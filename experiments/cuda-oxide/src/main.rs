//! Experimental Rust implementations of layer_ops.rs elementwise kernels.
//! Raw pointers preserve the existing cudarc ABI, including in-place output.

use cuda_device::{DynamicSharedArray, device, kernel, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// Add a residual with the same grid-stride loop as the NVRTC baseline.
    ///
    /// # Safety
    /// All pointers cover `n` floats, n is nonnegative, and the launch is 1-D
    /// with nonzero dimensions. Output may equal either input, but must not
    /// partially overlap it. No other stream may access output concurrently.
    #[kernel]
    pub unsafe fn tensor_add(a: *const f32, b: *const f32, out: *mut f32, n: i64) {
        let stride = thread::blockDim_x() as i64 * thread::gridDim_x() as i64;
        let mut i = thread::blockIdx_x() as i64 * thread::blockDim_x() as i64
            + thread::threadIdx_x() as i64;
        while i < n {
            // SAFETY: each thread owns its grid-stride indices; reads precede
            // the write so exact input/output aliasing is supported.
            unsafe {
                *out.add(i as usize) = *a.add(i as usize) + *b.add(i as usize);
            }
            i += stride;
        }
    }

    /// SwiGLU in the baseline operand order, using libdevice exp rather than
    /// an approximate exponential intrinsic.
    ///
    /// # Safety
    /// Pointers cover n floats, n >= 0, and the launch is nonempty and 1-D.
    /// Output may exactly alias an input; partial overlap is forbidden.
    #[kernel]
    pub unsafe fn swiglu_mul(gate: *const f32, up: *const f32, out: *mut f32, n: i64) {
        let stride = thread::blockDim_x() as i64 * thread::gridDim_x() as i64;
        let mut i = thread::blockIdx_x() as i64 * thread::blockDim_x() as i64
            + thread::threadIdx_x() as i64;
        while i < n {
            unsafe {
                let g = *gate.add(i as usize);
                *out.add(i as usize) = (g / (1.0 + (-g).exp())) * *up.add(i as usize);
            }
            i += stride;
        }
    }

    /// Elementwise or per-row sigmoid gating, preserving the sigmoid waypoint.
    ///
    /// # Safety
    /// x/out cover n floats; gate/sig cover n floats or ceil(n/width) when
    /// broadcast != 0. width > 0, n >= 0, and the launch is nonempty and 1-D.
    /// Only out and x may exactly alias; all other overlaps are forbidden.
    #[kernel]
    pub unsafe fn sigmoid_gate_mul(
        x: *const f32,
        gate: *const f32,
        sig: *mut f32,
        out: *mut f32,
        n: i64,
        width: i32,
        broadcast: i32,
    ) {
        let stride = thread::blockDim_x() as i64 * thread::gridDim_x() as i64;
        let mut i = thread::blockIdx_x() as i64 * thread::blockDim_x() as i64
            + thread::threadIdx_x() as i64;
        while i < n {
            let g = if broadcast != 0 { i / width as i64 } else { i };
            unsafe {
                let s = 1.0 / (1.0 + (-*gate.add(g as usize)).exp());
                if broadcast == 0 || i % width as i64 == 0 {
                    *sig.add(g as usize) = s;
                }
                *out.add(i as usize) = s * *x.add(i as usize);
            }
            i += stride;
        }
    }

    /// Same shuffle tree and serial warp accumulation as layer_ops.rs.
    ///
    /// # Safety
    /// Every thread participates in a 1-D block of 32..=1024 threads, a
    /// multiple of 32. Dynamic shared memory holds one f32 per warp.
    #[device]
    unsafe fn block_sum(mut v: f32) -> f32 {
        let mut offset = 16;
        while offset > 0 {
            v += warp::shuffle_down_f32_sync(u32::MAX, v, offset);
            offset >>= 1;
        }
        let tid = thread::threadIdx_x();
        let scratch = DynamicSharedArray::<f32>::get();
        unsafe {
            if tid & 31 == 0 {
                *scratch.add((tid >> 5) as usize) = v;
            }
            thread::sync_threads();
            if tid == 0 {
                let mut total = 0.0;
                let mut w = 0;
                let warps = thread::blockDim_x() / 32;
                #[unroll(4)]
                while w < warps {
                    total += *scratch.add(w as usize);
                    w += 1;
                }
                *scratch = total;
            }
            thread::sync_threads();
            *scratch
        }
    }

    /// Compute the inverse RMS using the baseline reduction and division.
    ///
    /// # Safety
    /// x covers width floats for every grid row, width > 0, and block_sum's
    /// collective participation and shared-memory requirements hold.
    #[device]
    unsafe fn inverse_rms(x: *const f32, width: i32, eps: f32) -> f32 {
        let base = thread::blockIdx_x() as usize * width as usize;
        let mut j = thread::threadIdx_x() as i32;
        let mut partial = 0.0f32;
        unsafe {
            while j < width {
                let v = *x.add(base + j as usize);
                // NVRTC contracts the baseline's partial += v * v.
                partial = v.mul_add(v, partial);
                j += thread::blockDim_x() as i32;
            }
            1.0 / (block_sum(partial) / width as f32 + eps).sqrt()
        }
    }

    /// Weighted RMSNorm with the existing per-row launch ABI.
    ///
    /// # Safety
    /// x/out cover gridDim.x * width floats and weight covers width floats.
    /// Only out may exactly alias x. The inverse_rms launch contract holds;
    /// no other stream accesses output concurrently.
    #[kernel]
    pub unsafe fn rms_norm_rows(
        x: *const f32,
        weight: *const f32,
        out: *mut f32,
        width: i32,
        eps: f32,
    ) {
        unsafe {
            let inv = inverse_rms(x, width, eps);
            let base = thread::blockIdx_x() as usize * width as usize;
            let mut j = thread::threadIdx_x() as i32;
            while j < width {
                *out.add(base + j as usize) =
                    *x.add(base + j as usize) * inv * *weight.add(j as usize);
                j += thread::blockDim_x() as i32;
            }
        }
    }

    /// Fused RMSNorm/SwiGLU, retaining the normalized intermediate waypoint.
    ///
    /// # Safety
    /// x/gate/normed/out cover gridDim.x * width floats, weight covers width,
    /// outputs are disjoint from each other and inputs, and inverse_rms's
    /// collective launch contract holds. No concurrent output access.
    #[kernel]
    pub unsafe fn rms_norm_swiglu_rows(
        x: *const f32,
        weight: *const f32,
        gate: *const f32,
        normed: *mut f32,
        out: *mut f32,
        width: i32,
        eps: f32,
    ) {
        unsafe {
            let inv = inverse_rms(x, width, eps);
            let base = thread::blockIdx_x() as usize * width as usize;
            let mut j = thread::threadIdx_x() as i32;
            while j < width {
                let nv = *x.add(base + j as usize) * inv * *weight.add(j as usize);
                *normed.add(base + j as usize) = nv;
                let g = *gate.add(base + j as usize);
                *out.add(base + j as usize) = (g / (1.0 + (-g).exp())) * nv;
                j += thread::blockDim_x() as i32;
            }
        }
    }
}

fn main() {}
