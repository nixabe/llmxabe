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

    /// Softplus with the baseline overflow guard and libdevice exp/log.
    ///
    /// # Safety
    /// x/out cover n floats, n >= 0, and the launch is nonempty and 1-D.
    /// Output may exactly alias x; partial overlap is forbidden.
    #[kernel]
    pub unsafe fn softplus_elementwise(x: *const f32, out: *mut f32, n: i64) {
        let stride = thread::blockDim_x() as i64 * thread::gridDim_x() as i64;
        let mut i = thread::blockIdx_x() as i64 * thread::blockDim_x() as i64
            + thread::threadIdx_x() as i64;
        while i < n {
            unsafe {
                let v = *x.add(i as usize);
                *out.add(i as usize) = if v > 20.0 { v } else { (1.0 + v.exp()).ln() };
            }
            i += stride;
        }
    }

    /// Split-half partial RoPE; paired outputs share double-precision angles.
    ///
    /// # Safety
    /// x/out are disjoint buffers covering gridDim.y * heads * head_dim
    /// floats. positions covers gridDim.y. gridDim.x == heads, blockDim.x ==
    /// head_dim, and 0 <= rope_dim <= head_dim is even. theta_base > 0.
    #[kernel]
    pub unsafe fn rope_partial(
        x: *const f32,
        positions: *const u32,
        out: *mut f32,
        heads: i32,
        head_dim: i32,
        rope_dim: i32,
        theta_base: f32,
    ) {
        let h = thread::blockIdx_x() as usize;
        let t = thread::blockIdx_y() as usize;
        let j = thread::threadIdx_x() as i32;
        let base = (t * heads as usize + h) * head_dim as usize;
        unsafe {
            if j >= rope_dim {
                *out.add(base + j as usize) = *x.add(base + j as usize);
            } else if j < rope_dim / 2 {
                let half = rope_dim / 2;
                let freq = (theta_base as f64).powf(-2.0 * j as f64 / rope_dim as f64);
                let angle = *positions.add(t) as f64 * freq;
                let sin_a = angle.sin() as f32;
                let cos_a = angle.cos() as f32;
                let x0 = *x.add(base + j as usize);
                let x1 = *x.add(base + (j + half) as usize);
                // Match NVRTC's contraction while sharing the angle and loads.
                *out.add(base + j as usize) = x0.mul_add(cos_a, -(x1 * sin_a));
                *out.add(base + (j + half) as usize) = x0.mul_add(sin_a, x1 * cos_a);
            }
        }
    }

    /// Production attention RoPE, using two token groups on underfilled grids.
    ///
    /// # Safety
    /// input/output are disjoint and cover tokens * heads * head_dim floats.
    /// position covers one i32; grid is (ceil(tokens/16), heads), blockDim.x
    /// equals head_dim, and rotated is even and between zero and head_dim.
    #[kernel]
    pub unsafe fn attn_rope_partial_neox(
        input: *const f32,
        output: *mut f32,
        heads: i32,
        head_dim: i32,
        rotated: i32,
        position: *const i32,
        theta: f32,
        tokens: i32,
    ) {
        let t0 = thread::blockIdx_x() as i32 * 16;
        // One loop bound, also valid for a partial tile at i32::MAX.
        let end = t0 + (tokens - t0).min(16);
        let h = thread::blockIdx_y() as usize;
        let d = thread::threadIdx_x() as i32;
        unsafe {
            if d >= rotated {
                let mut t = t0;
                while t < end {
                    let base = (t as usize * heads as usize + h) * head_dim as usize;
                    *output.add(base + d as usize) = *input.add(base + d as usize);
                    t += 1;
                }
                return;
            }
            let half = rotated / 2;
            // On the 72-SM target, a grid with at most one block per SM
            // needs more active rotary warps. Both halves then own alternating
            // tokens. Larger grids retain one frequency calculation per pair.
            let split_tokens = thread::gridDim_x() as u64 * thread::gridDim_y() as u64 <= 72;
            let stripe = if d < half { 0 } else { 1 };
            if stripe == 1 && !split_tokens {
                return;
            }
            let step = if split_tokens { 2 } else { 1 };
            let pair = if d < half { d } else { d - half };
            let mut u = stripe;
            let count = end - t0;
            if u >= count {
                return;
            }
            let freq = (theta as f64).powf(-2.0 * pair as f64 / rotated as f64);
            let pos = *position as f64;
            while u < count {
                let t = t0 + u;
                let base = (t as usize * heads as usize + h) * head_dim as usize;
                let angle = (pos + t as f64) * freq;
                let sin_a = angle.sin() as f32;
                let cos_a = angle.cos() as f32;
                let x0 = *input.add(base + pair as usize);
                let x1 = *input.add(base + (pair + half) as usize);
                *output.add(base + pair as usize) = x0.mul_add(cos_a, -(x1 * sin_a));
                *output.add(base + (pair + half) as usize) = x0.mul_add(sin_a, x1 * cos_a);
                u += step;
            }
        }
    }

    /// Two-level warp reduction with one block barrier.
    /// Each warp reduces the immutable partials and broadcasts locally.
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
            let lane = tid & 31;
            let warps = thread::blockDim_x() / 32;
            let mut total = if lane < warps {
                *scratch.add(lane as usize)
            } else {
                0.0
            };
            let mut offset = 16;
            while offset > 0 {
                total += warp::shuffle_down_f32_sync(u32::MAX, total, offset);
                offset >>= 1;
            }
            warp::shuffle_f32_sync(u32::MAX, total, 0)
        }
    }

    /// Compute the inverse RMS with the parallel reduction and exact division.
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
            // Small head rows fit one element per thread. Keep both operands
            // live across the reduction and overlap the gate's math with it.
            if width <= thread::blockDim_x() as i32 {
                let j = thread::threadIdx_x() as i32;
                let base = thread::blockIdx_x() as usize * width as usize;
                let v = if j < width {
                    *x.add(base + j as usize)
                } else {
                    0.0
                };
                let g = if j < width {
                    *gate.add(base + j as usize)
                } else {
                    0.0
                };
                let silu = g / (1.0 + (-g).exp());
                let inv = 1.0 / (block_sum(v * v) / width as f32 + eps).sqrt();
                if j < width {
                    let nv = v * inv * *weight.add(j as usize);
                    *normed.add(base + j as usize) = nv;
                    *out.add(base + j as usize) = silu * nv;
                }
                return;
            }
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
