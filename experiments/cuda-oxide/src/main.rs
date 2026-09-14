//! Experimental Rust implementations of layer_ops.rs elementwise kernels.
//! Raw pointers preserve the existing cudarc ABI, including in-place output.

use cuda_device::{kernel, thread};
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
}

fn main() {}
