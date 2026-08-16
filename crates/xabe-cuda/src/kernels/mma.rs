//! Turing integer tensor cores: `mma.sync.aligned.m8n8k16.s32.s8.s8.s32`.
//!
//! This is the instruction the whole prefill argument turns on.
//! `docs/OPTIMIZATION.md` §8.1 shows llama.cpp's prefill throughput is 61.9%
//! of this engine's absolute *fp32* ceiling, and `docs/BENCHMARKS.md` measures
//! the MoE grouped GEMM at 27% of fp32 peak against a ~50% structural ceiling
//! for its instruction mix. Both say the same thing: **fp32 cannot win
//! prefill, and no amount of tuning changes that.** Integer tensor cores are
//! the only path, and this module is the primitive they are built on.
//!
//! # Why int8 and not fp16
//!
//! Measured on this card: `m8n8k16` s8 runs at ~198 TOP/s against `m16n8k8`
//! fp16's ~99 TFLOP/s and scalar fp32's ~17.9 TFLOP/s. Twice the fp16 rate is
//! the smaller reason. The larger one is that **the weights are already
//! quantized**: the model ships Q6_K and Q8_0, so an int8 operand is the
//! native format, while an fp16 operand would mean dequantizing to a *wider*
//! type than the data actually carries.
//!
//! # What is and is not reachable here
//!
//! Verified by compiling to SASS on this host rather than trusting NVRTC:
//!
//! - `m8n8k16.s32.s8.s8.s32` assembles at `sm_75` and is what this module
//!   uses.
//! - `m16n8k8.f32.f16.f16.f32` assembles at `sm_75` and lowers to a genuine
//!   `HMMA.1688.F32`.
//! - **`m16n8k16` does not.** NVRTC *accepts* it at `compute_75` and emits
//!   PTX; ptxas then rejects it with `Feature '.m16n8k16' requires .target
//!   sm_80 or higher`. NVRTC success is therefore not evidence of
//!   reachability, and anything wanting a wider K must decompose into
//!   `m8n8k16` steps sharing one accumulator. Upstream does exactly this —
//!   llama.cpp's `ggml/src/ggml-cuda/mma.cuh`, TurboMind's
//!   `kernels/core/mma.h`, and vLLM's `marlin_mma.h` all split the Ampere
//!   shape into Turing halves.
//!
//! # The fragment layout, and why it is gated exactly
//!
//! A wrong fragment layout is *the* characteristic defect of a hand-written
//! MMA port: it produces finite, plausible, entirely wrong numbers. There is
//! no tolerance that separates it from arithmetic noise.
//!
//! So this kernel is gated on **bit-exact equality** with
//! [`xabe_kernels::mma::int8_gemm`], which is possible only because integer
//! addition is associative — the device and the reference sum the same
//! products in different orders and must still agree to the last bit. Every
//! fp32 kernel in this workspace has to accept a tolerance for exactly the
//! reason this one does not.
//!
//! Layout, from the PTX ISA, with lane `l` in `0..32`:
//!
//! ```text
//!   A (8x16 row-major)   row = l >> 2,  columns (l & 3) * 4 + {0,1,2,3}
//!   B (16x8 col-major)   col = l >> 2,  rows    (l & 3) * 4 + {0,1,2,3}
//!   C/D (8x8)            row = l >> 2,  columns (l & 3) * 2 + {0,1}
//! ```
//!
//! Note the accumulator's stride is 2 and the operands' is 4. Assuming they
//! match is the easy mistake, and `xabe_kernels::mma` spells all three in Rust
//! so a host-side test can check the tiling covers each element exactly once.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};

use super::compile;

/// Rows of the output tile, fixed by the instruction.
pub const MMA_M: usize = 8;
/// Columns of the output tile, fixed by the instruction.
pub const MMA_N: usize = 8;
/// Contraction elements consumed per instruction, fixed by the instruction.
pub const MMA_K: usize = 16;

const MMA_SRC: &str = r#"
extern "C" {

// d[m][n] = sum_k a[m][k] * b[n][k], int8 operands, int32 accumulate.
//
// `a` is [M][K] row-major and `b` is [N][K] row-major -- b is the transpose of
// the mathematical right operand, which is what `.col` means and what every
// quantized weight layout in this project already stores.
//
// One warp per output tile. grid.x tiles the N axis, grid.y the M axis, so the
// launch shape is a function of the declared geometry alone and stays
// capturable in a CUDA graph (AGENTS.md rule 5).
//
// K must be a multiple of 16. The caller checks it; a kernel-side check would
// have to be a branch on a host value in the inner loop.
__global__ void mma_int8_gemm(
    const signed char* __restrict__ a,
    const signed char* __restrict__ b,
    int* __restrict__ d,
    int m, int n, int k
) {
    int tile_n = blockIdx.x * 8;
    int tile_m = blockIdx.y * 8;
    int lane = threadIdx.x;

    // Operand slots. The `& 3` group indexes four *consecutive* contraction
    // elements, which is what makes each lane's load a single 4-byte access
    // rather than four scattered ones.
    int a_row = tile_m + (lane >> 2);
    int b_col = tile_n + (lane >> 2);
    int elem0 = (lane & 3) * 4;

    // The accumulator's column stride is 2, not 4. This is the asymmetry the
    // module docs warn about.
    int d_row = tile_m + (lane >> 2);
    int d_col = tile_n + (lane & 3) * 2;

    int acc0 = 0, acc1 = 0;

    for (int k0 = 0; k0 < k; k0 += 16) {
        // Rows past the end contribute zero rather than reading out of bounds.
        // Zero is the identity for this accumulation, so a padded tile is
        // arithmetically the same as a smaller one.
        unsigned int af = 0, bf = 0;
        if (a_row < m) {
            const signed char* p = a + (long long)a_row * k + k0 + elem0;
            af = *(const unsigned int*)p;
        }
        if (b_col < n) {
            const signed char* p = b + (long long)b_col * k + k0 + elem0;
            bf = *(const unsigned int*)p;
        }

        asm volatile(
            "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
            "{%0,%1}, {%2}, {%3}, {%0,%1};"
            : "+r"(acc0), "+r"(acc1)
            : "r"(af), "r"(bf)
        );
    }

    if (d_row < m) {
        if (d_col     < n) d[(long long)d_row * n + d_col    ] = acc0;
        if (d_col + 1 < n) d[(long long)d_row * n + d_col + 1] = acc1;
    }
}

}
"#;

/// Something went wrong compiling or launching the integer MMA kernel.
#[derive(Debug)]
pub enum MmaError {
    /// NVRTC rejected the source, or the module failed to load.
    ///
    /// On `sm_75` the likely cause is an instruction that needs `sm_80`. Note
    /// that such a failure surfaces here at *module load*, not at compile:
    /// NVRTC accepts `m16n8k16` and ptxas rejects it.
    Compile(String),
    /// The driver failed.
    Driver(DriverError),
    /// `k` is not a whole number of `MMA_K` steps.
    ///
    /// Rejected rather than padded: padding would need a masked tail load, and
    /// silently truncating would produce a plausible wrong product.
    RaggedContraction { k: usize },
    /// A buffer is not the size the declared shape implies.
    BufferShape {
        what: &'static str,
        expected: usize,
        actual: usize,
    },
}

impl std::fmt::Display for MmaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compile(m) => write!(f, "integer MMA kernel compilation failed: {m}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::RaggedContraction { k } => write!(
                f,
                "contraction length {k} is not a multiple of {MMA_K}, which is the \
                 instruction's K and not a tunable",
            ),
            Self::BufferShape {
                what,
                expected,
                actual,
            } => write!(
                f,
                "{what} holds {actual} elements, the shape implies {expected}"
            ),
        }
    }
}

impl std::error::Error for MmaError {}

impl From<DriverError> for MmaError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

/// The compiled integer tensor-core GEMM.
pub struct MmaKernels {
    gemm: CudaFunction,
}

impl MmaKernels {
    /// Compile for the context's device.
    ///
    /// A failure here on `sm_75` is the reachability signal that matters: the
    /// module is what ptxas has to accept, and ptxas is stricter than NVRTC.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, MmaError> {
        let ptx = compile(MMA_SRC, "mma_int8").map_err(MmaError::Compile)?;
        let module = ctx.load_module(ptx)?;
        Ok(Self {
            gemm: module.load_function("mma_int8_gemm")?,
        })
    }

    /// `d[m][n] = sum_k a[m][k] * b[n][k]`, exactly.
    ///
    /// `a` is `[m][k]` and `b` is `[n][k]`, both row-major and both int8.
    /// `d` is `[m][n]` int32. The result is **bit-exact** against
    /// [`xabe_kernels::mma::int8_gemm`] — integer addition is associative, so
    /// summation order cannot excuse a disagreement.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self,
        stream: &Arc<CudaStream>,
        a: &CudaSlice<i8>,
        b: &CudaSlice<i8>,
        d: &mut CudaSlice<i32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), MmaError> {
        if !k.is_multiple_of(MMA_K) {
            return Err(MmaError::RaggedContraction { k });
        }
        expect_len("a", a.len(), m * k)?;
        expect_len("b", b.len(), n * k)?;
        expect_len("d", d.len(), m * n)?;

        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(MMA_N) as u32, m.div_ceil(MMA_M) as u32, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        let mut builder = stream.launch_builder(&self.gemm);
        builder
            .arg(a)
            .arg(b)
            .arg(&mut *d)
            .arg(&mi)
            .arg(&ni)
            .arg(&ki);
        // SAFETY: one warp per 8x8 output tile over a grid that covers `m` x
        // `n` and bounds-checks both axes; every operand load is guarded by
        // the same bounds and every buffer was checked against the declared
        // shape above. `k` is a whole number of 16-element steps, so the inner
        // loop reads exactly `k` elements per row.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }
}

fn expect_len(what: &'static str, actual: usize, expected: usize) -> Result<(), MmaError> {
    if actual == expected {
        Ok(())
    } else {
        Err(MmaError::BufferShape {
            what,
            expected,
            actual,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_kernel_uses_the_turing_shape_and_not_the_ampere_one() {
        // `m16n8k16` compiles under NVRTC at compute_75 and is then rejected
        // by ptxas, so its presence would not be caught until module load on a
        // machine with a device. Caught here instead, on any machine.
        assert!(
            MMA_SRC.contains("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32"),
            "the int8 Turing shape must be the one emitted",
        );
        assert!(
            !MMA_SRC.contains("m16n8k16"),
            "m16n8k16 requires sm_80; it must be decomposed, not emitted",
        );
    }

    #[test]
    fn the_fragment_slots_match_the_rust_side_spelling() {
        // The kernel computes its slots inline; `xabe_kernels::mma` spells the
        // same layout for the host. If the two ever disagree the differential
        // test would fail, but it would fail without saying which side moved.
        assert!(
            MMA_SRC.contains("(lane >> 2)"),
            "row/col split is lane >> 2"
        );
        assert!(MMA_SRC.contains("(lane & 3) * 4"), "operand stride is 4");
        assert!(
            MMA_SRC.contains("(lane & 3) * 2"),
            "accumulator stride is 2, not 4 — this asymmetry is the trap",
        );
    }

    #[test]
    fn the_instruction_geometry_is_not_tunable() {
        // These are properties of the hardware instruction. A test exists so
        // that "tuning" them is a failing build rather than silent garbage.
        assert_eq!((MMA_M, MMA_N, MMA_K), (8, 8, 16));
    }
}
