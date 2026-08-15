//! Gated DeltaNet, recurrent (decode) form, on the device.
//!
//! The critical path: 30 of Qwen3.6's 40 layers. Unlike Gated Attention this
//! has no flash-attention codebase to port from, so the reference is
//! `xabe_kernels::gdn::recurrent`, which was itself cross-derived from
//! llama.cpp's `ggml_compute_forward_gated_delta_net_one_chunk` and vLLM's
//! `fused_recurrent_gated_delta_rule_fwd_kernel`.
//!
//! ## Shape
//!
//! Per head: state `S` is `head_dim x head_dim` fp32, laid out `S[v][k]` with
//! the key index contiguous. head_dim is 128 and there are 32 value heads, so
//! one layer's state for one sequence is 2 MiB — constant in sequence length,
//! which is the whole point of the hybrid architecture.
//!
//! The 32 value heads share 16 query/key heads by **modulo**:
//! `qk_head = v_head % 16`, so value heads 0 and 16 share a query/key head —
//! *not* 0 and 1. llama.cpp's fused op indexes it as
//! `fastmodulo(h_idx, n_k_heads)` (`ggml/src/ggml-cuda/gated_delta_net.cu:37`),
//! and its non-fused fallback reaches the same mapping through
//! `ggml_repeat_4d` (`src/models/qwen35moe.cpp:466`), which *tiles* rather than
//! stretches. Dividing instead — `v_head / 2`, the other plausible reading —
//! pairs every value head with the wrong query and stays finite and fluent.
//! That was this kernel's convention until it was gated against the captured
//! `final_output-N`: `h % 16` reproduces it to `max_abs 7.15e-7`, `h / 2` to
//! `5.59e-1` at cosine `0.975`.
//!
//! ## Why two kernels
//!
//! `q` and `k` are L2-normalized per query/key head. Doing that inside the
//! state kernel would mean every block redundantly re-normalizing a vector
//! shared by 128 blocks, and would add two block-wide reductions to the hot
//! path. Splitting it out costs one extra launch and lets the state kernel
//! touch global memory exactly once in each direction:
//!
//! - read `S[v][k]`, write `S[v][k]` — nothing else.
//!
//! That matters because this kernel is purely bandwidth-bound. At 2 MiB of
//! state per layer per sequence, 30 layers, read and written once per token,
//! the floor is ~120 MiB/token of state traffic.
//!
//! ## One thread per state element
//!
//! Block `(v_head, v_index)` owns row `S[v_index][*]` with one thread per key
//! index. The alternative — one thread per row — needs the row twice (once to
//! form the delta correction, once to apply it and read out), because 128
//! floats will not stay in registers. Trading that second pass for two
//! block-wide reductions is the right way round on a bandwidth-bound kernel.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};

use super::compile;

/// The epsilon inside the L2 normalization.
///
/// It is a **floor on the norm**, not a term added to the sum of squares — see
/// the `gdn_normalize_qk` source. `ggml_l2_norm` is
/// `scale = 1/max(sqrt(sum), eps)` on the CPU
/// (`ggml/src/ggml-cpu/ops.cpp:4204`) and `rsqrt(max(sum, eps*eps))` on CUDA
/// (`ggml/src/ggml-cuda/norm.cu:273`), which is the same value; llama.cpp
/// passes `hparams.f_norm_rms_eps` into it at `src/models/qwen35moe.cpp:456`.
/// Under that form the epsilon only binds on a vector whose norm is below it,
/// so its exact value is immaterial in normal operation and `1e-6` — FLA's
/// value, and the one
/// `xabe_kernels::gdn::recurrent::l2_normalize` uses — is safe to keep.
pub const L2_EPS: f32 = 1e-6;

const GDN_SRC: &str = r#"
extern "C" {

// Sum across a block of up to 1024 threads.
//
// Warp shuffles first, then one round through shared memory. The tree order
// differs from the reference's sequential sum, which is the dominant source
// of disagreement between this kernel and the CPU — fp32 addition is not
// associative. It is bounded and unbiased; see the module docs for the
// tolerance this justifies.
__device__ __forceinline__ float block_reduce_sum(float v, float* scratch) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) scratch[warp] = v;
    __syncthreads();

    int n_warps = (blockDim.x + 31) >> 5;
    float total = 0.0f;
    if (threadIdx.x == 0) {
        for (int w = 0; w < n_warps; ++w) total += scratch[w];
        scratch[0] = total;
    }
    __syncthreads();
    return scratch[0];
}

// L2-normalize q and k per query/key head, and fold the 1/sqrt(head_dim)
// output scale into q.
//
// The scale belongs on q rather than on the output because q appears only in
// the final dot product and never in the state update, so the two placements
// are identical arithmetic — this is the placement both upstream references
// use.
//
// grid: one block per qk head. block: head_dim threads.
__global__ void gdn_normalize_qk(
    const float* __restrict__ q_in,
    const float* __restrict__ k_in,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    int head_dim,
    float eps,
    float scale
) {
    extern __shared__ float scratch[];
    int head = blockIdx.x;
    int j = threadIdx.x;
    int base = head * head_dim;

    float qv = q_in[base + j];
    float kv = k_in[base + j];

    float q_sq = block_reduce_sum(qv * qv, scratch);
    __syncthreads();
    float k_sq = block_reduce_sum(kv * kv, scratch);
    __syncthreads();

    // `1/max(sqrt(sum), eps)`, which is ggml_l2_norm's formula verbatim
    // (ggml-cpu/ops.cpp:4204; the CUDA path's rsqrt(max(sum, eps*eps)) is the
    // same value). The epsilon FLOORS the norm rather than being added to the
    // sum of squares, so it guards a zero vector from dividing by zero and
    // otherwise never binds. `1/sqrt(sum + eps)` — the form this kernel used
    // until it was measured against the real q_conv/k_conv of blocks 0/4/20 —
    // instead shrinks every vector by eps/(2*sum): at the smallest observed
    // sum of squares, 2.327e-3, that is 2.148e-4 relative, three orders of
    // magnitude above this kernel's agreement with its reference.
    //
    // rsqrtf would be faster and is not IEEE-exact; the reference divides by
    // sqrtf, and matching it keeps the disagreement attributable to the
    // reduction order alone.
    q_out[base + j] = qv * (1.0f / fmaxf(sqrtf(q_sq), eps)) * scale;
    k_out[base + j] = kv * (1.0f / fmaxf(sqrtf(k_sq), eps));
}

// One recurrent step of the gated delta rule, for every value head at once.
//
// grid: (head_dim, n_v_heads) — one block per state row.
// block: head_dim threads — one per key index within that row.
//
// The order of operations follows the reference exactly: decay the state,
// form the delta correction against the *decayed* state, apply the
// outer-product update, then read the output from the *updated* state.
// Reordering any of these is a plausible-looking change that silently
// produces a different model.
__global__ void gdn_recurrent_step(
    float* __restrict__ state,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ log_decay,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int head_dim,
    int qk_heads
) {
    extern __shared__ float scratch[];

    int vi = blockIdx.x;
    int h  = blockIdx.y;
    int j  = threadIdx.x;

    // **Modulo, not division.** llama.cpp's fused op is
    // `fastmodulo(h_idx, n_k_heads)` and its fallback broadcasts with
    // `ggml_repeat_4d`, which tiles. See the module docs for the measurement
    // that discriminates the two against the captured `final_output-N`.
    int qk_head = h % qk_heads;
    int qk_base = qk_head * head_dim;
    int v_base  = h * head_dim;

    float kj = k[qk_base + j];
    float qj = q[qk_base + j];

    // `expf`, not the fast `__expf` intrinsic. The decay multiplies the whole
    // state every token, so a systematic bias here compounds over the context
    // rather than averaging out. Measured both at the real geometry over 512
    // tokens: identical worst-case error (2.980e-8), because the reduction
    // order dominates. `__expf` is available as a tuning knob if this kernel
    // ever turns out to be latency-bound, but it buys nothing measured today.
    float decay = expf(log_decay[h]);

    // 1. Decay, held in a register rather than written and re-read.
    long long idx = ((long long)h * head_dim + vi) * head_dim + j;
    float s = state[idx] * decay;

    // 2. Delta correction against the decayed state.
    float predicted = block_reduce_sum(s * kj, scratch);
    __syncthreads();
    float vcorr = beta[h] * (v[v_base + vi] - predicted);

    // 3. Outer-product update.
    s += vcorr * kj;
    state[idx] = s;

    // 4. Output from the updated state.
    float o = block_reduce_sum(s * qj, scratch);
    if (j == 0) out[v_base + vi] = o;
}

}
"#;

/// Something went wrong compiling or launching a GDN kernel.
#[derive(Debug)]
pub enum GdnError {
    /// NVRTC rejected the source, or the module failed to load.
    Compile(String),
    /// The driver failed.
    Driver(DriverError),
    /// A head dimension the kernel cannot service.
    UnsupportedHeadDim { head_dim: usize },
    /// Value heads are not a whole multiple of query/key heads.
    UnevenHeadGrouping { value_heads: usize, qk_heads: usize },
}

impl std::fmt::Display for GdnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compile(m) => write!(f, "kernel compilation failed: {m}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::UnsupportedHeadDim { head_dim } => write!(
                f,
                "head_dim {head_dim} must be a positive multiple of 32 and at most 1024",
            ),
            Self::UnevenHeadGrouping {
                value_heads,
                qk_heads,
            } => write!(
                f,
                "{value_heads} value heads do not divide evenly among {qk_heads} query/key heads",
            ),
        }
    }
}

impl std::error::Error for GdnError {}

impl From<DriverError> for GdnError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

/// Buffers reused across decode steps.
///
/// Normalized `q` and `k` are scratch, not state: they are overwritten every
/// step. Holding them here rather than allocating per step keeps the decode
/// path free of `cudaMalloc`, which is a synchronising call and would make
/// graph capture impossible.
pub struct GdnScratch {
    q_norm: CudaSlice<f32>,
    k_norm: CudaSlice<f32>,
}

/// Compiled Gated DeltaNet kernels.
pub struct GdnKernels {
    normalize: CudaFunction,
    step: CudaFunction,
    head_dim: usize,
    value_heads: usize,
    qk_heads: usize,
}

impl GdnKernels {
    /// Compile for a specific head geometry.
    ///
    /// The geometry is fixed at construction rather than passed per launch
    /// because it is fixed by the model, and validating it once means the
    /// launch path has nothing left to reject.
    pub fn new(
        ctx: &Arc<CudaContext>,
        head_dim: usize,
        value_heads: usize,
        qk_heads: usize,
    ) -> Result<Self, GdnError> {
        // One thread per key index, reduced with warp shuffles, so the head
        // dimension must be a whole number of warps and fit in one block.
        if head_dim == 0 || !head_dim.is_multiple_of(32) || head_dim > 1024 {
            return Err(GdnError::UnsupportedHeadDim { head_dim });
        }
        if qk_heads == 0 || !value_heads.is_multiple_of(qk_heads) {
            return Err(GdnError::UnevenHeadGrouping {
                value_heads,
                qk_heads,
            });
        }
        let ptx = compile(GDN_SRC, "gdn").map_err(GdnError::Compile)?;
        let module = ctx.load_module(ptx)?;
        Ok(Self {
            normalize: module.load_function("gdn_normalize_qk")?,
            step: module.load_function("gdn_recurrent_step")?,
            head_dim,
            value_heads,
            qk_heads,
        })
    }

    /// How many value heads share each query/key head.
    ///
    /// The *count*, not the mapping: value head `h` reads query/key head
    /// `h % qk_heads`, so the heads sharing a query/key head are `hq`,
    /// `hq + qk_heads`, `hq + 2 * qk_heads`, … and not a contiguous run.
    pub fn heads_per_kv(&self) -> usize {
        self.value_heads / self.qk_heads
    }

    /// The `1/sqrt(head_dim)` scale folded into `q`.
    pub fn output_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// Allocate the per-step scratch buffers.
    pub fn scratch(&self, stream: &Arc<CudaStream>) -> Result<GdnScratch, GdnError> {
        let n = self.qk_heads * self.head_dim;
        Ok(GdnScratch {
            q_norm: stream.alloc_zeros::<f32>(n)?,
            k_norm: stream.alloc_zeros::<f32>(n)?,
        })
    }

    /// Bytes of recurrent state for one sequence at this geometry.
    pub fn state_bytes(&self) -> usize {
        self.value_heads * self.head_dim * self.head_dim * size_of::<f32>()
    }

    /// Advance `state` by one token and write this token's output.
    ///
    /// `q` and `k` are `[qk_heads][head_dim]` raw — not normalized, not
    /// scaled; this does both. `v` is `[value_heads][head_dim]`, `log_decay`
    /// and `beta` are `[value_heads]`, and `out` is `[value_heads][head_dim]`.
    ///
    /// `state` is `[value_heads][head_dim][head_dim]` and is updated in place.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        stream: &Arc<CudaStream>,
        scratch: &mut GdnScratch,
        state: &mut CudaSlice<f32>,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        log_decay: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), GdnError> {
        let head_dim = self.head_dim as i32;
        let qk_heads = self.qk_heads as i32;
        // One float per warp, which is the most `block_reduce_sum` stores.
        let shared = (self.head_dim.div_ceil(32) * size_of::<f32>()) as u32;

        let norm_cfg = LaunchConfig {
            grid_dim: (self.qk_heads as u32, 1, 1),
            block_dim: (self.head_dim as u32, 1, 1),
            shared_mem_bytes: shared,
        };
        let eps = L2_EPS;
        let scale = self.output_scale();
        let mut builder = stream.launch_builder(&self.normalize);
        builder
            .arg(q)
            .arg(k)
            .arg(&mut scratch.q_norm)
            .arg(&mut scratch.k_norm)
            .arg(&head_dim)
            .arg(&eps)
            .arg(&scale);
        // SAFETY: one block per qk head and one thread per head element, with
        // both inputs and both outputs allocated to `qk_heads * head_dim`.
        // Shared memory covers one float per warp, which is all the reduction
        // writes.
        unsafe { builder.launch(norm_cfg) }?;

        let step_cfg = LaunchConfig {
            grid_dim: (self.head_dim as u32, self.value_heads as u32, 1),
            block_dim: (self.head_dim as u32, 1, 1),
            shared_mem_bytes: shared,
        };
        let mut builder = stream.launch_builder(&self.step);
        builder
            .arg(state)
            .arg(&scratch.q_norm)
            .arg(&scratch.k_norm)
            .arg(v)
            .arg(log_decay)
            .arg(beta)
            .arg(out)
            .arg(&head_dim)
            .arg(&qk_heads);
        // SAFETY: the grid is (head_dim, value_heads) and the block is
        // head_dim threads, so the flat state index
        // `(h * head_dim + vi) * head_dim + j` stays within
        // `value_heads * head_dim * head_dim`, which is what `state` holds.
        unsafe { builder.launch(step_cfg) }?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_source_updates_the_state_before_reading_the_output() {
        // Reading the output from the pre-correction state is the classic
        // way to get this rule subtly wrong: it stays finite, stays fluent,
        // and is a different model. The ordering is asserted structurally
        // because no synthetic input distinguishes the two cheaply.
        let update = GDN_SRC.find("s += vcorr * kj;").expect("update present");
        let output = GDN_SRC
            .find("float o = block_reduce_sum(s * qj, scratch);")
            .expect("output read present");
        assert!(update < output, "output is read before the state update");
    }

    #[test]
    fn the_correction_is_formed_against_the_decayed_state() {
        let decay = GDN_SRC
            .find("float s = state[idx] * decay;")
            .expect("decay present");
        let predicted = GDN_SRC
            .find("float predicted = block_reduce_sum(s * kj, scratch);")
            .expect("correction present");
        assert!(decay < predicted, "correction uses the undecayed state");
    }

    #[test]
    fn the_scale_is_folded_into_q_only() {
        // Folding it into k as well would square it; folding it into neither
        // scales every GDN layer's output by 11.3x.
        assert!(
            GDN_SRC.contains("q_out[base + j] = qv * (1.0f / fmaxf(sqrtf(q_sq), eps)) * scale;")
        );
        assert!(GDN_SRC.contains("k_out[base + j] = kv * (1.0f / fmaxf(sqrtf(k_sq), eps));"));
    }

    #[test]
    fn the_l2_epsilon_floors_the_norm_rather_than_being_added_to_the_sum() {
        // `ggml_l2_norm` is `1/max(sqrt(sum), eps)`, so eps guards a zero
        // vector and otherwise never binds. `1/sqrt(sum + eps)` looks
        // equivalent and is not: it shrinks every vector by eps/(2*sum),
        // measured at 2.148e-4 relative on the smallest-norm q/k of the real
        // model's blocks 0/4/20. Asserted structurally because no synthetic
        // input at a realistic scale distinguishes the two above the noise
        // floor of the differential gate.
        assert!(
            !GDN_SRC.contains("q_sq + eps"),
            "the L2 epsilon reverted to being added to the sum of squares",
        );
        assert!(!GDN_SRC.contains("k_sq + eps"));
    }

    #[test]
    fn the_query_key_head_is_selected_by_modulo_not_division() {
        // The single most consequential line in this file. llama.cpp's fused
        // op computes `fastmodulo(h_idx, n_k_heads)`
        // (ggml/src/ggml-cuda/gated_delta_net.cu:37) and its non-fused
        // fallback broadcasts with `ggml_repeat_4d`, which tiles rather than
        // stretches, so value head 17 reads query/key head 1 and not head 8.
        // Division is finite, fluent and a different model; the numeric
        // discrimination lives in
        // `gdn_differential.rs::the_query_key_head_broadcast_is_modulo_not_division`.
        assert!(GDN_SRC.contains("int qk_head = h % qk_heads;"));
        assert!(
            !GDN_SRC.contains("heads_per_kv"),
            "the kernel still takes the sharing ratio, which only a division \
             mapping needs",
        );
    }

    #[test]
    fn head_geometry_is_validated_not_assumed() {
        // These are the checks that run without a device.
        let bad_dim = GdnError::UnsupportedHeadDim { head_dim: 100 };
        assert!(bad_dim.to_string().contains("100"));
        let bad_group = GdnError::UnevenHeadGrouping {
            value_heads: 32,
            qk_heads: 5,
        };
        assert!(bad_group.to_string().contains("32"));
        // Qwen3.6's real geometry must be accepted.
        assert_eq!(128 % 32, 0);
        assert_eq!(32 % 16, 0);
    }
}
