//! RMSNorm, partial rotary embedding, SwiGLU, sigmoid gating, tensor add,
//! softplus, and the GDN short convolution.
//!
//! Every operation a forward pass needs that [`super::gdn`],
//! [`super::gdn_chunked`], [`super::attention`] and [`super::moe`] do not
//! already cover. They live here, as standalone tested kernels, rather than
//! inline in the blocks that use them: an engine-side inline kernel is
//! invisible to the kernel inventory and gets no differential test against the
//! `xabe-kernels` oracle. Fusing one *into* a block is still legitimate — the
//! GDN gate kernel computes softplus, the log-decay and beta's sigmoid in one
//! pass — but the standalone op has to exist and be the thing a fused variant
//! is checked against.
//!
//! References: [`xabe_kernels::norm::rms_norm`],
//! [`xabe_kernels::rope::apply_rope`], [`xabe_kernels::norm::swiglu`],
//! [`xabe_kernels::norm::sigmoid_gate`], [`xabe_kernels::norm::residual_add`],
//! [`xabe_kernels::norm::softplus`],
//! [`xabe_kernels::conv::causal_depthwise_conv1d`].
//!
//! # Geometry is per launch, not per instance
//!
//! [`super::gdn`] fixes its geometry at construction because the model fixes
//! it. These cannot: RMSNorm alone runs at three widths in one pass — hidden
//! 2048 for the input, post-mixer and final norms; head_dim 256 for the
//! attention layers' `attn_q_norm`/`attn_k_norm`; head_dim 128 for the GDN
//! output norm. So the width is a launch argument, validated per launch. The
//! sigmoid gate has the same problem in a sharper form: its gate is
//! elementwise for the attention output gate and **one scalar per token** for
//! the MoE shared expert, so the shape is a [`GateShape`] argument and the
//! gate buffer's length is checked against it on every launch.
//!
//! # What is exact, and what cannot be
//!
//! - **The tensor add, the rotary tail, and the convolution are bit-exact**
//!   and gated with `Tolerance::exact()`. The rotary tail is copied, not
//!   computed — partial rotary is easy to accidentally rotate, truncate or
//!   reorder past its boundary, and a tolerance would pass all three. The
//!   convolution spells its accumulation `__fadd_rn(acc, __fmul_rn(x, w))`
//!   because contracting into an FMA would round once where the reference
//!   rounds twice, and a tolerance can hide a reversed tap order on smooth
//!   input where exact equality cannot.
//! - **RMSNorm, SwiGLU, the sigmoid gate and softplus cannot be.** RMSNorm
//!   reduces 2048 squares in a shuffle tree where the reference sums
//!   sequentially. The other three are elementwise, so their entire
//!   disagreement is `expf` against `f32::exp` — about an ulp. All four are
//!   gated on a measured tolerance sized for that, not on a matmul-era one.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};

use super::compile;

/// Widest convolution the kernel will service.
///
/// The state-update kernel stages `conv_kernel - 1` window values in a
/// per-thread register array before writing any of them, which is what makes
/// updating the cache in place safe. That array needs a compile-time bound.
///
/// Qwen3.6 needs 4 (`qwen35moe.ssm.conv_kernel`). 8 is headroom, not a
/// surveyed requirement: llama.cpp reads `ssm_d_conv` from the file rather
/// than hardcoding it per architecture, so nothing here can claim what other
/// models in this family use. The cost of the headroom is four unused
/// registers per thread in one launch, and anything wider is rejected at the
/// launch path rather than overrunning the array.
pub const MAX_CONV_KERNEL: usize = 8;

/// Largest block the RMSNorm and rotary kernels will launch.
///
/// One thread per head element for the rotary kernel, and the upper bound on
/// the strided reduction for RMSNorm.
const MAX_BLOCK: usize = 1024;

const LAYER_OPS_SRC: &str = r#"
#define MAX_CONV_KERNEL 8

extern "C" {

// Sum across a block of up to 1024 threads. The same reduction as
// `gdn.rs`/`gdn_chunked.rs`, verbatim and for the same reason: warp shuffles
// then one pass through shared memory. The tree order differs from the
// reference's sequential sum, which is the entire source of disagreement
// between rms_norm_rows and xabe_kernels::norm::rms_norm.
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

// y = x / sqrt(mean(x^2) + eps) * weight, one row at a time.
//
// grid: one block per row. block: a whole number of warps, at most 1024;
// rows wider than the block are covered by a strided loop, which is what lets
// one kernel serve hidden 2048 and head_dim 128 without recompiling.
//
// `weight` is a single row of `width` floats shared by every row of `x`. That
// is exactly the shape of both uses: the hidden-size norms have one weight
// vector per layer applied to every token, and the per-head q/k and GDN
// output norms have one weight vector per layer applied to every (token,
// head) pair.
//
// The arithmetic transcribes the reference operand for operand: sum of
// squares, divide by width, add eps, one 1/sqrtf, then `xi * inv_rms * wi` in
// that order. rsqrtf would be faster and is not IEEE-exact; matching the
// reference's division keeps the disagreement attributable to the reduction
// order alone, as in the two GDN kernels.
__global__ void rms_norm_rows(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int width,
    float eps
) {
    extern __shared__ float scratch[];
    long long base = (long long)blockIdx.x * width;

    float partial = 0.0f;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        float v = x[base + j];
        partial += v * v;
    }
    float sum_sq = block_reduce_sum(partial, scratch);

    float inv_rms = 1.0f / sqrtf(sum_sq / (float)width + eps);

    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        out[base + j] = x[base + j] * inv_rms * weight[j];
    }
}

// Partial rotary embedding, NEOX (split-half) pairing: dimension i pairs with
// i + rope_dim/2, and dimensions [rope_dim, head_dim) pass through untouched.
//
// grid: (heads, tokens). block: head_dim threads, one per head element.
//
// **The tail is copied, not computed.** `out[base+j] = x[base+j]` for
// j >= rope_dim is a load and a store with no arithmetic between them, so the
// bits are preserved exactly and the differential test can gate that span
// with Tolerance::exact(). Any formulation that instead multiplied the tail
// by a cos of angle 0 would be "correct" to within an ulp and would fail that
// gate, which is the point of having it.
//
// Angles are computed in double precision because the reference does
// (`f64::from(theta_base).powf(...)`, `pos * freq`, then `as f32`). Turing
// runs fp64 at 1/32 rate, which is irrelevant here: this kernel is two loads
// and a store per thread and is bandwidth-bound, and computing the angle in
// fp32 would introduce a second disagreement with the reference on top of the
// one the tail gate exists to rule out.
//
// Each thread computes its own sin/cos rather than one thread per pair
// computing both, so that no thread writes two elements and the pass-through
// branch stays a plain copy.
__global__ void rope_partial(
    const float* __restrict__ x,
    const unsigned int* __restrict__ positions,
    float* __restrict__ out,
    int heads,
    int head_dim,
    int rope_dim,
    float theta_base
) {
    int h = blockIdx.x;
    int t = blockIdx.y;
    int j = threadIdx.x;
    long long base = ((long long)t * heads + h) * head_dim;

    if (j >= rope_dim) {
        out[base + j] = x[base + j];
        return;
    }

    int half = rope_dim >> 1;
    int i = (j < half) ? j : (j - half);

    double freq = pow((double)theta_base, -2.0 * (double)i / (double)rope_dim);
    double angle = (double)positions[t] * freq;
    float sin_a = (float)sin(angle);
    float cos_a = (float)cos(angle);

    float x0 = x[base + i];
    float x1 = x[base + i + half];
    out[base + j] = (j < half) ? (x0 * cos_a - x1 * sin_a)
                               : (x0 * sin_a + x1 * cos_a);
}

// out = silu(gate) * up, elementwise.
//
// grid-stride over a fixed grid so the launch shape does not depend on a
// host-side length, which is what keeps it capturable in a CUDA graph
// (AGENTS.md rule 5).
//
// `expf`, not `__expf`: the reference is `x / (1.0 + (-x).exp())` and the
// fast intrinsic is good to ~2 ulp of the *result*, not of the exponent,
// which on the negative tail is a relative error the tolerance would have to
// be widened for. This is not a bottleneck.
// RMSNorm and the SwiGLU multiply that consumes it, in one launch.
//
// The Gated DeltaNet's output norm is immediately multiplied by `silu(z)`,
// and the two kernels cover exactly the same elements: the norm's grid is one
// block per row of `width`, and the multiply is elementwise over
// `rows * width`. So the multiply rides along in the norm's second pass, and
// a decode step loses a launch per Gated DeltaNet layer.
//
// The reduction is untouched -- same block, same width, same
// `block_reduce_sum` -- so `normed` is bit-identical to what `rms_norm_rows`
// writes, and it is still written: it is an intermediate the block's own
// differential test compares.
//
// grid: (rows,). block: as `rms_norm_rows`.
__global__ void rms_norm_swiglu_rows(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ gate,
    float* __restrict__ normed,
    float* __restrict__ out,
    int width,
    float eps
) {
    extern __shared__ float scratch[];
    long long base = (long long)blockIdx.x * width;

    float partial = 0.0f;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        float v = x[base + j];
        partial += v * v;
    }
    float sum_sq = block_reduce_sum(partial, scratch);

    float inv_rms = 1.0f / sqrtf(sum_sq / (float)width + eps);

    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        float nv = x[base + j] * inv_rms * weight[j];
        normed[base + j] = nv;
        float g = gate[base + j];
        out[base + j] = (g / (1.0f + expf(-g))) * nv;
    }
}

__global__ void swiglu_mul(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    long long n
) {
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

// out = sigmoid(gate) * x, with the gate either x's own shape or one scalar
// per row broadcast across it.
//
// **Two shapes, because the model has two of these and they are not the same
// op.** Assuming one and reindexing at the call site would work for whichever
// caller guessed right and would silently read `rows` floats as `rows * width`
// for the other, so `broadcast` is an argument and the gate length is checked
// against it on every launch:
//
//  - the Gated Attention output gate is **elementwise** over
//    [tokens][q_heads * head_dim] — 4096 gate values per token
//    (`attn_gate` in `src/models/qwen35moe.cpp`, whose sigmoid is
//    `attn_gate_sigmoid`);
//  - the MoE shared expert's gate is **one scalar per token** —
//    `ffn_gate_inp_shexp` is a single `[hidden]` row, so its projection
//    collapses to a scalar — broadcast over the whole hidden dimension
//    (`build_layer_ffn`).
//
// `1.0f / (1.0f + expf(-g))` is ggml_sigmoid verbatim (`op_sigmoid` in
// `ggml/src/ggml-cuda/unary.cu`). The algebraically equal forms —
// 0.5f * (1 + tanhf(x/2)), or silu(x)/x — round differently, and
// `xabe_kernels::norm::sigmoid` spells it this same way.
//
// `sig` receives the nonlinearity on its own, so a divergence localizes to
// the sigmoid or to the product rather than to "the gate", and so the caller
// keeps llama.cpp's `attn_gate_sigmoid` waypoint. It is **gate's** shape, not
// x's: when broadcasting, only the thread holding column 0 of a row writes
// it, so no two threads write the same element.
__global__ void sigmoid_gate_mul(
    const float* __restrict__ x,
    const float* __restrict__ gate,
    float* __restrict__ sig,
    float* __restrict__ out,
    long long n,
    int width,
    int broadcast
) {
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        long long g = i;
        bool write_sig = true;
        if (broadcast) {
            long long row = i / (long long)width;
            write_sig = (i - row * (long long)width) == 0;
            g = row;
        }
        float s = 1.0f / (1.0f + expf(-gate[g]));
        if (write_sig) sig[g] = s;
        out[i] = x[i] * s;
    }
}

// out = a + b, elementwise. The residual add.
//
// One rounding, exactly as `xabe_kernels::norm::residual_add`, so this is
// bit-identical to the reference rather than close to it and the differential
// test gates it with Tolerance::exact(). Unlike the convolution it needs no
// __fadd_rn to get there: a lone add has nothing to contract into an FMA.
//
// There is no fused alternative to check this against — every residual add in
// the model currently has nowhere to go — so the whole content of this kernel
// is that the operands are added in the order given and nothing else happens
// to them.
__global__ void tensor_add(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    long long n
) {
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}

// out = log(1 + exp(x)), with a passthrough above 20. The GDN alpha gate's
// nonlinearity: `a_softplus = softplus(alpha + ssm_dt.bias)`, which is then
// multiplied by the already-negated `ssm_a` to give the log-decay
// (`qwen35moe.cpp`, `alpha_softplus` / `gate`).
//
// `(x > 20.0f) ? x : logf(1.0f + expf(x))` is ggml's formulation verbatim.
// The same expression appears three times in llama.cpp — `op_softplus` in
// `ggml/src/ggml-cuda/unary.cu` (the CUDA path), `op_softplus` in
// `ggml/src/ggml-cpu/unary-ops.cpp`, and `ggml_compute_softplus_f32` in
// `ggml/src/ggml-impl.h`.
//
// **The threshold is a correctness guard, not an optimization.** expf
// overflows fp32 above x ~ 88.7, so a branch-free softplus returns inf where
// the answer is x, and that inf becomes a non-finite log-decay that poisons
// the whole recurrence. Where the branch is taken it is also invisible:
// log(1 + e^20) - 20 is ~2.06e-9, about a thousandth of an fp32 ulp at that
// magnitude.
//
// (ggml's SYCL backend uses the numerically better
// `max(x,0) + log1p(exp(-|x|))` — `op_softplus` in
// `ggml/src/ggml-sycl/element_wise.cpp` — but that is not the form the CUDA
// and CPU paths take, and Qwen3.6's `a_softplus` comes off the CUDA one.)
__global__ void softplus_elementwise(
    const float* __restrict__ x,
    float* __restrict__ out,
    long long n
) {
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float v = x[i];
        out[i] = (v > 20.0f) ? v : logf(1.0f + expf(v));
    }
}

// Causal depthwise convolution over the fused GDN q/k/v stream.
//
// out[t][ch] = sum_{i<K} w[ch][i] * xw(t - (K-1) + i, ch)
//
// where xw reads this batch's tokens for non-negative positions and the
// carried cache for negative ones. Tap K-1 multiplies the *current* token and
// tap 0 the oldest, which is `ggml_compute_forward_ssm_conv_f32`'s window
// order (`ggml/src/ggml-cpu/ops.cpp`) and is the thing a reversed loop gets
// silently backwards.
//
// grid: (ceil(channels / 256), tokens). block: 256 threads over channels, so
// consecutive threads read consecutive channels of one token — the layout is
// [seq_len][channels] with channels contiguous, so every load coalesces.
//
// __fmul_rn/__fadd_rn rather than `acc += x * w`: see the module docs. The
// point is to deny nvcc the FMA contraction so this kernel is bit-identical
// to the scalar reference rather than merely close to it.
__global__ void conv1d_causal_depthwise(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ state,
    float* __restrict__ out,
    int channels,
    int conv_kernel
) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;
    int t = blockIdx.y;
    int carry = conv_kernel - 1;

    float acc = 0.0f;
    for (int i = 0; i < conv_kernel; ++i) {
        int u = t - carry + i;
        float xv = (u >= 0)
            ? x[(long long)u * channels + ch]
            : state[(long long)ch * carry + (u + carry)];
        acc = __fadd_rn(acc, __fmul_rn(xv, weight[(long long)ch * conv_kernel + i]));
    }
    out[(long long)t * channels + ch] = acc;
}

// The whole convolution for a one-token step: output and cache advance in
// one launch.
//
// `conv1d_causal_depthwise` and `conv1d_update_state` are two kernels because
// a batch's cache advance depends on the *last* `conv_kernel - 1` inputs of
// the whole batch, which the per-token blocks of the first kernel cannot see.
// At one token there is no batch: the window is the cache plus this token,
// and the new cache is the old one shifted by one with this token appended.
// A thread owns its channel's whole cache row, so the shift is safe in place
// once the row is staged in registers -- the same argument
// `conv1d_update_state` already makes, with `seq_len` pinned to 1.
//
// The accumulation keeps `__fadd_rn(acc, __fmul_rn(...))` in ascending tap
// order, so the output is bit-identical to the two-kernel path rather than
// merely close: see the module docs on why the FMA contraction is denied here.
//
// grid: (ceil(channels / 256),). block: 256.
__global__ void conv1d_step(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ state,
    float* __restrict__ out,
    int channels,
    int conv_kernel
) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;
    int carry = conv_kernel - 1;

    float staged[MAX_CONV_KERNEL];
    for (int j = 0; j < carry; ++j) {
        staged[j] = state[(long long)ch * carry + j];
    }
    float xv0 = x[ch];

    float acc = 0.0f;
    for (int i = 0; i < conv_kernel; ++i) {
        float xv = (i < carry) ? staged[i] : xv0;
        acc = __fadd_rn(acc, __fmul_rn(xv, weight[(long long)ch * conv_kernel + i]));
    }
    out[ch] = acc;

    for (int j = 0; j + 1 < carry; ++j) {
        state[(long long)ch * carry + j] = staged[j + 1];
    }
    if (carry > 0) {
        state[(long long)ch * carry + carry - 1] = xv0;
    }
}

// Advance the convolution cache to the last conv_kernel-1 inputs.
//
// grid-stride over channels, one thread per channel. Every window value is
// staged into registers before anything is written, so the update is safe
// in place: a thread owns its channel's whole cache row and no other thread
// touches it. That matters because a batch shorter than conv_kernel-1 — every
// decode step — reads entries of the old cache that it also overwrites.
// llama.cpp gets the same effect structurally, by taking the new state as a
// view into a freshly concatenated buffer
// (`llm_build_delta_net_base::build_conv_state`, `s_idx = conv_input->ne[0] -
// conv_states->ne[0]`); we have no such buffer and must not need one.
__global__ void conv1d_update_state(
    const float* __restrict__ x,
    float* __restrict__ state,
    int seq_len,
    int channels,
    int conv_kernel
) {
    int carry = conv_kernel - 1;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long ch = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         ch < channels;
         ch += stride) {
        float staged[MAX_CONV_KERNEL];
        for (int j = 0; j < carry; ++j) {
            int u = seq_len - carry + j;
            staged[j] = (u >= 0)
                ? x[(long long)u * channels + ch]
                : state[ch * carry + (u + carry)];
        }
        for (int j = 0; j < carry; ++j) {
            state[ch * carry + j] = staged[j];
        }
    }
}

}
"#;

/// Something went wrong compiling or launching a layer-op kernel.
#[derive(Debug)]
pub enum LayerOpsError {
    /// NVRTC rejected the source, or the module failed to load.
    Compile(String),
    /// The driver failed.
    Driver(DriverError),
    /// A row width the kernel cannot service.
    UnsupportedWidth { width: usize },
    /// A head geometry the rotary kernel cannot service.
    UnsupportedRotaryShape { head_dim: usize, rope_dim: usize },
    /// A convolution width the kernel cannot service.
    UnsupportedConvKernel { conv_kernel: usize },
    /// A buffer is not the length the declared geometry requires.
    ShapeMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for LayerOpsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compile(m) => write!(f, "kernel compilation failed: {m}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::UnsupportedWidth { width } => {
                write!(f, "row width {width} must be non-zero")
            }
            Self::UnsupportedRotaryShape { head_dim, rope_dim } => write!(
                f,
                "rope_dim {rope_dim} must be even and at most head_dim {head_dim}, which must \
                 itself be non-zero and at most 1024 (it is the block width)",
            ),
            Self::UnsupportedConvKernel { conv_kernel } => write!(
                f,
                "conv_kernel {conv_kernel} must be in 1..={MAX_CONV_KERNEL}",
            ),
            Self::ShapeMismatch {
                what,
                expected,
                got,
            } => write!(f, "{what} must hold {expected} floats, holds {got}"),
        }
    }
}

impl std::error::Error for LayerOpsError {}

impl From<DriverError> for LayerOpsError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

/// The shape of the gate tensor [`LayerOpsKernels::sigmoid_gate`] is given.
///
/// Qwen3.6 has two sigmoid gates and they are not the same op. This is an
/// argument rather than an assumption because guessing wrong is not a crash:
/// a broadcast gate read elementwise walks `rows * width` floats off the end
/// of a `rows`-float buffer, and an elementwise gate read per-row applies
/// every token's first gate value to that whole token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateShape {
    /// One gate value per element: `gate` is `[rows][width]`, like `x`.
    ///
    /// The Gated Attention output gate — `attn_gate` in
    /// `src/models/qwen35moe.cpp`, `q_heads * head_dim = 4096` values per
    /// token.
    Elementwise,
    /// One gate value per row, broadcast across that row: `gate` is `[rows]`.
    ///
    /// The MoE shared expert's gate. `ffn_gate_inp_shexp` is a single
    /// `[hidden]` row, so its projection collapses to one scalar per token,
    /// which `build_layer_ffn` multiplies the whole shared-expert output by.
    PerRow,
}

impl GateShape {
    /// How many gate values a `[rows][width]` activation needs in this shape.
    ///
    /// Also the length of the `sig` output, which is the gate's shape rather
    /// than `x`'s.
    const fn gate_len(self, rows: usize, width: usize) -> usize {
        match self {
            Self::Elementwise => rows * width,
            Self::PerRow => rows,
        }
    }
}

/// The compiled layer-op kernels.
pub struct LayerOpsKernels {
    rms_norm: CudaFunction,
    rms_norm_swiglu: CudaFunction,
    rope: CudaFunction,
    swiglu: CudaFunction,
    sigmoid_gate: CudaFunction,
    add: CudaFunction,
    softplus: CudaFunction,
    conv1d: CudaFunction,
    conv1d_state: CudaFunction,
    conv1d_step: CudaFunction,
}

impl LayerOpsKernels {
    /// Compile every layer op into one module.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, LayerOpsError> {
        let ptx = compile(LAYER_OPS_SRC, "layer_ops").map_err(LayerOpsError::Compile)?;
        let module = ctx.load_module(ptx)?;
        Ok(Self {
            rms_norm: module.load_function("rms_norm_rows")?,
            rms_norm_swiglu: module.load_function("rms_norm_swiglu_rows")?,
            rope: module.load_function("rope_partial")?,
            swiglu: module.load_function("swiglu_mul")?,
            sigmoid_gate: module.load_function("sigmoid_gate_mul")?,
            add: module.load_function("tensor_add")?,
            softplus: module.load_function("softplus_elementwise")?,
            conv1d: module.load_function("conv1d_causal_depthwise")?,
            conv1d_state: module.load_function("conv1d_update_state")?,
            conv1d_step: module.load_function("conv1d_step")?,
        })
    }

    /// RMSNorm over `[rows][width]`, then `out = silu(gate) * normed`.
    ///
    /// One launch for what `rms_norm` and `swiglu` did in two. The normed
    /// intermediate is still written; see the kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_swiglu(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        normed: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        width: usize,
        eps: f32,
    ) -> Result<(), LayerOpsError> {
        if width == 0 {
            return Err(LayerOpsError::UnsupportedWidth { width });
        }
        let n = rows * width;
        check_len("rms_norm x", n, x.len())?;
        check_len("rms_norm weight", width, weight.len())?;
        check_len("swiglu gate", n, gate.len())?;
        check_len("rms_norm out", n, normed.len())?;
        check_len("swiglu out", n, out.len())?;
        if rows == 0 {
            return Ok(());
        }

        let block = block_for_width(width);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block as u32, 1, 1),
            shared_mem_bytes: (block.div_ceil(32) * size_of::<f32>()) as u32,
        };
        let width_i32 = width as i32;
        let mut builder = stream.launch_builder(&self.rms_norm_swiglu);
        builder
            .arg(x)
            .arg(weight)
            .arg(gate)
            .arg(&mut *normed)
            .arg(&mut *out)
            .arg(&width_i32)
            .arg(&eps);
        // SAFETY: as `rms_norm` below, with `gate` and `out` checked to the
        // same `rows * width` and indexed by the same expression.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// RMSNorm over `rows` independent rows of `width` elements each.
    ///
    /// `x` and `out` are `[rows][width]`; `weight` is one `width`-long vector
    /// shared by every row. Both hidden-size norms (`rows = tokens`,
    /// `width = 2048`) and per-head norms (`rows = tokens * heads`,
    /// `width = head_dim`) are this one call.
    ///
    /// `out` may alias `x`: every thread reads `x[base + j]` and writes
    /// `out[base + j]` for the same `j`, after the reduction's final
    /// `__syncthreads()`, so no thread can read an element another has
    /// already overwritten.
    ///
    /// Eight plain arguments rather than a config struct, for the same reason
    /// [`super::gdn_chunked::GdnChunkedKernels::prefill`] takes ten: every one
    /// is a distinct per-call tensor or shape, not related configuration that
    /// would be set once and reused.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        width: usize,
        eps: f32,
    ) -> Result<(), LayerOpsError> {
        if width == 0 {
            return Err(LayerOpsError::UnsupportedWidth { width });
        }
        check_len("rms_norm x", rows * width, x.len())?;
        check_len("rms_norm weight", width, weight.len())?;
        check_len("rms_norm out", rows * width, out.len())?;
        if rows == 0 {
            return Ok(());
        }

        let block = block_for_width(width);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block as u32, 1, 1),
            // One float per warp, which is the most `block_reduce_sum` stores.
            shared_mem_bytes: (block.div_ceil(32) * size_of::<f32>()) as u32,
        };
        let width_i32 = width as i32;
        let mut builder = stream.launch_builder(&self.rms_norm);
        builder
            .arg(x)
            .arg(weight)
            .arg(out)
            .arg(&width_i32)
            .arg(&eps);
        // SAFETY: one block per row over buffers of `rows * width` floats,
        // with the in-row loop bounded by `width`, so `blockIdx.x * width + j`
        // is in range for every thread. `weight` is indexed by `j < width` and
        // holds `width` floats. Shared memory covers one float per warp, which
        // is all the reduction writes.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// Partial rotary embedding over `[seq_len][heads][head_dim]`.
    ///
    /// `positions` holds one absolute position per token. Dimensions
    /// `[rope_dim, head_dim)` are copied through bit-exactly.
    ///
    /// Unlike [`Self::rms_norm`], **`out` must not alias `x`**: element `j`
    /// of a rotated pair is written from elements `i` and `i + half`, so an
    /// in-place kernel would race. Enforced by the borrow checker taking `x`
    /// shared and `out` exclusive.
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        out: &mut CudaSlice<f32>,
        seq_len: usize,
        heads: usize,
        head_dim: usize,
        rope_dim: usize,
        theta_base: f32,
    ) -> Result<(), LayerOpsError> {
        if head_dim == 0
            || head_dim > MAX_BLOCK
            || !rope_dim.is_multiple_of(2)
            || rope_dim > head_dim
        {
            return Err(LayerOpsError::UnsupportedRotaryShape { head_dim, rope_dim });
        }
        let n = seq_len * heads * head_dim;
        check_len("rope x", n, x.len())?;
        check_len("rope out", n, out.len())?;
        check_len("rope positions", seq_len, positions.len())?;
        if seq_len == 0 || heads == 0 {
            return Ok(());
        }

        let cfg = LaunchConfig {
            grid_dim: (heads as u32, seq_len as u32, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let heads_i32 = heads as i32;
        let head_dim_i32 = head_dim as i32;
        let rope_dim_i32 = rope_dim as i32;
        let mut builder = stream.launch_builder(&self.rope);
        builder
            .arg(x)
            .arg(positions)
            .arg(out)
            .arg(&heads_i32)
            .arg(&head_dim_i32)
            .arg(&rope_dim_i32)
            .arg(&theta_base);
        // SAFETY: the grid is (heads, seq_len) and the block is head_dim
        // threads, so `(t * heads + h) * head_dim + j` covers exactly the
        // `seq_len * heads * head_dim` floats both buffers hold. The paired
        // reads at `i` and `i + half` stay below `rope_dim <= head_dim`.
        // `positions` is indexed by `t < seq_len` and holds `seq_len` entries.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// `out = silu(gate) * up` over `n` elements.
    ///
    /// The standalone form. [`super::moe`]'s grouped GEMM fuses the same
    /// activation into its epilogue; this exists for the paths that do not go
    /// through it and as the thing the fused version is checked against.
    pub fn swiglu(
        &self,
        stream: &Arc<CudaStream>,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), LayerOpsError> {
        check_len("swiglu gate", n, gate.len())?;
        check_len("swiglu up", n, up.len())?;
        check_len("swiglu out", n, out.len())?;
        if n == 0 {
            return Ok(());
        }

        let n_i64 = n as i64;
        let mut builder = stream.launch_builder(&self.swiglu);
        builder.arg(gate).arg(up).arg(out).arg(&n_i64);
        // SAFETY: the grid-stride loop is bounded by `n`, which is the length
        // checked above for all three buffers.
        unsafe { builder.launch(elementwise_cfg(n)) }?;
        Ok(())
    }

    /// `out = sigmoid(gate) * x` over a `[rows][width]` activation, writing
    /// the sigmoid itself to `sig`.
    ///
    /// `shape` says whether `gate` is elementwise or one scalar per row; see
    /// [`GateShape`]. `gate` and `sig` are both that shape's length —
    /// `rows * width` or `rows` — and `x` and `out` are always `rows * width`.
    /// Every one of those lengths is checked here rather than assumed, which
    /// is the point of taking the shape as an argument at all.
    ///
    /// `sig` exists so a divergence localizes to the nonlinearity or to the
    /// product rather than to "the gate", and so the caller keeps llama.cpp's
    /// `attn_gate_sigmoid` waypoint to compare against. In [`GateShape::PerRow`]
    /// it costs `rows` floats.
    ///
    /// `out` may alias `x`: each thread reads and writes the same index and
    /// reads `gate`, which is a different buffer. `sig` must not alias either.
    #[allow(clippy::too_many_arguments)]
    pub fn sigmoid_gate(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        sig: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        width: usize,
        shape: GateShape,
    ) -> Result<(), LayerOpsError> {
        if width == 0 {
            return Err(LayerOpsError::UnsupportedWidth { width });
        }
        let n = rows * width;
        let gate_len = shape.gate_len(rows, width);
        check_len("sigmoid_gate x", n, x.len())?;
        check_len("sigmoid_gate gate", gate_len, gate.len())?;
        check_len("sigmoid_gate sigmoid", gate_len, sig.len())?;
        check_len("sigmoid_gate out", n, out.len())?;
        if rows == 0 {
            return Ok(());
        }

        let n_i64 = n as i64;
        let width_i32 = width as i32;
        let broadcast_i32 = i32::from(shape == GateShape::PerRow);
        let mut builder = stream.launch_builder(&self.sigmoid_gate);
        builder
            .arg(x)
            .arg(gate)
            .arg(sig)
            .arg(out)
            .arg(&n_i64)
            .arg(&width_i32)
            .arg(&broadcast_i32);
        // SAFETY: the grid-stride loop is bounded by `n = rows * width`, the
        // checked length of `x` and `out`. The gate index is `i` when not
        // broadcasting and `i / width < rows` when broadcasting, which are
        // exactly the two lengths `gate` and `sig` were checked against just
        // above; `width != 0` so the division is defined.
        unsafe { builder.launch(elementwise_cfg(n)) }?;
        Ok(())
    }

    /// `out = a + b` over `n` elements — the residual add.
    ///
    /// Bit-identical to [`xabe_kernels::norm::residual_add`], not merely
    /// close: one rounding per element, in the operand order given.
    ///
    /// `out` may alias either input; each thread touches one index of each.
    pub fn add(
        &self,
        stream: &Arc<CudaStream>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), LayerOpsError> {
        check_len("add a", n, a.len())?;
        check_len("add b", n, b.len())?;
        check_len("add out", n, out.len())?;
        if n == 0 {
            return Ok(());
        }

        let n_i64 = n as i64;
        let mut builder = stream.launch_builder(&self.add);
        builder.arg(a).arg(b).arg(out).arg(&n_i64);
        // SAFETY: the grid-stride loop is bounded by `n`, which is the length
        // checked above for all three buffers.
        unsafe { builder.launch(elementwise_cfg(n)) }?;
        Ok(())
    }

    /// `out = log(1 + exp(x))` over `n` elements, with ggml's passthrough
    /// above `x = 20`.
    ///
    /// The GDN alpha gate's nonlinearity. The passthrough is not optional:
    /// without it `expf` overflows above `x ≈ 88.7` and the result is `inf`
    /// where the answer is `x`. See [`xabe_kernels::norm::softplus`].
    ///
    /// `out` may alias `x`.
    pub fn softplus(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), LayerOpsError> {
        check_len("softplus x", n, x.len())?;
        check_len("softplus out", n, out.len())?;
        if n == 0 {
            return Ok(());
        }

        let n_i64 = n as i64;
        let mut builder = stream.launch_builder(&self.softplus);
        builder.arg(x).arg(out).arg(&n_i64);
        // SAFETY: the grid-stride loop is bounded by `n`, which is the length
        // checked above for both buffers.
        unsafe { builder.launch(elementwise_cfg(n)) }?;
        Ok(())
    }

    /// The GDN short causal depthwise convolution, advancing `state`.
    ///
    /// `x` and `out` are `[seq_len][channels]` with channels contiguous;
    /// `weight` is `[channels][conv_kernel]`, the `ssm_conv1d.weight` layout
    /// verbatim; `state` is `[channels][conv_kernel - 1]` oldest-first and is
    /// updated in place to the last `conv_kernel - 1` inputs.
    ///
    /// Two launches, not one: the state update must observe the *old* state,
    /// which the convolution also reads, so they cannot be fused without
    /// either a second state buffer or a grid-wide barrier.
    ///
    /// This is the pure convolution, matching `ggml_ssm_conv`'s boundary.
    /// llama.cpp applies SiLU to the result as a separate op before slicing
    /// q/k/v apart (`src/models/qwen35moe.cpp`, `conv_output_silu`); that is
    /// the caller's job here too.
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        seq_len: usize,
        channels: usize,
        conv_kernel: usize,
    ) -> Result<(), LayerOpsError> {
        if conv_kernel == 0 || conv_kernel > MAX_CONV_KERNEL {
            return Err(LayerOpsError::UnsupportedConvKernel { conv_kernel });
        }
        if channels == 0 {
            return Err(LayerOpsError::UnsupportedWidth { width: channels });
        }
        check_len("conv1d x", seq_len * channels, x.len())?;
        check_len("conv1d weight", channels * conv_kernel, weight.len())?;
        check_len("conv1d state", channels * (conv_kernel - 1), state.len())?;
        check_len("conv1d out", seq_len * channels, out.len())?;
        if seq_len == 0 {
            return Ok(());
        }

        const BLOCK: usize = 256;
        let channels_i32 = channels as i32;
        let conv_kernel_i32 = conv_kernel as i32;
        let seq_len_i32 = seq_len as i32;

        // One token needs no batch-wide view of the window, so the output and
        // the cache advance are one launch. See `conv1d_step`.
        if seq_len == 1 {
            let cfg = LaunchConfig {
                grid_dim: (channels.div_ceil(BLOCK) as u32, 1, 1),
                block_dim: (BLOCK as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = stream.launch_builder(&self.conv1d_step);
            builder
                .arg(x)
                .arg(weight)
                .arg(&mut *state)
                .arg(&mut *out)
                .arg(&channels_i32)
                .arg(&conv_kernel_i32);
            // SAFETY: threads past `channels` return before touching memory;
            // `state` is indexed in `[ch * carry, ch * carry + carry)` and
            // `weight` in `[ch * conv_kernel, ...)`, both length-checked
            // above, and `conv_kernel <= MAX_CONV_KERNEL` is what makes the
            // `staged` register array large enough.
            unsafe { builder.launch(cfg) }?;
            return Ok(());
        }

        let cfg = LaunchConfig {
            grid_dim: (channels.div_ceil(BLOCK) as u32, seq_len as u32, 1),
            block_dim: (BLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&self.conv1d);
        builder
            .arg(x)
            .arg(weight)
            .arg(&*state)
            .arg(&mut *out)
            .arg(&channels_i32)
            .arg(&conv_kernel_i32);
        // SAFETY: threads past `channels` return before touching memory. The
        // window position `t - (conv_kernel - 1) + i` is either in
        // `[0, seq_len)` and indexes `x`, or in `[-(conv_kernel-1), 0)` and
        // indexes `state` at `ch * carry + (u + carry)` which is in
        // `[0, carry)`. Both buffers were length-checked above.
        unsafe { builder.launch(cfg) }?;

        let state_cfg = LaunchConfig {
            grid_dim: (channels.div_ceil(BLOCK).min(1024) as u32, 1, 1),
            block_dim: (BLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&self.conv1d_state);
        builder
            .arg(x)
            .arg(&mut *state)
            .arg(&seq_len_i32)
            .arg(&channels_i32)
            .arg(&conv_kernel_i32);
        // SAFETY: same index bounds as above, and `conv_kernel <=
        // MAX_CONV_KERNEL` is what makes the kernel's `staged` register array
        // large enough — checked at the top of this function, which is the
        // only path that launches it.
        unsafe { builder.launch(state_cfg) }?;
        Ok(())
    }
}

/// Block width for a row of `width` elements: a whole number of warps, capped
/// at [`MAX_BLOCK`], with wider rows covered by the kernel's strided loop.
///
/// A whole number of warps is required, not preferred: `block_reduce_sum`
/// shuffles with a full `0xffffffff` mask, so every lane of every warp in the
/// block must reach it. Threads past `width` contribute a zero partial.
fn block_for_width(width: usize) -> usize {
    width.next_multiple_of(32).clamp(32, MAX_BLOCK)
}

/// Launch geometry for the four grid-stride elementwise kernels.
///
/// A grid capped at 1024 blocks rather than one derived from `n`: grid-stride
/// keeps the launch shape independent of a host-side length, so the launch can
/// be captured in a CUDA graph and replayed at a different `n` (`AGENTS.md`
/// rule 5). The cap is what makes the shape *fixed*; without it the grid would
/// track `n` and every new length would need a new capture.
fn elementwise_cfg(n: usize) -> LaunchConfig {
    const BLOCK: usize = 256;
    LaunchConfig {
        grid_dim: (n.div_ceil(BLOCK).min(1024) as u32, 1, 1),
        block_dim: (BLOCK as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Rejects a buffer whose length disagrees with the declared geometry.
fn check_len(what: &'static str, expected: usize, got: usize) -> Result<(), LayerOpsError> {
    if got == expected {
        Ok(())
    } else {
        Err(LayerOpsError::ShapeMismatch {
            what,
            expected,
            got,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Where a snippet starts in the kernel source, or a failure naming it.
    fn at(needle: &str) -> usize {
        LAYER_OPS_SRC
            .find(needle)
            .unwrap_or_else(|| panic!("kernel source no longer contains `{needle}`"))
    }

    #[test]
    fn the_rotary_tail_is_copied_not_computed() {
        // The documented invariant, asserted structurally because it is a
        // property of the *code path*, not of any particular input: the
        // pass-through branch must be a bare load-store with no arithmetic,
        // or the bits stop being preserved and Tolerance::exact() stops being
        // reachable. Multiplying by cos(0) would be correct to an ulp and
        // would fail the differential gate.
        assert!(LAYER_OPS_SRC.contains(
            "    if (j >= rope_dim) {\n        out[base + j] = x[base + j];\n        return;\n    }"
        ));
    }

    #[test]
    fn the_rotary_pairing_is_split_half_neox_not_interleaved() {
        // NEOX pairs i with i + rope_dim/2. The other common convention pairs
        // 2i with 2i+1; both produce a rotation, both preserve norm, and only
        // one is Qwen's. `xabe_kernels::rope` cites
        // ggml_compute_forward_rope_flt's GGML_ROPE_TYPE_NEOX case for this.
        assert!(LAYER_OPS_SRC.contains("int half = rope_dim >> 1;"));
        assert!(LAYER_OPS_SRC.contains("float x1 = x[base + i + half];"));
    }

    #[test]
    fn the_rotation_signs_are_the_forward_rotation_not_its_transpose() {
        // out[i] = x0 cos - x1 sin, out[i+half] = x0 sin + x1 cos. Swapping
        // the two signs rotates by -position instead of +position, which is
        // undetectable by norm preservation and by position 0.
        assert!(LAYER_OPS_SRC.contains("(x0 * cos_a - x1 * sin_a)"));
        assert!(LAYER_OPS_SRC.contains("(x0 * sin_a + x1 * cos_a)"));
    }

    #[test]
    fn the_convolution_denies_the_compiler_its_fma_contraction() {
        // The one thing standing between this kernel and bit-exact agreement
        // with the scalar reference. `acc += x * w` would let nvcc emit an
        // fma, which rounds once where Rust rounds twice, and the
        // differential test's Tolerance::exact() would fail.
        assert!(LAYER_OPS_SRC.contains("acc = __fadd_rn(acc, __fmul_rn(xv,"));
        let conv_start = at("__global__ void conv1d_causal_depthwise(");
        let conv_end = at("__global__ void conv1d_update_state(");
        let body = &LAYER_OPS_SRC[conv_start..conv_end];
        assert!(
            !body.contains("acc +="),
            "the accumulation reverted to a contractible form",
        );
    }

    #[test]
    fn the_convolution_taps_run_forward_so_the_last_tap_is_the_current_token() {
        // u = t - (K-1) + i with i ascending: tap 0 reads the oldest input and
        // tap K-1 reads token t. Reversing this is anti-causal and stays
        // finite, fluent, and wrong.
        assert!(LAYER_OPS_SRC.contains("int u = t - carry + i;"));
        assert!(LAYER_OPS_SRC.contains("for (int i = 0; i < conv_kernel; ++i) {"));
    }

    #[test]
    fn the_convolution_reads_only_the_current_token_and_its_predecessors() {
        // Causality as a bound on the index rather than as a comment: the
        // largest window position any thread forms is `t - carry + (K-1)`,
        // which is `t` exactly. Checked arithmetically for the real geometry
        // so a future edit to the loop bound has to break this to land.
        let conv_kernel = 4i32;
        let carry = conv_kernel - 1;
        for t in 0..8i32 {
            let positions: Vec<i32> = (0..conv_kernel).map(|i| t - carry + i).collect();
            assert_eq!(*positions.iter().max().unwrap(), t);
            assert_eq!(*positions.iter().min().unwrap(), t - carry);
        }
    }

    #[test]
    fn the_state_update_stages_before_it_writes() {
        // In-place safety. A batch shorter than conv_kernel-1 — which is every
        // decode step — reads entries of the cache it also overwrites, so the
        // read loop must complete before the write loop begins.
        let read = at("staged[j] = (u >= 0)");
        let write = at("state[ch * carry + j] = staged[j];");
        assert!(read < write, "the cache is written before it is fully read");
    }

    #[test]
    fn the_state_array_bound_covers_the_widest_supported_kernel() {
        // The register array is `float staged[MAX_CONV_KERNEL]` and holds
        // conv_kernel-1 entries, so the host-side bound must not exceed it.
        assert!(LAYER_OPS_SRC.contains("#define MAX_CONV_KERNEL 8"));
        assert_eq!(MAX_CONV_KERNEL, 8);
        // Qwen3.6's real width — `qwen35moe.ssm.conv_kernel` — must be
        // accepted. Spelled as a literal because this crate deliberately does
        // not depend on `xabe-model`; the differential test in `xabe-engine`
        // reads it from `ModelConfig` and would fail if the two disagreed.
        // A const block, so a bound that stopped covering the model would be
        // a compile error rather than a test failure.
        const { assert!(4 <= MAX_CONV_KERNEL) };
    }

    #[test]
    fn rms_norm_divides_by_sqrt_rather_than_calling_rsqrtf() {
        // Same choice as both GDN kernels: rsqrtf is faster and not
        // IEEE-exact, and the reference divides. Matching it keeps the
        // measured disagreement attributable to the reduction order alone.
        assert!(
            LAYER_OPS_SRC.contains("float inv_rms = 1.0f / sqrtf(sum_sq / (float)width + eps);")
        );
        assert!(LAYER_OPS_SRC.contains("out[base + j] = x[base + j] * inv_rms * weight[j];"));
        // The negative half, scoped to the kernel body so that the comment
        // explaining the choice does not satisfy its own assertion.
        let body = &LAYER_OPS_SRC[at("__global__ void rms_norm_rows(")..at("// Partial rotary")];
        assert!(!body.contains("rsqrtf"), "rms_norm reverted to rsqrtf");
    }

    #[test]
    fn swiglu_uses_the_accurate_exponential() {
        assert!(LAYER_OPS_SRC.contains("out[i] = (g / (1.0f + expf(-g))) * up[i];"));
        let body = &LAYER_OPS_SRC[at("__global__ void swiglu_mul(")..at("// out = sigmoid(gate)")];
        assert!(!body.contains("__expf"), "swiglu reverted to __expf");
    }

    #[test]
    fn the_sigmoid_is_ggmls_reciprocal_form_and_not_an_algebraic_equivalent() {
        // `op_sigmoid` in ggml/src/ggml-cuda/unary.cu is 1/(1+expf(-x)). The
        // tanh form and silu(x)/x are equal on paper and round differently,
        // and this kernel is gated against the reciprocal one.
        assert!(LAYER_OPS_SRC.contains("float s = 1.0f / (1.0f + expf(-gate[g]));"));
        let body = &LAYER_OPS_SRC
            [at("__global__ void sigmoid_gate_mul(")..at("// out = a + b, elementwise")];
        assert!(
            !body.contains("tanh"),
            "the sigmoid was rewritten as a tanh"
        );
        assert!(!body.contains("__expf"), "the gate reverted to __expf");
    }

    #[test]
    fn the_sigmoid_gate_indexes_the_gate_by_row_only_when_it_broadcasts() {
        // The whole point of the `broadcast` argument: the attention gate is
        // elementwise and the shared expert's is one scalar per token. A
        // kernel that hardcoded either would read the other's buffer at the
        // wrong stride and stay finite while being wrong.
        assert!(LAYER_OPS_SRC.contains("long long row = i / (long long)width;"));
        assert!(LAYER_OPS_SRC.contains("            g = row;"));
        assert!(LAYER_OPS_SRC.contains("        long long g = i;"));
    }

    #[test]
    fn the_broadcast_sigmoid_is_written_once_per_row_not_once_per_element() {
        // `sig` is gate's shape, so in broadcast mode `width` threads share
        // one output element. They would all store the same value, but a
        // guarded single writer is the difference between a benign race and a
        // documented one — and it is what makes the differential test's
        // "every sig entry was written" sentinel check meaningful.
        assert!(
            LAYER_OPS_SRC.contains("write_sig = (i - row * (long long)width) == 0;"),
            "the broadcast sigmoid lost its single-writer guard",
        );
        assert!(LAYER_OPS_SRC.contains("if (write_sig) sig[g] = s;"));
    }

    #[test]
    fn the_residual_add_is_a_single_rounding() {
        // Nothing fused, nothing reassociated: `out[i] = a[i] + b[i]` is the
        // reference's exact operand sequence, which is what lets the
        // differential test gate it at exact equality rather than a
        // tolerance. A scale folded in here would still look plausible.
        assert!(LAYER_OPS_SRC.contains("        out[i] = a[i] + b[i];"));
        let body =
            &LAYER_OPS_SRC[at("__global__ void tensor_add(")..at("// out = log(1 + exp(x))")];
        assert!(!body.contains("fma"), "the residual add grew an FMA");
        // Scoped to the statement, not the whole body: the parameter list is
        // full of `*` and a naive search would match those.
        let store = body
            .lines()
            .find(|l| l.contains("out[i]"))
            .expect("the residual add still stores through out[i]");
        assert_eq!(
            store.trim(),
            "out[i] = a[i] + b[i];",
            "the residual add is no longer a bare sum",
        );
    }

    #[test]
    fn the_softplus_keeps_ggmls_large_argument_passthrough() {
        // `(x > 20) ? x : logf(1 + expf(x))` — op_softplus in
        // ggml/src/ggml-cuda/unary.cu, ggml/src/ggml-cpu/unary-ops.cpp, and
        // ggml_compute_softplus_f32 in ggml/src/ggml-impl.h, all three
        // identical. Dropping the branch returns inf above x ~ 88.7 rather
        // than saturating, and an inf log-decay poisons the recurrence.
        assert!(LAYER_OPS_SRC.contains("out[i] = (v > 20.0f) ? v : logf(1.0f + expf(v));"));
        // The threshold itself, checked arithmetically rather than trusted:
        // f32::exp overflows well above 20, so the branch is what stands
        // between the GDN alpha gate and a non-finite.
        assert!(100.0f32.exp().is_infinite());
        assert!(!(1.0f32 + 100.0f32.exp()).ln().is_finite());
        // And it is continuous to far below an ulp of 20.
        assert!((1.0f32 + 20.0f32.exp()).ln() - 20.0f32 < 1e-6);
    }

    #[test]
    fn the_gate_length_a_shape_demands_is_the_one_the_model_actually_has() {
        // Qwen3.6's two gates, spelled as literals because this crate does not
        // depend on `xabe-model`: attention q_heads 16 * head_dim 256 = 4096
        // values per token, and the MoE shared expert exactly one.
        const TOKENS: usize = 64;
        assert_eq!(
            GateShape::Elementwise.gate_len(TOKENS, 4096),
            TOKENS * 4096,
            "the attention output gate is elementwise over the q projection",
        );
        assert_eq!(
            GateShape::PerRow.gate_len(TOKENS, 2048),
            TOKENS,
            "the shared expert's gate is one scalar per token",
        );
        // A degenerate row count must not smuggle in a non-empty gate.
        assert_eq!(GateShape::Elementwise.gate_len(0, 4096), 0);
        assert_eq!(GateShape::PerRow.gate_len(0, 2048), 0);
    }

    #[test]
    fn the_elementwise_grid_is_capped_and_does_not_track_n() {
        // Grid-stride over a fixed grid: AGENTS.md rule 5 wants the launch
        // shape independent of a host-side length so it survives graph
        // capture and replay at a different n.
        for n in [1usize, 255, 256, 1 << 20, 1 << 28] {
            let cfg = elementwise_cfg(n);
            assert_eq!(cfg.block_dim, (256, 1, 1), "at n={n}");
            assert!((1..=1024).contains(&cfg.grid_dim.0), "at n={n}");
            assert_eq!(cfg.shared_mem_bytes, 0);
        }
        // Small launches still cover every element without the stride loop,
        // and large ones saturate the cap.
        assert_eq!(elementwise_cfg(1).grid_dim.0, 1);
        assert_eq!(elementwise_cfg(1 << 28).grid_dim.0, 1024);
    }

    #[test]
    fn the_block_width_is_always_a_whole_number_of_warps() {
        // block_reduce_sum shuffles with a full mask, so a partial warp would
        // hang or read garbage. Every width the model uses, plus the awkward
        // ones.
        for width in [1usize, 31, 32, 33, 128, 256, 2048, 4096, 8192] {
            let block = block_for_width(width);
            assert_eq!(block % 32, 0, "width {width} gave block {block}");
            assert!((32..=MAX_BLOCK).contains(&block), "width {width}");
        }
        // The three widths RMSNorm actually runs at in one forward pass.
        assert_eq!(block_for_width(2048), 1024);
        assert_eq!(block_for_width(256), 256);
        assert_eq!(block_for_width(128), 128);
    }

    #[test]
    fn geometry_is_validated_not_assumed() {
        // The checks that run without a device.
        assert!(
            LayerOpsError::UnsupportedWidth { width: 0 }
                .to_string()
                .contains('0')
        );
        assert!(
            LayerOpsError::UnsupportedRotaryShape {
                head_dim: 2048,
                rope_dim: 64,
            }
            .to_string()
            .contains("2048")
        );
        assert!(
            LayerOpsError::UnsupportedConvKernel { conv_kernel: 9 }
                .to_string()
                .contains('9')
        );
        assert!(
            LayerOpsError::ShapeMismatch {
                what: "conv1d state",
                expected: 24576,
                got: 8192,
            }
            .to_string()
            .contains("24576")
        );
    }

    #[test]
    fn a_length_that_matches_the_geometry_is_accepted_and_one_that_does_not_is_not() {
        assert!(check_len("x", 4, 4).is_ok());
        assert!(check_len("x", 4, 5).is_err());
    }
}
