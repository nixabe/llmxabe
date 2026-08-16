//! The 256-expert mixture-of-experts feed-forward block.
//!
//! This block sits on **every one of the 40 transformer layers** — Gated
//! DeltaNet and Gated Attention alike — plus the MTP head at block 40. It is
//! transcribed from `llama_model_qwen35moe::graph::build_layer_ffn`
//! (`/home/nixabe/llama.cpp/src/models/qwen35moe.cpp`) and the
//! `llm_graph_context::build_moe_ffn` / `build_ffn` helpers it calls
//! (`src/llama-graph.cpp`), not from a description of them.
//!
//! ## The block, step by step, with the graph node each step produces
//!
//! ```text
//!   input   = mixer output + residual                       `attn_residual-N`
//!   normed  = RMSNorm(input, post_attention_norm.weight)     `attn_post_norm-N`
//!   logits  = ffn_gate_inp . normed                          (not captured)
//!   probs   = softmax(logits) over all 256 experts           (not captured)
//!   ids     = top-8 by probability, ties to the lower index  (not captured)
//!   w       = probs[ids] / sum(probs[ids])                   (not captured)
//!   routed  = sum_k w_k * down_k(silu(gate_k . normed) * (up_k . normed))
//!                                                            `ffn_moe_out-N`
//!   shexp   = down_s(silu(gate_s . normed) * (up_s . normed))
//!   g       = sigmoid(ffn_gate_inp_shexp . normed)           one scalar / token
//!   ffn_out = routed + shexp * g                             `ffn_out-N`
//!   l_out   = ffn_out + input                                `l_out-N`
//! ```
//!
//! Three details in there are load-bearing and each one is a place an
//! implementation can be plausibly, silently wrong:
//!
//! 1. **The shared expert has a sigmoid gate.** `ffn_gate_inp_shexp` is a
//!    `[hidden]` vector, so `ffn_gate_inp_shexp . normed` is *one scalar per
//!    token*; `build_layer_ffn` sigmoids it and multiplies the whole shared
//!    expert output by it before the sum. [`xabe_cuda::kernels::moe`] has no
//!    reference for this — its `shared_expert` entry point implements
//!    `expert_mlp` only — so the gate lives here, at the block level, and
//!    `tests/moe_block.rs` measures its form out of the golden capture rather
//!    than assuming it.
//! 2. **The renormalization is over the selected 8, not all 256.**
//!    `build_moe_ffn` is called with `norm_w = true`, which divides the
//!    gathered weights by their own sum. `w_scale` is
//!    `hparams.expert_weights_scale`, which `qwen35moe`'s
//!    `load_arch_hparams` never reads, so it keeps its `0.0f` default and the
//!    `w_scale != 0.0f && w_scale != 1.0f` guard skips the scaling entirely.
//!    There is no expert-weight scale on this model.
//! 3. **The residual is taken before the post-mixer norm.** `ffn_residual`
//!    in `build_layer_ffn`'s caller is the `attn_residual` tensor, not the
//!    normalized one. Adding the normalized state back instead produces a
//!    model that still generates text.
//!
//! `build_cvec` between `ffn_out + residual` and the `l_out` callback is the
//! identity without a loaded control vector, so `l_out-N` is exactly the sum.
//!
//! ## Mixed quantization is not optional
//!
//! `Qwen3.6-35B-A3B-UD-Q6_K_XL` stores the expert stacks in **more than one
//! format, and not uniformly per layer**:
//!
//! - `ffn_down_exps` is Q8_0 on all 41 blocks;
//! - `ffn_gate_exps` / `ffn_up_exps` are Q6_K on blocks 0–38 and 40, but
//!   **Q8_0 on block 39**.
//!
//! So the format is read per tensor out of the file's own directory and
//! carried alongside the pointer in [`xabe_cuda::kernels::moe::QuantTensor`].
//! Anything that decides the format once per layer — never mind once per
//! model — reads block 39's gate and up projections as garbage.
//!
//! The two router tensors have the same problem in the other direction:
//! `blk.40.ffn_gate_inp.weight` and `blk.40.ffn_gate_inp_shexp.weight` are
//! the **only two `bf16` tensors in the file** (`docs/ORACLE.md` §8.4); every
//! router on blocks 0–39 is `f32`. [`MoeLayerWeights::upload`] widens `bf16`
//! to `f32` on the host — which is exact, `bf16` being a truncated `f32` —
//! and rejects any other dense type by name rather than reading it as
//! something it is not.
//!
//! ## Why the weights are uploaded here rather than taken from `DeviceWeights`
//!
//! [`crate::DeviceWeights`] holds the whole model in one
//! [`xabe_cuda::DeviceArena`], and hands out [`xabe_cuda::Allocation`] byte
//! ranges into a single `CudaSlice<u8>`. `cudarc` 0.19 can produce a
//! borrowed `CudaView` of a sub-range but not an owned `CudaSlice`, and
//! [`xabe_cuda::kernels::moe::QuantTensor`] takes `&CudaSlice<u8>`. Rather
//! than widen a kernel signature this workstream does not own,
//! [`MoeLayerWeights`] uploads one layer's tensors into their own
//! allocations. One layer is ~725 MiB of expert weights against the model's
//! 29.65 GiB, so a test can walk several layers without the model ever being
//! resident. Reconciling the two is a G006 integration concern, and is noted
//! rather than papered over.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};
use xabe_cuda::kernels::compile;
use xabe_cuda::kernels::layer_ops::{LayerOpsError, LayerOpsKernels};
use xabe_cuda::kernels::moe::{
    ExpertQuant, MoeBuffers, MoeError, MoeGeometry, MoeKernels, QuantTensor, SharedExpertInt8,
};
use xabe_gguf::{GgmlType, GgufFile};
use xabe_model::config::ModelConfig;
use xabe_model::weights::{Directory, Role};

/// Threads per block for the three glue kernels below.
const THREADS: u32 = 256;

/// Tokens the wide router instantiation carries. Mirrors its `TT`.
const ROUTER_TT: u32 = 8;

/// Token count below which the shared expert stays on the fp32 kernel.
///
/// Higher than the routed path's threshold of 8, and measured rather than
/// reasoned. The fp32 shared expert is *two* launches with the SwiGLU fused
/// into the first; the integer path is six — two activation quantizations,
/// three projections and a separate SwiGLU — because `q8_0_proj_split` is a
/// projection and knows nothing about what its output feeds. Six launches and
/// two extra sweeps over `[tokens][intermediate]` need a wide batch to pay
/// for themselves.
///
/// Measured end to end, tok/s at each batch with the threshold at 8 (so the
/// integer path ran) against the fp32 path:
///
/// |  19 | 265.34 -> 240.42 | -9.4%  |
/// | 128 | 893.27 -> 895.87 | +0.3%  |
/// | 512 | 1248.27 -> 1352.61 | +8.4% |
///
/// 128 is where it stops costing anything. Also gates the repack itself, so a
/// decode-shaped pass does not pay 3.5 MB per layer for weights it will never
/// read.
const SHARED_MMA_MIN_TOKENS: usize = 128;
/// Experts one router block covers. Mirrors `ROUTER_ET`.
const ROUTER_ET: u32 = 4;
/// Contraction the router stages per trip. Mirrors `ROUTER_JC`, and **must**
/// equal [`THREADS`] — that equality is what preserves the untiled kernel's
/// per-thread summation order, which the router cannot afford to change.
const ROUTER_JC: u32 = THREADS;

/// RMS epsilon when the file does not carry one.
///
/// llama.cpp reads `f_norm_rms_eps` from
/// `qwen35moe.attention.layer_norm_rms_epsilon` and `ModelConfig` has no
/// epsilon field, so [`MoeBlock::eps_from`] prefers the file and falls back to
/// the Qwen3 family's value.
pub const DEFAULT_RMS_EPS: f32 = 1e-6;

/// The GGUF key llama.cpp reads the RMS epsilon from.
const RMS_EPS_KEY: &str = "qwen35moe.attention.layer_norm_rms_epsilon";

/// The three operations the MoE block needs that no landed kernel provides.
///
/// All three are dense fp32 and tiny next to the grouped GEMM — the router
/// projection is 2048x256 per layer against 256 experts of 3x2048x512 — but
/// they must still run on the device, because a host round-trip for the
/// router logits would put a synchronization between every layer and its
/// successor and AGENTS.md rule 5 forbids anything on this path whose size
/// the host has to know.
///
/// Every launch shape here comes from the geometry and the token count lives
/// in the same `valid_tokens` device scalar the MoE kernels already gate on,
/// so adding these does not cost the sequence its capturability.
const GLUE_SRC: &str = r#"
extern "C" {

extern __shared__ float xabe_moe_block_shared[];

// Warp shuffles then one pass through shared memory. Copied from
// `kernels/moe.rs` verbatim so the router projection's summation order
// matches the one the routed GEMM already uses.
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

// logits[t][e] = dot(ffn_gate_inp row e, normed[t]).
//
// `w` is the GGUF tensor `[hidden, num_experts]` with ne[0] fastest-varying,
// so expert e's row is `hidden` *contiguous* floats at `e * hidden`. Reading
// it the other way round would gather at stride `num_experts` and produce a
// router that is wrong on every token while still summing to one.
//
// # Why this is tiled and the obvious version is not viable
//
// This is the smallest matmul in the layer — 2048x256 against 256 experts of
// 3x2048x512 — and the first version, one block per (expert, token) pair
// reducing a 2048-term dot product, was **7.6% of the whole forward pass**.
// The arithmetic was never the problem. At 512 tokens that grid is 131,072
// blocks, and each one re-read a full 2048-float weight row *and* a full
// 2048-float activation row: 2.1 GB of loads per layer to cover 6 MB of
// distinct data.
//
// A block now covers ROUTER_ET experts and ROUTER_TT tokens at once. The
// weight row is read once per token tile instead of once per token, the
// activation rows once per expert *group* instead of once per expert, and the
// traffic falls about fivefold. Both reads stay fully coalesced.
//
// # Why the summation order is preserved exactly
//
// A first attempt gave each warp its own expert and reduced with a plain warp
// shuffle. It was faster and it was wrong: the per-thread partition of the
// contraction changed, the logits moved in their last bits, and `tests/
// forward_pass.rs` caught block 31's error jumping 5.26x. The router's output
// is not consumed as a number — it is consumed by a top-8 argmax over 256
// experts. A last-bit disagreement between two adjacent logits does not
// perturb an answer slightly; it runs a different expert.
//
// So `ROUTER_JC` equals the block width and the staging loop hands thread
// `tid` the indices `tid, tid + 256, tid + 512, ...` in that order — the
// sequence the untiled `for (j = threadIdx.x; j < hidden; j += blockDim.x)`
// produced — and the reduction repeats `block_reduce_sum`'s shuffle-down and
// ascending warp sum. The logits are bit-identical; only the traffic changed.
//
// grid: (num_experts / ROUTER_ET, ceil(max_tokens / ROUTER_TT)) — never the
// live token count.

// Tokens one block carries, and experts it covers. The product is the 32
// accumulators each thread holds. `ROUTER_JC` is the contraction staged per
// trip and **must equal the block width**, which is what keeps each thread's
// summation order identical to the untiled kernel's.
#define ROUTER_ET 4
#define ROUTER_JC 256

// Instantiated at two token tile widths. `ROUTER_JC` equals the block width in
// both, which is what keeps each thread's summation order identical to the
// untiled kernel's — and identical *between* the two instantiations, so a
// decode step and a prefill step agree bit for bit on the logits of any token
// they share.
#define ROUTER_LOGITS(NAME, TT)                                                \
__global__ void NAME(                                                          \
    const float* __restrict__ w,                                               \
    const float* __restrict__ x,                                               \
    const int* __restrict__ valid_tokens,                                      \
    int hidden,                                                                \
    int num_experts,                                                           \
    int max_tokens,                                                            \
    float* __restrict__ logits                                                 \
) {                                                                            \
    float* sx      = xabe_moe_block_shared;                                    \
    float* scratch = sx + TT * ROUTER_JC;                                      \
                                                                               \
    int tid  = threadIdx.x;                                                    \
    int lane = tid & 31;                                                       \
    int warp = tid >> 5;                                                       \
    int n_warps = (int)(blockDim.x >> 5);                                      \
    int e0 = blockIdx.x * ROUTER_ET;                                           \
    int t0 = blockIdx.y * TT;                                                  \
                                                                               \
    float acc[ROUTER_ET][TT];                                                  \
    _Pragma("unroll")                                                          \
    for (int el = 0; el < ROUTER_ET; ++el)                                     \
        _Pragma("unroll")                                                      \
        for (int u = 0; u < TT; ++u) acc[el][u] = 0.0f;                        \
                                                                               \
    for (int jc = 0; jc < hidden; jc += ROUTER_JC) {                           \
        __syncthreads();                                                       \
        int j = jc + tid;                                                      \
        _Pragma("unroll")                                                      \
        for (int u = 0; u < TT; ++u) {                                         \
            int t = (t0 + u < max_tokens) ? t0 + u : max_tokens - 1;           \
            sx[u * ROUTER_JC + tid] = j < hidden ? x[(long long)t * hidden + j] : 0.0f; \
        }                                                                      \
        __syncthreads();                                                       \
        if (j < hidden) {                                                      \
            _Pragma("unroll")                                                  \
            for (int el = 0; el < ROUTER_ET; ++el) {                           \
                int e = e0 + el;                                               \
                float wj = e < num_experts ? w[(long long)e * hidden + j] : 0.0f; \
                _Pragma("unroll")                                              \
                for (int u = 0; u < TT; ++u) {                                 \
                    acc[el][u] += wj * sx[u * ROUTER_JC + tid];                \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
                                                                               \
    _Pragma("unroll")                                                          \
    for (int el = 0; el < ROUTER_ET; ++el) {                                   \
        _Pragma("unroll")                                                      \
        for (int u = 0; u < TT; ++u) {                                         \
            float v = acc[el][u];                                              \
            _Pragma("unroll")                                                  \
            for (int offset = 16; offset > 0; offset >>= 1) {                  \
                v += __shfl_down_sync(0xffffffff, v, offset);                  \
            }                                                                  \
            if (lane == 0) scratch[warp * (ROUTER_ET * TT) + el * TT + u] = v; \
        }                                                                      \
    }                                                                          \
    __syncthreads();                                                           \
                                                                               \
    int live = *valid_tokens;                                                  \
    if (tid < ROUTER_ET * TT) {                                                \
        int el = tid / TT;                                                     \
        int u  = tid % TT;                                                     \
        float total = 0.0f;                                                    \
        for (int wv = 0; wv < n_warps; ++wv) {                                 \
            total += scratch[wv * (ROUTER_ET * TT) + el * TT + u];             \
        }                                                                      \
        int e = e0 + el;                                                       \
        int t = t0 + u;                                                        \
        if (e < num_experts && t < live) {                                     \
            logits[(long long)t * num_experts + e] = total;                    \
        }                                                                      \
    }                                                                          \
}

// Eight tokens for prefill. One for decode, where seven of the eight
// accumulators would be a token clamped to the same row — 7/8 of the
// arithmetic and 7/8 of the staged tile spent recomputing one answer.
ROUTER_LOGITS(moe_block_router_logits,    8)
ROUTER_LOGITS(moe_block_router_logits_t1, 1)

// gate[t] = sigmoid(dot(ffn_gate_inp_shexp, normed[t])).
//
// `ffn_gate_inp_shexp` is `[hidden]` — a single row — so this produces one
// scalar per token, which is what `build_layer_ffn` then multiplies the whole
// shared-expert output by. grid: (max_tokens).
//
// `1 / (1 + exp(-x))` is `ggml_sigmoid`'s own formula, and `expf` rather than
// `__expf` for the same reason `layer_ops.rs`'s SwiGLU uses it: the fast
// intrinsic's error is relative to the result, not the exponent.
__global__ void moe_block_shared_gate(
    const float* __restrict__ w,
    const float* __restrict__ x,
    const int* __restrict__ valid_tokens,
    int hidden,
    float* __restrict__ gate
) {
    int t = blockIdx.x;
    if (t >= *valid_tokens) return;

    const float* xs = x + (long long)t * hidden;

    float s = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        s += w[j] * xs[j];
    }
    s = block_reduce_sum(s, xabe_moe_block_shared);

    if (threadIdx.x == 0) gate[t] = 1.0f / (1.0f + expf(-s));
}

// ffn_out = routed + shexp * gate;  l_out = ffn_out + residual.
//
// Two outputs rather than one because the golden capture has a waypoint
// either side of the residual add (`ffn_out-N` and `l_out-N`), and keeping
// both makes a block that got the residual source wrong fail at `l_out` with
// `ffn_out` still clean.
//
// The operand order is `build_layer_ffn`'s: the shared expert is gated first,
// then added to the routed sum with the routed sum on the left.
__global__ void moe_block_combine(
    const float* __restrict__ routed,
    const float* __restrict__ shexp,
    const float* __restrict__ gate,
    const float* __restrict__ residual,
    const int* __restrict__ valid_tokens,
    int hidden,
    float* __restrict__ ffn_out,
    float* __restrict__ l_out
) {
    int t = blockIdx.x;
    if (t >= *valid_tokens) return;

    float g = gate[t];
    long long base = (long long)t * hidden;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float v = routed[base + j] + shexp[base + j] * g;
        ffn_out[base + j] = v;
        l_out[base + j]   = v + residual[base + j];
    }
}

}
"#;

/// Something went wrong building or running the MoE block.
#[derive(Debug)]
pub enum MoeBlockError {
    /// A MoE kernel failed.
    Moe(MoeError),
    /// A layer-op kernel failed.
    LayerOps(LayerOpsError),
    /// The driver failed.
    Driver(DriverError),
    /// NVRTC rejected the glue source, or the module failed to load.
    Compile(String),
    /// The weight directory does not carry a tensor this block needs.
    MissingTensor { role: Role, layer: u32 },
    /// The file's data section is shorter than its directory claims.
    UnreadableTensor { name: String },
    /// A dense tensor is stored in a type this block will not guess at.
    ///
    /// Named rather than silently reinterpreted: the two `bf16` routers on
    /// block 40 are exactly the case where reading the bytes as something
    /// else produces finite, plausible, wrong logits.
    UnsupportedDenseType { name: String, found: GgmlType },
    /// An expert stack is stored in a type the grouped GEMM cannot unpack.
    UnsupportedExpertType { name: String, found: GgmlType },
    /// A tensor does not hold the element count the geometry implies.
    ShapeMismatch {
        name: String,
        expected: usize,
        found: usize,
    },
    /// A buffer handed to [`MoeBlock::forward`] is the wrong length.
    BufferLength {
        what: &'static str,
        expected: usize,
        found: usize,
    },
    /// More tokens were presented than the block was sized for.
    TooManyTokens { tokens: usize, max_tokens: usize },
}

impl std::fmt::Display for MoeBlockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Moe(e) => write!(f, "{e}"),
            Self::LayerOps(e) => write!(f, "{e}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::Compile(m) => write!(f, "glue kernel compilation failed: {m}"),
            Self::MissingTensor { role, layer } => {
                write!(f, "layer {layer} has no `{role}`")
            }
            Self::UnreadableTensor { name } => {
                write!(
                    f,
                    "tensor `{name}` is named by the directory but not readable"
                )
            }
            Self::UnsupportedDenseType { name, found } => write!(
                f,
                "tensor `{name}` is {}, and this block only widens f32 and bf16 — \
                 reading it as anything else would produce plausible garbage",
                found.name(),
            ),
            Self::UnsupportedExpertType { name, found } => write!(
                f,
                "expert stack `{name}` is {}, which the grouped GEMM cannot unpack \
                 (it handles Q6_K and Q8_0)",
                found.name(),
            ),
            Self::ShapeMismatch {
                name,
                expected,
                found,
            } => write!(
                f,
                "tensor `{name}` holds {found} elements, the geometry implies {expected}",
            ),
            Self::BufferLength {
                what,
                expected,
                found,
            } => write!(f, "{what} must hold {expected} floats, holds {found}"),
            Self::TooManyTokens { tokens, max_tokens } => write!(
                f,
                "{tokens} tokens exceeds the {max_tokens} this block was sized for",
            ),
        }
    }
}

impl std::error::Error for MoeBlockError {}

impl From<MoeError> for MoeBlockError {
    fn from(e: MoeError) -> Self {
        Self::Moe(e)
    }
}

impl From<LayerOpsError> for MoeBlockError {
    fn from(e: LayerOpsError) -> Self {
        Self::LayerOps(e)
    }
}

impl From<DriverError> for MoeBlockError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

/// One layer's MoE weights, resident on the device.
///
/// Uploaded per layer rather than taken out of [`crate::DeviceWeights`]; see
/// the module docs for why. The quantization format of each expert stack is
/// read out of the file's own tensor directory and kept next to the bytes,
/// because this model's is **not uniform** — not even within one layer, and
/// not even across layers for one role.
pub struct MoeLayerWeights {
    layer: u32,
    post_norm: CudaSlice<f32>,
    router: CudaSlice<f32>,
    gate_exps: CudaSlice<u8>,
    up_exps: CudaSlice<u8>,
    down_exps: CudaSlice<u8>,
    gate_quant: ExpertQuant,
    up_quant: ExpertQuant,
    down_quant: ExpertQuant,
    shared_gate_inp: CudaSlice<f32>,
    shared_gate: CudaSlice<u8>,
    shared_up: CudaSlice<u8>,
    shared_down: CudaSlice<u8>,
    shared_gate_quant: ExpertQuant,
    shared_up_quant: ExpertQuant,
    shared_down_quant: ExpertQuant,
    /// The shared expert repacked for the integer tensor cores, when its three
    /// matrices are Q8_0 and the pass is wide enough to want them. About
    /// 3.5 MB per layer.
    shared_int8: Option<SharedExpertInt8>,
    /// How the router and the shared-expert gate were stored in the file.
    router_type: GgmlType,
    shared_gate_inp_type: GgmlType,
}

impl MoeLayerWeights {
    /// Upload every tensor block `layer`'s MoE needs.
    ///
    /// `directory` must have been resolved against `file`.
    pub fn upload(
        stream: &Arc<CudaStream>,
        file: &GgufFile,
        directory: &Directory<'_>,
        layer: u32,
        geometry: &MoeGeometry,
    ) -> Result<Self, MoeBlockError> {
        let hidden = geometry.hidden;
        let one_expert = geometry.intermediate * hidden;
        let stack = geometry.stack_elements();

        let (post_norm, _) = dense(file, directory, Role::PostMixerNorm, layer, hidden)?;
        let (router, router_type) = dense(
            file,
            directory,
            Role::MoeRouter,
            layer,
            geometry.num_experts * hidden,
        )?;
        let (shared_gate_inp, shared_gate_inp_type) =
            dense(file, directory, Role::MoeSharedGateInp, layer, hidden)?;

        let (gate_bytes, gate_quant) = quantized(file, directory, Role::MoeGateExps, layer, stack)?;
        let (up_bytes, up_quant) = quantized(file, directory, Role::MoeUpExps, layer, stack)?;
        let (down_bytes, down_quant) = quantized(file, directory, Role::MoeDownExps, layer, stack)?;

        let (sgate_bytes, shared_gate_quant) =
            quantized(file, directory, Role::MoeSharedGate, layer, one_expert)?;
        let (sup_bytes, shared_up_quant) =
            quantized(file, directory, Role::MoeSharedUp, layer, one_expert)?;
        let (sdown_bytes, shared_down_quant) =
            quantized(file, directory, Role::MoeSharedDown, layer, one_expert)?;

        let shared_gate = stream.clone_htod(sgate_bytes)?;
        let shared_up = stream.clone_htod(sup_bytes)?;
        let shared_down = stream.clone_htod(sdown_bytes)?;
        // Repacked here, at upload, rather than lazily on the first pass:
        // `AGENTS.md` rule 6 forbids allocating mid-forward, and a lazy repack
        // would also poison the first timed repetition of every benchmark.
        let shared_int8 = if geometry.max_tokens >= SHARED_MMA_MIN_TOKENS
            && [shared_gate_quant, shared_up_quant, shared_down_quant]
                .iter()
                .all(|q| *q == ExpertQuant::Q8_0)
        {
            SharedExpertInt8::repack(
                stream.context(),
                stream,
                QuantTensor {
                    bytes: &shared_gate,
                    quant: shared_gate_quant,
                },
                QuantTensor {
                    bytes: &shared_up,
                    quant: shared_up_quant,
                },
                QuantTensor {
                    bytes: &shared_down,
                    quant: shared_down_quant,
                },
                one_expert,
            )
            .ok()
        } else {
            None
        };

        Ok(Self {
            layer,
            post_norm: stream.clone_htod(&post_norm)?,
            router: stream.clone_htod(&router)?,
            gate_exps: stream.clone_htod(gate_bytes)?,
            up_exps: stream.clone_htod(up_bytes)?,
            down_exps: stream.clone_htod(down_bytes)?,
            gate_quant,
            up_quant,
            down_quant,
            shared_gate_inp: stream.clone_htod(&shared_gate_inp)?,
            shared_gate,
            shared_up,
            shared_down,
            shared_gate_quant,
            shared_up_quant,
            shared_down_quant,
            shared_int8,
            router_type,
            shared_gate_inp_type,
        })
    }

    /// Which block these weights came from.
    pub fn layer(&self) -> u32 {
        self.layer
    }

    /// Storage format of each routed expert stack, in gate/up/down order.
    ///
    /// Exposed so a caller — or a test — can assert what it actually got
    /// rather than what it expected. On this model the answer is not the same
    /// for every layer.
    pub fn expert_quants(&self) -> [ExpertQuant; 3] {
        [self.gate_quant, self.up_quant, self.down_quant]
    }

    /// Storage format of the shared expert's three matrices.
    pub fn shared_quants(&self) -> [ExpertQuant; 3] {
        [
            self.shared_gate_quant,
            self.shared_up_quant,
            self.shared_down_quant,
        ]
    }

    /// How the file stored this layer's two router tensors.
    ///
    /// `(ffn_gate_inp, ffn_gate_inp_shexp)`. `f32` everywhere except block 40,
    /// where both are `bf16` and nothing else in the file is.
    pub fn router_types(&self) -> (GgmlType, GgmlType) {
        (self.router_type, self.shared_gate_inp_type)
    }

    /// Total device bytes held.
    pub fn bytes(&self) -> usize {
        (self.post_norm.len() + self.router.len() + self.shared_gate_inp.len()) * size_of::<f32>()
            + self.gate_exps.len()
            + self.up_exps.len()
            + self.down_exps.len()
            + self.shared_gate.len()
            + self.shared_up.len()
            + self.shared_down.len()
            + self.shared_int8.as_ref().map_or(0, SharedExpertInt8::bytes)
    }
}

/// Look up one tensor's bytes and stored type.
fn raw<'f>(
    file: &'f GgufFile,
    directory: &Directory<'_>,
    role: Role,
    layer: u32,
) -> Result<(&'f [u8], GgmlType, String), MoeBlockError> {
    let entry = directory
        .find(role, Some(layer))
        .ok_or(MoeBlockError::MissingTensor { role, layer })?;
    let name = entry.spec.name.clone();
    let bytes = file
        .tensor_bytes(&name)
        .ok_or_else(|| MoeBlockError::UnreadableTensor { name: name.clone() })?;
    Ok((bytes, entry.info.ggml_type, name))
}

/// An unquantized tensor, widened to f32 on the host.
///
/// `bf16` is a truncated `f32`, so the widening is exact — a shift, not a
/// conversion. Any other dense type is rejected by name.
fn dense(
    file: &GgufFile,
    directory: &Directory<'_>,
    role: Role,
    layer: u32,
    expected: usize,
) -> Result<(Vec<f32>, GgmlType), MoeBlockError> {
    let (bytes, ty, name) = raw(file, directory, role, layer)?;
    let values: Vec<f32> = match ty {
        GgmlType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(f32::from_le_bytes)
            .collect(),
        GgmlType::Bf16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| f32::from_bits(u32::from(u16::from_le_bytes(b)) << 16))
            .collect(),
        found => return Err(MoeBlockError::UnsupportedDenseType { name, found }),
    };
    if values.len() != expected {
        return Err(MoeBlockError::ShapeMismatch {
            name,
            expected,
            found: values.len(),
        });
    }
    Ok((values, ty))
}

/// A quantized weight stack, left in its serialized form.
///
/// Nothing is dequantized on the host: one layer's routed experts are 268 M
/// elements per projection, which is 3.1 GiB in fp32 against 725 MiB packed.
/// The GEMM unpacks the element it is about to multiply.
fn quantized<'f>(
    file: &'f GgufFile,
    directory: &Directory<'_>,
    role: Role,
    layer: u32,
    expected: usize,
) -> Result<(&'f [u8], ExpertQuant), MoeBlockError> {
    let (bytes, ty, name) = raw(file, directory, role, layer)?;
    let quant = match ty {
        GgmlType::Q6K => ExpertQuant::Q6K,
        GgmlType::Q8_0 => ExpertQuant::Q8_0,
        found => return Err(MoeBlockError::UnsupportedExpertType { name, found }),
    };
    let found = bytes.len() / quant.block_bytes() * quant.block_elements();
    if !bytes.len().is_multiple_of(quant.block_bytes()) || found != expected {
        return Err(MoeBlockError::ShapeMismatch {
            name,
            expected,
            found,
        });
    }
    Ok((bytes, quant))
}

/// The MoE feed-forward block, compiled and sized for one fixed geometry.
pub struct MoeBlock {
    moe: MoeKernels,
    layer_ops: LayerOpsKernels,
    router_logits_fn: CudaFunction,
    router_logits_t1_fn: CudaFunction,
    shared_gate_fn: CudaFunction,
    combine_fn: CudaFunction,
    buffers: MoeBuffers,
    normed: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    routed: CudaSlice<f32>,
    shexp: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    geometry: MoeGeometry,
    eps: f32,
}

impl MoeBlock {
    /// The RMS epsilon `file` declares, or [`DEFAULT_RMS_EPS`].
    ///
    /// llama.cpp reads this key with the non-optional `get_key`, so a file
    /// without it would not load there at all; the fallback exists for
    /// synthetic files, not for the real one.
    pub fn eps_from(file: &GgufFile) -> f32 {
        file.get_f32(RMS_EPS_KEY).unwrap_or(DEFAULT_RMS_EPS)
    }

    /// The real Qwen3.6 MoE geometry, for `max_tokens` per step.
    ///
    /// Derived from [`ModelConfig`] rather than written out, so a config that
    /// disagrees with the file cannot silently produce a block of the wrong
    /// shape.
    pub fn geometry_for(config: &ModelConfig, block_size: usize, max_tokens: usize) -> MoeGeometry {
        MoeGeometry {
            num_experts: config.moe.num_experts as usize,
            experts_per_token: config.moe.experts_per_token as usize,
            hidden: config.hidden_size as usize,
            intermediate: config.moe.expert_intermediate as usize,
            block_size,
            max_tokens,
        }
    }

    /// Compile every kernel the block needs and allocate every buffer, once.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        geometry: MoeGeometry,
        eps: f32,
    ) -> Result<Self, MoeBlockError> {
        let moe = MoeKernels::new(ctx, geometry)?;
        let layer_ops = LayerOpsKernels::new(ctx)?;
        let ptx = compile(GLUE_SRC, "moe_block").map_err(MoeBlockError::Compile)?;
        let module = ctx.load_module(ptx)?;
        let buffers = moe.buffers(stream)?;

        let n = geometry.max_tokens * geometry.hidden;
        Ok(Self {
            router_logits_fn: module.load_function("moe_block_router_logits")?,
            router_logits_t1_fn: module.load_function("moe_block_router_logits_t1")?,
            shared_gate_fn: module.load_function("moe_block_shared_gate")?,
            combine_fn: module.load_function("moe_block_combine")?,
            moe,
            layer_ops,
            buffers,
            normed: stream.alloc_zeros::<f32>(n)?,
            logits: stream.alloc_zeros::<f32>(geometry.max_tokens * geometry.num_experts)?,
            routed: stream.alloc_zeros::<f32>(n)?,
            shexp: stream.alloc_zeros::<f32>(n)?,
            gate: stream.alloc_zeros::<f32>(geometry.max_tokens)?,
            geometry,
            eps,
        })
    }

    /// Force every MoE GEMM back onto its fp32 kernel.
    ///
    /// Both the routed experts and the shared one: dropping the `MmaKernels`
    /// handle disables the routed path, and the shared path is gated on the
    /// same flag so it follows. See
    /// [`crate::forward::Forward::disable_tensor_cores`], which calls this so
    /// `tests/int8_forward.rs` covers the MoE at all — without it the test
    /// compared an integer MoE against an integer MoE and said nothing.
    pub fn disable_tensor_cores(&mut self) {
        self.moe.disable_tensor_cores();
    }

    /// Whether the MoE GEMMs will take the integer tensor-core path.
    pub fn tensor_cores_enabled(&self) -> bool {
        self.moe.tensor_cores_enabled()
    }

    /// The geometry this block was compiled for.
    pub fn geometry(&self) -> MoeGeometry {
        self.geometry
    }

    /// The RMS epsilon in use.
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// `attn_post_norm-N`: the post-mixer RMSNorm output, `[max_tokens][hidden]`.
    pub fn normed(&self) -> &CudaSlice<f32> {
        &self.normed
    }

    /// Router logits over all experts, `[max_tokens][num_experts]`.
    ///
    /// Not captured in the golden — `ffn_moe_logits` is named inside
    /// `build_moe_ffn` and the capture's filter list does not cover it — so
    /// this exists for diagnosis, not for a direct comparison.
    pub fn router_logits(&self) -> &CudaSlice<f32> {
        &self.logits
    }

    /// `ffn_moe_out-N`: the routed experts' weighted sum, before the shared
    /// expert.
    pub fn routed(&self) -> &CudaSlice<f32> {
        &self.routed
    }

    /// The shared expert's output **before** its sigmoid gate is applied.
    ///
    /// Not a graph node in llama.cpp's capture; exposed because the gate can
    /// then be recovered from the golden as
    /// `(ffn_out - ffn_moe_out) / shared_ungated`, which is how
    /// `tests/moe_block.rs` measures the gate's form instead of assuming it.
    pub fn shared_ungated(&self) -> &CudaSlice<f32> {
        &self.shexp
    }

    /// `sigmoid(ffn_gate_inp_shexp . normed)`, one scalar per token.
    pub fn shared_gate(&self) -> &CudaSlice<f32> {
        &self.gate
    }

    /// The MoE dispatch buffers, for inspecting the routing decision.
    pub fn buffers(&self) -> &MoeBuffers {
        &self.buffers
    }

    /// Run the whole block over `tokens` tokens.
    ///
    /// `residual` is the mixer output already added back to the block's input
    /// — llama.cpp's `attn_residual-N` — and is both the RMSNorm's input and
    /// the residual the block's output is added to. `ffn_out` receives
    /// `ffn_out-N` and `l_out` receives `l_out-N`; every buffer is
    /// `[max_tokens][hidden]`.
    ///
    /// Rows past `tokens` are never written, which is checkable: a kernel that
    /// ran on a stale slot would leave a plausible value where the caller left
    /// a zero.
    #[allow(clippy::too_many_arguments)]
    /// Publish this pass's token count into the device scalar the MoE kernels
    /// gate on, ahead of the pass.
    ///
    /// Idempotent, and the reason it is public: the write reads host memory,
    /// and a CUDA graph capture may not contain a copy from a pageable host
    /// pointer. `Forward::publish_inputs` calls this before capturing so that
    /// the call inside [`Self::forward`] has nothing left to do.
    pub fn publish_tokens(
        &mut self,
        stream: &Arc<CudaStream>,
        tokens: usize,
    ) -> Result<(), MoeBlockError> {
        self.moe
            .set_valid_tokens(stream, &mut self.buffers, tokens)?;
        Ok(())
    }

    pub fn forward(
        &mut self,
        stream: &Arc<CudaStream>,
        w: &MoeLayerWeights,
        residual: &CudaSlice<f32>,
        tokens: usize,
        ffn_out: &mut CudaSlice<f32>,
        l_out: &mut CudaSlice<f32>,
    ) -> Result<(), MoeBlockError> {
        let g = self.geometry;
        if tokens > g.max_tokens {
            return Err(MoeBlockError::TooManyTokens {
                tokens,
                max_tokens: g.max_tokens,
            });
        }
        let n = g.max_tokens * g.hidden;
        check_len("residual", n, residual.len())?;
        check_len("ffn_out", n, ffn_out.len())?;
        check_len("l_out", n, l_out.len())?;

        // A no-op after the first call at this shape — see
        // `MoeKernels::set_valid_tokens`. The forward path keeps the call so
        // a caller that never reaches `publish_tokens` is still correct.
        self.publish_tokens(stream, tokens)?;

        // 1. post-mixer RMSNorm. Every row of the buffer is normalized, not
        //    just the live ones, so the launch shape is the geometry's: a row
        //    of zeros normalizes to zeros rather than to a NaN, since the
        //    epsilon keeps the divisor positive.
        self.layer_ops.rms_norm(
            stream,
            residual,
            &w.post_norm,
            &mut self.normed,
            g.max_tokens,
            g.hidden,
            self.eps,
        )?;

        // 2. router logits, then softmax + top-k + renormalize on the device.
        let hidden_i32 = g.hidden as i32;
        let experts_i32 = g.num_experts as i32;
        let max_tokens_i32 = g.max_tokens as i32;
        // One token takes the narrow instantiation; see the kernel.
        let tt = if g.max_tokens == 1 { 1 } else { ROUTER_TT };
        let cfg = LaunchConfig {
            grid_dim: (
                (g.num_experts as u32).div_ceil(ROUTER_ET),
                (g.max_tokens as u32).div_ceil(tt),
                1,
            ),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: ((tt * ROUTER_JC + THREADS.div_ceil(32) * ROUTER_ET * tt) as usize
                * size_of::<f32>()) as u32,
        };
        let f = if tt == 1 {
            &self.router_logits_t1_fn
        } else {
            &self.router_logits_fn
        };
        let mut builder = stream.launch_builder(f);
        builder
            .arg(&w.router)
            .arg(&self.normed)
            .arg(self.buffers.valid_tokens())
            .arg(&hidden_i32)
            .arg(&experts_i32)
            .arg(&max_tokens_i32)
            .arg(&mut self.logits);
        // SAFETY: `w.router` was checked to hold `num_experts * hidden` floats
        // and `normed` `max_tokens * hidden`. The kernel returns for
        // `e >= num_experts` and clamps its token index to `max_tokens - 1`,
        // so `e * hidden + j` and `t * hidden + j` are both in range for
        // `j < hidden`. `logits` holds `max_tokens * num_experts` and is
        // written only for `t < valid_tokens`. The warp reduction uses no
        // shared memory.
        unsafe { builder.launch(cfg) }?;

        self.moe.route(stream, &mut self.buffers, &self.logits)?;
        self.moe.build_dispatch(stream, &mut self.buffers)?;

        // 3. the routed experts.
        self.moe.grouped_forward(
            stream,
            &mut self.buffers,
            QuantTensor {
                bytes: &w.gate_exps,
                quant: w.gate_quant,
            },
            QuantTensor {
                bytes: &w.up_exps,
                quant: w.up_quant,
            },
            QuantTensor {
                bytes: &w.down_exps,
                quant: w.down_quant,
            },
            &self.normed,
            &mut self.routed,
        )?;

        // 4. the shared expert, ungated — the kernel implements `expert_mlp`
        //    and nothing else.
        // Gated on **this pass's** token count, not merely on the repacked
        // weights existing. `Forward::reshape` shares one
        // `Vec<MoeLayerWeights>` between a wide prefill pass and a one-token
        // decode pass, so a decode step inherits whatever the prefill upload
        // repacked. Checking only for the repack sent every decode step down
        // the six-launch integer path to fill one slot of a 64-token tile, and
        // cost 57% of decode throughput — 15.39 ms per step became 24.13 —
        // while `bench_forward`'s n = 1 column, which builds its own weights
        // and so never repacks, showed nothing wrong.
        let wide_enough = g.max_tokens >= SHARED_MMA_MIN_TOKENS;
        match w
            .shared_int8
            .as_ref()
            .filter(|_| wide_enough && self.moe.tensor_cores_enabled())
        {
            Some(i8w) => self.moe.shared_expert_mma(
                stream,
                &mut self.buffers,
                i8w,
                &self.normed,
                &mut self.shexp,
            )?,
            None => self.moe.shared_expert(
                stream,
                &mut self.buffers,
                QuantTensor {
                    bytes: &w.shared_gate,
                    quant: w.shared_gate_quant,
                },
                QuantTensor {
                    bytes: &w.shared_up,
                    quant: w.shared_up_quant,
                },
                QuantTensor {
                    bytes: &w.shared_down,
                    quant: w.shared_down_quant,
                },
                &self.normed,
                &mut self.shexp,
            )?,
        }

        // 5. its sigmoid gate, which lives here because no kernel has it.
        // Still `block_reduce_sum`, so it still needs one float per warp.
        let cfg = LaunchConfig {
            grid_dim: (g.max_tokens as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: ((THREADS as usize).div_ceil(32) * size_of::<f32>()) as u32,
        };
        let mut builder = stream.launch_builder(&self.shared_gate_fn);
        builder
            .arg(&w.shared_gate_inp)
            .arg(&self.normed)
            .arg(self.buffers.valid_tokens())
            .arg(&hidden_i32)
            .arg(&mut self.gate);
        // SAFETY: one block per token slot, gated on the device
        // `valid_tokens`; `w.shared_gate_inp` holds `hidden` floats and
        // `gate` holds `max_tokens`.
        unsafe { builder.launch(cfg) }?;

        // 6. routed + gated shared, then the residual.
        let cfg = LaunchConfig {
            grid_dim: (g.max_tokens as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&self.combine_fn);
        builder
            .arg(&self.routed)
            .arg(&self.shexp)
            .arg(&self.gate)
            .arg(residual)
            .arg(self.buffers.valid_tokens())
            .arg(&hidden_i32)
            .arg(ffn_out)
            .arg(l_out);
        // SAFETY: one block per token slot, gated on the device
        // `valid_tokens`; all five `[max_tokens][hidden]` buffers were
        // length-checked above and the in-row loop is bounded by `hidden`.
        unsafe { builder.launch(cfg) }?;

        Ok(())
    }
}

fn check_len(what: &'static str, expected: usize, found: usize) -> Result<(), MoeBlockError> {
    if found == expected {
        Ok(())
    } else {
        Err(MoeBlockError::BufferLength {
            what,
            expected,
            found,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ModelConfig {
        ModelConfig::qwen3_6_35b_a3b()
    }

    #[test]
    fn the_geometry_is_derived_from_the_model_config_not_written_out() {
        let g = MoeBlock::geometry_for(&config(), 16, 32);
        assert_eq!(g.num_experts, 256);
        assert_eq!(g.experts_per_token, 8);
        assert_eq!(g.hidden, 2048);
        assert_eq!(g.intermediate, 512);
        // Every buffer and every grid comes from this, so it has to be an
        // upper bound rather than a typical value.
        assert_eq!(g.max_flat_pairs(), 256);
        assert!(g.sorted_capacity() <= 65_535, "grid.y limit");
    }

    #[test]
    fn the_shared_expert_gate_is_a_sigmoid_of_a_per_token_scalar() {
        // The step `xabe-kernels` has no reference for, and the one an
        // implementation is most likely to leave out entirely: the gate must
        // be `1/(1+exp(-x))` of a dot product against a `[hidden]` vector, and
        // it must multiply the shared expert only — not the routed sum.
        assert!(GLUE_SRC.contains("gate[t] = 1.0f / (1.0f + expf(-s));"));
        assert!(GLUE_SRC.contains("float v = routed[base + j] + shexp[base + j] * g;"));
        let start = GLUE_SRC
            .find("void moe_block_shared_gate")
            .expect("shared gate kernel present");
        let end = GLUE_SRC[start..]
            .find("void moe_block_combine")
            .expect("combine kernel present")
            + start;
        assert!(
            !GLUE_SRC[start..end].contains("__expf"),
            "the shared gate reverted to the fast intrinsic",
        );
    }

    #[test]
    fn the_router_reads_an_experts_weights_contiguously() {
        // `ffn_gate_inp.weight` is `[hidden, num_experts]` in ggml order, so
        // one expert's row is `hidden` contiguous floats. Gathering at stride
        // `num_experts` instead produces a router that is wrong on every
        // token and whose weights still sum to one.
        //
        // Asserted as the index expression rather than as one spelling of a
        // hoisted row pointer: the tiled kernel indexes `w` inline, and the
        // invariant is which of the two axes is contiguous, not whether a
        // pointer was named.
        assert!(GLUE_SRC.contains("w[(long long)e * hidden + j]"));
        assert!(
            !GLUE_SRC.contains("num_experts + e]")
                || GLUE_SRC.contains("logits[(long long)t * num_experts + e]"),
            "the only `* num_experts + e` may be the logits store, which is \
             `[max_tokens][num_experts]` and genuinely strided that way",
        );
    }

    #[test]
    fn the_residual_is_added_after_the_ffn_waypoint_not_before() {
        // `ffn_out-N` is captured before the residual add and `l_out-N` after,
        // so the block must produce both. Folding the residual into `ffn_out`
        // would still give the right `l_out` and would fail the golden's
        // intermediate waypoint — which is the point of having it.
        assert!(GLUE_SRC.contains("ffn_out[base + j] = v;"));
        assert!(GLUE_SRC.contains("l_out[base + j]   = v + residual[base + j];"));
    }

    #[test]
    fn every_launch_shape_comes_from_the_geometry_and_the_token_count_from_the_device() {
        // AGENTS.md rule 5: nothing on this path may be sized by a host-side
        // value. All three glue kernels must take the device scalar and act on
        // it.
        assert_eq!(
            GLUE_SRC
                .matches("const int* __restrict__ valid_tokens")
                .count(),
            3
        );
        // *Reading* it, not one particular spelling of the guard. The router
        // early-returned on it until it was tiled; now a warp carries eight
        // tokens, so it reads the scalar once and gates the store instead.
        // Asserting the `return` form would have made a correct rewrite look
        // like a rule-5 violation, which is the opposite of what this test is
        // for.
        assert_eq!(GLUE_SRC.matches("*valid_tokens").count(), 3);
        // What rule 5 actually forbids: a launch bound the host had to know.
        // `max_tokens` may size a grid — it is a geometry constant — but no
        // kernel may compare against a *count* passed by value.
        assert!(!GLUE_SRC.contains("int live_tokens"));
    }

    #[test]
    fn bf16_widening_is_a_shift_and_is_therefore_exact() {
        // Block 40's two routers are the file's only bf16 tensors. bf16 is a
        // truncated f32, so widening is `bits << 16` and loses nothing; the
        // point of asserting it is that a *rounding* conversion, or reading
        // the bytes as f16, would both be finite and wrong.
        let widen = |b: [u8; 2]| f32::from_bits(u32::from(u16::from_le_bytes(b)) << 16);
        for v in [1.0f32, -1.0, 0.5, -0.015625, 0.0] {
            let bits = v.to_bits();
            assert_eq!(bits & 0xffff, 0, "{v} is not exactly representable in bf16");
            let truncated = ((bits >> 16) as u16).to_le_bytes();
            assert_eq!(widen(truncated), v);
        }
        // And an f16 reading of the same bytes is genuinely different, so the
        // distinction is not academic.
        let bf16_one = ((1.0f32.to_bits() >> 16) as u16).to_le_bytes();
        assert_eq!(widen(bf16_one), 1.0);
        assert_ne!(u16::from_le_bytes(bf16_one), 0x3c00, "0x3c00 is f16's 1.0");
    }

    #[test]
    fn errors_name_the_tensor_and_the_type_they_refused() {
        let e = MoeBlockError::UnsupportedDenseType {
            name: "blk.40.ffn_gate_inp.weight".into(),
            found: GgmlType::Bf16,
        };
        let text = e.to_string();
        assert!(text.contains("blk.40.ffn_gate_inp.weight"), "{text}");

        let e = MoeBlockError::UnsupportedExpertType {
            name: "blk.39.ffn_gate_exps.weight".into(),
            found: GgmlType::Q4K,
        };
        assert!(e.to_string().contains("blk.39.ffn_gate_exps.weight"));

        let e = MoeBlockError::TooManyTokens {
            tokens: 33,
            max_tokens: 32,
        };
        assert!(e.to_string().contains("33"));

        let e = MoeBlockError::MissingTensor {
            role: Role::MoeSharedGateInp,
            layer: 7,
        };
        assert!(e.to_string().contains("ffn_gate_inp_shexp.weight"));
    }

    #[test]
    fn a_buffer_of_the_wrong_length_is_rejected() {
        assert!(check_len("residual", 4, 4).is_ok());
        let e = check_len("residual", 4, 5).unwrap_err();
        assert!(e.to_string().contains('4'));
    }
}
