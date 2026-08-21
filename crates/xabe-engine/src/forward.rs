//! The full forward pass: embedding, 40 blocks, final norm, LM head.
//!
//! This is the integration of everything in [`crate::block`]. Each block shape
//! was gated against llama.cpp's captured intermediates on its own
//! (`docs/ORACLE.md`, `tests/{gdn,attention,moe}_block.rs`); this file chains
//! them and `tests/forward_pass.rs` gates the chain — every `l_out-N` for
//! `N` in `0..=39`, then `h_nextn`, then `result_output`, then the argmax.
//!
//! ```text
//!   token ids -> get_rows(token_embd, ids)             model.input_embed
//!   for N in 0..40:
//!       mixer   = GDN if N % 4 != 3 else Gated Attention
//!       resid   = mixer(RMSNorm(h)) + h                attn_residual-N
//!       h       = MoE(resid) + resid                   l_out-N
//!   normed = RMSNorm(h, output_norm.weight)            h_nextn
//!   logits = output.weight . normed[last]              result_output
//! ```
//!
//! # The obstacle this file exists to have solved
//!
//! [`crate::DeviceWeights`] holds the model in **one** 29.6 GiB arena and
//! hands out byte ranges into a single `CudaSlice<u8>`. Every kernel entry
//! point takes `&CudaSlice<T>` and validates its whole element count, so a
//! range could not be passed to one. Both landed blocks worked around it by
//! copying their layer's weights into fresh allocations at construction, and
//! at 725 MiB per MoE layer that is 28.3 GiB of duplicate over 40 layers —
//! the model, twice, on a 48 GiB card.
//!
//! Three things keep fixed-token shapes from duplicating model weights:
//!
//! - **Gated DeltaNet, the embedding, the final norm and the LM head take
//!   [`crate::weights::ResidentTensor`] aliases.** Zero copy: the alias is the
//!   arena's own pointer arithmetic, wrapped back into a `CudaSlice` and
//!   sealed against `Drop`. 30 layers of GDN projections — 1.07 GiB — plus
//!   1.06 GiB of embedding and LM head are read in place.
//! - **The MoE owns its expert tensors, and the arena does not.**
//!   [`crate::block::moe::MoeLayerWeights`] uploads them into allocations of
//!   its own, so the nine roles it reads are filtered out of the arena
//!   entirely ([`crate::DeviceWeights::load_where`]). There is still exactly
//!   one copy of every tensor on the device — it is simply in 40 sets of
//!   allocations rather than in the slab. All 40 layers stay resident and
//!   nothing is re-uploaded per pass.
//! - **Gated Attention weights and their split-layout int8 repacks are shared
//!   across shapes.** The first shape still holds one copied set outside the
//!   arena, measured by [`ForwardReport::attention_duplicate_bytes`]. A
//!   reshape reuses that set and reports zero additional duplicate bytes;
//!   kernels and scratch remain shape-local.
//!
//! # Prefill and decode are the same code
//!
//! A pass is built for a fixed token count, so a 19-token prefill and a
//! one-token decode step are two `Forward` objects — but they run the same
//! forty blocks over one [`SequenceState`], which owns the KV caches and the
//! recurrent states. Everything that makes a step a *continuation* lives in
//! that state and nothing lives here.
//!
//! Build the second shape with [`Forward::reshape`], never a second
//! [`Forward::new`]: the 28.3 GiB of MoE weights are shared by reference
//! count, and two copies do not fit on a 48 GiB card.
//!
//! `tests/decode.rs` gates the equivalence directly — prefilling 19 tokens,
//! prefilling 18 and decoding 1, and prefilling 12 and decoding 7 all select
//! the same token with cosine 1.000000000 between their logits.
//!
//! A pass over a fresh or [`SequenceState::reset`] state is a cold prefill:
//! position 0, zeroed recurrent state, which is exactly what the oracle
//! captured (`state_predelta-N` is all zeros). That is what
//! `tests/forward_pass.rs` requires, and it is why the reset is the caller's
//! choice rather than something this file does unconditionally.
//!
//! # Where this pass disagrees with the capture
//!
//! llama.cpp quantizes its *activations* to `q8_1` and dots in int8, which is
//! 1,000-10,000x less accurate than accumulating in fp32 (`docs/ORACLE.md`
//! section 8 item 0). This pass takes the integer tensor cores only on wide
//! shapes — 64 tokens or more, where each block quantizes its own activations
//! — and dequantizes to fp32 below that. Neither path reproduces llama.cpp's
//! rounding, so a residual disagreement is expected on both, it accumulates
//! over 40 blocks, and `tests/forward_pass.rs` reports it as a curve rather
//! than a number.

use std::mem::ManuallyDrop;
use std::sync::Arc;

use smallvec::SmallVec;

use cudarc::driver::sys;
use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaGraph, CudaSlice, CudaStream, DriverError,
    LaunchConfig, PushKernelArg,
};

use xabe_cuda::kernels::attention::AttentionError;
use xabe_cuda::kernels::compile;
use xabe_cuda::kernels::layer_ops::{LayerOpsError, LayerOpsKernels};
use xabe_cuda::kernels::lm_head::{ARGMAX_BLOCKS, LmHeadError, LmHeadGeometry, LmHeadKernels};
use xabe_cuda::kernels::moe::{ExpertQuant, QuantTensor};
use xabe_gguf::{GgmlType, GgufFile};
use xabe_model::config::{LayerKind, ModelConfig};
use xabe_model::weights::{Directory, Role};

use crate::block::attention::{
    AttentionBlockError, AttentionKernelSet, AttentionLayerWeights, AttnScratch,
    GatedAttentionBlock, KvCache,
};
use crate::block::gdn::{
    GdnBlock, GdnBlockError, GdnGeometry, GdnLayerInt8, GdnLayerWeights, GdnState,
};
use crate::block::gdn_verify::{
    GdnSnapshotRing, GdnVerifyError, GdnVerifyScratch, run_layer_with_snapshots,
    run_layer_with_snapshots_batch,
};
use crate::block::moe::{MoeBlock, MoeBlockError, MoeLayerWeights};
use crate::state::{SequenceState, StateError};
use crate::weights::DeviceWeights;

/// Elements per Q8_0 block, and its serialized size.
///
/// Spelled here as well as in the kernel source below because NVRTC compiles
/// from a string with no access to Rust constants;
/// `the_embedding_row_stride_matches_the_kernel` asserts the two agree with
/// the upstream layout.
const QK8_0: usize = 32;
const BLOCK_Q8_0_BYTES: usize = 34;

/// Threads per block for the embedding gather.
const EMBED_THREADS: u32 = 256;

/// Grouped-GEMM tile width the MoE dispatch tables pad to, for a pass of
/// `tokens` positions.
///
/// `xabe_cuda::kernels::moe::MoeKernels` compiles two widths of
/// `moe_expert_ffn_mma`/`moe_expert_down_mma` -- `MMA_M_NARROW` (32) at
/// three blocks/SM, `MMA_M` (64) at two -- and picks between them by
/// `block_size` at construction. A wider dispatch tile means more routed
/// tokens share one staged weight tile before it is re-fetched, which is
/// where these kernels' traffic actually goes, but it costs occupancy to
/// fit the wider tile in shared memory. `bench_moe_mma` measured where
/// that trade crosses over, two interleaved rounds each: at 896 tokens the
/// narrow tile is ~22% faster, at 1,024 and 1,152 the two are within noise
/// of each other (narrow a hair ahead on both), and only at 1,280 does the
/// wide tile pull clearly ahead (~6%). 1,152 is the boundary this returns
/// -- the top of the measured wash rather than the middle of it, so a
/// token count landing in the noisy band never picks the tile that
/// measured behind. See "Two compiled widths instead of one" in
/// docs/BENCHMARKS.md for the numbers.
pub fn moe_block_size(tokens: usize) -> usize {
    if tokens <= 1152 { 32 } else { 64 }
}

/// The GGUF keys that are not in [`ModelConfig`] and must not be guessed.
pub(crate) const RMS_EPS_KEY: &str = "qwen35moe.attention.layer_norm_rms_epsilon";
pub(crate) const ROPE_FREQ_BASE_KEY: &str = "qwen35moe.rope.freq_base";

const EMBED_SRC: &str = r#"
extern "C" {

// Reinterpret two little-endian bytes as an IEEE half and widen to float.
//
// Copied verbatim from `xabe_cuda::kernels::dequant`: NVRTC compiles from a
// string with no include path, so <cuda_fp16.h> is unreachable.
__device__ __forceinline__ float load_half_le(const unsigned char* p) {
    unsigned short bits = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(bits));
    return f;
}

// out[t][j] = dequantize(token_embd)[ids[t]][j] -- ggml's `get_rows` over a
// Q8_0 table, which is what `model.input_embed` is.
//
// `token_embd.weight` is [hidden, vocab] in ggml order, so token `v`'s row is
// `hidden` *contiguous* elements at `v * (hidden/32) * 34` bytes. Reading it
// transposed gathers at stride `vocab` and produces a different model that
// still generates text; `docs/ORACLE.md` section 6.3 proves the layout
// bit-exactly against this same table.
//
// The arithmetic is `d * q` and nothing else, matching
// `dequantize_row_q8_0`, so the result is bit-identical to the capture rather
// than merely close.
//
// grid: (max_tokens). block: EMBED_THREADS, striding the row.
__global__ void fwd_embed_q8_0(
    const unsigned char* __restrict__ table,
    const int* __restrict__ ids,
    float* __restrict__ out,
    int hidden,
    int n_tokens
) {
    int t = blockIdx.x;
    if (t >= n_tokens) return;

    long long blocks = hidden / 32;
    const unsigned char* row = table + (long long)ids[t] * blocks * 34;
    float* dst = out + (long long)t * hidden;

    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        const unsigned char* blk = row + (long long)(j / 32) * 34;
        float d = load_half_le(blk);
        signed char q = (signed char)blk[2 + (j % 32)];
        dst[j] = d * (float)q;
    }
}

}
"#;

/// Something went wrong building or running the forward pass.
#[derive(Debug)]
pub enum ForwardError {
    /// NVRTC rejected the embedding source, or the module failed to load.
    Compile(String),
    /// The driver failed.
    Driver(DriverError),
    /// A Gated DeltaNet block failed.
    Gdn(GdnBlockError),
    /// A snapshotted Gated DeltaNet layer failed.
    GdnVerify(GdnVerifyError),
    /// A Gated Attention block failed.
    Attention(AttentionBlockError),
    /// The MoE block failed.
    Moe(MoeBlockError),
    /// A layer op failed.
    LayerOps(LayerOpsError),
    /// The LM head failed.
    LmHead(LmHeadError),
    /// The attention mixer failed.
    Mixer(AttentionError),
    /// A tensor the pass needs is not resident.
    ///
    /// Not recoverable by substituting anything: a missing projection would
    /// have to be replaced by *some* matrix, and every choice produces finite,
    /// plausible, wrong logits.
    MissingWeight { role: Role, layer: Option<u32> },
    /// A resident tensor is stored in a format this pass does not unpack.
    WrongQuant {
        role: Role,
        layer: Option<u32>,
        found: GgmlType,
        expected: GgmlType,
    },
    /// The embedding table or the LM head is not the size the vocabulary
    /// implies.
    ///
    /// Checked because the embedding gather indexes by a token id and has no
    /// other bound available: a short table reads past its own end.
    WrongTableSize {
        role: Role,
        expected: usize,
        got: usize,
    },
    /// The file does not declare a hyperparameter that must not be guessed.
    MissingMetadata { key: &'static str },
    /// The batch is not the size this instance was built for.
    ///
    /// Every block validates its buffer lengths exactly rather than accepting
    /// a prefix, so the token count is fixed at construction.
    WrongTokenCount { expected: usize, got: usize },
    /// `stage_image_rows` without `enable_image_injection` — vision serving
    /// must pre-allocate at load, never lazily (AGENTS.md rule 6).
    ImageInjectionNotEnabled,
    /// Sequence state could not be allocated or reset.
    State(StateError),
    /// [`Forward::capture_step`] was called on a pass with stage profiling on.
    ///
    /// The profiler records CUDA events between stages, and an event recorded
    /// inside a capture belongs to the graph rather than to the wall clock —
    /// so the two cannot both be true of one pass.
    CaptureWhileProfiling,
    /// The driver returned no graph from a capture. Nothing was recorded.
    EmptyCapture,
    /// A replayed step would write past the end of the key/value caches.
    CacheExhausted {
        /// Position the sequence had reached.
        position: usize,
        /// Positions this pass would add.
        tokens: usize,
        /// Positions the caches hold.
        max_seq: usize,
    },
    /// The state carries a different number of layers than this pass runs.
    ///
    /// A state is indexed by slot inside the pass, so a mismatch would silently
    /// fold layer 7's tokens into layer 6's recurrent matrix rather than fail.
    StateShape {
        expected_gdn: usize,
        expected_attention: usize,
        got_gdn: usize,
        got_attention: usize,
    },
    /// [`Forward::run_batch_decode`] was called before [`Forward::enable_batch_decode`].
    ///
    /// That method allocates the per-row LM head and argmax scratch batched
    /// decode needs, and `AGENTS.md` rule 6 makes allocating on this call a
    /// bug rather than a convenience — so it is a precondition the caller
    /// states explicitly, once, rather than something this method does for
    /// them on its first call.
    BatchDecodeNotEnabled,
    /// [`Forward::run_batch_prefill`] was called before its sequence-width
    /// output and position scratch was allocated.
    BatchPrefillNotEnabled,
    /// A batched-decode pass or its per-sequence states disagree with the
    /// batch width this pass was built for.
    BatchWidth {
        expected: usize,
        got: usize,
        what: &'static str,
    },
}

impl std::fmt::Display for ForwardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compile(m) => write!(f, "embedding kernel compilation failed: {m}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::Gdn(e) => write!(f, "{e}"),
            Self::GdnVerify(e) => write!(f, "{e}"),
            Self::Attention(e) => write!(f, "{e}"),
            Self::Moe(e) => write!(f, "{e}"),
            Self::LayerOps(e) => write!(f, "{e}"),
            Self::LmHead(e) => write!(f, "{e}"),
            Self::Mixer(e) => write!(f, "{e}"),
            Self::MissingWeight { role, layer } => match layer {
                Some(l) => write!(f, "block {l} has no resident `{role}`"),
                None => write!(f, "`{role}` is not resident"),
            },
            Self::WrongQuant {
                role,
                layer,
                found,
                expected,
            } => {
                let where_ = match layer {
                    Some(l) => format!("blk.{l}."),
                    None => String::new(),
                };
                write!(
                    f,
                    "`{where_}{role}` is {}, this pass unpacks {}",
                    found.name(),
                    expected.name(),
                )
            }
            Self::WrongTableSize {
                role,
                expected,
                got,
            } => write!(
                f,
                "`{role}` holds {got} B, the vocabulary implies {expected} B",
            ),
            Self::MissingMetadata { key } => write!(
                f,
                "the model file does not declare `{key}`, and guessing it would produce \
                 a finite, wrong forward pass",
            ),
            Self::WrongTokenCount { expected, got } => write!(
                f,
                "this pass was built for {expected} tokens and was given {got}",
            ),
            Self::ImageInjectionNotEnabled => write!(
                f,
                "image rows staged on a pass without enable_image_injection"
            ),
            Self::State(e) => write!(f, "{e}"),
            Self::CaptureWhileProfiling => {
                write!(f, "cannot capture a step while stage profiling is enabled",)
            }
            Self::EmptyCapture => write!(f, "the capture recorded no work"),
            Self::CacheExhausted {
                position,
                tokens,
                max_seq,
            } => write!(
                f,
                "{tokens} tokens at position {position} would need {} cache slots, \
                 but the state holds {max_seq}",
                position + tokens,
            ),
            Self::StateShape {
                expected_gdn,
                expected_attention,
                got_gdn,
                got_attention,
            } => write!(
                f,
                "this pass runs {expected_gdn} Gated DeltaNet and {expected_attention} attention \
                 layers, but the state carries {got_gdn} and {got_attention}",
            ),
            Self::BatchDecodeNotEnabled => write!(
                f,
                "run_batch_decode was called before enable_batch_decode allocated its scratch",
            ),
            Self::BatchPrefillNotEnabled => write!(
                f,
                "run_batch_prefill was called before enable_batch_prefill allocated its scratch",
            ),
            Self::BatchWidth {
                expected,
                got,
                what,
            } => write!(
                f,
                "this pass was built for a batch of {expected}, but {what} holds {got}",
            ),
        }
    }
}

impl std::error::Error for ForwardError {}

macro_rules! from_error {
    ($src:ty, $variant:ident) => {
        impl From<$src> for ForwardError {
            fn from(e: $src) -> Self {
                Self::$variant(e)
            }
        }
    };
}
from_error!(DriverError, Driver);
from_error!(GdnBlockError, Gdn);
from_error!(GdnVerifyError, GdnVerify);
from_error!(AttentionBlockError, Attention);
from_error!(MoeBlockError, Moe);
from_error!(LayerOpsError, LayerOps);
from_error!(LmHeadError, LmHead);
from_error!(AttentionError, Mixer);

/// Where the pass's device memory went, measured rather than predicted.
#[derive(Debug, Clone, Copy)]
pub struct ForwardReport {
    /// Bytes in the weight arena — everything read by alias, zero-copy.
    pub arena_bytes: u64,
    /// Bytes of MoE weights held outside the arena, across all 40 layers.
    ///
    /// Not a duplicate: these roles are filtered out of the arena, so this is
    /// the model's only copy of them.
    pub moe_bytes: u64,
    /// Bytes this shape newly owns for Gated Attention **in addition** to the
    /// arena's copy of the same tensors. A reshape reports zero because it
    /// shares its donor's weights.
    ///
    /// This one is genuine duplication. See the module docs.
    pub attention_duplicate_bytes: u64,
    /// Free VRAM before anything was allocated.
    pub free_before: u64,
    /// Free VRAM once every block was constructed.
    pub free_after: u64,
}

impl ForwardReport {
    /// VRAM the driver actually consumed, by its own accounting.
    pub fn vram_consumed(&self) -> u64 {
        self.free_before.saturating_sub(self.free_after)
    }

    /// Total weight bytes resident, arena and MoE together.
    pub fn weight_bytes(&self) -> u64 {
        self.arena_bytes + self.moe_bytes
    }
}

/// Roles the MoE block uploads for itself.
///
/// Filtered out of the weight arena so the model is resident exactly once.
/// Every entry is a role [`MoeLayerWeights::upload`] reads; if that list ever
/// changes, `the_moe_role_filter_is_exactly_what_the_moe_block_uploads`
/// notices, because the arena would then hold a tensor nothing reads or omit
/// one something does.
pub const MOE_OWNED_ROLES: [Role; 9] = [
    Role::PostMixerNorm,
    Role::MoeRouter,
    Role::MoeGateExps,
    Role::MoeUpExps,
    Role::MoeDownExps,
    Role::MoeSharedGateInp,
    Role::MoeSharedGate,
    Role::MoeSharedUp,
    Role::MoeSharedDown,
];

/// Whether the weight arena should hold `role`.
///
/// Pass this to [`DeviceWeights::load_where`] before constructing a
/// [`Forward`]: it is the other half of the arrangement the module docs
/// describe, and loading the full model instead would put 28.3 GiB of expert
/// weights on the card that nothing ever reads.
pub fn arena_holds(role: Role) -> bool {
    !MOE_OWNED_ROLES.contains(&role)
}

/// One timed span inside [`Forward::run`].
///
/// The spans partition the pass exactly: they are consecutive, none overlaps,
/// and their sum is the pass. That is what makes a percentage column
/// meaningful rather than merely suggestive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Zeroing the 30 Gated DeltaNet states and uploading the token ids.
    Reset,
    /// The `get_rows` embedding gather.
    Embed,
    /// One block's mixer — Gated DeltaNet or Gated Attention.
    Mixer { layer: u32, kind: LayerKind },
    /// One block's MoE: post-mixer norm, router, dispatch, grouped GEMM,
    /// shared expert, combine, residual add.
    Moe { layer: u32 },
    /// The final RMSNorm over all positions.
    FinalNorm,
    /// The last-position row copy and the LM head GEMV.
    LmHead,
}

/// Which point inside one layer a diagnostic waypoint was read at.
///
/// Every pass tags its waypoints this way; the ordinary entry points pass a
/// callback that ignores the tag, and the `*_with_stage_waypoints` siblings
/// are the ones that read it — the batch-vs-single-stream divergence audit's
/// instrumentation. `Stage` above exists for GPU-time profiling of the same
/// pass and is a different axis: this one names *what buffer* a callback is
/// looking at, not how long it took to produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaypointStage {
    /// Right after the embedding gather, before any layer runs.
    Embed,
    /// Right after the layer's mixer (Gated DeltaNet or Gated Attention),
    /// before the MoE block folds it into the residual stream. This is the
    /// buffer a divergence audit needs to tell "the mixer introduced this"
    /// from "the MoE block introduced this". [`Forward::run`]'s own callback
    /// filters this stage out, so the buffer stays private to the audit.
    Mixer,
    /// Right after the layer's MoE block, once its residual add has landed
    /// in the hidden state that feeds the next layer.
    Moe,
}

impl std::fmt::Display for Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reset => write!(f, "reset"),
            Self::Embed => write!(f, "embed"),
            Self::Mixer { layer, kind } => {
                let k = match kind {
                    LayerKind::GatedDeltaNet => "gdn",
                    LayerKind::GatedAttention => "attn",
                };
                write!(f, "mixer[{layer}:{k}]")
            }
            Self::Moe { layer } => write!(f, "moe[{layer}]"),
            Self::FinalNorm => write!(f, "final_norm"),
            Self::LmHead => write!(f, "lm_head"),
        }
    }
}

/// A per-stage GPU timeline for one pass, recorded with CUDA events.
///
/// # Why events and not `Instant`
///
/// Every launch in this pass is asynchronous, so a host clock read between two
/// stages measures *launch* time, not work. Getting wall clock out of an
/// `Instant` would need a `cuStreamSynchronize` per stage — 84 of them — which
/// drains the pipeline 84 times and inflates the total by far more than the
/// thing being measured. `cuEventRecord` is a stream-ordered marker: it costs
/// one enqueue at record time and the host reads the timeline once, after the
/// pass, from `cuEventElapsedTime`.
///
/// The events are created once, at [`Forward::enable_profiling`], and reused
/// every pass — `AGENTS.md` rule 6 forbids allocating on this path, and that
/// applies to the instrumentation as much as to the pass.
///
/// The residual overhead is real but small, and it is measured rather than
/// assumed: run `bench_forward` with and without `LLMXABE_PROFILE` and the
/// difference is the instrumentation. See `docs/BENCHMARKS.md`.
pub struct StageProfile {
    events: Vec<CudaEvent>,
    stages: Vec<Stage>,
    cursor: usize,
}

impl StageProfile {
    /// `spans` timed spans need `spans + 1` boundary markers.
    fn new(ctx: &Arc<CudaContext>, spans: usize) -> Result<Self, ForwardError> {
        let mut events = Vec::with_capacity(spans + 1);
        for _ in 0..=spans {
            // The default flag keeps timing enabled; `new_event(None)` would
            // create the event with `CU_EVENT_DISABLE_TIMING` and
            // `elapsed_ms` would then fail.
            events.push(ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?);
        }
        Ok(Self {
            events,
            stages: Vec::with_capacity(spans),
            cursor: 0,
        })
    }

    /// Record the opening marker and forget the previous pass.
    fn begin(&mut self, stream: &Arc<CudaStream>) -> Result<(), ForwardError> {
        self.stages.clear();
        self.cursor = 0;
        self.events[0].record(stream)?;
        self.cursor = 1;
        Ok(())
    }

    /// Close the span named `stage` at the stream's current position.
    fn mark(&mut self, stream: &Arc<CudaStream>, stage: Stage) -> Result<(), ForwardError> {
        self.events[self.cursor].record(stream)?;
        self.cursor += 1;
        self.stages.push(stage);
        Ok(())
    }

    /// The last pass's timeline, in milliseconds of GPU time per span.
    ///
    /// Synchronizes on the events it reads, so it is safe to call immediately
    /// after `run` returns.
    pub fn spans(&self) -> Result<Vec<(Stage, f64)>, ForwardError> {
        let mut out = Vec::with_capacity(self.stages.len());
        for (i, &stage) in self.stages.iter().enumerate() {
            let ms = self.events[i].elapsed_ms(&self.events[i + 1])?;
            out.push((stage, ms as f64));
        }
        Ok(out)
    }
}

/// The whole model, resident, wired end to end.
pub struct Forward {
    config: ModelConfig,
    tokens: usize,
    hidden: usize,
    vocab: usize,
    rms_eps: f32,

    embed_fn: CudaFunction,
    layer_ops: LayerOpsKernels,
    gdn: GdnBlock,
    attention: Vec<GatedAttentionBlock>,
    /// Shape-independent attention weights and lazy split-layout repacks.
    attention_weights: Arc<Vec<Arc<AttentionLayerWeights>>>,
    /// One set of per-pass buffers for all ten attention layers. They run in
    /// sequence and nothing crosses a layer boundary, so ten private copies
    /// were ten times the per-token VRAM for no benefit. See [`AttnScratch`].
    attn_scratch: AttnScratch,
    moe: MoeBlock,
    lm_head: LmHeadKernels,

    /// Zero-copy aliases into the weight arena. Never dropped — see
    /// [`crate::weights::ResidentTensor`].
    w_token_embd: ManuallyDrop<CudaSlice<u8>>,
    w_output_norm: ManuallyDrop<CudaSlice<f32>>,
    w_lm_head: ManuallyDrop<CudaSlice<u8>>,
    gdn_weights: Vec<ManuallyDrop<GdnLayerWeights>>,
    /// The Q8_0 projections in the split layout, one per Gated DeltaNet
    /// layer — empty only after [`Self::disable_tensor_cores`].
    ///
    /// Three consumers share this buffer: the integer tensor cores at
    /// `tokens >= MMA_SPLIT_TOKENS`, `gdn_proj_split_gemv` at one token,
    /// and `gdn_proj_split_t*` at `1 < tokens < MMA_SPLIT_TOKENS` (batch
    /// decode). Building it only at 1 and >=64 left a 2..63 prefill with
    /// an empty vector; `reshape` then handed that empty vector to a
    /// batch-decode shape and the projections fell back to the standard
    /// Q8_0 tile, which reopens the 7.15e-7 GDN layer-0 residual the split
    /// layout closed — invisibly at context 2048, where the prefill is
    /// already >=64.
    gdn_int8: Arc<Vec<GdnLayerInt8>>,
    /// Shared with every other shape built over the same model.
    ///
    /// At 725 MiB per layer these are 28.3 GiB — the single largest thing on
    /// the card, and the model's only copy of those roles. A second `Forward`
    /// for a different token count must borrow them rather than upload again;
    /// two copies do not fit on a 48 GiB device, which is why this is an
    /// `Arc` rather than a `Vec`. See [`Forward::reshape`].
    moe_weights: Arc<Vec<MoeLayerWeights>>,

    d_tokens: CudaSlice<i32>,
    hidden_state: CudaSlice<f32>,
    mixer_out: CudaSlice<f32>,
    ffn_out: CudaSlice<f32>,
    final_norm: CudaSlice<f32>,
    last_hidden: CudaSlice<f32>,
    logits: CudaSlice<f32>,

    /// Scratch for [`Self::sample_argmax`]: the first pass's per-block
    /// winners, and the single `i32` the second pass reduces them to.
    argmax_values: CudaSlice<f32>,
    argmax_indices: CudaSlice<i32>,
    argmax_out: CudaSlice<i32>,

    /// `None` unless [`Self::enable_profiling`] was called. The hot path pays
    /// one always-false branch per stage boundary when it is `None`.
    profile: Option<StageProfile>,

    /// Built by [`Self::enable_batch_decode`]; `None` until then.
    ///
    /// Sized for `self.tokens` rows rather than the fixed one [`Self::lm_head`]
    /// reads: batched decode's every row is a distinct sequence's last (and
    /// only) position, so there is no "select the last row" step to narrow
    /// `self.tokens` rows down to one the way `Self::body` does for every
    /// other shape this type builds.
    batch_lm_head: Option<LmHeadKernels>,
    /// `[tokens][vocab]`, one row per sequence in the batch.
    batch_logits: Option<CudaSlice<f32>>,
    /// Per-sequence argmax scratch, `[tokens][ARGMAX_BLOCKS]` each. The
    /// kernels themselves are geometry-free (see [`Self::lm_head`]'s
    /// `argmax`), so these are the only new allocation batched sampling needs.
    batch_argmax_values: Option<CudaSlice<f32>>,
    batch_argmax_indices: Option<CudaSlice<i32>>,
    /// One sampled id per sequence.
    batch_argmax_out: Option<CudaSlice<i32>>,
    /// Every sequence's own position, published in one copy instead of
    /// `tokens` separate ones. Gated Attention's per-sequence loop reads a
    /// one-element view of this rather than the [`SequenceState`]'s own
    /// `d_position` — see [`Self::publish_batch_inputs`].
    batch_positions: Option<CudaSlice<i32>>,
    /// Host staging for the copy above, pre-sized once so publishing a step
    /// never allocates (`AGENTS.md` rule 6).
    batch_positions_host: Vec<i32>,
    /// `(t, h, w)` rotary triples for one image-overlapping prefill chunk,
    /// `3 * tokens` i32. Published by [`Self::publish_mrope`] and consumed
    /// by [`Self::body`]'s attention layers when [`Self::mrope_active`];
    /// 12 KiB at the deepest chunk shape, so it is allocated
    /// unconditionally rather than gated on vision being enabled.
    d_mrope: CudaSlice<i32>,
    /// Whether the *next* pass rotates by [`Self::d_mrope`] instead of the
    /// per-sequence scalar. Never true on a captured pass — asserted at
    /// capture — so decode graphs always bind the scalar path.
    mrope_active: bool,
    /// Device staging for image embedding rows, `tokens * hidden` f32,
    /// allocated by [`Self::enable_image_injection`] when the server loads
    /// an mmproj — text-only serving never allocates it. Rows are copied
    /// here at chunk-preparation time and replayed over the freshly
    /// embedded `hidden_state` inside [`Self::body`], right after the
    /// token-id gather whose placeholder rows they replace.
    image_stage: Option<CudaSlice<f32>>,
    /// `(dst_token, stage_row, n_rows)` copies pending for the next pass.
    /// Always empty on a captured pass — asserted at capture.
    staged_rows: SmallVec<[(usize, usize, usize); 4]>,
    /// Rows of [`Self::image_stage`] already claimed by `staged_rows`.
    stage_used: usize,
    /// Stable pageable source used only while capturing the batch graph.
    batch_zero_tokens: Vec<i32>,
    /// Fixed fork/join resources for the sequence-local part of batched
    /// attention. Created before serving and retained with captured graphs.
    batch_attention: Option<BatchAttentionStreams>,

    /// Allocated by `enable_verify`, once per Gated DeltaNet layer.
    verify_scratch: Option<Vec<GdnVerifyScratch>>,

    report: ForwardReport,
}

/// One decode step, recorded once and launched as a unit.
///
/// Built by [`Forward::capture_step`] and launched by
/// [`Forward::replay_step`]. It holds device pointers into the `Forward` and
/// the `SequenceState` it was captured from, so replaying it against a
/// *different* pass or a different state would read and write the wrong
/// buffers — which is why neither of those is a parameter of the replay and
/// why this type carries no way to reach one.
pub struct StepGraph {
    graph: CudaGraph,
}

/// One batched decode step, recorded once and launched as a unit.
///
/// Built by [`Forward::capture_batch_step`] and launched by
/// [`Forward::replay_batch_step`]. It holds device pointers into the
/// batch-width `Forward` it was captured from and every [`SequenceState`] in
/// the batch — replaying it against a different pass, a different batch
/// width, or the states in a different order would read and write the wrong
/// buffers, the same way [`StepGraph`] cannot be replayed against a
/// different pass or state.
pub struct BatchStepGraph {
    graph: CudaGraph,
}

struct BatchAttentionStreams {
    secondary: Vec<Arc<CudaStream>>,
    fork: CudaEvent,
    joins: Vec<CudaEvent>,
}

impl Forward {
    /// Build the whole stack for exactly `tokens` positions.
    ///
    /// `weights` must have been loaded with [`arena_holds`] as the filter, and
    /// `directory` must be the one it was loaded from, resolved against
    /// `file`. The token count is fixed here rather than per call because
    /// every kernel in the chain validates its buffer lengths against the
    /// declared geometry exactly.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        file: &GgufFile,
        directory: &Directory<'_>,
        weights: &DeviceWeights,
        config: ModelConfig,
        tokens: usize,
    ) -> Result<Self, ForwardError> {
        Self::build(
            ctx, stream, file, directory, weights, config, tokens, None, None, None,
        )
    }

    /// Build a second pass over the **same resident weights**, for a different
    /// token count.
    ///
    /// This is what makes decode possible at all. Every kernel in the chain
    /// validates its buffer lengths against the declared geometry exactly, so
    /// a pass is built for one token count and cannot serve another — but the
    /// 28.3 GiB of MoE weights cannot be uploaded twice on a 48 GiB card, and
    /// re-uploading them per shape would cost more than the generation.
    ///
    /// So the shapes are separate and the weights are not. The returned pass
    /// shares this one's MoE weights by reference count and re-derives every
    /// other weight as a fresh alias into the same arena; only the scratch and
    /// the ten attention blocks' copies are genuinely new, which at `tokens =
    /// 1` is a few hundred MiB rather than thirty gigabytes.
    ///
    /// The two passes are still separate objects with separate buffers, so
    /// they must be driven over one [`SequenceState`] to be one sequence.
    ///
    /// [`ForwardReport::moe_bytes`] on the result is zero: the weights are
    /// real but they are not this pass's, and counting them twice would make
    /// the sum of two reports claim more VRAM than the card has.
    pub fn reshape(
        &self,
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        file: &GgufFile,
        directory: &Directory<'_>,
        weights: &DeviceWeights,
        tokens: usize,
    ) -> Result<Self, ForwardError> {
        Self::build(
            ctx,
            stream,
            file,
            directory,
            weights,
            self.config.clone(),
            tokens,
            Some(Arc::clone(&self.moe_weights)),
            Some(Arc::clone(&self.gdn_int8)),
            Some(Arc::clone(&self.attention_weights)),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        file: &GgufFile,
        directory: &Directory<'_>,
        weights: &DeviceWeights,
        config: ModelConfig,
        tokens: usize,
        shared_moe: Option<Arc<Vec<MoeLayerWeights>>>,
        gdn_int8: Option<Arc<Vec<GdnLayerInt8>>>,
        shared_attention: Option<Arc<Vec<Arc<AttentionLayerWeights>>>>,
    ) -> Result<Self, ForwardError> {
        let hidden = config.hidden_size as usize;
        let owns_attention_weights = shared_attention.is_none();
        let vocab = config.vocab_size as usize;
        let (free_before, _) = xabe_cuda::arena::memory_info(ctx)?;

        let rms_eps = file
            .get_f32(RMS_EPS_KEY)
            .ok_or(ForwardError::MissingMetadata { key: RMS_EPS_KEY })?;
        let rope_theta = file
            .get_f32(ROPE_FREQ_BASE_KEY)
            .ok_or(ForwardError::MissingMetadata {
                key: ROPE_FREQ_BASE_KEY,
            })?;

        let ptx = compile(EMBED_SRC, "forward_embed").map_err(ForwardError::Compile)?;
        let module = ctx.load_module(ptx)?;
        let embed_fn = module.load_function("fwd_embed_q8_0")?;

        let layer_ops = LayerOpsKernels::new(ctx)?;

        // --- the three global tensors, by alias ---------------------------
        //
        // The two Q8_0 tables are checked against `vocab * hidden` here rather
        // than trusted, because the embedding gather below indexes by a token
        // id and has no other bound to check against: a short table would read
        // past its own end for a high-numbered token.
        let w_token_embd = alias_q8_0(weights, stream, Role::TokenEmbedding, None)?;
        let w_lm_head = alias_q8_0(weights, stream, Role::LmHead, None)?;
        let table_bytes = vocab * hidden / QK8_0 * BLOCK_Q8_0_BYTES;
        for (role, len) in [
            (Role::TokenEmbedding, w_token_embd.len()),
            (Role::LmHead, w_lm_head.len()),
        ] {
            if len != table_bytes {
                return Err(ForwardError::WrongTableSize {
                    role,
                    expected: table_bytes,
                    got: len,
                });
            }
        }
        let w_output_norm = ManuallyDrop::new(
            // SAFETY: sealed in a `ManuallyDrop` that lives in `self` and is
            // never taken out, and `self` cannot outlive `weights` because the
            // caller holds both for the pass's lifetime.
            unsafe {
                weights
                    .f32_of(stream, Role::OutputNorm, None)
                    .ok_or(ForwardError::MissingWeight {
                        role: Role::OutputNorm,
                        layer: None,
                    })?
                    .into_aliasing_slice()
            },
        );

        // --- the 30 Gated DeltaNet layers, by alias -----------------------
        let gdn_geometry = GdnGeometry::from_config(&config, tokens, rms_eps);
        let gdn = GdnBlock::new(ctx, gdn_geometry)?;
        let mut gdn_weights = Vec::new();
        for layer in 0..config.num_layers {
            if config.layer_kind(layer) != LayerKind::GatedDeltaNet {
                continue;
            }
            gdn_weights.push(ManuallyDrop::new(alias_gdn_layer(weights, stream, layer)?));
        }

        // The repack serves three paths: the tensor cores above the
        // threshold, a one-token GEMV that wants the split layout for its
        // *alignment* rather than its arithmetic — Q8_0's 34-byte block stride
        // makes an in-place read straddle a sector boundary fifteen times in
        // sixteen — and the same GEMV tiled over `1 < tokens < MMA_SPLIT_
        // TOKENS` so batch decode and single-stream share a reduction order.
        // See `gdn_proj_split_gemv` / `gdn_proj_split_t*`.
        //
        // Shared through `reshape` like the MoE weights, and for the same
        // reason: a prefill pass and the decode pass driven over the same
        // sequence would otherwise hold two copies of 1.13 GiB.
        // **Sharing is conditional on the donor having built it.** A pass that
        // wants none of the three paths builds an empty vector, and `reshape`
        // hands that empty vector to the shape it spawns. That used to be
        // the 4..63-token hole: those prefills built nothing, so a one-token
        // decode they spawned fell back to the fp32 projection (13% of every
        // decode step — 9.7 ms became 11.0) and a 2..3-token batch decode
        // they spawned took the standard-layout tile (the 7.15e-7 GDN
        // layer-0 residual). Every width now has a consumer, so every
        // width builds it.
        let gdn_int8 = match gdn_int8 {
            Some(shared) if shared.is_empty() => {
                let mut built = Vec::new();
                for w in &gdn_weights {
                    built.push(gdn.repack(stream, w)?);
                }
                Arc::new(built)
            }
            Some(shared) => shared,
            None => {
                let mut built = Vec::new();
                for w in &gdn_weights {
                    built.push(gdn.repack(stream, w)?);
                }
                Arc::new(built)
            }
        };

        // --- the 10 Gated Attention layers, shared across shapes ----------
        let attn_kernels = Arc::new(AttentionKernelSet::new(ctx, &config, tokens)?);
        let mut attention = Vec::new();
        let mut shared_index = 0;
        for layer in 0..config.num_layers {
            if config.layer_kind(layer) != LayerKind::GatedAttention {
                continue;
            }
            let block = match &shared_attention {
                Some(shared) => {
                    let layer_weights = shared
                        .get(shared_index)
                        .expect("reshape attention weights match the model config");
                    GatedAttentionBlock::from_shared(
                        Arc::clone(&attn_kernels),
                        stream,
                        Arc::clone(layer_weights),
                        &config,
                        layer,
                        tokens,
                        rms_eps,
                        rope_theta,
                    )?
                }
                None => GatedAttentionBlock::new(
                    Arc::clone(&attn_kernels),
                    stream,
                    weights,
                    &config,
                    layer,
                    tokens,
                    rms_eps,
                    rope_theta,
                )?,
            };
            attention.push(block);
            shared_index += 1;
        }
        debug_assert!(
            shared_attention
                .as_ref()
                .is_none_or(|shared| shared_index == shared.len())
        );
        let attention_weights = shared_attention.unwrap_or_else(|| {
            Arc::new(
                attention
                    .iter()
                    .map(GatedAttentionBlock::shared_weights)
                    .collect(),
            )
        });

        // One scratch for all ten of them: they run in sequence and nothing
        // crosses a layer boundary. Ten private copies cost 1.68 MB of VRAM
        // per token of context against this one's 0.17 MB.
        let attn_scratch = AttnScratch::new(stream, &config, tokens)?;

        // --- the MoE, on every block ---------------------------------------
        let moe_geometry = MoeBlock::geometry_for(&config, moe_block_size(tokens), tokens);
        let moe = MoeBlock::new(ctx, stream, moe_geometry, rms_eps)?;
        let (moe_weights, moe_bytes) = match shared_moe {
            // Already on the card, uploaded by the pass this one was reshaped
            // from. Reported as zero bytes because they are not this pass's to
            // account for — see `reshape`.
            Some(shared) => (shared, 0u64),
            None => {
                let mut uploaded = Vec::with_capacity(config.num_layers as usize);
                let mut bytes = 0u64;
                for layer in 0..config.num_layers {
                    let w = MoeLayerWeights::upload(stream, file, directory, layer, &moe_geometry)?;
                    bytes += w.bytes() as u64;
                    uploaded.push(w);
                }
                (Arc::new(uploaded), bytes)
            }
        };

        let lm_head = LmHeadKernels::new(
            ctx,
            LmHeadGeometry {
                hidden,
                vocab,
                // One position: llama.cpp applies `get_rows(cur, inp_out_ids)`
                // before the head and a plain prefill asks for the last token
                // only, which is what `result_output` is.
                max_tokens: 1,
            },
        )?;

        let attention_duplicate_bytes = if owns_attention_weights {
            attention_bytes(weights, &config)
        } else {
            0
        };
        stream.synchronize()?;
        let (free_after, _) = xabe_cuda::arena::memory_info(ctx)?;

        Ok(Self {
            config,
            tokens,
            hidden,
            vocab,
            rms_eps,
            embed_fn,
            layer_ops,
            gdn,
            attention,
            attention_weights,
            attn_scratch,
            moe,
            lm_head,
            w_token_embd,
            w_output_norm,
            w_lm_head,
            gdn_weights,
            gdn_int8,
            moe_weights,
            d_tokens: stream.alloc_zeros::<i32>(tokens)?,
            hidden_state: stream.alloc_zeros::<f32>(tokens * hidden)?,
            mixer_out: stream.alloc_zeros::<f32>(tokens * hidden)?,
            ffn_out: stream.alloc_zeros::<f32>(tokens * hidden)?,
            final_norm: stream.alloc_zeros::<f32>(tokens * hidden)?,
            last_hidden: stream.alloc_zeros::<f32>(hidden)?,
            logits: stream.alloc_zeros::<f32>(vocab)?,
            argmax_values: stream.alloc_zeros::<f32>(ARGMAX_BLOCKS)?,
            argmax_indices: stream.alloc_zeros::<i32>(ARGMAX_BLOCKS)?,
            argmax_out: stream.alloc_zeros::<i32>(1)?,
            profile: None,
            batch_lm_head: None,
            batch_logits: None,
            batch_argmax_values: None,
            batch_argmax_indices: None,
            batch_argmax_out: None,
            batch_positions: None,
            batch_positions_host: Vec::new(),
            d_mrope: stream.alloc_zeros::<i32>(3 * tokens)?,
            mrope_active: false,
            image_stage: None,
            staged_rows: SmallVec::new(),
            stage_used: 0,
            batch_zero_tokens: Vec::new(),
            batch_attention: None,
            verify_scratch: None,
            report: ForwardReport {
                arena_bytes: weights.arena().capacity() as u64,
                moe_bytes,
                attention_duplicate_bytes,
                free_before,
                free_after,
            },
        })
    }

    /// Where this pass's device memory went.
    pub fn report(&self) -> ForwardReport {
        self.report
    }

    /// Start recording a per-stage GPU timeline on every subsequent [`Self::run`].
    ///
    /// Off by default: `bench_forward`'s figure must stay the uninstrumented
    /// one. The events are allocated here, once, so enabling this does not
    /// allocate on the pass itself.
    pub fn enable_profiling(&mut self, ctx: &Arc<CudaContext>) -> Result<(), ForwardError> {
        // Reset, embed, a mixer and a MoE per block, the final norm, the head.
        let spans = 2 * self.config.num_layers as usize + 4;
        self.profile = Some(StageProfile::new(ctx, spans)?);
        Ok(())
    }

    /// Stop recording, and release the events.
    pub fn disable_profiling(&mut self) {
        self.profile = None;
    }

    /// The last pass's per-stage timeline, if [`Self::enable_profiling`] was called.
    pub fn profile(&self) -> Option<&StageProfile> {
        self.profile.as_ref()
    }

    /// Close a timed span, when profiling is on.
    #[inline]
    fn mark(&mut self, stream: &Arc<CudaStream>, stage: Stage) -> Result<(), ForwardError> {
        match &mut self.profile {
            Some(p) => p.mark(stream, stage),
            None => Ok(()),
        }
    }

    /// Positions this pass was built for.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Output vocabulary, and so the length of [`Self::logits`].
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// `model.input_embed`: the embedding lookup, `[tokens][hidden]`.
    pub fn embedded(&self) -> &CudaSlice<f32> {
        // Valid only immediately after `run`'s first step, and overwritten by
        // block 0's residual; `run` therefore hands the caller a view of every
        // block boundary as it produces it. This accessor is the input.
        &self.hidden_state
    }

    /// `h_nextn`: the final output norm over **all** positions.
    pub fn final_norm(&self) -> &CudaSlice<f32> {
        &self.final_norm
    }

    /// `result_output`: the logits for the last position, `[vocab]`.
    pub fn logits(&self) -> &CudaSlice<f32> {
        &self.logits
    }

    /// `[tokens][vocab]` logits from the last [`Self::run_batch_decode`], one
    /// row per sequence in that call's `states` order. `None` until
    /// [`Self::enable_batch_decode`] has been called.
    pub fn batch_logits(&self) -> Option<&CudaSlice<f32>> {
        self.batch_logits.as_ref()
    }

    /// Greedy sampling: the id of the largest logit, ties to the lower id.
    ///
    /// Synchronizes the stream, because the caller needs the token before it
    /// can decide what to feed back in — the round-trip is unavoidable in an
    /// autoregressive loop. What is avoidable is its *width*: reducing on the
    /// device makes the transfer four bytes instead of the 993 KiB a
    /// 248,320-entry logit vector occupies, and it moves the scan itself off
    /// a single CPU core. That was worth about 4% of a decode step.
    ///
    /// The tie-break matches `xabe_kernels::gemv::argmax` exactly, which is
    /// what `tests/lm_head_differential.rs` asserts. A near-tie between two
    /// logits is the one place a kernel can be well inside every tolerance
    /// and still emit a different token.
    pub fn sample_argmax(&mut self, stream: &Arc<CudaStream>) -> Result<i32, ForwardError> {
        self.launch_argmax(stream)?;
        self.read_sampled(stream)
    }

    /// The two argmax kernels, without the read-back.
    ///
    /// Separate from [`Self::read_sampled`] so the reduction can live inside a
    /// captured step while the four-byte transfer, which needs host memory and
    /// a synchronize, stays outside it.
    fn launch_argmax(&mut self, stream: &Arc<CudaStream>) -> Result<(), ForwardError> {
        let vocab = self.vocab;
        self.lm_head.argmax(
            stream,
            &self.logits,
            vocab,
            &mut self.argmax_values,
            &mut self.argmax_indices,
            &mut self.argmax_out,
        )?;
        Ok(())
    }

    /// Read the token id the last [`Self::launch_argmax`] chose.
    ///
    /// Synchronizes: an autoregressive step is not finished until the host
    /// knows what to feed back in.
    fn read_sampled(&self, stream: &Arc<CudaStream>) -> Result<i32, ForwardError> {
        let host = stream.clone_dtoh(&self.argmax_out)?;
        stream.synchronize()?;
        Ok(host[0])
    }

    /// Copy the last position's logits to `host`, for host-side sampling.
    ///
    /// Synchronizes, exactly as [`Self::sample_argmax`] does and for the same
    /// reason: the caller is about to choose the next token from these
    /// values. `host` is caller-owned and resized here so a serving runtime
    /// can pre-allocate it once and reuse it (AGENTS.md rule 6).
    pub fn read_logits_into(
        &self,
        stream: &Arc<CudaStream>,
        host: &mut Vec<f32>,
    ) -> Result<(), ForwardError> {
        host.resize(self.vocab, 0.0);
        stream.memcpy_dtoh(&self.logits, host.as_mut_slice())?;
        stream.synchronize()?;
        Ok(())
    }

    /// Copy one sequence's logits row from the last batched decode step to
    /// `host`, for host-side sampling.
    ///
    /// `row` indexes the `states` order of that step. The batched step's
    /// argmax has already run on the device by the time a caller wants this,
    /// so reading the row is purely additive: the greedy path is untouched
    /// and pays nothing.
    pub fn read_batch_logits_row_into(
        &self,
        stream: &Arc<CudaStream>,
        row: usize,
        host: &mut Vec<f32>,
    ) -> Result<(), ForwardError> {
        let logits = self
            .batch_logits
            .as_ref()
            .ok_or(ForwardError::BatchDecodeNotEnabled)?;
        if row >= self.tokens {
            return Err(ForwardError::BatchWidth {
                expected: self.tokens,
                got: row,
                what: "logits row",
            });
        }
        let vocab = self.vocab;
        // SAFETY: `row < self.tokens` and `batch_logits` is
        // `self.tokens * vocab` elements, allocated by `enable_batch_decode`,
        // so the view is in bounds and outlived by the buffer.
        let view = unsafe { crate::viewslice::subslice(stream, logits, row * vocab, vocab) };
        host.resize(vocab, 0.0);
        stream.memcpy_dtoh(&*view, host.as_mut_slice())?;
        stream.synchronize()?;
        Ok(())
    }

    /// Record one whole step — embedding, forty blocks, LM head, argmax — as a
    /// CUDA graph, for [`Self::replay_step`] to launch at every later position.
    ///
    /// **Why.** A one-token step issues about 1,100 device operations, and
    /// `Forward::run` spends roughly 3 ms of host time issuing them against
    /// kernels that average 8.6 us. The GPU is idle for about 0.6 us between
    /// consecutive launches, which is ~0.63 ms per step, plus a ~0.31 ms host
    /// turnaround at the step boundary. A graph is the same sequence submitted
    /// as one object.
    ///
    /// **What made it possible.** Nothing in `Self::body` may take a value
    /// that changes between steps as a host argument, because a recorded
    /// launch keeps the arguments it was recorded with. The sequence position
    /// was the last one: the rotary embedding, the causal bound and the
    /// key/value append all read it from
    /// `SequenceState::publish_position`'s device scalar now.
    ///
    /// Capture executes nothing, so the state is not advanced and the caches
    /// are not written. It does have to run on a stream of its own — capture
    /// on the legacy default stream is rejected by the driver — so build the
    /// engine with `CudaContext::new_stream`.
    ///
    /// The bounds check that [`GatedAttentionBlock::forward`] makes against
    /// `max_seq` happens here, at the captured position, and cannot happen
    /// again inside a replay. [`Self::replay_step`] makes it itself.
    pub fn capture_step(
        &mut self,
        stream: &Arc<CudaStream>,
        state: &mut SequenceState,
    ) -> Result<StepGraph, ForwardError> {
        if self.profile.is_some() {
            return Err(ForwardError::CaptureWhileProfiling);
        }
        // A captured pass must bind the scalar rotary path: the graph
        // replays whatever kernel and buffer it recorded, and decode never
        // uses per-token rotary positions.
        assert!(
            !self.mrope_active,
            "capture with mrope active would bake the per-token rotary path into a decode graph"
        );
        assert!(
            self.staged_rows.is_empty(),
            "capture with staged image rows would bake an injection into a decode graph"
        );
        // The capture must be told about the position and the token count
        // before it starts, or the calls that publish them are recorded --
        // and a copy from pageable host memory is not something a graph may
        // contain.
        self.publish_inputs(stream, state, &vec![0i32; self.tokens])?;
        // Single-stream decode wants one block per split: at N=1 the 72
        // splits are already one thin wave and fewer blocks just halve the
        // memory parallelism (measured -13% at 32K). Reset here so a batch
        // capture that ran earlier on these shared kernels cannot leak its
        // narrower launch into this capture. Bit-identical either way.
        for block in &mut self.attention {
            block.set_decode_mma_blocks(0);
        }
        // Workers capture concurrently on three dedicated runtime threads.
        // Global mode lets unrelated CUDA work in either sibling thread
        // invalidate this stream's capture; thread-local mode scopes that
        // restriction to the worker that owns this context and stream.
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        // Whatever happens, the capture has to be closed before the error is
        // returned, or the stream stays in capture mode and every later launch
        // on it fails.
        let recorded = self
            .body(stream, state, &mut |_, _, _| {})
            .and_then(|()| self.launch_argmax(stream));
        // The only flag the driver accepts here without a stream parameter.
        // It concerns memory nodes the graph owns, and this graph allocates
        // nothing, so it is inert. `UPLOAD` and `USE_NODE_PRIORITY` were both
        // tried and both return `CUDA_ERROR_INVALID_VALUE` through
        // `cuGraphInstantiateWithFlags`.
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        recorded?;
        let graph = graph?.ok_or(ForwardError::EmptyCapture)?;
        graph.upload()?;
        Ok(StepGraph { graph })
    }

    /// Launch a captured step at the state's current position and return the
    /// sampled token id.
    ///
    /// The host work is three things: two small copies in, one graph launch,
    /// four bytes out.
    pub fn replay_step(
        &mut self,
        stream: &Arc<CudaStream>,
        state: &mut SequenceState,
        graph: &StepGraph,
        token_ids: &[i32],
    ) -> Result<i32, ForwardError> {
        if token_ids.len() != self.tokens {
            return Err(ForwardError::WrongTokenCount {
                expected: self.tokens,
                got: token_ids.len(),
            });
        }
        // The check `GatedAttentionBlock::forward` makes on every ordinary
        // pass. A replay never enters that function, so this is the only thing
        // standing between a too-long sequence and a cache overrun.
        if state.position() + self.tokens > state.max_seq() {
            return Err(ForwardError::CacheExhausted {
                position: state.position(),
                tokens: self.tokens,
                max_seq: state.max_seq(),
            });
        }
        self.publish_inputs(stream, state, token_ids)?;
        graph.graph.launch()?;
        let id = self.read_sampled(stream)?;
        state.advance(self.tokens);
        Ok(id)
    }

    /// Drop the repacked int8 weights, forcing every projection back to fp32.
    ///
    /// Exists so a differential test can run the *same shape* both ways and
    /// compare: the tensor-core path changes the numerics (activations are
    /// quantized to int8), and the golden capture is 19 tokens — below the
    /// threshold where that path engages — so the oracle gate alone would
    /// never exercise it. See `tests/int8_forward.rs`.
    ///
    /// Drops this shape's int8 handles. Shared attention repacks remain owned
    /// by the model so sibling shapes keep their independent choice; this
    /// shape cannot re-enable them without being rebuilt.
    ///
    /// Clears **both** mixers. Clearing only the Gated DeltaNet side would
    /// leave the ten Gated Attention layers on tensor cores in the supposed
    /// fp32 twin, and the differential test would silently stop covering them.
    pub fn disable_tensor_cores(&mut self) {
        self.gdn_int8 = Arc::new(Vec::new());
        for block in &mut self.attention {
            block.disable_tensor_cores();
        }
        self.moe.disable_tensor_cores();
    }

    /// Force decode off `attn_flash_decode_mma_wpo{4,2}` and back onto
    /// `attn_flash_decode_warp`'s per-key online softmax, in every attention
    /// block. Exists for `bench_decode`'s A/B lever; see
    /// `AttentionKernels::disable_decode_mma`.
    pub fn disable_decode_mma(&mut self) {
        for block in &mut self.attention {
            block.disable_decode_mma();
        }
    }

    /// Select which occupancy width the tensor-core decode kernel uses, in
    /// every attention block, if it is not disabled. `wpo` must be 2 or 4.
    /// Exists for `bench_decode`'s A/B lever.
    pub fn set_decode_mma_wpo(&mut self, wpo: usize) {
        for block in &mut self.attention {
            block.set_decode_mma_wpo(wpo);
        }
    }

    /// Whether this pass has the repacked int8 weights resident, in both
    /// mixers. A pass with one side repacked and not the other is a bug, so
    /// this reports the conjunction rather than either half.
    pub fn tensor_cores_enabled(&self) -> bool {
        !self.gdn_int8.is_empty()
            && self.attention.iter().all(|b| b.tensor_cores_enabled())
            && self.moe.tensor_cores_enabled()
    }

    /// Allocate the extra LM head and argmax scratch [`Self::run_batch_decode`]
    /// needs, sized for this pass's `tokens` rows.
    ///
    /// Idempotent, and free the second time. Split out of
    /// `run_batch_decode` rather than done on its first call because that
    /// call allocates, and `AGENTS.md` rule 6 makes an allocation the caller's
    /// explicit decision on a hot path rather than something a forward pass
    /// does for them. This pass must have been built for the batch width
    /// (`tokens`) [`Self::run_batch_decode`] will be called with; a pass
    /// built for one token can enable this too, but then serves batches of
    /// exactly one, same as [`Self::run`].
    pub fn enable_batch_decode(
        &mut self,
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
    ) -> Result<(), ForwardError> {
        if self.batch_lm_head.is_some() {
            return Ok(());
        }
        let tokens = self.tokens;
        self.batch_lm_head = Some(LmHeadKernels::new(
            ctx,
            LmHeadGeometry {
                hidden: self.hidden,
                vocab: self.vocab,
                max_tokens: tokens,
            },
        )?);
        self.batch_logits = Some(stream.alloc_zeros::<f32>(tokens * self.vocab)?);
        self.batch_argmax_values = Some(stream.alloc_zeros::<f32>(tokens * ARGMAX_BLOCKS)?);
        self.batch_argmax_indices = Some(stream.alloc_zeros::<i32>(tokens * ARGMAX_BLOCKS)?);
        self.batch_argmax_out = Some(stream.alloc_zeros::<i32>(tokens)?);
        self.batch_positions = Some(stream.alloc_zeros::<i32>(2 * tokens)?);
        self.batch_positions_host = vec![0i32; 2 * tokens];
        self.batch_zero_tokens = vec![0i32; tokens];
        let concurrent_attention = std::env::var_os("LLMXABE_SERIAL_BATCH_ATTENTION").is_none();
        let lanes = if concurrent_attention {
            tokens.saturating_sub(1)
        } else {
            0
        };
        let mut secondary = Vec::with_capacity(lanes);
        let mut joins = Vec::with_capacity(tokens.saturating_sub(1));
        for _ in 0..lanes {
            secondary.push(ctx.new_stream()?);
            joins.push(ctx.new_event(None)?);
        }
        self.batch_attention = Some(BatchAttentionStreams {
            secondary,
            fork: ctx.new_event(None)?,
            joins,
        });
        Ok(())
    }

    /// Allocate the fixed output and position scratch for equal-chunk batched
    /// prefill. `self.tokens` is the total physical row count; `sequences`
    /// names how those sequence-major rows are partitioned.
    ///
    /// This is separate from [`Self::enable_batch_decode`] because decode's
    /// output width equals `self.tokens`, while prefill keeps only one last
    /// row per sequence. Allocation remains an explicit cold-path operation.
    pub fn enable_batch_prefill(
        &mut self,
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        sequences: usize,
    ) -> Result<(), ForwardError> {
        if sequences == 0 || !self.tokens.is_multiple_of(sequences) {
            return Err(ForwardError::BatchWidth {
                expected: self.tokens,
                got: sequences,
                what: "a divisor number of sequences",
            });
        }
        if self.batch_lm_head.is_some() {
            let output_rows = self.batch_argmax_out.as_ref().map_or(0, CudaSlice::len);
            if output_rows != sequences {
                return Err(ForwardError::BatchWidth {
                    expected: output_rows,
                    got: sequences,
                    what: "sequences",
                });
            }
            return Ok(());
        }
        self.batch_lm_head = Some(LmHeadKernels::new(
            ctx,
            LmHeadGeometry {
                hidden: self.hidden,
                vocab: self.vocab,
                max_tokens: sequences,
            },
        )?);
        self.batch_logits = Some(stream.alloc_zeros::<f32>(sequences * self.vocab)?);
        self.batch_argmax_values = Some(stream.alloc_zeros::<f32>(sequences * ARGMAX_BLOCKS)?);
        self.batch_argmax_indices = Some(stream.alloc_zeros::<i32>(sequences * ARGMAX_BLOCKS)?);
        self.batch_argmax_out = Some(stream.alloc_zeros::<i32>(sequences)?);
        self.batch_positions = Some(stream.alloc_zeros::<i32>(2 * sequences)?);
        self.batch_positions_host = vec![0i32; 2 * sequences];
        self.batch_zero_tokens = vec![0i32; self.tokens];
        Ok(())
    }

    /// Batched decode: advance `states.len()` independent sequences by one
    /// token each, in a single pass, and return one sampled id per sequence
    /// in `states` order.
    ///
    /// # What this amortizes, and what it does not
    ///
    /// This pass (`self`) must have been built with `tokens == states.len()`
    /// — see the module docs on why a pass is a fixed width — and
    /// [`Self::enable_batch_decode`] must already have been called. Every
    /// weight-bound stage in it runs once over the whole `[states.len()]`
    /// batch: the embedding gather, [`crate::block::gdn::GdnBlock`]'s
    /// projections (see
    /// [`GdnBlock::forward_batch_decode`](crate::block::gdn::GdnBlock::forward_batch_decode)
    /// for why its two genuinely stateful steps still loop), the MoE
    /// dispatch, the final norm and the LM head. That is the whole win: a
    /// weight is read once for `states.len()` sequences' next token instead
    /// of once per sequence.
    ///
    /// Gated Attention batches the same way, through
    /// [`GatedAttentionBlock::forward_batch_decode`]: its four Q8_0
    /// projections, its norms and its output gate run once over the whole
    /// batch, and only rotary, the key/value append and the causal read loop
    /// per sequence — those three read a position or a sequence's own cache,
    /// which nothing here amortizes across sequences (see that method's
    /// docs). `self.attention` and `self.attn_scratch` are this pass's own,
    /// built for `tokens = states.len()` at construction like every other
    /// per-layer field here, so there is no second `Forward` to keep in step
    /// with this one.
    ///
    /// The per-sequence loops that remain — inside `GatedAttentionBlock` and
    /// inside `GdnBlock` — issue `states.len()` small launches rather than
    /// one, on the host. That cost is real and is not hidden by this
    /// function; [`Self::capture_batch_step`] amortizes it the same way
    /// [`Self::capture_step`] does for a single sequence.
    pub fn run_batch_decode(
        &mut self,
        stream: &Arc<CudaStream>,
        states: &mut [SequenceState],
        token_ids: &[i32],
    ) -> Result<Vec<i32>, ForwardError> {
        self.check_batch_shape(states, token_ids)?;
        self.publish_batch_inputs(stream, states, token_ids)?;
        self.body_batch_decode(stream, states, &mut |_, _, _| {})?;
        let host = self.read_batch_sampled(stream)?;
        for state in states.iter_mut() {
            state.advance(1);
        }
        Ok(host)
    }

    /// Prefill equal-length chunks for independent sequences in one physical
    /// pass. Token ids are sequence-major and must contain exactly
    /// `self.tokens` rows. Weight-bound stages see the flattened row axis;
    /// recurrent state, rotary positions and KV caches remain per sequence.
    pub fn run_batch_prefill(
        &mut self,
        stream: &Arc<CudaStream>,
        states: &mut [SequenceState],
        token_ids: &[i32],
    ) -> Result<Vec<i32>, ForwardError> {
        self.run_batch_prefill_tagged(stream, states, token_ids, &mut |_, _, _| {})
    }

    fn run_batch_prefill_tagged(
        &mut self,
        stream: &Arc<CudaStream>,
        states: &mut [SequenceState],
        token_ids: &[i32],
        on_waypoint: &mut impl FnMut(Option<u32>, WaypointStage, &CudaSlice<f32>),
    ) -> Result<Vec<i32>, ForwardError> {
        let n = states.len();
        if n == 0 || !self.tokens.is_multiple_of(n) {
            return Err(ForwardError::BatchWidth {
                expected: self.tokens,
                got: n,
                what: "a divisor number of states",
            });
        }
        if token_ids.len() != self.tokens {
            return Err(ForwardError::WrongTokenCount {
                expected: self.tokens,
                got: token_ids.len(),
            });
        }
        let chunk_tokens = self.tokens / n;
        let (gdn_layers, attn_layers) = (self.gdn_weights.len(), self.attention.len());
        for state in states.iter() {
            if state.gdn_layers() != gdn_layers || state.attention_layers() != attn_layers {
                return Err(ForwardError::StateShape {
                    expected_gdn: gdn_layers,
                    expected_attention: attn_layers,
                    got_gdn: state.gdn_layers(),
                    got_attention: state.attention_layers(),
                });
            }
            if state.position() + chunk_tokens > state.max_seq() {
                return Err(ForwardError::CacheExhausted {
                    position: state.position(),
                    tokens: chunk_tokens,
                    max_seq: state.max_seq(),
                });
            }
        }
        if self.batch_lm_head.is_none()
            || self.batch_argmax_out.as_ref().map_or(0, |x| x.len()) != n
        {
            return Err(ForwardError::BatchPrefillNotEnabled);
        }

        stream.memcpy_htod(token_ids, &mut self.d_tokens)?;
        self.moe.publish_tokens(stream, self.tokens)?;
        for state in states.iter_mut() {
            state
                .publish_position(stream)
                .map_err(ForwardError::State)?;
        }

        self.embed(stream)?;
        on_waypoint(None, WaypointStage::Embed, &self.hidden_state);
        let (mut gdn_slot, mut attn_slot) = (0usize, 0usize);
        for layer in 0..self.config.num_layers {
            match self.config.layer_kind(layer) {
                LayerKind::GatedDeltaNet => {
                    let mut gdn_states: SmallVec<[&mut GdnState; 3]> =
                        states.iter_mut().map(|s| s.gdn_mut(gdn_slot)).collect();
                    self.gdn.forward_batch_prefill(
                        stream,
                        &self.gdn_weights[gdn_slot],
                        self.gdn_int8.get(gdn_slot),
                        &mut gdn_states,
                        chunk_tokens,
                        &self.hidden_state,
                        &mut self.mixer_out,
                    )?;
                    gdn_slot += 1;
                }
                LayerKind::GatedAttention => {
                    let mut caches: SmallVec<[&mut KvCache; 3]> = SmallVec::new();
                    let mut offsets: SmallVec<[usize; 3]> = SmallVec::new();
                    let mut positions: SmallVec<[&CudaSlice<i32>; 3]> = SmallVec::new();
                    let mut rope_positions: SmallVec<[&CudaSlice<i32>; 3]> = SmallVec::new();
                    for state in states.iter_mut() {
                        offsets.push(state.position());
                        let (cache, position, rope) = state.kv_and_positions_mut(attn_slot);
                        caches.push(cache);
                        positions.push(position);
                        rope_positions.push(rope);
                    }
                    self.attention[attn_slot].forward_batch_prefill(
                        stream,
                        &mut self.attn_scratch,
                        &self.hidden_state,
                        &mut caches,
                        chunk_tokens,
                        &offsets,
                        &positions,
                        &rope_positions,
                        &mut self.mixer_out,
                    )?;
                    attn_slot += 1;
                }
            }
            on_waypoint(Some(layer), WaypointStage::Mixer, &self.mixer_out);
            self.moe.forward(
                stream,
                &self.moe_weights[layer as usize],
                &self.mixer_out,
                self.tokens,
                &mut self.ffn_out,
                &mut self.hidden_state,
            )?;
            on_waypoint(Some(layer), WaypointStage::Moe, &self.hidden_state);
        }

        self.layer_ops.rms_norm(
            stream,
            &self.hidden_state,
            &self.w_output_norm,
            &mut self.final_norm,
            self.tokens,
            self.hidden,
            self.rms_eps,
        )?;
        for seq in 0..n {
            let source = unsafe {
                crate::viewslice::subslice(
                    stream,
                    &self.final_norm,
                    (seq * chunk_tokens + chunk_tokens - 1) * self.hidden,
                    self.hidden,
                )
            };
            let mut destination = unsafe {
                crate::viewslice::subslice(stream, &self.mixer_out, seq * self.hidden, self.hidden)
            };
            stream.memcpy_dtod(&*source, &mut *destination)?;
        }
        let lm_head = self.batch_lm_head.as_ref().expect("checked above");
        let last_rows =
            unsafe { crate::viewslice::subslice(stream, &self.mixer_out, 0, n * self.hidden) };
        lm_head.forward(
            stream,
            QuantTensor {
                bytes: &self.w_lm_head,
                quant: ExpertQuant::Q8_0,
            },
            &last_rows,
            n,
            self.batch_logits.as_mut().expect("checked above"),
        )?;
        self.launch_batch_argmax(stream, n)?;
        let sampled = self.read_batch_sampled(stream)?;
        for state in states {
            state.advance(chunk_tokens);
        }
        Ok(sampled)
    }

    /// Diagnostic-only sibling of [`Self::run_batch_prefill`]: the same
    /// flattened chunk, with `on_waypoint` called after the embedding gather
    /// ([`WaypointStage::Embed`]), after each layer's mixer
    /// ([`WaypointStage::Mixer`]) and after each layer's MoE
    /// ([`WaypointStage::Moe`]).
    ///
    /// Written for the wide flattened-prefill divergence — identical prompts
    /// produced batch rows differing by 2.3e-1 from each other — which needs
    /// the first diverging layer and family, and no gate exposes the
    /// per-stage buffers.
    pub fn run_batch_prefill_with_stage_waypoints(
        &mut self,
        stream: &Arc<CudaStream>,
        states: &mut [SequenceState],
        token_ids: &[i32],
        mut on_waypoint: impl FnMut(Option<u32>, WaypointStage, &CudaSlice<f32>),
    ) -> Result<Vec<i32>, ForwardError> {
        self.run_batch_prefill_tagged(stream, states, token_ids, &mut on_waypoint)
    }

    /// Diagnostic-only sibling of [`Self::run_batch_decode`]: the same
    /// batched step, with `on_waypoint` called after the embedding gather
    /// ([`WaypointStage::Embed`]), after each layer's mixer
    /// ([`WaypointStage::Mixer`]) and after each layer's MoE
    /// ([`WaypointStage::Moe`]).
    ///
    /// Built for the batch-vs-single-stream divergence audit, which needs the
    /// first layer and family at which the two disagree. It drives the same
    /// `Self::body_batch_decode` the ordinary path and the graph capture do,
    /// so what it measures is what the engine runs.
    pub fn run_batch_decode_with_stage_waypoints(
        &mut self,
        stream: &Arc<CudaStream>,
        states: &mut [SequenceState],
        token_ids: &[i32],
        mut on_waypoint: impl FnMut(Option<u32>, WaypointStage, &CudaSlice<f32>),
    ) -> Result<Vec<i32>, ForwardError> {
        self.check_batch_shape(states, token_ids)?;
        self.publish_batch_inputs(stream, states, token_ids)?;
        self.body_batch_decode(stream, states, &mut on_waypoint)?;
        let host = self.read_batch_sampled(stream)?;
        for state in states.iter_mut() {
            state.advance(1);
        }
        Ok(host)
    }

    /// Record one batched decode step — embedding, forty blocks (Gated
    /// DeltaNet batched, Gated Attention looped per sequence), the LM head
    /// and every sequence's argmax — as a CUDA graph, for
    /// [`Self::replay_batch_step`] to launch at every later step.
    ///
    /// This is [`Self::capture_step`]'s reasoning, one level up: batched
    /// decode issues `states.len()` small launches per layer for the two
    /// steps `GdnBlock::forward_batch_decode` cannot batch and for the three
    /// steps `GatedAttentionBlock::forward_batch_decode` cannot, on top of
    /// the calls that are already batched. `bench_decode_batch` shows what
    /// that costs uncaptured. A graph is the same sequence of launches
    /// submitted as one object, the same trade `capture_step` already makes
    /// for a single sequence's ~1,100 operations, at `states.len()` times the
    /// operation count.
    ///
    /// The same constraints apply, for the same reasons: nothing in
    /// `Self::body_batch_decode` may take a value that changes between
    /// steps as a host argument, which is why every sequence's position lives
    /// in that [`SequenceState`]'s own device scalar and is read from there,
    /// not passed in. `self` and `states` must be the exact objects passed to
    /// every later [`Self::replay_batch_step`] — the graph holds their device
    /// pointers, not a description of the computation.
    pub fn capture_batch_step(
        &mut self,
        stream: &Arc<CudaStream>,
        states: &mut [SequenceState],
    ) -> Result<BatchStepGraph, ForwardError> {
        if self.profile.is_some() {
            return Err(ForwardError::CaptureWhileProfiling);
        }
        // A captured pass must bind the scalar rotary path: the graph
        // replays whatever kernel and buffer it recorded, and decode never
        // uses per-token rotary positions.
        assert!(
            !self.mrope_active,
            "capture with mrope active would bake the per-token rotary path into a decode graph"
        );
        assert!(
            self.staged_rows.is_empty(),
            "capture with staged image rows would bake an injection into a decode graph"
        );
        let n = self.tokens;
        self.check_batch_shape(states, &self.batch_zero_tokens)?;
        // As `capture_step`: the capture must already know about the token
        // ids and every sequence's position before it starts, or the copies
        // that publish them would be recorded into the graph, and a copy
        // from pageable host memory cannot be.
        stream.memcpy_htod(&self.batch_zero_tokens, &mut self.d_tokens)?;
        self.moe.publish_tokens(stream, n)?;
        for (dst, state) in self
            .batch_positions_host
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(states.iter())
        {
            let pos = state.position() as i32;
            dst[0] = pos;
            dst[1] = pos + state.rope_delta();
        }
        stream.memcpy_htod(
            &self.batch_positions_host,
            self.batch_positions
                .as_mut()
                .expect("enabled batch decode has positions"),
        )?;
        // Each worker owns its stream on a dedicated OS thread, and all three
        // workers may lazily capture a new batch width at the same time.
        // Three or more concurrent sequence-local decode-attention calls
        // overflow the 288 resident-block budget at one block per split, so
        // the batch capture bakes 48 blocks per call: at the 96 logical
        // splits the kernels now default to, each block computes exactly two
        // splits, the three calls' 288 blocks fill one wave, and the
        // partials are bit-identical at any block count — the block choice
        // is scheduling, not numerics (the split count is numerics, gated
        // separately; see MMA_DECODE_SPLITS). 96/48 measured 151.2/153.2
        // vs the prior 72/36's 149.5/148.3 at 32K N=3
        // (docs/BENCHMARKS.md 2026-08-20). Below three sequences the
        // default already fits one wave and stays.
        let batch_blocks = if n >= 3 { 48 } else { 0 };
        for block in &mut self.attention {
            block.set_decode_mma_blocks(batch_blocks);
        }
        // Global capture mode makes those independent contexts interfere.
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let recorded = self.body_batch_decode(stream, states, &mut |_, _, _| {});
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        recorded?;
        let graph = graph?.ok_or(ForwardError::EmptyCapture)?;
        graph.upload()?;
        Ok(BatchStepGraph { graph })
    }

    /// Launch a captured batched step and return one sampled id per sequence,
    /// in `states` order.
    ///
    /// `states` must be the exact states [`Self::capture_batch_step`] was
    /// called with, in the same order; see [`BatchStepGraph`].
    pub fn replay_batch_step(
        &mut self,
        stream: &Arc<CudaStream>,
        states: &mut [SequenceState],
        graph: &BatchStepGraph,
        token_ids: &[i32],
    ) -> Result<Vec<i32>, ForwardError> {
        let mut output = Vec::with_capacity(self.tokens);
        self.replay_batch_step_into(stream, states, graph, token_ids, &mut output)?;
        Ok(output)
    }

    /// Allocation-free sibling of [`Self::replay_batch_step`].
    ///
    /// `output` must already have room for the fixed batch width. Serving
    /// runtimes reserve it at startup and reuse it across decode rounds.
    pub fn replay_batch_step_into(
        &mut self,
        stream: &Arc<CudaStream>,
        states: &mut [SequenceState],
        graph: &BatchStepGraph,
        token_ids: &[i32],
        output: &mut Vec<i32>,
    ) -> Result<(), ForwardError> {
        let n = self.tokens;
        if token_ids.len() != n {
            return Err(ForwardError::BatchWidth {
                expected: n,
                got: token_ids.len(),
                what: "token_ids",
            });
        }
        for state in states.iter() {
            if state.position() + 1 > state.max_seq() {
                return Err(ForwardError::CacheExhausted {
                    position: state.position(),
                    tokens: 1,
                    max_seq: state.max_seq(),
                });
            }
        }
        self.publish_batch_inputs(stream, states, token_ids)?;
        graph.graph.launch()?;
        if output.capacity() < n {
            return Err(ForwardError::BatchWidth {
                expected: n,
                got: output.capacity(),
                what: "sample output capacity",
            });
        }
        output.resize(n, 0);
        self.read_batch_sampled_into(stream, output)?;
        for state in states.iter_mut() {
            state.advance(1);
        }
        Ok(())
    }

    /// The shape checks [`Self::run_batch_decode`] and
    /// [`Self::capture_batch_step`] both make, once.
    fn check_batch_shape(
        &self,
        states: &[SequenceState],
        token_ids: &[i32],
    ) -> Result<(), ForwardError> {
        let n = self.tokens;
        if states.len() != n {
            return Err(ForwardError::BatchWidth {
                expected: n,
                got: states.len(),
                what: "states",
            });
        }
        if token_ids.len() != n {
            return Err(ForwardError::BatchWidth {
                expected: n,
                got: token_ids.len(),
                what: "token_ids",
            });
        }
        let (gdn_layers, attn_layers) = (self.gdn_weights.len(), self.attention.len());
        for state in states {
            if state.gdn_layers() != gdn_layers || state.attention_layers() != attn_layers {
                return Err(ForwardError::StateShape {
                    expected_gdn: gdn_layers,
                    expected_attention: attn_layers,
                    got_gdn: state.gdn_layers(),
                    got_attention: state.attention_layers(),
                });
            }
            if state.position() + 1 > state.max_seq() {
                return Err(ForwardError::CacheExhausted {
                    position: state.position(),
                    tokens: 1,
                    max_seq: state.max_seq(),
                });
            }
        }
        if self.batch_lm_head.is_none() {
            return Err(ForwardError::BatchDecodeNotEnabled);
        }
        Ok(())
    }

    /// The batched decode step's host-touching half: the token ids and every
    /// sequence's own position. Split out for the same reason
    /// [`Self::publish_inputs`] is — a copy from pageable host memory cannot
    /// be recorded into a CUDA graph, so this runs before capture and again
    /// before every replay.
    ///
    /// Every sequence's position is copied in **one** call rather than
    /// `states.len()` — `SequenceState::publish_position` writes each
    /// state's own single-element device scalar, and looping that per state
    /// was `states.len()` separate host-to-device copies where one array
    /// does the same job, which measurably mattered at small batch widths
    /// where `states.len()` extra host round trips were a bigger fraction of
    /// a decode step than the compute they were standing in front of. See
    /// `docs/BENCHMARKS.md`. This writes
    /// [`Self::batch_positions`], not any state's own `d_position` — the
    /// batch-decode path never reads the latter.
    fn publish_batch_inputs(
        &mut self,
        stream: &Arc<CudaStream>,
        states: &[SequenceState],
        token_ids: &[i32],
    ) -> Result<(), ForwardError> {
        let n = self.tokens;
        stream.memcpy_htod(token_ids, &mut self.d_tokens)?;
        self.moe.publish_tokens(stream, n)?;
        for (dst, state) in self
            .batch_positions_host
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(states)
        {
            let pos = state.position() as i32;
            dst[0] = pos;
            dst[1] = pos + state.rope_delta();
        }
        let positions = self
            .batch_positions
            .as_mut()
            .expect("checked by the caller");
        stream.memcpy_htod(&self.batch_positions_host, positions)?;
        Ok(())
    }

    /// Embedding through every sequence's argmax: launches only.
    ///
    /// Nothing in here reads host memory, allocates, or synchronizes, which
    /// is what makes it capturable in [`Self::capture_batch_step`]. The
    /// per-sequence argmax launches are included, the same way
    /// [`Self::capture_step`] folds [`Self::launch_argmax`] into its captured
    /// region — only the four-byte-per-sequence read-back in
    /// [`Self::read_batch_sampled`] needs the host and stays outside.
    fn body_batch_decode(
        &mut self,
        stream: &Arc<CudaStream>,
        states: &mut [SequenceState],
        on_waypoint: &mut impl FnMut(Option<u32>, WaypointStage, &CudaSlice<f32>),
    ) -> Result<(), ForwardError> {
        let n = self.tokens;
        self.embed(stream)?;
        on_waypoint(None, WaypointStage::Embed, &self.hidden_state);

        let (mut gdn_slot, mut attn_slot) = (0usize, 0usize);
        for layer in 0..self.config.num_layers {
            match self.config.layer_kind(layer) {
                LayerKind::GatedDeltaNet => {
                    let mut gdn_states: SmallVec<[&mut GdnState; 3]> =
                        states.iter_mut().map(|st| st.gdn_mut(gdn_slot)).collect();
                    self.gdn
                        .forward_batch_decode(
                            stream,
                            &self.gdn_weights[gdn_slot],
                            self.gdn_int8.get(gdn_slot),
                            &mut gdn_states,
                            &self.hidden_state,
                            &mut self.mixer_out,
                        )
                        .map_err(ForwardError::Gdn)?;
                    gdn_slot += 1;
                }
                LayerKind::GatedAttention => {
                    // Every sequence's own cache, collected once so
                    // `forward_batch_decode` can batch the weight-bound
                    // steps and loop only the three that read a position or
                    // a cache. `self.attention` and `self.attn_scratch` are
                    // this pass's own -- built for `tokens = n` at
                    // construction, same as every other per-layer field
                    // here -- so there is no second `Forward` to keep in
                    // step with this one any more.
                    //
                    // Positions come from `self.batch_positions`, published
                    // once for all `n` sequences by `publish_batch_inputs`,
                    // not from any state's own `d_position` -- the
                    // batch-decode path never touches that field.
                    let batch_positions = self
                        .batch_positions
                        .as_ref()
                        .expect("published by publish_batch_inputs");
                    let mut caches: SmallVec<[&mut KvCache; 3]> = SmallVec::new();
                    let mut pos_offsets: SmallVec<[usize; 3]> = SmallVec::new();
                    let mut position_views: SmallVec<[_; 3]> = SmallVec::new();
                    let mut rope_views: SmallVec<[_; 3]> = SmallVec::new();
                    for (i, state) in states.iter_mut().enumerate() {
                        pos_offsets.push(state.position());
                        let (cache, _) = state.kv_and_position_mut(attn_slot);
                        caches.push(cache);
                        // SAFETY: `i < n` and `batch_positions` holds
                        // exactly `2 * n` elements — interleaved
                        // [slot, rope] pairs — allocated by
                        // `enable_batch_decode`.
                        position_views.push(unsafe {
                            crate::viewslice::subslice(stream, batch_positions, 2 * i, 1)
                        });
                        rope_views.push(unsafe {
                            crate::viewslice::subslice(stream, batch_positions, 2 * i + 1, 1)
                        });
                    }
                    let positions: SmallVec<[&CudaSlice<i32>; 3]> =
                        position_views.iter().map(|v| &**v).collect();
                    let rope_positions: SmallVec<[&CudaSlice<i32>; 3]> =
                        rope_views.iter().map(|v| &**v).collect();
                    self.attention[attn_slot]
                        .forward_batch_decode(
                            stream,
                            &self
                                .batch_attention
                                .as_ref()
                                .expect("enabled batch decode has attention streams")
                                .secondary,
                            &self
                                .batch_attention
                                .as_ref()
                                .expect("enabled batch decode has attention streams")
                                .fork,
                            &self
                                .batch_attention
                                .as_ref()
                                .expect("enabled batch decode has attention streams")
                                .joins,
                            &mut self.attn_scratch,
                            &self.hidden_state,
                            &mut caches,
                            &pos_offsets,
                            &positions,
                            &rope_positions,
                            &mut self.mixer_out,
                        )
                        .map_err(ForwardError::Attention)?;
                    attn_slot += 1;
                }
            }
            on_waypoint(Some(layer), WaypointStage::Mixer, &self.mixer_out);

            self.moe
                .forward(
                    stream,
                    &self.moe_weights[layer as usize],
                    &self.mixer_out,
                    n,
                    &mut self.ffn_out,
                    &mut self.hidden_state,
                )
                .map_err(ForwardError::Moe)?;
            on_waypoint(Some(layer), WaypointStage::Moe, &self.hidden_state);
        }

        self.layer_ops.rms_norm(
            stream,
            &self.hidden_state,
            &self.w_output_norm,
            &mut self.final_norm,
            n,
            self.hidden,
            self.rms_eps,
        )?;

        let vocab = self.vocab;
        {
            let lm_head = self.batch_lm_head.as_ref().expect("checked by the caller");
            let logits = self.batch_logits.as_mut().expect("checked by the caller");
            lm_head
                .forward(
                    stream,
                    QuantTensor {
                        bytes: &self.w_lm_head,
                        quant: ExpertQuant::Q8_0,
                    },
                    &self.final_norm,
                    n,
                    logits,
                )
                .map_err(ForwardError::LmHead)?;
        }

        // The argmax kernels are geometry-free -- they take `vocab` as a
        // plain argument -- so `self.lm_head`'s copies serve a batch of any
        // width; there is nothing `batch_lm_head`-specific to build for them.
        for i in 0..n {
            // SAFETY: `i < n`; `batch_logits` is `n * vocab`,
            // `batch_argmax_values`/`indices` are `n * ARGMAX_BLOCKS`, and
            // `batch_argmax_out` is `n`, all allocated by
            // `enable_batch_decode` to exactly those widths.
            let row = unsafe {
                crate::viewslice::subslice(
                    stream,
                    self.batch_logits.as_ref().expect("checked by the caller"),
                    i * vocab,
                    vocab,
                )
            };
            let mut values = unsafe {
                crate::viewslice::subslice(
                    stream,
                    self.batch_argmax_values
                        .as_ref()
                        .expect("checked by the caller"),
                    i * ARGMAX_BLOCKS,
                    ARGMAX_BLOCKS,
                )
            };
            let mut indices = unsafe {
                crate::viewslice::subslice(
                    stream,
                    self.batch_argmax_indices
                        .as_ref()
                        .expect("checked by the caller"),
                    i * ARGMAX_BLOCKS,
                    ARGMAX_BLOCKS,
                )
            };
            let mut out_one = unsafe {
                crate::viewslice::subslice(
                    stream,
                    self.batch_argmax_out
                        .as_ref()
                        .expect("checked by the caller"),
                    i,
                    1,
                )
            };
            self.lm_head
                .argmax(stream, &row, vocab, &mut values, &mut indices, &mut out_one)?;
        }
        Ok(())
    }

    /// Read back the last `Self::body_batch_decode`'s sampled ids.
    ///
    /// Synchronizes, for the reason [`Self::read_sampled`] does: a step is
    /// not finished until the host knows what to feed back in for every
    /// sequence.
    fn read_batch_sampled(&self, stream: &Arc<CudaStream>) -> Result<Vec<i32>, ForwardError> {
        let host = stream.clone_dtoh(
            self.batch_argmax_out
                .as_ref()
                .expect("checked by the caller"),
        )?;
        stream.synchronize()?;
        Ok(host)
    }

    fn launch_batch_argmax(
        &mut self,
        stream: &Arc<CudaStream>,
        rows: usize,
    ) -> Result<(), ForwardError> {
        let vocab = self.vocab;
        for i in 0..rows {
            let row = unsafe {
                crate::viewslice::subslice(
                    stream,
                    self.batch_logits.as_ref().expect("batch output enabled"),
                    i * vocab,
                    vocab,
                )
            };
            let mut values = unsafe {
                crate::viewslice::subslice(
                    stream,
                    self.batch_argmax_values
                        .as_ref()
                        .expect("batch output enabled"),
                    i * ARGMAX_BLOCKS,
                    ARGMAX_BLOCKS,
                )
            };
            let mut indices = unsafe {
                crate::viewslice::subslice(
                    stream,
                    self.batch_argmax_indices
                        .as_ref()
                        .expect("batch output enabled"),
                    i * ARGMAX_BLOCKS,
                    ARGMAX_BLOCKS,
                )
            };
            let mut output = unsafe {
                crate::viewslice::subslice(
                    stream,
                    self.batch_argmax_out
                        .as_ref()
                        .expect("batch output enabled"),
                    i,
                    1,
                )
            };
            self.lm_head
                .argmax(stream, &row, vocab, &mut values, &mut indices, &mut output)?;
        }
        Ok(())
    }

    fn read_batch_sampled_into(
        &self,
        stream: &Arc<CudaStream>,
        host: &mut [i32],
    ) -> Result<(), ForwardError> {
        stream.memcpy_dtoh(
            self.batch_argmax_out
                .as_ref()
                .expect("checked by the caller"),
            host,
        )?;
        stream.synchronize()?;
        Ok(())
    }

    /// Allocate the scratch [`Self::run_verify`] needs: one
    /// [`GdnVerifyScratch`] per Gated DeltaNet layer, sized for this pass's
    /// `tokens` (the verify window, `1 + d`). Idempotent, like
    /// [`Self::enable_batch_decode`], and for the same reason —
    /// `AGENTS.md` rule 6 makes allocating on `run_verify`'s call a bug
    /// rather than a convenience.
    pub fn enable_verify(&mut self, stream: &Arc<CudaStream>) -> Result<(), ForwardError> {
        if self.verify_scratch.is_some() {
            return Ok(());
        }
        // A verify window's rows must land exactly where plain decode's
        // would, and above `MOE_NARROW_DECODE_MAX` tokens the MoE's default
        // selection moves to tiled/tensor-core kernels that are not
        // bit-identical per row (`tests/batch_verify.rs` caught the flip).
        // Pin this pass's MoE to the flat decode regime for its lifetime —
        // a Forward with verify enabled is only ever used to verify.
        self.moe.set_exact_decode_regime(true);
        let geometry = self.gdn.geometry();
        let mut scratch = Vec::with_capacity(self.gdn_weights.len());
        for _ in &self.gdn_weights {
            scratch.push(GdnVerifyScratch::new(stream, &geometry, self.tokens)?);
        }
        self.verify_scratch = Some(scratch);
        Ok(())
    }

    /// A speculative-decode verify step: `token_ids` (`[id_last, draft_1,
    /// ..., draft_d]`, length `self.tokens = 1 + d`) run through this pass's
    /// full `1 + d`-position causal window in a single weight-read pass, with
    /// every Gated DeltaNet layer's recurrent state snapshotted at every
    /// position boundary along the way.
    ///
    /// Returns one greedy-argmax id per window position — row `i`'s id is
    /// the target model's own next-token prediction *after* having seen
    /// `token_ids[0..=i]`, which is what the caller compares against
    /// `token_ids[1..]` to find the accepted prefix.
    ///
    /// **Leaves `state` exactly where a plain [`Self::run`] over the whole
    /// window would** — every Gated DeltaNet layer is fully advanced, not
    /// rolled back — and does **not** call `state.advance()` at all. Rolling
    /// back to an accepted count `k` is the caller's job, once it has
    /// compared this call's return value against the drafts and decided `k`:
    /// call [`GdnSnapshotRing::commit`] with `i = k` on every ring in
    /// `rings`, then `state.advance(k)`. The attention key/value caches need
    /// no separate rollback — a rejected tail's cache entries sit at
    /// positions `state.position() + k ..` and are simply never read again
    /// until a later pass overwrites them at exactly those positions; see
    /// `crate::state`'s module docs on why a reset does not clear the cache.
    ///
    /// [`Self::enable_verify`] and [`Self::enable_batch_decode`] must both
    /// already have been called — the first allocates the per-layer
    /// snapshot scratch, the second the wide LM head and per-row argmax this
    /// method reuses unchanged (a verify window's positions are no
    /// different, to that tail, from batched decode's per-sequence rows: in
    /// both cases it is `tokens` independent hidden-state rows in, `tokens`
    /// logit rows and argmax ids out).
    ///
    /// `rings` must hold one [`GdnSnapshotRing`] of depth `self.tokens + 1`
    /// per Gated DeltaNet layer, in layer order — the caller's, because they
    /// are per-*sequence* state (like [`SequenceState`] itself) and this
    /// pass is shared across sequences.
    ///
    /// Gated DeltaNet projections take the split-layout tiled path when
    /// `gdn_int8` is resident — the same pairing one-token decode uses —
    /// so a verify window's first FMA agrees with the chain of
    /// [`Self::run`] calls the identity test compares against.
    pub fn run_verify(
        &mut self,
        stream: &Arc<CudaStream>,
        state: &mut SequenceState,
        rings: &mut [GdnSnapshotRing],
        token_ids: &[i32],
    ) -> Result<Vec<i32>, ForwardError> {
        if token_ids.len() != self.tokens {
            return Err(ForwardError::WrongTokenCount {
                expected: self.tokens,
                got: token_ids.len(),
            });
        }
        if self.batch_lm_head.is_none() {
            return Err(ForwardError::BatchDecodeNotEnabled);
        }
        let Some(mut scratch) = self.verify_scratch.take() else {
            return Err(ForwardError::BatchWidth {
                expected: self.gdn_weights.len(),
                got: 0,
                what: "verify scratch (call enable_verify first)",
            });
        };
        if rings.len() != self.gdn_weights.len() || scratch.len() != self.gdn_weights.len() {
            let got = rings.len().min(scratch.len());
            self.verify_scratch = Some(scratch);
            return Err(ForwardError::BatchWidth {
                expected: self.gdn_weights.len(),
                got,
                what: "gdn snapshot rings/scratch",
            });
        }

        self.publish_inputs(stream, state, token_ids)?;
        self.embed(stream)?;

        let (mut gdn_slot, mut attn_slot) = (0usize, 0usize);
        for layer in 0..self.config.num_layers {
            let result = match self.config.layer_kind(layer) {
                LayerKind::GatedDeltaNet => {
                    let r = run_layer_with_snapshots(
                        stream,
                        &mut self.gdn,
                        &self.gdn_weights[gdn_slot],
                        self.gdn_int8.get(gdn_slot),
                        state.gdn_mut(gdn_slot),
                        &self.hidden_state,
                        &mut self.mixer_out,
                        &mut scratch[gdn_slot],
                        &mut rings[gdn_slot],
                        self.tokens,
                    )
                    .map_err(ForwardError::from);
                    gdn_slot += 1;
                    r
                }
                LayerKind::GatedAttention => {
                    let pos_offset = state.position();
                    let (cache, positions, rope_scalar) = state.kv_and_positions_mut(attn_slot);
                    let rope = if self.mrope_active {
                        crate::block::attention::RopeSource::PerToken(&self.d_mrope)
                    } else {
                        crate::block::attention::RopeSource::Scalar(rope_scalar)
                    };
                    let r = self.attention[attn_slot]
                        .forward(
                            stream,
                            &mut self.attn_scratch,
                            &self.hidden_state,
                            cache,
                            pos_offset,
                            positions,
                            rope,
                            &mut self.mixer_out,
                        )
                        .map_err(ForwardError::from);
                    attn_slot += 1;
                    r
                }
            };
            if let Err(e) = result {
                self.verify_scratch = Some(scratch);
                return Err(e);
            }

            if let Err(e) = self.moe.forward(
                stream,
                &self.moe_weights[layer as usize],
                &self.mixer_out,
                self.tokens,
                &mut self.ffn_out,
                &mut self.hidden_state,
            ) {
                self.verify_scratch = Some(scratch);
                return Err(ForwardError::from(e));
            }
        }
        self.verify_scratch = Some(scratch);

        self.layer_ops.rms_norm(
            stream,
            &self.hidden_state,
            &self.w_output_norm,
            &mut self.final_norm,
            self.tokens,
            self.hidden,
            self.rms_eps,
        )?;

        let vocab = self.vocab;
        let n = self.tokens;
        {
            let lm_head = self.batch_lm_head.as_ref().expect("checked above");
            let logits = self.batch_logits.as_mut().expect("checked above");
            lm_head
                .forward(
                    stream,
                    QuantTensor {
                        bytes: &self.w_lm_head,
                        quant: ExpertQuant::Q8_0,
                    },
                    &self.final_norm,
                    n,
                    logits,
                )
                .map_err(ForwardError::LmHead)?;
        }
        for i in 0..n {
            // SAFETY: as `body_batch_decode`'s identical tail — `i < n`,
            // and `batch_logits`/`batch_argmax_values`/`batch_argmax_indices`/
            // `batch_argmax_out` are all sized from `enable_batch_decode` to
            // exactly `n` (times `vocab` or `ARGMAX_BLOCKS` where relevant).
            let row = unsafe {
                crate::viewslice::subslice(
                    stream,
                    self.batch_logits.as_ref().expect("checked above"),
                    i * vocab,
                    vocab,
                )
            };
            let mut values = unsafe {
                crate::viewslice::subslice(
                    stream,
                    self.batch_argmax_values.as_ref().expect("checked above"),
                    i * ARGMAX_BLOCKS,
                    ARGMAX_BLOCKS,
                )
            };
            let mut indices = unsafe {
                crate::viewslice::subslice(
                    stream,
                    self.batch_argmax_indices.as_ref().expect("checked above"),
                    i * ARGMAX_BLOCKS,
                    ARGMAX_BLOCKS,
                )
            };
            let mut out_one = unsafe {
                crate::viewslice::subslice(
                    stream,
                    self.batch_argmax_out.as_ref().expect("checked above"),
                    i,
                    1,
                )
            };
            self.lm_head
                .argmax(stream, &row, vocab, &mut values, &mut indices, &mut out_one)
                .map_err(ForwardError::LmHead)?;
        }
        self.read_batch_sampled(stream)
    }

    /// One [`GdnSnapshotRing`] per Gated DeltaNet layer for a verify window
    /// of `window` positions, in layer order — the per-sequence ring set
    /// [`Self::run_verify`] and [`Self::run_batch_verify`] take.
    ///
    /// Allocates, so this is a cold-path call: serving runtimes build one
    /// set per concurrent sequence at startup or admission, never per step.
    pub fn new_verify_rings(
        &self,
        stream: &Arc<CudaStream>,
        window: usize,
    ) -> Result<Vec<GdnSnapshotRing>, ForwardError> {
        let geometry = self.gdn.geometry();
        let mut rings = Vec::with_capacity(self.gdn_weights.len());
        for _ in &self.gdn_weights {
            rings.push(GdnSnapshotRing::new(stream, &geometry, window + 1)?);
        }
        Ok(rings)
    }

    /// The batched sibling of [`Self::run_verify`]: `states.len()`
    /// independent sequences, each contributing a window of `self.tokens /
    /// states.len()` positions (`[id_last, draft_1, ..., draft_d]`,
    /// sequence-major in `token_ids`), verified in **one** weight-read pass.
    ///
    /// Returns one greedy-argmax id per row, sequence-major: row
    /// `s * window + i` is the target's own next-token prediction after
    /// having seen sequence `s`'s `window[0..=i]`. The caller compares each
    /// sequence's rows `0..d` against its drafts to find that sequence's
    /// accepted prefix `k`, then commits it with
    /// [`Self::commit_verify_window`] at `positions = k + 1`. Like
    /// [`Self::run_verify`], this advances **nothing** itself, and the
    /// attention key/value caches need no rollback — a rejected tail's
    /// entries are overwritten in place by a later pass at the same
    /// positions.
    ///
    /// Weight-bound stages — the embedding gather, the Gated DeltaNet
    /// projections (via [`run_layer_with_snapshots_batch`]), Gated
    /// Attention's four projections (via
    /// [`GatedAttentionBlock::forward_batch_prefill`]), the MoE, the final
    /// norm and the LM head — each run once over all `states.len() *
    /// window` rows. That is the point: one pass's weight traffic verifies
    /// every sequence's whole window, where the serving decode loop paid
    /// one full pass per emitted token.
    ///
    /// [`Self::enable_verify`] and [`Self::enable_batch_decode`] must both
    /// have been called on this pass — the first sizes the per-layer
    /// snapshot scratch to `self.tokens`, the second the wide LM head and
    /// the per-row argmax used here unchanged. `rings[s]` must hold one
    /// ring of depth `window + 1` per Gated DeltaNet layer, in layer order,
    /// for sequence `s` (see [`Self::new_verify_rings`]).
    pub fn run_batch_verify(
        &mut self,
        stream: &Arc<CudaStream>,
        states: &mut [SequenceState],
        rings: &mut [Vec<GdnSnapshotRing>],
        token_ids: &[i32],
    ) -> Result<Vec<i32>, ForwardError> {
        self.run_batch_verify_tagged(stream, states, rings, token_ids, &mut |_, _, _| {})
    }

    /// Diagnostic-only sibling of [`Self::run_batch_verify`], with
    /// `on_waypoint` fired after the embedding gather and after each
    /// layer's mixer and MoE — the same probes the batch-decode and
    /// batch-prefill siblings expose, for the same reason: a cross-sequence
    /// divergence needs the first layer and family at which two sequences'
    /// rows disagree, and no public surface exposes the per-stage buffers.
    pub fn run_batch_verify_with_stage_waypoints(
        &mut self,
        stream: &Arc<CudaStream>,
        states: &mut [SequenceState],
        rings: &mut [Vec<GdnSnapshotRing>],
        token_ids: &[i32],
        mut on_waypoint: impl FnMut(Option<u32>, WaypointStage, &CudaSlice<f32>),
    ) -> Result<Vec<i32>, ForwardError> {
        self.run_batch_verify_tagged(stream, states, rings, token_ids, &mut on_waypoint)
    }

    fn run_batch_verify_tagged(
        &mut self,
        stream: &Arc<CudaStream>,
        states: &mut [SequenceState],
        rings: &mut [Vec<GdnSnapshotRing>],
        token_ids: &[i32],
        on_waypoint: &mut impl FnMut(Option<u32>, WaypointStage, &CudaSlice<f32>),
    ) -> Result<Vec<i32>, ForwardError> {
        let n = states.len();
        if n == 0 || !self.tokens.is_multiple_of(n) {
            return Err(ForwardError::BatchWidth {
                expected: self.tokens,
                got: n,
                what: "a divisor number of states",
            });
        }
        let window = self.tokens / n;
        if token_ids.len() != self.tokens {
            return Err(ForwardError::WrongTokenCount {
                expected: self.tokens,
                got: token_ids.len(),
            });
        }
        let (gdn_layers, attn_layers) = (self.gdn_weights.len(), self.attention.len());
        for state in states.iter() {
            if state.gdn_layers() != gdn_layers || state.attention_layers() != attn_layers {
                return Err(ForwardError::StateShape {
                    expected_gdn: gdn_layers,
                    expected_attention: attn_layers,
                    got_gdn: state.gdn_layers(),
                    got_attention: state.attention_layers(),
                });
            }
            if state.position() + window > state.max_seq() {
                return Err(ForwardError::CacheExhausted {
                    position: state.position(),
                    tokens: window,
                    max_seq: state.max_seq(),
                });
            }
        }
        // The per-row argmax needs `enable_batch_decode`'s sizing (one row
        // per token), not `enable_batch_prefill`'s (one row per sequence).
        if self.batch_lm_head.is_none()
            || self.batch_argmax_out.as_ref().map_or(0, CudaSlice::len) != self.tokens
        {
            return Err(ForwardError::BatchDecodeNotEnabled);
        }
        let Some(mut scratch) = self.verify_scratch.take() else {
            return Err(ForwardError::BatchWidth {
                expected: gdn_layers,
                got: 0,
                what: "verify scratch (call enable_verify first)",
            });
        };
        if rings.len() != n
            || rings.iter().any(|set| set.len() != gdn_layers)
            || scratch.len() != gdn_layers
        {
            self.verify_scratch = Some(scratch);
            return Err(ForwardError::BatchWidth {
                expected: gdn_layers,
                got: rings.iter().map(Vec::len).min().unwrap_or(0),
                what: "per-sequence gdn snapshot ring sets",
            });
        }

        // The publish half, as `run_batch_prefill`: token ids once, every
        // sequence's position from its own device scalar.
        stream.memcpy_htod(token_ids, &mut self.d_tokens)?;
        self.moe.publish_tokens(stream, self.tokens)?;
        for state in states.iter_mut() {
            state
                .publish_position(stream)
                .map_err(ForwardError::State)?;
        }

        self.embed(stream)?;
        on_waypoint(None, WaypointStage::Embed, &self.hidden_state);

        let (mut gdn_slot, mut attn_slot) = (0usize, 0usize);
        for layer in 0..self.config.num_layers {
            let result = match self.config.layer_kind(layer) {
                LayerKind::GatedDeltaNet => {
                    let mut gdn_states: SmallVec<[&mut GdnState; 3]> =
                        states.iter_mut().map(|s| s.gdn_mut(gdn_slot)).collect();
                    let mut layer_rings: SmallVec<[&mut GdnSnapshotRing; 3]> =
                        rings.iter_mut().map(|set| &mut set[gdn_slot]).collect();
                    let r = run_layer_with_snapshots_batch(
                        stream,
                        &mut self.gdn,
                        &self.gdn_weights[gdn_slot],
                        self.gdn_int8.get(gdn_slot),
                        &mut gdn_states,
                        &self.hidden_state,
                        &mut self.mixer_out,
                        &mut scratch[gdn_slot],
                        &mut layer_rings,
                        window,
                    )
                    .map_err(ForwardError::from);
                    gdn_slot += 1;
                    r
                }
                LayerKind::GatedAttention => {
                    let mut caches: SmallVec<[&mut KvCache; 3]> = SmallVec::new();
                    let mut offsets: SmallVec<[usize; 3]> = SmallVec::new();
                    let mut positions: SmallVec<[&CudaSlice<i32>; 3]> = SmallVec::new();
                    let mut rope_positions: SmallVec<[&CudaSlice<i32>; 3]> = SmallVec::new();
                    for state in states.iter_mut() {
                        offsets.push(state.position());
                        let (cache, position, rope) = state.kv_and_positions_mut(attn_slot);
                        caches.push(cache);
                        positions.push(position);
                        rope_positions.push(rope);
                    }
                    let r = self.attention[attn_slot]
                        .forward_batch_prefill(
                            stream,
                            &mut self.attn_scratch,
                            &self.hidden_state,
                            &mut caches,
                            window,
                            &offsets,
                            &positions,
                            &rope_positions,
                            &mut self.mixer_out,
                        )
                        .map_err(ForwardError::from);
                    attn_slot += 1;
                    r
                }
            };
            if let Err(e) = result {
                self.verify_scratch = Some(scratch);
                return Err(e);
            }
            on_waypoint(Some(layer), WaypointStage::Mixer, &self.mixer_out);

            if let Err(e) = self.moe.forward(
                stream,
                &self.moe_weights[layer as usize],
                &self.mixer_out,
                self.tokens,
                &mut self.ffn_out,
                &mut self.hidden_state,
            ) {
                self.verify_scratch = Some(scratch);
                return Err(ForwardError::from(e));
            }
            on_waypoint(Some(layer), WaypointStage::Moe, &self.hidden_state);
        }
        self.verify_scratch = Some(scratch);

        self.layer_ops.rms_norm(
            stream,
            &self.hidden_state,
            &self.w_output_norm,
            &mut self.final_norm,
            self.tokens,
            self.hidden,
            self.rms_eps,
        )?;

        let vocab = self.vocab;
        let rows = self.tokens;
        {
            let lm_head = self.batch_lm_head.as_ref().expect("checked above");
            let logits = self.batch_logits.as_mut().expect("checked above");
            lm_head
                .forward(
                    stream,
                    QuantTensor {
                        bytes: &self.w_lm_head,
                        quant: ExpertQuant::Q8_0,
                    },
                    &self.final_norm,
                    rows,
                    logits,
                )
                .map_err(ForwardError::LmHead)?;
        }
        for i in 0..rows {
            // SAFETY: as `run_verify`'s identical tail — `i < rows`, and the
            // batch logits/argmax buffers are all sized by
            // `enable_batch_decode` to exactly `rows` (times `vocab` or
            // `ARGMAX_BLOCKS` where relevant).
            let row = unsafe {
                crate::viewslice::subslice(
                    stream,
                    self.batch_logits.as_ref().expect("checked above"),
                    i * vocab,
                    vocab,
                )
            };
            let mut values = unsafe {
                crate::viewslice::subslice(
                    stream,
                    self.batch_argmax_values.as_ref().expect("checked above"),
                    i * ARGMAX_BLOCKS,
                    ARGMAX_BLOCKS,
                )
            };
            let mut indices = unsafe {
                crate::viewslice::subslice(
                    stream,
                    self.batch_argmax_indices.as_ref().expect("checked above"),
                    i * ARGMAX_BLOCKS,
                    ARGMAX_BLOCKS,
                )
            };
            let mut out_one = unsafe {
                crate::viewslice::subslice(
                    stream,
                    self.batch_argmax_out.as_ref().expect("checked above"),
                    i,
                    1,
                )
            };
            self.lm_head
                .argmax(stream, &row, vocab, &mut values, &mut indices, &mut out_one)
                .map_err(ForwardError::LmHead)?;
        }
        self.read_batch_sampled(stream)
    }

    /// Commit a verify step's outcome for one sequence: restore every Gated
    /// DeltaNet layer's state to the snapshot at `positions` window
    /// positions folded in, and advance the sequence by the same count.
    ///
    /// `positions` is the number of window rows that became real — the
    /// accepted draft count plus one for the window's leading `id_last`
    /// (`k + 1` in [`Self::run_verify`]'s and [`Self::run_batch_verify`]'s
    /// terms). The attention key/value caches need no touch-up; see
    /// [`Self::run_verify`]'s docs.
    pub fn commit_verify_window(
        &self,
        stream: &Arc<CudaStream>,
        state: &mut SequenceState,
        rings: &[GdnSnapshotRing],
        positions: usize,
    ) -> Result<(), ForwardError> {
        if rings.len() != self.gdn_weights.len() {
            return Err(ForwardError::BatchWidth {
                expected: self.gdn_weights.len(),
                got: rings.len(),
                what: "gdn snapshot rings",
            });
        }
        for (slot, ring) in rings.iter().enumerate() {
            ring.commit(stream, positions, state.gdn_mut(slot))?;
        }
        state.advance(positions);
        Ok(())
    }

    /// Allocate carried state for one sequence of up to `max_seq` positions.
    ///
    /// The state is separate from the pass because a pass is built for a fixed
    /// token count and a sequence is not: prefill and decode are two `Forward`
    /// objects of different shapes over one [`SequenceState`]. See that type's
    /// module docs.
    pub fn new_state(
        &self,
        stream: &Arc<CudaStream>,
        max_seq: usize,
    ) -> Result<SequenceState, ForwardError> {
        SequenceState::new(stream, &self.gdn, &self.config, max_seq).map_err(ForwardError::State)
    }

    /// Run the whole pass over `token_ids`, continuing `state`.
    ///
    /// `on_waypoint` is called with `(None, embedding)` once and then with
    /// `(Some(N), l_out_N)` after each of the 40 blocks, before the buffer is
    /// reused. That is what makes a wrong pass bisectable against the capture
    /// in one run rather than 40: a block whose output diverges while its
    /// input matched is the block at fault.
    ///
    /// The callback runs on the host, so it must synchronise if it reads.
    ///
    /// # Where this pass starts
    ///
    /// At `state.position()`. The attention blocks append their keys and
    /// values there and rotate by it, and the Gated DeltaNet blocks fold into
    /// whatever matrix the state already holds. So:
    ///
    /// - A fresh state, or one that has been [`SequenceState::reset`], gives a
    ///   **cold prefill**: position 0, zeroed recurrent state. That is a pure
    ///   function of `token_ids`, and it is what the oracle captured
    ///   (`state_predelta-N` is all zeros for every captured block), so it is
    ///   what a comparison against the capture requires.
    /// - A state left where a previous call finished gives a **continuation**,
    ///   which is what decode is.
    ///
    /// The distinction used to be made here, by zeroing unconditionally. It is
    /// the caller's now, because the whole difference between prefill and
    /// decode is which of the two they want, and a pass that always resets can
    /// only ever be the first. Forgetting to reset is not silent: a second
    /// cold prefill through a used state resumes the previous prompt's
    /// convolution taps and recurrent matrix, and produces a completely
    /// different, entirely finite result.
    pub fn run(
        &mut self,
        stream: &Arc<CudaStream>,
        state: &mut SequenceState,
        token_ids: &[i32],
        mut on_waypoint: impl FnMut(Option<u32>, &CudaSlice<f32>),
    ) -> Result<(), ForwardError> {
        // The mixer probe is [`Self::run_with_stage_waypoints`]'s alone; this
        // callback keeps the two it always had.
        self.run_tagged(stream, state, token_ids, &mut |layer, stage, buf| {
            if stage != WaypointStage::Mixer {
                on_waypoint(layer, buf);
            }
        })
    }

    fn run_tagged(
        &mut self,
        stream: &Arc<CudaStream>,
        state: &mut SequenceState,
        token_ids: &[i32],
        on_waypoint: &mut impl FnMut(Option<u32>, WaypointStage, &CudaSlice<f32>),
    ) -> Result<(), ForwardError> {
        if token_ids.len() != self.tokens {
            return Err(ForwardError::WrongTokenCount {
                expected: self.tokens,
                got: token_ids.len(),
            });
        }
        // A state built for a different model would index the wrong cache in
        // the middle of the pass, so it is checked once at the edge instead.
        let (gdn_layers, attn_layers) = (self.gdn_weights.len(), self.attention.len());
        if state.gdn_layers() != gdn_layers || state.attention_layers() != attn_layers {
            return Err(ForwardError::StateShape {
                expected_gdn: gdn_layers,
                expected_attention: attn_layers,
                got_gdn: state.gdn_layers(),
                got_attention: state.attention_layers(),
            });
        }

        if let Some(p) = &mut self.profile {
            p.begin(stream)?;
        }

        self.publish_inputs(stream, state, token_ids)?;
        self.body(stream, state, on_waypoint)?;
        // Last, and only on success: every block has now appended, so the
        // state's claim about how many positions its caches hold is true.
        // Advancing earlier would leave a failed pass claiming positions that
        // no cache was written for, and the next call would read them.
        state.advance(self.tokens);
        Ok(())
    }

    /// Diagnostic-only sibling of [`Self::run`]: the same pass, with
    /// `on_waypoint` also called right after each layer's mixer (Gated
    /// DeltaNet or Gated Attention), tagged [`WaypointStage::Mixer`], in
    /// addition to the post-embed and post-MoE calls `run` makes.
    ///
    /// Built for the batch-vs-single-stream divergence audit: telling "the
    /// mixer already disagreed" from "the mixer agreed and MoE introduced the
    /// disagreement" needs the buffer `run`'s callback never exposes. It runs
    /// the same `Self::body` the golden test and `capture_step` do, so an
    /// audit cannot measure a path the engine does not take.
    pub fn run_with_stage_waypoints(
        &mut self,
        stream: &Arc<CudaStream>,
        state: &mut SequenceState,
        token_ids: &[i32],
        mut on_waypoint: impl FnMut(Option<u32>, WaypointStage, &CudaSlice<f32>),
    ) -> Result<(), ForwardError> {
        self.run_tagged(stream, state, token_ids, &mut on_waypoint)
    }

    /// Publish `(t, h, w)` rotary triples for the next (single-sequence,
    /// uncaptured) pass, which will rotate by them instead of the scalar.
    ///
    /// `triples` holds exactly `3 * tokens` values in token order. The flag
    /// sticks until [`Self::clear_mrope`], so a caller that publishes for an
    /// image chunk must clear before the next text chunk.
    pub fn publish_mrope(
        &mut self,
        stream: &Arc<CudaStream>,
        triples: &[i32],
    ) -> Result<(), ForwardError> {
        if triples.len() != 3 * self.tokens {
            return Err(ForwardError::WrongTokenCount {
                expected: 3 * self.tokens,
                got: triples.len(),
            });
        }
        stream.memcpy_htod(triples, &mut self.d_mrope)?;
        self.mrope_active = true;
        Ok(())
    }

    /// Return the next pass to scalar rotary positions.
    pub fn clear_mrope(&mut self) {
        self.mrope_active = false;
    }

    /// Pre-allocate the image-row staging buffer (`tokens * hidden` f32).
    ///
    /// Called once at load when vision serving is enabled, per AGENTS.md
    /// rule 6 — [`Self::stage_image_rows`] on a pass without this is an
    /// error, never a lazy allocation.
    pub fn enable_image_injection(&mut self, stream: &Arc<CudaStream>) -> Result<(), ForwardError> {
        if self.image_stage.is_none() {
            self.image_stage = Some(stream.alloc_zeros::<f32>(self.tokens * self.hidden)?);
        }
        Ok(())
    }

    /// Queue `n_rows` embedding rows for injection at chunk token
    /// `dst_token` on the next pass, copying them device-to-device from
    /// `src` starting at its row `src_row` (rows are `hidden` floats).
    ///
    /// The copy into staging happens now; the copy into the residual
    /// stream happens inside [`Self::body`], after the embedding gather
    /// has written the placeholder rows this replaces.
    pub fn stage_image_rows(
        &mut self,
        stream: &Arc<CudaStream>,
        src: &CudaSlice<f32>,
        src_row: usize,
        dst_token: usize,
        n_rows: usize,
    ) -> Result<(), ForwardError> {
        let stage = self
            .image_stage
            .as_mut()
            .ok_or(ForwardError::ImageInjectionNotEnabled)?;
        if dst_token + n_rows > self.tokens
            || self.stage_used + n_rows > self.tokens
            || (src_row + n_rows) * self.hidden > src.len()
        {
            return Err(ForwardError::WrongTokenCount {
                expected: self.tokens,
                got: dst_token + n_rows,
            });
        }
        let h = self.hidden;
        // SAFETY: both views are inside their buffers by the checks above.
        let src_view = unsafe { crate::viewslice::subslice(stream, src, src_row * h, n_rows * h) };
        let mut dst_view =
            unsafe { crate::viewslice::subslice(stream, stage, self.stage_used * h, n_rows * h) };
        stream.memcpy_dtod(&*src_view, &mut *dst_view)?;
        self.staged_rows.push((dst_token, self.stage_used, n_rows));
        self.stage_used += n_rows;
        Ok(())
    }

    /// Drop any queued image rows without running them.
    pub fn clear_staged_image_rows(&mut self) {
        self.staged_rows.clear();
        self.stage_used = 0;
    }

    /// The two per-step host inputs: the token ids and the position.
    ///
    /// Split out of `Self::body` because these are the only two operations
    /// in a pass that read host memory, and a CUDA graph capture cannot
    /// contain a copy from a pageable host pointer. They run before the graph
    /// launches instead, on the same stream, which orders them ahead of
    /// everything the graph does.
    fn publish_inputs(
        &mut self,
        stream: &Arc<CudaStream>,
        state: &mut SequenceState,
        token_ids: &[i32],
    ) -> Result<(), ForwardError> {
        stream.memcpy_htod(token_ids, &mut self.d_tokens)?;
        state
            .publish_position(stream)
            .map_err(ForwardError::State)?;
        self.moe.publish_tokens(stream, self.tokens)?;
        Ok(())
    }

    /// Everything from the embedding gather to the LM head: launches only.
    ///
    /// Nothing in here reads host memory, allocates, or synchronizes, and
    /// nothing takes the sequence position as an argument — that is what makes
    /// it capturable as a CUDA graph and replayable at every later position.
    ///
    /// `on_waypoint` fires at all three [`WaypointStage`]s. Every caller
    /// supplies one — the graph capture a no-op, [`Forward::run`] one that
    /// drops the mixer probe — so there is one loop rather than a diagnostic
    /// copy of it that can drift from what the engine runs.
    fn body(
        &mut self,
        stream: &Arc<CudaStream>,
        state: &mut SequenceState,
        on_waypoint: &mut impl FnMut(Option<u32>, WaypointStage, &CudaSlice<f32>),
    ) -> Result<(), ForwardError> {
        self.mark(stream, Stage::Reset)?;
        self.embed(stream)?;
        // Image spans: overwrite the placeholder rows the gather just wrote
        // with the projector's embeddings. Empty on every decode step and on
        // every captured pass (asserted at capture), so the text path takes
        // no branch and records no copy.
        if !self.staged_rows.is_empty() {
            let stage = self
                .image_stage
                .as_ref()
                .expect("staged rows imply an enabled stage");
            let h = self.hidden;
            for &(dst_token, stage_row, n_rows) in &self.staged_rows {
                // SAFETY: bounds were checked when the rows were staged.
                let src =
                    unsafe { crate::viewslice::subslice(stream, stage, stage_row * h, n_rows * h) };
                let mut dst = unsafe {
                    crate::viewslice::subslice(
                        stream,
                        &self.hidden_state,
                        dst_token * h,
                        n_rows * h,
                    )
                };
                stream.memcpy_dtod(&*src, &mut *dst)?;
            }
        }
        self.mark(stream, Stage::Embed)?;
        on_waypoint(None, WaypointStage::Embed, &self.hidden_state);

        let (mut gdn_slot, mut attn_slot) = (0usize, 0usize);
        for layer in 0..self.config.num_layers {
            // 1. the mixer, into `attn_residual-N`.
            let kind = self.config.layer_kind(layer);
            match kind {
                LayerKind::GatedDeltaNet => {
                    self.gdn.forward(
                        stream,
                        &self.gdn_weights[gdn_slot],
                        self.gdn_int8.get(gdn_slot),
                        state.gdn_mut(gdn_slot),
                        &self.hidden_state,
                        &mut self.mixer_out,
                    )?;
                    gdn_slot += 1;
                }
                LayerKind::GatedAttention => {
                    let pos_offset = state.position();
                    let (cache, positions, rope_scalar) = state.kv_and_positions_mut(attn_slot);
                    let rope = if self.mrope_active {
                        crate::block::attention::RopeSource::PerToken(&self.d_mrope)
                    } else {
                        crate::block::attention::RopeSource::Scalar(rope_scalar)
                    };
                    self.attention[attn_slot].forward(
                        stream,
                        &mut self.attn_scratch,
                        &self.hidden_state,
                        cache,
                        pos_offset,
                        positions,
                        rope,
                        &mut self.mixer_out,
                    )?;
                    attn_slot += 1;
                }
            }
            self.mark(stream, Stage::Mixer { layer, kind })?;
            on_waypoint(Some(layer), WaypointStage::Mixer, &self.mixer_out);

            // 2. the MoE, whose residual is the mixer output and whose
            //    `l_out` becomes the next block's input.
            self.moe.forward(
                stream,
                &self.moe_weights[layer as usize],
                &self.mixer_out,
                self.tokens,
                &mut self.ffn_out,
                &mut self.hidden_state,
            )?;
            self.mark(stream, Stage::Moe { layer })?;
            on_waypoint(Some(layer), WaypointStage::Moe, &self.hidden_state);
        }

        // 3. `h_nextn`: the final norm, all positions.
        self.layer_ops.rms_norm(
            stream,
            &self.hidden_state,
            &self.w_output_norm,
            &mut self.final_norm,
            self.tokens,
            self.hidden,
            self.rms_eps,
        )?;
        self.mark(stream, Stage::FinalNorm)?;

        // 4. `result_output`: llama.cpp's `get_rows(cur, inp_out_ids)` keeps
        //    only the positions it was asked for, which for a prefill is the
        //    last. Selecting it here rather than running the 540 MB head over
        //    all 19 is the same arithmetic and 19x less bandwidth.
        let last = (self.tokens - 1) * self.hidden;
        let row = self.final_norm.slice(last..last + self.hidden);
        stream.memcpy_dtod(&row, &mut self.last_hidden)?;
        self.lm_head.forward(
            stream,
            QuantTensor {
                bytes: &self.w_lm_head,
                quant: ExpertQuant::Q8_0,
            },
            &self.last_hidden,
            1,
            &mut self.logits,
        )?;
        self.mark(stream, Stage::LmHead)?;
        Ok(())
    }

    /// `get_rows(token_embd.weight, ids)` into the residual stream.
    fn embed(&mut self, stream: &Arc<CudaStream>) -> Result<(), ForwardError> {
        let cfg = LaunchConfig {
            grid_dim: (self.tokens as u32, 1, 1),
            block_dim: (EMBED_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let hidden_i32 = self.hidden as i32;
        let tokens_i32 = self.tokens as i32;
        let mut builder = stream.launch_builder(&self.embed_fn);
        builder
            .arg(&*self.w_token_embd)
            .arg(&self.d_tokens)
            .arg(&mut self.hidden_state)
            .arg(&hidden_i32)
            .arg(&tokens_i32);
        // SAFETY: one block per token slot, returning above `n_tokens`; the
        // output holds `tokens * hidden` floats and the in-row loop is bounded
        // by `hidden`. The table was checked to be Q8_0 of `vocab * hidden`
        // elements at construction, so `ids[t] * (hidden/32) * 34 + hidden/32
        // * 34 - 1` is its last byte for any `ids[t] < vocab` — which the
        // vocabulary itself bounds, and a token id outside it is the caller's
        // to reject.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }
}

/// Alias one resident Q8_0 tensor, rejecting any other stored format.
fn alias_q8_0(
    weights: &DeviceWeights,
    stream: &Arc<CudaStream>,
    role: Role,
    layer: Option<u32>,
) -> Result<ManuallyDrop<CudaSlice<u8>>, ForwardError> {
    let placement = weights
        .find(role, layer)
        .ok_or(ForwardError::MissingWeight { role, layer })?;
    if placement.ggml_type != GgmlType::Q8_0 {
        return Err(ForwardError::WrongQuant {
            role,
            layer,
            found: placement.ggml_type,
            expected: GgmlType::Q8_0,
        });
    }
    let alias = weights
        .bytes_of(stream, role, layer)
        .ok_or(ForwardError::MissingWeight { role, layer })?;
    // SAFETY: the result is sealed in a `ManuallyDrop` that the caller stores
    // in `Forward` and never takes out of, and `Forward` is used only while
    // the `DeviceWeights` it was built from is alive.
    Ok(ManuallyDrop::new(unsafe { alias.into_aliasing_slice() }))
}

/// Alias one resident f32 tensor, rejecting any other stored format.
fn alias_f32(
    weights: &DeviceWeights,
    stream: &Arc<CudaStream>,
    role: Role,
    layer: u32,
) -> Result<CudaSlice<f32>, ForwardError> {
    let placement = weights
        .find(role, Some(layer))
        .ok_or(ForwardError::MissingWeight {
            role,
            layer: Some(layer),
        })?;
    if placement.ggml_type != GgmlType::F32 {
        return Err(ForwardError::WrongQuant {
            role,
            layer: Some(layer),
            found: placement.ggml_type,
            expected: GgmlType::F32,
        });
    }
    let alias = weights
        .f32_of(stream, role, Some(layer))
        .ok_or(ForwardError::MissingWeight {
            role,
            layer: Some(layer),
        })?;
    // SAFETY: every caller stores the result inside a `GdnLayerWeights` held
    // in a `ManuallyDrop` for the life of the `Forward`.
    Ok(unsafe { alias.into_aliasing_slice() })
}

/// One Gated DeltaNet layer's weights, entirely by alias.
///
/// The mirror image of [`GdnLayerWeights::upload`], which reads the same ten
/// roles out of the mapped file and copies each one onto the device. Nothing
/// is copied here: the fields are the arena's own bytes.
fn alias_gdn_layer(
    weights: &DeviceWeights,
    stream: &Arc<CudaStream>,
    layer: u32,
) -> Result<GdnLayerWeights, ForwardError> {
    let q8 = |role: Role| -> Result<CudaSlice<u8>, ForwardError> {
        Ok(ManuallyDrop::into_inner(alias_q8_0(
            weights,
            stream,
            role,
            Some(layer),
        )?))
    };
    Ok(GdnLayerWeights {
        input_norm: alias_f32(weights, stream, Role::InputNorm, layer)?,
        qkv: q8(Role::GdnQkv)?,
        gate: q8(Role::GdnGate)?,
        conv1d: alias_f32(weights, stream, Role::GdnConv1d, layer)?,
        alpha: alias_f32(weights, stream, Role::GdnAlpha, layer)?,
        beta: alias_f32(weights, stream, Role::GdnBeta, layer)?,
        dt_bias: alias_f32(weights, stream, Role::GdnDtBias, layer)?,
        a: alias_f32(weights, stream, Role::GdnA, layer)?,
        ssm_norm: alias_f32(weights, stream, Role::GdnNorm, layer)?,
        out: q8(Role::GdnOut)?,
    })
}

/// Bytes the Gated Attention blocks hold on top of the arena's own copy.
///
/// Summed from the placements rather than from the geometry, so it reports
/// what is really there. `GatedAttentionBlock::new` copies exactly these seven
/// roles per layer.
fn attention_bytes(weights: &DeviceWeights, config: &ModelConfig) -> u64 {
    const COPIED: [Role; 7] = [
        Role::InputNorm,
        Role::AttnQNorm,
        Role::AttnKNorm,
        Role::AttnQGate,
        Role::AttnK,
        Role::AttnV,
        Role::AttnOut,
    ];
    (0..config.num_layers)
        .filter(|&l| config.layer_kind(l) == LayerKind::GatedAttention)
        .flat_map(|l| COPIED.iter().map(move |&r| (r, l)))
        .filter_map(|(role, layer)| weights.find(role, Some(layer)))
        .map(|p| p.alloc.len as u64)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ModelConfig {
        ModelConfig::qwen3_6_35b_a3b()
    }

    #[test]
    fn the_embedding_row_stride_matches_the_kernel() {
        // The kernel spells 32 and 34 as literals because NVRTC has no access
        // to Rust constants. This is the check that they agree with the
        // upstream block layout, and that 2048 input elements is the 2,176
        // byte row `docs/ORACLE.md` section 6.3 dequantized bit-exactly.
        assert_eq!(QK8_0, xabe_cuda::kernels::dequant::QK8_0);
        assert_eq!(
            BLOCK_Q8_0_BYTES,
            xabe_cuda::kernels::dequant::BLOCK_Q8_0_BYTES
        );
        assert_eq!(BLOCK_Q8_0_BYTES, 2 + QK8_0);
        assert_eq!(2048 / QK8_0 * BLOCK_Q8_0_BYTES, 2176);
        assert!(
            EMBED_SRC
                .contains("const unsigned char* row = table + (long long)ids[t] * blocks * 34;")
        );
    }

    #[test]
    fn the_embedding_dequantizes_exactly_and_reads_signed_codes() {
        // `d * q`, one multiply, so the gather is bit-identical to
        // `dequantize_row_q8_0` and `model.input_embed` can be gated on exact
        // equality rather than a tolerance. Reading the codes as unsigned
        // would flip the sign of about half the table while leaving every
        // magnitude plausible.
        assert!(EMBED_SRC.contains("dst[j] = d * (float)q;"));
        assert!(EMBED_SRC.contains("signed char q = (signed char)blk[2 + (j % 32)];"));
    }

    #[test]
    fn the_embedding_grid_comes_from_the_geometry_and_the_count_from_an_argument() {
        // AGENTS.md rule 5. The grid is `max_tokens` blocks and the live count
        // is a guard inside the kernel, so the launch shape does not change
        // with the batch and stays replayable from a captured graph.
        assert!(EMBED_SRC.contains("if (t >= n_tokens) return;"));
    }

    #[test]
    fn the_moe_role_filter_is_exactly_the_moe_blocks_own_tensor_set() {
        // The arena and `MoeLayerWeights` must partition the model between
        // them. A role in neither is missing at runtime; a role in both is
        // 725 MiB per layer of duplicate, which is what this whole arrangement
        // exists to avoid.
        let c = config();
        let mixer_roles = [
            Role::AttnQGate,
            Role::AttnK,
            Role::AttnV,
            Role::AttnQNorm,
            Role::AttnKNorm,
            Role::AttnOut,
            Role::GdnQkv,
            Role::GdnGate,
            Role::GdnConv1d,
            Role::GdnA,
            Role::GdnAlpha,
            Role::GdnBeta,
            Role::GdnDtBias,
            Role::GdnNorm,
            Role::GdnOut,
            Role::InputNorm,
        ];
        for role in mixer_roles {
            assert!(arena_holds(role), "{role} must be in the arena");
        }
        for role in [Role::TokenEmbedding, Role::OutputNorm, Role::LmHead] {
            assert!(arena_holds(role), "{role} must be in the arena");
        }
        for role in MOE_OWNED_ROLES {
            assert!(!arena_holds(role), "{role} must not be in the arena");
        }
        // `PostMixerNorm` is the one that is easy to get wrong: it is a norm,
        // not an expert matrix, and it is read by the MoE block rather than by
        // the mixer, so it belongs on the MoE's side of the split.
        assert!(!arena_holds(Role::PostMixerNorm));
        assert_eq!(c.num_layers, 40);
    }

    #[test]
    fn the_layer_kinds_partition_the_stack_thirty_ten() {
        // The dispatch in `run` walks two slot counters in lockstep with the
        // layer index, so the two block vectors must be exactly as long as the
        // pattern says or a later layer reads an earlier layer's weights.
        let c = config();
        let gdn = (0..c.num_layers)
            .filter(|&l| c.layer_kind(l) == LayerKind::GatedDeltaNet)
            .count();
        let attn = (0..c.num_layers)
            .filter(|&l| c.layer_kind(l) == LayerKind::GatedAttention)
            .count();
        assert_eq!((gdn, attn), (30, 10));
        assert_eq!(gdn + attn, c.num_layers as usize);
        assert_eq!(c.layer_kind(3), LayerKind::GatedAttention);
        assert_eq!(c.layer_kind(39), LayerKind::GatedAttention);
        assert_eq!(c.layer_kind(0), LayerKind::GatedDeltaNet);
    }

    #[test]
    fn the_lm_head_runs_on_one_position_because_that_is_what_was_captured() {
        // `result_output` is `[248320]`, not `[248320, 19]`: llama.cpp selects
        // with `get_rows(cur, inp_out_ids)` between the final norm and the
        // head. Running the head over all 19 would be 19 passes over a 540 MB
        // tensor for 18 answers nothing asks for.
        let g = LmHeadGeometry {
            hidden: config().hidden_size as usize,
            vocab: config().vocab_size as usize,
            max_tokens: 1,
        };
        assert_eq!(g.weight_bytes(), 540_344_320);
        assert_eq!(g.weight_passes(1), 1);
    }

    #[test]
    fn errors_name_the_tensor_and_the_reason() {
        let e = ForwardError::MissingWeight {
            role: Role::LmHead,
            layer: None,
        };
        assert!(e.to_string().contains("output.weight"), "{e}");
        let e = ForwardError::WrongQuant {
            role: Role::GdnQkv,
            layer: Some(4),
            found: GgmlType::Q6K,
            expected: GgmlType::Q8_0,
        };
        let text = e.to_string();
        assert!(text.contains("blk.4."), "{text}");
        assert!(text.contains("q6_K"), "{text}");
        let e = ForwardError::MissingMetadata {
            key: ROPE_FREQ_BASE_KEY,
        };
        assert!(e.to_string().contains("rope.freq_base"));
        let e = ForwardError::WrongTokenCount {
            expected: 19,
            got: 20,
        };
        assert!(e.to_string().contains("19"));
    }
}
