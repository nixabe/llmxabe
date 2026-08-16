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
//! Two things fix it here, and one is not fixed:
//!
//! - **Gated DeltaNet, the embedding, the final norm and the LM head take
//!   [`crate::weights::ResidentTensor`] aliases.** Zero copy: the alias is the
//!   arena's own pointer arithmetic, wrapped back into a `CudaSlice` and
//!   sealed against `Drop`. 30 layers of GDN projections — 1.07 GiB — plus
//!   1.06 GiB of embedding and LM head are read in place.
//! - **The MoE keeps its own copy, and the arena does not.**
//!   [`crate::block::moe::MoeLayerWeights`]'s fields are private and only
//!   `upload` constructs one, so it cannot be given an alias without editing
//!   `block/moe.rs`, which this workstream does not own. Instead the nine
//!   roles it uploads are filtered out of the arena entirely
//!   ([`crate::DeviceWeights::load_where`]), so there is still exactly one
//!   copy of every tensor on the device — it is simply in 40 sets of
//!   allocations rather than in the slab. All 40 layers stay resident and
//!   nothing is re-uploaded per pass.
//! - **Gated Attention still duplicates.** `GatedAttentionBlock::new` copies
//!   through the host at construction and its scratch is private, so its ten
//!   layers cost 289 MiB twice. That one is left standing rather than papered
//!   over: it is bounded, it is measured by
//!   [`ForwardReport::attention_duplicate_bytes`], and the fix is the same
//!   `ResidentTensor` this file already uses for the other two shapes.
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
//! Every projection is fp32 against dequantized weights. llama.cpp quantizes
//! its *activations* to `q8_1` and dots in int8, which is 1,000-10,000x less
//! accurate (`docs/ORACLE.md` section 8 item 0) — so the residual disagreement
//! with the capture is llama.cpp's, it accumulates over 40 blocks, and
//! `tests/forward_pass.rs` reports it as a curve rather than a number.

use std::mem::ManuallyDrop;
use std::sync::Arc;

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig,
    PushKernelArg,
};

use xabe_cuda::kernels::attention::AttentionError;
use xabe_cuda::kernels::compile;
use xabe_cuda::kernels::layer_ops::{LayerOpsError, LayerOpsKernels};
use xabe_cuda::kernels::lm_head::{ARGMAX_BLOCKS, LmHeadError, LmHeadGeometry, LmHeadKernels};
use xabe_cuda::kernels::moe::{ExpertQuant, QuantTensor};
use xabe_gguf::{GgmlType, GgufFile};
use xabe_model::config::{LayerKind, ModelConfig};
use xabe_model::weights::{Directory, Role};

use crate::block::attention::{AttentionBlockError, AttentionKernelSet, GatedAttentionBlock};
use crate::block::gdn::{GdnBlock, GdnBlockError, GdnGeometry, GdnLayerInt8, GdnLayerWeights};
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

/// Grouped-GEMM tile width the MoE dispatch tables pad to.
const MOE_BLOCK_SIZE: usize = 16;

/// The GGUF keys that are not in [`ModelConfig`] and must not be guessed.
const RMS_EPS_KEY: &str = "qwen35moe.attention.layer_norm_rms_epsilon";
const ROPE_FREQ_BASE_KEY: &str = "qwen35moe.rope.freq_base";

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
    /// Sequence state could not be allocated or reset.
    State(StateError),
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
}

impl std::fmt::Display for ForwardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compile(m) => write!(f, "embedding kernel compilation failed: {m}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::Gdn(e) => write!(f, "{e}"),
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
            Self::State(e) => write!(f, "{e}"),
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
    /// Bytes the ten Gated Attention blocks hold **in addition** to the
    /// arena's copy of the same tensors.
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
    moe: MoeBlock,
    lm_head: LmHeadKernels,

    /// Zero-copy aliases into the weight arena. Never dropped — see
    /// [`crate::weights::ResidentTensor`].
    w_token_embd: ManuallyDrop<CudaSlice<u8>>,
    w_output_norm: ManuallyDrop<CudaSlice<f32>>,
    w_lm_head: ManuallyDrop<CudaSlice<u8>>,
    gdn_weights: Vec<ManuallyDrop<GdnLayerWeights>>,
    /// The Q8_0 projections repacked for integer tensor cores, one per Gated
    /// DeltaNet layer — empty when this shape will not use them.
    ///
    /// Built only for shapes above `GdnBlock::uses_tensor_cores`, so a decode
    /// pass does not pay 1.13 GiB for a path it never takes.
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

    report: ForwardReport,
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
            ctx, stream, file, directory, weights, config, tokens, None, None,
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
    ) -> Result<Self, ForwardError> {
        let hidden = config.hidden_size as usize;
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

        // The repack serves two different paths: the tensor cores above the
        // threshold, and a one-token GEMV that wants the split layout for its
        // *alignment* rather than its arithmetic — Q8_0's 34-byte block stride
        // makes an in-place read straddle a sector boundary fifteen times in
        // sixteen. See `gdn_proj_split_gemv`.
        //
        // Shared through `reshape` like the MoE weights, and for the same
        // reason: a prefill pass and the decode pass driven over the same
        // sequence would otherwise hold two copies of 1.13 GiB.
        let gdn_int8 = match gdn_int8 {
            Some(shared) => shared,
            None => {
                let mut built = Vec::new();
                if GdnBlock::uses_tensor_cores(tokens) || tokens == 1 {
                    for w in &gdn_weights {
                        built.push(gdn.repack(stream, w)?);
                    }
                }
                Arc::new(built)
            }
        };

        // --- the 10 Gated Attention layers, which copy --------------------
        let attn_kernels = Arc::new(AttentionKernelSet::new(ctx, &config, tokens)?);
        let mut attention = Vec::new();
        for layer in 0..config.num_layers {
            if config.layer_kind(layer) != LayerKind::GatedAttention {
                continue;
            }
            attention.push(GatedAttentionBlock::new(
                Arc::clone(&attn_kernels),
                stream,
                weights,
                &config,
                layer,
                tokens,
                rms_eps,
                rope_theta,
            )?);
        }

        // --- the MoE, on every block ---------------------------------------
        let moe_geometry = MoeBlock::geometry_for(&config, MOE_BLOCK_SIZE, tokens);
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

        let attention_duplicate_bytes = attention_bytes(weights, &config);
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

    /// Greedy sampling: the id of the largest logit, ties to the lower id.
    ///
    /// Synchronizes the stream, because the caller needs the token before it
    /// can decide what to feed back in — the round-trip is unavoidable in an
    /// autoregressive loop. What is avoidable is its *width*: reducing on the
    /// device makes the transfer four bytes instead of the 993 KiB a
    /// 248,320-entry logit vector occupies, and it moves the scan itself off
    /// a single CPU core. That was worth about 4% of a decode step.
    ///
    /// The tie-break matches [`xabe_kernels::gemv::argmax`] exactly, which is
    /// what `tests/lm_head_differential.rs` asserts. A near-tie between two
    /// logits is the one place a kernel can be well inside every tolerance
    /// and still emit a different token.
    pub fn sample_argmax(&mut self, stream: &Arc<CudaStream>) -> Result<i32, ForwardError> {
        let vocab = self.vocab;
        self.lm_head.argmax(
            stream,
            &self.logits,
            vocab,
            &mut self.argmax_values,
            &mut self.argmax_indices,
            &mut self.argmax_out,
        )?;
        let host = stream.clone_dtoh(&self.argmax_out)?;
        stream.synchronize()?;
        Ok(host[0])
    }

    /// Drop the repacked int8 weights, forcing every projection back to fp32.
    ///
    /// Exists so a differential test can run the *same shape* both ways and
    /// compare: the tensor-core path changes the numerics (activations are
    /// quantized to int8), and the golden capture is 19 tokens — below the
    /// threshold where that path engages — so the oracle gate alone would
    /// never exercise it. See `tests/int8_forward.rs`.
    ///
    /// Frees about 1.43 GiB. Not reversible without rebuilding the pass.
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

    /// Whether this pass has the repacked int8 weights resident, in both
    /// mixers. A pass with one side repacked and not the other is a bug, so
    /// this reports the conjunction rather than either half.
    pub fn tensor_cores_enabled(&self) -> bool {
        !self.gdn_int8.is_empty()
            && self.attention.iter().all(|b| b.tensor_cores_enabled())
            && self.moe.tensor_cores_enabled()
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

        let pos_offset = state.position();
        stream.memcpy_htod(token_ids, &mut self.d_tokens)?;
        self.mark(stream, Stage::Reset)?;
        self.embed(stream)?;
        self.mark(stream, Stage::Embed)?;
        on_waypoint(None, &self.hidden_state);

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
                    self.attention[attn_slot].forward(
                        stream,
                        &self.hidden_state,
                        state.kv_mut(attn_slot),
                        pos_offset,
                        &mut self.mixer_out,
                    )?;
                    attn_slot += 1;
                }
            }
            self.mark(stream, Stage::Mixer { layer, kind })?;

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
            on_waypoint(Some(layer), &self.hidden_state);
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
        // Last, and only on success: every block has now appended, so the
        // state's claim about how many positions its caches hold is true.
        // Advancing earlier would leave a failed pass claiming positions that
        // no cache was written for, and the next call would read them.
        state.advance(self.tokens);
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
