//! Gated Attention: 10 of Qwen3.6's 40 layers, plus the MTP head at block 40.
//!
//! Every layer whose index mod `pattern_period` equals `attention_offset` —
//! 3, 7, 11, ... 39 — is this shape. Geometry is 16 query heads against 2 KV
//! heads (GQA 8:1), head dimension 256, and **partial** rotary over the
//! leading 64 of those 256 dimensions.
//!
//! The order of operations is transcribed from
//! `llama.cpp/src/models/qwen35moe.cpp`, `llama_model_qwen35moe::graph::
//! build_layer_attn`, which comments it as *"joint QG projection, QG split, Q
//! norm, KV projection, K norm, RoPE, attention"*:
//!
//! | Step | This module | Golden node |
//! | --- | --- | --- |
//! | RMSNorm over the residual stream | [`LayerOpsKernels::rms_norm`] | `attn_norm-N` |
//! | packed query+gate projection | [`LmHeadKernels::forward`] | `Qcur_full-N` |
//! | deinterleave query from gate | [`AttentionKernels::split_query_and_gate`] | `Qcur_reshaped-N` / `gate_reshaped-N` |
//! | per-head RMSNorm on the query | [`LayerOpsKernels::rms_norm`] | `Qcur_normed-N` |
//! | key and value projections | [`LmHeadKernels::forward`] | `Kcur-N` / `Vcur-N` (first) |
//! | per-head RMSNorm on the key | [`LayerOpsKernels::rms_norm`] | `Kcur_normed-N` |
//! | partial rotary on query and key | [`AttentionKernels::rope`] | `Qcur-N` / `Kcur-N` (second) |
//! | append roped keys and raw values to the cache | [`KvCache`] | — |
//! | causal GQA attention | [`AttentionKernels::forward`] | `attn_pregate-N` |
//! | `sigmoid(gate)` and its product | [`LayerOpsKernels::sigmoid_gate`] | `gate_sigmoid-N` / `attn_gated-N` |
//! | output projection | [`LmHeadKernels::forward`] | `attn_output-N` |
//! | residual add | [`LayerOpsKernels::add`] | `attn_residual-N` |
//!
//! Every one of those tensors is exposed by an accessor on
//! [`GatedAttentionBlock`] after [`GatedAttentionBlock::forward`], because
//! that is what makes a wrong block *bisectable* against the capture rather
//! than merely wrong — see `crates/xabe-engine/tests/attention_block.rs`.
//!
//! # The trap: `attn_q.weight` is interleaved, not halved
//!
//! `blk.N.attn_q.weight` is `[2048, 8192]` — twice the query width — and
//! packs each head's query immediately followed by that head's **output
//! gate**: `[q_h0, gate_h0, q_h1, gate_h1, ...]`, per-head stride
//! `2 * head_dim`. Upstream reads the query with `ggml_view_3d(..., stride =
//! n_embd_head * 2, ...)` and the gate with the same view at byte offset
//! `n_embd_head`.
//!
//! Splitting the tensor into two contiguous halves instead is arithmetically
//! valid, produces finite plausible activations, and is a **different model**:
//! query heads 8..15 would be fed the gates of heads 0..7. The two readings
//! agree exactly on head 0, so a spot check passes. This module never does
//! the split itself — [`AttentionKernels::split_query_and_gate`] owns it, and
//! `docs/ORACLE.md` §6.4 proves the layout from the capture (72,960 of 77,824
//! elements disagree under the halves reading).
//!
//! # IMRoPE reduces to NEOX here, and that was checked rather than assumed
//!
//! `qwen35moe.cpp` applies `ggml_rope_multi` with `rope_type =
//! LLAMA_ROPE_TYPE_IMROPE` and the file's `rope.dimension_sections`, which is
//! `[11, 11, 10, 0]` for this model — three sections summing to 32, which is
//! `rope.dimension_count / 2`. Interleaved M-RoPE assigns pair `s` to the
//! t/h/w position channel by `s % 3`, and falls through to a **fourth**
//! channel only when `s` runs past `3 * sections[c]` for its own class. With
//! `[11, 11, 10, 0]` no pair in `0..32` falls through, and for a text batch
//! llama.cpp sets the t/h/w channels to the token position (and only the
//! unused fourth to 0, see `llm_graph_input_pos::set_input`). So every
//! rotated dimension is rotated by the token's ordinary position, at
//! `theta_base^(-2i/rope_dim)` — plain NEOX partial rotary, which is what
//! [`AttentionKernels::rope`] implements.
//!
//! That reduction is asserted from the model file's own metadata in
//! `tests/attention_block.rs`, so a model whose sections do not collapse this
//! way fails loudly instead of being silently rotated wrong.
//!
//! # Fixed token count, and the KV cache that works with it
//!
//! A block is constructed for exactly `tokens` positions per call.
//! [`LmHeadKernels::forward`] and [`AttentionKernels::forward`] both validate
//! buffer lengths against the declared geometry *exactly* rather than
//! accepting a prefix, and the two disagree about what that geometry is
//! (`max_tokens` versus `n_query`), so one buffer cannot serve both at two
//! different lengths. Chunked prefill and decode therefore need either a
//! block per shape or a prefix-tolerant length check in those two kernels.
//! This is a property of their validation, not of the arithmetic here.
//!
//! **A block per shape is the option taken**, and it is the one that keeps
//! `AGENTS.md` rule 5: every launch shape stays a function of the declared
//! geometry, so a decode block built at `tokens = 1` has identical launch
//! dimensions on every step and is capturable in a CUDA graph. A prefix
//! tolerance would have made the grid depend on a host value and given that
//! up for the entire pass.
//!
//! What is *not* duplicated per shape is the state. [`KvCache`] and
//! [`GdnState`](crate::block::gdn::GdnState) are owned by the caller and passed in, so a prefill block at
//! `tokens = 19` and a decode block at `tokens = 1` write to and read from the
//! same cache — which is what makes the second of those a continuation of the
//! first rather than a separate sequence. The weights are not duplicated
//! either: [`GatedAttentionBlock::new`] copies a layer out of
//! [`DeviceWeights`] into a shared `Arc`, and [`crate::forward::Forward`]
//! hands that same `Arc` — with its lazy split-layout repack — to every shape
//! it builds, so a second full-model shape adds only shape-local kernels and
//! scratch.

use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaContext, CudaEvent, CudaSlice, CudaStream, DriverError, PinnedHostSlice};

use xabe_cuda::arena::ArenaError;
use xabe_cuda::kernels::attention::{AttentionError, AttentionKernels, AttnDecodeScratch};
use xabe_cuda::kernels::layer_ops::{GateShape, LayerOpsError, LayerOpsKernels};
use xabe_cuda::kernels::lm_head::{LmHeadError, LmHeadGeometry, LmHeadKernels};
use xabe_cuda::kernels::mma::{MMA_SPLIT_TOKENS, MmaError, MmaKernels};
use xabe_cuda::kernels::moe::{ExpertQuant, QuantTensor};
use xabe_gguf::GgmlType;
use xabe_model::config::ModelConfig;
use xabe_model::weights::Role;

use crate::weights::DeviceWeights;

/// Something went wrong building or running a Gated Attention block.
#[derive(Debug)]
pub enum AttentionBlockError {
    /// The driver failed.
    Driver(DriverError),
    /// A reserved arena range could not be read back.
    Arena(ArenaError),
    /// One of the mixer kernels rejected a launch.
    Attention(AttentionError),
    /// The integer tensor-core path rejected a repack, a quantization, or a
    /// projection launch.
    Mma(MmaError),
    /// RMSNorm rejected a launch.
    LayerOps(LayerOpsError),
    /// A projection rejected a launch.
    Projection(LmHeadError),
    /// The device weights do not carry a tensor this block needs.
    ///
    /// A missing projection is not recoverable by falling back to something
    /// else: the block would produce finite, plausible, wrong activations.
    MissingWeight { role: Role, layer: u32 },
    /// A weight is not the element type this block unpacks.
    WrongQuant {
        role: Role,
        layer: u32,
        found: GgmlType,
    },
    /// A weight is not the shape the model geometry implies.
    WrongShape {
        role: Role,
        layer: u32,
        expected: Vec<u64>,
        found: Vec<u64>,
    },
    /// The layer index is not a Gated Attention layer for this config.
    NotAnAttentionLayer { layer: u32 },
    /// A buffer is not the length the declared geometry requires.
    BufferShape {
        what: &'static str,
        expected: usize,
        actual: usize,
    },
    /// The sequence would run past the end of its key/value cache.
    ///
    /// The cache is allocated once for the longest sequence a worker admits,
    /// so this is admission control having failed upstream rather than a
    /// recoverable condition. Rejected rather than wrapped: overwriting
    /// position 0 would silently answer from a different prompt.
    CacheExhausted {
        position: usize,
        tokens: usize,
        max_seq: usize,
    },
}

impl std::fmt::Display for AttentionBlockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::Arena(e) => write!(f, "{e}"),
            Self::Attention(e) => write!(f, "{e}"),
            Self::Mma(e) => write!(f, "{e}"),
            Self::LayerOps(e) => write!(f, "{e}"),
            Self::Projection(e) => write!(f, "{e}"),
            Self::MissingWeight { role, layer } => {
                write!(f, "block {layer} has no resident `{role}` tensor")
            }
            Self::WrongQuant { role, layer, found } => write!(
                f,
                "blk.{layer} `{role}` is {found:?}; this block unpacks Q8_0 projections \
                 and f32 norms only",
            ),
            Self::WrongShape {
                role,
                layer,
                expected,
                found,
            } => write!(
                f,
                "blk.{layer} `{role}` has dims {found:?}, the model geometry implies {expected:?}",
            ),
            Self::NotAnAttentionLayer { layer } => write!(
                f,
                "layer {layer} is not a Gated Attention layer under this ModelConfig",
            ),
            Self::BufferShape {
                what,
                expected,
                actual,
            } => write!(
                f,
                "{what} holds {actual} floats, but this geometry needs {expected}",
            ),
            Self::CacheExhausted {
                position,
                tokens,
                max_seq,
            } => write!(
                f,
                "{tokens} tokens at position {position} would need {} cache slots, but the cache holds {max_seq}",
                position + tokens,
            ),
        }
    }
}

impl std::error::Error for AttentionBlockError {}

impl From<DriverError> for AttentionBlockError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}
impl From<ArenaError> for AttentionBlockError {
    fn from(e: ArenaError) -> Self {
        Self::Arena(e)
    }
}
impl From<MmaError> for AttentionBlockError {
    fn from(e: MmaError) -> Self {
        Self::Mma(e)
    }
}

impl From<AttentionError> for AttentionBlockError {
    fn from(e: AttentionError) -> Self {
        Self::Attention(e)
    }
}
impl From<LayerOpsError> for AttentionBlockError {
    fn from(e: LayerOpsError) -> Self {
        Self::LayerOps(e)
    }
}
impl From<LmHeadError> for AttentionBlockError {
    fn from(e: LmHeadError) -> Self {
        Self::Projection(e)
    }
}

fn expect_len(
    what: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), AttentionBlockError> {
    if actual == expected {
        Ok(())
    } else {
        Err(AttentionBlockError::BufferShape {
            what,
            expected,
            actual,
        })
    }
}

/// Kernels shared by every Gated Attention block in a model.
///
/// Compiled once and shared: NVRTC compilation is a per-module cost, and ten
/// text-stack blocks plus the MTP head would otherwise pay it forty-four
/// times at startup. Everything here depends only on the geometry, never on
/// which layer it is.
pub struct AttentionKernelSet {
    mixer: AttentionKernels,
    ops: LayerOpsKernels,
    /// hidden -> `2 * q_heads * head_dim`, the packed query+gate projection.
    qgate: LmHeadKernels,
    /// hidden -> `kv_heads * head_dim`. Shared by the key and the value —
    /// they have identical geometry, so one compile serves both.
    kv: LmHeadKernels,
    /// `q_heads * head_dim` -> hidden.
    out: LmHeadKernels,
}

impl AttentionKernelSet {
    /// Compile every kernel a Gated Attention block launches, for `tokens`
    /// positions per call.
    pub fn new(
        ctx: &Arc<CudaContext>,
        config: &ModelConfig,
        tokens: usize,
    ) -> Result<Self, AttentionBlockError> {
        let a = config.attention;
        let hidden = config.hidden_size as usize;
        let head_dim = a.head_dim as usize;
        let q_dim = a.q_heads as usize * head_dim;
        let kv_dim = a.kv_heads as usize * head_dim;

        Ok(Self {
            mixer: AttentionKernels::new(ctx, a.q_heads as usize, a.kv_heads as usize, head_dim)?,
            ops: LayerOpsKernels::new(ctx)?,
            qgate: LmHeadKernels::new(
                ctx,
                LmHeadGeometry {
                    hidden,
                    vocab: 2 * q_dim,
                    max_tokens: tokens,
                },
            )?,
            kv: LmHeadKernels::new(
                ctx,
                LmHeadGeometry {
                    hidden,
                    vocab: kv_dim,
                    max_tokens: tokens,
                },
            )?,
            out: LmHeadKernels::new(
                ctx,
                LmHeadGeometry {
                    hidden: q_dim,
                    vocab: hidden,
                    max_tokens: tokens,
                },
            )?,
        })
    }
}

/// One attention layer's key/value cache for one sequence.
///
/// This is the memory that makes decode cheaper than re-prefilling: without
/// it, emitting token `n` costs a pass over all `n` positions, and generation
/// is quadratic in the length of what it has already said.
///
/// # Why it is a plain slab and not the paged pool
///
/// `xabe-cache` owns a two-group pager built for many concurrent sequences
/// sharing a pool, and that is the right structure for the server. This type
/// is the single-sequence case, allocated once for `max_seq` positions and
/// filled in order. Wiring the pager to the device is milestone 07's job; a
/// contiguous cache is what lets the decode path be measured before that
/// lands, and the kernel's absolute-position indexing is the same either way.
///
/// # Size
///
/// `2 * kv_heads * head_dim * 4` bytes per position per layer — 4 KiB for
/// Qwen3.6, whose 2 KV heads are the entire reason a 256-wide head dimension
/// is affordable. Ten attention layers of forty makes 40 KiB per token, so a
/// 32,768-token sequence costs 1.25 GiB. The other thirty layers are Gated
/// DeltaNet and carry a fixed-size recurrent state instead, which is what
/// [`GdnState`](crate::block::gdn::GdnState) holds and why this model's cache does not grow the way a
/// forty-layer dense model's would.
pub struct KvCache {
    k: CudaSlice<u16>,
    v: CudaSlice<u16>,
    kv_dim: usize,
    max_seq: usize,
}

/// One attention layer's preallocated retention-interval delta.
pub(crate) struct HostKvPrefix {
    pub k: PinnedHostSlice<u16>,
    pub v: PinnedHostSlice<u16>,
}

impl KvCache {
    /// Allocate a zeroed cache for `max_seq` positions of one attention layer.
    ///
    /// Zeroed rather than uninitialised because the value is read back for
    /// positions the kernel is allowed to reach, and a NaN left in an
    /// unwritten slot would propagate through the softmax rather than being
    /// masked away.
    pub fn new(
        stream: &Arc<CudaStream>,
        config: &ModelConfig,
        max_seq: usize,
    ) -> Result<Self, AttentionBlockError> {
        let kv_dim = config.attention.kv_heads as usize * config.attention.head_dim as usize;
        Self::with_kv_dim(stream, kv_dim, max_seq)
    }

    /// [`Self::new`] for a cache whose per-token width is not the target
    /// model's — the DFlash drafter's layers cache `8 heads × 128` where the
    /// target caches `2 × 256`.
    pub fn with_kv_dim(
        stream: &Arc<CudaStream>,
        kv_dim: usize,
        max_seq: usize,
    ) -> Result<Self, AttentionBlockError> {
        Ok(Self {
            k: stream.alloc_zeros::<u16>(max_seq * kv_dim)?,
            v: stream.alloc_zeros::<u16>(max_seq * kv_dim)?,
            kv_dim,
            max_seq,
        })
    }

    /// Positions this cache can hold.
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// Device bytes held, both halves.
    ///
    /// binary16, which is 20 KiB per token at this model's geometry rather than
    /// 40 — 2.53 GiB at 131,072 positions instead of 5.06. It is also the dtype
    /// llama.cpp's cache uses, and the one `xabe-cache`'s planner has always
    /// assumed (`DEFAULT_ELEM_SIZE = 2`), which this makes true rather than
    /// aspirational.
    pub fn bytes(&self) -> u64 {
        2 * (self.max_seq * self.kv_dim * size_of::<u16>()) as u64
    }

    /// The cached keys, `[max_seq][kv_heads][head_dim]`, rotary already
    /// applied. Positions at or above the sequence length are zero.
    pub fn keys(&self) -> &CudaSlice<u16> {
        &self.k
    }

    /// The cached values, same layout, not normed and not rotated.
    pub fn values(&self) -> &CudaSlice<u16> {
        &self.v
    }

    /// Both cache halves mutably, for a caller appending through
    /// `AttentionKernels::append_kv` directly rather than through
    /// [`GatedAttentionBlock`] — the DFlash drafter, whose injection path
    /// writes projected context K/V here.
    pub(crate) fn kv_mut(&mut self) -> (&mut CudaSlice<u16>, &mut CudaSlice<u16>) {
        (&mut self.k, &mut self.v)
    }

    pub(crate) fn snapshot_range_into(
        &self,
        stream: &Arc<CudaStream>,
        start: usize,
        positions: usize,
        prefix: &mut HostKvPrefix,
    ) -> Result<(), DriverError> {
        assert!(start <= positions);
        assert!(positions <= self.max_seq);
        let first = start * self.kv_dim;
        let elements = (positions - start) * self.kv_dim;
        assert_eq!(elements, prefix.k.len());
        assert_eq!(elements, prefix.v.len());
        stream.memcpy_dtoh(&self.k.slice(first..first + elements), &mut prefix.k)?;
        stream.memcpy_dtoh(&self.v.slice(first..first + elements), &mut prefix.v)?;
        Ok(())
    }

    pub(crate) fn restore_prefix(
        &mut self,
        stream: &Arc<CudaStream>,
        prefix: &HostKvPrefix,
        start: usize,
        positions: usize,
    ) -> Result<(), DriverError> {
        assert!(start <= positions);
        assert!(positions <= self.max_seq);
        let first = start * self.kv_dim;
        let elements = (positions - start) * self.kv_dim;
        assert_eq!(elements, prefix.k.len());
        assert_eq!(elements, prefix.v.len());
        stream.memcpy_htod(&prefix.k, &mut self.k.slice_mut(first..first + elements))?;
        stream.memcpy_htod(&prefix.v, &mut self.v.slice_mut(first..first + elements))?;
        Ok(())
    }
}

/// One Gated Attention block's weights, scratch, and forward pass.
pub struct GatedAttentionBlock {
    kernels: Arc<AttentionKernelSet>,
    layer: u32,
    tokens: usize,
    hidden: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    rms_eps: f32,
    rope_theta: f32,

    weights: Arc<AttentionLayerWeights>,

    /// The four Q8_0 projections repacked into the split layout the integer
    /// tensor cores can load, plus the handle that drives them.
    ///
    /// `None` below [`MMA_SPLIT_TOKENS`]. The MMA path pays a fixed cost — a
    /// quantization sweep over the activations and a grid quantized to 8-token
    /// tiles — that a short batch cannot amortize, and a decode step of one
    /// token would pay all of it to fill an eighth of a fragment.
    int8: Option<Arc<AttnInt8>>,

    /// Side streams and events for the batch-prefill per-sequence fan-out.
    ///
    /// The flattened pass runs the shared projections batch-wide, but rope,
    /// the KV append and the causal attention itself are per-sequence: their
    /// launches touch disjoint scratch slices and each sequence's own cache,
    /// and at deep KV the attention launch alone is several partial waves
    /// (512 blocks over 72 SMs at the 128K cells) whose scheduling tail is
    /// paid once per launch. Running the three sequences' chains serially
    /// pays that tail three times per layer per chunk; forking them merges
    /// the blocks into one occupancy pool with a single tail. Two side
    /// streams, one fork event recorded after the last shared input is
    /// written, and one join event each before the batch-wide gate resumes
    /// on the main stream. `None` when `LLMXABE_ATTN_PREFILL_SERIAL=1`
    /// keeps the serial order for A/B.
    batch_fork: Option<(Vec<Arc<CudaStream>>, CudaEvent, Vec<CudaEvent>)>,
}

/// Shape-independent device weights for one Gated Attention layer.
///
/// The ordinary and split-layout copies are shared by every fixed-token
/// shape. `int8` is populated while a shape is built, never in a forward
/// pass, so lazy construction does not put an allocation on the hot path.
pub(crate) struct AttentionLayerWeights {
    w_input_norm: CudaSlice<f32>,
    w_q_norm: CudaSlice<f32>,
    w_k_norm: CudaSlice<f32>,
    w_qgate: CudaSlice<u8>,
    w_k: CudaSlice<u8>,
    w_v: CudaSlice<u8>,
    w_out: CudaSlice<u8>,
    int8: Mutex<Option<Arc<AttnInt8>>>,
}

/// The per-pass buffers every Gated Attention layer needs, owned once.
///
/// These are pure scratch: each is written and consumed inside a single
/// layer's [`GatedAttentionBlock::forward`] and carries nothing across layers.
/// The ten layers run strictly in sequence, so one set serves all of them, and
/// holding one set instead of ten is the difference between 1.68 MB and
/// 0.17 MB of VRAM per token of context.
///
/// That is not a micro-optimization at this geometry. `q_dim` is 4,096 —
/// sixteen heads of 256 — and seven of these buffers are that wide, so a
/// private copy per layer was the single largest consumer of per-token memory
/// in the engine, larger than the Gated DeltaNet block and the MoE dispatch
/// put together. `GdnBlock` was already shared across its thirty layers; this
/// is the same arrangement for the other ten.
pub struct AttnScratch {
    tokens: usize,
    normed: CudaSlice<f32>,
    packed: CudaSlice<f32>,
    query: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    query_normed: CudaSlice<f32>,
    query_roped: CudaSlice<f32>,
    key: CudaSlice<f32>,
    key_normed: CudaSlice<f32>,
    key_roped: CudaSlice<f32>,
    value: CudaSlice<f32>,
    pregate: CudaSlice<f32>,
    gate_sigmoid: CudaSlice<f32>,
    gated: CudaSlice<f32>,
    projected: CudaSlice<f32>,
    /// Activations quantized once per contraction width and reused by every
    /// projection that shares it. Allocated on first use, then kept.
    xq: Option<(CudaSlice<i8>, CudaSlice<f32>)>,
    /// Per-slice partials for the flash-decoding path. Sized by the head
    /// geometry rather than by `tokens`, and used only at `n_query == 1`, but
    /// held here because this is the struct with a stream to allocate from and
    /// the one already shared by all ten attention layers.
    decode: Vec<AttnDecodeScratch>,
}

impl AttnScratch {
    /// Allocate for a fixed token count, once per [`crate::forward::Forward`].
    pub fn new(
        stream: &Arc<CudaStream>,
        config: &ModelConfig,
        tokens: usize,
    ) -> Result<Self, AttentionBlockError> {
        let hidden = config.hidden_size as usize;
        let a = &config.attention;
        let q_dim = a.q_heads as usize * a.head_dim as usize;
        let kv_dim = a.kv_heads as usize * a.head_dim as usize;
        Ok(Self {
            tokens,
            normed: stream.alloc_zeros::<f32>(tokens * hidden)?,
            packed: stream.alloc_zeros::<f32>(tokens * 2 * q_dim)?,
            query: stream.alloc_zeros::<f32>(tokens * q_dim)?,
            gate: stream.alloc_zeros::<f32>(tokens * q_dim)?,
            query_normed: stream.alloc_zeros::<f32>(tokens * q_dim)?,
            query_roped: stream.alloc_zeros::<f32>(tokens * q_dim)?,
            key: stream.alloc_zeros::<f32>(tokens * kv_dim)?,
            key_normed: stream.alloc_zeros::<f32>(tokens * kv_dim)?,
            key_roped: stream.alloc_zeros::<f32>(tokens * kv_dim)?,
            value: stream.alloc_zeros::<f32>(tokens * kv_dim)?,
            pregate: stream.alloc_zeros::<f32>(tokens * q_dim)?,
            gate_sigmoid: stream.alloc_zeros::<f32>(tokens * q_dim)?,
            gated: stream.alloc_zeros::<f32>(tokens * q_dim)?,
            projected: stream.alloc_zeros::<f32>(tokens * hidden)?,
            xq: None,
            // Prefill uses only entry zero; batched decode needs one entry per
            // independent sequence. The engine serves three and the isolated
            // width sweep exercises up to eight.
            // Allocating one ~4.6 MiB split-K arena per prefill token made a
            // 2,048-row pass reserve roughly 9.5 GiB for scratch it never
            // indexed. Keep the fixed serving maximum instead.
            decode: (0..tokens.min(8))
                .map(|_| AttnDecodeScratch::new(stream, a.q_heads as usize, a.head_dim as usize))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    /// The token count this scratch was sized for.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Drop the quantized activation mirror. See
    /// [`GatedAttentionBlock::disable_tensor_cores`].
    pub fn forget_quantized_activations(&mut self) {
        self.xq = None;
    }

    /// `attn_norm-N`: the input RMSNorm, `[tokens][hidden]`.
    pub fn normed_input(&self) -> &CudaSlice<f32> {
        &self.normed
    }
    /// `Qcur_full-N`: the packed query+gate projection, `[tokens][2*q_dim]`.
    pub fn packed_query_gate(&self) -> &CudaSlice<f32> {
        &self.packed
    }
    /// `Qcur_reshaped-N`: the deinterleaved query, `[tokens][q_heads][head_dim]`.
    pub fn query(&self) -> &CudaSlice<f32> {
        &self.query
    }
    /// `gate_reshaped-N`: the deinterleaved output gate, same shape.
    pub fn gate(&self) -> &CudaSlice<f32> {
        &self.gate
    }
    /// `Qcur_normed-N`: the query after its per-head RMSNorm.
    pub fn query_normed(&self) -> &CudaSlice<f32> {
        &self.query_normed
    }
    /// `Qcur-N`: the query after partial rotary.
    pub fn query_roped(&self) -> &CudaSlice<f32> {
        &self.query_roped
    }
    /// `Kcur-N` (first record): the raw key projection, `[tokens][kv_dim]`.
    pub fn key(&self) -> &CudaSlice<f32> {
        &self.key
    }
    /// `Kcur_normed-N`: the key after its per-head RMSNorm.
    pub fn key_normed(&self) -> &CudaSlice<f32> {
        &self.key_normed
    }
    /// `Kcur-N` (second record): the key after partial rotary.
    pub fn key_roped(&self) -> &CudaSlice<f32> {
        &self.key_roped
    }
    /// `Vcur-N`: the value projection. Neither normed nor rotated.
    pub fn value(&self) -> &CudaSlice<f32> {
        &self.value
    }
    /// `attn_pregate-N`: attention output before the gate.
    pub fn pregate(&self) -> &CudaSlice<f32> {
        &self.pregate
    }
    /// `gate_sigmoid-N`: `sigmoid(gate)`.
    pub fn gate_sigmoid(&self) -> &CudaSlice<f32> {
        &self.gate_sigmoid
    }
    /// `attn_gated-N`: the gated attention output.
    pub fn gated(&self) -> &CudaSlice<f32> {
        &self.gated
    }
    /// `attn_output-N`: the output projection, before the residual add.
    pub fn projected(&self) -> &CudaSlice<f32> {
        &self.projected
    }
}

/// One attention layer's Q8_0 projections in the split int8 layout.
///
/// Q8_0 interleaves a 2-byte scale with every 32 quants, so a 34-byte stride
/// puts every operand load off a 4-byte boundary and the tensor-core fragment
/// load faults. Splitting the quants from the scales into two arrays is what
/// makes the load legal, and it is worth about 12x over assembling the
/// operands byte by byte in place.
///
/// Costs about 30.7 MB per layer on top of the Q8_0 the block already holds —
/// int8 plus one fp32 scale per 32, against Q8_0's fp16 scale per 32.
struct AttnInt8 {
    mma: MmaKernels,
    qgate_q: CudaSlice<i8>,
    qgate_s: CudaSlice<f32>,
    k_q: CudaSlice<i8>,
    k_s: CudaSlice<f32>,
    v_q: CudaSlice<i8>,
    v_s: CudaSlice<f32>,
    out_q: CudaSlice<i8>,
    out_s: CudaSlice<f32>,
}

fn shared_or_try_init<T, E>(
    slot: &Mutex<Option<Arc<T>>>,
    build: impl FnOnce() -> Result<T, E>,
) -> Result<Arc<T>, E> {
    let mut cached = slot.lock().expect("attention repack mutex poisoned");
    if let Some(value) = cached.as_ref() {
        return Ok(Arc::clone(value));
    }
    let value = Arc::new(build()?);
    *cached = Some(Arc::clone(&value));
    Ok(value)
}

/// Which scratch buffer [`GatedAttentionBlock::quantize_activations`] reads.
///
/// The quantizer needs `&mut self` for the destination and `&self` for the
/// source, and both are fields of the same block. Naming the source instead of
/// passing it sidesteps the borrow rather than working around it with a clone.
#[derive(Clone, Copy)]
enum ScratchPick {
    /// `normed` — the RMSNormed residual stream, shared by q+gate, k and v.
    Normed,
    /// `gated` — the sigmoid-gated core output, read only by the output
    /// projection, which contracts over `q_dim` rather than `hidden`.
    Gated,
}

impl AttnInt8 {
    /// Repack this layer's four Q8_0 projections into the split layout.
    ///
    /// Runs once, at construction. The element counts are passed rather than
    /// derived because `qgate` is `hidden x 2*q_dim` and `out` is
    /// `q_dim x hidden` — deriving them here would duplicate the shape
    /// reasoning that `new` already did against the file's directory.
    #[allow(clippy::too_many_arguments)]
    fn repack(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        w_qgate: &CudaSlice<u8>,
        w_k: &CudaSlice<u8>,
        w_v: &CudaSlice<u8>,
        w_out: &CudaSlice<u8>,
        qgate_elems: usize,
        kv_elems: usize,
        out_elems: usize,
    ) -> Result<Self, AttentionBlockError> {
        let mma = MmaKernels::new(ctx)?;
        let one = |src: &CudaSlice<u8>, elements: usize| -> Result<_, AttentionBlockError> {
            let mut q = stream.alloc_zeros::<i8>(elements)?;
            let mut sc = stream.alloc_zeros::<f32>(elements / 32)?;
            mma.repack_q8_0(stream, src, &mut q, &mut sc, elements)?;
            Ok((q, sc))
        };
        let (qgate_q, qgate_s) = one(w_qgate, qgate_elems)?;
        let (k_q, k_s) = one(w_k, kv_elems)?;
        let (v_q, v_s) = one(w_v, kv_elems)?;
        let (out_q, out_s) = one(w_out, out_elems)?;
        Ok(Self {
            mma,
            qgate_q,
            qgate_s,
            k_q,
            k_s,
            v_q,
            v_s,
            out_q,
            out_s,
        })
    }
}

/// Where the rotary angle for one chunk comes from.
///
/// The cache-slot / causal position stays a scalar in every case — only the
/// *rotary* position ever needs three components, and only on prefill
/// chunks that overlap an image span. Text sequences and every decode step
/// use `Scalar`, whose value equals the slot position plus the sequence's
/// rope delta (0 for text-only — the byte-identical path).
#[derive(Clone, Copy)]
pub enum RopeSource<'a> {
    /// One device scalar; token `i` rotates by `*base + i`.
    Scalar(&'a CudaSlice<i32>),
    /// `(t, h, w)` i32 triples, one per token of the chunk — interleaved
    /// M-RoPE per `xabe_kernels::mrope::apply_imrope`, sections
    /// [`xabe_kernels::mrope::QWEN3_6_SECTIONS`].
    PerToken(&'a CudaSlice<i32>),
}

impl GatedAttentionBlock {
    /// Whether a batch of `tokens` is wide enough to be worth the integer
    /// tensor-core path. Shared with [`crate::block::gdn::GdnBlock`] so the
    /// two mixers never disagree about which arithmetic a pass is using.
    pub fn uses_tensor_cores(tokens: usize) -> bool {
        tokens >= MMA_SPLIT_TOKENS
    }

    /// Drop this block's repacked int8 weights, forcing its four projections
    /// back to the fp32 kernels. See [`crate::forward::Forward::disable_tensor_cores`].
    pub fn disable_tensor_cores(&mut self) {
        // Deliberately leave `weights.int8` intact: another shape may still
        // use it, and disabling this block is a shape-local A/B decision.
        self.int8 = None;
    }

    /// Whether this block has its repacked int8 weights resident.
    pub fn tensor_cores_enabled(&self) -> bool {
        self.int8.is_some()
    }

    /// Force this block's decode step off the tensor-core split-precision
    /// kernel and back onto `attn_flash_decode_warp`'s per-key online
    /// softmax. See [`crate::forward::Forward::disable_decode_mma`].
    pub fn disable_decode_mma(&mut self) {
        self.kernels.mixer.disable_decode_mma();
    }

    /// Select which occupancy width this block's tensor-core decode kernel
    /// uses. See [`crate::forward::Forward::set_decode_mma_wpo`].
    pub fn set_decode_mma_wpo(&mut self, wpo: usize) {
        self.kernels.mixer.set_decode_mma_wpo(wpo);
    }

    /// Set how many blocks this block's tensor-core decode launches over
    /// its logical splits — `0` restores one block per split. Bit-identical
    /// scheduling knob; see the mixer's own doc and
    /// `docs/BENCHMARKS.md` 2026-08-20 for the wave arithmetic.
    pub fn set_decode_mma_blocks(&mut self, blocks: usize) {
        self.kernels.mixer.set_decode_mma_blocks(blocks);
    }

    /// Quantize one scratch buffer to int8 for the projections that read it.
    ///
    /// Called twice per pass: once over `normed` for the three projections
    /// that contract over `hidden`, once over `gated` for the output
    /// projection that contracts over `q_dim`. The buffer is allocated once at
    /// the wider of the two so the second call cannot reallocate mid-pass,
    /// which `AGENTS.md` rule 6 forbids.
    fn quantize_activations(
        &self,
        stream: &Arc<CudaStream>,
        sc: &mut AttnScratch,
        pick: ScratchPick,
        tokens: usize,
        k_dim: usize,
    ) -> Result<(), AttentionBlockError> {
        let widest = tokens * self.hidden.max(self.q_heads * self.head_dim);
        let need = tokens * k_dim;
        debug_assert!(need <= widest);
        if !sc.xq.as_ref().is_some_and(|(q, _)| q.len() >= need) {
            sc.xq = Some((
                stream.alloc_zeros::<i8>(widest)?,
                stream.alloc_zeros::<f32>(widest / 32)?,
            ));
        }
        let src = match pick {
            ScratchPick::Normed => &sc.normed,
            ScratchPick::Gated => &sc.gated,
        };
        let i8w = self.int8.as_ref().expect("caller checked");
        let (q, sc) = sc.xq.as_mut().expect("just allocated");
        i8w.mma.quantize_rows(stream, src, q, sc, tokens, k_dim)?;
        Ok(())
    }

    /// Resolve layer `layer`'s weights out of `weights` and allocate scratch.
    ///
    /// `rms_eps` is the file's `*.attention.layer_norm_rms_epsilon` and
    /// `rope_theta` its `*.rope.freq_base`; neither is in [`ModelConfig`], and
    /// guessing either produces a block that is finite and wrong, so both are
    /// arguments rather than constants.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kernels: Arc<AttentionKernelSet>,
        stream: &Arc<CudaStream>,
        weights: &DeviceWeights,
        config: &ModelConfig,
        layer: u32,
        tokens: usize,
        rms_eps: f32,
        rope_theta: f32,
    ) -> Result<Self, AttentionBlockError> {
        let shared = Arc::new(Self::load_weights(stream, weights, config, layer)?);
        Self::from_shared(
            kernels, stream, shared, config, layer, tokens, rms_eps, rope_theta,
        )
    }

    fn load_weights(
        stream: &Arc<CudaStream>,
        weights: &DeviceWeights,
        config: &ModelConfig,
        layer: u32,
    ) -> Result<AttentionLayerWeights, AttentionBlockError> {
        let a = config.attention;
        let hidden = config.hidden_size as usize;
        let head_dim = a.head_dim as usize;
        let q_heads = a.q_heads as usize;
        let kv_heads = a.kv_heads as usize;
        let q_dim = q_heads * head_dim;
        let kv_dim = kv_heads * head_dim;

        let h = hidden as u64;
        let w_input_norm = f32_weight(weights, stream, Role::InputNorm, layer, &[h])?;
        let w_q_norm = f32_weight(weights, stream, Role::AttnQNorm, layer, &[head_dim as u64])?;
        let w_k_norm = f32_weight(weights, stream, Role::AttnKNorm, layer, &[head_dim as u64])?;
        let w_qgate = q8_0_weight(
            weights,
            stream,
            Role::AttnQGate,
            layer,
            &[h, 2 * q_dim as u64],
        )?;
        let w_k = q8_0_weight(weights, stream, Role::AttnK, layer, &[h, kv_dim as u64])?;
        let w_v = q8_0_weight(weights, stream, Role::AttnV, layer, &[h, kv_dim as u64])?;
        let w_out = q8_0_weight(weights, stream, Role::AttnOut, layer, &[q_dim as u64, h])?;

        Ok(AttentionLayerWeights {
            w_input_norm,
            w_q_norm,
            w_k_norm,
            w_qgate,
            w_k,
            w_v,
            w_out,
            int8: Mutex::new(None),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_shared(
        kernels: Arc<AttentionKernelSet>,
        stream: &Arc<CudaStream>,
        weights: Arc<AttentionLayerWeights>,
        config: &ModelConfig,
        layer: u32,
        tokens: usize,
        rms_eps: f32,
        rope_theta: f32,
    ) -> Result<Self, AttentionBlockError> {
        let a = config.attention;
        let hidden = config.hidden_size as usize;
        let head_dim = a.head_dim as usize;
        let q_heads = a.q_heads as usize;
        let kv_heads = a.kv_heads as usize;
        let q_dim = q_heads * head_dim;
        let kv_dim = kv_heads * head_dim;

        // Build on demand at shape construction, not on the hot path. The
        // mutex makes concurrent reshape construction share the same repack.
        let int8 = if Self::uses_tensor_cores(tokens) {
            Some(shared_or_try_init(&weights.int8, || {
                let repacked = AttnInt8::repack(
                    stream.context(),
                    stream,
                    &weights.w_qgate,
                    &weights.w_k,
                    &weights.w_v,
                    &weights.w_out,
                    hidden * 2 * q_dim,
                    hidden * kv_dim,
                    q_dim * hidden,
                )?;
                // Publish the cache entry only after its producing stream has
                // finished. Concurrent shape construction may use another
                // stream, which otherwise has no dependency on these writes.
                stream.synchronize()?;
                Ok::<_, AttentionBlockError>(repacked)
            })?)
        } else {
            None
        };

        Ok(Self {
            kernels,
            layer,
            tokens,
            hidden,
            q_heads,
            kv_heads,
            head_dim,
            rope_dim: a.rope_dim as usize,
            rms_eps,
            rope_theta,
            weights,
            int8,
            batch_fork: if std::env::var_os("LLMXABE_ATTN_PREFILL_SERIAL").is_some() {
                None
            } else {
                let ctx = stream.context();
                Some((
                    vec![ctx.new_stream()?, ctx.new_stream()?],
                    ctx.new_event(None)?,
                    vec![ctx.new_event(None)?, ctx.new_event(None)?],
                ))
            },
        })
    }

    pub(crate) fn shared_weights(&self) -> Arc<AttentionLayerWeights> {
        Arc::clone(&self.weights)
    }

    /// Which block this is.
    pub fn layer(&self) -> u32 {
        self.layer
    }

    /// Positions this block was sized for. See the module docs on why it is
    /// fixed.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Run the block over one self-contained window of `tokens` positions.
    ///
    /// `hidden_state` and `out` are both `[tokens][hidden]`, token-major, and
    /// may not alias — the residual add reads `hidden_state` after every
    /// projection has run, but `out` is written by a separate launch and
    /// aliasing would make the ordering a race rather than a dependency.
    ///
    /// `pos_offset` is the absolute position of token 0, which is what the
    /// rotary embedding rotates by and where this batch's keys and values are
    /// appended to `cache`. Query row `i` therefore attends to every key in
    /// `[0, pos_offset + i]` — this batch's own keys and every key the cache
    /// already held. `pos_offset == 0` is a cold prefill and `n_query == 1`
    /// with `pos_offset == n - 1` is a decode step; they are the same code.
    ///
    /// The keys written to the cache are the *roped* keys, and the values are
    /// the raw projections. That asymmetry is not an oversight: rotary is a
    /// function of absolute position, so a key rotated once at write time is
    /// correct for every later query, whereas rotating on read would redo the
    /// same work for the whole window on every step. Values are never rotated
    /// at all.
    /// Eight plain arguments rather than a config struct, for the reason the
    /// rest of this workspace gives: every one is a distinct per-call tensor or
    /// position, not related configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        stream: &Arc<CudaStream>,
        sc: &mut AttnScratch,
        hidden_state: &CudaSlice<f32>,
        cache: &mut KvCache,
        pos_offset: usize,
        positions: &CudaSlice<i32>,
        rope: RopeSource<'_>,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), AttentionBlockError> {
        let t = self.tokens;
        let hidden_elems = t * self.hidden;
        let q_dim = self.q_heads * self.head_dim;
        let kv_dim = self.kv_heads * self.head_dim;
        expect_len("block input", hidden_state.len(), hidden_elems)?;
        expect_len("block output", out.len(), hidden_elems)?;
        if pos_offset + t > cache.max_seq {
            return Err(AttentionBlockError::CacheExhausted {
                position: pos_offset,
                tokens: t,
                max_seq: cache.max_seq,
            });
        }

        // Cloned rather than borrowed: `quantize_activations` needs `&mut
        // self`, and an outstanding `&self.kernels` would make that a conflict
        // for the whole pass. The clone is one atomic increment on an `Arc`.
        let k = Arc::clone(&self.kernels);

        // 1. RMSNorm over the residual stream.
        k.ops.rms_norm(
            stream,
            hidden_state,
            &self.weights.w_input_norm,
            &mut sc.normed,
            t,
            self.hidden,
            self.rms_eps,
        )?;

        // 2. The packed query+gate projection.
        //
        // Steps 2 and 5 all read `normed` and all contract over `hidden`, so
        // one quantization serves three projections.
        if self.int8.is_some() {
            self.quantize_activations(stream, sc, ScratchPick::Normed, t, self.hidden)?;
        }
        if let Some(i8w) = self.int8.as_ref() {
            let (xq, xs) = sc.xq.as_ref().expect("quantized above");
            i8w.mma.q8_0_proj_split(
                stream,
                &i8w.qgate_q,
                &i8w.qgate_s,
                xq,
                xs,
                &mut sc.packed,
                self.hidden,
                2 * q_dim,
                t,
            )?;
        } else {
            k.qgate.forward(
                stream,
                QuantTensor {
                    bytes: &self.weights.w_qgate,
                    quant: ExpertQuant::Q8_0,
                },
                &sc.normed,
                t,
                &mut sc.packed,
            )?;
        }

        // 3. Deinterleave. See the module docs: this is the step that is a
        //    different model if it is done as a halves split.
        k.mixer
            .split_query_and_gate(stream, &sc.packed, &mut sc.query, &mut sc.gate, t)?;

        // 4. Per-head RMSNorm on the query.
        k.ops.rms_norm(
            stream,
            &sc.query,
            &self.weights.w_q_norm,
            &mut sc.query_normed,
            t * self.q_heads,
            self.head_dim,
            self.rms_eps,
        )?;

        // 5. Key and value projections, off the same normed input — and off
        //    the same quantization of it that step 2 already paid for.
        if let Some(i8w) = self.int8.as_ref() {
            let (xq, xs) = sc.xq.as_ref().expect("quantized at step 2");
            i8w.mma.q8_0_proj_split(
                stream,
                &i8w.k_q,
                &i8w.k_s,
                xq,
                xs,
                &mut sc.key,
                self.hidden,
                kv_dim,
                t,
            )?;
            i8w.mma.q8_0_proj_split(
                stream,
                &i8w.v_q,
                &i8w.v_s,
                xq,
                xs,
                &mut sc.value,
                self.hidden,
                kv_dim,
                t,
            )?;
        } else {
            k.kv.forward(
                stream,
                QuantTensor {
                    bytes: &self.weights.w_k,
                    quant: ExpertQuant::Q8_0,
                },
                &sc.normed,
                t,
                &mut sc.key,
            )?;
            k.kv.forward(
                stream,
                QuantTensor {
                    bytes: &self.weights.w_v,
                    quant: ExpertQuant::Q8_0,
                },
                &sc.normed,
                t,
                &mut sc.value,
            )?;
        }

        // 6. Per-head RMSNorm on the key. The value is not normed.
        k.ops.rms_norm(
            stream,
            &sc.key,
            &self.weights.w_k_norm,
            &mut sc.key_normed,
            t * self.kv_heads,
            self.head_dim,
            self.rms_eps,
        )?;

        // 7. Partial rotary, on the query and the key only, before the GQA
        //    broadcast — hence two head counts.
        for (src, dst, heads) in [
            (&sc.query_normed, &mut sc.query_roped, self.q_heads),
            (&sc.key_normed, &mut sc.key_roped, self.kv_heads),
        ] {
            match rope {
                RopeSource::Scalar(base) => k.mixer.rope(
                    stream,
                    src,
                    dst,
                    t,
                    heads,
                    self.rope_dim,
                    base,
                    self.rope_theta,
                )?,
                RopeSource::PerToken(mrope) => k.mixer.rope_imrope(
                    stream,
                    src,
                    dst,
                    t,
                    heads,
                    self.rope_dim,
                    mrope,
                    xabe_kernels::mrope::QWEN3_6_SECTIONS,
                    self.rope_theta,
                )?,
            }
        }

        // 8. Append this batch's keys and values to the cache, at the absolute
        //    positions they belong to. One kernel for both halves, reading the
        //    position from the device — this was two `cuMemcpyDtoDAsync` calls
        //    into host-computed slices, which is the same traffic but bakes a
        //    host-chosen destination address into the launch and so cannot be
        //    replayed from a CUDA graph.
        k.mixer.append_kv(
            stream,
            &sc.key_roped,
            &sc.value,
            &mut cache.k,
            &mut cache.v,
            t,
            cache.max_seq,
            positions,
        )?;

        // 9. Causal GQA attention over the whole cached window, not just this
        //    batch. `n_keys` is the filled length; the cache buffer itself is
        //    longer and the kernel reads none of the tail.
        k.mixer.forward(
            stream,
            &mut sc.decode[0],
            &sc.query_roped,
            &cache.k,
            &cache.v,
            &mut sc.pregate,
            t,
            cache.max_seq,
            pos_offset + t,
            positions,
        )?;

        // 10. The output gate.
        k.ops.sigmoid_gate(
            stream,
            &sc.pregate,
            &sc.gate,
            &mut sc.gate_sigmoid,
            &mut sc.gated,
            t,
            q_dim,
            GateShape::Elementwise,
        )?;

        // 11. Output projection. Contracts over `q_dim`, not `hidden`, and
        //     reads the gated core output — so it re-quantizes rather than
        //     reusing what step 2 produced.
        if self.int8.is_some() {
            self.quantize_activations(stream, sc, ScratchPick::Gated, t, q_dim)?;
        }
        if let Some(i8w) = self.int8.as_ref() {
            let (xq, xs) = sc.xq.as_ref().expect("quantized above");
            i8w.mma.q8_0_proj_split(
                stream,
                &i8w.out_q,
                &i8w.out_s,
                xq,
                xs,
                &mut sc.projected,
                q_dim,
                self.hidden,
                t,
            )?;
        } else {
            k.out.forward(
                stream,
                QuantTensor {
                    bytes: &self.weights.w_out,
                    quant: ExpertQuant::Q8_0,
                },
                &sc.gated,
                t,
                &mut sc.projected,
            )?;
        }

        // 12. Residual.
        k.ops
            .add(stream, hidden_state, &sc.projected, out, hidden_elems)?;

        Ok(())
    }

    /// Batched decode: `caches.len()` independent one-token sequences,
    /// advanced by amortizing every weight-bound projection across all of
    /// them and looping only the steps that read a **position** or a
    /// **sequence's own cache** — [`Self::forward`]'s steps 7, 8 and 9.
    ///
    /// # Why this split, and not [`Self::forward`] called `caches.len()` times
    ///
    /// Steps 1, 2, 4, 5, 6, 10, 11 and 12 have no notion of position or
    /// state at all: the input norm, the four Q8_0 projections, the
    /// deinterleave and the two per-head norms are a pure function of that
    /// token's own activations, contracting the same weight regardless of
    /// which sequence the token belongs to or where in its sequence it sits.
    /// Batching them the way a multi-token prefill chunk already does reads
    /// `w_qgate`, `w_k`, `w_v` and `w_out` once for the whole batch instead
    /// of once per sequence — measured, calling [`Self::forward`] per
    /// sequence instead paid the full ~29 MB per-layer weight read
    /// `caches.len()` times over, which was the largest single cost the
    /// batched-decode benchmark found. See
    /// `docs/BENCHMARKS.md`.
    ///
    /// Rotary position, the key/value append and the causal attention read
    /// cannot batch the same way: rotary needs each token's own absolute
    /// position and the mixer kernels' `positions` argument is one scalar a
    /// launch rotates *every* token in the call by (correct for a prefill
    /// chunk, where every token is a later position of the *same* sequence,
    /// and wrong for `N` different sequences at `N` unrelated positions).
    /// Those three steps loop once per sequence instead, each a one-token
    /// call against that sequence's own cache and its own device-resident
    /// position — exactly [`Self::forward`]'s own decode shape, just run
    /// `caches.len()` times rather than folded into the batch. None of the
    /// three touches a weight, so the loop costs small fixed-size launches,
    /// not weight re-reads.
    ///
    /// `hidden_state` and `out` are `[caches.len()][hidden]`, token `i`
    /// belonging to `caches[i]`. `pos_offsets[i]` and `positions[i]` are that
    /// sequence's own absolute position, host value and device scalar
    /// respectively — see [`Self::forward`]'s docs on what each is for.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_batch_decode(
        &mut self,
        stream: &Arc<CudaStream>,
        secondary_streams: &[Arc<CudaStream>],
        fork: &CudaEvent,
        joins: &[CudaEvent],
        sc: &mut AttnScratch,
        hidden_state: &CudaSlice<f32>,
        caches: &mut [&mut KvCache],
        pos_offsets: &[usize],
        positions: &[&CudaSlice<i32>],
        rope_positions: &[&CudaSlice<i32>],
        out: &mut CudaSlice<f32>,
    ) -> Result<(), AttentionBlockError> {
        let n = caches.len();
        let t = self.tokens;
        if t != n {
            return Err(AttentionBlockError::BufferShape {
                what: "batch decode caches",
                expected: t,
                actual: n,
            });
        }
        if pos_offsets.len() != n || positions.len() != n {
            return Err(AttentionBlockError::BufferShape {
                what: "batch decode positions",
                expected: n,
                actual: pos_offsets.len().min(positions.len()),
            });
        }
        let hidden_elems = t * self.hidden;
        let q_dim = self.q_heads * self.head_dim;
        let kv_dim = self.kv_heads * self.head_dim;
        expect_len("block input", hidden_state.len(), hidden_elems)?;
        expect_len("block output", out.len(), hidden_elems)?;
        for (i, cache) in caches.iter().enumerate() {
            if pos_offsets[i] + 1 > cache.max_seq {
                return Err(AttentionBlockError::CacheExhausted {
                    position: pos_offsets[i],
                    tokens: 1,
                    max_seq: cache.max_seq,
                });
            }
        }

        let k = Arc::clone(&self.kernels);

        // 1. RMSNorm over the residual stream. No position, no state: batches
        //    over every sequence's row in one launch.
        k.ops.rms_norm(
            stream,
            hidden_state,
            &self.weights.w_input_norm,
            &mut sc.normed,
            t,
            self.hidden,
            self.rms_eps,
        )?;

        // 2. The packed query+gate projection. See the method docs: this is
        //    the weight read batching exists to amortize.
        if self.int8.is_some() {
            self.quantize_activations(stream, sc, ScratchPick::Normed, t, self.hidden)?;
        }
        if let Some(i8w) = self.int8.as_ref() {
            let (xq, xs) = sc.xq.as_ref().expect("quantized above");
            i8w.mma.q8_0_proj_split(
                stream,
                &i8w.qgate_q,
                &i8w.qgate_s,
                xq,
                xs,
                &mut sc.packed,
                self.hidden,
                2 * q_dim,
                t,
            )?;
        } else {
            k.qgate.forward(
                stream,
                QuantTensor {
                    bytes: &self.weights.w_qgate,
                    quant: ExpertQuant::Q8_0,
                },
                &sc.normed,
                t,
                &mut sc.packed,
            )?;
        }

        // 3. Deinterleave. No position, no state: batches.
        k.mixer
            .split_query_and_gate(stream, &sc.packed, &mut sc.query, &mut sc.gate, t)?;

        // 4. Per-head RMSNorm on the query. Batches.
        k.ops.rms_norm(
            stream,
            &sc.query,
            &self.weights.w_q_norm,
            &mut sc.query_normed,
            t * self.q_heads,
            self.head_dim,
            self.rms_eps,
        )?;

        // 5. Key and value projections. The other half of the weight read
        //    batching exists to amortize.
        if let Some(i8w) = self.int8.as_ref() {
            let (xq, xs) = sc.xq.as_ref().expect("quantized at step 2");
            i8w.mma.q8_0_proj_split(
                stream,
                &i8w.k_q,
                &i8w.k_s,
                xq,
                xs,
                &mut sc.key,
                self.hidden,
                kv_dim,
                t,
            )?;
            i8w.mma.q8_0_proj_split(
                stream,
                &i8w.v_q,
                &i8w.v_s,
                xq,
                xs,
                &mut sc.value,
                self.hidden,
                kv_dim,
                t,
            )?;
        } else {
            k.kv.forward(
                stream,
                QuantTensor {
                    bytes: &self.weights.w_k,
                    quant: ExpertQuant::Q8_0,
                },
                &sc.normed,
                t,
                &mut sc.key,
            )?;
            k.kv.forward(
                stream,
                QuantTensor {
                    bytes: &self.weights.w_v,
                    quant: ExpertQuant::Q8_0,
                },
                &sc.normed,
                t,
                &mut sc.value,
            )?;
        }

        // 6. Per-head RMSNorm on the key. Batches.
        k.ops.rms_norm(
            stream,
            &sc.key,
            &self.weights.w_k_norm,
            &mut sc.key_normed,
            t * self.kv_heads,
            self.head_dim,
            self.rms_eps,
        )?;

        debug_assert!(secondary_streams.is_empty() || secondary_streams.len() == n - 1);
        debug_assert_eq!(joins.len(), secondary_streams.len());
        if !secondary_streams.is_empty() {
            fork.record(stream)?;
            for secondary in secondary_streams {
                secondary.wait(fork)?;
            }
        }

        // Queue auxiliary rows first so they are ready to issue as soon as
        // the main stream reaches the fork. Row zero is submitted last on
        // the already-busy main stream.
        for i in (0..n).rev() {
            let sequence_stream = if i == 0 || secondary_streams.is_empty() {
                stream
            } else {
                &secondary_streams[i - 1]
            };
            // SAFETY: `i < n = t`; `self.q_heads * self.head_dim` and
            // `self.kv_heads * self.head_dim` are `sc.query_normed`'s/
            // `sc.query_roped`'s and `sc.key_normed`'s/`sc.key_roped`'s own
            // per-token widths, and `kv_dim` is `sc.value`'s, all checked by
            // `AttnScratch::new` against `tokens * width`.
            let q_normed_i = unsafe {
                crate::viewslice::subslice(sequence_stream, &sc.query_normed, i * q_dim, q_dim)
            };
            let mut q_roped_i = unsafe {
                crate::viewslice::subslice(sequence_stream, &sc.query_roped, i * q_dim, q_dim)
            };
            k.mixer.rope(
                sequence_stream,
                &q_normed_i,
                &mut q_roped_i,
                1,
                self.q_heads,
                self.rope_dim,
                rope_positions[i],
                self.rope_theta,
            )?;

            let k_normed_i = unsafe {
                crate::viewslice::subslice(sequence_stream, &sc.key_normed, i * kv_dim, kv_dim)
            };
            let mut k_roped_i = unsafe {
                crate::viewslice::subslice(sequence_stream, &sc.key_roped, i * kv_dim, kv_dim)
            };
            k.mixer.rope(
                sequence_stream,
                &k_normed_i,
                &mut k_roped_i,
                1,
                self.kv_heads,
                self.rope_dim,
                rope_positions[i],
                self.rope_theta,
            )?;

            let value_i = unsafe {
                crate::viewslice::subslice(sequence_stream, &sc.value, i * kv_dim, kv_dim)
            };
            k.mixer.append_kv(
                sequence_stream,
                &k_roped_i,
                &value_i,
                &mut caches[i].k,
                &mut caches[i].v,
                1,
                caches[i].max_seq,
                positions[i],
            )?;

            let mut pregate_i = unsafe {
                crate::viewslice::subslice(sequence_stream, &sc.pregate, i * q_dim, q_dim)
            };
            k.mixer.forward(
                sequence_stream,
                &mut sc.decode[i],
                &q_roped_i,
                &caches[i].k,
                &caches[i].v,
                &mut pregate_i,
                1,
                caches[i].max_seq,
                pos_offsets[i] + 1,
                positions[i],
            )?;
            if !secondary_streams.is_empty() && i > 0 {
                joins[i - 1].record(sequence_stream)?;
            }
        }
        for join in joins {
            stream.wait(join)?;
        }

        // 10. The output gate. No position, no state: batches.
        k.ops.sigmoid_gate(
            stream,
            &sc.pregate,
            &sc.gate,
            &mut sc.gate_sigmoid,
            &mut sc.gated,
            t,
            q_dim,
            GateShape::Elementwise,
        )?;

        // 11. Output projection. The third weight read batching amortizes.
        if self.int8.is_some() {
            self.quantize_activations(stream, sc, ScratchPick::Gated, t, q_dim)?;
        }
        if let Some(i8w) = self.int8.as_ref() {
            let (xq, xs) = sc.xq.as_ref().expect("quantized above");
            i8w.mma.q8_0_proj_split(
                stream,
                &i8w.out_q,
                &i8w.out_s,
                xq,
                xs,
                &mut sc.projected,
                q_dim,
                self.hidden,
                t,
            )?;
        } else {
            k.out.forward(
                stream,
                QuantTensor {
                    bytes: &self.weights.w_out,
                    quant: ExpertQuant::Q8_0,
                },
                &sc.gated,
                t,
                &mut sc.projected,
            )?;
        }

        // 12. Residual. Batches.
        k.ops
            .add(stream, hidden_state, &sc.projected, out, hidden_elems)?;

        Ok(())
    }

    /// Batched prefill for independent equal-length chunks. Weight-bound
    /// projections run over the flattened sequence-major rows; rotary, KV
    /// append, and attention remain sequence-local.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_batch_prefill(
        &mut self,
        stream: &Arc<CudaStream>,
        sc: &mut AttnScratch,
        hidden_state: &CudaSlice<f32>,
        caches: &mut [&mut KvCache],
        chunk_tokens: usize,
        pos_offsets: &[usize],
        positions: &[&CudaSlice<i32>],
        rope_positions: &[&CudaSlice<i32>],
        out: &mut CudaSlice<f32>,
    ) -> Result<(), AttentionBlockError> {
        let n = caches.len();
        if n == 0 || chunk_tokens == 0 || n * chunk_tokens != self.tokens {
            return Err(AttentionBlockError::BufferShape {
                what: "batch prefill tokens",
                expected: self.tokens,
                actual: n * chunk_tokens,
            });
        }
        if pos_offsets.len() != n || positions.len() != n {
            return Err(AttentionBlockError::BufferShape {
                what: "batch prefill positions",
                expected: n,
                actual: pos_offsets.len().min(positions.len()),
            });
        }
        let total = self.tokens;
        let hidden_elems = total * self.hidden;
        let q_dim = self.q_heads * self.head_dim;
        let kv_dim = self.kv_heads * self.head_dim;
        expect_len("block input", hidden_state.len(), hidden_elems)?;
        expect_len("block output", out.len(), hidden_elems)?;
        for (i, c) in caches.iter().enumerate() {
            if pos_offsets[i] + chunk_tokens > c.max_seq {
                return Err(AttentionBlockError::CacheExhausted {
                    position: pos_offsets[i],
                    tokens: chunk_tokens,
                    max_seq: c.max_seq,
                });
            }
            // The attention kernels take the sequence's starting position as
            // one device scalar and add the row index internally.
            expect_len("batch prefill position", positions[i].len(), 1)?;
        }
        let k = Arc::clone(&self.kernels);
        k.ops.rms_norm(
            stream,
            hidden_state,
            &self.weights.w_input_norm,
            &mut sc.normed,
            total,
            self.hidden,
            self.rms_eps,
        )?;
        if self.int8.is_some() {
            self.quantize_activations(stream, sc, ScratchPick::Normed, total, self.hidden)?;
        }
        if let Some(i8w) = self.int8.as_ref() {
            let (xq, xs) = sc.xq.as_ref().expect("quantized above");
            i8w.mma.q8_0_proj_split(
                stream,
                &i8w.qgate_q,
                &i8w.qgate_s,
                xq,
                xs,
                &mut sc.packed,
                self.hidden,
                2 * q_dim,
                total,
            )?;
        } else {
            k.qgate.forward(
                stream,
                QuantTensor {
                    bytes: &self.weights.w_qgate,
                    quant: ExpertQuant::Q8_0,
                },
                &sc.normed,
                total,
                &mut sc.packed,
            )?;
        }
        k.mixer
            .split_query_and_gate(stream, &sc.packed, &mut sc.query, &mut sc.gate, total)?;
        k.ops.rms_norm(
            stream,
            &sc.query,
            &self.weights.w_q_norm,
            &mut sc.query_normed,
            total * self.q_heads,
            self.head_dim,
            self.rms_eps,
        )?;
        if let Some(i8w) = self.int8.as_ref() {
            let (xq, xs) = sc.xq.as_ref().expect("quantized above");
            i8w.mma.q8_0_proj_split(
                stream,
                &i8w.k_q,
                &i8w.k_s,
                xq,
                xs,
                &mut sc.key,
                self.hidden,
                kv_dim,
                total,
            )?;
            i8w.mma.q8_0_proj_split(
                stream,
                &i8w.v_q,
                &i8w.v_s,
                xq,
                xs,
                &mut sc.value,
                self.hidden,
                kv_dim,
                total,
            )?;
        } else {
            k.kv.forward(
                stream,
                QuantTensor {
                    bytes: &self.weights.w_k,
                    quant: ExpertQuant::Q8_0,
                },
                &sc.normed,
                total,
                &mut sc.key,
            )?;
            k.kv.forward(
                stream,
                QuantTensor {
                    bytes: &self.weights.w_v,
                    quant: ExpertQuant::Q8_0,
                },
                &sc.normed,
                total,
                &mut sc.value,
            )?;
        }
        k.ops.rms_norm(
            stream,
            &sc.key,
            &self.weights.w_k_norm,
            &mut sc.key_normed,
            total * self.kv_heads,
            self.head_dim,
            self.rms_eps,
        )?;
        // The per-sequence chains below — rope, the cache append, the causal
        // attention — touch disjoint scratch slices and each sequence's own
        // cache, so they are forked onto side streams and rejoined before the
        // batch-wide gate. See the `batch_fork` field for why. The fork event
        // orders them after the last shared input written above
        // (`key_normed`); each join orders the main stream after that
        // sequence's `pregate` slice is complete.
        let fork_streams = match self.batch_fork.as_ref() {
            Some((streams, fork, _)) if n > 1 => {
                fork.record(stream)?;
                for side in streams.iter().take(n - 1) {
                    side.wait(fork)?;
                }
                Some(streams)
            }
            _ => None,
        };
        for i in 0..n {
            // Sequence 0 stays on the main stream; the rest round-robin the
            // side streams (two cover the N=3 serving shape exactly, wider
            // batches share them and serialize pairwise, still correct).
            let seq_stream: &Arc<CudaStream> = match (i, fork_streams) {
                (0, _) | (_, None) => stream,
                (_, Some(streams)) => &streams[(i - 1) % streams.len()],
            };
            let base = i * chunk_tokens;
            let qn = unsafe {
                crate::viewslice::subslice(
                    stream,
                    &sc.query_normed,
                    base * q_dim,
                    chunk_tokens * q_dim,
                )
            };
            let mut qr = unsafe {
                crate::viewslice::subslice(
                    stream,
                    &sc.query_roped,
                    base * q_dim,
                    chunk_tokens * q_dim,
                )
            };
            let kn = unsafe {
                crate::viewslice::subslice(
                    stream,
                    &sc.key_normed,
                    base * kv_dim,
                    chunk_tokens * kv_dim,
                )
            };
            let mut kr = unsafe {
                crate::viewslice::subslice(
                    stream,
                    &sc.key_roped,
                    base * kv_dim,
                    chunk_tokens * kv_dim,
                )
            };
            k.mixer.rope(
                seq_stream,
                &qn,
                &mut qr,
                chunk_tokens,
                self.q_heads,
                self.rope_dim,
                rope_positions[i],
                self.rope_theta,
            )?;
            k.mixer.rope(
                seq_stream,
                &kn,
                &mut kr,
                chunk_tokens,
                self.kv_heads,
                self.rope_dim,
                rope_positions[i],
                self.rope_theta,
            )?;
            let val = unsafe {
                crate::viewslice::subslice(stream, &sc.value, base * kv_dim, chunk_tokens * kv_dim)
            };
            k.mixer.append_kv(
                seq_stream,
                &kr,
                &val,
                &mut caches[i].k,
                &mut caches[i].v,
                chunk_tokens,
                caches[i].max_seq,
                positions[i],
            )?;
            let mut pg = unsafe {
                crate::viewslice::subslice(stream, &sc.pregate, base * q_dim, chunk_tokens * q_dim)
            };
            k.mixer.forward(
                seq_stream,
                &mut sc.decode[0],
                &qr,
                &caches[i].k,
                &caches[i].v,
                &mut pg,
                chunk_tokens,
                caches[i].max_seq,
                pos_offsets[i] + chunk_tokens,
                positions[i],
            )?;
        }
        if let (Some(streams), Some((_, _, joins))) = (fork_streams, self.batch_fork.as_ref()) {
            for (side, join) in streams.iter().zip(joins).take(n - 1) {
                join.record(side)?;
                stream.wait(join)?;
            }
        }
        k.ops.sigmoid_gate(
            stream,
            &sc.pregate,
            &sc.gate,
            &mut sc.gate_sigmoid,
            &mut sc.gated,
            total,
            q_dim,
            GateShape::Elementwise,
        )?;
        if self.int8.is_some() {
            self.quantize_activations(stream, sc, ScratchPick::Gated, total, q_dim)?;
        }
        if let Some(i8w) = self.int8.as_ref() {
            let (xq, xs) = sc.xq.as_ref().expect("quantized above");
            i8w.mma.q8_0_proj_split(
                stream,
                &i8w.out_q,
                &i8w.out_s,
                xq,
                xs,
                &mut sc.projected,
                q_dim,
                self.hidden,
                total,
            )?;
        } else {
            k.out.forward(
                stream,
                QuantTensor {
                    bytes: &self.weights.w_out,
                    quant: ExpertQuant::Q8_0,
                },
                &sc.gated,
                total,
                &mut sc.projected,
            )?;
        }
        k.ops
            .add(stream, hidden_state, &sc.projected, out, hidden_elems)?;
        Ok(())
    }
}

/// Copy a resident Q8_0 tensor into a buffer of its own.
///
/// [`QuantTensor`] carries a whole `CudaSlice`, and its element count is
/// checked against the declared geometry, so a sub-range of the weight arena
/// cannot be passed directly — the arena is one 29.6 GiB slab. The copy is a
/// device-to-host-to-device round trip of at most 17.8 MiB (`attn_q`), paid
/// once at construction and never on the forward path. The fix is a borrowed
/// device view that `xabe-cuda` does not expose yet — `cudarc` 0.19 can make a
/// `CudaView` of a sub-range but not an owned `CudaSlice`, and every kernel
/// entry point takes the latter.
fn q8_0_weight(
    weights: &DeviceWeights,
    stream: &Arc<CudaStream>,
    role: Role,
    layer: u32,
    dims: &[u64],
) -> Result<CudaSlice<u8>, AttentionBlockError> {
    let placement = weights
        .find(role, Some(layer))
        .ok_or(AttentionBlockError::MissingWeight { role, layer })?;
    if placement.ggml_type != GgmlType::Q8_0 {
        return Err(AttentionBlockError::WrongQuant {
            role,
            layer,
            found: placement.ggml_type,
        });
    }
    if placement.dims != dims {
        return Err(AttentionBlockError::WrongShape {
            role,
            layer,
            expected: dims.to_vec(),
            found: placement.dims.clone(),
        });
    }
    let bytes = weights.arena().read(stream, &placement.alloc)?;
    Ok(stream.clone_htod(bytes.as_slice())?)
}

/// Copy a resident f32 norm vector into a typed buffer.
fn f32_weight(
    weights: &DeviceWeights,
    stream: &Arc<CudaStream>,
    role: Role,
    layer: u32,
    dims: &[u64],
) -> Result<CudaSlice<f32>, AttentionBlockError> {
    let placement = weights
        .find(role, Some(layer))
        .ok_or(AttentionBlockError::MissingWeight { role, layer })?;
    if placement.ggml_type != GgmlType::F32 {
        return Err(AttentionBlockError::WrongQuant {
            role,
            layer,
            found: placement.ggml_type,
        });
    }
    if placement.dims != dims {
        return Err(AttentionBlockError::WrongShape {
            role,
            layer,
            expected: dims.to_vec(),
            found: placement.dims.clone(),
        });
    }
    let bytes = weights.arena().read(stream, &placement.alloc)?;
    let values: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(f32::from_le_bytes)
        .collect();
    Ok(stream.clone_htod(values.as_slice())?)
}

/// Layers of `config` that are this block shape.
///
/// Derived from the pattern rather than listed, so a config change cannot
/// leave a hardcoded list behind. For Qwen3.6 this is 3, 7, ... 39.
pub fn attention_layers(config: &ModelConfig) -> impl Iterator<Item = u32> + '_ {
    (0..config.num_layers).filter(move |&l| l % config.pattern_period == config.attention_offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use xabe_model::config::LayerKind;

    #[test]
    fn the_attention_layers_are_the_ten_the_pattern_implies() {
        let config = ModelConfig::qwen3_6_35b_a3b();
        let layers: Vec<u32> = attention_layers(&config).collect();
        assert_eq!(layers, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39]);
        assert_eq!(layers.len(), 10);
        for &l in &layers {
            assert_eq!(config.layer_kind(l), LayerKind::GatedAttention);
        }
        // And nothing else is.
        for l in 0..config.num_layers {
            assert_eq!(
                layers.contains(&l),
                config.layer_kind(l) == LayerKind::GatedAttention,
                "layer {l} disagrees with ModelConfig::layer_kind",
            );
        }
    }

    #[test]
    fn the_projection_geometries_are_the_ones_the_file_carries() {
        // These four shapes are what `new` validates the resident tensors
        // against, and getting one backwards is the failure `docs/ORACLE.md`
        // §6.2 calls out: `attn_output` reads 4096 and writes 2048, the
        // opposite of every other row in its table.
        let config = ModelConfig::qwen3_6_35b_a3b();
        let a = config.attention;
        let hidden = u64::from(config.hidden_size);
        let head_dim = u64::from(a.head_dim);
        let q_dim = u64::from(a.q_heads) * head_dim;
        let kv_dim = u64::from(a.kv_heads) * head_dim;

        assert_eq!([hidden, 2 * q_dim], [2048, 8192]);
        assert_eq!([hidden, kv_dim], [2048, 512]);
        assert_eq!([q_dim, hidden], [4096, 2048]);
        assert_eq!(head_dim, 256);
    }

    #[test]
    fn the_projection_widths_satisfy_the_lm_head_kernels_staging_constraint() {
        // `LmHeadKernels::new` rejects a `hidden` that is not a whole number
        // of 16-block Q8_0 staging segments, i.e. a multiple of 512. Both
        // input widths this block uses must clear that, or three of the four
        // projections would fail to construct at runtime rather than here.
        let config = ModelConfig::qwen3_6_35b_a3b();
        let q_dim = (config.attention.q_heads * config.attention.head_dim) as usize;
        for width in [config.hidden_size as usize, q_dim] {
            assert_eq!(
                width % 512,
                0,
                "width {width} is not a whole staging segment"
            );
        }
    }

    #[test]
    fn a_length_that_matches_the_geometry_is_accepted_and_one_that_does_not_is_not() {
        assert!(expect_len("x", 4, 4).is_ok());
        let err = expect_len("x", 5, 4).unwrap_err();
        assert!(err.to_string().contains('5'));
        assert!(err.to_string().contains('4'));
    }

    #[test]
    fn a_lazy_repack_is_built_once_and_shared() {
        let slot = Mutex::new(None);
        let first = shared_or_try_init(&slot, || Ok::<_, ()>(17)).unwrap();
        let second = shared_or_try_init(&slot, || Ok::<_, ()>(99)).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(*second, 17);
    }

    #[test]
    fn a_failed_lazy_repack_can_be_retried() {
        let slot = Mutex::new(None);
        assert!(shared_or_try_init::<u32, _>(&slot, || Err("failed")).is_err());
        let recovered = shared_or_try_init(&slot, || Ok::<_, &str>(23)).unwrap();
        assert_eq!(*recovered, 23);
    }
}
