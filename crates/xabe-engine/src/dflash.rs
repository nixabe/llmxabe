//! The DFlash drafter: block in-fill speculative drafting for the serving
//! engine.
//!
//! `xabe_model::dflash` owns the shape (and the module docs explaining the
//! algorithm and which upstream's reading of the checkpoint this follows);
//! this module owns execution. Three pieces:
//!
//! - [`load_dflash_weights`]: the drafter GGUF's tensors, resident on the
//!   device (Q8_0 matrices as raw bytes, f32 norms as f32).
//! - [`DFlashForward::inject_context`]: the *encoder* — concatenated
//!   target-layer features through `fc`, RMSNorm, per-layer K/V projection,
//!   K-norm and rope, written into a sequence's [`DFlashDraftCache`].
//!   Nothing here reads token ids; the drafter's picture of the context is
//!   entirely the target's hidden states.
//! - [`DFlashForward::draft`]: the *query pass* — `[id_last, MASK × n]`
//!   through the six dense blocks, non-causally, and the target's own LM
//!   head; rows `1..=n` argmax into `n` drafted tokens in **one** drafter
//!   forward regardless of `n`.
//!
//! Every heavy kernel here is a reuse of an existing, differentially-tested
//! one: `LmHeadKernels` for all nine GEMM shapes, `AttentionKernels` (built
//! at the drafter's own 32/8/128 geometry) for rope, cache append and the
//! per-row decode attention, `LayerOpsKernels` for norms and SwiGLU. The
//! only new device code is the tiny glue below: a Q8_0 embedding gather
//! (the same kernel `block/mtp.rs` carries) — the target's embedding table
//! and LM head are aliased, not copied, exactly as the MTP head does.
//!
//! # Non-causal attention through the decode path
//!
//! The drafter attends non-causally (see `xabe_model::dflash`): every query
//! row sees the whole context *and every other query row*, bounded only by
//! the sliding window on the layers that have one. The engine's flash
//! prefill kernels are causal and unusable here, but its flash-*decode*
//! path is not causal at all — it attends one query row over keys
//! `0..=positions[0]`, a device scalar the caller sets. Appending the query
//! block's K/V into the cache at its positions and then running one decode
//! call per query row with that scalar set to the *block's* last position
//! is exactly non-causal block attention. A sliding-window layer offsets
//! the K/V base pointer to the window start instead of masking. At draft
//! widths (`n + 1 ≤ 16` rows) the per-row launches are noise next to the
//! target's verify pass.

use std::mem::ManuallyDrop;
use std::sync::Arc;

use cudarc::driver::PushKernelArg;
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig};

use xabe_cuda::kernels::attention::{AttentionError, AttentionKernels, AttnDecodeScratch};
use xabe_cuda::kernels::compile;
use xabe_cuda::kernels::layer_ops::{LayerOpsError, LayerOpsKernels};
use xabe_cuda::kernels::lm_head::{
    ARGMAX_BLOCKS, HeadTensor, LmHeadError, LmHeadGeometry, LmHeadKernels,
};
use xabe_gguf::{GgmlType, GgufFile};
use xabe_model::dflash::{DFlashConfig, DFlashRole, DFlashWeightSchema};
use xabe_model::weights::Role;

use crate::block::attention::KvCache;
use crate::viewslice::subslice;
use crate::weights::DeviceWeights;

const EMBED_THREADS: u32 = 256;

/// The one glue kernel: a Q8_0 embedding-table gather, byte-for-byte the
/// lookup `block/mtp.rs`'s `mtp_embed_q8_0` performs (and is tested
/// through). Duplicated because NVRTC compiles from a string with no
/// cross-string linking.
const DFLASH_GLUE_SRC: &str = r#"
extern "C" {

__device__ __forceinline__ float load_half_le(const unsigned char* p) {
    unsigned short bits = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(bits));
    return f;
}

// grid: (n_tokens). block: EMBED_THREADS, striding the row.
__global__ void dflash_embed_q8_0(
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

// One target-layer tap into the concatenated feature rows:
// aux[row][tap*hidden + j] = src[row][j]. grid: (rows), block strides hidden.
__global__ void dflash_tap(
    const float* __restrict__ src,
    float* __restrict__ aux,
    int hidden,
    int taps,
    int tap,
    int rows
) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* s = src + (long long)r * hidden;
    float* d = aux + ((long long)r * taps + tap) * hidden;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        d[j] = s[j];
    }
}

}
"#;

/// Something went wrong loading or running the DFlash drafter.
#[derive(Debug)]
pub enum DFlashError {
    /// NVRTC rejected the glue source, or the module failed to load.
    Compile(String),
    /// The driver failed.
    Driver(DriverError),
    /// The drafter GGUF does not match the schema.
    Schema(String),
    /// A drafter tensor is stored in a format this path cannot read.
    WrongQuant { name: String, found: GgmlType },
    /// A target tensor the drafter shares (embeddings, LM head) is absent
    /// or the wrong format.
    TargetWeight(String),
    /// Attention kernels refused.
    Attention(AttentionError),
    /// A norm or the SwiGLU refused.
    LayerOps(LayerOpsError),
    /// A projection GEMM refused.
    LmHead(LmHeadError),
    /// A caller shape disagrees with construction.
    Shape {
        what: &'static str,
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for DFlashError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compile(e) => write!(f, "DFlash glue compile failed: {e}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::Schema(e) => write!(f, "drafter GGUF mismatch: {e}"),
            Self::WrongQuant { name, found } => {
                write!(
                    f,
                    "drafter tensor {name} is {found:?}, not the expected format"
                )
            }
            Self::TargetWeight(e) => write!(f, "shared target tensor: {e}"),
            Self::Attention(e) => write!(f, "{e}"),
            Self::LayerOps(e) => write!(f, "{e}"),
            Self::LmHead(e) => write!(f, "{e}"),
            Self::Shape {
                what,
                expected,
                got,
            } => {
                write!(f, "{what}: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for DFlashError {}

macro_rules! from_error {
    ($src:ty, $variant:ident) => {
        impl From<$src> for DFlashError {
            fn from(e: $src) -> Self {
                Self::$variant(e)
            }
        }
    };
}
from_error!(DriverError, Driver);
from_error!(AttentionError, Attention);
from_error!(LayerOpsError, LayerOps);
from_error!(LmHeadError, LmHead);

/// One drafter block's resident tensors.
struct DFlashLayer {
    attn_norm: CudaSlice<f32>,
    wq: CudaSlice<u8>,
    wk: CudaSlice<u8>,
    wv: CudaSlice<u8>,
    wo: CudaSlice<u8>,
    q_norm: CudaSlice<f32>,
    k_norm: CudaSlice<f32>,
    ffn_norm: CudaSlice<f32>,
    w_gate: CudaSlice<u8>,
    w_up: CudaSlice<u8>,
    w_down: CudaSlice<u8>,
}

/// The drafter GGUF's tensors, resident on one device.
pub struct DFlashWeights {
    pub config: DFlashConfig,
    fc: CudaSlice<u8>,
    enc_norm: CudaSlice<f32>,
    output_norm: CudaSlice<f32>,
    layers: Vec<DFlashLayer>,
    /// Total resident bytes, for the engine's VRAM accounting.
    pub bytes: usize,
}

/// Upload every drafter tensor. `file` must match `config`'s schema — the
/// resolve error carries every mismatch, not the first.
pub fn load_dflash_weights(
    stream: &Arc<CudaStream>,
    file: &GgufFile,
    config: &DFlashConfig,
) -> Result<DFlashWeights, DFlashError> {
    let schema = DFlashWeightSchema::new(config);
    if let Err(errors) = schema.resolve(config, file) {
        let mut msg = format!("{} mismatches:", errors.len());
        for e in errors.iter().take(8) {
            msg.push_str(&format!("\n  {e}"));
        }
        return Err(DFlashError::Schema(msg));
    }

    let mut bytes = 0usize;
    let q8 = |stream: &Arc<CudaStream>, bytes: &mut usize, role, layer| {
        let spec = schema.find(role, layer).expect("schema covers its roles");
        let info = file.tensor(&spec.name).expect("resolve checked existence");
        if info.ggml_type != GgmlType::Q8_0 {
            return Err(DFlashError::WrongQuant {
                name: spec.name.clone(),
                found: info.ggml_type,
            });
        }
        let raw = file.tensor_bytes(&spec.name).expect("resolve checked");
        *bytes += raw.len();
        Ok(stream.clone_htod(raw)?)
    };
    let f32t = |stream: &Arc<CudaStream>, bytes: &mut usize, role, layer| {
        let spec = schema.find(role, layer).expect("schema covers its roles");
        let info = file.tensor(&spec.name).expect("resolve checked existence");
        if info.ggml_type != GgmlType::F32 {
            return Err(DFlashError::WrongQuant {
                name: spec.name.clone(),
                found: info.ggml_type,
            });
        }
        let raw = file.tensor_bytes(&spec.name).expect("resolve checked");
        *bytes += raw.len();
        let host: Vec<f32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        Ok::<_, DFlashError>(stream.clone_htod(&host)?)
    };

    use DFlashRole::*;
    let fc = q8(stream, &mut bytes, Fc, None)?;
    let enc_norm = f32t(stream, &mut bytes, EncNorm, None)?;
    let output_norm = f32t(stream, &mut bytes, OutputNorm, None)?;
    let mut layers = Vec::with_capacity(config.num_layers as usize);
    for i in 0..config.num_layers {
        let l = Some(i);
        layers.push(DFlashLayer {
            attn_norm: f32t(stream, &mut bytes, AttnNorm, l)?,
            wq: q8(stream, &mut bytes, AttnQ, l)?,
            wk: q8(stream, &mut bytes, AttnK, l)?,
            wv: q8(stream, &mut bytes, AttnV, l)?,
            wo: q8(stream, &mut bytes, AttnOut, l)?,
            q_norm: f32t(stream, &mut bytes, AttnQNorm, l)?,
            k_norm: f32t(stream, &mut bytes, AttnKNorm, l)?,
            ffn_norm: f32t(stream, &mut bytes, FfnNorm, l)?,
            w_gate: q8(stream, &mut bytes, FfnGate, l)?,
            w_up: q8(stream, &mut bytes, FfnUp, l)?,
            w_down: q8(stream, &mut bytes, FfnDown, l)?,
        });
    }
    Ok(DFlashWeights {
        config: config.clone(),
        fc,
        enc_norm,
        output_norm,
        layers,
        bytes,
    })
}

/// One sequence's drafter-side KV caches: one [`KvCache`] per drafter
/// layer, at the drafter's own kv width. Positions mirror the target's —
/// slot `p` holds the projection of the target's features at position `p`
/// (or, transiently, a query block's own K/V, overwritten by the next
/// injection; see [`DFlashForward::draft`]).
pub struct DFlashDraftCache {
    layers: Vec<KvCache>,
}

impl DFlashDraftCache {
    /// Bytes held across all layers.
    pub fn bytes(&self) -> u64 {
        self.layers.iter().map(KvCache::bytes).sum()
    }
}

/// The drafter's execution state: kernels, projections, and step scratch.
/// One per device runtime, shared by every sequence (per-sequence state is
/// only the [`DFlashDraftCache`]).
pub struct DFlashForward {
    config: DFlashConfig,
    weights: DFlashWeights,
    attn: AttentionKernels,
    ops: LayerOpsKernels,
    dec: AttnDecodeScratch,

    // One GEMM instance per (in, out) shape, each serving every layer's
    // tensor of that shape.
    fc_proj: LmHeadKernels,
    q_proj: LmHeadKernels,
    kv_proj_ctx: LmHeadKernels,
    kv_proj_q: LmHeadKernels,
    o_proj: LmHeadKernels,
    gu_proj: LmHeadKernels,
    down_proj: LmHeadKernels,
    lm_head: LmHeadKernels,

    /// The target's token embedding table and LM head, aliased from its
    /// resident weights — the drafter ships neither (see the schema docs).
    w_token_embd: ManuallyDrop<CudaSlice<u8>>,
    w_lm_head: ManuallyDrop<CudaSlice<u8>>,

    embed_fn: CudaFunction,
    tap_fn: CudaFunction,

    /// Concatenated target features, `[ctx_tokens][taps * hidden]` — filled
    /// tap by tap from the target pass's waypoints, consumed by
    /// [`Self::inject_context`].
    aux: CudaSlice<f32>,

    ctx_g: CudaSlice<f32>,
    ctx_g_normed: CudaSlice<f32>,
    ctx_k: CudaSlice<f32>,
    ctx_k_normed: CudaSlice<f32>,
    ctx_k_roped: CudaSlice<f32>,
    ctx_v: CudaSlice<f32>,

    /// Query-pass capacity (the trained block size) and its buffers.
    q_tokens: usize,
    d_tokens: CudaSlice<i32>,
    x: CudaSlice<f32>,
    normed: CudaSlice<f32>,
    q_buf: CudaSlice<f32>,
    q_normed: CudaSlice<f32>,
    q_roped: CudaSlice<f32>,
    k_buf: CudaSlice<f32>,
    k_normed: CudaSlice<f32>,
    k_roped: CudaSlice<f32>,
    v_buf: CudaSlice<f32>,
    attn_out: CudaSlice<f32>,
    proj_out: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    up: CudaSlice<f32>,
    act: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    argmax_values: CudaSlice<f32>,
    argmax_indices: CudaSlice<i32>,
    argmax_out: CudaSlice<i32>,
    argmax_probs: CudaSlice<f32>,

    /// Reused position/depth scalars: element 0 is the pass's base
    /// position; elements `1..=q_tokens` are per-row deepest-key scalars
    /// for the decode attention calls.
    positions: CudaSlice<i32>,
    pos_host: Vec<i32>,
}

impl DFlashForward {
    /// Build for one device. `ctx_tokens` bounds one injection call (the
    /// runtime passes its prefill chunk); the query capacity is the
    /// trained block size from the config.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        target_weights: &DeviceWeights,
        weights: DFlashWeights,
        ctx_tokens: usize,
        aux_slack_rows: usize,
        vocab: usize,
    ) -> Result<Self, DFlashError> {
        let config = weights.config.clone();
        let hidden = config.hidden_size as usize;
        let q_dim = config.q_dim() as usize;
        let kv_dim = config.kv_dim() as usize;
        let ffn = config.ffn_size as usize;
        let q_tokens = config.block_size as usize;
        let fc_in = config.fc_input_dim() as usize;
        let heads = config.num_q_heads as usize;
        let kv_heads = config.num_kv_heads as usize;
        let head_dim = config.head_dim as usize;

        let attn = AttentionKernels::new(ctx, heads, kv_heads, head_dim)?;
        let ops = LayerOpsKernels::new(ctx)?;
        let dec = AttnDecodeScratch::new(stream, heads, head_dim)?;

        let geom = |input: usize, output: usize, max_tokens: usize| LmHeadGeometry {
            hidden: input,
            vocab: output,
            max_tokens,
        };
        let big = ctx_tokens.max(q_tokens);
        // `LmHeadKernels::forward` checks its buffers at the geometry's full
        // capacity, so every call below passes whole buffers — which is why
        // the K/V projection needs two geometries (context chunks and query
        // blocks read differently-sized inputs) and why `aux` carries
        // `aux_slack_rows` of slack: a verify-pass injection reads from a row
        // offset, and the capacity-length view from that offset must stay in
        // bounds.
        let fc_proj = LmHeadKernels::new(ctx, geom(fc_in, hidden, big))?;
        let q_proj = LmHeadKernels::new(ctx, geom(hidden, q_dim, q_tokens))?;
        let kv_proj_ctx = LmHeadKernels::new(ctx, geom(hidden, kv_dim, big))?;
        let kv_proj_q = LmHeadKernels::new(ctx, geom(hidden, kv_dim, q_tokens))?;
        let o_proj = LmHeadKernels::new(ctx, geom(q_dim, hidden, q_tokens))?;
        let gu_proj = LmHeadKernels::new(ctx, geom(hidden, ffn, q_tokens))?;
        let down_proj = LmHeadKernels::new(ctx, geom(ffn, hidden, q_tokens))?;
        let lm_head = LmHeadKernels::new(ctx, geom(hidden, vocab, q_tokens))?;

        let alias = |role: Role| -> Result<ManuallyDrop<CudaSlice<u8>>, DFlashError> {
            let placement = target_weights
                .find(role, None)
                .ok_or_else(|| DFlashError::TargetWeight(format!("{role} not resident")))?;
            if placement.ggml_type != GgmlType::Q8_0 {
                return Err(DFlashError::TargetWeight(format!(
                    "{role} is {:?}, not Q8_0",
                    placement.ggml_type
                )));
            }
            let alias = target_weights
                .bytes_of(stream, role, None)
                .ok_or_else(|| DFlashError::TargetWeight(format!("{role} not resident")))?;
            // SAFETY: sealed in a ManuallyDrop stored in Self and never taken
            // out; Self is used only while the target DeviceWeights lives —
            // the same contract block/mtp.rs's aliases rely on.
            Ok(ManuallyDrop::new(unsafe { alias.into_aliasing_slice() }))
        };
        let w_token_embd = alias(Role::TokenEmbedding)?;
        let w_lm_head = alias(Role::LmHead)?;

        let ptx = compile(DFLASH_GLUE_SRC, "dflash_glue").map_err(DFlashError::Compile)?;
        let module = ctx.load_module(ptx)?;
        let embed_fn = module.load_function("dflash_embed_q8_0")?;
        let tap_fn = module.load_function("dflash_tap")?;
        let taps = config.target_layers.len();

        Ok(Self {
            attn,
            ops,
            dec,
            fc_proj,
            q_proj,
            kv_proj_ctx,
            kv_proj_q,
            o_proj,
            gu_proj,
            down_proj,
            lm_head,
            w_token_embd,
            w_lm_head,
            embed_fn,
            tap_fn,
            aux: stream.alloc_zeros::<f32>((big + aux_slack_rows) * taps * hidden)?,
            ctx_g: stream.alloc_zeros::<f32>(big * hidden)?,
            ctx_g_normed: stream.alloc_zeros::<f32>(big * hidden)?,
            ctx_k: stream.alloc_zeros::<f32>(big * kv_dim)?,
            ctx_k_normed: stream.alloc_zeros::<f32>(big * kv_dim)?,
            ctx_k_roped: stream.alloc_zeros::<f32>(big * kv_dim)?,
            ctx_v: stream.alloc_zeros::<f32>(big * kv_dim)?,
            q_tokens,
            d_tokens: stream.alloc_zeros::<i32>(q_tokens)?,
            x: stream.alloc_zeros::<f32>(q_tokens * hidden)?,
            normed: stream.alloc_zeros::<f32>(q_tokens * hidden)?,
            q_buf: stream.alloc_zeros::<f32>(q_tokens * q_dim)?,
            q_normed: stream.alloc_zeros::<f32>(q_tokens * q_dim)?,
            q_roped: stream.alloc_zeros::<f32>(q_tokens * q_dim)?,
            k_buf: stream.alloc_zeros::<f32>(q_tokens * kv_dim)?,
            k_normed: stream.alloc_zeros::<f32>(q_tokens * kv_dim)?,
            k_roped: stream.alloc_zeros::<f32>(q_tokens * kv_dim)?,
            v_buf: stream.alloc_zeros::<f32>(q_tokens * kv_dim)?,
            attn_out: stream.alloc_zeros::<f32>(q_tokens * q_dim)?,
            proj_out: stream.alloc_zeros::<f32>(q_tokens * hidden)?,
            gate: stream.alloc_zeros::<f32>(q_tokens * ffn)?,
            up: stream.alloc_zeros::<f32>(q_tokens * ffn)?,
            act: stream.alloc_zeros::<f32>(q_tokens * ffn)?,
            logits: stream.alloc_zeros::<f32>(q_tokens * vocab)?,
            argmax_values: stream.alloc_zeros::<f32>(q_tokens * ARGMAX_BLOCKS)?,
            argmax_indices: stream.alloc_zeros::<i32>(q_tokens * ARGMAX_BLOCKS)?,
            argmax_out: stream.alloc_zeros::<i32>(q_tokens)?,
            argmax_probs: stream.alloc_zeros::<f32>(q_tokens)?,
            positions: stream.alloc_zeros::<i32>(2 + q_tokens)?,
            pos_host: vec![0; 2 + q_tokens],
            config,
            weights,
        })
    }

    /// The drafter's configuration.
    pub fn config(&self) -> &DFlashConfig {
        &self.config
    }

    /// A fresh per-sequence draft cache covering `max_seq` positions.
    pub fn new_cache(
        &self,
        stream: &Arc<CudaStream>,
        max_seq: usize,
    ) -> Result<DFlashDraftCache, DFlashError> {
        let kv_dim = self.config.kv_dim() as usize;
        let mut layers = Vec::with_capacity(self.config.num_layers as usize);
        for _ in 0..self.config.num_layers {
            layers.push(
                KvCache::with_kv_dim(stream, kv_dim, max_seq)
                    .map_err(|e| DFlashError::TargetWeight(e.to_string()))?,
            );
        }
        Ok(DFlashDraftCache { layers })
    }

    /// Stage one target-layer tap into the concatenated feature rows.
    ///
    /// `src` is the target's residual stream, `[rows][hidden]` — the buffer
    /// its `Moe` waypoint hands out after layer `target_layers[tap] - 1`
    /// (the stream entering layer `target_layers[tap]`, which is the tap
    /// llama.cpp's `t_layer_inp` captures). Call once per tap per target
    /// pass; [`Self::inject_context`] then consumes any row range of the
    /// staged block.
    pub fn stage_tap(
        &mut self,
        stream: &Arc<CudaStream>,
        tap: usize,
        src: &CudaSlice<f32>,
        rows: usize,
    ) -> Result<(), DFlashError> {
        let taps = self.config.target_layers.len();
        let hidden = self.config.hidden_size as usize;
        if tap >= taps || rows * taps * hidden > self.aux.len() {
            return Err(DFlashError::Shape {
                what: "tap staging",
                expected: self.aux.len() / (taps * hidden),
                got: rows,
            });
        }
        debug_assert!(rows > 0);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (EMBED_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let hidden_i = hidden as i32;
        let taps_i = taps as i32;
        let tap_i = tap as i32;
        let rows_i = rows as i32;
        let mut builder = stream.launch_builder(&self.tap_fn);
        builder
            .arg(src)
            .arg(&mut self.aux)
            .arg(&hidden_i)
            .arg(&taps_i)
            .arg(&tap_i)
            .arg(&rows_i);
        // SAFETY: the grid covers exactly `rows` rows; the destination index
        // `(row * taps + tap) * hidden + j` is bounded by the length check
        // above, and `src` is the caller's `[rows][hidden]` waypoint buffer.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// Project `c` positions of staged target features into `cache`,
    /// starting at absolute position `start`, reading staged rows
    /// `aux_row..aux_row + c`.
    ///
    /// The staged rows are the target's residual stream at each configured
    /// tap, concatenated in tap order (what [`Self::stage_tap`] builds).
    /// This is the only writer of *real* context into the draft cache; the
    /// query pass's own K/V at these slots (from an earlier,
    /// partially-rejected block) are overwritten here.
    pub fn inject_context(
        &mut self,
        stream: &Arc<CudaStream>,
        aux_row: usize,
        c: usize,
        cache: &mut DFlashDraftCache,
        start: usize,
    ) -> Result<(), DFlashError> {
        if c == 0 {
            return Ok(());
        }
        let taps = self.config.target_layers.len();
        let hidden = self.config.hidden_size as usize;
        let fc_in = taps * hidden;
        if (aux_row + c) * fc_in > self.aux.len() {
            return Err(DFlashError::Shape {
                what: "injection chunk",
                expected: self.aux.len() / fc_in,
                got: aux_row + c,
            });
        }
        // Capacity-length view from the row offset — the GEMM checks its
        // input at full capacity and reads only `c` rows of it; the slack
        // rows allocated past `big` keep this in bounds for every verify
        // offset.
        let cap = self.ctx_g.len() / self.config.hidden_size as usize;
        if (aux_row + cap) * fc_in > self.aux.len() {
            return Err(DFlashError::Shape {
                what: "injection offset",
                expected: self.aux.len() / fc_in - cap,
                got: aux_row,
            });
        }
        let fused = unsafe { subslice(stream, &self.aux, aux_row * fc_in, cap * fc_in) };
        let kv_dim = self.config.kv_dim() as usize;
        let kv_heads = self.config.num_kv_heads as usize;
        let head_dim = self.config.head_dim as usize;
        let eps = self.config.rms_eps;

        // fc, then the post-fc RMSNorm (llama.cpp's `output_norm_enc`).
        self.fc_proj.forward(
            stream,
            HeadTensor::q8_0(&self.weights.fc),
            &fused,
            c,
            &mut self.ctx_g,
        )?;
        {
            let src = unsafe { subslice(stream, &self.ctx_g, 0, c * hidden) };
            let mut dst = unsafe { subslice(stream, &self.ctx_g_normed, 0, c * hidden) };
            self.ops.rms_norm(
                stream,
                &src,
                &self.weights.enc_norm,
                &mut dst,
                c,
                hidden,
                eps,
            )?;
        }

        stream.memcpy_htod(&[start as i32], &mut self.positions)?;
        let pos0 = unsafe { subslice(stream, &self.positions, 0, 1) };

        for l in 0..self.weights.layers.len() {
            let layer = &self.weights.layers[l];
            self.kv_proj_ctx.forward(
                stream,
                HeadTensor::q8_0(&layer.wk),
                &self.ctx_g_normed,
                c,
                &mut self.ctx_k,
            )?;
            self.kv_proj_ctx.forward(
                stream,
                HeadTensor::q8_0(&layer.wv),
                &self.ctx_g_normed,
                c,
                &mut self.ctx_v,
            )?;
            {
                let src = unsafe { subslice(stream, &self.ctx_k, 0, c * kv_dim) };
                let mut dst = unsafe { subslice(stream, &self.ctx_k_normed, 0, c * kv_dim) };
                self.ops.rms_norm(
                    stream,
                    &src,
                    &layer.k_norm,
                    &mut dst,
                    c * kv_heads,
                    head_dim,
                    eps,
                )?;
            }
            {
                let src = unsafe { subslice(stream, &self.ctx_k_normed, 0, c * kv_dim) };
                let mut dst = unsafe { subslice(stream, &self.ctx_k_roped, 0, c * kv_dim) };
                self.attn.rope(
                    stream,
                    &src,
                    &mut dst,
                    c,
                    kv_heads,
                    head_dim,
                    &pos0,
                    self.config.rope_theta,
                )?;
            }
            let k = unsafe { subslice(stream, &self.ctx_k_roped, 0, c * kv_dim) };
            let v = unsafe { subslice(stream, &self.ctx_v, 0, c * kv_dim) };
            let max_seq = cache.layers[l].max_seq();
            let (kc, vc) = cache.layers[l].kv_mut();
            self.attn
                .append_kv(stream, &k, &v, kc, vc, c, max_seq, &pos0)?;
        }
        Ok(())
    }

    /// One query pass: draft `n` tokens for a sequence whose context is
    /// injected through position `position - 1`, with `id_last` (the last
    /// committed/emitted token) at `position`.
    ///
    /// `p_min > 0` truncates the block at the first drafted position whose
    /// softmax probability (under the drafter's own head) falls below it —
    /// llama.cpp's greedy confidence gate. Zero skips the probability
    /// reduction entirely.
    ///
    /// Returns the drafted ids. Proposals only, for the target's batched
    /// verify — nothing here can change what is emitted.
    #[allow(clippy::too_many_arguments)]
    pub fn draft(
        &mut self,
        stream: &Arc<CudaStream>,
        id_last: i32,
        n: usize,
        cache: &mut DFlashDraftCache,
        position: usize,
        p_min: f32,
    ) -> Result<Vec<i32>, DFlashError> {
        let t = n + 1;
        if n == 0 || t > self.q_tokens {
            return Err(DFlashError::Shape {
                what: "draft block",
                expected: self.q_tokens - 1,
                got: n,
            });
        }
        let hidden = self.config.hidden_size as usize;
        let q_dim = self.config.q_dim() as usize;
        let kv_dim = self.config.kv_dim() as usize;
        let heads = self.config.num_q_heads as usize;
        let kv_heads = self.config.num_kv_heads as usize;
        let head_dim = self.config.head_dim as usize;
        let ffn = self.config.ffn_size as usize;
        let eps = self.config.rms_eps;
        let window = self.config.sliding_window as usize;

        // Base position, then one deepest-key scalar per query row. All
        // rows see through the block's end (non-causal); a sliding layer
        // additionally starts its K/V view at the row's window start, and
        // its scalar is relative to that base.
        let end = position + t; // one past the block's last position
        self.pos_host[0] = position as i32;
        for j in 0..t {
            let p = position + j;
            let swa_start = (p + 1).saturating_sub(window);
            self.pos_host[1 + j] = (end - 1 - swa_start) as i32;
        }
        // Full (non-sliding) layers share one deepest-key scalar at base 0.
        self.pos_host[1 + t] = (end - 1) as i32;
        {
            let mut dst = self.positions.slice_mut(0..2 + t);
            stream.memcpy_htod(&self.pos_host[..2 + t], &mut dst)?;
        }
        let pos0 = unsafe { subslice(stream, &self.positions, 0, 1) };

        // Token rows: [id_last, MASK × n].
        let mut ids = vec![self.config.mask_token_id as i32; t];
        ids[0] = id_last;
        stream.memcpy_htod(&ids, &mut self.d_tokens)?;
        {
            let cfg = LaunchConfig {
                grid_dim: (t as u32, 1, 1),
                block_dim: (EMBED_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            let hidden_i = hidden as i32;
            let t_i = t as i32;
            let mut builder = stream.launch_builder(&self.embed_fn);
            builder
                .arg(&*self.w_token_embd)
                .arg(&self.d_tokens)
                .arg(&mut self.x)
                .arg(&hidden_i)
                .arg(&t_i);
            // SAFETY: grid covers exactly `t` rows; `d_tokens` and `x` hold
            // at least `t` ids / `t * hidden` floats by construction, and
            // the token table was checked resident Q8_0.
            unsafe { builder.launch(cfg) }?;
        }

        for l in 0..self.weights.layers.len() {
            let layer = &self.weights.layers[l];
            let sliding = self.config.swa_pattern[l];

            {
                let x = unsafe { subslice(stream, &self.x, 0, t * hidden) };
                let mut normed = unsafe { subslice(stream, &self.normed, 0, t * hidden) };
                self.ops
                    .rms_norm(stream, &x, &layer.attn_norm, &mut normed, t, hidden, eps)?;
            }

            self.q_proj.forward(
                stream,
                HeadTensor::q8_0(&layer.wq),
                &self.normed,
                t,
                &mut self.q_buf,
            )?;
            self.kv_proj_q.forward(
                stream,
                HeadTensor::q8_0(&layer.wk),
                &self.normed,
                t,
                &mut self.k_buf,
            )?;
            self.kv_proj_q.forward(
                stream,
                HeadTensor::q8_0(&layer.wv),
                &self.normed,
                t,
                &mut self.v_buf,
            )?;

            {
                let src = unsafe { subslice(stream, &self.q_buf, 0, t * q_dim) };
                let mut dst = unsafe { subslice(stream, &self.q_normed, 0, t * q_dim) };
                self.ops.rms_norm(
                    stream,
                    &src,
                    &layer.q_norm,
                    &mut dst,
                    t * heads,
                    head_dim,
                    eps,
                )?;
            }
            {
                let src = unsafe { subslice(stream, &self.k_buf, 0, t * kv_dim) };
                let mut dst = unsafe { subslice(stream, &self.k_normed, 0, t * kv_dim) };
                self.ops.rms_norm(
                    stream,
                    &src,
                    &layer.k_norm,
                    &mut dst,
                    t * kv_heads,
                    head_dim,
                    eps,
                )?;
            }
            {
                let src = unsafe { subslice(stream, &self.q_normed, 0, t * q_dim) };
                let mut dst = unsafe { subslice(stream, &self.q_roped, 0, t * q_dim) };
                self.attn.rope(
                    stream,
                    &src,
                    &mut dst,
                    t,
                    heads,
                    head_dim,
                    &pos0,
                    self.config.rope_theta,
                )?;
            }
            {
                let src = unsafe { subslice(stream, &self.k_normed, 0, t * kv_dim) };
                let mut dst = unsafe { subslice(stream, &self.k_roped, 0, t * kv_dim) };
                self.attn.rope(
                    stream,
                    &src,
                    &mut dst,
                    t,
                    kv_heads,
                    head_dim,
                    &pos0,
                    self.config.rope_theta,
                )?;
            }

            // The query block's own K/V join the cache at its positions so
            // rows can attend to each other; the next injection overwrites
            // these slots with real context.
            {
                let k = unsafe { subslice(stream, &self.k_roped, 0, t * kv_dim) };
                let v = unsafe { subslice(stream, &self.v_buf, 0, t * kv_dim) };
                let max_seq = cache.layers[l].max_seq();
                let (kc, vc) = cache.layers[l].kv_mut();
                self.attn
                    .append_kv(stream, &k, &v, kc, vc, t, max_seq, &pos0)?;
            }

            // Per-row non-causal attention over `[window start, block end)`.
            // A sliding layer offsets the K/V base to the row's window
            // start and uses the row's own depth scalar (relative to that
            // base); a full layer starts at 0 and every row shares the
            // block-end scalar.
            for j in 0..t {
                let p = position + j;
                let swa_start = if sliding {
                    (p + 1).saturating_sub(window)
                } else {
                    0
                };
                let max_seq = cache.layers[l].max_seq();
                let q = unsafe { subslice(stream, &self.q_roped, j * q_dim, q_dim) };
                let mut out = unsafe { subslice(stream, &self.attn_out, j * q_dim, q_dim) };
                let k = unsafe {
                    subslice(
                        stream,
                        cache.layers[l].keys(),
                        swa_start * kv_dim,
                        (max_seq - swa_start) * kv_dim,
                    )
                };
                let v = unsafe {
                    subslice(
                        stream,
                        cache.layers[l].values(),
                        swa_start * kv_dim,
                        (max_seq - swa_start) * kv_dim,
                    )
                };
                let depth_index = if sliding { 1 + j } else { 1 + t };
                let depth = unsafe { subslice(stream, &self.positions, depth_index, 1) };
                self.attn.forward(
                    stream,
                    &mut self.dec,
                    &q,
                    &k,
                    &v,
                    &mut out,
                    1,
                    max_seq - swa_start,
                    end - swa_start,
                    &depth,
                )?;
            }

            // Output projection, residual add.
            {
                self.o_proj.forward(
                    stream,
                    HeadTensor::q8_0(&layer.wo),
                    &self.attn_out,
                    t,
                    &mut self.proj_out,
                )?;
                let o = unsafe { subslice(stream, &self.proj_out, 0, t * hidden) };
                let xa = unsafe { subslice(stream, &self.x, 0, t * hidden) };
                let mut xb = unsafe { subslice(stream, &self.x, 0, t * hidden) };
                self.ops.add(stream, &o, &xa, &mut xb, t * hidden)?;
            }

            // SwiGLU FFN, residual add.
            {
                let x = unsafe { subslice(stream, &self.x, 0, t * hidden) };
                let mut normed = unsafe { subslice(stream, &self.normed, 0, t * hidden) };
                self.ops
                    .rms_norm(stream, &x, &layer.ffn_norm, &mut normed, t, hidden, eps)?;
            }
            {
                self.gu_proj.forward(
                    stream,
                    HeadTensor::q8_0(&layer.w_gate),
                    &self.normed,
                    t,
                    &mut self.gate,
                )?;
                self.gu_proj.forward(
                    stream,
                    HeadTensor::q8_0(&layer.w_up),
                    &self.normed,
                    t,
                    &mut self.up,
                )?;
                let g = unsafe { subslice(stream, &self.gate, 0, t * ffn) };
                let u = unsafe { subslice(stream, &self.up, 0, t * ffn) };
                let mut act = unsafe { subslice(stream, &self.act, 0, t * ffn) };
                self.ops.swiglu(stream, &g, &u, &mut act, t * ffn)?;
            }
            {
                self.down_proj.forward(
                    stream,
                    HeadTensor::q8_0(&layer.w_down),
                    &self.act,
                    t,
                    &mut self.proj_out,
                )?;
                let d = unsafe { subslice(stream, &self.proj_out, 0, t * hidden) };
                let xa = unsafe { subslice(stream, &self.x, 0, t * hidden) };
                let mut xb = unsafe { subslice(stream, &self.x, 0, t * hidden) };
                self.ops.add(stream, &d, &xa, &mut xb, t * hidden)?;
            }
        }

        // Final norm, the (target's) LM head over the mask rows only, and
        // one argmax per drafted position. Row 0 (the committed token's
        // slot) predicts nothing this path uses.
        {
            let x = unsafe { subslice(stream, &self.x, 0, t * hidden) };
            let mut normed = unsafe { subslice(stream, &self.normed, 0, t * hidden) };
            self.ops.rms_norm(
                stream,
                &x,
                &self.weights.output_norm,
                &mut normed,
                t,
                hidden,
                eps,
            )?;
        }
        // All `t` rows through the head (the row tile makes n vs n+1 the
        // same number of weight passes); the drafted ids are the argmaxes of
        // the mask rows `1..=n`. Row 0's prediction is unused.
        let vocab = self.lm_head.geometry().vocab;
        self.lm_head.forward(
            stream,
            HeadTensor::q8_0(&self.w_lm_head),
            &self.normed,
            t,
            &mut self.logits,
        )?;
        for i in 0..n {
            let row = unsafe { subslice(stream, &self.logits, (1 + i) * vocab, vocab) };
            let mut values = unsafe {
                subslice(
                    stream,
                    &self.argmax_values,
                    i * ARGMAX_BLOCKS,
                    ARGMAX_BLOCKS,
                )
            };
            let mut indices = unsafe {
                subslice(
                    stream,
                    &self.argmax_indices,
                    i * ARGMAX_BLOCKS,
                    ARGMAX_BLOCKS,
                )
            };
            let mut out_one = unsafe { subslice(stream, &self.argmax_out, i, 1) };
            self.lm_head
                .argmax(stream, &row, vocab, &mut values, &mut indices, &mut out_one)?;
            if p_min > 0.0 {
                let mut prob = unsafe { subslice(stream, &self.argmax_probs, i, 1) };
                self.lm_head.argmax_prob(stream, &row, vocab, &mut prob)?;
            }
        }
        let host = stream.clone_dtoh(&self.argmax_out)?;
        let mut ids = host[..n].to_vec();
        if p_min > 0.0 {
            let probs = stream.clone_dtoh(&self.argmax_probs)?;
            stream.synchronize()?;
            if let Some(cut) = probs[..n].iter().position(|&p| p < p_min) {
                ids.truncate(cut);
            }
        } else {
            stream.synchronize()?;
        }
        Ok(ids)
    }
}
