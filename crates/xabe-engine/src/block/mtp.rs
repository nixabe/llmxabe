//! The MTP (multi-token-prediction) draft head: block 40 of the GGUF.
//!
//! The reference is llama.cpp's `graph_mtp`
//! (`src/models/qwen35moe.cpp:553-742`): `hnorm(h) ++ enorm(embed(tok))`,
//! concatenated and projected through `eh_proj` down to `hidden`, then one
//! ordinary dense-attention + MoE decoder block — this GGUF's block 40,
//! loaded exactly like the other ten attention layers (see
//! [`xabe_model::weights::WeightSchema::with_mtp`]) — then
//! `shared_head_norm` and the same LM head the main trunk uses. This file's
//! GGUF ships no `nextn.embed_tokens` / `nextn.shared_head_head`, so both are
//! shared with the main trunk; `crates/xabe-model/src/weights.rs` only
//! defines the four `Mtp*` roles that are actually present
//! (`gguf-dump`-confirmed against
//! `Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf`).
//!
//! # The self-referential chain
//!
//! Only the *first* draft position can use the target model's own hidden
//! state as `h` — the target has not run the second or third draft position
//! yet. Every later draft step feeds back the previous step's own emitted
//! `h_nextn` (post `shared_head_norm`, the same row the LM head reads)
//! instead. [`MtpBlock::forward`] returns that row for exactly this reason;
//! the caller is what decides which `h` to feed the next call.
//!
//! # Two geometries, one set of MoE weights
//!
//! Drafting is single-token, but the draft head's own attention layer needs
//! its **own** KV cache filled over the whole prompt before it can draft
//! anything — an empty draft KV does not error, it drafts garbage, which is
//! exactly the kind of quietly-wrong failure `AGENTS.md` warns about. That is
//! a wide, `tokens = prompt_len`-shaped pass (catch-up), while drafting
//! itself is `tokens = 1` — [`crate::forward::Forward`]'s prefill/decode
//! split again. [`MtpBlock::reshape`] builds the second geometry sharing the
//! first's upload of block 40's ~0.75 GiB of MoE weights, the same way
//! [`crate::forward::Forward::reshape`] shares the main trunk's.

use std::mem::ManuallyDrop;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig,
    PushKernelArg,
};

use xabe_cuda::kernels::compile;
use xabe_cuda::kernels::layer_ops::{LayerOpsError, LayerOpsKernels};
use xabe_cuda::kernels::lm_head::{ARGMAX_BLOCKS, LmHeadError, LmHeadGeometry, LmHeadKernels};
use xabe_cuda::kernels::moe::{ExpertQuant, QuantTensor};
use xabe_gguf::{GgmlType, GgufFile};
use xabe_model::config::ModelConfig;
use xabe_model::weights::{Directory, Role};

use crate::block::attention::{
    AttentionBlockError, AttentionKernelSet, AttnScratch, GatedAttentionBlock, KvCache,
};
use crate::block::moe::{MoeBlock, MoeBlockError, MoeLayerWeights};
use crate::weights::DeviceWeights;

const EMBED_THREADS: u32 = 256;

/// `out[t][j] = dequantize(token_embd)[ids[t]][j]`, and
/// `concat[t][j] = j < hidden ? e_norm[t][j] : h_norm[t][j - hidden]` —
/// `ggml_concat(.., dim = 0)`, e_norm first. One module because NVRTC
/// compiles from a string with no include path and no cross-string linking,
/// same reason `forward.rs`'s `EMBED_SRC` is inline.
const MTP_GLUE_SRC: &str = r#"
extern "C" {

__device__ __forceinline__ float load_half_le(const unsigned char* p) {
    unsigned short bits = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(bits));
    return f;
}

// grid: (n_tokens). block: EMBED_THREADS, striding the row.
__global__ void mtp_embed_q8_0(
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

// grid: (n_tokens). block: EMBED_THREADS, striding 2*hidden.
__global__ void mtp_concat_eh(
    const float* __restrict__ e_norm,
    const float* __restrict__ h_norm,
    float* __restrict__ out,
    int hidden,
    int n_tokens
) {
    int t = blockIdx.x;
    if (t >= n_tokens) return;
    const float* e = e_norm + (long long)t * hidden;
    const float* h = h_norm + (long long)t * hidden;
    float* dst = out + (long long)t * 2 * hidden;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        dst[j] = e[j];
        dst[hidden + j] = h[j];
    }
}

}
"#;

/// Something went wrong building or running the MTP draft head.
#[derive(Debug)]
pub enum MtpBlockError {
    /// NVRTC rejected the glue source, or the module failed to load.
    Compile(String),
    /// The driver failed.
    Driver(DriverError),
    /// The dense-attention half of block 40 failed.
    Attention(AttentionBlockError),
    /// Block 40's MoE failed.
    Moe(MoeBlockError),
    /// A norm or the concat failed.
    LayerOps(LayerOpsError),
    /// `eh_proj` or the LM head GEMM failed.
    LmHead(LmHeadError),
    /// A tensor the draft head needs is not resident.
    MissingWeight { role: Role, layer: Option<u32> },
    /// A resident tensor is not the element type this block unpacks.
    WrongQuant {
        role: Role,
        layer: Option<u32>,
        found: GgmlType,
    },
    /// The batch is not the size this instance was built for.
    WrongTokenCount { expected: usize, got: usize },
}

impl std::fmt::Display for MtpBlockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compile(m) => write!(f, "MTP glue kernel compilation failed: {m}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::Attention(e) => write!(f, "{e}"),
            Self::Moe(e) => write!(f, "{e}"),
            Self::LayerOps(e) => write!(f, "{e}"),
            Self::LmHead(e) => write!(f, "{e}"),
            Self::MissingWeight { role, layer } => match layer {
                Some(l) => write!(f, "block {l} has no resident `{role}`"),
                None => write!(f, "`{role}` is not resident"),
            },
            Self::WrongQuant { role, layer, found } => {
                let where_ = match layer {
                    Some(l) => format!("blk.{l}."),
                    None => String::new(),
                };
                write!(f, "`{where_}{role}` is {}, expected Q8_0", found.name())
            }
            Self::WrongTokenCount { expected, got } => write!(
                f,
                "this MTP pass was built for {expected} tokens and was given {got}",
            ),
        }
    }
}

impl std::error::Error for MtpBlockError {}

impl From<DriverError> for MtpBlockError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}
impl From<AttentionBlockError> for MtpBlockError {
    fn from(e: AttentionBlockError) -> Self {
        Self::Attention(e)
    }
}
impl From<MoeBlockError> for MtpBlockError {
    fn from(e: MoeBlockError) -> Self {
        Self::Moe(e)
    }
}
impl From<LayerOpsError> for MtpBlockError {
    fn from(e: LayerOpsError) -> Self {
        Self::LayerOps(e)
    }
}
impl From<LmHeadError> for MtpBlockError {
    fn from(e: LmHeadError) -> Self {
        Self::LmHead(e)
    }
}

/// Block 40, wired up as a self-contained one-layer forward pass.
///
/// Owns its own copy of block 40's attention weights (cheap — about 20 MiB,
/// the same duplication [`GatedAttentionBlock`] already accepts for the other
/// ten layers) and, unless [`Self::reshape`] shares them, its own upload of
/// block 40's MoE weights (~0.75 GiB, the expensive part).
pub struct MtpBlock {
    hidden: usize,
    tokens: usize,
    rms_eps: f32,

    embed_fn: CudaFunction,
    concat_fn: CudaFunction,
    layer_ops: LayerOpsKernels,
    attn: GatedAttentionBlock,
    attn_scratch: AttnScratch,
    /// A fork marker [`Self::forward_batch_decode`] hands the attention
    /// block. Drafting batches are 1-3 tokens, so the per-sequence chains
    /// run serially on the main stream (no side lanes) — the event is
    /// required by the signature, recorded and never waited on.
    batch_fork: CudaEvent,
    moe: MoeBlock,
    moe_weights: Arc<MoeLayerWeights>,
    eh_proj: LmHeadKernels,

    /// `None` for the wide catch-up geometry: nothing samples during
    /// catch-up, so building a `max_tokens`-wide LM head and logits buffer
    /// for it would allocate gigabytes that are never read.
    lm_head: Option<LmHeadKernels>,

    w_token_embd: ManuallyDrop<CudaSlice<u8>>,
    w_lm_head: ManuallyDrop<CudaSlice<u8>>,
    w_eh_proj: ManuallyDrop<CudaSlice<u8>>,
    w_enorm: ManuallyDrop<CudaSlice<f32>>,
    w_hnorm: ManuallyDrop<CudaSlice<f32>>,
    w_shared_head_norm: ManuallyDrop<CudaSlice<f32>>,

    d_tokens: CudaSlice<i32>,
    tok_embd: CudaSlice<f32>,
    e_norm: CudaSlice<f32>,
    h_norm: CudaSlice<f32>,
    concat: CudaSlice<f32>,
    eh_out: CudaSlice<f32>,
    attn_out: CudaSlice<f32>,
    ffn_out: CudaSlice<f32>,
    post_ffn: CudaSlice<f32>,
    /// `h_nextn`: `shared_head_norm(post_ffn)`, `[tokens][hidden]`. What the
    /// caller feeds back in as the next step's `h` — see the module docs.
    h_nextn: CudaSlice<f32>,

    logits: Option<CudaSlice<f32>>,
    argmax_values: Option<CudaSlice<f32>>,
    argmax_indices: Option<CudaSlice<i32>>,
    argmax_out: Option<CudaSlice<i32>>,
    argmax_probs: Option<CudaSlice<f32>>,
}

impl MtpBlock {
    /// Build the draft head for exactly `tokens` positions, uploading block
    /// 40's MoE weights fresh.
    ///
    /// `needs_lm_head` should be `false` for a catch-up pass over a whole
    /// prompt (nothing samples during catch-up) and `true` for a one-token
    /// draft step.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        file: &GgufFile,
        directory: &Directory<'_>,
        weights: &DeviceWeights,
        config: &ModelConfig,
        tokens: usize,
        rms_eps: f32,
        rope_theta: f32,
        needs_lm_head: bool,
    ) -> Result<Self, MtpBlockError> {
        let mtp_layer = config.num_layers;
        let moe_geometry =
            MoeBlock::geometry_for(config, crate::forward::moe_block_size(tokens), tokens);
        let moe_weights = Arc::new(MoeLayerWeights::upload(
            stream,
            file,
            directory,
            mtp_layer,
            &moe_geometry,
        )?);
        Self::build(
            ctx,
            stream,
            weights,
            config,
            tokens,
            rms_eps,
            rope_theta,
            needs_lm_head,
            moe_weights,
        )
    }

    /// Build a second geometry over the **same** uploaded MoE weights — the
    /// draft-step-shaped pass over a prompt that has already paid for
    /// catch-up's upload. See the module docs.
    #[allow(clippy::too_many_arguments)]
    pub fn reshape(
        &self,
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        weights: &DeviceWeights,
        config: &ModelConfig,
        tokens: usize,
        rms_eps: f32,
        rope_theta: f32,
        needs_lm_head: bool,
    ) -> Result<Self, MtpBlockError> {
        Self::build(
            ctx,
            stream,
            weights,
            config,
            tokens,
            rms_eps,
            rope_theta,
            needs_lm_head,
            Arc::clone(&self.moe_weights),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        weights: &DeviceWeights,
        config: &ModelConfig,
        tokens: usize,
        rms_eps: f32,
        rope_theta: f32,
        needs_lm_head: bool,
        moe_weights: Arc<MoeLayerWeights>,
    ) -> Result<Self, MtpBlockError> {
        let hidden = config.hidden_size as usize;
        let vocab = config.vocab_size as usize;
        let mtp_layer = config.num_layers;

        let ptx = compile(MTP_GLUE_SRC, "mtp_glue").map_err(MtpBlockError::Compile)?;
        let module = ctx.load_module(ptx)?;
        let embed_fn = module.load_function("mtp_embed_q8_0")?;
        let concat_fn = module.load_function("mtp_concat_eh")?;
        let layer_ops = LayerOpsKernels::new(ctx)?;

        let batch_fork = ctx.new_event(None)?;
        let attn_kernels = Arc::new(AttentionKernelSet::new(ctx, config, tokens)?);
        let attn = GatedAttentionBlock::new(
            attn_kernels,
            stream,
            weights,
            config,
            mtp_layer,
            tokens,
            rms_eps,
            rope_theta,
        )?;
        let attn_scratch = AttnScratch::new(stream, config, tokens)?;

        let moe_geometry =
            MoeBlock::geometry_for(config, crate::forward::moe_block_size(tokens), tokens);
        let moe = MoeBlock::new(ctx, stream, moe_geometry, rms_eps)?;

        let w_token_embd = alias_q8_0(weights, stream, Role::TokenEmbedding, None)?;
        let w_lm_head = alias_q8_0(weights, stream, Role::LmHead, None)?;
        let w_eh_proj = alias_q8_0(weights, stream, Role::MtpEhProj, Some(mtp_layer))?;
        let w_enorm = alias_f32(weights, stream, Role::MtpENorm, Some(mtp_layer))?;
        let w_hnorm = alias_f32(weights, stream, Role::MtpHNorm, Some(mtp_layer))?;
        let w_shared_head_norm =
            alias_f32(weights, stream, Role::MtpSharedHeadNorm, Some(mtp_layer))?;

        let eh_proj = LmHeadKernels::new(
            ctx,
            LmHeadGeometry {
                hidden: 2 * hidden,
                vocab: hidden,
                max_tokens: tokens,
            },
        )?;

        let (lm_head, logits, argmax_values, argmax_indices, argmax_out, argmax_probs) =
            if needs_lm_head {
                let lm_head = LmHeadKernels::new(
                    ctx,
                    LmHeadGeometry {
                        hidden,
                        vocab,
                        max_tokens: tokens,
                    },
                )?;
                (
                    Some(lm_head),
                    Some(stream.alloc_zeros::<f32>(tokens * vocab)?),
                    Some(stream.alloc_zeros::<f32>(tokens * ARGMAX_BLOCKS)?),
                    Some(stream.alloc_zeros::<i32>(tokens * ARGMAX_BLOCKS)?),
                    Some(stream.alloc_zeros::<i32>(tokens)?),
                    Some(stream.alloc_zeros::<f32>(tokens)?),
                )
            } else {
                (None, None, None, None, None, None)
            };

        Ok(Self {
            hidden,
            tokens,
            rms_eps,
            embed_fn,
            concat_fn,
            layer_ops,
            attn,
            attn_scratch,
            batch_fork,
            moe,
            moe_weights,
            eh_proj,
            lm_head,
            w_token_embd,
            w_lm_head,
            w_eh_proj,
            w_enorm,
            w_hnorm,
            w_shared_head_norm,
            d_tokens: stream.alloc_zeros::<i32>(tokens)?,
            tok_embd: stream.alloc_zeros::<f32>(tokens * hidden)?,
            e_norm: stream.alloc_zeros::<f32>(tokens * hidden)?,
            h_norm: stream.alloc_zeros::<f32>(tokens * hidden)?,
            concat: stream.alloc_zeros::<f32>(tokens * 2 * hidden)?,
            eh_out: stream.alloc_zeros::<f32>(tokens * hidden)?,
            attn_out: stream.alloc_zeros::<f32>(tokens * hidden)?,
            ffn_out: stream.alloc_zeros::<f32>(tokens * hidden)?,
            post_ffn: stream.alloc_zeros::<f32>(tokens * hidden)?,
            h_nextn: stream.alloc_zeros::<f32>(tokens * hidden)?,
            logits,
            argmax_values,
            argmax_indices,
            argmax_out,
            argmax_probs,
        })
    }

    /// Positions this pass was built for.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// `h_nextn` from the most recent [`Self::forward`]: `[tokens][hidden]`,
    /// what the caller feeds back in as the next call's `h`.
    pub fn h_nextn(&self) -> &CudaSlice<f32> {
        &self.h_nextn
    }

    /// Softmax probabilities of the most recent pass's per-row argmaxes —
    /// the confidence a greedy `p_min` gate compares against. One reduction
    /// per row plus a host read; call it only when a gate is configured.
    /// Requires an instance built with the LM head.
    pub fn last_argmax_probs(
        &mut self,
        stream: &Arc<CudaStream>,
    ) -> Result<Vec<f32>, MtpBlockError> {
        let lm_head = self.lm_head.as_ref().expect("built with lm_head");
        let logits = self.logits.as_ref().expect("built with lm_head");
        let probs = self.argmax_probs.as_ref().expect("built with lm_head");
        let vocab = lm_head.geometry().vocab;
        for i in 0..self.tokens {
            let row = unsafe { crate::viewslice::subslice(stream, logits, i * vocab, vocab) };
            let mut out = unsafe { crate::viewslice::subslice(stream, probs, i, 1) };
            lm_head.argmax_prob(stream, &row, vocab, &mut out)?;
        }
        let host = stream.clone_dtoh(probs)?;
        stream.synchronize()?;
        Ok(host)
    }

    /// One MTP forward pass: `token_ids` and `h` in, `h_nextn` (and, if this
    /// instance was built with a LM head, one sampled id per position) out.
    ///
    /// `h` is `[tokens][hidden]` — the target's own `h_nextn` for the first
    /// draft position after a verify, or this block's own previous
    /// [`Self::h_nextn`] for every later position in the chain. `cache` is
    /// this draft head's own key/value cache, entirely separate from the
    /// target model's; `pos_offset` is the absolute position of `token_ids[0]`
    /// in *that* cache.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        stream: &Arc<CudaStream>,
        token_ids: &[i32],
        h: &CudaSlice<f32>,
        cache: &mut KvCache,
        pos_offset: usize,
        positions: &CudaSlice<i32>,
    ) -> Result<Option<Vec<i32>>, MtpBlockError> {
        self.embed_tokens(stream, token_ids)?;
        self.finish_forward(stream, h, cache, pos_offset, positions)
    }

    /// One MTP step for `caches.len()` independent decoding sequences at
    /// once — one drafted token per sequence, each against its own draft
    /// KV cache and position scalar. The row layout matches the target's
    /// batch decode: row `i` belongs to sequence `i` everywhere.
    ///
    /// The weight-bound projections (`eh_proj`, the attention projections,
    /// the MoE, the LM head) batch across sequences; only rope, the cache
    /// append and the attention mix loop per sequence — the same split
    /// [`GatedAttentionBlock::forward_batch_decode`] makes for the target.
    /// `positions` serves as both the cache-slot scalar and the rope
    /// position, matching the single-sequence path's `RopeSource::Scalar`
    /// (the draft head never applies an M-RoPE delta; see
    /// [`Self::finish_forward`] on why that cannot affect exactness).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_batch_decode(
        &mut self,
        stream: &Arc<CudaStream>,
        token_ids: &[i32],
        h: &CudaSlice<f32>,
        caches: &mut [&mut KvCache],
        pos_offsets: &[usize],
        positions: &[&CudaSlice<i32>],
    ) -> Result<Option<Vec<i32>>, MtpBlockError> {
        self.embed_tokens(stream, token_ids)?;
        self.mix_input(stream, h)?;

        // No side lanes: at draft widths (≤ the worker's max batch) the
        // per-sequence loop is a handful of tiny launches, and a fork/join
        // would cost more than it hides. `batch_fork` satisfies the
        // signature and is never recorded when the lane list is empty.
        let (attn, attn_scratch, eh_out, attn_out) = (
            &mut self.attn,
            &mut self.attn_scratch,
            &self.eh_out,
            &mut self.attn_out,
        );
        attn.forward_batch_decode(
            stream,
            &[],
            &self.batch_fork,
            &[],
            attn_scratch,
            eh_out,
            caches,
            pos_offsets,
            positions,
            positions,
            attn_out,
        )?;

        self.finish_output(stream)
    }

    /// Steps 1-2: upload `token_ids`, embed, `enorm` — filling
    /// `self.e_norm` for [`Self::mix_input`].
    fn embed_tokens(
        &mut self,
        stream: &Arc<CudaStream>,
        token_ids: &[i32],
    ) -> Result<(), MtpBlockError> {
        let t = self.tokens;
        if token_ids.len() != t {
            return Err(MtpBlockError::WrongTokenCount {
                expected: t,
                got: token_ids.len(),
            });
        }
        stream.memcpy_htod(token_ids, &mut self.d_tokens)?;

        // 1. `model.input_embed`, one row per drafted/prompt token.
        let cfg = LaunchConfig {
            grid_dim: (t as u32, 1, 1),
            block_dim: (EMBED_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let hidden_i32 = self.hidden as i32;
        let tokens_i32 = t as i32;
        let mut builder = stream.launch_builder(&self.embed_fn);
        builder
            .arg(&*self.w_token_embd)
            .arg(&self.d_tokens)
            .arg(&mut self.tok_embd)
            .arg(&hidden_i32)
            .arg(&tokens_i32);
        // SAFETY: grid covers exactly `t` rows, `d_tokens` and `tok_embd` are
        // both sized `t` (elements / `hidden`) by construction, and the token
        // embedding table was checked resident and Q8_0 at construction.
        unsafe { builder.launch(cfg) }?;

        // 2. `mtp_enorm`.
        self.layer_ops.rms_norm(
            stream,
            &self.tok_embd,
            &self.w_enorm,
            &mut self.e_norm,
            t,
            self.hidden,
            self.rms_eps,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_forward(
        &mut self,
        stream: &Arc<CudaStream>,
        h: &CudaSlice<f32>,
        cache: &mut KvCache,
        pos_offset: usize,
        positions: &CudaSlice<i32>,
    ) -> Result<Option<Vec<i32>>, MtpBlockError> {
        self.mix_input(stream, h)?;

        // 5. block 40's dense-attention mixer, over its own KV cache.
        // The draft head rotates by the slot scalar. For image-bearing
        // sequences this ignores the M-RoPE delta, which can only cost
        // draft acceptance rate — the target pass re-scores every draft
        // token with the correct rotary positions, so exactness is
        // unaffected. Text-only sequences have delta 0 and are identical.
        let (attn, attn_scratch, eh_out, attn_out) = (
            &mut self.attn,
            &mut self.attn_scratch,
            &self.eh_out,
            &mut self.attn_out,
        );
        attn.forward(
            stream,
            attn_scratch,
            eh_out,
            cache,
            pos_offset,
            positions,
            crate::block::attention::RopeSource::Scalar(positions),
            attn_out,
        )?;

        self.finish_output(stream)
    }

    /// Steps 2h-4 of the head: `hnorm(h)`, the e/h concat and `eh_proj`,
    /// leaving the attention input in `self.eh_out`. `self.e_norm` must
    /// already hold the embedded, normalized token rows.
    fn mix_input(
        &mut self,
        stream: &Arc<CudaStream>,
        h: &CudaSlice<f32>,
    ) -> Result<(), MtpBlockError> {
        let t = self.tokens;
        let eps = self.rms_eps;
        self.layer_ops.rms_norm(
            stream,
            h,
            &self.w_hnorm,
            &mut self.h_norm,
            t,
            self.hidden,
            eps,
        )?;

        // 3. `mtp_concat`: e_norm first, then h_norm — `ggml_concat(dim=0)`.
        let cfg = LaunchConfig {
            grid_dim: (t as u32, 1, 1),
            block_dim: (EMBED_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let hidden_i32 = self.hidden as i32;
        let tokens_i32 = t as i32;
        let mut builder = stream.launch_builder(&self.concat_fn);
        builder
            .arg(&self.e_norm)
            .arg(&self.h_norm)
            .arg(&mut self.concat)
            .arg(&hidden_i32)
            .arg(&tokens_i32);
        // SAFETY: `e_norm`/`h_norm` are `t * hidden`, `concat` is
        // `t * 2 * hidden`, and the grid covers exactly `t` rows.
        unsafe { builder.launch(cfg) }?;

        // 4. `mtp_eh_proj`.
        self.eh_proj.forward(
            stream,
            QuantTensor {
                bytes: &self.w_eh_proj,
                quant: ExpertQuant::Q8_0,
            },
            &self.concat,
            t,
            &mut self.eh_out,
        )?;
        Ok(())
    }

    /// Steps 6-7 and the LM head: everything after the attention mixer.
    /// `self.attn_out` must hold the mixer's output for all `tokens` rows.
    fn finish_output(
        &mut self,
        stream: &Arc<CudaStream>,
    ) -> Result<Option<Vec<i32>>, MtpBlockError> {
        let t = self.tokens;
        let eps = self.rms_eps;
        // 6. block 40's MoE, residual against the attention output.
        self.moe.forward(
            stream,
            &self.moe_weights,
            &self.attn_out,
            t,
            &mut self.ffn_out,
            &mut self.post_ffn,
        )?;

        // 7. `h_nextn` = `shared_head_norm(post_ffn)` — fed to the LM head
        //    below and, by the caller, back into the next chained step.
        self.layer_ops.rms_norm(
            stream,
            &self.post_ffn,
            &self.w_shared_head_norm,
            &mut self.h_nextn,
            t,
            self.hidden,
            eps,
        )?;

        let Some(lm_head) = self.lm_head.as_ref() else {
            return Ok(None);
        };
        let logits = self.logits.as_mut().expect("built with lm_head");
        lm_head.forward(
            stream,
            QuantTensor {
                bytes: &self.w_lm_head,
                quant: ExpertQuant::Q8_0,
            },
            &self.h_nextn,
            t,
            logits,
        )?;

        let vocab = lm_head.geometry().vocab;
        for i in 0..t {
            let row = unsafe { crate::viewslice::subslice(stream, logits, i * vocab, vocab) };
            let mut values = unsafe {
                crate::viewslice::subslice(
                    stream,
                    self.argmax_values.as_ref().expect("built with lm_head"),
                    i * ARGMAX_BLOCKS,
                    ARGMAX_BLOCKS,
                )
            };
            let mut indices = unsafe {
                crate::viewslice::subslice(
                    stream,
                    self.argmax_indices.as_ref().expect("built with lm_head"),
                    i * ARGMAX_BLOCKS,
                    ARGMAX_BLOCKS,
                )
            };
            let mut out_one = unsafe {
                crate::viewslice::subslice(
                    stream,
                    self.argmax_out.as_ref().expect("built with lm_head"),
                    i,
                    1,
                )
            };
            lm_head.argmax(stream, &row, vocab, &mut values, &mut indices, &mut out_one)?;
        }
        let host = stream.clone_dtoh(self.argmax_out.as_ref().expect("built with lm_head"))?;
        stream.synchronize()?;
        Ok(Some(host))
    }
}

/// Alias one resident Q8_0 tensor, rejecting any other stored format.
///
/// Mirrors `forward.rs`'s private `alias_q8_0` — duplicated rather than
/// shared because that one is private to `Forward` and returns a
/// `ForwardError`, not an `MtpBlockError`.
fn alias_q8_0(
    weights: &DeviceWeights,
    stream: &Arc<CudaStream>,
    role: Role,
    layer: Option<u32>,
) -> Result<ManuallyDrop<CudaSlice<u8>>, MtpBlockError> {
    let placement = weights
        .find(role, layer)
        .ok_or(MtpBlockError::MissingWeight { role, layer })?;
    if placement.ggml_type != GgmlType::Q8_0 {
        return Err(MtpBlockError::WrongQuant {
            role,
            layer,
            found: placement.ggml_type,
        });
    }
    let alias = weights
        .bytes_of(stream, role, layer)
        .ok_or(MtpBlockError::MissingWeight { role, layer })?;
    // SAFETY: the result is sealed in a `ManuallyDrop` that the caller stores
    // in `MtpBlock` and never takes out of, and `MtpBlock` is used only while
    // the `DeviceWeights` it was built from is alive.
    Ok(ManuallyDrop::new(unsafe { alias.into_aliasing_slice() }))
}

/// Alias one resident f32 tensor, rejecting any other stored format.
fn alias_f32(
    weights: &DeviceWeights,
    stream: &Arc<CudaStream>,
    role: Role,
    layer: Option<u32>,
) -> Result<ManuallyDrop<CudaSlice<f32>>, MtpBlockError> {
    let placement = weights
        .find(role, layer)
        .ok_or(MtpBlockError::MissingWeight { role, layer })?;
    if placement.ggml_type != GgmlType::F32 {
        return Err(MtpBlockError::WrongQuant {
            role,
            layer,
            found: placement.ggml_type,
        });
    }
    let alias = weights
        .f32_of(stream, role, layer)
        .ok_or(MtpBlockError::MissingWeight { role, layer })?;
    // SAFETY: as `alias_q8_0`.
    Ok(ManuallyDrop::new(unsafe { alias.into_aliasing_slice() }))
}
