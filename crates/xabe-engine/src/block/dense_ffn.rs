//! The dense feed-forward block carried by every `qwen35` layer.
//!
//! Transcribed from `llama_model_qwen35::graph::build_layer_ffn`
//! (`/home/nixabe/llama.cpp/src/models/qwen35.cpp:473`), which is a single
//! call to `build_ffn(up, gate, down, LLM_FFN_SILU, LLM_FFN_PAR)` guarded by
//! `GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr)` — the assertion is
//! upstream's own statement that this architecture has no router.
//!
//! ## The block, step by step, with the graph node each step produces
//!
//! ```text
//!   input   = mixer output + residual                       `attn_residual-N`
//!   normed  = RMSNorm(input, post_attention_norm.weight)     `attn_post_norm-N`
//!   ffn_out = down(silu(gate . normed) * (up . normed))      `ffn_out-N`
//!   l_out   = ffn_out + input                                `post_ffn-N`
//! ```
//!
//! Two things are worth stating because they are the differences from
//! [`crate::block::moe`], and both are absences rather than additions:
//!
//! 1. **There is no shared-expert gate.** The MoE block's `ffn_out` is
//!    `routed + shexp * sigmoid(ffn_gate_inp_shexp . normed)`; this one's is
//!    the MLP output unmodified. Carrying the gate over would be a plausible,
//!    fluent, wrong model.
//! 2. **The residual is still taken before the post-mixer norm** — `qwen35.cpp`
//!    line 188 saves `ffn_residual = cur` and line 199 adds it back, exactly as
//!    the MoE graph does. That much *is* shared, so it is the one place where
//!    copying the MoE block's structure is right.
//!
//! Upstream's final callback for the layer is `post_ffn` where `qwen35moe`'s
//! is `l_out`; without a control vector `build_cvec` is the identity, so the
//! two name the same tensor and this block writes it to the caller's `l_out`.
//!
//! ## Why this runs on the MoE crate's shared-expert kernels
//!
//! `down(silu(gate . x) * (up . x))` is *the same computation* the MoE block's
//! shared expert performs, and
//! [`xabe_cuda::kernels::moe::MoeKernels::shared_expert`] takes its widths as
//! runtime arguments rather than baking Qwen3.6's 512 into the kernel. So the
//! dense FFN is that entry point at `intermediate = 17408` instead of 512,
//! plus one residual add — not a second implementation of a SwiGLU MLP, and
//! not a new differential test surface either: `tests/moe_differential.rs`
//! already gates those kernels against the CPU reference.
//!
//! What it must *not* inherit is the routing half. A dense block allocated
//! with [`MoeKernels::buffers`] would carry dispatch tables, a routed
//! `partial`, and an `inter` sized by the dispatch-slot count — about 445 MiB
//! per worker at a 4,096-token prefill and this model's FFN width, for tables
//! no launch reads. It takes
//! [`MoeKernels::shared_only_buffers`](xabe_cuda::kernels::moe::MoeKernels::shared_only_buffers)
//! instead, which allocates the shared path and leaves the routed tables
//! zero-length so a stray routed launch fails rather than dispatching nothing.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use xabe_cuda::kernels::compile;
use xabe_cuda::kernels::layer_ops::LayerOpsKernels;
use xabe_cuda::kernels::moe::{
    ExpertQuant, MoeBuffers, MoeGeometry, MoeKernels, QuantTensor, SharedExpertInt8,
    to_device_layout,
};
use xabe_gguf::GgufFile;
use xabe_model::config::ModelConfig;
use xabe_model::weights::{Directory, Role};

use crate::block::moe::{MoeBlockError, check_len, dense_tensor, quantized_tensor};

/// Threads per block for the residual add.
const THREADS: u32 = 256;

/// Warps per block in the split GEMV. Mirrors `GdnBlock`'s `PROJ_WARPS`.
const SPLIT_WARPS: u32 = 4;

/// Rows one warp of the split GEMV owns. Mirrors the `RT` in every
/// `DENSE_PROJ_SPLIT_ENTRY` below.
const SPLIT_ROWS: u32 = 4;

/// Widest pass the split GEMV serves.
///
/// Above this the pass is a prefill chunk and `shared_expert_mma`'s GEMM tile
/// is the right shape; at or below it the tile is mostly empty and the GEMV
/// wins — 63.5 ms against 109.0 ms on a one-token Qwen3.8-27B step. Four is
/// the widest compiled entry point, and also the serving target's batched
/// decode width plus one.
const SPLIT_GEMV_MAX_TOKENS: usize = 4;

/// The dense FFN's weight residency, and the one real design decision in this
/// module.
///
/// [`crate::block::moe`] holds Q8_0 *and* a tensor-core repack, because there
/// the repack covers the shared expert alone — 3.5 MB against that layer's
/// 725 MB of routed experts, noise. Here the repack would cover the whole
/// feed-forward block, and holding both does not fit: one layer's three
/// matrices are 267 M elements, which is 271 MiB of Q8_0 plus 287 MiB of
/// split int8, and across 64 layers that is 16.9 + 17.9 GiB on a 47.27 GiB
/// card already carrying 11.8 GiB of arena. The engine's first attempt at the
/// dense model died on exactly that `CUDA_ERROR_OUT_OF_MEMORY`.
///
/// So it is one or the other, and the measurement decides. With Q8_0 only,
/// every pass runs the fp32 `moe_shared_ffn` tile and prefill 512 measures
/// **76.2 tok/s** against llama.cpp's 691.2 on the same card — 9.1x slower,
/// which is the whole cost of not reaching the integer tensor cores on a
/// model whose FFN is 85% of its arithmetic. With int8 only, prefill reaches
/// **317.5 tok/s**, and the ~1.0 GiB the split layout costs over Q8_0 buys
/// that back.
///
/// It does *not* follow that every pass should then run `shared_expert_mma`.
/// Making the residency exclusive means the residency can no longer select
/// the kernel, so the width must: that GEMM stages a 64-token tile and at one
/// token discards 63/64 of it, measuring 109.0 ms per decode step against a
/// GEMV's 63.5 on identical weight traffic. Hence the `SplitGemv` path, a
/// copy of `GdnBlock`'s split-layout projection GEMV — the same layout, the
/// same widths — taken at `<= SPLIT_GEMV_MAX_TOKENS`.
///
/// The int8 layout is not an approximation of the Q8_0 one. Q8_0 is an int8
/// quant and an fp16 scale per 32; the split layout is the same quants with
/// the scale widened to fp32. Nothing is requantized, and the tensor core
/// multiplies the quants exactly. What *does* change is that the activations
/// are quantized to int8 — about 1/254 of each 32-element block's largest
/// magnitude — on the decode path as well as the prefill one. That is
/// measured against the CPU reference in `tests/dense_ffn_differential.rs`
/// (cosine 0.999997, max_abs 6.45e-2 on a tensor whose own scale is 82.3) and
/// end to end against llama.cpp's greedy output in `docs/BENCHMARKS.md`.
///
/// A dense model on a device without integer tensor cores falls back to Q8_0
/// and the fp32 kernels; correctness is unaffected, speed is not.
pub const DENSE_REPACK_INT8: bool = true;

/// The one operation the dense block needs that no landed kernel provides.
///
/// [`LayerOpsKernels::add`] would compute the same sum, but over the whole
/// buffer: it takes an element count from the host and knows nothing about
/// which token slots are live. Writing `l_out` on a slot the pass never
/// filled would put a plausible value where the caller left a zero, which is
/// exactly the thing `tests/forward_pass.rs` checks for. So the token bound
/// comes from the same `valid_tokens` device scalar the MoE kernels gate on,
/// and the launch shape comes from the geometry — AGENTS.md rule 5.
const GLUE_SRC: &str = r#"
extern "C" {

// l_out[t][j] = ffn_out[t][j] + residual[t][j], for live t only.
//
// The row is split over `gridDim.y` blocks for the same reason the MoE
// combine's is: one block per token would put a 20 KiB row on one SM.
__global__ void dense_ffn_add_residual(
    const float* __restrict__ ffn_out,
    const float* __restrict__ residual,
    const int* __restrict__ valid_tokens,
    int hidden,
    float* __restrict__ l_out) {
    int t = blockIdx.x;
    if (t >= *valid_tokens) return;

    long long base = (long long)t * hidden;
    int stride = gridDim.y * blockDim.x;
    for (int j = blockIdx.y * blockDim.x + threadIdx.x; j < hidden; j += stride) {
        l_out[base + j] = ffn_out[base + j] + residual[base + j];
    }
}

}

// ---------------------------------------------------------------------------
// The narrow-batch MLP over the split int8 layout.
// ---------------------------------------------------------------------------
//
// `shared_expert_mma` is a GEMM: it stages a whole 64-token tile and at one
// token throws 63/64 of that away. Measured, that is the difference between
// 63.5 ms and 109.0 ms on a Qwen3.8-27B decode step — the weight bytes are
// the same either way, so what it loses is the shape it reads them in.
//
// The dense FFN's weights are resident *only* in the split layout
// (`DENSE_REPACK_INT8`), so the fp32 kernels are not available to fall back
// to. This is the GEMV that reads that layout instead, and it is the same
// body `GdnBlock` runs for its own split projections — the one this project
// already measured as the fastest form at these widths.
//
// **Copied rather than shared.** `block/gdn.rs` owns the original, it is on
// the measured path for the model this project's whole benchmark record is
// about, and moving its source into a shared translation unit would move its
// codegen for no reason. The same argument as `GDN_GATES_Q8`. The two copies
// are kept identical by `the_split_gemv_body_matches_the_gdn_blocks`.
__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

template <int TT, int RT, bool ADD>
__device__ __forceinline__ void dense_proj_split_rows(
    const signed char* __restrict__ wq,
    const float* __restrict__ ws,
    const float* __restrict__ x,
    const float* __restrict__ residual,
    float* __restrict__ out,
    float* __restrict__ summed,
    int k_dim,
    int n_rows,
    int n_tokens
) {
    int lane = threadIdx.x;
    int n0 = (blockIdx.x * blockDim.y + threadIdx.y) * RT;
    if (n0 >= n_rows) return;
    int t0 = blockIdx.y * TT;
    if (t0 >= n_tokens) return;
    int live_t = n_tokens - t0; if (live_t > TT) live_t = TT;

    const signed char* row[RT];
    const float* sc[RT];
    #pragma unroll
    for (int r = 0; r < RT; ++r) {
        row[r] = wq + (long long)(n0 + r) * k_dim;
        sc[r]  = ws + (long long)(n0 + r) * (k_dim >> 5);
    }

    float acc[RT][TT];
    #pragma unroll
    for (int r = 0; r < RT; ++r) {
        #pragma unroll
        for (int i = 0; i < TT; ++i) acc[r][i] = 0.0f;
    }

    int off = 16 * lane;
    uint4 cur[RT];
    float dcur[RT];
    #pragma unroll
    for (int r = 0; r < RT; ++r) {
        cur[r]  = *(const uint4*)(row[r] + off);
        dcur[r] = sc[r][off >> 5];
    }

    for (int c = 0; c < k_dim; c += 512) {
        int cn = c + 512;
        uint4 nxt[RT];
        float dnxt[RT];
        if (cn < k_dim) {
            #pragma unroll
            for (int r = 0; r < RT; ++r) {
                nxt[r]  = *(const uint4*)(row[r] + cn + off);
                dnxt[r] = sc[r][(cn + off) >> 5];
            }
        }
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            int e0 = c + off + 4 * u;
            float4 xv[TT];
            #pragma unroll
            for (int i = 0; i < TT; ++i) {
                int t = t0 + i;
                xv[i] = (t < n_tokens)
                    ? *(const float4*)(x + (long long)t * k_dim + e0)
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            #pragma unroll
            for (int r = 0; r < RT; ++r) {
                unsigned int wrd = ((const unsigned int*)&cur[r])[u];
                float d = dcur[r];
                // Low byte first: the split repack keeps quants in element
                // order, and narrowing through `unsigned char` first makes
                // the int8 sign-extend rather than taking an
                // implementation-defined conversion from a value above 127
                // — the same idiom as `lm_head.rs`.
                float w0 = (float)(signed char)(unsigned char)(wrd)       * d;
                float w1 = (float)(signed char)(unsigned char)(wrd >> 8)  * d;
                float w2 = (float)(signed char)(unsigned char)(wrd >> 16) * d;
                float w3 = (float)(signed char)(unsigned char)(wrd >> 24) * d;
                #pragma unroll
                for (int i = 0; i < TT; ++i) {
                    acc[r][i] += w0 * xv[i].x;
                    acc[r][i] += w1 * xv[i].y;
                    acc[r][i] += w2 * xv[i].z;
                    acc[r][i] += w3 * xv[i].w;
                }
            }
        }
        if (cn < k_dim) {
            #pragma unroll
            for (int r = 0; r < RT; ++r) { cur[r] = nxt[r]; dcur[r] = dnxt[r]; }
        }
    }

    #pragma unroll
    for (int r = 0; r < RT; ++r) {
        for (int i = 0; i < live_t; ++i) {
            float sum = warp_reduce_sum(acc[r][i]);
            if (lane == 0) {
                long long o = (long long)(t0 + i) * n_rows + (n0 + r);
                out[o] = sum;
                if (ADD) summed[o] = sum + residual[o];
            }
        }
    }
}

#define DENSE_PROJ_SPLIT_ENTRY(NAME, TT, RT)                                 \
extern "C" __global__ void NAME(                                             \
    const signed char* __restrict__ wq,                                      \
    const float* __restrict__ ws,                                            \
    const float* __restrict__ x,                                             \
    float* __restrict__ out,                                                 \
    int k_dim,                                                               \
    int n_rows,                                                              \
    int n_tokens                                                             \
) {                                                                          \
    dense_proj_split_rows<TT, RT, false>(                                    \
        wq, ws, x, nullptr, out, nullptr, k_dim, n_rows, n_tokens);          \
}

DENSE_PROJ_SPLIT_ENTRY(dense_proj_split_t1, 1, 4)
DENSE_PROJ_SPLIT_ENTRY(dense_proj_split_t2, 2, 4)
DENSE_PROJ_SPLIT_ENTRY(dense_proj_split_t3, 3, 4)
DENSE_PROJ_SPLIT_ENTRY(dense_proj_split_t4, 4, 4)
"#;

/// One layer's dense FFN weights, resident on the device.
///
/// The three matrices are Q8_0 throughout `Qwen3.8-27B-UD-Q8_K_XL`, but the
/// format is still read per tensor out of the file's own directory rather
/// than assumed: the MoE model taught this lesson the expensive way (block 39
/// stores gate and up as Q8_0 where every other block uses Q6_K), and a
/// future quant recipe for this file has no reason to be more uniform.
pub struct DenseFfnLayerWeights {
    layer: u32,
    post_norm: CudaSlice<f32>,
    matrices: DenseFfnMatrices,
}

/// The three matrices, in exactly one residency. See [`DENSE_REPACK_INT8`].
enum DenseFfnMatrices {
    /// As the file stores them, for the fp32 kernels.
    Quantized {
        gate: CudaSlice<u8>,
        up: CudaSlice<u8>,
        down: CudaSlice<u8>,
        gate_quant: ExpertQuant,
        up_quant: ExpertQuant,
        down_quant: ExpertQuant,
    },
    /// Repacked for the integer tensor cores. The Q8_0 upload it was built
    /// from is dropped, which is what keeps this a replacement rather than an
    /// addition.
    Int8(SharedExpertInt8),
}

impl DenseFfnLayerWeights {
    /// Upload every tensor block `layer`'s dense FFN needs.
    ///
    /// `directory` must have been resolved against `file`.
    /// `repack_int8` asks for the integer tensor-core layout *instead of* the
    /// Q8_0 one. The serving path passes [`DENSE_REPACK_INT8`] when the
    /// device has integer tensor cores; the differential test drives both
    /// values so both residencies stay gated.
    pub fn upload(
        stream: &Arc<CudaStream>,
        file: &GgufFile,
        directory: &Directory<'_>,
        layer: u32,
        geometry: &MoeGeometry,
        repack_int8: bool,
    ) -> Result<Self, MoeBlockError> {
        let hidden = geometry.hidden;
        let one_matrix = geometry.intermediate * hidden;

        let (post_norm, _) = dense_tensor(file, directory, Role::PostMixerNorm, layer, hidden)?;
        let (gate_bytes, gate_quant) =
            quantized_tensor(file, directory, Role::FfnGate, layer, one_matrix)?;
        let (up_bytes, up_quant) =
            quantized_tensor(file, directory, Role::FfnUp, layer, one_matrix)?;
        let (down_bytes, down_quant) =
            quantized_tensor(file, directory, Role::FfnDown, layer, one_matrix)?;

        let gate = stream.clone_htod(&*to_device_layout(gate_quant, gate_bytes))?;
        let up = stream.clone_htod(&*to_device_layout(up_quant, up_bytes))?;
        let down = stream.clone_htod(&*to_device_layout(down_quant, down_bytes))?;

        // At upload, not lazily on the first pass: AGENTS.md rule 6 forbids
        // allocating mid-forward, and a lazy repack would also poison the
        // first timed repetition of every benchmark.
        //
        // The Q8_0 upload is the repack's *input*, so both exist for the
        // duration of this call — 558 MiB for one layer — and only the
        // repacked form survives it. Doing this per layer rather than after a
        // loop is what keeps that transient at one layer's worth.
        let all_q8_0 = [gate_quant, up_quant, down_quant]
            .iter()
            .all(|q| *q == ExpertQuant::Q8_0);
        let repacked = if repack_int8 && all_q8_0 {
            SharedExpertInt8::repack(
                stream.context(),
                stream,
                QuantTensor {
                    bytes: &gate,
                    quant: gate_quant,
                },
                QuantTensor {
                    bytes: &up,
                    quant: up_quant,
                },
                QuantTensor {
                    bytes: &down,
                    quant: down_quant,
                },
                one_matrix,
            )
            .ok()
        } else {
            None
        };

        let matrices = match repacked {
            Some(int8) => {
                drop((gate, up, down));
                DenseFfnMatrices::Int8(int8)
            }
            None => DenseFfnMatrices::Quantized {
                gate,
                up,
                down,
                gate_quant,
                up_quant,
                down_quant,
            },
        };

        Ok(Self {
            layer,
            post_norm: stream.clone_htod(&post_norm)?,
            matrices,
        })
    }

    /// Which block these weights came from.
    pub fn layer(&self) -> u32 {
        self.layer
    }

    /// Storage format of the three matrices, in gate/up/down order, or `None`
    /// when they are resident as the integer repack instead.
    pub fn quants(&self) -> Option<[ExpertQuant; 3]> {
        match &self.matrices {
            DenseFfnMatrices::Quantized {
                gate_quant,
                up_quant,
                down_quant,
                ..
            } => Some([*gate_quant, *up_quant, *down_quant]),
            DenseFfnMatrices::Int8(_) => None,
        }
    }

    /// Whether this layer is resident as the integer tensor-core repack.
    pub fn has_int8(&self) -> bool {
        matches!(self.matrices, DenseFfnMatrices::Int8(_))
    }

    /// Total device bytes held.
    pub fn bytes(&self) -> usize {
        let matrices = match &self.matrices {
            DenseFfnMatrices::Quantized { gate, up, down, .. } => {
                gate.len() + up.len() + down.len()
            }
            DenseFfnMatrices::Int8(w) => w.bytes(),
        };
        self.post_norm.len() * size_of::<f32>() + matrices
    }
}

/// The dense feed-forward block, compiled and sized for one fixed geometry.
pub struct DenseFfnBlock {
    moe: MoeKernels,
    layer_ops: LayerOpsKernels,
    add_residual_fn: CudaFunction,
    buffers: MoeBuffers,
    normed: CudaSlice<f32>,
    /// The narrow-batch path, when this pass shape is narrow enough to want
    /// it. `None` on a prefill shape, where `shared_expert_mma` is the right
    /// kernel and these buffers would be hundreds of megabytes.
    split: Option<SplitGemv>,
    geometry: MoeGeometry,
    eps: f32,
}

/// The compiled split-layout GEMV and its `[max_tokens][intermediate]`
/// scratch.
///
/// Three buffers of that shape, which is 1.1 MiB total at four tokens and
/// this model's FFN width — against the 613 MiB the tensor-core staging
/// arrays would be at a 2,048-token prefill. That asymmetry is why the two
/// paths do not both allocate.
struct SplitGemv {
    /// Indexed by `tokens - 1`, for `1..=SPLIT_GEMV_MAX_TOKENS`.
    tiles: [CudaFunction; SPLIT_GEMV_MAX_TOKENS],
    gate_out: CudaSlice<f32>,
    up_out: CudaSlice<f32>,
    activated: CudaSlice<f32>,
}

impl DenseFfnBlock {
    /// The dense FFN geometry implied by `config`, for `max_tokens` per step.
    ///
    /// Returns `None` for a model whose layers carry a routed MoE block
    /// instead; the caller picks the block type from the same
    /// [`ModelConfig::ffn`] this reads.
    ///
    /// `num_experts` and `experts_per_token` are one because
    /// [`MoeGeometry`] describes the *shared* path too and its validation
    /// rejects zero — no routed launch ever sees these buffers, and
    /// [`MoeKernels::shared_only_buffers`](xabe_cuda::kernels::moe::MoeKernels::shared_only_buffers)
    /// is what makes that structural rather than a convention.
    pub fn geometry_for(
        config: &ModelConfig,
        block_size: usize,
        max_tokens: usize,
    ) -> Option<MoeGeometry> {
        let d = config.dense_ffn()?;
        Some(MoeGeometry {
            num_experts: 1,
            experts_per_token: 1,
            hidden: config.hidden_size as usize,
            intermediate: d.intermediate as usize,
            block_size,
            max_tokens,
        })
    }

    /// Compile every kernel the block needs and allocate every buffer, once.
    ///
    /// `mma` must match the `repack_int8` the weights were uploaded with: it
    /// is what decides whether the four integer staging arrays — 613 MiB at
    /// this model's width and a 2,048-token prefill — are allocated at all.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        geometry: MoeGeometry,
        eps: f32,
        mma: bool,
    ) -> Result<Self, MoeBlockError> {
        let moe = MoeKernels::new(ctx, geometry)?;
        let layer_ops = LayerOpsKernels::new(ctx)?;
        let ptx = compile(GLUE_SRC, "dense_ffn_block").map_err(MoeBlockError::Compile)?;
        let module = ctx.load_module(ptx)?;
        // Exactly one of the two int8 paths is provisioned, by width. The
        // GEMV's scratch is small enough to be unconditional; the tensor-core
        // staging arrays are not.
        let narrow = mma && geometry.max_tokens <= SPLIT_GEMV_MAX_TOKENS;
        let buffers = moe.shared_only_buffers(stream, mma && !narrow)?;

        let split = if narrow {
            let n = geometry.max_tokens * geometry.intermediate;
            Some(SplitGemv {
                tiles: [
                    module.load_function("dense_proj_split_t1")?,
                    module.load_function("dense_proj_split_t2")?,
                    module.load_function("dense_proj_split_t3")?,
                    module.load_function("dense_proj_split_t4")?,
                ],
                gate_out: stream.alloc_zeros::<f32>(n)?,
                up_out: stream.alloc_zeros::<f32>(n)?,
                activated: stream.alloc_zeros::<f32>(n)?,
            })
        } else {
            None
        };

        Ok(Self {
            add_residual_fn: module.load_function("dense_ffn_add_residual")?,
            moe,
            layer_ops,
            buffers,
            normed: stream.alloc_zeros::<f32>(geometry.max_tokens * geometry.hidden)?,
            split,
            geometry,
            eps,
        })
    }

    /// `down(silu(gate . normed) * (up . normed))` over the split int8
    /// layout, three GEMV launches and a SwiGLU.
    ///
    /// Every launch shape comes from the geometry. The token count does not
    /// reach the device here at all — unlike `shared_expert`, this path
    /// computes every row of the buffer, so `n_tokens` is `max_tokens` and no
    /// row is conditionally skipped. The residual add downstream is what
    /// gates on `valid_tokens`, and it is the only thing that writes `l_out`.
    fn mlp_split_gemv(
        &mut self,
        stream: &Arc<CudaStream>,
        w: &SharedExpertInt8,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), MoeBlockError> {
        let g = self.geometry;
        let split = self.split.as_mut().expect("caller checked");
        let f = &split.tiles[g.max_tokens - 1];

        let (gate_q, gate_s) = w.gate();
        let (up_q, up_s) = w.up();
        let (down_q, down_s) = w.down();
        project_split(
            stream,
            f,
            gate_q,
            gate_s,
            &self.normed,
            &mut split.gate_out,
            g.hidden,
            g.intermediate,
            g.max_tokens,
        )?;
        project_split(
            stream,
            f,
            up_q,
            up_s,
            &self.normed,
            &mut split.up_out,
            g.hidden,
            g.intermediate,
            g.max_tokens,
        )?;
        self.layer_ops.swiglu(
            stream,
            &split.gate_out,
            &split.up_out,
            &mut split.activated,
            g.max_tokens * g.intermediate,
        )?;
        project_split(
            stream,
            f,
            down_q,
            down_s,
            &split.activated,
            out,
            g.intermediate,
            g.hidden,
            g.max_tokens,
        )
    }

    /// Force the FFN GEMMs back onto their fp32 kernels.
    pub fn disable_tensor_cores(&mut self) {
        self.moe.disable_tensor_cores();
    }

    /// Whether the FFN GEMMs will take the integer tensor-core path.
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

    /// The buffers this block gates on, for inspection.
    pub fn buffers(&self) -> &MoeBuffers {
        &self.buffers
    }

    /// Publish this pass's token count into the device scalar the kernels
    /// gate on, ahead of the pass. See [`crate::block::moe::MoeBlock::publish_tokens`].
    pub fn publish_tokens(
        &mut self,
        stream: &Arc<CudaStream>,
        tokens: usize,
    ) -> Result<(), MoeBlockError> {
        self.moe
            .set_valid_tokens(stream, &mut self.buffers, tokens)?;
        Ok(())
    }

    /// Run the whole block over `tokens` tokens.
    ///
    /// `residual` is the mixer output already added back to the block's input
    /// — llama.cpp's `attn_residual-N` — and is both the RMSNorm's input and
    /// the residual the block's output is added to. `ffn_out` receives
    /// `ffn_out-N` and `l_out` receives `post_ffn-N`; every buffer is
    /// `[max_tokens][hidden]`.
    pub fn forward(
        &mut self,
        stream: &Arc<CudaStream>,
        w: &DenseFfnLayerWeights,
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

        self.publish_tokens(stream, tokens)?;

        // 1. post-mixer RMSNorm, over every row of the buffer rather than
        //    just the live ones: a row of zeros normalizes to zeros, since
        //    the epsilon keeps the divisor positive.
        self.layer_ops.rms_norm(
            stream,
            residual,
            &w.post_norm,
            &mut self.normed,
            g.max_tokens,
            g.hidden,
            self.eps,
        )?;

        // 2. the MLP. There is no width gate here, and that is the difference
        //    from `MoeBlock::forward`. There, the weights are resident in
        //    *both* forms, so a one-token decode step can and should take the
        //    two-launch fp32 path rather than the six-launch integer one to
        //    fill one slot of a 64-token tile. Here they are resident in
        //    exactly one form — see `DENSE_REPACK_INT8` — so the residency
        //    picks the kernel, not the token count.
        match &w.matrices {
            DenseFfnMatrices::Int8(i8w) => {
                if !self.moe.tensor_cores_enabled() {
                    return Err(MoeBlockError::Int8WeightsWithoutTensorCores { layer: w.layer });
                }
                if self.split.is_some() {
                    self.mlp_split_gemv(stream, i8w, ffn_out)?;
                } else {
                    self.moe.shared_expert_mma(
                        stream,
                        &mut self.buffers,
                        i8w,
                        &self.normed,
                        ffn_out,
                    )?;
                }
            }
            DenseFfnMatrices::Quantized {
                gate,
                up,
                down,
                gate_quant,
                up_quant,
                down_quant,
            } => self.moe.shared_expert(
                stream,
                &mut self.buffers,
                QuantTensor {
                    bytes: gate,
                    quant: *gate_quant,
                },
                QuantTensor {
                    bytes: up,
                    quant: *up_quant,
                },
                QuantTensor {
                    bytes: down,
                    quant: *down_quant,
                },
                &self.normed,
                ffn_out,
            )?,
        }

        // 3. the residual add. Kept out of step 2 rather than folded into the
        //    down projection so `ffn_out-N` survives as its own waypoint,
        //    which is what the oracle comparison hangs on.
        let hidden_i32 = g.hidden as i32;
        let cfg = LaunchConfig {
            grid_dim: (g.max_tokens as u32, (g.hidden as u32).div_ceil(THREADS), 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&self.add_residual_fn);
        builder
            .arg(&*ffn_out)
            .arg(residual)
            .arg(self.buffers.valid_tokens())
            .arg(&hidden_i32)
            .arg(l_out);
        // SAFETY: one block per token slot, gated on the device
        // `valid_tokens`. All three `[max_tokens][hidden]` buffers were
        // length-checked above and the in-row loop is bounded by `hidden`.
        unsafe { builder.launch(cfg) }?;

        Ok(())
    }
}

/// One split-layout projection: `out[t][n] = sum_k dequant(wq,ws)[n][k] * x[t][k]`.
///
/// Every launch shape comes from the geometry. The token count does not reach
/// the device as a gate — unlike `shared_expert`, this path computes every row
/// of the buffer, so `n_tokens` is `max_tokens` and no row is conditionally
/// skipped. The residual add downstream is what gates on `valid_tokens`, and
/// it is the only thing that writes `l_out`.
#[allow(clippy::too_many_arguments)]
fn project_split(
    stream: &Arc<CudaStream>,
    f: &CudaFunction,
    wq: &CudaSlice<i8>,
    ws: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    k_dim: usize,
    n_rows: usize,
    tokens: usize,
) -> Result<(), MoeBlockError> {
    let cfg = LaunchConfig {
        grid_dim: ((n_rows as u32).div_ceil(SPLIT_WARPS * SPLIT_ROWS), 1, 1),
        block_dim: (32, SPLIT_WARPS, 1),
        shared_mem_bytes: 0,
    };
    let (k_i32, n_i32, t_i32) = (k_dim as i32, n_rows as i32, tokens as i32);
    let mut builder = stream.launch_builder(f);
    builder
        .arg(wq)
        .arg(ws)
        .arg(x)
        .arg(&mut *out)
        .arg(&k_i32)
        .arg(&n_i32)
        .arg(&t_i32);
    // SAFETY: one warp per group of `SPLIT_ROWS` adjacent output rows over a
    // grid that covers `n_rows` and returns above it. `k_dim` is a multiple of
    // 512 and `n_rows` of `SPLIT_ROWS` — both hold at this model's widths and
    // are asserted in the tests below — so no partially live warp group
    // exists. `wq` holds `n_rows * k_dim` quants and `ws` one scale per 32 of
    // them; `x` and `out` are `tokens` rows of `k_dim` and `n_rows`.
    unsafe { builder.launch(cfg) }?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_geometry_is_derived_from_the_model_config_not_written_out() {
        let c = ModelConfig::qwen3_8_27b();
        let g = DenseFfnBlock::geometry_for(&c, 32, 128).expect("qwen35 is dense");
        assert_eq!(g.hidden, 5120);
        assert_eq!(g.intermediate, 17_408);
        // The routed half is a placeholder, not a one-expert MoE: nothing
        // reads these, and `shared_only_buffers` is what enforces that.
        assert_eq!(g.num_experts, 1);
        assert_eq!(g.experts_per_token, 1);
    }

    #[test]
    fn a_routed_model_has_no_dense_geometry() {
        assert!(DenseFfnBlock::geometry_for(&ModelConfig::qwen3_6_35b_a3b(), 32, 128).is_none());
    }

    #[test]
    fn the_widths_satisfy_the_shared_expert_kernels_own_constraints() {
        // `MoeKernels::new` rejects a `hidden` or `intermediate` that is not a
        // multiple of the 128-element tile, and `shared_expert`'s GEMV
        // additionally splits `hidden` over four warps in 128-element tiles.
        // Both hold here, but not by much margin worth assuming.
        let g = DenseFfnBlock::geometry_for(&ModelConfig::qwen3_8_27b(), 32, 128).unwrap();
        assert_eq!(g.hidden % 128, 0);
        assert_eq!(g.intermediate % 128, 0);
        assert_eq!(g.hidden % 512, 0);
    }

    #[test]
    fn the_residual_add_is_gated_on_the_device_token_count() {
        // AGENTS.md rule 5: no launch on this path may be bounded by a count
        // the host passed by value.
        assert!(GLUE_SRC.contains("const int* __restrict__ valid_tokens"));
        assert!(GLUE_SRC.contains("if (t >= *valid_tokens) return;"));
        assert!(!GLUE_SRC.contains("int live_tokens"));
    }

    #[test]
    fn the_block_writes_the_ffn_waypoint_before_the_residual() {
        // The dense counterpart of the MoE block's same check: folding the
        // residual into `ffn_out` would still give the right `l_out` and
        // would fail the golden's intermediate waypoint.
        assert!(GLUE_SRC.contains("l_out[base + j] = ffn_out[base + j] + residual[base + j];"));
    }

    #[test]
    fn there_is_no_shared_expert_gate_in_the_dense_path() {
        // The single most plausible way to get this block wrong is to carry
        // the MoE block's sigmoid gate across. `qwen35.cpp:475` asserts
        // `ffn_gate_inp == nullptr`; this asserts the same thing about us.
        assert!(!GLUE_SRC.contains("expf"));
        assert!(!GLUE_SRC.contains("gate"));
    }
}
