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

/// Rows one warp of the split GEMV owns, per token tile.
///
/// Indexed by `tokens - 1`. It falls as the token tile widens for the reason
/// `GdnBlock`'s own table gives — the accumulator array is `RT * TT` floats
/// and the staged weight words another `2 * RT` — but where that block was
/// holding accumulators under a budget, this one is holding *occupancy*, and
/// on this model the two are the same constraint read from opposite ends.
/// ptxas puts `(TT, RT)` at, in registers per thread:
///
/// ```text
///        RT=1   RT=2   RT=4
/// TT=1     56     64     80
/// TT=2     64     64     96
/// TT=3     64     80    128
/// TT=4     64     80    128
/// ```
///
/// A Turing scheduler holds 16,384 registers, so 64 is the cliff: at or below
/// it eight warps fit per scheduler and the SM runs at 32 warps, at 80 it
/// runs 24 and at 128 it runs 16.
///
/// **The table is not the occupancy column.** `RT = 1` is the roomiest entry
/// at every width and the slowest at every width — 2.19 ms against `RT = 4`'s
/// 0.69 at four tokens, more than three times — because the activation
/// `float4` a lane loads feeds `RT` rows, so the activation reads per weight
/// byte are `4 * TT / RT` and that ratio, not the warp count, is what the
/// measurement follows. `RT` rises until the registers stop it: at four
/// tokens `RT = 8` runs a quarter of the warps and is still 10% faster.
/// Numbers and method in `docs/BENCHMARKS.md`.
///
/// `LLMXABE_DENSE_SPLIT_ROWS` overrides every entry with one value, so the
/// table can be re-measured in a single binary rather than argued about.
/// The compiled `(TT, RT)` pairs, in increasing width.
///
/// Not every width has an entry point: past four tokens the pass is a
/// speculative verify rather than a decode step, its width is
/// `(drafts + 1) * sequences` and lands on a handful of values, and each
/// extra instantiation is NVRTC time at every model load. A pass takes the
/// narrowest entry that covers it and masks the rest — the kernel already
/// gates every token on `n_tokens`, so a four-token pass through the
/// six-wide entry is correct, just two rows' worth of arithmetic wasted.
///
/// `RT` is 8 only where the activations are staged (`TT <= 4`). Above that
/// the weights are, and eight rows' worth of them do not fit — see the
/// staging note in `dense_proj_split_rows`.
const SPLIT_GEMV_ENTRIES: [(usize, u32); 8] = [
    (1, 4),
    (2, 4),
    (3, 8),
    (4, 8),
    (6, 4),
    (8, 4),
    (12, 4),
    (16, 4),
];

/// The narrowest compiled entry that covers a `tokens`-wide pass.
///
/// `LLMXABE_DENSE_SPLIT_ROWS` overrides the row tile, so the table can be
/// re-measured in a single binary rather than argued about; it is ignored
/// where the chosen width has no entry point at that tile.
fn split_entry_for(tokens: usize) -> (usize, u32) {
    let (tt, rt) = SPLIT_GEMV_ENTRIES
        .into_iter()
        .find(|&(tt, _)| tt >= tokens)
        .expect("caller checked tokens <= SPLIT_GEMV_MAX_TOKENS");
    let rt = match std::env::var("LLMXABE_DENSE_SPLIT_ROWS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
    {
        Some(r) if SPLIT_GEMV_ENTRIES.contains(&(tt, r)) => r,
        _ => rt,
    };
    (tt, rt)
}

/// Widest pass the split GEMV serves.
///
/// Above this the pass is a prefill chunk and `shared_expert_mma`'s GEMM tile
/// is the right shape; at or below it the tile is mostly empty and the GEMV
/// wins — 63.5 ms against 109.0 ms on a one-token Qwen3.8-27B step.
///
/// Sixteen, not four, because of the speculative verify pass. That pass is
/// `(drafts + 1) * sequences` wide — twelve at the serving target — and the
/// GEMM tile is no less empty there: measured, the block costs 1.374 ms a
/// layer at twelve tokens against 0.569 at four, a step that buys nothing
/// because the weight traffic is identical. Sixteen is the widest entry the
/// register budget reaches; above it the staged weights and `RT * TT`
/// accumulators stop fitting and the GEMM is the right shape again.
const SPLIT_GEMV_MAX_TOKENS: usize = 16;

/// The dense FFN's weight residency, and the one real design decision in this
/// module.
///
/// [`crate::block::moe`] holds Q8_0 *and* a tensor-core repack, because there
/// the repack covers the shared expert alone — 3.5 MB against that layer's
/// 725 MB of routed experts, noise. Here the repack would cover the whole
/// feed-forward block, and holding both does not fit: one layer's three
/// matrices are 267 M elements, which is 271 MiB either way — the split
/// layout keeps the file's own fp16 scale, so it is a re-layout at the same
/// 1.0625 bytes an element and not a widening — and across 64 layers that is
/// 16.9 GiB twice over on a 47.27 GiB card already carrying 11.8 GiB of
/// arena. The engine's first attempt at the dense model died on exactly that
/// `CUDA_ERROR_OUT_OF_MEMORY`.
///
/// The scale width was not always free. Holding it as fp32 cost 1.125 bytes
/// an element, which is 1.01 GiB of every decoded token on a model whose
/// decode step is a streaming problem; narrowing it to the fp16 the file
/// already stores was worth 4.2% of the block at one token and 1.4 GiB of
/// resident VRAM. See `MmaKernels::repack_q8_0_half`.
///
/// So it is still one or the other — the two layouts are the same size, and
/// two copies of 16.9 GiB do not fit beside the arena either. With Q8_0 only,
/// every pass runs the fp32 `moe_shared_ffn` tile and prefill 512 measures
/// **76.2 tok/s** against llama.cpp's 691.2 on the same card — 9.1x slower,
/// which is the whole cost of not reaching the integer tensor cores on a
/// model whose FFN is 85% of its arithmetic. With int8 only, prefill reaches
/// **317.5 tok/s** for no extra byte at all.
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
/// quant and an fp16 scale per 32; the split layout is the same quants and
/// the same fp16 scale, moved into two aligned arrays. Nothing is
/// requantized, and the tensor core multiplies the quants exactly. What *does* change is that the activations
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

// The post-mixer RMSNorm, emitting both widths in one pass.
//
// `LayerOpsKernels::rms_norm` produces `normed`, and the split GEMV then
// needs it narrowed -- two launches over a 20 KiB row, of which the second is
// almost entirely the card starting it. This is the first with an extra
// store: the fp32 output survives because `attn_post_norm-N` is a captured
// waypoint and the golden reads it, and the fp16 output is what the two
// projections actually consume.
//
// **The body is `rms_norm_rows` verbatim**, `block_reduce_sum` included, and
// it has to stay that way: a different reduction order here would make the
// dense block's own norm disagree with every other norm in the model.
// `the_dense_norm_matches_layer_ops` is what keeps the copy honest.
__device__ __forceinline__ float dense_block_reduce_sum(float v, float* scratch) {
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

__global__ void dense_rms_norm_narrow(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    unsigned short* __restrict__ out_h,
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
    float sum_sq = dense_block_reduce_sum(partial, scratch);

    float inv_rms = 1.0f / sqrtf(sum_sq / (float)width + eps);

    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        float v = x[base + j] * inv_rms * weight[j];
        out[base + j] = v;
        unsigned short h;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v));
        out_h[base + j] = h;
    }
}

// act_h[i] = (half)(silu(gate[i]) * up[i]).
//
// `LayerOpsKernels::swiglu` followed by a separate narrowing pass computes
// the same thing in two launches and a full-width round trip through memory;
// the down projection only ever reads the narrowed form, so the fp32
// intermediate has no other reader and does not need to exist.
//
// `expf`, not `__expf`. `layer_ops.rs` has a test asserting its own SwiGLU
// has not drifted to the fast intrinsic, and this has to agree with it byte
// for byte or the two residencies stop being comparable.
//
// This is *not* folded into the up projection's epilogue, which is where it
// was tried first: that epilogue runs on `lane == 0`, so 17,408 transcendental
// calls would land on one lane in thirty-two. It measured 0.585 ms a layer at
// four tokens against 0.560 here -- the launch it saved cost more than it was
// worth. See docs/BENCHMARKS.md.
__global__ void dense_swiglu_narrow(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    unsigned short* __restrict__ act_h,
    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float a = (g / (1.0f + expf(-g))) * up[i];
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(a));
    act_h[i] = h;
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

// One weight scale, widened from the fp16 the split layout stores.
//
// NVRTC compiles from a string with no include path, so <cuda_fp16.h> is
// unreachable and the conversion is the same inline `cvt` every other kernel
// in this workspace uses.
__device__ __forceinline__ float dense_scale(const unsigned short* p, long long i) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(p[i]));
    return f;
}

// Two activation halves out of one packed word.
//
// One `mov.b32` and two `cvt`s, which is what the hardware does anyway; the
// alternative of masking and shifting into `h` constraints costs a `PRMT`
// per half for the same result.
__device__ __forceinline__ void dense_half2(unsigned int packed, float& a, float& b) {
    asm("{ .reg .f16 hl, hh;             \n"
        "  mov.b32 {hl, hh}, %2;         \n"
        "  cvt.f32.f16 %0, hl;           \n"
        "  cvt.f32.f16 %1, hh;         }"
        : "=f"(a), "=f"(b)
        : "r"(packed));
}

// `epilogue`, which is uniform across the whole grid and so costs a branch
// the warp scheduler predicts perfectly:
//
//   DENSE_EP_PLAIN   write `out` and nothing else                 (gate, up)
//   DENSE_EP_ADD     also write `summed = out + residual`, live t only (down)
//
// The extra epilogue exists to retire a launch. At one token a 20 KiB
// elementwise residual add costs about as much as it takes the card to start
// it, 64 times a step, and the projection that would feed it already has the
// value in a register: the reduction has happened and `lane == 0` holds the
// sum, so the only new traffic is the residual row itself.
//
// What does *not* belong here is anything transcendental -- see
// `dense_swiglu_narrow` for the SwiGLU that was tried in this epilogue and
// measured slower, because `lane == 0` is one lane in thirty-two.
//
// `DENSE_EP_ADD` keeps its `valid_tokens` gate. Writing `l_out` on a slot the
// pass never filled would put a plausible value where the caller left a zero,
// which `tests/forward_pass.rs` checks for, and AGENTS.md rule 5 says the
// bound comes from the device scalar rather than the host.
#define DENSE_EP_PLAIN  0
#define DENSE_EP_ADD    2

// Eight quants of block `U` of `CUR`, dequantized by `D` into `W[0..8]`.
//
// Low byte first: the split repack keeps quants in element order, and
// narrowing through `unsigned char` first makes the int8 sign-extend rather
// than taking an implementation-defined conversion from a value above 127 —
// the same idiom as `lm_head.rs`. `GdnBlock`'s copy of this GEMV writes the
// same eight lines out longhand; they are compared by
// `both_split_gemvs_sign_extend_their_quants_the_same_way`.
#define DENSE_UNPACK_EIGHT(W, CUR, D, U)                                     \
    do {                                                                     \
        const unsigned int* q_ = (const unsigned int*)&(CUR);                \
        unsigned int lo_ = q_[2 * (U)];                                      \
        unsigned int hi_ = q_[2 * (U) + 1];                                  \
        float d_ = (D);                                                      \
        (W)[0] = (float)(signed char)(unsigned char)(lo_)       * d_;        \
        (W)[1] = (float)(signed char)(unsigned char)(lo_ >> 8)  * d_;        \
        (W)[2] = (float)(signed char)(unsigned char)(lo_ >> 16) * d_;        \
        (W)[3] = (float)(signed char)(unsigned char)(lo_ >> 24) * d_;        \
        (W)[4] = (float)(signed char)(unsigned char)(hi_)       * d_;        \
        (W)[5] = (float)(signed char)(unsigned char)(hi_ >> 8)  * d_;        \
        (W)[6] = (float)(signed char)(unsigned char)(hi_ >> 16) * d_;        \
        (W)[7] = (float)(signed char)(unsigned char)(hi_ >> 24) * d_;        \
    } while (0)

template <int TT, int RT>
__device__ __forceinline__ void dense_proj_split_rows(
    const signed char* __restrict__ wq,
    const unsigned short* __restrict__ ws,
    const unsigned short* __restrict__ x,
    const float* __restrict__ residual,
    float* __restrict__ out,
    float* __restrict__ summed,
    const int* __restrict__ valid_tokens,
    int epilogue,
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
    const unsigned short* sc[RT];
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
        dcur[r] = dense_scale(sc[r], off >> 5);
    }

    for (int c = 0; c < k_dim; c += 512) {
        int cn = c + 512;
        uint4 nxt[RT];
        float dnxt[RT];
        if (cn < k_dim) {
            #pragma unroll
            for (int r = 0; r < RT; ++r) {
                nxt[r]  = *(const uint4*)(row[r] + cn + off);
                dnxt[r] = dense_scale(sc[r], (cn + off) >> 5);
            }
        }
        // Two steps of eight elements rather than four of four: the lane's
        // sixteen contiguous activations are thirty-two bytes as halves, so
        // two `uint4` loads cover what four did. **The order of the `+=`
        // chain is unchanged** — element `off+0` through `off+15` in
        // sequence, then the next 512-element step — which is what keeps
        // this a change of activation *precision* and not of summation.
        //
        // Which operand is staged is chosen by `TT`, and both branches
        // accumulate in that same order, so they agree to the last bit.
        //
        //   TT <= 4   stage the tokens' activations, `TT * 8` live, and
        //             dequantize one row's weights at a time. This is the
        //             narrow decode shape, where `RT` is 8 and the other
        //             staging would hold 64 weights live.
        //   TT >  4   stage all `RT` rows' weights, `RT * 8` live, and take
        //             the tokens one at a time. `RT` is 4 here, so the live
        //             set stops growing with `TT` and a verify pass twelve
        //             wide fits where `TT * 8` activations would spill.
        //
        // Measured, not assumed: the wide staging costs 3% at three tokens
        // and 22% at four, which is why it is not simply used everywhere.
        #pragma unroll
        for (int u = 0; u < 2; ++u) {
            int e0 = c + off + 8 * u;
            if (TT <= 4) {
                float xv[TT][8];
                #pragma unroll
                for (int i = 0; i < TT; ++i) {
                    int t = t0 + i;
                    uint4 h = (t < n_tokens)
                        ? *(const uint4*)(x + (long long)t * k_dim + e0)
                        : make_uint4(0u, 0u, 0u, 0u);
                    const unsigned int* hw = (const unsigned int*)&h;
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        dense_half2(hw[e], xv[i][2 * e], xv[i][2 * e + 1]);
                    }
                }
                #pragma unroll
                for (int r = 0; r < RT; ++r) {
                    float w[8];
                    DENSE_UNPACK_EIGHT(w, cur[r], dcur[r], u);
                    #pragma unroll
                    for (int i = 0; i < TT; ++i) {
                        #pragma unroll
                        for (int e = 0; e < 8; ++e) acc[r][i] += w[e] * xv[i][e];
                    }
                }
            } else {
                float ww[RT][8];
                #pragma unroll
                for (int r = 0; r < RT; ++r) {
                    DENSE_UNPACK_EIGHT(ww[r], cur[r], dcur[r], u);
                }
                #pragma unroll
                for (int i = 0; i < TT; ++i) {
                    int t = t0 + i;
                    uint4 h = (t < n_tokens)
                        ? *(const uint4*)(x + (long long)t * k_dim + e0)
                        : make_uint4(0u, 0u, 0u, 0u);
                    const unsigned int* hw = (const unsigned int*)&h;
                    float xv[8];
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        dense_half2(hw[e], xv[2 * e], xv[2 * e + 1]);
                    }
                    #pragma unroll
                    for (int r = 0; r < RT; ++r) {
                        #pragma unroll
                        for (int e = 0; e < 8; ++e) acc[r][i] += ww[r][e] * xv[e];
                    }
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
                if (epilogue == DENSE_EP_ADD && (t0 + i) < *valid_tokens) {
                    summed[o] = sum + residual[o];
                }
            }
        }
    }
}


#define DENSE_PROJ_SPLIT_ENTRY(NAME, TT, RT)                                 \
extern "C" __global__ void NAME(                                             \
    const signed char* __restrict__ wq,                                      \
    const unsigned short* __restrict__ ws,                                   \
    const unsigned short* __restrict__ x,                                    \
    float* __restrict__ out,                                                 \
    const float* __restrict__ residual,                                      \
    float* __restrict__ summed,                                              \
    const int* __restrict__ valid_tokens,                                    \
    int epilogue,                                                            \
    int k_dim,                                                               \
    int n_rows,                                                              \
    int n_tokens                                                             \
) {                                                                          \
    dense_proj_split_rows<TT, RT>(                                           \
        wq, ws, x, residual, out, summed, valid_tokens,                      \
        epilogue, k_dim, n_rows, n_tokens);                                  \
}

DENSE_PROJ_SPLIT_ENTRY(dense_proj_split_t1_r4, 1, 4)
DENSE_PROJ_SPLIT_ENTRY(dense_proj_split_t2_r4, 2, 4)
DENSE_PROJ_SPLIT_ENTRY(dense_proj_split_t3_r4, 3, 4)
DENSE_PROJ_SPLIT_ENTRY(dense_proj_split_t4_r4, 4, 4)
DENSE_PROJ_SPLIT_ENTRY(dense_proj_split_t1_r8, 1, 8)
DENSE_PROJ_SPLIT_ENTRY(dense_proj_split_t2_r8, 2, 8)
DENSE_PROJ_SPLIT_ENTRY(dense_proj_split_t3_r8, 3, 8)
DENSE_PROJ_SPLIT_ENTRY(dense_proj_split_t4_r8, 4, 8)
DENSE_PROJ_SPLIT_ENTRY(dense_proj_split_t6_r4,   6, 4)
DENSE_PROJ_SPLIT_ENTRY(dense_proj_split_t8_r4,   8, 4)
DENSE_PROJ_SPLIT_ENTRY(dense_proj_split_t12_r4, 12, 4)
DENSE_PROJ_SPLIT_ENTRY(dense_proj_split_t16_r4, 16, 4)
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
    /// The one `(TT, RT)` entry point this geometry's width selects, and the
    /// rows per warp it was compiled for — the launch grid is derived from
    /// the same number, so the two cannot drift.
    tile: CudaFunction,
    rows_per_warp: u32,
    /// `dense_rms_norm_narrow`, which produces `normed` and `normed_h`
    /// together and so replaces the block's first two launches with one.
    rms_norm_narrow: CudaFunction,
    /// `dense_swiglu_narrow`, which produces `activated_h` from the gate and
    /// up projections in one pass.
    swiglu_narrow: CudaFunction,
    gate_out: CudaSlice<f32>,
    up_out: CudaSlice<f32>,
    /// The GEMV's activation operands, at half width. See
    /// [`DENSE_NARROW_ACTIVATIONS`].
    normed_h: CudaSlice<u16>,
    activated_h: CudaSlice<u16>,
    /// One element each, for the epilogue pointers a given launch does not
    /// read. See [`EpilogueArgs`].
    pad_f32: CudaSlice<f32>,
    pad_f32_out: CudaSlice<f32>,
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
            let (tile_tokens, rows_per_warp) = split_entry_for(geometry.max_tokens);
            let tokens = geometry.max_tokens;
            Some(SplitGemv {
                tile: module
                    .load_function(&format!("dense_proj_split_t{tile_tokens}_r{rows_per_warp}"))?,
                rows_per_warp,
                rms_norm_narrow: module.load_function("dense_rms_norm_narrow")?,
                swiglu_narrow: module.load_function("dense_swiglu_narrow")?,
                gate_out: stream.alloc_zeros::<f32>(n)?,
                up_out: stream.alloc_zeros::<f32>(n)?,
                normed_h: stream.alloc_zeros::<u16>(tokens * geometry.hidden)?,
                activated_h: stream.alloc_zeros::<u16>(n)?,
                pad_f32: stream.alloc_zeros::<f32>(1)?,
                pad_f32_out: stream.alloc_zeros::<f32>(1)?,
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
    #[allow(clippy::too_many_arguments)]
    fn mlp_split_gemv(
        &mut self,
        stream: &Arc<CudaStream>,
        w: &SharedExpertInt8,
        residual: &CudaSlice<f32>,
        ffn_out: &mut CudaSlice<f32>,
        l_out: &mut CudaSlice<f32>,
    ) -> Result<(), MoeBlockError> {
        let g = self.geometry;
        let valid = self.buffers.valid_tokens();
        let split = self.split.as_mut().expect("caller checked");
        let f = &split.tile;
        let rt = split.rows_per_warp;

        let (gate_q, gate_s) = w.gate();
        let (up_q, up_s) = w.up();
        let (down_q, down_s) = w.down();

        project_split(
            stream,
            f,
            gate_q,
            gate_s,
            &split.normed_h,
            &mut split.gate_out,
            EpilogueArgs {
                kind: 0,
                residual: &split.pad_f32,
                summed: &mut split.pad_f32_out,
                valid_tokens: valid,
            },
            g.hidden,
            g.intermediate,
            g.max_tokens,
            rt,
        )?;
        project_split(
            stream,
            f,
            up_q,
            up_s,
            &split.normed_h,
            &mut split.up_out,
            EpilogueArgs {
                kind: 0,
                residual: &split.pad_f32,
                summed: &mut split.pad_f32_out,
                valid_tokens: valid,
            },
            g.hidden,
            g.intermediate,
            g.max_tokens,
            rt,
        )?;
        swiglu_narrow(
            stream,
            &split.swiglu_narrow,
            &split.gate_out,
            &split.up_out,
            &mut split.activated_h,
            g.max_tokens * g.intermediate,
        )?;
        project_split(
            stream,
            f,
            down_q,
            down_s,
            &split.activated_h,
            ffn_out,
            EpilogueArgs {
                kind: 2,
                residual,
                summed: l_out,
                valid_tokens: valid,
            },
            g.intermediate,
            g.hidden,
            g.max_tokens,
            rt,
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
        //
        //    The split path runs its own copy of that kernel, which emits the
        //    fp16 operand the projections read from the same registers rather
        //    than in a second pass over the row. See `dense_rms_norm_narrow`.
        match &mut self.split {
            Some(split) => dense_rms_norm(
                stream,
                &split.rms_norm_narrow,
                residual,
                &w.post_norm,
                &mut self.normed,
                &mut split.normed_h,
                g.max_tokens,
                g.hidden,
                self.eps,
            )?,
            None => self.layer_ops.rms_norm(
                stream,
                residual,
                &w.post_norm,
                &mut self.normed,
                g.max_tokens,
                g.hidden,
                self.eps,
            )?,
        }

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
                    // The down projection's epilogue writes `l_out`, so this
                    // path skips the residual-add launch below.
                    self.mlp_split_gemv(stream, i8w, residual, ffn_out, l_out)?;
                    return Ok(());
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

        // 3. the residual add, for the tensor-core path only -- the split
        //    GEMV's down projection does it in its own epilogue and returned
        //    above. `ffn_out-N` survives as its own waypoint either way,
        //    which is what the oracle comparison hangs on; what the fused
        //    form removes is a launch, not a buffer.
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

/// The block's own post-mixer RMSNorm: `normed` and `normed_h` in one pass.
///
/// Launch shape and shared memory are [`LayerOpsKernels::rms_norm`]'s,
/// because the kernel body is — one block a row, a block as wide as the row
/// rounded up to a warp and capped at 1,024, one float of scratch a warp.
/// `the_dense_norm_matches_layer_ops` is what keeps the two from drifting.
#[allow(clippy::too_many_arguments)]
fn dense_rms_norm(
    stream: &Arc<CudaStream>,
    f: &CudaFunction,
    x: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    out_h: &mut CudaSlice<u16>,
    rows: usize,
    width: usize,
    eps: f32,
) -> Result<(), MoeBlockError> {
    check_len("dense_rms_norm x", rows * width, x.len())?;
    check_len("dense_rms_norm weight", width, weight.len())?;
    check_len("dense_rms_norm out", rows * width, out.len())?;
    check_len("dense_rms_norm out_h", rows * width, out_h.len())?;

    let block = width.next_multiple_of(32).clamp(32, 1024);
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (block as u32, 1, 1),
        // One float per warp, which is the most `dense_block_reduce_sum`
        // stores.
        shared_mem_bytes: (block.div_ceil(32) * size_of::<f32>()) as u32,
    };
    let width_i32 = width as i32;
    let mut builder = stream.launch_builder(f);
    builder
        .arg(x)
        .arg(weight)
        .arg(&mut *out)
        .arg(&mut *out_h)
        .arg(&width_i32)
        .arg(&eps);
    // SAFETY: one block per row over buffers of `rows * width`, checked
    // above, with the in-row loop bounded by `width`, so every thread's
    // `blockIdx.x * width + j` is in range. `weight` is indexed by `j <
    // width` and holds that many floats. Shared memory covers one float per
    // warp, which is all the reduction writes.
    unsafe { builder.launch(cfg) }?;
    Ok(())
}

/// `act_h = (half)(silu(gate) * up)`, one launch for what was three.
fn swiglu_narrow(
    stream: &Arc<CudaStream>,
    f: &CudaFunction,
    gate: &CudaSlice<f32>,
    up: &CudaSlice<f32>,
    act_h: &mut CudaSlice<u16>,
    elements: usize,
) -> Result<(), MoeBlockError> {
    let n = elements as i32;
    let cfg = LaunchConfig {
        grid_dim: ((elements as u32).div_ceil(THREADS), 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(f);
    builder.arg(gate).arg(up).arg(&mut *act_h).arg(&n);
    // SAFETY: one thread per element over a grid that covers `elements` and
    // returns above it; all three buffers are that long by the caller's
    // geometry.
    unsafe { builder.launch(cfg) }?;
    Ok(())
}

/// The five pointers and the selector a projection's epilogue needs.
///
/// Every pointer is passed on every launch, because the kernel's selector is
/// what decides which are read and cudarc has no null device pointer to pass
/// for the rest. The unused ones are handed the block's one-element pads,
/// which exist for exactly this and are never dereferenced.
struct EpilogueArgs<'a> {
    kind: i32,
    residual: &'a CudaSlice<f32>,
    summed: &'a mut CudaSlice<f32>,
    valid_tokens: &'a CudaSlice<i32>,
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
    ws: &CudaSlice<u16>,
    x: &CudaSlice<u16>,
    out: &mut CudaSlice<f32>,
    ep: EpilogueArgs<'_>,
    k_dim: usize,
    n_rows: usize,
    tokens: usize,
    rows_per_warp: u32,
) -> Result<(), MoeBlockError> {
    let cfg = LaunchConfig {
        grid_dim: ((n_rows as u32).div_ceil(SPLIT_WARPS * rows_per_warp), 1, 1),
        block_dim: (32, SPLIT_WARPS, 1),
        shared_mem_bytes: 0,
    };
    let (k_i32, n_i32, t_i32) = (k_dim as i32, n_rows as i32, tokens as i32);
    let kind = ep.kind;
    let EpilogueArgs {
        residual,
        summed,
        valid_tokens,
        ..
    } = ep;
    let mut builder = stream.launch_builder(f);
    builder
        .arg(wq)
        .arg(ws)
        .arg(x)
        .arg(&mut *out)
        .arg(residual)
        .arg(summed)
        .arg(valid_tokens)
        .arg(&kind)
        .arg(&k_i32)
        .arg(&n_i32)
        .arg(&t_i32);
    // SAFETY: one warp per group of `rows_per_warp` adjacent output rows over a
    // grid that covers `n_rows` and returns above it. `k_dim` is a multiple of
    // 512 and `n_rows` of `rows_per_warp` — both hold at this model's widths and
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

    /// The two copies of the split-layout projection GEMV read the weights
    /// in the same order, and that part must not drift.
    ///
    /// `dense_proj_split_rows` began as a copy of `GdnBlock`'s
    /// `gdn_proj_split_rows`, and the module comment above says why it is a
    /// copy rather than a shared translation unit. The two are **no longer
    /// identical**: this one takes its activations as halves and consumes
    /// them eight at a time, and it selects its epilogue from an argument
    /// instead of a template parameter. Both differences are on the
    /// activation side. See `DENSE_NARROW_ACTIVATIONS`.
    ///
    /// What is still a copy is the *weight* traversal — the row and scale
    /// base addresses, the 512-element step, the lane's 16-byte window, and
    /// the hand-rolled prefetch that Turing needs for want of `cp.async`.
    /// That is the half whose drift would be silent and expensive: both
    /// kernels would still compile, still run, and quietly read a different
    /// scale for a block. So it is compared token for token, and the
    /// activation half is left to the differential test.
    #[test]
    fn the_split_gemv_weight_traversal_matches_the_gdn_blocks() {
        /// From the first statement of the body to the end of the prefetch
        /// block, which is the last thing either kernel does before it
        /// touches an activation.
        fn weight_half(src: &str, prefix: &str) -> String {
            let open = format!("void {prefix}_proj_split_rows(");
            let at = src
                .find(&open)
                .unwrap_or_else(|| panic!("{prefix}_proj_split_rows is missing"));
            let from = at
                + src[at..]
                    .find("int lane = threadIdx.x;")
                    .unwrap_or_else(|| panic!("{prefix} body does not open on the lane index"));
            const END: &str = "(cn + off) >> 5);";
            let to = from
                + src[from..]
                    .find(END)
                    .unwrap_or_else(|| panic!("{prefix} body has no scale prefetch"))
                + END.len();
            src[from..to]
                .replace(&format!("{prefix}_"), "")
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        }
        let dense = weight_half(GLUE_SRC, "dense");
        let gdn = weight_half(crate::block::gdn::GDN_BLOCK_SRC, "gdn");
        assert_eq!(
            dense, gdn,
            "the dense FFN's split GEMV reads weights differently from \
             GdnBlock's; that half is kept identical on purpose -- see the \
             comment above `GLUE_SRC`"
        );
    }

    /// Both kernels narrow a weight byte through `unsigned char` first.
    ///
    /// Dropping the intermediate cast makes a quant above 127 take an
    /// implementation-defined conversion instead of sign-extending, which is
    /// wrong on half the weights and produces text that still reads fluently.
    #[test]
    fn both_split_gemvs_sign_extend_their_quants_the_same_way() {
        const IDIOM: &str = "(float)(signed char)(unsigned char)";
        assert!(GLUE_SRC.contains(IDIOM));
        assert!(crate::block::gdn::GDN_BLOCK_SRC.contains(IDIOM));
    }

    /// The block's own RMSNorm has not drifted from `LayerOpsKernels`'.
    ///
    /// `dense_rms_norm_narrow` exists only to add a second store; everything
    /// else is `rms_norm_rows` verbatim, `block_reduce_sum` included. A
    /// different summation order here would put this one layer's norm out of
    /// step with every other norm in the model, by an amount no benchmark
    /// would show and the `forward_pass` golden might not either.
    #[test]
    fn the_dense_norm_matches_layer_ops() {
        fn between<'a>(src: &'a str, open: &str, close: &str) -> &'a str {
            let at = src
                .find(open)
                .unwrap_or_else(|| panic!("the kernel source no longer has `{open}`"));
            let end = at
                + src[at..]
                    .find(close)
                    .unwrap_or_else(|| panic!("`{open}` no longer ends at `{close}`"));
            &src[at..end]
        }
        fn tokens(src: &str) -> String {
            src.split_whitespace().collect::<Vec<_>>().join(" ")
        }

        // The reduction, token for token. This is the part whose drift
        // would be silent: a different tree gives a different sum.
        let dense_reduce = tokens(
            &between(
                GLUE_SRC,
                "__device__ __forceinline__ float dense_block_reduce_sum(",
                "\n}\n",
            )
            .replace("dense_", ""),
        );
        let shared_reduce = tokens(between(
            xabe_cuda::kernels::layer_ops::LAYER_OPS_SRC,
            "__device__ __forceinline__ float block_reduce_sum(",
            "\n}\n",
        ));
        assert_eq!(
            dense_reduce, shared_reduce,
            "the dense block's copy of `block_reduce_sum` has drifted",
        );

        // And the arithmetic around it. The stores are what the copy exists
        // to change, so they are the only lines not compared.
        let dense_body = between(GLUE_SRC, "__global__ void dense_rms_norm_narrow(", "\n}\n");
        let shared_body = between(
            xabe_cuda::kernels::layer_ops::LAYER_OPS_SRC,
            "__global__ void rms_norm_rows(",
            "\n}\n",
        );
        for line in [
            "long long base = (long long)blockIdx.x * width;",
            "for (int j = threadIdx.x; j < width; j += blockDim.x) {",
            "partial += v * v;",
            "float inv_rms = 1.0f / sqrtf(sum_sq / (float)width + eps);",
        ] {
            assert!(dense_body.contains(line), "the dense norm lost `{line}`");
            assert!(
                shared_body.contains(line),
                "`rms_norm_rows` moved on: `{line}`"
            );
        }
        // `layer_ops.rs` has its own test pinning this; the copy needs the
        // same one or the two diverge the moment that one is relaxed.
        assert!(!dense_body.contains("rsqrtf"), "the dense norm took rsqrtf");
    }

    #[test]
    fn every_row_tile_the_table_selects_is_a_compiled_entry_point() {
        // `DenseFfnBlock::new` builds the entry-point name by formatting the
        // table's value, so a value with no matching `DENSE_PROJ_SPLIT_ENTRY`
        // is a runtime `load_function` failure at model load rather than a
        // compile error. This is what turns it back into one.
        for (tt, rt) in SPLIT_GEMV_ENTRIES {
            assert!(
                GLUE_SRC.contains(&format!("dense_proj_split_t{tt}_r{rt},")),
                "dense_proj_split_t{tt}_r{rt} is not instantiated",
            );
        }
        // Every width the block will accept has to land on one of them.
        for tokens in 1..=SPLIT_GEMV_MAX_TOKENS {
            let (tt, rt) = split_entry_for(tokens);
            assert!(
                tt >= tokens,
                "a {tokens}-token pass would take a {tt}-wide entry"
            );
            assert!(
                SPLIT_GEMV_ENTRIES.contains(&(tt, rt)),
                "a {tokens}-token pass selects ({tt}, {rt}), which is not a compiled entry",
            );
        }
        assert_eq!(
            SPLIT_GEMV_ENTRIES.last().expect("non-empty").0,
            SPLIT_GEMV_MAX_TOKENS,
            "the widest entry and the width gate have drifted apart",
        );
    }

    #[test]
    fn there_is_no_shared_expert_gate_in_the_dense_path() {
        // The single most plausible way to get this block wrong is to carry
        // the MoE block's sigmoid gate across. `qwen35.cpp:475` asserts
        // `ffn_gate_inp == nullptr`; this asserts the same thing about us.
        //
        // Scoped to the projection, not the module: `dense_swiglu_narrow`
        // legitimately has both an `expf` and a `gate` argument, because
        // SwiGLU's own gate half is not a router.
        let at = GLUE_SRC
            .find("void dense_proj_split_rows(")
            .expect("the projection is missing");
        let end = at
            + GLUE_SRC[at..]
                .find("\n#define ")
                .expect("it does not close");
        let proj = &GLUE_SRC[at..end];
        assert!(!proj.contains("expf"));
        assert!(!proj.contains("sigmoid"));
        assert!(!proj.contains("gate"));
    }
}
