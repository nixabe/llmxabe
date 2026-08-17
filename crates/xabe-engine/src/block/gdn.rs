//! The Gated DeltaNet block: 30 of Qwen3.6's 40 layers.
//!
//! This assembles the mixer that [`xabe_cuda::kernels::gdn`] (decode) and
//! [`xabe_cuda::kernels::gdn_chunked`] (prefill) sit at the centre of, plus
//! everything around them that turns a residual-stream hidden state into the
//! next one: the input norm, the fused q/k/v projection, the short causal
//! convolution, the gating projections, the output norm-and-gate, the output
//! projection, and the residual add.
//!
//! The authority for every step is `src/models/qwen35moe.cpp`
//! (`llama_model_qwen35moe::graph::build_layer_attn_linear`) and
//! `src/models/delta-net-base.cpp`, not a paper. `docs/ORACLE.md` describes the
//! capture of that implementation's intermediates, and
//! `crates/xabe-engine/tests/gdn_block.rs` gates this file against them
//! step by step.
//!
//! # The order, and what each step is checked against
//!
//! | Step | Golden node |
//! | --- | --- |
//! | RMSNorm over `hidden` with `attn_norm` | `attn_norm-N` |
//! | `attn_qkv` projection | `linear_attn_qkv_mixed-N` |
//! | causal depthwise conv1d, width 4 | `conv_output_raw-N` |
//! | SiLU | `conv_output_silu-N` |
//! | slice q/k/v, L2-normalize q/k | `q_conv_predelta-N`, … |
//! | `ssm_alpha` -> softplus -> `* ssm_a` | `alpha-N`, `a_softplus-N`, `gate-N` |
//! | `ssm_beta` -> sigmoid | `beta-N`, `beta_sigmoid-N` |
//! | the gated delta rule | (folded into `final_output-N`) |
//! | RMSNorm over `head_dim`, times `silu(z)` | `z-N`, `final_output-N` |
//! | `ssm_out` projection | `linear_attn_out-N` |
//! | residual add | `attn_residual-N` |
//!
//! # Three things that are easy to get wrong and are pinned here
//!
//! **The query/key heads are shared by *modulo*, not by division.** There are
//! 32 value heads and 16 query/key heads. llama.cpp's fused op indexes the
//! query/key head as `fastmodulo(h_idx, n_k_heads)`
//! (`ggml/src/ggml-cuda/gated_delta_net.cu`), and its non-fused fallback
//! reaches the same mapping through `ggml_repeat_4d`, which tiles rather than
//! stretches. So value head `h` reads query/key head `h % 16`, and value heads
//! 0 and 16 share a query/key head — *not* 0 and 1.
//!
//! [`xabe_cuda::kernels::gdn`] and [`xabe_cuda::kernels::gdn_chunked`] **used
//! to** hard-code the other convention, `qk_head = h / heads_per_kv`, and this
//! block worked around it: [`GdnBlock::split_qkv`] materialised the broadcast
//! itself and both kernels were constructed with `qk_heads == value_heads` so
//! their internal mapping became the identity. That cost the chunked form its
//! two Gram matrices 32 times per chunk instead of 16.
//!
//! Both kernels now compute `qk_head = h % qk_heads` themselves (`d4d2f4d`),
//! so the workaround is gone: they are constructed with the real 16 and
//! [`GdnBlock::split_qkv`] emits a plain `[tokens][qk_heads][head_dim]` slice.
//! `gdn_block.rs`'s `the_query_key_head_broadcast_is_modulo_not_division`
//! still discriminates the two mappings against the captured `final_output-N`,
//! on captured tensors only, so the convention stays pinned by a measurement
//! rather than by this paragraph.
//!
//! **What removing it bought, measured** — `gdn_chunked::prefill` alone, one
//! Quadro RTX 8000, 32 value heads, `chunk_len = 64`, the two constructions
//! driven back to back in the same process:
//!
//! | tokens | chunks | `qk_heads = 32` | `qk_heads = 16` |
//! | --- | --- | --- | --- |
//! | 19 | 1 | 0.280 ms | 0.267 ms |
//! | 512 | 8 | 8.798 ms | 7.581 ms |
//! | 2048 | 32 | 35.203 ms | 30.343 ms |
//!
//! Half the Gram work is **14%** of the prefill's wall clock, not half of it:
//! the two `chunk_len x chunk_len` Gram matrices are a minority of the kernel
//! sequence and the triangular solve dominates. The q/k buffers do halve
//! outright — 67 MiB to 33.5 MiB of scratch at 2048 tokens, plus the two Gram
//! buffers from 1 MiB to 512 KiB — which is the larger of the two wins and the
//! one that does not depend on how the solve is shaped.
//!
//! **`ssm_a` is stored already negated.** `gate-N = softplus(alpha + dt_bias) *
//! ssm_a` *is* the per-head log-decay; there is no second negation. Every entry
//! of `blk.N.ssm_a` is negative in the file.
//!
//! **The SiLU sits between the convolution and the q/k/v slice.** It applies to
//! the whole fused stream, so q, k and v are all convolved *and* activated
//! before anything is sliced apart.
//!
//! # A landed kernel that this block once could not use for prefill
//!
//! [`Self::forward`](GdnBlock::forward) dispatches a multi-token batch to
//! [`xabe_cuda::kernels::gdn_chunked`]. That kernel **used to** fail on this
//! model's numbers: `gdn_chunk_solve_and_apply` formed `v_t / lambda_t` with
//! `lambda_t = exp(sum_{i<=t} log_decay_i)`, and Qwen3.6's per-token
//! log-decays reach **-91.58** (block 0, head 9), so `lambda` was 2.5e-42 by
//! the second token, the quotient overflowed fp32, and the chunk filled with
//! `inf` and then `NaN`. Block 20 did the same by token 9; block 4, whose
//! worst per-token log-decay is only -5.99, came through cleanly throughout.
//!
//! It was fixed in `d4d2f4d` by substituting `u'_t = lambda_t u_t`, which
//! leaves every surviving factor as `exp` of a *non-positive* sum of real
//! per-token log-decays and therefore in `(0, 1]`. llama.cpp was immune in
//! both of its forms for the same reason: its fused CUDA op is recurrent and
//! never accumulates a decay, and `build_delta_net_chunking` only ever
//! *multiplies* by `exp(g_cum)` and `exp(g_cum_last - g_cum)`.
//! `gdn_block.rs`'s
//! `the_chunked_prefill_kernel_overflows_on_this_models_decay_rates` still
//! reports where the cumulative decay would have overflowed, and now gates the
//! chunked form against the recurrent one instead of failing.
//!
//! # What is deliberately not done here
//!
//! The projections are computed in fp32 against dequantized Q8_0 weights.
//! llama.cpp's CUDA backend instead quantizes the *activation* to Q8_1 and uses
//! integer dot products (`mul_mat_q` / `mul_mat_vec_q`), which is a different
//! and slightly lossier computation. That is the dominant term in this block's
//! disagreement with the capture, it is llama.cpp being less accurate rather
//! than this block being wrong, and `gdn_block.rs` measures both against an f64
//! host reference rather than asserting the claim in prose. Matching llama.cpp
//! bit-for-bit would mean reimplementing its activation quantization, which is
//! an engine-wide decision and not this file's to make.
//!
//! # Allocation
//!
//! [`LayerOpsKernels`] validates `rows * width == buffer.len()` exactly, so the
//! scratch cannot be sized once for a maximum and reused at a shorter length.
//! It is therefore sized to the live token count and cached: a decode loop at a
//! constant batch size allocates on its first call and never again. The
//! recurrent state and the convolution cache live in [`GdnState`] and are never
//! reallocated.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};

use xabe_cuda::kernels::compile;
use xabe_cuda::kernels::gdn::{GdnError, GdnKernels, GdnScratch};
use xabe_cuda::kernels::gdn_chunked::{GdnChunkedError, GdnChunkedKernels, GdnChunkedScratch};
use xabe_cuda::kernels::layer_ops::{LayerOpsError, LayerOpsKernels};
use xabe_cuda::kernels::mma::{MMA_SPLIT_TOKENS, MmaError, MmaKernels};
use xabe_gguf::{GgmlType, GgufFile};
use xabe_model::config::ModelConfig;
use xabe_model::weights::{Directory, Role};

/// Elements per Q8_0 block, and its serialized size.
///
/// Duplicated from [`xabe_cuda::kernels::dequant`] rather than imported so the
/// kernel source below and the host-side row-stride arithmetic are checked
/// against the same two literals; `the_q8_0_row_stride_matches_the_kernel`
/// asserts they agree with the upstream constants.
const QK8_0: usize = 32;
const BLOCK_Q8_0_BYTES: usize = 34;

/// Warps per projection block. One warp owns one output row.
const PROJ_WARPS: u32 = 4;

/// Tokens one warp of the fused alpha/beta gate kernel carries. Mirrors
/// `GATE_TT`.
///
/// `ssm_alpha.weight` and `ssm_beta.weight` are 256 KiB each, and a warp that
/// owned one (head, token) read all 16 KiB of its two rows to do 4,096
/// multiply-adds. Carrying a band of tokens reads them once for `GATE_TT`
/// times the arithmetic, and is bit-identical: the per-thread order over the
/// contraction and the closing `warp_reduce_sum` are both unchanged.
const GATE_TT: u32 = 8;

/// Token tile widths the projection is specialized for, ascending.
///
/// Spelled here and in the kernel source, which NVRTC compiles from a string
/// with no access to Rust constants;
/// `the_projection_tiles_match_the_kernel` asserts the two agree.
///
/// `2` and `4` exist for batched decode, where `tokens` is the number of
/// sequences advancing together rather than a prefill chunk. Below the old
/// floor of 8 this used to fall back to `project_per_sequence`, one untiled
/// launch per sequence — correct, but paying the weight traffic `tokens`
/// times over rather than once.
///
/// Adding a tile is not enough on its own — `proj_tile_for` also had to
/// change how it chooses one. Its old rule, "widest tile no wider than
/// `tokens`", picked tile 2 for 3 tokens: `ceil(3 / 2)` is two *independent*
/// grid.y slices, and each one re-walks the whole weight matrix on its own —
/// the reuse is within a slice, not across them. Two slices is two full
/// weight reads for 3 tokens, worse than it looks, and it measured worse in
/// practice: on `bench_decode_batch` at a 32,768-token context, adding tile
/// 2 under the old rule made N=3 *slower* than the per-sequence fallback it
/// replaced (41.1 ms/step against 38.6 ms). `proj_tile_for` now picks the
/// smallest tile that covers `tokens` in a *single* slice below the widest
/// declared tile, which sends 3 tokens to tile 4 instead — one guarded slice
/// at 75% live, one weight read, not two. 2 tokens still gets tile 2 as the
/// exact fit it always was; only the in-between counts changed. See
/// `docs/BENCHMARKS.md`'s batched-decode section for the regression and the
/// fix, and for the original 8/16 floor, which was measured at 19+ tokens
/// under the old multi-slice rule that still governs tokens at or above the
/// widest declared tile.
const PROJ_TILES: [u32; 4] = [2, 4, 8, 16];

/// Output rows each warp accumulates, per tile width above.
///
/// A warp loads one activation column into registers once and reuses it across
/// all of its rows, so this divides the activation traffic directly. It cannot
/// simply be raised — the accumulator array is `rows * tile` floats per thread,
/// and the register pressure eventually costs more occupancy than the traffic
/// saving buys. Measured at 512 tokens, sweeping (tile, rows):
///
/// ```text
///   (32, 2)  409.42 tok/s     64 accumulators
///   (16, 4)  423.30 tok/s     64 accumulators   <- chosen
///   (16, 8)  413.19 tok/s    128 accumulators
/// ```
///
/// (32, 2) and (16, 4) cost the same registers and differ only in how the
/// traffic splits between weight and activation: the narrower tile re-reads
/// the weight twice as often and the activation four times less, and the
/// activation is the larger term. (16, 8) halves the activation traffic again
/// and gives it back to occupancy.
///
/// Tiles 2 and 4 use `RR = 1`, not 4, and for a different reason than the
/// sweep above: at 512+ tokens grid.x is never the problem, `RR` only trades
/// activation traffic against it. At decode `n_rows` is 512-2048 and `tokens`
/// is 2-4, so grid.y is 1 and grid.x is the *only* axis with any blocks in
/// it — `n_rows.div_ceil(PROJ_WARPS * RR)` is 32-128 blocks at `RR = 4`
/// against 72 SMs, which leaves SMs with nothing to run rather than latency
/// to hide. `RR = 1` quadruples grid.x back to the untiled kernel's own
/// geometry (`n_rows.div_ceil(PROJ_WARPS)`), the one already known to stream
/// well, at the cost of activation traffic that is ~32 KB total at these
/// widths and was never the bottleneck to begin with. See
/// `docs/BENCHMARKS.md`'s batched-decode section for the nsys measurement
/// that found the collapsed grid axis rather than assuming it.
const PROJ_ROWS: [u32; 4] = [1, 1, 4, 4];

/// Which specialization to launch for `tokens`, and its width.
///
/// Two different rules, one on each side of the widest declared tile.
///
/// **Below it**, the smallest tile that covers `tokens` in a single grid.y
/// slice — not the widest tile no wider than `tokens`. A launch is
/// `ceil(tokens / tile)` *independent* slices, and each one re-walks the
/// whole weight matrix on its own regardless of how few of its rows are
/// live: the guarded path still pays close to a full slice's weight-read
/// cost (measured, using a 32-wide tile for a 19-token batch costs 21%,
/// 208.8 to 164.5 tok/s — expensive, but a single expensive slice, not two
/// or three of them). Picking the widest tile no wider than `tokens` instead
/// minimizes the *emptiest* slice's waste and, below the widest declared
/// tile, routinely costs more slices doing it — 3 tokens against tiles [2,
/// 4, 8, 16] picks 2 that way, and `ceil(3 / 2)` is two full weight reads for
/// three tokens, not one; picking the smallest covering tile sends it to 4
/// instead, one guarded slice at 75% live. See `PROJ_TILES`'s doc comment
/// for the measurement that changed this rule.
///
/// **At or above it**, the original "widest tile, however many slices"
/// rule: `tokens` is large enough that repeating the widest tile amortizes
/// the weight read across the most tokens per slice, proven at 128 and 512
/// tokens in `PROJ_ROWS`'s own sweep, and a single covering tile does not
/// exist up here regardless.
fn proj_tile_for(tokens: usize) -> (usize, u32) {
    if let Some(i) = PROJ_TILES.iter().position(|&t| tokens <= t as usize) {
        return (i, PROJ_TILES[i]);
    }
    let widest = PROJ_TILES.len() - 1;
    (widest, PROJ_TILES[widest])
}

/// Threads per block for the elementwise kernels.
const ELEMENTWISE_BLOCK: u32 = 256;

const GDN_BLOCK_SRC: &str = r#"
extern "C" {

// Sum across one warp, leaving the total in every lane.
//
// `__shfl_xor_sync` rather than `__shfl_down_sync` because every lane needs the
// result in the projection kernels' epilogue guard-free form; the butterfly
// costs the same five instructions.
__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    return v;
}

// Reinterpret two little-endian bytes as an IEEE half and widen to float.
//
// Copied verbatim from `xabe_cuda::kernels::dequant`. NVRTC compiles from a
// string with no include path, so <cuda_fp16.h> is unreachable and
// `__half2float` is not available; `cvt.f32.f16` is the same hardware
// conversion it lowers to.
__device__ __forceinline__ float load_half_le(const unsigned char* p) {
    unsigned short bits = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(bits));
    return f;
}

// out[t][n] = sum_k weight[n][k] * x[t][k], with `weight` a Q8_0 GGUF tensor.
//
// A GGUF tensor with dims [K, N] is N rows of K contiguous elements — ne[0] is
// the input width — so row n starts at byte `n * (K/32) * 34`. That is the
// whole of the layout convention `docs/ORACLE.md` section 6.1 proves, applied.
//
// grid: (ceil(N / warps), tokens). block: (32, warps) — one warp per output
// row, one lane per element within a Q8_0 block.
//
// fp32 accumulation against dequantized weights, deliberately: see the module
// docs on what llama.cpp does instead.
__global__ void gdn_proj_q8_0(
    const unsigned char* __restrict__ weight,
    const float* __restrict__ x,
    float* __restrict__ out,
    int k_dim,
    int n_rows
) {
    int lane = threadIdx.x;
    int n    = blockIdx.x * blockDim.y + threadIdx.y;
    if (n >= n_rows) return;
    int t = blockIdx.y;

    int blocks = k_dim / 32;
    const unsigned char* row = weight + (long long)n * blocks * 34;
    const float* xr = x + (long long)t * k_dim;

    float acc = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        const unsigned char* blk = row + (long long)b * 34;
        float d = load_half_le(blk);
        signed char q = (signed char)blk[2 + lane];
        acc += (float)q * d * xr[b * 32 + lane];
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        out[(long long)t * n_rows + n] = acc;
    }
}

// The same projection at one token, over the *repacked* split layout.
//
// Q8_0 interleaves a 2-byte scale with every 32 quants, so a row's quants sit
// at byte `b * 34 + 2` and a warp reading `blk[2 + lane]` covers 32 contiguous
// bytes that are 32-byte aligned only when `b == 15 (mod 16)`. Fifteen blocks
// in sixteen straddle a sector boundary, so the fetch is two sectors for
// thirty-two useful bytes — **half of every read is discarded**, which is most
// of why `gdn_proj_q8_0` moves its weights at 47% of this card's streaming
// roofline rather than nearer its ceiling.
//
// `GdnLayerInt8` already separates the quants from the scales for the tensor
// cores. Over that layout a row's quants are `k_dim` contiguous bytes from a
// 32-byte-aligned base, so the same warp reads exactly one sector.
//
// The scale is the identical fp16 value widened the identical way, and each
// product is still `q * d * x`. The *partition* differs — a lane owns four
// consecutive elements rather than one element of each block — so the warp
// reduction sums in a different order and this is equivalent rather than
// bit-identical. It is a plain dot product feeding a projection, not a
// selection, so reassociation here is the ordinary kind.
//
// grid: (ceil(N / warps),). block: (32, warps).
__global__ void gdn_proj_split_gemv(
    const signed char* __restrict__ wq,
    const float* __restrict__ ws,
    const float* __restrict__ x,
    float* __restrict__ out,
    int k_dim,
    int n_rows
) {
    int lane = threadIdx.x;
    int n    = blockIdx.x * blockDim.y + threadIdx.y;
    if (n >= n_rows) return;

    int blocks = k_dim / 32;
    const signed char* row = wq + (long long)n * k_dim;
    const float* sc = ws + (long long)n * (k_dim / 32);

    // Four quants and four activations per lane per step, so one instruction
    // moves 128 bytes of weight across the warp instead of 32. The split
    // layout is what makes this legal: `k_dim` is a multiple of 128 and the
    // row base a multiple of `k_dim`, so both the `char4` and the `float4` are
    // aligned by construction.
    //
    // A lane's four elements always share a Q8_0 block — they start at a
    // multiple of 4 and a block is 32 — so one scale covers all four.
    float acc = 0.0f;
    for (int c = 0; c < k_dim; c += 128) {
        int e0 = c + 4 * lane;
        char4 q = *(const char4*)(row + e0);
        float4 xv = *(const float4*)(x + e0);
        float d = sc[e0 >> 5];
        acc += (float)q.x * d * xv.x;
        acc += (float)q.y * d * xv.y;
        acc += (float)q.z * d * xv.z;
        acc += (float)q.w * d * xv.w;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        out[n] = acc;
    }
}

// The same projection with the block's residual added on the way out.
//
// `linear_attn_out-N` and `attn_residual-N` are the same numbers one add
// apart, and the add was its own launch over 2,048 floats -- 1.8 us against a
// decode step where the smallest kernels *are* their launch floor. The warp
// that produced `acc` is the one that would have read it back, so the add
// costs one instruction and one store here.
//
// `out` is still written: it is a captured waypoint, and a block that got the
// residual source wrong would otherwise have nothing to fail at before
// `attn_residual-N`.
__global__ void gdn_proj_split_gemv_add(
    const signed char* __restrict__ wq,
    const float* __restrict__ ws,
    const float* __restrict__ x,
    const float* __restrict__ residual,
    float* __restrict__ out,
    float* __restrict__ summed,
    int k_dim,
    int n_rows
) {
    int lane = threadIdx.x;
    int n    = blockIdx.x * blockDim.y + threadIdx.y;
    if (n >= n_rows) return;

    int blocks = k_dim / 32;
    const signed char* row = wq + (long long)n * k_dim;
    const float* sc = ws + (long long)n * (k_dim / 32);

    // Four quants and four activations per lane per step, so one instruction
    // moves 128 bytes of weight across the warp instead of 32. The split
    // layout is what makes this legal: `k_dim` is a multiple of 128 and the
    // row base a multiple of `k_dim`, so both the `char4` and the `float4` are
    // aligned by construction.
    //
    // A lane's four elements always share a Q8_0 block — they start at a
    // multiple of 4 and a block is 32 — so one scale covers all four.
    float acc = 0.0f;
    for (int c = 0; c < k_dim; c += 128) {
        int e0 = c + 4 * lane;
        char4 q = *(const char4*)(row + e0);
        float4 xv = *(const float4*)(x + e0);
        float d = sc[e0 >> 5];
        acc += (float)q.x * d * xv.x;
        acc += (float)q.y * d * xv.y;
        acc += (float)q.z * d * xv.z;
        acc += (float)q.w * d * xv.w;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        out[n] = acc;
        summed[n] = acc + residual[n];
    }
}

// The same projection, tiled over tokens.
//
// `gdn_proj_q8_0` above gives each (output row, token) pair its own warp, so
// the weight matrix is re-read once per token. That is invisible at decode,
// where there is one token and the weight is read exactly once — and it is
// catastrophic at prefill, where 512 tokens re-read 1.07 GiB of Gated
// DeltaNet projections 512 times. Measured before this kernel existed:
// `gdn_proj_q8_0` was 1,491.6 ms of a 2,523 ms prefill pass, 59.1%, moving
// 548 GB at 367 GB/s — an efficient kernel doing 512x the necessary work.
//
// Here a warp owns one output row and PROJ_TILE *tokens*. The dequantized
// weight element is loaded once and multiplied into all PROJ_TILE
// accumulators before being dropped, so the stack is read once per tile of
// tokens rather than once per token. This is exactly the transformation that
// took the MoE grouped GEMM from 110.9 ms to 11.9 ms per layer at 512 tokens.
//
// The activations are *not* staged in shared memory. One tile is
// PROJ_TILE * k_dim * 4 = 64 KiB at k_dim 2048, which does not fit in the
// 48 KiB a block may request. They do not need to be: every block in a row
// band reads the same tile, so the 6 MiB L2 serves them, and lanes read
// consecutive `kidx` so each access is coalesced.
//
// grid: (ceil(N / warps), ceil(tokens / PROJ_TILE)). block: (32, warps).
// A warp owns RR output rows and TT tokens.
//
// Two reuses, and both are load-bearing:
//
//   - the dequantized weight element is multiplied into all TT accumulators
//     before being dropped, so the weight stack is read once per *token tile*
//     rather than once per token. Without it, 512 tokens re-read 1.07 GiB of
//     projections 512 times — 59.1% of a prefill pass.
//   - the activation column is loaded into registers once and reused across
//     all RR rows. Without it, every weight row re-reads x, which is ~2.1 GB
//     of L2 traffic per call at 512 tokens.
//
// The tile width is specialized rather than fixed, because a fixed width is
// wrong at one end or the other. Measured tok/s with a fixed width and RR = 1:
//
//   tokens    tile 8   tile 16   tile 32   tile 64
//       19    208.80    208.57    164.52    115.85
//      128    319.15    324.89    332.28    328.05
//      512    339.94    347.20    359.18    360.12
//
// A wide tile wins at 512 and collapses at 19. The cause is the guarded path:
// when fewer tokens are live than the tile is wide, every thread still carries
// the full accumulator array — the register pressure is paid and the work is
// not done. `proj_tile_for` picks the widest fully-live tile.
#define GDN_PROJ_TILED(NAME, TT, RR)                                          \
__global__ void NAME(                                                         \
    const unsigned char* __restrict__ weight,                                 \
    const float* __restrict__ x,                                              \
    float* __restrict__ out,                                                  \
    int k_dim,                                                                \
    int n_rows,                                                               \
    int n_tokens                                                              \
) {                                                                           \
    int lane = threadIdx.x;                                                   \
    int n0   = (blockIdx.x * blockDim.y + threadIdx.y) * (RR);                \
    if (n0 >= n_rows) return;                                                 \
    int t0 = blockIdx.y * (TT);                                               \
    if (t0 >= n_tokens) return;                                               \
                                                                              \
    int blocks = k_dim / 32;                                                  \
    int live_t = n_tokens - t0; if (live_t > (TT)) live_t = (TT);             \
    int live_r = n_rows  - n0; if (live_r > (RR)) live_r = (RR);              \
                                                                              \
    float acc[RR][TT];                                                        \
    _Pragma("unroll")                                                         \
    for (int r = 0; r < (RR); ++r)                                            \
        _Pragma("unroll")                                                     \
        for (int i = 0; i < (TT); ++i) acc[r][i] = 0.0f;                      \
                                                                              \
    if (live_t == (TT) && live_r == (RR)) {                                   \
        for (int b = 0; b < blocks; ++b) {                                    \
            int kidx = b * 32 + lane;                                         \
            /* One load of the activation column, reused across all RR    */  \
            /* weight rows. This is the whole point: without it each row   */  \
            /* re-reads x, and x is re-read ~500 times per pass at 512     */  \
            /* tokens -- 2.1 GB of L2 traffic per projection call.         */  \
            float xv[TT];                                                     \
            _Pragma("unroll")                                                 \
            for (int i = 0; i < (TT); ++i)                                    \
                xv[i] = x[(long long)(t0 + i) * k_dim + kidx];                \
            _Pragma("unroll")                                                 \
            for (int r = 0; r < (RR); ++r) {                                  \
                const unsigned char* blk =                                    \
                    weight + (long long)(n0 + r) * blocks * 34 + b * 34;      \
                float w = (float)(signed char)blk[2 + lane]                   \
                        * load_half_le(blk);                                  \
                _Pragma("unroll")                                             \
                for (int i = 0; i < (TT); ++i) acc[r][i] += w * xv[i];        \
            }                                                                 \
        }                                                                     \
    } else {                                                                  \
        /* Same TT/RR-wide unrolled shape as the fully-live branch above --  \
         * the live count gates each element with a predicate, never bounds \
         * the loop, so `acc`/`xv` stay register-resident instead of        \
         * spilling to local memory the way a runtime trip count forces.    \
         * Out-of-range activation slots read as zero, which makes their    \
         * contribution to `acc` exactly zero; out-of-range row slots       \
         * re-address a real, already-in-bounds row (`n0`) rather than walk \
         * off the end of `weight`, and are simply never read back below.   \
         * See docs/BENCHMARKS.md's batched-decode section for the measured \
         * cost of the runtime-bound form this replaced. */                 \
        for (int b = 0; b < blocks; ++b) {                                    \
            int kidx = b * 32 + lane;                                         \
            float xv[TT];                                                     \
            _Pragma("unroll")                                                 \
            for (int i = 0; i < (TT); ++i)                                    \
                xv[i] = (t0 + i < n_tokens)                                    \
                    ? x[(long long)(t0 + i) * k_dim + kidx]                   \
                    : 0.0f;                                                    \
            _Pragma("unroll")                                                 \
            for (int r = 0; r < (RR); ++r) {                                  \
                int rr = (n0 + r < n_rows) ? (n0 + r) : n0;                    \
                const unsigned char* blk =                                    \
                    weight + (long long)rr * blocks * 34 + b * 34;            \
                float w = (float)(signed char)blk[2 + lane]                   \
                        * load_half_le(blk);                                  \
                _Pragma("unroll")                                             \
                for (int i = 0; i < (TT); ++i) acc[r][i] += w * xv[i];        \
            }                                                                 \
        }                                                                     \
    }                                                                         \
                                                                              \
    for (int r = 0; r < live_r; ++r) {                                        \
        for (int i = 0; i < live_t; ++i) {                                    \
            float sum = warp_reduce_sum(acc[r][i]);                           \
            if (lane == 0)                                                    \
                out[(long long)(t0 + i) * n_rows + (n0 + r)] = sum;           \
        }                                                                     \
    }                                                                         \
}

GDN_PROJ_TILED(gdn_proj_q8_0_t2,  2,  1)
GDN_PROJ_TILED(gdn_proj_q8_0_t4,  4,  1)
GDN_PROJ_TILED(gdn_proj_q8_0_t8,  8,  4)
GDN_PROJ_TILED(gdn_proj_q8_0_t16, 16, 4)

// out[t][n] = sum_k weight[n][k] * x[t][k], with `weight` an f32 GGUF tensor.
//
// `ssm_alpha.weight` and `ssm_beta.weight` are f32 in this file, and are only
// [2048, 32], so they get the simple strided form rather than the block-wise
// one above.
// alpha-N / beta-N and the three gate quantities they feed, in one launch.
//
// `ssm_alpha.weight` and `ssm_beta.weight` are both f32 `[hidden, heads]`,
// both contract the same `normed` activations, and `gdn_gates` reads nothing
// but their two outputs. That was three launches per layer to produce 32
// numbers each -- 0.32 ms of a 10 ms decode step spent on three kernel
// floors, for 131,072 multiply-adds that take microseconds.
//
// Both dot products keep `warp_reduce_sum` and the same lane-strided order
// `gdn_proj_f32` used, so the projections are bit-identical to the kernel
// they replace; the gate arithmetic is elementwise and unchanged. `alpha`
// and `beta_raw` are still written because the golden capture has a waypoint
// on each.
//
// Tokens one warp carries.
//
// The two weight matrices are `[hidden, heads]` f32 -- 256 KiB each -- and a
// warp that owns one (head, token) reads all 16 KiB of its two rows to do
// 4,096 multiply-adds. That is one and a half memory instructions per
// multiply-add, and it made a kernel with 2 GFLOP of work in it cost 4.2 ms of
// a 512-token pass. Carrying GATE_TT tokens reads the same two weight rows
// once for GATE_TT times the arithmetic.
//
// Bit-identical: each (head, token) accumulator still sums `i` lane-strided
// ascending and still finishes in the same `warp_reduce_sum`.
#define GATE_TT 8

#define GDN_GATES(NAME, TT)                                                    \
__global__ void NAME(                                                           \
    const float* __restrict__ w_alpha,                                          \
    const float* __restrict__ w_beta,                                           \
    const float* __restrict__ x,                                                \
    const float* __restrict__ dt_bias,                                          \
    const float* __restrict__ ssm_a,                                            \
    float* __restrict__ alpha,                                                  \
    float* __restrict__ beta_raw,                                               \
    float* __restrict__ a_softplus,                                             \
    float* __restrict__ log_decay,                                              \
    float* __restrict__ beta,                                                   \
    int k_dim,                                                                  \
    int heads,                                                                  \
    int tokens                                                                  \
) {                                                                             \
    int lane = threadIdx.x;                                                     \
    int n    = blockIdx.x * blockDim.y + threadIdx.y;                           \
    if (n >= heads) return;                                                     \
    int t0 = blockIdx.y * TT;                                                   \
                                                                                \
    const float* ra = w_alpha + (long long)n * k_dim;                           \
    const float* rb = w_beta  + (long long)n * k_dim;                           \
                                                                                \
    float aa[TT];                                                               \
    float bb[TT];                                                               \
    _Pragma("unroll")                                                           \
    for (int u = 0; u < TT; ++u) { aa[u] = 0.0f; bb[u] = 0.0f; }                \
                                                                                \
    for (int i = lane; i < k_dim; i += 32) {                                    \
        float av = ra[i];                                                       \
        float bv = rb[i];                                                       \
        _Pragma("unroll")                                                       \
        for (int u = 0; u < TT; ++u) {                                          \
            int t = t0 + u;                                                     \
            float xv = t < tokens ? x[(long long)t * k_dim + i] : 0.0f;         \
            aa[u] += av * xv;                                                   \
            bb[u] += bv * xv;                                                   \
        }                                                                       \
    }                                                                           \
                                                                                \
    _Pragma("unroll")                                                           \
    for (int u = 0; u < TT; ++u) {                                              \
        float a_sum = warp_reduce_sum(aa[u]);                                   \
        float b_sum = warp_reduce_sum(bb[u]);                                   \
        int t = t0 + u;                                                         \
        if (lane == 0 && t < tokens) {                                          \
            long long i = (long long)t * heads + n;                             \
            alpha[i] = a_sum;                                                   \
            beta_raw[i] = b_sum;                                                \
            float a = a_sum + dt_bias[n];                                       \
            float sp = (a > 20.0f) ? a : logf(1.0f + expf(a));                  \
            a_softplus[i] = sp;                                                 \
            log_decay[i] = sp * ssm_a[n];                                       \
            beta[i] = 1.0f / (1.0f + expf(-b_sum));                             \
        }                                                                       \
    }                                                                           \
}

// grid: (ceil(heads / warps), ceil(tokens / TT)). block: (32, warps).
//
// Eight tokens per warp for prefill. One for decode, where seven of the eight
// bands would be masked off and the warp would run eight times the
// multiply-adds and eight `warp_reduce_sum`s to throw seven of them away.
// Measured 105.1 -> 99.6 tok/s before this instantiation existed, which is the
// same trap the flash kernel's query tile set.
GDN_GATES(gdn_alpha_beta_gates,    8)
GDN_GATES(gdn_alpha_beta_gates_t1, 1)

__global__ void gdn_proj_f32(
    const float* __restrict__ weight,
    const float* __restrict__ x,
    float* __restrict__ out,
    int k_dim,
    int n_rows
) {
    int lane = threadIdx.x;
    int n    = blockIdx.x * blockDim.y + threadIdx.y;
    if (n >= n_rows) return;
    int t = blockIdx.y;

    const float* row = weight + (long long)n * k_dim;
    const float* xr  = x + (long long)t * k_dim;

    float acc = 0.0f;
    for (int i = lane; i < k_dim; i += 32) {
        acc += row[i] * xr[i];
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        out[(long long)t * n_rows + n] = acc;
    }
}

// out = silu(x), elementwise.
//
// `expf`, not `__expf`, for the same reason `layer_ops.rs` gives: the fast
// intrinsic's error is relative to the result rather than to the exponent, and
// this is not a bottleneck. The formula is `ggml_cuda_op_silu_single`'s
// verbatim: x / (1 + exp(-x)).
__global__ void gdn_silu(
    const float* __restrict__ x,
    float* __restrict__ out,
    long long n
) {
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float v = x[i];
        out[i] = v / (1.0f + expf(-v));
    }
}

// Slice the convolved q/k/v stream apart. No broadcast: q and k come out at
// their own 16 heads and the mixer kernels do `qk_head = h % qk_heads`.
//
// The stream is [2 * qk_heads * head_dim | value_heads * head_dim] per token,
// contiguous, which is the layout `attn_qkv.weight`'s output width implies and
// `build_layer_attn_linear`'s three `ggml_view_4d` offsets confirm. The three
// outputs are therefore exactly llama.cpp's `q_conv-N` / `k_conv-N`
// (`[head_dim, qk_heads, tokens]`) and `v_conv_predelta-N`
// (`[head_dim, value_heads, tokens]`) — the same shapes, not a widened form.
//
// This kernel **used to** materialise `qk_head = h % qk_heads` itself, because
// both GDN kernels indexed `h / heads_per_kv` internally and had to be handed
// an already-wide q and k. They index by modulo now, so the broadcast is
// theirs and this is a plain slice.
//
// grid: (value_heads, tokens). block: head_dim threads. Blocks with
// `h < qk_heads` write q and k as well as v, so the two ranges are covered by
// one launch and the low 16 heads do the extra pair of stores.
// conv_output_silu-N and the q/k/v split, in one launch.
//
// `gdn_silu` wrote one buffer that `gdn_split_qkv` immediately read, and the
// split touches every element of it exactly once -- the value part once, and
// the query and key parts once each -- so the nonlinearity can be applied on
// the way through. `silu` is still written because the golden capture has a
// waypoint on it, and because a block that got the split wrong would
// otherwise have no intermediate to fail at.
//
// `v / (1 + exp(-v))` is the same expression `gdn_silu` used, in the same
// order, so this is bit-identical to the pair it replaces.
//
// grid: (value_heads, tokens). block: head_dim.
__global__ void gdn_silu_split_qkv(
    const float* __restrict__ conv,
    float* __restrict__ silu,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    int head_dim,
    int qk_heads,
    int value_heads
) {
    int h = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;

    int key_dim  = qk_heads * head_dim;
    int conv_dim = 2 * key_dim + value_heads * head_dim;

    const float* c  = conv + (long long)t * conv_dim;
    float*       sr = silu + (long long)t * conv_dim;

    int iv = 2 * key_dim + h * head_dim + d;
    float vv = c[iv];
    vv = vv / (1.0f + expf(-vv));
    sr[iv] = vv;
    v[((long long)t * value_heads + h) * head_dim + d] = vv;

    if (h < qk_heads) {
        int iq = h * head_dim + d;
        int ik = key_dim + h * head_dim + d;
        float qv = c[iq];
        float kv = c[ik];
        qv = qv / (1.0f + expf(-qv));
        kv = kv / (1.0f + expf(-kv));
        sr[iq] = qv;
        sr[ik] = kv;
        long long o = ((long long)t * qk_heads + h) * head_dim + d;
        q[o] = qv;
        k[o] = kv;
    }
}

__global__ void gdn_split_qkv(
    const float* __restrict__ conv,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    int head_dim,
    int qk_heads,
    int value_heads
) {
    int h = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;

    int key_dim  = qk_heads * head_dim;
    int conv_dim = 2 * key_dim + value_heads * head_dim;

    const float* c = conv + (long long)t * conv_dim;

    v[((long long)t * value_heads + h) * head_dim + d] =
        c[2 * key_dim + h * head_dim + d];

    if (h < qk_heads) {
        long long o = ((long long)t * qk_heads + h) * head_dim + d;
        q[o] = c[h * head_dim + d];
        k[o] = c[key_dim + h * head_dim + d];
    }
}

// The per-head decay and write gates.
//
//   a_softplus = softplus(alpha + ssm_dt.bias)
//   log_decay  = a_softplus * ssm_a          <- already the log-decay
//   beta       = sigmoid(beta_raw)
//
// `ssm_a` is stored already negated, so the product is the log of a decay in
// (0, 1] and must not be negated again. The softplus is
// `ggml_cuda_op_softplus`'s verbatim, including its x > 20 passthrough, which
// matters because a lost passthrough overflows `expf` rather than saturating.
//
// grid-stride over tokens * heads.
__global__ void gdn_gates(
    const float* __restrict__ alpha,
    const float* __restrict__ beta_raw,
    const float* __restrict__ dt_bias,
    const float* __restrict__ ssm_a,
    float* __restrict__ a_softplus,
    float* __restrict__ log_decay,
    float* __restrict__ beta,
    int heads,
    long long n
) {
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        int h = (int)(i % heads);
        float a = alpha[i] + dt_bias[h];
        float sp = (a > 20.0f) ? a : logf(1.0f + expf(a));
        a_softplus[i] = sp;
        log_decay[i] = sp * ssm_a[h];
        beta[i] = 1.0f / (1.0f + expf(-beta_raw[i]));
    }
}

// out = a + b, elementwise. The residual add.
__global__ void gdn_add(
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

}
"#;

/// Something went wrong building or running a Gated DeltaNet block.
#[derive(Debug)]
pub enum GdnBlockError {
    /// The integer tensor-core kernels failed to build or launch.
    Mma(MmaError),
    /// NVRTC rejected this module's source, or the module failed to load.
    Compile(String),
    /// The driver failed.
    Driver(DriverError),
    /// One of the kernel families this block composes rejected a launch.
    LayerOps(LayerOpsError),
    /// The recurrent Gated DeltaNet kernel rejected a launch.
    Recurrent(GdnError),
    /// The chunked Gated DeltaNet kernel rejected a launch.
    Chunked(GdnChunkedError),
    /// The weight directory does not carry a tensor this layer needs.
    ///
    /// Reached when a layer index that is not a Gated DeltaNet layer is asked
    /// for, which is the mistake worth naming: the roles simply do not exist on
    /// an attention block.
    MissingWeight { role: Role, layer: u32 },
    /// A weight is stored in a format this block does not unpack.
    UnsupportedQuant {
        role: Role,
        found: GgmlType,
        expected: GgmlType,
    },
    /// A buffer length disagrees with the declared geometry.
    ShapeMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for GdnBlockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compile(m) => write!(f, "kernel compilation failed: {m}"),
            Self::Mma(e) => write!(f, "integer tensor cores: {e}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::LayerOps(e) => write!(f, "{e}"),
            Self::Recurrent(e) => write!(f, "{e}"),
            Self::Chunked(e) => write!(f, "{e}"),
            Self::MissingWeight { role, layer } => write!(
                f,
                "block {layer} has no `{role}`; is it a Gated DeltaNet layer?",
            ),
            Self::UnsupportedQuant {
                role,
                found,
                expected,
            } => write!(
                f,
                "`{role}` is {}, this block unpacks {}",
                found.name(),
                expected.name(),
            ),
            Self::ShapeMismatch {
                what,
                expected,
                got,
            } => write!(f, "{what} must hold {expected} floats, holds {got}"),
        }
    }
}

impl std::error::Error for GdnBlockError {}

impl From<DriverError> for GdnBlockError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

impl From<LayerOpsError> for GdnBlockError {
    fn from(e: LayerOpsError) -> Self {
        Self::LayerOps(e)
    }
}

impl From<GdnError> for GdnBlockError {
    fn from(e: GdnError) -> Self {
        Self::Recurrent(e)
    }
}

impl From<GdnChunkedError> for GdnBlockError {
    fn from(e: GdnChunkedError) -> Self {
        Self::Chunked(e)
    }
}

/// The shape one Gated DeltaNet block is built for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GdnGeometry {
    /// Residual stream width (2048).
    pub hidden: usize,
    /// Per-head width, shared by q, k and v (128).
    pub head_dim: usize,
    /// Value heads (32).
    pub value_heads: usize,
    /// Query/key heads (16). Value head `h` reads query/key head
    /// `h % qk_heads` — see the module docs.
    pub qk_heads: usize,
    /// Short convolution width (4).
    pub conv_kernel: usize,
    /// Tokens per triangular solve in the prefill form (64).
    pub chunk_len: usize,
    /// Longest sequence the chunked scratch is sized for.
    pub max_tokens: usize,
    /// RMSNorm epsilon, `qwen35moe.attention.layer_norm_rms_epsilon`.
    pub rms_eps: f32,
}

impl GdnGeometry {
    /// The real Qwen3.6 geometry, for sequences of up to `max_tokens`.
    ///
    /// `rms_eps` is not in [`ModelConfig`] — it is a GGUF metadata key — so it
    /// is a parameter rather than a derived constant, and
    /// [`Self::from_gguf`] reads it from the file.
    pub fn from_config(config: &ModelConfig, max_tokens: usize, rms_eps: f32) -> Self {
        Self {
            hidden: config.hidden_size as usize,
            head_dim: config.gdn.head_dim as usize,
            value_heads: config.gdn.value_heads as usize,
            qk_heads: config.gdn.qk_heads as usize,
            conv_kernel: config.gdn.conv_kernel as usize,
            chunk_len: config.gdn.chunk_len as usize,
            max_tokens,
            rms_eps,
        }
    }

    /// As [`Self::from_config`], with `rms_eps` read from the model file.
    ///
    /// Falling back to a literal would let a file with a different epsilon
    /// produce a silently different normalization, so the key is required.
    pub fn from_gguf(config: &ModelConfig, file: &GgufFile, max_tokens: usize) -> Option<Self> {
        let eps = file.get_f32("qwen35moe.attention.layer_norm_rms_epsilon")?;
        Some(Self::from_config(config, max_tokens, eps))
    }

    /// Width of the fused q/k/v stream the convolution runs over (8192).
    pub const fn conv_dim(&self) -> usize {
        2 * self.qk_heads * self.head_dim + self.value_heads * self.head_dim
    }

    /// Width of the value stream, and of the output gate `z` (4096).
    pub const fn value_dim(&self) -> usize {
        self.value_heads * self.head_dim
    }

    /// Width of one token's query or key stream (2048).
    ///
    /// Half `value_dim`, because there are half as many query/key heads. This
    /// is the width [`GdnBlock::split_qkv`] emits and both mixer kernels
    /// expect; handing them `value_dim` instead is the workaround this block
    /// carried until the kernels learned the modulo mapping themselves.
    pub const fn key_dim(&self) -> usize {
        self.qk_heads * self.head_dim
    }

    /// Floats in one sequence's convolution cache.
    pub const fn conv_state_len(&self) -> usize {
        self.conv_dim() * (self.conv_kernel - 1)
    }

    /// Floats in one sequence's recurrent state.
    pub const fn recurrent_state_len(&self) -> usize {
        self.value_heads * self.head_dim * self.head_dim
    }
}

/// One Gated DeltaNet layer's weights, resident on the device.
///
/// Held as owned allocations rather than as views into a
/// [`crate::DeviceWeights`] arena because [`LayerOpsKernels`] takes
/// `&CudaSlice<f32>` for its norm and convolution weights, which a byte-arena
/// view cannot satisfy. The three large matrices stay quantized exactly as the
/// file stores them; only the norm vectors, the convolution filters and the two
/// small gate projections are unpacked, and those are 660 KiB per layer.
pub struct GdnLayerWeights {
    /// `attn_norm.weight`, `[hidden]`.
    pub input_norm: CudaSlice<f32>,
    /// `attn_qkv.weight`, Q8_0 `[hidden, conv_dim]`.
    pub qkv: CudaSlice<u8>,
    /// `attn_gate.weight`, Q8_0 `[hidden, value_dim]` — the output gate `z`.
    pub gate: CudaSlice<u8>,
    /// `ssm_conv1d.weight`, `[conv_kernel, conv_dim]`, i.e. `conv_dim` rows of
    /// `conv_kernel` taps, which is what [`LayerOpsKernels::conv1d`] wants.
    pub conv1d: CudaSlice<f32>,
    /// `ssm_alpha.weight`, `[hidden, value_heads]`.
    pub alpha: CudaSlice<f32>,
    /// `ssm_beta.weight`, `[hidden, value_heads]`.
    pub beta: CudaSlice<f32>,
    /// `ssm_dt.bias`, `[value_heads]`.
    pub dt_bias: CudaSlice<f32>,
    /// `ssm_a`, `[value_heads]`, **already negated in the file**.
    pub a: CudaSlice<f32>,
    /// `ssm_norm.weight`, `[head_dim]`.
    pub ssm_norm: CudaSlice<f32>,
    /// `ssm_out.weight`, Q8_0 `[value_dim, hidden]`.
    pub out: CudaSlice<u8>,
}

/// One layer's Q8_0 projections, repacked for the integer tensor cores.
///
/// The same numbers [`GdnLayerWeights`] holds, with the scales lifted out of
/// the 34-byte blocks so every operand load in `mma_q8_0_proj_split` is
/// aligned. Reading the on-disk layout directly measured 1% of the tensor
/// cores' peak — slower than the fp32 kernel — so this repack is what makes
/// the instruction usable at all. See `docs/BENCHMARKS.md`.
///
/// Costs about 37.7 MiB per layer, 1.13 GiB over the 30 Gated DeltaNet
/// layers, held *in addition* to the arena's copy. That is the price of the
/// alignment, and it is why this is built once at construction rather than
/// per pass.
pub struct GdnLayerInt8 {
    qkv_q: CudaSlice<i8>,
    qkv_s: CudaSlice<f32>,
    gate_q: CudaSlice<i8>,
    gate_s: CudaSlice<f32>,
    out_q: CudaSlice<i8>,
    out_s: CudaSlice<f32>,
}

impl GdnLayerInt8 {
    /// Device bytes held.
    pub fn bytes(&self) -> u64 {
        let q = self.qkv_q.len() + self.gate_q.len() + self.out_q.len();
        let s = self.qkv_s.len() + self.gate_s.len() + self.out_s.len();
        (q + s * size_of::<f32>()) as u64
    }
}

impl GdnLayerWeights {
    /// Upload block `layer`'s Gated DeltaNet tensors from `file`.
    ///
    /// `directory` must have been resolved against `file`. Every tensor is
    /// looked up by [`Role`] rather than by name, so this file never types a
    /// GGUF tensor name.
    pub fn upload(
        stream: &Arc<CudaStream>,
        file: &GgufFile,
        directory: &Directory<'_>,
        layer: u32,
    ) -> Result<Self, GdnBlockError> {
        let raw = |role: Role| -> Result<(&[u8], GgmlType), GdnBlockError> {
            let entry = directory
                .find(role, Some(layer))
                .ok_or(GdnBlockError::MissingWeight { role, layer })?;
            let bytes = file
                .tensor_bytes(&entry.spec.name)
                .ok_or(GdnBlockError::MissingWeight { role, layer })?;
            Ok((bytes, entry.info.ggml_type))
        };

        let quantized = |role: Role| -> Result<CudaSlice<u8>, GdnBlockError> {
            let (bytes, ty) = raw(role)?;
            if ty != GgmlType::Q8_0 {
                return Err(GdnBlockError::UnsupportedQuant {
                    role,
                    found: ty,
                    expected: GgmlType::Q8_0,
                });
            }
            Ok(stream.clone_htod(bytes)?)
        };

        let floats = |role: Role| -> Result<CudaSlice<f32>, GdnBlockError> {
            let (bytes, ty) = raw(role)?;
            if ty != GgmlType::F32 {
                return Err(GdnBlockError::UnsupportedQuant {
                    role,
                    found: ty,
                    expected: GgmlType::F32,
                });
            }
            let values: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(f32::from_le_bytes)
                .collect();
            Ok(stream.clone_htod(&values)?)
        };

        Ok(Self {
            input_norm: floats(Role::InputNorm)?,
            qkv: quantized(Role::GdnQkv)?,
            gate: quantized(Role::GdnGate)?,
            conv1d: floats(Role::GdnConv1d)?,
            alpha: floats(Role::GdnAlpha)?,
            beta: floats(Role::GdnBeta)?,
            dt_bias: floats(Role::GdnDtBias)?,
            a: floats(Role::GdnA)?,
            ssm_norm: floats(Role::GdnNorm)?,
            out: quantized(Role::GdnOut)?,
        })
    }
}

/// One sequence's carried Gated DeltaNet state.
///
/// Both halves start zeroed, which is what
/// `build_layer_attn_linear`'s `state_predelta` shows for a fresh sequence.
/// Neither is ever reallocated: they are the block's only genuinely persistent
/// memory, and their size does not depend on the sequence length.
pub struct GdnState {
    /// `[conv_dim][conv_kernel - 1]`, oldest input first.
    pub conv: CudaSlice<f32>,
    /// `[value_heads][head_dim][head_dim]`, `S[v][k]` with the key contiguous.
    pub recurrent: CudaSlice<f32>,
}

/// Which form of the delta rule a call used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mixer {
    /// One token, one state update — the decode path.
    Recurrent,
    /// A whole sequence, `chunk_len` tokens per triangular solve.
    Chunked,
}

/// Borrowed views of everything the last [`GdnBlock::forward`] computed.
///
/// This is what makes a wrong block bisectable rather than merely wrong: each
/// field is one of the tensors `docs/ORACLE.md` captured, so a test can say
/// which *step* diverged instead of only that the block did.
pub struct GdnTrace<'a> {
    /// `attn_norm-N`, `[tokens][hidden]`.
    pub attn_norm: &'a CudaSlice<f32>,
    /// `linear_attn_qkv_mixed-N`, `[tokens][conv_dim]`.
    pub qkv_mixed: &'a CudaSlice<f32>,
    /// `conv_output_raw-N`, `[tokens][conv_dim]`.
    pub conv_raw: &'a CudaSlice<f32>,
    /// `conv_output_silu-N`, `[tokens][conv_dim]`.
    pub conv_silu: &'a CudaSlice<f32>,
    /// `q_conv-N`, `[tokens][qk_heads][head_dim]` — the capture's own shape.
    /// **Not** L2-normalized: the mixer kernels do that internally.
    pub q: &'a CudaSlice<f32>,
    /// `k_conv-N`, `[tokens][qk_heads][head_dim]`.
    pub k: &'a CudaSlice<f32>,
    /// `v_conv_predelta-N`, `[tokens][value_heads][head_dim]`.
    pub v: &'a CudaSlice<f32>,
    /// `z-N`, `[tokens][value_dim]`.
    pub z: &'a CudaSlice<f32>,
    /// `alpha-N`, `[tokens][value_heads]`.
    pub alpha: &'a CudaSlice<f32>,
    /// `a_softplus-N`, `[tokens][value_heads]`.
    pub a_softplus: &'a CudaSlice<f32>,
    /// `gate-N`, `[tokens][value_heads]` — the log-decay.
    pub log_decay: &'a CudaSlice<f32>,
    /// `beta-N`, `[tokens][value_heads]`.
    pub beta_raw: &'a CudaSlice<f32>,
    /// `beta_sigmoid-N`, `[tokens][value_heads]`.
    pub beta: &'a CudaSlice<f32>,
    /// The delta rule's output, before the norm and gate.
    pub core: &'a CudaSlice<f32>,
    /// `final_output-N`, `[tokens][value_dim]`.
    pub final_output: &'a CudaSlice<f32>,
    /// `linear_attn_out-N`, `[tokens][hidden]`.
    pub linear_attn_out: &'a CudaSlice<f32>,
    /// Which form of the delta rule ran.
    pub mixer: Mixer,
}

/// A projection weight and how it is stored.
#[derive(Clone, Copy)]
pub enum Projection<'a> {
    /// Q8_0 blocks exactly as the GGUF file holds them.
    Q8_0(&'a CudaSlice<u8>),
    /// Plain fp32.
    F32(&'a CudaSlice<f32>),
}

/// Scratch for one token count.
///
/// Sized to the live tokens rather than to a maximum, because
/// [`LayerOpsKernels`] checks `rows * width == len` exactly. Rebuilt only when
/// the token count changes.
struct Scratch {
    tokens: usize,
    normed: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    conv_raw: CudaSlice<f32>,
    conv_silu: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    z: CudaSlice<f32>,
    alpha: CudaSlice<f32>,
    beta_raw: CudaSlice<f32>,
    a_softplus: CudaSlice<f32>,
    log_decay: CudaSlice<f32>,
    beta: CudaSlice<f32>,
    core: CudaSlice<f32>,
    core_norm: CudaSlice<f32>,
    final_output: CudaSlice<f32>,
    projected: CudaSlice<f32>,
}

impl Scratch {
    fn new(
        stream: &Arc<CudaStream>,
        g: &GdnGeometry,
        tokens: usize,
    ) -> Result<Self, GdnBlockError> {
        let hidden = tokens * g.hidden;
        let conv = tokens * g.conv_dim();
        let heads = tokens * g.value_heads;
        let value = tokens * g.value_dim();
        // q and k are half the width of v: 16 query/key heads against 32 value
        // heads. They were `value` too while this block broadcast them itself.
        let key = tokens * g.key_dim();
        Ok(Self {
            tokens,
            normed: stream.alloc_zeros::<f32>(hidden)?,
            qkv: stream.alloc_zeros::<f32>(conv)?,
            conv_raw: stream.alloc_zeros::<f32>(conv)?,
            conv_silu: stream.alloc_zeros::<f32>(conv)?,
            q: stream.alloc_zeros::<f32>(key)?,
            k: stream.alloc_zeros::<f32>(key)?,
            v: stream.alloc_zeros::<f32>(value)?,
            z: stream.alloc_zeros::<f32>(value)?,
            alpha: stream.alloc_zeros::<f32>(heads)?,
            beta_raw: stream.alloc_zeros::<f32>(heads)?,
            a_softplus: stream.alloc_zeros::<f32>(heads)?,
            log_decay: stream.alloc_zeros::<f32>(heads)?,
            beta: stream.alloc_zeros::<f32>(heads)?,
            core: stream.alloc_zeros::<f32>(value)?,
            core_norm: stream.alloc_zeros::<f32>(value)?,
            final_output: stream.alloc_zeros::<f32>(value)?,
            projected: stream.alloc_zeros::<f32>(hidden)?,
        })
    }
}

/// The Gated DeltaNet block.
///
/// One instance serves every Gated DeltaNet layer: the kernels are compiled
/// once and the per-layer weights travel as an argument, because the geometry
/// is identical across all 30 of them and 30 NVRTC compiles at startup would
/// be 30 driver round-trips for the same PTX.
pub struct GdnBlock {
    /// Integer tensor cores for the Q8_0 projections at prefill shapes.
    mma: MmaKernels,
    /// Quantized activations and their scales, reused every projection.
    xq: Option<(CudaSlice<i8>, CudaSlice<f32>)>,
    layer_ops: LayerOpsKernels,
    recurrent: GdnKernels,
    chunked: GdnChunkedKernels,
    recurrent_scratch: GdnScratch,
    chunked_scratch: GdnChunkedScratch,
    proj_q8_0: CudaFunction,
    proj_split_gemv: CudaFunction,
    proj_split_gemv_add: CudaFunction,
    proj_tiled: [CudaFunction; 4],
    proj_f32: CudaFunction,
    alpha_beta_gates: CudaFunction,
    alpha_beta_gates_t1: CudaFunction,
    silu: CudaFunction,
    split: CudaFunction,
    silu_split: CudaFunction,
    gates: CudaFunction,
    add: CudaFunction,
    scratch: Option<Scratch>,
    mixer: Mixer,
    geometry: GdnGeometry,
}

impl GdnBlock {
    /// Compile every kernel this block needs, for `geometry`.
    ///
    /// Both mixer kernels get the model's real `qk_heads` (16), so they apply
    /// `qk_head = h % qk_heads` themselves over a narrow q and k. They were
    /// constructed with `qk_heads == value_heads` while their internal map was
    /// `h / heads_per_kv` and this block had to pre-broadcast; that made the
    /// chunked form compute its two `chunk_len x chunk_len` Gram matrices 32
    /// times per chunk rather than 16. See the module docs.
    pub fn new(ctx: &Arc<CudaContext>, geometry: GdnGeometry) -> Result<Self, GdnBlockError> {
        let stream = ctx.default_stream();

        let layer_ops = LayerOpsKernels::new(ctx)?;
        let recurrent = GdnKernels::new(
            ctx,
            geometry.head_dim,
            geometry.value_heads,
            geometry.qk_heads,
        )?;
        let chunked = GdnChunkedKernels::new(
            ctx,
            geometry.head_dim,
            geometry.value_heads,
            geometry.qk_heads,
            geometry.chunk_len,
        )?;
        let recurrent_scratch = recurrent.scratch(&stream)?;
        let chunked_scratch = chunked.scratch(&stream, geometry.max_tokens)?;

        let ptx = compile(GDN_BLOCK_SRC, "gdn_block").map_err(GdnBlockError::Compile)?;
        let module = ctx.load_module(ptx)?;

        Ok(Self {
            layer_ops,
            recurrent,
            chunked,
            recurrent_scratch,
            chunked_scratch,
            proj_q8_0: module.load_function("gdn_proj_q8_0")?,
            proj_split_gemv: module.load_function("gdn_proj_split_gemv")?,
            proj_split_gemv_add: module.load_function("gdn_proj_split_gemv_add")?,
            proj_tiled: [
                module.load_function("gdn_proj_q8_0_t2")?,
                module.load_function("gdn_proj_q8_0_t4")?,
                module.load_function("gdn_proj_q8_0_t8")?,
                module.load_function("gdn_proj_q8_0_t16")?,
            ],
            proj_f32: module.load_function("gdn_proj_f32")?,
            alpha_beta_gates: module.load_function("gdn_alpha_beta_gates")?,
            alpha_beta_gates_t1: module.load_function("gdn_alpha_beta_gates_t1")?,
            silu: module.load_function("gdn_silu")?,
            split: module.load_function("gdn_split_qkv")?,
            silu_split: module.load_function("gdn_silu_split_qkv")?,
            gates: module.load_function("gdn_gates")?,
            add: module.load_function("gdn_add")?,
            scratch: None,
            mixer: Mixer::Chunked,
            geometry,
            mma: MmaKernels::new(ctx).map_err(GdnBlockError::Mma)?,
            xq: None,
        })
    }

    /// Repack one layer's Q8_0 projections for the integer tensor cores.
    ///
    /// Called once per layer at construction. See [`GdnLayerInt8`] for what it
    /// costs and why it is worth it.
    pub fn repack(
        &self,
        stream: &Arc<CudaStream>,
        w: &GdnLayerWeights,
    ) -> Result<GdnLayerInt8, GdnBlockError> {
        let g = self.geometry;
        let one = |src: &CudaSlice<u8>, elements: usize| -> Result<_, GdnBlockError> {
            let mut q = stream.alloc_zeros::<i8>(elements)?;
            let mut sc = stream.alloc_zeros::<f32>(elements / 32)?;
            self.mma
                .repack_q8_0(stream, src, &mut q, &mut sc, elements)
                .map_err(GdnBlockError::Mma)?;
            Ok((q, sc))
        };
        let (qkv_q, qkv_s) = one(&w.qkv, g.hidden * g.conv_dim())?;
        let (gate_q, gate_s) = one(&w.gate, g.hidden * g.value_dim())?;
        let (out_q, out_s) = one(&w.out, g.value_dim() * g.hidden)?;
        Ok(GdnLayerInt8 {
            qkv_q,
            qkv_s,
            gate_q,
            gate_s,
            out_q,
            out_s,
        })
    }

    /// Whether a batch of `tokens` should take the integer tensor-core path.
    ///
    /// Below a full token tile the MMA kernel runs its guarded path and the
    /// tensor cores are starved of arithmetic intensity, so the fp32 kernel —
    /// which has its own tile specialization for small batches — wins. Decode
    /// is one token and never comes near this.
    pub fn uses_tensor_cores(tokens: usize) -> bool {
        tokens >= MMA_SPLIT_TOKENS
    }

    /// The geometry this block was compiled for.
    pub fn geometry(&self) -> GdnGeometry {
        self.geometry
    }

    /// The norm, convolution and activation kernels, for callers that want to
    /// drive one step of the block in isolation.
    pub fn layer_ops(&self) -> &LayerOpsKernels {
        &self.layer_ops
    }

    /// A zeroed convolution cache and recurrent state for one sequence.
    pub fn state(&self, stream: &Arc<CudaStream>) -> Result<GdnState, GdnBlockError> {
        Ok(GdnState {
            conv: stream.alloc_zeros::<f32>(self.geometry.conv_state_len())?,
            recurrent: stream.alloc_zeros::<f32>(self.geometry.recurrent_state_len())?,
        })
    }

    /// Every intermediate the last [`Self::forward`] produced.
    ///
    /// `None` before the first call, because the scratch is sized on demand.
    pub fn trace(&self) -> Option<GdnTrace<'_>> {
        let s = self.scratch.as_ref()?;
        Some(GdnTrace {
            attn_norm: &s.normed,
            qkv_mixed: &s.qkv,
            conv_raw: &s.conv_raw,
            conv_silu: &s.conv_silu,
            q: &s.q,
            k: &s.k,
            v: &s.v,
            z: &s.z,
            alpha: &s.alpha,
            a_softplus: &s.a_softplus,
            log_decay: &s.log_decay,
            beta_raw: &s.beta_raw,
            beta: &s.beta,
            core: &s.core,
            final_output: &s.final_output,
            linear_attn_out: &s.projected,
            mixer: self.mixer,
        })
    }

    /// Run the whole block: `out = mixer(rmsnorm(hidden)) + hidden`.
    ///
    /// `hidden` and `out` are `[tokens][hidden]` and their length fixes the
    /// token count. `state` is advanced in place — both the convolution cache
    /// and the recurrent state — so calling this once with 19 tokens and
    /// calling it nineteen times with one token each must produce the same
    /// outputs, which is what `gdn_block.rs` checks.
    ///
    /// A single token takes the recurrent form and anything longer takes the
    /// chunked form, which is the same dispatch `build_delta_net` makes.
    pub fn forward(
        &mut self,
        stream: &Arc<CudaStream>,
        weights: &GdnLayerWeights,
        int8: Option<&GdnLayerInt8>,
        state: &mut GdnState,
        hidden: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<Mixer, GdnBlockError> {
        let g = self.geometry;
        if !hidden.len().is_multiple_of(g.hidden) || hidden.is_empty() {
            return Err(GdnBlockError::ShapeMismatch {
                what: "forward hidden",
                expected: g.hidden,
                got: hidden.len(),
            });
        }
        let tokens = hidden.len() / g.hidden;
        check_len("forward out", tokens * g.hidden, out.len())?;
        if tokens > g.max_tokens {
            return Err(GdnBlockError::Chunked(GdnChunkedError::SequenceTooLong {
                seq_len: tokens,
                capacity: g.max_tokens,
            }));
        }

        if self.scratch.as_ref().map(|s| s.tokens) != Some(tokens) {
            self.scratch = Some(Scratch::new(stream, &g, tokens)?);
        }
        // Split the borrow: the scratch is `&mut` for the whole body while the
        // kernels are `&`, and the launch helpers below take `&self`.
        let mut scratch = self.scratch.take().expect("scratch was just installed");

        // All three Q8_0 projections read the same normed activations, so the
        // int8 conversion is done once here rather than three times inside
        // `project`. `None` keeps the fp32 path, which is what decode and
        // small batches take.

        let result = self.run(
            stream,
            weights,
            int8,
            state,
            hidden,
            out,
            &mut scratch,
            tokens,
        );
        self.scratch = Some(scratch);
        result?;
        Ok(self.mixer)
    }

    /// Batched decode: `states.len()` independent one-token sequences,
    /// advanced by amortizing every weight-bound projection across all of
    /// them and looping only the two steps that are intrinsically
    /// per-sequence.
    ///
    /// # Why this is a different method rather than `forward` at `tokens > 1`
    ///
    /// [`Self::forward`] treats every token beyond the first as a
    /// *continuation of the same sequence*: the causal convolution slides one
    /// window across all of them and the delta rule folds them in temporal
    /// order into one recurrent matrix (the chunked form, dispatched by
    /// [`Self::mix`]). That is correct for a prefill and wrong here — a
    /// four-token batch of four different sequences' next token is not four
    /// consecutive positions of one sequence, and folding them through one
    /// state would answer sequence 1's query with sequence 0's history.
    ///
    /// What *is* shared across sequences is every weight: the norm, the qkv
    /// and gate projections, the alpha/beta gate projections and the output
    /// projection contract the same matrices against every token regardless
    /// of which sequence it belongs to, and have no notion of state at all.
    /// Those run once over the whole `[states.len()][hidden]` batch, taking
    /// the same tiled kernel a `states.len()`-token prefill chunk would (see
    /// `GDN_PROJ_TILED` in the module source) — the weight is read once
    /// instead of once per sequence, which is `docs/OPTIMIZATION.md` R2's
    /// entire claimed win.
    ///
    /// The causal convolution and the delta-rule update are the two steps
    /// that read and write a *sequence's own* state, so they cannot be
    /// batched into one kernel call without a device-side index into
    /// `states` that no kernel in this workspace has (see
    /// `crate::viewslice`). They loop instead, one call per sequence, each
    /// over a **one-token** slice of the batch buffer — [`GdnKernels::step`]
    /// is a fixed-size `O(value_heads * head_dim^2)` update with no weight
    /// traffic at all, so the loop costs `states.len()` small launches, not
    /// `states.len()` weight re-reads. Every call in the loop is captured
    /// into the same CUDA graph as the batched calls around it when this
    /// runs inside [`crate::forward::Forward::capture_batch_step`], so the
    /// loop's *host* overhead — issuing `states.len()` launches instead of
    /// one — is paid once, at capture time, not on every replay.
    ///
    /// `hidden` and `out` are `[states.len()][hidden]`; token `i` belongs to
    /// `states[i]` and both must be the same length. Every state must already
    /// carry that sequence's Gated DeltaNet history, exactly as
    /// [`Self::forward`] expects — a fresh or [`GdnState`]-reset state starts
    /// that sequence cold.
    pub fn forward_batch_decode(
        &mut self,
        stream: &Arc<CudaStream>,
        weights: &GdnLayerWeights,
        int8: Option<&GdnLayerInt8>,
        states: &mut [&mut GdnState],
        hidden: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), GdnBlockError> {
        let g = self.geometry;
        let tokens = states.len();
        check_len("batch decode hidden", tokens * g.hidden, hidden.len())?;
        check_len("batch decode out", tokens * g.hidden, out.len())?;
        if tokens == 0 {
            return Ok(());
        }
        if tokens > g.max_tokens {
            return Err(GdnBlockError::Chunked(GdnChunkedError::SequenceTooLong {
                seq_len: tokens,
                capacity: g.max_tokens,
            }));
        }

        if self.scratch.as_ref().map(|s| s.tokens) != Some(tokens) {
            self.scratch = Some(Scratch::new(stream, &g, tokens)?);
        }
        let mut scratch = self.scratch.take().expect("scratch was just installed");
        let result =
            self.run_batch_decode(stream, weights, int8, states, hidden, out, &mut scratch);
        self.scratch = Some(scratch);
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn run_batch_decode(
        &mut self,
        stream: &Arc<CudaStream>,
        w: &GdnLayerWeights,
        tc: Option<&GdnLayerInt8>,
        states: &mut [&mut GdnState],
        hidden: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        s: &mut Scratch,
    ) -> Result<(), GdnBlockError> {
        let g = self.geometry;
        let tokens = states.len();
        // `gdn_proj_split_gemv` writes `out[n]` with no token axis at all --
        // wrong for `tokens > 1`, where a real per-token axis exists even
        // though each token is a different sequence's single step. At
        // exactly one sequence, though, "no token axis" and "the batch's
        // one token" are the same buffer layout, so `run`'s own `gemv` path
        // applies unchanged. Measured with `nsys --cuda-graph-trace=node`:
        // without this, batch(1)'s qkv/gate/out projections took the generic
        // tiled path's `tokens == 1` fallback (`gdn_proj_q8_0`, the untiled
        // kernel with no split-layout repack) at 38,608.3 ns/call average
        // against `gdn_proj_split_gemv`/`_add`'s 24,967 ns/call blended
        // average in `run` -- 94.7% of the entire measured N=1 batch-vs-
        // single-stream gap, 1.23 ms of 1.30 ms/step. See
        // `docs/BENCHMARKS.md`'s batched-decode section for the nsys-diff
        // that found it kernel by kernel rather than assuming it.
        let gemv = tc.filter(|_| tokens == 1);
        let tc = tc.filter(|_| Self::uses_tensor_cores(tokens));

        // 1. attn_norm-N, over every sequence's row.
        self.layer_ops.rms_norm(
            stream,
            hidden,
            &w.input_norm,
            &mut s.normed,
            tokens,
            g.hidden,
            g.rms_eps,
        )?;

        // 2. linear_attn_qkv_mixed-N and z-N. No state, so this is exactly
        //    the batched form `run` already has for a multi-token prefill —
        //    the weight is read once for the whole batch regardless of how
        //    many distinct sequences the tokens belong to.
        if let Some(i8w) = tc {
            self.quantize_activations(stream, &s.normed, tokens, g.hidden)?;
            let (xq, xs) = self.xq.as_ref().expect("quantized above");
            self.mma
                .q8_0_proj_split(
                    stream,
                    &i8w.qkv_q,
                    &i8w.qkv_s,
                    xq,
                    xs,
                    &mut s.qkv,
                    g.hidden,
                    g.conv_dim(),
                    tokens,
                )
                .map_err(GdnBlockError::Mma)?;
            self.mma
                .q8_0_proj_split(
                    stream,
                    &i8w.gate_q,
                    &i8w.gate_s,
                    xq,
                    xs,
                    &mut s.z,
                    g.hidden,
                    g.value_dim(),
                    tokens,
                )
                .map_err(GdnBlockError::Mma)?;
        } else if let Some(i8w) = gemv {
            self.project_split_gemv(
                stream,
                &i8w.qkv_q,
                &i8w.qkv_s,
                &s.normed,
                &mut s.qkv,
                g.hidden,
                g.conv_dim(),
            )?;
            self.project_split_gemv(
                stream,
                &i8w.gate_q,
                &i8w.gate_s,
                &s.normed,
                &mut s.z,
                g.hidden,
                g.value_dim(),
            )?;
        } else {
            self.project(
                stream,
                Projection::Q8_0(&w.qkv),
                &s.normed,
                &mut s.qkv,
                g.hidden,
                g.conv_dim(),
                tokens,
            )?;
            self.project(
                stream,
                Projection::Q8_0(&w.gate),
                &s.normed,
                &mut s.z,
                g.hidden,
                g.value_dim(),
                tokens,
            )?;
        }

        // 3. conv_output_raw-N, one sequence at a time: the short causal
        //    convolution reads and advances *that sequence's own* window, so
        //    it cannot run as one call over tokens that do not share a
        //    timeline. Each call is one token wide and touches only
        //    `states[i].conv` and a one-token slice of the batch buffers.
        let conv_dim = g.conv_dim();
        for (i, state) in states.iter_mut().enumerate() {
            // SAFETY: `i < tokens` and `conv_dim` is `s.qkv`'s and
            // `s.conv_raw`'s per-token width, checked by `Scratch::new`
            // against the same `tokens * conv_dim` this loop covers.
            let x = unsafe { crate::viewslice::subslice(stream, &s.qkv, i * conv_dim, conv_dim) };
            let mut y =
                unsafe { crate::viewslice::subslice(stream, &s.conv_raw, i * conv_dim, conv_dim) };
            self.layer_ops.conv1d(
                stream,
                &x,
                &w.conv1d,
                &mut state.conv,
                &mut y,
                1,
                conv_dim,
                g.conv_kernel,
            )?;
        }

        // 4. conv_output_silu-N and the q/k/v split. No state: every element
        //    of `s.conv_raw` — now correctly one sequence's own window per
        //    token, from step 3 — is read exactly once, so this batches over
        //    every sequence in one launch.
        self.silu_split_qkv(
            stream,
            &s.conv_raw,
            &mut s.conv_silu,
            &mut s.q,
            &mut s.k,
            &mut s.v,
            tokens,
        )?;

        // 5. alpha-N / a_softplus-N / gate-N and beta-N / beta_sigmoid-N. No
        //    state, batches the same way step 2 does.
        self.alpha_beta_gates(
            stream,
            &w.alpha,
            &w.beta,
            &s.normed,
            &w.dt_bias,
            &w.a,
            &mut s.alpha,
            &mut s.beta_raw,
            &mut s.a_softplus,
            &mut s.log_decay,
            &mut s.beta,
            tokens,
        )?;

        // 6. The delta rule, one sequence at a time — the other half of what
        //    step 3 could not batch, and for the same reason: it reads and
        //    writes `states[i].recurrent`, not a shared weight.
        //    `GdnKernels::step` is the one-token recurrent kernel `run` also
        //    uses at `tokens == 1`; here it runs once per sequence instead of
        //    once for the whole call, against that sequence's own state and a
        //    one-token slice of the batch's q/k/v/gates.
        let key_dim = g.key_dim();
        let value_dim = g.value_dim();
        let value_heads = g.value_heads;
        for (i, state) in states.iter_mut().enumerate() {
            // SAFETY: as the convolution loop above, with each buffer's own
            // per-token width, all checked by `Scratch::new` against
            // `tokens * width`.
            let q = unsafe { crate::viewslice::subslice(stream, &s.q, i * key_dim, key_dim) };
            let k = unsafe { crate::viewslice::subslice(stream, &s.k, i * key_dim, key_dim) };
            let v = unsafe { crate::viewslice::subslice(stream, &s.v, i * value_dim, value_dim) };
            let log_decay = unsafe {
                crate::viewslice::subslice(stream, &s.log_decay, i * value_heads, value_heads)
            };
            let beta = unsafe {
                crate::viewslice::subslice(stream, &s.beta, i * value_heads, value_heads)
            };
            let mut core_out =
                unsafe { crate::viewslice::subslice(stream, &s.core, i * value_dim, value_dim) };
            self.recurrent.step(
                stream,
                &mut self.recurrent_scratch,
                &mut state.recurrent,
                &q,
                &k,
                &v,
                &log_decay,
                &beta,
                &mut core_out,
            )?;
        }

        // 7. final_output-N = ssm_norm(core) * silu(z). No state: batches
        //    over every sequence's now-correct `core` row from step 6.
        self.layer_ops.rms_norm_swiglu(
            stream,
            &s.core,
            &w.ssm_norm,
            &s.z,
            &mut s.core_norm,
            &mut s.final_output,
            tokens * g.value_heads,
            g.head_dim,
            g.rms_eps,
        )?;

        // 8/9. linear_attn_out-N, then attn_residual-N. No state, batches the
        //      same way step 2 does.
        if let Some(i8w) = tc {
            self.quantize_activations(stream, &s.final_output, tokens, g.value_dim())?;
            let (xq, xs) = self.xq.as_ref().expect("quantized above");
            self.mma
                .q8_0_proj_split(
                    stream,
                    &i8w.out_q,
                    &i8w.out_s,
                    xq,
                    xs,
                    &mut s.projected,
                    g.value_dim(),
                    g.hidden,
                    tokens,
                )
                .map_err(GdnBlockError::Mma)?;
            self.add(stream, &s.projected, hidden, out, tokens * g.hidden)?;
        } else if let Some(i8w) = gemv {
            // The residual add rides out of the projection's own warp, as in
            // `run` -- at one token/one sequence, `hidden` and `out` are
            // exactly the buffers `project_split_gemv_add`'s `residual` and
            // `summed` expect: `n_rows` long, no token axis, because there
            // is only one token in the whole call.
            self.project_split_gemv_add(
                stream,
                &i8w.out_q,
                &i8w.out_s,
                &s.final_output,
                hidden,
                &mut s.projected,
                out,
                g.value_dim(),
                g.hidden,
            )?;
        } else {
            self.project(
                stream,
                Projection::Q8_0(&w.out),
                &s.final_output,
                &mut s.projected,
                g.value_dim(),
                g.hidden,
                tokens,
            )?;
            self.add(stream, &s.projected, hidden, out, tokens * g.hidden)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        stream: &Arc<CudaStream>,
        w: &GdnLayerWeights,
        tc: Option<&GdnLayerInt8>,
        state: &mut GdnState,
        hidden: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        s: &mut Scratch,
        tokens: usize,
    ) -> Result<(), GdnBlockError> {
        let g = self.geometry;
        // One token over the repacked layout is a GEMV, and a *different*
        // reason to want the repack than the tensor cores are: the split
        // layout's contiguous quants are worth having even when the
        // arithmetic stays fp32. See `gdn_proj_split_gemv`.
        let gemv = tc.filter(|_| tokens == 1);
        let tc = tc.filter(|_| Self::uses_tensor_cores(tokens));

        // 1. attn_norm-N
        self.layer_ops.rms_norm(
            stream,
            hidden,
            &w.input_norm,
            &mut s.normed,
            tokens,
            g.hidden,
            g.rms_eps,
        )?;

        // 2. linear_attn_qkv_mixed-N, and the output gate z-N alongside it.
        if let Some(i8w) = tc {
            self.quantize_activations(stream, &s.normed, tokens, g.hidden)?;
            let (xq, xs) = self.xq.as_ref().expect("quantized above");
            self.mma
                .q8_0_proj_split(
                    stream,
                    &i8w.qkv_q,
                    &i8w.qkv_s,
                    xq,
                    xs,
                    &mut s.qkv,
                    g.hidden,
                    g.conv_dim(),
                    tokens,
                )
                .map_err(GdnBlockError::Mma)?;
            self.mma
                .q8_0_proj_split(
                    stream,
                    &i8w.gate_q,
                    &i8w.gate_s,
                    xq,
                    xs,
                    &mut s.z,
                    g.hidden,
                    g.value_dim(),
                    tokens,
                )
                .map_err(GdnBlockError::Mma)?;
        } else if let Some(i8w) = gemv {
            self.project_split_gemv(
                stream,
                &i8w.qkv_q,
                &i8w.qkv_s,
                &s.normed,
                &mut s.qkv,
                g.hidden,
                g.conv_dim(),
            )?;
            self.project_split_gemv(
                stream,
                &i8w.gate_q,
                &i8w.gate_s,
                &s.normed,
                &mut s.z,
                g.hidden,
                g.value_dim(),
            )?;
        } else {
            self.project(
                stream,
                Projection::Q8_0(&w.qkv),
                &s.normed,
                &mut s.qkv,
                g.hidden,
                g.conv_dim(),
                tokens,
            )?;
            self.project(
                stream,
                Projection::Q8_0(&w.gate),
                &s.normed,
                &mut s.z,
                g.hidden,
                g.value_dim(),
                tokens,
            )?;
        }

        // 3/4. conv_output_raw-N, then conv_output_silu-N. The convolution
        // advances the cache, which is why it must run once per batch and not
        // once per comparison.
        self.layer_ops.conv1d(
            stream,
            &s.qkv,
            &w.conv1d,
            &mut state.conv,
            &mut s.conv_raw,
            tokens,
            g.conv_dim(),
            g.conv_kernel,
        )?;
        // 5. conv_output_silu-N and q/k/v in one launch: the split reads every
        //    element of the SiLU's output exactly once, so it can apply the
        //    nonlinearity on the way through.
        self.silu_split_qkv(
            stream,
            &s.conv_raw,
            &mut s.conv_silu,
            &mut s.q,
            &mut s.k,
            &mut s.v,
            tokens,
        )?;

        // 6. alpha-N / a_softplus-N / gate-N and beta-N / beta_sigmoid-N.
        //    One launch: both projections read the same `normed` and the
        //    gates read nothing but their outputs.
        self.alpha_beta_gates(
            stream,
            &w.alpha,
            &w.beta,
            &s.normed,
            &w.dt_bias,
            &w.a,
            &mut s.alpha,
            &mut s.beta_raw,
            &mut s.a_softplus,
            &mut s.log_decay,
            &mut s.beta,
            tokens,
        )?;

        // 7. The delta rule.
        self.mixer = self.mix(
            stream,
            state,
            &s.q,
            &s.k,
            &s.v,
            &s.log_decay,
            &s.beta,
            &mut s.core,
            tokens,
        )?;

        // 8. final_output-N = ssm_norm(core) * silu(z). One launch: the norm's
        //    grid is one block per head-row and the multiply is elementwise
        //    over the same rows, so it rides along in the norm's second pass.
        self.layer_ops.rms_norm_swiglu(
            stream,
            &s.core,
            &w.ssm_norm,
            &s.z,
            &mut s.core_norm,
            &mut s.final_output,
            tokens * g.value_heads,
            g.head_dim,
            g.rms_eps,
        )?;

        // 9/10. linear_attn_out-N, then attn_residual-N.
        //
        // This contracts over `value_dim`, not `hidden`, and reads the gated
        // core output rather than the normed input — so it re-quantizes rather
        // than reusing what step 2 produced.
        if let Some(i8w) = tc {
            self.quantize_activations(stream, &s.final_output, tokens, g.value_dim())?;
            let (xq, xs) = self.xq.as_ref().expect("quantized above");
            self.mma
                .q8_0_proj_split(
                    stream,
                    &i8w.out_q,
                    &i8w.out_s,
                    xq,
                    xs,
                    &mut s.projected,
                    g.value_dim(),
                    g.hidden,
                    tokens,
                )
                .map_err(GdnBlockError::Mma)?;
            self.add(stream, &s.projected, hidden, out, tokens * g.hidden)?;
        } else if let Some(i8w) = gemv {
            // The residual add rides out of the projection's own warp: at one
            // token it was a 1.8 us launch over 2,048 floats.
            self.project_split_gemv_add(
                stream,
                &i8w.out_q,
                &i8w.out_s,
                &s.final_output,
                hidden,
                &mut s.projected,
                out,
                g.value_dim(),
                g.hidden,
            )?;
        } else {
            self.project(
                stream,
                Projection::Q8_0(&w.out),
                &s.final_output,
                &mut s.projected,
                g.value_dim(),
                g.hidden,
                tokens,
            )?;
            self.add(stream, &s.projected, hidden, out, tokens * g.hidden)?;
        }
        Ok(())
    }

    /// Quantize `x` to int8 for the projections that take the tensor-core path.
    ///
    /// Called **twice** per layer, not once. `qkv` and `gate` both read the
    /// normed activations and contract over `hidden`, so they share a single
    /// quantization; `out` reads the gated core output and contracts over
    /// `value_dim`, so it needs its own. Only the first sharing is a saving —
    /// the second call genuinely re-quantizes different data.
    ///
    /// The buffer is allocated once, at the *widest* contraction the block
    /// runs, and the narrower call reuses the front of it. Sizing it to
    /// whichever call happens to come first would make the second call
    /// reallocate mid-pass, which `AGENTS.md` rule 6 forbids.
    fn quantize_activations(
        &mut self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        tokens: usize,
        k_dim: usize,
    ) -> Result<(), GdnBlockError> {
        let need = tokens * k_dim;
        let widest = tokens * self.geometry.hidden.max(self.geometry.value_dim());
        debug_assert!(
            need <= widest,
            "a projection contracts over more than the block's widest dimension"
        );
        let have = self.xq.as_ref().is_some_and(|(q, _)| q.len() >= need);
        if !have {
            self.xq = Some((
                stream.alloc_zeros::<i8>(widest)?,
                stream.alloc_zeros::<f32>(widest / QK8_0)?,
            ));
        }
        let (q, sc) = self.xq.as_mut().expect("just allocated");
        // The buffers are sized for the widest projection this block runs and
        // reused by the narrower ones, so they are routinely longer than this
        // call needs; `quantize_rows` checks a lower bound for exactly that.
        self.mma
            .quantize_rows(stream, x, q, sc, tokens, k_dim)
            .map_err(GdnBlockError::Mma)?;
        Ok(())
    }

    /// The one-token output projection with the block's residual folded in.
    ///
    /// See `gdn_proj_split_gemv_add`: `out` is `linear_attn_out-N` and
    /// `summed` is `attn_residual-N`.
    #[allow(clippy::too_many_arguments)]
    pub fn project_split_gemv_add(
        &self,
        stream: &Arc<CudaStream>,
        wq: &CudaSlice<i8>,
        ws: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        residual: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        summed: &mut CudaSlice<f32>,
        k_dim: usize,
        n_rows: usize,
    ) -> Result<(), GdnBlockError> {
        check_len("split gemv x", k_dim, x.len())?;
        check_len("split gemv residual", n_rows, residual.len())?;
        check_len("split gemv out", n_rows, out.len())?;
        check_len("split gemv summed", n_rows, summed.len())?;
        check_len("split gemv wq", n_rows * k_dim, wq.len())?;
        check_len("split gemv ws", n_rows * k_dim / QK8_0, ws.len())?;
        let cfg = LaunchConfig {
            grid_dim: ((n_rows as u32).div_ceil(PROJ_WARPS), 1, 1),
            block_dim: (32, PROJ_WARPS, 1),
            shared_mem_bytes: 0,
        };
        let (k_i32, n_i32) = (k_dim as i32, n_rows as i32);
        let mut builder = stream.launch_builder(&self.proj_split_gemv_add);
        builder
            .arg(wq)
            .arg(ws)
            .arg(x)
            .arg(residual)
            .arg(&mut *out)
            .arg(&mut *summed)
            .arg(&k_i32)
            .arg(&n_i32);
        // SAFETY: as `project_split_gemv` below, plus `residual` and `summed`
        // checked to `n_rows` and indexed by the same `n` the output is.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// The one-token projection over the repacked split layout.
    ///
    /// See `gdn_proj_split_gemv`: this exists because Q8_0's 34-byte block
    /// stride makes the untiled kernel's warp read straddle a sector boundary
    /// fifteen times in sixteen, and the repacked quants do not.
    #[allow(clippy::too_many_arguments)]
    pub fn project_split_gemv(
        &self,
        stream: &Arc<CudaStream>,
        wq: &CudaSlice<i8>,
        ws: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        k_dim: usize,
        n_rows: usize,
    ) -> Result<(), GdnBlockError> {
        check_len("split gemv x", k_dim, x.len())?;
        check_len("split gemv out", n_rows, out.len())?;
        check_len("split gemv wq", n_rows * k_dim, wq.len())?;
        check_len("split gemv ws", n_rows * k_dim / QK8_0, ws.len())?;
        let cfg = LaunchConfig {
            grid_dim: ((n_rows as u32).div_ceil(PROJ_WARPS), 1, 1),
            block_dim: (32, PROJ_WARPS, 1),
            shared_mem_bytes: 0,
        };
        let (k_i32, n_i32) = (k_dim as i32, n_rows as i32);
        let mut builder = stream.launch_builder(&self.proj_split_gemv);
        builder
            .arg(wq)
            .arg(ws)
            .arg(x)
            .arg(&mut *out)
            .arg(&k_i32)
            .arg(&n_i32);
        // SAFETY: one warp per output row over a grid covering `n_rows` and
        // returning above it; every buffer was length-checked immediately
        // above against exactly the extent the kernel indexes.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// `out[t][n] = sum_k weight[n][k] * x[t][k]`.
    ///
    /// `k_dim` is the GGUF tensor's `ne[0]` — the *input* width — and `n_rows`
    /// its `ne[1]`. Getting those the wrong way round is the transposed
    /// comparison `docs/ORACLE.md` section 6.1 exists to prevent, so they are
    /// named for the file's convention rather than for a matrix one.
    #[allow(clippy::too_many_arguments)]
    pub fn project(
        &self,
        stream: &Arc<CudaStream>,
        weight: Projection<'_>,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        k_dim: usize,
        n_rows: usize,
        tokens: usize,
    ) -> Result<(), GdnBlockError> {
        check_len("project x", tokens * k_dim, x.len())?;
        check_len("project out", tokens * n_rows, out.len())?;
        if !k_dim.is_multiple_of(QK8_0) {
            return Err(GdnBlockError::ShapeMismatch {
                what: "project k_dim (must be a whole number of Q8_0 blocks)",
                expected: k_dim.next_multiple_of(QK8_0),
                got: k_dim,
            });
        }
        if tokens == 0 || n_rows == 0 {
            return Ok(());
        }

        let cfg = LaunchConfig {
            grid_dim: ((n_rows as u32).div_ceil(PROJ_WARPS), tokens as u32, 1),
            block_dim: (32, PROJ_WARPS, 1),
            shared_mem_bytes: 0,
        };
        let k_i32 = k_dim as i32;
        let n_i32 = n_rows as i32;
        let t_i32 = tokens as i32;

        match weight {
            Projection::Q8_0(bytes) => {
                let expected = n_rows * k_dim / QK8_0 * BLOCK_Q8_0_BYTES;
                if bytes.len() != expected {
                    return Err(GdnBlockError::ShapeMismatch {
                        what: "project Q8_0 weight (bytes)",
                        expected,
                        got: bytes.len(),
                    });
                }
                // One token has nothing to tile: the weight is already read
                // exactly once, and the tiled kernel would carry seven unused
                // accumulators per thread for no benefit. Above one token the
                // untiled kernel re-reads the whole weight matrix per token,
                // which was 59.1% of a 512-token prefill pass.
                if tokens == 1 {
                    let mut builder = stream.launch_builder(&self.proj_q8_0);
                    builder
                        .arg(bytes)
                        .arg(x)
                        .arg(&mut *out)
                        .arg(&k_i32)
                        .arg(&n_i32);
                    // SAFETY: one warp per output row over a grid covering
                    // `n_rows` rows and returning above it, so the last byte
                    // any lane reads is `(n_rows - 1) * (k_dim/32) * 34 + 33`,
                    // which is the length checked immediately above. `x` and
                    // `out` were length-checked against `tokens * k_dim` and
                    // `tokens * n_rows`.
                    unsafe { builder.launch(cfg) }?;
                } else {
                    let (slot, tile) = proj_tile_for(tokens);
                    let rows = PROJ_ROWS[slot];
                    let tiled = LaunchConfig {
                        grid_dim: (
                            (n_rows as u32).div_ceil(PROJ_WARPS * rows),
                            (tokens as u32).div_ceil(tile),
                            1,
                        ),
                        block_dim: (32, PROJ_WARPS, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut builder = stream.launch_builder(&self.proj_tiled[slot]);
                    builder
                        .arg(bytes)
                        .arg(x)
                        .arg(&mut *out)
                        .arg(&k_i32)
                        .arg(&n_i32)
                        .arg(&t_i32);
                    // SAFETY: as above for the weight, which is indexed
                    // identically. The token axis is covered by
                    // `ceil(tokens / PROJ_TILE)` blocks that return above
                    // `n_tokens`, and the ragged final tile takes the guarded
                    // path, so no lane reads `x` or writes `out` past
                    // `tokens - 1`.
                    unsafe { builder.launch(tiled) }?;
                }
            }
            Projection::F32(values) => {
                check_len("project f32 weight", n_rows * k_dim, values.len())?;
                let mut builder = stream.launch_builder(&self.proj_f32);
                builder
                    .arg(values)
                    .arg(x)
                    .arg(&mut *out)
                    .arg(&k_i32)
                    .arg(&n_i32);
                // SAFETY: as above, with the weight indexed by
                // `n * k_dim + i` for `i < k_dim`, which is the length checked.
                unsafe { builder.launch(cfg) }?;
            }
        }
        Ok(())
    }

    /// `out = silu(x)` over `n` elements. `out` may alias `x`.
    pub fn silu(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), GdnBlockError> {
        check_len("silu x", n, x.len())?;
        check_len("silu out", n, out.len())?;
        if n == 0 {
            return Ok(());
        }
        let n_i64 = n as i64;
        let mut builder = stream.launch_builder(&self.silu);
        builder.arg(x).arg(&mut *out).arg(&n_i64);
        // SAFETY: the grid-stride loop is bounded by `n`, the checked length of
        // both buffers.
        unsafe { builder.launch(elementwise_cfg(n)) }?;
        Ok(())
    }

    /// SiLU and the q/k/v split, in one launch.
    ///
    /// `silu` is `conv_output_silu-N` and is still produced; see the kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn silu_split_qkv(
        &self,
        stream: &Arc<CudaStream>,
        conv: &CudaSlice<f32>,
        silu: &mut CudaSlice<f32>,
        q: &mut CudaSlice<f32>,
        k: &mut CudaSlice<f32>,
        v: &mut CudaSlice<f32>,
        tokens: usize,
    ) -> Result<(), GdnBlockError> {
        let g = self.geometry;
        let narrow = tokens * g.key_dim();
        let wide = tokens * g.value_dim();
        check_len("split conv", tokens * g.conv_dim(), conv.len())?;
        check_len("split silu", tokens * g.conv_dim(), silu.len())?;
        check_len("split q", narrow, q.len())?;
        check_len("split k", narrow, k.len())?;
        check_len("split v", wide, v.len())?;
        if tokens == 0 {
            return Ok(());
        }

        let cfg = LaunchConfig {
            grid_dim: (g.value_heads as u32, tokens as u32, 1),
            block_dim: (g.head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let (head_dim, qk_heads, value_heads) =
            (g.head_dim as i32, g.qk_heads as i32, g.value_heads as i32);
        let mut builder = stream.launch_builder(&self.silu_split);
        builder
            .arg(conv)
            .arg(&mut *silu)
            .arg(&mut *q)
            .arg(&mut *k)
            .arg(&mut *v)
            .arg(&head_dim)
            .arg(&qk_heads)
            .arg(&value_heads);
        // SAFETY: as `split_qkv` below, with the addition that `silu` is
        // written at the same `conv`-shaped indices it is read from, and both
        // were checked to hold `tokens * conv_dim`.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// Slice the convolved stream into q, k and v.
    ///
    /// `q` and `k` come out `[tokens][qk_heads][head_dim]` and `v`
    /// `[tokens][value_heads][head_dim]` — the shapes llama.cpp captures, and
    /// the shapes both mixer kernels want now that they index the query/key
    /// head as `h % qk_heads` themselves. This used to broadcast q and k up to
    /// `value_heads` because they did not.
    ///
    /// Superseded on the forward path by [`Self::silu_split_qkv`]. Kept
    /// because `tests/gdn_block.rs` runs it on the golden capture's own
    /// `conv_output_silu-N`, which checks the split without the nonlinearity
    /// in front of it.
    pub fn split_qkv(
        &self,
        stream: &Arc<CudaStream>,
        conv: &CudaSlice<f32>,
        q: &mut CudaSlice<f32>,
        k: &mut CudaSlice<f32>,
        v: &mut CudaSlice<f32>,
        tokens: usize,
    ) -> Result<(), GdnBlockError> {
        let g = self.geometry;
        let narrow = tokens * g.key_dim();
        let wide = tokens * g.value_dim();
        check_len("split conv", tokens * g.conv_dim(), conv.len())?;
        check_len("split q", narrow, q.len())?;
        check_len("split k", narrow, k.len())?;
        check_len("split v", wide, v.len())?;
        if tokens == 0 {
            return Ok(());
        }

        let cfg = LaunchConfig {
            grid_dim: (g.value_heads as u32, tokens as u32, 1),
            block_dim: (g.head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let (head_dim, qk_heads, value_heads) =
            (g.head_dim as i32, g.qk_heads as i32, g.value_heads as i32);
        let mut builder = stream.launch_builder(&self.split);
        builder
            .arg(conv)
            .arg(&mut *q)
            .arg(&mut *k)
            .arg(&mut *v)
            .arg(&head_dim)
            .arg(&qk_heads)
            .arg(&value_heads);
        // SAFETY: the grid is (value_heads, tokens) and the block head_dim
        // threads, so every write to `v` is at
        // `(t * value_heads + h) * head_dim + d` inside `wide`, and the writes
        // to `q` and `k` are guarded by `h < qk_heads`, which puts them inside
        // `narrow`. Every read is inside `t * conv_dim + conv_dim`.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// alpha, beta, and the three gate quantities, in one launch.
    ///
    /// Replaces two `gdn_proj_f32` launches and a `gdn_gates` launch. See the
    /// kernel for why three launches for 32 output rows each was the cost
    /// rather than the arithmetic.
    #[allow(clippy::too_many_arguments)]
    pub fn alpha_beta_gates(
        &self,
        stream: &Arc<CudaStream>,
        w_alpha: &CudaSlice<f32>,
        w_beta: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        a: &CudaSlice<f32>,
        alpha: &mut CudaSlice<f32>,
        beta_raw: &mut CudaSlice<f32>,
        a_softplus: &mut CudaSlice<f32>,
        log_decay: &mut CudaSlice<f32>,
        beta: &mut CudaSlice<f32>,
        tokens: usize,
    ) -> Result<(), GdnBlockError> {
        let g = self.geometry;
        let heads = g.value_heads;
        let n = tokens * heads;
        check_len("alpha weight", heads * g.hidden, w_alpha.len())?;
        check_len("beta weight", heads * g.hidden, w_beta.len())?;
        check_len("alpha-beta x", tokens * g.hidden, x.len())?;
        check_len("gates dt_bias", heads, dt_bias.len())?;
        check_len("gates ssm_a", heads, a.len())?;
        check_len("gates alpha", n, alpha.len())?;
        check_len("gates beta_raw", n, beta_raw.len())?;
        check_len("gates a_softplus", n, a_softplus.len())?;
        check_len("gates log_decay", n, log_decay.len())?;
        check_len("gates beta", n, beta.len())?;
        if n == 0 {
            return Ok(());
        }

        // A partial token band costs its empty lanes in full, so below a whole
        // band the one-token instantiation is launched instead. See the kernel.
        let (f, tt) = if tokens < GATE_TT as usize {
            (&self.alpha_beta_gates_t1, 1)
        } else {
            (&self.alpha_beta_gates, GATE_TT)
        };
        let cfg = LaunchConfig {
            grid_dim: (
                (heads as u32).div_ceil(PROJ_WARPS),
                (tokens as u32).div_ceil(tt),
                1,
            ),
            block_dim: (32, PROJ_WARPS, 1),
            shared_mem_bytes: 0,
        };
        let (k_i32, h_i32, t_i32) = (g.hidden as i32, heads as i32, tokens as i32);
        let mut builder = stream.launch_builder(f);
        builder
            .arg(w_alpha)
            .arg(w_beta)
            .arg(x)
            .arg(dt_bias)
            .arg(a)
            .arg(&mut *alpha)
            .arg(&mut *beta_raw)
            .arg(&mut *a_softplus)
            .arg(&mut *log_decay)
            .arg(&mut *beta)
            .arg(&k_i32)
            .arg(&h_i32)
            .arg(&t_i32);
        // SAFETY: one warp per (head, GATE_TT-wide token band) over a grid
        // that covers both and returns above `heads`; every buffer was
        // length-checked immediately above against exactly the extent the
        // kernel indexes, the band is masked against `tokens`, and the two
        // per-head tables are indexed by `n < heads`.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// The decay and write gates: `a_softplus`, the log-decay, and `beta`.
    ///
    /// Superseded on the forward path by [`Self::alpha_beta_gates`], which
    /// folds it into the two projections that feed it. Kept because
    /// `tests/gdn_block.rs` runs it on the golden capture's own `alpha-N` and
    /// `beta-N` to check the gate arithmetic on its own, which the fused
    /// kernel cannot be asked to do.
    #[allow(clippy::too_many_arguments)]
    pub fn gates(
        &self,
        stream: &Arc<CudaStream>,
        alpha: &CudaSlice<f32>,
        beta_raw: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        a: &CudaSlice<f32>,
        a_softplus: &mut CudaSlice<f32>,
        log_decay: &mut CudaSlice<f32>,
        beta: &mut CudaSlice<f32>,
        tokens: usize,
    ) -> Result<(), GdnBlockError> {
        let heads = self.geometry.value_heads;
        let n = tokens * heads;
        check_len("gates alpha", n, alpha.len())?;
        check_len("gates beta_raw", n, beta_raw.len())?;
        check_len("gates dt_bias", heads, dt_bias.len())?;
        check_len("gates ssm_a", heads, a.len())?;
        check_len("gates a_softplus", n, a_softplus.len())?;
        check_len("gates log_decay", n, log_decay.len())?;
        check_len("gates beta", n, beta.len())?;
        if n == 0 {
            return Ok(());
        }

        let heads_i32 = heads as i32;
        let n_i64 = n as i64;
        let mut builder = stream.launch_builder(&self.gates);
        builder
            .arg(alpha)
            .arg(beta_raw)
            .arg(dt_bias)
            .arg(a)
            .arg(&mut *a_softplus)
            .arg(&mut *log_decay)
            .arg(&mut *beta)
            .arg(&heads_i32)
            .arg(&n_i64);
        // SAFETY: the grid-stride loop is bounded by `n = tokens * heads`, the
        // checked length of five buffers, and `i % heads < heads` bounds the
        // two per-head ones.
        unsafe { builder.launch(elementwise_cfg(n)) }?;
        Ok(())
    }

    /// `out = a + b` over `n` elements.
    pub fn add(
        &self,
        stream: &Arc<CudaStream>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), GdnBlockError> {
        check_len("add a", n, a.len())?;
        check_len("add b", n, b.len())?;
        check_len("add out", n, out.len())?;
        if n == 0 {
            return Ok(());
        }
        let n_i64 = n as i64;
        let mut builder = stream.launch_builder(&self.add);
        builder.arg(a).arg(b).arg(&mut *out).arg(&n_i64);
        // SAFETY: the grid-stride loop is bounded by `n`, checked for all three.
        unsafe { builder.launch(elementwise_cfg(n)) }?;
        Ok(())
    }

    /// Advance the recurrent state over `tokens` tokens and write their output.
    ///
    /// `q` and `k` are `[tokens][qk_heads][head_dim]`, `v` is
    /// `[tokens][value_heads][head_dim]`, and `log_decay` and `beta` are
    /// `[tokens][value_heads]`. The kernels pair value head `h` with query/key
    /// head `h % qk_heads`; they took an already-broadcast q and k until they
    /// learned to do that themselves.
    ///
    /// One token takes the recurrent kernel and more than one takes the chunked
    /// kernel. The two must agree: prefill fills the state a later decode
    /// resumes from.
    #[allow(clippy::too_many_arguments)]
    pub fn mix(
        &mut self,
        stream: &Arc<CudaStream>,
        state: &mut GdnState,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        log_decay: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        tokens: usize,
    ) -> Result<Mixer, GdnBlockError> {
        let g = self.geometry;
        let narrow = tokens * g.key_dim();
        let wide = tokens * g.value_dim();
        check_len("mix q", narrow, q.len())?;
        check_len("mix k", narrow, k.len())?;
        check_len("mix v", wide, v.len())?;
        check_len("mix log_decay", tokens * g.value_heads, log_decay.len())?;
        check_len("mix beta", tokens * g.value_heads, beta.len())?;
        check_len("mix out", wide, out.len())?;
        check_len("mix state", g.recurrent_state_len(), state.recurrent.len())?;

        if tokens == 1 {
            self.recurrent.step(
                stream,
                &mut self.recurrent_scratch,
                &mut state.recurrent,
                q,
                k,
                v,
                log_decay,
                beta,
                out,
            )?;
            Ok(Mixer::Recurrent)
        } else {
            self.chunked.scan(
                stream,
                &mut self.chunked_scratch,
                &mut state.recurrent,
                q,
                k,
                v,
                log_decay,
                beta,
                out,
                tokens,
            )?;
            Ok(Mixer::Chunked)
        }
    }
}

/// A fixed grid over a grid-stride loop, so the launch shape does not depend on
/// a host-side length and stays capturable in a CUDA graph (AGENTS.md rule 5).
fn elementwise_cfg(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (
            n.div_ceil(ELEMENTWISE_BLOCK as usize).min(1024) as u32,
            1,
            1,
        ),
        block_dim: (ELEMENTWISE_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Rejects a buffer whose length disagrees with the declared geometry.
fn check_len(what: &'static str, expected: usize, got: usize) -> Result<(), GdnBlockError> {
    if got == expected {
        Ok(())
    } else {
        Err(GdnBlockError::ShapeMismatch {
            what,
            expected,
            got,
        })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_projection_tiles_match_the_kernel() {
        // NVRTC compiles from a string with no access to Rust constants, so
        // the tile width is spelled twice. If they drift, the launch grid
        // stops covering the token axis and the tail tokens are silently never
        // written — finite, plausible, wrong.
        for t in PROJ_TILES {
            assert!(
                GDN_BLOCK_SRC
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
                    .contains(&format!(
                        "GDN_PROJ_TILED(gdn_proj_q8_0_t{t}, {t}, {})",
                        PROJ_ROWS[PROJ_TILES.iter().position(|&w| w == t).unwrap()],
                    )),
                "the kernel must instantiate a {t}-wide tile, because \
                 `proj_tile_for` will try to launch one",
            );
        }
        // Below the widest declared tile, the chooser picks the *smallest*
        // tile that covers `tokens` in one slice — which can be wider than
        // `tokens`, deliberately, to avoid a second slice. `proj_tile_for` is
        // never called below 2 tokens: `project` special-cases `tokens == 1`
        // to the untiled kernel before this chooser is reached.
        for (tokens, want) in [
            (2usize, 2u32),
            (3, 4),
            (4, 4),
            (5, 8),
            (7, 8),
            (8, 8),
            (9, 16),
            (15, 16),
            (16, 16),
        ] {
            let (slot, tile) = proj_tile_for(tokens);
            assert_eq!(tile, PROJ_TILES[slot]);
            assert_eq!(
                tile, want,
                "{tokens} tokens should take the smallest tile that covers it in one slice",
            );
            assert!(
                tile as usize >= tokens,
                "{tokens} tokens chose a {tile}-wide tile that needs a second slice",
            );
        }
        // At and above the widest declared tile, no single tile covers
        // `tokens` any more, and the chooser falls back to the original
        // rule: the widest tile, however many slices that takes. Proven at
        // 128 and 512 tokens in `PROJ_ROWS`'s own sweep.
        for tokens in [17usize, 19, 31, 32, 128, 512] {
            let (slot, tile) = proj_tile_for(tokens);
            assert_eq!(tile, PROJ_TILES[slot]);
            assert_eq!(
                tile, 16,
                "{tokens} tokens is past every single-slice tile and must fall back to the widest",
            );
        }
    }

    #[test]
    fn the_tiled_projection_reads_each_operand_once_per_tile() {
        // Both defects this kernel exists to fix are structural, not numeric —
        // every version computes the same thing, just at very different cost,
        // so there is nothing numeric to assert on and the source is the only
        // place the property is visible.
        //
        // 1. The weight must be dequantized outside the *token* loop, or each
        //    token re-reads the whole weight matrix (the original defect: 59%
        //    of prefill).
        // 2. The activation column must be loaded outside the *row* loop, or
        //    each weight row re-reads x (~2.1 GB of L2 traffic per call).
        let tiled = GDN_BLOCK_SRC
            .split("#define GDN_PROJ_TILED")
            .nth(1)
            .expect("the tiled kernel must exist");
        let activation_load = tiled
            .find("xv[i] = x[")
            .expect("the tiled kernel loads an activation column");
        let row_loop = tiled
            .find("for (int r = 0; r < (RR); ++r) {")
            .expect("the tiled kernel loops over its row band");
        assert!(
            activation_load < row_loop,
            "the activation column must be loaded before the row loop, or every \
             weight row re-reads x and the row band buys nothing",
        );

        let weight_load = tiled[row_loop..]
            .find("load_half_le(blk)")
            .expect("the tiled kernel dequantizes inside the row loop");
        let token_loop = tiled[row_loop..]
            .find("for (int i = 0; i < (TT); ++i) acc[r][i]")
            .expect("the tiled kernel accumulates over its token tile");
        assert!(
            weight_load < token_loop,
            "the weight must be dequantized before the token loop, or the tile \
             buys nothing",
        );
    }

    use super::*;

    /// Where a snippet starts in the kernel source, or a failure naming it.
    fn at(needle: &str) -> usize {
        GDN_BLOCK_SRC
            .find(needle)
            .unwrap_or_else(|| panic!("kernel source no longer contains `{needle}`"))
    }

    fn geometry() -> GdnGeometry {
        GdnGeometry::from_config(&ModelConfig::qwen3_6_35b_a3b(), 19, 1e-6)
    }

    #[test]
    fn the_split_emits_the_captures_own_query_key_width_and_broadcasts_nothing() {
        // The query/key head map — `fastmodulo(h_idx, n_k_heads)` in
        // llama.cpp's fused op, `ggml_repeat_4d`'s tiling in its fallback —
        // now lives in the two GDN kernels, whose own structural tests assert
        // `int qk_head = h % qk_heads;` and that `heads_per_kv` does not
        // appear. What this file must not do is re-introduce a broadcast on
        // top of that, which would pair every value head with query/key head
        // `(h % 16) % 16` at 2x the Gram work and *still* look correct.
        let body =
            &GDN_BLOCK_SRC[at("__global__ void gdn_split_qkv(")..at("__global__ void gdn_gates(")];
        assert!(
            body.contains("if (h < qk_heads) {"),
            "the split no longer writes q and k at their own head count",
        );
        assert!(
            !body.contains("h % qk_heads") && !body.contains("h / qk_heads"),
            "the split re-introduced a head map; that belongs in the kernels",
        );
        // And the widths it writes into differ, which is what makes the
        // absence of a broadcast observable rather than merely stated.
        let g = geometry();
        assert_eq!(g.key_dim(), 2048);
        assert_eq!(g.value_dim(), 4096);
        assert_eq!(g.value_dim(), 2 * g.key_dim());
    }

    #[test]
    fn the_log_decay_is_the_product_and_is_not_negated_again() {
        // `ssm_a` is stored already negated — every entry of `blk.N.ssm_a` is
        // negative — so `softplus(alpha + dt_bias) * ssm_a` *is* the log-decay.
        // A second negation makes the state grow instead of decay, which stays
        // finite over 19 tokens and diverges over a real context.
        assert!(GDN_BLOCK_SRC.contains("log_decay[i] = sp * ssm_a[h];"));
        let body = &GDN_BLOCK_SRC[at("__global__ void gdn_gates(")..at("__global__ void gdn_add(")];
        assert!(!body.contains("-ssm_a"), "ssm_a was negated a second time");
        assert!(!body.contains("-sp *"), "the softplus was negated");
    }

    #[test]
    fn the_softplus_keeps_its_large_argument_passthrough() {
        // `ggml_cuda_op_softplus` is `(x > 20) ? x : log(1 + exp(x))`. Dropping
        // the branch overflows `expf` at x > 88 and returns inf rather than
        // saturating, which would poison every subsequent decay.
        assert!(GDN_BLOCK_SRC.contains("float sp = (a > 20.0f) ? a : logf(1.0f + expf(a));"));
    }

    #[test]
    fn the_projection_reads_a_gguf_row_as_ne0_contiguous_elements() {
        // The layout claim `docs/ORACLE.md` section 6.1 proves, as code: a
        // tensor with dims [K, N] is N rows of K contiguous elements, so row n
        // begins at `n * (K/32) * 34` bytes. Reading it the other way round
        // produces a transposed comparison that looks like a kernel bug.
        assert!(
            GDN_BLOCK_SRC
                .contains("const unsigned char* row = weight + (long long)n * blocks * 34;")
        );
        assert!(GDN_BLOCK_SRC.contains("const float* row = weight + (long long)n * k_dim;"));
    }

    #[test]
    fn the_q8_0_row_stride_matches_the_kernel() {
        // The kernel spells 32 and 34 as literals because NVRTC compiles from a
        // string with no access to Rust constants. This is the check that the
        // two spellings agree, and that they agree with the upstream block
        // layout in `ggml/src/ggml-common.h`.
        assert_eq!(QK8_0, xabe_cuda::kernels::dequant::QK8_0);
        assert_eq!(
            BLOCK_Q8_0_BYTES,
            xabe_cuda::kernels::dequant::BLOCK_Q8_0_BYTES
        );
        assert_eq!(BLOCK_Q8_0_BYTES, 2 + QK8_0);
        // 2048 input elements is 64 blocks, 2,176 bytes — the same row stride
        // `golden.rs` dequantizes the embedding table with.
        assert_eq!(2048 / QK8_0 * BLOCK_Q8_0_BYTES, 2176);
    }

    #[test]
    fn the_quants_are_read_as_signed() {
        // Reading int8 codes as unsigned flips the sign of roughly half of
        // every tensor while leaving magnitudes plausible.
        assert!(GDN_BLOCK_SRC.contains("signed char q = (signed char)blk[2 + lane];"));
    }

    #[test]
    fn the_geometry_derives_the_real_widths_from_model_config() {
        let g = geometry();
        assert_eq!(g.hidden, 2048);
        assert_eq!(g.head_dim, 128);
        assert_eq!(g.value_heads, 32);
        assert_eq!(g.qk_heads, 16);
        assert_eq!(g.conv_kernel, 4);
        assert_eq!(g.chunk_len, 64);
        // 2 * 16 * 128 + 32 * 128 — the width of `attn_qkv.weight`'s output and
        // of `ssm_conv1d.weight`'s channel count.
        assert_eq!(g.conv_dim(), 8192);
        assert_eq!(g.value_dim(), 4096);
        assert_eq!(g.key_dim(), 2048);
        // The fused stream is two key widths and one value width, which is
        // what makes the slice offsets in `gdn_split_qkv` a partition.
        assert_eq!(g.conv_dim(), 2 * g.key_dim() + g.value_dim());
        // Three carried inputs per channel, and 2 MiB of recurrent state.
        assert_eq!(g.conv_state_len(), 8192 * 3);
        assert_eq!(g.recurrent_state_len(), 32 * 128 * 128);
        assert_eq!(g.recurrent_state_len() * 4, 2 << 20);
    }

    #[test]
    fn a_prompt_shorter_than_a_chunk_still_takes_the_chunked_path() {
        // The golden's 19 tokens are well under `chunk_len = 64`, so the
        // chunked form runs a single ragged chunk. That is the path the oracle
        // exercises, and reading `19 < 64` as "use the recurrent form" would
        // silently test something else.
        let g = geometry();
        assert!(g.max_tokens < g.chunk_len);
        assert!(g.max_tokens > 1);
    }

    #[test]
    fn geometry_is_validated_not_assumed() {
        // The checks that run without a device.
        assert!(
            GdnBlockError::MissingWeight {
                role: Role::GdnQkv,
                layer: 3,
            }
            .to_string()
            .contains("attn_qkv.weight")
        );
        assert!(
            GdnBlockError::UnsupportedQuant {
                role: Role::GdnOut,
                found: GgmlType::Q6K,
                expected: GgmlType::Q8_0,
            }
            .to_string()
            .contains("q6_K")
        );
        assert!(
            GdnBlockError::ShapeMismatch {
                what: "mix q",
                expected: 77824,
                got: 4096,
            }
            .to_string()
            .contains("77824")
        );
    }

    #[test]
    fn the_elementwise_grid_is_capped_so_it_stays_graph_capturable() {
        // A grid derived from `n` would change shape with the batch size and
        // break replay of a captured graph. The loop is grid-stride, so a
        // capped grid is still correct at any length.
        assert_eq!(elementwise_cfg(1).grid_dim.0, 1);
        assert_eq!(elementwise_cfg(256).grid_dim.0, 1);
        assert_eq!(elementwise_cfg(257).grid_dim.0, 2);
        assert_eq!(elementwise_cfg(1 << 30).grid_dim.0, 1024);
    }
}
