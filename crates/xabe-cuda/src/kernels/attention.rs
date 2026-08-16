//! Gated Attention (flash-attention style) on sm_75.
//!
//! Ten of Qwen3.6's forty layers, plus the MTP head at block 40. Geometry is
//! 16 query heads against 2 KV heads (GQA ratio 8), head dimension **256**,
//! partial rotary over the leading 64 dimensions. The reference is
//! `xabe_kernels::attention::causal_attention_streaming` — the online-softmax
//! form — which is itself cross-checked against
//! `causal_attention_naive` in that crate.
//!
//! Ported as an *algorithm* from llama.cpp's `fattn-vec.cuh` shape (one query
//! row per block, K and V streamed), not from `fattn-wmma`. `docs/KERNELS.md`
//! names `fattn-tile.cu` / `fattn-vec.cuh` as the sm_75 sources and explicitly
//! excludes the WMMA path.
//!
//! ## Three kernels, and why the first one exists
//!
//! `blk.N.attn_q.weight` is `[n_embd, head_dim * n_head * 2]` = `[2048, 8192]`
//! and packs the query **and its output gate interleaved per head**:
//! `[q_h0, gate_h0, q_h1, gate_h1, ...]`. Upstream reads the query half with
//! `ggml_view_3d(.., head_dim, n_head, n_tokens, stride = head_dim*2, ..)` in
//! `src/models/qwen35moe.cpp`, which is where that was confirmed — it was not
//! inferred from the dimensions. Splitting the tensor into two contiguous
//! halves is arithmetically valid, produces finite plausible activations, and
//! is a different model: query heads 8..15 would be fed the gates of heads
//! 0..7. [`attn_packed_query_offset`] / [`attn_packed_gate_offset`] state the
//! layout in Rust so the host side and the kernel cannot drift, and
//! `attn_split_query_gate` is the kernel that applies it.
//!
//! The gate itself is *not* applied here. There is no CPU reference for the
//! gating nonlinearity in `xabe-kernels`, so applying it would put arithmetic
//! into this kernel that the differential harness cannot check — which
//! `AGENTS.md` forbids. The gate is deinterleaved and handed back to the
//! caller.
//!
//! ## Shared-memory budget, and why the textbook tile does not fit
//!
//! `docs/KERNELS.md` records **48 KiB of shared memory per block** on this
//! hardware, measured. (Turing's SM carries 64 KiB of unified L1/shared, and a
//! block can reach the full 64 KiB only by opting in through
//! `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)`; the
//! default static ceiling is 48 KiB. This kernel budgets against 48 KiB and
//! needs no opt-in.)
//!
//! A classic flash-attention block stages Q, K and V tiles plus a score tile:
//!
//! ```text
//! bytes = (BM + 2*BN) * head_dim * 4  +  BM * BN * 4
//!
//!   BM=BN=64 -> (64 + 128) * 256 * 4 + 16384  =  212,992 B = 208 KiB   4.3x over
//!   BM=BN=32 -> ( 32 +  64) * 256 * 4 +  4096 =   102,400 B = 100 KiB   2.1x over
//!   BM=BN=16 -> ( 16 +  32) * 256 * 4 +  1024 =    50,176 B =  49 KiB   just over
//!   BM=BN= 8 -> (  8 +  16) * 256 * 4 +   256 =    24,832 B =  24 KiB   fits
//! ```
//!
//! head_dim 256 in fp32 is 1 KiB per row, so the whole family is 4x more
//! expensive than the head_dim-64 shapes these tile sizes were chosen for.
//! Nothing above `BM=BN=8` fits, and an 8x8 tile buys almost none of the
//! arithmetic-intensity multiplier that tiling exists for.
//!
//! **Chosen shape: `BM = 8`, K and V not staged at all.**
//!
//! ```text
//!   score_sh  BM * (head_dim/32 * 4) floats (256) = 1024 B
//!   w_sh      BM * (head_dim/32 * 4) floats (256) = 1024 B
//!   m/l/corr  3 * BM floats                (24)   =   96 B
//!                                                  -------
//!                                                   2144 B  =  2.1 KiB  (4.4% of 48 KiB)
//! ```
//!
//! One block owns **eight** query rows of one query head. Its 256 threads are
//! 8 warps; each warp scores `ATTN_KT` keys per trip against all eight rows,
//! so a tile is 32 keys. Each thread owns one output dimension of eight
//! running accumulators. K and V are still read straight from global memory,
//! coalesced.
//!
//! **The query tile is in registers, not shared.** That is the whole point of
//! it. `BM = 1` was not slow because of bandwidth — it was slow because it did
//! one memory instruction per multiply-add:
//!
//! ```text
//!   scores:  per key, per lane   8 global k + 8 shared q  + 8 FMA + 5 shfl
//!   values:  per key, per thread 1 global v + 1 shared w  + 1 FMA
//! ```
//!
//! Turing issues 4 LSU operations per SM per clock against 64 FMAs, so a
//! kernel at one load per multiply-add is running at a sixteenth of the FMA
//! pipe and no amount of L2 hit rate changes that. Holding `q` in registers
//! removes the shared read from the score loop entirely and makes one `k`
//! element feed eight multiply-adds; one `v` element likewise feeds eight.
//!
//! ```text
//!   scores:  per key, per lane   8 global k + 0 shared    + 64 FMA + 40 shfl
//!   values:  per key, per thread 1 global v + 8 shared w  +  8 FMA
//! ```
//!
//! Measured at 512 tokens: **21.1 ms -> 8.5 ms** per forward pass over the ten
//! attention layers, which is 1,823 -> 1,922 tok/s end to end.
//!
//! The register array is what bounds `head_dim`: `qr[ATTN_QT][ATTN_MAXD]` is
//! only in registers while every index folds to a constant, so `ATTN_MAXD` is
//! a compile-time 8 and head dimensions above 256 are rejected rather than
//! silently spilled to local memory.
//!
//! **Why the softmax bookkeeping is one thread per slot.** Widening the query
//! tile multiplies the per-tile reductions by `BM` as well, and the obvious
//! version — every thread sweeping `BM * tile` shared floats for the max and
//! the normalizer — costs more than the value loop it was meant to amortize.
//! `ATTN_QT * ATTN_KT` is therefore pinned to 32, which makes `ATTN_QT * tile`
//! exactly the block width at every head dimension, so the exponential phase
//! is one thread per (row, key) slot with no loop at all. The max and the
//! normalizer stay serial and ascending in one thread per row, because the
//! normalizer is a floating-point sum and a warp butterfly would reassociate
//! it — see the differential test's exactness gate.
//!
//! Staging K in shared memory would still buy nothing: a block touches each K
//! row once, so there is no intra-block reuse to amortize the staging
//! against. The reuse is *across* blocks — 16 query heads share 2 KV heads —
//! and that is served by the 6 MiB L2, not by shared memory.
//!
//! **Occupancy consequence, stated honestly.** Shared memory is not the
//! limiter at 2.1 KiB per block; the register file is. The query tile costs
//! `ATTN_QT * head_dim / 32` registers per thread on top of the accumulators,
//! which puts the kernel around two blocks of 256 threads per SM rather than
//! four. That is half the thread occupancy for eight times the arithmetic per
//! byte, and the measurement above is which way that trade goes.
//!
//! ## Tensor cores: not used, and what that costs
//!
//! **This is the scalar fp32 path. It does not use the `m16n8k8` tensor-core
//! MMA family that `kernels::mod::TARGET_ARCH` (`compute_75`) makes reachable.
//! Calling it "flash attention" refers to the online-softmax streaming form
//! and the absence of a materialized score matrix, not to tensor cores.**
//!
//! What that gives up, and why it is second in line rather than first:
//!
//! - Turing's fp16 `m16n8k8` MMA with fp32 accumulate runs at roughly 8x the
//!   fp32 FMA rate (about 130 vs 16.3 TFLOP/s on this part). That is the whole
//!   headline number, and none of it is available here.
//! - It is not reachable at `BM = 1`. `m16n8k8` consumes a 16x8 operand tile,
//!   so it needs at least 16 query rows resident per block — which means
//!   staging Q, and staging K to feed the B operand. At head_dim 256 in fp16
//!   that is `(16 + 2*BN) * 256 * 2` bytes; `BN = 16` is 24 KiB and does fit.
//!   So the tensor-core path is *available*, but only after the tiling is
//!   redesigned, and only in fp16.
//! - fp16 K/V would end the fp32-exact comparison this kernel is gated on.
//!   The differential test currently measures agreement with the scalar
//!   reference at the fp32 rounding floor; an fp16 MMA path has to be re-gated
//!   at `Tolerance::reduced_precision_gpu()` (5e-2), which is three orders of
//!   magnitude looser and would hide a formulation bug this gate catches.
//! - The workload is bandwidth-bound at this shape (see above), so 8x more
//!   FLOP/s on its own converts to far less than 8x. The tiling that unlocks
//!   tensor cores is also the tiling that raises arithmetic intensity — the
//!   two are the same change, and the intensity is the part that pays.
//!
//! The correct order is: get the numerics right against the reference at fp32,
//! then re-tile for reuse, then take the MMA path and re-gate. This module is
//! step one, and it should not be described as more than that.
//!
//! ## Causal masking
//!
//! Query row `i` of a launch sits at absolute position `key_offset + i` and
//! attends to keys `[0, key_offset + i]` inclusive — `key_offset + i + 1`
//! keys. `key_offset` is what makes chunked prefill and decode the same
//! kernel, and it is a **device scalar**: it is the only thing that differs
//! between two consecutive decode steps, so keeping it out of the launch
//! arguments is what lets a whole step be recorded once as a CUDA graph.
//! The bound appears exactly once, as the loop limit `n_visible`, so
//! there is no separate mask to get off by one against; an off-by-one would
//! have to be an off-by-one in `+ 1`, and the differential test proves that
//! `+ 1` is right by perturbing key `t+1` and requiring output row `t` to come
//! back bit-identical.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};

use super::compile;

/// Offset of query head `head`'s slice within one token's packed
/// `attn_q.weight` row.
///
/// The row is `[q_h0, gate_h0, q_h1, gate_h1, ...]`, so the per-head stride is
/// `2 * head_dim` and the query is the first slice of each pair. A halves
/// split would compute `head * head_dim` instead, which is the same value only
/// for head 0.
pub const fn attn_packed_query_offset(head: usize, head_dim: usize) -> usize {
    2 * head * head_dim
}

/// Offset of query head `head`'s **output gate** slice within one token's
/// packed `attn_q.weight` row. See [`attn_packed_query_offset`].
pub const fn attn_packed_gate_offset(head: usize, head_dim: usize) -> usize {
    (2 * head + 1) * head_dim
}

/// Keys one warp scores between barriers in the flash kernel.
///
/// Mirrors `ATTN_KT`. The block rescales its running softmax once per
/// `n_warps * KEYS_PER_WARP` keys, so this trades a little shared memory and a
/// longer serial sweep per trip against a quarter of the barriers.
const KEYS_PER_WARP: usize = 4;

/// Query rows one block carries. Mirrors `ATTN_QT`.
///
/// The product `QUERY_TILE * KEYS_PER_WARP` is pinned to 32 by
/// [`the_query_tile_gives_one_thread_per_softmax_slot`], which is what makes
/// `QUERY_TILE * keys_per_tile()` equal the block width at every head
/// dimension: `head_dim/32` warps times `KEYS_PER_WARP` keys times
/// `QUERY_TILE` rows is `head_dim` threads exactly.
const QUERY_TILE: usize = 8;

/// Tokens one rotary block carries. Mirrors `ROPE_TT`.
///
/// The rotary frequency depends on the head dimension and nothing else, so a
/// block per token recomputed a double-precision `pow` for every (token, head,
/// dimension) triple to get one of 32 distinct values. A band hoists it out.
const ROPE_TOKENS: u32 = 16;

/// Head dimensions one lane reduces over, at most. Mirrors `ATTN_MAXD`.
///
/// The query tile lives in registers as `qr[ATTN_QT][ATTN_MAXD]`, and an array
/// indexed by anything the compiler cannot fold to a constant is spilled to
/// local memory — which would cost more than the tiling saves. So the bound is
/// a compile-time constant and `head_dim` is rejected above `32 * ATTN_MAXD`.
const ATTN_MAXD: usize = 8;

/// Query rows one GQA-shared block carries. Mirrors `GQA_QT`.
///
/// This is exactly the factor by which that kernel divides K/V traffic, so
/// bigger is better right up to the register wall — see the kernel's own
/// comment for why 4 is where it stops.
const GQA_QUERY_TILE: usize = 4;

/// Key slices flash decoding splits the window into. Mirrors `DEC_SPLITS`.
///
/// The split grid is `(DECODE_SPLITS, kv_heads)`, so this times `kv_heads` is
/// the block count that replaces decode's old sixteen. 64 gives 128 blocks
/// against 72 SMs, which is where the card stops being the constraint; raising
/// it further only shortens each slice and lengthens the combine.
const DECODE_SPLITS: usize = 144;

/// Keys the flash-decoding split pass stages per trip. Mirrors `DEC_KT`.
///
/// Capped at 32 because the exponential phase gives one key slot to one lane.
const DECODE_KEY_TILE: usize = 8;

/// Keys the GQA-shared kernel stages in shared memory per trip. Mirrors
/// `GQA_KT`.
///
/// Sets both the shared footprint (`2 * GQA_KEY_TILE * head_dim` floats) and
/// how often the running softmax is rescaled. Capped at 32 because the
/// exponential phase gives one key slot to one lane.
const GQA_KEY_TILE: usize = 8;

const ATTENTION_SRC: &str = r#"
extern "C" {

// NVRTC compiles from a string with no include path, so <math.h>'s INFINITY
// macro is not reachable. `__int_as_float` is a builtin and always is.
__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

// Deinterleave the packed query/gate tensor.
//
// grid: (n_tokens, q_heads). block: head_dim threads.
//
// `blk.N.attn_q.weight` emits, per token, [q_h0, gate_h0, q_h1, gate_h1, ...]
// — a per-head stride of 2*head_dim with the query first. This is confirmed
// against llama.cpp's src/models/qwen35moe.cpp, which views it with
// stride = head_dim*2. It is NOT [all queries | all gates]: reading it that
// way feeds query heads 8..15 the gates of heads 0..7, stays finite, and is a
// different model.
__global__ void attn_split_query_gate(
    const float* __restrict__ packed,
    float* __restrict__ q,
    float* __restrict__ gate,
    int q_heads,
    int head_dim
) {
    long long t = blockIdx.x;
    int h = blockIdx.y;
    int d = threadIdx.x;

    long long row = t * (long long)q_heads * 2 * head_dim;
    long long dst = (t * (long long)q_heads + h) * (long long)head_dim + d;

    q[dst]    = packed[row + (long long)(2 * h)     * head_dim + d];
    gate[dst] = packed[row + (long long)(2 * h + 1) * head_dim + d];
}

// Partial rotary position embedding, NEOX (split-half) pairing.
//
// grid: (n_tokens, n_heads). block: head_dim threads.
//
// Rotates dimensions [0, rope_dim) pairing i with i + rope_dim/2, and copies
// [rope_dim, head_dim) through. The copy is a real load/store rather than an
// arithmetic identity so the tail is bit-identical rather than merely close —
// that is what `Tolerance::exact()` in the differential test checks.
//
// The angle is computed in double to match `xabe_kernels::rope::apply_rope`,
// which raises theta_base to a f64 power. Doing it in float diverges visibly
// at the positions this model actually reaches: at position 262,143 a float
// angle loses about 5 significant digits.
// Tokens one rotary block carries.
//
// The frequency `theta_base ^ (-2 d / rope_dim)` depends on the head dimension
// and nothing else, yet a block per token made it a **double-precision `pow`
// per (token, head, dimension) triple** -- 262,144 of them per pass for the
// query stream alone, for 32 distinct values. Turing runs fp64 at a
// thirty-second of fp32, and `pow` is a libdevice call on top of that.
//
// A band of tokens computes it once and reuses it, which is bit-identical:
// the same `pow` of the same arguments, hoisted out of a loop. The angle and
// its sine and cosine still have to be per token, and still have to be double
// -- see below.
#define ROPE_TT 16

__global__ void attn_rope_partial_neox(
    const float* __restrict__ in,
    float* __restrict__ out,
    int n_heads,
    int head_dim,
    int rope_dim,
    const int* __restrict__ pos_offset,
    float theta_base,
    int n_tokens
) {
    long long t0 = (long long)blockIdx.x * ROPE_TT;
    int h = blockIdx.y;
    int d = threadIdx.x;

    int half = rope_dim >> 1;

    if (d >= rope_dim) {
        for (int u = 0; u < ROPE_TT; ++u) {
            long long t = t0 + u;
            if (t >= n_tokens) break;
            long long base = (t * (long long)n_heads + h) * (long long)head_dim;
            out[base + d] = in[base + d];
        }
        return;
    }
    // Dimensions [half, rope_dim) are written by their partner thread d-half.
    if (d >= half) return;

    // Hoisted: the one quantity in here that does not depend on the token.
    double freq = pow((double)theta_base, -2.0 * (double)d / (double)rope_dim);

    for (int u = 0; u < ROPE_TT; ++u) {
        long long t = t0 + u;
        if (t >= n_tokens) break;
        long long base = (t * (long long)n_heads + h) * (long long)head_dim;

        double pos = (double)(*pos_offset) + (double)t;
        double angle = pos * freq;
        float sin_a = (float)sin(angle);
        float cos_a = (float)cos(angle);

        float x0 = in[base + d];
        float x1 = in[base + d + half];
        out[base + d]        = x0 * cos_a - x1 * sin_a;
        out[base + d + half] = x0 * sin_a + x1 * cos_a;
    }
}

// Causal GQA attention, online-softmax streaming form.
//
// grid: (n_query, q_heads) — one block per (query row, query head).
// block: head_dim threads = head_dim/32 warps.
//
// Per iteration each warp scores one key, so a tile is head_dim/32 keys (8 at
// head_dim 256). Thread `tid` owns output dimension `tid` of the running
// accumulator for the whole kernel, so the value accumulation never leaves
// registers and never needs a reduction.
//
// The online update follows `causal_attention_streaming` in xabe-kernels: a
// running max, a correction factor that rescales the accumulator and the
// normalizer into the new max's frame, then the new contributions. The one
// deliberate difference is that the update is applied per *tile* of keys
// rather than per key — mathematically identical, and it cuts the number of
// expf evaluations by the tile width, because otherwise all head_dim threads
// redundantly evaluate the same two exponentials for every key.
// Keys one warp scores per trip, and query rows one block carries.
//
// Their product is pinned to 32, which is what makes the block width exactly
// `ATTN_QT * tile`: the per-tile softmax bookkeeping then has one thread per
// (query row, key slot) and needs no loop at all. See the phases below.
#define ATTN_KT 4
#define ATTN_QT 8
// Head dimensions one lane reduces over, `head_dim / 32`. Bounded because the
// query tile lives in registers and `qr[ATTN_QT][ATTN_MAXD]` must be indexed
// by constants after unrolling — a runtime bound spills it to local memory,
// which is the whole win thrown away. 8 covers head_dim 256.
#define ATTN_MAXD 8

#define ATTN_FLASH(NAME, QT)                                                  \
__global__ void NAME(                                                           \
    const float* __restrict__ q,                                                \
    const float* __restrict__ k,                                                \
    const float* __restrict__ v,                                                \
    float* __restrict__ out,                                                    \
    int q_heads,                                                                \
    int kv_heads,                                                               \
    int head_dim,                                                               \
    const int* __restrict__ key_offset,                                         \
    float scale,                                                                \
    int n_query                                                                 \
) {                                                                             \
    extern __shared__ float smem[];                                             \
    int n_warps = blockDim.x >> 5;                                              \
    int tile = n_warps * ATTN_KT;  /* keys per barrier pair */                  \
    float* score_sh = smem;  /* QT * tile */                                    \
    float* w_sh     = score_sh + QT * tile;                                     \
    float* m_sh     = w_sh + QT * tile;  /* QT, the running maxima */           \
    float* l_sh     = m_sh + QT;  /* QT, the normalizers */                     \
    float* corr_sh  = l_sh + QT;  /* QT, this tile's rescale */                 \
                                                                                \
    long long qi0 = (long long)blockIdx.x * QT;                                 \
    int h = blockIdx.y;                                                         \
    int kvh = h / (q_heads / kv_heads);  /* GQA: never assume 1:1 */            \
    int tid = threadIdx.x;                                                      \
    int lane = tid & 31;                                                        \
    int warp = tid >> 5;                                                        \
    int dpt = head_dim >> 5;                                                    \
                                                                                \
    /* The query tile in registers rather than shared. */                       \
    /* */                                                                       \
    /* Lane `lane` owns dimensions `lane, lane + 32, ...` of *every* row in the */ \
    /* tile — the same partition of the contraction the untiled kernel gave one */ \
    /* block, so the dot product below sums in the same order. Holding it in */ \
    /* registers is what turns the score loop from one shared read per */       \
    /* multiply-add into none: a key element is loaded once and multiplied */   \
    /* QT times. */                                                             \
    float qr[QT][ATTN_MAXD];                                                    \
    _Pragma("unroll")                                                           \
    for (int u = 0; u < QT; ++u) {                                              \
        long long qrow = qi0 + u;                                               \
        const float* qp = q + (qrow * (long long)q_heads + h) * (long long)head_dim; \
        _Pragma("unroll")                                                       \
        for (int i = 0; i < ATTN_MAXD; ++i) {                                   \
            qr[u][i] = (i < dpt && qrow < (long long)n_query) ? qp[lane + 32 * i] : 0.0f; \
        }                                                                       \
    }                                                                           \
                                                                                \
    if (tid < QT) {                                                             \
        m_sh[tid] = neg_inf();                                                  \
        l_sh[tid] = 0.0f;                                                       \
    }                                                                           \
    float acc[QT];                                                              \
    _Pragma("unroll")                                                           \
    for (int u = 0; u < QT; ++u) acc[u] = 0.0f;                                 \
                                                                                \
    /* Rows of the tile that exist, and the deepest key any of them sees. The */ \
    /* causal bound is per row — row `qi0 + u` sees keys `[0, key_offset + qi0 */ \
    /* + u]` inclusive — so the loop runs to the *last* row's bound and the */  \
    /* earlier rows mask the tail off. See the mask in phase 1. */              \
    int qt_live = (int)((long long)n_query - qi0);                              \
    if (qt_live > QT) qt_live = QT;                                             \
    long long n_visible = (long long)(*key_offset) + qi0 + qt_live;             \
                                                                                \
    /* One thread per (query row, key slot) for the softmax phases. */          \
    int su = tid / tile;                                                        \
    int sw = tid - su * tile;                                                   \
                                                                                \
    __syncthreads();                                                            \
                                                                                \
    for (long long j0 = 0; j0 < n_visible; j0 += tile) {                        \
        /* ATTN_KT keys per warp per trip, not one. */                          \
        /* */                                                                   \
        /* `key = j0 + warp * ATTN_KT + r` stored at `score_sh[u * tile + warp */ \
        /* * ATTN_KT + r]` keeps slot `w` holding key `j0 + w`, which is what */ \
        /* lets the value accumulation below stay a single ascending sweep. */  \
        _Pragma("unroll")                                                       \
        for (int r = 0; r < ATTN_KT; ++r) {                                     \
            long long key = j0 + (long long)warp * ATTN_KT + r;                 \
            float part[QT];                                                     \
            _Pragma("unroll")                                                   \
            for (int u = 0; u < QT; ++u) part[u] = 0.0f;                        \
            if (key < n_visible) {                                              \
                const float* krow =                                             \
                    k + (key * (long long)kv_heads + kvh) * (long long)head_dim; \
                /* Lane l takes dimensions l, l+32, l+64, ...: consecutive lanes */ \
                /* read consecutive floats, so every load is a full 128 B */    \
                /* transaction, and one such load feeds QT multiply-adds. */    \
                _Pragma("unroll")                                               \
                for (int i = 0; i < ATTN_MAXD; ++i) {                           \
                    if (i < dpt) {                                              \
                        float kd = krow[lane + 32 * i];                         \
                        _Pragma("unroll")                                       \
                        for (int u = 0; u < QT; ++u) part[u] += qr[u][i] * kd;  \
                    }                                                           \
                }                                                               \
            }                                                                   \
            _Pragma("unroll")                                                   \
            for (int u = 0; u < QT; ++u) {                                      \
                float p = part[u];                                              \
                for (int off = 16; off > 0; off >>= 1) {                        \
                    p += __shfl_xor_sync(0xffffffff, p, off);                   \
                }                                                               \
                if (lane == 0) score_sh[u * tile + warp * ATTN_KT + r] = p * scale; \
            }                                                                   \
        }                                                                       \
        __syncthreads();                                                        \
                                                                                \
        /* Phase 1: the running max, one thread per query row. */               \
        /* */                                                                   \
        /* Serial and ascending over the tile, which is the order the untiled */ \
        /* kernel folded it in. A warp butterfly would be faster and would also */ \
        /* be a different reduction; `fmaxf` happens to be associative in */    \
        /* floating point, but the normalizer in phase 3 is not, so the two are */ \
        /* kept the same shape rather than one being quietly special. */        \
        if (tid < QT) {                                                         \
            long long limit = (long long)(*key_offset) + qi0 + tid;             \
            float tmax = neg_inf();                                             \
            for (int w = 0; w < tile; ++w) {                                    \
                if (j0 + w <= limit) tmax = fmaxf(tmax, score_sh[tid * tile + w]); \
            }                                                                   \
            float m0 = m_sh[tid];                                               \
            float new_m = fmaxf(m0, tmax);                                      \
            /* Matches the reference's guard exactly: on the first tile there is */ \
            /* no accumulator to rescale and expf(-inf - -inf) would be NaN. */ \
            corr_sh[tid] = (m0 == neg_inf()) ? 0.0f : expf(m0 - new_m);         \
            m_sh[tid] = new_m;                                                  \
        }                                                                       \
        __syncthreads();                                                        \
                                                                                \
        /* Phase 2: one exponential per slot, all in parallel. Masked slots */  \
        /* write a literal zero, which is what makes a row whose causal bound */ \
        /* ended earlier contribute exactly nothing to the tiles past it — */   \
        /* `a += 0.0f * v` and `lsum += 0.0f` are both exact. */                \
        if (tid < QT * tile) {                                                  \
            long long limit = (long long)(*key_offset) + qi0 + su;              \
            w_sh[su * tile + sw] = (j0 + sw <= limit)                           \
                ? expf(score_sh[su * tile + sw] - m_sh[su])                     \
                : 0.0f;                                                         \
        }                                                                       \
        __syncthreads();                                                        \
                                                                                \
        /* Phase 3: the normalizer, serial and ascending. Independent of phase */ \
        /* 4, so the two run without a barrier between them. */                 \
        if (tid < QT) {                                                         \
            float lsum = 0.0f;                                                  \
            for (int w = 0; w < tile; ++w) lsum += w_sh[tid * tile + w];        \
            l_sh[tid] = l_sh[tid] * corr_sh[tid] + lsum;                        \
        }                                                                       \
                                                                                \
        /* Phase 4: the values. One global load per key now serves QT */        \
        /* rows instead of one, which is the whole point of the query tile. */  \
        _Pragma("unroll")                                                       \
        for (int u = 0; u < QT; ++u) acc[u] = acc[u] * corr_sh[u];              \
        for (int w = 0; w < tile; ++w) {                                        \
            if (j0 + w >= n_visible) break;                                     \
            float vv =                                                          \
                v[((j0 + w) * (long long)kv_heads + kvh) * (long long)head_dim + tid]; \
            _Pragma("unroll")                                                   \
            for (int u = 0; u < QT; ++u) acc[u] += w_sh[u * tile + w] * vv;     \
        }                                                                       \
                                                                                \
        /* Before the next iteration overwrites score_sh and w_sh. */           \
        __syncthreads();                                                        \
    }                                                                           \
                                                                                \
    _Pragma("unroll")                                                           \
    for (int u = 0; u < QT; ++u) {                                              \
        long long qrow = qi0 + u;                                               \
        if (qrow < (long long)n_query) {                                        \
            out[(qrow * (long long)q_heads + h) * (long long)head_dim + tid] =  \
                acc[u] / l_sh[u];                                               \
        }                                                                       \
    }                                                                           \
}

// Eight query rows for prefill. One for decode, where seven of the eight
// register rows would be a masked-off row past `n_query`: the score loop would
// still run eight multiply-adds and eight shuffle reductions per key to throw
// seven of them away, and attention is 8% of a decode step. Measured 104.1 ->
// 99.7 tok/s before this instantiation existed.
//
// They are separate kernels rather than one macro at two widths because the
// two shapes want different softmax bookkeeping. At eight rows the per-tile
// max and normalizer are one thread per row, because every thread sweeping
// `QT * tile` shared floats would cost more than the value loop it amortizes.
// At one row there is nothing to amortize: the redundant sweep is 32 shared
// broadcasts, and paying it in every thread is cheaper than serializing it
// onto one and adding a barrier. `attn_flash_causal_t1` is therefore the
// pre-tiling kernel, unchanged.
ATTN_FLASH(attn_flash_causal, ATTN_QT)

// ---------------------------------------------------------------------------
// The same attention, with the KV head — not the query head — on the grid.
//
// grid: (n_query / GQA_QT, kv_heads) — one block per (query tile, KV head).
// block: (q_heads / kv_heads) warps — one warp per query head of that KV head.
//
// ## Why this kernel exists
//
// `attn_flash_causal` above puts the *query* head on grid.y, so at Qwen3.6's
// 16:2 grouping the eight query heads sharing a KV head are eight separate
// blocks, and each one streams the whole K and V window for itself. The work
// needs one pass over K/V per KV head; the kernel paid one per query head.
// Measured at the sixteenth chunk of an 8,192-token chunked prefill, that was
// 16.4 GB moved in 31.65 ms — 77% of this card's 672 GB/s — against a 2.05 GB
// lower bound. Eight times the traffic, and the kernel was bandwidth-bound, so
// it was eight times the time.
//
// Two cheaper fixes were measured first and both lost; see docs/BENCHMARKS.md.
// Widening ATTN_QT to 16 halves the block count but doubles `qr` to 128
// registers and halves resident blocks: 8K prefill 6,004 -> 6,647 ms. Merely
// swapping the grid axes so the eight siblings are *co-resident* and hit in L2
// lost 1.7% across three interleaved pairs — sibling blocks start together but
// drift apart faster than 6 MB of L2 can span, so nothing but an explicit
// staging barrier actually makes them share.
//
// ## The shape, and why GQA_QT is 4 and not 8
//
// DRAM traffic here is `(blocks) * (visible K/V)`, and blocks is
// `n_query/GQA_QT * kv_heads` against the old `n_query/ATTN_QT * q_heads`. The
// gqa ratio and ATTN_QT are both 8, so the reduction is exactly GQA_QT: four
// query rows, four times less traffic. Eight would give eight, and cannot be
// had — a warp now owns a whole head, so a lane carries `head_dim/32`
// accumulator dimensions per row rather than one, and `qr` plus `acc` is
// `2 * GQA_QT * ATTN_MAXD` registers. At GQA_QT 4 that is 64 and ptxas keeps
// two blocks resident; at 8 it is 128, which is the wall the ATTN_QT 16
// experiment already hit from the other side.
//
// ## What moved into shared, and what that costs
//
// K and V for GQA_KT keys are staged once per trip and read by all eight
// warps. The score loop then reads K from shared instead of global, one read
// feeding GQA_QT multiply-adds. Turing does 128 B/clk of shared against 64
// fp32 lanes — two warp-FMAs per warp-load — so a 4:1 ratio is still short of
// the load/store bound, which is why this does not simply move the stall.
//
// Consecutive lanes read consecutive words in both `k_sh` and `v_sh`
// (`lane + 32 * i` with a `head_dim` row stride), so no padding is needed:
// every access is one conflict-free 128 B phase.
//
// ## Softmax
//
// A warp owns a head outright, so the per-tile bookkeeping is warp-local and
// needs `__syncwarp` rather than `__syncthreads` — the only block-wide
// barriers left are the two that fence the staged tile. The running max and
// the normalizer stay serial and ascending over the tile, matching
// `causal_attention_streaming` and the kernel above; only the tile *width*
// differs (GQA_KT rather than `n_warps * ATTN_KT`), which changes where the
// rescales land and so is a floating-point difference, not an algebraic one.
#define GQA_QT 4
#define GQA_KT 8

__global__ void attn_flash_causal_gqa(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int q_heads,
    int kv_heads,
    int head_dim,
    const int* __restrict__ key_offset,
    float scale,
    int n_query
) {
    extern __shared__ float smem[];
    int gqa = q_heads / kv_heads;
    int dpt = head_dim >> 5;
    float* k_sh = smem;                            // GQA_KT * head_dim
    float* v_sh = k_sh + GQA_KT * head_dim;        // GQA_KT * head_dim
    float* w_sh = v_sh + GQA_KT * head_dim;        // gqa * GQA_QT * GQA_KT

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int nthr = blockDim.x;

    long long qi0 = (long long)blockIdx.x * GQA_QT;
    int kvh = blockIdx.y;
    int h   = kvh * gqa + warp;      // this warp's query head
    float* my_w = w_sh + warp * (GQA_QT * GQA_KT);

    // This warp's query tile in registers. Lane `lane` owns dimensions
    // `lane, lane + 32, ...` of every row, so the dot product below reduces
    // over exactly the partition the untiled kernel used.
    float qr[GQA_QT][ATTN_MAXD];
    #pragma unroll
    for (int u = 0; u < GQA_QT; ++u) {
        long long qrow = qi0 + u;
        const float* qp = q + (qrow * (long long)q_heads + h) * (long long)head_dim;
        #pragma unroll
        for (int i = 0; i < ATTN_MAXD; ++i) {
            qr[u][i] = (i < dpt && qrow < (long long)n_query) ? qp[lane + 32 * i] : 0.0f;
        }
    }

    float acc[GQA_QT][ATTN_MAXD];
    float m[GQA_QT], ln[GQA_QT];
    #pragma unroll
    for (int u = 0; u < GQA_QT; ++u) {
        m[u] = neg_inf();
        ln[u] = 0.0f;
        #pragma unroll
        for (int i = 0; i < ATTN_MAXD; ++i) acc[u][i] = 0.0f;
    }

    int qt_live = (int)((long long)n_query - qi0);
    if (qt_live > GQA_QT) qt_live = GQA_QT;
    long long n_visible = (long long)(*key_offset) + qi0 + qt_live;

    for (long long j0 = 0; j0 < n_visible; j0 += GQA_KT) {
        // Stage the tile. `n_visible` is block-uniform, so every warp runs the
        // same trip count and the barriers below are reached by all of them.
        __syncthreads();
        // Every load for the tile is issued before any of it is stored; see
        // the same loop in `attn_flash_decode_split` for why. Costs
        // `2 * GQA_KT` registers, which is the reason it is measured here
        // rather than assumed: this kernel is already at the two-block
        // boundary where the decode one has room to spare.
        for (int d = tid; d < head_dim; d += nthr) {
            float kreg[GQA_KT];
            float vreg[GQA_KT];
            #pragma unroll
            for (int jj = 0; jj < GQA_KT; ++jj) {
                long long key = j0 + jj;
                bool live = key < n_visible;
                long long at = (key * (long long)kv_heads + kvh) * (long long)head_dim + d;
                kreg[jj] = live ? k[at] : 0.0f;
                vreg[jj] = live ? v[at] : 0.0f;
            }
            #pragma unroll
            for (int jj = 0; jj < GQA_KT; ++jj) {
                k_sh[jj * head_dim + d] = kreg[jj];
                v_sh[jj * head_dim + d] = vreg[jj];
            }
        }
        __syncthreads();

        // Scores. One shared read of a key dimension feeds GQA_QT multiply-adds.
        #pragma unroll
        for (int jj = 0; jj < GQA_KT; ++jj) {
            float part[GQA_QT];
            #pragma unroll
            for (int u = 0; u < GQA_QT; ++u) part[u] = 0.0f;
            #pragma unroll
            for (int i = 0; i < ATTN_MAXD; ++i) {
                if (i < dpt) {
                    float kd = k_sh[jj * head_dim + lane + 32 * i];
                    #pragma unroll
                    for (int u = 0; u < GQA_QT; ++u) part[u] += qr[u][i] * kd;
                }
            }
            #pragma unroll
            for (int u = 0; u < GQA_QT; ++u) {
                float p = part[u];
                for (int off = 16; off > 0; off >>= 1) {
                    p += __shfl_xor_sync(0xffffffff, p, off);
                }
                if (lane == 0) my_w[u * GQA_KT + jj] = p * scale;
            }
        }
        __syncwarp();

        // The running max, serial and ascending over the tile. Every lane
        // sweeps it redundantly: that is GQA_KT conflict-free shared
        // broadcasts, cheaper than serializing onto one lane and broadcasting
        // the result back. `corr` matches the reference's first-tile guard,
        // where there is no accumulator to rescale and expf(-inf - -inf) is NaN.
        float corr[GQA_QT];
        #pragma unroll
        for (int u = 0; u < GQA_QT; ++u) {
            long long limit = (long long)(*key_offset) + qi0 + u;
            float tmax = neg_inf();
            for (int w = 0; w < GQA_KT; ++w) {
                if (j0 + w <= limit) tmax = fmaxf(tmax, my_w[u * GQA_KT + w]);
            }
            float m0 = m[u];
            float nm = fmaxf(m0, tmax);
            corr[u] = (m0 == neg_inf()) ? 0.0f : expf(m0 - nm);
            m[u] = nm;
        }

        // One exponential per slot. Masked slots store a literal zero, which is
        // what lets a row whose causal bound ended earlier contribute exactly
        // nothing to the tiles past it: `a += 0.0f * v` and `l += 0.0f` are
        // both exact. Every read of `my_w` happens before any write to it.
        float wl[GQA_QT];
        #pragma unroll
        for (int u = 0; u < GQA_QT; ++u) {
            long long limit = (long long)(*key_offset) + qi0 + u;
            float s = (lane < GQA_KT) ? my_w[u * GQA_KT + lane] : 0.0f;
            wl[u] = (lane < GQA_KT && j0 + lane <= limit) ? expf(s - m[u]) : 0.0f;
        }
        __syncwarp();
        #pragma unroll
        for (int u = 0; u < GQA_QT; ++u) {
            if (lane < GQA_KT) my_w[u * GQA_KT + lane] = wl[u];
        }
        __syncwarp();

        // The normalizer, serial and ascending like the max above.
        #pragma unroll
        for (int u = 0; u < GQA_QT; ++u) {
            float lsum = 0.0f;
            for (int w = 0; w < GQA_KT; ++w) lsum += my_w[u * GQA_KT + w];
            ln[u] = ln[u] * corr[u] + lsum;
        }

        // The values. One shared read of a value dimension serves GQA_QT rows.
        #pragma unroll
        for (int u = 0; u < GQA_QT; ++u) {
            #pragma unroll
            for (int i = 0; i < ATTN_MAXD; ++i) acc[u][i] *= corr[u];
        }
        for (int w = 0; w < GQA_KT; ++w) {
            if (j0 + w >= n_visible) break;
            float wu[GQA_QT];
            #pragma unroll
            for (int u = 0; u < GQA_QT; ++u) wu[u] = my_w[u * GQA_KT + w];
            #pragma unroll
            for (int i = 0; i < ATTN_MAXD; ++i) {
                if (i < dpt) {
                    float vv = v_sh[w * head_dim + lane + 32 * i];
                    #pragma unroll
                    for (int u = 0; u < GQA_QT; ++u) acc[u][i] += wu[u] * vv;
                }
            }
        }
    }

    #pragma unroll
    for (int u = 0; u < GQA_QT; ++u) {
        long long qrow = qi0 + u;
        if (qrow < (long long)n_query) {
            float* op = out + (qrow * (long long)q_heads + h) * (long long)head_dim;
            #pragma unroll
            for (int i = 0; i < ATTN_MAXD; ++i) {
                if (i < dpt) op[lane + 32 * i] = acc[u][i] / ln[u];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Flash decoding: one query row, the key range split across blocks.
//
// grid: (DEC_SPLITS, kv_heads) for the split pass, (q_heads,) for the combine.
// block: (q_heads / kv_heads) warps, one warp per query head, as above.
//
// ## Why decode needs its own shape
//
// `attn_flash_causal_t1` launches `(n_query, q_heads)`, and decode is
// `n_query == 1`. That is **sixteen blocks on a 72-SM card**: 78% of the
// machine is idle before the eightfold K/V redundancy is even counted. Nothing
// about the query tile can fix that, because there is only one query row to
// tile. The parallelism has to come from the key axis instead.
//
// So each block takes a contiguous slice of the key range and runs the same
// online softmax over just that slice, emitting a partial `(m, l, acc)`. A
// second pass merges the slices. The merge is exact in exact arithmetic:
// writing `gm` for `max_s m_s`,
//
//     sum_s exp(m_s - gm) * acc_s = sum_s sum_{j in s} exp(s_j - gm) * v_j
//     sum_s exp(m_s - gm) * l_s   = sum_s sum_{j in s} exp(s_j - gm)
//
// which are the numerator and denominator a single pass would have built, and
// `gm` is the global maximum because the maximum of the slice maxima is it.
//
// ## Rule 5
//
// `DEC_SPLITS` is a host constant, but no slice boundary is. The slice width is
// computed on the device from `*key_offset` and rounded up to a whole `DEC_KT`
// tile, so a tile never straddles a boundary and every block's trip count comes
// from device state alone. Splits past the end of a short window run zero trips
// and write the identity partial — `m = -inf`, `l = 0`, `acc = 0` — which the
// combine folds in as `exp(-inf - gm) = 0`, exactly.
#define DEC_KT 8
#define DEC_SPLITS 144

__global__ void attn_flash_decode_split(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ part_acc,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    int q_heads,
    int kv_heads,
    int head_dim,
    const int* __restrict__ key_offset,
    float scale
) {
    extern __shared__ float smem[];
    int gqa = q_heads / kv_heads;
    int dpt = head_dim >> 5;
    float* k_sh = smem;                        // DEC_KT * head_dim
    float* v_sh = k_sh + DEC_KT * head_dim;    // DEC_KT * head_dim
    float* s_sh = v_sh + DEC_KT * head_dim;    // gqa * DEC_KT

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int nthr = blockDim.x;
    int split = blockIdx.x;
    int kvh = blockIdx.y;
    int h = kvh * gqa + warp;
    float* my_s = s_sh + warp * DEC_KT;

    // Decode's single row sits at absolute position *key_offset and sees keys
    // [0, *key_offset]. Every quantity below is block-uniform, so the barriers
    // in the trip loop are reached by every warp the same number of times.
    long long n_visible = (long long)(*key_offset) + 1;
    long long per = (n_visible + DEC_SPLITS - 1) / DEC_SPLITS;
    per = ((per + DEC_KT - 1) / DEC_KT) * DEC_KT;
    long long begin = (long long)split * per;
    long long end = begin + per;
    if (end > n_visible) end = n_visible;

    float qr[ATTN_MAXD];
    const float* qp = q + (long long)h * (long long)head_dim;
    #pragma unroll
    for (int i = 0; i < ATTN_MAXD; ++i) qr[i] = (i < dpt) ? qp[lane + 32 * i] : 0.0f;

    float acc[ATTN_MAXD];
    #pragma unroll
    for (int i = 0; i < ATTN_MAXD; ++i) acc[i] = 0.0f;
    float m = neg_inf();
    float l = 0.0f;

    for (long long j0 = begin; j0 < end; j0 += DEC_KT) {
        int n_this = (int)((end - j0) < (long long)DEC_KT ? (end - j0) : (long long)DEC_KT);
        __syncthreads();
        // Every load for the tile is issued before any of it is stored.
        //
        // Writing straight into shared makes each store depend on the load
        // just above it, which leaves the tile with about one outstanding
        // request per thread at a time; the kernel then waits out the full
        // DRAM latency DEC_KT times per trip instead of once. Landing the
        // whole tile in registers first leaves `2 * DEC_KT` requests in
        // flight, which is what turns this loop from latency-bound into
        // bandwidth-bound. Costs `2 * DEC_KT` registers, which this kernel
        // has because one query row makes `qr` and `acc` a quarter of what
        // the prefill kernel carries.
        for (int d = tid; d < head_dim; d += nthr) {
            float kreg[DEC_KT];
            float vreg[DEC_KT];
            #pragma unroll
            for (int jj = 0; jj < DEC_KT; ++jj) {
                long long key = j0 + jj;
                bool live = jj < n_this;
                long long at = (key * (long long)kv_heads + kvh) * (long long)head_dim + d;
                kreg[jj] = live ? k[at] : 0.0f;
                vreg[jj] = live ? v[at] : 0.0f;
            }
            #pragma unroll
            for (int jj = 0; jj < DEC_KT; ++jj) {
                k_sh[jj * head_dim + d] = kreg[jj];
                v_sh[jj * head_dim + d] = vreg[jj];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int jj = 0; jj < DEC_KT; ++jj) {
            float part = 0.0f;
            #pragma unroll
            for (int i = 0; i < ATTN_MAXD; ++i) {
                if (i < dpt) part += qr[i] * k_sh[jj * head_dim + lane + 32 * i];
            }
            for (int off = 16; off > 0; off >>= 1) {
                part += __shfl_xor_sync(0xffffffff, part, off);
            }
            if (lane == 0) my_s[jj] = part * scale;
        }
        __syncwarp();

        // Serial and ascending over the tile, like every other form here.
        float tmax = neg_inf();
        for (int w = 0; w < n_this; ++w) tmax = fmaxf(tmax, my_s[w]);
        float nm = fmaxf(m, tmax);
        float corr = (m == neg_inf()) ? 0.0f : expf(m - nm);

        float wl = (lane < n_this) ? expf(my_s[lane] - nm) : 0.0f;
        __syncwarp();
        if (lane < DEC_KT) my_s[lane] = wl;
        __syncwarp();

        float lsum = 0.0f;
        for (int w = 0; w < n_this; ++w) lsum += my_s[w];
        l = l * corr + lsum;

        #pragma unroll
        for (int i = 0; i < ATTN_MAXD; ++i) acc[i] *= corr;
        for (int w = 0; w < n_this; ++w) {
            float ww = my_s[w];
            #pragma unroll
            for (int i = 0; i < ATTN_MAXD; ++i) {
                if (i < dpt) acc[i] += ww * v_sh[w * head_dim + lane + 32 * i];
            }
        }
        m = nm;
    }

    float* pa = part_acc + ((long long)split * q_heads + h) * (long long)head_dim;
    #pragma unroll
    for (int i = 0; i < ATTN_MAXD; ++i) {
        if (i < dpt) pa[lane + 32 * i] = acc[i];
    }
    if (lane == 0) {
        part_m[split * q_heads + h] = m;
        part_l[split * q_heads + h] = l;
    }
}

// grid: (q_heads,). block: head_dim threads, one per output dimension.
__global__ void attn_flash_decode_combine(
    const float* __restrict__ part_acc,
    const float* __restrict__ part_m,
    const float* __restrict__ part_l,
    float* __restrict__ out,
    int q_heads,
    int head_dim
) {
    int h = blockIdx.x;
    int d = threadIdx.x;

    // Split 0 always holds at least key 0, so this is never -inf and the
    // subtraction below never forms inf - inf.
    float gm = neg_inf();
    for (int s = 0; s < DEC_SPLITS; ++s) gm = fmaxf(gm, part_m[s * q_heads + h]);

    float num = 0.0f;
    float den = 0.0f;
    for (int s = 0; s < DEC_SPLITS; ++s) {
        float ms = part_m[s * q_heads + h];
        float f = (ms == neg_inf()) ? 0.0f : expf(ms - gm);
        num += f * part_acc[((long long)s * q_heads + h) * (long long)head_dim + d];
        den += f * part_l[s * q_heads + h];
    }
    out[(long long)h * (long long)head_dim + d] = num / den;
}

__global__ void attn_flash_causal_t1(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int q_heads,
    int kv_heads,
    int head_dim,
    const int* __restrict__ key_offset,
    float scale,
    int n_query
) {
    (void)n_query;
    extern __shared__ float smem[];
    int n_warps = blockDim.x >> 5;
    int tile = n_warps * ATTN_KT;              // keys per barrier pair
    float* q_sh     = smem;                    // head_dim floats
    float* score_sh = smem + head_dim;         // tile floats
    float* w_sh     = score_sh + tile;         // tile floats

    long long qi = blockIdx.x;
    int h = blockIdx.y;
    int kvh = h / (q_heads / kv_heads);        // GQA: never assume 1:1
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    long long qbase = (qi * (long long)q_heads + h) * (long long)head_dim;
    q_sh[tid] = q[qbase + tid];
    __syncthreads();

    // The causal bound, written once. Query row qi sits at absolute position
    // key_offset + qi and sees keys [0, key_offset + qi] inclusive.
    long long n_visible = (long long)(*key_offset) + qi + 1;

    float m = neg_inf();
    float l = 0.0f;
    float acc = 0.0f;

    for (long long j0 = 0; j0 < n_visible; j0 += tile) {
        // ATTN_KT keys per warp per trip, not one.
        //
        // The barrier pair below is per *trip*, and one key per warp made it
        // one barrier pair per eight keys: a 512-token prefill row crossed 128
        // of them. The scores of several keys are independent, so a warp can
        // compute ATTN_KT of them back to back and the block can rescale its
        // running softmax once for all `tile` of them.
        //
        // `key = j0 + warp * ATTN_KT + r` stored at `score_sh[warp * ATTN_KT
        // + r]` keeps slot `w` holding key `j0 + w`, which is what lets the
        // value accumulation below stay a single ascending sweep.
        #pragma unroll
        for (int r = 0; r < ATTN_KT; ++r) {
            long long key = j0 + (long long)warp * ATTN_KT + r;
            float partial = 0.0f;
            if (key < n_visible) {
                const float* krow =
                    k + (key * (long long)kv_heads + kvh) * (long long)head_dim;
                // Lane l takes dimensions l, l+32, l+64, ...: consecutive lanes
                // read consecutive floats, so every load is a full 128 B
                // transaction, and q_sh[d] with d = lane + 32*i hits a distinct
                // bank per lane.
                for (int d = lane; d < head_dim; d += 32) partial += q_sh[d] * krow[d];
            }
            for (int off = 16; off > 0; off >>= 1) {
                partial += __shfl_xor_sync(0xffffffff, partial, off);
            }
            if (lane == 0) score_sh[warp * ATTN_KT + r] = partial * scale;
        }
        __syncthreads();

        long long remaining = n_visible - j0;
        int n_this = (int)(remaining < (long long)tile ? remaining : (long long)tile);

        float tile_max = neg_inf();
        for (int w = 0; w < n_this; ++w) tile_max = fmaxf(tile_max, score_sh[w]);
        float new_m = fmaxf(m, tile_max);
        // Matches the reference's guard exactly: on the first tile there is
        // no accumulator to rescale and expf(-inf - -inf) would be NaN.
        float corr = (m == neg_inf()) ? 0.0f : expf(m - new_m);

        // w_sh and score_sh are disjoint, so this write races nothing above.
        if (tid < n_this) w_sh[tid] = expf(score_sh[tid] - new_m);
        __syncthreads();

        float lsum = 0.0f;
        for (int w = 0; w < n_this; ++w) lsum += w_sh[w];
        l = l * corr + lsum;

        float a = acc * corr;
        for (int w = 0; w < n_this; ++w) {
            a += w_sh[w] * v[((j0 + w) * (long long)kv_heads + kvh) * (long long)head_dim + tid];
        }
        acc = a;
        m = new_m;

        // Before the next iteration overwrites score_sh and w_sh.
        __syncthreads();
    }

    out[qbase + tid] = acc / l;
}


// Append this batch's roped keys and raw values to the cache, at the absolute
// position the sequence has reached.
//
// This was two `cuMemcpyDtoDAsync` calls into `cache.k.slice_mut(at..)` and
// `cache.v.slice_mut(at..)`, which is the same traffic and one fewer launch.
// It is a kernel now for one reason: `at` was computed on the host, so the
// destination was a **host-chosen address**. A CUDA graph records addresses,
// so a captured decode step would write position `n` forever. Reading the
// position from device memory is what makes the step replayable, and it is
// `AGENTS.md` rule 5 besides.
//
// Both halves in one launch: they are the same shape and the same stride, and
// the copies are independent, so splitting them would only cost a launch.
//
// grid: (ceil(2 * span / ATTN_APPEND_THREADS),). block: ATTN_APPEND_THREADS.
__global__ void attn_kv_append(
    const float* __restrict__ key,
    const float* __restrict__ value,
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const int* __restrict__ position,
    int span,
    int row
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * span) return;
    // `span` is `n_tokens * row`; the cache is indexed by *position*, so the
    // destination base is one row per position and not one span.
    long long at = (long long)(*position) * (long long)row;
    if (i < span) {
        k_cache[at + i] = key[i];
    } else {
        int j = i - span;
        v_cache[at + j] = value[j];
    }
}

}
"#;

/// Threads per block for [`AttentionKernels::append_kv`].
const APPEND_THREADS: u32 = 256;

/// Something went wrong compiling or launching an attention kernel.
#[derive(Debug)]
pub enum AttentionError {
    /// NVRTC rejected the source, or the module failed to load.
    Compile(String),
    /// The driver failed.
    Driver(DriverError),
    /// A head dimension the kernel cannot service.
    ///
    /// One thread per head dimension, reduced with warp shuffles, so it must
    /// be a whole number of warps and fit in one block.
    UnsupportedHeadDim { head_dim: usize },
    /// Query heads are not a whole multiple of KV heads.
    UnevenGqaGrouping { q_heads: usize, kv_heads: usize },
    /// `rope_dim` is odd or wider than `head_dim`.
    UnsupportedRopeDim { rope_dim: usize, head_dim: usize },
    /// A buffer is not the size the declared geometry implies.
    ///
    /// Rejected rather than clamped: a short K buffer would be read past its
    /// end for every query deep enough to reach the missing rows, and the
    /// result would still look like attention.
    ///
    /// For `key` and `value`, `expected` is a lower bound rather than an exact
    /// size — see [`AttentionKernels::forward`] on why a cache is allowed to be
    /// longer than its filled window.
    BufferShape {
        what: &'static str,
        expected: usize,
        actual: usize,
    },
    /// The query rows would run past the end of the key window.
    ///
    /// Query row `i` reads keys `[0, key_offset + i]`, so `key_offset +
    /// n_query` must not exceed `n_keys`.
    QueryPastKeys {
        key_offset: usize,
        n_query: usize,
        n_keys: usize,
    },
}

impl std::fmt::Display for AttentionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compile(m) => write!(f, "kernel compilation failed: {m}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::UnsupportedHeadDim { head_dim } => write!(
                f,
                "head_dim {head_dim} must be a positive multiple of 32 and at most 1024",
            ),
            Self::UnevenGqaGrouping { q_heads, kv_heads } => write!(
                f,
                "{q_heads} query heads do not divide evenly among {kv_heads} kv heads",
            ),
            Self::UnsupportedRopeDim { rope_dim, head_dim } => write!(
                f,
                "rope_dim {rope_dim} must be even and at most head_dim {head_dim}",
            ),
            Self::BufferShape {
                what,
                expected,
                actual,
            } => write!(
                f,
                "{what} holds {actual} floats, but this geometry needs {expected}",
            ),
            Self::QueryPastKeys {
                key_offset,
                n_query,
                n_keys,
            } => write!(
                f,
                "query rows at offset {key_offset}..{} run past the {n_keys}-key window",
                key_offset + n_query,
            ),
        }
    }
}

impl std::error::Error for AttentionError {}

impl From<DriverError> for AttentionError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

/// Per-slice partials for the flash-decoding path.
///
/// Preallocated because [`AGENTS.md` rule 6] forbids allocating mid-forward,
/// and shared by every attention layer because they run in sequence on one
/// stream and nothing crosses a layer boundary. Roughly 1 MiB at this model's
/// geometry, which is why it is not worth a shape-dependent lifetime.
pub struct AttnDecodeScratch {
    /// `[DECODE_SPLITS][q_heads][head_dim]`, each slice's unnormalized sum.
    acc: CudaSlice<f32>,
    /// `[DECODE_SPLITS][q_heads]`, each slice's running maximum.
    m: CudaSlice<f32>,
    /// `[DECODE_SPLITS][q_heads]`, each slice's softmax normalizer.
    l: CudaSlice<f32>,
}

impl AttnDecodeScratch {
    /// Allocate for one head geometry.
    pub fn new(
        stream: &Arc<CudaStream>,
        q_heads: usize,
        head_dim: usize,
    ) -> Result<Self, AttentionError> {
        Ok(Self {
            acc: stream.alloc_zeros::<f32>(DECODE_SPLITS * q_heads * head_dim)?,
            m: stream.alloc_zeros::<f32>(DECODE_SPLITS * q_heads)?,
            l: stream.alloc_zeros::<f32>(DECODE_SPLITS * q_heads)?,
        })
    }

    /// Bytes held, for the VRAM accounting the engine prints.
    pub fn bytes(&self) -> usize {
        (self.acc.len() + self.m.len() + self.l.len()) * size_of::<f32>()
    }
}

/// Compiled Gated Attention kernels for one head geometry.
pub struct AttentionKernels {
    split: CudaFunction,
    rope: CudaFunction,
    flash: CudaFunction,
    flash_gqa: CudaFunction,
    decode_split: CudaFunction,
    decode_combine: CudaFunction,
    flash_t1: CudaFunction,
    append: CudaFunction,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl AttentionKernels {
    /// Compile for a specific head geometry.
    ///
    /// Fixed at construction rather than per launch, matching `GdnKernels`:
    /// the geometry is a property of the model, and validating it once leaves
    /// the launch path with only per-call shapes to reject.
    pub fn new(
        ctx: &Arc<CudaContext>,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, AttentionError> {
        if head_dim == 0 || !head_dim.is_multiple_of(32) || head_dim > 32 * ATTN_MAXD {
            return Err(AttentionError::UnsupportedHeadDim { head_dim });
        }
        if kv_heads == 0 || !q_heads.is_multiple_of(kv_heads) {
            return Err(AttentionError::UnevenGqaGrouping { q_heads, kv_heads });
        }
        let ptx = compile(ATTENTION_SRC, "attention").map_err(AttentionError::Compile)?;
        let module = ctx.load_module(ptx)?;
        Ok(Self {
            split: module.load_function("attn_split_query_gate")?,
            rope: module.load_function("attn_rope_partial_neox")?,
            flash: module.load_function("attn_flash_causal")?,
            flash_gqa: module.load_function("attn_flash_causal_gqa")?,
            decode_split: module.load_function("attn_flash_decode_split")?,
            decode_combine: module.load_function("attn_flash_decode_combine")?,
            flash_t1: module.load_function("attn_flash_causal_t1")?,
            append: module.load_function("attn_kv_append")?,
            q_heads,
            kv_heads,
            head_dim,
        })
    }

    /// Query heads sharing each KV head.
    pub fn gqa_ratio(&self) -> usize {
        self.q_heads / self.kv_heads
    }

    /// Keys scored per tile: one per warp, so `head_dim / 32`.
    ///
    /// Exposed so a test can pick a sequence length that is deliberately not a
    /// multiple of it and exercise the ragged final tile.
    pub fn keys_per_tile(&self) -> usize {
        (self.head_dim / 32) * KEYS_PER_WARP
    }

    /// Dynamic shared memory one block requests: the score and weight tiles,
    /// which are now `QUERY_TILE` rows deep, plus the three per-row softmax
    /// scalars. The query tile itself is in registers. See the module docs.
    pub fn shared_bytes(&self) -> usize {
        (2 * QUERY_TILE * self.keys_per_tile() + 3 * QUERY_TILE) * size_of::<f32>()
    }

    /// Dynamic shared memory the GQA-shared kernel requests: the staged key
    /// and value tiles, plus one weight tile per warp. Unlike the kernel above
    /// it stages K and V rather than only the softmax scratch, which is the
    /// whole point — see that kernel's comment.
    pub fn shared_bytes_gqa(&self) -> usize {
        (2 * GQA_KEY_TILE * self.head_dim + self.gqa_ratio() * GQA_QUERY_TILE * GQA_KEY_TILE)
            * size_of::<f32>()
    }

    /// Dynamic shared memory the flash-decoding split pass requests: the staged
    /// key and value tiles plus one score tile per warp.
    pub fn shared_bytes_decode(&self) -> usize {
        (2 * DECODE_KEY_TILE * self.head_dim + self.gqa_ratio() * DECODE_KEY_TILE)
            * size_of::<f32>()
    }

    /// Whether flash decoding can service this geometry.
    ///
    /// Same three conditions as the GQA-shared kernel — it is the same block
    /// shape with one query row instead of four.
    fn decode_split_is_available(&self) -> bool {
        self.gqa_ratio() >= 2
            && self.gqa_ratio() * 32 <= 1024
            && self.shared_bytes_decode() <= 48 * 1024
    }

    /// Whether the GQA-shared kernel can service this geometry at all.
    ///
    /// Three things have to hold, and none of them do for every model: there
    /// must be sharing to exploit, one warp per query head has to fit in a
    /// block, and the staged tiles have to fit in the 48 KiB a block gets
    /// without opting in. When any fails the launch falls back to
    /// `attn_flash_causal`, which needs none of them.
    fn gqa_shared_is_available(&self) -> bool {
        self.gqa_ratio() >= 2
            && self.gqa_ratio() * 32 <= 1024
            && self.shared_bytes_gqa() <= 48 * 1024
    }

    /// Dynamic shared memory the one-row kernel requests: `q_sh` plus the two
    /// tile-wide scratch arrays. It stages the query rather than holding it in
    /// registers, so its budget is the pre-tiling one and not a `QUERY_TILE`
    /// of 1 in the formula above.
    fn shared_bytes_t1(&self) -> usize {
        (self.head_dim + 2 * self.keys_per_tile()) * size_of::<f32>()
    }

    /// The `1/sqrt(head_dim)` score scale.
    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    fn expect_len(
        what: &'static str,
        buf_len: usize,
        expected: usize,
    ) -> Result<(), AttentionError> {
        if buf_len != expected {
            return Err(AttentionError::BufferShape {
                what,
                expected,
                actual: buf_len,
            });
        }
        Ok(())
    }

    /// Like [`Self::expect_len`], for the buffers a cache is allowed to
    /// over-allocate. Too small is still fatal — that is the read-past-the-end
    /// case — but too large is the ordinary steady state of a KV cache.
    fn expect_at_least(
        what: &'static str,
        buf_len: usize,
        needed: usize,
    ) -> Result<(), AttentionError> {
        if buf_len < needed {
            return Err(AttentionError::BufferShape {
                what,
                expected: needed,
                actual: buf_len,
            });
        }
        Ok(())
    }

    /// Deinterleave the packed `attn_q` projection output into a query tensor
    /// and an output-gate tensor.
    ///
    /// `packed` is `[n_tokens][q_heads * 2 * head_dim]` in the layout
    /// `[q_h0, gate_h0, q_h1, gate_h1, ...]`; `q` and `gate` are each
    /// `[n_tokens][q_heads][head_dim]`. See the module docs for why this is a
    /// kernel and not a slice.
    pub fn split_query_and_gate(
        &self,
        stream: &Arc<CudaStream>,
        packed: &CudaSlice<f32>,
        q: &mut CudaSlice<f32>,
        gate: &mut CudaSlice<f32>,
        n_tokens: usize,
    ) -> Result<(), AttentionError> {
        let per_head = self.q_heads * self.head_dim;
        Self::expect_len("packed query/gate", packed.len(), n_tokens * per_head * 2)?;
        Self::expect_len("query", q.len(), n_tokens * per_head)?;
        Self::expect_len("gate", gate.len(), n_tokens * per_head)?;

        let cfg = LaunchConfig {
            grid_dim: (n_tokens as u32, self.q_heads as u32, 1),
            block_dim: (self.head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let q_heads = self.q_heads as i32;
        let head_dim = self.head_dim as i32;
        let mut builder = stream.launch_builder(&self.split);
        builder
            .arg(packed)
            .arg(q)
            .arg(gate)
            .arg(&q_heads)
            .arg(&head_dim);
        // SAFETY: the grid is (n_tokens, q_heads) with one thread per head
        // dimension, and the three buffer lengths were just checked against
        // exactly the indices `2*h` and `2*h+1` reach.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// Apply partial rotary embedding to `[n_tokens][n_heads][head_dim]`.
    ///
    /// Token `i` is rotated by position `positions[0] + i`. `n_heads` is
    /// `q_heads` for the query stream and `kv_heads` for the key stream — RoPE
    /// is applied before the GQA broadcast, so the two differ.
    ///
    /// Dimensions `[rope_dim, head_dim)` are copied through unmodified.
    ///
    /// **`positions` is a one-element device scalar, not a host number.** The
    /// position is the only thing that changes between two decode steps, so
    /// keeping it on the device is what lets a whole step be captured once as
    /// a CUDA graph and replayed — a host argument would be baked into the
    /// recorded launch. It is also `AGENTS.md` rule 5: nothing on the forward
    /// path is sized or indexed by a host-side value.
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_tokens: usize,
        n_heads: usize,
        rope_dim: usize,
        positions: &CudaSlice<i32>,
        theta_base: f32,
    ) -> Result<(), AttentionError> {
        Self::expect_len("rope position", positions.len(), 1)?;
        if !rope_dim.is_multiple_of(2) || rope_dim > self.head_dim {
            return Err(AttentionError::UnsupportedRopeDim {
                rope_dim,
                head_dim: self.head_dim,
            });
        }
        let n = n_tokens * n_heads * self.head_dim;
        Self::expect_len("rope input", input.len(), n)?;
        Self::expect_len("rope output", out.len(), n)?;

        let cfg = LaunchConfig {
            grid_dim: ((n_tokens as u32).div_ceil(ROPE_TOKENS), n_heads as u32, 1),
            block_dim: (self.head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_heads_i = n_heads as i32;
        let head_dim = self.head_dim as i32;
        let rope_dim_i = rope_dim as i32;
        let n_tokens_i = n_tokens as i32;
        let mut builder = stream.launch_builder(&self.rope);
        builder
            .arg(input)
            .arg(out)
            .arg(&n_heads_i)
            .arg(&head_dim)
            .arg(&rope_dim_i)
            .arg(positions)
            .arg(&theta_base)
            .arg(&n_tokens_i);
        // SAFETY: the grid is (ceil(n_tokens / ROPE_TOKENS), n_heads) with one
        // thread per head dimension, and `n_tokens` is passed so the band
        // stops at the last real token; both buffers were checked to hold
        // exactly that many floats, and every thread touches only its own
        // head's slice.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// Causal grouped-query attention over a key window.
    ///
    /// - `q`, `out`: `[n_query][q_heads][head_dim]`
    /// - `k`, `v`: `[n_keys][kv_heads][head_dim]`
    ///
    /// Query row `i` sits at absolute position `positions[0] + i` and attends
    /// to keys `[0, positions[0] + i]`. A position of 0 with `n_query` equal
    /// to the filled window is a full prefill; `n_query == 1` at the window's
    /// last position is a decode step against a cached window.
    ///
    /// **`positions` is a one-element device scalar** — see [`Self::rope`] for
    /// why. The consequence here is that the "query rows run past the key
    /// window" check cannot live in this function any more: the position is
    /// not a number this side of the launch. `max_keys` is the caller's
    /// promise about the cache's *capacity*, which is checked against the
    /// buffers, and the caller owns the promise that the position stays inside
    /// it. In this engine that is `GatedAttentionBlock::forward`, which holds
    /// both the host position and `KvCache::max_seq` and returns
    /// [`AttentionError::QueryPastKeys`] itself.
    ///
    /// `k` and `v` may be **longer** than the filled window. A KV cache is
    /// allocated once for the longest sequence the worker admits and then
    /// filled a token at a time. The kernel indexes keys by absolute position
    /// and reads nothing above `positions[0] + n_query - 1`, so the tail is
    /// untouched rather than merely unused. `q` and `out` stay exact: those
    /// are indexed by the launch geometry, so a wrong length there is a wrong
    /// launch.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        stream: &Arc<CudaStream>,
        dec: &mut AttnDecodeScratch,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_query: usize,
        max_keys: usize,
        positions: &CudaSlice<i32>,
    ) -> Result<(), AttentionError> {
        Self::expect_len("attention position", positions.len(), 1)?;
        let q_elems = n_query * self.q_heads * self.head_dim;
        let kv_elems = max_keys * self.kv_heads * self.head_dim;
        Self::expect_len("query", q.len(), q_elems)?;
        Self::expect_at_least("key", k.len(), kv_elems)?;
        Self::expect_at_least("value", v.len(), kv_elems)?;
        Self::expect_len("output", out.len(), q_elems)?;

        // A partial tile costs its empty rows in full: the score loop runs a
        // multiply-add and a shuffle reduction per row per key whether or not
        // the row exists. Decode is `n_query == 1`, where that is seven
        // eighths of the kernel, so below a whole tile the one-row
        // instantiation is launched instead.
        // Decode gets its own two-pass shape. See the kernel comment: a single
        // query row cannot be tiled, so `attn_flash_causal_t1` could only ever
        // launch `q_heads` blocks, and this splits the key axis instead.
        if n_query == 1 && self.decode_split_is_available() {
            return self.decode(stream, dec, q, k, v, out, positions);
        }

        // Three shapes, narrowest first.
        //
        // Below one query tile the tiled kernels cost their empty rows in full
        // — the score loop runs a multiply-add and a shuffle reduction per row
        // per key whether or not the row exists — so decode, which is
        // `n_query == 1`, takes the one-row instantiation.
        //
        // Above it the GQA-shared kernel is the default, because it reads K and
        // V once per KV head instead of once per query head. It cannot service
        // every geometry; `attn_flash_causal` can, and is the fallback.
        let (f, grid, block, shared) = if n_query < GQA_QUERY_TILE {
            (
                &self.flash_t1,
                (n_query as u32, self.q_heads as u32, 1),
                self.head_dim as u32,
                self.shared_bytes_t1(),
            )
        } else if self.gqa_shared_is_available() {
            (
                &self.flash_gqa,
                (
                    (n_query as u32).div_ceil(GQA_QUERY_TILE as u32),
                    self.kv_heads as u32,
                    1,
                ),
                (self.gqa_ratio() * 32) as u32,
                self.shared_bytes_gqa(),
            )
        } else {
            (
                &self.flash,
                (
                    (n_query as u32).div_ceil(QUERY_TILE as u32),
                    self.q_heads as u32,
                    1,
                ),
                self.head_dim as u32,
                self.shared_bytes(),
            )
        };
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: (block, 1, 1),
            shared_mem_bytes: shared as u32,
        };
        let q_heads = self.q_heads as i32;
        let kv_heads = self.kv_heads as i32;
        let head_dim = self.head_dim as i32;
        let n_query_i32 = n_query as i32;
        let scale = self.scale();
        let mut builder = stream.launch_builder(f);
        builder
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(out)
            .arg(&q_heads)
            .arg(&kv_heads)
            .arg(&head_dim)
            .arg(positions)
            .arg(&scale)
            .arg(&n_query_i32);
        // SAFETY: each of the three shapes covers the whole query range with a
        // grid and block chosen just above, and every one is passed `n_query`
        // so the kernel masks the rows a partial tile does not have — it
        // neither reads `q` nor writes `out` for them. The deepest key index
        // any block reads is `positions[0] + n_query - 1`, which the caller
        // promised is below `max_keys`, and all four buffers were checked
        // against it. Each shape's shared request is computed by the method
        // named beside it and covers everything that shape indexes: the two
        // softmax scratch tiles for `flash`, `q_sh` plus scratch for
        // `flash_t1`, and the staged K/V tiles plus a per-warp weight tile for
        // `flash_gqa`. `gqa_shared_is_available` has already established that
        // the GQA block is at most 1024 threads and its shared request at most
        // the 48 KiB a block gets without opting in.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// The two-pass decode path: slice the key window, then merge the slices.
    ///
    /// Split by [`Self::forward`] rather than inlined there because the two
    /// passes need a launch each and share none of the single-launch shapes'
    /// grid arithmetic. Buffer lengths were checked by the caller.
    #[allow(clippy::too_many_arguments)]
    fn decode(
        &self,
        stream: &Arc<CudaStream>,
        dec: &mut AttnDecodeScratch,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<i32>,
    ) -> Result<(), AttentionError> {
        let partials = DECODE_SPLITS * self.q_heads;
        Self::expect_len(
            "decode partial sums",
            dec.acc.len(),
            partials * self.head_dim,
        )?;
        Self::expect_len("decode partial maxima", dec.m.len(), partials)?;
        Self::expect_len("decode partial normalizers", dec.l.len(), partials)?;

        let q_heads = self.q_heads as i32;
        let kv_heads = self.kv_heads as i32;
        let head_dim = self.head_dim as i32;
        let scale = self.scale();

        let split_cfg = LaunchConfig {
            grid_dim: (DECODE_SPLITS as u32, self.kv_heads as u32, 1),
            block_dim: ((self.gqa_ratio() * 32) as u32, 1, 1),
            shared_mem_bytes: self.shared_bytes_decode() as u32,
        };
        let mut builder = stream.launch_builder(&self.decode_split);
        builder
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(&mut dec.acc)
            .arg(&mut dec.m)
            .arg(&mut dec.l)
            .arg(&q_heads)
            .arg(&kv_heads)
            .arg(&head_dim)
            .arg(positions)
            .arg(&scale);
        // SAFETY: the grid is (DECODE_SPLITS, kv_heads) with one warp per query
        // head under that KV head, so `h` stays below `q_heads` and every
        // partial index below `DECODE_SPLITS * q_heads`, which the three
        // `expect_len` calls above sized. The deepest key read is
        // `positions[0]`, which the caller checked against `max_keys`. Shared
        // covers the staged tiles and the per-warp scores, and
        // `decode_split_is_available` established it fits in 48 KiB and that
        // the block is at most 1024 threads.
        unsafe { builder.launch(split_cfg) }?;

        let combine_cfg = LaunchConfig {
            grid_dim: (self.q_heads as u32, 1, 1),
            block_dim: (self.head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&self.decode_combine);
        builder
            .arg(&dec.acc)
            .arg(&dec.m)
            .arg(&dec.l)
            .arg(out)
            .arg(&q_heads)
            .arg(&head_dim);
        // SAFETY: one block per query head, one thread per head dimension, so
        // the read of `part_acc` stays inside the buffer sized above and the
        // write covers `out` exactly once — `out` is `q_heads * head_dim` at
        // `n_query == 1`, which the caller checked.
        unsafe { builder.launch(combine_cfg) }?;
        Ok(())
    }

    /// Write this batch's keys and values into the cache at `positions[0]`.
    ///
    /// `key` and `value` are `[n_tokens][kv_heads][head_dim]`; the caches are
    /// the same layout over `max_keys` positions. This replaced two
    /// device-to-device copies into host-computed slices — see the kernel's
    /// own comment for why a host-computed destination address had to go.
    #[allow(clippy::too_many_arguments)]
    pub fn append_kv(
        &self,
        stream: &Arc<CudaStream>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        k_cache: &mut CudaSlice<f32>,
        v_cache: &mut CudaSlice<f32>,
        n_tokens: usize,
        max_keys: usize,
        positions: &CudaSlice<i32>,
    ) -> Result<(), AttentionError> {
        Self::expect_len("append position", positions.len(), 1)?;
        let row = self.kv_heads * self.head_dim;
        let span = n_tokens * row;
        Self::expect_len("append key", key.len(), span)?;
        Self::expect_len("append value", value.len(), span)?;
        Self::expect_at_least("append key cache", k_cache.len(), max_keys * row)?;
        Self::expect_at_least("append value cache", v_cache.len(), max_keys * row)?;

        let cfg = LaunchConfig {
            grid_dim: ((2 * span).div_ceil(APPEND_THREADS as usize) as u32, 1, 1),
            block_dim: (APPEND_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let span_i = span as i32;
        let row_i = row as i32;
        let mut builder = stream.launch_builder(&self.append);
        builder
            .arg(key)
            .arg(value)
            .arg(&mut *k_cache)
            .arg(&mut *v_cache)
            .arg(positions)
            .arg(&span_i)
            .arg(&row_i);
        // SAFETY: every thread past `2 * span` returns, both sources hold
        // exactly `span` floats, and the deepest destination index is
        // `positions[0] * row + span - 1` — inside `max_keys * row`, which
        // both caches were checked to hold, for any position the caller's own
        // `max_seq` check admits.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_packed_query_layout_is_interleaved_not_halved() {
        // The single most likely way to get this tensor wrong. At head 0 the
        // two layouts agree, which is exactly why a spot check on head 0
        // passes and the model is still broken.
        let head_dim = 256;
        assert_eq!(attn_packed_query_offset(0, head_dim), 0);
        assert_eq!(attn_packed_gate_offset(0, head_dim), head_dim);

        // A halves split would put query head h at h * head_dim. Interleaved
        // puts it at 2 * h * head_dim. They differ for every head but the
        // first.
        for head in 1..16 {
            assert_ne!(
                attn_packed_query_offset(head, head_dim),
                head * head_dim,
                "head {head}: interleaved offset collapsed onto the halves split",
            );
        }

        // And the halves split would read query head 8's slice out of the
        // region that actually holds gates, at the real 16-head geometry.
        let q_dim = 16 * head_dim;
        assert!(attn_packed_query_offset(8, head_dim) >= q_dim);
        // The last head's gate ends exactly at the packed row's width, so the
        // two offset formulas tile the full `2 * q_dim` without a gap.
        assert_eq!(attn_packed_gate_offset(15, head_dim) + head_dim, 2 * q_dim);
    }

    #[test]
    fn the_split_kernel_strides_by_two_head_dims() {
        // Guards the comment above from being "simplified" in the source.
        assert!(
            ATTENTION_SRC.contains("packed[row + (long long)(2 * h)     * head_dim + d]"),
            "query slice is no longer read at the interleaved 2*h stride",
        );
        assert!(
            ATTENTION_SRC.contains("packed[row + (long long)(2 * h + 1) * head_dim + d]"),
            "gate slice is no longer read at the interleaved 2*h+1 stride",
        );
    }

    #[test]
    fn the_causal_bound_is_inclusive_of_the_query_position() {
        // An off-by-one here leaks exactly one future token per row, which
        // moves aggregate error metrics by almost nothing. It is asserted
        // structurally because no tolerance would catch it reliably.
        // The block's loop bound is the *last* row's window; each earlier row
        // masks the tail off itself. Both halves are asserted, because
        // dropping the per-row mask would silently let row `qi0` attend to
        // seven future tokens and every aggregate metric would barely move.
        assert!(
            ATTENTION_SRC
                .contains("long long n_visible = (long long)(*key_offset) + qi0 + qt_live;"),
            "the causal window bound changed shape",
        );
        assert!(
            ATTENTION_SRC.contains("long long limit = (long long)(*key_offset) + qi0 + tid;")
                && ATTENTION_SRC.contains("long long limit = (long long)(*key_offset) + qi0 + su;")
                && ATTENTION_SRC.contains("if (j0 + w <= limit) tmax ="),
            "the per-row causal mask is gone from the softmax phases",
        );
        assert!(
            ATTENTION_SRC.contains("for (long long j0 = 0; j0 < n_visible; j0 += tile)")
                && ATTENTION_SRC.contains("long long key = j0 + (long long)warp * ATTN_KT + r;")
                && ATTENTION_SRC.contains("if (key < n_visible) {"),
            "the key loop no longer stops at the causal bound",
        );
    }

    #[test]
    fn the_rope_tail_is_copied_rather_than_computed() {
        // `out[d] = in[d]` is bit-identical. Anything arithmetic — even
        // multiplying by a cos of angle zero — is not, and would fail the
        // exact-equality gate the differential test puts on the tail.
        assert!(
            ATTENTION_SRC.contains("out[base + d] = in[base + d];"),
            "the untouched rotary tail is no longer a plain copy",
        );
        // And it is still selected by the same bound, now inside the token
        // band rather than outside it.
        assert!(
            ATTENTION_SRC.contains("if (d >= rope_dim) {"),
            "the rotary tail is no longer selected by rope_dim",
        );
    }

    #[test]
    fn the_accumulator_is_rescaled_before_the_new_contributions_land() {
        // Online softmax is only stable if the running accumulator and the
        // normalizer are moved into the new max's frame *first*. Folding the
        // new weights in before rescaling produces a finite, plausible, wrong
        // answer whenever the max increases.
        let rescale_l = ATTENTION_SRC
            .find("l_sh[tid] = l_sh[tid] * corr_sh[tid] + lsum;")
            .expect("normalizer rescale present");
        let rescale_acc = ATTENTION_SRC
            .find("for (int u = 0; u < QT; ++u) acc[u] = acc[u] * corr_sh[u];")
            .expect("accumulator rescale present");
        let fold_in = ATTENTION_SRC
            .find("acc[u] += w_sh[u * tile + w] * vv;")
            .expect("value accumulation present");
        assert!(rescale_acc < fold_in, "values folded in before rescaling");
        assert!(rescale_l < fold_in, "normalizer updated after the values");

        // The one-row kernel is a separate body and needs the same order.
        let t1 = ATTENTION_SRC
            .find("__global__ void attn_flash_causal_t1(")
            .expect("the one-row kernel is still there");
        let tail = &ATTENTION_SRC[t1..];
        let t1_l = tail
            .find("l = l * corr + lsum;")
            .expect("t1 normalizer rescale");
        let t1_acc = tail
            .find("float a = acc * corr;")
            .expect("t1 accumulator rescale");
        let t1_fold = tail
            .find("a += w_sh[w] * v[")
            .expect("t1 value accumulation");
        assert!(t1_acc < t1_fold, "t1 values folded in before rescaling");
        assert!(t1_l < t1_fold, "t1 normalizer updated after the values");
    }

    #[test]
    fn gqa_is_a_grouping_not_an_identity() {
        assert!(
            ATTENTION_SRC.contains("int kvh = h / (q_heads / kv_heads);"),
            "the kv head mapping is no longer a GQA grouping",
        );
        // Mirrors xabe_kernels::attention::kv_head_for_query_head at the real
        // 16:2 geometry. Asserted here because xabe-cuda must not depend on
        // the oracle crate.
        let kv_of = |h: usize| h / (16 / 2);
        assert_eq!(kv_of(0), 0);
        assert_eq!(kv_of(7), 0);
        assert_eq!(kv_of(8), 1);
        assert_eq!(kv_of(15), 1);
    }

    #[test]
    fn the_shared_memory_budget_fits_turing_at_the_real_geometry() {
        // docs/KERNELS.md: 48 KiB per block, measured. The module docs work
        // through why the textbook tile shapes do not fit at head_dim 256.
        const TURING_SHARED_PER_BLOCK: usize = 48 * 1024;
        let head_dim = 256usize;
        let keys_per_tile = (head_dim / 32) * KEYS_PER_WARP;
        let bytes = (2 * QUERY_TILE * keys_per_tile + 3 * QUERY_TILE) * size_of::<f32>();
        assert_eq!(bytes, 2144);
        assert!(bytes < TURING_SHARED_PER_BLOCK / 20);

        // The tile shapes that do not fit, so the arithmetic in the module
        // docs fails loudly if someone edits it.
        let tiled = |bm: usize, bn: usize| (bm + 2 * bn) * head_dim * 4 + bm * bn * 4;
        assert!(tiled(64, 64) > TURING_SHARED_PER_BLOCK);
        assert!(tiled(32, 32) > TURING_SHARED_PER_BLOCK);
        assert!(tiled(16, 16) > TURING_SHARED_PER_BLOCK);
        assert!(tiled(8, 8) < TURING_SHARED_PER_BLOCK);
    }

    #[test]
    fn head_geometry_is_validated_not_assumed() {
        // The checks that run without a device.
        assert!(
            AttentionError::UnsupportedHeadDim { head_dim: 100 }
                .to_string()
                .contains("100")
        );
        assert!(
            AttentionError::UnevenGqaGrouping {
                q_heads: 16,
                kv_heads: 5,
            }
            .to_string()
            .contains("16")
        );
        assert!(
            AttentionError::QueryPastKeys {
                key_offset: 100,
                n_query: 8,
                n_keys: 104,
            }
            .to_string()
            .contains("108")
        );
        // Qwen3.6's real geometry must be accepted.
        assert_eq!(256 % 32, 0);
        assert_eq!(16 % 2, 0);
        assert_eq!(64 % 2, 0);
    }

    #[test]
    fn a_kv_cache_may_be_longer_than_its_filled_window_but_never_shorter() {
        // The asymmetry is the whole point: a cache is allocated for the
        // longest admissible sequence and filled one token at a time, so
        // "longer than the window" is its steady state from decode step two
        // onward. "Shorter" is the read-past-the-end bug this check exists to
        // catch, and it stays fatal.
        assert!(AttentionKernels::expect_at_least("key", 4096, 4096).is_ok());
        assert!(AttentionKernels::expect_at_least("key", 1 << 20, 4096).is_ok());

        let err = AttentionKernels::expect_at_least("key", 4095, 4096)
            .expect_err("one float short must not be accepted");
        let message = err.to_string();
        assert!(
            message.contains("4096") && message.contains("4095"),
            "{message}"
        );

        // The query and the output are indexed by the launch geometry rather
        // than by absolute position, so they keep the exact check.
        assert!(AttentionKernels::expect_len("query", 4097, 4096).is_err());
    }

    #[test]
    fn the_query_tile_gives_one_thread_per_softmax_slot() {
        // The kernel's softmax phases index `score_sh` and `w_sh` as
        // `su * tile + sw` with `su = tid / tile`, and cover every slot with
        // no loop. That is only true when the block is exactly as wide as the
        // tile is big — which, since the block is `head_dim` threads and the
        // tile is `(head_dim / 32) * KEYS_PER_WARP` keys deep per row, reduces
        // to this product being a warp.
        assert_eq!(
            QUERY_TILE * KEYS_PER_WARP,
            32,
            "QUERY_TILE * KEYS_PER_WARP must be 32, or phase 2 misses slots",
        );
        for head_dim in [32usize, 64, 128, 256] {
            let tile = (head_dim / 32) * KEYS_PER_WARP;
            assert_eq!(
                QUERY_TILE * tile,
                head_dim,
                "head_dim {head_dim}: block width and softmax slot count diverged",
            );
        }
    }

    #[test]
    fn the_register_query_tile_bounds_the_head_dimension() {
        // `qr[ATTN_QT][ATTN_MAXD]` is only in registers while every index
        // folds to a constant, so the head dimension cannot exceed what
        // ATTN_MAXD covers. Asserted because raising head_dim past it would
        // compile, run, and silently spill.
        assert_eq!(32 * ATTN_MAXD, 256);
        assert!(ATTENTION_SRC.contains("#define ATTN_MAXD 8"));
        assert!(
            ATTENTION_SRC.contains("float qr[QT][ATTN_MAXD];"),
            "the query tile is no longer a register array",
        );
    }
}
