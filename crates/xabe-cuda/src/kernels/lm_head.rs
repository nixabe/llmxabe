//! LM head GEMV: hidden 2048 -> vocab 248,320, dequantizing Q8_0 in the
//! prologue.
//!
//! This is the last op of a forward pass and the one that turns a residual
//! stream into something samplable. It is also, per [`docs/MODEL.md`], the
//! single most bandwidth-expensive tensor in the model: 248,320 x 2048 at
//! Q8_0 is **540,344,320 bytes read for one decoded token**, against roughly
//! 2.86 GB of total weight traffic per token — about 19% of the whole decode
//! step for one matrix.
//!
//! ## Why there is no split-K
//!
//! [`docs/KERNELS.md`] plans this kernel as "split-K". **It is not, and the
//! arithmetic says it should not be.** Split-K exists to manufacture
//! parallelism when the output dimension is too small to fill the machine —
//! which is exactly the MoE decode situation that recommendation came from,
//! where a batch of three tokens meets a 512-row expert matrix.
//!
//! The LM head is the opposite regime. Assigning one warp per output row:
//!
//! - work available: 248,320 warps, one per vocabulary entry;
//! - work resident: 72 SMs x 32 warps/SM (sm_75's 1024-thread occupancy
//!   limit) = 2,304 warps.
//!
//! That is a **107x surplus** ([`LmHeadGeometry::occupancy_surplus`], which
//! is asserted in this module's tests so the justification fails loudly if
//! the vocabulary ever shrinks). Splitting K would multiply an already
//! 107x-oversubscribed grid, add a second launch, add a `vocab * K` partial
//! buffer, and make the summation order depend on the split — for zero
//! occupancy gain. It is a pure loss here.
//!
//! ## What the kernel is bound by
//!
//! Arithmetic intensity is 2 FLOP per weight element over 34/32 = 1.0625
//! bytes per element, i.e. **1.88 FLOP/byte**. The card's ridge point is
//! 16.3 TFLOP/s / 672 GB/s = 24 FLOP/byte, so this is bandwidth-bound by an
//! order of magnitude and nothing about the arithmetic is worth tuning. The
//! floor is 540,344,320 B / 672 GB/s = **804 us per weight pass**.
//!
//! What *is* worth tuning is everything that makes a bandwidth-bound kernel
//! fail to reach its bandwidth. Two things did, and both were measured
//! rather than reasoned about.
//!
//! ### 1. The 34-byte block breaks sector alignment (worth 1.35x)
//!
//! The obvious kernel gives lane `l` element `l` of each Q8_0 block, so a
//! warp reads the block's 32 contiguous quant bytes. Those 32 bytes are
//! never 32-byte aligned, because the block is 34 bytes and the alignment
//! cycles, so nearly every request straddles two sectors.
//!
//! Measured on the real head, one token: **1.692 ms, 319 GB/s, 47.5% of
//! spec**. The same kernel over a deliberately falsified 32-byte-strided
//! layout — same instruction count, same everything but the alignment — ran
//! in 0.880 ms, which is what identified the cause rather than guessing at
//! it.
//!
//! The fix is that **global memory is read only through 16-byte `uint4`
//! loads at 32-byte-aligned addresses**, into a small per-warp shared-memory
//! staging buffer; the 34-byte-granular unpacking then happens against
//! shared memory, where alignment is free. A staged segment is
//! [`STAGE_BLOCKS`] = 16 blocks = 544 bytes, and 544 is a multiple of 32,
//! which is what makes every segment start on a sector boundary. That took
//! it to **1.256 ms, 430 GB/s, 64.0%**.
//!
//! ### 2. The activation read, not the weight read (worth another 1.29x)
//!
//! At that point the kernel was no longer DRAM-bound. Deleting the
//! activation multiply — keeping every weight load and the whole staging
//! path — ran in **0.922 ms (87.2% of spec)**, so 0.33 ms of the 1.26 was
//! the `hidden` vector, not the weights.
//!
//! It is not DRAM traffic: `hidden` is 8 KB and lives in cache. It is
//! *memory-pipe* traffic — at one element per lane, every weight element
//! cost a separate scalar load of a separate activation, 4 bytes per lane
//! against 1.06 bytes of weight.
//!
//! The fix is a register tile: **four consecutive elements per lane, eight
//! lanes per Q8_0 block, four blocks per warp iteration**. The activation
//! read becomes one `float4` per lane — 512 contiguous bytes per warp
//! instruction, the widest a load can be — and the block's fp16 delta is
//! loaded once per four elements instead of once per one. Measured over
//! several runs: **0.900–0.972 ms, 556–600 GB/s, 82.7–89.3% of spec**.
//!
//! Cumulative: **1.7–1.9x over the naive form**, against llama.cpp's 44.5%
//! of peak on its weight path ([`docs/BENCHMARKS.md`]). What remains is
//! within a few percent of the 0.922 ms weight-only ceiling, so there is
//! very little left here that is not DRAM.
//!
//! ### The batch tile, and where it stops paying
//!
//! The one lever that changes *bytes per token* rather than bandwidth is
//! reusing a pass over the weights across several tokens: a warp holds `BT`
//! accumulators, dequantizes each weight element once, and multiplies it
//! into all `BT`. A prefill chunk of 8 tokens costs one 540 MB pass, not
//! eight.
//!
//! It does not scale linearly, and the measurement says so plainly. Every
//! row below is a single pass over the weights:
//!
//! | tokens | ms/call | ms/token |
//! |---:|---:|---:|
//! | 1 | 0.99 | 0.99 |
//! | 2 | 1.35 | 0.67 |
//! | 5 | 2.37 | 0.47 |
//! | 8 | 2.9–3.4 | 0.37–0.43 |
//!
//! Eight tokens is **2.4–2.9x** a single token's throughput, not 8x, for
//! exactly the reason section 2 identified: past `BT = 1` the activation
//! loads scale with `BT` while the weight read does not, and the kernel
//! stops being weight-bound almost immediately. It is also the widest
//! spread of any figure here, so treat the 8-token row as the least
//! reliable one.
//!
//! Per-token cost is still improving at `BT = 8`, which is why
//! [`MAX_BATCH_TILE`] is 8 and not smaller. The next lever would be a second
//! register tile over *rows*, so one activation load feeds several
//! vocabulary rows; that is not implemented and its size is unmeasured.
//!
//! ## Reused from the landed MoE kernel
//!
//! [`super::moe`]'s `q8_0_element` prologue, verbatim, including the operand
//! order `(float)q * d` that the milestone-04 dequant gate proved
//! bit-identical to [`xabe_kernels::quant::dequantize_q8_0`], and the
//! `load_half_le` inline-PTX widening from [`super::dequant`] (NVRTC has no
//! include path, so `<cuda_fp16.h>` is unreachable). [`super::moe::QuantTensor`]
//! is the weight-plus-format pair, unchanged.
//!
//! ## Why the result cannot be bit-identical to the CPU reference
//!
//! The dequantized *weights* are bit-identical — multiplication only. The
//! 2,048-term dot product is not: [`xabe_kernels::gemv::gemv`] sums
//! sequentially, each lane here sums its 64 terms in four-element groups and
//! the 32 lane totals are then combined in a shuffle tree, and fp32 addition
//! is not associative. The kernel also lets nvcc contract `acc += w * x` into an
//! FMA, which rounds once where the reference rounds twice; unlike
//! [`super::dequant`] and the convolution in [`super::layer_ops`], there is
//! no exactness left to protect here, so denying the contraction would cost
//! throughput for nothing. The gate is a measured tolerance plus **argmax
//! agreement**, which is the property that actually reaches the user.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};

use super::compile;
use super::moe::{ExpertQuant, QuantTensor};

/// Elements per Q8_0 block, and serialized bytes per block.
///
/// Duplicated from [`super::dequant`] rather than imported for the same
/// reason [`super::moe`] duplicates them: a change there must not silently
/// alter this module's alignment validation.
const QK8_0: usize = 32;
const BLOCK_Q8_0_BYTES: usize = 34;

/// Warps per block. One warp owns one output row, so this is also rows per
/// block.
///
/// 256 threads is four blocks per SM against sm_75's 1,024-thread limit, and
/// eight warps read eight adjacent rows — 8 x 2,176 = 17,408 contiguous
/// bytes per block. **The value is not load-bearing**: 16 warps per block
/// measured 1.267 ms against 8 warps' 1.256 ms on the real head, i.e. the
/// same within run-to-run variation. It is 8 because that is the smaller
/// shared-memory footprint, not because it was found to be faster.
const WARPS_PER_BLOCK: u32 = 8;

/// Threads per warp. Spelled out because the reduction shuffles with a full
/// `0xffffffff` mask and the block's x-dimension must be exactly this.
const WARP: u32 = 32;

/// Largest number of tokens one pass over the weights may serve.
///
/// Each token in the tile costs one fp32 accumulator register per lane and
/// one extra `float4` activation load per four weight elements. The measured
/// table in the module docs shows the saving is real but strongly
/// sublinear — 8 tokens is 2.85x one token's throughput, not 8x, because the
/// activation traffic scales with the tile while the weight read does not.
///
/// Eight is chosen because per-token cost is still improving there
/// (0.37–0.43 ms/token against 0.47 at five and 0.99 at one) and because the
/// accumulator array still fits comfortably in registers. It is not a knee;
/// the knee is at 2, and everything past it is buying progressively less.
pub const MAX_BATCH_TILE: usize = 8;

/// Streaming multiprocessors on a Quadro RTX 8000 (Turing TU102).
const SM_COUNT: usize = 72;

/// Resident warps per SM on sm_75: the 1,024-thread occupancy limit / 32.
const WARPS_PER_SM: usize = 32;

/// Q8_0 blocks one warp stages into shared memory at a time.
///
/// 16 blocks is 544 bytes, which is 34 `uint4` and — the property the whole
/// staging scheme rests on — a multiple of 32. A row starts at a multiple of
/// `(hidden/32) * 34`, which under [`LmHeadKernels::new`]'s divisibility
/// check is also a multiple of 32, so every stage begins on a sector
/// boundary and a 32-lane `uint4` load covers exactly 16 whole sectors.
///
/// Shared memory cost is `WARPS_PER_BLOCK * 544` = 4,352 B per block, which
/// leaves occupancy limited by the 1,024-thread cap rather than by shared
/// memory (sm_75 has 64 KiB of shared per SM; four 256-thread blocks want
/// 17 KiB of it).
const STAGE_BLOCKS: usize = 16;

/// Threads per block in both argmax passes.
const ARGMAX_THREADS: u32 = 256;

/// Upper bound on the first argmax pass's grid, and so the length of the
/// partial-winner buffers the caller supplies.
///
/// 512 blocks of 256 threads is 131,072 lanes over a 248,320-entry vocabulary
/// — about two entries each, which is enough to saturate a card whose whole
/// job here is to stream 993 KiB once. Capping it also bounds the second
/// pass, which is a single block and would rather reduce 512 winners than
/// 970.
pub const ARGMAX_BLOCKS: usize = 512;

const LM_HEAD_SRC: &str = r#"
#define ARGMAX_THREADS 256
#define LM_HEAD_STAGE_BLOCKS 16
#define LM_HEAD_STAGE_U4 ((LM_HEAD_STAGE_BLOCKS * 34) / 16)
#define LM_HEAD_STAGE_PER_LANE ((LM_HEAD_STAGE_U4 + 31) / 32)

// Reinterpret two little-endian bytes as an IEEE half and widen to float.
//
// Copied verbatim from `kernels/dequant.rs`. NVRTC compiles from a string
// with no include path, so <cuda_fp16.h> is unreachable and `__half2float`
// is not available; `cvt.f32.f16` is the same hardware conversion it lowers
// to, which is what makes the widening bit-identical to the host's.
__device__ __forceinline__ float load_half_le(const unsigned char* p) {
    unsigned short bits = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(bits));
    return f;
}

// One warp per vocabulary row; BT tokens share the pass over that row.
//
// grid.x  : ceil(vocab / warps_per_block)
// blockDim: (32, warps_per_block)
//
// `weight` is the raw Q8_0 tensor exactly as it appears in the GGUF file,
// [vocab][hidden] with hidden fastest-varying (GGUF dims [2048, 248320]), so
// row v starts at v * (hidden/32) * 34 and is contiguous.
//
// `hidden_states` is [max_tokens][hidden_dim] and `logits` is
// [max_tokens][vocab]; `token_base` selects this tile's slice of both. The
// tile is a *host* decision, not a device-side gate as in `moe.rs` — the LM
// head has no indirection to build and runs once per step rather than once
// per layer, and at decode the token count is fixed by the step shape, so
// the launch sequence is constant across replays of a given shape.
//
// The dequant prologue is `moe.rs`'s `q8_0_element`, specialized to the
// sequential walk: the delta is loaded once per block instead of once per
// element, and the operand order `(float)q * d` is preserved because that is
// what makes the weight bit-identical to the scalar reference.
//
// THE ALIGNMENT STAGE is the reason this kernel is not a plain byte-load
// loop, and it is worth 1.9x on real hardware -- see the module docs for the
// measurement. Global memory is read only through 16-byte `uint4` loads at
// 32-byte-aligned addresses, into a per-warp shared-memory buffer; the
// 34-byte-granular unpacking then happens against shared memory, where
// alignment costs nothing.
template <int BT>
__device__ __forceinline__ void lm_head_rows(
    const unsigned char* __restrict__ weight,
    const float* __restrict__ hidden_states,
    int hidden_dim,
    int vocab,
    int token_base,
    float* __restrict__ logits
) {
    // One staging buffer per warp: LM_HEAD_STAGE_U4 uint4 (544 B) each.
    // Declared as uint4 rather than unsigned char so the array's 16-byte
    // alignment is a property of its type, which is what every vector load
    // and store below depends on.
    extern __shared__ uint4 lm_head_stage[];

    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    uint4* s4 = lm_head_stage + threadIdx.y * LM_HEAD_STAGE_U4;
    const unsigned char* mine = (const unsigned char*)s4;

    // Uniform across the warp: blockDim.x is exactly one warp, so `row`
    // depends only on threadIdx.y. The early return therefore never strands
    // a lane inside a warp barrier or the shuffle reduction below -- and it
    // is the reason the staging buffer is per warp rather than per block: a
    // block-wide barrier behind this return would hang.
    if (row >= vocab) return;

    int nblocks = hidden_dim >> 5;
    const uint4* g4 = (const uint4*)(weight + (long long)row * nblocks * 34);
    const float* x = hidden_states + (long long)token_base * hidden_dim;

    // Four consecutive elements per lane, eight lanes per Q8_0 block, four
    // blocks per warp iteration. See the module docs: this is what turns the
    // activation read into one `float4` per lane -- 512 contiguous bytes per
    // warp instruction, the widest a load can be -- instead of four separate
    // scalar loads, and it amortizes the block's delta over four elements.
    int blk = lane >> 3;
    int sub = lane & 7;

    float acc[BT];
    #pragma unroll
    for (int t = 0; t < BT; ++t) acc[t] = 0.0f;

    // LM_HEAD_STAGE_BLOCKS * 34 = 544 bytes = 34 uint4 per warp per stage.
    // 544 is a multiple of 32, and a row starts at a multiple of
    // (hidden/32)*34 which is also a multiple of 32, so every stage begins at
    // a 32-byte boundary and the 32-lane uint4 load below covers exactly 16
    // whole sectors. Nothing is fetched that is not used.
    //
    // `hidden` is validated to be a multiple of 32 * LM_HEAD_STAGE_BLOCKS at
    // construction, so the last stage is never ragged and no load runs past
    // the row.
    //
    // The stage is double-buffered *through registers*, which is what
    // hand-rolled prefetch looks like on a part with no `cp.async`: the
    // global loads for segment s+1 are issued immediately after segment s is
    // published to shared memory, so their DRAM latency overlaps segment s's
    // arithmetic instead of stalling in front of it. Keeping the second
    // buffer in registers rather than in shared memory costs two `uint4` per
    // lane and leaves the shared footprint at one segment.
    uint4 pre[LM_HEAD_STAGE_PER_LANE];
    #pragma unroll
    for (int k = 0; k < LM_HEAD_STAGE_PER_LANE; ++k) {
        int i = lane + k * 32;
        if (i < LM_HEAD_STAGE_U4) pre[k] = g4[i];
    }

    for (int seg = 0; seg < nblocks; seg += LM_HEAD_STAGE_BLOCKS) {
        #pragma unroll
        for (int k = 0; k < LM_HEAD_STAGE_PER_LANE; ++k) {
            int i = lane + k * 32;
            if (i < LM_HEAD_STAGE_U4) s4[i] = pre[k];
        }
        __syncwarp();

        int next = seg + LM_HEAD_STAGE_BLOCKS;
        if (next < nblocks) {
            const uint4* src = g4 + (next / LM_HEAD_STAGE_BLOCKS) * LM_HEAD_STAGE_U4;
            #pragma unroll
            for (int k = 0; k < LM_HEAD_STAGE_PER_LANE; ++k) {
                int i = lane + k * 32;
                if (i < LM_HEAD_STAGE_U4) pre[k] = src[i];
            }
        }

        #pragma unroll
        for (int r = 0; r < LM_HEAD_STAGE_BLOCKS; r += 4) {
            const unsigned char* bp = mine + (r + blk) * 34;
            float d = load_half_le(bp);

            // The four quants as two 16-bit shared loads. `bp` inherits the
            // staging buffer's 16-byte alignment through a 34-byte block
            // stride, and `2 + 4*sub` is even, so the address is 2-byte
            // aligned -- but never reliably 4-byte aligned, which is why this
            // is two `ushort` loads and not one `uint`. The high byte is
            // narrowed through `unsigned char` first so that the int8 quant
            // sign-extends rather than taking an implementation-defined
            // conversion from a value above 127.
            const unsigned short* qp = (const unsigned short*)(bp + 2 + 4 * sub);
            unsigned short p0 = qp[0];
            unsigned short p1 = qp[1];
            // Operand order `(float)q * d` throughout, exactly as in
            // `moe.rs`'s q8_0_element; reassociating to `d * q` rounds
            // differently and costs bit-identical weights.
            float w0 = (float)(signed char)(unsigned char)(p0 & 0xFF) * d;
            float w1 = (float)(signed char)(unsigned char)(p0 >> 8) * d;
            float w2 = (float)(signed char)(unsigned char)(p1 & 0xFF) * d;
            float w3 = (float)(signed char)(unsigned char)(p1 >> 8) * d;

            int j = ((seg + r + blk) << 5) + 4 * sub;
            #pragma unroll
            for (int t = 0; t < BT; ++t) {
                // 16-byte aligned: `hidden_dim` is a multiple of 512 and `j`
                // a multiple of 4, and the base allocation is 256-aligned.
                float4 xv = *(const float4*)(x + (long long)t * hidden_dim + j);
                acc[t] += w0 * xv.x + w1 * xv.y + w2 * xv.z + w3 * xv.w;
            }
        }
        // The next stage overwrites what the loop above is still reading.
        __syncwarp();
    }

    // 32 lane subtotals -> one. Five shuffles per token and no barrier of
    // any kind: the warp is the whole reduction domain, which is the second
    // reason a row is owned by a warp rather than by a block.
    #pragma unroll
    for (int t = 0; t < BT; ++t) {
        float v = acc[t];
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, off);
        }
        if (lane == 0) {
            logits[(long long)(token_base + t) * vocab + row] = v;
        }
    }
}

// One entry point per batch tile. The tile is a template parameter and not a
// runtime argument on purpose: `acc[]` must live in registers, and a
// register array cannot be indexed by a loop whose bound the compiler does
// not know. A runtime bound would either spill `acc` to local memory or, if
// unrolled to MAX_BATCH_TILE with a `t < batch` predicate, issue eight times
// the loads and FMAs for a decode step that needs one -- about 0.5 ms of
// issue against a 0.8 ms DRAM floor, i.e. enough to stop being
// bandwidth-bound.
//
// Every tile from 1 to 8 gets an entry point, not just the powers of two.
// Eight extra kernels is eight extra template instantiations in one NVRTC
// compile; the alternative costs a whole extra 540 MB pass on the decode
// batches that are most common. A greedy power-of-two decomposition would
// serve 3 tokens as 2+1 and 7 as 4+2+1 -- two and three passes over the
// weights for batches that fit in one.
#define LM_HEAD_ENTRY(NAME, BT)                                             \
extern "C" __global__ void NAME(                                            \
    const unsigned char* __restrict__ weight,                               \
    const float* __restrict__ hidden_states,                                \
    int hidden_dim,                                                         \
    int vocab,                                                              \
    int token_base,                                                         \
    float* __restrict__ logits                                              \
) {                                                                         \
    lm_head_rows<BT>(weight, hidden_states, hidden_dim, vocab,              \
                     token_base, logits);                                   \
}

LM_HEAD_ENTRY(lm_head_gemv_b1, 1)
LM_HEAD_ENTRY(lm_head_gemv_b2, 2)
LM_HEAD_ENTRY(lm_head_gemv_b3, 3)
LM_HEAD_ENTRY(lm_head_gemv_b4, 4)
LM_HEAD_ENTRY(lm_head_gemv_b5, 5)
LM_HEAD_ENTRY(lm_head_gemv_b6, 6)
LM_HEAD_ENTRY(lm_head_gemv_b7, 7)
LM_HEAD_ENTRY(lm_head_gemv_b8, 8)

// ---------------------------------------------------------------------------
// Greatest logit, lowest index on a tie: `xabe_kernels::gemv::argmax`.
// ---------------------------------------------------------------------------
//
// The sampled token is the only part of a 248,320-entry logit vector anybody
// sees, and getting it to the host used to mean copying all 993 KiB of the
// vector and scanning it there. That cost 0.5 ms of a 11.6 ms decode step —
// 4% of the whole model — to move 993 KiB across PCIe and touch every one of
// a quarter of a million floats on one CPU core, in order to learn four
// bytes. Reducing on the device and copying the four bytes is the same
// answer for about 20 us.
//
// Two launches rather than one. A single block reading 993 KiB occupies one
// SM of seventy-two, and this reduction is bandwidth-bound: the first pass
// spreads the vector over a full grid, and the second reduces the per-block
// winners, a vector short enough that one block is no longer the wrong shape.
//
// Splitting it that way is only safe because of the tie-break. "Greatest
// value, lowest index" is associative *and* commutative, so the answer does
// not depend on how the vector is partitioned or in what order the partitions
// are merged. That is the same argument `moe_route`'s max and argmax
// reductions rest on — and the same one its softmax denominator cannot make,
// which is why that one still reduces through shared memory.
__device__ __forceinline__ void argmax_merge(float& bv, int& bi, float v, int i) {
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
}

// Reduce one block's per-thread candidate to thread 0.
//
// Every thread seeds from element 0 rather than from a -INFINITY sentinel, so
// a vector that is entirely NaN answers 0 — which is what the scalar
// reference does, because `NaN > NaN` is false and its running best never
// moves off index 0. A sentinel would answer "no candidate" instead, and the
// two would disagree on the one input where disagreement is hardest to spot.
__device__ __forceinline__ void argmax_reduce_block(float& bv, int& bi) {
    __shared__ float sv[ARGMAX_THREADS / 32];
    __shared__ int   si[ARGMAX_THREADS / 32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xffffffff, bv, off);
        int   oi = __shfl_down_sync(0xffffffff, bi, off);
        argmax_merge(bv, bi, ov, oi);
    }
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < ARGMAX_THREADS / 32; ++w) {
            argmax_merge(bv, bi, sv[w], si[w]);
        }
    }
}

extern "C" __global__ void argmax_partial(
    const float* __restrict__ x,
    int n,
    float* __restrict__ pv,
    int* __restrict__ pi
) {
    float bv = x[0];
    int   bi = 0;
    for (int i = blockIdx.x * ARGMAX_THREADS + threadIdx.x;
         i < n;
         i += gridDim.x * ARGMAX_THREADS) {
        argmax_merge(bv, bi, x[i], i);
    }
    argmax_reduce_block(bv, bi);
    if (threadIdx.x == 0) { pv[blockIdx.x] = bv; pi[blockIdx.x] = bi; }
}

extern "C" __global__ void argmax_final(
    const float* __restrict__ pv,
    const int* __restrict__ pi,
    int n,
    int* __restrict__ out
) {
    float bv = pv[0];
    int   bi = pi[0];
    for (int i = threadIdx.x; i < n; i += ARGMAX_THREADS) {
        argmax_merge(bv, bi, pv[i], pi[i]);
    }
    argmax_reduce_block(bv, bi);
    if (threadIdx.x == 0) out[0] = bi;
}
"#;

/// The LM head shape this instance is compiled and sized for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LmHeadGeometry {
    /// Residual stream width (2048).
    pub hidden: usize,
    /// Output vocabulary (248,320, and untied from the input embedding).
    pub vocab: usize,
    /// Largest token count any single step may present.
    ///
    /// Sizes the caller's `hidden_states` and `logits` buffers; the launch
    /// shape itself depends only on `vocab`.
    pub max_tokens: usize,
}

impl LmHeadGeometry {
    /// The real Qwen3.6 LM head, for `max_tokens` per step.
    pub const fn qwen3_6(max_tokens: usize) -> Self {
        Self {
            hidden: 2048,
            vocab: 248_320,
            max_tokens,
        }
    }

    /// Serialized bytes per vocabulary row (2,176 at the real geometry).
    pub const fn row_bytes(&self) -> usize {
        self.hidden / QK8_0 * BLOCK_Q8_0_BYTES
    }

    /// Serialized bytes of the whole head (540,344,320 at the real geometry).
    pub const fn weight_bytes(&self) -> usize {
        self.vocab * self.row_bytes()
    }

    /// Weight elements (508,559,360 at the real geometry).
    pub const fn elements(&self) -> usize {
        self.vocab * self.hidden
    }

    /// How many times the grid oversubscribes the card's resident warps.
    ///
    /// One warp per row against 72 SMs x 32 resident warps. This is the
    /// number that makes split-K pointless here; see the module docs.
    pub const fn occupancy_surplus(&self) -> usize {
        self.vocab / (SM_COUNT * WARPS_PER_SM)
    }

    /// Passes over the weight tensor a batch of `tokens` costs.
    ///
    /// There is a compiled entry point for every tile from 1 to
    /// [`MAX_BATCH_TILE`], so [`LmHeadKernels::forward`]'s decomposition is
    /// simply `ceil(tokens / MAX_BATCH_TILE)`: 37 prefill tokens cost five
    /// passes over the 540 MB tensor rather than thirty-seven.
    pub const fn weight_passes(&self, tokens: usize) -> usize {
        tokens.div_ceil(MAX_BATCH_TILE)
    }

    /// Weight bytes a batch of `tokens` actually pulls across the bus.
    ///
    /// The denominator of any honest GB/s figure for this kernel: it is
    /// [`Self::weight_bytes`] times [`Self::weight_passes`], not times the
    /// token count.
    pub const fn weight_bytes_read(&self, tokens: usize) -> usize {
        self.weight_passes(tokens) * self.weight_bytes()
    }
}

/// Largest tile that serves `remaining` tokens.
///
/// Every value in `1..=MAX_BATCH_TILE` has its own compiled entry point, so
/// this is just a cap — no power-of-two rounding, and therefore no batch
/// that pays an extra pass over the weights for being an awkward size.
const fn batch_tile(remaining: usize) -> usize {
    if remaining > MAX_BATCH_TILE {
        MAX_BATCH_TILE
    } else {
        remaining
    }
}

/// Something went wrong compiling, sizing, or launching the LM head.
#[derive(Debug)]
pub enum LmHeadError {
    /// NVRTC rejected the source, or the module failed to load.
    Compile(String),
    /// The driver failed.
    Driver(DriverError),
    /// A geometry the kernel cannot service, with the reason.
    UnsupportedGeometry {
        geometry: Box<LmHeadGeometry>,
        reason: &'static str,
    },
    /// A weight format this kernel does not unpack.
    ///
    /// Only Q8_0 is implemented, because `output.weight` in
    /// `Qwen3.6-35B-A3B-UD-Q6_K_XL` is Q8_0 — verified against the file's
    /// tensor directory. [`docs/KERNELS.md`] proposes re-quantizing the head
    /// to Q6_K to cut its 540 MB/token; that would slot [`super::moe`]'s
    /// `q6k_element` into the prologue unchanged, but shipping an untested
    /// second path ahead of that decision would be worse than rejecting it
    /// here.
    UnsupportedQuant(ExpertQuant),
    /// The weight tensor is not a whole number of Q8_0 blocks.
    ///
    /// Rejected rather than truncated: a partial trailing block would make
    /// the last vocabulary rows read past the tensor.
    RaggedWeights { bytes: usize, block_bytes: usize },
    /// The weight tensor does not hold `vocab * hidden` elements.
    WrongElementCount { expected: usize, found: usize },
    /// A buffer length disagrees with the declared geometry.
    ShapeMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    /// More tokens were presented than the buffers were sized for, or none.
    TooManyTokens { tokens: usize, max_tokens: usize },
}

impl std::fmt::Display for LmHeadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compile(m) => write!(f, "kernel compilation failed: {m}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::UnsupportedGeometry { geometry, reason } => {
                write!(f, "unsupported LM head geometry {geometry:?}: {reason}")
            }
            Self::UnsupportedQuant(q) => write!(
                f,
                "the LM head kernel unpacks Q8_0 only, not {q:?}; see LmHeadError::UnsupportedQuant",
            ),
            Self::RaggedWeights { bytes, block_bytes } => write!(
                f,
                "LM head: {bytes} bytes is not a whole number of {block_bytes} B Q8_0 blocks",
            ),
            Self::WrongElementCount { expected, found } => write!(
                f,
                "LM head: expected {expected} elements for this geometry, tensor holds {found}",
            ),
            Self::ShapeMismatch {
                what,
                expected,
                got,
            } => write!(f, "{what}: expected {expected} elements, got {got}"),
            Self::TooManyTokens { tokens, max_tokens } => write!(
                f,
                "{tokens} tokens is outside 1..={max_tokens}, the range the buffers were sized for",
            ),
        }
    }
}

impl std::error::Error for LmHeadError {}

impl From<DriverError> for LmHeadError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

/// The compiled LM head GEMV, one entry point per batch tile.
pub struct LmHeadKernels {
    /// Indexed by `tile - 1`, for tiles `1..=MAX_BATCH_TILE`.
    tiles: [CudaFunction; MAX_BATCH_TILE],
    argmax_partial: CudaFunction,
    argmax_final: CudaFunction,
    geometry: LmHeadGeometry,
}

impl LmHeadKernels {
    /// Compile and validate for `geometry`.
    ///
    /// The geometry is checked once here so the launch path has nothing left
    /// to reject, matching `MoeKernels::new` and `GdnKernels::new`.
    pub fn new(ctx: &Arc<CudaContext>, geometry: LmHeadGeometry) -> Result<Self, LmHeadError> {
        let bad = |reason: &'static str| LmHeadError::UnsupportedGeometry {
            geometry: Box::new(geometry),
            reason,
        };
        if geometry.hidden == 0 || geometry.vocab == 0 {
            return Err(bad("hidden and vocab must be non-zero"));
        }
        // A whole number of Q8_0 blocks, *and* a whole number of staging
        // segments. The second is what lets the staging loop read `uint4`s
        // with no ragged tail and no load past the end of a row: it makes
        // `(hidden/32) * 34` — the row stride — a multiple of 32, so every
        // row and every stage within it starts on a sector boundary.
        if !geometry.hidden.is_multiple_of(QK8_0 * STAGE_BLOCKS) {
            return Err(bad(
                "hidden must be a whole number of 16-block staging segments (a multiple of 512)",
            ));
        }
        if geometry.max_tokens == 0 {
            return Err(bad("max_tokens must be non-zero"));
        }
        // grid.x is capped at 2^31-1; 248,320 rows over 8 warps per block is
        // 31,040, so this only bites a synthetic geometry.
        if geometry.vocab.div_ceil(WARPS_PER_BLOCK as usize) > u32::MAX as usize {
            return Err(bad("vocabulary exceeds the grid.x limit"));
        }

        let ptx = compile(LM_HEAD_SRC, "lm_head").map_err(LmHeadError::Compile)?;
        let module = ctx.load_module(ptx)?;
        Ok(Self {
            tiles: [
                module.load_function("lm_head_gemv_b1")?,
                module.load_function("lm_head_gemv_b2")?,
                module.load_function("lm_head_gemv_b3")?,
                module.load_function("lm_head_gemv_b4")?,
                module.load_function("lm_head_gemv_b5")?,
                module.load_function("lm_head_gemv_b6")?,
                module.load_function("lm_head_gemv_b7")?,
                module.load_function("lm_head_gemv_b8")?,
            ],
            argmax_partial: module.load_function("argmax_partial")?,
            argmax_final: module.load_function("argmax_final")?,
            geometry,
        })
    }

    /// The geometry this instance was compiled for.
    pub fn geometry(&self) -> LmHeadGeometry {
        self.geometry
    }

    /// `logits[t][v] = sum_h weight[v][h] * hidden_states[t][h]`.
    ///
    /// `weight` is the raw Q8_0 tensor from the GGUF file, `hidden_states`
    /// is `[max_tokens][hidden]` and `logits` is `[max_tokens][vocab]`; rows
    /// past `tokens` in either are neither read nor written.
    ///
    /// A batch of `tokens` costs [`LmHeadGeometry::weight_passes`] passes
    /// over the 540 MB tensor, not `tokens` passes — see the module docs.
    pub fn forward(
        &self,
        stream: &Arc<CudaStream>,
        weight: QuantTensor<'_>,
        hidden_states: &CudaSlice<f32>,
        tokens: usize,
        logits: &mut CudaSlice<f32>,
    ) -> Result<(), LmHeadError> {
        let g = self.geometry;
        if weight.quant != ExpertQuant::Q8_0 {
            return Err(LmHeadError::UnsupportedQuant(weight.quant));
        }
        if !weight.bytes.len().is_multiple_of(BLOCK_Q8_0_BYTES) {
            return Err(LmHeadError::RaggedWeights {
                bytes: weight.bytes.len(),
                block_bytes: BLOCK_Q8_0_BYTES,
            });
        }
        let found = weight.bytes.len() / BLOCK_Q8_0_BYTES * QK8_0;
        if found != g.elements() {
            return Err(LmHeadError::WrongElementCount {
                expected: g.elements(),
                found,
            });
        }
        if tokens == 0 || tokens > g.max_tokens {
            return Err(LmHeadError::TooManyTokens {
                tokens,
                max_tokens: g.max_tokens,
            });
        }
        check_len(
            "hidden states",
            g.max_tokens * g.hidden,
            hidden_states.len(),
        )?;
        check_len("logits", g.max_tokens * g.vocab, logits.len())?;

        let hidden_dim = g.hidden as i32;
        let vocab = g.vocab as i32;
        let cfg = LaunchConfig {
            grid_dim: (g.vocab.div_ceil(WARPS_PER_BLOCK as usize) as u32, 1, 1),
            block_dim: (WARP, WARPS_PER_BLOCK, 1),
            // One staging buffer per warp. 4,352 B, well under the 48 KiB a
            // block may request without the `MAX_DYNAMIC_SHARED_SIZE_BYTES`
            // opt-in that cudarc's LaunchConfig does not expose.
            shared_mem_bytes: (WARPS_PER_BLOCK as usize * STAGE_BLOCKS * BLOCK_Q8_0_BYTES) as u32,
        };

        let mut base = 0usize;
        while base < tokens {
            let tile = batch_tile(tokens - base);
            let token_base = base as i32;
            let mut builder = stream.launch_builder(&self.tiles[tile - 1]);
            builder
                .arg(weight.bytes)
                .arg(hidden_states)
                .arg(&hidden_dim)
                .arg(&vocab)
                .arg(&token_base)
                .arg(&mut *logits);
            // SAFETY: one warp per row over a grid that covers `vocab` rows
            // and returns above it; the weight element count was checked
            // against `vocab * hidden` above, so `row * (hidden/32) * 34 +
            // 33` is the last byte and is in range. `token_base + tile <=
            // tokens <= max_tokens` bounds both the `hidden_states` reads and
            // the `logits` writes, and both buffers were length-checked.
            unsafe { builder.launch(cfg) }?;
            base += tile;
        }
        Ok(())
    }
}

impl LmHeadKernels {
    /// `out[0] = argmax(values[..n])`, ties going to the lower index.
    ///
    /// `partial_values` and `partial_indices` are scratch of exactly
    /// [`ARGMAX_BLOCKS`] elements each; `out` is one `i32`. All three are
    /// caller-owned so a decode step allocates nothing — see `AGENTS.md`
    /// rule 6.
    ///
    /// This is the device half of sampling. It exists so the host reads four
    /// bytes per step instead of the whole logit vector; see the kernel's own
    /// comment for what that was costing.
    pub fn argmax(
        &self,
        stream: &Arc<CudaStream>,
        values: &CudaSlice<f32>,
        n: usize,
        partial_values: &mut CudaSlice<f32>,
        partial_indices: &mut CudaSlice<i32>,
        out: &mut CudaSlice<i32>,
    ) -> Result<(), LmHeadError> {
        if n == 0 || n > values.len() {
            return Err(LmHeadError::ShapeMismatch {
                what: "argmax length",
                expected: values.len(),
                got: n,
            });
        }
        check_len("argmax partial values", ARGMAX_BLOCKS, partial_values.len())?;
        check_len(
            "argmax partial indices",
            ARGMAX_BLOCKS,
            partial_indices.len(),
        )?;
        check_len("argmax output", 1, out.len())?;

        // `n >= 1` was checked above, so the ceiling is at least one and there
        // is no lower bound left to apply.
        let blocks = n.div_ceil(ARGMAX_THREADS as usize).min(ARGMAX_BLOCKS);
        let n_i32 = n as i32;
        let one = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (ARGMAX_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&self.argmax_partial);
        builder
            .arg(values)
            .arg(&n_i32)
            .arg(&mut *partial_values)
            .arg(&mut *partial_indices);
        // SAFETY: the grid-stride loop is bounded by `n`, which was checked
        // against `values.len()`; `blocks <= ARGMAX_BLOCKS` is the length of
        // both partial buffers, and each block writes exactly its own slot.
        unsafe { builder.launch(one) }?;

        let blocks_i32 = blocks as i32;
        let two = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (ARGMAX_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&self.argmax_final);
        builder
            .arg(&*partial_values)
            .arg(&*partial_indices)
            .arg(&blocks_i32)
            .arg(&mut *out);
        // SAFETY: reads exactly the `blocks` slots the first pass wrote and
        // writes the single `i32` checked above.
        unsafe { builder.launch(two) }?;
        Ok(())
    }
}

/// Rejects a buffer whose length disagrees with the declared geometry.
fn check_len(what: &'static str, expected: usize, got: usize) -> Result<(), LmHeadError> {
    if got == expected {
        Ok(())
    } else {
        Err(LmHeadError::ShapeMismatch {
            what,
            expected,
            got,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen() -> LmHeadGeometry {
        LmHeadGeometry::qwen3_6(64)
    }

    #[test]
    fn the_dequant_prologue_multiplies_in_the_reference_order() {
        // Bit-identical weights depend on `q * d`. This guards all four
        // unpacked elements from being "simplified" into a different
        // rounding, the same way `moe.rs` and `dequant.rs` guard theirs.
        for lane in 0..4 {
            let expected = format!(
                "float w{lane} = (float)(signed char)(unsigned char)(p{} {}) * d;",
                lane / 2,
                if lane % 2 == 0 { "& 0xFF" } else { ">> 8" },
            );
            assert!(
                LM_HEAD_SRC.contains(&expected),
                "quant {lane} reassociated away from `(float)q * d`, or stopped \
                 narrowing through `unsigned char` first: expected `{expected}`",
            );
        }
    }

    #[test]
    fn quants_are_read_as_signed_and_the_high_byte_is_narrowed_first() {
        // Two errors in one place. Reading the int8 codes as unsigned flips
        // the sign of half the tensor while leaving magnitudes plausible.
        // And the high byte of each `ushort` arrives as a value in 0..=255,
        // so converting it straight to `signed char` is an
        // implementation-defined narrowing; going through `unsigned char`
        // first makes the two's-complement reinterpretation explicit.
        assert_eq!(
            LM_HEAD_SRC
                .matches("(float)(signed char)(unsigned char)")
                .count(),
            4,
            "the four quants are no longer all read as narrowed signed bytes",
        );
        assert!(
            !LM_HEAD_SRC.contains("(signed char)(p0 >> 8)"),
            "the high byte skipped the `unsigned char` narrowing",
        );
    }

    #[test]
    fn the_delta_is_loaded_once_per_four_elements_not_once_per_element() {
        // `moe.rs`'s `q8_0_element` re-derives the fp16 delta for every
        // element it unpacks, because a grouped GEMM addresses weights by
        // flat index. This kernel knows it is walking a block, so one
        // `load_half_le` serves the four quants a lane owns. Reverting to a
        // per-element call computes the same numbers with 4x the
        // cvt.f32.f16 and 4x the shared-memory traffic for the delta.
        let body = &LM_HEAD_SRC[at("void lm_head_rows")..at("#define LM_HEAD_ENTRY")];
        assert_eq!(
            body.matches("load_half_le").count(),
            1,
            "the delta load moved into, or out of, the four-element group",
        );
        assert!(body.contains("float d = load_half_le(bp);"));
    }

    #[test]
    fn a_row_is_owned_by_one_warp_so_the_reduction_needs_no_block_barrier() {
        // The structural claim in the module docs. `blockDim.x` is exactly
        // one warp, which is what makes `row` warp-uniform and the early
        // return safe in front of both `__syncwarp` and a full-mask shuffle.
        // A block-wide staging buffer would need `__syncthreads`, and the
        // early return would then be a hang — which is precisely why the
        // staging buffer is per warp.
        assert!(LM_HEAD_SRC.contains("int row = blockIdx.x * blockDim.y + threadIdx.y;"));
        assert!(LM_HEAD_SRC.contains("v += __shfl_down_sync(0xffffffff, v, off);"));
        let body = &LM_HEAD_SRC[at("void lm_head_rows")..at("#define LM_HEAD_ENTRY")];
        assert!(
            !body.contains("__syncthreads"),
            "a block-wide barrier appeared behind a warp-uniform early return",
        );
        assert!(
            body.contains("__syncwarp()"),
            "the shared staging buffer lost the warp barrier that publishes it",
        );
        assert_eq!(WARP, 32, "the full shuffle mask assumes a 32-lane warp");
    }

    #[test]
    fn every_global_weight_read_is_a_sector_aligned_vector_load() {
        // The 1.35x from the module docs, as a property of the code rather
        // than of a benchmark run. If any scalar load of `weight` comes
        // back, the 34-byte block stride starts straddling sectors again.
        let body = &LM_HEAD_SRC[at("void lm_head_rows")..at("#define LM_HEAD_ENTRY")];
        assert!(body.contains("const uint4* g4 = (const uint4*)(weight"));
        assert!(
            !body.contains("weight["),
            "a scalar index into `weight` came back; the 34-byte block stride \
             makes any sub-16-byte global load straddle sectors",
        );

        // And the arithmetic that makes those loads sector-aligned: a stage
        // is a whole number of 32-byte sectors, and so is a row.
        let g = qwen();
        let stage_bytes = STAGE_BLOCKS * BLOCK_Q8_0_BYTES;
        assert_eq!(stage_bytes, 544);
        assert!(stage_bytes.is_multiple_of(32), "a stage straddles sectors");
        assert!(
            stage_bytes.is_multiple_of(16),
            "a stage is not a whole uint4 count"
        );
        assert!(
            g.row_bytes().is_multiple_of(stage_bytes),
            "the last stage of a row would be ragged and read past it",
        );
        assert!(g.row_bytes().is_multiple_of(32), "rows straddle sectors");
        assert_eq!(g.row_bytes() / stage_bytes, 4, "four stages per row");
    }

    #[test]
    fn the_activation_read_is_one_float4_per_lane_over_four_blocks_per_warp() {
        // The second 1.29x. Four consecutive elements per lane and eight
        // lanes per block is what makes the warp's activation read 512
        // contiguous bytes in one instruction instead of four separate
        // 128-byte ones; reverting to a scalar `x[...]` load puts 0.33 ms
        // back on a 0.9 ms kernel.
        assert!(LM_HEAD_SRC.contains("int blk = lane >> 3;"));
        assert!(LM_HEAD_SRC.contains("int sub = lane & 7;"));
        assert!(
            LM_HEAD_SRC
                .contains("float4 xv = *(const float4*)(x + (long long)t * hidden_dim + j);")
        );
        assert!(LM_HEAD_SRC.contains("for (int r = 0; r < LM_HEAD_STAGE_BLOCKS; r += 4) {"));

        // Every element of every block is covered exactly once, and the
        // four a lane owns are contiguous so the float4 is legal. Checked by
        // enumeration rather than by argument, at the real stage size.
        let mut seen = vec![0u8; STAGE_BLOCKS * QK8_0];
        for r in (0..STAGE_BLOCKS).step_by(4) {
            for lane in 0..32usize {
                let (blk, sub) = (lane >> 3, lane & 7);
                let j = (r + blk) * QK8_0 + 4 * sub;
                assert!(j.is_multiple_of(4), "float4 load would be misaligned");
                for n in seen.iter_mut().skip(j).take(4) {
                    *n += 1;
                }
            }
        }
        assert!(
            seen.iter().all(|&n| n == 1),
            "the lane mapping double-counts or drops elements of a stage",
        );
    }

    #[test]
    fn there_is_one_compiled_entry_point_for_every_tile_not_just_powers_of_two() {
        // `forward` indexes `tiles` by `tile - 1`, so a gap in the sequence
        // would silently run the wrong batch width.
        for tile in 1..=MAX_BATCH_TILE {
            assert!(
                LM_HEAD_SRC.contains(&format!("LM_HEAD_ENTRY(lm_head_gemv_b{tile}, {tile})")),
                "no entry point for a batch tile of {tile}",
            );
        }
        assert_eq!(
            LM_HEAD_SRC.matches("LM_HEAD_ENTRY(lm_head_gemv_b").count(),
            MAX_BATCH_TILE,
        );
        assert_eq!(MAX_BATCH_TILE, 8);
    }

    #[test]
    fn the_batch_tiling_reads_the_weights_once_per_tile_not_once_per_token() {
        // The only lever that changes bytes/token for this kernel, stated as
        // arithmetic: 37 prefill tokens must cost 5 passes over 540 MB, not
        // 37.
        let g = qwen();
        assert_eq!(g.weight_passes(1), 1);
        assert_eq!(g.weight_passes(8), 1);
        assert_eq!(g.weight_passes(9), 2);
        assert_eq!(g.weight_passes(37), 5);
        assert_eq!(g.weight_passes(64), 8);

        // `weight_passes` is the ceiling only because every tile from 1 to 8
        // exists. Had `batch_tile` rounded down to a power of two, 3 tokens
        // would cost two passes and 7 would cost three; this checks the two
        // agree, which is what makes the ceiling claim true rather than
        // aspirational.
        for tokens in 1..=g.max_tokens {
            let mut remaining = tokens;
            let mut passes = 0;
            while remaining > 0 {
                let tile = batch_tile(remaining);
                assert!((1..=MAX_BATCH_TILE).contains(&tile));
                remaining -= tile;
                passes += 1;
            }
            assert_eq!(passes, g.weight_passes(tokens), "at {tokens} tokens");
            assert!(passes <= tokens);
        }

        assert_eq!(g.weight_bytes_read(8), g.weight_bytes());
        assert_eq!(g.weight_bytes_read(9), 2 * g.weight_bytes());
    }

    #[test]
    fn the_real_geometry_costs_the_540_mb_per_token_the_model_docs_claim() {
        // docs/MODEL.md's bandwidth table says 540 MB for the LM head. If
        // this ever stops matching, one of the two is wrong and this says
        // which.
        let g = qwen();
        assert_eq!(g.hidden, 2048);
        assert_eq!(g.vocab, 248_320);
        assert_eq!(g.row_bytes(), 2176);
        assert_eq!(g.elements(), 508_559_360);
        assert_eq!(g.weight_bytes(), 540_344_320);
        // The header of the real file reports exactly this many bytes for
        // `output.weight`, and `output_norm.weight` starts at that offset.
        assert_eq!(g.weight_bytes() / BLOCK_Q8_0_BYTES, 15_892_480);
        // 540.3 MB at the card's 672 GB/s spec bandwidth.
        let floor_us = g.weight_bytes() as f64 / 672.0e9 * 1.0e6;
        assert!(
            (800.0..810.0).contains(&floor_us),
            "the roofline floor moved to {floor_us:.0} us",
        );
    }

    #[test]
    fn the_grid_oversubscribes_the_card_enough_that_split_k_would_buy_nothing() {
        // The module docs reject docs/KERNELS.md's "split-K" plan on this
        // number. Asserting it means the rejection stops being true out loud
        // if the vocabulary ever shrinks to where split-K would help.
        let g = qwen();
        assert_eq!(SM_COUNT * WARPS_PER_SM, 2304);
        assert_eq!(g.occupancy_surplus(), 107);
        assert!(
            g.occupancy_surplus() >= 8,
            "one warp per row no longer fills the card by a wide margin; \
             split-K may now be worth what it costs",
        );
        // Wave quantization is not the concern either: 31,040 blocks over 72
        // SMs is 431 whole waves.
        let blocks = g.vocab.div_ceil(WARPS_PER_BLOCK as usize);
        assert_eq!(blocks, 31_040);
        assert!(blocks / SM_COUNT > 100);
    }

    #[test]
    fn arithmetic_intensity_puts_this_kernel_far_below_the_ridge_point() {
        // 2 FLOP per element over 34/32 bytes per element, against the
        // card's 16.3 TFLOP/s fp32 and 672 GB/s. If this were near the ridge
        // point the batch tiling would be the wrong optimization.
        let g = qwen();
        let intensity = 2.0 * g.elements() as f64 / g.weight_bytes() as f64;
        let ridge = 16.3e12 / 672.0e9;
        assert!((1.87..1.89).contains(&intensity), "{intensity}");
        assert!(
            intensity * 10.0 < ridge,
            "{intensity:.2} FLOP/byte is no longer an order of magnitude below \
             the {ridge:.1} FLOP/byte ridge point",
        );
    }

    #[test]
    fn geometry_is_validated_not_assumed() {
        // The checks that run without a device.
        let bad = LmHeadError::UnsupportedGeometry {
            geometry: Box::new(LmHeadGeometry {
                hidden: 100,
                ..qwen()
            }),
            reason: "hidden must be a whole number of 32-element Q8_0 blocks",
        };
        assert!(bad.to_string().contains("Q8_0 blocks"));
        assert!(
            LmHeadError::UnsupportedQuant(ExpertQuant::Q6K)
                .to_string()
                .contains("Q8_0 only"),
        );
        assert!(
            LmHeadError::RaggedWeights {
                bytes: 2177,
                block_bytes: 34,
            }
            .to_string()
            .contains("2177"),
        );
        assert!(
            LmHeadError::WrongElementCount {
                expected: 508_559_360,
                found: 2048,
            }
            .to_string()
            .contains("508559360"),
        );
        assert!(
            LmHeadError::ShapeMismatch {
                what: "logits",
                expected: 15_892_480,
                got: 2048,
            }
            .to_string()
            .contains("15892480"),
        );
        assert!(
            LmHeadError::TooManyTokens {
                tokens: 0,
                max_tokens: 64,
            }
            .to_string()
            .contains("1..=64"),
        );
        assert!(check_len("logits", 4, 4).is_ok());
        assert!(check_len("logits", 4, 5).is_err());
    }

    /// Where a snippet starts in the kernel source, or a failure naming it.
    fn at(needle: &str) -> usize {
        LM_HEAD_SRC
            .find(needle)
            .unwrap_or_else(|| panic!("kernel source no longer contains `{needle}`"))
    }
}
