//! LM head GEMV: hidden 2048 -> vocab 248,320, dequantizing Q8_0 inline.
//!
//! The last op of a forward pass, and the single most bandwidth-expensive
//! tensor in the model: 248,320 x 2048 at Q8_0 is **540 MB read per decoded
//! token**, about 19% of the step's whole weight traffic for one matrix. The
//! floor is 540,344,320 B / 672 GB/s = **804 us per weight pass**.
//!
//! Arithmetic intensity is 1.88 FLOP/byte against a 24 FLOP/byte ridge point,
//! so this is bandwidth-bound by an order of magnitude and nothing about the
//! arithmetic is worth tuning. Everything below is about not *wasting*
//! bandwidth.
//!
//! # Three design choices, each measured
//!
//! **No split-K.** Split-K manufactures parallelism when the output dimension
//! cannot fill the machine. Here one warp per vocabulary row is 248,320 warps
//! against 2,304 resident — a 107x surplus
//! ([`LmHeadGeometry::occupancy_surplus`], asserted in this module's tests so
//! the justification fails loudly if the vocabulary ever shrinks). Splitting K
//! would add a launch, a `vocab * K` partial buffer and a split-dependent
//! summation order for no occupancy gain.
//!
//! **Global memory is read only through 16-byte `uint4` loads at 32-byte
//! aligned addresses**, into a per-warp shared staging buffer; the 34-byte
//! granular unpack then runs against shared memory, where alignment is free. A
//! Q8_0 block is 34 bytes, so a warp reading its 32 contiguous quants is never
//! sector-aligned and nearly every request straddles two sectors. `STAGE_BLOCKS`
//! is 16 blocks = 544 bytes, a multiple of 32, which is what makes every
//! staged segment start on a sector boundary. Worth 1.35x.
//!
//! **Four consecutive elements per lane, eight lanes per block, four blocks
//! per warp iteration.** With one element per lane, every weight element cost
//! a separate scalar load of a separate activation — 4 bytes of activation
//! against 1.06 bytes of weight, and the kernel was memory-pipe bound rather
//! than DRAM bound. The tile makes the activation read one `float4` per lane
//! and loads the block's fp16 delta once per four elements. Worth another
//! 1.29x, and lands the kernel within a few percent of a weight-only ablation.
//!
//! # Two tile axes, and what each buys
//!
//! - **Batch tile `BT`** — one warp holds `BT` accumulators and multiplies
//!   each dequantized weight into all of them, so a prefill chunk costs one
//!   540 MB pass rather than `BT`. Sub-linear by construction: past `BT = 1`
//!   the activation loads scale with the tile and the weight read does not, so
//!   eight tokens are ~2.4-2.9x one token, not 8x. Per-token cost is still
//!   improving at 8, which is why [`MAX_BATCH_TILE`] is 8.
//! - **Row tile `RT`** — one warp takes `RT` adjacent vocabulary rows so the
//!   activation `float4` loads sit outside the row loop and feed every row's
//!   FMAs from the same registers. Per-row arithmetic and its order are
//!   untouched, so the tiled kernels are **bit-identical** to the untiled one
//!   and gated as such. `vocab % RT == 0` is required so no partially-live
//!   warp group exists.
//!
//! # Why the result is not bit-identical to the CPU reference
//!
//! The dequantized weights are — multiplication only. The 2,048-term dot
//! product is not: `xabe_kernels::gemv::gemv` sums sequentially, each lane
//! here sums 64 terms in four-element groups and the 32 lane totals combine in
//! a shuffle tree. The kernel also lets nvcc contract `acc += w * x` into an
//! FMA, which rounds once where the reference rounds twice; there is no
//! exactness left to protect here, so denying the contraction would cost
//! throughput for nothing. The gate is a measured tolerance plus **argmax
//! agreement**, which is the property that reaches the user.
//!
//! The Q8_0 prologue, the operand order `(float)q * d`, and the `load_half_le`
//! inline-PTX widening are shared verbatim with [`super::moe`] and
//! [`super::dequant`] — NVRTC has no include path, so `<cuda_fp16.h>` is
//! unreachable.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};

use super::compile;

/// Elements per Q8_0 block, and serialized bytes per block.
///
/// Duplicated from [`super::dequant`] rather than imported for the same
/// reason [`super::moe`] duplicates them: a change there must not silently
/// alter this module's alignment validation.
const QK8_0: usize = 32;
const BLOCK_Q8_0_BYTES: usize = 34;

/// Elements per k-quant superblock, and serialized bytes per Q6_K superblock.
/// Duplicated for the same reason as the Q8_0 pair above.
const QK_K: usize = 256;
const BLOCK_Q6_K_BYTES: usize = 210;

/// The remaining block geometries, same provenance: `ggml-common.h`'s
/// `block_q4_0`, `block_q4_K` and `block_q5_K` static asserts.
const QK4_0: usize = 32;
const BLOCK_Q4_0_BYTES: usize = 18;
const BLOCK_Q4_K_BYTES: usize = 144;
const BLOCK_Q5_K_BYTES: usize = 176;

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

/// Weight elements one warp of the bf16 body covers per iteration: 32 lanes
/// times the 8 elements a 16-byte load carries. `hidden` must be a multiple
/// of this for the loop to have no ragged tail. Mirrors the `32 * 8` in
/// `lm_head_rows_bf16`.
const BF16_WARP_STEP: usize = 256;

/// Weight elements one warp of the Q6_K body covers per iteration: the whole
/// 256-element superblock, since the two 128-element halves are an inner
/// unrolled pair. Mirrors the `hidden_dim >> 8` in `lm_head_rows_q6_k`.
const Q6_K_WARP_STEP: usize = QK_K;

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

// One warp per RT adjacent vocabulary rows; BT tokens share the pass.
//
// grid.x  : ceil(vocab / (warps_per_block * RT))
// blockDim: (32, warps_per_block)
//
// RT is the register tile over rows the module docs name as the next lever
// past the batch tile: past BT = 1 the activation loads scale with BT while
// the weight read does not, so a second row per warp lets one `float4`
// activation load feed two rows' FMAs and halves the activation-pipe traffic
// per weight element. The weight traffic itself is unchanged — different
// rows are different bytes — which is why RT tiles the *activation* cost
// only. RT > 1 requires `vocab % RT == 0` (the launch path checks it), so
// every row of a warp's group is live together and no per-row guard is
// needed inside the loop.
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
template <int BT, int RT>
__device__ __forceinline__ void lm_head_rows(
    const unsigned char* __restrict__ weight,
    const float* __restrict__ hidden_states,
    int hidden_dim,
    int vocab,
    int token_base,
    float* __restrict__ logits
) {
    // One staging buffer per warp: LM_HEAD_STAGE_U4 uint4 (544 B) per row.
    // Declared as uint4 rather than unsigned char so the array's 16-byte
    // alignment is a property of its type, which is what every vector load
    // and store below depends on.
    extern __shared__ uint4 lm_head_stage[];

    int row = (blockIdx.x * blockDim.y + threadIdx.y) * RT;
    int lane = threadIdx.x;
    uint4* s4 = lm_head_stage + threadIdx.y * (LM_HEAD_STAGE_U4 * RT);
    const unsigned char* mine = (const unsigned char*)s4;

    // Uniform across the warp: blockDim.x is exactly one warp, so `row`
    // depends only on threadIdx.y. The early return therefore never strands
    // a lane inside a warp barrier or the shuffle reduction below -- and it
    // is the reason the staging buffer is per warp rather than per block: a
    // block-wide barrier behind this return would hang. At RT > 1 the launch
    // path guarantees `vocab % RT == 0`, so the whole group is live or the
    // whole group is past the end -- there is no partially live warp.
    if (row >= vocab) return;

    int nblocks = hidden_dim >> 5;
    const uint4* g4[RT];
    #pragma unroll
    for (int rt = 0; rt < RT; ++rt) {
        g4[rt] = (const uint4*)(weight + (long long)(row + rt) * nblocks * 34);
    }
    const float* x = hidden_states + (long long)token_base * hidden_dim;

    // Four consecutive elements per lane, eight lanes per Q8_0 block, four
    // blocks per warp iteration. See the module docs: this is what turns the
    // activation read into one `float4` per lane -- 512 contiguous bytes per
    // warp instruction, the widest a load can be -- instead of four separate
    // scalar loads, and it amortizes the block's delta over four elements.
    int blk = lane >> 3;
    int sub = lane & 7;

    float acc[RT * BT];
    #pragma unroll
    for (int t = 0; t < RT * BT; ++t) acc[t] = 0.0f;

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
    uint4 pre[RT][LM_HEAD_STAGE_PER_LANE];
    #pragma unroll
    for (int rt = 0; rt < RT; ++rt) {
        #pragma unroll
        for (int k = 0; k < LM_HEAD_STAGE_PER_LANE; ++k) {
            int i = lane + k * 32;
            if (i < LM_HEAD_STAGE_U4) pre[rt][k] = g4[rt][i];
        }
    }

    for (int seg = 0; seg < nblocks; seg += LM_HEAD_STAGE_BLOCKS) {
        #pragma unroll
        for (int rt = 0; rt < RT; ++rt) {
            #pragma unroll
            for (int k = 0; k < LM_HEAD_STAGE_PER_LANE; ++k) {
                int i = lane + k * 32;
                if (i < LM_HEAD_STAGE_U4) s4[rt * LM_HEAD_STAGE_U4 + i] = pre[rt][k];
            }
        }
        __syncwarp();

        int next = seg + LM_HEAD_STAGE_BLOCKS;
        if (next < nblocks) {
            #pragma unroll
            for (int rt = 0; rt < RT; ++rt) {
                const uint4* src = g4[rt] + (next / LM_HEAD_STAGE_BLOCKS) * LM_HEAD_STAGE_U4;
                #pragma unroll
                for (int k = 0; k < LM_HEAD_STAGE_PER_LANE; ++k) {
                    int i = lane + k * 32;
                    if (i < LM_HEAD_STAGE_U4) pre[rt][k] = src[i];
                }
            }
        }

        #pragma unroll
        for (int r = 0; r < LM_HEAD_STAGE_BLOCKS; r += 4) {
            int j = ((seg + r + blk) << 5) + 4 * sub;
            // One `float4` per token per group, loaded before the row loop so
            // all RT rows' FMAs feed from the same registers. This is the
            // whole point of the row tile: the activation read no longer
            // scales with RT while the weight read (which does scale) is the
            // part DRAM was already paying for.
            float4 xv[BT];
            #pragma unroll
            for (int t = 0; t < BT; ++t) {
                // 16-byte aligned: `hidden_dim` is a multiple of 512 and `j`
                // a multiple of 4, and the base allocation is 256-aligned.
                xv[t] = *(const float4*)(x + (long long)t * hidden_dim + j);
            }
            #pragma unroll
            for (int rt = 0; rt < RT; ++rt) {
                const unsigned char* bp = mine + rt * (LM_HEAD_STAGE_U4 * 16) + (r + blk) * 34;
                float d = load_half_le(bp);

                // The four quants as two 16-bit shared loads. `bp` inherits
                // the staging buffer's 16-byte alignment through a 34-byte
                // block stride, and `2 + 4*sub` is even, so the address is
                // 2-byte aligned -- but never reliably 4-byte aligned, which
                // is why this is two `ushort` loads and not one `uint`. The
                // high byte is narrowed through `unsigned char` first so that
                // the int8 quant sign-extends rather than taking an
                // implementation-defined conversion from a value above 127.
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

                #pragma unroll
                for (int t = 0; t < BT; ++t) {
                    acc[rt * BT + t] += w0 * xv[t].x + w1 * xv[t].y + w2 * xv[t].z + w3 * xv[t].w;
                }
            }
        }
        // The next stage overwrites what the loop above is still reading.
        __syncwarp();
    }

    // 32 lane subtotals -> one. Five shuffles per token per row and no
    // barrier of any kind: the warp is the whole reduction domain, which is
    // the second reason a row is owned by a warp rather than by a block.
    #pragma unroll
    for (int rt = 0; rt < RT; ++rt) {
        #pragma unroll
        for (int t = 0; t < BT; ++t) {
            float v = acc[rt * BT + t];
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_down_sync(0xffffffff, v, off);
            }
            if (lane == 0) {
                logits[(long long)(token_base + t) * vocab + row + rt] = v;
            }
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
    lm_head_rows<BT, 1>(weight, hidden_states, hidden_dim, vocab,           \
                        token_base, logits);                                \
}

LM_HEAD_ENTRY(lm_head_gemv_b1, 1)
LM_HEAD_ENTRY(lm_head_gemv_b2, 2)
LM_HEAD_ENTRY(lm_head_gemv_b3, 3)
LM_HEAD_ENTRY(lm_head_gemv_b4, 4)
LM_HEAD_ENTRY(lm_head_gemv_b5, 5)
LM_HEAD_ENTRY(lm_head_gemv_b6, 6)
LM_HEAD_ENTRY(lm_head_gemv_b7, 7)
LM_HEAD_ENTRY(lm_head_gemv_b8, 8)

// The row-tiled decode entry points. Only the three-token tile gets them:
// that is the N=3 batched decode shape, the one place the profile shows the
// activation-pipe cost (2.22 ms against a 0.92 ms weight-read floor,
// docs/BENCHMARKS.md 2026-08-20). Row arithmetic and order are identical to
// the RT=1 instantiations -- only the number of accumulators in flight
// differs -- so `b3r2`/`b3r4` against three `b1` launches must be
// bit-identical, and the differential test asserts exactly that.
#define LM_HEAD_ENTRY_RT(NAME, BT, RT)                                      \
extern "C" __global__ void NAME(                                            \
    const unsigned char* __restrict__ weight,                               \
    const float* __restrict__ hidden_states,                                \
    int hidden_dim,                                                         \
    int vocab,                                                              \
    int token_base,                                                         \
    float* __restrict__ logits                                              \
) {                                                                         \
    lm_head_rows<BT, RT>(weight, hidden_states, hidden_dim, vocab,          \
                         token_base, logits);                               \
}

LM_HEAD_ENTRY_RT(lm_head_gemv_b3r2, 3, 2)
LM_HEAD_ENTRY_RT(lm_head_gemv_b3r4, 3, 4)

// ---------------------------------------------------------------------------
// The same GEMV over a bf16 weight tensor.
// ---------------------------------------------------------------------------
//
// `Qwen3.8-27B-UD-Q8_K_XL` stores `output.weight`, every attention `attn_q` /
// `attn_k` / `attn_v`, and `nextn.eh_proj` as bf16 — 53 of its 866 tensors,
// and the only ones this kernel family is asked for that are not Q8_0. They
// are also the *largest* ones it is asked for: the head alone is 2.54 GiB.
// Widening them to fp32 on the host would cost 5.1 GiB on the card and double
// the per-token read of the single most bandwidth-expensive tensor in the
// model, and requantizing them to Q8_0 would change the model. So the kernel
// reads them where they are.
//
// bf16 is a truncated fp32, so widening is `bits << 16` — exact, and with no
// hardware conversion instruction needed. That is also why there is no
// staging buffer here and no alignment prologue: a bf16 row is dense, every
// byte of a fetched sector is used, and a row starts at `v * hidden * 2` with
// `hidden` a multiple of 512, so every load below is naturally aligned. The
// Q8_0 path needs its staging pass only because a 34-byte block stride puts
// useful bytes across sector boundaries; that problem does not exist here.
//
// Each lane takes 8 consecutive elements — one `uint4`, 16 bytes, which is
// the widest load there is and matches what the Q8_0 path's staging pass
// achieves. A warp therefore covers 256 elements per iteration, and
// `hidden_dim` must be a multiple of 256 for the loop to have no ragged
// tail. `LmHeadKernels::with_row_tile` already requires a multiple of 512.
__device__ __forceinline__ float widen_bf16(unsigned int bits) {
    return __int_as_float((int)(bits << 16));
}

template <int BT, int RT>
__device__ __forceinline__ void lm_head_rows_bf16(
    const unsigned char* __restrict__ weight,
    const float* __restrict__ hidden_states,
    int hidden_dim,
    int vocab,
    int token_base,
    float* __restrict__ logits
) {
    int row = (blockIdx.x * blockDim.y + threadIdx.y) * RT;
    int lane = threadIdx.x;
    if (row >= vocab) return;

    const uint4* g4[RT];
    #pragma unroll
    for (int rt = 0; rt < RT; ++rt) {
        g4[rt] = (const uint4*)(weight + (long long)(row + rt) * hidden_dim * 2);
    }
    const float* x = hidden_states + (long long)token_base * hidden_dim;

    float acc[RT * BT];
    #pragma unroll
    for (int t = 0; t < RT * BT; ++t) acc[t] = 0.0f;

    for (int base = 0; base < hidden_dim; base += 32 * 8) {
        int j = base + lane * 8;
        // Two `float4` per token, loaded before the row loop so all RT rows'
        // FMAs feed from the same registers — the row tile's whole purpose,
        // as in the Q8_0 body above.
        float4 xa[BT];
        float4 xb[BT];
        #pragma unroll
        for (int t = 0; t < BT; ++t) {
            const float* xt = x + (long long)t * hidden_dim + j;
            xa[t] = *(const float4*)xt;
            xb[t] = *(const float4*)(xt + 4);
        }
        #pragma unroll
        for (int rt = 0; rt < RT; ++rt) {
            uint4 p = g4[rt][(base >> 3) + lane];
            float w0 = widen_bf16(p.x & 0xFFFFu);
            float w1 = widen_bf16(p.x >> 16);
            float w2 = widen_bf16(p.y & 0xFFFFu);
            float w3 = widen_bf16(p.y >> 16);
            float w4 = widen_bf16(p.z & 0xFFFFu);
            float w5 = widen_bf16(p.z >> 16);
            float w6 = widen_bf16(p.w & 0xFFFFu);
            float w7 = widen_bf16(p.w >> 16);
            #pragma unroll
            for (int t = 0; t < BT; ++t) {
                acc[rt * BT + t] += w0 * xa[t].x + w1 * xa[t].y
                                  + w2 * xa[t].z + w3 * xa[t].w
                                  + w4 * xb[t].x + w5 * xb[t].y
                                  + w6 * xb[t].z + w7 * xb[t].w;
            }
        }
    }

    #pragma unroll
    for (int rt = 0; rt < RT; ++rt) {
        #pragma unroll
        for (int t = 0; t < BT; ++t) {
            float v = acc[rt * BT + t];
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_down_sync(0xffffffff, v, off);
            }
            if (lane == 0) {
                logits[(long long)(token_base + t) * vocab + row + rt] = v;
            }
        }
    }
}

#define LM_HEAD_BF16_ENTRY(NAME, BT, RT)                                    \
extern "C" __global__ void NAME(                                            \
    const unsigned char* __restrict__ weight,                               \
    const float* __restrict__ hidden_states,                                \
    int hidden_dim,                                                         \
    int vocab,                                                              \
    int token_base,                                                         \
    float* __restrict__ logits                                              \
) {                                                                         \
    lm_head_rows_bf16<BT, RT>(weight, hidden_states, hidden_dim, vocab,     \
                              token_base, logits);                          \
}

LM_HEAD_BF16_ENTRY(lm_head_bf16_b1, 1, 1)
LM_HEAD_BF16_ENTRY(lm_head_bf16_b2, 2, 1)
LM_HEAD_BF16_ENTRY(lm_head_bf16_b3, 3, 1)
LM_HEAD_BF16_ENTRY(lm_head_bf16_b4, 4, 1)
LM_HEAD_BF16_ENTRY(lm_head_bf16_b5, 5, 1)
LM_HEAD_BF16_ENTRY(lm_head_bf16_b6, 6, 1)
LM_HEAD_BF16_ENTRY(lm_head_bf16_b7, 7, 1)
LM_HEAD_BF16_ENTRY(lm_head_bf16_b8, 8, 1)
LM_HEAD_BF16_ENTRY(lm_head_bf16_b3r2, 3, 2)
LM_HEAD_BF16_ENTRY(lm_head_bf16_b3r4, 3, 4)

// ---------------------------------------------------------------------------
// The same GEMV over an F16 weight tensor.
// ---------------------------------------------------------------------------
//
// Structurally the bf16 body with one line changed: an IEEE half needs a real
// conversion where bf16 needs only a shift. Everything else — 8 elements per
// lane in one `uint4`, 256 per warp per step, no staging and no alignment
// prologue — holds for exactly the same reason, because an f16 row is dense
// and `hidden` is a multiple of 512.
//
// It is a separate body rather than a template parameter over the bf16 one on
// purpose. The bf16 GEMV serves `qwen35`'s head and its attn_q/k/v — 2.54 GiB
// of the shipped dense model — and it has **no differential test of its own**
// in `crates/xabe-engine/tests`. Refactoring a live, unguarded path to save
// forty lines trades a real risk for a cosmetic gain. If a bf16 differential
// lands later, folding the two together is a safe follow-up.
//
// `cvt.f32.f16` rather than a hand-rolled widening: NVRTC has no include path
// so `__half2float` is unreachable, and the hardware instruction gets
// subnormal halves right, which a shift-and-mask version has to be told about.
// The quantizer emits subnormals for near-zero values, so that path is live.
__device__ __forceinline__ float widen_f16(unsigned int bits) {
    unsigned short h = (unsigned short)(bits & 0xFFFFu);
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

template <int BT>
__device__ __forceinline__ void lm_head_rows_f16(
    const unsigned char* __restrict__ weight,
    const float* __restrict__ hidden_states,
    int hidden_dim,
    int vocab,
    int token_base,
    float* __restrict__ logits
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    if (row >= vocab) return;

    const uint4* g4 = (const uint4*)(weight + (long long)row * hidden_dim * 2);
    const float* x = hidden_states + (long long)token_base * hidden_dim;

    float acc[BT];
    #pragma unroll
    for (int t = 0; t < BT; ++t) acc[t] = 0.0f;

    for (int base = 0; base < hidden_dim; base += 32 * 8) {
        int j = base + lane * 8;
        uint4 p = g4[(base >> 3) + lane];
        float w0 = widen_f16(p.x);
        float w1 = widen_f16(p.x >> 16);
        float w2 = widen_f16(p.y);
        float w3 = widen_f16(p.y >> 16);
        float w4 = widen_f16(p.z);
        float w5 = widen_f16(p.z >> 16);
        float w6 = widen_f16(p.w);
        float w7 = widen_f16(p.w >> 16);

        #pragma unroll
        for (int t = 0; t < BT; ++t) {
            const float* xt = x + (long long)t * hidden_dim + j;
            float4 xa = *(const float4*)xt;
            float4 xb = *(const float4*)(xt + 4);
            acc[t] += w0 * xa.x + w1 * xa.y + w2 * xa.z + w3 * xa.w
                    + w4 * xb.x + w5 * xb.y + w6 * xb.z + w7 * xb.w;
        }
    }

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

#define LM_HEAD_F16_ENTRY(NAME, BT)                                         \
extern "C" __global__ void NAME(                                            \
    const unsigned char* __restrict__ weight,                               \
    const float* __restrict__ hidden_states,                                \
    int hidden_dim,                                                         \
    int vocab,                                                              \
    int token_base,                                                         \
    float* __restrict__ logits                                              \
) {                                                                         \
    lm_head_rows_f16<BT>(weight, hidden_states, hidden_dim, vocab,          \
                         token_base, logits);                               \
}

LM_HEAD_F16_ENTRY(lm_head_f16_b1, 1)
LM_HEAD_F16_ENTRY(lm_head_f16_b2, 2)
LM_HEAD_F16_ENTRY(lm_head_f16_b3, 3)
LM_HEAD_F16_ENTRY(lm_head_f16_b4, 4)
LM_HEAD_F16_ENTRY(lm_head_f16_b5, 5)
LM_HEAD_F16_ENTRY(lm_head_f16_b6, 6)
LM_HEAD_F16_ENTRY(lm_head_f16_b7, 7)
LM_HEAD_F16_ENTRY(lm_head_f16_b8, 8)

// ---------------------------------------------------------------------------
// The same GEMV over Q4_0, Q4_K and Q5_K weight tensors.
// ---------------------------------------------------------------------------
//
// These three are what an ordinary community quant is built from. `Q4_K_M` in
// particular is a *mixture* — Q4_K for most projections, Q5_K and Q6_K for the
// ones llama.cpp's heuristics protect — so a file of that name needs all of
// these bodies plus the Q6_K one above, not any single reader.
//
// All three are fallbacks, written the same way as the Q6_K body and for the
// same reason: their block strides (18, 144 and 176 bytes) do not put a warp's
// weight read on a sector boundary the way Q8_0's staging pass arranges, so
// they take the format's own indexing instead and accept the coalescing that
// gives.
//
// # The affine formats need their rounding pinned
//
// Q6_K and Q8_0 reconstruct with multiplications only, so the compiler has no
// addition to contract and the device value is bit-identical to the scalar
// reference for free. Q4_K and Q5_K are *affine* — `d*sc*q - dmin*m` — and
// `nvcc` will happily contract that subtraction into an `fma(d1, q, -m1)`,
// which rounds once where the reference rounds twice. That is a real
// divergence in the weight itself, not in the sum, and it would turn an exact
// agreement into a fuzzy one for no gain.
//
// `k_affine_value` pins it with `mul.rn` and `sub.rn` in inline PTX. Inline
// PTX rather than `__fmul_rn`/`__fsub_rn` because NVRTC compiles from a string
// with no include path — the same constraint that makes `load_half_le` use
// `cvt.f32.f16` — and the two instructions were going to be issued anyway, so
// this costs nothing but the compiler's freedom to be wrong.
__device__ __forceinline__ float k_affine_value(float d1, int code, float m1) {
    float c = (float)code;
    float t, r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(t) : "f"(d1), "f"(c));
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(t), "f"(m1));
    return r;
}

// The packed 6-bit sub-scale/sub-min reader shared by Q4_K and Q5_K.
//
// A transcription of `get_scale_min_k4` in `ggml-quants.c`. Eight pairs in
// twelve bytes, and the packing is not uniform: pairs 0-3 are the low 6 bits
// of `q[j]` and `q[j+4]`, pairs 4-7 take a low nibble from `q[j+4]` and borrow
// a high bit-pair from `q[j-4]` (scale) or `q[j]` (min). Reading the second
// branch as if it were the first yields scales up to 4x too small — finite,
// plausible, and invisible except as a quietly worse model.
__device__ __forceinline__ void get_scale_min_k4(
    int j, const unsigned char* __restrict__ q, int* sc, int* m
) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m  = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m  = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4);
    }
}

// Q4_0: 32 elements per 18-byte block, one fp16 delta, codes centred by -8.
//
// A warp covers two blocks (64 elements) per step: lane L takes block `L>>4`
// of the pair and byte `L&15` of it. The two nibbles of that byte are elements
// `j` and `j + 16` of the block — **not** adjacent outputs, which is the one
// way this format is easy to get wrong. `hidden_dim` must be a multiple of 64;
// the geometry check already requires a multiple of 512.
template <int BT>
__device__ __forceinline__ void lm_head_rows_q4_0(
    const unsigned char* __restrict__ weight,
    const float* __restrict__ hidden_states,
    int hidden_dim,
    int vocab,
    int token_base,
    float* __restrict__ logits
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    if (row >= vocab) return;

    int nblocks = hidden_dim >> 5;
    const unsigned char* w = weight + (long long)row * nblocks * 18;
    const float* x = hidden_states + (long long)token_base * hidden_dim;

    float acc[BT];
    #pragma unroll
    for (int t = 0; t < BT; ++t) acc[t] = 0.0f;

    int sub = lane >> 4;        // which block of the pair
    int j   = lane & 15;        // which byte within it

    for (int b = 0; b < nblocks; b += 2) {
        const unsigned char* blk = w + (long long)(b + sub) * 18;
        float d = load_half_le(blk);
        unsigned int q = blk[2 + j];

        // Operand order `q * d`, as in `dequantize_row_q4_0` and every other
        // Q8_0-family unpack here; `d * q` rounds differently.
        float w0 = (float)((int)(q & 0xFu) - 8) * d;
        float w1 = (float)((int)(q >> 4) - 8) * d;

        int i = ((b + sub) << 5) + j;
        #pragma unroll
        for (int t = 0; t < BT; ++t) {
            const float* xt = x + (long long)t * hidden_dim + i;
            acc[t] += w0 * xt[0] + w1 * xt[16];
        }
    }

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

// Q4_K (144 B) and Q5_K (176 B): 256 elements as four 64-element passes, each
// pass split into a low-nibble half under sub-scale `2g` and a high-nibble
// half under `2g+1`.
//
// One lane per `l`, so a lane owns eight elements per superblock at flat
// positions `64g + 32*sub + l`. Byte `qs[32g + l]` carries the two of them
// that share `g`.
//
// `HIGH` selects Q5_K, whose fifth bit lives in a separate 32-byte plane that
// is **not advanced between passes**: all four index `qh[l]` and it is the bit
// position `2g + sub` that moves. The reference spells this as two masks
// shifted left by two each pass, which is the same statement made less
// directly.
template <bool HIGH, int BT>
__device__ __forceinline__ void lm_head_rows_k_affine(
    const unsigned char* __restrict__ weight,
    const float* __restrict__ hidden_states,
    int hidden_dim,
    int vocab,
    int token_base,
    float* __restrict__ logits
) {
    const int SB = HIGH ? 176 : 144;
    const int QS = HIGH ? 48 : 16;   // byte offset of `qs` within a superblock

    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    if (row >= vocab) return;

    int nsb = hidden_dim >> 8;
    const unsigned char* w = weight + (long long)row * nsb * SB;
    const float* x = hidden_states + (long long)token_base * hidden_dim;

    float acc[BT];
    #pragma unroll
    for (int t = 0; t < BT; ++t) acc[t] = 0.0f;

    for (int sb = 0; sb < nsb; ++sb) {
        const unsigned char* base = w + (long long)sb * SB;
        float d    = load_half_le(base);
        float dmin = load_half_le(base + 2);
        const unsigned char* scales = base + 4;
        const unsigned char* qh     = base + 16;   // Q5_K only
        const unsigned char* qs     = base + QS;

        unsigned int hbits = HIGH ? qh[lane] : 0u;

        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            unsigned int byte = qs[g * 32 + lane];
            #pragma unroll
            for (int sub = 0; sub < 2; ++sub) {
                int js = 2 * g + sub;
                int sc, m;
                get_scale_min_k4(js, scales, &sc, &m);
                float d1 = d * (float)sc;
                float m1 = dmin * (float)m;

                int code = (int)(sub ? (byte >> 4) : (byte & 0xFu));
                if (HIGH) code |= (int)((hbits >> js) & 1u) << 4;

                float wv = k_affine_value(d1, code, m1);
                int i = (sb << 8) + g * 64 + sub * 32 + lane;
                #pragma unroll
                for (int t = 0; t < BT; ++t) {
                    acc[t] += wv * x[(long long)t * hidden_dim + i];
                }
            }
        }
    }

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

#define LM_HEAD_Q4_0_ENTRY(NAME, BT)                                        \
extern "C" __global__ void NAME(                                            \
    const unsigned char* __restrict__ weight,                               \
    const float* __restrict__ hidden_states,                                \
    int hidden_dim, int vocab, int token_base,                              \
    float* __restrict__ logits                                              \
) {                                                                         \
    lm_head_rows_q4_0<BT>(weight, hidden_states, hidden_dim, vocab,         \
                          token_base, logits);                              \
}

#define LM_HEAD_KAFF_ENTRY(NAME, HIGH, BT)                                  \
extern "C" __global__ void NAME(                                            \
    const unsigned char* __restrict__ weight,                               \
    const float* __restrict__ hidden_states,                                \
    int hidden_dim, int vocab, int token_base,                              \
    float* __restrict__ logits                                              \
) {                                                                         \
    lm_head_rows_k_affine<HIGH, BT>(weight, hidden_states, hidden_dim,      \
                                    vocab, token_base, logits);             \
}

LM_HEAD_Q4_0_ENTRY(lm_head_q4_0_b1, 1)
LM_HEAD_Q4_0_ENTRY(lm_head_q4_0_b2, 2)
LM_HEAD_Q4_0_ENTRY(lm_head_q4_0_b3, 3)
LM_HEAD_Q4_0_ENTRY(lm_head_q4_0_b4, 4)
LM_HEAD_Q4_0_ENTRY(lm_head_q4_0_b5, 5)
LM_HEAD_Q4_0_ENTRY(lm_head_q4_0_b6, 6)
LM_HEAD_Q4_0_ENTRY(lm_head_q4_0_b7, 7)
LM_HEAD_Q4_0_ENTRY(lm_head_q4_0_b8, 8)

LM_HEAD_KAFF_ENTRY(lm_head_q4_k_b1, false, 1)
LM_HEAD_KAFF_ENTRY(lm_head_q4_k_b2, false, 2)
LM_HEAD_KAFF_ENTRY(lm_head_q4_k_b3, false, 3)
LM_HEAD_KAFF_ENTRY(lm_head_q4_k_b4, false, 4)
LM_HEAD_KAFF_ENTRY(lm_head_q4_k_b5, false, 5)
LM_HEAD_KAFF_ENTRY(lm_head_q4_k_b6, false, 6)
LM_HEAD_KAFF_ENTRY(lm_head_q4_k_b7, false, 7)
LM_HEAD_KAFF_ENTRY(lm_head_q4_k_b8, false, 8)

LM_HEAD_KAFF_ENTRY(lm_head_q5_k_b1, true, 1)
LM_HEAD_KAFF_ENTRY(lm_head_q5_k_b2, true, 2)
LM_HEAD_KAFF_ENTRY(lm_head_q5_k_b3, true, 3)
LM_HEAD_KAFF_ENTRY(lm_head_q5_k_b4, true, 4)
LM_HEAD_KAFF_ENTRY(lm_head_q5_k_b5, true, 5)
LM_HEAD_KAFF_ENTRY(lm_head_q5_k_b6, true, 6)
LM_HEAD_KAFF_ENTRY(lm_head_q5_k_b7, true, 7)
LM_HEAD_KAFF_ENTRY(lm_head_q5_k_b8, true, 8)

// ---------------------------------------------------------------------------
// The same GEMV over a Q6_K weight tensor.
// ---------------------------------------------------------------------------
//
// Neither shipped file stores a tensor this kernel family is asked for as
// Q6_K — both put their projections and their head at Q8_0 or bf16, and Q6_K
// appears only in the expert stacks, which `moe.rs` owns. This body exists for
// the *other* files: the ordinary community quant of either architecture is
// uniform, so `token_embd.weight` and every projection arrive as Q6_K and the
// model previously stopped at the first one with "is q6_K, this pass unpacks
// q8_0".
//
// It is a fallback and is written as one. There is no staging pass, no row
// tile and no prefetch — a 210-byte superblock stride defeats the alignment
// trick the Q8_0 body is built around, and the file that needs this path is
// not the file whose numbers are in docs/BENCHMARKS.md. What it has instead is
// the indexing the format already wants: one lane per `l`, which is exactly
// how `dequantize_row_q6_K` walks a superblock, so every load below is a
// 32-lane contiguous byte read and the 6-bit unpacking is the reference's
// arithmetic with nothing rearranged.
//
// A warp covers one 128-element half per step and two halves per superblock,
// so `hidden_dim` must be a multiple of 256. GGUF enforces that already —
// `ne[0] % blck_size != 0` is refused at parse — and `with_row_tile` requires
// a multiple of 512 on top, so the check in `forward` is a guard rail rather
// than a live branch.
//
// Operand order is `(d * scale) * q`, the same as `dequant.rs`'s
// `dequantize_q6_k` and `xabe_kernels::quant::dequantize_q6_k`. Reassociating
// to `d * (scale * q)` is mathematically equal, rounds differently, and would
// cost the weight-level bit-identity for nothing.
template <int BT>
__device__ __forceinline__ void lm_head_rows_q6_k(
    const unsigned char* __restrict__ weight,
    const float* __restrict__ hidden_states,
    int hidden_dim,
    int vocab,
    int token_base,
    float* __restrict__ logits
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    if (row >= vocab) return;

    int nsb = hidden_dim >> 8;
    const unsigned char* w = weight + (long long)row * nsb * 210;
    const float* x = hidden_states + (long long)token_base * hidden_dim;

    float acc[BT];
    #pragma unroll
    for (int t = 0; t < BT; ++t) acc[t] = 0.0f;

    // `is` selects which of the two per-16-element scales in a lane's quarter
    // applies, and depends only on the lane, so it is hoisted out of both
    // loops.
    int is = lane >> 4;

    for (int sb = 0; sb < nsb; ++sb) {
        const unsigned char* base = w + (long long)sb * 210;
        // One broadcast load of the superblock delta for all 256 elements.
        float d = load_half_le(base + 208);

        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const unsigned char* ql = base + half * 64;
            const unsigned char* qh = base + 128 + half * 32;
            const signed char*   sc = (const signed char*)(base + 192 + half * 8);

            // Three contiguous 32-byte warp reads. The four codes a lane owns
            // are interleaved across the half at l, l+32, l+64, l+96 — the
            // format's own layout, not a choice made here.
            unsigned int ql0 = ql[lane];
            unsigned int ql1 = ql[lane + 32];
            unsigned int qhb = qh[lane];

            int raw1 = (int)((ql0 & 0xFu) | ((qhb & 3u) << 4));
            int raw2 = (int)((ql1 & 0xFu) | (((qhb >> 2) & 3u) << 4));
            int raw3 = (int)((ql0 >> 4)   | (((qhb >> 4) & 3u) << 4));
            int raw4 = (int)((ql1 >> 4)   | (((qhb >> 6) & 3u) << 4));

            float w0 = d * (float)sc[is]     * (float)(raw1 - 32);
            float w1 = d * (float)sc[is + 2] * (float)(raw2 - 32);
            float w2 = d * (float)sc[is + 4] * (float)(raw3 - 32);
            float w3 = d * (float)sc[is + 6] * (float)(raw4 - 32);

            int j = (sb << 8) + half * 128 + lane;
            #pragma unroll
            for (int t = 0; t < BT; ++t) {
                const float* xt = x + (long long)t * hidden_dim + j;
                acc[t] += w0 * xt[0] + w1 * xt[32] + w2 * xt[64] + w3 * xt[96];
            }
        }
    }

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

#define LM_HEAD_Q6_K_ENTRY(NAME, BT)                                        \
extern "C" __global__ void NAME(                                            \
    const unsigned char* __restrict__ weight,                               \
    const float* __restrict__ hidden_states,                                \
    int hidden_dim,                                                         \
    int vocab,                                                              \
    int token_base,                                                         \
    float* __restrict__ logits                                              \
) {                                                                         \
    lm_head_rows_q6_k<BT>(weight, hidden_states, hidden_dim, vocab,         \
                          token_base, logits);                              \
}

LM_HEAD_Q6_K_ENTRY(lm_head_q6_k_b1, 1)
LM_HEAD_Q6_K_ENTRY(lm_head_q6_k_b2, 2)
LM_HEAD_Q6_K_ENTRY(lm_head_q6_k_b3, 3)
LM_HEAD_Q6_K_ENTRY(lm_head_q6_k_b4, 4)
LM_HEAD_Q6_K_ENTRY(lm_head_q6_k_b5, 5)
LM_HEAD_Q6_K_ENTRY(lm_head_q6_k_b6, 6)
LM_HEAD_Q6_K_ENTRY(lm_head_q6_k_b7, 7)
LM_HEAD_Q6_K_ENTRY(lm_head_q6_k_b8, 8)

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

// Softmax probability of the row's maximum: p = 1 / sum(exp(x - max)).
// One block per call; two grid-stride passes over the row (max, then the
// stabilized exp-sum). The 2 MB the second pass rereads is noise next to
// the GEMM that produced the row.
extern "C" __global__ void argmax_prob(
    const float* __restrict__ x,
    int n,
    float* __restrict__ out
) {
    __shared__ float sv[ARGMAX_THREADS / 32];

    float bv = x[0];
    int   bi = 0;
    for (int i = threadIdx.x; i < n; i += ARGMAX_THREADS) {
        argmax_merge(bv, bi, x[i], i);
    }
    argmax_reduce_block(bv, bi);
    if (threadIdx.x == 0) sv[0] = bv;
    __syncthreads();
    const float row_max = sv[0];
    __syncthreads();

    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += ARGMAX_THREADS) {
        acc += expf(x[i] - row_max);
    }
    // warp then cross-warp sum
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if ((threadIdx.x & 31) == 0) sv[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int w = 0; w < ARGMAX_THREADS / 32; ++w) total += sv[w];
        out[0] = 1.0f / total;
    }
}
"#;

/// How a tensor this kernel family reads is stored in the GGUF.
///
/// Both models put their per-layer projections and their LM head through this
/// same GEMV, and the two files do not agree on the format:
/// `Qwen3.6-35B-A3B-UD-Q6_K_XL` stores all of them Q8_0, while
/// `Qwen3.8-27B-UD-Q8_K_XL` stores `output.weight`, every `attn_q`/`attn_k`/
/// `attn_v` and `nextn.eh_proj` as bf16 and everything else Q8_0. So the
/// format travels with the pointer, read out of the file's own tensor
/// directory, rather than being a property of the kernel or of the model.
///
/// Neither shipped file uses Q6_K here; the ordinary *uniform* community
/// quant of either architecture does, for every tensor including the head,
/// and that is who [`Self::Q6K`] is for. It takes the fallback body — no
/// staging, no row tile — and it does not reach the split-int8 tensor-core
/// repack at all, so a file quantized that way trades prefill throughput for
/// loading. See `docs/MODEL.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadFormat {
    /// 32 quants and one fp16 delta per 34-byte block.
    Q8_0,
    /// A dense 2-byte truncated fp32. Widening is a shift, so it is exact.
    Bf16,
    /// "K-quant" 6-bit: 256 quants, 16 int8 group scales and one fp16
    /// super-scale per 210-byte superblock.
    Q6K,
    /// A dense IEEE half. Unlike bf16 the widening is a real conversion, so
    /// it needs `cvt.f32.f16` rather than a shift.
    F16,
    /// Legacy 4-bit: 32 codes and one fp16 delta per 18-byte block, centred
    /// by -8. The two nibbles of a byte are elements `j` and `j + 16`.
    Q4_0,
    /// "K-quant" 4-bit, **affine**: 256 codes over eight 32-element groups,
    /// each with a packed 6-bit scale *and min*, over two fp16 super-scales.
    Q4K,
    /// "K-quant" 5-bit: as [`Self::Q4K`] plus a separate high-bit plane.
    Q5K,
}

impl HeadFormat {
    /// Serialized bytes per weight element.
    pub const fn bytes_per_element(self) -> f64 {
        match self {
            Self::Q8_0 => BLOCK_Q8_0_BYTES as f64 / QK8_0 as f64,
            Self::Bf16 => 2.0,
            Self::Q6K => BLOCK_Q6_K_BYTES as f64 / QK_K as f64,
            Self::F16 => 2.0,
            Self::Q4_0 => BLOCK_Q4_0_BYTES as f64 / QK4_0 as f64,
            Self::Q4K => BLOCK_Q4_K_BYTES as f64 / QK_K as f64,
            Self::Q5K => BLOCK_Q5_K_BYTES as f64 / QK_K as f64,
        }
    }

    /// The format a GGUF type code names, or `None` if this kernel family
    /// does not read it.
    pub const fn from_ggml(name: &str) -> Option<Self> {
        // `GgmlType` lives in `xabe-gguf`, which `xabe-cuda` does not depend
        // on, so the mapping is by name at the one call site that has both.
        match name.as_bytes() {
            b"q8_0" => Some(Self::Q8_0),
            b"bf16" => Some(Self::Bf16),
            b"q6_K" => Some(Self::Q6K),
            b"f16" => Some(Self::F16),
            b"q4_0" => Some(Self::Q4_0),
            b"q4_K" => Some(Self::Q4K),
            b"q5_K" => Some(Self::Q5K),
            _ => None,
        }
    }
}

/// A weight tensor for [`LmHeadKernels::forward`], with its storage format.
#[derive(Clone, Copy)]
pub struct HeadTensor<'a> {
    /// The tensor exactly as it appears in the file.
    pub bytes: &'a CudaSlice<u8>,
    /// How to unpack it.
    pub format: HeadFormat,
}

impl<'a> HeadTensor<'a> {
    /// A Q8_0 tensor — the common case, and what every `qwen35moe` tensor is.
    pub fn q8_0(bytes: &'a CudaSlice<u8>) -> Self {
        Self {
            bytes,
            format: HeadFormat::Q8_0,
        }
    }

    /// A bf16 tensor.
    pub fn bf16(bytes: &'a CudaSlice<u8>) -> Self {
        Self {
            bytes,
            format: HeadFormat::Bf16,
        }
    }

    /// A Q6_K tensor — what a uniform community quant stores these as.
    pub fn q6_k(bytes: &'a CudaSlice<u8>) -> Self {
        Self {
            bytes,
            format: HeadFormat::Q6K,
        }
    }

    /// An f16 tensor.
    pub fn f16(bytes: &'a CudaSlice<u8>) -> Self {
        Self {
            bytes,
            format: HeadFormat::F16,
        }
    }
}

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

    /// Serialized bytes per vocabulary row **at Q8_0** (2,176 at the real
    /// geometry).
    ///
    /// Qwen3.6's head is Q8_0; Qwen3.8's is bf16, for which the figure is
    /// `hidden * 2`. Use [`Self::row_bytes_for`] where the format is not
    /// already known to be Q8_0 — a GB/s number computed with the wrong one
    /// is off by 1.88x.
    pub const fn row_bytes(&self) -> usize {
        self.hidden / QK8_0 * BLOCK_Q8_0_BYTES
    }

    /// Serialized bytes per vocabulary row at `format`.
    pub const fn row_bytes_for(&self, format: HeadFormat) -> usize {
        match format {
            HeadFormat::Q8_0 => self.row_bytes(),
            HeadFormat::Bf16 => self.hidden * 2,
            HeadFormat::Q6K => self.hidden / QK_K * BLOCK_Q6_K_BYTES,
            HeadFormat::F16 => self.hidden * 2,
            HeadFormat::Q4_0 => self.hidden / QK4_0 * BLOCK_Q4_0_BYTES,
            HeadFormat::Q4K => self.hidden / QK_K * BLOCK_Q4_K_BYTES,
            HeadFormat::Q5K => self.hidden / QK_K * BLOCK_Q5_K_BYTES,
        }
    }

    /// Serialized bytes of the whole head at Q8_0 (540,344,320 at the real
    /// geometry). See [`Self::row_bytes`].
    pub const fn weight_bytes(&self) -> usize {
        self.vocab * self.row_bytes()
    }

    /// Serialized bytes of the whole head at `format`.
    pub const fn weight_bytes_for(&self, format: HeadFormat) -> usize {
        self.vocab * self.row_bytes_for(format)
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
    /// `hidden` is not a whole number of the named body's warp step.
    ///
    /// Structurally unreachable behind the multiple-of-512 geometry check,
    /// and kept because that check is about the Q8_0 staging pass and could
    /// reasonably be relaxed for a format that does not stage.
    RaggedContraction {
        format: &'static str,
        hidden: usize,
        step: usize,
    },
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
            Self::RaggedContraction {
                format,
                hidden,
                step,
            } => write!(
                f,
                "the {format} GEMV covers {step} elements per warp step, so `hidden` must be a \
                 multiple of {step}, not {hidden}",
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
    /// The same, over a bf16 weight tensor.
    bf16_tiles: [CudaFunction; MAX_BATCH_TILE],
    /// The same, over a Q6_K weight tensor. No row tile: the fallback body
    /// has none.
    q6_k_tiles: [CudaFunction; MAX_BATCH_TILE],
    /// The same, over an f16 weight tensor. No row tile, for the same reason.
    f16_tiles: [CudaFunction; MAX_BATCH_TILE],
    /// The three community-quant bodies, likewise untiled.
    q4_0_tiles: [CudaFunction; MAX_BATCH_TILE],
    q4_k_tiles: [CudaFunction; MAX_BATCH_TILE],
    q5_k_tiles: [CudaFunction; MAX_BATCH_TILE],
    /// The bf16 row-tiled three-token entry point, paired with
    /// [`Self::b3_row_tile`]'s row count.
    bf16_b3_row_tile: Option<CudaFunction>,
    /// The row-tiled three-token entry point and its row count, or `None`
    /// for the untiled path.
    ///
    /// [`LmHeadKernels::new`] always takes the four-row tile;
    /// [`LmHeadKernels::with_row_tile`] is what reaches the others, and only
    /// the differential test does.
    b3_row_tile: Option<(CudaFunction, usize)>,
    argmax_partial: CudaFunction,
    argmax_final: CudaFunction,
    argmax_prob: CudaFunction,
    geometry: LmHeadGeometry,
}

impl LmHeadKernels {
    /// Compile and validate for `geometry`.
    ///
    /// The geometry is checked once here so the launch path has nothing left
    /// to reject, matching `MoeKernels::new` and `GdnKernels::new`.
    pub fn new(ctx: &Arc<CudaContext>, geometry: LmHeadGeometry) -> Result<Self, LmHeadError> {
        // The four-row tile is the fastest correct point of the isolated
        // sweep — 1.116 ms against RT=2's 1.376 and the untiled 1.662 at
        // three tokens — and a 4.5--4.8% whole-pass N=3 win in all three
        // interleaved 2K pairs (docs/BENCHMARKS.md).
        Self::with_row_tile(ctx, geometry, Some(4))
    }

    /// [`Self::new`] with the three-token row tile chosen explicitly rather
    /// than from the environment: `None` is the untiled path, `Some(2)` and
    /// `Some(4)` the compiled row tiles.
    ///
    /// Public so the differential test can gate every compiled path in one
    /// process without mutating the environment under other threads.
    pub fn with_row_tile(
        ctx: &Arc<CudaContext>,
        geometry: LmHeadGeometry,
        row_tile: Option<usize>,
    ) -> Result<Self, LmHeadError> {
        let bad = |reason: &'static str| LmHeadError::UnsupportedGeometry {
            geometry: Box::new(geometry),
            reason,
        };
        if let Some(rt) = row_tile
            && rt != 2
            && rt != 4
        {
            return Err(bad("the only compiled row tiles are 2 and 4"));
        }
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
        // The row tile only serves whole groups of RT adjacent rows; a vocab
        // that is not a multiple of RT would leave the last group partially
        // live, which the kernel does not guard. The real 248,320-entry head
        // is a multiple of both tiles.
        let b3_row_tile = match row_tile {
            Some(rt) if geometry.vocab.is_multiple_of(rt) => {
                let name = if rt == 4 {
                    "lm_head_gemv_b3r4"
                } else {
                    "lm_head_gemv_b3r2"
                };
                Some((module.load_function(name)?, rt))
            }
            _ => None,
        };
        let bf16_b3_row_tile = match &b3_row_tile {
            Some((_, 4)) => Some(module.load_function("lm_head_bf16_b3r4")?),
            Some((_, _)) => Some(module.load_function("lm_head_bf16_b3r2")?),
            None => None,
        };
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
            bf16_tiles: [
                module.load_function("lm_head_bf16_b1")?,
                module.load_function("lm_head_bf16_b2")?,
                module.load_function("lm_head_bf16_b3")?,
                module.load_function("lm_head_bf16_b4")?,
                module.load_function("lm_head_bf16_b5")?,
                module.load_function("lm_head_bf16_b6")?,
                module.load_function("lm_head_bf16_b7")?,
                module.load_function("lm_head_bf16_b8")?,
            ],
            q6_k_tiles: [
                module.load_function("lm_head_q6_k_b1")?,
                module.load_function("lm_head_q6_k_b2")?,
                module.load_function("lm_head_q6_k_b3")?,
                module.load_function("lm_head_q6_k_b4")?,
                module.load_function("lm_head_q6_k_b5")?,
                module.load_function("lm_head_q6_k_b6")?,
                module.load_function("lm_head_q6_k_b7")?,
                module.load_function("lm_head_q6_k_b8")?,
            ],
            f16_tiles: [
                module.load_function("lm_head_f16_b1")?,
                module.load_function("lm_head_f16_b2")?,
                module.load_function("lm_head_f16_b3")?,
                module.load_function("lm_head_f16_b4")?,
                module.load_function("lm_head_f16_b5")?,
                module.load_function("lm_head_f16_b6")?,
                module.load_function("lm_head_f16_b7")?,
                module.load_function("lm_head_f16_b8")?,
            ],
            q4_0_tiles: [
                module.load_function("lm_head_q4_0_b1")?,
                module.load_function("lm_head_q4_0_b2")?,
                module.load_function("lm_head_q4_0_b3")?,
                module.load_function("lm_head_q4_0_b4")?,
                module.load_function("lm_head_q4_0_b5")?,
                module.load_function("lm_head_q4_0_b6")?,
                module.load_function("lm_head_q4_0_b7")?,
                module.load_function("lm_head_q4_0_b8")?,
            ],
            q4_k_tiles: [
                module.load_function("lm_head_q4_k_b1")?,
                module.load_function("lm_head_q4_k_b2")?,
                module.load_function("lm_head_q4_k_b3")?,
                module.load_function("lm_head_q4_k_b4")?,
                module.load_function("lm_head_q4_k_b5")?,
                module.load_function("lm_head_q4_k_b6")?,
                module.load_function("lm_head_q4_k_b7")?,
                module.load_function("lm_head_q4_k_b8")?,
            ],
            q5_k_tiles: [
                module.load_function("lm_head_q5_k_b1")?,
                module.load_function("lm_head_q5_k_b2")?,
                module.load_function("lm_head_q5_k_b3")?,
                module.load_function("lm_head_q5_k_b4")?,
                module.load_function("lm_head_q5_k_b5")?,
                module.load_function("lm_head_q5_k_b6")?,
                module.load_function("lm_head_q5_k_b7")?,
                module.load_function("lm_head_q5_k_b8")?,
            ],
            bf16_b3_row_tile,
            b3_row_tile,
            argmax_partial: module.load_function("argmax_partial")?,
            argmax_final: module.load_function("argmax_final")?,
            argmax_prob: module.load_function("argmax_prob")?,
            geometry,
        })
    }

    /// The geometry this instance was compiled for.
    pub fn geometry(&self) -> LmHeadGeometry {
        self.geometry
    }

    /// `logits[t][v] = sum_h weight[v][h] * hidden_states[t][h]`.
    ///
    /// `weight` is the raw tensor from the GGUF file in whichever of the
    /// formats [`HeadFormat`] names, `hidden_states`
    /// is `[max_tokens][hidden]` and `logits` is `[max_tokens][vocab]`; rows
    /// past `tokens` in either are neither read nor written.
    ///
    /// A batch of `tokens` costs [`LmHeadGeometry::weight_passes`] passes
    /// over the 540 MB tensor, not `tokens` passes — see the module docs.
    pub fn forward(
        &self,
        stream: &Arc<CudaStream>,
        weight: HeadTensor<'_>,
        hidden_states: &CudaSlice<f32>,
        tokens: usize,
        logits: &mut CudaSlice<f32>,
    ) -> Result<(), LmHeadError> {
        let g = self.geometry;
        let found = match weight.format {
            HeadFormat::Q8_0 => {
                if !weight.bytes.len().is_multiple_of(BLOCK_Q8_0_BYTES) {
                    return Err(LmHeadError::RaggedWeights {
                        bytes: weight.bytes.len(),
                        block_bytes: BLOCK_Q8_0_BYTES,
                    });
                }
                weight.bytes.len() / BLOCK_Q8_0_BYTES * QK8_0
            }
            HeadFormat::Bf16 => {
                if !g.hidden.is_multiple_of(BF16_WARP_STEP) {
                    return Err(LmHeadError::RaggedContraction {
                        format: "bf16",
                        hidden: g.hidden,
                        step: BF16_WARP_STEP,
                    });
                }
                if !weight.bytes.len().is_multiple_of(2) {
                    return Err(LmHeadError::RaggedWeights {
                        bytes: weight.bytes.len(),
                        block_bytes: 2,
                    });
                }
                weight.bytes.len() / 2
            }
            HeadFormat::Q6K => {
                if !g.hidden.is_multiple_of(Q6_K_WARP_STEP) {
                    return Err(LmHeadError::RaggedContraction {
                        format: "Q6_K",
                        hidden: g.hidden,
                        step: Q6_K_WARP_STEP,
                    });
                }
                if !weight.bytes.len().is_multiple_of(BLOCK_Q6_K_BYTES) {
                    return Err(LmHeadError::RaggedWeights {
                        bytes: weight.bytes.len(),
                        block_bytes: BLOCK_Q6_K_BYTES,
                    });
                }
                weight.bytes.len() / BLOCK_Q6_K_BYTES * QK_K
            }
            HeadFormat::F16 => {
                // Same 256-element warp step as the bf16 body it mirrors.
                if !g.hidden.is_multiple_of(BF16_WARP_STEP) {
                    return Err(LmHeadError::RaggedContraction {
                        format: "f16",
                        hidden: g.hidden,
                        step: BF16_WARP_STEP,
                    });
                }
                if !weight.bytes.len().is_multiple_of(2) {
                    return Err(LmHeadError::RaggedWeights {
                        bytes: weight.bytes.len(),
                        block_bytes: 2,
                    });
                }
                weight.bytes.len() / 2
            }
            HeadFormat::Q4_0 => {
                // A warp covers two 32-element blocks per step.
                if !g.hidden.is_multiple_of(2 * QK4_0) {
                    return Err(LmHeadError::RaggedContraction {
                        format: "Q4_0",
                        hidden: g.hidden,
                        step: 2 * QK4_0,
                    });
                }
                if !weight.bytes.len().is_multiple_of(BLOCK_Q4_0_BYTES) {
                    return Err(LmHeadError::RaggedWeights {
                        bytes: weight.bytes.len(),
                        block_bytes: BLOCK_Q4_0_BYTES,
                    });
                }
                weight.bytes.len() / BLOCK_Q4_0_BYTES * QK4_0
            }
            HeadFormat::Q4K | HeadFormat::Q5K => {
                let block_bytes = if weight.format == HeadFormat::Q4K {
                    BLOCK_Q4_K_BYTES
                } else {
                    BLOCK_Q5_K_BYTES
                };
                if !g.hidden.is_multiple_of(QK_K) {
                    return Err(LmHeadError::RaggedContraction {
                        format: if weight.format == HeadFormat::Q4K {
                            "Q4_K"
                        } else {
                            "Q5_K"
                        },
                        hidden: g.hidden,
                        step: QK_K,
                    });
                }
                if !weight.bytes.len().is_multiple_of(block_bytes) {
                    return Err(LmHeadError::RaggedWeights {
                        bytes: weight.bytes.len(),
                        block_bytes,
                    });
                }
                weight.bytes.len() / block_bytes * QK_K
            }
        };
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
            // The three-token tile is the N=3 batched decode shape; it goes
            // through the row-tiled entry point when one was selected at
            // construction. A warp then owns RT adjacent rows and one
            // activation `float4` feeds all of them, which is the lever the
            // module docs name against the activation-pipe cost that scales
            // with BT. Row arithmetic is identical, so the logits are
            // bit-identical to the untiled path.
            // Only the Q8_0 body stages; the others read global memory
            // directly, and asking for a staging buffer no lane touches would
            // cost them occupancy for nothing.
            let staged = weight.format == HeadFormat::Q8_0;
            let cfg = if staged {
                cfg
            } else {
                LaunchConfig {
                    shared_mem_bytes: 0,
                    ..cfg
                }
            };
            // Q6_K has no row-tiled instantiation — it is the fallback body,
            // and a row tile is an optimization for the formats the shipped
            // files actually use.
            let row_tiled = self.b3_row_tile.as_ref().filter(|_| {
                tile == 3 && matches!(weight.format, HeadFormat::Q8_0 | HeadFormat::Bf16)
            });
            let (func, cfg) = match row_tiled {
                Some((f, rt)) => {
                    let rows_per_block = WARPS_PER_BLOCK as usize * rt;
                    let rt_cfg = LaunchConfig {
                        grid_dim: (g.vocab.div_ceil(rows_per_block) as u32, 1, 1),
                        block_dim: (WARP, WARPS_PER_BLOCK, 1),
                        shared_mem_bytes: if staged {
                            (WARPS_PER_BLOCK as usize * rt * STAGE_BLOCKS * BLOCK_Q8_0_BYTES) as u32
                        } else {
                            0
                        },
                    };
                    let func = match (weight.format, self.bf16_b3_row_tile.as_ref()) {
                        (HeadFormat::Bf16, Some(b)) => b,
                        (HeadFormat::Bf16, None) => {
                            unreachable!("the bf16 row tile is loaded exactly when the Q8_0 one is")
                        }
                        _ => f,
                    };
                    (func, rt_cfg)
                }
                None => match weight.format {
                    HeadFormat::Q8_0 => (&self.tiles[tile - 1], cfg),
                    HeadFormat::Bf16 => (&self.bf16_tiles[tile - 1], cfg),
                    HeadFormat::Q6K => (&self.q6_k_tiles[tile - 1], cfg),
                    HeadFormat::F16 => (&self.f16_tiles[tile - 1], cfg),
                    HeadFormat::Q4_0 => (&self.q4_0_tiles[tile - 1], cfg),
                    HeadFormat::Q4K => (&self.q4_k_tiles[tile - 1], cfg),
                    HeadFormat::Q5K => (&self.q5_k_tiles[tile - 1], cfg),
                },
            };
            let mut builder = stream.launch_builder(func);
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

    /// Softmax probability of `values[..n]`'s maximum, written to `out[0]`:
    /// `1 / sum(exp(x - max))`, the confidence a greedy drafter's `p_min`
    /// gate compares against. One block, two passes over the row — noise
    /// next to the GEMM that produced it, and launched only when a gate is
    /// actually configured.
    pub fn argmax_prob(
        &self,
        stream: &Arc<CudaStream>,
        values: &CudaSlice<f32>,
        n: usize,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), LmHeadError> {
        if n == 0 || n > values.len() {
            return Err(LmHeadError::ShapeMismatch {
                what: "argmax_prob length",
                expected: values.len(),
                got: n,
            });
        }
        check_len("argmax_prob output", 1, out.len())?;
        let n_i32 = n as i32;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (ARGMAX_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&self.argmax_prob);
        builder.arg(values).arg(&n_i32).arg(&mut *out);
        // SAFETY: both grid-stride loops are bounded by `n`, checked against
        // `values.len()`, and the write is the single `f32` checked above.
        unsafe { builder.launch(cfg) }?;
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
        assert!(LM_HEAD_SRC.contains("int row = (blockIdx.x * blockDim.y + threadIdx.y) * RT;"));
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
        assert!(body.contains("g4[rt] = (const uint4*)(weight"));
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
            LM_HEAD_SRC.contains("xv[t] = *(const float4*)(x + (long long)t * hidden_dim + j);")
        );
        // And the row tile's whole point: the float4 loads sit *outside* the
        // row loop, so RT rows' FMAs feed from the same registers and the
        // activation read does not scale with RT.
        assert!(
            LM_HEAD_SRC
                .find("xv[t] = *(const float4*)")
                .expect("xv load present")
                < LM_HEAD_SRC
                    .find(
                        "for (int rt = 0; rt < RT; ++rt) {\n                const unsigned char* bp"
                    )
                    .expect("row unpack loop present"),
            "the activation load moved inside the row loop; it would then \
             scale with RT and the tile would buy nothing",
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
    fn the_row_tiled_entry_points_exist_and_fit_the_real_geometry() {
        // The two row tiles the launch path can select. They are separate
        // macro instantiations so the per-tile entry-point count above stays
        // a truthful guard on the untiled set.
        assert!(LM_HEAD_SRC.contains("LM_HEAD_ENTRY_RT(lm_head_gemv_b3r2, 3, 2)"));
        assert!(LM_HEAD_SRC.contains("LM_HEAD_ENTRY_RT(lm_head_gemv_b3r4, 3, 4)"));

        // Whole groups only: the kernel has no partially-live-group guard,
        // so the launch path requires vocab % RT == 0 and the real head
        // satisfies it for both tiles.
        let g = qwen();
        assert!(g.vocab.is_multiple_of(2) && g.vocab.is_multiple_of(4));

        // The staging buffer scales with RT; at RT=4 it must still sit under
        // the 48 KiB a block may request without the opt-in cudarc's
        // LaunchConfig does not expose.
        let shared_rt4 = WARPS_PER_BLOCK as usize * 4 * STAGE_BLOCKS * BLOCK_Q8_0_BYTES;
        assert_eq!(shared_rt4, 17_408);
        assert!(shared_rt4 <= 48 * 1024);
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
        // The ragged-contraction message names the format it is about, so a
        // Q6_K tensor with a bad `hidden` cannot be misread as a bf16 one.
        assert!(
            LmHeadError::RaggedContraction {
                format: "Q6_K",
                hidden: 100,
                step: QK_K,
            }
            .to_string()
            .contains("Q6_K"),
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
