//! Turing integer tensor cores: `mma.sync.aligned.m8n8k16.s32.s8.s8.s32`.
//!
//! The instruction the prefill argument turns on. `docs/OPTIMIZATION.md` §4.1
//! shows llama.cpp's prefill throughput sitting at ~61% of this engine's
//! absolute *fp32* ceiling: fp32 cannot win prefill, and no amount of tuning
//! changes that. Integer tensor cores are the only path, and this module is
//! the primitive they are built on.
//!
//! # Why int8 and not fp16
//!
//! Measured on this card: `m8n8k16` s8 runs at ~198 TOP/s against `m16n8k8`
//! fp16's ~99 TFLOP/s and scalar fp32's ~17.9 TFLOP/s. Twice the fp16 rate is
//! the smaller reason. The larger one is that the weights are **already**
//! quantized — the model ships Q6_K and Q8_0, so an int8 operand is the native
//! format, while an fp16 operand would mean dequantizing to a type *wider*
//! than the data carries.
//!
//! # What is reachable, verified against SASS rather than NVRTC
//!
//! `m8n8k16.s32.s8.s8.s32` and `m16n8k8.f32.f16.f16.f32` both assemble at
//! `sm_75`. **`m16n8k16` does not**: NVRTC accepts it at `compute_75` and
//! emits PTX, then ptxas rejects it (`Feature '.m16n8k16' requires .target
//! sm_80 or higher`). NVRTC success is not evidence of reachability. Anything
//! wanting a wider K decomposes into `m8n8k16` steps sharing one accumulator,
//! which is what llama.cpp's `mma.cuh`, TurboMind's `core/mma.h` and vLLM's
//! `marlin_mma.h` all do.
//!
//! # The fragment layout, and why it is gated exactly
//!
//! A wrong fragment layout is *the* characteristic defect of a hand-written
//! MMA port: it produces finite, plausible, entirely wrong numbers, and no
//! tolerance separates it from arithmetic noise.
//!
//! So this kernel is gated on **bit-exact equality** with
//! `xabe_kernels::mma::int8_gemm` — possible only because integer addition
//! is associative, so the device and the reference may sum the same products
//! in different orders and must still agree to the last bit. Every fp32 kernel
//! in this workspace accepts a tolerance for exactly the reason this one does
//! not.
//!
//! Layout, from the PTX ISA, with lane `l` in `0..32`:
//!
//! ```text
//!   A (8x16 row-major)   row = l >> 2,  columns (l & 3) * 4 + {0,1,2,3}
//!   B (16x8 col-major)   col = l >> 2,  rows    (l & 3) * 4 + {0,1,2,3}
//!   C/D (8x8)            row = l >> 2,  columns (l & 3) * 2 + {0,1}
//! ```
//!
//! The accumulator's stride is 2 and the operands' is 4. Assuming they match
//! is the easy mistake, and `xabe_kernels::mma` spells all three in Rust so a
//! host test can check the tiling covers each element exactly once.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};

use super::compile;

/// Rows of the output tile, fixed by the instruction.
pub const MMA_M: usize = 8;
/// Columns of the output tile, fixed by the instruction.
pub const MMA_N: usize = 8;
/// Contraction elements consumed per instruction, fixed by the instruction.
pub const MMA_K: usize = 16;

const MMA_SRC: &str = r#"
extern "C" {

// Two little-endian bytes as an IEEE half, widened. NVRTC has no include path,
// so <cuda_fp16.h> is unreachable; `cvt.f32.f16` is the same hardware
// conversion `__half2float` lowers to.
// Assemble four bytes into an MMA operand register.
//
// A Q8_0 block is 34 bytes -- a 2-byte scale then 32 quants -- so the quants
// start at an odd offset and a block's stride is not a multiple of 4. A
// `*(const unsigned int*)` load of a weight fragment is therefore *never*
// guaranteed aligned and faults with CUDA_ERROR_MISALIGNED_ADDRESS. The
// activation stream has no such problem (plain int8, offsets are multiples of
// 4), so only the weight side needs this.
__device__ __forceinline__ unsigned int pack4(const unsigned char* p) {
    return (unsigned int)p[0]
         | ((unsigned int)p[1] << 8)
         | ((unsigned int)p[2] << 16)
         | ((unsigned int)p[3] << 24);
}

__device__ __forceinline__ float load_half_le_mma(const unsigned char* p) {
    unsigned short bits = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(bits));
    return f;
}


// d[m][n] = sum_k a[m][k] * b[n][k], int8 operands, int32 accumulate.
//
// `a` is [M][K] row-major and `b` is [N][K] row-major -- b is the transpose of
// the mathematical right operand, which is what `.col` means and what every
// quantized weight layout in this project already stores.
//
// One warp per output tile. grid.x tiles the N axis, grid.y the M axis, so the
// launch shape is a function of the declared geometry alone and stays
// capturable in a CUDA graph (AGENTS.md rule 5).
//
// K must be a multiple of 16. The caller checks it; a kernel-side check would
// have to be a branch on a host value in the inner loop.
// Quantize activations to int8, one scale per 32-element block.
//
// This is the operand the model does not already provide. Weights ship as
// Q8_0 -- int8 with an fp16 scale per 32 elements -- so they feed the MMA
// directly; activations are fp32 and must be narrowed to match.
//
// Symmetric, absmax, round-to-nearest, exactly llama.cpp's `quantize_q8_1`
// shape minus the sum term this formulation does not need (no zero point, so
// no correction term). One warp per (row, block): 32 lanes, one element each.
__global__ void mma_quantize_rows_q8(
    const float* __restrict__ x,
    signed char* __restrict__ q,
    float* __restrict__ scales,
    int k_dim
) {
    // Row on x, contraction block on y -- deliberately, and not the other way
    // round. grid.y and grid.z are capped at 65,535 while grid.x is capped at
    // 2^31-1, and `rows` here is the MoE dispatch *slot* count, which is
    // `max_tokens * top_k` plus padding. Putting it on y capped the model's
    // context at about 7,168 tokens, with the failure surfacing as a bare
    // CUDA_ERROR_INVALID_VALUE from the launch.
    int row   = blockIdx.x;
    int block = blockIdx.y * blockDim.y + threadIdx.y;
    int lane  = threadIdx.x;
    int blocks = k_dim / 32;
    if (block >= blocks) return;

    long long base = (long long)row * k_dim + block * 32;
    float v = x[base + lane];

    // Absmax across the warp. `fmaxf` of absolute values, butterfly so every
    // lane ends with the same scale and none has to broadcast.
    float a = fabsf(v);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, off));

    // A block of exact zeros has no scale; 1.0 keeps the dequantization
    // well-defined and every quant is zero anyway.
    float d = a > 0.0f ? a / 127.0f : 1.0f;
    float inv = a > 0.0f ? 127.0f / a : 0.0f;

    // rintf, not truncation: truncation biases every magnitude downward and
    // the bias survives the sum over k, which a symmetric rounding does not.
    int qi = (int)rintf(v * inv);
    qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);

    q[base + lane] = (signed char)qi;
    if (lane == 0) scales[(long long)row * blocks + block] = d;
}

// out[t][n] = sum_k w[n][k] * x[t][k], with `w` a Q8_0 GGUF tensor and `x`
// pre-quantized by `mma_quantize_rows_q8`, computed on integer tensor cores.
//
// The scales cannot be folded into the operands -- they would not fit in int8
// -- so they are applied *after* each 32-element block's int32 accumulation,
// which is exactly where llama.cpp's MMQ applies them. One Q8_0 block is two
// m8n8k16 steps, and both share the same pair of scales, so the scaling costs
// 2 FMA per 4096 FLOP.
//
// Fragment roles: A is the activation tile (MMA's M axis is tokens), B is the
// weight tile (N axis is output rows). So lane l accumulates
// D[token t0 + (l>>2)][rows n0 + (l&3)*2 + {0,1}].
//
// A warp owns 8 tokens and MMA_ROWS output rows, so the activation fragment is
// loaded once and reused across every row tile -- the same reuse the fp32
// projection needed, for the same reason.
#define MMA_ROWS 64
#define MMA_TOKS 32

__global__ void mma_q8_0_proj(
    const unsigned char* __restrict__ weight,
    const signed char* __restrict__ xq,
    const float* __restrict__ xs,
    float* __restrict__ out,
    int k_dim,
    int n_rows,
    int n_tokens
) {
    int lane = threadIdx.x;
    int t0   = blockIdx.y * 8;
    int n0   = (blockIdx.x * blockDim.y + threadIdx.y) * MMA_ROWS;
    if (t0 >= n_tokens || n0 >= n_rows) return;

    int blocks = k_dim / 32;
    int trow = t0 + (lane >> 2);          // this lane's token, for A and D
    int quad = (lane & 3) * 4;            // its 4 consecutive k elements
    int dcol = (lane & 3) * 2;            // its 2 output rows within a tile

    const int tiles = MMA_ROWS / 8;
    float facc[tiles][2];
#pragma unroll
    for (int r = 0; r < tiles; ++r) { facc[r][0] = 0.0f; facc[r][1] = 0.0f; }

    bool live_t = trow < n_tokens;

    for (int b = 0; b < blocks; ++b) {
        // The activation half-blocks, loaded once for all MMA_ROWS rows.
        unsigned int a0 = 0, a1 = 0;
        float dx = 0.0f;
        if (live_t) {
            const signed char* xp = xq + (long long)trow * k_dim + b * 32 + quad;
            a0 = *(const unsigned int*)(xp);
            a1 = *(const unsigned int*)(xp + 16);
            dx = xs[(long long)trow * blocks + b];
        }

#pragma unroll
        for (int r = 0; r < tiles; ++r) {
            int nb = n0 + r * 8;
            // The lane's B fragment row, and the two D rows it accumulates.
            int brow = nb + (lane >> 2);
            unsigned int b0 = 0, b1 = 0;
            if (brow < n_rows) {
                const unsigned char* blk = weight + (long long)brow * blocks * 34 + b * 34;
                b0 = pack4(blk + 2 + quad);
                b1 = pack4(blk + 2 + 16 + quad);
            }

            int acc0 = 0, acc1 = 0;
            asm volatile(
                "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                "{%0,%1}, {%2}, {%3}, {%0,%1};"
                : "+r"(acc0), "+r"(acc1) : "r"(a0), "r"(b0));
            asm volatile(
                "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                "{%0,%1}, {%2}, {%3}, {%0,%1};"
                : "+r"(acc0), "+r"(acc1) : "r"(a1), "r"(b1));

            // Weight scales for the two output rows this lane holds. Read
            // after the MMA so the loads overlap the tensor-core latency.
            int r0 = nb + dcol, r1 = r0 + 1;
            float dw0 = 0.0f, dw1 = 0.0f;
            if (r0 < n_rows) dw0 = load_half_le_mma(weight + (long long)r0 * blocks * 34 + b * 34);
            if (r1 < n_rows) dw1 = load_half_le_mma(weight + (long long)r1 * blocks * 34 + b * 34);

            facc[r][0] += (float)acc0 * dx * dw0;
            facc[r][1] += (float)acc1 * dx * dw1;
        }
    }

    if (!live_t) return;
#pragma unroll
    for (int r = 0; r < tiles; ++r) {
        int r0 = n0 + r * 8 + dcol, r1 = r0 + 1;
        if (r0 < n_rows) out[(long long)trow * n_rows + r0] = facc[r][0];
        if (r1 < n_rows) out[(long long)trow * n_rows + r1] = facc[r][1];
    }
}

// The same projection over a *repacked* weight layout.
//
// `mma_q8_0_proj` reads GGUF Q8_0 in place, and that costs it everything: a
// block is 34 bytes -- a 2-byte scale then 32 quants -- so a fragment's four
// bytes are never word-aligned and must be assembled with four scalar loads.
// Counted per Q8_0 block per warp at MMA_ROWS = 32, that is ~48 memory
// instructions for 8 MMA instructions, and the tensor cores sit idle behind
// the load unit. Measured: 2.0 TOP/s, 1% of this part's ~198 TOP/s peak, and
// slower than the fp32 kernel it was meant to replace.
//
// Splitting the scales out of the quants makes every operand load aligned and
// contiguous: one `ld.global.u32` per fragment and one `ld.global.f32` per
// scale. This is why llama.cpp's MMQ, TurboMind and vLLM's Marlin all carry
// their own packed weight layouts rather than reading the on-disk format --
// the repack is a one-time cost at load and the alignment is worth it every
// pass afterwards.
//
//   quants: int8  [n_rows][k]
//   scales: fp32  [n_rows][k / 32]   (`mma_repack_q8_0`)
//   scales: fp16  [n_rows][k / 32]   (`mma_repack_q8_0_h`)
// Split a Q8_0 tensor into aligned quants and per-32 scales.
//
// One warp per 32-element block: each lane moves one quant, lane 0 writes the
// scale. Run once at load; see `mma_q8_0_proj_split` for why the on-disk
// layout cannot be read directly at speed.
//
// **Which width the scales get is a bandwidth decision, not a numeric one.**
// Q8_0 stores an fp16 scale per 32 quants, so the fp32 form widens a value
// that has no more precision to give: 1.125 bytes an element against Q8_0's
// own 1.0625, which is 5.9% more traffic for nothing. That is noise where the
// repack covers a shared expert (3.5 MB in a layer of 725) and it is not
// noise where it covers a 17,408-wide dense FFN -- there the same 5.9% is
// 1.01 GiB of every decoded token. `_h` keeps the file's own fp16 bits
// verbatim, so it is neither a requantization nor a rounding.
__global__ void mma_repack_q8_0(
    const unsigned char* __restrict__ src,
    signed char* __restrict__ q,
    float* __restrict__ scales,
    long long n_blocks
) {
    long long b = (long long)blockIdx.x * blockDim.y + threadIdx.y;
    if (b >= n_blocks) return;
    int lane = threadIdx.x;
    const unsigned char* blk = src + b * 34;
    q[b * 32 + lane] = (signed char)blk[2 + lane];
    if (lane == 0) scales[b] = load_half_le_mma(blk);
}

__global__ void mma_repack_q8_0_h(
    const unsigned char* __restrict__ src,
    signed char* __restrict__ q,
    unsigned short* __restrict__ scales,
    long long n_blocks
) {
    long long b = (long long)blockIdx.x * blockDim.y + threadIdx.y;
    if (b >= n_blocks) return;
    int lane = threadIdx.x;
    const unsigned char* blk = src + b * 34;
    q[b * 32 + lane] = (signed char)blk[2 + lane];
    // The two source bytes, moved rather than converted. GGUF is
    // little-endian and so is the device, so this is the identity on the
    // value as well as on the bits.
    if (lane == 0) scales[b] = (unsigned short)blk[0] | ((unsigned short)blk[1] << 8);
}

// Contraction the block stages per trip, and the padded row stride it stages
// into.
//
// The inner loop used to read its B fragment straight from global on every
// 32-element block: four warps sharing one row band meant four fetches of the
// same bytes, and every one of them was a dependent global load in a kernel
// that only fits eight warps to an SM. L1 caught most of the traffic and none
// of the latency.
//
// Staged, each weight byte is fetched once per block and the mainloop reads
// shared. `PROJ_WSTRIDE` is 144 rather than 128 because the fragment load is
// `row = lane >> 2`, four consecutive bytes per lane: at a 128-byte stride
// that is 32 words, every row lands on the same bank, and the access is an
// eight-way conflict. 144 bytes is 36 words and `36 mod 32 == 4`, which tiles
// the 32 banks exactly -- the same rule `kernels::moe` documents.
#define PROJ_KC      128
#define PROJ_WSTRIDE 144

extern __shared__ signed char xabe_proj_shared[];

}  // extern "C" -- a template cannot have C linkage, and the body below is
   // one so that the two scale widths share it. The `extern "C" __global__`
   // entry points that instantiate it are declared individually and keep
   // their unmangled names; `extern "C" {` reopens after them.

// One weight scale, from either width. The fp16 form is `mma_repack_q8_0_h`'s
// output: the file's own bits, so this widens and does not round.
__device__ __forceinline__ float xabe_proj_scale(const float* p, long long i) {
    return p[i];
}
__device__ __forceinline__ float xabe_proj_scale(const unsigned short* p, long long i) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(p[i]));
    return f;
}

template <typename ST>
__device__ __forceinline__ void mma_q8_0_proj_split_body(
    const signed char* __restrict__ wq,
    const ST* __restrict__ ws,
    const signed char* __restrict__ xq,
    const float* __restrict__ xs,
    float* __restrict__ out,
    int k_dim,
    int n_rows,
    int n_tokens
) {
    int lane = threadIdx.x;
    // The warps of a block walk **the same weight rows** and different tokens,
    // not the other way round.
    //
    // Every block in `y` re-reads the whole weight band -- that is the shape
    // of this kernel and the reason the token tile was swept end to end rather
    // than in isolation. What the sweep could not change is that the warps of
    // a block used to take *different* row bands, so four warps meant four
    // disjoint weight reads and the block shared nothing. Giving them one row
    // band and four token tiles makes three of the four reads L1 hits and
    // divides the band's DRAM traffic by `blockDim.y` -- at 512 tokens, eight
    // passes over the weights become two.
    int t0   = (blockIdx.y * blockDim.y + threadIdx.y) * MMA_TOKS;
    int n0   = blockIdx.x * MMA_ROWS;
    // No early return: the staging loop below is a block-wide barrier pair and
    // a warp that left would deadlock the rest. `t0 >= n_tokens` is instead
    // masked at the epilogue, and `n0 >= n_rows` cannot happen because the
    // grid is `ceil(n_rows / MMA_ROWS)`.
    if (n0 >= n_rows) return;

    int blocks = k_dim / 32;
    int quad = (lane & 3) * 4;
    int dcol = (lane & 3) * 2;
    int trow = lane >> 2;                 // token within a tile, for A and D
    int brow = lane >> 2;                 // row within a tile, for B

    const int ntile = MMA_ROWS / 8;
    const int ttile = MMA_TOKS / 8;

    // Arithmetic intensity is the whole game here. With one token tile a warp
    // does 8 tokens x 2 ops per weight byte = 16 OP/byte, against an int8
    // ridge point of 198e12 / 672e9 = 295 OP/byte -- deeply memory-bound, and
    // measured at 3.2% of peak. Each extra token tile multiplies the ops per
    // weight byte without changing the weight traffic at all, because the same
    // B fragment feeds every token tile.
    float facc[ttile][ntile][2];
#pragma unroll
    for (int t = 0; t < ttile; ++t)
#pragma unroll
        for (int r = 0; r < ntile; ++r) { facc[t][r][0] = 0.0f; facc[t][r][1] = 0.0f; }

    signed char* sw  = xabe_proj_shared;                            // [MMA_ROWS][PROJ_WSTRIDE]
    float*       sws = (float*)(sw + MMA_ROWS * PROJ_WSTRIDE);      // [MMA_ROWS][PROJ_KC/32]
    int tid      = threadIdx.y * 32 + lane;
    int nthreads = blockDim.y * 32;

    for (int kc = 0; kc < k_dim; kc += PROJ_KC) {
        __syncthreads();
        // Sixteen bytes a lane, so a row of the tile is eight instructions for
        // the whole block rather than four per warp per 32-element block.
        for (int idx = tid; idx < MMA_ROWS * (PROJ_KC / 16); idx += nthreads) {
            int r = idx / (PROJ_KC / 16);
            int c = (idx % (PROJ_KC / 16)) * 16;
            int wr = n0 + r;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (wr < n_rows) {
                v = *(const uint4*)(wq + (long long)wr * k_dim + kc + c);
            }
            *(uint4*)(sw + r * PROJ_WSTRIDE + c) = v;
        }
        for (int idx = tid; idx < MMA_ROWS * (PROJ_KC / 32); idx += nthreads) {
            int r = idx / (PROJ_KC / 32);
            int bb = idx % (PROJ_KC / 32);
            int wr = n0 + r;
            // A row past `n_rows` stages a zero scale, which is enough to make
            // every product it feeds exactly zero without zeroing its quants.
            sws[r * (PROJ_KC / 32) + bb] =
                wr < n_rows
                    ? xabe_proj_scale(ws, (long long)wr * blocks + (kc >> 5) + bb)
                    : 0.0f;
        }
        __syncthreads();

#pragma unroll
    for (int bb = 0; bb < PROJ_KC / 32; ++bb) {
        int b = (kc >> 5) + bb;
        // One A fragment pair and one scale per token tile.
        unsigned int a0[ttile], a1[ttile];
        float dx[ttile];
#pragma unroll
        for (int t = 0; t < ttile; ++t) {
            int tr = t0 + t * 8 + trow;
            if (tr < n_tokens) {
                const signed char* xp = xq + (long long)tr * k_dim + b * 32 + quad;
                a0[t] = *(const unsigned int*)(xp);
                a1[t] = *(const unsigned int*)(xp + 16);
                dx[t] = xs[(long long)tr * blocks + b];
            } else {
                a0[t] = 0; a1[t] = 0; dx[t] = 0.0f;
            }
        }

#pragma unroll
        for (int r = 0; r < ntile; ++r) {
            const signed char* wp =
                sw + (r * 8 + brow) * PROJ_WSTRIDE + bb * 32 + quad;
            unsigned int b0 = *(const unsigned int*)(wp);
            unsigned int b1 = *(const unsigned int*)(wp + 16);
            float dw0 = sws[(r * 8 + dcol) * (PROJ_KC / 32) + bb];
            float dw1 = sws[(r * 8 + dcol + 1) * (PROJ_KC / 32) + bb];

            // The B fragment is loaded once and consumed by every token tile.
#pragma unroll
            for (int t = 0; t < ttile; ++t) {
                int acc0 = 0, acc1 = 0;
                asm volatile(
                    "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                    "{%0,%1}, {%2}, {%3}, {%0,%1};"
                    : "+r"(acc0), "+r"(acc1) : "r"(a0[t]), "r"(b0));
                asm volatile(
                    "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                    "{%0,%1}, {%2}, {%3}, {%0,%1};"
                    : "+r"(acc0), "+r"(acc1) : "r"(a1[t]), "r"(b1));
                facc[t][r][0] += (float)acc0 * dx[t] * dw0;
                facc[t][r][1] += (float)acc1 * dx[t] * dw1;
            }
        }
    }
    }

#pragma unroll
    for (int t = 0; t < ttile; ++t) {
        int tr = t0 + t * 8 + trow;
        if (tr >= n_tokens) continue;
#pragma unroll
        for (int r = 0; r < ntile; ++r) {
            int r0 = n0 + r * 8 + dcol, r1 = r0 + 1;
            if (r0 < n_rows) out[(long long)tr * n_rows + r0] = facc[t][r][0];
            if (r1 < n_rows) out[(long long)tr * n_rows + r1] = facc[t][r][1];
        }
    }
}

#define MMA_PROJ_SPLIT_ENTRY(NAME, ST)                                       \
extern "C" __global__ void NAME(                                             \
    const signed char* __restrict__ wq,                                      \
    const ST* __restrict__ ws,                                               \
    const signed char* __restrict__ xq,                                      \
    const float* __restrict__ xs,                                            \
    float* __restrict__ out,                                                 \
    int k_dim,                                                               \
    int n_rows,                                                              \
    int n_tokens                                                             \
) {                                                                          \
    mma_q8_0_proj_split_body<ST>(wq, ws, xq, xs, out, k_dim, n_rows, n_tokens); \
}

MMA_PROJ_SPLIT_ENTRY(mma_q8_0_proj_split, float)
MMA_PROJ_SPLIT_ENTRY(mma_q8_0_proj_split_h, unsigned short)

extern "C" {

__global__ void mma_int8_gemm(
    const signed char* __restrict__ a,
    const signed char* __restrict__ b,
    int* __restrict__ d,
    int m, int n, int k
) {
    int tile_n = blockIdx.x * 8;
    int tile_m = blockIdx.y * 8;
    int lane = threadIdx.x;

    // Operand slots. The `& 3` group indexes four *consecutive* contraction
    // elements, which is what makes each lane's load a single 4-byte access
    // rather than four scattered ones.
    int a_row = tile_m + (lane >> 2);
    int b_col = tile_n + (lane >> 2);
    int elem0 = (lane & 3) * 4;

    // The accumulator's column stride is 2, not 4. This is the asymmetry the
    // module docs warn about.
    int d_row = tile_m + (lane >> 2);
    int d_col = tile_n + (lane & 3) * 2;

    int acc0 = 0, acc1 = 0;

    for (int k0 = 0; k0 < k; k0 += 16) {
        // Rows past the end contribute zero rather than reading out of bounds.
        // Zero is the identity for this accumulation, so a padded tile is
        // arithmetically the same as a smaller one.
        unsigned int af = 0, bf = 0;
        if (a_row < m) {
            const signed char* p = a + (long long)a_row * k + k0 + elem0;
            af = *(const unsigned int*)p;
        }
        if (b_col < n) {
            const signed char* p = b + (long long)b_col * k + k0 + elem0;
            bf = *(const unsigned int*)p;
        }

        asm volatile(
            "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
            "{%0,%1}, {%2}, {%3}, {%0,%1};"
            : "+r"(acc0), "+r"(acc1)
            : "r"(af), "r"(bf)
        );
    }

    if (d_row < m) {
        if (d_col     < n) d[(long long)d_row * n + d_col    ] = acc0;
        if (d_col + 1 < n) d[(long long)d_row * n + d_col + 1] = acc1;
    }
}

}
"#;

/// Something went wrong compiling or launching the integer MMA kernel.
#[derive(Debug)]
pub enum MmaError {
    /// NVRTC rejected the source, or the module failed to load.
    ///
    /// On `sm_75` the likely cause is an instruction that needs `sm_80`. Note
    /// that such a failure surfaces here at *module load*, not at compile:
    /// NVRTC accepts `m16n8k16` and ptxas rejects it.
    Compile(String),
    /// The driver failed.
    Driver(DriverError),
    /// `k` is not a whole number of `MMA_K` steps.
    ///
    /// Rejected rather than padded: padding would need a masked tail load, and
    /// silently truncating would produce a plausible wrong product.
    RaggedContraction { k: usize },
    /// A buffer is not the size the declared shape implies.
    BufferShape {
        what: &'static str,
        expected: usize,
        actual: usize,
    },
}

impl std::fmt::Display for MmaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compile(m) => write!(f, "integer MMA kernel compilation failed: {m}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::RaggedContraction { k } => write!(
                f,
                "contraction length {k} is not a multiple of {MMA_K}, which is the \
                 instruction's K and not a tunable",
            ),
            Self::BufferShape {
                what,
                expected,
                actual,
            } => write!(
                f,
                "{what} holds {actual} elements, the shape implies {expected}"
            ),
        }
    }
}

impl std::error::Error for MmaError {}

impl From<DriverError> for MmaError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

/// Output rows one warp accumulates.
///
/// Spelled here and as `MMA_ROWS` in the kernel;
/// `the_row_band_matches_the_kernel` asserts they agree.
pub const MMA_ROWS: usize = 64;

/// Tokens one warp accumulates — the instruction's M, not a tunable.
pub const MMA_TOKENS: usize = MMA_M;

/// Tokens one warp accumulates in the split-layout projection.
///
/// Unlike [`MMA_TOKENS`] this *is* a tunable. Together with [`MMA_ROWS`] it
/// sets both the arithmetic intensity — one B fragment feeds every token tile,
/// so extra tiles multiply the operations per weight byte without touching the
/// weight traffic — and the register pressure, which is
/// `(MMA_SPLIT_TOKS / 8) * (MMA_ROWS / 8) * 2` accumulators per thread.
///
/// # A tuning result that reversed twice
///
/// This constant was 64 for most of the project's life, and the reason was
/// weight traffic. The token tile divides the grid — `grid.y` is `tokens /
/// MMA_SPLIT_TOKS` and **every block in `y` re-reads the whole weight band** —
/// so a narrow tile multiplied the DRAM traffic. A full sweep at the time,
/// end to end, put 64 ahead of everything:
///
/// | tile | n = 128 | n = 512 |
/// |---|---:|---:|
/// | 64 x 64  | 889.34 | **1341.39** |
/// | 64 x 32  | 933.32 | 1309.01 |
/// | 32 x 64  | 861.89 | 1284.06 |
/// | 16 x 128 | 905.99 | 1228.07 |
/// | 32 x 128 | 684.86 | 1211.75 |
/// | 16 x 64  | 915.18 | 1164.79 |
///
/// and a later sweep under the shared-row-band mapping agreed: `8 x 32 x 64`
/// measured 1,369 against `4 x 64 x 64`'s 1,435.
///
/// **Staging the weight tile in shared memory reversed it.** Once each block
/// fetches the band once, with `uint4` loads, instead of once per warp with
/// scattered four-byte reads, the traffic argument for a wide token tile is
/// gone — and what is left is the register pressure the wide tile was buying
/// its way out of: `(MMA_SPLIT_TOKS / 8) * (MMA_ROWS / 8) * 2` accumulators
/// per thread, 128 of them at 64 x 64, which pins ptxas at 255 registers and
/// eight warps to an SM. Halving it to 64 roughly doubles the resident warps.
///
/// Re-swept end to end with the staged tile, tok/s at n = 512:
///
/// | warps x tokens x rows | tok/s |
/// | --- | ---: |
/// | **4 x 32 x 64** | **2,255.68** |
/// | 2 x 32 x 64 | 2,239.56 |
/// | 4 x 16 x 64 | 2,212.23 |
/// | 8 x 16 x 64 | 2,152.69 |
/// | 4 x 64 x 64 | 2,142.98 |
/// | 4 x 32 x 32 | 2,123.91 |
/// | 4 x 32 x 128 | 2,085.36 |
/// | 8 x 64 x 64 | 2,077.50 |
///
/// The kernel went 53.0 -> 38.4 ms per 512-token pass across the two changes
/// together, and neither is worth much without the other: staging alone was
/// 2.0 ms, and the narrow tile alone had already been measured as a loss,
/// twice.
///
/// Recorded at length because the isolated `bench_mma` benchmark points the
/// wrong way here — a microbenchmark of a kernel that re-reads a weight
/// measures cache residency it will not have in situ — and because a tuning
/// result is only valid against the kernel structure it was measured on.
pub const MMA_SPLIT_TOKS: usize = 32;

/// Contraction the split projection stages per trip. Mirrors `PROJ_KC`.
pub const PROJ_KC: usize = 128;

/// Padded row stride of the staged weight tile, in bytes. Mirrors
/// `PROJ_WSTRIDE`.
///
/// 144 and not `PROJ_KC`: the fragment load is `row = lane >> 2` with four
/// consecutive bytes per lane, so at a 128-byte stride all eight rows of a
/// fragment land on the same bank and the read is an eight-way conflict. 144
/// bytes is 36 words and `36 mod 32 == 4`, which tiles the banks exactly.
const PROJ_WSTRIDE: usize = 144;

/// Shared bytes one split-projection block needs: the staged weight tile plus
/// its per-32 scales.
const fn proj_shared_bytes() -> u32 {
    (MMA_ROWS * PROJ_WSTRIDE + MMA_ROWS * (PROJ_KC / 32) * size_of::<f32>()) as u32
}

/// Batch width at or above which the engine's blocks route their projections
/// through the integer tensor cores.
///
/// Deliberately **not** [`MMA_SPLIT_TOKS`], though it was the same constant
/// until the tile was retuned. They answer different questions: the tile is
/// how much work one warp holds, and this is whether a batch is wide enough to
/// be worth quantizing activations for at all. A decode step of one token
/// would pay the whole fixed cost to fill an eighth of a fragment.
pub const MMA_SPLIT_TOKENS: usize = 64;

/// The compiled integer tensor-core kernels.
pub struct MmaKernels {
    gemm: CudaFunction,
    quantize: CudaFunction,
    proj: CudaFunction,
    proj_split: CudaFunction,
    proj_split_h: CudaFunction,
    repack: CudaFunction,
    repack_h: CudaFunction,
}

impl MmaKernels {
    /// Compile for the context's device.
    ///
    /// A failure here on `sm_75` is the reachability signal that matters: the
    /// module is what ptxas has to accept, and ptxas is stricter than NVRTC.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, MmaError> {
        let ptx = compile(MMA_SRC, "mma_int8").map_err(MmaError::Compile)?;
        let module = ctx.load_module(ptx)?;
        Ok(Self {
            gemm: module.load_function("mma_int8_gemm")?,
            quantize: module.load_function("mma_quantize_rows_q8")?,
            proj: module.load_function("mma_q8_0_proj")?,
            proj_split: module.load_function("mma_q8_0_proj_split")?,
            proj_split_h: module.load_function("mma_q8_0_proj_split_h")?,
            repack: module.load_function("mma_repack_q8_0")?,
            repack_h: module.load_function("mma_repack_q8_0_h")?,
        })
    }

    /// Quantize `x` (`[rows][k]` fp32) to int8 with one scale per 32 elements.
    ///
    /// `q` is `[rows][k]` and `scales` is `[rows][k / 32]`. This is the operand
    /// the model does not already ship: weights are Q8_0 and feed the tensor
    /// cores directly, activations are fp32 and must be narrowed to match.
    ///
    /// **This is the step that costs accuracy**, and it is the same step
    /// llama.cpp takes before its own int8 matmuls. `docs/ORACLE.md` §8 item 0
    /// measures llama.cpp's activation quantization as 1,000-10,000x less
    /// accurate than this engine's fp32 path on a single projection; adopting
    /// it here trades that advantage for the tensor cores it unlocks.
    pub fn quantize_rows(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scales: &mut CudaSlice<f32>,
        rows: usize,
        k: usize,
    ) -> Result<(), MmaError> {
        if !k.is_multiple_of(32) {
            return Err(MmaError::RaggedContraction { k });
        }
        expect_at_least("quantize x", x.len(), rows * k)?;
        expect_at_least("quantize q", q.len(), rows * k)?;
        expect_at_least("quantize scales", scales.len(), rows * k / 32)?;
        if rows == 0 {
            return Ok(());
        }

        const WARPS: u32 = 4;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, (k / 32).div_ceil(WARPS as usize) as u32, 1),
            block_dim: (32, WARPS, 1),
            shared_mem_bytes: 0,
        };
        let k_i = k as i32;
        let mut builder = stream.launch_builder(&self.quantize);
        builder.arg(x).arg(&mut *q).arg(&mut *scales).arg(&k_i);
        // SAFETY: one warp per (row, 32-element block) over a grid covering
        // every block and returning above it, so no lane touches an index
        // beyond `rows * k`, which all three buffers were checked against.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// Split a Q8_0 tensor into aligned quants and fp32 scales, on device.
    ///
    /// Prefer [`Self::repack_q8_0_half`] for anything large: widening the
    /// scale buys no precision and costs 5.9% more traffic on every read.
    ///
    /// `src` is the GGUF byte stream, `q` is `[elements]` int8 and `scales` is
    /// `[elements / 32]` fp32. Run once at load: the repacked form is what
    /// [`Self::q8_0_proj_split`] reads, and the difference between the two
    /// layouts is 3.2x.
    pub fn repack_q8_0(
        &self,
        stream: &Arc<CudaStream>,
        src: &CudaSlice<u8>,
        q: &mut CudaSlice<i8>,
        scales: &mut CudaSlice<f32>,
        elements: usize,
    ) -> Result<(), MmaError> {
        if !elements.is_multiple_of(32) {
            return Err(MmaError::RaggedContraction { k: elements });
        }
        let blocks = elements / 32;
        expect_len("repack src", src.len(), blocks * 34)?;
        expect_len("repack q", q.len(), elements)?;
        expect_len("repack scales", scales.len(), blocks)?;
        if blocks == 0 {
            return Ok(());
        }

        const WARPS: u32 = 4;
        let cfg = LaunchConfig {
            grid_dim: (blocks.div_ceil(WARPS as usize) as u32, 1, 1),
            block_dim: (32, WARPS, 1),
            shared_mem_bytes: 0,
        };
        let n = blocks as i64;
        let mut builder = stream.launch_builder(&self.repack);
        builder.arg(src).arg(&mut *q).arg(&mut *scales).arg(&n);
        // SAFETY: one warp per 32-element block over a grid covering every
        // block and returning above it; all three buffers were checked against
        // the block count.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// As [`Self::repack_q8_0`], with the scales left at the width the file
    /// stores them in.
    ///
    /// `scales` is `[elements / 32]` raw IEEE-half bits — the two bytes at the
    /// head of each Q8_0 block, moved rather than converted, so this is the
    /// identity on the value. Prefer it wherever the repacked tensor is large
    /// enough for its scale array to be traffic: the fp32 form costs 1.125
    /// bytes an element against Q8_0's own 1.0625 and buys no precision, which
    /// is 5.9% of every read of a weight that is only ever read once per pass.
    ///
    /// [`Self::q8_0_proj_split_half`] is the matching consumer.
    pub fn repack_q8_0_half(
        &self,
        stream: &Arc<CudaStream>,
        src: &CudaSlice<u8>,
        q: &mut CudaSlice<i8>,
        scales: &mut CudaSlice<u16>,
        elements: usize,
    ) -> Result<(), MmaError> {
        if !elements.is_multiple_of(32) {
            return Err(MmaError::RaggedContraction { k: elements });
        }
        let blocks = elements / 32;
        expect_len("repack src", src.len(), blocks * 34)?;
        expect_len("repack q", q.len(), elements)?;
        expect_len("repack scales", scales.len(), blocks)?;
        if blocks == 0 {
            return Ok(());
        }

        const WARPS: u32 = 4;
        let cfg = LaunchConfig {
            grid_dim: (blocks.div_ceil(WARPS as usize) as u32, 1, 1),
            block_dim: (32, WARPS, 1),
            shared_mem_bytes: 0,
        };
        let n = blocks as i64;
        let mut builder = stream.launch_builder(&self.repack_h);
        builder.arg(src).arg(&mut *q).arg(&mut *scales).arg(&n);
        // SAFETY: one warp per 32-element block over a grid covering every
        // block and returning above it; all three buffers were checked against
        // the block count.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// As [`Self::q8_0_proj`], over a weight layout with the scales split out.
    ///
    /// `wq` is `[n_rows][k]` int8 and `ws` is `[n_rows][k / 32]` fp32 — the
    /// same numbers a Q8_0 tensor carries, rearranged so every operand load is
    /// aligned and contiguous. See the kernel's comment for why that is worth
    /// a repack.
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_proj_split(
        &self,
        stream: &Arc<CudaStream>,
        wq: &CudaSlice<i8>,
        ws: &CudaSlice<f32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        k: usize,
        n_rows: usize,
        tokens: usize,
    ) -> Result<(), MmaError> {
        // The staged weight tile is `PROJ_KC` wide, so the contraction has to
        // be a whole number of chunks and not merely of MMA steps.
        if !k.is_multiple_of(PROJ_KC) {
            return Err(MmaError::RaggedContraction { k });
        }
        expect_len("split wq", wq.len(), n_rows * k)?;
        expect_len("split ws", ws.len(), n_rows * k / 32)?;
        expect_at_least("split xq", xq.len(), tokens * k)?;
        expect_at_least("split xs", xs.len(), tokens * k / 32)?;
        expect_len("split out", out.len(), tokens * n_rows)?;
        if tokens == 0 || n_rows == 0 {
            return Ok(());
        }

        const WARPS: u32 = 4;
        let cfg = LaunchConfig {
            grid_dim: (
                (n_rows as u32).div_ceil(MMA_ROWS as u32),
                (tokens as u32).div_ceil(WARPS * MMA_SPLIT_TOKS as u32),
                1,
            ),
            block_dim: (32, WARPS, 1),
            shared_mem_bytes: proj_shared_bytes(),
        };
        let (k_i, n_i, t_i) = (k as i32, n_rows as i32, tokens as i32);
        let mut builder = stream.launch_builder(&self.proj_split);
        builder
            .arg(wq)
            .arg(ws)
            .arg(xq)
            .arg(xs)
            .arg(&mut *out)
            .arg(&k_i)
            .arg(&n_i)
            .arg(&t_i);
        // SAFETY: as `q8_0_proj`, with both weight buffers checked against the
        // declared shape and every load and store guarded against `n_rows` and
        // `n_tokens`.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// As [`Self::q8_0_proj_split`], reading [`Self::repack_q8_0_half`]'s
    /// narrower scales.
    ///
    /// Same kernel body, same shared-memory staging — the scales are widened
    /// to fp32 on the way into shared, so only the global read differs and the
    /// mainloop is identical.
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_proj_split_half(
        &self,
        stream: &Arc<CudaStream>,
        wq: &CudaSlice<i8>,
        ws: &CudaSlice<u16>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        k: usize,
        n_rows: usize,
        tokens: usize,
    ) -> Result<(), MmaError> {
        if !k.is_multiple_of(PROJ_KC) {
            return Err(MmaError::RaggedContraction { k });
        }
        expect_len("split wq", wq.len(), n_rows * k)?;
        expect_len("split ws", ws.len(), n_rows * k / 32)?;
        expect_at_least("split xq", xq.len(), tokens * k)?;
        expect_at_least("split xs", xs.len(), tokens * k / 32)?;
        expect_len("split out", out.len(), tokens * n_rows)?;
        if tokens == 0 || n_rows == 0 {
            return Ok(());
        }

        const WARPS: u32 = 4;
        let cfg = LaunchConfig {
            grid_dim: (
                (n_rows as u32).div_ceil(MMA_ROWS as u32),
                (tokens as u32).div_ceil(WARPS * MMA_SPLIT_TOKS as u32),
                1,
            ),
            block_dim: (32, WARPS, 1),
            shared_mem_bytes: proj_shared_bytes(),
        };
        let (k_i, n_i, t_i) = (k as i32, n_rows as i32, tokens as i32);
        let mut builder = stream.launch_builder(&self.proj_split_h);
        builder
            .arg(wq)
            .arg(ws)
            .arg(xq)
            .arg(xs)
            .arg(&mut *out)
            .arg(&k_i)
            .arg(&n_i)
            .arg(&t_i);
        // SAFETY: as `q8_0_proj_split`; the only difference is the width of
        // the scale array, which was length-checked against the same shape.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// `out[t][n] = sum_k weight[n][k] * x[t][k]` on integer tensor cores.
    ///
    /// `weight` is a Q8_0 GGUF tensor of `[n_rows][k]`; `xq` and `xs` are the
    /// output of [`Self::quantize_rows`] over `[tokens][k]`.
    ///
    /// The scales are applied after each 32-element block's int32
    /// accumulation, because they do not fit in int8 — which is where
    /// llama.cpp's MMQ applies them too.
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_proj(
        &self,
        stream: &Arc<CudaStream>,
        weight: &CudaSlice<u8>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        k: usize,
        n_rows: usize,
        tokens: usize,
    ) -> Result<(), MmaError> {
        if !k.is_multiple_of(32) {
            return Err(MmaError::RaggedContraction { k });
        }
        expect_len("proj weight", weight.len(), n_rows * k / 32 * 34)?;
        expect_len("proj xq", xq.len(), tokens * k)?;
        expect_len("proj xs", xs.len(), tokens * k / 32)?;
        expect_len("proj out", out.len(), tokens * n_rows)?;
        if tokens == 0 || n_rows == 0 {
            return Ok(());
        }

        const WARPS: u32 = 4;
        let cfg = LaunchConfig {
            grid_dim: (
                (n_rows as u32).div_ceil(WARPS * MMA_ROWS as u32),
                (tokens as u32).div_ceil(MMA_TOKENS as u32),
                1,
            ),
            block_dim: (32, WARPS, 1),
            shared_mem_bytes: 0,
        };
        let (k_i, n_i, t_i) = (k as i32, n_rows as i32, tokens as i32);
        let mut builder = stream.launch_builder(&self.proj);
        builder
            .arg(weight)
            .arg(xq)
            .arg(xs)
            .arg(&mut *out)
            .arg(&k_i)
            .arg(&n_i)
            .arg(&t_i);
        // SAFETY: the grid covers `n_rows` in bands of `WARPS * MMA_ROWS` and
        // `tokens` in tiles of 8, and every operand load and every store is
        // guarded against both bounds. The weight's last byte for row
        // `n_rows - 1` is `(n_rows - 1) * (k/32) * 34 + 33`, which is the
        // length checked above.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// `d[m][n] = sum_k a[m][k] * b[n][k]`, exactly.
    ///
    /// `a` is `[m][k]` and `b` is `[n][k]`, both row-major and both int8.
    /// `d` is `[m][n]` int32. The result is **bit-exact** against
    /// `xabe_kernels::mma::int8_gemm` — integer addition is associative, so
    /// summation order cannot excuse a disagreement.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self,
        stream: &Arc<CudaStream>,
        a: &CudaSlice<i8>,
        b: &CudaSlice<i8>,
        d: &mut CudaSlice<i32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), MmaError> {
        if !k.is_multiple_of(MMA_K) {
            return Err(MmaError::RaggedContraction { k });
        }
        expect_len("a", a.len(), m * k)?;
        expect_len("b", b.len(), n * k)?;
        expect_len("d", d.len(), m * n)?;

        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(MMA_N) as u32, m.div_ceil(MMA_M) as u32, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        let mut builder = stream.launch_builder(&self.gemm);
        builder
            .arg(a)
            .arg(b)
            .arg(&mut *d)
            .arg(&mi)
            .arg(&ni)
            .arg(&ki);
        // SAFETY: one warp per 8x8 output tile over a grid that covers `m` x
        // `n` and bounds-checks both axes; every operand load is guarded by
        // the same bounds and every buffer was checked against the declared
        // shape above. `k` is a whole number of 16-element steps, so the inner
        // loop reads exactly `k` elements per row.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }
}

/// Like [`expect_len`], for buffers a caller is allowed to over-allocate.
///
/// The activation scratch is sized once for the largest projection a block
/// runs and then reused by smaller ones, so "longer than needed" is its
/// ordinary state. Too small stays fatal: that is the read-past-the-end case.
fn expect_at_least(what: &'static str, actual: usize, needed: usize) -> Result<(), MmaError> {
    if actual >= needed {
        Ok(())
    } else {
        Err(MmaError::BufferShape {
            what,
            expected: needed,
            actual,
        })
    }
}

fn expect_len(what: &'static str, actual: usize, expected: usize) -> Result<(), MmaError> {
    if actual == expected {
        Ok(())
    } else {
        Err(MmaError::BufferShape {
            what,
            expected,
            actual,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_row_axis_of_the_quantizer_is_x_because_only_x_is_unbounded() {
        // `rows` is the MoE dispatch slot count -- `max_tokens * top_k` plus
        // padding -- so it is the one launch dimension in this file that scales
        // with context. grid.y and grid.z stop at 65,535; grid.x stops at
        // 2^31-1. With rows on y the model could not exceed about 7,168 tokens
        // and the failure was an unattributed CUDA_ERROR_INVALID_VALUE, which
        // is why this is asserted rather than left to a comment.
        assert!(MMA_SRC.contains("int row   = blockIdx.x;"));
        assert!(MMA_SRC.contains("int block = blockIdx.y * blockDim.y + threadIdx.y;"));
        assert!(
            !MMA_SRC.contains("int row   = blockIdx.y;"),
            "rows on grid.y caps context at 65,535 slots",
        );
    }

    #[test]
    fn the_kernel_uses_the_turing_shape_and_not_the_ampere_one() {
        // `m16n8k16` compiles under NVRTC at compute_75 and is then rejected
        // by ptxas, so its presence would not be caught until module load on a
        // machine with a device. Caught here instead, on any machine.
        assert!(
            MMA_SRC.contains("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32"),
            "the int8 Turing shape must be the one emitted",
        );
        assert!(
            !MMA_SRC.contains("m16n8k16"),
            "m16n8k16 requires sm_80; it must be decomposed, not emitted",
        );
    }

    #[test]
    fn the_fragment_slots_match_the_rust_side_spelling() {
        // The kernel computes its slots inline; `xabe_kernels::mma` spells the
        // same layout for the host. If the two ever disagree the differential
        // test would fail, but it would fail without saying which side moved.
        assert!(
            MMA_SRC.contains("(lane >> 2)"),
            "row/col split is lane >> 2"
        );
        assert!(MMA_SRC.contains("(lane & 3) * 4"), "operand stride is 4");
        assert!(
            MMA_SRC.contains("(lane & 3) * 2"),
            "accumulator stride is 2, not 4 — this asymmetry is the trap",
        );
    }

    #[test]
    fn the_row_band_matches_the_kernel() {
        assert!(
            MMA_SRC.contains(&format!("#define MMA_ROWS {MMA_ROWS}\n")),
            "the kernel's row band must equal the host's, or the launch grid \
             stops covering the output rows",
        );
        assert!(
            MMA_SRC.contains(&format!("#define MMA_TOKS {MMA_SPLIT_TOKS}\n")),
            "the kernel's token tile must equal the host's, or the launch grid \
             stops covering the tokens",
        );
        assert_eq!(MMA_ROWS % MMA_N, 0, "the band must be whole 8-row tiles");
        assert_eq!(
            MMA_SPLIT_TOKENS % MMA_M,
            0,
            "the token tile must be whole 8-token MMA tiles",
        );
    }

    #[test]
    fn the_activation_quantizer_rounds_rather_than_truncates() {
        // Truncation biases every magnitude toward zero, and the bias survives
        // the sum over k rather than cancelling — a systematic error in every
        // projection, not a rounding one.
        assert!(MMA_SRC.contains("rintf(v * inv)"), "must round to nearest");
        assert!(
            MMA_SRC.contains("qi < -127 ? -127"),
            "must clamp symmetrically; -128 has no positive counterpart and \
             would skew the scale",
        );
    }

    #[test]
    fn the_instruction_geometry_is_not_tunable() {
        // These are properties of the hardware instruction. A test exists so
        // that "tuning" them is a failing build rather than silent garbage.
        assert_eq!((MMA_M, MMA_N, MMA_K), (8, 8, 16));
    }
}
