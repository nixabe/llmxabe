//! Mixture-of-Experts routing and grouped GEMM on sm_75.
//!
//! Qwen3.6 puts a 256-expert, top-8 MoE block on **every one of the 40
//! layers** plus the MTP head. That makes this the single most launch-
//! sensitive path in the engine: the naive form is 8 routed experts x 3
//! matrices + 3 shared-expert matrices = 27 GEMVs per token per layer, or
//! 1,080 launches per decoded token (`MoeConfig::naive_gemvs_per_token` in
//! `xabe-model`). The whole point of the grouped form is to replace that
//! with a fixed, small number of launches whose *shapes do not depend on the
//! routing decision*.
//!
//! ## Four kernels, and why the split falls where it does
//!
//! 1. `moe_route` — softmax over all 256 experts, top-k selection,
//!    renormalization. **On the device.** A host round-trip here would cost
//!    a synchronization per layer per token; at 40 layers that dominates
//!    decode latency regardless of how fast the GEMM is.
//! 2. `moe_align_block_size` — the sorted-token indirection, ported from
//!    `xabe_kernels::moe::dispatch`. Built into **fixed-size** buffers
//!    allocated once, because `AGENTS.md` rule 5 forbids anything sized by a
//!    host-side value on this path: a host-sized allocation cannot be inside
//!    a captured CUDA graph, and graph capture over this indirection is the
//!    project's single largest expected win.
//! 3. `moe_expert_ffn` / `moe_expert_down` — the grouped GEMM proper, with
//!    **Q6_K / Q8_0 dequantization in the prologue**. Materializing all 256
//!    experts in fp32 would turn one layer's 630 MiB of quantized expert
//!    weights into 3.1 GiB, times 41 blocks; the tile being multiplied is
//!    dequantized instead, and nothing is written back in fp32.
//! 4. `moe_reduce` — the fp32 weighted sum of each token's 8 routed
//!    contributions.
//!
//! The **shared expert is hoisted out** into `moe_shared_ffn` /
//! `moe_shared_down`: it is active for every token unconditionally, so it
//! has no routing, no sorting, and no indirection to pay for.
//!
//! ## The value the host is not allowed to know
//!
//! Every kernel here is launched at a grid sized from `MoeGeometry`, which is
//! fixed at construction. The per-step token count lives in a device scalar
//! (`MoeBuffers::valid_tokens`) that the kernels read and early-out on, and
//! `num_tokens_post_pad` — the one genuinely data-dependent size — is
//! written to device memory and never read back on the hot path. That is
//! what makes the launch shapes constant across steps.
//!
//! ## Ported from
//!
//! - `xabe_kernels::moe::router::route_token` for the selection semantics.
//!   The tie-break (equal probability -> lower expert index) is transcribed
//!   exactly; a different tie-break silently runs a different expert.
//! - `xabe_kernels::moe::dispatch::moe_align_block_size` for the tables,
//!   which in turn came from vLLM's `moe_align_block_size`. The padding
//!   sentinel is `num_tokens * top_k` and inactive blocks are `-1`, matching
//!   [`xabe_kernels::moe::dispatch::padding_sentinel`] and `INACTIVE_EXPERT`
//!   exactly. Get the sentinel wrong and a consumer either indexes past the
//!   token array or silently drops tokens.
//! - `kernels::dequant` for the bit-unpacking, including the `load_half_le`
//!   inline-PTX helper and the operand order `(d * scale) * q` that the
//!   milestone-04 gate proved bit-identical to the scalar reference.
//!
//! ## Why the result cannot be bit-identical to the CPU reference
//!
//! The dequantized *weights* are bit-identical (multiplication only). The
//! dot products are not: the reference sums sequentially, the kernel reduces
//! in a warp-shuffle tree, and fp32 addition is not associative. The gate is
//! therefore a tolerance on the GEMM output and exact equality on the
//! routing decision and the dispatch tables, which have no rounding freedom
//! in their *discrete* content.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};

use super::compile;

/// Elements per Q8_0 block, and per k-quant superblock.
///
/// Duplicated from [`super::dequant`] rather than imported so that a change
/// there cannot silently alter this module's alignment validation.
const QK8_0: usize = 32;
const QK_K: usize = 256;
const BLOCK_Q8_0_BYTES: usize = 34;
const BLOCK_Q6_K_BYTES: usize = 210;

/// Threads per block for every kernel in this module.
///
/// A power of two, because the routing reductions are plain shared-memory
/// tree reductions that halve the active range each round.
const THREADS: u32 = 256;

/// Storage format of one expert weight stack.
///
/// The real `Qwen3.6-35B-A3B-UD-Q6_K_XL` file is **mixed**: `ffn_gate_exps`
/// and `ffn_up_exps` are Q6_K, but `ffn_down_exps` is Q8_0 (verified against
/// the file's tensor directory, not assumed). A kernel that hard-coded Q6_K
/// would read the down projection as garbage, so the format travels with the
/// pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertQuant {
    /// 256 elements per 210-byte superblock.
    Q6K,
    /// 32 elements per 34-byte block.
    Q8_0,
}

impl ExpertQuant {
    /// The integer the kernel switches on.
    const fn code(self) -> i32 {
        match self {
            Self::Q6K => 0,
            Self::Q8_0 => 1,
        }
    }

    /// Elements per serialized block.
    pub const fn block_elements(self) -> usize {
        match self {
            Self::Q6K => QK_K,
            Self::Q8_0 => QK8_0,
        }
    }

    /// Serialized bytes per block.
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::Q6K => BLOCK_Q6_K_BYTES,
            Self::Q8_0 => BLOCK_Q8_0_BYTES,
        }
    }
}

/// A quantized weight stack resident on the device, with its format.
#[derive(Clone, Copy)]
pub struct QuantTensor<'a> {
    /// Raw serialized blocks, exactly as they appear in the GGUF file.
    pub bytes: &'a CudaSlice<u8>,
    /// How to unpack them.
    pub quant: ExpertQuant,
}

impl QuantTensor<'_> {
    /// Elements this buffer holds, given its format.
    fn elements(&self) -> usize {
        self.bytes.len() / self.quant.block_bytes() * self.quant.block_elements()
    }

    fn is_whole_blocks(&self) -> bool {
        self.bytes.len().is_multiple_of(self.quant.block_bytes())
    }
}

const MOE_SRC: &str = r#"
extern "C" {

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

// One Q6_K element, addressed by its flat index in the dequantized tensor.
//
// The standalone kernel in `kernels/dequant.rs` assigns one thread to four
// outputs at flat offsets l, l+32, l+64, l+96 within a 128-element half.
// Inverting that mapping: flat index r within a superblock decomposes as
// half = r/128, group = (r%128)/32, l = r%32 — which is what lets a GEMM
// walk a weight row in natural order and still land on the right nibble.
//
// The multiply order `(d * scale) * q` is load-bearing and matches the
// scalar reference; reassociating to `d * (scale * q)` is mathematically
// equal, rounds differently, and would cost bit-identical weights.
__device__ __forceinline__ float q6k_element(const unsigned char* src, long long i) {
    long long sb = i >> 8;
    int r    = (int)(i & 255);
    int half = r >> 7;
    int hr   = r & 127;
    int grp  = hr >> 5;
    int l    = hr & 31;

    const unsigned char* base = src + sb * 210;
    const unsigned char* ql = base + half * 64;
    const unsigned char* qh = base + 128 + half * 32;
    const signed char*   sc = (const signed char*)(base + 192 + half * 8);
    float d = load_half_le(base + 208);

    int is = l >> 4;
    unsigned char h = qh[l];
    int raw, si;
    if (grp == 0)      { raw = (ql[l]      & 0xF) | ((h & 3) << 4);        si = is;     }
    else if (grp == 1) { raw = (ql[l + 32] & 0xF) | (((h >> 2) & 3) << 4); si = is + 2; }
    else if (grp == 2) { raw = (ql[l]      >> 4)  | (((h >> 4) & 3) << 4); si = is + 4; }
    else               { raw = (ql[l + 32] >> 4)  | (((h >> 6) & 3) << 4); si = is + 6; }

    return d * (float)sc[si] * (float)(raw - 32);
}

// One Q8_0 element. Reading the code as `signed char` is load-bearing: the
// quants are int8 on disk, and reading them unsigned flips the sign of
// roughly half of every tensor while leaving magnitudes plausible.
__device__ __forceinline__ float q8_0_element(const unsigned char* src, long long i) {
    long long block = i >> 5;
    int lane = (int)(i & 31);
    const unsigned char* base = src + block * 34;
    float d = load_half_le(base);
    signed char q = (signed char)base[2 + lane];
    return (float)q * d;
}

// The GEMM prologue: unpack exactly the weight element about to be
// multiplied. Nothing is materialized in fp32.
__device__ __forceinline__ float dequant_element(
    const unsigned char* src, int quant, long long i
) {
    return quant == 0 ? q6k_element(src, i) : q8_0_element(src, i);
}

// Shared-memory pool. One declaration, one type, for every kernel below:
// two `extern __shared__` arrays of different element types in the same
// translation unit is a redeclaration error, so integer users cast.
extern __shared__ float xabe_shared[];

// Sum across a block of up to 1024 threads. Warp shuffles first, then one
// round through shared memory. The tree order differs from the reference's
// sequential sum, which is the dominant source of disagreement between this
// kernel and the CPU; see the module docs.
__device__ __forceinline__ float block_reduce_sum(float v, float* scratch) {
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

// -------------------------------------------------------------------------
// 1. Routing: softmax over all experts, top-k, renormalize.
// -------------------------------------------------------------------------
//
// grid: one block per token slot (always `max_tokens`, never the live count).
// block: THREADS threads, strided over the experts.
//
// This transcribes `xabe_kernels::moe::router::route_token`:
//   probs = softmax(logits); rank by (prob desc, index asc); take k;
//   weights = prob / sum(selected probs).
//
// Two details are the whole correctness story:
//
// - **Selection is by probability, not by logit.** Softmax is monotonic so
//   the two orderings agree, but the reference's tie-break fires on equal
//   *probabilities*, and reproducing it means comparing the same quantity.
// - **Ties go to the lower expert index.** Every reduction below carries the
//   index alongside the value for exactly this reason. Without it a tie
//   resolves by whichever thread happened to win the shuffle, which is not
//   even stable across runs.
//
// `sum_exp` is reduced in a tree here and sequentially in the reference, so
// the two differ in the last ulp — but it divides every probability equally,
// so the *ordering* (and therefore the selection) is untouched, and it
// cancels out of the renormalized weights almost exactly.
__global__ void moe_route(
    const float* __restrict__ logits,
    const int* __restrict__ valid_tokens,
    int num_experts,
    int top_k,
    int* __restrict__ topk_ids,
    float* __restrict__ topk_weights
) {
    float* probs = xabe_shared;
    float* rval  = xabe_shared + num_experts;
    int*   ridx  = (int*)(rval + blockDim.x);

    int token = blockIdx.x;
    if (token >= *valid_tokens) return;

    const float* row = logits + (long long)token * num_experts;

    // --- max logit (exact: max is associative and rounds nothing) ---------
    float local = row[0];
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        local = fmaxf(local, row[e]);
    }
    rval[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) rval[threadIdx.x] = fmaxf(rval[threadIdx.x], rval[threadIdx.x + s]);
        __syncthreads();
    }
    float max_logit = rval[0];
    __syncthreads();

    // --- exp, sum, normalize ---------------------------------------------
    float esum = 0.0f;
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        float p = expf(row[e] - max_logit);
        probs[e] = p;
        esum += p;
    }
    rval[threadIdx.x] = esum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) rval[threadIdx.x] += rval[threadIdx.x + s];
        __syncthreads();
    }
    float sum_exp = rval[0];
    __syncthreads();
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        probs[e] = probs[e] / sum_exp;
    }
    __syncthreads();

    // --- k successive argmax passes, ties to the lower index --------------
    //
    // Equivalent to the reference's full sort-then-truncate, and cheaper:
    // k is 8 against 256 experts. A selected expert is masked with -1.0f,
    // which no softmax probability can reach, so it can never be re-picked.
    for (int j = 0; j < top_k; ++j) {
        float bv = -1.0f;
        int   bi = -1;
        for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
            float p = probs[e];
            if (p > bv || (p == bv && (bi < 0 || e < bi))) { bv = p; bi = e; }
        }
        rval[threadIdx.x] = bv;
        ridx[threadIdx.x] = bi;
        __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                float av = rval[threadIdx.x];
                float cv = rval[threadIdx.x + s];
                int   ai = ridx[threadIdx.x];
                int   ci = ridx[threadIdx.x + s];
                if (cv > av || (cv == av && ci >= 0 && (ai < 0 || ci < ai))) {
                    rval[threadIdx.x] = cv;
                    ridx[threadIdx.x] = ci;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            int best = ridx[0];
            topk_ids[(long long)token * top_k + j] = best;
            topk_weights[(long long)token * top_k + j] = rval[0];
            probs[best] = -1.0f;
        }
        __syncthreads();
    }

    // --- renormalize over just the selected k -----------------------------
    //
    // Summed in selection order by a single thread, matching the reference's
    // `ranked.iter().map(|&(_, p)| p).sum()`. k is 8; parallelizing this
    // would only add a reduction-order difference for nothing.
    if (threadIdx.x == 0) {
        long long b = (long long)token * top_k;
        float s = 0.0f;
        for (int j = 0; j < top_k; ++j) s += topk_weights[b + j];
        if (s > 0.0f) {
            for (int j = 0; j < top_k; ++j) topk_weights[b + j] = topk_weights[b + j] / s;
        } else {
            // Unreachable with a real softmax (every entry is > 0), but the
            // reference guards it rather than propagating NaN, so this does
            // too.
            for (int j = 0; j < top_k; ++j) topk_weights[b + j] = 1.0f / (float)top_k;
        }
    }
}

// -------------------------------------------------------------------------
// 2. Sorted-token indirection, into fixed-size buffers.
// -------------------------------------------------------------------------
//
// One block, THREADS threads, one owning thread per expert (strided if there
// are more experts than threads). Single-block on purpose: the exclusive
// prefix sum over per-expert padded counts is a global dependency, and at
// 256 experts a device-wide multi-pass scan costs more in launches than the
// scan costs in arithmetic.
//
// **The ordering guarantee is why the cursor is not an atomic counter.**
// The reference places an expert's tokens in ascending flat `(token, k)`
// index. A cursor bumped atomically gives whatever order the warps happened
// to arrive in, which is not reproducible run to run and would make an exact
// comparison against the reference impossible. Instead each expert's owning
// thread scans the flat array once in ascending order and appends. That
// costs `num_experts / blockDim * numel` iterations per thread — 296 at the
// decode shapes this is written for.
//
// `numel = valid_tokens * top_k` is simultaneously the number of valid flat
// indices and the padding sentinel (`padding_sentinel` in the reference is
// exactly `num_tokens * top_k`), so a consumer's test is `flat >= numel`.
//
// The sentinel and INACTIVE_EXPERT fills cover the **whole capacity**, not
// just `num_tokens_post_pad`. Filling only the live prefix would leave a
// previous, longer step's entries visible past it — tokens that no longer
// exist, pointing into a token array that has since shrunk.
__global__ void moe_align_block_size(
    const int* __restrict__ topk_ids,
    const int* __restrict__ valid_tokens,
    int top_k,
    int num_experts,
    int block_size,
    int sorted_capacity,
    int expert_capacity,
    int* __restrict__ sorted_token_ids,
    int* __restrict__ expert_ids,
    int* __restrict__ num_tokens_post_pad
) {
    int* cumsum = (int*)xabe_shared;
    int numel = (*valid_tokens) * top_k;
    int tid = threadIdx.x;

    // 1. per-expert counts, padded up to a whole number of blocks. An expert
    //    with no tokens gets *zero* blocks, not a padded-empty one — the
    //    reference's `CEILDIV(0, block_size) == 0`.
    for (int e = tid; e < num_experts; e += blockDim.x) {
        int c = 0;
        for (int i = 0; i < numel; ++i) {
            if (topk_ids[i] == e) ++c;
        }
        cumsum[e + 1] = ((c + block_size - 1) / block_size) * block_size;
    }
    __syncthreads();

    // 2. exclusive prefix sum. Sequential in one thread: 256 iterations,
    //    and it is the only ordering the whole kernel depends on.
    if (tid == 0) {
        cumsum[0] = 0;
        for (int e = 0; e < num_experts; ++e) cumsum[e + 1] += cumsum[e];
        *num_tokens_post_pad = cumsum[num_experts];
    }
    __syncthreads();

    // 3. sentinel / inactive fill over the fixed capacity.
    for (int s = tid; s < sorted_capacity; s += blockDim.x) sorted_token_ids[s] = numel;
    for (int b = tid; b < expert_capacity; b += blockDim.x) expert_ids[b] = -1;
    __syncthreads();

    // 4. scatter in ascending flat index, then claim this expert's blocks.
    for (int e = tid; e < num_experts; e += blockDim.x) {
        int slot = cumsum[e];
        for (int i = 0; i < numel; ++i) {
            if (topk_ids[i] == e) {
                if (slot < sorted_capacity) sorted_token_ids[slot] = i;
                ++slot;
            }
        }
        int first = cumsum[e] / block_size;
        int last  = cumsum[e + 1] / block_size;
        for (int b = first; b < last && b < expert_capacity; ++b) expert_ids[b] = e;
    }
}

// -------------------------------------------------------------------------
// 3. Grouped GEMM: gate/up + SwiGLU, then down.
// -------------------------------------------------------------------------
//
// grid: (intermediate, sorted_capacity). One block per (output row, slot).
// The grid is the *capacity*, never `num_tokens_post_pad` — that value lives
// only on the device. Slots past the live region carry INACTIVE_EXPERT and
// slots inside it that are padding carry the sentinel; both exit after one
// or two loads. The predicate is uniform across the block, so the early
// `return` never strands a `__syncthreads()`.
__global__ void moe_expert_ffn(
    const unsigned char* __restrict__ gate_q, int gate_quant,
    const unsigned char* __restrict__ up_q,   int up_quant,
    const float* __restrict__ hidden_states,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids,
    const int* __restrict__ valid_tokens,
    int top_k,
    int block_size,
    int hidden,
    int intermediate,
    float* __restrict__ inter
) {
    int r    = blockIdx.x;
    int slot = blockIdx.y;

    int e = expert_ids[slot / block_size];
    if (e < 0) return;
    int numel = (*valid_tokens) * top_k;
    int flat = sorted_token_ids[slot];
    if (flat >= numel) return;

    int token = flat / top_k;
    const float* x = hidden_states + (long long)token * hidden;
    // Both projections are [intermediate x hidden] per expert, stacked over
    // experts — the GGUF layout `[hidden, intermediate, experts]` with
    // dims[0] fastest-varying.
    long long base = ((long long)e * intermediate + r) * hidden;

    float sg = 0.0f;
    float su = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float xv = x[j];
        sg += dequant_element(gate_q, gate_quant, base + j) * xv;
        su += dequant_element(up_q,   up_quant,   base + j) * xv;
    }
    sg = block_reduce_sum(sg, xabe_shared);
    __syncthreads();
    su = block_reduce_sum(su, xabe_shared);

    if (threadIdx.x == 0) {
        // SwiGLU, written exactly as `xabe_kernels::norm::silu`:
        // x / (1 + exp(-x)), not the algebraically equal x * sigmoid(x).
        float act = sg / (1.0f + expf(-sg));
        inter[(long long)slot * intermediate + r] = act * su;
    }
}

// grid: (hidden, sorted_capacity).
//
// Writes each (token, k) contribution to its own slice of `partial` rather
// than accumulating into the output with atomics. Two reasons: atomicAdd
// makes the summation order non-deterministic, so the same input would give
// bit-different output run to run; and the deterministic reduction below can
// then sum in ascending k, matching the reference's per-token loop.
__global__ void moe_expert_down(
    const unsigned char* __restrict__ down_q, int down_quant,
    const float* __restrict__ inter,
    const float* __restrict__ topk_weights,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids,
    const int* __restrict__ valid_tokens,
    int top_k,
    int block_size,
    int hidden,
    int intermediate,
    float* __restrict__ partial
) {
    int h    = blockIdx.x;
    int slot = blockIdx.y;

    int e = expert_ids[slot / block_size];
    if (e < 0) return;
    int numel = (*valid_tokens) * top_k;
    int flat = sorted_token_ids[slot];
    if (flat >= numel) return;

    // [hidden x intermediate] per expert — GGUF `[intermediate, hidden, experts]`.
    long long base = ((long long)e * hidden + h) * intermediate;
    const float* a = inter + (long long)slot * intermediate;

    float s = 0.0f;
    for (int j = threadIdx.x; j < intermediate; j += blockDim.x) {
        s += dequant_element(down_q, down_quant, base + j) * a[j];
    }
    s = block_reduce_sum(s, xabe_shared);

    if (threadIdx.x == 0) {
        partial[(long long)flat * hidden + h] = topk_weights[flat] * s;
    }
}

// -------------------------------------------------------------------------
// 4. fp32 weighted sum of each token's top-k contributions.
// -------------------------------------------------------------------------
//
// grid: max_tokens. Ascending k, matching the reference's per-token loop.
__global__ void moe_reduce(
    const float* __restrict__ partial,
    const int* __restrict__ valid_tokens,
    int top_k,
    int hidden,
    float* __restrict__ out
) {
    int token = blockIdx.x;
    if (token >= *valid_tokens) return;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            acc += partial[((long long)token * top_k + k) * hidden + h];
        }
        out[(long long)token * hidden + h] = acc;
    }
}

// -------------------------------------------------------------------------
// 5. The shared expert, hoisted out of the routed path.
// -------------------------------------------------------------------------
//
// No routing, no sorting, no indirection, no routing weight: one expert
// applied to every token unconditionally. grid: (intermediate, max_tokens).
__global__ void moe_shared_ffn(
    const unsigned char* __restrict__ gate_q, int gate_quant,
    const unsigned char* __restrict__ up_q,   int up_quant,
    const float* __restrict__ hidden_states,
    const int* __restrict__ valid_tokens,
    int hidden,
    int intermediate,
    float* __restrict__ inter
) {
    int r     = blockIdx.x;
    int token = blockIdx.y;
    if (token >= *valid_tokens) return;

    const float* x = hidden_states + (long long)token * hidden;
    long long base = (long long)r * hidden;

    float sg = 0.0f;
    float su = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float xv = x[j];
        sg += dequant_element(gate_q, gate_quant, base + j) * xv;
        su += dequant_element(up_q,   up_quant,   base + j) * xv;
    }
    sg = block_reduce_sum(sg, xabe_shared);
    __syncthreads();
    su = block_reduce_sum(su, xabe_shared);

    if (threadIdx.x == 0) {
        float act = sg / (1.0f + expf(-sg));
        inter[(long long)token * intermediate + r] = act * su;
    }
}

// grid: (hidden, max_tokens).
__global__ void moe_shared_down(
    const unsigned char* __restrict__ down_q, int down_quant,
    const float* __restrict__ inter,
    const int* __restrict__ valid_tokens,
    int hidden,
    int intermediate,
    float* __restrict__ out
) {
    int h     = blockIdx.x;
    int token = blockIdx.y;
    if (token >= *valid_tokens) return;

    long long base = (long long)h * intermediate;
    const float* a = inter + (long long)token * intermediate;

    float s = 0.0f;
    for (int j = threadIdx.x; j < intermediate; j += blockDim.x) {
        s += dequant_element(down_q, down_quant, base + j) * a[j];
    }
    s = block_reduce_sum(s, xabe_shared);

    if (threadIdx.x == 0) out[(long long)token * hidden + h] = s;
}

}
"#;

/// The MoE shapes this instance is compiled and sized for.
///
/// Fixed at construction, exactly as `ModelConfig::qwen3_6_35b_a3b().moe`
/// fixes them for the real model: nothing downstream may vary a launch shape
/// per step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeGeometry {
    /// Routed experts available (256 for Qwen3.6).
    pub num_experts: usize,
    /// Routed experts selected per token (8).
    pub experts_per_token: usize,
    /// Residual stream width (2048).
    pub hidden: usize,
    /// Per-expert FFN width (512).
    pub intermediate: usize,
    /// Grouped-GEMM tile width the dispatch tables pad to.
    pub block_size: usize,
    /// Largest token count any single step may present.
    ///
    /// This is what every buffer and every grid is sized from, so it is the
    /// one number that has to be a genuine upper bound rather than a typical
    /// value.
    pub max_tokens: usize,
}

impl MoeGeometry {
    /// The real Qwen3.6 MoE shape, for `max_tokens` per step.
    pub const fn qwen3_6(block_size: usize, max_tokens: usize) -> Self {
        Self {
            num_experts: 256,
            experts_per_token: 8,
            hidden: 2048,
            intermediate: 512,
            block_size,
            max_tokens,
        }
    }

    /// Largest number of flat `(token, k)` pairs a step can produce.
    pub const fn max_flat_pairs(&self) -> usize {
        self.max_tokens * self.experts_per_token
    }

    /// Fixed capacity of `sorted_token_ids`, in slots.
    ///
    /// `num_tokens_post_pad` is the sum over *active* experts of
    /// `ceil(count_e / block_size) * block_size`. Each term is at most
    /// `count_e + block_size - 1`, the counts sum to `max_flat_pairs`, and
    /// at most `min(num_experts, max_flat_pairs)` experts can be active, so
    /// the total is bounded by
    /// `max_flat_pairs + active * (block_size - 1)`.
    ///
    /// At Qwen3.6's 256 experts and top-8 this is worst-case tight: with a
    /// batch big enough that every expert is hit, and every expert hit a
    /// number of times that is one more than a multiple of `block_size`,
    /// every one of the 256 runs pays the full `block_size - 1` of padding.
    /// Rounding up to a whole number of blocks keeps
    /// `sorted_capacity / block_size` exact.
    pub const fn sorted_capacity(&self) -> usize {
        let numel = self.max_flat_pairs();
        let active = if self.num_experts < numel {
            self.num_experts
        } else {
            numel
        };
        let bound = numel + active * (self.block_size - 1);
        bound.div_ceil(self.block_size) * self.block_size
    }

    /// Fixed capacity of `expert_ids`, in blocks.
    pub const fn expert_block_capacity(&self) -> usize {
        self.sorted_capacity() / self.block_size
    }

    /// Elements one expert stack must hold for this geometry.
    ///
    /// Gate and up are `[intermediate x hidden]` per expert; down is
    /// `[hidden x intermediate]`. Both come to the same count.
    pub const fn stack_elements(&self) -> usize {
        self.num_experts * self.intermediate * self.hidden
    }
}

/// Something went wrong compiling, sizing, or launching a MoE kernel.
#[derive(Debug)]
pub enum MoeError {
    /// NVRTC rejected the source, or the module failed to load.
    Compile(String),
    /// The driver failed.
    Driver(DriverError),
    /// A geometry the kernels cannot service, with the reason.
    UnsupportedGeometry {
        geometry: Box<MoeGeometry>,
        reason: &'static str,
    },
    /// A weight stack is not a whole number of quantization blocks.
    ///
    /// Rejected rather than truncated: a partial trailing block would make
    /// the last expert's rows read past the tensor.
    RaggedWeights {
        which: &'static str,
        bytes: usize,
        block_bytes: usize,
    },
    /// A weight stack does not hold the element count the geometry implies.
    ///
    /// This is the check that catches gate/up (`[hidden, intermediate,
    /// experts]`) being handed where down (`[intermediate, hidden, experts]`)
    /// belongs on a model where those differ, and any expert-count mismatch.
    WrongElementCount {
        which: &'static str,
        expected: usize,
        found: usize,
    },
    /// More tokens were presented than the buffers were sized for.
    TooManyTokens { tokens: usize, max_tokens: usize },
}

impl std::fmt::Display for MoeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compile(m) => write!(f, "kernel compilation failed: {m}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::UnsupportedGeometry { geometry, reason } => {
                write!(f, "unsupported MoE geometry {geometry:?}: {reason}")
            }
            Self::RaggedWeights {
                which,
                bytes,
                block_bytes,
            } => write!(
                f,
                "{which}: {bytes} bytes is not a whole number of {block_bytes} B blocks",
            ),
            Self::WrongElementCount {
                which,
                expected,
                found,
            } => write!(
                f,
                "{which}: expected {expected} elements for this geometry, stack holds {found}",
            ),
            Self::TooManyTokens { tokens, max_tokens } => write!(
                f,
                "{tokens} tokens exceeds the {max_tokens} the buffers were sized for",
            ),
        }
    }
}

impl std::error::Error for MoeError {}

impl From<DriverError> for MoeError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

/// Every buffer the MoE path touches, allocated once.
///
/// Nothing here is sized by a per-step value. `valid_tokens` is the device
/// scalar the kernels gate on, and `num_tokens_post_pad` is the
/// data-dependent size the dispatch kernel produces — held in device memory
/// so no launch shape ever has to wait for it.
pub struct MoeBuffers {
    topk_ids: CudaSlice<i32>,
    topk_weights: CudaSlice<f32>,
    sorted_token_ids: CudaSlice<i32>,
    expert_ids: CudaSlice<i32>,
    num_tokens_post_pad: CudaSlice<i32>,
    valid_tokens: CudaSlice<i32>,
    inter: CudaSlice<f32>,
    partial: CudaSlice<f32>,
    shared_inter: CudaSlice<f32>,
}

impl MoeBuffers {
    /// Selected expert ids, `[max_tokens][experts_per_token]`.
    pub fn topk_ids(&self) -> &CudaSlice<i32> {
        &self.topk_ids
    }

    /// Renormalized routing weights, same layout as [`Self::topk_ids`], and
    /// indexed by the flat `(token, k)` id the dispatch tables carry.
    pub fn topk_weights(&self) -> &CudaSlice<f32> {
        &self.topk_weights
    }

    /// Flat `(token, k)` ids grouped by expert and padded to `block_size`.
    pub fn sorted_token_ids(&self) -> &CudaSlice<i32> {
        &self.sorted_token_ids
    }

    /// Expert owning each `block_size`-sized run, or `-1` for inactive.
    pub fn expert_ids(&self) -> &CudaSlice<i32> {
        &self.expert_ids
    }

    /// The single-element device scalar holding `num_tokens_post_pad`.
    pub fn num_tokens_post_pad(&self) -> &CudaSlice<i32> {
        &self.num_tokens_post_pad
    }

    /// The single-element device scalar holding this step's token count.
    pub fn valid_tokens(&self) -> &CudaSlice<i32> {
        &self.valid_tokens
    }

    /// Total device bytes held.
    pub fn bytes(&self) -> usize {
        (self.topk_ids.len()
            + self.sorted_token_ids.len()
            + self.expert_ids.len()
            + self.num_tokens_post_pad.len()
            + self.valid_tokens.len())
            * size_of::<i32>()
            + (self.topk_weights.len()
                + self.inter.len()
                + self.partial.len()
                + self.shared_inter.len())
                * size_of::<f32>()
    }
}

/// Compiled MoE kernels for one fixed geometry.
pub struct MoeKernels {
    route: CudaFunction,
    align: CudaFunction,
    expert_ffn: CudaFunction,
    expert_down: CudaFunction,
    reduce: CudaFunction,
    shared_ffn: CudaFunction,
    shared_down: CudaFunction,
    geometry: MoeGeometry,
}

impl MoeKernels {
    /// Compile and validate for `geometry`.
    ///
    /// The geometry is checked once here so the launch path has nothing left
    /// to reject — the same reasoning as `GdnKernels::new`.
    pub fn new(ctx: &Arc<CudaContext>, geometry: MoeGeometry) -> Result<Self, MoeError> {
        let bad = |reason: &'static str| MoeError::UnsupportedGeometry {
            geometry: Box::new(geometry),
            reason,
        };
        if geometry.block_size == 0 {
            return Err(bad("block_size must be non-zero"));
        }
        if geometry.experts_per_token == 0 {
            return Err(bad("experts_per_token must be non-zero"));
        }
        if geometry.experts_per_token > geometry.num_experts {
            return Err(bad("experts_per_token must not exceed num_experts"));
        }
        if geometry.max_tokens == 0 {
            return Err(bad("max_tokens must be non-zero"));
        }
        if geometry.hidden == 0 || geometry.intermediate == 0 {
            return Err(bad("hidden and intermediate must be non-zero"));
        }
        // grid.y is the slot capacity and grid.x an output row; both must fit
        // the driver's per-dimension limits. x is capped at 2^31-1 but y and
        // z at 65535, which is the one that can actually bite.
        if geometry.sorted_capacity() > 65_535 || geometry.max_tokens > 65_535 {
            return Err(bad("slot capacity exceeds the 65535 grid.y limit"));
        }

        let ptx = compile(MOE_SRC, "moe").map_err(MoeError::Compile)?;
        let module = ctx.load_module(ptx)?;
        Ok(Self {
            route: module.load_function("moe_route")?,
            align: module.load_function("moe_align_block_size")?,
            expert_ffn: module.load_function("moe_expert_ffn")?,
            expert_down: module.load_function("moe_expert_down")?,
            reduce: module.load_function("moe_reduce")?,
            shared_ffn: module.load_function("moe_shared_ffn")?,
            shared_down: module.load_function("moe_shared_down")?,
            geometry,
        })
    }

    /// The geometry this instance was compiled for.
    pub fn geometry(&self) -> MoeGeometry {
        self.geometry
    }

    /// Allocate every buffer, once.
    pub fn buffers(&self, stream: &Arc<CudaStream>) -> Result<MoeBuffers, MoeError> {
        let g = self.geometry;
        Ok(MoeBuffers {
            topk_ids: stream.alloc_zeros::<i32>(g.max_flat_pairs())?,
            topk_weights: stream.alloc_zeros::<f32>(g.max_flat_pairs())?,
            sorted_token_ids: stream.alloc_zeros::<i32>(g.sorted_capacity())?,
            expert_ids: stream.alloc_zeros::<i32>(g.expert_block_capacity())?,
            num_tokens_post_pad: stream.alloc_zeros::<i32>(1)?,
            valid_tokens: stream.alloc_zeros::<i32>(1)?,
            inter: stream.alloc_zeros::<f32>(g.sorted_capacity() * g.intermediate)?,
            partial: stream.alloc_zeros::<f32>(g.max_flat_pairs() * g.hidden)?,
            shared_inter: stream.alloc_zeros::<f32>(g.max_tokens * g.intermediate)?,
        })
    }

    /// Publish this step's token count into the device scalar the kernels
    /// gate on.
    ///
    /// This is a data write, not a shape decision: no allocation happens and
    /// no launch geometry changes, which is what keeps the sequence
    /// capturable.
    pub fn set_valid_tokens(
        &self,
        stream: &Arc<CudaStream>,
        buffers: &mut MoeBuffers,
        tokens: usize,
    ) -> Result<(), MoeError> {
        if tokens > self.geometry.max_tokens {
            return Err(MoeError::TooManyTokens {
                tokens,
                max_tokens: self.geometry.max_tokens,
            });
        }
        stream.memcpy_htod(&[tokens as i32], &mut buffers.valid_tokens)?;
        Ok(())
    }

    /// Softmax + top-k over all experts, entirely on the device.
    ///
    /// `logits` is `[max_tokens][num_experts]`; only the first
    /// `valid_tokens` rows are read.
    pub fn route(
        &self,
        stream: &Arc<CudaStream>,
        buffers: &mut MoeBuffers,
        logits: &CudaSlice<f32>,
    ) -> Result<(), MoeError> {
        let g = self.geometry;
        let num_experts = g.num_experts as i32;
        let top_k = g.experts_per_token as i32;
        // probs[num_experts] + one (float, int) reduction slot per thread.
        let shared = ((g.num_experts + THREADS as usize) * size_of::<f32>()
            + THREADS as usize * size_of::<i32>()) as u32;

        let cfg = LaunchConfig {
            grid_dim: (g.max_tokens as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: shared,
        };
        let mut builder = stream.launch_builder(&self.route);
        builder
            .arg(logits)
            .arg(&buffers.valid_tokens)
            .arg(&num_experts)
            .arg(&top_k)
            .arg(&mut buffers.topk_ids)
            .arg(&mut buffers.topk_weights);
        // SAFETY: one block per token slot, bounded by the device
        // `valid_tokens`; `logits` holds `max_tokens * num_experts` floats
        // and both outputs `max_tokens * top_k`. Shared memory covers the
        // probability array plus one reduction slot per thread, which is
        // everything the kernel indexes.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// Build the sorted-token indirection into the fixed-size buffers.
    ///
    /// Consumes [`Self::route`]'s `topk_ids` and writes `sorted_token_ids`,
    /// `expert_ids` and the device-side `num_tokens_post_pad`. Nothing is
    /// read back.
    pub fn build_dispatch(
        &self,
        stream: &Arc<CudaStream>,
        buffers: &mut MoeBuffers,
    ) -> Result<(), MoeError> {
        let g = self.geometry;
        let top_k = g.experts_per_token as i32;
        let num_experts = g.num_experts as i32;
        let block_size = g.block_size as i32;
        let sorted_capacity = g.sorted_capacity() as i32;
        let expert_capacity = g.expert_block_capacity() as i32;
        let shared = ((g.num_experts + 1) * size_of::<i32>()) as u32;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: shared,
        };
        let mut builder = stream.launch_builder(&self.align);
        builder
            .arg(&buffers.topk_ids)
            .arg(&buffers.valid_tokens)
            .arg(&top_k)
            .arg(&num_experts)
            .arg(&block_size)
            .arg(&sorted_capacity)
            .arg(&expert_capacity)
            .arg(&mut buffers.sorted_token_ids)
            .arg(&mut buffers.expert_ids)
            .arg(&mut buffers.num_tokens_post_pad);
        // SAFETY: a single block; the shared array is `num_experts + 1`
        // ints, exactly what the prefix sum indexes, and every global write
        // is bounds-checked against the capacities passed in.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// The routed half of the MoE block: grouped GEMM over the dispatch
    /// tables, then the fp32 weighted sum of each token's `top_k`
    /// contributions.
    ///
    /// `hidden_states` is `[max_tokens][hidden]`, `out` is the same shape.
    /// Requires [`Self::route`] and [`Self::build_dispatch`] to have run on
    /// `buffers` for this step.
    #[allow(clippy::too_many_arguments)]
    pub fn grouped_forward(
        &self,
        stream: &Arc<CudaStream>,
        buffers: &mut MoeBuffers,
        gate: QuantTensor<'_>,
        up: QuantTensor<'_>,
        down: QuantTensor<'_>,
        hidden_states: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), MoeError> {
        let g = self.geometry;
        check_stack("gate", gate, g.stack_elements())?;
        check_stack("up", up, g.stack_elements())?;
        check_stack("down", down, g.stack_elements())?;

        let top_k = g.experts_per_token as i32;
        let block_size = g.block_size as i32;
        let hidden = g.hidden as i32;
        let intermediate = g.intermediate as i32;
        let gate_code = gate.quant.code();
        let up_code = up.quant.code();
        let down_code = down.quant.code();
        // One float per warp, which is the most `block_reduce_sum` stores.
        let shared = ((THREADS as usize).div_ceil(32) * size_of::<f32>()) as u32;

        let ffn_cfg = LaunchConfig {
            grid_dim: (g.intermediate as u32, g.sorted_capacity() as u32, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: shared,
        };
        let mut builder = stream.launch_builder(&self.expert_ffn);
        builder
            .arg(gate.bytes)
            .arg(&gate_code)
            .arg(up.bytes)
            .arg(&up_code)
            .arg(hidden_states)
            .arg(&buffers.sorted_token_ids)
            .arg(&buffers.expert_ids)
            .arg(&buffers.valid_tokens)
            .arg(&top_k)
            .arg(&block_size)
            .arg(&hidden)
            .arg(&intermediate)
            .arg(&mut buffers.inter);
        // SAFETY: grid.y is the slot capacity, which is exactly what
        // `sorted_token_ids` holds and `block_size` times what `expert_ids`
        // holds; `inter` is `sorted_capacity * intermediate` floats, the
        // range `(slot, r)` covers. Weight indexing is bounded by the
        // element-count check above.
        unsafe { builder.launch(ffn_cfg) }?;

        // Every valid flat id is written exactly once by the down kernel, so
        // this zeroing is belt-and-braces — but a dispatch bug that dropped a
        // slot would otherwise read a previous step's contribution and
        // produce a plausible wrong answer instead of an obviously wrong one.
        stream.memset_zeros(&mut buffers.partial)?;

        let down_cfg = LaunchConfig {
            grid_dim: (g.hidden as u32, g.sorted_capacity() as u32, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: shared,
        };
        let mut builder = stream.launch_builder(&self.expert_down);
        builder
            .arg(down.bytes)
            .arg(&down_code)
            .arg(&buffers.inter)
            .arg(&buffers.topk_weights)
            .arg(&buffers.sorted_token_ids)
            .arg(&buffers.expert_ids)
            .arg(&buffers.valid_tokens)
            .arg(&top_k)
            .arg(&block_size)
            .arg(&hidden)
            .arg(&intermediate)
            .arg(&mut buffers.partial);
        // SAFETY: as above; `partial` is `max_flat_pairs * hidden` floats and
        // is indexed by `flat * hidden + h` with `flat < valid_tokens *
        // top_k <= max_flat_pairs`.
        unsafe { builder.launch(down_cfg) }?;

        let reduce_cfg = LaunchConfig {
            grid_dim: (g.max_tokens as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&self.reduce);
        builder
            .arg(&buffers.partial)
            .arg(&buffers.valid_tokens)
            .arg(&top_k)
            .arg(&hidden)
            .arg(out);
        // SAFETY: one block per token slot, gated on the device
        // `valid_tokens`; `out` is `max_tokens * hidden` floats.
        unsafe { builder.launch(reduce_cfg) }?;
        Ok(())
    }

    /// The shared expert, applied to every token unconditionally.
    ///
    /// Deliberately a separate entry point with no `buffers.sorted_*` in
    /// sight: the shared expert is always active, so paying for routing,
    /// sorting, or indirection on it would be pure overhead — 41 blocks'
    /// worth per token.
    ///
    /// The stacks here are single-expert: `[intermediate x hidden]` for gate
    /// and up, `[hidden x intermediate]` for down.
    #[allow(clippy::too_many_arguments)]
    pub fn shared_expert(
        &self,
        stream: &Arc<CudaStream>,
        buffers: &mut MoeBuffers,
        gate: QuantTensor<'_>,
        up: QuantTensor<'_>,
        down: QuantTensor<'_>,
        hidden_states: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), MoeError> {
        let g = self.geometry;
        let one_expert = g.intermediate * g.hidden;
        check_stack("shared gate", gate, one_expert)?;
        check_stack("shared up", up, one_expert)?;
        check_stack("shared down", down, one_expert)?;

        let hidden = g.hidden as i32;
        let intermediate = g.intermediate as i32;
        let gate_code = gate.quant.code();
        let up_code = up.quant.code();
        let down_code = down.quant.code();
        let shared = ((THREADS as usize).div_ceil(32) * size_of::<f32>()) as u32;

        let ffn_cfg = LaunchConfig {
            grid_dim: (g.intermediate as u32, g.max_tokens as u32, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: shared,
        };
        let mut builder = stream.launch_builder(&self.shared_ffn);
        builder
            .arg(gate.bytes)
            .arg(&gate_code)
            .arg(up.bytes)
            .arg(&up_code)
            .arg(hidden_states)
            .arg(&buffers.valid_tokens)
            .arg(&hidden)
            .arg(&intermediate)
            .arg(&mut buffers.shared_inter);
        // SAFETY: grid is (intermediate, max_tokens) and `shared_inter` is
        // `max_tokens * intermediate` floats; the token index is gated on
        // the device `valid_tokens`.
        unsafe { builder.launch(ffn_cfg) }?;

        let down_cfg = LaunchConfig {
            grid_dim: (g.hidden as u32, g.max_tokens as u32, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: shared,
        };
        let mut builder = stream.launch_builder(&self.shared_down);
        builder
            .arg(down.bytes)
            .arg(&down_code)
            .arg(&buffers.shared_inter)
            .arg(&buffers.valid_tokens)
            .arg(&hidden)
            .arg(&intermediate)
            .arg(out);
        // SAFETY: as above; `out` is `max_tokens * hidden` floats.
        unsafe { builder.launch(down_cfg) }?;
        Ok(())
    }
}

fn check_stack(which: &'static str, t: QuantTensor<'_>, expected: usize) -> Result<(), MoeError> {
    if !t.is_whole_blocks() {
        return Err(MoeError::RaggedWeights {
            which,
            bytes: t.bytes.len(),
            block_bytes: t.quant.block_bytes(),
        });
    }
    let found = t.elements();
    if found != expected {
        return Err(MoeError::WrongElementCount {
            which,
            expected,
            found,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen() -> MoeGeometry {
        MoeGeometry::qwen3_6(16, 64)
    }

    #[test]
    fn the_dequant_prologue_multiplies_in_the_reference_order() {
        // Bit-identical weights depend on `(d * scale) * q` for Q6_K and
        // `q * d` for Q8_0. This guards the two expressions from being
        // "simplified" into a different rounding.
        assert!(
            MOE_SRC.contains("return d * (float)sc[si] * (float)(raw - 32);"),
            "Q6_K prologue reassociated away from (d * scale) * q",
        );
        assert!(
            MOE_SRC.contains("return (float)q * d;"),
            "Q8_0 prologue reassociated away from q * d",
        );
    }

    #[test]
    fn quants_and_scales_are_read_as_signed() {
        // Reading int8 codes or the per-group scales as unsigned is the most
        // likely transcription error and produces plausible magnitudes.
        assert!(MOE_SRC.contains("signed char q = (signed char)base[2 + lane];"));
        assert!(MOE_SRC.contains("(const signed char*)(base + 192 + half * 8)"));
    }

    #[test]
    fn the_padding_sentinel_is_num_tokens_times_top_k() {
        // `padding_sentinel(num_tokens, top_k) == num_tokens * top_k` in
        // `xabe_kernels::moe::dispatch`, and consumers test `flat >= numel`.
        // A sentinel of, say, -1 would still "work" for the fill but would
        // make `flat / top_k` index before the token array.
        assert!(MOE_SRC.contains("int numel = (*valid_tokens) * top_k;"));
        assert!(MOE_SRC.contains("sorted_token_ids[s] = numel;"));
        assert!(MOE_SRC.contains("if (flat >= numel) return;"));
    }

    #[test]
    fn inactive_blocks_are_minus_one() {
        assert!(MOE_SRC.contains("expert_ids[b] = -1;"));
        assert!(MOE_SRC.contains("if (e < 0) return;"));
    }

    #[test]
    fn the_dispatch_scatter_is_ordered_not_atomic() {
        // The reference places an expert's tokens in ascending flat index.
        // An atomic cursor would give arrival order instead, which is not
        // reproducible run to run and cannot be compared exactly. Scoped to
        // the dispatch kernel's body rather than the whole source so that a
        // legitimate atomic elsewhere would not trip it.
        let start = MOE_SRC
            .find("void moe_align_block_size")
            .expect("dispatch kernel present");
        let end = MOE_SRC[start..]
            .find("void moe_expert_ffn")
            .expect("next kernel present")
            + start;
        assert!(
            !MOE_SRC[start..end].contains("atomic"),
            "the dispatch scatter must not use atomics; ordering is the contract",
        );
    }

    #[test]
    fn routing_selection_carries_the_index_so_ties_break_low() {
        // `route_token` breaks ties on equal probability by lower expert
        // index. Without the index in the reduction the winner is whichever
        // thread got there first, which is not even stable across runs.
        assert!(MOE_SRC.contains("if (p > bv || (p == bv && (bi < 0 || e < bi)))"));
        assert!(MOE_SRC.contains("if (cv > av || (cv == av && ci >= 0 && (ai < 0 || ci < ai)))"));
    }

    #[test]
    fn sorted_capacity_bounds_the_worst_case_at_the_real_geometry() {
        let g = qwen();
        // 64 tokens x top-8 = 512 flat pairs. Every one of the 256 experts
        // can be active, and each active run pays up to block_size - 1 of
        // padding: 512 + 256*15 = 4352, already a multiple of 16.
        assert_eq!(g.max_flat_pairs(), 512);
        assert_eq!(g.sorted_capacity(), 4352);
        assert_eq!(g.expert_block_capacity(), 272);
        assert!(g.sorted_capacity().is_multiple_of(g.block_size));

        // Brute force the bound on a small geometry: enumerate every way of
        // splitting `numel` selections across experts is too many, so check
        // the analytic worst case instead — one expert per selection until
        // experts run out, each padded to a full block.
        let small = MoeGeometry {
            num_experts: 4,
            experts_per_token: 2,
            hidden: 8,
            intermediate: 4,
            block_size: 4,
            max_tokens: 3,
        };
        let numel = small.max_flat_pairs(); // 6
        let mut worst = 0usize;
        // All distributions of 6 selections over 4 experts.
        for a in 0..=numel {
            for b in 0..=numel - a {
                for c in 0..=numel - a - b {
                    let d = numel - a - b - c;
                    let total: usize = [a, b, c, d]
                        .iter()
                        .map(|&n| n.div_ceil(small.block_size) * small.block_size)
                        .sum();
                    worst = worst.max(total);
                }
            }
        }
        assert!(
            small.sorted_capacity() >= worst,
            "capacity {} is below the enumerated worst case {worst}",
            small.sorted_capacity(),
        );
    }

    #[test]
    fn every_buffer_is_sized_from_the_geometry_alone() {
        // The property AGENTS.md rule 5 is really asking for: nothing below
        // depends on a per-step token count.
        let g = qwen();
        assert_eq!(g.stack_elements(), 256 * 512 * 2048);
        // 8.9 MiB of intermediate, 4 MiB of per-(token,k) partials.
        assert_eq!(g.sorted_capacity() * g.intermediate, 4352 * 512);
        assert_eq!(g.max_flat_pairs() * g.hidden, 512 * 2048);
    }

    #[test]
    fn geometry_is_validated_not_assumed() {
        // The checks that run without a device.
        let bad = MoeError::UnsupportedGeometry {
            geometry: Box::new(MoeGeometry {
                block_size: 0,
                ..qwen()
            }),
            reason: "block_size must be non-zero",
        };
        assert!(bad.to_string().contains("block_size must be non-zero"));

        let ragged = MoeError::RaggedWeights {
            which: "gate",
            bytes: 211,
            block_bytes: 210,
        };
        assert!(ragged.to_string().contains("211"));

        let wrong = MoeError::WrongElementCount {
            which: "down",
            expected: 268_435_456,
            found: 1024,
        };
        assert!(wrong.to_string().contains("268435456"));

        let too_many = MoeError::TooManyTokens {
            tokens: 65,
            max_tokens: 64,
        };
        assert!(too_many.to_string().contains("65"));
    }

    #[test]
    fn quant_block_geometry_matches_the_ggml_layout() {
        assert_eq!(ExpertQuant::Q6K.block_elements(), 256);
        assert_eq!(ExpertQuant::Q6K.block_bytes(), 210);
        assert_eq!(ExpertQuant::Q8_0.block_elements(), 32);
        assert_eq!(ExpertQuant::Q8_0.block_bytes(), 34);
        assert_eq!(BLOCK_Q6_K_BYTES, QK_K / 2 + QK_K / 4 + QK_K / 16 + 2);
        assert_eq!(BLOCK_Q8_0_BYTES, 2 + QK8_0);
        // The real file is mixed, so the codes must be distinct and stable.
        assert_ne!(ExpertQuant::Q6K.code(), ExpertQuant::Q8_0.code());
    }

    #[test]
    fn the_shared_expert_has_no_indirection() {
        // If the shared-expert kernels ever grow a `sorted_token_ids`
        // argument, the hoist has been undone.
        let start = MOE_SRC
            .find("void moe_shared_ffn")
            .expect("shared ffn present");
        let tail = &MOE_SRC[start..];
        assert!(
            !tail.contains("sorted_token_ids"),
            "the shared expert must not consult the routed dispatch tables",
        );
        assert!(
            !tail.contains("topk_weights"),
            "the shared expert has no routing weight",
        );
    }
}
