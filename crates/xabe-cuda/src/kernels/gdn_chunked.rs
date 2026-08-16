//! Gated DeltaNet, chunked-parallel (prefill) form, on the device.
//!
//! The sibling of [`super::gdn`]. Decode advances the recurrence one token at
//! a time; prefill cannot afford to, because the per-token dependency chain
//! serialises a 32K-token prompt into 32K dependent kernel launches. The
//! chunked form breaks `chunk_len` tokens out of that chain at a time by
//! solving for all of their state corrections *simultaneously*, at the cost of
//! one triangular solve per chunk per head.
//!
//! The reference is `xabe_kernels::gdn::chunked::chunked_forward`, whose
//! module doc carries the full derivation. It is reproduced here only as far
//! as the kernel needs it. Within one chunk of `C` tokens, with `S_in` the
//! state carried in, `g_t = sum_{i<=t} log_decay_i` the cumulative in-chunk
//! log-decay, `lambda_t = exp(g_t)`, and unit-norm keys `k_t`:
//!
//! ```text
//! W[t]      = beta_t * (v_t / lambda_t  -  S_in k_t)
//! (I + A) U = W,   A[t][i] = beta_t (k_i . k_t) for i < t, 0 otherwise
//! o_t       = lambda_t * (S_in q_t + sum_{i<=t} u_i (k_i . q_t))
//! S_out     = lambda_{C-1} * (S_in + sum_i u_i k_i^T)
//! ```
//!
//! ## Why the kernel does not solve for `U`, but for `lambda_t u_t`
//!
//! That statement is unusable in fp32 on this model. `1/lambda_t` is
//! unbounded: Qwen3.6's per-token log-decays reach **-91.58** (block 0, head
//! 9), so `lambda_1` is already `1.691e-40` (subnormal) and `v_1 / lambda_1`
//! overflows fp32 at `|v| > 5.75e-2` — the measured `max|v_1|` there is `6.95`,
//! two orders past it. Run as written,
//! block 0's chunked prefill was 38912/38912 `NaN` and block 20's was
//! 20480/38912. Block 4, whose worst per-token log-decay is only -5.99, came
//! through and agreed with the recurrent form to `5.96e-8`, so the formulation
//! was right and only its *frame* was wrong.
//!
//! The fix is a change of unknown, not a clamp. Substitute `u'_t = lambda_t
//! u_t` into the system above and multiply the `t`-th row through by
//! `lambda_t`:
//!
//! ```text
//! u'_t + beta_t sum_{i<t} (k_i . k_t) (lambda_t/lambda_i) u'_i
//!                                 = beta_t * (v_t - lambda_t * (S_in k_t))
//! o_t   = lambda_t * (S_in q_t) + sum_{i<=t} (lambda_t/lambda_i) u'_i (k_i . q_t)
//! S_out = lambda_{C-1} * S_in + sum_i (lambda_{C-1}/lambda_i) u'_i k_i^T
//! ```
//!
//! Every surviving decay factor is either `lambda_t = exp(g_t)` or a ratio
//! `lambda_a / lambda_i = exp(g_a - g_i)` with `a >= i`, and `g_a - g_i =
//! sum_{i<m<=a} log_decay_m` is a sum of *actual per-token log-decays over a
//! sub-range of the chunk*. **No division by a decay remains anywhere.** For
//! this model every `log_decay <= 0` structurally — `gate-N = softplus(alpha +
//! dt_bias) * ssm_a` with every entry of `ssm_a` negative and softplus
//! positive — so every exponent is `<= 0`, every factor is in `(0, 1]`, and
//! `expf` of a non-positive argument cannot overflow. The failure mode that
//! remains is underflow to `+0`, which is the correct limit: a state that has
//! decayed below fp32 really has stopped contributing. The kernel therefore
//! keeps `g_t` in shared memory and never materialises `1/lambda_t`.
//!
//! This is also what llama.cpp does. `build_delta_net_chunking`
//! (`src/models/delta-net-base.cpp:89-216`) builds `decay_mask = exp(g_cs_j -
//! g_cs_i)` over the lower-diagonal triangle and `g_diff = exp(g_last -
//! g_cum)`, multiplies them in, and never divides. Its comment quotes the
//! PyTorch reference's `torch.clamp(..., max=50.0)` around both exponentials;
//! **llama.cpp's own graph does not emit that clamp**, and it cannot bind
//! here, because a clamp at `+50` only fires on a positive exponent and every
//! exponent above is non-positive. No clamp is added: one would change the
//! answer silently on exactly the inputs it fired for, and there is nothing
//! for it to protect against.
//!
//! ## Shape
//!
//! Identical to the recurrent form: state `S` is `head_dim x head_dim` fp32
//! per value head, laid out `S[v][k]` with the key index contiguous, 32 value
//! heads sharing 16 query/key heads by **modulo** — `qk_head = v_head %
//! qk_heads`, see [`super::gdn`]'s module docs for the measurement that
//! settles it — `head_dim` 128, `chunk_len` 64. The triangular system is
//! therefore 64x64 per chunk per head, and `U` is 64x128.
//!
//! ## The triangular solve: forward substitution, not explicit inversion
//!
//! This is the one place where the device kernel deliberately does *not*
//! transcribe the reference. `chunked_forward` calls
//! `tri::invert_unit_lower_triangular` to form `M^-1 = (I + A)^-1` explicitly
//! and then computes `U = M^-1 W`. This kernel forward-substitutes for `U`
//! directly and never materialises `M^-1`. In exact arithmetic the two are the
//! same answer; in floating point and in silicon they are not equivalent, and
//! three arguments all point the same way.
//!
//! **Shared memory.** This argument held while a block owned a whole head's
//! `U` — `C x head_dim` = 32 KiB at `C = 64`, against `M^-1`'s `C x C` = 16
//! KiB, for 48 KiB exactly at Turing's per-block limit with nothing left over.
//! It no longer does: a solve block now owns a `SOLVE_VB`-wide band of value
//! indices, so its `U` slice is 8 KiB and an inverse would fit beside it.
//! **The shared-memory case against inversion is void; the arithmetic one
//! below is what the choice now rests on.** Recorded rather than deleted
//! because the conclusion outlived its first reason, and a reader who
//! rediscovers the 48 KiB sum should know it was already accounted for.
//!
//! **Arithmetic.** Inverting costs `C^3/6` ~= 44k multiply-adds per head per
//! chunk and the following `C x C x head_dim` product costs 524k. Forward
//! substitution costs `C^2 head_dim / 2` = 262k and there is no inversion —
//! 2.2x less work for the same result.
//!
//! **Numerics.** Substitution is backward stable for triangular systems: the
//! computed `U` is the exact solution of `(M + dM) U = W` with `|dM| <=
//! gamma_C |M|` elementwise (Higham, *Accuracy and Stability of Numerical
//! Algorithms*, 2nd ed., Thm. 8.5). Inverting and then multiplying is not
//! backward stable, and it materialises intermediates the substitution never
//! forms: row `t` of the inverse of a unit lower-triangular matrix can grow
//! like `2^(t-1)`, so at `C = 64` the worst case is ~9.2e18 — far past what an
//! fp32 significand can carry into the subsequent product without the small
//! entries of `W` being annihilated by cancellation. Whether that growth is
//! *realised* is data-dependent, and here it is not: keys are L2-normalized so
//! `|A[t][i]| = beta_t |k_i . k_t| <= beta_t < 1`, and at `head_dim = 128` the
//! observed off-diagonal magnitudes are ~`beta / sqrt(128)` ~= 0.04, nowhere
//! near the bound. That is exactly the point — forward substitution does not
//! depend on that remaining true and explicit inversion does. The reference
//! keeps the explicit inverse because it is graded on being a legible
//! transcription of the derivation; this kernel is graded on agreeing with it,
//! and `crates/xabe-engine/tests/gdn_chunked_differential.rs` measures the gap
//! rather than assuming it.
//!
//! ## Why four kernels
//!
//! Chunks are sequentially dependent through `S`, so the host loops over
//! chunks and each chunk runs four launches:
//!
//! 1. `gdn_chunk_normalize_qk` — L2-normalize `q`/`k` per (token, qk head) and
//!    fold `1/sqrt(head_dim)` into `q`. Runs **once for the whole sequence**,
//!    not per chunk: normalization has no cross-token dependency.
//! 2. `gdn_chunk_gram` — the two Gram matrices `k_i . k_t` and `k_i . q_t`.
//!    Both depend only on the query/key head, not the value head, so computing
//!    them once and letting the two value heads that share a qk head read them
//!    halves the work. `beta_t`, which *is* per value head, is folded in later.
//! 3. `gdn_chunk_inter` — `S_in k_t` and `S_in q_t` for every token in the
//!    chunk. Must run before the state is overwritten.
//! 4. `gdn_chunk_solve_and_apply` — the substitution, the per-token output and
//!    the chunk-end state, fused into one block per value head because all
//!    three consume `U`, and `U` is 32 KiB of shared memory that would
//!    otherwise round-trip through global.
//!
//! The fused kernel launches one block per value head — 32 blocks — which
//! leaves most of a 72-SM card idle. That is a known and untuned cost: the
//! solve is inherently sequential over the `C` tokens of a chunk, so the
//! parallelism available inside it is `head_dim` threads, and batching several
//! sequences is what fills the machine. This module is gated on correctness
//! against the reference, not on throughput.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};

use super::compile;

/// The epsilon inside the L2 normalization — a floor on the norm, as
/// `ggml_l2_norm` has it.
///
/// The same `1e-6` as [`super::gdn::L2_EPS`] and as
/// `xabe_kernels::gdn::recurrent::recurrent_forward`. Duplicated as a
/// re-export rather than a second literal so the two forms cannot drift.
pub use super::gdn::L2_EPS;

/// Shared memory a Turing block gets without the opt-in carve-out.
///
/// Turing has 64 KiB of shared memory per SM, of which a block may claim 48
/// KiB by default; the remaining 16 KiB needs
/// `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`. Staying under the
/// default keeps the launch path free of a per-function attribute call, and
/// the chunked geometry fits with room to spare (32.5 KiB at `chunk_len = 64`,
/// `head_dim = 128`).
pub const MAX_SHARED_BYTES: usize = 48 * 1024;

/// Tokens one `gdn_chunk_inter` block carries. Mirrors the kernel's
/// `INTER_TT`; the drift test below is what keeps the two in step, because a
/// mismatch would size the shared staging for a different tile than the
/// kernel indexes into and read past its end.
///
/// `grid.x` is `chunk_len / INTER_TT` and every block in it reads the whole
/// `head_dim x head_dim` state slice for its head, so this is also how many
/// times per launch that slice is read.
///
/// | `INTER_TT` | tok/s |
/// | --- | ---: |
/// | 8 | 1,727.74 |
/// | **16** | **1,735.42** |
/// | 32 | 1,729.09 |
const INTER_TT: u32 = 16;

/// Value indices one solve block owns. Mirrors the kernel's `VB`.
const SOLVE_VB: u32 = 32;

/// Token positions the solve's output section computes side by side.
///
/// Mirrors `SOLVE_TT`. The block is `SOLVE_VB * SOLVE_TT` threads: the forward
/// substitution is sequential in `t` and runs on the first row of them, and
/// the per-token output is not, so it runs on all of them.
const SOLVE_TT: u32 = 8;
/// Warps in a scan block, each owning one value index. Mirrors `SCAN_WARPS`.
const SCAN_WARPS: u32 = 4;

/// Head dimensions one scan lane carries, at most. Mirrors `SCAN_MAXR`.
///
/// The state is a register array indexed by constants after unrolling, so the
/// bound is compile-time and head dimensions above `32 * SCAN_MAXR` are
/// rejected rather than silently spilled to local memory.
const SCAN_MAXR: usize = 4;

/// Value indices one state-update block owns. Mirrors `STATE_VB`.
const STATE_VB: u32 = STATE_VLANES * STATE_VT;
/// State columns one state-update block owns. Mirrors `STATE_JB`.
const STATE_JB: u32 = STATE_JLANES * STATE_JT;
/// Value indices and state columns one state-update *thread* owns. Mirror
/// `STATE_VT` and `STATE_JT`.
///
/// The state update is a rank-`c` outer product, so a thread that owns one
/// cell spends two memory instructions per multiply-add. A 4x4 tile reads one
/// `float4` from each operand and does sixteen. See the kernel.
const STATE_VT: u32 = 4;
const STATE_JT: u32 = 4;
/// Threads spanning the block's value band and its column band. Mirror
/// `STATE_VLANES` and `STATE_JLANES`; the block is their product.
const STATE_VLANES: u32 = 8;
const STATE_JLANES: u32 = 32;

const GDN_CHUNKED_SRC: &str = r#"
extern "C" {

// Sum across a block of up to 1024 threads. Identical to the recurrent
// kernel's reduction, and identical for the same reason: warp shuffles then
// one pass through shared memory. The tree order differs from the reference's
// sequential sum, which is one of the two sources of disagreement between this
// kernel and the CPU.
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

// L2-normalize q and k for every (token, query/key head), folding the
// 1/sqrt(head_dim) output scale into q.
//
// Same arithmetic as the recurrent form's normalizer, batched over the
// sequence: one block per (token, qk head) pair instead of one per qk head.
// The scale goes on q alone because q appears only in the final dot product
// and never in the state update.
//
// grid: seq_len * qk_heads blocks. block: head_dim threads.
__global__ void gdn_chunk_normalize_qk(
    const float* __restrict__ q_in,
    const float* __restrict__ k_in,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    int head_dim,
    float eps,
    float scale
) {
    extern __shared__ float scratch[];
    long long base = (long long)blockIdx.x * head_dim;
    int j = threadIdx.x;

    float qv = q_in[base + j];
    float kv = k_in[base + j];

    float q_sq = block_reduce_sum(qv * qv, scratch);
    __syncthreads();
    float k_sq = block_reduce_sum(kv * kv, scratch);
    __syncthreads();

    // `1/max(sqrt(sum), eps)`, ggml_l2_norm's formula verbatim — the epsilon
    // floors the norm and never binds in normal operation. See
    // `super::gdn`'s normalizer for the measurement that rejected
    // `1/sqrt(sum + eps)`.
    //
    // 1.0f/sqrtf rather than rsqrtf, matching the reference's division, so
    // that the disagreement stays attributable to the reduction order alone.
    q_out[base + j] = qv * (1.0f / fmaxf(sqrtf(q_sq), eps)) * scale;
    k_out[base + j] = kv * (1.0f / fmaxf(sqrtf(k_sq), eps));
}

// The whole sequence as a sequential scan, with the state in registers.
//
// This is the prefill path. The chunked kernels below it are the reference
// implementation and stay gated by their own differential tests, but they are
// not what a forward pass runs, and the reason is arithmetic rather than
// engineering.
//
// At head_dim 128 and chunk 64 the chunked form does about 42.3 M multiply-adds
// per head per 512 tokens against the scan's `4 * D^2` per token, which is
// 33.5 M -- **26% more work**. Chunking is not a FLOP reduction here; it is a
// reshaping that turns the recurrence into matmuls, and it pays only when the
// matmul shape buys tensor cores. Turing has no fp32 tensor cores, so on this
// card in fp32 the chunked form is pure overhead. llama.cpp reached the same
// conclusion: its CUDA gated-delta op is a scan, and the chunked form survives
// only as the slower ggml-graph fallback.
//
// grid: (value_heads, head_dim / SCAN_WARPS). block: (32, SCAN_WARPS).
//
// **One warp owns one (value head, value index) pair for the whole sequence.**
// Lane `l` holds `S[vi][l], S[vi][l + 32], ...` -- `head_dim / 32` floats, four
// at this geometry -- and never writes them out until the last token. The
// contraction of both `S k` and `S q` is over `j`, which is exactly the axis
// the lanes span, so both reductions are warp shuffles: no shared memory, no
// `__syncthreads`, and no global round trip for the state at any point in the
// sequence. The chunked path wrote and re-read the whole state eight times per
// layer to achieve the same thing.
//
// Every warp of a head re-reads the same `q` and `k` rows, which is a 128-fold
// read amplification and is deliberate: the per-token working set is a few
// tens of KiB across all heads and never leaves L1, and sharing it through
// shared memory would reintroduce the barriers this shape exists to avoid.
#define SCAN_WARPS 4
// Head dimensions one lane carries. `head_dim / 32`, bounded so the state is a
// register array indexed by constants after unrolling -- a runtime bound
// spills it to local memory and the whole design with it.
#define SCAN_MAXR 4

__global__ void gdn_scan_prefill(
    float* __restrict__ state,
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ log_decay,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int head_dim,
    int value_heads,
    int qk_heads,
    int seq_len
) {
    int h  = blockIdx.x;
    int vi = blockIdx.y * SCAN_WARPS + threadIdx.y;
    if (vi >= head_dim) return;
    int lane = threadIdx.x;
    int nr = head_dim >> 5;

    // Modulo, not division. See `super::gdn`'s module docs.
    int hq = h % qk_heads;

    float sreg[SCAN_MAXR];
    long long sbase = ((long long)h * head_dim + vi) * head_dim;
    #pragma unroll
    for (int r = 0; r < SCAN_MAXR; ++r) {
        sreg[r] = (r < nr) ? state[sbase + r * 32 + lane] : 0.0f;
    }

    for (int t = 0; t < seq_len; ++t) {
        long long ht = (long long)t * value_heads + h;
        long long qk = ((long long)t * qk_heads + hq) * head_dim;

        float kj[SCAN_MAXR], qj[SCAN_MAXR];
        #pragma unroll
        for (int r = 0; r < SCAN_MAXR; ++r) {
            if (r < nr) {
                kj[r] = k_norm[qk + r * 32 + lane];
                qj[r] = q_norm[qk + r * 32 + lane];
            } else {
                kj[r] = 0.0f;
                qj[r] = 0.0f;
            }
        }

        // `expf`, not `__expf`: the decay multiplies the whole state every
        // token, so a systematic bias compounds over the context rather than
        // averaging out. Same choice, and same reasoning, as the decode step.
        float decay = expf(log_decay[ht]);

        // 1. Decay, held in registers rather than written and re-read.
        // 2. Delta correction against the *decayed* state.
        float predicted = 0.0f;
        #pragma unroll
        for (int r = 0; r < SCAN_MAXR; ++r) {
            sreg[r] *= decay;
            predicted += sreg[r] * kj[r];
        }
        for (int off = 16; off > 0; off >>= 1) {
            predicted += __shfl_xor_sync(0xffffffff, predicted, off);
        }
        float vcorr = beta[ht] * (v[ht * head_dim + vi] - predicted);

        // 3. Outer-product update, and 4. the output from the *updated* state,
        //    in one pass: the freshly written state is consumed for the output
        //    while it is still in the register.
        float o = 0.0f;
        #pragma unroll
        for (int r = 0; r < SCAN_MAXR; ++r) {
            sreg[r] += vcorr * kj[r];
            o += sreg[r] * qj[r];
        }
        for (int off = 16; off > 0; off >>= 1) {
            o += __shfl_xor_sync(0xffffffff, o, off);
        }
        if (lane == 0) out[ht * head_dim + vi] = o;
    }

    #pragma unroll
    for (int r = 0; r < SCAN_MAXR; ++r) {
        if (r < nr) state[sbase + r * 32 + lane] = sreg[r];
    }
}


// The two intra-chunk Gram matrices, per query/key head:
//
//   kk[hq][t][i] = k_i . k_t      (the delta rule's key interference)
//   kq[hq][t][i] = k_i . q_t      (the intra-chunk causal attention)
//
// Both are computed over the full C x C square even though only the lower
// triangle is read. Branching to skip the upper half saves half the work of a
// kernel that is not the bottleneck, and costs a divergent warp plus an
// uninitialised-memory hazard if a later reader's bounds ever slip.
//
// beta_t is deliberately *not* folded into kk here: it is per value head, and
// two value heads share each qk head, so folding it in would double the work
// this kernel exists to halve.
//
// grid: (C, qk_heads). block: C threads, one per source token i.
__global__ void gdn_chunk_gram(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    float* __restrict__ kk,
    float* __restrict__ kq,
    int head_dim,
    int qk_heads,
    int chunk_start,
    int c
) {
    // k_t and q_t, staged once so the C threads broadcast from shared rather
    // than each re-reading the same 512 bytes from global.
    extern __shared__ float staged[];
    float* sk = staged;
    float* sq = staged + head_dim;

    int t = blockIdx.x;
    int hq = blockIdx.y;
    long long base_t = ((long long)(chunk_start + t) * qk_heads + hq) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        sk[d] = k_norm[base_t + d];
        sq[d] = q_norm[base_t + d];
    }
    __syncthreads();

    int i = threadIdx.x;
    long long base_i = ((long long)(chunk_start + i) * qk_heads + hq) * head_dim;
    float dot_kk = 0.0f;
    float dot_kq = 0.0f;
    // Sequential ascending accumulation over the head dimension, which is the
    // reference's order too — these two dot products are the one part of the
    // chunked form where the device and the host agree operand for operand.
    // The `float4` below is a **load** widening and nothing else: the four
    // components are consumed x, y, z, w, so the additions happen in the same
    // sequence they did one scalar at a time.
    //
    // It is worth widening because thread `i` owns a whole row and consecutive
    // threads are `qk_heads * head_dim` floats apart, so this read cannot be
    // coalesced across the warp however it is written. What it can be is
    // wider: 16 bytes of every 32-byte sector used instead of 4, and a quarter
    // of the load instructions. `head_dim` is validated to be a multiple of
    // 32 and every base offset is a multiple of it, so the alignment holds by
    // construction.
    const float4* ki4 = (const float4*)(k_norm + base_i);
    int nd4 = head_dim >> 2;
    for (int d4 = 0; d4 < nd4; ++d4) {
        float4 kv = ki4[d4];
        int d = d4 << 2;
        dot_kk += kv.x * sk[d];
        dot_kq += kv.x * sq[d];
        dot_kk += kv.y * sk[d + 1];
        dot_kq += kv.y * sq[d + 1];
        dot_kk += kv.z * sk[d + 2];
        dot_kq += kv.z * sq[d + 2];
        dot_kk += kv.w * sk[d + 3];
        dot_kq += kv.w * sq[d + 3];
    }

    long long o = ((long long)hq * c + t) * c + i;
    kk[o] = dot_kk;
    kq[o] = dot_kq;
}

// The two products against the state carried into this chunk:
//
//   sik[h][t][vi]  = sum_j S_in[h][vi][j] k_t[j]     (what the state predicts)
//   oint[h][t][vi] = sum_j S_in[h][vi][j] q_t[j]     (the inter-chunk output)
//
// Must run before gdn_chunk_solve_and_apply overwrites the state. Splitting it
// out rather than fusing it is what lets the fused kernel spend its whole
// shared-memory budget on U.
//
// grid: (C, value_heads). block: head_dim threads, one per value index.
// Tokens one block carries. The state row this kernel contracts against does
// not depend on `t`, so a block per token re-read the whole `head_dim x
// head_dim` state matrix `c` times per chunk — 134 MB per chunk at Qwen3.6's
// geometry, for 2 MB of distinct state. Carrying INTER_TT tokens divides both
// that traffic and the load instructions that fetch it by INTER_TT.
//
// Eight and not sixteen: the accumulators are 2 * INTER_TT registers per
// thread and the grid is `ceil(c / INTER_TT) * value_heads` blocks. At c = 64
// and 32 value heads, eight gives 256 blocks over 72 SMs; sixteen would halve
// that to 128 and buy only another factor of two on traffic that is already
// L2-resident.
#define INTER_TT 16

__global__ void gdn_chunk_inter(
    const float* __restrict__ state,
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    float* __restrict__ sik,
    float* __restrict__ oint,
    int head_dim,
    int qk_heads,
    int chunk_start,
    int c
) {
    extern __shared__ float staged[];
    float* sk = staged;                             // [INTER_TT][head_dim]
    float* sq = staged + INTER_TT * head_dim;       // [INTER_TT][head_dim]

    int t0 = blockIdx.x * INTER_TT;
    int h = blockIdx.y;
    int vi = threadIdx.x;
    // Modulo, not division: llama.cpp tiles the query/key heads across the
    // value heads. See `super::gdn`'s module docs.
    int hq = h % qk_heads;

    #pragma unroll
    for (int u = 0; u < INTER_TT; ++u) {
        int t = t0 + u;
        if (t < c) {
            long long base_t = ((long long)(chunk_start + t) * qk_heads + hq) * head_dim;
            sk[u * head_dim + vi] = k_norm[base_t + vi];
            sq[u * head_dim + vi] = q_norm[base_t + vi];
        } else {
            // Multiplied into accumulators that are never stored, so the
            // value only has to be finite.
            sk[u * head_dim + vi] = 0.0f;
            sq[u * head_dim + vi] = 0.0f;
        }
    }
    __syncthreads();

    const float* row = state + ((long long)h * head_dim + vi) * head_dim;
    float acc_k[INTER_TT];
    float acc_q[INTER_TT];
    #pragma unroll
    for (int u = 0; u < INTER_TT; ++u) { acc_k[u] = 0.0f; acc_q[u] = 0.0f; }

    // `s` is loaded once and multiplied into every carried token, which is the
    // whole point: the inner loop is now INTER_TT multiply-adds per state
    // element instead of one. The `float4` widens the load only — the four
    // components are consumed x, y, z, w, so each accumulator still sums over
    // `j` ascending.
    const float4* row4 = (const float4*)row;
    int nj4 = head_dim >> 2;
    for (int j4 = 0; j4 < nj4; ++j4) {
        float4 sv = row4[j4];
        int j = j4 << 2;
        // The staged reads are `float4` for the same reason the state read
        // is: eight scalar shared loads per eight multiply-adds is one memory
        // instruction per unit of arithmetic. Widening them is a load
        // widening only — the components are still consumed x, y, z, w, so
        // each accumulator sums over `j` ascending exactly as before.
        #pragma unroll
        for (int u = 0; u < INTER_TT; ++u) {
            float4 kv = *(const float4*)(sk + u * head_dim + j);
            float4 qv = *(const float4*)(sq + u * head_dim + j);
            acc_k[u] += sv.x * kv.x;
            acc_q[u] += sv.x * qv.x;
            acc_k[u] += sv.y * kv.y;
            acc_q[u] += sv.y * qv.y;
            acc_k[u] += sv.z * kv.z;
            acc_q[u] += sv.z * qv.z;
            acc_k[u] += sv.w * kv.w;
            acc_q[u] += sv.w * qv.w;
        }
    }

    #pragma unroll
    for (int u = 0; u < INTER_TT; ++u) {
        int t = t0 + u;
        if (t < c) {
            long long o = ((long long)h * c + t) * head_dim + vi;
            sik[o] = acc_k[u];
            oint[o] = acc_q[u];
        }
    }
}

// The chunk: solve for U', emit every token's output, advance the state.
//
// grid: (value_heads,). block: head_dim threads, one per value index vi.
// shared: U'[c][head_dim], then gcum[c], then decay[c].
//
// The unknown is u'_t = lambda_t u_t, not u_t. See the module docs for the
// substitution and for why the textbook `u_t` form is unusable in fp32 on this
// model: it needs 1/lambda_t, and lambda_t reaches 2.5e-42 by the second token
// of block 0. Here `gcum[t] = sum_{i<=t} log_decay_i` is kept in log space and
// every decay factor is `expf` of a non-positive argument, so nothing can
// overflow and an underflow to +0 is the correct limit.
//
// The order of operations is load-bearing in exactly the way the recurrent
// form's is, restated for a whole chunk at once:
//
//   1. The carried-in state is multiplied by lambda_t before the correction is
//      formed against it. That is the chunked form's "decay the state first",
//      written in the frame where the unknown is u'_t rather than u_t: the
//      value stays put and the state is brought forward to meet it. Dropping
//      it leaves a finite, fluent, different model.
//   2. The substitution sums over i < t, strictly. Token t's correction is
//      formed against the state as it stood *before* token t updated it, and
//      each earlier u'_i is carried forward by exp(gcum[t] - gcum[i]) — the
//      decay accumulated strictly between token i and token t.
//   3. The output sums over i <= t, inclusively. Token t's output is read from
//      the state *after* token t's own update. Making the two triangles agree
//      — the obvious "simplification" — is precisely the recurrent kernel's
//      read-before-update bug, transposed.
// Value indices one solve block owns. Mirrors `SOLVE_VB` on the Rust side.
//
// `vi` is a pure spectator in the forward substitution — thread `vi` reads
// only its own column of U' and never another's — so the axis splits freely
// across blocks. Splitting it four ways turns 32 blocks into 128, which is
// what puts the kernel on most of the 72 SMs instead of fewer than half.
// `gcum` is recomputed per block; that is 64 sequential adds against a kernel
// whose shortest section is 2,016 iterations.
#define VB 32

// Token positions the block's *output* section works on at once.
//
// Sections 1 and 2 are a forward substitution and are sequential in `t`, so
// they run on one VB-wide row of threads and the rest of the block waits at
// the barriers. Section 3 is not: every token's output reads a U' that is
// already final, so `SOLVE_TT` of them can be computed side by side. The block
// is `VB * SOLVE_TT` threads and the launch grid is unchanged, which takes the
// kernel from 128 warps -- about 5% of what 72 SMs hold -- to 1,024 for the
// section that is nearly half its arithmetic.
#define SOLVE_TT 8

__global__ void gdn_chunk_solve_and_apply(
    float* __restrict__ uprime,
    float* __restrict__ lambda_last,
    const float* __restrict__ v,
    const float* __restrict__ log_decay,
    const float* __restrict__ beta,
    const float* __restrict__ k_norm,
    const float* __restrict__ kk,
    const float* __restrict__ kq,
    const float* __restrict__ sik,
    const float* __restrict__ oint,
    float* __restrict__ out,
    int head_dim,
    int value_heads,
    int qk_heads,
    int chunk_start,
    int c
) {
    extern __shared__ float shared[];
    float* u     = shared;                              // [c][VB], holds U'
    float* gcum  = shared + (long long)c * VB;          // [c], log space
    float* decay = gcum + c;                            // [SOLVE_TT][c]

    int h = blockIdx.x;
    // `vl` is the value index within the band and `tsub` the output section's
    // token lane. Sections 1, 2 and 4a use `vl` alone and idle the rest.
    int vl = threadIdx.x & (VB - 1);
    int tsub = threadIdx.x / VB;
    int vi = blockIdx.y * VB + vl;
    // Modulo, not division. See `super::gdn`'s module docs.
    int hq = h % qk_heads;

    // Cumulative in-chunk log-decay, kept in log space. One thread,
    // accumulated sequentially in the reference's exact order: a parallel
    // prefix scan would sum the log-decays in a different association and give
    // a different decay for the same input, which is a gratuitous divergence
    // on a quantity that multiplies every output in the chunk.
    //
    // Exponentiating here — the previous `lambda[t] = expf(running)` — is what
    // made this kernel unusable: `running` reaches -91.58 on this model, so
    // `lambda[t]` underflows and the `v_t / lambda[t]` it fed overflowed to
    // inf and then NaN. Everything below exponentiates a *difference* of two
    // gcum entries instead, which is bounded above by 1 for non-positive
    // log-decays.
    if (threadIdx.x == 0) {
        float running = 0.0f;
        for (int t = 0; t < c; ++t) {
            running += log_decay[(long long)(chunk_start + t) * value_heads + h];
            gcum[t] = running;
        }
    }
    __syncthreads();

    // --- 1/2. Forward substitution for U', one row per iteration. ----------
    //
    // u'_t = W'_t - beta_t * sum_{i<t} (k_i . k_t) exp(gcum_t - gcum_i) u'_i,
    // W'_t = beta_t * (v_t - exp(gcum_t) * (S_in k_t)).
    //
    // Thread vi owns column vi of U' throughout, so the inner reduction is a
    // per-thread sequential loop over i with no cross-thread communication —
    // only the __syncthreads() that publishes row t before row t+1 reads it.
    // u[i * VB + vl] is bank-conflict free: consecutive vl land in
    // consecutive banks.
    //
    // decay[i] is rebuilt cooperatively each iteration rather than recomputed
    // per thread: the row is `c` exponentials shared by all `head_dim` threads,
    // so computing it once costs 1/head_dim of the alternative on a kernel
    // whose inner loop is otherwise pure multiply-add.
    for (int t = 0; t < c; ++t) {
        long long ht = (long long)(chunk_start + t) * value_heads + h;
        float beta_t = beta[ht];
        const float* kk_row = kk + ((long long)hq * c + t) * c;

        // The whole block fills `decay`, and it fills the *whole* coefficient
        // rather than the exponential alone.
        //
        // `beta_t * kk_row[i] * decay[i]` does not depend on `vl`, so the
        // inner loop below was making all VB lanes recompute the same two
        // products and reload the same `kk_row[i]` from global. Hoisting them
        // here leaves that loop one shared read and one multiply-add, which
        // matters more than the arithmetic saved: sections 1 and 2 are a
        // forward substitution, sequential in `t`, running on one VB-wide row
        // of threads, and they are this kernel's critical path.
        //
        // Bit-identical. `acc -= beta_t * kk_row[i] * decay[i] * u[...]`
        // associates left to right, so `(beta_t * kk_row[i]) * expf(...)` is
        // the same product in the same order, formed once instead of VB
        // times.
        for (int i = threadIdx.x; i < t; i += blockDim.x) {
            decay[i] = beta_t * kk_row[i] * expf(gcum[t] - gcum[i]);
        }
        __syncthreads();

        if (tsub == 0) {
        float v_t = v[ht * head_dim + vi];

        float acc = beta_t * (v_t - expf(gcum[t]) * sik[((long long)h * c + t) * head_dim + vi]);

        for (int i = 0; i < t; ++i) {
            acc -= decay[i] * u[i * VB + vl];
        }

        u[t * VB + vl] = acc;
        }
        // Publishes row t, and holds every thread until the whole block is
        // done reading `decay` before the next iteration overwrites it.
        __syncthreads();
    }

    // --- 3. Per-token output, from the state *including* token t's update. --
    for (int t0 = 0; t0 < c; t0 += SOLVE_TT) {
        int t = t0 + tsub;
        // One `decay` row per token lane, so the SOLVE_TT tokens in flight do
        // not overwrite each other's.
        float* dec = decay + tsub * c;
        if (t < c) {
            // As in sections 1 and 2: `kq_row[i]` does not depend on `vl`, so
            // it is folded into the row here instead of being reloaded from
            // global by all VB lanes. `kq_row[i] * dec[i] * u[...]` associates
            // left to right, so this is the same product in the same order.
            const float* kq_row = kq + ((long long)hq * c + t) * c;
            for (int i = vl; i <= t; i += VB) {
                dec[i] = kq_row[i] * expf(gcum[t] - gcum[i]);
            }
        }
        __syncthreads();

        if (t < c) {
        float o_intra = 0.0f;
        for (int i = 0; i <= t; ++i) {
            o_intra += dec[i] * u[i * VB + vl];
        }
        float o_inter = expf(gcum[t]) * oint[((long long)h * c + t) * head_dim + vi];
        long long ht = (long long)(chunk_start + t) * value_heads + h;
        out[ht * head_dim + vi] = o_inter + o_intra;
        }
        __syncthreads();
    }

    // --- 4a. Publish U' with the chunk-end decay folded in. ---------------
    //
    // `r_i = exp(gcum_{c-1} - gcum_i)` is llama.cpp's `g_diff = g_last -
    // g_cum` exactly (`delta-net-base.cpp:201-227`), reached from the
    // substitution above rather than transcribed. Folding it here rather than
    // inside the state update costs `c` multiplies per column instead of
    // `c * head_dim`.
    //
    // The state update itself is `gdn_chunk_state_update`, a separate launch,
    // and the reason is occupancy. This kernel has one thread per (head, vi)
    // — 4,096 threads, 128 warps, about 5% of what 72 SMs can hold — because
    // `vi` is the only axis sections 1 through 3 are parallel over. The state
    // update is parallel over (head, vi, j) as well: 524,288 independent
    // outputs. Leaving it here would run two thirds of this kernel's
    // arithmetic at a twentieth of the machine.
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        decay[i] = expf(gcum[c - 1] - gcum[i]);
    }
    __syncthreads();

    // Every `i` is independent, so this walks the token lanes as well.
    for (int i = tsub; i < c; i += SOLVE_TT) {
        uprime[((long long)h * c + i) * head_dim + vi] = u[i * VB + vl] * decay[i];
    }
    if (threadIdx.x == 0 && blockIdx.y == 0) lambda_last[h] = expf(gcum[c - 1]);
}

// The chunk-end state update, split out of the solve for occupancy:
//
//   S[h][vi][j] = lambda_last[h] * S[h][vi][j] + sum_i U'[h][i][vi] k[i][j]
//
// grid: (value_heads, head_dim / STATE_VB, head_dim / STATE_JB), one thread
// per (vi, j) pair. Every output is independent, so this saturates where its
// parent could not.
//
// The `i` loop runs ascending and the update is a single fused
// `lambda * S + acc`, both as the solve had them: reassociating the sum or
// splitting the fused multiply-add into a scale pass and an accumulate pass
// would round differently on a value that is carried into every later chunk.
#define STATE_VT 4
// State columns one thread owns, and value indices it owns. Both 4, so the
// two operands of the outer product are one `float4` each and the thread does
// sixteen multiply-adds with them.
//
// The first version gave each thread one `(vi, j)` cell and looped `i`:
//
//   float kv = k_norm[...j];              // one global load
//   acc += su[i * STATE_VB + vl] * kv;    // one shared load, one FMA
//
// Two memory instructions per multiply-add, on a part that issues 4 of the
// former and 64 of the latter per SM per clock. The state update is a rank-`c`
// outer product -- `S[vi][j] += sum_i u[i][vi] * k[i][j]` -- and neither
// operand depends on the other's index, so a 4x4 register tile reads two
// `float4`s and does sixteen multiply-adds with them. Sixteen times the
// arithmetic per instruction.
#define STATE_JT 4
// Threads spanning the block's value band and its column band. The `j` lanes
// are the fast axis so that a warp's 32 `float4` column loads are 512
// contiguous bytes -- four full transactions -- while its `su` load is one
// address broadcast to all 32.
#define STATE_VLANES 8
#define STATE_JLANES 32
// The bands themselves. `STATE_VB` is unchanged at 32, so the staged `su`
// tile is the same 8 KiB it was; `STATE_JB` is now the whole 128-wide state
// row, which is what removes `grid.z` and with it the repeated staging that
// the sweep below was working around.
//
//   4   1,705.97 tok/s
//   8   1,727.74
//   16  1,721.89
//   32  1,710.30
//
// Those were measured with one cell per thread, where `STATE_JB` traded the
// number of passes over `uprime` against the block size. With a register tile
// the block covers 32x128 cells with the same 256 threads and `grid.z` is 1,
// so `uprime` is staged once per value band and the trade is gone.
#define STATE_VB (STATE_VLANES * STATE_VT)
#define STATE_JB (STATE_JLANES * STATE_JT)

__global__ void gdn_chunk_state_update(
    float* __restrict__ state,
    const float* __restrict__ uprime,
    const float* __restrict__ k_norm,
    const float* __restrict__ lambda_last,
    int head_dim,
    int qk_heads,
    int chunk_start,
    int c
) {
    extern __shared__ float su[];                       // [c][STATE_VB]

    int h  = blockIdx.x;
    int v0 = blockIdx.y * STATE_VB;
    int j0 = blockIdx.z * STATE_JB;
    int hq = h % qk_heads;

    // `j` is the fast axis: consecutive threads take consecutive columns, so
    // a warp's column loads are contiguous and its `su` load is a broadcast.
    int jl = threadIdx.x & (STATE_JLANES - 1);
    int vl = threadIdx.x / STATE_JLANES;
    int vi = v0 + vl * STATE_VT;
    int j  = j0 + jl * STATE_JT;

    // Staged once per block and read `c` times by every one of the block's
    // `j` lanes. A row past `head_dim` stages zeros, which contribute exactly
    // nothing rather than needing a guard in the inner loop.
    for (int idx = threadIdx.x; idx < c * STATE_VB; idx += blockDim.x) {
        int i = idx / STATE_VB;
        int lv = idx % STATE_VB;
        su[idx] = v0 + lv < head_dim
            ? uprime[((long long)h * c + i) * head_dim + v0 + lv]
            : 0.0f;
    }
    __syncthreads();

    // `head_dim` is a multiple of 32 and `vi` and `j` are multiples of 4, so
    // one bound checked here covers all four cells and both `float4` loads.
    int live = vi < head_dim && j < head_dim;

    float acc[STATE_VT][STATE_JT];
    #pragma unroll
    for (int a = 0; a < STATE_VT; ++a) {
        #pragma unroll
        for (int b = 0; b < STATE_JT; ++b) acc[a][b] = 0.0f;
    }

    if (live) {
        for (int i = 0; i < c; ++i) {
            float4 uu = *(const float4*)(su + i * STATE_VB + vl * STATE_VT);
            float4 kk = *(const float4*)(
                k_norm + ((long long)(chunk_start + i) * qk_heads + hq) * head_dim + j);
            const float* up = (const float*)&uu;
            const float* kp = (const float*)&kk;
            #pragma unroll
            for (int a = 0; a < STATE_VT; ++a) {
                #pragma unroll
                for (int b = 0; b < STATE_JT; ++b) acc[a][b] += up[a] * kp[b];
            }
        }

        float lam = lambda_last[h];
        #pragma unroll
        for (int a = 0; a < STATE_VT; ++a) {
            float* cell = state + ((long long)h * head_dim + vi + a) * head_dim + j;
            #pragma unroll
            for (int b = 0; b < STATE_JT; ++b) cell[b] = lam * cell[b] + acc[a][b];
        }
    }
}

}
"#;

/// Something went wrong compiling or launching a chunked GDN kernel.
#[derive(Debug)]
pub enum GdnChunkedError {
    /// NVRTC rejected the source, or the module failed to load.
    Compile(String),
    /// The driver failed.
    Driver(DriverError),
    /// A head dimension the kernel cannot service.
    UnsupportedHeadDim { head_dim: usize },
    /// Value heads are not a whole multiple of query/key heads.
    UnevenHeadGrouping { value_heads: usize, qk_heads: usize },
    /// A chunk length the kernel cannot service.
    UnsupportedChunkLen { chunk_len: usize },
    /// `U` plus the cumulative decay do not fit in a block's shared memory.
    SharedMemoryExceeded { needed: usize, available: usize },
    /// The sequence is longer than the scratch buffers were sized for.
    SequenceTooLong { seq_len: usize, capacity: usize },
}

impl std::fmt::Display for GdnChunkedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compile(m) => write!(f, "kernel compilation failed: {m}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::UnsupportedHeadDim { head_dim } => write!(
                f,
                "head_dim {head_dim} must be a positive multiple of 32 and at most 1024",
            ),
            Self::UnevenHeadGrouping {
                value_heads,
                qk_heads,
            } => write!(
                f,
                "{value_heads} value heads do not divide evenly among {qk_heads} query/key heads",
            ),
            Self::UnsupportedChunkLen { chunk_len } => write!(
                f,
                "chunk_len {chunk_len} must be non-zero and at most 1024 (it is the block width \
                 of the Gram kernel)",
            ),
            Self::SharedMemoryExceeded { needed, available } => write!(
                f,
                "the chunk needs {needed} bytes of shared memory per block, {available} available",
            ),
            Self::SequenceTooLong { seq_len, capacity } => write!(
                f,
                "sequence of {seq_len} tokens exceeds the scratch capacity of {capacity}",
            ),
        }
    }
}

impl std::error::Error for GdnChunkedError {}

impl From<DriverError> for GdnChunkedError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

/// Buffers reused across prefill calls.
///
/// Sized at construction from the longest sequence a chunk will ever carry, so
/// that prefill itself allocates nothing — `cudaMalloc` synchronises, and a
/// synchronising call on this path would both stall the other workers and make
/// graph capture impossible.
pub struct GdnChunkedScratch {
    /// `[max_seq_len][qk_heads][head_dim]`, `q` normalized and pre-scaled.
    q_norm: CudaSlice<f32>,
    /// `[max_seq_len][qk_heads][head_dim]`, `k` normalized.
    k_norm: CudaSlice<f32>,
    /// `[qk_heads][chunk_len][chunk_len]`, `k_i . k_t`.
    kk: CudaSlice<f32>,
    /// `[qk_heads][chunk_len][chunk_len]`, `k_i . q_t`.
    kq: CudaSlice<f32>,
    /// `[value_heads][chunk_len][head_dim]`, `S_in k_t`.
    sik: CudaSlice<f32>,
    /// `[value_heads][chunk_len][head_dim]`, `S_in q_t`.
    oint: CudaSlice<f32>,
    /// `[value_heads][chunk_len][head_dim]`, the solved `U'` with the
    /// chunk-end decay `r_i` already folded in.
    ///
    /// The solve holds `U'` in shared memory while it needs it and publishes
    /// it here for `gdn_chunk_state_update`, which is a separate launch
    /// because the state update is parallel over one more axis than the solve
    /// is. See the kernel.
    uprime: CudaSlice<f32>,
    /// `[value_heads]`, `exp(gcum[c-1])` for the chunk just solved.
    lambda_last: CudaSlice<f32>,
    max_seq_len: usize,
}

impl GdnChunkedScratch {
    /// The longest sequence this scratch can service.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

/// Compiled chunked-parallel Gated DeltaNet kernels.
pub struct GdnChunkedKernels {
    normalize: CudaFunction,
    gram: CudaFunction,
    inter: CudaFunction,
    solve: CudaFunction,
    state_update: CudaFunction,
    scan: CudaFunction,
    head_dim: usize,
    value_heads: usize,
    qk_heads: usize,
    chunk_len: usize,
}

impl GdnChunkedKernels {
    /// Compile for a specific head geometry and chunk length.
    ///
    /// Geometry is fixed at construction, as it is in [`super::gdn`], because
    /// the model fixes it — validating once means the launch path has nothing
    /// left to reject.
    pub fn new(
        ctx: &Arc<CudaContext>,
        head_dim: usize,
        value_heads: usize,
        qk_heads: usize,
        chunk_len: usize,
    ) -> Result<Self, GdnChunkedError> {
        // One thread per key index inside a warp-shuffle reduction, so the head
        // dimension must be a whole number of warps and fit in one block.
        if head_dim == 0 || !head_dim.is_multiple_of(32) || head_dim > 1024 {
            return Err(GdnChunkedError::UnsupportedHeadDim { head_dim });
        }
        if qk_heads == 0 || !value_heads.is_multiple_of(qk_heads) {
            return Err(GdnChunkedError::UnevenHeadGrouping {
                value_heads,
                qk_heads,
            });
        }
        // The Gram kernel runs one thread per source token in the chunk.
        if chunk_len == 0 || chunk_len > 1024 {
            return Err(GdnChunkedError::UnsupportedChunkLen { chunk_len });
        }
        let needed = shared_bytes_for_solve(chunk_len);
        if needed > MAX_SHARED_BYTES {
            return Err(GdnChunkedError::SharedMemoryExceeded {
                needed,
                available: MAX_SHARED_BYTES,
            });
        }

        let ptx = compile(GDN_CHUNKED_SRC, "gdn_chunked").map_err(GdnChunkedError::Compile)?;
        let module = ctx.load_module(ptx)?;
        Ok(Self {
            normalize: module.load_function("gdn_chunk_normalize_qk")?,
            gram: module.load_function("gdn_chunk_gram")?,
            inter: module.load_function("gdn_chunk_inter")?,
            solve: module.load_function("gdn_chunk_solve_and_apply")?,
            state_update: module.load_function("gdn_chunk_state_update")?,
            scan: module.load_function("gdn_scan_prefill")?,
            head_dim,
            value_heads,
            qk_heads,
            chunk_len,
        })
    }

    /// How many value heads share each query/key head.
    ///
    /// The *count*, not the mapping: value head `h` reads query/key head
    /// `h % qk_heads`, so the heads sharing a query/key head are `hq`,
    /// `hq + qk_heads`, … and not a contiguous run.
    pub fn heads_per_kv(&self) -> usize {
        self.value_heads / self.qk_heads
    }

    /// Tokens processed per triangular solve.
    pub fn chunk_len(&self) -> usize {
        self.chunk_len
    }

    /// The `1/sqrt(head_dim)` scale folded into `q`.
    pub fn output_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// Bytes of recurrent state for one sequence at this geometry.
    pub fn state_bytes(&self) -> usize {
        self.value_heads * self.head_dim * self.head_dim * size_of::<f32>()
    }

    /// Allocate scratch for sequences of up to `max_seq_len` tokens.
    pub fn scratch(
        &self,
        stream: &Arc<CudaStream>,
        max_seq_len: usize,
    ) -> Result<GdnChunkedScratch, GdnChunkedError> {
        let qk = max_seq_len * self.qk_heads * self.head_dim;
        let gram = self.qk_heads * self.chunk_len * self.chunk_len;
        let inter = self.value_heads * self.chunk_len * self.head_dim;
        Ok(GdnChunkedScratch {
            q_norm: stream.alloc_zeros::<f32>(qk)?,
            k_norm: stream.alloc_zeros::<f32>(qk)?,
            kk: stream.alloc_zeros::<f32>(gram)?,
            kq: stream.alloc_zeros::<f32>(gram)?,
            sik: stream.alloc_zeros::<f32>(inter)?,
            oint: stream.alloc_zeros::<f32>(inter)?,
            uprime: stream.alloc_zeros::<f32>(inter)?,
            lambda_last: stream.alloc_zeros::<f32>(self.value_heads)?,
            max_seq_len,
        })
    }

    /// L2-normalize the whole sequence's queries and keys into the scratch.
    ///
    /// Shared by [`Self::prefill`] and [`Self::scan`]; the query is also
    /// pre-scaled by `1/sqrt(head_dim)` here, which is why neither mixer
    /// applies an output scale of its own.
    fn normalize_sequence(
        &self,
        stream: &Arc<CudaStream>,
        scratch: &mut GdnChunkedScratch,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        seq_len: usize,
    ) -> Result<(), GdnChunkedError> {
        // One float per warp, which is all `block_reduce_sum` stores.
        let reduce_shared = (self.head_dim.div_ceil(32) * size_of::<f32>()) as u32;
        let norm_cfg = LaunchConfig {
            grid_dim: ((seq_len * self.qk_heads) as u32, 1, 1),
            block_dim: (self.head_dim as u32, 1, 1),
            shared_mem_bytes: reduce_shared,
        };
        let head_dim = self.head_dim as i32;
        let eps = L2_EPS;
        let scale = self.output_scale();
        let mut builder = stream.launch_builder(&self.normalize);
        builder
            .arg(q)
            .arg(k)
            .arg(&mut scratch.q_norm)
            .arg(&mut scratch.k_norm)
            .arg(&head_dim)
            .arg(&eps)
            .arg(&scale);
        // SAFETY: one block per (token, qk head) pair and one thread per head
        // element, over inputs and outputs of `seq_len * qk_heads * head_dim`.
        // Shared memory covers one float per warp, all the reduction writes.
        unsafe { builder.launch(norm_cfg) }?;
        Ok(())
    }

    /// The whole sequence as a sequential scan, with the state in registers.
    ///
    /// This is what a forward pass runs. [`Self::prefill`] is the chunked
    /// reference implementation and keeps its own differential tests, but at
    /// this head dimension the chunked form does about 26% more arithmetic
    /// than the scan and only pays when its matmul shape buys tensor cores,
    /// which fp32 on `sm_75` does not have. See `gdn_scan_prefill`.
    ///
    /// The normalization pass is shared with the chunked path and runs first,
    /// unchanged: the scan consumes `q_norm` and `k_norm`, not `q` and `k`.
    #[allow(clippy::too_many_arguments)]
    pub fn scan(
        &self,
        stream: &Arc<CudaStream>,
        scratch: &mut GdnChunkedScratch,
        state: &mut CudaSlice<f32>,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        log_decay: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        seq_len: usize,
    ) -> Result<(), GdnChunkedError> {
        if seq_len > scratch.max_seq_len {
            return Err(GdnChunkedError::SequenceTooLong {
                seq_len,
                capacity: scratch.max_seq_len,
            });
        }
        if seq_len == 0 {
            return Ok(());
        }
        if self.head_dim > 32 * SCAN_MAXR {
            return Err(GdnChunkedError::UnsupportedHeadDim {
                head_dim: self.head_dim,
            });
        }

        let head_dim = self.head_dim as i32;
        let value_heads = self.value_heads as i32;
        let qk_heads = self.qk_heads as i32;

        self.normalize_sequence(stream, scratch, q, k, seq_len)?;

        let cfg = LaunchConfig {
            grid_dim: (
                self.value_heads as u32,
                (self.head_dim as u32).div_ceil(SCAN_WARPS),
                1,
            ),
            block_dim: (32, SCAN_WARPS, 1),
            shared_mem_bytes: 0,
        };
        let seq_i32 = seq_len as i32;
        let mut builder = stream.launch_builder(&self.scan);
        builder
            .arg(&mut *state)
            .arg(&scratch.q_norm)
            .arg(&scratch.k_norm)
            .arg(v)
            .arg(log_decay)
            .arg(beta)
            .arg(&mut *out)
            .arg(&head_dim)
            .arg(&value_heads)
            .arg(&qk_heads)
            .arg(&seq_i32);
        // SAFETY: one warp per (value head, value index), both covered by the
        // grid and the `vi >= head_dim` guard. The state index
        // `(h * head_dim + vi) * head_dim + r * 32 + lane` stays inside
        // `value_heads * head_dim * head_dim`; the token indices stay below
        // `seq_len`, which was checked against the scratch capacity and which
        // bounds every `q_norm`, `k_norm`, `v`, `log_decay`, `beta` and `out`
        // access. No shared memory is requested and none is indexed.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// Run the chunked form over `seq_len` tokens, advancing `state`.
    ///
    /// This is the chunked-parallel reference. [`Self::scan`] is what a
    /// forward pass runs; both are held to the same differential gate against
    /// the same two host forms, because neither is a reference for the other.
    ///
    /// `q` and `k` are `[seq_len][qk_heads][head_dim]` raw — not normalized,
    /// not scaled; this does both. `v` is `[seq_len][value_heads][head_dim]`,
    /// `log_decay` and `beta` are `[seq_len][value_heads]`, and `out` is
    /// `[seq_len][value_heads][head_dim]`.
    ///
    /// `state` is `[value_heads][head_dim][head_dim]` and is updated in place,
    /// so passing a non-zero state resumes from a prefix-cache hit and passing
    /// zeros starts a fresh sequence. On return it holds the state after the
    /// final token — the same value the recurrent form would have reached.
    ///
    /// Ten plain arguments rather than a config struct, for the same reason
    /// `chunked_forward` takes eight: every one is a distinct per-call tensor,
    /// not related configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill(
        &self,
        stream: &Arc<CudaStream>,
        scratch: &mut GdnChunkedScratch,
        state: &mut CudaSlice<f32>,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        log_decay: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        seq_len: usize,
    ) -> Result<(), GdnChunkedError> {
        if seq_len > scratch.max_seq_len {
            return Err(GdnChunkedError::SequenceTooLong {
                seq_len,
                capacity: scratch.max_seq_len,
            });
        }
        if seq_len == 0 {
            return Ok(());
        }

        let head_dim = self.head_dim as i32;
        let value_heads = self.value_heads as i32;
        let qk_heads = self.qk_heads as i32;

        self.normalize_sequence(stream, scratch, q, k, seq_len)?;

        // --- chunks, in order: they are dependent through the state -------
        let staged_shared = (2 * self.head_dim * size_of::<f32>()) as u32;
        // `gdn_chunk_inter` stages INTER_TT tokens' key and query vectors, not
        // one of each.
        let inter_shared = (2 * INTER_TT as usize * self.head_dim * size_of::<f32>()) as u32;
        let mut start = 0usize;
        while start < seq_len {
            let c = (start + self.chunk_len).min(seq_len) - start;
            let chunk_start = start as i32;
            let c_i32 = c as i32;

            let gram_cfg = LaunchConfig {
                grid_dim: (c as u32, self.qk_heads as u32, 1),
                block_dim: (c as u32, 1, 1),
                shared_mem_bytes: staged_shared,
            };
            let mut builder = stream.launch_builder(&self.gram);
            builder
                .arg(&scratch.q_norm)
                .arg(&scratch.k_norm)
                .arg(&mut scratch.kk)
                .arg(&mut scratch.kq)
                .arg(&head_dim)
                .arg(&qk_heads)
                .arg(&chunk_start)
                .arg(&c_i32);
            // SAFETY: the grid is (c, qk_heads) and the block is c threads, so
            // the Gram index `(hq * c + t) * c + i` stays inside
            // `qk_heads * chunk_len * chunk_len`, and the token index
            // `chunk_start + t` stays below `seq_len` by the loop bound.
            // Shared memory holds the two staged head vectors.
            unsafe { builder.launch(gram_cfg) }?;

            let inter_cfg = LaunchConfig {
                grid_dim: ((c as u32).div_ceil(INTER_TT), self.value_heads as u32, 1),
                block_dim: (self.head_dim as u32, 1, 1),
                shared_mem_bytes: inter_shared,
            };
            let mut builder = stream.launch_builder(&self.inter);
            builder
                .arg(&*state)
                .arg(&scratch.q_norm)
                .arg(&scratch.k_norm)
                .arg(&mut scratch.sik)
                .arg(&mut scratch.oint)
                .arg(&head_dim)
                .arg(&qk_heads)
                .arg(&chunk_start)
                .arg(&c_i32);
            // SAFETY: the grid is (c, value_heads) and the block is head_dim
            // threads, so the state index `(h * head_dim + vi) * head_dim + j`
            // stays inside `value_heads * head_dim * head_dim` and the output
            // index `(h * c + t) * head_dim + vi` inside
            // `value_heads * chunk_len * head_dim`.
            unsafe { builder.launch(inter_cfg) }?;

            let solve_cfg = LaunchConfig {
                grid_dim: (
                    self.value_heads as u32,
                    (self.head_dim as u32).div_ceil(SOLVE_VB),
                    1,
                ),
                block_dim: (SOLVE_VB * SOLVE_TT, 1, 1),
                shared_mem_bytes: shared_bytes_for_solve(c) as u32,
            };
            let mut builder = stream.launch_builder(&self.solve);
            builder
                .arg(&mut scratch.uprime)
                .arg(&mut scratch.lambda_last)
                .arg(v)
                .arg(log_decay)
                .arg(beta)
                .arg(&scratch.k_norm)
                .arg(&scratch.kk)
                .arg(&scratch.kq)
                .arg(&scratch.sik)
                .arg(&scratch.oint)
                .arg(&mut *out)
                .arg(&head_dim)
                .arg(&value_heads)
                .arg(&qk_heads)
                .arg(&chunk_start)
                .arg(&c_i32);
            // SAFETY: one block per (value head, SOLVE_VB-wide band of value
            // indices), SOLVE_VB threads per block, and shared memory sized
            // for exactly the `c * SOLVE_VB` entries of U' this band holds
            // plus the `c` cumulative log-decays and the `c` decay ratios the
            // kernel indexes. `uprime` is `value_heads * chunk_len * head_dim`
            // and `lambda_last` is `value_heads`. Every other global index is
            // bounded by the same reasoning as the two kernels above, and
            // `c >= 1` inside the loop so `gcum[c - 1]` is in range.
            unsafe { builder.launch(solve_cfg) }?;

            let update_cfg = LaunchConfig {
                grid_dim: (
                    self.value_heads as u32,
                    (self.head_dim as u32).div_ceil(STATE_VB),
                    (self.head_dim as u32).div_ceil(STATE_JB),
                ),
                block_dim: (STATE_VLANES * STATE_JLANES, 1, 1),
                shared_mem_bytes: (c * STATE_VB as usize * size_of::<f32>()) as u32,
            };
            let mut builder = stream.launch_builder(&self.state_update);
            builder
                .arg(&mut *state)
                .arg(&scratch.uprime)
                .arg(&scratch.k_norm)
                .arg(&scratch.lambda_last)
                .arg(&head_dim)
                .arg(&qk_heads)
                .arg(&chunk_start)
                .arg(&c_i32);
            // SAFETY: one thread per (value index, state column) pair, so the
            // state index `(h * head_dim + vi) * head_dim + j` covers
            // `value_heads * head_dim * head_dim` exactly once. Shared memory
            // holds the `c * STATE_VB` slice of `uprime` this block reads.
            unsafe { builder.launch(update_cfg) }?;

            start += c;
        }
        Ok(())
    }
}

/// Shared memory the fused solve kernel needs for a chunk of `c` tokens.
///
/// `U'` is `c x head_dim`, the cumulative log-decay is `c` more floats, and one
/// row of decay ratios is `c` more. Nothing else is resident — that is the
/// whole reason the triangular system is solved by substitution rather than by
/// materialising the `c x c` inverse, and the same reason the `c x c` decay
/// mask llama.cpp builds as a tensor is rebuilt here one row at a time.
fn shared_bytes_for_solve(c: usize) -> usize {
    // U', the cumulative log-decays, and one decay row per token lane.
    (c * SOLVE_VB as usize + c + SOLVE_TT as usize * c) * size_of::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The kernel source with `//` comments stripped.
    ///
    /// A structural assertion about what the *code* does must not be
    /// satisfiable — or defeatable — by a comment that quotes the very thing
    /// it forbids, which is exactly what the module docs above do when they
    /// explain the formulation this kernel replaced.
    fn code_only(src: &str) -> String {
        src.lines()
            .map(|l| match l.find("//") {
                Some(i) => &l[..i],
                None => l,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Where a snippet starts in the kernel source, or a failure naming it.
    fn at(needle: &str) -> usize {
        GDN_CHUNKED_SRC
            .find(needle)
            .unwrap_or_else(|| panic!("kernel source no longer contains `{needle}`"))
    }

    #[test]
    fn the_substitution_excludes_the_diagonal_and_the_output_includes_it() {
        // The single most important structural property of the chunked form,
        // and the one a well-meaning "make these two loops consistent" edit
        // destroys. `i < t` in the solve means token t's correction is formed
        // against the state before its own update; `i <= t` in the output means
        // token t reads the state after it. That asymmetry is the chunked
        // transcription of the recurrent kernel's decay-correct-update-read
        // order, and both halves are needed: making the solve inclusive
        // double-counts token t's own key, and making the output exclusive is
        // the classic read-before-update bug.
        assert!(GDN_CHUNKED_SRC.contains(
            "for (int i = 0; i < t; ++i) {\n            acc -= decay[i] * u[i * VB + vl];"
        ));
        // And the coefficient that was hoisted into `decay` is still the
        // exclusive one: the fill runs to `i < t` in the substitution and
        // `i <= t` in the output, which is where the asymmetry now lives.
        assert!(
            GDN_CHUNKED_SRC.contains("for (int i = threadIdx.x; i < t; i += blockDim.x) {"),
            "the substitution's coefficient row is no longer filled exclusively",
        );
        assert!(
            GDN_CHUNKED_SRC.contains("for (int i = vl; i <= t; i += VB) {"),
            "the output's coefficient row is no longer filled inclusively",
        );
        assert!(GDN_CHUNKED_SRC.contains(
            "for (int i = 0; i <= t; ++i) {\n            o_intra += dec[i] * u[i * VB + vl];"
        ));
    }

    #[test]
    fn the_state_is_brought_forward_to_the_value_rather_than_the_value_back_to_the_state() {
        // The chunked form's "decay the state first", in the frame where the
        // unknown is u'_t = lambda_t u_t: the carried-in prediction S_in k_t is
        // multiplied by exp(gcum_t) before the residual is formed, and v_t is
        // left alone. The equivalent-looking `v_t / lambda_t - S_in k_t` is the
        // same algebra and is *not* the same computation — it needs the
        // reciprocal of a cumulative decay that reaches 2.5e-42 on this model,
        // and overflows fp32. See the module docs.
        let rescale = at("float acc = beta_t * (v_t - expf(gcum[t]) * sik[");
        let subtract = at("acc -= decay[i] * u[i * VB + vl];");
        assert!(
            rescale < subtract,
            "the interference sum is subtracted before the carried-in state is decayed",
        );
    }

    #[test]
    fn no_cumulative_decay_is_ever_divided_by() {
        // The defect this kernel shipped with, asserted so it cannot come
        // back. Every decay factor must be `expf` of a cumulative log-decay or
        // of a difference of two, both non-positive for this model's gates, so
        // the worst case is underflow to +0 rather than overflow to inf.
        let code = code_only(GDN_CHUNKED_SRC);
        let solve = code
            .find("__global__ void gdn_chunk_solve_and_apply(")
            .expect("the fused solve kernel is still there");
        let body = &code[solve..];
        for forbidden in ["/ lambda", "v_t /", "1.0f / expf", "lambda[t] = expf"] {
            assert!(
                !body.contains(forbidden),
                "the solve kernel divides by a cumulative decay again: `{forbidden}`",
            );
        }
        // And the positive statement: the cumulative decay stays in log space.
        assert!(GDN_CHUNKED_SRC.contains("gcum[t] = running;"));
        // Both per-token rows now carry a lane-invariant coefficient folded
        // in, but the exponential itself is still a *difference* of two `gcum`
        // entries, which is the property this test exists to hold.
        assert!(
            GDN_CHUNKED_SRC.contains("decay[i] = beta_t * kk_row[i] * expf(gcum[t] - gcum[i]);")
        );
        assert!(GDN_CHUNKED_SRC.contains("dec[i] = kq_row[i] * expf(gcum[t] - gcum[i]);"));
        assert!(GDN_CHUNKED_SRC.contains("decay[i] = expf(gcum[c - 1] - gcum[i]);"));
    }

    #[test]
    fn the_inter_tile_constant_matches_the_kernel_define() {
        // The launch sizes shared memory from the Rust constant and the kernel
        // indexes it with the `#define`. If they disagree the kernel reads
        // past the staged tile, which is a silent wrong answer rather than a
        // fault, so this is checked rather than trusted.
        let needle = format!("#define INTER_TT {INTER_TT}\n");
        assert!(
            GDN_CHUNKED_SRC.contains(&needle),
            "the kernel's INTER_TT is not {INTER_TT}; the shared staging \
             `gdn_chunked_forward` allocates would be the wrong size",
        );
    }

    #[test]
    fn the_output_and_the_state_are_rescaled_by_the_right_cumulative_decay() {
        // Per-token output carries lambda_t on the *inter*-chunk term only —
        // the intra-chunk term already carries exp(gcum_t - gcum_i) per
        // summand, because the unknown solved for is lambda_i u_i. The
        // chunk-end state carries lambda_{c-1} on S_in and exp(gcum_{c-1} -
        // gcum_i) per summand. Swapping any of these is dimensionally
        // invisible and silently changes every layer's output after the first
        // chunk.
        assert!(GDN_CHUNKED_SRC.contains(
            "float o_inter = expf(gcum[t]) * oint[((long long)h * c + t) * head_dim + vi];"
        ));
        assert!(GDN_CHUNKED_SRC.contains("out[ht * head_dim + vi] = o_inter + o_intra;"));
        // `lambda_last` is now published by the solve and consumed by
        // `gdn_chunk_state_update`, which is a separate launch. Both halves are
        // asserted: computing it and applying it are in different kernels, so
        // an edit could plausibly drop either.
        assert!(GDN_CHUNKED_SRC.contains("lambda_last[h] = expf(gcum[c - 1]);"));
        assert!(GDN_CHUNKED_SRC.contains("float lam = lambda_last[h];"));
        assert!(GDN_CHUNKED_SRC.contains("cell[b] = lam * cell[b] + acc[a][b];"));
    }

    #[test]
    fn the_query_key_head_is_selected_by_modulo_not_division() {
        // Every kernel that indexes a query/key head must tile, not block:
        // llama.cpp's fused op is `fastmodulo(h_idx, n_k_heads)` and its
        // fallback broadcasts with `ggml_repeat_4d`. See `super::gdn`.
        //
        // Four, not three: `gdn_chunk_state_update` was split out of the solve
        // and reads `k_norm` itself, and `gdn_scan_prefill` — the kernel a
        // forward pass actually runs — makes the same choice independently of
        // all three. Each can get it wrong the same way, and the failure mode
        // is a model that still generates fluent text.
        assert_eq!(
            GDN_CHUNKED_SRC.matches("int hq = h % qk_heads;").count(),
            4,
            "gdn_scan_prefill, gdn_chunk_inter, gdn_chunk_solve_and_apply and \
             gdn_chunk_state_update must all use modulo",
        );
        assert!(
            !GDN_CHUNKED_SRC.contains("heads_per_kv"),
            "a kernel still takes the sharing ratio, which only division needs",
        );
    }

    #[test]
    fn the_l2_epsilon_floors_the_norm_rather_than_being_added_to_the_sum() {
        // Same invariant as the recurrent normalizer's, and the same reason:
        // `ggml_l2_norm` is `1/max(sqrt(sum), eps)` and `1/sqrt(sum + eps)`
        // shrinks the smallest-norm q/k by 2.148e-4 relative on this model.
        assert!(!GDN_CHUNKED_SRC.contains("q_sq + eps"));
        assert!(!GDN_CHUNKED_SRC.contains("k_sq + eps"));
    }

    #[test]
    fn the_state_update_consumes_the_solved_corrections_not_the_raw_residuals() {
        // S_out = lambda_{c-1} (S_in + sum_i u_i k_i^T) uses U, the solution of
        // the triangular system — not W, its right-hand side. Using W would be
        // the ungated delta rule with the intra-chunk key interference dropped,
        // which is a plausible-looking kernel that diverges from the recurrent
        // form only as keys within a chunk start to correlate.
        // The state update is a separate kernel now, so "before" is a data
        // dependency rather than a source ordering: the solve publishes U'
        // into `uprime` and `gdn_chunk_state_update` is the only reader.
        let publish = at("u[t * VB + vl] = acc;");
        let fold =
            at("uprime[((long long)h * c + i) * head_dim + vi] = u[i * VB + vl] * decay[i];");
        assert!(
            publish < fold,
            "U' is published to global before it is solved"
        );
        assert!(
            GDN_CHUNKED_SRC
                .contains("for (int b = 0; b < STATE_JT; ++b) acc[a][b] += up[a] * kp[b];"),
            "the state update must consume the solved U', not the raw residuals",
        );
    }

    #[test]
    fn the_scale_is_folded_into_q_only() {
        // Same invariant as the recurrent kernel's: on both, or on neither,
        // and every GDN layer's output is off by 11.3x or its square.
        assert!(
            GDN_CHUNKED_SRC
                .contains("q_out[base + j] = qv * (1.0f / fmaxf(sqrtf(q_sq), eps)) * scale;")
        );
        assert!(
            GDN_CHUNKED_SRC.contains("k_out[base + j] = kv * (1.0f / fmaxf(sqrtf(k_sq), eps));")
        );
    }

    #[test]
    fn beta_is_not_folded_into_the_shared_gram_matrix() {
        // kk is shared between the value heads that share a query/key head, and
        // beta is per value head. Folding beta into the Gram kernel would make
        // the two heads read each other's gate.
        let gram_start = at("__global__ void gdn_chunk_gram(");
        let gram_end = at("__global__ void gdn_chunk_inter(");
        let gram_body = &GDN_CHUNKED_SRC[gram_start..gram_end];
        assert!(
            !gram_body.contains("beta"),
            "the Gram kernel must not see beta; it is per value head",
        );
    }

    #[test]
    fn the_cumulative_decay_is_accumulated_sequentially() {
        // A parallel scan would associate the log-decay sum differently and
        // produce a different lambda for identical input, for no benefit on a
        // 64-element prefix that one thread computes while the rest of the
        // block waits on a single __syncthreads().
        assert!(GDN_CHUNKED_SRC.contains("running += log_decay["));
        assert!(GDN_CHUNKED_SRC.contains("gcum[t] = running;"));
    }

    #[test]
    fn no_inverse_is_materialised() {
        // The decision recorded in the module docs, asserted rather than only
        // described: nothing in the source builds an explicit inverse.
        assert!(!GDN_CHUNKED_SRC.contains("inv"));
        assert!(GDN_CHUNKED_SRC.contains("Forward substitution for U"));
    }

    #[test]
    fn shared_memory_at_the_real_geometry_fits_a_turing_block() {
        // chunk_len 64 over a SOLVE_VB-wide band, plus one decay row per
        // token lane: 10.5 KiB, well inside the 48 KiB a block gets without
        // the opt-in carve-out. It was 32.5 KiB when a block owned a whole
        // head, and 8.5 KiB when the output section ran one token at a time.
        let needed = shared_bytes_for_solve(64);
        assert_eq!(
            needed,
            (64 * SOLVE_VB as usize + 64 + SOLVE_TT as usize * 64) * 4,
        );
        // The block is this wide, and the decay rows have to keep up with it.
        assert_eq!(
            SOLVE_VB * SOLVE_TT,
            256,
            "the solve block is SOLVE_VB * SOLVE_TT threads",
        );
        assert!(needed < MAX_SHARED_BYTES, "{needed} bytes");
        // A 64x64 inverse *would* now fit beside it — 8.5 + 16 KiB — which is
        // why the module docs no longer offer shared memory as a reason to
        // prefer forward substitution. The arithmetic reason is untouched and
        // is asserted separately by `no_inverse_is_materialised`.
        assert!(needed + 64 * 64 * 4 < MAX_SHARED_BYTES);
    }

    #[test]
    fn geometry_is_validated_not_assumed() {
        // The checks that run without a device.
        assert!(
            GdnChunkedError::UnsupportedHeadDim { head_dim: 100 }
                .to_string()
                .contains("100")
        );
        assert!(
            GdnChunkedError::UnevenHeadGrouping {
                value_heads: 32,
                qk_heads: 5,
            }
            .to_string()
            .contains("32")
        );
        assert!(
            GdnChunkedError::UnsupportedChunkLen { chunk_len: 0 }
                .to_string()
                .contains('0')
        );
        assert!(
            GdnChunkedError::SharedMemoryExceeded {
                needed: 99_000,
                available: MAX_SHARED_BYTES,
            }
            .to_string()
            .contains("99000")
        );
        assert!(
            GdnChunkedError::SequenceTooLong {
                seq_len: 4096,
                capacity: 512,
            }
            .to_string()
            .contains("4096")
        );
    }
}
