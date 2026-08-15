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
//! as the kernel needs it:
//!
//! Within one chunk of `C` tokens, with `S_in` the state carried in,
//! `lambda_t = exp(sum_{i<=t} log_decay_i)` the cumulative in-chunk decay, and
//! unit-norm keys `k_t`:
//!
//! ```text
//! W[t]      = beta_t * (v_t / lambda_t  -  S_in k_t)
//! (I + A) U = W,   A[t][i] = beta_t (k_i . k_t) for i < t, 0 otherwise
//! o_t       = lambda_t * (S_in q_t + sum_{i<=t} u_i (k_i . q_t))
//! S_out     = lambda_{C-1} * (S_in + sum_i u_i k_i^T)
//! ```
//!
//! ## Shape
//!
//! Identical to the recurrent form: state `S` is `head_dim x head_dim` fp32
//! per value head, laid out `S[v][k]` with the key index contiguous, 32 value
//! heads sharing 16 query/key heads (`qk_head = v_head / 2`), `head_dim` 128,
//! `chunk_len` 64. The triangular system is therefore 64x64 per chunk per
//! head, and `U` is 64x128.
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
//! **Shared memory.** `M^-1` is `C x C` = 16 KiB at `C = 64`, and `U` is
//! `C x head_dim` = 32 KiB. Holding both resident is 48 KiB — exactly the
//! per-block shared-memory limit on Turing without the opt-in carve-out, with
//! nothing left for the cumulative decay or any staging, and it stops fitting
//! the moment `chunk_len` or `head_dim` moves. Forward substitution needs only
//! `U`.
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

/// The epsilon inside the L2 normalization.
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
/// the chunked geometry fits with room to spare (33 KiB at `chunk_len = 64`,
/// `head_dim = 128`).
pub const MAX_SHARED_BYTES: usize = 48 * 1024;

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

    // 1.0f/sqrtf rather than rsqrtf, matching the reference's division, so
    // that the disagreement stays attributable to the reduction order alone.
    q_out[base + j] = qv * (1.0f / sqrtf(q_sq + eps)) * scale;
    k_out[base + j] = kv * (1.0f / sqrtf(k_sq + eps));
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
    for (int d = 0; d < head_dim; ++d) {
        float ki = k_norm[base_i + d];
        dot_kk += ki * sk[d];
        dot_kq += ki * sq[d];
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
__global__ void gdn_chunk_inter(
    const float* __restrict__ state,
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    float* __restrict__ sik,
    float* __restrict__ oint,
    int head_dim,
    int qk_heads,
    int heads_per_kv,
    int chunk_start,
    int c
) {
    extern __shared__ float staged[];
    float* sk = staged;
    float* sq = staged + head_dim;

    int t = blockIdx.x;
    int h = blockIdx.y;
    int vi = threadIdx.x;
    int hq = h / heads_per_kv;

    long long base_t = ((long long)(chunk_start + t) * qk_heads + hq) * head_dim;
    sk[vi] = k_norm[base_t + vi];
    sq[vi] = q_norm[base_t + vi];
    __syncthreads();

    const float* row = state + ((long long)h * head_dim + vi) * head_dim;
    float acc_k = 0.0f;
    float acc_q = 0.0f;
    for (int j = 0; j < head_dim; ++j) {
        float s = row[j];
        acc_k += s * sk[j];
        acc_q += s * sq[j];
    }

    long long o = ((long long)h * c + t) * head_dim + vi;
    sik[o] = acc_k;
    oint[o] = acc_q;
}

// The chunk: solve for U, emit every token's output, advance the state.
//
// grid: (value_heads,). block: head_dim threads, one per value index vi.
// shared: U[c][head_dim] followed by lambda[c].
//
// The order of operations is load-bearing in exactly the way the recurrent
// form's is, restated for a whole chunk at once:
//
//   1. lambda_t is the cumulative decay through and including token t, and v_t
//      is divided by it before the correction is formed. That division is the
//      chunked form's "decay the state first": it moves the value into the
//      undecayed frame the substitution solves in, and dropping it leaves a
//      finite, fluent, different model.
//   2. The substitution sums over i < t, strictly. Token t's correction is
//      formed against the state as it stood *before* token t updated it.
//   3. The output sums over i <= t, inclusively. Token t's output is read from
//      the state *after* token t's own update. Making the two triangles agree
//      — the obvious "simplification" — is precisely the recurrent kernel's
//      read-before-update bug, transposed.
__global__ void gdn_chunk_solve_and_apply(
    float* __restrict__ state,
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
    int heads_per_kv,
    int chunk_start,
    int c
) {
    extern __shared__ float shared[];
    float* u = shared;                       // [c][head_dim]
    float* lambda = shared + (long long)c * head_dim;  // [c]

    int h = blockIdx.x;
    int vi = threadIdx.x;
    int hq = h / heads_per_kv;

    // Cumulative in-chunk decay. One thread, accumulated sequentially in the
    // reference's exact order: a parallel prefix scan would sum the log-decays
    // in a different association and give a different lambda for the same
    // input, which is a gratuitous divergence on a quantity that multiplies
    // every output in the chunk.
    if (vi == 0) {
        float running = 0.0f;
        for (int t = 0; t < c; ++t) {
            running += log_decay[(long long)(chunk_start + t) * value_heads + h];
            lambda[t] = expf(running);
        }
    }
    __syncthreads();

    // --- 1/2. Forward substitution for U, one row per iteration. -----------
    //
    // u_t = W_t - beta_t * sum_{i<t} (k_i . k_t) u_i,
    // W_t = beta_t * (v_t / lambda_t - S_in k_t).
    //
    // Thread vi owns column vi of U throughout, so the inner reduction is a
    // per-thread sequential loop over i with no cross-thread communication —
    // only the __syncthreads() that publishes row t before row t+1 reads it.
    // u[i * head_dim + vi] is bank-conflict free: consecutive vi land in
    // consecutive banks.
    for (int t = 0; t < c; ++t) {
        long long ht = (long long)(chunk_start + t) * value_heads + h;
        float beta_t = beta[ht];
        float v_t = v[ht * head_dim + vi];

        float acc = beta_t * (v_t / lambda[t] - sik[((long long)h * c + t) * head_dim + vi]);

        const float* kk_row = kk + ((long long)hq * c + t) * c;
        for (int i = 0; i < t; ++i) {
            acc -= beta_t * kk_row[i] * u[i * head_dim + vi];
        }

        u[t * head_dim + vi] = acc;
        __syncthreads();
    }

    // --- 3. Per-token output, from the state *including* token t's update. --
    for (int t = 0; t < c; ++t) {
        const float* kq_row = kq + ((long long)hq * c + t) * c;
        float o_intra = 0.0f;
        for (int i = 0; i <= t; ++i) {
            o_intra += kq_row[i] * u[i * head_dim + vi];
        }
        float o_inter = oint[((long long)h * c + t) * head_dim + vi];
        long long ht = (long long)(chunk_start + t) * value_heads + h;
        out[ht * head_dim + vi] = lambda[t] * (o_inter + o_intra);
    }

    // --- 4. Chunk-end state: S = lambda_{c-1} * (S_in + sum_i u_i k_i^T). ---
    //
    // Thread vi owns row vi of the state, so the read-modify-write is private
    // to the thread and needs no synchronisation. Every thread reads the same
    // k_norm element at the same time, which the hardware broadcasts.
    float lambda_last = lambda[c - 1];
    float* row = state + ((long long)h * head_dim + vi) * head_dim;
    for (int j = 0; j < head_dim; ++j) {
        float acc = 0.0f;
        for (int i = 0; i < c; ++i) {
            long long ki = ((long long)(chunk_start + i) * qk_heads + hq) * head_dim + j;
            acc += u[i * head_dim + vi] * k_norm[ki];
        }
        row[j] = lambda_last * (row[j] + acc);
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
        let needed = shared_bytes_for_solve(chunk_len, head_dim);
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
            head_dim,
            value_heads,
            qk_heads,
            chunk_len,
        })
    }

    /// Value heads sharing each query/key head.
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
            max_seq_len,
        })
    }

    /// Run the chunked form over `seq_len` tokens, advancing `state`.
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
        let heads_per_kv = self.heads_per_kv() as i32;

        // --- normalize the whole sequence once ---------------------------
        // One float per warp, which is all `block_reduce_sum` stores.
        let reduce_shared = (self.head_dim.div_ceil(32) * size_of::<f32>()) as u32;
        let norm_cfg = LaunchConfig {
            grid_dim: ((seq_len * self.qk_heads) as u32, 1, 1),
            block_dim: (self.head_dim as u32, 1, 1),
            shared_mem_bytes: reduce_shared,
        };
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

        // --- chunks, in order: they are dependent through the state -------
        let staged_shared = (2 * self.head_dim * size_of::<f32>()) as u32;
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
                grid_dim: (c as u32, self.value_heads as u32, 1),
                block_dim: (self.head_dim as u32, 1, 1),
                shared_mem_bytes: staged_shared,
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
                .arg(&heads_per_kv)
                .arg(&chunk_start)
                .arg(&c_i32);
            // SAFETY: the grid is (c, value_heads) and the block is head_dim
            // threads, so the state index `(h * head_dim + vi) * head_dim + j`
            // stays inside `value_heads * head_dim * head_dim` and the output
            // index `(h * c + t) * head_dim + vi` inside
            // `value_heads * chunk_len * head_dim`.
            unsafe { builder.launch(inter_cfg) }?;

            let solve_cfg = LaunchConfig {
                grid_dim: (self.value_heads as u32, 1, 1),
                block_dim: (self.head_dim as u32, 1, 1),
                shared_mem_bytes: shared_bytes_for_solve(c, self.head_dim) as u32,
            };
            let mut builder = stream.launch_builder(&self.solve);
            builder
                .arg(&mut *state)
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
                .arg(&heads_per_kv)
                .arg(&chunk_start)
                .arg(&c_i32);
            // SAFETY: one block per value head, head_dim threads per block, and
            // shared memory sized for exactly the `c * head_dim` entries of U
            // plus the `c` cumulative decays the kernel indexes. Every global
            // index is bounded by the same reasoning as the two kernels above,
            // and `c >= 1` inside the loop so `lambda[c - 1]` is in range.
            unsafe { builder.launch(solve_cfg) }?;

            start += c;
        }
        Ok(())
    }
}

/// Shared memory the fused solve kernel needs for a chunk of `c` tokens.
///
/// `U` is `c x head_dim` and the cumulative decay is `c` more floats. Nothing
/// else is resident — that is the whole reason the triangular system is solved
/// by substitution rather than by materialising the `c x c` inverse.
fn shared_bytes_for_solve(c: usize, head_dim: usize) -> usize {
    (c * head_dim + c) * size_of::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(GDN_CHUNKED_SRC.contains("for (int i = 0; i < t; ++i) {\n            acc -= beta_t * kk_row[i] * u[i * head_dim + vi];"));
        assert!(GDN_CHUNKED_SRC.contains(
            "for (int i = 0; i <= t; ++i) {\n            o_intra += kq_row[i] * u[i * head_dim + vi];"
        ));
    }

    #[test]
    fn the_value_is_moved_into_the_undecayed_frame_before_the_correction() {
        // v_t / lambda_t is the chunked form's "decay the state first". The
        // substitution solves in a frame where the decay has been divided out,
        // so a v_t that skipped the division would be measured against a state
        // it does not belong to. Dropping it leaves every output finite and
        // every norm plausible.
        let rescale = at("float acc = beta_t * (v_t / lambda[t] - sik[");
        let subtract = at("acc -= beta_t * kk_row[i] * u[i * head_dim + vi];");
        assert!(
            rescale < subtract,
            "the interference sum is subtracted before v is rescaled by 1/lambda",
        );
    }

    #[test]
    fn the_output_and_the_state_are_rescaled_by_the_right_cumulative_decay() {
        // Per-token output carries lambda_t; the chunk-end state carries
        // lambda_{c-1}. Swapping them is dimensionally invisible and silently
        // changes every layer's output after the first chunk.
        assert!(
            GDN_CHUNKED_SRC.contains("out[ht * head_dim + vi] = lambda[t] * (o_inter + o_intra);")
        );
        assert!(GDN_CHUNKED_SRC.contains("float lambda_last = lambda[c - 1];"));
        assert!(GDN_CHUNKED_SRC.contains("row[j] = lambda_last * (row[j] + acc);"));
    }

    #[test]
    fn the_state_update_consumes_the_solved_corrections_not_the_raw_residuals() {
        // S_out = lambda_{c-1} (S_in + sum_i u_i k_i^T) uses U, the solution of
        // the triangular system — not W, its right-hand side. Using W would be
        // the ungated delta rule with the intra-chunk key interference dropped,
        // which is a plausible-looking kernel that diverges from the recurrent
        // form only as keys within a chunk start to correlate.
        let publish = at("u[t * head_dim + vi] = acc;");
        let update = at("acc += u[i * head_dim + vi] * k_norm[ki];");
        assert!(publish < update, "the state is updated before U is solved");
    }

    #[test]
    fn the_scale_is_folded_into_q_only() {
        // Same invariant as the recurrent kernel's: on both, or on neither,
        // and every GDN layer's output is off by 11.3x or its square.
        assert!(
            GDN_CHUNKED_SRC.contains("q_out[base + j] = qv * (1.0f / sqrtf(q_sq + eps)) * scale;")
        );
        assert!(GDN_CHUNKED_SRC.contains("k_out[base + j] = kv * (1.0f / sqrtf(k_sq + eps));"));
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
        assert!(GDN_CHUNKED_SRC.contains("lambda[t] = expf(running);"));
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
        // head_dim 128, chunk_len 64: 33 KiB, comfortably inside the 48 KiB a
        // block gets without the opt-in carve-out.
        let needed = shared_bytes_for_solve(64, 128);
        assert_eq!(needed, (64 * 128 + 64) * 4);
        assert!(needed < MAX_SHARED_BYTES, "{needed} bytes");
        // Holding a 64x64 inverse as well would not fit, which is the shared
        // memory half of the substitution argument.
        assert!(needed + 64 * 64 * 4 > MAX_SHARED_BYTES);
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
