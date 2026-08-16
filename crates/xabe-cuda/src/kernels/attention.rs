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
//! **Chosen shape: `BM = 1`, K and V not staged at all.**
//!
//! ```text
//!   q_sh     head_dim floats            = 1024 B
//!   score_sh head_dim/32 floats (8)     =   32 B
//!   w_sh     head_dim/32 floats (8)     =   32 B
//!                                        -------
//!                                         1088 B  =  1.06 KiB  (2.2% of 48 KiB)
//! ```
//!
//! One block owns one (query row, query head). Its 256 threads are 8 warps;
//! each warp computes one key's `q . k` per iteration with a shuffle
//! reduction, so a tile is 8 keys. Each thread then owns one output dimension
//! of the running accumulator. K and V are read straight from global memory,
//! coalesced, exactly once each per block.
//!
//! Staging K in shared memory would buy nothing at `BM = 1`: a block touches
//! each K row once, so there is no intra-block reuse to amortize the staging
//! against. The reuse is *across* blocks — 16 query heads share 2 KV heads,
//! and neighbouring query rows share nearly the whole window — and that is
//! served by the 6 MiB L2, not by shared memory.
//!
//! **Occupancy consequence, stated honestly.** Shared memory is not the
//! limiter at 1.06 KiB per block; the limiter is Turing's 1024 threads per SM,
//! giving 4 blocks of 256 threads per SM. That is full thread occupancy but
//! low arithmetic intensity: this shape reads ~2 KiB of K+V per key per query
//! row and does ~1024 FLOP with it, about 0.5 FLOP/byte before L2. The kernel
//! is therefore bandwidth-bound rather than compute-bound, and its ceiling is
//! set by how much of the KV window L2 can hold across a wave of blocks — not
//! by FMA throughput.
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
__global__ void attn_rope_partial_neox(
    const float* __restrict__ in,
    float* __restrict__ out,
    int n_heads,
    int head_dim,
    int rope_dim,
    const int* __restrict__ pos_offset,
    float theta_base
) {
    long long t = blockIdx.x;
    int h = blockIdx.y;
    int d = threadIdx.x;
    long long base = (t * (long long)n_heads + h) * (long long)head_dim;

    int half = rope_dim >> 1;

    if (d >= rope_dim) {
        out[base + d] = in[base + d];
        return;
    }
    // Dimensions [half, rope_dim) are written by their partner thread d-half.
    if (d >= half) return;

    double pos = (double)(*pos_offset) + (double)t;
    double freq = pow((double)theta_base, -2.0 * (double)d / (double)rope_dim);
    double angle = pos * freq;
    float sin_a = (float)sin(angle);
    float cos_a = (float)cos(angle);

    float x0 = in[base + d];
    float x1 = in[base + d + half];
    out[base + d]        = x0 * cos_a - x1 * sin_a;
    out[base + d + half] = x0 * sin_a + x1 * cos_a;
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
// Keys one warp scores per trip. See the loop below.
#define ATTN_KT 4

__global__ void attn_flash_causal(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int q_heads,
    int kv_heads,
    int head_dim,
    const int* __restrict__ key_offset,
    float scale
) {
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

/// Compiled Gated Attention kernels for one head geometry.
pub struct AttentionKernels {
    split: CudaFunction,
    rope: CudaFunction,
    flash: CudaFunction,
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
        if head_dim == 0 || !head_dim.is_multiple_of(32) || head_dim > 1024 {
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

    /// Dynamic shared memory one block requests: `q_sh` plus the two
    /// tile-wide scratch arrays. See the module docs for the budget.
    pub fn shared_bytes(&self) -> usize {
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
            grid_dim: (n_tokens as u32, n_heads as u32, 1),
            block_dim: (self.head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_heads_i = n_heads as i32;
        let head_dim = self.head_dim as i32;
        let rope_dim_i = rope_dim as i32;
        let mut builder = stream.launch_builder(&self.rope);
        builder
            .arg(input)
            .arg(out)
            .arg(&n_heads_i)
            .arg(&head_dim)
            .arg(&rope_dim_i)
            .arg(positions)
            .arg(&theta_base);
        // SAFETY: the grid is (n_tokens, n_heads) with one thread per head
        // dimension; both buffers were checked to hold exactly that many
        // floats, and every thread touches only its own head's slice.
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
    pub fn forward(
        &self,
        stream: &Arc<CudaStream>,
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

        let cfg = LaunchConfig {
            grid_dim: (n_query as u32, self.q_heads as u32, 1),
            block_dim: (self.head_dim as u32, 1, 1),
            shared_mem_bytes: self.shared_bytes() as u32,
        };
        let q_heads = self.q_heads as i32;
        let kv_heads = self.kv_heads as i32;
        let head_dim = self.head_dim as i32;
        let scale = self.scale();
        let mut builder = stream.launch_builder(&self.flash);
        builder
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(out)
            .arg(&q_heads)
            .arg(&kv_heads)
            .arg(&head_dim)
            .arg(positions)
            .arg(&scale);
        // SAFETY: the grid is (n_query, q_heads) with one thread per head
        // dimension. The deepest key index any block reads is
        // `positions[0] + n_query - 1`, which the caller promised is below
        // `max_keys`, and all four buffers were checked against it.
        // Shared memory covers q_sh plus the two `head_dim/32`-wide scratch
        // arrays, which is everything the kernel indexes.
        unsafe { builder.launch(cfg) }?;
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
        assert!(
            ATTENTION_SRC.contains("long long n_visible = (long long)(*key_offset) + qi + 1;"),
            "the causal window bound changed shape",
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
            ATTENTION_SRC.contains("if (d >= rope_dim) {\n        out[base + d] = in[base + d];"),
            "the untouched rotary tail is no longer a plain copy",
        );
    }

    #[test]
    fn the_accumulator_is_rescaled_before_the_new_contributions_land() {
        // Online softmax is only stable if the running accumulator and the
        // normalizer are moved into the new max's frame *first*. Folding the
        // new weights in before rescaling produces a finite, plausible, wrong
        // answer whenever the max increases.
        let rescale_l = ATTENTION_SRC
            .find("l = l * corr + lsum;")
            .expect("normalizer rescale present");
        let rescale_acc = ATTENTION_SRC
            .find("float a = acc * corr;")
            .expect("accumulator rescale present");
        let fold_in = ATTENTION_SRC
            .find("a += w_sh[w] * v[")
            .expect("value accumulation present");
        assert!(rescale_acc < fold_in, "values folded in before rescaling");
        assert!(rescale_l < fold_in, "normalizer updated after the values");
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
        let keys_per_tile = head_dim / 32;
        let bytes = (head_dim + 2 * keys_per_tile) * size_of::<f32>();
        assert_eq!(bytes, 1088);
        assert!(bytes < TURING_SHARED_PER_BLOCK / 40);

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
}
