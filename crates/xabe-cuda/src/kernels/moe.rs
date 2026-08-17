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
//! 2. `moe_align_count` + `moe_align_block_size` — the sorted-token
//!    indirection, ported from `xabe_kernels::moe::dispatch`. Built into
//!    **fixed-size** buffers allocated once, because `AGENTS.md` rule 5
//!    forbids anything sized by a host-side value on this path: a host-sized
//!    allocation cannot be inside a captured CUDA graph, and graph capture
//!    over this indirection is the project's single largest expected win.
//!    Two launches, one block per expert each, because a 256-bin histogram
//!    with no atomics is a scan per bin and running all 256 of them in one
//!    block used one SM of 72.
//! 3. `moe_expert_ffn` / `moe_expert_down` — the grouped GEMM proper, with
//!    **Q6_K / Q8_0 dequantization in the prologue**. Materializing all 256
//!    experts in fp32 would turn one layer's 630 MiB of quantized expert
//!    weights into 3.1 GiB, times 41 blocks; the tile being multiplied is
//!    dequantized instead, and nothing is written back in fp32.
//! 4. `moe_reduce` — the fp32 weighted sum of each token's 8 routed
//!    contributions.
//!
//! ## How the grouped GEMM is tiled, and why each choice
//!
//! One block owns one **dispatch block** — a `block_size` run of slots that
//! by construction belongs to one expert — and a band of `TILE_ROWS` output
//! rows, one warp each. The slots' activations are staged in shared memory,
//! and every dequantized weight element is multiplied into all of the tile's
//! accumulators before being dropped. That is the whole point: an expert's
//! stack is read once per *tile of tokens*, not once per token.
//!
//! The first version gave each `(output row, slot)` pair its own block and so
//! re-read the whole stack for every token routed to an expert. Measured on
//! this card at Qwen3.6's geometry, `grouped_forward` ms per layer:
//!
//! ```text
//!                             1 tok   19 tok   128 tok   512 tok
//!   one block per (row,slot)   0.774   10.48     36.55    110.93
//!   tiled (this module)        0.174    2.00      5.37     11.92
//! ```
//!
//! Four things had to be true for the tiled form to be worth it, and only
//! three of them turned out to matter — see the comments at each site:
//!
//! - a warp owns a whole output row, so the dot product reduces in shuffles
//!   and never crosses a warp boundary (no block-wide reduction, no barrier
//!   in the epilogue);
//! - each lane takes four *consecutive* contraction elements, which makes the
//!   Q6_K superblock header a once-per-tile read and the staged activations a
//!   single `LDS.128`;
//! - the staging is double-buffered by hand through registers, because
//!   Turing has no `cp.async` and the barrier otherwise exposed a full global
//!   round trip on every pass;
//! - the tile height is specialized to the number of *live* rows, because a
//!   decode step puts one token in a 16-slot block.
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

use std::borrow::Cow;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};

use super::compile;
use super::mma::MmaKernels;

/// Elements per Q8_0 block, and per k-quant superblock.
///
/// Duplicated from [`super::dequant`] rather than imported so that a change
/// there cannot silently alter this module's alignment validation.
const QK8_0: usize = 32;
const QK_K: usize = 256;
const BLOCK_Q8_0_BYTES: usize = 34;
const BLOCK_Q6_K_BYTES: usize = 210;

/// Bytes per Q6_K superblock once it is on the device.
///
/// 224 rather than the file's 210 so that every superblock base, and with it
/// every field inside one, is 16-byte aligned. See [`ExpertQuant::block_bytes`].
pub const BLOCK_Q6_K_DEVICE_BYTES: usize = 224;

/// Threads per block for the routing, dispatch and reduction kernels.
///
/// A power of two, because their reductions are plain shared-memory tree
/// reductions that halve the active range each round.
const THREADS: u32 = 256;

/// Token slots one weight read is amortized over in the grouped GEMM.
///
/// This is the whole reason the tiled form is faster: a weight element is
/// dequantized once and multiplied into `TILE_M` tokens' accumulators, so an
/// expert's stack is read once per *tile of tokens* instead of once per
/// token. At Qwen3.6's `block_size` of 16 one tile is exactly one dispatch
/// block, which is why 16 and not 8 or 32 — a tile must never straddle two
/// experts, and a tile smaller than `block_size` gives up reuse for nothing.
const TILE_M: usize = 16;

/// Contraction elements staged into shared memory per pass.
///
/// 128 is forced by the Q6_K prologue, not chosen for occupancy: 128
/// consecutive elements are exactly one *half* of a 256-element superblock,
/// so the fp16 delta, the half's scale pointer and the `ql`/`qh` byte for a
/// lane are read **once** for the four elements that lane unpacks. A tile
/// that straddled a superblock boundary would have to re-read the header.
const TILE_K: usize = 128;

/// Output rows a grouped-GEMM block computes, one warp each.
///
/// Each warp owns a whole output row and reduces its dot product with warp
/// shuffles, so nothing crosses a warp boundary and the only `__syncthreads`
/// is the activation staging. Eight warps is the largest that keeps the
/// `TILE_M` gate and up accumulators in registers at 256 threads.
const TILE_ROWS: u32 = 8;

/// Threads per grouped-GEMM block.
const GEMM_THREADS: u32 = TILE_ROWS * 32;

/// Experts one lane of `moe_route`'s selection warp holds in registers.
///
/// Spelled here as well as in the kernel source, because NVRTC compiles from
/// a string with no access to Rust constants;
/// `the_routing_warp_bound_matches_the_kernel` asserts the two agree.
const ROUTE_LANE_EXPERTS: usize = 16;

/// Warps splitting the contraction in the one-token shared expert.
///
/// Mirrors `MOE_SHARED_WARPS`. Four rather than eight: eight leaves each warp
/// two 128-element tiles, which is short enough that the block spends more
/// time being scheduled than reading.
const SHARED_WARPS: u32 = 4;

/// Warps per block on the integer tensor-core path. Mirrors `MOE_MMA_WARPS`.
const MMA_WARPS: u32 = 8;
/// Output rows one warp owns — one `m8n8k16` N fragment. Mirrors `MOE_MMA_N`.
const MMA_N: u32 = 8;
/// Output rows one block covers. Mirrors `MOE_MMA_ROWS`.
const MMA_ROWS: u32 = MMA_WARPS * MMA_N;
/// Dispatch slots one block stages. Mirrors `MOE_MMA_M`.
///
/// A `block_size` above this would have its tail silently dropped, so
/// [`MoeKernels::new`] rejects such a geometry rather than trusting callers.
const MMA_M: usize = 32;
/// Token count below which the fp32 grouped GEMM is faster.
///
/// Measured per layer at Qwen3.6's geometry, `grouped_forward` end to end
/// including the Q8_0 down projection both paths share:
///
/// | tokens | fp32      | int8 tensor cores |
/// |--------|-----------|-------------------|
/// | 1      | 0.175 ms  | 0.195 ms          |
/// | 19     | 1.962 ms  | 1.718 ms          |
/// | 128    | 5.357 ms  | 4.333 ms          |
/// | 512    | 11.869 ms | 8.544 ms          |
///
/// The crossover is somewhere in `(1, 19]`. Eight is chosen as the height of
/// one M fragment rather than as a measured point: below a full fragment the
/// tile is more than half padding, and the fixed cost — an activation
/// quantization sweep plus a shared staging pass per k-chunk — has nothing to
/// amortize against. What this exists to prevent is the measured case: a
/// decode step of one token, where the integer path loses.
const MMA_MIN_TOKENS: usize = 8;

/// Bytes per staged activation row. Mirrors `MOE_MMA_ASTRIDE`.
///
/// 16 more than the contraction it holds: a 128-byte stride is 32 shared
/// banks, which makes every fragment load an eight-way conflict. See the
/// kernel.
const MMA_ASTRIDE: usize = MMA_KC + 16;

/// Contraction staged per trip. Mirrors `MOE_MMA_KC`.
const MMA_KC: usize = 128;
/// Bytes per staged weight row. Mirrors `MOE_MMA_WSTRIDE` — see the kernel on
/// why it is 112 and not the 96 the payload needs.
const MMA_WSTRIDE: usize = 112;
/// Bytes per staged scale row. Mirrors `MOE_MMA_SSTRIDE`.
const MMA_SSTRIDE: usize = 16;

/// Bytes per staged weight row in the down projection. Mirrors
/// `MOE_MMA_DSTRIDE`: the quants, then one fp32 scale per 32 of them.
const MMA_DSTRIDE: usize = MMA_KC + (MMA_KC / 32) * 4;

/// Shared bytes the down projection needs: one staged weight tile, the int8
/// activation tile, its per-32 scales, and one row index plus one slot id per
/// staged slot.
const fn mma_down_shared_bytes() -> u32 {
    (MMA_ROWS as usize * MMA_DSTRIDE
        + MMA_M * MMA_ASTRIDE
        + MMA_M * (MMA_KC / 32) * size_of::<f32>()
        + MMA_M * size_of::<i64>()
        + MMA_M * size_of::<i32>()) as u32
}

/// Shared bytes the Q8_0 gate/up tensor-core GEMM needs: two staged weight
/// tiles in the `MMA_DSTRIDE` layout (quants then fp32 block scales), the int8
/// activation tile, its per-32 scales, and one activation row index per slot.
const fn mma_ffn_q8_shared_bytes() -> u32 {
    (2 * MMA_ROWS as usize * MMA_DSTRIDE
        + MMA_M * MMA_ASTRIDE
        + MMA_M * (MMA_KC / 32) * size_of::<f32>()
        + MMA_M * size_of::<i64>()) as u32
}

/// Shared bytes the tensor-core grouped GEMM needs: two staged weight tiles,
/// their scale tiles, the int8 activation tile, its per-32 scales, and one
/// activation row index per staged slot.
const fn mma_shared_bytes() -> u32 {
    (2 * MMA_ROWS as usize * MMA_WSTRIDE
        + 2 * MMA_ROWS as usize * MMA_SSTRIDE
        + MMA_M * MMA_ASTRIDE
        + MMA_M * (MMA_KC / 32) * size_of::<f32>()
        + MMA_M * size_of::<i64>()) as u32
}

/// Storage format of one expert weight stack.
///
/// The real `Qwen3.6-35B-A3B-UD-Q6_K_XL` file is **mixed**: `ffn_gate_exps`
/// and `ffn_up_exps` are Q6_K, but `ffn_down_exps` is Q8_0 (verified against
/// the file's tensor directory, not assumed). A kernel that hard-coded Q6_K
/// would read the down projection as garbage, so the format travels with the
/// pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertQuant {
    /// 256 elements per superblock: 210 bytes in the file, 224 on the device.
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

    /// Bytes per block **as the file stores them**.
    ///
    /// What a GGUF tensor's length must be a multiple of. Not what the device
    /// copy is strided by — see [`Self::block_bytes`].
    pub const fn file_block_bytes(self) -> usize {
        match self {
            Self::Q6K => BLOCK_Q6_K_BYTES,
            Self::Q8_0 => BLOCK_Q8_0_BYTES,
        }
    }

    /// Bytes per block **as the device holds them**.
    ///
    /// Q6_K is padded from 210 to 224 on the way in. 210 is even and nothing
    /// else: `210 * s` is 4-byte aligned only for even `s`, so every load from
    /// a superblock has to be 16 bits wide, and the grouped GEMM's staging
    /// loop -- which is a third of the kernel that is a fifth of prefill --
    /// spends four instructions per weight row moving 106 bytes. 224 is a
    /// multiple of 16, so every field of every superblock is 16-byte aligned
    /// and one `int4` load stages eight rows at a time.
    ///
    /// The 14 pad bytes are never read; they cost 6.7% more sectors and 1.2
    /// GiB of the 47 GiB card. See [`to_device_layout`].
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::Q6K => BLOCK_Q6_K_DEVICE_BYTES,
            Self::Q8_0 => BLOCK_Q8_0_BYTES,
        }
    }
}

/// Rewrite a GGUF expert stack into the layout the device kernels index.
///
/// Q6_K superblocks are re-strided from 210 bytes to
/// [`BLOCK_Q6_K_DEVICE_BYTES`]; the 210 payload bytes are copied unchanged and
/// the rest is left zero. Q8_0 is returned as-is. Every upload of an
/// [`ExpertQuant`] tensor must go through this, because the kernels address
/// superblocks by [`ExpertQuant::block_bytes`].
pub fn to_device_layout(quant: ExpertQuant, bytes: &[u8]) -> Cow<'_, [u8]> {
    let src_stride = quant.file_block_bytes();
    let dst_stride = quant.block_bytes();
    if src_stride == dst_stride {
        // Q8_0 is already the layout the kernels index; borrow it rather than
        // copying 285 MiB per down projection for nothing.
        return Cow::Borrowed(bytes);
    }
    let blocks = bytes.len() / src_stride;
    let mut out = vec![0u8; blocks * dst_stride];

    // Threaded, because this runs over 17.6 GiB of expert weights at load and
    // single-threaded it added 28 s to building a shape. The output is split
    // into disjoint block ranges, so no two threads touch the same byte.
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(blocks.max(1));
    let per = blocks.div_ceil(threads);
    std::thread::scope(|scope| {
        for (t, dst) in out.chunks_mut(per * dst_stride).enumerate() {
            let src = &bytes[t * per * src_stride..];
            scope.spawn(move || {
                for (b, chunk) in src
                    .chunks_exact(src_stride)
                    .take(dst.len() / dst_stride)
                    .enumerate()
                {
                    dst[b * dst_stride..b * dst_stride + src_stride].copy_from_slice(chunk);
                }
            });
        }
    });
    Cow::Owned(out)
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
// Tile shape. Mirrored by `TILE_M` / `TILE_K` / `TILE_ROWS` on the Rust side,
// which size the shared memory and the grid; `tile_shape_is_mirrored_in_rust`
// asserts the two never drift.
// Experts one lane of the routing warp can hold in registers. 16 covers 512
// experts; the launch path rejects anything wider.
#define Q6K_SB 224
#define MOE_SHARED_WARPS 4
#define MOE_ROUTE_LANE_EXPERTS 16
#define MOE_TM   16
#define MOE_TK   128
#define MOE_TN   4
#define MOE_ROWS 8
// Contraction passes a decode GEMV issues before consuming any of them.
//
// These loops are bounded by `hidden` or `intermediate`, both runtime values,
// so ptxas cannot unroll them on its own: the next pass's dequantization
// cannot issue until this pass's arithmetic has freed its registers, and each
// thread keeps one weight tile in flight. That is a memory-parallelism bound
// rather than a bandwidth one, and it is why these kernels sat at a third of
// the streaming roofline while the LM head's own GEMV reached 89%.
//
// The gate/up kernels keep their own literal 2 -- they dequantize two matrices
// a pass, so they already had two independent streams before any unrolling,
// and four measured worse there. The down projections have one stream and need
// this.
#define MOE_DOWN_UNROLL 4

// Derived: activation floats each thread stages per pass, and the tile-row
// stride between the rows one thread owns. Both are compile-time so the
// staging loops unroll and hold their prefetch in registers.
#define MOE_MSTEP ((MOE_ROWS * 32) / MOE_TK)
#define MOE_STAGE (MOE_TM / MOE_MSTEP)

// The device helpers below are C++ templates, so they cannot live inside the
// `extern "C"` block the kernels need — a template cannot have C linkage.
// The `__global__` entry points start further down.

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

// The same conversion for a half already in a register.
//
// The staging loop below fetches a Q6_K superblock's delta in the same warp
// instruction as its `qh` and its sub-scales -- one lane's two bytes of a
// twenty-one-lane load -- so by the time it is converted it is no longer
// addressable.
__device__ __forceinline__ float half_bits_to_float(unsigned short bits) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(bits));
    return f;
}

// The Q6_K reconstruction, written once so there is one place to get it
// wrong. The multiply order `(d * scale) * q` is load-bearing and matches
// the scalar reference operand for operand; reassociating to
// `d * (scale * q)` is mathematically equal, rounds differently, and would
// cost bit-identical weights.
__device__ __forceinline__ float q6k_value(float d, const signed char* sc, int si, int raw) {
    return d * (float)sc[si] * (float)(raw - 32);
}

// The Q8_0 reconstruction, same reasoning.
__device__ __forceinline__ float q8_0_value(signed char q, float d) {
    return (float)q * d;
}

// Unpack the MOE_TK consecutive Q6_K elements starting at `i0`. Warp lane L
// owns the **four consecutive** elements 4L .. 4L+3.
//
// Two things ride on "consecutive" rather than the stride-32 assignment the
// standalone dequant kernel uses:
//
// 1. **The superblock header is hoisted.** Addressing an element by its flat
//    index alone — the shape the first version of this kernel used — re-reads
//    the fp16 delta and re-derives the scale pointer for *every* element,
//    2,048 redundant header loads per weight row per token. `i0` is a
//    multiple of 128, so the tile lies inside one 128-element half of one
//    superblock: `half`, `sc` and `d` are read once, and because four
//    consecutive `l` values never cross a 16-element scale boundary, the
//    group index `si` is constant across all four too.
// 2. **The four activations the lane needs are then contiguous in shared
//    memory**, which is what lets the inner product load them as one
//    `LDS.128` instead of four `LDS.32`. Shared-load issue, not arithmetic,
//    was the measured limiter of the stride-32 form.
//
// The mapping is `kernels/dequant.rs`'s, read backwards: a flat index r
// within a superblock decomposes as half = r/128, group = (r%128)/32 and
// l = r%32. Over this tile half is fixed, group is `lane / 8`, and l runs
// 4*(lane%8) .. +3.
__device__ __forceinline__ void dequant_tile_q6k(
    const unsigned char* __restrict__ src, long long i0, int wlane, float* out
) {
    const unsigned char* base = src + (i0 >> 8) * Q6K_SB;
    int half = (int)((i0 >> 7) & 1);
    const unsigned char* ql = base + half * 64;
    const unsigned char* qh = base + 128 + half * 32;
    const signed char*   sc = (const signed char*)(base + 192 + half * 8);
    float d = load_half_le(base + 208);

    int grp = wlane >> 3;
    int l   = (wlane * 4) & 31;
    int si  = (l >> 4) + 2 * grp;
    // Groups 1 and 3 live in the second 32 bytes of the half's `ql`; groups
    // 2 and 3 take the high nibble rather than the low one.
    const unsigned char* qlp = ql + ((grp & 1) ? l + 32 : l);
    int shift = 2 * grp;

    // One 32-bit load each, which the *device* superblock stride is what makes
    // legal. The file's 210 bytes are even and nothing more, so `210 * s` is
    // 4-byte aligned only for even `s` and this had to be two 16-bit loads
    // assembled with a shift. `Q6K_SB` is 224, a multiple of 16, so every
    // superblock base is word-aligned and so is every field inside one --
    // `l` is a multiple of 4 and `half * 64` and `128 + half * 32` both are.
    //
    // This is the decode side of the same padding that lets the grouped GEMM
    // stage eight rows in one `int4`. It halves the load instructions of the
    // one-token expert GEMV's inner loop, which is 17% of a decode step.
    unsigned int qlw = *(const unsigned int*)(qlp);
    unsigned int qhw = *(const unsigned int*)(qh + l);

    // Unpack all four codes at word width rather than one byte at a time.
    //
    // The four elements a lane owns take the *same* field of four different
    // bytes: the same nibble of `ql`, the same bit pair of `qh`. So one mask
    // over the whole word does what four shift-and-mask pairs did, and the
    // per-element work drops to a shift, a mask and the subtraction.
    //
    // Both shifts are safe across byte lanes because the field never reaches
    // the top of its byte: `(qlw >> 4) & 0x0F0F0F0F` takes bits 4..7 of each
    // byte, and `shift` is 0, 2, 4 or 6 so `(qhw >> shift) & 0x03030303`
    // takes bits `shift..shift+1` — neither can pull a bit in from the byte
    // above.
    //
    // This is worth doing because the decode-shape kernels that call it are
    // bound by the **integer pipe**, not by DRAM. `moe_expert_ffn_gemv` moved
    // its weights at 43% of the card's streaming roofline while the LM head's
    // own GEMV reached 82% on the same card, and the difference is that Q6_K
    // costs about nine integer instructions per element to unpack where Q8_0
    // costs none. Fewer instructions per byte, identical bytes: `raw` is the
    // same integer, so every reconstructed weight is bit-for-bit what it was.
    unsigned int lo4 = (grp < 2) ? (qlw & 0x0F0F0F0Fu) : ((qlw >> 4) & 0x0F0F0F0Fu);
    unsigned int raw4 = lo4 | (((qhw >> shift) & 0x03030303u) << 4);

    #pragma unroll
    for (int t = 0; t < MOE_TN; ++t) {
        int raw = (int)((raw4 >> (8 * t)) & 0xFFu);
        out[t] = q6k_value(d, sc, si, raw);
    }
}

// The same tile in Q8_0. A Q8_0 block *is* 32 elements, so a lane's four
// consecutive elements always share one block and its fp16 delta is read
// once for all four.
//
// Reading the code as `signed char` is load-bearing: the quants are int8 on
// disk, and reading them unsigned flips the sign of roughly half of every
// tensor while leaving magnitudes plausible.
__device__ __forceinline__ void dequant_tile_q8_0(
    const unsigned char* __restrict__ src, long long i0, int wlane, float* out
) {
    const unsigned char* base = src + ((i0 >> 5) + (wlane >> 3)) * 34;
    float d = load_half_le(base);
    int first = (wlane * 4) & 31;

    #pragma unroll
    for (int t = 0; t < MOE_TN; ++t) {
        int lane = first + t;
        signed char q = (signed char)base[2 + lane];
        out[t] = q8_0_value(q, d);
    }
}

// The GEMM prologue: unpack exactly the weight tile about to be multiplied.
// Nothing is materialized in fp32 beyond the four values in registers.
__device__ __forceinline__ void dequant_tile(
    const unsigned char* __restrict__ src, int quant, long long i0, int wlane, float* out
) {
    if (quant == 0) dequant_tile_q6k(src, i0, wlane, out);
    else            dequant_tile_q8_0(src, i0, wlane, out);
}

// Stage MOE_TM rows x MOE_TK columns of activations into shared memory, in
// two halves so the global read of one pass overlaps the arithmetic of the
// previous one.
//
// **Turing has no `cp.async`, so the double buffer is by hand** — the buffer
// is the `pf` register array, not a second shared tile. `prefetch_tile`
// issues the loads for pass j+1 immediately after pass j's tile becomes
// visible; `commit_tile` writes them to shared at the top of pass j+1. With
// the two fused into one `stage` call the block stalled on a full global
// round trip at every one of the 16 barriers per row, which measured as the
// single largest remaining cost.
//
// `rows[m]` is the flat element offset of tile row m's source row, or
// negative for a row that contributes nothing — a padding slot, or a token
// past `valid_tokens`. Those rows are **zeroed rather than skipped**, which
// makes their contribution exactly zero and lets the inner product run an
// unconditional, fully unrolled MOE_TM-wide accumulate.
//
// The thread -> (row, column) map is deliberately rigid: a thread's column is
// `threadIdx.x % MOE_TK` for the whole kernel and only its row advances, so
// both halves are a compile-time-bounded unrolled loop over MOE_STAGE
// elements. An earlier version indexed the tile linearly and recovered the
// row and column with a division, which cost a *software integer division*
// (`flat / top_k` divides by a runtime value — twenty-odd instructions on
// sm_75) on every staged float. The row offsets are divided once per tile row
// instead, by the MOE_TM threads that fill `rows`.
//
// `TM` is the *live* tile height, which is not always MOE_TM — see
// `moe_rounded_rows`.
template<int TM>
__device__ __forceinline__ void prefetch_tile(
    const float* __restrict__ src, const long long* rows, int j0, float* pf
) {
    int k = threadIdx.x & (MOE_TK - 1);
    int m0 = threadIdx.x / MOE_TK;
    #pragma unroll
    for (int i = 0; i < TM / MOE_MSTEP; ++i) {
        long long row = rows[m0 + i * MOE_MSTEP];
        pf[i] = row < 0 ? 0.0f : src[row + j0 + k];
    }
}

template<int TM>
__device__ __forceinline__ void commit_tile(const float* pf, float* xs) {
    int k = threadIdx.x & (MOE_TK - 1);
    int m0 = threadIdx.x / MOE_TK;
    #pragma unroll
    for (int i = 0; i < TM / MOE_MSTEP; ++i) {
        xs[(m0 + i * MOE_MSTEP) * MOE_TK + k] = pf[i];
    }
}

// How many of a tile's MOE_TM rows carry a real token, and the compile-time
// tile height that covers them.
//
// A dispatch block holds `block_size` slots but only its leading `count %
// block_size` are real at the end of an expert's run, and at decode a whole
// block holds **one** token: the tile is 16 rows wide and 15 of them are the
// padding sentinel. Multiplying those rows anyway costs a 16x arithmetic
// overhead on exactly the shape where the MoE is 77% of the step. `rows[m]`
// is negative for a row that contributes nothing, so the live height is one
// past the last non-negative entry — computed rather than assumed to be a
// prefix, so a dispatch defect cannot silently drop tokens.
//
// The result is uniform across the block (every warp sees the same slots), so
// selecting a specialization on it never diverges.
__device__ __forceinline__ int live_tile_rows(const long long* rows) {
    int bm = 0;
    #pragma unroll
    for (int m = 0; m < MOE_TM; ++m) {
        if (rows[m] >= 0) bm = m + 1;
    }
    return bm;
}

// The four staged activations lane `wlane` needs from tile row `m`, as one
// 128-bit shared load. `xabe_shared` is 16-byte aligned and the row stride is
// MOE_TK floats, so both the base and the `4 * wlane` offset are multiples of
// 16 bytes; consecutive lanes then cover all 32 banks exactly once.
__device__ __forceinline__ float4 tile_row4(const float* xs, int m, int wlane) {
    return *(const float4*)(xs + m * MOE_TK + 4 * wlane);
}

// Shared-memory pool. One declaration, one type, for every kernel below:
// two `extern __shared__` arrays of different element types in the same
// translation unit is a redeclaration error, so integer users cast. The
// explicit alignment is what makes `tile_row4`'s 128-bit load legal.
extern __shared__ __align__(16) float xabe_shared[];

// Sum TM independent accumulators across a warp, leaving the totals in every
// lane. Replaces the old block-wide reduction: a warp now owns a whole output
// row, so nothing has to cross a warp boundary and the epilogue needs no
// `__syncthreads` at all.
template<int TM>
__device__ __forceinline__ void warp_reduce_tile(float* acc) {
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc[m] += __shfl_xor_sync(0xffffffff, acc[m], off);
        }
    }
}

// The gate/up half of one expert's FFN for a TM-row tile: contract two weight
// rows against the staged activations and leave the two dot products in every
// lane of the owning warp.
//
// The staging is double-buffered by hand — `prefetch_tile` issues pass j+1's
// global loads *before* pass j's arithmetic, so the round trip overlaps the
// multiply-adds instead of stalling the block at the barrier.
template<int TM>
__device__ __forceinline__ void tile_gemm_pair(
    const unsigned char* __restrict__ gate_q, int gate_quant,
    const unsigned char* __restrict__ up_q,   int up_quant,
    const float* __restrict__ src, const long long* rows, float* xs,
    long long wrow, int k_len, int lane, int live, float* ag, float* au
) {
    #pragma unroll
    for (int m = 0; m < TM; ++m) { ag[m] = 0.0f; au[m] = 0.0f; }

    float pf[TM / MOE_MSTEP];
    prefetch_tile<TM>(src, rows, 0, pf);

    for (int j0 = 0; j0 < k_len; j0 += MOE_TK) {
        __syncthreads();
        commit_tile<TM>(pf, xs);
        __syncthreads();
        if (j0 + MOE_TK < k_len) prefetch_tile<TM>(src, rows, j0 + MOE_TK, pf);
        if (live) {
            float wg[MOE_TN];
            float wu[MOE_TN];
            dequant_tile(gate_q, gate_quant, wrow + j0, lane, wg);
            dequant_tile(up_q,   up_quant,   wrow + j0, lane, wu);
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                float4 xv = tile_row4(xs, m, lane);
                ag[m] += wg[0] * xv.x;  au[m] += wu[0] * xv.x;
                ag[m] += wg[1] * xv.y;  au[m] += wu[1] * xv.y;
                ag[m] += wg[2] * xv.z;  au[m] += wu[2] * xv.z;
                ag[m] += wg[3] * xv.w;  au[m] += wu[3] * xv.w;
            }
        }
    }
    warp_reduce_tile<TM>(ag);
    warp_reduce_tile<TM>(au);
}

// As above for a single weight matrix: the down projection.
template<int TM>
__device__ __forceinline__ void tile_gemm_single(
    const unsigned char* __restrict__ w_q, int w_quant,
    const float* __restrict__ src, const long long* rows, float* xs,
    long long wrow, int k_len, int lane, int live, float* ad
) {
    #pragma unroll
    for (int m = 0; m < TM; ++m) ad[m] = 0.0f;

    float pf[TM / MOE_MSTEP];
    prefetch_tile<TM>(src, rows, 0, pf);

    for (int j0 = 0; j0 < k_len; j0 += MOE_TK) {
        __syncthreads();
        commit_tile<TM>(pf, xs);
        __syncthreads();
        if (j0 + MOE_TK < k_len) prefetch_tile<TM>(src, rows, j0 + MOE_TK, pf);
        if (live) {
            float wd[MOE_TN];
            dequant_tile(w_q, w_quant, wrow + j0, lane, wd);
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                float4 xv = tile_row4(xs, m, lane);
                ad[m] += wd[0] * xv.x;
                ad[m] += wd[1] * xv.y;
                ad[m] += wd[2] * xv.z;
                ad[m] += wd[3] * xv.w;
            }
        }
    }
    warp_reduce_tile<TM>(ad);
}

// Dispatch the two workhorses on the live tile height, rounded **up** to the
// nearest specialization.
//
// Rounding up is not a preference. Picking a `TM` below `bm` silently drops
// the rows past it, and the failure is invisible at small batches: an earlier
// try at `bm > 2 -> CALL(4)` was the fastest variant measured and was wrong
// for every tile with five to eight live rows. `tests/moe_differential.rs`
// caught it on token 36 of 37. Every threshold below is `bm > TM_lower`.
//
// **Three instantiations, and the count is measured rather than assumed.**
// Each one is a full copy of the inner loop and ptxas budgets registers for
// the widest, so a fourth costs occupancy on every path including the narrow
// ones. Measured at Qwen3.6's geometry, `grouped_forward` ms per layer:
//
// ```text
//   specializations   1 tok   19 tok   128 tok   512 tok
//   {16} only         0.405    4.53      8.68     13.29
//   {2,16}            0.172    2.03      7.59     12.70
//   {2,4,16}          0.172    1.87      6.05     12.04
//   {2,8,16}          0.174    2.00      5.37     11.92   <- this
//   {2,4,8,16}        0.415    4.72      8.68     13.29   <- register cliff
// ```
//
// The last row is not a typo: the fourth copy pushed `moe_expert_ffn` past
// the register count that fits three blocks on an sm_75 SM, and the occupancy
// loss ate the entire arithmetic saving — it is no better than no
// specialization at all.
#define MOE_TILE_DISPATCH(CALL)        \
    if      (bm > 8) { CALL(16); }     \
    else if (bm > 2) { CALL(8);  }     \
    else             { CALL(2);  }

// Write one slot's weighted contribution, or nothing at all if the slot is
// padding.
//
// The sentinel is exactly `valid_tokens * top_k`
// (`xabe_kernels::moe::dispatch::padding_sentinel`), so `flat >= numel` is
// the padding test and `flat` is a valid index into `topk_weights` whenever
// it passes.
__device__ __forceinline__ void store_slot_contribution(
    float* __restrict__ partial, const float* __restrict__ topk_weights,
    int flat, int numel, int hidden, int h, float s
) {
    if (flat >= numel) return;
    partial[(long long)flat * hidden + h] = topk_weights[flat] * s;
}

extern "C" {

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
// The routing itself, so the fused one-token kernel below can run it without
// a second copy of two hundred lines. Every barrier inside is reached by every
// thread of the block, which is what makes it safe to call from a
// `__device__` function.
__device__ void route_token(
    const float* __restrict__ logits,
    int token,
    int num_experts,
    int top_k,
    float* probs,
    float* rval,
    int* ridx,
    int* __restrict__ topk_ids,
    float* __restrict__ topk_weights
) {

    const float* row = logits + (long long)token * num_experts;

    // --- max logit (exact: max is associative and rounds nothing) ---------
    float local = row[0];
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        local = fmaxf(local, row[e]);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local = fmaxf(local, __shfl_down_sync(0xffffffff, local, off));
    }
    if ((threadIdx.x & 31) == 0) rval[threadIdx.x >> 5] = local;
    __syncthreads();
    float max_logit = rval[0];
    for (int w = 1; w < (int)(blockDim.x >> 5); ++w) max_logit = fmaxf(max_logit, rval[w]);
    __syncthreads();

    // --- exp, sum, normalize ---------------------------------------------
    float esum = 0.0f;
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        float p = expf(row[e] - max_logit);
        probs[e] = p;
        esum += p;
    }
    // **This one keeps its shared-memory tree.** The other two reductions in
    // this kernel were replaced with warp shuffles because `fmaxf` and
    // "greatest value, lowest index on a tie" are associative *and*
    // commutative, so their answers do not depend on the shape of the
    // reduction. A floating-point sum is neither, and this one divides every
    // probability. Reshaping it moved the renormalized routing weights in
    // their last bits, which was enough for `tests/forward_pass.rs` to report
    // a genuine ranking disagreement with llama.cpp at rank 4 of the top 8.
    //
    // Nine barriers, against the eighty-one the other two shed between them.
    rval[threadIdx.x] = esum;
    __syncthreads();
    // The same additions in the same order, with the last five steps under a
    // warp barrier instead of a block barrier: once `s <= 16` every reader and
    // every writer is lane 0..31 of warp 0, and `__syncwarp` fences shared
    // memory for them. Five `__syncthreads()` on a kernel that runs as one
    // block, forty times a decode step.
    for (int s = blockDim.x >> 1; s > 16; s >>= 1) {
        if (threadIdx.x < s) rval[threadIdx.x] += rval[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        #pragma unroll
        for (int s = 16; s > 0; s >>= 1) {
            if (threadIdx.x < s) rval[threadIdx.x] += rval[threadIdx.x + s];
            __syncwarp();
        }
    }
    __syncthreads();
    float sum_exp = rval[0];
    // `probs` stays *unnormalized* in shared memory. The only consumer is the
    // selection warp below, which divides in registers -- `probs[e] /
    // sum_exp` is the same float either way -- so the pass that used to write
    // the normalized array back, and the barrier after it, are gone.

    // --- k successive argmax passes, ties to the lower index --------------
    //
    // Equivalent to the reference's full sort-then-truncate, and cheaper:
    // k is 8 against 256 experts. A selected expert is masked with -1.0f,
    // which no softmax probability can reach, so it can never be re-picked.
    // **On one warp, and with no barriers at all.** The block-wide form cost
    // two `__syncthreads()` and a serial pass over the warp winners per
    // selected expert — sixteen barriers and eight serial passes for eight
    // experts, in a kernel that runs as a single block on a 72-SM card, so
    // the whole machine waits through them forty times a decode step.
    //
    // "Greatest value, lowest index on a tie" is associative *and*
    // commutative, so the answer does not depend on the shape of the
    // reduction the way a floating-point sum would — which is what makes it
    // safe to change the shape at all, on a quantity that selects experts.
    // That is the same argument the softmax denominator above cannot make,
    // which is why that one still reduces through shared memory.
    //
    // Each lane holds its experts' probabilities in registers, and the
    // butterfly shuffle leaves the winner in *every* lane, so the lane that
    // owns it masks it in place. The mask loop is fully unrolled with a
    // predicate rather than indexed by the winner, because a dynamic index
    // into a register array spills it to local memory and undoes the point.
    if (threadIdx.x < 32) {
        int lane = threadIdx.x;
        float p[MOE_ROUTE_LANE_EXPERTS];
        #pragma unroll
        for (int i = 0; i < MOE_ROUTE_LANE_EXPERTS; ++i) {
            int e = lane + 32 * i;
            p[i] = (e < num_experts) ? probs[e] / sum_exp : -1.0f;
        }
        for (int j = 0; j < top_k; ++j) {
            float bv = -1.0f;
            int   bi = -1;
            #pragma unroll
            for (int i = 0; i < MOE_ROUTE_LANE_EXPERTS; ++i) {
                int e = lane + 32 * i;
                if (p[i] > bv || (p[i] == bv && (bi < 0 || e < bi))) { bv = p[i]; bi = e; }
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                float cv = __shfl_xor_sync(0xffffffff, bv, off);
                int   ci = __shfl_xor_sync(0xffffffff, bi, off);
                if (cv > bv || (cv == bv && ci >= 0 && (bi < 0 || ci < bi))) { bv = cv; bi = ci; }
            }
            #pragma unroll
            for (int i = 0; i < MOE_ROUTE_LANE_EXPERTS; ++i) {
                if (bi >= 0 && lane + 32 * i == bi) p[i] = -1.0f;
            }
            if (lane == 0) {
                topk_ids[(long long)token * top_k + j] = bi;
                topk_weights[(long long)token * top_k + j] = bv;
            }
        }
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

// grid: (max_tokens,). block: MOE_THREADS.
__global__ void moe_route(
    const float* __restrict__ logits,
    const int* __restrict__ valid_tokens,
    int num_experts,
    int top_k,
    int* __restrict__ topk_ids,
    float* __restrict__ topk_weights
) {
    int token = blockIdx.x;
    if (token >= *valid_tokens) return;
    float* probs = xabe_shared;
    float* rval  = xabe_shared + num_experts;
    int*   ridx  = (int*)(rval + blockDim.x);
    route_token(logits, token, num_experts, top_k, probs, rval, ridx,
                topk_ids, topk_weights);
}


// -------------------------------------------------------------------------
// 2. Sorted-token indirection, into fixed-size buffers.
// -------------------------------------------------------------------------
//
// Two launches, both at grids fixed by `MoeGeometry`: **one block per
// expert**, THREADS threads. The first version ran the whole thing in a
// single block — one SM of 72 — and its own docs admitted it was "not tuned
// for large prefill batches". Both phases are `O(num_experts * numel)` in
// total work (a 256-bin histogram with no atomics is a scan per bin), so on
// one block that is 8,192 serial iterations at a 512-token step; spread one
// expert per block it is 32.
//
// Phase 1 counts and fills; phase 2 scans and scatters. The split is a
// launch rather than a grid-wide barrier because the exclusive prefix sum
// over per-expert padded counts is a genuine global dependency and Turing
// has no device-wide sync inside a kernel.
//
// `numel = valid_tokens * top_k` is simultaneously the number of valid flat
// indices and the padding sentinel (`padding_sentinel` in the reference is
// exactly `num_tokens * top_k`), so a consumer's test is `flat >= numel`.
//
// The sentinel and INACTIVE_EXPERT fills cover the **whole capacity**, not
// just `num_tokens_post_pad`. Filling only the live prefix would leave a
// previous, longer step's entries visible past it — tokens that no longer
// exist, pointing into a token array that has since shrunk. They ride on
// phase 1 because they depend on nothing phase 1 computes, and spreading
// them over `num_experts` blocks costs nothing.
//
// grid: (num_experts, 1, 1). Block `e` owns expert `e`.
// The whole dispatch table in one block, for a one-token pass.
//
// `moe_align_count` and `moe_align_block_size` are shaped for a batch: 256
// blocks, a block-wide tree reduction to count one expert's selections, a
// 256-element serial prefix sum recomputed in every block, and a Hillis-Steele
// scan per 256 flat pairs to place them. At one token there are **eight** flat
// pairs. Every block was spending sixteen barriers and a 256-add serial scan
// to discover that seven of its 256 threads had nothing to do, and the pair
// cost 0.37 ms of a 10.3 ms decode step between them -- more than the shared
// expert's entire feed-forward.
//
// With `numel` at eight, a thread can simply read all eight ids twice: once to
// count its expert's selections and once to place them in ascending flat
// order, which is the same order the scan produced. That makes the counts
// private to a thread, which removes the reduction, the intermediate
// `counts` array and the second launch along with it.
//
// The two barriers that remain are real: the fill must land before the
// scatter overwrites part of it, and the prefix sum must be complete before
// any thread reads its base.
//
// grid: (1,). block: MOE_THREADS.
__device__ void dispatch_token(
    const int* __restrict__ topk_ids,
    const int* __restrict__ valid_tokens,
    int top_k,
    int num_experts,
    int block_size,
    int sorted_capacity,
    int expert_capacity,
    int* cumsum,
    int* __restrict__ sorted_token_ids,
    int* __restrict__ expert_ids,
    int* __restrict__ num_tokens_post_pad
) {
    int tid = threadIdx.x;
    int numel = (*valid_tokens) * top_k;

    for (int s = tid; s < sorted_capacity; s += blockDim.x) sorted_token_ids[s] = numel;
    for (int b = tid; b < expert_capacity; b += blockDim.x) expert_ids[b] = -1;

    // An expert with no tokens gets *zero* blocks, not a padded-empty one —
    // the reference's `CEILDIV(0, block_size) == 0`.
    for (int e = tid; e < num_experts; e += blockDim.x) {
        int c = 0;
        for (int i = 0; i < numel; ++i) {
            if (topk_ids[i] == e) ++c;
        }
        cumsum[e + 1] = ((c + block_size - 1) / block_size) * block_size;
    }
    __syncthreads();

    if (tid == 0) {
        cumsum[0] = 0;
        for (int i = 0; i < num_experts; ++i) cumsum[i + 1] += cumsum[i];
        *num_tokens_post_pad = cumsum[num_experts];
    }
    __syncthreads();

    for (int e = tid; e < num_experts; e += blockDim.x) {
        int base = cumsum[e];
        int w = 0;
        for (int i = 0; i < numel; ++i) {
            if (topk_ids[i] == e) {
                int slot = base + w;
                if (slot < sorted_capacity) sorted_token_ids[slot] = i;
                ++w;
            }
        }
        int first = cumsum[e] / block_size;
        int last  = cumsum[e + 1] / block_size;
        for (int b = first; b < last && b < expert_capacity; ++b) expert_ids[b] = e;
    }
}

// grid: (1,). block: MOE_THREADS.
__global__ void moe_dispatch_t1(
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
    dispatch_token(topk_ids, valid_tokens, top_k, num_experts, block_size,
                   sorted_capacity, expert_capacity, (int*)xabe_shared,
                   sorted_token_ids, expert_ids, num_tokens_post_pad);
}

// Routing and the dispatch table it feeds, in one launch.
//
// Both run as a single block at one token, one immediately after the other,
// with nothing between them but the eight expert ids the first wrote. Two
// launches for that is two kernel floors -- about 13 us of a 9.8 ms decode
// step across forty layers -- to move eight integers through global memory
// and back.
//
// The `__syncthreads()` between them is what makes the hand-off legal: lane 0
// of warp 0 wrote `topk_ids`, and a block barrier fences those writes for
// every thread that is about to read them.
//
// grid: (1,). block: MOE_THREADS. Shared: probs + rval + ridx + cumsum.
__global__ void moe_route_dispatch_t1(
    const float* __restrict__ logits,
    const int* __restrict__ valid_tokens,
    int num_experts,
    int top_k,
    int block_size,
    int sorted_capacity,
    int expert_capacity,
    int* __restrict__ topk_ids,
    float* __restrict__ topk_weights,
    int* __restrict__ sorted_token_ids,
    int* __restrict__ expert_ids,
    int* __restrict__ num_tokens_post_pad
) {
    if (*valid_tokens < 1) return;
    float* probs  = xabe_shared;
    float* rval   = xabe_shared + num_experts;
    int*   ridx   = (int*)(rval + blockDim.x);
    int*   cumsum = ridx + blockDim.x;

    route_token(logits, 0, num_experts, top_k, probs, rval, ridx,
                topk_ids, topk_weights);
    __syncthreads();
    dispatch_token(topk_ids, valid_tokens, top_k, num_experts, block_size,
                   sorted_capacity, expert_capacity, cumsum,
                   sorted_token_ids, expert_ids, num_tokens_post_pad);
}


__global__ void moe_align_count(
    const int* __restrict__ topk_ids,
    const int* __restrict__ valid_tokens,
    int top_k,
    int sorted_capacity,
    int expert_capacity,
    int* __restrict__ sorted_token_ids,
    int* __restrict__ expert_ids,
    int* __restrict__ counts
) {
    int* scratch = (int*)xabe_shared;
    int numel = (*valid_tokens) * top_k;
    int e = blockIdx.x;
    int tid = threadIdx.x;

    int gtid = blockIdx.x * blockDim.x + tid;
    int gstride = gridDim.x * blockDim.x;
    for (int s = gtid; s < sorted_capacity; s += gstride) sorted_token_ids[s] = numel;
    for (int b = gtid; b < expert_capacity; b += gstride) expert_ids[b] = -1;

    int c = 0;
    for (int i = tid; i < numel; i += blockDim.x) {
        if (topk_ids[i] == e) ++c;
    }
    scratch[tid] = c;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    // Integer addition is associative, so the tree order here cannot change
    // the count the way it would change a float sum.
    if (tid == 0) counts[e] = scratch[0];
}

// grid: (num_experts, 1, 1). Block `e` places expert `e`'s tokens.
//
// **The ordering guarantee is why the cursor is not an atomic counter.**
// The reference places an expert's tokens in ascending flat `(token, k)`
// index. A cursor bumped atomically gives whatever order the warps happened
// to arrive in, which is not reproducible run to run and would make an exact
// comparison against the reference impossible. Instead the block walks the
// flat array in ascending tiles and a per-tile exclusive scan hands each hit
// its rank — the same placement the reference's sequential append produces,
// arrived at in parallel.
__global__ void moe_align_block_size(
    const int* __restrict__ topk_ids,
    const int* __restrict__ valid_tokens,
    const int* __restrict__ counts,
    int top_k,
    int num_experts,
    int block_size,
    int sorted_capacity,
    int expert_capacity,
    int* __restrict__ sorted_token_ids,
    int* __restrict__ expert_ids,
    int* __restrict__ num_tokens_post_pad
) {
    int* cumsum = (int*)xabe_shared;              // num_experts + 1
    int* scan   = cumsum + num_experts + 1;       // blockDim.x

    int numel = (*valid_tokens) * top_k;
    int e = blockIdx.x;
    int tid = threadIdx.x;

    // 1. per-expert counts, padded up to a whole number of blocks. An expert
    //    with no tokens gets *zero* blocks, not a padded-empty one — the
    //    reference's `CEILDIV(0, block_size) == 0`.
    for (int i = tid; i < num_experts; i += blockDim.x) {
        int c = counts[i];
        cumsum[i + 1] = ((c + block_size - 1) / block_size) * block_size;
    }
    __syncthreads();

    // 2. exclusive prefix sum, recomputed identically in every block rather
    //    than read from a third launch: 256 integer adds against a kernel
    //    launch is not a close call, and integer addition is associative so
    //    every block lands on the same cumsum.
    if (tid == 0) {
        cumsum[0] = 0;
        for (int i = 0; i < num_experts; ++i) cumsum[i + 1] += cumsum[i];
        if (e == 0) *num_tokens_post_pad = cumsum[num_experts];
    }
    __syncthreads();

    // 3. scatter in ascending flat index.
    int base = cumsum[e];
    int written = 0;
    for (int o = 0; o < numel; o += blockDim.x) {
        int i = o + tid;
        int hit = (i < numel && topk_ids[i] == e) ? 1 : 0;
        scan[tid] = hit;
        __syncthreads();
        for (int d = 1; d < blockDim.x; d <<= 1) {
            int v = tid >= d ? scan[tid - d] : 0;
            __syncthreads();
            scan[tid] += v;
            __syncthreads();
        }
        int rank  = scan[tid] - hit;
        int total = scan[blockDim.x - 1];
        if (hit) {
            int slot = base + written + rank;
            if (slot < sorted_capacity) sorted_token_ids[slot] = i;
        }
        written += total;
        __syncthreads();
    }

    // 4. claim this expert's blocks.
    int first = cumsum[e] / block_size;
    int last  = cumsum[e + 1] / block_size;
    for (int b = first + tid; b < last && b < expert_capacity; b += blockDim.x) expert_ids[b] = e;
}

// -------------------------------------------------------------------------
// 3. Grouped GEMM: gate/up + SwiGLU, then down.
// -------------------------------------------------------------------------
//
// grid: (ceil(intermediate / MOE_ROWS), expert_block_capacity). One block per
// (band of MOE_ROWS output rows, dispatch block). The grid is the fixed
// *capacity*, never `num_tokens_post_pad` — that value lives only on the
// device. Blocks past the live region carry INACTIVE_EXPERT and exit after
// one load. The predicate is uniform across the block, so the early `return`
// never strands a `__syncthreads()`.
//
// ## Why one block per dispatch block and not one per slot
//
// The first version gave every (row, slot) pair its own block, so the tokens
// sharing an expert each re-read that expert's whole weight stack: at a
// 512-token step that is 16x the traffic a tiled form needs, and the measured
// result was 6.5 GB/s against a 672 GB/s card. Here a block owns a whole
// `block_size` run — which by construction belongs to *one* expert — stages
// those slots' activations in shared memory, and multiplies each dequantized
// weight into all MOE_TM accumulators before dropping it. The stack is then
// read once per dispatch block instead of once per token, and the dequant
// work falls by the same factor.
//
// One warp per output row, MOE_TM accumulators per lane, contraction split
// across the warp's 32 lanes: consecutive lanes read consecutive weight
// elements, which is what makes the Q6_K `ql`/`qh` loads coalesce into whole
// sectors, and no partial sum ever has to cross a warp boundary.
//
// Padding slots are *not* branched around; their staged activation row is
// zeroed, which makes their contribution exactly zero and keeps the
// MOE_TM-wide accumulate unconditional and fully unrolled. Branching per slot
// would put a runtime index on the accumulator array and spill it to local
// memory, which costs far more than the wasted multiply-adds.
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
    float*     xs   = xabe_shared;                              // [MOE_TM][MOE_TK]
    long long* rows = (long long*)(xs + MOE_TM * MOE_TK);       // [MOE_TM]

    int blk = blockIdx.y;
    int e = expert_ids[blk];
    if (e < 0) return;

    int numel = (*valid_tokens) * top_k;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int r = blockIdx.x * MOE_ROWS + warp;
    int live = r < intermediate;
    // Both projections are [intermediate x hidden] per expert, stacked over
    // experts — the GGUF layout `[hidden, intermediate, experts]` with
    // dims[0] fastest-varying.
    long long wrow = ((long long)e * intermediate + r) * hidden;

    for (int m0 = 0; m0 < block_size; m0 += MOE_TM) {
        __syncthreads();
        if (threadIdx.x < MOE_TM) {
            int m = m0 + threadIdx.x;
            int flat = m < block_size
                ? sorted_token_ids[(long long)blk * block_size + m]
                : numel;
            rows[threadIdx.x] =
                flat < numel ? (long long)(flat / top_k) * hidden : -1;
        }
        __syncthreads();

        int bm = live_tile_rows(rows);
        float ag[MOE_TM];
        float au[MOE_TM];
#define MOE_FFN_TILE(TM) tile_gemm_pair<TM>(                                  \
            gate_q, gate_quant, up_q, up_quant, hidden_states, rows, xs,      \
            wrow, hidden, lane, live, ag, au)
        MOE_TILE_DISPATCH(MOE_FFN_TILE)
#undef MOE_FFN_TILE

        if (live && lane == 0) {
            #pragma unroll
            for (int m = 0; m < MOE_TM; ++m) {
                if (m < bm) {
                    // SwiGLU, written exactly as `xabe_kernels::norm::silu`:
                    // x / (1 + exp(-x)), not the algebraically equal
                    // x * sigmoid(x).
                    float act = ag[m] / (1.0f + expf(-ag[m]));
                    inter[((long long)blk * block_size + m0 + m) * intermediate + r] =
                        act * au[m];
                }
            }
        }
    }
}

// -------------------------------------------------------------------------
// 3a. The same two projections when there is exactly one token.
// -------------------------------------------------------------------------
//
// A decode step routes one token to `top_k` distinct experts, so every
// dispatch block holds exactly one live slot and fifteen of padding. The
// kernel above still runs its whole GEMM apparatus for that: it stages an
// activation tile in shared memory, crosses two `__syncthreads()` per
// 128-element slice of the contraction, and dispatches on a tile height that
// is always one. None of it buys anything when the weight is multiplied by a
// single token — there is no reuse to capture, because each dequantized value
// is used once and dropped.
//
// What is left after removing it is a GEMV: one warp per output row, streaming
// its weights and dotting them against an activation vector every warp in the
// grid shares. Measured against the tiled kernel it replaces, the MoE at one
// token goes from 27% of the card's streaming roofline toward what the LM
// head's own GEMV already reaches on the same card (54%).
//
// The per-lane accumulation and the `__shfl_xor` reduction are the tiled
// kernel's, operand for operand, so this is not a different summation order —
// only a different way of arriving at it.
//
// grid: (ceil(intermediate / MOE_ROWS), expert_block_capacity).
__global__ void moe_expert_ffn_gemv(
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
    int blk = blockIdx.y;
    int e = expert_ids[blk];
    if (e < 0) return;

    // Slot 0 and no other: the Rust side launches this only when the pass is
    // one token wide, and one token cannot fill a second slot of a block that
    // belongs to a single expert.
    int flat = sorted_token_ids[(long long)blk * block_size];
    if (flat >= (*valid_tokens) * top_k) return;
    const float* xs = hidden_states + (long long)(flat / top_k) * hidden;

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int r = blockIdx.x * MOE_ROWS + warp;
    if (r >= intermediate) return;
    long long wrow = ((long long)e * intermediate + r) * hidden;

    float ag[1] = {0.0f};
    float au[1] = {0.0f};
    // Two tiles in flight. The two `dequant_tile` calls of one iteration are
    // already independent, but the *next* iteration's loads cannot issue
    // until this one's arithmetic has consumed its registers unless the loop
    // is unrolled — and this kernel moves its weights at 46% of the card's
    // streaming roofline, which is a memory-parallelism number rather than a
    // bandwidth one. Two is the measured knee: 0.65% of the whole decode
    // step, where four gave it back.
    #pragma unroll 2
    for (int j0 = 0; j0 < hidden; j0 += MOE_TK) {
        float wg[MOE_TN];
        float wu[MOE_TN];
        dequant_tile(gate_q, gate_quant, wrow + j0, lane, wg);
        dequant_tile(up_q,   up_quant,   wrow + j0, lane, wu);
        // `hidden` is a multiple of MOE_TK and the row base of a multiple of
        // it, so this 16-byte load is aligned by construction.
        float4 xv = *(const float4*)(xs + j0 + 4 * lane);
        ag[0] += wg[0] * xv.x;  au[0] += wu[0] * xv.x;
        ag[0] += wg[1] * xv.y;  au[0] += wu[1] * xv.y;
        ag[0] += wg[2] * xv.z;  au[0] += wu[2] * xv.z;
        ag[0] += wg[3] * xv.w;  au[0] += wu[3] * xv.w;
    }
    warp_reduce_tile<1>(ag);
    warp_reduce_tile<1>(au);

    if (lane == 0) {
        float act = ag[0] / (1.0f + expf(-ag[0]));
        inter[(long long)blk * block_size * intermediate + r] = act * au[0];
    }
}

// The down projection of the same step. Contracts over `intermediate` and
// writes through `store_slot_contribution`, exactly as the tiled kernel does.
//
// grid: (ceil(hidden / MOE_ROWS), expert_block_capacity).
__global__ void moe_expert_down_gemv(
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
    int blk = blockIdx.y;
    int e = expert_ids[blk];
    if (e < 0) return;

    int numel = (*valid_tokens) * top_k;
    int flat = sorted_token_ids[(long long)blk * block_size];
    if (flat >= numel) return;
    const float* xs = inter + (long long)blk * block_size * intermediate;

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int h = blockIdx.x * MOE_ROWS + warp;
    if (h >= hidden) return;
    long long wrow = ((long long)e * hidden + h) * intermediate;

    float ad[1] = {0.0f};
    // One dequantized matrix a pass, against the gate/up kernel's two, so this
    // loop starts with half the requests in flight and needs the deeper
    // unroll. Accumulation stays sequential into `ad[0]`, so the result is
    // bit-identical to the rolled form.
    #pragma unroll MOE_DOWN_UNROLL
    for (int j0 = 0; j0 < intermediate; j0 += MOE_TK) {
        float wd[MOE_TN];
        dequant_tile(down_q, down_quant, wrow + j0, lane, wd);
        float4 xv = *(const float4*)(xs + j0 + 4 * lane);
        ad[0] += wd[0] * xv.x;
        ad[0] += wd[1] * xv.y;
        ad[0] += wd[2] * xv.z;
        ad[0] += wd[3] * xv.w;
    }
    warp_reduce_tile<1>(ad);

    if (lane == 0) {
        store_slot_contribution(partial, topk_weights, flat, numel, hidden, h, ad[0]);
    }
}

// -------------------------------------------------------------------------
// 3b. The same gate/up projection on the integer tensor cores.
// -------------------------------------------------------------------------
//
// # Why Q6_K is an integer format and not a float one
//
// A Q6_K weight is `d * scale[sub] * (raw - 32)`, where `raw - 32` lands in
// `[-32, 31]` — an int8 with two bits of headroom. The two multipliers are
// constant over a 16-element run and a 256-element superblock respectively,
// so they factor straight out of the contraction:
//
//     sum_k w[k] x[k]  =  d * sum_sub scale[sub] * ( sum_{k in sub} q[k] x[k] )
//
// The inner sum is 16 int8 products. `mma.m8n8k16` contracts **exactly 16**,
// so one instruction is one sub-block, the scale application lands on a clean
// boundary, and no partial scale ever has to be carried across an MMA. The
// shapes were not chosen to match; they already did.
//
// This is why the file's dominant format is a gift rather than an obstacle.
// Dequantizing Q6_K to fp32 and multiplying — what `moe_expert_ffn` above
// does — discards an integer representation the hardware multiplies eight
// times faster than the float one it is being converted into.
//
// # Accumulator width
//
// `|q| <= 32` and `|x| <= 127`, so one MMA accumulates at most
// `32 * 127 * 16 = 65,024`. int32 is not close to overflowing and the integer
// half of every dot product is **exact** — all of the error in this kernel is
// the activation quantization, none of it is accumulation.
//
// # What this kernel does not do
//
// `ffn_down` is Q8_0 in this file, not Q6_K, and stays on the fp32 path. Its
// 34-byte block stride puts every fragment load off a word boundary, which is
// a different problem with a different fix — see `kernels::mma`.
//
// grid: (ceil(intermediate / MOE_MMA_ROWS), expert_block_capacity),
// blockDim MOE_MMA_WARPS * 32. One warp owns 8 output rows — one N fragment —
// and every warp in the block shares the staged activation tile.

#define MOE_MMA_WARPS 8
#define MOE_MMA_N     8
#define MOE_MMA_ROWS  (MOE_MMA_WARPS * MOE_MMA_N)
// Slots staged per block. `block_size` must not exceed this; the Rust side
// checks it, because a larger dispatch block would silently drop its tail.
#define MOE_MMA_M     32
#define MOE_MMA_MF    (MOE_MMA_M / 8)
// Contraction staged per trip: one Q6_K *half*, which is the unit the format's
// `ql`/`qh` split is addressed in. Half a superblock rather than a whole one
// halves the shared footprint for no extra loop overhead.
#define MOE_MMA_KC    128

// Bytes per staged activation row.
//
// 144 and not `MOE_MMA_KC`. A 128-byte row stride is exactly 32 shared banks,
// and the fragment load `sa + (mf * 8 + arow) * stride + kk + quad` varies
// `arow` over eight rows and `quad` over four words -- so with a 32-word
// stride all eight rows land on the same four banks and every one of these
// loads is an **eight-way conflict**, on the kernel that is a quarter of
// prefill. 144 bytes is 36 words and `36 mod 32 = 4`, so row `i` starts four
// banks along from row `i - 1` and the eight rows tile the 32 banks exactly
// once. The sixteen wasted bytes a row buy a conflict-free load.
#define MOE_MMA_ASTRIDE (MOE_MMA_KC + 16)

// Bytes per staged weight row: 64 of `ql`, then 32 of `qh` at offset 64.
//
// 112 and not the 96 the payload needs. The operand load varies the row over
// eight values and the word within a row over four, so a conflict-free stride
// has to send each row exactly four banks along from the last: **the stride in
// words must be 4 mod 32**. 112 bytes is 28 words, `28 * i mod 32` walks
// 0, 28, 24, ..., 4, and the four words each row contributes fill the gaps, so
// the warp's 32 lanes tile the 32 banks exactly once.
//
// 96 is 24 words and `gcd(24, 32) = 8`, which collapses the eight rows into
// four bank groups -- a two-way conflict. 100 was the first fix and only
// spread the rows: 25 is coprime with 32 so the eight row bases are distinct,
// but they are not four apart, and three of the eight collided once the word
// offset was added. Worth about nothing next to
// `MOE_MMA_ASTRIDE`, and kept because it is the shape that is provably right
// rather than the shape that happened to measure the same.
#define MOE_MMA_WSTRIDE 112
// Bytes per staged scale row: 8 int8 sub-scales, then the fp32 superblock
// delta at offset 8 (which is where the 4-byte alignment requirement lands).
#define MOE_MMA_SSTRIDE 16

// Three blocks per SM, asked for explicitly.
//
// This kernel is bandwidth-bound, not compute-bound: at 512 tokens it moves
// 470 MB of Q6_K per layer and issues about 5% of the card's int8 throughput
// doing it, so what it needs from the scheduler is loads in flight, and what
// puts loads in flight is resident warps. Its shared footprint is 21,760 bytes
// and Turing's SM has 65,536 to give, so three blocks fit with 256 bytes to
// spare -- but only if the register allocation also fits three, and ptxas has
// no reason to aim for that unless told.
//
// The second argument is the one that matters. The first is redundant with the
// launch's `block_dim` and is stated so the pair cannot drift apart silently.
#define MOE_MMA_BLOCKS_PER_SM 3

__global__ void __launch_bounds__(MOE_MMA_WARPS * 32, MOE_MMA_BLOCKS_PER_SM)
moe_expert_ffn_mma(
    const unsigned char* __restrict__ gate_q,
    const unsigned char* __restrict__ up_q,
    const signed char* __restrict__ xq,
    const float* __restrict__ xscale,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids,
    const int* __restrict__ valid_tokens,
    int top_k,
    int block_size,
    int hidden,
    int intermediate,
    float* __restrict__ inter
) {
    unsigned char* swg = (unsigned char*)xabe_shared;
    unsigned char* swu = swg + MOE_MMA_ROWS * MOE_MMA_WSTRIDE;
    unsigned char* ssg = swu + MOE_MMA_ROWS * MOE_MMA_WSTRIDE;
    unsigned char* ssu = ssg + MOE_MMA_ROWS * MOE_MMA_SSTRIDE;
    signed char*   sa  = (signed char*)(ssu + MOE_MMA_ROWS * MOE_MMA_SSTRIDE);
    float*         sas = (float*)(sa + MOE_MMA_M * MOE_MMA_ASTRIDE);
    long long*     rows = (long long*)(sas + MOE_MMA_M * (MOE_MMA_KC / 32));

    int blk = blockIdx.y;
    int e = expert_ids[blk];
    if (e < 0) return;
    int numel = (*valid_tokens) * top_k;

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int r0   = blockIdx.x * MOE_MMA_ROWS;

    if (tid < MOE_MMA_M) {
        int flat = tid < block_size
            ? sorted_token_ids[(long long)blk * block_size + tid]
            : numel;
        // The activation *row*, not a byte offset: this tile indexes int8.
        rows[tid] = flat < numel ? (long long)(flat / top_k) : -1;
    }
    __syncthreads();

    // Lane roles. The operand split (stride 4) and the accumulator split
    // (stride 2) are different, which is the characteristic MMA trap: `nload`
    // is the row this lane *loads* an operand for, `ccol` the two columns it
    // *owns* in the accumulator. They are not the same rows.
    int nload = warp * MOE_MMA_N + (lane >> 2);
    int arow  = lane >> 2;
    int quad  = (lane & 3) * 4;
    int ccol  = warp * MOE_MMA_N + (lane & 3) * 2;

    long long ebase = (long long)e * intermediate * hidden;
    int kblocks = hidden >> 5;

    float accg[MOE_MMA_MF][2];
    float accu[MOE_MMA_MF][2];
    #pragma unroll
    for (int mf = 0; mf < MOE_MMA_MF; ++mf) {
        accg[mf][0] = 0.0f; accg[mf][1] = 0.0f;
        accu[mf][0] = 0.0f; accu[mf][1] = 0.0f;
    }

    for (int kc = 0; kc < hidden; kc += MOE_MMA_KC) {
        __syncthreads();

        // Stage the weights. This is the whole point of the rewrite: read
        // straight from global in the fragment layout and consecutive lanes
        // land on rows `hidden` elements apart, so every four-byte operand
        // costs a full 32-byte sector and eight-ninths of the fetch is
        // thrown away. Here a warp walks one row's bytes contiguously — one
        // sector per 32 lanes — and the scatter happens in shared, which has
        // no coalescing to lose.
        // Eight rows per instruction, not one row per four.
        //
        // A warp stages exactly the eight rows it will compute with, and the
        // lane split is `row = lane & 7`, `chunk = lane >> 3`. That does two
        // things at once. The global side becomes `int4`: with the device
        // stride padded to 224 every field of every superblock is 16-byte
        // aligned, so 8 rows x 64 bytes of `ql` is one load where the 210-byte
        // file layout forced 16-bit loads and eight of them. The shared side
        // becomes conflict-free: a 128-bit store is serviced eight lanes at a
        // time and those eight lanes hold eight *different* rows, so with a
        // stride of 28 words -- `28 mod 32 = -4` -- each lane's four banks sit
        // four along from the last and the eight tile the 32 banks exactly.
        //
        // Staging measured 32 ms of this kernel's 62 and stayed that expensive
        // with every byte already in L1, so what it cost was the count of
        // loads. Four per row per matrix becomes four per *eight* rows.
        int half = (kc >> 7) & 1;
        {
            int jq = lane & 7;
            int c4 = lane >> 3;
            int rq = warp * MOE_MMA_N + jq;
            int nq = r0 + rq;
            int live = nq < intermediate;
            long long iq = ebase + (long long)nq * hidden + kc;
            const unsigned char* gq = gate_q + (iq >> 8) * Q6K_SB;
            const unsigned char* uq = up_q   + (iq >> 8) * Q6K_SB;

            if (live) {
                *(uint4*)(swg + rq * MOE_MMA_WSTRIDE + c4 * 16) =
                    *(const uint4*)(gq + half * 64 + c4 * 16);
                *(uint4*)(swu + rq * MOE_MMA_WSTRIDE + c4 * 16) =
                    *(const uint4*)(uq + half * 64 + c4 * 16);
                if (lane < 16) {
                    *(uint4*)(swg + rq * MOE_MMA_WSTRIDE + 64 + c4 * 16) =
                        *(const uint4*)(gq + 128 + half * 32 + c4 * 16);
                    *(uint4*)(swu + rq * MOE_MMA_WSTRIDE + 64 + c4 * 16) =
                        *(const uint4*)(uq + 128 + half * 32 + c4 * 16);
                }
            }
            if (lane < 8) {
                if (live) {
                    *(uint2*)(ssg + rq * MOE_MMA_SSTRIDE) =
                        *(const uint2*)(gq + 192 + half * 8);
                    *(uint2*)(ssu + rq * MOE_MMA_SSTRIDE) =
                        *(const uint2*)(uq + 192 + half * 8);
                    *(float*)(ssg + rq * MOE_MMA_SSTRIDE + 8) =
                        half_bits_to_float(*(const unsigned short*)(gq + 208));
                    *(float*)(ssu + rq * MOE_MMA_SSTRIDE + 8) =
                        half_bits_to_float(*(const unsigned short*)(uq + 208));
                } else {
                    // A row past `intermediate` contributes nothing, and
                    // zeroing the *scale* is enough to guarantee that without
                    // zeroing 96 bytes of quants: every product it feeds is
                    // multiplied by it.
                    *(float*)(ssg + rq * MOE_MMA_SSTRIDE + 8) = 0.0f;
                    *(float*)(ssu + rq * MOE_MMA_SSTRIDE + 8) = 0.0f;
                }
            }
        }

        // Stage the activations as words: `hidden` and `MOE_MMA_KC` are
        // multiples of 4, so every one of these is aligned. A padding slot
        // stages zeros, which makes its products exactly zero and keeps the
        // inner loop branch-free.
        for (int idx = tid; idx < MOE_MMA_M * (MOE_MMA_KC / 4); idx += blockDim.x) {
            int m  = idx / (MOE_MMA_KC / 4);
            int k4 = (idx % (MOE_MMA_KC / 4)) * 4;
            long long row = rows[m];
            unsigned int v = row >= 0
                ? *(const unsigned int*)(xq + row * hidden + kc + k4)
                : 0u;
            *(unsigned int*)(sa + m * MOE_MMA_ASTRIDE + k4) = v;
        }
        for (int idx = tid; idx < MOE_MMA_M * (MOE_MMA_KC / 32); idx += blockDim.x) {
            int m  = idx / (MOE_MMA_KC / 32);
            int kb = idx % (MOE_MMA_KC / 32);
            long long row = rows[m];
            sas[m * (MOE_MMA_KC / 32) + kb] =
                row >= 0 ? xscale[row * kblocks + (kc >> 5) + kb] : 0.0f;
        }
        __syncthreads();

        // Hoisted: the superblock delta moves once per staged half, not once
        // per sub-block.
        float dg0 = *(const float*)(ssg + ccol * MOE_MMA_SSTRIDE + 8);
        float dg1 = *(const float*)(ssg + (ccol + 1) * MOE_MMA_SSTRIDE + 8);
        float du0 = *(const float*)(ssu + ccol * MOE_MMA_SSTRIDE + 8);
        float du1 = *(const float*)(ssu + (ccol + 1) * MOE_MMA_SSTRIDE + 8);

        // Two 16-wide sub-blocks per trip, because 32 is the span one
        // activation scale covers and that is what lets the pair share a
        // single conversion to float.
        //
        // # Why the loop is shaped around `I2F`
        //
        // The obvious inner loop converts each MMA result to float and scales
        // it there: four accumulators times four activation fragments is
        // sixteen `I2F` per 16 elements of contraction, plus four more for the
        // sub-scales. SASS says 20 of the loop's 105 instructions were `I2F`,
        // and on Turing integer-to-float runs on the conversion pipe at a
        // quarter of the FMA pipe's rate -- so those 20 cost as much as the
        // other 85 together, in a kernel that is a quarter of prefill.
        //
        // The sub-scale is an int8 and the MMA result is at most
        // `32 * 127 * 16 = 65,024`, so their product fits in 23 bits and the
        // two sub-blocks' products sum to at most 16.5 M. Accumulating *that*
        // in int32 is exact, needs one `IMAD` per sub-block, and leaves one
        // conversion per accumulator per 32 elements where there were four.
        // The int8 sub-scales never become floats at all.
        //
        // The float arithmetic that remains is `(float)acc * (dx * d)`, where
        // `d` is the superblock delta hoisted above and `dx` the activation
        // scale for these 32 elements. Against the old expression this is one
        // rounding instead of three per pair of sub-blocks, so the result is
        // not bit-identical to what this kernel produced before -- it is
        // slightly *more* accurate, and `tests/forward_pass.rs` gates the
        // difference against llama.cpp's own activations.
        for (int kk = 0; kk < MOE_MMA_KC; kk += 32) {
            unsigned int bg[2], bu[2];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                int rp  = kk + 16 * h + quad;
                int grp = rp >> 5;
                int l   = rp & 31;
                int shift = 2 * grp;
                // `l` is a multiple of 4 and the shared stride is too, so these
                // are aligned word loads where the equivalent global reads had to
                // be assembled byte by byte: Q6_K's 210-byte block stride leaves
                // no alignment to rely on, but a layout this kernel chose does.
                unsigned int qlg = *(const unsigned int*)(
                    swg + nload * MOE_MMA_WSTRIDE + ((grp & 1) ? l + 32 : l));
                unsigned int qhg = *(const unsigned int*)(
                    swg + nload * MOE_MMA_WSTRIDE + 64 + l);
                unsigned int qlu = *(const unsigned int*)(
                    swu + nload * MOE_MMA_WSTRIDE + ((grp & 1) ? l + 32 : l));
                unsigned int qhu = *(const unsigned int*)(
                    swu + nload * MOE_MMA_WSTRIDE + 64 + l);

                // Four Q6_K codes to four signed bytes with no per-element work
                // at all. Every step below acts on all four lanes of the word at
                // once, and none of them can carry a bit across a byte boundary:
                //
                //   nibble   `(q >> 4) & 0x0F0F0F0F` takes bits 4..7 of each byte
                //   high two `(qh >> shift) & 0x03030303`, shift <= 6, so bits
                //            shift..shift+1 of each byte and no further
                //   bias     Q6_K stores `raw - 32` in offset binary, and offset
                //            binary *is* two's complement with the sign bit
                //            flipped -- so `^ 0x20` converts all four codes at
                //            once, leaving a 6-bit signed value per byte
                //   extend   bit 5 is now the sign; copying it into bits 6 and 7
                //            with two shifted ORs widens all four to int8
                //
                // Nine word operations per matrix where the per-element loop
                // needed about forty, on the kernel that is 25% of prefill. The
                // byte patterns are identical, so the MMA sees the same operands
                // it always did.
                unsigned int tg = ((((grp < 2) ? qlg : (qlg >> 4)) & 0x0F0F0F0Fu)
                    | (((qhg >> shift) & 0x03030303u) << 4)) ^ 0x20202020u;
                unsigned int tu = ((((grp < 2) ? qlu : (qlu >> 4)) & 0x0F0F0F0Fu)
                    | (((qhu >> shift) & 0x03030303u) << 4)) ^ 0x20202020u;
                unsigned int sg = tg & 0x20202020u;
                unsigned int su = tu & 0x20202020u;
                bg[h] = tg | (sg << 1) | (sg << 2);
                bu[h] = tu | (su << 1) | (su << 2);
            }

            int sub = kk >> 4;
            int cg0[2], cg1[2], cu0[2], cu1[2];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                cg0[h] = (signed char)ssg[ccol * MOE_MMA_SSTRIDE + sub + h];
                cg1[h] = (signed char)ssg[(ccol + 1) * MOE_MMA_SSTRIDE + sub + h];
                cu0[h] = (signed char)ssu[ccol * MOE_MMA_SSTRIDE + sub + h];
                cu1[h] = (signed char)ssu[(ccol + 1) * MOE_MMA_SSTRIDE + sub + h];
            }

            #pragma unroll
            for (int mf = 0; mf < MOE_MMA_MF; ++mf) {
                float dx = sas[(mf * 8 + arow) * (MOE_MMA_KC / 32) + (kk >> 5)];
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    unsigned int a = *(const unsigned int*)(
                        sa + (mf * 8 + arow) * MOE_MMA_ASTRIDE + kk + 16 * h + quad);
                    int g0 = 0, g1 = 0, u0 = 0, u1 = 0;
                    asm volatile(
                        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                        "{%0,%1}, {%2}, {%3}, {%0,%1};"
                        : "+r"(g0), "+r"(g1) : "r"(a), "r"(bg[h]));
                    asm volatile(
                        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                        "{%0,%1}, {%2}, {%3}, {%0,%1};"
                        : "+r"(u0), "+r"(u1) : "r"(a), "r"(bu[h]));
                    accg[mf][0] += (float)g0 * dx * (dg0 * (float)cg0[h]);
                    accg[mf][1] += (float)g1 * dx * (dg1 * (float)cg1[h]);
                    accu[mf][0] += (float)u0 * dx * (du0 * (float)cu0[h]);
                    accu[mf][1] += (float)u1 * dx * (du1 * (float)cu1[h]);
                }
            }
        }
    }

    // SwiGLU, written exactly as `xabe_kernels::norm::silu`: x / (1 + exp(-x)),
    // not the algebraically equal x * sigmoid(x). Padding slots are dropped
    // rather than written, which is what lets `moe_expert_down` treat an
    // unwritten `inter` row as unreachable instead of as zero.
    #pragma unroll
    for (int mf = 0; mf < MOE_MMA_MF; ++mf) {
        int m = mf * 8 + arow;
        if (m < block_size && rows[m] >= 0) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int n = r0 + ccol + j;
                if (n < intermediate) {
                    float g = (j == 0) ? accg[mf][0] : accg[mf][1];
                    float u = (j == 0) ? accu[mf][0] : accu[mf][1];
                    float act = g / (1.0f + expf(-g));
                    inter[((long long)blk * block_size + m) * intermediate + n] = act * u;
                }
            }
        }
    }
}

// grid: (ceil(hidden / MOE_ROWS), expert_block_capacity). The same tiling as
// above with the contraction running over `intermediate` instead of `hidden`.
//
// Writes each (token, k) contribution to its own slice of `partial` rather
// than accumulating into the output with atomics. Two reasons: atomicAdd
// makes the summation order non-deterministic, so the same input would give
// bit-different output run to run; and the deterministic reduction below can
// then sum in ascending k, matching the reference's per-token loop.
//
// Padding slots read `inter` as zero rather than as whatever the previous
// step left there, and `store_slot_contribution` drops their output entirely.
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
    float*     xs        = xabe_shared;                          // [MOE_TM][MOE_TK]
    long long* rows      = (long long*)(xs + MOE_TM * MOE_TK);   // [MOE_TM]
    int*       slot_flat = (int*)(rows + MOE_TM);                // [MOE_TM]

    int blk = blockIdx.y;
    int e = expert_ids[blk];
    if (e < 0) return;

    int numel = (*valid_tokens) * top_k;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int h = blockIdx.x * MOE_ROWS + warp;
    int live = h < hidden;
    // [hidden x intermediate] per expert — GGUF `[intermediate, hidden, experts]`.
    long long wrow = ((long long)e * hidden + h) * intermediate;

    for (int m0 = 0; m0 < block_size; m0 += MOE_TM) {
        __syncthreads();
        if (threadIdx.x < MOE_TM) {
            int m = m0 + threadIdx.x;
            int flat = m < block_size
                ? sorted_token_ids[(long long)blk * block_size + m]
                : numel;
            slot_flat[threadIdx.x] = flat;
            rows[threadIdx.x] = flat < numel
                ? ((long long)blk * block_size + m) * intermediate
                : -1;
        }
        __syncthreads();

        int bm = live_tile_rows(rows);
        float ad[MOE_TM];
#define MOE_DOWN_TILE(TM) tile_gemm_single<TM>(                               \
            down_q, down_quant, inter, rows, xs,                              \
            wrow, intermediate, lane, live, ad)
        MOE_TILE_DISPATCH(MOE_DOWN_TILE)
#undef MOE_DOWN_TILE

        if (live && lane == 0) {
            #pragma unroll
            for (int m = 0; m < MOE_TM; ++m) {
                if (m < bm) {
                    store_slot_contribution(
                        partial, topk_weights, slot_flat[m], numel, hidden, h, ad[m]);
                }
            }
        }
    }
}

// -------------------------------------------------------------------------
// 3c. The down projection on the integer tensor cores.
// -------------------------------------------------------------------------
//
// Same shape of argument as `moe_expert_ffn_mma`, one format down. A Q8_0
// block is 32 int8 with an fp16 scale, so the scale factors out of *two*
// consecutive `m8n8k16` contractions rather than one, and the quants need no
// bit-unpacking at all — the only thing standing between them and a tensor
// core is the 34-byte block stride, which puts every four-byte operand off a
// word boundary.
//
// `kernels::mma` solves that for the dense projections by repacking the tensor
// into split quant and scale arrays. That is not affordable here: `ffn_down_
// exps` is 10.7 G weights across the 40 layers, and a second copy would not
// fit beside the model. Staging through shared memory solves it for free —
// the kernel chooses the shared layout, so it can put the quants on a word
// boundary and the scales somewhere else entirely.
//
// grid: (ceil(hidden / MOE_MMA_ROWS), expert_block_capacity). The contraction
// runs over `intermediate` rather than `hidden`.

// Bytes per staged weight row: MOE_MMA_KC quants, then one fp32 scale per 32
// at offset MOE_MMA_KC.
//
// 144 is 36 words, and the eight rows a warp reads land on banks
// `4r + (quad/4)` — thirty-two distinct banks across the warp, no conflict.
#define MOE_MMA_DSTRIDE (MOE_MMA_KC + (MOE_MMA_KC / 32) * 4)

// Four blocks per SM, for the same reason the gate/up kernel asks for three.
// Q8_0 stages one weight tile rather than two, so this kernel's footprint is
// 14,720 bytes and four of them fit in 65,536 with room left.
#define MOE_DOWN_BLOCKS_PER_SM 4

__global__ void __launch_bounds__(MOE_MMA_WARPS * 32, MOE_DOWN_BLOCKS_PER_SM)
moe_expert_down_mma(
    const unsigned char* __restrict__ down_q,
    const signed char* __restrict__ iq,
    const float* __restrict__ iscale,
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
    unsigned char* sw  = (unsigned char*)xabe_shared;
    signed char*   sa  = (signed char*)(sw + MOE_MMA_ROWS * MOE_MMA_DSTRIDE);
    float*         sas = (float*)(sa + MOE_MMA_M * MOE_MMA_ASTRIDE);
    long long*     rows = (long long*)(sas + MOE_MMA_M * (MOE_MMA_KC / 32));
    int*           slot_flat = (int*)(rows + MOE_MMA_M);

    int blk = blockIdx.y;
    int e = expert_ids[blk];
    if (e < 0) return;
    int numel = (*valid_tokens) * top_k;

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int r0   = blockIdx.x * MOE_MMA_ROWS;

    if (tid < MOE_MMA_M) {
        int m = tid;
        int flat = m < block_size
            ? sorted_token_ids[(long long)blk * block_size + m]
            : numel;
        slot_flat[m] = flat;
        // The `inter` row, which is the *slot* index and not the token index:
        // the ffn kernel wrote one row per dispatch slot.
        rows[m] = flat < numel ? (long long)blk * block_size + m : -1;
    }
    __syncthreads();

    int nload = warp * MOE_MMA_N + (lane >> 2);
    int arow  = lane >> 2;
    int quad  = (lane & 3) * 4;
    int ccol  = warp * MOE_MMA_N + (lane & 3) * 2;

    // [hidden x intermediate] per expert — GGUF `[intermediate, hidden, experts]`.
    long long ebase = (long long)e * hidden * intermediate;
    int kblocks = intermediate >> 5;

    float acc[MOE_MMA_MF][2];
    #pragma unroll
    for (int mf = 0; mf < MOE_MMA_MF; ++mf) { acc[mf][0] = 0.0f; acc[mf][1] = 0.0f; }

    for (int kc = 0; kc < intermediate; kc += MOE_MMA_KC) {
        __syncthreads();

        for (int r = warp; r < MOE_MMA_ROWS; r += MOE_MMA_WARPS) {
            int n = r0 + r;
            if (n < hidden) {
                const unsigned char* src =
                    down_q + ((ebase + (long long)n * intermediate + kc) >> 5) * 34;
                // Each trip reads 32 bytes contiguous within one block, so the
                // fetch coalesces even though the block stride does not let it
                // be a word load.
                // Two bytes per lane. The quants of a Q8_0 block start at
                // byte 2 of a 34-byte block, so they are even-aligned and
                // never word-aligned; 16-bit is the widest legal load, and it
                // halves the instructions this copy costs.
                for (int t = lane * 2; t < MOE_MMA_KC; t += 64) {
                    *(unsigned short*)(sw + r * MOE_MMA_DSTRIDE + t) =
                        *(const unsigned short*)(src + (t >> 5) * 34 + 2 + (t & 31));
                }
                if (lane < (MOE_MMA_KC / 32)) {
                    // One 16-bit load, not `load_half_le`'s two 8-bit ones: a
                    // Q8_0 block starts on an even byte, so the fp16 scale at
                    // its head is 2-byte aligned even though the 34-byte
                    // stride never makes it 4-byte aligned. Staging is what
                    // this kernel spends its time on, and this is one of the
                    // four loads a row was costing.
                    *(float*)(sw + r * MOE_MMA_DSTRIDE + MOE_MMA_KC + lane * 4) =
                        half_bits_to_float(*(const unsigned short*)(src + lane * 34));
                }
            } else if (lane < (MOE_MMA_KC / 32)) {
                // Zeroing the scale is enough: every product it feeds is
                // multiplied by it.
                *(float*)(sw + r * MOE_MMA_DSTRIDE + MOE_MMA_KC + lane * 4) = 0.0f;
            }
        }

        for (int idx = tid; idx < MOE_MMA_M * (MOE_MMA_KC / 4); idx += blockDim.x) {
            int m  = idx / (MOE_MMA_KC / 4);
            int k4 = (idx % (MOE_MMA_KC / 4)) * 4;
            long long row = rows[m];
            unsigned int v = row >= 0
                ? *(const unsigned int*)(iq + row * intermediate + kc + k4)
                : 0u;
            *(unsigned int*)(sa + m * MOE_MMA_ASTRIDE + k4) = v;
        }
        for (int idx = tid; idx < MOE_MMA_M * (MOE_MMA_KC / 32); idx += blockDim.x) {
            int m  = idx / (MOE_MMA_KC / 32);
            int kb = idx % (MOE_MMA_KC / 32);
            long long row = rows[m];
            sas[m * (MOE_MMA_KC / 32) + kb] =
                row >= 0 ? iscale[row * kblocks + (kc >> 5) + kb] : 0.0f;
        }
        __syncthreads();

        for (int kk = 0; kk < MOE_MMA_KC; kk += 16) {
            unsigned int b = *(const unsigned int*)(
                sw + nload * MOE_MMA_DSTRIDE + kk + quad);

            // One Q8_0 scale spans 32 contraction elements, so it is the same
            // for this k-step and the next.
            int sb = kk >> 5;
            float w0 = *(const float*)(
                sw + ccol * MOE_MMA_DSTRIDE + MOE_MMA_KC + sb * 4);
            float w1 = *(const float*)(
                sw + (ccol + 1) * MOE_MMA_DSTRIDE + MOE_MMA_KC + sb * 4);

            #pragma unroll
            for (int mf = 0; mf < MOE_MMA_MF; ++mf) {
                unsigned int a = *(const unsigned int*)(
                    sa + (mf * 8 + arow) * MOE_MMA_ASTRIDE + kk + quad);
                int d0 = 0, d1 = 0;
                asm volatile(
                    "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                    "{%0,%1}, {%2}, {%3}, {%0,%1};"
                    : "+r"(d0), "+r"(d1) : "r"(a), "r"(b));
                float dx = sas[(mf * 8 + arow) * (MOE_MMA_KC / 32) + sb];
                acc[mf][0] += (float)d0 * dx * w0;
                acc[mf][1] += (float)d1 * dx * w1;
            }
        }
    }

    #pragma unroll
    for (int mf = 0; mf < MOE_MMA_MF; ++mf) {
        int m = mf * 8 + arow;
        if (m < block_size) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int n = r0 + ccol + j;
                if (n < hidden) {
                    store_slot_contribution(
                        partial, topk_weights, slot_flat[m], numel, hidden, n,
                        (j == 0) ? acc[mf][0] : acc[mf][1]);
                }
            }
        }
    }
}

// The gate/up half of the grouped GEMM when both stacks are Q8_0.
//
// `Qwen3.6-35B-A3B-UD-Q6_K_XL` stores `ffn_gate_exps` and `ffn_up_exps` as
// Q6_K on every block but **39**, where they are Q8_0. One layer in forty took
// the fp32 fallback for that, and it cost 5.8 ms of a 246 ms pass against
// 1.5 ms for a tensor-core layer -- 2.4% of prefill for 2.5% of the model.
//
// It is `moe_expert_ffn_mma`'s body with `moe_expert_down_mma`'s operand
// handling: the same tile shape, the same lane roles, the same SwiGLU
// epilogue, but a Q8_0 weight tile and an fp32 block scale where Q6_K needed a
// packed six-bit tile and an int8 sub-scale. The staged layout is
// `MOE_MMA_DSTRIDE` -- `MOE_MMA_KC` quant bytes then `MOE_MMA_KC / 32` fp32
// scales -- for both matrices.
//
// grid and block are `moe_expert_ffn_mma`'s exactly, so the two are
// interchangeable at the launch site.
__global__ void moe_expert_ffn_mma_q8(
    const unsigned char* __restrict__ gate_q,
    const unsigned char* __restrict__ up_q,
    const signed char* __restrict__ xq,
    const float* __restrict__ xscale,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids,
    const int* __restrict__ valid_tokens,
    int top_k,
    int block_size,
    int hidden,
    int intermediate,
    float* __restrict__ inter
) {
    unsigned char* swg = (unsigned char*)xabe_shared;
    unsigned char* swu = swg + MOE_MMA_ROWS * MOE_MMA_DSTRIDE;
    signed char*   sa  = (signed char*)(swu + MOE_MMA_ROWS * MOE_MMA_DSTRIDE);
    float*         sas = (float*)(sa + MOE_MMA_M * MOE_MMA_ASTRIDE);
    long long*     rows = (long long*)(sas + MOE_MMA_M * (MOE_MMA_KC / 32));

    int blk = blockIdx.y;
    int e = expert_ids[blk];
    if (e < 0) return;
    int numel = (*valid_tokens) * top_k;

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int r0   = blockIdx.x * MOE_MMA_ROWS;

    if (tid < MOE_MMA_M) {
        int flat = tid < block_size
            ? sorted_token_ids[(long long)blk * block_size + tid]
            : numel;
        rows[tid] = flat < numel ? (long long)(flat / top_k) : -1;
    }
    __syncthreads();

    int nload = warp * MOE_MMA_N + (lane >> 2);
    int arow  = lane >> 2;
    int quad  = (lane & 3) * 4;
    int ccol  = warp * MOE_MMA_N + (lane & 3) * 2;

    long long ebase = (long long)e * intermediate * hidden;
    int kblocks = hidden >> 5;

    float accg[MOE_MMA_MF][2];
    float accu[MOE_MMA_MF][2];
    #pragma unroll
    for (int mf = 0; mf < MOE_MMA_MF; ++mf) {
        accg[mf][0] = 0.0f; accg[mf][1] = 0.0f;
        accu[mf][0] = 0.0f; accu[mf][1] = 0.0f;
    }

    for (int kc = 0; kc < hidden; kc += MOE_MMA_KC) {
        __syncthreads();

        // A warp stages whole rows of both matrices. The quants of a Q8_0
        // block start at byte 2 of a 34-byte block, so they are even-aligned
        // and never word-aligned; 16 bits is the widest legal load, and `t` is
        // even so a pair never straddles a block boundary.
        for (int r = warp; r < MOE_MMA_ROWS; r += MOE_MMA_WARPS) {
            int n = r0 + r;
            if (n < intermediate) {
                long long off = (ebase + (long long)n * hidden + kc) >> 5;
                const unsigned char* sg = gate_q + off * 34;
                const unsigned char* su = up_q   + off * 34;
                for (int t = lane * 2; t < MOE_MMA_KC; t += 64) {
                    int at = (t >> 5) * 34 + 2 + (t & 31);
                    *(unsigned short*)(swg + r * MOE_MMA_DSTRIDE + t) =
                        *(const unsigned short*)(sg + at);
                    *(unsigned short*)(swu + r * MOE_MMA_DSTRIDE + t) =
                        *(const unsigned short*)(su + at);
                }
                if (lane < (MOE_MMA_KC / 32)) {
                    *(float*)(swg + r * MOE_MMA_DSTRIDE + MOE_MMA_KC + lane * 4) =
                        half_bits_to_float(*(const unsigned short*)(sg + lane * 34));
                    *(float*)(swu + r * MOE_MMA_DSTRIDE + MOE_MMA_KC + lane * 4) =
                        half_bits_to_float(*(const unsigned short*)(su + lane * 34));
                }
            } else if (lane < (MOE_MMA_KC / 32)) {
                // A row past `intermediate` contributes nothing, and zeroing
                // the scale is enough to guarantee it: every product it feeds
                // is multiplied by it.
                *(float*)(swg + r * MOE_MMA_DSTRIDE + MOE_MMA_KC + lane * 4) = 0.0f;
                *(float*)(swu + r * MOE_MMA_DSTRIDE + MOE_MMA_KC + lane * 4) = 0.0f;
            }
        }

        for (int idx = tid; idx < MOE_MMA_M * (MOE_MMA_KC / 4); idx += blockDim.x) {
            int m  = idx / (MOE_MMA_KC / 4);
            int k4 = (idx % (MOE_MMA_KC / 4)) * 4;
            long long row = rows[m];
            unsigned int v = row >= 0
                ? *(const unsigned int*)(xq + row * hidden + kc + k4)
                : 0u;
            *(unsigned int*)(sa + m * MOE_MMA_ASTRIDE + k4) = v;
        }
        for (int idx = tid; idx < MOE_MMA_M * (MOE_MMA_KC / 32); idx += blockDim.x) {
            int m  = idx / (MOE_MMA_KC / 32);
            int kb = idx % (MOE_MMA_KC / 32);
            long long row = rows[m];
            sas[m * (MOE_MMA_KC / 32) + kb] =
                row >= 0 ? xscale[row * kblocks + (kc >> 5) + kb] : 0.0f;
        }
        __syncthreads();

        for (int kk = 0; kk < MOE_MMA_KC; kk += 16) {
            unsigned int bg = *(const unsigned int*)(
                swg + nload * MOE_MMA_DSTRIDE + kk + quad);
            unsigned int bu = *(const unsigned int*)(
                swu + nload * MOE_MMA_DSTRIDE + kk + quad);

            // One Q8_0 scale spans 32 contraction elements, so it is the same
            // for this k-step and the next.
            int sb = kk >> 5;
            float wg0 = *(const float*)(
                swg + ccol * MOE_MMA_DSTRIDE + MOE_MMA_KC + sb * 4);
            float wg1 = *(const float*)(
                swg + (ccol + 1) * MOE_MMA_DSTRIDE + MOE_MMA_KC + sb * 4);
            float wu0 = *(const float*)(
                swu + ccol * MOE_MMA_DSTRIDE + MOE_MMA_KC + sb * 4);
            float wu1 = *(const float*)(
                swu + (ccol + 1) * MOE_MMA_DSTRIDE + MOE_MMA_KC + sb * 4);

            #pragma unroll
            for (int mf = 0; mf < MOE_MMA_MF; ++mf) {
                float dx = sas[(mf * 8 + arow) * (MOE_MMA_KC / 32) + sb];
                unsigned int a = *(const unsigned int*)(
                    sa + (mf * 8 + arow) * MOE_MMA_ASTRIDE + kk + quad);
                int g0 = 0, g1 = 0, u0 = 0, u1 = 0;
                asm volatile(
                    "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                    "{%0,%1}, {%2}, {%3}, {%0,%1};"
                    : "+r"(g0), "+r"(g1) : "r"(a), "r"(bg));
                asm volatile(
                    "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                    "{%0,%1}, {%2}, {%3}, {%0,%1};"
                    : "+r"(u0), "+r"(u1) : "r"(a), "r"(bu));
                accg[mf][0] += (float)g0 * dx * wg0;
                accg[mf][1] += (float)g1 * dx * wg1;
                accu[mf][0] += (float)u0 * dx * wu0;
                accu[mf][1] += (float)u1 * dx * wu1;
            }
        }
    }

    // SwiGLU, written exactly as `xabe_kernels::norm::silu`, and padding slots
    // dropped rather than written -- both as `moe_expert_ffn_mma` has them.
    #pragma unroll
    for (int mf = 0; mf < MOE_MMA_MF; ++mf) {
        int m = mf * 8 + arow;
        if (m < block_size && rows[m] >= 0) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int n = r0 + ccol + j;
                if (n < intermediate) {
                    float g = (j == 0) ? accg[mf][0] : accg[mf][1];
                    float u = (j == 0) ? accu[mf][0] : accu[mf][1];
                    float act = g / (1.0f + expf(-g));
                    inter[((long long)blk * block_size + m) * intermediate + n] = act * u;
                }
            }
        }
    }
}


// SwiGLU for the shared expert's tensor-core path.
//
// The fp32 shared-expert kernel fuses this into its epilogue; the tensor-core
// path cannot, because `mma_q8_0_proj_split` is a projection and knows nothing
// about what its output feeds. One elementwise pass over `[max_tokens]
// [intermediate]` is a rounding error against the two projections it sits
// between.
//
// `x / (1 + exp(-x))`, exactly as `xabe_kernels::norm::silu` and the fp32
// kernel above write it — not the algebraically equal `x * sigmoid(x)`.
__global__ void moe_swiglu(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    long long n,
    float* __restrict__ out
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    out[i] = (g / (1.0f + expf(-g))) * up[i];
}

// The shared expert at one token, for the same reason as
// `moe_expert_ffn_gemv`: at this shape the tiled kernel's staging and barriers
// buy nothing, because each dequantized weight is multiplied once.
//
// Simpler than the routed pair — the shared expert consults no dispatch table
// and is applied to every token unconditionally, so there is no slot to look
// up and no routing weight to apply.
//
// **Split over the contraction, one output row per block.** The obvious shape
// — one warp per row, MOE_ROWS rows per block — gives this kernel 64 blocks
// at the real geometry, because the shared expert's intermediate is 512 and
// MOE_ROWS is 8. Sixty-four blocks on a 72-SM card leaves eight SMs with no
// work at all and the other sixty-four running one block of eight warps, a
// quarter of the threads sm_75 will hold. It measured at **16% of the card's
// streaming roofline**, the worst of any kernel in a decode step, while the
// routed pair next door — same arithmetic, same format, 512 blocks — reached
// 47%.
//
// So the block owns one row and its MOE_ROWS warps split the contraction
// between them, which multiplies the block count by MOE_ROWS instead of
// dividing the row count by it. The per-warp partial sums are combined
// through shared memory at the end: eight additions, once per row, against
// the eight-fold parallelism they buy.
//
// grid: (intermediate,). block: MOE_ROWS warps, all on one row.
__global__ void moe_shared_ffn_gemv(
    const unsigned char* __restrict__ gate_q, int gate_quant,
    const unsigned char* __restrict__ up_q,   int up_quant,
    const float* __restrict__ hidden_states,
    const int* __restrict__ valid_tokens,
    int hidden,
    int intermediate,
    float* __restrict__ inter
) {
    if (*valid_tokens < 1) return;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int r = blockIdx.x;
    if (r >= intermediate) return;
    long long wrow = (long long)r * hidden;

    // The launch path rejects a `hidden` this does not divide evenly.
    int chunk = hidden / MOE_SHARED_WARPS;
    int stop = warp * chunk + chunk;

    float ag[1] = {0.0f};
    float au[1] = {0.0f};
    // Two streams like the routed gate/up kernel, but its literal 2 was tuned
    // against a `hidden`-long contraction; this one walks only `hidden /
    // MOE_SHARED_WARPS`, so it has fewer passes to hide the same latency.
    #pragma unroll 2
    for (int j0 = warp * chunk; j0 < stop; j0 += MOE_TK) {
        float wg[MOE_TN];
        float wu[MOE_TN];
        dequant_tile(gate_q, gate_quant, wrow + j0, lane, wg);
        dequant_tile(up_q,   up_quant,   wrow + j0, lane, wu);
        float4 xv = *(const float4*)(hidden_states + j0 + 4 * lane);
        ag[0] += wg[0] * xv.x;  au[0] += wu[0] * xv.x;
        ag[0] += wg[1] * xv.y;  au[0] += wu[1] * xv.y;
        ag[0] += wg[2] * xv.z;  au[0] += wu[2] * xv.z;
        ag[0] += wg[3] * xv.w;  au[0] += wu[3] * xv.w;
    }
    warp_reduce_tile<1>(ag);
    warp_reduce_tile<1>(au);

    __shared__ float sg[MOE_SHARED_WARPS];
    __shared__ float su[MOE_SHARED_WARPS];
    if (lane == 0) { sg[warp] = ag[0]; su[warp] = au[0]; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float g = 0.0f;
        float u = 0.0f;
        for (int w = 0; w < MOE_SHARED_WARPS; ++w) { g += sg[w]; u += su[w]; }
        inter[r] = (g / (1.0f + expf(-g))) * u;
    }
}

// grid: (ceil(hidden / MOE_ROWS),).
__global__ void moe_shared_down_gemv(
    const unsigned char* __restrict__ down_q, int down_quant,
    const float* __restrict__ inter,
    const int* __restrict__ valid_tokens,
    int hidden,
    int intermediate,
    float* __restrict__ out
) {
    if (*valid_tokens < 1) return;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int h = blockIdx.x * MOE_ROWS + warp;
    if (h >= hidden) return;
    long long wrow = (long long)h * intermediate;

    float ad[1] = {0.0f};
    // Same single-stream contraction as the routed down projection above.
    #pragma unroll MOE_DOWN_UNROLL
    for (int j0 = 0; j0 < intermediate; j0 += MOE_TK) {
        float wd[MOE_TN];
        dequant_tile(down_q, down_quant, wrow + j0, lane, wd);
        float4 xv = *(const float4*)(inter + j0 + 4 * lane);
        ad[0] += wd[0] * xv.x;
        ad[0] += wd[1] * xv.y;
        ad[0] += wd[2] * xv.z;
        ad[0] += wd[3] * xv.w;
    }
    warp_reduce_tile<1>(ad);

    if (lane == 0) out[h] = ad[0];
}

// -------------------------------------------------------------------------
// 4. fp32 weighted sum of each token's top-k contributions.
// -------------------------------------------------------------------------
//
// grid: max_tokens. Ascending k, matching the reference's per-token loop.
// grid: (max_tokens, ceil(hidden / THREADS)).
//
// The second grid dimension is the whole point at one token. A block per
// token means *one block* on a 72-SM card summing eight 2,048-float vectors:
// 72 KiB of traffic in 5.2 us, which is 14 GB/s and entirely the launch's
// own latency. Splitting the row gives the same work eight blocks.
//
// The k loop stays ascending and stays inside one thread, so the summation
// order is untouched — the routed contributions of one expert are added in
// selection order exactly as the reference adds them.
__global__ void moe_reduce(
    const float* __restrict__ partial,
    const int* __restrict__ valid_tokens,
    int top_k,
    int hidden,
    float* __restrict__ out
) {
    int token = blockIdx.x;
    if (token >= *valid_tokens) return;
    int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= hidden) return;
    float acc = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        acc += partial[((long long)token * top_k + k) * hidden + h];
    }
    out[(long long)token * hidden + h] = acc;
}

// -------------------------------------------------------------------------
// 5. The shared expert, hoisted out of the routed path.
// -------------------------------------------------------------------------
//
// No routing, no sorting, no indirection, no routing weight: one expert
// applied to every token unconditionally.
//
// It is tiled over tokens for the same reason the routed path is, and the
// payoff is larger: the shared expert's stack is *one* expert, so before
// tiling a 512-token step re-read the same 2.7 MiB 512 times. grid:
// (ceil(intermediate / MOE_ROWS), ceil(max_tokens / MOE_TM)) — both from the
// geometry, neither from the live token count, which is still the device
// scalar the staging gates on.
__global__ void moe_shared_ffn(
    const unsigned char* __restrict__ gate_q, int gate_quant,
    const unsigned char* __restrict__ up_q,   int up_quant,
    const float* __restrict__ hidden_states,
    const int* __restrict__ valid_tokens,
    int hidden,
    int intermediate,
    float* __restrict__ inter
) {
    float*     xs   = xabe_shared;                              // [MOE_TM][MOE_TK]
    long long* rows = (long long*)(xs + MOE_TM * MOE_TK);       // [MOE_TM]

    int nvalid = *valid_tokens;
    int t0 = blockIdx.y * MOE_TM;
    if (t0 >= nvalid) return;

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int r = blockIdx.x * MOE_ROWS + warp;
    int live = r < intermediate;
    long long wrow = (long long)r * hidden;

    if (threadIdx.x < MOE_TM) {
        int token = t0 + threadIdx.x;
        rows[threadIdx.x] = token < nvalid ? (long long)token * hidden : -1;
    }
    __syncthreads();

    int bm = live_tile_rows(rows);
    float ag[MOE_TM];
    float au[MOE_TM];
#define MOE_SHARED_FFN_TILE(TM) tile_gemm_pair<TM>(                           \
        gate_q, gate_quant, up_q, up_quant, hidden_states, rows, xs,          \
        wrow, hidden, lane, live, ag, au)
    MOE_TILE_DISPATCH(MOE_SHARED_FFN_TILE)
#undef MOE_SHARED_FFN_TILE

    if (live && lane == 0) {
        #pragma unroll
        for (int m = 0; m < MOE_TM; ++m) {
            if (m < bm) {
                float act = ag[m] / (1.0f + expf(-ag[m]));
                inter[(long long)(t0 + m) * intermediate + r] = act * au[m];
            }
        }
    }
}

// grid: (ceil(hidden / MOE_ROWS), ceil(max_tokens / MOE_TM)).
__global__ void moe_shared_down(
    const unsigned char* __restrict__ down_q, int down_quant,
    const float* __restrict__ inter,
    const int* __restrict__ valid_tokens,
    int hidden,
    int intermediate,
    float* __restrict__ out
) {
    float*     xs   = xabe_shared;                              // [MOE_TM][MOE_TK]
    long long* rows = (long long*)(xs + MOE_TM * MOE_TK);       // [MOE_TM]

    int nvalid = *valid_tokens;
    int t0 = blockIdx.y * MOE_TM;
    if (t0 >= nvalid) return;

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int h = blockIdx.x * MOE_ROWS + warp;
    int live = h < hidden;
    long long wrow = (long long)h * intermediate;

    if (threadIdx.x < MOE_TM) {
        int token = t0 + threadIdx.x;
        rows[threadIdx.x] = token < nvalid ? (long long)token * intermediate : -1;
    }
    __syncthreads();

    int bm = live_tile_rows(rows);
    float ad[MOE_TM];
#define MOE_SHARED_DOWN_TILE(TM) tile_gemm_single<TM>(                        \
        down_q, down_quant, inter, rows, xs,                                  \
        wrow, intermediate, lane, live, ad)
    MOE_TILE_DISPATCH(MOE_SHARED_DOWN_TILE)
#undef MOE_SHARED_DOWN_TILE

    if (live && lane == 0) {
        #pragma unroll
        for (int m = 0; m < MOE_TM; ++m) {
            if (m < bm) out[(long long)(t0 + m) * hidden + h] = ad[m];
        }
    }
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
    /// The integer tensor-core path rejected the activation quantization.
    Mma(super::mma::MmaError),
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
            Self::Mma(e) => write!(f, "{e}"),
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
    /// Per-expert selection counts, handed from the dispatch kernel's
    /// counting pass to its scatter pass. Device-side only: no host ever
    /// reads it, which is what lets the two passes be separate launches
    /// without breaking graph capture.
    expert_counts: CudaSlice<i32>,
    num_tokens_post_pad: CudaSlice<i32>,
    valid_tokens: CudaSlice<i32>,
    /// What `valid_tokens` was last set to. See
    /// [`MoeKernels::set_valid_tokens`].
    valid_published: i32,
    inter: CudaSlice<f32>,
    partial: CudaSlice<f32>,
    shared_inter: CudaSlice<f32>,
    /// This step's activations quantized to int8, and one scale per 32.
    ///
    /// Quantized **once per layer**, not once per block: `grid.x` is
    /// `intermediate / 64`, so every block that shares a dispatch block would
    /// otherwise redo the same sweep, and the redundant fp32 reads would cost
    /// more traffic than the weights the kernel exists to stream.
    xq: CudaSlice<i8>,
    xq_scales: CudaSlice<f32>,
    /// The gate/up output quantized to int8 for the down projection.
    ///
    /// One row per dispatch *slot*, not per token: the down projection
    /// contracts a slot's own intermediate vector, and padding slots are
    /// staged as zero rather than read.
    iq: CudaSlice<i8>,
    iq_scales: CudaSlice<f32>,
    /// The shared expert's gate projection, held until `up` is ready to be
    /// combined with it. The fp32 kernel needs no such buffer because it
    /// computes both halves in one block and fuses the SwiGLU inline.
    shared_gate_out: CudaSlice<f32>,
    /// `silu(gate) * up`, its own buffer rather than either operand's so the
    /// SwiGLU kernel's three pointers can stay `__restrict__`.
    shared_swiglu: CudaSlice<f32>,
    /// The shared expert's SwiGLU output quantized for its down projection.
    siq: CudaSlice<i8>,
    siq_scales: CudaSlice<f32>,
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

    /// The per-`(token, k)` routed contributions, `[max_flat_pairs][hidden]`,
    /// already scaled by their routing weight.
    ///
    /// What [`MoeKernels::grouped_forward_partial`] leaves behind, for a
    /// caller that sums them itself. Slot `(t, k)` starts at
    /// `(t * experts_per_token + k) * hidden`; summing ascending in `k` is
    /// what reproduces the reference's order.
    pub fn partial(&self) -> &CudaSlice<f32> {
        &self.partial
    }

    /// Total device bytes held.
    pub fn bytes(&self) -> usize {
        (self.topk_ids.len()
            + self.sorted_token_ids.len()
            + self.expert_ids.len()
            + self.expert_counts.len()
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
/// One layer's shared expert repacked for the integer tensor cores.
///
/// About 3.5 MB per layer — three `intermediate * hidden` matrices as int8
/// plus one fp32 scale per 32, against the Q8_0 the block already holds.
/// Trivial beside the 692 MB of routed experts, which is why this one gets a
/// repack where those get shared-memory staging: a second copy of the routed
/// stacks would not fit beside the model, and a second copy of this one is
/// noise.
pub struct SharedExpertInt8 {
    gate_q: CudaSlice<i8>,
    gate_s: CudaSlice<f32>,
    up_q: CudaSlice<i8>,
    up_s: CudaSlice<f32>,
    down_q: CudaSlice<i8>,
    down_s: CudaSlice<f32>,
}

impl SharedExpertInt8 {
    /// Repack all three matrices. `elements` is `intermediate * hidden`, the
    /// same for each — `down` is the transpose of the other two, not a
    /// different size.
    pub fn repack(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        gate: QuantTensor<'_>,
        up: QuantTensor<'_>,
        down: QuantTensor<'_>,
        elements: usize,
    ) -> Result<Self, MoeError> {
        for (which, t) in [("gate", gate), ("up", up), ("down", down)] {
            if t.quant != ExpertQuant::Q8_0 {
                return Err(MoeError::UnsupportedGeometry {
                    geometry: Box::new(MoeGeometry::qwen3_6(16, 1)),
                    reason: match which {
                        "gate" => "shared gate is not Q8_0",
                        "up" => "shared up is not Q8_0",
                        _ => "shared down is not Q8_0",
                    },
                });
            }
        }
        let mma = MmaKernels::new(ctx).map_err(MoeError::Mma)?;
        // The source is already resident — `QuantTensor` carries a device
        // slice — so this repacks in place on the card rather than round
        // tripping through the host.
        let one = |src: &CudaSlice<u8>| -> Result<(CudaSlice<i8>, CudaSlice<f32>), MoeError> {
            let mut q = stream.alloc_zeros::<i8>(elements)?;
            let mut sc = stream.alloc_zeros::<f32>(elements / 32)?;
            mma.repack_q8_0(stream, src, &mut q, &mut sc, elements)
                .map_err(MoeError::Mma)?;
            Ok((q, sc))
        };
        let (gate_q, gate_s) = one(gate.bytes)?;
        let (up_q, up_s) = one(up.bytes)?;
        let (down_q, down_s) = one(down.bytes)?;
        Ok(Self {
            gate_q,
            gate_s,
            up_q,
            up_s,
            down_q,
            down_s,
        })
    }

    /// Device bytes held.
    pub fn bytes(&self) -> usize {
        self.gate_q.len()
            + self.up_q.len()
            + self.down_q.len()
            + (self.gate_s.len() + self.up_s.len() + self.down_s.len()) * size_of::<f32>()
    }
}

pub struct MoeKernels {
    route: CudaFunction,
    dispatch_t1: CudaFunction,
    route_dispatch_t1: CudaFunction,
    align_count: CudaFunction,
    align: CudaFunction,
    expert_ffn: CudaFunction,
    expert_ffn_gemv: CudaFunction,
    expert_down_gemv: CudaFunction,
    expert_ffn_mma: CudaFunction,
    expert_ffn_mma_q8: CudaFunction,
    /// Drives the activation quantization the tensor-core path consumes.
    /// `None` if the integer path is unavailable on this device.
    mma: Option<MmaKernels>,
    expert_down: CudaFunction,
    expert_down_mma: CudaFunction,
    reduce: CudaFunction,
    shared_ffn: CudaFunction,
    shared_down: CudaFunction,
    shared_ffn_gemv: CudaFunction,
    shared_down_gemv: CudaFunction,
    swiglu: CudaFunction,
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
        // Both contraction lengths are walked in `TILE_K`-wide slices, and
        // the Q6_K prologue hoists the superblock header on the strength of a
        // slice never straddling a 128-element half. A row whose length is
        // not a multiple of `TILE_K` would break that, so it is rejected
        // rather than silently handled by a slower per-element path: at
        // Qwen3.6's 2048 x 512 the condition holds with room to spare, and a
        // geometry where it does not is a design question, not a fallback.
        if !geometry.hidden.is_multiple_of(TILE_K) {
            return Err(bad("hidden must be a multiple of the 128-element tile"));
        }
        if !geometry.intermediate.is_multiple_of(TILE_K) {
            return Err(bad(
                "intermediate must be a multiple of the 128-element tile",
            ));
        }
        // The tensor-core path stages a whole dispatch block at once, so a
        // `block_size` past `MMA_M` would drop its tail. Rejected rather than
        // handled: every slot past the sixteenth would be silently ignored,
        // and the wrong answer would be finite and plausible.
        if geometry.block_size > MMA_M {
            return Err(bad(
                "block_size exceeds the 16 slots the tensor-core tile stages",
            ));
        }
        // grid.y is the dispatch-block capacity or a token tile, grid.x a
        // band of output rows; both must fit the driver's per-dimension
        // limits. x is capped at 2^31-1 but y and z at 65535, which is the
        // one that can actually bite.
        //
        // The quantity to bound is `expert_block_capacity`, which is what every
        // dispatch launch puts in grid.y. This used to bound `sorted_capacity`
        // — the *slot* count, `block_size` times larger — and so capped context
        // at about 7,168 tokens where grid.y is 2,040 of an available 65,535.
        //
        // That bound was not arbitrary, though, and relaxing it alone was not
        // enough: `MmaKernels::quantize_rows` took `rows` on grid.y, and the
        // down projection passes `sorted_capacity` as `rows`. Past 7,168 tokens
        // the launch failed with a bare `CUDA_ERROR_INVALID_VALUE`. That kernel
        // now takes rows on grid.x, where the limit is 2^31-1, so the slot
        // count no longer reaches a 65,535 axis anywhere. Nothing else needs it
        // to: inside the dispatch kernels it is a loop bound, and outside it is
        // a length for `sorted_token_ids` and `inter`, both indexed by `int`.
        if geometry.expert_block_capacity() > 65_535 {
            return Err(bad(
                "dispatch-block capacity exceeds the 65535 grid.y limit",
            ));
        }
        // grid.x takes `max_tokens` directly in the quantize and reduce
        // launches. The driver allows 2^31-1 there, but a token count that
        // large is a sizing mistake somewhere upstream rather than a workload,
        // and every `sorted_capacity`-sized allocation would be terabytes.
        if geometry.max_tokens > 65_535 {
            return Err(bad("max_tokens exceeds the 65535 grid limit"));
        }

        let ptx = compile(MOE_SRC, "moe").map_err(MoeError::Compile)?;
        let module = ctx.load_module(ptx)?;
        Ok(Self {
            route: module.load_function("moe_route")?,
            dispatch_t1: module.load_function("moe_dispatch_t1")?,
            route_dispatch_t1: module.load_function("moe_route_dispatch_t1")?,
            align_count: module.load_function("moe_align_count")?,
            align: module.load_function("moe_align_block_size")?,
            expert_ffn: module.load_function("moe_expert_ffn")?,
            expert_ffn_gemv: module.load_function("moe_expert_ffn_gemv")?,
            expert_down_gemv: module.load_function("moe_expert_down_gemv")?,
            expert_ffn_mma: module.load_function("moe_expert_ffn_mma")?,
            expert_ffn_mma_q8: module.load_function("moe_expert_ffn_mma_q8")?,
            // Compiled eagerly so a device that cannot reach the integer
            // tensor cores fails here, at construction, rather than mid-pass.
            // A failure is not fatal: the fp32 path stays available and the
            // launch below falls back to it.
            mma: MmaKernels::new(ctx).ok(),
            expert_down: module.load_function("moe_expert_down")?,
            expert_down_mma: module.load_function("moe_expert_down_mma")?,
            reduce: module.load_function("moe_reduce")?,
            shared_ffn: module.load_function("moe_shared_ffn")?,
            shared_down: module.load_function("moe_shared_down")?,
            shared_ffn_gemv: module.load_function("moe_shared_ffn_gemv")?,
            shared_down_gemv: module.load_function("moe_shared_down_gemv")?,
            swiglu: module.load_function("moe_swiglu")?,
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
            expert_counts: stream.alloc_zeros::<i32>(g.num_experts)?,
            num_tokens_post_pad: stream.alloc_zeros::<i32>(1)?,
            valid_tokens: stream.alloc_zeros::<i32>(1)?,
            valid_published: 0,
            inter: stream.alloc_zeros::<f32>(g.sorted_capacity() * g.intermediate)?,
            partial: stream.alloc_zeros::<f32>(g.max_flat_pairs() * g.hidden)?,
            shared_inter: stream.alloc_zeros::<f32>(g.max_tokens * g.intermediate)?,
            xq: stream.alloc_zeros::<i8>(g.max_tokens * g.hidden)?,
            xq_scales: stream.alloc_zeros::<f32>(g.max_tokens * g.hidden / 32)?,
            iq: stream.alloc_zeros::<i8>(g.sorted_capacity() * g.intermediate)?,
            iq_scales: stream.alloc_zeros::<f32>(g.sorted_capacity() * g.intermediate / 32)?,
            shared_gate_out: stream.alloc_zeros::<f32>(g.max_tokens * g.intermediate)?,
            shared_swiglu: stream.alloc_zeros::<f32>(g.max_tokens * g.intermediate)?,
            siq: stream.alloc_zeros::<i8>(g.max_tokens * g.intermediate)?,
            siq_scales: stream.alloc_zeros::<f32>(g.max_tokens * g.intermediate / 32)?,
        })
    }

    /// Force the grouped GEMM back onto the fp32 kernel.
    ///
    /// Exists for `tests/moe_differential.rs`, which has to gate both paths:
    /// the fp32 kernel against the host at fp32 tolerance — an invariant worth
    /// keeping tight — and the integer kernel against the same host reference
    /// at the looser bound int8 activations actually permit. Selecting the
    /// path internally would make the tight gate untestable, and loosening the
    /// one tolerance to cover both would stop it catching anything.
    ///
    /// Not reversible: the compiled function stays loaded, but the handle it
    /// needs to quantize activations is dropped.
    pub fn disable_tensor_cores(&mut self) {
        self.mma = None;
    }

    /// Whether the grouped GEMM will take the integer tensor-core path for a
    /// Q6_K gate/up pair.
    pub fn tensor_cores_enabled(&self) -> bool {
        self.mma.is_some()
    }

    /// Publish this step's token count into the device scalar the kernels
    /// gate on.
    ///
    /// This is a data write, not a shape decision: no allocation happens and
    /// no launch geometry changes, which is what keeps the sequence
    /// capturable.
    /// A repeat with the same count writes nothing. That is not only an
    /// elided copy — forty layers share one `MoeBuffers`, so a pass used to
    /// publish the same number forty times — it is what keeps the write out
    /// of a CUDA graph capture. The source is a host slice, and a copy from
    /// pageable host memory is not something a capture may contain; hoisting
    /// it to `Forward::publish_inputs` and making the repeats free is how the
    /// step below it became recordable.
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
        if buffers.valid_published == tokens as i32 {
            return Ok(());
        }
        stream.memcpy_htod(&[tokens as i32], &mut buffers.valid_tokens)?;
        buffers.valid_published = tokens as i32;
        Ok(())
    }

    /// Routing and the dispatch table, in as few launches as the shape allows.
    ///
    /// At one token that is **one** launch: both kernels run as a single block
    /// and the second reads nothing from the first but eight expert ids. Above
    /// one token the dispatch genuinely needs a grid, so it stays two calls.
    pub fn route_and_dispatch(
        &self,
        stream: &Arc<CudaStream>,
        buffers: &mut MoeBuffers,
        logits: &CudaSlice<f32>,
    ) -> Result<(), MoeError> {
        let g = self.geometry;
        if g.max_tokens != 1 {
            self.route(stream, buffers, logits)?;
            return self.build_dispatch(stream, buffers);
        }
        if g.num_experts > 32 * ROUTE_LANE_EXPERTS {
            return Err(MoeError::UnsupportedGeometry {
                geometry: Box::new(g),
                reason: "the routing warp holds 16 experts per lane, so at most \
                         512 experts",
            });
        }
        let num_experts = g.num_experts as i32;
        let top_k = g.experts_per_token as i32;
        let block_size = g.block_size as i32;
        let sorted_capacity = g.sorted_capacity() as i32;
        let expert_capacity = g.expert_block_capacity() as i32;
        // probs + one (float, int) reduction slot per thread + cumsum.
        let shared = ((g.num_experts + THREADS as usize) * size_of::<f32>()
            + (THREADS as usize + g.num_experts + 1) * size_of::<i32>())
            as u32;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: shared,
        };
        let mut builder = stream.launch_builder(&self.route_dispatch_t1);
        builder
            .arg(logits)
            .arg(&buffers.valid_tokens)
            .arg(&num_experts)
            .arg(&top_k)
            .arg(&block_size)
            .arg(&sorted_capacity)
            .arg(&expert_capacity)
            .arg(&mut buffers.topk_ids)
            .arg(&mut buffers.topk_weights)
            .arg(&mut buffers.sorted_token_ids)
            .arg(&mut buffers.expert_ids)
            .arg(&mut buffers.num_tokens_post_pad);
        // SAFETY: the union of the two kernels' own bounds, both checked in
        // their single-launch forms above and below; the shared request is
        // the sum of the two layouts, which the kernel splits at the same
        // offsets.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// Softmax + top-k over all experts, entirely on the device.
    ///
    /// `logits` is `[max_tokens][num_experts]`; only the first
    /// `valid_tokens` rows are read.
    ///
    /// [`Self::route_and_dispatch`] is what the forward path calls; this is
    /// the batch half of it, and what the differential test drives directly.
    pub fn route(
        &self,
        stream: &Arc<CudaStream>,
        buffers: &mut MoeBuffers,
        logits: &CudaSlice<f32>,
    ) -> Result<(), MoeError> {
        let g = self.geometry;
        // The top-k selection holds every expert in one warp's registers.
        if g.num_experts > 32 * ROUTE_LANE_EXPERTS {
            return Err(MoeError::UnsupportedGeometry {
                geometry: Box::new(g),
                reason: "the routing warp holds 16 experts per lane, so at most \
                         512 experts",
            });
        }
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
    ///
    /// Two launches, both at `num_experts` blocks — a fixed grid, from the
    /// geometry. The first counts each expert's selections and lays down the
    /// sentinel / INACTIVE fill; the second recomputes the shared prefix sum
    /// and scatters. The intermediate counts live in
    /// [`MoeBuffers::expert_counts`] on the device, so the pair is still one
    /// capturable sequence with no host round trip between the passes.
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

        // One token is eight flat pairs, and one block can place them all —
        // see `moe_dispatch_t1`.
        if g.max_tokens == 1 {
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: ((g.num_experts + 1) * size_of::<i32>()) as u32,
            };
            let mut builder = stream.launch_builder(&self.dispatch_t1);
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
            // SAFETY: every write is bounded by `sorted_capacity` or
            // `expert_capacity`, which are this geometry's own buffer lengths;
            // `cumsum` holds `num_experts + 1` ints and the shared request
            // above is exactly that.
            unsafe { builder.launch(cfg) }?;
            return Ok(());
        }

        let cfg = LaunchConfig {
            grid_dim: (g.num_experts as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: (THREADS as usize * size_of::<i32>()) as u32,
        };
        let mut builder = stream.launch_builder(&self.align_count);
        builder
            .arg(&buffers.topk_ids)
            .arg(&buffers.valid_tokens)
            .arg(&top_k)
            .arg(&sorted_capacity)
            .arg(&expert_capacity)
            .arg(&mut buffers.sorted_token_ids)
            .arg(&mut buffers.expert_ids)
            .arg(&mut buffers.expert_counts);
        // SAFETY: one block per expert, so `blockIdx.x` is a valid expert id;
        // the shared array is one int per thread, exactly what the count
        // reduction indexes, and the two fills are bounded by the capacities
        // passed in.
        unsafe { builder.launch(cfg) }?;

        let scatter_shared = ((g.num_experts + 1 + THREADS as usize) * size_of::<i32>()) as u32;
        let scatter_cfg = LaunchConfig {
            grid_dim: (g.num_experts as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: scatter_shared,
        };
        let mut builder = stream.launch_builder(&self.align);
        builder
            .arg(&buffers.topk_ids)
            .arg(&buffers.valid_tokens)
            .arg(&buffers.expert_counts)
            .arg(&top_k)
            .arg(&num_experts)
            .arg(&block_size)
            .arg(&sorted_capacity)
            .arg(&expert_capacity)
            .arg(&mut buffers.sorted_token_ids)
            .arg(&mut buffers.expert_ids)
            .arg(&mut buffers.num_tokens_post_pad);
        // SAFETY: one block per expert; the shared array is `num_experts + 1`
        // ints for the prefix sum plus one per thread for the placement scan,
        // and every global write is bounds-checked against the capacities
        // passed in.
        unsafe { builder.launch(scatter_cfg) }?;
        Ok(())
    }

    /// The grouped GEMM over the dispatch tables, stopping at the per-(token,
    /// k) contributions in [`MoeBuffers::partial`].
    ///
    /// `hidden_states` is `[max_tokens][hidden]`. Requires [`Self::route`] and
    /// [`Self::build_dispatch`] to have run on `buffers` for this step.
    ///
    /// Callers that want the summed `[max_tokens][hidden]` result want
    /// [`Self::grouped_forward`]. This entry point exists for the one caller
    /// that folds the sum into a kernel it was going to launch anyway — the
    /// sum is one fused multiply-add per element and the launch that performs
    /// it is 2.2 us of a 9.6 ms step, so whoever can absorb it should.
    pub fn grouped_forward_partial(
        &self,
        stream: &Arc<CudaStream>,
        buffers: &mut MoeBuffers,
        gate: QuantTensor<'_>,
        up: QuantTensor<'_>,
        down: QuantTensor<'_>,
        hidden_states: &CudaSlice<f32>,
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
        let shared = tile_shared_bytes();

        // One token is a GEMV, not a GEMM, and gets its own pair of kernels.
        // See `moe_expert_ffn_gemv` on why the tiled kernel's staging and
        // barriers are pure overhead at this shape.
        let gemv = g.max_tokens == 1;

        // The integer path needs both projections it fuses to be the *same*
        // format, and there is a kernel for each: Q6_K for blocks 0-38 and 40,
        // Q8_0 for block 39. A mixed pair has no kernel and takes the fp32
        // fallback, which no shipping file has ever produced.
        let mma_quant = match (gate.quant, up.quant) {
            (ExpertQuant::Q6K, ExpertQuant::Q6K) => Some(ExpertQuant::Q6K),
            (ExpertQuant::Q8_0, ExpertQuant::Q8_0) => Some(ExpertQuant::Q8_0),
            _ => None,
        };
        let use_mma = self.mma.is_some() && g.max_tokens >= MMA_MIN_TOKENS && mma_quant.is_some();

        if gemv {
            let cfg = LaunchConfig {
                grid_dim: (
                    (g.intermediate as u32).div_ceil(TILE_ROWS),
                    g.expert_block_capacity() as u32,
                    1,
                ),
                block_dim: (GEMM_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = stream.launch_builder(&self.expert_ffn_gemv);
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
            // SAFETY: the same bounds as the tiled launch below, minus the
            // shared memory it does not use. The kernel reads slot 0 of its
            // dispatch block only, which is in range because
            // `sorted_token_ids` holds `expert_block_capacity * block_size`.
            unsafe { builder.launch(cfg) }?;
        } else if use_mma {
            let mma = self.mma.as_ref().expect("checked above");
            let rows = g.max_tokens;
            mma.quantize_rows(
                stream,
                hidden_states,
                &mut buffers.xq,
                &mut buffers.xq_scales,
                rows,
                g.hidden,
            )
            .map_err(MoeError::Mma)?;

            let q8 = mma_quant == Some(ExpertQuant::Q8_0);
            let cfg = LaunchConfig {
                grid_dim: (
                    (g.intermediate as u32).div_ceil(MMA_ROWS),
                    g.expert_block_capacity() as u32,
                    1,
                ),
                block_dim: (MMA_WARPS * 32, 1, 1),
                shared_mem_bytes: if q8 {
                    mma_ffn_q8_shared_bytes()
                } else {
                    mma_shared_bytes()
                },
            };
            let f = if q8 {
                &self.expert_ffn_mma_q8
            } else {
                &self.expert_ffn_mma
            };
            let mut builder = stream.launch_builder(f);
            builder
                .arg(gate.bytes)
                .arg(up.bytes)
                .arg(&buffers.xq)
                .arg(&buffers.xq_scales)
                .arg(&buffers.sorted_token_ids)
                .arg(&buffers.expert_ids)
                .arg(&buffers.valid_tokens)
                .arg(&top_k)
                .arg(&block_size)
                .arg(&hidden)
                .arg(&intermediate)
                .arg(&mut buffers.inter);
            // SAFETY: as for the fp32 launch below, plus: `xq` is
            // `max_tokens * hidden` int8 and is indexed by `row * hidden + k`
            // with `row < max_tokens` (the dispatch tables cannot name a
            // larger token) and `k < hidden`; `xq_scales` is that divided by
            // 32 and indexed by `row * (hidden/32) + k/32`.
            unsafe { builder.launch(cfg) }?;
        } else {
            let ffn_cfg = LaunchConfig {
                grid_dim: (
                    (g.intermediate as u32).div_ceil(TILE_ROWS),
                    g.expert_block_capacity() as u32,
                    1,
                ),
                block_dim: (GEMM_THREADS, 1, 1),
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
            // SAFETY: grid.y is the dispatch-block capacity, exactly what
            // `expert_ids` holds and `block_size` times fewer than what
            // `sorted_token_ids` holds; `inter` is `sorted_capacity *
            // intermediate` floats, the range `(blk * block_size + m, r)`
            // covers. Shared memory covers the activation tile plus one slot id
            // per tile row. Weight indexing is bounded by the element-count
            // check above.
            unsafe { builder.launch(ffn_cfg) }?;
        }

        // Every valid flat id is written exactly once by the down kernel, so
        // this zeroing is belt-and-braces — but a dispatch bug that dropped a
        // slot would otherwise read a previous step's contribution and
        // produce a plausible wrong answer instead of an obviously wrong one.
        stream.memset_zeros(&mut buffers.partial)?;

        if gemv {
            let cfg = LaunchConfig {
                grid_dim: (
                    (g.hidden as u32).div_ceil(TILE_ROWS),
                    g.expert_block_capacity() as u32,
                    1,
                ),
                block_dim: (GEMM_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = stream.launch_builder(&self.expert_down_gemv);
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
            // SAFETY: as above.
            unsafe { builder.launch(cfg) }?;
        }

        // The down projection takes the integer path on the same terms, but
        // gated on its *own* format: `ffn_down_exps` is Q8_0 where gate/up are
        // Q6_K, so the two halves of the block can legitimately disagree about
        // which arithmetic they use.
        let down_mma = !gemv
            && self.mma.is_some()
            && g.max_tokens >= MMA_MIN_TOKENS
            && down.quant == ExpertQuant::Q8_0;

        if gemv {
            // Launched above, alongside its gate/up half. Falls through to the
            // reduction the other two paths also reach.
        } else if down_mma {
            let mma = self.mma.as_ref().expect("checked above");
            // One row per dispatch slot. Padding slots hold whatever the last
            // step left in `inter` and are quantized along with the rest; the
            // kernel stages them as zero rather than reading them, so their
            // contents never reach an accumulator.
            mma.quantize_rows(
                stream,
                &buffers.inter,
                &mut buffers.iq,
                &mut buffers.iq_scales,
                g.sorted_capacity(),
                g.intermediate,
            )
            .map_err(MoeError::Mma)?;

            let cfg = LaunchConfig {
                grid_dim: (
                    (g.hidden as u32).div_ceil(MMA_ROWS),
                    g.expert_block_capacity() as u32,
                    1,
                ),
                block_dim: (MMA_WARPS * 32, 1, 1),
                shared_mem_bytes: mma_down_shared_bytes(),
            };
            let mut builder = stream.launch_builder(&self.expert_down_mma);
            builder
                .arg(down.bytes)
                .arg(&buffers.iq)
                .arg(&buffers.iq_scales)
                .arg(&buffers.topk_weights)
                .arg(&buffers.sorted_token_ids)
                .arg(&buffers.expert_ids)
                .arg(&buffers.valid_tokens)
                .arg(&top_k)
                .arg(&block_size)
                .arg(&hidden)
                .arg(&intermediate)
                .arg(&mut buffers.partial);
            // SAFETY: as for the fp32 launch below, plus: `iq` is
            // `sorted_capacity * intermediate` int8 and is indexed by
            // `slot * intermediate + k` with `slot < sorted_capacity` by
            // construction of the dispatch tables.
            unsafe { builder.launch(cfg) }?;
        } else {
            let down_cfg = LaunchConfig {
                grid_dim: (
                    (g.hidden as u32).div_ceil(TILE_ROWS),
                    g.expert_block_capacity() as u32,
                    1,
                ),
                block_dim: (GEMM_THREADS, 1, 1),
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
        }

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
        self.grouped_forward_partial(stream, buffers, gate, up, down, hidden_states)?;
        let g = self.geometry;
        let top_k = g.experts_per_token as i32;
        let hidden = g.hidden as i32;
        let reduce_cfg = LaunchConfig {
            grid_dim: (g.max_tokens as u32, (g.hidden as u32).div_ceil(THREADS), 1),
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

    /// The shared expert on the integer tensor cores.
    ///
    /// Three `q8_0_proj_split` calls with a SwiGLU between the second and the
    /// third, against the fp32 kernel's two fused launches. The fusion is what
    /// is given up; the projection kernel it buys is the one already measured
    /// at 27 TOP/s on the dense projections, and the extra elementwise pass
    /// over `[max_tokens][intermediate]` is a rounding error beside it.
    ///
    /// Falls back to [`Self::shared_expert`] — the caller's job — when the
    /// weights are not Q8_0 or the batch is too short to amortize the
    /// quantization sweeps.
    pub fn shared_expert_mma(
        &self,
        stream: &Arc<CudaStream>,
        buffers: &mut MoeBuffers,
        w: &SharedExpertInt8,
        hidden_states: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), MoeError> {
        let g = self.geometry;
        let mma = self.mma.as_ref().ok_or(MoeError::UnsupportedGeometry {
            geometry: Box::new(g),
            reason: "the integer tensor cores are unavailable on this device",
        })?;

        // `normed` again, not the routed path's leftovers: the two are the
        // same buffer today, but depending on that would make this correct by
        // coincidence. The sweep is microseconds.
        mma.quantize_rows(
            stream,
            hidden_states,
            &mut buffers.xq,
            &mut buffers.xq_scales,
            g.max_tokens,
            g.hidden,
        )
        .map_err(MoeError::Mma)?;

        for (wq, ws, dst) in [(&w.gate_q, &w.gate_s, 0usize), (&w.up_q, &w.up_s, 1usize)] {
            let target = if dst == 0 {
                &mut buffers.shared_gate_out
            } else {
                &mut buffers.shared_inter
            };
            mma.q8_0_proj_split(
                stream,
                wq,
                ws,
                &buffers.xq,
                &buffers.xq_scales,
                target,
                g.hidden,
                g.intermediate,
                g.max_tokens,
            )
            .map_err(MoeError::Mma)?;
        }

        let elems = g.max_tokens * g.intermediate;
        let n = elems as i64;
        let cfg = LaunchConfig {
            grid_dim: ((elems as u32).div_ceil(THREADS), 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&self.swiglu);
        builder
            .arg(&buffers.shared_gate_out)
            .arg(&buffers.shared_inter)
            .arg(&n)
            .arg(&mut buffers.shared_swiglu);
        // SAFETY: every buffer is `max_tokens * intermediate` floats and the
        // kernel bounds-checks its flat index against `n`.
        unsafe { builder.launch(cfg) }?;

        mma.quantize_rows(
            stream,
            &buffers.shared_swiglu,
            &mut buffers.siq,
            &mut buffers.siq_scales,
            g.max_tokens,
            g.intermediate,
        )
        .map_err(MoeError::Mma)?;

        mma.q8_0_proj_split(
            stream,
            &w.down_q,
            &w.down_s,
            &buffers.siq,
            &buffers.siq_scales,
            out,
            g.intermediate,
            g.hidden,
            g.max_tokens,
        )
        .map_err(MoeError::Mma)?;
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
        let shared = tile_shared_bytes();
        let token_tiles = (g.max_tokens as u32).div_ceil(TILE_M as u32);

        // One token is a GEMV; see `moe_shared_ffn_gemv`.
        if g.max_tokens == 1 {
            // One block per output row, its warps splitting the contraction.
            // See `moe_shared_ffn_gemv` for why the row-per-warp shape was
            // the wrong one at this geometry.
            if !g.hidden.is_multiple_of(SHARED_WARPS as usize * TILE_K) {
                return Err(MoeError::UnsupportedGeometry {
                    geometry: Box::new(g),
                    reason: "the one-token shared expert splits `hidden` over \
                             8 warps in 128-element tiles, so it must be a \
                             multiple of 1024",
                });
            }
            let ffn_cfg = LaunchConfig {
                grid_dim: (g.intermediate as u32, 1, 1),
                block_dim: (SHARED_WARPS * 32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = stream.launch_builder(&self.shared_ffn_gemv);
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
            // SAFETY: `shared_inter` is `max_tokens * intermediate` floats and
            // this writes its first `intermediate`; the weight bounds are the
            // element-count checks above.
            unsafe { builder.launch(ffn_cfg) }?;

            let down_cfg = LaunchConfig {
                grid_dim: ((g.hidden as u32).div_ceil(TILE_ROWS), 1, 1),
                block_dim: (GEMM_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = stream.launch_builder(&self.shared_down_gemv);
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
            return Ok(());
        }

        let ffn_cfg = LaunchConfig {
            grid_dim: ((g.intermediate as u32).div_ceil(TILE_ROWS), token_tiles, 1),
            block_dim: (GEMM_THREADS, 1, 1),
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
            grid_dim: ((g.hidden as u32).div_ceil(TILE_ROWS), token_tiles, 1),
            block_dim: (GEMM_THREADS, 1, 1),
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

/// Dynamic shared memory one grouped-GEMM block needs.
///
/// The staged activation tile, then one source-row offset per tile row, then
/// one flat slot id per tile row (the routed down projection needs it to
/// address `partial`). 8 KiB and change at Qwen3.6's shape, so several blocks
/// fit an sm_75 SM's 64 KiB and the limit on occupancy is registers, not
/// this.
const fn tile_shared_bytes() -> u32 {
    (TILE_M * TILE_K * size_of::<f32>() + TILE_M * size_of::<i64>() + TILE_M * size_of::<i32>())
        as u32
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
    fn tile_shape_is_mirrored_in_rust() {
        // The kernel's `#define`s decide the shared-memory layout and the
        // per-lane unroll; the Rust constants decide the grid, the block and
        // the `shared_mem_bytes` handed to the driver. If the two drift, the
        // symptom is an out-of-bounds shared read, not a compile error.
        for (name, value) in [
            ("MOE_TM", TILE_M),
            ("MOE_TK", TILE_K),
            // One value per lane per Q6_K group: 128 elements over 32 lanes.
            ("MOE_TN", TILE_K / 32),
            ("MOE_ROWS", TILE_ROWS as usize),
            // The routing warp's per-lane register array. Too small and the
            // top-k silently ignores the tail of the expert list; the launch
            // path rejects a geometry above it, and this keeps the two ends
            // of that check the same number.
            ("MOE_ROUTE_LANE_EXPERTS", ROUTE_LANE_EXPERTS),
            // The tensor-core tile. `MOE_MMA_WARPS` sets the block width and
            // the rows one block covers; `MOE_MMA_M` sets the staged
            // activation tile and is the bound the launch path checks
            // `block_size` against. Both were retuned once already, and both
            // read shared memory the Rust side sized.
            ("MOE_MMA_WARPS", MMA_WARPS as usize),
            ("MOE_MMA_M", MMA_M),
            // Warps splitting one row in the one-token shared expert.
            ("MOE_SHARED_WARPS", SHARED_WARPS as usize),
        ] {
            // The source aligns its values into a column, so the separator is
            // one or more spaces rather than exactly one.
            let found = MOE_SRC
                .split(&format!("#define {name} "))
                .skip(1)
                .any(|rest| rest.trim_start().starts_with(&format!("{value}\n")));
            assert!(
                found,
                "{name} is {value} in Rust but not in the kernel source",
            );
        }
        assert_eq!(GEMM_THREADS, TILE_ROWS * 32, "one warp per output row");
        // A tile is one half of a 256-element Q6_K superblock, which is what
        // makes the header read once per tile instead of once per element.
        assert_eq!(TILE_K * 2, 256, "a tile must be one Q6_K superblock half");
        assert_eq!(
            tile_shared_bytes(),
            16 * 128 * 4 + 16 * 8 + 16 * 4,
            "8 KiB of activation tile plus the row offsets and slot ids",
        );
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
    fn the_grid_guard_bounds_the_block_count_and_not_the_slot_count() {
        // These differ by `block_size`, and guarding the wrong one cost this
        // model a factor of `block_size` in supported context. At 32,768
        // tokens and 16 slots to a block the dispatch needs 16,624 blocks of
        // an available 65,535 — comfortable — while the slot count is 265,984,
        // which the old guard compared against 65,535 and rejected.
        let g = MoeGeometry::qwen3_6(16, 32_768);
        assert_eq!(g.sorted_capacity(), 265_984);
        assert_eq!(g.expert_block_capacity(), 16_624);
        assert!(
            g.expert_block_capacity() <= 65_535,
            "32K tokens must be inside the grid.y limit, not outside it",
        );
        assert!(
            g.sorted_capacity() > 65_535,
            "this test is only meaningful while the two quantities disagree",
        );
    }

    #[test]
    fn every_grid_dimension_is_a_function_of_the_geometry_alone() {
        // The other half of AGENTS.md rule 5: the *buffers* not being sized by
        // a per-step value is checked above, and this is the launch shapes.
        // Two geometries differing only in `max_tokens` must still each
        // produce one fixed grid, and the tiled GEMM's grid must not depend on
        // `num_tokens_post_pad` — that value exists only in device memory, so
        // anything reading it would have to synchronize and could not be
        // captured.
        for max_tokens in [1usize, 19, 128, 512] {
            let g = MoeGeometry::qwen3_6(16, max_tokens);
            // Routed GEMM: (row band, dispatch-block capacity).
            let routed_y = g.expert_block_capacity() as u32;
            assert_eq!(routed_y, (g.sorted_capacity() / g.block_size) as u32);
            // Shared expert: (row band, token-tile count).
            let shared_y = (g.max_tokens as u32).div_ceil(TILE_M as u32);
            assert_eq!(shared_y as usize, g.max_tokens.div_ceil(TILE_M));
            // Both grid.y values must fit the driver's 65535 limit for every
            // geometry `MoeKernels::new` accepts.
            assert!(routed_y <= 65_535 && shared_y <= 65_535);
            // grid.x is a band of output rows, and both contraction lengths
            // are whole tiles.
            assert_eq!((g.intermediate as u32).div_ceil(TILE_ROWS), 64);
            assert_eq!((g.hidden as u32).div_ceil(TILE_ROWS), 256);
            assert!(g.hidden.is_multiple_of(TILE_K) && g.intermediate.is_multiple_of(TILE_K));
        }
    }

    #[test]
    fn routing_selection_carries_the_index_so_ties_break_low() {
        // `route_token` breaks ties on equal probability by lower expert
        // index. Without the index in the reduction the winner is whichever
        // lane got there first, which is not even stable across runs.
        //
        // Both halves are asserted: the per-lane scan over the experts it
        // owns, and the cross-lane butterfly that merges the lanes. The
        // second is what makes the selection independent of the reduction's
        // shape, which is the property the kernel's own comment rests on.
        assert!(MOE_SRC.contains("if (p[i] > bv || (p[i] == bv && (bi < 0 || e < bi)))"));
        assert!(MOE_SRC.contains("if (cv > bv || (cv == bv && ci >= 0 && (bi < 0 || ci < bi)))"));
        // And the mask that stops a selected expert being picked again must
        // stay a predicated sweep, not a dynamic index: `p` is a register
        // array, and indexing it by the winner spills it to local memory.
        assert!(MOE_SRC.contains("if (bi >= 0 && lane + 32 * i == bi) p[i] = -1.0f;"));
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
        assert_eq!(ExpertQuant::Q6K.file_block_bytes(), 210);
        assert_eq!(ExpertQuant::Q6K.block_bytes(), 224);
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
