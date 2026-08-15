//! RMSNorm, partial rotary embedding, SwiGLU, and the GDN short convolution.
//!
//! The glue between the three big kernels. Individually none of these is
//! interesting; collectively they are every operation a forward pass needs
//! that [`super::gdn`], [`super::gdn_chunked`], [`super::attention`] and
//! [`super::moe`] do not already cover, so G006 cannot exist without them.
//!
//! References: [`xabe_kernels::norm::rms_norm`],
//! [`xabe_kernels::rope::apply_rope`], [`xabe_kernels::norm::swiglu`], and
//! [`xabe_kernels::conv::causal_depthwise_conv1d`].
//!
//! ## One module, one compile, geometry per launch
//!
//! [`super::gdn`] and [`super::gdn_chunked`] fix their geometry at
//! construction because the model fixes it. These four cannot: RMSNorm alone
//! runs at **three different widths** in one forward pass —
//!
//! - hidden 2048, for each layer's input norm and post-mixer norm and for the
//!   final output norm;
//! - head_dim 256, for the attention layers' per-head `attn_q_norm` /
//!   `attn_k_norm` (`ssm`-free layers in `src/models/qwen35moe.cpp` create
//!   both at `{ n_embd_head_k }`);
//! - head_dim 128, for the GDN output norm (`ssm_norm`, created at
//!   `{ head_v_dim }`).
//!
//! so the width is a launch argument and is validated per launch. The four
//! kernels share one NVRTC module because they share nothing else and five
//! separate compiles would be five separate driver round-trips at startup.
//!
//! ## What is exact and what is not
//!
//! - **The rotary tail is bit-exact.** Dimensions `[rope_dim, head_dim)` are
//!   copied, not computed, so they come back byte-identical and the
//!   differential test gates them with `Tolerance::exact()`. This is the
//!   documented invariant from `xabe_kernels::rope`, not a nicety: partial
//!   rotary is easy to accidentally rotate, truncate, or reorder past its
//!   boundary, and a tolerance would pass all three.
//! - **The convolution is bit-exact.** Four taps accumulated in ascending
//!   order is the reference's exact operand sequence, so the only thing that
//!   could break equality is the compiler contracting the multiply and the
//!   add into an FMA — which rounds once where the reference rounds twice.
//!   The kernel therefore spells the accumulation `__fadd_rn(acc,
//!   __fmul_rn(x, w))`, which nvcc may not contract, and the gate is exact
//!   equality. Exactness is worth the two intrinsics here for the same reason
//!   it was in [`super::dequant`]: a tolerance can hide a reversed tap order
//!   on smooth input, and exact equality cannot.
//! - **RMSNorm and SwiGLU are not exact**, and cannot be. RMSNorm reduces
//!   2048 squares in a warp-shuffle tree where the reference sums them
//!   sequentially, and fp32 addition is not associative; SwiGLU's `expf`
//!   agrees with `f32::exp` to about an ulp. Both are unbiased and bounded,
//!   so both are gated on a measured tolerance.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};

use super::compile;

/// Widest convolution the kernel will service.
///
/// The state-update kernel stages `conv_kernel - 1` window values in a
/// per-thread register array before writing any of them, which is what makes
/// updating the cache in place safe. That array needs a compile-time bound.
///
/// Qwen3.6 needs 4 (`qwen35moe.ssm.conv_kernel`). 8 is headroom, not a
/// surveyed requirement: llama.cpp reads `ssm_d_conv` from the file rather
/// than hardcoding it per architecture, so nothing here can claim what other
/// models in this family use. The cost of the headroom is four unused
/// registers per thread in one launch, and anything wider is rejected at the
/// launch path rather than overrunning the array.
pub const MAX_CONV_KERNEL: usize = 8;

/// Largest block the RMSNorm and rotary kernels will launch.
///
/// One thread per head element for the rotary kernel, and the upper bound on
/// the strided reduction for RMSNorm.
const MAX_BLOCK: usize = 1024;

const LAYER_OPS_SRC: &str = r#"
#define MAX_CONV_KERNEL 8

extern "C" {

// Sum across a block of up to 1024 threads. The same reduction as
// `gdn.rs`/`gdn_chunked.rs`, verbatim and for the same reason: warp shuffles
// then one pass through shared memory. The tree order differs from the
// reference's sequential sum, which is the entire source of disagreement
// between rms_norm_rows and xabe_kernels::norm::rms_norm.
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

// y = x / sqrt(mean(x^2) + eps) * weight, one row at a time.
//
// grid: one block per row. block: a whole number of warps, at most 1024;
// rows wider than the block are covered by a strided loop, which is what lets
// one kernel serve hidden 2048 and head_dim 128 without recompiling.
//
// `weight` is a single row of `width` floats shared by every row of `x`. That
// is exactly the shape of both uses: the hidden-size norms have one weight
// vector per layer applied to every token, and the per-head q/k and GDN
// output norms have one weight vector per layer applied to every (token,
// head) pair.
//
// The arithmetic transcribes the reference operand for operand: sum of
// squares, divide by width, add eps, one 1/sqrtf, then `xi * inv_rms * wi` in
// that order. rsqrtf would be faster and is not IEEE-exact; matching the
// reference's division keeps the disagreement attributable to the reduction
// order alone, as in the two GDN kernels.
__global__ void rms_norm_rows(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int width,
    float eps
) {
    extern __shared__ float scratch[];
    long long base = (long long)blockIdx.x * width;

    float partial = 0.0f;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        float v = x[base + j];
        partial += v * v;
    }
    float sum_sq = block_reduce_sum(partial, scratch);

    float inv_rms = 1.0f / sqrtf(sum_sq / (float)width + eps);

    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        out[base + j] = x[base + j] * inv_rms * weight[j];
    }
}

// Partial rotary embedding, NEOX (split-half) pairing: dimension i pairs with
// i + rope_dim/2, and dimensions [rope_dim, head_dim) pass through untouched.
//
// grid: (heads, tokens). block: head_dim threads, one per head element.
//
// **The tail is copied, not computed.** `out[base+j] = x[base+j]` for
// j >= rope_dim is a load and a store with no arithmetic between them, so the
// bits are preserved exactly and the differential test can gate that span
// with Tolerance::exact(). Any formulation that instead multiplied the tail
// by a cos of angle 0 would be "correct" to within an ulp and would fail that
// gate, which is the point of having it.
//
// Angles are computed in double precision because the reference does
// (`f64::from(theta_base).powf(...)`, `pos * freq`, then `as f32`). Turing
// runs fp64 at 1/32 rate, which is irrelevant here: this kernel is two loads
// and a store per thread and is bandwidth-bound, and computing the angle in
// fp32 would introduce a second disagreement with the reference on top of the
// one the tail gate exists to rule out.
//
// Each thread computes its own sin/cos rather than one thread per pair
// computing both, so that no thread writes two elements and the pass-through
// branch stays a plain copy.
__global__ void rope_partial(
    const float* __restrict__ x,
    const unsigned int* __restrict__ positions,
    float* __restrict__ out,
    int heads,
    int head_dim,
    int rope_dim,
    float theta_base
) {
    int h = blockIdx.x;
    int t = blockIdx.y;
    int j = threadIdx.x;
    long long base = ((long long)t * heads + h) * head_dim;

    if (j >= rope_dim) {
        out[base + j] = x[base + j];
        return;
    }

    int half = rope_dim >> 1;
    int i = (j < half) ? j : (j - half);

    double freq = pow((double)theta_base, -2.0 * (double)i / (double)rope_dim);
    double angle = (double)positions[t] * freq;
    float sin_a = (float)sin(angle);
    float cos_a = (float)cos(angle);

    float x0 = x[base + i];
    float x1 = x[base + i + half];
    out[base + j] = (j < half) ? (x0 * cos_a - x1 * sin_a)
                               : (x0 * sin_a + x1 * cos_a);
}

// out = silu(gate) * up, elementwise.
//
// grid-stride over a fixed grid so the launch shape does not depend on a
// host-side length, which is what keeps it capturable in a CUDA graph
// (AGENTS.md rule 5).
//
// `expf`, not `__expf`: the reference is `x / (1.0 + (-x).exp())` and the
// fast intrinsic is good to ~2 ulp of the *result*, not of the exponent,
// which on the negative tail is a relative error the tolerance would have to
// be widened for. This is not a bottleneck.
__global__ void swiglu_mul(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    long long n
) {
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

// Causal depthwise convolution over the fused GDN q/k/v stream.
//
// out[t][ch] = sum_{i<K} w[ch][i] * xw(t - (K-1) + i, ch)
//
// where xw reads this batch's tokens for non-negative positions and the
// carried cache for negative ones. Tap K-1 multiplies the *current* token and
// tap 0 the oldest, which is `ggml_compute_forward_ssm_conv_f32`'s window
// order (`ggml/src/ggml-cpu/ops.cpp`) and is the thing a reversed loop gets
// silently backwards.
//
// grid: (ceil(channels / 256), tokens). block: 256 threads over channels, so
// consecutive threads read consecutive channels of one token — the layout is
// [seq_len][channels] with channels contiguous, so every load coalesces.
//
// __fmul_rn/__fadd_rn rather than `acc += x * w`: see the module docs. The
// point is to deny nvcc the FMA contraction so this kernel is bit-identical
// to the scalar reference rather than merely close to it.
__global__ void conv1d_causal_depthwise(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ state,
    float* __restrict__ out,
    int channels,
    int conv_kernel
) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;
    int t = blockIdx.y;
    int carry = conv_kernel - 1;

    float acc = 0.0f;
    for (int i = 0; i < conv_kernel; ++i) {
        int u = t - carry + i;
        float xv = (u >= 0)
            ? x[(long long)u * channels + ch]
            : state[(long long)ch * carry + (u + carry)];
        acc = __fadd_rn(acc, __fmul_rn(xv, weight[(long long)ch * conv_kernel + i]));
    }
    out[(long long)t * channels + ch] = acc;
}

// Advance the convolution cache to the last conv_kernel-1 inputs.
//
// grid-stride over channels, one thread per channel. Every window value is
// staged into registers before anything is written, so the update is safe
// in place: a thread owns its channel's whole cache row and no other thread
// touches it. That matters because a batch shorter than conv_kernel-1 — every
// decode step — reads entries of the old cache that it also overwrites.
// llama.cpp gets the same effect structurally, by taking the new state as a
// view into a freshly concatenated buffer
// (`llm_build_delta_net_base::build_conv_state`, `s_idx = conv_input->ne[0] -
// conv_states->ne[0]`); we have no such buffer and must not need one.
__global__ void conv1d_update_state(
    const float* __restrict__ x,
    float* __restrict__ state,
    int seq_len,
    int channels,
    int conv_kernel
) {
    int carry = conv_kernel - 1;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long ch = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         ch < channels;
         ch += stride) {
        float staged[MAX_CONV_KERNEL];
        for (int j = 0; j < carry; ++j) {
            int u = seq_len - carry + j;
            staged[j] = (u >= 0)
                ? x[(long long)u * channels + ch]
                : state[ch * carry + (u + carry)];
        }
        for (int j = 0; j < carry; ++j) {
            state[ch * carry + j] = staged[j];
        }
    }
}

}
"#;

/// Something went wrong compiling or launching a layer-op kernel.
#[derive(Debug)]
pub enum LayerOpsError {
    /// NVRTC rejected the source, or the module failed to load.
    Compile(String),
    /// The driver failed.
    Driver(DriverError),
    /// A row width the kernel cannot service.
    UnsupportedWidth { width: usize },
    /// A head geometry the rotary kernel cannot service.
    UnsupportedRotaryShape { head_dim: usize, rope_dim: usize },
    /// A convolution width the kernel cannot service.
    UnsupportedConvKernel { conv_kernel: usize },
    /// A buffer is not the length the declared geometry requires.
    ShapeMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for LayerOpsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compile(m) => write!(f, "kernel compilation failed: {m}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::UnsupportedWidth { width } => {
                write!(f, "row width {width} must be non-zero")
            }
            Self::UnsupportedRotaryShape { head_dim, rope_dim } => write!(
                f,
                "rope_dim {rope_dim} must be even and at most head_dim {head_dim}, which must \
                 itself be non-zero and at most 1024 (it is the block width)",
            ),
            Self::UnsupportedConvKernel { conv_kernel } => write!(
                f,
                "conv_kernel {conv_kernel} must be in 1..={MAX_CONV_KERNEL}",
            ),
            Self::ShapeMismatch {
                what,
                expected,
                got,
            } => write!(f, "{what} must hold {expected} floats, holds {got}"),
        }
    }
}

impl std::error::Error for LayerOpsError {}

impl From<DriverError> for LayerOpsError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

/// The compiled layer-op kernels.
pub struct LayerOpsKernels {
    rms_norm: CudaFunction,
    rope: CudaFunction,
    swiglu: CudaFunction,
    conv1d: CudaFunction,
    conv1d_state: CudaFunction,
}

impl LayerOpsKernels {
    /// Compile all four operations into one module.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, LayerOpsError> {
        let ptx = compile(LAYER_OPS_SRC, "layer_ops").map_err(LayerOpsError::Compile)?;
        let module = ctx.load_module(ptx)?;
        Ok(Self {
            rms_norm: module.load_function("rms_norm_rows")?,
            rope: module.load_function("rope_partial")?,
            swiglu: module.load_function("swiglu_mul")?,
            conv1d: module.load_function("conv1d_causal_depthwise")?,
            conv1d_state: module.load_function("conv1d_update_state")?,
        })
    }

    /// RMSNorm over `rows` independent rows of `width` elements each.
    ///
    /// `x` and `out` are `[rows][width]`; `weight` is one `width`-long vector
    /// shared by every row. Both hidden-size norms (`rows = tokens`,
    /// `width = 2048`) and per-head norms (`rows = tokens * heads`,
    /// `width = head_dim`) are this one call.
    ///
    /// `out` may alias `x`: every thread reads `x[base + j]` and writes
    /// `out[base + j]` for the same `j`, after the reduction's final
    /// `__syncthreads()`, so no thread can read an element another has
    /// already overwritten.
    ///
    /// Eight plain arguments rather than a config struct, for the same reason
    /// [`super::gdn_chunked::GdnChunkedKernels::prefill`] takes ten: every one
    /// is a distinct per-call tensor or shape, not related configuration that
    /// would be set once and reused.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        width: usize,
        eps: f32,
    ) -> Result<(), LayerOpsError> {
        if width == 0 {
            return Err(LayerOpsError::UnsupportedWidth { width });
        }
        check_len("rms_norm x", rows * width, x.len())?;
        check_len("rms_norm weight", width, weight.len())?;
        check_len("rms_norm out", rows * width, out.len())?;
        if rows == 0 {
            return Ok(());
        }

        let block = block_for_width(width);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block as u32, 1, 1),
            // One float per warp, which is the most `block_reduce_sum` stores.
            shared_mem_bytes: (block.div_ceil(32) * size_of::<f32>()) as u32,
        };
        let width_i32 = width as i32;
        let mut builder = stream.launch_builder(&self.rms_norm);
        builder
            .arg(x)
            .arg(weight)
            .arg(out)
            .arg(&width_i32)
            .arg(&eps);
        // SAFETY: one block per row over buffers of `rows * width` floats,
        // with the in-row loop bounded by `width`, so `blockIdx.x * width + j`
        // is in range for every thread. `weight` is indexed by `j < width` and
        // holds `width` floats. Shared memory covers one float per warp, which
        // is all the reduction writes.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// Partial rotary embedding over `[seq_len][heads][head_dim]`.
    ///
    /// `positions` holds one absolute position per token. Dimensions
    /// `[rope_dim, head_dim)` are copied through bit-exactly.
    ///
    /// Unlike [`Self::rms_norm`], **`out` must not alias `x`**: element `j`
    /// of a rotated pair is written from elements `i` and `i + half`, so an
    /// in-place kernel would race. Enforced by the borrow checker taking `x`
    /// shared and `out` exclusive.
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        out: &mut CudaSlice<f32>,
        seq_len: usize,
        heads: usize,
        head_dim: usize,
        rope_dim: usize,
        theta_base: f32,
    ) -> Result<(), LayerOpsError> {
        if head_dim == 0
            || head_dim > MAX_BLOCK
            || !rope_dim.is_multiple_of(2)
            || rope_dim > head_dim
        {
            return Err(LayerOpsError::UnsupportedRotaryShape { head_dim, rope_dim });
        }
        let n = seq_len * heads * head_dim;
        check_len("rope x", n, x.len())?;
        check_len("rope out", n, out.len())?;
        check_len("rope positions", seq_len, positions.len())?;
        if seq_len == 0 || heads == 0 {
            return Ok(());
        }

        let cfg = LaunchConfig {
            grid_dim: (heads as u32, seq_len as u32, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let heads_i32 = heads as i32;
        let head_dim_i32 = head_dim as i32;
        let rope_dim_i32 = rope_dim as i32;
        let mut builder = stream.launch_builder(&self.rope);
        builder
            .arg(x)
            .arg(positions)
            .arg(out)
            .arg(&heads_i32)
            .arg(&head_dim_i32)
            .arg(&rope_dim_i32)
            .arg(&theta_base);
        // SAFETY: the grid is (heads, seq_len) and the block is head_dim
        // threads, so `(t * heads + h) * head_dim + j` covers exactly the
        // `seq_len * heads * head_dim` floats both buffers hold. The paired
        // reads at `i` and `i + half` stay below `rope_dim <= head_dim`.
        // `positions` is indexed by `t < seq_len` and holds `seq_len` entries.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// `out = silu(gate) * up` over `n` elements.
    ///
    /// The standalone form. [`super::moe`]'s grouped GEMM fuses the same
    /// activation into its epilogue; this exists for the paths that do not go
    /// through it and as the thing the fused version is checked against.
    pub fn swiglu(
        &self,
        stream: &Arc<CudaStream>,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), LayerOpsError> {
        check_len("swiglu gate", n, gate.len())?;
        check_len("swiglu up", n, up.len())?;
        check_len("swiglu out", n, out.len())?;
        if n == 0 {
            return Ok(());
        }

        const BLOCK: usize = 256;
        // A fixed grid, not one derived from `n`: grid-stride keeps the launch
        // shape independent of a host-side length so the launch can be
        // captured in a CUDA graph and replayed at a different `n`.
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(BLOCK).min(1024) as u32, 1, 1),
            block_dim: (BLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i64 = n as i64;
        let mut builder = stream.launch_builder(&self.swiglu);
        builder.arg(gate).arg(up).arg(out).arg(&n_i64);
        // SAFETY: the grid-stride loop is bounded by `n`, which is the length
        // checked above for all three buffers.
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }

    /// The GDN short causal depthwise convolution, advancing `state`.
    ///
    /// `x` and `out` are `[seq_len][channels]` with channels contiguous;
    /// `weight` is `[channels][conv_kernel]`, the `ssm_conv1d.weight` layout
    /// verbatim; `state` is `[channels][conv_kernel - 1]` oldest-first and is
    /// updated in place to the last `conv_kernel - 1` inputs.
    ///
    /// Two launches, not one: the state update must observe the *old* state,
    /// which the convolution also reads, so they cannot be fused without
    /// either a second state buffer or a grid-wide barrier.
    ///
    /// This is the pure convolution, matching `ggml_ssm_conv`'s boundary.
    /// llama.cpp applies SiLU to the result as a separate op before slicing
    /// q/k/v apart (`src/models/qwen35moe.cpp`, `conv_output_silu`); that is
    /// the caller's job here too.
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        seq_len: usize,
        channels: usize,
        conv_kernel: usize,
    ) -> Result<(), LayerOpsError> {
        if conv_kernel == 0 || conv_kernel > MAX_CONV_KERNEL {
            return Err(LayerOpsError::UnsupportedConvKernel { conv_kernel });
        }
        if channels == 0 {
            return Err(LayerOpsError::UnsupportedWidth { width: channels });
        }
        check_len("conv1d x", seq_len * channels, x.len())?;
        check_len("conv1d weight", channels * conv_kernel, weight.len())?;
        check_len("conv1d state", channels * (conv_kernel - 1), state.len())?;
        check_len("conv1d out", seq_len * channels, out.len())?;
        if seq_len == 0 {
            return Ok(());
        }

        const BLOCK: usize = 256;
        let channels_i32 = channels as i32;
        let conv_kernel_i32 = conv_kernel as i32;
        let seq_len_i32 = seq_len as i32;

        let cfg = LaunchConfig {
            grid_dim: (channels.div_ceil(BLOCK) as u32, seq_len as u32, 1),
            block_dim: (BLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&self.conv1d);
        builder
            .arg(x)
            .arg(weight)
            .arg(&*state)
            .arg(&mut *out)
            .arg(&channels_i32)
            .arg(&conv_kernel_i32);
        // SAFETY: threads past `channels` return before touching memory. The
        // window position `t - (conv_kernel - 1) + i` is either in
        // `[0, seq_len)` and indexes `x`, or in `[-(conv_kernel-1), 0)` and
        // indexes `state` at `ch * carry + (u + carry)` which is in
        // `[0, carry)`. Both buffers were length-checked above.
        unsafe { builder.launch(cfg) }?;

        let state_cfg = LaunchConfig {
            grid_dim: (channels.div_ceil(BLOCK).min(1024) as u32, 1, 1),
            block_dim: (BLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&self.conv1d_state);
        builder
            .arg(x)
            .arg(&mut *state)
            .arg(&seq_len_i32)
            .arg(&channels_i32)
            .arg(&conv_kernel_i32);
        // SAFETY: same index bounds as above, and `conv_kernel <=
        // MAX_CONV_KERNEL` is what makes the kernel's `staged` register array
        // large enough — checked at the top of this function, which is the
        // only path that launches it.
        unsafe { builder.launch(state_cfg) }?;
        Ok(())
    }
}

/// Block width for a row of `width` elements: a whole number of warps, capped
/// at [`MAX_BLOCK`], with wider rows covered by the kernel's strided loop.
///
/// A whole number of warps is required, not preferred: `block_reduce_sum`
/// shuffles with a full `0xffffffff` mask, so every lane of every warp in the
/// block must reach it. Threads past `width` contribute a zero partial.
fn block_for_width(width: usize) -> usize {
    width.next_multiple_of(32).clamp(32, MAX_BLOCK)
}

/// Rejects a buffer whose length disagrees with the declared geometry.
fn check_len(what: &'static str, expected: usize, got: usize) -> Result<(), LayerOpsError> {
    if got == expected {
        Ok(())
    } else {
        Err(LayerOpsError::ShapeMismatch {
            what,
            expected,
            got,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Where a snippet starts in the kernel source, or a failure naming it.
    fn at(needle: &str) -> usize {
        LAYER_OPS_SRC
            .find(needle)
            .unwrap_or_else(|| panic!("kernel source no longer contains `{needle}`"))
    }

    #[test]
    fn the_rotary_tail_is_copied_not_computed() {
        // The documented invariant, asserted structurally because it is a
        // property of the *code path*, not of any particular input: the
        // pass-through branch must be a bare load-store with no arithmetic,
        // or the bits stop being preserved and Tolerance::exact() stops being
        // reachable. Multiplying by cos(0) would be correct to an ulp and
        // would fail the differential gate.
        assert!(LAYER_OPS_SRC.contains(
            "    if (j >= rope_dim) {\n        out[base + j] = x[base + j];\n        return;\n    }"
        ));
    }

    #[test]
    fn the_rotary_pairing_is_split_half_neox_not_interleaved() {
        // NEOX pairs i with i + rope_dim/2. The other common convention pairs
        // 2i with 2i+1; both produce a rotation, both preserve norm, and only
        // one is Qwen's. `xabe_kernels::rope` cites
        // ggml_compute_forward_rope_flt's GGML_ROPE_TYPE_NEOX case for this.
        assert!(LAYER_OPS_SRC.contains("int half = rope_dim >> 1;"));
        assert!(LAYER_OPS_SRC.contains("float x1 = x[base + i + half];"));
    }

    #[test]
    fn the_rotation_signs_are_the_forward_rotation_not_its_transpose() {
        // out[i] = x0 cos - x1 sin, out[i+half] = x0 sin + x1 cos. Swapping
        // the two signs rotates by -position instead of +position, which is
        // undetectable by norm preservation and by position 0.
        assert!(LAYER_OPS_SRC.contains("(x0 * cos_a - x1 * sin_a)"));
        assert!(LAYER_OPS_SRC.contains("(x0 * sin_a + x1 * cos_a)"));
    }

    #[test]
    fn the_convolution_denies_the_compiler_its_fma_contraction() {
        // The one thing standing between this kernel and bit-exact agreement
        // with the scalar reference. `acc += x * w` would let nvcc emit an
        // fma, which rounds once where Rust rounds twice, and the
        // differential test's Tolerance::exact() would fail.
        assert!(LAYER_OPS_SRC.contains("acc = __fadd_rn(acc, __fmul_rn(xv,"));
        let conv_start = at("__global__ void conv1d_causal_depthwise(");
        let conv_end = at("__global__ void conv1d_update_state(");
        let body = &LAYER_OPS_SRC[conv_start..conv_end];
        assert!(
            !body.contains("acc +="),
            "the accumulation reverted to a contractible form",
        );
    }

    #[test]
    fn the_convolution_taps_run_forward_so_the_last_tap_is_the_current_token() {
        // u = t - (K-1) + i with i ascending: tap 0 reads the oldest input and
        // tap K-1 reads token t. Reversing this is anti-causal and stays
        // finite, fluent, and wrong.
        assert!(LAYER_OPS_SRC.contains("int u = t - carry + i;"));
        assert!(LAYER_OPS_SRC.contains("for (int i = 0; i < conv_kernel; ++i) {"));
    }

    #[test]
    fn the_convolution_reads_only_the_current_token_and_its_predecessors() {
        // Causality as a bound on the index rather than as a comment: the
        // largest window position any thread forms is `t - carry + (K-1)`,
        // which is `t` exactly. Checked arithmetically for the real geometry
        // so a future edit to the loop bound has to break this to land.
        let conv_kernel = 4i32;
        let carry = conv_kernel - 1;
        for t in 0..8i32 {
            let positions: Vec<i32> = (0..conv_kernel).map(|i| t - carry + i).collect();
            assert_eq!(*positions.iter().max().unwrap(), t);
            assert_eq!(*positions.iter().min().unwrap(), t - carry);
        }
    }

    #[test]
    fn the_state_update_stages_before_it_writes() {
        // In-place safety. A batch shorter than conv_kernel-1 — which is every
        // decode step — reads entries of the cache it also overwrites, so the
        // read loop must complete before the write loop begins.
        let read = at("staged[j] = (u >= 0)");
        let write = at("state[ch * carry + j] = staged[j];");
        assert!(read < write, "the cache is written before it is fully read");
    }

    #[test]
    fn the_state_array_bound_covers_the_widest_supported_kernel() {
        // The register array is `float staged[MAX_CONV_KERNEL]` and holds
        // conv_kernel-1 entries, so the host-side bound must not exceed it.
        assert!(LAYER_OPS_SRC.contains("#define MAX_CONV_KERNEL 8"));
        assert_eq!(MAX_CONV_KERNEL, 8);
        // Qwen3.6's real width — `qwen35moe.ssm.conv_kernel` — must be
        // accepted. Spelled as a literal because this crate deliberately does
        // not depend on `xabe-model`; the differential test in `xabe-engine`
        // reads it from `ModelConfig` and would fail if the two disagreed.
        // A const block, so a bound that stopped covering the model would be
        // a compile error rather than a test failure.
        const { assert!(4 <= MAX_CONV_KERNEL) };
    }

    #[test]
    fn rms_norm_divides_by_sqrt_rather_than_calling_rsqrtf() {
        // Same choice as both GDN kernels: rsqrtf is faster and not
        // IEEE-exact, and the reference divides. Matching it keeps the
        // measured disagreement attributable to the reduction order alone.
        assert!(
            LAYER_OPS_SRC.contains("float inv_rms = 1.0f / sqrtf(sum_sq / (float)width + eps);")
        );
        assert!(LAYER_OPS_SRC.contains("out[base + j] = x[base + j] * inv_rms * weight[j];"));
        // The negative half, scoped to the kernel body so that the comment
        // explaining the choice does not satisfy its own assertion.
        let body = &LAYER_OPS_SRC[at("__global__ void rms_norm_rows(")..at("// Partial rotary")];
        assert!(!body.contains("rsqrtf"), "rms_norm reverted to rsqrtf");
    }

    #[test]
    fn swiglu_uses_the_accurate_exponential() {
        assert!(LAYER_OPS_SRC.contains("out[i] = (g / (1.0f + expf(-g))) * up[i];"));
        let body = &LAYER_OPS_SRC[at("__global__ void swiglu_mul(")..at("// Causal depthwise")];
        assert!(!body.contains("__expf"), "swiglu reverted to __expf");
    }

    #[test]
    fn the_block_width_is_always_a_whole_number_of_warps() {
        // block_reduce_sum shuffles with a full mask, so a partial warp would
        // hang or read garbage. Every width the model uses, plus the awkward
        // ones.
        for width in [1usize, 31, 32, 33, 128, 256, 2048, 4096, 8192] {
            let block = block_for_width(width);
            assert_eq!(block % 32, 0, "width {width} gave block {block}");
            assert!((32..=MAX_BLOCK).contains(&block), "width {width}");
        }
        // The three widths RMSNorm actually runs at in one forward pass.
        assert_eq!(block_for_width(2048), 1024);
        assert_eq!(block_for_width(256), 256);
        assert_eq!(block_for_width(128), 128);
    }

    #[test]
    fn geometry_is_validated_not_assumed() {
        // The checks that run without a device.
        assert!(
            LayerOpsError::UnsupportedWidth { width: 0 }
                .to_string()
                .contains('0')
        );
        assert!(
            LayerOpsError::UnsupportedRotaryShape {
                head_dim: 2048,
                rope_dim: 64,
            }
            .to_string()
            .contains("2048")
        );
        assert!(
            LayerOpsError::UnsupportedConvKernel { conv_kernel: 9 }
                .to_string()
                .contains('9')
        );
        assert!(
            LayerOpsError::ShapeMismatch {
                what: "conv1d state",
                expected: 24576,
                got: 8192,
            }
            .to_string()
            .contains("24576")
        );
    }

    #[test]
    fn a_length_that_matches_the_geometry_is_accepted_and_one_that_does_not_is_not() {
        assert!(check_len("x", 4, 4).is_ok());
        assert!(check_len("x", 4, 5).is_err());
    }
}
