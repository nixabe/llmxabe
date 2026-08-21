//! Device ops for the qwen3vl vision tower: LayerNorm with bias, GELU,
//! vision M-RoPE, row softmax, f16 utility copies, and cuBLASLt f16 GEMM.
//!
//! Reference: `xabe_kernels::vision` — the tower structure, layouts and the
//! cited llama.cpp provenance live there; this module only reproduces those
//! ops on the device. Activations are f16 (matching llama.cpp's CUDA clip
//! path); every kernel computes in f32 and rounds once on store.
//!
//! # Why cuBLASLt, in a workspace that hand-writes its matmuls
//!
//! The tower is dense f16 GEMM over up to a few thousand rows — exactly the
//! shape cuBLASLt covers well on Turing — and it runs once per image at
//! admission, never inside a captured graph and never on the decode path
//! this project's kernels are specialized for. `MmaKernels` is int8 and
//! `LmHeadKernels` is Q6_K/Q8_0; neither speaks f16 weights, and writing a
//! third GEMM family for an off-hot-path pass would buy correctness risk
//! with no benchmark upside. llama.cpp makes the same call: its prefill
//! matmuls go through cuBLAS.
//!
//! # Geometry is per launch
//!
//! Image grids vary per request, so like [`super::layer_ops`] every launch
//! takes its shape as arguments and validates buffer lengths against them.
//! The vision pass is never captured (it runs before prefill on the same
//! stream), so host-sized launches are sound here; AGENTS.md rule 5 binds
//! captured paths only.

use std::sync::Arc;

use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};
use half::f16;

use super::compile;

/// CUDA C++ source for the elementwise and reduction kernels.
///
/// Kept as one translation unit so the whole set is one NVRTC invocation.
const VISION_SRC: &str = r#"
extern "C" {

// binary16 <-> fp32 by hand: NVRTC compiles from a string with no include
// path, so cuda_fp16.h is not reachable — same convention as
// `attention.rs`. Activations are stored binary16 and every kernel
// computes in fp32.
__device__ __forceinline__ float h2f(unsigned short h) {
    float f;
    asm("{ .reg .f16 a; mov.b16 a, %1; cvt.f32.f16 %0, a; }" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ unsigned short f2h(float f) {
    unsigned short h;
    asm("{ .reg .f16 a; cvt.rn.f16.f32 a, %1; mov.b16 %0, a; }" : "=h"(h) : "f"(f));
    return h;
}

__global__ void vision_layer_norm(
    const unsigned short* __restrict__ x,
    unsigned short* __restrict__ out,
    const unsigned short* __restrict__ gamma,
    const unsigned short* __restrict__ beta,
    int width,
    float eps)
{
    // One block per row. Two-pass mean/variance in f32.
    const unsigned short* row = x + (long long)blockIdx.x * width;
    unsigned short* dst = out + (long long)blockIdx.x * width;

    __shared__ float warp_sums[32];
    float sum = 0.0f;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        sum += h2f(row[j]);
    }
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    }
    if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) >> 5;
        float v = threadIdx.x < warps ? warp_sums[threadIdx.x] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, off);
        }
        if (threadIdx.x == 0) warp_sums[0] = v;
    }
    __syncthreads();
    float mean = warp_sums[0] / (float)width;
    __syncthreads();

    float var = 0.0f;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        float d = h2f(row[j]) - mean;
        var += d * d;
    }
    for (int off = 16; off > 0; off >>= 1) {
        var += __shfl_down_sync(0xffffffffu, var, off);
    }
    if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = var;
    __syncthreads();
    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) >> 5;
        float v = threadIdx.x < warps ? warp_sums[threadIdx.x] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, off);
        }
        if (threadIdx.x == 0) warp_sums[0] = v;
    }
    __syncthreads();
    float inv = rsqrtf(warp_sums[0] / (float)width + eps);

    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        float g = h2f(gamma[j]);
        float b = h2f(beta[j]);
        float centered = (h2f(row[j]) - mean) * inv;
        dst[j] = f2h(centered * g + b);
    }
}

__global__ void vision_gelu(unsigned short* __restrict__ x, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // ggml's tanh approximation, computed in f32.
    float v = h2f(x[i]);
    float inner = 0.79788456f * v * (1.0f + 0.044715f * v * v);
    x[i] = f2h(0.5f * v * (1.0f + tanhf(inner)));
}

// Rotates the query and key slices of the fused qkv buffer in place.
// Layout per token: [q 0..hidden | k hidden..2*hidden | v untouched],
// heads of head_dim within each slice. positions holds (row, col) per
// token. Pairs (j, j + head_dim/2); pairs below the quarter rotate by the
// patch row, the rest by the column, each with frequencies restarting at
// theta^0 — GGML_ROPE_TYPE_VISION with independent sections, see
// xabe_kernels::vision::tower.
__global__ void vision_rope_qk(
    unsigned short* __restrict__ qkv,
    const int* __restrict__ positions,
    int hidden,
    int head_dim,
    float theta_base)
{
    int token = blockIdx.x;
    int slice = blockIdx.y;         // 0 = q, 1 = k
    int head = blockIdx.z;
    int j = threadIdx.x;            // pair index, [0, head_dim/2)
    int half_dim = head_dim >> 1;
    int quarter = head_dim >> 2;
    if (j >= half_dim) return;

    unsigned short* base = qkv + (long long)token * 3 * hidden + slice * hidden + head * head_dim;
    int pos = j < quarter ? positions[2 * token] : positions[2 * token + 1];
    int k = j < quarter ? j : j - quarter;
    float freq = powf(theta_base, -(float)k / (float)quarter);
    float angle = (float)pos * freq;
    float sin_a, cos_a;
    sincosf(angle, &sin_a, &cos_a);
    float x0 = h2f(base[j]);
    float x1 = h2f(base[j + half_dim]);
    base[j] = f2h(x0 * cos_a - x1 * sin_a);
    base[j + half_dim] = f2h(x0 * sin_a + x1 * cos_a);
}

__global__ void vision_softmax_rows(
    unsigned short* __restrict__ scores,
    int row_len)
{
    // One block per row, rows are (batch * n) in grid.x.
    unsigned short* row = scores + (long long)blockIdx.x * row_len;
    __shared__ float warp_red[32];

    float local_max = -3.4e38f;
    for (int j = threadIdx.x; j < row_len; j += blockDim.x) {
        local_max = fmaxf(local_max, h2f(row[j]));
    }
    for (int off = 16; off > 0; off >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffu, local_max, off));
    }
    if ((threadIdx.x & 31) == 0) warp_red[threadIdx.x >> 5] = local_max;
    __syncthreads();
    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) >> 5;
        float v = threadIdx.x < warps ? warp_red[threadIdx.x] : -3.4e38f;
        for (int off = 16; off > 0; off >>= 1) {
            v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
        }
        if (threadIdx.x == 0) warp_red[0] = v;
    }
    __syncthreads();
    float row_max = warp_red[0];
    __syncthreads();

    float total = 0.0f;
    for (int j = threadIdx.x; j < row_len; j += blockDim.x) {
        float e = expf(h2f(row[j]) - row_max);
        row[j] = f2h(e);
        total += e;
    }
    for (int off = 16; off > 0; off >>= 1) {
        total += __shfl_down_sync(0xffffffffu, total, off);
    }
    if ((threadIdx.x & 31) == 0) warp_red[threadIdx.x >> 5] = total;
    __syncthreads();
    if (threadIdx.x < 32) {
        int warps = (blockDim.x + 31) >> 5;
        float v = threadIdx.x < warps ? warp_red[threadIdx.x] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, off);
        }
        if (threadIdx.x == 0) warp_red[0] = v;
    }
    __syncthreads();
    float inv = 1.0f / warp_red[0];
    for (int j = threadIdx.x; j < row_len; j += blockDim.x) {
        row[j] = f2h(h2f(row[j]) * inv);
    }
}

__global__ void vision_add_f16(
    unsigned short* __restrict__ a,
    const unsigned short* __restrict__ b,
    long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = f2h(h2f(a[i]) + h2f(b[i]));
}

__global__ void vision_half_to_float(
    const unsigned short* __restrict__ src,
    float* __restrict__ dst,
    long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = h2f(src[i]);
}

} // extern "C"
"#;

/// Errors from the vision device ops.
#[derive(Debug)]
pub enum VisionError {
    /// NVRTC rejected the source, or the module failed to load.
    Build(String),
    /// cuBLASLt refused the matmul (no heuristic, bad descriptor).
    Blas(cudarc::cublaslt::result::CublasError),
    /// The driver failed.
    Driver(DriverError),
    /// A buffer is not the length the declared geometry requires.
    WrongLength {
        name: &'static str,
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for VisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Build(msg) => write!(f, "vision kernel build failed: {msg}"),
            Self::Blas(e) => write!(f, "cublasLt: {e:?}"),
            Self::Driver(e) => write!(f, "driver: {e}"),
            Self::WrongLength {
                name,
                expected,
                got,
            } => write!(f, "{name}: expected {expected} elements, got {got}"),
        }
    }
}

impl std::error::Error for VisionError {}

impl From<DriverError> for VisionError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

impl From<cudarc::cublaslt::result::CublasError> for VisionError {
    fn from(e: cudarc::cublaslt::result::CublasError) -> Self {
        Self::Blas(e)
    }
}

fn check_len(name: &'static str, expected: usize, got: usize) -> Result<(), VisionError> {
    if expected == got {
        Ok(())
    } else {
        Err(VisionError::WrongLength {
            name,
            expected,
            got,
        })
    }
}

/// Pre-sized buffers may exceed the launch's geometry; equality is only
/// demanded of weights.
fn check_min(name: &'static str, expected: usize, got: usize) -> Result<(), VisionError> {
    if got >= expected {
        Ok(())
    } else {
        Err(VisionError::WrongLength {
            name,
            expected,
            got,
        })
    }
}

const BLOCK: u32 = 256;

fn grid_1d(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(BLOCK as usize) as u32, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// The compiled vision kernel set plus a cuBLASLt handle, bound to one
/// stream.
pub struct VisionKernels {
    layer_norm: CudaFunction,
    gelu: CudaFunction,
    rope_qk: CudaFunction,
    softmax_rows: CudaFunction,
    add_f16: CudaFunction,
    half_to_float: CudaFunction,
    blas: CudaBlasLT,
}

impl VisionKernels {
    /// Compile and load the kernels on `ctx`, binding GEMMs to `stream`.
    pub fn new(ctx: &Arc<CudaContext>, stream: Arc<CudaStream>) -> Result<Self, VisionError> {
        let ptx = compile(VISION_SRC, "vision").map_err(VisionError::Build)?;
        let module = ctx.load_module(ptx).map_err(VisionError::Driver)?;
        let f = |name: &str| {
            module
                .load_function(name)
                .map_err(|e| VisionError::Build(format!("{name}: {e}")))
        };
        Ok(Self {
            layer_norm: f("vision_layer_norm")?,
            gelu: f("vision_gelu")?,
            rope_qk: f("vision_rope_qk")?,
            softmax_rows: f("vision_softmax_rows")?,
            add_f16: f("vision_add_f16")?,
            half_to_float: f("vision_half_to_float")?,
            blas: CudaBlasLT::new(stream)?,
        })
    }

    /// LayerNorm with weight and bias over each of `rows` rows of `width`.
    ///
    /// `out` must not alias `x` (the residual stream keeps the un-normed
    /// value); enforced by the exclusive borrow.
    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f16>,
        out: &mut CudaSlice<f16>,
        gamma: &CudaSlice<f16>,
        beta: &CudaSlice<f16>,
        rows: usize,
        width: usize,
        eps: f32,
    ) -> Result<(), VisionError> {
        check_min("layer_norm x", rows * width, x.len())?;
        check_min("layer_norm out", rows * width, out.len())?;
        check_len("layer_norm gamma", width, gamma.len())?;
        check_len("layer_norm beta", width, beta.len())?;
        if rows == 0 {
            return Ok(());
        }
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let width_i32 = width as i32;
        let mut b = stream.launch_builder(&self.layer_norm);
        b.arg(x)
            .arg(out)
            .arg(gamma)
            .arg(beta)
            .arg(&width_i32)
            .arg(&eps);
        // SAFETY: one block per row; every in-row index is bounded by
        // `width`, and both buffers hold `rows * width` halves.
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    /// In-place tanh-approximation GELU over the first `n` elements.
    pub fn gelu(
        &self,
        stream: &Arc<CudaStream>,
        x: &mut CudaSlice<f16>,
        n: usize,
    ) -> Result<(), VisionError> {
        if x.len() < n {
            return Err(VisionError::WrongLength {
                name: "gelu x",
                expected: n,
                got: x.len(),
            });
        }
        if n == 0 {
            return Ok(());
        }
        let n_ll = n as i64;
        let mut b = stream.launch_builder(&self.gelu);
        b.arg(x).arg(&n_ll);
        // SAFETY: the guard `i >= n` bounds every access; `x` holds at
        // least `n` halves.
        unsafe { b.launch(grid_1d(n)) }?;
        Ok(())
    }

    /// Vision M-RoPE over the q and k slices of the fused qkv buffer.
    ///
    /// `positions` holds `(row, col)` i32 pairs per token, in the same
    /// token order as the rows of `qkv`.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_qk(
        &self,
        stream: &Arc<CudaStream>,
        qkv: &mut CudaSlice<f16>,
        positions: &CudaSlice<i32>,
        tokens: usize,
        hidden: usize,
        head_dim: usize,
        theta_base: f32,
    ) -> Result<(), VisionError> {
        check_min("rope qkv", tokens * 3 * hidden, qkv.len())?;
        check_min("rope positions", tokens * 2, positions.len())?;
        assert!(
            head_dim.is_multiple_of(4) && hidden.is_multiple_of(head_dim),
            "vision rope needs head_dim % 4 == 0 and hidden % head_dim == 0"
        );
        if tokens == 0 {
            return Ok(());
        }
        let heads = hidden / head_dim;
        let cfg = LaunchConfig {
            grid_dim: (tokens as u32, 2, heads as u32),
            block_dim: ((head_dim / 2) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let hidden_i32 = hidden as i32;
        let head_dim_i32 = head_dim as i32;
        let mut b = stream.launch_builder(&self.rope_qk);
        b.arg(qkv)
            .arg(positions)
            .arg(&hidden_i32)
            .arg(&head_dim_i32)
            .arg(&theta_base);
        // SAFETY: grid is (tokens, 2, heads), block head_dim/2 threads; the
        // furthest access is token*3*hidden + slice*hidden + head*head_dim
        // + head_dim - 1, inside the checked qkv length. positions is
        // indexed by 2*token + 1 < 2*tokens.
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    /// Row-wise softmax in place over `rows` rows of `row_len`.
    pub fn softmax_rows(
        &self,
        stream: &Arc<CudaStream>,
        scores: &mut CudaSlice<f16>,
        rows: usize,
        row_len: usize,
    ) -> Result<(), VisionError> {
        if scores.len() < rows * row_len {
            return Err(VisionError::WrongLength {
                name: "softmax scores",
                expected: rows * row_len,
                got: scores.len(),
            });
        }
        if rows == 0 {
            return Ok(());
        }
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let row_len_i32 = row_len as i32;
        let mut b = stream.launch_builder(&self.softmax_rows);
        b.arg(scores).arg(&row_len_i32);
        // SAFETY: one block per row, in-row indices bounded by row_len, and
        // the buffer holds at least rows * row_len halves (checked above).
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    /// `a[..n] += b[..n]` elementwise.
    pub fn add_assign(
        &self,
        stream: &Arc<CudaStream>,
        a: &mut CudaSlice<f16>,
        b: &CudaSlice<f16>,
        n: usize,
    ) -> Result<(), VisionError> {
        if a.len() < n || b.len() < n {
            return Err(VisionError::WrongLength {
                name: "add_assign",
                expected: n,
                got: a.len().min(b.len()),
            });
        }
        if n == 0 {
            return Ok(());
        }
        let n_ll = n as i64;
        let mut builder = stream.launch_builder(&self.add_f16);
        builder.arg(a).arg(b).arg(&n_ll);
        // SAFETY: guarded by `i >= n`; both buffers checked to hold n.
        unsafe { builder.launch(grid_1d(n)) }?;
        Ok(())
    }

    /// Widen `src[..n]` into `dst[..n]`.
    pub fn to_f32(
        &self,
        stream: &Arc<CudaStream>,
        src: &CudaSlice<f16>,
        dst: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), VisionError> {
        if src.len() < n || dst.len() < n {
            return Err(VisionError::WrongLength {
                name: "to_f32",
                expected: n,
                got: src.len().min(dst.len()),
            });
        }
        if n == 0 {
            return Ok(());
        }
        let n_ll = n as i64;
        let mut b = stream.launch_builder(&self.half_to_float);
        b.arg(src).arg(dst).arg(&n_ll);
        // SAFETY: guarded by `i >= n`; both buffers checked to hold n.
        unsafe { b.launch(grid_1d(n)) }?;
        Ok(())
    }

    /// `y[tokens × out] = x[tokens × in] · wᵀ + bias`, all row-major f16,
    /// f32 accumulation.
    ///
    /// `w` is a GGUF-layout projection: `out` rows of length `in`. The
    /// row-major views map onto cuBLASLt's column-major world as
    /// `C(out×n) = Aᵀ(out×in) · B(in×n)` with A = w and B = x, both at
    /// their natural leading dimensions, so no copies or transposes are
    /// materialized. `x_ld`/`y_ld` allow reading from and writing into
    /// slices of wider row buffers (the fused qkv rows).
    #[allow(clippy::too_many_arguments)]
    pub fn proj(
        &self,
        x: &CudaSlice<f16>,
        w: &CudaSlice<f16>,
        bias: Option<&CudaSlice<f16>>,
        y: &mut CudaSlice<f16>,
        tokens: usize,
        in_dim: usize,
        out_dim: usize,
        beta: f32,
    ) -> Result<(), VisionError> {
        check_len("proj w", in_dim * out_dim, w.len())?;
        if x.len() < tokens * in_dim || y.len() < tokens * out_dim {
            return Err(VisionError::WrongLength {
                name: "proj x/y",
                expected: tokens * in_dim.max(out_dim),
                got: x.len().min(y.len()),
            });
        }
        if let Some(b) = bias {
            check_len("proj bias", out_dim, b.len())?;
        }
        if tokens == 0 {
            return Ok(());
        }
        let cfg = MatmulConfig {
            transa: true,
            transb: false,
            transc: false,
            m: out_dim as u64,
            n: tokens as u64,
            k: in_dim as u64,
            alpha: 1.0,
            lda: in_dim as i64,
            ldb: in_dim as i64,
            beta,
            ldc: out_dim as i64,
            stride_a: None,
            stride_b: None,
            stride_c: None,
            stride_bias: None,
            batch_size: None,
        };
        // SAFETY: dimensions and leading strides are validated against the
        // buffer lengths above; bias is out_dim long, matching m.
        unsafe { self.blas.matmul(cfg, w, x, y, bias, None) }?;
        Ok(())
    }

    /// Batched per-head attention scores:
    /// `scores[h][q][k] = scale · Q[q] · K[k]` over `heads` consecutive
    /// heads, where token `t`'s slice for the first head starts at
    /// `t * row_len + q_offset` (resp. `k_offset`) and later heads advance
    /// by `head_dim` — the fused-qkv layout, with the q/k slice base and
    /// any head-chunk offset folded into the offsets.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_scores(
        &self,
        buf: &CudaSlice<f16>,
        q_offset: usize,
        k_offset: usize,
        scores: &mut CudaSlice<f16>,
        tokens: usize,
        row_len: usize,
        head_dim: usize,
        heads: usize,
        scale: f32,
    ) -> Result<(), VisionError> {
        if scores.len() < heads * tokens * tokens {
            return Err(VisionError::WrongLength {
                name: "attn scores",
                expected: heads * tokens * tokens,
                got: scores.len(),
            });
        }
        if tokens == 0 || heads == 0 {
            return Ok(());
        }
        let cfg = MatmulConfig {
            // C(k_tokens × q_tokens) per head = Kᵀ-of-memory transposed ·
            // Q-memory: both operands are the (head_dim × tokens)
            // column-major views their row-major memory naturally is.
            transa: true,
            transb: false,
            transc: false,
            m: tokens as u64,
            n: tokens as u64,
            k: head_dim as u64,
            alpha: scale,
            lda: row_len as i64,
            ldb: row_len as i64,
            beta: 0.0,
            ldc: tokens as i64,
            stride_a: Some(head_dim as i64),
            stride_b: Some(head_dim as i64),
            stride_c: Some((tokens * tokens) as i64),
            stride_bias: None,
            batch_size: Some(heads as i32),
        };
        let q_view = slice_from(buf, q_offset);
        let k_view = slice_from(buf, k_offset);
        // SAFETY: per-head strides walk `heads` slices of `head_dim` within
        // rows of `row_len`; the furthest element read is
        // (tokens-1)*row_len + offset + heads*head_dim - 1, within the
        // caller's buffer; scores length is checked above.
        unsafe { self.blas.matmul(cfg, &k_view, &q_view, scores, None, None) }?;
        Ok(())
    }

    /// Batched per-head attention output:
    /// `out[q] = Σ_k probs[h][q][k] · V[k]`, mirroring
    /// [`Self::attn_scores`]'s offset convention: token `t`'s first-head
    /// value slice starts at `t * v_row_len + v_offset`, the output's at
    /// `t * out_row_len + out_offset`.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_output(
        &self,
        probs: &CudaSlice<f16>,
        v: &CudaSlice<f16>,
        v_offset: usize,
        out: &mut CudaSlice<f16>,
        out_offset: usize,
        tokens: usize,
        v_row_len: usize,
        out_row_len: usize,
        head_dim: usize,
        heads: usize,
    ) -> Result<(), VisionError> {
        if probs.len() < heads * tokens * tokens {
            return Err(VisionError::WrongLength {
                name: "attn probs",
                expected: heads * tokens * tokens,
                got: probs.len(),
            });
        }
        if tokens == 0 || heads == 0 {
            return Ok(());
        }
        let cfg = MatmulConfig {
            // C(head_dim × q_tokens) per head = V-memory (head_dim × k) ·
            // probsᵀ-memory (k × q).
            transa: false,
            transb: false,
            transc: false,
            m: head_dim as u64,
            n: tokens as u64,
            k: tokens as u64,
            alpha: 1.0,
            lda: v_row_len as i64,
            ldb: tokens as i64,
            beta: 0.0,
            ldc: out_row_len as i64,
            stride_a: Some(head_dim as i64),
            stride_b: Some((tokens * tokens) as i64),
            stride_c: Some(head_dim as i64),
            stride_bias: None,
            batch_size: Some(heads as i32),
        };
        let v_view = slice_from(v, v_offset);
        let probs_view = slice_from(probs, 0);
        let mut out_view = slice_from_mut(out, out_offset);
        // SAFETY: same windowing argument as attn_scores; the output's
        // furthest write is (tokens-1)*out_row_len + out_offset +
        // heads*head_dim - 1, within the caller's buffer.
        unsafe {
            self.blas
                .matmul(cfg, &v_view, &probs_view, &mut out_view, None, None)
        }?;
        Ok(())
    }
}

/// A shared sub-view of a device buffer starting at `offset` elements.
fn slice_from(s: &CudaSlice<f16>, offset: usize) -> cudarc::driver::CudaView<'_, f16> {
    s.slice(offset..)
}

/// An exclusive sub-view of a device buffer starting at `offset` elements.
fn slice_from_mut(s: &mut CudaSlice<f16>, offset: usize) -> cudarc::driver::CudaViewMut<'_, f16> {
    s.slice_mut(offset..)
}
