//! Gated Attention block against the llama.cpp capture, step by step.
//!
//! [`xabe_engine::block::attention`] assembles ten of Qwen3.6's forty layers.
//! This is its gate: the real block, on the real Q8_0 weights, over the real
//! 19-token prompt `docs/ORACLE.md` describes, compared against **every**
//! intermediate llama.cpp recorded for blocks 3 and 39 — not just the block
//! output, so a divergence localizes to one operation instead of to "the
//! attention block".
//!
//! ## Three columns, because one number cannot separate three causes
//!
//! For each step the test reports, and gates, three comparisons:
//!
//! | Column | What it compares | What a failure means |
//! | --- | --- | --- |
//! | `host|gold` | a host reference of **that one op**, fed llama.cpp's own input for that step, against llama.cpp's output | this repository has the *operation* wrong — wrong epsilon, wrong pairing, wrong head mapping |
//! | `dev |host` | the device block's intermediate against a host fp32 reference of the **same chain** | a *kernel* is wrong |
//! | `dev |gold` | the device block's intermediate against llama.cpp | the end-to-end number, which inherits both of the above plus llama.cpp's own arithmetic |
//!
//! The third column is the loosest and it is loose for a reason that is
//! measured here rather than asserted:
//!
//! ## llama.cpp's projections are int8, and this test proves it
//!
//! `attn_q`, `attn_k`, `attn_v` and `attn_output` are Q8_0. On CUDA with 19
//! tokens in the batch, `ggml_cuda_mul_mat` routes a quantized `src0` past
//! MMVF and MMVQ to **MMQ**, which quantizes the *activations* to `q8_1`
//! blocks of 32 (`d = amax/127`, `roundf`) and does the dot product in int8.
//! So the captured `Qcur_full-N` is not a float matmul at all.
//!
//! Measured, on the golden's own inputs (see the `q8_1` rows the test
//! prints): an exact fp32 matmul of the dequantized weights disagrees with
//! the capture by **2.1e-2 to 1.1e-1 of the tensor's RMS**, while the same
//! matmul with the activations put through llama.cpp's `q8_1` quantization
//! first agrees to **6.1e-6 to 9.4e-5** — between 1,186x and 10,384x tighter
//! over the eight projections, at cosine 1.0000000000 to ten digits. The test
//! asserts both halves of that contrast, so the loose `dev|gold` tolerance on
//! the projections stops being justified the moment the explanation stops
//! holding.
//!
//! This project's fp32 GEMV is therefore *more* accurate than the oracle at
//! these four steps, and the residual gap is llama.cpp's activation
//! quantization rather than an error here. That is stated because it is the
//! one place in this file where "does not match" is the correct outcome.
//!
//! ## IMRoPE, and why a NEOX kernel is the right one
//!
//! `qwen35moe.cpp` calls `ggml_rope_multi` with `LLAMA_ROPE_TYPE_IMROPE`.
//! The test reads `qwen35moe.rope.dimension_sections` out of the model file
//! and proves, from the section widths alone, that every rotated pair draws
//! the ordinary text position — so interleaved M-RoPE collapses to plain NEOX
//! partial rotary here. A model whose sections did not collapse would fail
//! that assertion rather than be silently rotated wrong.
//!
//! ## The interleave trap
//!
//! `attn_q.weight` packs each head's query followed by that head's output
//! gate. The test checks the deinterleave on **this repository's own** device
//! output (not only on the capture, which `golden.rs` already does): the
//! kernel's `query`/`gate` must be bit-identical to a host deinterleave of
//! the device's own packed projection, and a halves split must disagree on
//! every head but the first — exactly `tokens * head_dim` of `tokens *
//! q_heads * head_dim` elements agreeing, because head 0 coincides under both
//! readings.
//!
//! SKIPS — reporting that it skipped — without a CUDA driver, a supported
//! device, the model file, or the golden capture. Per `AGENTS.md`, a skip is
//! not a pass.

#[path = "golden.rs"]
mod golden;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_engine::block::attention::{
    AttentionKernelSet, GatedAttentionBlock, KvCache, attention_layers,
};
use xabe_engine::weights::DeviceWeights;
use xabe_gguf::{GgufFile, GgufValue};
use xabe_kernels::attention::{causal_attention_streaming, kv_head_for_query_head};
use xabe_kernels::gemv::gemv_batch;
use xabe_kernels::norm::rms_norm;
use xabe_kernels::quant::{BLOCK_Q8_0_BYTES, QK8_0, dequantize_row_q8_0};
use xabe_kernels::rope::apply_rope;
use xabe_model::config::{LayerKind, ModelConfig};
use xabe_model::weights::{Role, WeightSchema};

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llama.cpp/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

/// The two Gated Attention blocks whose internals the capture carries.
const CAPTURED_LAYERS: [u32; 2] = [3, 39];

/// Elements per `q8_1` activation block, and the divisor llama.cpp's
/// `quantize_q8_1` / `quantize_mmq_q8_1` use for the scale. Spelled here
/// because this file emulates that quantization to explain the projections;
/// see the module docs.
const QK8_1: usize = 32;
const Q8_1_LEVELS: f32 = 127.0;

/// How closely a host reference of one operation, fed llama.cpp's own input
/// for that step, must reproduce llama.cpp's output for that step.
///
/// `abs_over_rms` is the worst absolute error as a fraction of the reference
/// tensor's RMS — scale-free, unlike `compare()`'s `max_rel_error`, whose
/// `REL_EPS = 1e-6` floor makes it report `abs_error / 1e-6` for the many
/// near-zero elements in these tensors (see `.omc/handoffs/team-plan.md`).
///
/// Every op in this class is a pure fp32 formula with no reduced-precision
/// step anywhere in llama.cpp's version of it, so agreement is at the fp32
/// round-off floor. Measured over both blocks: the deinterleave and the
/// residual add are **bit-identical** (0.0), the two per-head norms, the
/// rotary, the sigmoid and the gate product land at 8.5e-7 to 6.8e-6, and the
/// worst is the hidden-size input norm at **2.97e-5** — that one because
/// `ggml_compute_forward_rms_norm_f32` accumulates its sum of squares in
/// `ggml_float` (double) over 2,048 terms where
/// `xabe_kernels::norm::rms_norm` accumulates in f32. The 1e-4 bound is 3.4x
/// the worst observation.
const OP_DEFINITION: Gate = Gate {
    abs_over_rms: 1e-4,
    min_cosine: 1.0 - 1e-9,
};

/// The attention step's own floor.
///
/// Looser than [`OP_DEFINITION`] because llama.cpp ran this capture with
/// **flash attention enabled and `type_k = type_v = f16`**
/// (`docs/ORACLE.md` §1), so the captured `attn_pregate-N` is a softmax over
/// keys and values that were rounded to fp16 on their way into the KV cache,
/// summed in the flash kernel's order. Neither is reproducible by an fp32
/// scalar reference. The test measures the f16-rounded variant alongside the
/// fp32 one and prints both, so how much of the gap the cache dtype explains
/// is visible rather than assumed.
///
/// Worst measured: **5.55e-3 of RMS** at block 39 (4.47e-3 at block 3),
/// cosine 0.9999999576. Rounding K and V to fp16 first accounts for only part
/// of it — 3.93e-3 and 4.04e-3 respectively — so the rest is the flash
/// kernel's summation order, not the cache dtype. The 2e-2 bound is 3.6x the
/// worst observation.
const ATTENTION_FLOOR: Gate = Gate {
    abs_over_rms: 2e-2,
    min_cosine: 1.0 - 1e-6,
};

/// How closely llama.cpp's `q8_1`-quantized-activation matmul must be
/// reproduced by emulating that quantization on the host.
///
/// This is the assertion that *justifies* [`PROJECTION_VS_GOLDEN`]. Measured
/// over all eight projections: 6.08e-6 to **9.44e-5** of RMS, cosine
/// 1.0000000000 throughout. The worst is `attn_output-39`, whose dot product
/// runs over 4,096 terms rather than 2,048; the 1e-3 bound is 10.6x it.
const PROJECTION_Q8_1: Gate = Gate {
    abs_over_rms: 1e-3,
    min_cosine: 1.0 - 1e-9,
};

/// How much worse an exact fp32 matmul must be than the `q8_1` emulation
/// before the "llama.cpp quantized its activations" explanation counts as
/// demonstrated rather than asserted.
///
/// Measured ratios over the eight projections span 1,186x (`attn_output-39`)
/// to 10,384x (`Vcur-3`). The bound is 100x — an order of magnitude below the
/// smallest observation — so it fails loudly if the two ever converge, which
/// is the only thing that would invalidate [`PROJECTION_VS_GOLDEN`]'s width.
const FP32_MUST_BE_WORSE_BY: f64 = 100.0;

/// How closely a device kernel must match a host fp32 reference of the same
/// chain.
///
/// This is the classic differential bound and the one that catches a wrong
/// *kernel*. Differences are reduction order (warp-shuffle trees against
/// sequential sums, in RMSNorm, the GEMV and the attention dot product),
/// `expf` against `f32::exp`, and the flash kernel's per-tile rather than
/// per-key online-softmax rescale. All are unbiased and bounded.
///
/// Measured over all thirty steps of both blocks: **1.22e-4** at worst
/// (`attn_output-3`, whose absolute error is 1.16e-6 against an unusually
/// small tensor), and 7.6e-5 or better everywhere else, with every cosine at
/// 1.0000000000. The 1e-3 bound is 8x the worst observation — two to three
/// orders of magnitude below what any formulation error in an fp32 kernel
/// would produce, and comfortably above driver and hardware variation.
const DEVICE_VS_HOST: Gate = Gate {
    abs_over_rms: 1e-3,
    min_cosine: 1.0 - 1e-7,
};

/// End-to-end agreement with llama.cpp for the steps that are downstream of
/// at least one int8 projection.
///
/// Everything from `Qcur_full-N` onwards inherits the activation
/// quantization documented in the module docs, and the two per-head RMSNorms
/// amplify it slightly (they divide by an RMS that is itself perturbed).
/// This bound is a coarse outlier catch; **the cosine is the real gate**, and
/// the `q8_1` contrast above is what licenses the width.
///
/// Worst measured across both blocks: **0.233 of the normalizing RMS**
/// (`attn_output-3` / `attn_residual-3`) and cosine **0.9998550** — the 0.5
/// bound is 2.1x the first and the 5e-4 cosine bound is 3.4x the second. The
/// worst end-to-end block output is `attn_residual-39` at 2.118e-2 absolute,
/// 7.21e-2 of the residual stream's RMS, cosine 0.9999814.
const PROJECTION_VS_GOLDEN: Gate = Gate {
    abs_over_rms: 0.5,
    min_cosine: 1.0 - 5e-4,
};

/// End-to-end agreement for the one step upstream of every projection.
///
/// `attn_norm-N` reads the residual stream directly, so nothing about the
/// projections can reach it and it is held to the fp32 floor. Measured:
/// 7.29e-6 of RMS at worst, cosine 1.0000000000 — the device is in fact
/// *closer* to llama.cpp here than this file's own host reference is
/// (2.97e-5), because the kernel's warp-shuffle reduction tree happens to
/// track ggml's double-precision accumulation better than a sequential f32
/// sum does.
const PRE_PROJECTION_VS_GOLDEN: Gate = Gate {
    abs_over_rms: 1e-4,
    min_cosine: 1.0 - 1e-9,
};

/// A scale-free agreement bound.
#[derive(Debug, Clone, Copy)]
struct Gate {
    /// Worst absolute error, as a fraction of the reference tensor's RMS.
    abs_over_rms: f64,
    /// Minimum cosine similarity, computed in f64 — `compare()` returns an
    /// f32 cosine, which cannot represent `1 - 1e-9` distinctly from `1.0`.
    min_cosine: f64,
}

/// What one comparison measured.
#[derive(Debug, Clone, Copy)]
struct Agreement {
    max_abs: f64,
    max_abs_index: usize,
    rms_ref: f64,
    cosine: f64,
}

impl Agreement {
    fn abs_over_rms(&self) -> f64 {
        if self.rms_ref > 0.0 {
            self.max_abs / self.rms_ref
        } else {
            self.max_abs
        }
    }
}

impl std::fmt::Display for Agreement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "max_abs={:.3e} ({:.2e} x rms) cos={:.10}",
            self.max_abs,
            self.abs_over_rms(),
            self.cosine,
        )
    }
}

/// Root mean square of a tensor, in f64.
fn rms_of(x: &[f32]) -> f64 {
    (x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / x.len() as f64).sqrt()
}

fn measure(candidate: &[f32], reference: &[f32]) -> Agreement {
    assert_eq!(
        candidate.len(),
        reference.len(),
        "measure(): length mismatch — a layout error, not a numeric one",
    );
    assert!(!candidate.is_empty(), "measure(): nothing to compare");

    let mut max_abs = 0.0f64;
    let mut max_abs_index = 0usize;
    let (mut dot, mut nc, mut nr, mut sq) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (i, (&c, &r)) in candidate.iter().zip(reference).enumerate() {
        assert!(c.is_finite(), "candidate element {i} is {c}");
        let (c, r) = (f64::from(c), f64::from(r));
        let e = (c - r).abs();
        if e > max_abs {
            max_abs = e;
            max_abs_index = i;
        }
        dot += c * r;
        nc += c * c;
        nr += r * r;
        sq += r * r;
    }
    let denom = nc.sqrt() * nr.sqrt();
    Agreement {
        max_abs,
        max_abs_index,
        rms_ref: (sq / reference.len() as f64).sqrt(),
        cosine: if denom > 0.0 { dot / denom } else { 1.0 },
    }
}

/// Check one agreement, normalizing the absolute bound by `scale`.
///
/// `scale` is the reference tensor's own RMS for almost every step. It is
/// **not** for a step whose output is a product with a factor that can be
/// much smaller than one: the absolute error there is set by the magnitude of
/// the factor that carries it, not by the magnitude of the product. Block 3's
/// output gate is mostly near zero, so `attn_gated-3` has an RMS of 8.8e-3
/// while inheriting `attn_pregate-3`'s error at RMS 3.8e-1 — normalizing by
/// its own RMS would report a 118% error for a step that is doing nothing
/// wrong. Those call sites pass the RMS that actually sets the scale, and say
/// which one.
fn assert_gate(label: &str, a: &Agreement, gate: &Gate, scale: f64) {
    assert!(
        a.max_abs <= gate.abs_over_rms * scale,
        "{label}: worst absolute error is {:.3e}, {:.3e} x the scale RMS {:.3e}, \
         above the bound {:.3e} (element {})",
        a.max_abs,
        a.max_abs / scale,
        scale,
        gate.abs_over_rms,
        a.max_abs_index,
    );
    assert!(
        a.cosine >= gate.min_cosine,
        "{label}: cosine similarity {:.12} is below {:.12}",
        a.cosine,
        gate.min_cosine,
    );
}

fn model_path() -> PathBuf {
    std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
}

fn setup() -> Option<(Arc<CudaContext>, GgufFile, golden::Golden)> {
    if !driver_available() {
        println!("SKIPPED: no CUDA driver present");
        return None;
    }
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            println!("SKIPPED: could not create a context on device 0: {e}");
            return None;
        }
    };
    let info = DeviceInfo::from_context(0, &ctx).expect("device properties must be readable");
    if !info.is_supported() {
        println!(
            "SKIPPED: device 0 is {} ({}), below the sm_75 minimum",
            info.name,
            info.compute_capability.sm_arch(),
        );
        return None;
    }
    let path = model_path();
    if !path.exists() {
        println!(
            "SKIPPED: model file not found at {}; set LLMXABE_MODEL to override",
            path.display(),
        );
        return None;
    }
    let file = GgufFile::open(&path).expect("model file must parse as valid GGUF v3");
    // `golden::setup` prints its own skip reason, and panics rather than
    // skipping on a corrupt capture.
    let g = golden::setup()?;
    println!(
        "device 0: {} ({}); golden: {} records, {} tokens",
        info.name,
        info.compute_capability.sm_arch(),
        g.records().len(),
        g.n_tokens(),
    );
    Some((ctx, file, g))
}

// ---------------------------------------------------------------------------
// Host references
// ---------------------------------------------------------------------------

/// Split a token-major flat buffer into per-row vectors.
fn rows(flat: &[f32], width: usize) -> Vec<Vec<f32>> {
    flat.chunks_exact(width).map(<[f32]>::to_vec).collect()
}

/// RMSNorm over `flat.len() / width` independent rows.
fn rms_norm_rows(flat: &[f32], weight: &[f32], width: usize, eps: f32) -> Vec<f32> {
    flat.chunks_exact(width)
        .flat_map(|r| rms_norm(r, weight, eps))
        .collect()
}

/// Dequantize a whole Q8_0 tensor into a row-major `[out_dim][in_dim]` matrix.
///
/// GGUF `[A, B]` is `B` rows of `A` contiguous elements (`docs/ORACLE.md`
/// §6.1), which is exactly `gemv_batch`'s `weight` layout, so no transpose
/// happens anywhere in this file.
fn dequantize_q8_0_tensor(bytes: &[u8], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let row_bytes = in_dim / QK8_0 * BLOCK_Q8_0_BYTES;
    assert_eq!(
        bytes.len(),
        out_dim * row_bytes,
        "tensor is not {out_dim} rows of {in_dim} Q8_0 elements",
    );
    let mut out = Vec::with_capacity(out_dim * in_dim);
    for r in 0..out_dim {
        out.extend(
            dequantize_row_q8_0(&bytes[r * row_bytes..(r + 1) * row_bytes])
                .expect("a whole number of Q8_0 blocks"),
        );
    }
    out
}

/// llama.cpp's `q8_1` activation quantization, round-tripped back to f32.
///
/// `d = amax / 127` over each block of 32, `q = roundf(x / d)`, value
/// `q * d`. `roundf` is half-away-from-zero, which is `f32::round`. The
/// scale is kept in fp32: the test asserts below that this reproduces the
/// capture, and an fp16 scale — which is what the non-MMQ `quantize_q8_1`
/// stores — does **not**, by three orders of magnitude.
fn quantize_activations_q8_1(x: &[f32]) -> Vec<f32> {
    assert!(
        x.len().is_multiple_of(QK8_1),
        "activation row is not a whole number of q8_1 blocks",
    );
    let mut out = Vec::with_capacity(x.len());
    for block in x.as_chunks::<QK8_1>().0 {
        let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if amax == 0.0 {
            out.extend(std::iter::repeat_n(0.0f32, QK8_1));
            continue;
        }
        let d = amax / Q8_1_LEVELS;
        out.extend(block.iter().map(|&v| (v / d).round() * d));
    }
    out
}

/// `out[t][o] = sum_i weight[o][i] * x[t][i]`, token-major in and out.
fn project(weight: &[f32], out_dim: usize, in_dim: usize, x: &[f32]) -> Vec<f32> {
    let xs = rows(x, in_dim);
    gemv_batch(weight, out_dim, in_dim, &xs).concat()
}

/// The interleaved reading of `attn_q`'s output: `[q_h0, gate_h0, q_h1, ...]`.
fn deinterleave(packed: &[f32], q_heads: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let stride = 2 * q_heads * head_dim;
    let tokens = packed.len() / stride;
    let mut q = Vec::with_capacity(tokens * q_heads * head_dim);
    let mut gate = Vec::with_capacity(tokens * q_heads * head_dim);
    for t in 0..tokens {
        let row = &packed[t * stride..(t + 1) * stride];
        for h in 0..q_heads {
            q.extend_from_slice(&row[2 * h * head_dim..(2 * h + 1) * head_dim]);
            gate.extend_from_slice(&row[(2 * h + 1) * head_dim..(2 * h + 2) * head_dim]);
        }
    }
    (q, gate)
}

/// The wrong reading: `attn_q` split into two contiguous halves.
///
/// Kept as a function so the test can show it disagreeing rather than
/// describe it disagreeing.
fn halves_split(packed: &[f32], q_heads: usize, head_dim: usize) -> Vec<f32> {
    let stride = 2 * q_heads * head_dim;
    let tokens = packed.len() / stride;
    let mut q = Vec::with_capacity(tokens * q_heads * head_dim);
    for t in 0..tokens {
        q.extend_from_slice(&packed[t * stride..t * stride + q_heads * head_dim]);
    }
    q
}

/// Partial rotary over `[tokens][heads][head_dim]`, token `t` at position
/// `pos_offset + t`.
fn rope_rows(
    x: &[f32],
    heads: usize,
    head_dim: usize,
    rope_dim: u32,
    pos_offset: u32,
    theta: f32,
) -> Vec<f32> {
    let tokens = x.len() / (heads * head_dim);
    let mut out = Vec::with_capacity(x.len());
    for t in 0..tokens {
        for h in 0..heads {
            let base = (t * heads + h) * head_dim;
            out.extend(apply_rope(
                &x[base..base + head_dim],
                pos_offset + t as u32,
                rope_dim,
                theta,
            ));
        }
    }
    out
}

/// Causal GQA attention over one self-contained window.
fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let tokens = q.len() / (q_heads * head_dim);
    let head = |buf: &[f32], heads: usize, h: usize| -> Vec<Vec<f32>> {
        (0..tokens)
            .map(|t| {
                let base = (t * heads + h) * head_dim;
                buf[base..base + head_dim].to_vec()
            })
            .collect()
    };

    let mut out = vec![0.0f32; tokens * q_heads * head_dim];
    for h in 0..q_heads {
        let kvh = kv_head_for_query_head(h as u32, q_heads as u32, kv_heads as u32) as usize;
        let o = causal_attention_streaming(
            &head(q, q_heads, h),
            &head(k, kv_heads, kvh),
            &head(v, kv_heads, kvh),
        );
        for (t, row) in o.iter().enumerate() {
            let base = (t * q_heads + h) * head_dim;
            out[base..base + head_dim].copy_from_slice(row);
        }
    }
    out
}

/// `sigmoid(gate)` and `pregate * sigmoid(gate)`, in ggml's spelling.
fn sigmoid_gate(pregate: &[f32], gate: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let sig: Vec<f32> = gate.iter().map(|&g| 1.0 / (1.0 + (-g).exp())).collect();
    let out = pregate.iter().zip(&sig).map(|(&p, &s)| p * s).collect();
    (sig, out)
}

/// Round a buffer through fp16, to show how much of the attention gap the
/// captured run's f16 KV cache accounts for.
fn through_f16(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| f16_round_trip(v)).collect()
}

/// One fp32 -> fp16 -> fp32 round trip, round-to-nearest-even, without a
/// dependency: this crate does not pull in `half`.
fn f16_round_trip(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let bits = x.to_bits();
    let sign = bits >> 31;
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    let mant = bits & 0x007f_ffff;

    // Overflow to infinity, underflow to (sub)normal, or a normal half.
    let (half_exp, half_mant, shift) = if exp > 15 {
        return if sign == 1 {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
    } else if exp < -14 {
        // Subnormal half: the implicit 1 becomes explicit and the mantissa
        // shifts by however far below the smallest normal exponent we are.
        let extra = (-14 - exp) as u32;
        if extra > 24 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        (0i32, mant | 0x0080_0000, 13 + extra)
    } else {
        (exp + 15, mant, 13)
    };

    let round_bit = 1u32 << (shift - 1);
    let sticky_mask = round_bit - 1;
    let mut q = half_mant >> shift;
    let rem = half_mant & (round_bit | sticky_mask);
    if rem > round_bit || (rem == round_bit && (q & 1) == 1) {
        q += 1;
    }
    // A carry out of the mantissa is a clean exponent increment.
    let mut e = half_exp;
    if q > 0x3ff {
        q &= 0x3ff;
        e += 1;
        if e > 30 {
            return if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            };
        }
    }

    let magnitude = if e <= 0 {
        q as f32 * 2.0f32.powi(-24)
    } else {
        (1.0 + q as f32 / 1024.0) * 2.0f32.powi(e - 15)
    };
    if sign == 1 { -magnitude } else { magnitude }
}

// ---------------------------------------------------------------------------
// Model metadata
// ---------------------------------------------------------------------------

/// `rope.dimension_sections` as four `i32`s.
fn rope_sections(file: &GgufFile) -> [i32; 4] {
    let value = file
        .get("qwen35moe.rope.dimension_sections")
        .expect("the file must declare rope.dimension_sections");
    let list: Vec<i32> = match value {
        GgufValue::Array(xabe_gguf::GgufArray::I32(v)) => v.clone(),
        GgufValue::Array(xabe_gguf::GgufArray::U32(v)) => v.iter().map(|&x| x as i32).collect(),
        other => panic!("rope.dimension_sections is {other:?}, expected an i32/u32 array"),
    };
    assert_eq!(list.len(), 4, "rope.dimension_sections must have 4 entries");
    [list[0], list[1], list[2], list[3]]
}

/// Which of llama.cpp's four M-RoPE position channels pair `sector` draws.
///
/// Transcribed from `rope_vision`/`rope_multi`'s `is_imrope` branch in
/// `ggml/src/ggml-cuda/rope.cu`: interleaved M-RoPE assigns by `sector % 3`
/// and falls through to channel 3 when the sector runs past its class's
/// `3 * sections[c]`.
fn imrope_channel(sector: i32, sections: [i32; 4]) -> usize {
    if sector % 3 == 1 && sector < 3 * sections[1] {
        1
    } else if sector % 3 == 2 && sector < 3 * sections[2] {
        2
    } else if sector % 3 == 0 && sector < 3 * sections[0] {
        0
    } else {
        3
    }
}

// ---------------------------------------------------------------------------
// The test
// ---------------------------------------------------------------------------

/// One isolated-host-op comparison: this repository's reference for a single
/// operation, fed llama.cpp's own input for that step.
struct IsolatedStep<'a> {
    name: &'static str,
    host: Vec<f32>,
    gold: &'a [f32],
    gate: Gate,
}

/// One of the four Q8_0 projections, for the fp32-versus-`q8_1` contrast.
struct Projection<'a> {
    name: &'static str,
    /// Dequantized weight, `[out_dim][in_dim]` row-major.
    weight: &'a [f32],
    /// llama.cpp's own input to this projection.
    input: &'a [f32],
    out_dim: usize,
    in_dim: usize,
    /// llama.cpp's own output.
    gold: &'a [f32],
}

/// One step of the chained device run, against both references.
struct DeviceStep<'a> {
    name: &'static str,
    device: Vec<f32>,
    host: &'a [f32],
    gold: &'a [f32],
    gate: Gate,
    /// RMS to normalize the `dev|gold` absolute bound by, when the step's own
    /// output does not set the scale of its error. See [`assert_gate`].
    scale: Option<(f64, &'static str)>,
}

fn iso_step<'a>(
    name: &'static str,
    host: Vec<f32>,
    gold: &'a [f32],
    gate: Gate,
) -> IsolatedStep<'a> {
    IsolatedStep {
        name,
        host,
        gold,
        gate,
    }
}

fn dev_step<'a>(
    name: &'static str,
    device: Vec<f32>,
    host: &'a [f32],
    gold: &'a [f32],
    gate: Gate,
) -> DeviceStep<'a> {
    DeviceStep {
        name,
        device,
        host,
        gold,
        gate,
        scale: None,
    }
}

/// Everything one host reference pass produced.
struct HostChain {
    normed: Vec<f32>,
    packed: Vec<f32>,
    query: Vec<f32>,
    gate: Vec<f32>,
    query_normed: Vec<f32>,
    query_roped: Vec<f32>,
    key: Vec<f32>,
    key_normed: Vec<f32>,
    key_roped: Vec<f32>,
    value: Vec<f32>,
    pregate: Vec<f32>,
    gate_sigmoid: Vec<f32>,
    gated: Vec<f32>,
    projected: Vec<f32>,
    residual: Vec<f32>,
}

/// The dequantized projections and norm vectors for one layer.
struct HostWeights {
    input_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    qgate: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    out: Vec<f32>,
}

impl HostWeights {
    fn load(file: &GgufFile, config: &ModelConfig, layer: u32) -> Self {
        let hidden = config.hidden_size as usize;
        let a = config.attention;
        let head_dim = a.head_dim as usize;
        let q_dim = a.q_heads as usize * head_dim;
        let kv_dim = a.kv_heads as usize * head_dim;

        let f32_tensor = |role: Role| -> Vec<f32> {
            let name = format!("blk.{layer}.{}", role.suffix());
            let bytes = file
                .tensor_bytes(&name)
                .unwrap_or_else(|| panic!("{name} must be resident in the file"));
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(f32::from_le_bytes)
                .collect()
        };
        let q8_0_tensor = |role: Role, out_dim: usize, in_dim: usize| -> Vec<f32> {
            let name = format!("blk.{layer}.{}", role.suffix());
            let info = file
                .tensor(&name)
                .unwrap_or_else(|| panic!("{name} must be in the tensor directory"));
            assert_eq!(
                info.ggml_type,
                xabe_gguf::GgmlType::Q8_0,
                "{name} is not Q8_0; this reference dequantizes Q8_0 only",
            );
            assert_eq!(
                info.dims,
                vec![in_dim as u64, out_dim as u64],
                "{name} is not [{in_dim}, {out_dim}] in ggml order",
            );
            let bytes = file.tensor_bytes(&name).expect("tensor bytes are mapped");
            dequantize_q8_0_tensor(bytes, out_dim, in_dim)
        };

        Self {
            input_norm: f32_tensor(Role::InputNorm),
            q_norm: f32_tensor(Role::AttnQNorm),
            k_norm: f32_tensor(Role::AttnKNorm),
            qgate: q8_0_tensor(Role::AttnQGate, 2 * q_dim, hidden),
            key: q8_0_tensor(Role::AttnK, kv_dim, hidden),
            value: q8_0_tensor(Role::AttnV, kv_dim, hidden),
            out: q8_0_tensor(Role::AttnOut, hidden, q_dim),
        }
    }
}

/// Run the whole block on the host in fp32, from `hidden_state`.
fn host_chain(
    w: &HostWeights,
    config: &ModelConfig,
    hidden_state: &[f32],
    eps: f32,
    theta: f32,
) -> HostChain {
    let hidden = config.hidden_size as usize;
    let a = config.attention;
    let head_dim = a.head_dim as usize;
    let q_heads = a.q_heads as usize;
    let kv_heads = a.kv_heads as usize;
    let q_dim = q_heads * head_dim;
    let kv_dim = kv_heads * head_dim;

    let normed = rms_norm_rows(hidden_state, &w.input_norm, hidden, eps);
    let packed = project(&w.qgate, 2 * q_dim, hidden, &normed);
    let (query, gate) = deinterleave(&packed, q_heads, head_dim);
    let query_normed = rms_norm_rows(&query, &w.q_norm, head_dim, eps);
    let key = project(&w.key, kv_dim, hidden, &normed);
    let value = project(&w.value, kv_dim, hidden, &normed);
    let key_normed = rms_norm_rows(&key, &w.k_norm, head_dim, eps);
    let query_roped = rope_rows(&query_normed, q_heads, head_dim, a.rope_dim, 0, theta);
    let key_roped = rope_rows(&key_normed, kv_heads, head_dim, a.rope_dim, 0, theta);
    let pregate = attention(
        &query_roped,
        &key_roped,
        &value,
        q_heads,
        kv_heads,
        head_dim,
    );
    let (gate_sigmoid, gated) = sigmoid_gate(&pregate, &gate);
    let projected = project(&w.out, hidden, q_dim, &gated);
    let residual = hidden_state
        .iter()
        .zip(&projected)
        .map(|(&a, &b)| a + b)
        .collect();

    HostChain {
        normed,
        packed,
        query,
        gate,
        query_normed,
        query_roped,
        key,
        key_normed,
        key_roped,
        value,
        pregate,
        gate_sigmoid,
        gated,
        projected,
        residual,
    }
}

fn read(stream: &Arc<CudaStream>, buf: &CudaSlice<f32>) -> Vec<f32> {
    let v = stream.clone_dtoh(buf).expect("device read-back");
    stream.synchronize().expect("sync");
    v
}

#[test]
fn the_gated_attention_block_reproduces_every_captured_intermediate_of_blocks_3_and_39() {
    let Some((ctx, file, g)) = setup() else {
        return;
    };
    let config = ModelConfig::qwen3_6_35b_a3b();
    let a = config.attention;
    let hidden = config.hidden_size as usize;
    let head_dim = a.head_dim as usize;
    let q_heads = a.q_heads as usize;
    let kv_heads = a.kv_heads as usize;
    let q_dim = q_heads * head_dim;
    let kv_dim = kv_heads * head_dim;
    let tokens = g.n_tokens();
    let stream = ctx.default_stream();

    // ---- 0. The two hyperparameters that are in the file and not in
    //         ModelConfig, and the IMRoPE reduction. -----------------------
    let eps = file
        .get_f32("qwen35moe.attention.layer_norm_rms_epsilon")
        .expect("the file must declare the RMSNorm epsilon");
    let theta = file
        .get_f32("qwen35moe.rope.freq_base")
        .expect("the file must declare the rotary frequency base");
    let n_rot = file
        .get_u32("qwen35moe.rope.dimension_count")
        .expect("the file must declare the rotary dimension count");
    assert_eq!(
        n_rot, a.rope_dim,
        "the file rotates {n_rot} dimensions, ModelConfig says {}",
        a.rope_dim,
    );
    let sections = rope_sections(&file);
    let sect_sum: i32 = sections.iter().sum();
    assert_eq!(
        sect_sum,
        (n_rot / 2) as i32,
        "M-RoPE sections {sections:?} must sum to rope_dim/2",
    );
    // The claim the NEOX kernel rests on: no rotated pair falls through to
    // channel 3, which is the only channel llama.cpp sets to 0 for text
    // (`llm_graph_input_pos::set_input`). If one did, that pair would not be
    // rotated at all and a plain NEOX kernel would be wrong.
    let fell_through: Vec<i32> = (0..sect_sum)
        .filter(|&s| imrope_channel(s, sections) == 3)
        .collect();
    assert!(
        fell_through.is_empty(),
        "IMRoPE sections {sections:?} leave pairs {fell_through:?} on the zero-position \
         channel, so this model is NOT plain NEOX partial rotary and \
         AttentionKernels::rope is the wrong kernel for it",
    );
    println!(
        "rope: n_rot={n_rot}, freq_base={theta:.1}, sections={sections:?} -> all {sect_sum} \
         rotated pairs draw the text position, so IMRoPE == NEOX here; rms_eps={eps:.3e}",
    );

    // The layer set this module claims, checked against ModelConfig.
    let layers: Vec<u32> = attention_layers(&config).collect();
    assert_eq!(layers, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39]);
    for &l in &CAPTURED_LAYERS {
        assert_eq!(config.layer_kind(l), LayerKind::GatedAttention);
        assert!(layers.contains(&l));
    }
    println!(
        "block shape covers {} of {} layers: {layers:?}",
        layers.len(),
        config.num_layers,
    );

    // ---- 1. The model, resident. --------------------------------------
    let schema = WeightSchema::new(&config);
    let directory = schema
        .resolve(&file)
        .expect("schema must resolve against the model file");
    let started = Instant::now();
    let (weights, report) = match DeviceWeights::load(&ctx, &stream, &file, &directory) {
        Ok(pair) => pair,
        Err(e) => panic!("weight load failed: {e}"),
    };
    println!(
        "loaded {} tensors, {:.2} GiB in {:.1} s",
        report.tensors,
        report.bytes as f64 / (1u64 << 30) as f64,
        started.elapsed().as_secs_f64(),
    );

    let kernels = Arc::new(
        AttentionKernelSet::new(&ctx, &config, tokens).expect("attention kernels must compile"),
    );

    for &layer in &CAPTURED_LAYERS {
        println!("\n================ block {layer} ================");
        let hw = HostWeights::load(&file, &config, layer);

        let block_in = g.block_input(layer).f32_data.clone();
        assert_eq!(block_in.len(), tokens * hidden);

        // ---- 2. The device block. ------------------------------------
        let mut block = GatedAttentionBlock::new(
            Arc::clone(&kernels),
            &stream,
            &weights,
            &config,
            layer,
            tokens,
            eps,
            theta,
        )
        .expect("block weights resolve and scratch allocates");
        assert_eq!(block.layer(), layer);
        assert_eq!(block.tokens(), tokens);

        let d_in = stream
            .clone_htod(block_in.as_slice())
            .expect("upload input");
        let mut d_out = stream
            .alloc_zeros::<f32>(tokens * hidden)
            .expect("alloc output");
        // A cache sized to this window and written from position 0 makes the
        // pass a cold prefill, which is what the capture recorded.
        let mut cache = KvCache::new(&stream, &config, tokens).expect("kv cache allocates");
        block
            .forward(&stream, &d_in, &mut cache, 0, &mut d_out)
            .expect("block forward launches");
        stream.synchronize().expect("sync after forward");

        // ---- 3. The host chain over the same input. ------------------
        let host = host_chain(&hw, &config, &block_in, eps, theta);

        // ---- 4. Step by step. ----------------------------------------
        // Each entry: the golden tensor, the device buffer, the host chain's
        // value, the host reference of that one op fed the golden's own
        // input for it, and the gate for the `dev|gold` column.
        let gold = |name: &str| g.f32(&format!("{name}-{layer}")).to_vec();
        let gold_first = |name: &str| {
            g.all(&format!("{name}-{layer}"))
                .first()
                .expect("record present")
                .f32_data
                .clone()
        };
        let gold_second = |name: &str| {
            g.all(&format!("{name}-{layer}"))
                .get(1)
                .expect("second record present")
                .f32_data
                .clone()
        };

        let gold_norm = gold("attn_norm");
        let gold_full = gold("Qcur_full");
        let gold_q = gold("Qcur_reshaped");
        let gold_gate = gold("gate_reshaped");
        let gold_qn = gold("Qcur_normed");
        let gold_k = gold_first("Kcur");
        let gold_kn = gold("Kcur_normed");
        let gold_kr = gold_second("Kcur");
        let gold_qr = gold("Qcur");
        let gold_v = gold_first("Vcur");
        let gold_pre = gold("attn_pregate");
        let gold_sig = gold("gate_sigmoid");
        let gold_gated = gold("attn_gated");
        let gold_proj = gold("attn_output");
        let gold_res = gold("attn_residual");

        // The isolated host op for each step, fed llama.cpp's own input.
        let (iso_q, iso_gate) = deinterleave(&gold_full, q_heads, head_dim);
        let (iso_sig, iso_gated) = sigmoid_gate(&gold_pre, &gold_gate);
        let iso = vec![
            iso_step(
                "attn_norm",
                rms_norm_rows(&block_in, &hw.input_norm, hidden, eps),
                &gold_norm,
                OP_DEFINITION,
            ),
            iso_step("Qcur_reshaped", iso_q, &gold_q, OP_DEFINITION),
            iso_step("gate_reshaped", iso_gate, &gold_gate, OP_DEFINITION),
            iso_step(
                "Qcur_normed",
                rms_norm_rows(&gold_q, &hw.q_norm, head_dim, eps),
                &gold_qn,
                OP_DEFINITION,
            ),
            iso_step(
                "Kcur_normed",
                rms_norm_rows(&gold_k, &hw.k_norm, head_dim, eps),
                &gold_kn,
                OP_DEFINITION,
            ),
            iso_step(
                "Qcur(rope)",
                rope_rows(&gold_qn, q_heads, head_dim, a.rope_dim, 0, theta),
                &gold_qr,
                OP_DEFINITION,
            ),
            iso_step(
                "Kcur(rope)",
                rope_rows(&gold_kn, kv_heads, head_dim, a.rope_dim, 0, theta),
                &gold_kr,
                OP_DEFINITION,
            ),
            iso_step(
                "attn_pregate",
                attention(&gold_qr, &gold_kr, &gold_v, q_heads, kv_heads, head_dim),
                &gold_pre,
                ATTENTION_FLOOR,
            ),
            iso_step("gate_sigmoid", iso_sig, &gold_sig, OP_DEFINITION),
            iso_step("attn_gated", iso_gated, &gold_gated, OP_DEFINITION),
            iso_step(
                "attn_residual",
                block_in
                    .iter()
                    .zip(&gold_proj)
                    .map(|(&x, &y)| x + y)
                    .collect(),
                &gold_res,
                OP_DEFINITION,
            ),
        ];
        println!("-- isolated host op on llama.cpp's own input for that step --");
        for s in &iso {
            let m = measure(&s.host, s.gold);
            println!("  {:16} host|gold {m}", s.name);
            assert_gate(
                &format!("{}-{layer} host|gold", s.name),
                &m,
                &s.gate,
                m.rms_ref,
            );
        }

        // The four projections, both ways round.
        println!("-- projections: fp32 matmul vs llama.cpp's q8_1 activations --");
        let projections = [
            Projection {
                name: "Qcur_full",
                weight: &hw.qgate,
                input: &gold_norm,
                out_dim: 2 * q_dim,
                in_dim: hidden,
                gold: &gold_full,
            },
            Projection {
                name: "Kcur",
                weight: &hw.key,
                input: &gold_norm,
                out_dim: kv_dim,
                in_dim: hidden,
                gold: &gold_k,
            },
            Projection {
                name: "Vcur",
                weight: &hw.value,
                input: &gold_norm,
                out_dim: kv_dim,
                in_dim: hidden,
                gold: &gold_v,
            },
            Projection {
                name: "attn_output",
                weight: &hw.out,
                input: &gold_gated,
                out_dim: hidden,
                in_dim: q_dim,
                gold: &gold_proj,
            },
        ];
        for p in projections {
            let Projection {
                name,
                weight,
                input,
                out_dim,
                in_dim,
                gold: reference,
            } = p;
            let exact = project(weight, out_dim, in_dim, input);
            let quantized = project(weight, out_dim, in_dim, &quantize_activations_q8_1(input));
            let m_exact = measure(&exact, reference);
            let m_quant = measure(&quantized, reference);
            println!("  {name:16} fp32 {m_exact}");
            println!("  {:16} q8_1 {m_quant}", "");
            assert_gate(
                &format!("{name}-{layer} q8_1|gold"),
                &m_quant,
                &PROJECTION_Q8_1,
                m_quant.rms_ref,
            );
            // The contrast is what licenses PROJECTION_VS_GOLDEN's width. If
            // the two ever converge, the explanation is wrong and this fails.
            assert!(
                m_exact.abs_over_rms() > FP32_MUST_BE_WORSE_BY * m_quant.abs_over_rms(),
                "{name}-{layer}: an exact fp32 matmul is only {:.1}x worse than the q8_1 \
                 emulation ({:.3e} vs {:.3e} of RMS). The claim that llama.cpp quantizes \
                 its activations no longer explains the gap, so PROJECTION_VS_GOLDEN is \
                 no longer justified.",
                m_exact.abs_over_rms() / m_quant.abs_over_rms(),
                m_exact.abs_over_rms(),
                m_quant.abs_over_rms(),
            );
        }

        // How much of the attention gap the captured run's f16 KV explains.
        let f16_kv = attention(
            &gold_qr,
            &through_f16(&gold_kr),
            &through_f16(&gold_v),
            q_heads,
            kv_heads,
            head_dim,
        );
        println!(
            "  {:16} attn with f16-rounded K/V (the capture's cache dtype): {}",
            "",
            measure(&f16_kv, &gold_pre),
        );

        // ---- 5. The device, against both references. -----------------
        let dev_out = read(&stream, &d_out);
        // `attn_pregate-N`'s RMS, for the two steps whose error is inherited
        // from it rather than generated by them. See `assert_gate`.
        let pregate_rms = rms_of(&gold_pre);
        let steps = vec![
            dev_step(
                "attn_norm",
                read(&stream, block.normed_input()),
                &host.normed,
                &gold_norm,
                PRE_PROJECTION_VS_GOLDEN,
            ),
            dev_step(
                "Qcur_full",
                read(&stream, block.packed_query_gate()),
                &host.packed,
                &gold_full,
                PROJECTION_VS_GOLDEN,
            ),
            dev_step(
                "Qcur_reshaped",
                read(&stream, block.query()),
                &host.query,
                &gold_q,
                PROJECTION_VS_GOLDEN,
            ),
            dev_step(
                "gate_reshaped",
                read(&stream, block.gate()),
                &host.gate,
                &gold_gate,
                PROJECTION_VS_GOLDEN,
            ),
            dev_step(
                "Qcur_normed",
                read(&stream, block.query_normed()),
                &host.query_normed,
                &gold_qn,
                PROJECTION_VS_GOLDEN,
            ),
            dev_step(
                "Kcur",
                read(&stream, block.key()),
                &host.key,
                &gold_k,
                PROJECTION_VS_GOLDEN,
            ),
            dev_step(
                "Vcur",
                read(&stream, block.value()),
                &host.value,
                &gold_v,
                PROJECTION_VS_GOLDEN,
            ),
            dev_step(
                "Kcur_normed",
                read(&stream, block.key_normed()),
                &host.key_normed,
                &gold_kn,
                PROJECTION_VS_GOLDEN,
            ),
            dev_step(
                "Qcur(rope)",
                read(&stream, block.query_roped()),
                &host.query_roped,
                &gold_qr,
                PROJECTION_VS_GOLDEN,
            ),
            dev_step(
                "Kcur(rope)",
                read(&stream, block.key_roped()),
                &host.key_roped,
                &gold_kr,
                PROJECTION_VS_GOLDEN,
            ),
            dev_step(
                "attn_pregate",
                read(&stream, block.pregate()),
                &host.pregate,
                &gold_pre,
                PROJECTION_VS_GOLDEN,
            ),
            DeviceStep {
                // sigmoid(x) is in (0, 1) whatever x is, so an error in this
                // tensor is bounded by 1, not by its own RMS — which is
                // 2.3e-2 at block 3 because that block's gate is mostly
                // saturated toward zero.
                scale: Some((1.0, "sigmoid's unit range")),
                ..dev_step(
                    "gate_sigmoid",
                    read(&stream, block.gate_sigmoid()),
                    &host.gate_sigmoid,
                    &gold_sig,
                    PROJECTION_VS_GOLDEN,
                )
            },
            DeviceStep {
                // `attn_gated = attn_pregate * sigmoid(gate)`, and the
                // sigmoid factor is at most 1, so the error this step carries
                // is `attn_pregate`'s error — not something its own
                // (suppressed) magnitude can normalize.
                scale: Some((pregate_rms, "attn_pregate's RMS")),
                ..dev_step(
                    "attn_gated",
                    read(&stream, block.gated()),
                    &host.gated,
                    &gold_gated,
                    PROJECTION_VS_GOLDEN,
                )
            },
            DeviceStep {
                // Block 3's attention contribution is small — `attn_output-3`
                // has an RMS of 9.5e-3 against the residual stream's 3.8e-2 —
                // so its own magnitude is not the scale at which its error
                // matters. The scale that does is the stream it is added
                // into, which is the next step's tensor.
                scale: Some((
                    rms_of(&gold_res),
                    "the residual stream this block writes into",
                )),
                ..dev_step(
                    "attn_output",
                    read(&stream, block.projected()),
                    &host.projected,
                    &gold_proj,
                    PROJECTION_VS_GOLDEN,
                )
            },
            dev_step(
                "attn_residual",
                dev_out.clone(),
                &host.residual,
                &gold_res,
                PROJECTION_VS_GOLDEN,
            ),
        ];

        println!("-- device block, chained from l_out-{} --", layer - 1);
        for s in &steps {
            let vs_host = measure(&s.device, s.host);
            let vs_gold = measure(&s.device, s.gold);
            println!("  {:16} dev|host {vs_host}", s.name);
            match s.scale {
                None => println!("  {:16} dev|gold {vs_gold}", ""),
                Some((scale, why)) => println!(
                    "  {:16} dev|gold {vs_gold} [bound normalized by {why} = {scale:.3e}: \
                     {:.2e} x]",
                    "",
                    vs_gold.max_abs / scale,
                ),
            }
            assert_gate(
                &format!("{}-{layer} dev|host", s.name),
                &vs_host,
                &DEVICE_VS_HOST,
                vs_host.rms_ref,
            );
            assert_gate(
                &format!("{}-{layer} dev|gold", s.name),
                &vs_gold,
                &s.gate,
                s.scale.map_or(vs_gold.rms_ref, |(scale, _)| scale),
            );
        }

        // ---- 6. The interleave, on this repository's own output. ------
        let dev_packed = read(&stream, block.packed_query_gate());
        let dev_query = read(&stream, block.query());
        let dev_gate = read(&stream, block.gate());
        let (host_q, host_gate) = deinterleave(&dev_packed, q_heads, head_dim);
        let bad_q = dev_query
            .iter()
            .zip(&host_q)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let bad_gate = dev_gate
            .iter()
            .zip(&host_gate)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let split = halves_split(&dev_packed, q_heads, head_dim);
        let bad_split = dev_query
            .iter()
            .zip(&split)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let total = tokens * q_dim;
        println!(
            "  deinterleave over {total} elements: stride-2*head_dim mismatches q={bad_q} \
             gate={bad_gate}; halves-split mismatches {bad_split}",
        );
        assert_eq!(bad_q, 0, "the device query is not at stride 2*head_dim");
        assert_eq!(bad_gate, 0, "the device gate is not at stride 2*head_dim");
        // Head 0 coincides under both readings, so a discriminating check
        // requires every other head to disagree.
        assert_eq!(
            bad_split,
            total - tokens * head_dim,
            "a halves split of attn_q should disagree on every head but the first; \
             if it does not, this comparison proves nothing",
        );

        // ---- 7. The rotary tail, bit-exact. ---------------------------
        let dev_qn = read(&stream, block.query_normed());
        let dev_qr = read(&stream, block.query_roped());
        let dev_kn = read(&stream, block.key_normed());
        let dev_kr = read(&stream, block.key_roped());
        let rope_dim = a.rope_dim as usize;
        let tail_mismatches = |before: &[f32], after: &[f32], heads: usize| -> usize {
            let mut bad = 0;
            for row in 0..tokens * heads {
                let base = row * head_dim;
                for d in rope_dim..head_dim {
                    if before[base + d].to_bits() != after[base + d].to_bits() {
                        bad += 1;
                    }
                }
            }
            bad
        };
        let q_tail = tail_mismatches(&dev_qn, &dev_qr, q_heads);
        let k_tail = tail_mismatches(&dev_kn, &dev_kr, kv_heads);
        println!(
            "  partial rotary: dims [{rope_dim}, {head_dim}) unchanged — query mismatches \
             {q_tail}, key mismatches {k_tail}",
        );
        assert_eq!(q_tail, 0, "the query's un-rotated tail was modified");
        assert_eq!(k_tail, 0, "the key's un-rotated tail was modified");

        // And the rotated head is genuinely rotated, so the check above is
        // not passing because rope did nothing at all.
        let rotated: usize = (0..tokens * q_heads)
            .map(|row| {
                (0..rope_dim)
                    .filter(|&d| {
                        dev_qn[row * head_dim + d].to_bits() != dev_qr[row * head_dim + d].to_bits()
                    })
                    .count()
            })
            .sum();
        // Token 0 sits at position 0, where the rotation is the identity, so
        // its head is expected to come back unchanged.
        let rotatable = (tokens - 1) * q_heads * rope_dim;
        println!(
            "  and {rotated} of {} rotated dimensions changed ({rotatable} are at a \
             non-zero position)",
            tokens * q_heads * rope_dim,
        );
        assert!(
            rotated > rotatable / 2,
            "the rotary barely changed anything; it may not have run",
        );

        println!(
            "block {layer}: device output vs attn_residual-{layer} -> {}",
            measure(&dev_out, &gold_res),
        );
    }

    println!(
        "\ngates: op-definition <= {:.0e} x rms and cosine >= {:.9}; \
         device vs host fp32 <= {:.0e} x rms; device vs llama.cpp <= {:.2} x rms and \
         cosine >= {:.6}",
        OP_DEFINITION.abs_over_rms,
        OP_DEFINITION.min_cosine,
        DEVICE_VS_HOST.abs_over_rms,
        PROJECTION_VS_GOLDEN.abs_over_rms,
        PROJECTION_VS_GOLDEN.min_cosine,
    );
}

#[test]
fn the_imrope_section_mapping_matches_the_upstream_branch() {
    // Runs without a device or a model file: it is arithmetic on the section
    // widths, and it is what the device test's assertion rests on.
    //
    // Qwen3.6's [11, 11, 10, 0]: every pair in 0..32 is claimed by its own
    // `s % 3` class, so none reaches the fourth channel — the only one
    // llama.cpp zeroes for a text batch.
    let qwen = [11, 11, 10, 0];
    for s in 0..32 {
        assert_ne!(
            imrope_channel(s, qwen),
            3,
            "pair {s} fell through to the zero-position channel",
        );
        assert_eq!(imrope_channel(s, qwen), (s % 3) as usize);
    }

    // A section set that does *not* collapse, so the assertion above is
    // shown to be capable of failing. [4, 4, 4, 20] leaves everything from
    // pair 12 onward on channel 3.
    let vision = [4, 4, 4, 20];
    let fell_through: Vec<i32> = (0..32)
        .filter(|&s| imrope_channel(s, vision) == 3)
        .collect();
    assert!(
        !fell_through.is_empty(),
        "the mapping never reports a fall-through, so the model test proves nothing",
    );
    assert!(fell_through.contains(&12));
}

#[test]
fn the_q8_1_emulation_is_the_quantization_it_claims_to_be() {
    // No device, no model file. `d = amax/127`, `q = roundf(x/d)`, and the
    // reconstruction is `q * d` — so the block maximum is reproduced exactly
    // and everything else lands on a multiple of `d`.
    let block: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) * 0.37).collect();
    let out = quantize_activations_q8_1(&block);
    let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let d = amax / 127.0;

    for (&x, &y) in block.iter().zip(&out) {
        assert!(
            (y / d - (y / d).round()).abs() < 1e-3,
            "{y} is not a multiple of the block scale {d}",
        );
        assert!(
            (x - y).abs() <= d * 0.5 + 1e-6,
            "{x} quantized to {y}, further than half a step {d} away",
        );
    }
    // An all-zero block has no scale and must not divide by it.
    assert!(
        quantize_activations_q8_1(&[0.0f32; 32])
            .iter()
            .all(|&v| v == 0.0)
    );
}

#[test]
fn the_fp16_round_trip_agrees_with_known_values() {
    // Used only to show how much of the attention gap the capture's f16 KV
    // cache explains, but a wrong rounding would make that diagnosis wrong,
    // so it is checked against values with known half representations.
    assert_eq!(f16_round_trip(1.0), 1.0);
    assert_eq!(f16_round_trip(-2.5), -2.5);
    assert_eq!(f16_round_trip(0.0), 0.0);
    // 1 + 2^-11 is exactly half way between 1.0 and the next half; ties go
    // to even, which is 1.0.
    assert_eq!(f16_round_trip(1.0 + 2f32.powi(-11)), 1.0);
    // 1 + 2^-10 is representable exactly.
    assert_eq!(f16_round_trip(1.0 + 2f32.powi(-10)), 1.0 + 2f32.powi(-10));
    // Subnormal halves go down to 2^-24.
    assert_eq!(f16_round_trip(2f32.powi(-24)), 2f32.powi(-24));
    assert_eq!(f16_round_trip(2f32.powi(-30)), 0.0);
    // And the largest half is 65504; beyond it saturates to infinity.
    assert_eq!(f16_round_trip(65504.0), 65504.0);
    assert!(f16_round_trip(1e30).is_infinite());
    // A value with more mantissa than a half can hold loses the tail.
    assert!((f16_round_trip(0.1) - 0.099975586).abs() < 1e-8);
}
