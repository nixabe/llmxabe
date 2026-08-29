//! The Gated DeltaNet block against llama.cpp's own intermediates.
//!
//! `gdn_differential.rs` and `gdn_chunked_differential.rs` check the two mixer
//! kernels against a CPU reference on synthetic input. This file checks the
//! *block* — every step of `build_layer_attn_linear`, in order, on the real
//! Qwen3.6 weights — against the captured llama.cpp forward pass described in
//! `docs/ORACLE.md`. It is the difference between "the delta rule is
//! transcribed correctly" and "layer 20 produces what llama.cpp produces".
//!
//! ## What is compared
//!
//! Blocks 0, 4 and 20 are the three Gated DeltaNet layers whose internals the
//! capture holds. For each of them, every captured waypoint is compared
//! **twice**:
//!
//! 1. *Anchored* ([`each_step_matches_llama_cpp_when_fed_its_own_input`]) —
//!    each step runs on llama.cpp's own captured input for that step, so a
//!    disagreement is that step's and nothing else's.
//! 2. *Chained* ([`the_whole_block_matches_llama_cpp_token_by_token`]) — the
//!    block runs end to end from `l_out-(N-1)` and every intermediate is
//!    compared, so the numbers show where error enters and how it accumulates.
//!
//! ## Two findings this file exists to have produced
//!
//! **The query/key heads are shared by modulo, and both landed GDN kernels
//! divided.** llama.cpp's fused op indexes the query/key head as
//! `fastmodulo(h_idx, n_k_heads)` and its non-fused fallback broadcasts with
//! `ggml_repeat_4d`, which tiles; `xabe_cuda::kernels::gdn` and
//! `::gdn_chunked` both computed `h / heads_per_kv` internally.
//! [`the_query_key_head_broadcast_is_modulo_not_division`] discriminates the
//! two against the captured `final_output-N` using nothing but captured
//! tensors, and still does. Both kernels were corrected in `d4d2f4d`, so
//! `block::gdn` no longer materialises the broadcast and this file no longer
//! widens the capture's 16-head `q_conv_predelta` / `k_conv_predelta` before
//! feeding them to the mixer.
//!
//! **The chunked prefill kernel overflowed on this model's real decay rates.**
//! `gdn_chunk_solve_and_apply` formed `v_t / lambda_t` where
//! `lambda_t = exp(sum_{i<=t} log_decay_i)`. Block 0 head 9 has a per-token
//! log-decay of -91.58, so `lambda` reached 2.5e-42 by the second token and the
//! quotient was `inf`. See
//! [`the_chunked_prefill_kernel_overflows_on_this_models_decay_rates`], which
//! localizes where that would have happened and now gates the chunked form
//! against the recurrent one instead of failing. The chained end-to-end test
//! still drives the block one token at a time, because that also threads the
//! convolution cache and the recurrent state through 19 separate calls — the
//! carry a decode loop performs, and exercised nowhere else.
//!
//! ## The one large tolerance, and why it is not this block's
//!
//! `linear_attn_qkv_mixed`, `z` and `linear_attn_out` come from Q8_0 weight
//! matrices. llama.cpp's CUDA backend does not compute those in fp32: for a
//! quantized `src0` it quantizes the *activation* to Q8_1 and uses integer dot
//! products (`ggml_cuda_mul_mat` dispatches to `mul_mat_q` /
//! `mul_mat_vec_q`). This block accumulates in fp32 against dequantized
//! weights instead.
//!
//! That difference is measured rather than asserted:
//! [`the_projection_gap_is_llama_cpps_activation_quantization`] computes the
//! same projection in f64 on the host from the same dequantized weights and the
//! same captured input and compares *both* implementations to it. The f32
//! `ssm_alpha` projection, which takes llama.cpp's unquantized path, is the
//! control that rules out a wrong GEMM here.
//!
//! ## Limits of the oracle, restated so nothing is claimed that is not covered
//!
//! One prompt, 19 tokens, one ubatch, and `state_predelta-N` is all zeros. So
//! this file checks the recurrence **only from a cold start**, and never across
//! a `chunk_len = 64` boundary. A carried-in non-zero state is not exercised
//! against llama.cpp anywhere here; closing that needs a second capture over a
//! longer prompt.
//!
//! SKIPS — reporting that it skipped — without a driver, a supported device,
//! the golden capture, or the model file.

#![allow(clippy::too_many_arguments)]

use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_engine::block::gdn::{GdnBlock, GdnGeometry, GdnLayerWeights, Mixer, Projection};
use xabe_gguf::{GgmlType, GgufFile};
use xabe_kernels::compare::{ComparisonResult, Tolerance, ToleranceCheck, check, compare};
use xabe_model::config::ModelConfig;
use xabe_model::weights::{Directory, WeightSchema};

#[path = "golden.rs"]
mod golden;

/// The three Gated DeltaNet blocks whose internals the capture holds.
const LAYERS: [u32; 3] = [0, 4, 20];

/// Where the model lives, matching `golden.rs`'s own default.
const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llmxabe/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

// ---------------------------------------------------------------------------
// Gates
//
// Every bound below was set from the measurements this file prints, after they
// were taken. Each carries the measured worst case it came from, so a
// regression past it fails rather than being absorbed.
//
// `max_rel_error` is a tripwire and not the gate, for the reason
// `gdn_chunked_differential.rs` documents at length: `compare()` divides by
// `max(|reference|, 1e-6)`, and this block drives many components toward zero,
// so for those the ratio reports `abs_error / 1e-6` rather than anything about
// accuracy. `max_abs_error` and `cosine` are the gates.
// ---------------------------------------------------------------------------

/// Steps computed in fp32 on both sides from the same input, with no quantized
/// matrix in the way: the two norms, the convolution, the activations, the
/// gates, the slice, and the delta rule itself.
///
/// Measured worst, anchored, over blocks 0/4/20: `max_abs 1.144e-5` on
/// `gate-0`, whose own values reach 91.6 in magnitude — a relative error of
/// 1.2e-7. `cosine` is 1.000000 to six places on every one of them, and
/// `max_rel` peaks at 1.852e-2 on `a_softplus-4`, where the reference element
/// is near zero. The bounds are 4.4x, 1e-7 and 5.4x those.
const ANCHORED_ELEMENTWISE: Tolerance = Tolerance {
    max_abs_error: 5e-5,
    max_rel_error: 1e-1,
    min_cosine_similarity: 1.0 - 1e-7,
    allow_non_finite: false,
};

/// Steps that go through a Q8_0 weight matrix, where llama.cpp quantizes the
/// activation to Q8_1 and this block does not.
///
/// [`the_projection_gap_is_llama_cpps_activation_quantization`] is what
/// licenses this bound; without that test it would be an unexplained
/// four-order-of-magnitude loosening and would not be defensible. Measured
/// there: on `attn_qkv` this block is 1.9e-6 from an f64 reference and
/// llama.cpp is 7.1e-2 — 37,000x further; on `ssm_out` it is 2.2e-8 against
/// llama.cpp's 1.6e-3 — 73,000x further.
///
/// Measured worst, anchored, over blocks 0/4/20: `max_abs 1.002e-1` on
/// `linear_attn_qkv_mixed-0`, and a cosine of 0.999920 on
/// `linear_attn_out-0`. The bounds are 3x and 2.5x those.
const ANCHORED_QUANTIZED_PROJECTION: Tolerance = Tolerance {
    max_abs_error: 3e-1,
    max_rel_error: 1e6,
    min_cosine_similarity: 1.0 - 3e-4,
    allow_non_finite: false,
};

/// The block's output on the residual stream, run end to end, against
/// `attn_residual-N`.
///
/// Measured over blocks 0/4/20: `max_abs` 1.72e-3 / 2.03e-3 / 9.88e-3 against
/// reference peaks of 0.77 / 1.46 / 34.1, and cosine 0.999948 / 0.999996 /
/// 0.999999. The bounds are 3x and 3.8x the worst of those. Almost all of it
/// is the Q8_1 projection gap: the anchored `attn_residual` comparison, which
/// takes llama.cpp's own `linear_attn_out`, is bit-identical.
///
/// Also gated in the test against 2% of the reference tensor's own peak
/// magnitude, which is the bound that stays meaningful if the activations ever
/// change scale.
const CHAINED_BLOCK_OUTPUT: Tolerance = Tolerance {
    max_abs_error: 3e-2,
    max_rel_error: 1e6,
    min_cosine_similarity: 1.0 - 2e-4,
    allow_non_finite: false,
};

/// Prefill against decode: the same tokens through the chunked kernel and
/// through nineteen single-token recurrent calls.
///
/// Same weights, same gates, same convolution cache; only the delta-rule kernel
/// differs, so this is far tighter than anything gated against llama.cpp.
/// Measured on block 4, the one layer whose decay rates the chunked kernel
/// survives: `max_abs 5.96e-8` on the output and `4.17e-7` on the final state.
const CHUNKED_VS_RECURRENT: Tolerance = Tolerance {
    max_abs_error: 5e-5,
    max_rel_error: 1e2,
    min_cosine_similarity: 1.0 - 1e-8,
    allow_non_finite: false,
};

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

struct Fixture {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    file: GgufFile,
    golden: golden::Golden,
    config: ModelConfig,
    geometry: GdnGeometry,
}

fn model_path() -> PathBuf {
    std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
}

/// A device, the golden capture and the model file, or a reported skip.
fn setup() -> Option<Fixture> {
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
    let info = DeviceInfo::from_context(0, &ctx).expect("device properties readable");
    if !info.is_supported() {
        println!("SKIPPED: device 0 is below the sm_75 minimum");
        return None;
    }
    let g = golden::setup()?;
    let path = model_path();
    if !path.exists() {
        println!(
            "SKIPPED: model file not found at {}; set LLMXABE_MODEL to override",
            path.display(),
        );
        return None;
    }
    let file = GgufFile::open(&path).expect("model file must parse as valid GGUF v3");
    let config = ModelConfig::qwen3_6_35b_a3b();
    let geometry = GdnGeometry::from_gguf(&config, &file, g.n_tokens())
        .expect("the file must carry qwen35moe.attention.layer_norm_rms_epsilon");
    let stream = ctx.default_stream();

    println!(
        "device 0: {}, golden {} records / {} tokens, rms_eps {:e}",
        info.name,
        g.records().len(),
        g.n_tokens(),
        geometry.rms_eps,
    );
    Some(Fixture {
        ctx,
        stream,
        file,
        golden: g,
        config,
        geometry,
    })
}

/// The resolved weight directory.
///
/// The schema is leaked so the borrow outlives this call; it is a few hundred
/// KiB and the process is a test binary.
fn directory<'a>(file: &'a GgufFile, config: &ModelConfig) -> Directory<'a> {
    let schema: &'static WeightSchema = Box::leak(Box::new(WeightSchema::new(config)));
    schema.resolve(file).expect("schema resolves")
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

fn dtoh(stream: &Arc<CudaStream>, buf: &CudaSlice<f32>) -> Vec<f32> {
    let out = stream.clone_dtoh(buf).expect("device read back");
    stream.synchronize().expect("sync");
    out
}

fn htod(stream: &Arc<CudaStream>, values: &[f32]) -> CudaSlice<f32> {
    stream.clone_htod(values).expect("upload")
}

/// One row of a GGUF matrix, dequantized on the host.
///
/// `dims[0]` is the input width and one row is `dims[0]` contiguous elements —
/// the convention `docs/ORACLE.md` section 6.1 proves against this
/// repository's own reader.
fn weight_row(file: &GgufFile, name: &str, row: usize) -> Vec<f32> {
    let info = file.tensor(name).expect("tensor is in the directory");
    let k = info.dims[0] as usize;
    let bytes = file.tensor_bytes(name).expect("tensor bytes are mapped");
    match info.ggml_type {
        GgmlType::Q8_0 => {
            let row_bytes = k / xabe_kernels::quant::QK8_0 * xabe_kernels::quant::BLOCK_Q8_0_BYTES;
            xabe_kernels::quant::dequantize_row_q8_0(&bytes[row * row_bytes..(row + 1) * row_bytes])
                .expect("a whole number of Q8_0 blocks")
        }
        GgmlType::F32 => bytes[row * k * 4..(row + 1) * k * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(f32::from_le_bytes)
            .collect(),
        other => panic!(
            "{name} is {}, which this test does not unpack",
            other.name()
        ),
    }
}

/// `ggml_l2_norm`: `scale = 1 / max(sqrt(sum(x^2)), eps)`.
///
/// Note this is **not** `1 / sqrt(sum + eps)`. ggml's CPU form is
/// `1.0f/fmaxf(sqrtf(sum), eps)` and its CUDA form
/// `rsqrtf(fmaxf(sum, eps*eps))` — the epsilon floors the norm rather than
/// being added under the root, and at `eps = 1e-6` it never binds. Both GDN
/// kernels use `1/sqrt(sum + 1e-6)` instead;
/// [`the_mixer_kernels_l2_epsilon_is_a_different_formula_than_ggmls`] measures
/// what that costs on this data.
fn l2_norm_ggml(x: &[f32]) -> Vec<f32> {
    let sum: f32 = x.iter().map(|v| v * v).sum();
    let scale = 1.0 / sum.sqrt().max(1e-6);
    x.iter().map(|v| v * scale).collect()
}

/// `ggml_rms_norm` followed by the weight multiply, over one row.
fn rms_norm_ggml(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mean: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean + eps).sqrt();
    x.iter().zip(weight).map(|(v, w)| v * scale * w).collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Print a comparison and hand it back.
fn report(step: &str, candidate: &[f32], reference: &[f32]) -> ComparisonResult {
    let r = compare(candidate, reference);
    println!("    {step:<26} {r}");
    r
}

/// Assert a comparison against a gate, naming the step in the failure.
fn gate(step: &str, r: &ComparisonResult, tolerance: &Tolerance) {
    if let ToleranceCheck::Fail { reason, .. } = check(r, tolerance) {
        panic!("{step}: {reason}\n  full metrics: {r}");
    }
}

// A host `broadcast_heads` used to live here, to widen the capture's 16-head
// `q_conv_predelta` / `k_conv_predelta` to 32 heads for a mixer that could not
// do the mapping itself. Both kernels index `h % qk_heads` now, so nothing in
// this file broadcasts and the head map is discriminated only where it belongs
// — inside `delta_rule_host`, against `final_output-N`.

// ---------------------------------------------------------------------------
// A host transcription of llama.cpp's fused gated delta net
// ---------------------------------------------------------------------------

/// `gated_delta_net_cuda`, non-KDA branch, transcribed operand for operand.
///
/// `q`/`k` are `[tokens][qk_heads][head_dim]`, already L2-normalized — the
/// captured `q_conv_predelta` / `k_conv_predelta`. `v` is
/// `[tokens][value_heads][head_dim]`. `log_decay` and `beta` are
/// `[tokens][value_heads]`. The state starts at zero, which is what
/// `state_predelta` holds for a fresh sequence.
///
/// This exists to answer one question numerically that no synthetic input can
/// answer cheaply: whether value head `h` reads query/key head `h % qk_heads`
/// or `h / (value_heads / qk_heads)`.
fn delta_rule_host(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    log_decay: &[f32],
    beta: &[f32],
    tokens: usize,
    qk_heads: usize,
    value_heads: usize,
    head_dim: usize,
    modulo: bool,
) -> Vec<f32> {
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    // state[h][col][i] — `col` is the value index and `i` the key index, the
    // layout the fused kernel documents as `M[col][i] = S[i][col]`.
    let mut state = vec![0.0f32; value_heads * head_dim * head_dim];
    let mut out = vec![0.0f32; tokens * value_heads * head_dim];

    for t in 0..tokens {
        for h in 0..value_heads {
            let hq = if modulo {
                h % qk_heads
            } else {
                h / (value_heads / qk_heads)
            };
            let qk_base = (t * qk_heads + hq) * head_dim;
            let v_base = (t * value_heads + h) * head_dim;
            let g = log_decay[t * value_heads + h].exp();
            let b = beta[t * value_heads + h];

            for col in 0..head_dim {
                let row = (h * head_dim + col) * head_dim;
                let mut kv = 0.0f32;
                for i in 0..head_dim {
                    kv += state[row + i] * k[qk_base + i];
                }
                let delta = (v[v_base + col] - g * kv) * b;
                let mut attn = 0.0f32;
                for i in 0..head_dim {
                    let s = g * state[row + i] + k[qk_base + i] * delta;
                    state[row + i] = s;
                    attn += s * q[qk_base + i];
                }
                out[v_base + col] = attn * scale;
            }
        }
    }
    out
}

/// `build_norm_gated`: `ssm_norm(core) * silu(z)`, per head.
fn norm_and_gate_host(
    core: &[f32],
    z: &[f32],
    ssm_norm: &[f32],
    tokens: usize,
    value_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; tokens * value_heads * head_dim];
    for t in 0..tokens {
        for h in 0..value_heads {
            let base = (t * value_heads + h) * head_dim;
            let normed = rms_norm_ggml(&core[base..base + head_dim], ssm_norm, eps);
            for (d, n) in normed.iter().enumerate() {
                out[base + d] = n * silu(z[base + d]);
            }
        }
    }
    out
}

/// Drive the mixer one token at a time over a whole sequence.
///
/// The recurrent kernel is a decode kernel: it advances the state by exactly
/// one token. Running a sequence through it is what a decode loop does, and it
/// is what anchors the chunked form, which now also runs on this model's decay
/// rates.
///
/// `q` and `k` are `[tokens][qk_heads][head_dim]`, `v` is
/// `[tokens][value_heads][head_dim]`.
fn mix_recurrently(
    block: &mut GdnBlock,
    stream: &Arc<CudaStream>,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    log_decay: &[f32],
    beta: &[f32],
    tokens: usize,
) -> Vec<f32> {
    let geo = block.geometry();
    let wide = geo.value_dim();
    let narrow = geo.key_dim();
    let heads = geo.value_heads;
    assert_eq!(
        q.len(),
        tokens * narrow,
        "q is [tokens][qk_heads][head_dim]"
    );
    assert_eq!(
        k.len(),
        tokens * narrow,
        "k is [tokens][qk_heads][head_dim]"
    );
    let mut state = block.state(stream).expect("state allocates");
    let mut out = Vec::with_capacity(tokens * wide);
    for t in 0..tokens {
        let d_q = htod(stream, &q[t * narrow..(t + 1) * narrow]);
        let d_k = htod(stream, &k[t * narrow..(t + 1) * narrow]);
        let d_v = htod(stream, &v[t * wide..(t + 1) * wide]);
        let d_g = htod(stream, &log_decay[t * heads..(t + 1) * heads]);
        let d_b = htod(stream, &beta[t * heads..(t + 1) * heads]);
        let mut d_out = stream.alloc_zeros::<f32>(wide).expect("mixer out");
        let mixer = block
            .mix(
                stream, &mut state, &d_q, &d_k, &d_v, &d_g, &d_b, &mut d_out, 1,
            )
            .expect("mix");
        assert_eq!(mixer, Mixer::Recurrent);
        out.extend_from_slice(&dtoh(stream, &d_out));
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// The one claim no synthetic input settles: which query/key head a value head
/// reads.
///
/// llama.cpp's fused op indexes it as `fastmodulo(h_idx, n_k_heads)` and its
/// non-fused fallback broadcasts with `ggml_repeat_4d`, which tiles — so value
/// head 17 reads query/key head 1, not 8. Both landed GDN kernels implemented
/// `h / heads_per_kv` until `d4d2f4d`; `block::gdn` materialised the broadcast
/// itself in the meantime, and now delegates it.
///
/// The discrimination runs entirely on captured tensors: golden
/// `q_conv_predelta` / `k_conv_predelta` / `v_conv_predelta`, golden `gate` and
/// `beta_sigmoid`, a zero initial state, then golden `ssm_norm` and golden `z`,
/// compared against golden `final_output`. Nothing from this repository is in
/// the loop except the mapping under test.
#[test]
fn the_query_key_head_broadcast_is_modulo_not_division() {
    let Some(fx) = setup() else { return };
    let g = &fx.golden;
    let geo = fx.geometry;
    let tokens = g.n_tokens();

    for layer in LAYERS {
        let q = g.f32(&format!("q_conv_predelta-{layer}"));
        let k = g.f32(&format!("k_conv_predelta-{layer}"));
        let v = g.f32(&format!("v_conv_predelta-{layer}"));
        let decay = g.f32(&format!("gate-{layer}"));
        let beta = g.f32(&format!("beta_sigmoid-{layer}"));
        let z = g.f32(&format!("z-{layer}"));
        let reference = g.f32(&format!("final_output-{layer}"));
        let ssm_norm = weight_row(&fx.file, &format!("blk.{layer}.ssm_norm.weight"), 0);

        // A fresh sequence starts from a zeroed state; assert that rather than
        // assume it, because the whole recurrence hangs off it.
        assert!(
            g.f32(&format!("state_predelta-{layer}"))
                .iter()
                .all(|&x| x == 0.0),
            "block {layer} did not start from a zero recurrent state",
        );

        println!("block {layer}: head-broadcast discrimination on captured tensors");
        let mut results = Vec::new();
        for modulo in [true, false] {
            let core = delta_rule_host(
                q,
                k,
                v,
                decay,
                beta,
                tokens,
                geo.qk_heads,
                geo.value_heads,
                geo.head_dim,
                modulo,
            );
            let final_output = norm_and_gate_host(
                &core,
                z,
                &ssm_norm,
                tokens,
                geo.value_heads,
                geo.head_dim,
                geo.rms_eps,
            );
            let label = if modulo {
                "h % qk_heads (ggml)"
            } else {
                "h / heads_per_kv"
            };
            results.push(report(label, &final_output, reference));
        }

        let (modulo, division) = (&results[0], &results[1]);
        assert!(
            modulo.max_abs_error < 1e-5,
            "block {layer}: the modulo mapping does not reproduce final_output: {modulo}",
        );
        // The check is only meaningful if the alternative is visibly wrong.
        assert!(
            division.max_abs_error > 1000.0 * modulo.max_abs_error,
            "block {layer}: the two mappings are indistinguishable here, so this \
             test proves nothing: modulo {modulo}, division {division}",
        );
    }
}

/// Whether this block's fp32 projection or llama.cpp's integer one is the
/// implementation that departs from exact arithmetic.
///
/// `ggml_cuda_mul_mat` sends a Q8_0 `src0` down `mul_mat_q` / `mul_mat_vec_q`,
/// which quantize the fp32 activation to Q8_1 before the integer dot product.
/// This block accumulates in fp32 against dequantized weights. The two
/// therefore cannot agree to fp32 round-off, and the interesting question is
/// which of them is closer to the answer.
///
/// Both are compared against the same projection computed in f64 on the host
/// from the same dequantized Q8_0 weights and llama.cpp's own captured
/// `attn_norm-N`. The f32-weighted `ssm_alpha` projection, which takes
/// llama.cpp's unquantized path, is the control: if *it* disagreed the fault
/// would be this block's GEMM rather than the quantization.
#[test]
fn the_projection_gap_is_llama_cpps_activation_quantization() {
    let Some(fx) = setup() else { return };
    let dir = directory(&fx.file, &fx.config);
    let geo = fx.geometry;
    let tokens = fx.golden.n_tokens();
    let layer = LAYERS[0];

    let block = GdnBlock::new(&fx.ctx, geo).expect("kernels compile");
    let weights =
        GdnLayerWeights::upload(&fx.stream, &fx.file, &dir, layer).expect("weights upload");

    // llama.cpp's own input to the projection.
    let attn_norm = fx.golden.f32(&format!("attn_norm-{layer}")).to_vec();
    let d_norm = htod(&fx.stream, &attn_norm);

    let mut d_qkv = fx
        .stream
        .alloc_zeros::<f32>(tokens * geo.conv_dim())
        .expect("qkv buffer");
    block
        .project(
            &fx.stream,
            Projection::Q8_0(&weights.qkv),
            &d_norm,
            &mut d_qkv,
            geo.hidden,
            geo.conv_dim(),
            tokens,
        )
        .expect("qkv projection");
    let device_qkv = dtoh(&fx.stream, &d_qkv);
    let golden_qkv = fx.golden.f32(&format!("linear_attn_qkv_mixed-{layer}"));

    // A sample of output rows spanning the q, k and v thirds of the fused
    // stream. All 8,192 rows in f64 on the host would be 319M multiply-adds in
    // a debug build for no extra information.
    let key_dim = geo.qk_heads * geo.head_dim;
    let sample: Vec<usize> = (0..128)
        .chain(key_dim..key_dim + 128)
        .chain(2 * key_dim..2 * key_dim + 128)
        .collect();

    let (mut ours, mut theirs, mut exact) = (Vec::new(), Vec::new(), Vec::new());
    for &n in &sample {
        let row = weight_row(&fx.file, &format!("blk.{layer}.attn_qkv.weight"), n);
        for t in 0..tokens {
            let x = &attn_norm[t * geo.hidden..(t + 1) * geo.hidden];
            let acc: f64 = row
                .iter()
                .zip(x)
                .map(|(&w, &v)| f64::from(w) * f64::from(v))
                .sum();
            exact.push(acc as f32);
            ours.push(device_qkv[t * geo.conv_dim() + n]);
            theirs.push(golden_qkv[t * geo.conv_dim() + n]);
        }
    }

    println!(
        "block {layer} attn_qkv projection, {} sampled elements",
        exact.len()
    );
    let ours_vs_exact = report("this block vs f64", &ours, &exact);
    let theirs_vs_exact = report("llama.cpp vs f64", &theirs, &exact);
    let ours_vs_theirs = report("this block vs llama.cpp", &ours, &theirs);

    // The same measurement on `ssm_out`, which is where the gap is largest:
    // its input is `final_output`, whose silu gate gives it a much wider
    // dynamic range than `attn_norm` has, and Q8_1's per-32-element scale
    // therefore costs more there.
    let final_output = fx.golden.f32(&format!("final_output-{layer}")).to_vec();
    let d_final = htod(&fx.stream, &final_output);
    let mut d_proj = fx
        .stream
        .alloc_zeros::<f32>(tokens * geo.hidden)
        .expect("out buffer");
    block
        .project(
            &fx.stream,
            Projection::Q8_0(&weights.out),
            &d_final,
            &mut d_proj,
            geo.value_dim(),
            geo.hidden,
            tokens,
        )
        .expect("out projection");
    let device_out = dtoh(&fx.stream, &d_proj);
    let golden_out = fx.golden.f32(&format!("linear_attn_out-{layer}"));

    let (mut ours2, mut theirs2, mut exact2) = (Vec::new(), Vec::new(), Vec::new());
    for n in 0..128usize {
        let row = weight_row(&fx.file, &format!("blk.{layer}.ssm_out.weight"), n);
        for t in 0..tokens {
            let x = &final_output[t * geo.value_dim()..(t + 1) * geo.value_dim()];
            let acc: f64 = row
                .iter()
                .zip(x)
                .map(|(&w, &v)| f64::from(w) * f64::from(v))
                .sum();
            exact2.push(acc as f32);
            ours2.push(device_out[t * geo.hidden + n]);
            theirs2.push(golden_out[t * geo.hidden + n]);
        }
    }
    println!(
        "block {layer} ssm_out projection, {} sampled elements",
        exact2.len()
    );
    let ours_out = report("this block vs f64", &ours2, &exact2);
    let theirs_out = report("llama.cpp vs f64", &theirs2, &exact2);

    // The control: a weight matrix llama.cpp does not route through its
    // integer path. It is f32 in this file — the dense `qwen35` sibling
    // stores the same tensor Q8_0, which is what `as_projection` is for.
    let mut d_alpha = fx
        .stream
        .alloc_zeros::<f32>(tokens * geo.value_heads)
        .expect("alpha buffer");
    block
        .project(
            &fx.stream,
            weights.alpha.as_projection(),
            &d_norm,
            &mut d_alpha,
            geo.hidden,
            geo.value_heads,
            tokens,
        )
        .expect("alpha projection");
    let device_alpha = dtoh(&fx.stream, &d_alpha);
    let golden_alpha = fx.golden.f32(&format!("alpha-{layer}"));
    println!("block {layer} ssm_alpha projection (f32 weights, llama.cpp's unquantized path)");
    let alpha = report("this block vs llama.cpp", &device_alpha, golden_alpha);

    assert!(
        ours_vs_exact.max_abs_error < theirs_vs_exact.max_abs_error,
        "this block is further from the f64 reference than llama.cpp is, so the \
         projection gap is this block's after all: ours {ours_vs_exact}, theirs \
         {theirs_vs_exact}",
    );
    assert!(
        theirs_vs_exact.max_abs_error > 100.0 * ours_vs_exact.max_abs_error,
        "llama.cpp is no longer meaningfully further from exact than this block; \
         the attribution above needs re-measuring: ours {ours_vs_exact}, theirs \
         {theirs_vs_exact}",
    );
    // The two implementations' disagreement is essentially all of llama.cpp's
    // own departure from exact arithmetic, which is what "the gap is theirs"
    // means quantitatively.
    assert!(
        ours_vs_theirs.max_abs_error < 2.0 * theirs_vs_exact.max_abs_error,
        "the disagreement exceeds llama.cpp's own error, so something else is \
         contributing: {ours_vs_theirs} vs {theirs_vs_exact}",
    );
    assert!(
        theirs_out.max_abs_error > 100.0 * ours_out.max_abs_error,
        "the ssm_out gap is no longer attributable to llama.cpp's activation \
         quantization: ours {ours_out}, theirs {theirs_out}",
    );
    // And the unquantized control agrees far better, which is what rules out a
    // wrong GEMM in this block.
    assert!(
        alpha.max_abs_error < 1e-4,
        "the f32-weight projection disagrees too: this is a GEMM fault, not a \
         quantization one: {alpha}",
    );
}

/// Every captured step, run on llama.cpp's own input for that step.
///
/// This is the localizing test: each comparison depends on exactly one step's
/// arithmetic, so a failure names the step rather than the block.
#[test]
fn each_step_matches_llama_cpp_when_fed_its_own_input() {
    let Some(fx) = setup() else { return };
    let dir = directory(&fx.file, &fx.config);
    let geo = fx.geometry;
    let tokens = fx.golden.n_tokens();
    let stream = fx.stream.clone();
    let mut block = GdnBlock::new(&fx.ctx, geo).expect("kernels compile");

    for layer in LAYERS {
        let weights =
            GdnLayerWeights::upload(&stream, &fx.file, &dir, layer).expect("weights upload");
        let mut state = block.state(&stream).expect("state allocates");
        println!("block {layer}: anchored, one step at a time");

        let named = |tag: &str| format!("{tag}-{layer}");
        let gold = |tag: &str| fx.golden.f32(&named(tag));

        // --- 1. attn_norm-N, from l_out-(N-1) --------------------------------
        let input = fx.golden.block_input(layer).f32_data.clone();
        let d_input = htod(&stream, &input);
        let mut d_norm = stream
            .alloc_zeros::<f32>(tokens * geo.hidden)
            .expect("norm buffer");
        block
            .layer_ops()
            .rms_norm(
                &stream,
                &d_input,
                &weights.input_norm,
                &mut d_norm,
                tokens,
                geo.hidden,
                geo.rms_eps,
            )
            .expect("rms_norm");
        let r = report("attn_norm", &dtoh(&stream, &d_norm), gold("attn_norm"));
        gate("attn_norm", &r, &ANCHORED_ELEMENTWISE);

        // --- 2. linear_attn_qkv_mixed-N and z-N, from attn_norm-N ------------
        let d_gold_norm = htod(&stream, gold("attn_norm"));
        let mut d_qkv = stream
            .alloc_zeros::<f32>(tokens * geo.conv_dim())
            .expect("qkv");
        block
            .project(
                &stream,
                Projection::Q8_0(&weights.qkv),
                &d_gold_norm,
                &mut d_qkv,
                geo.hidden,
                geo.conv_dim(),
                tokens,
            )
            .expect("qkv projection");
        let r = report(
            "linear_attn_qkv_mixed",
            &dtoh(&stream, &d_qkv),
            gold("linear_attn_qkv_mixed"),
        );
        gate("linear_attn_qkv_mixed", &r, &ANCHORED_QUANTIZED_PROJECTION);

        let mut d_z = stream
            .alloc_zeros::<f32>(tokens * geo.value_dim())
            .expect("z");
        block
            .project(
                &stream,
                Projection::Q8_0(&weights.gate),
                &d_gold_norm,
                &mut d_z,
                geo.hidden,
                geo.value_dim(),
                tokens,
            )
            .expect("z projection");
        let r = report("z", &dtoh(&stream, &d_z), gold("z"));
        gate("z", &r, &ANCHORED_QUANTIZED_PROJECTION);

        // --- 3/4. conv_output_raw-N and conv_output_silu-N -------------------
        let d_gold_qkv = htod(&stream, gold("linear_attn_qkv_mixed"));
        let mut d_conv = stream
            .alloc_zeros::<f32>(tokens * geo.conv_dim())
            .expect("conv");
        block
            .layer_ops()
            .conv1d(
                &stream,
                &d_gold_qkv,
                &weights.conv1d,
                &mut state.conv,
                &mut d_conv,
                tokens,
                geo.conv_dim(),
                geo.conv_kernel,
            )
            .expect("conv1d");
        let r = report(
            "conv_output_raw",
            &dtoh(&stream, &d_conv),
            gold("conv_output_raw"),
        );
        gate("conv_output_raw", &r, &ANCHORED_ELEMENTWISE);

        let d_gold_conv = htod(&stream, gold("conv_output_raw"));
        let mut d_silu = stream
            .alloc_zeros::<f32>(tokens * geo.conv_dim())
            .expect("silu");
        block
            .silu(&stream, &d_gold_conv, &mut d_silu, tokens * geo.conv_dim())
            .expect("silu");
        let r = report(
            "conv_output_silu",
            &dtoh(&stream, &d_silu),
            gold("conv_output_silu"),
        );
        gate("conv_output_silu", &r, &ANCHORED_ELEMENTWISE);

        // --- 5. the q/k/v slice, and the L2 norm the mixer folds in ----------
        let d_gold_silu = htod(&stream, gold("conv_output_silu"));
        let wide = tokens * geo.value_dim();
        let narrow = tokens * geo.key_dim();
        let mut d_q = stream.alloc_zeros::<f32>(narrow).expect("q");
        let mut d_k = stream.alloc_zeros::<f32>(narrow).expect("k");
        let mut d_v = stream.alloc_zeros::<f32>(wide).expect("v");
        block
            .split_qkv(&stream, &d_gold_silu, &mut d_q, &mut d_k, &mut d_v, tokens)
            .expect("split");

        // v is a plain slice with no arithmetic, so it must be bit-identical.
        let r = report(
            "v_conv_predelta",
            &dtoh(&stream, &d_v),
            gold("v_conv_predelta"),
        );
        gate("v_conv_predelta", &r, &Tolerance::exact());

        // q and k leave the block un-normalized — the mixer kernels normalize
        // internally — so the comparison against `*_conv_predelta` applies
        // ggml's own l2 norm on the host, per (token, query/key head). The
        // split emits exactly the capture's `[head_dim, qk_heads, tokens]`, so
        // there is nothing to broadcast on either side.
        for (tag, buf) in [("q_conv_predelta", &d_q), ("k_conv_predelta", &d_k)] {
            let raw = dtoh(&stream, buf);
            assert_eq!(raw.len(), narrow);
            let mut normed = vec![0.0f32; narrow];
            for chunk in 0..tokens * geo.qk_heads {
                let base = chunk * geo.head_dim;
                normed[base..base + geo.head_dim]
                    .copy_from_slice(&l2_norm_ggml(&raw[base..base + geo.head_dim]));
            }
            let r = report(tag, &normed, fx.golden.f32(&named(tag)));
            gate(tag, &r, &ANCHORED_ELEMENTWISE);
        }

        // --- 6. the gates ----------------------------------------------------
        let mut d_alpha = stream
            .alloc_zeros::<f32>(tokens * geo.value_heads)
            .expect("alpha");
        let mut d_beta_raw = stream
            .alloc_zeros::<f32>(tokens * geo.value_heads)
            .expect("beta");
        block
            .project(
                &stream,
                weights.alpha.as_projection(),
                &d_gold_norm,
                &mut d_alpha,
                geo.hidden,
                geo.value_heads,
                tokens,
            )
            .expect("alpha projection");
        block
            .project(
                &stream,
                weights.beta.as_projection(),
                &d_gold_norm,
                &mut d_beta_raw,
                geo.hidden,
                geo.value_heads,
                tokens,
            )
            .expect("beta projection");
        let r = report("alpha", &dtoh(&stream, &d_alpha), gold("alpha"));
        gate("alpha", &r, &ANCHORED_ELEMENTWISE);
        let r = report("beta", &dtoh(&stream, &d_beta_raw), gold("beta"));
        gate("beta", &r, &ANCHORED_ELEMENTWISE);

        let d_gold_alpha = htod(&stream, gold("alpha"));
        let d_gold_beta = htod(&stream, gold("beta"));
        let n_gates = tokens * geo.value_heads;
        let mut d_softplus = stream.alloc_zeros::<f32>(n_gates).expect("softplus");
        let mut d_decay = stream.alloc_zeros::<f32>(n_gates).expect("decay");
        let mut d_beta = stream.alloc_zeros::<f32>(n_gates).expect("beta_sigmoid");
        block
            .gates(
                &stream,
                &d_gold_alpha,
                &d_gold_beta,
                &weights.dt_bias,
                &weights.a,
                &mut d_softplus,
                &mut d_decay,
                &mut d_beta,
                tokens,
            )
            .expect("gates");
        let r = report(
            "a_softplus",
            &dtoh(&stream, &d_softplus),
            gold("a_softplus"),
        );
        gate("a_softplus", &r, &ANCHORED_ELEMENTWISE);
        let r = report("gate (log-decay)", &dtoh(&stream, &d_decay), gold("gate"));
        gate("gate", &r, &ANCHORED_ELEMENTWISE);
        let r = report(
            "beta_sigmoid",
            &dtoh(&stream, &d_beta),
            gold("beta_sigmoid"),
        );
        gate("beta_sigmoid", &r, &ANCHORED_ELEMENTWISE);

        // --- 7/8. the delta rule, then final_output-N ------------------------
        //
        // Fed llama.cpp's already-L2-normalized q/k, at their own 16 heads —
        // the kernels pair value head `h` with query/key head `h % 16`
        // themselves. The mixer normalizes again, which is idempotent to about
        // 5e-7 on a unit vector, and folds in the `1/sqrt(head_dim)` output
        // scale that llama.cpp applies at the end instead.
        let core = mix_recurrently(
            &mut block,
            &stream,
            gold("q_conv_predelta"),
            gold("k_conv_predelta"),
            gold("v_conv_predelta"),
            gold("gate"),
            gold("beta_sigmoid"),
            tokens,
        );

        let d_core = htod(&stream, &core);
        let mut d_core_norm = stream.alloc_zeros::<f32>(wide).expect("core_norm");
        block
            .layer_ops()
            .rms_norm(
                &stream,
                &d_core,
                &weights.ssm_norm,
                &mut d_core_norm,
                tokens * geo.value_heads,
                geo.head_dim,
                geo.rms_eps,
            )
            .expect("ssm_norm");
        let d_gold_z = htod(&stream, gold("z"));
        let mut d_final = stream.alloc_zeros::<f32>(wide).expect("final");
        block
            .layer_ops()
            .swiglu(&stream, &d_gold_z, &d_core_norm, &mut d_final, wide)
            .expect("swiglu");
        let r = report(
            "final_output",
            &dtoh(&stream, &d_final),
            gold("final_output"),
        );
        gate("final_output", &r, &ANCHORED_ELEMENTWISE);

        // --- 9/10. linear_attn_out-N and attn_residual-N ---------------------
        let d_gold_final = htod(&stream, gold("final_output"));
        let mut d_out = stream.alloc_zeros::<f32>(tokens * geo.hidden).expect("out");
        block
            .project(
                &stream,
                Projection::Q8_0(&weights.out),
                &d_gold_final,
                &mut d_out,
                geo.value_dim(),
                geo.hidden,
                tokens,
            )
            .expect("out projection");
        let r = report(
            "linear_attn_out",
            &dtoh(&stream, &d_out),
            gold("linear_attn_out"),
        );
        gate("linear_attn_out", &r, &ANCHORED_QUANTIZED_PROJECTION);

        let d_gold_mix = htod(&stream, gold("linear_attn_out"));
        let mut d_residual = stream
            .alloc_zeros::<f32>(tokens * geo.hidden)
            .expect("residual");
        block
            .layer_ops()
            .add(
                &stream,
                &d_gold_mix,
                &d_input,
                &mut d_residual,
                tokens * geo.hidden,
            )
            .expect("residual add");
        let r = report(
            "attn_residual",
            &dtoh(&stream, &d_residual),
            gold("attn_residual"),
        );
        // Two fp32 additions of the same two operands. Nothing may round
        // differently, so this is the one step gated on exact equality.
        gate("attn_residual", &r, &Tolerance::exact());
    }
}

/// The whole block, end to end, on the real weights for the golden prompt.
///
/// Driven one token at a time — the decode path — because
/// [`the_chunked_prefill_kernel_overflows_on_this_models_decay_rates`] shows
/// the prefill kernel cannot run this data. Every waypoint is reported so drift
/// is visible where it enters; the gate is on the block's output against
/// `attn_residual-N`.
///
/// Running the prompt token by token also threads the convolution cache and the
/// recurrent state through 19 separate calls, which is exactly the carry a
/// decode loop performs and is not exercised anywhere else.
#[test]
fn the_whole_block_matches_llama_cpp_token_by_token() {
    let Some(fx) = setup() else { return };
    let dir = directory(&fx.file, &fx.config);
    let geo = fx.geometry;
    let tokens = fx.golden.n_tokens();
    let stream = fx.stream.clone();
    let mut block = GdnBlock::new(&fx.ctx, geo).expect("kernels compile");

    // Every captured waypoint, in graph order, in the order the trace exposes
    // them.
    const TAGS: [&str; 13] = [
        "attn_norm",
        "linear_attn_qkv_mixed",
        "conv_output_raw",
        "conv_output_silu",
        "v_conv_predelta",
        "z",
        "alpha",
        "a_softplus",
        "gate",
        "beta",
        "beta_sigmoid",
        "final_output",
        "linear_attn_out",
    ];

    for layer in LAYERS {
        let weights =
            GdnLayerWeights::upload(&stream, &fx.file, &dir, layer).expect("weights upload");
        let mut state = block.state(&stream).expect("state allocates");
        let input = fx.golden.block_input(layer).f32_data.clone();

        let mut collected: Vec<Vec<f32>> = vec![Vec::new(); TAGS.len()];
        let mut output = Vec::with_capacity(tokens * geo.hidden);

        for t in 0..tokens {
            let d_token = htod(&stream, &input[t * geo.hidden..(t + 1) * geo.hidden]);
            let mut d_out = stream.alloc_zeros::<f32>(geo.hidden).expect("out");
            let mixer = block
                .forward(&stream, &weights, None, &mut state, &d_token, &mut d_out)
                .expect("forward");
            assert_eq!(
                mixer,
                Mixer::Recurrent,
                "a one-token batch must take the decode form",
            );
            output.extend_from_slice(&dtoh(&stream, &d_out));

            let trace = block.trace().expect("a forward ran");
            for (slot, buf) in [
                trace.attn_norm,
                trace.qkv_mixed,
                trace.conv_raw,
                trace.conv_silu,
                trace.v,
                trace.z,
                trace.alpha,
                trace.a_softplus,
                trace.log_decay,
                trace.beta_raw,
                trace.beta,
                trace.final_output,
                trace.linear_attn_out,
            ]
            .into_iter()
            .enumerate()
            {
                collected[slot].extend_from_slice(&dtoh(&stream, buf));
            }
        }

        println!(
            "block {layer}: chained from {}, {tokens} tokens one at a time (decode form)",
            fx.golden.block_input(layer).name,
        );
        for (slot, tag) in TAGS.iter().enumerate() {
            report(
                tag,
                &collected[slot],
                fx.golden.f32(&format!("{tag}-{layer}")),
            );
        }

        let reference = fx.golden.f32(&format!("attn_residual-{layer}"));
        let r = report("attn_residual", &output, reference);
        gate(
            &format!("block {layer} attn_residual"),
            &r,
            &CHAINED_BLOCK_OUTPUT,
        );

        // The residual stream is what the next block reads, so state its
        // magnitude alongside the error rather than leaving the reader to guess
        // whether 1e-1 is large.
        let max_ref = reference.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let rms = (reference.iter().map(|v| v * v).sum::<f32>() / reference.len() as f32).sqrt();
        println!("    reference magnitude:       max |x| = {max_ref:.4e}, rms = {rms:.4e}");
        assert!(
            r.max_abs_error < 0.02 * max_ref,
            "block {layer}: the worst error is more than 2% of the tensor's own \
             peak magnitude ({:.4e} vs {max_ref:.4e})",
            r.max_abs_error,
        );
    }
}

/// **A defect in a landed kernel, localized.**
///
/// `gdn_chunk_solve_and_apply` in `crates/xabe-cuda/src/kernels/gdn_chunked.rs`
/// forms the right-hand side of its triangular system as
///
/// ```text
/// W_t      = beta_t * (v_t / lambda_t - S_in k_t),
/// lambda_t = exp(sum_{i<=t} log_decay_i)
/// ```
///
/// The division is the problem. `lambda` is a *cumulative* decay, so it shrinks
/// monotonically across the chunk, and Qwen3.6's real per-token log-decays are
/// far more negative than any synthetic input used to gate that kernel:
///
/// ```text
/// block  0: min per-token log-decay -91.578  -> lambda 2.5e-42 by token 1 (head 9)
/// block  4: min per-token log-decay  -5.988  -> no overflow over 19 tokens
/// block 20: min per-token log-decay -12.436  -> lambda 1.3e-41 by token 9 (head 7)
/// ```
///
/// `|v|` reaches 17.3, so `v / lambda` reaches 2.8e42 — past `f32::MAX` — and
/// the chunk fills with `inf`, then `NaN`. At `chunk_len = 64` rather than 19
/// tokens this gets worse, not better.
///
/// llama.cpp is immune twice over: its fused CUDA op is the recurrent form and
/// never accumulates a decay at all, and its non-fused chunked form
/// (`build_delta_net_chunking`) only ever *multiplies* by `exp(g_cum)` and
/// `exp(g_cum_last - g_cum)`, both bounded above by 1 because the log-decays
/// are negative.
///
/// This is not a tolerance question — the output is not finite — so there is
/// nothing to widen. The test is what a fix has to make pass.
#[test]
fn the_chunked_prefill_kernel_overflows_on_this_models_decay_rates() {
    let Some(fx) = setup() else { return };
    let dir = directory(&fx.file, &fx.config);
    let geo = fx.geometry;
    let tokens = fx.golden.n_tokens();
    let stream = fx.stream.clone();
    let mut block = GdnBlock::new(&fx.ctx, geo).expect("kernels compile");

    let mut failures = Vec::new();
    for layer in LAYERS {
        let weights =
            GdnLayerWeights::upload(&stream, &fx.file, &dir, layer).expect("weights upload");
        let input = fx.golden.block_input(layer).f32_data.clone();
        let reference = fx.golden.f32(&format!("attn_residual-{layer}"));

        // Where the cumulative decay first drives `v / lambda` past f32::MAX,
        // computed from llama.cpp's own captured `gate-N` and
        // `v_conv_predelta-N` so the diagnosis does not depend on this
        // repository being right about anything.
        let decay = fx.golden.f32(&format!("gate-{layer}"));
        let v = fx.golden.f32(&format!("v_conv_predelta-{layer}"));
        let mut first_overflow = None;
        let mut min_step = f32::INFINITY;
        for h in 0..geo.value_heads {
            let mut cumulative = 0.0f32;
            for t in 0..tokens {
                let step = decay[t * geo.value_heads + h];
                min_step = min_step.min(step);
                cumulative += step;
                let lambda = cumulative.exp();
                let base = (t * geo.value_heads + h) * geo.head_dim;
                let peak = v[base..base + geo.head_dim]
                    .iter()
                    .fold(0.0f32, |m, x| m.max(x.abs()));
                if first_overflow.is_none() && peak > 0.0 && !(peak / lambda).is_finite() {
                    first_overflow = Some((t, h, cumulative, lambda, peak));
                }
            }
        }
        let where_ = match first_overflow {
            Some((t, h, c, lambda, peak)) => format!(
                "at token {t} head {h} (cumulative log-decay {c:.3}, lambda {lambda:.3e}, \
                 max |v_t| {peak:.3})"
            ),
            None => "none over these 19 tokens".to_string(),
        };
        println!(
            "block {layer}: min per-token log-decay {min_step:.3}, first v/lambda overflow {where_}",
        );

        // Prefill: 19 tokens in one chunked call.
        let mut chunked_state = block.state(&stream).expect("state");
        let d_input = htod(&stream, &input);
        let mut d_chunked = stream.alloc_zeros::<f32>(tokens * geo.hidden).expect("out");
        let mixer = block
            .forward(
                &stream,
                &weights,
                None,
                &mut chunked_state,
                &d_input,
                &mut d_chunked,
            )
            .expect("chunked forward");
        assert_eq!(
            mixer,
            Mixer::Chunked,
            "19 tokens must take the prefill form"
        );
        let chunked = dtoh(&stream, &d_chunked);

        // Decode: the same 19 tokens, one at a time.
        let mut recurrent_state = block.state(&stream).expect("state");
        let mut recurrent = Vec::with_capacity(tokens * geo.hidden);
        for t in 0..tokens {
            let d_token = htod(&stream, &input[t * geo.hidden..(t + 1) * geo.hidden]);
            let mut d_out = stream.alloc_zeros::<f32>(geo.hidden).expect("out");
            block
                .forward(
                    &stream,
                    &weights,
                    None,
                    &mut recurrent_state,
                    &d_token,
                    &mut d_out,
                )
                .expect("recurrent forward");
            recurrent.extend_from_slice(&dtoh(&stream, &d_out));
        }

        // The convolution cache is advanced by the same kernel either way, once
        // over 19 tokens or nineteen times over one, so it must land on the
        // same three carried inputs bit for bit regardless of the mixer. This
        // half passes, which is what narrows the defect to the mixer.
        let r = report(
            "convolution cache",
            &dtoh(&stream, &chunked_state.conv),
            &dtoh(&stream, &recurrent_state.conv),
        );
        gate(
            &format!("block {layer} convolution cache"),
            &r,
            &Tolerance::exact(),
        );

        let vs_recurrent = report("chunked vs recurrent", &chunked, &recurrent);
        let vs_golden = report("chunked vs llama.cpp", &chunked, reference);
        let state = report(
            "final state, chunked vs rec.",
            &dtoh(&stream, &chunked_state.recurrent),
            &dtoh(&stream, &recurrent_state.recurrent),
        );

        // Each comparison against the bound that is right for it: the two
        // internal-consistency checks against CHUNKED_VS_RECURRENT, and the
        // one against llama.cpp against the same bound the recurrent path
        // clears in `the_whole_block_matches_llama_cpp_token_by_token`. Using
        // one flat threshold would have flagged block 4, which is perfectly
        // consistent and only carries the usual Q8_1 projection gap.
        for (what, r, tol) in [
            ("output vs recurrent", &vs_recurrent, &CHUNKED_VS_RECURRENT),
            ("output vs llama.cpp", &vs_golden, &CHAINED_BLOCK_OUTPUT),
            ("final state vs recurrent", &state, &CHUNKED_VS_RECURRENT),
        ] {
            if let ToleranceCheck::Fail { reason, .. } = check(r, tol) {
                failures.push(format!(
                    "  block {layer} chunked {what}: {reason}\n      {r}\n      overflow {where_}"
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "the chunked prefill kernel does not reproduce the recurrent form on this \
         model's real decay rates.\n\n{}\n\n\
         Where to look: the chunked form is only usable here because it solves \
         for u'_t = lambda_t u_t rather than u_t, so no cumulative decay is ever \
         divided by. Qwen3.6's per-token log-decays reach -91.58 (block 0), so \
         lambda underflows to a subnormal within two tokens and any quotient by \
         it exceeds f32::MAX. `gdn_chunked.rs` carries the derivation and a \
         source-level test forbidding the division; a non-finite result here \
         means the substitution was undone or a new one was introduced.\n\
         The synthetic activations gdn_chunked_differential.rs uses never reach \
         those decay rates, so its own differential test would not catch it.\n\
         llama.cpp is immune in both of its forms: the fused CUDA op is recurrent \
         and never accumulates a decay, and build_delta_net_chunking only multiplies \
         by exp(g_cum) and exp(g_cum_last - g_cum), both bounded above by 1.",
        failures.join("\n"),
    );
}

/// What the mixer kernels' L2 epsilon costs, measured on real activations.
///
/// `ggml_l2_norm` scales by `1 / max(sqrt(sum), eps)` — the epsilon floors the
/// norm and never binds. Both GDN kernels use `1 / sqrt(sum + 1e-6)` instead,
/// which is a different formula, not a different constant. To first order the
/// resulting relative difference in the scale is `eps / (2 * sum)`, so it grows
/// as the activations shrink.
///
/// Recorded here as a measured cost of a landed kernel's choice rather than
/// left to be rediscovered downstream: it is small, but it is systematic
/// shrinkage of the smallest-norm q and k vectors, not round-off.
#[test]
fn the_mixer_kernels_l2_epsilon_is_a_different_formula_than_ggmls() {
    let Some(fx) = setup() else { return };
    let geo = fx.geometry;
    let tokens = fx.golden.n_tokens();

    let mut worst_rel = 0.0f32;
    let mut smallest_sum = f32::INFINITY;
    for layer in LAYERS {
        for tag in ["q_conv", "k_conv"] {
            let raw = fx.golden.f32(&format!("{tag}-{layer}"));
            for chunk in 0..tokens * geo.qk_heads {
                let base = chunk * geo.head_dim;
                let x = &raw[base..base + geo.head_dim];
                let sum: f32 = x.iter().map(|v| v * v).sum();
                smallest_sum = smallest_sum.min(sum);
                // The two scales, as each side computes them.
                let ggml = 1.0 / sum.sqrt().max(1e-6);
                let kernel = 1.0 / (sum + 1e-6).sqrt();
                worst_rel = worst_rel.max(((ggml - kernel) / ggml).abs());
            }
        }
    }
    let predicted = 1e-6 / (2.0 * smallest_sum);
    println!(
        "l2 norm over blocks {LAYERS:?}: smallest sum of squares {smallest_sum:.4e}, \
         worst relative scale difference between ggml's 1/sqrt(sum) and the kernels' \
         1/sqrt(sum + 1e-6) = {worst_rel:.4e} (first-order prediction eps/(2*sum) = \
         {predicted:.4e})",
    );

    // The mechanism, asserted rather than described: if the measured difference
    // stopped tracking `eps / (2 * sum)` the explanation above would be wrong.
    assert!(
        (worst_rel - predicted).abs() < 0.05 * predicted,
        "the difference no longer matches eps/(2*sum): measured {worst_rel:.4e}, \
         predicted {predicted:.4e}",
    );
    // And a ceiling, so a future layer whose q or k came close to the epsilon
    // would say so here rather than three steps downstream.
    assert!(
        worst_rel < 5e-4,
        "the mixer kernels' L2 epsilon has started to bind: worst relative scale \
         difference {worst_rel:.4e} on a smallest sum of squares of {smallest_sum:.4e}",
    );
}

/// The capture's own limits, asserted so nothing above claims more than it
/// covers.
#[test]
fn the_oracle_does_not_cover_a_carried_in_state_or_a_chunk_boundary() {
    let Some(fx) = setup() else { return };
    let geo = fx.geometry;

    // 19 tokens is one ragged chunk, so no chunk-to-chunk state threading is
    // exercised anywhere in this file.
    assert!(
        fx.golden.n_tokens() < geo.chunk_len,
        "the prompt now crosses a chunk boundary; the limits documented in this \
         file and in docs/ORACLE.md section 9 are stale",
    );
    // And every captured GDN block starts from a zero state.
    for layer in LAYERS {
        assert!(
            fx.golden
                .f32(&format!("state_predelta-{layer}"))
                .iter()
                .all(|&v| v == 0.0),
        );
    }
    println!(
        "oracle limits hold: {} tokens < chunk_len {}, all three captured GDN \
         blocks start from a zero recurrent state, so nothing here checks a \
         prefix-cache resume",
        fx.golden.n_tokens(),
        geo.chunk_len,
    );
}
