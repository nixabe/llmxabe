//! Differential test: the device layer ops against the scalar references, at
//! the real Qwen3.6 geometry.
//!
//! These kernels are the glue G006 (full forward pass) needs and the
//! three big kernels do not provide: RMSNorm, partial rotary embedding,
//! SwiGLU, sigmoid gating, the residual add, softplus, and the Gated DeltaNet
//! short causal convolution.
//!
//! ## What is compared
//!
//! - `xabe_kernels::norm::rms_norm` at all three widths one forward pass uses
//!   — hidden 2048, attention head_dim 256 (`attn_q_norm` / `attn_k_norm`),
//!   and GDN head_dim 128 (`ssm_norm`).
//! - `xabe_kernels::rope::apply_rope` at head_dim 256 / rope_dim 64, with the
//!   untouched tail checked **separately and exactly**.
//! - `xabe_kernels::norm::swiglu` over a million elements.
//! - `xabe_kernels::norm::sigmoid_gate` in **both** shapes the model uses: the
//!   attention output gate, elementwise over `[tokens][4096]`, and the MoE
//!   shared expert's gate, one scalar per token broadcast over hidden 2048.
//! - `xabe_kernels::norm::residual_add` at hidden 2048, **exactly**.
//! - `xabe_kernels::norm::softplus` over a range that straddles ggml's `x > 20`
//!   passthrough and reaches past the point where an unguarded `exp` overflows.
//! - `xabe_kernels::conv::causal_depthwise_conv1d` at the real 8192-channel,
//!   4-tap geometry, batched and streamed one token at a time, plus the
//!   convolution cache, plus a causality probe run on the device.
//!
//! ## Which metric gates, and why
//!
//! `compare()` computes relative error as `|c - r| / max(|r|, 1e-6)`, so on
//! any tensor with near-zero elements `max_rel_error` degenerates into
//! `abs_error / 1e-6` and reports nothing about accuracy. That is the trap
//! `gdn_differential.rs` documents, and it applies to exactly one of the
//! kernels here — RoPE, whose rotation produces genuine cancellation. The
//! others were **measured** first and turned out to have informative
//! relative errors, so they are gated on all three metrics rather than
//! inheriting a loose `max_rel_error` they do not need. Each tolerance below
//! quotes the number it was set against.
//!
//! Softplus is the one case where `max_rel_error` is informative *and* large
//! (`7.9e-6` against the sigmoid gate's `2.7e-7`), for a third reason that is
//! neither the floor nor a kernel defect: ggml's `log(1 + exp(x))` carries the
//! answer in the low bits of a number near 1 for mid-negative `x`, so one ulp
//! of that intermediate is eight parts per million of the result. Both sides
//! suffer it identically; the gate quotes it and asserts the driving element's
//! absolute error separately so the explanation stays falsifiable.
//!
//! **None of these is a matmul.** llama.cpp's CUDA matmuls quantize their
//! activations to q8_1, which makes a loose tolerance defensible *there*;
//! these are elementwise or a single reduction against an exact fp32
//! reference, and their disagreement is one ulp of `expf`/`logf`. The gates
//! below are set from the measurement, not inherited.
//!
//! Three are gated at **exact equality** instead, which is a much
//! stronger statement than any tolerance:
//!
//! - the rotary tail, because dimensions `[rope_dim, head_dim)` are copied
//!   rather than computed;
//! - the whole convolution, because four taps in ascending order is the
//!   reference's exact operand sequence and the kernel spells the
//!   accumulation with `__fadd_rn`/`__fmul_rn` so nvcc cannot contract it
//!   into an FMA;
//! - the residual add, because a lone `a + b` has one rounding and nothing to
//!   contract.
//!
//! SKIPS — reporting that it skipped — without a driver or a supported
//! device. It needs no model file: the geometry comes from `ModelConfig` and
//! the activations are synthetic, because there is no captured Qwen3.6
//! activation to compare against.

use std::sync::Arc;

use cudarc::driver::CudaContext;
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_cuda::kernels::layer_ops::{GateShape, LayerOpsKernels};
use xabe_kernels::compare::{Tolerance, assert_matches, compare};
use xabe_kernels::conv::{causal_depthwise_conv1d, gdn_conv_channels};
use xabe_kernels::norm::{residual_add, rms_norm, sigmoid, sigmoid_gate, softplus, swiglu};
use xabe_kernels::rng::Xorshift64Star;
use xabe_kernels::rope::apply_rope;
use xabe_model::config::ModelConfig;

/// RMSNorm: the device reduces 2048 squares in a warp-shuffle tree where
/// `xabe_kernels::norm::rms_norm` sums them sequentially, and fp32 addition
/// is not associative. That is the only difference between the two, so the
/// disagreement is bounded reassociation noise.
///
/// **All three metrics gate.** Measured at the widest and therefore worst
/// case (hidden 2048, 64 rows): `max_abs = 3.099e-6`, `max_rel = 1.331e-6` on
/// a reference value of `2.911e-1`, cosine `1.000000000`. The relative error
/// is informative here — its driving element is nowhere near the 1e-6 floor —
/// so leaving `max_rel_error` at the 5e-2 that `gdn_differential.rs` needs
/// would be throwing away a check this kernel passes comfortably. 1e-4 is 75x
/// the measured worst case; 1e-5 absolute is 3.2x it.
const RMS_GATE: Tolerance = Tolerance {
    max_abs_error: 1e-5,
    max_rel_error: 1e-4,
    min_cosine_similarity: 1.0 - 1e-7,
    allow_non_finite: false,
};

/// SwiGLU: elementwise, so there is no reduction and no reassociation. The
/// entire disagreement is `expf` against `f32::exp`, about an ulp.
///
/// **All three metrics gate.** Measured over 1,048,576 elements with `gate`
/// spanning both saturating limbs of the sigmoid: `max_abs = 2.861e-6`,
/// `max_rel = 3.153e-7` on a reference value of `1.772e-2`, cosine
/// `1.000000000`. The absolute error scales with the output magnitude (up to
/// ~36 here, from `silu(12) * 3`), which is why 1e-5 absolute is the looser
/// of the two bounds despite being 3.5x the measurement.
const SWIGLU_GATE: Tolerance = Tolerance {
    max_abs_error: 1e-5,
    max_rel_error: 1e-4,
    min_cosine_similarity: 1.0 - 1e-7,
    allow_non_finite: false,
};

/// The sigmoid gate: elementwise, no reduction, so the entire disagreement is
/// `expf` against `f32::exp` propagated through one multiply.
///
/// **All three metrics gate, at the fp32 rounding floor.** This is not a
/// matmul — there is no q8_1 activation quantization anywhere near it — so it
/// does not get a matmul-era tolerance. Worst measurement over both shapes,
/// 393,216 elements and 262,208 gate values in total:
///
/// | tensor | max_abs | max_rel | on reference | cosine |
/// |---|---|---|---|---|
/// | attention gate, elementwise, 64x4096 | `4.768e-7` | `2.703e-7` | `2.692e-5` | `1.000000000` |
/// | its sigmoid, 262,144 values | `1.192e-7` | `2.297e-7` | `2.534e-4` | `1.000000000` |
/// | MoE shared gate, per-row, 64x2048 | `2.980e-8` | `2.384e-7` | `9.768e-4` | `1.000000000` |
/// | its sigmoid, 64 values | `7.451e-9` | `1.228e-7` | `2.962e-5` | `1.000000000` |
///
/// `4.768e-7` is one ulp of the largest output (`|x| <= 3`), which is the
/// floor: `sigmoid(g) * x` rounds once after a sigmoid that itself agrees to
/// an ulp. `5e-6` is 10.5x that and `3e-6` is 11.1x the measured relative
/// error, whose driving reference values (`2.7e-5`, `9.8e-4`) are 27x and
/// 977x `compare()`'s `1e-6` floor and therefore informative.
const SIGMOID_GATE: Tolerance = Tolerance {
    max_abs_error: 5e-6,
    max_rel_error: 3e-6,
    min_cosine_similarity: 1.0 - 1e-7,
    allow_non_finite: false,
};

/// Softplus: elementwise, so again no reassociation. `logf(1.0f + expf(x))`
/// against `(1.0 + x.exp()).ln()` is two libm calls' worth of disagreement,
/// and above `x = 20` both sides take the passthrough and agree bit for bit.
///
/// **All three metrics gate.** Measured over 16,384 elements spanning
/// `x in [-40, 30]` plus the branch probes: `max_abs = 9.537e-7`,
/// `max_rel = 7.941e-6` on a reference value of `1.466e-2`, cosine
/// `1.000000000`. `1e-5` is 10.5x the absolute error and `1e-4` is 12.6x the
/// relative one.
///
/// **`max_rel_error` is two orders looser than the sigmoid gate's, and not
/// because of `compare()`'s `1e-6` floor** — the driving reference value is
/// `1.466e-2`, four orders above it. It is ggml's formula: for `x` around
/// `-4`, `exp(x)` is `~0.0148` and `1 + exp(x)` is `~1.0148`, so the quantity
/// the logarithm actually needs is carried in the low bits of a number near 1
/// and one ulp *of that sum* (`1.19e-7`) is `8e-6` *of the result*. Both sides
/// suffer that amplification identically; what is left is whether their `expf`
/// results straddle the same rounding boundary. The test asserts that
/// interpretation via [`SOFTPLUS_MAX_REL_DRIVER_ABS_ERROR`] rather than
/// asserting it in prose.
const SOFTPLUS_GATE: Tolerance = Tolerance {
    max_abs_error: 1e-5,
    max_rel_error: 1e-4,
    min_cosine_similarity: 1.0 - 1e-7,
    allow_non_finite: false,
};

/// The absolute error permitted at the element driving softplus's
/// `max_rel_error`.
///
/// This is what makes the looser `SOFTPLUS_GATE.max_rel_error` honest: the
/// ratio is large because `1 + exp(x)` throws away the result's leading bits,
/// not because the kernel is wrong. Measured worst case is `1.164e-7` — one
/// ulp of the intermediate sum, exactly as predicted. `1e-6` is 8.6x that and
/// 10x below `SOFTPLUS_GATE.max_abs_error`, so a kernel whose error is
/// concentrated on the small-output region cannot satisfy both.
const SOFTPLUS_MAX_REL_DRIVER_ABS_ERROR: f32 = 1e-6;

/// The value `sig` is pre-filled with before a sigmoid-gate launch.
///
/// A sigmoid is in `(0, 1)`, so `-1` is a value the kernel cannot produce. In
/// [`GateShape::PerRow`] only one thread per row writes `sig`, and this is what
/// proves that guard selected exactly one writer per row rather than none for
/// some of them — a check `alloc_zeros` could not support, because 0 is what a
/// deeply negative gate legitimately rounds to.
const SIG_SENTINEL: f32 = -1.0;

/// RoPE, rotated span only — the tail is gated with [`Tolerance::exact`].
///
/// **`max_abs_error` and cosine gate; `max_rel_error` deliberately does not.**
/// This is the one kernel here where the relative error is uninformative, and
/// not because of `compare()`'s 1e-6 floor: `x0*cos - x1*sin` is a genuine
/// subtractive cancellation, so an output can be four orders of magnitude
/// smaller than its two inputs while carrying their rounding error. Measured:
/// `max_abs = 1.192e-7`, but `max_rel = 4.756e-4` on a reference value of
/// `1.524e-5` — an absolute error of about `7e-9` on an element whose inputs
/// are O(1).
///
/// The test asserts that interpretation rather than asserting it in prose:
/// the absolute error at the element driving `max_rel_error` must stay below
/// `1e-6`, so if the ratio ever grows because the *error* grew rather than
/// because the denominator shrank, this fails loudly.
const ROPE_GATE: Tolerance = Tolerance {
    max_abs_error: 1e-5,
    max_rel_error: 5e-2,
    min_cosine_similarity: 1.0 - 1e-7,
    allow_non_finite: false,
};

/// The absolute error permitted at the element driving RoPE's `max_rel_error`.
///
/// This is what makes the loose `ROPE_GATE.max_rel_error` honest. Measured
/// worst case is ~7.25e-9; 1e-6 is ~140x that and still 10x below
/// `ROPE_GATE.max_abs_error`, so the two bounds cannot both be satisfied by a
/// kernel whose error is concentrated on small elements.
const ROPE_MAX_REL_DRIVER_ABS_ERROR: f32 = 1e-6;

/// The RMS epsilon. `ModelConfig` carries no epsilon field, and llama.cpp
/// reads `f_norm_rms_eps` from the file; 1e-6 is the value the Qwen3 family
/// ships and is what `xabe_kernels`' own tests use. Whatever the loader ends
/// up reading, host and device get the same number here, which is what this
/// test is measuring.
const RMS_EPS: f32 = 1e-6;

/// RoPE frequency base, matching `xabe_kernels::rope`'s own tests.
const THETA_BASE: f32 = 10_000.0;

fn setup() -> Option<Arc<CudaContext>> {
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
    Some(ctx)
}

#[test]
fn device_rms_norm_matches_the_reference_at_every_width_a_forward_pass_uses() {
    let Some(ctx) = setup() else { return };
    let config = ModelConfig::qwen3_6_35b_a3b();
    let stream = ctx.default_stream();
    let kernels = LayerOpsKernels::new(&ctx).expect("kernels must compile");

    // The three distinct widths, each with the row count it actually runs at:
    //   hidden 2048   — every layer's input and post-mixer norm, and the
    //                   final output norm; one row per token.
    //   head_dim 256  — attention attn_q_norm / attn_k_norm; one row per
    //                   (token, q head).
    //   head_dim 128  — the GDN output norm ssm_norm; one row per
    //                   (token, value head).
    const TOKENS: usize = 64;
    let cases: [(&str, usize, usize); 3] = [
        ("hidden", TOKENS, config.hidden_size as usize),
        (
            "attn q/k norm",
            TOKENS * config.attention.q_heads as usize,
            config.attention.head_dim as usize,
        ),
        (
            "gdn output norm",
            TOKENS * config.gdn.value_heads as usize,
            config.gdn.head_dim as usize,
        ),
    ];

    for (name, rows, width) in cases {
        let mut rng = Xorshift64Star::new(0x2222_0000 + width as u64);
        let x = rng.vec_f32(rows * width, -4.0, 4.0);
        // Real RMSNorm weights sit near 1, not near 0; a weight vector centred
        // on zero would shrink every output and flatter max_abs_error.
        let weight: Vec<f32> = rng.vec_f32(width, 0.5, 1.5);

        let reference: Vec<f32> = (0..rows)
            .flat_map(|r| rms_norm(&x[r * width..(r + 1) * width], &weight, RMS_EPS))
            .collect();

        let d_x = stream.clone_htod(&x).expect("upload x");
        let d_w = stream.clone_htod(&weight).expect("upload weight");
        let mut d_out = stream.alloc_zeros::<f32>(rows * width).expect("alloc out");
        kernels
            .rms_norm(&stream, &d_x, &d_w, &mut d_out, rows, width, RMS_EPS)
            .expect("rms_norm launches");
        let candidate = stream.clone_dtoh(&d_out).expect("read back");
        stream.synchronize().expect("sync");

        let result = compare(&candidate, &reference);
        println!(
            "rms_norm {name}: {rows} rows x {width} -> max_abs={:.3e} cosine={:.9} \
             max_rel={:.3e} on reference {:.3e}",
            result.max_abs_error,
            result.cosine_similarity,
            result.max_rel_error,
            reference[result.max_rel_error_index].abs(),
        );

        // All three metrics, including the relative error — see RMS_GATE.
        assert_matches(&candidate, &reference, &RMS_GATE);
    }

    println!(
        "gate: max_abs<{:.0e}, max_rel<{:.0e}, cosine>{:.9}",
        RMS_GATE.max_abs_error, RMS_GATE.max_rel_error, RMS_GATE.min_cosine_similarity,
    );
}

#[test]
fn device_rms_norm_writes_in_place_correctly() {
    // The API documents that `out` may alias `x`, because the reduction's
    // final __syncthreads() separates every read from every write. The engine
    // will use that to avoid a second hidden-size buffer per layer, so it has
    // to be true rather than plausible.
    let Some(ctx) = setup() else { return };
    let stream = ctx.default_stream();
    let kernels = LayerOpsKernels::new(&ctx).expect("kernels must compile");

    let rows = 8usize;
    let width = 2048usize;
    let mut rng = Xorshift64Star::new(0x3333);
    let x = rng.vec_f32(rows * width, -4.0, 4.0);
    let weight = rng.vec_f32(width, 0.5, 1.5);

    let mut d_x = stream.clone_htod(&x).expect("upload");
    let d_w = stream.clone_htod(&weight).expect("upload weight");
    let d_x_ro = d_x.clone();
    kernels
        .rms_norm(&stream, &d_x_ro, &d_w, &mut d_x, rows, width, RMS_EPS)
        .expect("in-place rms_norm launches");
    let candidate = stream.clone_dtoh(&d_x).expect("read back");
    stream.synchronize().expect("sync");

    let reference: Vec<f32> = (0..rows)
        .flat_map(|r| rms_norm(&x[r * width..(r + 1) * width], &weight, RMS_EPS))
        .collect();
    let result = compare(&candidate, &reference);
    println!("rms_norm in place: max_abs={:.3e}", result.max_abs_error);
    assert_matches(&candidate, &reference, &RMS_GATE);
}

#[test]
fn device_rope_matches_the_reference_and_passes_the_tail_through_bit_exactly() {
    let Some(ctx) = setup() else { return };
    let config = ModelConfig::qwen3_6_35b_a3b();
    let attn = config.attention;
    let head_dim = attn.head_dim as usize;
    let rope_dim = attn.rope_dim as usize;
    let heads = attn.q_heads as usize;
    const TOKENS: usize = 128;

    // Positions well away from zero: at position 0 every rotation is the
    // identity and a kernel that ignored `positions` entirely would pass.
    const POS_BASE: u32 = 100_000;

    println!(
        "rope: head_dim={head_dim}, rope_dim={rope_dim} ({} of {head_dim} rotated), \
         {heads} heads x {TOKENS} tokens, positions {POS_BASE}..",
        rope_dim,
    );

    let stream = ctx.default_stream();
    let kernels = LayerOpsKernels::new(&ctx).expect("kernels must compile");

    let mut rng = Xorshift64Star::new(0x4444);
    let x = rng.vec_f32(TOKENS * heads * head_dim, -1.0, 1.0);
    let positions: Vec<u32> = (0..TOKENS as u32).map(|t| POS_BASE + t).collect();

    let d_x = stream.clone_htod(&x).expect("upload x");
    let d_pos = stream.clone_htod(&positions).expect("upload positions");
    let mut d_out = stream
        .alloc_zeros::<f32>(TOKENS * heads * head_dim)
        .expect("alloc out");
    kernels
        .rope(
            &stream, &d_x, &d_pos, &mut d_out, TOKENS, heads, head_dim, rope_dim, THETA_BASE,
        )
        .expect("rope launches");
    let candidate = stream.clone_dtoh(&d_out).expect("read back");
    stream.synchronize().expect("sync");

    let mut reference: Vec<f32> = Vec::with_capacity(TOKENS * heads * head_dim);
    for (t, &pos) in positions.iter().enumerate() {
        for h in 0..heads {
            let base = (t * heads + h) * head_dim;
            reference.extend(apply_rope(
                &x[base..base + head_dim],
                pos,
                rope_dim as u32,
                THETA_BASE,
            ));
        }
    }

    // --- the tail, alone, exactly ----------------------------------------
    //
    // Checked against the *input*, not against the reference's copy of it, so
    // this measures pass-through rather than agreement between two copies.
    let tail_candidate: Vec<f32> = (0..TOKENS * heads)
        .flat_map(|row| x_span(&candidate, row, head_dim, rope_dim))
        .collect();
    let tail_input: Vec<f32> = (0..TOKENS * heads)
        .flat_map(|row| x_span(&x, row, head_dim, rope_dim))
        .collect();
    assert_eq!(
        tail_candidate.len(),
        TOKENS * heads * (head_dim - rope_dim),
        "the tail span must be head_dim - rope_dim wide",
    );
    assert_eq!(
        tail_candidate, tail_input,
        "dimensions [{rope_dim}, {head_dim}) must pass through bit-identical, not merely close",
    );
    assert_matches(&tail_candidate, &tail_input, &Tolerance::exact());
    println!(
        "rope tail: {} elements in [{rope_dim}, {head_dim}) bit-identical to the input",
        tail_candidate.len(),
    );

    // --- the rotated span, against the reference --------------------------
    let rot_candidate: Vec<f32> = (0..TOKENS * heads)
        .flat_map(|row| candidate[row * head_dim..row * head_dim + rope_dim].to_vec())
        .collect();
    let rot_reference: Vec<f32> = (0..TOKENS * heads)
        .flat_map(|row| reference[row * head_dim..row * head_dim + rope_dim].to_vec())
        .collect();
    let result = compare(&rot_candidate, &rot_reference);
    println!(
        "rope rotated span: max_abs={:.3e} cosine={:.9} max_rel={:.3e} on reference {:.3e}",
        result.max_abs_error,
        result.cosine_similarity,
        result.max_rel_error,
        rot_reference[result.max_rel_error_index].abs(),
    );
    // The evidence for leaving ROPE_GATE.max_rel_error loose: the ratio is
    // large because the denominator is small, not because the error is. If
    // that stops being true this fails before the loose bound can hide it.
    let driver = result.max_rel_error_index;
    let driver_abs_error = (rot_candidate[driver] - rot_reference[driver]).abs();
    assert!(
        driver_abs_error < ROPE_MAX_REL_DRIVER_ABS_ERROR,
        "max_rel_error {:.3e} is driven by an absolute error of {driver_abs_error:.3e} on a \
         reference value of {:.3e} — that is a real error, not a cancellation artefact, so \
         max_abs_error alone is no longer a sufficient gate",
        result.max_rel_error,
        rot_reference[driver].abs(),
    );
    println!(
        "rope max_rel driver: abs_error={driver_abs_error:.3e} (bound {:.0e}) on reference {:.3e}",
        ROPE_MAX_REL_DRIVER_ABS_ERROR,
        rot_reference[driver].abs(),
    );
    assert_matches(&rot_candidate, &rot_reference, &ROPE_GATE);

    // --- and the whole tensor, so a mis-sliced test cannot hide a bug -----
    assert_matches(&candidate, &reference, &ROPE_GATE);
}

/// The `[rope_dim, head_dim)` tail of row `row` of a `[rows][head_dim]` tensor.
fn x_span(v: &[f32], row: usize, head_dim: usize, rope_dim: usize) -> Vec<f32> {
    v[row * head_dim + rope_dim..(row + 1) * head_dim].to_vec()
}

#[test]
fn device_swiglu_matches_the_reference() {
    let Some(ctx) = setup() else { return };
    let stream = ctx.default_stream();
    let kernels = LayerOpsKernels::new(&ctx).expect("kernels must compile");

    // One token's worth of routed-expert activation for a whole batch:
    // expert_intermediate 512 x 8 experts x 256 tokens.
    let config = ModelConfig::qwen3_6_35b_a3b();
    let n = config.moe.expert_intermediate as usize * config.moe.experts_per_token as usize * 256;

    // Wide enough to cover both saturating limbs of the sigmoid: at -12 the
    // silu output is ~-7e-5 and at +12 it is ~12, so a kernel that got the
    // sign of the exponent backwards cannot hide in the middle of the range.
    let mut rng = Xorshift64Star::new(0x5555);
    let gate = rng.vec_f32(n, -12.0, 12.0);
    let up = rng.vec_f32(n, -3.0, 3.0);

    let reference = swiglu(&gate, &up);

    let d_gate = stream.clone_htod(&gate).expect("upload gate");
    let d_up = stream.clone_htod(&up).expect("upload up");
    let mut d_out = stream.alloc_zeros::<f32>(n).expect("alloc out");
    kernels
        .swiglu(&stream, &d_gate, &d_up, &mut d_out, n)
        .expect("swiglu launches");
    let candidate = stream.clone_dtoh(&d_out).expect("read back");
    stream.synchronize().expect("sync");

    let result = compare(&candidate, &reference);
    println!(
        "swiglu: {n} elements -> max_abs={:.3e} cosine={:.9} max_rel={:.3e} on reference {:.3e}",
        result.max_abs_error,
        result.cosine_similarity,
        result.max_rel_error,
        reference[result.max_rel_error_index].abs(),
    );
    // All three metrics, including the relative error — see SWIGLU_GATE.
    assert_matches(&candidate, &reference, &SWIGLU_GATE);
}

#[test]
fn device_sigmoid_gate_matches_the_reference_in_both_shapes_the_model_uses() {
    // Two callers, two gate shapes, one kernel. The Gated Attention output
    // gate is elementwise over the q projection; the MoE shared expert's gate
    // is a single scalar per token broadcast over the whole hidden dimension.
    // Both run here at the real widths, because a kernel that quietly assumed
    // one of them would still produce finite, plausible numbers for the other.
    let Some(ctx) = setup() else { return };
    let config = ModelConfig::qwen3_6_35b_a3b();
    let stream = ctx.default_stream();
    let kernels = LayerOpsKernels::new(&ctx).expect("kernels must compile");

    const TOKENS: usize = 64;
    let q_dim = config.attention.q_heads as usize * config.attention.head_dim as usize;
    assert_eq!(q_dim, 4096, "the attention output gate is [4096, n_tokens]");
    let hidden = config.hidden_size as usize;

    let cases: [(&str, usize, usize, GateShape); 2] = [
        (
            "attention output gate",
            TOKENS,
            q_dim,
            GateShape::Elementwise,
        ),
        ("moe shared expert gate", TOKENS, hidden, GateShape::PerRow),
    ];

    for (name, rows, width, shape) in cases {
        let n = rows * width;
        let gate_len = match shape {
            GateShape::Elementwise => n,
            GateShape::PerRow => rows,
        };

        let mut rng = Xorshift64Star::new(0x9999_0000 + width as u64);
        let x = rng.vec_f32(n, -3.0, 3.0);
        // Wide enough to reach both saturating limbs: sigmoid(-12) is ~6.1e-6
        // and sigmoid(12) is ~1 - 6.1e-6, so a flipped exponent sign cannot
        // hide in the middle of the range.
        let gate = rng.vec_f32(gate_len, -12.0, 12.0);

        // One oracle for both shapes: expand the broadcast gate on the host
        // and call the same elementwise reference.
        let expanded: Vec<f32> = match shape {
            GateShape::Elementwise => gate.clone(),
            GateShape::PerRow => (0..n).map(|i| gate[i / width]).collect(),
        };
        let reference = sigmoid_gate(&x, &expanded);
        let sig_reference: Vec<f32> = gate.iter().map(|&g| sigmoid(g)).collect();

        let d_x = stream.clone_htod(&x).expect("upload x");
        let d_gate = stream.clone_htod(&gate).expect("upload gate");
        let mut d_sig = stream
            .clone_htod(&vec![SIG_SENTINEL; gate_len])
            .expect("upload sigmoid sentinel");
        let mut d_out = stream.alloc_zeros::<f32>(n).expect("alloc out");
        kernels
            .sigmoid_gate(
                &stream, &d_x, &d_gate, &mut d_sig, &mut d_out, rows, width, shape,
            )
            .expect("sigmoid_gate launches");
        let candidate = stream.clone_dtoh(&d_out).expect("read back out");
        let sig = stream.clone_dtoh(&d_sig).expect("read back sigmoid");
        stream.synchronize().expect("sync");

        // Every gate element was written exactly once. In PerRow mode a single
        // thread per row owns `sig[row]`, and a guard that selected no writer
        // for some row would leave the sentinel behind here rather than
        // showing up as a plausible-looking product downstream.
        assert!(
            !sig.contains(&SIG_SENTINEL),
            "{name}: {} of {gate_len} sigmoid entries were never written",
            sig.iter().filter(|&&v| v == SIG_SENTINEL).count(),
        );

        let sig_result = compare(&sig, &sig_reference);
        println!(
            "sigmoid_gate {name}: sigmoid over {gate_len} gate values -> \
             max_abs={:.3e} cosine={:.9} max_rel={:.3e} on reference {:.3e}",
            sig_result.max_abs_error,
            sig_result.cosine_similarity,
            sig_result.max_rel_error,
            sig_reference[sig_result.max_rel_error_index].abs(),
        );
        assert_matches(&sig, &sig_reference, &SIGMOID_GATE);

        let result = compare(&candidate, &reference);
        println!(
            "sigmoid_gate {name}: {rows} rows x {width} ({n} elements, gate {gate_len}) -> \
             max_abs={:.3e} cosine={:.9} max_rel={:.3e} on reference {:.3e}",
            result.max_abs_error,
            result.cosine_similarity,
            result.max_rel_error,
            reference[result.max_rel_error_index].abs(),
        );
        assert_matches(&candidate, &reference, &SIGMOID_GATE);
    }

    println!(
        "gate: max_abs<{:.0e}, max_rel<{:.0e}, cosine>{:.9}",
        SIGMOID_GATE.max_abs_error, SIGMOID_GATE.max_rel_error, SIGMOID_GATE.min_cosine_similarity,
    );
}

#[test]
fn device_sigmoid_gate_broadcast_really_varies_per_token_and_not_per_element() {
    // The probe that makes the shape claim falsifiable. Give one token a gate
    // that is wide open and another one that is shut; if the kernel indexed a
    // PerRow gate elementwise it would apply token 0's scalar to the first
    // `width` elements of the *flattened* tensor and read past the buffer for
    // the rest, and if it ignored `broadcast` entirely the length check would
    // have rejected the launch. Neither of those reproduces this pattern.
    let Some(ctx) = setup() else { return };
    let stream = ctx.default_stream();
    let kernels = LayerOpsKernels::new(&ctx).expect("kernels must compile");

    let rows = 4usize;
    let width = 2048usize;
    let x = vec![1.0f32; rows * width];
    // sigmoid(30) rounds to 1.0 and sigmoid(-30) to ~9.36e-14 in fp32.
    let gate = vec![30.0f32, -30.0, 30.0, -30.0];

    let d_x = stream.clone_htod(&x).expect("upload x");
    let d_gate = stream.clone_htod(&gate).expect("upload gate");
    let mut d_sig = stream
        .clone_htod(&vec![SIG_SENTINEL; rows])
        .expect("upload");
    let mut d_out = stream.alloc_zeros::<f32>(rows * width).expect("alloc out");
    kernels
        .sigmoid_gate(
            &stream,
            &d_x,
            &d_gate,
            &mut d_sig,
            &mut d_out,
            rows,
            width,
            GateShape::PerRow,
        )
        .expect("sigmoid_gate launches");
    let out = stream.clone_dtoh(&d_out).expect("read back");
    stream.synchronize().expect("sync");

    for row in 0..rows {
        let span = &out[row * width..(row + 1) * width];
        let open = row % 2 == 0;
        for (j, &v) in span.iter().enumerate() {
            if open {
                assert!(
                    (v - 1.0).abs() < 1e-6,
                    "row {row} col {j}: an open gate must pass x through, got {v}",
                );
            } else {
                assert!(
                    v.abs() < 1e-6,
                    "row {row} col {j}: a shut gate must silence x, got {v}",
                );
            }
        }
    }
    println!(
        "sigmoid_gate broadcast: {rows} tokens x {width} — alternating open/shut per-token \
         scalars reached all {} elements of their own row and none of any other",
        rows * width,
    );
}

#[test]
fn device_tensor_add_is_bit_identical_to_the_reference() {
    // The residual add. There is no rounding freedom in `a + b`, so anything
    // short of exact equality means the kernel is not doing what it says —
    // a fused scale, a reassociation, an FMA with something else. A tolerance
    // here would pass all three.
    let Some(ctx) = setup() else { return };
    let config = ModelConfig::qwen3_6_35b_a3b();
    let stream = ctx.default_stream();
    let kernels = LayerOpsKernels::new(&ctx).expect("kernels must compile");

    // The real shape: one hidden-size residual per token, twice per layer.
    const TOKENS: usize = 256;
    let n = TOKENS * config.hidden_size as usize;

    let mut rng = Xorshift64Star::new(0xAAAA);
    // Operands of very different magnitudes on purpose: `a + b` with
    // |a| >> |b| is where a reassociated or fused form would lose the small
    // operand entirely and still look right on smooth input.
    let a = rng.vec_f32(n, -100.0, 100.0);
    let b = rng.vec_f32(n, -1e-3, 1e-3);

    let reference = residual_add(&a, &b);

    let d_a = stream.clone_htod(&a).expect("upload a");
    let d_b = stream.clone_htod(&b).expect("upload b");
    let mut d_out = stream.alloc_zeros::<f32>(n).expect("alloc out");
    kernels
        .add(&stream, &d_a, &d_b, &mut d_out, n)
        .expect("add launches");
    let candidate = stream.clone_dtoh(&d_out).expect("read back");
    stream.synchronize().expect("sync");

    let result = compare(&candidate, &reference);
    println!(
        "tensor_add: {n} elements ({TOKENS} tokens x {}) -> max_abs={:.3e} cosine={:.9} \
         (gate: exact equality)",
        config.hidden_size, result.max_abs_error, result.cosine_similarity,
    );
    assert_eq!(
        result.max_abs_error, 0.0,
        "the residual add must be bit-identical to the reference",
    );
    assert_eq!(candidate, reference, "the residual add diverged");
    assert_matches(&candidate, &reference, &Tolerance::exact());

    // And in place, which is how a residual stream is actually updated.
    let mut d_acc = stream.clone_htod(&a).expect("upload a");
    let d_acc_ro = d_acc.clone();
    kernels
        .add(&stream, &d_acc_ro, &d_b, &mut d_acc, n)
        .expect("in-place add launches");
    let in_place = stream.clone_dtoh(&d_acc).expect("read back");
    stream.synchronize().expect("sync");
    assert_eq!(
        in_place, reference,
        "the in-place residual add diverged from the out-of-place one",
    );
    println!("tensor_add in place: {n} elements bit-identical to the reference");
}

#[test]
fn device_softplus_matches_the_reference_including_its_large_argument_passthrough() {
    // The GDN alpha gate's nonlinearity. What is actually being checked is
    // ggml's `x > 20` branch: without it `expf` overflows above x ~ 88.7 and
    // softplus returns inf where the answer is x, which becomes a non-finite
    // log-decay and poisons the whole recurrence.
    let Some(ctx) = setup() else { return };
    let config = ModelConfig::qwen3_6_35b_a3b();
    let stream = ctx.default_stream();
    let kernels = LayerOpsKernels::new(&ctx).expect("kernels must compile");

    // The real shape: one alpha per (token, value head).
    const TOKENS: usize = 512;
    let heads = config.gdn.value_heads as usize;
    let n = TOKENS * heads;

    // Exact probes on the branch and well past where an unguarded exp dies.
    // 88.0 is the last argument `expf` survives; 89.0 is the first it does
    // not; both are above the threshold and must therefore come back as
    // themselves.
    let probes: [f32; 10] = [
        0.0, 19.0, 19.999_998, 20.0, 20.000_002, 88.0, 89.0, 100.0, 1000.0, -80.0,
    ];
    assert!(
        88.0f32.exp().is_finite() && 89.0f32.exp().is_infinite(),
        "the probes no longer straddle fp32 exp overflow",
    );

    let mut rng = Xorshift64Star::new(0xBBBB);
    let mut x = rng.vec_f32(n - probes.len(), -40.0, 30.0);
    x.extend_from_slice(&probes);
    assert_eq!(x.len(), n);

    let reference: Vec<f32> = x.iter().map(|&v| softplus(v)).collect();
    assert!(
        reference.iter().all(|v| v.is_finite()),
        "the reference itself overflowed; the CPU passthrough is broken",
    );

    let d_x = stream.clone_htod(&x).expect("upload x");
    let mut d_out = stream.alloc_zeros::<f32>(n).expect("alloc out");
    kernels
        .softplus(&stream, &d_x, &mut d_out, n)
        .expect("softplus launches");
    let candidate = stream.clone_dtoh(&d_out).expect("read back");
    stream.synchronize().expect("sync");

    // The passthrough, checked exactly and by index rather than inferred from
    // an aggregate: above 20 the kernel is a copy, so these are bit-identical.
    for (k, &p) in probes.iter().enumerate() {
        let i = n - probes.len() + k;
        if p > 20.0 {
            assert_eq!(
                candidate[i], p,
                "softplus({p}) must pass through unchanged, got {}",
                candidate[i],
            );
        }
        assert!(
            candidate[i].is_finite(),
            "softplus({p}) returned {} — the x > 20 passthrough is gone",
            candidate[i],
        );
    }
    println!(
        "softplus passthrough: {:?} all returned themselves bit-exactly",
        probes.iter().filter(|&&p| p > 20.0).collect::<Vec<_>>(),
    );

    let result = compare(&candidate, &reference);
    println!(
        "softplus: {n} elements ({TOKENS} tokens x {heads} value heads) over x in [-40, 30] \
         plus probes -> max_abs={:.3e} cosine={:.9} max_rel={:.3e} on reference {:.3e}",
        result.max_abs_error,
        result.cosine_similarity,
        result.max_rel_error,
        reference[result.max_rel_error_index].abs(),
    );
    assert_eq!(result.non_finite_count, 0, "the device produced NaN/Inf");

    // The evidence for leaving SOFTPLUS_GATE.max_rel_error two orders looser
    // than the sigmoid gate's: the ratio is large because `1 + exp(x)` carries
    // the answer in the low bits of a number near 1, not because the error is.
    // If the error itself ever grows, this fails before the loose bound can
    // absorb it.
    let driver = result.max_rel_error_index;
    let driver_abs_error = (candidate[driver] - reference[driver]).abs();
    assert!(
        driver_abs_error < SOFTPLUS_MAX_REL_DRIVER_ABS_ERROR,
        "max_rel_error {:.3e} is driven by an absolute error of {driver_abs_error:.3e} on a \
         reference value of {:.3e} (x = {:.3e}) — that is larger than one ulp of the \
         intermediate `1 + exp(x)`, so it is a real error rather than the formula's own \
         cancellation and max_abs_error alone is no longer a sufficient gate",
        result.max_rel_error,
        reference[driver].abs(),
        x[driver],
    );
    println!(
        "softplus max_rel driver: x={:.3e} abs_error={driver_abs_error:.3e} (bound {:.0e}) \
         on reference {:.3e}",
        x[driver],
        SOFTPLUS_MAX_REL_DRIVER_ABS_ERROR,
        reference[driver].abs(),
    );

    assert_matches(&candidate, &reference, &SOFTPLUS_GATE);
    println!(
        "gate: max_abs<{:.0e}, max_rel<{:.0e}, cosine>{:.9}",
        SOFTPLUS_GATE.max_abs_error,
        SOFTPLUS_GATE.max_rel_error,
        SOFTPLUS_GATE.min_cosine_similarity,
    );
}

#[test]
fn device_gdn_conv1d_matches_the_reference_bit_for_bit() {
    let Some(ctx) = setup() else { return };
    let config = ModelConfig::qwen3_6_35b_a3b();
    let g = config.gdn;
    let conv_kernel = g.conv_kernel as usize;
    let channels = gdn_conv_channels(
        g.qk_heads as usize,
        g.value_heads as usize,
        g.head_dim as usize,
    );
    // The real `ssm_conv1d.weight` is [4, 8192]; if this ever stops being
    // 8192 the tensor and the kernel have parted company.
    assert_eq!(channels, 8192, "conv channel count must match [4, 8192]");
    const SEQ_LEN: usize = 512;

    println!(
        "conv1d: {channels} channels, conv_kernel={conv_kernel}, {SEQ_LEN} tokens \
         (state = {} floats)",
        channels * (conv_kernel - 1),
    );

    let stream = ctx.default_stream();
    let kernels = LayerOpsKernels::new(&ctx).expect("kernels must compile");

    let mut rng = Xorshift64Star::new(0x6666);
    let x = rng.vec_f32(SEQ_LEN * channels, -2.0, 2.0);
    let weight = rng.vec_f32(channels * conv_kernel, -0.6, 0.6);
    // A non-zero carried cache: zeros would make the first three tokens pass
    // whether or not the kernel reads the state at all.
    let state0 = rng.vec_f32(channels * (conv_kernel - 1), -2.0, 2.0);

    let (reference, ref_state) =
        causal_depthwise_conv1d(&x, &weight, &state0, SEQ_LEN, channels, conv_kernel);

    let d_x = stream.clone_htod(&x).expect("upload x");
    let d_w = stream.clone_htod(&weight).expect("upload weight");
    let mut d_state = stream.clone_htod(&state0).expect("upload state");
    let mut d_out = stream
        .alloc_zeros::<f32>(SEQ_LEN * channels)
        .expect("alloc out");
    kernels
        .conv1d(
            &stream,
            &d_x,
            &d_w,
            &mut d_state,
            &mut d_out,
            SEQ_LEN,
            channels,
            conv_kernel,
        )
        .expect("conv1d launches");
    let candidate = stream.clone_dtoh(&d_out).expect("read output");
    let cand_state = stream.clone_dtoh(&d_state).expect("read state");
    stream.synchronize().expect("sync");

    // Exact, not close. Four taps in ascending order is the reference's exact
    // operand sequence, and the kernel uses __fadd_rn/__fmul_rn so nvcc
    // cannot contract the multiply-add into a single-rounding FMA. A
    // tolerance here would pass a reversed tap order on smooth input; exact
    // equality cannot.
    let result = compare(&candidate, &reference);
    println!(
        "conv1d output: max_abs={:.3e} cosine={:.9} (gate: exact equality)",
        result.max_abs_error, result.cosine_similarity,
    );
    assert_eq!(
        result.max_abs_error, 0.0,
        "conv1d must be bit-identical to the reference; \
         a non-zero max_abs means nvcc contracted the accumulation into an FMA \
         or the operand order drifted",
    );
    assert_matches(&candidate, &reference, &Tolerance::exact());

    // The cache is what the next decode step depends on, so it is checked
    // independently: a kernel can produce correct output from a cache that is
    // about to be wrong.
    assert_eq!(
        cand_state,
        ref_state,
        "the convolution cache must hold the last {} inputs, bit-identical",
        conv_kernel - 1,
    );
    println!(
        "conv1d cache: {} floats bit-identical to the reference",
        cand_state.len(),
    );
}

#[test]
fn device_gdn_conv1d_streamed_one_token_at_a_time_reproduces_the_batched_result() {
    // The decode path. Output at token t must depend only on tokens
    // t-3 ..= t, which means feeding the same sequence one token at a time
    // through the cache has to give the same answer as feeding it in one
    // batch. If it does not, prefill and decode are two different models.
    let Some(ctx) = setup() else { return };
    let config = ModelConfig::qwen3_6_35b_a3b();
    let g = config.gdn;
    let conv_kernel = g.conv_kernel as usize;
    let channels = gdn_conv_channels(
        g.qk_heads as usize,
        g.value_heads as usize,
        g.head_dim as usize,
    );
    const SEQ_LEN: usize = 64;

    let stream = ctx.default_stream();
    let kernels = LayerOpsKernels::new(&ctx).expect("kernels must compile");

    let mut rng = Xorshift64Star::new(0x7777);
    let x = rng.vec_f32(SEQ_LEN * channels, -2.0, 2.0);
    let weight = rng.vec_f32(channels * conv_kernel, -0.6, 0.6);
    let state0 = rng.vec_f32(channels * (conv_kernel - 1), -2.0, 2.0);

    let (reference, ref_state) =
        causal_depthwise_conv1d(&x, &weight, &state0, SEQ_LEN, channels, conv_kernel);

    let d_w = stream.clone_htod(&weight).expect("upload weight");
    let mut d_state = stream.clone_htod(&state0).expect("upload state");
    let mut d_out = stream.alloc_zeros::<f32>(channels).expect("alloc out");

    let mut streamed = Vec::with_capacity(SEQ_LEN * channels);
    for t in 0..SEQ_LEN {
        let token = &x[t * channels..(t + 1) * channels];
        let d_token = stream.clone_htod(token).expect("upload token");
        kernels
            .conv1d(
                &stream,
                &d_token,
                &d_w,
                &mut d_state,
                &mut d_out,
                1,
                channels,
                conv_kernel,
            )
            .expect("conv1d step launches");
        streamed.extend_from_slice(&stream.clone_dtoh(&d_out).expect("read step output"));
    }
    let streamed_state = stream.clone_dtoh(&d_state).expect("read state");
    stream.synchronize().expect("sync");

    // A single-token step reads state entries it also overwrites, which is
    // exactly the aliasing the state kernel stages registers to survive. If
    // that staging were wrong this is where it would show.
    assert_eq!(
        streamed, reference,
        "streaming {SEQ_LEN} single-token steps must reproduce the batched result exactly",
    );
    assert_eq!(streamed_state, ref_state, "the streamed cache diverged");
    println!(
        "conv1d streaming: {SEQ_LEN} single-token steps over {channels} channels, \
         bit-identical to the {SEQ_LEN}-token batch (and so is the cache)",
    );
}

#[test]
fn device_gdn_conv1d_is_causal_a_later_token_cannot_change_an_earlier_output() {
    // Causality measured on the device rather than inferred from the source.
    // Rewrite one token and every output before it must be bit-identical; the
    // perturbed token's own output must change, or the probe proved nothing.
    let Some(ctx) = setup() else { return };
    let stream = ctx.default_stream();
    let kernels = LayerOpsKernels::new(&ctx).expect("kernels must compile");

    let channels = 512usize;
    let conv_kernel = 4usize;
    let seq_len = 32usize;
    const PERTURBED: usize = 20;

    let mut rng = Xorshift64Star::new(0x8888);
    let x = rng.vec_f32(seq_len * channels, -1.0, 1.0);
    let weight = rng.vec_f32(channels * conv_kernel, -1.0, 1.0);
    let state0 = rng.vec_f32(channels * (conv_kernel - 1), -1.0, 1.0);

    let run = |x: &[f32]| -> Vec<f32> {
        let d_x = stream.clone_htod(x).expect("upload x");
        let d_w = stream.clone_htod(&weight).expect("upload weight");
        let mut d_state = stream.clone_htod(&state0).expect("upload state");
        let mut d_out = stream
            .alloc_zeros::<f32>(seq_len * channels)
            .expect("alloc out");
        kernels
            .conv1d(
                &stream,
                &d_x,
                &d_w,
                &mut d_state,
                &mut d_out,
                seq_len,
                channels,
                conv_kernel,
            )
            .expect("conv1d launches");
        let out = stream.clone_dtoh(&d_out).expect("read out");
        stream.synchronize().expect("sync");
        out
    };

    let base = run(&x);
    let mut perturbed_x = x.clone();
    for ch in 0..channels {
        perturbed_x[PERTURBED * channels + ch] += 1000.0;
    }
    let perturbed = run(&perturbed_x);

    assert_eq!(
        &base[..PERTURBED * channels],
        &perturbed[..PERTURBED * channels],
        "token {PERTURBED} changed an output that precedes it — the device conv is not causal",
    );
    assert_ne!(
        base[PERTURBED * channels],
        perturbed[PERTURBED * channels],
        "the perturbed token did not reach its own output; the probe proved nothing",
    );
    // And it must stop influencing outputs after conv_kernel - 1 more tokens.
    let last_affected = PERTURBED + conv_kernel - 1;
    assert_eq!(
        &base[(last_affected + 1) * channels..],
        &perturbed[(last_affected + 1) * channels..],
        "the perturbation outlived its {conv_kernel}-tap window",
    );
    println!(
        "conv1d causality: perturbing token {PERTURBED} left outputs 0..{PERTURBED} and \
         {}..{seq_len} bit-identical, and changed exactly outputs {PERTURBED}..={last_affected}",
        last_affected + 1,
    );
}

#[test]
fn a_shape_that_does_not_match_the_declared_geometry_is_rejected_not_run() {
    // The launch path validates rather than trusting the caller, so a buffer
    // that is one head short fails loudly instead of reading past its end.
    let Some(ctx) = setup() else { return };
    let stream = ctx.default_stream();
    let kernels = LayerOpsKernels::new(&ctx).expect("kernels must compile");

    let d_x = stream.alloc_zeros::<f32>(100).expect("alloc");
    let d_w = stream.alloc_zeros::<f32>(10).expect("alloc");
    let mut d_out = stream.alloc_zeros::<f32>(100).expect("alloc");
    // 11 rows of 10 is 110 floats; x holds 100.
    let err = kernels
        .rms_norm(&stream, &d_x, &d_w, &mut d_out, 11, 10, RMS_EPS)
        .expect_err("a short buffer must be rejected");
    println!("rejected as expected: {err}");

    // rope_dim must not exceed head_dim.
    let d_pos = stream.alloc_zeros::<u32>(1).expect("alloc");
    let err = kernels
        .rope(&stream, &d_x, &d_pos, &mut d_out, 1, 1, 64, 128, THETA_BASE)
        .expect_err("rope_dim > head_dim must be rejected");
    println!("rejected as expected: {err}");

    // conv_kernel beyond the register array the state kernel stages into.
    let mut d_state = stream.alloc_zeros::<f32>(10).expect("alloc");
    let err = kernels
        .conv1d(&stream, &d_x, &d_w, &mut d_state, &mut d_out, 1, 10, 99)
        .expect_err("an oversized conv_kernel must be rejected");
    println!("rejected as expected: {err}");

    // The gate shape, which is the one that matters most here: a PerRow gate
    // read elementwise would walk 100 floats off the end of a 10-float buffer,
    // so the length check is what stands between a mis-declared shape and an
    // out-of-bounds read. Both directions are rejected.
    let mut d_sig = stream.alloc_zeros::<f32>(10).expect("alloc");
    let err = kernels
        .sigmoid_gate(
            &stream,
            &d_x,
            &d_w,
            &mut d_sig,
            &mut d_out,
            10,
            10,
            GateShape::Elementwise,
        )
        .expect_err("a per-row gate declared elementwise must be rejected");
    println!("rejected as expected: {err}");

    let mut d_sig100 = stream.alloc_zeros::<f32>(100).expect("alloc");
    let d_gate100 = stream.alloc_zeros::<f32>(100).expect("alloc");
    let err = kernels
        .sigmoid_gate(
            &stream,
            &d_x,
            &d_gate100,
            &mut d_sig100,
            &mut d_out,
            10,
            10,
            GateShape::PerRow,
        )
        .expect_err("an elementwise gate declared per-row must be rejected");
    println!("rejected as expected: {err}");
    // And the correctly-declared per-row form is accepted, so the two
    // rejections above are about the shape and not about the call failing
    // for some unrelated reason.
    let d_gate10 = stream.alloc_zeros::<f32>(10).expect("alloc");
    kernels
        .sigmoid_gate(
            &stream,
            &d_x,
            &d_gate10,
            &mut d_sig,
            &mut d_out,
            10,
            10,
            GateShape::PerRow,
        )
        .expect("a correctly declared per-row gate must be accepted");
    stream.synchronize().expect("sync");

    // The add and softplus validate too, on the same helper.
    let err = kernels
        .add(&stream, &d_x, &d_w, &mut d_out, 100)
        .expect_err("a short operand must be rejected");
    println!("rejected as expected: {err}");
    let err = kernels
        .softplus(&stream, &d_x, &mut d_out, 99)
        .expect_err("a length that matches neither buffer must be rejected");
    println!("rejected as expected: {err}");
}
