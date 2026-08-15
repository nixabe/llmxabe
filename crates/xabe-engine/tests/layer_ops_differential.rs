//! Differential test: the device layer ops against the scalar references, at
//! the real Qwen3.6 geometry.
//!
//! These four kernels are the glue G006 (full forward pass) needs and the
//! three big kernels do not provide: RMSNorm, partial rotary embedding,
//! SwiGLU, and the Gated DeltaNet short causal convolution.
//!
//! ## What is compared
//!
//! - `xabe_kernels::norm::rms_norm` at all three widths one forward pass uses
//!   — hidden 2048, attention head_dim 256 (`attn_q_norm` / `attn_k_norm`),
//!   and GDN head_dim 128 (`ssm_norm`).
//! - `xabe_kernels::rope::apply_rope` at head_dim 256 / rope_dim 64, with the
//!   untouched tail checked **separately and exactly**.
//! - `xabe_kernels::norm::swiglu` over a million elements.
//! - `xabe_kernels::conv::causal_depthwise_conv1d` at the real 8192-channel,
//!   4-tap geometry, batched and streamed one token at a time, plus the
//!   convolution cache, plus a causality probe run on the device.
//!
//! ## Which metric gates, and why
//!
//! `compare()` computes relative error as `|c - r| / max(|r|, 1e-6)`, so on
//! any tensor with near-zero elements `max_rel_error` degenerates into
//! `abs_error / 1e-6` and reports nothing about accuracy. That is the trap
//! `gdn_differential.rs` documents, and it applies to exactly one of the four
//! kernels here — RoPE, whose rotation produces genuine cancellation. The
//! other three were **measured** first and turned out to have informative
//! relative errors, so they are gated on all three metrics rather than
//! inheriting a loose `max_rel_error` they do not need. Each tolerance below
//! quotes the number it was set against.
//!
//! Two of the four are gated at **exact equality** instead, which is a much
//! stronger statement than any tolerance:
//!
//! - the rotary tail, because dimensions `[rope_dim, head_dim)` are copied
//!   rather than computed;
//! - the whole convolution, because four taps in ascending order is the
//!   reference's exact operand sequence and the kernel spells the
//!   accumulation with `__fadd_rn`/`__fmul_rn` so nvcc cannot contract it
//!   into an FMA.
//!
//! SKIPS — reporting that it skipped — without a driver or a supported
//! device. It needs no model file: the geometry comes from `ModelConfig` and
//! the activations are synthetic, because there is no captured Qwen3.6
//! activation to compare against.

use std::sync::Arc;

use cudarc::driver::CudaContext;
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_cuda::kernels::layer_ops::LayerOpsKernels;
use xabe_kernels::compare::{Tolerance, assert_matches, compare};
use xabe_kernels::conv::{causal_depthwise_conv1d, gdn_conv_channels};
use xabe_kernels::norm::{rms_norm, swiglu};
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
}
