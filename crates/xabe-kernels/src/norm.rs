//! RMSNorm, SwiGLU, sigmoid gating, softplus, and residual-add reference
//! kernels.
//!
//! These are the simple building blocks around the attention/GDN/MoE cores;
//! they still need CPU references because every fused GPU kernel that folds
//! them in (e.g. RMSNorm + quantize, or SwiGLU inside the MoE expert MLP)
//! gets validated against a plain, obviously-correct version of the same
//! math.
//!
//! ## Why [`silu`] and [`sigmoid`] spell the same sigmoid two different ways
//!
//! [`silu`] is `x / (1 + exp(-x))` — one division — and [`sigmoid`] is
//! `1 / (1 + exp(-x))`, so `silu(x)` is *not* `x * sigmoid(x)` bit for bit:
//! the second form rounds the reciprocal and then the product where the first
//! rounds once. That is not an oversight to be tidied away. Each matches the
//! ggml op whose output the corresponding device kernel is checked against
//! (`ggml_silu` and `ggml_sigmoid` respectively), and rewriting either in
//! terms of the other would move every measured differential by an ulp for no
//! gain.

/// Root-mean-square layer normalization.
///
/// `y_i = x_i / sqrt(mean(x^2) + eps) * weight_i`
///
/// This is the standard formulation used throughout LLaMA-family and Qwen
/// models (no mean-subtraction, unlike LayerNorm). `weight` must have the
/// same length as `x`.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(
        x.len(),
        weight.len(),
        "rms_norm: weight must match x length"
    );
    assert!(!x.is_empty(), "rms_norm: x must be non-empty");

    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();

    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| xi * inv_rms * wi)
        .collect()
}

/// SiLU (swish) activation: `x * sigmoid(x)`.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Logistic sigmoid: `1 / (1 + exp(-x))`.
///
/// Spelled as the reciprocal rather than `0.5 * (1 + tanh(x/2))` or
/// `silu(x) / x`, because that is `ggml_sigmoid`'s own formula — `op_sigmoid`
/// in `ggml/src/ggml-cuda/unary.cu` and in `ggml/src/ggml-cpu/unary-ops.cpp`.
/// The three forms are algebraically equal and round differently, and this is
/// the one whose output the Gated Attention output gate and the MoE shared
/// expert's gate are compared against.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Softplus: `log(1 + exp(x))`, with a passthrough above `x = 20`.
///
/// `(x > 20) ? x : log(1 + exp(x))` is ggml's formulation verbatim. The same
/// expression appears three times in llama.cpp — `op_softplus` in
/// `ggml/src/ggml-cuda/unary.cu` (the CUDA path), `op_softplus` in
/// `ggml/src/ggml-cpu/unary-ops.cpp`, and `ggml_compute_softplus_f32` in
/// `ggml/src/ggml-impl.h`, which `ggml_compute_forward_ssm_scan_f32` calls.
///
/// **The threshold is a correctness guard, not an optimization.** `exp(x)`
/// overflows fp32 above `x ≈ 88.7`, so a branch-free softplus returns `inf`
/// where the true value is `x`, and in the GDN alpha gate that `inf` becomes
/// a non-finite log-decay that poisons the whole recurrence. Where the branch
/// is *taken* it is also invisible: `log(1 + e^20) - 20` is about `2.06e-9`,
/// roughly a thousandth of an fp32 ulp at that magnitude.
///
/// (ggml's SYCL backend uses the numerically better
/// `max(x, 0) + log1p(exp(-|x|))` — `op_softplus` in
/// `ggml/src/ggml-sycl/element_wise.cpp` — but that is not the form the CUDA
/// and CPU paths take, and Qwen3.6's `a_softplus` tensor comes off the CUDA
/// one. Matching the oracle beats improving on it.)
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// SwiGLU gated feed-forward activation.
///
/// `out_i = silu(gate_i) * up_i`
///
/// This is the elementwise gate applied between the gate/up projections and
/// the down projection in every expert MLP (see `MoeConfig` in
/// `xabe-model`). `gate` and `up` must be the same length.
pub fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(
        gate.len(),
        up.len(),
        "swiglu: gate and up must be equal length"
    );
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| silu(g) * u)
        .collect()
}

/// Sigmoid gating: `out_i = sigmoid(gate_i) * x_i`.
///
/// Qwen3.6 has two of these and they differ only in the *shape* of `gate`,
/// which is the caller's problem, not this function's:
///
/// - the Gated Attention output gate is elementwise over `[tokens][q_dim]`
///   (`attn_gate` in `src/models/qwen35moe.cpp`);
/// - the MoE shared expert's gate is one scalar per token — `ffn_gate_inp_shexp`
///   is a single `[hidden]` row, so its projection collapses to a scalar —
///   broadcast over the whole hidden dimension (`build_layer_ffn`).
///
/// A broadcast gate is checked by expanding it and calling this, so there is
/// one oracle rather than two that could drift apart.
///
/// `gate` and `x` must be the same length.
pub fn sigmoid_gate(x: &[f32], gate: &[f32]) -> Vec<f32> {
    assert_eq!(
        x.len(),
        gate.len(),
        "sigmoid_gate: x and gate must be equal length"
    );
    x.iter()
        .zip(gate.iter())
        .map(|(&v, &g)| v * sigmoid(g))
        .collect()
}

/// Elementwise residual addition: `out_i = a_i + b_i`.
///
/// Trivial, but every transformer block does this at least twice per layer
/// (post-mixer, post-MLP), and a fused GPU kernel that folds the add into
/// something else still needs this as its oracle.
pub fn residual_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(
        a.len(),
        b.len(),
        "residual_add: operands must be equal length"
    );
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compare::{Tolerance, assert_matches};
    use crate::rng::Xorshift64Star;

    #[test]
    fn rms_norm_of_unit_weight_matches_hand_computed_value() {
        // x = [3, 4], mean(x^2) = (9+16)/2 = 12.5, rms = sqrt(12.5) ~ 3.5355
        let x = [3.0f32, 4.0];
        let w = [1.0f32, 1.0];
        let out = rms_norm(&x, &w, 0.0);
        let expected = [3.0 / 12.5f32.sqrt(), 4.0 / 12.5f32.sqrt()];
        assert_matches(&out, &expected, &Tolerance::tight_fp32());
    }

    #[test]
    fn rms_norm_scales_output_norm_to_sqrt_n_when_weight_is_one() {
        let mut rng = Xorshift64Star::new(11);
        let n = 128;
        let x = rng.vec_f32(n, -3.0, 3.0);
        let w = vec![1.0f32; n];
        let out = rms_norm(&x, &w, 1e-6);
        let out_rms = (out.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt();
        assert!(
            (out_rms - 1.0).abs() < 1e-3,
            "normalized RMS should be ~1, got {out_rms}"
        );
    }

    #[test]
    fn rms_norm_respects_per_channel_weight() {
        let x = [1.0f32, 1.0];
        let w = [2.0f32, 0.5];
        let out = rms_norm(&x, &w, 0.0);
        // mean(x^2) = 1, inv_rms = 1
        assert_matches(&out, &[2.0, 0.5], &Tolerance::tight_fp32());
    }

    #[test]
    fn eps_prevents_division_by_zero_on_an_all_zero_input() {
        let x = [0.0f32, 0.0, 0.0];
        let w = [1.0f32, 1.0, 1.0];
        let out = rms_norm(&x, &w, 1e-6);
        for v in out {
            assert!(v.is_finite());
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn silu_of_zero_is_zero_and_silu_is_monotonic_for_large_x() {
        assert_eq!(silu(0.0), 0.0);
        // silu(x) -> x for large positive x (sigmoid -> 1).
        assert!((silu(10.0) - 10.0).abs() < 1e-3);
        // silu(x) -> 0 for large negative x.
        assert!(silu(-10.0).abs() < 1e-3);
    }

    #[test]
    fn swiglu_gate_of_zero_silences_the_up_projection() {
        let gate = [0.0f32, 0.0];
        let up = [5.0f32, -5.0];
        let out = swiglu(&gate, &up);
        assert_matches(&out, &[0.0, 0.0], &Tolerance::tight_fp32());
    }

    #[test]
    fn swiglu_matches_naive_per_element_definition() {
        let mut rng = Xorshift64Star::new(21);
        let gate = rng.vec_f32(64, -2.0, 2.0);
        let up = rng.vec_f32(64, -2.0, 2.0);
        let out = swiglu(&gate, &up);
        let naive: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        assert_matches(&out, &naive, &Tolerance::tight_fp32());
    }

    #[test]
    fn sigmoid_of_zero_is_one_half_and_it_saturates_symmetrically() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert!((sigmoid(20.0) - 1.0).abs() < 1e-6);
        assert!(sigmoid(-20.0).abs() < 1e-6);
        // 1 - sigmoid(x) == sigmoid(-x), to within fp32 rounding.
        for x in [-8.0f32, -1.5, 0.25, 3.0, 11.0] {
            assert!((1.0 - sigmoid(x) - sigmoid(-x)).abs() < 1e-6, "at {x}");
        }
    }

    #[test]
    fn sigmoid_is_the_reciprocal_form_and_not_silu_divided_by_x() {
        // The two spellings are algebraically equal and round differently.
        // This asserts which one this crate is, because the device kernel is
        // gated against ggml's `sigmoid`, not against `silu(x)/x`.
        for x in [-3.0f32, -0.5, 0.7, 4.25] {
            assert_eq!(sigmoid(x), 1.0 / (1.0 + (-x).exp()));
        }
    }

    #[test]
    fn softplus_matches_log1p_exp_below_the_threshold() {
        // Away from the branch, softplus is just log(1 + e^x).
        for x in [-10.0f32, -1.0, 0.0, 1.0, 19.0] {
            assert_eq!(softplus(x), (1.0 + x.exp()).ln());
        }
        // log(1 + e^0) = log 2.
        assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 1e-7);
    }

    #[test]
    fn softplus_passes_large_arguments_through_instead_of_overflowing() {
        // The reason ggml's threshold exists: e^100 is not representable in
        // fp32, so the unguarded formula returns inf where the answer is 100.
        assert!(
            100.0f32.exp().is_infinite(),
            "the premise of the passthrough no longer holds",
        );
        assert!(!(1.0 + 100.0f32.exp()).ln().is_finite());
        assert_eq!(softplus(100.0), 100.0);
        assert_eq!(softplus(1000.0), 1000.0);
        // And the branch itself is invisible: at the threshold the difference
        // between the two formulas is ~2.06e-9, a thousandth of an ulp of 20.
        let jump = (1.0f32 + 20.0f32.exp()).ln() - 20.0f32;
        assert!(jump < 1e-6, "the passthrough is discontinuous: {jump:e}");
    }

    #[test]
    fn softplus_is_monotonic_and_never_negative() {
        let mut rng = Xorshift64Star::new(41);
        let mut xs = rng.vec_f32(256, -40.0, 40.0);
        xs.sort_by(f32::total_cmp);
        let mut prev = f32::NEG_INFINITY;
        for x in xs {
            let y = softplus(x);
            assert!(y >= 0.0, "softplus({x}) = {y} is negative");
            assert!(y >= prev, "softplus is not monotonic at {x}");
            prev = y;
        }
    }

    #[test]
    fn sigmoid_gate_of_a_zero_gate_halves_the_input() {
        let x = [4.0f32, -6.0];
        let gate = [0.0f32, 0.0];
        assert_matches(&sigmoid_gate(&x, &gate), &[2.0, -3.0], &Tolerance::exact());
    }

    #[test]
    fn sigmoid_gate_saturates_to_the_input_and_to_zero() {
        let x = [1.0f32, 1.0];
        let out = sigmoid_gate(&x, &[30.0, -30.0]);
        assert!((out[0] - 1.0).abs() < 1e-6, "an open gate must pass x");
        assert!(out[1].abs() < 1e-6, "a closed gate must silence x");
    }

    #[test]
    fn sigmoid_gate_matches_the_naive_per_element_definition() {
        let mut rng = Xorshift64Star::new(51);
        let x = rng.vec_f32(64, -3.0, 3.0);
        let gate = rng.vec_f32(64, -12.0, 12.0);
        let naive: Vec<f32> = x
            .iter()
            .zip(gate.iter())
            .map(|(&v, &g)| v * (1.0 / (1.0 + (-g).exp())))
            .collect();
        assert_matches(&sigmoid_gate(&x, &gate), &naive, &Tolerance::exact());
    }

    #[test]
    fn sigmoid_gate_is_not_swiglu() {
        // silu(g)*u = g*sigmoid(g)*u, which is the extra factor of g that
        // separates the MoE expert MLP's activation from the attention output
        // gate. Confusing the two is the kind of bug a tolerance would pass
        // wherever g happens to sit near 1.
        let x = [2.0f32];
        let gate = [3.0f32];
        let gated = sigmoid_gate(&x, &gate);
        let swi = swiglu(&gate, &x);
        assert!(
            (gated[0] - swi[0]).abs() > 1.0,
            "sigmoid_gate {:?} and swiglu {:?} must not coincide",
            gated,
            swi,
        );
    }

    #[test]
    fn residual_add_matches_elementwise_sum() {
        let a = [1.0f32, -2.0, 3.5];
        let b = [0.5f32, 2.0, -3.5];
        let out = residual_add(&a, &b);
        assert_matches(&out, &[1.5, 0.0, 0.0], &Tolerance::tight_fp32());
    }

    #[test]
    fn residual_add_is_commutative() {
        let mut rng = Xorshift64Star::new(31);
        let a = rng.vec_f32(32, -5.0, 5.0);
        let b = rng.vec_f32(32, -5.0, 5.0);
        let ab = residual_add(&a, &b);
        let ba = residual_add(&b, &a);
        assert_matches(&ab, &ba, &Tolerance::tight_fp32());
    }
}
