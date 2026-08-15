//! RMSNorm, SwiGLU, and residual-add reference kernels.
//!
//! These are the simple building blocks around the attention/GDN/MoE cores;
//! they still need CPU references because every fused GPU kernel that folds
//! them in (e.g. RMSNorm + quantize, or SwiGLU inside the MoE expert MLP)
//! gets validated against a plain, obviously-correct version of the same
//! math.

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
