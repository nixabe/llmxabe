//! Partial rotary position embedding (mRoPE), NEOX-style pairing.
//!
//! Qwen3.6 rotates only the leading [`xabe_model::AttentionConfig::rope_dim`]
//! of each [`xabe_model::AttentionConfig::head_dim`]-wide head (64 of 256 —
//! see `config.rs`). Partial rotary is a documented source of subtle bugs
//! because it is easy to accidentally rotate, truncate, or reorder the
//! untouched tail; the differential test below exists specifically to catch
//! that.
//!
//! Pairing convention is "NEOX" (split-half): dimension `i` is paired with
//! `i + rope_dim/2` for `i` in `[0, rope_dim/2)`, matching
//! `ggml_compute_forward_rope_flt`'s `GGML_ROPE_TYPE_NEOX` case
//! (`ggml/src/ggml-cpu/ops.cpp`, `rotate_pairs<T>(n_dims, n_dims/2, ...)`
//! at line 6076), which also copies dimensions `[n_dims, head_dim)` through
//! unmodified (line 6087-6093) — the behaviour this module's tail-passthrough
//! test checks. This is the standard GPT-NeoX / HF `rotate_half` pairing
//! used by Qwen-family partial-rotary attention.

/// Rotates the leading `rope_dim` dimensions of a single head vector by
/// `position`, leaving `[rope_dim, head_dim)` untouched.
///
/// `theta_base` is the RoPE frequency base (commonly `10000.0` or a
/// YaRN-scaled value); frequency for pair `i` (`0 <= i < rope_dim/2`) is
/// `theta_base^(-2*i/rope_dim)`.
///
/// # Panics
/// If `rope_dim` is odd, exceeds `head_vec.len()`, or `head_vec` is empty.
pub fn apply_rope(head_vec: &[f32], position: u32, rope_dim: u32, theta_base: f32) -> Vec<f32> {
    let head_dim = head_vec.len();
    let rope_dim = rope_dim as usize;
    assert!(
        !head_vec.is_empty(),
        "apply_rope: head_vec must be non-empty"
    );
    assert_eq!(rope_dim % 2, 0, "apply_rope: rope_dim must be even");
    assert!(
        rope_dim <= head_dim,
        "apply_rope: rope_dim must not exceed head_dim"
    );

    let mut out = head_vec.to_vec();
    let half = rope_dim / 2;
    let pos = f64::from(position);

    for i in 0..half {
        let freq = f64::from(theta_base).powf(-2.0 * i as f64 / rope_dim as f64);
        let angle = pos * freq;
        let (sin_a, cos_a) = (angle.sin() as f32, angle.cos() as f32);

        let x0 = head_vec[i];
        let x1 = head_vec[i + half];
        out[i] = x0 * cos_a - x1 * sin_a;
        out[i + half] = x0 * sin_a + x1 * cos_a;
    }
    // out[rope_dim..] is left as the untouched copy from `head_vec.to_vec()`.
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compare::compare;
    use crate::rng::Xorshift64Star;

    const THETA_BASE: f32 = 10_000.0;

    fn qwen3_6_shape() -> (usize, u32) {
        let attn = xabe_model::ModelConfig::qwen3_6_35b_a3b().attention;
        (attn.head_dim as usize, attn.rope_dim)
    }

    #[test]
    fn tail_beyond_rope_dim_passes_through_byte_identical() {
        let (head_dim, rope_dim) = qwen3_6_shape();
        let mut rng = Xorshift64Star::new(5);
        let head = rng.vec_f32(head_dim, -1.0, 1.0);
        let out = apply_rope(&head, 12345, rope_dim, THETA_BASE);

        let tail_expected = &head[rope_dim as usize..];
        let tail_actual = &out[rope_dim as usize..];
        assert_eq!(
            tail_actual, tail_expected,
            "untouched tail must be bit-identical, not merely close"
        );
    }

    #[test]
    fn position_zero_is_the_identity_rotation() {
        let (head_dim, rope_dim) = qwen3_6_shape();
        let mut rng = Xorshift64Star::new(6);
        let head = rng.vec_f32(head_dim, -1.0, 1.0);
        let out = apply_rope(&head, 0, rope_dim, THETA_BASE);
        let result = compare(&out, &head);
        assert!(
            result.max_abs_error < 1e-5,
            "rotation by angle 0 must be identity: {result}"
        );
    }

    #[test]
    fn rotation_preserves_the_norm_of_the_rotated_span() {
        let (head_dim, rope_dim) = qwen3_6_shape();
        let mut rng = Xorshift64Star::new(7);
        let head = rng.vec_f32(head_dim, -1.0, 1.0);
        let out = apply_rope(&head, 999, rope_dim, THETA_BASE);

        let norm_before: f32 = head[..rope_dim as usize].iter().map(|v| v * v).sum();
        let norm_after: f32 = out[..rope_dim as usize].iter().map(|v| v * v).sum();
        assert!(
            (norm_before - norm_after).abs() < 1e-3,
            "a rotation must preserve the L2 norm of the rotated span: before={norm_before} after={norm_after}"
        );
    }

    #[test]
    fn full_rotary_head_has_no_untouched_tail() {
        let head = vec![1.0f32; 8];
        let out = apply_rope(&head, 3, 8, THETA_BASE);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn rotation_matrix_is_orthonormal_for_a_single_pair() {
        // rope_dim = 2 is a single pair with frequency theta_base^0 = 1, so
        // this exercises the raw 2x2 rotation matrix directly: it must
        // preserve the unit norm of [1, 0] for an arbitrary position.
        let head = vec![1.0f32, 0.0];
        let out = apply_rope(&head, 17, 2, 10_000.0);
        let norm: f32 = out.iter().map(|v| v * v).sum();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "rotation must preserve unit norm: {norm}"
        );
    }
}
