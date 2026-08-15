//! Gated DeltaNet — recurrent (per-token) form.
//!
//! This is the critical-path kernel: 30 of Qwen3.6's 40 layers are Gated
//! DeltaNet, and unlike Gated Attention it has no equivalent in any
//! flash-attention codebase to lean on.
//!
//! The formulation below was cross-derived from **two independent**
//! upstream implementations that turned out to agree exactly once put in
//! the same notation:
//!
//! 1. `llama.cpp`'s CPU reference,
//!    `ggml_compute_forward_gated_delta_net_one_chunk`
//!    (`ggml/src/ggml-cpu/ops.cpp:10745-10896`). State is stored
//!    "transposed" (`s_out[j*S_v+i] = S[i][j]`, row `j` = value index, so a
//!    row is contiguous over the key index) and it explicitly documents the
//!    per-token update: decay the state, form the delta correction, apply
//!    the outer-product update, *then* read the output — in that order.
//!    It also applies an output scale `1/sqrt(head_dim)` that isn't in most
//!    textbook statements of the delta rule.
//! 2. vLLM's Triton kernel for the same op,
//!    `fused_recurrent_gated_delta_rule_fwd_kernel`
//!    (`vllm/third_party/flash_linear_attention/ops/fused_recurrent.py:27-176`),
//!    which is what `Qwen3NextGatedDeltaNetAttention`
//!    (`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`) — the
//!    module Qwen3-Next/Qwen3.5/Qwen3.6 actually instantiate — calls in
//!    `forward_native`/`forward_cpu`. It keeps state as `b_h[v, k]` (same
//!    convention as llama.cpp's transposed layout, just not transposed-in-
//!    memory), L2-normalizes `q` and `k` (`eps=1e-6`) when
//!    `use_qk_l2norm_in_kernel=True` (which Qwen's call sites always pass),
//!    and scales `q` by `1/sqrt(head_dim)` *before* the update — the same
//!    scale as llama.cpp's, just applied to the other operand (they are
//!    mathematically identical: `q` only ever appears in the final output
//!    dot product, never in the state update, so scaling it before or
//!    after makes no difference).
//!
//! Both sources agree on every operational detail: L2-normalized `q`/`k`,
//! a *scalar* per-(token, head) decay (not the per-channel "KDA" variant
//! `llama.cpp` also supports, which Qwen does not use — confirmed by
//! `A_log`/`dt_bias` in `qwen_gdn_linear_attn.py` being shaped
//! `[num_v_heads]`, one scalar per head, not per channel), decay applied to
//! the state *before* the correction is computed, and the correction
//! applied to the *decayed* state before computing the output for that
//! token. Given two independently-written, differently-structured upstream
//! implementations produce byte-identical math once translated to the same
//! notation, confidence that this matches Qwen3.6's actual GDN is high.
//! What is *not* independently confirmed: the exact epsilon inside the L2
//! norm for Qwen3.6 specifically (this uses vLLM's `1e-6`, since llama.cpp's
//! op assumes q/k are pre-normalized by a separate node and doesn't state
//! one), and whether Qwen3.6 (as opposed to Qwen3-Next/3.5, which share this
//! code path in vLLM) changed anything about the gating. Given the model
//! family lineage this is treated as extremely likely to be correct, not
//! externally verified against a captured Qwen3.6 activation.

/// Per-head recurrent state: a `head_dim x head_dim` matrix, row-major,
/// `state[v * head_dim + k]` — row `v` (value index) is contiguous over the
/// key index `k`. Matches `GdnConfig::state_bytes_per_layer`'s shape
/// `[value_heads, head_dim, head_dim]` in `xabe-model` for one head.
pub type GdnState = Vec<f32>;

/// A fresh, zeroed state for one head.
pub fn zero_state(head_dim: usize) -> GdnState {
    vec![0.0f32; head_dim * head_dim]
}

/// L2-normalizes a vector: `x / sqrt(sum(x^2) + eps)`.
///
/// `eps` matches the FLA Triton kernel's `1e-6` inside the square root
/// (`fused_recurrent.py:128-129`), guarding an all-zero vector from
/// producing `NaN` rather than `0`.
pub fn l2_normalize(x: &[f32], eps: f32) -> Vec<f32> {
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let inv_norm = 1.0 / (sum_sq + eps).sqrt();
    x.iter().map(|&v| v * inv_norm).collect()
}

/// One recurrent step of the gated delta rule for a single head.
///
/// `q`, `k` must already be L2-normalized (see [`l2_normalize`]) and `k`
/// must *not* be pre-scaled; `q` must additionally already carry the
/// `1/sqrt(head_dim)` output scale (see [`output_scale`]). `state` is
/// mutated in place (decayed, then corrected) and the per-token output is
/// returned.
///
/// `log_decay` is the *log-space* decay for this token (matching both
/// upstream sources: llama.cpp's `g` tensor and vLLM's `g` argument are
/// both pre-exponential; the actual multiplicative decay applied to the
/// state is `exp(log_decay)`). `beta` is the scalar update gate.
pub fn recurrent_step(
    state: &mut GdnState,
    head_dim: usize,
    q_scaled: &[f32],
    k: &[f32],
    v: &[f32],
    log_decay: f32,
    beta: f32,
) -> Vec<f32> {
    assert_eq!(state.len(), head_dim * head_dim);
    assert_eq!(q_scaled.len(), head_dim);
    assert_eq!(k.len(), head_dim);
    assert_eq!(v.len(), head_dim);

    let decay = log_decay.exp();

    // 1. Decay the state: S *= decay, uniformly (scalar per-head gating,
    //    not the per-channel KDA variant — see module docs).
    for s in state.iter_mut() {
        *s *= decay;
    }

    // 2. Delta correction: vcorr[vi] = beta * (v[vi] - sum_k S[vi,k]*k[k]).
    let mut vcorr = vec![0.0f32; head_dim];
    for vi in 0..head_dim {
        let row = &state[vi * head_dim..vi * head_dim + head_dim];
        let predicted: f32 = row.iter().zip(k.iter()).map(|(&s, &kk)| s * kk).sum();
        vcorr[vi] = beta * (v[vi] - predicted);
    }

    // 3. Outer-product update: S[vi,k] += vcorr[vi] * k[k].
    for vi in 0..head_dim {
        let row = &mut state[vi * head_dim..vi * head_dim + head_dim];
        for (kk, r) in k.iter().zip(row.iter_mut()) {
            *r += vcorr[vi] * kk;
        }
    }

    // 4. Output: o[vi] = sum_k S[vi,k] * q_scaled[k], using the *updated*
    //    (post-correction) state, matching both upstream sources.
    let mut out = vec![0.0f32; head_dim];
    for vi in 0..head_dim {
        let row = &state[vi * head_dim..vi * head_dim + head_dim];
        out[vi] = row
            .iter()
            .zip(q_scaled.iter())
            .map(|(&s, &qq)| s * qq)
            .sum();
    }

    out
}

/// The output scale applied to (already L2-normalized) `q`: `1/sqrt(head_dim)`.
///
/// Matches llama.cpp's `scale = 1.0f/sqrtf(S_v)` (`ops.cpp:10815`, applied
/// to the output) and FLA's default `scale = k.shape[-1] ** -0.5` (applied
/// to `q` before the update) — see module docs for why these are the same
/// computation despite being applied at different points.
pub fn output_scale(head_dim: usize) -> f32 {
    1.0 / (head_dim as f32).sqrt()
}

/// Runs the recurrent form over a whole sequence for one head.
///
/// `q`, `k`, `v` are `[seq_len][head_dim]`, raw (not yet normalized or
/// scaled — this function does that internally, once, consistently, so
/// that [`crate::gdn::chunked::chunked_forward`] can be compared against it
/// with the exact same preprocessing). `log_decay`, `beta` are `[seq_len]`.
///
/// Returns `(outputs, final_state)`.
pub fn recurrent_forward(
    head_dim: usize,
    q: &[Vec<f32>],
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    log_decay: &[f32],
    beta: &[f32],
    initial_state: Option<&GdnState>,
) -> (Vec<Vec<f32>>, GdnState) {
    let seq_len = q.len();
    assert_eq!(k.len(), seq_len);
    assert_eq!(v.len(), seq_len);
    assert_eq!(log_decay.len(), seq_len);
    assert_eq!(beta.len(), seq_len);

    let mut state = initial_state
        .cloned()
        .unwrap_or_else(|| zero_state(head_dim));
    let scale = output_scale(head_dim);

    let mut outputs = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let q_norm = l2_normalize(&q[t], 1e-6);
        let k_norm = l2_normalize(&k[t], 1e-6);
        let q_scaled: Vec<f32> = q_norm.iter().map(|&x| x * scale).collect();
        let out = recurrent_step(
            &mut state,
            head_dim,
            &q_scaled,
            &k_norm,
            &v[t],
            log_decay[t],
            beta[t],
        );
        outputs.push(out);
    }

    (outputs, state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compare::{Tolerance, assert_matches};
    use crate::rng::Xorshift64Star;

    #[test]
    fn l2_normalize_produces_unit_norm_for_nonzero_input() {
        let x = [3.0f32, 4.0];
        let n = l2_normalize(&x, 0.0);
        let norm: f32 = n.iter().map(|v| v * v).sum();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn l2_normalize_of_zero_vector_does_not_produce_nan() {
        let x = [0.0f32, 0.0, 0.0];
        let n = l2_normalize(&x, 1e-6);
        assert!(n.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn zero_decay_and_zero_beta_leaves_state_and_output_at_zero() {
        let head_dim = 4;
        let mut state = zero_state(head_dim);
        let q = vec![1.0f32; head_dim];
        let k = vec![1.0f32; head_dim];
        let v = vec![5.0f32; head_dim];
        // decay = exp(-inf) would be zero, but use a large negative decay
        // that still evaluates to ~0 in f32, and beta = 0 so no update
        // happens either.
        let out = recurrent_step(&mut state, head_dim, &q, &k, &v, -1e6, 0.0);
        assert!(out.iter().all(|&x| x == 0.0));
        assert!(state.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn full_decay_full_beta_makes_state_equal_the_outer_product_of_k_and_v() {
        // decay = exp(0) = 1 (no decay), beta = 1 (full update), starting
        // from a zero state: S[vi,k] should become v[vi]*k[k] exactly,
        // since the "predicted" term is zero when the state starts at zero.
        let head_dim = 3;
        let mut state = zero_state(head_dim);
        let k = l2_normalize(&[1.0, 2.0, 3.0], 1e-6);
        let v = vec![10.0f32, -5.0, 2.0];
        let q = vec![0.0f32; head_dim]; // unused for this check
        let _out = recurrent_step(&mut state, head_dim, &q, &k, &v, 0.0, 1.0);

        let mut expected = vec![0.0f32; head_dim * head_dim];
        for vi in 0..head_dim {
            for kk in 0..head_dim {
                expected[vi * head_dim + kk] = v[vi] * k[kk];
            }
        }
        assert_matches(&state, &expected, &Tolerance::tight_fp32());
    }

    #[test]
    fn recurrent_forward_matches_a_hand_unrolled_two_step_sequence() {
        let head_dim = 2;
        let q = vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0]];
        let k = vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0]];
        let v = vec![vec![2.0f32, 0.0], vec![0.0f32, 3.0]];
        let log_decay = vec![0.0f32, 0.0];
        let beta = vec![1.0f32, 1.0];

        let (outputs, final_state) =
            recurrent_forward(head_dim, &q, &k, &v, &log_decay, &beta, None);

        // Manually unroll using the same primitives to build an
        // independent expectation via a different call path (single-shot
        // vs. step-by-step with pre-normalized inputs supplied directly).
        let mut state = zero_state(head_dim);
        let scale = output_scale(head_dim);
        let mut expected_outputs = Vec::new();
        for t in 0..2 {
            let qn = l2_normalize(&q[t], 1e-6);
            let kn = l2_normalize(&k[t], 1e-6);
            let qs: Vec<f32> = qn.iter().map(|&x| x * scale).collect();
            expected_outputs.push(recurrent_step(
                &mut state, head_dim, &qs, &kn, &v[t], 0.0, 1.0,
            ));
        }

        for (o, e) in outputs.iter().zip(expected_outputs.iter()) {
            assert_matches(o, e, &Tolerance::tight_fp32());
        }
        assert_matches(&final_state, &state, &Tolerance::tight_fp32());
    }

    #[test]
    fn output_scale_is_inverse_sqrt_head_dim() {
        assert!((output_scale(256) - 1.0 / 16.0).abs() < 1e-6);
        assert!((output_scale(128) - 1.0 / (128.0f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn recurrent_forward_is_deterministic_across_runs() {
        let mut rng = Xorshift64Star::new(77);
        let head_dim = 16;
        let seq_len = 20;
        let q: Vec<Vec<f32>> = (0..seq_len)
            .map(|_| rng.vec_f32(head_dim, -1.0, 1.0))
            .collect();
        let k: Vec<Vec<f32>> = (0..seq_len)
            .map(|_| rng.vec_f32(head_dim, -1.0, 1.0))
            .collect();
        let v: Vec<Vec<f32>> = (0..seq_len)
            .map(|_| rng.vec_f32(head_dim, -1.0, 1.0))
            .collect();
        let log_decay = rng.vec_f32(seq_len, -1.0, 0.0);
        let beta = rng.vec_f32(seq_len, 0.0, 1.0);

        let (out1, state1) = recurrent_forward(head_dim, &q, &k, &v, &log_decay, &beta, None);
        let (out2, state2) = recurrent_forward(head_dim, &q, &k, &v, &log_decay, &beta, None);

        for (a, b) in out1.iter().zip(out2.iter()) {
            assert_eq!(
                a, b,
                "recurrent_forward must be bit-deterministic on repeated runs"
            );
        }
        assert_eq!(state1, state2);
    }
}
