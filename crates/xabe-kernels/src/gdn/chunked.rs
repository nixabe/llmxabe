//! Gated DeltaNet — chunked parallel form (the prefill path).
//!
//! Processes `chunk_len` tokens at a time using the standard "WY
//! representation" trick for the delta rule (Yang et al., *Parallelizing
//! Linear Transformers with the Delta Rule over Sequence Length*, extended
//! here with per-token gating/decay to match Gated DeltaNet): the
//! chunk-local correction vectors `u_t` for every token in the chunk can be
//! solved for *simultaneously* via one triangular solve, instead of the
//! `recurrent` module's token-by-token dependency chain.
//!
//! # Derivation
//!
//! This module's algorithm is derived directly from the recurrent
//! recurrence in `crates/xabe-kernels/src/gdn/recurrent.rs` (itself
//! cross-checked against two independent upstream sources — see that
//! module's docs) rather than transcribed from an upstream chunked kernel:
//! vLLM's chunked path (`chunk_gated_delta_rule`,
//! `vllm/third_party/flash_linear_attention/ops/chunk.py` and
//! `chunk_delta_h.py`) is Triton-JIT code with no plain scalar reference to
//! port, and llama.cpp does not implement a chunked GDN prefill kernel at
//! all (only the recurrent form). **The one thing this module is graded
//! against is agreement with `recurrent::recurrent_forward` on the same
//! input** — see `equivalence` tests below — which is the project's stated
//! milestone gate (cosine within `1e-3`; `Tolerance::gdn_chunk_vs_recurrent`).
//!
//! Per-token recurrence (state `S`, `head_dim x head_dim`, `v`-major):
//! `S_t = a_t S_{t-1} + beta_t (v_t - a_t S_{t-1} k_t) k_t^T`, `a_t =
//! exp(log_decay_t)`.
//!
//! Within one chunk of length `C` (0-indexed `t = 0..C`), let `S_in` be the
//! state carried in from before the chunk, `lambda_t = exp(sum_{i<=t}
//! log_decay_i)` the cumulative in-chunk decay through and including token
//! `t`, and `T_t = S_t / lambda_t` the "undecayed" state. Substituting into
//! the recurrence and simplifying (the `a_t` cancels) gives a *decay-free*
//! delta rule on `T`, with value rescaled by the *inverse* of the
//! cumulative decay: `T_t = T_{t-1} + beta_t (v_t/lambda_t - T_{t-1} k_t)
//! k_t^T`, `T_{-1} = S_in`.
//!
//! Because every step of that recurrence adds a single outer product to
//! `T`, `T_t = S_in + sum_{i<=t} u_i k_i^T` for some per-token vector
//! `u_i`, by induction. Substituting this ansatz back into the recurrence
//! and requiring it to hold for every `t` gives a *linear system* for all
//! `u_t` simultaneously:
//!
//! ```text
//! u_t + beta_t * sum_{i<t} (k_i . k_t) u_i = beta_t * (v_t/lambda_t - S_in . k_t)   for t = 0..C
//! ```
//!
//! In matrix form, with `A[t][i] = beta_t (k_i . k_t)` for `i < t` (zero
//! elsewhere — strictly lower triangular) and `W[t] = beta_t * (v_t/lambda_t
//! minus S_in k_t)`: `(I + A) U = W`, so `U = (I + A)^{-1} W`. `I + A` is
//! unit lower triangular (ones on the diagonal, since `A`'s diagonal is
//! zero) — exactly what
//! [`crate::gdn::tri::invert_unit_lower_triangular`] solves by forward
//! substitution.
//!
//! (Some presentations of the ungated delta rule instead write this as
//! `(I - A')^{-1}` for a differently-signed `A'`; that is an equivalent
//! statement of the same system reached by defining the unknown as `-u_t`
//! instead of `u_t`. This derivation was worked from first principles
//! against the verified recurrence above — see the C=2 hand-expansion in
//! git history / the derivation notes — and validated by the
//! recurrent-vs-chunked equivalence tests rather than matched to a
//! particular paper's sign convention.)
//!
//! Once `U` is known, the per-token output and the chunk-end state follow
//! directly: `o_t = S_t q_t = lambda_t (S_in q_t + sum_{i<=t} u_i (k_i .
//! q_t))`, and `S_{C-1} = lambda_{C-1} (S_in + sum_i u_i k_i^T)`.

use crate::gdn::recurrent::{GdnState, l2_normalize, output_scale, zero_state};
use crate::gdn::tri::{invert_unit_lower_triangular, matmul};

fn transpose(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = m[i * cols + j];
        }
    }
    out
}

/// Runs the chunked parallel form over a whole sequence for one head,
/// processing `chunk_len` tokens at a time.
///
/// Same call convention as [`crate::gdn::recurrent::recurrent_forward`]:
/// `q`, `k`, `v` are `[seq_len][head_dim]` raw (normalized/scaled
/// internally, identically to the recurrent form, so the two are directly
/// comparable), `log_decay`/`beta` are `[seq_len]`.
///
/// Returns `(outputs, final_state)`.
///
/// Eight plain arguments (over clippy's default limit of seven) rather than
/// a config struct: every argument here is a distinct per-call tensor or
/// scalar, not related configuration, so bundling them would just move the
/// same list into a struct literal at every call site without making
/// either clearer.
#[allow(clippy::too_many_arguments)]
pub fn chunked_forward(
    head_dim: usize,
    chunk_len: usize,
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
    assert!(chunk_len > 0, "chunked_forward: chunk_len must be non-zero");

    let scale = output_scale(head_dim);
    let mut state = initial_state
        .cloned()
        .unwrap_or_else(|| zero_state(head_dim));
    let mut outputs = vec![Vec::new(); seq_len];

    let mut start = 0usize;
    while start < seq_len {
        let end = (start + chunk_len).min(seq_len);
        let c = end - start;

        // Preprocess this chunk's q/k exactly as the recurrent form does.
        let q_scaled: Vec<Vec<f32>> = (start..end)
            .map(|t| {
                l2_normalize(&q[t], 1e-6)
                    .iter()
                    .map(|&x| x * scale)
                    .collect()
            })
            .collect();
        let k_norm: Vec<Vec<f32>> = (start..end).map(|t| l2_normalize(&k[t], 1e-6)).collect();

        // Cumulative in-chunk decay, inclusive of token t's own decay.
        let mut lambda = vec![0.0f32; c];
        let mut running = 0.0f32;
        for i in 0..c {
            running += log_decay[start + i];
            lambda[i] = running.exp();
        }

        // W[t] = beta_t * (v_t/lambda_t - S_in . k_t)
        let s_in_t = transpose(&state, head_dim, head_dim); // [head_dim(k), head_dim(v)]
        let s_in_dot_k = matmul(&flatten(&k_norm), &s_in_t, c, head_dim, head_dim); // [c, head_dim(v)]
        let mut w = vec![0.0f32; c * head_dim];
        for t in 0..c {
            let beta_t = beta[start + t];
            for vi in 0..head_dim {
                let v_tilde = v[start + t][vi] / lambda[t];
                w[t * head_dim + vi] = beta_t * (v_tilde - s_in_dot_k[t * head_dim + vi]);
            }
        }

        // A[t][i] = beta_t * (k_i . k_t) for i < t, else 0.
        let mut a = vec![0.0f32; c * c];
        for t in 0..c {
            for i in 0..t {
                let dot: f32 = k_norm[i]
                    .iter()
                    .zip(k_norm[t].iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                a[t * c + i] = beta[start + t] * dot;
            }
        }
        let m_inv = invert_unit_lower_triangular(&a, c);
        let u = matmul(&m_inv, &w, c, c, head_dim); // [c, head_dim(v)]

        // Causal-inclusive intra-chunk attention: P[t][i] = k_i . q_t for i <= t.
        let mut p = vec![0.0f32; c * c];
        for t in 0..c {
            for i in 0..=t {
                let dot: f32 = k_norm[i]
                    .iter()
                    .zip(q_scaled[t].iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                p[t * c + i] = dot;
            }
        }
        let o_intra = matmul(&p, &u, c, c, head_dim); // [c, head_dim(v)]
        let o_inter = matmul(&flatten(&q_scaled), &s_in_t, c, head_dim, head_dim); // [c, head_dim(v)]

        for t in 0..c {
            let mut o = vec![0.0f32; head_dim];
            for vi in 0..head_dim {
                o[vi] = lambda[t] * (o_inter[t * head_dim + vi] + o_intra[t * head_dim + vi]);
            }
            outputs[start + t] = o;
        }

        // Chunk-end state: S_end = lambda_{C-1} * (S_in + sum_i u_i k_i^T).
        let u_t = transpose(&u, c, head_dim); // [head_dim(v), c]
        let sum_term = matmul(&u_t, &flatten(&k_norm), head_dim, c, head_dim); // [head_dim(v), head_dim(k)]
        let lambda_last = lambda[c - 1];
        for idx in 0..head_dim * head_dim {
            state[idx] = lambda_last * (state[idx] + sum_term[idx]);
        }

        start = end;
    }

    (outputs, state)
}

fn flatten(rows: &[Vec<f32>]) -> Vec<f32> {
    rows.iter().flatten().copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compare::{Tolerance, assert_matches};
    use crate::gdn::recurrent::recurrent_forward;
    use crate::rng::Xorshift64Star;

    /// `(q, k, v, log_decay, beta)` for a random test sequence.
    type RandomGdnSequence = (
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
        Vec<f32>,
        Vec<f32>,
    );

    fn random_sequence(
        rng: &mut Xorshift64Star,
        seq_len: usize,
        head_dim: usize,
    ) -> RandomGdnSequence {
        let q: Vec<Vec<f32>> = (0..seq_len)
            .map(|_| rng.vec_f32(head_dim, -1.0, 1.0))
            .collect();
        let k: Vec<Vec<f32>> = (0..seq_len)
            .map(|_| rng.vec_f32(head_dim, -1.0, 1.0))
            .collect();
        let v: Vec<Vec<f32>> = (0..seq_len)
            .map(|_| rng.vec_f32(head_dim, -1.0, 1.0))
            .collect();
        // Decay in log-space, negative (decay <= 1), a realistic range for
        // a sigmoid/softplus-derived gate.
        let log_decay = rng.vec_f32(seq_len, -0.5, 0.0);
        let beta = rng.vec_f32(seq_len, 0.1, 0.9);
        (q, k, v, log_decay, beta)
    }

    fn flatten_outputs(o: &[Vec<f32>]) -> Vec<f32> {
        o.iter().flatten().copied().collect()
    }

    /// The headline test: chunked and recurrent forms must agree closely
    /// on identical input. This is the project's stated correctness gate
    /// for Gated DeltaNet (`Tolerance::gdn_chunk_vs_recurrent`) and the
    /// single most important assertion in this crate — a failure here is a
    /// real finding about the derivation above, not a flaky test.
    #[test]
    fn chunked_and_recurrent_forms_agree_on_a_multi_chunk_sequence() {
        let mut rng = Xorshift64Star::new(2024);
        let head_dim = 32;
        let chunk_len = 16;
        let seq_len = 100; // spans multiple chunks, last chunk partial.
        let (q, k, v, log_decay, beta) = random_sequence(&mut rng, seq_len, head_dim);

        let (rec_out, rec_state) = recurrent_forward(head_dim, &q, &k, &v, &log_decay, &beta, None);
        let (chunk_out, chunk_state) =
            chunked_forward(head_dim, chunk_len, &q, &k, &v, &log_decay, &beta, None);

        // Gate on the full tolerance -- max-abs, max-rel, non-finite count
        // and cosine. Cosine alone would accept a uniformly scaled output,
        // which is exactly the drift this crate exists to catch.
        let tol = Tolerance::gdn_chunk_vs_recurrent();
        assert_matches(
            &flatten_outputs(&chunk_out),
            &flatten_outputs(&rec_out),
            &tol,
        );
        assert_matches(&chunk_state, &rec_state, &tol);
    }

    #[test]
    fn chunked_and_recurrent_agree_when_chunk_len_equals_sequence_length() {
        let mut rng = Xorshift64Star::new(7);
        let head_dim = 16;
        let seq_len = 24;
        let (q, k, v, log_decay, beta) = random_sequence(&mut rng, seq_len, head_dim);

        let (rec_out, _) = recurrent_forward(head_dim, &q, &k, &v, &log_decay, &beta, None);
        let (chunk_out, _) =
            chunked_forward(head_dim, seq_len, &q, &k, &v, &log_decay, &beta, None);

        assert_matches(
            &flatten_outputs(&chunk_out),
            &flatten_outputs(&rec_out),
            &Tolerance::gdn_chunk_vs_recurrent(),
        );
    }

    #[test]
    fn chunked_and_recurrent_agree_when_chunk_len_is_one() {
        // chunk_len=1 degenerates the chunked form to (almost) the
        // recurrent form token-by-token; this exercises the triangular
        // solve at its smallest non-trivial size (1x1).
        let mut rng = Xorshift64Star::new(8);
        let head_dim = 8;
        let seq_len = 10;
        let (q, k, v, log_decay, beta) = random_sequence(&mut rng, seq_len, head_dim);

        let (rec_out, _) = recurrent_forward(head_dim, &q, &k, &v, &log_decay, &beta, None);
        let (chunk_out, _) = chunked_forward(head_dim, 1, &q, &k, &v, &log_decay, &beta, None);

        assert_matches(
            &flatten_outputs(&chunk_out),
            &flatten_outputs(&rec_out),
            &Tolerance::gdn_chunk_vs_recurrent(),
        );
    }

    #[test]
    fn chunked_forward_respects_a_nonzero_initial_state() {
        let mut rng = Xorshift64Star::new(9);
        let head_dim = 12;
        let seq_len = 30;
        let (q, k, v, log_decay, beta) = random_sequence(&mut rng, seq_len, head_dim);
        let initial_state: GdnState = rng.vec_f32(head_dim * head_dim, -0.5, 0.5);

        let (rec_out, rec_state) = recurrent_forward(
            head_dim,
            &q,
            &k,
            &v,
            &log_decay,
            &beta,
            Some(&initial_state),
        );
        let (chunk_out, chunk_state) = chunked_forward(
            head_dim,
            8,
            &q,
            &k,
            &v,
            &log_decay,
            &beta,
            Some(&initial_state),
        );

        let tol = Tolerance::gdn_chunk_vs_recurrent();
        assert_matches(
            &flatten_outputs(&chunk_out),
            &flatten_outputs(&rec_out),
            &tol,
        );
        assert_matches(&chunk_state, &rec_state, &tol);
    }

    /// The equivalence test at Qwen3.6's actual Gated DeltaNet shape
    /// (`ModelConfig::qwen3_6_35b_a3b().gdn`: `head_dim=128`,
    /// `chunk_len=64`), rather than an arbitrary small size — this is the
    /// shape every real GDN layer in the model runs at.
    #[test]
    fn chunked_and_recurrent_agree_at_the_configured_qwen3_6_gdn_shape() {
        let cfg = xabe_model::ModelConfig::qwen3_6_35b_a3b();
        let head_dim = cfg.gdn.head_dim as usize;
        let chunk_len = cfg.gdn.chunk_len as usize;

        let mut rng = Xorshift64Star::new(36);
        let seq_len = chunk_len * 3 + 5; // several full chunks plus a partial one.
        let (q, k, v, log_decay, beta) = random_sequence(&mut rng, seq_len, head_dim);

        let (rec_out, rec_state) = recurrent_forward(head_dim, &q, &k, &v, &log_decay, &beta, None);
        let (chunk_out, chunk_state) =
            chunked_forward(head_dim, chunk_len, &q, &k, &v, &log_decay, &beta, None);

        let tol = Tolerance::gdn_chunk_vs_recurrent();
        assert_matches(
            &flatten_outputs(&chunk_out),
            &flatten_outputs(&rec_out),
            &tol,
        );
        assert_matches(&chunk_state, &rec_state, &tol);
    }

    #[test]
    fn chunk_boundary_placement_does_not_change_the_result() {
        // The same sequence split into differently-sized chunks must
        // produce the same output: chunking is an implementation detail of
        // how the (identical) recurrence is computed, not a change to it.
        let mut rng = Xorshift64Star::new(10);
        let head_dim = 8;
        let seq_len = 48;
        let (q, k, v, log_decay, beta) = random_sequence(&mut rng, seq_len, head_dim);

        let (out_a, _) = chunked_forward(head_dim, 16, &q, &k, &v, &log_decay, &beta, None);
        let (out_b, _) = chunked_forward(head_dim, 7, &q, &k, &v, &log_decay, &beta, None);

        assert_matches(
            &flatten_outputs(&out_a),
            &flatten_outputs(&out_b),
            &Tolerance::gdn_chunk_vs_recurrent(),
        );
    }
}
