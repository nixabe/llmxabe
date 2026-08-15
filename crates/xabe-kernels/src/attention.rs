//! Causal grouped-query attention reference: a plain two-pass softmax form
//! and an online-softmax streaming form, cross-checked against each other.
//!
//! Sized for Qwen3.6's Gated Attention layers: 16 query heads, 2 KV heads
//! (GQA ratio 8, see `AttentionConfig::gqa_ratio` in `xabe-model`), head
//! dimension 256. This is the oracle the eventual sm_75 flash-attention
//! port (`fattn-tile.cu` / `fattn-vec.cuh` in llama.cpp) gets checked
//! against, so the two forms here are kept algorithmically independent: the
//! naive form materializes the full score matrix, the streaming form never
//! does, and they must still agree.

/// Plain causal softmax attention for one (query head, batch) pair.
///
/// `q`: `[seq_len, head_dim]`, `k`/`v`: `[seq_len, head_dim]` — already the
/// key/value head selected for this query head via GQA broadcast (see
/// [`kv_head_for_query_head`]). Returns `[seq_len, head_dim]`.
///
/// Computes, for each query position `t`: attend only to keys `<= t`
/// (causal mask), `softmax(q_t . k_i / sqrt(head_dim))` weighting of `v_i`.
pub fn causal_attention_naive(q: &[Vec<f32>], k: &[Vec<f32>], v: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let seq_len = q.len();
    assert_eq!(k.len(), seq_len);
    assert_eq!(v.len(), seq_len);
    assert!(seq_len > 0, "causal_attention_naive: empty sequence");
    let head_dim = q[0].len();
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let mut out = vec![vec![0.0f32; head_dim]; seq_len];

    for t in 0..seq_len {
        // Pass 1: raw scores over the causal window [0, t].
        let mut scores = vec![0.0f32; t + 1];
        for (i, score) in scores.iter_mut().enumerate() {
            let dot: f32 = q[t].iter().zip(k[i].iter()).map(|(&a, &b)| a * b).sum();
            *score = dot * scale;
        }

        // Pass 2: numerically stable softmax over that window.
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for score in &mut scores {
            *score = (*score - max_score).exp();
            sum_exp += *score;
        }
        for score in &mut scores {
            *score /= sum_exp;
        }

        // Weighted sum of values.
        for (i, weight) in scores.iter().enumerate() {
            for d in 0..head_dim {
                out[t][d] += weight * v[i][d];
            }
        }
    }

    out
}

/// Online-softmax ("flash attention") streaming form of the same
/// computation: keys/values are visited once, in order, maintaining a
/// running max, running normalizer, and running weighted-value accumulator
/// that get rescaled as the running max increases (Milakov & Gimelshein's
/// online softmax, as used by every flash-attention kernel). Never
/// materializes the full score row.
pub fn causal_attention_streaming(q: &[Vec<f32>], k: &[Vec<f32>], v: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let seq_len = q.len();
    assert_eq!(k.len(), seq_len);
    assert_eq!(v.len(), seq_len);
    assert!(seq_len > 0, "causal_attention_streaming: empty sequence");
    let head_dim = q[0].len();
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let mut out = vec![vec![0.0f32; head_dim]; seq_len];

    for t in 0..seq_len {
        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;
        let mut acc = vec![0.0f32; head_dim];

        for i in 0..=t {
            let dot: f32 = q[t].iter().zip(k[i].iter()).map(|(&a, &b)| a * b).sum();
            let score = dot * scale;

            let new_max = running_max.max(score);
            // Rescale the existing accumulator and normalizer into the new
            // max's frame before folding in this key's contribution.
            let correction = if running_max == f32::NEG_INFINITY {
                0.0
            } else {
                (running_max - new_max).exp()
            };
            let weight = (score - new_max).exp();

            running_sum = running_sum * correction + weight;
            for d in 0..head_dim {
                acc[d] = acc[d] * correction + weight * v[i][d];
            }
            running_max = new_max;
        }

        for d in 0..head_dim {
            out[t][d] = acc[d] / running_sum;
        }
    }

    out
}

/// Maps a query head index to its key/value head under grouped-query
/// attention: query heads are partitioned into `kv_heads` contiguous
/// groups, all sharing one KV head, matching `AttentionConfig::gqa_ratio`
/// (`q_heads / kv_heads`) in `xabe-model`.
pub const fn kv_head_for_query_head(query_head: u32, q_heads: u32, kv_heads: u32) -> u32 {
    let group_size = q_heads / kv_heads;
    query_head / group_size
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compare::{Tolerance, assert_matches};
    use crate::rng::Xorshift64Star;

    fn random_seq(rng: &mut Xorshift64Star, seq_len: usize, head_dim: usize) -> Vec<Vec<f32>> {
        (0..seq_len)
            .map(|_| rng.vec_f32(head_dim, -1.0, 1.0))
            .collect()
    }

    fn flatten(v: &[Vec<f32>]) -> Vec<f32> {
        v.iter().flatten().copied().collect()
    }

    #[test]
    fn naive_and_streaming_forms_agree_on_random_input() {
        let mut rng = Xorshift64Star::new(123);
        let seq_len = 37;
        let head_dim = xabe_model::ModelConfig::qwen3_6_35b_a3b()
            .attention
            .head_dim as usize;
        let q = random_seq(&mut rng, seq_len, head_dim);
        let k = random_seq(&mut rng, seq_len, head_dim);
        let v = random_seq(&mut rng, seq_len, head_dim);

        let naive = causal_attention_naive(&q, &k, &v);
        let streaming = causal_attention_streaming(&q, &k, &v);

        // Both forms are provably the same softmax computed in a different
        // summation order; max-abs error is at the fp32 machine-epsilon
        // floor. Relative error is not a meaningful metric on its own
        // near-zero output components (a convex combination of many
        // random values can land arbitrarily close to zero, where any
        // nonzero rounding difference reads as a huge ratio) — so this
        // checks max-abs and cosine directly instead of relying on
        // `Tolerance`'s relative-error bound.
        let result = crate::compare::compare(&flatten(&streaming), &flatten(&naive));
        assert!(
            result.max_abs_error < 1e-4 && result.cosine_similarity > 1.0 - 1e-6,
            "naive vs streaming attention diverged beyond fp32 rounding: {result}"
        );
    }

    #[test]
    fn single_token_sequence_attends_only_to_itself() {
        let q = vec![vec![1.0f32, 0.0]];
        let k = vec![vec![1.0f32, 0.0]];
        let v = vec![vec![5.0f32, -5.0]];
        let out = causal_attention_naive(&q, &k, &v);
        assert_matches(&out[0], &v[0], &Tolerance::tight_fp32());
    }

    #[test]
    fn causal_mask_prevents_attending_to_future_tokens() {
        // Query at t=0 must reproduce v[0] exactly regardless of what comes
        // after it in the sequence (nothing else is in its causal window).
        let mut rng = Xorshift64Star::new(9);
        let head_dim = 16;
        let q = random_seq(&mut rng, 5, head_dim);
        let k = random_seq(&mut rng, 5, head_dim);
        let v = random_seq(&mut rng, 5, head_dim);
        let out = causal_attention_naive(&q, &k, &v);
        assert_matches(&out[0], &v[0], &Tolerance::tight_fp32());
    }

    #[test]
    fn attention_output_is_a_convex_combination_of_values() {
        // Softmax weights are non-negative and sum to one, so every output
        // dimension must lie within [min(v), max(v)] over the causal window.
        let mut rng = Xorshift64Star::new(44);
        let head_dim = 8;
        let seq_len = 6;
        let q = random_seq(&mut rng, seq_len, head_dim);
        let k = random_seq(&mut rng, seq_len, head_dim);
        let v = random_seq(&mut rng, seq_len, head_dim);
        let out = causal_attention_naive(&q, &k, &v);

        for t in 0..seq_len {
            for d in 0..head_dim {
                let window: Vec<f32> = v[..=t].iter().map(|row| row[d]).collect();
                let lo = window.iter().copied().fold(f32::INFINITY, f32::min);
                let hi = window.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                assert!(
                    out[t][d] >= lo - 1e-4 && out[t][d] <= hi + 1e-4,
                    "output[{t}][{d}]={} outside convex hull [{lo}, {hi}]",
                    out[t][d]
                );
            }
        }
    }

    #[test]
    fn gqa_16_to_2_maps_eight_query_heads_per_kv_head() {
        let attn = xabe_model::ModelConfig::qwen3_6_35b_a3b().attention;
        assert_eq!(attn.gqa_ratio(), 8);
        assert_eq!(kv_head_for_query_head(0, attn.q_heads, attn.kv_heads), 0);
        assert_eq!(kv_head_for_query_head(7, attn.q_heads, attn.kv_heads), 0);
        assert_eq!(kv_head_for_query_head(8, attn.q_heads, attn.kv_heads), 1);
        assert_eq!(kv_head_for_query_head(15, attn.q_heads, attn.kv_heads), 1);
    }
}
