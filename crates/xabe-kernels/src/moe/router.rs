//! MoE router: softmax over all experts, top-k selection, renormalization.
//!
//! Sized for Qwen3.6's `MoeConfig { num_experts: 256, experts_per_token: 8,
//! .. }` (see `xabe-model::config::MoeConfig`), though the functions here
//! are generic in `num_experts`/`k`.
//!
//! Tie-breaking on equal logits is deterministic (lower expert index wins)
//! so that this reference produces the *same* selection on every run and
//! on every platform — without that, a differential test against a GPU
//! kernel could fail on a tie that both implementations resolved
//! "correctly" but differently, which is a false alarm, not a bug.

/// One token's routing decision: which experts were selected and their
/// (renormalized) weights, in the order selected — highest logit first,
/// ties broken by lower expert index.
#[derive(Debug, Clone, PartialEq)]
pub struct RoutingDecision {
    /// Selected expert indices, length `k`.
    pub expert_ids: Vec<u32>,
    /// Routing weight per selected expert, same order as `expert_ids`,
    /// summing to `1.0` (renormalized over just the selected `k`, not all
    /// `num_experts`).
    pub weights: Vec<f32>,
}

/// Routes one token's router logits: softmax over all `num_experts`
/// entries, select the top `k` by softmax probability, then renormalize
/// those `k` weights to sum to `1.0`.
///
/// # Panics
/// If `logits` is empty, `k` is zero, or `k > logits.len()`.
pub fn route_token(logits: &[f32], k: usize) -> RoutingDecision {
    assert!(!logits.is_empty(), "route_token: logits must be non-empty");
    assert!(k > 0, "route_token: k must be non-zero");
    assert!(
        k <= logits.len(),
        "route_token: k must not exceed num_experts"
    );

    // Numerically stable softmax over all experts.
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp: f32 = exp.iter().sum();
    let probs: Vec<f32> = exp.iter().map(|&x| x / sum_exp).collect();

    // Select top-k by probability, ties broken by lower index: sort
    // (index, prob) pairs by (-prob, index) so the comparison is a total
    // order (probabilities can repeat; indices never do).
    let mut ranked: Vec<(u32, f32)> = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| (i as u32, p))
        .collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    ranked.truncate(k);

    let selected_sum: f32 = ranked.iter().map(|&(_, p)| p).sum();
    let expert_ids: Vec<u32> = ranked.iter().map(|&(i, _)| i).collect();
    let weights: Vec<f32> = if selected_sum > 0.0 {
        ranked.iter().map(|&(_, p)| p / selected_sum).collect()
    } else {
        // All-zero probability mass among the selected set cannot happen
        // with a real softmax (every entry is > 0), but guard division by
        // zero defensively rather than propagate NaN.
        vec![1.0 / k as f32; k]
    };

    RoutingDecision {
        expert_ids,
        weights,
    }
}

/// Routes every token in a batch. `logits` is `[num_tokens][num_experts]`.
pub fn route_batch(logits: &[Vec<f32>], k: usize) -> Vec<RoutingDecision> {
    logits.iter().map(|row| route_token(row, k)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Xorshift64Star;

    #[test]
    fn selects_exactly_k_experts_with_weights_summing_to_one() {
        let logits = [0.1f32, 3.0, -1.0, 2.5, 0.0, 1.0, -2.0, 4.0];
        let decision = route_token(&logits, 3);
        assert_eq!(decision.expert_ids.len(), 3);
        let sum: f32 = decision.weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "weights should sum to 1, got {sum}"
        );
    }

    #[test]
    fn selects_the_highest_logit_experts() {
        let logits = [0.1f32, 3.0, -1.0, 2.5, 0.0, 1.0, -2.0, 4.0];
        let decision = route_token(&logits, 3);
        // Highest logits are at indices 7 (4.0), 1 (3.0), 3 (2.5).
        let mut ids = decision.expert_ids.clone();
        ids.sort_unstable();
        assert_eq!(ids, vec![1, 3, 7]);
    }

    #[test]
    fn ties_are_broken_by_lower_expert_index_deterministically() {
        let logits = [1.0f32, 1.0, 1.0, 1.0];
        let decision = route_token(&logits, 2);
        assert_eq!(
            decision.expert_ids,
            vec![0, 1],
            "tie-break must prefer lower indices"
        );
    }

    #[test]
    fn tie_breaking_is_stable_across_repeated_calls() {
        let logits = [0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5];
        let first = route_token(&logits, 3);
        for _ in 0..10 {
            assert_eq!(
                route_token(&logits, 3),
                first,
                "routing must be deterministic on ties"
            );
        }
    }

    #[test]
    fn selected_weights_are_ordered_by_selection_probability() {
        let logits = [5.0f32, 1.0, 3.0];
        let decision = route_token(&logits, 3);
        assert_eq!(decision.expert_ids, vec![0, 2, 1]);
        assert!(decision.weights[0] > decision.weights[1]);
        assert!(decision.weights[1] > decision.weights[2]);
    }

    #[test]
    fn k_equal_to_num_experts_selects_everyone() {
        let logits = [1.0f32, 2.0, 3.0];
        let decision = route_token(&logits, 3);
        let mut ids = decision.expert_ids.clone();
        ids.sort_unstable();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn routing_matches_qwen3_6_top8_of_256_shape() {
        let moe = xabe_model::ModelConfig::qwen3_6_35b_a3b().moe;
        let num_experts = moe.num_experts as usize;
        let k = moe.experts_per_token as usize;

        let mut rng = Xorshift64Star::new(256);
        let logits = rng.vec_f32(num_experts, -10.0, 10.0);
        let decision = route_token(&logits, k);
        assert_eq!(decision.expert_ids.len(), k);
        assert_eq!(decision.weights.len(), k);
        let sum: f32 = decision.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
        // No duplicate experts within one token's selection.
        let mut ids = decision.expert_ids.clone();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), k);
    }

    #[test]
    fn route_batch_routes_each_row_independently() {
        let logits = vec![vec![3.0f32, 1.0, 2.0], vec![1.0f32, 3.0, 2.0]];
        let decisions = route_batch(&logits, 2);
        assert_eq!(decisions.len(), 2);
        assert_eq!(decisions[0].expert_ids, vec![0, 2]);
        assert_eq!(decisions[1].expert_ids, vec![1, 2]);
    }

    #[test]
    #[should_panic]
    fn k_greater_than_num_experts_panics() {
        route_token(&[1.0, 2.0], 3);
    }
}
