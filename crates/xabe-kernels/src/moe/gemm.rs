//! Grouped MoE GEMM reference: consumes the block-aligned dispatch tables
//! from [`crate::moe::dispatch::moe_align_block_size`], applies each
//! token's assigned expert's fused gate/up projection, SwiGLU, and down
//! projection, then accumulates each token's top-k contributions (weighted
//! by its routing weights) in fp32.
//!
//! The correctness statement for the whole fused MoE path is that this
//! *must* agree with [`naive_forward`], which recomputes the identical
//! result by looping directly over each token's routed experts with no
//! dispatch table involved — two structurally different code paths over
//! the same math. See the `equivalence` test below.

/// One expert's three matrices, row-major, `[out_dim x in_dim]` (matches
/// `MoeConfig::MATS_PER_EXPERT` = gate, up, down in `xabe-model`).
#[derive(Debug, Clone)]
pub struct ExpertWeights {
    /// `[intermediate x hidden]`.
    pub gate: Vec<f32>,
    /// `[intermediate x hidden]`.
    pub up: Vec<f32>,
    /// `[hidden x intermediate]`.
    pub down: Vec<f32>,
}

fn matvec(w: &[f32], out_dim: usize, in_dim: usize, x: &[f32]) -> Vec<f32> {
    assert_eq!(w.len(), out_dim * in_dim);
    assert_eq!(x.len(), in_dim);
    let mut out = vec![0.0f32; out_dim];
    for o in 0..out_dim {
        let row = &w[o * in_dim..o * in_dim + in_dim];
        out[o] = row.iter().zip(x.iter()).map(|(&wi, &xi)| wi * xi).sum();
    }
    out
}

/// One expert's forward pass: `down(swiglu(gate @ x, up @ x))`.
pub fn expert_mlp(
    expert: &ExpertWeights,
    x: &[f32],
    hidden: usize,
    intermediate: usize,
) -> Vec<f32> {
    let gate_out = matvec(&expert.gate, intermediate, hidden, x);
    let up_out = matvec(&expert.up, intermediate, hidden, x);
    let activated = crate::norm::swiglu(&gate_out, &up_out);
    matvec(&expert.down, hidden, intermediate, &activated)
}

/// Grouped-GEMM-style forward pass using the dispatch tables.
///
/// `hidden_states` is `[num_tokens][hidden]`. `flat_weights[flat_index]` is
/// the routing weight for `(token, k)` at `flat_index = token * top_k + k`
/// — the same flat indexing [`crate::moe::dispatch::moe_align_block_size`]
/// uses for `sorted_token_ids` entries (see
/// [`crate::moe::router::RoutingDecision`]). `experts[e]` is expert `e`'s
/// weights. Returns `[num_tokens][hidden]`, each token's top-k
/// contributions summed, accumulated in fp32.
pub fn grouped_forward(
    hidden_states: &[Vec<f32>],
    dispatch: &super::dispatch::MoeDispatch,
    flat_weights: &[f32],
    top_k: usize,
    experts: &[ExpertWeights],
    hidden: usize,
    intermediate: usize,
) -> Vec<Vec<f32>> {
    let num_tokens = hidden_states.len();
    let sentinel = super::dispatch::padding_sentinel(num_tokens, top_k);
    let mut out = vec![vec![0.0f32; hidden]; num_tokens];

    let block_size = if dispatch.expert_ids.is_empty() {
        1
    } else {
        dispatch.sorted_token_ids.len() / dispatch.expert_ids.len()
    };

    for (block_idx, &expert_id) in dispatch.expert_ids.iter().enumerate() {
        if expert_id == super::dispatch::INACTIVE_EXPERT {
            continue;
        }
        let expert = &experts[expert_id as usize];
        let slot_range = block_idx * block_size..(block_idx + 1) * block_size;
        for slot in slot_range {
            let flat_id = dispatch.sorted_token_ids[slot];
            if flat_id == sentinel {
                continue; // padding, no token here.
            }
            let token_idx = (flat_id as usize) / top_k;
            let weight = flat_weights[flat_id as usize];
            let contribution = expert_mlp(expert, &hidden_states[token_idx], hidden, intermediate);
            for (o, c) in out[token_idx].iter_mut().zip(contribution.iter()) {
                *o += weight * c;
            }
        }
    }

    out
}

/// Naive per-token loop over the token's own top-k routed experts, with no
/// dispatch table: the oracle [`grouped_forward`] is checked against.
pub fn naive_forward(
    hidden_states: &[Vec<f32>],
    routing: &[super::router::RoutingDecision],
    experts: &[ExpertWeights],
    hidden: usize,
    intermediate: usize,
) -> Vec<Vec<f32>> {
    assert_eq!(hidden_states.len(), routing.len());
    hidden_states
        .iter()
        .zip(routing.iter())
        .map(|(x, decision)| {
            let mut acc = vec![0.0f32; hidden];
            for (&expert_id, &weight) in decision.expert_ids.iter().zip(decision.weights.iter()) {
                let contribution =
                    expert_mlp(&experts[expert_id as usize], x, hidden, intermediate);
                for (a, c) in acc.iter_mut().zip(contribution.iter()) {
                    *a += weight * c;
                }
            }
            acc
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compare::compare;
    use crate::moe::dispatch::moe_align_block_size;
    use crate::moe::router::{RoutingDecision, route_batch};
    use crate::rng::Xorshift64Star;

    fn random_expert(
        rng: &mut Xorshift64Star,
        hidden: usize,
        intermediate: usize,
    ) -> ExpertWeights {
        ExpertWeights {
            gate: rng.vec_f32(intermediate * hidden, -0.3, 0.3),
            up: rng.vec_f32(intermediate * hidden, -0.3, 0.3),
            down: rng.vec_f32(hidden * intermediate, -0.3, 0.3),
        }
    }

    fn flatten(rows: &[Vec<f32>]) -> Vec<f32> {
        rows.iter().flatten().copied().collect()
    }

    /// The correctness statement for the whole fused MoE path: dispatch +
    /// grouped GEMM must reproduce the naive per-token loop exactly (up to
    /// floating-point reassociation), on a realistically-shaped batch.
    #[test]
    fn grouped_dispatch_path_matches_the_naive_per_token_loop() {
        let mut rng = Xorshift64Star::new(2025);
        let num_tokens = 23;
        let num_experts = 16;
        let top_k = 4;
        let hidden = 12;
        let intermediate = 20;
        let block_size = 8;

        let experts: Vec<ExpertWeights> = (0..num_experts)
            .map(|_| random_expert(&mut rng, hidden, intermediate))
            .collect();
        let hidden_states: Vec<Vec<f32>> = (0..num_tokens)
            .map(|_| rng.vec_f32(hidden, -1.0, 1.0))
            .collect();
        let router_logits: Vec<Vec<f32>> = (0..num_tokens)
            .map(|_| rng.vec_f32(num_experts, -3.0, 3.0))
            .collect();

        let routing = route_batch(&router_logits, top_k);
        let topk_ids: Vec<Vec<u32>> = routing.iter().map(|d| d.expert_ids.clone()).collect();
        let flat_weights: Vec<f32> = routing.iter().flat_map(|d| d.weights.clone()).collect();

        let dispatch = moe_align_block_size(&topk_ids, block_size, num_experts);

        let grouped = grouped_forward(
            &hidden_states,
            &dispatch,
            &flat_weights,
            top_k,
            &experts,
            hidden,
            intermediate,
        );
        let naive = naive_forward(&hidden_states, &routing, &experts, hidden, intermediate);

        let result = compare(&flatten(&grouped), &flatten(&naive));
        assert!(
            result.max_abs_error < 1e-3 && result.cosine_similarity > 1.0 - 1e-5,
            "grouped dispatch path diverged from the naive per-token loop: {result}"
        );
    }

    #[test]
    fn a_single_expert_used_by_every_token_reduces_to_a_shared_expert() {
        let mut rng = Xorshift64Star::new(2026);
        let hidden = 6;
        let intermediate = 10;
        let num_tokens = 5;
        let num_experts = 3;
        let top_k = 1;
        let experts: Vec<ExpertWeights> = (0..num_experts)
            .map(|_| random_expert(&mut rng, hidden, intermediate))
            .collect();
        let hidden_states: Vec<Vec<f32>> = (0..num_tokens)
            .map(|_| rng.vec_f32(hidden, -1.0, 1.0))
            .collect();
        // Route every token to expert 0 with full weight.
        let topk_ids: Vec<Vec<u32>> = (0..num_tokens).map(|_| vec![0u32]).collect();
        let flat_weights = vec![1.0f32; num_tokens];
        let routing: Vec<RoutingDecision> = (0..num_tokens)
            .map(|_| RoutingDecision {
                expert_ids: vec![0],
                weights: vec![1.0],
            })
            .collect();

        let dispatch = moe_align_block_size(&topk_ids, 4, num_experts);
        let grouped = grouped_forward(
            &hidden_states,
            &dispatch,
            &flat_weights,
            top_k,
            &experts,
            hidden,
            intermediate,
        );
        let naive = naive_forward(&hidden_states, &routing, &experts, hidden, intermediate);

        for (t, x) in hidden_states.iter().enumerate() {
            let direct = expert_mlp(&experts[0], x, hidden, intermediate);
            let g = compare(&grouped[t], &direct);
            assert!(g.max_abs_error < 1e-4, "token {t}: {g}");
            let n = compare(&naive[t], &direct);
            assert!(n.max_abs_error < 1e-4, "token {t}: {n}");
        }
    }

    #[test]
    fn zero_routing_weight_contributes_nothing() {
        let hidden = 4;
        let intermediate = 6;
        let expert = ExpertWeights {
            gate: vec![1.0; intermediate * hidden],
            up: vec![1.0; intermediate * hidden],
            down: vec![1.0; hidden * intermediate],
        };
        let x = vec![1.0f32; hidden];
        let full = expert_mlp(&expert, &x, hidden, intermediate);
        assert!(
            full.iter().any(|&v| v != 0.0),
            "sanity: expert output should be nonzero"
        );

        let hidden_states = vec![x];
        let routing = vec![RoutingDecision {
            expert_ids: vec![0],
            weights: vec![0.0],
        }];
        let naive = naive_forward(&hidden_states, &routing, &[expert], hidden, intermediate);
        assert!(naive[0].iter().all(|&v| v == 0.0));
    }
}
