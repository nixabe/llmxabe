//! `moe_align_block_size`: converts scattered per-token top-k expert
//! assignments into contiguous, block-padded per-expert runs so a single
//! grouped GEMM launch can walk blocks and gather rows, instead of issuing
//! one GEMV per (token, selected expert) pair — the 1,080-launch problem
//! described in `MoeConfig::naive_gemvs_per_token` (`xabe-model`).
//!
//! Contract ported from vLLM's `moe_align_block_size`
//! (`vllm/model_executor/layers/fused_moe/moe_align_block_size.py`) and its
//! CUDA kernel (`csrc/libtorch_stable/moe/moe_align_sum_kernels.cu`,
//! `_moe_align_block_size`): per-expert token counts are padded up to the
//! next multiple of `block_size`; experts with zero assigned tokens get
//! *zero* blocks (not a padded-empty one — `CEILDIV(0, block_size) == 0` in
//! the CUDA kernel); tokens within an expert's run are placed in ascending
//! order of their flat `(token, k)` index (`token * top_k + k`); padding
//! slots in `sorted_token_ids` are filled with the sentinel `num_tokens *
//! top_k` (one past the last valid flat index — matches the CUDA kernel's
//! `SENTINEL`/`numel` and the Python docstring's worked example); trailing
//! `expert_ids` blocks beyond the last real block are filled with `-1`.

/// Sentinel token id used to mark a padding slot in `sorted_token_ids`.
/// Equal to `num_tokens * top_k` — one past the last valid flat
/// `(token, k)` index — so a consumer can check `id < num_tokens * top_k`
/// to know whether a slot is real.
pub const fn padding_sentinel(num_tokens: usize, top_k: usize) -> u32 {
    (num_tokens * top_k) as u32
}

/// Marks an `expert_ids` block as inactive (no tokens, skip in the grouped
/// GEMM).
pub const INACTIVE_EXPERT: i32 = -1;

/// The dispatch tables one grouped-GEMM launch consumes.
#[derive(Debug, Clone, PartialEq)]
pub struct MoeDispatch {
    /// Flat `(token, k)` indices, grouped by assigned expert and padded to
    /// `block_size` per expert; padding slots hold [`padding_sentinel`].
    /// Length is a multiple of `block_size`.
    pub sorted_token_ids: Vec<u32>,
    /// Expert id owning each `block_size`-sized block of `sorted_token_ids`
    /// (`expert_ids.len() == sorted_token_ids.len() / block_size`); trailing
    /// unused blocks are [`INACTIVE_EXPERT`].
    pub expert_ids: Vec<i32>,
    /// Number of valid (non-padding) slots at the front of
    /// `sorted_token_ids`, rounded up to `block_size` — i.e. the sum of
    /// every active expert's padded run length.
    pub num_tokens_post_pad: usize,
}

/// Builds the block-aligned dispatch tables for one batch's routing
/// decisions.
///
/// `topk_expert_ids` is `[num_tokens][top_k]`, matching
/// [`crate::moe::router::RoutingDecision::expert_ids`] stacked across a
/// batch; `block_size` is the grouped-GEMM tile width; `num_experts` bounds
/// the valid expert-id range (`MoeConfig::num_experts` — 256 for
/// Qwen3.6, see `xabe-model`).
///
/// # Panics
/// If `block_size` is zero, or any expert id is `>= num_experts`.
pub fn moe_align_block_size(
    topk_expert_ids: &[Vec<u32>],
    block_size: usize,
    num_experts: usize,
) -> MoeDispatch {
    assert!(
        block_size > 0,
        "moe_align_block_size: block_size must be non-zero"
    );

    let num_tokens = topk_expert_ids.len();
    let top_k = topk_expert_ids.first().map_or(0, Vec::len);
    for row in topk_expert_ids {
        assert_eq!(
            row.len(),
            top_k,
            "moe_align_block_size: ragged top_k across tokens"
        );
    }

    // Flatten to (flat_index, expert_id) in token-major, k-minor order —
    // this order is what "ascending flat index within an expert" sorts by.
    let flat: Vec<u32> = topk_expert_ids.iter().flatten().copied().collect();
    for &e in &flat {
        assert!(
            (e as usize) < num_experts,
            "moe_align_block_size: expert id {e} out of range"
        );
    }

    let mut counts = vec![0u32; num_experts];
    for &e in &flat {
        counts[e as usize] += 1;
    }

    let padded_counts: Vec<u32> = counts
        .iter()
        .map(|&c| c.div_ceil(block_size as u32) * block_size as u32)
        .collect();

    // Exclusive prefix sum over padded per-expert run lengths.
    let mut cumsum = vec![0u32; num_experts + 1];
    for e in 0..num_experts {
        cumsum[e + 1] = cumsum[e] + padded_counts[e];
    }
    let num_tokens_post_pad = cumsum[num_experts] as usize;

    let sentinel = padding_sentinel(num_tokens, top_k);
    let mut sorted_token_ids = vec![sentinel; num_tokens_post_pad];
    let mut running = vec![0u32; num_experts];
    for (flat_idx, &e) in flat.iter().enumerate() {
        let slot = cumsum[e as usize] + running[e as usize];
        sorted_token_ids[slot as usize] = flat_idx as u32;
        running[e as usize] += 1;
    }

    let num_blocks = num_tokens_post_pad / block_size;
    let mut expert_ids = vec![INACTIVE_EXPERT; num_blocks];
    for e in 0..num_experts {
        let first_block = (cumsum[e] as usize) / block_size;
        let last_block = (cumsum[e + 1] as usize) / block_size;
        for block in expert_ids.iter_mut().take(last_block).skip(first_block) {
            *block = e as i32;
        }
    }

    MoeDispatch {
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_vllm_docstring_worked_example() {
        // topk_ids = [[2,3,4],[1,2,4],[1,3,4],[1,2,3]], block_size=4, num_experts=4
        let topk = vec![
            vec![2u32, 3, 4],
            vec![1, 2, 4],
            vec![1, 3, 4],
            vec![1, 2, 3],
        ];
        // vLLM's docstring example uses experts 1..4 (num_experts=4 there
        // means valid ids 0..4, but the example only uses 1..=4 — adjust to
        // 5 experts here so id 4 is in range, matching the *placement*
        // logic the docstring demonstrates rather than its literal
        // num_experts value).
        let dispatch = moe_align_block_size(&topk, 4, 5);

        // Expert 1 appears at flat indices {3, 6, 9} (ascending).
        // Expert 2 appears at flat indices {0, 4, 10}.
        // Expert 3 appears at flat indices {1, 7, 11}.
        // Expert 4 appears at flat indices {2, 5, 8}.
        let sentinel = padding_sentinel(4, 3);
        assert_eq!(
            dispatch.sorted_token_ids,
            vec![
                3, 6, 9, sentinel, // expert 1's block (padded)
                0, 4, 10, sentinel, // expert 2's block (padded)
                1, 7, 11, sentinel, // expert 3's block (padded)
                2, 5, 8, sentinel, // expert 4's block (padded)
            ]
        );
        assert_eq!(dispatch.expert_ids, vec![1, 2, 3, 4]);
        assert_eq!(dispatch.num_tokens_post_pad, 16);
    }

    #[test]
    fn every_token_k_pair_appears_exactly_once_among_valid_slots() {
        let moe = xabe_model::ModelConfig::qwen3_6_35b_a3b()
            .moe()
            .expect("the routed model");
        let mut rng = crate::rng::Xorshift64Star::new(42);
        let num_tokens = 37;
        let top_k = moe.experts_per_token as usize;
        let num_experts = moe.num_experts as usize;
        let topk: Vec<Vec<u32>> = (0..num_tokens)
            .map(|_| {
                (0..top_k)
                    .map(|_| rng.next_u32_below(num_experts as u32))
                    .collect()
            })
            .collect();

        let dispatch = moe_align_block_size(&topk, 32, num_experts);
        let sentinel = padding_sentinel(num_tokens, top_k);

        let mut seen = vec![0u32; num_tokens * top_k];
        for &id in &dispatch.sorted_token_ids {
            if id == sentinel {
                continue;
            }
            assert!(
                (id as usize) < num_tokens * top_k,
                "id {id} out of the flat range"
            );
            seen[id as usize] += 1;
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "every (token, k) flat index must appear exactly once"
        );
    }

    #[test]
    fn every_block_belongs_to_exactly_one_expert_and_matches_its_tokens() {
        let mut rng = crate::rng::Xorshift64Star::new(43);
        let num_tokens = 20;
        let top_k = 4;
        let num_experts = 16;
        let block_size = 8;
        let topk: Vec<Vec<u32>> = (0..num_tokens)
            .map(|_| {
                (0..top_k)
                    .map(|_| rng.next_u32_below(num_experts as u32))
                    .collect()
            })
            .collect();
        let flat: Vec<u32> = topk.iter().flatten().copied().collect();

        let dispatch = moe_align_block_size(&topk, block_size, num_experts);
        let sentinel = padding_sentinel(num_tokens, top_k);

        for (block_idx, &expert) in dispatch.expert_ids.iter().enumerate() {
            let slot_range = block_idx * block_size..(block_idx + 1) * block_size;
            for slot in slot_range {
                let id = dispatch.sorted_token_ids[slot];
                if id == sentinel {
                    continue; // padding: ignored by consumers.
                }
                assert_ne!(
                    expert, INACTIVE_EXPERT,
                    "a real token landed in an inactive block"
                );
                assert_eq!(
                    flat[id as usize], expert as u32,
                    "token {id} in block {block_idx} does not belong to block's expert {expert}"
                );
            }
        }
    }

    #[test]
    fn padding_slots_are_the_sentinel_and_appear_only_after_an_experts_real_tokens() {
        let topk = vec![vec![0u32], vec![0], vec![1]];
        let dispatch = moe_align_block_size(&topk, 4, 2);
        let sentinel = padding_sentinel(3, 1);
        // Expert 0: 2 tokens padded to 4 -> [t0, t1, sentinel, sentinel]
        // Expert 1: 1 token padded to 4 -> [t2, sentinel, sentinel, sentinel]
        assert_eq!(dispatch.sorted_token_ids[2], sentinel);
        assert_eq!(dispatch.sorted_token_ids[3], sentinel);
        assert_eq!(dispatch.sorted_token_ids[5], sentinel);
    }

    #[test]
    fn experts_with_zero_tokens_consume_no_blocks() {
        // num_experts=4 but only experts 0 and 2 are ever selected.
        let topk = vec![vec![0u32], vec![2u32]];
        let dispatch = moe_align_block_size(&topk, 4, 4);
        // 2 experts * 1 padded block of 4 = 8 total slots, not 4*4=16.
        assert_eq!(dispatch.num_tokens_post_pad, 8);
        assert_eq!(dispatch.expert_ids, vec![0, 2]);
    }

    #[test]
    fn already_block_aligned_counts_need_no_padding() {
        let topk = vec![vec![0u32], vec![0], vec![0], vec![0]];
        let dispatch = moe_align_block_size(&topk, 4, 1);
        assert_eq!(dispatch.num_tokens_post_pad, 4);
        assert!(
            dispatch
                .sorted_token_ids
                .iter()
                .all(|&id| id != padding_sentinel(4, 1))
        );
    }

    #[test]
    #[should_panic]
    fn zero_block_size_panics() {
        moe_align_block_size(&[vec![0u32]], 0, 1);
    }

    #[test]
    #[should_panic]
    fn out_of_range_expert_id_panics() {
        moe_align_block_size(&[vec![5u32]], 4, 4);
    }
}
