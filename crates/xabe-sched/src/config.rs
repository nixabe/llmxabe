//! Scheduler tunables, validated at construction.
//!
//! Ported design from vLLM's `SchedulerConfig` / `Scheduler.schedule()`
//! (`vllm/v1/core/sched/scheduler.py`): a per-step token budget spent first
//! on running (decode) requests, then on waiting (prefill) requests, with a
//! watermark reserved for future admissions and a draft-token allowance for
//! speculative decoding entering the budget calculation up front.

use crate::error::SchedulerConfigError;

/// Both MTP and n-gram speculation commonly draft 2-3 tokens per step;
/// default to the conservative (larger) end so the budget is sized for the
/// drafted case rather than the accepted case.
pub const DEFAULT_DRAFT_TOKENS_PER_STEP: u32 = 3;

/// Fraction of total attention blocks reserved as headroom when admitting
/// from the waiting queue, so admission doesn't repeatedly evict and
/// preempt the same requests under load.
pub const DEFAULT_WATERMARK_FRACTION: f64 = 0.01;
/// Bound waiting requests so host-side admission cannot grow without limit.
pub const DEFAULT_WAITING_REQUESTS_MULTIPLIER: u32 = 4;

/// Scheduler configuration.
///
/// Constructed only through [`SchedulerConfig::new`], which enforces
/// AGENTS.md rule 3 as a hard, typed error — not a `debug_assert`, not a
/// log line.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchedulerConfig {
    /// Total tokens (decode + prefill, across all requests) schedulable in
    /// one step.
    token_budget: u32,
    /// Attention block size in tokens. Must match the
    /// `xabe_cache::CacheConfig` this scheduler is paired with.
    block_size: u32,
    /// Maximum number of requests concurrently decoding.
    max_concurrent_decodes: u32,
    /// Fraction of total attention blocks reserved as admission headroom.
    watermark_fraction: f64,
    /// Maximum speculative tokens produced per decode step. The draft source
    /// may be MTP, n-gram lookup, or none. Each drafted token
    /// consumes one slot of the per-step token budget and reserves one KV
    /// block-worth of capacity that may be discarded on rejection, so it
    /// must be budgeted for up front, not after the fact.
    draft_tokens_per_step: u32,
    /// Maximum requests held in the waiting queue; running requests are not
    /// counted because their memory is already reserved.
    max_waiting_requests: u32,
    /// Most prefill tokens one sequence may be granted in a single step.
    ///
    /// This is what keeps concurrent sessions concurrent. Without a cap the
    /// first prompt that does not fit in one step takes the whole budget
    /// every step until it finishes, and every other admitted session gets
    /// nothing — measured on one card with four sessions, first tokens
    /// arrived at 34, 71, 136 and 203 seconds, which is one prompt at a time
    /// dressed up as four slots.
    ///
    /// Set it to the cache's snapshot retention interval. A prefill pass may
    /// not straddle a snapshot boundary, so that interval is the widest pass
    /// the engine can issue, and slicing on it costs no throughput: the same
    /// tokens move at the same width, spread across sessions instead of
    /// stacked behind one. Zero means no cap, which is the old behaviour.
    prefill_slice: u32,
}

impl SchedulerConfig {
    /// Build a validated scheduler configuration.
    ///
    /// # Errors
    ///
    /// Returns [`SchedulerConfigError::BudgetTooSmall`] if
    /// `token_budget <= block_size + max_concurrent_decodes` — AGENTS.md
    /// rule 3. At that boundary, once `max_concurrent_decodes` requests are
    /// decoding they alone can consume the entire step budget, so no
    /// waiting request can ever start prefill and the engine serializes to
    /// batch 1 (benchmarked at ~7x slower upstream, LMCache Qwen3.6
    /// recipe).
    pub fn new(
        token_budget: u32,
        block_size: u32,
        max_concurrent_decodes: u32,
        watermark_fraction: f64,
        draft_tokens_per_step: u32,
    ) -> Result<Self, SchedulerConfigError> {
        if block_size == 0 {
            return Err(SchedulerConfigError::ZeroBlockSize);
        }
        if !(0.0..1.0).contains(&watermark_fraction) {
            return Err(SchedulerConfigError::WatermarkOutOfRange(
                watermark_fraction,
            ));
        }
        if token_budget <= block_size + max_concurrent_decodes {
            return Err(SchedulerConfigError::BudgetTooSmall {
                token_budget,
                block_size,
                max_concurrent_decodes,
            });
        }
        Ok(Self {
            token_budget,
            block_size,
            max_concurrent_decodes,
            watermark_fraction,
            draft_tokens_per_step,
            max_waiting_requests: max_concurrent_decodes
                .saturating_mul(DEFAULT_WAITING_REQUESTS_MULTIPLIER),
            prefill_slice: 0,
        })
    }

    /// [`SchedulerConfig::new`] with the project defaults for watermark
    /// fraction and draft token count.
    pub fn with_defaults(
        token_budget: u32,
        block_size: u32,
        max_concurrent_decodes: u32,
    ) -> Result<Self, SchedulerConfigError> {
        Self::new(
            token_budget,
            block_size,
            max_concurrent_decodes,
            DEFAULT_WATERMARK_FRACTION,
            DEFAULT_DRAFT_TOKENS_PER_STEP,
        )
    }

    /// Most prefill tokens one sequence may be granted in a step; 0 means
    /// no cap. See the field for why this is the retention interval.
    pub fn prefill_slice(&self) -> u32 {
        self.prefill_slice
    }

    /// Cap each sequence's per-step prefill grant, so several sessions share
    /// a step rather than queueing behind the first.
    #[must_use]
    pub fn with_prefill_slice(mut self, prefill_slice: u32) -> Self {
        self.prefill_slice = prefill_slice;
        self
    }

    pub fn token_budget(&self) -> u32 {
        self.token_budget
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    pub fn max_concurrent_decodes(&self) -> u32 {
        self.max_concurrent_decodes
    }

    pub fn watermark_fraction(&self) -> f64 {
        self.watermark_fraction
    }

    pub fn draft_tokens_per_step(&self) -> u32 {
        self.draft_tokens_per_step
    }

    pub fn max_waiting_requests(&self) -> u32 {
        self.max_waiting_requests
    }

    /// Tokens one decoding request consumes from the step budget: one real
    /// token plus every drafted token, since a rejected draft still had to
    /// be scheduled and materialized in KV before rejection could happen.
    ///
    /// Sizing the budget for this (the drafted case) rather than
    /// `1` (the accepted case) is what AGENTS.md's speculative-decode note
    /// requires — sizing for the accepted case causes admission to
    /// oscillate as the draft acceptance rate varies step to step.
    pub fn tokens_per_decode_step(&self) -> u32 {
        1 + self.draft_tokens_per_step
    }

    /// Watermark, in blocks, given a pool of `total_attention_blocks`.
    pub fn watermark_blocks(&self, total_attention_blocks: u32) -> u32 {
        (self.watermark_fraction * f64::from(total_attention_blocks)).floor() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_construct_successfully() {
        let cfg = SchedulerConfig::with_defaults(4096, 256, 32).unwrap();
        assert_eq!(cfg.token_budget(), 4096);
        assert_eq!(
            cfg.tokens_per_decode_step(),
            1 + DEFAULT_DRAFT_TOKENS_PER_STEP
        );
        assert_eq!(cfg.max_waiting_requests(), 128);
    }

    /// AGENTS.md rule 3, the named regression: a config at the exact
    /// boundary (`token_budget == block_size + max_concurrent_decodes`)
    /// must be rejected, because at that budget a full house of decoding
    /// requests alone consumes the whole step and no prefill can ever be
    /// admitted — the ~7x throughput collapse this rule cites.
    #[test]
    fn budget_at_block_size_plus_concurrency_boundary_would_serialize_to_batch_one_regression() {
        let err = SchedulerConfig::new(288, 256, 32, DEFAULT_WATERMARK_FRACTION, 0).unwrap_err();
        assert_eq!(
            err,
            SchedulerConfigError::BudgetTooSmall {
                token_budget: 288,
                block_size: 256,
                max_concurrent_decodes: 32,
            }
        );
    }

    /// The degenerate case named directly in AGENTS.md rule 3's prose:
    /// budget exactly equal to block size, with any concurrent decode
    /// capacity at all, must be rejected.
    #[test]
    fn budget_equal_to_bare_block_size_would_serialize_to_batch_one_regression() {
        assert!(SchedulerConfig::new(256, 256, 1, DEFAULT_WATERMARK_FRACTION, 0).is_err());
    }

    #[test]
    fn budget_below_the_boundary_is_also_rejected() {
        assert!(SchedulerConfig::new(100, 256, 32, DEFAULT_WATERMARK_FRACTION, 0).is_err());
    }

    #[test]
    fn budget_one_above_the_boundary_is_accepted() {
        assert!(SchedulerConfig::new(289, 256, 32, DEFAULT_WATERMARK_FRACTION, 0).is_ok());
    }

    #[test]
    fn zero_block_size_is_rejected() {
        assert_eq!(
            SchedulerConfig::new(4096, 0, 32, DEFAULT_WATERMARK_FRACTION, 0).unwrap_err(),
            SchedulerConfigError::ZeroBlockSize
        );
    }

    #[test]
    fn watermark_out_of_range_is_rejected() {
        assert!(SchedulerConfig::new(4096, 256, 32, 1.0, 0).is_err());
        assert!(SchedulerConfig::new(4096, 256, 32, -0.1, 0).is_err());
    }

    #[test]
    fn watermark_blocks_rounds_down() {
        let cfg = SchedulerConfig::new(4096, 256, 32, 0.1, 0).unwrap();
        assert_eq!(cfg.watermark_blocks(999), 99);
    }

    #[test]
    fn tokens_per_decode_step_includes_draft_tokens() {
        let cfg = SchedulerConfig::new(4096, 256, 32, 0.01, 3).unwrap();
        assert_eq!(cfg.tokens_per_decode_step(), 4);
    }
}
