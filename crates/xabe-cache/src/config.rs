//! Per-group page geometry, derived from [`xabe_model::ModelConfig`].
//!
//! This is the one place attention block size and GDN retention interval are
//! chosen. Everything downstream — [`crate::pool::BlockPool`] sizing,
//! [`crate::radix::RadixTree`] truncation — reads its geometry from here
//! rather than recomputing it, for the same reason `xabe-model::ModelConfig`
//! exists: a wrong number should be wrong in exactly one place.

use xabe_model::ModelConfig;

use crate::error::CacheConfigError;

/// Attention block size vLLM and llama.cpp both converge on for this class
/// of hardware: large enough to amortize kernel launch overhead, small
/// enough to keep prefix-cache hit granularity fine. Not load-bearing beyond
/// being a reasonable default — override via [`CacheConfig::new`].
pub const DEFAULT_ATTENTION_BLOCK_SIZE: u32 = 256;

/// Default GDN snapshot retention interval, in tokens.
///
/// One retained recurrent-state snapshot per 8 attention blocks at the
/// default block size. See AGENTS.md rule 2 and
/// `crate::config::tests::tiny_retention_interval_would_make_snapshots_dominate_the_pool_regression`
/// for why this is not derived from the attention block size.
pub const DEFAULT_GDN_RETENTION_INTERVAL: u32 = 2048;

/// Bytes per KV element. f16, the model's native cache dtype.
pub const DEFAULT_ELEM_SIZE: u64 = 2;

/// Per-group page geometry for one model.
///
/// Holds exactly the two independent parameters AGENTS.md rules 1 and 2
/// require: `attention_block_size` sets the attention group's natural page
/// size, and `gdn_retention_interval` (`R`) sets how often a GDN recurrent
/// state snapshot is retained for prefix reuse — a parameter with no
/// required relationship to `attention_block_size` beyond alignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheConfig {
    model: ModelConfig,
    /// Tokens per attention block.
    attention_block_size: u32,
    /// Tokens between retained GDN snapshots (`R`).
    gdn_retention_interval: u32,
    /// Bytes per KV element (e.g. 2 for f16).
    elem_size: u64,
}

impl CacheConfig {
    /// Build a validated cache configuration.
    ///
    /// Rejects a zero block size, a zero or misaligned retention interval,
    /// and a zero element size. The alignment check exists so that
    /// [`crate::radix::RadixTree::match_prefix`] can truncate a matched
    /// prefix down to a retained snapshot boundary with plain integer
    /// division — see AGENTS.md rule 2.
    pub fn new(
        model: ModelConfig,
        attention_block_size: u32,
        gdn_retention_interval: u32,
        elem_size: u64,
    ) -> Result<Self, CacheConfigError> {
        if attention_block_size == 0 {
            return Err(CacheConfigError::ZeroAttentionBlockSize);
        }
        if gdn_retention_interval == 0 {
            return Err(CacheConfigError::ZeroRetentionInterval);
        }
        if elem_size == 0 {
            return Err(CacheConfigError::ZeroElemSize);
        }
        if !gdn_retention_interval.is_multiple_of(attention_block_size) {
            return Err(CacheConfigError::RetentionNotBlockAligned {
                retention_interval: gdn_retention_interval,
                attention_block_size,
            });
        }
        Ok(Self {
            model,
            attention_block_size,
            gdn_retention_interval,
            elem_size,
        })
    }

    /// [`CacheConfig::new`] with the project defaults for block size,
    /// retention interval, and element size.
    pub fn with_defaults(model: ModelConfig) -> Result<Self, CacheConfigError> {
        Self::new(
            model,
            DEFAULT_ATTENTION_BLOCK_SIZE,
            DEFAULT_GDN_RETENTION_INTERVAL,
            DEFAULT_ELEM_SIZE,
        )
    }

    pub fn model(&self) -> &ModelConfig {
        &self.model
    }

    pub fn attention_block_size(&self) -> u32 {
        self.attention_block_size
    }

    pub fn gdn_retention_interval(&self) -> u32 {
        self.gdn_retention_interval
    }

    pub fn elem_size(&self) -> u64 {
        self.elem_size
    }

    /// Bytes held by one attention block: `block_size` tokens' worth of KV
    /// across every attention layer. This is the attention group's natural
    /// page size — never padded to the GDN group's page size (AGENTS.md
    /// rule 1).
    pub fn attention_page_bytes(&self) -> u64 {
        self.model.kv_bytes_per_token(self.elem_size) * u64::from(self.attention_block_size)
    }

    /// Bytes held by one GDN snapshot: the full recurrent state across every
    /// GDN layer, for one sequence, at one retained point in time. This is
    /// the GDN group's natural page size, independent of block size.
    pub fn gdn_page_bytes(&self) -> u64 {
        self.model.gdn_state_bytes_per_sequence()
    }

    /// How many attention blocks span one retention interval.
    ///
    /// `gdn_retention_interval` is validated at construction to be a
    /// multiple of `attention_block_size`, so this divides evenly.
    pub fn attention_blocks_per_retention_interval(&self) -> u32 {
        self.gdn_retention_interval / self.attention_block_size
    }

    /// Largest retention boundary at or below `matched_tokens`.
    ///
    /// This is the truncation rule from AGENTS.md rule 2: a matched prefix
    /// of KV blocks does not imply a usable GDN snapshot at the same length,
    /// because linear-attention state at position `p` cannot be sliced —
    /// only reused at a point where a snapshot was actually retained.
    pub fn retention_floor(&self, matched_tokens: u32) -> u32 {
        (matched_tokens / self.gdn_retention_interval) * self.gdn_retention_interval
    }

    /// Ratio of bytes spent on one GDN snapshot to bytes of attention KV
    /// accumulated over one retention interval.
    ///
    /// This is the number to watch when tuning `R`: a ratio near or above 1
    /// means snapshot overhead is comparable to (or dominates) the KV growth
    /// it's meant to sit alongside, which is exactly the failure mode
    /// AGENTS.md rule 2 describes (snapshots consuming ~80% of the pool at
    /// too-small an `R`). Lower `R` trades snapshot memory for finer GDN
    /// reuse granularity; this ratio quantifies that trade.
    pub fn snapshot_to_kv_ratio(&self) -> f64 {
        let kv_bytes_over_interval = self.model.kv_bytes_per_token(self.elem_size) as f64
            * f64::from(self.gdn_retention_interval);
        self.gdn_page_bytes() as f64 / kv_bytes_over_interval
    }
}

/// Free/total capacity for both cache groups, reported separately.
///
/// There is deliberately no field that sums attention and GDN capacity into
/// one number. AGENTS.md rule 1: a GDN block is on the order of ten times an
/// attention block's bytes (and vLLM's padded-to-max design produced ~7x
/// capacity misreporting on this architecture family) precisely because
/// someone summed them. Token capacity is counted from the attention group
/// only, since only attention grows with sequence position; the GDN group's
/// cost is a fixed per-slot number, not a per-token one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CapacityReport {
    pub attention_free_blocks: u32,
    pub attention_total_blocks: u32,
    pub attention_block_size: u32,
    pub gdn_free_slots: u32,
    pub gdn_total_slots: u32,
    pub gdn_bytes_per_slot: u64,
}

impl CapacityReport {
    /// Tokens of attention context the currently free attention blocks can
    /// hold. This is the only "capacity" number this crate produces; there
    /// is intentionally no analogous single figure spanning both groups.
    pub fn attention_token_capacity(&self) -> u64 {
        u64::from(self.attention_free_blocks) * u64::from(self.attention_block_size)
    }

    /// Fixed bytes reserved if one more sequence needs a live GDN state slot
    /// or one more retained snapshot. Constant regardless of how long that
    /// sequence's attention context is.
    pub fn gdn_bytes_per_additional_slot(&self) -> u64 {
        self.gdn_bytes_per_slot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> ModelConfig {
        ModelConfig::qwen3_6_35b_a3b()
    }

    #[test]
    fn defaults_construct_successfully() {
        let cfg = CacheConfig::with_defaults(model()).unwrap();
        assert_eq!(cfg.attention_block_size(), DEFAULT_ATTENTION_BLOCK_SIZE);
        assert_eq!(cfg.gdn_retention_interval(), DEFAULT_GDN_RETENTION_INTERVAL);
        assert_eq!(cfg.attention_blocks_per_retention_interval(), 8);
    }

    #[test]
    fn zero_block_size_is_rejected() {
        assert_eq!(
            CacheConfig::new(model(), 0, 2048, 2),
            Err(CacheConfigError::ZeroAttentionBlockSize)
        );
    }

    #[test]
    fn zero_retention_interval_is_rejected() {
        assert_eq!(
            CacheConfig::new(model(), 256, 0, 2),
            Err(CacheConfigError::ZeroRetentionInterval)
        );
    }

    #[test]
    fn misaligned_retention_interval_is_rejected() {
        // 2000 is not a multiple of 256.
        assert_eq!(
            CacheConfig::new(model(), 256, 2000, 2),
            Err(CacheConfigError::RetentionNotBlockAligned {
                retention_interval: 2000,
                attention_block_size: 256,
            })
        );
    }

    /// AGENTS.md rule 1: attention and GDN pages must never share a page
    /// size, or the smaller group's capacity gets misreported by the ratio
    /// between the two natural sizes (~7x in the upstream incident this
    /// rule cites). Ported design: vLLM's `unify_kv_cache_spec_page_size`
    /// existing at all is the bug shape this test forbids reintroducing.
    #[test]
    fn attention_and_gdn_page_sizes_are_never_unified_to_a_shared_max_regression() {
        let cfg = CacheConfig::with_defaults(model()).unwrap();
        let attn_page = cfg.attention_page_bytes();
        let gdn_page = cfg.gdn_page_bytes();

        // The two pages must differ substantially — that's the whole point
        // of keeping them separate. If a future change makes these equal
        // (e.g. by padding one to the other), this is the bug rule 1
        // exists to catch.
        assert_ne!(attn_page, gdn_page);
        let ratio = gdn_page as f64 / attn_page as f64;
        assert!(
            ratio > 5.0,
            "expected GDN page to dwarf an attention page (got ratio {ratio}); \
             if this shrank toward 1.0, something is unifying page geometry"
        );
    }

    /// AGENTS.md rule 1, other direction: a capacity report must not fuse
    /// the two groups into one token figure. Changing GDN slot availability
    /// must have zero effect on reported attention token capacity, and vice
    /// versa — they are unrelated resources.
    #[test]
    fn capacity_report_keeps_attention_and_gdn_capacity_independent_regression() {
        let report_a = CapacityReport {
            attention_free_blocks: 100,
            attention_total_blocks: 200,
            attention_block_size: 256,
            gdn_free_slots: 0,
            gdn_total_slots: 50,
            gdn_bytes_per_slot: 60 * 1024 * 1024,
        };
        let report_b = CapacityReport {
            gdn_free_slots: 50,
            ..report_a
        };

        assert_eq!(report_a.attention_token_capacity(), 100 * 256);
        assert_eq!(
            report_a.attention_token_capacity(),
            report_b.attention_token_capacity(),
            "attention token capacity must not depend on GDN slot count"
        );
    }

    /// AGENTS.md rule 2: retention interval is independent of block size,
    /// and shrinking it toward block size is exactly the upstream failure
    /// (snapshots consuming ~80% of the pool, hit rate 85% -> 75%, ~18%
    /// throughput loss, p99 ~3x worse — vLLM PR #45845). This test doesn't
    /// reproduce those measurements (no GPU, no workload here) but it does
    /// pin the *mechanism*: shrinking R toward block_size must make the
    /// snapshot-to-KV ratio blow up, and the chosen default must keep it
    /// well below that regime.
    #[test]
    fn tiny_retention_interval_would_make_snapshots_dominate_the_pool_regression() {
        let default_cfg = CacheConfig::with_defaults(model()).unwrap();
        let worst_case_cfg = CacheConfig::new(model(), 256, 256, 2).unwrap();

        let default_ratio = default_cfg.snapshot_to_kv_ratio();
        let worst_ratio = worst_case_cfg.snapshot_to_kv_ratio();

        assert!(
            default_ratio < 2.0,
            "default R={} should keep snapshot overhead well under KV growth, got ratio {default_ratio}",
            DEFAULT_GDN_RETENTION_INTERVAL
        );
        assert!(
            worst_ratio > 10.0,
            "R == block_size should make snapshot bytes dwarf per-interval KV bytes, got ratio {worst_ratio}"
        );
        assert!(
            worst_ratio > default_ratio * 5.0,
            "shrinking R toward block_size must sharply worsen the ratio"
        );
    }

    #[test]
    fn retention_floor_rounds_down_to_the_nearest_interval() {
        let cfg = CacheConfig::with_defaults(model()).unwrap(); // R = 2048
        assert_eq!(cfg.retention_floor(0), 0);
        assert_eq!(cfg.retention_floor(2047), 0);
        assert_eq!(cfg.retention_floor(2048), 2048);
        assert_eq!(cfg.retention_floor(4095), 2048);
        assert_eq!(cfg.retention_floor(4096), 4096);
    }
}
