//! Error types for cache configuration and block allocation.

use thiserror::Error;

/// A structurally invalid [`crate::config::CacheConfig`].
///
/// Both variants below exist to enforce AGENTS.md rules 1 and 2: the two
/// cache groups have independent geometry, and the GDN retention interval
/// must be expressible on attention block boundaries so that a prefix match
/// can be truncated to a retained snapshot deterministically (see
/// [`crate::radix::RadixTree::match_prefix`]).
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CacheConfigError {
    /// `attention_block_size` was zero, so no page geometry can be derived.
    #[error("attention_block_size must be non-zero")]
    ZeroAttentionBlockSize,
    /// `gdn_retention_interval` was zero, so no snapshot would ever be taken
    /// and every prefix match on the GDN group would truncate to zero.
    #[error("gdn_retention_interval must be non-zero")]
    ZeroRetentionInterval,
    /// The retention interval is not a whole number of attention blocks.
    ///
    /// vLLM's coordinator enforces the equivalent constraint
    /// (`retention_interval % scheduler_block_size == 0`) for the same
    /// reason: retention boundaries must land on block boundaries or the
    /// truncation in [`crate::radix::RadixTree::match_prefix`] has no clean
    /// definition.
    #[error(
        "gdn_retention_interval ({retention_interval}) must be a multiple of \
         attention_block_size ({attention_block_size})"
    )]
    RetentionNotBlockAligned {
        retention_interval: u32,
        attention_block_size: u32,
    },
    /// `elem_size` (bytes per KV element) was zero.
    #[error("elem_size must be non-zero")]
    ZeroElemSize,
}

/// A block allocator with no free blocks left.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
#[error("{label} block pool exhausted: 0 of {total} blocks free")]
pub struct PoolExhausted {
    pub label: &'static str,
    pub total: u32,
}
