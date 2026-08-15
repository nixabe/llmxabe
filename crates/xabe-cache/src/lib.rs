//! Two-group KV/GDN pager and shared prefix radix tree.
//!
//! This crate exists to enforce AGENTS.md rules 1 and 2, both paid for in
//! someone else's production incident:
//!
//! 1. Attention KV and Gated DeltaNet recurrent state are never given a
//!    shared page size. [`config::CacheConfig`] derives each group's
//!    natural page size independently from [`xabe_model::ModelConfig`], and
//!    [`pool::BlockPool`] allocates one pool per group. [`config::CapacityReport`]
//!    reports the two groups' capacity separately — there is no combined
//!    "capacity" field to accidentally read as one number.
//! 2. The GDN snapshot retention interval `R` is an independent knob, not
//!    derived from the attention block size, and a prefix match against the
//!    GDN group truncates to the largest retained snapshot boundary at or
//!    below the matched length. See [`radix::RadixTree::match_prefix`] and
//!    [`config::CacheConfig::retention_floor`].
//!
//! Start at [`config::CacheConfig`] for geometry, [`pool::BlockPool`] for
//! allocation, and [`radix::RadixTree`] for prefix sharing.

pub mod config;
pub mod error;
pub mod pool;
pub mod radix;

pub use config::{CacheConfig, CapacityReport};
pub use error::{CacheConfigError, PoolExhausted};
pub use pool::{BlockId, BlockPool};
pub use radix::{BlockHash, GdnSlot, PrefixBlock, PrefixMatch, RadixTree, hash_block};
