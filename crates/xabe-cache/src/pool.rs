//! Per-group block allocator.
//!
//! Each cache group (attention KV, GDN recurrent state) gets its own
//! [`BlockPool`] at its own natural page size — see `docs/CACHE.md` and
//! AGENTS.md rule 1. The two pools are never merged and their free lists are
//! never sized against each other.
//!
//! Ported design, not code: this mirrors the free-list structure of vLLM's
//! `FreeKVCacheBlockQueue` (`vllm/v1/core/kv_cache_utils.py`) in spirit — a
//! pre-sized structure that never allocates on `alloc`/`free` — but uses a
//! flat `Vec`-backed stack instead of an intrusive doubly linked list, since
//! we don't need vLLM's O(1) arbitrary-position removal (eviction here goes
//! through [`crate::radix::RadixTree`], which returns whole block ids to
//! free rather than removing them from mid-list).

use crate::error::PoolExhausted;

/// Opaque handle to one block within a single [`BlockPool`].
///
/// Block ids are only comparable within the pool that issued them — a
/// `BlockId` from the attention pool and one from the GDN pool may carry the
/// same numeric value but refer to unrelated memory. Callers are expected to
/// keep pool identity out-of-band (e.g. by not mixing ids from different
/// pools in the same collection).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId(pub u32);

/// A single group's block allocator: one page size, one free list.
///
/// `alloc`, `free_block`, and `alloc_into` never allocate host memory after
/// construction — the free list is a `Vec<BlockId>` reserved to
/// `total_blocks` capacity up front and used as a stack, per AGENTS.md rule
/// 6. `alloc_n` is a convenience wrapper for setup/test code and does
/// allocate a new `Vec`; it is not meant for the per-step hot path.
#[derive(Debug)]
pub struct BlockPool {
    /// Human-readable group name, used only for diagnostics.
    label: &'static str,
    /// Bytes occupied by a single block/page in this group.
    page_bytes: u64,
    /// Total blocks this pool was constructed with.
    total: u32,
    /// Free blocks, used as a LIFO stack. Pre-reserved to `total` capacity.
    free: Vec<BlockId>,
}

impl BlockPool {
    /// Build a pool of `total_blocks` blocks, each `page_bytes` large.
    ///
    /// All blocks start free. The free list's backing `Vec` is allocated
    /// once, here, at its final capacity — no further allocation happens on
    /// `alloc`/`free_block` for the lifetime of the pool.
    pub fn new(label: &'static str, page_bytes: u64, total_blocks: u32) -> Self {
        let mut free = Vec::with_capacity(total_blocks as usize);
        for id in (0..total_blocks).rev() {
            free.push(BlockId(id));
        }
        Self {
            label,
            page_bytes,
            total: total_blocks,
            free,
        }
    }

    /// Take one free block. O(1), no allocation.
    pub fn alloc(&mut self) -> Result<BlockId, PoolExhausted> {
        self.free.pop().ok_or(PoolExhausted {
            label: self.label,
            total: self.total,
        })
    }

    /// Take `n` free blocks into `out`, appending. No allocation as long as
    /// `out` already has spare capacity (the hot-path-safe variant of
    /// `alloc_n`). On insufficient free blocks, nothing is taken and an
    /// error is returned.
    pub fn alloc_into(&mut self, n: u32, out: &mut Vec<BlockId>) -> Result<(), PoolExhausted> {
        if self.free.len() < n as usize {
            return Err(PoolExhausted {
                label: self.label,
                total: self.total,
            });
        }
        for _ in 0..n {
            out.push(self.free.pop().expect("checked len above"));
        }
        Ok(())
    }

    /// Convenience wrapper over [`Self::alloc_into`] that allocates a fresh
    /// `Vec`. Intended for setup and test code, not the per-step hot path.
    pub fn alloc_n(&mut self, n: u32) -> Result<Vec<BlockId>, PoolExhausted> {
        let mut out = Vec::with_capacity(n as usize);
        self.alloc_into(n, &mut out)?;
        Ok(out)
    }

    /// Return a block to the free list. O(1), no allocation: capacity for
    /// every possible block id was reserved in [`Self::new`].
    pub fn free_block(&mut self, id: BlockId) {
        debug_assert!(id.0 < self.total, "block id from a different pool");
        self.free.push(id);
    }

    /// Return many blocks to the free list.
    pub fn free_blocks<I: IntoIterator<Item = BlockId>>(&mut self, ids: I) {
        for id in ids {
            self.free_block(id);
        }
    }

    /// Number of currently free blocks.
    pub fn free_count(&self) -> u32 {
        self.free.len() as u32
    }

    /// Total blocks this pool was constructed with.
    pub fn total(&self) -> u32 {
        self.total
    }

    /// Bytes occupied by a single block in this group.
    pub fn page_bytes(&self) -> u64 {
        self.page_bytes
    }

    /// This pool's diagnostic label.
    pub fn label(&self) -> &'static str {
        self.label
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_pool_has_all_blocks_free() {
        let pool = BlockPool::new("attention", 4096, 8);
        assert_eq!(pool.free_count(), 8);
        assert_eq!(pool.total(), 8);
    }

    #[test]
    fn alloc_then_free_round_trips() {
        let mut pool = BlockPool::new("attention", 4096, 2);
        let a = pool.alloc().unwrap();
        let b = pool.alloc().unwrap();
        assert_ne!(a, b);
        assert_eq!(pool.free_count(), 0);
        assert!(pool.alloc().is_err());

        pool.free_block(a);
        assert_eq!(pool.free_count(), 1);
        let c = pool.alloc().unwrap();
        assert_eq!(c, a, "freed block should be reusable");
    }

    #[test]
    fn alloc_n_exhaustion_leaves_pool_untouched() {
        let mut pool = BlockPool::new("gdn", 1 << 20, 3);
        assert!(pool.alloc_n(4).is_err());
        // Nothing should have been taken on a failed bulk allocation.
        assert_eq!(pool.free_count(), 3);
    }

    #[test]
    fn alloc_into_does_not_grow_out_beyond_its_reserved_capacity() {
        let mut pool = BlockPool::new("attention", 4096, 4);
        let mut out = Vec::with_capacity(4);
        pool.alloc_into(4, &mut out).unwrap();
        assert_eq!(out.len(), 4);
        assert_eq!(pool.free_count(), 0);
    }

    #[test]
    fn free_blocks_bulk_returns_all() {
        let mut pool = BlockPool::new("attention", 4096, 4);
        let ids = pool.alloc_n(4).unwrap();
        assert_eq!(pool.free_count(), 0);
        pool.free_blocks(ids);
        assert_eq!(pool.free_count(), 4);
    }
}
