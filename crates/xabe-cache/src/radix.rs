//! Prefix radix tree over token sequences, shared across workers.
//!
//! Ported design, not code, from vLLM's block-hash-chain prefix cache
//! (`vllm/v1/core/kv_cache_utils.py`'s `hash_block_tokens` /
//! `BlockHashWithGroupId`, and `vllm/v1/core/block_pool.py`'s hash-keyed
//! block table): each cached block's hash chains in its parent's hash, so
//! two blocks only compare equal if their entire prefix, not just their own
//! tokens, matches. That makes the "tree" a flat hash map keyed by chained
//! hash rather than a literal trie — matching a hash is already sufficient
//! proof of a full-prefix match, so there is no need for an explicit
//! branching structure beyond parent/child links for eviction bookkeeping.
//!
//! The one piece that is *not* upstream vLLM: AGENTS.md rule 2. Recurrent
//! (GDN) state at position `p` summarizes `[0, p)` in full and cannot be
//! sliced the way KV blocks can — a matched prefix is only reusable for the
//! GDN group at the retained snapshot boundaries it actually stored. See
//! [`RadixTree::match_prefix`] and `PrefixMatch::gdn_matched_tokens`.

use rustc_hash::{FxHashMap, FxHashSet};
use std::hash::{Hash, Hasher};

use parking_lot::RwLock;
use rustc_hash::FxHasher;

use crate::pool::BlockId;

/// Chained content hash of a block: `hash(parent_hash, tokens)`.
///
/// Because the parent hash feeds into the child hash, identical token
/// content occurring at two different points in two different sequences
/// only produces the same `BlockHash` if their entire prefix, block for
/// block, was identical too.
pub type BlockHash = u64;

/// GDN snapshot handle. A distinct alias of [`BlockId`] because it indexes
/// into the GDN group's [`crate::pool::BlockPool`], a different id space
/// from the attention group's blocks.
pub type GdnSlot = BlockId;

/// Sentinel parent hash for the first block of any sequence.
pub const ROOT_HASH: BlockHash = 0;

/// Compute a chained block hash. `parent` is [`ROOT_HASH`] for a sequence's
/// first block, or the previous block's hash otherwise.
pub fn hash_block(parent: BlockHash, tokens: &[u32]) -> BlockHash {
    let mut hasher = FxHasher::default();
    parent.hash(&mut hasher);
    tokens.hash(&mut hasher);
    hasher.finish()
}

/// One block of a prefix being inserted into the tree.
#[derive(Debug, Clone)]
pub struct PrefixBlock {
    /// This block's chained hash (see [`hash_block`]).
    pub hash: BlockHash,
    /// The attention-group block holding this block's KV.
    pub block: BlockId,
    /// A retained GDN snapshot, if this block's end position is a retention
    /// boundary (`token_end % R == 0`). `None` for every other block.
    pub gdn_snapshot: Option<GdnSlot>,
}

#[derive(Debug)]
struct Node {
    parent: BlockHash,
    children: FxHashSet<BlockHash>,
    block: BlockId,
    token_end: u32,
    gdn_snapshot: Option<GdnSlot>,
    ref_count: u32,
    pinned: bool,
    /// Logical clock value at last release (ref_count hit zero). Used only
    /// to order LRU eviction; irrelevant while ref_count > 0.
    last_used: u64,
}

struct Inner {
    nodes: FxHashMap<BlockHash, Node>,
    /// Hashes of nodes whose parent is the implicit root, tracked
    /// separately since [`ROOT_HASH`] itself has no `Node`.
    root_children: FxHashSet<BlockHash>,
    clock: u64,
}

/// Result of a longest-prefix-match lookup.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PrefixMatch {
    /// Hash of the last matched node, if any matched. Pass this (and every
    /// hash that matched, in order) to [`RadixTree::incr_ref_chain`] /
    /// [`RadixTree::decr_ref_chain`] to protect the whole matched chain.
    pub matched_hash: Option<BlockHash>,
    /// Tokens covered by the matched attention blocks.
    pub matched_tokens: u32,
    /// Attention blocks for the matched prefix, in order.
    pub blocks: Vec<BlockId>,
    /// Reusable GDN snapshot, if any retained snapshot falls within the
    /// matched range.
    pub gdn_snapshot: Option<GdnSlot>,
    /// Tokens the reused GDN snapshot actually covers.
    ///
    /// This is AGENTS.md rule 2 made concrete: it is the largest retained
    /// snapshot boundary `<= matched_tokens`, not `matched_tokens` itself.
    /// A caller that reuses `gdn_snapshot` and then treats it as valid for
    /// `matched_tokens` positions instead of `gdn_matched_tokens` positions
    /// is reintroducing the bug this field exists to prevent.
    pub gdn_matched_tokens: u32,
}

/// Prefix radix tree, held in host RAM and shared across workers.
///
/// All mutation and lookup goes through a single [`parking_lot::RwLock`], so
/// concurrent workers can share one cache without duplicating prefill work
/// for a shared system prompt.
pub struct RadixTree {
    block_size: u32,
    inner: RwLock<Inner>,
}

impl RadixTree {
    pub fn new(block_size: u32) -> Self {
        Self {
            block_size,
            inner: RwLock::new(Inner {
                nodes: FxHashMap::default(),
                root_children: FxHashSet::default(),
                clock: 0,
            }),
        }
    }

    /// Insert a prefix chain with its cached blocks.
    ///
    /// `prefix` must be in sequence order, each hash chained from the
    /// previous (see [`hash_block`]). Nodes that already exist (a shared
    /// prefix with something already cached) are left untouched rather than
    /// overwritten — their ref count and pin state must survive a re-insert
    /// of the same prefix by a different worker.
    pub fn insert(&self, prefix: &[PrefixBlock]) {
        let mut inner = self.inner.write();
        let mut parent_hash = ROOT_HASH;
        let mut token_start = 0u32;

        for pb in prefix {
            let token_end = token_start + self.block_size;

            if let std::collections::hash_map::Entry::Vacant(entry) = inner.nodes.entry(pb.hash) {
                entry.insert(Node {
                    parent: parent_hash,
                    children: FxHashSet::default(),
                    block: pb.block,
                    token_end,
                    gdn_snapshot: pb.gdn_snapshot,
                    ref_count: 0,
                    pinned: false,
                    last_used: 0,
                });
                if parent_hash == ROOT_HASH {
                    inner.root_children.insert(pb.hash);
                } else if let Some(parent_node) = inner.nodes.get_mut(&parent_hash) {
                    parent_node.children.insert(pb.hash);
                }
            }

            parent_hash = pb.hash;
            token_start = token_end;
        }
    }

    /// Longest-prefix-match lookup.
    ///
    /// `hashes` is the querying sequence's own chained hashes, in order.
    /// Matching stops at the first hash absent from the tree (equivalently,
    /// the first point the two sequences' content diverged, since the chain
    /// makes a hash match imply the whole prefix up to it matched).
    pub fn match_prefix(&self, hashes: &[BlockHash]) -> PrefixMatch {
        let inner = self.inner.read();
        let mut result = PrefixMatch::default();

        for &h in hashes {
            let Some(node) = inner.nodes.get(&h) else {
                break;
            };
            result.blocks.push(node.block);
            result.matched_tokens = node.token_end;
            result.matched_hash = Some(h);
            if let Some(slot) = node.gdn_snapshot {
                result.gdn_snapshot = Some(slot);
                result.gdn_matched_tokens = node.token_end;
            }
        }

        result
    }

    /// Increment the reference count of every node along a matched chain.
    ///
    /// Call with the full list of hashes a request is actively using (e.g.
    /// `PrefixMatch::blocks`' corresponding hashes), not just the leaf —
    /// every block in the chain must be protected from eviction while in
    /// use, not only the deepest one.
    pub fn incr_ref_chain(&self, hashes: &[BlockHash]) {
        let mut inner = self.inner.write();
        for h in hashes {
            if let Some(n) = inner.nodes.get_mut(h) {
                n.ref_count += 1;
            }
        }
    }

    /// Decrement the reference count of every node along a chain, releasing
    /// any that reach zero for LRU eviction.
    pub fn decr_ref_chain(&self, hashes: &[BlockHash]) {
        let mut inner = self.inner.write();
        inner.clock += 1;
        let clock = inner.clock;
        for h in hashes {
            if let Some(n) = inner.nodes.get_mut(h) {
                n.ref_count = n.ref_count.saturating_sub(1);
                if n.ref_count == 0 {
                    n.last_used = clock;
                }
            }
        }
    }

    /// Mark every node in a chain as pinned: exempt from eviction regardless
    /// of reference count. Intended for the system-prompt prefix, prefilled
    /// once per worker at boot.
    pub fn pin_chain(&self, hashes: &[BlockHash]) {
        let mut inner = self.inner.write();
        for h in hashes {
            if let Some(n) = inner.nodes.get_mut(h) {
                n.pinned = true;
            }
        }
    }

    /// Remove the pinned flag from every node in a chain.
    pub fn unpin_chain(&self, hashes: &[BlockHash]) {
        let mut inner = self.inner.write();
        for h in hashes {
            if let Some(n) = inner.nodes.get_mut(h) {
                n.pinned = false;
            }
        }
    }

    pub fn is_pinned(&self, hash: BlockHash) -> bool {
        self.inner.read().nodes.get(&hash).is_some_and(|n| n.pinned)
    }

    pub fn ref_count(&self, hash: BlockHash) -> u32 {
        self.inner
            .read()
            .nodes
            .get(&hash)
            .map_or(0, |n| n.ref_count)
    }

    pub fn contains(&self, hash: BlockHash) -> bool {
        self.inner.read().nodes.contains_key(&hash)
    }

    pub fn len(&self) -> usize {
        self.inner.read().nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Evict up to `max_nodes` unreferenced, unpinned leaf nodes, LRU first.
    ///
    /// Only leaves (nodes with no cached children) are ever evicted in one
    /// pass, mirroring vLLM's tail-of-chain eviction: a node with children
    /// still cached is still useful as a prefix for those children, even if
    /// nothing currently references it directly. Evicting a leaf may turn
    /// its parent into a leaf, which becomes eligible on a subsequent call.
    ///
    /// Returns the attention blocks freed, in eviction order.
    ///
    /// This does a linear scan over all nodes per evicted block, which is
    /// adequate for a maintenance operation triggered when a pool runs low,
    /// but is not the hot path (allocation/free on the block pools is; see
    /// [`crate::pool::BlockPool`]). Not benchmarked at scale — see the
    /// crate's top-level report for what was and wasn't measured.
    pub fn evict_unreferenced(&self, max_nodes: usize) -> Vec<BlockId> {
        let mut inner = self.inner.write();
        let mut evicted = Vec::new();

        while evicted.len() < max_nodes {
            let candidate = inner
                .nodes
                .iter()
                .filter(|(_, n)| n.children.is_empty() && n.ref_count == 0 && !n.pinned)
                .min_by_key(|(_, n)| n.last_used)
                .map(|(h, _)| *h);

            let Some(hash) = candidate else {
                break;
            };

            let node = inner.nodes.remove(&hash).expect("just found it");
            evicted.push(node.block);

            if node.parent == ROOT_HASH {
                inner.root_children.remove(&hash);
            } else if let Some(parent) = inner.nodes.get_mut(&node.parent) {
                parent.children.remove(&hash);
            }
        }

        evicted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a linear chain of `n` blocks, each `block_size` tokens, with a
    /// GDN snapshot retained every `retention_interval` tokens.
    fn build_chain(
        block_size: u32,
        retention_interval: u32,
        n_blocks: u32,
    ) -> (Vec<PrefixBlock>, Vec<BlockHash>) {
        let mut parent = ROOT_HASH;
        let mut prefix = Vec::new();
        let mut hashes = Vec::new();
        for i in 0..n_blocks {
            // Distinct "tokens" per block so hashes differ; content doesn't
            // matter for these tests beyond uniqueness.
            let tokens = [i, i + 1000];
            let hash = hash_block(parent, &tokens);
            let token_end = (i + 1) * block_size;
            let gdn_snapshot = token_end
                .is_multiple_of(retention_interval)
                .then_some(BlockId(i));
            prefix.push(PrefixBlock {
                hash,
                block: BlockId(i),
                gdn_snapshot,
            });
            hashes.push(hash);
            parent = hash;
        }
        (prefix, hashes)
    }

    #[test]
    fn insert_then_match_returns_the_full_chain() {
        let tree = RadixTree::new(256);
        let (prefix, hashes) = build_chain(256, 2048, 4);
        tree.insert(&prefix);

        let m = tree.match_prefix(&hashes);
        assert_eq!(m.matched_tokens, 4 * 256);
        assert_eq!(m.blocks.len(), 4);
        assert_eq!(m.matched_hash, Some(hashes[3]));
    }

    #[test]
    fn match_stops_at_first_divergence() {
        let tree = RadixTree::new(256);
        let (prefix, hashes) = build_chain(256, 2048, 4);
        tree.insert(&prefix);

        let mut query = hashes[0..2].to_vec();
        query.push(0xDEAD_BEEF); // diverges after 2 matched blocks
        query.push(hashes[3]); // would-be match, but chain already broke

        let m = tree.match_prefix(&query);
        assert_eq!(m.matched_tokens, 2 * 256);
        assert_eq!(m.blocks.len(), 2);
    }

    #[test]
    fn empty_query_matches_nothing() {
        let tree = RadixTree::new(256);
        let (prefix, _) = build_chain(256, 2048, 4);
        tree.insert(&prefix);

        let m = tree.match_prefix(&[]);
        assert_eq!(m.matched_tokens, 0);
        assert!(m.blocks.is_empty());
        assert!(m.gdn_snapshot.is_none());
    }

    /// AGENTS.md rule 2, the load-bearing test for this module: a matched
    /// KV prefix does not imply a GDN snapshot at the same length. GDN
    /// reuse must truncate down to the largest retained boundary at or
    /// below the matched length, exactly like `CacheConfig::retention_floor`
    /// computes — this test proves the tree actually enforces that via
    /// real inserted nodes, not just the arithmetic helper.
    #[test]
    fn gdn_reuse_truncates_to_retained_snapshot_boundary_not_full_match_length_regression() {
        let block_size = 256;
        let retention_interval = 2048; // snapshot every 8 blocks
        let tree = RadixTree::new(block_size);
        // 20 blocks = 5120 tokens; snapshots retained at block 8 (2048) and
        // block 16 (4096). Block 20 (5120) is not itself a boundary.
        let (prefix, hashes) = build_chain(block_size, retention_interval, 20);
        tree.insert(&prefix);

        // Full match: 20/20 blocks, 5120 tokens matched.
        let m = tree.match_prefix(&hashes);
        assert_eq!(m.matched_tokens, 5120);
        assert_eq!(
            m.gdn_matched_tokens, 4096,
            "GDN reuse must truncate to the last retained snapshot (4096), \
             not the full matched length (5120)"
        );

        // Partial match landing between two snapshot boundaries (18 blocks
        // = 4608 tokens, between the 4096 and the next-would-be 6144
        // boundary) must still only offer the 4096 snapshot.
        let partial = &hashes[0..18];
        let m2 = tree.match_prefix(partial);
        assert_eq!(m2.matched_tokens, 4608);
        assert_eq!(m2.gdn_matched_tokens, 4096);

        // A match that doesn't even reach the first snapshot boundary (5
        // blocks = 1280 tokens < 2048) offers no GDN reuse at all.
        let too_short = &hashes[0..5];
        let m3 = tree.match_prefix(too_short);
        assert_eq!(m3.matched_tokens, 1280);
        assert_eq!(m3.gdn_snapshot, None);
        assert_eq!(m3.gdn_matched_tokens, 0);
    }

    #[test]
    fn ref_counted_nodes_are_not_evicted() {
        let tree = RadixTree::new(256);
        let (prefix, hashes) = build_chain(256, 2048, 3);
        tree.insert(&prefix);
        tree.incr_ref_chain(&hashes);

        let evicted = tree.evict_unreferenced(10);
        assert!(evicted.is_empty(), "referenced nodes must not be evicted");

        tree.decr_ref_chain(&hashes);
        let evicted = tree.evict_unreferenced(10);
        assert_eq!(evicted.len(), 3, "released nodes should now be evictable");
    }

    #[test]
    fn pinned_nodes_are_never_evicted_even_at_zero_refs() {
        let tree = RadixTree::new(256);
        let (prefix, hashes) = build_chain(256, 2048, 2);
        tree.insert(&prefix);
        tree.pin_chain(&hashes);

        let evicted = tree.evict_unreferenced(10);
        assert!(evicted.is_empty(), "pinned nodes must survive eviction");
        assert_eq!(tree.len(), 2);
    }

    #[test]
    fn eviction_only_removes_leaves_first() {
        let tree = RadixTree::new(256);
        let (prefix, hashes) = build_chain(256, 2048, 3);
        tree.insert(&prefix);
        // Nothing referenced; root->A->B->C chain, all unreferenced.

        let evicted = tree.evict_unreferenced(1);
        assert_eq!(evicted, vec![BlockId(2)], "leaf (last block) evicted first");
        assert_eq!(tree.len(), 2);
        assert!(tree.contains(hashes[0]));
        assert!(tree.contains(hashes[1]));
        assert!(!tree.contains(hashes[2]));
    }

    #[test]
    fn eviction_is_lru_ordered() {
        let tree = RadixTree::new(256);
        // Two independent single-block chains (both children of root).
        let a = PrefixBlock {
            hash: hash_block(ROOT_HASH, &[1]),
            block: BlockId(0),
            gdn_snapshot: None,
        };
        let b = PrefixBlock {
            hash: hash_block(ROOT_HASH, &[2]),
            block: BlockId(1),
            gdn_snapshot: None,
        };
        tree.insert(std::slice::from_ref(&a));
        tree.insert(std::slice::from_ref(&b));

        // Touch `a` (ref then release) after `b`, so `a` becomes more
        // recently used than `b`.
        tree.incr_ref_chain(&[b.hash]);
        tree.decr_ref_chain(&[b.hash]);
        tree.incr_ref_chain(&[a.hash]);
        tree.decr_ref_chain(&[a.hash]);

        let evicted = tree.evict_unreferenced(1);
        assert_eq!(
            evicted,
            vec![BlockId(1)],
            "b (LRU) must be evicted before a"
        );
    }

    #[test]
    fn reinserting_an_existing_prefix_preserves_ref_count() {
        let tree = RadixTree::new(256);
        let (prefix, hashes) = build_chain(256, 2048, 2);
        tree.insert(&prefix);
        tree.incr_ref_chain(&hashes);

        // A second worker inserts the same prefix (e.g. after its own
        // prefill of the same content).
        tree.insert(&prefix);

        assert_eq!(
            tree.ref_count(hashes[0]),
            1,
            "re-inserting an existing node must not reset its ref count"
        );
    }
}
