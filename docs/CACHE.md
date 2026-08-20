# Hybrid cache

Implemented in [`xabe-cache`](../crates/xabe-cache). This document explains
the two design rules the crate exists to enforce, and why each has a
regression test rather than a comment.

## Two geometries, never unified

Qwen3.6 has two kinds of layer, and their cache behaviour has nothing in
common:

| Group | Layers | Natural page | Scaling |
| --- | --- | --- | --- |
| Attention | 10 | 5 MiB (256 tokens × 20,480 B) | **O(n)** — grows with position |
| Gated DeltaNet | 30 | 62.8125 MiB (full recurrent state) | **O(1)** — constant per sequence |

The attention page is 256 tokens of KV across all ten attention layers. The
GDN page is one complete recurrent-state snapshot across all thirty GDN
layers, including the convolution history as well as the 32 × 128 × 128 fp32
recurrent matrix per layer.

They differ by roughly 12.6×, and that difference is the whole point.

### Rule 1: allocate at natural page sizes

vLLM's allocator originally padded every cache group's page size up to the
maximum across groups. On Qwen3.5 — the same architecture family, 24 GDN and 8
attention layers of 32 — this produced roughly **7× KV memory
overestimation**, because every attention page was inflated to match a GDN
block. Both the reported token capacity and the actual allocation were wrong.

`CacheConfig` therefore computes `attention_page_bytes()` and
`gdn_page_bytes()` independently, and `BlockPool` is instantiated per group
with its own page size and free list.

**Capacity is reported per group and never summed.** `CapacityReport` has no
field that combines the two, and `attention_token_capacity()` is documented as
the only token figure the crate produces. Only attention grows with position;
the GDN group is a fixed per-slot cost, so adding them yields a number that
means nothing.

Regression tests:

- `attention_and_gdn_page_sizes_are_never_unified_to_a_shared_max_regression`
  — asserts the two page sizes differ by more than 5×. If a future change
  padded one to the other, the ratio would collapse toward 1.0 and this fails.
- `capacity_report_keeps_attention_and_gdn_capacity_independent_regression`

## Rule 2: retention interval is independent of block size

A GDN prefix snapshot is the full recurrent and convolution state — 62.8125
MiB — regardless of how much context it summarizes. Retaining one at every
attention block boundary is therefore catastrophic at small block sizes.

vLLM PR #45845 measured exactly that. At block size 128, **snapshots occupied
roughly 80% of the KV pool**, leaving no uncached headroom and forcing the
allocator to evict live attention prefixes:

| Metric | Effect |
| --- | --- |
| Prefix hit rate | ~85% → ~75% |
| Throughput | −18% |
| p99 latency | ~3× worse |

Block sizes of 256 and 512 were unaffected — which is precisely why coupling
the two parameters is a trap. It looks fine until someone tunes block size
down for hit-rate granularity and silently destroys throughput.

So the snapshot retention interval `R` is a separate parameter. Defaults:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `DEFAULT_ATTENTION_BLOCK_SIZE` | 256 tokens | Attention page granularity |
| `DEFAULT_GDN_RETENTION_INTERVAL` | 2048 tokens | One snapshot per 8 attention blocks |

`R` is validated at construction to be a multiple of the block size, so
truncation to a snapshot boundary is plain integer division.

### The ratio to instrument

`CacheConfig::snapshot_to_kv_ratio()` reports snapshot bytes against the
attention KV accumulated over one retention interval. At the defaults:

```
62.8125 MiB snapshot / (20,480 B/token × 2048 tokens = 40 MiB KV) = 1.5703125
```

A ratio near or above 1 means snapshot overhead is comparable to the KV growth
it sits alongside. 1.5703125 is worth watching but is far from the pathological
case that produced the numbers above.

**Tune `R` against measured hit rate, not intuition.** Lowering it buys finer
GDN reuse granularity and pays in snapshot memory; this ratio is what
quantifies the trade.

Regression test:
`tiny_retention_interval_would_make_snapshots_dominate_the_pool_regression`.

## The exact-match constraint

This is the subtle one, and it is why `R` is a first-order parameter rather
than a tuning detail.

Linear-attention state at position *p* summarizes `[0, p)` **completely**.
That is excellent for resumption — one 62.8125 MiB blob restores thirty layers
of history. But it means the state cannot be sliced: you cannot reconstruct a
mid-prefix state from a longer one the way you can simply drop KV blocks past a
cut point.

**Reuse is possible only at points where a snapshot was actually retained.**

So a prefix match must be truncated back to the largest retained snapshot
boundary at or below the matched length. `CacheConfig::retention_floor()` does
that, and `RadixTree::match_prefix` applies it.

A 5,000-token prefix match with `R = 2048` yields a usable GDN resumption point
at token 4,096 — the remaining 904 tokens must be recomputed. That granularity
applies across **75% of the model's layers**, which is the entire reason `R`
matters more than it looks like it should.

Regression test:
`gdn_reuse_truncates_to_retained_snapshot_boundary_not_full_match_length_regression`.

## The prefix tree

A radix tree over token-block hashes, held in host RAM behind an `RwLock` and
shared by all three workers. Structure mirrors vLLM's block-hash-chain design:
each node is keyed by a hash chaining its parent's hash with its own token
block, so identical prefixes converge on identical node chains regardless of
which worker inserted them.

| Operation | Purpose |
| --- | --- |
| `insert` | Publish a computed prefix |
| `match_prefix` | Longest-prefix lookup, truncated per rule 2 |
| `incr_ref_chain` / `decr_ref_chain` | Reference counting along a chain |
| `pin_chain` / `unpin_chain` | Eviction exemption |
| `evict_unreferenced` | LRU reclamation of unreferenced leaves |

**Pinning** exists for the platform system prompt: prefilled once per worker at
boot into an eviction-exempt slot. Three prefills at startup, zero at steady
state.

### Why one process wins

The alternative was `--slot-save-path` on tmpfs as an L2 tier behind each
`llama-server` process's private cache. That design carried six risks — blob
corruption from parameter drift, weak-hash privacy leakage on a multi-user
platform, tmpfs competing with the page cache, torn reads needing atomic
rename, no reaper, and restore latency blocking sibling slots.

All six are artifacts of the replicas being separate OS processes. See
[ARCHITECTURE.md](ARCHITECTURE.md#why-one-process) for the mapping.

## Migration on cold hit

When the best-matching worker is saturated, the prefix is looked up in the host
tree, copied to a less-loaded worker, and only the tail is prefilled.

Staging goes through a **pre-allocated pinned arena** — pageable
host-to-device transfers roughly halve effective bandwidth. Each slot keeps
the two natural geometries separate: 40 MiB of attention KV for one 2,048-token
interval and 62.8125 MiB of GDN state, or 102.8125 MiB in total.

Each worker owns 24 slots (2.41 GiB); the three-worker process reserves 7.23
GiB of pinned host memory at startup. A snapshot holds its slot for as long as
it or any child snapshot remains reachable. If all slots are retained, the
request that reaches the next boundary stops publishing further snapshots but
continues inference. It does not later publish a delta spanning the missed
boundary. One chain can therefore retain at most 49,152 tokens at the default
interval when it owns every slot; this is bounded storage, not full 128K
retention coverage.

Snapshot copies currently run on and synchronize the serving stream at each
retention boundary. Moving them to a dedicated copy stream remains future
work; the arena removes hot-path allocation, not the copy stall.

## Known limitations

Stated rather than discovered later:

- `evict_unreferenced` is an O(n) linear scan per evicted node, finding the
  minimum-`last_used` leaf by scanning all nodes. Adequate for a maintenance
  operation, not benchmarked at scale. A production version wants an LRU heap
  or intrusive list, as vLLM uses in `FreeKVCacheBlockQueue`.
- `xabe-cache` itself does not touch device memory. `xabe-engine` accounts the
  per-worker device pools during admission and owns the pinned migration arena.
  The cache crate's tests remain host-side logic.
- **Only the prompt is shared.** A request is admitted with the block hashes
  of its prompt, and nothing extends that chain as the sequence generates. So
  a snapshot taken at a retention boundary the prompt does not reach has no
  name to be filed under, and `Engine::install_snapshot` declines to share it
  — the sequence keeps using it locally, but no other request can find it.

  The cost is cross-turn reuse: the second turn of a conversation matches only
  as far as the first turn's *prompt*, not through the reply the model
  generated. Closing this means hashing generated tokens into the chain as
  they are produced, which must agree exactly with the runtime's own idea of
  the sequence position — a chain that disagrees files a snapshot under a
  prefix it does not describe, and the next request to match that hash resumes
  from the wrong state. That is worth building deliberately, with a test that
  pins the agreement, rather than as a side effect.

  Until then, note that this is *not* free: declining is the safe branch, and
  it silently gives up the reuse rather than misreporting it.
