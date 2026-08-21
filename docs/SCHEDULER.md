# Scheduler

Implemented in [`xabe-sched`](../crates/xabe-sched). The design is ported from
vLLM's `Scheduler.schedule()` (`vllm/v1/core/sched/scheduler.py`) — the
algorithm, not the code. It is well-tested upstream and, more usefully, its
failure modes are documented.

## Chunked prefill with decode priority

Each step, in order:

1. **Batch every decode-ready running request.** Decode is memory-bound and
   latency-sensitive; it goes first, unconditionally.
2. **Continue any request already mid-chunk**, spending remaining budget.
3. **Admit from the waiting queue**, gated by the free-block watermark,
   chunking the last request that does not fit.

This deliberately mixes compute-bound prefill with memory-bound decode in a
single batch — the right pairing for an MoE model whose decode is
bandwidth-starved and whose prefill is not.

`step()` returns a `BatchDescription` listing which requests decode and which
prefill for how many tokens. It is deterministic and touches no device, so
scheduling policy is testable without a GPU.

### The token budget

The central knob. Smaller improves inter-token latency, because fewer prefills
interrupt decodes; larger improves time-to-first-token. The llama.cpp
baseline's `-ub 4096` is a reasonable starting equivalent.

## Rule 3: the budget-versus-block trap

**Never set the per-step token budget equal to the block size.**

At budget = N, once any request is decoding it consumes at least one token of
the per-step budget, so no new request can begin prefill and execution
serializes to one request at a time. LMCache's Qwen3.6 recipe benchmarked this
on Qwen3.6-27B: roughly **7× slower, with the GPU batch stuck at 1**. Setting
the budget to 2N−1 restored full batching.

The failure is silent. Throughput is bad, nothing errors, and the GPU looks
busy.

So `SchedulerConfig::new` **rejects** a configuration where

```
token_budget <= block_size + max_concurrent_decodes
```

as a typed error at construction. Not a `debug_assert`, not a log line — a hard
constructor failure, because the whole point is that this must be impossible to
misconfigure into production.

Regression tests:

- `budget_equal_to_bare_block_size_would_serialize_to_batch_one_regression`
- `budget_at_block_size_plus_concurrency_boundary_would_serialize_to_batch_one_regression`

## Rule 4: admission checks the full sequence

**Chunked prefill splits compute, not memory.**

A 32K request reserves 32K of KV for its entire life regardless of how the
prefill is chunked. Admitting on the basis of the first chunk fitting
over-admits, and the pool thrashes: requests are admitted, evicted, preempted,
and readmitted.

`admit()` therefore checks `prompt_tokens + max_output_tokens` against
capacity, not the chunk size.

Regression test:
`admission_rejects_full_sequence_exceeding_capacity_even_when_first_chunk_fits_regression`
— a request whose first chunk fits comfortably but whose full lifetime does not
is rejected.

## Free-block watermark

A fraction of blocks (`DEFAULT_WATERMARK_FRACTION`, 1%) is held in reserve when
admitting from the waiting queue.

Without it, the scheduler admits right up to the last free block, then
immediately needs to evict, which preempts a request that is likely to be
readmitted and preempted again. The watermark buys hysteresis.

## Bounded waiting queue

Admission is also bounded independently of device capacity. By default each
worker holds at most four times `max_concurrent_decodes` requests in its
waiting queue; with the shipped three-wide worker that is 12 queued requests,
or 36 across the three-card engine. Running requests do not count against this
limit because their full-lifetime KV reservation is already accounted for.

Once the limit is reached, `can_admit()` reports the worker as saturated and
`admit()` returns `WaitingQueueFull`. The HTTP layer exposes the engine's
result as 503, allowing an upstream load balancer to retry instead of letting
prompts and response channels grow without bound in host memory. Internal
preemption may still put a running request back at the front: dropping already
admitted work to enforce an external-admission limit would be a different and
more destructive policy.

## Preemption by recompute

When capacity runs short, requests are preempted and requeued at the **front**
of the waiting queue, and their work is recomputed rather than swapped to host
memory.

Recompute has lower overhead than swapping, and the shared prefix cache
([CACHE.md](CACHE.md)) already provides the useful half of swap: a preempted
request's prefix is in the host radix tree, so readmission does not start from
zero.

## Speculative decoding

A draft source produces 2–3 draft tokens per step. Every draft token consumes
token budget and KV blocks that may be discarded on rejection.

Which source, and whether any, is `--spec-type` (see [CLI.md](CLI.md#speculative-decoding)).
It ships as `none`. The only source wired into serving is n-gram suffix
lookup; the trained MTP head has a driver but was measured and not adopted
(milestone 09).

`tokens_per_decode_step()` charges `1 + draft_tokens_per_step` against the
budget, so the budget is **sized for the drafted case, not the accepted case**.
Sizing for the accepted case makes admission oscillate: the scheduler admits
based on optimistic capacity, drafts overshoot it, and requests preempt.

`DEFAULT_DRAFT_TOKENS_PER_STEP` is 3, matching the vLLM and SGLang recipes for
this model's trained MTP head. It is the default draft count for whichever
source `--spec-type` selects, not a claim that MTP is the one running.

## Async scheduling

Overlapping CPU-side scheduling and input preparation with GPU compute keeps
the device from idling between steps.

On a 12-vCPU host running three workers this is **not free** — the baseline
already partitions cores four ways per replica. Measure the overlap against the
contention before committing to it. Not implemented.

## Known limitations

Stated explicitly rather than discovered later:

- **KV is reserved for the full worst-case lifetime at admission**
  (`prompt + max_output_tokens`), not grown incrementally as generation
  proceeds. This satisfies rule 4 literally and is the simplest correct model,
  but it under-utilizes the pool for requests that stop early. A production
  engine grows allocation lazily while still gating admission on the full
  length.
- **Draft-token rejection is modeled at the budget level only.** There is no
  simulation of discarding individual speculative KV blocks when a draft is
  rejected.
- **The scheduler does not hold a live `BlockPool`.** It tracks free and total
  attention block counts as integers that a caller keeps in sync with the real
  pool. This keeps `step()` device-free and deterministic; wiring the two
  together belongs to `xabe-engine`.
- No device memory is touched. The scheduler tests are host-side logic.
