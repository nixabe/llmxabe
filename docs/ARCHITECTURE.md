# Architecture

## The shape

```
+-- single process, one address space -------------------------+
|   HTTP / admission queue                                     |
|            |                                                 |
|   cache-aware router  -- score(w) = a*prefix_match(w)        |
|            |                       - b*queued_tokens(w)      |
|            |                       - c*kv_utilization(w)     |
|      +-----+-----+                                           |
|      v     v     v                                           |
|   worker0 worker1 worker2   each: own CUDA context, stream,  |
|    GPU 0   GPU 1   GPU 2    two-group paged pool, captured   |
|      |     |     |          graph, chunked-prefill scheduler |
|      +-----+-----+                                           |
|            v                                                 |
|   shared prefix cache -- radix tree in host RAM              |
|   (pinned arena, cross-worker, no serialization)             |
+--------------------------------------------------------------+
```

Three workers, one per GPU, each holding a complete copy of the model. Workers
never touch each other's device memory: no P2P, no NCCL, nothing crosses PCIe
on the decode path.

What makes them one engine lives entirely above the device — one admission
queue, one router, one radix tree.

## Why one process

The obvious alternative is three `llama-server` processes. That is the
baseline, it works, and it is not obviously worse. The difference is the prefix
cache.

Sharing a prefix cache across OS processes was the design that prompted this
project: `--slot-save-path` on tmpfs as an L2 tier behind each process's
private `-cram`. Its risk list ran to six items — silent blob corruption from
parameter drift, weak-hash privacy leakage on a multi-user platform, tmpfs
competing with the page cache, torn reads needing atomic rename, no reaper, and
restore latency blocking sibling slots.

**All six are artifacts of the replicas being separate OS processes.** In one
address space:

| Cross-process problem | In-process resolution |
| --- | --- |
| Blob corruption from parameter drift | No serialization format, so no version to drift |
| Weak-hash collision / privacy leak | No filenames, so no collision surface |
| tmpfs competing with page cache | No tmpfs |
| Torn reads needing atomic rename | Tree mutation behind an `RwLock` |
| No reaper | The arena allocator already owns eviction |
| Restore latency blocking siblings | A device copy on a dedicated low-priority stream |

This is the project's one *architectural* claim, and the only one made without
qualification. It is a structural property, not a measurement — though it is
now supported by one: resubmitting a 25,136-token prompt drops
time-to-first-token from 9.7 s to 0.3 s, a 32× improvement that is currently
confined to a single process.

Everything else is a performance bet, and benchmarking has since resolved two
of them against the project: fused MoE dispatch and graph capture are **already
implemented in llama.cpp**, so they are the bar rather than the advantage. What
remains is the measured 44.5% efficiency of the MoE weight path, and
compile-time specialization on fixed shapes, which is still unmeasured. See
[BENCHMARKS.md](BENCHMARKS.md).

## Crate boundaries

Dependencies point one way. If `xabe-cache` needs to know something about
scheduling, the boundary is wrong.

```
xabe-gguf ──> xabe-model ──┬──> xabe-cache ──┐
                           ├──> xabe-sched ──┼──> xabe-engine ──> xabe-server
                           └──> xabe-kernels ┘
              xabe-cuda ───────────────────────┘
```

| Crate | Owns | Deliberately does not know |
| --- | --- | --- |
| `xabe-gguf` | Container parsing, tensor layout, mmap | Anything about Qwen |
| `xabe-model` | Architecture, VRAM and bandwidth budgets | Anything about devices |
| `xabe-cache` | Two-group pager, radix prefix tree | Scheduling policy |
| `xabe-sched` | Chunked prefill, admission, preemption | Device memory |
| `xabe-kernels` | CPU reference kernels, differential harness | CUDA |
| `xabe-cuda` | Driver API, NVRTC, graphs, capability gate | Layers and experts |
| `xabe-engine` | Worker lifecycle, router, orchestration | HTTP |
| `xabe-server` | HTTP surface, admission queue | Everything below the engine |
| `xabe-log` | `tracing` setup, `--log-level` parsing, output format | Every other crate |

`xabe-log` sits outside the dependency chain above: it is a leaf that anything
may depend on and that depends on nothing in the workspace.

The rule that matters is about *who installs the subscriber*. Binaries call
`xabe_log::init_from_args()` exactly once at startup; **library code emits
`tracing` events and never installs a subscriber**, so an embedding process
keeps control of where output goes and a test binary is free to install its
own. No `xabe_log` symbol appears outside a `src/bin/`, `examples/` or
`main.rs` file, and `crates_do_not_install_a_subscriber` in `xabe-log`'s test
module asserts that by scanning the workspace.

Cargo has no per-target dependency section, so the crates whose *binaries*
need `xabe-log` list it as an ordinary dependency even though their libraries
do not use it. That is a manifest limitation, not a layering claim.

The split that earns its keep is `xabe-kernels` knowing nothing about CUDA.
Reference implementations are the oracle for every GPU kernel, and they must be
runnable and debuggable on a machine with no GPU. See [TESTING.md](TESTING.md).

## The worker

Each worker owns one GPU and, on it:

- a CUDA context and a small set of streams,
- the full model weights in Q6_K (~29.6 GiB),
- a two-group paged cache pool ([CACHE.md](CACHE.md)),
- a captured decode graph,
- a chunked-prefill scheduler ([SCHEDULER.md](SCHEDULER.md)).

Workers are symmetric. Any request can go to any worker; the router's job is
only to pick the one where it will be cheapest.

### Why decode is one graph replay

A decoded token naively costs 1,080 small GEMVs in the MoE blocks alone — nine
active experts × three matrices × forty layers — plus attention, GDN, norms,
and the LM head. That is hundreds of kernel launches per token, on a workload
where each kernel is memory-bound and short.

Capturing the whole decode step and replaying it once per token removes launch
overhead from the critical path, and it is why the MoE indirection tables must
be built **on-device into fixed-size buffers**: anything sized by a host value
makes the graph topology vary between replays, and the capture becomes
worthless.

> This was originally described as the largest single expected win.
> Benchmarking showed **llama.cpp already captures graphs on every decode
> step** — server logs report `graphs reused = 1195` over a 1,200-token
> generation — and already fuses MoE dispatch. Graph capture is therefore the
> bar to clear, not an advantage to claim. The measured headroom is elsewhere:
> the MoE weight path runs at 44.5% of peak bandwidth against flash attention's
> ~80%. See [BENCHMARKS.md](BENCHMARKS.md).

The technique is to gate work on a `valid_tokens` scalar held in device memory,
so the topology stays static while routing content varies freely. See
[KERNELS.md](KERNELS.md).

Graph capture also imposes a constraint the safe Rust wrapper does not survive
— see [TOOLCHAIN.md](TOOLCHAIN.md#the-event-tracking-constraint).

## The router

Requests are scored per worker:

```
score(w) = a * prefix_match(w) - b * queued_tokens(w) - c * kv_utilization(w)
```

Cache affinity pulls a request toward the worker that already holds its prefix;
the load terms push back when that worker is saturated. The coefficients are
tuning parameters, not constants — they trade time-to-first-token against
inter-token latency, and the right balance depends on traffic shape.

**Migration on cold hit.** When the best-matching worker is saturated, the
prefix is looked up in the host tree, copied to a less-loaded worker, and only
the tail is prefilled. Staging goes through a pre-allocated pinned arena —
pageable host-to-device transfers roughly halve effective bandwidth — on a
dedicated low-priority stream per device so it never preempts decode.

**Pinned system prefix.** The platform's system prompt is prefilled once per
worker at boot into an eviction-exempt slot. Three prefills at startup, zero at
steady state.

## Scope

**Text, and images behind `--mmproj`.** The model's vision tower (the
`mmproj-F16.gguf` shipped beside the weights) is implemented: a SigLIP
encoder plus merger runs per worker on the device, image embeddings are
injected over the `<|image_pad|>` positions after the token embedding, and
attention layers switch to interleaved M-RoPE on image-bearing chunks. The
prefix cache stays correct under images by substituting per-slot content-hash
lanes for pad tokens when naming blocks. All of it is opt-in: without
`--mmproj` nothing vision-related is allocated, decode graphs capture the
scalar-rope path unchanged, and image requests are refused with a 400.

Video stays out of scope, and so does fetching remote image URLs.

## Reading order

0. [BENCHMARKS.md](BENCHMARKS.md) — what the baseline actually achieves
1. [MODEL.md](MODEL.md) — what the architecture costs, and why decode is
   KV-bound at long context
2. [CACHE.md](CACHE.md) — the two-group pager and prefix sharing
3. [SCHEDULER.md](SCHEDULER.md) — chunked prefill and admission
4. [KERNELS.md](KERNELS.md) — what has to be written, and in what order
5. [TESTING.md](TESTING.md) — how any of it is known to be correct
