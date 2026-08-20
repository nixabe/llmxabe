# AGENTS.md

Operating instructions for AI agents working in this repository. Humans should
read [CONTRIBUTING.md](CONTRIBUTING.md) instead; it covers the same ground with
less prescription.

## What this project is

`llmxabe` is a single-process, three-worker CUDA inference engine for
`Qwen3.6-35B-A3B` on 3× Quadro RTX 8000 (sm_75, 48 GB, 672 GB/s each).

Its one *structural* advantage over running three `llama-server` processes is
that the prefix cache lives in one address space. Every other claim — fused MoE
dispatch, CUDA graph capture, compile-time shape specialization — is a
performance bet whose size is unknown until measured. Do not write documentation,
commit messages, or comments that assert those wins as fact.

## Current standing — read this before trusting any number you find

The head-to-head against llama.cpp moves, and every stale copy of it in these
documents has misled someone. The current numbers live in **"Current standing"
at the top of [docs/BENCHMARKS.md](docs/BENCHMARKS.md)**, and that section is
the only place they belong — do not restate a ratio anywhere else without a
link to it.

The standing is that every cell of the one-card N=3 target is clear of
llama.cpp at its own best settings, by +0.8% to +18.6% on prefill and +4.4% at
2K decode, with 32K decode level-to-slightly-ahead. The two thinnest margins
are inside this card's own thermal drift and stand only on same-hour
alternating pairs. Treat them as parity you have to keep, not as headroom.

Two standing corrections to older text you may encounter below or elsewhere:

- **CUDA graph capture is not "the largest expected win".** It was measured:
  ~0.4% at prefill shape, and bounded above by 5.8% on a decode step. The
  design rules that keep the engine graph-capturable stay, because they cost
  nothing — but do not rank work by that old claim.
- **Compare against llama.cpp's best settings, not its defaults.** Its prefill
  wants `-ub 4096` at depth (its default `-ub 512` is ~40% slower at 8K+), and
  its decode wants `-fa 1` and backend sampling. A ratio measured against its
  defaults is not a result; this mistake was made three times before it was
  named in BENCHMARKS.md ("The baseline was llama.cpp's default, not its
  best").

## Non-negotiable design rules

These encode failure modes that were paid for in someone else's production
system. Each one has a regression test. Do not relax them to make a test pass.

1. **Never unify page geometry across cache groups.** Attention blocks are
   ~32 KiB; a Gated DeltaNet state block is ~1.1 MiB — about 35×. Padding every
   group to the max produced ~7× KV capacity misreporting upstream. Allocate at
   natural per-group page sizes and report capacity per group.
   Enforced by `xabe-cache`.

2. **GDN snapshot retention interval `R` is independent of attention block
   size.** Snapshotting recurrent state at every block boundary let snapshots
   consume ~80% of the pool, collapsing prefix hit rate ~85% → ~75% and
   tripling p99. `R` defaults to 2048 tokens against a 256-token attention
   block. Enforced by `xabe-cache`.

3. **Per-step token budget must exceed `block_size + max_concurrent_decodes`.**
   At budget == block size, a single decoding request starves prefill admission
   and execution serializes to batch 1 (~7× throughput loss). This is asserted
   at startup, not documented and hoped for. Enforced by `xabe-sched`.

4. **Admission checks the full sequence length, not the first chunk.** Chunked
   prefill splits compute, not memory. A 32K request reserves 32K of KV for its
   whole life. Enforced by `xabe-sched`.

5. **MoE indirection tables are built on-device into fixed-size buffers.**
   Anything sized by a host-side value breaks CUDA graph capture. Capture has
   since been *measured* at near-zero value for today's shapes (see
   [docs/BENCHMARKS.md](docs/BENCHMARKS.md)) — the rule stays anyway, because
   device-side gating on a `valid_tokens` scalar costs nothing now and reverting
   it would foreclose capture later. Fix grids by packing to a device-side live
   count, not by host-sized launches.

6. **No allocation on the hot path.** Pinned arenas are pre-allocated at
   startup. Three workers contending on the host allocator produce latency
   spikes that read like GPU stalls and cost days to diagnose.

## Correctness before speed

Numerics drift is the highest-likelihood risk in this project: the engine stays
fluent while getting quietly worse, and no benchmark catches it.

Every kernel ships with a CPU reference implementation in `xabe-kernels` and a
differential test against it. Thresholds are per-tensor max-abs and cosine
similarity, not eyeball comparison of generated text. See
[docs/TESTING.md](docs/TESTING.md).

A kernel without a passing differential test is not done, regardless of how fast
it runs.

## Crate map

| Crate | Owns | Depends on |
| --- | --- | --- |
| `xabe-gguf` | GGUF container parsing, tensor layout, mmap | — |
| `xabe-model` | Qwen3.6 config, VRAM and bandwidth budgets | `xabe-gguf` |
| `xabe-cache` | Two-group pager, radix prefix tree, snapshot retention | `xabe-model` |
| `xabe-sched` | Chunked prefill, decode priority, admission control | `xabe-model`, `xabe-cache` |
| `xabe-kernels` | CPU reference kernels, differential harness | `xabe-model` |
| `xabe-cuda` | Driver API, streams, graphs, device probe | — |
| `xabe-engine` | Worker, cache-aware router, orchestration | all of the above |
| `xabe-server` | HTTP surface, admission queue | `xabe-engine` |
| `xabe-log` | `tracing` setup, `--log-level`, output format | — |

Dependencies point one way only. If you need `xabe-cache` to know something
about scheduling, the abstraction is wrong — fix the boundary, do not add the
edge.

## Where the answers live

Do not design from first principles. Two upstream projects have already made
the mistakes in this exact architecture class.

- **llama.cpp** (`/home/nixabe/llama.cpp`) is the source for anything that must
  run on sm_75. It is the only one validated on Turing. Relevant:
  `ggml/src/ggml-cuda/gated_delta_net.cu`, `fattn-mma-f16.cuh` (the path this
  model's head_dim 256 actually takes on Turing — an older note here pointed at
  `fattn-tile.cu`/`fattn-vec.cuh`, which was wrong for this geometry),
  `dequantize.cuh`, `vecdotq.cuh`, `mmvq.cu`, `src/llama-memory-recurrent.cpp`.
- **vLLM** (`/home/nixabe/vllm`) is the source for everything above the kernel:
  dispatch strategy, cache group structure, scheduling policy, admission
  control. Relevant: `vllm/model_executor/layers/fused_moe/fused_moe.py`,
  `vllm/v1/core/kv_cache_utils.py`, `vllm/v1/core/sched/scheduler.py`, and
  `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` for how
  per-sequence recurrent state is indexed by a device-side index tensor.
- **exllamav3** (`/home/nixabe/exllamav3`) is a third reference for
  quantized-kernel and attention tricks; check it before concluding an idea is
  novel.

**Port algorithms, not kernels.** vLLM's grouped-GEMM paths assume fp8 or bf16
and Ampere-or-later features. Reimplement its *indexing strategy* against
llama.cpp's Turing-proven primitives.

Turing has no `cp.async`. Double-buffering is by hand — and it has mattered:
software-pipelined prefetch was worth 12% on prefill attention. "This workload
is bandwidth-bound" is true of decode's weight path and nothing else; prefill
attention is on the tensor cores and traffic/issue-bound, decode attention is
bound by its softmax rather than its loads (a calibration kernel with the
identical grid and loads reaches 90% of streaming roofline), and the decode
MoE GEMVs are bound by the integer pipe unpacking Q6_K. Establish which bound
you are under before optimizing — BENCHMARKS.md's WHY NOT list records several
optimizations that were correct for the wrong bound and measured slower.

## Working rules

- **Rust 2024 edition.** Nightly is pinned in `rust-toolchain.toml`.
- **Commit subjects are sentences, one logical change per commit.** The
  repository's practice moved past the Conventional Commits rule still written
  in CONTRIBUTING.md: read `git log --oneline -20` and match it. The house
  style names what was done and, where it fits, what was learned — "Split the
  key axis at decode, and stop leaving 78% of the card idle". The commit
  message is where the blow-by-blow lives, so it carries the numbers and the
  method; BENCHMARKS.md carries only what outlives the change. Negative results
  get commits too ("…and be wrong about how much that buys").
- **Never commit `qwen36-rust-engine-plan.md`.** It is a local design draft and
  is listed in `.gitignore`. Its content belongs in `docs/` once settled.
- **Never commit model weights**, captured goldens, or benchmark output.
- `cargo fmt --all` and `cargo clippy --workspace --all-targets` must be clean
  before you commit.
- `cargo test --workspace` must pass before you commit.
- **Never `println!` outside a test.** Binaries and examples log through
  `tracing`; libraries emit events and never install a subscriber. Tool
  output — tables, results — is `info!`, because `INFO` is the level that
  means "appears by default". `xabe-log`'s `tests/layering.rs` scans the
  workspace and fails the build otherwise. Levels are documented in
  [CONTRIBUTING.md](CONTRIBUTING.md#console-output).

## How to measure

The project's measurement discipline is what has kept it honest; follow it.

- **Kernel-level first.** `bench_attention` (~6 s per A/B) exists because
  whole-forward A/Bs are so expensive that the honest response to a small
  change was to not measure it. Prefer the narrow bench, then confirm
  end-to-end: `bench_forward` (chunked prefill via `LLMXABE_BENCH_CHUNK`),
  `bench_decode`, `bench_moe`, `bench_mma`, `profile_forward`.
- **Interleaved A/B pairs**, at least three, spreads reported. A single pair
  proves nothing on this host; run-to-run drift has eaten 10%+ "wins" before.
- **CUDA events, not `Instant`**, for anything inside a pass. Host clocks
  measure enqueue latency.
- **`ncu` does not work on this host** (`ERR_NVGPUCTRPERM`). The working
  substitutes are `nsys`, `cuobjdump -sass` on the cubin, roofline arithmetic
  from the GGUF tensor directory, and ablation. Do not burn time re-trying
  `ncu`.
- **Pin a GPU and check it is idle.** Three cards; sibling agents may be
  benchmarking. `nvidia-smi` first, then `CUDA_VISIBLE_DEVICES=<n>` on every
  run.
- **Record rejects with their numbers** in [docs/BENCHMARKS.md](docs/BENCHMARKS.md)'s
  **WHY NOT** section, in its voice, one row with the mechanism. A rejected
  attempt that is not written down will be re-attempted by the next agent;
  those tables have already prevented repeat work several times.
- **BENCHMARKS.md is not a journal.** It carries the current standing, the
  method, and the two durable lists — why the engine is shaped as it is, and
  what was measured and rejected. When a new measurement supersedes an old
  number, **replace it**; when a change lands, fold its *mechanism* into WHY
  rather than appending a dated section. The blow-by-blow belongs in commit
  messages, which is what `git log` is for.

## Reporting results honestly

Overstated progress reports are actively harmful — they erode the only signal
that says whether the remaining gaps are closing. Two of this project's three
original performance bets turned out to be things the baseline already did;
that was only discovered because measurements were reported against the
baseline's best configuration rather than around it.

- Do not describe a component as working because it compiles.
- Do not describe a kernel as correct because it produced plausible text.
- If you did not run it on the GPU, say that you did not run it on the GPU.
- If a test is skipped because no device is present, say it was skipped. A
  skipped test is not a passing test.

State what you measured, on what input, and what you did not check.
