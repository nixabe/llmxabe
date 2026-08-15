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
   Anything sized by a host-side value breaks CUDA graph capture, which is the
   project's single largest expected win. Gate on a `valid_tokens` scalar held
   in device memory instead.

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
  run on sm_75. It is the only one of the two validated on Turing. Relevant:
  `ggml/src/ggml-cuda/gated_delta_net.cu`, `fattn-tile.cu`, `fattn-vec.cuh`,
  `dequantize.cuh`, `vecdotq.cuh`, `mmvq.cu`, `src/llama-memory-recurrent.cpp`.
- **vLLM** (`/home/nixabe/vllm`) is the source for everything above the kernel:
  dispatch strategy, cache group structure, scheduling policy, admission
  control. Relevant: `vllm/model_executor/layers/fused_moe/fused_moe.py`,
  `vllm/v1/core/kv_cache_utils.py`, `vllm/v1/core/sched/scheduler.py`.

**Port algorithms, not kernels.** vLLM's grouped-GEMM paths assume fp8 or bf16
and Ampere-or-later features. Reimplement its *indexing strategy* against
llama.cpp's Turing-proven primitives.

Turing has no `cp.async`. Double-buffering is by hand. Accept it; this workload
is bandwidth-bound, not latency-bound.

## Working rules

- **Rust 2024 edition.** Nightly is pinned in `rust-toolchain.toml`.
- **Conventional Commits**, one logical phase per commit. Scope is the crate
  name where it applies: `feat(xabe-cache): add two-group pager`.
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

## Reporting results honestly

This project's whole justification is a measurement that has not been taken yet
(milestone 06, CUDA graph capture over on-device MoE indirection). That makes
overstated progress reports actively harmful — they erode the only signal that
tells us whether to continue.

- Do not describe a component as working because it compiles.
- Do not describe a kernel as correct because it produced plausible text.
- If you did not run it on the GPU, say that you did not run it on the GPU.
- If a test is skipped because no device is present, say it was skipped. A
  skipped test is not a passing test.

State what you measured, on what input, and what you did not check.
