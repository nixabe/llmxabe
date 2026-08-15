# Contributing

## Setup

```sh
rustup toolchain install nightly     # rust-toolchain.toml pins it
cargo build --workspace
cargo test --workspace
```

The host-side crates need no GPU and no CUDA toolkit. Everything under
`xabe-gguf`, `xabe-model`, `xabe-cache`, `xabe-sched`, and `xabe-kernels`
builds and tests on any machine.

For device work you additionally need CUDA 12.x on `PATH` and a Turing-or-later
GPU. Check what the host offers:

```sh
cargo run -p xabe-cuda --bin probe
```

Tests requiring a device or the model file skip when they are absent, and they
say so. A skipped test is not a passing test — do not read a green run on a
GPU-less machine as validation of kernel work.

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for reference checkouts, model
paths, and the llama.cpp baseline.

## Where to start

The crate map is in [AGENTS.md](AGENTS.md#crate-map), and the milestone table
is in [README.md](README.md#milestones). Work is roughly ordered by that table.

The highest-value work available right now is Gated DeltaNet on the GPU
(milestone 01). It covers 30 of 40 layers, it is the project's critical path,
and there is a CPU reference plus a differential harness already waiting for it
in `xabe-kernels`.

## Design rules that are not up for negotiation

Five rules encode failure modes documented in upstream production systems. Each
has a regression test. If your change makes one of those tests fail, the change
is wrong — do not adjust the test.

They are stated with their evidence in
[AGENTS.md](AGENTS.md#non-negotiable-design-rules). In brief:

1. Cache groups keep their natural page sizes; never unified.
2. GDN snapshot retention interval is independent of attention block size.
3. Token budget > `block_size + max_concurrent_decodes`, asserted at startup.
4. Admission checks full sequence length, not the first chunk.
5. MoE indirection is built on-device into fixed-size buffers.

## Correctness standard

Numerics drift is this project's highest-likelihood risk: the engine stays
fluent while getting quietly worse, and no throughput benchmark catches it.

Every kernel needs:

- a CPU reference implementation in `xabe-kernels`,
- a differential test against that reference with explicit per-tensor max-abs
  and cosine thresholds,
- and, for GPU kernels, the same test running against the device.

Reading generated text and judging it plausible is not a test. See
[docs/TESTING.md](docs/TESTING.md) for thresholds and harness usage.

## Porting from upstream

Two reference checkouts back this work, and they are used for different things:

- **llama.cpp** for anything that must run on sm_75. It is the only backend of
  the two actually validated on Turing.
- **vLLM** for everything above the kernel — dispatch strategy, cache group
  structure, scheduling policy, admission control.

Port *algorithms*, not kernels. vLLM's grouped-GEMM paths assume fp8 or bf16 and
Ampere-or-later features; reimplement the indexing strategy against llama.cpp's
Turing-proven primitives.

When you port something, cite what you ported from in the commit message — file
and function, not just project name. Upstream paths drift, and a future reader
needs to find the thing you were looking at.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/), scoped to the
crate:

```
feat(xabe-cache): add two-group pager with per-group page geometry
fix(xabe-sched): reject token budget <= block size at construction
docs: document snapshot retention interval rationale
test(xabe-kernels): add cosine threshold for chunked delta rule
perf(xabe-cuda): hoist shared expert out of the routed path
```

One logical change per commit. A commit should build and pass tests on its own.

Before committing:

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## Things that must never be committed

- `qwen36-rust-engine-plan.md` — local design draft, gitignored. Settled
  content belongs in `docs/`.
- Model weights, GGUF files, captured activation goldens.
- Benchmark output.

## Reporting results

The project's justification rests on a measurement nobody has taken yet
(milestone 06). Overstated progress reports destroy the only signal that says
whether to continue.

When you report on work:

- Compiling is not working.
- Plausible output is not correct output.
- If you did not run it on the GPU, say so.
- If a test skipped, say it skipped.

State what you measured, on what input, and what you did not check. "I
implemented the kernel; the differential test skipped because this machine has
no CUDA device" is a good report. "Implemented and working" is not.

## Performance claims

Anything of the form "X is faster" needs the measurement next to it: hardware,
context length, batch size, and what it was compared against. The roofline
numbers in [docs/MODEL.md](docs/MODEL.md) are ceilings derived from bandwidth,
not observed rates, and are labelled as such. Keep that distinction.
