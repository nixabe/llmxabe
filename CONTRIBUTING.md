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

Commit style and the pre-commit checklist live in
[AGENTS.md](AGENTS.md)'s Working rules — Conventional Commits scoped to the
crate, one logical change per commit, fmt/clippy/test clean before
committing.

## Things that must never be committed

- `qwen36-rust-engine-plan.md` — local design draft, gitignored. Settled
  content belongs in `docs/`.
- Model weights, GGUF files, captured activation goldens.
- Benchmark output.

## Console output

Nothing outside a test prints directly. Binaries and examples log through
`tracing`; libraries emit events and never install a subscriber. Two tests in
`xabe-log` enforce this by scanning the workspace, so a stray `println!` in a
binary fails the build rather than quietly ignoring `--log-level`.

Choosing a level:

| Level | For | Rule of thumb |
| --- | --- | --- |
| `error!` | the run cannot produce its result | followed by a non-zero exit |
| `warn!` | the result is real but something was surprising | a reader who ignores it may be misled |
| `info!` | **the tool's own output** — tables, results, summaries | must appear with no flags |
| `debug!` | setup: geometry, buffer sizes, compilation, resolved paths | O(components), not O(items) |
| `trace!` | per-item: per-tensor, per-kernel, per-token | fine to be thousands of lines |

`info!` for tool output is the part that surprises people. It is deliberate:
`INFO` is the level that means "appears by default", and a table that used
`println!` instead would escape both filtering and redirection. The default
format prints an `INFO` event as its message and nothing else, so the
rendering is unchanged from the `println!` it replaced — verified by
byte-diffing four tools' output across the migration.

Do not add a `trace!` inside a per-element loop. Per-tensor and per-layer are
fine; the disabled-level check is cheap against work measured in milliseconds
and expensive against work measured in nanoseconds.

Every binary takes the same flag, `--log-level info | debug | trace`
(default `info`). There is deliberately no `warn` or `error` setting:
`tracing`'s filter is an ordering, so warnings and errors are visible at
every level the flag accepts, and offering them as values would only let a
caller hide problems. `INFO`, `DEBUG` and `TRACE` go to stdout and
`WARN`/`ERROR` to stderr, so piping a table somewhere still leaves
diagnostics on the terminal. `RUST_LOG` is honoured when the flag is absent,
including per-target directives (`RUST_LOG=xabe_engine::weights=trace`); an
explicit `--log-level` overrides it and says so.

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
