# Toolchain

**Decision: `cudarc` 0.19 + NVRTC, with CUDA C++ kernel sources compiled at
load time.**

This resolves milestone 00. The evidence below was produced by
`cargo run -p xabe-cuda --bin probe` on the target host, and the same checks
run as a test (`spike::tests::milestone_00_gating_spike`), skipping loudly
when no device is present.

## The choice

The design plan offered two options.

| Option | For | Against |
| --- | --- | --- |
| `cudarc` + CUDA C++ via NVRTC | Stable on sm_75; JIT specialization at load; inline PTX always available | Two languages; kernel code outside Rust's type system |
| NVlabs `cuda-oxide` | Single-source Rust; compile-time kernel policies; monomorphization; PTX interop with existing launchers | Alpha; separate pinned nightly; sm_75 needs an explicit target and workload validation |
| NVlabs `cutile-rs` | Rust tile DSL; compiler-managed layout; stable Rust; graph replay | Explicitly excludes sm_75; cannot target the deployment cards |

The decision rule was explicit: any "no" on the inline-PTX question collapses
the choice to cudarc, because without an escape hatch you cannot reach
`mma.sync.m16n8k8.f16` or `ldmatrix`, and you cannot route around a codegen
bug.

**cuda-oxide is obtainable from NVlabs.** The original availability rejection
is obsolete. Do not confuse it with Protryon's older `cuda-oxide` driver
wrapper. Availability alone does not validate a replacement kernel.

## Rust compiler evaluation

Source revisions checked on 2026-09-10:

- [cuda-oxide `26754ae5`](https://github.com/NVlabs/cuda-oxide/tree/26754ae52c26c097dc1c465a1e42c4c5d05a3d40)
  declares version 0.2.1 and pins `nightly-2026-08-28`. Its installation guide
  lists sm_80+, but `cargo-oxide/src/commands/codegen_env.rs` explicitly handles
  Turing and accepts `--arch sm_75`. Its `ptx_asm!` macro supplies an inline-PTX
  escape hatch. The residual-add probe ran on sm_75; this does not establish
  support or performance for all of our tensor-core kernels.
- [cutile-rs `2eed75e8`](https://github.com/NVlabs/cutile-rs/tree/2eed75e8f552be31216ddf2019288f03dfec4939)
  declares version 0.3.1. Its README explicitly excludes architectures below
  sm_80, including sm_75, with no plan to support them. CUDA 13.2 adds Ampere
  support; 13.3 adds Hopper. Neither makes Turing a supported target.

The candidate in [`experiments/cuda-oxide`](../experiments/cuda-oxide) ports
`layer_ops.rs::tensor_add` to Rust, preserving its raw-pointer ABI, grid-stride
loop and input/output aliasing. It uses a separate Cargo workspace so compiler
experiments do not change the engine's nightly or dependency graph. The
[`bench_rust_add`](../crates/xabe-engine/src/bin/bench_rust_add.rs) gate loads
its PTX through the engine's existing cudarc runtime, compares both compilers
against `xabe_kernels::norm::residual_add`, and measures alternating pairs
with CUDA events. The opt-in `xabe-cuda/rust-kernels` feature embeds the
generated PTX, so using it requires neither the experimental compiler nor its
host runtime at engine build or run time. Source changes require regeneration.
The server and engine expose a forwarding `rust-kernels` feature: build with
`cargo build --release -p xabe-server --features rust-kernels` to enable it.
It is disabled by default and has no runtime switch; see the
[build instructions](DEVELOPMENT.md#optional-rust-cuda-kernels).

On GPU 1, the residual matched its CPU oracle bit for bit, including both
in-place aliases, signed zero, subnormals and ragged tails. All 12 existing
layer-op differentials passed with the feature enabled. The 19-token Qwen3.6
forward golden also passed all 40 block gates and selected token 25358.
These checks do not establish long-context performance or validate another
Rust kernel. [The paired measurements](BENCHMARKS.md#rust-kernel-authoring-can-keep-the-existing-launcher)
show a faster standalone kernel and model-level parity at the measured N=3/2K
configurations. The default remains NVRTC.

The current host runtime dependency (`cuda-core`/`cuda-bindings` 0.3.1) requires
CUDA 13.0+ headers. On this host the shell selects CUDA 12.4 even though CUDA
13.0 is also installed. Select the latter explicitly for the experiment;
ordinary engine builds continue to use NVRTC as before.

## Gating results

Measured on 3× Quadro RTX 8000, driver 595.84, CUDA 12.4, Rust nightly.

| # | Question | Result |
| --- | --- | --- |
| Q1 | Does the toolchain emit sm_75 PTX? | **Pass** — NVRTC accepts `compute_75` and the emitted PTX carries `.target sm_75` |
| Q2 | Is inline PTX available inside kernels? | **Pass** — `asm volatile("fma.rn.f32 …")` compiled, launched, and matched host arithmetic exactly over 1024 elements |
| Q3 | Do warp shuffle/vote intrinsics lower correctly? | **Pass** — `__shfl_xor_sync` butterfly reduction and `__ballot_sync`/`__popc` matched host results across 8 warps (max abs error 1.9e-6) |
| Q4 | Are CUDA Graphs reachable? | **Pass** — a captured launch replayed 5× with the accumulator advancing exactly once per replay |

Each check verifies numerical output. None is satisfied by the API returning
`Ok`, and a check that cannot run reports `Skipped` rather than `Pass` — see
[`CheckOutcome`](../crates/xabe-cuda/src/spike.rs).

Q1 was answered more strictly than "it compiled": NVRTC accepting
`--gpu-architecture=compute_75` does not by itself prove the right target came
out, so the `.target` directive in the emitted PTX is inspected directly.

Q4 is the one that mattered most. Graph capture over on-device MoE indirection
is the thesis of this project (milestone 06), and the plan flagged graphs as
"not in the crate list; `cuda-bindings` raw FFI is the fallback". That turned
out to be unnecessary — cudarc 0.19 exposes a safe graph API
(`begin_capture`, `end_capture`, `CudaGraph::launch`) over 90 `cuGraph*`
bindings.

## The event-tracking constraint

Q4 failed on first attempt with:

```
CUDA_ERROR_STREAM_CAPTURE_ISOLATION
  "dependency created on uncaptured work in another stream"
```

This is **not** a hardware or driver limitation, and it is worth understanding
because it constrains milestone 06 permanently.

cudarc tracks a read event and a write event per device allocation. When the
context is in multi-stream mode it injects `cuStreamWaitEvent` before any
launch touching a buffer that another stream last used. For ordinary work that
is exactly right — it is what makes the safe API safe.

During stream capture it is fatal. The injected wait refers to an event
recorded *before* capture began, so the captured graph would depend on
uncaptured work, and CUDA rejects the entire capture.

The resolution is `CudaContext::disable_event_tracking()`, which is `unsafe`
precisely because it hands responsibility for cross-stream ordering back to the
caller.

**Consequence for the engine.** The decode step cannot rely on the wrapper for
ordering. It must:

- own its buffers for the lifetime of the captured graph,
- fully synchronize before capture begins,
- order its own streams explicitly, and
- keep event tracking disabled only across the captured region.

That is consistent with the design anyway — §"No allocation on the hot path"
in [AGENTS.md](../AGENTS.md) already requires pre-allocated arenas, and a
captured graph requires stable device pointers regardless. But it means the
safe wrapper's guarantees do not extend into the hot path, and reviewers should
treat capture-region code as manually ordered.

## What this does not settle

- **Kernel language.** CUDA C++ through NVRTC means kernel code sits outside
  Rust's type system. Shapes are passed as scalars and validated at the call
  boundary in `xabe-cuda`, not by the compiler.
- **Tensor-core paths.** Q2 proves inline PTX works; it does not prove any
  particular `mma.sync` shape is correct or profitable on this workload. The
  MoE decode shapes are extremely skinny (batch 3 against 2048×512), so SplitK
  is expected to matter more than MMA. Unmeasured.
- **cuBLASLt.** Enabled as a cudarc feature for the prefill GEMM path but not
  yet exercised by any code here.

## Reproducing

```sh
cargo run -p xabe-cuda --bin probe    # full device inventory + gate
cargo test -p xabe-cuda               # same checks as tests
```

Exits non-zero if the fleet fails the capability gate or a check that ran did
not pass. On a host with no CUDA driver, the binary reports that and exits
non-zero; the test skips with a printed message.
