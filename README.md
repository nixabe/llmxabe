# llmxabe

A single-process CUDA inference engine for `Qwen3.6-35B-A3B` on 3× Quadro
RTX 8000, written in Rust.

> **Status: foundation.** The host-side engine — GGUF loading, hybrid cache,
> scheduler, CPU reference kernels — is implemented and tested. The CUDA
> kernels are not. This does not yet run a model. See
> [Milestones](#milestones) for exactly what is and is not done.

## Why this exists

The obvious way to serve this model on three GPUs is three `llama-server`
processes, one per card. That works, and it is the baseline this project is
measured against.

It has one structural weakness: each process has a private prefix cache. Three
replicas serving one workload cache the same system prompt three times, and a
request routed to the wrong replica pays full prefill for a prefix that another
replica already holds. Sharing that cache across OS processes means a
serialization format, a filesystem, a hashing scheme, and an eviction
policy — every one of which is a source of silent corruption or a privacy leak
on a multi-user platform.

In one address space it is a radix tree behind a lock.

That is the *architectural* claim, and it is the only one made without
qualification. Three further bets are being placed, and their size is unknown
until measured:

- **Fused MoE dispatch.** Naively, 9 active experts × 3 matrices × 40 layers is
  1,080 small GEMVs per decoded token. Sorting tokens by expert and running one
  grouped GEMM over contiguous runs collapses that to a single launch.
- **CUDA graph capture.** One replay per decoded token instead of hundreds of
  launches. This is the project's largest expected single win, and the reason
  the MoE indirection tables must be built on-device.
- **Compile-time specialization.** Shapes are fixed and known. A
  general-purpose engine cannot assume that; this one can.

This project does not claim it will be faster than llama.cpp. That backend has
years of CUDA tuning behind it.

## Target hardware and model

| | |
| --- | --- |
| GPUs | 3× Quadro RTX 8000 — 48 GB, sm_75 (Turing), 672 GB/s |
| Host | 125 GB RAM, 12 vCPU |
| Model | `unsloth/Qwen3.6-35B-A3B-GGUF`, `UD-Q6_K_XL` (31.8 GB) |
| CUDA | 12.4 |

Turing constrains the design in two ways worth stating early: there is no
`cp.async`, so double-buffering is by hand, and the tensor-core paths available
are the `m16n8k8` fp16 MMA family rather than anything newer.

## Architecture

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

Workers never touch each other's device memory. No P2P, no NCCL, nothing
crosses PCIe on the decode path. What makes them one engine lives above the
device: one queue, one router, one radix tree.

## The model, and what follows from it

Qwen3.6-35B-A3B is 40 layers in a repeating pattern of three Gated DeltaNet
layers followed by one Gated Attention layer, with a 256-expert MoE block on
*every* layer.

Three consequences drive most of the design:

1. **Only 10 of 40 layers hold growing KV.** A 384K-token pool costs 7.5 GiB,
   not the 30 GiB a full-attention 35B would need. The other 30 layers hold
   fixed-size recurrent state — ~60 MiB per sequence, constant regardless of
   position.
2. **Long-context decode becomes KV-bound**, despite only a quarter of layers
   holding any KV at all. KV reads overtake weight reads at about 139K tokens;
   at 128K the split is 2.68 GB/token of KV against 2.86 GB/token of weights.
3. **The LM head alone costs 540 MB/token** — roughly 58% of what all forty MoE
   layers read combined — because the vocabulary is 248,320 and untied.

`cargo run -p xabe-model --example budget` prints the full VRAM and bandwidth
tables for any context length.

## Building

```sh
git clone <repo> && cd llmxabe
cargo build --workspace
cargo test --workspace
```

The host-side crates build and test without a GPU or CUDA toolkit. Tests that
need a device detect its absence and skip, reporting that they skipped.

CUDA work additionally needs the CUDA 12.x toolkit on `PATH`:

```sh
cargo run -p xabe-cuda --bin probe     # device inventory and sm_75 gate
cargo run -p xabe-server               # full engine preflight
```

The preflight validates the whole startup path — model config, cache geometry,
scheduler construction, device gate, VRAM budget against the card's *measured*
memory, and engine assembly. Three design rules are enforced by construction,
so a preflight that builds these types has checked them. It does not serve
requests; there is no HTTP surface yet.

Tests that read the real model file look for it at
`$LLMXABE_MODEL`, falling back to the path in `docs/DEVELOPMENT.md`. They skip
if it is absent rather than failing.

## Milestones

Numbering follows the design plan. "Gate" is the condition for calling it done.

| # | Milestone | Gate | Status |
| --- | --- | --- | --- |
| 00 | Toolchain spike | Inline PTX confirmed, or cudarc chosen | done |
| 01 | GDN kernel prototype | Cosine ≥ 1e-3 vs reference on 4K tokens | CPU reference only |
| 02 | Differential harness | Per-tensor max-abs + cosine thresholds | done |
| 03 | FP16 dense forward | Correct logits, any speed | not started |
| 04 | Q6_K dequant + MoE grouped GEMM | Correct, single GPU | CPU reference only |
| 05 | Flash attention port, sm_75 | Correct at 128K | not started |
| 06 | CUDA graph capture | **The justification gate — measure here** | not started |
| 07 | Two-group pager + scheduler | 3 slots, matches llama.cpp `-np 3` | host side done |
| 08 | Multi-worker + router + shared cache | Hit rate ≥ llama.cpp baseline | host side done |
| 09 | MTP speculative decode | Accept rate vs n-gram baseline | not started |

Milestone 01 deliberately precedes the harness. Gated DeltaNet covers 75% of
layers, has no equivalent in any flash-attention codebase, and its prefill form
needs per-chunk triangular matrix inversion. If it does not come together,
nothing else matters — and that should be discovered in week three, not week
twelve.

Milestone 06 is the continue/stop gate for the project as a whole.

## Scope

**Text only.** The model ships a vision encoder, and reimplementing a ViT is a
separate project. Multimodal requests route to a retained `llama-server`
instance; text-only requests go to this engine.

## Documentation

| | |
| --- | --- |
| [AGENTS.md](AGENTS.md) | Instructions for AI agents; the binding design rules |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Setup, workflow, commit and review conventions |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Component design and rationale |
| [docs/TOOLCHAIN.md](docs/TOOLCHAIN.md) | Milestone-00 gate results and the cudarc decision |
| [docs/MODEL.md](docs/MODEL.md) | Qwen3.6 structure, VRAM and bandwidth analysis |
| [docs/CACHE.md](docs/CACHE.md) | Hybrid two-group cache and prefix sharing |
| [docs/SCHEDULER.md](docs/SCHEDULER.md) | Chunked prefill, admission, preemption |
| [docs/KERNELS.md](docs/KERNELS.md) | Kernel inventory, risk, and porting notes |
| [docs/TESTING.md](docs/TESTING.md) | Differential harness and numerics thresholds |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Environment, reference checkouts, baseline |

## Baseline

The configuration this project is measured against:

```sh
llama-server -a qwen3.6-35b-a3b -m Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf \
  -sm none -ngl 99 --fit off -t 4 -tb 4 --poll 0 \
  -c 393216 -np 3 -b 4096 -ub 4096 \
  -fa on -ctk f16 -ctv f16 -cram 10240 \
  --jinja --reasoning-preserve \
  --temp 1.0 --top-p 0.95 --top-k 20 --presence-penalty 1.5
```

Three improvements to that baseline are available today and need no Rust at
all — see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md#baseline-improvements-available-now).

## License

MIT OR Apache-2.0.
