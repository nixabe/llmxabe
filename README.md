# llmxabe

A single-process CUDA inference engine for `Qwen3.6-35B-A3B` on 3× Quadro
RTX 8000, written in Rust.

> **Status: the engine serves, and it is ahead of the baseline.** The model is
> resident on one GPU, a full 40-block forward pass runs entirely on the
> device — Gated DeltaNet, gated attention, and the 256-expert MoE — it
> decodes autoregressively against a KV cache and a carried recurrent state,
> and it batches prefill and decode across parallel sequences. Against golden
> data captured from llama.cpp it produces **the same argmax token**, and a
> sequence decodes **bit-identically** whether batched with others or run
> alone.
>
> Measured one card, three concurrent sequences, against llama.cpp at its own
> best settings: **ahead on every cell, prefill 512–128K and decode 2K/32K**.
> The current numbers live in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).
>
> It serves OpenAI's completions, chat completions and responses dialects and
> Anthropic's messages, streaming or not, behind an optional API key, and
> tokenizes from the vocabulary embedded in the GGUF.
>
> It samples — `temperature`, `top_p`, `top_k`, `seed`, with `temperature: 0`
> as greedy argmax — it speaks tool calls in all three chat dialects, and with
> `--mmproj` it takes inline images in all three as well. See
> [docs/API.md](docs/API.md) for what the endpoints accept and refuse, and
> [docs/MILESTONES.md](docs/MILESTONES.md) for the engine milestones.

## Why this exists

The obvious way to serve this model on three GPUs is three `llama-server`
processes, one per card. That works, and it is the baseline this project is
measured against — the exact configuration is in
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

It has one structural weakness: each process has a private prefix cache. Three
replicas serving one workload cache the same system prompt three times, and a
request routed to the wrong replica pays full prefill for a prefix that another
replica already holds. Sharing that cache across OS processes means a
serialization format, a filesystem, a hashing scheme, and an eviction
policy — every one of which is a source of silent corruption or a privacy leak
on a multi-user platform.

In one address space it is a radix tree behind a lock.

That is the *architectural* claim, and it is measured: resubmitting a
25,136-token prompt takes 0.3 s warm against 9.7 s cold — a **32× improvement
in time-to-first-token** that today is confined to one process, so three
replicas hold three copies of it.

The *performance* claim — faster than llama.cpp at its own best settings, on
its own strongest workload — is measured too. How it was won, what was tried
and rejected, and the discipline behind the numbers are all in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md); the model that ranks the remaining
work is in [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md).

**Accuracy is a hard gate, not a budget.** Nothing was bought with a loosened
tolerance. A sequence decodes bit-identically whether batched with others or
alone — the serving contract three instances on one card require — and several
real speedups were rejected for breaking it.

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
   holding any KV at all. KV reads overtake weight reads at about 139K tokens.
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

The whole workspace builds and tests without a GPU or CUDA toolkit — that is
what CI runs. Tests that need a device detect its absence and skip, reporting
that they skipped. Tests that read the real model file look for it at
`$LLMXABE_MODEL`, falling back to the path in
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md), and skip if it is absent.

Running device work needs a Turing-or-later GPU and its driver at runtime:

```sh
cargo run -p xabe-cuda --bin probe     # device inventory and sm_75 gate
cargo run -p xabe-server               # full engine preflight
```

The preflight validates the whole startup path — model config, cache geometry,
scheduler construction, device gate, VRAM budget against the card's *measured*
memory, and engine assembly.

The HTTP surface serves OpenAI's `/v1/completions`, `/v1/chat/completions` and
`/v1/responses`, and Anthropic's `/v1/messages`, across all three cards —
streaming or not, with optional API-key authentication, with sampling and
tool calling in every chat dialect. See
[docs/API.md](docs/API.md) for the endpoints and what they refuse, and
[docs/TESTING.md](docs/TESTING.md) for what the serving checks do and do not
cover.

To serve it in a container instead, with the models directory mounted and
images enabled:

```sh
docker compose up --build
```

That runs the configuration this project measured, on every card the NVIDIA
container runtime exposes. What it sets and why is in
[docs/DOCKER.md](docs/DOCKER.md).

Every binary logs through `tracing` and takes `--log-level info | debug |
trace`; the levels and their meanings are in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Scope

**Text, and images behind `--mmproj`.** The model's vision tower runs on the
device per worker, differential-tested against a CPU reference that is itself
validated against llama.cpp executing the same `mmproj-F16.gguf`. Vision is
opt-in: a server started without `--mmproj` allocates nothing for it and
serves the text path unchanged. Video is out of scope. See the scope section
of [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Documentation

| | |
| --- | --- |
| [AGENTS.md](AGENTS.md) | Instructions for AI agents; the binding design rules |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Setup, workflow, console output, commit and review conventions |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | **Current standing, and why / why not — start here for performance** |
| [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md) | The bandwidth model, what transfers from vLLM, and the ceilings |
| [docs/MILESTONES.md](docs/MILESTONES.md) | The milestone plan, its gates, and their status |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Component design and rationale |
| [docs/TOOLCHAIN.md](docs/TOOLCHAIN.md) | Milestone-00 gate results and the cudarc decision |
| [docs/MODEL.md](docs/MODEL.md) | Qwen3.6 structure, VRAM and bandwidth analysis |
| [docs/CACHE.md](docs/CACHE.md) | Hybrid two-group cache and prefix sharing |
| [docs/SCHEDULER.md](docs/SCHEDULER.md) | Chunked prefill, admission, preemption |
| [docs/KERNELS.md](docs/KERNELS.md) | Kernel inventory, risk, and porting notes |
| [docs/TESTING.md](docs/TESTING.md) | Differential harness and numerics thresholds |
| [docs/API.md](docs/API.md) | HTTP endpoints, streaming, authentication, and what they refuse |
| [docs/CLI.md](docs/CLI.md) | The server binary's command-line arguments |
| [docs/DOCKER.md](docs/DOCKER.md) | Serving from the container image, and what the compose defaults mean |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Environment, reference checkouts, and the llama.cpp baseline configuration |

## License

MIT OR Apache-2.0.
