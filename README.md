# llmxabe

A single-process CUDA inference engine for `Qwen3.6-35B-A3B` on 3× Quadro
RTX 8000, written in Rust.

> **Status: the model generates.** It is resident on one GPU (733 tensors,
> 29.65 GiB, read back bit-identical), a full 40-block forward pass runs
> entirely on the device — Gated DeltaNet, gated attention, and the
> 256-expert MoE — and it now decodes autoregressively against a KV cache and
> a carried recurrent state. Against golden data captured from llama.cpp it
> produces **the same argmax token** (25358, `' Tokyo'`) at logit 19.936268,
> and prefilling 12 tokens then decoding 7 reaches that same token with
> cosine 1.000000000 against prefilling all 19 at once.
>
> What it cannot do: there is **no HTTP surface**, no tokenizer, and no
> batching across sequences. It generates token ids for one sequence at a
> time — decode steps replay from a captured CUDA graph. See
> [Milestones](#milestones) for the itemized state.
>
> Measured against llama.cpp on the same card: prefill **1,362.83 tok/s**
> at 512 tokens against 2,070.50 (**1.52× slower**), decode **104.3 tok/s**
> against `tg128`'s 104.72 (**level**). See
> [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

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
qualification. It is now also measured: resubmitting a 25,136-token prompt
takes 0.3 s warm against 9.7 s cold — a **32× improvement in time-to-first-token**
that today is confined to one process, so three replicas hold three copies of
it.

The performance bets have been benchmarked, and the results were not what the
design plan expected. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

- **Fused MoE dispatch** and **CUDA graph capture** are *already implemented in
  llama.cpp* (`mmid.cu`, `topk-moe.cu`, `USE_CUDA_GRAPH` — confirmed live at
  runtime), as is Turing-specific kernel tuning. These are not wins this
  project can claim; they are the bar it has to clear.
- **The real headroom is efficiency.** llama.cpp reaches 104.79 tok/s at short
  context, which is 44.5% of the 235 tok/s bandwidth roofline. That gap
  decomposes: the MoE weight path runs at 44.5% of peak bandwidth while flash
  attention streams KV at ~80%. Closing it is worth roughly 1.8× — but it means
  beating an already-fused, already-graphed, already-tuned implementation.
- **A GPU-side sampler looked like a win the plan missed** — a 248,320-token
  vocabulary makes CPU penalty sampling cost 22–24% of decode throughput. On
  testing, llama.cpp already ships one (`--backend-sampling`, off by default),
  and enabling it recovers the whole tax (+36%). That is the second identified
  opportunity to turn out already-implemented upstream.
- **Compile-time specialization.** Shapes are fixed and known. A
  general-purpose engine cannot assume that; this one can. Still unmeasured.

This project does not claim it will be faster than llama.cpp. That backend has
years of CUDA tuning behind it, and the benchmarks make that concrete.

**Where it actually stands** (2026-08-16, `bench_forward` n = 512 and
`bench_decode` on one card):

| | llama.cpp | llmxabe | position |
| --- | ---: | ---: | --- |
| Prefill, 512 tokens | 2,070.50 ± 160.35 tok/s | **1,361.42 ± 5.52** | 1.52× slower |
| Decode, warm | 104.72 ± 0.36 tok/s | **104.3**, 9.59 ms/step | 1.00× — level |

Prefill was 29.6× slower, then 10.3×, and is now 1.52×. Decode was 1.61× and
is now 1.01×. **Both are still losses**, and the project does not claim
otherwise.

The move that closed most of it was putting every quantized matmul on
Turing's integer tensor cores — `mma.m8n8k16.s32.s8.s8.s32`, ~198 TOP/s
against fp32's 16.3 TFLOP/s. Q6_K and Q8_0 weights are *already integers*,
so this is not a precision downgrade imposed on float weights; it is
declining to convert integers into floats in order to multiply them more
slowly. Roughly half the remaining gain, though, came not from arithmetic but
from finding kernels that re-read the same bytes — the recurrent state read
once per token instead of once per chunk, the router's weight row read once
per (expert, token) pair.

Decode needed almost none of that. At one token every matmul is a GEMV, and
the wins were removing GEMM machinery that had nothing to do at that shape,
plus one layout accident: Q8_0 puts a row's quants at byte `b * 34 + 2`, so
fifteen warp reads in sixteen straddle a 32-byte sector boundary and half of
every fetch is discarded. See [BENCHMARKS.md](docs/BENCHMARKS.md).

`ncu` still cannot read performance counters on this host
(`ERR_NVGPUCTRPERM`), so every attribution above is from `nsys` kernel
timings and arithmetic, not from hardware counters.

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

### Console output

Every binary routes its output through [`tracing`] and takes the same flag:

```sh
--log-level info | debug | trace     # default: info
```

`info` is the tool's own output — tables, results, summaries — and is what you
get with no flag. `debug` adds setup detail: NVRTC compilation per kernel with
timings, arena staging, resolved geometry. `trace` adds per-item detail, such
as all 753 tensors as they are uploaded.

There is deliberately no `warn` or `error` setting. `tracing`'s filter is an
ordering, so warnings and errors are visible at every level the flag accepts;
offering them as values would only let a caller hide problems. `INFO`, `DEBUG`
and `TRACE` go to stdout and `WARN`/`ERROR` to stderr, so piping a table
somewhere still leaves diagnostics on the terminal.

`RUST_LOG` is honoured when the flag is absent, including per-target
directives (`RUST_LOG=xabe_engine::weights=trace`). An explicit `--log-level`
overrides it and says so.

[`tracing`]: https://docs.rs/tracing

Tests that read the real model file look for it at
`$LLMXABE_MODEL`, falling back to the path in `docs/DEVELOPMENT.md`. They skip
if it is absent rather than failing.

## Milestones

Numbering follows the design plan. "Gate" is the condition for calling it done.

| # | Milestone | Gate | Status |
| --- | --- | --- | --- |
| 00 | Toolchain spike | Inline PTX confirmed, or cudarc chosen | done |
| 01 | GDN kernel prototype | Cosine ≥ 1e-3 vs reference | **done** — decode form max_abs 2.98e-8, cosine 1.000000000 over 512 tokens; chunked prefill form landed and matched against golden data |
| 02 | Differential harness | Per-tensor max-abs + cosine thresholds | done |
| 03 | FP16 dense forward | Correct logits, any speed | **done** (fp32, not fp16) — full 40-block pass, argmax 25358 matching llama.cpp |
| 04 | Q6_K dequant + MoE grouped GEMM | Correct, single GPU | **done** — dequant bit-identical over 8.4 M elements; grouped GEMM tiled, 9.3× on the MoE path, expert ids exact on all 37 tokens × top-8 |
| 05 | Flash attention port, sm_75 | Correct at 128K | **correct at batch and decode shape, unverified at 128K** — scalar fp32, `BM = 1`, no tensor cores; gated against golden data at 19 and 512 tokens and against the batch path at decode shape. At `n_query = 1` the grid is 16 blocks on 72 SMs and the KV read runs at ~2.9% of peak bandwidth — see BENCHMARKS.md |
| 05b | Autoregressive decode | Incremental path equals batch path | **done** — KV cache + carried recurrent state; prefill-12-then-decode-7 matches a 19-token prefill at cosine 1.000000000, same argmax. 65.25 tok/s at 128-token context vs llama.cpp's 104.72 |
| 06 | CUDA graph capture | Was "the justification gate"; llama.cpp already does this — see BENCHMARKS.md | not started; launch overhead measured at ~0.4% at n=512, so it is no longer a gate |
| 07 | Two-group pager + scheduler | 3 slots, matches llama.cpp `-np 3` | host side done; the device side runs a **single-sequence contiguous** KV cache (`block::attention::KvCache`), not yet the pager |
| 08 | Multi-worker + router + shared cache | Hit rate ≥ llama.cpp baseline | host side done |
| 09 | MTP speculative decode | Accept rate vs n-gram baseline | not started |

Milestone 01 deliberately precedes the harness. Gated DeltaNet covers 75% of
layers, has no equivalent in any flash-attention codebase, and its prefill form
needs per-chunk triangular matrix inversion. If it does not come together,
nothing else matters — and that should be discovered in week three, not week
twelve.

Milestone 06 was designated the continue/stop gate for the project as a whole,
on the assumption that graph capture over on-device MoE indirection was an
unclaimed win. Benchmarking showed llama.cpp already captures graphs on every
decode step, so that gate needs restating in terms of measured throughput
against 104.79 tok/s rather than "does graph capture help".

## Scope

**Text only.** The model ships a vision encoder, and reimplementing a ViT is a
separate project. Multimodal requests route to a retained `llama-server`
instance; text-only requests go to this engine.

## Documentation

| | |
| --- | --- |
| [AGENTS.md](AGENTS.md) | Instructions for AI agents; the binding design rules |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Setup, workflow, commit and review conventions |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | **Measured llama.cpp baseline — start here for performance** |
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
  -bs \
  --temp 1.0 --top-p 0.95 --top-k 20 --presence-penalty 1.5
```

`-bs` (`--backend-sampling`) is off by default and is the largest free win
measured here. It keeps sampling on the GPU, which makes penalty samplers cost
nothing — **+36%** on this configuration — so Qwen's published thinking-mode
defaults, `--presence-penalty 1.5` included, become free.

Measured baseline tuning results need no Rust at all — see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md#baseline-tuning--now-measured).

## License

MIT OR Apache-2.0.
