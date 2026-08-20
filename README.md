# llmxabe

A single-process CUDA inference engine for `Qwen3.6-35B-A3B` on 3× Quadro
RTX 8000, written in Rust.

> **Status: the engine serves, and it is ahead of the baseline.** The model is
> resident on one GPU, a full 40-block forward pass runs entirely on the
> device — Gated DeltaNet, gated attention, and the 256-expert MoE — it
> decodes autoregressively against a KV cache and a carried recurrent state,
> and it batches prefill and decode across parallel sequences. Against golden
> data captured from llama.cpp it produces **the same argmax token** (25358,
> `' Tokyo'`), and a sequence decodes **bit-identically** whether batched with
> others or run alone.
>
> Measured one card, three concurrent sequences, against llama.cpp at its own
> best settings, same card and same hour, alternating processes: **prefill
> ahead at every context from 512 to 128K** (+0.8% to +18.6%) and **decode
> ahead at 2K (+4.4%) and 32K (won every pair)**. See
> [docs/BENCHMARKS.md](docs/BENCHMARKS.md).
>
> What it cannot do: streaming responses, tokenization, and multimodal input.
> See [Milestones](#milestones) for the itemized state.

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

The *performance* bets were benchmarked, and two of the three turned out to be
things the baseline already did. **Fused MoE dispatch** and **CUDA graph
capture** are already implemented in llama.cpp (`mmid.cu`, `topk-moe.cu`,
`USE_CUDA_GRAPH` — confirmed live at runtime), as is Turing-specific kernel
tuning, and so is the GPU-side sampler that looked like a third opening
(`--backend-sampling`, off by default, worth +36% when enabled). None of those
are wins this project can claim; they are the bar it had to clear.

What was left was efficiency, and that turned out to be enough.

**Where it stands** — one card, three concurrent sequences, against llama.cpp
at `-np 3 -b 4096 -ub 4096`, measured same card, same hour, alternating
processes:

| cell | llmxabe | llama.cpp | margin |
| :--- | ---: | ---: | ---: |
| prefill 512 | 3,281 tok/s | 3,008 | **+9.1%** |
| prefill 2K | 3,797 | 3,202 | **+18.6%** |
| prefill 8K | 3,666 | 3,202 | **+14.5%** |
| prefill 32K | 2,790 | 2,655 | **+5.1%** |
| prefill 65K | 2,226 | 2,143 | **+3.9%** |
| prefill 128K | 1,565 | 1,552 | **+0.8%** |
| decode 2K | 193.3 | 185.2 | **+4.4%** |
| decode 32K | 151.4 (median) | 150.7 | won every pair |

Read the two thinnest rows — 128K prefill and 32K decode — as **level to
slightly ahead**, not as headlines. This card's thermal drift is ~1.3% and
card-to-card spread ~1.5%, both larger than those margins; they stand only
because they were taken as alternating same-hour pairs on one card and won all
of them. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for the method.

Prefill was 29.6× slower, then 10.3×, then 1.20×, and is now ahead. The step
that closed it was not the MoE GEMM, which was already faster than llama.cpp's:
it was four kernels — the flash attention block, the Gated DeltaNet state
update and solve, and the alpha/beta gates — that had each been written with
one memory instruction per multiply-add, on a part that issues four of the
former per SM per clock against sixty-four of the latter.

The move that closed most of it was putting every quantized matmul on Turing's
integer tensor cores — `mma.m8n8k16.s32.s8.s8.s32`, ~198 TOP/s against fp32's
16.3 TFLOP/s. Q6_K and Q8_0 weights are *already integers*, so this is not a
precision downgrade imposed on float weights; it is declining to convert
integers into floats in order to multiply them more slowly. Roughly half the
remaining gain, though, came not from arithmetic but from finding kernels that
re-read the same bytes — the recurrent state read once per token instead of
once per chunk, the router's weight row read once per (expert, token) pair.

Decode needed almost none of that. At one token every matmul is a GEMV, and the
wins were removing GEMM machinery that had nothing to do at that shape, filling
the card's block slots exactly once per step, and one layout accident: Q8_0
puts a row's quants at byte `b * 34 + 2`, so fifteen warp reads in sixteen
straddle a 32-byte sector boundary and half of every fetch is discarded.

**Accuracy is a hard gate, not a budget.** Nothing above cost a loosened
tolerance. A sequence decodes bit-identically whether batched with others or
alone, which is the serving contract three instances on one card require, and
several real speedups were rejected for breaking it.

`ncu` cannot read performance counters on this host (`ERR_NVGPUCTRPERM`), so
every attribution above is from `nsys`/`nvprof` timings, `ptxas`/`cuobjdump`
and arithmetic, not from hardware counters.

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
so a preflight that builds these types has checked them.

The HTTP surface serves non-streaming `/v1/completions` across all three cards.
Streaming, disconnect cancellation and overload behaviour are not implemented —
see [docs/TESTING.md](docs/TESTING.md) for what the serving checks do and do
not cover.

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
| 05 | Flash attention port, sm_75 | Correct at 128K | **done** — `m16n8k8` tensor cores with fp32 accumulation for prefill, split-K flash decoding with a depth-aware tensor-core path for decode, binary16 KV. Gated at nine depths including a 128K window |
| 05b | Autoregressive decode | Incremental path equals batch path | **done** — KV cache + carried recurrent state; incremental and batch paths agree at cosine 1.000000000, same argmax |
| 06 | CUDA graph capture | Was "the justification gate"; llama.cpp already does this — see BENCHMARKS.md | **done**, and worth ~0 on throughput; kept for host cost and launch-shape discipline |
| 07 | Two-group pager + scheduler | 3 slots, matches llama.cpp `-np 3` | **done** — batched decode and flattened batch prefill at N=1–8, scheduler path within 0.7% of the isolated kernel path |
| 08 | Multi-worker + router + shared cache | Hit rate ≥ llama.cpp baseline | **done** — nine requests balanced three per card, mixed decode/prefill steps on every worker, cross-worker restore emits identical token ids |
| 09 | MTP speculative decode | Accept rate vs n-gram baseline | implemented, **not adopted** — 65.2% acceptance for ~7%, and unbatched across sequences |

Milestone 01 deliberately precedes the harness. Gated DeltaNet covers 75% of
layers, has no equivalent in any flash-attention codebase, and its prefill form
needs per-chunk triangular matrix inversion. If it does not come together,
nothing else matters — and that should be discovered in week three, not week
twelve.

Milestone 06 was designated the continue/stop gate for the project as a whole,
on the assumption that graph capture over on-device MoE indirection was an
unclaimed win. Benchmarking showed llama.cpp already captures graphs on every
decode step, and that on Turing a graph node costs roughly what the launch gap
it replaces cost. The gate was restated as measured throughput against
llama.cpp's own best settings, which is what the table above reports.

## Scope

**Text only.** The model ships a vision encoder, and reimplementing a ViT is a
separate project. Multimodal requests route to a retained `llama-server`
instance; text-only requests go to this engine.

## Documentation

| | |
| --- | --- |
| [AGENTS.md](AGENTS.md) | Instructions for AI agents; the binding design rules |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Setup, workflow, commit and review conventions |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | **Current standing, and why / why not — start here for performance** |
| [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md) | The bandwidth model, what transfers from vLLM, and the ceilings |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Component design and rationale |
| [docs/TOOLCHAIN.md](docs/TOOLCHAIN.md) | Milestone-00 gate results and the cudarc decision |
| [docs/MODEL.md](docs/MODEL.md) | Qwen3.6 structure, VRAM and bandwidth analysis |
| [docs/CACHE.md](docs/CACHE.md) | Hybrid two-group cache and prefix sharing |
| [docs/SCHEDULER.md](docs/SCHEDULER.md) | Chunked prefill, admission, preemption |
| [docs/KERNELS.md](docs/KERNELS.md) | Kernel inventory, risk, and porting notes |
| [docs/TESTING.md](docs/TESTING.md) | Differential harness and numerics thresholds |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Environment, reference checkouts, baseline |

## Baseline

The serving configuration this project replaces:

```sh
llama-server -a qwen3.6-35b-a3b -m Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf \
  -sm none -ngl 99 --fit off -t 4 -tb 4 --poll 0 \
  -c 393216 -np 3 -b 4096 -ub 4096 \
  -fa on -ctk f16 -ctv f16 -cram 10240 \
  --jinja --reasoning-preserve \
  -bs \
  --temp 1.0 --top-p 0.95 --top-k 20 --presence-penalty 1.5
```

`-bs` (`--backend-sampling`) is off by default and is the largest free win in
it. It keeps sampling on the GPU, which makes penalty samplers cost nothing —
**+36%** — so Qwen's published thinking-mode defaults, `--presence-penalty 1.5`
included, become free.

The throughput bar is the same model and flags under `llama-batched-bench` at
`-npl 3`, taken at whichever of `-ub 2048` / `-ub 4096` is faster for the cell.
Comparing against llama.cpp's *defaults* is not a result; see
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## License

MIT OR Apache-2.0.
