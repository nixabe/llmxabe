# Development environment

## Target host

| | |
| --- | --- |
| GPUs | 3× Quadro RTX 8000 — 47.3 GiB usable each, sm_75, 672 GB/s |
| Driver | 595.84 |
| CUDA | 12.4 |
| Host | 125 GB RAM, 12 vCPU |
| Rust | nightly, pinned by `rust-toolchain.toml` |

Device figures above are measured, not quoted — run
`cargo run -p xabe-cuda --bin probe` to reproduce them. Two are worth noting
because the design plan carried them as estimates:

- Peak bandwidth derives to **exactly 672 GB/s** from a 384-bit bus at 7001 MHz
  effective. Every roofline in [MODEL.md](MODEL.md) is stated against this.
- Usable memory is **47.3 GiB**, against the plan's ~47.5 GiB assumption. The
  VRAM budget in [MODEL.md](MODEL.md) uses the measured figure.

## Building

```sh
cargo build --workspace
cargo test --workspace
```

The host-side crates — `xabe-gguf`, `xabe-model`, `xabe-cache`, `xabe-sched`,
`xabe-kernels` — need no GPU and no CUDA toolkit. This is deliberate: the
correctness work that matters most (kernel references, cache geometry,
scheduler invariants) must not be gated on device access.

`xabe-cuda` needs CUDA 12.x. It builds against `cudarc` with the `cuda-12040`
feature and `fallback-dynamic-loading`, so it links against whatever driver is
present at runtime.

## Reference checkouts

Two upstream projects are checked out locally. They are used for different
purposes and should not be substituted for one another — see
[AGENTS.md](../AGENTS.md#where-the-answers-live).

| Project | Path | Used for |
| --- | --- | --- |
| llama.cpp | `/home/nixabe/llama.cpp` | Anything that must run on sm_75 |
| vLLM | `/home/nixabe/vllm` | Everything above the kernel |

> The design plan gives vLLM's path as `/home/nixabe/vLLM`. It is actually
> lowercase `vllm`.

Paths inside those trees drift on master. Treat the tables in
[KERNELS.md](KERNELS.md) as areas to search, not addresses, and confirm against
the checkout in front of you.

llama.cpp master already ships `ggml/src/ggml-cuda/gated_delta_net.cu`, which
is the only Turing-validated reference for this project's critical-path kernel.
Its HEAD also serializes MTP ubatches, which answers the plan's open question
about whether MTP is supported for this architecture.

## Model files

| | |
| --- | --- |
| Weights | `/home/nixabe/llama.cpp/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf` (32 GB) |
| Vision encoder | `mmproj-F16.gguf` (899 MB), same directory |

Tests that read the model honour `$LLMXABE_MODEL` and fall back to the path
above. They **skip** when it is absent rather than failing, so the workspace
stays testable on a machine without 32 GB of weights.

Never load the weights into memory — `mmap` them. Never commit them; `*.gguf`
is gitignored.

## Baseline

The llama.cpp configuration this project is measured against:

```sh
llama-server -a qwen3.6-35b-a3b \
  -m Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf \
  -sm none -ngl 99 --fit off \
  -t 4 -tb 4 --poll 0 \
  -c 393216 -np 3 -b 4096 -ub 4096 \
  -fa on -ctk f16 -ctv f16 -cram 10240 \
  --mmproj mmproj-F16.gguf --image-min-tokens 1024 \
  --jinja --reasoning-preserve \
  --temp 1.0 --top-p 0.95 --min-p 0.0
```

`-sm none` gives one full model instance per card rather than splitting one
model across three, which is correct here: the model fits, and splitting would
put PCIe on the decode path. `-t 4 -tb 4` partitions 12 vCPU across three
replicas, and `--poll 0` avoids three spinning pollers competing for them.

### Baseline improvements available now

Three changes need no Rust at all and should be made independently of this
project.

1. **Add `--top-k 20` and `--presence-penalty 1.5`.** These are Qwen's
   published thinking-mode defaults and are currently unset. Presence penalty
   is their stated remedy for repetition loops.

2. **A/B n-gram speculation against MTP.** The baseline uses
   `--spec-type ngram-map-k4v`. n-gram wins on repetitive spans — code edits,
   file rewrites — but this model ships a *trained* MTP head, and both the vLLM
   and SGLang recipes recommend 2–3 speculative tokens for it. On varied prose
   MTP is the likelier winner. llama.cpp master supports MTP for this
   architecture, so this is runnable today.

3. **Benchmark `-ctk q8_0 -ctv q8_0`.** This is the highest-value experiment
   available without writing any Rust. It halves the dominant bandwidth term at
   long context — 4.92 GB → 3.58 GB per token at 128K, a 37% roofline
   improvement. The cost is dequantization work inside the attention kernel,
   which on Turing is not free, so it must be measured rather than assumed.

## Open questions

Carried from the design plan. Answered ones are struck through with what was
found.

1. Measured decode rate at 4K, 32K, and 128K today. Every roofline in
   [MODEL.md](MODEL.md) is a ceiling with no measured floor beside it. **Still
   open, and the most valuable single measurement available.**
2. VRAM cost of `mmproj-F16.gguf`. The file is 899 MB on disk; its resident
   cost is still an estimate.
3. What fraction of platform traffic is multimodal — determines whether the
   retained llama.cpp instance is permanent or transitional.
4. ~~Confirm the GDN state shape from `config.json`.~~ Derived as
   32 × 128 × 128 × fp32 per layer; see [MODEL.md](MODEL.md) for what the GGUF
   metadata shows.
5. ~~Does llama.cpp master support MTP for this architecture?~~ Yes — HEAD
   includes MTP ubatch serialization.
6. Observed `sim_best` distribution in the server logs. That is the current
   cache hit rate and the baseline [CACHE.md](CACHE.md) must beat. **Open.**
7. Does vLLM's GDN prefill kernel depend on Ampere-or-later features? If the
   chunked delta rule assumes `cp.async` or bf16, llama.cpp is the only viable
   reference. **Open.**

## Conventions

Commit format, review expectations, and the correctness standard are in
[CONTRIBUTING.md](../CONTRIBUTING.md). Agent-specific rules are in
[AGENTS.md](../AGENTS.md).
