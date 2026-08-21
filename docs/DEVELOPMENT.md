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

No crate needs a GPU or a CUDA toolkit to build and test, including
`xabe-cuda`. This is deliberate: the correctness work that matters most
(kernel references, cache geometry, scheduler invariants) must not be gated on
device access.

`xabe-cuda` has no build script, and builds against `cudarc` with the
`cuda-12040` feature, `fallback-dynamic-loading` and `nvrtc` — so it needs no
headers and no `nvcc` at build time, resolving the driver and compiling
kernels at runtime instead. Nothing is needed on `PATH` to compile it.

That property is load-bearing rather than incidental, and it is easy to break
by accident. Everything device-gated funnels through
`xabe_cuda::device::driver_available`, which has to answer `false` — not
abort — on a machine with no driver, because `cudarc` panics rather than
erroring when it cannot find `libcuda` at all.

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

The tuned form of this, after the measurements below, adds `-bs` and Qwen's
sampling defaults:

```sh
  ... -bs --temp 1.0 --top-p 0.95 --top-k 20 --presence-penalty 1.5
```

`-sm none` gives one full model instance per card rather than splitting one
model across three, which is correct here: the model fits, and splitting would
put PCIe on the decode path. `-t 4 -tb 4` partitions 12 vCPU across three
replicas, and `--poll 0` avoids three spinning pollers competing for them.

### Baseline tuning

Benchmarked on this host; the reasoning behind each is in
[BENCHMARKS.md](BENCHMARKS.md).

1. **Add `-bs` (`--backend-sampling`). This is the largest free win available.**
   It moves sampling into the compute graph and is `false` by default.

   Without it, **any** penalty sampler — `presence_penalty`, `repeat_penalty`,
   `frequency_penalty` — costs 22–24% of decode throughput, binary rather than
   proportional (`0.1` costs the same as `1.5`), because a 248,320-token
   vocabulary is processed on the CPU every step. With `-bs`, that cost is
   **zero**: throughput stays flat at 102–103 tok/s whatever the penalty
   settings, a **+36%** improvement on Qwen's recommended configuration. It is
   also 4.7% faster with no penalty at all, because the 993 KB per-token logits
   copy back to the host disappears.

   Verified sampling correctly, not collapsing to greedy: three seeds give
   three distinct outputs, one seed reproduces exactly, and reasoning and
   long-context retrieval both stay correct.

2. **Then add `--top-k 20` and `--presence-penalty 1.5`.** With `-bs` enabled
   these are effectively free, so Qwen's published thinking-mode defaults —
   including their stated remedy for repetition loops — cost nothing.

3. **Keep `-ctk f16 -ctv f16`. Do not switch to `q8_0`.** This was predicted to
   be the highest-value experiment available; it is measurably worse at every
   depth and dramatically worse where it was supposed to help most: −15.5% at
   32K and **−35.8% at 128K**. The KV path already runs at ~80% of peak
   bandwidth, so there was little to buy, and Turing dequantization inside the
   attention kernel costs more than the saving.

4. **Choose `-np` deliberately.** Three slots buy 1.77× aggregate throughput
   and cost 41% per-request latency, plus a further ~9.5% per-request
   throughput at depth versus `-np 1`. Latency-sensitive, rarely-concurrent
   workloads are measurably better served by `-np 1`.

5. **A/B n-gram speculation against MTP.** Still unmeasured. The baseline uses
   `--spec-type ngram-map-k4v`; n-gram wins on repetitive spans, but this model
   ships a *trained* MTP head and both the vLLM and SGLang recipes recommend
   2–3 speculative tokens for it. llama.cpp master supports MTP for this
   architecture, so this is runnable today.

## Open questions

Answered, so they are not re-asked:

- **Decode rate at depth on the baseline.** 103.4 / 92.4 / 68.8 tok/s at
  4K / 32K / 128K single stream, against rooflines of 229 / 191 / 121. The
  weight path runs at 44.5% of peak bandwidth and the KV path at ~80%.
- **The GDN state shape.** 32 × 128 × 128 × fp32 per layer, confirmed against
  the file's own metadata (`ssm.state_size 128`, `ssm.inner_size 4096`,
  `ssm.time_step_rank 32`) and every GDN tensor shape. The device kernel runs
  at this geometry and agrees with the reference to 2.98e-8.
- **Was `ModelConfig`'s parameter count 2.4% low?** No — an accounting error.
  The gap was entirely `blk.40`, the MTP head, counted on the file side and not
  the config side. Like for like on the text path the derivation is accurate to
  0.001%. See [MODEL.md](MODEL.md).
- **Does llama.cpp master support MTP for this architecture?** Yes.

Still open:

- VRAM cost of `mmproj-F16.gguf`. The file is 899 MB on disk; its resident cost
  is still an estimate.
- What fraction of platform traffic is multimodal — determines whether the
  retained llama.cpp instance is permanent or transitional.
- The GDN short convolution cache (~2.8 MiB per sequence) is not modelled by
  `gdn_state_bytes_per_sequence()`. Small, but a real omission rather than a
  rounding choice.
- Observed `sim_best` distribution in the server logs — the cache hit rate
  [CACHE.md](CACHE.md) must beat.
- Whether vLLM's GDN prefill kernel depends on Ampere-or-later features. If the
  chunked delta rule assumes `cp.async` or bf16, llama.cpp is the only viable
  reference.

## Conventions

Commit format, review expectations, and the correctness standard are in
[CONTRIBUTING.md](../CONTRIBUTING.md). Agent-specific rules are in
[AGENTS.md](../AGENTS.md).
