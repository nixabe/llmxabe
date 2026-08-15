# Baseline benchmarks

Measured llama.cpp performance on the target host. Until this document
existed, every performance number in this project was a bandwidth ceiling with
no floor beside it.

**These are measurements, not estimates.** Where they contradict the design
plan, the measurement wins and the contradiction is stated.

## Setup

| | |
| --- | --- |
| Hardware | 3× Quadro RTX 8000, sm_75, 47.3 GiB usable, 672 GB/s |
| Host | 125 GB RAM, 12 vCPU, CUDA 12.4, driver 595.84 |
| llama.cpp | build 10456, commit `fd6863a69` |
| Model | `Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf` — 30.36 GiB, 35.51 B params |
| Decode measurements | `llama-bench`, `-sm none -ngl 99 -fa on -t 4`, 3 repetitions |
| Serving measurements | `llama-server`, one GPU, `/completion` endpoint timings |

llama-bench independently reports 35.51 B parameters and 30.36 GiB, which
matches the tensor-directory sum in [MODEL.md](MODEL.md) exactly.

## Headline result

**Peak observed decode: 104.79 ± 0.21 tok/s** at short context, single stream,
f16 KV, no penalty samplers.

That is **44.5% of the 235 tok/s bandwidth roofline**. The gap is the
opportunity this project is trying to capture, and it is now a number rather
than a hope.

## Decode versus context depth

`llama-bench -p 0 -n 128 -d <depth> -r 3`, f16 KV.

| Depth | Measured tok/s | Roofline | Fraction of peak bandwidth |
| ---: | ---: | ---: | ---: |
| 0 | 104.79 ± 0.21 | 235 | 44.5% |
| 4,096 | 103.39 ± 0.65 | 229 | 45.2% |
| 32,768 | 92.36 ± 0.50 | 191 | 48.5% |
| 131,072 | 68.76 ± 0.43 | 121 | 56.7% |

**Efficiency rises with depth.** That is the most informative thing in the
table, and it decomposes cleanly. Fitting a two-term model — weights read at
efficiency `E_w`, KV read at efficiency `E_kv` — to the depth-0 and
depth-131,072 points gives:

| Path | Bytes/token | Achieved bandwidth | Fraction of 672 GB/s |
| --- | ---: | ---: | ---: |
| Weights (MoE + LM head + projections) | 2.86 GB | 299 GB/s | **44.5%** |
| KV (flash attention) | 20,480 B × depth | 537 GB/s | **~80%** |

That model then predicts the two depths it was not fitted to:

| Depth | Predicted | Measured | Error |
| ---: | ---: | ---: | ---: |
| 4,096 | 103.1 | 103.39 | 0.3% |
| 32,768 | 92.7 | 92.36 | 0.4% |

**The weight path is the inefficient one.** Flash attention streams KV at ~80%
of peak; the MoE weight path runs at 44.5%. As context grows, the efficient
term grows as a share of traffic, so aggregate efficiency improves even though
absolute throughput falls.

This is the quantified version of the design plan's central claim that MoE
decode is an indirection problem. It is real, and it is worth about **1.8× on
the weight-bound term** if the weight path could be brought to the KV path's
efficiency — 104.79 → roughly 188 tok/s at short context.

## Prefill

| Test | tok/s |
| --- | ---: |
| `pp512` | 2,010 ± 150 |
| `pp4096` | 1,950 ± 10 |
| Real 25,136-token prompt (server) | 2,732 |

Prefill is compute-bound and roughly 20–26× decode throughput, which is what
makes chunked prefill worth mixing with decode in one batch — see
[SCHEDULER.md](SCHEDULER.md).

## Three findings that contradict the plan

### 1. `q8_0` KV cache is much slower on Turing, not faster

The design plan called this "the highest-value experiment available without
writing any Rust", predicting a 37% roofline improvement at 128K from halving
the dominant bandwidth term.

| Depth | f16 KV | q8_0 KV | Change |
| ---: | ---: | ---: | ---: |
| 0 | 104.79 | 102.13 | −2.5% |
| 32,768 | 92.36 | 78.04 | **−15.5%** |
| 131,072 | 68.76 | 44.11 | **−35.8%** |

It is worse everywhere, and worst exactly where it was supposed to help most.

The plan flagged the risk — "the cost is dequant work inside the attention
kernel, which on Turing is not free" — and on Turing it is expensive enough to
more than consume the bandwidth saving. The KV path already runs at ~80% of
peak, so there was far less headroom to buy than the roofline suggested.

**Recommendation: keep `-ctk f16 -ctv f16`.**

### 2. Any penalty sampler costs 22–24% of decode throughput

Measured on the server at short context, 300–400 tokens generated:

| Sampling | tok/s | vs. no penalty |
| --- | ---: | ---: |
| Greedy (`temp=0, top_k=1`) | 99.14 | — |
| Qwen defaults, no penalty | 98.10 | −1% |
| `presence_penalty=0.0` (path off) | 98.61 | — |
| `presence_penalty=0.1` | 75.12 | **−24%** |
| `presence_penalty=1.5` | 75.53 | **−23%** |
| `repeat_penalty=1.1` | 76.98 | **−22%** |
| `frequency_penalty=0.5` | 77.34 | **−22%** |
| `top_k=0` (full-vocab sort) | 80.07 | −19% |

**The cost is binary, not proportional.** `presence_penalty=0.1` costs the same
as `1.5`; what matters is whether the penalty path runs at all. The mechanism
is the vocabulary: 248,320 untied tokens, processed on the CPU every step.
`top_k=0` corroborates it — any operation touching all logits costs about the
same.

This directly revises advice previously given in
[DEVELOPMENT.md](DEVELOPMENT.md): adding `--presence-penalty 1.5` is Qwen's
published remedy for repetition loops, but it is **not free**, and the ~23%
cost was not known when it was recommended. It is a quality-versus-throughput
trade, and it should be made deliberately.

It is also fixable **today**, and the fix was tested — see
[GPU-side sampling](#gpu-side-sampling-recovers-all-of-it) below.

## GPU-side sampling recovers all of it

The previous section identified a GPU-side sampler as a fix. **llama.cpp
already has one**: `-bs` / `--backend-sampling`, which builds the sampler into
the compute graph. It is `false` by default (`common/common.h`).

Enabling it removes the penalty tax completely.

| Sampling | CPU (default) | GPU (`-bs`) | Change |
| --- | ---: | ---: | ---: |
| `presence_penalty=0.0` (path off) | 98.61 | 103.23 | +4.7% |
| `presence_penalty=0.1` | 75.12 | 102.93 | **+37.0%** |
| `presence_penalty=1.5` | 75.53 | 102.85 | **+36.2%** |
| `repeat_penalty=1.1` | 76.98 | 102.18 | **+32.7%** |
| `frequency_penalty=0.5` | 77.34 | 102.50 | **+32.5%** |

Throughput becomes flat at 102–103 tok/s regardless of penalty configuration.
The penalty is no longer a trade at all.

Note the first row: GPU sampling is **4.7% faster even with no penalty
enabled**. That is the 248,320 × 4 B = 993 KB per-token logits copy back to the
host that no longer has to happen.

### Verified, not assumed

A sampler that silently collapsed to greedy would look fast and be wrong.

| Check | Result |
| --- | --- |
| Three different seeds | 3 distinct outputs — genuinely sampling |
| Same seed twice | Identical — reproducible |
| Reasoning (sheep, 137×24) | **9** and **3,288**, both correct, at 102.6 tok/s *with* `presence_penalty=1.5` |
| 25K needle retrieval | **4471**, correct |

### In the production configuration

With `-np 3 -c 393216 --backend-sampling`, penalties enabled throughout:

| Sampling | tok/s |
| --- | ---: |
| `presence_penalty=1.5` | 102.34 |
| `repeat_penalty=1.1` | 101.67 |
| `frequency_penalty=0.5` | 101.93 |

| Concurrency | Per-request | Aggregate |
| ---: | ---: | ---: |
| 1 | 102.29 | 82.96 |
| 2 | 77.08 | 129.54 |
| 3 | 62.97 | **160.93** |

That c=3 aggregate is measured **with penalties on**. The earlier 162.4 figure
was measured with penalties *off* and CPU sampling, so backend sampling buys
Qwen's recommended sampling configuration for essentially nothing.

Long context is unaffected: 82.36 tok/s at 25K depth with penalties, against
80.07 without `-bs` and without penalties.

**Recommendation: add `-bs` to the serving configuration.** It is the largest
free win found in this exercise — roughly **+36% on the recommended sampling
configuration** — and it requires no code.

### 3. llama.cpp already implements the plan's three performance bets

This is the finding that most changes the project's outlook, and it is worth
stating plainly.

| Plan's bet | Status in llama.cpp |
| --- | --- |
| Fused MoE dispatch | **Already done** — `ggml/src/ggml-cuda/mmid.cu`, `topk-moe.cu`, plus `ffn_up`+`ffn_gate`+GLU fusion in `ggml_cuda_should_fuse_mul_mat` |
| CUDA graph capture | **Already done** — `USE_CUDA_GRAPH` / `cudaGraphLaunch`, and confirmed live at runtime |
| Turing-specific tuning | **Already done** — `mmq-config-turing.cuh` carries hand-tuned per-type tile configurations for sm_75 |

Graph capture is not merely compiled in; it is active. Server logs report
`graphs reused = 1195` over a 1,200-token generation, and 5,175 across a longer
session — essentially every decode step replays a captured graph.

**Milestone 06 was designated "the justification gate" and "the largest single
expected win".** It is measuring a capability the baseline already has. That
does not make the milestone worthless — a purpose-built graph over fixed shapes
may still beat a general one — but it is no longer a step-change, and the
project's continue/stop criterion needs restating.

## Serving configuration

### Concurrency

`-np 3`, one GPU, 300 tokens per request:

| Concurrent requests | Per-request tok/s | Aggregate tok/s |
| ---: | ---: | ---: |
| 1 | 98.7 | 91.6 |
| 2 | 72.3 | 136.3 |
| 3 | 58.6 | **162.4** |

Batching buys **1.77×** aggregate throughput at three slots, costing 41%
per-request latency.

That it is 1.77× and not near 3× is itself informative: for MoE, batching does
not amortize expert weights well, because three tokens route to up to 24
distinct experts with little overlap. What does amortize is the shared expert,
the projections, and the LM head. The measurement is consistent with the sparse
routing structure.

Across three cards this implies roughly **487 tok/s aggregate**.

### Slot count costs throughput at depth

Same 25,136-token prompt, one GPU:

| Configuration | tok/s at 25K depth |
| --- | ---: |
| `-np 1 -c 131072` | 87.66 |
| `-np 3 -c 393216` | 80.07 |

About **9.5%**, paid for the concurrency above. Worth choosing deliberately
rather than inheriting: if the workload is latency-sensitive and rarely
concurrent, `-np 1` is measurably better.

### Prefix caching

The 25,136-token prompt, resubmitted:

| | Prompt tokens processed | Wall time |
| --- | ---: | ---: |
| Cold | 25,136 | 9.7 s |
| Warm (`cache_prompt`) | 4 | **0.3 s** |

A **32× improvement in time-to-first-token**, and the clearest empirical
support for this project's one unqualified architectural claim: that cache is
per-process today, so three `llama-server` replicas hold three copies and a
request routed to the wrong replica pays the full 9.7 s. See
[ARCHITECTURE.md](ARCHITECTURE.md#why-one-process).

## VRAM

Measured resident on GPU 0 with `-c 393216 -np 3`: **38.57 GiB**.

The preflight in `xabe-server` predicted **39.98 GiB** — 3.6% high, which is
the safe direction for a capacity planner. Model load took about 20 s with the
file in page cache.

## Correctness

End-to-end behaviour was verified, not assumed.

| Test | Result |
| --- | --- |
| Thinking mode | 2,832 characters of `reasoning_content`, correctly separated from `content` |
| "17 sheep, all but 9 run away" | **9** — correct, with correct idiom reasoning |
| 137 × 24 | **3,288** — correct |
| Needle retrieval in a 25,136-token log | **4471** — correct, needle at ~50% depth |

The chat template, reasoning-content separation, long-context attention, and
prefix cache all work as expected.

## Implications for this project

**What got weaker.** Two of the three performance bets — fused MoE dispatch and
CUDA graph capture — are already implemented in the baseline, and the Turing
kernels are already hand-tuned. The project cannot claim these as wins; it must
beat them. Milestone 06 is no longer a step-change and the continue/stop gate
should be reformulated around measured throughput against 104.79 tok/s rather
than against "does graph capture help".

**What got stronger.** The shared prefix cache is now empirically supported: a
32× TTFT improvement that is currently confined to one process. That remains
the project's one structural, non-speculative advantage.

**What was new, and then wasn't.** The penalty-sampler measurement looked like a
differentiator the plan had missed: a 248,320-token vocabulary makes CPU-side
sampling cost 22–24% of decode throughput. Testing it showed llama.cpp already
ships the fix (`--backend-sampling`), merely disabled by default, and enabling
it recovers the entire tax.

That is the second time an identified opportunity turned out to be already
implemented upstream — graph capture was the first. The pattern is worth
naming: **this baseline is more complete than the design plan assumed, and
candidate differentiators should be tested against it before being planned
around.** The remaining genuinely-unclaimed items are the shared prefix cache
and compile-time shape specialization.

**The real headroom** is the 44.5% weight-path efficiency. It is genuine, and
it is worth about 1.8× if fully captured — but capturing it means beating an
already-fused, already-graphed, already-Turing-tuned implementation, which is a
materially harder proposition than the plan assumed.

---

# llmxabe's own forward pass, and where its time goes

Everything above measures llama.cpp. This section measures **llmxabe**, on the
same host, the same card, the same file, on the same day, and puts the two
side by side.

It exists because the engine now runs end to end (`crates/xabe-engine/src/forward.rs`,
gated against the llama.cpp capture by `tests/forward_pass.rs`) and the first
question after "is it right" is "is it fast". The answer is no, and the rest of
this section is the arithmetic of *why*, measured rather than guessed.

## Head to head

> **Superseded by [the re-measurement below](#head-to-head-after-the-moe-tiling).**
> Everything in this section, and every per-stage figure that follows it, was
> measured *before* the grouped-GEMM tiling landed. It is kept because the
> per-stage breakdown is what identified the defect, and because the
> after-figures are only meaningful against it. **Do not quote 73.48 tok/s or
> 27.9× as current.**

Both sides measured on **GPU 2** of this host on 2026-08-16, one process on the
card at a time.

```sh
# llama.cpp, build fd6863a69 (10456)
CUDA_VISIBLE_DEVICES=2 llama-bench \
  -m models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf \
  -p 512 -n 128 -ngl 99 -r 3

# llmxabe
CUDA_VISIBLE_DEVICES=2 LLMXABE_BENCH_N=1,19,128,512 LLMXABE_BENCH_REPS=10 \
  cargo run --release -p xabe-engine --bin bench_forward
```

| | llama.cpp | llmxabe | llmxabe's position |
| --- | ---: | ---: | --- |
| Prefill, 512 tokens — a full forward over a batch | `pp512` **2,051.30 ± 168.24 tok/s** | n = 512, **73.48 ± 0.23 tok/s** | **27.9× slower** |
| Decode | `tg128` **104.72 ± 0.36 tok/s**, 9.55 ms/token | *no KV cache at the time — see "Decode, measured" below* | — |
| Cost of one token's worth of work, best case | 9.55 ms (a warm decode step) | 32.61 ms (a cold n = 1 pass) | **3.4× slower**, and the comparison flatters llmxabe |

llmxabe's full sweep, 2 warmup passes discarded, 10 timed repetitions, stream
synchronized inside the timed region:

| tokens | ms/pass | tok/s |
| ---: | ---: | ---: |
| 1 | 34.31 ± 3.10 | 29.14 ± 2.63 |
| 19 | 485.33 ± 0.60 | 39.15 ± 0.05 |
| 128 | 2,020.89 ± 9.03 | 63.34 ± 0.28 |
| 512 | 6,967.54 ± 22.16 | 73.48 ± 0.23 |

**Read the n = 1 row as a latency floor, not a decode rate.** llmxabe carries
no KV cache and no recurrent state between calls, so every pass is a cold full
forward; the module docs on `bench_forward` explain why comparing it to
llama.cpp's `tg` would be comparing a cold pass against a warm incremental one.
The n = 1 row is what one pass costs, and a real decode step cannot be faster
than the work it shares with it. Against llama.cpp's *actual* decode it is
3.4× behind, and that is the charitable reading.

The ± 3.10 ms on the n = 1 row is host jitter, not GPU variance: the same
configuration measured by CUDA events gives 32.61 ms with a 0.18 ms standard
deviation, and the uninstrumented wall clock reaches 32.26 ± 0.01 ms once the
host settles.

**llmxabe loses to llama.cpp on every axis measured.** 27.9× at prefill, 3.4×
at the decode floor. On the card's 672 GB/s streaming roofline, llama.cpp's
decode moves the necessary weight bytes at **47.4% of peak** and llmxabe's
n = 1 pass at **13.9%** — a figure derived below from the tensor sizes, and one
that lands within a percentage point of the independent two-term fit earlier in
this document (299 GB/s, 44.5%).

## Head to head, after the MoE tiling

The grouped GEMM and the shared expert were tiled on 2026-08-16
(`perf(moe): tile the grouped GEMM and the shared expert`). Re-measured:

| | llama.cpp | llmxabe | position | was |
| --- | ---: | ---: | --- | ---: |
| Prefill, 512 tokens | `pp512` **2,070.50 ± 160.35 tok/s** | n = 512, **200.87 ± 0.75 tok/s** | **10.3× slower** | 29.6× |
| Cost of one token's worth of work, best case | 9.55 ms (warm decode step) | 18.16 ms (cold n = 1 pass) | **1.90× slower** | 3.9× |

| tokens | ms/pass | tok/s | before | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 18.16 ± 0.01 | 55.07 ± 0.04 | 26.80 | 2.05× |
| 19 | 123.01 ± 0.10 | 154.46 ± 0.12 | 37.16 | 4.16× |
| 128 | 665.18 ± 1.53 | 192.43 ± 0.44 | 60.53 | 3.18× |
| 512 | 2,548.90 ± 9.50 | 200.87 ± 0.75 | 69.84 | 2.88× |

Peak VRAM 31.002 GiB of 47.27 GiB.

**Two caveats on comparing these tables.** This sweep ran on **GPU 0** with
`LLMXABE_BENCH_REPS=5`; the superseded one ran on GPU 2 with 10 repetitions.
The cards are identical models on the same host and the run-to-run spread is
under 0.4% at every batch size, so the comparison is sound, but it is not a
same-configuration A/B. The "before" column is the GPU-0, 5-repetition
baseline measured immediately prior to the change on the same card, not the
GPU-2 numbers in the superseded table — which is why it reads 69.84 rather
than 73.48.

**The n = 1 row is a latency floor, not a decode rate** — it was measured
before there was a KV cache. Real decode is measured in the next section, and
it turns out the floor was the *pessimistic* proxy, not the flattering one.

## Decode, measured (2026-08-16)

`tests/decode.rs` landed the KV cache and the carried recurrent state, so
there is now a number that compares to `llama-bench -n` rather than a proxy.
From `bench_decode`, greedy argmax fed back in, 4 warmup steps discarded:

```sh
CUDA_VISIBLE_DEVICES=0 ./target/release/bench_decode 128 64
```

| | llama.cpp | llmxabe | position |
| --- | ---: | ---: | --- |
| Decode, ~128-token context | `tg128` **104.72 ± 0.36 tok/s**, 9.55 ms/token | **65.25 tok/s**, 15.33 ± 0.10 ms/token | **1.61× slower** |

That is the first honest decode comparison this project has had. The old
"1.90× at the decode floor" line used a cold `n = 1` pass as a stand-in, and
it was **too harsh**: a real decode step does not re-zero 30 recurrent states
and reaches 15.33 ms where the cold pass needed 18.16 ms.

Per-step latency is tight — sd 0.10 ms, p99 15.56 ms — so the mean is
meaningful rather than an average over a bimodal distribution.

### Decode gets slower with context, and by how much

Over 2,000 steps from a 128-token prompt, context 132 → 2,132:

| | ms/step | tok/s |
| --- | ---: | ---: |
| first 10% (context ~132) | 15.56 | 64.3 |
| last 10% (context ~2,132) | 19.79 | 50.5 |
| mean over the whole run | 17.73 ± 1.37 | 56.4 |

**+27.2% over 2,000 tokens**, and that slope is the attention kernel reading
a growing cache. The arithmetic says the slope is far worse than it should
be:

```text
KV read per step at ctx 2,132
  = 10 attention layers x 2,132 positions x 2 kv_heads x 256 x 4 B x 2 (K,V)
  = 83 MiB
observed cost of the growth   4.23 ms   (19.79 - 15.56)
implied bandwidth             19.6 GB/s     = 2.9% of the card's 672 GB/s
```

**The cause is parallelism, not bandwidth.** The flash kernel's grid is
`(n_query, q_heads)`, so at `n_query = 1` it launches **16 blocks on a
72-SM card** — 56 SMs idle, and each surviving block streams its whole KV
window sequentially. The kernel is latency-bound at decode shape, and the
`BM = 1` tiling that is correct for prefill is the wrong shape here.

The standard fix is to split the KV window across blocks and combine the
partial softmaxes — flash-decoding, which is what vLLM's paged attention and
llama.cpp's parallel-block `fattn` path both do. That is a decode-specific
kernel, not a re-tiling of the existing one, and it is the largest identified
win left on the decode path. It is **not yet implemented**, and the 65.25
tok/s above is the number without it.

### The MoE path in isolation

From `bench_moe`, per layer, on a Quadro RTX 8000 (before → after, ms):

| tokens | route | dispatch | grouped GEMM | shared | MoE total | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.022 → 0.022 | 0.011 → 0.016 | 0.774 → **0.174** | 0.034 → 0.035 | 0.841 → 0.247 | **3.4×** |
| 19 | 0.017 → 0.022 | 0.015 → 0.016 | 10.482 → **2.036** | 0.410 → **0.071** | 10.920 → 2.150 | **5.1×** |
| 128 | 0.018 → 0.018 | 0.053 → 0.017 | 36.545 → **5.357** | 2.746 → **0.304** | 39.360 → 5.700 | **6.9×** |
| 512 | 0.033 → 0.033 | 0.185 → 0.035 | 110.932 → **11.895** | 11.233 → **1.132** | 122.400 → 13.100 | **9.3×** |

The shared expert is 9.9× on its own at 512 tokens. It had been re-reading
its entire 2.7 MiB weight stack once per token, a defect the pre-tiling
analysis in this document never named — finding 3 below, which diagnosed the
routed GEMM correctly, does not mention the shared expert at all.

Routing and dispatch are, as the earlier profile said, negligible: the
dispatch parallelization is worth 0.12% of runtime. It was done because it
was cheap, not because it mattered.

### Correctness across the change

Not "the tests still pass". The golden-data forward pass yields the same
argmax token **25358 (' Tokyo')** at logit **19.936268** against
**19.936270** before — a 2e-6 move, consistent with reassociation in the dot
product and nothing else. Dispatch tables are bit-exact including their
padding sentinels, and expert ids are exact on all 37 tokens × top-8. No
tolerance was loosened. Full workspace: 35 test binaries, 445 passed, 0
failed, 0 ignored.

### Against the roofline, after

13.3% of fp32 peak at 512 tokens, against the 25% the plan targeted. The
shortfall could not be attributed: `ncu` cannot read counters on this host
(`ERR_NVGPUCTRPERM`), so the analysis behind it is `nsys`, an offline SASS
dump, and ablation only. The SASS inner loop is **511 instructions for 128
FFMA — 25% instruction density** — which puts roughly **50% of fp32 peak** as
the structural ceiling for this instruction mix. 13.3% is therefore about
half of what this kernel shape can reach, not half of what the card can.

### What did not work, with numbers

Recorded so it is not re-attempted:

| Attempt | Effect |
| --- | --- |
| `LDS.128` vectorization of weight loads, alone | 2% |
| Staging activations without tiling | 2.5% |
| Ablating **all** weight loads and dequant entirely | 6% |
| Cutting shared-memory loads from 16 to 1 per iteration | **slower** |
| `MOE_ROWS = 16` instead of 8 | worse at every batch size; reverted |
| Tile rounding `bm > 2 -> CALL(4)` | 24% faster and **wrong** — drops rows for tiles with 5–8 live slots; `moe_differential` caught it on token 36 of 37; discarded |

The 6% ablation row is the load-bearing one: it is what showed the cost was
neither DRAM bandwidth nor dequant ALU, and redirected the work to weight
reuse.

## How the breakdown was measured

**Which code this is.** Every llmxabe figure in this section was measured
against commit **`89a38ba`** ("the full forward pass reproduces llama.cpp's
token"), built `--release`, in a clean `git worktree` so that a concurrent
workstream editing `crates/xabe-cuda/src/kernels/moe.rs` could not move the
numbers mid-run. That precaution turned out to be necessary: a rebuild against
the working tree part-way through this exercise already showed a materially
faster MoE. **These numbers are a snapshot of `89a38ba` and will go stale.**
The method below is the part meant to outlive them — re-run `profile_forward`
rather than trusting the tables.

`crates/xabe-engine/src/bin/profile_forward.rs` partitions one pass into 84
consecutive, non-overlapping spans: the state reset, the embedding gather, a
mixer and a MoE for each of the 40 blocks, the final norm, and the LM head.
They sum to the pass, which is what makes the percentage column arithmetic
rather than rhetoric.

**CUDA events, not `Instant`.** Every launch in the pass is asynchronous, so a
host clock read between two stages measures enqueue latency and not work.
Timing 84 stages with `Instant` would need 84 `cuStreamSynchronize` calls per
pass, draining the pipeline every time. `cuEventRecord` is a stream-ordered
marker: the enqueue is cheap and the host reads the whole timeline once, after
the pass.

The instrumentation is **off by default**. `Forward::run` costs one
always-false branch per stage boundary unless `enable_profiling` was called, so
`bench_forward`'s figure remains the uninstrumented one.

**Instrumentation overhead, measured rather than asserted.** `profile_forward`
brackets the instrumented run with two uninstrumented ones:

| n | events off (before) | events on | events off (after) |
| ---: | ---: | ---: | ---: |
| 1 | 40.52 ± 3.70 ms | 32.62 ± 0.18 ms | 32.26 ± 0.01 ms |
| 512 | 6,833.18 ± 53.70 ms | 6,938.52 ± 19.26 ms | 6,967.73 ± 5.00 ms |

At n = 1 the overhead is **+0.37 ms, +1.1%**. At n = 512 the instrumented run
falls *between* the two uninstrumented brackets, so the overhead (+105 ms
nominal, +1.5%) is **not distinguishable from run-to-run drift**. Neither
figure changes any conclusion below, all of which turn on factors of 10 or
more.

A second instrument, `nsys profile -t cuda`, gives the per-kernel composition
inside each stage. Its inflation was measured too: at n = 512 it reports
7,096.55 ms/pass against 6,967.54 uninstrumented (**+1.9%**), so its absolute
numbers are usable; at n = 1 it reports 44.32 ms against 34.31 (**+29%**),
because the pass is 1,053 launches of mostly-small kernels and CUPTI's
per-launch cost is not small relative to them. **The n = 1 kernel figures below
are therefore rescaled to the CUDA-event total and should be read as shares,
not as absolutes.**

`ncu` was **not** usable: this host has
`NVreg_RestrictProfilingToAdminUsers` set and the run fails with
`ERR_NVGPUCTRPERM`. So there are no measured DRAM-traffic or achieved-FLOP
hardware counters here. Every bandwidth and FLOP figure below is *necessary
work divided by measured time* — a lower bound on what the kernel really moved,
which is the conservative direction for every claim made from it.

## Per-stage breakdown

```sh
CUDA_VISIBLE_DEVICES=2 LLMXABE_PROFILE_N=1,512 LLMXABE_PROFILE_REPS=5 \
  cargo run --release -p xabe-engine --bin profile_forward
```

### n = 1 — 32.61 ms of GPU time

| stage | ms total | ms each | % of pass | sd |
| --- | ---: | ---: | ---: | ---: |
| reset (30 GDN state memsets + id upload) | 0.194 | 0.194 | 0.60% | 0.004 |
| embedding gather | 0.006 | 0.006 | 0.02% | 0.000 |
| GDN mixer ×30 | 4.743 | 0.158 | 14.55% | 0.017 |
| Gated Attention mixer ×10 | 0.894 | 0.089 | 2.74% | 0.005 |
| **MoE on GDN layers ×30** | **19.455** | 0.648 | **59.66%** | 0.119 |
| **MoE on attention layers ×10** | **6.410** | 0.641 | **19.66%** | 0.037 |
| final RMSNorm | 0.005 | 0.005 | 0.02% | 0.001 |
| LM head (1 position) | 0.904 | 0.904 | 2.77% | 0.001 |
| **total** | **32.611** | | 100.00% | |

**The MoE is 79.3% of the pass.** Everything else together is 20.7%.

### n = 512 — 6,938.54 ms of GPU time

| stage | ms total | ms each | % of pass | sd |
| --- | ---: | ---: | ---: | ---: |
| reset (30 GDN state memsets + id upload) | 0.261 | 0.261 | 0.00% | 0.076 |
| embedding gather | 0.026 | 0.026 | 0.00% | 0.005 |
| **GDN mixer ×30** | **1,782.98** | 59.433 | **25.70%** | 3.976 |
| Gated Attention mixer ×10 | 157.66 | 15.766 | 2.27% | 0.240 |
| **MoE on GDN layers ×30** | **3,792.71** | 126.424 | **54.66%** | 11.591 |
| **MoE on attention layers ×10** | **1,203.97** | 120.397 | **17.35%** | 3.551 |
| final RMSNorm | 0.023 | 0.023 | 0.00% | 0.000 |
| LM head (1 position) | 0.910 | 0.910 | 0.01% | 0.001 |
| **total** | **6,938.54** | | 100.00% | |

**The MoE is 72.0% and the GDN mixer 25.7%.** Attention, the embedding, the
final norm and the LM head together are 2.3%.

The MoE costs the same on both layer kinds (126.4 ms vs 120.4 ms), which is
what it should do — the MoE does not know what mixer preceded it. That
agreement is a check on the instrumentation, not a finding.

### Inside the stages: per-kernel, from `nsys`

n = 512, rescaled from the nsys total (7,055.95 ms) to the CUDA-event total:

| kernel | % of pass | ms/pass | launches/pass | grid | block |
| --- | ---: | ---: | ---: | --- | --- |
| `moe_expert_ffn` | **38.75%** | 2,688.9 | 40 | (512, 7936, 1) | (256,1,1) |
| `moe_expert_down` | **28.65%** | 1,988.1 | 40 | (2048, 7936, 1) | (256,1,1) |
| `gdn_proj_q8_0` | **20.80%** | 1,443.1 | 90 | (2048\|1024\|512, 512, 1) | (32,4,1) |
| `moe_shared_down` | 2.34% | 162.1 | 40 | (2048, 512, 1) | (256,1,1) |
| `moe_shared_ffn` | 2.28% | 158.1 | 40 | (512, 512, 1) | (256,1,1) |
| `gdn_chunk_inter` | 2.11% | 146.5 | 240 | (64, 32, 1) | (128,1,1) |
| `lm_head_gemv_b8` (attention projections) | 1.80% | 124.8 | 2,560 | (1024\|256\|64, 1, 1) | (32,8,1) |
| `gdn_chunk_solve_and_apply` | 1.18% | 81.6 | 240 | (32, 1, 1) | (128,1,1) |
| `moe_block_router_logits` | 0.70% | 48.6 | 40 | (256, 512, 1) | (256,1,1) |
| `gdn_chunk_gram` | 0.54% | 37.8 | 240 | (64, 16, 1) | (64,1,1) |
| `attn_flash_causal` | **0.33%** | 22.9 | 10 | (512, 16, 1) | (256,1,1) |
| `moe_align_block_size` | **0.12%** | 8.6 | 40 | **(1, 1, 1)** | (256,1,1) |
| everything else (19 kernels) | 0.32% | 22.3 | 583 | | |

n = 1, shares only (see the inflation note above):

| kernel | % of pass | ms/pass (rescaled) | grid |
| --- | ---: | ---: | --- |
| `moe_expert_down` | **48.87%** | 15.94 | (2048, 128, 1) |
| `moe_expert_ffn` | **27.94%** | 9.11 | (512, 128, 1) |
| `gdn_proj_q8_0` | 8.70% | 2.84 | (2048\|1024\|512, 1, 1) |
| `lm_head_gemv_b1` | 3.60% | 1.17 | (31040, 1, 1) |
| `moe_shared_ffn` | 1.60% | 0.52 | (512, 1, 1) |
| `moe_route` | 1.42% | 0.46 | **(1, 1, 1)** |
| `moe_shared_down` | 1.25% | 0.41 | (2048, 1, 1) |
| `gdn_recurrent_step` | 1.07% | 0.35 | (128, 32, 1) |
| `rms_norm_rows` | 0.90% | 0.29 | **(1, 1, 1)** |
| `moe_block_router_logits` | 0.60% | 0.20 | (256, 1, 1) |
| `moe_reduce` | 0.54% | 0.18 | **(1, 1, 1)** |
| `moe_align_block_size` | 0.47% | 0.15 | **(1, 1, 1)** |
| `attn_flash_causal` | **0.08%** | 0.03 | (1, 16, 1) |

**Two kernels — `moe_expert_ffn` and `moe_expert_down` — are 67.4% of the pass
at n = 512 and 76.8% at n = 1.** A third, `gdn_proj_q8_0`, brings it to 88.2%
and 85.5%. Everything else in this engine is noise by comparison.

## Against the rooflines

The card is 672 GB/s and 16.3 TFLOP/s fp32. Every projection in this pass is a
matvec against a dequantized weight matrix, so its arithmetic is exactly
`2 × elements` FLOP per activation row and its necessary traffic is exactly the
tensor's stored size per distinct matrix touched. Both come from the GGUF
directory — `profile_forward` prints the per-tensor table it sums, so the model
is auditable against the file rather than being a shape guess.

### n = 1 — bandwidth-bound, and nowhere near the bandwidth

| stage | necessary bytes | achieved GB/s | % of 672 GB/s | ideal ms |
| --- | ---: | ---: | ---: | ---: |
| GDN projections ×30 | 1,089,477,120 | 229.68 | 34.18% | 1.621 |
| Gated Attention projections ×10 | 289,771,520 | 324.30 | 48.26% | 0.431 |
| **MoE ×40** | 1,125,253,120 | **43.51** | **6.47%** | 1.674 |
| LM head | 540,344,320 | **597.93** | **88.98%** | 0.804 |

Whole pass: **3.045 GB of necessary traffic in 32.61 ms = 93.4 GB/s = 13.9% of
peak.** A perfect implementation of the same arithmetic would take 4.53 ms —
**221 tok/s**, which reproduces the project's own 235 tok/s roofline to within
6%, from the tensor sizes rather than by assumption. llama.cpp's measured
9.55 ms/token moves the same bytes at 318.9 GB/s, **47.4% of peak** — itself
within a percentage point of the 299 GB/s two-term fit earlier in this
document, which is a useful cross-check on both.

The LM head is the outlier in the good direction: **88.98% of streaming peak**,
almost nothing left on the table.

### n = 512 — compute-bound, and nowhere near the compute

| stage | FLOP | achieved GFLOP/s | % of 16.3 TFLOP/s | ideal ms |
| --- | ---: | ---: | ---: | ---: |
| GDN projections ×30 | 1,030.8 G | 578.13 | **3.55%** | 63.24 |
| Gated Attention projections ×10 | 279.2 G | 1,770.70 | **10.86%** | 17.13 |
| MoE ×40 | 1,181.1 G | 236.38 | **1.45%** | 72.46 |
| LM head | 1.0 G | 1,117.41 | 6.86% | 0.06 |
| **all matmuls** | **2,491.1 G** | **357.5** | **2.19%** | **152.83** |

llama.cpp does the same 2,491 GFLOP in 249.6 ms — **9.98 TFLOP/s, 61.2% of the
card's fp32 peak.** It clears 60% of fp32 peak because it is not spending fp32:
its `mmq` path quantizes activations to `q8_1` and dots in int8 on the tensor
cores, whose published peak on this card is 130.5 TOPS — **8× the fp32 peak**.
Against *that* denominator llama.cpp is at 7.6%, an ordinary number for a GEMM.
Against fp32 it looks superhuman only because it is using a different unit.
(130.5 TOPS is the datasheet figure for this card, not something measured
here.)

**That is the single most important structural fact in this section.** llmxabe
dequantizes to fp32 and dots in fp32, so its ceiling is 16.3 TFLOP/s. fp32 is
not disqualifying by itself — a *flawless* fp32 pass would need 152.83 ms of
matmul against llama.cpp's whole 249.6 ms. But a realistic well-tiled fp32 SIMT
GEMM reaches perhaps 25% of peak, which is 611 ms of matmul and, with the
non-matmul stages left as they are, about 980 ms for the pass: **3.9× behind
llama.cpp**, as the ranked list below works out independently. Matching the
baseline at prefill requires int8 tensor cores (`mma.m8n8k16.s8`, available on
sm_75), not merely better tiling.

## The suspected causes, confirmed and refuted

Six causes were suspected before any of this was measured. **Four are
essentially refuted, one is confirmed and dominant, one is confirmed but
unquantifiable on this host.**

### 1. "Attention is the scalar fp32 path, `BM = 1`, no `m16n8k8` tensor cores" — REFUTED as a cost

The description is accurate: `attn_flash_causal` launches grid `(512, 16, 1)`,
one query row per block, `BM = 1`, and it is fp32 throughout.

It is also **0.33% of the pass at n = 512 and 0.08% at n = 1**. Deleting the
attention kernel entirely — making it free — would recover **22.9 ms of 6,938**
and **0.03 ms of 32.6**.

The whole Gated Attention mixer is only 2.27% of the pass, and 79% of *that* is
its projections (`lm_head_gemv_b8`, 124.8 ms) rather than the attention maths
(22.9 ms). Attention is not where llmxabe's time goes.

**Limit on this conclusion:** this is a 512-token self-contained window with no
KV cache. Attention cost grows quadratically with context while everything else
grows linearly, so at 32K this conclusion would flip. It is a statement about
*this* benchmark, not about the engine at length.

### 2. "The MoE dispatch kernel is single-block (one SM of 72)" — CONFIRMED as a fact, REFUTED as a cost

`moe_align_block_size` does launch with grid `(1, 1, 1)`, at both sizes. At
n = 1, seven more kernels degenerate to a single block because there is only
one token to spread over: `moe_route`, `moe_reduce`, `moe_block_combine`,
`moe_block_shared_gate`, `rms_norm_rows`, `gdn_gates`, `fwd_embed_q8_0`.

Every kernel whose grid is exactly one block, summed:

| | n = 1 | n = 512 |
| --- | ---: | ---: |
| single-block kernels | 8 | 1 (`moe_align_block_size` only) |
| their total cost | **1.32 ms** | **8.58 ms** |
| share of the pass | **4.03%** | **0.12%** |

Parallelizing every one of them perfectly recovers **4.0% at n = 1** and
**0.12% at n = 512**. It is real and it is small. `moe_align_block_size`
specifically — the dispatch kernel the suspicion named — is **0.47% at n = 1
and 0.12% at n = 512**.

### 3. "The MoE grouped GEMM has no tiling or shared-memory reuse; its grid is the full fixed slot capacity, so most blocks early-out" — CONFIRMED, and it is the whole story

Both halves are true, and together they are **67.4% of the pass at n = 512 and
76.8% at n = 1**.

**No tiling.** `moe_expert_ffn` launches one block per `(output row, slot)` and
`moe_expert_down` one per `(output element, slot)`. Each block re-reads the
activation row from global memory, streams its weight row element by element,
and finishes with a full 256-thread block reduction. Nothing is staged in
shared memory and nothing is reused across the 16 slots of a tile. The
consequence is a **15.9× redundant weight read** at n = 512: an expert's
matrices are re-fetched once per assigned token instead of once per tile of 16.
How much of that redundancy L2 absorbs cannot be measured on this host (no
`ncu`), so the true DRAM traffic lies somewhere between the **29.2 GB** the
arithmetic needs and the **464.4 GB** a zero-reuse implementation would move.

**Fixed-capacity grid.** `grid.y` is `sorted_capacity`, a compile-time bound
(AGENTS.md rule 5 requires it, so that the launch shape stays replayable from a
captured graph). Live slots are the rest:

| n | `sorted_capacity` | live `(token, expert)` pairs | blocks doing work | early-out |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 8 | **6.25%** | **93.75%** |
| 512 | 7,936 | 4,096 | **51.61%** | **48.39%** |

At n = 1 the grid is **16× larger than the work in it**; at n = 512 it is 1.94×.

**The sharpest single number in this whole section** is the comparison between
the two expert kernels at n = 512:

| kernel | quant | reduction length | MACs/thread | GFLOP | ms | GFLOP/s | % fp32 peak |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `moe_expert_ffn` | q6_K | 2,048 | 8 | 687.2 | 2,688.9 | 255.6 | 1.57% |
| `moe_expert_down` | q8_0 | 512 | 2 | 343.6 | 1,988.1 | 172.8 | 1.06% |

`moe_expert_down` does **half** the arithmetic of `moe_expert_ffn` and takes
**74%** as long — it is **1.48× less efficient per FLOP** — while using the
*cheaper* quantization format. The difference between them is the reduction
length: 2 MACs per thread before an 8-step block reduction, against 8. That is
the cost of no tiling, isolated, in one comparison.

### 4. "Per-element Q6_K/Q8_0 dequant re-reads the superblock header for every element" — CONFIRMED as a fact, but bounded small

It is true: `dequant_element(gate_q, gate_quant, base + j)` is called inside the
innermost loop, per element, and recomputes the superblock offset each time.

It cannot be isolated by measurement here — doing so means editing
`crates/xabe-cuda/src/kernels/moe.rs`, which this workstream does not own, and
`ncu`'s instruction counters are unavailable. But row 3's table **bounds it**:
the q6_K kernel, whose per-element unpack is far more expensive (a 12-byte
scale array plus two bit planes, against q8_0's single scale and one byte), is
the *more* efficient of the two per FLOP. If header re-reads were the dominant
term, the ordering would be the other way round.

**Conclusion: real, worth fixing, but not the reason the MoE is 72% of the
n = 512 pass and 79% of the n = 1 pass.** The reduction shape is.

### 5. "The LM head computes logits for ALL positions; llama.cpp's pp computes only the last" — REFUTED outright

It does not. `Forward::run` already does what llama.cpp does — it copies the
last position's row out of the final norm and runs the head on one row, exactly
mirroring `get_rows(cur, inp_out_ids)`. `forward.rs` has a test asserting it
(`the_lm_head_runs_on_one_position_because_that_is_what_was_captured`).

The measurement confirms it: **`lm_head_gemv_b1` launches once per pass**, and
the LM head stage costs **0.904 ms at n = 1 and 0.910 ms at n = 512** — the
same, because it does the same work either way. That is 2.77% of the n = 1 pass
and **0.01%** of the n = 512 pass, at **88.98% of streaming peak**.

The LM head is the best-optimized thing in this engine. There is nothing to
recover here.

### 6. "40 layers × many small launches, no fusion, no CUDA graph" — REFUTED

**Launch overhead is already fully hidden.** At n = 1 the pass issues **1,053
kernel launches** and the uninstrumented wall clock settles at 32.26 ± 0.01 ms
against 32.61 ms of CUDA-event GPU time — the wall clock is *below* the
instrumented GPU total, the difference being the event markers themselves. The
exposed launch overhead is **not distinguishable from zero**. At n = 512 there
are 4,263 launches and the gap is ~0.4%.

The reason is unflattering: the kernels are slow enough to hide their own
launches. `moe_expert_down` at n = 1 launches 262,144 blocks; the CPU has
plenty of time to enqueue the next kernel. **CUDA graph capture would recover
approximately nothing at these shapes** — and would only start to matter after
the kernels got roughly 10× faster, at which point it should be re-measured.

**Fusion of the elementwise glue is also small.** Every elementwise, norm,
rope, conv and reduction kernel together (`gdn_silu`, `swiglu_mul`, `gdn_add`,
`moe_block_combine`, `rms_norm_rows`, `attn_residual_add`, `moe_route`,
`moe_reduce`, and eleven more) is **21.4 ms, 0.31% of the pass at n = 512** —
and 1.71 ms, 5.25%, at n = 1. Perfect fusion recovers under a third of a
percent where the pass is slowest.

This one matters beyond itself: **milestone 06, CUDA graph capture, was
designated "the justification gate" and "the largest single expected win".**
The section above already showed llama.cpp has it. This measurement shows that
in llmxabe, at the shapes it runs today, it would buy nothing at all.

## Ranked by measured headroom

Ordered by how many milliseconds each would recover **if perfect**, not by how
appealing it sounds. "Realistic" assumes a well-tiled fp32 SIMT GEMM at 25% of
the card's fp32 peak, which is an ordinary result for such a kernel and roughly
7× better than what these kernels do now.

### At n = 512 (prefill), out of 6,938.54 ms

| # | Change | Now | Perfect | Realistic | Recovers (realistic) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Tile the MoE routed grouped GEMM (`moe_expert_ffn` + `moe_expert_down`) | 4,677 ms | 63 ms | 253 ms | **4,424 ms (63.8%)** |
| 2 | Tile the GDN projections (`gdn_proj_q8_0`) | 1,443 ms | 63 ms | 253 ms | **1,190 ms (17.2%)** |
| 3 | Tile the MoE shared expert (`moe_shared_ffn` + `moe_shared_down`) | 320 ms | 8 ms | 32 ms | **288 ms (4.2%)** |
| 4 | The GDN chunked delta rule (`gdn_chunk_*`) | 266 ms | — | — | ≤ 266 ms (3.8%) |
| 5 | Tile the attention projections (`lm_head_gemv_b8`) | 125 ms | 17 ms | 68 ms | **57 ms (0.8%)** |
| 6 | Parallelize `moe_block_router_logits` | 48.6 ms | — | — | ≤ 48.6 ms (0.70%) |
| 7 | Tensor-core / tiled flash attention | 22.9 ms | — | — | ≤ 22.9 ms (0.33%) |
| 8 | Fuse the elementwise glue | 21.4 ms | — | — | ≤ 21.4 ms (0.31%) |
| 9 | Parallelize `moe_align_block_size` (the dispatch kernel) | 8.6 ms | — | — | ≤ 8.6 ms (0.12%) |
| 10 | CUDA graph capture | ~29 ms exposed | — | — | **≤ 29 ms (0.42%)** |
| 11 | LM head | 0.91 ms | 0.80 ms | 0.80 ms | **0.11 ms (0.002%)** |

Items 1–3 are **92.8% of the pass** and recover **85.1% of it**, and they are
one change: *stage a tile of the weight matrix in shared memory and reuse it
across a tile of tokens.* Items 6–11 together are **1.9%**.

Doing 1, 2, 3 and 5 realistically gives ~980 ms → **522 tok/s**, still **3.9×
behind llama.cpp's 2,051**. Closing the rest needs int8 tensor cores.

### At n = 1 (latency floor), out of 32.61 ms

At one token nothing is compute-bound; the binding roofline is bandwidth.

| # | Change | Now | Perfect | Recovers |
| ---: | --- | ---: | ---: | ---: |
| 1 | MoE: pack the grid to live slots and tile the GEMM | 25.87 ms | 1.67 ms | **24.20 ms (74.2%)** |
| 2 | GDN projections | 4.74 ms | 1.62 ms | **3.12 ms (9.6%)** |
| 3 | Parallelize the 8 single-block kernels | 1.32 ms | — | ≤ 1.32 ms (4.0%) |
| 4 | Gated Attention mixer | 0.89 ms | 0.43 ms | **0.46 ms (1.4%)** |
| 5 | Skip the 30 GDN state memsets (a decode would not do them) | 0.19 ms | 0 | **0.19 ms (0.6%)** |
| 6 | LM head | 0.90 ms | 0.80 ms | **0.10 ms (0.3%)** |
| 7 | CUDA graph capture | ~0 ms exposed | 0 | **~0 ms** |

Item 1 alone is three quarters of the pass. Its mechanism at n = 1 is the
**93.75% early-out**: 128 grid slots for 8 live pairs.

Doing 1, 2 and 4 gives ~4.8 ms → **207 tok/s**, which would put llmxabe's
latency floor **ahead** of llama.cpp's 104.72 tok/s decode. That is the
optimistic reading and it should be treated with suspicion until a KV cache
exists: the n = 1 pass has no cache to read, and llama.cpp's `tg128` does. It
is nonetheless the clearest evidence that the decode-side gap is a kernel
problem and not an architectural one.

## What this changes

**The engine's problem is one kernel shape, not six things.** Every suspected
cause that could be isolated is worth under 3% of the pass. The one that could
not be isolated — the per-element dequant — is bounded small by the q6_K/q8_0
comparison. The remaining one, no tiling and no shared-memory reuse in the
grouped GEMM, is worth **64% at prefill and 74% at the latency floor**. Every
hour spent on tensor-core attention, dispatch parallelism, dequant
micro-optimization or CUDA graph capture before that one is fixed is an hour
spent on the 3%.

**Prefill cannot be won in fp32.** llama.cpp runs the same 2,491 GFLOP at 61%
of this card's fp32 peak because it is not spending fp32 — it dots in int8 on
the tensor cores, which are 8× faster. A perfectly tiled fp32 llmxabe lands
around 3.9× behind. This is a stated design consequence, not a tuning gap, and
it needs a decision rather than more optimization.

**The decode-side story is better than the prefill one**, and it is the side
the project cares about. The n = 1 pass is bandwidth-bound at 13.9% of peak
against llama.cpp's 47.4%, and the gap is the same MoE grouped GEMM. Fixing it
is bounded, local, and does not require changing the arithmetic.

**Milestone 06's premise is now measured and does not hold.** Graph capture
buys ~0.4% at n = 512 and nothing measurable at n = 1, because launch overhead
is already hidden behind slow kernels. The earlier finding was that llama.cpp
already has graph capture; this one is that llmxabe would not benefit from it
yet. The continue/stop gate needs restating around the grouped GEMM.

## Not measured, in this section

- **Any hardware counter.** `ncu` fails with `ERR_NVGPUCTRPERM` on this host, so
  there is no measured DRAM traffic, L2 hit rate, occupancy, or instruction mix.
  Every efficiency figure here is necessary-work ÷ measured-time, which
  understates real traffic and therefore understates how far off peak the
  kernels are.
- **How much of the grouped GEMM's 16× redundant weight read L2 absorbs.** The
  true DRAM traffic is bounded between 29.2 GB and 464.4 GB per pass at n = 512
  and was not narrowed.
- **The dequant header re-read in isolation.** Bounded by inference from the
  q6_K/q8_0 comparison, not measured directly; measuring it means editing a
  kernel this workstream does not own.
- **Anything at long context.** The largest *prefill* measured is 512 tokens.
  The conclusion that attention is negligible is a statement about that shape
  and does not survive decode: at a 2,132-token context the KV read is 27% of
  a decode step and climbing. See "Decode, measured".
- **Decode past ~2K context.** Measured to 2,132 positions. The +27.2% slope
  over that range is real and unresolved; nothing here says where it goes at
  32K.
- **Multi-GPU.** Every figure is one card.
- The GDN chunked delta rule kernels (`gdn_chunk_*`, 3.8% at n = 512) were
  timed but not analyzed against a roofline.
- **Anything past `89a38ba`.** A concurrent workstream is rewriting the very
  kernel this section identifies as dominant. The ranked list is a statement
  about that commit; the first thing to do with it is re-measure.

## Reproducing

Raw commands are recorded in the tables above. The decode sweep is:

```sh
llama-bench -m Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf \
  -ngl 99 -sm none -fa on -ctk f16 -ctv f16 -t 4 \
  -p 0 -n 128 -d 0,4096,32768,131072 -r 3
```

## Not measured

- Multi-GPU aggregate throughput under real routed traffic (single-card figures
  extrapolated).
- Whether `--backend-sampling` changes output *quality* over long runs; it was
  verified correct and diverse over short samples only.
- MTP speculative decode acceptance rate against the n-gram baseline.
- Any effect of `-b`/`-ub` tuning.
- The `mmproj` vision encoder's resident VRAM cost.
- Sustained thermal behaviour; all runs were short.
