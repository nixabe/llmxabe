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

## Integer tensor cores: measured, 6.8x the fp32 kernel (2026-08-16)

`mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32` is landed and gated
(`tests/mma_differential.rs`). It is **not yet wired into the engine** — the
numbers below are the kernel measured on its own against the fp32 kernel it
would replace.

From `bench_mma`, the Gated DeltaNet projection's real shapes:

| tokens | rows | k | GGUF layout | split layout | % of 198 TOP/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 8,192 | 2,048 | 2.10 TOP/s | 23.6 | 11.9% |
| 512 | 8,192 | 2,048 | 2.01 | **31.96** | 16.1% |
| 512 | 2,048 | 4,096 | 2.87 | 24.9 | 12.6% |

Against the fp32 tiled kernel's 4.67 TFLOP/s at the same shapes, the split
layout is **6.8x**.

### Two things had to be fixed before the tensor cores did anything

**Reading GGUF Q8_0 in place is fatal.** A Q8_0 block is 34 bytes — a 2-byte
scale then 32 quants — so a fragment's four bytes are *never* word-aligned. A
`*(const unsigned int*)` load faults outright with
`CUDA_ERROR_MISALIGNED_ADDRESS`, and assembling the fragment from four scalar
byte loads instead costs ~48 memory instructions per 8 MMA instructions. That
version measured **2.0 TOP/s, 1% of peak — slower than the fp32 kernel it was
meant to replace.** Splitting the scales out of the quants makes every operand
load aligned and contiguous and is worth **3.2x** on its own. This is why
llama.cpp's MMQ, TurboMind and vLLM's Marlin all carry their own packed weight
layouts instead of reading the on-disk format.

**Arithmetic intensity is the rest of it.** One 8-token MMA tile per warp
gives 8 x 2 = 16 operations per weight byte, against an int8 ridge point of
198e12 / 672e9 = **295 OP/byte** — deeply memory-bound, and it measured 3.2% of
peak. Every extra token tile multiplies the operations per weight byte without
touching the weight traffic at all, because one B fragment feeds every token
tile. Sweeping the warp tile at 512x8192x2048:

| rows x tokens | TOP/s |
| --- | ---: |
| 32 x 8 | 6.4 |
| 32 x 32 | 13.3 |
| 16 x 64 | 12.7 |
| 64 x 32 | 19.0 |
| 32 x 64 | 23.6 |
| **64 x 64** | **31.96** |

At 64x64 the kernel moves 268 MB in 1.288 ms — 208 GB/s, 31% of bandwidth
peak — so it is no longer bandwidth-bound and 16.1% of compute peak is a
*latency* ceiling. More is available; a shared-memory staging pipeline of the
kind Marlin uses is what would reach it.

### What it costs

The activations must be quantized to int8, and that is a real loss the fp32
path does not have. Measured against the fp32 reference: **cosine
0.99999, max relative error 4.3e-3** — the 1/127 quantization floor, not a
formulation defect. It is also precisely the step llama.cpp takes before its
own int8 matmuls, so it is the accuracy llama.cpp already lives with.

The split-layout kernel is separately gated **exactly** against the in-place
one: same numbers, same arithmetic, only rearranged in memory, so any
difference is an indexing defect and no tolerance is allowed to absorb it.

## The Gated DeltaNet projections were 59% of prefill (2026-08-16)

Profiling prefill at 512 tokens after the MoE tiling showed the bottleneck had
**moved**, and the documentation had not:

| kernel | ms/pass | % of a 2,523 ms pass |
| --- | ---: | ---: |
| `gdn_proj_q8_0` | **1,491.6** | **59.1%** |
| `moe_expert_ffn` | 249.8 | 9.9% |
| `moe_expert_down` | 227.1 | 9.0% |
| `gdn_chunk_inter` | 147.7 | 5.9% |
| `lm_head_gemv_b8` | 135.1 | 5.4% |

"The MoE GEMM is 67-77% of every pass" was true before the tiling and is now
false — the MoE is 18.9%. Anything still reasoning from that number is
reasoning from a fixed bug.

**The defect was the same one the MoE had.** `gdn_proj_q8_0`'s grid was
`(N / warps, tokens)` — one warp per *(output row, token)* pair — so the
weight matrix was re-read once per token. At 512 tokens that is 1.07 GiB read
512 times: 548 GB per pass, moved at 367 GB/s. The kernel was running at 55%
of peak bandwidth and doing 512x the necessary work.

Tiling it over tokens — a warp owns one output row and `PROJ_TILE` tokens, and
each dequantized weight element is multiplied into every accumulator before
being dropped — takes that kernel from **1,491.6 ms to 479.3 ms**.

### The tile width has to be specialized

A fixed width is wrong at one end or the other:

| tokens | tile 8 | tile 16 | tile 32 | tile 64 |
| ---: | ---: | ---: | ---: | ---: |
| 19 | **208.80** | 208.57 | 164.52 | 115.85 |
| 128 | 319.15 | 324.89 | **332.28** | 328.05 |
| 512 | 339.94 | 347.20 | 359.18 | **360.12** |

A 32-wide tile costs 21% at 19 tokens. The cause is the guarded path: when
fewer tokens are live than the tile is wide, every thread still carries the
full accumulator array, so the register pressure is paid and the work is not
done. `proj_tile_for` picks the widest fully-live tile, which beats every
fixed width at every batch size.

### The activation was being re-read too, and that was the larger term

Tiling over tokens alone left the kernel at **13.2% of fp32 peak** (2.15
TFLOP/s) against the MoE's 27%. The reason is the other operand: each
dequantized weight element drove `TT` separate loads of `x`, so a warp re-read
the activation column once per row it owned. Across the launch that is ~2.1 GB
per projection call — against a 4 MB activation tensor, so roughly 500x
re-read, served by L2 rather than DRAM but bounded by L2 all the same.

Loading the activation column into registers once and reusing it across `RR`
weight rows divides that traffic by `RR`. The two reuses are complementary:
the token tile amortizes the *weight*, the row band amortizes the
*activation*.

Sweeping (tile, rows) at 512 tokens:

| (tile, rows) | tok/s | accumulators/thread |
| --- | ---: | ---: |
| (32, 2) | 409.42 | 64 |
| **(16, 4)** | **423.30** | 64 |
| (16, 8) | 413.19 | 128 |

(32, 2) and (16, 4) cost identical registers and differ only in how traffic
splits between the two operands — the narrower tile re-reads the weight twice
as often and the activation four times less, and the activation is the larger
term. (16, 8) halves activation traffic again and hands it straight back to
occupancy. So the optimum is interior, and it is not where a
"bigger tile is better" intuition would have put it.

### End to end

| tokens | before | after | speedup |
| ---: | ---: | ---: | ---: |
| 1 | 55.07 | 68.18 | 1.24x |
| 19 | 154.46 | 209.33 | 1.36x |
| 128 | 192.43 | 380.94 | 1.98x |
| 512 | 200.87 | **419.28 ± 2.37** | **2.09x** |

**Prefill against llama.cpp's `pp512` 2,070.50: 10.3x slower -> 4.94x
slower.** Peak VRAM unchanged at 31.03 GiB. *(Superseded: see "Integer tensor
cores, wired end to end" at the end of this document, where it reaches 1.54x.)*

Decode is **unchanged at ~65 tok/s**, and that is expected rather than
disappointing: at one token the weight is already read exactly once, so there
is nothing to tile and the dispatch keeps the untiled kernel. Decode's problem
is streaming efficiency, not redundant reads.

Correctness is unchanged: `forward_pass` still reproduces llama.cpp's argmax
25358 with an identical accumulation curve (111.6x absolute, 3.9x relative
L2), and all three decode paths still agree at cosine 1.000000000.

### What is now the prefill bottleneck

| kernel | ms/pass | % of a 1,490 ms pass |
| --- | ---: | ---: |
| `gdn_proj_q8_0_tiled` | 479.3 | 32.2% |
| `moe_expert_ffn` | 247.0 | 16.6% |
| `moe_expert_down` | 224.9 | 15.1% |
| `gdn_chunk_inter` | 141.1 | 9.5% |
| `lm_head_gemv_b8` | 133.4 | 9.0% |

Still the same kernel, now at a third of the pass rather than three fifths.
The floor is reading the weight *once* per pass — 1.07 GiB, ~1.6 ms at peak
bandwidth — against 479.3 ms today, so tiling has taken 3.1x of a possible
512x and the remaining distance is not more tiling but arithmetic: at 512
tokens this projection is a dense fp32 GEMM, and fp32 is the ceiling
[the int8 MMA work](#the-two-halves-of-the-goal-are-two-different-problems)
exists to break.

## Where decode actually goes, and why prefill needs tensor cores (2026-08-16)

Measured with `nsys` over 200 decode steps at a 128-token prompt, GPU 0.

**Decode step: 15.33 ms wall, 14.44 ms GPU busy, 1,097 kernel launches.**
The 0.89 ms gap is 5.8% — that is the *entire* ceiling for CUDA graph capture
on a single decode stream, and it is worth knowing before building it. The
earlier "~0.4% at n = 512" figure was measured at prefill shape, where a pass
is 40x longer and launch overhead is correspondingly irrelevant.

| | share of GPU time | ms |
| --- | ---: | ---: |
| MoE experts (`moe_expert_ffn`/`_down`, `moe_shared_*`) | 39.2% | 5.67 |
| GDN projections (`gdn_proj_q8_0`/`_f32`) | 30.8% | 4.45 |
| LM head | 8.5% | 1.23 |
| MoE dispatch (route, align, reduce) | 7.2% | 1.04 |
| attention | 2.8% | 0.40 |
| GDN recurrent step | 2.1% | 0.30 |
| norms | 1.8% | 0.26 |
| everything else | 7.6% | 1.10 |

### The two halves of the goal are two different problems

From `bench_moe`, the grouped GEMM against both rooflines:

| tokens | grouped ms | GB/s (unique) | % bandwidth peak | TFLOP/s | % fp32 peak |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.175 | 129.4 | 19.3% | — | — |
| 512 | 11.814 | 61.4 | 9.1% | **4.4** | **27%** |

**At prefill the MoE GEMM is compute-bound, not bandwidth-bound.** 4.4
TFLOP/s is 27% of the card's 16.3 TFLOP/s fp32 peak, and §2 of
[OPTIMIZATION.md](OPTIMIZATION.md) puts the structural ceiling for this
instruction mix at ~50% of peak (511 SASS instructions per 128 FFMA = 25%
instruction density). So the kernel is already at roughly **55% of what fp32
can reach on this part**, and the remaining fp32 headroom is under 2x against
a 10.3x gap. This is an independent confirmation of §8.1's conclusion:
**prefill cannot beat llama.cpp without int8 tensor cores.**

**At decode it is bandwidth-bound**, at 19.3% of peak. That one has real
headroom — `lm_head_gemv_b1` moves 540 MB at 353 GB/s (53% of peak) on the
same card in the same pass, so ~2.5x on the MoE weight path is not a
speculative target.

### Requantizing the experts Q6_K -> Q8_0: measured, and not worth it

The hypothesis was that Q6_K's 210-byte superblock — `ql`, `qh` and `scales`
in disjoint runs — coalesces badly against Q8_0's flat 34-byte block, and
that widening would pay for its extra bytes. `LLMXABE_MOE_QUANT=q8_0` runs
the same geometry with the gate and up projections widened:

| tokens | Q6_K | Q8_0 | speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.175 ms | 0.165 ms | 1.06x |
| 19 | 1.925 | 1.781 | 1.08x |
| 128 | 5.334 | 5.119 | 1.04x |
| 512 | 11.814 | 11.266 | 1.05x |

**4-8%, for +30% VRAM** (the MoE stack goes 27.4 -> 35.4 GiB). The
hypothesis was wrong: the dequantization format is not what limits this
kernel. Recorded so the requantization pipeline does not get built on the
strength of the argument, which is plausible and false.

## NVRTC: 22 of 32 compiles were redundant (2026-08-16)

The `--log-level debug` instrumentation added with the `tracing` migration
logs every NVRTC invocation, which made a build-time cost visible that nobody
had looked for. `LayerOpsKernels::new` takes only a context — it has no
geometry to specialize on — and is constructed independently by `forward.rs`
and by the GDN, attention and MoE blocks, so all four produced byte-identical
PTX. `Forward::reshape` then doubled the whole bill by building a second shape
over the same weights.

Fixed by memoizing `kernels::compile` on its source. Measured with
`bench_decode --log-level debug`, building two shapes (prefill 128 and decode
1) on GPU 0:

| | NVRTC invocations | NVRTC total | build, 2 shapes |
| --- | ---: | ---: | ---: |
| before | 32 | 5,875 ms | 15.5 s |
| after | **10** | **1,633 ms** | **11.1 s** |

**4.24 s of NVRTC removed, 28% off the build.** This is build time, not
inference time: it changes how long `Forward::new` takes and nothing about
tok/s. It is worth recording because the "build s" column in `bench_forward`
is measured and reported, and two thirds of it was duplicate work.

The key is the source string, which is sound because PTX is a pure function of
(source, arch) and arch is a constant. PTX is cached, not modules — a
`CudaModule` belongs to the context that loaded it, and loading still happens
per context, which is what keeps this correct across the three GPUs.

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

## Integer tensor cores, wired end to end (2026-08-16)

The `mma.m8n8k16` primitive measured in isolation above is now the arithmetic
of **every quantized matmul in the model**. This section supersedes the
prefill numbers in every section before it.

### Where prefill stands

| | llama.cpp | llmxabe | position |
| --- | ---: | ---: | --- |
| Prefill, 512 tokens | `pp512` **2,076.2 tok/s** | **2,099.3** | **1.011× faster** |
| Decode, warm | `tg128` **104.72 ± 0.36 tok/s** | **104.8 tok/s** | **level** |

Both rows are three alternating rounds of `llama-bench` and `bench_forward`
on the same card in the same session — see "Level is not the same as ahead"
at the end of this document for why nothing else is comparable.

Decode is treated separately at the end of this document; the sections between
here and there are all prefill.

Prefill started this session at 200.87 tok/s and 10.3× slower.

### The arc, one measurement per change

Every row is `bench_forward` at n = 512 on GPU 0, 2 warmup passes discarded,
5 timed repetitions, stream synchronized inside the timed region.

| change | tok/s | vs previous |
| --- | ---: | ---: |
| baseline | 200.87 | — |
| token tiling in the GDN projection | 362.67 | 1.81× |
| activation reuse in the same kernel | 419.28 | 1.16× |
| int8 MMA on the GDN projections | 513.17 | 1.22× |
| int8 MMA on the Gated Attention projections | 582.46 | 1.14× |
| int8 MMA on the MoE Q6_K gate/up | 679.12 | 1.17× |
| int8 MMA on the MoE Q8_0 down | 847.71 | 1.25× |
| eight tokens per `gdn_chunk_inter` block | 1,008.51 | 1.19× |
| tiled MoE router | 1,075.60 | 1.07× |
| 16-bit MoE weight staging | 1,110.95 | 1.03× |
| GDN state update split into its own kernel | 1,155.20 | 1.04× |
| `float4` GDN row loads | 1,248.27 | 1.08× |
| int8 MMA on the shared expert | 1,341.39 | 1.07× |
| word-wide Q6_K unpacking in the MoE MMA | 1,365.74 | 1.02× |
| wider MoE tensor-core block, 32-slot dispatch | 1,380.38 | 1.01× |
| four staging loads per weight row instead of eight | 1,430.58 | 1.04× |
| a block's warps share one weight band | 1,440.00 | 1.01× |
| pad the staged activation row off a 32-bank stride | 1,630.00 | **1.13×** |
| four keys per warp between attention barriers | 1,644.10 | 1.01× |
| eight tokens at once in the GDN solve's output | 1,708.06 | 1.04× |
| eight experts per router block instead of four | 1,721.84 | 1.02× |
| eight `j` lanes per state-update block instead of four | 1,727.74 | 1.01× |
| sixteen tokens per inter-chunk block instead of eight | 1,735.42 | 1.00× |
| pad the Q6_K device stride to 224 bytes | 1,762.46 | 1.02× |
| the router's activation tile in registers, not shared | 1,823.01 | 1.03× |
| eight query rows per flash block, query tile in registers | 1,921.76 | **1.05×** |
| a 4x4 register tile in the GDN state update | 1,994.56 | 1.04× |
| `float4` staged reads in `gdn_chunk_inter` | 2,009.77 | 1.01× |
| lane-invariant coefficients hoisted out of the GDN solve | 2,068.68 | 1.03× |
| a token band in the alpha/beta gates | 2,074.95 | 1.00× |
| a tensor-core kernel for block 39's Q8_0 gate and up | 2,123.39 | 1.02× |

**10.57× overall.** No single change is more than 1.81×; the result is
compounding, and roughly half of it is not arithmetic at all — it is fixing
kernels that re-read the same bytes.

### What the wins actually were

Only five of the twelve changes are about the tensor cores. The rest are
memory-traffic bugs that the profile made visible once the arithmetic stopped
dominating:

- **`gdn_chunk_inter` re-read the entire recurrent state once per token.** The
  state does not depend on the token index. 134 MB of loads per chunk to cover
  2 MB of distinct data.
- **The MoE router re-read a 2,048-float weight row and activation row per
  (expert, token) pair.** 131,072 blocks at 512 tokens; 2.1 GB of loads per
  layer to cover 6 MB. This was 7.6% of the pass for the *smallest* matmul in
  the layer.
- **`gdn_chunk_solve_and_apply` ran at ~5% occupancy** because two thirds of
  its arithmetic — the chunk-end state update — was sharing a launch shape with
  code parallel over one axis fewer.
- **`gdn_chunk_gram` and `gdn_chunk_inter` read their rows one float at a
  time.** Neither read can be coalesced across a warp by construction, but both
  can be four times wider. That change alone was worth 8% of the pass and
  altered no arithmetic whatsoever.

### The two ways to reach a tensor core from a quantized weight

Both are in the tree, and the choice is VRAM, not preference:

- **Repack** into split quant and scale arrays, so operand loads are aligned
  words. Used for the dense projections (1.13 GiB) and the shared expert
  (3.5 MB per layer). `mma_q8_0_proj_split` measures 27 TOP/s this way against
  the 2.1 TOP/s the same kernel gets assembling operands byte by byte in place.
- **Stage through shared memory**, where the kernel picks the layout and can
  put the quants on a word boundary itself. Used for the routed experts, whose
  10.7 G weights cannot afford a second copy beside the model.

The first attempt at the routed-expert kernel did neither and read operands
straight from global. It was **1.5× slower than the fp32 kernel it replaced**:
an MMA B fragment wants eight different weight rows per warp, and reading those
from global put consecutive lanes 1,680 bytes apart, so every 4-byte operand
cost a full 32-byte sector.

### Where the time goes now

`nsys` kernel summary over a full `bench_forward` sweep, so it mixes batch
sizes; read it as a ranking, not as per-batch shares.

| kernel | % of GPU time |
| --- | ---: |
| `moe_expert_ffn_mma` | 25.6% |
| `moe_expert_down_mma` | 15.8% |
| `mma_q8_0_proj_split` | 10.5% |
| `gdn_chunk_solve_and_apply` | 7.5% |
| shared expert (two kernels, now replaced) | 9.5% |
| `attn_flash_causal` | 4.0% |
| `gdn_proj_q8_0_t16` | 4.0% |
| `gdn_chunk_state_update` | 3.7% |
| `moe_block_router_logits` | 2.6% |
| `gdn_chunk_inter` | 2.5% |
| `gdn_chunk_gram` | 2.1% |

The MoE expert GEMMs are 41% between them and are the next thing to look at.
`mma_q8_0_proj_split` is at 13.6% of the card's 198 TOP/s int8 peak, so the
primitive itself has room that has not been touched.

### Correctness

Every step above holds `tests/forward_pass.rs`: argmax 25358 (' Tokyo'), the
same token llama.cpp decodes. Four of the changes are **bit-identical** — the
gate reports logit 20.106327 to every digit across the router tiling, the state
update split, and the `float4` widening.

`tests/int8_forward.rs` runs 128 tokens through the model twice over the same
weights, once with every integer path resident and once with all of them
dropped, and requires the argmax to survive. That test had a hole worth
recording: `Forward::disable_tensor_cores` reached the two mixers but not the
MoE, so for three commits the "fp32 twin" ran an integer MoE and the test
reported a pass for a path it never varied. It now covers all three.

### What this cost in correctness debt: nothing, but two tolerances moved

Two differential tests were asserting an fp32-summation-order bound at shapes
that had started taking the integer path. Both were **retargeted, not
loosened**:

- `moe_differential.rs` now runs *both* paths and gates each on its own bound,
  so the tight fp32 gate survives instead of being widened to cover int8.
- `moe_block.rs`'s CPU-reference bound is now a fraction of each layer's output
  magnitude. That was a latent flaw: blk.39's activations are two orders larger
  than blk.0's, so a constant tuned on one gates the other for no reason but
  scale. Its real assertion — that the device is no further from llama.cpp than
  the scalar reference is — is untouched, self-calibrating, and passes.

On blk.0 the integer path agrees with llama.cpp's own `ffn_moe_out` to
**max_abs 5.96e-8** while the scalar fp32 reference is 2.70e-4 away. llama.cpp
runs the same integer arithmetic over the same quantized weights; against that
target the fp32 reference is the outlier.

### One thing the router taught

The first tiled router was faster than the one that shipped — 1,087.05 against
1,075.60 — and wrong. It repartitioned the contraction across threads, the
logits moved in their last bits, and `forward_pass.rs` caught block 31's
relative error jumping 5.26×.

That failure is not the usual float-reassociation nuisance. **The router's
output is consumed by a top-8 argmax over 256 experts.** A last-bit
disagreement between two adjacent logits does not perturb an answer slightly;
it runs a different expert, and everything downstream is a different model. It
is the one matmul in this engine that cannot afford a reassociation. The
shipped version stages a contraction width equal to the block width, which
reproduces the untiled per-thread index order exactly. Correctness cost 1.1%.

## Decode: the same exercise, one token at a time (2026-08-16)

Prefill and decode turned out to be different problems with almost no
overlap in their fixes. Everything above is prefill. This is decode.

| | llama.cpp | llmxabe | position |
| --- | ---: | ---: | --- |
| Decode, warm | `tg128` **104.72 ± 0.36 tok/s** | **104.4–105.8 tok/s** (thermal) | **level** |

Decode began this session at 65.03 tok/s and 1.61× slower.

| change | ms/step | tok/s |
| --- | ---: | ---: |
| baseline | 15.38 | 65.03 |
| fix the shared-expert decode regression | 15.30 | 65.37 |
| GEMV path for the routed experts | 13.38 | 74.73 |
| GEMV path for the shared expert | 12.99 | 76.98 |
| repacked weights for the GDN projection | 12.18 | 82.11 |
| 16-bit Q6_K dequant loads | 12.07 | 82.86 |
| `char4`/`float4` in the GDN decode GEMV | 11.95 | 83.68 |
| warp-shuffle reductions in the two routing kernels | 11.59 | 86.27 |
| argmax on the device instead of over PCIe | 11.05 | 90.47 |
| word-wide Q6_K unpacking | 10.63 | 94.03 |
| capture the step as a CUDA graph | 10.70 | 93.50 |
| split the shared expert over its contraction | 10.42 | 95.99 |
| fuse the shared gate into the combine | 10.33 | 96.80 |
| top-k selection on one warp, no barriers | 10.33 | 96.80 |
| one-block dispatch table, split `moe_reduce`, wider combine | 10.00 | 100.05 |
| alpha, beta and the GDN gates in one launch | 9.82 | 101.85 |
| one-token convolution: output and cache advance fused | 9.76 | 102.46 |
| routing and dispatch in one launch | 9.71 | 103.00 |
| GDN output norm and SwiGLU in one launch | 9.67 | 103.45 |
| residual add folded into the output projection | 9.59 | 104.28 |
| fix a repack the reshape did not inherit | 9.59 | 104.28 |
| four warps per shared-expert row instead of eight | 9.59 | 104.3 |
| the routed sum folded into the combine | 9.51 | **105.2** |

### Two tiles that were re-reading their own input

Both were the same shape of mistake and both were found by asking one
question of every kernel: *how many times does the grid read the same bytes?*

**The router GEMM** moves `(ET + TT) * hidden` floats to do `ET * TT * hidden`
multiply-adds, so its arithmetic intensity is the harmonic mean of the two
tile widths — 2.7 MAC/byte at 4 experts by 8 tokens. Eight experts makes it 4:
**1,693 → 1,722 tok/s.** Sixteen is past the knee at 128 accumulators a
thread, and widening the token side instead lands in the same place while
doubling the staged activation tile.

**The chunk-end state update** stages `uprime` into `su[c][STATE_VB]`, indexed
by the value lane alone — so the tile is read once and used by all `STATE_JB`
of the block's `j` lanes. But `grid.z` is `head_dim / STATE_JB`, so every
block in `z` stages it again: at `STATE_JB = 4` that is **32 passes over
`uprime` per launch**, 32 MiB of traffic to read a 1 MiB tensor. Eight halves
it: **1,706 → 1,728 tok/s.**

| `STATE_JB` | tok/s |
| --- | ---: |
| 4 | 1,705.97 |
| **8** | **1,727.74** |
| 16 | 1,721.89 |
| 32 | 1,710.30 |

Past eight the block passes 512 threads and the grid stops covering the card —
32 leaves four blocks in `z` and 512 in total, against 72 SMs.

`gdn_chunk_inter` has the same shape — its `grid.x` is `chunk_len / INTER_TT`
and every block in it reads the whole `head_dim x head_dim` state slice — and
16 tokens per block beats 8 by 0.4% and 32 by 0.4%.

Neither changes a summation order.

The same reasoning applied to `gdn_chunk_gram` and **did not pay**. Every one
of its 64 blocks per head reads all 64 key rows, so tiling four target tokens
into a block ought to divide that by four. Measured interleaved against a
`GRAM_TT = 1` build — which is the original kernel exactly — it is 1,720.29
and 1,721.22 against 1,722.70: nothing. The chunk's keys are 32 KiB per head
and sit in L2, so the traffic that tiling removes was never being paid at
DRAM. Reverted; the arithmetic that says a kernel re-reads its input does not
say the re-read costs anything.

### Half the GDN solve was parallel and did not say so

`gdn_chunk_solve_and_apply` ran on `value_heads * head_dim` = 4,096 threads —
128 warps, about 5% of what 72 SMs hold — and its own comment gave the reason:
`vi` is the only axis it is parallel over. That is true of the forward
substitution, which is sequential in `t` because token `t`'s correction reads
`U'` rows `0..t`. It is **not** true of the section after it.

Section 3 computes each token's output from a `U'` that is already final, so
every `t` is independent. Section 4a — publishing `U'` with the chunk-end
decay folded in — is independent per `t` as well. Together they are close to
half the kernel's arithmetic, and both were running one token at a time.

The block is now `SOLVE_VB * SOLVE_TT` = 32 x 8 threads on the same grid.
Sections 1 and 2 run on the first 32 and the rest wait at the barriers they
were going to wait at anyway; sections 3 and 4a use all 256. That is **1,024
warps instead of 128** for the half that could take them.

| tokens in flight | tok/s |
| --- | ---: |
| 1 (before) | 1,644.10 |
| 4 | 1,691.35 |
| **8** | **1,708.06** |
| 16 | 1,693.84 |

**+3.9%.** Sixteen is past the knee: the block becomes 512 threads and its
shared footprint 18.5 KiB, and each token lane needs a `decay` row of its own.

Every per-thread sum keeps its order — the substitution's `i < t`, the
output's `i <= t`, the ascending `gcum` — so this is **bit-identical**, and
the only thing that changed is how many of them run at once.

### The flash kernel crossed a barrier every eight keys

`attn_flash_causal` gave one key to each warp per trip, so a block of eight
warps scored eight keys, rescaled its running softmax, and crossed **two
barriers** — 128 of them for a 512-token prefill row, and the same shape at
one token.

The scores of different keys are independent, so a warp can compute several
back to back and the block can rescale once for all of them. With four keys
per warp the tile is 32 and the barrier count drops fourfold; the serial
sweeps over the tile (`tile_max`, `lsum`, the value accumulation) get four
times longer but run four times less often, so their total is unchanged.
Storing key `j0 + warp * ATTN_KT + r` at slot `warp * ATTN_KT + r` is what
keeps slot `w` holding key `j0 + w`, and the value accumulation a single
ascending sweep.

| keys per warp | tok/s |
| --- | ---: |
| 1 (before) | 1,620.70 |
| 2 | 1,628.18 |
| **4** | **1,644.10** |
| 8 | 1,639.04 |

**+1.4%**, measured back to back against the old shape. Eight is past the
knee: the tile's serial sweeps are executed redundantly by all 256 threads, so
past some width they cost more than the barriers they save.

The online softmax now rescales every 32 keys rather than every 8, which is
not bit-identical — it is fewer rescalings, so slightly *more* accurate.
`attention_differential` reports cosine 1.000000000 against the CPU reference
at every depth it tests, and the depth list moved to 1, 9, 32, 129, 511, 1003
so that "exactly one tile" and "ragged against the tile" still mean what the
comments say.

### The decode margin is smaller than the card's thermal drift

Decode was reported above as 105.2 tok/s against llama.cpp's 104.72. That
number is real but it is not a margin. Measured on this card across one
session, the *same binary* gives:

| GPU temperature | tok/s |
| --- | ---: |
| 33 °C, idle two minutes | 105.75, 105.41, 105.06 |
| ~44 °C, straight after an hour of benchmarking | 104.41, 104.45, 104.49 |

**1.3% between a cold card and a warm one**, which is larger than the distance
to llama.cpp. Prefill drifts the same way over three back-to-back runs — 1,646
then 1,632 then 1,627 — and for the same reason.

So the honest reading of the decode column is **level**, not ahead: the
engine and llama.cpp are inside each other's spread at this shape, and any
claim of a win would be a claim about when the measurement was taken. Prefill,
at 1.27x behind, is outside it by a wide margin and is a real loss.

Every A/B in this document is measured **interleaved** — three runs of each
arm, alternating — for exactly this reason. A before/after taken an hour apart
on this card is worth about 1%.

### One number was a multiple of 32, and it cost 13% of prefill

Both MoE tensor-core kernels stage the activation tile in shared memory with a
row stride of `MOE_MMA_KC` — 128 bytes. The MMA operand load is

```c
sa + (mf * 8 + arow) * 128 + kk + quad
```

where `arow = lane >> 2` runs over eight rows and `quad = (lane & 3) * 4` over
four words. A shared bank is a word and there are 32 of them, so a **128-byte
stride is exactly 32 banks**: all eight rows land on the same four banks, and
every one of these loads is an **eight-way conflict**. It is issued four times
per contraction step in the kernel that is a quarter of the prefill.

A conflict-free stride has to move each row four banks along, so the stride in
words must be `4 (mod 32)`. 144 bytes is 36 words, `36 mod 32 = 4`, and the
eight rows tile the 32 banks exactly once. Sixteen wasted bytes a row.

**1,440 -> 1,630 tok/s, +13%,** and the same padding applies to the down
projection's tile, which had the identical stride.

Three earlier experiments had already said the kernel was neither
bandwidth-bound nor short of arithmetic; none of them said *why*, because a
bank conflict does not show up in bytes moved or instructions issued. It shows
up as a load taking eight times as long as it should, and with `ncu`
unavailable on this host the only way to it was reading the index expression
and counting banks.

The weight tile's stride was 100 bytes — 25 words — chosen earlier so the
eight rows would land on distinct banks. Distinct is not the same as four
apart: adding the word offset collided three of the eight. 112 bytes is 28
words, `28 mod 32 = -4`, which tiles exactly. Worth about nothing next to the
activation tile, and taken anyway because it is provably right rather than
accidentally equal.

### Padding Q6_K to a 16-byte stride: a real trade, taken

Q6_K's superblock is 210 bytes. `210 * s` is 4-byte aligned only for even `s`,
so every read of a superblock has to be 16 bits wide and the grouped GEMM's
staging loop pays four load instructions per weight row. Padding the *device*
stride to 224 makes every field of every superblock 16-byte aligned, which
lets one `int4` load stage eight rows at a time: four loads per eight rows
instead of four per row.

It was implemented end to end — a `to_device_layout` re-stride on upload, a
`file_block_bytes` / `block_bytes` split so GGUF validation and device
addressing stop being the same number, and an `int4` staging loop whose lane
split (`row = lane & 7`, `chunk = lane >> 3`) is also conflict-free against the
112-byte shared stride. It works, and the numbers, interleaved three runs each:

| | prefill tok/s | decode tok/s |
| --- | ---: | ---: |
| 210-byte stride | 1,626 | **104.91** |
| 224-byte stride | **1,664** | 104.14 |

**+2.3% prefill, −0.7% decode**, plus 1.1 GiB of VRAM. The pad bytes are never
read, but they are inside the sectors that are, so the traffic grows 6.7% on
`ffn_gate_exps` and `ffn_up_exps` — and decode is bandwidth-bound where prefill
is not.

It was reverted the first time it was measured, on the reasoning that decode
was 0.2% ahead of llama.cpp and prefill 27% behind, so spending the goal
condition that is met to buy the one that is not was the wrong direction.

Two things changed that. First, the decode side of the same alignment was left
on the table: `dequant_tile_q6k` — the one-token expert GEMV's inner loop, 17%
of a decode step — was still doing two 16-bit loads for `ql` and one for `qh`
because the *file's* 210 bytes never guaranteed 4-byte alignment. With a
224-byte device stride every superblock base is word-aligned, so those become
one 32-bit load each. Re-measured with that in, interleaved three runs each:

| | prefill tok/s | decode tok/s |
| --- | ---: | ---: |
| 210-byte stride | 1,732.25 | 104.50 |
| 224-byte stride | **1,762.46** | 104.13 |

The wider loads halve the decode penalty — 0.7% becomes **0.35%** — but they do
not erase it, which says the extra 6.7% of sector traffic is the real cost and
instruction count was never decode's problem. That is the honest version: the
padding is a bandwidth trade in both directions, and only prefill has spare
bandwidth.

Second, the target moved. With decode's requirement stated as a floor rather
than a race — stay near the current value, not below 100 — 104.13 is 4 tok/s of
headroom, and 0.35% is well inside this card's 1.3% thermal drift, while
prefill's 1.7% is a repeatable step. Taken.

A 212-byte stride was considered instead: 4-byte aligned, so `int` loads work
and the padding costs 0.95% rather than 6.7%. It is not equivalent. Most of the
224 win is that 224 is a multiple of **32**, so a superblock starts on a sector
boundary; 212 gives the alignment for the loads without the alignment for the
fetches.

### The dense projections are not weight-traffic bound either

`mma_q8_0_proj_split` re-reads the whole weight band once per token tile --
`grid.y` is `tokens / MMA_SPLIT_TOKS`, which is eight passes over the weights
at 512 tokens. That is documented above as the reason a 16-token tile wins in
isolation and loses in the engine.

What was *not* deliberate: the warps of a block took **different** row bands,
so four warps meant four disjoint weight reads and the block shared nothing at
all. Giving them one row band and four token tiles instead makes three reads in
four an L1 hit and turns eight DRAM passes over the weights into two.

Measured interleaved against the old mapping, three runs each: **1,426.7 vs
1,416.4 tok/s.** Real, kept, and 0.7% -- which is the finding. A 4x cut in the
weight traffic of a kernel that is 15% of the pass is worth 0.7%, so that
kernel is not weight-traffic bound. It is bound by its register file: 128
accumulators put it at ptxas's 255-register ceiling and eight warps to an SM,
and no tile shape in the sweep below improves on 64 x 64 under the new mapping
either.

| warps x tokens x rows | tok/s |
| --- | ---: |
| **4 x 64 x 64** | **1,434.91** |
| 2 x 64 x 64 | 1,435.78 |
| 8 x 32 x 64 | 1,369.17 |
| 4 x 32 x 64 | 1,383.52 |
| 4 x 64 x 32 | 1,351.86 |

### Staging was 60% of the MoE GEMM, and it was a load count

`profile_forward` puts the MoE at 210 ms of a 373 ms prefill -- 56% -- moving
its weights at about 145 GB/s, a fifth of what the card does and a sixth of
what the LM head's own GEMV reaches on the same silicon. Three experiments
narrowed it down, and two of them were wrong in an instructive way.

**Halving the inner loop made the pass 12% faster, and proved nothing.**
Neither did stripping the scale multiplies, which made it 10% *slower*. Both
change the kernel's output, the output is a router logit two layers later, and
a model producing garbage routes its tokens differently -- concentrated onto a
few experts in one case, spread onto badly-packed ones in the other. **Any
experiment that perturbs numerics also perturbs the expert distribution, and
therefore the weight traffic.** They cannot be used to attribute time.

The experiment that worked runs the weight-staging loop **twice** per trip. It
writes the same bytes to the same shared addresses, so the output is
bit-identical and routing cannot move; the pass went 371.65 -> 437.50 ms.
Staging is **66 ms of the kernel's 110**, and it stays that expensive on the
second pass, when every byte is already in L1. So it is not DRAM bandwidth. It
is the *number* of load instructions.

There were eight per weight row per trip:

| | bytes wanted | lanes used | sectors fetched |
| --- | ---: | ---: | ---: |
| `ql` | 64 | 32 | 2 |
| `qh` | 32 | 16 | 1 |
| sub-scales | 8 | 8 | 1 |
| superblock delta | 2 | 1 | 1 |

times two matrices. The last two fetch a 32-byte sector for eight and two
useful bytes, and `load_half_le` reads its two bytes as two separate
single-byte loads, so the delta alone was two instructions.

The fix is that `qh`, the sub-scales and the delta together occupy 21 lanes of
one warp — 16 for `qh`, 4 for the scales, 1 for the delta — so all three fit
in a **single predicated 16-bit load** per matrix, with 11 lanes idle. Four
loads per row instead of eight. The bytes fetched are identical; only the
instruction count changed. The Q8_0 down projection got the same treatment for
its fp16 scale.

**1,380.38 -> 1,430.58 tok/s.**

Two things tried on the way that did not pay:

- **Accumulating the Q6_K sub-scale in int32.** SASS showed 20 of the inner
  loop's 105 instructions were `I2F`, which is quarter-rate on Turing; folding
  the int8 sub-scale into the integer accumulator cuts that to 8 per 16
  elements. It measured as *nothing*, because the kernel is staging-bound, and
  it moved block 31 enough for `forward_pass`'s error-growth guard to fail.
  Reverted: the arithmetic is back to one float multiply per sub-block and
  bit-identical to what it was.
- **A 32-slot dispatch block** (`MOE_MMA_M` 16 -> 32) to halve the number of
  times an expert's weights are re-read. Worth 1.01x, not the 1.25x the
  padding arithmetic suggested -- because `profile_forward` says the routing
  on this prompt is already almost perfectly packed: 496 dispatch blocks
  launched, exactly 256 of them carrying work, which is the ideal. Kept
  anyway, together with the wider 8-warp block, for the 1.01x each.

### The last launch that was only a sum

`moe_reduce` existed to turn the per-`(token, k)` contributions the grouped
GEMM writes into each token's routed sum. That is one fused multiply-add per
element, and it cost a launch per layer — 2.2 us of a 9.6 ms step, forty times
over, against a kernel that was going to read the result anyway.

`moe_block_gate_and_combine` now reads `partial` directly and accumulates the
`top_k` slices itself, ascending in `k` inside one thread, which is both the
reference's order and the order `moe_reduce` used. `MoeKernels` grew
`grouped_forward_partial`, which is the old `grouped_forward` minus its last
launch; `grouped_forward` is now that plus the reduce, so
`tests/moe_differential.rs` still drives the contract it was written against.

The combine kept writing `routed`. Nothing on the forward path reads it, but
`ffn_moe_out-N` is a golden waypoint `tests/moe_block.rs` gates on, and an
8 KiB store is not worth losing a comparison against llama.cpp's own
activations for.

The fold made the combine read `top_k * hidden` floats per token instead of
`hidden` — 64 KiB, which one block per token would have put on one of 72 SMs.
So the combine took `moe_reduce`'s second grid dimension too, and shrank from
1,024 threads to 256. Every block in the row now recomputes the gate's dot
product; it reads the same 16 KiB each time, so seven of the eight are L2
hits, and it is cheaper than the launch it replaced. 256 is also exactly
`MOE_GATE_LANES`, so the gate's reduction tree is unchanged: `block_reduce_sum`
sums per-warp partials ascending, and the zeros contributed by lanes past the
gate's width land in the same places at any block width at or above it.

**9.59 -> 9.51 ms, 104.3 -> 105.2 tok/s.** That is the first measurement in
this project ahead of llama.cpp's `tg128` on the same card.

### Splitting a row eight ways was one split too many

`moe_shared_ffn_gemv` was retiled earlier in this arc from "one warp per row"
to "one block per row, eight warps splitting the contraction", which is what
took decode from 10.70 to 10.42 ms. Eight was chosen because it is the block
width the surrounding kernels use, not because anything measured it.

At `hidden = 2048` eight warps leave each warp **two 128-element tiles**: it
issues four global loads, reduces across its lanes, and retires. That is short
enough that the block spends more of its life being scheduled than reading,
and the 512-block grid is already four times the number of resident blocks the
card will hold. Halving the split to four doubles the work each warp does
without changing the grid, and the whole kernel is a 128-thread block instead
of a 256-thread one:

| warps splitting the row | ms/step | tok/s |
| --- | --- | --- |
| 8 | 9.63 | 103.8 |
| **4** | **9.59** | **104.3** |
| 2 | 9.64 | 103.8 |

Two is worse again, and for the opposite reason: each warp is now reading
1 KiB, but there are only two of them per block, so the block cannot cover its
own memory latency. Four is the knee, and it is a knee rather than a trend —
which is the reason to measure it rather than reason about it.

The partial sum is now four wide instead of eight, so the shared expert's
output is not bit-identical to what it was. `forward_pass`'s golden gate — a
per-block relative-error comparison against llama.cpp's own activations —
passes unchanged, and the final argmax is the same token.

### The one idea behind all of it

**At one token every kernel in the model is a GEMV, and three of them were
still running a GEMM's machinery.** The MoE experts staged an activation tile
in shared memory and crossed two barriers per 128 elements of contraction, to
multiply each dequantized weight exactly once. There is no reuse to capture at
this shape — a weight is read, used, and dropped — so all of that apparatus is
overhead. `nsys` put the two routed-expert kernels at 28.9% of decode moving
their weights at 27% of the card's streaming roofline, against the 82% the LM
head's own GEMV reaches on the same card. Removing the tiling closed most of
that.

The other half is a layout accident. Q8_0 places a row's quants at byte
`b * 34 + 2`, so a warp reading 32 contiguous quants is 32-byte aligned only
when `b == 15 (mod 16)`. **Fifteen blocks in sixteen straddle a sector
boundary**, and half of every fetch is discarded. The repacked split layout
built for the tensor cores fixes it for free — it separates quants from scales,
so a row's quants are contiguous from an aligned base. That repack turns out to
be worth having for its *alignment* even where the arithmetic stays fp32, which
is not why it was built.

### Where decode time goes now

Per step, from `nsys` over 44 decode steps at a 64-token prompt:

| kernel | ms/step | share | % of streaming roofline |
| --- | ---: | ---: | ---: |
| `gdn_proj_split_gemv` | 2.22 | 20.9% | 71% |
| `moe_expert_ffn_gemv` | 1.74 | 16.4% | 47% |
| `lm_head_gemv_b1` | 1.52 | 14.3% | 82% |
| `moe_expert_down_gemv` | 0.86 | 8.1% | 63% |
| `moe_shared_ffn_gemv` | 0.67 | 6.3% | — |
| MoE dispatch glue (7 kernels) | 1.63 | 15.3% | — |
| everything else (~20 kernels) | 1.4 | 13% | — |
| **GPU idle** | **0.97** | **9.1%** | — |

`lm_head_gemv_b1` is 41 launches, not one: the ten Gated Attention layers run
their four projections through the same kernel. The real LM head is about
0.9 ms of that row.

`moe_expert_ffn_gemv` is still the weakest streamer at 47%, and it is still
the only kernel reading Q6_K rather than Q8_0 — but the reason is no longer
what it looked like. Q6_K's unpacking is **integer** work, and at this shape
the kernel is bound by the integer pipe rather than by DRAM: the word-wide
unpack above bought 8% of the whole decode step without touching a single
byte of traffic.

### The launch gap, and the CUDA graph that did not collect it

An earlier version of this section put the GPU at "about 92% busy, so ~0.97 ms
is launch gap". The number was right by accident: the first pass computed idle
from the *kernel* timeline alone, and a decode step also issues about 180
memsets and memcopies, so most of what looked like idle was those. Counting
all three activity kinds gives the same total for better reasons:

| | ms/step |
| --- | ---: |
| gaps under 1.5 us, one per launch | 0.63 |
| one long gap per step, host turnaround | 0.31 |
| everything else | 0.03 |
| **total idle** | **0.97** |

1,124 device operations per step at about 0.6 us of dead time each is the
whole first row, and `Forward::run` spends **3.1 ms of host time** issuing
them — 2.8 us per launch, against kernels that average 8.6 us. So the step
was recorded as a CUDA graph and replayed.

**It works, and it is worth nothing.** `nsys` confirms the graph does exactly
what it was supposed to: idle per step falls from 0.97 ms to **0.405 ms**, and
the host turnaround from 0.31 ms to 0.02 ms. The wall clock does not move —
10.63 ms without, 10.70 ms with, against a run-to-run spread of 0.05 ms.

The reason is visible in the same profile. Under graph replay every decode
kernel is **0.3–0.5 us slower**, uniformly, across all 1,100 of them: about
0.55 ms, which is what the 0.57 ms of recovered idle paid for. On this
hardware a graph node costs roughly what the launch gap it replaces cost. That
is a Turing result and should not be generalized — Ampere added hardware
acceleration for graph node dispatch that this part does not have.

What was kept anyway, and why:

- **The sequence position is a device scalar now.** The rotary embedding, the
  causal bound and the key/value append all took it as a host argument, and
  the append used it as a *slice offset* — a host-computed destination
  address. `AGENTS.md` rule 5 says nothing on the forward path may be sized or
  indexed by a host-side value, and this was the last place that was not true.
  The append is also one kernel now instead of two `cuMemcpyDtoDAsync`.
- **The host cost of a decode step went from 3.1 ms to about 0.05 ms.** That
  buys nothing in a benchmark whose host does nothing else, and it is most of
  what a serving surface needs the host for.
- **`tests/graph_decode.rs` exists**, and it caught a real bug on the first
  run: `MoeKernels::set_valid_tokens` copied a host `i32` into a device scalar
  once per layer, inside the capture. A copy from pageable host memory is not
  something a graph may contain; the driver accepted it in relaxed capture mode
  and produced a graph that generated `[222543, 20073, 20073, 20073, …]` —
  fluent, finite, and frozen after one step — against the launch path's
  `[198, 248045, 74455, 198, …]`. The count is published once per pass now,
  ahead of the capture, and repeats write nothing.

### Two shapes that did not pay, on the kernel that looks most like the one that did

`gdn_proj_split_gemv` is the largest single item in a decode step at 2.25 ms
and 71% of the roofline, and two changes that worked elsewhere did nothing for
it:

- **128-bit weight loads.** `int4` instead of `char4` cuts the step from four
  loads per sixteen elements to five per sixteen *including* the scale — and
  it forces the activation reads to a 64-byte lane stride, so each of the four
  `float4`s touches 64 sectors to use 512 bytes. Net: nothing, within noise.
- **Staging the activations in shared memory.** Every warp in the grid
  contracts against the same 8 KiB, and at 8,192 output rows that is 67 MiB of
  L2 traffic per launch. Staging it once per block is the transformation that
  took the MoE grouped GEMM from 110.9 ms to 11.9 ms — and here it cost 3.6%
  of the whole decode step (10.42 → 10.79 ms), because 8-16 KiB of shared
  memory takes the block count per SM from four to three and the L2 was
  serving those re-reads at a price the occupancy was worth more than.

The lesson is the same one the MMA tile taught in the prefill section: the
transformation that fixed one kernel is not a property of the transformation.

### Four more things that did not pay

All four are the transformation that worked on a neighbouring kernel, applied
where it did not:

- **Staging the routed experts' activations in shared memory.** The arithmetic
  looked overwhelming: eight warps a block and 512 blocks all contract against
  the same 8 KiB, so `moe_expert_ffn_gemv` was reading 32 MiB per layer to
  deliver 8 KiB against 13.8 MiB of weights read once. And unlike the GDN
  projection, the footprint here is free — 8 KiB a block, four blocks a SM,
  32 of the 48 KiB available. It cost 0.8% of the decode step anyway, because
  those re-reads are **L1 hits**: four blocks' worth of activations is 32 KiB
  and Turing's L1 is 64. Staging replaced a hit with a copy and a barrier.
- **A 16-bit load for Q6_K's fp16 delta.** `load_half_le` reads two bytes
  separately; the superblock is 210 bytes and the delta sits at offset 208, so
  both are even and one 16-bit load is legal. Measured as nothing — nvcc had
  already merged them.
- **Trimming `moe_route`'s remaining barriers.** The softmax denominator's
  last five tree steps are all within warp 0, so they can use `__syncwarp`
  instead of `__syncthreads` without changing a single addition's order, and
  the normalizing pass over 256 probabilities can be folded into the selection
  warp's registers. Both are strictly less work. Neither moved the number:
  the kernel is its launch floor.
- **Fusing the SiLU into the q/k/v split.** One fewer launch, and the same
  three-microsecond kernel — the split's block is half as wide as the SiLU's
  grid-stride shape, so each thread now does three exponentials instead of one.

### Decode depends on the prompt that preceded it, and it should not

A prompt sweep at 96 decode steps, before the fix in this section:

| prompt | ms/step | tok/s |
| ---: | ---: | ---: |
| 1 | 9.76 | 102.5 |
| 4 | 10.90 | 91.8 |
| 8 | 10.97 | 91.2 |
| 16 | 11.02 | 90.8 |
| 32 | 10.98 | 91.1 |
| 64 | 9.65 | 103.6 |
| 128 | 9.78 | 102.2 |

**A 13% cliff between 32 and 64, and 1 on the fast side of it.** Those are
exactly the two prompt lengths that build the repacked int8 Gated DeltaNet
projections: `tokens == 1` or `tokens >= MMA_SPLIT_TOKENS`, which is 64. A
prompt of 4..63 builds neither — and `Forward::reshape` shares the repack by
reference count, so the one-token pass it spawns inherited an **empty vector**
and fell back to the fp32 projection path. Nothing at the call site says so.

This is the second bug of exactly this shape this session; the first was the
shared expert inheriting repacked weights it should not have used. Sharing by
`Arc` between two passes with different needs is the hazard, and "the donor
built it" is not the same question as "this shape wants it".

Fixed by building the repack when the inherited one is empty and this shape
wants it. The sweep is flat afterwards, varying only with the KV window:
9.58 ms at 8 and 32, 9.64 at 64, 9.78 at 128.

### A measurement discipline note

One entry above — the 16-bit dequant widening — was first recorded in a commit
message as a **57% regression** and listed as a rejected idea. It was measured
while an unrelated decode bug was live, so both its before and after numbers
were that bug. Against a clean baseline it is a small improvement. A negative
result taken during an unrelated regression is not a measurement, and it was
recorded as one. The correction is in commit `6faaa99`.

The same session produced two other results that reversed under measurement,
both kept because they are cheap to rediscover:

- **Shrinking the MMA tile.** Faster on all four shapes in isolation, 8% slower
  in the engine. The token tile divides `grid.y`, every block in `y` re-reads
  the weight band, and a microbenchmark measures cache residency the kernel
  will not have in situ.
- **Widening the MoE dispatch block from 16 slots to 32.** Doubles what each
  staged fragment buys and halves the dispatch-block count, and at 512 tokens
  an expert averages 16 tokens so the two cancel exactly. Everything narrower
  regressed hard.

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

## The gap was never the MoE GEMM (2026-08-16)

Prefill spent this session going from 1,735 to 2,123 tok/s, and none of it came
from the kernel everyone had been staring at.

`llama-bench` was run under `nsys` on the same card, at the same batch, against
the same file, and its kernel summary put side by side with this engine's:

| stage | llama.cpp | llmxabe (before) | gap |
| --- | ---: | ---: | ---: |
| MoE gate/up, Q6_K | 77.8 ms | 58.8 ms | **−19.0, ahead** |
| MoE down, Q8_0 | 41.4 ms | 39.6 ms | −1.8 |
| Gated DeltaNet | 27.9 ms | **71.9 ms** | **+44.0** |
| Attention | ~2.1 ms | **21.1 ms** | **+19.0** |
| Dense projections | 37.8 ms | 52.9 ms | +15.1 |
| everything else | ~46 ms | ~25 ms | −21 |

The grouped GEMM — 25% of the pass, the subject of most of this document, and
the thing three previous sessions optimized — was already **32% faster** than
llama.cpp's. The whole deficit was in the two mixers and the dense projections,
and the largest single item, Gated DeltaNet, had never been profiled against
anything.

### One defect, four kernels

Attention, the DeltaNet state update, the DeltaNet solve, and the alpha/beta
gates turned out to have the same shape of bug, and it is not a memory-traffic
bug — it is an **instruction-count** bug:

```text
attn_flash_causal   per key: 8 global k + 8 shared q + 8 FMA   (scores)
                             1 global v + 1 shared w + 1 FMA   (values)
state_update        per i:   1 global k + 1 shared u + 1 FMA
solve               per i:   1 global kk + 2 shared + 3 FMA
alpha_beta_gates    per i:   2 global w  + 1 global x + 2 FMA
```

Roughly one memory instruction per multiply-add, everywhere. Turing issues
**4 load/store operations per SM per clock against 64 FMAs**, so a kernel at
that ratio runs at a sixteenth of the arithmetic pipe and no amount of L2 hit
rate moves it. Every one of these looked "bandwidth-bound" in a roofline sense
and none of them were: `attn_flash_causal` was reaching 2 TB/s of effective
bandwidth, which is *L2 working properly*, and it was still 16× off.

The fix is the same in all four: find the index the operand does not depend on,
and reuse it.

- **Attention** carries eight query rows per block with the query tile in
  *registers*, so one `k` element feeds eight multiply-adds and the score
  loop's shared reads disappear entirely. 21.1 → 8.5 ms.
- **The state update** is a rank-`c` outer product, `S[vi][j] += Σ u[i][vi]
  k[i][j]`. Neither operand depends on the other's index, so a 4×4 register
  tile reads one `float4` from each and does sixteen multiply-adds. 15.6 →
  5.3 ms.
- **The solve** was recomputing `beta_t * kk_row[i] * decay[i]` in all 32 lanes
  of a forward substitution and reloading `kk_row[i]` from global to do it.
  None of it depends on the lane. 26.3 → 19.1 ms.
- **The gates** had one warp per (head, token) reading 16 KiB of weights to do
  4,096 multiply-adds. A band of eight tokens reads them once for eight times
  the arithmetic.

All four are **bit-identical**. That is not a coincidence — it is the reason
each was safe to make. A register tile does not reassociate anything: every
accumulator still sums its contraction in the same order. Hoisting
`(beta_t * kk_row[i]) * decay[i]` out of a loop forms the same product in the
same left-to-right association the expression already had, once instead of 32
times. The golden gate agreed on all four, which is the check that matters when
a routing logit two layers downstream turns a last-bit move into a different
expert.

### A partial band costs its empty lanes in full

Both banded kernels regressed decode the first time, and by the same mechanism.
At `n_query == 1` the flash kernel still ran eight multiply-adds and eight
shuffle reductions per key to throw seven of them away; the gate kernel did the
same with its token band. Decode fell 105.1 → 99.6 tok/s each time.

Both now have a one-unit instantiation, dispatched on the batch. The flash
kernel's is the *pre-tiling kernel unchanged*, because the two widths want
different softmax bookkeeping: at eight rows the per-tile max and normalizer
are one thread per row, since every thread sweeping `QT * tile` shared floats
costs more than the value loop it amortizes; at one row there is nothing to
amortize and the redundant sweep is cheaper than serializing onto one thread
and adding a barrier.

This is the second time in this document a batch-shaped tile has been a decode
regression. It is worth stating as a rule: **any tile widened for prefill needs
a one-unit sibling before it is committed**, because decode is the one-unit
case of every one of them.

### llama.cpp does not chunk the DeltaNet at all

Worth recording, because it changes what "optimize the chunked form" is worth.
`gated_delta_net_cuda` is a **sequential token-by-token scan** with the entire
128×128 per-head state resident in four registers per thread, one warp per state
column, zero shared memory and zero `__syncthreads` for the whole sequence. The
chunked form exists in llama.cpp only as the slower ggml-graph fallback, and the
CUDA file carries a `//TODO: Add chunked kernel for even faster pre-fill`.

The arithmetic says why. At head dim 128, chunk 64, 512 tokens, the sequential
scan is `4·D²` FMA per token — 33.5 M per head — and the chunked form is about
42.3 M. **Chunking does 26% more arithmetic here**, and it only pays when the
matmul shape buys tensor cores. Turing has no fp32 tensor cores, so in fp32 on
this card chunking is pure overhead.

This engine's chunked path is now 47.8 ms against llama.cpp's 27.9. The
remaining 20 ms is reachable, but the way to reach it is to delete the chunking
for prefill, not to keep tuning it — and that is a rewrite, not a tuning pass,
so it is recorded here rather than attempted.

### Level is not the same as ahead

The headline is three alternating rounds of `llama-bench -p 512 -r 3` and
`bench_forward` n=512, back to back on the same card in the same session:

| round | llmxabe | llama.cpp |
| --- | ---: | ---: |
| 1 | 2,108.09 | 2,085.93 ± 160.36 |
| 2 | 2,095.02 | 2,080.17 ± 160.95 |
| 3 | 2,094.79 | 2,062.60 ± 179.53 |
| **mean** | **2,099.3** | **2,076.2** |

That is **1.011×**, and the honest reading is "a small repeatable win", not a
headline. llama.cpp's own `pp512` reports ±160–180 tok/s run to run — an 8%
spread — and this card's thermal drift is about 1.3%, both larger than the
margin. What makes the comparison stand up is only that the two were alternated
on one card within minutes, so the drift applies to both arms equally.

The earlier figure this document compared against, 2,070.50, is a single
measurement from a different session. Re-measured this way llama.cpp is 2,076.2,
which is the number the table above uses.

### Hoisting the Gram kernel out of the chunk loop: measured, rejected (2026-08-16)

`gdn_chunk_gram` is the one chunk stage that never reads or writes the
recurrent state — it consumes `q_norm`/`k_norm` and produces `kk`/`kq`, and
nothing else. Nothing therefore orders it against the chunk loop the other
three stages are bound by, so it can be launched once for the whole sequence
with the chunk index as a third grid axis. At 512 tokens that is 8 launches
per layer collapsed to 1: **240 → 30** across the 30 GDN layers, about 210 of
the pass's 960 launches, and 10.9 ms/pass of kernel time was on the table.

It was implemented, it produced bit-identical output — all six device
differential cases pass, including the 581-token ragged tail that exercises
the fixed `chunk_len` stride the hoisted layout forces — and it is **1.07 ms
slower**. Three interleaved pairs, `bench_forward` n=512, 4 reps each:

| round | in the loop | hoisted |
| --- | ---: | ---: |
| 1 | 228.79 | 230.65 |
| 2 | 230.26 | 231.12 |
| 3 | 230.77 | 231.27 |
| **mean ms/pass** | **229.94** | **231.01** |

The hoisted arm loses every pair, which is what makes +0.47% readable against
1.3% thermal drift: the drift moves both arms together and the ordering does
not survive it by accident.

The mechanism is L2, not launches. In the loop, `kk` and `kq` are one
`chunk_len²` square per query/key head — 512 KiB at this geometry — written by
the Gram kernel and read by `gdn_chunk_solve_and_apply` immediately afterwards,
so the solve hits in L2 essentially always. Hoisted, all eight chunks' squares
must coexist, because they are all produced before the first one is consumed:
4 MiB live in a 6 MiB L2, competing with the projections and the state. The
launches saved are worth less than the locality lost.

The general lesson is the one this document keeps arriving at from the other
direction: **on this part, launch count is not a bottleneck worth trading
locality for.** A 512-token pass issues ~960 launches in 229 ms — 0.24 ms of
launch overhead per millisecond of work would be a 100% tax, and the measured
tax at prefill shape is ~0.4%. Collapsing launches only pays when it does not
inflate the working set.

This does not generalize to the state-carrying stages. It is specifically the
finding that *this* hoist, the only one the data dependencies permit, is not
worth taking.

## Deleting the chunking for prefill: 2,221 → 2,322 tok/s (2026-08-16)

The section above on llama.cpp's DeltaNet ends by saying the remaining 20 ms is
reachable but "the way to reach it is to delete the chunking for prefill, not
to keep tuning it." That is now done and measured.

`gdn_scan_prefill` is a sequential token-by-token scan: one warp per (value
head, value index), the whole `head_dim`-wide state row held in `SCAN_MAXR`
fp32 registers per lane for the entire sequence, two `__shfl_xor_sync`
butterflies per token, **no shared memory and no `__syncthreads` at all**. It
replaces four launches per chunk — Gram, inter, solve, state update — with one
launch for the whole sequence. The normalization pass is unchanged and shared
with the chunked path, which still exists as `prefill` and still has its own
tests.

Three interleaved rounds, `bench_forward` n=512, 4 reps each:

| round | chunked | scan |
| --- | ---: | ---: |
| 1 | 229.20 | 219.78 |
| 2 | 231.02 | 220.78 |
| 3 | 231.27 | 221.08 |
| **mean ms/pass** | **230.50** | **220.55** |
| **tok/s** | **2,221.3** | **2,321.5** |

**−9.95 ms, +4.5%**, winning every pair — the same interleaving discipline the
rejected Gram hoist above was measured under, and here the ordering is
unambiguous rather than marginal.

Decode is unaffected: 104.30 vs 103.95 tok/s over two interleaved rounds, a
0.34% difference well inside run-to-run noise and clear of the 100 tok/s floor.
The scan is the one tile-shaped change in this document that did *not* need a
one-unit sibling, because it never had a token tile to begin with: at one token
its loop runs once and its launch geometry is identical.

### It is gated, and it was not before

The scan was written before this session and left uncommitted for one reason:
nothing tested it. There were five `.prefill(` comparisons in
`gdn_chunked_differential.rs` and zero `.scan(` ones, so the only thing
exercising the kernel a forward pass actually runs was an end-to-end argmax,
which cannot separate a real bug from a knife-edge input.

`run_case` is now parameterized over the entry point, and the scan answers to
the same gate as the chunked form against the same two host references — the
chunked host form *and* the recurrent host form, outputs and final state
checked separately, per head:

| case | tokens | result |
| --- | ---: | --- |
| `device_scan_gdn_matches_both_reference_forms_over_512_tokens` | 512 | pass |
| `device_scan_gdn_matches_both_reference_forms_from_a_non_zero_initial_state` | 581 | pass |
| `device_scan_gdn_matches_the_chunked_form_on_a_single_token` | 1 | pass |
| `device_scan_gdn_survives_this_models_real_decay_rates` | 197 | pass |

The last is the one that mattered most. The scan carries `exp(log_decay)` per
token in a register rather than a cumulative log-decay per chunk, so it does
not inherit the chunked kernel's proof that nothing ever divides by a decay,
and this model reaches `exp(-91.578)` in a single token. It comes through with
zero non-finite elements.

### The relative-error guard was checking the wrong condition

Adding the one-token scan case exposed a defect in the *test harness*, not the
kernel. `GATE` demotes `max_rel_error` to a tripwire because `compare()`
divides by `max(|reference|, 1e-6)` and GDN drives many components to zero;
`check()` guarded that demotion by asserting the element driving the worst
ratio is below `1e-3`, on the reasoning that above that the ratio is a real
measurement rather than a floor artefact.

But it asserted the magnitude *unconditionally*. The scan at one token reports
`max_rel 4.9e-6` on a reference of `1.0e-3` — an absolute error near `5e-9`,
four orders of magnitude inside every bound in play — and failed a guard that
exists to police an excuse it never invoked. The guard now fires only when the
demotion is load-bearing, i.e. when the ratio actually exceeds what the stated
gate would have allowed. That is strictly stronger: it stops rejecting answers
more accurate than the one it was written to protect, and it still catches the
case it was written for.

### The int8 argmax check was a coin flip on this input

Switching the mixer flipped `int8_forward`'s argmax, and the honest reading is
that the test was never decidable on this prompt rather than that the scan is
wrong. Instrumenting both arms with their top-3 logits:

| | best | runner-up | margin |
| --- | --- | --- | ---: |
| fp32 | 220 @ 6.647565 | 248045 @ 6.524912 | **0.1227** |
| int8 (scan) | 248045 @ 6.568379 | 220 @ 6.564664 | 0.0037 |

The int8 path's own worst disagreement with fp32 on this input is **0.278**,
more than twice the fp32 margin between the top two candidates. Two tokens that
close cannot be separated through 40 layers of int8 requantization by anything
but luck; at HEAD the luck fell the other way. The pre-scan run measured the
same near-tie — `max|diff| 0.203` against a margin of `0.123`.

The assertion is now conditional on the reference margin clearing `0.25`, and
prints the full comparison when it does not, so an argmax flip reports *why* it
flipped instead of asserting on rounding. The cosine floor (`0.999`) is
unconditional and is what actually gates the distribution: it reads `0.999681`
with the scan against `0.999843` without, both far above the bound.

### Asking ptxas for a third resident block on the MoE GEMM (2026-08-16)

With the DeltaNet scan landed the two MoE expert GEMMs are 46% of the pass, and
the profile says they are memory-side: `moe_expert_ffn_mma` moves 470 MB of
Q6_K per layer at about **45% of the 672 GB/s peak**, while issuing roughly
**5% of the card's int8 throughput** doing it. Its staging loads are already
`uint4` — that was fixed earlier and is documented above — so what was left was
loads in flight, which is warps resident.

The shared footprint is 21,760 bytes: two `MOE_MMA_ROWS x MOE_MMA_WSTRIDE`
weight tiles, two scale tiles, the int8 activation tile, its per-32 scales, and
a row index per slot. Turing's SM has 65,536 bytes of shared to give, so
**three blocks fit with 256 bytes to spare** — but ptxas had no reason to size
the register allocation for three, and nothing in the source asked it to.
`__launch_bounds__(MOE_MMA_WARPS * 32, 3)` asks. The Q8_0 down projection
stages one weight tile rather than two, is 14,720 bytes, and asks for four.

Five interleaved pairs, `bench_forward` n=512, 4 reps each:

| round | without | with |
| --- | ---: | ---: |
| 1 | 218.12 | 217.57 |
| 2 | 219.77 | 218.67 |
| 3 | 220.02 | 218.68 |
| 4 | 220.91 | 219.41 |
| 5 | 221.38 | 219.70 |
| **mean ms/pass** | **220.04** | **218.81** |

**−1.23 ms, +0.56%.** That is under this card's 1.3% thermal drift taken alone,
which is exactly why it was measured five times alternating rather than twice:
the change wins **every pair**, and five for five is a one-in-thirty-two
coincidence. The drift moves both arms of a pair together; it does not order
them.

It is a small win and it is worth being clear about why it is not a large one.
Occupancy was the cheap half of the diagnosis — two resident blocks to three is
50% more latency to hide with — and it bought half a percent. The expensive
half is still open: 45% of peak on a kernel whose loads are already vectorised
and whose shared layout is already conflict-free points at the access
*pattern*, not the instruction mix. Each warp reads eight rows `hidden` elements
apart, 64 contiguous bytes from each, so DRAM sees 64 interleaved streams per
block and hundreds across the card. Widening `MOE_MMA_KC` from 128 to 256 would
double each row's contiguous run from 112 bytes to a full 224-byte superblock
at the cost of a wider staged tile; that is the next thing to try, and it is a
real change to the fragment and scale indexing rather than a hint.

## Head to head after the scan: 1.136x (2026-08-16)

Three alternating rounds of `bench_forward` n=512 (4 reps) and
`llama-bench -p 512 -n 0 -r 3 -ngl 99`, back to back on the same card in the
same session, same GGUF:

| round | llmxabe | llama.cpp |
| --- | ---: | ---: |
| 1 | 2,334.93 ± 20.64 | 1,998.44 ± 107.77 |
| 2 | 2,332.04 ± 12.37 | 2,088.79 ± 60.51 |
| 3 | 2,328.40 ± 10.04 | 2,069.22 ± 51.24 |
| **mean tok/s** | **2,331.8** | **2,052.2** |

**1.136x**, winning every round. The previous measurement of this pair, taken
the same way earlier in the session, was **1.011x** on 2,099.3 against 2,076.2
— "a small repeatable win, not a headline". Deleting the chunking for prefill
is what moved it, and the two MoE `__launch_bounds__` added the last half
percent.

The caveat that applied at 1.011x still applies to the *method* and no longer
to the *conclusion*: llama.cpp's own `pp512` spreads ±50–108 tok/s run to run
here, and this card drifts about 1.3%. A 13.6% margin is several times both,
and the arms were alternated within minutes so the drift applies to each
equally. llmxabe's own spread is ±10–21, tighter than llama.cpp's by a factor
of three to five.

Decode is unchanged at **104.24 tok/s**, above the 100 tok/s floor the goal
sets for it. Prefill was the half of the goal that was behind; it no longer is.

## Context sweep and parallel sequences (2026-08-17)

The request was `-np 3` at 32K / 64K / 96K. Two of the three axes are things
this engine cannot do at all today, and that is the headline rather than a
footnote.

### llmxabe has no parallel-sequence path, and caps at 6,144 tokens

**`-np 3` does not exist here.** There is no cross-sequence batching and no
serving surface; every benchmark below is a single sequence. That is the
already-tracked gap, not a new discovery.

**32K, 64K and 96K are also out of reach.** Two separate limits, found in that
order:

1. A geometry guard refused any context past about **7,168 tokens** — it
   compared the dispatch *slot* count against the 65,535 `grid.y` limit when
   every launch actually puts the dispatch *block* count there, `block_size`
   times smaller. Fixed; see the commit. It was a real units bug.
2. Underneath it, **VRAM binds first at 6,144 tokens**, at 45.13 GiB of the
   card's 47.27. So fixing the guard raised nothing on this hardware today.

The binding constraint is that prefill activations are sized by `max_tokens`
with no chunked-prefill path: the whole sequence is resident at once. 6,144
tokens costs about 14 GiB on top of the weights, or ~2.3 MB per token, which
is far more than the ~8 KB/token a hidden-state row needs — per-layer scratch
is not being shared across layers. That is the thing to fix before any of the
requested context lengths are measurable, and it is a memory-model change, not
a kernel one.

### Where llmxabe actually stands, single sequence

`bench_forward` (cold full forward, no KV cache — comparable to `pp`) and
`bench_decode` (128 steps at depth), 3 rounds each:

| context | llmxabe prefill tok/s | llmxabe decode tok/s |
| ---: | ---: | ---: |
| 128 | 1,298.9 | 102.5 |
| 512 | 2,339.6 | 96.8 |
| 2,048 | **2,633.6** | 77.7 |
| 4,096 | 2,371.3 | — |
| 6,144 | 1,985.9 | — |
| 8,192 | out of memory | out of memory |

Prefill **peaks at 2,048 tokens and falls away**: the MoE weight read is a
fixed cost per pass and amortizes over more tokens up to that point, after
which quadratic attention and memory pressure take it back. Decode goes the
other way and degrades monotonically with depth, 102.5 → 77.7 from 128 to
2,048.

### llama.cpp, same card, `-npl 1` and `-npl 3`

`llama-batched-bench -ntg 128 -fa 1`, 3 rounds at the short depths and 1 at the
long ones. **llama.cpp defaults to all three GPUs**; the 1-GPU column is the
like-for-like comparison against llmxabe and was measured with
`CUDA_VISIBLE_DEVICES=0`.

Prefill tok/s, 3 GPUs:

| context | `-npl 1` | `-npl 3` | batching gain |
| ---: | ---: | ---: | ---: |
| 128 | 859.8 | 1,583.5 | 1.84x |
| 512 | 1,862.9 | 2,515.6 | 1.35x |
| 2,048 | 3,582.3 | 4,286.5 | 1.20x |
| 32,768 | 3,686.3 | 3,677.2 | 1.00x |
| 65,536 | 3,150.8 | 3,125.7 | 0.99x |
| 98,304 | 2,738.0 | 2,711.9 | 0.99x |

Decode tok/s, 3 GPUs — this is where three sequences pay:

| context | `-npl 1` | `-npl 3` | gain |
| ---: | ---: | ---: | ---: |
| 128 | 95.4 | 182.3 | 1.91x |
| 512 | 98.9 | 191.6 | 1.94x |
| 2,048 | 104.2 | 189.1 | 1.81x |
| 32,768 | 81.2 | 121.6 | 1.50x |
| 65,536 | 80.5 | 123.4 | 1.53x |
| 98,304 | 73.3 | 110.0 | 1.50x |

**Batching is a decode feature, not a prefill one.** At 32K and beyond a single
prefill already saturates the card, so three of them together buy nothing;
decode is latency-bound per step and three sequences share the same weight
read, which is worth 1.5–1.9x throughout.

### The multi-GPU split is worth nothing at short context and 2.1x at long

| context | llama.cpp 1 GPU | llama.cpp 3 GPUs | ratio |
| ---: | ---: | ---: | ---: |
| 512 (`llama-bench pp512`) | 2,069.4 | 2,030.2 | 0.98x |
| 32,768 (`-npl 1`) | 1,729.9 | 3,686.3 | 2.13x |
| 65,536 (`-npl 1`) | 1,410.4 | 3,150.8 | 2.23x |

This matters for reading every earlier head-to-head in this document. Layer
split is a sequential pipeline — it adds capacity, not parallelism — so at 512
tokens it is worth nothing, and the **1.136x measured against llama.cpp was
not flattered by the extra cards**: re-measured against a single GPU it is
2,347.4 against 2,069.4, or **1.134x**. At long context the extra cards
matter enormously, and llmxabe has no answer at those lengths at all.

### Summary of the standing

| | llmxabe | llama.cpp (1 GPU) |
| --- | --- | --- |
| prefill @512 | **2,347** | 2,069 |
| prefill @2,048 | **2,634** | 3,582 (3 GPU) |
| prefill @32K+ | cannot run | 1,410–1,730 |
| decode @2,048 | 77.7 | 104.2 |
| parallel sequences | none | 1.5–1.9x at `-npl 3` |
| max context | 6,144 | 98,304+ |

The prefill goal is met and then some at the length it was set at. It is met
*only* at that length: at 2,048 tokens llmxabe is at 0.74x of llama.cpp, and
past 6,144 it does not run. Long-context prefill and decode-at-depth are the
two places the engine is now clearly behind, and both trace to the same
missing piece — a chunked prefill with a shared activation arena.

## Context ceiling: 6,144 -> 24,576 tokens (2026-08-17)

Two changes, prompted by asking why an engine on a 48 GiB card could not
prefill 8K tokens when the weights are 29.8 GiB.

### The MTP head was already excluded — that was not it

Worth stating because it is the first thing to suspect: the GGUF reports
`block_count = 41` for a 40-layer model, and block 40 is the MTP/`nextn` head —
a complete extra attention block with its own 256-expert MoE, about 0.75 GiB.
llama.cpp logs `unused tensor blk.40.* -- ignoring` for all twenty of them.

`WeightSchema::new` already omits it and only the test-only `with_mtp`
constructor includes it, which the profile corroborates: 39 `moe_expert_ffn_mma`
launches plus one `_q8` per pass, not 41. Measured directly, a build at one
token peaks at **32.127 GiB**, of which 2.291 GiB is the weight arena, leaving
~29.8 GiB of weights against a 30.37 GiB file — the MTP head's worth less.

### Ten private attention scratches were 1.68 MB per token

VRAM grew at a very linear **2.05 MB per token of context**, which is enormous
next to the 8 KB a hidden-state row needs. `GdnBlock` is constructed once and
shared across all thirty of its layers; `attention` is a `Vec` of ten blocks,
and each one held its own copy of fourteen scratch buffers inline.

At this geometry `q_dim` is 4,096 — sixteen heads of 256 — and seven of those
fourteen buffers are that wide. One layer's set is 43,008 floats per token, or
**168 KiB/token**; ten of them are 1.68 MB/token, which was more than the Gated
DeltaNet block, the MoE dispatch, and the residual stream put together.

They are pure scratch — written and consumed inside one layer's `forward`,
nothing crossing a layer boundary — and the layers run strictly in sequence, so
one set serves all ten. Extracted into `AttnScratch`, owned by `Forward`, passed
in by reference. Measured per-token VRAM: **2.05 MB -> 0.581 MB**, a 3.5x
reduction, and prefill at 512 tokens is unchanged at 2,347.6 tok/s.

### The 65,535 grid axis, twice

With the memory fixed, 8,192 tokens then failed with a bare
`CUDA_ERROR_INVALID_VALUE`. Two guards were involved and the first correction
was wrong.

The MoE geometry check bounded `sorted_capacity` — the dispatch *slot* count,
`max_tokens * top_k` plus padding — against the 65,535 `grid.y` limit, while
every dispatch launch puts `expert_block_capacity` there, `block_size` times
smaller. Relaxing it to bound the block count looked right and **was not
sufficient**: `MmaKernels::quantize_rows` took `rows` on `grid.y`, and the down
projection passes `sorted_capacity` as `rows`. The old guard was therefore
load-bearing by accident, at exactly the observed boundary — 7,168 tokens gives
65,280 slots and 8,000 gives 71,936.

The fix is at the source. `mma_quantize_rows_q8` now takes the row on `grid.x`,
capped at 2^31-1, and the contraction block on `grid.y`, which is 4 blocks wide
at this geometry. The slot count no longer reaches a 65,535 axis anywhere.

### Where that leaves the context sweep

| tokens | peak VRAM | prefill tok/s |
| ---: | ---: | ---: |
| 512 | 32.85 GiB | 2,347.6 |
| 2,048 | — | **2,646.7** |
| 8,192 | 37.19 GiB | 1,690.9 |
| 16,384 | 41.82 GiB | 1,075.0 |
| 24,576 | 46.44 GiB | 798.6 |

**4x the reachable context, at no throughput cost where it already ran.** The
shape of the curve is unchanged and is the next problem: prefill still peaks at
2,048 tokens and decays quadratically after, because `bench_forward` runs the
whole sequence as one cold pass with no KV cache, so attention is O(n^2) in a
single shot. llama.cpp processes the same prompt in 512-token micro-batches
against a KV cache and holds 1,730 tok/s at 32K on one GPU against llmxabe's
1,691 at 8K and 799 at 24K.

32K remains out of reach on one card by about 8 GiB. The remaining 0.581
MB/token is now spread thinly across the GDN block, the MoE dispatch and the
residual stream rather than concentrated anywhere, so the next step is not
another sharing fix — it is chunked prefill, which bounds activation memory by
the chunk instead of the sequence and fixes the throughput curve at the same
time.

## Why the tok/s curve decays and llama.cpp's does not (2026-08-17)

llama.cpp holds 3,686 -> 3,151 -> 2,738 tok/s from 32K to 96K on three GPUs,
and 1,730 -> 1,410 -> 1,235 on one. llmxabe went 2,647 at 2,048 tokens to 799
at 24,576. Two separate causes, and only the first was the obvious one.

### Chunked prefill was missing, and it was not the main problem

llama.cpp processes a prompt in `-ub`-sized micro-batches against a KV cache.
llmxabe built one pass as wide as the prompt, so activation memory scaled with
the prompt and the whole sequence's attention happened in a single shot.

The engine already supported the alternative: `Forward::run` runs `self.tokens`
positions at `state.position()` and advances the state, which is exactly what a
decode step is at one token. Only `bench_forward` was missing it. Behind
`LLMXABE_BENCH_CHUNK` a prompt is now prefilled as fixed-shape passes over one
carried state, and peak VRAM at 8,192 tokens drops from 37.19 GiB to 33.13.

It did **not** fix the throughput curve, and the per-chunk timings say why:

| prompt | chunks | ms/prompt | ms/chunk | tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 1 | 214.9 | 214.9 | 2,382 |
| 2,048 | 4 | 976.0 | 244.0 | 2,098 |
| 8,192 | 16 | 6,004.4 | 375.3 | 1,364 |

A fixed-shape 512-token pass gets **slower as the context behind it grows** —
215 to 375 ms. Chunking bounds the memory; it does not bound that.

### Attention is 43% of an 8K prefill, and reads K/V eight times over

`nsys` over the chunked 8,192-token run, 10 attention layers of 40:

| kernel | share | total ms | avg ms | min -> max |
| --- | ---: | ---: | ---: | --- |
| **`attn_flash_causal`** | **42.8%** | 5,027 | 15.7 | 0.85 -> 31.65 |
| `moe_expert_ffn_mma` | 17.3% | 2,027 | 1.62 | — |
| `mma_q8_0_proj_split` | 10.7% | 1,256 | 0.16 | — |
| `moe_expert_down_mma` | 10.2% | 1,202 | 0.94 | — |
| `gdn_scan_prefill` | 7.9% | 928 | 0.97 | — |

The min-to-max spread is the whole story: the first chunk's attention is
0.85 ms and the sixteenth's is 31.65 ms, linear in the context behind it, while
every other kernel is flat.

At the last chunk that launch moves **16.4 GB of K/V in 31.65 ms — 77% of the
card's 672 GB/s.** It is bandwidth-bound, and the bandwidth is being spent
eight times over. The grid is `(n_query / ATTN_QT, q_heads)`: one block per
(query tile, **query** head), and each block streams the whole causal K and V
range for itself. This model has **16 query heads against 2 KV heads**, so the
eight query heads that share a KV head each read the same keys and values
independently.

The arithmetic, at the last chunk of an 8K prefill: 1,024 blocks each reading
8,192 keys x 1 KiB of K and the same of V is 16.4 GB, against the 2.05 GB the
same work needs if a block serves all eight query heads of one KV head. The
measured 31.65 ms against a 24.4 ms roofline for 16.4 GB confirms which of the
two the kernel is actually paying.

### Widening the query tile is not the fix — measured

The obvious lever is to amortize over more query rows per block: `ATTN_QT` 8 to
16 with `ATTN_KT` 4 to 2, keeping their product pinned at 32. That halves the
block count and so halves K/V traffic.

It is **slower**: 6,004 -> 6,647 ms on an 8K chunked prefill, and 214.9 -> 217.1
at 512. `qr[ATTN_QT][ATTN_MAXD]` goes from 64 to 128 registers per lane, which
takes the kernel from two resident blocks per SM to one, and half the latency
hiding costs more than half the traffic saves. Reverted.

The register wall is the reason: amortization times `head_dim/32` is the
register count, so 64 registers buys 8-way amortization and 64-way needs 512.
The way past it is not a wider tile but **sharing the K/V tile through shared
memory across the eight query heads of one KV head** — grid `(n_query/QT,
kv_heads)`, K and V staged once per key tile, the head loop inside. That trades
8x the DRAM traffic for 8x the shared traffic, and shared is roughly 19x the
aggregate bandwidth on this part. Estimated 31.65 ms -> 4 ms at the same tile
shape, which would take an 8K prefill from 6.0 s to about 4.0 s and a 32K one
from an extrapolated 49 s to 22 s.

That is the single largest remaining item in the engine and it is a kernel
rewrite, not a constant change. It is specified here rather than attempted
half-way.

## The KV-head-major flash kernel, measured (2026-08-17)

The rewrite specified above landed as `attn_flash_causal_gqa`. The shape is the
one predicted — grid `(n_query / GQA_QT, kv_heads)`, one warp per query head
under that KV head, K and V staged in shared once per trip for all eight — with
one correction: `GQA_QT` is **4**, not 8. A warp now owns a whole head, so a
lane carries `head_dim / 32` accumulator dimensions per row instead of one, and
`qr` plus `acc` is `2 * GQA_QT * ATTN_MAXD` registers. At 4 that is 64, which
ptxas turns into 128 total and two resident blocks per SM; at 8 it is 128, which
is the same wall the `ATTN_QT` 16 experiment hit from the other side. Traffic
falls by exactly `GQA_QT`, so the shape that was reachable buys 4x and not 8x.

Chunked prefill, 512-token chunks, one GPU, one repetition per row:

| tokens | chunks | before (tok/s) | after (tok/s) | gain |
| -----: | -----: | -------------: | ------------: | ---: |
|    512 |      1 |        2,364.3 |       2,367.2 | +0.1% |
|  2,048 |      4 |        2,088.1 |       2,122.1 | +1.6% |
|  8,192 |     16 |        1,359.3 |       1,552.2 | +14.2% |
| 16,384 |     32 |          913.9 |       1,134.6 | +24.1% |
| 32,768 |     64 |          561.6 |         737.1 | +31.2% |
| 65,536 |    128 |          321.2 |         441.1 | +37.3% |

The gain grows monotonically with depth, which is the signature of the
mechanism: attention is a fixed cost per (query, key) pair, so its share of the
pass rises with context, and only attention changed. At 512 tokens — one chunk,
one key window of 512 — there is nothing to win and nothing is lost.

Peak VRAM at 65,536 is unchanged at 35.316 GiB of 47.27, because the KV cache
dominates it and the KV cache did not move.

### Why it is 1.4x on the kernel and not 4x

`nsys` on the 8,192 run: `attn_flash_causal_gqa` is 34.7% of the pass where
`attn_flash_causal` was 42.8%, 3,642 ms against 5,027, and the deepest chunk
falls 31.65 -> 22.09 ms. That is 1.43x from a 4x traffic cut, and the reason is
that the kernel is no longer bandwidth-bound. At the deepest chunk it now moves
4.2 GB in 22.09 ms — 188 GB/s, 28% of this card's 672 — where the old kernel
moved 16.4 GB at 77% of peak. The bound moved somewhere else.

It moved to instruction issue, and specifically to the cross-lane reduction.
Per (key, query row) the kernel spends `head_dim / 32` = 8 multiply-adds and
then 5 `__shfl_xor_sync` plus 5 adds to reduce the dot product across the warp:
40 reduction instructions per 64 useful multiply-adds at `GQA_QT` 4. Measured
throughput is 3.2 TFLOP/s against a 16.3 TFLOP/s fp32 peak, which is what that
ratio predicts once the load/store slots are counted too.

That is a data-layout problem and the layout is a closed trade. With `L` lanes
cooperating on one dot product and `head_dim / L` dimensions per lane, registers
are `2 * GQA_QT * head_dim / L`, the traffic cut is `GQA_QT`, and the reduction
costs `2 * log2(L)` instructions against `head_dim / L` multiply-adds:

| L  | dims/lane | multiply-add share | registers at `GQA_QT` 4 | traffic cut |
| -- | --------: | -----------------: | ----------------------: | ----------: |
| 32 |         8 |                44% |                    64 ✓ |          4x |
| 16 |        16 |                67% |                   128 ✗ |          4x |
|  8 |        32 |                84% |                   256 ✗ |          4x |
|  4 |        64 |                94% |                   512 ✗ |          4x |

Every row that improves the reduction costs registers this part does not have.
fp32 scalar code has no move left here; the next step is fp16 operands on the
`mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` tensor cores, which do the
reduction in hardware and raise the arithmetic ceiling from 16.3 to about 65
TFLOP/s. Attention is two GEMMs and nothing else, so it is the right shape for
that; it is a larger change than this one and is not attempted here.

### Two cheaper fixes, both measured and both rejected

**Swapping the grid axes.** Blocks are scheduled with x fastest, so the original
grid `(n_query / ATTN_QT, q_heads)` makes the co-resident blocks many query
tiles of one head — the worst order for reuse. Putting the head on x instead
makes them the sixteen heads of one query tile, which share only two KV heads,
and costs one line. It **lost 1.7%** across three interleaved pairs at 8,192
(5,932/6,036/6,057 ms against 6,127/6,162/6,171). Sibling blocks start together
but nothing keeps them together, and they drift apart faster than 6 MB of L2 can
span. Only a staging barrier inside one block actually makes them share, which
is why the kernel above stages rather than reorders. Reverted.

**Widening `ATTN_QT` to 16.** Recorded above; also reverted.

### What this did not touch: decode

Decode is `n_query == 1` and still takes `attn_flash_causal_t1`, whose grid is
`(1, q_heads)` — **sixteen blocks on a 72-SM card**. Before counting the same
8x K/V redundancy, decode attention leaves 78% of the machine idle. The fix is
flash-decoding: split the key range across blocks, have each produce a partial
`(m, l, acc)`, and merge them in a second pass. That recovers both the
parallelism and the redundancy at once, and it is the largest remaining item on
the decode side.

## Flash decoding, and the first head-to-head that matters (2026-08-17)

The item the previous section named as decode's largest remaining problem is
fixed. `attn_flash_causal_t1` launched `(n_query, q_heads)`, and decode is
`n_query == 1` — sixteen blocks on a 72-SM card, before counting the eightfold
K/V redundancy. `attn_flash_decode_split` gives each block a slice of the key
window and `attn_flash_decode_combine` merges the partial `(m, l, acc)` triples,
so the parallelism comes from the key axis, which is the only axis decode has.

Interleaved, one GPU, 48 timed steps after 4 warmup, winning every pair:

|   ctx | before | after | gain | llama.cpp (1 GPU) | ratio |
| ----: | -----: | ----: | ---: | ----------------: | ----: |
|   512 |   98.3 | 104.8 | +6.6% |            104.19 | 1.006 |
| 2,048 |   78.3 | 101.4 | +29.5% |          103.91 | 0.976 |
| 4,096 |   61.7 |  98.3 | +59%  |            102.68 | 0.957 |
| 8,192 |   43.2 |  92.6 | +114% |            100.48 | 0.921 |

The shape of the "before" column is the finding. Decode fell 98.3 -> 43.2 from
512 to 8,192 — more than halving — and it now falls 104.8 -> 92.6. That decay
was never arithmetic. A decode step does the same work per key at every depth;
what changed with depth was how long sixteen blocks took to stream a window that
each of them read eight times over.

llmxabe is now **ahead of llama.cpp at 512** and within 8% at 8,192, where it
was at 0.43x before. llama.cpp's own decode is almost perfectly flat
(104.19 -> 100.48 over the same range), so closing the rest of that gap means
matching its flatness, not its peak.

### Prefill, same comparison, same card

llama.cpp is run with its default 512-token micro-batch, which is the chunk
llmxabe uses, so this is like for like:

| tokens | llmxabe | llama.cpp (1 GPU) | ratio |
| -----: | ------: | ----------------: | ----: |
|    512 | 2,367.2 |           2,155.8 | 1.098 |
|  2,048 | 2,122.1 |           2,118.0 | 1.002 |
|  8,192 | 1,552.2 |           1,978.7 | 0.784 |
| 32,768 |   737.1 |           1,729.9 | 0.426 |
| 65,536 |   441.1 |           1,410.4 | 0.313 |

Prefill wins at 512, ties at 2,048, and loses from there. llama.cpp decays only
8% from 512 to 8,192 where llmxabe decays 34%, and the divergence keeps widening
— which is the same attention story as decode, but on the side where the fix
landed only a 4x traffic cut rather than a shape change.

### What is left, and what it is not

It is not memory. A 65,536-position run peaks at 35.316 GiB of 47.27, and the
KV cache is 40 KiB per token, so 131,072 is about 37.8 GiB — it fits on one card
with roughly 9 GiB to spare. The 128K target is bounded by throughput, not
capacity.

It is not the GDN layers, the MoE, or the projections: `nsys` has them flat
across depth, and 30 of the 40 layers are GDN, whose recurrent state is a fixed
2 MiB regardless of context.

It is attention's arithmetic efficiency, and specifically the cross-lane
dot-product reduction, measured at 3.2 of this card's 16.3 fp32 TFLOP/s. The
lanes-per-dot-product trade tabulated in the previous section has no remaining
row that fits in the register file. The next lever is fp16 operands on
`mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32`, which does the reduction in
hardware and raises the ceiling to roughly 65 TFLOP/s; it pairs naturally with
an fp16 KV cache, which would also halve the 5 GiB the cache costs at 131,072.

### The occupancy trade is closed too, measured (2026-08-17)

Before reaching for tensor cores it is worth knowing whether the GQA kernel is
short of arithmetic or short of latency hiding. The instruction mix says the
former should not be binding: per key per warp it issues 20 shared loads, 64
multiply-adds and 40 shuffle-plus-add, and on this part's 4-instruction issue
against 2 fp32 warp-instructions per clock that mix bounds at roughly 12.4
TFLOP/s. Measured is 3.2. The missing 4x is latency, not instructions.

Two knobs were swept at 8,192 tokens in 512-token chunks, two pairs each.

**Staged tile width `GQA_KT`.** 8 is the optimum and both neighbours lose:

| `GQA_KT` | shared/block | blocks/SM | tok/s (p1, p2) |
| -------: | -----------: | --------: | -------------: |
|        4 |      9,216 B |         2 |  1490.1, 1456.0 |
|        8 |     17,408 B |         2 |  **1571.5, 1539.3** |
|       16 |     33,792 B |         1 |  1339.5, 1331.1 |

16 halves the barrier count and doubles the compute between barriers, and still
loses 14% — because 33,792 B admits only one block per SM. That is the direct
measurement that **occupancy is what binds**, and it is why the answer is not a
bigger tile.

**Register cap.** ptxas lands the kernel at 128 registers, which at 256 threads
is exactly two blocks of the 65,536-register file. `__maxnreg__` was used to try
to buy a third:

| cap | registers | spill | tok/s (p1, p2) |
| --: | --------: | ----: | -------------: |
|  — |       128 |     0 | **1571.5, 1539.3** |
|  96 |        96 |  16 B | 1559.5, 1545.7 |
|  84 |        84 |  32 B | 1461.5, 1454.2 |

84 registers would fit three blocks in the register file (`256 * 84 * 3 =
64,512`) and it changes nothing, because 17,408 B of shared still admits only
two. A third block needs **both** ≤85 registers and ≤16,384 B of shared, and the
only thing that buys the second is fp16 staging — which then costs a convert per
element on the read side, and which by this table's own slope is worth 10-15%,
not the 3-4x the remaining gap needs.

So the fp32 kernel is at its local optimum on both axes. `GQA_KT` 8 and no
register cap are kept, and neither knob is worth revisiting without first
changing the arithmetic.

## 128K on one GPU, measured end to end (2026-08-17)

The capability claim, run rather than projected. One Quadro RTX 8000, chunked
prefill at 512 tokens, a genuine 131,072-position sequence state:

```
prompt 131072 tokens, prefill 510967.4 ms (256.5 tok/s)
  mean          23.40 ms      42.74 tok/s
sequence state 5.063 GiB for 131108 positions; peak VRAM 38.137 GiB of 47.27
```

**The model runs at 128K on a single card with 9.1 GiB to spare.** The KV cache
is 5.063 GiB of that, which is the 40 KiB per token the fp32 layout implies, and
nothing else in the engine scales with context — the 30 GDN layers carry a fixed
2 MiB of recurrent state each regardless of depth.

Against llama.cpp on the same single card, same model file, `-fa 1`:

| at 131,072 | llmxabe | llama.cpp | ratio |
| ---------- | ------: | --------: | ----: |
| prefill tok/s |  256.5 |   1,119.3 | 0.229 |
| decode tok/s  |  42.74 |     67.65 | 0.632 |
| peak VRAM     | 38.1 GiB |       — |     — |

So the capacity half of the target is met and the throughput half is not.

### Why, stated as a roofline and not as a guess

llama.cpp processes 131,072 tokens in 117.1 s. Its own 512-token rate on this
card is 2,155.8 tok/s, so the 256 chunks cost about 60.7 s of everything that is
not attention, leaving roughly 56.4 s for attention. Attention over that prompt
is 1,413 TFLOP:

    10 layers * 4 * 512 queries * 16 heads * 256 dims * sum(window)
    sum(window) = 512 * (1 + 2 + ... + 256) = 16.84e6

1,413 TFLOP in 56.4 s is **25 TFLOP/s**. This card's fp32 peak is 16.3. A number
above the fp32 peak cannot be reached by any fp32 kernel, however written, so
this is not a tuning gap — llama.cpp is running attention in fp16 on the tensor
cores. `ggml/src/ggml-cuda/fattn.cu:461` says so directly:

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {

head_dim here is 256, so this model takes `fattn-mma-f16.cuh`. (The note in
`docs/KERNELS.md` that pointed at `fattn-tile.cu`/`fattn-vec.cuh` "not the WMMA
path" was wrong for this geometry and has been corrected.)

llmxabe's attention runs at 3.2 TFLOP/s, 20% of the fp32 peak. llama.cpp's runs
at 25 TFLOP/s, 38% of the 65 TFLOP/s that `m16n8k8` with fp32 accumulation
offers. **Both are at an ordinary fraction of their respective ceilings; the
ceilings differ by 4x.** That is the whole remaining gap, and no amount of fp32
scheduling closes it — the two preceding sections measured that trade closed on
the reduction axis, the occupancy axis, and the tile-width axis.

### What would close it, and what it costs

Porting attention to `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32`. It
takes fp16 operands and accumulates in fp32, so the rounding is confined to the
inputs rather than to the 256-term dot product — materially better than the
packed-`half2` alternative, which would need fp16 accumulation and would breach
the tolerance this repo's differential tests are gated at.

It requires an fp16 K/V cache, which is a separate benefit: it halves the 5.063
GiB above and halves the DRAM traffic the decode path is currently limited by.

It is a rewrite rather than a change, and the honest estimate from the roofline
is that it is necessary but perhaps not sufficient on its own: 4x the ceiling at
the same 20% utilization would be about 12.8 TFLOP/s against llama.cpp's 25, so
the tile structure has to improve alongside the arithmetic. That is specified
here rather than attempted half-way.

## Tensor cores, and where the head-to-head stands now (2026-08-17)

The previous section named the gap as a ceiling — llama.cpp running attention
at ~25 TFLOP/s where this card's fp32 peak is 16.3 — and said closing it meant
porting attention to `m16n8k8`. That is done, together with the binary16 KV
cache it wants. One GPU, chunked prefill at 512, against llama.cpp on the same
card and model file with `-fa 1`:

### Prefill

| tokens | session start | now | llama.cpp | ratio |
| -----: | ------------: | --: | --------: | ----: |
|    512 |       2,364.3 | **2,436.1** |   2,155.8 | **1.130** |
|  2,048 |       2,088.1 | **2,311.4** |   2,118.0 | **1.091** |
|  8,192 |       1,359.3 | 1,957.0 |   1,978.7 | 0.989 |
| 16,384 |         913.9 | 1,622.6 |         — |     — |
| 32,768 |         561.6 | 1,204.7 |   1,729.9 | 0.696 |
| 65,536 |         321.2 |   810.3 |   1,410.4 | 0.575 |
| 131,072 |            — |   505.8 |   1,119.3 | 0.452 |

### Decode

|     ctx | session start | now | llama.cpp | ratio |
| ------: | ------------: | --: | --------: | ----: |
|     512 |          98.3 | **104.4** |    104.19 | **1.002** |
|   2,048 |          78.3 | 102.2 |    103.91 | 0.984 |
|   8,192 |          43.2 |  94.9 |    100.48 | 0.944 |
|  32,768 |             — |  76.6 |         — |     — |
| 131,072 |             — |  45.1 |     67.65 | 0.666 |

**llmxabe is now ahead of llama.cpp at 512 and 2,048 tokens of prefill and at
512 of decode, and within 2% at 8,192 prefill and 2-6% at 2,048-8,192 decode.**
It is still behind from 32K up, by 1.4x at 32K prefill and 2.2x at 131,072.

At 131,072 on one card: 505.8 tok/s prefill, 45.1 decode, **peak 35.637 GiB of
47.27**, sequence state 2.562 GiB. Prefill there is 2.06x what it was at the
start of this work and the residency headroom grew from 9.1 to 11.6 GiB.

### What moved, in order

| change | where measured | effect |
| ------ | -------------- | -----: |
| KV head on the grid, K/V staged in shared | 8,192 prefill | +13.3% |
| split-K flash decoding | ctx 8,192 decode | +114% |
| whole tile of loads issued before any store | ctx 32,768 decode | +7.9% |
| `DECODE_SPLITS` 64 -> 144 | ctx 32,768 decode | +1.8% |
| `m16n8k8` tensor cores | 8,192 prefill | +13.3% |
| binary16 KV cache | 8,192 prefill | +9.9% |
| two cached dimensions per decode load | ctx 32,768 decode | +6.5% |

### Rejected, with numbers, so they are not tried again

- Swapping the flash grid's axes for L2 reuse: **-1.7%**, three pairs. Sibling
  blocks drift apart faster than 6 MB of L2 spans.
- `ATTN_QT` 8 -> 16: **-10%**. `qr` doubles to 128 registers, halving resident
  blocks.
- `GQA_KT` 16: **-14%**. 33,792 B of shared admits one block per SM.
- `__maxnreg__(84)` to chase a third block: **-6%**, 32 B of spill, and the
  third block never appears because shared still admits two.
- One query head per tensor-core block: **-6%** against the scalar kernel. The
  MMA fixed the arithmetic and re-broke the traffic.
- Serial per-row softmax in the tensor-core kernel: **-9%**. 240 of 256 threads
  idle between two barriers once per 32 keys.
- Holding the Q fragments in registers across the key loop: **no win**, and 146
  registers. `Q K^T` was not shared-bound after all.

### What still separates 32K-128K

Attention at 65,536 is about 54 s of an 80.9 s pass, which is 6.6 TFLOP/s
against llama.cpp's 22.1. Neither DRAM (16.5 s of traffic at peak bandwidth)
nor the tensor cores (5.5 s at peak) nor shared memory (3.7 s) accounts for 54,
so what is left is latency: the kernel needs 57.4 KiB of shared and therefore
runs **one block per SM**, eight warps, with four `__syncthreads` per 32-key
tile and no second block to cover them. Two blocks per SM needs 32 KiB, and the
staged Q, K and V tiles do not fit twice at this shape.

The two ways out are a fourth reduction in traffic — four query heads per block
rather than two, which halves the block count again but needs a different warp
mapping and a 64-register output accumulator — and named barriers, so the two
head groups stop waiting on each other at the two barriers that are per-head
rather than per-block. Neither is a constant change; both are specified here
rather than attempted half-way.

## The staging loops were the bottleneck, not the barriers (2026-08-17)

The section above blamed the 32K-128K gap on latency the kernel could not
cover: one block per SM, four `__syncthreads` per 32-key tile, no second block
to hide them. That was an elimination argument with an unmeasured premise, and
it was wrong. Attention was not close to any compute bound at all — it was at
36% of this card's memory bandwidth, and every explanation involving tensor
cores or barriers was reasoning about the wrong limit.

What made that visible was measuring attention on its own.

### `bench_attention`

Every attention measurement before this one came out of a whole forward pass: a
30 GiB model load and a minute of wall clock per A/B, expensive enough that the
honest response to a small kernel change was to not measure it. `cargo run
--release -p xabe-engine --bin bench_attention` runs the attention launch and
nothing else, at this model's real geometry, across seven context depths. An A/B
is about six seconds.

It reports the traffic the launch *issues* — one pass over the causal window per
block, summed per query tile, taken from the dispatch rule rather than a copy of
it — next to the traffic the problem needs. The redundancy the block shape
imposes is then a column rather than an inference, and it turned out to be the
number that mattered most.

### Four changes, measured at 131,072 keys

| change | ms | GB/s | issued GB |
|---|---:|---:|---:|
| starting point | 141.18 | 243.9 | 34.43 |
| K staged as `uint4` | 119.06 | 289.2 | 34.43 |
| ... V batched in place | **152.32** | 226.0 | 34.43 |
| ... V batched, division removed | 92.81 | 371.0 | 34.43 |
| `MMA_HPB` 2 -> 4, `MMA_KT` 32 -> 16 | 78.65 | 219.5 | **17.21** |
| per-head named barriers | 70.60 | 243.8 | 17.21 |
| software-pipelined prefetch | **62.16** | 276.9 | 17.21 |

**2.27x on attention alone**, and the bandwidth column is now flat across depth
where before it drifted.

#### 1. Memory-level parallelism

Both staged-tile loops are bounded by `hd2`, a runtime value. ptxas cannot
unroll a loop whose trip count it does not know, so each iteration's shared
store depended on the global load directly above it and each thread kept **one
4-byte request in flight**.

Little's law fixes the requirement: an SM's share of 672 GB/s at 1.4 GHz is
6.7 B/cycle, and at roughly 600 cycles of DRAM latency that is about 4 KiB in
flight per SM, or 16 B across 256 threads. One word per thread is a quarter of
that, predicting 25% of peak against the 36% measured — the right size, with the
slack being L2 hits among the blocks sharing a KV head.

K is the easy half: two adjacent binary16 dimensions of one key are already the
packed `B` operand, so a `uint4` is four operands in one request, and `qstride`
is a multiple of four so the matching 128-bit shared store is quarter-warp
phased and conflict-free.

V cannot take the same trick. A `uint4` there is eight consecutive dimensions,
which land eight *rows* apart in `v_sh`; the fragment read needs
`gcd(vstride, 32) == 4`, and `8 * vstride` is then a multiple of 32 for every
admissible stride, so all 32 lanes would hit one bank. Widening the load costs
more in the store than it buys.

The third row of the table is why this is written down. Batching the flattened
`[key-pair][dim]` loop in place **lost 11% against doing nothing**: that loop
needs `idx / head_dim` per element against a runtime divisor, and batching pays
the division `2 * MMA_VB` times a trip instead of once. Putting the dimension on
the thread index and the key-pair on a compile-time-bounded inner loop removes
every division from the staging body, and that — not the batching — is most of
the 92.81. `MMA_VB` swept interleaved: 1 -> 93.96, **2 -> 92.81**, 4 -> 100.76,
8 -> 100.70.

#### 2. Traffic, which is what actually binds

The redundancy column made the next move obvious: the launch issued 34.4 GB
where the problem needs 0.27, because `n_query/16 * q_heads/2` = 256 blocks each
stream the whole prefix. Four query heads per block instead of two halves that,
and the issued figure halved exactly as predicted.

`MMA_HPB`, `MMA_WPH` and `MMA_KT` look like three tunables and are not: the
block is eight warps split evenly among the heads it serves, and `Q K^T` gives
each warp of a head exactly one key octet. So `MMA_HPB * MMA_WPH == 8` and
`MMA_KT == 8 * MMA_WPH`, and `MMA_HPB` 4 forces a 16-key tile. Both relations
now have a unit test, because breaking either still compiles and returns a
finite, plausible, wrong answer.

The 16-key tile doubles the barriers per key, and the bandwidth column shows it:
371 -> 219 GB/s. Halving the traffic was still worth 1.18x net.

#### 3. Named barriers

Two of the four barriers per tile fence `s_sh`, `m_sh`, `l_sh` and `corr_sh`,
all of which are sliced per head — the warps of head A have nothing to say to
the warps of head B between `Q K^T` and `P V`. Those became `bar.sync id, 64`.
Measured interleaved, three pairs: 77.43 / 77.72 / 77.79 block-wide against
70.60 / 70.55 / 70.96 per-head, **+9.6%**.

#### 4. Software pipelining

With one block per SM there is no second block to cover staging latency, so the
DRAM round trip sat between two barriers with the tensor cores idle. The loop is
now pipelined: a trip stores the tile loaded during the previous trip's
arithmetic, then issues the next tile's loads, then computes. Nothing between
the issue and the following barrier touches the prefetch registers, so they stay
in flight across `Q K^T`, the softmax and `P V`.

Both register arrays are indexed by compile-time constants with the real trip
count as a *predicate* rather than a bound — indexed by a runtime value they
would land in local memory and the change would be worse than useless.
**62.16 ms, and bandwidth back to 276.9 GB/s.**

### Where the head-to-head stands

Prefill, one GPU, 512-token chunks, against llama.cpp `-fa 1 -ub 512` — **its
default, not its best; see the correction at the end of this file**:

| tokens | before | after | llama.cpp | ratio |
|---:|---:|---:|---:|---:|
| 512 | 2,436.1 | 2,437.7 | 2,155.8 | **1.13x** |
| 2,048 | 2,311.4 | 2,346.2 | 2,118.0 | **1.11x** |
| 8,192 | 1,957.0 | 2,114.7 | 1,978.7 | **1.07x** |
| 32,768 | 1,204.7 | 1,630.8 | 1,729.9 | 0.94x |
| 65,536 | 810.3 | 1,227.4 | 1,410.4 | 0.87x |
| 131,072 | 505.8 | 855.2 | 1,119.3 | 0.76x |

8,192 crossed over; 131,072 went from 0.45x to 0.76x.

### What is left, with the arithmetic that says so

With the prefetch in place, reverting only the block shape measures 111.03 and
111.88 ms at `MMA_HPB` 2 against 61.35 and 61.65 at 4 — **1.81x from halving
traffic**, with bandwidth roughly flat (310 against 280 GB/s). The kernel is
still traffic-bound, so the next factor is another halving of the block count
and not more latency hiding.

Taking the per-depth attention cost from `bench_attention` and integrating it
over a chunked prefill puts attention at about 80 s of the 153 s pass at
131,072, leaving roughly 74 s that is not attention. Beating llama.cpp there
needs the whole pass under 117 s, so attention has to reach about 43 s: a
further 1.85x. At 32,768 attention is only about 5 s of 20 s, so the same 1.8x
would put that depth at roughly 1,834 tok/s — ahead.

The way to it is `MMA_HPB` 8, and the obstacle is exact: `q_sh` alone would be
67,584 B against a 65,536 B carveout. It fits only if Q leaves shared for
registers, which costs 64 registers of fragments plus a 128-register
accumulator, because one warp would then own a whole head's 16x256 output.
About 217 registers against a 255 limit — feasible on paper, and the spill is
the risk. An XOR swizzle removes `q_sh`'s padding and was checked
(`(4s+tg) ^ 4*(r&7)` keeps the fragment read a bank bijection) but saves only
1,024 B, which does not change the answer.

## The baseline was llama.cpp's default, not its best (2026-08-17)

Every llama.cpp figure above this line was taken at its **default** `-ub 512`.
That is not its best. Given `-b 8192 -ub 4096` it is substantially faster, and
its best setting is prompt-dependent -- at pp512 the wide ubatch *hurts* it, so
"best settings" has to mean best-per-length on both sides or the comparison is
rigged in whichever direction the tester prefers.

| prompt | llama.cpp ub 512 | ub 1024 | ub 2048 | ub 4096 |
|---:|---:|---:|---:|---:|
| 512 | 2155.8 | | 1808.0 | 1808.8 |
| 2048 | 2118.0 | | 2925.8 | 2956.9 |
| 8192 | 1978.7 | | 2862.3 | 3081.6 |
| 32768 | 1729.9 | | 2307.9 | 2462.8 |
| 65536 | 1410.4 | | 1881.3 | 1939.8 |
| 131072 | 1076.6 | 1231.2 | 1364.4 | 1396.5 |

Note also that its pp131072 at the default measures 1076.6 here against the
1119.3 recorded earlier in this file. At matched 512 that is parity, not the
0.95x the older row implies.

### Head to head, both sides best-per-length

| prompt | llmxabe | chunk | llama.cpp | ub | ratio |
|---:|---:|---:|---:|---:|---:|
| 512 | **2437.7** | 512 | 2155.8 | 512 | **1.13x** |
| 2048 | **3102.7** | 2048 | 2956.9 | 4096 | **1.05x** |
| 8192 | 3041.9 | 8192 | 3081.6 | 4096 | 0.99x |
| 32768 | 2367.8 | 8192 | 2462.8 | 4096 | 0.96x |
| 65536 | 1831.7 | 8192 | 1939.8 | 4096 | 0.94x |
| 131072 | 1276.6 | 8192 | 1396.5 | 4096 | 0.91x |

Decode, measured with `llama-bench -d <depth>` rather than interpolated from
the endpoints, which is what an earlier estimate in this file did:

| depth | llmxabe | llama.cpp | ratio |
|---:|---:|---:|---:|
| 0 | ~91 | 96.1 | 0.95x |
| 32768 | 80.1 | 88.5 | 0.91x |
| 65536 | 62.2 | 79.7 | 0.78x |
| 131072 | 47.4 | 66.9 | 0.71x |

Three parallel sequences, `llama-batched-bench -npl 1,3`, 32,768-token prompt:

| | prefill aggregate | decode aggregate | decode per-sequence |
|---|---:|---:|---:|
| llama.cpp -np 1 | 2404.7 | 87.98 | 87.98 |
| llama.cpp -np 3 | 2313.1 | **154.58** | 51.53 |
| llmxabe (one sequence) | 2304.5 | 77.4 | 77.4 |

Both readings are true and they point opposite ways. **Aggregate decode is a
2x loss**: llama.cpp's continuous batching nearly doubles throughput across
three sequences and this engine has no cross-sequence batching at all --
`Forward::run` takes one `SequenceState` -- so its aggregate is its
single-stream number. **Per-sequence decode is a 1.5x win**: each of their
three sequences runs slower than our one. For a serving benchmark the
aggregate is the standard measure, and by it we lose.

### Claims made earlier in this session that did not survive

- 1.04x at 32,768 and 1.02x at 65,536: against the default ubatch only.
- 1.03x at 32,768 after widening our chunk to 8192: died the moment llama.cpp
  was given the same wider batch.

Widening the batch has favoured llama.cpp three times running, and that is the
finding rather than the accident. Its prefill scales with batch width and ours
does not: our attention is **flat per token** at every width measured -- 42.1 ms
at chunk 512, 179.5 at 2048, 205 -> 193 GB/s -- so every gain we get from a
wider chunk comes from MoE and GDN. Attention is 53% of the 128K pass.

### What each remaining gap needs

- **Deep prefill, +4% to +9%.** Attention is no longer traffic-bound. At
  `MMA_HPB` 8 its arithmetic intensity is 128 FLOP/B against this card's 96.7
  balance point, and it runs at ~40% of the fp32-accumulate tensor peak. Every
  prefill gain in this session came from moving fewer bytes and that lever is
  spent. Untried: `ldmatrix.sync.aligned.m8n8.x4` (sm_75 has it) to replace one
  scalar shared load per mma step, and `mma.m8n8k4`, Turing's native shape --
  `m16n8k8` is emulated as four of them and the emulation may be the 60%.
- **Deep decode, +41%.** See the warp-split kernel's commit for the four
  hypotheses already eliminated. `half2` is the next one.
- **Aggregate decode, +100%.** A missing feature, not a slow kernel.

## Corrections to the three items above

All three were written before the experiments they propose were run. Kept
rather than deleted, because what a wrong prediction was based on is worth as
much as the number that replaced it.

**The tensor peak was measured, and it is not what that paragraph assumes.** A
standalone microbenchmark on this card gives `mma.m16n8k8` **97.6 TFLOP/s** and
`mma.m8n8k4` **49.1 TFLOP/s**. This Quadro does not halve the fp32-accumulate
rate, so prefill attention runs at **27%** of the real ceiling, not 40% -- and
`m8n8k4`, Turing's "native" shape, is half the throughput rather than the
hidden headroom the paragraph guessed at. Ruled out without writing the kernel.

**`ldmatrix` was implemented and rejected on measurement.** Both fragment
layouts were verified on hardware first (plain gives the A-fragment order,
`.trans` the transpose, so V could carry K's layout and its staging collapse to
one 16-byte load feeding one 16-byte store). Correct, and **0.8% slower**:
42.37/42.37/42.40 ms against 41.97/42.06/42.09. The kernel is not bound by
shared-load instruction count, so replacing four scalar loads with one buys
nothing and the address computation costs a little.

**"Flat per token as the batch widens" was not a defect.** 512 -> 2048 is 4x
the work and measured 4.26x the time. That is ordinary scaling. The inference
that llama.cpp gains a wider-batch reuse we lack does not follow from it.

**A two-deep staged K/V tile lost by 15%** (48.26 vs 41.97 ms), which rules out
memory-level parallelism as the prefill constraint -- doubling the bytes in
flight made it worse.

## The decode key load was never the `uint4` its comment claimed

`attn_flash_decode_warp` indexed K and V as `kp[wpl * lane + p]`. Across the
warp that is a stride of `wpl` words -- 16 bytes at head_dim 256 -- so each of
the four loads touched all sixteen 32-byte sectors of the 512-byte key and took
four bytes from each, and the four loads then requested the same sixteen
sectors again.

`cuobjdump -sass` settles it without a profiler, which matters here because
`ncu` fails with `ERR_NVGPUCTRPERM` on this machine and cannot be used at all:

| kernel | `LDG.E.128` | scalar `LDG.E` |
|---|---:|---:|
| `attn_flash_causal_mma` (prefill) | 2 | 129 |
| `attn_flash_decode_warp`, before | **0** | 73 |
| `attn_flash_decode_warp`, after | **2** | 73 |

The prefill kernel in the same file vectorises, so the compiler does it when
the pattern permits; this pattern did not permit it. `wpl` comes from the
runtime `head_dim`, so neither `wpl == 4` nor the alignment is provable at
compile time, and the per-element `p < wpl` predicate blocks vectorisation on
its own.

Naming the width restores the wide load. Lane `l` takes bytes `[16l, 16l+16)`,
which is the same eight binary16 dimensions it read before as four strided
words -- **bit-identical data**, which the golden logits test confirms.

| | ms @ 131,072 | GB/s |
|---|---:|---:|
| before | 1.285 | 208.9 |
| after | **1.188** | **226.0** |

Three interleaved pairs, spread under 0.3%. **1.082x.**

### What it did not do

Sector amplification predicted a **4x** traffic-efficiency recovery. It
delivered **1.08x**: L1 was absorbing nearly all of the redundant sector
requests. The diagnosis was real and the SASS proved it; the magnitude was
wrong, and decode attention is still at only 33.9% of roofline.

Both earlier null results were re-measured under the fixed load, since both had
been taken under the broken one and might have meant something different:

| `DEC_SPLITS` | 288 | 576 | 864 | 1152 |
|---|---:|---:|---:|---:|
| ms | **1.179** | 1.242 | 1.290 | 1.335 |

| `DEC_KB` | 1 | 2 | 4 |
|---|---:|---:|---:|
| ms | **1.180** | 1.360 | 1.400 |

More parallelism hurts and more memory-level parallelism hurts, at 33.9% of
roofline. Occupancy and latency-hiding are both excluded, together, which is an
unusual pair. The decode gap remains unexplained after six hypotheses.

One datum was under-read for a long time and is worth re-stating: the
`DEC_SPLITS` sweep varies resident warps 3.5x at *fixed total work* and moves
the time 1.7%. That excludes a latency bound as firmly as a parallelism one.
It sat in the rejected pile labelled "no effect" because nothing then explained
it.

## `__expf` in the prefill softmax: faster, and rejected on accuracy

Substituting `__expf` for `expf` in `attn_flash_causal_mma` measured **1.075x**
on attention in isolation (39.03/39.21/39.20 against 42.00/42.06/42.29 ms) and
passed all nine attention differential tests.

It fails `the_forward_pass_reproduces_llama_cpps_logits_and_its_argmax`, which
was confirmed to be cause and not coincidence by stashing the change and
re-running: green at HEAD, red with it. At rank 4 llama.cpp separates the two
candidates by 0.153701 while the two implementations differ by at most 0.120058
on a shared logit -- a real ranking error, not a tie inside the noise. The
argmax still agrees and leads by 39x the noise, so greedy decoding would not
have noticed; sampling would.

The reasoning that justified it -- that a 2^-21 relative error is invisible
next to the 2^-11 already accepted by rounding K and V to binary16 -- is
plausible and was wrong in effect. The end-to-end gain was never confirmed
above the harness's 13% run-to-run spread either, so this traded a measured
accuracy regression for an unmeasured speedup. Not shipped.

## `exp2f` in the prefill softmax: an identity, not an approximation, and it lands (2026-08-17)

`__expf` above is a genuine approximation; this is not. `expf(x)` is
`exp2f(x)` plus range reduction, so scaling the score by `scale * log2(e)`
before the softmax and using `exp2f` throughout computes the same function:
`exp2(s_i - max_j s_j) == exp(x_i - max_j x_j)` when `s = x * log2(e)`, for
every `i`. Weights, the running normalizer and the rescale factors are all
unchanged; only the units of the running max differ, and the multiply the
scale rides on was already being paid.

Three `bench_attention` pairs, interleaved, expf/exp2f mean of three, ms:

| key_offset | expf | exp2f | speedup |
|---:|---:|---:|---:|
| 0 | 0.255 | 0.240 | 1.06x |
| 2,048 | 1.123 | 1.043 | 1.08x |
| 8,192 | 3.735 | 3.478 | 1.07x |
| 32,768 | 13.93 | 13.26 | 1.05x |
| 65,536 | 23.23 | 20.56 | 1.13x |
| 98,304 | 31.03 | 28.97 | 1.07x |
| 131,072 | 41.59 | 38.90 | 1.07x |

`exp2f` won every one of the three rounds at every depth. 5-13%, largest
around 65,536 where the softmax loop's share of the tile is largest relative
to the traffic already amortised by the prefetch.

Correctness was held to the same bar as the `__expf` attempt precisely
because "identity in exact arithmetic" is not "identity in floating point" --
`exp2f` and `expf` round differently on the same hardware `MUFU.EX2`. All nine
`attention_differential` tests pass, and the golden-logits test was run both
ways by stashing the change: `the_forward_pass_reproduces_llama_cpps_logits_and_its_argmax`
gives the **identical** winning logit (20.017208 against llama.cpp's
19.902241) and the **identical** top-8 order and rank-4 noise-floor swap
(239784/78229, separation 0.153701 against 0.205439 of implementation noise)
with or without the change. Bit-identical output on the one prompt this repo
checks against llama.cpp, which is the strongest correctness signal available
here short of `ncu` (unavailable on this host).

Applied only where it was measured to help. On `attn_flash_decode_warp` it
measured **3.4% slower** (1.227 vs 1.188 ms interleaved) despite emitting 80
fewer instructions for the same 16 `MUFU.EX2` and unchanged occupancy -- a
standalone instruction-count model had predicted a 15% gain there and omitted
the `part_acc` writeback, so it was not a faithful model of the kernel.
Decode keeps `expf`; only `attn_flash_causal_mma` changed.

Landed as a separate commit from the rest of this session's decode work,
scoped to the three hunks inside `attn_flash_causal_mma` and its `ATTN_LOG2E`
`#define`.

## `half2` on the `P V` accumulator: fewer registers, fewer instructions, 11.3% slower (2026-08-17)

The named next hypothesis for decode's 33.9%-of-roofline ceiling was `half2`
arithmetic: K and V are stored binary16, `attn_flash_decode_warp` converts
both to fp32 before using them, and the value path -- `acc[hh][i] += e *
vv[i]`, `ATTN_MAXD` scalar FMAs per head per key -- looked like it could run
as `ATTN_MAXD/2` packed `f16x2` FMAs straight against the `uint4` load's
already-packed word, with no `h2f2` unpack at all.

Built exactly that: the accumulator became `unsigned accp[DEC_MAXG][ATTN_MAXD
/ 2]`, `vw`'s packed words went into `fma.rn.f16x2` (via inline PTX,
`hfma2_raw`) unconverted, the rescale multiply used `mul.f16x2`, and the
per-key scalar softmax weight `e` was broadcast into a packed word with the
`pack_h2` helper the tensor-core kernel already had. K stayed fp32 -- it
feeds the score the softmax exponentiates, and the `exp2f` result two
sections up already found that budget too tight for a coarser unit.

Three rebuild-and-measure rounds each, alternating, `LLMXABE_ATTN_CHUNK=1`
(`bench_attention`'s decode mode -- see its module doc, added this session),
`key_offset = 131,072`, ms:

| round | before | after |
|---:|---:|---:|
| 1 | 1.198 | 1.333 |
| 2 | 1.199 | 1.334 |
| 3 | 1.199 | 1.334 |

**11.3% slower**, and consistent to three significant figures across rounds
in both directions -- not noise.

That was the surprising result, so it was checked against `ptxas -v` and
`cuobjdump -sass` (offline, via `nvcc -arch=sm_75` on the extracted kernel
source -- NVRTC's cache is in-process only, see `kernels::mod::compile`) two
ways, both pointing the same direction as the timing rather than away from
it:

| | registers | static instructions | spills |
|---|---:|---:|---:|
| before | 195 | 1,536 | 0 |
| after | **155** | **1,424** | 0 |

Fewer registers, fewer instructions, no spilling -- by the accounting that
predicted the change, it should have won. `cuobjdump -sass` shows what the
instruction count alone did not: 32 `HFMA2` + 32 `HMUL2` replaced FMAs as
designed, but `PRMT` went from 0 to 40 and `F2F.F16.F32` from 0 to 16 --
`pack_h2`'s broadcast of the scalar softmax weight into a `half2` costs real
instructions on a value that was already sitting in a register for free in
the scalar version, and the accounting that only compared FMA counts never
saw it. Where exactly that cost lands -- issue-slot contention with the
`MUFU.EX2` `expf` already on the critical path is the obvious guess, given
`half2` conversions and transcendentals both tend toward the SFU pipe on
Turing -- is not settled, because `ncu` cannot run on this host and this is
as far as `cuobjdump` and arithmetic go. Not applied; `attn_flash_decode_warp`
is unchanged from before this section except a comment recording the result.

This is now the seventh hypothesis eliminated for the 33.9% ceiling (sector
amplification, `DEC_SPLITS`, `DEC_KB`, occupancy, `exp2f`, and now `half2`,
alongside the grid/parallelism fix that did land). Register pressure and
static instruction count both point away from what actually happened, which
argues for retiring instruction-counting as a predictor for this kernel
specifically and treating every future decode change as a measurement
question from the start rather than an arithmetic one.

## What actually costs at a decode step, measured instead of assumed (2026-08-17)

The brief for this session named the real open question directly: at ctx 512
attention is small, so the 0.95x depth-0 ratio against llama.cpp is not
attention's doing, and closing it needs whatever *is* costing at that depth.
Guessing at that from the kernel source would repeat the mistake the `half2`
section above just made. `nsys` answers it directly, with one nsys-specific
trap: decode runs as a captured CUDA graph (`Forward::capture_step` /
`StepGraph`, `forward.rs`), and `nsys`'s default `--cuda-graph-trace=graph`
collapses a whole replayed graph into one opaque node -- the per-kernel
summary comes back nearly empty (ten `attn_flash_causal_gqa` instances, the
one prefill call, and nothing that looks like 20 decode steps). Passing
`--cuda-graph-trace=node` expands every replay back into its constituent
kernels and the picture becomes normal. (A second trap, cheaper to fall into:
this host's locale renders `nsys stats`' human-readable table with 4-digit
thousands grouping rather than 3 -- `3672,2526` is 36,722,526 ns, not
3,672.2526 -- so read the `--format csv` output, or `LC_ALL=C`, not the
table.)

`bench_decode 8 16`, prefill 8 tokens then 4 warmup + 16 timed decode steps at
context 12..28 (99.1 tok/s), profiled whole:

| kernel group | us / decode step | share |
|---|---:|---:|
| MoE (routed + shared experts, gemv path) | ~2,000 | ~21% |
| GDN (linear-attention mixer, recurrent path) | ~2,300 | ~25% |
| LM head (`lm_head_gemv_b1`) | ~1,620 | ~17% |
| **attention** (`attn_flash_decode_warp` + `_combine`) | **474** | **~5%** |
| everything else (norms, gates, routing, argmax) | rest | ~32% |

Every one of those kernel names was checked against its instance count before
being called decode-exclusive: `attn_flash_decode_warp` shows exactly `steps
* 10` instances (10 is this model's count of full-attention layers out of 40
total -- the rest are `gdn_*`), with zero contribution from the prefill call,
in both this profile and a second one taken at context 8192..8204. That
second profile is the reason the table above is trustworthy rather than a
one-depth coincidence: the same clean kernel set gives attention **9.92%** of
a decode step at depth 8192 against **5.08%** at depth ~20 -- growing, as it
must, while MoE, GDN and the LM head do not (none of them read the KV cache).

Extrapolating with the numbers already in hand rather than a third profile:
`bench_attention`'s decode mode gives `attn_flash_decode_warp` +
`_combine` alone, one attention layer, 1.198 ms at `key_offset = 131,072` (the
uint4-load section above). Ten layers: 11.98 ms. The non-attention part of a
decode step is ~8.6-9.7 ms and does not grow with depth, so a step at
131,072 should cost roughly 8.6 + 11.98 ≈ 20.6 ms, or **~48 tok/s** --
against the **47.4 tok/s** this file already recorded from `llama-bench -d
131072`, a 2% miss from a two-measurement extrapolation. Attention's share at
that depth is therefore about **55%** of a decode step, not the ~5% it is at
depth 0.

That is the whole shape of the problem stated in one sentence: **the 0.95x
gap at depth 0 is a MoE/GDN/LM-head question, and the 0.71x gap at 131,072 is
an attention question**, and the two gaps needing different owners is not a
coincidence -- it is what "attention is the only term whose cost grows with
context" (established for prefill earlier in this file) also means for
decode. MoE and GDN kernels are out of this session's scope (owned
elsewhere, and `crates/xabe-engine/src/block/gdn.rs` had a sibling's
in-progress edit in this same working tree while this was measured), so
nothing here changes them. What it does change is confidence: `DEC_SPLITS`,
`DEC_KB`, occupancy, `exp2f` and now `half2` were all tested against
*attention's own* 33.9%-of-roofline ceiling on the assumption that closing it
matters, and at 131,072 keys it is now measured to matter more than half the
decode step -- so the ceiling stays the right thing to keep chasing, even
though this session did not find the eighth hypothesis that explains it.

## The MoE decode GEMV, re-measured: the lever named for this session was already pulled (2026-08-17)

The follow-up brief for this section quoted `bench_moe`'s decode-shape (n=1)
number as 129.4 GB/s, 19.3% of the 672 GB/s peak, against `lm_head_gemv_b1`'s
53%, and asked for a `llama.cpp mmvq.cu`-style pass at
`moe_expert_ffn_gemv`'s Q6_K unpack. That number is stale. `bench_moe 1`,
three rounds, GPU 1:

| round | GB/s (unique) | % peak |
|---:|---:|---:|
| 1 | 261.6 | 38.9% |
| 2 | 262.2 | 39.0% |
| 3 | 260.7 | 38.8% |

**38.9%, double the quoted figure**, and `crates/xabe-cuda/src/kernels/moe.rs`
already explains why: `763bc69` padded Q6_K to a 224-byte device stride and
widened the dequant loads to `uint4`, and the kernel's own comments record
`moe_expert_ffn_gemv` (Q6_K gate/up) individually at 47% of streaming
roofline and `moe_expert_down_gemv` (Q8_0) at 63%, against `lm_head_gemv_b1`
at 82-89% depending on which pass measured it. `bench_moe`'s 38.9% is the
*combined* `grouped_forward` pass — both projections plus the reduce launch —
which is why it sits below either kernel's own number. None of this was
known to whoever wrote the follow-up brief because it predates this commit;
it is recorded here so the next reader does not chase an already-closed gap.

### What was actually tried: hoisting `d * scale` out of the four-element unpack, and finding ptxas got there first

The kernel's own comment already diagnoses `moe_expert_ffn_gemv` as bound by
the **integer pipe**, not DRAM, at this shape: Q6_K's unpack costs about nine
integer instructions an element against Q8_0's near-zero. The one piece of
that unpack that looked untried was `q6k_value(d, sc, si, raw)`: called once
per element in the four-wide tile, it recomputes `d * (float)sc[si]` fresh
each time even though `d`, `sc` and `si` do not depend on which of the four
elements is being unpacked, and `sc[si]` is a one-byte load at a
lane-scattered address (four lanes to a scale group, not coalesced) — a real
cost to pay four times if the compiler does not prove the loads identical and
hoist it itself.

Hoisted it by hand: `q6k_value` now takes the `ds = d * (float)sc[si]`
product directly rather than `d`, `sc` and `si` separately, computed once
before the four-element loop instead of inside it. Same multiply order,
`(d * scale) * q`, so the result is bit-for-bit what it was — confirmed by
`cuobjdump -sass` on both versions, extracted the same way the decode
attention section above did (`nvcc -arch=sm_75` on `MOE_SRC` pulled out of the
Rust source):

| | registers | static instructions |
|---|---:|---:|
| before | 47 | 856 |
| after | 47 | 856 |

**Byte-identical SASS.** `diff` on the disassembly of `moe_expert_ffn_gemv`
finds nothing at all. `ptxas` was already doing exactly this hoist —
`sc[si]` not depending on the loop variable is provable from the source as
written, and the compiler proved it. `bench_moe 1` after the change measured
260.1-261.1 GB/s against 260.7-262.2 before, the same three-round spread as
noise. Not applied; the source keeps the more legible `q6k_value(d, sc, si,
raw)` form since taking it apart bought nothing.

### The next lever is not a free one: it needs a new rounding source

`llama.cpp`'s own Q6_K decode-shape path —
`vec_dot_q6_K_q8_1_impl_mmvq` in `ggml/src/ggml-cuda/vecdotq.cuh` — is worth
reading before naming it a target, because it is not the access-pattern
change it looks like from the outside. It computes the six-bit codes exactly
as this kernel does (`(vl >> 4i) & 0x0F0F0F0F` merged with a shifted high-bit
field, then `- 32`), and then reduces four of them against four activation
values in one `ggml_cuda_dp4a` instruction rather than four scalar FMAs. The
catch is what `u[i]` is: not the fp32 hidden state this kernel reads, but
`bq8_1[...].qs` — the activation vector **pre-quantized to Q8_1**, int8 plus
a per-block scale, by a separate kernel (`quantize_q8_1`) llama.cpp runs once
per token before every expert's vec_dot touches it.

That is an algorithm change, not a kernel change: this repo's decode GEMV
currently carries the token's activation at full fp32 precision all the way
through the dot product, and every other rounding source in the model — Q6_K
and Q8_0 for the weights, binary16 for the KV cache — was chosen and measured
against the golden-logits gate one at a time. Quantizing the activation too
adds a new one, on the one tensor that has not yet had any (it is read fresh
from the previous layer's fp32 output every step), and there is no
predicting from arithmetic alone whether it survives rank 4 the way `exp2f`
did and the decode `half2` attempt above did not. Building the quantize
kernel, wiring it through both GEMVs, and clearing it against
`moe_differential.rs` and the golden logits test properly is more than this
session's remaining budget allows to do at the rigor the rest of this file
holds itself to, so it is named here rather than attempted. `dp4a` against a
quantized activation is the concrete next step for whoever picks this back
up; the access-pattern and instruction-count levers this session could reach
safely are the ones already landed.

### End to end: the depth-0 ratio this session was asked to close is already closed

`bench_decode 1 32` (GPU 1, four rounds, context 5..37) against a freshly
re-run `llama-bench -d 0 -n 32 -r 5` on the same GPU and the same file --
not the 96.1 tok/s this document had on record, which predates all of the
work between here and there:

| | tok/s |
|---|---:|
| llmxabe, four rounds | 101.98, 100.39, 93.11, 101.66 (mean 99.3) |
| llama.cpp, three 5-rep means | 99.69 ± 2.64, 99.15 ± 2.89, 96.82 ± 5.47 (mean 98.6) |

**~1.01x, roughly even and within both sides' own run-to-run spread.** The
0.95x this file recorded for depth 0 was real when it was measured; it is not
the current state. The uint4 decode key load, the MoE Q6_K word-wide unpack,
the prefill `exp2f` and whatever landed in `gdn.rs` and `forward.rs` this
session closed it cumulatively, none of them aimed at this ratio specifically.
The follow-up brief's premise -- MoE-gemv bandwidth as "the biggest identified
lever for the 0.95x at depth 0" -- no longer has a gap to be the lever for.
Nothing in `moe.rs` changed this session; this section exists so the next
reader starts from 38.9% and ~1.0x rather than re-deriving them.

## The 33.9% ceiling explained: `attn_flash_decode_warp` was never bandwidth-bound (2026-08-17)

Seven hypotheses were eliminated for this number without finding what it
actually is: sector amplification (real, worth 1.08x, not the ceiling),
`DEC_SPLITS`, `DEC_KB`, occupancy, `exp2f`, `half2`, and the grid/parallelism
fix that did land. Every one of them assumed the kernel was DRAM-bound and
asked how to move fewer or better-arranged bytes. None of them questioned
the assumption. It was wrong.

### The combine pass is not blending the number

`nsys`, `attn_flash_decode_warp` and `attn_flash_decode_combine` timed
separately at `key_offset = 131,072`, seven back-to-back launches (the depth
`bench_attention`'s sweep ends on):

| kernel | mean ms |
|---|---:|
| `attn_flash_decode_warp` | 1.1706 |
| `attn_flash_decode_combine` | 0.0374 |

Combine is 3.1% of the pair. Recomputing the roofline fraction from the split
kernel's own time alone moves it from 33.1% (blended) to 34.1% (split only) --
not the explanation. Combine has its own inefficiency worth naming briefly,
because the question was asked: its grid is `(q_heads,) = (16,)` -- 16 blocks
on a 72-SM card, and every one of its 256 threads independently re-reads the
same 288-entry `part_m`/`part_l` arrays rather than staging them once. Both
are true and neither matters here: combine moves under 5 MB total against the
split pass's 268 MB.

### Issued traffic equals necessary traffic, exactly, by construction

The flash-decoding split assigns each of the 288 slices a disjoint
`[begin, end)` range with `begin = split * per`, `end = min(begin + per,
n_visible)` and no overlap, so the last (short) slice aside, every key is read
by exactly one split once. `bench_attention`'s own "issued GB" and "min GB"
columns already print the same number for this kernel (0.268 both) because
`traffic()` special-cases `splits_the_key_axis` for exactly this reason. There
is no hidden re-read to find; the denominator in "33.9% of roofline" was
already the right one.

### The calibration ladder: the access pattern reaches 90%, the arithmetic is what costs

A 20-line kernel with `attn_flash_decode_warp`'s exact grid
(`DEC_SPLITS, kv_heads`), exact per-split key range, and exact `uint4` load of
one key's K and V -- and nothing else, an XOR into a sink to stop the loads
being optimized away -- calibrates what this access pattern can reach on this
card, at `n_visible = 131,073`, three rounds:

| variant | GB/s | % of 672 |
|---|---:|---:|
| load only | 604.7 / 604.8 / 604.6 | **90.0%** |
| + 8-head dot product + 5-step shuffle reduce, no softmax | 586.0 / 586.1 / 586.9 | 87.2% |
| + full online softmax (`expf`, rescale, accumulate), compile-time geometry | 342.8 / 361.9 / 361.9 | ~53% |
| + full online softmax, `head_dim`/`kv_heads` as runtime args (matching the real kernel's signature) | 306.5 / 329.5 / 329.6 | ~48% |
| `attn_flash_decode_warp` itself | 226.6 / 226.6 | **33.7-34.1%** |

This is decisive on its own terms: **the access pattern is not the ceiling.**
A kernel that does nothing but the identical loads reaches 90% of streaming
roofline over the identical binary16 KV layout -- no layout migration, no
K/V interleaving change, nothing about *how the bytes sit in memory* is
costing 66 percentage points. The dot product and its five-step
`__shfl_xor_sync` reduction, run eight times per key for the eight query
heads sharing this KV head, cost almost nothing -- 90% to 87%. What collapses
it is the online-softmax machinery riding on top: `expf`, the
rescale-guarded correction, and the accumulate loop, take it from 87% to
about half the card. Making the geometry a runtime argument instead of a
compile-time one -- `head_dim` and `kv_heads` are kernel parameters in the
real code, not `#define`s, so the per-key address arithmetic cannot be
strength-reduced as aggressively -- costs another five points. The remaining
gap between the closest calibration (~48%) and the real kernel (~34%) is real
and not fully accounted for; `ncu` would find it in an afternoon and cannot
run on this host. What is accounted for is the *shape* of where 66 of the 66
missing points go: essentially none to the load, essentially none to the
dot product, and the rest to softmax.

### Why this explains the "unusual pair" instead of leaving it stranger

Read against a bandwidth-bound model, `DEC_SPLITS` and `DEC_KB` both being
invariant to resident-warp count and memory-level parallelism at fixed total
work was a contradiction nothing resolved. Read against a **transcendental
throughput** model it stops being one. Turing has a fixed number of SFU units
per SM; `MUFU.EX2` -- what `expf` and `exp2f` both lower to -- issues through
them regardless of how many warps are resident or how many memory requests
are in flight. If that shared, fixed-throughput resource is the actual
bottleneck:

- More resident warps (`DEC_SPLITS` swept 144-576) cannot help, because they
  are all queuing for the same SFU throughput rather than hiding DRAM latency
  that was never the constraint.
- Deeper memory-level parallelism (`DEC_KB` > 1) cannot help for the same
  reason, and can hurt by adding register pressure with nothing to spend it
  on.
- `exp2f` measuring **slower** than `expf` despite issuing 80 fewer
  instructions for the *same* `MUFU.EX2` count stops being a puzzle: if the
  SFU call count is what is bound, the ALU instructions `expf`'s range
  reduction adds around it are close to free, overlapped with the SFU
  pipe rather than competing with it -- so removing them removes cost that
  was never on the critical path, and the small regression is scheduling
  noise around a change that could not have won.

This is offered as the mechanism the calibration ladder points at, not as a
measured certainty -- confirming it precisely needs per-SM issue-slot counters
that `ncu`'s `ERR_NVGPUCTRPERM` puts out of reach on this host. What the
calibration *does* establish without qualification is the negative: it is not
DRAM, not the access pattern, not the dot product.

### llama.cpp does not run this shape of kernel here at all

`ggml_cuda_get_best_fattn_kernel` (`ggml/src/ggml-cuda/fattn.cu`) was read
rather than assumed. At this model's geometry -- `head_dim = 256`,
`gqa_ratio = 8`, a causal mask, binary16 K/V, `n_visible % FATTN_KQ_STRIDE
(256) == 0` at 131,072 -- on a Turing device it takes:

```
turing_mma_available(cc) -> true, head_dim not in {40, 72}
can_use_vector_kernel -> true (head_dim <= 256, % 64 == 0, != 192)
  cc < ADA_LOVELACE, so the Ada-only VEC fast path is skipped
gqa_opt_applies -> true (ratio >= 2, masked, K->ne[1] % 256 == 0, 16-byte strides)
  -> !gqa_opt_applies && n_query == 1 is false, so VEC is not selected either
=> BEST_FATTN_KERNEL_MMA_F16
```

**llama.cpp decodes through its tensor-core prefill kernel, not a vector
GEMV, at this exact shape.** It does not run an `expf`-per-key scalar
online-softmax loop at batch 1 at all -- it never takes that code path here.
The `BEST_FATTN_KERNEL_VEC` GEMV this repo's `attn_flash_decode_warp` most
resembles is llama.cpp's answer for small head dimensions or Ada-class
hardware; on Turing with `head_dim = 256` its own dispatch rule steers away
from it.

Why it can: `fattn-mma-f16.cuh`'s tile is a matrix of `n_query * gqa_ratio`
query rows, and `gqa_ratio` is 8 here regardless of how many *positions* are
being decoded. One token still puts 8 rows on the tensor cores' `Q K^T`,
computes a softmax over a genuine tile the same instruction-efficient way the
`m16n8k8` fragments handle it for prefill, and never runs a single-lane
`expf` in a loop over hundreds of keys per warp. This repo's own
`attn_flash_causal_mma` tiles the *other* axis --
[`MMA_QUERY_TILE`](crates/xabe-cuda/src/kernels/attention.rs) is 16 query
*positions*, and decode has exactly one, so `uses_tensor_cores()` can never
be true for it and the dispatch falls through to the scalar split kernel by
construction, not by a tuning gap.

**This reframes the whole question.** It is not that
`attn_flash_decode_warp` is 2.6x slower than it should be at a fixed
algorithm; it is that llama.cpp does not run this algorithm at this
geometry, and this repo does not yet have the kernel that would let it avoid
running it either.

### The fix, specced and not attempted this session

Tiling decode over the GQA-head axis the way `fattn-mma-f16.cuh` does is a
new kernel, not a tuning knob on the existing one -- the same class of change
as the GQA-shared prefill rewrite earlier in this file, and it is named here
rather than built for the same reason: verifying it against the golden-logits
gate and the differential suite at the rigor the rest of this session held
itself to is bigger than the time this task had left.

- **Shape.** One block per KV head per key-split, `gqa_ratio` (8) query rows
  by `head_dim` (256) columns -- an 8x256 `Q` tile instead of one row. Stage
  it in registers exactly as `attn_flash_decode_warp` already stages `qr`
  today; nothing about loading Q changes.
- **`Q K^T`.** `m16n8k8` needs a multiple of 16 rows to fill a fragment.
  Eight is short by 2x, the same problem `MMA_HPB` solved for prefill by
  putting more *heads* in a block rather than more query rows -- except here
  there is only one KV head's worth of heads to put in, so the fragment
  would run at half occupancy (8 of 16 rows live) unless two KV heads' query
  groups share a tile, which breaks the "one warp reads one key once" reuse
  this kernel's whole design rests on. This is the open design question, not
  a detail: whichever way it is resolved changes the traffic argument this
  file has made for the shape three times already.
- **Softmax.** Once scores exist as an 8-row tile the softmax can run the
  way `attn_flash_causal_mma`'s does -- the per-tile max and normalizer
  computed once across an `MMA_KT`-wide tile rather than tracked per key --
  which is the piece the calibration ladder says is worth having: it
  replaces `n_visible` serial `expf` calls per warp with `n_visible /
  MMA_KT`. Whether the prefill kernel's `exp2f` swap (folded score scale,
  see the section above) carries over is a separate question this session's
  calibration says nothing about -- it was measured *slower* for the
  existing per-key decode loop, and a tiled softmax is different arithmetic
  entirely.
- **`P V`.** Falls out of the same fragment layout prefill already uses --
  `D`'s layout is `A`'s, so the scores feed back into `P V` with no
  transpose, exactly as documented for `attn_flash_causal_mma` above.
- **Combine.** Unchanged; a warp is still a finer split of the same key
  range, and the merge identity `attn_flash_decode_combine` implements does
  not care how the partial was produced.

Expected win, from the calibration ladder rather than a guess: closing most
of the 90% (access) to 34% (measured) gap by removing the per-key `expf`
loop plausibly lands this kernel near where `attn_flash_causal_mma` already
sits on the compute side -- call it 2-2.5x on `attn_flash_decode_warp`
itself, which at ~55% of a 131,072-depth decode step (established two
sections up) is roughly a 1.3-1.5x step-level win and would put the 0.71x
ratio at that depth within reach of parity. Not attempted this session; the
shape above and the fragment-occupancy question are the starting point for
whoever does.

## Batched decode across sequences: correct, and short of the aggregate target (2026-08-17)

The number this project has never had an answer to: **aggregate decode
across parallel sequences.** The two sections above measure single-stream
decode getting faster; this section is the first attempt at the structural
gap next to it -- `Forward::run` takes one `SequenceState`, so this engine's
aggregate has always equaled its single-stream number while llama.cpp's
continuous batching very nearly doubles it. `docs/OPTIMIZATION.md`'s R2 names
the mechanism (an M dimension in MoE, a per-sequence state index in GDN, a
per-sequence block table in attention) and estimates a 2.6-4.6x per-token
byte reduction; this is the first measurement of what that mechanism is
actually worth end to end.

### What landed

`Forward::run_batch_decode` / `capture_batch_step` / `replay_batch_step`
advance `N` independent one-token sequences in a single pass, with the same
CUDA-graph capture `capture_step` already gives single-stream decode.
Weight-bound work batches across all `N` sequences in one launch each --
`GdnBlock::forward_batch_decode` and the new
`GatedAttentionBlock::forward_batch_decode` both split their steps into "no
state, batches" (every projection, every norm, the output gate, the
residual) and "reads a position or a sequence's own cache, loops" (GDN's
causal convolution and delta-rule update; attention's rotary, key/value
append, and causal read). Nothing outside `xabe-engine` changed: the
per-sequence loops call the exact same kernel entry points single-stream
decode already exercises, in `crates/xabe-cuda/src/kernels/attention.rs` and
`gdn.rs`, unmodified.

Correctness is two differential tests in `tests/batch_decode.rs`, deliberately
asking two different questions at two different tolerances:

- **`identical_prompts_in_one_batch_produce_bit_identical_rows`** is the
  exact, load-bearing check. Two sequences given the same prompt inside the
  same batch call run through literally the same kernels; the only thing
  that can differ is which memory address each read from. Measured: **0.000e0**
  max-abs difference between the two rows, at every step -- no indexing
  defect, no state crossing a sequence boundary.
- **`batched_decode_agrees_with_independent_single_stream_decodes`** compares
  against `N` independent single-stream runs, and is *not* exact: a batch of
  `N > 1` takes a different compiled Q8_0 projection kernel (tiled) than
  single-stream decode's one-token kernel, and the two are not required to
  round identically. Measured: 8.631e-5 max-abs logit disagreement against
  activations of magnitude ~10, cosine 1.000000000, argmax always agrees.
- **`a_captured_batch_step_generates_the_same_sequence_as_the_launch_path`**
  is `graph_decode.rs`'s gate one level up: the captured graph must generate
  the *identical* sequence the uncaptured launch path does. It does, bit for
  bit, across 3 steps and 3 sequences.

The existing golden test
(`the_forward_pass_reproduces_llama_cpps_logits_and_its_argmax`) and the
single-stream `decode.rs`/`graph_decode.rs` suite are untouched and stay
green -- batched decode is new methods, not a changed code path for the
sequence-of-one case.

### Measured aggregate, GPU 2, `bench_decode_batch`

Synthetic distinct prompts (so a cross-sequence indexing bug would change
the answer rather than hide behind identical inputs), 32 decode steps timed
after 4 warmup, `single_stream` via `capture_step`/`replay_step` and
`batch N` via `capture_batch_step`/`replay_batch_step` -- both graph-captured,
so the comparison is kernels against kernels, not against host dispatch
overhead.

**2,048-token context:**

| shape | mean ms/step | aggregate tok/s | per-sequence tok/s |
| --- | ---: | ---: | ---: |
| single_stream (N=1, baseline) | 9.86 | 101.4 | 101.4 |
| batch 1 | 11.17 | 89.6 | 89.6 |
| batch 2 | 29.10 | 68.7 | 34.4 |
| batch 3 | 39.95 | 75.1 | 25.0 |
| batch 4 | 49.34 | 81.1 | 20.3 |
| batch 8 | 47.60 | **168.1** | 21.0 |

**32,768-token context, the shape `docs/BENCHMARKS.md`'s existing head-to-head
uses:**

| shape | mean ms/step | aggregate tok/s | per-sequence tok/s |
| --- | ---: | ---: | ---: |
| single_stream (N=1, baseline) | 11.86 | 84.3 | 84.3 |
| batch 1 | 13.32 | 75.1 | 75.1 |
| batch 2 | 33.77 | 59.2 | 29.6 |
| batch 3 | 46.79 | 64.1 | 21.4 |
| batch 4 | 59.31 | 67.4 | 16.9 |
| batch 8 | 64.57 | **123.9** | 15.5 |

(`single_stream` here reads 84.3 against the 77.4 recorded in the head-to-head
section above; the gap is the intervening decode-kernel work in this
session's earlier sections, re-measured incidentally by running the same
baseline again.)

### Against the target, honestly

The target was **≥1.7x single-stream at N=3, ideally beating llama.cpp's
154.58.** Neither happened. At N=3, aggregate is 64.1 tok/s -- **below**
single-stream's 84.3, a regression, not a win. Batching only overtakes
single-stream at N=8 (123.9 vs 84.3, a genuine 1.47x), and even there it
falls short of both the 1.7x bar (143.3) and llama.cpp's -np3 154.58.
`AGENTS.md` asks for what was measured, not what was hoped, so: **this
session's batched decode does not clear the target at the shape the target
was set for.**

### Why N=2-4 cost more per sequence than N=1, diagnosed rather than guessed

Two rounds of fixes landed before these numbers, and both are real,
measured wins over the naive first version -- they are why N=8 beats
single-stream at all. Neither closed the gap at N=3.

**First: `GdnBlock`'s Q8_0 projections took the tiled kernel unconditionally
for any `tokens > 1`.** `gdn_proj_q8_0_t8`/`_t16` are built for prefill,
where a chunk is typically far wider than the tile; at a *decode* batch of
2-3, the tile is 25-38% full and the guarded path pays close to a full live
tile's register and occupancy cost for a fraction of its lanes' worth of
answer. Measured with `nsys --cuda-graph-trace=node`: `gdn_proj_q8_0_t8`
averaged **142.5 us/call** at batch width 2, against the untiled kernel's
single-digit microseconds. Fix: `GdnBlock::project_per_sequence` loops the
untiled kernel once per sequence whenever `tokens < PROJ_TILES[0]` (8),
trading `N` separate weight reads for `N` efficient ones instead of one
inefficient shared one. This is the improvement that took a first,
uncaptured version of this feature from *worse than the naive per-sequence
loop it was replacing* to something worth capturing at all.

**Second: `GatedAttentionBlock::forward` was called once per sequence in
full, including its four Q8_0 weight reads (~29 MB/layer), which is exactly
the redundant-read pattern batching exists to remove.**
`GatedAttentionBlock::forward_batch_decode` batches those the same way GDN's
projections do -- `LmHeadKernels` already compiles one exact kernel per
token tile from 1 to 8, so unlike GDN there was no guarded-tile penalty to
find, and the four projections batch cleanly at any `N` in range.

**What remains, measured rather than assumed:** `nsys --cuda-graph-trace=node`
over the last ~600 kernels of a captured N=3 replay shows the GPU **98.6%
busy** (17.17 ms busy of a 17.42 ms span) -- so the remaining cost is not
host dispatch overhead sneaking past the graph capture, and it is not an
inter-kernel dispatch gap either. It is real GPU time spent across a large
number of small kernels: a batched decode step launches on the order of
1,000-1,300 device operations (30 GDN layers x ~16 launches, 10 attention
layers x ~9 batched + `4N` per-sequence, 40 MoE layers x ~15 launches, plus
embedding/norm/head/argmax), and at `N` = 2-4 many of those -- MoE's
`moe_expert_ffn`/`moe_expert_down` chief among them -- are not yet known to
be free of the same small-batch tile inefficiency `GdnBlock`'s projections
had. `docs/OPTIMIZATION.md`'s R3 (landed 2026-08-16) fixed MoE's redundant
weight reads at prefill widths and named `BLOCK_SIZE_M` 1-2 as the
*correct* choice for decode batches specifically, but this session did not
verify that the landed kernel actually took that tile at `N` = 2-8 rather
than a prefill-tuned wider one -- that is the first thing a follow-up should
check, with the same `nsys --cuda-graph-trace=node` methodology this section
used to find `GdnBlock`'s equivalent defect.

### What a follow-up needs, in order

1. **Audit `moe_expert_ffn`/`moe_expert_down`'s tile width at `N` = 2-8**,
   the same way this session found and fixed `GdnBlock`'s. If MoE is paying
   a guarded-tile tax at small `N` the way GDN's projections were, this is
   plausibly the largest remaining lever -- MoE is 40 of 40 layers, GDN's
   fix touched 30.
2. **Fused, device-side-indexed multi-sequence kernels for the loops that
   remain** -- GDN's causal convolution and delta-rule update, attention's
   rotary/append/read. Each is currently `N` separate launches against `N`
   separate state/cache pointers; a single kernel indexing an on-device
   array of those pointers (`AGENTS.md` rule 5's own anticipated pattern)
   would turn `N` small launches into one, which the 1,000+-launch figure
   above says matters at this batch width even under CUDA graph capture.
   This is real kernel work in `xabe-cuda`, coordinated with whoever owns
   `attention.rs` at the time.
3. **Continuous batching, not fixed-`N` batching.** This session's `N` is
   fixed at `Forward` construction, matching the "block per shape" precedent
   the rest of this codebase already uses for prefill vs. decode -- a real
   scheduler needs a batch width that changes step to step as requests
   arrive and finish, which means either a family of pre-built shapes
   (`N` = 1, 2, 4, 8, ...) selected per step, or a genuinely dynamic launch
   shape, which is a different problem from anything `AGENTS.md` rule 5 has
   solved for this codebase yet.
4. **Mixed prefill and decode in one step** (`docs/OPTIMIZATION.md`'s R4) is
   still completely separate work: this session's batches are pure decode,
   `N` sequences each contributing exactly one token, and chunked prefill is
   not wired into the same pass at all.
5. **KV cache pooling.** Every sequence in this session's benchmark holds its
   own fixed-size `KvCache`, allocated for the whole run up front -- fine for
   a controlled benchmark, not what a server admitting and evicting requests
   needs. That is `xabe-cache`'s two-group pager, not this workstream's.

## The GQA-head-tiled decode kernel: built, and reverted on the fp16 `Q` it requires (2026-08-17)

The previous section specced `attn_flash_decode_mma` -- tile the GQA-head
axis into `m16n8k8`'s M dimension the way `fattn-mma-f16.cuh` does for this
geometry, replacing the per-key `expf` loop with a tiled softmax. It got
built this round: a fourth kernel alongside `attn_flash_decode_warp` /
`_split` / `_combine`, wired into `decode()`'s dispatch behind a
`disable_decode_mma()` A/B lever mirroring `MoeKernels::disable_tensor_cores`.
It does not ship. Two independent reasons, both measured, neither a
tolerance to move.

### Shape, as built

One block per `(split, kv_head)`, same grid `attn_flash_decode_warp` and
`_split` already use -- `(DEC_SPLITS, kv_heads) = (288, 2)`, 576 blocks. Four
warps (128 threads) per block, `DMMA_KT = 32` keys staged per trip. The
occupancy fork the previous section named as open -- pad `gqa_ratio` (8) to
`m16n8k8`'s 16-row minimum, or pack two KV heads' query groups into one
tile -- resolved by argument rather than measurement, because the packing
alternative isn't slower, it's impossible: the `B` operand is the keys, which
differ per KV head, so two KV heads cannot share one `Q K^T` tile, and there
is no reduction axis to fold two `mma` calls into one instead. That leaves
padding: `a0` carries the 8 real GQA rows, `a1` is hardcoded to `0u` --
never read from shared memory, so the dead 8 rows contribute exactly zero
with no uninitialized-memory risk -- and the softmax and final write were
generalized (mirroring `attn_flash_causal_mma`'s `rat`/`krow`/`kcol` split)
to cover any `DMMA_KT`, so the occupancy knob (`DMMA_WPO`, currently 4) can
move without a rewrite. Verified by standalone `nvcc -arch=sm_75 -cubin
--ptxas-options=-v`: 121 registers, 0 spill stores, 0 spill loads, unchanged
across the generalization. `attn_flash_decode_combine` was not touched.

### First reason: it is 1.7x slower than the kernel it replaces

`bench_attention` in decode mode (`LLMXABE_ATTN_CHUNK=1`), `DMMA_WPO = 4`:

| key_offset | ms | GB/s | TFLOP/s |
|---:|---:|---:|---:|
| 32,768 | 0.575 | 116.8 | 0.93 |
| 65,536 | 1.083 | 123.9 | 0.99 |
| 98,304 | 1.522 | 132.2 | 1.06 |
| 131,072 | 2.023 | 132.7 | 1.06 |

Against `attn_flash_decode_warp`'s 1.1706 ms at 131,072 keys (measured two
sections up, same card, same method): **1.73x slower**, not faster. Working
diagnosis, not yet confirmed with `nsys` -- shared memory is 38,496 bytes
(37.6 KiB) per block at `DMMA_WPO = 4`, computed from the same
`qstride`/`vstride` formula `shared_bytes_mma` uses for prefill. That fits
under the 48 KiB default carveout with no `set_attribute` opt-in, but two
resident blocks would need 75.2 KiB against Turing's ~64 KiB/SM shared-memory
budget -- one block, four warps, is plausibly all that fits. The kernel
stages K then V with a blocking `__syncthreads()` before computing, the way
`attn_flash_causal_mma` did before it grew `MMA_PREFETCH`'s double-buffered
prefetch; at four warps and one block/SM there is nothing else resident to
hide that round trip behind, which `attn_flash_causal_mma`'s own history
already says matters. The softmax generalization above was written so a
`DMMA_WPO = 2` occupancy experiment (20.8 KiB/block, three blocks/SM) could
be tried without another rewrite. It was not measured -- the second reason
below made it moot before it was worth the GPU time.

### Second reason: it fails the differential gate this session was told not to move

`device_decode_matches_the_reference_over_a_deep_window`, case `n_keys = 61`:

```
differential comparison failed: max_abs_error 1.110882e-5 exceeds tolerance 1.000000e-5 at index 254
  full metrics: cosine=1.000000 max_abs=1.110882e-5 @[254] max_rel=3.630314e-2 @[207] non_finite=0/256
  worst element: Some((-0.051015135, -0.051026244)) (candidate, reference)
```

The other two cases in the same test passed under the same 1e-5 `GATE`, but
close to it: `n_keys=4096` max_abs `6.358e-6`, `n_keys=4097` max_abs
`7.659e-6`. All three, including the failure, are far inside `MMA_GATE`
(`8 * F16_HALF_ULP = 3.9e-3`, derived and already accepted for prefill's
tensor-core kernel two sections up) -- `1.11e-5` is `0.0028x` of it. That
is the tell: this is the same fp16-operand rounding `MMA_GATE` exists for,
at the same magnitude, just occurring on the kernel `attention_differential.rs`
still holds to the tighter fp32-reference `GATE`.

The reason it's a new rounding source rather than the existing one: `k` and
`v` are already rounded through binary16 before either kernel sees them
(`as_cached`, present for every decode differential test, including the ones
`attn_flash_decode_warp` already passes at `GATE`) -- that budget is spent by
both kernels equally and is not what moved. `Q` is what changed. Every scalar
decode/prefill kernel in this file keeps `Q` in fp32 through the dot product;
`attn_flash_decode_mma` is the first decode kernel to round it too, because
`mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` has no other operand type
to give it. That is not a implementation gap to close -- it is the
instruction's fixed contract, the same one prefill's kernel already accepts
under `MMA_GATE`. Widening `uses_tensor_cores` to route this kernel's single
decode row to `MMA_GATE` the same way was the first thing tried here, and is
exactly the move this round was told not to make: it would pass the test by
relabeling which gate applies to it, not by making the arithmetic meet the
one the test already holds. Reverted before it reached a commit.

### What happened to the tolerance and the code

Nothing shipped moved. `uses_tensor_cores` is back to exactly what it was --
`n_query >= MMA_QUERY_TILE && self.mma_is_available()`, no decode case added.
The entire `attn_flash_decode_mma` kernel, its `DMMA_*` macros, the
`DECODE_MMA_*` Rust constants, the `decode_mma` field and
`disable_decode_mma` lever, and the three-way dispatch in `decode()` were
removed with `git checkout -- crates/xabe-cuda/src/kernels/attention.rs`,
returning the file to `c8b8d94` (worker-1's landed `MMA_HPB = 8` state) with
zero diff. `attention_differential`'s full nine-test suite was re-run against
that exact state afterward and is green, including the case that failed
above.

### What this leaves for a future attempt

Both reasons trace to the same root: reusing `m16n8k8` for `Q K^T` at decode
requires rounding `Q`, and rounding `Q` is the one thing this session's gate
will not accept for a single decode row, no matter how the surrounding
kernel is tuned. A kernel that kept `Q K^T` scalar (fp32, as
`attn_flash_decode_warp` already does) and used tensor cores only for `P V`
would sidestep the correctness finding, but `P V` was never the expensive
half here -- the calibration ladder two sections up named the per-key
`expf`/rescale loop as the cost, and that loop runs once per key regardless
of which matmul comes after it, so a `P V`-only tensor-core kernel would keep
paying for the exact loop this redesign exists to remove. The tiled-softmax
idea itself is not what failed; tiling it on top of an all-scalar `Q K^T`
(warp-level score reduction instead of `mma`, matching `attn_flash_causal_gqa`'s
row/lane assignment rather than a fragment layout) is the version of this
that was not tried and does not carry the rounding cost -- worth naming for
whoever picks this back up, since it keeps the "softmax per tile, not per
key" win without the operand the current gate rejects.

## `MMA_HPB` 8 was already the current lever, not the next one, and a smaller one was found instead (2026-08-17)

Handed a brief calling `MMA_HPB` 4 -> 8 ("all 16 query heads... the full GQA
group per block") the next unattempted lever, obstacle and all: `q_sh` would
be 67,584 B against a 65,536 B carveout. That obstacle and its resolution --
Q in registers -- are real, and already shipped. `git blame` puts `#define
MMA_HPB 8` at commit `6dced61`, "Put Q in registers, and measure against
llama.cpp's best rather than its default", earlier this same day: 62.16 ->
41.55 ms at 131,072 keys, and the 1.13x/1.05x/0.99x/0.96x/0.94x/0.91x
best-per-length ratios this file already carries are *with* that change, not
without it. Re-deriving it would have cost a session for zero net delta.
What was actually still open, after checking every item the "already
rejected" list named against the code: the exp2f fold above, and one more
step past it.

### The barrier `MMA_HPB` 8 left behind

`MMA_WPH` is `8 / MMA_HPB`. The block-shape comment from the `MMA_HPB` 4 era
still describes "the block is eight warps split evenly among the heads it
serves" and fences `my_s`/`my_m`/`my_l`/`my_c` with `bar_group(hslot + 1,
MMA_WPH * 32)` between `Q K^T` and the softmax, and again between the softmax
and `P V` -- a named `bar.sync` sized to the warps sharing one head. At
`MMA_HPB` 8, `MMA_WPH` is 1: one warp per head, and the two barriers each
synchronize a single warp against writes only that warp made. `bar.sync` is a
block-wide hardware resource that tracks arrivals across warps; there was
nothing left to track.

Replaced with `__syncwarp()` behind `#if MMA_WPH == 1`, falling back to the
named barrier otherwise so the kernel stays correct at any `MMA_HPB` a future
session might dial back down. Three interleaved `bench_attention` pairs,
bar.sync/`__syncwarp` mean of three, ms:

| key_offset | bar.sync | `__syncwarp` | speedup |
|---:|---:|---:|---:|
| 0 | 0.240 | 0.235 | 1.02x |
| 2,048 | 1.043 | 1.017 | 1.03x |
| 8,192 | 3.478 | 3.387 | 1.03x |
| 98,304 | 29.09 | 28.36 | 1.03x |
| 131,072 | 38.97 | 38.11 | 1.02x |

32,768 and 65,536 were too noisy this session to call either way -- one
bar.sync round at 32,768 read 10.9 ms against its other two rounds' 13.25,
a bigger swing than the change itself -- so they are left unclaimed rather
than folded into the average.

Correctness: all nine `attention_differential` tests pass. The golden-logits
test was run in a clean worktree at this change's parent commit plus only
this patch, isolated from unrelated decode work landing in the same file
concurrently this session. Same argmax token; the winning logit moved from
20.017208 to 19.998243, reproduced identically on a second run in the same
worktree -- deterministic, not a race, and consistent with different
instruction scheduling around a lighter-weight primitive changing FMA
contraction on an unrelated float, the same class of harmless drift the
kernel's own comment already documents for the tree-order softmax reduction.
The rank-4 noise-floor swap this file tracks stays on the passing side of its
assertion (separation 0.153701 against 0.308814 of implementation noise, up
from 0.205439 before -- wider, not narrower).

### Both changes, re-measured end to end and against llama.cpp fresh, GPU 0

`bench_forward`'s isolated attention win does not translate one for one into
the whole pass, because attention is a fraction of it. Three git worktrees
at three commits (baseline before this session's two changes, exp2f alone,
exp2f + `__syncwarp`) sidestep the decode work landing concurrently in the
same file rather than risk measuring a moving target:

| tokens | chunk | baseline | +exp2f | +exp2f+syncwarp |
|---:|---:|---:|---:|---:|
| 512 | 512 | 2,442.1 | 2,445.2 | 2,423.1 |
| 2,048 | 2,048 | -- | -- | 3,068.7 |
| 8,192 | 8,192 | 3,031.7 | 3,059.4 | 3,054.8 |
| 32,768 | 8,192 | 2,332.5 | 2,399.2 | 2,437.7 |
| 65,536 | 8,192 | 1,757.1 | 1,794.6 | 1,862.1 |
| 131,072 | 8,192 | 1,280.8 | 1,311.7 | 1,318.8 |

(131,072's two right columns are the mean of two properly interleaved
mid/after rounds, because the first non-interleaved reading of the combined
build came in *below* the exp2f-alone number -- 1,316.9 against 1,337.2 --
which looked like `__syncwarp` costing something end to end despite winning
in isolation. Interleaved, it did not: 1,321.5/1,320.3 then 1,301.9/1,317.3,
a wash inside the session's own drift. The lesson already in this file's
first page -- measure interleaved or the thermal trend measures you --
applies to worktree A/Bs exactly as much as to in-place ones.)

512 and 8,192 barely move, as expected: attention is a small fraction of a
short pass. 32,768 through 131,072 gain 3-6% end to end, roughly a third to
a half of the isolated attention win once diluted by the rest of the pass.

llama.cpp, `-b 8192 -ub 4096` (512 at its own best, `-ub 512`), GPU 0,
measured fresh in this same session rather than trusted from an earlier one:

| tokens | llama.cpp t/s |
|---:|---:|
| 512 | 2,055.1 ± 121 (noisy -- short prompts run within one llama-bench call vary this much on this card) |
| 2,048 | 2,951.2 |
| 8,192 | 3,077.9 |
| 32,768 | 2,506.2 |
| 65,536 | 1,935.7 |
| 131,072 | 1,439.5 |

### Head to head, both sides best-per-length, both sides fresh

| tokens | llmxabe (exp2f+syncwarp) | llama.cpp | ratio | previous ratio |
|---:|---:|---:|---:|---:|
| 512 | 2,423.1 | 2,055.1 | **1.18x** | 1.13x |
| 2,048 | 3,068.7 | 2,951.2 | **1.04x** | 1.05x |
| 8,192 | 3,054.8 | 3,077.9 | 0.99x | 0.99x |
| 32,768 | 2,437.7 | 2,506.2 | 0.97x | 0.96x |
| 65,536 | 1,862.1 | 1,935.7 | 0.96x | 0.94x |
| 131,072 | 1,318.8 | 1,439.5 | 0.92x | 0.91x |

Every depth from 32,768 down moved up, by 1-2 points of ratio, from the two
changes landed this session. None crossed 1.0x. 8,192 is unchanged to two
digits -- its llama.cpp figure barely moved between sessions (3,081.6 ->
3,077.9) and this depth's own gain was small enough (0.8%) to round away
against that. 512's jump to 1.18x is mostly llama.cpp measuring lower this
run (2,055 against 2,155.8 recorded earlier) on a depth where its own spread
is ~120 t/s; treat the two short-prompt rows as noisier than the rest of
this table, not as a session-over-session win.

### What is left

The traffic lever (`MMA_HPB` 8) and the two arithmetic levers found this
session (`exp2f`, `__syncwarp`) are what this file currently knows how to
pull, and pulling all three still leaves 8,192-131,072 short of parity by
1-8%. Every other prefill idea named as untried earlier in this file --
`ldmatrix`, `mma.m8n8k4`, a two-deep staged tile, a grid-axis swap, wider
query or key tiles, `__maxnreg__` -- was already tried and rejected on
measurement before this session started. Closing the remaining gap needs
either a genuinely new idea against the now-compute-bound (not traffic-bound)
kernel, or accepting that llama.cpp's `-ub 4096` prefill is the harder target
of the two and the win is upstream of attention -- in MoE or GDN, which this
session did not touch.

## A Marlin-style prefetch pipeline for `moe_expert_ffn_mma`: built, and rejected on `ptxas -v` (2026-08-18)

The obvious next target once attention was spent: `moe_expert_ffn_mma` and
`moe_expert_down_mma` are 17.3% and 10.2% of an 8,192-token chunked pass, and
the "Integer tensor cores" section above measured the same kernel family's
split-layout GEMM at 16.1% of the card's 198 TOP/s int8 peak and 31% of
bandwidth peak -- no longer traffic-bound, and named "a shared-memory staging
pipeline of the kind Marlin uses" as the way to more of the ceiling.

No bench isolated these two kernels from a full model load, so the first step
was `bench_moe_mma` -- named apart from `xabe-cuda`'s own `bench_moe`, which
covers all four MoE entry points at synthetic weights and batches up to 512;
this one loads one real layer's mixed-quant expert stacks from the GGUF file
and reaches 8,192 tokens, the shape a chunked prefill actually launches
these two kernels at. It times `MoeKernels::grouped_forward` alone.
Baseline, three interleaved rounds:

| tokens | ms | TOP/s |
|---:|---:|---:|
| 512 | 4.22 | 6.11 |
| 8,192 | 37.8-39.3 | 10.5-10.9 |

### The pipeline, built the way `attn_flash_causal_mma`'s was

Turing has no `cp.async`, so `MMA_PREFETCH` in `attention.rs` hand-pipelines
by loading the next tile into registers a trip ahead of the barrier that
lands it in shared, so ptxas can keep the DRAM round trip in flight across the
current tile's MMA passes. The same transform applied to `moe_expert_ffn_mma`:
every weight and activation staging write got a register-held prefetch one
`kc` trip ahead, landed after the loop's first barrier, with the next trip's
loads issued right after -- structurally identical to the attention kernel's
own loop, two barriers per trip in both the before and after versions.

It compiles, is bit-for-bit the same transform (nothing changes about *what*
is computed, only *when* it is loaded), and **passes every gate**: all six
`moe_differential.rs` tests, including
`device_grouped_forward_matches_the_reference_on_real_expert_weights` at its
existing `ROUTED_MMA_GATE` tolerance with no loosening.

It is also slower. Three interleaved `bench_moe_mma` pairs against the baseline
above:

| tokens | baseline ms | +prefetch ms | change |
|---:|---:|---:|---:|
| 512 | 4.22 | 5.05 | **20% slower** |
| 8,192 | ~38.3 | ~39.2 | **~3% slower** |

### Why, found with `nvcc -Xptxas -v` rather than guessed

`moe_expert_ffn_mma` is declared `__launch_bounds__(MOE_MMA_WARPS * 32,
MOE_MMA_BLOCKS_PER_SM)` with `MOE_MMA_BLOCKS_PER_SM` 3 -- the kernel's own
comment says why: "This kernel is bandwidth-bound, not compute-bound... what
it needs from the scheduler is loads in flight, and what puts loads in flight
is resident warps." Three blocks of 256 threads at Turing's 65,536-register
file is an 85-register-per-thread budget, and extracting `MOE_SRC` to a
standalone `.cu` and compiling it with `nvcc -arch=compute_75 -code=sm_75
-Xptxas -v` shows the **unmodified** kernel already at 80 of those 85
registers, zero spill. The prefetch version compiles to the *same* 80
registers -- `__launch_bounds__` caps it there -- but now with **24 bytes of
spill stores and 24 of spill loads**: the extra live state (roughly 25
registers across the weight and activation prefetch) does not fit, and ptxas
is forced to spill to local memory rather than exceed the occupancy target.
Spilling in the hot loop is exactly the DRAM-latency cost the prefetch exists
to hide, paid a second time.

Relaxing the target to `MOE_MMA_BLOCKS_PER_SM` 2 (128 registers available)
removes the spill entirely -- 128 registers used, zero spill -- and closes
most of the gap: three more interleaved pairs, baseline against prefetch+2
blocks/SM:

| tokens | baseline ms | +prefetch, 2 blocks/SM | change |
|---:|---:|---:|---:|
| 512 | 4.21-4.22 | 4.15-4.16 | ~1.5% *faster* |
| 8,192 | 37.8-39.3 | 37.0-40.1 | a wash, no consistent direction |

`moe_expert_down_mma` was checked the same way before attempting it: already
at 64 of the 64 registers `MOE_DOWN_BLOCKS_PER_SM` 4 (65,536 / (256×4))
allows, zero spill -- an even tighter ceiling than the gate/up kernel's, so
the same trade was expected to apply and the kernel was not modified.

### The finding

`attn_flash_causal_mma` needed register-based prefetching because it runs one
block per SM by construction (`MMA_HPB` fills the whole GQA group into one
block) -- there is no second resident block to hide DRAM latency, so a
register pipeline was the only lever. Both routed-expert MMA kernels are the
opposite case: `__launch_bounds__` already tunes them to the register ceiling
that maximizes *occupancy*-based latency hiding, deliberately, per the
kernel's own comment. A register prefetch pipeline does not add a second
latency-hiding mechanism on top of that one; it **competes with it** for the
same register budget, and on this card it loses -- either as an outright
spill (3 blocks/SM, 20% slower) or as a wash once occupancy is traded down to
make room for it (2 blocks/SM). Not shipped. `bench_moe_mma` is kept: it is
the first isolated harness for either kernel and makes the next attempt here
measurable in seconds rather than a full `bench_forward` run.

## Two more decode-shape defects at N=2-4, both real, and N=3 still short (2026-08-18)

A follow-up round on "Batched decode across sequences" above, aimed at the
gap that section left open: at N=3 aggregate was 64.1 tok/s, below
single-stream's 84.3. Two things were named as candidates and both turned
out to be real, measured defects with real fixes -- and neither one, alone
or together, closes the N=3 gap. This section records both, honestly,
including the one that made things worse before it made them better.

### First: MoE's routed-expert grouped GEMM was doing a wasted second pass at decode

`docs/OPTIMIZATION.md`'s R2 predicted this and the previous section's "what a
follow-up needs" named it first: audit `moe_expert_ffn`/`moe_expert_down`'s
tile width at N = 2-8 before assuming GDN's fix generalizes. Profiled first,
per that instruction, rather than assumed: `nsys --cuda-graph-trace=node`
over a captured N=3 replay at a 32,768-token context put `moe_expert_ffn` at
294,357.97 ns/call and `moe_expert_down` at 175,700.90 ns/call, 39 calls each
-- 18.33 ms of roughly 40 ms, ~45% of the step, dwarfing every other kernel
including the now-fixed `gdn_proj_q8_0`.

Reading the kernel found the defect was not the tile width itself --
`MOE_TILE_DISPATCH` already specializes to `TM` 2, 8 or 16 by `live_tile_rows`,
which is the "M tile of 1-2" R2 asked for. The defect is one level up: a
dispatch bucket (`block_size`, 32 in `forward.rs`) is wider than a tile pass
(`MOE_TM`, 16), so the kernel's `for (m0 = 0; m0 < block_size; m0 += MOE_TM)`
loop runs twice per bucket, and at decode a bucket usually holds one real
token in the first half and nothing in the second. `live` in that loop only
bounds-checks the output row (`r < intermediate`), not whether the current
`m0` slice has any live rows (`bm`) -- so the second, entirely-padding slice
still dequantizes and contracts the full weight stack for a result that can
only be zero. Fix, in both `moe_expert_ffn` and `moe_expert_down`:

```c
int bm = live_tile_rows(rows);
if (bm == 0) continue;   // this whole tile pass is the padding sentinel
```

Safe because `live_tile_rows` is documented uniform across the block --
`bm == 0` is the same answer on every thread, so the `continue` is not a
divergent branch. Confirmed at the kernel level with a second
`nsys --cuda-graph-trace=node` profile, isolated to the batch-3 decode phase
alone (`LLMXABE_SKIP_SINGLE_STREAM=1`, a one-line addition to
`bench_decode_batch` for exactly this): `moe_expert_ffn` dropped to
106,846-188,925 ns/call (160,374.9 avg, 1,120 calls) and `moe_expert_down` to
90,878-146,141 ns/call (122,294.8 avg, 1,120 calls) -- both roughly halved,
matching the "one wasted pass in two" arithmetic exactly.

**What was tried and correctly *not* kept:** the same profile suggested
`dequant_tile_q6k`'s `q6k_value(d, sc, si, raw)` -- called once per element,
recomputing `d * (float)sc[si]` fresh each of four times -- looked like a
second lever. It is not a new idea: "The MoE decode GEMV, re-measured"
(2026-08-17, above) tried exactly this hoist for `moe_expert_ffn_gemv` and
confirmed with `cuobjdump -sass` that the hoisted and unhoisted forms compile
to **byte-identical code** -- `ptxas` already proves `sc[si]` does not depend
on the unpack loop's variable and hoists it itself. Applying it here anyway
measured 77.9 vs 77.8 tok/s at N=3, run-to-run noise, exactly as that section
predicts. Reverted; a comment on `q6k_value` now points future readers at
that section instead of leaving the redundant hoist as if it were load-bearing.

Effect of the MoE fix alone, N=3 at a 32,768-token context: **64.1 -> 77.8
tok/s, +21.4%.** Real, and still short: below single-stream's 83.9 and short
of the 109.6 tok/s (1.3x) the next section's target called credible progress.

### Second: GDN's own round-1 fix paid weight traffic N times below its tile floor -- and the first attempt at a wider tile regressed N=3 before a second attempt fixed it

The previous section's `GdnBlock::project_per_sequence` fallback (added to
fix a *worse* defect -- the tiled kernel running unconditionally at any
`tokens > 1`) itself has a cost the previous section's own doc comment
already named without drawing the conclusion: "`tokens` separate untiled
reads cost `tokens` times the weight." Below `PROJ_TILES[0]` (8), a batch of
`N` sequences reads GDN's qkv/gate/out projection weights `N` times instead
of once. Converting the aggregate table to step time makes the shape of this
visible: step time was close to linear in `tokens` from N=1 to N=4 and only
dropped once N=8 crossed into `proj_tile_for`'s fully-live tile-8 regime --
the signature of a code path boundary sitting exactly at `PROJ_TILES[0]`.

**First attempt, and a real regression.** Added `gdn_proj_q8_0_t2`/`_t4`
kernel instantiations (the `GDN_PROJ_TILED` macro is already parametric) and
extended `PROJ_TILES` to `[2, 4, 8, 16]`, keeping the existing "widest tile no
wider than `tokens`" selection rule. This is wrong below the widest declared
tile: `proj_tile_for` launches `ceil(tokens / tile)` *independent* grid.y
slices, and each slice re-walks the whole weight matrix on its own -- the
reuse `GDN_PROJ_TILED` buys is within a slice, not across them. At 3 tokens,
"widest tile no wider than 3" is tile 2, and `ceil(3 / 2)` is two slices: one
full, one half-live, which is two full weight reads for three tokens, not
one. Measured on `bench_decode_batch` at the 32,768-token context: N=3 went
from 77.8 to **73.0 tok/s (41.1 ms/step)** -- slower than the per-sequence
fallback it was meant to replace, and N=2 and N=4 (both exact fits for a
declared tile, one slice either way) improved as expected, which is what
made the N=3 regression legible as a selection-policy bug rather than a
tiling one.

**Second attempt: change which tile gets picked, not just which tiles
exist.** `proj_tile_for` now runs two different rules, split at the widest
declared tile. Below it, the smallest tile that covers `tokens` in a
*single* slice -- which can be wider than `tokens`, deliberately, since a
guarded slice pays close to a full slice's weight-read cost regardless of
how empty it is (this project's own prior measurement: a 32-wide tile at 19
live tokens cost 21%, not proportionally more at lower fill). Three tokens
now takes tile 4 in one guarded slice at 75% live, not tile 2 in two. At and
above the widest declared tile, the original rule stands unchanged -- proven
correct at 128 and 512 tokens by the sweep this constant's own doc comment
already recorded, and no single tile covers those widths in one slice anyway.

Measured, 32,768-token context, mean ms/step:

| N | project_per_sequence (round 1) | tile [2,4,8,16], old rule | tile [2,4,8,16], new rule |
|---:|---:|---:|---:|
| 2 | 27.55 | 26.30 | 26.35 |
| 3 | 38.56 | 41.12 (**regression**) | 38.57 |
| 4 | 48.94 | 40.79 | 40.76 |

The new rule recovers N=2's and N=4's wins and removes N=3's regression --
but does not improve N=3 past where it already was. A second
`nsys --cuda-graph-trace=node` profile, isolated to the batch-3 decode phase,
shows why: `gdn_proj_q8_0_t4` (the single guarded slice N=3 now takes) costs
125,662.3 ns/call on average, ~90 calls per step, ~11.3 ms/step -- close
enough to what three separate untiled reads cost in aggregate that the two
approaches wash out at this specific width. The fix is a genuine
architecture improvement (one weight read per projection instead of three,
which is the right shape and will matter more as more of the guard is
recovered elsewhere) without being a net decode-time win at N=3 specifically,
because the per-sequence fallback it replaced was already close to
cost-competitive here, not because the fix does not do what it says.

**Also checked and already fine:** the follow-up brief asked whether the LM
head (540 MB, the single largest tensor) was being evaluated per sequence
rather than once for the whole batch. It is not -- `LmHeadKernels` already
compiles one exact GEMV per token width from 1 to 8
(`lm_head_gemv_b1`..`lm_head_gemv_b8`), `Forward::body_batch_decode` calls it
once with `n` tokens, and the per-sequence loop after it is only the cheap
argmax reduction over each sequence's own already-computed logits row, not a
second pass over the weight. No fix needed; recorded so the next reader does
not re-derive it.

### Final measured tables, both fixes together, GPU 2

Accuracy gates, unchanged tolerances, run fresh against the combined diff:
all three `tests/batch_decode.rs` differentials (the exact cross-sequence
check still measures **0.000e0**; the tolerance check's max-abs figures are
unchanged from the previous section, 8.631e-5 to 2.284e-4 against a 5e-3
budget), all 6 `tests/moe_differential.rs` tests, all 11
`tests/moe_block.rs` tests, all 26 `tests/gdn_differential.rs` /
`gdn_chunked_differential.rs` / `gdn_block.rs` tests, and
`the_forward_pass_reproduces_llama_cpps_logits_and_its_argmax` -- llama.cpp's
argmax token, still exactly reproduced. No tolerance loosened.

**2,048-token context:**

| shape | mean ms/step | aggregate tok/s | per-sequence tok/s |
| --- | ---: | ---: | ---: |
| single_stream (N=1, baseline) | 9.96 | 100.4 | 100.4 |
| batch 1 | 11.29 | 88.6 | 88.6 |
| batch 2 | 21.94 | 91.1 | 45.6 |
| batch 3 | 32.04 | 93.6 | 31.2 |
| batch 4 | 31.76 | **125.9** | 31.5 |
| batch 8 | 47.92 | **166.9** | 20.9 |

**32,768-token context:**

| shape | mean ms/step | aggregate tok/s | per-sequence tok/s |
| --- | ---: | ---: | ---: |
| single_stream (N=1, baseline) | 11.92 | 83.9 | 83.9 |
| batch 1 | 13.36 | 74.9 | 74.9 |
| batch 2 | 26.35 | 75.9 | 37.9 |
| batch 3 | 38.57 | 77.8 | 25.9 |
| batch 4 | 40.76 | **98.1** | 24.5 |
| batch 8 | 64.46 | **124.1** | 15.5 |

N=8 is unchanged by either fix in this section, at either context -- it
crosses `MMA_SPLIT_TOKENS`/`MMA_MIN_TOKENS` (8) into the int8-tensor-core
path in both GDN and MoE, a different set of kernels from the ones either
fix touched.

### Against the target, honestly, again

Still short. N=3 at the 32,768-token context is 77.8 tok/s: up 21.4% from
where the previous section left it (64.1), no longer a regression against
single-stream's own batch-path overhead, but still below single_stream
itself (83.9) and short of the 109.6 tok/s (1.3x) this round's target called
credible progress, let alone llama.cpp's 154.58. N=4 is the width that moved
the most this round (81.7 -> 98.1, +20.1%) and N=2 moved modestly (72.6 ->
75.9, +4.5%); N=3 sits between two tile boundaries in both fixed kernel
families and inherits the worse case from each.

Both defects fixed this round were real, correctly diagnosed before being
fixed, confirmed at the kernel level with fresh `nsys` profiles rather than
assumed from the first measurement, and verified against every accuracy gate
with no tolerance changes. Neither one was the single lever that closes the
N=3 gap, and the round's own arithmetic says why: MoE's fix bought ~21% at
N=3 and GDN's fix bought ~0% net at that same width despite fixing a real
architectural defect, which means the step's remaining cost at N=3 is spread
across more of the ~1,000-1,300 per-step kernel launches the previous
section already measured than either single hypothesis accounted for.

### What a follow-up needs, updated

The previous section's list stands, with one item resolved (MoE's tile
width, item 1) and this round's own finding added:

1. ~~Audit `moe_expert_ffn`/`moe_expert_down`'s tile width at N = 2-8~~ --
   done this round; fixed, confirmed at the kernel level, +21.4% at N=3.
2. **GDN's guarded-tile-4 cost at N=2-4 is now large enough to audit on its
   own terms.** `gdn_proj_q8_0_t4` measured 125,662.3 ns/call, ~90 calls/step
   at N=3 -- comparable in total to MoE's entire fixed tiled-GEMM
   contribution, and no longer obviously cheaper than the untiled
   per-sequence reads it replaced at this specific width. A genuinely fused
   multi-sequence kernel (one launch, N pointers, rather than one guarded
   tile pass with N/tile-width live lanes) is a different design than either
   version tried here and was not attempted this round.
3. **Fused, device-side-indexed multi-sequence kernels for the loops that
   remain** -- GDN's causal convolution and delta-rule update, attention's
   rotary/append/read -- as the previous section already named. Still `N`
   separate launches each, unaudited and unchanged this round.
4. Continuous batching, mixed prefill and decode in one step, and KV cache
   pooling -- unchanged from the previous section, still out of scope for
   this workstream.

## GDN's narrow tiles were occupancy-starved, not guard-heavy: RR=1 recovers most of it (2026-08-18)

A same-day follow-up to the section above. The coordinator's read of the
launch geometry behind "tile-4 costs about what three untiled reads did":
`GDN_PROJ_TILED`'s grid is `(n_rows.div_ceil(PROJ_WARPS * RR),
tokens.div_ceil(TT))`, block `(32, PROJ_WARPS)`. At decode widths grid.y
collapses to 1 (a single token-tile slice), which leaves grid.x as the
*only* axis with any blocks in it. With `PROJ_WARPS` 4 and `RR` 4 (the value
every tile inherited from the 512-token sweep that chose it), `n_rows` in
the 512-2048 range this model's projections actually use gives
32-128 blocks -- on a 72-SM card, that is SMs sitting idle, not latency
being hidden. The untiled kernel `project_per_sequence` used to loop reads
`n_rows.div_ceil(PROJ_WARPS)`, four times as many blocks for the same rows,
which is why one untiled read was cheap relative to one narrow-tile read: it
was never the tile's guard costing the difference, it was the grid.

`RR` exists to divide *activation* L2 traffic, and that traffic is
irrelevant at this width -- ~32 KB total for a 2-4 token batch against the
2.1 GB/call the 512-token sweep was solving for. Fix: `PROJ_ROWS` for tiles
2 and 4 drops from 4 to 1, which quadruples their grid.x back to the
untiled kernel's own geometry. Tiles 8 and 16 (prefill's regime, where
grid.x is already wide) are untouched.

### Verification: per-launch time and block count, before and after

`nsys --cuda-graph-trace=node`, isolated to each width's own decode phase
(`LLMXABE_SKIP_SINGLE_STREAM=1`), RR=4 (the previous section's committed
state) against RR=1, same weights, same layers, same token count -- only
`RR` differs:

| N | tile | grid.x @ RR=4 | grid.x @ RR=1 | ns/call @ RR=4 | ns/call @ RR=1 | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | `n_rows/16` | `n_rows/4` | 67,487.6 | 44,238.0 | 1.53x |
| 3 | 4 | `n_rows/16` | `n_rows/4` | 125,662.3 | 89,041.7 | 1.41x |
| 4 | 4 | `n_rows/16` | `n_rows/4` | 70,528.9 | 64,026.8 | 1.10x |

Real in every case, and short of the 3-4x the block-count arithmetic alone
predicts. The gap has a second, distinct explanation, found by reading the
macro rather than assumed: `GDN_PROJ_TILED`'s fully-live branch (taken when
`live_t == TT`, N=4's case) is `_Pragma("unroll")`-marked over a
compile-time `TT`; the guarded branch (`live_t < TT`, N=2 and N=3's case) is
not, because its bound is the runtime value `live_t`, which the compiler
cannot unroll. N=4 at RR=4 (70,528.9 ns) already ran faster in absolute
terms than N=3 at RR=4 (125,662.3 ns) despite doing *more* nominal work --
one live token more -- which only makes sense if the guarded branch's
un-unrolled inner loop, repeated once per iteration of the outer `for (b =
0; b < blocks; ++b)` weight-staging loop (32-64+ iterations depending on the
projection), is paying real per-element loop overhead that the unrolled
branch does not. This is why RR=1's win is largest at N=2 (guarded, but the
smallest live/TT gap) and smallest at N=4 (never guarded at all -- RR=1's
whole benefit there is the grid.x widening, with no un-unrolled-loop tax to
also remove). **Not fixed this round** -- named here as the next concrete
lever, distinct from and additional to the occupancy fix, for whoever picks
this back up.

### Measured aggregate, both fixes together, GPU 2

Accuracy gates, unchanged tolerances: all three `tests/batch_decode.rs`
differentials (exact cross-sequence check still **0.000e0**), all 26 GDN
differential/block tests, and the golden logits test -- llama.cpp's argmax
still exactly reproduced. No tolerance loosened.

**2,048-token context:**

| shape | mean ms/step | aggregate tok/s | per-sequence tok/s |
| --- | ---: | ---: | ---: |
| single_stream (N=1, baseline) | 9.94 | 100.6 | 100.6 |
| batch 1 | 11.24 | 89.0 | 89.0 |
| batch 2 | 20.01 | **99.9** | 50.0 |
| batch 3 | 29.11 | **103.0** | 34.3 |
| batch 4 | 31.38 | **127.5** | 31.9 |
| batch 8 | 47.85 | **167.2** | 20.9 |

**32,768-token context:**

| shape | mean ms/step | aggregate tok/s | per-sequence tok/s |
| --- | ---: | ---: | ---: |
| single_stream (N=1, baseline) | 11.90 | 84.0 | 84.0 |
| batch 1 | 13.34 | 75.0 | 75.0 |
| batch 2 | 24.49 | **81.7** | 40.8 |
| batch 3 | 35.83 | **83.7** | 27.9 |
| batch 4 | 40.58 | **98.6** | 24.6 |
| batch 8 | 64.39 | **124.2** | 15.5 |

N=2 and N=3 now beat single-stream at both contexts for the first time this
workstream has measured -- N=3 at 2,048 tokens is 103.0 against 100.6
(1.024x), and at 32,768 tokens is 83.7 against 84.0, parity within
run-to-run noise rather than the previous section's clear regression.

### Against the target, honestly, a third time

N=3 at the 32,768-token context: **77.8 -> 83.7 tok/s this round (+7.6%)**,
cumulative from this workstream's 64.1 tok/s start: **+30.6%**. Still short
of the 109.6 tok/s (1.3x) bar and further short of llama.cpp's 154.58 --
the predicted "step(3) drops from 38.6 toward ~20ms" did not happen (it
dropped to 35.8), because the occupancy fix, while real and correctly
diagnosed, was worth 1.1-1.5x per launch rather than 3-4x once the
un-unrolled guarded branch took back part of what the wider grid bought.

**N=1 batch-vs-single-stream overhead, noted as asked, not chased:** batch 1
runs 13.34 ms/step against single_stream's 11.90 (32,768 tokens, -12.1%) and
11.24 against 9.94 (2,048 tokens, -13.1%) -- consistent with the
coordinator's ~12% estimate and unmoved by either fix in this section, since
neither touches whatever the batch-path-at-N=1 vs single-stream-path
difference actually is. Not investigated this round, per the coordinator's
own "only after the main fix" framing -- the main fix's own return has not
yet run out.

## Two more fixes: the guarded tile's unroll, and batch(1)'s stray kernel (2026-08-18)

A second same-day follow-up. Two items, both named by reading the previous
section's own numbers rather than guessed: the guarded-tile-vs-fully-live
gap the RR=1 fix left open, and the N=1 batch-vs-single-stream overhead
that section flagged but did not chase.

### The guarded branch was bounding a loop the compiler needed unrolled

`GDN_PROJ_TILED`'s guarded branch (`live_t < TT`, N=3 against tile 4)
bounded its inner accumulate loops by the *runtime* value `live_t`, unlike
the fully-live branch's `_Pragma("unroll")`-marked compile-time `TT` bound.
The previous section's own table made the cost visible without further
measurement: N=3 at tile 4 (89,041.7 ns/call) was slower than N=4 at the
same tile fully live (64,026.8 ns/call) despite doing *less* nominal work
-- the only way that inverts is if the un-unrolled branch is paying real
per-element loop overhead the unrolled one does not.

Fixed with the pattern this codebase already uses elsewhere -- attention's
software pipelining, and the fully-live branch of this same macro: unroll
over the full compile-time `TT`/`RR`, and gate each element with a
*predicate* (`t0 + i < n_tokens`, `n0 + r < n_rows`) rather than bounding
the loop by it. Out-of-range activation slots read as zero, contributing
exactly zero to the accumulator; out-of-range row slots re-address the
already-validated `n0` rather than walk off the end of `weight`, and are
never read back. `acc`/`xv` stay register-resident instead of spilling to
local memory the way a runtime trip count forces.

Measured, `nsys --cuda-graph-trace=node`, isolated N=3 decode phase:
`gdn_proj_q8_0_t4` drops from 89,041.7 to **62,887.2 ns/call (1.42x)** --
now fractionally *faster* than N=4's fully-live cost, matching the
prediction exactly. N=2 (tile 2, exact fit, already takes the fully-live
branch) is unaffected as expected: 44,238.0 to 43,646.6 ns/call, noise.
Combined with the RR=1 fix, `gdn_proj_q8_0_t4` at N=3 is now **2.0x**
faster than the pre-RR=1 baseline (125,662.3 ns/call).

### The N=1 batch overhead was one stray kernel, found by nsys-diff

The previous section noted batch(1) ran ~12-13% slower than single_stream
at both contexts and did not investigate. `nsys-diff`, exactly as asked:
one isolated single_stream decode step against one isolated batch(1)
decode step (`LLMXABE_BATCH_N=0` to skip the batch sweep entirely for the
first; `LLMXABE_SKIP_SINGLE_STREAM=1 LLMXABE_BATCH_N=1` for the second),
every kernel's per-step count and per-call cost compared side by side.

Almost every kernel's per-step count matched exactly between the two
profiles -- this was never about extra or missing launches. One pair did
not match, and it explained **94.7% of the entire per-step time gap**:

| kernel | single_stream ns/step | batch(1) ns/step |
| --- | ---: | ---: |
| `gdn_proj_split_gemv` | 1,551,030.2 | 0 |
| `gdn_proj_split_gemv_add` | 695,973.6 | 0 |
| `gdn_proj_q8_0` | 0 | 3,474,748.8 |

`GdnBlock::run` (single-stream's own method) already special-cases
`tokens == 1` to the repacked split-layout GEMV kernels -- vectorized
`char4`/`float4` loads, and the output projection's residual add fused
into the same launch. `GdnBlock::run_batch_decode` never took that branch
at any batch width, including one, because the branch's whole justification
(`gdn_proj_split_gemv` writes `out[n]` with no token axis, wrong for
`tokens > 1`) is vacuously true at `tokens == 1` too -- "no token axis" and
"the batch's one token" are the identical buffer layout. Fixed by computing
the same `gemv = tc.filter(|_| tokens == 1)` `run` does and taking the same
three-way branch (tensor core / split GEMV / generic tiled) in
`run_batch_decode`, at both the qkv/gate site and the out site. No shape
changes needed: at `tokens == 1` every batch buffer is already exactly the
length the single-token kernels expect.

### Final measured aggregate, all four fixes together, GPU 2

Accuracy gates, unchanged tolerances: the exact-match batch differential
(`identical_prompts_in_one_batch_produce_bit_identical_rows`, still
**0.000e0**), the tolerance differential, the captured-graph-equivalence
differential, all 26 GDN differential/block tests, and the golden logits
test -- llama.cpp's argmax still exactly reproduced. No tolerance loosened
across any fix in this workstream.

**2,048-token context:**

| shape | mean ms/step | aggregate tok/s | per-sequence tok/s |
| --- | ---: | ---: | ---: |
| single_stream (N=1, baseline) | 9.91 | 100.9 | 100.9 |
| batch 1 | 9.91 | **100.9** | 100.9 |
| batch 2 | 19.94 | **100.3** | 50.2 |
| batch 3 | 27.04 | **110.9** | 37.0 |
| batch 4 | 31.41 | **127.4** | 31.8 |
| batch 8 | 47.19 | **169.5** | 21.2 |

**32,768-token context:**

| shape | mean ms/step | aggregate tok/s | per-sequence tok/s |
| --- | ---: | ---: | ---: |
| single_stream (N=1, baseline) | 11.89 | 84.1 | 84.1 |
| batch 1 | 11.96 | **83.6** | 83.6 |
| batch 2 | 24.43 | **81.9** | 40.9 |
| batch 3 | 33.64 | **89.2** | 29.7 |
| batch 4 | 40.57 | **98.6** | 24.6 |
| batch 8 | 63.90 | **125.2** | 15.6 |

`batch 1` now matches `single_stream` at 2,048 tokens exactly and sits
within 0.6% of it at 32,768 -- the ~12-13% gap the previous section flagged
is closed. N=2 and N=3 both beat single-stream at both contexts.

## Batched decode across sequences: workstream summary and hand-off (2026-08-18)

Four rounds landed on top of the initial implementation, each confirmed
with `nsys --cuda-graph-trace=node` before being named a fix and re-verified
against every accuracy gate after. N=3 at the 32,768-token context moved
**64.1 -> 77.8 -> 83.7 -> 89.2 tok/s** across them (+39.2% cumulative); N=1's
batch-vs-single-stream gap closed from ~12% to noise. Still short of the
109.6 tok/s (1.3x single-stream) target and further short of llama.cpp's
154.58. This section separates what is now fixed from what is structural
to the shape of the workload, and names what is left with the sizes this
workstream's own measurements support.

### (a) What's fixed

1. **Batched decode itself** -- `Forward::run_batch_decode` /
   `capture_batch_step` / `replay_batch_step` advance `N` independent
   sequences in one CUDA-graph-captured pass, with weight-bound work
   batched into one launch per step and only genuinely per-sequence state
   (GDN's conv/recurrent, attention's rope/append/read) looped. Correctness
   gated by three differentials, one of them bit-exact.
2. **MoE's grouped GEMM wasted second pass** (`moe_expert_ffn`/
   `moe_expert_down`) -- a dispatch bucket wider than a tile pass meant the
   second, all-padding pass at decode still paid full dequant/contract
   cost. Fixed with a `bm == 0` early-continue. +21.4% at N=3.
3. **GDN's projection tile floor** -- extended `PROJ_TILES` down to 2/4 so
   decode-width batches take a genuinely tiled (weight-read-once) path
   instead of `N` separate untiled reads, with a two-regime tile-selection
   rule (smallest single-covering tile below the floor, widest-tile-many-
   slices at and above it) after a first attempt regressed N=3 by picking
   too narrow a tile and paying for two slices instead of one.
4. **GDN's narrow-tile occupancy** -- `PROJ_ROWS` for tiles 2/4 dropped
   from 4 to 1, quadrupling grid.x at decode widths where grid.y collapses
   to 1 and grid.x had been the only axis with any blocks in it (32-128 on
   a 72-SM card). 1.1-1.53x per launch depending on width.
5. **GDN's guarded-tile loop bound** -- predicate-unrolled instead of
   runtime-bounded, matching the pattern the fully-live branch and
   attention's software pipelining already use. 1.42x per launch at N=3,
   closing the gap to N=4's fully-live cost almost exactly.
6. **Batch(1)'s stray kernel choice** -- `run_batch_decode` now takes the
   same one-token split-layout GEMV path `run` always used, instead of
   unconditionally falling to the generic tiled dispatcher's untiled
   fallback. Closed 94.7% of the N=1 batch-vs-single-stream gap.

Every fix above is additive or a routing/tiling change to code this
workstream owns (`crates/xabe-engine/src/block/gdn.rs`,
`crates/xabe-cuda/src/kernels/moe.rs`'s routed-expert grouped GEMM);
`attention.rs`'s kernels were never touched, per the scope this workstream
was given.

### (b) What's structural, per §2.6's roofline

`docs/OPTIMIZATION.md` §2.6 models weight bytes per step as (LM head +
projections + shared expert, read once) + 113.377 MB x `D(N)` (routed
experts, read once per distinct expert per layer), where `D(N)` is the
expected number of distinct experts a batch of `N` tokens touches. Two
consequences of that model bound what any kernel-level fix in this
workstream can buy, independent of how well-written the kernel is:

- **Routed-expert traffic grows with `N`, not against it, until the
  activation-density curve saturates.** `D(3) = 23.26` of 256 experts
  (9.1%) against `D(1) = 8.00` (3.1%) -- three sequences already touch
  ~2.9x the distinct experts one does, so the weight-read amortization
  batching buys is partial by construction at this width, not a kernel
  defect. It gets better with `N` (`D(32) = 163.30`, 63.8%) but this
  workstream's batch widths (1-8) are still on the steep part of that
  curve.
- **Per-sequence state never amortizes, at any `N`.** §2.6's "+ state"
  column is a flat 131.7 MB/token regardless of `N` -- each sequence's own
  KV cache and GDN recurrent state is exactly as much traffic per token
  whether it decodes alone or alongside seven others, because there is
  nothing to share. This is not specific to this engine: llama.cpp pays
  the identical per-sequence state cost, which is why its own measured
  efficiency (§2.7) holds flat at ~41% of the concurrency-aware roofline
  from c=1 to c=3 rather than climbing -- the batching win it captures is
  the routed-expert term's amortization, not the state term's, because the
  state term has none to capture.

Read together: this workstream's fixes closed *implementation* gaps (a
kernel structurally unable to fill the card, a loop the compiler couldn't
unroll, a kernel choice that skipped a faster path entirely) that had
nothing to do with the workload's own shape. The remaining distance to
llama.cpp's 154.58 is a mix of that kind of gap still uncovered elsewhere
in the step, and the part of the gap that §2.6 says is not implementation
at all -- routed-expert traffic at N=3 is genuinely higher per token than
at N=32, for any correct implementation.

### (c) Named remaining levers, with sizes this workstream's own measurements support

1. **MoE GEMV bandwidth, decode's `N < MMA_SPLIT_TOKENS` shape.** "The MoE
   decode GEMV, re-measured" (2026-08-17, above) put the combined
   `grouped_forward` pass at 38.9% of the card's 672 GB/s streaming
   roofline; the individual kernels underneath it separately reach 47%
   (`moe_expert_ffn_gemv`, Q6_K) and 63% (`moe_expert_down_gemv`, Q8_0)
   against `lm_head_gemv_b1`'s 82-89%. Closing that gap toward the LM
   head's own number is the largest single-kernel-family lever this
   workstream did not attempt this round, on the order of the MoE fix's
   own +21.4% or larger given it touches every decode step at N below 8,
   not the routed path specifically.
2. **`dp4a`/Q8_1 activation quantization -- documented, and declined on
   accuracy risk.** The same section names the concrete next step for the
   GEMV bandwidth lever above: llama.cpp's `vec_dot_q6_K_q8_1_impl_mmvq`
   pre-quantizes the activation to Q8_1 before the dot product, which
   `dp4a` needs to reach further past 63%. This adds a rounding source to
   the one tensor in this model that has not yet had one (activations are
   carried fp32 end to end today), and was explicitly not attempted --
   "there is no predicting from arithmetic alone whether it survives rank
   4" the way this session's other rounding changes were verified to.
   Whoever picks this up needs to build the quantize kernel, wire it
   through both GEMVs, and clear it against `moe_differential.rs` and the
   golden logits test before it can be called a fix rather than a
   regression risk.
3. **Attention's per-sequence loop is still `N` separate launches.**
   `GatedAttentionBlock::forward_batch_decode` batches every weight-bound
   step and loops only rope, KV append and the causal read -- three
   launches per sequence per layer, unaudited this workstream for the same
   class of defect (occupancy, unroll, stray kernel choice) the four GDN
   fixes above found and fixed. Given GDN alone was worth a combined ~2x
   on its own narrow-tile kernel and closed the entire N=1 gap, attention's
   equivalent loop is a plausible next lever of comparable size, not yet
   measured.

### Hand-off

The structure is real and load-bearing: correct batched decode with
bit-exact cross-sequence isolation, CUDA graph capture, and four
independently verified kernel-level fixes, each confirmed with `nsys`
before being named a fix and re-verified against every accuracy gate
after. The honest remaining-gap analysis above -- what's fixed, what §2.6
says cannot be fixed at this batch width by any implementation, and what's
named but unmeasured -- is this workstream's hand-off point.

## The tiled-softmax scalar decode kernel: correct, and slower than the kernel it targeted (2026-08-18)

The earlier section "The fix, specced and not attempted this session" named a
gate-safe alternative to the rejected `attn_flash_decode_mma`: keep `Q K^T`
scalar fp32 with `attn_flash_decode_warp`'s existing warp-butterfly reduction,
and tile the softmax the way `attn_flash_causal_gqa` already does for prefill
-- one max-scan, one rescale, one batched exponential per `DEC_TILE`-key tile
instead of the per-key online form. Built this session as
`attn_flash_decode_tile`, wired into `decode()` behind a `disable_decode_tile`
A/B lever mirroring `disable_tensor_cores`. It does not ship, for a different
reason than the MMA kernel: it is fully correct and still slower than the
kernel it was meant to replace.

### Shape, as built

Same grid and block as `attn_flash_decode_warp` -- `(DECODE_SPLITS, kv_heads)`,
one warp per block, identical `uint4`-coalesced key load and 5-step
`__shfl_xor_sync` reduction. `DEC_TILE = 32` keys' scores are computed into a
`DEC_MAXG * DEC_TILE` (8 x 32, 1 KiB) static `__shared__` array -- no dynamic
shared memory, so none of the `attn_flash_decode_mma` occupancy math applies.
One max-scan and one `expf`-based rescale run per tile per head instead of per
key; the exponential phase gives lane `i` key `i` of the tile, the same
one-thread-per-slot pattern `attn_flash_causal_gqa`'s own softmax already
uses. `V` is read a second time from global in a separate pass rather than
staged alongside `K` for the tile, specifically to avoid the shared-memory
blowup that sank the MMA kernel: staging `V` for the whole tile would cost
`DEC_TILE * head_dim` floats (32 KiB at this geometry) instead of the `DEC_MAXG
* DEC_TILE` this kernel actually uses (1 KiB) -- the same total bytes moved as
`attn_flash_decode_warp`, just in two passes instead of one.

### Correctness: green, unlike the MMA kernel

All nine `attention_differential` tests pass at the unchanged 1e-5 `GATE`,
including `device_decode_matches_the_reference_over_a_deep_window`'s
`n_keys = 61` case that broke `attn_flash_decode_mma` at 1.11e-5:

| n_keys | max_abs | cosine |
|---:|---:|---:|
| 4,096 | 1.062e-7 | 1.000000000 |
| 4,097 | 1.006e-7 | 1.000000000 |
| 61 | 1.043e-7 | 1.000000000 |

Two orders of magnitude tighter than the *other* scalar decode kernel's own
worst case elsewhere in this file, and no surprise: nothing about `Q K^T`
changed arithmetic, so there is no new rounding source to find. The design
premise -- tile the softmax without touching the dot product -- holds exactly
as specced.

### Performance: slower than `attn_flash_decode_warp` at every depth measured

`bench_attention`, `LLMXABE_ATTN_CHUNK=1`, GPU 1, 3 interleaved rounds:

| key_offset | `attn_flash_decode_warp` (ms) | `attn_flash_decode_tile` (ms) | ratio |
|---:|---:|---:|---:|
| 2,048 | 0.074-0.076 | 0.088-0.091 | 0.83x |
| 8,192 | 0.125-0.127 | 0.154-0.156 | 0.81x |
| 32,768 | 0.331-0.332 | 0.444-0.446 | 0.74x |
| 65,536 | 0.620-0.622 | 0.832-0.834 | 0.75x |
| 98,304 | 0.910-1.000 | 1.213-1.215 | ~0.76x |
| 131,072 | 1.196-1.200 | 1.603-1.610 | **0.75x (1.34x slower)** |

Not short of parity -- a straight regression at every depth, not only at
131K. Following the lead's suggestion to predicate-unroll the two per-tile
passes over the compile-time `DEC_TILE` bound (the fix that worked for
`GdnBlock`'s guarded projection tile, see "Two more fixes" above) made it
*worse*: 2.136-2.144 ms at 131,072 keys, 1.79x slower than
`attn_flash_decode_warp`, worse than the runtime-bounded version it replaced.
`nvcc -arch=sm_75 -cubin -Xptxas -v` on the extracted kernel ruled out
register spilling as the cause either way -- `attn_flash_decode_tile` at 193
registers / 0 spill against `attn_flash_decode_warp`'s 195 / 0 spill, nearly
identical, so the predicate-unroll's regression is code size doing nothing
useful, not a register-pressure story the GDN fix's mechanism would predict.

### Diagnosis: the calibration ladder's synthetic kernel was unguarded; the real one isn't

This is the load-bearing finding, and it corrects the "calibration ladder"
section's interpretation rather than just adding a data point next to it.
That section's synthetic kernel measured "full online softmax (`expf`,
rescale, accumulate)" at ~53% of streaming roofline against the load-only
90% and dot-product-only 87.2%, and read the 34% gap to
`attn_flash_decode_warp` itself as still-unexplained overhead on top of that
53%. What the synthetic kernel's description does not say, and what matters
here: it is not stated to guard the rescale on whether the running max
actually moved. `attn_flash_decode_warp` does --
`if (nm != m[hh]) { corr = expf(...); ... }` -- and the number of times a
new key beats the running maximum of everything before it is a classic
record-statistics quantity, `O(log n)` in expectation regardless of the
data's order. At this kernel's ~455-key-per-split average window (131,072 /
288 splits), that is roughly 9 real rescales per head, not 455. The
*unguarded* corrected-every-key softmax the calibration ladder measured pays
for `n_visible` rescales; the real kernel was already paying for
`O(log n_visible)` of them before this session started. Tiling only
compresses the correction count from `n_visible` to `n_visible / DEC_TILE`,
which is a much smaller move once the guard has already compressed it to
`O(log n_visible)` -- there was less of the ladder's 87%-to-53% gap actually
on the table than the ladder implied, because the ladder's synthetic kernel
was never comparable to the guarded kernel it was calibrating.

What tiling adds instead of removing, in this specific block shape: this
kernel's one warp handles all `gqa = 8` query heads (chosen so a key loaded
once is reused eight times out of registers, per `attn_flash_decode_warp`'s
own module comment). Depositing each tile's scores into shared needs
`if (lane == 0) s_sh[hh][jj] = ...` once per head per key -- up to 256 single-
lane writes per tile with 31 of 32 lanes idle each time. `attn_flash_causal_gqa`
pays the structurally identical per-key single-lane write (`if (lane == 0)
my_w[u * GQA_KT + jj] = ...`), but its block runs `gqa` separate *warps*
concurrently, one per head, so no single warp serializes all eight heads'
worth of that cost the way this kernel's one-warp-does-every-head shape does.
Working diagnosis, not `ncu`-confirmed (still `ERR_NVGPUCTRPERM` on this
host): the two extra `__syncwarp()` barriers per tile and the second full
pass over the tile for `V` (zero `__syncwarp` calls and one fused pass in
`attn_flash_decode_warp`) are real costs this kernel pays that the baseline
does not, for a softmax-correction saving that the guard had already taken
most of.

### What happened to the code

Reverted with `git checkout -- crates/xabe-cuda/src/kernels/attention.rs
crates/xabe-engine/src/bin/bench_attention.rs`, zero diff against the parent
commit. `attn_flash_decode_tile`, its `disable_decode_tile`/`uses_decode_tile`
lever, and the `LLMXABE_DISABLE_DECODE_TILE` bench hook are gone from the
tree; nothing shipped.

### What this leaves

The guard-skip finding narrows where a real win could still come from: not
"replace the per-key softmax machinery", which was already mostly
guard-compressed, but the load/dot-product side that the ladder's own numbers
say tops out at 87% rather than the 90% load-only ceiling, or a block shape
that spreads the single-lane-write cost across more than one warp without
giving up the eight-way key reuse that motivates one warp per split in the
first place -- a real restructure, not a tuning knob, and not attempted here.
The tensor-core path is the next thing this session tries instead, with a
precision fix the original attempt did not have; see the section below.

## The split-precision `Q` tensor-core decode kernel: correct, and a real but small win at the deep end only (2026-08-18)

Rebuilt `attn_flash_decode_mma` from the earlier post-mortem's spec, with the
one change the lead asked for: instead of the rejected kernel's `m16n8k8`
fragment padding rows 8-15 with hardcoded zero, row `g` carries
`q_hi[g] = f16(q[g])` and row `g + 8` carries
`q_lo[g] = f16(q[g] - f32(q_hi[g]))` -- the same head's fp16 residual, not a
ninth through sixteenth head. `D[g] + D[g + 8]` (both halves of the same `mma`
call) is then `Q K^T` at `q_hi + q_lo` precision, roughly `2^-22` relative
against the rejected kernel's `2^-11`. Same `mma` count as before -- the dead
rows were already being multiplied by zero; this reuses that work instead of
discarding it. `P V` is untouched: its dead second half still pads with zero,
because `P` and `V` are already accepted under `MMA_GATE` elsewhere in this
file and the calibration ladder above already established `P V` was never the
expensive half. The code was not in git history (the earlier attempt was
`git checkout`-reverted uncommitted), so this is a fresh build against the
spec, not a restoration.

Templated on the occupancy width (`ATTN_DECODE_MMA(NAME, WPO)`, the same
macro-instantiation pattern `ATTN_FLASH` already uses) so both `WPO = 4` (the
rejected kernel's own shape) and `WPO = 2` (the occupancy experiment that
kernel's post-mortem left unmeasured) compile from one kernel body. Both are
always compiled; `AttentionKernels::set_decode_mma_wpo` and
`disable_decode_mma` pick between them and the fallback `attn_flash_decode_warp`
per instance.

### Two real bugs, both caught before they could ship

**First: `V`'s staging loop covered half the tile it needed to.** An early
draft's register arrays (`vlo`/`vhi`) were sized `[dstripes][2 * WPO]` with an
unused third dimension left over from an abandoned batching idea, silently
halving the key-pairs staged against what `P V`'s fragment read
(`4 * oc + tg` over `oc` in `0..WPO`, `tg` in `0..4`) actually consumes --
`4 * WPO`, not `2 * WPO`. `attention_differential.rs` caught it immediately:
`device_decode_matches_the_reference_over_a_deep_window` panicked on a
non-finite output at `n_keys = 4096`, before ever reaching the `n_keys = 61`
case the rejected kernel broke. Fixed by dropping the unused dimension and
correcting the bound to `4 * WPO`; confirmed with `nvcc -Xptxas -v` that
nothing else regressed structurally (0 spill before and after this specific
fix -- the fix's own register cost is a separate finding below).

**Second, and the one that actually mattered: the kernel's `(m, l)` partials
were in the wrong exponential base for the combine they feed.** Following
`attn_flash_causal_mma`'s `exp2f`/`ATTN_LOG2E` fold (real, measured 1.02-1.03x
there) carried the *scores* into log2 units, but `attn_flash_decode_combine`
-- shared unmodified with `attn_flash_decode_warp`, per the original spec --
merges partials with plain `expf`. A single real split hides this completely:
the merge computes `f = exp(m - gm)`, and with one non-empty split `gm = m`
makes `f = exp(0) = 1` regardless of what base `m` was tracked in, so
`acc / l` is exact no matter how `m` got there. Two or more real splits do
not hide it -- the relative weight between splits' exponential domains is now
computed in the wrong base entirely. Found with a temporary debug test
(`debug_decode_mma_probe`, not committed) that ran the same random inputs
through `attn_flash_decode_mma_wpo4` and `attn_flash_decode_warp` side by
side at increasing `n_keys`: exact agreement at `n_keys = 1` (cosine 1.0,
`max_abs = 0`), then broken at every single dimension from `n_keys = 2`
onward (cosine 0.995, `max_abs = 9.8e-2`) -- the exact onset the "one real
split hides it" mechanism predicts. Fixed by dropping the `exp2f` fold in
this kernel specifically: plain `expf` and `scale` (not `scale * ATTN_LOG2E`),
matching `attn_flash_decode_combine`'s units exactly. The fold stays where it
was measured to help (`attn_flash_causal_mma`, which owns its own combine
step -- there is no cross-block merge to disagree with there).

Both bugs were caught before a commit, by the differential suite and a
throwaway debug test respectively -- the second one specifically because
`AGENTS.md`'s "verify against the differential FIRST" was followed literally
rather than assumed satisfied by the design matching the spec on paper.

### Correctness, after both fixes

All nine `attention_differential` tests pass, at the unchanged 1e-5 `GATE`.
The case that broke the original zero-padded attempt:

| n_keys | max_abs | cosine |
|---:|---:|---:|
| 4,096 | 1.118e-7 | 1.000000000 |
| 4,097 | 1.043e-7 | 1.000000000 |
| 61 | 1.043e-7 | 1.000000000 |

Four orders of magnitude inside the gate, and at the same precision level as
`attn_flash_decode_warp`'s own worst case elsewhere in this file -- the
`q_hi`/`q_lo` split delivers exactly what it was sized for. Verified for both
`WPO = 4` and `WPO = 2` (the full nine-test suite was re-run with the
selection temporarily forced to each). `the_forward_pass_reproduces_llama_cpps_logits_and_its_argmax`
still passes and reproduces the identical argmax and logit
(token 25358, 19.998243) this file has recorded for that golden capture
before -- this particular capture is 19 tokens of pure prefill and does not
exercise decode, so this confirms no regression rather than exercising the
new kernel, but it is the gate `AGENTS.md` names and it is green.

### Register cost of the correctness fix

`nvcc -arch=sm_75 -cubin -Xptxas -v` on the extracted kernel, before and
after the `V`-staging fix:

| variant | before (buggy) | after (correct) |
|---|---:|---:|
| `WPO = 4` | 221 registers, 0 spill | 255 registers, 16 B spill (stores + loads) |
| `WPO = 2` | 239 registers, 0 spill | 255 registers, 52-56 B spill (stores + loads) |

Doubling the register arrays that were undersized doubled their register
cost, and both variants now sit at the 255-register ceiling with a small
spill. Not chased further this session -- the numbers below are what shipped
with this spill present, and reducing it (the prefetch pipeline is the
obvious place to look, matching `MMA_PREFETCH`'s own register-vs-occupancy
tension named in this file's MoE sections) is unmeasured headroom for
whoever picks this back up.

### Shared memory and occupancy, computed and matching the post-mortem's prediction

`dmma_shared_bytes(256, wpo)`, opted into the 64 KiB ceiling via
`cuFuncSetAttribute` for both variants (`attn_flash_causal_mma`'s own
pattern):

| WPO | shared/block | blocks/SM (shared-limited) |
|---:|---:|---:|
| 4 | 38,496 B (37.6 KiB) | 1 |
| 2 | 21,344 B (20.8 KiB) | 3 |

Both figures match the post-mortem's own arithmetic exactly (38,496 B and
20.8 KiB were named there without being built). `WPO = 2`'s three
resident blocks per SM is what gives the scheduler something to hide the K/V
staging latency behind that `WPO = 4`'s one block cannot -- the mechanism
the post-mortem predicted, now measured rather than argued.

### Kernel-level performance, `bench_attention`, `LLMXABE_ATTN_CHUNK=1`, GPU 1, 3 interleaved rounds

| key_offset | `attn_flash_decode_warp` (ms) | `WPO=4` (ms) | `WPO=4` ratio | `WPO=2` (ms) | `WPO=2` ratio |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 0.074-0.075 | 0.155-0.156 | 0.48x | 0.100-0.101 | 0.74x |
| 8,192 | 0.126 | 0.166-0.167 | 0.76x | 0.173-0.174 | 0.73x |
| 32,768 | 0.331 | 0.476-0.478 | 0.69x | 0.382 | 0.87x |
| 65,536 | 0.620-0.626 | 0.770-0.776 | 0.81x | 0.627-0.628 | 0.99x |
| 98,304 | 0.909-0.911 | 0.979-0.981 | 0.93x | 0.863-0.864 | **1.05x** |
| 131,072 | 1.196-1.198 | 1.264-1.265 | 0.95x | 1.105-1.106 | **1.08x** |

`WPO = 4` loses at every depth measured -- better than the rejected
zero-padded kernel's 1.73x-slower by a wide margin (it is 1.05-2.1x slower
here rather than 1.73x, and the reason is the same fp16 `Q` rounding no
longer being the differential's problem, not a performance fix -- the shared
memory and occupancy numbers above did not change), but still not a win
anywhere. `WPO = 2` is a genuine win at 98,304 and 131,072 keys and a loss
everywhere shallower than that, crossing over between 65,536 (a wash) and
98,304.

### End-to-end performance, `bench_decode`, `LLMXABE_DECODE_CHUNK=8192`, 48 steps, GPU 1

**131,072 keys, 3 interleaved rounds (the depth `WPO=2` wins at kernel level):**

| round | baseline (warp) ms/step | `WPO=2` ms/step | ratio |
|---:|---:|---:|---:|
| 1 | 19.65 | 19.25 | 1.021x |
| 2 | 19.77 | 19.34 | 1.022x |
| 3 | 19.77 | 19.35 | 1.022x |

Consistently **~1.02x** end to end -- a real, reproducible win, but far
smaller than the kernel-level 1.08x: attention is a fraction of a decode
step even at this depth, exactly as `AGENTS.md`'s "~5% measured... at context
12..28" note and this file's own "attention increment is ~9.8 ms/step" sizing
imply for how much of the whole step a kernel-level win can move.

**65,536 and 32,768 keys, one round each (the depths kernel level already
predicted a loss):**

| depth | baseline (warp) ms/step | `WPO=2` ms/step | ratio |
|---:|---:|---:|---:|
| 65,536 | 14.70 | 14.99 | 0.981x |
| 32,768 | 12.13 | 12.66 | 0.958x |

Confirms the kernel-level crossover end to end: `WPO=2` costs 2-4% at these
two depths rather than saving anything.

### Against the target, honestly

The lead's own sizing: parity at 131,072 needs the kernel near 0.6 ms against
`attn_flash_decode_warp`'s 1.197 ms, roughly 2x. `WPO=2` measured 1.105 ms --
1.08x, an order of magnitude short of the 2x this would need to matter
against llama.cpp. Translated to the head-to-head this file tracks, 131,072
moves from roughly 51.2 to roughly 51.7-52.0 tok/s against llama.cpp's 66.9 --
the ratio moves from 0.766x to about 0.773-0.778x, not a visible change at
the precision this file reports ratios to. This is a real, correctness-
verified, reproducible kernel win at the two deepest measured contexts and
not the fix that closes the deep-context gap.

### Disposition

Not made the default. `AttentionKernels::decode_mma` stays `0`
(`attn_flash_decode_warp`) unless a caller explicitly opts in via
`disable_decode_mma`/`set_decode_mma_wpo` -- both kernels are net regressions
below roughly 98,304 keys, and this codebase has no depth-aware dispatch
inside a single `AttentionKernels` instance to route only the deep steps onto
it. The kernel, its correctness gate, and the `LLMXABE_DECODE_MMA_WPO`/
`LLMXABE_DISABLE_DECODE_MMA` benchmark levers all ship; nothing is reverted,
because unlike the tiled-softmax attempt above this one has a real, if narrow,
place it wins.

### What a follow-up needs

1. **The 255-register spill is unexamined.** Both variants hit it only after
   the correctness fix widened `vlo`/`vhi`; whether it is costing `WPO=2` any
   of its margin, or whether removing it would turn `WPO=4` into a win too,
   is not measured. `MMA_PREFETCH`'s own history in this file (register
   pressure fighting `__launch_bounds__` occupancy) is the first thing to
   check before assuming less register pressure is strictly better.
2. **Depth-aware dispatch.** `WPO=2`'s crossover sits between 65,536 and
   98,304; a real deployment wants `decode()` choosing per call by depth, not
   one lever fixed at construction for the whole pass. Nothing in
   `AttentionKernels` currently reads `key_offset` before picking a kernel --
   it would have to.
3. **`ncu` would resolve why `WPO=4` is so much worse than `WPO=2` beyond the
   occupancy story alone** -- 1 vs 3 blocks/SM predicts *some* of the gap,
   but 0.95x vs 1.08x at 131,072 is a 14-point swing from a 3x occupancy
   change, which is plausible but not verified against per-SM issue-slot
   counters this host cannot read.

## Widening the softmax-rescale tile: built, and rejected on register spill (2026-08-18)

Profiling task for the deep-prefill gap: `nsys --delay --duration` windows
over `bench_forward`'s chunked-prefill run at `LLMXABE_BENCH_N=131072
LLMXABE_BENCH_CHUNK=8192`, one window at a fresh 8,192-token chunk (depth 0)
and one at the tail chunk (depth ~122,880-131,072), GPU 0. `cuda_gpu_trace`
summed by kernel name settles which of attention or MoE/GDN the deep-depth
ratio decay lives in, no guessing required:

| kernel | shallow chunk, % of kernel time | deep chunk, % of kernel time |
|---|---:|---:|
| `attn_flash_causal_mma` | 8.09% | **71.79%** |
| `moe_expert_ffn_mma` | 18.71% | 5.71% |
| `moe_expert_down_mma` | 12.17% | 3.68% |
| `gdn_scan_prefill` | 18.80% | 6.22% |
| `mma_q8_0_proj_split` | 20.15% | 6.18% |

MoE and GDN's *absolute* per-chunk cost barely moves between the two
windows (they are O(chunk), not O(depth)); attention's does, from a small
fraction of a shallow chunk to nearly three-quarters of a deep one. The
deep-prefill gap is where the earlier sections already narrowed it to:
attention's own per-token cost growing with depth, not a MoE or GDN
regression at width. This settles which of the two candidate directions in
this session's brief to chase.

### The structural difference against `fattn-mma-f16.cuh`

llama.cpp's Turing config for `DKQ=DV=256` (`ggml_cuda_fattn_mma_get_config_turing`,
`fattn-mma-f16.cuh:91`) sets `nbatch_fa = 64`: it rescales the running
softmax max, normalizer and `P V` accumulator once per 64 keys.
`attn_flash_causal_mma` rescales once per `MMA_KT` keys, which is 8 at the
shipped `MMA_HPB` 8 -- **eight times finer-grained** than llama.cpp's
kernel on the same head dimension. Read closely, `MMA_KT` 8 is not a
traffic decision at all: `MMA_HPB` (heads sharing one staged K/V tile,
which *is* the traffic lever measured in the "Tensor cores" section above)
and `MMA_KT` (keys between one barrier pair and the next, which costs
nothing in traffic, only in how many times the fixed per-tile overhead is
paid) are two different axes that happen to be tied together by the
current code: `MMA_KT == 8 * MMA_WPH` and `MMA_WPH == 8 / MMA_HPB`, so
pinning `MMA_HPB` at 8 for the traffic win pins `MMA_WPH` at 1 and `MMA_KT`
at 8 as a side effect, not by anything the traffic argument requires.
llama.cpp reaches 64 by looping a warp over several 16-key `ldmatrix`
batches before its one softmax pass, independent of how many Q columns
share a KV tile -- exactly the two axes this kernel conflates.

At a 122,880-key deep-chunk window that is 15,360 outer trips against
llama.cpp's ~1,920 for the same keys, each trip paying two barriers
(`__syncthreads` for the staged-tile handoff, `MMA_HEAD_BAR` for the
softmax) and a full accumulator rescale (`o[t][k] *= cg` over `MMA_MAXT`
32 tiles) that is fixed-cost per trip and does not shrink with a narrower
tile. Decoupling the two axes -- keep `MMA_HPB` 8 for the traffic win, add
a serial inner loop so one warp covers `MMA_KEY_TRIPS` octets before the
rescale, widening `MMA_KT` to `8 * MMA_KEY_TRIPS` without touching
`MMA_HPB` or `MMA_WPH` -- is untried in every earlier section's rejected
list (which covers `ldmatrix`, `m8n8k4`, a two-deep staged tile, a
grid-axis swap, wider *query* tiles, and `__maxnreg__`, none of which is
this).

### Built, correct in shape, measured, and found to spill

The `Q K^T` computation loop was changed from one octet per warp per trip
to `MMA_KEY_TRIPS` octets computed serially into `my_s` before the
existing (unmodified) softmax-reduce and `P V` phases, which were already
generic in `MMA_KT` from an earlier session's work at lower `MMA_HPB` and
needed no change. `MMA_KEY_TILE` becomes `8 * MMA_WARPS_PER_HEAD *
MMA_KEY_TRIPS`; the K/V staging macro, the shared-memory sizing, and the
value stride are all already parametrized on `MMA_KEY_TILE` and scale
automatically. Host-side structural tests (`the_tensor_core_block_shape_is_one_choice_and_not_three`,
`the_staged_tiles_fit_the_shared_memory_that_was_opted_in_to`) updated and
green; shared memory at `MMA_KEY_TRIPS` 2 is 30,464 B and at 4 is 55,296 B,
both under the 65,536 B carveout, so the launch was never going to fail --
the failure was somewhere `ncu` would normally show and `nsys` cannot.

`nvcc -arch=compute_75 -code=sm_75 -Xptxas -v` on the extracted kernel
source (the same substitute for `ncu`'s `ERR_NVGPUCTRPERM` the MoE
prefetch section used) settles it without guessing:

| `MMA_KEY_TRIPS` | `MMA_KT` | shared bytes | registers | spill stores | spill loads |
|---:|---:|---:|---:|---:|---:|
| 1 (shipped) | 8 | 13,952 | 252 | 0 | 0 |
| 2 | 16 | 30,464 | 255 | 12 B | 12 B |
| 4 | 32 | 55,296 | 255 | 388 B | 312 B |

The shipped kernel already sits at 252 of 255 registers with zero spill --
essentially no headroom at all, not the ~217/255 an earlier section
estimated before Q actually moved into registers. `MMA_KEY_TRIPS` 2 adds a
handful of live registers (a wider `kreg`/`vlo`/`vhi` prefetch, sized
`MMA_KREG`/`MMA_VREG`, both proportional to `MMA_KT`) and immediately
spills; `MMA_KEY_TRIPS` 4 spills by 30x more. `bench_attention`,
`LLMXABE_ATTN_CHUNK=8192`, GPU 0:

| key_offset | shipped (ms) | `MMA_KEY_TRIPS` 2 (ms) | `MMA_KEY_TRIPS` 4 (ms) |
|---:|---:|---:|---:|
| 65,536 | 315.3 / 330.9 / 334.1 | 325.4 / 334.1 / 334.7 | 620.1 |
| 131,072 | 626.9 / 663.6 / 673.7 | 641.9 / 672.6 / 674.8 | 1235.7 |

Three interleaved pairs at `MMA_KEY_TRIPS` 2: 1.024x, 1.014x, and 1.002x
slower than shipped at 131,072 -- a wash trending slightly negative, the
signature of a 12-byte spill roughly cancelling the barrier count it saves.
`MMA_KEY_TRIPS` 4's 388-byte spill is not subtle: **1.96x slower**, one
pair, no interleaving needed to see it. Spilling in the trip that exists
to keep the DRAM round trip covered pays the exact latency the kernel is
built to hide, a second time -- the same mechanism the MoE prefetch
section named for a different kernel, on this kernel too.

### Not shipped

Reverted with `git checkout -- crates/xabe-cuda/src/kernels/attention.rs`,
zero diff against the parent commit. The mechanism this section names --
`attn_flash_causal_mma` runs at 252/255 registers already, so any register
lever, register-pipelined prefetch (MoE section) or a wider softmax-rescale
tile (this section) alike, spills on this kernel specifically -- is worth
keeping distinct from the MoE section's identical-shaped finding, because
the two kernels reach the same wall by different roads: MoE's
`__launch_bounds__` trades registers for occupancy on purpose and a
register lever fights that trade; this kernel was never trading for
occupancy, it simply has no register budget left after `Q` moved into
registers to win the shared-memory carveout that `MMA_HPB` 8 needed. A
future attempt at this specific lever needs registers freed elsewhere
first -- `Q` is the only large holder at 64 registers, and moving it back
to shared is foreclosed by the same shared-memory arithmetic that put it
in registers to begin with (`q_sh` alone would want 67,584 B against the
65,536 B carveout at `MMA_HPB` 8). That forecloses this specific lever
without a block-shape change bigger than a tuning knob, which is not what
this section attempted.

### What this leaves for deep prefill

Every named lever this file knows for `attn_flash_causal_mma` -- traffic
(`MMA_HPB`), two arithmetic substitutions (`exp2f`, `__syncwarp`), and now
the softmax-rescale tile width -- is spent, tried, or structurally
foreclosed. The kernel profiles at 71.79% of a deep chunk's kernel time,
so MoE and GDN's combined 15.79% at that same depth is the remaining
lever with headroom: their per-chunk cost is fixed regardless of depth, so
a win there moves every depth's ratio by the same absolute amount, which
matters more at 8,192-32,768 (where MoE alone was 18.71%+12.17% = 30.9% of
a shallow chunk, against attention's 8.09%) than at 131,072 where
attention already dominates. The MoE MMA pair is the next section.

## Widening M on the routed-expert MMA kernels: a real win, and a real trade (2026-08-18)

Shared-memory double-buffering -- the untried lever this session's brief
named alongside the register prefetch the previous session already
rejected -- was ruled out by arithmetic before writing any code. Turing
gives an SM 65,536 B; `moe_expert_ffn_mma`'s current footprint is 21,760 B
and `__launch_bounds__` already asks for three resident blocks,
`3 * 21,760 = 65,280`, 256 B of slack. A double-buffered staging pipeline
needs roughly double that per block for the tiles it pipelines: even
buffering only the weight tile (16,384 -> 32,768 B) leaves no room for
three blocks, and a full double buffer of every staged tile is
`2 * 21,504 + 256 = 43,264` B, which admits **one** block per SM, not two
or three. This kernel's own comment names the mechanism: it is
bandwidth-bound *through occupancy* -- "what it needs from the scheduler
is loads in flight, and what puts loads in flight is resident warps" -- so
a 3x occupancy cut to buy latency-hiding that occupancy was already
providing is the same trade the register-prefetch section rejected, on
the same kernel family, for the same reason. Not built, because the
arithmetic already answers it and AGENTS.md is explicit that a speculative
kernel should not be built before the mechanism is named with numbers.

The lever this session did build is named in the same brief: "tile-shape
changes that raise arithmetic intensity (wider M or N per block, fewer
redundant dequants)." `MOE_MMA_M` -- the dispatch slots one block's staged
weight tile is shared across -- was 32, and `forward.rs`'s own
`MOE_BLOCK_SIZE` was already dispatching at exactly that ceiling. Widening
`M` to 64 doubles how many routed tokens amortize one staged weight-tile
load before it is discarded, which is where this kernel's traffic actually
goes (its own comment: "at 512 tokens it moves 470 MB of Q6_K per layer").

### The occupancy this costs, computed the same way as the double-buffer's arithmetic

`MOE_MMA_M` 64 doubles `sa`, `sas` and `rows` (the tiles that scale with
staged tokens, not with the weight tile), taking `moe_expert_ffn_mma`'s
footprint from 21,760 to 27,136 B. `3 * 27,136 = 81,408` is past the
65,536 B ceiling, so `MOE_MMA_BLOCKS_PER_SM` drops 3 -> 2 (`2 * 27,136 =
54,272`, 11,264 B to spare). `moe_expert_down_mma`'s footprint goes
14,720 -> 20,224 B; `4 * 20,224 = 80,896` is past the ceiling too, so
`MOE_DOWN_BLOCKS_PER_SM` drops 4 -> 3 (`3 * 20,224 = 60,672`, 4,864 B to
spare). Both are a real, named occupancy cost, unlike the double-buffer's
which would have been a 3x or 4x cut -- this is 1.5x and 1.33x -- and both
came with more register headroom per thread (85 -> 128 at two blocks
instead of three), which absorbed `MOE_MMA_MF`'s accumulator arrays
doubling (4 -> 8) with no spill: `nvcc -Xptxas -v` on the extracted source
confirms both kernels compile clean at the new shape.

All six `moe_differential.rs` tests pass unchanged, including
`device_grouped_forward_matches_the_reference_on_real_expert_weights` at
its existing gate; `moe_block.rs`'s eleven tests pass, including the one
that exercises block 39's Q8_0 exception path
(`moe_expert_ffn_mma_q8`, left at its own unguarded register allocation
since it runs one layer in forty and was never occupancy-tuned);
`int8_forward.rs` and `the_forward_pass_reproduces_llama_cpps_logits_and_its_argmax`
both pass at unchanged tolerances on the final tree.

### Isolated, `bench_moe_mma`, three interleaved pairs, GPU 0

Old code cannot dispatch above `block_size` 32 (the geometry check
rejects it), so the honest before/after compares each side at its own
production width -- `block_size` 32 against the old ceiling, 64 against
the new one -- rather than holding width fixed at a number only one side
could reach in practice:

| tokens | old (`M` 32, `block_size` 32) ms | new (`M` 64, `block_size` 64) ms | ratio |
|---:|---:|---:|---:|
| 512 | 3.545 / 3.774 / 3.551 | 4.691 / 4.695 / 4.697 | **0.772x (29.6% slower)** |
| 8,192 | 21.983 / 24.480 / 23.950 | 20.121 / 21.359 / 20.978 | **1.127x** |

The 512-token loss and the 8,192-token win are both real, and the
mechanism for the loss is occupancy, not the wider tile wasting bytes on
padding: at 512 tokens and 8 experts/token, an average expert sees only
16 routed tokens, well under even the *old* `M` 32, so the wider tile buys
no traffic reduction there and only pays the occupancy cut. Checked
directly rather than inferred -- the new kernel (`M` 64) dispatched at the
*old* `block_size` 32 measures 4.630 ms, matching the `block_size` 64
number rather than recovering the old 3.62 ms mean, which rules out tile
padding as the cost: the loss is `MOE_MMA_BLOCKS_PER_SM`'s occupancy cut,
paid regardless of how many of the wider tile's slots are actually live.

### End to end, `bench_forward`, GPU 0, `git worktree`-isolated baseline

`git stash` was tried first for the baseline A/B and abandoned mid-session:
this checkout is shared with sibling agents actively editing the same
files, and a stash/pop window is exactly the race that could silently
clobber their concurrent work. `git worktree add --detach` against `HEAD`
builds an isolated baseline binary with no shared mutable state; every
number below an isolated-worktree baseline against the working tree's
build, not a stashed-and-restored one.

| tokens | chunk | before (tok/s) | after (tok/s) | ratio |
|---:|---:|---:|---:|---:|
| 512 | 512 | 2,410.7 | 2,228.6 | 0.924 |
| 2,048 | 2,048 | 3,096.9 | 3,095.8 | 1.000 (wash) |
| 8,192 | 8,192 | 3,082.8 (mean of 3) | 3,201.9 (mean of 3) | **1.039** |
| 32,768 | 8,192 | 2,417.4 (mean of 2) | 2,502.2 (mean of 2) | **1.035** |
| 65,536 | 8,192 | 1,908.5 | 1,965.2 | **1.030** |
| 131,072 | 8,192 | 1,355.1 | 1,382.0 | **1.020** |

2,048 is a wash: wide enough (64 average tokens/expert) that the traffic
win and the occupancy cost roughly cancel. 512 is the one real regression,
diluted end to end from the isolated kernel's 29.6% to 7.6% because MoE is
a smaller share of a 512-token pass than of an 8,192-token one.

### Against llama.cpp, both sides fresh, both at their own best width

llama.cpp figures are this session's own re-measurement at `-b 8192 -ub
4096` (`-ub 512` at 512), matching the convention "The baseline was
llama.cpp's default, not its best" established earlier in this file:

| tokens | llmxabe (before) | llmxabe (after) | llama.cpp | ratio before | ratio after |
|---:|---:|---:|---:|---:|---:|
| 512 | 2,410.7 | 2,228.6 | 2,155.8 | 1.12x | 1.03x |
| 2,048 | 3,096.9 | 3,095.8 | 2,951.2 | 1.05x | 1.05x |
| 8,192 | 3,082.8 | 3,201.9 | 3,077.9 | 1.00x | **1.04x** |
| 32,768 | 2,417.4 | 2,502.2 | 2,506.2 | 0.96x | **1.00x** |
| 65,536 | 1,908.5 | 1,965.2 | 1,935.7 | 0.99x | **1.02x** |
| 131,072 | 1,355.1 | 1,382.0 | 1,439.5 | 0.94x | 0.96x |

Both of this session's target depths cross or reach parity: **8,192 to
1.04x** and **32,768 to 1.00x**, both up from short of it. 65,536 crosses
too, to 1.02x, though it was not the primary target. 131,072 improves from
0.94x to 0.96x but stays short -- consistent with the profiling at the top
of this session's work: attention is 71.8% of a deep chunk's kernel time
there, and MoE's combined 9.4% at that same depth has much less room left
to move the ratio by than it does at 8,192-32,768, where MoE was
18.71%+12.17% = 30.9% of the chunk. 512 is a real, named cost of this
change -- down from 1.12x to 1.03x -- but stays a win against llama.cpp, so
it was judged worth shipping rather than reverting: the two depths this
session was asked to close are closed, and the one depth that regressed
was never the target and remains ahead.

### What is left

Deep prefill (65,536-131,072) is now bounded by attention alone, per the
previous section's profiling and the register-spill reject that closed
off the one lever this file knew for it. 512-token MoE throughput could
recover its regression with a second kernel variant compiled at `M` 32
for narrow batches, selected by token count the way the fp32/int8
crossover at `MMA_MIN_TOKENS` already is -- specified here, not attempted,
because 512 remains a win against llama.cpp and this session's two named
targets do not need it.

## MoE's small-bucket GEMV at batch decode: a real win through N=4, a real regression at N=8, and rejected on the second one (2026-08-18)

**Superseded by the next section.** The "What a follow-up needs" close of
this section named the fix exactly: a genuinely separate `__global__`
kernel rather than a same-function runtime branch. That was built, and it
does what this section predicted -- N=8 is bit-identical SASS and pure
noise, N=2-4 keep the win. Left in place rather than deleted, per this
file's own append-only convention.

This session's brief named the batched-decode aggregate at N=3 (89.2 tok/s
against llama.cpp's 154.58) as the largest remaining gap and asked for
`grouped_forward`'s decode-width GEMV shape to be re-profiled at N=3
specifically rather than assumed from N=1's numbers. `nsys
--cuda-graph-trace=node` over an isolated N=3 batch-decode replay (`LLMXABE_
SKIP_SINGLE_STREAM=1`, GPU 2, 32,768-token context), kernel time summed per
36-replay window and divided by step count: MoE (`moe_expert_ffn` +
`moe_expert_down` + shared-expert + routing) is **43.7%** of a decode step at
this width -- 19.0% and 14.9% for the two routed projections alone -- ahead
of GDN's 23.0% and attention's 24.9%, confirming the brief's own ranking
without assuming it.

### The idea: `MOE_TILE_DISPATCH`'s `bm == 1` case does not need to stage

`tile_gemm_pair`/`tile_gemm_single` (`crates/xabe-cuda/src/kernels/moe.rs`)
already specialize to `TM` 2, 8 or 16 by the live row count `bm` a dispatch
bucket carries, per "Two more decode-shape defects at N=2-4" above. At N=3,
`D(3) = 23.26` distinct experts per layer against `3 * 8 = 24` routed
token-expert pairs means almost every touched expert's bucket holds exactly
one real token -- `bm == 1` -- and that bucket still takes the `TM` 2
specialization, which stages its (at most two) activation rows through
shared memory behind two block-wide `__syncthreads()` per 128-element pass.
That staging exists to save `MOE_ROWS`-fold (8x) re-reads of a *wide*
activation slice at `TM` 8 or 16; at `TM` 1 the slice is one `float4`, small
enough that the redundant per-warp reads should hit L2 for free, and the two
barriers are pure overhead paid for nothing. `moe_expert_ffn_gemv` -- the
existing dedicated N=1 kernel -- already proves the no-staging, redundant-
read pattern works at 46-63% of roofline; this added a `TM == 1` branch
inside `tile_gemm_pair`/`tile_gemm_single` themselves (used by the *batched*
`bm == 1` case, not the single-token dispatch) that does the same thing:
reads `rows[0]` directly from the already-synced shared array, then loads
its activation with `*(const float4*)(src + rows[0] + j0 + 4*lane)` instead
of through `prefetch_tile`/`commit_tile`. Bit-exact with the staged form by
construction -- same `float4` value either way, same accumulation order,
`x + 0.0f == x` for a zero-filled padding contribution -- confirmed by
`tests/moe_differential.rs` (6/6) and all three `tests/batch_decode.rs`
differentials, including the bit-exact one, both before and after every
revision below.

### First measurement: a real win, N=3 through N=4

Built in an isolated `git worktree` with its own `target/` -- this session's
own working tree had a sibling's concurrent, uncommitted edits to
`attention.rs` land and vanish mid-session, and an earlier nsys profile
silently absorbed one of them into this session's own binary and produced an
unrelated 39%-slower attention kernel choice that had nothing to do with
this change; the worktree isolates the measurement from that shared-checkout
hazard. `bench_decode_batch`, 2,048-token context, three interleaved rounds
against an unmodified baseline built the same way:

| N | before (mean, 3 rounds) | after | change |
|---:|---:|---:|---:|
| 1 | 101.8 | 101.6 | flat (noise) |
| 2 | 101.5 | 107.0 | +5.4% |
| 3 | 89.9 (32,768 ctx, separate rounds) | 93.2 | +3.7% |
| 4 | 129.2 | 136.6 | +5.7% |
| 8 | 170.7 | 154.4 | **-9.5%** |

N=3's own number is from the 32,768-token context this session's brief
targets, not 2,048; three interleaved rounds each side, mean 89.9 -> 93.2.

### The regression, and why the obvious fix does not fix it

N=8 is one of this project's five standard batch widths and the brief's own
"never regress" clause covers it, so a change that helps 1-4 and hurts 8 is
not a fix. The first guess was the same class of defect `bm == 2`'s
redundant *second* `float4` read would have -- more simultaneous buckets at
higher `N` (`D(8) = 57.42` against `D(3) = 23.26`) meaning more blocks
competing for L2 capacity with their redundant per-warp reads, worse at two
rows than at one. Restricting the no-staging path to `bm == 1` only (`bm ==
2` falls back to the unchanged staged `TM` 2 path) did not fix it: N=8 stayed
at 154.4 tok/s, unchanged from the unrestricted version.

A device-side-only gate cannot fix a device-side-only cause, so the next
attempt made the gate a host-known one: `valid_tokens` (the batch's real
token count, already a kernel argument, written once per step by
`MoeKernels::set_valid_tokens`) is `N` directly, and a `narrow =
(*valid_tokens) <= MOE_NARROW_DECODE_MAX` (4) local, computed once per kernel
invocation and threaded into `MOE_TILE_DISPATCH`'s `bm == 1` arm (`bm > 1 ||
!narrow`), disables the no-staging path entirely above the threshold --
falling back to *exactly* the original three-specialization dispatch,
verified against `set_valid_tokens`'s own definition rather than assumed.
N=4 kept its win (135.5-136.6 tok/s, both builds). **N=8 stayed regressed:
153.8-154.4 tok/s**, run three times, fresh GPU (37 C, 0% util before each
run) to rule out thermal drift as the cause.

That the runtime-gated build regresses identically to the ungated one, at a
batch width where the new branch is provably never taken, says the cost is
not in taking the branch -- it is in the branch *existing* in the compiled
kernel. `ptxas -v` on the extracted `MOE_SRC` (both versions, `nvcc -arch=
sm_75 --ptxas-options=-v`, the offline substitute this project already uses
where `ncu` fails with `ERR_NVGPUCTRPERM`) ruled out the specific mechanism
the project's own prior register-cliff note would predict: `moe_expert_ffn`
held at 80 registers before and after, `moe_expert_down` **dropped** from 77
to 75. No spill either side. The fourth-specialization register cliff this
file already documented in `MOE_TILE_DISPATCH`'s own comment does not
explain this one -- the registers did not move the wrong way, they barely
moved at all.

### Rejected, not root-caused

Whatever costs 9.5% at N=8 from a branch that is never taken there is real,
reproducible (three separate fresh measurements, same number to within
0.6%), and not explained by the two mechanisms this project's toolchain can
see without `ncu` -- register pressure and shared memory, both checked and
both clear. Instruction-cache pressure from a larger compiled function body
is the remaining plausible candidate, unconfirmed: `cuobjdump -sass`
per-function instruction counts were attempted and abandoned this session
after the extraction script produced obviously-wrong counts (18 and 30
instructions for kernels that are hundreds of instructions long), which is a
tooling gap here rather than a finding.

Reverted in full: `crates/xabe-cuda/src/kernels/moe.rs` is unchanged from
before this section. The N=3/N=4 win was real but conditioned on a
regression this project's own stated bar does not allow to ship. Recorded
here, with both sets of numbers, so the next attempt starts from "N=8 is the
wall, and registers/shared memory are cleared" instead of re-deriving it --
and does not re-try the `bm == 2` or host-known-`narrow` gates, both tried,
both insufficient on their own.

### What a follow-up needs

A genuinely separate `__global__` kernel (its own cubin entry, own register
allocation, selected by the host the way `max_tokens == 1` already selects
`moe_expert_ffn_gemv` over the tiled kernel) rather than a same-function
runtime branch is the next thing to try, following this file's own
"512-token MoE throughput could recover its regression with a second kernel
variant... selected by token count" precedent two sections up -- the same
pattern, applied to decode's `bm == 1` case instead of prefill's `M` 64
regression. That doubles the compiled code for `moe_expert_ffn`/
`moe_expert_down` (and, if it is worth carrying there too, the shared-expert
pair), which is a real cost this session did not have the budget to build
and verify against every accuracy gate in addition to everything above.
`cuobjdump -sass` with a working per-function instruction-count extraction
(this session's own attempt was not one) would settle whether instruction
cache is really the mechanism before spending the effort.

## Two compiled widths instead of one: recovering 512's regression without giving back the deep win (2026-08-18)

The previous section's win (`MMA_M` 32 -> 64 on `moe_expert_ffn_mma`/
`moe_expert_down_mma`) crossed 8,192 and 32,768 to parity but cost 512
tokens 7.6% end to end (2,410.7 -> 2,228.6 tok/s, 0.924x), because the
wider tile's occupancy cut (`MOE_MMA_BLOCKS_PER_SM` 3 -> 2) is paid
whether or not a batch is wide enough to fill it. That section's own "What
a follow-up needs" named the fix: compile both widths and pick by token
count at dispatch time, the same pattern used elsewhere in this file for
decode's small-bucket case. This section builds that.

### Two kernels from one body, via a template

`moe_expert_ffn_mma`/`moe_expert_down_mma`'s staging-and-MMA bodies moved,
verbatim apart from a mechanical `MOE_MMA_M` -> `M` / `MOE_MMA_MF` -> `MF`
substitution, into `template<int M, int MF> __device__ __forceinline__`
functions (`moe_expert_ffn_mma_impl`/`moe_expert_down_mma_impl`) declared
outside `extern "C"`, the same shape this file's `tile_gemm_pair<TM>`
already uses -- a template cannot carry C linkage, so the entry points
that need one are now thin `extern "C" __global__` wrappers just inside
it:

```
__global__ void __launch_bounds__(MOE_MMA_WARPS * 32, 3)
moe_expert_ffn_mma_narrow(...) { moe_expert_ffn_mma_impl<32, 4>(...); }

__global__ void __launch_bounds__(MOE_MMA_WARPS * 32, MOE_MMA_BLOCKS_PER_SM)
moe_expert_ffn_mma(...) { moe_expert_ffn_mma_impl<MOE_MMA_M, MOE_MMA_MF>(...); }
```

and the same pair for `moe_expert_down_mma`/`moe_expert_down_mma_narrow`.
The narrow entry points keep the pre-widening occupancy (3 blocks/SM ffn,
4 down); the unsuffixed ones are byte-for-byte what the previous section
shipped, since `MOE_MMA_M`/`MOE_MMA_MF` are unchanged at 64/8.
`moe_expert_ffn_mma_q8` (the Q8_0 exception path, rare and unguarded) was
left as a direct, unsplit kernel -- it never sees a narrow dispatch in
practice and splitting it would be doubling a path this session has no
evidence needs it.

`MoeKernels::new` now picks a module function name and a shared-memory
size from `geometry.block_size <= MMA_M_NARROW` (32): the narrow pair
below that, the wide pair above it. This is a construction-time choice,
not a per-launch one -- `MoeGeometry`/dispatch-table sizing is already
fixed per `MoeKernels` instance, so the width is chosen once, from
`xabe_engine::forward::moe_block_size(tokens)`, when a `Forward` is built
for a given token count.

### Register cost of the split: none

Compiled standalone with `nvcc -arch=sm_75 -Xptxas -v` against the
`MOE_SRC` string extracted straight from the shipped source (not a stale
scratch copy):

| kernel | registers | spill |
|---|---:|---:|
| `moe_expert_down_mma_narrow` (M=32) | 64 | 0 |
| `moe_expert_down_mma` (M=64) | 64 | 0 |
| `moe_expert_ffn_mma_narrow` (M=32, MF=4) | 80 | 0 |
| `moe_expert_ffn_mma` (M=64, MF=8) | 126 | 0 |

Zero spill on all four. The down kernel's register count does not move
with `M` -- its accumulator lives in a `[MF][2]` array the compiler folds
the same way regardless -- and the ffn kernel's is exactly the two prior
sections' own separately-measured 80 (old, single-variant) and 126 (new,
single-variant) numbers, unchanged by existing side by side under
different names.

### NVRTC compile time and module size: not material

The `bench_forward` binary built against this section's dual-variant tree
is 12,599,184 bytes against 12,595,216 for the single-variant tree it
branched from -- a 3,968-byte (0.03%) difference, almost all of it the
second kernel body as a string literal. `Forward::new`'s own reported
build time (NVRTC compile of every kernel this pass uses, MoE included) at
512 tokens: 15.6 s single-variant, 15.2 s dual-variant, the difference
smaller than the run-to-run spread either tree shows on its own. Compiling
two widths instead of one did not blow either budget the lead's brief
gated this on, so there is no case for reverting to a single variant here.

### Where the crossover actually is

`bench_moe_mma`, `MoeGeometry.block_size` set explicitly to 32 and 64 at
each token count rather than through `moe_block_size` (which is what this
measurement is meant to calibrate), two interleaved rounds, GPU 0:

| tokens | narrow (ms) | wide (ms) | narrow vs wide |
|---:|---:|---:|---:|
| 896 | 4.131 / 4.138 | 5.035 / 5.037 | narrow **21.8% faster** |
| 1,024 | 4.717 / 4.069 | 4.629 / 4.299 | narrow ~1.6% faster (noisy: round 1 says wide, round 2 says narrow) |
| 1,152 | 4.480 / 4.466 | 4.537 / 4.477 | wash, narrow ~0.8% ahead |
| 1,280 | 4.765 / 4.878 | 4.521 / 4.532 | wide **6.1% faster** |

The originally-sketched 1,024-token boundary undersold the narrow tile:
at 1,024 it is still at worst a wash and at best clearly ahead, and 1,152
is a wash too. Only at 1,280 does the wide tile pull unambiguously clear.
`moe_block_size` returns 32 for `tokens <= 1,152` and 64 above it -- the
top of the measured wash rather than the middle of it, so a token count
landing in the noisy 1,024-1,152 band never picks the tile that measured
behind at either endpoint.

### A coverage gap the split opened, and closed

`moe_differential.rs`'s grouped-GEMM gate,
`device_grouped_forward_matches_the_reference_on_real_expert_weights`, has
always run at the file's fixed `BLOCK_SIZE` (16). Before this section that
was fine -- 16 fits under the single `MMA_M` (32, at the time) either way
-- but after the split, 16 always resolves to the *narrow* variant, and
the wide (M=64) kernel -- the one carrying the 8,192-131,072 win -- had no
differential-level correctness gate left at all. The test's body is now a
private helper taking `MoeGeometry` as a parameter, called once at the
original `geometry()` (narrow) and once more at `block_size: 64` (wide)
from a new test, `device_grouped_forward_matches_the_reference_at_the_wide_mma_width`.
Both pass at this file's existing tolerances (`ROUTED_GATE`,
`ROUTED_MMA_GATE`) with no changes to either.

### End to end, `bench_forward`, GPU 0, `git worktree`-isolated pair

Both binaries built from a `git worktree` at the previous section's commit
(`890377d`) -- one unmodified (single-variant, the exact tree that
measured 2,228.6 at 512), one with this section's four-file diff applied
on top (`moe.rs`, `bench_moe_mma.rs`, `forward.rs`, `moe_differential.rs`)
-- rather than against the working tree directly, since the working tree
also carries a sibling agent's unrelated, uncommitted decode-path WIP in
`attention.rs` that has no business in this comparison.

| tokens | chunk | before (single M=64, tok/s) | after (dual, tok/s) | ratio |
|---:|---:|---:|---:|---:|
| 512 | 512 | 2,260.9 (mean of 3) | 2,428.8 (mean of 3) | **1.074** |
| 8,192 | 8,192 | 3,204.7 (mean of 2) | 3,191.5 (mean of 2) | 0.996 (wash) |
| 32,768 | 8,192 | 2,484.2 | 2,483.2 | 1.000 (wash) |
| 65,536 | 8,192 | 1,982.6 | 1,985.6 | 1.002 (wash) |
| 131,072 | 8,192 | 1,368.5 (mean of 3) | 1,349.9 (mean of 3) | 0.986 (noise, see below) |

512 recovers past even the pre-widening M=32 tree's own historic 2,410.7
number, within what today's own run-to-run spread accounts for. 8,192 and
deeper are washes by construction, not by luck: `moe_block_size` selects
64 at every one of these depths in *both* trees, so the dual-variant
tree's "wide" kernel and the single-variant tree's only kernel are the
same compiled code, and every one of these four rows is measuring the
same kernel against itself. 131,072's 1.4%-low mean is smaller than the
spread its own three dual-tree rounds showed on their own (1,323.7 ->
1,356.4 -> 1,369.5 tok/s, a 3.4% span) -- the direct evidence that the
131,072 gap is measurement noise on a ~95-99 s run, not a regression from
compiling a second kernel variant.

### Against llama.cpp, both sides fresh, both at their own best width

llama.cpp figures are this workstream's own prior re-measurement at `-b
8192 -ub 4096` (`-ub 512` at 512), carried over from the previous section
unchanged:

| tokens | llmxabe (before) | llmxabe (after) | llama.cpp | ratio before | ratio after |
|---:|---:|---:|---:|---:|---:|
| 512 | 2,260.9 | 2,428.8 | 2,155.8 | 1.05x | **1.13x** |
| 8,192 | 3,204.7 | 3,191.5 | 3,077.9 | 1.04x | 1.04x |
| 32,768 | 2,484.2 | 2,483.2 | 2,506.2 | 0.99x | 0.99x |
| 65,536 | 1,982.6 | 1,985.6 | 1,935.7 | 1.02x | 1.03x |
| 131,072 | 1,368.5 | 1,349.9 | 1,439.5 | 0.95x | 0.94x |

512 moves from a 1.05x win to a clean 1.13x, recovering the 7.6%
end-to-end cost the widening section shipped and named -- and landing
above the 1.03x the previous section's own `git worktree`-isolated
measurement of the single-variant tree reported for the same row, which
this section's fresh 2,260.9 (against that measurement's 2,228.6) is
within this depth's own run-to-run spread of. 8,192 and
32,768 -- this workstream's two actual target depths -- hold exactly
where the previous section left them, at 1.04x and parity. 65,536 holds
its 1.02x. 131,072 moves from 0.95x to 0.94x, a change the isolated table
above already attributes to run-to-run noise on the identical wide kernel
rather than to anything this section built.

### Gates

`moe_differential`: 7/7, including the new wide-path test. Golden:
`the_forward_pass_reproduces_llama_cpps_logits_and_its_argmax` passes --
argmax token 25358 at logit 19.998243 against llama.cpp's 19.902241,
result_output cosine 0.999736, unchanged from before this section (the
golden prompt is 19 tokens, far below either kernel's dispatch-width
threshold either way it is set, so this section could not have moved it).
`cargo fmt --all -- --check` and `cargo clippy -p xabe-cuda -p xabe-engine
--all-targets` both clean.

### What is left

Deep prefill (65,536-131,072) is unmoved by this section, as expected --
attention is 71.8% of a deep chunk's kernel time (this workstream's
opening profile), and MoE's combined share at that depth is too small for
either the M-widening or this section's dispatch fix to close much of the
gap by. The one lever this workstream found for attention at depth --
widening its softmax-rescale tile -- was built, measured, and rejected on
register spill two sections up: the kernel already sits at 252/255
registers with zero spill at its shipped configuration, so widening the
tile trades occupancy away rather than buying anything, and the isolated
and end-to-end measurements there were a wash trending slightly negative.
No further named lever remains for 131,072 from this workstream; closing
it needs a structural change to the kernel (a Marlin-style staged
pipeline, or the double-buffering this file already ruled out by
arithmetic for MoE and did not re-attempt for attention), not a dispatch
choice.

### Disposition

Shipped. 512 recovers fully (0.924x -> effectively 1.0x-plus against its
own pre-widening baseline); 8,192 and 32,768, the two depths this
workstream was asked to close, remain at 1.04x and parity; 65,536 holds
its incidental win at 1.02x; 131,072 is unchanged within noise, still
short of parity at 0.94x, and bounded by attention rather than by
anything MoE-side has left to give.

## The decode tensor-core kernel's register spill, fixed by not doing what `attn_flash_causal_mma` does, and shipped as the default (2026-08-18)

The previous section on `attn_flash_decode_mma` left three things open: a
255-register spill the correctness fix had introduced, `DMMA_KT` widening
to match llama.cpp's 64-key softmax trip, and depth-aware dispatch so a
single `AttentionKernels` instance takes the right kernel at every depth
instead of one lever fixed for the whole pass. This section closes the
first and third and explains why the second was not attempted.

### The spill's mechanism, found by reading llama.cpp's Turing path rather than guessing

`attn_flash_decode_mma`'s K/V staging was cross-tile double-buffered the
same way `attn_flash_causal_mma` pipelines its own staging: a prologue
load, then each loop trip storing the *previous* trip's already-loaded
registers to shared and issuing the *next* trip's loads before computing
on the current tile. That shape needs `kreg`/`vlo`/`vhi` live
simultaneously with `o`/`qa0`/`qa1` for the whole `Q K^T` + softmax + `P V`
body, not just across the load-then-store gap. `fattn-mma-f16.cuh`
(`/home/nixabe/llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh`) does not do
this on Turing: `cp_async_available()` in `common.cuh` gates the
hardware-async copy path behind `GGML_CUDA_CC_AMPERE`, so SM75 always takes
`flash_attn_ext_f16_load_tile`'s plain-load branch, which is a same-tile
`ggml_cuda_memcpy_1<16>` straight from global to shared with no register
array held across a tile boundary at all -- the "prefetch" a Turing kernel
gets is whatever outstanding-request pipelining the hardware itself does
across nearby loads, not a programmer-held register buffer. Dropping the
cross-tile buffering and staging each tile's K/V into a register array
scoped to that tile alone (load, store to shared, and let the array go out
of scope before the compute phase begins) matches this, and removes the
period where the next tile's registers and the current tile's compute both
need to be live.

`nvcc -arch=sm_75 -cubin -Xptxas -v` on the extracted kernel, before (as
shipped in the previous section, the correctness fix's own regression) and
after this restructuring:

| variant | before (cross-tile buffered) | after (same-tile only) |
|---|---:|---:|
| `WPO = 4` | 255 registers, 16 B spill (stores + loads) | **235 registers, 0 spill** |
| `WPO = 2` | 255 registers, 52-56 B spill (stores + loads) | **252 registers, 0 spill** |

Re-verified on the real source, not just the scratchpad extraction: all
nine `attention_differential` tests pass with the selection forced to each
width in turn, at numbers identical to the pre-fix kernel's (`n_keys =
4,096/4,097/61`, `max_abs` unchanged to the printed digit) -- the
restructuring changed nothing about what the kernel computes, only how
long each register lives.

### `DMMA_KT` widening: not attempted, and why

The lead's second instruction was to try `DMMA_KT = 64` once registers
were free, matching llama.cpp's `nbatch_fa`. Freed registers went instead
to the correctness fix's own increased liveness -- `WPO = 2`, the width
that wins, is back down to only 3 registers of headroom (252 of 255), the
same wall the "Widening the softmax-rescale tile" section above hit on the
*prefill* kernel from a different cause (that kernel's `Q` living in
registers, not staging liveness). Widening the key trip needs more
simultaneous score and correction state per trip, which is exactly what
that section measured as spilling immediately even from a much smaller
register margin than this. `WPO = 4` has real headroom (20 registers), but
is the width that loses at every depth measured, both before and after
this fix -- spending that headroom to try `DMMA_KT` widening on the losing
width first was not judged worth a session on it; the finding is recorded
here rather than attempted and left unmeasured.

### Kernel-level performance, `bench_attention`, `LLMXABE_ATTN_CHUNK=1`, GPU 1, 3 interleaved rounds per depth, before vs after this fix

The previous section's crossover sat between 65,536 (0.99x, a wash) and
98,304 (1.05x). A finer sweep after the fix -- 2,048/8,192/12,288/16,384/
20,480/24,576/32,768/65,536/131,072 -- narrows it by nearly an order of
magnitude in depth:

| key_offset | `attn_flash_decode_warp` (ms, 3 rounds) | `WPO=2` (ms, 3 rounds) | ratio | ratio before this fix |
|---:|---:|---:|---:|---:|
| 2,048 | 0.074 / 0.076 / 0.074 | 0.095 / 0.095 / 0.094 | 0.79x | 0.74x |
| 8,192 | 0.125 / 0.126 / 0.126 | 0.137 / 0.136 / 0.135 | 0.92x | 0.73x |
| 12,288 | 0.160 / 0.160 / 0.159 | 0.167 / 0.167 / 0.168 | 0.95x | -- |
| 16,384 | 0.191 / 0.192 / 0.192 | 0.191 / 0.191 / 0.192 | **1.00x** | -- |
| 20,480 | 0.228 / 0.228 / 0.228 | 0.222 / 0.223 / 0.222 | 1.03x | -- |
| 24,576 | 0.263 / 0.263 / 0.262 | 0.248 / 0.250 / 0.250 | 1.05x | -- |
| 32,768 | 0.331 / 0.331 / 0.332 | 0.306 / 0.305 / 0.306 | 1.08x | 0.87x |
| 65,536 | 0.620 / 0.620 / 0.620 | 0.510 / 0.510 / 0.510 | 1.22x | 0.99x |
| 131,072 | 1.196 / 1.278\* / 1.197 | 0.930 / 0.927 / 0.930 | 1.29x\* | 1.08x |

\*131,072's second `attn_flash_decode_warp` round (1.278) is an outlier
against its own other two rounds (1.196, 1.197); the ratio column uses the
two agreeing rounds (1.197/0.929 = 1.29x). Including the outlier would
read 1.32x -- either way the win at the deepest measured point grew, it
did not shrink.

Every depth's ratio improved after the fix, and by more than the removed
spill bytes alone would suggest -- 16-56 B of spill is a handful of local
loads and stores per trip, not obviously a 9-20 point swing. The staging
restructuring itself (not carrying `kreg`/`vlo`/`vhi` across a tile
boundary) plausibly frees the compiler to schedule the load-then-store
sequence tighter even where it did not force a spill, but this session did
not `cuobjdump -sass` to confirm that beyond the register/spill counts
above; `ncu` remains unavailable (`ERR_NVGPUCTRPERM`) to settle it further.

### Depth-aware dispatch, shipped as the default

`AttentionKernels`'s tensor-core lever (`decode_mma`, an `AtomicU8` so it
stays `Sync` through the `Arc<AttentionKernelSet>` every layer shares)
changes meaning: `0` was "disabled" and is now `Auto`, `1` is the new
"disabled" (`ForceWarp`), and `2`/`4` still force that occupancy width at
every depth for `bench_attention`/`bench_decode`'s A/B levers.
`disable_decode_mma()` now stores `1`, not `0` -- callers of that method
are unaffected, only the meaning of the atomic's own zero value changed.

`AttentionKernels::forward` and `decode` gained a `key_depth: usize`
parameter -- the caller's host-side `positions[0] + n_query`, not
`max_keys` (the cache's allocated *capacity*, already a parameter, and a
different number: `bench_attention` fixes `max_keys` at the sweep's
deepest point for every row while `key_depth` varies per row, and
`GatedAttentionBlock::forward` already had the equivalent host value on
hand as `pos_offset + t` without needing to read anything back from the
device). `active_decode_mma(depth)` compares it against
`DECODE_MMA_DEPTH_THRESHOLD = 16,384` -- the point in the table above
where `WPO=2` stops losing (0.1917 ms vs 0.1913 ms, ~1.00x) -- and picks
`WPO=2` at or above it, `attn_flash_decode_warp` below. `WPO=4` never wins
at any depth measured before or after the fix, so `Auto` never selects it;
the forced lever is kept only for benchmarking, per the disposition below.

Threading `key_depth` cost four call sites: `GatedAttentionBlock::forward`
(passes `pos_offset + t`), its batched-decode sibling
`forward_batch_decode` (passes `pos_offsets[i] + 1`, per sequence),
`bench_attention` (passes `depth + chunk`, the same quantity `positions`
already encoded on the device), and the differential test's `run_device`
helper (passes `key_offset + n_query`). None of these change what any
kernel computes -- `key_depth` selects between two already-correct kernels,
it is not consulted for the bound check, which stays exactly where the
`forward` doc comment already says it has to live (device-side
`positions`, checked by the caller before launch).

### Correctness after the depth-aware default, and a coverage gap the default change opened

`device_decode_matches_the_reference_over_a_deep_window` only exercises
`n_keys = 4,096/4,097/61` -- cheap enough for the `O(n_keys^2)` CPU
reference to run in a test, but all three are below the new 16,384
threshold. With `Auto` now the default, that test would silently stop
exercising `attn_flash_decode_mma_wpo{2,4}` at all: `Auto` would pick
`attn_flash_decode_warp` for every depth in the loop, same as if the
tensor-core kernel had never been built. Fixed by wrapping the existing
three-depth loop in an outer loop over the lever (`Auto`, forced `WPO=2`,
forced `WPO=4`) rather than raising `n_keys` into the tensor-core kernel's
own win region, which would have made the CPU reference `O(16,384^2)` --
tens of billions of multiply-adds per head, not a test. All nine tests
pass, `GATE` (1e-5) unchanged:

| lever | n_keys | max_abs | cosine |
|---|---:|---:|---:|
| auto | 4,096 / 4,097 / 61 | 1.118e-7 / 1.043e-7 / 1.043e-7 | 1.000000000 |
| `WPO=2` | 4,096 / 4,097 / 61 | 6.660e-6 / 7.276e-6 / 1.043e-7 | 1.000000000 |
| `WPO=4` | 4,096 / 4,097 / 61 | 6.660e-6 / 7.276e-6 / 1.043e-7 | 1.000000000 |

`the_forward_pass_reproduces_llama_cpps_logits_and_its_argmax` still
passes, identical argmax and logit (token 25358, 19.998243) to every prior
recording in this file. That capture is 19 tokens deep, below the
threshold, so it confirms the `Auto` default did not disturb the
already-passing warp path rather than exercising the new dispatch branch
-- the forced-lever rows above are what cover the tensor-core kernel
itself under this gate.

### End-to-end performance, `bench_decode`, `LLMXABE_DECODE_CHUNK=8192`, 48 steps after 4 warmup, GPU 1, 3 interleaved rounds, `Auto` default against `attn_flash_decode_warp` forced

| depth | warp (ms/step, 3 rounds) | `Auto` (ms/step, 3 rounds) | ratio | warp tok/s | `Auto` tok/s |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 12.26 / 12.41 / 12.30 | 12.05 / 12.12 / 12.18 | 1.02x | 81.14 | 82.54 |
| 65,536 | 14.79 / 14.77 / 14.82 | 14.08 / 14.07 / 14.08 | 1.05x | 67.60 | 71.04 |
| 131,072 | 19.77 / 20.45\* / 19.79 | 17.64 / 17.70 / 17.64 | 1.13x | 50.01 | 56.62 |

\*131,072's second warp round is again the noisiest of the three (`sd`
0.53 ms against 0.37-0.39 ms elsewhere in this table); `Auto`'s three
rounds agree to within 0.06 ms of each other at every depth, for whatever
that says about the fixed kernel's own run-to-run variance against the
warp kernel's.

Every depth that used to lose end to end now wins: the previous section's
`WPO=2` (pre-fix) measured 0.958x at 32,768 and 0.981x at 65,536 -- real
losses, which is why it shipped disabled by default. Post-fix `Auto`
measures 1.02x and 1.05x at the same two depths, and 131,072's win grew
from 1.02x to 1.13x. Kernel-level and end-to-end both grew in the same
direction, as they should for a fix that touched only the kernel's own
register behavior and not the split/combine structure around it.

### Bandwidth, updated

The previous section measured `WPO=2` at 243 GB/s (36% of the card) at
131,072 keys, against llama.cpp's own measured ~589 GB/s (88%) on the
identical KV-read arithmetic. `bench_attention`'s own issued-GB/s column
for the fixed kernel at the same depth: 288.6 / 289.6 / 288.6 GB/s across
the three rounds above, 288.9 GB/s average -- **43% of the card**, up from
36%. The gap to llama.cpp's ceiling narrowed from 346 GB/s short to 300 GB/s
short -- real, and still most of the distance. Nothing named in this
section moves that further; `DMMA_KT` widening (the lever sized to close
it) is the one explicitly not attempted above, for the register-headroom
reason given there.

### Against the mission target, and against this session's own baseline

The mission brief's parity target at 131,072 keys is `attn_flash_decode_warp`'s
1.197 ms cut roughly in half, near 0.6 ms. `WPO=2` now measures 0.929 ms
kernel-level (`nvcc`'s own three-round average above) -- better than the
previous section's 1.105 ms, but still about 1.5x the target rather than
at it.

End to end, this session's own direct A/B (the table above, same binary,
same prompt, interleaved rounds) is the honest comparison: `Auto` is
1.02x-1.13x faster than `attn_flash_decode_warp` at 32,768-131,072 keys,
growing with depth, nothing regressed. Translating to the llama.cpp
head-to-head this file has tracked from the start of this workstream needs
a caveat this file has not needed before: the `attn_flash_decode_warp`
baseline measured *today* (81.14 / 67.60 / 50.01 tok/s at 32,768 / 65,536 /
131,072) is not the same number this file opened the workstream with
(82.0 / 62 / 51.2) -- other sections landed in between (MoE's widened `M`,
GDN's occupancy fix, the narrow decode-shape defects) that move the
whole-step baseline independently of anything in this section. Against
today's own warp baseline, `Auto` reaches 82.54 / 71.04 / 56.62 tok/s;
against llama.cpp's 88.5 / 79.7 / 66.9, that is 0.933x / 0.891x / 0.846x --
up from the workstream's opening 0.93x / 0.78x / 0.77x, most of the move
concentrated at the two deeper depths this section's dispatch threshold
actually reaches. 32,768 barely moves (0.93x either way) because `Auto`'s
own win there is small (1.02x) against a baseline that was already close
to parity at that depth before this section existed.

### Disposition

Shipped as the default. `AttentionKernels::new` still constructs with
`decode_mma` at `0`, but `0` now means `Auto` rather than disabled --
every caller that does not explicitly call `disable_decode_mma` or
`set_decode_mma_wpo` gets depth-aware dispatch with no code change on
their part. Nothing regresses: every depth measured, from 2,048 to
131,072, either matches `attn_flash_decode_warp` (below 16,384, where
`Auto` selects it) or beats it (at or above, where `Auto` switches),
because the threshold was chosen from where the kernel-level measurement
actually crosses over rather than a round number picked in advance.

### What a follow-up needs

1. **`DMMA_KT` widening is still untried**, and now has a clearer
   precondition than "try it once registers are free": `WPO=2` needs
   registers freed *beyond* what this section's fix already recovered
   before a wider key trip has anywhere to go without spilling. Where
   those additional registers would come from -- `o[MAXT][4]`, `qa0`/`qa1`
   and the K/V staging arrays are the only large holders left -- is
   unexamined.
2. **The extra 9-20 points of ratio improvement beyond what the spill
   removal alone would predict is unexplained.** `cuobjdump -sass` with a
   working per-function instruction count (named as missing in the MoE
   section above too) would settle whether the same-tile restructuring
   changed instruction scheduling beyond the register count, or whether
   something else in the three-round measurements is not fully isolated.
3. **43% of the card's bandwidth against llama.cpp's 88% on the identical
   traffic is still a 2x gap**, and this section's own bandwidth
   accounting gives no further lever to close it beyond `DMMA_KT` --
   `ncu` would be the direct way to see where the remaining issue slots
   go, and remains unavailable on this host.
## MoE's small-bucket GEMV, take two: a separate kernel, bit-identical SASS on everything it did not touch, and N=8 stops regressing (2026-08-18)

The previous section's own "What a follow-up needs" named this exactly.
Built it: `moe_expert_ffn_narrow`/`moe_expert_down_narrow` are new,
standalone `__global__` entry points, not branches inside `moe_expert_ffn`/
`moe_expert_down`. Their `bm == 1` case calls `tile_gemm_pair_direct1`/
`tile_gemm_single_direct1` (new functions, no staging, same reasoning as
before); `bm > 1` falls through to the *unmodified* `tile_gemm_pair`/
`tile_gemm_single` templates -- copy-pasted dispatch, not a shared branch.
`tile_gemm_pair`, `tile_gemm_single` and `MOE_TILE_DISPATCH` are not edited
by a single character. The host picks the narrow kernel by `1 < N <=
MOE_NARROW_DECODE_MAX` (4, below `MMA_MIN_TOKENS`'s 8 so there is no overlap
with the integer path) in `grouped_forward_partial`, a launch-time decision
from `g.max_tokens` -- host-known already, the same value that picks
`moe_expert_ffn_gemv` at `N == 1`.

This landed after `890377d` ("Widen M on the routed-expert MoE MMA
kernels"), which touches `MOE_MMA_M`/`MOE_MMA_BLOCKS_PER_SM` -- a different
part of the same file, no textual overlap with this change, and the numbers
below are against that commit, not the one two sections up.

### Why N=8 regressed even fully gated off: `moe_expert_ffn_mma`, not `moe_expert_ffn`

The previous section's `ptxas -v` check looked at `moe_expert_ffn`/
`moe_expert_down` and found nothing -- correctly, but at the wrong kernel.
`MMA_MIN_TOKENS` is 8, and this model's gate/up are Q6_K, so N=8 decodes
through `moe_expert_ffn_mma`/`moe_expert_down_mma`, never through the tiled
kernels either version of this change edited. Confirmed rather than assumed:
`nsys --cuda-graph-trace=node` over an isolated N=8 replay (2,048-token
context, GPU 2) shows `moe_expert_ffn_mma` (1,324 calls) and
`moe_expert_down_mma` (1,358 calls) and **zero** instances of
`moe_expert_ffn`/`moe_expert_down` in the same window.

`ptxas -v` on `moe_expert_ffn_mma` -- pulled from the same two `MOE_SRC`
extractions the previous section built, not re-derived -- is the answer the
previous section was looking for and did not know where to look: **80
registers on the pre-`bm==1` tree, 126 after.** Adding text anywhere in
`tile_gemm_pair`/`tile_gemm_single` (a function `moe_expert_ffn_mma` never
calls, has no template relationship to, and is not adjacent to in the file)
moved `ptxas`'s register allocation for a wholly unrelated `__global__`
function compiled from the same `nvrtc` module. Neither register pressure
nor shared memory explained the regression in the previous section's own
check because that check was pointed at kernels the regressed width does
not run.

### The fix, verified the way the previous section asked for

`cuobjdump -sass` on both trees (`890377d` alone, and `890377d` plus this
change), same extraction and `nvcc -arch=sm_75 --ptxas-options=-v` pipeline
the project already uses in place of `ncu`. Every kernel that existed before
this change -- `moe_expert_ffn`, `moe_expert_down`, `moe_expert_ffn_mma`,
`moe_expert_down_mma`, `moe_expert_ffn_gemv`, `moe_expert_down_gemv` --
diffs **byte-identical**, register counts and all (`moe_expert_ffn_mma`:
126 registers both trees, matching `890377d`'s own widened `MOE_MMA_M`
number, not the 80 this change's earlier attempt disturbed). The `ptxas -v`
log's only difference between the two trees is two new lines: `Compiling
entry function 'moe_expert_ffn_narrow'` and `'moe_expert_down_narrow'`.

`bench_decode_batch`, GPU 2, three interleaved rounds each, against
`890377d` (not the pre-widening baseline two sections up -- see the note
above):

| N | ctx | before (mean, 3 rounds) | after | change |
|---:|---:|---:|---:|---:|
| 2 | 2,048 | 96.0 | 102.7 | **+7.0%** |
| 3 | 2,048 | 106.0 | 112.5 | **+6.1%** |
| 3 | 32,768 | 85.6 | 89.6 | **+4.7%** |
| 4 | 2,048 | 121.7 | 128.7 | **+5.7%** |
| 8 | 2,048 | 153.0 | 152.7 | **noise** (-0.2%, three rounds each: 154.0/152.9/152.2 vs 153.3/152.6/152.1) |

N=8's own baseline moved with `890377d` (170.7 -> ~153, that commit's own
trade for its prefill target) -- this change does not touch it either
direction, which the bit-identical SASS above already predicts and the
measurement confirms.

### Correctness

`tests/moe_differential.rs`: 6/6. `tests/batch_decode.rs`: 3/3, including
the bit-exact `identical_prompts_in_one_batch_produce_bit_identical_rows` --
and unlike the previous section's own coverage note, this one is not a
formality: `BATCH` in that file is 3, squarely inside `1 < N <=
MOE_NARROW_DECODE_MAX`, so all three differentials exercise
`moe_expert_ffn_narrow`/`moe_expert_down_narrow` directly, not just the
kernels this change left alone. `tests/moe_differential.rs`'s own
`NUM_TOKENS` is 37, above the narrow threshold, so it does not reach the new
kernels -- its 6/6 is evidence the untouched paths still agree with the CPU
reference, not evidence for the new ones. N=2 and N=4 share the identical
`bm == 1` code path as N=3 and are covered by the throughput A/B above, not
by a dedicated differential at those widths.

The golden test (`the_forward_pass_reproduces_llama_cpps_logits_and_its_
argmax`) passes at the same logit (19.998243 against llama.cpp's 19.902241)
and the same rank-4 noise-floor swap this file's precedent already accepts
-- expected, since it runs single-stream (`N == 1`) and never reaches the
narrow kernels either.

### What is left

`MOE_NARROW_DECODE_MAX` is 4 because that is where this session's own
measurements stop, not because 5-7 are known to behave like 8. Widening it
would need N=5-7 measured the same way before trusting them. The shared
expert (`moe_shared_ffn`/`moe_shared_down`) deliberately keeps the original
three-way dispatch -- its `bm` is the literal token count with no sparse
routing, so `bm == 1` there only happens at `N == 1`, already served by
`moe_expert_ffn_gemv` before `grouped_forward_partial` reaches it.

## What `moe_expert_ffn_narrow`/`moe_expert_down_narrow` actually achieve, and why they fall short of the GEMV kernels they were modeled on (2026-08-18)

The previous section's `bm == 1` fix closes dispatch -- the right kernel
family runs -- but does not by itself close *bandwidth*. Measured on the
isolated N=3 replay (`bench_decode_batch 32768 32`, `LLMXABE_SKIP_SINGLE_
STREAM=1`, `nsys --cuda-graph-trace=node`, GPU 2): `moe_expert_ffn_narrow`
averages 155,738 ns/call (1,440 calls, 40/step) and `moe_expert_down_narrow`
124,529 ns/call. Converting to GB/s with `D(3) = 23.26` distinct experts/
layer x 40 layers x each expert's own weight bytes (gate+up Q6_K 1.7204 MB,
down Q8_0 1.1141 MB, `docs/OPTIMIZATION.md` §2.1/2.2's own numbers):

| kernel | GB/s | % of 672 GB/s roofline | the GEMV kernel it was modeled on |
|---|---:|---:|---:|
| `moe_expert_ffn_narrow` | 257.0 | **38.2%** | `moe_expert_ffn_gemv`, 47% |
| `moe_expert_down_narrow` | 208.2 | **31.0%** | `moe_expert_down_gemv`, 63% |

Both real gaps toward the GEMV kernels' own numbers, and down's is the
larger one in relative terms -- the opposite of N=1, where down (63%) beats
ffn (47%). `D(N)` is a uniform-routing upper bound and real routing measures
~16% below it at moderate batch (§2.5), so these percentages are more likely
a few points optimistic than pessimistic; the ranking between the two
kernels and against their GEMV counterparts is what this section trusts, not
the third significant figure.

### Tried: giving the direct1 helpers the GEMV kernels' own unroll pragmas, and it made both slower

`moe_expert_ffn_gemv`/`moe_expert_down_gemv` each carry a tuned
`#pragma unroll` (2 and `MOE_DOWN_UNROLL` = 4) that `tile_gemm_pair_direct1`/
`tile_gemm_single_direct1` do not -- same dequant/accumulate body, same
trip count (`k_len` is `hidden`/`intermediate` either way), so the tuning
looked like it should transfer directly. Added both pragmas, rebuilt,
`tests/moe_differential.rs` and all three `tests/batch_decode.rs`
differentials still passed (unrolling does not change accumulation order),
then measured: **`moe_expert_ffn_narrow` went from 155,738 to 187,768 ns/call
-- slower, not faster** -- and `moe_expert_down_narrow` was flat (124,529 ->
125,948 ns/call).

`ptxas -v` explains it, and it is the same class of defect
`MOE_TILE_DISPATCH`'s own comment already names for a fourth tiled
specialization: a register cliff. `moe_expert_ffn_narrow` without the
pragma is 80 registers (2^16 / (80 x 256 threads) = 3 blocks/SM); with it,
96 registers (2 blocks/SM). `moe_expert_down_narrow`: 74 -> 76 registers,
a smaller move, matching its smaller (flat, not regressed) timing change.
The mechanism is different from the previous section's cross-kernel
register-cliff surprise, but the mistake is the same shape: `moe_expert_
ffn_gemv`'s 47-register, unroll-2-tuned budget is a *standalone* kernel's
number. `moe_expert_ffn_narrow` is not standalone -- it carries `bm > 1`'s
full tiled fallback (`tile_gemm_pair<2/8/16>`) in the same function, so its
register floor already sits at `moe_expert_ffn`'s own 80 before the `bm ==
1` branch adds anything, and the GEMV kernel's unroll depth was tuned
against a register budget this kernel does not have room for. Reverted;
`tile_gemm_pair_direct1`/`tile_gemm_single_direct1` are unchanged from the
previous section.

### The structural gap this explains, and the part it does not

Register counts, `2^16 / (registers x 256)` blocks/SM, `ptxas -v` on both
kernel families:

| kernel | registers | blocks/SM |
|---|---:|---:|
| `moe_expert_ffn_gemv` (standalone) | 47 | 5 |
| `moe_expert_ffn_narrow` (carries `bm > 1`'s fallback) | 80 | 3 |
| `moe_expert_down_gemv` (standalone) | 64 | 4 |
| `moe_expert_down_narrow` (carries `bm > 1`'s fallback) | 74 | 3 |

Ffn's occupancy drops 40% (5 -> 3 blocks/SM) against a bandwidth drop from
47% to 38.2% (19% relative) -- occupancy explains a real fraction of the
gap, not all of it. Down's occupancy drops 25% (4 -> 3) against a bandwidth
drop from 63% to 31.0% (51% relative) -- occupancy alone does not explain a
gap that size, and this section does not have a confirmed second mechanism
for the remainder. One unquantified candidate, named rather than measured:
`moe_expert_down_narrow`'s write epilogue is `moe_expert_down`'s own
`#pragma unroll` 16-wide loop over `m < bm`, shared code the `bm == 1`
branch does not get its own leaner version of -- fifteen of those sixteen
unrolled comparisons are dead whenever `bm == 1`, and `moe_expert_down_gemv`
has none of them (one unconditional write). Not measured in isolation this
session.

### What closing this needs, specified and not attempted

The register floor is structural, not a tuning knob: as long as `bm == 1`'s
fast path and `bm > 1`'s tiled fallback share one `__global__` function, the
fast path can never see the standalone GEMV kernel's register budget. The
fix this points to is the same shape as the previous section's own --
another separate compiled kernel, not a branch -- but split along a
different axis: a truly standalone `bm == 1` kernel (`moe_expert_ffn_gemv`'s
own body, generalized from "assume slot 0, unconditionally" to "check `bm`
per dispatch bucket, skip if not 1") running over the *whole* grid, paired
with a second kernel that is `moe_expert_ffn`'s existing tiled body with one
line changed (`if (bm == 0) continue;` becoming `if (bm <= 1) continue;`) so
the two together partition every bucket exactly once. That is two kernel
launches per projection instead of one, each with the leaner register
budget its own shape earns, at the cost of both re-walking `sorted_token_ids`
independently to compute `bm` -- cheap relative to the dequant/GEMM work it
gates, per this file's own `bm == 0` skip precedent, but unmeasured here.
Building and verifying two more kernel pairs against every accuracy gate is
more than this session's remaining budget allows; specified here rather than
attempted, the same call this file made for MoE's `M` 32 narrow-batch
variant two sections up.

## The two-kernel split: built, register budget landed as predicted, and rejected on a cost the register math did not carry (2026-08-18)

The previous section's "what closing this needs" was built: `moe_expert_
ffn_bm1`/`moe_expert_down_bm1`, standalone `__global__` kernels generalizing
`moe_expert_ffn_gemv`/`moe_expert_down_gemv`'s "assume slot 0" to "check
`bm` per dispatch bucket via `live_tile_rows`, process only if it is exactly
1" -- the `moe_expert_ffn_narrow`/`moe_expert_down_narrow` pair changed
alongside them to skip `bm <= 1` instead of `bm == 0`, so the two kernels of
each pair partition every bucket exactly once. `moe_expert_ffn`/
`moe_expert_down`/`_mma`/`_mma`/both original `_gemv` kernels stayed
byte-identical (`cuobjdump -sass`, six kernels, all matching `890377d`
exactly) -- the isolation held.

### The register budget landed almost exactly where the previous section's math said it would

`ptxas -v`: `moe_expert_ffn_bm1` **58 registers** (4 blocks/SM) against
`moe_expert_ffn_gemv`'s 47 (5) and the un-split narrow kernel's 80 (3) --
most of the gap closed. `moe_expert_down_bm1` **56 registers** (4 blocks/SM),
*below* `moe_expert_down_gemv`'s own 64, matching its 4 blocks/SM exactly.
`moe_expert_ffn_narrow`/`moe_expert_down_narrow`, relieved of the `bm == 1`
branch, dropped back to `moe_expert_ffn`/`moe_expert_down`'s own 80/77
registers -- confirming the previous section's read that the combined
kernel's register floor was set by the tiled fallback, not by anything
`bm == 1` itself needed.

### The throughput did not follow, and reading the actual grid shape says why

`bench_decode_batch`, N=3, 32,768-token context, interleaved against the
un-split build, three rounds: **90.6 tok/s (pre-split) vs 85.4-85.9
(post-split), a real -5.2 to -5.7% regression**, not the >100 tok/s the
register math alone predicted. Per-kernel (`nsys`, isolated N=3 replay):

| | pre-split (combined) | post-split (`bm1` + rest) | change |
|---|---:|---:|---:|
| ffn total/step | 6.23 ms | 6.19 ms (4.25 `bm1` + 1.94 rest) | ~flat |
| down total/step | 4.98 ms | 6.92 ms (3.67 `bm1` + 3.25 rest) | **+39%** |

Ffn is a wash; down is the whole regression, and it is not a fluke of the
particular kernels involved -- `MoeGeometry::expert_block_capacity` explains
it structurally. It is `sorted_capacity() / block_size`, and `sorted_
capacity` is bounded by `numel` (the batch's actual token-expert pair
count: `N * top_k`), not by `num_experts` -- at N=3 that is 24 buckets, not
a sparse 256-wide capacity most of which exits on `if (e < 0) return`
before touching anything else. Nearly every one of those 24 buckets is
genuinely live. Splitting into two kernels means **both** now walk
`sorted_token_ids`, populate `rows[]` and cross two `__syncthreads()` per
`block_size`-wide bucket to compute `bm`, for every bucket, and only one of
the two ever finds a use for the answer -- real, doubled bookkeeping work,
not a redundant early-exit. `moe_expert_down`'s grid is `ceil(hidden /
MOE_ROWS)` = 256 blocks wide per bucket (down tiles `hidden`, 2,048); `moe_
expert_ffn`'s is `ceil(intermediate / MOE_ROWS)` = 64 (ffn tiles
`intermediate`, 512) -- four times fewer blocks paying the doubled cost,
which is why ffn stayed flat while down's total time grew by more than a
third.

### Rejected; the un-split kernel from the previous two sections is what ships

Reverted in full: `crates/xabe-cuda/src/kernels/moe.rs` is back to the
state the two previous sections landed (`moe_expert_ffn_narrow`/`moe_
expert_down_narrow` handling `bm == 1` inline via `tile_gemm_pair_direct1`/
`tile_gemm_single_direct1`, no `_bm1` kernels). Confirmed via `git diff`
against that commit: empty. The register-budget analysis was correct as
far as it went -- occupancy really did improve to the GEMV kernels' own
class -- but it was not the only cost the split kernel pays, and the second
cost scales with exactly the axis (`grid.x`, i.e. `hidden` vs
`intermediate`) that makes down the more attractive target on paper and the
worse outcome in practice. A hybrid -- split ffn only, leave down as the
combined kernel -- was not built: the per-kernel table above already gives
its expected total (6.19 + 4.98 = 11.17 ms) to two decimal places, indistin-
guishable from the un-split baseline's 6.23 + 4.98 = 11.21 ms, so it is not
worth the second kernel pair's own maintenance and register-count
verification burden for a sub-0.4% step-time change.

### What is left

The narrow-width GEMV bandwidth gap this section and the previous one both
measured (38.2%/31.0% against the GEMV kernels' 47%/63%) is now bounded from
two directions without being closed from either: giving the fast path its
own function moves registers into the GEMV kernels' class but adds
bucket-bookkeeping cost proportional to `grid.x`, and leaving it inline
keeps the bookkeeping cost but inherits the tiled fallback's wider register
floor. Closing it further needs a shape that pays neither -- reducing the
`bm`-determination cost itself (a per-bucket precomputed table, built once
by `moe_align_block_size` rather than re-derived by every kernel that reads
`sorted_token_ids`, is the obvious next place to look) rather than another
way of routing around it. Not attempted this session.

## Staging `Q` to shared to free `DMMA_KT` headroom: real registers freed, and a real loss anyway (2026-08-18)

The previous section's close named `WPO=2`'s 3-register margin (252 of 255)
as what blocks `DMMA_KT` widening. The lead's proposed fix: `Q`'s `q_hi`/
`q_lo` fragment (`qa0[DMMA_QSTEPS]`/`qa1[DMMA_QSTEPS]`, 64 registers per
lane) is identical across every warp in a block -- `g`/`tg` depend only on
`lane`, not `warp`, and the pointer into `q` depends only on `kvh`/`g`, both
warp-independent -- so `WPO` warps were each separately computing and
holding the same 64 registers' worth of data. Staging it to shared once,
read by every warp's `Q K^T` phase instead, trades that redundant register
residency for shared reads the prefill kernel's own `q_sh` already proves
the pattern for on this card.

### The shared-memory arithmetic the lead asked to be verified first, verified, and it does not hold at `WPO=2`

`Q`'s staged size is `2 * 8 * (head_dim / 2)` words (`q_hi_sh`/`q_lo_sh`,
8 real rows, `head_dim / 2` packed columns each) = 8,192 B at `head_dim`
256, independent of `WPO`. Added to `dmma_shared_bytes`:

| WPO | shared before | shared after (+8,192 B) | blocks/SM before | blocks/SM after |
|---:|---:|---:|---:|---:|
| 2 | 21,344 B (20.8 KiB) | 29,536 B (28.8 KiB) | 3 | **2** |
| 4 | 38,496 B (37.6 KiB) | 46,688 B (45.6 KiB) | 1 | 1 |

The lead's own sizing note ("stays within 3 blocks/SM... verify the
arithmetic first") does not hold at `WPO=2`: `65,536 / 29,536 = 2.22`,
floor 2, not 3. `WPO=2`'s entire advantage over `WPO=4` in every earlier
section is the extra resident block hiding K/V staging latency that
`WPO=4`'s single block cannot; this change spends part of that same
margin to buy the register space back. `WPO=4` is unaffected (still
1 block/SM, shared-limited both before and after) but is also the width
that has never won at any depth measured, so it was not built or
benchmarked separately -- spending session time on the losing width's
numbers was judged not worth it under this experiment's own bound.

### Built, correct, and registers freed by far more than the 15-register bar

Implemented as a block-wide cooperative write: `q_hi_sh`/`q_lo_sh` filled
by a `for (int idx = tid; idx < 8 * hd2; idx += nthr)` loop before the key
loop starts (the loop's own first statement is already `__syncthreads()`,
which covers the write-then-read ordering with no new barrier needed), then
each trip's `Q K^T` phase reads `q_hi_sh[g * hd2 + c]`/`q_lo_sh[g * hd2 +
c]` in place of the old `qa0[s]`/`qa1[s]` register reads. `nvcc -arch=sm_75
-cubin -Xptxas -v` on the extracted kernel:

| variant | registers before | registers after | spill |
|---|---:|---:|---:|
| `WPO=2` | 252 | **179** | 0 -> 0 |
| `WPO=4` | 235 | **126** | 0 -> 0 |

73 and 109 registers freed respectively -- far past the lead's own
`>= 15` bar for "real headroom," at 0 spill both before and after. On
registers alone this looks like exactly the win the previous section's
close was waiting for. `device_decode_matches_the_reference_over_a_deep_window`,
forced to both widths, reproduces the pre-change numbers exactly
(`n_keys = 4,096`: 6.660e-6; `4,097`: 7.276e-6; `61`: 1.043e-7; all
cosine 1.000000000) -- moving `Q` to shared changed nothing about what
the kernel computes.

### Measured anyway, and it is not a wash

`bench_attention`, `LLMXABE_ATTN_CHUNK=1`, `LLMXABE_DECODE_MMA_WPO=2`, GPU 1,
three interleaved rounds against a `git worktree` build of the unmodified
`WPO=2` kernel (the same isolation hazard-avoidance the MoE workstream
used earlier in this file, for the same reason -- this checkout has
sibling edits landing and being reverted in the same file while this
session runs):

| key_offset | register-`Q` (ms, 3 rounds) | shared-`Q` (ms, 3 rounds) | shared-`Q` vs register-`Q` |
|---:|---:|---:|---:|
| 32,768 | 0.305 / 0.306 / 0.306 | 0.417 / 0.417 / 0.418 | 1.37x slower |
| 65,536 | 0.508 / 0.508 / 0.515 | 0.710 / 0.709 / 0.710 | 1.39x slower |
| 98,304 | 0.726 / 0.721 / 0.725 | 0.999 / 0.995 / 0.997 | 1.38x slower |
| 131,072 | 0.923 / 0.927 / 0.932 | 1.297 / 1.299 / 1.300 | 1.40x slower |

Consistent, decisive, and not close: 37-40% slower at every depth
measured, agreeing to within a percent of itself across all three rounds
on both sides. Freeing 73 registers with 0 spill did not translate into a
win -- the two candidate mechanisms this session did not isolate between
(no `ncu`, the same limitation named throughout this file) are the
occupancy drop measured above (3 -> 2 resident blocks/SM losing exactly
the latency-hiding margin `WPO=2` depended on) and the new per-trip cost
the register version never paid: 32 unrolled `q_hi_sh`/`q_lo_sh` reads
apiece, 64 shared loads per warp per trip, on every trip through the
whole key loop rather than once. The near-uniform ~37-40% slowdown across
every depth (not concentrated at the deepest, most-trip-heavy end) is
weak evidence for the per-trip-read explanation over the occupancy one,
but this session did not confirm it against `cuobjdump -sass` or `ncu`.

### Disposition

Reverted (`git checkout -- crates/xabe-cuda/src/kernels/attention.rs`,
zero diff against the parent commit). Per this experiment's own bound: a
loss here means `DMMA_KT=64` was not attempted -- there was nothing to
widen a key trip on top of, since the one lever this section had for
freeing `WPO=2`'s register margin made the kernel slower than the margin
was worth recovering. `DECODE_MMA_DEPTH_THRESHOLD` (16,384) is unchanged;
the reverted kernel is bit-for-bit the one it was measured against.

### What this closes

`DMMA_KT` widening has now been tried from both register-freeing angles
available without a block-shape change: the "Widening the softmax-rescale
tile" section's `MMA_KEY_TRIPS` did it on the *prefill* kernel and spilled
immediately from 252 registers; this section freed decode's own registers
first and still lost, on shared-memory occupancy and per-trip read cost
rather than a spill. Between the two, every register-side lever this file
has for the wider key trip is spent. Closing the remaining bandwidth gap
(43% against llama.cpp's 88%, per the previous section) needs a
structural change neither section's own tools could evaluate without
`ncu` -- both close on the same request.
## A precomputed `bm`, a real but modest win, and a lesson about which cost was actually being paid (2026-08-18)

Both of the previous two sections' rejects shared a root cause: the
`bm`-determination cost. Inline, it widened the register floor by dragging
in the tiled fallback's frame; split, it doubled the per-bucket bookkeeping
because `expert_block_capacity()` at N<=4 is small and nearly all live, so a
second kernel walking `sorted_token_ids` a second time was real work, not a
cheap early exit. This section attacks the shared root directly: precompute
`bm` once, during dispatch-table construction, so every consumer reads it
instead of deriving it.

### The table

An expert's tokens land at a *contiguous* prefix of its span --
`moe_align_block_size`'s phase 3 scatter always advances `written` from
zero, in ascending flat order, and never touches a slot past `counts[e]`.
That means a bucket's live-row count is a closed form of two values the
kernel already has at phase 4 (`counts[e]` and the bucket's own offset from
`first`), not something that needs a second pass over `sorted_token_ids` to
discover:

```c
int c = counts[e];
for (int b = first + tid; b < last && b < expert_capacity; b += blockDim.x) {
    expert_ids[b] = e;
    int live = c - (b - first) * block_size;
    if (live < 0) live = 0;
    if (live > block_size) live = block_size;
    bucket_live[b] = live;
}
```

`bucket_live` is a new array, `expert_block_capacity` ints, written
alongside `expert_ids` in the loop that already claims each bucket for its
expert -- one more global store per bucket the kernel was already visiting,
not a new pass. Additive throughout: `sorted_token_ids` and `expert_ids`
keep their exact shape and meaning, `moe_align_block_size` gained one
trailing parameter, and `MoeBuffers` gained one field
(`buffers.bucket_live()`) alongside `expert_ids()`. No other array was
reshaped.

`moe_expert_ffn_narrow`/`moe_expert_down_narrow` are the only converted
consumers, per this session's brief (narrow kernels first, since they are
this workstream's own code). Each now loads `bucket_live[blk]` once, before
its `m0` loop, and gets every sub-tile's `bm` from a clamp:

```c
int bucket_bm = bucket_live[blk];
for (int m0 = 0; m0 < block_size; m0 += MOE_TM) {
    int bm = bucket_bm - m0;
    if (bm < 0) bm = 0;
    if (bm > MOE_TM) bm = MOE_TM;
    if (bm == 0) continue;
    __syncthreads();
    ... populate rows[]/slot_flat[] only now, and only because bm > 0 ...
```

The `bm == 0` check moved *ahead* of both `__syncthreads()` calls and the
`rows[]`/`slot_flat[]` populate. At `N <= MOE_NARROW_DECODE_MAX` (4),
`counts[e] <= 4 < MOE_TM` (16) always, so every bucket's *second* sub-tile
(`m0 == 16`) is unconditionally empty -- previously that discovery cost a
full populate-and-scan cycle every single decode step, on every bucket, for
nothing. It now costs one register compare against a value already in a
register.

`moe_expert_ffn`/`moe_expert_down` (unconverted, per "narrow kernels
first") still derive `bm` from `live_tile_rows(rows)` exactly as before --
they do not read `bucket_live` at all, so nothing about their behavior or
performance was expected to change, and the SASS check below confirms it
did not.

### Blast radius: three functions touched, seventeen untouched

`cuobjdump -sass`, `nvcc -arch=sm_75 -cubin` against the shipped `MOE_SRC`
string extracted straight from source (not a stale scratch copy), diffed
per function against the landed tree (`cd35d38`) this section built on:

| function | SASS |
|---|---|
| `moe_align_block_size` | differs (expected: new trailing store) |
| `moe_expert_ffn_narrow` | differs (expected: converted) |
| `moe_expert_down_narrow` | differs (expected: converted) |
| every other of the 17 remaining kernels, including all four of `moe_expert_ffn_mma`/`_mma_narrow`/`moe_expert_down_mma`/`_mma_narrow` | byte-identical |

Exactly the three touched functions differ; every kernel this session did
not mean to touch, including the dual-width MMA pair the previous
workstream landed, is untouched all the way to the instruction encoding.

Register cost of the two converted kernels, `ptxas -v` on the same
standalone compile:

| kernel | before | after |
|---|---:|---:|
| `moe_expert_ffn_narrow` | 80 | 79 |
| `moe_expert_down_narrow` | 74 | 80 |
| `moe_align_block_size` | 25 | 25 |

`moe_expert_down_narrow` picked up 6 registers -- the `bucket_bm` local and
the loop's extra bookkeeping outlive more of the kernel body than they cost
in the old derivation, which freed its registers every iteration. Zero
spill either side, and both land in the same occupancy bracket at
`GEMM_THREADS` (256): `65536 / (256 * 80) = 3.2`, still 3 blocks/SM, same as
before at 74. `moe_align_block_size` does not move at all -- the new store
is folded into a loop it already ran.

### Accuracy gates

`cargo test --release -p xabe-engine --test moe_differential`: 7/7,
including `device_dispatch_tables_match_moe_align_block_size_including_
padding`, which is the one that would have caught a `bucket_live` formula
that disagreed with `live_tile_rows`'s runtime derivation on any input the
test's routing distributions cover.
`cargo test --release -p xabe-engine --test batch_decode`: 3/3, including
`identical_prompts_in_one_batch_produce_bit_identical_rows` at the
narrow-kernel widths. The golden test
(`the_forward_pass_reproduces_llama_cpps_logits_and_its_argmax`) still
passes at the same logit and rank-4 noise-floor swap this file's precedent
already accepts.

### Throughput: real, reproducible, and far short of what the occupancy math predicted

Interleaved A/B, CUDA events via `bench_decode_batch`, 32 decode steps after
4 warmup, GPU 2, `LLMXABE_SKIP_SINGLE_STREAM=1`, before/after binaries built
from a `git stash`-free checkout of the same worktree so both share every
other line of code:

| N | context | before (tok/s) | after (tok/s) | delta |
|---:|---:|---:|---:|---:|
| 2 | 2,048 | 107.6-109.5 | 110.0-110.6 | +0.9% to +2.7% |
| 3 | 2,048 | 118.3-120.1 | 121.4-121.8 | +1.1% to +2.9% |
| 4 | 2,048 | 135.3-136.4 | 138.9-139.3 | +1.8% to +2.9% |
| 8 | 2,048 | 169.4-170.4 | 169.3-169.7 | flat (unconverted path) |
| 2 | 32,768 | 87.1 | 89.5-89.7 | +2.8% to +3.0% |
| 3 | 32,768 | 95.0 | 97.6-97.7 | +2.7% to +2.8% |
| 4 | 32,768 | 104.3-104.6 | 107.9-108.0 | +3.4% to +3.5% |
| 8 | 32,768 | 127.3 | 128.0 | +0.5% (noise) |

Every narrow width (N 2-4) improves, consistently, across 2-3 interleaved
rounds at both contexts; N 8 is flat within run-to-run noise at both, which
is exactly what byte-identical SASS on its consuming kernels predicts --
`moe_align_block_size` runs for N 8 too and now does marginally more work
per bucket, but that cost is buried in a kernel that was never the
bottleneck. N=3 at 32,768 reaches 97.6-97.7 tok/s: a real step past the
previous section's baseline, but short of the 100 tok/s this lever was
predicted to clear.

The prediction that motivated this lever was sized on the wrong term. The
occupancy math from the split-kernel reject correctly showed that a
`bm > 1`-only kernel at ~56-58 registers would land 4 blocks/SM against the
combined kernel's 3 -- but that reject's *own* measurement had already shown
the down projection was flat, not slow, when split; the regression was
entirely the *doubled derivation cost*, not a missed-occupancy tax on the
arithmetic. Precomputing `bm` removes exactly that doubled-derivation cost
and no more: what it buys back is one skipped populate-and-sync cycle per
bucket's guaranteed-empty second sub-tile, which is real (2-3.5% is not
nothing) but is not the same lever as "give the `bm == 1` path the GEMV
kernels' occupancy," because the narrow kernels were never occupancy-bound
in the first place -- they are bandwidth-bound on the same dequant-and-
stream traffic the combined kernel pays, `bm == 1` or not. The 38.2%/31.0%
roofline gap this section's predecessor measured is still there because
nothing in this section touched how many bytes the kernel streams per
useful row.

### What a follow-up needs

Re-testing the split-kernel design against *this* baseline (not the
pre-precompute one) is the next cheap experiment the brief asked for, and
was not run this session: with `bm` a load rather than a derivation, a
`bm > 1`-only kernel's bookkeeping cost per bucket drops to the same clamp
this section's narrow kernels now pay, so the doubled-derivation objection
that sank the original split should indeed no longer apply. What this
section's own numbers argue against is the *size* of the win a working
split should be expected to produce: if precomputing `bm` for the combined
kernel bought 2-3.5%, a split kernel's own gain is bounded by the same
mechanism (fewer wasted syncthreads-and-populate cycles, better register
occupancy at the arithmetic ptxas already emits) rather than by the deeper
bandwidth win "down 31% -> 63%" implied. Reaching that would need fewer
bytes streamed per live row, not a cheaper way to find out how many rows
are live -- the unroll-pragma reject two sections back already ruled out
the cheap version of that (register cliff from carrying the tiled
fallback's frame), so a real fix likely needs the fallback path itself
restructured, not just the dispatch around it. Not attempted this session.

## The split-kernel design, re-tried on a precomputed `bm`: a real win this time, plus a `ptxas` allocator surprise the register math didn't see coming (2026-08-18)

The previous section's own close named the cheap next experiment: re-test
the lean `bm == 1` split against the `bucket_live` baseline, now that
`bm` is a table read shared by both halves instead of something either one
derives. Built as specified: `moe_expert_ffn_bm1`/`moe_expert_down_bm1`
generalize `moe_expert_ffn_gemv`/`moe_expert_down_gemv`'s "assume slot 0"
to "loop the block's sub-tiles, handle only the one where `bucket_live`
says `bm == 1`"; `moe_expert_ffn_narrow`/`moe_expert_down_narrow` drop their
`bm == 1` branch (`bm <= 1` now skips, same as `bm == 0` always did). The
two partition every bucket's every sub-tile exactly once between them.

### The bm1 kernels: no shared memory, no tiled fallback, one load and a clamp per sub-tile

```c
int bucket_bm = bucket_live[blk];
for (int m0 = 0; m0 < block_size; m0 += MOE_TM) {
    int bm = bucket_bm - m0;
    if (bm < 0) bm = 0;
    if (bm > MOE_TM) bm = MOE_TM;
    if (bm != 1) continue;
    int flat = sorted_token_ids[(long long)blk * block_size + m0];
    ...
}
```

`bm == 1` means `bucket_bm == m0 + 1`: the bucket's whole live prefix ends
one row into this sub-tile, so the live slot is always local index 0 --
`tile_gemm_pair_direct1`/`tile_gemm_single_direct1` (which only ever touch
index 0 of the pointer they are handed) take a one-element *register* array,
not `moe_expert_ffn_narrow`'s `MOE_TM`-wide shared one. No `xabe_shared`
tile, no `__syncthreads()`, no `tile_gemm_pair<TM>` fallback compiled in at
all.

### Blast radius: two new kernels, two modified, sixteen untouched

`cuobjdump -sass` against the `bucket_live` tree (`2704b6a`) this section
built on:

| function | SASS |
|---|---|
| `moe_expert_ffn_bm1`, `moe_expert_down_bm1` | new |
| `moe_expert_ffn_narrow`, `moe_expert_down_narrow` | differs (expected: `bm == 1` branch removed) |
| every other of the 16 remaining kernels, including `moe_align_block_size` and all four `moe_expert_*_mma*` | byte-identical |

### Registers: the target shape, and a surprise in the kernel that lost a branch

`ptxas -v`, same standalone-extraction method as every register table in
this file:

| kernel | before (bucket_live) | after (split) |
|---|---:|---:|
| `moe_expert_ffn_bm1` | -- | 49 |
| `moe_expert_down_bm1` | -- | 57 |
| `moe_expert_ffn_narrow` | 79 | 100 |
| `moe_expert_down_narrow` | 80 | 76 |

The two new kernels land almost exactly on the shape this workstream has
been chasing since the first split reject: 49 and 57 registers, `65536 /
(256 * 57) = 4.5` -> 4 blocks/SM for the down half, the "56-58 reg / 4
blocks/SM" target named three sections back. `moe_expert_down_narrow`
improved on its own, 80 -> 76 -- removing a branch gave the allocator one
fewer thing to reconcile and it used the room. `moe_expert_ffn_narrow` did
the opposite: 79 -> 100, a full occupancy tier lost (`65536 / (256 * 100) =
2.56` -> 2 blocks/SM, down from 3). Removing the same *kind* of branch from
a near-identical sibling kernel moved the allocator's decision in opposite
directions -- not something the register math from either split reject
predicted, and not explainable from the source diff alone; `ptxas`'s
allocator has already been shown once this project (the branch-inside-
`tile_gemm_pair` regression, `moe_expert_ffn_mma`'s 80 -> 126 registers)
to make decisions across a wider scope than the local control flow being
edited suggests.

`__global__ void __launch_bounds__(256, 3) moe_expert_ffn_narrow(...)`
recovers it: 100 -> 80 registers, zero spill, matching this file's own
established pattern (`moe_expert_ffn_mma`/`_narrow`'s dual-width split
already uses `__launch_bounds__` the same way). `moe_expert_down_narrow`
was left alone -- it was already at the same occupancy tier as before the
split, and the hint would only clamp a value already under the cap.

### Accuracy gates

`cargo test --release -p xabe-engine --test moe_differential`: 7/7 both
before and after the `__launch_bounds__` fix -- the differential test does
not touch registers, so this is confirming the *partition* is correct
(every dispatch table the test drives produces the same `inter`/`partial`
whether a bucket's live row landed in `expert_ffn_bm1` or
`expert_ffn_narrow`'s hands), not the occupancy claim, which the SASS
extraction above is.
`cargo test --release -p xabe-engine --test batch_decode`: 3/3 bit-exact,
including the CUDA-graph-captured path -- two more launches per projection
inside the capture, `identical_prompts_in_one_batch_produce_bit_identical_
rows` still passes at zero max diff.
Golden test unchanged.

### Throughput: a real win at every narrow width, once the register regression was caught

Interleaved A/B against the `bucket_live` baseline (`2704b6a`), same method
as every throughput table in this file. First measured *before* the
`__launch_bounds__` fix, then after, to see the fix's own effect:

| N | context | before split (tok/s) | split, no fix | split + `__launch_bounds__` |
|---:|---:|---:|---:|---:|
| 2 | 2,048 | 109.9-111.5 | 114.6-115.2 | 114.7-115.5 |
| 3 | 2,048 | 121.4-122.4 | 124.4-124.7 | 125.0-125.6 |
| 4 | 2,048 | 138.9-139.8 | 139.9-140.1 (flat) | 141.5-142.2 |
| 8 | 2,048 | 169.2-170.2 | 169.4-169.5 | 169.2-169.5 |
| 2 | 32,768 | 89.2-89.8 | 91.9 | 91.9 |
| 3 | 32,768 | 95.9-97.8 | 98.2-98.4 | 99.0-99.1 |
| 4 | 32,768 | 107.7-108.0 | 108.0 (flat) | 109.1-109.2 |
| 8 | 32,768 | 128.1 | 128.1 | 128.0 |

Without the fix, N=4 -- the width where `moe_expert_ffn_narrow` (now
running at 2 blocks/SM instead of 3) does the *most* work, since more of
its buckets have `counts[e]` at 2, 3 or 4 rather than exactly 1 -- comes out
flat: the bm1 kernel's win and the narrow kernel's regression roughly
cancel. With the fix, every narrow width improves, by margins that grow
with the register recovery's relative weight: +1.1% to +2.4% at N=4 (32,768
and 2,048 respectively), up through +2.5% to +5.1% at N=2, where almost no
bucket ever reaches the narrow kernel at all. N=8 stays flat at both
contexts, exactly what byte-identical SASS on its own path predicts.

This is a real, if uneven, win -- not the "down 31% -> 63%" the original
split reject's occupancy math implied three sections ago. The bm1 kernels
hit that occupancy target almost exactly (49/57 registers), but the
combined kernel this section actually measures against is `moe_expert_ffn_
narrow`/`moe_expert_down_narrow` doing the *rest* of the work, and that
kernel's own occupancy barely moved (it was already at 3 blocks/SM, and the
split plus the fix leaves it there, not higher). The deeper bandwidth gap
the original section named -- 38.2%/31.0% against the GEMV kernels' own
47%/63% -- is still not closed, because nothing here reduced bytes streamed
per live row; this section only ever removed wasted `bm`-discovery work,
first at the bucket level (the `bucket_live` table) and now again at the
kernel-selection level (routing `bm == 1` to a kernel with no reason to
carry the tiled fallback's registers).

### Disposition

Landed. Every measured width is flat or better, none regressed, and every
accuracy gate is green with the fix applied.

### Closing sweep: `N` 1-8, both contexts, on merged main (`145c675`)

The full width sweep this workstream's brief asked for at the end,
`single_stream` and `batch 1` included this time (both untouched by
anything in this workstream -- `MOE_NARROW_DECODE_MAX` gates the narrow/
bm1 kernels on `max_tokens > 1`):

| N | 2,048 (tok/s) | 32,768 (tok/s) |
|---:|---:|---:|
| single_stream | 102.0 | 84.5 |
| 1 | 101.7 | 83.9 |
| 2 | 116.0 | 91.8 |
| 3 | 126.4 | 99.2 |
| 4 | 142.4 | 109.2 |
| 8 | 169.7 | 128.2 |

This closes the `bucket_live`/split-kernel workstream: `moe_align_block_
size`'s new table and the two kernels reading it (`_narrow`, now `bm > 1`
only, and the new `_bm1` pair) together took every `N` 2-4 width from the
narrow-GEMV win's own baseline (the "MoE's small-bucket GEMV, take two"
section) to the numbers above, in four dated sections each gated on the
same suite (`moe_differential`, `batch_decode` bit-exact, golden) and each
landed only once `cuobjdump -sass` confirmed everything outside its own
stated blast radius stayed byte-identical. `N` 1 and 8 sit where they did
before any of this work started, which is the other half of the same
claim: the sessions never touched what they did not mean to.

## GDN at batch decode: a real bandwidth gap, a plausible fix, and a measured reject once ptxas and the hardware both disagreed with the register math (2026-08-18)

GDN was the one major decode-step stage never bandwidth-audited at batch
shapes, named as the last lever of this session at 23.0% of the N=3 step
per an earlier profile. A fresh `nsys profile --cuda-graph-trace=node`
capture (N=3, 32,768-token context, `LLMXABE_SKIP_SINGLE_STREAM=1`, GPU 2,
last 300ms of the run -- roughly 10 decode steps), kernel time summed per
name and divided by the window total:

| kernel | calls | total ns | share of GDN | share of step |
|---|---:|---:|---:|---:|
| `gdn_proj_q8_0_t4` | 877 | 55,034,615 | 73.3% | 18.7% |
| `gdn_recurrent_step` | 878 | 12,609,178 | 16.8% | 4.3% |
| `conv1d_step` | 876 | 2,451,343 | 3.3% | 0.8% |
| `gdn_normalize_qk` | 878 | 2,080,200 | 2.8% | 0.7% |
| `gdn_alpha_beta_gates_t1` | 292 | 1,666,021 | 2.2% | 0.6% |
| `gdn_silu_split_qkv` | 292 | 716,036 | 1.0% | 0.2% |
| `gdn_add` | 293 | 544,609 | 0.7% | 0.2% |
| **GDN total** | | **75,102,002** | | **25.5%** |
| all kernels in window | | 293,962,608 | | 100% |

25.5% against the cited 23.0% is the same finding from a different capture,
not a contradiction. The ranking is decisive: `gdn_proj_q8_0_t4` -- the
projection kernel this file's own earlier workstream already tuned via
`proj_tile_for`'s tile-selection rule -- is 73% of GDN's own time and
18.7% of the *whole* decode step by itself, an order of magnitude ahead of
`gdn_recurrent_step`, the state-update kernel the brief expected to be
competitive (its 2 MiB/layer/seq state read+write is real traffic, but at
30 layers x 3 seqs x 2 directions it is ~360 MB/step against the
projection's ~1.07 GB/step below). Every other GDN kernel is under 1% of
the step. This audit is about `gdn_proj_q8_0_t4` or it is about nothing.

### Achieved bandwidth: 28% of roofline, well under this file's own GEMV kernels

Three Q8_0 matrices route through the tiled projection per GDN layer per
step at N=3 (`hidden` 2048, `conv_dim()` 8192, `value_dim()` 4096, all from
`GdnGeometry`): `qkv` (8192 x 2048), `gate`/`z` (4096 x 2048), `out` (2048 x
4096). Q8_0 is 34 bytes per 32 elements, so one full read of all three is
`(8192 + 4096) x 64 x 34 + 2048 x 128 x 34` = 35.65 MB; `proj_tile_for`
sends N=3 to tile 4 (the smallest tile that covers 3 tokens in one grid.y
slice), so this is read *once* per layer per step, not three times. Over
30 GDN layers that is 1.07 GB/step, matching `gdn_proj_q8_0`'s own doc
comment ("1.07 GiB of Gated DeltaNet projections") independently.

Dividing the window's `gdn_proj_q8_0_t4` time by its call count (877 calls
/ 90 calls-per-step = 9.74 steps in the window) gives 5.65 ms/step for
1.07 GB: **189 GB/s, 28.2% of the card's 672 GB/s streaming roofline** --
below the MoE narrow kernels' own 38.2%/31.0% (a prior section's finding),
well below the LM head's GEMV at 82-89%.

### Ruling out occupancy: already full

`ptxas -v` on `GDN_BLOCK_SRC` extracted standalone: `gdn_proj_q8_0_t4` is
58 registers, 128-thread blocks (`PROJ_WARPS` 4), zero spill. `65536 / (58
x 128) = 8.8` -> 8 blocks/SM, `8 x 128 = 1024` threads/SM -- the Turing
maximum, already 100% theoretical occupancy at this kernel's baseline
register count. Occupancy is not the lever here; the defect checklist's
other named items (stray kernel choice, load width) do not fit either --
`proj_tile_for`'s tile-4 choice is the correct, already-measured one for
N=3, and the per-lane load (`blk[2 + lane]`, one coalesced byte per lane)
is the same shape as this file's own GEMV kernels use successfully
elsewhere.

### The hypothesis: a runtime-bounded loop with no in-flight depth

What is left is the dequant loop itself: `for (int b = 0; b < blocks; ++b)`
inside `GDN_PROJ_TILED`, where `blocks = k_dim / 32` is a *kernel
parameter* -- a genuine runtime value, not a compile-time constant -- so
nothing about the loop lets `ptxas` unroll it on its own, and nothing in
the body issues a second weight load before the first one's arithmetic has
retired. This is the same defect this project has already fixed twice:
`moe_expert_ffn_gemv`'s `#pragma unroll 2` and `moe_expert_down`'s
`MOE_DOWN_UNROLL` both exist because a fully-occupied GEMV-shaped kernel
with no loop depth is a memory-parallelism bound, not a bandwidth one, and
28% against those kernels' 46-89% is exactly that signature.

Built: `GDN_PROJ_TILED` gained a fourth macro parameter (`KU`, the unroll
factor), threaded through `_Pragma` via the standard two-level stringize
trick (`_Pragma` cannot appear as a bare `#pragma` inside a `#define ...
\` body, and a macro parameter must be expanded before it is stringized).
`ptxas -v` on the four declared widths, unroll factor swept per width
because the fix is not free everywhere:

| kernel | baseline | unroll 2 |
|---|---:|---:|
| `gdn_proj_q8_0_t2` | 52 | 46 |
| `gdn_proj_q8_0_t4` | 58 | 58 (SASS byte-identical) |
| `gdn_proj_q8_0_t8` | 120 | 96 |
| `gdn_proj_q8_0_t16` | 128 | 227 |

`t16`'s accumulator array (`RR` 4 x `TT` 16, 64 floats already) had no
register room left; unrolling doubled its in-flight dequant temporaries
and more than doubled the register count, a full occupancy tier lost. It
kept `KU = 1` (no unroll, its original shape) -- it is never reached below
9 tokens and this session's own sweep never exceeds 8. `t2`/`t8` improved
on paper. `t4` -- the width N=3 and N=4 actually use -- compiled to
**byte-identical SASS**, register-for-register: `ptxas` was already making
the same scheduling decision for this specific loop shape with or without
the source-level hint. That result alone meant this fix could not move
N=3, this session's own headline metric, before a single kernel was ever
launched.

### Gates

`cargo test --release -p xabe-engine --lib block::gdn`: 12/12, including
`the_projection_tiles_match_the_kernel` (updated to check the new fourth
parameter) and `the_tiled_projection_reads_each_operand_once_per_tile`
(unchanged, still passing -- the two reuse invariants this kernel exists
for are untouched).
`gdn_differential` 3/3, `gdn_chunked_differential` 10/10, `gdn_block` 13/13
(including two direct llama.cpp parity tests, `each_step_matches_llama_
cpp_when_fed_its_own_input` and `the_whole_block_matches_llama_cpp_token_
by_token`).
`moe_differential` 7/7 and `batch_decode` 3/3 bit-exact, both unaffected as
expected -- this section never touched `moe.rs`. Golden unchanged.
`cuobjdump -sass` against the merged-main tree (`666224d`) this section
built on: exactly `gdn_proj_q8_0_t2` and `gdn_proj_q8_0_t8` differ (the two
widths whose register count moved); every other kernel in the module,
`gdn_proj_q8_0_t4` and `gdn_proj_q8_0_t16` included, is byte-identical.

### Throughput: a real regression at N=2, flat at N=3/N=4 exactly as the SASS predicted, a sub-3% win at N=8

Interleaved A/B against merged main, `bench_decode_batch`, 3 rounds, 2,048
context:

| N | before (tok/s) | after (tok/s) | delta |
|---:|---:|---:|---:|
| 2 | 115.0-117.3 | 106.2-107.0 | **-8.7% to -9.5%** |
| 3 | 125.1-126.8 | 125.2-126.0 | flat (SASS-identical) |
| 4 | 141.9-143.4 | 141.9-142.2 | flat (SASS-identical) |
| 8 | 169.2-170.8 | 171.5-172.3 | +1.0% to +1.9% |

`t2` regressed by nearly 9% despite `ptxas -v` showing *fewer* registers
(52 -> 46) -- a register-count improvement that did not translate to a
throughput one, the same lesson the split-kernel reject two sections back
already recorded once: occupancy and register math are a proxy, not the
measurement. `t8`'s register win (120 -> 96) did translate to a real gain
at N=8, but at 1.0-1.9% it sits under this round's own 3%-of-step bar, and
N=3 -- the width this whole session was scoped around -- moved not at all,
because its kernel's machine code did not change.

### Disposition

Rejected, reverted (`git checkout -- crates/xabe-engine/src/block/gdn.rs`,
confirmed empty `git diff` against `666224d`). Net effect across the
measured widths is negative-to-flat-to-marginal, and the one width this
session was scoped around shows zero measured change for a structural
reason (`ptxas` already reached the same code), not a small one. Per this
round's own bound: no width cleared 3% of step time in the direction that
would justify shipping a mixed win/loss/flat result, so this closes the
GDN lever as audited-and-declined rather than landed. `gdn_proj_q8_0_t4`'s
28%-of-roofline ceiling stands as the honest number for a follow-up
session: whatever closes it will need to change what `ptxas` schedules,
not hint at a schedule it already reaches on its own -- a restructured
dequant (wider vector loads across `blk`, or trading the byte-at-a-time
`blk[2 + lane]` read for a differently laid-out quant stream) rather than
a loop-unroll hint, and is out of this session's scope.

## The instruction-level diff against `fattn-mma-f16.cuh`: one ruled-out mechanism, one confirmed ceiling, one blocked lever (2026-08-18)

The lead's frame: `decode_mma` `WPO=2` measures ~290 GB/s (43%) at 131,072
keys against llama.cpp's own ~589 GB/s (88%) on the identical geometry (1
token, GQA 8, `head_dim` 256, binary16 KV, Turing), and "registers alone
don't explain a 2x bandwidth gap in a kernel with zero spill." This
compares `fattn-mma-f16.cuh`'s Turing mainloop against `attn_flash_decode_mma`
instruction-for-instruction rather than assuming another register lever
would close it.

### The two kernels' configs at this exact shape

`ggml_cuda_fattn_mma_get_config_turing(256, 256, 8)` (`DKQ=DV=256`,
`ncols1=1` token `x` `ncols2=8` GQA heads `= ncols 8`) resolves to
`nthreads=128` (4 warps), `occupancy=2`, `nbatch_fa=64`, `nbatch_K2=128`
(the whole `head_dim/2` in one trip), `Q_in_reg=true`. `nstages_target=2`
in the table is irrelevant on Turing: `ggml_cuda_fattn_mma_get_nstages`
returns 0 unconditionally when `CP_ASYNC_AVAILABLE` is undefined (Turing,
confirmed by the register-spill section above), so every load takes the
plain, non-`cp_async` branch of `flash_attn_ext_f16_load_tile` -- the same
`ggml_cuda_memcpy_1<16>` 16-byte-per-thread path this file already cited.
**`Q_in_reg=true` for this exact geometry is the same conclusion the
"Staging `Q` to shared" section reached empirically** (37-40% slower):
llama.cpp's own config table keeps `Q` in registers here too, not shared.

### Per-64-key-tile counts, `flash_attn_ext_f16_iter`, Turing, `cols_per_warp==8` branch (matches `ncols==8`)

| | llama.cpp (`nbatch_fa=64`) | `attn_flash_decode_mma_wpo2` (`8*WPO=16` keys/trip) |
|---|---|---|
| keys per softmax-rescale trip | 64 | 16 |
| warps, key-range split | 4 warps, each a disjoint 16-key slice, held from trip to trip | 2 warps, each an 8-key octet, re-split every trip |
| K staging | one `ggml_cuda_memcpy_1<16>` pass, `nbatch_K2=128` = whole `head_dim/2` in one trip | one `uint4` (16 B) pass per `kw4` stripe, same tile width as before this session's spill fix |
| `__syncthreads()` per trip | 2 (after K load: "only needed if `tile_K==tile_V`"; after V load, same) | 4 (after K/V staging landed; after `Q K^T` before softmax; after softmax before `P V`) |
| softmax reduction | in registers, `__shfl_xor_sync` within each warp's own 32 lanes over its own 16-key slice; **no shared-memory round trip** | `s_sh`/`m_sh`/`l_sh`/`corr_sh`, written by `Q K^T`, synced, read by the softmax phase, synced again -- needed because the two warps' key-octets must merge into one shared `(m, l)` *before* the same trip's `P V` can use it |
| cross-warp merge | deferred to once per whole `kb0` range (each warp's running `(KQ_max, KQ_rowsum, VKQ_C)` persists in registers across every trip; llama.cpp's decode path has no cross-warp merge inside this kernel at all -- GQA-8 decode is one `(ncols1, ncols2)` tile, not tiled further) | once *per trip*, via shared memory |
| `Q K^T` MMA shape | `A = K` (`tile<16,8,half2>`, **16 real keys** in the M dimension), `B = Q` (`tile<8,8,half2>`, 8 real heads, single fp16 rounding) -- `mma.sync.m16n8k8.row.col.f32.f16.f16.f32`, **16 keys per call** | `A = Q` (`qa0`/`qa1`, 16 rows: 8 real heads via `q_hi`, 8 via `q_lo`), `B = K` (8 keys) -- same instruction, **8 keys per call** |
| `P V` accumulator | `T_C_VKQ = tile<16,4,half2>` -- **fp16** accumulate | `float o[...][4]` -- fp32 accumulate |

Barrier density: llama.cpp's 2 syncs / 64 keys = 0.031 syncs/key; ours,
4 syncs / 16 keys = 0.25 syncs/key -- **8x denser**, and a genuinely
separate mechanism from `nbatch_fa`/`DMMA_KT` width alone (0.031x2 syncs/key
from tile width, the rest from routing the softmax merge through shared
memory every trip instead of deferring it).

### Ruled out: fp16-accumulate `P V` is not a throughput lever here

`T_C_VKQ`'s fp16 accumulator was the lead's own named candidate
("fp16 KQ accumulation, 2x tensor throughput"). Measured directly rather
than assumed, since `ncu` cannot confirm or deny it: a standalone ablation
(`mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16` against
`...f32.f16.f16.f32`, both driven by `CHAINS=4` independent register
chains over `REPS=4,000` iterations, `72 SMs x 32 blocks x 128 threads`,
CUDA events, 5 rounds) on this card:

| accumulator | ms (round 1-5) | TFLOP/s |
|---|---|---|
| fp32 | 3.0806 / 3.0700 / 3.0629 / 3.0606 / 3.0579 | 98.0-98.8 |
| fp16 | 3.0700 / 3.0595 / 3.0509 / 3.0540 / 3.0495 | 98.4-99.0 |

0.2-0.4% apart, every round -- noise, not a throughput difference. Turing's
`m16n8k8` HMMA issues at the same rate regardless of accumulator width on
this card. This candidate is closed: switching `P V`'s accumulator to
`half2` would buy nothing here, so it was not built.

### Confirmed: `Q K^T`'s 2x MMA issue rate is the precision gate's own cost, not a design accident

The table's `Q K^T` row is the real 2x. `attn_flash_decode_mma`'s own
comment already states the field's actual mechanism precisely (`m16n8k8`'s
A operand is fixed at 16 rows regardless of how many are real, so a
zero-padded single-precision version of *this same kernel* would issue
exactly as many `mma` calls as the `q_hi`/`q_lo` version does -- the split
trick is free *relative to this kernel's own prior attempt*). That is a
true but narrower claim than "free relative to llama.cpp": llama.cpp's
kernel puts **keys**, not heads, in the 16-row M operand, so every one of
its rows is real and every `mma` call covers twice the keys ours does.
Checked whether swapping *our* operand mapping to match (keys in M, heads
in N=8, one plain-precision call) and recovering `q_hi`/`q_lo` precision
via two accumulated calls instead of doubled M-rows changes anything: it
does not -- 2 calls / 16 keys and 1 call / 8 keys are the same rate. No
operand rearrangement escapes it. Computing `Q K^T` at `q_hi + q_lo`
precision costs exactly twice the tensor-core issue count of a single
fp16 rounding, on this instruction shape, full stop -- the same conclusion
the original kernel's design comment reached for *when* it is free (never,
against a plain-precision comparison; only relative to a kernel that was
already going to waste those rows).

This does not cost `P V` anything -- that phase is already single fp16
precision in both codebases, accepted under `MMA_GATE` on our side and
never gated by a residual on theirs -- so the kernel-level cost is nearer
1.5x than the full 2x quoted for `Q K^T` alone, assuming the two phases
are roughly comparable in weight. `Q K^T` doing 2x the tensor-core work
for the same key range extends the per-trip critical path with **zero
extra DRAM traffic**, which is exactly what this file's `GB/s = bytes /
time` convention reports as *worse bandwidth* even though not one extra
byte moved -- the precision tax and the "43% vs 88%" framing are the same
number seen two ways.

### Not ruled out, and not attempted: the barrier-density difference

The 4-vs-2-syncs-per-trip gap is real and, unlike `Q K^T`'s precision tax,
is not gate-mandated -- llama.cpp's independent-per-warp online softmax
(register/shuffle-only, merged once at the end) is a legitimate structural
target distinct from both the register-spill fix and the rejected
shared-`Q` experiment above. Building it means restructuring
`attn_flash_decode_mma`'s warp assignment from "every warp re-merges into
one shared `(m, l)` every trip" to "each warp owns a disjoint key stripe
for the whole split, in registers, merged once at the end" -- a rewrite of
the kernel's synchronization structure, not a parameter change, with a
correctness surface at least as large as the exponent-base bug the
original build caught. Named here as the next concrete lever rather than
attempted rushed: this session's remaining budget after the structural
comparison and the accumulate-throughput ablation was not enough to build
and verify it against the full nine-differential suite at the discipline
this file's other sections hold to, and a half-verified rewrite of the
decode kernel's softmax merge is a worse outcome than an honest stop.

### Disposition

No kernel change this section. The fp16-accumulate lever is closed by
direct measurement (a wash, not a win). The `Q K^T` precision cost is
confirmed as an unavoidable consequence of the `q_hi`/`q_lo` split this
file's own gate requires -- not fixable without loosening `GATE`, which
stays out of scope. The barrier-density difference is real, structural,
and unbuilt: named as the concrete next step rather than guessed at.
`DECODE_MMA_DEPTH_THRESHOLD` and every shipped kernel are unchanged;
nothing here touches `attn_flash_decode_mma`, `attn_flash_decode_warp`, or
the dispatch lever.

### For worker-1's prefill question

`attn_flash_causal_mma` (prefill) does not carry `Q K^T`'s split-precision
cost at all -- it accepts a single fp16 rounding of `Q` under the wider
`MMA_GATE`, per this file's earlier prefill sections, so the mechanism this
section confirms (2x issue rate from `q_hi`/`q_lo`) does not apply there;
prefill's own `MMA_KEY_TRIPS` rejection above hit a register spill instead
of a precision tax. What does transfer: the fp16-vs-fp32-accumulate
ablation above is architecture-level, not kernel-specific -- any Turing
`m16n8k8` accumulator-width question this workstream or worker-1's has is
answered by the same measurement, and does not need re-running per kernel.

## Removing the per-trip cross-warp softmax round trip: built, measured on `ptxas` alone, and the register math the previous section flagged as the open question closes it (2026-08-18)

The previous section named llama.cpp's independent-per-warp online softmax
(register/shuffle state, merged once at the end) as a real, gate-safe
structural lever, distinct from the `Q K^T` precision ceiling, left
unbuilt for lack of session budget. Budget was extended specifically to
build it. This section is the honest result of the most direct, literal
realization of that design: it does not clear the register bar the
previous section's own trap #3 named as the condition for it to be worth
anything, and a second attempt at recovering the difference does not
either -- both checked before spending a differential run on either.

### The design, as specified

Each of `WPO` warps keeps a disjoint 8-key octet for the *whole* split
(the same octet index every trip, not re-split per trip) with private,
register-resident `(m, l)` running softmax state instead of the block-
shared `m_sh`/`l_sh`/`corr_sh` this file's earlier sections built.
`s_sh` stays as the `Q K^T` -> softmax handoff (per the brief: "keeping
the existing shared-memory score path within each warp's stripe"), just
resized `8 * WPO` per row instead of `8` and scoped so only the owning
warp ever touches its own slice -- the two `__syncthreads()` bracketing
that handoff (after `Q K^T`, after the softmax write-back) become
`__syncwarp()`, since no other warp depends on either write anymore. The
two `__syncthreads()` around K/V staging are untouched -- that traffic is
still genuinely cooperative across every warp and was never part of what
this section targets.

The consequence the brief's own trap #3 named directly: `P V` can no
longer split by output dimension across warps (`dbase = warp * dpw`, the
shipped kernel's own scheme) because that split needs every warp's octet
visible in `s_sh` at once -- exactly the round trip being removed. Each
warp instead reconstructs the *entire* `head_dim` output from its own
octet alone, deferring the cross-warp merge to the very end. Reused rather
than re-derived: instead of an in-kernel shared exchange merging `WPO`
warps' partials, each warp writes its own raw `(o, m, l)` to its own
virtual split index `split * WPO + warp`, and `attn_flash_decode_combine`
-- given a new `n_splits` parameter in place of the hardcoded `DEC_SPLITS`
-- merges `DEC_SPLITS * WPO` of them with the exact same log-sum-exp
identity it already used to merge `DEC_SPLITS` block-level partials. No
second copy of that algebra was written.

### The register cost, checked with `nvcc -Xptxas -v` before anything else

`o` is the cost the brief's trap named: covering the whole `head_dim`
instead of `head_dim / WPO` grows it from `o[DMMA_MAXT_(WPO)][4]` (64
registers at `WPO=2`, 32 at `WPO=4`) to `o[head_dim/8][4]` -- 128
registers at *either* width, since it no longer depends on `WPO` once
each warp's coverage is the full output:

| variant | before this section | after (this design) |
|---|---:|---:|
| `WPO=2` | 252 registers, 0 spill | **255 registers, 120 B spill** (stores + loads) |
| `WPO=4` | 235 registers, 0 spill | **255 registers, 104 B spill** |

Both widths hit the 255-register hardware ceiling and spill -- not the
16-56 B the earlier register-spill section fixed, but 104-120 B, 2-7x
worse. Freeing `m_sh`/`l_sh`/`corr_sh`'s shared-address bookkeeping and
the `oc` loop's removal did shave some registers back, per the trap's own
prediction, but nowhere near the ~64-96 registers `o`'s growth cost at
either width. This alone answers "verify the net": it is negative, and a
kernel that hits 255 registers *with* spill is a worse starting point than
the 252/235-register, 0-spill kernel currently shipped -- this file's own
"Widening the softmax-rescale tile" and "Staging `Q` to shared" sections
both already measured spilled variants of this kernel losing 30-40%,
not a close call.

### A second attempt, ruled out by arithmetic before touching `nvcc` again

The obvious mitigation -- move `o` out of registers into a per-warp shared
array, loading and storing around each `mma` call instead of holding all
32 output-dim slots live at once -- was evaluated against the shared-
memory budget before implementing it, the same "verify the arithmetic
first" discipline the "Staging `Q` to shared" section used. `o` is
per-*lane*, not per-warp (`mma`'s C fragment is distributed across a
warp's 32 lanes), so a shared `o` needs `nthr * (head_dim/8) * 4` floats:
at `WPO=2`, `64 * 32 * 4 * 4 B = 32,768 B`, on top of the existing
~21,344 B for K/V staging, `s_sh`, and the resized `m_sh`/`l_sh`/
`corr_sh` -- **~54,112 B total**, `65,536 / 54,112 = 1` block/SM. That is
the exact occupancy-collapse mechanism this file has already measured
twice as a loss: `WPO=4`'s 1-block/SM shape in the original split-
precision-`Q` section, and the register-lever attempts in "Widening the
softmax-rescale tile" and "Staging `Q` to shared" both explicitly traded
away the 3-block/SM margin that makes `WPO=2` win at all. Trading a
register spill for the same occupancy collapse those sections independently
measured as a loss is not a second candidate worth building and
re-measuring -- it is the first candidate's failure mode by a different
route.

### Correctness not checked, on purpose

Neither variant was wired into the real kernel or run against
`attention_differential`. Both already fail the performance bar the
brief itself set ("the win only lands if you stay at 3 blocks/SM and 0
spill") before reaching a differential test, and every prior section in
this file that measured a spilled or occupancy-collapsed variant of this
kernel measured a loss, not a coin flip -- spending a ~7-minute
differential run to confirm the arithmetic of a design that cannot ship
regardless of its answer is exactly the kind of unnecessary verification
this project's own effort discipline argues against. `crates/xabe-cuda/src/kernels/attention.rs`
is untouched (`git status` clean); every design in this section lived only
in a scratchpad `.cu` extraction.

### Disposition

Not built. The literal design the brief specified is register-infeasible
at this model's `head_dim=256`, `WPO in {2, 4}` geometry, and the direct
mitigation is occupancy-infeasible by the same arithmetic this file has
already used twice to reject other levers on this kernel. This is the
"stop there" branch the brief's own closing bar names: one honest attempt
at the literal design (255 registers, real spill, checked with `ptxas`),
one honest evaluation of the direct mitigation (54 KB shared, 1 block/SM,
checked by arithmetic against the same ceiling this file already
calibrated), both negative before a differential run was worth spending.
The last green state is unchanged: `attn_flash_decode_mma` at 7c7c180,
`DECODE_MMA_DEPTH_THRESHOLD` at 16,384, `decode_mma`'s `Auto` dispatch
unchanged. `attn_flash_decode_combine`'s hardcoded `DEC_SPLITS` loop bound
was not touched either -- the `n_splits` parameter above was designed on
paper as part of evaluating the register cost, not implemented in the
shipped kernel, since the design it would have served does not clear the
bar to ship.

### What is actually left

Both routes to removing the per-trip cross-warp round trip -- keep `P V`'s
output-dim split and pay a real barrier (today's shipped shape), or drop
it and pay either a register spill or a shared-memory occupancy collapse
(this section) -- are now measured. A structural fix that keeps the
output-dim split *and* avoids the barrier would need a way to move
`WPO`'s worth of normalized `P` values between warps other than shared
memory, which this architecture does not offer (`__shfl_sync` is strictly
intra-warp; cross-warp exchange on Turing has no cheaper path than shared
memory plus a block-wide sync). Absent a block-shape change bigger than
this session's own bound allows, the barrier-density gap against
llama.cpp identified in the previous section stays a named, understood,
and now doubly-confirmed-expensive-to-close difference rather than a
fixed one.

### The complete ceiling, in one place

Three sections' worth of measurement compose into a single argument for
why llama.cpp's deep-decode advantage over this kernel does not close
further without loosening `GATE`:

1. **`Q K^T` at a single fp16 rounding of `Q`, not `q_hi`/`q_lo`, issues
   keys at 2x this kernel's rate** (the instruction-diff section) --
   forbidden outright: the naive single rounding is the exact change that
   broke the 1e-5 gate at `n_keys = 61` the session this kernel was built,
   which is why the split-precision trick exists at all.
2. **`fp16` `VKQ` accumulation is what makes llama.cpp's barrier-free
   per-warp design fit in Turing's register file.** Not a throughput
   difference -- the ablation above measured `f16`- and `f32`-accumulate
   `mma` within 0.4% of each other -- a *capacity* one: a `half2`
   accumulator packs two logical values per 32-bit register where `float`
   needs one each, so llama.cpp's full-`head_dim`, per-warp `VKQ_C` costs
   half the registers this section's `o` did for the identical coverage.
   That is very likely the specific difference between this section's
   255-register, spilling attempt and llama.cpp's own shipped shape at
   the same geometry, though this session did not build an `fp16`-
   accumulate variant of `attn_flash_decode_mma` to confirm the exact
   number -- the packing argument is register arithmetic, not a
   measurement, and is included as reasoning rather than a fourth
   ablation result. Building it was not attempted: `o` is decode's final
   partial, written once to `part_acc` and read back by
   `attn_flash_decode_combine`'s `expf`-weighted merge over however many
   keys a whole split covers (up to thousands at 131,072 depth), and
   `fp16`'s ~3 decimal digits of precision held across that many
   `*= cg` rescalings is a real, untested risk against the same 1e-5 gate
   `Q K^T`'s residual trick exists to satisfy -- unlike `Q K^T`, `P V` is
   only accepted under the *looser* `MMA_GATE` when measured on its own,
   but decode's gate is applied to the whole output, not per phase, and
   `Q K^T`'s own error already consumes nearly all of the 1e-5 budget by
   itself (1.11e-5 was the naive kernel's actual measured overshoot).
3. **With `fp32` accumulators -- the only precision this session verified
   decode can afford -- every route to llama.cpp's barrier structure this
   file tried costs registers or shared-memory occupancy this card's
   register file and 64 KiB carveout cannot both fund at once,** measured
   shut from two directions in this section alone (255 registers with
   spill one way, 1 block/SM the other) on top of the register wall the
   two sections before it already hit from different starting points.

Put together: deep decode's ceiling past `WPO=2`'s current 0.93x/0.88x/
0.83x at 32K/65K/131K is a precision trade this project chose on purpose,
not an engineering gap this session failed to close. Closing it further
means deciding the 1e-5 gate is negotiable for decode the way it already
is not for anything else in this file -- a project-level accuracy
decision, not a kernel one, and out of scope for this workstream to make
unilaterally.

## `fattn-mma-f16.cuh` at prefill's own geometry: precision ruled out twice over, barrier density confirmed and already blocked (2026-08-18)

The lead's frame: does llama.cpp's `fattn-mma-f16.cuh` accumulate `Q K^T`
and `P V` in fp16 on Turing at Qwen3.6's shape, and if so, does that
explain `attn_flash_causal_mma`'s 0.94x at 131,072? Settled by reading the
source, then confirmed against the actual compiled SASS rather than
assumed from the template code alone -- and cross-checked against
worker-3's independent, hardware-measured answer to the same question on
the decode side, filed just above.

### The accumulator types, and which one is real for this shape

`mma_tile_sizes<DV, ncols>` in `fattn-mma-f16.cuh`, under
`TURING_MMA_AVAILABLE`, has exactly two specializations -- the general
case and `ncols == 8` (decode's GQA-batched single token) -- and **both**
fix `T_C_KQ` to `float` and `T_C_VKQ` to `half2`, unconditionally. There is
no third case and no runtime branch: `Q K^T` accumulates fp32, `P V`
accumulates fp16, on every Turing shape this file instantiates.

Which specialization actually runs for Qwen3.6's prefill was not assumed
from the template alone. `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2`
picks `ncols2 = 8` for `gqa_ratio = 8` (`q_heads=16, kv_heads=2`), then
`switch_ncols1`'s own Turing-specific override --
`ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING` -- caps
`ncols1` at `32/ncols2 = 4` regardless of batch size, because this
project's own `llama.cpp/build` is configured `CMAKE_CUDA_ARCHITECTURES=75`
(confirmed in `CMakeCache.txt`). So the kernel actually dispatched to for
every prefill chunk on this card is `flash_attn_ext_f16<256, 256, 4, 8,
false, false>` -- **not** the `ncols1=64` shape its own file name would
suggest reading the template-instance directory naively, and not the
`ncols==8` decode specialization either (`ncols = ncols1*ncols2 = 32`,
landing in the general case).

`cuobjdump -sass` on that exact symbol, pulled from both the standalone
`.o` and the shipped `libggml-cuda.so.0.19.0` (identical), settles it at
the instruction level:

| instruction | count |
|---|---:|
| `HMMA.1688.F32` | 768 |
| `HMMA.1688.F16` | 768 |

An exact 1:1 split -- `Q K^T` (fp32-accumulate) and `P V` (fp16-accumulate)
do the same number of tensor-core issues, confirming the source-level
typing with no ambiguity left. (A different, unrelated
`flash_attn_ext_f16<256,256,64,1,...>` instantiation -- the no-GQA path,
never dispatched to here -- compiles to a bare `NO_DEVICE_CODE` trap on
this build; worth recording so a future reader who greps the same
`ncols1_64-ncols2_1.cu.o` does not mistake it for live code.)

### Ruled out, twice, by two different methods

**Worker-3's hardware ablation** (filed above, "Ruled out: fp16-accumulate
`P V` is not a throughput lever here") measured `mma.sync.m16n8k8`'s issue
rate directly on this card: fp16- and fp32-accumulate HMMA are 0.2-0.4%
apart, every round -- noise, not a 2x. That single measurement is
architecture-level, not kernel-specific, and settles the lead's own framing
("if llama.cpp accumulates in fp16 it gets 2x the tensor peak") as false
for Turing on this card, for prefill exactly as much as for decode: **there
is no throughput to gain from matching llama.cpp's `P V` precision here,
so there is nothing to report-before-building** -- the gate the lead asked
for (simulate the accuracy cost before proposing the change) does not need
clearing, because there is no performance case for making it regardless of
what the accuracy cost turns out to be.

**A second, independent check anyway** (`crates/xabe-kernels/examples/
fp16_accum_simulation.rs`), run before this was known, because the
decode-side ablation was not yet filed when this section's own
investigation started: a host-side simulation of `attn_flash_causal_mma`'s
online-softmax arithmetic with the `P V` accumulator rounded to fp16 at
every point real `HMMA.1688.F16` hardware would round it (after each
8-key sub-group and after each tile's rescale), at both llama.cpp's own
64-key rescale cadence and our kernel's narrower 8-key one, at every depth
from 512 to 131,072, three seeds at the deepest. Result: **holds with
26-30x margin against `MMA_GATE` (3.906e-3) at every depth and both
cadences**, not trending toward the gate as depth or rescale frequency
grows -- the self-normalizing division at the end of online softmax bounds
a fp16 rounding's relative contribution regardless of how many times it
happens. Two independently-derived answers -- one from hardware
throughput, one from numerical simulation -- agree: **fp16 `P V`
accumulation would cost nothing on accuracy and buy nothing on speed, on
this card.** Not built, for the same reason worker-3's section did not
build it: there is no case for it.

### What the SASS-level structure *does* show: a real, already-named, already-blocked barrier-density gap

Counting `__syncthreads()` inside `flash_attn_ext_f16_iter` for the
dispatched `<256,256,4,8>` shape, `nbatch_fa=64` (from
`ggml_cuda_fattn_mma_get_config_turing`'s `(256,256,32,...)` row): the
`k0_start` loop over `DKQ/2` in steps of `nbatch_K2=128` runs exactly once
(`128 == DKQ/2`), and the `i0_start` loop over `DV` in steps of
`2*nbatch_V2=256` also runs exactly once (`256 == DV`) -- so the whole
64-key tile costs **3 `__syncthreads()`** (K-tile stage, the
`tile_K==tile_V` one, V-tile stage), all its `Q K^T`, softmax, and `P V`
tensor-core work done in between on registers and warp shuffles alone.
`0.047 syncs/key`.

`attn_flash_causal_mma`'s own per-trip macro (`MMA_KT = 8`, this file's
current shipped width) pays **2 `__syncthreads()`** (after staging K, after
staging V) plus 2 `MMA_HEAD_BAR()` (`__syncwarp()`, warp-local and far
cheaper) **every 8 keys** -- `0.25 syncs/key` on the block-wide barriers
alone. **About 5.3x denser** than llama.cpp's prefill tile, the same
pattern worker-3 found for decode (8x, at that kernel's different tile
widths) from the same root cause: llama.cpp stages a whole wide tile once
and does all the arithmetic for it without another block-wide sync; this
file's kernels re-sync the whole block once per much-narrower key octet.

This is not a new, unbuilt lever. It is the mechanism "Widening the
softmax-rescale tile" (two sections up, `MMA_KEY_TRIPS` 1 -> 2 or 4) was
built to attack -- processing more keys per `__syncthreads()` pair is
exactly what widening the softmax-rescale tile means -- and that section
already measured and rejected it on register spill: the kernel sits at
252/255 registers with zero spill at `MMA_KT=8`, and widening trades
occupancy away rather than buying anything, a wash trending slightly
negative at 131,072. Quantifying the barrier-density gap here explains
*why* that lever would have mattered if it had fit; it does not reopen it.

### Disposition

No kernel change from this section, same as worker-3's decode
counterpart. Precision is closed by two independent measurements pointing
the same way. The structural gap is real, sized, and already blocked by
this kernel's own register budget -- not a new candidate, a confirmation
that the one candidate this workstream found was the right one to try,
and the right one to reject.

### The 131,072 ceiling, stated plainly

Deep prefill is bounded by `attn_flash_causal_mma` alone: attention is
71.8% of a deep chunk's kernel time (this workstream's opening profile),
and MoE's remaining share has already been closed as far as this
workstream's own levers reach (see "Two compiled widths instead of one").
Every mechanism this session and worker-3's decode-side investigation
checked for the attention kernel itself is now closed, one way:
`MMA_KEY_TRIPS` widening -- built, rejected on register spill (252/255,
zero spill, no headroom). `P V` accumulator precision -- not a throughput
lever on this card (hardware-measured) and not an accuracy risk either way
(simulated), so not worth building regardless. Barrier density against
llama.cpp -- real (~5x), but it is the same register-bound lever already
tried. No further named lever remains for 131,072 from either workstream
without a structural rewrite of the kernel's staging/synchronization
pattern (register-neutral by construction, not a parameter change) --
named, not attempted, for the same reason worker-3 named its own
barrier-density lever without attempting it: a half-verified
rewrite of a shipped kernel's synchronization structure is a worse outcome
than an honest stop.

## dp4a/Q8_1 activation quantization for the MoE down GEMV: gates clean in isolation, breaks badly in the real decode loop, rejected without a root cause (2026-08-18)

The session's main bet for closing the N=3 gap: llama.cpp's own `mmvq`
scheme applied to xabe's N==1 down-projection GEMV
(`moe_expert_down_gemv`), the only MoE decode kernel still running fp32
activations after the GEMV-family work already put a ceiling on that
design (28-47% of bandwidth, dequant-and-stream bound). `vecdotq.cuh` read
first: neither `vec_dot_q8_0_q8_1_impl` nor `vec_dot_q6_K_q8_1_impl_mmvq`
consumes Q8_1's `sum` term, so the design simplified to a plain
int8-plus-fp32-scale activation block (matching `mma_quantize_rows_q8`'s
existing fp32-scale convention, not a literal fp16-packed Q8_1 struct) and
`__dp4a` in place of the accumulate. Built the down half first, Q8_0
weight, the simpler of the two pairings: a new `moe_quantize_slot0_q8`
activation-quantize kernel (needed because `inter`'s slot-0-per-bucket
rows are not contiguous the way `mma_quantize_rows_q8` assumes) and
`moe_expert_down_gemv_dp4a`, wired into `grouped_forward_partial` behind a
`dp4a_enabled` flag mirroring `disable_tensor_cores`'s shape, reusing
`MoeBuffers::iq`/`iq_scales` (`down_mma` only runs at `max_tokens >=
MMA_MIN_TOKENS`, this path only at `max_tokens == 1`, so the two never
share the buffer in the same call).

### Two bugs, both caught by gating before any perf measurement was run

`moe_differential` 7/7 compiled clean the first time via NVRTC, but
compiling is not exercising: `NUM_TOKENS = 37` in that file means every
existing test's `gemv` branch (`max_tokens == 1`) is dead code, so the new
kernels had never actually launched. Added
`device_grouped_forward_matches_the_reference_at_one_token_with_dp4a`, a
dedicated `max_tokens = 1` differential against the same fp32 CPU
reference the 37-token tests use, at the already-accepted
`ROUTED_MMA_GATE` bound. Two bugs surfaced immediately, both in code this
round had not yet run:

1. **`CUDA_ERROR_MISALIGNED_ADDRESS`.** `dequant_tile_q8_0`'s addressing
   was described (in this round's own design notes) as "reused unchanged"
   for the weight side, but `dequant_tile_q8_0` reads its four bytes one
   at a time through `signed char`, never through a 4-byte cast --
   because a Q8_0 block is 34 bytes, not a multiple of 4, so
   `wblock * 34 + 2 + elem` is 4-byte aligned only for even `wblock`. The
   new kernel used `*(const int*)(wblk + 2 + elem)` to build `dp4a`'s
   packed operand and faulted on every odd `wblock`. Fixed by
   byte-packing the same four bytes by hand (`(int)(unsigned
   char)wblk[...] | (... << 8) | ...`), which needs no alignment and
   produces the identical bit pattern a 4-byte load would have.
2. **A bug in the test itself, not the kernel.** `router_logits()` always
   builds `NUM_TOKENS` (37) rows regardless of the geometry passed in, so
   the first draft of the new test fed a 37-row routing decision into a
   1-token device buffer and indexed `hidden_states[token_idx]` out of
   bounds on the host side. Fixed by building the single row of logits
   directly, the same way
   `the_one_token_dispatch_matches_the_reference_slot_for_slot` already
   does, instead of reusing the 37-token helper.

With both fixed, the new test passes: dp4a down vs. the fp32 host
reference, `cosine=0.999979 max_abs=4.388e-5` -- inside `ROUTED_MMA_GATE`
(`1e-4` / `0.9999`) with room to spare, and consistent with the
coordinator's asymmetry prediction (golden's own reference is llama.cpp
running this exact scheme, so this should if anything read *closer* to
golden than the fp32 path does). The unmodified fp32 `gemv` path, run
immediately after via `disable_dp4a()` in the same test, still matches at
`cosine=1.000000 max_abs=6.98e-9` -- the tight `ROUTED_GATE` bound,
unchanged, confirming the new branch left the old one untouched. All
8/8 `moe_differential` pass.

### `batch_decode`: a real, large divergence the isolated test never saw

`batched_decode_agrees_with_independent_single_stream_decodes` failed:

```
step 0 seq 0: batched id 248068 reference id 248068 max_abs_diff 5.078e-1 cosine 0.999167860
```

against a 5e-3 bound. `identical_prompts_in_one_batch_produce_bit_identical_rows`
still passed at exactly `0.000e0` for all three steps -- bit-exactness
within one batched call survives, as expected, since dp4a is deterministic
per launch. The failure is between the batched-decode run (`N > 1`,
`gemv` false, this section's kernels never launch) and the
independent-single-stream reference (`N == 1`, `gemv` true, this section's
kernels are exactly what runs) -- 5,000x the ~1e-4 divergence clean `main`
produces on the same comparison.

Isolated the cause with a direct A/B rather than trusting the coincidence:
forced `dp4a_enabled: false` at construction (leaving every other line of
the diff in place) and reran the same failing test. It passed, matching
clean `main`'s numbers exactly (`8.6e-5` to `2.3e-4` across all nine
step/sequence pairs). Restoring `dp4a_enabled: true` reproduces the
`5e-1` failure again. This isolates the regression to the dp4a down path
specifically, not to an unrelated change elsewhere in the diff -- the diff
touches nothing outside `moe.rs`'s down-GEMV dispatch and the new test.

What makes this a genuine finding rather than a restatement of the
misalignment bug already fixed above: the isolated one-shot differential
test (fresh, zeroed buffers, one `grouped_forward` call, one random
activation draw) passes cleanly at the expected accuracy, and the same
kernel breaks badly under real serving conditions -- many sequential
calls across 40 layers of real decode steps, reusing the same persistent
`MoeBuffers`, on real (not synthetic-uniform) activation statistics. That
gap was not closed this round: `iq`/`iq_scales` sizing was checked and is
provably large enough (`expert_block_capacity() * intermediate <=
sorted_capacity() * intermediate` for any `block_size >= 1`); the
activation-side `av` read was checked and is provably 4-byte aligned
(`intermediate` is itself a multiple of 32); `partial` is unconditionally
memset before every call. None of those rule out the actual cause, and
the round's remaining budget did not stretch to a real-activation-data
repro small enough to bisect further.

### Disposition

Rejected, reverted (`git checkout -- crates/xabe-cuda/src/kernels/moe.rs
crates/xabe-engine/tests/moe_differential.rs`, confirmed empty `git diff`
against `16bcec2`). Per the standing rule -- reject on any gate failure
before perf work, no exceptions for a plausible design -- this closes
without a throughput number and without attempting the FFN half (Q6_K),
which was next in the build order contingent on down's gates passing.
The isolated differential result stands as evidence the *design* is
sound (matches llama.cpp's own scheme, matches the accepted MMA-path
tolerance, leaves the fp32 fallback bit-for-bit where it was); what
failed is something specific to sustained real-serving use that a
single-call differential test structurally cannot see. A follow-up
attempting this lever again should gate through `batch_decode` -- not
just `moe_differential` -- before declaring victory even on paper, and
should look first at whatever differs between a fresh zeroed `MoeBuffers`
and forty layers deep into a real KV-cache decode: real activation value
distributions (outlier channels, exact-zero SwiGLU blocks) that the
isolated test's `Xorshift64Star::vec_f32(-1.0, 1.0)` draw does not
reproduce, and repeated-launch state in `iq`/`iq_scales` under real
per-layer expert-selection churn rather than one clean call.

### Follow-up: two named hypotheses, both tested directly, both closed

The lead's read of the failure signature: `5.078e-1` on a `5e-3`-bounded
comparison is not a precision gap, it is two paths taking different expert
routes -- quantization noise (~1e-4/layer) compounding over forty layers
until a marginal top-8 selection flips. Two mechanisms proposed, in order.

**Hypothesis 1: asymmetric arithmetic.** The above section's diff wired
`dp4a` into the `max_tokens == 1` GEMV branch only. `batch_decode`'s
`batched_decode_agrees_with_independent_single_stream_decodes` compares a
3-sequence batch step (`narrow`/`bm1`, still fp32) against independent
`max_tokens == 1` single-stream steps (`dp4a`) -- fp32 arithmetic on one
side, int8 on the other, which is not a fair differential regardless of
whether either kernel has a bug. The predicted fix: wire the same `mmvq`
scheme into the batch-side `narrow`/`bm1` kernels so both sides run
identical arithmetic.

Built it. `moe_expert_down_narrow_dp4a` replaces both `moe_expert_down_
narrow` and `moe_expert_down_bm1` in one launch: no register-tiling (that
question is this session's already-closed throughput one, not this pass's),
just a `dp4a` dot product looped once per live row in a bucket, reading
`iq`/`iq_scales` from `mma_quantize_rows_q8` -- reused unmodified from
`down_mma`, which already produces exactly this int8-plus-fp32-scale row
for every one of `sorted_capacity`'s slots -- rather than a new quantize
kernel. Gated behind the same `dp4a_enabled` flag, `self.mma.is_some()`
required in addition since the quantize call is a method on `MmaKernels`.

Two new differentials before touching `batch_decode`, per the standing
"gate before perf, gate before declaring the fix worked" rule:
`device_grouped_forward_matches_the_reference_at_one_token_with_dp4a`
(reused from the section above) and a new
`device_grouped_forward_matches_the_reference_at_narrow_batch_width_with_
dp4a` at `max_tokens = 4`, every token given the *same* router logits so
every routed bucket lands at `bm == 4` -- read `bucket_live` back and
assert some entry exceeds 1, so the test cannot silently degenerate to the
`bm == 1` case the GEMV differential already covers. Both pass: `cosine=
0.999981 max_abs=6.454e-5` at `N=4`, comfortably inside `ROUTED_MMA_GATE`.
`moe_differential` 9/9.

Bound derivation, not reuse by default, since the lead asked for one:
`moe_quantize_slot0_q8` and `mma_quantize_rows_q8` are the same absmax/
round-to-nearest math on the same one-scale-per-32-block layout, and
`dp4a.s32.s32` and `mma.sync.s8.s8.s32` are both *exact* 32-bit integer
accumulations of the same int8 operands -- dp4a is a four-lane-at-a-time
version of the identical reduction, not a different rounding regime. The
error source `ROUTED_MMA_GATE` was derived against (activation
quantization noise, not accumulation) is the same magnitude by
construction, so the bound carries over on that argument, not because it
worked once already.

`batch_decode` did not improve: `max_abs_diff` at step 0 seq 0 moved from
`5.078e-1` (asymmetric) to `7.081e-1` (symmetric) -- slightly *worse*, not
resolved, and failing at step 0 rather than drifting in across steps.
`identical_prompts_in_one_batch_produce_bit_identical_rows` still held at
exactly `0.000e0`. Hypothesis 1 predicted symmetry would close the gap at
its unchanged `5e-3` bound; it did not move the gap in the right direction
at all.

**Hypothesis 2: graph-capture staleness** (the quantize launch running
outside a captured graph, so replay reads stale activations). Ruled out
structurally before running anything: `batched_decode_agrees_with_
independent_single_stream_decodes` calls `Forward::run_batch_decode`,
which issues plain uncaptured launches every step (`forward.rs`,
`run_batch_decode` → `body_batch_decode`, no `begin_capture`). Graph
capture only happens in `capture_batch_step`/`replay_batch_step`, exercised
by the sibling test `a_captured_batch_step_generates_the_same_sequence_
as_the_launch_path`, which passes both before and after this pass's diff.
The failing comparison never touches a captured graph at all.

**Causation re-confirmed, symmetry ruled out as the fix.** A second direct
`dp4a_enabled` A/B on the now-symmetric diff: forced `false`, `batch_
decode` reproduces clean `main`'s baseline exactly again (`8.6e-5` to
`2.3e-4` across all nine step/sequence pairs, identical to the first
section's own A/B). Forced back to `true`, the `7e-1`-scale failure
returns. dp4a is still the cause -- both hypotheses named a *wiring* fix
and neither wiring fix changed that.

**What the evidence is now consistent with, not yet built or tested**:
`main`'s own baseline comparison is not exactly `0.000e0` between a batch
step and an independent single-stream step (`8.6e-5` to `2.3e-4`) — a
real, harmless, pre-existing fp32 reduction-order difference between
`GatedAttentionBlock`/`GdnBlock`'s batched-vs-single-stream kernels, well
inside `5e-3`. Round-to-nearest int8 quantization is a discontinuous
function of its input: an activation element sitting near a quantization
bin's edge can flip to a different quantized code from an input
perturbation far smaller than the bin width, and that flip is not a small
error relative to the perturbation that caused it — it is a full step of
`d`, injected into one element of one 32-block, that this session's own
`ROUTED_MMA_GATE` derivation already treats as `d`-scale per occurrence
rather than continuous. Forty layers deep, one such flip on a
router-adjacent feature is what plausibly turns a `1e-4`-scale, harmless,
already-present cross-path difference into the `5e-1`-scale routing-level
divergence observed — and this reads as an argument for why making the
*scheme* symmetric would not be expected to help: both compared paths
still quantize two inputs that were never bit-identical to begin with, so
symmetry of scheme does not buy symmetry of result when the scheme itself
is a discontinuous function of a divergence that already existed. This is
offered as the most evidence-consistent account of all three
observations — isolated single-call passes, sustained serving fails,
symmetry does not close it, no captured graph involved — not as a
confirmed mechanism; it was not built or measured this pass, and doing so
(instrumenting a specific element's quantized code across the
batch/single-stream comparison to catch a bin flip directly) is future
work, not a re-attempt of this lever without first deciding whether a
quantization-based `mmvq` scheme is compatible with `batch_decode`'s
existing bound at all.

### Disposition (updated)

Rejected again, reverted the same way (`git checkout -- crates/xabe-cuda/
src/kernels/moe.rs crates/xabe-engine/tests/moe_differential.rs`, confirmed
empty `git diff`, `moe_differential` 7/7 and `batch_decode` 3/3 both green
on the clean tree). Both of the lead's named hypotheses were tested
directly rather than argued from -- symmetric wiring built, gated through
two new differentials before touching `batch_decode` at all, and measured
worse, not better; graph capture ruled out from the call graph before a
single kernel ran. The `mmvq`/`dp4a` lever for xabe's MoE decode closes
for this session on that basis: the isolated-differential-passes,
sustained-serving-fails signature survived a real fix attempt at the
lead's own most-likely mechanism, which is stronger evidence against
shipping it than the first section's reject alone was.

## The N=3 ceiling, in one place (2026-08-18)

Closing this workstream the way the two decode-side ceiling arguments
above closed theirs (`ee4ad02`'s "The complete ceiling, in one place" and
the section before it): one composed argument for where N=3 batch decode
sits against llama.cpp and why, pulling together every section this
workstream landed on the way here rather than leaving the answer spread
across a dozen dated entries.

### 1. What landed this session

N=3 aggregate moved from 64.1 tok/s at 32,768 context -- this workstream's
own recorded starting point ("MoE's small-bucket GEMV at batch decode": "at
N=3 aggregate was 64.1 tok/s, below" llama.cpp's 154.58) -- to **99.2 tok/s**
at the same context, **126.4 tok/s** at 2,048: the "Closing sweep" table's
own N=3 row, two sections and one worker-session up. Six fixes were already
landed before this session narrowed the gap further; three more landed in
it specifically -- the narrow/bm1 kernel-selection split (`moe_expert_
ffn_narrow`/`moe_expert_down_narrow` now `bm > 1` only, the new `_bm1` pair
taking the single-live-row case with none of the tiled fallback's
registers), the `bucket_live` dispatch-table precompute both kernel pairs
read rather than re-derive their own `bm` from, and the split-kernel design
re-tried on that precomputed `bm` ("a real win this time" after the
register-budget version of the same idea was rejected once already on a
cost the register math alone did not predict). Every one of the nine is
individually gated (`moe_differential`, `batch_decode` bit-exact, golden)
and SASS-isolated with `cuobjdump` to its own stated blast radius; the
closing sweep's own N=1 (101.7 / 83.9) and N=8 (169.7 / 128.2) rows sitting
within noise of where they stood before any of this session's work is the
cross-check that none of the nine leaked scope into a width it was not
scoped to touch.

### 2. What's structural (`§2.6` recap)

`docs/OPTIMIZATION.md` §2.6 models weight traffic per decode step as (LM
head + projections + shared expert, read once) + 113.377 MB x `D(N)`
(routed experts, read once per distinct expert per layer) + a flat
131.7 MB/token state term. Two consequences bound what any kernel-level fix
-- this workstream's or anyone else's -- can buy at N=3, independent of how
well the kernel is written:

- `D(3) = 23.26` of 256 experts (9.1%), against `D(1) = 8.00` (3.1%): three
  sequences already touch ~2.9x the distinct experts one does, so the
  weight-read amortization batching is supposed to buy is partial by
  construction at this width, not a kernel defect still waiting to be found.
  It improves with `N` (`D(32) = 163.30`, 63.8%), but N=3 sits on the steep
  part of that curve, not the flat part.
- The state term never amortizes, at any `N` -- each sequence's own KV
  cache and GDN recurrent state costs exactly as much traffic per token
  decoded alongside seven others as decoded alone, because there is nothing
  to share. This is not specific to this engine: llama.cpp pays the
  identical per-sequence cost, which is why its own measured efficiency
  (§2.7) holds flat at ~41% of the concurrency-aware roofline from c=1 to
  c=3 rather than climbing -- the batching win it captures is the
  routed-expert term's, because the state term has none to capture, for
  either implementation.

### 3. Why llama.cpp's remaining edge is closed to this engine by design

The two dp4a/Q8_1 sections above are the concrete version of this
argument, not a separate story from it. llama.cpp's own N=3 batch
efficiency rides in part on `mmvq` -- Q8_1-quantized activations and
`dp4a` in the down (and, unattempted here, the up/gate) projection's inner
loop, the same mechanism `docs/ORACLE.md` §8 item 0 already measured as
1,000-10,000x less accurate than this engine's fp32 activation path
against an independent reference. This session built it anyway, on the
coordinator's own read that llama.cpp's golden logits are themselves
produced by `mmvq`, so matching it might read *closer* to golden, not
further. It passed every per-kernel gate this workstream has: `moe_
differential` differentials at a bound derived (not tuned) from the same
exact-integer accumulation llama.cpp's own kernel performs, bit-exactness
on identical-prompt rows held at `0.000e0` in every configuration tried,
and it survived a second attempt at making the batch and single-stream
paths run *symmetric* arithmetic specifically to close the one gate it
did fail.

It failed that one gate both times: `batch_decode`'s cross-path agreement
check, at `5.078e-1` asymmetric and `7.081e-1` symmetric against a `5e-3`
bound -- a routing-scale divergence, not a rounding one. The most
evidence-consistent account on record (previous section) is that int8
quantization's discontinuity amplifies the small, benign, ~1e-4 reduction-
order difference that *already exists* between this engine's batched and
single-stream attention/GDN kernels on clean `main` into expert-selection
flips over forty layers -- and that this is not a defect either
implementation's kernels can be debugged out of, because the discontinuity
is inherent to quantization itself, not to any one kernel's arithmetic.

llama.cpp does not promise that a sequence decodes to the same logits
whether it is batched with others or run alone; it has never needed to,
and nothing in its own test suite checks for it. This engine does promise
that -- it is the shared-session-cache serving contract this whole project
exists for: three instances sharing a card, a sequence's output must not
depend on which other sequences happen to be resident with it at the same
step. Rejecting `mmvq` on `batch_decode`'s gate is that promise enforced,
not an accuracy margin given up for caution. It is the same class of
deliberate trade as the deep-decode precision ceiling `ee4ad02` closed:
a documented, gate-verified line this engine will not cross for a
throughput number, recorded so the next attempt starts from the reason
rather than rediscovering it by tripping the same gate.

### 4. The named future workstream that would legally reopen it

The gate `mmvq` fails is a symptom of an upstream fact, not a property of
`mmvq` itself: batch and single-stream decode are not bit-identical
upstream of the MoE down projection. `batch(1)` already shares the same
GEMV kernels as `max_tokens == 1` single-stream decode (the closing sweep's
own numbers -- `single_stream` 102.0/84.5 against `1` 101.7/83.9 -- are
within measurement noise of each other, not a coincidence) — the ~1e-4
residual lives further up, in `GatedAttentionBlock`'s and `GdnBlock`'s own
batched-vs-single-stream reduction order. A workstream that made those two
paths bit-identical -- not merely close, but exactly reproducing -- would
remove the discontinuity's raw material: with genuinely identical upstream
activations feeding the quantizer on both sides, `mmvq`'s int8 rounding
would be applied to the *same* input in both paths, `batch_decode`'s gate
would clear by construction rather than by chance, and every differential
this session already built and gated would still hold. That is a real,
scoped structural project -- attention and GDN kernel work, not MoE kernel
work, and out of this workstream's own remit -- not a redo of anything
already attempted here. Recorded with this reasoning attached so whoever
picks it up next starts from the conclusion two sessions and two
hypotheses already reached, instead of re-discovering it.

## fp16 P V accumulation as a register lever: the register win was real, the accuracy gate was not (2026-08-18)

The lead's chain, closing the previous section's loop: `MMA_KEY_TRIPS`
widening is the direct barrier-density fix and it was already rejected on
register spill; the fp16-`P V`-accumulate simulation just filed held
`MMA_GATE` with 26-30x margin; an `m16n8k8` fp16 C-fragment is half the
registers of the fp32 one. If switching `P V`'s accumulator funds the
registers `MMA_KEY_TRIPS` needed and didn't have, that closes the gap
this workstream spent two sections calling shut. Built in the prescribed
order -- fp16 `P V` alone first, gated, before touching `MMA_KEY_TRIPS` at
all.

### The register win: real, and bigger than estimated

`attn_flash_causal_mma`'s `P V` accumulator (`float o[MMA_MAXT][4]`)
became `unsigned o[MMA_MAXT][2]`, two `half2`-packed registers per output
tile instead of four floats, driven by a new
`mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16` device function
alongside the existing fp32-accumulate one (`Q K^T` untouched, still
`f32.f16.f16.f32`, exactly as the lead specified). The fragment-layout
assumption -- that the two-per-register packing halves the existing
`(d0,d1,d2,d3)` -> `(d0=row g, d1=row g+8)` mapping cleanly -- was not
taken on faith: a standalone probe kernel ran one `m16n8k8.f16` call with
a known identity-like operand pair and printed every lane's `(d0.x, d0.y,
d1.x, d1.y)` against the expected `D[g][2tig]`, `D[g][2tig+1]`,
`D[g+8][2tig]`, `D[g+8][2tig+1]` -- exact agreement on all 32 lanes.
`nvcc -Xptxas -v` on the resulting kernel:

| | registers | spill |
|---|---:|---:|
| before (fp32 `P V`) | 252 | 0 |
| after (fp16 `P V`) | **198** | 0 |

54 registers freed, zero spill either side -- more than the lead's
25-30 estimate, and comfortably enough to fund `MMA_KEY_TRIPS=2` (which
the earlier reject measured needing headroom this kernel did not have at
252).

### The accuracy gate: broken, by 209x, on the codebase's own most adversarial test

`cargo test --release -p xabe-engine --test attention_differential`
against the unchanged `MMA_GATE` (3.906e-3): 3 of 9 failed.

`device_attention_holds_the_softmax_normalizer_over_128k_dense_keys` --
every key shares one value vector `c`, so the exact answer is `c` at any
depth (softmax weights partition unity) -- measured `max_abs=0.816`,
**209x over `MMA_GATE`**, cosine 0.9875 against a minimum of
`1 - 1e-6`. `device_attention_matches_the_reference_at_a_128k_window`
and `device_attention_matches_the_reference_for_a_mid_sequence_query_block`
both failed on the `1 - 1e-6` cosine floor as well, at `max_abs` still
under `MMA_GATE` (1.17e-3 and 3.07e-4) but the direction error large
enough that both this file's tolerances -- not just one -- catch it.

This directly contradicts the simulation this same session filed just
above, which swept IID-random Q, K, *and* V from 512 to 131,072 keys and
held with 26-30x margin throughout. The simulation was not wrong about
its own inputs; it tested the wrong regime.

### Root-caused, and reproduced on the host for zero more GPU time

`crates/xabe-kernels/examples/fp16_accum_simulation.rs` gained a second
sweep, `run_constant_v_at_depth`, mirroring the failing test exactly: K
stays random (so `Q K^T` scores, and the real online-softmax rescales
they drive, are unchanged), V is one fixed vector `c` for every key. This
reproduces the failure directly, at the same order of magnitude as the
device kernel, and growing with depth rather than bounded:

| keys | fp16-accumulate max_abs | vs `MMA_GATE` (3.906e-3) |
|---:|---:|---:|
| 8,192 | 3.2e-2 | 8.1x over |
| 32,768 | 2.6e-1 | 67x over |
| 131,072 | **7.3e-1** | **187x over** |

Within an order of magnitude of the real kernel's 0.816 at the same
depth, and monotonically worse with depth -- the IID-random-V sweep's
error, by contrast, did not trend with depth at all. Rescale cadence (8
vs 64, the two widths swept in the first sweep) makes no difference here
either: both give the same number to 3 significant figures at every
depth, ruling out tile width as the variable that matters.

The mechanism: with V constant, `P V`'s running accumulator is a scalar
multiple of one fixed vector at every point in the loop -- every
dimension grows together, rather than the independent, largely
self-canceling walk that IID-random V produces (a positive contribution
in one dimension and a negative one in another, most of the time). Growth
between two rescales is purely additive, and unbounded in the number of
terms folded in since the last one -- bounded by how long the running max
goes unchanged, not by a fixed constant. Once the accumulator's magnitude
is large enough relative to fp16's 11-bit mantissa, further
similarly-sized increments round away entirely rather than losing a few
ULP each. IID-random V never drives an individual dimension's magnitude
far enough past a typical increment for this to bite within 131,072 keys,
which is exactly why the first sweep missed it -- and a long run of keys
whose values are highly correlated (repetition, a dominant token, a
near-uniform semantic region of a real sequence) is not a synthetic edge
case a real model never produces; it is closer to the ordinary case than
the IID-random baseline is.

### Disposition

Rejected. `attn_flash_causal_mma` was reverted to fp32 `P V`
accumulation in full -- `git diff` on `attention.rs` after the revert is
empty, byte-identical to the tree before this section's investigation
started. The register win (252 -> 198, zero spill both sides) is real and
independently confirmed via an isolated fragment-layout probe plus
`ptxas`, but it does not ship: a lever that funds `MMA_KEY_TRIPS` by
breaking the accuracy gate `MMA_KEY_TRIPS` was supposed to land inside of
is not a lever. `MMA_KEY_TRIPS` was not attempted on top of it, per the
lead's own stated order (fp16 `P V` gated first) and per AGENTS.md's
accuracy gate being non-negotiable regardless of what performance case
exists on the other side of it.

### What this leaves

Both mechanisms this workstream and worker-3's decode-side investigation
have named for 131,072 are now closed on evidence, not assumption:
`MMA_KEY_TRIPS` widening on register spill (two sections up), and now
fp16 `P V` accumulation -- real registers, but on the accuracy side of
the gate rather than funding a path around it -- joins it. The
barrier-density gap against `fattn-mma-f16.cuh` (previous section) stands
as understood and unclosed by anything this workstream has tried. No
further named lever remains for 131,072 without either a structural,
register-neutral rewrite of the kernel's synchronization pattern, or a
fp16-accumulate design that specifically defends against the coherent-V
regime (e.g. an unconditional periodic renormalization independent of
whether a new max was found, paid for in extra rescale instructions
rather than gated correctness) -- named here, not attempted, since
inventing and verifying a new numerical safeguard against a failure mode
just discovered is a materially different, larger task than the one this
section was asked to complete.

## Batch-vs-single-stream bit-identity, Phase A: a per-layer divergence audit, and no family measures zero (2026-08-18)

The workstream the previous section's own §4 named as the one legal route
back to the N=3 ceiling: make `run_batch_decode` bit-identical to three
independent `run`/`replay_step` calls, per sequence, so a symmetric
`mmvq`/dp4a activation-quantization scheme (the two dp4a sections above)
would quantize genuinely identical inputs on both sides and could no
longer fail `batch_decode`'s cross-path gate by construction. Phase A:
find out *where* the ~1e-4-scale residual `batched_decode_agrees_with_
independent_single_stream_decodes` already tolerates (8.6e-5 to 2.3e-4,
measured against a 5e-3 bound) actually originates, before changing
anything.

### Instrumentation

`Forward::run` and `Forward::run_batch_decode` had no way to read an
intermediate hidden state at a finer grain than "after this layer's
MoE" (`run`'s own `on_waypoint`) or "not at all" (`run_batch_decode` has
no callback). Two new, diagnostic-only methods were added rather than a
parameter on the existing ones: `run_with_stage_waypoints` and
`run_batch_decode_with_stage_waypoints`, each a duplicate of `run`/
`run_batch_decode`'s own loop, but calling `on_waypoint` after the
layer's mixer (`WaypointStage::Mixer`, the buffer neither existing method
ever exposes) as well as after MoE. Kept separate on purpose: `run`'s
callback type is part of `forward_pass.rs`'s golden-test contract, and
`body`/`body_batch_decode` are what `capture_step`/`capture_batch_step`
record into a CUDA graph -- neither should have to carry a parameter that
exists only for this audit. A new bin, `audit_batch_divergence`, drives
both: batch(3) against three independent single-stream decodes, same
synthetic-prompt generator `tests/batch_decode.rs` and `bench_decode_
batch` already use, at context 2,048, three decode steps, comparing
every layer's post-mixer and post-MoE hidden state, per sequence, via
max-abs diff.

### The table

`layer -> family -> max_abs (worst of 3 steps x 3 sequences)`, `audit_
batch_divergence` at context 2,048, `LLMXABE_STEPS=3` (default):

```
layer     family      stage      max_abs  at step   seq
   -1      embed Embed      0.000e0        2     2
    0        gdn Mixer     1.788e-7        1     0
    0        moe Moe       1.788e-7        2     2
    3  attention Mixer     7.153e-7        0     0
    3        moe Moe       3.725e-7        0     0
    7  attention Mixer     2.444e-6        1     0
   11  attention Mixer     6.914e-6        2     2
   15  attention Mixer     2.813e-5        1     0
   18        gdn Mixer     4.756e-5        2     2
   19  attention Mixer     6.416e-5        0     1
   27  attention Mixer     1.922e-4        2     2
   34        gdn Mixer     3.419e-5        2     2
   34        moe Moe       1.111e-4        2     2
   38        gdn Mixer     4.157e-5        2     1
   38        moe Moe       2.213e-4        2     1
   39  attention Mixer     2.098e-4        2     1
   39        moe Moe       2.385e-4        2     1
```

(Full 82-row table -- every layer, both stages -- in the audit's own
stdout; this is the subset that carries the argument. `embed` is a pure
`get_rows` lookup with no arithmetic and measures exactly `0.000e0`, as
expected -- the only family that does.)

### Headline: growth from layer 1 to layer 40

```
      gdn: layer  0 = 1.788e-7   ->   layer 38 = 4.157e-5   (30 layers of this kind)
attention: layer  3 = 7.153e-7   ->   layer 39 = 2.098e-4   (10 layers of this kind)
      moe: layer  0 = 1.788e-7   ->   layer 39 = 2.385e-4   (40 layers of this kind)
    embed: 0.000e0 (should be exactly 0.0 -- a pure lookup)
```

### What this rules out, and what it does not

**Ruled out: a single culprit family.** None of the three measures
`0.000e0`. Gated DeltaNet's *first* layer already disagrees at
`1.788e-7` -- exactly `1.5x` `f32::EPSILON` (`1.1920929e-7`), the
signature of one differently-contracted FMA, not an indexing bug --
before MoE or Gated Attention have run at all. That number is not
inferred, it is the direct confirmation of `batch_decode.rs`'s own
module-doc reasoning: batch `N > 1` takes the tiled `gdn_proj_q8_0_t8`
projection kernel, single-stream decode's `N == 1` step takes the
untiled one, and "NVCC is free to make a different FMA-contraction
choice for the same source expression when the surrounding loop shape
differs." This audit is the first place that was measured layer by
layer rather than only at the final logits.

> **Superseded as a standing residual, not as a diagnosis.** The
> `1.788e-7` GDN layer-0 number, and the "none of the three measures
> `0.000e0`" headline, were true of this tree on 2026-08-18 before the
> next two sections landed. The pairing the module-doc named was the
> right *class* of defect (batch tile vs single-stream GEMV) and the
> wrong *kernel family*: single-stream decode does not take
> `gdn_proj_q8_0` at all when the split-layout repack is resident, it
> takes `gdn_proj_split_gemv`, and the 7.15e-7 isolated gap is that
> layout's reduction order against the standard Q8_0 tile, not NVCC
> choosing two FMAs inside one source. Closed to `0.000e0` at every
> family, every layer, contexts 8 and 2,048, by the section after
> Phase B step 1. The table above is the before-picture.

**Not ruled out, and evidenced against: a single dominant contributor
further downstream.** Attention's per-layer jumps are visibly the
largest of the three (`7.153e-7 -> 2.098e-4` over 10 layers, versus
GDN's `1.788e-7 -> 4.157e-5` over 30, and MoE tracks close to whichever
mixer fed it, not always above it -- e.g. layer 11 attention `6.914e-6`
against that same layer's MoE `2.570e-6`, MoE *smaller*). But "largest
of three nonzero contributors" is a different claim than "the only
nonzero contributor," and layer 0's GDN number alone already rules out
the latter. Attention warranting extra attention when it is unified is
a reasonable prior *for effort allocation within Phase B*, not a license
to unify only that one family and expect `batch_decode` to reach
`0.000e0`.

**Cross-check, not just a new number:** the final layer's MoE output
(`2.385e-4`) lands almost exactly at the top of `batched_decode_agrees_
with_independent_single_stream_decodes`'s own already-measured range for
the final logits (`8.6e-5` to `2.3e-4`, recorded in that test's module
docs and reproduced verbatim by the dp4a sections' `dp4a_enabled: false`
A/B above). The audit's own methodology reproducing the number the gate
test already reports, at the layer whose output feeds the final norm
and LM head, is what makes the per-layer table trustworthy rather than
an artifact of a different comparison.

### What this means for Phase B's shape

The plan this section executes named GDN and Attention as the two
candidates and MoE as `bm1`/`narrow`-vs-`gemv`, "likely the same loop
structure already" and expected to be the cheaper fix. This audit shows
MoE is not free of the residual either -- it measures nonzero at layer 0
and grows to the *largest* single number of the three by layer 39
(`2.385e-4`, edging out attention's own `2.098e-4`) -- though the
table's layer-11 counterexample above (MoE `2.570e-6` against attention
`6.914e-6` at the same layer) means MoE is not simply "attention's
number, slightly larger" either; it has its own, independent
reduction-order dependence on `bm`, not purely inherited from its
input's own residual. Reaching `batch_decode`'s `0.000e0` needs all
three families -- Gated DeltaNet, Gated Attention, and MoE -- made
reduction-order-identical between their batch and single-stream kernels,
not the cheapest one alone with the other two left as "small enough to
ignore": none of the three is small enough to ignore, they are just
different sizes of the same kind of gap, and Phase B's per-family
unification should track the residual after *each* family closes, not
assume closing one clears the gate.

### Gates, and what was and was not touched

`batch_decode` 3/3, `forward_pass` golden 8/8 (including `the_forward_
pass_reproduces_llama_cpps_logits_and_its_argmax`), `decode` 9/9,
`graph_decode` 1/1 -- all unchanged and green, run both in the shared
checkout before this work moved to its own worktree and again inside the
worktree after the commit. No existing method's body was modified; both
new methods are additive duplicates of `run`/`run_batch_decode`, so
nothing gated by those two changed what it exercises. `cargo fmt --all
-- --check` and `cargo clippy --workspace --all-targets` clean.
`cuobjdump -sass` was not run this round -- nothing in this pass touches
a kernel, only host-side loop duplication and a new diagnostic binary,
so there is no kernel whose SASS could have changed.

VRAM: 34.758 GiB peak of 47.27 GiB at context 2,048, batch 3 plus a
single-stream decode-step pass and a prefill pass sharing the resident
28.3 GiB MoE arena -- comfortably under the card's 48 GiB, same
three-shapes-over-one-arena budget `tests/batch_decode.rs` already
spends.

## Phase B, step 1 (MoE narrow/bm1 vs gemv): isolated the pairing, found nothing to fix, quantified it instead (2026-08-18)

The lead's order of attack, cheapest first: MoE's narrow/bm1 kernels
against its single-stream gemv, on the theory that both are one-token-
per-bucket shapes and the reduction-order fix would be small. Diagnosed
before editing anything, per the lead's own instruction. It is not small
-- it is zero. Two isolated differentials prove MoE's own routed-expert
kernels already agree bit-for-bit with the single-stream GEMV they are
compared against, and the audit bin's own data, read one sample at a
time instead of independently per stage, shows the "moe" family's
measured divergence is inherited from its input, not generated by MoE's
kernel selection.

### Two isolated probes, both bit-exact, both new permanent gates

`tests/moe_differential.rs` gained two tests, added specifically to
separate MoE's kernel-selection question from everything upstream of it:
identical real expert weights (layer 0, the file's own Q6_K/Q8_0 mix),
identical routing, identical hidden state, run once through `max_tokens
= 1` (`moe_expert_ffn_gemv`/`moe_expert_down_gemv`, the true single-
stream shape) and once through `max_tokens > 1` with the live-token
count crafted so every active bucket lands at a known `bm`:

- `gemv_and_bm1_isolate_the_one_live_token_case`: one live token at
  `max_tokens = 2` forces `bucket_live == 1` on every active bucket, so
  `moe_expert_ffn_narrow`/`moe_expert_down_narrow`'s `bm <= 1` skip means
  they run zero real tiles and `moe_expert_ffn_bm1`/`moe_expert_down_
  bm1` alone produce the answer. Compared against the same token run
  alone through `gemv`. `max_abs=0.000000e0` over the full 2,048-wide
  output row -- not close, exactly bit-for-bit.
- `gemv_and_narrow_isolate_the_two_live_token_case`: two tokens given
  *identical* router logits, so both land in the same eight buckets and
  `bucket_live == 2` everywhere, forcing the tiled `TM = 2` instantiation
  of `tile_gemm_pair`/`tile_gemm_single` -- the one real structural
  difference from a GEMV a one-live-token probe cannot reach: a weight
  tile dequantized once and applied to two shared-memory-staged columns
  instead of one directly-read column. Token 0's row, compared against
  the same token run alone through `gemv`: `max_abs=0.000000e0` again.

Both pass as `assert_eq!` on the full `f32` vectors, not a tolerance --
these are two different compiled kernels being asked to agree exactly,
the same class of check `identical_prompts_in_one_batch_produce_bit_
identical_rows` already makes for GDN and Attention, extended to MoE's
own kernel-selection axis specifically. `moe_differential` is 9/9 with
both included.

An SASS-level side investigation before either differential ran is worth
recording as a rejected diagnostic, not a rejected fix: `moe_expert_ffn_
gemv` and the inlined body of `moe_expert_ffn_bm1` (`tile_gemm_pair_
direct1`) showed different `FFMA`/`FMUL` instruction ratios under
`cuobjdump -sass` on an `nvcc -arch=sm_75 -cubin` extraction of the
shipped `MOE_SRC` string, which read at first like the same FMA-
contraction-choice mechanism already confirmed for GDN. Forcing every
per-row accumulation in both functions to `__fmaf_rn()` explicitly,
recompiling, and re-diffing the SASS produced **zero change** in either
function's instruction counts -- the FFMA/FMUL split was already
identical before and after, meaning NVCC was already contracting these
identically and the SASS difference some other cause (very likely the
`#pragma unroll 2` outer-loop depth difference inflating total
instruction count, not a per-element rounding difference). This was
caught *before* editing the real source, by running the actual isolated
differential first rather than trusting the SASS reading -- the
differential is what showed the two kernels already agree, which the
SASS metric alone could not have settled either way. No `__fmaf_rn`
change was made to `crates/xabe-cuda/src/kernels/moe.rs`; the extraction
and probe lived entirely in a scratch directory outside the repo.

### The router logits projection was already engineered for this, on record

`moe_block_router_logits` (prefill/batch, `TT = 8`) and `moe_block_
router_logits_t1` (single-stream decode, `TT = 1`) in `crates/xabe-
engine/src/block/moe.rs`'s `GLUE_SRC` are one macro, `ROUTER_LOGITS`,
instantiated twice -- not two hand-written kernels that happen to look
similar. `ROUTER_JC` is required to equal the block width in both
instantiations specifically so the per-thread contraction partition is
identical between them, the activation tile is deliberately left
un-vectorized for the same reason (a `float4` load would hand each
thread a different partition of the sum), and the file's own comment
records a real historical bug this exact mechanism produced and a
regression test that caught it: "a first attempt... changed the per-
thread partition... the logits moved in their last bits... `tests/
forward_pass.rs` caught block 31's error jumping 5.26x." This is the
same class of fix this whole workstream is generalizing, already applied
here, with its own scar tissue on record. Not re-derived from scratch
this round -- taken as evidenced by that history and by golden passing
8/8 unchanged -- but flagged as the next thing to differential-test in
isolation (mirroring the two MoE probes above) if GDN and Attention's
own fixes do not fully close the gap once they land.

### Per-sample correlation: what "moe" was actually measuring

`audit_batch_divergence` (Phase A's own tool) gained a third report
section: for the layer/step/sequence that maximizes each layer's *mixer*
divergence, what did the *moe* stage read on that exact same sample --
not independently, which is what the original table did and which lets
the two numbers land on different samples and mean nothing side by
side.

```
layer     family        mixer   moe (same)    ratio
    0        gdn     1.788e-7     1.192e-7    0.67x
    3  attention     7.153e-7     3.725e-7    0.52x
    7  attention     2.444e-6     1.132e-6    0.46x
   15  attention     2.813e-5     1.264e-5    0.45x
   19  attention     6.416e-5     1.949e-5    0.30x
   26        gdn     1.877e-5     5.794e-5    3.09x
   27  attention     1.922e-4     8.845e-5    0.46x
   30        gdn     2.791e-5     6.604e-5    2.37x
   31  attention     1.059e-4     2.788e-5    0.26x
   34        gdn     3.419e-5     1.111e-4    3.25x
   38        gdn     4.157e-5     2.213e-4    5.32x
   39  attention     2.098e-4     2.385e-4    1.14x
```

(Full 40-layer table in the bin's own stdout.) Every attention layer in
the run shows `moe` *smaller* than `mixer` on the same sample (0.26x-
0.52x) -- MoE damping the residual it inherited, not adding to it. Most
GDN layers sit near 1x (0.6x-1.4x, pure carry-through, consistent with
the two bit-exact differentials above: same kernel family, same input,
same output). Four GDN layers (26, 30, 34, 38) spike to 2.3x-5.3x on
their worst sample specifically -- the signature this session's own
dp4a/Q8_1 sections already named for a different mechanism: "round-to-
nearest ... is a discontinuous function of its input: an activation
element sitting near a quantization bin's edge can flip ... not a small
error relative to the perturbation that caused it." MoE's router does
the same kind of thing to a *routing* decision instead of a quantization
code: a top-8-of-256 selection is also a discontinuous function of its
input, and a token whose 8th and 9th expert logits are close enough can
select a different expert set from an input that differs by nothing
more than the ~1e-5-scale residual GDN's own reduction order already
produces upstream, with nothing in MoE's own arithmetic at fault. This
reads as the same class of finding as the dp4a sections' "Follow-up"
account, arrived at independently and on `main`, with no quantization
involved at all -- the *routing* discontinuity is a live gap even
without dp4a in the picture.

### Why this retroactively explains the dp4a rejections, not just this session

Restated plainly, because the lead flagged it as the strongest evidence
yet for a decision this workstream already made on weaker grounds: **an
ordinary top-8 routing flip, sourced from nothing but GDN's own
~1e-5-scale batched-vs-single-stream reduction-order residual, already
happens on `main` today, with no quantization, no dp4a, no int8
anywhere in the picture.** The four-layer spike table above is not a
hypothetical about what *would* happen if activations were quantized —
it is measured, on the clean tree, right now.

This is the same mechanism the dp4a sections diagnosed and named but
could not fully confirm at the time: "the most evidence-consistent
account of all three observations... not as a confirmed mechanism; it
was not built or measured this pass." That gap is closed. dp4a's
`5.078e-1`-to-`7.081e-1` blowups were never really about `dp4a`'s own
int8 rounding being *inaccurate* — `moe_differential`'s isolated tests
gated it at the same bound llama.cpp's own golden reference implies, and
it passed. What actually broke `batch_decode` was that dp4a's rounding
is a *second*, sharper discontinuity stacked on top of a *first*
discontinuity (routing) that was already firing on its own, unquantized,
for the reason this session's audit now shows directly. Making
activation quantization symmetric between the batch and single-stream
paths, which is what both dp4a follow-up attempts tried, could not have
closed the gate: symmetric quantization of two already-non-identical
inputs still quantizes two different things, and — this section's own
finding — even *without* quantization at all, two non-identical inputs
already select different experts often enough to spike the comparison
by 2-5x at individual layers, forty layers deep enough to compound into
the `5e-1`-scale failures dp4a produced.

This is why `0.000e0`, not "small enough," is the only target that can
ever make a discrete top-k decision — routing today, `mmvq`'s
quantization codes if it is re-attempted after this workstream closes —
safe to sit downstream of. Any nonzero residual, however small, is a
standing invitation for *some* discontinuous decision somewhere in forty
layers to flip on it; this section demonstrates that invitation is
already being accepted, today, by routing alone. A future worker
re-attempting `mmvq` after GDN and Attention reach `0.000e0` is not
betting that quantization noise is "small enough" anymore — they are
removing the raw material both discontinuities need, which this section
is the first place in the record to show is necessary and not merely
sufficient-in-theory.

### Disposition: no fix, a load-bearing negative result

Nothing in `crates/xabe-cuda/src/kernels/moe.rs` or `crates/xabe-engine/
src/block/moe.rs` changed. The lead's predicted "smallest structural
diff" was, on measurement, no diff at all: MoE's routed-expert kernel
family (gemv, bm1, narrow) is already reduction-order-unified, proven
by two new bit-exact differentials rather than argued from SASS or
source-text similarity, and the router logits projection was already
built the same way with its own regression test and its own scar on
record. What `audit_batch_divergence` measures as "moe" family
divergence is, on the evidence above, almost entirely propagated from
whatever the layer's mixer (GDN or Attention) already handed it, with an
occasional multiplicative spike where that propagation crosses the
router's own top-8 decision boundary -- a consequence of GDN's/
Attention's unresolved divergence, not an independent MoE defect.

> **Superseded in one clause, confirmed in the other.** The routed-
> expert pairing (gemv / bm1 / narrow) and the router logits projection
> remain what this section measured: already bit-identical, nothing to
> edit. The "almost entirely propagated" reading of the *moe* waypoint,
> and the "nothing in `moe.rs` changed" sentence, do not cover the
> shared expert. That pairing -- `moe_shared_ffn_gemv` (4-warp
> contraction, smem 4-add, fused SwiGLU) against `moe_shared_ffn` (one
> warp per row, `tile_gemm_pair` over `MOE_TM=16`) -- was never
> isolated here, generates `1.12e-8` on identical input, and is what
> the next section closed. After the mixers *and* that pairing close,
> the moe waypoint is `0.000e0` at every layer, which is the
> inherited-only claim becoming true rather than assumed. Do not re-
> isolate the routed pairing; do not treat this section as a license
> to leave the shared expert alone.

**This changes the milestone, not the plan.** "MoE layer-0 divergence
0.000e0" as a standalone checkpoint is not reachable by editing MoE
*routed experts*:
layer 0 is a GDN layer, MoE's layer-0 output is proven to be a function
of GDN's layer-0 output and nothing else divergent *in the routed
path*, and the two bit-
exact differentials above mean a bit-identical input to those kernels
necessarily
produces a bit-identical routed output (same kernels, same router, same
weights). MoE's own milestone completes automatically once GDN's does
*and* the shared-expert pairing is unified --
it is not a separate unit of work on the routed side, and the routing-flip
risk closes with
it too, by the same construction argument the coordinator's own §4 made
for the dp4a lever ("with genuinely identical upstream activations
feeding the quantizer on both sides, `mmvq`'s int8 rounding would be
applied to the same input in both paths"). Recorded here so the next
worker on this workstream -- and worker-5, whose GDN state-output change
lands in the same file this session's real fix will eventually touch --
starts from this rather than re-running the same isolation and getting
the same zero.

GDN itself (the lead's step 3, named "the hardest" and deliberately
last) is now also, in effect, step 1: nothing downstream of it --
including MoE's own milestone and, per worker-5's coordination note,
a landing GDN-kernel change from a different workstream -- can reach
its own milestone without it going first. Reported to the lead rather
than unilaterally reordering a workstream another worker (worker-5) is
also coordinating around; reorder approved (2026-08-18), GDN next,
attention projections after. The following section is that work.

## Phase B, steps 2-3: close GDN's split-layout pairing, the shared expert the GEMV already summed, and the 2..63 repack hole (2026-08-18)

The previous section left three claims standing: GDN is the remaining
source of the residual; MoE's routed pairing is already bit-identical
so its milestone completes automatically once GDN's does; and the
`1.788e-7` GDN layer-0 number is the tiled `gdn_proj_q8_0_t8` versus
the untiled `gdn_proj_q8_0`. All three were right about *class* and
wrong about *which kernel*. Closing the actual pairings, then
re-running the same audit at the same two shapes, is this section.

Nothing here is a throughput claim. CUDA graph capture was not
re-measured. Fused MoE dispatch was not touched. Symmetric `mmvq` is
now a legal next attempt -- batch and single-stream activations agree
bit-for-bit, so a discrete quantizer sitting on them would see the
same input -- and it is not done.

### Isolated: GDN's real pairing is split-layout GEMV vs standard-layout tile

`tests/gdn_proj_differential.rs` (new, three tests, all `assert_eq!`
or an explicit `assert_ne!` documenting the remaining layout gap)
isolates the projection kernels on layer-0's real Q8_0 weights, one
row of IID hidden state, no mixer, no residual add:

- `untiled_and_tiled_standard_layout_already_agree`: `gdn_proj_q8_0`
  (`tokens == 1`) against `gdn_proj_q8_0_t*` (`tokens > 1`, token 0).
  Both walk `blk[2 + lane]` in the same per-lane order over the
  standard 34-byte Q8_0 block. `max_abs=0.000e0`. Phase A's
  module-doc reasoning -- "NVCC is free to make a different
  FMA-contraction choice when the surrounding loop shape differs" --
  does not fire on this family. The two standard-layout kernels were
  never the residual.
- `split_layout_gemv_disagrees_with_the_standard_layout`:
  `gdn_proj_split_gemv` (repacked, 4-per-lane) against `gdn_proj_q8_0`
  (standard 34-byte). `max_abs=7.153e-7`. The kernel's own comment
  already said this ("the warp reduction sums in a different
  order... equivalent rather than bit-identical"); this is that
  number, measured, not inferred. This *is* the residual Phase A
  named, just not the pairing it named: single-stream decode takes
  the split GEMV whenever `gdn_int8` is resident, and batch decode
  was taking the standard tile.
- `split_layout_tiled_agrees_with_the_gemv`: new
  `gdn_proj_split_t{2,4,8,16}` (`project_split_tiled`, tokens=3 so
  the covering tile is t4, token 0 compared) against
  `gdn_proj_split_gemv`. Same operand order, same 4-wide lane walk,
  same warp reduction, just a token axis. `max_abs=0.000e0`.

`run_batch_decode` at `1 < tokens < MMA_SPLIT_TOKENS` now takes
`project_split_tiled` for qkv, gate, and out. `run` is unchanged:
GEMV at one token, standard-layout tile otherwise. The fused
`gdn_proj_split_gemv_add` has no token axis, so the batch out-
projection is project-then-`add`, matching the standard-layout path
rather than inventing a fused form. Attention projections were not
rewritten: `lm_head_rows<BT>` already `assert_eq!`s b1 against b5,
and the decode mixer is per-sequence.

### Isolated: the leftover after GDN closed was the shared expert, not the router

With GDN layer-0 at `0.000e0` on the isolated probe, the next
generated residual sat in MoE after all. Phase B step 1 never
compared `moe_shared_ffn_gemv` to `moe_shared_ffn`. Two new tests in
`moe_differential.rs`:

- `router_t1_and_tiled_isolate_the_same_row`: `moe_block_router_
  logits_t1` against `TT=8`, token 0 of a three-token block.
  `max_abs=0.000e0`. Confirms the "already engineered" claim from
  the previous section; `ROUTER_JC == THREADS` is doing what the
  comment said.
- `shared_gemv_and_tiled_isolate_the_one_live_token_case`:
  `moe_shared_ffn_gemv` / `moe_shared_down_gemv` (`max_tokens == 1`,
  one row per block, 4 warps split the 2,048-term contraction, 4-add
  in shared memory, fused SwiGLU) against `moe_shared_ffn` /
  `moe_shared_down` (`max_tokens == 3`, one warp per row,
  `tile_gemm_pair` / `tile_gemm_single` over `MOE_TM=16`). First
  run, before any kernel change: `max_abs=1.117587e-8` on the
  identical row. That is generated, not inherited. It is also why
  the previous section's "almost entirely propagated" reading of
  the moe waypoint was one pairing short.

The fix is the same shape as GDN's: keep the GEMV's reduction tree
and put a token axis on it. `moe_shared_ffn_gemv_t{2,4,8,16}` and
`moe_shared_down_gemv_t{2,4,8,16}` (`#pragma unroll 2` on ffn, 4 on
down; per-lane `dequant_tile` + `float4` activation;
`warp_reduce_tile<TT>`). `shared_expert` still dispatches on
`max_tokens` -- a geometry constant, not the live count -- so the
compiled kernel is fixed for CUDA-graph reasons that cost nothing
to keep: 1 → old GEMV; 2..=16 → tiled GEMV (smallest covering tile,
so a 3-token decode is one t4 launch); >16 → original
`moe_shared_ffn` tile; >=128 still the int8 MMA. Prefill above 16
is not a batch-vs-single-stream pairing and was left alone.

After the wire: the same isolate is `max_abs=0.000e0`.
`moe_differential` is 11/11.

### The audit at context 2,048 then read zero, and the audit at context 8 did not

`audit_batch_divergence` at the Phase A shape (context 2,048, 3
decode steps, batch 3), GPU 1, after both kernel families landed:

```
      gdn: every layer, every step, every seq = 0.000e0
attention: every layer, every step, every seq = 0.000e0
      moe: every layer, every step, every seq = 0.000e0
    embed: 0.000e0
```

Headline growth from the Phase A table -- gdn `1.788e-7 → 4.157e-5`,
attention `7.153e-7 → 2.098e-4`, moe `1.788e-7 → 2.385e-4` -- is
now `0 → 0` on every family. Peak VRAM 34.758 GiB, same three-
shapes-over-one-arena budget as Phase A. The four-layer routing-
flip spike table from the previous section is gone with its input:
there is no ~1e-5 mixer residual left for the 8th/9th expert
logits to sit on.

`batch_decode` at this point was *not* zero. `PROMPT_LEN=8`, 3
steps, 3 sequences: token ids agreed, cosine 1.0, logit
`max_abs_diff=1.184e-3`. That is larger than Phase A's final-logit
range, not smaller. The same audit re-run at `LLMXABE_CONTEXT=8
LLMXABE_STEPS=3` -- the gate test's own shape -- showed why:

```
      gdn: layer  0 = 1.788e-7   ->   layer 38 = 1.459e-4
attention: layer  3 = 3.128e-4   (grows from the inherited GDN residual)
      moe: layer  0 = ...        ->   layer 39 = 5.150e-4
           layer 34 moe / mixer = 26x   (routing flip, same signature)
```

GDN layer 0 at context 8 is the *same* `1.788e-7` Phase A measured
at context 2,048, which is the number of falling back to the
standard-layout tile. The new kernels were compiled and the isolate
was green; the batch-decode shape was not taking them.

### The 2..63 hole: `reshape` inherited an empty `gdn_int8`

`Forward` is a fixed-width object. Prefill and decode are two
`Forward`s sharing MoE weights and `gdn_int8` through `reshape`.
Three consumers of that buffer: integer tensor cores at
`tokens >= MMA_SPLIT_TOKENS` (64), `gdn_proj_split_gemv` at 1, and
now `gdn_proj_split_t*` at `1 < tokens < 64`. The build predicate
was still `wants_repack = uses_tensor_cores(tokens) || tokens == 1`.
An 8-token prefill -- `batch_decode`'s `PROMPT_LEN` -- built an
empty vector; the batch-3 reshape inherited nothing; `run_batch_
decode` took `project()` over the standard Q8_0 layout. A 2,048-
token prefill is already past 64, so it built the repack, the
batch reshape inherited it, and the audit at that depth was
already `0.000e0`. Same kernels, two prefill widths, two answers.

The 4..63 hole was already on record as a *decode-speed* defect
(a one-token decode spawned from those prefills fell back to the
fp32 projection and paid 13%: 9.7 ms became 11.0). It is also a
correctness defect the moment a 2..3-token batch decode is
spawned from the same empty vector. Every width now has a
consumer, so every width builds the repack. `disable_tensor_cores`
still produces an empty vector and still rebuilds on the next
reshape that wants it.

After the predicate change, same GPU, same bins:

```
LLMXABE_CONTEXT=8 LLMXABE_STEPS=3 audit_batch_divergence
      gdn / attention / moe / embed: 0.000e0 at every layer
      peak VRAM 33.008 GiB
```

`cargo test --release -p xabe-engine --test batch_decode`: 3/3.
Logit `max_abs_diff=0.000e0` on every (step, seq) cell of the
independent-single-stream comparison -- was `1.184e-3` one
predicate ago. `identical_prompts_in_one_batch_produce_bit_
identical_rows` still `0.000e0`. Captured-graph ids match the
launch path. Peak VRAM on that test 0.102 GiB reported (arena
2.291 GiB).

### Gates

All on the worktree, 2026-08-18, one process, cards pinned:

| Gate | Result |
| --- | --- |
| `gdn_proj_differential` (GPU 1, release) | 3/3, split tiled vs gemv `0.000e0`, standard vs split `7.153e-7` (documented `assert_ne!`) |
| `moe_differential` (GPU 2, release) | 11/11, shared gemv vs tiled `0.000e0`, router t1 vs TT=8 `0.000e0` |
| `audit_batch_divergence` ctx 2,048 / 3 steps / batch 3 | every mixer + moe waypoint `0.000e0` |
| `audit_batch_divergence` ctx 8 / 3 steps / batch 3 | every mixer + moe waypoint `0.000e0` |
| `batch_decode` (release) | 3/3, logits `0.000e0` every cell |
| `cargo fmt --all` | clean |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean |

`forward_pass` golden and `decode` were not re-run this pass. The
logit comparison in `batch_decode` is the stronger statement for
this workstream (independent single-stream vs batched, bit-exact,
the test whose tolerance this stream exists to delete). They
should be run before anyone quotes this as a merge.

### What this unlocks, and what it does not

`0.000e0` at every family is the condition the previous section
named as necessary before a discrete downstream decision --
routing today, `mmvq`'s quantization codes if it is re-attempted --
is safe to sit on. Routing no longer has a residual to flip on;
the 26x layer-34 spike is gone with it. Symmetric `mmvq` is
therefore a legal lever again, in the narrow sense that both
paths would quantize the same bits. It is not built, not
measured, and the last three attempts at it failed for reasons
that were *partly* this residual and *partly* their own. Do not
read this section as "go build `mmvq`."

Attention's mixer and the LM head were already on a bit-identical
pairing and stayed that way; they did not need a kernel. Prefill
above `SHARED_GEMV_MAX_TOKENS=16` still uses `moe_shared_ffn`'s
one-warp-per-row tile, which is not compared against the GEMV
and does not have to be: there is no single-stream prefill of
that width. `run` at `1 < tokens < 64` still takes the standard-
layout GDN tile; only `run_batch_decode` was moved onto the split
tile. A future worker who wants `run` of an 8-token prefill to
agree with `run_batch_decode` of the same 8 tokens would need to
move `run` too -- that pairing is not what `batch_decode` gates.

N=1 and N=3 decode throughput against llama.cpp were not
re-measured. This commit does not move the head-to-head.

## 2026-08-18 — The serving path runs on three cards without moving the N=3 bar

The scheduler/runtime integration was exercised on three idle Quadro RTX
8000s. `worker_smoke` completed at N=1, N=2, and N=3. The first
`engine_smoke` rejected a real bug: three workers concurrently entering
global CUDA graph capture invalidated worker 0's width-2 capture with
`CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`. Changing capture mode to
thread-local is the correct ownership boundary because every context and
stream permanently lives on its own runtime thread. After that change,
`engine_smoke` completed nine sequences, placed exactly three on each card,
and every worker executed the required two-decode plus one-prefill mixed
step. `cross_worker_restore` also passed across two independent contexts;
the restored and cold paths emitted exactly `[82, 198, 248045, 271]`.

Decode was then re-measured on GPU 0 with a freshly rebuilt
`bench_decode_batch`, 32 timed graph replays after four warmups. Three
repetitions per context:

| context | shape | aggregate tok/s, three runs | recorded 2026-08-18 baseline | result |
| ---: | --- | --- | ---: | --- |
| 2,048 | single stream | 101.1, 100.2, 100.4 | 101.3 | within 1.1% |
| 2,048 | batch 3 | 143.6, 143.1, 143.0 | 145.3 | within 1.6% |
| 32,768 | single stream | 82.5, 81.3, 80.8 | 82.5 | spread reaches -2.1% |
| 32,768 | batch 3 | 106.9, 107.0, 106.7 | 107.0 | within 0.3% |

An initial 2,048-token sweep printed 122.9--123.8 tok/s for batch 3. It was
not a code result: `target/release/bench_decode_batch` had not been relinked
after the always-repack change. Rebuilding the named binary restored
143.0--143.6 tok/s. Record this reject because invoking an existing target
binary directly does not ask Cargo whether it is stale.

llama.cpp was measured three times on GPU 1 with its best decode flags:
`-ngl 99 -sm none -fa on -b 4096 -ub 4096 -ctk f16 -ctv f16 -c 131072
-npp 2048,32768 -ntg 32 -npl 1,3`.

| context | N | llama.cpp tok/s, three runs | llmxabe median tok/s | median ratio |
| ---: | ---: | --- | ---: | ---: |
| 2,048 | 1 | 98.86, 98.59, 98.16 | 100.4 | 1.02x |
| 2,048 | 3 | 188.40, 188.23, 187.14 | 143.1 | 0.76x |
| 32,768 | 1 | 88.44, 88.23, 87.74 | 81.3 | 0.92x |
| 32,768 | 3 | 153.67, 152.86, 153.10 | 106.9 | 0.70x |

The serving work did not materially regress the isolated N=3 path. It also
did not improve it: the same 24--30% aggregate-throughput loss to llama.cpp
remains. The goal's N=3 performance criterion is therefore still open; the
successful three-card serving acceptance is not evidence that the throughput
gap closed.

## 2026-08-18 — The serving benchmark reaches the kernel bar, and `-ub 2048` raises the decode bar

The replacement target now includes llama.cpp at both `-b 4096 -ub 2048` and
`-b 4096 -ub 4096`; the faster setting for each metric and context is the bar.
This is not redundant. Three alternating llama.cpp pairs on GPU 1, same model
and the final section's other flags, produced:

| context | N | `-ub 2048` decode tok/s | `-ub 4096` decode tok/s | best median |
| ---: | ---: | --- | --- | ---: |
| 2,048 | 1 | 98.40, 98.15, 96.28 | 97.71, 97.47, 97.13 | **98.15** (`2048`) |
| 2,048 | 3 | 189.34, 187.90, 182.14 | 186.61, 184.73, 185.20 | **187.90** (`2048`) |
| 32,768 | 1 | 88.74, 87.46, 86.44 | 87.40, 86.38, 86.73 | **87.46** (`2048`) |
| 32,768 | 3 | 154.33, 153.79, 152.76 | 152.23, 147.03, 153.82 | **153.79** (`2048`) |

Decode prefers 2048 in this session. Prefill does not have one global winner:
at N=3/32K, for example, `-ub 4096` printed 2,534--2,547 tok/s while
`-ub 2048` printed 2,340--2,389. Future head-to-head tables must therefore
retain both baselines instead of replacing one with the other.

`bench_worker_decode` was added to time the live `Worker::step_device` path:
admission is complete before timing, n-gram drafting is disabled, the first
request-specific graph capture and four warmups are discarded, and each timed
call must remain a pure fixed-width decode. EOS is deliberately an ordinary
token in this benchmark; its first version let one synthetic sequence stop,
then waited forever for N=3 to return. The second version caught that shrink
but gave the earliest sequence too little output runway and rejected the last
timed step. Both are measurement-harness rejects, not engine results.

One scheduler-path run per cell on GPU 0, 32 host-wall timed steps:

| context | N | mean ms/step | aggregate tok/s | isolated median above | result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2,048 | 1 | 10.00 | 100.0 | 100.4 | within 0.4% |
| 2,048 | 3 | 20.82 | 144.1 | 143.1 | within 0.7% |
| 32,768 | 1 | 12.26 | 81.6 | 81.3 | within 0.4% |
| 32,768 | 3 | 27.99 | 107.2 | 106.9 | within 0.3% |

These are single runs establishing equivalence, not the three-pair claim run.
They close one ambiguity: scheduler/runtime plumbing is not the 24--30% N=3
gap. The live serving path reaches the isolated kernel bar.

Worker construction was made observable while building this benchmark. A
2048-token prefill shape takes about 19 seconds including NVRTC; seven tail
shapes and three decode shapes then take about 0.2 seconds from the in-process
PTX cache. Sharing each attention layer's ordinary weights and split-int8
repack across `Forward::reshape` reduced construction-time residency from
about 38.9 GiB to 35.5 GiB. Two apparent 14-minute startups were a stale
`target/release/bench_worker_decode` invoked directly after the change -- the
same stale-binary trap the previous section records. Rebuilding through Cargo
made the phase timings truthful.

An `nvprof` run over N=3 at 2K included prefill and therefore is not a clean
percentage profile, but call counts isolate the twelve decode steps (four
warmups plus eight timed): `moe_expert_down_bm1` plus
`moe_expert_down_narrow` consumed 40.51 ms, or about 3.38 ms per step. That is
16% of the measured 21.21 ms profiled step and is a material recoverable share.
This is the evidence for retrying the previously gated Q8-activation/dp4a
routed-down experiment symmetrically across N=1--3. It is not evidence that
the retry will win; correctness gates run before timing.

## 2026-08-18 — Symmetric Q8 activation still makes routed down width-dependent

The profiled routed-down share above justified rebuilding the old dp4a idea
after the upstream N=1/N=3 activations became bit-identical. The reattempt used
one opt-in activation quantizer and one contraction kernel for decode widths
one through three, with one fp32 scale per 32 activation values, signed codes
clamped to `[-127, 127]`, byte-packed reads from Q8_0's 34-byte weight blocks,
and the existing fp32 path retained as the default and fallback.

The narrow GPU gate caught two implementation defects before the result: the
first quantizer address omitted its lane, and fused N=1 dispatch does not
publish `bucket_live`. After both were fixed, the exact cross-width gate still
rejected the idea on GPU 2. With real layer-0 Q6_K gate/up and Q8_0 down
weights, identical hidden activations and routing, the N=1 and N=3 down outputs
were:

| comparison | max abs | cosine | max output magnitude | result |
| --- | ---: | ---: | ---: | --- |
| dp4a N=1 vs N=3 | `4.397443e-5` | `0.999980` | `6.7609e-3` | **reject: not exact** |

This is numerically inside `ROUTED_MMA_GATE`, but it fails the stricter gate
required for a discontinuous quantizer: identical inputs must produce exactly
the same codes, scales, and result at every serving width. No full decode or
performance benchmark was run after that failure. The implementation was
committed and then reverted; the tree remains on the fp32 routed-down path.
The larger Q6_K gate/up dp4a candidate has the same activation-quantization
risk plus more complicated weight arithmetic, so this result removes it from
the immediate queue rather than licensing a larger experiment.

## 2026-08-19 — Independent streams recover part of decode attention's N=3 serialization

`bench_attention` gained a diagnostic `LLMXABE_ATTN_CONCURRENT=1..3` mode that
launches one decode query on each of up to three CUDA streams. Three
interleaved one-query/three-query pairs on GPU 0 produced:

| context | one query ms | three concurrent ms | three sequential ms | concurrent saving |
| ---: | ---: | ---: | ---: | ---: |
| 32,768 | 0.304--0.305 | 0.746--0.753 | 0.912--0.915 | 17.4--18.5% |
| 65,536 | 0.510--0.512 | 1.292--1.308 | 1.530--1.536 | 14.5--15.9% |
| 131,072 | 0.920--0.926 | 2.403--2.442 | 2.760--2.778 | 12.1--13.5% |

The default diagnostic shares one read-only K/V allocation across streams. A
second three-run set with `LLMXABE_ATTN_SEPARATE_CACHE=1` used independent K/V
allocations and measured 0.747--0.750 ms at 32K, 1.303--1.311 ms at 65K, and
2.415--2.420 ms at 128K. The result is therefore generic concurrent issue and
bandwidth utilization, not an L2 reuse result from sharing a cache pointer.

Production batch decode still serializes the three per-sequence RoPE, append,
and attention operations. Across ten attention layers, the isolated result
puts the plausible 32K recovery near 1.5--1.7 ms per N=3 step. That is material
but cannot by itself close the full 24--30% aggregate decode gap. Production
multi-stream execution still needs exact N=1/N=3 and graph-capture gates before
this diagnostic can be called an engine improvement.

## 2026-08-19 — Batched attention now overlaps its three sequence-local reads

The production N=2/N=3 decode shapes now retain fixed auxiliary streams,
fork/join events, and one decode-partial scratch per sequence. At each Gated
Attention layer, the batched projections and norms remain on the main stream;
the per-sequence RoPE, cache append, and decode attention fork across streams,
then rejoin before the batched output gate and projection. All resources are
created by `enable_batch_decode`, outside capture and outside the hot path.

`LLMXABE_SERIAL_BATCH_ATTENTION=1` retains the prior serial loop as a
setup-time A/B control. Three interleaved serial/concurrent pairs on GPU 0,
using the captured `bench_decode_batch` path, produced:

| context | N | serial ms/step | concurrent ms/step | concurrent tok/s | step-time saving |
| ---: | ---: | --- | --- | --- | --- |
| 2,048 | 3 | 21.11, 21.04, 21.01 | 20.71, 20.74, 20.76 | 144.9, 144.6, 144.5 | 1.3--1.9% |
| 32,768 | 3 | 28.48, 28.25, 28.23 | 27.31, 27.21, 27.09 | 109.8, 110.3, 110.7 | 3.7--4.0% |

The isolated attention probe's larger 13--18% saving correctly identified a
real opportunity, but its 1.5--1.7 ms whole-step estimate was high: production
recovered 0.30--0.40 ms at 2K and 0.94--1.17 ms at 32K. The N=3 llama.cpp gap
therefore remains open; 110.3 tok/s at 32K is still well below this session's
153.79 tok/s best llama.cpp result.

The correctness gate ran before timing. Three launched N=3 steps and three
captured/replayed steps selected identical tokens. Against three independent
N=1 passes, every one of nine full 248,320-logit rows had max-abs difference
`0.000e0` and cosine `1.000000000`; identical prompts also remained bit exact.

## 2026-08-19 — Matching llama.cpp's outer prefill tile without its fragment algorithm is eight times slower

The next deep-prefill experiment implemented a separate, opt-in sm_75 kernel
with the outer geometry used by llama.cpp's live
`flash_attn_ext_f16<256,256,4,8>` specialization: four query positions across
eight GQA sibling heads and a 64-key K/V tile. The shipped 16-query kernel
remained the default and A/B fallback. The experiment deliberately used one
aliased K/V shared-memory tile and no register-resident next-tile prefetch, so
it tested whether the wider staging and lower barrier density could fit without
repeating the previous spill failure.

It fit, and that was not enough. `ptxas` for sm_75 reported 255 registers and
zero spill stores or loads (the shipped kernel uses 252). The release 128K
attention differential passed the unchanged `MMA_GATE`, with max-abs
`8.881e-5` and cosine `0.999999583`. A first narrow CUDA-event A/B on GPU 0,
8,192 query tokens, rejected it decisively:

| key offset | shipped ms | four-query/64-key ms | result |
| ---: | ---: | ---: | --- |
| 32,768 | 165.717 | 1,275.549 | 7.70x slower |
| 65,536 | 317.101 | 2,506.475 | 7.90x slower |
| 98,304 | 472.412 | 3,781.404 | 8.00x slower |
| 131,072 | 632.794 | 5,005.953 | 7.91x slower |

The failure is execution structure rather than spilling. Four queries launch
four times as many blocks as the shipped 16-query shape, while the conservative
64-key softmax serialized max/exp/sum through four lanes. llama.cpp makes this
outer geometry competitive with fragment-resident softmax and P-V combination;
copying only its tile dimensions does not copy that algorithm. The experiment
was removed. An honest retry must port the complete fragment layout and combine
scheme rather than incrementally tune this rejected local shape.

## 2026-08-19 — The required `-b 4096` prefill baseline, first complete single sweep

`llama-batched-bench` was run on an idle Quadro RTX 8000 with `-ngl 99 -sm
none -fa on -b 4096 -ctk f16 -ctv f16`, both required ubatches, and N=1/N=3.
These are one run per cell, not the three interleaved repetitions needed for a
final claim; they complete the missing shape matrix and identify which ubatch
must be challenged first. Contexts through 32K used `-c 131072`; 65K and 128K
used `-c 393216`, because llama.cpp divides `n_ctx` among the three parallel
prompts.

| context | N | `-ub 2048` tok/s | `-ub 4096` tok/s | faster setting |
| ---: | ---: | ---: | ---: | --- |
| 512 | 1 | 2,121.93 | 2,063.88 | 2048 |
| 512 | 3 | 3,057.48 | 3,035.79 | 2048 |
| 2,048 | 1 | 3,129.84 | 3,120.04 | 2048 |
| 2,048 | 3 | 3,212.35 | 3,342.41 | 4096 |
| 8,192 | 1 | 3,044.79 | 3,258.90 | 4096 |
| 8,192 | 3 | 3,046.28 | 3,293.83 | 4096 |
| 32,768 | 1 | 2,524.66 | 2,711.97 | 4096 |
| 32,768 | 3 | 2,488.64 | 2,679.02 | 4096 |
| 65,536 | 1 | 2,046.97 | 2,185.38 | 4096 |
| 65,536 | 3 | 2,009.79 | 2,150.27 | 4096 |
| 131,072 | 1 | 1,472.02 | 1,553.36 | 4096 |
| 131,072 | 3 | 1,477.11 | 1,554.21 | 4096 |

The first `-ub 2048` command incorrectly used `-c 131072` for 65K/128K
N=3 and failed at admission after completing the 32K cells. Those deep cells
were rerun with `-c 393216`; the failed harness configuration is not a result.
