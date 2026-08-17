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
