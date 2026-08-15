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
