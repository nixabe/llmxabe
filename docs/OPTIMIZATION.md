# Optimization plan

A ranked, arithmetic-backed plan for making `llmxabe` faster than the
`llama.cpp` baseline, derived from vLLM's source at `/home/nixabe/vllm`
(commit `d4801990a`, 2026-08-15), llama.cpp's source at
`/home/nixabe/llama.cpp` (commit `fd6863a69`, build 10456), and published
work.

**Nothing in this document was measured by its author** — all three GPUs
were in use by sibling workers. Every number here is either quoted from
[BENCHMARKS.md](BENCHMARKS.md)/[MODEL.md](MODEL.md), derived with the
arithmetic shown, or cited to a source. Where a claim could not be verified
it is marked **unverified**. The suggested experiments in
[§10](#10-suggested-experiments) are the things that would turn the remaining
derivations into measurements.

A detailed per-stage profile of llmxabe's forward pass landed in
[BENCHMARKS.md](BENCHMARKS.md#llmxabes-own-forward-pass-and-where-its-time-goes)
while this was being written. It was produced independently, it agrees with
§2's derivations to within 2% (§2.8), and it **changed this document's
ranking** — promoting the grouped-GEMM tiling work and demoting CUDA graph
capture to near-zero. Where the two differ, the measurement wins and the
difference is stated.

---

## 1. Where we are

> **Superseded in part.** [R3](#r3--tile-the-routed-expert-grouped-gemm), the
> grouped-GEMM tiling, has since landed and been measured. §1.0 is the
> current state; the tables that follow it are the pre-tiling baseline the
> rest of the document reasons from, kept because §2–§9's arithmetic is
> calibrated against them. Where a later section quotes 69.84 t/s or a 29.6×
> ratio, read it as "before R3".

### 1.0 Current state (2026-08-16, after R3 landed)

Release build, Quadro RTX 8000 sm_75, `Qwen3.6-35B-A3B-UD-Q6_K_XL`, 5 timed
repetitions after 2 discarded warmups, stream synchronized inside the timed
region:

| | llmxabe | llama.cpp | ratio | was |
| --- | ---: | ---: | ---: | ---: |
| `pp512` (prefill) | **200.87 ± 0.75 t/s** | 2,070.50 t/s | **10.3× slower** | 29.6× |

> **Superseded.** This table is the state at the tiling commit. Prefill is now
> **1,365.74 ± 6.60 t/s, 1.52× slower**, and decode **94.03 tok/s, 1.11×
> slower**. See "Integer tensor cores, wired end to end" in
> [BENCHMARKS.md](BENCHMARKS.md) for the twelve-step arc and where the time
> goes now. The rest of this section is kept because its *reasoning* — the
> roofline arithmetic and the structural fp32 ceiling — is what motivated the
> integer path, and that reasoning held.
| decode floor, n=1 | **55.07 ± 0.04 t/s** | 104.74 t/s | **1.90× slower** | 3.9× |
| fraction of fp32 peak at 512 | 13.3% | | | 4.6% |

Intermediate batch sizes, for the shape of the curve:

| tokens | ms/pass | tok/s | was |
| ---: | ---: | ---: | ---: |
| 1 | 18.16 ± 0.01 | 55.07 ± 0.04 | 26.80 |
| 19 | 123.01 ± 0.10 | 154.46 ± 0.12 | 37.16 |
| 128 | 665.18 ± 1.53 | 192.43 ± 0.44 | 60.53 |
| 512 | 2548.90 ± 9.50 | 200.87 ± 0.75 | 69.84 |

Peak VRAM 31.002 GiB of 47.27 GiB. The same argmax token 25358 (' Tokyo')
comes out at logit 19.936268 against 19.936270 before — a 2e-6 move
consistent with reassociation in the dot product and nothing else.

**What this does and does not close.** The prefill gap is now 10.3×, not
29.6×, and the n=1 floor is 1.90×, not 3.9×. Neither axis is won. The
remaining prefill distance is the tensor-core gap analysed in §7 plus the
~2× of unlocalized latency inside the tiled kernel itself (§1.0 note below);
the remaining decode distance is the missing KV cache and recurrent-state
carry (G007), which is a structural absence, not a tuning gap.

The tiled grouped GEMM reaches **13.3% of fp32 peak** at 512 tokens against
the 25% §2.7 targeted. That shortfall could not be attributed, because `ncu`
cannot read counters on this host (`ERR_NVGPUCTRPERM`); the analysis behind
it is `nsys`, an offline SASS dump, and ablation only. The SASS inner loop
is 511 instructions for 128 FFMA — 25% instruction density — which puts
roughly **50% of fp32 peak** as the structural ceiling for this instruction
mix, so 13.3% is about half of what this kernel shape can reach, not half of
what the card can.

### 1.1 Baseline the rest of this document reasons from

Measured on this host before R3 landed (release, Quadro RTX 8000 sm_75,
`Qwen3.6-35B-A3B-UD-Q6_K_XL`):

| | llmxabe | llama.cpp | ratio |
| --- | ---: | ---: | ---: |
| `pp512` (prefill) | 69.84 t/s | 2,070.50 t/s | **29.6× slower** |
| decode floor, n=1 | 26.80 t/s | 104.74 t/s | **3.9× slower** |
| fraction of the 235 t/s weight roofline | 11.4% | 44.5% | |

A second, more detailed measurement landed in
[BENCHMARKS.md](BENCHMARKS.md#llmxabes-own-forward-pass-and-where-its-time-goes)
while this document was being written, on GPU 2 on 2026-08-16 with CUDA-event
timing and a per-stage profile. Its figures are slightly different and
should be preferred, because it states its instrumentation overhead:

| | llmxabe | llama.cpp | ratio |
| --- | ---: | ---: | ---: |
| `pp512` | 73.48 ± 0.23 t/s | 2,051.30 ± 168.24 t/s | **27.9× slower** |
| single-pass latency floor, n=1 | 32.61 ms | 9.55 ms/token (`tg128`) | **3.4× slower** |
| fraction of bandwidth peak, n=1 | 13.9% | 47.4% | |

The two sets agree on everything that matters. **We currently lose on both
axes, by a lot.** The forward pass reproduces llama.cpp's token exactly, so
the remaining distance is entirely engineering — but it is a long distance.

Everything derived in §2 below was worked out independently of that profile
and then checked against it. The two agree closely (§2.8), which is the only
reason the derivations in this document should be trusted at all.

What exists and what does not:

| Capability | llmxabe | llama.cpp |
| --- | --- | --- |
| Correct forward pass | yes | yes |
| KV cache | **no** — `block/attention.rs:574-576` says "the attention window is this batch alone; carrying a cache across calls is G007's job" | yes |
| Batched decode across sequences | **no** — the batch dimension is tokens within one sequence | yes (`-np N`) |
| Continuous batching / chunked prefill | host-side only (`xabe-sched`, 18 tests, no device) | yes |
| Paged prefix cache | host-side only (`xabe-cache`, 22 tests, no device) | yes (per process) |
| CUDA graph capture | **no** — no `cuGraph*` symbol anywhere in `crates/xabe-cuda/src` | yes, active (`graphs reused = 1195` over 1,200 tokens) |
| Tensor cores | **no** — everything is scalar fp32 | yes (see [§7](#7-what-is-reachable-on-sm_75)) |
| MTP speculative decode | no | **yes** (`--spec-type`, `common/arg.cpp:4139`; `nextn` tensors load for `LLM_ARCH_QWEN35MOE`, `src/models/qwen35moe.cpp:137-142`) |

---

## 2. Method: the bandwidth model everything below is scored against

All rankings are derived from one model. It is stated here so every later
number can be checked.

### 2.1 Bytes per weight

Quantization rates, from the ggml block layouts:

| Format | Block | Bytes | Bits/weight | Bytes/param |
| --- | --- | ---: | ---: | ---: |
| `Q6_K` | 256 weights | 210 (`ql` 128 + `qh` 64 + `scales` 16 + `d` 2) | 6.5625 | 0.8203 |
| `Q8_0` | 32 weights | 34 (`d` 2 + `qs` 32) | 8.5 | 1.0625 |

### 2.2 Per-token weight traffic at batch 1

Using the per-tensor quantization [MODEL.md](MODEL.md) verified against the
real GGUF (expert `gate`/`up` are `Q6_K`, expert `down` is `Q8_0`, shared
expert and all projections and the LM head are `Q8_0`):

| Component | Arithmetic | MB/token |
| --- | --- | ---: |
| Routed experts | 8 × 40 × (2·2048·512·0.8203 + 512·2048·1.0625) = 8 × 40 × 2.834 MB | 906.9 |
| Shared expert | 40 × 3·2048·512·1.0625 | 133.7 |
| LM head | 248,320 × 2048 × 1.0625 | 540.3 |
| Projections | 1.305 B × 1.0625 | 1,386.6 |
| **Total weights** | | **2,967.5** |

> This is 3.8% above [MODEL.md](MODEL.md)'s 2.86 GB. The difference is the
> `Q8_0` expert `down` projections, which MODEL.md prices at a single `Q6_K`
> rate. MODEL.md already flags its own figure as "a floor, not a ceiling".

### 2.3 Per-sequence state traffic — the term the existing docs omit

A Gated DeltaNet decode step must **read and write** the full recurrent
state. It does not fit anywhere on chip (62.9 MB), so both directions hit
HBM:

| Component | Arithmetic | MB/token/sequence |
| --- | --- | ---: |
| GDN recurrent state (read + write) | 2 × 30 × 32 × 128 × 128 × 4 B | 125.8 |
| GDN conv state (read + write) | 2 × 30 × 96 KiB | 5.9 |
| **Total** | | **131.7** |
| Attention KV (read) | 20,480 B × depth | 0.0205 × depth |

**This term does not amortize across a batch.** Each sequence owns its own
state. It is the single most important structural fact for the concurrency
plan, and neither [MODEL.md](MODEL.md) nor [BENCHMARKS.md](BENCHMARKS.md)
counts it (see [§9](#9-contradictions-with-the-existing-docs)).

### 2.4 Corrected single-stream roofline

At batch 1, depth 0: 2,967.5 + 131.7 = **3,099.2 MB/token** →
672,000 MB/s ÷ 3,099.2 = **216.8 tok/s**, not the 235 tok/s in
MODEL.md. llama.cpp's 104.79 tok/s is **48.3%** of this.

### 2.5 Expert-activation density under batching

For 256 experts with top-8 routing and *uniform* routing, the expected
number of distinct experts touched per layer by a batch of N tokens is

```
D(N) = 256 × (1 − (1 − 8/256)^N)
```

Real routing is measurably **less** diverse than uniform, which helps us.
DynaExq (arXiv:2511.15015, Table 1, "Expert activation ratio (%) in decode
stage") measures Qwen3-Next-80B (512 experts, top-10) at **1.9%** activated
at batch 1 and **39.2%** at batch 32. The uniform model predicts 1.95% and
46.8%. So treat `D(N)` as an **upper bound** on traffic; observed routing
lands roughly 16% below it at moderate batch. The same table reports
**86.2%** activation for that model in *prefill* at batch 32 — prefill is
effectively dense.

### 2.6 The concurrency table

Weight bytes per step = 2,060.6 MB (LM head + projections + shared expert,
read once) + 113.377 MB × `D(N)` (routed experts, read once per distinct
expert per layer).

| N | `D(N)` | % of 256 | weights MB/step | weights MB/token | + state | **total MB/token** | roofline tok/s | at llama.cpp's 41% |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.00 | 3.1% | 2,967.5 | 2,967.5 | 131.7 | 3,099.2 | 216.8 | 88.9 |
| 2 | 15.75 | 6.2% | 3,846.3 | 1,923.2 | 131.7 | 2,054.9 | 327.0 | 134.1 |
| 3 | 23.26 | 9.1% | 4,697.8 | 1,565.9 | 131.7 | 1,697.6 | 395.9 | 162.3 |
| 4 | 30.53 | 11.9% | 5,522.0 | 1,380.5 | 131.7 | 1,512.2 | 444.4 | 182.2 |
| 8 | 57.42 | 22.4% | 8,570.7 | 1,071.3 | 131.7 | 1,203.0 | 558.6 | 229.0 |
| 16 | 101.95 | 39.8% | 13,618.9 | 851.2 | 131.7 | 982.9 | 683.7 | 280.3 |
| 32 | 163.30 | 63.8% | 20,575.1 | 643.0 | 131.7 | 774.7 | 867.4 | 355.6 |
| 64 | 222.44 | 86.9% | 27,280.0 | 426.3 | 131.7 | 558.0 | 1,204.3 | 493.8 |

### 2.7 The result that reframes the whole strategy

Compare the roofline column against llama.cpp's **measured** aggregate
throughput at `-np 3` ([BENCHMARKS.md](BENCHMARKS.md#concurrency)):

| Concurrency | llama.cpp aggregate | roofline | efficiency |
| ---: | ---: | ---: | ---: |
| 1 | 91.6 (82.96 with `-bs`) | 216.8 | 42.2% (38.3%) |
| 2 | 136.3 (129.54) | 327.0 | 41.7% (39.6%) |
| 3 | 162.4 (160.93) | 395.9 | 41.0% (40.7%) |

**llama.cpp holds a flat ~41% of the concurrency-aware roofline from c=1 to
c=3.** Its batching captures essentially the entire arithmetic benefit that
expert-activation density allows. The 1.77× it achieves at c=3 is not
"MoE batches badly" — 1.77× is *what the arithmetic permits*
(3,099.2 / 1,697.6 = 1.83, and it gets 97% of that).

This **contradicts the strategic hypothesis** as posed. There is no
"llama.cpp batches poorly" headroom to harvest. The two real openings are:

1. ~~**More slots.**~~ **MEASURED, AND THIS OPENING IS CLOSED.** E1 has been
   run. llama.cpp does not merely hold its efficiency as slots rise — it
   scales well past what the `-np 3` extrapolation predicted:

   ```
   llama-batched-bench -c 32768 -npp 128 -ntg 128 -npl 1,2,4,8,16 -ngl 99
   Quadro RTX 8000, build fd6863a69 (10456), same model file

   |  B | S_TG t/s | vs B=1 | aggregate t/s |
   |----|----------|--------|---------------|
   |  1 |   105.09 |  1.00x |        189.65 |
   |  2 |   159.63 |  1.52x |        287.68 |
   |  4 |   219.64 |  2.09x |        396.37 |
   |  8 |   262.08 |  2.49x |        464.34 |
   | 16 |   382.02 |  3.64x |        637.42 |
   ```

   **llama.cpp gains 3.64× on decode from 1 to 16 concurrent sequences.**
   The predicted "280 tok/s at N=16 if it holds 41%" was too pessimistic by
   36%: it actually reaches 382.02.

   So the bar for any concurrency claim is **382 tok/s decode at 16
   concurrent, per card** — not the 104.74 single-stream figure, and not the
   162.4 at `-np 3`. That is a 3.65× higher bar than the number this
   document was originally ranked against, and every projection in
   [§7](#7-aggregate-concurrency) must be read against it.

   This does not make concurrency worthless — llmxabe's three-worker
   architecture still multiplies whatever per-card figure it achieves, and
   so would three llama.cpp instances. It removes the *asymmetry*. There is
   no batching deficit in llama.cpp to exploit; there is only the kernel
   deficit, which is opening 2.
2. **Efficiency above 41%.** The KV path already reaches ~80%. Every point
   of weight-path efficiency is a point of throughput at every concurrency
   simultaneously. This is the kernel program.

### 2.8 Cross-check against the measured profile

`profile_forward` sums the GGUF tensor directory to get necessary traffic
per stage and divides by CUDA-event time
([BENCHMARKS.md](BENCHMARKS.md#llmxabes-own-forward-pass-and-where-its-time-goes)).
It was produced independently of §2.2–§2.4. The two agree:

| Quantity | derived here | measured | agreement |
| --- | ---: | ---: | --- |
| Projections, MB/token | 1,386.6 | 1,379.2 (1,089.5 GDN + 289.8 attention) | 0.5% |
| MoE, MB/token | 1,040.6 | 1,125.3 | 8.1% — see below |
| LM head, MB/token | 540.3 | 540.3 | **exact** |
| Total necessary bytes | 3,099.2 | 3,045 + 62.9 GDN state = **3,108** | 0.3% |
| Perfect-implementation ceiling | 216.8 tok/s | 221 tok/s (their 4.53 ms) | 1.9% |
| llama.cpp fraction of peak, n=1 | 48.3% | 47.4% | 0.9% |
| llama.cpp prefill, % of fp32 peak | 61.9% | 61.2% | 1.1% |

Two residuals, both stated rather than smoothed over:

- The 1.9% gap on the ceiling is the GDN recurrent-state **read**, which the
  profile's byte table omits (it counts the reset *write* under a separate
  stage). Once R1 lands and the state is read rather than memset,
  **216.8 tok/s** is the number.
- The 84.7 MB gap on the MoE row is **not fully accounted for**. The router
  weights explain about 22 MB (40 × 2048 × 256 × 1.0625 B); the rest is
  presumably the gate/norm tensors my §2.2 does not enumerate. The profile's
  figure comes from summing the actual GGUF directory and should be
  preferred; my §2.6 concurrency table is therefore **slightly optimistic**,
  by at most 84.7 MB on the once-per-step term — 2.7% at N=1, 0.3% at N=32.

### 2.9 The measured breakdown, and what it does to the ranking

This is the most decision-relevant table in the project, and it is measured,
not derived:

| Stage | achieved GB/s at n=1 | % of 672 GB/s |
| --- | ---: | ---: |
| **MoE ×40** | **43.51** | **6.47%** |
| GDN projections ×30 | 229.68 | 34.18% |
| Gated Attention projections ×10 | 324.30 | 48.26% |
| **LM head** | **597.93** | **88.98%** |

**Our own LM head reaches 89% of streaming peak on this card.** That single
number retires an entire class of doubt: the hardware, the driver, the NVRTC
toolchain and the fp32 arithmetic are all capable of saturating HBM. The MoE
at 6.47% is a kernel-shape problem and nothing else.

The cause is identified and quantified in that profile: `moe_expert_ffn` and
`moe_expert_down` stage nothing in shared memory and reuse nothing across
tokens, so an expert's weights are re-fetched once per assigned token instead
of once per tile — a **15.9× redundant weight read at n=512** — and at n=1 the
fixed-capacity grid runs 128 slots for 8 live pairs, a **93.75% early-out**.
Those two kernels are **67.4% of the pass at n=512 and 76.8% at n=1**.

Three consequences for this document's ranking, all of which I applied:

1. **R3 is not "make the memory path faster" in the abstract.** It is one
   specific change — stage a weight tile in shared memory and reuse it across
   a tile of tokens — and it is worth a measured **74.2% at n=1** and
   **63.8% at n=512**.
2. **CUDA graph capture (R5) is demoted to near-worthless for now.** Measured
   exposed launch overhead is ~0.4% at n=512 and *not distinguishable from
   zero* at n=1: 1,053 launches per pass, and the uninstrumented wall clock
   (32.26 ms) sits *below* the instrumented GPU total (32.61 ms). The
   10–20% figure I cited from secondary sources is **refuted for our current
   shapes**. It becomes relevant only after the kernels get roughly 10×
   faster.
3. **Attention is not where our time goes.** `attn_flash_causal` is 0.33% of
   the pass at n=512 and 0.08% at n=1. Tensor-core attention — the thing
   [KERNELS.md](KERNELS.md) flags most prominently as missing — is worth at
   most 0.33%. (Caveat from that profile, which I endorse: this is a 512-token
   window with no KV cache, and the conclusion would flip at 32K.)

---

## 3. What vLLM actually does, and what of it transfers

### (a) Continuous batching and the step loop

`vllm/v1/core/sched/scheduler.py:476-1290`.

The design note at `scheduler.py:478-487` is the whole idea:

> "There's no 'decoding phase' nor 'prefill phase' in the scheduler. Each
> request just has the `num_computed_tokens` and `num_tokens_with_spec` […]
> At each step, the scheduler tries to assign tokens to the requests so that
> each request's `num_computed_tokens` can catch up its
> `num_tokens_with_spec`."

The loop:

1. Running requests first, unconditionally, until the token budget is spent
   (`scheduler.py:525-700`). Each gets
   `num_tokens_with_spec − num_computed_tokens` tokens, clipped by
   `token_budget`, by `long_prefill_token_threshold`
   (`scheduler.py:563-564`), and by `max_model_len` (`:571-576`).
2. Then the waiting queue, chunking whatever does not fit
   (`scheduler.py:751-1133`). If `enable_chunked_prefill` is false, a
   request larger than the remaining budget stops the loop instead of being
   chunked (`:959-966`).
3. Preemption removes a request from the *running* list and gives its tokens
   back to the budget (`scheduler.py:658-662`).

`xabe-sched` already implements this shape, and its rules 3 and 4
([SCHEDULER.md](SCHEDULER.md)) are correct and stricter than vLLM's. Two
things vLLM does that `xabe-sched` does not:

- **`_mamba_block_aligned_split`** (`scheduler.py:366-406`). A prefill chunk
  must *end* where recurrent state can be checkpointed, because "slot p
  holds the state after exactly (p+1) × block_size tokens. State is written
  at chunk ends, so chunk ends must be block aligned" (`:401-405`). This is
  the scheduler-side half of `xabe-cache`'s exact-match rule, and
  `xabe-sched` has no equivalent. **Without it, a chunked prefill of a GDN
  layer produces a state at a position that can never be reused.**
- **Padding decode batches to a uniform shape to preserve full CUDA graphs**
  (`scheduler.py:936-950`): when spec decode is on, a 1-token decode is
  padded to `1 + num_spec_tokens` so the step stays graph-capturable, and
  the scheduler would rather *not schedule* than schedule unpadded
  (`:944-949`). Scheduling policy is subordinated to graph capture.

Quantitatively, chunked prefill is well-supported in the literature:
Sarathi-Serve (OSDI '24) reports up to **2.6× serving capacity within
latency SLO** for Mistral-7B on one A100 and up to 6.9× for Falcon-180B on
8×A100 over Orca and vLLM. A production report puts plain chunked prefill at
**+50% total token throughput** for evenly-sized requests. Both are
latency-SLO-shaped wins; neither changes the raw arithmetic in §2.6.

**Reachable on sm_75: entirely.** This is host-side policy.

### (b) Paged KV cache and block management

`vllm/v1/core/kv_cache_manager.py`, `block_pool.py`, `kv_cache_utils.py`.

What `xabe-cache` already matches:

- Block-hash chaining so identical prefixes converge regardless of who
  inserted them (`block_pool.py:225-300`, `cache_full_blocks`).
- Reference counting and eviction of unreferenced blocks.
- A retention interval for recurrent-state checkpoints. vLLM's is
  `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` (`kv_cache_coordinator.py:154`),
  validated to be a multiple of the scheduler block size
  (`kv_cache_coordinator.py:56-60`) — exactly `xabe-cache`'s rule that `R`
  must be a multiple of the attention block size. vLLM rejects the setting
  outright if the model has no sliding-window or Mamba group
  (`kv_cache_coordinator.py:43-54`). **vLLM's default is `None` = dense
  checkpointing**; `xabe-cache` defaults `R` to 2048 tokens.

Where `xabe-cache` is **behind**:

- `FreeKVCacheBlockQueue` is an intrusive doubly-linked free list giving O(1)
  removal from the middle (`kv_cache_utils.py:185-269`). `xabe-cache`'s
  `evict_unreferenced` is an O(n) scan, which [CACHE.md](CACHE.md) already
  states.
- vLLM caches *partial* blocks too (`block_pool.py:445`,
  `cache_partial_block`), so a hit does not have to land on a block
  boundary.
- vLLM pins the "shared prefix junction" so the retention interval cannot
  drop the one boundary that enables cross-request reuse
  (`kv_cache_manager.py:238-247`, described as "Marconi-style APC").

Where `xabe-cache` is **ahead, and vLLM's own source now proves it**:

[CACHE.md](CACHE.md) rule 1 says never unify page geometry across cache
groups. vLLM's `_get_kv_cache_groups_uniform_page_size` still *requires*
uniform page size — assumption 1 at `kv_cache_utils.py:1136-1138`:
"Physical memory per block: Must be the same across all KV cache groups.
Breaking this assumption is non-trivial due to memory fragmentation
concerns."

But vLLM does **not** achieve it by padding attention pages up to the Mamba
page. It raises the attention **block size in tokens** until the attention
page is at least as large as the Mamba page, then pads the Mamba page the
last few percent to match exactly
(`vllm/platforms/interface.py:888-938`):

```
attn_tokens_per_mamba_state = cdiv(mamba_page_size, attn_page_size_1_token)
attn_block_size = chunk_size * cdiv(attn_tokens_per_mamba_state, chunk_size)
```

For our geometry: `attn_page_size_1_token` = 2 × 2 kv-heads × 256 head-dim ×
2 B = 2,048 B per layer; `mamba_page_size` = 32 × 128 × 128 × 4 B =
2,097,152 B per layer. So
`attn_tokens_per_mamba_state` = 1,024, and with `chunk_size` = 64 the
attention block size lands at **1,024 tokens** — four times
`xabe-cache`'s 256, with one recurrent checkpoint per attention block
(`R` = block size, i.e. `mamba_cache_mode = "align"`,
`vllm/platforms/interface.py:915-916`).

So the two designs make the *same* trade differently:

| | `xabe-cache` | vLLM |
| --- | --- | --- |
| Attention block | 256 tokens | ~1,024 tokens (derived) |
| Checkpoint interval `R` | 2,048 tokens (8 blocks) | = block size (1 block) |
| Snapshot : KV ratio over one `R` | 60 MiB : 40 MiB = 1.5 | 60 MiB : 20 MiB = 3.0 |
| Prefix-hit granularity | 256 tokens for attention, 2,048 for GDN | 1,024 for both |
| Page geometry | independent per group | uniform, enforced |

**`xabe-cache`'s geometry is genuinely better on both axes** — finer
attention hit granularity *and* a lower snapshot ratio — and it is better
precisely because it refuses the uniform-page constraint. That is a real
architectural advantage and it is not speculative; it is visible in vLLM's
source as an acknowledged limitation.

Primary source for the whole problem class: **Marconi: Prefix Caching for
the Era of Hybrid LLMs** (MLSys '25, Outstanding Paper Honorable Mention,
arXiv:2411.19379), which is where the exact-match constraint on recurrent
state and reuse-forecasting admission policies come from; it reports up to
**34.4× higher token hit rate** and **71.1% lower TTFT** against prior
prefix-caching systems.

**Reachable on sm_75: entirely.**

### (c) The fused MoE path

`vllm/model_executor/layers/fused_moe/fused_moe.py`,
`moe_align_block_size.py`, `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu`.

**The `moe_align_block_size` kernel.**
`moe_align_sum_kernels.cu:100-193` is a single-block kernel: per-warp shared
counters incremented by `atomicAdd` (`:144-153`), a `cub::BlockScan`
exclusive sum over the per-expert counts rounded up to `block_size`
(`:157-174`), then each thread writes the `expert_ids` runs for its own
expert (`:182-188`) and fills the tail with a sentinel (`:190-194`). A
second thread block (selected by `blockIdx.x % 2`, `:120-127`) fills
`sorted_token_ids` with the out-of-range sentinel. `num_tokens_post_pad`
lands in **device memory** (`:176-178`) and the consuming GEMM early-exits
on it (`fused_moe.py:405-407`).

That is exactly `llmxabe`'s design rule 5, independently arrived at, and it
confirms `xabe-cuda`'s `moe_align_block_size` equivalent (single-block,
tables exact vs reference, per [KERNELS.md](KERNELS.md)) is the right shape.

**The decisive finding for our MoE worker: vLLM skips the sort entirely at
decode shapes.** `_prepare_expert_assignment` (`fused_moe.py:1556-1576`):

```python
naive_block_assignment = (
    expert_map is None
    and num_tokens * top_k_num * 4 <= global_num_experts
    and ...
)
```

For our geometry (`global_num_experts` = 256, `top_k` = 8) that condition is
`N × 32 ≤ 256`, i.e. **N ≤ 8**. Below that, `sorted_token_ids` is `None`,
`moe_align_block_size` is not called at all, and the GEMM kernel gives
**each (token, expert) pair its own block** (`fused_moe.py:408-416`):

```python
offs_token = tl.where(offs == 0, pid_m, num_valid_tokens)
```

— row 0 of the M tile is the one real token, every other row is masked out.

**Why this matters, with arithmetic.** At decode concurrency the routed
experts are almost empty. At N=32 there are 256 (token, expert) pairs spread
over `D(32)` = 163.3 distinct experts — **1.57 tokens per expert**. If you
pad each expert's run to vLLM's small-batch `BLOCK_SIZE_M` of 16
(`fused_moe.py:1371-1372`), the padded M total is 163 × 16 = 2,608 rows
against 256 real ones: a **10.2× compute inflation**. Check that against
the compute budget:

- FLOPs at N=32 = 32 × 5.89 GFLOP = 188.5 GFLOP → 11.6 ms at fp32 peak
  (16.31 TFLOP/s).
- Memory at N=32 = 24.79 GB → 36.9 ms at 672 GB/s.
- Headroom = 3.2×. A 10.2× inflation **overruns it** and turns a
  bandwidth-bound decode into a compute-bound one.

So: **the M tile must be 1 or 2 at decode, not 16.** At `M_block` = 2 the
inflation is 326/256 = 1.27× → 14.7 ms compute against 36.9 ms memory, still
comfortably bandwidth-bound. Padding costs *compute*, not bandwidth (the
expert's weight tile is read once per block regardless of how many M rows
are real), which is exactly why the trade flips on a machine with only
16.3 TFLOP/s of fp32.

**Tiling heuristics worth copying** (`fused_moe.py:1366-1412`):

| Knob | vLLM's rule | Value for our decode (N ≤ 32) |
| --- | --- | --- |
| `BLOCK_SIZE_M` | 16 if M ≤ 32, 32 if ≤ 96, 64 if ≤ 512, else 128 | override to 1–2, per the arithmetic above |
| `BLOCK_SIZE_N` | 64 if M ≤ 64 else 128 | 64 |
| `BLOCK_SIZE_K` | 128 if M ≤ 64 else 64 — "small batches benefit from longer reduction" | 128 |
| `GROUP_SIZE_M` | `16 if M // E > 128 else 1` (`:1390-1391`) — "with many experts each one sees few tokens so grouping is useless" | **1** |
| `num_warps` | 4 if M ≤ 128 else 8 | 4 |

Note `GROUP_SIZE_M` = 1 for us: the L2-locality grouped launch order that
`fused_moe_kernel` implements (`fused_moe.py:386-396`) **buys nothing at
decode**, because `M // E` = 0. It matters only at prefill.
[KERNELS.md](KERNELS.md#turing-adjustments) currently recommends grouped
launch order for decode; that is the wrong regime for it.

**SplitK.** [KERNELS.md](KERNELS.md) cites "roughly 20% from SplitK alone"
on vLLM's Mixtral kernel. The primary source is PyTorch's *Accelerating MoE
model inference with Locality-Aware Kernel Design*, which reports ~18–20%
from SplitK over the data-parallel decomposition and then addresses a low L2
hit rate with grouped launch order; the SplitK decomposition itself is
described in arXiv:2402.00025. **Note this vLLM tree still carries
`SPLIT_K` as a kernel parameter (`fused_moe.py:343`) but every default
config sets `"SPLIT_K": 1`** (`fused_moe.py:1308, 1344, 1358-1365, 1409`).
Treat the 20% as a claim about a specific 8-expert model on Ampere, not a
promise for a 256-expert model on Turing.

**Where llama.cpp draws the GEMV/GEMM line, which is the number that
actually governs our design.** `ggml_cuda_should_use_mmvq`
(`ggml/src/ggml-cuda/mmvq.cu:282-330`) returns
`ne11 <= MMVQ_MAX_BATCH_SIZE` on all NVIDIA parts, and
`MMVQ_MAX_BATCH_SIZE` is **8** (`mmvq.cuh:3`). So llama.cpp uses the
`dp4a`-based quantized GEMV up to 8 columns and switches to the
tensor-core MMQ kernel at 9. Applied to us:

- Routed experts see ~1.6 columns even at N=32 → **permanently in GEMV
  territory**. Tensor cores are irrelevant there.
- Projections, shared expert and LM head see the full N → cross into MMQ
  territory at **N ≥ 9**.

That is a clean split for the MoE worker: build a bandwidth-optimal gather
GEMV for the routed path, and worry about tensor cores only for the dense
path and prefill.

**What of vLLM's MoE is *not* reachable:** the Triton tensor-descriptor
gather path (`USE_TD`, `fused_moe.py:446-461`) needs TMA; the fp8 blockwise
configs (`:1314-1347`) need Ada+; `moe_align_sum_kernels.cu:508`'s
`__reduce_add_sync` needs sm_80 — Marlin's sm_75 branch replaces it with a
`__shfl_down_sync` ladder (`marlin_template.h:493-508`), which is the
pattern to copy.

### (d) CUDA graph capture

`vllm/config/compilation.py`.

vLLM has five modes (`compilation.py:53-63`): `NONE`, `PIECEWISE`, `FULL`,
`FULL_DECODE_ONLY`, `FULL_AND_PIECEWISE`. The distinction that matters is
between a **uniform decode batch** (every request contributes the same
number of query tokens) and a mixed prefill/decode batch. Only the former
gets a full graph by default; mixed batches fall back to piecewise, in which
attention stays eager and everything between attention calls is captured.

Bucketing (`compilation.py:698-705`, `vllm/config/vllm.py:1888-1891`):

```
[1, 2, 4] + list(range(8, 256, 8)) + list(range(256, max_size + 1, 16))
```

capped at 512 by default. Each captured size can only serve that size, so
runtime pads up to the next bucket.

Graph capture is subordinated to hardware capability, not the reverse:
`compilation.py:1388-1473` *downgrades* the mode when the attention backend
cannot support graphs for mixed batches, and `:1499-1510` refuses to start
if a Mamba model has fewer state blocks than the largest capture size.

**Expected win.** Published estimates put CUDA graphs at **10–20%** on
small-batch decode, with launch overhead 20–40% of end-to-end time at batch
1 and each replay saving 50–100 µs. **These are secondary-source numbers and
I could not verify them against a primary benchmark** — vLLM's own design
doc gives no figure. What *is* verified is that llama.cpp already replays a
graph on essentially every decode step (`graphs reused = 1195` over 1,200
tokens, [BENCHMARKS.md](BENCHMARKS.md)), so **graph capture is the bar, not
the advantage**, as [ARCHITECTURE.md](ARCHITECTURE.md) already concedes.

For us the naive launch count is worse than llama.cpp's: 1,080 MoE GEMVs per
token before fusion. If a launch costs ~5 µs of host time and the step
budget at 100 tok/s is 10 ms, 1,080 launches is 5.4 ms — over half the
budget. That is an argument for fusion *and* capture, not capture alone.

**Reachable on sm_75: yes.** CUDA graphs are a driver feature with no
architecture floor.

### (e) Gated DeltaNet and hybrid handling

`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`,
`vllm/model_executor/layers/mamba/ops/`.

Structure worth copying:

- The forward core splits on `attn_metadata.num_prefills` vs
  `num_decodes` and runs *different kernels* for each
  (`qwen_gdn_linear_attn.py:1348-1372`, `:1385-1541`), with a fast path when
  the batch is pure decode (`:1269-1280`, `_forward_core_decode_non_spec`).
  A mixed batch runs both and stitches. `xabe-engine` will need the same
  split; the chunked and recurrent GDN kernels already exist
  ([KERNELS.md](KERNELS.md)).
- Recurrent state is indexed by a **device-side index tensor**
  (`non_spec_state_indices_tensor`, `:1291`), not by a host-computed
  pointer — the same discipline as `llmxabe`'s design rule 5, and for the
  same reason.

**The batching property that matters.** Recurrent state is per-sequence and
constant-size. At batch N the GDN step reads and writes N × 62.9 MB while
the weights are read once. From §2.6:

| N | state share of per-token bytes |
| ---: | ---: |
| 1 | 131.7 / 3,099.2 = **4.2%** |
| 8 | 131.7 / 1,203.0 = **10.9%** |
| 32 | 131.7 / 774.7 = **17.0%** |
| 64 | 131.7 / 558.0 = **23.6%** |

The share grows monotonically because the numerator is fixed and the
denominator falls. Routed-expert weights stay the larger term until about
**N ≈ 220** — at N=128 they contribute 28,526/128 = 223 MB/token against
131.7 MB of state; at N=256, 113 MB against 131.7 — but that crossover is
far beyond what 47.3 GiB of VRAM allows (§6). What matters at the
concurrency we can actually reach is that state is **17% of per-token
bytes at N=32 and rising**, and no amount of expert-batching touches it.

vLLM has two named mitigations:

- **`mamba_ssm_cache_dtype`** (`vllm/config/cache.py:134-136`): store the
  SSM state at a narrower dtype than the model. fp32 → fp16 halves 125.8 MB
  to 62.9 MB — worth 2.1% at N=1 and **8.4% at N=32**.
- **`use_replayssm`** (`vllm/config/cache.py:151-158`): "cache recent SSM
  inputs and skip the per-step full-state store, writing the checkpoint back
  only on flush", with a ring buffer of length B = 16 by default. This
  removes the *store* half of the traffic on 15 of every 16 steps —
  62.9 MB × 15/16 ≈ 59 MB/token/sequence, i.e. **7.6% at N=32**. It is
  restricted to non-speculative decode and the Triton Mamba backend. The
  implementation is Mamba2-specific (`ops/replayssm_config.py`,
  `selective_state_update_replayssm_output_only.py`); **whether the same
  trick is sound for the delta rule is unverified.**

**Reachable on sm_75:** the algorithmic structure, yes. The specific vLLM
kernels are Triton and several are Blackwell-tuned
(`replayssm_config.py:47-52`); the CUTE-DSL chunked path
(`ops/gdn_chunk_cutedsl/`) is Hopper+. Use llama.cpp's
`ggml/src/ggml-cuda/gated_delta_net.cu`, which is Turing-validated, as
[KERNELS.md](KERNELS.md) already directs.

---

## 4. Ranked plan

Ranked by expected win per unit of risk. "Risk" folds in build cost,
dependency depth, and the chance the win fails to materialize.

Every item is marked **sm_75: yes / no / partial**.

**Read the ordering as two lists, not one.** R1 and R2 are *prerequisites*:
without them there is no decode number and no concurrency to measure. R3 is
the largest *measured* win in the project and is the one to build first if
throughput is the goal — it needs neither of the others and it can be
measured today. R5 is included for completeness and has been measured at
approximately zero (§2.9); it is ranked where it is because it becomes real
after R3, not because it is worth doing now.

---

### R1 — KV cache and persistent recurrent state — **DONE (2026-08-16)**

Landed as `block::attention::KvCache` plus `state::SequenceState`, gated by
`tests/decode.rs`. What it is **not**: `xabe-cache`'s two-group pager, which
is still host-side only. This is one contiguous cache for one sequence, which
is what makes a single-stream decode number measurable; the pager is what
makes three concurrent sequences work, and that is R2's and milestone 07's.

**Measured.** 65.25 tok/s at a 128-token context against llama.cpp's `tg128`
104.72 — **1.61× slower**. The prediction below that this "is not an
optimization, it is the feature that makes every other decode number
meaningful" held, and it corrected a number in the other direction than
expected: the cold `n = 1` pass had been used as a decode proxy at 18.16 ms,
and a real decode step is **15.33 ms**, because it does not re-zero 30
recurrent states. The proxy was pessimistic, not flattering.

**What it exposed.** Decode cost grows +27.2% from context 132 to 2,132, and
the implied bandwidth on the KV read is 19.6 GB/s — 2.9% of peak. The cause
is that the flash kernel's grid is `(n_query, q_heads)`, so `n_query = 1`
launches 16 blocks on a 72-SM card. That is a new item, R10 below, and it is
now the largest identified win on the decode path.

The entry as it was written before the work, kept because its reasoning is
what the measurement above is scored against:

**sm_75: yes.** No architecture dependency.

**What.** Carry attention KV and GDN recurrent state across forward calls
instead of recomputing the window. `block/attention.rs:574-576` currently
documents the absence.

**Mechanism.** Without it, decoding token *t* costs a prefill of *t* tokens.
Decode at depth 4K costs ~4,000× a real decode step.

**Expected win.** Unbounded as a function of depth; **zero at depth 0**,
which is why the measured 26.80 tok/s decode floor does not show it. It is
not an optimization, it is the feature that makes every other decode number
meaningful.

**Cost.** Wire `xabe-cache`'s `BlockPool` to device allocations, add a block
table to the attention kernel, add state indices to the GDN kernels. Both
crates are tested host-side; neither has touched a device
([CACHE.md](CACHE.md), [SCHEDULER.md](SCHEDULER.md) both say so).

**Depends on.** Nothing.

---

### R2 — Batched decode across sequences

**sm_75: yes.** From §5 below, decode stays bandwidth-bound in pure fp32 up
to about batch 256, so no tensor cores are needed anywhere in this item.

**What.** One forward pass over N sequences: an M dimension in the MoE
GEMM, a per-sequence state index in GDN, a per-sequence block table in
attention.

**Mechanism.** Weights are shared across the batch; state and KV are not.
§2.6.

**Expected win.** Per-token weight bytes fall 2,967.5 → 1,071.3 (N=8) →
643.0 (N=32), a **2.77×** and **4.61×** reduction. Including the
non-amortizing state term, total per-token bytes fall **2.58×** (N=8) and
**4.00×** (N=32). At constant kernel efficiency that is the aggregate
throughput multiplier.

**Critical implementation constraint, with arithmetic.** The routed-expert
kernel must tile over M with `BLOCK_SIZE_M` of 1–2, not vLLM's small-batch
default of 16. At N=32, `D(32)` = 163.3 distinct experts hold 256 rows —
1.57 per expert. `BLOCK_SIZE_M` = 16 inflates compute 10.2× against only
3.2× of compute headroom and makes decode compute-bound. See §3(c).

**Second constraint.** Dequantize each weight tile **once per tile**, not
once per token. Unpacking 22.6 G params at N=32 at ~4 ops each is 5.5 ms
against 36.9 ms of memory time — fine. Redoing it per token is 176 ms —
fatal.

**Cost.** High. Touches every kernel.

**Depends on.** R1.

---

### R3 — Tile the routed-expert grouped GEMM

> **DONE (2026-08-16).** Landed and measured; see §1.0 for the end-to-end
> figures. What follows is the pre-landing analysis, kept because the
> outcome checks it. **Where the prediction held:** this was indeed the
> largest single item, and tiling was indeed the mechanism — ablating all
> weight loads and dequant entirely bought only 6%, which is what proved the
> cost was reuse rather than bandwidth or dequant ALU. **Where it did not:**
> the predicted "74.2% of the n=1 pass" recovery assumed 25% of fp32 peak;
> the landed kernel reaches 13.3% at 512 tokens, so the n=1 pass went 32.61
> → 18.16 ms (44.3% recovered, not 74.2%) and 55.07 tok/s, not the 207
> tok/s projected here. The projection was optimistic by roughly the ratio
> of the peak fractions, which is the honest reading: the model of *what*
> was slow was right, the model of *how fast the fix would be* was not.
>
> Two things the analysis below did not anticipate. The **shared expert**
> had the same defect and was 9.9× on its own — it is not mentioned anywhere
> in this section. And **grid over-provisioning turned out to be a minor
> term**: the dispatch parallelization it motivated was worth 0.12% of
> runtime when finally measured. The redundant-read half of the diagnosis
> carried essentially all of the win.

**This should probably be done first**, before R1 and R2. It is ranked third
only because R1 and R2 are prerequisites for *measuring* concurrency at all.
By measured headroom it is the largest single item in the project, and unlike
R1 and R2 it can be built and measured today.

**sm_75: yes.** Decode is a memory-access problem, not a compute problem
(§5.1). Tensor cores contribute nothing here.

**What.** `moe_expert_ffn` and `moe_expert_down` currently stage nothing in
shared memory and reuse nothing across tokens. Stage a tile of the expert's
weight matrix in shared memory and reuse it across a tile of tokens; pack the
grid to live slots instead of the fixed `sorted_capacity`.

**Mechanism, measured.** Two effects, both quantified in
[BENCHMARKS.md](BENCHMARKS.md#llmxabes-own-forward-pass-and-where-its-time-goes):

- **Redundant weight reads.** An expert's matrices are re-fetched once per
  assigned token instead of once per tile — a **15.9× redundant read** at
  n=512. How much L2 absorbs is unmeasurable on this host (`ncu` fails with
  `ERR_NVGPUCTRPERM`), so true DRAM traffic is bounded between 29.2 GB and
  282 GB per pass.
- **Grid over-provisioning.** `grid.y` is the compile-time `sorted_capacity`
  that design rule 5 requires for graph replay. At n=1 that is 128 slots for
  8 live `(token, expert)` pairs — **93.75% early-out**; at n=512, 7,936 for
  4,096 — 48.4%.

The isolating comparison is `moe_expert_down` (q8_0, 512-long reduction, 2
MACs/thread) taking **74% as long as** `moe_expert_ffn` (q6_K, 2,048-long,
8 MACs/thread) while doing **half** the arithmetic — 1.48× less efficient per
FLOP with the *cheaper* quant format. The difference is reduction shape, not
dequantization.

**Expected win, measured.** Recovers **74.2% of the n=1 pass** and **63.8% of
the n=512 pass** (realistic, at 25% of fp32 peak). The MoE currently runs at
**6.47% of bandwidth peak** while our own LM head runs at **88.98%** — the
capability is demonstrated on this exact hardware by this exact codebase.

Fixing R3 plus the GDN and attention projections takes the n=1 pass to ~4.8 ms
= **207 tok/s**, ahead of llama.cpp's 104.72. Treat that with suspicion until
R1 exists — that pass has no cache to read — though §2.3's arithmetic says the
recurrent-state read only costs it 62.9 MB, i.e. ~2%, so the conclusion
survives.

**Cost.** High, but bounded and local: two kernels.

**Depends on.** Nothing. Its *concurrency* value depends on R2.

**Corollary the profile makes unavoidable:** items worth under 3% each —
tensor-core attention, dispatch parallelism, dequant micro-optimization, graph
capture, elementwise fusion — should not be touched before this lands.

> **The corollary has expired, and its percentages with it.** Now that R3 has
> landed, every one of those items is a larger share of a smaller pass and
> must be re-profiled before being ranked. Dispatch parallelism was done
> anyway (it was cheap) and measured at 0.12%; the rest are unmeasured
> against the new profile.

---

### R4 — Chunked prefill mixed with decode, wired to the device

**sm_75: yes.**

**What.** Connect `xabe-sched`'s `step()` to a real batch. Add
`_mamba_block_aligned_split` (§3(a)) — currently missing.

**Mechanism.** Prefill is compute-bound, decode is memory-bound; mixing them
uses both. Prefill is 20–26× decode throughput on this model
([BENCHMARKS.md](BENCHMARKS.md)).

**Expected win.** Sarathi-Serve reports up to 2.6× serving capacity within
SLO (Mistral-7B, one A100); a production deployment reports +50% total token
throughput. Both are latency-SLO-shaped. **On raw aggregate decode
throughput it adds nothing beyond R2** — it changes when prefill is allowed
to interrupt decode, not how many bytes a step reads.

**Cost.** Low; the policy is written and tested. The missing piece is the
Mamba-aligned split, ~50 lines.

**Depends on.** R1, R2.

---

### R5 — CUDA graph capture of the decode step

**sm_75: yes.**

**What.** Capture the fixed-topology decode step. The MoE indirection tables
are already built on-device into fixed-size buffers gated by a
`valid_tokens` device scalar, precisely so this is possible
([KERNELS.md](KERNELS.md)); no capture has been performed.

**Mechanism.** Removes host launch latency from the critical path.

**Expected win: measured at approximately zero, today.**
[BENCHMARKS.md](BENCHMARKS.md#llmxabes-own-forward-pass-and-where-its-time-goes)
measures exposed launch overhead at **~0.4% at n=512** and **not
distinguishable from zero at n=1** — the n=1 pass issues 1,053 launches and
the uninstrumented wall clock (32.26 ± 0.01 ms) sits *below* the instrumented
CUDA-event GPU total (32.61 ms). The reason is unflattering: our kernels are
slow enough to hide their own launches.

The 10–20% figure that circulates for vLLM (secondary sources, and I could
not trace it to a primary benchmark) does not transfer to our shapes. **It
becomes relevant only after R3 makes the kernels roughly 10× faster**, at
which point it must be re-measured rather than assumed.

> **That condition is now partly met.** R3 landed at 9.3× on the MoE path
> and 2.05× on the n=1 pass overall (§1.0). Launch count per pass is
> unchanged at 1,053, so the same ~1,053 launches now hide behind 18.16 ms
> instead of 32.61 ms — launch overhead has roughly doubled as a *fraction*
> of the pass without changing in absolute terms. It is still not measured
> to be non-zero. **Re-measure before building capture**; do not carry the
> "approximately zero" verdict forward on the strength of the old numbers.

llama.cpp already captures graphs, so even at its best this closes a gap
rather than opening one — a point [ARCHITECTURE.md](ARCHITECTURE.md) already
concedes and which is now doubly true.

**Cost.** Medium. Requires bucketed capture sizes (copy vLLM's
`[1, 2, 4] + range(8, 256, 8)` shape) and a decode batch padded to a bucket,
plus the event-tracking constraint in
[TOOLCHAIN.md](TOOLCHAIN.md#the-event-tracking-constraint).

**Depends on.** R2 (you need a fixed batch shape to capture) **and R3** (there
is nothing to recover until the kernels are fast).

> Design rule 5 — MoE indirection tables built on-device into fixed-size
> buffers — should **not** be relaxed on the strength of this. It costs a
> 93.75% early-out at n=1 (R3), but the fix for that is packing the grid to a
> device-side live count, not reverting to host-sized launches. Keep the rule;
> fix the grid.

---

### R6 — MTP speculative decode

**sm_75: yes.**

**What.** Use block 40's `nextn` head to draft 2–3 tokens per step and
verify them in one batched forward.

**Mechanism.** This is the **only** lever that changes the arithmetic of
*single-stream* decode. A verify step over 1 + d drafted tokens reads the
weights once, exactly like a batch of 1 + d.

**Expected win, with arithmetic.** Drafting 3 tokens makes the verify a
batch-4 step: 5,522.0 MB of weights + 4 × 131.7 MB of state = 6,048.8 MB.
With acceptance rate *a*, accepted tokens per step = 1 + 3a:

| acceptance *a* | tokens/step | MB/accepted token | vs 3,099.2 baseline |
| ---: | ---: | ---: | ---: |
| 0.4 | 2.2 | 2,749.5 | 1.13× |
| 0.6 | 2.8 | 2,160.3 | **1.44×** |
| 0.8 | 3.4 | 1,779.1 | **1.74×** |

**Depends on.** R2's batch machinery and a correct MTP head. `xabe-sched`
already budgets for it (`DEFAULT_DRAFT_TOKENS_PER_STEP` = 3,
[SCHEDULER.md](SCHEDULER.md)).

**The catch, and it is a large one.** llama.cpp **already ships MTP
speculation for this architecture** — `--spec-type`
(`common/arg.cpp:4138-4140`), `COMMON_SPECULATIVE_TYPE_DRAFT_MTP`
(`common/arg.cpp:3042`), and `qwen35moe.cpp:137-142` loads the `nextn`
tensors. The 104.79 tok/s baseline was measured **without** it
([BENCHMARKS.md](BENCHMARKS.md) lists MTP acceptance rate under "Not
measured"). **The real bar may be materially higher than 104.79.** This is
the third time an identified opportunity turned out to be already
implemented upstream, after graph capture and `--backend-sampling`.

---

### R7 — Requantize the projections and LM head from Q8_0 to Q6_K

**sm_75: yes.**

**What.** The projections are the largest single weight term
(1,386.6 MB/token) purely because they are `Q8_0`.

**Expected win.** Q8_0 → Q6_K saves 22.8% of those bytes:

| Tensor group | MB/token at Q8_0 | at Q6_K | saved |
| --- | ---: | ---: | ---: |
| Projections | 1,386.6 | 1,070.5 | 316.1 |
| LM head | 540.3 | 417.2 | 123.1 |
| **Total** | | | **439.2** |

439.2 / 3,099.2 = **14.2% of single-stream decode bytes → +16.5%
throughput**.

**Its value collapses under concurrency**, and this is the important part.
Those tensors are read once per *step*, not once per token, so at N=32 the
saving is 439.2 / 32 = 13.7 MB/token against 774.7 — **1.8%**. R7 is a
single-stream optimization and nothing else. If Program B (§6) is the goal,
skip it.

**Cost.** Low to build (requantize the GGUF), but it **breaks bit-exact
agreement with llama.cpp on the same file**, which is the entire correctness
strategy in [TESTING.md](TESTING.md) and [ORACLE.md](ORACLE.md). It cannot
land until the oracle gate is complete.

**Alternative with the same shape and no new file:** `UD-Q5_K_XL` is
26.6 GB against 30.36 GB, ~12.4% less weight traffic overall. Quality cost
unmeasured.

---

### R8 — Narrow the GDN recurrent state

**sm_75: yes** for fp16 storage. `use_replayssm`'s specific kernels are not
portable but the idea is.

**What.** Store the recurrent state as fp16 rather than fp32
(vLLM's `mamba_ssm_cache_dtype`, `config/cache.py:134-136`), and/or elide
the per-step store (vLLM's `use_replayssm`, `config/cache.py:151-158`).

**Expected win.** fp16 halves 125.8 → 62.9 MB/token/sequence:

| N | saving as % of per-token bytes |
| ---: | ---: |
| 1 | 2.0% |
| 8 | 5.2% |
| 32 | **8.1%** |
| 64 | **11.3%** |

Store elision at B = 16 saves a further ~7% at N=32.

**Cost.** Low to build, **high numerical risk.** Error in a recurrent state
compounds along the sequence; there is no per-step reset. The chunked-vs-
recurrent equivalence test in `xabe-kernels` would catch a systematic bias
but not slow drift over 10,000 tokens. Gate it on a long-generation
comparison, not a single-step differential.

**Rank rationale:** low at the concurrency levels we can reach today, rising
sharply if R2 succeeds past N=32.

---

### R9 — Tensor cores for prefill

**sm_75: partial** — `m16n8k8` fp16 and `m8n8k16` int8 only; see §7.

**What.** Replace the scalar fp32 GEMM in the prefill path with an MMA
tile.

**Mechanism.** From §5.2, prefill is the *only* regime where compute is the
binding constraint, and the fp32 ceiling is below llama.cpp's measured
throughput.

**Expected win.** Necessary to reach parity, not sufficient for a win. See
§8 for the numbers, which are the most important negative result in this
document.

**Cost.** Very high. Re-tiling attention alone is called out in
[KERNELS.md](KERNELS.md#what-the-landed-kernels-do-not-yet-do): `m16n8k8`
needs 16 query rows resident and the current shape is `BM = 1`, and fp16
operands would end the fp32 comparison the kernel is gated on.

**Blocked today by a documented-but-solvable limitation.** llama.cpp's MMQ
path refuses to run if `cudaDevAttrMaxSharedMemoryPerBlockOptin < 48 KiB`
(`ggml/src/ggml-cuda/mmq.cu:303-310`), and on sm_75 that attribute is
**64 KiB**. [KERNELS.md](KERNELS.md) states "shared memory on this hardware
is 48 KiB per block, not 64 KiB … a block reaches it only by opting in
through `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`, which cudarc's
`LaunchConfig` does not expose." The attribute is settable directly through
the driver API with `cuFuncSetAttribute`, independent of what cudarc's
`LaunchConfig` exposes. **Unblocking this is a small change and it is a
prerequisite for any MMQ-style port.**

---

### Not recommended

- **Split the LM head across cards and all-reduce the argmax.**
  [KERNELS.md](KERNELS.md) and [MODEL.md](MODEL.md) both float it. It saves
  360 MB/token (11.6%) at batch 1 and essentially nothing under
  concurrency, it violates the rule that nothing crosses PCIe on the decode
  path, and it requires all three cards to serve one sequence — which
  destroys the three-worker concurrency architecture that
  [ARCHITECTURE.md](ARCHITECTURE.md) is built on. The two goals are
  mutually exclusive; pick concurrency.
- **`q8_0` KV cache.** Already measured, already worse: −15.5% at 32K,
  −35.8% at 128K ([BENCHMARKS.md](BENCHMARKS.md)).
- **SplitK in the routed-expert GEMM.** The 20% figure is from an 8-expert
  model on Ampere, and this vLLM tree sets `SPLIT_K: 1` in every default
  config. At `D(N)` ≈ 164 experts × 512 output rows we have 84,000 output
  rows against 2,304 resident warps — a 36× surplus. SplitK manufactures
  parallelism we do not lack. This is the same argument that already
  correctly rejected split-K for the LM head.
- **L2-grouped launch order at decode.** `GROUP_SIZE_M` = 1 is vLLM's own
  rule when `M // E` is small (`fused_moe.py:1390-1391`), and ours is 0.
  Reserve it for prefill.

---

### R10 — Flash-decoding: split the KV window across blocks

**sm_75: yes.** No architecture dependency; it is a grid-shape change plus a
softmax combine, not a tensor-core path.

**What.** At `n_query = 1` the attention grid is `(1, q_heads)` = 16 blocks.
Split each query head's KV window into `S` chunks, run `16 * S` blocks each
computing a partial online softmax over its chunk, then combine the partials
with their running maxima and sums. This is flash-decoding; vLLM's paged
attention and llama.cpp's parallel-block `fattn` path both do it.

**Mechanism.** 16 blocks on 72 SMs leaves 78% of the machine idle, and each
block streams its whole window sequentially, so the kernel is latency-bound
rather than bandwidth-bound. `S = 8` would fill the card.

**Expected win.** Bounded by the measured slope: the KV read costs 4.23 ms of
a 19.79 ms step at 2,132 positions. Perfect parallelism removes most of that
4.23 ms and none of the other 15.56, so at this context it is worth about 27%
— and it grows with context, which is the point. It does **nothing** at short
context, where the 16 blocks are already enough to cover a small window.

**Cost.** A second attention kernel specialized for `n_query = 1`, plus the
combine. The existing kernel stays for prefill, where `BM = 1` over many query
rows already fills the grid. The partial-softmax combine is the part that is
easy to get subtly wrong, and it needs its own differential test against the
single-block path — which now exists, because `tests/decode.rs` gates the
decode shape against the batch shape.

**Depends on.** R1, which is done.

---

## 5. Where the binding constraint actually is

Two separate regimes, and confusing them is the fastest way to waste
months.

### 5.1 Decode is a memory problem, not a compute problem

FLOPs per decoded token = 2 × 2.946 B active params = **5.89 GFLOP**.
At the 216.8 tok/s roofline that is 1.28 TFLOP/s — **7.8% of the card's
16.31 TFLOP/s fp32 peak**.

Decode stays bandwidth-bound in pure fp32 up to:

| N | FLOPs/step | fp32 time | bytes/step | memory time | ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 188.5 GFLOP | 11.6 ms | 24.79 GB | 36.9 ms | 3.2× |
| 64 | 377 GFLOP | 23.1 ms | 35.71 GB | 53.1 ms | 2.3× |
| 128 | 754 GFLOP | 46.2 ms | 47.44 GB | 70.6 ms | 1.53× |
| 256 | 1,508 GFLOP | 92.5 ms | 64.79 GB | 96.4 ms | **1.04×** |

**Crossover at about batch 256.** Every concurrency level we can plausibly
reach on 47.3 GiB of VRAM is far below it.

> **Therefore: no part of the decode program needs tensor cores.** "We have
> no tensor cores" is not why we lose at decode. We lose at decode because
> the memory path runs at 11.4% instead of 44.5%.

### 5.2 Prefill is a compute problem

At `pp512` all 256 experts are touched (`D(512)` ≈ 256.0), so a step reads
40 × 256 × 2.834 MB = 29,024 MB of expert weights plus 2,060.6 MB of
everything else = **31,085 MB per 512 tokens = 60.7 MB/token**.

- Memory roofline: 672,000 / 60.7 = **11,070 tok/s**.
- fp32 compute ceiling: 16.31 TFLOP/s ÷ 4.874 GFLOP/token (experts +
  projections, LM head applies to one token) = **3,346 tok/s**.
- llama.cpp measured: **2,070 tok/s** = 18.7% of the memory roofline,
  **61.9% of the fp32 ceiling**, 3.9% of the int8-tensor ceiling.
- llmxabe measured: **69.84 tok/s** = 0.34 TFLOP/s = **2.1% of fp32 peak**.

That 61.9%-of-fp32 figure is the tell. No dequantize-then-GEMM pipeline
reaches 62% of fp32 peak. llama.cpp is using tensor cores, and its source
confirms it: `ggml_cuda_should_use_mmq` returns `true` unconditionally when
`turing_mma_available(cc)` (`ggml/src/ggml-cuda/mmq.cu:312`), and the MMA
tiles issue `mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32`
(`ggml/src/ggml-cuda/mma.cuh:929-937`) — **Turing int8 tensor cores, on
`Q6_K` and `Q8_0` weights, today.**

The profile in [BENCHMARKS.md](BENCHMARKS.md#llmxabes-own-forward-pass-and-where-its-time-goes)
reaches the same conclusion from an independent direction: it sums the same
2,491.1 GFLOP from the GGUF tensor directory, measures llama.cpp doing it in
249.6 ms, and gets **9.98 TFLOP/s = 61.2% of fp32 peak** against my 10.09 and
61.9%. It measures llmxabe at **357.5 GFLOP/s = 2.19% of fp32 peak**.

---

## 6. The two programs, kept separate

### Program A — close the gap to llama.cpp on single stream

Target: beat 104.79 tok/s at short context. Roofline is 216.8.

| Step | Multiplier | Cumulative tok/s |
| --- | ---: | ---: |
| Today (latency floor, no KV cache) | — | 30.7 (32.61 ms) |
| **R3: tile the MoE GEMM + GDN/attention projections** | measured headroom 74.2% + 9.6% + 1.4% | **~207** (4.8 ms) |
| R1: subtract the recurrent-state read the floor does not do | ×0.98 | ~203 |
| R5: graph capture | ~1.00× (**measured ≈ 0**) | ~203 |
| R7: requantize projections + LM head | 1.165× | ~236 |
| R6: MTP at *a* = 0.6 | 1.44× | ~340 |

Row 2 comes from the measured per-stage headroom in
[BENCHMARKS.md](BENCHMARKS.md#llmxabes-own-forward-pass-and-where-its-time-goes),
not from my model. Rows 4–6 exceed the 216.8 tok/s roofline for a single
non-speculative token; row 5 does so by shrinking the numerator
(requantization) and row 6 by changing the denominator (speculation), so
neither violates it.

**Every one of these is reachable on sm_75, and none of them needs tensor
cores.** But treat the whole column with suspicion:

- Row 2's ~207 tok/s comes from a pass with no cache to read. §2.3 says the
  recurrent-state read costs only ~2% at short context, so the conclusion
  should survive R1 — but that is a derivation, not a measurement, and it
  degrades with depth as KV traffic grows.
- R6 is available to llama.cpp today and unmeasured there (§9.3), so it may
  raise the bar as much as it raises us.
- R7 breaks bit-exact agreement with the oracle and cannot land until the
  oracle gate is complete.

Strip R6 and R7 out and the honest single-stream program is
**30.7 → ~203 tok/s, roughly 1.9× llama.cpp**, achieved entirely by fixing
one kernel shape. That is a much better position than the 1.1–1.2× I
estimated before reading the profile, and the difference is precisely that
the profile identified *which* inefficiency dominates instead of assuming a
uniform 44.5% ceiling.

### Program B — win on aggregate concurrent throughput

Target: beat llama.cpp's measured 162.4 tok/s aggregate per card, and
whatever it does at higher `-np` (unmeasured).

At llama.cpp's own 41% efficiency, purely from raising concurrency:

| N | roofline | at 41% | vs 162.4 |
| ---: | ---: | ---: | ---: |
| 3 | 395.9 | 162.3 | 1.00× (this is the measured point) |
| 8 | 558.6 | 229.0 | 1.41× |
| 16 | 683.7 | 280.3 | 1.73× |
| 32 | 867.4 | 355.6 | 2.19× |

Combined with R3 taking sustained efficiency to 50% and to 70%:

| N | at 50% | at 70% | vs 162.4 | across 3 cards @70% |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 341.9 | 478.6 | 2.11× / 2.95× | 1,436 |
| 32 | 433.7 | 607.2 | 2.67× / 3.74× | 1,822 |

against llama.cpp's implied 487 tok/s across three cards at `-np 3`.

**The efficiency figure is the whole uncertainty here, and it is bracketed
by two measurements on this card rather than by guesswork:** llama.cpp's MoE
weight path sustains 41–47%, and llmxabe's own LM head sustains **89%**. A
routed-expert gather will not reach 89% — it is a strided read through a
routing table, not a contiguous stream — but 41% is a floor set by another
implementation's choices, not by the hardware. Anything in 50–70% is
defensible as a target and none of it needs tensor cores (§5.1).

**This is where the architecture is aimed and it is where the arithmetic
supports a win.** With two enormous caveats:

1. **llama.cpp can also be run at `-np 16`.** If it holds 41% there, the
   concurrency column is not a differentiator at all, only the efficiency
   column is. E1 below settles this.
2. VRAM. At 47.3 GiB with 29.6 GiB of weights there is ~17.7 GiB for cache.
   Each slot costs 60 MiB of GDN state plus 20,480 B/token of KV. Sixteen
   slots at 32K context each: 16 × (0.0586 + 0.625) GiB = **10.9 GiB** —
   fits. Thirty-two slots at 32K: 21.9 GiB — **does not fit**. Thirty-two
   slots at 16K: 11.9 GiB — fits. **The concurrency ceiling on this card is
   about 16–32 sequences at realistic context depth**, which happens to be
   exactly where the §2.6 curve is still steep.

**The structural advantage remains what it always was**, and it is a
prefill/TTFT advantage, not a decode one: one shared radix tree across three
GPUs, measured at 32× TTFT improvement on a re-submitted 25K prompt, which
three `llama-server` processes cannot replicate
([ARCHITECTURE.md](ARCHITECTURE.md#why-one-process)).

---

## 6b. Tensor cores and TensorRT: what was actually checked (2026-08-16)

Prompted by "try AMP for FP32, or maybe TensorRT if they're reachable". Both
were investigated on this host. Claims below are marked by how they are known.

### TensorRT — not installed, and not applicable if it were

**VERIFIED.** No `libnvinfer*` anywhere on the filesystem, nothing in
`ldconfig -p`, no `tensorrt` Python module, no Rust bindings vendored, and no
`onnx` string anywhere in this repo.

More important than the absence: **it is not a drop-in even installed.**
TensorRT consumes ONNX or its network-definition API. This engine loads GGUF
Q6_K/Q8_0 directly and implements Gated DeltaNet over 30 of 40 layers plus a
256-expert MoE on every layer. Q6_K's two-level superblock scales are not a
TensorRT weight format and GDN is not a builtin layer, so adopting it means
writing an ONNX exporter *and* plugins that reimplement the kernels this
project already has. It subtracts nothing. **Do not pursue it.**

### "AMP" is the wrong frame; the right one is operand precision

Autocast and loss scaling are training concepts and irrelevant here. The
inference equivalent is narrower operands with fp32 accumulate.

**VERIFIED by compiling to SASS on this host:**

- `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` compiles for `sm_75` and
  lowers to a genuine **`HMMA.1688.F32`** instruction. The tensor-core path is
  real and reachable through inline PTX in the NVRTC source, which is the
  mechanism cudarc leaves available and which `dequant.rs` already uses for
  `cvt.f32.f16`.
- **`m16n8k16` is a trap.** NVRTC *accepts* it at `compute_75` and emits PTX;
  **ptxas then rejects it** — `Feature '.m16n8k16' requires .target sm_80 or
  higher`. NVRTC success is therefore not proof of reachability; only
  PTX→SASS is. Any MMA work must hand-decompose to `m16n8k8`.

**MEASURED in-session, harness not committed — re-measure before relying on
it.** A microbenchmark on GPU 0 reported: fp32 FMA 17.90 TFLOP/s; `m16n8k8`
with fp32 accumulate **99.46 TFLOP/s**; with fp16 accumulate 104.76; int8
`m8n8k16` s8→s32 **198.80 TOP/s**; `__dp4a` 50.92 TOP/s.

Two consequences, if those hold up:

1. **§7's open question "whether Quadro RTX 8000 runs fp32-accumulate MMA at
   full rate" appears to be answered yes** — 99.46 vs 104.76 is a ~5% gap, not
   the 2× halving GeForce 20-series suffers. fp32 accumulation would be
   essentially free, which matters because it confines any accuracy loss to
   operand rounding.
2. **The fp16:fp32 ratio is ~5.6×, not the 8× claimed** in `attention.rs` and
   `KERNELS.md`. Those figures are optimistic by ~1.4×.

### int8, not fp16, is the target

The weights are already Q6_K/Q8_0. int8 MMA measured 2× the fp16 rate and is
the *native* format rather than a precision downgrade imposed on a quantized
model — and it is what llama.cpp's MMQ already uses on this same card. So
tensor cores in the **MoE GEMM are not parity, they are the thing llama.cpp
has and this engine does not**, which is consistent with §8.1's conclusion
that a perfect fp32 kernel still loses at prefill.

Note the asymmetry with decode: §8.2 says decode compute is 7.8% of fp32
peak, so **no MMA work addresses the 1.61× decode gap** — that one is
bandwidth efficiency, and R10 (flash-decoding) is its item.

### The cost this understates

`AGENTS.md` requires a differential test per kernel. fp16 or int8 operands
would force `moe_differential.rs`'s `ROUTED_GATE` from `max_abs 5e-7` to
roughly `5e-2` — a **100,000× loosening** — and `attention_differential.rs`
already warns that anything near `reduced_precision_gpu()` "would be a
formulation error hiding behind a loose threshold". That gate is the best
defense against exactly the bug class hand-written MMA fragment indexing
produces. A replacement strategy — matching llama.cpp's quantized arithmetic
bit-for-bit, per §8.1 — needs to exist **before** the first MMA line, not
after. The exact-gated items that are pure data movement (partial-rotary
tail, query/gate deinterleave, causal mask) survive untouched.

## 7. What is reachable on sm_75

### Not available — do not plan around any of these

| Feature | Requires | Evidence |
| --- | --- | --- |
| `bfloat16` | cc ≥ 8.0 | `vllm/platforms/cuda.py:626-634` raises for cc < 8.0 |
| fp8 `e4m3` compute | cc ≥ 8.9 | `marlin_template.h:296-299`: "FP8 computation is only supported for Ada Lovelace or newer" |
| `cp.async` / `ldgsts` | cc ≥ 8.0 | Marlin's sm_75 configs drop to `stages = 2`, `generate_kernels.py:226-228`. [AGENTS.md](../AGENTS.md) already states it. |
| `mma.m16n8k16` fp16 | cc ≥ 8.0 | llama.cpp decomposes into 2× `m16n8k8` on Turing, `mma.cuh:981-989` |
| `mma.m16n8k16` / `m16n8k32` int8 | cc ≥ 8.0 | decomposed into 2×/4× `m8n8k16`, `mma.cuh:928-937`, `:950-966` |
| `__reduce_add_sync` | cc ≥ 8.0 | Marlin uses a `__shfl_down_sync` ladder on sm_75, `marlin_template.h:493-508` |
| TMA / tensor descriptors | Hopper | `fused_moe.py:446-461` (`USE_TD`) |
| `wgmma`, thread-block clusters, async barriers | Hopper | — |
| FlashAttention 2 / 3 | cc ≥ 8.0 / Hopper | use llama.cpp's `fattn-tile.cu` / `fattn-vec.cuh` |
| DeepGEMM, CUTE-DSL GDN chunk path | Hopper+ | `vllm/model_executor/layers/mamba/ops/gdn_chunk_cutedsl/` |
| Blackwell MMA paths in MMQ | cc ≥ 10.0 | `mmq.cuh:274`, `:715` |

### Available — and mostly unexploited by us

| Feature | Peak / detail | Evidence |
| --- | --- | --- |
| fp32 FMA | **16.31 TFLOP/s** (4,608 cores × 2 × 1.77 GHz) | NVIDIA Quadro RTX 8000 datasheet |
| `mma.m16n8k8` fp16 | **130.5 TFLOP/s** tensor, 8× fp32 | datasheet "Tensor Performance"; `mma.cuh:972-992` |
| `mma.m8n8k16` s8 → s32 | **~261 TOP/s** (Turing int8 tensor = 2× fp16 tensor; verified by the T4's 65 TFLOP fp16 / 130 TOPS int8 pairing) | Turing whitepaper; `mma.cuh:922-939` |
| `__dp4a` | **65.8 TOP/s** (datasheet INT8 figure, = 4× fp32 rate) | datasheet; `common.cuh:733-739` |
| Shared memory | 64 KiB/SM; **48 KiB per block by default, 64 KiB per block via opt-in** | CUDA C Programming Guide §5.1; llama.cpp requires the 48 KiB opt-in floor at `mmq.cu:303-310` |
| CUDA graphs | no arch floor | driver feature |
| Marlin MoE, sm_75 variant | fp16 or int8 **activations** only, int4/int8 weights | `CMakeLists.txt:1316` (`MARLIN_MOE_SM75_ARCHS "7.5"`); `marlin_template.h:301-305`: "Turing TensorCore only supports fp16 and int8" |
| Turing fp16-accumulate MMA | `use_fp16_accum` is enabled on sm_75 for fp16 activations | `marlin_template.h:308-320`. **Whether Quadro RTX 8000 runs fp32-accumulate MMA at full rate (unlike GeForce RTX 20-series, which halves it) is unverified.** |

**Note the Marlin constraint carefully.** vLLM ships a real fused-MoE
tensor-core kernel for sm_75, but it accepts only int4 or int8 weights with
fp16 or int8 activations. `Q6_K`'s two-level superblock scale hierarchy is
not one of those formats. **Do not plan to reuse Marlin.** llama.cpp's MMQ
handles `Q6_K` and `Q8_0` natively with the same Turing int8 MMA
instruction, which is why [AGENTS.md](../AGENTS.md)'s rule — port vLLM's
*indexing strategy* onto llama.cpp's *primitives* — is right, and this
finding is a concrete instance of it.

---

## 8. The honest ceiling: what would make the goal unreachable

### 8.1 Prefill: beating llama.cpp on sm_75 without tensor cores is arithmetically impossible

| Bound | tok/s | derivation |
| --- | ---: | --- |
| Memory roofline at `pp512` | 11,070 | 672,000 MB/s ÷ 60.7 MB/token |
| **fp32 compute ceiling** | **3,346** | 16.31 TFLOP/s ÷ 4.874 GFLOP/token |
| fp32 at a realistic 40% of peak | 1,338 | |
| **llama.cpp measured** | **2,070** | [BENCHMARKS.md](BENCHMARKS.md) |
| llmxabe measured | 69.84 | |

llama.cpp's 2,070 tok/s is **61.9% of the absolute fp32 ceiling**. A pure
fp32 kernel would have to sustain more than 62% of theoretical peak FMA
throughput — with K-quant unpacking, gather indexing and masked M tiles in
the same loop — merely to *tie*. That does not happen.

The measured profile puts a concrete number on the shortfall: fixing the MoE
grouped GEMM, the GDN projections, the shared expert and the attention
projections to a *realistic* 25% of fp32 peak takes the n=512 pass from
6,938 ms to ~980 ms = **522 tok/s — still 3.9× behind llama.cpp's 2,051**
([BENCHMARKS.md](BENCHMARKS.md#llmxabes-own-forward-pass-and-where-its-time-goes)).
That is an independently derived confirmation of this section's conclusion.

**Statement, plainly: `llmxabe` cannot beat llama.cpp's prefill throughput
without using Turing's tensor cores.** The margin is 1.6× at 100% of fp32
peak, and no real kernel gets there.

The tensor-core path requires, at minimum:
1. Opting into 64 KiB of shared memory per block via
   `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)`
   — currently documented as unavailable in [KERNELS.md](KERNELS.md), which
   is a cudarc-API observation, not a hardware one.
2. Re-tiling to `m16n8k8` (fp16) or `m8n8k16` (int8) shapes, decomposed by
   hand because the wider Ampere MMA shapes do not exist.
3. Quantizing activations to `q8_1` and dotting in int8, exactly as
   llama.cpp does — which, per [ORACLE.md](ORACLE.md) §8, is 1,000–10,000×
   less accurate than exact fp32 and changes every tolerance in the test
   suite.
4. Hand double-buffering, because there is no `cp.async`.

Item 3 is the one to think about before starting: the current correctness
strategy is exact fp32 agreement with a CPU reference. The prefill win
requires abandoning it in favour of matching llama.cpp's *quantized*
arithmetic instead.

### 8.2 Decode single-stream: possible, but not by much

The roofline is 216.8 tok/s. llama.cpp sits at 48.3% of it. Compute is not
the constraint (7.8% of fp32 peak, §5.1), so **tensor cores are not
required** and there is no arithmetic impossibility.

But the target is a strided gather over K-quant superblocks with a
routing-dependent access pattern, and the incumbent is an implementation
that has been tuned for this exact architecture (`mmq-config-turing.cuh`)
over years. Getting from 11.4% to 44.5% is credible. Getting meaningfully
past 44.5% is a research project, and the flash-attention path's ~80%
should not be read as evidence it is easy — flash attention streams KV
contiguously; the MoE weight path does not.

**Realistic single-stream outcome: parity to +20%, not a rout.**

### 8.3 Aggregate concurrency: the ceiling is the recurrent state

The §2.6 curve flattens because the per-sequence terms stop amortizing. The
asymptote as N → ∞, at depth 0:

```
weights → 2,060.6/N + 113.377 × 256/N → 0
state   → 131.7 MB/token, constant
```

So the absolute ceiling on aggregate decode throughput per card is
672,000 / 131.7 = **5,102 tok/s**, and it is set entirely by GDN state
traffic. At depth 4K, KV adds 84 MB/token and the ceiling falls to
3,116 tok/s. These are far above anything reachable, but they say what the
final wall is made of: **not weights, not experts — recurrent state.**

The practical ceiling is VRAM (§6, Program B): about 16–32 concurrent
sequences at realistic depth.

### 8.4 The condition under which the whole goal fails — RESOLVED

This section originally read: *"If llama.cpp at `-np 16` holds ~41%
efficiency, then llmxabe's only path to a win is R3 — raising bandwidth
efficiency above 44.5% — and nothing else in this document matters."*

**E1 has been run, and the answer is worse than that condition.** llama.cpp
at 16 concurrent sequences does not merely hold its efficiency; it reaches
**382.02 tok/s decode, 3.64× its single-stream 105.09** (§2.7). It scales
better than this document projected, not worse.

So the conclusion stands in its strong form: **this project has a kernel
story, not a concurrency story.** Concurrency is table stakes — llmxabe
needs batched decode to be in the same conversation at all, and it does not
have it — but building it buys parity with a capability llama.cpp already
ships, not an advantage over it.

Every path to actually winning runs through R3 and the tiling program:
raising the weight path above its measured 6.47% of bandwidth peak. The MoE
GEMM is 67–77% of every pass, and the same codebase already reaches 88.98%
in the LM head, so the deficit is the kernel and nothing else. Concurrency
then multiplies whatever efficiency that program achieves — for llmxabe and
for llama.cpp equally.

The honest summary: **there is no shortcut here.** The gap must be closed
where it is, in the grouped GEMM.

---

## 9. Contradictions with the existing docs

Stated plainly, as the project's own rules require.

1. **The 235 tok/s roofline in [MODEL.md](MODEL.md) is optimistic by ~8%.**
   It counts weight bytes only. A GDN decode step also reads and writes
   62.9 MB of recurrent state and ~2.9 MB of conv state, which is
   131.7 MB/token — 4.2% of the total at batch 1, and 16.9% at batch 32.
   The corrected batch-1 roofline is **216.8 tok/s** (§2.4). Every
   "fraction of roofline" figure in [BENCHMARKS.md](BENCHMARKS.md) is
   correspondingly a few points low.

2. **The strategic premise "llama.cpp batches poorly" is not supported by
   its own measurements.** llama.cpp holds a flat 41% of the
   concurrency-aware roofline from c=1 to c=3 (§2.7). Its 1.77× at c=3 is
   97% of the 1.83× the expert-activation arithmetic permits. The
   concurrency opportunity is *more slots* and *higher efficiency*, not
   *better batching*.

3. **[BENCHMARKS.md](BENCHMARKS.md)'s 104.79 tok/s may not be the real
   bar.** llama.cpp ships MTP speculative decoding for `LLM_ARCH_QWEN35MOE`
   (`common/arg.cpp:3042`, `:4138-4140`; `src/models/qwen35moe.cpp:137-142`)
   and the benchmark did not enable it. By the §R6 arithmetic a 60%
   acceptance rate would put llama.cpp at ~150 tok/s. This is the third
   "already implemented upstream" finding after graph capture and
   `--backend-sampling`.

4. **[KERNELS.md](KERNELS.md)'s "48 KiB shared memory per block, not
   64 KiB" is a cudarc limitation, not a hardware one.** sm_75's
   `MaxSharedMemoryPerBlockOptin` is 64 KiB, `cuFuncSetAttribute` is a
   driver-API call available regardless of what cudarc's `LaunchConfig`
   exposes, and llama.cpp's MMQ path *requires* the 48 KiB opt-in floor
   (`mmq.cu:303-310`). As written, the doc reads as a hardware ceiling and
   would quietly rule out the MMQ port.

5. **[KERNELS.md](KERNELS.md)'s Turing adjustments recommend two things
   that are wrong for the decode regime.** SplitK "supplies parallelism the
   M dimension cannot" — but at `D(N)` ≈ 164 experts × 512 output rows
   there are 84,000 output rows against 2,304 resident warps, a 36× surplus.
   And L2-grouped launch order is exactly what vLLM disables when
   `M // E` is small (`fused_moe.py:1390-1391`). Both belong to prefill.

6. **[CACHE.md](CACHE.md)'s account of vLLM's page-size unification is
   right about the constraint and wrong about the remedy.** vLLM still
   requires uniform page size across groups (`kv_cache_utils.py:1136-1138`),
   but it satisfies it by *raising the attention block size in tokens*
   until the attention page matches the Mamba page
   (`platforms/interface.py:888-913`), not by padding attention pages to
   waste. For our geometry that yields a ~1,024-token attention block. The
   memory is not wasted; the *hit granularity* is. `xabe-cache`'s design is
   still better, for a slightly different reason than the doc gives (§3(b)).

7. **The tensor-core headroom is 16× fp32, not 8×.**
   [BENCHMARKS.md](BENCHMARKS.md#llmxabes-own-forward-pass-and-where-its-time-goes)
   states that llama.cpp's `mmq` path "dots in int8 on the tensor cores,
   whose published peak on this card is 130.5 TOPS — 8× the fp32 peak", and
   scores llama.cpp at 7.6% of it. 130.5 is the **FP16** tensor rate in
   TFLOPS (the datasheet's "Tensor Performance"). Turing's **INT8** tensor
   rate is 2× the FP16 rate — corroborated by the Tesla T4's published
   65 TFLOPS FP16 / 130 TOPS INT8 pairing — so the figure for
   `mma.m8n8k16.s8` is **~261 TOPS, 16× fp32**, and llama.cpp sits at ~3.8%
   of it. This strengthens that section's own conclusion rather than
   weakening it: there is twice as much tensor-core headroom as it claims,
   and correspondingly more distance for an fp32 implementation to make up.

8. **[BENCHMARKS.md](BENCHMARKS.md)'s prefill headline and my §1 table
   disagree slightly** — 2,070.50 vs 2,051.30 tok/s for llama.cpp, 69.84 vs
   73.48 for llmxabe. Different runs on different cards on different days.
   Neither changes any conclusion; the ratio moves from 29.6× to 27.9×.
   Quoted both in §1 rather than silently picking one.

9. **[ARCHITECTURE.md](ARCHITECTURE.md)'s LM-head-split idea and its
   three-worker concurrency architecture are mutually exclusive.** Splitting
   the LM head across cards requires all three cards to serve one sequence.
   See "Not recommended" in §4.

---

## 10. Suggested experiments

I had no GPU for this task; all three cards were in use by sibling workers.
These are the measurements that would convert the derivations above into
facts, ordered by how much they change the plan.

**E1 — llama.cpp aggregate throughput at `-np 8`, `-np 16`, `-np 32`.**
This is the single highest-value measurement available and it requires no
code. §8.4: if llama.cpp holds ~41% efficiency at 16 slots, the entire
concurrency argument for this project collapses to the kernel argument.
Watch VRAM; at `-np 16 -c 524288` the KV pool alone is ~10 GiB.

**E2 — llama.cpp with MTP speculation enabled.** `--spec-type` with the MTP
type (the exact type string is enumerated by
`common_speculative_all_types_str()`, `common/arg.cpp:4139` — confirm before
running). Report the acceptance rate and the resulting tok/s. This tells us
what the real single-stream bar is (§9.3).

**E3 — Confirm `MaxSharedMemoryPerBlockOptin` on the Quadro RTX 8000.**
One `cuDeviceGetAttribute` call. Settles §9.4 and unblocks any MMQ-style
port.

**E4 — ~~Microbenchmark the routed-expert gather GEMV~~ — done.** Answered
while this document was being written: the MoE runs at **43.51 GB/s = 6.47%
of peak** at n=1 against the LM head's 597.93 GB/s = 88.98%
([BENCHMARKS.md](BENCHMARKS.md#llmxabes-own-forward-pass-and-where-its-time-goes)).
The follow-up worth running is the same measurement **after** R3 lands, at
N = 1, 8, 16, 32, to find where on the 41%–89% bracket the tiled kernel
actually sits. That number sizes all of Program B.

**E4b — Bound the redundant-read term.** The profile could not narrow the
true DRAM traffic at n=512 below "somewhere between 29.2 GB and 282 GB"
because `ncu` fails with `ERR_NVGPUCTRPERM` on this host. Either get
`NVreg_RestrictProfilingToAdminUsers=0` set, or bound it indirectly by
varying `BLOCK_SIZE_M` and reading the time curve.

**E5 — Measure fp32-accumulate `m16n8k8` MMA throughput on this card.**
Settles whether the Quadro RTX 8000 pays the GeForce fp32-accumulate
penalty (§7, marked unverified). Changes the R9 arithmetic by 2× if it does.

**E6 — Measure real expert-activation density at N = 2, 4, 8, 16, 32** on
representative traffic, against the uniform `D(N)` in §2.5. DynaExq's
Qwen3-Next data suggests real routing is ~16% less diverse than uniform at
N=32, which would make every concurrency figure in §2.6 *better* than
stated. Worth knowing before sizing anything.

**E7 — Long-generation numerics check for an fp16 GDN state** (R8), over
≥10,000 tokens against the fp32 path. A single-step differential will not
catch compounding drift.

---

## 11. Sources

### Source trees read directly

- vLLM, `/home/nixabe/vllm`, commit `d4801990a45792c7081652f8ebea4ee56ceb67f9`
  (2026-08-15). All `vllm/…` and `csrc/…` line references above are against
  this commit.
- llama.cpp, `/home/nixabe/llama.cpp`, commit
  `fd6863a69542c74a617a1219f1b18ccf773f41ed` (2026-08-15), build 10456 —
  the same build [BENCHMARKS.md](BENCHMARKS.md) measured.

### Papers

- Rui Pan et al., *Marconi: Prefix Caching for the Era of Hybrid LLMs*,
  MLSys 2025 (Outstanding Paper, Honorable Mention).
  <https://arxiv.org/abs/2411.19379> — exact-match constraint on recurrent
  state; reuse-forecasting admission; up to 34.4× token hit rate, 71.1%
  lower TTFT. Artifact: <https://github.com/ruipeterpan/marconi>
- Amey Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference
  with Sarathi-Serve*, OSDI 2024.
  <https://www.usenix.org/conference/osdi24/presentation/agrawal>,
  <https://arxiv.org/pdf/2403.02310> — chunked prefill and stall-free
  scheduling; up to 2.6× (Mistral-7B, 1×A100) and 6.9× (Falcon-180B,
  8×A100) serving capacity within SLO.
- Kexin Chu et al., *Dynamic Expert Quantization for Scalable
  Mixture-of-Experts Inference* (DynaExq).
  <https://arxiv.org/abs/2511.15015> — Table 1, "Expert activation ratio (%)
  in decode stage": Qwen3-Next-80B 1.9% at batch 1, **39.2% at batch 32**;
  Table 2, prefill: **86.2% at batch 32**. Used in §2.5 to calibrate the
  uniform-routing model.
- *Sparse Prefix Caching for Hybrid and Recurrent LLM Serving*.
  <https://arxiv.org/pdf/2605.05219> — claims up to 2.3× on hybrid models
  by choosing checkpoint positions rather than caching densely.
  **The specific configuration behind that number could not be verified
  from the fetched text; treat the figure as unconfirmed.**
- Less Wright et al., *Accelerating a Triton Fused Kernel for W4A16
  Quantized Inference with SplitK work decomposition*.
  <https://arxiv.org/pdf/2402.00025>

### Vendor and engine documentation

- NVIDIA Quadro RTX 8000 datasheet — 16.3 TFLOPS FP32, 130.5 TFLOPS tensor,
  672 GB/s, 4,608 CUDA cores, 576 tensor cores.
  <https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/quadro-rtx-8000-us-nvidia-946977-r1-web.pdf>
- NVIDIA Turing Architecture Whitepaper — second-generation tensor cores add
  INT8 and INT4; INT8 runs at 2× the FP16 tensor rate (corroborated by the
  Tesla T4's 65 TFLOPS FP16 / 130 TOPS INT8 pairing).
  <https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf>
- CUDA C Programming Guide §5.1, Compute Capabilities — 48 KiB default
  shared memory per block, 64 KiB opt-in on compute capability 7.x.
  <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html>
- vLLM CUDA Graphs design doc.
  <https://docs.vllm.ai/en/stable/design/cuda_graphs/>
- vLLM Optimization and Tuning.
  <https://docs.vllm.ai/en/stable/configuration/optimization/>
- PyTorch blog, *Accelerating MoE model inference with Locality-Aware Kernel
  Design* — SplitK ~18–20%, then grouped launch order for L2.
  <https://pytorch.org/blog/accelerating-moe-model/>
- vLLM blog, *vLLM Now Supports Qwen3-Next*.
  <https://vllm.ai/blog/2025-09-11-qwen3-next>

### Secondary sources, used only where labelled

- TNG Technology Consulting, *Prefill and Decode for Concurrent Requests*
  — chunked prefill worth +50% total token throughput in a production vLLM
  deployment.
  <https://huggingface.co/blog/tngtech/llm-performance-prefill-decode-concurrent-requests>
- A. Kuo, *Qwen3.6-35B-A3B on Desktop Blackwell* — **our exact model** on an
  RTX PRO 6000 Blackwell (96 GB), vLLM FP8: 208.6 tok/s decode, 438 tok/s
  maximum concurrent, 25 ms TTFT; Ollama Q4_K_M 144.2 tok/s decode. Note the
  concurrency ratio is only 438/208.6 = **2.1×**, which is a useful reality
  check against §6 Program B's optimism — though the concurrency level
  behind "maximum concurrent" is not stated, so the comparison is not
  clean.
  <https://allenkuo.medium.com/qwen3-6-35b-a3b-on-desktop-blackwell-the-first-time-vllm-beats-ollama-on-decode-f139f445f926>
- CUDA graph decode speedup of 10–20% at small batch, 20–40% launch overhead
  at batch 1, 50–100 µs saved per replay. **These circulate widely but I
  could not trace them to a primary benchmark**; used only in §3(d) and R5,
  and flagged there.
