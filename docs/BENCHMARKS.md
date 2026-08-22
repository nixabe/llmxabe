# Benchmarks

Where `llmxabe` stands against `llama.cpp` on the target host, and the
reasoning that produced it.

This file is **not a journal**. It carries the current standing, the method
that makes it trustworthy, and two lists that outlive any particular number:
**why** the engine is shaped the way it is, and **why not** — the things that
were built, measured, and rejected. Per-session narratives, superseded tables
and the dated arc of individual changes live in `git log`, which is where a
reader who wants the day-by-day should go.

**These are measurements, not estimates.** Where a measurement contradicted a
plan, the measurement won.

## Setup

| | |
| --- | --- |
| Hardware | 3× Quadro RTX 8000, sm_75 (Turing), 47.3 GiB usable, 672 GB/s, 72 SMs, 6 MiB L2 |
| Host | 125 GB RAM, 12 vCPU, CUDA 12.4, driver 595.84 |
| llama.cpp | build 10456, commit `fd6863a69` |
| Model | `Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf` — 30.36 GiB, 35.51 B params |

The comparison configuration — llama.cpp at its own best settings, three
parallel sequences, one card:

```sh
llama-batched-bench -m Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf \
  -ngl 99 -sm none -fa on -b 4096 -ub 4096 -ctk f16 -ctv f16 \
  -npl 3 -npp <ctx> -ntg 32
```

`-ub 2048` is llama.cpp's better decode setting and `-ub 4096` its better
prefill setting at most depths, so both are kept as bars and the faster of the
two is the one to clear. Its decode bars at `-ub 2048` are **187.90 tok/s** at
2K and **153.79 tok/s** at 32K, N=3.

## How these numbers are measured

The discipline is the reason the standing can be trusted at all; several
"wins" of 5–10% turned out to be the card warming up.

- **Same card, same hour, alternating processes.** llama.cpp drifts about a
  point a day on this host, this card's thermal drift is ~1.3% between a cold
  and a warm run, and card-to-card spread is ~1.5% (GPU 2 fast, GPU 1 the
  house standard). Only alternated pairs are comparable — a before/after taken
  an hour apart carries about a percent of drift, which is the size of several
  of this project's real margins.
- **At least three interleaved pairs**, spreads reported. A change that wins
  every pair is readable below the drift; a change that wins on the mean but
  loses a pair is not.
- **Never bench two GPUs at once on this host** — host contention costs ~6%
  and has manufactured a phantom regression at 65K prefill.
- **CUDA events, not `Instant`**, for anything inside a pass. Host clocks
  measure enqueue latency.
- **Kernel-level first, then end to end.** `bench_attention` (~6 s per A/B)
  exists because whole-forward A/Bs are expensive enough that the honest
  response to a small change was to not measure it.
- **`ncu` does not work on this host** (`ERR_NVGPUCTRPERM`). The working
  substitutes are `nsys`/`nvprof` timelines, `nvcc -Xptxas -v` and
  `cuobjdump -sass` on an extracted kernel, roofline arithmetic from the GGUF
  tensor directory, and ablation. Do not spend time re-trying `ncu`.
- **Rebuild through Cargo.** Invoking `target/release/<bin>` directly does not
  ask Cargo whether it is stale; this has twice produced numbers that were a
  stale binary rather than a result.
- **Compare against llama.cpp's best settings, never its defaults.** Measuring
  against `-ub 512` inflated three separate claims before the rule was named.

## Current standing

Every cell of the serving target — one card, N=3, llama.cpp at
`-np 3 -b 4096 -ub 4096` — measured same-hour, same-card, alternating
processes. Prefill on GPU 2, decode on GPU 1.

| cell | llmxabe tok/s | llama.cpp tok/s | margin |
| :--- | ---: | ---: | ---: |
| prefill 512 | 3,281.0 ± 8.6 | 3,007.5 | **+9.1%** |
| prefill 2K | 3,819.4 ± 23.4 | 3,201.5 | **+19.3%** |
| prefill 8K | 3,665.5 ± 16.8 | 3,201.8 | **+14.5%** |
| prefill 32K | 2,790.3 ± 5.8 | 2,655.1 | **+5.1%** |
| prefill 65K | 2,226.0 ± 0.2 | 2,142.8 | **+3.9%** |
| prefill 128K | 1,564.5 ± 0.3 | 1,551.6 | **+0.8%** |
| decode 2K | 206.4 | 185.2 | **+11.4%** |
| decode 32K | 158.3 / 157.8 / 158.2 | 150.7 / 150.8 / 150.6 | **+4.9%** |

Provenance, in the spirit of the drift rules above: the 512–32K prefill rows
ran a binary predating the per-sequence prefill fork, whose effect at those
depths is orders of magnitude inside the reported margins; the 65K row was
measured while gates ran on a neighbouring card and its margin is therefore a
floor; the 128K rows ran clean and uncontended. The prefill 2K and both
decode rows are alternating same-card runs of the current tree; their
llama.cpp columns are the standing head-to-head rather than a same-hour
re-run, so read those three margins with llama.cpp's ~1%/day drift in mind.

Single sequence, same tree, for reference: **~87.5 tok/s** decode at 32K and
~101 tok/s at 2K. Aggregate across three cards, one session each, is a
deployment figure and **not** the per-instance N=3 target — do not cite it as
a replacement claim.

### Capability

| | |
| --- | --- |
| Max context, one card | 131,072 positions, peak ~35.6 GiB of 47.27 |
| KV cache | binary16, 20 KiB/token over 10 attention layers |
| Recurrent state | fixed ~2 MiB per GDN layer per sequence, independent of depth |
| Parallel sequences | batched decode and flattened batch prefill at N=1–8 |
| Speculative decode | seven drafters (`ngram`, `ngram-simple`, `ngram-mod`, `ngram-map-k`, `ngram-map-k4v`, `draft-mtp`, `spec-dflash`), each bit-exact against plain decode; off by default — a net win at N=1 only, see WHY and WHY NOT |

## Correctness gates

Throughput claims in this file were all taken with these green. They are hard
gates: a speedup that flips a logit ranking or needs a loosened tolerance is a
reject regardless of size.

| Gate | What it asserts |
| --- | --- |
| `forward_pass` golden | Reproduces llama.cpp's argmax token 25358 (`' Tokyo'`) and its logit ordering on the captured 19-token prompt |
| `batch_decode` | A sequence decodes **bit-identically** whether batched with others or run alone — `max_abs 0.000e0` on every full 248,320-logit row |
| `batch_prefill` | A 6,138-row flattened N=3 pass is bit-exact against three serial 2,046-row passes |
| `attention_differential` | Nine cases including a 128K window, at `GATE` 1e-5 for scalar paths and `MMA_GATE` (8 × binary16 half-ulp) for tensor-core paths |
| `moe_differential`, `gdn_*_differential`, `lm_head_differential` | Every kernel against its CPU reference, plus exact cross-kernel identity where two compiled kernels must agree |

The batch-vs-single-stream gate is a **serving contract**, not a tolerance:
three instances share a card, and a sequence's output must not depend on which
other sequences happen to be resident with it. `0.000e0`, not "small enough",
is the only target that makes a downstream discrete decision — top-8 expert
routing today, activation quantization if it is ever re-attempted — safe to
sit on. Measured directly: a ~1e-5 reduction-order residual upstream was
enough to flip expert selection and spike a layer's divergence 26×.

---

# WHY

The reasons the engine is shaped the way it is. Each is a mechanism that paid,
stated so it transfers to the next kernel rather than as a changelog entry.

## Arithmetic: integer tensor cores are the only way to win prefill

llama.cpp runs the same 2,491 GFLOP of a 512-token pass at ~61% of this card's
16.31 TFLOP/s fp32 peak, which no dequantize-then-FMA pipeline reaches. It is
not spending fp32: `mmq` dots in int8 on Turing's tensor cores. Measured on
this card: `mma.m16n8k8` fp32-accumulate **97.6 TFLOP/s**, `mma.m8n8k16` s8→s32
**~198 TOP/s**, against fp32 FMA's 16.3 TFLOP/s. A perfectly tiled fp32 engine
lands ~3.9× behind; the ceiling is arithmetic, not tuning.

Q6_K and Q8_0 weights are *already integers*, so using the integer tensor cores
is not a precision downgrade imposed on float weights — it is declining to
convert integers into floats in order to multiply them more slowly.

Two ways to reach a tensor core from a quantized weight, and the choice is
VRAM, not preference:

- **Repack** into split quant/scale arrays so operand loads are aligned words.
  Used for the dense projections and the shared expert. The same kernel
  measures 27 TOP/s repacked against 2.1 TOP/s assembling operands byte by byte
  from the on-disk layout — reading GGUF Q8_0 in place is fatal, because a
  34-byte block means a fragment's four bytes are never word-aligned.
- **Stage through shared memory**, where the kernel picks the layout. Used for
  the routed experts, whose 10.7 G weights cannot afford a second copy.

This is why llama.cpp's MMQ, TurboMind and vLLM's Marlin all carry their own
packed weight layouts instead of reading the on-disk format.

## Memory: find the index the operand does not depend on

Roughly half of every large win in this project's history was not arithmetic.
It was a kernel re-reading the same bytes, found by asking one question of
every kernel: *how many times does this grid read the same byte?*

- The GDN projection's grid was one warp per (output row, token), so the weight
  matrix was read once per token — 512× at prefill width.
- `gdn_chunk_inter` re-read the entire recurrent state once per token; the
  state does not depend on the token index.
- The MoE router re-read a 2,048-float weight row per (expert, token) pair —
  2.1 GB of loads per layer to cover 6 MB.
- Prefill attention's grid was one block per (query tile, **query** head), and
  this model has 16 query heads against 2 KV heads, so eight blocks each
  streamed the same K and V independently. Grid on the **KV** head with K/V
  staged in shared is a straight 4–8× traffic cut.
- The GDN scan gives one warp four adjacent value columns: they share the same
  normalized q/k row, decay, beta and address arithmetic, and only the
  accumulators differ.
- The LM head's four-row tile hoists the activation `float4` loads outside the
  row loop, so the activation-pipe cost per weight element divides by the tile.

Two operands, two reuses, and they are complementary: a **token tile**
amortizes the weight, a **row band** amortizes the activation. The optimum is
interior and is not where "bigger tile is better" would put it.

## The instruction-count bug that looks like a bandwidth bug

Turing issues **4 load/store operations per SM per clock against 64 FMAs**. Any
kernel written at roughly one memory instruction per multiply-add runs at a
sixteenth of the arithmetic pipe, and no amount of L2 hit rate moves it. Four
kernels — flash attention, the GDN state update, the GDN solve, the alpha/beta
gates — each looked bandwidth-bound in a roofline sense and none were: the
attention kernel was reaching 2 TB/s of *effective* bandwidth, which is L2
working perfectly, and it was still 16× off.

The fix is the same in all four: a register tile, so one loaded element feeds
many multiply-adds. All four were **bit-identical** afterwards, which is not a
coincidence — a register tile reassociates nothing, and that is why each was
safe to make.

## Layout: banks and sectors

Three separate wins came from arithmetic on addresses, and none of them shows
up as bytes moved or instructions issued:

- **A shared stride that is a multiple of 32 words is an 8-way bank
  conflict.** Both MoE tensor-core kernels staged their activation tile at a
  128-byte row stride — exactly 32 banks — so all eight fragment rows landed on
  the same four banks. 144 bytes (36 words, `36 mod 32 = 4`) tiles the banks
  exactly. **+13% of prefill**, from sixteen wasted bytes a row.
- **Q6_K's 210-byte superblock is padded to 224 on the device.** 224 is a
  multiple of 16 *and* of 32, so every field is aligned for wide loads and a
  superblock starts on a sector boundary. A 212-byte stride buys the load
  alignment without the fetch alignment and is not equivalent. The pad costs
  6.7% more sector traffic on two tensors, which prefill has spare bandwidth
  for and decode does not — a deliberate trade, taken.
- **Q8_0 puts a row's quants at byte `b * 34 + 2`.** Fifteen warp reads in
  sixteen straddle a 32-byte sector boundary, so half of every fetch is
  discarded. The split repack built for the tensor cores fixes this for free,
  and turns out to be worth having for its *alignment* even where the
  arithmetic stays fp32. Where the on-disk layout must be read directly, two
  aligned 16-bit loads recover four signed bytes without the misalignment.

Naming a load's width matters too: a runtime-derived stride and a per-element
predicate both block vectorisation, and `cuobjdump -sass` counting
`LDG.E.128` against scalar `LDG.E` is how that is settled without a profiler.

## Decode is a GEMV, and a GEMM's machinery is pure overhead there

At one token every matmul in the model is a GEMV. Three kernels were still
staging an activation tile in shared memory and crossing two barriers per 128
elements of contraction in order to multiply each dequantized weight exactly
once. There is no reuse to capture at that shape — a weight is read, used, and
dropped — so all of that apparatus is cost.

The same principle repeats at every level of the dispatch:

- A dispatch bucket wider than a tile pass meant the second, all-padding pass
  still dequantized and contracted the full weight stack for a result that
  could only be zero. One `bm == 0` early-continue: **+21% at N=3**.
- The live-row count per bucket is a closed form of two values the dispatch
  kernel already has, so it is precomputed into a table rather than re-derived
  by every consumer.
- At N=3 there are 24 routed token/expert pairs against 23.26 distinct experts
  in expectation, so almost every bucket holds exactly one real token. Routing
  those pairs **directly** — a fixed grid over `max_tokens * top_k` reading
  `topk_ids[flat]`, with the device-side `valid_tokens` scalar gating it so
  graph capture is preserved — beats the bucketed mapping, and lets the two
  dispatch-table launches be skipped entirely at those widths.
- A tile widened for prefill needs a one-unit sibling before it is committed,
  because decode is the one-unit case of every one of them. A partial band
  costs its empty lanes in full: at `n_query == 1` a banded flash kernel still
  ran eight multiply-adds per key to throw seven away.

## Parallelism: fill the wave, and split the only axis decode has

Decode attention's grid was `(1, q_heads)` — **sixteen blocks on a 72-SM
card**, each streaming its whole KV window sequentially. The fix is
flash-decoding: split the key range across blocks, have each produce a partial
`(m, l, acc)`, and merge them. That recovers the parallelism *and* the GQA
redundancy at once, and it is worth 114% at 8K depth.

The wave shape then matters as much as the split count, and the two are
different axes:

- The **logical** split count sets the numerics (each split's reduction length
  is what the deep-window 1e-5 gate sees). 36 and 48 logical splits fail it at
  `2.28e-5`; 72 and 96 pass.
- The **block** count is pure scheduling once the kernel grid-strides its
  splits, and the partials are bit-identical at any block count because no
  reduction boundary moves.

At 96 logical splits, 48 blocks per call grid-stride exactly two splits each,
and the N=3 shape's three concurrent calls fill all 288 resident block slots in
one wave. Uneven division is what sank the earlier 48-block attempt at 72
splits: half the blocks took two splits and the long blocks set the makespan.

Per-sequence work that touches disjoint state — rope, cache append, the causal
read — forks onto side streams and rejoins before the batched projections. This
is worth 2–4% where the launches are several partial waves whose scheduling
tail was otherwise paid once per sequence per layer.

Sequence-owned state forbids batching over the *token* axis, and for a long
time that was read as forcing one launch per sequence — 2n GDN launches per
layer per decode step, and n serialized whole-chunk scans per layer at
prefill. It only forces one *state pointer* per sequence: the kernels take
`STEP_MAX_BATCH` scalar pointer slots, `grid.z` picks a sequence, and one
launch advances every sequence's state against the batch scratch it was
already reading. Graph capture sees exactly the pointer stability the
per-sequence launches gave it, and batch-vs-single bit-equality is a property
of the source — each z slice runs the identical arithmetic. The recurrent
step went from 291 GB/s across three serial launches to **~583 GB/s (87% of
streaming peak)** in one; the batched scan is bounded instead by its own
register-limited occupancy (~1.8 waves at N=3), which is why prefill gained
one point where decode gained several.

A warp per state row, not a block. The delta-rule step's block-per-row form
paid two block-wide reductions — two `__syncthreads`, a shared round trip
and a serial cross-warp combine — per 512-byte row, and moved state at 44% of
peak. One warp per row holds the row in four `float4` registers per lane,
reduces with barrier-free shuffle butterflies, and loads and stores nothing
narrower than 16 bytes.

## Prefill: delete the chunking, and widen the pass

llama.cpp does not chunk the Gated DeltaNet at all — `gated_delta_net_cuda` is
a sequential token-by-token scan with the whole per-head state in registers,
zero shared memory and zero `__syncthreads`, and its CUDA file carries a
`//TODO: Add chunked kernel for even faster pre-fill`. The arithmetic says why:
at this geometry the chunked form does **26% more arithmetic**, and it only
pays when the matmul shape buys tensor cores. Turing has no fp32 tensor cores,
so in fp32 on this card chunking is pure overhead. Deleting it for prefill was
worth 4.5%.

Above that, the physical pass width is a throughput lever in its own right.
Flattening all three N=3 prompts into one physical pass over independent
carried KV and recurrent state is **not** a win at equal total ubatch — three
2,046-token serial prompts and three 682-row chunks are the same three
2,046-row passes — but it is a large win when it makes the pass *wider* than
any single sequence could: 6,138 rows for 2K, the whole 24,576-row prompt for
8K, 12,288 for 32K and above. That is the shape llama.cpp's own tuning points
at when it prefers `-ub 4096` over `-ub 2048`.

## Precision is a design constraint, not a tuning knob

The engine carries activations in fp32 end to end and gates every kernel
against a CPU reference. Two consequences shape the kernels:

- **`Q K^T` at decode uses a split-precision `Q`.** `m16n8k8` takes fp16
  operands only, and a single fp16 rounding of `Q` breaks the 1e-5 decode gate
  at `n_keys = 61`. Row `g` carries `f16(q[g])` and row `g+8` carries
  `f16(q[g] - f32(q_hi[g]))`, so `D[g] + D[g+8]` is `Q K^T` at roughly `2^-22`.
  The fragment's dead rows were being multiplied by zero anyway, so the split
  is free *relative to this kernel* — though it costs exactly 2× the tensor-core
  issue count of a single rounding, which is a real and permanent tax against
  llama.cpp's own shape.
- **The router cannot afford a reassociation.** Its output feeds a top-8 argmax
  over 256 experts, so a last-bit disagreement does not perturb an answer
  slightly — it runs a different expert, and everything downstream is a
  different model. The tiled router stages a contraction width equal to the
  block width, reproducing the untiled per-thread index order exactly.
  Correctness cost 1.1% and a faster wrong version was caught by the golden
  test's error-growth guard.

`exp2f` in the prefill softmax lands because it is an **identity**, not an
approximation: `expf(x)` is `exp2f(x)` plus range reduction, so folding
`log2(e)` into the score scale computes the same function. It was still held to
the full gate, because "identity in exact arithmetic" is not "identity in
floating point".

## CUDA graph capture: kept for the host, not for throughput

Capture works and does exactly what it should — idle per decode step falls from
0.97 ms to 0.405 ms, host turnaround from 0.31 ms to 0.02 ms — and the wall
clock does not move. Under replay every decode kernel is 0.3–0.5 µs slower,
uniformly, across all ~1,100 of them: on Turing a graph node costs roughly what
the launch gap it replaces cost. (Ampere added hardware acceleration for graph
node dispatch; do not generalize this result forward.)

It is kept anyway, for reasons that are not throughput: the host cost of a
decode step went from 3.1 ms to ~0.05 ms, which is most of what a serving
surface needs the host for, and capture forces every launch shape to be a
function of geometry rather than of a host-side value — design rule 5, which
costs nothing and would be expensive to reinstate later.

## Vision rides along without touching the text path

Image support (the `--mmproj` tower, M-RoPE, embedding injection) is shaped so
the text path cannot pay for it. The rotary base and the cache slot are two
values in one position buffer with persistent subslice views, so captured
decode graphs bind the rope view once and a text sequence publishes the same
number to both; the per-token M-RoPE kernel shares the scalar kernel's
double-precision angle arithmetic and collapses bit-exactly when t == h == w,
so switching kernels is a dispatch decision, not a numerics decision; and the
prefix cache hashes per-slot content lanes in place of `<|image_pad|>` ids so
identical pad runs from different images cannot cross-hit — the SGLang pad
substitution, strengthened to one lane per slot. Verified by alternating-pair
A/B of the pre-vision and post-vision binaries with vision off (three decode
pairs on one card, nine prefill pairs including order-reversed ones on
another): decode N=3 +0.5% with the new binary winning 2 of 3, prefill at
parity once position-in-session drift (~0.8%, first runner wins regardless of
binary) is controlled for. Numbers in the enabling commits.

## Speculative decode: exact by construction, priced by the verify step

Seven drafters — five model-free n-gram policies (`ngram`, and llama.cpp's
`ngram-simple`, `ngram-mod`, `ngram-map-k`, `ngram-map-k4v`), the trained
MTP head, and the DFlash block drafter — feed one serving mechanism:
per-sequence draft windows, then a single batched verify pass of
`width × (draft + 1)` rows whose acceptance rule is token equality against
the target's own output. A wrong drafter can only waste compute, never
change a token (`serving_speculative_identity` asserts bit-identity for all
of them), so speculation is purely a throughput lever, and the lever's sign
is set by batch width, not by acceptance.

Every drafter at its shipped default, against plain decode on the same
card and prompts — `bench_worker_spec`, context 256, 512 tokens/sequence,
medians of interleaved rounds with the order reversed on alternate ones:

| Drafter | draft cap | N=1 | N=3 |
| --- | --- | --- | --- |
| plain (`none`) | — | 103.7 tok/s | 211 tok/s |
| `ngram-map-k` | 48 | **+20.9%** | −77.8% |
| `ngram-map-k4v` | 48 | **+20.8%** | −78.8% |
| `ngram-simple` | 48 | **+16.3%** | −22.8% |
| `ngram-mod` | 64 | **+11.1%** | out of memory |
| `ngram` | 3 | **+9.2%** | −57.7% |
| `draft-mtp` | 3 | **+1.4%** | −27.2% |
| `spec-dflash` | 3 | −14.8% | −39.5% |

Two mechanisms produce that whole table, and they pull opposite ways.

The verify pass is the prefill-shaped pass at tiny token counts, where
fixed per-pass cost dominates — and at N=1 that cost is nearly flat in the
window. `ngram`'s 4-row window averages 16.72 ms/step against
`ngram-map-k`'s 49-row window at 16.86 ms, both over a 9.64 ms plain step:
twelve times the rows for about 1% more time. Acceptance therefore
converts almost directly into throughput at N=1, and the widest drafters
win — `ngram-map-k`/`k4v` +20.9%/+20.8%, `ngram-simple` +16.3%,
`ngram-mod` +11.1%, `ngram` +9.2% over 103.7 tok/s, six interleaved rounds,
spreads under 0.7% and order bias under 0.2%.

At N=3 the same 49-row window is 147 rows, well past that flat regime,
while plain decode has meanwhile become *cheaper* per token by amortizing
one weight read across three sequences inside a captured graph the verify
path does not use. Every drafter loses there: `ngram-simple` −22.8%,
`ngram` −57.7%, the map pair −77.8%/−78.8% against 210.9 tok/s, and
`ngram-mod` at its own default cannot allocate at all. The
`LLMXABE_NGRAM_GATED` lever isolates the same mechanism from the other
side: the round-gated fallback (accepted tokens ride plain decode steps) at
*identical* acceptance is 1.9× the batched verify at N=3 (166 vs 87 tok/s)
and 8% behind it at N=1 (104 vs 113) — the batched verify wins exactly
where its row count stays near the plain step's and loses where it
multiplies it.

The trained drafters turn the window trade upside down, and that is the
most useful thing measured about them. They verify every step at
near-identical cost to each other, so at a 3-token block their gap is pure
acceptance (2.86 vs 2.44 of a 4-row window: MTP **+1.4%**, DFlash
**−14.8%** at N=1). But their *draft* side is a model pass, not a host
lookup, and it is the half that scales. At the same 4-row verify window
`ngram` costs 16.72 ms/step where `draft-mtp` costs 27.15 and
`spec-dflash` 27.64 — that ~10.5 ms is the drafter's own pass. So widening
is the opposite trade from widening an n-gram window: 4 → 49 verify rows
costs **+0.14 ms/step**, while a 3 → 15 trained block costs **+57.2 ms**
(MTP) and **+52.6 ms** (DFlash). Both collapse: MTP +1.4% → **−55.4%** and
DFlash −14.8% → **−65.8%** at N=1; −27.2% → −67.3% and −39.5% → −54.8% at
N=3. The trained drafters' best configuration is their smallest, and only
MTP at a 3-token block is not a loss. One asymmetry inside the collapse is
worth keeping: at a 15-token block DFlash outlasts MTP at N=3 (−54.8%
against −67.3%, 24.1 vs 15.4 tokens per step), which is block in-fill
still accepting where a chained head has drifted.

Serving therefore defaults to `--spec-type none`. The win is real, and
large, only for single-stream deployments, and there it belongs to the
model-free drafters — the ones whose draft side costs nothing to widen.

---

# WHY NOT

Everything below was built or derived far enough to measure, and rejected.
They are recorded so they are not re-attempted: several have already been
proposed twice.

## Rejected on measurement

| Attempt | Result |
| --- | --- |
| GDN split projection at row tiles 1 and 2 (`uint4` form, N=3) | 78.4 and 46.4 us per qkv/gate call against RT=4's 33.9. RT=1 octuples the warps and the in-flight bytes and is the *worst* of the three, so memory-level parallelism was never the binding constraint — instruction count per byte is, and it falls with RT. |
| GDN split projection, char4 partition + row tile + prefetch | 40.0/36.7 us against the shipped `uint4` form's 33.9/29.3, despite fully-coalesced activation loads and ~2.7x fewer L1 wavefronts per byte on paper. The wavefront model predicted the wrong winner; the wide weight load won anyway. |
| `q8_0` KV cache (llama.cpp side) | −2.5% at depth 0, **−15.5%** at 32K, **−35.8%** at 128K. Turing's in-kernel dequant costs more than the halved traffic saves, and the KV path already ran at ~80% of peak. Keep `-ctk f16 -ctv f16`. |
| Requantize experts Q6_K → Q8_0 | 4–8% for **+30% VRAM**. The dequantization format is not what limits that kernel. |
| Marlin-style register prefetch on the MoE MMA kernels | 20% slower at 512 tokens. `__launch_bounds__` already trades registers for occupancy on purpose; a register pipeline competes with that trade and ptxas spills instead of exceeding the occupancy target. |
| Widening the prefill softmax-rescale tile (`MMA_KEY_TRIPS` 2/4) | A wash then **1.96× slower**. The kernel sits at 252 of 255 registers with zero spill; the wider prefetch arrays spill immediately. |
| `MMA_KOCT=4` (32-key prefill staging tile) | **−30%**. Same mechanism: the cross-tile prefetch arrays grow with the tile and spill. One octet is the largest tile whose prefetch fits beside `o` and `qa` at head_dim 256. |
| Staging `Q` to shared in the decode MMA kernel | 73 registers freed, zero spill, and **37–40% slower** at every depth. Shared grew past the 3-blocks/SM line, and 64 shared loads per warp per trip replaced registers that were free. llama.cpp's own Turing config keeps `Q` in registers here too. |
| Per-warp online softmax without the cross-warp round trip | 255 registers and 104–120 B of spill at both occupancy widths; the shared-memory mitigation collapses to 1 block/SM by arithmetic. Not built past `ptxas`. |
| fp16 `P V` accumulation | 54 registers freed — and **209× over `MMA_GATE`** on the constant-`V` test, growing with depth. With `V` coherent the accumulator grows purely additively between rescales until increments round away entirely; an IID-random-`V` simulation held with 26–30× margin and tested the wrong regime. Also buys nothing: fp16- and fp32-accumulate `m16n8k8` measured within 0.4% on this card. |
| `__expf` in the prefill softmax | 1.075× on attention, and a **real rank-4 ranking error** against llama.cpp's separation. Greedy decoding would not notice; sampling would. |
| `half2` on the decode `P V` accumulator | Fewer registers, fewer instructions, **11.3% slower**. `pack_h2`'s broadcast of the scalar softmax weight added 40 `PRMT` and 16 `F2F` on a value that was already in a register. Instruction counting is retired as a predictor for this kernel. |
| `dp4a` / Q8_1 activation quantization (llama.cpp's `mmvq`) | Passes every isolated per-kernel gate at a bound derived from llama.cpp's own arithmetic, then fails `batch_decode` at **5.1e-1** (asymmetric) and **7.1e-1** (symmetric) against a 5e-3 bound. Round-to-nearest is discontinuous, so it amplifies any upstream residual into expert-selection flips. Re-attempted after the paths became bit-identical and still failed the exact cross-width gate at `4.4e-5`: identical inputs must produce identical codes at every serving width. |
| `ldmatrix.sync.aligned.m8n8.x4` | Both fragment layouts verified correct on hardware, and **0.8% slower**. The kernel is not bound by shared-load instruction count. |
| `mma.m8n8k4` (Turing's "native" shape) | **49.1 TFLOP/s** measured against `m16n8k8`'s 97.6. Half the throughput, not hidden headroom. Ruled out without writing the kernel. |
| Swapping the flash grid's axes for L2 reuse | **−1.7%**. Sibling blocks start together but drift apart faster than 6 MB of L2 spans; only a staging barrier inside one block makes them share. |
| `ATTN_QT` 16, `GQA_KT` 16, `__maxnreg__(84)` | −10%, −14%, −6%. All three trade the second resident block per SM for something worth less than it. |
| Copying llama.cpp's 4-query/64-key outer prefill tile | **~8× slower**, `ptxas` clean at 255 registers with zero spill. Four queries launch four times the blocks and a conservative 64-key softmax serialized through four lanes. Copying tile dimensions does not copy the fragment-resident softmax and combine that make them competitive. |
| Hoisting the GDN Gram kernel out of the chunk loop | Bit-identical, 240 launches → 30, and **0.5% slower**, losing every interleaved pair. In the loop the Gram output hits L2 immediately; hoisted, all eight chunks' squares must coexist in a 6 MiB L2. On this part, launch count is not worth trading locality for. |
| Tiling `gdn_chunk_gram` over target tokens | Nothing. Its keys are 32 KiB per head and sit in L2 — the arithmetic that says a kernel re-reads its input does not say the re-read costs anything. |
| Shared-memory staging on the GDN decode GEMV and the routed-expert GEMV | −3.6% and −0.8%. The re-reads staging removes were L1 hits; staging replaced a hit with a copy and a barrier, and cost a resident block. |
| Shared-staged flat decode GEMVs (`int4` staging, as the LM head uses) | Bit-identical by construction and **1.3–2.4% slower**. These kernels are bound by the **integer pipe**, not by load issue: Q6_K unpack costs ~9 integer ops per element against a ~10-op budget at the streaming roofline. |
| I2F-free unpack (exact-mantissa trick) | Bit-exact, every gate green, **flat**. With both the load-issue and XU-pipe hypotheses dead, the flat decode GEMVs at 473–519 GB/s read as at their practical equilibrium for this quantization on this card. |
| Even/odd MMA accumulator chains | −2% prefill, noise at decode. The compiler's schedule was not accumulator-stalled, and eight more registers on kernels already at ~230 costs more than the chain relief. |
| Skipping the online-softmax rescale when the running max did not move | Bit-identical by construction and **2.8% slower** at 128K prefill. The identity multiplies hid under staged-load latency the schedule pays anyway; the vote-and-branch costs more than the work it skips. |
| llama.cpp's n-gram drafters at N=3, at llama.cpp's own 48-token defaults | **−22.8%** (`ngram-simple`), **−77.8%**/**−78.8%** (`ngram-map-k`/`k4v`) against plain decode's 210.9 tok/s; three interleaved rounds, spreads ≤0.4%. Not an acceptance failure — the map pair fills 22 of 49 rows per sequence — but a row-count one: 3 × 49 = 147 verify rows against a captured 3-row graph step. The same policies win 16–21% at N=1, where the window is free. The map pair's *magnitude* is additionally inflated by `bench_worker_spec` letting a high-acceptance sequence run far past the timed quota while a straggler gates the window; the sign is not in doubt, the exact figure is. |
| Widening the trained drafters to the drafter's trained maximum (`--spec-draft-n-max 15`) | MTP **+1.4% → −55.4%** and DFlash **−14.8% → −65.8%** at N=1; −27.2% → −67.3% and −39.5% → −54.8% at N=3. Three interleaved rounds, spreads ≤1.0%. Acceptance did rise (MTP 2.86 → 3.90 tokens per step at N=1) and was nowhere near enough: a 3 → 15 block adds **+57.2 ms/step** (MTP) and **+52.6 ms** (DFlash) onto a 9.64 ms plain step, because the *drafter* pass scales with the block even though the verify pass does not (4 → 49 verify rows is +0.14 ms). A wider window is free only when the draft side is a host lookup; making a trained drafter pay means making its pass cheaper, not longer. |
| `ngram-mod` at llama.cpp's default `n_max` 64, N=3 | `CUDA_ERROR_OUT_OF_MEMORY` before the first step, in every round. The verify path holds one GDN snapshot ring set per decode slot — 30 layers × (32·128·128 + 8192·3) fp32 × `(drafts + 2)` — which is 4.05 GiB per slot at 64 drafts and 12.15 GiB for three, beside a 29.6 GiB model on a 48 GiB card. 48 drafts (9.2 GiB for three) fits and still loses. The draft cap is a VRAM knob, not only a scheduling one. |
| Tensor cores for routed MoE at N=3 (`MMA_MIN_TOKENS` 8 → 3) | 135.2 vs 147.7 tok/s. Padding a three-row dispatch into MMA fragments and quantizing its activations costs more than the arithmetic recovers. |
| The general fp32 expert tiles at N=3 (`MOE_NARROW_DECODE_MAX` 4 → 2) | 129.6 vs 147.7 tok/s. With the tensor-core result above, this brackets the current N=3 choice: both neighbouring kernel paths are slower. |
| The scalar warp decode kernel at depth | 133.0 vs 148–151 tok/s at 32K N=3. The depth-aware dispatch onto the tensor-core kernel stands. |
| `WPO=4` decode occupancy width | ~4.5% slower than `WPO=2` at 32K N=3, before and after the register-spill fix. Never wins at any depth measured. |
| Decode blocks 48 or 60 at 72 logical splits | 150.5 and 139.5 against 36 blocks' 152–153. 48 divides 72 unevenly and the long blocks set the makespan; 60 spills into a second wave. |
| Three independent N=1 decode graphs on one card | 129.9 vs 147.7 tok/s at 2K. Separate streams repeat weight traffic and contend more than their overlap recovers; serving keeps batched weight reuse. |
| Shared-expert side-stream overlap | Flat at N=3 (no SM or DRAM slack to hide in), **−2.5–3%** at N=1 (the step is launch-latency-bound and two events per MoE layer is pure overhead). |
| Independent GDN streams at prefill | Within thermal drift, reversing across warmed pairs. Removed rather than kept on a cold-card gain. |
| Cross-sequence prefill flattening at *equal* total ubatch | 0.99×. At fixed physical width, batching does not reduce the number of full-model passes. It wins only when the pass gets wider — see WHY. |
| The two-kernel `bm == 1` split, first attempt | −5.2% before `bucket_live` existed: both kernels then walked `sorted_token_ids` and crossed two barriers per bucket to compute `bm`, and only one used the answer. It landed as a win only once `bm` became a table read. |
| GEMV unroll pragmas on the direct-1 helpers | 20% slower on ffn. A register cliff — the standalone GEMV kernel's unroll depth was tuned against a register budget the narrow kernel, which carries the tiled fallback in the same function, does not have. |
| A `KU` unroll hint on the GDN tiled projection | Byte-identical SASS at the width N=3 actually uses; −9% at N=2. `ptxas` already reached the same schedule. Whatever closes that kernel's 28%-of-roofline ceiling must change what ptxas schedules, not hint at a schedule it already finds. |
| Speculative decode as the N=3 lever (n-gram, MTP, DFlash; serving A/B, four interleaved rounds with a reversal) | Every arm loses at N=3: 87.5 / 150.7 / 125.2 tok/s against plain 211.4, despite up to 10.3 accepted tokens per 12-row verify window. The batched verify's fixed cost (~4.8× a plain batch-3 step) outruns the tokens it saves, and the round-gated fallback at identical acceptance also loses (166 vs 211). The measured net wins are n-gram **+9%** and MTP **+1.5%**, both at N=1 only — see WHY. Making the verify pass ride decode's launch machinery instead of the prefill shape is the untried lever. |
| Removing the routed-partial clear | Below run-to-run spread, and reversing. Deleting a defensive correctness aid for a result smaller than host drift is not justified. |

## Rejected on arithmetic, before building

- **TensorRT.** Not installed, and not a drop-in if it were: it consumes ONNX
  or its network-definition API, Q6_K's two-level superblock scales are not a
  TensorRT weight format, and Gated DeltaNet is not a builtin layer. Adopting
  it means writing an exporter *and* plugins that reimplement the kernels this
  project already has.
- **vLLM's Marlin MoE.** It ships a real fused sm_75 tensor-core kernel, and it
  accepts only int4/int8 weights with fp16/int8 activations. Q6_K is not one of
  those formats. Port vLLM's *indexing strategy* onto llama.cpp's *primitives*.
- **Shared-memory double-buffering in the MoE MMA kernels.** The footprint is
  21,760 B and `__launch_bounds__` already asks for three resident blocks —
  `3 × 21,760 = 65,280` of 65,536, with 256 B of slack. Any double buffer
  admits one block per SM. That is the same trade the register prefetch lost,
  3× worse.
- **Split-K in the LM head.** Split-K manufactures parallelism when the output
  dimension cannot fill the machine. The LM head is the opposite: 248,320 warps
  against 2,304 resident, a 107× surplus. Splitting K adds a launch, a
  `vocab × K` partial buffer and a split-dependent summation order for no
  occupancy gain. Asserted in a unit test so it fails loudly if the vocabulary
  ever shrinks.
- **`mma.m16n8k16`.** NVRTC accepts it at `compute_75` and emits PTX; `ptxas`
  then rejects it (`requires .target sm_80 or higher`). NVRTC success is not
  proof of reachability — only PTX→SASS is. Any MMA work hand-decomposes to
  `m16n8k8` / `m8n8k16`.

## Diagnoses that were right about the fact and wrong about the cost

Recorded because the *reasoning* is what misleads, not the number.

- **"The MoE dispatch kernel is single-block."** True, and worth **0.12%** of
  runtime. Parallelized anyway because it was cheap.
- **"The LM head computes logits for all positions."** It never did; it already
  mirrors `get_rows(cur, inp_out_ids)` and runs on one row.
- **"Per-element dequant re-reads the superblock header."** True, and bounded
  small — the Q6_K kernel, whose unpack is far more expensive, is the *more*
  efficient of the two per FLOP. Hand-hoisting it produced **byte-identical
  SASS**: `ptxas` was already doing it.
- **"Decode attention is bandwidth-bound."** Seven hypotheses were tested
  against that assumption before it was checked. A calibration ladder — the
  identical grid and loads with nothing else — reaches **90% of streaming
  roofline**, and adding the dot product costs 3 points. The cost is the
  softmax machinery, not the access pattern.
- **"Sector amplification is costing 4×."** The diagnosis was real and `SASS`
  proved it; the recovery was **1.08×**, because L1 was absorbing nearly all
  the redundant sector requests.
- **Any experiment that perturbs numerics also perturbs expert routing**, and
  therefore weight traffic. Halving an inner loop made a pass 12% faster and
  stripping scale multiplies made it 10% slower — neither measured anything.
  The valid ablation runs the staging loop *twice*, writing the same bytes to
  the same addresses, so the output is bit-identical and routing cannot move.
- **A negative result taken during an unrelated regression is not a
  measurement.** One change was recorded as a 57% regression while a decode bug
  was live; against a clean baseline it is a small improvement.
- **`ptxas` allocates across a wider scope than the edit.** Adding a branch to
  one device function moved an unrelated `__global__` function's register count
  from 80 to 126, and removing a branch from two near-identical sibling kernels
  moved their allocations in opposite directions. Register math is a proxy;
  `cuobjdump -sass` diffing every kernel against its stated blast radius is the
  check.

---

# Ceilings that are closed by design

These are not open work items. Each is a trade the project made on purpose,
verified, and would have to be *un*-made to reopen.

**Deep decode, against llama.cpp's own kernel.** llama.cpp decodes through its
tensor-core prefill kernel at this geometry (head_dim 256, GQA 8, Turing) — it
never runs a scalar per-key softmax loop at batch 1 here. Its edge over this
engine decomposes into three things this engine forbids:

1. A **single fp16 rounding of `Q`**, which issues keys at 2× this kernel's
   rate. Forbidden: that exact rounding broke the 1e-5 gate at `n_keys = 61`,
   which is why the `q_hi`/`q_lo` split exists.
2. **fp16 `VKQ` accumulation**, which is not a throughput lever (measured
   within 0.4%) but a *capacity* one — a `half2` accumulator packs two values
   per register, which is what makes its barrier-free per-warp design fit
   Turing's register file. With fp32 accumulators, every route to that
   structure costs registers or shared-memory occupancy this card cannot fund
   at once, measured shut from four directions.
3. **`mmvq`/dp4a activation quantization**, rejected above on the serving
   contract rather than on accuracy in isolation.

**Deep prefill (128K).** `attn_flash_causal_mma` is 60–72% of a deep chunk's
kernel time and is latency-bound at one resident block per SM. Its barrier
density is ~5× llama.cpp's (2 block-wide syncs per 8-key octet against 3 per
64-key tile), which is real, sized, and the same register-bound lever already
tried: the kernel is at 252/255 registers with zero spill, so widening the tile
trades occupancy away rather than buying anything. Closing it needs a
structural, register-neutral rewrite of the staging and synchronization
pattern, not a parameter change.

**Aggregate concurrency.** The per-sequence state term — 131.7 MB/token of GDN
recurrent and conv state — **never amortizes at any batch width**, because
there is nothing to share. llama.cpp pays it identically, which is why its own
efficiency holds flat at ~41% of the concurrency-aware roofline from one
sequence to three rather than climbing. The batching win either engine can
capture is the routed-expert term's, and at N=3 that term is on the steep part
of its own curve: three sequences touch `D(3) = 23.26` distinct experts against
one sequence's 8.00, so amortization is partial by construction, not by defect.

The named workstreams that would legally reopen (1) and (3) are recorded in
[OPTIMIZATION.md](OPTIMIZATION.md), not here.

---

## Reproducing

```sh
# llmxabe, N=3 decode and prefill
CUDA_VISIBLE_DEVICES=1 LLMXABE_BATCH_N=3 \
  ./target/release/bench_decode_batch 32768 16
CUDA_VISIBLE_DEVICES=2 LLMXABE_PREFILL_SEQUENCES=3 LLMXABE_BENCH_CHUNK=6138 \
  LLMXABE_BENCH_N=2046 cargo run --release -p xabe-engine --bin bench_forward

# llama.cpp, same card, alternating with the above
CUDA_VISIBLE_DEVICES=1 llama-batched-bench \
  -m Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf \
  -ngl 99 -sm none -fa on -b 4096 -ub 4096 -ctk f16 -ctv f16 \
  -c 131072 -npp 32768 -ntg 32 -npl 3
```

Narrow harnesses, for anything smaller than a whole-pass change:
`bench_attention` (`LLMXABE_ATTN_CHUNK=1` for decode shape), `bench_moe`,
`bench_moe_mma`, `bench_mma`, `bench_decode`, `bench_worker_decode`,
`profile_forward`, and `audit_batch_divergence` for per-layer
batch-vs-single-stream divergence.

## Not measured

- Any hardware counter. Every efficiency figure here is necessary-work ÷
  measured-time, which understates real traffic and therefore understates how
  far off peak a kernel is.
- Multi-GPU aggregate under real routed traffic. The three-card figures are
  one session per card, not three sessions per card.
- Output *quality* over long runs under any of the precision trades; they were
  verified against gates and short samples, not long-horizon generation.
- Vision serving throughput. The tower's resident VRAM was measured
  (1152 MiB at load, ~1320 MiB after first use, per worker — see
  [DEVELOPMENT.md](DEVELOPMENT.md)), and text-path preservation was A/B'd
  with vision off; encode latency and image-heavy throughput were not
  benchmarked.
- Sustained thermal behaviour; all runs are short.
