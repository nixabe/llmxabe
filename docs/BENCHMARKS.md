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
| Second model | `Qwen3.8-27B-UD-Q8_K_XL.gguf` (`qwen35`, dense) — 29.30 GiB, 26.90 B params |

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
  and has manufactured a phantom regression at 65K prefill. **Numbers need
  the whole box, not just the card.** A *correctness* run on another GPU is
  enough to poison a pair, and not because it competes for SMs you are not
  using: the CUDA context spin-up and the NVRTC compile around it take host
  CPU. Two sub-second test runs on a neighbouring card cost one measured pair
  in this file's history. Outside a pairs window a sibling can use the other
  cards freely; inside one, nothing else runs anywhere, builds included.
- **Re-measure after a rebase, not before.** A branch that measured a win
  against its own base can be net negative once it sits on a moved `main`,
  and the arithmetic does not warn you: +1.1% over `ac341cc` landed on a
  `main` that had meanwhile gone −5.0%, which three interleaved pairs found
  only because the rebased tree was benched against the original baseline
  rather than assumed equivalent to it.
- **CUDA events, not `Instant`**, for anything inside a pass. Host clocks
  measure enqueue latency.
- **Kernel-level first, then end to end.** `bench_attention` (~6 s per A/B)
  and `bench_dense_ffn` (~10 s, and it prints the card's measured streaming
  ceiling beside every row) exist because whole-forward A/Bs are expensive
  enough that the honest response to a small change was to not measure it.
  **Then confirm in the model.** A narrow bench and a real step disagree about
  cache state, and at least one change has won every interleaved pair of the
  former while losing the latter — see `ld.global.cs` in the WHY NOT list.
- **`ncu` does not work on this host** (`ERR_NVGPUCTRPERM`). The working
  substitutes are `nsys`/`nvprof` timelines, `nvcc -Xptxas -v` and
  `cuobjdump -sass` on an extracted kernel, roofline arithmetic from the GGUF
  tensor directory, and ablation. Do not spend time re-trying `ncu`.
- **Rebuild through Cargo.** Invoking `target/release/<bin>` directly does not
  ask Cargo whether it is stale; this has twice produced numbers that were a
  stale binary rather than a result.
- **Compare against llama.cpp's best settings, never its defaults.** Measuring
  against `-ub 512` inflated three separate claims before the rule was named.
- **A sweep decides nothing, and one three-pair set decides less than it
  looks.** One run per point does not resolve anything below roughly 1.5% on
  this host, so a sweep is for choosing *where* to put pairs and never for
  choosing a value. The router's expert tile read 208.0 / 208.7 / 207.6 /
  207.6 across `ET` 8/4/2/1 — non-monotonic, and read by two agents
  independently as drift rather than a knee. Three interleaved reversed pairs
  then had `ET = 4` winning every one at +1.31%, which was believed and
  written up. Re-measured against a *pinned baseline built from the same
  commit* rather than a stale binary, the same change reads **+0.24 / −0.19 /
  −0.05%** at N=3 and the sign flips inside the set. The +1.31% was an
  artifact and there is no effect to find. Non-monotonic single runs mean
  *unmeasured*; three pairs against a drifting box mean *barely measured*.
  The floor is set by the drift, and the drift is what you have to show.
- **Report the pairs, not the mean.** A three-pair set whose arms are
  208.7/206.3, 207.9/201.4 and 207.3/205.3 has a mean of +1.79% and an honest
  reading of about +1.1%: the 201.4 is 2.4% below its own siblings, so it is a
  bad reading of the baseline rather than a good result. A mean cannot tell
  those apart and a list of pairs can.

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
103.2 tok/s at 2K. Aggregate across three cards, one session each, is a
deployment figure and **not** the per-instance N=3 target — do not cite it as
a replacement claim.

### Capability

| | |
| --- | --- |
| Max context, one card | 131,072 positions, peak ~35.6 GiB of 47.27 |
| KV cache | binary16, 20 KiB/token over 10 attention layers |
| Recurrent state | fixed ~2 MiB per GDN layer per sequence, independent of depth |
| Parallel sequences | batched decode and flattened batch prefill at N=1–8 |
| Speculative decode | seven drafters (`ngram`, `ngram-simple`, `ngram-mod`, `ngram-map-k`, `ngram-map-k4v`, `draft-mtp`, `spec-dflash`), each bit-exact against plain decode; off by default. On `qwen35` `draft-mtp` is now a win at N=1 **and** N=3; it stays opt-in because on the shipped `UD-Q8_K_XL` file its extra layer and per-sequence draft cache do not fit beside three 128K-capable caches. See WHY. |

## A second `qwen35moe` checkpoint

`ornith-ai/Ornith-1.5-35B-A3B` is a different finetune of the same
architecture, requantized to the shipped mix (see
[MODEL.md](MODEL.md#serving-a-second-qwen35moe-checkpoint)). It is here
because it is the only evidence that the standing above is a property of the
*engine* and not of one file.

Same method as the table above — one card, N=3, llama.cpp at its own best
settings, three interleaved alternating reps, prefill on GPU 2 and decode on
GPU 1. llama.cpp's column is the faster of `-ub 2048` and `-ub 4096` per cell.

| cell | Ornith | sd | llama.cpp | margin | Qwen3.6 above |
| :--- | ---: | ---: | ---: | ---: | ---: |
| prefill 512 | 3,322.8 | 0.00% | 2,910.3 | **+14.2%** | +9.1% |
| prefill 2K | 3,869.9 | 0.04% | 3,251.4 | **+19.0%** | +19.3% |
| prefill 8K | 3,563.2 | 0.17% | 3,180.0 | **+12.0%** | +14.5% |
| prefill 32K | 2,803.4 | 0.06% | 2,649.3 | **+5.8%** | +5.1% |
| prefill 65K | 2,218.4 | 0.11% | 2,152.3 | **+3.1%** | +3.9% |
| prefill 128K | 1,567.1 | 0.20% | 1,555.9 | **+0.7%** | +0.8% |
| decode 2K | 207.8 | 0.05% | 187.3 | **+10.9%** | +11.4% |
| decode 32K | 158.6 | 0.22% | 151.6 | **+4.6%** | +5.0% |

**Every cell won all three interleaved pairs**, which is the claim that
matters at 128K prefill — the thinnest margin here, as it is for Qwen3.6, and
one this file elsewhere says stands only on alternating pairs: 1,567.6 v
1,555.9, then 1,570.0 v 1,555.8, then 1,563.7 v 1,553.0.

Two readings of the right-hand column, and only one of them is safe. The
margins agreeing to within half a point on five of eight cells says the two
checkpoints are interchangeable to this engine. It does **not** say either
model moved: those Qwen3.6 numbers were taken on other days, and this file's
own provenance note says its 512–32K prefill rows predate the per-sequence
prefill fork.

The 512 row is llama.cpp's noise, not ours. It ranged 2,770.8–2,910.3 across
the ladder while Ornith sat at 3,322.8 with no measurable spread, and `-c` was
ruled out as the cause by direct control — 2,843.8 mean at `-c 4096` against
2,852.3 at the model default of 262,144. The margin is quoted against
llama.cpp's *best* of three; against its mean it reads +16.6%.

**The bar was re-checked and holds.** The decode 2K row's opponent was
re-measured on a later day, same card, same settings: **186.22 tok/s** at
`-ub 2048` against the 187.3 recorded above, which is inside this host's own
run-to-run spread. That matters beyond one row — llama.cpp not having drifted
is what keeps the other seven cells on live footing rather than stale, and it
is the cheap half of any head-to-head because that side runs whether or not
our column has moved. The llmxabe column was **not** re-measured into this
table on that day: the run that would have supplied it was taken on a tree
carrying an unrelated 5% regression, so it was discarded rather than
published. Re-checking the opponent is worth doing on its own.

## Qwen3.8-27B (`qwen35`)

The dense sibling landed after the standing above was taken, and **nothing in
this section is comparable to it**: a different model and a different
arithmetic mix. Its own baseline is the same `llama-batched-bench`, same card
(GPU 1), alternating processes, three interleaved reps.

The dense model is the smaller file and reads **15× the feed-forward weight
per token** (17.11 B parameters against 35B-A3B's 1.13 B) and 3.2× the KV. No
expectation carries across from the table above — and none of it did.

**Read the file column before the margin column.** On this model the
quantization of the *file* moves the result more than anything either engine
does, and a margin quoted without it is meaningless. Two files, both measured
same-card, three interleaved reps, llmxabe's spread under 0.2% and
llama.cpp's up to 3% at N=3:

| cell | file | llmxabe agg | per slot | llama.cpp agg | per slot | margin |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| prefill 512, N=1 | UD-Q8_K_XL | 361.6 | 361.6 | 681.8 | 681.8 | −47.0% |
| prefill 512, N=1 | all-Q8_0 | 666.2 | 666.2 | 769.3 | 769.3 | **−13.4%** |
| prefill 512, N=3 | UD-Q8_K_XL | 370.0 | 123.3 | 756.4 | 252.1 | −51.1% |
| prefill 512, N=3 | all-Q8_0 | 704.2 | 234.7 | 853.1 | 284.4 | **−17.5%** |
| decode, N=1 | UD-Q8_K_XL | 18.0 | 18.0 | 18.05 | 18.05 | −0.3% |
| decode, N=1 | all-Q8_0 | 19.2 | 19.2 | 19.40 | 19.40 | **−1.0%** |
| decode, N=3 | UD-Q8_K_XL | 46.7 | 15.6 | 43.36 | 14.45 | **+7.7%** |
| decode, N=3 | all-Q8_0 | 49.1 | 16.4 | 48.15 | 16.05 | **+2.0%** |
| decode, N=1, `draft-mtp` | all-Q8_0 | 35.1 | 35.1 | 19.40 | 19.40 | **+81%** |
| decode, N=3, `draft-mtp` | all-Q8_0 | 50.0 | 16.7 | 48.15 | 16.05 | **+3.8%** |

The two `draft-mtp` rows are **not the default** and are not a like-for-like
kernel comparison: they are this engine running the file's own trained
next-token-prediction head as a drafter, with a bit-exact verify, against
llama.cpp running the same file the only way it can. Upstream discards that
head — `model has unused tensor blk.64.nextn.eh_proj.weight -- ignoring`, on
every load — so there is no setting that gives it the same option. What the
rows are for is the size of the gap between a decode step that reads 27 GB
for one token and one that reads it for 2.86. See the speculative section
for why it stays opt-in.

`UD-Q8_K_XL` is the shipped Unsloth file: Q8_0 everywhere except
`output.weight` and every `attn_q`/`attn_k`/`attn_v`, which are bf16.
`all-Q8_0` is that file requantized with `llama-quantize --allow-requantize
... q8_0`, which is the same weights with the bf16 tensors folded down —
29.30 GiB becomes 27.05. **Its numerics are unverified**: the greedy-agreement
table below was run on the shipped file, and no differential or agreement
check has been run on the requant. It is a speed measurement only.

**Per slot is uniform by construction on both sides**, not three separately
observed rates: a batched step advances all N sequences through one set of
launches, so each slot's rate is exactly the aggregate over N. These columns
answer "what does one user see while N are served", not "do the slots differ".
The second question needs a running server under mixed arrival, which is the
serving section below and is not comparable to this table.

What the two rows of each pair say together:

- **Decode wins at N=3 and is at parity at N=1; prefill is not.** Both N=3
  cells are clear of llama.cpp, by 2.0% and 7.7%. Both N=1 cells sit just
  under it, by 1.0% and 0.3% — the second is inside llama.cpp's own spread
  across the three reps, the first is not. Every prefill cell loses, by 13%
  on the file both engines read best and by half on the shipped one. Prefill
  is where this model's remaining gap lives, and it is an arithmetic gap
  rather than a bandwidth one — see "Where the dense decode step goes" below
  for why the N=1 decode cell is not going to be closed by a faster kernel.
- **The bf16 tensors still cost prefill, and no longer cost decode.** Folding
  them to Q8_0 moves prefill 360 → 661 at N=1; the per-tensor int8 gate
  recovered part of that in the engine, and the rest is `attn_q`/`k`/`v`
  having no integer tensor-core path at bf16 at all. At decode the same
  tensors cost 6.5%, which is the extra bytes and nothing else.
- **The decode margins have moved several points across two revisions of
  this table, and llama.cpp has not moved.** Every point of it is on this
  side: the split int8 layout stopped widening the file's fp16 scales to
  fp32 (WHY, "Layout"), the GEMV's row tile is chosen per token width, its
  activations are read as halves, and the block's fp16 narrowing and residual
  add were folded into launches that were already running. llama.cpp's column
  is a same-hour alternating re-run in every row, three reps, and its spread
  is 0.4% at N=1 and under 0.6% at N=3 on both files.

And llama.cpp's outright best on this model is neither file: see "What the
quantization is worth" below. Do not quote any row here as a claim about the
engine as a whole, and do not let them soften the MoE standing above.

llama.cpp at `-b 2048 -ub 2048 -fa on -ctk f16 -ctv f16 -sm none`, `-npl 1`
or `-npl 3`, `-npp 512 -ntg 64`. Ours is `bench_forward`'s 512-token column (a
cold full forward, comparable to `pp`), `bench_decode` at N=1 and
`bench_decode_batch`'s `batch 3` row — which carries its own `single_stream`
baseline in the same process, and it agrees with `bench_decode` to 0.3%. Peak
VRAM 39.39 GiB of 47.27 at N=3 prefill on the shipped file, 37.39 on the
requant — 1.4 GiB below the earlier revision of this table, because the split
int8 layout no longer widens the file's scales.

### `-ub` is not a knob on this workload

Measured, because "llama.cpp's best settings" is a rule this project has been
burned by three times. Three reps each, `-npl 1,3 -npp 512 -ntg 64`:

| llama.cpp config | PP N=1 | TG N=1 | PP N=3 | TG N=3 |
| :--- | ---: | ---: | ---: | ---: |
| default `-c`, `-ub 2048` | 693.1 | 18.05 | **744.7** | **43.98** |
| `-c 6144 -ub 2048` | 692.6 | 18.05 | 741.6 | 42.33 |
| `-c 6144 -ub 4096` | 692.0 | 18.05 | 742.9 | 43.89 |
| `-c 6144 -ub 512` | 692.2 | 18.03 | 674.3 | 42.65 |

`-ub 4096` changes nothing, and it cannot: at `-npp 512 -npl 3` the physical
batch is 1536 rows, already under the 2048 cap, so raising the cap has nothing
to raise. (At the default `-c` it does not even fit — llama.cpp reserves the
full 262K context, and a 4.04 GiB compute buffer will not go beside ~29.3 GiB
of weights and ~16 GiB of KV. Capping `-c` makes it fit and still buys
nothing.) `-ub 512` is the only setting that moves anything and it *loses* 9%
of N=3 prefill by splitting 1536 rows into three ubatches. Decode ignores `-ub`
entirely — it is one token per slot. The `-b 2048 -ub 2048` baseline above is
therefore llama.cpp's best of these, not its default.

### Speculative decode: what the width band was hiding

This section used to end with an experiment it had not run — "instantiating
`dense_proj_split_t5..t16` would test whether the N=3 result inverts too;
that is unmeasured and is the obvious next experiment, not a claim." It was
run, and it inverted.

The MoE model's finding — "verify-step cost sets the sign; a net win at N=1
only" — transfers in *shape* but not in size, because dense decode is at the
memory wall and a verify step reads the same 27 GB whether it checks one
token or nine. `bench_worker_spec` on the `all-Q8_0` file, GPU 1, context
256, 512 tokens per sequence:

| drafter | N=1 tok/s | tokens/step | N=3 tok/s | tokens/step |
| :--- | ---: | ---: | ---: | ---: |
| none | 19.2 | 1.000 | 49.1 | 3.000 |
| `ngram` | 20.2 | 1.101 | 23.5 | 4.152 |
| `draft-mtp` | **35.1** | 2.860 | **50.0** | 9.350 |

**`draft-mtp` is worth +83% at N=1** — far above the +21% the same class of
change bought on `qwen35moe`, and for the reason the roofline predicts. What
changed is N=3, which used to lose 7% and now wins. `ngram` still loses
there, and heavily — it drafts on 1.101 tokens a step at N=1, so at N=3 it
buys a twelve-row verify pass with almost no acceptance to pay for it.

Two constants were holding it down, and neither was the drafter's:

- **The FFN's GEMV/GEMM crossover.** A verify step presents
  `N x (1 + drafts)` rows to the feed-forward block — twelve at N=3 — and
  above four rows that fell onto `shared_expert_mma`, the GEMM that stages a
  64-token tile and discards most of it. Carrying the GEMV to sixteen tokens
  is 1.374 -> 0.974 ms a layer at twelve, and took N=3 from 42.2 to 48.2.
- **The prefill attention kernels' query-axis parallelism.** At twelve query
  rows the GQA-shared kernel launches `kv_heads` blocks — two, on a 72-SM
  card — and scans the whole window with them, 433 us a layer. Running those
  rows through the flash-decode split instead, which carries its parallelism
  in the key axis, took N=1 from 32.0 to 35.5 and N=3 from 48.2 to 50.5.

So the honest reading was never "speculation loses at N=3 on this model". It
was that a verify pass is a *shape* — a handful of rows against a deep
window — that both the decode kernels and the prefill kernels were written
to exclude, and once something serves that shape the N=3 result follows the
N=1 one.

**It is still off by default, and the reason is memory, not speed.** The
head is one more transformer layer of weights and one more draft cache per
resident sequence. On `all-Q8_0` that fits; on the shipped `UD-Q8_K_XL`,
which is 2.25 GiB larger, `bench_worker_spec` at N=3 dies in the Gated
DeltaNet state allocation with 34.99 GiB free after the weights. Turning it
on by default would trade the 128K-at-N=3 capability for the decode number
on one file, and it would do it silently, because admission (AGENTS.md
rule 4) does not yet know a drafter changes the per-sequence reservation.
That is the work this would need, and it is scheduler work, not kernel work.

### A whittled MoE is the structural version of requantizing

`Qwen3.8-Whittle-MoE-27B-A17.8B` (community, `research-preview`) is the same
64-layer, 5120-wide skeleton with the 17,408 dense MLP replaced by 64 experts
of width 192, 16 active, plus a 5120 shared expert — 26.9 B parameters of
which ~17.8 B are read per token. It reaches the same lever as a requant, by
reading fewer weights rather than smaller ones. llama.cpp, same card, three
interleaved reps, everything Q8_0 except Qwen3.6:

| model | GiB | PP N=1 | TG N=1 | PP N=3 | TG N=3 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-27B dense (`all-Q8_0`) | 27.05 | 792.3 | 19.25 | 851.9 | 48.37 |
| Qwen3.8-Whittle-MoE A17.8B | 26.71 | 665.5 | **25.73** | 889.5 | 51.51 |
| Qwen3.6-35B-A3B (Q6_K) | 30.36 | 2101.7 | **102.57** | 2912.1 | **194.19** |

**+33.7% decode at N=1** for −16% prefill, shrinking to +6.5% at N=3 as three
tokens activate the union of more of the 64 experts. It is *less* bandwidth
efficient than the dense model (72% of peak against 79%) and wins only by
reading ~32% fewer bytes.

And the row that decides deployment: **Qwen3.6-35B-A3B is 4x Whittle and 5.3x
the dense model on decode, and 3x on prefill.** A17.8B is still 17.8 B active
where A3B is 3 B. For serving several sessions on one card the routed model is
not a close call, and no variant of the 27B family changes that ranking.

This engine cannot load the Whittle file. It declares `general.architecture =
qwen35moe` and every tensor is Q8_0 or F32 — no bf16, no k-quant, so every
kernel it needs exists — but `ModelConfig::for_architecture("qwen35moe")`
returns the hardcoded Qwen3.6-35B-A3B geometry and the schema then fails on
about a thousand shapes. **`general.architecture` names a family, not a
model**, and this file is the counterexample: three files now claim two
architecture names across three geometries. Every field a config needs is
*declared* in its metadata (`block_count`, `embedding_length`, `expert_count`,
`expert_used_count`, `expert_feed_forward_length`, the four `ssm.*` keys), so
deriving one would be reading rather than inferring from tensor shapes — but
that is a design change, and the failure today is a wall of `ShapeMismatch`
where it should be one line naming the geometry.

### What the quantization is worth, and where the ceiling is

Decode reads the whole model once per token, so bytes per token *is* the
decode budget. From the file's own tensor directory, per decoded token:
`DenseFfn` 17.198 GiB + `Projections` 8.363 + `LmHead` 2.368 + norms =
**27.93 GiB = 29.99 GB**. On a 672 GB/s card that is 44.6 ms, so **22.4 tok/s
is the theoretical ceiling** at N=1 and neither engine is far off it:
llama.cpp's 18.11 is 542 GB/s (80.7% of peak), our 17.2 is 516 GB/s (76.8%).
This is why the N=1 decode margin is small and why it will stay small — both
engines are streaming the same 30 GB against the same wall. For contrast the
MoE model reads 2.86 GB/token, which is the whole reason it decodes at ~100
tok/s single-sequence; no expectation transfers between the two.

Which makes requantizing the only large decode lever. llama.cpp, same card,
three interleaved reps per file:

| file | GiB | PP N=1 | TG N=1 | PP N=3 | TG N=3 | achieved BW at N=1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| UD-Q8_K_XL (shipped) | 29.30 | 689.7 | 18.11 | 746.2 | 44.09 | 542 GB/s |
| all-Q8_0 | 27.05 | 779.0 | 19.41 | 837.2 | 48.22 | 538 GB/s |
| Q6_K | 20.89 | 669.3 | 22.13 | 674.0 | 52.64 | 463 GB/s |
| Q4_K_M | 15.66 | 714.1 | **29.51** | 718.2 | **60.11** | 453 GB/s |

**+63% decode from Q4_K_M**, and the k-quants get it while *losing* streaming
efficiency — 453 GB/s against Q8_0's 538, because unpacking a k-quant costs
integer-pipe work. They win anyway by reading half as much. That is the same
bound the WHY list records for the MoE decode GEMVs, seen from the other side.

**We cannot load either k-quant.** Both are refused at build by name rather
than breaking: ``token_embd.weight` is q6_K, this pass unpacks q8_0``. The
dense FFN loader already accepts Q6_K; what does not is the embedding gather,
the LM-head GEMV family (`HeadFormat` is Q8_0 or bf16) and the attention
projections. So llama.cpp's best configuration on this model is one this
engine has no answer to, and the margins in the standing table above are
like-for-like margins at a file we can both read — not a claim about the
fastest way to serve Qwen3.8-27B on this card. Today that is llama.cpp at
Q4_K_M.

### Where the dense decode step goes, and what the card can actually stream

The dense model's decode step is one streaming problem with a small tail, and
the only way to read the numbers below is against a measured ceiling rather
than the pin rate. `bench_dense_ffn`'s calibration kernel — a fully coalesced
`uint4` read of exactly the resident weight bytes, no arithmetic — streams
**604.5 GB/s, 90.0% of the 672 GB/s pin rate** (sd 0.43%). That is the number
a kernel on this card is allowed to be compared against.

One `bench_decode` step on `all-Q8_0` at N=1, from an `nsys` capture with
`--cuda-graph-trace=node` (decode replays a captured graph, and without that
flag none of it is attributed). Weight bytes are counted from the GGUF tensor
directory, not inferred from the time:

| stage | launches | ms | weight bytes | GB/s | of 604.5 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| dense FFN GEMVs | 192 | 31.70 | 18.18 GB | 573 | 95% |
| GDN projections | 144 | 11.07 | 5.85 GB | 528 | 87% |
| attention projections | 64 | 3.36 | 1.78 GB | 530 | 88% |
| LM head | 1 | 2.23 | 1.35 GB | 606 | **100%** |
| GDN alpha/beta gates | 48 | 1.02 | 0.03 GB | 25 | 4% |
| everything else | ~596 | 2.60 | — | — | — |
| **step** | ~1,045 | **51.98** | 27.19 GB | 523 | **87%** |

Three things follow, and together they say where a win on this model is and
is not.

- **The LM head is at 100% of what this card streams** — 1.35 GB in 2.23 ms
  is 606 GB/s against a 604.5 GB/s calibration. It is the cleanest available
  proof that the ceiling is the ceiling. The three GEMV families beside it
  are at 87–95%. Weighted together the streaming kernels are at 97% of
  calibration; the "13% off peak" that the step total reads as is almost all
  the tail, not the GEMVs.
- **Both engines are against the same wall.** llama.cpp's 19.40 tok/s on this
  file is 51.55 ms for the same 27.19 GB, 528 GB/s, 87% of the same ceiling.
  The 1.0% between us is 0.43 ms of launch tail on a 52 ms step. It is not a
  bandwidth difference and it will not be closed by a faster GEMV.
- **What is left is the tail** — 2.60 ms across ~596 launches that move almost
  no weight, 5.0% of the step. The gates are 1.02 ms of it for 30 MB, because
  one warp per head is 48 warps on a 72-SM card. Every attempt to recover that
  particular millisecond is now in the WHY NOT list; the ones that worked
  elsewhere in the tail are single fused launches worth ~0.13 ms each.

The way past the wall is therefore not a kernel. It is reading those 27.19 GB
for more than one token, which is what the speculative section is about.

An earlier version of this table had one `attention projections + LM head`
row, because the two share an entry point — `GatedAttentionBlock` runs its
four projections through `LmHeadKernels::forward`, so `lm_head_gemv_b1`
appears 65 times a step. Splitting them by duration histogram is what showed
the head at 606 GB/s; the merged row read as 94% and hid both halves.

### What the FFN residency is worth

The one real decision in the dense port, measured rather than argued. One
layer's three matrices are 267 M elements and 271 MiB in *either* layout —
the split form keeps the file's own fp16 scale, so it is a re-layout at the
same 1.0625 bytes an element — and across 64 layers holding **both** is
16.9 GiB twice over on a card already carrying 11.8 GiB of arena. It does not
fit — the first attempt died on `CUDA_ERROR_OUT_OF_MEMORY` — so it is one or
the other, and then the kernel follows from the residency.

(The split layout used to be the larger of the two, at 287 MiB a layer,
because the repack widened those scales to fp32 for no precision. That cost
is gone; the rows below were measured before it was, and their *ranking* is
what they are kept for.)

| residency | prefill 512 | decode | note |
| :--- | ---: | ---: | :--- |
| Q8_0, fp32 `moe_shared_ffn` everywhere | 76.2 | 15.74 | 9.1× behind llama.cpp on prefill |
| split int8, `shared_expert_mma` everywhere | 321.3 | 9.17 | prefill 4.2×; decode −47% |
| split int8, GEMV at ≤ 16 tokens (**shipped**) | 317.5 | 17.26 | both |

The first two rows are single runs on the same card, taken to choose between
the three; only the shipped row is the three-pair figure from the table above.

The middle row is the lesson. `shared_expert_mma` is a GEMM: at one token it
stages a 64-token tile and throws 63/64 of it away, and the weight bytes are
the same either way — what it loses is the *shape* it reads them in, 63.5 ms
becoming 109.0. The fix was not a new kernel but an existing one:
`GdnBlock`'s split-layout projection GEMV already reads exactly this layout at
exactly these widths, and it is the form this project measured as the fastest
at one token. The dense block runs a copy of it.

That copy also made decode the *accurate* path rather than the fast-and-loose
one: the GEMV dequantizes the weights and multiplies in fp32, where the GEMM
quantizes activations to int8. Against the CPU reference, `max_abs` 3.36e-4
and cosine 1.000000 for the GEMV against 6.45e-2 and 0.999997 for the GEMM.

### What int8 activations cost in the generated text

Stated because it is the one place the dense port trades exactness for speed,
and because "it produced plausible text" is not a result.

Greedy (`--temp 0 --top-k 1`) against `llama-completion` on the same file, 160
tokens:

| prompt | Q8_0 residency | shipped residency |
| :--- | :--- | :--- |
| "The capital of France is" | identical, 717/717 chars | identical, 690/690 chars |
| a B-tree explanation | identical, 717/717 chars | diverges at char 596 of 710 |

So the fp32 path reproduces llama.cpp character for character on both, and the
shipped one flips a near-tie about 135 tokens in on one of them — the two
continuations at that point are both correct English about B-trees. The
divergence comes from the *prefill* pass, which is the only one that quantizes
activations; it is one constant (`DENSE_REPACK_INT8`) away, at a quarter of
the prefill.

## Serving concurrent sessions

The numbers above are single-sequence kernel throughput against llama.cpp.
These are the *server*: real HTTP requests through `/v1/chat/completions`,
streamed, measured end to end. They are not comparable to the standing table
and must not be quoted against llama.cpp.

Measured with `bench_server.py` (see Reproducing): every prompt's length is
read from the response's own `usage.prompt_tokens` rather than estimated, and
each cell's prompts carry a unique leading token so nothing is served from the
prefix cache. Both precautions exist because their absence produced wrong
numbers here — see WHY NOT. Prefill rate is total prompt tokens over time to
the *last* first-token; decode per-slot is the reciprocal of median
inter-token latency. Trials agree to within 0.5% except where noted.

`--spec-type none -c 405504 -s 3 -tb 4096 -pc 2048`, three slots per card.

### One card, prefill

Aggregate is the card's total; per-slot is what one session sees.

| depth | 1 session | 2 sessions | 3 sessions |
| :--- | ---: | ---: | ---: |
| 16K | 2,409 agg / 2,409 slot | 2,350 / 1,175 | 2,290 / 763 |
| 32K | 2,191 / 2,191 | 2,196 / 1,098 | 2,154 / 718 |
| 64K | 1,785 / 1,785 | 1,784 / 892 | 1,709 / 570 |
| 128K | 1,310 / 1,310 | — | 1,310 / 437 |

Aggregate is flat across session count: adding sessions divides the card, it
does not cost it. The one visible dip — 16K at three sessions, −4.9% — is the
kernel's own price for interleaving three sequences (measured separately at
2,795 → 2,656 tok/s, ratio 0.95), not scheduling overhead. At 128K the
aggregate is identical to within 0.2% at one and three sessions.

### One card, decode

| depth | 1 session | 2 sessions | 3 sessions |
| :--- | ---: | ---: | ---: |
| 16K | 63.6 agg / 64.1 slot | 95.8 / 48.6 | 117.0 / 39.7 |
| 32K | 69.8 / 70.5 | 94.9 / 48.4 | 113.4 / 38.7 |
| 64K | 61.9 / 62.7 | 79.3 / 41.5 | 96.1 / 33.3 |
| 128K | 51.6 / 52.4 | — | 71.5 / 26.1 |

Decode aggregate *rises* with session count — 63.6 → 117.0 at 16K — because
concurrent sequences share one weight read per step. Per-slot falls, as it
must; the batched read is what makes the aggregate grow anyway.

### Three cards, the nine-session target

| depth | 3 sessions | 6 sessions | 9 sessions |
| :--- | ---: | ---: | ---: |
| 16K prefill | 6,803 agg / 2,268 slot | 7,029 / 1,171 | 6,101 / 678 |
| 64K prefill | 5,290 / 1,764 | 5,281 / 880 | 5,071 / 564 |
| 16K decode | 202.3 / 68.4 | 297.4 / 52.2 | 368.6 / 42.5 |
| 64K decode | 193.7 / 66.8 | 244.4 / 44.0 | 304.0 / 35.0 |

The nine-session cells are the noisiest on this host: their two trials spread
6.1% on 16K prefill and 3.9% on 64K decode, against 0.7% or better everywhere
else. Read them as the pair, not the digit.

Nine sessions at 64K reach 5,071 tok/s of prefill against one card's
three-session 1,709 — **2.97x for 3x the cards**, and every session makes
progress throughout: first tokens land at [93.9, 99.3, 108.5 x5, 110.7 x2],
grouped by card rather than staggered one prompt at a time.

### A session beside busy neighbours

Aggregate throughput says nothing about what one session feels while the other
cards are working, and that is the number a person actually notices. One
session decodes 200 tokens while two 64K prompts prefill on the *other* two
cards; its inter-token latency is the measurement, and its own baseline on an
idle fleet is measured in the same run.

| decoding session | mean | median | p90 | p99 | max | wall spent stalled |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| fleet stepped in lockstep | 146.0 ms | 12.1 | 958 | 1,579 | 2,959 | 92% |
| workers on their own loops | 11.8 ms | 11.6 | 12.1 | 14.2 | 21.1 | 0% |
| the same fleet, idle | 11.7 ms | 11.6 | 11.8 | 12.4 | 19.8 | — |

The median is the trap: it is 12.1 ms in every row, and a probe that reported
it concluded the fleet cost a decoding session 1.3x. The cost was entirely in
the tail — 21 of 199 tokens absorbed 26.9 s of the 29.0 s that session took,
each stall the length of one 2048-token prefill chunk on a card that session
was not running on. Decoding beside two busy cards is now indistinguishable
from decoding on an idle fleet, and the prefills alongside it finished no
slower.

### Prefix cache: coverage decides whether it hits at all

Six-turn conversations, each turn extending the last, three of them at once on
one card. `reused_prefix_tokens` is the server's own admission telemetry, not
inferred from latency.

| arena | snapshots/worker | coverage per slot | prefix reuse | wall |
| :--- | ---: | ---: | ---: | ---: |
| old fixed default | 24 | 16,384 | **0% every request** | 319.1 s |
| `--cache-ram 12GiB` | 119 | 79,872 | 73% → 86% | 133.2 s |
| current default | 198 | 135,168 (100%) | 73% → 86% | 132.2 s |

The 0% is the point. An undersized arena does not cache less — it caches
*nothing*: publishing yields rather than evicts when slots are scarce, so a
worker that cannot hold its sessions' retention points publishes nothing, the
next turn has nothing to match, and every turn re-prefills from zero. There is
no partial-credit region, which is why the default is now derived from the
serving configuration rather than fixed.

Reuse is quantised to the retention interval — 43,008 of 49,879 tokens is
21 × 2048 — so the last partial interval is always re-prefilled. A single
conversation never shows this: 40K tokens needs ~20 snapshots and even the old
default held 24. It takes concurrent sessions to exceed the arena.

A larger arena than the context needs buys nothing, and the preflight says so.
Measured on three cards at `--cache-ram 64GiB`: 63.86 GiB pinned, 212
snapshots per worker, **106% of the context** covered, 60 s to healthy against
34 s at 19.88 GiB — and reuse identical to full coverage, 73% → 85% over six
turns. Host memory was fully released on shutdown. `RLIMIT_MEMLOCK` does not
bind: 63.86 GiB pinned against a 15.72 GiB soft *and hard* limit, because the
NVIDIA driver pins outside it.

The plateau at ~85% is not a shortfall. Each of these turns adds 6,179 tokens
that did not previously exist, and against the *reusable* prefix the cache
takes 97.1% → 98.4%. What is left is the partial retention interval below the
last snapshot boundary — 552 to 692 tokens per turn, bounded by `R` and
averaging about `R/2`. Only token-granular reuse would recover it; see the WHY
entry on what llama.cpp does with in-place slots.


### What this cost to get right

Two lock faults in the driver loop, not the scheduler, dominated everything
else. Before them, four sessions on one card produced first tokens at 34, 71,
136 and 203 seconds — one prompt at a time wearing three slots — and three
sessions at 64K were bimodal on a lock race, 1,734 tok/s or 826. Both are in
WHY. The scheduler's own contribution, sharing each step across prefilling
sessions, is worth +3.0% at 16K and +3.4% at 64K with worst-case time to
first token 3.5% lower; it is small next to the locks and was unmeasurable
until they were fixed.

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

## Compile-time formats free registers that ptxas then spends

The MoE expert prologue took its format as a runtime argument inside a
`__forceinline__` function, so every arm was emitted at all nine call sites
and the arms paid for each other — the mechanism that made adding three
community formats cost 5.05% (see WHY NOT). The fix is to make the format a
template parameter, `dequant_tile_ct<Q>`, and compile each tuned family once
per format: one unpacker per instantiation, branch gone before ptxas sees it.

**That is not automatically free, and the direction of the surprise is worth
carrying.** Removing the branch *raised* register counts on the hot kernels,
because ptxas allocates against an occupancy target and freed budget gets
spent on unrolling and loads in flight rather than returned:

    moe_expert_ffn        80 -> 96 registers   24 -> 16 warps/SM
    moe_expert_ffn_gemv   44 -> 59             40 -> 32 (both cap at 32)
    moe_shared_ffn_gemv   42 -> 57             40 -> 32 (both cap at 32)
    moe_shared_ffn_gemv_t3  64 -> 66           32 -> 24

Only two of those are real. sm_75 holds 64 K registers and 32 warps per SM, so
**64 registers per thread is the full-occupancy budget at any block size**, and
a kernel moving 44 -> 59 changes nothing because the warp ceiling was already
binding. `moe_expert_ffn` crossing into 96 and `..._t3` crossing 64 are the two
that cost blocks.

Both are recovered with `__launch_bounds__(GEMM_THREADS_CU, N)` at the blocks
per SM the runtime-ladder form reached — 3 for `moe_expert_ffn`, 4 for `_t3` —
and **ptxas hits those caps with zero spill stores and zero spill loads**, so
the branch removal is kept and the occupancy with it. Final state: 57 kernels,
no occupancy regression against the 33-kernel baseline anywhere, no spills
anywhere, and module compile 3.13 s -> 3.82 s.

**And it is a win, not merely a wash.** `bench_decode_batch 2048 32`,
`LLMXABE_BATCH_N=1,2,3`, GPU 0, box otherwise idle, three interleaved pairs
with the order reversed in the middle one, against the immediately preceding
commit:

    N=1   +6.98%  +2.86%  +6.92%   (worst +2.86%)
    N=2   +1.58%  +1.46%  +0.98%   (worst +0.98%)
    N=3   +1.36%  +1.51%  +1.21%   (worst +1.21%)

Nine of nine comparisons favour the specialized form, including the reversed
pair where run order works against it. N=1's spread is wide and its smallest
delta is the base-first pair, which is what first-runner-wins predicts; treat
+2.9% as the N=1 claim rather than +7%. N=2 and N=3 are tight enough to read
directly. Binaries were checked with `strings` to confirm each arm carried the
kernels it was supposed to before any of this was believed.

**The standing table has not been re-measured against llama.cpp**, so these
are deltas against our own previous commit and nothing more.

Two traps in the tool, both paid for here. `__launch_bounds__(T, 1)` is not a
no-op — it tells ptxas one block per SM suffices, which *relaxes* its default
heuristic and made `_t4` and `_t8` worse than leaving the attribute off. And a
register count is not an outcome: it is only evidence once turned into blocks
per SM, which needs the granularity (8 on Turing) and the warp ceiling.

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

And the corollary, which cost 2× prefill before it was measured: **a weight
that is not an integer used to take the whole block off the tensor cores.**
The attention int8 repack was gated on `weights.formats.all_q8_0()` — every
projection in the layer, not each one on its own. Qwen3.8-27B's shipped
`UD-Q8_K_XL` stores `attn_q`/`attn_k`/`attn_v` as bf16, so that predicate was
false, and the entire attention block fell back to the fp32 GEMV path at
prefill width. Measured in that state, requantizing the file to plain Q8_0
took N=1 prefill from 319.4 to 655.4 tok/s and N=3 from 326.3 to 692.0 —
**2.05× and 2.12×**, with no engine change at all.

The gate was not *wrong* — an int8 repack of a bf16 tensor is a quantization
decision, not a re-layout — but it was all-or-nothing where it could be
per-tensor, and its cost went unmeasured until a second model arrived carrying
mixed formats. It is now per-tensor: each projection takes the integer path if
its own weight is Q8_0. On the shipped `qwen35` file that means `attn_output`
alone, and it is worth **+13.7%** of prefill (316.0 → 359.3 tok/s, three
interleaved pairs, won 3/3) — not the 2x, because `attn_output` is ~30% of a
layer's projection elements and only 16 of the 64 layers are attention layers.
The other 70% needs the file to change, not the engine.

(The current numbers for that same comparison are 360.4 → 660.7 and
368.8 → 700.3, a 1.83× and 1.90×: the engine collected the part of the gap
that was its own, and the rest belongs to the file.)

Two costs, both measured, neither hidden: the repack is now built for a file
that previously built none, so **+0.50 GiB** resident; and dense N=3 decode
went 43.87 → 43.63 tok/s, **−0.5%**, consistently across all three pairs. At
three tokens `uses_tensor_cores` is false and the decode step's launches are
unchanged, so that delta is *unexplained* — it is not the integer path being
taken. N=1 decode is unaffected. On `qwen35moe`, where all four projections
are Q8_0, the new predicate and the old one are the same predicate:
prefill 2452.4 → 2446.3 (spreads overlap), decode 100.87 → 100.83 at N=1 and
204.73 → 204.73 at N=3, peak VRAM byte-identical.
`LLMXABE_ATTN_INT8_ALL_OR_NOTHING=1` restores the old predicate in the same
binary, which is how those two rows were measured rather than asserted.

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

## Residency: every projection on the card once, in the form its reader wants

The same question, asked of the card instead of a grid: *how many copies of
this tensor are resident, and how many are read?* The answer was two and
one, for every mixer projection. The arena held the file's layout of the
three Gated DeltaNet projections and the four attention projections, and
nothing read those copies: the GDN block reads its split int8 repack at
every width (the repack was built *from* the arena's copy and then the copy
sat there), and the attention block copied its four out of the arena into
allocations of its own for the decode GEMV and repacked from *those* for the
tensor cores. Per tensor directory that was 8.3 GiB held twice on the
dense file and about 1.3 GiB on `qwen35moe`.

The fix is a filter, not a kernel. The engine loads with
`arena_holds_entry`, which sees the stored format beside the role: the
attention projections stay out of the arena in every format, the GDN
projections stay out when Q8_0, and `Forward::new` repacks a Q8_0 GDN
projection from a transient upload of the file's bytes that is freed once
the repack kernel has read it — one tensor at a time, so the build's peak
is one 83 MiB tensor above steady state. A GDN projection in any other
format has no repack and stays in the arena, read in place by the generic
kernels as before. The golden found the one width that still read the
stored layout — the single-sequence pass at 2..63 tokens took the
standard-layout Q8_0 tile where the batch paths took the split tile — and
that path now takes the split tile too, which moved the 19-token golden's
margins toward llama.cpp (final-logit cosine 0.999736 → 0.999759, max-abs
0.464 → 0.415). Measured with `bench_forward` (512 tokens, three prompts,
driver accounting), against the tree before it:

| file | arena | peak VRAM |
| :--- | ---: | ---: |
| `Qwen3.8-27B-UD-Q8_K_XL` | 11.822 → 3.658 GiB | 39.387 → **31.262 GiB** |
| `Qwen3.6-35B-A3B-UD-Q6_K_XL` | 2.291 → 1.025 GiB | 33.479 → **32.229 GiB** |

Nothing any kernel reads changed — the repack and the block's copy are the
same bytes they were — so throughput was not measured and none is claimed.
What the 8.1 GiB buys on the dense card (a deeper KV pool, `draft-mtp`
beside three 128K caches, a fourth slot) is not measured either; the
Capability table above still says what was. The role-only filter
`arena_holds` still exists for one reader: `tests/int8_forward.rs` builds
an fp32 twin over the stored-format path, and that twin needs the arena to
carry the bytes. Under the engine's filter `Forward::disable_tensor_cores`
refuses rather than build a pass with nothing to read.

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
- **The split repack's scale width is traffic, not precision.** Splitting the
  scales out of the quants said nothing about how wide to store them, and they
  were stored fp32 — which widens a value that has no more precision to give,
  because a Q8_0 block's scale *is* an fp16. The layout cost 1.125 bytes an
  element against the on-disk 1.0625, so the repack that exists to make the
  loads aligned was also making them 5.9% more numerous. Invisible where it
  covers a MoE shared expert (0.1 MB in a layer of 725); on `qwen35`, where the
  same struct holds a 17,408-wide dense FFN, it was **1.01 GiB of every decoded
  token**. Keeping the file's own fp16 bits — moved, not converted, so the
  values are identical to the last bit — is worth 4.2% of the dense FFN block
  at one token and 1.4 GiB of resident VRAM, and every differential test is
  unchanged because nothing about the arithmetic changed.

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
- The rule survives a change of *model*. Qwen3.8-27B's dense FFN is resident
  in one layout only (VRAM decides that), so its residency cannot select the
  kernel — its width has to. Running the GEMM at one token cost 109.0 ms
  against a GEMV's 63.5 on identical weight traffic, and the fix was not a new
  kernel: `GdnBlock`'s split-layout projection GEMV already reads that exact
  layout at that exact width. A second architecture is a good test of whether
  a lesson was learned as a mechanism or as a constant.
- The rule does **not** reach the router's expert tile, though it looks like
  it should and that reasoning was acted on twice. At decode `grid.y` is 1, so
  `num_experts / ET` is the entire grid — 32 blocks, 256 warps against the
  2,304 this part holds — and halving `ET` to 4 doubles the blocks. It is
  worth nothing measurable; see the WHY NOT row. The token tile is a different
  matter and does pay: `TT` 8 at one token spends 7/8 of the arithmetic and
  7/8 of the staged tile recomputing one clamped row, which is why `_t1` and
  `_t3` exist.

Two launches that each fail to fill the card *look* like they can be made to
fill it together, and the Gated DeltaNet block's input projections are the
strongest case for it this engine has: `qkv` and `gate` read the same
normalized activation, write disjoint scratch, and are ordered by nothing but
the stream they are issued on. At N=3 with the row tile 4 they are 2,048 and
1,024 warps against the 1,152 this part holds at that kernel's 128 registers —
1.78 waves and then 0.89, each paying its own ramp against a half-empty
machine, reading **407** and **351 GB/s** where the routed-expert GEMVs beside
them run 21 waves deep and read 503–518. Forked onto a side stream they do
overlap, 1.46x, and the main stream's busy time falls 12.95 → 12.21 ms per
step in `nsys`.

**The fork is not in the engine; the merge is.** The fork was built,
measured at +0.79% at N=3, found to be −1.2% at N=1, reverted for a defect,
and never re-measured against a clean baseline — see WHY NOT. What it left
behind was the second data point on where this step's ceiling comes from:
two independent measurements saying the step is launch-latency-bound at one
token, so anything that *adds* work to the stream — an event pair per layer,
sixty graph nodes — loses there whatever it overlaps elsewhere.

What survives that constraint is putting both matrices under **one grid**.
`gdn_proj_split_pair_*` takes the two weight pointers, the two outputs and
the row seam, and each warp group runs the single-matrix body on whichever
matrix its rows fall in — no event, no second stream, and every output is
the same chain of additions the separate launches produced, asserted bit for
bit at every token width the tile table has structure at. The two partial
waves become one grid of 2.67, and the ramp and drain are paid once. Three
interleaved pairs at 2K on GPU 1, order reversed in the middle pair, the
baseline arm flat across the sitting (210.1–210.6):

| width | pair 1 | pair 2 | pair 3 |
| :--- | ---: | ---: | ---: |
| N=3 | 210.3 → 213.4 | 210.1 → 213.7 | 210.6 → 212.9 |
| N=1 | 109.3 → 111.1 | 108.9 → 111.3 | 109.3 → 111.7 |

**+1.1–1.7% at N=3 and +1.6–2.2% at N=1**, every pair won, and the N=1 side
is the tell: the fork *lost* there, and a merge that pays no events wins
there, which is the launch-bound reading confirmed from the other direction.
Depth was not measured; the saving is a fixed amount per layer and is a
smaller fraction of a deeper step.

The one thing it cost was registers, and only until it was told not to.
Left to itself ptxas spent the second pointer set: the one-token tile went
80 → 96 registers and crossed from six resident blocks to five for no
arithmetic. `__launch_bounds__(128, 6)` put it back at 80 with no spill and
`t2`–`t4` held their counts under the matching bound; the two widest tiles
are left unbounded, because the bound that would restore `t8`'s fourth
block costs 24 bytes of spill and `(128, 2)` on `t16` sends ptxas to 255
registers *and* a spill against 208 unbounded — the `(T, 1)` trap from
"Compile-time formats free registers", one notch up. Check the count on
every tile of a family, not the one the target width uses.

**The same seam carries a whole small kernel into a big one's grid.** The
shared expert at decode width was two launches of its own — 2.2 MB of
gate/up and 1.1 MB of down at 6.8 and 4.1 us, which is 240–330 GB/s and
mostly ramp and drain — beside a routed-expert launch several waves deep
that reads the same activation and is ready at the same moment. Its
side-stream overlap is in WHY NOT (flat at N=3, −2.5–3% at N=1: events).
`moe_fused_{ffn,down}_*` give the shared expert's blocks `blockIdx.y` slots
past the routed grid and run the shared body there — two rows per
256-thread block for the gate/up, since that body was written for four
warps splitting one row's contraction, and one row per warp for the down —
while the routed blocks run the routed body on the indices the separate
launch gave them. Forty layers lose two launches each, `partial`,
`shared_inter` and the shared output are bit-identical to the four-launch
form at every width the fused entries cover (`moe_fused_shared_
differential`, widths 1–4 with every slot live and with one live token in a
wider pass), and the register count is the max of the two bodies rather
than their sum. Three interleaved pairs against the qkv+gate merge as the
baseline, same method and card, baseline arm drifting −0.5% over the
sitting:

| width | pair 1 | pair 2 | pair 3 |
| :--- | ---: | ---: | ---: |
| N=3 | 213.6 → 216.8 | 212.7 → 214.9 | 212.6 → 216.0 |
| N=1 | 111.9 → 114.7 | 111.1 → 114.2 | 111.2 → 113.9 |

**+1.0–1.6% at N=3 and +2.4–2.8% at N=1**, every pair won; by `nsys` the
step lost 80 launches (745 → 665 at N=1) and 0.25 ms of wall at N=1 and
0.28 ms at N=3. The rule the two merges share: at one token every launch
under ten microseconds is mostly its own floor, and the floor is paid once
per grid, not once per body. What a side stream cannot buy — because the
events it needs are themselves graph nodes on a step with no slack — a
seam in `blockIdx` buys for free, as long as each body is left exactly as
its separate launch ran it.

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

## The snapshot arena is all-or-nothing, so size it from the configuration

A prefix cache that is too small is intuitively a cache that hits less often.
This one hits *never*. Publishing a snapshot yields rather than evicts when
slots are scarce (`Engine::publish` calls `reclaim_for` and returns without
publishing if it cannot reserve), so a worker whose arena cannot hold its
sessions' retention points publishes nothing at all — and with nothing
published, the next turn has nothing to match, so it re-prefills from zero and
publishes nothing in turn. Measured with three growing conversations on one
card: 0% reuse on every request against 73–86% once the arena covered the
context, 319 s of wall clock against 132.

The fixed 24-slot default was sized against the host's locked-memory limit and
covered 16,384 tokens per slot — under one agent turn at this context. It is
now derived from what is being served: one snapshot per retention interval,
per slot, across that slot's share of `--total-context`, capped at a quarter
of the host's `MemAvailable` because the arena is page-locked and the weights
still need the page cache to be read through. `--cache-ram full` lifts the
cap.

llama.cpp reaches the same place from the other side and is worth reading on
this: `--cache-ram` defaults to 8192 MiB there against our former 2.41 GiB
effective, and its `server_prompt_cache` is an LRU bounded in *bytes* rather
than a fixed pool, so it degrades by evicting rather than by declining to
publish. It also has a tier we do not: a slot keeps its KV in place between
requests and takes the common prefix against what it already holds
(`get_common_prefix` → `n_past`), so a continuing conversation that lands on
its own slot costs nothing at all. Ours always pays a 102.81 MiB host round
trip for the same continuation. Making same-slot continuation free is the
obvious next lever and is not built.

## A speculative output ceiling is not a prediction, so it is not a reservation

Admission reserves `prompt + max_output_tokens` and the pool hands out real
pages for all of it, held for the sequence's whole life — rule 4, and right
about the prompt, because chunked prefill splits compute and not memory. It is
wrong about the output. `max_output_tokens` is the caller's ceiling, not its
estimate: an agent that asks for 65,536 tokens and emits fifty has taken 256
blocks — 1.28 GiB at this model's 5 MiB pages — from its neighbours for
nothing, and the arithmetic is unforgiving. Against one card's 1,584 blocks,
three sessions on 100K prompts fit at `max_tokens` 4,096 and only two fit at
65,536; at 150K prompts it is two against one.

llama.cpp does not do this. It checks the prompt against the slot
(`tools/server/server-context.cpp`, `slot.task->n_tokens() >= slot.n_ctx`) and
treats `n_predict` purely as a stopping condition (`server_slot::n_remaining`,
`has_budget`), allocating as the sequence grows.

So the ceiling is capped at one slot's share of the pool rather than honoured
in full, with a floor of one block so a prompt longer than the share can still
generate. It caps the *reservation*: a sequence that genuinely runs that far
stops there and reports `length`, which is what llama.cpp does when a slot's
context fills. Capping is also what makes the refusal legible — a caller who
asked for 410,000 output tokens used to get `503 every worker refused
admission` and no way to tell an impossible prompt from a full queue, because
`can_admit` was a bool. Refusals now carry the scheduler's own reason.

## Concurrency is a lock property before it is a scheduler property

Nine sessions across three cards were not concurrent for four reasons, and
not one of them lived in `xabe-sched`. They all presented identically — as a
scheduler that refused to share — and each was diagnosed only once a step
logged what it actually carried, which is why that log is in the tree.

**The driver loop must yield the lock it steps under.** A driver loop takes
it, holds it for a whole GPU step, releases it and takes it straight back, so
a handler blocked in `place_tokens` loses that race indefinitely. This was
first found when that lock was one `Mutex<Engine>`; it survives the split into
per-worker locks, because a submission still has to score every worker. Two symptoms, one fault. A client is registered
*before* its request reaches the engine, so between those two moments there is
nothing to run and the loop span on empty steps — 20,636 of them in a 27 s
run against 15 in a run that happened to win the race. And while a long prompt
is prefilling every step is productive, so no idle back-off can help: at 64K,
sessions dispatched 2 ms apart were not merely unscheduled but never
submitted, `running=1 waiting=0` for twenty-two consecutive steps while the
first prompt had the card to itself. Handlers now raise an atomic before they
block and the loop stands aside 1 ms when it is non-zero. Three sessions at
64K went from bimodal — 1,734 tok/s at [36.5, 107, 110] or 826 at [36.5, 228,
231] — to 1,635–1,708 with first tokens clustered at [105, 117, 117], spread
2.2%.

**A step is shared, capped at one retention interval per session.** Phase 2
stopped at the first request whose prompt did not fit in one step, so that
request took the whole budget every step until it finished. It now walks the
running set from a rotating start, granting each prefilling session at most
`--prefill-slice` tokens — the snapshot retention interval, which is already
the widest pass the engine can issue because a pass may not straddle a
boundary. The same tokens therefore move at the same width, spread across
sessions rather than stacked behind one; the rotation decides who is short
when the budget covers fewer slices than there are sessions. Worth +3.0% at
16K and +3.4% at 64K on aggregate prefill, with worst-case time to first token
3.5% lower.

**The fleet must not step in lockstep.** One driver thread spawned all three
workers every step and joined them before starting the next, so every card ran
at the speed of the slowest card *in that step*. A card with one decode token
to emit finished in 12 ms and then sat at the join for the rest of a 2048-token
prefill chunk on somebody else's card. The engine is now one lock per worker
with a driver thread each, taking only its own worker's lock and releasing it
before the step's cache bookkeeping; `step_devices` survives for the smoke
binary and the tests, which do want one bounded unit of fleet-wide progress.
Worth 4.7–12.1% on aggregate decode and 0.6–16.2% on aggregate prefill across
the nine-session table, and it is what makes a session beside busy neighbours
cost nothing rather than 12x.

**Routing is the one decision that must stay atomic.** Scoring reads every
worker's load and then admits to the cheapest one, and that pair was atomic
only because one `Mutex<Engine>` happened to make it so. Per-worker locks took
that away, and concurrent handlers all scored the same idle fleet and all
chose the same card: nine simultaneous 64K sessions placed 4/2/3, and the
fourth session on the oversubscribed card waited for a free slot — 150 s to
its first token against 113 s for its neighbours, and 269 s on a worse split,
which read as the barrier not being fixed at all. A routing lock held across
score-and-admit restores it. It never covers a GPU step, so the driver loops
never wait on it, and placements are exact thirds again.

The order matters for anyone reading the history: the scheduler change was
measured as a 3% *regression* until the locks were fixed, because it could not
get sessions to schedule. The routing race is the same lesson one level up —
removing a lock removed an invariant nothing had written down.

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

**Everything above is a shallow-context result (256-token prompts), and it
does not survive depth.** Re-measured at this project's actual N=3 target —
three slots of ~120K, the most that fits beside its own output in a
393,216-token pool — the ranking does not shrink, it inverts:

| Drafter at N=3, ~120K/slot | prefill | decode | tokens/step |
| --- | --- | --- | --- |
| plain (`none`) | **821 tok/s** | **99.7 tok/s** | 3.0 |
| `ngram-map-k4v`, cap 8 | 745 (−9.3%) | 19.3 (**−80.6%**) | 27.0 |
| `draft-mtp`, 3 | 642 (−21.8%) | 9.2 (**−90.8%**) | 12.0 |
| `ngram-map-k4v`, cap 32 or 48 | — | — | out of memory |
| `spec-dflash`, 2 or 3 | — | — | out of memory |

Acceptance is not what fails. It is *perfect*: MTP took 12.0 of 12
available tokens per step and the map drafter 27.0 of 27, and both still
lost by an order of magnitude. What fails is that **at depth the verify
step's cost is set by context, not by window** — 1.30 s carrying 12 rows,
1.38 s carrying 27 — against a 30 ms captured plain decode step. A ~45×
per-step premium cannot be repaid by a 9× token multiple, and no drafting
policy can change the premium. At 256 tokens that premium was 1.7×, which
is the entire reason the sign flips.

Two second-order effects are worth carrying. Speculation costs *prefill*
at N=3 even when nothing is drafted during it: a decoding sequence charges
`1 + drafts` against the per-step token budget, so concurrent prefill
chunks get less of it — 9.3% for a cap of 8, before MTP's own per-chunk
catch-up pass takes it to 21.8%. And VRAM at this depth is the binding
constraint on the drafters that were best when shallow: the KV pool leaves
so little room that `ngram-map-k4v` cannot run above a cap of ~8, and
DFlash — whose six draft-layer caches are sized to the *full sequence*, so
they scale with context rather than with the draft count — does not run at
all.

Serving therefore defaults to `--spec-type none`, and at the N=3 deep
target that default is not a compromise but the best configuration
measured. The win is real, and large, only for **shallow single-stream**
work, and there it belongs to the model-free drafters — the ones whose
draft side costs nothing to widen.

---

# WHY NOT

Everything below was built or derived far enough to measure, and rejected.
They are recorded so they are not re-attempted: several have already been
proposed twice.

## Rejected on measurement

| Attempt | Result |
| --- | --- |
| GDN split projection at row tiles 1 and 2 (`uint4` form, N=3) | 78.4 and 46.4 us per qkv/gate call against RT=4's 33.9. RT=1 octuples the warps and the in-flight bytes and is the *worst* of the three, so memory-level parallelism was never the binding constraint — instruction count per byte is, and it falls with RT. |
| The output projection at row tile 2 (2,048 rows, 512 warps at RT = 4 — under half a wave) | Bit-identical by construction and **flat at N=1** (23.0 us either way), **+25% at N=3** (28.2 → 35.2 us). Same finding as the row above on a second shape: the grid was 0.44 waves deep and doubling its warps bought nothing, because the kernel's cost at three tokens is the activation instruction count per weight byte, which RT halving doubles. Wave fill is not what binds a decode-width GEMV on this card; do not re-derive it from the grid arithmetic. |
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
| Prefetching the GDN alpha/beta gate contraction | 21.2 us a call becomes 28.5 (four blocks staged), 30.8 (two), or 46.3 (one block ahead, the `gdn_proj_split_rows` double-buffer idiom). The kernel is 12 blocks on a 72-SM card and looks latency-bound, so this should have worked; every restructuring made it worse, which says ptxas was already scheduling the tight loop better than the source could. The gates stay as written, and their 1.0 ms stays on the table. |
| Staging the weights instead of the activations at every GEMV width | 0.549 -> 0.585 ms a layer at three tokens and 0.570 -> 0.695 at four. Hoisting `RT * 8` dequantized weights is what lets `TT` pass four, but at `TT <= 4` the row tile is 8 and sixty-four weights live costs more than the `TT * 8` activations it replaces. Both stagings ship, chosen by `TT`. |
| Hoisting the GDN Gram kernel out of the chunk loop | Bit-identical, 240 launches → 30, and **0.5% slower**, losing every interleaved pair. In the loop the Gram output hits L2 immediately; hoisted, all eight chunks' squares must coexist in a 6 MiB L2. On this part, launch count is not worth trading locality for. |
| `shared_expert_mma` on the dense FFN at decode width (Qwen3.8-27B) | 109.0 ms per step against the fp32 path's 63.5, and 9.17 tok/s against 15.74. A GEMM stages a 64-token tile and at one token discards 63/64 of it; the weight bytes are identical either way, so what it loses is the shape it reads them in, not the traffic. Replaced by a copy of `GdnBlock`'s split-layout GEMV — 17.26 tok/s, and *more* accurate than the GEMM besides, since it dequantizes the weights and multiplies in fp32 rather than quantizing activations. |
| Tiling `gdn_chunk_gram` over target tokens | Nothing. Its keys are 32 KiB per head and sit in L2 — the arithmetic that says a kernel re-reads its input does not say the re-read costs anything. |
| Shared-memory staging on the GDN decode GEMV and the routed-expert GEMV | −3.6% and −0.8%. The re-reads staging removes were L1 hits; staging replaced a hit with a copy and a barrier, and cost a resident block. |
| Shared-staged flat decode GEMVs (`int4` staging, as the LM head uses) | Bit-identical by construction and **1.3–2.4% slower**. These kernels are bound by the **integer pipe**, not by load issue: Q6_K unpack costs ~9 integer ops per element against a ~10-op budget at the streaming roofline. |
| Shared-memory activation staging in the dense split GEMV | 0.528 ms against 0.512 at one token and 0.862 against 0.684 at four, losing every pair. The staged form reads activations coalesced once per *block* instead of half-used once per *warp* — 2x fewer sector requests, exactly what the address arithmetic predicts — and the two barriers a 512-element step needs cost more than the sectors save. Third time shared staging has lost to registers on a decode-width GEMV in this file. |
| `ld.global.cs` (evict-first) on the dense split GEMV's weight loads | Wins every cell of the isolated bench — 0.508 against 0.512 at one token, 0.601 against 0.690 at four, three interleaved pairs — and **loses in the model**: `dense_proj_split_t1_r4` 166.6 -> 167.6 us and the GDN out projection 67.8 -> 71.9, for a net 52.28 -> 52.65 ms step. The hypothesis (streamed weights evicting the re-read activations from L1) is right about the isolated kernel and wrong about a step where twenty other kernels have already decided what is in cache. A narrow bench that wins is a candidate, not a result. |
| `CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 0` on the dense split GEMV | Nothing, at any width. The kernel already requests zero shared memory, so the driver was already giving it the large-L1 split; the hint had nothing to ask for. |
| Splitting the GDN gate contraction across eight warps | 20.9 -> 5.4 us on the kernel, 52.28 -> 51.57 ms on the step (**+1.3%**), and it moves the model's logits: `1 - cosine` against llama.cpp's own logits grows 2.64e-4 -> 2.94e-4 and `forward_pass` fails. The reduction is *more* accurate on the top-8 logits (max disagreement 0.309 -> 0.147) and less accurate in L2, which is the metric that gate is written on. One warp per head leaves 48 warps on a 72-SM card and the kernel at 25 GB/s; that is a real 1.9% of the decode step sitting behind a summation order the goldens are gated on, and any replacement has to be bit-identical to collect it. |
| Unrolling the GDN gate loop for memory-level parallelism | Exactly nothing — 20.9 us before and after. `ptxas` was already issuing the loads ahead; the kernel is short of *warps*, not of in-flight loads per warp. |
| I2F-free unpack (exact-mantissa trick) | Bit-exact, every gate green, **flat**. With both the load-issue and XU-pipe hypotheses dead, the flat decode GEMVs at 473–519 GB/s read as at their practical equilibrium for this quantization on this card. |
| Even/odd MMA accumulator chains | −2% prefill, noise at decode. The compiler's schedule was not accumulator-stalled, and eight more registers on kernels already at ~230 costs more than the chain relief. |
| Skipping the online-softmax rescale when the running max did not move | Bit-identical by construction and **2.8% slower** at 128K prefill. The identity multiplies hid under staged-load latency the schedule pays anyway; the vote-and-branch costs more than the work it skips. |
| llama.cpp's n-gram drafters at N=3, at llama.cpp's own 48-token defaults | **−22.8%** (`ngram-simple`), **−77.8%**/**−78.8%** (`ngram-map-k`/`k4v`) against plain decode's 210.9 tok/s; three interleaved rounds, spreads ≤0.4%. Not an acceptance failure — the map pair fills 22 of 49 rows per sequence — but a row-count one: 3 × 49 = 147 verify rows against a captured 3-row graph step. The same policies win 16–21% at N=1, where the window is free. The map pair's *magnitude* is additionally inflated by `bench_worker_spec` letting a high-acceptance sequence run far past the timed quota while a straggler gates the window; the sign is not in doubt, the exact figure is. |
| Any speculative decoding at the N=3 deep-context target (~120K per slot) | **−80.6%** (`ngram-map-k4v` at a cap of 8) and **−90.8%** (`draft-mtp` at 3) on decode, plus −9.3% and −21.8% on *prefill*; `spec-dflash` and the wider n-gram caps do not fit in VRAM at all. Interleaved rounds, decode spreads ≤2.6%. Acceptance was perfect in both survivors (12.0 of 12 and 27.0 of 27 tokens per step) and irrelevant: the verify step costs 1.30 s at 12 rows and 1.38 s at 27 against a 30 ms captured decode step, so its price is set by context depth rather than window width and no policy can repay it. The same drafters win 16–21% at 256 tokens; **a speculative result measured shallow says nothing about the deep target.** |
| Widening the trained drafters to the drafter's trained maximum (`--spec-draft-n-max 15`) | MTP **+1.4% → −55.4%** and DFlash **−14.8% → −65.8%** at N=1; −27.2% → −67.3% and −39.5% → −54.8% at N=3. Three interleaved rounds, spreads ≤1.0%. Acceptance did rise (MTP 2.86 → 3.90 tokens per step at N=1) and was nowhere near enough: a 3 → 15 block adds **+57.2 ms/step** (MTP) and **+52.6 ms** (DFlash) onto a 9.64 ms plain step, because the *drafter* pass scales with the block even though the verify pass does not (4 → 49 verify rows is +0.14 ms). A wider window is free only when the draft side is a host lookup; making a trained drafter pay means making its pass cheaper, not longer. |
| `ngram-mod` at llama.cpp's default `n_max` 64, N=3 | `CUDA_ERROR_OUT_OF_MEMORY` before the first step, in every round. The verify path holds one GDN snapshot ring set per decode slot — 30 layers × (32·128·128 + 8192·3) fp32 × `(drafts + 2)` — which is 4.05 GiB per slot at 64 drafts and 12.15 GiB for three, beside a 29.6 GiB model on a 48 GiB card. 48 drafts (9.2 GiB for three) fits and still loses. The draft cap is a VRAM knob, not only a scheduling one. |
| Tensor cores for routed MoE at N=3 (`MMA_MIN_TOKENS` 8 → 3) | 135.2 vs 147.7 tok/s. Padding a three-row dispatch into MMA fragments and quantizing its activations costs more than the arithmetic recovers. |
| The general fp32 expert tiles at N=3 (`MOE_NARROW_DECODE_MAX` 4 → 2) | 129.6 vs 147.7 tok/s. With the tensor-core result above, this brackets the current N=3 choice: both neighbouring kernel paths are slower. |
| The scalar warp decode kernel at depth | 133.0 vs 148–151 tok/s at 32K N=3. The depth-aware dispatch onto the tensor-core kernel stands. |
| `WPO=4` decode occupancy width | ~4.5% slower than `WPO=2` at 32K N=3, before and after the register-spill fix. Never wins at any depth measured. |
| Decode blocks 48 or 60 at 72 logical splits | 150.5 and 139.5 against 36 blocks' 152–153. 48 divides 72 unevenly and the long blocks set the makespan; 60 spills into a second wave. |
| Three independent N=1 decode graphs on one card | 129.9 vs 147.7 tok/s at 2K. Separate streams repeat weight traffic and contend more than their overlap recovers; serving keeps batched weight reuse. |
| Shared-expert side-stream overlap | Flat at N=3 (no SM or DRAM slack to hide in), **−2.5–3%** at N=1 (the step is launch-latency-bound and two events per MoE layer is pure overhead). The launches it tried to hide were later removed instead — the shared expert's blocks now ride the routed expert's grid, no events; see WHY, "The same seam carries a whole small kernel". |
| Independent GDN streams at prefill | Within thermal drift, reversing across warmed pairs. Removed rather than kept on a cold-card gain. |
| Cross-sequence prefill flattening at *equal* total ubatch | 0.99×. At fixed physical width, batching does not reduce the number of full-model passes. It wins only when the pass gets wider — see WHY. |
| The two-kernel `bm == 1` split, first attempt | −5.2% before `bucket_live` existed: both kernels then walked `sorted_token_ids` and crossed two barriers per bucket to compute `bm`, and only one used the answer. It landed as a win only once `bm` became a table read. |
| GEMV unroll pragmas on the direct-1 helpers | 20% slower on ffn. A register cliff — the standalone GEMV kernel's unroll depth was tuned against a register budget the narrow kernel, which carries the tiled fallback in the same function, does not have. |
| A `KU` unroll hint on the GDN tiled projection | Byte-identical SASS at the width N=3 actually uses; −9% at N=2. `ptxas` already reached the same schedule. Whatever closes that kernel's 28%-of-roofline ceiling must change what ptxas schedules, not hint at a schedule it already finds. |
| Speculative decode as the N=3 lever (n-gram, MTP, DFlash; serving A/B, four interleaved rounds with a reversal) | Every arm loses at N=3: 87.5 / 150.7 / 125.2 tok/s against plain 211.4, despite up to 10.3 accepted tokens per 12-row verify window. The batched verify's fixed cost (~4.8× a plain batch-3 step) outruns the tokens it saves, and the round-gated fallback at identical acceptance also loses (166 vs 211). The measured net wins are n-gram **+9%** and MTP **+1.5%**, both at N=1 only — see WHY. Making the verify pass ride decode's launch machinery instead of the prefill shape was named here as the untried lever; it has since been *partly* built — a narrow query window now goes through the flash-decode split rather than a two-block prefill launch (WHY, "Speculative decode"), worth 21 ms of a 194 ms N=3 verify step on `qwen35`. **The MoE serving arm above has not been re-measured against it**, so these numbers stand as the last measurement, not as a current claim. |
| Removing the routed-partial clear | Below run-to-run spread, and reversing. Deleting a defensive correctness aid for a result smaller than host drift is not justified. |
| Grant alignment as a prefill lever (`ADMISSION_RESERVE_FRACTION` 4 → 2) | Predicted **+27%**, measured **+1.5%**. The width curve is real — 2,048-wide passes run at 2,760 tok/s against 256-wide at 1,510, and the ratio holds at depth (1,896 vs 1,050 at 64K) — and `choose_prefill_width` does decompose a 3,072-token grant into 2,048 + 4×256. `max_batch = 3`, the 256 tail ceiling, and the grant reaching `execute_prefill` intact were all verified. The penalty still does not appear end to end. The constant stays at 2 because it is never worse and an aligned grant is the honest default, but **do not rank work by that width arithmetic**; the mechanism is confirmed and its cost is not. |
| Snapshot retention as the cause of the concurrent-session penalty | Nothing. `--cache-ram 0` (no snapshots) and `16GiB` (159 per worker) both land within noise of the 2.41 GiB default's 24, at 64K × 3 sessions. The arena arithmetic is seductive — a 64K prompt needs 31 snapshots at R=2048, three sessions ~93 against 24 — and wrong. Worse, it was first "ruled out" at 16K, where three sessions need exactly 24 and the arena *cannot* bind, which proved nothing in either direction. Test a capacity hypothesis at a depth where the capacity is actually exceeded. |
| Community-quant formats in the runtime `dequant_tile` ladder, and `__noinline__` to contain them | The ladder is `__forceinline__` with a runtime `quant`, so every arm is emitted at every call site — nine of them across `tile_gemm_pair`, `tile_gemm_single`, the `_direct1` variants and the shared-expert bodies. Three extra formats cost `moe_shared_ffn_gemv` +20 registers (42 → 62), `moe_expert_ffn_flat` +18, `moe_expert_ffn` +16, `moe_expert_ffn_gemv` +11; 20 kernels moved, **−5.05%** on Ornith N=3 2K (195.5 against 205.9, three interleaved pairs, same card two minutes apart). `moe_expert_ffn_flat_q6` did **not** move — it is specialized with no runtime `quant` — which localizes the cost to the ladder rather than to the bodies existing. **`__noinline__` is worse, not better**: `moe_expert_ffn` 80 → 124 and `moe_shared_ffn` 92 → 130, because a non-inlined device call spills live values across the ABI boundary. Reverted. **The compile-time flag it prescribed has since landed** — `dequant_tile_ct<Q>`, one instantiation per format, no runtime branch; see "Compile-time formats free registers that ptxas then spends" in WHY for the before/after and for the occupancy trap it turned up on the way. **The generalisation is structural**: a runtime ladder inside a `__forceinline__` function is a *shared resource*, so an arm added for one format is paid for by every format already using it; a macro that emits a separate `__global__` per format — as `GDN_GATES` does — is not, and a new one there touches nothing existing. Which of the two shapes a kernel family uses decides whether adding a format is free, and it is worth knowing before writing the format rather than after measuring it. |
| Forking the Gated DeltaNet input projections onto a side stream | **−1.2%** at N=1 2K, losing both interleaved pairs, against +0.79% winning all three at N=3; a first pair read −7.5% and is quoted only as the reason the set was extended. The overlap is real at both widths (1.46x in `nsys`) and irrelevant at one token: the step is launch-latency-bound there, so two events per layer across thirty layers is sixty graph nodes bought against a machine with no wave to fill. Second independent change to hit that wall — see the shared-expert side-stream row above. **Reverted, and the N=3 figure is not a current claim**: a `replace_all` had put the fork into `run_batch_prefill` as well as `run_batch_decode` — one uncaptured path and one inside a graph capture, both forking onto one side stream recording one pair of events, on a `GdnBlock` that serving alternates between them. The +0.79% was measured against a stale pinned binary by the same method that produced the router tile's +1.31%, which did not survive re-measurement against a baseline built from its own commit. Anyone re-attempting this owes it a clean baseline before quoting a number. |
| Narrowing the router's expert tile at decode width (`ET` 8 → 4) | **Nothing — and the sweep that motivated it cannot resolve anything it claimed.** This host drifts **−0.8% at N=3 over ten minutes**, monotonically and in one direction, which is larger than every effect the `ET` sweep reported; a sequential sweep crossing that drift produces exactly the ragged curve that invites an occupancy-knee story, so **neither the sweep's shape nor its knee is evidence of anything until it is re-run interleaved**. Nor can a sweep be corrected for order after the fact: drift favours the first position and cold caches, first-touch faults and the NVRTC compile penalise it, so the sign of the bias is not even known without measuring it. Re-measured against a baseline built from the pinned commit rather than a stale binary: three interleaved reversed pairs at 2K read +0.24 / −0.19 / −0.05% at N=3, +0.30 / +0.18 / −0.12% at N=2 and +0.49 / +0.29 / 0.00% at N=1 — the sign flips inside every set. The +1.31% recorded here before came from pairs against a stale pinned binary, and a control A/B of that binary against current main read +0.53 / −0.05% at N=3, so the baseline gap does not explain it; the original sweep's run order was not kept, so it cannot be checked against the drift either. The grid arithmetic is not wrong — `grid.y` is 1 at decode, so `ET = 8` is 32 blocks and 256 warps against the 2,304 this part holds — it is just not what binds. `ptxas -v` either side: the prefill kernel unmoved at 96 registers, so nothing was traded for the non-result. Prefill cannot be hiding a win (it selects that unmoved kernel) and depth cannot be (router cost is independent of KV depth, so the tile is a *smaller* fraction of a deeper step). |
| Reading the Ornith fork's 32K result at all | Three interleaved pairs gave −0.31%, +2.62% and +1.39% while the *baseline* arm ranged 1.85% across its own three runs. The prediction on record was +0.55–0.6%, from a fixed per-layer saving over a step that grows 14.5 → 18.9 ms. The set can neither confirm nor refute that, so no 32K figure is quoted for this change. Recorded because the temptation was to take the mean and call it +1.2%. |
| Inferring scheduler behaviour from client-side timings | Wrong three times: a lock-starvation fault was read as a scheduler refusing to share, an admission-pacing artefact as a race, and a 2x throughput collapse as a kernel concurrency limit (the kernel charges 5%, not 50%). All three fell out immediately once a step logged its own grants. Instrument the component before theorising about it. |

## Rejected on arithmetic, before building

- **Widening Qwen3.8-27B's bf16 tensors on the host.** 53 of its 866 tensors
  are bf16 and they are among its largest: `output.weight` alone is 2.54 GiB,
  and every attention `attn_q`/`attn_k`/`attn_v` besides. fp32 would put
  5.1 GiB on the card for the head alone and *double* the per-token read of
  the model's most bandwidth-expensive tensor. Requantizing them to Q8_0
  instead changes the model. The LM-head GEMV grew a bf16 body instead — the
  simpler of its two, since bf16 widening is a shift and a bf16 row is dense
  enough to need neither the staging pass nor the alignment prologue Q8_0's
  34-byte stride forces.
- **Holding the dense FFN in both Q8_0 and the split int8 layout.** 16.9 GiB
  twice over across 64 layers, on a card already carrying 11.8 GiB of arena.
  This one was not rejected on arithmetic in time — the first attempt at the
  dense model died on `CUDA_ERROR_OUT_OF_MEMORY`, which is what the arithmetic
  would have said. See "What the FFN residency is worth" above for what
  choosing between them measured.
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

- **"The fleet-wide step barrier costs a neighbouring session almost
  nothing."** The probe reported **1.3x** and it was believed long enough to
  nearly abandon the fix. It had taken the *median* inter-token latency, which
  is 12.1 ms whether the other cards are busy or idle, because the barrier
  does not slow tokens down — it freezes them, 21 times out of 199, for the
  length of somebody else's prefill chunk. The mean was **12.1x**. Summary
  statistics choose which failures they can see: report the mean and the tail
  for anything whose cost arrives in stalls.
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
- **Two agents agreeing on a mechanism is the same guess twice.** A −5%
  regression was correctly localized to a force-inlined runtime dispatch
  ladder, and both agents then independently reasoned that hiding the extra
  arms behind `__noinline__` would contain it. It made things worse —
  `moe_expert_ffn` 80 → 124 registers — because the ABI boundary spills what
  the inlining had kept live. The diagnosis was measured and right; the fix
  was reasoned and wrong, and it was written into a commit message as *the*
  fix before being compiled once. Concurrence is not corroboration when both
  parties are reasoning from the same model.
- **"The sweep is non-monotonic, so there is nothing there" — right answer,
  and the reasoning was still wrong.** Two agents read the router expert
  tile's sweep that way. Non-monotonicity across single runs is the signature
  of a method that cannot resolve the effect, not of an absent effect, and
  inferring absence from a measurement that could not have detected presence
  is an error whatever the subject — so three interleaved pairs were run, the
  tile won all three at +1.31%, and that was written up as the sweep having
  understated it threefold. It had not. Re-measured against a baseline built
  from the pinned commit, the same change reads ±0.2% with the sign flipping
  inside the set. **Both readings were unresolved, and the confident one was
  the more expensive mistake**: an absent effect confidently rejected costs
  an opportunity, an absent effect confidently *measured* costs the standing
  and everything reasoned from it afterwards. Ask what the method's
  resolution is before reading either shape — on this host one run per point
  resolves nothing below about 1.5%, and one three-pair set against a box
  drifting 0.8% in ten minutes resolves less than it appears to.
- **A green test target is not a test that ran.** `forward_pass` reports
  `ok, 8 passed` in 0.36 s when it cannot find the golden, because a skip is
  a pass and cargo captures stdout. `.golden/` is gitignored, so **every run
  from a `git worktree` skips it silently** unless `LLMXABE_GOLDEN` points at
  the main checkout's copy. Related: a `cargo test | grep` pipeline reports
  grep's exit status, and a sweep that died at target 29 of 60 is
  indistinguishable from one that passed. All three were live in one
  afternoon; `docs/TESTING.md` carries the checks.
- **`ptxas` allocates across a wider scope than the edit.** Adding a branch to
  one device function moved an unrelated `__global__` function's register count
  from 80 to 126, and removing a branch from two near-identical sibling kernels
  moved their allocations in opposite directions. Register math is a proxy;
  `cuobjdump -sass` diffing every kernel against its stated blast radius is the
  check.
- **"It was verified" — at the widths it was benched at.** The router's narrow
  expert tile was swept at N=3, confirmed under pairs at N=3, and checked for
  bit-identity at N=1 and N=3. It shipped sizing a launch for one
  instantiation's dynamic shared memory and dispatching another's kernel at
  exactly `max_tokens == 2`. Two rules derived the tile widths; they agreed at
  every width anyone had reason to look at and disagreed at the one no
  benchmark produces. Nothing asked which widths **exist** — speculative
  decode drives the scheduler at width 2 and no bench does. A gate scoped by
  *what the change is* only ever confirms what is already believed; scope it
  by the entry points the edit lands in and the shapes the scheduler puts
  through them. The same afternoon and the same mistake in the other
  direction: an edit's *shape* was verified and not its *location*, and a
  `replace_all` put a decode-only fork into the uncaptured prefill path as
  well. The tile was then re-measured against a clean baseline and had no win
  to defend; the fork has not been re-measured and its number stands only as
  history. **The correctness fault is what sent anyone back to the
  measurement at all** — without it both would have kept their figures.

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

The dense (`qwen35`) rows use the same harnesses, which take the model by
environment variable — nothing else about them changes for a second
architecture:

```sh
M=Qwen3.8-27B-UD-Q8_K_XL.gguf
LLMXABE_MODEL="$M" CUDA_VISIBLE_DEVICES=1 ./target/release/bench_forward
LLMXABE_MODEL="$M" CUDA_VISIBLE_DEVICES=1 ./target/release/bench_decode

# N=3. bench_decode_batch prints aggregate and per-seq columns itself, and
# carries its own single_stream baseline in the same process.
LLMXABE_MODEL="$M" LLMXABE_PREFILL_SEQUENCES=3 LLMXABE_BENCH_CHUNK=1536 \
  LLMXABE_BENCH_N=512 CUDA_VISIBLE_DEVICES=1 ./target/release/bench_forward
LLMXABE_MODEL="$M" LLMXABE_BATCH_N=1,3 CUDA_VISIBLE_DEVICES=1 \
  ./target/release/bench_decode_batch 512 64

CUDA_VISIBLE_DEVICES=1 llama-batched-bench -m "$M" \
  -ngl 99 -sm none -fa on -b 2048 -ub 2048 -ctk f16 -ctv f16 \
  -npl 1 -npp 512 -ntg 64        # -npl 3 for the N=3 rows

# The dense FFN alone, at decode widths, with the card's measured streaming
# ceiling printed above the table. ~10 s per A/B against `bench_decode`'s
# ~90 s, which is what makes an inner-loop change measurable at all.
LLMXABE_MODEL="$M" CUDA_VISIBLE_DEVICES=1 ./target/release/bench_dense_ffn
LLMXABE_DENSE_SPLIT_ROWS=8 ...                  # override the RT table
LLMXABE_FFN_N=4,6,8,12,16,32 ...                # the GEMV/GEMM width sweep

# A second qwen35moe checkpoint. Requant to the shipped mix first — a
# uniformly-Q6_K file stops at the first projection. Recipe in MODEL.md.
LLMXABE_MODEL=Ornith-1.5-35B-A3B-UD-Q6_K_XL.gguf CUDA_VISIBLE_DEVICES=2 \
  LLMXABE_PREFILL_SEQUENCES=3 LLMXABE_BENCH_CHUNK=6138 LLMXABE_BENCH_N=130944 \
  ./target/release/bench_forward     # depths are multiples of 6138/3 = 2046

# The speculative table. `LLMXABE_SPEC` takes any of the seven drafter names.
LLMXABE_MODEL="$M" LLMXABE_SPEC=draft-mtp CUDA_VISIBLE_DEVICES=1 \
  ./target/release/bench_worker_spec           # context 256, 512 tok/seq

# Per-kernel attribution of a decode step. `--cuda-graph-trace=node` is not
# optional: decode replays a captured graph and without it nsys attributes one
# pass and drops the rest.
nsys profile -t cuda --cuda-graph-trace=node -s none -o d \
  ./target/release/bench_decode 128 20
nsys export --type sqlite -o d.sqlite d.nsys-rep
```

The requantized files, and the two the engine cannot read. These are
generated artifacts rather than anything shipped, so a checkout will not have
them until this is run:

```sh
llama-quantize --allow-requantize "$M" Qwen3.8-27B-req-q8_0.gguf  q8_0   12
llama-quantize --allow-requantize "$M" Qwen3.8-27B-req-q6_k.gguf  q6_k   12
llama-quantize --allow-requantize "$M" Qwen3.8-27B-req-q4_k_m.gguf q4_k_m 12
```

*Why* the last two cannot be read — and which formats each kernel family has
a reader for at all — is
[MODEL.md](MODEL.md#which-formats-the-engine-actually-reads). It is not an
oversight: a k-quant projection has no path to the integer tensor cores, so
reading one would cost the prefill advantage it was quantized to preserve.

Interleave the *files* within each repetition, not the repetitions within each
file — this table compares models, and the card drifts.

For the greedy-agreement table, `llama-completion ... -no-cnv < /dev/null` —
`llama-cli -no-cnv` is not honoured in build 10456 and enters conversation
mode, which applies the chat template and makes the two sides incomparable.

The serving tables come from a running server rather than a kernel harness:

```sh
CUDA_VISIBLE_DEVICES=0 ./target/release/llmxabe \
  --spec-type none -c 405504 -s 3 --token-budget 4096 --prefill-chunk 2048 &
python3 tools/serving/bench_server.py http://127.0.0.1:8000 out.json \
  '{"depths":[2800,5500,11000,22000],"sessions":[1,2,3],"trials":3,"max_tokens":48}'
```

`depths` are word counts, not tokens — the harness measures each prompt's real
length through `usage.prompt_tokens` and reports against that. Drop
`CUDA_VISIBLE_DEVICES` for the three-card fleet and raise `sessions` to 9.
`--prefill-slice 0` restores whole-step prefill, which is how the sharing A/B
was run.

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
- The dense model anywhere except the cells in its own section: N=1 and N=3 at
  one depth only — no depth sweep, no N>3, no server, no vision.
- **Any accuracy check on the requantized files.** The `all-Q8_0`, Q6_K and
  Q4_K_M files were produced with `--allow-requantize` from an already-quantized
  source and measured for speed alone. The greedy-agreement table is about the
  shipped `UD-Q8_K_XL` file. Do not ship a requant on the strength of the
  throughput table.
- **Narrowing the activations, in the dense GEMV and the GDN gates alike.**
  Both read them fp32 against int8 weights, so an activation costs four bytes
  where the weight it multiplies costs one, and that ratio — not the weight
  traffic — is what each kernel's remaining cost is made of.

  In the dense split GEMV it is `4 * TT / RT` reads per weight byte, which is
  why the block costs 0.51 ms at one token and 0.68 at four for *identical*
  weight bytes, and why `RT` is worth raising until the register file stops
  it. In the GDN gate kernel it is starker: per warp per 32-element step the
  activations are 8 sectors against the quants' 4, so the misaligned Q8_0
  read that the Layout section calls out is only 17% of that kernel's sector
  budget. **Fixing the alignment there is not worth building** — it was
  costed at 2x on the strength of the layout rule and is worth ~0.3% of the
  step once the activation sectors are counted. This entry exists so the
  next reader does not re-derive that the expensive way.

  fp16 activations halve the ratio in both, and cost the decode path the
  exactness the residency section credits it with — the one place this engine
  is more accurate than llama.cpp, which quantizes activations to Q8_1 here.
  Not built, not measured, and the trade is a decision rather than a tuning
  question.
- **Fusing the dense FFN's SwiGLU into the up projection and its residual add
  into the down projection.** Both are arithmetically free — the ADD path is
  already a template parameter the dense entry points do not instantiate — and
  together they retire 128 launches of a 51.8 ms step, worth about 0.5%. Not
  taken: the residual add is the only thing on that path gated on the device
  `valid_tokens` scalar (AGENTS.md rule 5), and folding it into a GEMV that
  writes every row of the buffer would put a plausible value on a slot the
  pass never filled, which is exactly what `forward_pass` checks for. It is
  recoverable by threading the scalar into the GEMV's epilogue; 0.5% did not
  justify moving that gate.
- **k-quants anywhere but the dense FFN loader.** Q6_K halves the decode
  budget and llama.cpp reaches 22.13 tok/s with it on this model; the
  embedding gather, the LM-head GEMV family and the attention projections all
  refuse it by name. What a Q6_K *GEMV* would achieve on this card is
  unmeasured — llama.cpp's own Q6_K row streams 463 GB/s against Q8_0's 538,
  so the win is smaller than the byte count suggests.
