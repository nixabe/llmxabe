# Milestones

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
| 06 | CUDA graph capture | Was "the justification gate"; llama.cpp already does this — see [BENCHMARKS.md](BENCHMARKS.md) | **done**, and worth ~0 on throughput; kept for host cost and launch-shape discipline |
| 07 | Two-group pager + scheduler | 3 slots, matches llama.cpp `-np 3` | **done** — batched decode and flattened batch prefill at N=1–8, scheduler path within 0.7% of the isolated kernel path |
| 08 | Multi-worker + router + shared cache | Hit rate ≥ llama.cpp baseline | **done** — nine requests balanced three per card, mixed decode/prefill steps on every worker, cross-worker restore emits identical token ids |
| 09 | MTP speculative decode | Accept rate vs n-gram baseline | implemented, **not adopted** — 65.2% acceptance for ~7%, and unbatched across sequences |

## What the numbering does not cover

The plan's list ends at 09, and the serving surface sits outside it. The HTTP
dialects and their streaming, the tokenizer built from the GGUF's own
vocabulary, the prefix-cache sizing and the command line all landed after the
engine milestones and have no gate in the plan, so they get no row here rather
than an invented one.

Their state is [API.md](API.md) and [CLI.md](CLI.md). What is and is not
actually checked about them is the "Serving acceptance status" section of
[TESTING.md](TESTING.md) — worth reading before trusting this table's "done"
column to mean the whole product works, because the two cover different
things.

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
llama.cpp's own best settings, which is what
[BENCHMARKS.md](BENCHMARKS.md)'s standing table reports.
