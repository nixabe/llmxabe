# The performance model

The arithmetic every performance decision in this project is scored against,
and what of vLLM's and llama.cpp's designs transfers to sm_75.

This is the *model*; [BENCHMARKS.md](BENCHMARKS.md) is the *measurement*.
Where they disagree, the measurement wins. Numbers here are either quoted from
[BENCHMARKS.md](BENCHMARKS.md) / [MODEL.md](MODEL.md), derived with the
arithmetic shown, or cited to a source read directly.

Source trees this document was written against:

- vLLM, `/home/nixabe/vllm`, commit `d4801990a` (2026-08-15)
- llama.cpp, `/home/nixabe/llama.cpp`, commit `fd6863a69`, build 10456

Upstream paths drift. Treat the file references as areas, not addresses.

---

## 1. The bandwidth model

All rankings come from one model. It is stated here so every later number can
be checked.

### 1.1 Bytes per weight

From the ggml block layouts:

| Format | Block | Bytes | Bits/weight | Bytes/param |
| --- | --- | ---: | ---: | ---: |
| `Q6_K` | 256 weights | 210 (`ql` 128 + `qh` 64 + `scales` 16 + `d` 2) | 6.5625 | 0.8203 |
| `Q8_0` | 32 weights | 34 (`d` 2 + `qs` 32) | 8.5 | 1.0625 |

### 1.2 Per-token weight traffic at batch 1

Using the per-tensor quantization in [MODEL.md](MODEL.md), verified against the
real GGUF (expert `gate`/`up` are `Q6_K`, expert `down` is `Q8_0`; the shared
expert, every projection and the LM head are `Q8_0`):

| Component | Arithmetic | MB/token |
| --- | --- | ---: |
| Routed experts | 8 × 40 × (2·2048·512·0.8203 + 512·2048·1.0625) | 906.9 |
| Shared expert | 40 × 3·2048·512·1.0625 | 133.7 |
| LM head | 248,320 × 2048 × 1.0625 | 540.3 |
| Projections | 1.305 B × 1.0625 | 1,386.6 |
| **Total weights** | | **2,967.5** |

### 1.3 Per-sequence state traffic — the term that never amortizes

A Gated DeltaNet decode step must **read and write** the full recurrent state.
It does not fit anywhere on chip (62.9 MB), so both directions hit HBM:

| Component | Arithmetic | MB/token/sequence |
| --- | --- | ---: |
| GDN recurrent state (read + write) | 2 × 30 × 32 × 128 × 128 × 4 B | 125.8 |
| GDN conv state (read + write) | 2 × 30 × 96 KiB | 5.9 |
| **Total** | | **131.7** |
| Attention KV (read) | 20,480 B × depth | 0.0205 × depth |

**This term does not amortize across a batch.** Each sequence owns its own
state. It is the single most important structural fact for the concurrency
plan, and it is why the aggregate ceiling is what it is (§4.3).

### 1.4 Single-stream roofline

At batch 1, depth 0: 2,967.5 + 131.7 = **3,099.2 MB/token** →
672,000 ÷ 3,099.2 = **216.8 tok/s**, not the 235 tok/s
[MODEL.md](MODEL.md) states from weight bytes alone. Every "fraction of
roofline" figure taken against 235 is correspondingly a few points low.

An independent per-stage profile that sums the GGUF tensor directory and
divides by CUDA-event time agrees to within 0.3% on total necessary bytes and
1.9% on the ceiling, which is the only reason the derivations here should be
trusted at all.

### 1.5 Expert-activation density under batching

For 256 experts with top-8 routing and *uniform* routing, the expected number
of distinct experts touched per layer by a batch of N tokens is

```
D(N) = 256 × (1 − (1 − 8/256)^N)
```

Real routing is measurably **less** diverse than uniform, which helps.
DynaExq (arXiv:2511.15015, Table 1) measures Qwen3-Next-80B at 1.9% activated
at batch 1 and 39.2% at batch 32 where the uniform model predicts 1.95% and
46.8%. Treat `D(N)` as an **upper bound**; observed routing lands roughly 16%
below it at moderate batch. The same table reports 86.2% activation in
*prefill* at batch 32 — prefill is effectively dense.

### 1.6 The concurrency table

Weight bytes per step = 2,060.6 MB (LM head + projections + shared expert, read
once) + 113.377 MB × `D(N)` (routed experts, read once per distinct expert per
layer).

| N | `D(N)` | % of 256 | weights MB/token | + state | **total MB/token** | roofline tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.00 | 3.1% | 2,967.5 | 131.7 | 3,099.2 | 216.8 |
| 2 | 15.75 | 6.2% | 1,923.2 | 131.7 | 2,054.9 | 327.0 |
| 3 | 23.26 | 9.1% | 1,565.9 | 131.7 | 1,697.6 | 395.9 |
| 4 | 30.53 | 11.9% | 1,380.5 | 131.7 | 1,512.2 | 444.4 |
| 8 | 57.42 | 22.4% | 1,071.3 | 131.7 | 1,203.0 | 558.6 |
| 16 | 101.95 | 39.8% | 851.2 | 131.7 | 982.9 | 683.7 |
| 32 | 163.30 | 63.8% | 643.0 | 131.7 | 774.7 | 867.4 |
| 64 | 222.44 | 86.9% | 426.3 | 131.7 | 558.0 | 1,204.3 |

The table is slightly optimistic — by at most 84.7 MB on the once-per-step
term, 2.7% at N=1 and 0.3% at N=32 — because §1.2 does not enumerate the gate
and norm tensors the GGUF directory sum includes.

### 1.7 What the table says about strategy

llama.cpp holds a **flat ~41% of the concurrency-aware roofline** from one
sequence to three. Its 1.77× at three slots is 97% of the 1.83× the
expert-activation arithmetic permits, so "llama.cpp batches poorly" was never
true; batching captures essentially the entire benefit the arithmetic allows,
for either engine.

It also scales further than a `-np 3` extrapolation predicts: at 16 concurrent
sequences it reaches **382 tok/s decode, 3.64× its single-stream 105**. There
is no batching deficit to exploit. **This project has a kernel story, not a
concurrency story** — concurrency is table stakes, and every point of weight-
path efficiency is a point of throughput at every concurrency simultaneously.

---

## 2. What transfers from vLLM, and what does not

**Port algorithms, not kernels.** vLLM is the reference for everything above
the kernel — dispatch strategy, cache group structure, scheduling policy,
admission control. llama.cpp is the reference for anything that must run on
Turing. vLLM's grouped-GEMM paths assume fp8 or bf16 and Ampere-or-later
features; reimplement its *indexing strategy* against llama.cpp's Turing-proven
primitives.

### Continuous batching and the step loop

`vllm/v1/core/sched/scheduler.py`. One step admits a token budget rather than a
request count, so prefill chunks and decode steps share a batch. The rule that
falls out of it and is enforced in `xabe-sched`: the per-step token budget must
exceed `block_size + max_concurrent_decodes`, or a single decoding request
starves prefill admission and execution serializes to batch 1.

### Paged KV cache and block management

`vllm/v1/core/kv_cache_manager.py`, `block_pool.py`, `kv_cache_utils.py`.

Already matched by `xabe-cache`: block-hash chaining so identical prefixes
converge regardless of who inserted them, reference counting and eviction of
unreferenced blocks, and a retention interval for recurrent-state checkpoints
that must be a multiple of the attention block size.

Where `xabe-cache` is behind: vLLM's free list is intrusive and doubly linked,
giving O(1) removal from the middle where `evict_unreferenced` is an O(n) scan;
vLLM caches *partial* blocks, so a hit need not land on a block boundary; and
it pins the shared-prefix junction so the retention interval cannot drop the
one boundary that enables cross-request reuse.

Where `xabe-cache` is ahead, and vLLM's own source proves it: vLLM *requires*
uniform page size across cache groups (`kv_cache_utils.py`, "Breaking this
assumption is non-trivial due to memory fragmentation concerns"). It satisfies
that by raising the attention block size in tokens until the attention page
matches the Mamba page — for this geometry, a ~1,024-token attention block with
one recurrent checkpoint per block:

| | `xabe-cache` | vLLM |
| --- | --- | --- |
| Attention block | 256 tokens | ~1,024 tokens (derived) |
| Checkpoint interval `R` | 2,048 tokens (8 blocks) | = block size (1 block) |
| Snapshot : KV ratio over one `R` | 60 MiB : 40 MiB = 1.5 | 60 MiB : 20 MiB = 3.0 |
| Prefix-hit granularity | 256 attention / 2,048 GDN | 1,024 both |
| Page geometry | independent per group | uniform, enforced |

Finer attention hit granularity *and* a lower snapshot ratio, precisely because
it refuses the uniform-page constraint. The problem class is Marconi (MLSys '25,
arXiv:2411.19379), which is where the exact-match constraint on recurrent state
and reuse-forecasting admission come from.

### The fused MoE path

`vllm/model_executor/layers/fused_moe/fused_moe.py`, `moe_align_block_size.py`.
The sorted-token indirection — count per expert, exclusive-scan to bucket
offsets, scatter token ids into per-expert buckets padded to the tile width — is
the structure `xabe-cuda`'s dispatch uses. Two of vLLM's own tuning rules do
**not** transfer to this shape: `SPLIT_K` is 1 in every default config, and
`GROUP_SIZE_M` is 1 whenever `M // E` is small, which at decode it always is.

### CUDA graph capture

vLLM distinguishes a uniform decode batch from a mixed prefill/decode batch and
only fully captures the former, subordinating capture to what the attention
backend supports rather than the reverse. The widely-circulated "10–20% on
small-batch decode" figures could not be traced to a primary benchmark and are
**refuted for this engine's shapes** — see [BENCHMARKS.md](BENCHMARKS.md).
Capture is kept for host cost and for the launch-shape discipline it forces,
not for throughput.

### Gated DeltaNet and hybrid handling

`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`. Two structural
points worth copying: the forward core splits on prefill vs decode counts and
runs *different kernels* for each, with a fast path when the batch is pure
decode; and recurrent state is indexed by a **device-side index tensor**, not a
host-computed pointer — the same discipline as design rule 5, for the same
reason.

The batching property that matters: state is 4.2% of per-token bytes at N=1,
10.9% at N=8, 17.0% at N=32 and rising, because the numerator is fixed and the
denominator falls. Routed-expert weights stay the larger term until about
N ≈ 220, far beyond what 47.3 GiB of VRAM allows. vLLM's two named mitigations
are a narrower SSM state dtype (fp32 → fp16 halves 125.8 MB, worth 8.4% at
N=32) and `use_replayssm`, which skips the per-step full-state store on 15 of
every 16 steps. The latter is Mamba2-specific and **whether it is sound for the
delta rule is unverified**.

The specific vLLM kernels do not transfer: several are Blackwell-tuned and the
CUTE-DSL chunked path is Hopper+. Use
`ggml/src/ggml-cuda/gated_delta_net.cu`, which is Turing-validated.

---

## 3. What is reachable on sm_75

### Not available — do not plan around any of these

| Feature | Requires | Evidence |
| --- | --- | --- |
| `bfloat16` | cc ≥ 8.0 | `vllm/platforms/cuda.py` raises for cc < 8.0 |
| fp8 `e4m3` compute | cc ≥ 8.9 | `marlin_template.h`: "only supported for Ada Lovelace or newer" |
| `cp.async` / `ldgsts` | cc ≥ 8.0 | Marlin's sm_75 configs drop to `stages = 2`; llama.cpp's `cp_async_available()` gates on Ampere |
| `mma.m16n8k16` fp16 | cc ≥ 8.0 | llama.cpp decomposes into 2× `m16n8k8` on Turing |
| `mma.m16n8k16` / `m16n8k32` int8 | cc ≥ 8.0 | decomposed into 2×/4× `m8n8k16` |
| `__reduce_add_sync` | cc ≥ 8.0 | Marlin uses a `__shfl_down_sync` ladder on sm_75 |
| TMA, `wgmma`, thread-block clusters, async barriers | Hopper | — |
| FlashAttention 2 / 3 | cc ≥ 8.0 / Hopper | use llama.cpp's Turing path |
| DeepGEMM, CUTE-DSL GDN chunk path | Hopper+ | — |

`mma.m16n8k16` is a specific trap: NVRTC *accepts* it at `compute_75` and emits
PTX, then `ptxas` rejects it (`Feature '.m16n8k16' requires .target sm_80 or
higher`). NVRTC success is not proof of reachability — only PTX→SASS is.

### Available

| Feature | Rate on this card | How known |
| --- | ---: | --- |
| fp32 FMA | 16.31 TFLOP/s (datasheet), 17.9 measured | datasheet + microbenchmark |
| `mma.m16n8k8` fp16, fp32 accumulate | **97.6 TFLOP/s** | measured on this card |
| `mma.m16n8k8` fp16, fp16 accumulate | within 0.4% of fp32 accumulate | measured; **not** a throughput lever here |
| `mma.m8n8k4` | 49.1 TFLOP/s | measured — half of `m16n8k8`, not hidden headroom |
| `mma.m8n8k16` s8 → s32 | **~198 TOP/s** | measured |
| `__dp4a` | 50.9 TOP/s | measured |
| Shared memory | 64 KiB/SM; 48 KiB per block by default, **64 KiB via opt-in** | `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)`; cudarc 0.19.9 exposes it as a safe `CudaFunction::set_attribute` |
| CUDA graphs | no arch floor | driver feature |

This Quadro does **not** pay the GeForce fp32-accumulate MMA penalty: 97.6
against 104.8 fp16-accumulate is a ~7% gap, not a halving. fp32 accumulation is
therefore essentially free, which is what confines accuracy loss to operand
rounding and makes the tensor-core paths gateable at all.

**vLLM's Marlin MoE does not apply.** It ships a real fused sm_75 tensor-core
kernel, and it accepts only int4/int8 weights with fp16/int8 activations.
`Q6_K`'s two-level superblock scale hierarchy is not one of those formats.
llama.cpp's MMQ handles `Q6_K` and `Q8_0` natively with the same Turing int8
MMA instruction, which is the concrete instance of "port vLLM's indexing onto
llama.cpp's primitives".

---

## 4. The ceilings

### 4.1 Prefill is a compute problem, and fp32 cannot win it

At prefill width all 256 experts are touched, so a step reads
40 × 256 × 2.834 MB + 2,060.6 MB = **60.7 MB/token**.

| Bound | tok/s | derivation |
| --- | ---: | --- |
| Memory roofline | 11,070 | 672,000 MB/s ÷ 60.7 MB/token |
| **fp32 compute ceiling** | **3,346** | 16.31 TFLOP/s ÷ 4.874 GFLOP/token |
| fp32 at a realistic 40% of peak | 1,338 | |

llama.cpp reaches **~61% of the absolute fp32 ceiling**, which no
dequantize-then-FMA pipeline does — it is dotting in int8 on the tensor cores
(`ggml_cuda_should_use_mmq` returns true unconditionally when
`turing_mma_available`). A pure fp32 kernel would have to sustain more than 62%
of theoretical peak FMA throughput, with K-quant unpacking, gather indexing and
masked M tiles in the same loop, merely to tie. An independent estimate from
the measured per-stage profile agrees: fixing every matmul to a *realistic* 25%
of fp32 peak still lands **3.9× behind**.

**Stated plainly: prefill cannot be won on sm_75 without the integer tensor
cores.** That conclusion drove the whole integer path, and it held.

### 4.2 Decode is a memory problem, and needs no tensor cores

FLOPs per decoded token = 2 × 2.946 B active params = 5.89 GFLOP. At the
216.8 tok/s roofline that is 1.28 TFLOP/s — **7.8% of fp32 peak**. Decode stays
bandwidth-bound in pure fp32 up to roughly batch 256, far beyond anything
47.3 GiB of VRAM reaches.

The corollary held throughout: decode gains came from streaming efficiency,
grid shape and wave occupancy, not from arithmetic. Where a tensor-core decode
kernel does win — deep windows, past ~16K keys — it wins on softmax structure,
and it pays a 2× tensor-core issue tax for the precision split it needs.

### 4.3 Aggregate concurrency bottoms out on recurrent state

The §1.6 curve flattens because the per-sequence terms stop amortizing. As
N → ∞ at depth 0, the weight terms go to zero and state stays at 131.7
MB/token, so the absolute ceiling on aggregate decode per card is
672,000 / 131.7 = **5,102 tok/s** — set entirely by GDN state traffic. At 4K
depth, KV adds 84 MB/token and it falls to 3,116.

The practical ceiling is VRAM, not that asymptote. With 29.6 GiB of weights
there is ~17.7 GiB left; a slot costs 60 MiB of GDN state plus 20,480 B/token
of KV. Sixteen slots at 32K fit (10.9 GiB); thirty-two at 32K do not (21.9);
thirty-two at 16K do. **The concurrency ceiling on this card is 16–32 sequences
at realistic depth**, which is exactly where the §1.6 curve is still steep.

### 4.4 The precision line, and what would legally move it

The remaining measured gaps against llama.cpp decompose into precision and
consistency trades this engine forbids by mission — single-rounded fp16 `Q`,
fp16 `P·V` accumulation, and `mmvq`/dp4a activation quantization. Each was
built or derived, gated, and rejected; the evidence is in
[BENCHMARKS.md](BENCHMARKS.md).

Three named workstreams would reopen them legally rather than by loosening a
gate:

1. **Register-neutral rewrite of the MMA kernels' synchronization pattern.**
   The barrier-density gap against `fattn-mma-f16.cuh` is real (~5× at prefill,
   ~8× at decode) and every parameter-level route to it costs registers or
   shared-memory occupancy this card cannot fund at once. A structural rewrite
   that is register-neutral by construction is the only remaining route.
2. **fp16 accumulation with unconditional periodic renormalization.** fp16
   `P·V` fails on coherent `V` because growth between rescales is purely
   additive and unbounded in the number of terms. A renormalization that does
   not depend on a new maximum being found would defend against exactly that,
   paid for in rescale instructions rather than in gated correctness. It buys
   register *capacity*, not throughput — the accumulate rate is a wash on this
   card.
3. **Symmetric activation quantization, now that batch and single-stream decode
   are bit-identical.** The raw material both discontinuities needed is gone, so
   `mmvq` is a legal lever again in the narrow sense that both paths would
   quantize the same bits. It is still gated on producing identical codes,
   scales and results at every serving width, which the last attempt failed at
   `4.4e-5`. This is not a licence to go build it.

### 4.5 Not recommended

- **Splitting the LM head across cards and all-reducing the argmax.** Saves
  360 MB/token (11.6%) at batch 1 and essentially nothing under concurrency,
  violates the rule that nothing crosses PCIe on the decode path, and requires
  all three cards to serve one sequence — which destroys the three-worker
  architecture the project is built on. The two goals are mutually exclusive;
  pick concurrency.
- **Split-K anywhere in this model.** It manufactures parallelism when the
  output dimension cannot fill the machine. Neither the LM head (107× warp
  surplus) nor the routed-expert GEMM (36× surplus) lacks it.
- **L2-grouped launch order at decode.** vLLM's own rule sets `GROUP_SIZE_M`
  to 1 when `M // E` is small. Reserve it for prefill.
- **`q8_0` KV cache.** Measured worse everywhere, worst where it was supposed
  to help most.
- **TensorRT.** Not installed, and not a drop-in if it were — see
  [BENCHMARKS.md](BENCHMARKS.md).

---

## 5. Sources

### Papers

- Rui Pan et al., *Marconi: Prefix Caching for the Era of Hybrid LLMs*,
  MLSys 2025. <https://arxiv.org/abs/2411.19379> — exact-match constraint on
  recurrent state, reuse-forecasting admission; up to 34.4× token hit rate,
  71.1% lower TTFT.
- Amey Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference
  with Sarathi-Serve*, OSDI 2024. <https://arxiv.org/pdf/2403.02310> — chunked
  prefill and stall-free scheduling.
- Kexin Chu et al., *Dynamic Expert Quantization for Scalable Mixture-of-Experts
  Inference* (DynaExq). <https://arxiv.org/abs/2511.15015> — expert activation
  ratios used in §1.5 to calibrate the uniform-routing model.
- Less Wright et al., *Accelerating a Triton Fused Kernel for W4A16 Quantized
  Inference with SplitK work decomposition*. <https://arxiv.org/pdf/2402.00025>

### Vendor documentation

- NVIDIA Quadro RTX 8000 datasheet — 16.3 TFLOPS FP32, 130.5 TFLOPS tensor,
  672 GB/s, 4,608 CUDA cores, 576 tensor cores.
- NVIDIA Turing Architecture Whitepaper — second-generation tensor cores add
  INT8 and INT4; INT8 runs at 2× the FP16 tensor rate, corroborated by the
  Tesla T4's published 65 TFLOPS FP16 / 130 TOPS INT8 pairing.
- CUDA C Programming Guide §5.1 — 48 KiB default shared memory per block,
  64 KiB opt-in on compute capability 7.x.
- vLLM CUDA Graphs design doc, and Optimization and Tuning.
