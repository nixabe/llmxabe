# Kernels

## Inventory

| Kernel | Layers | Risk | Plan | Status |
| --- | --- | --- | --- | --- |
| Gated DeltaNet, recurrent (decode) | 30 | **critical** | Delta rule, one token at a time | **sm_75 kernel, max_abs 2.98e-8 vs reference** |
| Gated DeltaNet, chunked (prefill) | 30 | **critical** | Forward substitution per chunk, not explicit inverses | **sm_75 kernel, max_abs 2.61e-8 vs reference** |
| GDN short convolution (depthwise, width 4) | 30 | medium | Causal depthwise conv over the fused qkv stream, before the delta rule | **sm_75 kernel, bit-identical to reference** |
| MoE dispatch + grouped GEMM | 40 | high | Port algorithm from vLLM; mixed Q6_K/Q8_0 dequant in prologue | **sm_75 kernel, max_abs 9.78e-9 vs reference** |
| Flash attention (GQA 16:2, head 256) | 10 | medium | Online softmax, `BM = 1`, scalar fp32 (no tensor cores yet) | **sm_75 kernel, max_abs 1.60e-6 at a 128K window** |
| LM head GEMV (2048 × 248,320) | 1 | medium | ~~split-K~~ — one warp per row; dominates weight bandwidth | **sm_75 kernel, argmax exact, 81–89% of roofline** |
| `moe_align_block_size` equivalent | 40 | medium | Write; must be on-device for graph capture | **sm_75 kernel, tables exact vs reference** |
| mRoPE (64 of 256 dims) | 10 | low | Write; partial rotary is unusual — test carefully | **sm_75 kernel, tail bit-identical** |
| Dequant (Q6_K, Q8_0) | all | low | Port llama.cpp K-quant unpacking | **sm_75 kernel, bit-identical to reference** |
| Router top-k over 256 experts | 40 | low | Write; warp-level bitonic | **sm_75 kernel, expert IDs exact vs reference** |
| RMSNorm, SwiGLU, residual | all | low | Write; RMSNorm runs at three widths (2048, 256, 128) | **sm_75 kernels, max_abs 3.10e-6 vs reference** |
| Vision encoder | — | deferred | Out of scope — images stay on llama.cpp | n/a |

"CPU reference" means a scalar fp32 implementation exists in `xabe-kernels`
with differential tests, and no GPU kernel has been written. Where a device
kernel exists the measured agreement against that reference is quoted, because
"implemented" without a number is not a status. See [TESTING.md](TESTING.md).

The short convolution was missing from this table entirely until the weight
schema made it visible — `qwen35moe.ssm.conv_kernel = 4`, one
`ssm_conv1d.weight` per GDN layer. It is not optional and it is not folded
into the delta rule; see [MODEL.md](MODEL.md).

> **A SiLU follows the convolution, and it is a separate op.**
> `src/models/qwen35moe.cpp:422` applies `ggml_silu` to the convolution output
> *before* q, k and v are sliced apart and L2-normalized. Both this table and
> [MODEL.md](MODEL.md) previously described only the convolution, so anyone
> wiring a GDN layer from the docs alone would have dropped the activation and
> produced a plausible, wrong model.
>
> `xabe_kernels::conv::causal_depthwise_conv1d` is deliberately the pure
> convolution, matching `ggml_ssm_conv`'s boundary exactly; the caller applies
> the SiLU. Whoever assembles the GDN block owns that step.

### What the landed kernels do not yet do

Every kernel above is gated on **correctness against the CPU reference**, and
none of them has been tuned. Three limits are worth stating so the numbers
above are not read as more than they are:

- **No tensor cores anywhere.** Attention is the scalar fp32 path. The
  `m16n8k8` MMA family that `compute_75` makes reachable runs at roughly 8×
  the fp32 FMA rate, and none of that is in hand. It is also unreachable at
  the current `BM = 1` shape — `m16n8k8` needs 16 query rows resident, which
  means re-tiling — and fp16 operands would end the fp32 comparison the kernel
  is gated on. The re-tiling that unlocks MMA is the same change that raises
  arithmetic intensity, and at ~0.5 FLOP/byte attention is bandwidth-bound, so
  the intensity is the half that pays.
- **The MoE dispatch kernel is single-block.** Correct and deterministic, fine
  at decode shapes, not tuned for large prefill batches.
- **Graph capture is argued, not demonstrated.** The MoE path has fixed grids,
  fixed buffers and a device-side token count precisely so it can be captured,
  but no capture has been performed yet. That is milestone 06's gate, and
  until it runs this remains a structural claim.
- **Batch tiling in the LM head is sublinear.** Eight tokens cost 2.58× one
  token, not the 8× that reading the weights once instead of eight times would
  allow, because activation loads scale with the tile while weight loads do
  not. A register tile over rows is the next lever and is neither implemented
  nor measured.

**This table planned the LM head as split-K. That was wrong**, and the row now
says so. Split-K manufactures parallelism when the output dimension is too
small to fill the machine — the MoE decode regime, where three tokens meet a
512-row expert matrix. The LM head is the opposite: one warp per output row is
248,320 warps against 2,304 resident, a 107× surplus. Splitting K would add a
launch, a `vocab × K` partial buffer and a split-dependent summation order for
no occupancy gain. The rejection is asserted in a unit test so it fails loudly
if the vocabulary ever shrinks.

**Shared memory on this hardware is 48 KiB per block**, not 64 KiB. The 64 KiB
figure is per-SM; a block reaches it only by opting in through
`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`, which cudarc's
`LaunchConfig` does not expose. Both landed kernels that budget shared memory
(attention at 1,088 B and the chunked GDN solve at 33 KiB) are sized against
48 KiB and need no opt-in.

## Order of work

**Gated DeltaNet is the critical path, not attention.** It covers 30 of 40
layers, has no equivalent in any flash-attention codebase, and its prefill form
requires per-chunk triangular matrix inversion. If it does not come together,
nothing else matters.

That is why milestone 01 deliberately precedes the differential harness in
milestone 02: you want to know in week three, not week twelve.

Milestone 06 — graph capture over on-device MoE indirection — was the
continue/stop gate for the project as a whole. Benchmarking has since shown
llama.cpp already does both graph capture and MoE fusion, so the gate must be
restated as measured throughput against the baseline's 104.79 tok/s rather than
as "does graph capture help". See [BENCHMARKS.md](BENCHMARKS.md).

### What the baseline already does

Anything written here has to beat, not merely match, the following:

| Capability | Where it lives in llama.cpp |
| --- | --- |
| MoE indirection (`mul_mat_id`) | `ggml/src/ggml-cuda/mmid.cu` |
| Fused top-k routing | `ggml/src/ggml-cuda/topk-moe.cu` |
| `ffn_up` + `ffn_gate` + GLU fusion | `ggml_cuda_should_fuse_mul_mat` |
| CUDA graph capture and replay | `USE_CUDA_GRAPH`, `cudaGraphLaunch` |
| Turing-tuned quantized matmul tiles | `mmq-config-turing.cuh` |

Measured consequence: 104.79 tok/s at short context, 44.5% of the bandwidth
roofline, with the weight path at 44.5% of peak and the KV path at ~80%. The
inefficiency is concentrated in the weight path, which is where kernel work
should go.

## Gated DeltaNet

Thirty of forty layers. Fixed-size recurrent state, ~2 MiB per layer per
sequence, constant regardless of position.

Two forms are needed:

- **Recurrent**, for decode. One token at a time, state carried forward.
  Straightforward, and the reference against which everything else is checked.
- **Chunked parallel**, for prefill. Processes `chunk_len` tokens at once,
  which requires inverting a lower-triangular matrix per chunk. This is where
  the difficulty lives.

The two forms must agree to tight tolerance on identical input. That
equivalence test is the single most valuable check in `xabe-kernels`, because
it is what will later reveal whether a GPU chunked kernel is wrong.

Reference: `ggml/src/ggml-cuda/gated_delta_net.cu` in llama.cpp, which is
Turing-validated, plus its CPU counterpart. vLLM's
`model_executor/layers/mamba/` GDN modules are a useful cross-check but assume
Ampere-or-later features in places.

## MoE

Every layer has a 256-expert MoE block, including all thirty GDN layers. There
is no dense-layer shortcut.

Naively that is 1,080 small GEMVs per decoded token — nine active experts,
three matrices each, forty layers. The fused path exists to collapse that into
one launch per layer.

### Dispatch

1. The router produces `topk_ids` — eight routed experts per token, plus the
   shared expert.
2. Tokens are sorted by assigned expert and each expert's run is padded to a
   block multiple, producing `sorted_token_ids` and `expert_ids`: a flat list
   where every block of rows belongs to exactly one expert.
3. One grouped-GEMM kernel walks blocks, reads its expert index from
   `expert_ids`, and gathers rows through `sorted_token_ids`. Scattered access
   becomes contiguous per block.
4. Gate and up projections fuse into a single pass, then SwiGLU.
5. Routing-weight application and the top-k reduction fuse into one kernel,
   accumulating in fp32.

Reference: `vllm/model_executor/layers/fused_moe/fused_moe.py` —
`fused_moe_kernel` and `moe_align_block_size`. The algorithm transfers to
Turing unchanged because it is an indexing strategy, not a tensor-core trick.

### Making it graph-capturable

This is the part that matters most, because dynamic routing is the obvious
thing that breaks capture.

**Build the indirection tables entirely on-device into fixed-size buffers,
gated by a `valid_tokens` scalar in device memory.** Nothing is sized by a
host-side value, so graph topology stays static across replays while routing
content varies freely.

Second: **size the grid from a typical-case token-count hint rather than the
worst-case padded length**, and have each CTA stride over additional tiles
through an outer loop. Worst-case grid sizing wastes launch resources on a
decode batch of three.

### Turing adjustments

- **SplitK is the decode-shape optimization.** At batch 3 against 2048×512
  matrices the GEMM is extremely skinny; SplitK supplies parallelism the M
  dimension cannot. Published work on vLLM's Mixtral kernel measured roughly
  20% from SplitK alone.
- **Block launch ordering matters for L2.** The same work found a low L2 hit
  rate in vLLM's MoE kernel and fixed it with grouped rather than row- or
  column-major launch order. Turing's L2 is 6 MiB — measured, see
  [DEVELOPMENT.md](DEVELOPMENT.md) — so this matters more here, not less.
- **The shared expert is always active.** Hoist it out of the routed path
  entirely: it never needs sorting or indirection, and it is one ninth of
  expert traffic.
- **Weights stay in Q6_K.** vLLM's grouped-GEMM paths assume fp8 or bf16; here
  dequantization fuses into the kernel prologue instead. Take llama.cpp's
  K-quant superblock unpacking for that, not vLLM's quantization path.

## The LM head

One matrix, 2048 × 248,320, and it costs **540 MB per decoded token** — roughly
58% of what all forty MoE layers read combined. See [MODEL.md](MODEL.md).

It deserves its own optimization for that reason alone. Two options:

- Keep it at Q6_K rather than Q8_0.
- Split it across cards and all-reduce the argmax. That is 2 KB of traffic,
  viable even over PCIe — but note it violates the otherwise-absolute rule that
  nothing crosses PCIe on the decode path, so it needs measuring before it is
  adopted.

## Turing constraints

- **No `cp.async`.** Double-buffering is by hand. Accept it; this workload is
  bandwidth-bound rather than latency-bound, so the loss is smaller here than
  it would be on a compute-bound one.
- **Tensor cores are the `m16n8k8` fp16 MMA family.** Reachable through inline
  PTX, which the milestone-00 spike verified works — see
  [TOOLCHAIN.md](TOOLCHAIN.md).
- **48 KiB shared memory per block**, 72 SMs, 6 MiB L2. Measured.
- **Warp shuffle and vote intrinsics lower correctly**, verified numerically.
  The router's top-k over 256 experts depends on them.

## Porting rules

**Port algorithms, not kernels.**

llama.cpp is the source for anything that must run on sm_75 — its CUDA backend
is the only one of the two references actually validated on Turing. vLLM is the
source for everything above the kernel: dispatch strategy, cache group
structure, scheduling policy, admission control.

vLLM's kernels assume fp8 or bf16 and Ampere-or-later features. Reimplement
their indexing strategy against llama.cpp's Turing-proven primitives.

When you port something, cite the file and function in the commit message.
Upstream paths drift on master, and a future reader needs to find what you were
looking at.

Relevant areas — treat as areas, not addresses:

| Need | llama.cpp |
| --- | --- |
| sm_75 flash attention | `ggml/src/ggml-cuda/fattn-tile.cu`, `fattn-vec.cuh` (not the WMMA path) |
| K-quant superblock unpacking | `ggml/src/ggml-cuda/dequantize.cuh`, `vecdotq.cuh`, `ggml/src/ggml-common.h` |
| Quantized matvec at decode shapes | `ggml/src/ggml-cuda/mmvq.cu` |
| Gated DeltaNet | `ggml/src/ggml-cuda/gated_delta_net.cu` |
| Recurrent vs KV memory split | `src/llama-memory-recurrent.cpp`, `src/llama-kv-cache*.cpp` |
| GGUF layout | `ggml/src/gguf.cpp` |

| Need | vLLM |
| --- | --- |
| Fused MoE, sorted-token indirection | `vllm/model_executor/layers/fused_moe/fused_moe.py` |
| Gated DeltaNet forward | `vllm/model_executor/layers/mamba/` |
| Hybrid cache groups, page geometry | `vllm/v1/core/kv_cache_utils.py`, `kv_cache_coordinator.py` |
| Chunked prefill, admission | `vllm/v1/core/sched/scheduler.py` |

## Correctness

A kernel without a passing differential test is not done, regardless of how
fast it runs. Numerics drift is the highest-likelihood risk in this project:
the engine stays fluent while getting quietly worse, and no throughput
benchmark catches it.

See [TESTING.md](TESTING.md) for thresholds and harness usage.
