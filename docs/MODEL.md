# Model structure and resource budgets

Everything here derives from
[`xabe_model::ModelConfig`](../crates/xabe-model/src/config.rs). Run the
tables yourself:

```sh
cargo run -p xabe-model --example budget
```

## Two architectures

The engine serves two models. They share the hybrid layer pattern, the
tokenizer, the partial rotary geometry and the vision tower, and differ in the
feed-forward block and in every width:

| | `qwen35moe` | `qwen35` |
| --- | --- | --- |
| Model | Qwen3.6-35B-A3B | Qwen3.8-27B |
| Feed-forward | 256 experts, 8 routed + 1 gated shared | one dense SwiGLU MLP |
| Layers (+ MTP) | 40 (+1) | 64 (+1) |
| Hidden | 2048 | 5120 |
| GDN heads | 32 V, 16 QK, head dim 128 | 48 V, 16 QK, head dim 128 |
| Attention heads | 16 Q, 2 KV, head dim 256 | 24 Q, 4 KV, head dim 256 |
| FFN intermediate | 512 per expert | 17,408 |
| Total / active params | 35 B / ~3 B | 27 B / 27 B |
| KV per token, f16 | 20 KiB (10 layers) | 64 KiB (16 layers) |
| GDN state per sequence | ~60 MiB (30 layers) | ~144 MiB (48 layers) |

Which one a file is comes from its own `general.architecture`, read at
startup; `ModelConfig::for_architecture` has no fallback, because inferring
hyperparameters from tensor shapes would produce a model that loads and is
wrong. The rest of this document describes `qwen35moe` unless it says
otherwise; the dense model has its own section at the end.

## Structure

| Property | Value |
| --- | --- |
| Total / active params | 35B / ~3B |
| Layers | 40 |
| Hidden layout | 10 × ( 3 × (Gated DeltaNet → MoE) + 1 × (Gated Attention → MoE) ) |
| Gated DeltaNet layers | 30 |
| Gated Attention layers | 10 |
| Hidden dimension | 2048 |
| GDN heads | 32 V, 16 QK — head dim 128 |
| Attention heads | 16 Q, 2 KV — head dim 256, RoPE dim 64 (25%) |
| MoE | 256 experts, 8 routed + 1 shared, expert intermediate 512 |
| Vocabulary (untied) | 248,320 in and out |
| MTP | trained multi-step |
| Context | 262,144 native, 1,010,000 with YaRN |

`verify::check_config` re-derives the parameter count from these fields and
rejects anything outside a plausible band, which catches a transcription error
in a field that is individually plausible — setting `hidden_size` to 4096, for
instance.

### Three consequences

1. **MoE is on every layer**, including all 30 GDN layers. No dense-layer
   shortcut.
2. **Only 10 layers hold growing KV.** A 384K pool costs 7.5 GiB, not the
   30 GiB a full-attention 35B would need.
3. **30 layers hold fixed-size recurrent state** — ~2 MiB per layer, ~60 MiB
   per sequence, constant regardless of position.

## Verified against the real file

The following were checked against
`Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf` (32.6 GB, 753 tensors, 55 metadata keys)
rather than taken on trust.

| Quantity | `ModelConfig` | Real GGUF | Verdict |
| --- | --- | --- | --- |
| Embeddings (in + out) | 1.017 B | 1.017 B | **exact** |
| LM head alone | 0.509 B | 0.509 B | **exact** |
| Expert weights | 32.338 B | 32.212 B | +0.39% |
| Total (text path) | 34.660 B | 34.661 B | **−0.001%** |

These are now produced by a test rather than by hand:
`xabe_model::weights::WeightSchema` derives every expected tensor name and
shape from `ModelConfig` alone, and
`crates/xabe-model/tests/real_model_weights.rs` resolves it against the file.
All 753 tensors match, with none left unclaimed.

Tensor type histogram: `q6_K` 80 tensors / 16.41 GiB, `q8_0` 303 / 13.86 GiB,
`f32` 368 / 0.10 GiB, `bf16` 2. Total tensor data 30.36 GiB.

**Quantization is mixed, and not the way the planning document assumed.**
The LM head *and the projections* are Q8_0. That matters for the bandwidth
arithmetic below.

It is also mixed *within the expert stacks*, which an earlier revision of this
document got wrong by calling them simply "Q6_K". Verified tensor by tensor
against the file's directory, not inferred:

| Expert tensor | Blocks 0–38 | Block 39 | Block 40 (MTP) |
|---|---|---|---|
| `ffn_gate_exps` | `q6_K` | **`q8_0`** | `q6_K` |
| `ffn_up_exps` | `q6_K` | **`q8_0`** | `q6_K` |
| `ffn_down_exps` | `q8_0` | `q8_0` | `q8_0` |

That accounts for the histogram exactly: 80 `q6_K` tensors is 40 blocks × 2
(gate and up) with block 39 excluded, and the 41 down projections plus block
39's gate and up are 43 of the `q8_0` count.

Two consequences, both load-bearing:

- A dequantization prologue that hard-codes Q6_K reads every expert down
  projection as garbage. The format must travel with the pointer — see
  `QuantTensor` in `crates/xabe-cuda/src/kernels/moe.rs`.
- Nothing may assume a uniform format *per layer* either. Block 39 is the
  counterexample, and it sits between two blocks that do match the pattern, so
  a spot check on layers 0 and 20 would miss it.

The shared-expert tensors (`ffn_*_shexp`) are Q8_0.

### Three findings from the real file

1. **The GDN state shape is confirmed, not merely derived.** The planning
   document carried 32 × 128 × 128 × fp32 per layer as an unverified
   derivation. It matches llama.cpp's own `n_embd_s()` formula
   (`ssm_d_state 128 × ssm_d_inner 4096`) computed from the file's metadata,
   and every GDN weight tensor shape agrees. This closes an open question.

2. **`block_count` is 41, not 40.** Block 40 is an extra attention-type layer
   carrying the MTP (`nextn.*`) tensors — about 0.008 B parameters.
   `ModelConfig::num_layers = 40` correctly describes the repeating base
   stack, so this is not a bug, but a loader that assumes `num_layers` covers
   every `blk.N` tensor in the file will be wrong.

3. **There is a short convolution the kernel inventory did not list.**
   Every GDN layer carries `ssm_conv1d.weight` of shape `[4, 8192]` —
   `qwen35moe.ssm.conv_kernel = 4` — a causal depthwise convolution over the
   fused q/k/v stream, applied before the delta rule. `ModelConfig` counts its
   parameters, but [KERNELS.md](KERNELS.md) had no entry for the kernel and
   `gdn_state_bytes_per_sequence()` does not model its cache: llama.cpp keeps
   a separate conv cache (`n_embd_r()`) of roughly 96 KiB per layer, ~2.8 MiB
   per sequence across 30 layers. That is ~5% on top of the 60 MiB state —
   small, but structurally missing rather than rounded away.

> **A previously reported 2.4% gap was an accounting error, not a config
> error.** An earlier hand-rolled comparison put `ModelConfig` 2.4% below the
> file. The whole difference was `blk.40`: the MTP head is a complete
> dense-attention block with its own 256-expert MoE — 0.84 B parameters — and
> it was being counted on the file side but not the config side. Compared like
> for like on the text path, the derivation is accurate to **0.001%**.

## VRAM budget per card

At context 393,216, 3 slots, f16 KV, Q6_K_XL weights, against 47.3 GiB usable
(measured — see [DEVELOPMENT.md](DEVELOPMENT.md)):

| Segment | GiB | Confidence |
| --- | --- | --- |
| Weights | 29.60 | measured |
| KV pool | 7.50 | derived |
| GDN recurrent state × 3 slots | 0.18 | derived |
| Compute buffers | 2.20 | **estimate** |
| CUDA context + cuBLAS workspace | 0.50 | **estimate** |
| **Total** | **39.98** | |
| **Headroom** | **7.52** | |

This runs below the planning document's ~41.3 GiB because it **excludes the
vision encoder**. This engine is text-only, so the ~1.5 GiB `mmproj` cost does
not apply; that is a scope difference, not an error.

The two estimate rows are inherited unmeasured and are the next thing to
measure.

**Trade available:** `UD-Q5_K_XL` (26.6 GB) frees ~4.9 GiB — another ~250K
tokens of KV pool, or a fourth slot. The quality cost is a measurement, not an
argument.

## Bandwidth

This is the section that most changed on contact with the real file.

### Per-token weight read

Using the quantization actually observed tensor by tensor — Q6_K experts,
Q8_0 LM head and projections:

| Component | Params | Bytes/token |
| --- | --- | --- |
| Active experts (9 × 3 mats × 40 layers) | 1.132 B | 929 MB |
| LM head (248,320 × 2048, Q8_0) | 0.509 B | 540 MB |
| GDN + attention projections (Q8_0) | 1.305 B | 1,387 MB |
| **Total weights** | | **2.86 GB** |

> **This differs from the planning document's 2.24 GB by about 28%.** Two
> causes, both verified against the file: the projections are structurally
> larger than that document's 0.944 B estimate, and they are stored at Q8_0
> rather than the Q6_K its arithmetic implied. The active-expert (929 MB) and
> LM-head (540 MB) figures match it to within 1%.
>
> Since the config's projection count is itself ~9% *below* what the real file
> holds, 2.86 GB is a floor, not a ceiling.

### Per-token KV read

20,480 B/token × context, re-read every decoded token:

| Context | KV GB/tok | Weights GB/tok | Total | Roofline @ 672 GB/s |
| --- | --- | --- | --- | --- |
| 4K | 0.084 | 2.856 | 2.94 | 229 tok/s |
| 32K | 0.671 | 2.856 | 3.53 | 191 tok/s |
| 128K | 2.684 | 2.856 | 5.54 | **121 tok/s** |
| 256K | 5.369 | 2.856 | 8.23 | 82 tok/s |

**These are ceilings derived from bandwidth.** They have now been measured
against llama.cpp on this hardware — see [BENCHMARKS.md](BENCHMARKS.md):

| Context | Roofline | Measured (llama.cpp) | Fraction |
| ---: | ---: | ---: | ---: |
| 4K | 229 | 103.4 | 45.2% |
| 32K | 191 | 92.4 | 48.5% |
| 128K | 121 | 68.8 | 56.7% |

The gap decomposes cleanly: the weight path achieves 44.5% of peak bandwidth,
the KV path about 80%. The rooflines are sound; the MoE weight path is what
falls short of them.

### Conclusions

1. **Long-context decode is KV-bound, but the crossover is later than
   assumed.** KV reads overtake weight reads at about **139K tokens**, not the
   ~109K the planning document implied. The hybrid architecture already saved
   4× on KV; what remains still comes to dominate, just further out.

2. **`-ctk q8_0 -ctv q8_0` was benchmarked, and it is worse.** −15.5% at 32K
   and −35.8% at 128K. The roofline argument for it assumed the KV path had
   headroom; it does not, running already at ~80% of peak. Turing
   dequantization inside the attention kernel costs more than halving the term
   saves. Keep f16.

3. **The LM head deserves its own optimization.** 540 MB/token for a single
   matrix — roughly 58% of what all forty MoE layers read combined. Either
   keep it at Q6_K rather than Q8_0, or split it across cards and all-reduce
   the argmax (2 KB of traffic, viable even over PCIe, but it breaks the
   otherwise-absolute rule that nothing crosses PCIe on the decode path). See
   [KERNELS.md](KERNELS.md#the-lm-head).

4. **Projections are now the largest single weight term**, at 1,387 MB/token —
   larger than the active experts. That is a direct consequence of their being
   Q8_0, and it makes requantizing them the most obvious available bandwidth
   win. Unmeasured.

---

# Qwen3.8-27B (`qwen35`)

The dense sibling. Everything above describes `qwen35moe`; this section is the
delta, and only the delta — the layer pattern, the partial rotary, the
tokenizer, the two-group cache geometry and the vision tower are the same
design at different widths.

## Structure

| Property | Value |
| --- | --- |
| Total / active params | 27 B / 27 B (26.90 B derived) |
| Layers | 64 (+ 1 MTP block, `block_count` 65) |
| Hidden layout | 16 × ( 3 × (Gated DeltaNet → FFN) + 1 × (Gated Attention → FFN) ) |
| Gated DeltaNet layers | 48 |
| Gated Attention layers | 16 |
| Hidden dimension | 5120 |
| GDN heads | 48 V, 16 QK — head dim 128 |
| Attention heads | 24 Q, 4 KV — head dim 256, RoPE dim 64 (25%) |
| Feed-forward | one dense SwiGLU MLP, intermediate 17,408 |
| Vocabulary (untied) | 248,320 in and out — byte-identical to Qwen3.6's |
| MTP | trained multi-step |
| Context | 262,144 native |

The GDN head split is **derived, not stated**. The file gives
`ssm.inner_size 6144`, `ssm.time_step_rank 48`, `ssm.group_count 16` and
`ssm.state_size 128`; llama.cpp's `qwen35.cpp` reads `n_v_heads` from
`ssm_dt_rank` and `n_k_heads` from `ssm_n_group`, with both head dimensions
equal to `ssm_d_state`. That gives 48 value heads and 16 q/k heads of 128, and
it is checked against the file's own tensor shapes rather than trusted:
`attn_qkv.weight` is `[5120, 10240]` and `2·16·128 + 48·128 = 10240`.
`crates/xabe-model/tests/real_dense_model_weights.rs` asserts all of it.

## Verified against the real file

`Qwen3.8-27B-UD-Q8_K_XL.gguf` — 29.29 GiB, **866 tensors**, 51 metadata keys.
`WeightSchema::with_mtp(&ModelConfig::qwen3_8_27b())` resolves against it with
zero mismatches and nothing left unclaimed.

Tensor type histogram: `q8_0` 453 tensors / 24.49 GiB, `bf16` 53 / 4.79 GiB,
`f32` 360 / 0.01 GiB.

**The bf16 tensors are the interesting part**, and they are not a curiosity:
they are 53 of the file's largest.

| Tensor | Count | Format |
| --- | ---: | --- |
| `output.weight` | 1 | **bf16**, 2.54 GiB |
| `blk.N.attn_q` / `attn_k` / `attn_v` | 51 | **bf16** |
| `blk.64.nextn.eh_proj.weight` | 1 | **bf16** |
| `blk.N.ffn_gate` / `ffn_up` / `ffn_down` | 195 | `q8_0` |
| `blk.N.attn_output`, `attn_qkv`, `attn_gate`, `ssm_out` | — | `q8_0` |
| `blk.N.ssm_alpha` / `ssm_beta` | 96 | **`q8_0`** (f32 in Qwen3.6) |
| norms, `ssm_a`, `ssm_dt.bias`, `ssm_conv1d` | 360 | `f32` |

Two of those rows cost kernel work rather than a config field:

- The LM head and the attention q/k/v projections go through the *same*
  GEMV (`xabe_cuda::kernels::lm_head`), which read Q8_0 only. It now has a
  bf16 body as well, selected per tensor from the file's directory — see
  [KERNELS.md](KERNELS.md). Widening those on the host was rejected on
  arithmetic: it would put 5.1 GiB on the card for the head alone and double
  the per-token read of the most bandwidth-expensive tensor in the model.
  Reading them where they are is correct, but it is not free: bf16 has no
  integer tensor-core path, so those three projections stay on the fp32 GEMV
  at prefill width. The repack is gated per tensor rather than per block, so
  `attn_output` still reaches the tensor cores beside them; what remains is
  the format itself. Requantizing the file to plain Q8_0 takes prefill from
  360 to 661 tok/s with no engine change, and costs 6.5% of decode besides —
  measured, in [BENCHMARKS.md](BENCHMARKS.md).
- `ssm_alpha` / `ssm_beta` are f32 in Qwen3.6 and Q8_0 here. The fused gate
  kernel has a Q8_0 instantiation rather than a host-side widening, because
  the forward path *aliases* the weight arena and an owned widened copy inside
  a `ManuallyDrop<GdnLayerWeights>` would leak once per pass shape.

## Per-token cost, and why the smaller model is the expensive one

This is the number to carry away, and it is the opposite of what the parameter
counts suggest:

| | Qwen3.6-35B-A3B | Qwen3.8-27B |
| --- | ---: | ---: |
| File | 30.36 GiB | **29.29 GiB** |
| FFN params read per token | 1.13 B | **17.11 B** |
| KV read per token per 1K context, f16 | 20 MiB | **64 MiB** |
| GDN state per slot | 60 MiB | **144 MiB** |

The dense model is the smaller file and reads **15× the feed-forward weight
per token**. That is the entire point of an A3B mixture, seen from the other
side, and it means no decode expectation may be carried across: a roofline
built for 2.86 GB/token does not describe a model that reads roughly ten times
that.
