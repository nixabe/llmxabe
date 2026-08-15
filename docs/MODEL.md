# Model structure and resource budgets

Everything here derives from
[`xabe_model::ModelConfig`](../crates/xabe-model/src/config.rs). Run the
tables yourself:

```sh
cargo run -p xabe-model --example budget
```

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
| Expert weights | 32.34 B | 33.02 B | ~2% low |
| Projections | 1.305 B | ~1.44 B | ~9% low |
| Total (excl. MTP) | 34.66 B | 35.50 B | ~2.4% low |

Tensor type histogram: `q6_K` 80 tensors / 16.41 GiB, `q8_0` 303 / 13.86 GiB,
`f32` 368 / 0.10 GiB, `bf16` 2. Total tensor data 30.36 GiB.

**Quantization is mixed, and not the way the planning document assumed.**
Expert weights are Q6_K; the LM head *and the projections* are Q8_0. That
matters for the bandwidth arithmetic below.

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

3. **There is a short-convolution cache the config does not model.**
   llama.cpp maintains a separate conv cache (`n_embd_r()`) alongside the
   recurrent state — roughly 96 KiB per layer, ~2.8 MiB per sequence across
   30 layers. That is about 5% on top of the 60 MiB state, small but
   structurally absent from `gdn_state_bytes_per_sequence()`.

The residual 2.4% gap in total parameters is unexplained and should be closed
before any VRAM figure is treated as exact. It is a slight *under*-estimate,
so budgets derived from it are optimistic rather than dangerous.

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
