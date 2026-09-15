# Model-dependent constants audit

Scope: production model loading, auxiliary-model setup, image preprocessing,
and tokenizer construction. These changes derive configuration before worker
allocation; they do not add metadata reads or allocation to the inference loop.

| Location / assumption | Implementation |
| --- | --- |
| Text loader selects a fixed model-size preset | `ModelConfig::from_gguf` reads the supported architecture's metadata; tensor schemas independently validate shapes. |
| MTP draft count alone enables the head | CLI selection and the complete declared MTP tensor set are both required; missing weights fall back to ordinary decode. |
| Vision worker assumes the Qwen3.6 tower and a 2048-wide projector | `VisionConfig::from_gguf` reads the mmproj's layer count, widths, heads, native image and patch size, merge size, epsilon, channel means and standard deviations. It checks output width against the target. |
| HTTP preprocessing uses a target-derived vision preset | Server preflight reads the same mmproj metadata as the worker, including normalization and patch geometry. Patch capacity derives from the image-token budget and parsed merge factor. |
| DFlash worker assumes six blocks, fixed heads/widths, tap layers, mask ID and draft block size | `DFlashConfig::from_gguf` reads those fields, normalization, rotary base and sliding-window pattern from the drafter. Target width, mask bounds, tap bounds/uniqueness and pattern lengths are checked before upload. |
| Token IDs >= 248,000 are treated as special | Tokenizer construction and grammar pieces use `tokenizer.ggml.token_type`; shifted special IDs work without a new cutoff. Missing/invalid type tables fail explicitly. |
| Qwen pre-tokenization silently applies to every tokenizer | Construction verifies `tokenizer.ggml.model = gpt2` and `tokenizer.ggml.pre = qwen35` before selecting the implemented regex. |

## Constants retained deliberately

- GGUF quantization block sizes and byte layouts describe the serialized format,
  not a model size. Changing Q8_0's 32 elements or Q6_K's 256 elements would corrupt
  decoding.
- CUDA warp sizes, launch widths, prefill tail buckets, GDN chunk length, and
  measured tile choices are hardware or execution policy. Replacing them with
  metadata has no defined meaning. Performance changes still require the
  measurement discipline in [BENCHMARKS.md](BENCHMARKS.md).
- The implemented vision family is `qwen3vl_merger`, with two temporal slices,
  a 2x2 merger, GELU, and rotary base 10,000. The installed GGUFs do not encode
  every one of these family semantics. Unsupported merger/deepstack variants
  fail instead of inheriting a configuration that would execute incorrectly.
- DFlash execution supports the Qwen-style dense backbone sharing the target's
  embeddings and output head. Separate heads and DSpark/DeepSeek variants need
  execution support; parsing their dimensions alone does not implement them.
- Qwen chat/tool marker spellings and pre-tokenizer regex are protocol and
  tokenizer-family semantics. Numeric token IDs are resolved from vocabulary.
- CLI defaults (budgets, batch/chunk sizes, image limits) are user-overridable
  resource policies. Reference presets and fixed kernel benchmark fixtures
  intentionally keep the target model geometry for reproducible comparisons.

## Validation

Synthetic auxiliary GGUFs exercise altered geometry and malformed metadata;
real-file tests compare both installed mmproj schemas and the installed DFlash
schema to parsed configuration. Tokenizer tests cover token types below the old
ID cutoff and the real model vocabulary. GPU numerical validation is separate:
passing configuration or schema tests does not establish that a new model size
runs correctly or quickly on the GPU.
