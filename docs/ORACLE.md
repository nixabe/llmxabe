# The llama.cpp oracle

Story G006's gate is *"a full forward pass whose logits match llama.cpp"*.
Until this document existed there was no mechanism to obtain llama.cpp's answer
for a given prompt, so that gate was not executable. This is how it becomes
executable.

**What is captured:** one llama.cpp forward pass over a fixed 19-token prompt,
recording the final logits over all 248,320 vocabulary entries **and** 288
intermediate graph tensors — every block boundary in the 40-layer stack, plus
the internals of three Gated DeltaNet blocks and two Gated Attention blocks.

**Where it lives:** `.golden/qwen36-golden.bin`, 57,602,888 bytes.
`.golden/` is gitignored (`.gitignore:15`). It is regenerated, never committed.

**How it is read:** `crates/xabe-engine/tests/golden.rs`, which skips and says
so when the file is absent.

---

## 1. Provenance

| | |
| --- | --- |
| Model | `/home/nixabe/llama.cpp/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf` (32,611,711,264 B) |
| llama.cpp | commit `fd6863a69542c74a617a1219f1b18ccf773f41ed`, `b10430-26-gfd6863a69` |
| Built libraries | `/home/nixabe/llama.cpp/build/bin/libllama.so`, `libggml*.so` |
| GPU | Quadro RTX 8000, sm_75, driver 595.84, `CUDA_VISIBLE_DEVICES=2` |
| Compiler | g++ 11.4.0 |
| Context | `n_ctx = n_batch = n_ubatch = 4096`, `n_seq_max = 1` |
| Attention | Flash Attention **enabled**, `type_k = type_v = f16` |
| Offload | `n_gpu_layers = 99`, `split_mode = NONE` |

The same facts are written to `.golden/meta.txt` at capture time, alongside a
SHA-256 of the container.

This is the same llama.cpp build [BENCHMARKS.md](BENCHMARKS.md) was measured
against, so the oracle and the performance baseline describe the same binary.

### The prompt

```
The capital of France is Paris. The capital of Germany is Berlin. The capital of Japan is
```

Chosen because it is short, has no chat template, tokenizes deterministically,
and has an unambiguous continuation — so a capture that silently went wrong is
visible without reading any floats.

`add_bos` is **false** for this model: no BOS token is prepended. 19 tokens:

```
  [ 0]    760  'The'        [ 7]    561  ' The'       [14]    561  ' The'
  [ 1]   6511  ' capital'   [ 8]   6511  ' capital'   [15]   6511  ' capital'
  [ 2]    314  ' of'        [ 9]    314  ' of'        [16]    314  ' of'
  [ 3]   9338  ' France'    [10]   9564  ' Germany'   [17]   6124  ' Japan'
  [ 4]    369  ' is'        [11]    369  ' is'        [18]    369  ' is'
  [ 5]  11751  ' Paris'     [12]  19241  ' Berlin'
  [ 6]     13  '.'          [13]     13  '.'
```

llama.cpp's argmax over the resulting logits is token **25358 `' Tokyo'`**,
logit **19.902241**. `golden.rs` asserts that value, so a stale or corrupt
capture fails rather than silently comparing against nonsense.

---

## 2. Why not `llama-eval-callback`

`llama-eval-callback` and `llama-debug --verbose --tensor-filter` register the
same callback this capture uses, and they were the obvious starting point.
They are not sufficient:

- **They truncate.** `common_debug_print_tensor` (`common/debug.cpp`) prints
  the first and last three indices of every dimension and a scalar `sum`. For a
  `[2048, 19]` hidden state that is 6 floats per token out of 2048. You cannot
  gate a differential test on that.
- `llama-debug --save-logits` *does* write full final logits
  (`examples/debug/debug.cpp`), and that half would have been enough — but it
  writes nothing intermediate, and a forward pass that is wrong at layer 12
  needs to be localized, not just declared wrong.

So the capture uses the same public callback hook
(`llama_context_params::cb_eval`) with a binary sink instead of a printer. No
llama.cpp source is modified; the tool builds out of tree against the installed
headers and shared libraries.

`llama-perplexity` was also considered and rejected: it reports an aggregate
over a corpus, which cannot localize anything.

---

## 3. Regenerating the capture

The capture program is **`tools/oracle/capture.cpp`**, tracked in this
repository and built by `tools/oracle/Makefile`. The filter list it was run
with is `tools/oracle/filters.txt`. It is also reproduced in full in
[appendix A](#appendix-a-capturecpp) so this document stands alone.

The golden data it produces lives in `.golden/`, which is **gitignored** — 56
MB of f32 tensors does not belong in git history. The generator is tracked and
the output is not, so anyone can reproduce the data but nobody has to clone it.

### Build

```sh
cd tools/oracle && make
```

which is exactly:

```sh
LLAMA=/home/nixabe/llama.cpp
g++ -O2 -std=c++17 capture.cpp -o capture \
    -I$LLAMA/include -I$LLAMA/ggml/include \
    -L$LLAMA/build/bin -lllama -lggml-base -lggml \
    -Wl,-rpath,$LLAMA/build/bin
```

No llama.cpp source is modified. The program registers the same public
`llama_context_params::cb_eval` hook that `llama-eval-callback` uses, with a
binary sink instead of a printer, and builds out of tree against the installed
headers and shared objects.

### Run

The filter argument is a comma-separated list of regexes, each **fully**
matched against the ggml node name. The exact list used is in
`.golden/filters.txt`; it is reproduced here verbatim.

```sh
FILTERS='model\.input_embed,result_norm,h_nextn,result_output,attn_norm-[0-9]+,attn_residual-[0-9]+,attn_post_norm-[0-9]+,ffn_out-[0-9]+,l_out-[0-9]+,(linear_attn_qkv_mixed|z|beta|beta_sigmoid|alpha|a_softplus|gate|conv_output_raw|conv_output_silu|q_conv|k_conv|v_conv|q_conv_predelta|k_conv_predelta|v_conv_predelta|state_predelta|final_output|linear_attn_out|ffn_moe_out)-(0|4|20),(Qcur_full|Qcur_reshaped|Qcur_normed|Qcur|Kcur|Kcur_normed|Vcur|gate_reshaped|attn_pregate|gate_sigmoid|attn_gated|attn_output|ffn_moe_out)-(3|39)'

mkdir -p .golden
CUDA_VISIBLE_DEVICES=2 tools/oracle/capture \
  /home/nixabe/llama.cpp/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf \
  .golden/qwen36-golden.bin \
  "The capital of France is Paris. The capital of Germany is Berlin. The capital of Japan is" \
  "$FILTERS" | tee .golden/capture.log
```

Takes about 40 s including model load. Expected tail:

```
captured result_norm                  type=f32   ne=[2048,1,1,1] n=2048 sum=-47.401514
captured result_output                type=f32   ne=[248320,1,1,1] n=248320 sum=-572014.570270
argmax token = 25358 ' Tokyo' logit = 19.902241
records = 290 (callback matched 288)
```

### Verify

```sh
CUDA_VISIBLE_DEVICES=2 cargo test -p xabe-engine --test golden -- --nocapture
```

Six tests. They check the container parses, the prompt is the expected one,
every block boundary is present and finite, the mixer internals match the layer
kind `ModelConfig` derives, the logits agree with the public API, and the two
layout claims in [§6](#6-tensor-name-and-layout-mapping) hold against the real
file.

---

## 4. What was captured

288 graph tensors + 2 synthesized records. All payloads are f32.

### Global

| Name | `ne` | What it is |
| --- | --- | --- |
| `model.input_embed` | `[2048, 19]` | `get_rows(token_embd.weight, tokens)` — the stack's input |
| `h_nextn` | `[2048, 19]` | final `output_norm`, **all** positions |
| `result_norm` | `[2048]` | final `output_norm`, last position only |
| `result_output` | `[248320]` | logits, last position |
| `api.logits` | `[248320]` | same, via `llama_get_logits_ith` — see below |
| `api.tokens` | `[19]` (i32) | the prompt's token ids |

`result_norm` and `result_output` cover **one** position because llama.cpp
applies `ggml_get_rows(cur, inp_out_ids)` before the LM head and a plain
prefill only requests logits for the last token. `h_nextn` is the same
normalized hidden state *before* that selection, so it is the tensor to compare
a full-sequence final norm against.

`api.logits` is read from the public API after `llama_decode` returned;
`result_output` comes off the graph callback mid-compute. `golden.rs` asserts
they are **bit-identical**, which is what licenses treating the intermediate
captures as belonging to the same execution as the logits.

### Every block, 0..=39

| Name | `ne` | What it is |
| --- | --- | --- |
| `attn_norm-N` | `[2048, 19]` | input RMSNorm, before the mixer |
| `attn_residual-N` | `[2048, 19]` | mixer output + residual |
| `attn_post_norm-N` | `[2048, 19]` | post-mixer RMSNorm, before the MoE |
| `ffn_out-N` | `[2048, 19]` | routed MoE + shared expert, before residual |
| `l_out-N` | `[2048, 19]` | the block's output on the residual stream |

That gives 200 waypoints across the stack. It is what makes a wrong forward
pass **bisectable**: a mismatch at `l_out-17` with `l_out-16` clean localizes
the fault to one block without re-running anything.

Hidden state *entering* block `N` is `l_out-(N-1)`, or `model.input_embed` for
block 0. `Golden::block_input` / `block_output` name that so the off-by-one
lives in one place.

### Gated DeltaNet internals — blocks 0, 4, 20

| Name | `ne` | Produced by |
| --- | --- | --- |
| `linear_attn_qkv_mixed-N` | `[8192, 19]` | `attn_qkv.weight` projection |
| `conv_output_raw-N` | `[8192, 19]` | `ggml_ssm_conv` over the fused q/k/v stream |
| `conv_output_silu-N` | `[8192, 19]` | `ggml_silu` of the above |
| `q_conv-N`, `k_conv-N` | `[128, 16, 19]` | the sliced q/k views |
| `q_conv_predelta-N`, `k_conv_predelta-N` | `[128, 16, 19]` | after `ggml_l2_norm` |
| `v_conv_predelta-N` | `[128, 32, 19]` | the sliced v view |
| `alpha-N` | `[32, 19]` | `ssm_alpha.weight` projection |
| `a_softplus-N` | `[32, 19]` | `softplus(alpha + ssm_dt.bias)` |
| `gate-N` | `[32, 19]` | `ssm_a * a_softplus` — the per-head **log-decay** |
| `beta-N`, `beta_sigmoid-N` | `[1, 32, 19]` | write strength, pre/post sigmoid |
| `state_predelta-N` | `[128, 128, 32]` | recurrent state entering the block |
| `z-N` | `[4096, 19]` | `attn_gate.weight` projection (the output gate) |
| `final_output-N` | `[4096, 19]` | `ssm_norm(delta_out) * silu(z)` |
| `linear_attn_out-N` | `[2048, 19]` | `ssm_out.weight` projection |
| `ffn_moe_out-N` | `[2048, 19]` | routed experts only, before the shared expert |

### Gated Attention internals — blocks 3, 39

| Name | `ne` | Produced by |
| --- | --- | --- |
| `Qcur_full-N` | `[8192, 19]` | `attn_q.weight` projection — query **and** gate |
| `Qcur_reshaped-N` | `[256, 16, 19]` | the query half, strided view |
| `Qcur_normed-N` | `[256, 16, 19]` | after `attn_q_norm` |
| `Qcur-N` | `[256, 16, 19]` | after IMRoPE |
| `Kcur-N` (first) | `[512, 19]` | `attn_k.weight` projection |
| `Kcur_normed-N` | `[256, 2, 19]` | after `attn_k_norm` |
| `Kcur-N` (second) | `[256, 2, 19]` | after IMRoPE |
| `Vcur-N` (two records) | `[512, 19]`, `[256, 2, 19]` | projection, then reshape |
| `gate_reshaped-N` | `[4096, 19]` | the gate half, made contiguous |
| `attn_pregate-N` | `[4096, 19]` | attention output before gating |
| `gate_sigmoid-N` | `[4096, 19]` | `sigmoid(gate)` |
| `attn_gated-N` | `[4096, 19]` | their product |
| `attn_output-N` | `[2048, 19]` | `attn_output.weight` projection |
| `ffn_moe_out-N` | `[2048, 19]` | routed experts only |

**Names are not unique.** `cb(t, "Kcur", il)` is called twice — once on the raw
projection, once after RoPE — so `Kcur-3` names two tensors of two different
shapes. Same for `Vcur-3`. `Golden::get` returns the last; `Golden::all`
returns both, and `golden.rs` asserts both shapes so a consumer that picks the
wrong one has a documented reason available.

---

## 5. File format

Little-endian, flat, length-prefixed. Deliberately not JSON or npz:
`xabe-engine` has no serialization dependency and this reader needs none.

```
  magic     "XABEGOLD"                 8 B
  u32       version = 1
  u32       n_records
  record * n_records:
      u32   name_len
      u8    name[name_len]             UTF-8, no NUL
      u32   dtype                      0 = f32, 1 = i32
      i64   ne[4]                      ggml order, ne[0] fastest-varying
      u64   n_elements                 = ne0*ne1*ne2*ne3
      u8    payload[n_elements * 4]
```

Payloads are written in **ggml logical order**,
`i0 + ne0*(i1 + ne1*(i2 + ne2*i3))`. Strided views are de-strided through
`nb[]` at capture time, so a reader never has to know ggml's stride rules — it
only has to know which dimension is fastest.

`Golden::load` rejects a bad magic, an unknown version, a `ne`/`n_elements`
disagreement, truncation, and trailing bytes. `setup()` **panics** on a corrupt
file rather than skipping: a test that quietly passed on a truncated capture
would be worse than no oracle.

---

## 6. Tensor name and layout mapping

This is the section that exists because getting it wrong produces a transposed
comparison that reads like a kernel bug.

### 6.1 The matrix convention

ggml stores `ne[0]` as the **fastest-varying** dimension. A GGUF matrix with
dims `[A, B]` is therefore, in ordinary row-major terms, a `B × A` matrix: `B`
rows of `A` contiguous elements. `ggml_mul_mat(W, X)` with `W = [A, B]` and
`X = [A, n]` produces `[B, n]` — so **`ne[0]` is the input width and `ne[1]` is
the output width**.

Applied to this model: `output.weight` is `[2048, 248320]`, i.e. 248,320 rows
of 2,048 contiguous elements, not 2,048 rows of 248,320.
`xabe_model::weights::TensorSpec::dims` already documents itself as being in
this order (`dims[0]` fastest-varying), and `xabe-gguf` reads it that way. The
two proofs below are the evidence, not the assertion.

### 6.2 `Role` ↔ GGUF name ↔ llama.cpp graph node

`Role` is `xabe_model::weights::Role`. The GGUF name is `Role::suffix()` under
a `blk.N.` prefix. The graph node is what the golden holds.

| `Role` | GGUF name | `ne` | Graph node it produces |
| --- | --- | --- | --- |
| `TokenEmbedding` | `token_embd.weight` | `[2048, 248320]` | `model.input_embed` |
| `OutputNorm` | `output_norm.weight` | `[2048]` | `h_nextn`, `result_norm` |
| `LmHead` | `output.weight` | `[2048, 248320]` | `result_output` |
| `InputNorm` | `blk.N.attn_norm.weight` | `[2048]` | `attn_norm-N` |
| `PostMixerNorm` | `blk.N.post_attention_norm.weight` | `[2048]` | `attn_post_norm-N` |
| `AttnQGate` | `blk.N.attn_q.weight` | `[2048, 8192]` | `Qcur_full-N` |
| `AttnQNorm` | `blk.N.attn_q_norm.weight` | `[256]` | `Qcur_normed-N` |
| `AttnK` | `blk.N.attn_k.weight` | `[2048, 512]` | `Kcur-N` (first) |
| `AttnKNorm` | `blk.N.attn_k_norm.weight` | `[256]` | `Kcur_normed-N` |
| `AttnV` | `blk.N.attn_v.weight` | `[2048, 512]` | `Vcur-N` (first) |
| `AttnOut` | `blk.N.attn_output.weight` | `[4096, 2048]` | `attn_output-N` |
| `GdnQkv` | `blk.N.attn_qkv.weight` | `[2048, 8192]` | `linear_attn_qkv_mixed-N` |
| `GdnGate` | `blk.N.attn_gate.weight` | `[2048, 4096]` | `z-N` |
| `GdnConv1d` | `blk.N.ssm_conv1d.weight` | `[4, 8192]` | `conv_output_raw-N` |
| `GdnAlpha` | `blk.N.ssm_alpha.weight` | `[2048, 32]` | `alpha-N` |
| `GdnDtBias` | `blk.N.ssm_dt.bias` | `[32]` | folded into `a_softplus-N` |
| `GdnA` | `blk.N.ssm_a` | `[32]` | folded into `gate-N` |
| `GdnBeta` | `blk.N.ssm_beta.weight` | `[2048, 32]` | `beta-N` |
| `GdnNorm` | `blk.N.ssm_norm.weight` | `[128]` | folded into `final_output-N` |
| `GdnOut` | `blk.N.ssm_out.weight` | `[4096, 2048]` | `linear_attn_out-N` |
| `MoeRouter`, `Moe*Exps` | `blk.N.ffn_*.weight` | — | `ffn_moe_out-N` |
| `MoeShared*` | `blk.N.ffn_*_shexp.weight` | — | `ffn_out-N` (with the routed part) |

Note the direction of `AttnOut` and `GdnOut`: `[4096, 2048]` is input 4096,
output 2048 — the opposite reading of the pair from every other row in the
table, and it is correct.

### 6.3 Proof 1 — the embedding table, bit-exact

`input_embedding_matches_this_repos_own_gguf_reader` in `golden.rs`:

1. `xabe-gguf` opens the model and confirms `token_embd.weight` is `Q8_0` with
   dims `[2048, 248320]`.
2. For each of the 19 prompt tokens `t`, it takes bytes
   `[t * 2176, (t+1) * 2176)` of the tensor — 64 `Q8_0` blocks of 34 bytes,
   which is one row **only if `ne[0]` is the fastest-varying dimension** — and
   dequantizes them with `xabe_kernels::quant::dequantize_row_q8_0`.
3. It compares against column `t` of the captured `model.input_embed`.

Measured:

```
token_embd.weight dequantized by xabe-gguf + xabe-kernels vs llama.cpp's
model.input_embed: 38912/38912 elements bit-identical, max_abs 0.000e0
  correct column 0 head: [-0.0011072159, 0.009411335, -0.008304119, 0.0027680397, 0.0022144318]
  transposed misreading: [-0.0011072159, 0.0073566437, 0.0006380081, -0.009645939, -0.0009417534]
```

All 38,912 elements bit-identical, not merely close — `Q8_0` dequantization is
`d * q`, multiply-only, so exactness is the right gate and reassociation cannot
hide a layout error behind a tolerance. The second line is the same slice read
with the dimensions swapped, and it disagrees on element 1 onward, so the match
is discriminating rather than a coincidence.

### 6.4 Proof 2 — the packed query/gate projection

`the_packed_query_gate_projection_is_interleaved_per_head_not_split_in_half`
checks the other documented trap, using only captured data:

```
Qcur_reshaped[d, h, t]      == Qcur_full[h*2*head_dim + d,            t]
gate_reshaped[h*hd + d, t]  == Qcur_full[h*2*head_dim + head_dim + d, t]
```

Measured:

```
attn_q interleave over 77824 elements: stride-2*head_dim mismatches q=0 gate=0;
split-in-half mismatches 72960
```

Zero mismatches under the interleaved reading. Under the "split `attn_q` down
the middle" reading, 72,960 of 77,824 disagree — the 4,864 that agree are
exactly head 0 (`256 × 19`), where both readings coincide. So `attn_q` really
is `[q_h0, gate_h0, q_h1, gate_h1, …]` at stride `head_dim * 2`, as
`xabe_model::weights` says.

---

## 7. Fidelity of the capture

Three properties were measured, not assumed. All three matter because the whole
value of an oracle is that it is the same computation the engine must match.

### The callback does not perturb the result

`ggml_backend_sched_compute_splits` breaks the graph at every node the callback
asks for, which can prevent op fusion inside a backend and change reduction
order. So the capture was run twice: once with the full 288-node filter, once
with a filter matching only `result_output` (a single break at the final node,
which is equivalent to no callback at all).

```
result_output   BIT-IDENTICAL
api.logits      BIT-IDENTICAL
api.tokens      BIT-IDENTICAL
```

**No perturbation on this model and this build.** If a future filter set does
perturb the logits, this comparison is how you would find out; re-run it after
changing the filter.

### The run is deterministic

The full capture was run twice and the two 57,602,888-byte containers compared
with `cmp`:

```
RERUN BIT-IDENTICAL TO GOLDEN
```

Byte-for-byte, including every intermediate. The oracle is a fixed target, not
a moving one.

### The graph node is the API's logits

Asserted in `golden.rs`: `result_output` and `llama_get_logits_ith(ctx, 18)`
are bit-identical over all 248,320 entries.

---

## 8. Things found that the docs do not say

Stated because they were discovered while doing this, and each one could
silently produce a wrong implementation.

0. **llama.cpp's matmuls quantize the *activations* to `q8_1`. It is measurably
   less accurate than an exact fp32 path, and this accounts for essentially
   every residual gap against this oracle.**

   This is the single most important entry here, because without it every
   block workstream independently concludes its own matmul is slightly wrong.
   Three did, and all three arrived at the same cause.

   With 19 tokens, `ggml_cuda_mul_mat` routes a Q8_0 `src0` past MMVF/MMVQ to
   **MMQ**, which quantizes the activation into `q8_1` blocks of 32
   (`d = amax/127`, `roundf`) and does the dot in int8. Reproducing that
   quantization on the host recovers llama.cpp's own output almost exactly,
   which is what makes this a demonstration rather than a hypothesis:

   | projection | exact fp32 vs golden | same matmul, `q8_1` activations | ratio |
   | --- | --- | --- | --- |
   | `Qcur_full-3` | 7.00e-2 | 2.05e-5 | 3,421× |
   | `Kcur-3` | 5.49e-2 | 9.06e-6 | 6,069× |
   | `Vcur-3` | 3.71e-2 | 3.58e-6 | 10,384× |
   | `attn_output-3` | 7.79e-4 | 3.28e-7 | 2,379× |
   | `Qcur_full-39` | 7.38e-2 | 2.19e-5 | 3,361× |
   | `Vcur-39` | 3.49e-2 | 1.62e-5 | 2,154× |

   The scale is **fp32**, not fp16: an fp16 scale (what the non-MMQ
   `quantize_q8_1` stores) does *not* reproduce it (8.0e-2 / 1.6e-2 / 3.2e-2).

   Corroborated independently against an f64 host reference from the same
   weights and the same captured input:

   ```
   attn_qkv:  llmxabe vs f64  1.907e-6   |  llama.cpp vs f64  7.059e-2   (37,000x)
   ssm_out:   llmxabe vs f64  2.235e-8   |  llama.cpp vs f64  1.629e-3   (73,000x)
   control (ssm_alpha, f32 weights, llama.cpp's unquantized path):  2.384e-6
   ```

   And in the MoE, where device and the structurally unrelated CPU reference
   agree to 4.47e-8 while *both* sit the same 2.6997e-4 away from llama.cpp —
   agreeing to six significant figures on the distance.

   **Consequences.** A tolerance against this oracle at a quantized matmul is
   bounded below by llama.cpp's error, not ours, and is therefore loose for a
   reason that has nothing to do with our kernel. Any test relying on that
   should assert the explanation — that a `q8_1` reconstruction is orders
   closer than the exact one — so the loose bound stops being justified the
   moment the explanation stops holding. Tightening the *measured* agreement
   would require adopting `q8_1` activation quantization ourselves, which is an
   engine-wide accuracy decision and a deliberate loss of precision.

   Elementwise ops are unaffected; they are not matmuls and should agree at the
   fp32 rounding floor.

1. **`q_conv_predelta` / `k_conv_predelta` are *not* broadcast to 32 heads.**
   They stay at `[128, 16, 19]` — 16 qk heads — while `v_conv_predelta` is
   `[128, 32, 19]`. `build_layer_attn_linear` only emits the
   `ggml_repeat_4d` to 32 heads when the fused GDN path is off
   (`cparams.fused_gdn_ar`/`fused_gdn_ch`), and this build has it on. Anything
   comparing against these tensors must do the broadcast itself.

   **The broadcast is `qk_head = v_head % n_qk_heads`, not
   `v_head / heads_per_kv`.** `ggml_repeat_4d` tiles; it does not block.
   `ggml/src/ggml-cuda/gated_delta_net.cu:37` computes
   `fastmodulo(h_idx, neqk1_magic)` directly. An earlier revision of this
   document said only "do the broadcast yourself" without stating the
   direction, and two landed kernels got it backwards as a result —
   discriminated numerically against `final_output-N`:

   | block | `h % 16` (correct) | `h / 2` (wrong) |
   | --- | --- | --- |
   | 0 | max_abs 7.15e-7, cosine 1.000000 | max_abs 5.59e-1, cosine 0.974880 |
   | 4 | max_abs 1.19e-7, cosine 1.000000 | max_abs 5.66e-1, cosine 0.467691 |
   | 20 | max_abs 1.04e-7, cosine 1.000000 | max_abs 5.24e-1, cosine 0.622903 |

   A differential test against a single-head CPU reference **cannot** catch
   this: the caller does the broadcast, so a test that uses the same wrong
   convention as the kernel sees perfect agreement. Only the oracle catches it.

2. **`ssm_a` is stored already negated.** `blk.0.ssm_a` begins
   `[-0.03642, -0.03116, -0.13747, …]` and every one of its 32 entries is
   negative. So `gate-N = ssm_a * softplus(alpha + dt_bias)` *is* the log-decay
   directly; llama.cpp's `// -A_log.exp() * softplus` comment describes what
   the stored value already contains, not an operation the graph performs.

3. **`token_embd.weight` is `Q8_0`, not a float type.** The embedding lookup
   is a dequantization. [MODEL.md](MODEL.md) says the LM head and projections
   are Q8_0 but does not call out the input embedding, and a loader that
   assumed f16/bf16 there would read garbage.

4. **The file's two `bf16` tensors are `blk.40.ffn_gate_inp.weight` and
   `blk.40.ffn_gate_inp_shexp.weight`** — the MTP block's routers, and nothing
   else. [MODEL.md](MODEL.md) reports the count but not which tensors. Every
   router in blocks 0–39 is `f32`; only the MTP block's two are `bf16`. A
   router dequant path that handles f32 only works for the whole text stack and
   fails exactly at block 40.

5. **`add_bos` is false for this model.** No BOS is prepended. A comparison
   that prepends one would be off by a token everywhere.

6. **`result_norm` is one position, not 19.** `inp_out_ids` selection happens
   between the final norm and the LM head. Use `h_nextn` for a full-sequence
   comparison of the final norm.

Nothing here contradicts [MODEL.md](MODEL.md), [KERNELS.md](KERNELS.md) or the
team plan. Every fact those documents state that this capture touches — 40
layers with attention at offset 3 of period 4, `block_count` 41, the
interleaved `attn_q` packing, the SiLU after the GDN convolution, mixed expert
quantization — held. The SiLU in particular is now checked numerically rather
than by reading source: `golden.rs` asserts
`conv_output_silu-0 == silu(conv_output_raw-0)` to 9.537e-7, which is fp32
round-off on a 155,648-element tensor.

---

## 9. Limits

- **One prompt, 19 tokens, one ubatch.** Nothing here exercises the GDN chunked
  prefill path across a chunk boundary (`chunk_len = 64`), multi-ubatch
  scheduling, or decode with a carried recurrent state — `state_predelta-0` is
  all zeros, so the recurrence is only checked from a cold start. A second
  capture over a >64-token prompt would close the first of those; the procedure
  is unchanged apart from the prompt string.
- **Flash attention is on**, so the attention numerics in the golden are the
  flash path's. A non-flash reference would differ within fp32 round-off.
  Re-capture with `LLAMA_FLASH_ATTN_TYPE_DISABLED` if that distinction matters.
- **The MoE internals are not captured** beyond `ffn_moe_out` and `ffn_out`.
  Router logits, top-k ids and per-expert outputs are named inside
  `build_moe_ffn`, not in `qwen35moe.cpp`, and were not filtered for. Adding
  them is a filter change, not a code change.
- **The MTP block (40) never executes** in the main graph, so nothing about it
  is captured.

---

## Appendix A: `capture.cpp`

Tracked at `tools/oracle/capture.cpp`; build it with `tools/oracle/Makefile`.

```cpp
// Golden-oracle capture tool for llmxabe.
//
// Runs one llama.cpp forward pass over a fixed prompt and writes the full,
// untruncated contents of selected intermediate graph tensors plus the final
// logits to a single self-describing binary container.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <regex>
#include <string>
#include <vector>

static const char MAGIC[8] = { 'X','A','B','E','G','O','L','D' };

struct writer {
    FILE * f = nullptr;
    uint32_t n_records = 0;

    void open(const char * path) {
        f = fopen(path, "wb");
        if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
        fwrite(MAGIC, 1, 8, f);
        uint32_t v = 1;
        fwrite(&v, 4, 1, f);
        fwrite(&n_records, 4, 1, f); // patched on close
    }

    void record(const std::string & name, uint32_t dtype,
                const int64_t ne[4], const void * data, uint64_t n_elem) {
        uint32_t nl = (uint32_t) name.size();
        fwrite(&nl, 4, 1, f);
        fwrite(name.data(), 1, nl, f);
        fwrite(&dtype, 4, 1, f);
        fwrite(ne, 8, 4, f);
        fwrite(&n_elem, 8, 1, f);
        fwrite(data, 4, n_elem, f);
        n_records++;
    }

    void close() {
        fseek(f, 12, SEEK_SET);
        fwrite(&n_records, 4, 1, f);
        fclose(f);
        f = nullptr;
    }
};

struct cb_state {
    writer * w = nullptr;
    std::vector<std::regex> filters;
    std::vector<uint8_t> raw;
    std::vector<float>   flat;
    int matched = 0;
};

static float read_scalar(const uint8_t * d, ggml_type type, size_t off) {
    switch (type) {
        case GGML_TYPE_F32:  return *(const float *) (d + off);
        case GGML_TYPE_F16:  return ggml_fp16_to_fp32(*(const ggml_fp16_t *) (d + off));
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*(const ggml_bf16_t *) (d + off));
        case GGML_TYPE_I32:  return (float) *(const int32_t *) (d + off);
        case GGML_TYPE_I16:  return (float) *(const int16_t *) (d + off);
        case GGML_TYPE_I8:   return (float) *(const int8_t  *) (d + off);
        default: fprintf(stderr, "unhandled type %s\n", ggml_type_name(type)); exit(1);
    }
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (cb_state *) user_data;

    bool match = false;
    for (const auto & re : st->filters) {
        if (std::regex_match(t->name, re)) { match = true; break; }
    }

    if (ask) {
        // Returning false means the scheduler will not call back with the data.
        return match;
    }
    if (!match) {
        return true;
    }
    if (ggml_is_quantized(t->type)) {
        fprintf(stderr, "skipping quantized tensor %s (%s)\n", t->name, ggml_type_name(t->type));
        return true;
    }

    const size_t nbytes = ggml_nbytes(t);
    const uint8_t * src;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        src = (const uint8_t *) t->data;
    } else {
        st->raw.resize(nbytes);
        ggml_backend_tensor_get(t, st->raw.data(), 0, nbytes);
        src = st->raw.data();
    }

    const int64_t * ne = t->ne;
    const size_t  * nb = t->nb;
    const uint64_t n_elem = (uint64_t) ne[0] * ne[1] * ne[2] * ne[3];
    st->flat.resize(n_elem);

    uint64_t o = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++)
    for (int64_t i2 = 0; i2 < ne[2]; i2++)
    for (int64_t i1 = 0; i1 < ne[1]; i1++)
    for (int64_t i0 = 0; i0 < ne[0]; i0++) {
        st->flat[o++] = read_scalar(src, t->type, i3*nb[3] + i2*nb[2] + i1*nb[1] + i0*nb[0]);
    }

    double sum = 0.0;
    for (uint64_t i = 0; i < n_elem; i++) sum += st->flat[i];

    st->w->record(t->name, 0, ne, st->flat.data(), n_elem);
    st->matched++;
    printf("captured %-28s type=%-5s ne=[%lld,%lld,%lld,%lld] n=%llu sum=%.6f\n",
           t->name, ggml_type_name(t->type),
           (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3],
           (unsigned long long) n_elem, sum);
    fflush(stdout);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <model.gguf> <out.bin> <prompt> <regex>[,<regex>...]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * out_path   = argv[2];
    const char * prompt     = argv[3];

    cb_state st;
    writer w;
    w.open(out_path);
    st.w = &w;
    {
        std::string spec = argv[4];
        size_t p = 0;
        while (p <= spec.size()) {
            size_t q = spec.find(',', p);
            if (q == std::string::npos) q = spec.size();
            std::string pat = spec.substr(p, q - p);
            if (!pat.empty()) st.filters.emplace_back(pat, std::regex::optimize);
            p = q + 1;
        }
    }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    mp.split_mode   = LLAMA_SPLIT_MODE_NONE;
    mp.main_gpu     = 0;   // relative to CUDA_VISIBLE_DEVICES

    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx             = 4096;
    cp.n_batch           = 4096;
    cp.n_ubatch          = 4096;
    cp.n_seq_max         = 1;
    cp.n_threads         = 4;
    cp.n_threads_batch   = 4;
    cp.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k            = GGML_TYPE_F16;
    cp.type_v            = GGML_TYPE_F16;
    cp.cb_eval           = cb_eval;
    cp.cb_eval_user_data = &st;
    cp.no_perf           = false;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    std::vector<llama_token> tokens(512);
    int32_t n = llama_tokenize(vocab, prompt, (int32_t) strlen(prompt),
                               tokens.data(), (int32_t) tokens.size(),
                               /*add_special =*/ true, /*parse_special =*/ true);
    if (n < 0) { fprintf(stderr, "tokenize failed (%d)\n", n); return 1; }
    tokens.resize(n);

    printf("add_bos=%d n_tokens=%d\n", (int) llama_vocab_get_add_bos(vocab), n);
    for (int i = 0; i < n; i++) {
        char piece[256];
        int m = llama_token_to_piece(vocab, tokens[i], piece, sizeof(piece), 0, true);
        printf("  [%2d] %6d  '%.*s'\n", i, tokens[i], m > 0 ? m : 0, piece);
    }
    fflush(stdout);

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), n)) != 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits  = llama_get_logits_ith(ctx, n - 1);
    if (!logits) { fprintf(stderr, "no logits\n"); return 1; }
    {
        const int64_t ne[4] = { n_vocab, 1, 1, 1 };
        w.record("api.logits", 0, ne, logits, (uint64_t) n_vocab);
    }
    {
        std::vector<int32_t> tk(tokens.begin(), tokens.end());
        const int64_t ne[4] = { n, 1, 1, 1 };
        w.record("api.tokens", 1, ne, tk.data(), (uint64_t) n);
    }

    int best = 0;
    for (int i = 1; i < n_vocab; i++) if (logits[i] > logits[best]) best = i;
    char piece[256];
    int m = llama_token_to_piece(vocab, best, piece, sizeof(piece), 0, true);
    printf("argmax token = %d '%.*s' logit = %.6f\n", best, m > 0 ? m : 0, piece, logits[best]);
    printf("records = %u (callback matched %d)\n", w.n_records, st.matched);

    w.close();

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
```
