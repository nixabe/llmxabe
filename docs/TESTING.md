# Testing and numerics

## The risk this exists for

Numerics drift is the highest-likelihood risk in this project. A subtly wrong
kernel does not crash and does not produce garbage — the model stays fluent
and gets quietly worse. No throughput benchmark catches it, and reading
generated text and judging it plausible catches it least of all.

So: **a kernel without a passing differential test is not done, regardless of
how fast it runs.**

## The structure

Every kernel has a scalar fp32 CPU reference in
[`xabe-kernels`](../crates/xabe-kernels), written for obvious correctness
rather than speed. These are the oracle. A clever reference that is subtly
wrong is worse than no reference at all, so they are deliberately naive.

`xabe-kernels` depends on no CUDA and no device. That is a design property,
not an accident: the correctness work must be runnable and debuggable on a
laptop, and must never be blocked on GPU access.

| Module | Reference |
| --- | --- |
| `gdn::recurrent` | Gated DeltaNet, recurrent (decode) form |
| `gdn::chunked` | Gated DeltaNet, chunked parallel (prefill) form |
| `gdn::tri` | Lower-triangular inverse by forward substitution |
| `moe::router` | Softmax + top-k over 256 experts, deterministic ties |
| `moe::dispatch` | `moe_align_block_size` equivalent |
| `moe::gemm` | Grouped MoE GEMM with SwiGLU and top-k reduction |
| `attention` | GQA 16:2, head dim 256, plus online-softmax form |
| `rope` | Partial rotary — 64 of 256 dims |
| `norm` | RMSNorm, SwiGLU, residual |
| `quant` | Q6_K and Q8_0 dequantization |
| `compare` | The differential harness |
| `rng` | Seeded xorshift, for reproducible inputs |

## Metrics: report all of them

`compare()` returns a `ComparisonResult` carrying:

| Metric | Catches | Misses |
| --- | --- | --- |
| `max_abs_error` (+ index) | Magnitude errors | Direction errors on small values |
| `max_rel_error` (+ index) | Proportional errors on small values | Errors on near-zero references |
| `cosine_similarity` | Direction errors | **Uniform scaling** |
| `non_finite_count` | NaN/Inf | Everything else |

**Cosine alone is not sufficient.** A kernel returning exactly half the correct
output has cosine similarity 1.0. `compare.rs` carries a test named
`cosine_alone_hides_a_magnitude_error_but_max_abs_catches_it` that pins this,
and every equivalence assertion in the crate gates on the full tolerance via
`assert_matches`, not on cosine.

Results report **which element** was worst and at what index.
`worst_abs_pair()` returns the offending `(candidate, reference)` values.
"Cosine 0.994" is not actionable; "worst element at index 3117, expected 0.42,
got 0.31" is.

## Tolerances

| Preset | Bounds | For |
| --- | --- | --- |
| `exact()` | 0, 0, cosine 1.0 | Bit-identical paths |
| `tight_fp32()` | 1e-3 abs/rel, cosine 1−1e-5 | Two fp32 implementations of the same formula differing only in operation order |
| `gdn_chunk_vs_recurrent()` | 5e-2 abs/rel, cosine 1−1e-3 | The milestone-01 gate |
| `reduced_precision_gpu()` | looser | fp16/tf32 device kernels vs fp32 reference |

fp32 summation is not associative, so exact equality is the wrong bar for a
reformulation. These are tight enough to catch a real formulation bug while
absorbing reassociation noise. Each carries its rationale in a doc comment.

Inputs come from a seeded xorshift generator (`rng::Xorshift64Star`) so a
failure reproduces exactly. No `rand` dependency.

## The GDN equivalence test

This is the most valuable check in the crate, and the reason milestone 01
precedes the harness in the plan.

The chunked parallel form (prefill) and the recurrent form (decode) compute the
same function by very different routes — the chunked form requires inverting a
lower-triangular matrix per chunk. **They must agree on identical input.**

That equivalence is what will later reveal whether a GPU chunked kernel is
wrong, because it is checkable without any reference activations from
`transformers`.

Current status: passes at the **real Qwen3.6 shape** — head dim 128, chunk
length 64, taken from `ModelConfig` rather than hardcoded — over a 197-token
sequence spanning three full chunks and a partial one, gated on the full
tolerance, comparing both the per-token outputs and the final recurrent state.
Also covered: `chunk_len = 1`, `chunk_len = seq_len`, nonzero initial state,
and chunk-boundary independence.

## Provenance: ported versus derived

Formulations are cited in doc comments, distinguishing what was ported from
upstream from what was inferred. That distinction is the honest measure of how
much a reference should be trusted.

**Ported and cross-checked.** The GDN *recurrent* form was derived from two
independent upstream implementations that were compared line by line and found
to agree exactly:

- llama.cpp, `ggml_compute_forward_gated_delta_net_one_chunk`
  (`ggml/src/ggml-cpu/ops.cpp`)
- vLLM, `fused_recurrent_gated_delta_rule_fwd_kernel`, which is what
  `Qwen3NextGatedDeltaNetAttention` — the module this model family actually
  instantiates — calls on CPU

Both confirm L2-normalized q/k with eps 1e-6, a **scalar per-head** decay
(confirmed from `A_log`/`dt_bias` having shape `[num_v_heads]`, not the
per-channel variant llama.cpp also supports but Qwen does not use),
decay-then-correction-then-output ordering, and an output scale of
`1/sqrt(head_dim)`.

Two independently written implementations converging exactly is strong
evidence. It is **not** verification against a captured Qwen3.6 activation, and
it does not rule out Qwen3.6 having changed the gating relative to the
Qwen3-Next/3.5 code path they share.

**Derived, not ported.** No scalar chunked reference exists upstream — vLLM's
is Triton-JIT and llama.cpp has none — so the chunked form was derived from the
verified recurrence via the WY representation. The derivation yields
`(I + A)^{-1}` rather than the `(I − A)^{-1}` the planning document paraphrased;
both are valid under different sign conventions, and the equivalence test is
what actually validates it.

Other provenance: `moe::dispatch` follows vLLM's `moe_align_block_size` output
contract, verified against the CUDA source for padding and sentinel semantics.
Q6_K/Q8_0 *dequantization* is transcribed from `ggml-quants.c`; the
*quantizer* is a simple encoder written for round-trip testing only and is
explicitly **not** llama.cpp's least-squares quantizer. RoPE's NEOX pairing was
confirmed from `ggml_compute_forward_rope_flt`.

**Vision tower: ported and validated against upstream execution.** The
`vision` reference (patch conv, position-grid interpolation, 27 SigLIP
blocks, `qwen3vl_merger` projector) is ported from
`tools/mtmd/models/qwen3vl.cpp` and `tools/mtmd/clip.cpp`, and — unusually
for this crate — validated not just by construction but against llama.cpp
*running the same mmproj file*: `tests/vision_golden.rs` replays
`llama-mtmd-debug`'s synthetic images through the reference and compares the
final embeddings value-for-value against that tool's tensor dumps. Captured
logs are never committed; the test reads them from
`LLMXABE_MTMD_GOLDEN_DIR` and skips without it. Generate with, for `IMG` in
`gray`/`red`/`cb` and `N` in `64`/`96`:

```sh
llama-mtmd-debug -m <model.gguf> --mmproj <mmproj-F16.gguf> \
  -p encode --image $IMG -n $N --no-mmproj-offload -ngl 0 --no-warmup \
  > $LLMXABE_MTMD_GOLDEN_DIR/golden-$IMG-$N.log 2>&1
```

Agreement at last check: ≤ 3.2e-4 on output mass, ≤ 5e-3 per printed value —
the budget covers llama.cpp's f16 GELU table (`GGML_GELU_FP16`) against the
reference's exact tanh form. Text-side M-RoPE (`mrope`) is ported from
`ggml_mrope_cache_init`'s `IMROPE` branch and carries a bit-exactness test
that scalar positions collapse to `rope::apply_rope` — the same reduction
`xabe-engine`'s text path depends on.

## The dense architecture (`qwen35`)

Qwen3.8-27B shares every kernel with Qwen3.6-35B-A3B except the feed-forward
block, and the shared ones are gated by the tests above at Qwen3.6's widths.
What is new is gated by two files:

| Test | What it asserts | Needs |
| --- | --- | --- |
| `xabe-model/tests/real_dense_model_weights.rs` | `WeightSchema` resolves against all 866 tensors of the real file with none left unclaimed; every derived hyperparameter (the GDN head split above all) agrees with the file's own metadata; the file carries no routed tensor; and the *routed* schema refuses it by architecture rather than by a wall of shapes | the file |
| `xabe-engine/tests/dense_ffn_differential.rs` | The whole dense block against `expert_mlp` + `rms_norm` on real Q8_0 weights, at all three of its kernel paths | the file, a device |

The differential is the interesting one, because the dense block runs on the
MoE crate's shared-expert kernels and so what is genuinely new is the *block*:
that the post-mixer RMSNorm feeds the MLP, that the MLP has no gate of its
own, and that the residual added back is the block's input rather than its
normalized form. Each of those three has a wrong version that still generates
fluent text.

It also gates all three residency/width combinations against one reference,
which is what says they are the same computation rather than three:

| Path | `max_tokens` | `max_abs` | cosine |
| --- | ---: | ---: | ---: |
| Q8_0, fp32 `moe_shared_ffn` | 16 | 3.20e-4 | 1.000000 |
| split int8, `shared_expert_mma` | 128 | 6.45e-2 | 0.999997 |
| split int8, `dense_proj_split_t3` | 3 | 3.36e-4 | 1.000000 |

On a tensor whose own max magnitude is 82.3. The middle row is looser because
it is the only one that quantizes *activations* to int8; the GEMV dequantizes
the weights and multiplies in fp32, which is why the decode path is the
accurate one and the prefill path is the fast one.

`max_rel_error` is excluded from both gates for the floor reason the routed
MoE differential records, and the test asserts the element driving it really
is small against the tensor's own scale — so the exclusion fails loudly if it
stops being justified.

### End to end, against llama.cpp

The `forward_pass` golden is a Qwen3.6 capture and there is no dense
equivalent, so the end-to-end check is greedy agreement with
`llama-completion` at `--temp 0 --top-k 1` on the same file. Measured:

| Prompt | Q8_0 residency | int8 residency (shipped) |
| --- | --- | --- |
| "The capital of France is", 160 tokens | identical, 717/717 chars | identical, 690/690 chars |
| B-tree explanation, 160 tokens | identical, 717/717 chars | diverges at char 596 of 710 |

The divergence is the price of int8 activations in prefill and is stated
rather than smoothed over: with the fp32 residency the same prompt matches
llama.cpp character for character, and it is one constant
(`DENSE_REPACK_INT8`) away. What it buys is in [BENCHMARKS.md](BENCHMARKS.md).
The two continuations at the divergence are both correct English about
B-trees, i.e. a near-tie flipped, not a trajectory drifting.

## Serving acceptance status

Stated plainly, because a gap you know about is manageable and one you assume
away is not.

What has passed on the three local RTX 8000s: `worker_smoke`, `engine_smoke`
and `cross_worker_restore`. The engine run serves nine sequences, balances
three onto each worker, and exercises a mixed two-decode/one-prefill step on
every card; the cross-worker restore emits exactly the same token ids as the
cold path. Nine simultaneous HTTP `/v1/completions` requests are served by one
server bound to all three cards, every response producing the same
continuation.

The HTTP surface itself is checked by hand against a live server, because the
engine below it needs a GPU and a 30 GiB model and so cannot be stood up in a
`cargo test`. What that check has covered, on one RTX 8000:

- All four generation endpoints, streaming and not, plus `/v1/models` and
  `/v1/messages/count_tokens`.
- API-key authentication: accepted via both header spellings, refused with
  each dialect's own `401` envelope, and open when no key is set.
- Every refusal in [API.md](API.md): `n > 1`, a forcing `tool_choice`, a
  late system message, an image content part, `previous_response_id`,
  malformed JSON.
- Sampling: `temperature: 0` still greedy and exact; equal seeds replaying
  byte-identical completions on chat and raw completions; different seeds
  diverging; out-of-range `temperature` refused.
- Tool calling in all three chat dialects, streaming and not: calls parsed
  with schema-typed arguments, `tool_calls` / `tool_use` / `function_call`
  wire shapes, and tool-result round-trips answered from the result.
- Three concurrent streams on one worker, each returning its own correct
  answer.
- Disconnect cancellation: a prompt that runs 42 s to `max_tokens` leaves the
  card idle within 3 s of the client vanishing.
- A 4000-token generation, which crosses a GDN retention boundary — the case
  that used to fail the whole scheduler step.
- **Prefix reuse through generated tokens, against a cold control.** A
  14-token prompt generated 2,600 tokens; a second request whose prompt was
  that prompt plus that output reused 2,048 tokens, none of which the first
  request's prompt could have named. The resumed continuation was compared
  against the identical request on a freshly started server that had never
  seen the text — byte-identical, which is the check that matters, because a
  chain naming the wrong prefix produces fluent output rather than an error.
  Four interleaved pairs on unique prompts put the saving at 2.1× on time to
  first token for a 2,621-token prompt, and every pair's resumed answer
  matched its cold arm.

Everything below the wire format is covered by the unit tests in
`crates/xabe-server/src/http/`: the ChatML rendering is pinned string by
string, and stop-sequence hold-back, header parsing, and constant-time key
comparison have their own tests.

What none of that covers: overload behaviour, and any of it under sustained
load rather than by hand.

The scheduler-driven path measures within 0.7% of the isolated kernel path, so
runtime plumbing is not where throughput goes. Serving numbers belong in
`BENCHMARKS.md`, not here.

## Running

```sh
cargo test --workspace                # everything
cargo test -p xabe-kernels            # references and harness
cargo test -p xabe-kernels gdn        # the critical path
cargo run -p xabe-cuda --bin probe    # device gate and milestone-00 spike
```

Serving acceptance uses the scheduler-driven runtime rather than the isolated
forward benchmarks:

```sh
# One visible card, N=1 or N=3 through Worker::step_device.
CUDA_VISIBLE_DEVICES=0 LLMXABE_BATCH_N=3 \
  cargo run --release -p xabe-engine --bin worker_smoke

# Three visible cards, nine requests, including a 2-decode + 1-prefill step
# on every worker.
CUDA_VISIBLE_DEVICES=0,1,2 \
  cargo run --release -p xabe-engine --bin engine_smoke

# Two separate CUDA contexts: cold 2K prefill versus pinned-host KV/GDN
# restore, compared by exact emitted token ids.
CUDA_VISIBLE_DEVICES=0,1 \
  cargo run --release -p xabe-engine --bin cross_worker_restore
```

These commands are acceptance checks, not benchmarks. Run the interleaved
benchmark commands from `BENCHMARKS.md` before making a throughput claim, and
follow its measurement discipline — a single pair on this host proves nothing.

Tests needing the 32 GB model file or a GPU **skip and say so**. A skipped test
is not a passing test — do not read a green run on a GPU-less machine as
validation of device work. See
[CONTRIBUTING.md](../CONTRIBUTING.md#reporting-results).

The corollary bites on a machine where the cards *are* free: `cargo test
--workspace --release` runs each integration binary concurrently, and the
device ones each load the whole model, so `batch_decode` and `batch_prefill`
together exhaust a 48 GiB card and fail with `CUDA_ERROR_OUT_OF_MEMORY`. That
is contention, not a regression — the same tests pass serialized:

```sh
CUDA_VISIBLE_DEVICES=0 cargo test --release -p xabe-engine --test batch_decode  -- --test-threads=1
CUDA_VISIBLE_DEVICES=0 cargo test --release -p xabe-engine --test batch_prefill -- --test-threads=1
cargo test --workspace --release --lib --bins
```

Read an OOM in a whole-workspace run as "run them one at a time", and check
before concluding a device kernel broke.
