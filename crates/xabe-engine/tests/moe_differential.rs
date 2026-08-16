//! Differential test: the device MoE path against the `xabe-kernels` CPU
//! reference, at Qwen3.6's real MoE geometry and on real quantized expert
//! weights taken from the model file.
//!
//! This is the milestone-05 gate. MoE runs on **every one of the 40 layers**
//! plus the MTP head, so an error here is an error on every token; and
//! unlike the numeric kernels, most of what can go wrong is *discrete* — a
//! different expert selected, a token placed in the wrong run, a padding
//! slot read as a real token. Those do not show up as a slightly worse
//! cosine similarity. They show up as a different model.
//!
//! ## What is compared, and how strictly
//!
//! | Stage | Reference | Gate |
//! |---|---|---|
//! | routing | `route_batch` | selected expert ids **exactly**, weights under tolerance |
//! | dispatch tables | `moe_align_block_size` | `sorted_token_ids` and `expert_ids` **exactly**, padding slots included |
//! | grouped GEMM | `grouped_forward`, cross-checked against `naive_forward` | tolerance |
//! | shared expert | `expert_mlp` | tolerance |
//!
//! The first two are exact because they have no floating-point freedom in
//! their content: a top-k selection is a set of integers and a dispatch
//! table is a permutation with padding. Only the GEMM output is gated on a
//! tolerance, and only because the reference sums a 2,048-term dot product
//! sequentially while the kernel reduces it in a warp-shuffle tree.
//!
//! ## Why real weights
//!
//! Synthetic fp32 experts would exercise none of the Q6_K per-group scale
//! distribution, none of its sign patterns, and none of the denormal deltas
//! the quantizer emits for near-zero blocks. They would also hide the fact —
//! verified here against the file's own tensor directory, not assumed — that
//! `Qwen3.6-35B-A3B-UD-Q6_K_XL` is **mixed**: `ffn_gate_exps` and
//! `ffn_up_exps` are Q6_K but `ffn_down_exps` is Q8_0. A kernel that
//! hard-coded one format passes on synthetic data and reads garbage on the
//! real file.
//!
//! Only one layer's expert stacks are loaded (about 700 MiB of the model's
//! 29.65 GiB); the full model is never resident.
//!
//! SKIPS — reporting that it skipped — without a driver, a supported device,
//! or the model file.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_cuda::kernels::moe::{ExpertQuant, MoeBuffers, MoeGeometry, MoeKernels, QuantTensor};
use xabe_gguf::{GgmlType, GgufFile};
use xabe_kernels::compare::{Tolerance, assert_matches, compare};
use xabe_kernels::moe::dispatch::{INACTIVE_EXPERT, moe_align_block_size, padding_sentinel};
use xabe_kernels::moe::gemm::{ExpertWeights, expert_mlp, grouped_forward, naive_forward};
use xabe_kernels::moe::router::{RoutingDecision, route_batch};
use xabe_kernels::quant::{dequantize_row_q6_k, dequantize_row_q8_0};
use xabe_kernels::rng::Xorshift64Star;
use xabe_model::config::ModelConfig;
use xabe_model::weights::{Role, WeightSchema};

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llama.cpp/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

/// Tokens in the batch under test.
///
/// 37 is deliberately awkward: it is not a multiple of [`BLOCK_SIZE`], and
/// `37 * 8 = 296` flat `(token, k)` pairs is not either, so essentially
/// every active expert's run ends mid-block and the padding slots are
/// exercised rather than incidentally absent.
const NUM_TOKENS: usize = 37;

/// Grouped-GEMM tile width the dispatch tables pad to.
const BLOCK_SIZE: usize = 16;

/// Buffer capacity, in tokens. Larger than [`NUM_TOKENS`] on purpose: it is
/// what proves the launch shapes come from the geometry rather than from the
/// live batch, and it means the sentinel fill has to cover slots the live
/// step never touches.
const MAX_TOKENS: usize = 64;

/// The layer whose expert stacks are pulled from the file.
const LAYER: u32 = 0;

/// Fixed seeds, one per test, so a failure is reproducible from the name
/// alone rather than from whatever order the harness ran things in.
const ROUTE_SEED: u64 = 0x_5EED_0A01;
const DISPATCH_SEED: u64 = 0x_5EED_0B02;
const GEMM_ROUTE_SEED: u64 = 0x_5EED_0C03;
const GEMM_INPUT_SEED: u64 = 0x_5EED_0D04;
const SHARED_INPUT_SEED: u64 = 0x_5EED_0E05;

/// Tolerance for the routed grouped GEMM against the fp32 scalar reference.
///
/// The dequantized *weights* are bit-identical to the reference — the
/// milestone-04 gate proved that, and this module reuses the same operand
/// order. What is left is a 2,048-term dot product summed sequentially on
/// the host and in a warp-shuffle tree on the device, plus the same again
/// over 512 terms in the down projection. fp32 addition is not associative,
/// so the gate is a tolerance.
///
/// **`max_abs_error` and `min_cosine_similarity` are the gate;
/// `max_rel_error` is not.** `compare()` computes relative error as
/// `|c - r| / max(|r|, 1e-6)`, and a MoE output whose mean magnitude is
/// ~1.5e-3 is full of elements below that floor — for those the denominator
/// *is* the floor, so the ratio reports `abs_error / 1e-6` rather than
/// anything about accuracy. The tests assert, below, that the element
/// driving `max_rel_error` really is near zero, so if that stops being true
/// the justification fails loudly instead of quietly covering a real error.
///
/// Measured worst case at the real geometry, 37 tokens x top-8 of 256, real
/// Q6_K gate/up and Q8_0 down from layer 0: `max_abs = 9.78e-9`,
/// `cosine = 1.000000`. The bound below is ~51x that — room for hardware and
/// driver variation, not room for a formulation bug, which would land orders
/// of magnitude away on an output whose own max magnitude is only 1.0e-2.
/// The same output from the integer tensor-core path, which is a different
/// arithmetic and needs a different bound.
///
/// [`ROUTED_GATE`] gates fp32 summation order — a disagreement in the last
/// bits. This gates something larger and deliberate: the activations are
/// quantized to int8 with one fp32 scale per 32, which costs about `1/254` of
/// the block's largest magnitude per element. The *weights* are not
/// approximated at all — a Q6_K quant is an integer in `[-32, 31]` and the
/// tensor core multiplies it exactly — and the int32 accumulation is exact, so
/// activation quantization is the whole of the error.
///
/// Measured at the real geometry, 37 tokens x top-8 of 256, real Q6_K gate/up
/// from layer 0: `max_abs = 4.59e-5`, `cosine = 0.999988`, on an output whose
/// own max magnitude is 1.0e-2. The bound is ~2x the measured worst case.
///
/// It is still a real gate. The characteristic defect of hand-written MMA is a
/// wrong fragment layout — mixing up the operand split (stride 4) with the
/// accumulator split (stride 2) — and that does not produce a slightly worse
/// answer, it produces a differently-shaped one. The cosine floor is what
/// catches it; `max_abs_error` alone would not.
///
/// `max_rel_error` is excluded for the same floor reason as [`ROUTED_GATE`].
const ROUTED_MMA_GATE: Tolerance = Tolerance {
    max_abs_error: 1.0e-4,
    max_rel_error: f32::INFINITY,
    min_cosine_similarity: 0.9999,
    allow_non_finite: false,
};

const ROUTED_GATE: Tolerance = Tolerance {
    max_abs_error: 5e-7,
    max_rel_error: 5e-2,
    min_cosine_similarity: 1.0 - 1e-6,
    allow_non_finite: false,
};

/// As [`ROUTED_GATE`], for the shared expert.
///
/// Looser in absolute terms for one reason: the shared expert carries no
/// routing weight, so its output is not divided across 8 contributions and
/// its magnitude is correspondingly larger. Measured worst case on layer 0's
/// real Q8_0 shared expert: `max_abs = 2.98e-7`, `cosine = 1.000000`; the
/// bound is ~17x that.
const SHARED_GATE: Tolerance = Tolerance {
    max_abs_error: 5e-6,
    max_rel_error: 5e-2,
    min_cosine_similarity: 1.0 - 1e-6,
    allow_non_finite: false,
};

fn model_path() -> PathBuf {
    std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
}

/// A context on the only visible device, or `None` with a printed reason.
fn device() -> Option<Arc<CudaContext>> {
    if !driver_available() {
        println!("SKIPPED: no CUDA driver present");
        return None;
    }
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            println!("SKIPPED: could not create a context on device 0: {e}");
            return None;
        }
    };
    let info = DeviceInfo::from_context(0, &ctx).expect("device properties readable");
    if !info.is_supported() {
        println!("SKIPPED: device 0 is below the sm_75 minimum");
        return None;
    }
    Some(ctx)
}

fn device_and_model() -> Option<(Arc<CudaContext>, GgufFile)> {
    let ctx = device()?;
    let path = model_path();
    if !path.exists() {
        println!("SKIPPED: model file not found at {}", path.display());
        return None;
    }
    Some((ctx, GgufFile::open(&path).expect("valid GGUF v3")))
}

fn geometry() -> MoeGeometry {
    let config = ModelConfig::qwen3_6_35b_a3b();
    MoeGeometry {
        num_experts: config.moe.num_experts as usize,
        experts_per_token: config.moe.experts_per_token as usize,
        hidden: config.hidden_size as usize,
        intermediate: config.moe.expert_intermediate as usize,
        block_size: BLOCK_SIZE,
        max_tokens: MAX_TOKENS,
    }
}

/// Router logits with realistic spread, plus the same rows padded to the
/// buffer capacity.
///
/// Rows past `NUM_TOKENS` are filled with a hugely negative constant rather
/// than zeros: if a kernel ever routed a slot it should not, the resulting
/// selection would be the fixed set `0..k` rather than something that blends
/// in with real routing.
fn router_logits(g: &MoeGeometry, seed: u64) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let live: Vec<Vec<f32>> = (0..NUM_TOKENS)
        .map(|_| rng.vec_f32(g.num_experts, -8.0, 8.0))
        .collect();
    let mut flat: Vec<f32> = Vec::with_capacity(g.max_tokens * g.num_experts);
    for row in &live {
        flat.extend_from_slice(row);
    }
    flat.resize(g.max_tokens * g.num_experts, -1.0e30);
    (live, flat)
}

fn dequant_slice(ty: GgmlType, bytes: &[u8]) -> Vec<f32> {
    match ty {
        GgmlType::Q6K => dequantize_row_q6_k(bytes).expect("reference q6_K"),
        GgmlType::Q8_0 => dequantize_row_q8_0(bytes).expect("reference q8_0"),
        other => panic!(
            "expert stack is {}, which this test cannot unpack",
            other.name()
        ),
    }
}

fn quant_of(ty: GgmlType) -> ExpertQuant {
    match ty {
        GgmlType::Q6K => ExpertQuant::Q6K,
        GgmlType::Q8_0 => ExpertQuant::Q8_0,
        other => panic!("unexpected expert stack type {}", other.name()),
    }
}

/// Serialized bytes and elements per block for a GGUF quantized type.
fn block_shape(ty: GgmlType) -> (usize, usize) {
    let q = quant_of(ty);
    (q.block_bytes(), q.block_elements())
}

/// Guard against a tensor that would compare perfectly while proving
/// nothing.
fn assert_carries_signal(name: &str, v: &[f32]) {
    let nonzero = v.iter().filter(|x| **x != 0.0).count();
    let frac = nonzero as f64 / v.len() as f64;
    assert!(
        frac > 0.25,
        "{name}: only {:.1}% of {} sampled values are non-zero — an all-zero \
         tensor would pass every comparison below vacuously",
        frac * 100.0,
        v.len(),
    );
    assert!(
        v.iter().all(|x| x.is_finite()),
        "{name}: contains a non-finite value",
    );
}

/// Route on the device and read the decision back.
fn device_routing(
    kernels: &MoeKernels,
    stream: &Arc<CudaStream>,
    buffers: &mut MoeBuffers,
    logits: &CudaSlice<f32>,
) -> (Vec<i32>, Vec<f32>) {
    kernels.route(stream, buffers, logits).expect("route");
    let ids = stream.clone_dtoh(buffers.topk_ids()).expect("ids back");
    let weights = stream
        .clone_dtoh(buffers.topk_weights())
        .expect("weights back");
    stream.synchronize().expect("sync");
    (ids, weights)
}

/// The device's top-k for one token, as `u32` for comparison against
/// [`RoutingDecision::expert_ids`].
fn device_ids_for(ids: &[i32], token: usize, top_k: usize) -> Vec<u32> {
    ids[token * top_k..(token + 1) * top_k]
        .iter()
        .map(|&v| v as u32)
        .collect()
}

// ---------------------------------------------------------------------------
// (a) routing
// ---------------------------------------------------------------------------

#[test]
fn device_routing_selects_exactly_the_same_experts_as_route_batch() {
    let Some(ctx) = device() else { return };
    let g = geometry();
    let stream = ctx.default_stream();
    let kernels = MoeKernels::new(&ctx, g).expect("kernels must compile for sm_75");
    let mut buffers = kernels.buffers(&stream).expect("buffers allocate");

    println!(
        "geometry: {} experts, top-{}, hidden {}, intermediate {}, block_size {}, \
         {NUM_TOKENS} live tokens in a {MAX_TOKENS}-token buffer",
        g.num_experts, g.experts_per_token, g.hidden, g.intermediate, g.block_size,
    );

    let (live, flat) = router_logits(&g, ROUTE_SEED);
    let all: Vec<f32> = live.concat();
    assert_carries_signal("router logits", &all);

    let d_logits = stream.clone_htod(&flat).expect("upload logits");
    kernels
        .set_valid_tokens(&stream, &mut buffers, NUM_TOKENS)
        .expect("valid_tokens");
    let (ids, weights) = device_routing(&kernels, &stream, &mut buffers, &d_logits);

    let reference: Vec<RoutingDecision> = route_batch(&live, g.experts_per_token);

    let mut mismatched = 0usize;
    for (t, decision) in reference.iter().enumerate() {
        let got = device_ids_for(&ids, t, g.experts_per_token);
        if got != decision.expert_ids {
            mismatched += 1;
            if mismatched <= 3 {
                println!(
                    "  token {t}: device {got:?} vs reference {:?}",
                    decision.expert_ids
                );
            }
        }
    }
    assert_eq!(
        mismatched, 0,
        "{mismatched} of {NUM_TOKENS} tokens selected a different expert set — \
         this comparison is exact on purpose: a different tie-break or a \
         different top-k silently runs a different model",
    );

    let candidate: Vec<f32> = weights[..NUM_TOKENS * g.experts_per_token].to_vec();
    let expected: Vec<f32> = reference.iter().flat_map(|d| d.weights.clone()).collect();
    let result = compare(&candidate, &expected);
    println!(
        "routing weights over {} (token, k) pairs: {result}",
        candidate.len(),
    );
    assert_matches(
        &candidate,
        &expected,
        &Tolerance {
            max_abs_error: 1e-6,
            max_rel_error: 1e-3,
            min_cosine_similarity: 1.0 - 1e-9,
            allow_non_finite: false,
        },
    );

    // Every token's weights must sum to 1 after renormalization; a kernel
    // that renormalized over all 256 experts instead of the selected 8 would
    // score near-perfect on cosine and fail here outright.
    for t in 0..NUM_TOKENS {
        let base = t * g.experts_per_token;
        let sum: f32 = candidate[base..base + g.experts_per_token].iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "token {t}: weights sum to {sum}");
    }

    println!(
        "expert-id selection matched EXACTLY on all {NUM_TOKENS} tokens \
         x top-{} over {} experts",
        g.experts_per_token, g.num_experts,
    );
}

#[test]
fn an_exact_tie_across_every_expert_resolves_to_the_lowest_indices() {
    // The one case where "close enough" is no defence: with identical logits
    // the reference picks experts 0..k-1, and any tie-break that depends on
    // which thread won a shuffle picks something else — and something
    // different again on the next run.
    let Some(ctx) = device() else { return };
    let g = geometry();
    let stream = ctx.default_stream();
    let kernels = MoeKernels::new(&ctx, g).expect("compiles");
    let mut buffers = kernels.buffers(&stream).expect("buffers");

    let flat = vec![0.5f32; g.max_tokens * g.num_experts];
    let d_logits = stream.clone_htod(&flat).expect("upload");
    kernels
        .set_valid_tokens(&stream, &mut buffers, NUM_TOKENS)
        .expect("valid_tokens");

    let expected: Vec<i32> = (0..g.experts_per_token as i32).collect();
    for run in 0..4 {
        let (ids, weights) = device_routing(&kernels, &stream, &mut buffers, &d_logits);
        for t in 0..NUM_TOKENS {
            let base = t * g.experts_per_token;
            assert_eq!(
                &ids[base..base + g.experts_per_token],
                &expected[..],
                "run {run}, token {t}: tie-break did not prefer the lowest indices",
            );
        }
        let w = 1.0 / g.experts_per_token as f32;
        for &x in &weights[..NUM_TOKENS * g.experts_per_token] {
            assert!(
                (x - w).abs() < 1e-6,
                "tied weights should be uniform, got {x}"
            );
        }
    }
    println!("all-equal logits resolve to experts {expected:?} on every run");
}

// ---------------------------------------------------------------------------
// (b) dispatch tables
// ---------------------------------------------------------------------------

#[test]
fn device_dispatch_tables_match_moe_align_block_size_including_padding() {
    let Some(ctx) = device() else { return };
    let g = geometry();
    let stream = ctx.default_stream();
    let kernels = MoeKernels::new(&ctx, g).expect("compiles");
    let mut buffers = kernels.buffers(&stream).expect("buffers");

    let (live, flat) = router_logits(&g, DISPATCH_SEED);
    let d_logits = stream.clone_htod(&flat).expect("upload logits");
    kernels
        .set_valid_tokens(&stream, &mut buffers, NUM_TOKENS)
        .expect("valid_tokens");
    let (ids, _) = device_routing(&kernels, &stream, &mut buffers, &d_logits);

    let reference = route_batch(&live, g.experts_per_token);
    let topk_ids: Vec<Vec<u32>> = reference.iter().map(|d| d.expert_ids.clone()).collect();

    // The tables mean nothing unless both sides built them from the same
    // selection, so establish that before comparing them.
    for (t, row) in topk_ids.iter().enumerate() {
        assert_eq!(
            &device_ids_for(&ids, t, g.experts_per_token),
            row,
            "token {t}: routing diverged before dispatch",
        );
    }

    kernels
        .build_dispatch(&stream, &mut buffers)
        .expect("dispatch");
    let d_sorted = stream
        .clone_dtoh(buffers.sorted_token_ids())
        .expect("sorted back");
    let d_experts = stream
        .clone_dtoh(buffers.expert_ids())
        .expect("expert ids back");
    let d_post_pad = stream
        .clone_dtoh(buffers.num_tokens_post_pad())
        .expect("post-pad back");
    stream.synchronize().expect("sync");

    let expected = moe_align_block_size(&topk_ids, g.block_size, g.num_experts);
    let sentinel = padding_sentinel(NUM_TOKENS, g.experts_per_token);
    let post_pad = d_post_pad[0] as usize;

    assert_eq!(
        post_pad, expected.num_tokens_post_pad,
        "num_tokens_post_pad disagrees; the device value lives in device \
         memory and is what a captured graph would gate on",
    );

    let device_sorted: Vec<u32> = d_sorted[..post_pad].iter().map(|&v| v as u32).collect();
    assert_eq!(
        device_sorted, expected.sorted_token_ids,
        "sorted_token_ids differ — compared exactly, padding slots included",
    );

    let num_blocks = post_pad / g.block_size;
    assert_eq!(
        &d_experts[..num_blocks],
        &expected.expert_ids[..],
        "expert_ids differ",
    );

    // Slots past the live region must read as padding, not as a previous
    // step's tokens.
    assert!(
        d_sorted[post_pad..].iter().all(|&v| v as u32 == sentinel),
        "capacity past num_tokens_post_pad is not filled with the sentinel",
    );
    assert!(
        d_experts[num_blocks..]
            .iter()
            .all(|&v| v == INACTIVE_EXPERT),
        "capacity past the last active block is not INACTIVE_EXPERT",
    );

    let padding_slots = device_sorted.iter().filter(|&&v| v == sentinel).count();
    let real_slots = device_sorted.len() - padding_slots;
    let active_experts: BTreeSet<i32> = d_experts[..num_blocks].iter().copied().collect();
    assert_eq!(
        real_slots,
        NUM_TOKENS * g.experts_per_token,
        "every (token, k) pair must appear exactly once among the valid slots",
    );
    assert!(
        padding_slots > 0,
        "no padding slots were exercised — {NUM_TOKENS} tokens x top-{} must \
         not divide evenly into blocks of {}",
        g.experts_per_token,
        g.block_size,
    );

    println!(
        "dispatch: {NUM_TOKENS} tokens x top-{} = {real_slots} pairs over {} active \
         experts; num_tokens_post_pad={post_pad} in {num_blocks} blocks of {} \
         ({padding_slots} padding slots); fixed capacity {} slots / {} blocks",
        g.experts_per_token,
        active_experts.len(),
        g.block_size,
        g.sorted_capacity(),
        g.expert_block_capacity(),
    );
    println!("sorted_token_ids and expert_ids matched EXACTLY, padding included");
}

// ---------------------------------------------------------------------------
// (c) + (d) grouped GEMM on real quantized weights
// ---------------------------------------------------------------------------

#[test]
fn device_grouped_forward_matches_the_reference_on_real_expert_weights() {
    let Some((ctx, file)) = device_and_model() else {
        return;
    };
    let config = ModelConfig::qwen3_6_35b_a3b();
    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).expect("schema must resolve");
    let g = geometry();
    let stream = ctx.default_stream();
    let mut kernels = MoeKernels::new(&ctx, g).expect("compiles");
    let mut buffers = kernels.buffers(&stream).expect("buffers");

    // --- real expert stacks for one layer --------------------------------
    let stacks: Vec<_> = [Role::MoeGateExps, Role::MoeUpExps, Role::MoeDownExps]
        .iter()
        .map(|&role| {
            directory
                .find(role, Some(LAYER))
                .unwrap_or_else(|| panic!("{role} on layer {LAYER} missing"))
        })
        .collect();
    println!(
        "layer {LAYER} expert stacks: {}",
        stacks
            .iter()
            .map(|e| format!(
                "{}={} {:?}",
                e.spec.role,
                e.info.ggml_type.name(),
                e.spec.dims
            ))
            .collect::<Vec<_>>()
            .join("  "),
    );
    // The file is mixed; if that ever stops being true this test should say
    // so rather than quietly stop covering one of the two prologues.
    assert!(
        stacks.iter().any(|e| e.info.ggml_type == GgmlType::Q6K),
        "no Q6_K expert stack on layer {LAYER}; the Q6_K prologue would go untested",
    );

    let bytes: Vec<&[u8]> = stacks
        .iter()
        .map(|e| file.tensor_bytes(&e.spec.name).expect("tensor readable"))
        .collect();

    let t0 = Instant::now();
    let d_gate = stream.clone_htod(bytes[0]).expect("upload gate");
    let d_up = stream.clone_htod(bytes[1]).expect("upload up");
    let d_down = stream.clone_htod(bytes[2]).expect("upload down");
    stream.synchronize().expect("sync");
    println!(
        "uploaded {:.1} MiB of quantized expert weights in {:.2?}",
        bytes.iter().map(|b| b.len()).sum::<usize>() as f64 / (1024.0 * 1024.0),
        t0.elapsed(),
    );

    // --- activations ------------------------------------------------------
    let mut rng = Xorshift64Star::new(GEMM_INPUT_SEED);
    let hidden_states: Vec<Vec<f32>> = (0..NUM_TOKENS)
        .map(|_| rng.vec_f32(g.hidden, -1.0, 1.0))
        .collect();
    let live_hidden: Vec<f32> = hidden_states.concat();
    assert_carries_signal("hidden states", &live_hidden);
    let mut flat_hidden = live_hidden.clone();
    flat_hidden.resize(g.max_tokens * g.hidden, 0.0);
    let d_hidden = stream.clone_htod(&flat_hidden).expect("upload hidden");

    let (live_logits, flat_logits) = router_logits(&g, GEMM_ROUTE_SEED);
    let d_logits = stream.clone_htod(&flat_logits).expect("upload logits");

    // --- device ------------------------------------------------------------
    kernels
        .set_valid_tokens(&stream, &mut buffers, NUM_TOKENS)
        .expect("valid_tokens");
    kernels
        .route(&stream, &mut buffers, &d_logits)
        .expect("route");
    kernels
        .build_dispatch(&stream, &mut buffers)
        .expect("dispatch");

    let mut d_out = stream
        .alloc_zeros::<f32>(g.max_tokens * g.hidden)
        .expect("out allocates");
    let gate = QuantTensor {
        bytes: &d_gate,
        quant: quant_of(stacks[0].info.ggml_type),
    };
    let up = QuantTensor {
        bytes: &d_up,
        quant: quant_of(stacks[1].info.ggml_type),
    };
    let down = QuantTensor {
        bytes: &d_down,
        quant: quant_of(stacks[2].info.ggml_type),
    };

    // The integer path first, while `kernels` still has its `MmaKernels`.
    // Both runs go through the same dispatch and the same down projection;
    // only the gate/up GEMM differs, which is what makes the two comparisons
    // below attributable.
    assert!(
        kernels.tensor_cores_enabled(),
        "this device compiled the integer kernels, so the test must exercise          them — a silent fp32-only run would report a passing gate for a path          it never touched",
    );
    let t_mma = Instant::now();
    kernels
        .grouped_forward(&stream, &mut buffers, gate, up, down, &d_hidden, &mut d_out)
        .expect("grouped forward, integer tensor cores");
    stream.synchronize().expect("sync");
    let mma_time = t_mma.elapsed();
    let mma_full = stream.clone_dtoh(&d_out).expect("out back");
    stream.synchronize().expect("sync");
    let mma_out: Vec<f32> = mma_full[..NUM_TOKENS * g.hidden].to_vec();

    kernels.disable_tensor_cores();
    let t1 = Instant::now();
    kernels
        .grouped_forward(&stream, &mut buffers, gate, up, down, &d_hidden, &mut d_out)
        .expect("grouped forward");
    stream.synchronize().expect("sync");
    let gpu_time = t1.elapsed();
    let full_out = stream.clone_dtoh(&d_out).expect("out back");
    let device_ids = stream.clone_dtoh(buffers.topk_ids()).expect("ids back");
    stream.synchronize().expect("sync");
    let device_out: Vec<f32> = full_out[..NUM_TOKENS * g.hidden].to_vec();
    println!(
        "device grouped forward (3 launches + one memset): fp32 {gpu_time:.2?},          int8 tensor cores {mma_time:.2?}"
    );

    // --- host reference ---------------------------------------------------
    let routing = route_batch(&live_logits, g.experts_per_token);
    for (t, decision) in routing.iter().enumerate() {
        assert_eq!(
            &device_ids_for(&device_ids, t, g.experts_per_token),
            &decision.expert_ids,
            "token {t}: routing diverged",
        );
    }

    let topk_ids: Vec<Vec<u32>> = routing.iter().map(|d| d.expert_ids.clone()).collect();
    let flat_weights: Vec<f32> = routing.iter().flat_map(|d| d.weights.clone()).collect();
    let dispatch = moe_align_block_size(&topk_ids, g.block_size, g.num_experts);

    // Only the experts this batch routes to are unpacked on the host: all 256
    // in fp32 would be 3.1 GiB for one layer, and neither reference path ever
    // indexes an unselected expert.
    let selected: BTreeSet<u32> = topk_ids.iter().flatten().copied().collect();
    let per_expert = g.intermediate * g.hidden;
    let t2 = Instant::now();
    let mut experts: Vec<ExpertWeights> = (0..g.num_experts)
        .map(|_| ExpertWeights {
            gate: Vec::new(),
            up: Vec::new(),
            down: Vec::new(),
        })
        .collect();
    let mut sampled_weights: Vec<f32> = Vec::new();
    for &e in &selected {
        let e = e as usize;
        let cut = |i: usize| {
            let ty = stacks[i].info.ggml_type;
            let (block_bytes, block_elems) = block_shape(ty);
            let span = per_expert / block_elems * block_bytes;
            dequant_slice(ty, &bytes[i][e * span..(e + 1) * span])
        };
        let w = ExpertWeights {
            gate: cut(0),
            up: cut(1),
            down: cut(2),
        };
        if sampled_weights.is_empty() {
            sampled_weights.extend_from_slice(&w.gate[..4096]);
            sampled_weights.extend_from_slice(&w.down[..4096]);
        }
        experts[e] = w;
    }
    println!(
        "unpacked {} of {} experts on the host in {:.2?} ({:.2} GiB fp32)",
        selected.len(),
        g.num_experts,
        t2.elapsed(),
        (selected.len() * per_expert * 3 * 4) as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    assert_carries_signal("real dequantized expert weights", &sampled_weights);

    let t3 = Instant::now();
    let reference_grouped = grouped_forward(
        &hidden_states,
        &dispatch,
        &flat_weights,
        g.experts_per_token,
        &experts,
        g.hidden,
        g.intermediate,
    );
    let reference_naive =
        naive_forward(&hidden_states, &routing, &experts, g.hidden, g.intermediate);
    println!("host references: {:.2?}", t3.elapsed());

    let flat_grouped: Vec<f32> = reference_grouped.concat();
    let flat_naive: Vec<f32> = reference_naive.concat();
    assert_carries_signal("reference grouped output", &flat_grouped);

    // The reference's own cross-check: the dispatch path and the naive
    // per-token loop are structurally different code over the same math.
    let oracle = compare(&flat_grouped, &flat_naive);
    println!("host grouped_forward vs naive_forward: {oracle}");
    assert!(
        oracle.max_abs_error < 1e-4 && oracle.cosine_similarity > 1.0 - 1e-6,
        "the CPU oracle disagrees with itself: {oracle}",
    );

    let vs_grouped = compare(&device_out, &flat_grouped);
    let vs_naive = compare(&device_out, &flat_naive);
    println!("device vs host grouped_forward: {vs_grouped}");
    println!("device vs host naive_forward:   {vs_naive}");
    println!(
        "output magnitude: max |ref| = {:.4e}, mean |ref| = {:.4e}",
        flat_grouped.iter().fold(0.0f32, |m, v| m.max(v.abs())),
        flat_grouped.iter().map(|v| v.abs()).sum::<f32>() / flat_grouped.len() as f32,
    );

    // Evidence for GATE's claim that max_rel_error is a floor artefact: the
    // element driving it must be near zero. If it stops being, the absolute
    // bound is no longer a sufficient gate and this fails loudly.
    let driver = flat_grouped[vs_grouped.max_rel_error_index].abs();
    assert!(
        driver < 1e-3,
        "max_rel_error {:.3e} sits on a reference value of {driver:.3e}, large \
         enough for the ratio to mean something — max_abs_error alone is no \
         longer a sufficient gate",
        vs_grouped.max_rel_error,
    );

    assert_matches(&device_out, &flat_grouped, &ROUTED_GATE);
    assert_matches(&device_out, &flat_naive, &ROUTED_GATE);

    // The integer path against the same host reference, at the bound int8
    // activations permit rather than the one fp32 arithmetic does.
    let vs_mma = compare(&mma_out, &flat_grouped);
    println!("device int8 tensor cores vs host grouped_forward: {vs_mma}");
    assert_matches(&mma_out, &flat_grouped, &ROUTED_MMA_GATE);
    assert!(
        mma_full[NUM_TOKENS * g.hidden..].iter().all(|&v| v == 0.0),
        "the integer path wrote past the live token range",
    );

    // Slots the live batch never used must be untouched, not filled with a
    // stale or wrapped-around token's output.
    assert!(
        full_out[NUM_TOKENS * g.hidden..].iter().all(|&v| v == 0.0),
        "the kernel wrote past the {NUM_TOKENS} live tokens",
    );

    println!(
        "gate: max_abs<{:.0e}, cosine>{:.9} (max_rel is not the gate; see ROUTED_GATE)",
        ROUTED_GATE.max_abs_error, ROUTED_GATE.min_cosine_similarity,
    );
    println!(
        "fixed buffers: {:.2} MiB, allocated once",
        buffers.bytes() as f64 / (1024.0 * 1024.0),
    );
}

// ---------------------------------------------------------------------------
// shared expert, hoisted out of the routed path
// ---------------------------------------------------------------------------

#[test]
fn the_shared_expert_runs_for_every_token_with_no_routing() {
    let Some((ctx, file)) = device_and_model() else {
        return;
    };
    let config = ModelConfig::qwen3_6_35b_a3b();
    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).expect("schema resolves");
    let g = geometry();
    let stream = ctx.default_stream();
    let kernels = MoeKernels::new(&ctx, g).expect("compiles");
    let mut buffers = kernels.buffers(&stream).expect("buffers");

    let entries: Vec<_> = [Role::MoeSharedGate, Role::MoeSharedUp, Role::MoeSharedDown]
        .iter()
        .map(|&r| {
            directory
                .find(r, Some(LAYER))
                .unwrap_or_else(|| panic!("{r} missing"))
        })
        .collect();
    let bytes: Vec<&[u8]> = entries
        .iter()
        .map(|e| file.tensor_bytes(&e.spec.name).expect("readable"))
        .collect();
    println!(
        "layer {LAYER} shared expert: {}",
        entries
            .iter()
            .map(|e| format!("{}={}", e.spec.role, e.info.ggml_type.name()))
            .collect::<Vec<_>>()
            .join("  "),
    );

    let host = ExpertWeights {
        gate: dequant_slice(entries[0].info.ggml_type, bytes[0]),
        up: dequant_slice(entries[1].info.ggml_type, bytes[1]),
        down: dequant_slice(entries[2].info.ggml_type, bytes[2]),
    };
    assert_carries_signal("shared expert weights", &host.gate);

    let d_gate = stream.clone_htod(bytes[0]).expect("upload");
    let d_up = stream.clone_htod(bytes[1]).expect("upload");
    let d_down = stream.clone_htod(bytes[2]).expect("upload");

    let mut rng = Xorshift64Star::new(SHARED_INPUT_SEED);
    let hidden_states: Vec<Vec<f32>> = (0..NUM_TOKENS)
        .map(|_| rng.vec_f32(g.hidden, -1.0, 1.0))
        .collect();
    let mut flat_hidden: Vec<f32> = hidden_states.concat();
    assert_carries_signal("hidden states", &flat_hidden);
    flat_hidden.resize(g.max_tokens * g.hidden, 0.0);
    let d_hidden = stream.clone_htod(&flat_hidden).expect("upload");

    let mut d_out = stream
        .alloc_zeros::<f32>(g.max_tokens * g.hidden)
        .expect("out");
    kernels
        .set_valid_tokens(&stream, &mut buffers, NUM_TOKENS)
        .expect("valid_tokens");
    kernels
        .shared_expert(
            &stream,
            &mut buffers,
            QuantTensor {
                bytes: &d_gate,
                quant: quant_of(entries[0].info.ggml_type),
            },
            QuantTensor {
                bytes: &d_up,
                quant: quant_of(entries[1].info.ggml_type),
            },
            QuantTensor {
                bytes: &d_down,
                quant: quant_of(entries[2].info.ggml_type),
            },
            &d_hidden,
            &mut d_out,
        )
        .expect("shared expert");
    let device_out = stream.clone_dtoh(&d_out).expect("out back");
    stream.synchronize().expect("sync");

    let reference: Vec<f32> = hidden_states
        .iter()
        .flat_map(|x| expert_mlp(&host, x, g.hidden, g.intermediate))
        .collect();
    let candidate = &device_out[..NUM_TOKENS * g.hidden];
    assert_carries_signal("shared expert reference output", &reference);
    let result = compare(candidate, &reference);
    println!("shared expert vs expert_mlp, {NUM_TOKENS} tokens: {result}");
    println!(
        "output magnitude: max |ref| = {:.4e}, mean |ref| = {:.4e}",
        reference.iter().fold(0.0f32, |m, v| m.max(v.abs())),
        reference.iter().map(|v| v.abs()).sum::<f32>() / reference.len() as f32,
    );

    // Same evidence as the routed path: the element driving max_rel_error is
    // below `compare()`'s 1e-6 relative-error floor, so its ratio is an
    // artefact of the floor rather than a measurement.
    let driver = reference[result.max_rel_error_index].abs();
    assert!(
        driver < 1e-3,
        "max_rel_error {:.3e} sits on a reference value of {driver:.3e}, large \
         enough for the ratio to be meaningful — max_abs_error alone is no \
         longer a sufficient gate",
        result.max_rel_error,
    );

    assert_matches(candidate, &reference, &SHARED_GATE);
    assert!(
        device_out[NUM_TOKENS * g.hidden..]
            .iter()
            .all(|&v| v == 0.0),
        "the shared expert wrote past the live tokens",
    );
    println!("shared expert ran for every token with no dispatch table consulted");
}
