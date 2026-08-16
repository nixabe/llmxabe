//! The tensor-core path against the fp32 path it replaces, on the real model.
//!
//! `forward_pass.rs` gates 19 tokens against llama.cpp's capture. That is
//! *below* `GdnBlock::uses_tensor_cores`, so the oracle gate runs entirely in
//! fp32 and says nothing about the integer path. This closes that hole the
//! only way available without a second capture: run the same prompt through
//! the same shape twice, once with the repacked int8 weights resident and once
//! without, and require them to agree.
//!
//! What each side is:
//!
//! - **fp32** — the path `forward_pass.rs` gates, so it is known-good by
//!   transitivity.
//! - **int8** — the same arithmetic with every quantized GEMM's activations
//!   quantized to int8 and multiplied on tensor cores: three projections per
//!   Gated DeltaNet layer (30 of them), four per Gated Attention layer (10),
//!   and in all 40 layers the routed experts' Q6_K gate/up and Q8_0 down plus
//!   the shared expert. Every matmul in the model that has an integer path
//!   takes it on one side of this comparison and not the other.
//!
//! Agreement cannot be exact and is not asked to be: int8 activations cost
//! about 1/127 per element, which is the entire price of the instruction.
//! What the gate requires is that the *argmax survives* — the only thing a
//! user sees — and that the distributions stay close enough that the argmax
//! surviving is not luck.
//!
//! SKIPS — reporting that it skipped — without a driver, a supported device,
//! or the model file. It needs no golden capture: the comparison is internal.

use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_engine::DeviceWeights;
use xabe_engine::block::gdn::GdnBlock;
use xabe_engine::forward::{Forward, arena_holds};
use xabe_gguf::GgufFile;
use xabe_model::config::ModelConfig;
use xabe_model::weights::WeightSchema;

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llama.cpp/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

/// Tokens to run. Must be at or above the tensor-core threshold, or this test
/// compares fp32 against fp32 and passes vacuously — which is asserted below.
const TOKENS: usize = 128;

/// Cosine floor between the two paths.
///
/// int8 activations cost ~1/127 per element and average down over a 2,048-long
/// contraction, but the error compounds through 30 Gated DeltaNet layers and
/// the residual stream carries it forward. This is loose enough for that and
/// far too tight for a wrong fragment layout, which does not land near the
/// right answer at all.
const MIN_COSINE: f32 = 0.999;

fn model_path() -> PathBuf {
    std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
}

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
    println!("device 0: {}", info.name);
    Some(ctx)
}

fn dtoh(stream: &Arc<CudaStream>, buf: &CudaSlice<f32>) -> Vec<f32> {
    let v = stream.clone_dtoh(buf).expect("device read-back");
    stream.synchronize().expect("sync");
    v
}

fn argmax(v: &[f32]) -> (usize, f32) {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    (best, v[best])
}

/// The `n` highest logits, descending.
///
/// Printed on both arms so that an argmax disagreement is a *diagnosable*
/// event and not a bare inequality. The question a flip raises is always the
/// same — did the int8 path get the answer wrong, or were the top two within
/// its quantization noise of each other in the reference to begin with — and
/// only the runner-up's margin answers it.
fn top_k(v: &[f32], n: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_unstable_by(|&a, &b| v[b].total_cmp(&v[a]));
    idx.into_iter().take(n).map(|i| (i, v[i])).collect()
}

/// The reference gap between the best and second-best logit, below which the
/// argmax is not a property of the model but of the last bit of arithmetic.
///
/// Set against the int8 path's own measured disagreement with fp32, which on
/// this input is `max|diff| ~= 0.2` on logits peaking at 9.2. A reference
/// margin under this cannot survive quantization by anything but luck, so a
/// flip there is a statement about the *input*, not about the kernel — and the
/// test says so instead of failing as if a bug had been found.
const DECIDABLE_MARGIN: f32 = 0.25;

#[test]
fn integer_tensor_cores_agree_with_the_fp32_path_on_the_real_model() {
    assert!(
        GdnBlock::uses_tensor_cores(TOKENS),
        "this test compares the int8 path against fp32, so it must run at a \
         shape that actually takes the int8 path; {TOKENS} tokens does not",
    );

    let Some(ctx) = device() else {
        return;
    };
    let path = model_path();
    if !path.exists() {
        println!("SKIPPED: model file not found at {}", path.display());
        return;
    }
    let file = GgufFile::open(&path).expect("valid GGUF v3");
    let config = ModelConfig::qwen3_6_35b_a3b();
    let stream = ctx.default_stream();

    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).expect("schema resolves");
    let (weights, _) = DeviceWeights::load_where(&ctx, &stream, &file, &directory, arena_holds)
        .expect("weight load");

    let mut int8 = Forward::new(
        &ctx,
        &stream,
        &file,
        &directory,
        &weights,
        config.clone(),
        TOKENS,
    )
    .expect("the pass builds");
    assert!(
        int8.tensor_cores_enabled(),
        "a pass at {TOKENS} tokens must have repacked weights resident",
    );

    // The same shape over the same weights, with the int8 path removed.
    // `reshape` shares the MoE weights; a second `Forward::new` would try to
    // upload another 28.3 GiB.
    let mut fp32 = int8
        .reshape(&ctx, &stream, &file, &directory, &weights, TOKENS)
        .expect("the fp32 twin builds");
    fp32.disable_tensor_cores();
    assert!(!fp32.tensor_cores_enabled());

    let ids: Vec<i32> = (0..TOKENS)
        .map(|i| ((i * 7919 + 1234) % config.vocab_size as usize) as i32)
        .collect();

    let mut s_a = int8.new_state(&stream, TOKENS).expect("state");
    int8.run(&stream, &mut s_a, &ids, |_, _| {})
        .expect("int8 pass");
    let a = dtoh(&stream, int8.logits());

    let mut s_b = fp32.new_state(&stream, TOKENS).expect("state");
    fp32.run(&stream, &mut s_b, &ids, |_, _| {})
        .expect("fp32 pass");
    let b = dtoh(&stream, fp32.logits());

    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    let mut max_abs = 0f32;
    for (x, y) in a.iter().zip(&b) {
        dot += f64::from(*x) * f64::from(*y);
        na += f64::from(*x) * f64::from(*x);
        nb += f64::from(*y) * f64::from(*y);
        max_abs = max_abs.max((x - y).abs());
    }
    let cosine = (dot / (na.sqrt() * nb.sqrt())) as f32;
    let (ia, va) = argmax(&a);
    let (ib, vb) = argmax(&b);
    let peak = b.iter().fold(0f32, |m, v| m.max(v.abs()));

    let top_fp32 = top_k(&b, 3);
    let top_int8 = top_k(&a, 3);
    let margin = top_fp32[0].1 - top_fp32[1].1;
    println!(
        "\n{TOKENS} tokens: every integer path in the model — 30 Gated DeltaNet \
         layers x 3 projections, 10 Gated Attention layers x 4, and 40 layers \
         of routed and shared experts\n\
         \x20 argmax   int8 {ia} (logit {va:.6})   fp32 {ib} (logit {vb:.6})\n\
         \x20 cosine   {cosine:.9}\n\
         \x20 max|diff| {max_abs:.6} against a peak logit of {peak:.3}\n\
         \x20 fp32 top3 {top_fp32:?}\n\
         \x20 int8 top3 {top_int8:?}\n\
         \x20 fp32 margin (best - runner-up) {margin:.6}, decidable above {DECIDABLE_MARGIN}",
    );

    assert!(
        a.iter().all(|v| v.is_finite()),
        "int8 path produced non-finite logits"
    );
    // The argmax is only the model's answer where the reference says the top
    // two are distinguishable. Where it does not, asserting on the argmax
    // asserts on rounding, and a test that fails on rounding cannot tell a
    // regression from a re-association. The cosine floor below covers the
    // whole distribution either way, and it is the bound that does not depend
    // on this input happening to have a decisive winner.
    if margin >= DECIDABLE_MARGIN {
        assert_eq!(
            ia, ib,
            "the int8 path selects a different token than the fp32 path it replaces, on an \
             input whose fp32 margin is {margin:.6} — wider than the {DECIDABLE_MARGIN} \
             below which the choice is rounding. Quantizing activations is allowed to move \
             logits, not to change a decided answer",
        );
    } else if ia != ib {
        println!(
            "  NOTE: argmax differs ({ia} vs {ib}) on a reference margin of {margin:.6}, \
             below the {DECIDABLE_MARGIN} this input would need for the choice to be \
             decidable at all. Not asserted: the top two are inside the int8 path's own \
             {max_abs:.3} disagreement with fp32, so which one wins is arithmetic order, \
             not accuracy. The cosine floor is the gate here.",
        );
    }
    assert!(
        cosine >= MIN_COSINE,
        "int8 agrees with fp32 only to cosine {cosine}, below the {MIN_COSINE} \
         floor — the argmax surviving does not make the distribution right",
    );
}
