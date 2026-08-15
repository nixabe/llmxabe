//! Autoregressive decode: the incremental path must equal the batch path.
//!
//! `forward_pass.rs` gates one 19-token prefill against llama.cpp's capture,
//! which says the model is right when it sees the whole prompt at once. It
//! says nothing about decode, because the capture holds no decode step —
//! `docs/ORACLE.md` section 9 lists what a second capture would need to cover.
//!
//! This file gates decode without one, using the model against itself:
//!
//! ```text
//!   A:  prefill all 19 tokens                       -> logits at position 18
//!   B:  prefill 18, then decode token 18            -> logits at position 18
//!   C:  prefill 12, then decode tokens 12..18       -> logits at position 18
//! ```
//!
//! All three predict the token after the prompt, so all three must agree. That
//! is a real gate rather than a tautology, because B and C reach it through
//! machinery A never touches:
//!
//! - **The KV cache.** In A every key and value is computed in the same batch.
//!   In B and C the last query attends to keys written by an *earlier call*,
//!   so a wrong write offset, a wrong `key_offset`, or a cache that is not
//!   actually read makes B and C disagree with A.
//! - **The rotary offset.** A rotates the whole batch from position 0. C
//!   rotates its decode steps from 12, 13, ... 17. Rotating a decode step from
//!   0 produces finite, plausible, wrong attention — and it is invisible in a
//!   single-step test where position 0 and the offset happen to coincide,
//!   which is why C exists alongside B.
//! - **The Gated DeltaNet handoff.** A runs 30 layers of the *chunked* form
//!   over 19 tokens. B and C run the chunked form over a prefix and then the
//!   *recurrent* form per step, resuming the carried matrix and convolution
//!   window. These are different algorithms over the same recurrence, and this
//!   is the only test in the workspace that runs one into the other.
//!
//! # What agreement is expected, and why not bit-equality
//!
//! Not bit-identical, and the gate does not ask for it. The chunked and
//! recurrent forms sum the same terms in a different order and the MoE's
//! grouped GEMM tiles differently at 19 rows than at 1, so the two paths round
//! differently at every one of 40 blocks. The argmax is the gate that matters
//! and it is exact; the cosine floor is there so a *near*-miss that happens to
//! keep the argmax still fails.
//!
//! SKIPS — reporting that it skipped — without a driver, a supported device,
//! or the model file. It does **not** need the golden capture: the comparison
//! is internal. `GOLDEN_TOKENS` and `GOLDEN_ARGMAX` are used as the prompt and
//! the expected answer because a prompt with a known answer makes a failure
//! legible, not because the capture is read.

#[path = "golden.rs"]
mod golden;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use xabe_cuda::arena::memory_info;
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_engine::DeviceWeights;
use xabe_engine::forward::{Forward, arena_holds};
use xabe_gguf::GgufFile;
use xabe_model::config::ModelConfig;
use xabe_model::weights::WeightSchema;

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llama.cpp/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

/// Cosine floor between the batch path and an incremental one.
///
/// The two paths differ by summation order in the delta rule and by tile shape
/// in the MoE, compounded over 40 blocks. Measured agreement is reported by
/// the test itself, and the observed value is 1.000000000 with a max absolute
/// disagreement of 4.0e-6 — three orders of magnitude inside this bound.
///
/// The floor is set here rather than at the observed value because it is
/// guarding against a *formulation* error, not against drift: a wrong write
/// offset or an unrotated key does not nudge the cosine, it changes which
/// function is being computed. How far below this such a mistake would land
/// has not been measured, so this is a bound chosen to leave room for
/// rounding, not one calibrated against a known-bad run.
const MIN_COSINE: f32 = 0.999;

/// Where the 19-token prompt is split for the multi-step case.
///
/// 12 leaves 7 decode steps, which is enough that a position offset that
/// increments wrongly has diverged visibly by the end, and short enough that
/// the test stays a test rather than a benchmark.
const SPLIT: usize = 12;

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
    println!(
        "device 0: {} sm_{}{}, {:.1} GiB",
        info.name,
        info.compute_capability.major,
        info.compute_capability.minor,
        info.total_memory as f64 / (1u64 << 30) as f64,
    );
    Some(ctx)
}

fn setup() -> Option<(Arc<CudaContext>, GgufFile)> {
    let ctx = device()?;
    let path = model_path();
    if !path.exists() {
        println!(
            "SKIPPED: model file not found at {}; set LLMXABE_MODEL to override",
            path.display(),
        );
        return None;
    }
    Some((ctx, GgufFile::open(&path).expect("valid GGUF v3")))
}

fn dtoh(stream: &Arc<CudaStream>, buf: &CudaSlice<f32>) -> Vec<f32> {
    let v = stream.clone_dtoh(buf).expect("device read-back");
    stream.synchronize().expect("sync");
    v
}

/// Index and value of the largest element.
fn argmax(v: &[f32]) -> (usize, f32) {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    (best, v[best])
}

/// Cosine similarity, in f64 so the accumulation is not itself the error.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..a.len() {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// Largest absolute disagreement.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn decoding_one_token_at_a_time_agrees_with_prefilling_the_whole_prompt() {
    let Some((ctx, file)) = setup() else {
        return;
    };
    let config = ModelConfig::qwen3_6_35b_a3b();
    let stream = ctx.default_stream();
    let prompt = golden::GOLDEN_TOKENS;
    let tokens = prompt.len();
    assert!(SPLIT < tokens && SPLIT > 0);

    let (free_at_start, total) = memory_info(&ctx).expect("memory info");

    // ---- 1. The model, resident, once. --------------------------------
    let schema = WeightSchema::new(&config);
    let directory = schema
        .resolve(&file)
        .expect("the schema must resolve against the model file");
    let loaded = Instant::now();
    let (weights, load) = DeviceWeights::load_where(&ctx, &stream, &file, &directory, arena_holds)
        .expect("weight load");
    println!(
        "\n=== load ===\narena: {:.3} GiB in {:.1} s",
        load.bytes as f64 / (1u64 << 30) as f64,
        loaded.elapsed().as_secs_f64(),
    );

    // ---- 2. Four shapes over one set of weights. ----------------------
    //
    // `reshape` shares the 28.3 GiB of MoE weights by reference count. Four
    // independent `Forward::new` calls would try to upload them four times and
    // exhaust the card on the second.
    let built = Instant::now();
    let mut full = Forward::new(
        &ctx,
        &stream,
        &file,
        &directory,
        &weights,
        config.clone(),
        tokens,
    )
    .expect("the 19-token pass builds");
    let mut short = full
        .reshape(&ctx, &stream, &file, &directory, &weights, tokens - 1)
        .expect("the 18-token pass builds");
    let mut head = full
        .reshape(&ctx, &stream, &file, &directory, &weights, SPLIT)
        .expect("the split-prefix pass builds");
    let mut step = full
        .reshape(&ctx, &stream, &file, &directory, &weights, 1)
        .expect("the single-token pass builds");

    let (free_after_build, _) = memory_info(&ctx).expect("memory info");
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    println!(
        "built 4 shapes ({tokens}, {}, {SPLIT}, 1) in {:.1} s\n\
         \x20 MoE weights uploaded once      {:8.3} GiB\n\
         \x20 the other three report          {:8.3} GiB  <- shared, not re-uploaded\n\
         \x20 PEAK VRAM                      {:8.3} GiB of {:.2} GiB",
        tokens - 1,
        built.elapsed().as_secs_f64(),
        gib(full.report().moe_bytes),
        gib(step.report().moe_bytes),
        gib(free_at_start.saturating_sub(free_after_build)),
        gib(total),
    );
    assert_eq!(
        step.report().moe_bytes,
        0,
        "a reshaped pass must not claim the weights it borrowed",
    );

    // ---- A. The whole prompt in one pass. -----------------------------
    let mut state = full
        .new_state(&stream, tokens)
        .expect("sequence state allocates");
    println!(
        "\n=== state ===\n{:.3} GiB for {tokens} positions ({} GDN, {} attention layers)",
        state.bytes() as f64 / (1u64 << 30) as f64,
        state.gdn_layers(),
        state.attention_layers(),
    );

    full.run(&stream, &mut state, &prompt, |_, _| {})
        .expect("batch prefill runs");
    let batch = dtoh(&stream, full.logits());
    assert_eq!(
        state.position(),
        tokens,
        "a {tokens}-token pass must leave the state at position {tokens}",
    );
    let (batch_id, batch_logit) = argmax(&batch);

    // ---- B. Prefill all but the last token, then decode it. -----------
    state.reset(&stream).expect("back to a cold start");
    assert_eq!(state.position(), 0);
    short
        .run(&stream, &mut state, &prompt[..tokens - 1], |_, _| {})
        .expect("18-token prefill runs");
    assert_eq!(state.position(), tokens - 1);
    step.run(&stream, &mut state, &prompt[tokens - 1..], |_, _| {})
        .expect("one decode step runs");
    assert_eq!(state.position(), tokens);
    let one_step = dtoh(&stream, step.logits());

    // ---- C. Prefill a short prefix, then decode the rest. -------------
    state.reset(&stream).expect("back to a cold start");
    head.run(&stream, &mut state, &prompt[..SPLIT], |_, _| {})
        .expect("prefix prefill runs");
    let decode_started = Instant::now();
    for (n, id) in prompt[SPLIT..].iter().enumerate() {
        assert_eq!(
            state.position(),
            SPLIT + n,
            "the state must advance exactly one position per decode step",
        );
        step.run(&stream, &mut state, std::slice::from_ref(id), |_, _| {})
            .expect("decode step runs");
    }
    stream.synchronize().expect("sync");
    let decode_wall = decode_started.elapsed();
    assert_eq!(state.position(), tokens);
    let many_steps = dtoh(&stream, step.logits());

    // ---- The gate. ----------------------------------------------------
    println!(
        "\n=== incremental vs batch, logits at position {} ===\n\
         {:<34} {:>12} {:>14} {:>10}\n\
         {:<34} {:>12} {:>14} {:>10}\n\
         {:<34} {:>12} {:>14} {:>10}",
        tokens - 1,
        "path",
        "argmax",
        "cosine vs A",
        "max|diff|",
        format!("A: prefill {tokens}"),
        batch_id,
        1.0f32,
        0.0f32,
        format!("B: prefill {} + 1 decode", tokens - 1),
        argmax(&one_step).0,
        cosine(&batch, &one_step),
        max_abs_diff(&batch, &one_step),
    );
    println!(
        "{:<34} {:>12} {:>14.9} {:>10.6}",
        format!("C: prefill {SPLIT} + {} decodes", tokens - SPLIT),
        argmax(&many_steps).0,
        cosine(&batch, &many_steps),
        max_abs_diff(&batch, &many_steps),
    );
    // Seven steps with no warmup, whatever profile this test was built in.
    // It is here so a catastrophic regression is visible from the gate; the
    // throughput claim lives in `bench_decode`, which warms up and repeats.
    println!(
        "\n{} decode steps in {:.1} ms — {:.2} ms/token, {:.2} tok/s \
         (no warmup, 7 samples; not a performance claim)",
        tokens - SPLIT,
        decode_wall.as_secs_f64() * 1e3,
        decode_wall.as_secs_f64() * 1e3 / (tokens - SPLIT) as f64,
        (tokens - SPLIT) as f64 / decode_wall.as_secs_f64(),
    );

    // The argmax is the only thing here a user would ever notice, and it is
    // exact. It is checked against the batch path *and* against llama.cpp's
    // own answer, so a decode that agrees with a broken prefill still fails.
    assert_eq!(
        batch_id,
        golden::GOLDEN_ARGMAX,
        "the batch path no longer reproduces llama.cpp's token; \
         fix forward_pass.rs before reading anything below",
    );
    for (name, logits) in [
        ("B (one decode step)", &one_step),
        ("C (7 decodes)", &many_steps),
    ] {
        let (id, logit) = argmax(logits);
        assert_eq!(
            id, batch_id,
            "{name} selected token {id} (logit {logit}) where the batch path \
             selected {batch_id} (logit {batch_logit})",
        );
        let c = cosine(&batch, logits);
        assert!(
            c >= MIN_COSINE,
            "{name} agrees with the batch path only to cosine {c}, below the \
             {MIN_COSINE} floor — the argmax surviving does not make the \
             distribution right",
        );
    }
}

#[test]
fn a_decode_step_cannot_run_past_the_end_of_its_cache() {
    // The failure this guards is silent if it is allowed to wrap: writing at
    // `position % max_seq` would answer the next query from a different
    // prompt's keys. Checked without a device, on the error type itself, so it
    // runs everywhere.
    use xabe_engine::block::attention::AttentionBlockError;
    let e = AttentionBlockError::CacheExhausted {
        position: 4096,
        tokens: 8,
        max_seq: 4096,
    };
    let m = e.to_string();
    assert!(m.contains("4104"), "must name the slot it needed: {m}");
    assert!(m.contains("4096"), "must name the cache it has: {m}");
}
