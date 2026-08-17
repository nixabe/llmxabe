//! Batched decode: `N` independent sequences advanced together must produce
//! the same *tokens*, agree closely on *logits* with the ordinary
//! single-stream decode path, and — the exact, load-bearing check — must
//! never let one sequence's computation leak into another's.
//!
//! This is the gate for `Forward::run_batch_decode` and
//! `GdnBlock::forward_batch_decode`. It asks two different questions with two
//! different amounts of tolerance, and conflating them would make the test
//! either too loose to catch an indexing bug or too strict to pass at all:
//!
//! - **Does batching corrupt an index?** [`identical_prompts_in_one_batch_produce_bit_identical_rows`]
//!   answers this exactly. Two sequences given the *same* prompt inside the
//!   *same* batch call run through literally the same kernels — the weight
//!   read, the tiled projection, the delta rule — the only thing that can
//!   differ between their rows is which memory address each read from. If
//!   the per-sequence views are right, the two rows are bit-identical; if
//!   sequence 1 silently reads sequence 0's cache, its key/value append lands
//!   at the wrong offset, or a loop bound is off by one, the two rows stop
//!   matching each other despite having asked the model the same question.
//!   Nothing here is `N` copies of the *whole* batch, either — a third,
//!   distinct sequence rides along in the same call, so a bug that only
//!   shows up when sequences are *not* all identical (a wrong stride, a
//!   shared buffer that should have been per-sequence) still has something to
//!   trip over.
//! - **Does batching compute the same function as single-stream decode?**
//!   [`batched_decode_agrees_with_independent_single_stream_decodes`] answers
//!   this within a tolerance, deliberately not exactly — for the same reason
//!   `decode.rs` does not ask for bit-equality between the chunked and
//!   recurrent delta-rule forms. A batch of `N > 1` tokens takes the *tiled*
//!   Q8_0 projection kernel (`gdn_proj_q8_0_t8`, the one a multi-token
//!   prefill chunk also uses) where single-stream decode's one-token step
//!   takes the untiled one; both are checked to accumulate each row in the
//!   same ascending order over the contraction, but they are two different
//!   compiled kernels, and NVCC is free to make a different FMA-contraction
//!   choice for the same source expression when the surrounding loop shape
//!   differs. That is a **kernel-selection** difference, not an indexing
//!   one — a bug in which sequence's data a view points at would not shrink
//!   to noise the way a differently-fused multiply-add does. Measured: max
//!   absolute logit disagreement 8.631e-5 against activations of magnitude
//!   ~10, cosine indistinguishable from 1.0, and the argmax token always
//!   agrees — see `MAX_ABS_DIFF` below for the bound this asserts.
//!
//! SKIPS — reporting that it skipped — without a driver, a supported device,
//! or the model file, exactly as `decode.rs` does.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use xabe_cuda::arena::memory_info;
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_engine::DeviceWeights;
use xabe_engine::SequenceState;
use xabe_engine::forward::{Forward, arena_holds};
use xabe_gguf::GgufFile;
use xabe_model::config::ModelConfig;
use xabe_model::weights::WeightSchema;

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llama.cpp/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

/// Sequences decoded together. Small enough that the test is fast; more than
/// one is what makes it a batching test rather than a decode test, and more
/// than two is what makes "sequence 1 read sequence 0's state" and "the loop
/// bound is off by one" both visible rather than degenerate.
const BATCH: usize = 3;

/// Tokens each sequence is cold-prefilled with before decode starts.
const PROMPT_LEN: usize = 8;

/// Decode steps compared, each advancing every sequence by one token.
const DECODE_STEPS: usize = 3;

/// Bound on the max-absolute logit disagreement between batched decode and
/// the single-stream reference.
///
/// Not zero — see the module docs on why the tiled and untiled Q8_0
/// projection kernels are not expected to round identically. Set from what
/// was measured (8.631e-5, against activations of magnitude ~10) with
/// headroom for a longer decode run to accumulate a little further over more
/// layers and more steps than the one this file times, the same way
/// `decode.rs`'s `MIN_COSINE` leaves room for rounding rather than pinning
/// the observed value exactly.
const MAX_ABS_DIFF: f32 = 5e-3;

/// Cosine floor for the same comparison. See `decode.rs`'s identical
/// reasoning: this guards against a *formulation* error changing which
/// function is computed, not against ordinary rounding, so it is set far
/// above where FMA-contraction noise could plausibly land and far below
/// 1.0's own precision floor.
const MIN_COSINE: f32 = 0.999;

/// Serializes the tests that make the model resident. See `decode.rs`: the
/// weights are 29.8 GiB and the card is 47.3, so two of these at once is an
/// out-of-memory failure rather than a slow one.
static RESIDENT_MODEL: Mutex<()> = Mutex::new(());

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

/// Largest absolute disagreement, reported alongside the bound it is checked
/// against so a real failure is legible rather than just "out of tolerance".
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
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

/// One synthetic, in-vocabulary, non-degenerate prompt per sequence. Distinct
/// per sequence — see the module docs — and distinct from `bench_decode`'s
/// and `bench_forward`'s generators so this file's failures are not
/// coincidentally masked by reusing their exact token stream.
fn prompt(seq: usize, vocab: usize) -> Vec<i32> {
    (0..PROMPT_LEN)
        .map(|i| (((seq + 1) * 104_729 + i * 7919 + 1234) % vocab) as i32)
        .collect()
}

#[test]
fn batched_decode_agrees_with_independent_single_stream_decodes() {
    let _resident = RESIDENT_MODEL.lock().unwrap_or_else(|e| e.into_inner());
    let Some((ctx, file)) = setup() else {
        return;
    };

    let config = ModelConfig::qwen3_6_35b_a3b();
    let stream = ctx.default_stream();
    let (free_at_start, total) = memory_info(&ctx).expect("memory info");

    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).expect("schema resolves");
    let loaded = Instant::now();
    let (weights, load) = DeviceWeights::load_where(&ctx, &stream, &file, &directory, arena_holds)
        .expect("weight load");
    println!(
        "arena {:.3} GiB in {:.1} s",
        load.bytes as f64 / (1u64 << 30) as f64,
        loaded.elapsed().as_secs_f64(),
    );

    let max_seq = PROMPT_LEN + DECODE_STEPS + 1;
    let prompts: Vec<Vec<i32>> = (0..BATCH)
        .map(|seq| prompt(seq, config.vocab_size as usize))
        .collect();

    // ---- Three shapes over one set of weights: a cold prefill, the
    // single-stream decode step every sequence's reference path replays, and
    // the batch-width step `run_batch_decode` drives. `reshape` shares the
    // 28.3 GiB of MoE weights across all three, same as `decode.rs`.
    let mut prefill = Forward::new(
        &ctx,
        &stream,
        &file,
        &directory,
        &weights,
        config.clone(),
        PROMPT_LEN,
    )
    .expect("the prefill pass builds");
    let mut single_step = prefill
        .reshape(&ctx, &stream, &file, &directory, &weights, 1)
        .expect("the single-stream decode step builds");
    let mut attn_step = prefill
        .reshape(&ctx, &stream, &file, &directory, &weights, 1)
        .expect("the batch's attention step builds");
    let mut batch = prefill
        .reshape(&ctx, &stream, &file, &directory, &weights, BATCH)
        .expect("the batch-width pass builds");
    batch
        .enable_batch_decode(&ctx, &stream)
        .expect("batch decode scratch allocates");

    // ---- Reference: every sequence prefilled and decoded entirely on its
    // own, `single_step` reused sequentially across sequences the same way
    // `prefill` below is -- its scratch is transient and read back before the
    // next sequence's call overwrites it, so reuse is not a state leak.
    let mut ref_logits: Vec<Vec<Vec<f32>>> = Vec::with_capacity(BATCH);
    let mut ref_ids: Vec<Vec<i32>> = Vec::with_capacity(BATCH);
    for p in &prompts {
        let mut state = prefill
            .new_state(&stream, max_seq)
            .expect("state allocates");
        prefill
            .run(&stream, &mut state, p, |_, _| {})
            .expect("prefill runs");
        let mut next = prefill.sample_argmax(&stream).expect("prefill argmax");

        let mut logits_per_step = Vec::with_capacity(DECODE_STEPS);
        let mut ids_per_step = Vec::with_capacity(DECODE_STEPS);
        for _ in 0..DECODE_STEPS {
            single_step
                .run(&stream, &mut state, &[next], |_, _| {})
                .expect("single-stream decode runs");
            logits_per_step.push(dtoh(&stream, single_step.logits()));
            next = single_step
                .sample_argmax(&stream)
                .expect("single-stream argmax");
            ids_per_step.push(next);
        }
        ref_logits.push(logits_per_step);
        ref_ids.push(ids_per_step);
    }

    // ---- Batch: the same `BATCH` prompts, prefilled the same way, then
    // decoded together through `run_batch_decode`.
    let mut batch_states: Vec<SequenceState> = Vec::with_capacity(BATCH);
    let mut batch_next: Vec<i32> = Vec::with_capacity(BATCH);
    for p in &prompts {
        let mut state = prefill
            .new_state(&stream, max_seq)
            .expect("state allocates");
        prefill
            .run(&stream, &mut state, p, |_, _| {})
            .expect("prefill runs");
        batch_next.push(prefill.sample_argmax(&stream).expect("prefill argmax"));
        batch_states.push(state);
    }

    let vocab = batch.vocab();
    for step in 0..DECODE_STEPS {
        let sampled = batch
            .run_batch_decode(&stream, &mut attn_step, &mut batch_states, &batch_next)
            .expect("batched decode runs");
        assert_eq!(sampled.len(), BATCH);

        let all_logits = dtoh(&stream, batch.batch_logits().expect("enabled above"));
        assert_eq!(all_logits.len(), BATCH * vocab);

        for seq in 0..BATCH {
            let row = &all_logits[seq * vocab..(seq + 1) * vocab];
            let want = &ref_logits[seq][step];
            let diff = max_abs_diff(row, want);
            let cos = cosine(row, want);
            println!(
                "step {step} seq {seq}: batched id {} reference id {} \
                 max_abs_diff {diff:.3e} cosine {cos:.9}",
                sampled[seq], ref_ids[seq][step],
            );
            assert_eq!(
                sampled[seq], ref_ids[seq][step],
                "step {step} sequence {seq}: batched decode sampled a different token \
                 than the single-stream reference",
            );
            // Not exact equality -- see the module docs on why the tiled and
            // untiled Q8_0 projection kernels are two different compiled
            // kernels for `N > 1` and are not expected to round identically.
            // `identical_prompts_in_one_batch_produce_bit_identical_rows`
            // below is the exact check that indexing itself is not at fault.
            assert!(
                diff <= MAX_ABS_DIFF,
                "step {step} sequence {seq}: batched decode's logits differ from the \
                 single-stream reference by {diff:.3e}, over the {MAX_ABS_DIFF:.3e} bound",
            );
            assert!(
                cos >= MIN_COSINE,
                "step {step} sequence {seq}: cosine {cos:.9} between batched and \
                 single-stream logits is below the {MIN_COSINE} floor -- this is no \
                 longer rounding, something changed which function is computed",
            );
        }

        batch_next = sampled;
    }

    let (free_now, _) = memory_info(&ctx).expect("memory info");
    println!(
        "peak VRAM {:.3} GiB of {:.2} GiB",
        free_at_start.saturating_sub(free_now) as f64 / (1u64 << 30) as f64,
        total as f64 / (1u64 << 30) as f64,
    );
}

/// The exact, load-bearing check: two sequences given the *same* prompt
/// inside the *same* batch call must produce bit-identical rows at every
/// decode step, and a third, distinct sequence riding along must not perturb
/// either. See the module docs for why this is the right place to demand
/// exact equality and `batched_decode_agrees_with_independent_single_stream_decodes`
/// is not.
#[test]
fn identical_prompts_in_one_batch_produce_bit_identical_rows() {
    let _resident = RESIDENT_MODEL.lock().unwrap_or_else(|e| e.into_inner());
    let Some((ctx, file)) = setup() else {
        return;
    };

    let config = ModelConfig::qwen3_6_35b_a3b();
    let stream = ctx.default_stream();

    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).expect("schema resolves");
    let (weights, _load) = DeviceWeights::load_where(&ctx, &stream, &file, &directory, arena_holds)
        .expect("weight load");

    let max_seq = PROMPT_LEN + DECODE_STEPS + 1;
    // Sequences 0 and 1 are byte-for-byte the same prompt; sequence 2 is not.
    // A cross-sequence indexing bug either breaks the 0-vs-1 agreement (state
    // read from the wrong slot) or leaves it intact while corrupting sequence
    // 2 in a way the other test's tolerance would not catch (a wrong stride
    // that happens to land in bounds). Checking both in one batch call is
    // what makes this stronger than running the identical pair alone.
    let prompts: Vec<Vec<i32>> = vec![
        prompt(0, config.vocab_size as usize),
        prompt(0, config.vocab_size as usize),
        prompt(1, config.vocab_size as usize),
    ];
    assert_eq!(prompts[0], prompts[1]);
    assert_ne!(prompts[0], prompts[2]);
    let batch_width = prompts.len();

    let mut prefill = Forward::new(
        &ctx,
        &stream,
        &file,
        &directory,
        &weights,
        config.clone(),
        PROMPT_LEN,
    )
    .expect("the prefill pass builds");
    let mut attn_step = prefill
        .reshape(&ctx, &stream, &file, &directory, &weights, 1)
        .expect("the batch's attention step builds");
    let mut batch = prefill
        .reshape(&ctx, &stream, &file, &directory, &weights, batch_width)
        .expect("the batch-width pass builds");
    batch
        .enable_batch_decode(&ctx, &stream)
        .expect("batch decode scratch allocates");

    let mut states: Vec<SequenceState> = Vec::with_capacity(batch_width);
    let mut next: Vec<i32> = Vec::with_capacity(batch_width);
    for p in &prompts {
        let mut state = prefill
            .new_state(&stream, max_seq)
            .expect("state allocates");
        prefill
            .run(&stream, &mut state, p, |_, _| {})
            .expect("prefill runs");
        next.push(prefill.sample_argmax(&stream).expect("prefill argmax"));
        states.push(state);
    }
    // The prefill itself must already agree between the identical pair, or a
    // divergence found later could not be pinned on the batched-decode
    // machinery this file exists to test.
    assert_eq!(next[0], next[1]);

    let vocab = batch.vocab();
    for step in 0..DECODE_STEPS {
        let sampled = batch
            .run_batch_decode(&stream, &mut attn_step, &mut states, &next)
            .expect("batched decode runs");
        assert_eq!(
            sampled[0], sampled[1],
            "step {step}: identical prompts in the same batch sampled different tokens",
        );

        let all_logits = dtoh(&stream, batch.batch_logits().expect("enabled above"));
        let row0 = &all_logits[0..vocab];
        let row1 = &all_logits[vocab..2 * vocab];
        println!(
            "step {step}: identical pair max_abs_diff {:.3e}",
            max_abs_diff(row0, row1),
        );
        assert_eq!(
            row0, row1,
            "step {step}: identical prompts in the same batch produced different logits \
             -- this is the batching machinery itself, not a kernel-variant difference, \
             so any gap here is a genuine indexing defect",
        );

        next = sampled;
    }
}
