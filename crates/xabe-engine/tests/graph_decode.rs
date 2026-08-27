//! A captured decode step must generate exactly the sequence the ordinary
//! launch path generates.
//!
//! [`Forward::capture_step`] records one token's worth of work — the
//! embedding gather, forty blocks, the final norm, the LM head and the argmax
//! — as a CUDA graph, and [`Forward::replay_step`] launches it again at every
//! later position. That is worth 3.8 ms of a 10.6 ms step, and it is only
//! correct if the recorded launches contain **nothing that changes between
//! steps**. A recorded launch keeps the arguments it was recorded with, so a
//! single host-side position left anywhere on the path would make every
//! replayed step rotate by, attend up to, and append at the position that was
//! captured — and the model would still emit fluent, entirely finite,
//! completely wrong tokens. No tolerance catches that; only a comparison
//! against the path that does not have the problem does.
//!
//! So this test runs the same prompt twice over the same weights:
//!
//! | | how it decodes |
//! | --- | --- |
//! | reference | [`Forward::run`] then [`Forward::sample_argmax`], per step |
//! | candidate | [`Forward::capture_step`] once, then [`Forward::replay_step`] |
//!
//! and requires the two to agree **bit for bit**. Not within a tolerance:
//! the replay issues the identical kernels with the identical arguments over
//! the identical buffers, so anything other than equality means the graph is
//! not the pass.
//!
//! ## What each step of the sequence proves
//!
//! One step would not be enough. A graph captured at position `p` and
//! replayed once at position `p` is trivially right, and that is exactly the
//! bug this is looking for. The interesting evidence is in the steps after
//! the first:
//!
//! - **the rotary embedding** rotates token `i` by absolute position, so a
//!   frozen position makes every generated token carry the first one's angle;
//! - **the causal bound** is `position + 1` keys, so a frozen position makes
//!   every later step attend to a window that stops growing;
//! - **the key/value append** writes at `position * row`, so a frozen
//!   position makes every later step overwrite the same cache slot.
//!
//! All three are wrong in the same direction and all three are invisible in
//! the output's *shape*. Requiring the whole generated sequence to match is
//! what separates them from a working capture.
//!
//! SKIPS — reporting that it skipped — without a driver, a supported device,
//! or the model file.

use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_engine::DeviceWeights;
use xabe_engine::forward::{Forward, arena_holds};
use xabe_gguf::GgufFile;
use xabe_model::config::ModelConfig;
use xabe_model::weights::WeightSchema;

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llmxabe/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

/// Prompt length. Short: the point is the decode steps, not the prefill.
const PROMPT: usize = 24;

/// Decode steps compared.
///
/// Eight, because the failure this is looking for is invisible at one and
/// obvious by the second — and because a frozen key/value append would
/// overwrite the same slot eight times, which is enough for the attention
/// window to be visibly wrong rather than marginally so.
const STEPS: usize = 8;

fn model_path() -> PathBuf {
    std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
}

fn dtoh(stream: &Arc<CudaStream>, buf: &CudaSlice<f32>) -> Vec<f32> {
    let v = stream.clone_dtoh(buf).expect("device read-back");
    stream.synchronize().expect("sync");
    v
}

fn main_or_skip() -> Option<(Arc<CudaContext>, GgufFile)> {
    if !driver_available() {
        println!("SKIPPED: no CUDA driver on this host");
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
    let path = model_path();
    if !path.exists() {
        println!("SKIPPED: model file not found at {}", path.display());
        return None;
    }
    Some((ctx, GgufFile::open(&path).expect("valid GGUF v3")))
}

#[test]
fn a_captured_step_generates_the_same_sequence_as_the_launch_path() {
    let Some((ctx, file)) = main_or_skip() else {
        return;
    };
    let config = ModelConfig::qwen3_6_35b_a3b();

    // Capture is rejected on the legacy default stream, so the whole test
    // runs on one created stream. Creating it puts cudarc into multi-stream
    // mode, where every slice records an event per use and later uses wait on
    // it; a wait on an event recorded outside a capture is
    // `CUDA_ERROR_STREAM_CAPTURE_ISOLATION`.
    //
    // SAFETY: one stream, so there is no cross-stream ordering for those
    // events to enforce. It has to happen before the first allocation,
    // because only slices created afterwards are untracked.
    let stream = ctx.new_stream().expect("create stream");
    unsafe { ctx.disable_event_tracking() };

    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).expect("schema resolves");
    let (weights, _) = DeviceWeights::load_where(&ctx, &stream, &file, &directory, arena_holds)
        .expect("weight load");

    let mut prefill = Forward::new(
        &ctx,
        &stream,
        &file,
        &directory,
        &weights,
        config.clone(),
        PROMPT,
    )
    .expect("the prefill shape builds");
    let mut step = prefill
        .reshape(&ctx, &stream, &file, &directory, &weights, 1)
        .expect("the single-token shape builds");

    let prompt: Vec<i32> = (0..PROMPT)
        .map(|i| ((i * 7919 + 1234) % config.vocab_size as usize) as i32)
        .collect();
    let mut state = prefill
        .new_state(&stream, PROMPT + STEPS)
        .expect("sequence state allocates");

    // ---- reference: one ordinary launch sequence per step ----------------
    prefill
        .run(&stream, &mut state, &prompt, |_, _| {})
        .expect("prefill runs");
    let mut token = prefill.sample_argmax(&stream).expect("prefill argmax");
    let mut want_ids = Vec::with_capacity(STEPS);
    let mut want_logits = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        step.run(&stream, &mut state, &[token], |_, _| {})
            .expect("decode step runs");
        token = step.sample_argmax(&stream).expect("decode argmax");
        want_ids.push(token);
        want_logits.push(dtoh(&stream, step.logits()));
    }
    let end_position = state.position();

    // ---- candidate: the same steps, replayed from one capture ------------
    //
    // The state is reset to a cold start and the prompt is prefilled again,
    // so the replayed run begins where the reference run began rather than
    // continuing it.
    state.reset(&stream).expect("state resets");
    prefill
        .run(&stream, &mut state, &prompt, |_, _| {})
        .expect("prefill runs again");
    let mut token = prefill.sample_argmax(&stream).expect("prefill argmax");

    let graph = step
        .capture_step(&stream, &mut state)
        .expect("a decode step captures");
    assert_eq!(
        state.position(),
        PROMPT,
        "capture executes nothing, so it must not advance the state",
    );

    let mut got_ids = Vec::with_capacity(STEPS);
    let mut got_logits = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        token = step
            .replay_step(&stream, &mut state, &graph, &[token])
            .expect("replay runs");
        got_ids.push(token);
        got_logits.push(dtoh(&stream, step.logits()));
    }

    // ---- the comparison --------------------------------------------------
    assert_eq!(
        state.position(),
        end_position,
        "the replayed run must leave the state where the launched run did",
    );
    println!("launched: {want_ids:?}");
    println!("replayed: {got_ids:?}");
    assert_eq!(
        got_ids, want_ids,
        "the captured step generated a different sequence than the launch path",
    );

    for (i, (got, want)) in got_logits.iter().zip(&want_logits).enumerate() {
        let differing = got.iter().zip(want).filter(|(a, b)| a != b).count();
        assert_eq!(
            differing,
            0,
            "step {i}: {differing} of {} logits differ between the replayed and \
             the launched step — the replay issues the identical kernels with \
             the identical arguments over the identical buffers, so this is \
             not a tolerance question",
            got.len(),
        );
    }
    println!(
        "{STEPS} replayed steps are bit-identical to {STEPS} launched steps on \
         all {} logits each, and select the same {STEPS} tokens",
        got_logits[0].len(),
    );

    // A sequence of eight identical tokens would satisfy everything above
    // while proving nothing about the position advancing, so say out loud
    // that it is not one.
    let distinct = {
        let mut v = got_ids.clone();
        v.sort_unstable();
        v.dedup();
        v.len()
    };
    assert!(
        distinct > 1,
        "all {STEPS} generated tokens are the same id ({}), which would make \
         the comparison above pass for a graph frozen at one position",
        got_ids[0],
    );
}
