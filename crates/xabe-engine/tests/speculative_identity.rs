//! The correctness contract R6 is legal under: speculative decode must emit
//! **exactly** what plain, non-speculative greedy decode emits.
//!
//! This is a hard gate rather than a tolerance because under greedy
//! decoding, a drafted token is accepted only if it equals the
//! target model's own argmax at that position, and a rejected draft is
//! replaced by the target's own argmax (the "bonus" token) rather than
//! dropped. So the emitted sequence is a pure function of the target model
//! and the prompt — speculation changes how many weight reads it costs to
//! produce it, never what it is. If this file ever fails, that is a bug in
//! acceptance or in the Gated DeltaNet rollback
//! (`crates/xabe-engine/src/block/gdn_verify.rs`), not a numerics question.
//!
//! Two regimes, both required by the brief:
//!
//! - **Natural text** ([`speculative_decode_matches_plain_greedy_on_a_natural_prompt`]):
//!   the golden capture's own 19-token prompt, which the model was captured
//!   actually continuing — acceptance should be real, not incidental.
//! - **Adversarial** ([`speculative_decode_matches_plain_greedy_on_a_random_prompt`]):
//!   token ids with no linguistic structure, where the draft head should
//!   agree with the target rarely if at all — the regime that exercises
//!   [`GdnSnapshotRing::commit`] at a small (often 1) accepted count on
//!   nearly every step, which is exactly the case a rollback bug would show
//!   up in and a mostly-successful draft would not.
//!
//! Both decode at least 64 tokens, per the brief.
//!
//! SKIPS — reporting that it skipped — without a driver, a supported device,
//! or the model file.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaContext, CudaStream};
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_engine::forward::Forward;
use xabe_engine::speculative::SpeculativeSession;
use xabe_engine::weights::DeviceWeights;
use xabe_gguf::GgufFile;
use xabe_model::config::ModelConfig;
use xabe_model::weights::{Directory, WeightSchema};

#[path = "golden.rs"]
mod golden;

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llmxabe/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";
const RMS_EPS_KEY: &str = "qwen35moe.attention.layer_norm_rms_epsilon";
const ROPE_FREQ_BASE_KEY: &str = "qwen35moe.rope.freq_base";

/// `docs/SCHEDULER.md`'s `DEFAULT_DRAFT_TOKENS_PER_STEP`.
const DRAFT_TOKENS: usize = 3;
/// The brief's floor.
const MIN_STEPS: usize = 64;

/// Each case makes roughly 30 GiB of model weights resident. Rust's test
/// harness otherwise runs both cases concurrently and asks a 48 GiB card to
/// hold two copies, turning the correctness gate into an OOM race.
static GPU_CASE: Mutex<()> = Mutex::new(());

fn model_path() -> PathBuf {
    std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
}

struct Fixture {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    file: GgufFile,
    config: ModelConfig,
    rms_eps: f32,
    rope_theta: f32,
}

fn setup() -> Option<Fixture> {
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
    let path = model_path();
    if !path.exists() {
        println!(
            "SKIPPED: model file not found at {}; set LLMXABE_MODEL to override",
            path.display(),
        );
        return None;
    }
    let stream = ctx.default_stream();
    let file = GgufFile::open(&path).expect("model file must parse as valid GGUF v3");
    let config = ModelConfig::qwen3_6_35b_a3b();
    let rms_eps = file.get_f32(RMS_EPS_KEY).expect("rms eps present");
    let rope_theta = file
        .get_f32(ROPE_FREQ_BASE_KEY)
        .expect("rope theta present");
    println!("device 0: {}", info.name);
    Some(Fixture {
        ctx,
        stream,
        file,
        config,
        rms_eps,
        rope_theta,
    })
}

fn directory<'a>(file: &'a GgufFile, config: &ModelConfig) -> Directory<'a> {
    let schema: &'static WeightSchema = Box::leak(Box::new(WeightSchema::with_mtp(config)));
    schema.resolve(file).expect("schema (with MTP) resolves")
}

fn plain_greedy_decode(
    fx: &Fixture,
    directory: &Directory<'_>,
    weights: &DeviceWeights,
    prompt_ids: &[i32],
    n_steps: usize,
    max_seq: usize,
) -> Vec<i32> {
    let mut prefill = Forward::new(
        &fx.ctx,
        &fx.stream,
        &fx.file,
        directory,
        weights,
        fx.config.clone(),
        prompt_ids.len(),
    )
    .expect("prefill pass builds");
    let mut state = prefill
        .new_state(&fx.stream, max_seq)
        .expect("sequence state allocates");
    prefill
        .run(&fx.stream, &mut state, prompt_ids, |_, _| {})
        .expect("prefill runs");
    let mut tok = prefill.sample_argmax(&fx.stream).expect("prefill samples");
    let mut out = vec![tok];

    let mut decode = prefill
        .reshape(&fx.ctx, &fx.stream, &fx.file, directory, weights, 1)
        .expect("decode pass reshapes");
    while out.len() < n_steps {
        decode
            .run(&fx.stream, &mut state, &[tok], |_, _| {})
            .expect("decode step runs");
        tok = decode
            .sample_argmax(&fx.stream)
            .expect("decode step samples");
        out.push(tok);
    }
    out
}

fn speculative_decode(
    fx: &Fixture,
    directory: &Directory<'_>,
    weights: &DeviceWeights,
    prompt_ids: &[i32],
    n_steps: usize,
    max_seq: usize,
) -> (Vec<i32>, usize, usize) {
    let (mut session, id_last) = SpeculativeSession::new(
        &fx.ctx,
        &fx.stream,
        &fx.file,
        directory,
        weights,
        &fx.config,
        fx.rms_eps,
        fx.rope_theta,
        prompt_ids,
        max_seq,
        DRAFT_TOKENS,
    )
    .expect("speculative session builds");

    let mut out = vec![id_last];
    let mut total_accepted = 0usize;
    let mut total_drafted = 0usize;
    while out.len() < n_steps {
        let outcome = session.step(&fx.stream).expect("speculative step runs");
        total_accepted += outcome.accepted;
        total_drafted += outcome.drafted;
        out.extend(outcome.emitted);
    }
    (out, total_accepted, total_drafted)
}

fn assert_sequences_match(regime: &str, plain: &[i32], speculative: &[i32], n: usize) {
    let plain = &plain[..n];
    let speculative = &speculative[..n];
    if plain != speculative {
        let first_diff = plain
            .iter()
            .zip(speculative)
            .position(|(a, b)| a != b)
            .unwrap_or(plain.len().min(speculative.len()));
        panic!(
            "[{regime}] speculative decode diverged from plain greedy decode at position \
             {first_diff}: plain={:?} speculative={:?}\nfull plain:       {plain:?}\nfull speculative: {speculative:?}",
            plain.get(first_diff),
            speculative.get(first_diff),
        );
    }
    println!("[{regime}] {n} tokens: speculative decode is bit-identical to plain greedy decode");
}

#[test]
fn speculative_decode_matches_plain_greedy_on_a_natural_prompt() {
    let _gpu_case = GPU_CASE.lock().expect("GPU test lock poisoned");
    let Some(fx) = setup() else { return };
    // The golden file is gitignored and lives at the main checkout's
    // `.golden/`; a worktree does not inherit it. `golden::setup` already
    // prints SKIPPED and returns None — do not panic over a missing capture.
    let Some(g) = golden::setup() else { return };
    let prompt_ids = g.tokens().to_vec();

    let dir = directory(&fx.file, &fx.config);
    let (weights, report) = DeviceWeights::load_where(
        &fx.ctx,
        &fx.stream,
        &fx.file,
        &dir,
        xabe_engine::forward::arena_holds,
    )
    .expect("weight load");
    println!(
        "loaded {} tensors, {:.2} GiB (block 40 included)",
        report.tensors,
        report.bytes as f64 / (1u64 << 30) as f64,
    );

    let max_seq = prompt_ids.len() + MIN_STEPS + DRAFT_TOKENS + 8;
    let plain = plain_greedy_decode(&fx, &dir, &weights, &prompt_ids, MIN_STEPS, max_seq);
    let (speculative, accepted, drafted) =
        speculative_decode(&fx, &dir, &weights, &prompt_ids, MIN_STEPS, max_seq);

    println!(
        "[natural] acceptance: {accepted}/{drafted} drafted tokens accepted ({:.1}%)",
        100.0 * accepted as f64 / drafted.max(1) as f64,
    );
    assert_sequences_match("natural", &plain, &speculative, MIN_STEPS);
}

#[test]
fn speculative_decode_matches_plain_greedy_on_a_random_prompt() {
    let _gpu_case = GPU_CASE.lock().expect("GPU test lock poisoned");
    let Some(fx) = setup() else { return };

    // Deterministic, structureless token ids — see the module docs on why
    // this regime matters: it is the one where nearly every step rolls back
    // a partially- or fully-rejected draft, which a mostly-accepting
    // natural-text run would rarely exercise.
    let vocab = fx.config.vocab_size as i64;
    let mut state = 0x5EED_u64;
    let prompt_ids: Vec<i32> = (0..16)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 33) % vocab as u64) as i32
        })
        .collect();

    let dir = directory(&fx.file, &fx.config);
    let (weights, report) = DeviceWeights::load_where(
        &fx.ctx,
        &fx.stream,
        &fx.file,
        &dir,
        xabe_engine::forward::arena_holds,
    )
    .expect("weight load");
    println!(
        "loaded {} tensors, {:.2} GiB (block 40 included)",
        report.tensors,
        report.bytes as f64 / (1u64 << 30) as f64,
    );

    let max_seq = prompt_ids.len() + MIN_STEPS + DRAFT_TOKENS + 8;
    let plain = plain_greedy_decode(&fx, &dir, &weights, &prompt_ids, MIN_STEPS, max_seq);
    let (speculative, accepted, drafted) =
        speculative_decode(&fx, &dir, &weights, &prompt_ids, MIN_STEPS, max_seq);

    println!(
        "[random] acceptance: {accepted}/{drafted} drafted tokens accepted ({:.1}%)",
        100.0 * accepted as f64 / drafted.max(1) as f64,
    );
    assert_sequences_match("random", &plain, &speculative, MIN_STEPS);
}
