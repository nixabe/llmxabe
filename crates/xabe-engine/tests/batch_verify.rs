//! `Forward::run_batch_verify`'s correctness contract: a batch of sequences
//! verifying drafted windows in one pass must emit **exactly** what plain,
//! non-speculative greedy decode emits for every sequence — the same hard
//! gate `tests/speculative_identity.rs` holds the single-sequence path to.
//!
//! Drafts here are crafted, not modeled: each step's draft for a sequence is
//! the plain run's own continuation for the first `k_target` positions and a
//! deliberately wrong token after, with `k_target` cycling `0..=d` across
//! steps. That drives every acceptance count through both sequences on a
//! fixed schedule, so the full-accept commit, the partial rollback and the
//! full rejection are all exercised many times each — and because the
//! expected acceptance is known per step, the test asserts it exactly,
//! which a modeled drafter cannot.
//!
//! The two sequences use different prompts and different lengths, so their
//! attention caches, GDN states and snapshot rings are genuinely distinct;
//! a cross-sequence indexing bug in the batched verify (rows, rings, or
//! per-sequence state) diverges one of them immediately.
//!
//! SKIPS — reporting that it skipped — without a driver, a supported device,
//! or the model file.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaContext, CudaStream};
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_engine::forward::Forward;
use xabe_engine::weights::DeviceWeights;
use xabe_gguf::GgufFile;
use xabe_model::config::ModelConfig;
use xabe_model::weights::{Directory, WeightSchema};

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llama.cpp/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

/// `docs/SCHEDULER.md`'s `DEFAULT_DRAFT_TOKENS_PER_STEP`.
const DRAFT_TOKENS: usize = 3;
const WINDOW: usize = 1 + DRAFT_TOKENS;
const MIN_STEPS: usize = 64;

/// Batch width, overridable for divergence bisection: `LLMXABE_TEST_BATCH=1`
/// runs the same code path at the single-sequence shape the identity test
/// already proves for `run_verify`, isolating width-dependent numerics.
fn sequences() -> usize {
    std::env::var("LLMXABE_TEST_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2)
}

/// One weights copy is ~30 GiB; serialize GPU cases as the other GPU tests
/// in this crate do.
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
    println!("device 0: {}", info.name);
    Some(Fixture {
        ctx,
        stream,
        file,
        config,
    })
}

fn xorshift_prompt(seed: u64, len: usize, vocab: i64) -> Vec<i32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 33) % vocab as u64) as i32
        })
        .collect()
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

/// Contamination bisector: two sequences with *identical* prompts, drafts
/// and schedules must produce identical rows at every stage of every layer
/// — their halves of each stage buffer are compared within the same pass.
/// The first differing stage names the layer and family where one
/// sequence's data leaks into the other's. Purely diagnostic; run with
/// `LLMXABE_TEST_TWIN=1` (skips otherwise so the suite stays cheap).
#[test]
fn twin_sequences_stay_identical_through_every_stage() {
    if std::env::var_os("LLMXABE_TEST_TWIN").is_none() {
        println!("SKIPPED: set LLMXABE_TEST_TWIN=1 to run the twin bisector");
        return;
    }
    let _gpu_case = GPU_CASE.lock().expect("GPU test lock poisoned");
    let Some(fx) = setup() else { return };

    let vocab = fx.config.vocab_size as i64;
    let prompt = xorshift_prompt(0x5EED, 16, vocab);
    let n_seq = 2usize;

    let schema: &'static WeightSchema = Box::leak(Box::new(WeightSchema::new(&fx.config)));
    let dir = schema.resolve(&fx.file).expect("schema resolves");
    let (weights, _) = DeviceWeights::load_where(
        &fx.ctx,
        &fx.stream,
        &fx.file,
        &dir,
        xabe_engine::forward::arena_holds,
    )
    .expect("weight load");

    let slack = MIN_STEPS + WINDOW + 4;
    let max_seq = prompt.len() + slack + 8;
    let plain = plain_greedy_decode(&fx, &dir, &weights, &prompt, slack, max_seq);

    let mut prefill = Forward::new(
        &fx.ctx,
        &fx.stream,
        &fx.file,
        &dir,
        &weights,
        fx.config.clone(),
        prompt.len(),
    )
    .expect("prefill pass builds");
    let mut states = Vec::new();
    let mut emitted: Vec<Vec<i32>> = Vec::new();
    for _ in 0..n_seq {
        let mut state = prefill
            .new_state(&fx.stream, max_seq)
            .expect("sequence state allocates");
        prefill
            .run(&fx.stream, &mut state, &prompt, |_, _| {})
            .expect("prefill runs");
        let first = prefill.sample_argmax(&fx.stream).expect("prefill samples");
        states.push(state);
        emitted.push(vec![first]);
    }

    let mut verify = prefill
        .reshape(
            &fx.ctx,
            &fx.stream,
            &fx.file,
            &dir,
            &weights,
            n_seq * WINDOW,
        )
        .expect("verify pass reshapes");
    verify
        .enable_batch_decode(&fx.ctx, &fx.stream)
        .expect("batch decode enables");
    verify
        .enable_verify(&fx.stream)
        .expect("verify scratch enables");
    let mut rings: Vec<_> = (0..n_seq)
        .map(|_| {
            verify
                .new_verify_rings(&fx.stream, WINDOW)
                .expect("snapshot rings allocate")
        })
        .collect();

    let mut step = 0usize;
    let mut window_ids = vec![0i32; n_seq * WINDOW];
    let mut draft = [0i32; DRAFT_TOKENS];
    while emitted[0].len() < MIN_STEPS {
        let k_target = step % (DRAFT_TOKENS + 1);
        let e = emitted[0].len();
        for (s, em) in emitted.iter().enumerate() {
            window_ids[s * WINDOW] = *em.last().expect("prefill emitted a token");
            for j in 0..DRAFT_TOKENS {
                let correct = plain[e + j];
                let tok = if j < k_target {
                    correct
                } else {
                    (correct + 1) % vocab as i32
                };
                draft[j] = tok;
                window_ids[s * WINDOW + 1 + j] = tok;
            }
        }

        let stream = Arc::clone(&fx.stream);
        let this_step = step;
        let rows = verify
            .run_batch_verify_with_stage_waypoints(
                &fx.stream,
                &mut states,
                &mut rings,
                &window_ids,
                |layer, stage, buffer| {
                    let mut host = vec![0f32; buffer.len()];
                    stream
                        .memcpy_dtoh(buffer, &mut host)
                        .expect("waypoint read-back");
                    let half = host.len() / 2;
                    let (a, b) = host.split_at(half);
                    let diff = a
                        .iter()
                        .zip(b)
                        .map(|(x, y)| (x - y).abs())
                        .fold(0f32, f32::max);
                    assert!(
                        diff == 0.0,
                        "step {this_step}: twin halves diverge at layer {layer:?} stage \
                         {stage:?}, max abs diff {diff:e}",
                    );
                },
            )
            .expect("batched verify runs");

        let (head, tail) = rows.split_at(WINDOW);
        assert_eq!(head, tail, "step {step}: twin argmax rows diverge");

        for s in 0..n_seq {
            let seq_rows = &rows[s * WINDOW..(s + 1) * WINDOW];
            let mut accepted = 0usize;
            while accepted < DRAFT_TOKENS
                && seq_rows[accepted] == window_ids[s * WINDOW + 1 + accepted]
            {
                accepted += 1;
            }
            let bonus = seq_rows[accepted];
            verify
                .commit_verify_window(&fx.stream, &mut states[s], &rings[s], accepted + 1)
                .expect("verify window commits");
            let mut new_tokens: Vec<i32> =
                window_ids[s * WINDOW + 1..s * WINDOW + 1 + accepted].to_vec();
            new_tokens.push(bonus);
            emitted[s].extend_from_slice(&new_tokens);
        }
        step += 1;
    }
    println!("twins stayed identical for {} steps", step);
}

#[test]
fn batched_verify_matches_plain_greedy_decode_for_every_sequence() {
    let _gpu_case = GPU_CASE.lock().expect("GPU test lock poisoned");
    let Some(fx) = setup() else { return };

    let vocab = fx.config.vocab_size as i64;
    let n_seq = sequences();
    let prompts: Vec<Vec<i32>> = [(0x5EED, 16), (0xB0A7, 24), (0xD1CE, 20)][..n_seq]
        .iter()
        .map(|&(seed, len)| xorshift_prompt(seed, len, vocab))
        .collect();

    let schema: &'static WeightSchema = Box::leak(Box::new(WeightSchema::new(&fx.config)));
    let dir = schema.resolve(&fx.file).expect("schema resolves");
    let (weights, report) = DeviceWeights::load_where(
        &fx.ctx,
        &fx.stream,
        &fx.file,
        &dir,
        xabe_engine::forward::arena_holds,
    )
    .expect("weight load");
    println!(
        "loaded {} tensors, {:.2} GiB",
        report.tensors,
        report.bytes as f64 / (1u64 << 30) as f64,
    );

    // Plain greedy runs first: they are both the identity baseline and the
    // source of the crafted drafts. Decode a margin past MIN_STEPS so a
    // full-accept step near the end still has known continuations to draft.
    let slack = MIN_STEPS + WINDOW + 4;
    let max_seq = prompts.iter().map(Vec::len).max().unwrap() + slack + 8;
    let plain: Vec<Vec<i32>> = prompts
        .iter()
        .map(|p| plain_greedy_decode(&fx, &dir, &weights, p, slack, max_seq))
        .collect();

    // The speculative side: prefill each sequence, then verify crafted
    // windows batched across both sequences in one pass per step.
    let mut prefill = Forward::new(
        &fx.ctx,
        &fx.stream,
        &fx.file,
        &dir,
        &weights,
        fx.config.clone(),
        prompts[0].len(),
    )
    .expect("prefill pass builds");
    let mut states = Vec::with_capacity(n_seq);
    let mut emitted: Vec<Vec<i32>> = Vec::with_capacity(n_seq);
    for (i, prompt) in prompts.iter().enumerate() {
        if prefill.tokens() != prompt.len() {
            prefill = prefill
                .reshape(&fx.ctx, &fx.stream, &fx.file, &dir, &weights, prompt.len())
                .expect("prefill reshapes to the next prompt length");
        }
        let mut state = prefill
            .new_state(&fx.stream, max_seq)
            .expect("sequence state allocates");
        prefill
            .run(&fx.stream, &mut state, prompt, |_, _| {})
            .expect("prefill runs");
        let first = prefill.sample_argmax(&fx.stream).expect("prefill samples");
        states.push(state);
        emitted.push(vec![first]);
        println!("seq {i}: prefilled {} tokens", prompt.len());
    }

    let mut verify = prefill
        .reshape(
            &fx.ctx,
            &fx.stream,
            &fx.file,
            &dir,
            &weights,
            n_seq * WINDOW,
        )
        .expect("verify pass reshapes");
    verify
        .enable_batch_decode(&fx.ctx, &fx.stream)
        .expect("batch decode enables");
    verify
        .enable_verify(&fx.stream)
        .expect("verify scratch enables");
    let mut rings: Vec<_> = (0..n_seq)
        .map(|_| {
            verify
                .new_verify_rings(&fx.stream, WINDOW)
                .expect("snapshot rings allocate")
        })
        .collect();

    let mut step = 0usize;
    let mut window_ids = vec![0i32; n_seq * WINDOW];
    let mut drafts = vec![[0i32; DRAFT_TOKENS]; n_seq];
    while emitted.iter().any(|e| e.len() < MIN_STEPS) {
        // Craft each sequence's draft from its own plain continuation:
        // correct for the first k_target positions, deliberately wrong
        // after. Different target counts per sequence in the same step
        // exercise per-sequence rollback within one batched pass.
        for s in 0..n_seq {
            let k_target = (step + s) % (DRAFT_TOKENS + 1);
            let e = emitted[s].len();
            window_ids[s * WINDOW] = *emitted[s].last().expect("prefill emitted a token");
            for j in 0..DRAFT_TOKENS {
                let correct = plain[s][e + j];
                let tok = if j < k_target {
                    correct
                } else {
                    (correct + 1) % vocab as i32
                };
                drafts[s][j] = tok;
                window_ids[s * WINDOW + 1 + j] = tok;
            }
        }

        let rows = verify
            .run_batch_verify(&fx.stream, &mut states, &mut rings, &window_ids)
            .expect("batched verify runs");

        for s in 0..n_seq {
            let k_target = (step + s) % (DRAFT_TOKENS + 1);
            let seq_rows = &rows[s * WINDOW..(s + 1) * WINDOW];
            let mut accepted = 0usize;
            while accepted < DRAFT_TOKENS && seq_rows[accepted] == drafts[s][accepted] {
                accepted += 1;
            }
            assert_eq!(
                accepted, k_target,
                "seq {s} step {step}: crafted draft should be accepted for exactly \
                 {k_target} positions, got {accepted} (rows {seq_rows:?}, draft {:?})",
                drafts[s],
            );
            let bonus = seq_rows[accepted];
            verify
                .commit_verify_window(&fx.stream, &mut states[s], &rings[s], accepted + 1)
                .expect("verify window commits");
            emitted[s].extend_from_slice(&drafts[s][..accepted]);
            emitted[s].push(bonus);
            // The bonus token is the divergence canary: a wrong bonus at any
            // step silently shifts the context, and every later crafted
            // draft is then rejected at k = 0 — the failure would surface
            // steps later, far from its cause. Catch it at the step.
            let e = emitted[s].len();
            assert_eq!(
                emitted[s][e - 1],
                plain[s][e - 1],
                "seq {s} step {step} (k_target {k_target}): bonus diverged from plain \
                 greedy decode at emitted position {} (rows {seq_rows:?})",
                e - 1,
            );
        }
        step += 1;
    }

    for s in 0..n_seq {
        let n = MIN_STEPS.min(emitted[s].len());
        let plain_prefix = &plain[s][..n];
        let spec_prefix = &emitted[s][..n];
        if plain_prefix != spec_prefix {
            let first_diff = plain_prefix
                .iter()
                .zip(spec_prefix)
                .position(|(a, b)| a != b)
                .unwrap_or(n);
            panic!(
                "seq {s}: batched verify diverged from plain greedy decode at position \
                 {first_diff}: plain={:?} speculative={:?}\nplain:       {plain_prefix:?}\nspeculative: {spec_prefix:?}",
                plain_prefix.get(first_diff),
                spec_prefix.get(first_diff),
            );
        }
        println!(
            "seq {s}: {n} tokens bit-identical to plain greedy decode across {step} batched \
             verify steps"
        );
    }
}
