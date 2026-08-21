//! The serving path's speculative contract, end to end: a worker configured
//! with `Speculation::Ngram` must emit **exactly** the token streams the
//! same worker emits with `Speculation::None`, for every request — through
//! the real scheduler, admission, chunked prefill, batched decode and the
//! batched verify step, not a hand-driven session.
//!
//! Two requests decode together so the batched verify runs at width 2 with
//! genuinely different sequences: one strongly periodic prompt (the regime
//! n-gram drafting exists for — the test asserts at least one step emitted
//! more than one token, so it cannot silently degrade into testing the
//! fallback path), and one structureless prompt (drafts fire rarely and are
//! mostly rejected — the rollback regime).
//!
//! SKIPS — reporting that it skipped — without a driver, a supported device,
//! or the model file.

use std::path::PathBuf;
use std::sync::Mutex;

use xabe_cache::CacheConfig;
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_engine::sampling::SamplingParams;
use xabe_engine::worker::{ServingConfig, Speculation, Worker, WorkerId};
use xabe_model::config::ModelConfig;
use xabe_sched::config::{DEFAULT_WATERMARK_FRACTION, SchedulerConfig};
use xabe_sched::request::{NewRequest, RequestId};

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llama.cpp/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";
const DEFAULT_DFLASH_PATH: &str =
    "/home/nixabe/llama.cpp/models/Qwen3.6-35B-A3B-GGUF/qwen36-35b-a3b-dflash-Q8_0.gguf";

const PREFILL_CHUNK: usize = 64;
const TOKEN_BUDGET: u32 = 4_096;
const TOTAL_CONTEXT: u32 = 32_768;
const WIDTH: u32 = 2;
/// `docs/SCHEDULER.md`'s `DEFAULT_DRAFT_TOKENS_PER_STEP`.
const DRAFTS: u32 = 3;
const MIN_TOKENS: usize = 48;

static GPU_CASE: Mutex<()> = Mutex::new(());

fn model_path() -> PathBuf {
    std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
}

fn dflash_path() -> PathBuf {
    std::env::var_os("LLMXABE_DFLASH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_DFLASH_PATH))
}

fn gpu_available() -> bool {
    if !driver_available() {
        println!("SKIPPED: no CUDA driver present");
        return false;
    }
    let Ok(ctx) = cudarc::driver::CudaContext::new(0) else {
        println!("SKIPPED: could not create a context on device 0");
        return false;
    };
    let info = DeviceInfo::from_context(0, &ctx).expect("device properties readable");
    if !info.is_supported() {
        println!("SKIPPED: device 0 is below the sm_75 minimum");
        return false;
    }
    if !model_path().exists() {
        println!(
            "SKIPPED: model file not found at {}; set LLMXABE_MODEL to override",
            model_path().display(),
        );
        return false;
    }
    true
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

struct ServingRun {
    /// Emitted tokens per request, in emission order.
    outputs: Vec<Vec<i32>>,
    /// Scheduler/device steps taken to produce them.
    steps: usize,
    /// Steps in which some request emitted more than one token — only a
    /// working accept path can produce these.
    multi_token_steps: usize,
}

fn run_serving(
    speculation: Speculation,
    drafts: u32,
    prompts: &[Vec<i32>],
) -> Result<ServingRun, String> {
    let model = ModelConfig::qwen3_6_35b_a3b();
    let cache = CacheConfig::with_defaults(model.clone()).map_err(|e| e.to_string())?;
    let scheduler = SchedulerConfig::new(
        TOKEN_BUDGET,
        cache.attention_block_size(),
        WIDTH,
        DEFAULT_WATERMARK_FRACTION,
        drafts,
    )
    .map_err(|e| e.to_string())?;
    let attention_blocks = TOTAL_CONTEXT / cache.attention_block_size();
    let mut worker = Worker::new(WorkerId(0), 0, cache, scheduler, attention_blocks, WIDTH);
    let serving = ServingConfig {
        speculation,
        dflash_gguf: matches!(speculation, Speculation::DFlash).then(dflash_path),
        ..ServingConfig::new(PREFILL_CHUNK)
    };
    // The benchmark bind leaves EOS an ordinary token, so both runs decode
    // the same fixed number of steps regardless of what the text "says".
    worker
        .bind_device_for_benchmark(&model_path(), model, serving)
        .map_err(|e| e.to_string())?;

    for (i, prompt) in prompts.iter().enumerate() {
        let request = NewRequest {
            id: RequestId(i as u64 + 1),
            prompt_tokens: prompt.len() as u32,
            max_output_tokens: (MIN_TOKENS + 16) as u32,
        };
        worker
            .admit_tokens(request, prompt.clone(), Vec::new(), SamplingParams::GREEDY)
            .map_err(|e| e.to_string())?;
    }

    let mut outputs = vec![Vec::new(); prompts.len()];
    let mut steps = 0usize;
    let mut multi_token_steps = 0usize;
    while outputs.iter().any(|o: &Vec<i32>| o.len() < MIN_TOKENS) {
        if steps > 64 * prompts.len() + 64 {
            return Err(format!(
                "no progress after {steps} steps; emitted {:?}",
                outputs.iter().map(Vec::len).collect::<Vec<_>>(),
            ));
        }
        let step = worker.step_device().map_err(|e| e.to_string())?;
        let mut per_request = vec![0usize; prompts.len()];
        for (id, token) in &step.generated {
            let index = (id.0 - 1) as usize;
            outputs[index].push(*token);
            per_request[index] += 1;
        }
        if per_request.iter().any(|&n| n > 1) {
            multi_token_steps += 1;
        }
        steps += 1;
    }
    Ok(ServingRun {
        outputs,
        steps,
        multi_token_steps,
    })
}

/// Same contract for the trained MTP head: `Speculation::Mtp` through the
/// real serving loop — chunked prefill catch-up, chained batch drafting,
/// batched verify — must emit exactly what `Speculation::None` emits.
#[test]
fn serving_with_mtp_speculation_matches_serving_without() {
    let _gpu_case = GPU_CASE.lock().expect("GPU test lock poisoned");
    if !gpu_available() {
        return;
    }
    let vocab = ModelConfig::qwen3_6_35b_a3b().vocab_size as i64;

    // 96 tokens: longer than `PREFILL_CHUNK`, so the draft-head catch-up
    // must cross a chunk boundary and carry `pending_h` between chunks —
    // the path a one-chunk prompt would leave untested.
    let cycle = [791i32, 1131, 1721, 2217];
    let periodic: Vec<i32> = cycle.iter().copied().cycle().take(96).collect();
    let random = xorshift_prompt(0x5EED_CAFE, 16, vocab);
    let prompts = vec![periodic, random];

    let plain = run_serving(Speculation::None, 0, &prompts).expect("plain serving runs");
    let spec = run_serving(Speculation::Mtp, DRAFTS, &prompts).expect("mtp serving runs");

    println!(
        "plain: {} steps; mtp: {} steps, {} multi-token steps",
        plain.steps, spec.steps, spec.multi_token_steps,
    );
    for (i, (a, b)) in plain.outputs.iter().zip(&spec.outputs).enumerate() {
        let n = MIN_TOKENS.min(a.len()).min(b.len());
        assert_eq!(
            a[..n],
            b[..n],
            "request {i}: MTP serving diverged from plain serving",
        );
        println!("request {i}: {n} tokens identical");
    }
    // The trained head must actually get drafts accepted — a run where every
    // draft was rejected would make this test a no-op on the accept path.
    assert!(
        spec.multi_token_steps > 0,
        "no step emitted more than one token; the MTP head never had a draft accepted",
    );
}

/// Same contract for the DFlash drafter: `Speculation::DFlash` through the
/// real serving loop — feature taps riding prefill and verify, context
/// injection, one block-in-fill drafter pass per step — must emit exactly
/// what `Speculation::None` emits. SKIPS without the drafter GGUF.
#[test]
fn serving_with_dflash_speculation_matches_serving_without() {
    let _gpu_case = GPU_CASE.lock().expect("GPU test lock poisoned");
    if !gpu_available() {
        return;
    }
    if !dflash_path().exists() {
        println!(
            "SKIPPED: DFlash drafter not found at {}; set LLMXABE_DFLASH to override",
            dflash_path().display(),
        );
        return;
    }
    let vocab = ModelConfig::qwen3_6_35b_a3b().vocab_size as i64;

    // Natural-ish structure for the drafter (a trained model, not a suffix
    // matcher) plus a structureless prompt, and a 96-token prompt so the
    // feature taps cross a prefill-chunk boundary.
    let cycle = [791i32, 1131, 1721, 2217];
    let periodic: Vec<i32> = cycle.iter().copied().cycle().take(96).collect();
    let random = xorshift_prompt(0x5EED_CAFE, 16, vocab);
    let prompts = vec![periodic, random];

    let plain = run_serving(Speculation::None, 0, &prompts).expect("plain serving runs");
    let spec = run_serving(Speculation::DFlash, DRAFTS, &prompts).expect("dflash serving runs");

    println!(
        "plain: {} steps; dflash: {} steps, {} multi-token steps",
        plain.steps, spec.steps, spec.multi_token_steps,
    );
    for (i, (a, b)) in plain.outputs.iter().zip(&spec.outputs).enumerate() {
        let n = MIN_TOKENS.min(a.len()).min(b.len());
        assert_eq!(
            a[..n],
            b[..n],
            "request {i}: DFlash serving diverged from plain serving",
        );
        println!("request {i}: {n} tokens identical");
    }
    // The drafter must actually get drafts accepted. This doubles as the
    // behavioral gate on the whole feature pipeline: a wrong tap layer, a
    // wrong rope, or a wrong mask cannot break exactness (the verify
    // guarantees that) — they break *this*, by dragging acceptance to zero.
    assert!(
        spec.multi_token_steps > 0,
        "no step emitted more than one token; the DFlash drafter never had a draft accepted",
    );
}

#[test]
fn serving_with_ngram_speculation_matches_serving_without() {
    let _gpu_case = GPU_CASE.lock().expect("GPU test lock poisoned");
    if !gpu_available() {
        return;
    }
    let vocab = ModelConfig::qwen3_6_35b_a3b().vocab_size as i64;

    // A strongly periodic prompt: greedy continuation of a short cycle is
    // the canonical n-gram-draftable stream. And a structureless one.
    let cycle = [791i32, 1131, 1721, 2217];
    let periodic: Vec<i32> = cycle.iter().copied().cycle().take(32).collect();
    let random = xorshift_prompt(0x5EED_CAFE, 16, vocab);
    let prompts = vec![periodic, random];

    let plain = run_serving(Speculation::None, 0, &prompts).expect("plain serving runs");
    let spec =
        run_serving(Speculation::Ngram { min: 2, max: 4 }, DRAFTS, &prompts).expect("spec serving");

    println!(
        "plain: {} steps; spec: {} steps, {} multi-token steps",
        plain.steps, spec.steps, spec.multi_token_steps,
    );
    for (i, (a, b)) in plain.outputs.iter().zip(&spec.outputs).enumerate() {
        let n = MIN_TOKENS.min(a.len()).min(b.len());
        assert_eq!(
            a[..n],
            b[..n],
            "request {i}: speculative serving diverged from plain serving",
        );
        println!("request {i}: {n} tokens identical");
    }
    // The periodic request must actually exercise acceptance — a run where
    // every draft was rejected (or never proposed) would make this test a
    // no-op on the code it exists to gate.
    assert!(
        spec.multi_token_steps > 0,
        "no step emitted more than one token; the verify path never accepted a draft",
    );
    assert!(
        spec.steps <= plain.steps,
        "speculation took more steps ({}) than plain decoding ({})",
        spec.steps,
        plain.steps,
    );
}
