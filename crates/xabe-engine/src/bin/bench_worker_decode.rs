//! Scheduler-driven aggregate decode throughput for one serving worker.
//!
//! Unlike `bench_decode_batch`, this binary goes through `Worker::step_device`:
//! admission, scheduler batching, the thread-owned CUDA runtime, graph replay,
//! sampling, and scheduler accounting are all in the measured path. Prefill is
//! completed before timing so the reported number is directly comparable to
//! `llama-batched-bench -npl N`'s generation result.
//!
//! ```sh
//! CUDA_VISIBLE_DEVICES=0 LLMXABE_BATCH_N=1,3 \
//!   cargo run --release -p xabe-engine --bin bench_worker_decode -- 2048 32
//! ```
//!
//! `LLMXABE_MODEL` overrides the model path. `LLMXABE_BATCH_N` is a
//! comma-separated list of widths in `1..=3` (default `1,3`).

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use tracing::{error, info};
use xabe_cache::CacheConfig;
use xabe_engine::{Worker, WorkerId};
use xabe_model::ModelConfig;
use xabe_sched::config::{DEFAULT_WATERMARK_FRACTION, SchedulerConfig};
use xabe_sched::request::{NewRequest, RequestId};

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llama.cpp/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";
const DEFAULT_CONTEXT: usize = 2_048;
const DEFAULT_STEPS: u32 = 32;
const WARMUP: u32 = 4;
const PREFILL_CHUNK: usize = 2_048;
const TOKEN_BUDGET: u32 = 4_096;
const TOTAL_CONTEXT: u32 = 393_216;

fn model_path() -> PathBuf {
    std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
}

fn widths() -> Result<Vec<usize>, &'static str> {
    let raw = std::env::var("LLMXABE_BATCH_N").unwrap_or_else(|_| "1,3".to_owned());
    let parsed: Vec<usize> = raw
        .split(',')
        .filter_map(|part| part.trim().parse().ok())
        .collect();
    if parsed.is_empty() || parsed.iter().any(|width| !(1..=3).contains(width)) {
        return Err("LLMXABE_BATCH_N must contain comma-separated widths in 1..=3");
    }
    Ok(parsed)
}

fn synthetic_prompt(sequence: usize, len: usize, vocab: usize) -> Vec<i32> {
    (0..len)
        .map(|position| (((sequence + 1) * 104_729 + position * 7919 + 1234) % vocab) as i32)
        .collect()
}

fn stats(samples: &[f64]) -> (f64, f64) {
    let count = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / count;
    let variance = samples
        .iter()
        .map(|sample| (sample - mean).powi(2))
        .sum::<f64>()
        / (count - 1.0).max(1.0);
    (mean, variance.sqrt())
}

fn run_width(
    path: &std::path::Path,
    model: &ModelConfig,
    context: usize,
    steps: u32,
    width: usize,
) -> Result<(f64, f64), String> {
    let cache = CacheConfig::with_defaults(model.clone()).map_err(|error| error.to_string())?;
    // Compare plain target-model decode with llama-batched-bench. The serving
    // default budgets n-gram drafts, which can emit a variable number of
    // tokens per scheduler call and would make this a different workload.
    let scheduler = SchedulerConfig::new(
        TOKEN_BUDGET,
        cache.attention_block_size(),
        width as u32,
        DEFAULT_WATERMARK_FRACTION,
        0,
    )
    .map_err(|error| error.to_string())?;
    let attention_blocks = TOTAL_CONTEXT / cache.attention_block_size();
    let mut worker = Worker::new(
        WorkerId(0),
        0,
        cache,
        scheduler,
        attention_blocks,
        width as u32,
    );
    worker
        .bind_device(path, model.clone(), PREFILL_CHUNK.min(context).max(1))
        .map_err(|error| error.to_string())?;

    for sequence in 0..width {
        let request = NewRequest {
            id: RequestId(sequence as u64 + 1),
            prompt_tokens: context as u32,
            // Earlier requests may decode while later long prompts are still
            // being chunk-prefilled. Leave enough runway for all requests to
            // become decode-ready before the steady-state window begins.
            max_output_tokens: WARMUP
                + steps
                + u32::try_from(context.div_ceil(PREFILL_CHUNK)).unwrap_or(u32::MAX)
                + 2,
        };
        worker
            .admit_tokens(
                request,
                synthetic_prompt(sequence, context, model.vocab_size as usize),
            )
            .map_err(|error| error.to_string())?;
    }

    // Admission and all chunked prefill stay outside the timed region. The
    // first decode step can share the scheduler batch with the final prefill,
    // so count emitted tokens rather than assuming a fixed number of calls.
    let mut warmed = 0u32;
    // The first full-width step captures the request-id-specific graph. Drop
    // it in addition to the ordinary warmups.
    while warmed < WARMUP + 1 {
        let step = worker.step_device().map_err(|error| error.to_string())?;
        if step.decode_items == width && step.prefill_items == 0 && step.generated.len() == width {
            if !step.completed.is_empty() || !step.stopped.is_empty() {
                return Err("a request completed during decode warmup".to_owned());
            }
            warmed += 1;
        }
    }

    let mut samples = Vec::with_capacity(steps as usize);
    let mut emitted = 0u32;
    while emitted < steps * width as u32 {
        let started = Instant::now();
        let step = worker.step_device().map_err(|error| error.to_string())?;
        let elapsed = started.elapsed().as_secs_f64() * 1e3;
        if step.decode_items != width || step.prefill_items != 0 {
            return Err(format!(
                "timed step was not steady-state decode: {} decodes, {} prefills",
                step.decode_items, step.prefill_items
            ));
        }
        if step.generated.len() != width {
            return Err(format!(
                "timed step emitted {} tokens for width {width}",
                step.generated.len()
            ));
        }
        if !step.completed.is_empty() || !step.stopped.is_empty() {
            return Err("a request completed during the timed decode window".to_owned());
        }
        samples.push(elapsed);
        emitted += width as u32;
    }
    Ok(stats(&samples))
}

fn main() -> ExitCode {
    let rest = xabe_log::init_from_args();
    let numbers: Vec<usize> = rest.iter().filter_map(|arg| arg.parse().ok()).collect();
    let context = numbers.first().copied().unwrap_or(DEFAULT_CONTEXT);
    let steps = numbers
        .get(1)
        .and_then(|value| u32::try_from(*value).ok())
        .unwrap_or(DEFAULT_STEPS);
    if context == 0 || steps == 0 {
        error!("context and steps must be positive");
        return ExitCode::FAILURE;
    }
    let widths = match widths() {
        Ok(widths) => widths,
        Err(message) => {
            error!(message);
            return ExitCode::FAILURE;
        }
    };
    let path = model_path();
    if !path.exists() {
        error!("model not found at {}", path.display());
        return ExitCode::FAILURE;
    }
    let model = ModelConfig::qwen3_6_35b_a3b();

    info!("scheduler-driven decode, context {context}, {steps} timed steps after {WARMUP} warmup");
    info!(
        "{:<10} {:>10} {:>10} {:>16} {:>16}",
        "shape", "mean ms", "sd ms", "aggregate tok/s", "per-seq tok/s"
    );
    for width in widths {
        match run_width(&path, &model, context, steps, width) {
            Ok((mean, sd)) => info!(
                "{:<10} {:>10.2} {:>10.2} {:>16.1} {:>16.1}",
                format!("N={width}"),
                mean,
                sd,
                width as f64 * 1e3 / mean,
                1e3 / mean,
            ),
            Err(error) => {
                error!("N={width} failed: {error}");
                return ExitCode::FAILURE;
            }
        }
    }
    ExitCode::SUCCESS
}
