//! Autoregressive decode throughput: the number that compares to `llama-bench
//! -n`.
//!
//! `bench_forward` measures a *prefill* — one pass over N tokens from a cold
//! start — which is the `pp` column. Its `n = 1` row is often quoted as a
//! decode floor, and that is wrong in a way worth stating: a cold one-token
//! pass zeroes 30 recurrent states, attends to a one-key window, and reads no
//! cache. Real decode carries state and attends to everything before it, and
//! it gets more expensive as the context grows.
//!
//! This measures the real thing:
//!
//! ```text
//!   prefill `prompt` tokens          (not timed, it is the `pp` column's job)
//!   decode `warmup` tokens           (not timed)
//!   decode `steps` tokens            timed, per step
//! ```
//!
//! Per-step latency is reported as a distribution rather than a mean, because
//! decode cost is a function of context length and a mean hides the slope. If
//! the last decile is much slower than the first, the KV window is what is
//! costing, and that is a finding rather than noise.
//!
//! # What is and is not measured
//!
//! Greedy sampling **is** inside the timed region, because an autoregressive
//! step is not finished until the host knows which token to feed back in.
//! [`Forward::sample_argmax`] reduces on the device and returns four bytes;
//! doing it on the host instead moved the whole 993 KiB logit vector across
//! PCIe every step and cost 4% of the step.
//!
//! What is not measured is everything a sampler does past argmax: no top-p,
//! no repetition penalty, and no tokenizer to print what it said. Those are
//! host-side work on a 248,320-entry vector and would be a real cost; this
//! benchmark does not pay it, and neither does `llama-bench -n`.
//!
//! Usage:
//!
//! ```sh
//! cargo run --release -p xabe-engine --bin bench_decode -- [prompt] [steps]
//! ```

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use cudarc::driver::CudaContext;
use tracing::{debug, error, info, warn};

use xabe_cuda::arena::memory_info;
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_engine::DeviceWeights;
use xabe_engine::forward::{Forward, arena_holds};
use xabe_gguf::GgufFile;
use xabe_model::config::ModelConfig;
use xabe_model::weights::WeightSchema;

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llama.cpp/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

/// Tokens decoded before the clock starts.
///
/// The first step after a prefill pays for whatever the driver still has in
/// flight from it, and the second is the first that is representative.
const WARMUP: usize = 4;

/// Default prompt length and timed step count.
const DEFAULT_PROMPT: usize = 128;
const DEFAULT_STEPS: usize = 64;

fn model_path() -> PathBuf {
    std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
}

/// Mean and sample standard deviation.
fn stats(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, var.sqrt())
}

/// The `p`-th percentile of an already-sorted slice, nearest rank.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    let i = ((p / 100.0 * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len()) - 1;
    sorted[i]
}

fn main() -> ExitCode {
    let rest = xabe_log::init_from_args();
    let nums: Vec<usize> = rest.iter().filter_map(|a| a.parse().ok()).collect();
    let prompt_len = nums.first().copied().unwrap_or(DEFAULT_PROMPT);
    let steps = nums.get(1).copied().unwrap_or(DEFAULT_STEPS);

    if !driver_available() {
        error!("No CUDA driver reachable on this host.");
        return ExitCode::FAILURE;
    }
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            error!("Could not create a context on device 0: {e}");
            return ExitCode::FAILURE;
        }
    };
    let info = DeviceInfo::from_context(0, &ctx).expect("device properties readable");
    if !info.is_supported() {
        error!("Device 0 is below the sm_75 minimum.");
        return ExitCode::FAILURE;
    }
    let path = model_path();
    if !path.exists() {
        warn!(
            "SKIPPED: model file not found at {}; set LLMXABE_MODEL to override",
            path.display(),
        );
        return ExitCode::SUCCESS;
    }

    let config = ModelConfig::qwen3_6_35b_a3b();
    let stream = ctx.default_stream();
    let file = GgufFile::open(&path).expect("valid GGUF v3");
    let (free_at_start, total) = memory_info(&ctx).expect("memory info");

    info!(
        "device 0: {} sm_{}{}, {:.1} GiB",
        info.name,
        info.compute_capability.major,
        info.compute_capability.minor,
        total as f64 / (1u64 << 30) as f64,
    );

    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).expect("schema resolves");
    let (weights, load) = DeviceWeights::load_where(&ctx, &stream, &file, &directory, arena_holds)
        .expect("weight load");
    debug!(
        "arena {:.3} GiB in {:.1} s",
        load.bytes as f64 / (1u64 << 30) as f64,
        load.elapsed.as_secs_f64(),
    );

    // Two shapes over one set of weights: the prefill, and the single-token
    // step that every decode replays. `reshape` shares the 28.3 GiB of MoE
    // weights, which is the only reason two shapes fit at all.
    let built = Instant::now();
    let mut prefill = match Forward::new(
        &ctx,
        &stream,
        &file,
        &directory,
        &weights,
        config.clone(),
        prompt_len,
    ) {
        Ok(f) => f,
        Err(e) => {
            error!("FAILED to build the {prompt_len}-token prefill: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut step = match prefill.reshape(&ctx, &stream, &file, &directory, &weights, 1) {
        Ok(f) => f,
        Err(e) => {
            error!("FAILED to build the decode step: {e}");
            return ExitCode::FAILURE;
        }
    };
    debug!("built 2 shapes in {:.1} s", built.elapsed().as_secs_f64());

    let max_seq = prompt_len + WARMUP + steps;
    let mut state = match prefill.new_state(&stream, max_seq) {
        Ok(s) => s,
        Err(e) => {
            error!("FAILED to allocate sequence state for {max_seq} positions: {e}");
            return ExitCode::FAILURE;
        }
    };

    // In-vocabulary, non-degenerate ids: the same generator `bench_forward`
    // uses, so the two are measuring the same routing spread.
    let ids: Vec<i32> = (0..prompt_len)
        .map(|i| ((i * 7919 + 1234) % config.vocab_size as usize) as i32)
        .collect();

    let t = Instant::now();
    if let Err(e) = prefill.run(&stream, &mut state, &ids, |_, _| {}) {
        error!("FAILED during prefill: {e}");
        return ExitCode::FAILURE;
    }
    stream.synchronize().expect("sync");
    let prefill_ms = t.elapsed().as_secs_f64() * 1e3;

    let mut next = prefill.sample_argmax(&stream).expect("prefill argmax");

    // Sampling is on the device and synchronizes there, so the timed region
    // holds a full autoregressive step: forward, greedy pick, and the four
    // bytes the host needs to choose the next input. Reducing the 248,320
    // logits on the host instead moved 993 KiB across PCIe and scanned them
    // on one core every step, which measured as 0.5 ms of an 11.6 ms step.
    let mut decode_one = |state: &mut _, token: i32| -> Result<i32, String> {
        step.run(&stream, state, &[token], |_, _| {})
            .map_err(|e| e.to_string())?;
        step.sample_argmax(&stream).map_err(|e| e.to_string())
    };

    for _ in 0..WARMUP {
        match decode_one(&mut state, next) {
            Ok(id) => next = id,
            Err(e) => {
                error!("FAILED during decode warmup: {e}");
                return ExitCode::FAILURE;
            }
        }
    }

    let mut samples = Vec::with_capacity(steps);
    for _ in 0..steps {
        let t = Instant::now();
        match decode_one(&mut state, next) {
            Ok(id) => next = id,
            Err(e) => {
                error!("FAILED during decode: {e}");
                return ExitCode::FAILURE;
            }
        }
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }

    let (free_now, _) = memory_info(&ctx).expect("memory info");
    let (mean, sd) = stats(&samples);
    let mut sorted = samples.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in a wall clock"));

    // The slope over the timed window. Decode cost grows with context because
    // attention reads a longer cache each step; if this is flat, the KV window
    // is not what is costing at this length, and that is worth knowing before
    // anyone optimizes it.
    let tenth = samples.len().max(10) / 10;
    let (first_mean, _) = stats(&samples[..tenth]);
    let (last_mean, _) = stats(&samples[samples.len() - tenth..]);

    info!("");
    info!(
        "prompt {prompt_len} tokens, prefill {prefill_ms:.1} ms ({:.1} tok/s)",
        prompt_len as f64 / (prefill_ms / 1e3),
    );
    info!(
        "decode {steps} steps after {WARMUP} warmup, context {}..{}",
        prompt_len + WARMUP,
        prompt_len + WARMUP + steps,
    );
    info!("");
    info!("  mean       {mean:8.2} ms   {:8.2} tok/s", 1e3 / mean);
    info!("  sd         {sd:8.2} ms");
    info!(
        "  min / p50  {:8.2} / {:.2} ms",
        sorted[0],
        percentile(&sorted, 50.0),
    );
    info!(
        "  p90 / p99  {:8.2} / {:.2} ms",
        percentile(&sorted, 90.0),
        percentile(&sorted, 99.0),
    );
    info!(
        "  first 10% vs last 10%: {first_mean:.2} -> {last_mean:.2} ms ({:+.1}% over {steps} steps)",
        (last_mean / first_mean - 1.0) * 100.0,
    );
    info!("");
    info!(
        "sequence state {:.3} GiB for {max_seq} positions; peak VRAM {:.3} GiB of {:.2} GiB",
        state.bytes() as f64 / (1u64 << 30) as f64,
        free_at_start.saturating_sub(free_now) as f64 / (1u64 << 30) as f64,
        total as f64 / (1u64 << 30) as f64,
    );
    ExitCode::SUCCESS
}
