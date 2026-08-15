//! Times the forward pass at several batch sizes, for comparison against
//! `llama-bench`.
//!
//! # What this measures, and what it does not
//!
//! One call to [`Forward::run`] is a complete pass over `n` tokens: embedding
//! lookup, 40 blocks, final norm, LM head. It carries **no state between
//! calls** — `run` zeroes the Gated DeltaNet recurrent matrices and
//! convolution caches every time, and there is no KV cache for the ten
//! attention layers, so each pass recomputes its whole window.
//!
//! That makes this directly comparable to llama.cpp's **`pp` (prompt
//! processing)** figure, which is also a full forward over a batch. It is
//! **not** comparable to llama.cpp's `tg` (token generation): a decode step
//! reuses a KV cache and resumes a recurrent state, and llmxabe has neither
//! yet — that is story G007. Reporting the `n = 1` row as a decode rate would
//! be comparing a cold full pass against a warm incremental one, so it is
//! labelled a *latency floor* instead: it is what one pass costs, and a real
//! decode step cannot be slower than the work it shares with this.
//!
//! Methodology mirrors `llama-bench`: the model is loaded once, each batch
//! size gets its own `Forward` (the token count is fixed at construction),
//! warmup passes are discarded, and the reported figure is the mean and
//! sample standard deviation over the timed repetitions. The stream is
//! synchronized inside the timed region, so the number is wall clock for
//! completed work rather than launch latency.
//!
//! ```text
//! cargo run --release -p xabe-engine --bin bench_forward
//! ```
//!
//! Environment: `LLMXABE_MODEL` overrides the model path, `LLMXABE_BENCH_N`
//! overrides the comma-separated batch sizes, `LLMXABE_BENCH_REPS` the
//! repetition count.

use std::path::PathBuf;
use std::time::Instant;

use cudarc::driver::CudaContext;
use tracing::{error, info, warn};
use xabe_cuda::arena::memory_info;
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_engine::forward::{Forward, arena_holds};
use xabe_engine::weights::DeviceWeights;
use xabe_gguf::GgufFile;
use xabe_model::config::ModelConfig;
use xabe_model::weights::WeightSchema;

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llama.cpp/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

/// Discarded passes before timing starts.
///
/// The first pass pays NVRTC module loads and first-touch page faults on
/// every scratch buffer; the second is representative. Two is what
/// `llama-bench` uses for the same reason.
const WARMUP: usize = 2;

fn env_usize_list(key: &str, default: &[usize]) -> Vec<usize> {
    match std::env::var(key) {
        Ok(v) => v
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .filter(|n| *n > 0)
            .collect(),
        Err(_) => default.to_vec(),
    }
}

/// Mean and sample standard deviation, in milliseconds.
fn stats(samples: &[f64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    if samples.len() < 2 {
        return (mean, 0.0);
    }
    let var = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt())
}

fn main() {
    xabe_log::init_from_args();

    if !driver_available() {
        warn!("SKIPPED: no CUDA driver present");
        return;
    }
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            warn!("SKIPPED: could not create a context on device 0: {e}");
            return;
        }
    };
    let info = DeviceInfo::from_context(0, &ctx).expect("device properties readable");
    if !info.is_supported() {
        warn!("SKIPPED: device 0 is below the sm_75 minimum");
        return;
    }
    let path = std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH));
    if !path.exists() {
        warn!("SKIPPED: model file not found at {}", path.display());
        return;
    }

    let batches = env_usize_list("LLMXABE_BENCH_N", &[1, 19, 128, 512]);
    let reps: usize = std::env::var("LLMXABE_BENCH_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    let file = GgufFile::open(&path).expect("valid GGUF v3");
    let config = ModelConfig::qwen3_6_35b_a3b();
    let stream = ctx.default_stream();
    let (free_at_start, total) = memory_info(&ctx).expect("memory info");

    info!("device: {} ({})", info.name, info.compute_capability);
    info!(
        "model:  {} ({:.2} GiB free of {:.2} GiB)",
        path.display(),
        free_at_start as f64 / (1u64 << 30) as f64,
        total as f64 / (1u64 << 30) as f64,
    );

    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).expect("schema resolves");
    let load = Instant::now();
    let (weights, report) =
        DeviceWeights::load_where(&ctx, &stream, &file, &directory, arena_holds)
            .expect("weight load");
    info!(
        "arena:  {:.3} GiB in {:.1} s ({:.2} GB/s)\n",
        report.bytes as f64 / (1u64 << 30) as f64,
        load.elapsed().as_secs_f64(),
        report.throughput_gb_s(),
    );

    info!(
        "{:>7} | {:>9} | {:>16} | {:>14}",
        "tokens", "build s", "ms/pass", "tok/s"
    );
    info!("{:->7}-+-{:->9}-+-{:->16}-+-{:->14}", "", "", "", "");

    let mut peak_used = 0u64;
    for &n in &batches {
        let built = Instant::now();
        let mut forward = match Forward::new(
            &ctx,
            &stream,
            &file,
            &directory,
            &weights,
            config.clone(),
            n,
        ) {
            Ok(f) => f,
            Err(e) => {
                error!("{n:>7} | FAILED to build: {e}");
                continue;
            }
        };
        let build_s = built.elapsed().as_secs_f64();

        // Arbitrary but in-vocabulary ids. The routing decision depends on the
        // token, and a degenerate all-zero prompt would route every position
        // to the same experts and understate the MoE's real dispatch spread.
        let ids: Vec<i32> = (0..n)
            .map(|i| ((i * 7919 + 1234) % config.vocab_size as usize) as i32)
            .collect();

        let mut failed = false;
        for _ in 0..WARMUP {
            if let Err(e) = forward.run(&stream, &ids, |_, _| {}) {
                error!("{n:>7} | FAILED during warmup: {e}");
                failed = true;
                break;
            }
        }
        if failed {
            continue;
        }
        stream.synchronize().expect("sync after warmup");

        let (free_now, _) = memory_info(&ctx).expect("memory info");
        peak_used = peak_used.max(free_at_start.saturating_sub(free_now));

        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t = Instant::now();
            forward.run(&stream, &ids, |_, _| {}).expect("forward pass");
            stream.synchronize().expect("sync");
            samples.push(t.elapsed().as_secs_f64() * 1e3);
        }

        let (mean, sd) = stats(&samples);
        info!(
            "{n:>7} | {build_s:>9.1} | {mean:>8.2} ± {sd:>5.2} | {:>8.2} ± {:>3.2}",
            n as f64 / (mean / 1e3),
            n as f64 / (mean / 1e3) * (sd / mean),
        );
    }

    info!(
        "\npeak VRAM {:.3} GiB of {:.2} GiB",
        peak_used as f64 / (1u64 << 30) as f64,
        total as f64 / (1u64 << 30) as f64,
    );
    info!(
        "warmup {WARMUP} discarded, {reps} timed repetitions, stream synchronized inside \
         the timed region",
    );
    info!(
        "NOTE: no KV cache and no carried recurrent state — every pass is a cold full \
         forward. Comparable to llama.cpp `pp`, NOT to `tg`.",
    );
}
