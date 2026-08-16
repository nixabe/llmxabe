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
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaStream};
use tracing::{error, info, warn};
use xabe_cuda::arena::memory_info;
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_engine::forward::{Forward, arena_holds};
use xabe_engine::weights::DeviceWeights;
use xabe_gguf::GgufFile;
use xabe_model::config::ModelConfig;
use xabe_model::weights::{Directory, WeightSchema};

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

    // Chunked prefill: run the prompt through a fixed-shape pass repeatedly,
    // carrying the KV cache and the recurrent state, instead of building one
    // pass as wide as the prompt.
    //
    // This is what llama.cpp does with `-ub`, and it is the difference between
    // activation memory that scales with the prompt and activation memory that
    // is constant. The engine already supports it — `Forward::run` runs
    // `self.tokens` positions at `state.position()` and advances the state, and
    // a one-token `Forward` chained this way is exactly what decode is. Only
    // this harness was missing it.
    if let Some(chunk) = env_usize("LLMXABE_BENCH_CHUNK") {
        chunked_prefill(
            &ctx,
            &stream,
            &file,
            &directory,
            &weights,
            &config,
            &batches,
            chunk,
            reps,
            free_at_start,
            total,
        );
        return;
    }

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

        // One state, reset before every pass. This benchmark measures a cold
        // prefill, so each repetition must start from position 0 with a zeroed
        // recurrent state; without the reset the second pass would be a
        // continuation of the first and would measure something else.
        let mut state = match forward.new_state(&stream, n) {
            Ok(s) => s,
            Err(e) => {
                error!("{n:>7} | FAILED to allocate sequence state: {e}");
                continue;
            }
        };

        let mut failed = false;
        for _ in 0..WARMUP {
            state.reset(&stream).expect("reset");
            if let Err(e) = forward.run(&stream, &mut state, &ids, |_, _| {}) {
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
            // The reset is inside the timed region because it used to be
            // inside `run`, and moving it out would make these numbers
            // quietly incomparable with every prefill measurement already in
            // docs/BENCHMARKS.md.
            let t = Instant::now();
            state.reset(&stream).expect("reset");
            forward
                .run(&stream, &mut state, &ids, |_, _| {})
                .expect("forward pass");
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

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok().and_then(|v| v.trim().parse().ok())
}

/// Prefill a prompt as a chain of fixed-shape passes over one sequence state.
///
/// The pass is built once at `chunk` positions and reused for every prompt
/// length, because its shape no longer depends on the prompt — which is the
/// whole point. What scales with the prompt is the `SequenceState`: the KV
/// cache for the ten attention layers, and nothing else, since the Gated
/// DeltaNet's recurrent state is a fixed `[value_heads][head_dim][head_dim]`
/// per layer however long the sequence gets.
#[allow(clippy::too_many_arguments)]
fn chunked_prefill(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    file: &GgufFile,
    directory: &Directory<'_>,
    weights: &DeviceWeights,
    config: &ModelConfig,
    batches: &[usize],
    chunk: usize,
    reps: usize,
    free_at_start: u64,
    total: u64,
) {
    let max_seq = batches.iter().copied().max().unwrap_or(chunk);
    let built = Instant::now();
    let mut forward =
        match Forward::new(ctx, stream, file, directory, weights, config.clone(), chunk) {
            Ok(f) => f,
            Err(e) => {
                error!("FAILED to build the {chunk}-token pass: {e}");
                return;
            }
        };
    let build_s = built.elapsed().as_secs_f64();

    let mut state = match forward.new_state(stream, max_seq) {
        Ok(s) => s,
        Err(e) => {
            error!("FAILED to allocate a {max_seq}-position sequence state: {e}");
            return;
        }
    };

    info!(
        "chunked prefill: {chunk}-token pass built in {build_s:.1} s, state for {max_seq} positions"
    );
    info!(
        "{:>8} | {:>7} | {:>16} | {:>14}",
        "tokens", "chunks", "ms/prompt", "tok/s"
    );
    info!("{:->8}-+-{:->7}-+-{:->16}-+-{:->14}", "", "", "", "");

    let mut peak_used = 0u64;
    for &n in batches {
        if !n.is_multiple_of(chunk) {
            warn!("{n:>8} | skipped: not a multiple of the {chunk}-token chunk");
            continue;
        }
        let chunks = n / chunk;
        // Ids for the whole prompt, sliced per chunk. Same generator as the
        // single-pass path so the routing spread is identical.
        let ids: Vec<i32> = (0..n)
            .map(|i| ((i * 7919 + 1234) % config.vocab_size as usize) as i32)
            .collect();

        let mut run_once = |state: &mut _| -> Result<(), String> {
            for c in 0..chunks {
                let slice = &ids[c * chunk..(c + 1) * chunk];
                forward
                    .run(stream, state, slice, |_, _| {})
                    .map_err(|e| format!("chunk {c} at position {}: {e}", c * chunk))?;
            }
            Ok(())
        };

        state.reset(stream).expect("reset");
        if let Err(e) = run_once(&mut state) {
            error!("{n:>8} | FAILED during warmup: {e}");
            continue;
        }
        stream.synchronize().expect("sync after warmup");
        let (free_now, _) = memory_info(ctx).expect("memory info");
        peak_used = peak_used.max(free_at_start.saturating_sub(free_now));

        let mut samples = Vec::with_capacity(reps);
        let mut failed = false;
        for _ in 0..reps {
            let t = Instant::now();
            state.reset(stream).expect("reset");
            if let Err(e) = run_once(&mut state) {
                error!("{n:>8} | FAILED: {e}");
                failed = true;
                break;
            }
            stream.synchronize().expect("sync");
            samples.push(t.elapsed().as_secs_f64() * 1e3);
        }
        if failed {
            continue;
        }

        let (mean, sd) = stats(&samples);
        info!(
            "{n:>8} | {chunks:>7} | {mean:>9.2} ± {sd:>4.2} | {:>8.2} ± {:>3.2}",
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
        "NOTE: one warmup discarded, {reps} timed repetitions. The prompt is prefilled as \
         {chunk}-token passes over a carried KV cache and recurrent state — the same thing \
         llama.cpp does with `-ub {chunk}`, and directly comparable to its `S_PP`.",
    );
}
