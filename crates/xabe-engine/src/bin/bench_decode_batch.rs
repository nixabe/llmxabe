//! Aggregate decode throughput across batched sequences: the number that
//! compares to `llama-batched-bench -npl N`.
//!
//! `bench_decode` measures one sequence decoding alone — 77.4 tok/s at a
//! 32,768-token context, `docs/BENCHMARKS.md`'s baseline. This measures `N`
//! sequences decoding *together* through [`Forward::run_batch_decode`], which
//! is the feature that number has no answer to: llama.cpp reaches 154.58
//! tok/s aggregate at three parallel sequences over the same context by
//! batching them, and this engine had nothing that did that until now.
//!
//! # What is and is not measured
//!
//! Two different things are timed, and they should not be confused:
//!
//! - **`single_stream`**: one sequence, decoded through
//!   [`Forward::capture_step`]/[`Forward::replay_step`] — the CUDA-graph path
//!   `bench_decode` also uses. This is the baseline the batched numbers must
//!   beat.
//! - **`batch N`**: `N` sequences, decoded together through
//!   [`Forward::capture_batch_step`]/[`Forward::replay_batch_step`] — the same
//!   graph-capture trade, one level up. An uncaptured batched step issues `N`
//!   small per-sequence launches for the Gated Attention loop and for the two
//!   steps `GdnBlock::forward_batch_decode` cannot batch, on top of every
//!   already-batched call; `tests/batch_decode.rs`'s
//!   `a_captured_batch_step_generates_the_same_sequence_as_the_launch_path`
//!   is what gates the capture against the uncaptured path it replaces here.
//!
//! Both include greedy sampling in the timed region, for the same reason
//! `bench_decode` does: an autoregressive step is not finished until the host
//! knows what to feed back in.
//!
//! ```sh
//! cargo run --release -p xabe-engine --bin bench_decode_batch -- [context] [steps]
//! ```
//!
//! Environment: `LLMXABE_MODEL` overrides the model path,
//! `LLMXABE_BATCH_N` overrides the comma-separated batch widths (default
//! `1,2,3,4,8`), `LLMXABE_DECODE_CHUNK` overrides the prefill chunk width.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use cudarc::driver::CudaContext;
use tracing::{error, info, warn};

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

/// Decode steps discarded before timing starts, per shape. See
/// `bench_decode`: the first step after a prefill pays for whatever the
/// driver still has in flight from it.
const WARMUP: usize = 4;

const DEFAULT_CONTEXT: usize = 2048;
const DEFAULT_STEPS: usize = 32;
const DEFAULT_BATCH_WIDTHS: &[usize] = &[1, 2, 3, 4, 8];

/// Largest single prefill pass. Wider contexts are prefilled in chunks of
/// this width so scratch stays bounded — see `bench_decode`'s identical
/// reasoning.
// A 2,048-row MoE prefill workspace cannot coexist with the resident 35B
// weights and a decode shape on the 48 GiB deployment card. Prefill happens
// before timing, so use the largest setup chunk that leaves room for the
// actual N-way decode measurement. The resulting recurrent/KV state is the
// same sequence of chunked prefill operations production uses.
const MAX_PREFILL_CHUNK: usize = 512;

fn model_path() -> PathBuf {
    std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
}

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

fn stats(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, var.sqrt())
}

/// A chunk width that divides `context` and is at most `max_chunk`.
fn chunk_for(context: usize, max_chunk: usize) -> usize {
    let mut c = context.min(max_chunk);
    while c > 1 && !context.is_multiple_of(c) {
        c -= 1;
    }
    c
}

fn decode_chunk_limit() -> usize {
    std::env::var("LLMXABE_DECODE_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&chunk| chunk > 0)
        .unwrap_or(MAX_PREFILL_CHUNK)
}

/// One synthetic, in-vocabulary, non-degenerate prompt. Distinct per sequence
/// so a cross-sequence indexing bug would show up as a wrong answer rather
/// than being hidden behind identical inputs — the same reasoning
/// `tests/batch_decode.rs` documents.
fn synthetic_prompt(seq: usize, len: usize, vocab: usize) -> Vec<i32> {
    (0..len)
        .map(|i| (((seq + 1) * 104_729 + i * 7919 + 1234) % vocab) as i32)
        .collect()
}

fn main() -> ExitCode {
    let rest = xabe_log::init_from_args();
    let nums: Vec<usize> = rest.iter().filter_map(|a| a.parse().ok()).collect();
    let context = nums.first().copied().unwrap_or(DEFAULT_CONTEXT);
    let steps = nums.get(1).copied().unwrap_or(DEFAULT_STEPS);
    let batch_widths = env_usize_list("LLMXABE_BATCH_N", DEFAULT_BATCH_WIDTHS);

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
    // A stream of its own: the single-stream baseline captures a step, and
    // `cuStreamBeginCapture` rejects the legacy default stream.
    let stream = ctx.new_stream().expect("create stream");
    // SAFETY: as `bench_decode` — this process has exactly one stream, so
    // there is no cross-stream ordering for cudarc's per-slice event tracking
    // to manage, and capture and tracking cannot both be on. Must run before
    // the first allocation.
    unsafe { ctx.disable_event_tracking() };
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
    info!(
        "arena {:.3} GiB in {:.1} s",
        load.bytes as f64 / (1u64 << 30) as f64,
        load.elapsed.as_secs_f64(),
    );

    let chunk = chunk_for(context, decode_chunk_limit());
    if chunk != context {
        info!("prefill chunk {chunk} (context {context} is not <= {MAX_PREFILL_CHUNK})");
    }
    let max_seq = context + WARMUP + steps + 1;

    info!("");
    info!("context {context}, {steps} steps timed after {WARMUP} warmup");
    info!("");
    info!(
        "{:<16} {:>10} {:>14} {:>16} {:>14}",
        "shape", "mean ms", "aggregate tok/s", "per-seq tok/s", "peak VRAM GiB"
    );

    // ---- The baseline: one sequence, graph-captured, exactly `bench_decode`'s
    // methodology. -----------------------------------------------------------
    // `LLMXABE_SKIP_SINGLE_STREAM` is a profiling aid only: it drops this
    // block's kernels out of an `nsys` capture so a per-kernel breakdown of
    // the batch sweep below is not diluted by the N=1 baseline's own calls.
    if std::env::var_os("LLMXABE_SKIP_SINGLE_STREAM").is_none() {
        let mut prefill = Forward::new(
            &ctx,
            &stream,
            &file,
            &directory,
            &weights,
            config.clone(),
            chunk,
        )
        .expect("prefill pass builds");
        let mut step = prefill
            .reshape(&ctx, &stream, &file, &directory, &weights, 1)
            .expect("decode step builds");
        // Same A/B lever bench_decode carries: force the warp decode kernel
        // over the depth-dispatched tensor-core one.
        if std::env::var("LLMXABE_DISABLE_DECODE_MMA").is_ok() {
            step.disable_decode_mma();
        }
        let mut state = prefill
            .new_state(&stream, max_seq)
            .expect("state allocates");

        let ids = synthetic_prompt(0, context, config.vocab_size as usize);
        for piece in ids.chunks(chunk) {
            prefill
                .run(&stream, &mut state, piece, |_, _| {})
                .expect("prefill runs");
        }
        stream.synchronize().expect("sync");
        let mut next = prefill.sample_argmax(&stream).expect("prefill argmax");

        let graph = step
            .capture_step(&stream, &mut state)
            .expect("capture the decode step");
        for _ in 0..WARMUP {
            next = step
                .replay_step(&stream, &mut state, &graph, &[next])
                .expect("warmup decode step");
        }

        let mut samples = Vec::with_capacity(steps);
        for _ in 0..steps {
            let t = Instant::now();
            next = step
                .replay_step(&stream, &mut state, &graph, &[next])
                .expect("timed decode step");
            samples.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let (mean, _sd) = stats(&samples);
        let (free_now, _) = memory_info(&ctx).expect("memory info");
        info!(
            "{:<16} {:>10.2} {:>14.1} {:>16.1} {:>14.3}",
            "single_stream",
            mean,
            1e3 / mean,
            1e3 / mean,
            free_at_start.saturating_sub(free_now) as f64 / (1u64 << 30) as f64,
        );
    }

    // ---- The batch sweep: `N` sequences advanced together through
    // `run_batch_decode`, for every width in `batch_widths`. -----------------
    for &n in &batch_widths {
        let mut prefill = Forward::new(
            &ctx,
            &stream,
            &file,
            &directory,
            &weights,
            config.clone(),
            chunk,
        )
        .expect("prefill pass builds");
        let mut batch = prefill
            .reshape(&ctx, &stream, &file, &directory, &weights, n)
            .expect("batch-width pass builds");
        batch
            .enable_batch_decode(&ctx, &stream)
            .expect("batch decode scratch allocates");
        if std::env::var("LLMXABE_DISABLE_DECODE_MMA").is_ok() {
            batch.disable_decode_mma();
        } else if let Ok(wpo) = std::env::var("LLMXABE_DECODE_MMA_WPO")
            && let Ok(wpo) = wpo.parse::<usize>()
        {
            batch.set_decode_mma_wpo(wpo);
        }

        let mut states: Vec<SequenceState> = Vec::with_capacity(n);
        let mut next: Vec<i32> = Vec::with_capacity(n);
        for seq in 0..n {
            let mut state = prefill
                .new_state(&stream, max_seq)
                .expect("state allocates");
            let ids = synthetic_prompt(seq, context, config.vocab_size as usize);
            for piece in ids.chunks(chunk) {
                prefill
                    .run(&stream, &mut state, piece, |_, _| {})
                    .expect("prefill runs");
            }
            stream.synchronize().expect("sync");
            next.push(prefill.sample_argmax(&stream).expect("prefill argmax"));
            states.push(state);
        }

        // Captured once, exactly as the single-stream baseline above: a
        // batched step issues `n` small per-sequence launches on top of every
        // already-batched call, and a capture amortizes that issue cost the
        // same way `capture_step` does for one sequence.
        let graph = batch
            .capture_batch_step(&stream, &mut states)
            .expect("capture the batched decode step");

        for _ in 0..WARMUP {
            next = batch
                .replay_batch_step(&stream, &mut states, &graph, &next)
                .expect("warmup batch decode step");
        }

        let mut samples = Vec::with_capacity(steps);
        for _ in 0..steps {
            let t = Instant::now();
            next = batch
                .replay_batch_step(&stream, &mut states, &graph, &next)
                .expect("timed batch decode step");
            samples.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let (mean, _sd) = stats(&samples);
        let (free_now, _) = memory_info(&ctx).expect("memory info");
        info!(
            "{:<16} {:>10.2} {:>14.1} {:>16.1} {:>14.3}",
            format!("batch {n}"),
            mean,
            n as f64 * 1e3 / mean,
            1e3 / mean,
            free_at_start.saturating_sub(free_now) as f64 / (1u64 << 30) as f64,
        );
    }

    info!("");
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::chunk_for;

    #[test]
    fn prefill_chunk_is_a_context_divisor_within_its_limit() {
        assert_eq!(chunk_for(2048, 512), 512);
        assert_eq!(chunk_for(2048, 750), 512);
        assert_eq!(chunk_for(513, 512), 171);
        assert_eq!(chunk_for(127, 512), 127);
    }
}
