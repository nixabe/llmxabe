//! The dense feed-forward block alone, at the widths a `qwen35` step runs it.
//!
//! The dense model's decode step is a streaming problem and this block is
//! most of it: 17.20 GiB of the 25.73 GiB a token reads on
//! `Qwen3.8-27B-req-q8_0`, which is 67%. `bench_decode` measures the whole
//! step at 0.06% spread, which is precise enough to *confirm* a change but
//! costs a 30 s model load per side and says nothing about which of the three
//! projections moved. This runs one real layer's `gate`/`up`/`down` and
//! reports each separately, so a change to the GEMV's inner loop is a
//! ten-second A/B.
//!
//! # What the columns mean
//!
//! `GB/s` divides the *resident* weight bytes by the measured time — resident,
//! not as-stored, because the two differ here: `DENSE_REPACK_INT8` holds the
//! split int8 layout, and its scale array is what this benchmark exists to
//! put a number on. The card streams 672 GB/s, so the last column is the
//! fraction of that a perfect implementation of the same reads could not
//! exceed.
//!
//! `x64` scales one layer to the model's 64, which is the figure that
//! compares to `bench_decode`'s per-step millisecond — not equal to it, since
//! a step also pays the mixers and the LM head.
//!
//! # What is not measured
//!
//! One layer, resident, re-read every repetition. At 300 MiB against a 6 MiB
//! L2 that is a cold read every time, which is what a real step does; but a
//! real step also interleaves the mixer's launches, and this does not.
//!
//! ```text
//! CUDA_VISIBLE_DEVICES=1 LLMXABE_MODEL=<dense.gguf> \
//!   cargo run --release -p xabe-engine --bin bench_dense_ffn
//! ```
//!
//! `LLMXABE_MODEL` overrides the path, `LLMXABE_FFN_LAYER` the layer,
//! `LLMXABE_FFN_REPS` the repetitions, `LLMXABE_FFN_N` the comma-separated
//! token counts.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaContext, CudaStream, PushKernelArg};
use tracing::{error, info, warn};

use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_engine::block::dense_ffn::{DenseFfnBlock, DenseFfnLayerWeights};
use xabe_engine::forward::moe_block_size;
use xabe_gguf::GgufFile;
use xabe_model::config::ModelConfig;
use xabe_model::weights::WeightSchema;

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llmxabe/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-req-q8_0.gguf";

/// Quadro RTX 8000, sm_75: GDDR6 streaming peak.
const PEAK_GB_S: f64 = 672.0;

/// Wall-clock spent warming before the clock starts, per token count.
///
/// A repetition count is the wrong unit here. At one token the block is 0.55
/// ms, so three of them is under two milliseconds — far too short for the
/// card to leave its idle clock, and the first row of the table measured 9%
/// slow with a 7.7% spread until this was a duration instead. The card is
/// pinned at its boost clock well inside a quarter second.
const WARMUP_MS: u128 = 400;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_usize_list(key: &str, default: &[usize]) -> Vec<usize> {
    match std::env::var(key) {
        Ok(v) => v
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect::<Vec<_>>(),
        Err(_) => default.to_vec(),
    }
}

/// A pure streaming read over the same bytes, to put a ceiling on the GEMV.
///
/// The GEMV's `GB/s` column means nothing without knowing what this card can
/// actually do on a read stream, and 672 is the pin count, not a measurement.
/// This kernel reads exactly the resident weight bytes with fully coalesced
/// `uint4` loads and no arithmetic beyond the reduction that keeps ptxas from
/// deleting them — the number the GEMV is allowed to be compared against.
const CALIBRATE_SRC: &str = r#"
extern "C" __global__ void stream_read_u4(
    const uint4* __restrict__ src,
    long long n_vec,
    float* __restrict__ sink
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    unsigned int acc = 0u;
    for (; i < n_vec; i += stride) {
        uint4 v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    // Never true, and ptxas cannot prove it.
    if (acc == 0xdeadbeefu && blockIdx.x == 0xffffffu) sink[0] = (float)acc;
}
"#;

fn main() -> ExitCode {
    xabe_log::init_from_args();
    if !driver_available() {
        warn!("SKIPPED: no CUDA driver on this host");
        return ExitCode::SUCCESS;
    }
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            error!("{e}");
            ExitCode::FAILURE
        }
    }
}

/// Mean and sample standard deviation, in the same units as the input.
fn stats(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, var.sqrt())
}

/// Time `reps` calls of `body`, each bracketed by CUDA events on `stream`.
///
/// Events rather than `Instant` for the reason AGENTS.md gives: every launch
/// is asynchronous, so a host clock around one repetition measures enqueue
/// latency. Each repetition is timed on its own so the report can carry a
/// spread rather than only a mean.
fn time_reps<F>(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    reps: usize,
    mut body: F,
) -> Result<Vec<f64>, Box<dyn std::error::Error>>
where
    F: FnMut() -> Result<(), Box<dyn std::error::Error>>,
{
    let start = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let stop = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let t0 = Instant::now();
    while t0.elapsed().as_millis() < WARMUP_MS {
        body()?;
        stream.synchronize()?;
    }
    let mut out = Vec::with_capacity(reps);
    for _ in 0..reps {
        start.record(stream)?;
        body()?;
        stop.record(stream)?;
        stream.synchronize()?;
        out.push(f64::from(start.elapsed_ms(&stop)?));
    }
    Ok(out)
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let info = DeviceInfo::from_context(0, &ctx)?;
    if !info.is_supported() {
        warn!("SKIPPED: device 0 is below the sm_75 minimum");
        return Ok(());
    }
    let stream = ctx.default_stream();

    let path = std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH));
    if !path.exists() {
        warn!("SKIPPED: model file not found at {}", path.display());
        return Ok(());
    }
    let file = GgufFile::open(&path)?;
    let config = ModelConfig::from_gguf(&file)?;
    let Some(dense) = config.dense_ffn() else {
        warn!(
            "SKIPPED: {} is not a dense model; this benchmark measures the `qwen35` FFN",
            path.display()
        );
        return Ok(());
    };
    let layers = config.num_blocks();
    let layer = env_usize("LLMXABE_FFN_LAYER", 0).min(layers as usize - 1) as u32;
    let reps = env_usize("LLMXABE_FFN_REPS", 12);
    let batches = env_usize_list("LLMXABE_FFN_N", &[1, 2, 3, 4, 512]);

    let schema = WeightSchema::with_mtp(&config);
    let directory = schema.resolve(&file).map_err(|e| {
        format!(
            "{} tensor(s) did not match the dense schema; first is {:?}",
            e.len(),
            e.first()
        )
    })?;

    info!(
        "device 0: {} sm_{}",
        info.name,
        info.compute_capability.to_string().replace('.', "")
    );
    info!(
        "model:    {} -- {} layers, hidden {}, intermediate {}",
        path.display(),
        layers,
        config.hidden_size,
        dense.intermediate,
    );

    // One layer, both residencies decided by `DenseFfnBlock::new`'s `mma`
    // flag exactly as the serving path decides it.
    let geom_of = |tokens: usize| {
        DenseFfnBlock::geometry_for(&config, moe_block_size(tokens), tokens)
            .expect("dense_ffn() returned Some above")
    };
    let weights = DenseFfnLayerWeights::upload(
        &stream,
        &file,
        &directory,
        layer,
        &geom_of(1),
        xabe_engine::block::dense_ffn::DENSE_REPACK_INT8,
    )?;
    let resident = weights.bytes();
    info!(
        "layer {layer}: {:.1} MiB resident as {}\n",
        resident as f64 / (1024.0 * 1024.0),
        if weights.has_int8() {
            "split int8"
        } else {
            "the file's own quant"
        },
    );

    // The ceiling first, so every row below can be read against it.
    {
        let ptx = xabe_cuda::kernels::compile(CALIBRATE_SRC, "dense_ffn_calibrate")?;
        let module = ctx.load_module(ptx)?;
        let f = module.load_function("stream_read_u4")?;
        let n_vec = resident / 16;
        let src = stream.alloc_zeros::<u8>(n_vec * 16)?;
        let mut sink = stream.alloc_zeros::<f32>(1)?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (72 * 8, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let n = n_vec as i64;
        let samples = time_reps(&ctx, &stream, reps, || {
            let mut b = stream.launch_builder(&f);
            b.arg(&src).arg(&n).arg(&mut sink);
            // SAFETY: a grid-stride loop bounded by `n_vec`, over a buffer of
            // exactly `n_vec` 16-byte vectors; `sink` is never written.
            unsafe { b.launch(cfg) }?;
            Ok(())
        })?;
        let (ms, sd) = stats(&samples);
        info!(
            "calibration: a coalesced read of the same {:.1} MiB streams {:.1} GB/s, \
             {:.1}% of the {PEAK_GB_S} GB/s pin rate (sd {:.2}%)\n",
            resident as f64 / (1024.0 * 1024.0),
            resident as f64 / (ms * 1e-3) / 1e9,
            resident as f64 / (ms * 1e-3) / 1e9 / PEAK_GB_S * 100.0,
            sd / ms * 100.0,
        );
    }

    info!(
        "{:>6}  {:>10}  {:>8}  {:>10}  {:>8}  {:>7}",
        "tokens", "ms/layer", "sd %", "x64 (ms)", "GB/s", "of peak"
    );
    info!(
        "{:->6}--{:->10}--{:->8}--{:->10}--{:->8}--{:->7}",
        "", "", "", "", "", ""
    );

    for &tokens in &batches {
        let geometry = geom_of(tokens);
        let eps = 1e-6;
        let mut block = DenseFfnBlock::new(&ctx, &stream, geometry, eps, true)?;
        let n = geometry.max_tokens * geometry.hidden;
        let residual = stream.alloc_zeros::<f32>(n)?;
        let mut ffn_out = stream.alloc_zeros::<f32>(n)?;
        let mut l_out = stream.alloc_zeros::<f32>(n)?;
        // A realistic activation magnitude: a normed row is unit-ish, and a
        // buffer of zeros would let the int8 activation quantizer pick a
        // degenerate scale.
        let host: Vec<f32> = (0..n)
            .map(|i| ((i % 97) as f32 - 48.0) / 48.0)
            .collect::<Vec<_>>();
        let residual = {
            let mut r = residual;
            stream.memcpy_htod(&host, &mut r)?;
            r
        };

        let samples = time_reps(&ctx, &stream, reps, || {
            block.forward(
                &stream,
                &weights,
                &residual,
                tokens,
                &mut ffn_out,
                &mut l_out,
            )?;
            Ok(())
        })?;
        let (ms, sd) = stats(&samples);
        let gbs = resident as f64 / (ms * 1e-3) / 1e9;
        info!(
            "{tokens:>6}  {ms:>10.3}  {:>8.2}  {:>10.2}  {gbs:>8.1}  {:>6.1}%",
            sd / ms * 100.0,
            ms * layers as f64,
            gbs / PEAK_GB_S * 100.0,
        );
    }

    info!(
        "\n{WARMUP_MS} ms of warmup discarded, {reps} repetitions timed with CUDA events. `GB/s` is the \
         resident weight bytes over the measured time; at <= 4 tokens the block is the split \
         GEMV and above it the tensor-core GEMM, which is not a streaming kernel and whose \
         percentage is not a target."
    );
    Ok(())
}
