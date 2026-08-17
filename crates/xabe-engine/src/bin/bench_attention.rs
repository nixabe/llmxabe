//! Attention alone, at the context depths where the throughput curve bends.
//!
//! `docs/BENCHMARKS.md` establishes that prefill throughput falls from 1.13x
//! llama.cpp at 512 tokens to 0.45x at 131,072, and that the decay is
//! attention: it is the only term whose cost per token grows with the context
//! already in the cache. Every measurement of it so far has come from a whole
//! forward pass, which means a 30 GiB model load and a minute of wall clock per
//! A/B — expensive enough that the honest response to a small change was to not
//! measure it.
//!
//! This runs the attention launch and nothing else, at the geometry Qwen3.6
//! actually has, so a kernel change can be measured in seconds and interleaved
//! against its predecessor rather than compared across a thermal drift.
//!
//! ## What the columns mean
//!
//! `key_offset` is the depth already in the cache; `n_query` is the chunk being
//! prefilled on top of it. A chunked prefill of `C` tokens at chunk width `W`
//! runs this once per `key_offset` in `0, W, 2W, ...`, so the row at
//! `key_offset = C/2` is the average cost of a token in that prefill and the
//! row at `C` is the worst.
//!
//! `GB/s` is the traffic the kernel *issues*, not the traffic the problem
//! needs: one pass of K and V per block, counted per block. That is the number
//! to compare against the card's 672 GB/s, and the gap between it and the
//! `min GB` column — the same traffic counted once per KV head, which is what
//! an ideal kernel would move — is the redundancy the block shape imposes.
//!
//! `TFLOP/s` counts the causal half only, so it is comparable to the ~22
//! TFLOP/s llama.cpp reaches on this card and to the ~65 the fp16 tensor cores
//! peak at.

use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaStream};
use tracing::{error, info};

use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_cuda::kernels::attention::{AttentionKernels, AttnDecodeScratch};

/// Qwen3.6-35B-A3B's Gated Attention geometry.
const Q_HEADS: usize = 16;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 256;

/// The chunk width `Forward` prefills at, so a row is one real launch.
const CHUNK: usize = 512;

/// `key_offset` values: the depth already cached when the chunk arrives.
const DEPTHS: [usize; 7] = [0, 2_048, 8_192, 32_768, 65_536, 98_304, 131_072];

const WARMUP: usize = 2;
const REPS: usize = 5;

fn main() -> ExitCode {
    xabe_log::init_from_args();

    if !driver_available() {
        info!("SKIPPED: no CUDA driver present");
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

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let info = DeviceInfo::from_context(0, &ctx)?;
    info!(
        "device 0: {} ({} SMs, {:.0} GB/s peak)",
        info.name,
        info.sm_count,
        info.peak_bandwidth_gb_s(),
    );

    let stream = ctx.default_stream();
    let kernels = AttentionKernels::new(&ctx, Q_HEADS, KV_HEADS, HEAD_DIM)?;
    let mut dec = AttnDecodeScratch::new(&stream, Q_HEADS, HEAD_DIM)?;

    let max_keys = DEPTHS[DEPTHS.len() - 1] + CHUNK;
    let kv_elems = max_keys * KV_HEADS * HEAD_DIM;
    info!(
        "cache {:.2} GiB, chunk {CHUNK}, tensor cores: {}",
        (2 * kv_elems * size_of::<u16>()) as f64 / (1 << 30) as f64,
        kernels.uses_tensor_cores(CHUNK),
    );

    // Values, not zeros: the softmax is data-dependent and a cache of zeros
    // would give every key the same score. This is a timing harness, so the
    // numbers only have to be plausible in magnitude — correctness belongs to
    // `tests/attention_differential.rs`.
    let q_host: Vec<f32> = (0..CHUNK * Q_HEADS * HEAD_DIM)
        .map(|i| ((i % 97) as f32 - 48.0) / 64.0)
        .collect();
    let kv_host: Vec<u16> = (0..kv_elems)
        .map(|i| (0x3800 | (i % 1024)) as u16)
        .collect();

    let q = stream.clone_htod(q_host.as_slice())?;
    let k = stream.clone_htod(kv_host.as_slice())?;
    let v = stream.clone_htod(kv_host.as_slice())?;
    let mut out = stream.alloc_zeros::<f32>(CHUNK * Q_HEADS * HEAD_DIM)?;
    drop(q_host);
    drop(kv_host);

    info!("");
    info!(
        "{:>10}  {:>8}  {:>9}  {:>9}  {:>9}  {:>8}  {:>8}",
        "key_offset", "n_query", "ms", "GB/s", "issued GB", "min GB", "TFLOP/s",
    );

    for &depth in &DEPTHS {
        let positions = stream.clone_htod([depth as i32].as_slice())?;
        let launch = |stream: &Arc<CudaStream>,
                      dec: &mut AttnDecodeScratch,
                      out: &mut _|
         -> Result<(), Box<dyn std::error::Error>> {
            kernels.forward(stream, dec, &q, &k, &v, out, CHUNK, max_keys, &positions)?;
            Ok(())
        };

        for _ in 0..WARMUP {
            launch(&stream, &mut dec, &mut out)?;
        }
        stream.synchronize()?;

        let start = Instant::now();
        for _ in 0..REPS {
            launch(&stream, &mut dec, &mut out)?;
        }
        stream.synchronize()?;
        let ms = start.elapsed().as_secs_f64() * 1e3 / REPS as f64;

        let (issued, minimum) = traffic(&kernels, depth);
        let flops = causal_flops(depth);
        info!(
            "{depth:>10}  {CHUNK:>8}  {ms:>9.3}  {:>9.1}  {:>9.3}  {:>8.3}  {:>8.2}",
            issued as f64 / (ms * 1e6),
            issued as f64 / 1e9,
            minimum as f64 / 1e9,
            flops / (ms * 1e9),
        );
    }
    Ok(())
}

/// `(bytes the launch issues, bytes the problem needs)`.
///
/// A block streams every key its query tile can see, so the issued figure sums
/// the causal window per tile and multiplies by the blocks sharing that tile;
/// the minimum is one pass over the whole window per KV head. Their ratio is
/// the redundancy the block shape imposes, which is what a change to `MMA_HPB`
/// moves.
fn traffic(kernels: &AttentionKernels, depth: usize) -> (usize, usize) {
    let rows = kernels.query_tile(CHUNK);
    let tiles = CHUNK.div_ceil(rows);
    let heads = kernels.blocks_per_launch(CHUNK) / tiles;
    let per_key = HEAD_DIM * 2 * size_of::<u16>();

    let issued: usize = (0..tiles)
        .map(|t| heads * (depth + (rows * (t + 1)).min(CHUNK)) * per_key)
        .sum();
    (issued, KV_HEADS * (depth + CHUNK) * per_key)
}

/// Multiply-adds under the causal mask, counted as two flops each, over both
/// `Q K^T` and `P V`.
fn causal_flops(depth: usize) -> f64 {
    // Row `i` of the chunk sees `depth + i + 1` keys.
    let visible: f64 = (0..CHUNK).map(|i| (depth + i + 1) as f64).sum();
    2.0 * 2.0 * visible * HEAD_DIM as f64 * Q_HEADS as f64
}
