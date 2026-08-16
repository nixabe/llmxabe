//! Integer tensor cores against the fp32 path, at the shapes that matter.
//!
//! `docs/BENCHMARKS.md` establishes that prefill is compute-bound and that
//! fp32 is within ~2x of its structural ceiling while the gap to llama.cpp is
//! ~5x. This measures the only instruction that changes that arithmetic.
//!
//! The shapes are the Gated DeltaNet projection's real ones: `[tokens, 2048]`
//! against `[8192, 2048]` is `blk.N.attn_qkv.weight`, which is 30 of 40 layers
//! and was 59.1% of a prefill pass before it was tiled.
//!
//! Reported per shape: the fp32 rate, the int8 rate, and the ratio. The int8
//! column includes the activation quantization, because that is a cost the
//! fp32 path does not pay and excluding it would flatter the result.

use std::process::ExitCode;
use std::time::Instant;

use cudarc::driver::CudaContext;
use tracing::{error, info};

use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_cuda::kernels::mma::MmaKernels;

/// `(tokens, output rows, contraction)`.
const SHAPES: [(usize, usize, usize); 4] = [
    (128, 8192, 2048),
    (512, 8192, 2048),
    (512, 2048, 4096),
    (512, 248_320, 2048),
];

const WARMUP: usize = 3;
const REPS: usize = 10;

/// Deterministic bytes in the Q8_0 layout: `{ ggml_half d; int8_t qs[32]; }`.
///
/// Written inline rather than through `xabe_kernels::quant` because this is a
/// timing harness and `xabe-kernels` is a dev-dependency, reachable from
/// `tests/` and not from a binary. Nothing here checks a value — correctness
/// belongs to `tests/mma_differential.rs`, which does use the validated
/// quantizer. The only property this needs is that the bytes are the right
/// shape and the scales are not degenerate, because a stack of zeros would let
/// the hardware skip work and flatter the measurement.
fn q8_0_rows(rows: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut bytes = Vec::with_capacity(rows * k / 32 * 34);
    for _ in 0..rows * k / 32 {
        // A plausible fp16 scale, written as raw bits so this needs no fp16
        // crate. 0x2000..0x27ff is roughly 0.002 to 0.031 — never zero, never
        // subnormal, and never large enough to saturate the accumulator.
        let bits: u16 = 0x2000 | (next() % 0x800) as u16;
        bytes.extend_from_slice(&bits.to_le_bytes());
        for _ in 0..32 {
            bytes.push(((next() % 255) as i32 - 127) as i8 as u8);
        }
    }
    bytes
}

/// IEEE half bits to f32, without a dependency.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1f);
    let man = u32::from(bits & 0x3ff);
    let f = match exp {
        0 if man == 0 => sign,
        0 => {
            // Subnormal: renormalize into a float32 exponent.
            let shift = man.leading_zeros() - 21;
            sign | ((127 - 15 - shift) << 23) | ((man << (shift + 1)) & 0x7f_ffff)
        }
        31 => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(f)
}

fn main() -> ExitCode {
    xabe_log::init_from_args();

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
    let stream = ctx.default_stream();
    let kernels = match MmaKernels::new(&ctx) {
        Ok(k) => k,
        Err(e) => {
            error!("integer MMA kernels did not build: {e}");
            return ExitCode::FAILURE;
        }
    };

    info!("device 0: {}", info.name);
    info!("");
    info!(
        "{:>7} {:>8} {:>6} {:>10} {:>9} {:>10} {:>9} {:>9}",
        "tokens", "rows", "k", "gguf ms", "TOP/s", "split ms", "TOP/s", "% of 198",
    );

    for (tokens, rows, k) in SHAPES {
        let w = q8_0_rows(rows, k, 0x5EED ^ rows as u64);
        let mut st = 0xA11CEu64;
        let x: Vec<f32> = (0..tokens * k)
            .map(|_| {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                (st >> 40) as f32 / 8388608.0 - 2.0
            })
            .collect();

        // The same numbers in the split layout: scales lifted out, quants
        // contiguous. Built here rather than repacked on device because this
        // measures the steady state, and the repack is a one-time load cost.
        let mut wq = Vec::with_capacity(rows * k);
        let mut ws = Vec::with_capacity(rows * k / 32);
        for blk in w.chunks(34) {
            let bits = u16::from_le_bytes([blk[0], blk[1]]);
            ws.push(f16_to_f32(bits));
            wq.extend(blk[2..34].iter().map(|&b| b as i8));
        }

        let d_w = stream.clone_htod(w.as_slice()).expect("weight upload");
        let d_wq = stream.clone_htod(wq.as_slice()).expect("wq upload");
        let d_ws = stream.clone_htod(ws.as_slice()).expect("ws upload");
        let d_x = stream.clone_htod(x.as_slice()).expect("activation upload");
        let mut d_q = stream.alloc_zeros::<i8>(tokens * k).expect("q");
        let mut d_s = stream.alloc_zeros::<f32>(tokens * k / 32).expect("s");
        let mut d_o = stream.alloc_zeros::<f32>(tokens * rows).expect("out");

        let mut once = |kern: &MmaKernels| {
            kern.quantize_rows(&stream, &d_x, &mut d_q, &mut d_s, tokens, k)
                .expect("quantize");
            kern.q8_0_proj(&stream, &d_w, &d_q, &d_s, &mut d_o, k, rows, tokens)
                .expect("project");
        };
        for _ in 0..WARMUP {
            once(&kernels);
        }
        stream.synchronize().expect("sync");

        let t = Instant::now();
        for _ in 0..REPS {
            once(&kernels);
        }
        stream.synchronize().expect("sync");
        let ms = t.elapsed().as_secs_f64() * 1e3 / REPS as f64;

        let mut split = |kern: &MmaKernels| {
            kern.quantize_rows(&stream, &d_x, &mut d_q, &mut d_s, tokens, k)
                .expect("quantize");
            kern.q8_0_proj_split(&stream, &d_wq, &d_ws, &d_q, &d_s, &mut d_o, k, rows, tokens)
                .expect("project");
        };
        for _ in 0..WARMUP {
            split(&kernels);
        }
        stream.synchronize().expect("sync");
        let t2 = Instant::now();
        for _ in 0..REPS {
            split(&kernels);
        }
        stream.synchronize().expect("sync");
        let ms2 = t2.elapsed().as_secs_f64() * 1e3 / REPS as f64;

        // Two operations per multiply-accumulate, as everywhere else here.
        let ops = 2.0 * tokens as f64 * rows as f64 * k as f64;
        let tops = ops / (ms / 1e3) / 1e12;
        let tops2 = ops / (ms2 / 1e3) / 1e12;
        info!(
            "{tokens:>7} {rows:>8} {k:>6} {ms:>10.3} {tops:>9.2} {ms2:>10.3} {tops2:>9.2} {:>8.1}%",
            tops2 / 198.0 * 100.0,
        );
    }

    info!("");
    info!("fp32 peak is 16.3 TFLOP/s; int8 tensor-core peak measured at ~198 TOP/s.");
    info!("The int8 column includes activation quantization, which fp32 does not pay.");
    ExitCode::SUCCESS
}
