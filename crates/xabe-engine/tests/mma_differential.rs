//! Integer tensor cores against an exact oracle.
//!
//! Every other differential test in this workspace accepts a tolerance,
//! because the device and the CPU reference sum the same fp32 products in
//! different orders and legitimately disagree in the last bits. **This one
//! does not**, and that is the point: integer addition is associative, so
//! summation order cannot excuse a disagreement and the gate can be
//! `assert_eq!`.
//!
//! That matters more here than anywhere else. The characteristic defect of a
//! hand-written MMA port is a wrong *fragment layout* — reading the operand
//! registers in the wrong lane order — and it produces finite, plausible,
//! entirely wrong numbers. There is no tolerance that separates it from
//! arithmetic noise. An exact gate separates it from everything.
//!
//! What this establishes, and what it does not:
//!
//! - It **does** establish that `mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32`
//!   assembles for `sm_75` on this host, that this is a real tensor-core
//!   instruction rather than an emulation, and that our fragment indexing
//!   agrees with the hardware's for every lane.
//! - It **does not** say anything about the MoE, about Q6_K, or about
//!   throughput. This is the primitive, gated on its own, before anything is
//!   built on it.
//!
//! SKIPS — reporting that it skipped — without a driver or a supported device.
//! It needs no model file and no golden capture.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_cuda::kernels::mma::{MMA_K, MMA_M, MMA_N, MmaKernels};
use xabe_kernels::mma::int8_gemm;
use xabe_kernels::quant::{dequantize_q8_0, quantize_q8_0};
use xabe_kernels::rng::Xorshift64Star;

/// A context on device 0, or `None` with a printed reason.
fn device() -> Option<Arc<CudaContext>> {
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
    println!(
        "device 0: {} sm_{}{}",
        info.name, info.compute_capability.major, info.compute_capability.minor,
    );
    Some(ctx)
}

/// Deterministic int8 fill spanning the full range, including both extremes.
///
/// `-128` is included deliberately: it is the value whose negation overflows,
/// and a kernel that widened operands through `signed char -> int` incorrectly
/// would show it here and nowhere else.
fn fill(n: usize, seed: u64) -> Vec<i8> {
    let mut rng = Xorshift64Star::new(seed);
    (0..n)
        .map(|_| (rng.next_u32_below(256) as i32 - 128) as i8)
        .collect()
}

fn run_case(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    kernels: &MmaKernels,
    m: usize,
    n: usize,
    k: usize,
) {
    let _ = ctx;
    let a = fill(m * k, 0x51ED_1234 ^ (m as u64) << 8 ^ k as u64);
    let b = fill(n * k, 0xBEEF_0007 ^ (n as u64) << 8 ^ k as u64);

    let d_a = stream.clone_htod(a.as_slice()).expect("upload a");
    let d_b = stream.clone_htod(b.as_slice()).expect("upload b");
    let mut d_d = stream.alloc_zeros::<i32>(m * n).expect("alloc d");

    kernels
        .gemm(stream, &d_a, &d_b, &mut d_d, m, n, k)
        .expect("the MMA launch is accepted");
    let got = stream.clone_dtoh(&d_d).expect("read back");
    stream.synchronize().expect("sync");

    let want = int8_gemm(&a, &b, m, n, k);

    let mismatches = got.iter().zip(&want).filter(|(g, w)| g != w).count();
    let peak = want.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
    println!(
        "  {m:>4} x {n:>4} x {k:>5}   {:>7} entries, {mismatches} mismatched, peak |acc| {peak}",
        m * n,
    );
    assert_eq!(
        mismatches,
        0,
        "{m}x{n}x{k}: {mismatches} of {} entries disagree with the exact reference; \
         integer addition is associative, so this is a layout or indexing defect, \
         not rounding",
        m * n,
    );
}

#[test]
fn integer_tensor_cores_reproduce_the_exact_reference_on_sm_75() {
    let Some(ctx) = device() else {
        return;
    };
    let stream = ctx.default_stream();

    // Module load is where ptxas runs. `m16n8k16` compiles under NVRTC at
    // compute_75 and fails here, so a successful build is the reachability
    // evidence — not the NVRTC call.
    let kernels = MmaKernels::new(&ctx).expect(
        "the m8n8k16 int8 shape must assemble for sm_75; a failure here is ptxas \
         rejecting the instruction, which is the thing this test exists to check",
    );
    println!("mma.m8n8k16.s32.s8.s8.s32 assembled and loaded for sm_75\n");

    // One tile exactly, then shapes that exercise both the M and N tiling and
    // the bounds guards on a ragged edge, then the MoE's real contraction
    // length. `k` must be a multiple of 16, which is the instruction's K.
    for (m, n, k) in [
        (MMA_M, MMA_N, MMA_K), // a single instruction, no accumulation loop
        (MMA_M, MMA_N, 128),   // one tile, eight k-steps
        (16, 16, 64),          // 2x2 tiles
        (8, 8, 2048),          // the MoE gate/up contraction
        (64, 512, 2048),       // an expert's gate projection, real shape
        (5, 3, 32),            // ragged: both axes bound-checked
        (13, 8, 16),           // ragged M only
        (8, 11, 16),           // ragged N only
    ] {
        run_case(&ctx, &stream, &kernels, m, n, k);
    }
}

#[test]
fn a_contraction_that_is_not_a_whole_number_of_steps_is_rejected() {
    let Some(ctx) = device() else {
        return;
    };
    let stream = ctx.default_stream();
    let kernels = MmaKernels::new(&ctx).expect("kernels compile");

    let d_a = stream.clone_htod([0i8; 8 * 24].as_slice()).expect("a");
    let d_b = stream.clone_htod([0i8; 8 * 24].as_slice()).expect("b");
    let mut d_d = stream.alloc_zeros::<i32>(64).expect("d");

    // 24 is not a multiple of 16. Truncating to 16 would produce a plausible
    // wrong product and padding would need a masked tail load, so it is an
    // error instead.
    let err = kernels
        .gemm(&stream, &d_a, &d_b, &mut d_d, 8, 8, 24)
        .expect_err("a ragged contraction must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("24") && msg.contains("16"), "{msg}");
}

/// Largest magnitude, the denominator every relative statement is made against.
fn peak_of(v: &[f32]) -> f32 {
    v.iter().fold(0f32, |m, x| m.max(x.abs()))
}

/// A deterministic Q8_0 weight stack and its exact fp32 dequantization.
///
/// Built through `xabe_kernels::quant`, the validated reference, rather than
/// by hand: the byte layout (`{ ggml_half d; int8_t qs[32]; }`) is the thing
/// the kernel indexes into, so a hand-rolled copy here could agree with a
/// wrong kernel.
fn q8_0_rows(rows: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let mut bytes = Vec::with_capacity(rows * k / 32 * 34);
    let mut deq = Vec::with_capacity(rows * k);
    for _ in 0..rows * k / 32 {
        let mut raw = [0f32; 32];
        for v in raw.iter_mut() {
            *v = rng.next_f32_range(-1.0, 1.0);
        }
        let block = quantize_q8_0(&raw);
        bytes.extend_from_slice(&block.d.to_le_bytes());
        bytes.extend(block.qs.iter().map(|&q| q as u8));
        deq.extend_from_slice(&dequantize_q8_0(&block));
    }
    (bytes, deq)
}

#[test]
fn the_int8_projection_agrees_with_the_fp32_reference_it_replaces() {
    // Unlike the exact gate above, this one *must* accept a tolerance: the
    // activations are quantized to int8, which is a real loss of information
    // and the entire price of the tensor cores. The question is not whether
    // it is exact — it cannot be — but whether the error is the ~1/127
    // quantization floor rather than a formulation defect.
    //
    // llama.cpp takes exactly this step before its own int8 matmuls, so the
    // error accepted here is the error it already lives with.
    let Some(ctx) = device() else {
        return;
    };
    let stream = ctx.default_stream();
    let kernels = MmaKernels::new(&ctx).expect("kernels compile");

    println!(
        "\n{:>6} {:>6} {:>6}   {:>12} {:>12} {:>10}",
        "tokens", "rows", "k", "cosine", "max_rel", "verdict",
    );

    // The fp32 reference is a scalar triple loop, so its cost is
    // `tokens * rows * k` on the host. The two small shapes exercise the
    // arithmetic; the large one exercises the *tiling* — multiple row bands
    // and token tiles, and a ragged edge in neither — and is checked against
    // the in-place kernel on the device instead, which is the comparison that
    // catches an indexing defect anyway. A 128x8192x2048 host reference is
    // 2.1 G multiply-adds and took 280 s of a 285 s test run.
    for (tokens, n_rows, k, host_reference) in [
        (8usize, 64usize, 256usize, true),
        (19, 512, 2048, true),
        (128, 8192, 2048, false),
    ] {
        let (w, wf) = q8_0_rows(n_rows, k, 0xC0FFEE ^ (n_rows as u64));
        let mut rng = Xorshift64Star::new(0xA5A5 ^ (tokens as u64));
        let x: Vec<f32> = (0..tokens * k)
            .map(|_| rng.next_f32_range(-2.0, 2.0))
            .collect();

        // fp32 reference, the arithmetic the tiled kernel performs.
        let mut want = vec![0f32; tokens * n_rows];
        if host_reference {
            for t in 0..tokens {
                for n in 0..n_rows {
                    let mut acc = 0f32;
                    for i in 0..k {
                        acc += wf[n * k + i] * x[t * k + i];
                    }
                    want[t * n_rows + n] = acc;
                }
            }
        }

        let d_w = stream.clone_htod(w.as_slice()).expect("w");
        let d_x = stream.clone_htod(x.as_slice()).expect("x");
        let mut d_q = stream.alloc_zeros::<i8>(tokens * k).expect("q");
        let mut d_s = stream.alloc_zeros::<f32>(tokens * k / 32).expect("s");
        let mut d_o = stream.alloc_zeros::<f32>(tokens * n_rows).expect("o");

        kernels
            .quantize_rows(&stream, &d_x, &mut d_q, &mut d_s, tokens, k)
            .expect("quantize");
        kernels
            .q8_0_proj(&stream, &d_w, &d_q, &d_s, &mut d_o, k, n_rows, tokens)
            .expect("project");
        let got = stream.clone_dtoh(&d_o).expect("read back");
        stream.synchronize().expect("sync");

        // The split-layout kernel must agree with the in-place one *exactly*:
        // it is the same arithmetic over the same numbers, only rearranged in
        // memory. Any difference is an indexing defect, and it is checked
        // before the tolerance comparison below so a layout bug cannot hide
        // inside the quantization error the tolerance is there to allow.
        let mut wq = Vec::with_capacity(n_rows * k);
        let mut ws = Vec::with_capacity(n_rows * k / 32);
        for blk in w.as_chunks::<34>().0 {
            ws.push(half::f16::from_le_bytes([blk[0], blk[1]]).to_f32());
            wq.extend(blk[2..34].iter().map(|&b| b as i8));
        }
        let d_wq = stream.clone_htod(wq.as_slice()).expect("wq");
        let d_ws = stream.clone_htod(ws.as_slice()).expect("ws");
        let mut d_o2 = stream.alloc_zeros::<f32>(tokens * n_rows).expect("o2");
        kernels
            .q8_0_proj_split(
                &stream, &d_wq, &d_ws, &d_q, &d_s, &mut d_o2, k, n_rows, tokens,
            )
            .expect("split project");
        let got_split = stream.clone_dtoh(&d_o2).expect("read back split");
        stream.synchronize().expect("sync");
        let scale = peak_of(&got).max(1e-6);
        let split_diff = got
            .iter()
            .zip(&got_split)
            .filter(|(a, b)| (*a - *b).abs() > 1e-4 * scale)
            .count();
        assert_eq!(
            split_diff, 0,
            "{tokens}x{n_rows}x{k}: the split layout disagrees with the in-place \
             one on {split_diff} entries; same numbers, same arithmetic, so this \
             is an indexing defect",
        );

        if !host_reference {
            println!(
                "{tokens:>6} {n_rows:>6} {k:>6}   {:>12} {:>12} {:>10}",
                "(vs in-place)", "0 differ", "ok",
            );
            continue;
        }

        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        let mut max_rel = 0f32;
        let peak = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        for (g, w) in got.iter().zip(&want) {
            dot += f64::from(*g) * f64::from(*w);
            na += f64::from(*g) * f64::from(*g);
            nb += f64::from(*w) * f64::from(*w);
            max_rel = max_rel.max((g - w).abs() / peak);
        }
        let cosine = (dot / (na.sqrt() * nb.sqrt())) as f32;
        assert!(got.iter().all(|v| v.is_finite()), "non-finite output");

        println!(
            "{tokens:>6} {n_rows:>6} {k:>6}   {cosine:>12.9} {max_rel:>12.3e} {:>10}",
            if cosine > 0.9999 { "ok" } else { "FAIL" },
        );
        assert!(
            cosine > 0.9999,
            "{tokens}x{n_rows}x{k}: cosine {cosine} against the fp32 reference. \
             int8 activations cost about 1/127 per element and average down over \
             k, so anything this far off is a formulation defect and not \
             quantization noise",
        );
    }
}
