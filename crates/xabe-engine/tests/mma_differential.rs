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
