//! Milestone 00 — the toolchain gating spike.
//!
//! Before writing kernels, four questions had to be answered on the actual
//! target hardware, because a "no" on any of them changes the toolchain
//! decision rather than merely inconveniencing it:
//!
//! 1. Does the toolchain emit sm_75 PTX at all?
//! 2. Is there an inline-PTX escape hatch inside kernels? Without one you
//!    cannot reach `mma.sync.m16n8k8.f16` or `ldmatrix`, and you cannot route
//!    around a codegen bug.
//! 3. Do warp shuffle and vote intrinsics lower correctly on sm_75? The MoE
//!    router's top-k over 256 experts is a warp-level bitonic selection, so
//!    these are load-bearing, not conveniences.
//! 4. Are CUDA Graphs reachable? Graph capture over on-device MoE indirection
//!    is the thesis of the whole project (milestone 06). If it were not
//!    reachable from Rust the architecture would need rethinking.
//!
//! Each check below actually compiles, launches, and verifies numerical
//! output. None of them is satisfied by "the API exists".
//!
//! The outcome is recorded in `docs/TOOLCHAIN.md`.

use crate::device::DeviceInfo;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg, sys};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

/// Outcome of a single gating check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckOutcome {
    /// The capability is present and was demonstrated by running code.
    Pass(String),
    /// The capability is absent. Carries what that implies for the design.
    Fail(String),
    /// The check could not run — no driver, no device, no toolkit.
    ///
    /// Deliberately distinct from `Pass`. A check that did not run has not
    /// passed, and reporting it as green is how a toolchain decision gets made
    /// on no evidence.
    Skipped(String),
}

impl CheckOutcome {
    /// Whether this outcome demonstrates the capability.
    pub fn is_pass(&self) -> bool {
        matches!(self, Self::Pass(_))
    }

    /// Whether the check ran at all.
    pub fn ran(&self) -> bool {
        !matches!(self, Self::Skipped(_))
    }
}

impl core::fmt::Display for CheckOutcome {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Pass(d) => write!(f, "PASS    {d}"),
            Self::Fail(d) => write!(f, "FAIL    {d}"),
            Self::Skipped(d) => write!(f, "SKIPPED {d}"),
        }
    }
}

/// The full result of the gating spike.
#[derive(Debug, Clone)]
pub struct SpikeReport {
    /// Q1: sm_75 PTX generation.
    pub emits_sm75_ptx: CheckOutcome,
    /// Q2: inline-PTX escape hatch inside kernels.
    pub inline_ptx: CheckOutcome,
    /// Q3: warp shuffle and vote intrinsics.
    pub warp_intrinsics: CheckOutcome,
    /// Q4: CUDA graph capture and replay.
    pub cuda_graphs: CheckOutcome,
}

impl SpikeReport {
    /// Whether every check that ran passed.
    ///
    /// Skipped checks do not count toward success; consult [`Self::all_ran`].
    pub fn all_ran_checks_passed(&self) -> bool {
        self.checks()
            .iter()
            .filter(|c| c.ran())
            .all(|c| c.is_pass())
    }

    /// Whether every check actually executed.
    pub fn all_ran(&self) -> bool {
        self.checks().iter().all(|c| c.ran())
    }

    /// The checks in gating order.
    pub fn checks(&self) -> [&CheckOutcome; 4] {
        [
            &self.emits_sm75_ptx,
            &self.inline_ptx,
            &self.warp_intrinsics,
            &self.cuda_graphs,
        ]
    }

    /// Labels matching [`Self::checks`].
    pub const LABELS: [&'static str; 4] = [
        "Q1 sm_75 PTX emission",
        "Q2 inline PTX escape hatch",
        "Q3 warp shuffle/vote intrinsics",
        "Q4 CUDA graph capture and replay",
    ];
}

/// A trivial kernel used to prove sm_75 PTX comes out of the compiler.
const VECADD_SRC: &str = r#"
extern "C" __global__ void vecadd(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
"#;

/// A kernel whose arithmetic happens entirely in inline PTX.
///
/// If inline assembly were unavailable or miscompiled, the output would not
/// match the expected value — this is checked numerically rather than by
/// looking for the instruction in the generated PTX.
const INLINE_PTX_SRC: &str = r#"
extern "C" __global__ void inline_ptx_fma(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i], y = b[i], r;
    // Fused multiply-add expressed directly in PTX: r = x * y + 1.0
    asm volatile("fma.rn.f32 %0, %1, %2, 0f3F800000;" : "=f"(r) : "f"(x), "f"(y));
    out[i] = r;
}
"#;

/// An in-place accumulate, used to prove a captured graph replays.
///
/// Accumulating in place makes each replay observable: after `k` replays the
/// buffer must hold exactly `k`. A kernel that merely wrote a value would look
/// identical whether the graph ran once or five times.
const ACCUM_SRC: &str = r#"
extern "C" __global__ void accum(const float* a, float* acc, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += a[i];
}
"#;

/// A kernel exercising the warp primitives the MoE router depends on.
///
/// `__shfl_xor_sync` performs a butterfly reduction across the warp;
/// `__ballot_sync` and `__popc` count how many lanes satisfy a predicate.
/// Both are used by the top-k selection over 256 experts.
const WARP_SRC: &str = r#"
extern "C" __global__ void warp_ops(const float* in, float* sums, int* counts, int n) {
    int lane = threadIdx.x & 31;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? in[i] : 0.0f;

    // Butterfly all-reduce across the warp.
    float s = v;
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_xor_sync(0xffffffff, s, off);
    }

    // Count lanes holding a value above threshold.
    unsigned mask = __ballot_sync(0xffffffff, v > 0.5f);
    int c = __popc(mask);

    if (lane == 0) {
        int w = i >> 5;
        sums[w] = s;
        counts[w] = c;
    }
}
"#;

/// Compile CUDA C++ to PTX for a specific architecture.
fn compile_for(src: &str, arch: &'static str, name: &str) -> Result<cudarc::nvrtc::Ptx, String> {
    compile_ptx_with_opts(
        src,
        CompileOptions {
            arch: Some(arch),
            name: Some(name.to_string()),
            ..Default::default()
        },
    )
    .map_err(|e| format!("{e:?}"))
}

/// Q1 — does the toolchain emit PTX targeting sm_75?
///
/// Checked two ways: NVRTC must accept `--gpu-architecture=compute_75`, and
/// the emitted PTX must carry a matching `.target` directive. Compiling
/// without error is not sufficient evidence that the right target came out.
pub fn check_sm75_ptx() -> CheckOutcome {
    match compile_for(VECADD_SRC, "compute_75", "vecadd") {
        Err(e) => CheckOutcome::Fail(format!("NVRTC rejected compute_75: {e}")),
        Ok(ptx) => {
            let src = ptx.to_src();
            if src.contains(".target sm_75") {
                CheckOutcome::Pass("NVRTC emits PTX with `.target sm_75`".into())
            } else {
                let found = src
                    .lines()
                    .find(|l| l.trim_start().starts_with(".target"))
                    .unwrap_or("<no .target directive>")
                    .trim()
                    .to_string();
                CheckOutcome::Fail(format!("expected `.target sm_75`, PTX carries `{found}`"))
            }
        }
    }
}

/// Q2 — is inline PTX usable inside a kernel, and does it produce correct
/// results?
///
/// This is the question the plan called decisive: without an escape hatch you
/// cannot reach the `mma.sync` and `ldmatrix` instructions the tensor-core
/// paths need, and you cannot work around a compiler codegen bug.
pub fn check_inline_ptx(ctx: &std::sync::Arc<CudaContext>) -> CheckOutcome {
    const N: usize = 1024;

    let ptx = match compile_for(INLINE_PTX_SRC, "compute_75", "inline_ptx_fma") {
        Ok(p) => p,
        Err(e) => {
            return CheckOutcome::Fail(format!("inline `asm volatile` rejected by NVRTC: {e}"));
        }
    };

    let a: Vec<f32> = (0..N).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..N).map(|i| i as f32 * 0.25 + 1.0).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y + 1.0).collect();

    match run_binary_kernel(ctx, &ptx, "inline_ptx_fma", &a, &b, N) {
        Err(e) => CheckOutcome::Fail(format!("inline-PTX kernel failed to run: {e}")),
        Ok(got) => match max_abs_diff(&got, &expected) {
            d if d < 1e-4 => CheckOutcome::Pass(format!(
                "`asm volatile(\"fma.rn.f32\")` compiled, launched, and matched \
                 host arithmetic (max abs error {d:.2e} over {N} elements)"
            )),
            d => CheckOutcome::Fail(format!(
                "inline PTX ran but produced wrong results: max abs error {d:.2e}"
            )),
        },
    }
}

/// Q3 — do warp shuffle and vote intrinsics lower correctly on sm_75?
pub fn check_warp_intrinsics(ctx: &std::sync::Arc<CudaContext>) -> CheckOutcome {
    const WARPS: usize = 8;
    const N: usize = WARPS * 32;

    let ptx = match compile_for(WARP_SRC, "compute_75", "warp_ops") {
        Ok(p) => p,
        Err(e) => return CheckOutcome::Fail(format!("warp intrinsics rejected by NVRTC: {e}")),
    };

    // Values chosen so both the sum and the ballot count are non-trivial and
    // differ per warp.
    let input: Vec<f32> = (0..N).map(|i| ((i % 7) as f32) * 0.2).collect();

    let stream = ctx.default_stream();
    let result = (|| -> Result<(Vec<f32>, Vec<i32>), cudarc::driver::DriverError> {
        let module = ctx.load_module(ptx)?;
        let func = module.load_function("warp_ops")?;

        let d_in = stream.clone_htod(&input)?;
        let mut d_sums = stream.alloc_zeros::<f32>(WARPS)?;
        let mut d_counts = stream.alloc_zeros::<i32>(WARPS)?;
        let n = N as i32;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&d_in)
            .arg(&mut d_sums)
            .arg(&mut d_counts)
            .arg(&n);
        unsafe { builder.launch(cfg) }?;

        stream.synchronize()?;
        Ok((stream.clone_dtoh(&d_sums)?, stream.clone_dtoh(&d_counts)?))
    })();

    let (sums, counts) = match result {
        Ok(v) => v,
        Err(e) => return CheckOutcome::Fail(format!("warp kernel failed to run: {e:?}")),
    };

    // Recompute both reductions on the host.
    let mut ok = true;
    let mut worst = 0.0f32;
    for w in 0..WARPS {
        let slice = &input[w * 32..(w + 1) * 32];
        let expect_sum: f32 = slice.iter().sum();
        let expect_count = slice.iter().filter(|v| **v > 0.5).count() as i32;
        worst = worst.max((sums[w] - expect_sum).abs());
        if counts[w] != expect_count {
            ok = false;
        }
    }

    if ok && worst < 1e-3 {
        CheckOutcome::Pass(format!(
            "__shfl_xor_sync butterfly reduction and __ballot_sync/__popc both \
             match host results across {WARPS} warps (max abs error {worst:.2e})"
        ))
    } else {
        CheckOutcome::Fail(format!(
            "warp intrinsics produced wrong results: shuffle max abs error {worst:.2e}, \
             ballot counts {}",
            if ok { "matched" } else { "MISMATCHED" }
        ))
    }
}

/// Q4 — can a launch be captured into a CUDA graph and replayed?
///
/// This is the project's largest expected single win, so it is verified by
/// replaying a captured graph and checking the device state actually advanced
/// once per replay — not merely by observing that the capture API returned
/// `Ok`.
///
/// # The event-tracking constraint
///
/// cudarc tracks a read and a write event per device allocation and, when the
/// context is in multi-stream mode, injects `cuStreamWaitEvent` before a
/// launch that touches a buffer another stream last used. That is the right
/// default for ordinary work and **fatal during capture**: the injected wait
/// refers to an event recorded before capture began, and CUDA rejects the
/// whole capture with `CUDA_ERROR_STREAM_CAPTURE_ISOLATION`
/// ("dependency created on uncaptured work in another stream").
///
/// So capture requires [`CudaContext::disable_event_tracking`], which hands
/// responsibility for cross-stream ordering back to us. That is not a
/// workaround — it is a real design constraint on milestone 06, and it is why
/// the engine's decode step must own its buffers and order its own streams
/// explicitly rather than relying on the wrapper. Recorded in
/// `docs/TOOLCHAIN.md`.
pub fn check_cuda_graphs(ctx: &std::sync::Arc<CudaContext>) -> CheckOutcome {
    const N: usize = 256;
    const REPLAYS: usize = 5;

    let ptx = match compile_for(ACCUM_SRC, "compute_75", "accum") {
        Ok(p) => p,
        Err(e) => return CheckOutcome::Fail(format!("kernel compile failed: {e}")),
    };

    let result = (|| -> Result<Vec<f32>, cudarc::driver::DriverError> {
        let module = ctx.load_module(ptx)?;
        let func = module.load_function("accum")?;
        let stream = ctx.new_stream()?;

        let ones = vec![1.0f32; N];
        let d_a = stream.clone_htod(&ones)?;
        let mut d_acc = stream.alloc_zeros::<f32>(N)?;
        let n = N as i32;

        // Drain every outstanding operation before capture, so that turning
        // off event tracking below cannot reorder anything still in flight.
        stream.synchronize()?;
        ctx.synchronize()?;

        // SAFETY: the context is fully synchronized above, this thread is the
        // only submitter, and the captured region touches only `d_a` and
        // `d_acc`, whose ordering is established by that synchronization. See
        // the doc comment for why this is required rather than merely
        // convenient.
        unsafe { ctx.disable_event_tracking() };

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Capture `acc += a` into a graph. Relaxed capture mode keeps
        // unrelated driver activity in this process from invalidating it.
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
        {
            let mut builder = stream.launch_builder(&func);
            builder.arg(&d_a).arg(&mut d_acc).arg(&n);
            unsafe { builder.launch(cfg) }?;
        }
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        )?;

        let Some(graph) = graph else {
            return Ok(Vec::new());
        };

        for _ in 0..REPLAYS {
            graph.launch()?;
        }
        stream.synchronize()?;
        stream.clone_dtoh(&d_acc)
    })();

    // SAFETY: restore the wrapper's default behaviour for the rest of the
    // process regardless of how the capture went. Done outside the closure so
    // that an early `?` return cannot leave tracking disabled.
    unsafe { ctx.enable_event_tracking() };

    match result {
        Err(e) => CheckOutcome::Fail(format!("graph capture or replay failed: {e:?}")),
        Ok(v) if v.is_empty() => CheckOutcome::Fail("stream capture produced no graph".into()),
        Ok(v) => {
            // Each replay adds 1.0 to every element.
            let expected = REPLAYS as f32;
            let worst = v
                .iter()
                .map(|x| (x - expected).abs())
                .fold(0.0f32, f32::max);
            if worst < 1e-4 {
                CheckOutcome::Pass(format!(
                    "captured a kernel launch and replayed it {REPLAYS}x; accumulator \
                     advanced exactly once per replay (max abs error {worst:.2e})"
                ))
            } else {
                CheckOutcome::Fail(format!(
                    "graph replayed but device state is wrong: expected {expected}, \
                     max abs error {worst:.2e}"
                ))
            }
        }
    }
}

/// Run the whole gating spike against device `ordinal`.
///
/// Every check is reported as skipped, with a reason, when no driver or device
/// is available. A skipped spike is not a passing spike.
pub fn run(ordinal: usize) -> SpikeReport {
    let skip = |why: &str| SpikeReport {
        emits_sm75_ptx: CheckOutcome::Skipped(why.to_string()),
        inline_ptx: CheckOutcome::Skipped(why.to_string()),
        warp_intrinsics: CheckOutcome::Skipped(why.to_string()),
        cuda_graphs: CheckOutcome::Skipped(why.to_string()),
    };

    let ctx = match CudaContext::new(ordinal) {
        Ok(c) => c,
        Err(e) => return skip(&format!("no CUDA context on device {ordinal}: {e:?}")),
    };

    // PTX emission needs only the compiler, so it is checked first and
    // independently of whether the device is usable.
    let emits_sm75_ptx = check_sm75_ptx();

    let info = DeviceInfo::from_context(ordinal, &ctx).ok();
    if let Some(info) = &info
        && !info.compute_capability.is_supported()
    {
        let why = format!(
            "device {ordinal} is compute {} , below the sm_75 minimum",
            info.compute_capability
        );
        return SpikeReport {
            emits_sm75_ptx,
            inline_ptx: CheckOutcome::Skipped(why.clone()),
            warp_intrinsics: CheckOutcome::Skipped(why.clone()),
            cuda_graphs: CheckOutcome::Skipped(why),
        };
    }

    SpikeReport {
        emits_sm75_ptx,
        inline_ptx: check_inline_ptx(&ctx),
        warp_intrinsics: check_warp_intrinsics(&ctx),
        cuda_graphs: check_cuda_graphs(&ctx),
    }
}

/// Launch a two-input, one-output float kernel and return the result.
fn run_binary_kernel(
    ctx: &std::sync::Arc<CudaContext>,
    ptx: &cudarc::nvrtc::Ptx,
    name: &str,
    a: &[f32],
    b: &[f32],
    n: usize,
) -> Result<Vec<f32>, String> {
    let stream = ctx.default_stream();
    (|| -> Result<Vec<f32>, cudarc::driver::DriverError> {
        let module = ctx.load_module(ptx.clone())?;
        let func = module.load_function(name)?;

        let d_a = stream.clone_htod(a)?;
        let d_b = stream.clone_htod(b)?;
        let mut d_out = stream.alloc_zeros::<f32>(n)?;
        let n_i32 = n as i32;

        let block = 256u32;
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(block as usize) as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&func);
        builder.arg(&d_a).arg(&d_b).arg(&mut d_out).arg(&n_i32);
        unsafe { builder.launch(cfg) }?;

        stream.synchronize()?;
        stream.clone_dtoh(&d_out)
    })()
    .map_err(|e| format!("{e:?}"))
}

/// Largest absolute elementwise difference between two slices.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The toolchain decision this spike resolves.
///
/// The plan offered two options: `cudarc` with CUDA C++ through NVRTC, or
/// NVlabs' `cuda-oxide` single-source Rust. The decision rule was explicit —
/// any "no" on the inline-PTX question collapses it to cudarc.
pub const TOOLCHAIN_DECISION: &str = "cudarc 0.19 + NVRTC";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_skipped_check_is_not_a_passing_check() {
        let skipped = CheckOutcome::Skipped("no device".into());
        assert!(!skipped.is_pass());
        assert!(!skipped.ran());
    }

    #[test]
    fn a_report_of_all_skips_does_not_claim_success_by_running() {
        let r = SpikeReport {
            emits_sm75_ptx: CheckOutcome::Skipped("x".into()),
            inline_ptx: CheckOutcome::Skipped("x".into()),
            warp_intrinsics: CheckOutcome::Skipped("x".into()),
            cuda_graphs: CheckOutcome::Skipped("x".into()),
        };
        assert!(!r.all_ran(), "nothing ran, so the spike is not complete");
    }

    #[test]
    fn a_failure_is_reported_even_when_other_checks_pass() {
        let r = SpikeReport {
            emits_sm75_ptx: CheckOutcome::Pass("ok".into()),
            inline_ptx: CheckOutcome::Fail("no escape hatch".into()),
            warp_intrinsics: CheckOutcome::Pass("ok".into()),
            cuda_graphs: CheckOutcome::Pass("ok".into()),
        };
        assert!(!r.all_ran_checks_passed());
    }

    #[test]
    fn labels_line_up_with_checks() {
        let r = SpikeReport {
            emits_sm75_ptx: CheckOutcome::Pass("a".into()),
            inline_ptx: CheckOutcome::Pass("b".into()),
            warp_intrinsics: CheckOutcome::Pass("c".into()),
            cuda_graphs: CheckOutcome::Pass("d".into()),
        };
        assert_eq!(r.checks().len(), SpikeReport::LABELS.len());
    }

    /// The spike proper. Skips loudly rather than failing when no GPU is
    /// present, because this crate must remain testable on a laptop.
    #[test]
    fn milestone_00_gating_spike() {
        if !crate::device::driver_available() {
            eprintln!("SKIPPED milestone_00_gating_spike: no CUDA driver on this host");
            return;
        }
        let report = run(0);
        for (label, outcome) in SpikeReport::LABELS.iter().zip(report.checks()) {
            eprintln!("  {label}: {outcome}");
        }
        assert!(
            report.all_ran_checks_passed(),
            "a gating check failed; see the lines above — this changes the \
             toolchain decision, it is not a flake"
        );
    }
}
