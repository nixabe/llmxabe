//! CPU differential and six alternating CUDA-event pairs for Rust activations.
//! Run on an idle sm_75 GPU with CUDA_VISIBLE_DEVICES set explicitly.
use std::sync::Arc;

use cudarc::driver::sys::{CUevent_flags, CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use tracing::info;
use xabe_kernels::compare::{Tolerance, assert_matches};
use xabe_kernels::norm::{sigmoid, sigmoid_gate, softplus, swiglu};

type Error = Box<dyn std::error::Error>;

#[derive(Clone, Copy, Debug)]
enum Op {
    Swiglu,
    Softplus,
    Sigmoid { width: usize, broadcast: bool },
}
impl Op {
    fn name(self) -> &'static str {
        match self {
            Self::Swiglu => "swiglu_mul",
            Self::Softplus => "softplus_elementwise",
            Self::Sigmoid { .. } => "sigmoid_gate_mul",
        }
    }
    fn gate_len(self, n: usize) -> usize {
        match self {
            Self::Sigmoid {
                width,
                broadcast: true,
            } => n.div_ceil(width),
            _ => n,
        }
    }
    fn tolerance(self) -> Tolerance {
        // Same gates as tests/layer_ops_differential.rs; never relaxed for Rust.
        let (abs, rel) = match self {
            Self::Swiglu | Self::Softplus => (1e-5, 1e-4),
            _ => (5e-6, 3e-6),
        };
        Tolerance {
            max_abs_error: abs,
            max_rel_error: rel,
            min_cosine_similarity: 1.0 - 1e-7,
            allow_non_finite: false,
        }
    }
}

fn launch(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    op: Op,
    pointers: [u64; 4],
    n: usize,
) -> Result<(), Error> {
    let mut args = stream.launch_builder(kernel);
    let n = n as i64;
    let (width, broadcast) = match op {
        Op::Sigmoid { width, broadcast } => (width as i32, i32::from(broadcast)),
        _ => (1, 0),
    };
    match op {
        Op::Softplus => {
            args.arg(&pointers[0]).arg(&pointers[3]).arg(&n);
        }
        Op::Swiglu => {
            args.arg(&pointers[1])
                .arg(&pointers[0])
                .arg(&pointers[3])
                .arg(&n);
        }
        Op::Sigmoid { .. } => {
            args.arg(&pointers[0])
                .arg(&pointers[1])
                .arg(&pointers[2])
                .arg(&pointers[3])
                .arg(&n)
                .arg(&width)
                .arg(&broadcast);
        }
    }
    // SAFETY: callers retain guarded allocations covering n and gate_len(n).
    // Only exact output/input aliases are tested. Every readback synchronizes.
    unsafe {
        args.launch(LaunchConfig {
            grid_dim: ((n as usize).div_ceil(256).clamp(1, 1024) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })?;
    }
    Ok(())
}

fn main() -> Result<(), Error> {
    xabe_log::init_from_args();
    let ctx = CudaContext::new(0)?;
    // SAFETY: a single stream, allocations retained through synchronization.
    unsafe { ctx.disable_event_tracking() };
    let stream = ctx.new_stream()?;
    let base = ctx.load_module(xabe_cuda::kernels::compile(
        xabe_cuda::kernels::layer_ops::LAYER_OPS_SRC,
        "activations_baseline",
    )?)?;
    let rust = ctx.load_module(Ptx::from_src(include_str!(
        "../../../xabe-cuda/src/kernels/rust/tensor_add.ptx"
    )))?;
    for op in [
        Op::Swiglu,
        Op::Softplus,
        Op::Sigmoid {
            width: 4096,
            broadcast: false,
        },
        Op::Sigmoid {
            width: 2048,
            broadcast: true,
        },
        Op::Sigmoid {
            width: 7,
            broadcast: true,
        },
    ] {
        let kernels = [
            base.load_function(op.name())?,
            rust.load_function(op.name())?,
        ];
        for n in [
            0, 1, 96, 255, 256, 257, 4096, 12288, 49152, 262145, 8_388_608,
        ] {
            let mut x: Vec<f32> = (0..n).map(|i| ((i % 997) as f32 - 498.0) / 166.0).collect();
            if matches!(op, Op::Softplus) {
                for (v, edge) in x.iter_mut().zip([
                    -100.0, -20.0, -0.0, 0.0, 19.999998, 20.0, 20.000002, 100.0, 1000.0,
                ]) {
                    *v = edge;
                }
            }
            let mut gate: Vec<f32> = (0..op.gate_len(n))
                .map(|i| ((i % 991) as f32 - 495.0) / 41.25)
                .collect();
            // Saturation, underflow, signed zero, and the sigmoid midpoint.
            for (g, edge) in gate.iter_mut().zip([-100.0, 100.0, -0.0, 0.0, 1e-7, -1e-7]) {
                *g = edge;
            }
            let reference = match op {
                Op::Swiglu => swiglu(&gate, &x),
                Op::Softplus => x.iter().copied().map(softplus).collect(),
                Op::Sigmoid { width, broadcast } => {
                    let expanded: Vec<_> = (0..n)
                        .map(|i| gate[if broadcast { i / width } else { i }])
                        .collect();
                    sigmoid_gate(&x, &expanded)
                }
            };
            let sig_ref: Vec<_> = gate.iter().copied().map(sigmoid).collect();
            let tol = op.tolerance();
            for (arm, kernel) in kernels.iter().enumerate() {
                for guard in [1, 4] {
                    let guarded = |v: &[f32]| {
                        let mut a = vec![12345.0; guard];
                        a.extend_from_slice(v);
                        a.push(12345.0);
                        a
                    };
                    let aliases = if matches!(op, Op::Swiglu) { 3 } else { 2 };
                    for alias in 0..aliases {
                        let dx = stream.clone_htod(&guarded(&x))?;
                        let dg = stream.clone_htod(&guarded(&gate))?;
                        let ds = stream.clone_htod(&guarded(&vec![0.0; gate.len()]))?;
                        let dout = stream.clone_htod(&guarded(&vec![0.0; n]))?;
                        let output = [&dout, &dx, &dg][alias];
                        let ptr = |v: &cudarc::driver::CudaSlice<f32>| {
                            v.device_ptr(&stream).0 + (guard * 4) as u64
                        };
                        launch(
                            &stream,
                            kernel,
                            op,
                            [ptr(&dx), ptr(&dg), ptr(&ds), ptr(output)],
                            n,
                        )?;
                        for (buffer, expected) in [(output, &reference), (&ds, &sig_ref)] {
                            if std::ptr::eq(buffer, &ds) && !matches!(op, Op::Sigmoid { .. }) {
                                continue;
                            }
                            let actual = stream.clone_dtoh(buffer)?;
                            assert!(actual[..guard].iter().all(|&v| v == 12345.0));
                            assert_eq!(actual[guard + expected.len()], 12345.0);
                            if !expected.is_empty() {
                                assert_matches(
                                    &actual[guard..guard + expected.len()],
                                    expected,
                                    &tol,
                                );
                            }
                        }
                    }
                }
                info!(
                    ?op,
                    n, arm, "CPU differential passed including aliases and guards"
                );
            }
            if n < 4096 && !(matches!(op, Op::Softplus) && n == 96) {
                continue;
            }
            let dx = stream.clone_htod(&x)?;
            let dg = stream.clone_htod(&gate)?;
            let ds = stream.alloc_zeros::<f32>(gate.len())?;
            let dout = stream.alloc_zeros::<f32>(n)?;
            let pointers = [
                dx.device_ptr(&stream).0,
                dg.device_ptr(&stream).0,
                ds.device_ptr(&stream).0,
                dout.device_ptr(&stream).0,
            ];
            stream.synchronize()?;
            let mut graphs = Vec::new();
            for kernel in &kernels {
                stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                for _ in 0..100 {
                    launch(&stream, kernel, op, pointers, n)?;
                }
                graphs.push(stream.end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)?.ok_or("empty graph")?);
            }
            for _ in 0..10 {
                for graph in &graphs {
                    graph.launch()?;
                }
            }
            stream.synchronize()?;
            let start = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
            let stop = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
            for pair in 0..6 {
                let mut us = [0.0; 2];
                for arm in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
                    start.record(&stream)?;
                    graphs[arm].launch()?;
                    stop.record(&stream)?;
                    stream.synchronize()?;
                    us[arm] = f64::from(start.elapsed_ms(&stop)?) * 10.0;
                }
                info!(
                    ?op,
                    n,
                    pair,
                    base_us = us[0],
                    rust_us = us[1],
                    delta_pct = 100.0 * (us[0] / us[1] - 1.0),
                    "CUDA event time per launch"
                );
            }
            // Validate each graph outside the timing window, avoiding host
            // readback and comparison gaps between paired measurements.
            for graph in &graphs {
                graph.launch()?;
                assert_matches(&stream.clone_dtoh(&dout)?, &reference, &tol);
                if matches!(op, Op::Sigmoid { .. }) {
                    assert_matches(&stream.clone_dtoh(&ds)?, &sig_ref, &tol);
                }
            }
        }
    }
    Ok(())
}
