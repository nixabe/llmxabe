//! CPU differential and interleaved CUDA-event gate for Rust normalization.
use std::sync::Arc;

use cudarc::driver::sys::{CUevent_flags, CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use tracing::info;
use xabe_kernels::compare::{Tolerance, assert_matches};
use xabe_kernels::norm::{rms_norm, swiglu};

type Error = Box<dyn std::error::Error>;
// The existing layer-ops RMSNorm and SwiGLU gates, unchanged.
const GATE: Tolerance = Tolerance {
    max_abs_error: 1e-5,
    max_rel_error: 1e-4,
    min_cosine_similarity: 1.0 - 1e-7,
    allow_non_finite: false,
};

#[derive(Clone, Copy)]
struct Shape {
    rows: usize,
    width: usize,
    eps: f32,
    fused: bool,
}

fn launch(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    pointers: [u64; 5],
    shape: Shape,
) -> Result<(), Error> {
    let width = shape.width as i32;
    let block = shape.width.next_multiple_of(32).clamp(32, 1024) as u32;
    let mut args = stream.launch_builder(kernel);
    args.arg(&pointers[0]).arg(&pointers[1]);
    if shape.fused {
        args.arg(&pointers[2]).arg(&pointers[3]);
    }
    args.arg(&pointers[4]).arg(&width).arg(&shape.eps);
    // SAFETY: callers retain every allocation and provide width weights and
    // rows*width data. The grid/block/shared geometry matches LayerOpsKernels.
    unsafe {
        args.launch(LaunchConfig {
            grid_dim: (shape.rows as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: block / 32 * 4,
        })?;
    }
    Ok(())
}

fn main() -> Result<(), Error> {
    xabe_log::init_from_args();
    let ctx = CudaContext::new(0)?;
    // SAFETY: one stream, allocations retained until synchronized readback.
    unsafe { ctx.disable_event_tracking() };
    let stream = ctx.new_stream()?;
    let base = ctx.load_module(xabe_cuda::kernels::compile(
        xabe_cuda::kernels::layer_ops::LAYER_OPS_SRC,
        "norm_baseline",
    )?)?;
    let rust = ctx.load_module(Ptx::from_src(include_str!(
        "../../../xabe-cuda/src/kernels/rust/tensor_add.ptx"
    )))?;
    for fused in [false, true] {
        let entry = if fused {
            "rms_norm_swiglu_rows"
        } else {
            "rms_norm_rows"
        };
        let kernels = [base.load_function(entry)?, rust.load_function(entry)?];
        // Sub-warp/ragged widths and multiple iterations per thread, followed
        // by Qwen3.6 hidden, GDN-head and attention-head decode/prefill shapes.
        for (rows, width) in [
            (2, 1),
            (2, 31),
            (2, 33),
            (2, 129),
            (2, 1025),
            (2, 5120),
            (3, 2048),
            (96, 128),
            (48, 256),
            (1536, 2048),
            (49152, 128),
            (24576, 256),
        ] {
            let n = rows * width;
            let mut x: Vec<f32> = (0..n).map(|i| ((i % 997) as f32 - 498.0) / 166.0).collect();
            // A zero row exercises epsilon and a row with a single nonzero
            // value checks partial warps and ownership of the final column.
            x[..width].fill(0.0);
            x[width..2 * width].fill(0.0);
            x[2 * width - 1] = 1.0;
            // Keep sparse-row magnitudes bounded even at adversarial widths.
            let w: Vec<f32> = (0..width)
                .map(|i| ((i % 71) as f32 - 35.0) / 128.0)
                .collect();
            let gate: Vec<f32> = (0..n).map(|i| ((i % 991) as f32 - 495.0) / 165.0).collect();
            for eps in [1e-6, 1e-5] {
                let shape = Shape {
                    rows,
                    width,
                    eps,
                    fused,
                };
                let reference: Vec<f32> = x
                    .chunks_exact(width)
                    .flat_map(|row| rms_norm(row, &w, eps))
                    .collect();
                let expected = if fused {
                    swiglu(&gate, &reference)
                } else {
                    reference.clone()
                };
                for (arm, kernel) in kernels.iter().enumerate() {
                    for guard in [1, 4] {
                        let guarded = |v: &[f32]| {
                            let mut a = vec![12345.0; guard];
                            a.extend_from_slice(v);
                            a.push(12345.0);
                            a
                        };
                        for alias in 0..if fused { 1 } else { 2 } {
                            let dx = stream.clone_htod(&guarded(&x))?;
                            let dw = stream.clone_htod(&guarded(&w))?;
                            let dg = stream.clone_htod(&guarded(&gate))?;
                            let dn = stream.clone_htod(&guarded(&vec![0.0; n]))?;
                            let dout = stream.clone_htod(&guarded(&vec![0.0; n]))?;
                            let output = if alias == 0 { &dout } else { &dx };
                            let ptr = |v: &cudarc::driver::CudaSlice<f32>| {
                                v.device_ptr(&stream).0 + (guard * 4) as u64
                            };
                            launch(
                                &stream,
                                kernel,
                                [ptr(&dx), ptr(&dw), ptr(&dg), ptr(&dn), ptr(output)],
                                shape,
                            )?;
                            for (buffer, values) in [(output, &expected), (&dn, &reference)] {
                                if std::ptr::eq(buffer, &dn) && !fused {
                                    continue;
                                }
                                let actual = stream.clone_dtoh(buffer)?;
                                assert!(actual[..guard].iter().all(|&v| v == 12345.0));
                                assert_eq!(actual[n + guard], 12345.0);
                                assert_matches(&actual[guard..n + guard], values, &GATE);
                            }
                        }
                    }
                    info!(
                        entry,
                        rows,
                        width,
                        eps,
                        arm,
                        "CPU differential passed; guards and supported alias intact"
                    );
                }
                if rows == 2 || eps != 1e-6 {
                    continue;
                }
                let dx = stream.clone_htod(&x)?;
                let dw = stream.clone_htod(&w)?;
                let dg = stream.clone_htod(&gate)?;
                let dn = stream.alloc_zeros::<f32>(n)?;
                let dout = stream.alloc_zeros::<f32>(n)?;
                let ptrs = [
                    dx.device_ptr(&stream).0,
                    dw.device_ptr(&stream).0,
                    dg.device_ptr(&stream).0,
                    dn.device_ptr(&stream).0,
                    dout.device_ptr(&stream).0,
                ];
                stream.synchronize()?;
                let mut graphs = Vec::new();
                for kernel in &kernels {
                    stream
                        .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                    for _ in 0..100 {
                        launch(&stream, kernel, ptrs, shape)?;
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
                        entry,
                        rows,
                        width,
                        pair,
                        base_us = us[0],
                        rust_us = us[1],
                        delta_pct = 100.0 * (us[0] / us[1] - 1.0),
                        "CUDA event time per launch"
                    );
                }
                for graph in &graphs {
                    graph.launch()?;
                    assert_matches(&stream.clone_dtoh(&dout)?, &expected, &GATE);
                    if fused {
                        assert_matches(&stream.clone_dtoh(&dn)?, &reference, &GATE);
                    }
                }
            }
        }
    }
    Ok(())
}
