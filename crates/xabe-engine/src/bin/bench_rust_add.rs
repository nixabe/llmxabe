//! Differential and CUDA-event A/B gate for the experimental Rust residual.
//! Usage: bench_rust_add <oxide.ptx> <entry-name> [baseline.ptx]
use std::sync::Arc;

use cudarc::driver::sys::{CUevent_flags, CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use tracing::info;
use xabe_kernels::compare::{Tolerance, assert_matches};
use xabe_kernels::norm::residual_add;

type Error = Box<dyn std::error::Error>;

fn launch(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    pointers: [u64; 3],
    n: usize,
) -> Result<(), Error> {
    let n_i64 = n as i64;
    let mut args = stream.launch_builder(kernel);
    args.arg(&pointers[0])
        .arg(&pointers[1])
        .arg(&pointers[2])
        .arg(&n_i64);
    // SAFETY: callers retain all allocations, provide n valid elements, and
    // synchronize before reading. The zero case deliberately launches once.
    unsafe {
        args.launch(LaunchConfig {
            grid_dim: (n.div_ceil(256).clamp(1, 1024) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })?;
    }
    Ok(())
}

fn main() -> Result<(), Error> {
    xabe_log::init_from_args();
    let args: Vec<_> = std::env::args().collect();
    if !(3..=4).contains(&args.len()) {
        return Err("usage: bench_rust_add <oxide.ptx> <entry-name> [baseline.ptx]".into());
    }
    let source = std::fs::read_to_string(&args[1])?;
    if !source.lines().any(|line| line.trim() == ".target sm_75") {
        return Err("candidate PTX must explicitly target sm_75".into());
    }
    let ctx = CudaContext::new(0)?;
    // SAFETY: one stream, all storage retained until synchronization below.
    unsafe { ctx.disable_event_tracking() };
    let stream = ctx.new_stream()?;
    let baseline_ptx = if let Some(path) = args.get(3) {
        Ptx::from_src(std::fs::read_to_string(path)?)
    } else {
        xabe_cuda::kernels::compile(
            xabe_cuda::kernels::layer_ops::LAYER_OPS_SRC,
            "rust_add_baseline",
        )?
    };
    let baseline = ctx.load_module(baseline_ptx)?;
    let candidate = ctx.load_module(Ptx::from_src(source))?;
    let kernels = [
        baseline.load_function("tensor_add")?,
        candidate.load_function(&args[2])?,
    ];

    // Ragged tails, empty input, decode, prefill, and multiple grid strides.
    for n in [
        0, 1, 3, 4, 5, 255, 256, 257, 2048, 6144, 16383, 16384, 16385, 16386, 16387, 1_048_576,
        8_388_608,
    ] {
        let mut a: Vec<f32> = (0..n)
            .map(|i| ((i % 1009) as f32 - 504.0) / 127.0)
            .collect();
        let mut b: Vec<f32> = (0..n)
            .map(|i| ((i % 1013) as f32 - 506.0) / 131.0)
            .collect();
        // Signed zero, subnormals, cancellation, and a rounding tie. All
        // expected sums stay finite so the ordinary numerical gate applies.
        let edges = [
            (-0.0, -0.0),
            (0.0, -0.0),
            (f32::from_bits(1), f32::from_bits(1)),
            (f32::MIN_POSITIVE, -f32::from_bits(1)),
            (f32::MAX, -f32::MAX),
            (1.0, f32::EPSILON / 2.0),
            (-1.0, -f32::EPSILON / 2.0),
        ];
        for (i, &(x, y)) in edges.iter().take(n).enumerate() {
            a[i] = x;
            b[i] = y;
        }
        let reference = residual_add(&a, &b);
        for (arm, kernel) in kernels.iter().enumerate() {
            // Exercise every float alignment modulo 16, including independently
            // offset pointers that must disable a candidate's vector path.
            for guards in [
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
                [4, 4, 4],
                [4, 1, 4],
                [4, 4, 1],
                [1, 4, 4],
            ] {
                for alias in 0..3 {
                    let guarded = |values: &[f32], guard: usize| {
                        let mut v = vec![12345.0; guard];
                        v.extend_from_slice(values);
                        v.push(12345.0);
                        v
                    };
                    let da = stream.clone_htod(&guarded(&a, guards[0]))?;
                    let db = stream.clone_htod(&guarded(&b, guards[1]))?;
                    let output = stream.clone_htod(&guarded(&vec![0.0; n], guards[2]))?;
                    let pa = da.device_ptr(&stream).0 + (guards[0] * 4) as u64;
                    let pb = db.device_ptr(&stream).0 + (guards[1] * 4) as u64;
                    let pc = output.device_ptr(&stream).0 + (guards[2] * 4) as u64;
                    let out_ptr = [pc, pa, pb][alias];
                    launch(&stream, kernel, [pa, pb, out_ptr], n)?;
                    let actual = stream.clone_dtoh([&output, &da, &db][alias])?;
                    let guard = [guards[2], guards[0], guards[1]][alias];
                    assert!(
                        actual[..guard].iter().all(|&v| v == 12345.0),
                        "leading guard arm={arm}"
                    );
                    assert_eq!(actual[n + guard], 12345.0, "trailing guard arm={arm}");
                    if n > 0 {
                        assert_matches(&actual[guard..n + guard], &reference, &Tolerance::exact());
                    }
                    assert!(
                        actual[guard..n + guard]
                            .iter()
                            .zip(&reference)
                            .all(|(x, y)| x.to_bits() == y.to_bits())
                    );
                }
            }
        }
        info!(
            n,
            "both kernels match CPU exactly; separate output and both aliases, guards intact"
        );
        if n < 2048 {
            continue;
        }

        let da = stream.clone_htod(&a)?;
        let db = stream.clone_htod(&b)?;
        let output = stream.alloc_zeros::<f32>(n)?;
        let pointers = [
            da.device_ptr(&stream).0,
            db.device_ptr(&stream).0,
            output.device_ptr(&stream).0,
        ];
        stream.synchronize()?;
        // Amortize event overhead and remove host enqueue gaps. Output is
        // overwritten each time, so replay has the same arithmetic input.
        let mut graphs = Vec::new();
        for kernel in &kernels {
            stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            for _ in 0..100 {
                launch(&stream, kernel, pointers, n)?;
            }
            graphs.push(
                stream
                    .end_capture(
                        CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                    )?
                    .ok_or("empty graph")?,
            );
        }
        for _ in 0..10 {
            for graph in &graphs {
                graph.launch()?;
            }
        }
        stream.synchronize()?;
        let start = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        let stop = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        let mut samples = [Vec::new(), Vec::new()];
        for pair in 0..6 {
            for arm in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
                start.record(&stream)?;
                graphs[arm].launch()?;
                stop.record(&stream)?;
                stream.synchronize()?;
                samples[arm].push(f64::from(start.elapsed_ms(&stop)?) * 10.0);
            }
            let base_us = samples[0][pair];
            let rust_us = samples[1][pair];
            info!(
                n,
                pair,
                base_us,
                rust_us,
                delta_pct = 100.0 * (base_us / rust_us - 1.0),
                "CUDA event time per launch"
            );
        }
        let actual = stream.clone_dtoh(&output)?;
        assert_matches(&actual, &reference, &Tolerance::exact());
        for (arm, times) in samples.iter().enumerate() {
            info!(
                n,
                arm,
                min_us = times.iter().copied().fold(f64::INFINITY, f64::min),
                max_us = times.iter().copied().fold(0.0, f64::max),
                "spread"
            );
        }
    }
    Ok(())
}
