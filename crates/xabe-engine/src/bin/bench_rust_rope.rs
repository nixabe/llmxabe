//! Partial RoPE CPU differentials and six interleaved CUDA-event pairs.
use cudarc::driver::sys::{CUevent_flags, CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;
use tracing::info;
use xabe_kernels::compare::{Tolerance, assert_matches, compare};
use xabe_kernels::rope::apply_rope;
type Error = Box<dyn std::error::Error>;

#[derive(Clone, Copy, Debug)]
struct Shape {
    tokens: usize,
    heads: usize,
    width: usize,
    rotated: usize,
    tiled: bool,
}
fn launch(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    pointers: [u64; 3],
    s: Shape,
) -> Result<(), Error> {
    let (heads, width, rotated, theta) = (
        s.heads as i32,
        s.width as i32,
        s.rotated as i32,
        10_000_000.0f32,
    );
    let mut args = stream.launch_builder(kernel);
    let tokens = s.tokens as i32;
    if s.tiled {
        args.arg(&pointers[0])
            .arg(&pointers[2])
            .arg(&heads)
            .arg(&width)
            .arg(&rotated)
            .arg(&pointers[1])
            .arg(&theta)
            .arg(&tokens);
    } else {
        args.arg(&pointers[0])
            .arg(&pointers[1])
            .arg(&pointers[2])
            .arg(&heads)
            .arg(&width)
            .arg(&rotated)
            .arg(&theta);
    }
    // SAFETY: guarded buffers cover every token/head/column and positions cover
    // tokens; input/output are disjoint. Each rotated pair has one writer.
    unsafe {
        args.launch(LaunchConfig {
            grid_dim: if s.tiled {
                (s.tokens.div_ceil(16) as u32, s.heads as u32, 1)
            } else {
                (s.heads as u32, s.tokens as u32, 1)
            },
            block_dim: (s.width as u32, 1, 1),
            shared_mem_bytes: 0,
        })?;
    }
    Ok(())
}
fn validate(actual: &[f32], reference: &[f32], input: &[f32], s: Shape) {
    let mut a = Vec::new();
    let mut r = Vec::new();
    for row in 0..s.tokens * s.heads {
        let base = row * s.width;
        for j in s.rotated..s.width {
            assert_eq!(actual[base + j].to_bits(), input[base + j].to_bits());
        }
        a.extend_from_slice(&actual[base..base + s.rotated]);
        r.extend_from_slice(&reference[base..base + s.rotated]);
    }
    if !a.is_empty() {
        // Unchanged layer_ops_differential rotated-span and cancellation gates.
        assert_matches(
            &a,
            &r,
            &Tolerance {
                max_abs_error: 1e-5,
                max_rel_error: 5e-2,
                min_cosine_similarity: 1.0 - 1e-7,
                allow_non_finite: false,
            },
        );
        let c = compare(&a, &r);
        let i = c.max_rel_error_index;
        assert!((a[i] - r[i]).abs() < 1e-6);
    }
}
fn main() -> Result<(), Error> {
    xabe_log::init_from_args();
    let ctx = CudaContext::new(0)?;
    // SAFETY: single stream, all buffers retained through synchronization.
    unsafe { ctx.disable_event_tracking() };
    let stream = ctx.new_stream()?;
    let base = ctx.load_module(xabe_cuda::kernels::compile(
        xabe_cuda::kernels::layer_ops::LAYER_OPS_SRC,
        "rope_baseline",
    )?)?;
    let rust = ctx.load_module(Ptx::from_src(include_str!(
        "../../../xabe-cuda/src/kernels/rust/tensor_add.ptx"
    )))?;
    let attention = ctx.load_module(xabe_cuda::kernels::compile(
        xabe_cuda::kernels::attention::ATTENTION_SRC,
        "rope_attention_baseline",
    )?)?;
    for tiled in [false, true] {
        let entry = if tiled {
            "attn_rope_partial_neox"
        } else {
            "rope_partial"
        };
        let kernels = [
            if tiled { &attention } else { &base }.load_function(entry)?,
            rust.load_function(entry)?,
        ];
        for (tokens, heads, width, rotated) in [
            (5, 2, 32, 0),
            (5, 2, 96, 2),
            (5, 2, 128, 128),
            (17, 2, 256, 64),
            (3, 16, 256, 64),
            (3, 2, 256, 64),
            (512, 16, 256, 64),
            (512, 2, 256, 64),
            (1536, 16, 256, 64),
            (1536, 2, 256, 64),
        ] {
            let s = Shape {
                tokens,
                heads,
                width,
                rotated,
                tiled,
            };
            let n = tokens * heads * width;
            let mut x: Vec<f32> = (0..n).map(|i| ((i % 997) as f32 - 498.0) / 498.0).collect();
            for row in 0..tokens * heads {
                if rotated < width {
                    x[row * width + rotated] = -0.0;
                }
            }
            let pos: Vec<u32> = (0..tokens)
                .map(|i| {
                    if tiled && (tokens == 5 || tokens == 17) {
                        262_143 + i as u32
                    } else if tokens == 5 {
                        [0, 1, 100_000, 262_143, u32::MAX][i % 5]
                    } else {
                        2048 + i as u32
                    }
                })
                .collect();
            let mut reference = Vec::with_capacity(n);
            for (t, &position) in pos.iter().enumerate() {
                for h in 0..heads {
                    let at = (t * heads + h) * width;
                    reference.extend(apply_rope(
                        &x[at..at + width],
                        position,
                        rotated as u32,
                        10_000_000.0,
                    ));
                }
            }
            let positions = stream.clone_htod(&pos)?;
            for guard in [1, 4] {
                let mut input = vec![12345.0; n + guard + 1];
                input[guard..guard + n].copy_from_slice(&x);
                let dx = stream.clone_htod(&input)?;
                let sentinel = vec![12345.0; n + guard + 1];
                let mut out = stream.clone_htod(&sentinel)?;
                let pointers = [
                    dx.device_ptr(&stream).0 + (guard * 4) as u64,
                    positions.device_ptr(&stream).0,
                    out.device_ptr(&stream).0 + (guard * 4) as u64,
                ];
                for (arm, kernel) in kernels.iter().enumerate() {
                    stream.memcpy_htod(&sentinel, &mut out)?;
                    launch(&stream, kernel, pointers, s)?;
                    let actual = stream.clone_dtoh(&out)?;
                    assert!(actual[..guard].iter().all(|&v| v == 12345.0));
                    assert_eq!(actual[n + guard], 12345.0);
                    validate(&actual[guard..guard + n], &reference, &x, s);
                    info!(
                        ?s,
                        arm, guard, "CPU rotated-span and exact tail/guard gates passed"
                    );
                }
                if guard != 4 || tokens == 5 || tokens == 17 {
                    continue;
                }
                let mut graphs = Vec::new();
                for kernel in &kernels {
                    stream
                        .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                    for _ in 0..100 {
                        launch(&stream, kernel, pointers, s)?;
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
                        ?s,
                        pair,
                        base_us = us[0],
                        rust_us = us[1],
                        delta_pct = 100.0 * (us[0] / us[1] - 1.0),
                        "CUDA event time per launch"
                    );
                }
                for graph in &graphs {
                    stream.memcpy_htod(&sentinel, &mut out)?;
                    graph.launch()?;
                    let actual = stream.clone_dtoh(&out)?;
                    validate(&actual[guard..guard + n], &reference, &x, s);
                }
            }
        }
    }
    Ok(())
}
