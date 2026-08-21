//! Differential test: the device interleaved M-RoPE kernel
//! (`attn_rope_partial_imrope`) against `xabe_kernels::mrope::apply_imrope`,
//! at the real Qwen3.6 attention geometry (head_dim 256, rope_dim 64,
//! sections [11, 11, 10]).
//!
//! Two properties gate:
//!
//! 1. **Reference agreement** on genuinely diverging `(t, h, w)` triples —
//!    the image-span case. Same metric policy as
//!    `layer_ops_differential.rs`'s rope test: cosine and max-abs gate,
//!    `max_rel_error` is reported but not gated (rotation produces genuine
//!    cancellation near zero), and the untouched tail `[64, 256)` is
//!    checked bit-exactly.
//! 2. **Bit-exact collapse to the scalar kernel** when `t == h == w` —
//!    the same reduction the engine's text path relies on. The two kernels
//!    share their double-precision angle arithmetic by construction; this
//!    asserts nothing drifted.
//!
//! SKIPS without a driver or a supported device.

use std::sync::Arc;

use cudarc::driver::CudaContext;
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_cuda::kernels::attention::AttentionKernels;
use xabe_kernels::compare::compare;
use xabe_kernels::mrope::{MropePos, QWEN3_6_SECTIONS, apply_imrope};
use xabe_kernels::rng::Xorshift64Star;

const Q_HEADS: usize = 16;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 256;
const ROPE_DIM: usize = 64;
const THETA: f32 = 10_000_000.0;

fn setup() -> Option<Arc<CudaContext>> {
    if !driver_available() {
        println!("SKIPPED: no CUDA driver present");
        return None;
    }
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("SKIPPED: could not create a context on device 0: {e}");
            return None;
        }
    };
    match DeviceInfo::from_context(0, &ctx) {
        Ok(info) if info.is_supported() => Some(ctx),
        _ => {
            println!("SKIPPED: device 0 unusable or below sm_75");
            None
        }
    }
}

/// Positions shaped like a real mixed chunk: text, an 8x6 image grid at
/// base 3, text continuing at the max-rule base.
fn mixed_positions(n_tokens: usize) -> Vec<MropePos> {
    let span = xabe_kernels::mrope::ImageSpan {
        start: 3,
        grid_h: 8,
        grid_w: 6,
    };
    let (pos, _next) = xabe_kernels::mrope::assign_positions(n_tokens, &[span]);
    pos
}

#[test]
fn device_imrope_matches_the_reference_on_diverging_components() {
    let Some(ctx) = setup() else { return };
    let stream = ctx.default_stream();
    let kernels = AttentionKernels::new(&ctx, Q_HEADS, KV_HEADS, HEAD_DIM).expect("compiles");

    let n_tokens = 3 + 48 + 5;
    let positions = mixed_positions(n_tokens);
    let mut rng = Xorshift64Star::new(41);
    let input = rng.vec_f32(n_tokens * Q_HEADS * HEAD_DIM, -1.0, 1.0);

    let triples: Vec<i32> = positions
        .iter()
        .flat_map(|p| [p.t as i32, p.h as i32, p.w as i32])
        .collect();
    let d_in = stream.clone_htod(&input).expect("upload");
    let d_pos = stream.clone_htod(&triples).expect("upload");
    let mut d_out = stream.alloc_zeros::<f32>(input.len()).expect("alloc");
    kernels
        .rope_imrope(
            &stream,
            &d_in,
            &mut d_out,
            n_tokens,
            Q_HEADS,
            ROPE_DIM,
            &d_pos,
            QWEN3_6_SECTIONS,
            THETA,
        )
        .expect("launch");
    let device = stream.clone_dtoh(&d_out).expect("download");
    stream.synchronize().expect("sync");

    let mut reference = Vec::with_capacity(input.len());
    for (t, pos) in positions.iter().enumerate() {
        for h in 0..Q_HEADS {
            let base = (t * Q_HEADS + h) * HEAD_DIM;
            reference.extend(apply_imrope(
                &input[base..base + HEAD_DIM],
                *pos,
                ROPE_DIM as u32,
                QWEN3_6_SECTIONS,
                THETA,
            ));
        }
    }

    let result = compare(&device, &reference);
    println!(
        "imrope: max_abs {:.3e}  max_rel {:.3e}  cosine {:.9}",
        result.max_abs_error, result.max_rel_error, result.cosine_similarity
    );
    // Measured 1.2e-7 max_abs (one ulp of sin/cos rounded to f32); gate
    // leaves ~80x headroom without admitting a wrong channel selection,
    // which moves whole pairs by O(1).
    assert!(result.max_abs_error < 1e-5, "max_abs degraded");
    assert!(result.cosine_similarity > 0.999_999, "cosine degraded");

    // The untouched tail is copied, not computed: bit-exact.
    for t in 0..n_tokens {
        for h in 0..Q_HEADS {
            let base = (t * Q_HEADS + h) * HEAD_DIM;
            assert_eq!(
                &device[base + ROPE_DIM..base + HEAD_DIM],
                &input[base + ROPE_DIM..base + HEAD_DIM],
                "tail modified at token {t} head {h}"
            );
        }
    }
}

#[test]
fn device_imrope_with_scalar_positions_is_bit_identical_to_the_scalar_kernel() {
    let Some(ctx) = setup() else { return };
    let stream = ctx.default_stream();
    let kernels = AttentionKernels::new(&ctx, Q_HEADS, KV_HEADS, HEAD_DIM).expect("compiles");

    let n_tokens = 19;
    let base_pos = 131_009i32; // deep enough that f32 angles would diverge
    let mut rng = Xorshift64Star::new(43);
    let input = rng.vec_f32(n_tokens * KV_HEADS * HEAD_DIM, -1.0, 1.0);
    let d_in = stream.clone_htod(&input).expect("upload");

    // Scalar kernel: base device scalar, angle = *base + t.
    let d_base = stream.clone_htod(&[base_pos]).expect("upload");
    let mut d_scalar_out = stream.alloc_zeros::<f32>(input.len()).expect("alloc");
    kernels
        .rope(
            &stream,
            &d_in,
            &mut d_scalar_out,
            n_tokens,
            KV_HEADS,
            ROPE_DIM,
            &d_base,
            THETA,
        )
        .expect("scalar launch");

    // IMROPE kernel with t == h == w == base + i.
    let triples: Vec<i32> = (0..n_tokens as i32)
        .flat_map(|i| [base_pos + i; 3])
        .collect();
    let d_pos = stream.clone_htod(&triples).expect("upload");
    let mut d_imrope_out = stream.alloc_zeros::<f32>(input.len()).expect("alloc");
    kernels
        .rope_imrope(
            &stream,
            &d_in,
            &mut d_imrope_out,
            n_tokens,
            KV_HEADS,
            ROPE_DIM,
            &d_pos,
            QWEN3_6_SECTIONS,
            THETA,
        )
        .expect("imrope launch");

    let scalar = stream.clone_dtoh(&d_scalar_out).expect("download");
    let imrope = stream.clone_dtoh(&d_imrope_out).expect("download");
    stream.synchronize().expect("sync");
    assert_eq!(scalar, imrope, "scalar collapse must be bit-exact");
}
