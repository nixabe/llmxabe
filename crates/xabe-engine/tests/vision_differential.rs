//! Differential test: the device vision tower against the scalar
//! reference, at a small synthetic geometry and at the real mmproj.
//!
//! The reference (`xabe_kernels::vision::encode`) is itself validated
//! against llama.cpp executing the same mmproj file
//! (`xabe-kernels/tests/vision_golden.rs`), so agreement here chains the
//! device path back to upstream.
//!
//! ## Tolerances
//!
//! The device path stores every activation as f16 (matching llama.cpp's
//! CUDA clip path) while the reference is scalar f32, so this is *not* a
//! same-formula fp32 comparison: 27 blocks of f16 rounding compound.
//! The gates below were set from measurement — see each assertion — and
//! cosine similarity is the primary gate, per `docs/TESTING.md`, with
//! max-abs quoted against the observed value on the real weights.
//!
//! SKIPS without a driver, an sm_75 device, or (for the real-weights case)
//! the mmproj file.

use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use xabe_engine::vision::{VisionForward, load_vision_weights};
use xabe_kernels::compare::compare;
use xabe_kernels::rng::Xorshift64Star;
use xabe_kernels::vision::tower::{VisionBlockWeights, VisionWeights, encode};
use xabe_kernels::vision::{PreprocessedImage, preprocess};
use xabe_model::VisionConfig;

const DEFAULT_MMPROJ_PATH: &str =
    "/home/nixabe/llmxabe/models/Qwen3.6-35B-A3B-GGUF/mmproj-F16.gguf";

fn setup() -> Option<Arc<CudaContext>> {
    if !xabe_cuda::device::driver_available() {
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
    match xabe_cuda::device::DeviceInfo::from_context(0, &ctx) {
        Ok(info) if info.is_supported() => Some(ctx),
        Ok(_) => {
            println!("SKIPPED: device 0 is below the sm_75 minimum");
            None
        }
        Err(e) => {
            println!("SKIPPED: could not probe device 0: {e}");
            None
        }
    }
}

fn tiny_cfg() -> VisionConfig {
    VisionConfig {
        num_layers: 3,
        hidden_size: 64,
        num_heads: 2,
        ffn_size: 128,
        image_size: 64, // 4x4 position grid
        patch_size: 16,
        temporal_patch_size: 2,
        spatial_merge: 2,
        projection_dim: 32,
        ln_eps: 1e-6,
        rope_theta: 10_000.0,
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
    }
}

fn tiny_weights(cfg: &VisionConfig, seed: u64) -> VisionWeights {
    let mut rng = Xorshift64Star::new(seed);
    let h = cfg.hidden_size as usize;
    let ffn = cfg.ffn_size as usize;
    let patch = (cfg.patch_elems() / cfg.temporal_patch_size) as usize;
    let edge = cfg.pos_grid_edge() as usize;
    let merged = cfg.merger_input_dim() as usize;
    let proj = cfg.projection_dim as usize;
    let mut v = |n: usize| rng.vec_f32(n, -0.08, 0.08);
    VisionWeights {
        patch_embed: v(patch * h),
        patch_bias: v(h),
        pos_embed: v(edge * edge * h),
        blocks: (0..cfg.num_layers)
            .map(|_| VisionBlockWeights {
                ln1_w: v(h).iter().map(|x| 1.0 + x).collect(),
                ln1_b: v(h),
                qkv_w: v(h * 3 * h),
                qkv_b: v(3 * h),
                out_w: v(h * h),
                out_b: v(h),
                ln2_w: v(h).iter().map(|x| 1.0 + x).collect(),
                ln2_b: v(h),
                up_w: v(h * ffn),
                up_b: v(ffn),
                down_w: v(ffn * h),
                down_b: v(h),
            })
            .collect(),
        post_ln_w: v(h).iter().map(|x| 1.0 + x).collect(),
        post_ln_b: v(h),
        fc1_w: v(merged * merged),
        fc1_b: v(merged),
        fc2_w: v(merged * proj),
        fc2_b: v(proj),
    }
}

#[test]
fn device_tower_matches_the_reference_at_a_synthetic_geometry() {
    let Some(ctx) = setup() else { return };
    let stream = ctx.default_stream();
    let cfg = tiny_cfg();
    let weights = tiny_weights(&cfg, 11);
    let mut fwd =
        VisionForward::new(&ctx, stream, &cfg, &weights, 64).expect("device tower builds");

    // A non-square grid exercises the position-embedding resize, the rope
    // row/col split, and the cell ordering asymmetrically.
    let patch_len = (cfg.patch_elems() / cfg.temporal_patch_size) as usize;
    let (gh, gw) = (4u32, 8u32);
    let mut rng = Xorshift64Star::new(23);
    let image = PreprocessedImage {
        patches: rng.vec_f32((gh * gw) as usize * patch_len, -1.0, 1.0),
        grid_h: gh,
        grid_w: gw,
    };

    let device = fwd.encode_to_host(&image).expect("device encode");
    let reference = encode(&cfg, &weights, &image.patches, gh, gw);

    let result = compare(&device, &reference);
    println!(
        "tiny: max_abs {:.3e}  max_rel {:.3e}  cosine {:.6}",
        result.max_abs_error, result.max_rel_error, result.cosine_similarity
    );
    // Measured on this geometry: max_abs 1.1e-3, cosine 1.000000. The
    // gates leave ~4x headroom without admitting a transposed weight or a
    // head-offset bug, both of which push cosine far below 0.99.
    assert!(result.cosine_similarity > 0.9999, "cosine degraded");
    assert!(result.max_abs_error < 8e-3, "max_abs degraded");
}

#[test]
fn device_tower_matches_the_reference_on_the_real_mmproj() {
    let Some(ctx) = setup() else { return };
    let path = std::env::var_os("LLMXABE_MMPROJ")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MMPROJ_PATH));
    if !path.exists() {
        println!("SKIPPED: mmproj not found at {}", path.display());
        return;
    }
    let file = xabe_gguf::GgufFile::open(&path).expect("mmproj parses");
    let cfg = VisionConfig::qwen3_6_35b_a3b();
    let weights = load_vision_weights(&file, &cfg).expect("mmproj loads");

    let stream = ctx.default_stream();
    let mut fwd =
        VisionForward::new(&ctx, stream, &cfg, &weights, 256).expect("device tower builds");

    // A 96x96 gradient image through the real preprocessing: 6x6 patch
    // grid, 9 output tokens.
    let mut rgb = vec![0u8; 96 * 96 * 3];
    for y in 0..96u32 {
        for x in 0..96u32 {
            let i = ((y * 96 + x) * 3) as usize;
            rgb[i] = (x * 255 / 95) as u8;
            rgb[i + 1] = (y * 255 / 95) as u8;
            rgb[i + 2] = ((x + y) * 255 / 190) as u8;
        }
    }
    let image = preprocess(&cfg, &rgb, 96, 96);
    assert_eq!((image.grid_h, image.grid_w), (6, 6));

    let device = fwd.encode_to_host(&image).expect("device encode");
    let reference = encode(&cfg, &weights, &image.patches, image.grid_h, image.grid_w);

    let result = compare(&device, &reference);
    println!(
        "real: max_abs {:.3e}  max_rel {:.3e}  cosine {:.6}",
        result.max_abs_error, result.max_rel_error, result.cosine_similarity
    );
    // 27 blocks of f16 activation rounding against scalar f32; measured
    // max_abs 1.02e-2, cosine 0.999990 (max_rel is uninformative here —
    // near-zero elements, the gdn_differential.rs trap). Gates leave ~5x
    // headroom on max_abs without admitting a transposed weight or a
    // head-offset bug, which push cosine far below 0.99.
    assert!(result.cosine_similarity > 0.999, "cosine degraded");
    assert!(result.max_abs_error < 5e-2, "max_abs degraded");
}
