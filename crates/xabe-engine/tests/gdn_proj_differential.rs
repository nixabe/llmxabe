//! Isolates the exact kernel pairing that was the origin of the
//! batch-vs-single-stream residual in Gated DeltaNet: `GdnBlock::project`'s untiled and tiled forms (standard Q8_0
//! layout) against `GdnBlock::project_split_gemv` (the repacked split
//! layout, the one-token path single-stream decode actually takes whenever
//! the int8 repack is resident).
//!
//! Same method `moe_proj_differential`... no, `moe_differential.rs`'s
//! `gemv_and_direct_isolate_the_one_live_token_case` / `gemv_and_direct_
//! isolate_the_two_live_token_case` established: real weights, identical
//! input, two different compiled entry points, direct comparison, no
//! tolerance. Two questions, two tests:
//!
//! 1. Are the untiled `gdn_proj_q8_0` and the tiled `gdn_proj_q8_0_t*`
//!    family -- both over the *standard*, non-repacked Q8_0 layout --
//!    already bit-identical to each other? Both walk the weight in the
//!    same per-lane order (`blk[2 + lane]` for each 32-element block in
//!    turn), so the prediction is yes.
//! 2. Is `gdn_proj_split_gemv` -- the repacked, vectorized-4-per-lane
//!    layout -- bit-identical to either of the above? The kernel's own
//!    comment already says no ("the warp reduction sums in a different
//!    order... equivalent rather than bit-identical"); this measures it
//!    rather than trusting the comment.
//!
//! Only `project()`'s public dispatch is exercised for (1): `tokens == 1`
//! takes the untiled kernel and `tokens > 1` takes the tiled one, so
//! feeding it two tokens and reading back token 0's row is `GdnBlock`'s own
//! routing choosing the tiled kernel for us, not a private kernel handle
//! reached around it.
//!
//! SKIPS -- reporting that it skipped -- without a driver, a supported
//! device, or the model file, exactly as `gdn_block.rs` does.

use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_engine::block::gdn::{GdnBlock, GdnGeometry, GdnLayerWeights, Projection};
use xabe_gguf::GgufFile;
use xabe_kernels::rng::Xorshift64Star;
use xabe_model::config::ModelConfig;
use xabe_model::weights::WeightSchema;

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llmxabe/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

/// The Gated DeltaNet layer whose real weights are used. Layer 0, matching
/// `gdn_block.rs`'s own first anchor layer.
const LAYER: u32 = 0;

fn model_path() -> PathBuf {
    std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
}

fn setup() -> Option<(Arc<CudaContext>, GgufFile, ModelConfig)> {
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
    let path = model_path();
    if !path.exists() {
        println!(
            "SKIPPED: model file not found at {}; set LLMXABE_MODEL to override",
            path.display(),
        );
        return None;
    }
    let file = GgufFile::open(&path).expect("valid GGUF v3");
    let config = ModelConfig::qwen3_6_35b_a3b();
    Some((ctx, file, config))
}

fn dtoh(stream: &Arc<CudaStream>, buf: &CudaSlice<f32>) -> Vec<f32> {
    let v = stream.clone_dtoh(buf).expect("device read-back");
    stream.synchronize().expect("sync");
    v
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// `project()`'s untiled (`tokens == 1`) and tiled (`tokens > 1`) forms,
/// both over the standard Q8_0 layout, compared on token 0 of a two-token
/// call against the same row run alone.
#[test]
fn untiled_and_tiled_standard_layout_already_agree() {
    let Some((ctx, file, config)) = setup() else {
        return;
    };
    let stream = ctx.default_stream();
    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).expect("schema resolves");
    let geometry = GdnGeometry::from_config(&config, 8, 1e-6);
    let block = GdnBlock::new(&ctx, geometry).expect("kernels compile");
    let weights =
        GdnLayerWeights::upload(&stream, &file, &directory, LAYER).expect("weights upload");

    let hidden = geometry.hidden;
    let conv_dim = geometry.conv_dim();
    let mut rng = Xorshift64Star::new(0x_5EED_9D01);
    let row0 = rng.vec_f32(hidden, -1.0, 1.0);
    let row1 = rng.vec_f32(hidden, -1.0, 1.0);

    // Alone: `tokens == 1`, the untiled `gdn_proj_q8_0`.
    let x_alone = stream.clone_htod(&row0).expect("upload x");
    let mut out_alone = stream.alloc_zeros::<f32>(conv_dim).expect("out allocates");
    block
        .project(
            &stream,
            Projection::Q8_0(&weights.qkv),
            &x_alone,
            &mut out_alone,
            hidden,
            conv_dim,
            1,
        )
        .expect("untiled projection runs");
    let via_untiled = dtoh(&stream, &out_alone);

    // Paired: `tokens == 2`, the tiled `gdn_proj_q8_0_t2`. Token 1 is a
    // distinct random row, not a copy of token 0 -- if the tiled kernel's
    // cross-token accumulator indexing were wrong, reusing the same row
    // twice could hide it.
    let mut paired = row0.clone();
    paired.extend_from_slice(&row1);
    let x_paired = stream.clone_htod(&paired).expect("upload x");
    let mut out_paired = stream
        .alloc_zeros::<f32>(2 * conv_dim)
        .expect("out allocates");
    block
        .project(
            &stream,
            Projection::Q8_0(&weights.qkv),
            &x_paired,
            &mut out_paired,
            hidden,
            conv_dim,
            2,
        )
        .expect("tiled projection runs");
    let via_tiled = dtoh(&stream, &out_paired);
    let token0_via_tiled = &via_tiled[..conv_dim];

    let diff = max_abs_diff(&via_untiled, token0_via_tiled);
    println!(
        "untiled (tokens=1) vs tiled (tokens=2, token 0), standard Q8_0 layout: \
         max_abs_diff {diff:.3e} over {conv_dim} elements",
    );
    assert_eq!(
        via_untiled, token0_via_tiled,
        "gdn_proj_q8_0 and gdn_proj_q8_0_t2 disagree on the same row -- the \
         standard-layout untiled and tiled kernels are not reduction-order- \
         identical after all",
    );
}

/// `project_split_gemv` (the repacked split layout, the kernel real
/// single-stream decode actually takes) against `project()`'s untiled
/// standard-layout form, on the identical row.
///
/// Not gated on a tolerance: this documents where the two-kernel-family
/// mismatch actually lives, per the kernel's own comment
/// ("the warp reduction sums in a different order... equivalent rather
/// than bit-identical").
#[test]
fn split_layout_gemv_disagrees_with_the_standard_layout() {
    let Some((ctx, file, config)) = setup() else {
        return;
    };
    let stream = ctx.default_stream();
    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).expect("schema resolves");
    let geometry = GdnGeometry::from_config(&config, 8, 1e-6);
    let block = GdnBlock::new(&ctx, geometry).expect("kernels compile");
    let weights =
        GdnLayerWeights::upload(&stream, &file, &directory, LAYER).expect("weights upload");
    let int8 = block.repack(&stream, &weights).expect("repack builds");

    let hidden = geometry.hidden;
    let conv_dim = geometry.conv_dim();
    let mut rng = Xorshift64Star::new(0x_5EED_9D02);
    let row0 = rng.vec_f32(hidden, -1.0, 1.0);

    let x = stream.clone_htod(&row0).expect("upload x");

    let mut out_standard = stream.alloc_zeros::<f32>(conv_dim).expect("out allocates");
    block
        .project(
            &stream,
            Projection::Q8_0(&weights.qkv),
            &x,
            &mut out_standard,
            hidden,
            conv_dim,
            1,
        )
        .expect("standard-layout projection runs");
    let via_standard = dtoh(&stream, &out_standard);

    let (qkv_q, qkv_s) = int8.qkv();
    let mut out_split = stream.alloc_zeros::<f32>(conv_dim).expect("out allocates");
    block
        .project_split_gemv(&stream, qkv_q, qkv_s, &x, &mut out_split, hidden, conv_dim)
        .expect("split-layout gemv runs");
    let via_split = dtoh(&stream, &out_split);

    let diff = max_abs_diff(&via_standard, &via_split);
    println!(
        "standard layout (gdn_proj_q8_0) vs split layout (gdn_proj_split_gemv): \
         max_abs_diff {diff:.3e} over {conv_dim} elements, output magnitude \
         max |standard| = {:.4e}",
        via_standard.iter().fold(0.0f32, |m, v| m.max(v.abs())),
    );
    assert_ne!(
        diff, 0.0,
        "split-layout gemv and the standard layout now agree bit-for-bit -- \
         if a fix landed, replace this assertion with the bit-exact one \
         `untiled_and_tiled_standard_layout_already_agree` uses and update \
         this test's module docs",
    );
}

/// `project_split_tiled` (the new batch path at `1 < tokens < MMA_SPLIT_
/// TOKENS`) against `project_split_gemv` (the one-token path) on the
/// identical row. This is the pairing `run_batch_decode` vs `run` now
/// actually take once the tiled split kernel is wired in.
///
/// Bit-exact: column 0 of the tiled launch must see the GEMV's own
/// sequence of additions, operand for operand. A nonzero here means the
/// tiled kernel is not the generalization it claims to be.
#[test]
fn split_layout_tiled_agrees_with_the_gemv() {
    let Some((ctx, file, config)) = setup() else {
        return;
    };
    let stream = ctx.default_stream();
    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).expect("schema resolves");
    let geometry = GdnGeometry::from_config(&config, 8, 1e-6);
    let block = GdnBlock::new(&ctx, geometry).expect("kernels compile");
    let weights =
        GdnLayerWeights::upload(&stream, &file, &directory, LAYER).expect("weights upload");
    let int8 = block.repack(&stream, &weights).expect("repack builds");
    let (qkv_q, qkv_s) = int8.qkv();

    let hidden = geometry.hidden;
    let conv_dim = geometry.conv_dim();
    let mut rng = Xorshift64Star::new(0x_5EED_9D03);
    let row0 = rng.vec_f32(hidden, -1.0, 1.0);
    let row1 = rng.vec_f32(hidden, -1.0, 1.0);
    let row2 = rng.vec_f32(hidden, -1.0, 1.0);

    let x_alone = stream.clone_htod(&row0).expect("upload x");
    let mut out_alone = stream.alloc_zeros::<f32>(conv_dim).expect("out allocates");
    block
        .project_split_gemv(
            &stream,
            qkv_q,
            qkv_s,
            &x_alone,
            &mut out_alone,
            hidden,
            conv_dim,
        )
        .expect("split gemv runs");
    let via_gemv = dtoh(&stream, &out_alone);

    // Three distinct rows so a ragged tile (TT=4 covering 3) is exercised
    // as well as the live columns. Token 0 is the compared row.
    let mut paired = row0.clone();
    paired.extend_from_slice(&row1);
    paired.extend_from_slice(&row2);
    let x_paired = stream.clone_htod(&paired).expect("upload x");
    let mut out_paired = stream
        .alloc_zeros::<f32>(3 * conv_dim)
        .expect("out allocates");
    block
        .project_split_tiled(
            &stream,
            qkv_q,
            qkv_s,
            &x_paired,
            &mut out_paired,
            hidden,
            conv_dim,
            3,
        )
        .expect("split tiled runs");
    let via_tiled = dtoh(&stream, &out_paired);
    let token0_via_tiled = &via_tiled[..conv_dim];

    let diff = max_abs_diff(&via_gemv, token0_via_tiled);
    println!(
        "split gemv (tokens=1) vs split tiled (tokens=3, token 0): \
         max_abs_diff {diff:.3e} over {conv_dim} elements",
    );
    assert_eq!(
        via_gemv, token0_via_tiled,
        "gdn_proj_split_gemv and gdn_proj_split_t* disagree on the same row \
         -- the tiled split kernel is not reduction-order-identical to the \
         GEMV it claims to generalize",
    );
}

/// `project_split_pair` (the `qkv` and `gate` projections under one grid)
/// against the two single-matrix launches it replaces, at every token
/// width the launch has structure at: one, several, exactly a tile, a tile
/// plus a tail, several tiles plus a tail (`(1,4) (2,4) (3,4) (4,4) (8,2)
/// (16,1)` are the tiles).
///
/// Bit-exact, with no tolerance: a warp group in the pair kernel runs the
/// single-matrix body on its own matrix, so every output must be the same
/// chain of additions. A nonzero difference means the seam between the two
/// matrices landed inside a row tile, or the row re-basing is wrong.
#[test]
fn split_layout_pair_agrees_with_the_two_single_launches() {
    let Some((ctx, file, config)) = setup() else {
        return;
    };
    let stream = ctx.default_stream();
    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).expect("schema resolves");
    let geometry = GdnGeometry::from_config(&config, 32, 1e-6);
    let block = GdnBlock::new(&ctx, geometry).expect("kernels compile");
    let weights =
        GdnLayerWeights::upload(&stream, &file, &directory, LAYER).expect("weights upload");
    let int8 = block.repack(&stream, &weights).expect("repack builds");
    let (qkv_q, qkv_s) = int8.qkv();
    let (gate_q, gate_s) = int8.gate();

    let hidden = geometry.hidden;
    let conv_dim = geometry.conv_dim();
    let value_dim = geometry.value_dim();
    let mut rng = Xorshift64Star::new(0x_5EED_9D04);

    for tokens in [1usize, 2, 3, 4, 5, 8, 9, 17] {
        let rows = rng.vec_f32(tokens * hidden, -1.0, 1.0);
        let x = stream.clone_htod(&rows).expect("upload x");

        let mut qkv_single = stream
            .alloc_zeros::<f32>(tokens * conv_dim)
            .expect("out allocates");
        let mut z_single = stream
            .alloc_zeros::<f32>(tokens * value_dim)
            .expect("out allocates");
        if tokens == 1 {
            block
                .project_split_gemv(&stream, qkv_q, qkv_s, &x, &mut qkv_single, hidden, conv_dim)
                .expect("split gemv runs");
            block
                .project_split_gemv(
                    &stream,
                    gate_q,
                    gate_s,
                    &x,
                    &mut z_single,
                    hidden,
                    value_dim,
                )
                .expect("split gemv runs");
        } else {
            block
                .project_split_tiled(
                    &stream,
                    qkv_q,
                    qkv_s,
                    &x,
                    &mut qkv_single,
                    hidden,
                    conv_dim,
                    tokens,
                )
                .expect("split tiled runs");
            block
                .project_split_tiled(
                    &stream,
                    gate_q,
                    gate_s,
                    &x,
                    &mut z_single,
                    hidden,
                    value_dim,
                    tokens,
                )
                .expect("split tiled runs");
        }

        let mut qkv_pair = stream
            .alloc_zeros::<f32>(tokens * conv_dim)
            .expect("out allocates");
        let mut z_pair = stream
            .alloc_zeros::<f32>(tokens * value_dim)
            .expect("out allocates");
        block
            .project_split_pair(
                &stream,
                (qkv_q, qkv_s, &mut qkv_pair, conv_dim),
                (gate_q, gate_s, &mut z_pair, value_dim),
                &x,
                hidden,
                tokens,
            )
            .expect("split pair runs");

        let (a, b) = (dtoh(&stream, &qkv_single), dtoh(&stream, &qkv_pair));
        let (c, d) = (dtoh(&stream, &z_single), dtoh(&stream, &z_pair));
        println!(
            "tokens {tokens}: qkv max_abs_diff {:.3e}, gate max_abs_diff {:.3e}",
            max_abs_diff(&a, &b),
            max_abs_diff(&c, &d),
        );
        assert!(
            a.iter().all(|v| v.is_finite()) && c.iter().all(|v| v.is_finite()),
            "tokens {tokens}: the single launches produced a non-finite value"
        );
        assert_eq!(
            a, b,
            "tokens {tokens}: gdn_proj_split_pair disagrees with the single qkv launch"
        );
        assert_eq!(
            c, d,
            "tokens {tokens}: gdn_proj_split_pair disagrees with the single gate launch"
        );
    }
}
