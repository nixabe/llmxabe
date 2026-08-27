//! `block::gdn_verify`'s window-local snapshot composition, checked against
//! the plain chunked forward pass on the real model's weights.
//!
//! A speculative-decode verify step needs the
//! Gated DeltaNet recurrent state snapshotted at every position boundary
//! inside the window, not just at the end, so a partially-accepted draft can
//! roll back to exactly the right point without a second weight-read pass.
//! `crates/xabe-engine/src/block/gdn_verify.rs` gets there by replicating
//! `GdnBlock::run`'s ten steps with the two state-touching ones (the
//! convolution and the delta rule) unrolled to one token at a time — see
//! that file's module docs for why that stays correct without changing
//! `xabe-cuda` or `gdn.rs` at all.
//!
//! This file is the check that the unrolling really did stay correct:
//! [`GdnBlock::forward`] over a whole window, committing
//! [`run_layer_with_snapshots`]'s ring at "everything accepted", must land on
//! the same output and the same final `conv`/`recurrent` state as the plain
//! call — `Tolerance::gdn_chunk_vs_recurrent()`, the tolerance this project
//! already uses for chunked-vs-recurrent equivalence, because that is
//! exactly what this comparison is: two different decompositions of the same
//! arithmetic, not two different formulas.
//!
//! SKIPS — reporting that it skipped — without a driver, a supported device,
//! or the model file.

use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_engine::block::gdn::{GdnBlock, GdnGeometry, GdnLayerWeights};
use xabe_engine::block::gdn_verify::{GdnSnapshotRing, GdnVerifyScratch, run_layer_with_snapshots};
use xabe_gguf::GgufFile;
use xabe_kernels::compare::{Tolerance, check, compare};
use xabe_model::config::ModelConfig;
use xabe_model::weights::WeightSchema;

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llmxabe/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

/// A real Gated DeltaNet layer (not the boundary attention layer 3/7/...).
const LAYER: u32 = 4;

/// `1 + d` at `d = 3` — the width `docs/SCHEDULER.md`'s
/// `DEFAULT_DRAFT_TOKENS_PER_STEP` budgets a verify step for.
const WINDOW: usize = 4;

fn model_path() -> PathBuf {
    std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
}

struct Fixture {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    file: GgufFile,
    config: ModelConfig,
    geometry: GdnGeometry,
}

fn setup() -> Option<Fixture> {
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
    let file = GgufFile::open(&path).expect("model file must parse as valid GGUF v3");
    let config = ModelConfig::qwen3_6_35b_a3b();
    let geometry = GdnGeometry::from_gguf(&config, &file, WINDOW)
        .expect("the file must carry qwen35moe.attention.layer_norm_rms_epsilon");
    let stream = ctx.default_stream();
    println!(
        "device 0: {}, layer {LAYER}, window {WINDOW}, rms_eps {:e}",
        info.name, geometry.rms_eps,
    );
    Some(Fixture {
        ctx,
        stream,
        file,
        config,
        geometry,
    })
}

/// A small deterministic pseudo-random hidden state — this test checks
/// internal self-consistency between two code paths on the same real
/// weights, not agreement with a captured oracle, so any finite input that
/// does not drive this layer's real decay rates to overflow will do. Values
/// are kept small (`[-0.5, 0.5)`) for the same reason
/// `gdn_chunked_differential.rs`'s synthetic inputs are: this layer's real
/// `ssm_a` reaches a per-token log-decay of -91.58 (block 0, head 9), and a
/// large-magnitude activation is what turned that into an `inf` the first
/// time (see that file's module docs).
fn pseudo_random_hidden(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            // xorshift64*, cheap and deterministic; the exact distribution
            // does not matter here.
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * 1.0
        })
        .collect()
}

fn dtoh(stream: &Arc<CudaStream>, buf: &CudaSlice<f32>) -> Vec<f32> {
    let host = stream.clone_dtoh(buf).expect("copy back");
    stream.synchronize().expect("sync");
    host
}

fn directory<'a>(file: &'a GgufFile, config: &ModelConfig) -> xabe_model::weights::Directory<'a> {
    let schema: &'static WeightSchema = Box::leak(Box::new(WeightSchema::new(config)));
    schema.resolve(file).expect("schema resolves")
}

#[test]
fn committing_the_full_ring_matches_a_plain_chunked_forward() {
    let Some(fx) = setup() else { return };
    let dir = directory(&fx.file, &fx.config);
    let weights =
        GdnLayerWeights::upload(&fx.stream, &fx.file, &dir, LAYER).expect("layer 4 weights upload");

    let hidden_vals = pseudo_random_hidden(WINDOW * fx.geometry.hidden, 0xC0FFEE);
    let hidden = fx.stream.clone_htod(&hidden_vals).expect("hidden upload");

    let mut block = GdnBlock::new(&fx.ctx, fx.geometry).expect("kernels compile");

    // --- reference: one plain forward over the whole window -----------
    let mut ref_state = block.state(&fx.stream).expect("ref state");
    let mut ref_out = fx
        .stream
        .alloc_zeros::<f32>(WINDOW * fx.geometry.hidden)
        .expect("ref out");
    block
        .forward(
            &fx.stream,
            &weights,
            None,
            &mut ref_state,
            &hidden,
            &mut ref_out,
        )
        .expect("reference forward");

    // --- candidate: the snapshotted composition, committed at "accept
    //     everything" -----------------------------------------------------
    let mut cand_state = block.state(&fx.stream).expect("candidate state");
    let mut cand_out = fx
        .stream
        .alloc_zeros::<f32>(WINDOW * fx.geometry.hidden)
        .expect("candidate out");
    let mut scratch =
        GdnVerifyScratch::new(&fx.stream, &fx.geometry, WINDOW).expect("verify scratch");
    let mut ring =
        GdnSnapshotRing::new(&fx.stream, &fx.geometry, WINDOW + 1).expect("snapshot ring");
    run_layer_with_snapshots(
        &fx.stream,
        &mut block,
        &weights,
        None,
        &mut cand_state,
        &hidden,
        &mut cand_out,
        &mut scratch,
        &mut ring,
        WINDOW,
    )
    .expect("snapshotted forward");
    ring.commit(&fx.stream, WINDOW, &mut cand_state)
        .expect("commit at full acceptance");

    let tol = Tolerance::gdn_chunk_vs_recurrent();

    let out_result = compare(&dtoh(&fx.stream, &cand_out), &dtoh(&fx.stream, &ref_out));
    println!("out:       {out_result}");
    assert!(
        check(&out_result, &tol).is_pass(),
        "mixer output diverges: {out_result}",
    );

    let conv_result = compare(
        &dtoh(&fx.stream, &cand_state.conv),
        &dtoh(&fx.stream, &ref_state.conv),
    );
    println!("conv:      {conv_result}");
    assert!(
        check(&conv_result, &tol).is_pass(),
        "committed conv state diverges: {conv_result}",
    );

    let rec_result = compare(
        &dtoh(&fx.stream, &cand_state.recurrent),
        &dtoh(&fx.stream, &ref_state.recurrent),
    );
    println!("recurrent: {rec_result}");
    assert!(
        check(&rec_result, &tol).is_pass(),
        "committed recurrent state diverges: {rec_result}",
    );

    println!(
        "committing GdnSnapshotRing at full acceptance reproduces a plain \
         chunked forward: output, conv state and recurrent state all within \
         {tol:?}",
    );
}

#[test]
fn committing_a_partial_ring_slot_matches_a_recurrent_prefix() {
    // The rollback case: accept only the first two of four window positions,
    // and check that committing slot 2 leaves the state a plain forward over
    // *only* the first two tokens would have left. This is the invariant the
    // whole mechanism exists for — a rejected draft's tail must vanish from
    // the recurrent state as if it had never been folded in, at zero extra
    // weight-read cost.
    let Some(fx) = setup() else { return };
    let dir = directory(&fx.file, &fx.config);
    let weights =
        GdnLayerWeights::upload(&fx.stream, &fx.file, &dir, LAYER).expect("layer 4 weights upload");

    let hidden_vals = pseudo_random_hidden(WINDOW * fx.geometry.hidden, 0xC0FFEE);
    let hidden = fx.stream.clone_htod(&hidden_vals).expect("hidden upload");
    let accepted = 2usize;
    let prefix_vals = &hidden_vals[..accepted * fx.geometry.hidden];
    let prefix_hidden = fx.stream.clone_htod(prefix_vals).expect("prefix upload");

    let mut block = GdnBlock::new(&fx.ctx, fx.geometry).expect("kernels compile");

    // --- reference: a plain forward over only the accepted prefix ------
    let mut ref_state = block.state(&fx.stream).expect("ref state");
    let mut ref_out = fx
        .stream
        .alloc_zeros::<f32>(accepted * fx.geometry.hidden)
        .expect("ref out");
    block
        .forward(
            &fx.stream,
            &weights,
            None,
            &mut ref_state,
            &prefix_hidden,
            &mut ref_out,
        )
        .expect("reference forward");

    // --- candidate: the full four-token window, committed at slot 2 ----
    let mut cand_state = block.state(&fx.stream).expect("candidate state");
    let mut cand_out = fx
        .stream
        .alloc_zeros::<f32>(WINDOW * fx.geometry.hidden)
        .expect("candidate out");
    let mut scratch =
        GdnVerifyScratch::new(&fx.stream, &fx.geometry, WINDOW).expect("verify scratch");
    let mut ring =
        GdnSnapshotRing::new(&fx.stream, &fx.geometry, WINDOW + 1).expect("snapshot ring");
    run_layer_with_snapshots(
        &fx.stream,
        &mut block,
        &weights,
        None,
        &mut cand_state,
        &hidden,
        &mut cand_out,
        &mut scratch,
        &mut ring,
        WINDOW,
    )
    .expect("snapshotted forward");
    ring.commit(&fx.stream, accepted, &mut cand_state)
        .expect("commit at partial acceptance");

    let tol = Tolerance::gdn_chunk_vs_recurrent();

    let conv_result = compare(
        &dtoh(&fx.stream, &cand_state.conv),
        &dtoh(&fx.stream, &ref_state.conv),
    );
    println!("conv (rollback):      {conv_result}");
    assert!(
        check(&conv_result, &tol).is_pass(),
        "rolled-back conv state diverges from the accepted-prefix-only state: {conv_result}",
    );

    let rec_result = compare(
        &dtoh(&fx.stream, &cand_state.recurrent),
        &dtoh(&fx.stream, &ref_state.recurrent),
    );
    println!("recurrent (rollback): {rec_result}");
    assert!(
        check(&rec_result, &tol).is_pass(),
        "rolled-back recurrent state diverges from the accepted-prefix-only state: {rec_result}",
    );

    println!(
        "committing GdnSnapshotRing at a partial acceptance ({accepted}/{WINDOW}) reproduces \
         a plain forward over only the accepted prefix — the rollback the whole mechanism \
         exists for.",
    );
}
