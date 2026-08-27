//! Locate the ~1e-4-scale batch-vs-single-stream residual that
//! `tests/batch_decode.rs`'s
//! `batched_decode_agrees_with_independent_single_stream_decodes` tolerates:
//! *which layer, and which family within it*, first disagrees.
//!
//! Its rough location -- "further up, in `GatedAttentionBlock`'s and
//! `GdnBlock`'s own batched-vs-single-stream reduction order" -- was inferred
//! from the final logits, not measured layer by layer. This dumps the
//! per-layer, per-stage hidden state both paths produce, via
//! [`Forward::run_with_stage_waypoints`] and
//! [`Forward::run_batch_decode_with_stage_waypoints`], and reports max-abs
//! divergence per layer, tagged by which family produced it: the layer's
//! mixer (Gated DeltaNet or Gated Attention, whichever `layer_kind` says that
//! layer is) and, separately, that layer's MoE block.
//!
//! Both of those drive the same `body`/`body_batch_decode` the engine and the
//! graph capture do -- they only pass a callback that reads the mixer
//! waypoint -- so what this measures is what the engine runs.
//!
//! ```sh
//! CUDA_VISIBLE_DEVICES=<n> cargo run --release -p xabe-engine --bin audit_batch_divergence
//! ```
//!
//! Environment: `LLMXABE_MODEL` overrides the model path, `LLMXABE_CONTEXT`
//! overrides the prefill length (default 2048), `LLMXABE_STEPS` overrides
//! the number of decode steps compared (default 3).

use std::path::PathBuf;
use std::process::ExitCode;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use std::sync::Arc;
use tracing::{error, info, warn};

use xabe_cuda::arena::memory_info;
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_engine::DeviceWeights;
use xabe_engine::SequenceState;
use xabe_engine::forward::{Forward, WaypointStage, arena_holds_for};
use xabe_gguf::GgufFile;
use xabe_model::config::{LayerKind, ModelConfig};
use xabe_model::weights::WeightSchema;

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llmxabe/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

/// Sequences decoded together in the batch path. Matches
/// `tests/batch_decode.rs`'s `BATCH`: three is enough to exercise the
/// batching machinery without inflating the audit's own runtime, and every
/// number reported below is per-sequence so the width does not change what
/// a "family" comparison means.
const BATCH: usize = 3;

/// Largest single prefill pass, same bound `bench_decode_batch` uses so a
/// context above this still prefills in chunks.
const MAX_PREFILL_CHUNK: usize = 2048;

fn model_path() -> PathBuf {
    std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// A chunk width that divides `context` and is at most `MAX_PREFILL_CHUNK`.
fn chunk_for(context: usize) -> usize {
    let mut c = context.min(MAX_PREFILL_CHUNK);
    while c > 1 && !context.is_multiple_of(c) {
        c -= 1;
    }
    c
}

/// One synthetic, in-vocabulary, non-degenerate prompt per sequence. Same
/// generator `tests/batch_decode.rs` and `bench_decode_batch` use, so this
/// audit's inputs are not a coincidentally-easier case than the tests that
/// already measure the residual this exists to locate.
fn synthetic_prompt(seq: usize, len: usize, vocab: usize) -> Vec<i32> {
    (0..len)
        .map(|i| (((seq + 1) * 104_729 + i * 7919 + 1234) % vocab) as i32)
        .collect()
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

/// One waypoint's reading for one sequence: which layer, which stage, and the
/// max-abs divergence between the batch and single-stream hidden state at
/// that point. Family is derived from `(layer, stage)` on demand via
/// [`family_of`] rather than stored, so there is one place that mapping is
/// defined.
#[derive(Clone, Copy)]
struct Reading {
    step: usize,
    layer: Option<u32>,
    stage: WaypointStage,
    seq: usize,
    max_abs: f32,
}

fn family_of(config: &ModelConfig, layer: Option<u32>, stage: WaypointStage) -> &'static str {
    match (layer, stage) {
        (None, WaypointStage::Embed) => "embed",
        (Some(l), WaypointStage::Mixer) => match config.layer_kind(l) {
            LayerKind::GatedDeltaNet => "gdn",
            LayerKind::GatedAttention => "attention",
        },
        (Some(_), WaypointStage::Moe) => "moe",
        _ => "other",
    }
}

fn main() -> ExitCode {
    let _rest = xabe_log::init_from_args();

    if !driver_available() {
        error!("No CUDA driver reachable on this host.");
        return ExitCode::FAILURE;
    }
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            error!("Could not create a context on device 0: {e}");
            return ExitCode::FAILURE;
        }
    };
    let info_dev = DeviceInfo::from_context(0, &ctx).expect("device properties readable");
    if !info_dev.is_supported() {
        error!("Device 0 is below the sm_75 minimum.");
        return ExitCode::FAILURE;
    }
    let path = model_path();
    if !path.exists() {
        warn!(
            "SKIPPED: model file not found at {}; set LLMXABE_MODEL to override",
            path.display(),
        );
        return ExitCode::SUCCESS;
    }

    let context = env_usize("LLMXABE_CONTEXT", 2048);
    let steps = env_usize("LLMXABE_STEPS", 3);

    let stream = ctx.default_stream();
    let file = GgufFile::open(&path).expect("valid GGUF v3");
    let config = ModelConfig::from_gguf(&file).expect("a supported architecture");
    let (free_at_start, total) = memory_info(&ctx).expect("memory info");

    info!(
        "device 0: {} sm_{}{}, {:.1} GiB",
        info_dev.name,
        info_dev.compute_capability.major,
        info_dev.compute_capability.minor,
        total as f64 / (1u64 << 30) as f64,
    );

    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).expect("schema resolves");
    let (weights, load) = DeviceWeights::load_where(&ctx, &stream, &file, &directory, |role| {
        arena_holds_for(config.ffn, role)
    })
    .expect("weight load");
    info!(
        "arena {:.3} GiB in {:.1} s",
        load.bytes as f64 / (1u64 << 30) as f64,
        load.elapsed.as_secs_f64(),
    );

    let chunk = chunk_for(context);
    let max_seq = context + steps + 1;
    let hidden = config.hidden_size as usize;
    let vocab = config.vocab_size as usize;

    info!("context {context} (chunk {chunk}), {steps} decode steps, batch {BATCH}");

    // ---- Build: one prefill pass, a single-stream decode-step pass, and a
    // batch-width decode-step pass, all sharing the resident weights -- same
    // three-shapes-over-one-arena pattern `tests/batch_decode.rs` uses. -----
    let mut prefill = Forward::new(
        &ctx,
        &stream,
        &file,
        &directory,
        &weights,
        config.clone(),
        chunk,
    )
    .expect("prefill pass builds");
    let mut single_step = prefill
        .reshape(&ctx, &stream, &file, &directory, &weights, 1)
        .expect("single-stream decode step builds");
    let mut batch = prefill
        .reshape(&ctx, &stream, &file, &directory, &weights, BATCH)
        .expect("batch-width pass builds");
    batch
        .enable_batch_decode(&ctx, &stream)
        .expect("batch decode scratch allocates");

    let prompts: Vec<Vec<i32>> = (0..BATCH)
        .map(|seq| synthetic_prompt(seq, context, vocab))
        .collect();

    // ---- Reference: each sequence prefilled and decoded entirely on its
    // own, capturing every waypoint's hidden state along the way. -----------
    // The waypoint order is deterministic from the model's own layer
    // pattern -- one `Embed`, then one `Mixer`/`Moe` pair per layer, in the
    // exact order `run_with_stage_waypoints`/`run_batch_decode_with_stage_
    // waypoints` call `on_waypoint` -- so it does not need to be recorded
    // from a live call.
    let waypoint_order: Vec<(Option<u32>, WaypointStage)> = {
        let mut order = vec![(None, WaypointStage::Embed)];
        for layer in 0..config.num_layers {
            order.push((Some(layer), WaypointStage::Mixer));
            order.push((Some(layer), WaypointStage::Moe));
        }
        order
    };

    let mut ref_readings: Vec<Vec<Vec<f32>>> = vec![Vec::new(); BATCH]; // [seq][waypoint idx] -> hidden
    let mut ref_states: Vec<SequenceState> = Vec::with_capacity(BATCH);
    let mut ref_next: Vec<i32> = Vec::with_capacity(BATCH);
    for p in &prompts {
        let mut state = prefill
            .new_state(&stream, max_seq)
            .expect("state allocates");
        for piece in p.chunks(chunk) {
            prefill
                .run(&stream, &mut state, piece, |_, _| {})
                .expect("prefill runs");
        }
        ref_next.push(prefill.sample_argmax(&stream).expect("prefill argmax"));
        ref_states.push(state);
    }
    for _step in 0..steps {
        for seq in 0..BATCH {
            let mut waypoints = Vec::new();
            single_step
                .run_with_stage_waypoints(
                    &stream,
                    &mut ref_states[seq],
                    &[ref_next[seq]],
                    |layer, stage, buf| {
                        waypoints.push((layer, stage, dtoh(&stream, buf)));
                    },
                )
                .expect("single-stream decode step runs");
            for (_, _, host) in &waypoints {
                ref_readings[seq].push(host.clone());
            }
            ref_next[seq] = single_step.sample_argmax(&stream).expect("argmax");
        }
    }

    // ---- Batch: the same `BATCH` prompts, decoded together, capturing the
    // same waypoints as one `n * hidden` buffer per call. --------------------
    let mut batch_readings: Vec<Vec<f32>> = Vec::new(); // [waypoint idx] -> n*hidden
    let mut batch_states: Vec<SequenceState> = Vec::with_capacity(BATCH);
    let mut batch_next: Vec<i32> = Vec::with_capacity(BATCH);
    for p in &prompts {
        let mut state = prefill
            .new_state(&stream, max_seq)
            .expect("state allocates");
        for piece in p.chunks(chunk) {
            prefill
                .run(&stream, &mut state, piece, |_, _| {})
                .expect("prefill runs");
        }
        batch_next.push(prefill.sample_argmax(&stream).expect("prefill argmax"));
        batch_states.push(state);
    }
    for _step in 0..steps {
        let mut waypoints = Vec::new();
        let sampled = batch
            .run_batch_decode_with_stage_waypoints(
                &stream,
                &mut batch_states,
                &batch_next,
                |layer, stage, buf| {
                    waypoints.push((layer, stage, dtoh(&stream, buf)));
                },
            )
            .expect("batch decode step runs");
        for (_, _, host) in waypoints {
            batch_readings.push(host);
        }
        batch_next = sampled;
    }

    // ---- Compare: for every (step, waypoint, sequence), max-abs diff
    // between the single-stream reference row and the batch row. ------------
    let n_waypoints = waypoint_order.len();
    let mut readings: Vec<Reading> = Vec::new();
    for step in 0..steps {
        for (wi, &(layer, stage)) in waypoint_order.iter().enumerate() {
            let batch_buf = &batch_readings[step * n_waypoints + wi];
            debug_assert_eq!(batch_buf.len(), BATCH * hidden);
            for seq in 0..BATCH {
                let ref_buf = &ref_readings[seq][step * n_waypoints + wi];
                let batch_row = &batch_buf[seq * hidden..(seq + 1) * hidden];
                let diff = max_abs_diff(ref_buf, batch_row);
                readings.push(Reading {
                    step,
                    layer,
                    stage,
                    seq,
                    max_abs: diff,
                });
            }
        }
    }

    // ---- Report: per-layer, per-family max-abs divergence, plus the
    // headline table the audit exists to produce. ---------------------------
    info!("");
    info!(
        "{:>5} {:>10} {:>10} {:>12} {:>8} {:>5}",
        "layer", "family", "stage", "max_abs", "at step", "seq"
    );
    let mut by_layer_family: Vec<(Option<u32>, &'static str, f32)> = Vec::new();
    for &(layer, stage) in &waypoint_order {
        let family = family_of(&config, layer, stage);
        // The full reading at the worst-case (step, seq), not just its
        // number -- a max that always lands at the same step/seq pair looks
        // different from one that wanders, and only the readings carry that.
        let worst = readings
            .iter()
            .filter(|r| r.layer == layer && r.stage == stage)
            .max_by(|a, b| a.max_abs.total_cmp(&b.max_abs))
            .expect("every waypoint has at least one reading");
        info!(
            "{:>5} {:>10} {:>10?} {:>12.3e} {:>8} {:>5}",
            layer.map(|l| l as i64).unwrap_or(-1),
            family,
            stage,
            worst.max_abs,
            worst.step,
            worst.seq,
        );
        by_layer_family.push((layer, family, worst.max_abs));
    }

    info!("");
    info!("=== headline: kernel family -> max_abs divergence at layer 1 -> growth by layer 40 ===");
    for target_family in ["gdn", "attention", "moe"] {
        let mut points: Vec<(u32, f32)> = by_layer_family
            .iter()
            .filter(|(_, f, _)| *f == target_family)
            .filter_map(|(l, _, v)| l.map(|l| (l, *v)))
            .collect();
        points.sort_by_key(|(l, _)| *l);
        if let (Some(&(first_l, first_v)), Some(&(last_l, last_v))) =
            (points.first(), points.last())
        {
            info!(
                "{target_family:>10}: layer {first_l:>2} = {first_v:.3e}   ->   layer {last_l:>2} = {last_v:.3e}   ({} layers of this kind)",
                points.len(),
            );
        } else {
            info!("{target_family:>10}: no layers of this kind");
        }
    }
    let embed_diff = by_layer_family
        .iter()
        .find(|(l, f, _)| l.is_none() && *f == "embed")
        .map(|(_, _, v)| *v)
        .unwrap_or(0.0);
    info!(
        "{:>10}: {:.3e} (should be exactly 0.0 -- a pure lookup)",
        "embed", embed_diff
    );

    // ---- Per-sample correlation: for the exact (layer, step, seq) that
    // maximizes the mixer's own diff, what did the MoE stage on that SAME
    // sample read? A `moe/mixer` ratio near 1x is pure inheritance -- MoE
    // faithfully carrying forward whatever the mixer already handed it,
    // with nothing of its own to fix. A ratio far above 1x is MoE (or the
    // router's discrete top-8 decision it feeds) adding real divergence of
    // its own on top. Independent per-stage maxes (the table above) cannot
    // tell these apart, because they are free to land on different
    // (step, seq) pairs; this reads both stages off the *same* sample. ----
    info!("");
    info!("=== per-sample correlation: mixer diff vs moe diff on the same (layer, step, seq) ===");
    info!(
        "{:>5} {:>10} {:>12} {:>12} {:>8}",
        "layer", "family", "mixer", "moe (same)", "ratio"
    );
    for &(layer, stage) in &waypoint_order {
        if stage != WaypointStage::Mixer {
            continue;
        }
        let Some(layer) = layer else { continue };
        let family = family_of(&config, Some(layer), WaypointStage::Mixer);
        let worst_mixer = readings
            .iter()
            .filter(|r| r.layer == Some(layer) && r.stage == WaypointStage::Mixer)
            .max_by(|a, b| a.max_abs.total_cmp(&b.max_abs))
            .expect("every mixer waypoint has at least one reading");
        let same_sample_moe = readings
            .iter()
            .find(|r| {
                r.layer == Some(layer)
                    && r.stage == WaypointStage::Moe
                    && r.step == worst_mixer.step
                    && r.seq == worst_mixer.seq
            })
            .expect("every (layer, step, seq) has both a mixer and a moe reading");
        let ratio = if worst_mixer.max_abs > 0.0 {
            same_sample_moe.max_abs / worst_mixer.max_abs
        } else {
            f32::INFINITY
        };
        info!(
            "{:>5} {:>10} {:>12.3e} {:>12.3e} {:>7.2}x",
            layer, family, worst_mixer.max_abs, same_sample_moe.max_abs, ratio,
        );
    }

    let (free_now, _) = memory_info(&ctx).expect("memory info");
    info!(
        "peak VRAM {:.3} GiB of {:.2} GiB",
        free_at_start.saturating_sub(free_now) as f64 / (1u64 << 30) as f64,
        total as f64 / (1u64 << 30) as f64,
    );

    ExitCode::SUCCESS
}
