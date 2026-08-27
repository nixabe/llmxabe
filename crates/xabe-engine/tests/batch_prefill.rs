//! Equal-chunk N=3 prefill must agree with three independent serial passes,
//! including across a carried chunk boundary.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaContext, CudaStream};
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_engine::forward::{Forward, arena_holds};
use xabe_engine::{DeviceWeights, SequenceState};
use xabe_gguf::GgufFile;
use xabe_model::config::ModelConfig;
use xabe_model::weights::WeightSchema;

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llmxabe/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";
const N: usize = 3;
// Keep both the serial and flattened shapes on the production tensor-core
// projection path. Tiny shapes deliberately select different projection
// kernels and test their cross-kernel rounding instead of batch indexing.
const CHUNK: usize = 64;
const CHUNKS: usize = 2;
const MAX_ABS_DIFF: f32 = 5e-3;
const MIN_COSINE: f32 = 0.999;

static RESIDENT_MODEL: Mutex<()> = Mutex::new(());

fn setup() -> Option<(Arc<CudaContext>, GgufFile)> {
    if !driver_available() {
        println!("SKIPPED: no CUDA driver present");
        return None;
    }
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(error) => {
            println!("SKIPPED: could not create a context on device 0: {error}");
            return None;
        }
    };
    let info = DeviceInfo::from_context(0, &ctx).expect("device properties readable");
    if !info.is_supported() {
        println!("SKIPPED: device 0 is below the sm_75 minimum");
        return None;
    }
    let path = std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH));
    if !path.exists() {
        println!(
            "SKIPPED: model file not found at {}; set LLMXABE_MODEL to override",
            path.display()
        );
        return None;
    }
    Some((ctx, GgufFile::open(path).expect("valid GGUF v3")))
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in a.iter().zip(b) {
        dot += f64::from(a) * f64::from(b);
        aa += f64::from(a) * f64::from(a);
        bb += f64::from(b) * f64::from(b);
    }
    (dot / (aa.sqrt() * bb.sqrt())) as f32
}

fn prompt(sequence: usize, vocab: usize) -> Vec<i32> {
    (0..CHUNK * CHUNKS)
        .map(|i| (((sequence + 3) * 65_537 + i * 7_919 + 17) % vocab) as i32)
        .collect()
}

fn read(stream: &Arc<CudaStream>, data: &cudarc::driver::CudaSlice<f32>) -> Vec<f32> {
    let host = stream.clone_dtoh(data).expect("device read-back");
    stream.synchronize().expect("read-back completes");
    host
}

/// The full-prompt flattened shape the 2K prefill throughput claim rests on:
/// one 6,138-row physical pass carrying three whole 2,046-token prompts,
/// against three serial 2,046-row passes.
///
/// The carried-chunk test above gates the flattening math at 64-token
/// chunks; this gates the exact shape `docs/BENCHMARKS.md`'s 2026-08-20
/// prefill head-to-head measured (3,737--3,804 tok/s against llama.cpp's
/// tuned 3,342.41), so that number is tied to a correctness result at its
/// own width rather than to a smaller one's. Same gates as above: per-token
/// logit tolerance and exact argmax agreement per sequence.
///
/// SKIPS — reporting that it skipped — without a driver, a supported device,
/// or the model file.
#[test]
fn n3_full_prompt_flattened_prefill_matches_serial_at_2046_rows_per_sequence() {
    let _resident = RESIDENT_MODEL.lock().unwrap_or_else(|e| e.into_inner());
    let Some((ctx, file)) = setup() else {
        return;
    };
    // 2,046 rows per sequence is the claimed shape; `LLMXABE_TEST_WIDE_CHUNK`
    // narrows it for width bisection when this gate fails.
    let wide_chunk: usize = std::env::var("LLMXABE_TEST_WIDE_CHUNK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2046);
    let stream = ctx.default_stream();
    let config = ModelConfig::qwen3_6_35b_a3b();
    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).expect("schema resolves");
    let (weights, _) = DeviceWeights::load_where(&ctx, &stream, &file, &directory, arena_holds)
        .expect("weights load");
    let vocab = config.vocab_size as usize;
    // `LLMXABE_TEST_IDENTICAL_PROMPTS=1` feeds every sequence the same
    // prompt: the flattened pass must then produce three bit-identical
    // logit rows, so any disagreement between them is cross-sequence
    // contamination rather than width-dependent rounding.
    let identical = std::env::var_os("LLMXABE_TEST_IDENTICAL_PROMPTS").is_some();
    let prompts: Vec<Vec<i32>> = (0..N)
        .map(|sequence| {
            let seed = if identical { 3 } else { sequence + 3 };
            (0..wide_chunk)
                .map(|i| ((seed * 65_537 + i * 7_919 + 17) % vocab) as i32)
                .collect()
        })
        .collect();

    let mut serial = Forward::new(
        &ctx,
        &stream,
        &file,
        &directory,
        &weights,
        config.clone(),
        wide_chunk,
    )
    .expect("serial 2046-row shape builds");
    let mut batch = serial
        .reshape(&ctx, &stream, &file, &directory, &weights, N * wide_chunk)
        .expect("flattened 6138-row shape builds");
    batch
        .enable_batch_prefill(&ctx, &stream, N)
        .expect("batch output scratch allocates");

    let mut serial_states: Vec<SequenceState> = (0..N)
        .map(|_| serial.new_state(&stream, wide_chunk).expect("serial state"))
        .collect();
    let mut batch_states: Vec<SequenceState> = (0..N)
        .map(|_| batch.new_state(&stream, wide_chunk).expect("batch state"))
        .collect();

    let mut reference_logits = Vec::with_capacity(N);
    let mut reference_ids = Vec::with_capacity(N);
    for sequence in 0..N {
        serial
            .run(
                &stream,
                &mut serial_states[sequence],
                &prompts[sequence],
                |_, _| {},
            )
            .expect("serial prompt runs");
        reference_logits.push(read(&stream, serial.logits()));
        reference_ids.push(serial.sample_argmax(&stream).expect("serial argmax"));
    }

    let mut flattened = Vec::with_capacity(N * wide_chunk);
    for prompt in &prompts {
        flattened.extend_from_slice(prompt);
    }
    let batch_ids = batch
        .run_batch_prefill(&stream, &mut batch_states, &flattened)
        .expect("flattened pass runs");
    let batch_logits = read(
        &stream,
        batch.batch_logits().expect("batch logits were enabled"),
    );
    if identical {
        for sequence in 1..N {
            let a = &batch_logits[..vocab];
            let b = &batch_logits[sequence * vocab..(sequence + 1) * vocab];
            let diff = max_abs_diff(a, b);
            println!("identical prompts: batch row {sequence} vs row 0 max_abs={diff:.6e}");
        }
    }
    for sequence in 0..N {
        let row = &batch_logits[sequence * vocab..(sequence + 1) * vocab];
        let max_abs = max_abs_diff(row, &reference_logits[sequence]);
        let cos = cosine(row, &reference_logits[sequence]);
        println!("wide sequence {sequence}: max_abs={max_abs:.6e}, cosine={cos:.9}");
        assert!(
            max_abs <= MAX_ABS_DIFF,
            "max_abs {max_abs:e} exceeds {MAX_ABS_DIFF:e}"
        );
        assert!(
            cos >= MIN_COSINE,
            "cosine {cos:.9} is below {MIN_COSINE:.9}"
        );
        assert_eq!(
            batch_ids[sequence], reference_ids[sequence],
            "sequence {sequence}: the flattened pass sampled a different token",
        );
    }
    assert!(
        batch_states
            .iter()
            .all(|state| state.position() == wide_chunk)
    );
}

/// Diagnostic, not a gate: finds the first layer and stage where the wide
/// flattened pass diverges from serial. Runs only when
/// `LLMXABE_AUDIT_FLAT_PREFILL=1` — it prints a per-layer table and always
/// "passes", because its output is the evidence a fix starts from, not a
/// bound. All three sequences get the SAME prompt, so one serial run is the
/// reference for every flattened row range and any disagreement BETWEEN
/// batch rows is cross-sequence contamination.
#[test]
fn audit_where_the_wide_flattened_prefill_first_diverges() {
    if std::env::var_os("LLMXABE_AUDIT_FLAT_PREFILL").is_none() {
        println!("SKIPPED: set LLMXABE_AUDIT_FLAT_PREFILL=1 to run this audit");
        return;
    }
    let _resident = RESIDENT_MODEL.lock().unwrap_or_else(|e| e.into_inner());
    let Some((ctx, file)) = setup() else {
        return;
    };
    let wide_chunk: usize = std::env::var("LLMXABE_TEST_WIDE_CHUNK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let stream = ctx.default_stream();
    let config = ModelConfig::qwen3_6_35b_a3b();
    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).expect("schema resolves");
    let (weights, _) = DeviceWeights::load_where(&ctx, &stream, &file, &directory, arena_holds)
        .expect("weights load");
    let vocab = config.vocab_size as usize;
    let hidden = config.hidden_size as usize;
    let prompt: Vec<i32> = (0..wide_chunk)
        .map(|i| ((3 * 65_537 + i * 7_919 + 17) % vocab) as i32)
        .collect();

    let mut serial = Forward::new(
        &ctx,
        &stream,
        &file,
        &directory,
        &weights,
        config.clone(),
        wide_chunk,
    )
    .expect("serial shape builds");
    let mut batch = serial
        .reshape(&ctx, &stream, &file, &directory, &weights, N * wide_chunk)
        .expect("flattened shape builds");
    batch
        .enable_batch_prefill(&ctx, &stream, N)
        .expect("batch scratch allocates");

    // One serial reference pass, all per-stage buffers kept on the host.
    let mut serial_state = serial.new_state(&stream, wide_chunk).expect("state");
    let mut reference: Vec<(Option<u32>, xabe_engine::forward::WaypointStage, Vec<f32>)> =
        Vec::new();
    serial
        .run_with_stage_waypoints(&stream, &mut serial_state, &prompt, |layer, stage, buf| {
            let host = stream.clone_dtoh(buf).expect("waypoint read");
            stream.synchronize().expect("sync");
            reference.push((layer, stage, host[..wide_chunk * hidden].to_vec()));
        })
        .expect("serial waypoint run");

    let mut batch_states: Vec<SequenceState> = (0..N)
        .map(|_| batch.new_state(&stream, wide_chunk).expect("state"))
        .collect();
    let mut flattened = Vec::with_capacity(N * wide_chunk);
    for _ in 0..N {
        flattened.extend_from_slice(&prompt);
    }

    let mut cursor = 0usize;
    let mut first_disagreement: Option<String> = None;
    batch
        .run_batch_prefill_with_stage_waypoints(
            &stream,
            &mut batch_states,
            &flattened,
            |layer, stage, buf| {
                let host = stream.clone_dtoh(buf).expect("waypoint read");
                stream.synchronize().expect("sync");
                let (ref_layer, ref_stage, ref_rows) = &reference[cursor];
                assert_eq!((*ref_layer, *ref_stage), (layer, stage), "waypoint order");
                let mut per_seq = [0.0f32; N];
                for (s, slot) in per_seq.iter_mut().enumerate() {
                    let rows = &host[s * wide_chunk * hidden..(s + 1) * wide_chunk * hidden];
                    *slot = max_abs_diff(rows, ref_rows);
                }
                let cross = max_abs_diff(
                    &host[..wide_chunk * hidden],
                    &host[wide_chunk * hidden..2 * wide_chunk * hidden],
                );
                if per_seq.iter().any(|&d| d > 0.0) || cross > 0.0 {
                    let line = format!(
                        "layer {layer:?} stage {stage:?}: vs-serial {per_seq:?}, seq1-vs-seq0 {cross:.3e}",
                    );
                    println!("{line}");
                    if first_disagreement.is_none() {
                        first_disagreement = Some(line);
                    }
                }
                cursor += 1;
            },
        )
        .expect("batch waypoint run");
    match first_disagreement {
        Some(line) => println!("FIRST DIVERGENCE: {line}"),
        None => println!("no divergence at any waypoint — the tail must be downstream"),
    }
}

#[test]
fn n3_batch_prefill_matches_serial_across_a_carried_chunk() {
    let _resident = RESIDENT_MODEL.lock().unwrap_or_else(|e| e.into_inner());
    let Some((ctx, file)) = setup() else {
        return;
    };
    let stream = ctx.default_stream();
    let config = ModelConfig::qwen3_6_35b_a3b();
    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).expect("schema resolves");
    let (weights, _) = DeviceWeights::load_where(&ctx, &stream, &file, &directory, arena_holds)
        .expect("weights load");
    let prompts: Vec<_> = (0..N)
        .map(|sequence| prompt(sequence, config.vocab_size as usize))
        .collect();

    let mut serial = Forward::new(
        &ctx,
        &stream,
        &file,
        &directory,
        &weights,
        config.clone(),
        CHUNK,
    )
    .expect("serial shape builds");
    let mut batch = serial
        .reshape(&ctx, &stream, &file, &directory, &weights, N * CHUNK)
        .expect("flattened batch shape builds");
    batch
        .enable_batch_prefill(&ctx, &stream, N)
        .expect("batch output scratch allocates");

    let max_seq = CHUNK * CHUNKS;
    let mut serial_states: Vec<SequenceState> = (0..N)
        .map(|_| serial.new_state(&stream, max_seq).expect("serial state"))
        .collect();
    let mut batch_states: Vec<SequenceState> = (0..N)
        .map(|_| batch.new_state(&stream, max_seq).expect("batch state"))
        .collect();

    for chunk in 0..CHUNKS {
        let range = chunk * CHUNK..(chunk + 1) * CHUNK;
        let mut reference_logits = Vec::with_capacity(N);
        let mut reference_ids = Vec::with_capacity(N);
        for sequence in 0..N {
            serial
                .run(
                    &stream,
                    &mut serial_states[sequence],
                    &prompts[sequence][range.clone()],
                    |_, _| {},
                )
                .expect("serial chunk runs");
            reference_logits.push(read(&stream, serial.logits()));
            reference_ids.push(serial.sample_argmax(&stream).expect("serial argmax"));
        }

        let mut flattened = Vec::with_capacity(N * CHUNK);
        for prompt in &prompts {
            flattened.extend_from_slice(&prompt[range.clone()]);
        }
        let batch_ids = batch
            .run_batch_prefill(&stream, &mut batch_states, &flattened)
            .expect("batch chunk runs");
        let batch_logits = read(
            &stream,
            batch.batch_logits().expect("batch logits were enabled"),
        );
        let vocab = config.vocab_size as usize;
        for sequence in 0..N {
            let row = &batch_logits[sequence * vocab..(sequence + 1) * vocab];
            let max_abs = max_abs_diff(row, &reference_logits[sequence]);
            let cos = cosine(row, &reference_logits[sequence]);
            println!("chunk {chunk} sequence {sequence}: max_abs={max_abs:.6e}, cosine={cos:.9}");
            assert!(
                max_abs <= MAX_ABS_DIFF,
                "max_abs {max_abs:e} exceeds {MAX_ABS_DIFF:e}"
            );
            assert!(
                cos >= MIN_COSINE,
                "cosine {cos:.9} is below {MIN_COSINE:.9}"
            );
            assert_eq!(batch_ids[sequence], reference_ids[sequence]);
        }
        assert!(
            batch_states
                .iter()
                .all(|state| state.position() == range.end)
        );
    }
}
