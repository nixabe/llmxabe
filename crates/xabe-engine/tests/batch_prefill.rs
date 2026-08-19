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
    "/home/nixabe/llama.cpp/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";
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
