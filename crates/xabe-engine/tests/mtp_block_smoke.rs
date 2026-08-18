//! `block::mtp::MtpBlock` wired against the real model file.
//!
//! Not a differential test against a captured oracle — no MTP capture exists
//! yet (`docs/ORACLE.md` does not cover block 40) — so this checks the thing
//! a wrong shape or a wrong tensor role would fail first: the pass runs
//! without erroring, and every output is finite and in the vocabulary's
//! range. `tests/gdn_verify_differential.rs` covers the numerically
//! sensitive half of R6 (the recurrent-state rollback); this file covers
//! block 40's own wiring — the loader, the `eh_proj` GEMM, the concat, and
//! the dense-attention + MoE block reused from layer index `config.
//! num_layers`.
//!
//! SKIPS — reporting that it skipped — without a driver, a supported device,
//! or the model file.

use std::path::PathBuf;

use cudarc::driver::CudaContext;
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_engine::block::attention::KvCache;
use xabe_engine::block::mtp::MtpBlock;
use xabe_engine::weights::DeviceWeights;
use xabe_gguf::GgufFile;
use xabe_model::config::ModelConfig;
use xabe_model::weights::WeightSchema;

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llama.cpp/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";
const RMS_EPS_KEY: &str = "qwen35moe.attention.layer_norm_rms_epsilon";
const ROPE_FREQ_BASE_KEY: &str = "qwen35moe.rope.freq_base";

fn model_path() -> PathBuf {
    std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
}

#[test]
fn the_draft_head_runs_and_emits_a_token_in_vocabulary_range() {
    if !driver_available() {
        println!("SKIPPED: no CUDA driver present");
        return;
    }
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            println!("SKIPPED: could not create a context on device 0: {e}");
            return;
        }
    };
    let info = DeviceInfo::from_context(0, &ctx).expect("device properties readable");
    if !info.is_supported() {
        println!("SKIPPED: device 0 is below the sm_75 minimum");
        return;
    }
    let path = model_path();
    if !path.exists() {
        println!(
            "SKIPPED: model file not found at {}; set LLMXABE_MODEL to override",
            path.display(),
        );
        return;
    }

    let file = GgufFile::open(&path).expect("model file must parse as valid GGUF v3");
    let config = ModelConfig::qwen3_6_35b_a3b();
    let rms_eps = file.get_f32(RMS_EPS_KEY).expect("rms eps present");
    let rope_theta = file
        .get_f32(ROPE_FREQ_BASE_KEY)
        .expect("rope theta present");

    let schema = WeightSchema::with_mtp(&config);
    let directory = schema
        .resolve(&file)
        .expect("schema (with MTP) must resolve against the model file");

    let stream = ctx.default_stream();
    let (weights, report) =
        DeviceWeights::load(&ctx, &stream, &file, &directory).expect("weight load");
    println!(
        "loaded {} tensors, {:.2} GiB (block 40 included)",
        report.tensors,
        report.bytes as f64 / (1u64 << 30) as f64,
    );

    let mut draft = MtpBlock::new(
        &ctx, &stream, &file, &directory, &weights, &config, 1, rms_eps, rope_theta, true,
    )
    .expect("MTP draft head builds");

    let mut cache = KvCache::new(&stream, &config, 8).expect("draft KV cache");
    let positions = stream.alloc_zeros::<i32>(1).expect("positions");

    // A real token id (0 is always valid — token_embd's row 0) and a zero
    // seed `h`. This is a smoke test, not a numerical one: the point is that
    // every kernel in the chain accepts the shapes and produces something
    // finite, not that a zero `h` is a realistic target hidden state.
    let h = stream
        .alloc_zeros::<f32>(config.hidden_size as usize)
        .expect("h seed");
    let sampled = draft
        .forward(&stream, &[0i32], &h, &mut cache, 0, &positions)
        .expect("draft forward");
    let ids = sampled.expect("this instance was built with a LM head");
    assert_eq!(ids.len(), 1);
    let id = ids[0];
    println!("drafted token id: {id}");
    assert!(
        (0..config.vocab_size as i32).contains(&id),
        "drafted id {id} outside vocabulary [0, {})",
        config.vocab_size,
    );

    let h_nextn = stream.clone_dtoh(draft.h_nextn()).expect("h_nextn back");
    stream.synchronize().expect("sync");
    assert_eq!(h_nextn.len(), config.hidden_size as usize);
    assert!(
        h_nextn.iter().all(|v| v.is_finite()),
        "h_nextn has a non-finite element",
    );
    let norm: f32 = h_nextn.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("h_nextn: finite, ||h_nextn||_2 = {norm}");
    assert!(norm > 0.0, "h_nextn is exactly zero — the pass did nothing");
}
