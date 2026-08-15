//! Load the real model onto a real GPU and verify it arrived intact.
//!
//! This is the milestone-01 gate for device residency. It is the first test in
//! the project that moves model weights across PCIe, and it asserts four
//! things that a build success does not:
//!
//! 1. Every tensor the schema names is resident, with no byte left unwritten.
//! 2. Read-back is bit-identical to the memory-mapped file, sampled across
//!    every layer kind and every element type.
//! 3. VRAM actually consumed is within tolerance of the documented budget.
//! 4. Every tensor the forward pass will ask for is addressable, and no two
//!    share an arena offset.
//!
//! ## Why this is one test and not four
//!
//! The model is 29.65 GiB on a 47.3 GiB card. `cargo test` runs test functions
//! on parallel threads by default, so four functions that each load the model
//! would ask one device for ~119 GiB and fail with `CUDA_ERROR_OUT_OF_MEMORY`.
//! Serialising them would work and would also triple the runtime for no gain:
//! these are four assertions about one load, so they share one.
//!
//! SKIPS — reporting that it skipped — when there is no CUDA driver, no
//! supported device, or no model file. Per `AGENTS.md`, a skip is not a pass.

use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_engine::weights::DeviceWeights;
use xabe_gguf::GgufFile;
use xabe_model::config::ModelConfig;
use xabe_model::weights::{Role, WeightSchema};

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llama.cpp/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

/// `docs/MODEL.md` budgets this much for weights.
const DOCUMENTED_WEIGHTS_GIB: f64 = 29.6;

/// Tolerance on that budget. Anything worse and the budget cannot be trusted
/// to decide whether the KV pool fits alongside the weights.
const VRAM_TOLERANCE: f64 = 0.05;

fn model_path() -> PathBuf {
    std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// Everything the test needs, or `None` with a printed reason for skipping.
fn setup() -> Option<(Arc<CudaContext>, GgufFile)> {
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
    let info = DeviceInfo::from_context(0, &ctx).expect("device properties must be readable");
    if !info.is_supported() {
        println!(
            "SKIPPED: device 0 is {} ({}), below the sm_75 minimum",
            info.name,
            info.compute_capability.sm_arch(),
        );
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
    println!(
        "device 0: {} ({}), {:.1} GiB total",
        info.name,
        info.compute_capability.sm_arch(),
        gib(info.total_memory),
    );
    Some((ctx, file))
}

#[test]
fn the_model_becomes_resident_on_a_gpu_intact_and_within_budget() {
    let Some((ctx, file)) = setup() else { return };
    let config = ModelConfig::qwen3_6_35b_a3b();
    let schema = WeightSchema::new(&config);
    let directory = schema
        .resolve(&file)
        .expect("schema must resolve against the model file");
    let stream = ctx.default_stream();

    let (weights, report) = match DeviceWeights::load(&ctx, &stream, &file, &directory) {
        Ok(pair) => pair,
        Err(e) => panic!("load failed: {e}"),
    };

    println!(
        "uploaded {} tensors, {:.2} GiB in {:.1} s ({:.2} GB/s)",
        report.tensors,
        gib(report.bytes),
        report.elapsed.as_secs_f64(),
        report.throughput_gb_s(),
    );
    println!(
        "arena {:.2} GiB, VRAM consumed {:.2} GiB (driver overhead {:.0} MiB)",
        gib(report.arena_bytes),
        gib(report.vram_consumed()),
        (report.vram_consumed().saturating_sub(report.arena_bytes)) as f64 / (1u64 << 20) as f64,
    );

    // --- 1. Nothing was skipped -----------------------------------------

    assert_eq!(
        report.tensors,
        directory.len(),
        "not every tensor in the directory was uploaded",
    );
    assert!(report.complete(), "load reported no bytes written");

    // Alignment padding is the only legitimate gap between what the file
    // holds and what the arena reserves.
    let padding = report.arena_bytes - report.bytes;
    let max_padding = directory.len() as u64 * 256;
    assert!(
        padding < max_padding,
        "arena is {padding} B larger than the tensor data, more than alignment explains",
    );

    // --- 2. Bit-identical read-back --------------------------------------

    // 48 samples is a stride of ~15 across the directory, which covers every
    // layer kind, both mixer types, and all four element types in the file.
    let (checked, verified_bytes) = weights
        .verify(&stream, &file, &directory, 48)
        .expect("read-back must match the file");
    println!(
        "verified {checked} tensors ({:.2} GiB) bit-identical against the mapping",
        gib(verified_bytes),
    );
    assert!(checked >= 40, "verification sampled only {checked} tensors");

    // --- 3. VRAM against the documented budget ---------------------------

    let consumed_gib = gib(report.vram_consumed());
    let error = (consumed_gib - DOCUMENTED_WEIGHTS_GIB) / DOCUMENTED_WEIGHTS_GIB;
    println!(
        "budget {DOCUMENTED_WEIGHTS_GIB:.2} GiB, measured {consumed_gib:.2} GiB ({:+.2}%)",
        error * 100.0,
    );
    assert!(
        error.abs() < VRAM_TOLERANCE,
        "resident weights are {:.1}% off the documented budget",
        error * 100.0,
    );

    // --- 4. Everything the forward pass needs is addressable -------------

    for role in [Role::TokenEmbedding, Role::OutputNorm, Role::LmHead] {
        assert!(weights.find(role, None).is_some(), "{role} not resident");
    }

    let mut gdn = 0;
    let mut attn = 0;
    for layer in 0..config.num_layers {
        // Every layer, without exception, has a full MoE block.
        for role in [Role::MoeRouter, Role::MoeGateExps, Role::MoeSharedDown] {
            assert!(
                weights.find(role, Some(layer)).is_some(),
                "layer {layer} is missing {role}",
            );
        }
        if weights.find(Role::GdnQkv, Some(layer)).is_some() {
            gdn += 1;
            // The short convolution is easy to forget — it has no entry in
            // `docs/KERNELS.md` — so assert it explicitly.
            assert!(
                weights.find(Role::GdnConv1d, Some(layer)).is_some(),
                "GDN layer {layer} has no short convolution",
            );
        } else {
            attn += 1;
            assert!(
                weights.find(Role::AttnQGate, Some(layer)).is_some(),
                "attention layer {layer} has no query projection",
            );
        }
    }
    assert_eq!(
        (gdn, attn),
        (30, 10),
        "layer pattern is wrong on the device"
    );

    // Two tensors sharing a byte range would each read back correctly on
    // their own and corrupt each other in use, so distinctness is checked
    // rather than assumed from the bump allocator being obviously correct.
    let mut offsets: Vec<usize> = weights
        .placements()
        .iter()
        .map(|p| p.alloc.offset)
        .collect();
    offsets.sort_unstable();
    let before = offsets.len();
    offsets.dedup();
    assert_eq!(offsets.len(), before, "two tensors share an arena offset");

    println!(
        "{} tensors addressable across {gdn} GDN and {attn} attention layers",
        weights.placements().len()
    );
}

#[test]
fn a_card_too_small_for_the_model_is_refused_before_any_copying() {
    // The load path checks free memory up front so an undersized card reports
    // a number instead of failing partway through a 30 GiB copy. This checks
    // the arithmetic without needing an undersized card: `required_bytes` is
    // what the check compares against.
    let Some((_ctx, file)) = setup() else { return };
    let config = ModelConfig::qwen3_6_35b_a3b();
    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).expect("schema must resolve");

    let needed = DeviceWeights::required_bytes(&directory);
    assert!(
        needed >= directory.total_bytes(),
        "arena sizing must not under-count the tensor data",
    );
    assert!(
        needed - directory.total_bytes() < directory.len() as u64 * 256,
        "arena sizing overshoots by more than alignment",
    );
    println!(
        "required {:.2} GiB for {:.2} GiB of tensor data",
        gib(needed),
        gib(directory.total_bytes()),
    );
}
