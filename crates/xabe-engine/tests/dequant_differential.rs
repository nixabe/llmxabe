//! Differential test: device dequantization against the scalar reference,
//! on bytes taken from the real model file.
//!
//! This is the milestone-04 gate for the unpacking half of the MoE work. It
//! is deliberately stricter than the tolerance-based harness in
//! `xabe_kernels::compare`: both formats compute their result with
//! multiplications only, so there is no FMA contraction to diverge on, and
//! with matching operand order the device result must be **bit-identical** to
//! the CPU reference.
//!
//! That strictness is the point. A tolerance of 1e-5 would pass an
//! implementation that read the per-group scales unsigned for the handful of
//! blocks where they happen to be positive, or that reassociated the scale
//! multiply. Exact equality passes only a correct transcription.
//!
//! Synthetic inputs would not do here. The scale distributions, the sign
//! patterns in `scales`, and the occurrence of denormal deltas are properties
//! of the actual quantizer that produced this file, and a hand-built block
//! exercises none of them.
//!
//! SKIPS — reporting that it skipped — without a driver, a supported device,
//! or the model file.

use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_cuda::kernels::dequant::{BLOCK_Q6_K_BYTES, BLOCK_Q8_0_BYTES, Dequantizer};
use xabe_gguf::{GgmlType, GgufFile};
use xabe_kernels::quant::{dequantize_row_q6_k, dequantize_row_q8_0};
use xabe_model::config::ModelConfig;
use xabe_model::weights::{Role, WeightSchema};

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llama.cpp/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

/// How many blocks of each sampled tensor to compare.
///
/// Blocks are independent, so a prefix is representative of the unpacking.
/// 8,192 superblocks is 2,097,152 elements — enough that a rare bit pattern
/// shows up, small enough that the fp32 read-back stays under 10 MB.
const SAMPLE_BLOCKS: usize = 8_192;

fn model_path() -> PathBuf {
    std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
}

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
    let info = DeviceInfo::from_context(0, &ctx).expect("device properties readable");
    if !info.is_supported() {
        println!("SKIPPED: device 0 is below the sm_75 minimum");
        return None;
    }
    let path = model_path();
    if !path.exists() {
        println!("SKIPPED: model file not found at {}", path.display());
        return None;
    }
    Some((ctx, GgufFile::open(&path).expect("valid GGUF v3")))
}

/// Report the first disagreement, with enough context to identify the block.
fn assert_bit_identical(name: &str, cpu: &[f32], gpu: &[f32], block_elems: usize) {
    assert_eq!(
        cpu.len(),
        gpu.len(),
        "{name}: reference produced {} elements, device produced {}",
        cpu.len(),
        gpu.len(),
    );
    for (i, (&a, &b)) in cpu.iter().zip(gpu).enumerate() {
        if a.to_bits() != b.to_bits() {
            panic!(
                "{name}: element {i} (block {}, lane {}) differs — \
                 reference {a:e} ({:#010x}), device {b:e} ({:#010x})",
                i / block_elems,
                i % block_elems,
                a.to_bits(),
                b.to_bits(),
            );
        }
    }
}

#[test]
fn device_dequantization_is_bit_identical_to_the_reference_on_real_weights() {
    let Some((ctx, file)) = setup() else { return };
    let config = ModelConfig::qwen3_6_35b_a3b();
    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).expect("schema must resolve");
    let stream = ctx.default_stream();
    let dequant = Dequantizer::new(&ctx).expect("kernels must compile for sm_75");

    // Sample across layer kinds and roles so the comparison covers tensors
    // the quantizer treated differently: expert stacks, the LM head, the
    // mixer projections of both layer types.
    let sampled: &[(Role, Option<u32>)] = &[
        (Role::LmHead, None),
        (Role::TokenEmbedding, None),
        (Role::MoeGateExps, Some(0)),
        (Role::MoeUpExps, Some(0)),
        (Role::MoeGateExps, Some(19)),
        (Role::MoeDownExps, Some(19)),
        (Role::GdnQkv, Some(0)),
        (Role::GdnOut, Some(38)),
        (Role::AttnQGate, Some(3)),
        (Role::AttnOut, Some(39)),
        (Role::MoeSharedGate, Some(7)),
    ];

    let mut compared_tensors = 0usize;
    let mut compared_elements = 0u64;
    let mut seen_q6k = 0usize;
    let mut seen_q8_0 = 0usize;

    for &(role, layer) in sampled {
        let entry = directory
            .find(role, layer)
            .unwrap_or_else(|| panic!("{role} on layer {layer:?} is not in the directory"));
        let name = entry.spec.name.as_str();
        let all_bytes = file.tensor_bytes(name).expect("tensor data readable");

        let (block_bytes, block_elems) = match entry.info.ggml_type {
            GgmlType::Q6K => (BLOCK_Q6_K_BYTES, 256),
            GgmlType::Q8_0 => (BLOCK_Q8_0_BYTES, 32),
            // f32 and bf16 tensors need no unpacking; they are covered by
            // the byte-identical residency check in `device_weights.rs`.
            other => {
                println!(
                    "  {name}: {} needs no dequantization, skipped",
                    other.name()
                );
                continue;
            }
        };

        let take = (SAMPLE_BLOCKS * block_bytes).min(all_bytes.len());
        let take = take - take % block_bytes;
        let bytes = &all_bytes[..take];

        let device_src = stream.clone_htod(bytes).expect("upload sample");
        let (gpu, cpu) = match entry.info.ggml_type {
            GgmlType::Q6K => {
                seen_q6k += 1;
                (
                    dequant.q6_k(&stream, &device_src).expect("q6_K launch"),
                    dequantize_row_q6_k(bytes).expect("reference q6_K"),
                )
            }
            _ => {
                seen_q8_0 += 1;
                (
                    dequant.q8_0(&stream, &device_src).expect("q8_0 launch"),
                    dequantize_row_q8_0(bytes).expect("reference q8_0"),
                )
            }
        };
        let gpu = stream.clone_dtoh(&gpu).expect("read back");
        stream.synchronize().expect("sync");

        assert_bit_identical(name, &cpu, &gpu, block_elems);

        // A tensor of all zeros would compare bit-identical while proving
        // nothing, so require the sample to carry real signal.
        let nonzero = cpu.iter().filter(|v| **v != 0.0).count();
        assert!(
            nonzero * 4 > cpu.len(),
            "{name}: only {nonzero} of {} sampled values are non-zero",
            cpu.len(),
        );
        assert!(
            cpu.iter().all(|v| v.is_finite()),
            "{name}: reference produced a non-finite value",
        );

        compared_tensors += 1;
        compared_elements += cpu.len() as u64;
        println!(
            "  {name:<38} {:<5} {:>9} elements  bit-identical",
            entry.info.ggml_type.name(),
            cpu.len(),
        );
    }

    println!(
        "{compared_tensors} tensors, {compared_elements} elements, all bit-identical \
         ({seen_q6k} q6_K, {seen_q8_0} q8_0)",
    );
    // Both formats must actually have been exercised — a sample list that
    // drifted to one type would still pass every assertion above.
    assert!(seen_q6k >= 3, "only {seen_q6k} q6_K tensors compared");
    assert!(seen_q8_0 >= 3, "only {seen_q8_0} q8_0 tensors compared");
}

#[test]
fn a_ragged_input_is_rejected_rather_than_read_past() {
    let Some((ctx, _file)) = setup() else { return };
    let stream = ctx.default_stream();
    let dequant = Dequantizer::new(&ctx).expect("kernels must compile");

    // One byte short of a whole superblock. Truncating instead of rejecting
    // would dequantize whatever follows the tensor in device memory.
    let ragged = stream
        .clone_htod(&vec![0u8; BLOCK_Q6_K_BYTES + 1])
        .expect("upload");
    assert!(dequant.q6_k(&stream, &ragged).is_err());

    let ragged = stream
        .clone_htod(&vec![0u8; BLOCK_Q8_0_BYTES - 1])
        .expect("upload");
    assert!(dequant.q8_0(&stream, &ragged).is_err());
    println!("ragged inputs rejected for both formats");
}
