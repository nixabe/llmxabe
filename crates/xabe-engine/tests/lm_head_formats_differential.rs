//! Differential tests for the LM head GEMV's **fallback** weight formats,
//! against the `xabe-kernels` scalar reference.
//!
//! ## What these are and why they are not the Q8_0 test
//!
//! `lm_head_differential.rs` covers Q8_0, the format the shipped
//! `qwen35moe` file stores its head and every projection in, and it covers it
//! over the whole 248,320-entry vocabulary because that head's argmax *is*
//! the sampled token. This file covers the other bodies: the formats an
//! ordinary community quant uses, which no file on this host stores.
//!
//! That last clause is why these tests build their own weights instead of
//! reading them:
//!
//! 1. Take real rows of the real Q8_0 `output.weight` and dequantize them, so
//!    the values carry the trained head's actual magnitude spread and sign
//!    pattern rather than a uniform random one.
//! 2. Requantize each block to the format under test and serialize it.
//! 3. Dequantize *that* back with the scalar reference. Those floats, not the
//!    step-1 floats, are what the kernel is claimed to compute with.
//! 4. Run the scalar `gemv_batch` over them for the reference logits.
//!
//! Step 3 is load-bearing. Comparing against step 1 would fold the format's
//! own quantization error into the gate and measure the *format*; comparing
//! against step 3 measures exactly one thing — whether the device unpacks the
//! bits the way `dequantize_row_*` does.
//!
//! The scalar references themselves were checked against gguf-py's
//! independent numpy `dequantize` and agree bit-for-bit; see the commit that
//! added them. So a failure here is the device kernel, not the reference.
//!
//! ## What is compared, and how strictly
//!
//! | Property | Gate |
//! |---|---|
//! | logits, one token, every row | tolerance |
//! | logits, a 5-token batch | tolerance |
//! | argmax, every token | **exact** |
//! | every batch tile 1..=8 against the single-token path | **bit-identical** |
//!
//! Nothing here can be bit-identical to the host: the reference sums the
//! contraction sequentially while the kernel sums a few elements per lane and
//! then combines 32 lanes in a shuffle tree. The tile-versus-single-token
//! comparison *is* exact, because all eight instantiations do identical
//! arithmetic in identical order and differ only in how many accumulators are
//! in flight.
//!
//! ## Why a 4,096-row head and not the full 248,320
//!
//! The Q8_0 sibling argues for the full vocabulary because a sampled
//! comparison cannot see a wrong row. That argument is about the shipped
//! model's own head. These bodies are fallbacks for files that are not the
//! benchmark target, and the property they have to prove — that the
//! bit-unpacking matches the reference — is a property of a block, exercised
//! identically by row 5 and by row 200,000. 4,096 rows is 8.4 M weights,
//! which covers every code and both scale-group selectors many times over,
//! and keeps the host reference to seconds rather than minutes.
//!
//! SKIPS — reporting that it skipped — without a driver, a supported device,
//! or the model file.

use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_cuda::kernels::lm_head::{
    HeadFormat, HeadTensor, LmHeadGeometry, LmHeadKernels, MAX_BATCH_TILE,
};
use xabe_gguf::{GgmlType, GgufFile};
use xabe_kernels::compare::{Tolerance, assert_matches, compare};
use xabe_kernels::gemv::{argmax, gemv_batch};
use xabe_kernels::quant::{
    QK_K, QK4_0, dequantize_q4_0, dequantize_q4_k, dequantize_q5_k, dequantize_q6_k,
    dequantize_row_f16, dequantize_row_q8_0, quantize_q4_0, quantize_q4_k, quantize_q5_k,
    quantize_q6_k,
};
use xabe_kernels::rng::Xorshift64Star;
use xabe_model::config::ModelConfig;
use xabe_model::weights::{Role, WeightSchema};

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llmxabe/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

/// Rows of the real head requantized into the format under test.
const ROWS: usize = 4_096;

/// Residual width, and so the contraction. Stays a multiple of 512:
/// `LmHeadKernels::with_row_tile` requires it for the Q8_0 staging pass, and
/// the k-quant bodies need a multiple of 256 for their superblock loop.
const HIDDEN: usize = 2_048;

/// Tokens in the batched test. Deliberately not a power of two, so the `b5`
/// instantiation is reached.
const BATCH: usize = 5;

/// The same gate `lm_head_differential.rs` applies to the Q8_0 body, and for
/// the same reason: both sides multiply bit-identical weights by bit-identical
/// activations, so summation order is the entire difference and the bar can be
/// set at fp32 reassociation noise over a 2,048-term contraction.
/// `max_rel_error` is 1.0 because a logit near zero makes relative error
/// meaningless; `max_abs_error` and cosine are the real gates.
const GATE: Tolerance = Tolerance {
    max_abs_error: 5e-5,
    max_rel_error: 1.0,
    min_cosine_similarity: 1.0 - 1e-9,
    allow_non_finite: false,
};

fn model_path() -> PathBuf {
    std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
}

fn device() -> Option<(Arc<CudaContext>, DeviceInfo)> {
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
    Some((ctx, info))
}

fn geometry() -> LmHeadGeometry {
    LmHeadGeometry {
        hidden: HIDDEN,
        vocab: ROWS,
        max_tokens: MAX_BATCH_TILE,
    }
}

/// The first [`ROWS`] rows of the real `output.weight`, dequantized.
///
/// The common source for every format below, so they are all compared on the
/// same trained values rather than on separate random draws.
fn real_head_rows(file: &GgufFile, g: &LmHeadGeometry) -> Vec<f32> {
    let config = ModelConfig::qwen3_6_35b_a3b();
    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(file).expect("schema must resolve");
    let entry = directory
        .find(Role::LmHead, None)
        .expect("output.weight present — the LM head is untied");
    assert_eq!(
        entry.info.ggml_type,
        GgmlType::Q8_0,
        "this file requantizes the real Q8_0 head; if the file's head is not \
         Q8_0 the source data is not what the module docs claim",
    );
    let bytes = file
        .tensor_bytes(&entry.spec.name)
        .expect("tensor readable through the mmap");
    let row_bytes = g.hidden / 32 * 34;
    let src = dequantize_row_q8_0(&bytes[..ROWS * row_bytes]).expect("reference q8_0 unpacking");
    assert_eq!(src.len(), ROWS * g.hidden);
    src
}

/// Pack `src` into `format` and dequantize it back with the scalar reference.
///
/// Returns the serialized bytes the device will read and the fp32 the device
/// is claimed to reconstruct from them.
fn pack(format: HeadFormat, src: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let mut bytes = Vec::new();
    let mut back = Vec::with_capacity(src.len());
    match format {
        HeadFormat::Q6K => {
            for x in src.as_chunks::<QK_K>().0 {
                let b = quantize_q6_k(x);
                bytes.extend_from_slice(&b.to_bytes());
                back.extend_from_slice(&dequantize_q6_k(&b));
            }
        }
        HeadFormat::F16 => {
            // Dense, so "packing" is just the narrowing conversion — and the
            // round trip through `dequantize_row_f16` is what the kernel is
            // claimed to reconstruct. Values here are order 0.25, far inside
            // f16's range, so nothing saturates; a head whose weights reached
            // 65504 would be a different test.
            for &v in src {
                bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
            back = dequantize_row_f16(&bytes);
        }
        HeadFormat::Q4_0 => {
            for x in src.as_chunks::<QK4_0>().0 {
                let b = quantize_q4_0(x);
                bytes.extend_from_slice(&b.to_bytes());
                back.extend_from_slice(&dequantize_q4_0(&b));
            }
        }
        HeadFormat::Q4K => {
            for x in src.as_chunks::<QK_K>().0 {
                let b = quantize_q4_k(x);
                bytes.extend_from_slice(&b.to_bytes());
                back.extend_from_slice(&dequantize_q4_k(&b));
            }
        }
        HeadFormat::Q5K => {
            for x in src.as_chunks::<QK_K>().0 {
                let b = quantize_q5_k(x);
                bytes.extend_from_slice(&b.to_bytes());
                back.extend_from_slice(&dequantize_q5_k(&b));
            }
        }
        other => panic!("no packer for {other:?}"),
    }
    (bytes, back)
}

/// Guard against a tensor that would compare perfectly while proving nothing.
fn assert_carries_signal(label: &str, w: &[f32]) {
    let amax = w.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let nonzero = w.iter().filter(|v| **v != 0.0).count();
    println!(
        "{label}: |w| max {amax:.4e}, {:.1}% non-zero",
        100.0 * nonzero as f64 / w.len() as f64,
    );
    assert!(amax > 0.0, "{label}: the requantized head is all zeros");
    assert!(
        nonzero > w.len() / 2,
        "{label}: over half the requantized head is exactly zero, which no \
         trained head is; the quantizer or the serializer is wrong",
    );
    assert!(
        w.iter().all(|v| v.is_finite()),
        "{label}: contains a non-finite weight",
    );
}

/// Random hidden states, plus the same rows padded to the buffer capacity.
fn hidden_states(g: &LmHeadGeometry, seed: u64, tokens: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let live: Vec<Vec<f32>> = (0..tokens)
        .map(|_| rng.vec_f32(g.hidden, -1.0, 1.0))
        .collect();
    let mut flat: Vec<f32> = live.concat();
    flat.resize(g.max_tokens * g.hidden, 0.0);
    (live, flat)
}

fn assert_argmax_agrees(label: &str, candidate: &[f32], reference: &[f32]) {
    let ci = argmax(candidate);
    let ri = argmax(reference);
    assert_eq!(
        ci, ri,
        "{label}: device argmax is row {ci} ({}), reference is row {ri} ({}); \
         a head that agrees within tolerance and still flips the argmax has \
         produced a different model",
        candidate[ci], reference[ri],
    );
}

/// The whole body of every per-format test: reference agreement on one token
/// and on a batch, exact argmax, and every tile bit-identical to the
/// single-token path.
fn check(format: HeadFormat, label: &str, seed: u64) {
    let Some((ctx, info)) = device() else { return };
    let path = model_path();
    if !path.exists() {
        println!("SKIPPED: model file not found at {}", path.display());
        return;
    }
    let file = GgufFile::open(&path).expect("valid GGUF v3");
    println!("device: {}, format: {label}", info.name);

    let g = geometry();
    let src = real_head_rows(&file, &g);
    let (packed, weights) = pack(format, &src);
    assert_eq!(weights.len(), g.vocab * g.hidden);
    assert_carries_signal(label, &weights);

    let stream = ctx.default_stream();
    let d_weight = stream.clone_htod(packed.as_slice()).expect("upload head");
    stream.synchronize().expect("sync");
    let weight = HeadTensor {
        bytes: &d_weight,
        format,
    };
    let kernels = LmHeadKernels::new(&ctx, g).expect("compile the LM head kernels");

    let (xs, flat) = hidden_states(&g, seed, MAX_BATCH_TILE);
    let reference = gemv_batch(&weights, g.vocab, g.hidden, &xs);
    let d_x = stream
        .clone_htod(flat.as_slice())
        .expect("upload activations");

    // One token, then a batch, both against the scalar reference.
    for tokens in [1, BATCH] {
        let mut d_logits = stream
            .alloc_zeros::<f32>(g.max_tokens * g.vocab)
            .expect("allocate logits");
        kernels
            .forward(&stream, weight, &d_x, tokens, &mut d_logits)
            .expect("launch the head");
        let got = stream.clone_dtoh(&d_logits).expect("copy logits back");
        for t in 0..tokens {
            let span = &got[t * g.vocab..(t + 1) * g.vocab];
            let l = format!("{label}, {tokens} token(s), token {t}");
            println!("{l}: {}", compare(span, &reference[t]));
            assert_matches(span, &reference[t], &GATE);
            assert_argmax_agrees(&l, span, &reference[t]);
        }
    }

    // Every tile against the one-token path, bit for bit. A tolerance cannot
    // catch one instantiation drifting from the others, because all eight are
    // within tolerance of the reference by construction.
    let mut single = vec![0.0f32; MAX_BATCH_TILE * g.vocab];
    for t in 0..MAX_BATCH_TILE {
        let mut one = vec![0.0f32; g.max_tokens * g.hidden];
        one[..g.hidden].copy_from_slice(&xs[t]);
        let d_one = stream.clone_htod(one.as_slice()).expect("upload");
        let mut d_out = stream
            .alloc_zeros::<f32>(g.max_tokens * g.vocab)
            .expect("allocate logits");
        kernels
            .forward(&stream, weight, &d_one, 1, &mut d_out)
            .expect("launch the head");
        let out = stream.clone_dtoh(&d_out).expect("copy logits back");
        single[t * g.vocab..(t + 1) * g.vocab].copy_from_slice(&out[..g.vocab]);
    }
    for tile in 1..=MAX_BATCH_TILE {
        let mut d_out = stream
            .alloc_zeros::<f32>(g.max_tokens * g.vocab)
            .expect("allocate logits");
        kernels
            .forward(&stream, weight, &d_x, tile, &mut d_out)
            .expect("launch the head");
        let out = stream.clone_dtoh(&d_out).expect("copy logits back");
        for t in 0..tile {
            assert_eq!(
                &out[t * g.vocab..(t + 1) * g.vocab],
                &single[t * g.vocab..(t + 1) * g.vocab],
                "{label} tile {tile}, token {t}: the batched entry point is \
                 not bit-identical to the single-token one, so the eight \
                 instantiations are not the same kernel",
            );
        }
    }
    println!("{label}: all {MAX_BATCH_TILE} batch tiles bit-identical to the single-token path");
}

#[test]
fn the_q6_k_head_body_matches_the_scalar_reference() {
    check(HeadFormat::Q6K, "q6_K", 0x51D6_C0DE);
}

#[test]
fn the_f16_head_body_matches_the_scalar_reference() {
    check(HeadFormat::F16, "f16", 0x00F1_6DEF);
}

#[test]
fn the_q4_0_head_body_matches_the_scalar_reference() {
    check(HeadFormat::Q4_0, "q4_0", 0x0000_4004);
}

#[test]
fn the_q4_k_head_body_matches_the_scalar_reference() {
    check(HeadFormat::Q4K, "q4_K", 0x0004_4B4B);
}

#[test]
fn the_q5_k_head_body_matches_the_scalar_reference() {
    check(HeadFormat::Q5K, "q5_K", 0x0005_4B5B);
}
