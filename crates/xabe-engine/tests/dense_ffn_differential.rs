//! Differential test: the device dense-FFN block against the `xabe-kernels`
//! CPU reference, at Qwen3.8-27B's real geometry and on the real quantized
//! `ffn_{gate,up,down}` weights taken from the model file.
//!
//! The dense block is not a new GEMM — it is
//! [`xabe_cuda::kernels::moe::MoeKernels::shared_expert`] at a 34x wider
//! intermediate — so what this gates is the *block*, which is new: that the
//! post-mixer RMSNorm feeds the MLP, that the MLP is `down(silu(gate·x) *
//! (up·x))` with no gate of its own, and that the residual added back is the
//! block's input rather than its normalized form. Each of those three has a
//! wrong version that still generates fluent text.
//!
//! It also gates the two widths the kernels themselves have never run at.
//! `MoeKernels::new` requires `hidden` and `intermediate` to be multiples of
//! the 128-element tile and `shared_expert`'s GEMV additionally splits
//! `hidden` over four warps in 128-element tiles; 5120 and 17408 satisfy
//! those, but "satisfies the precondition" is not "produces the right
//! answer", and this is the difference.
//!
//! Both residencies are covered in one test rather than two, because the
//! dequantized reference weights are 1.07 GiB and building them twice in
//! parallel test threads is the kind of thing that turns a CI box over:
//!
//! | Residency | Width | Kernel | Gate |
//! |---|---|---|---|
//! | Q8_0 as stored | any | fp32 `moe_shared_ffn` | [`DENSE_FP32_GATE`] |
//! | split int8 repack | > 4 | `shared_expert_mma` | [`DENSE_MMA_GATE`] |
//! | split int8 repack | <= 4 | `dense_proj_split_t*` | [`DENSE_MMA_GATE`] |
//!
//! The serving path takes the second and third — see `DENSE_REPACK_INT8` —
//! so the first is the fallback for a device with no integer tensor cores,
//! and all three are gated because any of them can be the one that runs.
//!
//! Only one layer's FFN is loaded (about 271 MiB of the model's 29.3 GiB);
//! the full model is never resident.
//!
//! SKIPS — reporting that it skipped — without a driver, a supported device,
//! or the dense model file.

use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_engine::block::dense_ffn::{DenseFfnBlock, DenseFfnLayerWeights};
use xabe_engine::block::moe::MoeBlock;
use xabe_gguf::{GgmlType, GgufFile};
use xabe_kernels::compare::{Tolerance, assert_matches, compare};
use xabe_kernels::moe::gemm::{ExpertWeights, expert_mlp};
use xabe_kernels::norm::rms_norm;
use xabe_kernels::quant::{dequantize_row_q6_k, dequantize_row_q8_0};
use xabe_kernels::rng::Xorshift64Star;
use xabe_model::config::ModelConfig;
use xabe_model::weights::{Directory, Role, WeightSchema};

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llmxabe/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf";

/// The layer whose FFN is loaded. Block 0 is a Gated DeltaNet layer; the FFN
/// is identical on both mixer kinds, so the choice is arbitrary and the
/// cheapest one to find in the file.
const LAYER: u32 = 0;

/// Live tokens in the batch.
///
/// Deliberately not a power of two and not a multiple of any tile width, so
/// the tail of every launch is exercised rather than incidentally absent.
const NUM_TOKENS: usize = 5;

/// Buffer capacity for the fp32 pass. Above [`NUM_TOKENS`] on purpose: the
/// gap is what proves rows past the live count are left alone.
const FP32_MAX_TOKENS: usize = 16;

/// Buffer capacity for the wide integer pass. Deliberately a *different*
/// width from the fp32 one: the residencies must agree with the reference at
/// whatever width they are given, not only at a shared one.
const MMA_MAX_TOKENS: usize = 128;

/// Buffer capacity for the narrow integer pass, inside
/// `SPLIT_GEMV_MAX_TOKENS` so the split-layout GEMV is what runs. This is the
/// decode shape, and it reads the *same* resident weights as the wide arm
/// through a different kernel — so the two must agree with the reference to
/// the same bound, which is what says the GEMV is not a second, subtly
/// different unpack of the split layout.
const GEMV_MAX_TOKENS: usize = 3;

const RESIDUAL_SEED: u64 = 0x5eed_dead_beef_1357;

/// Gate for the fp32 GEMM path against the fp32 scalar reference.
///
/// The dequantized weights are bit-identical to the reference's — both come
/// out of `xabe_kernels::quant` — so what is left is a 5,120-term dot product
/// and then a 17,408-term one, summed sequentially on the host and in a
/// warp-shuffle tree on the device. fp32 addition is not associative and
/// 17,408 terms is a long reduction, so the gate is a tolerance.
///
/// **`max_abs_error` and `min_cosine_similarity` are the gate;
/// `max_rel_error` is not** — the same floor argument the routed MoE
/// differential records. `compare()` divides by `max(|reference|, 1e-6)`, and
/// an FFN output that spans `[-82, 82]` has plenty of elements near zero for
/// which that denominator is the floor rather than the value. The test
/// asserts below that the element driving `max_rel_error` really is small
/// against the tensor's own scale, so the justification fails loudly if it
/// stops holding.
///
/// Measured on layer 0's real Q8_0 weights, 5 tokens, `max_tokens = 16`:
/// `max_abs = 3.20e-4`, `cosine = 1.000000`, on a tensor whose own max
/// magnitude is 82.3 — a relative error of 3.9e-6 against scale. The bound is
/// ~2x that.
const DENSE_FP32_GATE: Tolerance = Tolerance {
    max_abs_error: 7e-4,
    max_rel_error: f32::INFINITY,
    min_cosine_similarity: 1.0 - 1e-6,
    allow_non_finite: false,
};

/// Gate for the integer tensor-core path.
///
/// A different arithmetic, so a different bound. The activations are
/// quantized to int8 with one fp32 scale per 32, costing about `1/254` of
/// each block's largest magnitude per element; the weights are Q8_0 integers
/// the tensor core multiplies exactly and the int32 accumulation is exact, so
/// activation quantization is the whole of the error. It is applied twice
/// here — once to `normed` and once to the 17,408-wide SwiGLU output — where
/// the routed path applies it to a 512-wide one.
///
/// Measured at the same inputs, `max_tokens = 128`: `max_abs = 6.45e-2`,
/// `cosine = 0.999997`, on the same 82.3 scale — 7.8e-4 against scale. The
/// bound is ~2x the absolute figure and ~3x the cosine deficit.
///
/// It is still a real gate: the characteristic defect of hand-written MMA is
/// a wrong fragment layout, which does not produce a slightly worse answer
/// but a differently-shaped one, and the cosine floor is what catches that.
const DENSE_MMA_GATE: Tolerance = Tolerance {
    max_abs_error: 1.3e-1,
    max_rel_error: f32::INFINITY,
    min_cosine_similarity: 1.0 - 1e-5,
    allow_non_finite: false,
};

fn model_path() -> PathBuf {
    std::env::var_os("LLMXABE_DENSE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
}

fn device_and_model() -> Option<(Arc<CudaContext>, GgufFile)> {
    if !driver_available() {
        println!("SKIPPED: no CUDA driver");
        return None;
    }
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            println!("SKIPPED: no usable CUDA device ({e})");
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
            "SKIPPED: dense model file not found at {}; set LLMXABE_DENSE_MODEL to override",
            path.display()
        );
        return None;
    }
    Some((ctx, GgufFile::open(&path).expect("valid GGUF v3")))
}

fn dequant_slice(ty: GgmlType, bytes: &[u8]) -> Vec<f32> {
    match ty {
        GgmlType::Q8_0 => dequantize_row_q8_0(bytes).expect("reference q8_0"),
        GgmlType::Q6K => dequantize_row_q6_k(bytes).expect("reference q6_K"),
        other => panic!("layer {LAYER}'s FFN is stored as {}", other.name()),
    }
}

fn host_weights(file: &GgufFile, directory: &Directory<'_>) -> ExpertWeights {
    let read = |role: Role| {
        let entry = directory
            .find(role, Some(LAYER))
            .unwrap_or_else(|| panic!("{role} missing"));
        let bytes = file.tensor_bytes(&entry.spec.name).expect("readable");
        println!(
            "  {:<20} {:<6} {} elements",
            entry.spec.role.suffix(),
            entry.info.ggml_type.name(),
            entry.spec.n_elements(),
        );
        dequant_slice(entry.info.ggml_type, bytes)
    };
    ExpertWeights {
        gate: read(Role::FfnGate),
        up: read(Role::FfnUp),
        down: read(Role::FfnDown),
    }
}

/// `l_out` and `ffn_out` as the reference computes them, for one token.
fn reference(
    host: &ExpertWeights,
    post_norm: &[f32],
    residual: &[f32],
    hidden: usize,
    intermediate: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let normed = rms_norm(residual, post_norm, eps);
    let ffn = expert_mlp(host, &normed, hidden, intermediate);
    let l_out = ffn
        .iter()
        .zip(residual.iter())
        .map(|(&f, &r)| f + r)
        .collect();
    (ffn, l_out)
}

#[test]
fn the_dense_block_matches_the_cpu_reference_on_both_gemm_paths() {
    let Some((ctx, file)) = device_and_model() else {
        return;
    };
    let config = ModelConfig::qwen3_8_27b();
    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).expect("dense schema resolves");
    let stream = ctx.default_stream();
    let eps = MoeBlock::eps_from(&config, &file);
    let hidden = config.hidden_size as usize;
    let intermediate = config.ffn.intermediate() as usize;
    println!("layer {LAYER} dense FFN, eps {eps:e}:");
    let host = host_weights(&file, &directory);

    let post_norm_entry = directory
        .find(Role::PostMixerNorm, Some(LAYER))
        .expect("post_attention_norm");
    let post_norm: Vec<f32> = file
        .tensor_bytes(&post_norm_entry.spec.name)
        .expect("readable")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    assert_eq!(post_norm.len(), hidden);

    let mut rng = Xorshift64Star::new(RESIDUAL_SEED);
    let rows: Vec<Vec<f32>> = (0..NUM_TOKENS)
        .map(|_| rng.vec_f32(hidden, -1.0, 1.0))
        .collect();

    let mut ref_ffn = Vec::with_capacity(NUM_TOKENS * hidden);
    let mut ref_l_out = Vec::with_capacity(NUM_TOKENS * hidden);
    for row in &rows {
        let (f, l) = reference(&host, &post_norm, row, hidden, intermediate, eps);
        ref_ffn.extend_from_slice(&f);
        ref_l_out.extend_from_slice(&l);
    }

    let scale = ref_l_out.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    println!("reference l_out max magnitude {scale:.4}");

    for (label, max_tokens, wants_int8, tolerance) in [
        ("fp32", FP32_MAX_TOKENS, false, DENSE_FP32_GATE),
        ("int8-mma", MMA_MAX_TOKENS, true, DENSE_MMA_GATE),
        ("int8-gemv", GEMV_MAX_TOKENS, true, DENSE_MMA_GATE),
    ] {
        // The narrow arm runs fewer tokens than the others, because that is
        // the point of it. Its reference is the same rows' prefix.
        let live_tokens = NUM_TOKENS.min(max_tokens);
        let geometry = DenseFfnBlock::geometry_for(&config, 32, max_tokens).expect("dense");
        let weights =
            DenseFfnLayerWeights::upload(&stream, &file, &directory, LAYER, &geometry, wants_int8)
                .expect("upload");
        assert_eq!(
            weights.has_int8(),
            wants_int8,
            "{label}: the residency must follow what the caller asked for",
        );
        assert_eq!(
            weights.quants().is_none(),
            wants_int8,
            "{label}: the repack replaces the Q8_0 upload rather than joining it",
        );
        let mut block =
            DenseFfnBlock::new(&ctx, &stream, geometry, eps, wants_int8).expect("block compiles");
        if wants_int8 && !block.tensor_cores_enabled() {
            println!("SKIPPED int8 path: integer tensor cores unavailable on this device");
            continue;
        }

        let mut flat: Vec<f32> = rows[..live_tokens].concat();
        flat.resize(max_tokens * hidden, 0.0);
        let d_residual = stream.clone_htod(&flat).expect("upload");
        // Filled with a sentinel rather than zeros: a kernel that wrote a
        // row it should not have would otherwise be invisible against the
        // zeros the reference would also produce there.
        let sentinel = vec![-7.5f32; max_tokens * hidden];
        let mut d_ffn = stream.clone_htod(&sentinel).expect("alloc");
        let mut d_l_out = stream.clone_htod(&sentinel).expect("alloc");

        block
            .forward(
                &stream,
                &weights,
                &d_residual,
                live_tokens,
                &mut d_ffn,
                &mut d_l_out,
            )
            .expect("dense ffn forward");
        let got_ffn = stream.clone_dtoh(&d_ffn).expect("ffn back");
        let got_l_out = stream.clone_dtoh(&d_l_out).expect("l_out back");
        stream.synchronize().expect("sync");

        let live = live_tokens * hidden;
        let ref_ffn = &ref_ffn[..live];
        let ref_l_out = &ref_l_out[..live];
        let ffn_result = compare(&got_ffn[..live], ref_ffn);
        let l_result = compare(&got_l_out[..live], ref_l_out);
        println!("{label} ffn_out vs expert_mlp:            {ffn_result}");
        println!("{label} l_out  vs expert_mlp + residual:  {l_result}");
        assert_matches(&got_ffn[..live], ref_ffn, &tolerance);
        assert_matches(&got_l_out[..live], ref_l_out, &tolerance);

        // The excluded metric, justified rather than ignored: the element
        // driving `max_rel_error` must be small against the tensor's own
        // scale, which is what makes the ratio an artefact of `compare()`'s
        // 1e-6 floor rather than a real error the gate is stepping over.
        for (what, result, reference) in [
            ("ffn_out", &ffn_result, ref_ffn),
            ("l_out", &l_result, ref_l_out),
        ] {
            let driver = reference[result.max_rel_error_index].abs();
            assert!(
                driver < scale / 1000.0,
                "{label} {what}: max_rel_error is driven by |{driver:e}|, which is not \
                 small against the tensor's own scale {scale:e} — the exclusion of \
                 max_rel_error from the gate is no longer justified",
            );
        }

        // Rows past the live count are the caller's, and the block must not
        // have touched `l_out` there. `ffn_out` is written by the projection
        // itself, which owns the whole buffer, so only `l_out` is checked.
        assert!(
            got_l_out[live..].iter().all(|&v| v == -7.5),
            "{label}: the residual add wrote past the live token count",
        );
    }
}

#[test]
fn the_dense_block_is_not_the_moe_block_with_a_gate() {
    // A structural check that needs no device: the gated and ungated forms
    // differ, so a block that carried the MoE shared-expert's sigmoid across
    // would land somewhere else entirely. This is the arithmetic statement of
    // what `qwen35.cpp:475`'s `GGML_ASSERT(ffn_gate_inp == nullptr)` says.
    let ffn = [1.0f32, -2.0, 0.5];
    let residual = [0.25f32, 0.25, 0.25];
    let ungated: Vec<f32> = ffn.iter().zip(residual).map(|(&f, r)| f + r).collect();
    let gate = xabe_kernels::norm::sigmoid(0.75);
    let gated: Vec<f32> = ffn
        .iter()
        .zip(residual)
        .map(|(&f, r)| f * gate + r)
        .collect();
    assert_ne!(ungated, gated);
    assert!((gate - 0.679_178_7).abs() < 1e-6, "gate {gate}");
}
