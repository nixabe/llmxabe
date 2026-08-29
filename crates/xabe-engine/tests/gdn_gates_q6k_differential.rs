//! `gdn_alpha_beta_gates_q6k{,_t1}` against the f32 gate kernel, on weights
//! the two are guaranteed to agree on.
//!
//! The α/β gates were the last consumer in the engine with no reader beyond
//! f32 and Q8_0, so a uniform Q6_K file could not load even once every other
//! tensor had one. The new kernel unpacks Q6_K in its inner loop. Nothing
//! shipped stores these two tensors that way — Qwen3.6 has them f32, Qwen3.8
//! Q8_0 — so there is no golden to compare against and no file to read. This
//! test makes one.
//!
//! The method is `lm_head_formats_differential.rs`'s: take the real weights,
//! requantize them on the host, and feed *the same numbers* to two kernels.
//! Concretely, with `w` the file's f32 `ssm_alpha` / `ssm_beta`:
//!
//!   packed    = quantize_q6_k(w)          -> the Q6_K kernel's input
//!   reference = dequantize_q6_k(packed)   -> the f32 kernel's input
//!
//! `reference` is exactly the set of values `packed` denotes, so the two
//! kernels are being asked to compute the same dot products from the same
//! floats. Quantization error cancels: it is in both sides. What is left is
//! the unpacking — the nibble/high-bit interleave, the sub-scale index, and
//! the operand order — which is the only thing here that has never run.
//!
//! **The bar is exact equality, not a tolerance.** The Q6_K body was written
//! to walk a superblock as (half, group) pairs so that lane `l` accumulates
//! flat offsets `l, l+32, l+64, ...` ascending — the same sequence in the
//! same order as the f32 body. Same addends, same order, same result, to the
//! last bit. A tolerance here would hide precisely the reordering bugs the
//! test exists to catch, so any difference at all is a failure.
//!
//! Both token tiles are covered: `tokens == 1` takes the `_t1` instantiation
//! and `tokens == 8` takes the `GATE_TT` one, and they are separate compiled
//! entry points that share only the macro body.
//!
//! SKIPS — reporting that it skipped — without a driver, a supported device,
//! or the model file, exactly as `gdn_proj_differential.rs` does.

use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_engine::block::gdn::{GateProjection, GdnBlock, GdnGeometry, GdnLayerWeights};
use xabe_gguf::GgufFile;
use xabe_kernels::quant::{BLOCK_Q6_K_BYTES, QK_K, dequantize_row_q6_k, quantize_q6_k};
use xabe_kernels::rng::Xorshift64Star;
use xabe_model::config::ModelConfig;
use xabe_model::weights::WeightSchema;

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llmxabe/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

/// Layer 0, matching the anchor layer every other GDN differential uses.
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

/// `quantize_q6_k` over a whole tensor, returning the packed bytes and the
/// f32 values those bytes denote. The caller feeds the first to the Q6_K
/// kernel and the second to the f32 kernel.
fn requantize(values: &[f32]) -> (Vec<u8>, Vec<f32>) {
    assert!(
        values.len().is_multiple_of(QK_K),
        "a Q6_K tensor is a whole number of 256-element superblocks; \
         got {} elements",
        values.len(),
    );
    let mut packed = Vec::with_capacity(values.len() / QK_K * BLOCK_Q6_K_BYTES);
    let (blocks, rest) = values.as_chunks::<QK_K>();
    assert!(rest.is_empty(), "checked above");
    for block in blocks {
        packed.extend_from_slice(&quantize_q6_k(block).to_bytes());
    }
    let reference = dequantize_row_q6_k(&packed).expect("a whole number of Q6_K blocks");
    assert_eq!(reference.len(), values.len());
    (packed, reference)
}

/// The five outputs `alpha_beta_gates` writes, read back.
struct Gates {
    alpha: Vec<f32>,
    beta_raw: Vec<f32>,
    a_softplus: Vec<f32>,
    log_decay: Vec<f32>,
    beta: Vec<f32>,
}

#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the kernel's own five-output signature"
)]
fn run_gates(
    block: &GdnBlock,
    stream: &Arc<CudaStream>,
    w_alpha: &GateProjection,
    w_beta: &GateProjection,
    x: &CudaSlice<f32>,
    dt_bias: &CudaSlice<f32>,
    a: &CudaSlice<f32>,
    heads: usize,
    tokens: usize,
) -> Gates {
    let n = tokens * heads;
    let mut alpha = stream.alloc_zeros::<f32>(n).expect("alpha allocates");
    let mut beta_raw = stream.alloc_zeros::<f32>(n).expect("beta_raw allocates");
    let mut a_softplus = stream.alloc_zeros::<f32>(n).expect("a_softplus allocates");
    let mut log_decay = stream.alloc_zeros::<f32>(n).expect("log_decay allocates");
    let mut beta = stream.alloc_zeros::<f32>(n).expect("beta allocates");
    block
        .alpha_beta_gates(
            stream,
            w_alpha,
            w_beta,
            x,
            dt_bias,
            a,
            &mut alpha,
            &mut beta_raw,
            &mut a_softplus,
            &mut log_decay,
            &mut beta,
            tokens,
        )
        .expect("the gate kernel runs");
    Gates {
        alpha: dtoh(stream, &alpha),
        beta_raw: dtoh(stream, &beta_raw),
        a_softplus: dtoh(stream, &a_softplus),
        log_decay: dtoh(stream, &log_decay),
        beta: dtoh(stream, &beta),
    }
}

/// Every element of every output, or the first disagreement with enough
/// context to locate it.
fn assert_identical(label: &str, candidate: &Gates, reference: &Gates) {
    for (name, (c, r)) in [
        ("alpha", (&candidate.alpha, &reference.alpha)),
        ("beta_raw", (&candidate.beta_raw, &reference.beta_raw)),
        ("a_softplus", (&candidate.a_softplus, &reference.a_softplus)),
        ("log_decay", (&candidate.log_decay, &reference.log_decay)),
        ("beta", (&candidate.beta, &reference.beta)),
    ] {
        assert_eq!(c.len(), r.len(), "{label}: {name} length");
        for (i, (cv, rv)) in c.iter().zip(r).enumerate() {
            assert_eq!(
                cv.to_bits(),
                rv.to_bits(),
                "{label}: {name}[{i}] differs — Q6_K {cv:e} against f32 {rv:e}. \
                 These are the same numbers summed in the same order, so any \
                 difference is an unpacking or ordering defect, not rounding.",
            );
        }
    }
}

/// The whole point: Q6_K in, and the same numbers out as the f32 kernel
/// given exactly what the Q6_K bytes denote. Run at both token tiles.
#[test]
fn the_q6_k_gate_kernel_matches_the_f32_one_on_the_values_it_encodes() {
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

    // This file stores the gates f32, which is what makes it the right
    // fixture: the reference side needs no unpacking of its own.
    let (GateProjection::F32(alpha_f32), GateProjection::F32(beta_f32)) =
        (&weights.alpha, &weights.beta)
    else {
        panic!(
            "expected {} to store ssm_alpha/ssm_beta as f32; if that changed, \
             this test needs a different fixture",
            model_path().display(),
        );
    };

    let hidden = geometry.hidden;
    let heads = geometry.value_heads;
    let (packed_alpha, ref_alpha) = requantize(&dtoh(&stream, alpha_f32));
    let (packed_beta, ref_beta) = requantize(&dtoh(&stream, beta_f32));
    assert_eq!(ref_alpha.len(), hidden * heads);

    let w_q6k_alpha = GateProjection::Q6K(stream.clone_htod(&packed_alpha).expect("upload alpha"));
    let w_q6k_beta = GateProjection::Q6K(stream.clone_htod(&packed_beta).expect("upload beta"));
    let w_f32_alpha = GateProjection::F32(stream.clone_htod(&ref_alpha).expect("upload alpha f32"));
    let w_f32_beta = GateProjection::F32(stream.clone_htod(&ref_beta).expect("upload beta f32"));

    let dt_bias = weights.dt_bias;
    let a = weights.a;

    // Both entry points, and every shape of grid either can be launched with.
    // `tokens < GATE_TT` takes `_t1` with `tt = 1`, otherwise the tiled
    // kernel with `tt = GATE_TT`; the grid's y extent is `tokens / tt`
    // rounded up, from the same `tt`, so:
    //
    //   1   `_t1`,   one block   — the decode width
    //   2   `_t1`,   two blocks  — multi-block on the untiled kernel
    //   7   `_t1`,   seven       — the widest the untiled path takes
    //   8   tiled,   one block   — exactly one full tile
    //   9   tiled,   two blocks  — a full tile plus a one-row tail
    //   17  tiled,   three       — two full tiles plus a tail
    //
    // 2 and 9 are the ones worth naming. A token count that is neither one
    // nor a multiple of the tile is where a kernel and the grid it was sized
    // with can disagree, and that disagreement — `tt` and a second width
    // derived by rules that agreed at 1 and 3 and not at 2 — is what faulted
    // main in e4d43e0. It cannot happen here, because `tt` and the function
    // come out of one match arm and the grid comes out of that same `tt`.
    // Tested rather than argued, because that is what the router's own
    // differential had missing at exactly this width.
    for tokens in [1usize, 2, 7, 8, 9, 17] {
        let mut rng = Xorshift64Star::new(0x_6A11_E500 ^ tokens as u64);
        let x_host = rng.vec_f32(tokens * hidden, -1.0, 1.0);
        let x = stream.clone_htod(&x_host).expect("upload x");

        let candidate = run_gates(
            &block,
            &stream,
            &w_q6k_alpha,
            &w_q6k_beta,
            &x,
            &dt_bias,
            &a,
            heads,
            tokens,
        );
        let reference = run_gates(
            &block,
            &stream,
            &w_f32_alpha,
            &w_f32_beta,
            &x,
            &dt_bias,
            &a,
            heads,
            tokens,
        );
        assert_identical(&format!("tokens={tokens}"), &candidate, &reference);
        println!("tokens={tokens}: Q6_K gates bit-identical to f32 across all five outputs");
    }
}

/// One launch serves both gates, so a pair in two different formats has no
/// kernel. It must be refused rather than silently reading one of them
/// through the other's unpacker.
#[test]
fn a_mixed_format_gate_pair_is_rejected() {
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

    let GateProjection::F32(alpha_f32) = &weights.alpha else {
        panic!("this fixture stores the gates f32");
    };
    let (packed_alpha, _) = requantize(&dtoh(&stream, alpha_f32));
    let w_q6k = GateProjection::Q6K(stream.clone_htod(&packed_alpha).expect("upload alpha"));

    let hidden = geometry.hidden;
    let heads = geometry.value_heads;
    let n = heads;
    let x = stream.alloc_zeros::<f32>(hidden).expect("x allocates");
    let mut alpha = stream.alloc_zeros::<f32>(n).expect("alpha allocates");
    let mut beta_raw = stream.alloc_zeros::<f32>(n).expect("beta_raw allocates");
    let mut a_softplus = stream.alloc_zeros::<f32>(n).expect("a_softplus allocates");
    let mut log_decay = stream.alloc_zeros::<f32>(n).expect("log_decay allocates");
    let mut beta = stream.alloc_zeros::<f32>(n).expect("beta allocates");

    // Q6_K alpha against the file's own f32 beta.
    let err = block
        .alpha_beta_gates(
            &stream,
            &w_q6k,
            &weights.beta,
            &x,
            &weights.dt_bias,
            &weights.a,
            &mut alpha,
            &mut beta_raw,
            &mut a_softplus,
            &mut log_decay,
            &mut beta,
            1,
        )
        .expect_err("a mixed-format pair has no kernel and must be refused");
    let text = err.to_string();
    assert!(
        text.contains("same format"),
        "the error should say the formats must agree; got {text}",
    );
}
