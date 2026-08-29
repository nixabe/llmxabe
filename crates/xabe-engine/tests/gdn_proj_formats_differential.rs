//! `gdn_proj_qgeneric_*` against `gdn_proj_f32`, on weights the two are
//! guaranteed to agree on.
//!
//! The GDN block's own projections — `attn_qkv`, `attn_gate`, `ssm_out` —
//! were the last wall in `docs/MODEL.md`'s format table. `Projection` had two
//! variants because `gdn_proj_*` had two readers, so a uniformly Q6_K file
//! loaded its experts, head, embedding and gates and stopped here.
//!
//! Both shipped files store all three Q8_0, so as with the gates there is no
//! golden and no fixture. This makes one, the same way
//! `gdn_gates_q6k_differential.rs` does: take the file's real `attn_qkv`,
//! dequantize it, requantize to the format under test, and feed
//!
//!   packed    = quantize(w)              -> the generic kernel
//!   reference = dequantize(packed)       -> `gdn_proj_f32`
//!
//! `reference` is exactly the set of values `packed` denotes, so both kernels
//! are asked for the same dot products over the same floats. Quantization
//! error cancels: it is on both sides. What is left is the unpacking — the
//! nibble and bit-plane interleave, the sub-scale index, the row stride, and
//! the affine formats' rounding — which is the only thing here that has never
//! run.
//!
//! **The bar is exact equality.** `gdn_proj_f32` and `gdn_proj_qgeneric_*`
//! have the same body apart from the unpack: `for (i = lane; i < k_dim; i +=
//! 32) acc += w(i) * x[i]`, then one `warp_reduce_sum`. Same addends, same
//! order, same result to the last bit. A tolerance would hide exactly the
//! reordering and stride defects this exists to catch.
//!
//! Q8_0 is in the list as a **control that runs the code under test**: it has
//! its own faster kernel, but `Projection::Quant(bytes, ProjQuant::Q8_0)`
//! goes through the generic one, so a green result on the other six is not
//! resting on a control that took a different path. That distinction caught a
//! real double-dispatch bug in the MoE community routing the same day.
//!
//! SKIPS — reporting that it skipped — without a driver, a supported device,
//! or the model file.

use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::{bf16, f16};
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_engine::block::gdn::{GdnBlock, GdnGeometry, GdnLayerWeights, ProjQuant, Projection};
use xabe_gguf::GgufFile;
use xabe_kernels::quant::{
    QK_K, QK4_0, QK8_0, dequantize_row_bf16, dequantize_row_f16, dequantize_row_q4_0,
    dequantize_row_q4_k, dequantize_row_q5_k, dequantize_row_q6_k, dequantize_row_q8_0,
    quantize_q4_0, quantize_q4_k, quantize_q5_k, quantize_q6_k, quantize_q8_0,
};
use xabe_kernels::rng::Xorshift64Star;
use xabe_model::config::ModelConfig;
use xabe_model::weights::WeightSchema;

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llmxabe/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

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
    Some((ctx, file, ModelConfig::qwen3_6_35b_a3b()))
}

fn dtoh_f32(stream: &Arc<CudaStream>, buf: &CudaSlice<f32>) -> Vec<f32> {
    let v = stream.clone_dtoh(buf).expect("device read-back");
    stream.synchronize().expect("sync");
    v
}

/// Requantize `rows x k` values, returning the packed bytes and the values
/// those bytes denote. Packing is per row, because the row stride is what the
/// kernel indexes by and a format whose superblocks straddled a row boundary
/// would be a different (wrong) layout.
fn pack(fmt: ProjQuant, values: &[f32], rows: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(values.len(), rows * k);
    assert!(
        k.is_multiple_of(fmt.k_multiple()),
        "k_dim {k} is not a whole number of {fmt:?} blocks",
    );
    let mut packed = Vec::with_capacity(rows * fmt.row_bytes(k));
    for r in 0..rows {
        let row = &values[r * k..(r + 1) * k];
        match fmt {
            ProjQuant::Q8_0 => {
                let (blocks, rest) = row.as_chunks::<QK8_0>();
                debug_assert!(rest.is_empty(), "checked above");
                for c in blocks {
                    packed.extend_from_slice(&quantize_q8_0(c).to_bytes());
                }
            }
            ProjQuant::Q4_0 => {
                let (blocks, rest) = row.as_chunks::<QK4_0>();
                debug_assert!(rest.is_empty(), "checked above");
                for c in blocks {
                    packed.extend_from_slice(&quantize_q4_0(c).to_bytes());
                }
            }
            ProjQuant::Q6K => {
                let (blocks, rest) = row.as_chunks::<QK_K>();
                debug_assert!(rest.is_empty(), "checked above");
                for c in blocks {
                    packed.extend_from_slice(&quantize_q6_k(c).to_bytes());
                }
            }
            ProjQuant::Q4K => {
                let (blocks, rest) = row.as_chunks::<QK_K>();
                debug_assert!(rest.is_empty(), "checked above");
                for c in blocks {
                    packed.extend_from_slice(&quantize_q4_k(c).to_bytes());
                }
            }
            ProjQuant::Q5K => {
                let (blocks, rest) = row.as_chunks::<QK_K>();
                debug_assert!(rest.is_empty(), "checked above");
                for c in blocks {
                    packed.extend_from_slice(&quantize_q5_k(c).to_bytes());
                }
            }
            ProjQuant::F16 => {
                for v in row {
                    packed.extend_from_slice(&f16::from_f32(*v).to_le_bytes());
                }
            }
            ProjQuant::Bf16 => {
                for v in row {
                    packed.extend_from_slice(&bf16::from_f32(*v).to_le_bytes());
                }
            }
        }
    }
    assert_eq!(packed.len(), rows * fmt.row_bytes(k), "{fmt:?} row stride");

    let reference: Vec<f32> = (0..rows)
        .flat_map(|r| {
            let rb = fmt.row_bytes(k);
            let bytes = &packed[r * rb..(r + 1) * rb];
            match fmt {
                ProjQuant::Q8_0 => dequantize_row_q8_0(bytes).expect("q8_0 blocks"),
                ProjQuant::Q4_0 => dequantize_row_q4_0(bytes).expect("q4_0 blocks"),
                ProjQuant::Q6K => dequantize_row_q6_k(bytes).expect("q6_K blocks"),
                ProjQuant::Q4K => dequantize_row_q4_k(bytes).expect("q4_K blocks"),
                ProjQuant::Q5K => dequantize_row_q5_k(bytes).expect("q5_K blocks"),
                ProjQuant::F16 => dequantize_row_f16(bytes),
                ProjQuant::Bf16 => dequantize_row_bf16(bytes),
            }
        })
        .collect();
    assert_eq!(reference.len(), values.len());
    (packed, reference)
}

fn check(fmt: ProjQuant, label: &str) {
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

    let k = geometry.hidden;
    let rows = geometry.conv_dim();

    // The file's own `attn_qkv`, dequantized: real weights, real magnitudes.
    let q8_bytes = stream.clone_dtoh(&weights.qkv).expect("qkv back");
    stream.synchronize().expect("sync");
    let dense = dequantize_row_q8_0(&q8_bytes).expect("a whole number of Q8_0 blocks");
    assert_eq!(dense.len(), rows * k, "attn_qkv is [hidden, conv_dim]");

    let (packed, reference) = pack(fmt, &dense, rows, k);
    let d_packed = stream.clone_htod(&packed).expect("packed up");
    let d_reference = stream.clone_htod(&reference).expect("reference up");

    // One token exercises a single-block grid; several exercise `blockIdx.y`
    // and the `t * k_dim` / `t * n_rows` strides, which no single-token run
    // can distinguish from ignoring `t` entirely.
    for tokens in [1usize, 2, 5] {
        let mut rng = Xorshift64Star::new(0x_9C0F_1200 ^ tokens as u64);
        let x_host = rng.vec_f32(tokens * k, -1.0, 1.0);
        let x = stream.clone_htod(&x_host).expect("x up");

        let mut got = stream.alloc_zeros::<f32>(tokens * rows).expect("out alloc");
        block
            .project(
                &stream,
                Projection::Quant(&d_packed, fmt),
                &x,
                &mut got,
                k,
                rows,
                tokens,
            )
            .expect("generic projection runs");
        let candidate = dtoh_f32(&stream, &got);

        let mut want = stream.alloc_zeros::<f32>(tokens * rows).expect("ref alloc");
        block
            .project(
                &stream,
                Projection::F32(&d_reference),
                &x,
                &mut want,
                k,
                rows,
                tokens,
            )
            .expect("f32 projection runs");
        let expected = dtoh_f32(&stream, &want);

        for (i, (c, r)) in candidate.iter().zip(&expected).enumerate() {
            assert_eq!(
                c.to_bits(),
                r.to_bits(),
                "{label} tokens={tokens}: element {i} differs — generic {c:e} \
                 against f32 {r:e}. Both kernels sum the same floats in the \
                 same lane-strided order, so any difference is an unpacking \
                 or row-stride defect, not rounding.",
            );
        }
        println!("{label}, tokens={tokens}: bit-identical over {rows} rows");
    }
}

#[test]
fn the_generic_projection_reads_q4_0() {
    check(ProjQuant::Q4_0, "q4_0");
}

#[test]
fn the_generic_projection_reads_q6_k() {
    check(ProjQuant::Q6K, "q6_K");
}

#[test]
fn the_generic_projection_reads_q4_k() {
    check(ProjQuant::Q4K, "q4_K");
}

#[test]
fn the_generic_projection_reads_q5_k() {
    check(ProjQuant::Q5K, "q5_K");
}

#[test]
fn the_generic_projection_reads_f16() {
    check(ProjQuant::F16, "f16");
}

#[test]
fn the_generic_projection_reads_bf16() {
    check(ProjQuant::Bf16, "bf16");
}

/// The control, and it runs the same kernel as the cases above.
///
/// Q8_0 has its own tiled path, so routing it through `Projection::Quant`
/// deliberately bypasses that and exercises `gdn_proj_qgeneric_q8_0`. If this
/// fails, the harness or the generic kernel is wrong and none of the six
/// results above is evidence about a format.
#[test]
fn the_shipped_format_goes_through_the_generic_kernel_too() {
    check(ProjQuant::Q8_0, "q8_0 via generic");
}
