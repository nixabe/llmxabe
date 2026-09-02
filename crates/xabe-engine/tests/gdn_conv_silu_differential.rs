//! `gdn_conv_silu_split_step_batch` against the two launches it replaces.
//!
//! At decode width the Gated DeltaNet step ran `conv1d_step_batch` and then
//! `gdn_silu_split_qkv`, the second reading back what the first wrote. The
//! fused launch convolves a channel and gates its own accumulator, with the
//! same per-channel arithmetic as each separate kernel, so its five outputs
//! and the advanced convolution caches must be **bit-identical** to the
//! two-launch form — not close, identical — at every batch width the pointer
//! slots cover. Each sequence gets its own random cache and its own random
//! token so a wrong slot or stride has something to trip over.

use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_cuda::kernels::gdn::STEP_MAX_BATCH;
use xabe_engine::block::gdn::{GdnBlock, GdnGeometry, GdnLayerWeights};
use xabe_gguf::GgufFile;
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
    let config = ModelConfig::qwen3_6_35b_a3b();
    Some((ctx, file, config))
}

/// `conv_raw`, `conv_silu`, `q`, `k`, `v`, and the advanced cache per sequence.
type Outputs = (
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<Vec<f32>>,
);

fn dtoh(stream: &Arc<CudaStream>, buf: &CudaSlice<f32>) -> Vec<f32> {
    let v = stream.clone_dtoh(buf).expect("device read-back");
    stream.synchronize().expect("sync");
    v
}

#[test]
fn fused_conv_silu_split_agrees_with_the_two_launches() {
    let Some((ctx, file, config)) = setup() else {
        return;
    };
    let stream = ctx.default_stream();
    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).expect("schema resolves");
    let geometry = GdnGeometry::from_config(&config, STEP_MAX_BATCH, 1e-6);
    let block = GdnBlock::new(&ctx, geometry).expect("kernels compile");
    let weights =
        GdnLayerWeights::upload(&stream, &file, &directory, LAYER).expect("weights upload");

    let conv_dim = geometry.conv_dim();
    let key_dim = geometry.key_dim();
    let value_dim = geometry.value_dim();
    let cache_len = geometry.conv_state_len();
    let mut rng = Xorshift64Star::new(0x_C0DE_5EED_0002);

    for &n in &[1usize, 2, 3, 4, STEP_MAX_BATCH] {
        // One token per sequence, one cache per sequence, every value
        // distinct.
        let x_host = rng.vec_f32(n * conv_dim, -2.0, 2.0);
        let caches_host: Vec<Vec<f32>> =
            (0..n).map(|_| rng.vec_f32(cache_len, -2.0, 2.0)).collect();
        let x = stream.clone_htod(&x_host).expect("upload x");

        let run = |fused: bool| -> Outputs {
            let mut caches: Vec<CudaSlice<f32>> = caches_host
                .iter()
                .map(|c| stream.clone_htod(c).expect("upload cache"))
                .collect();
            let ptrs: Vec<u64> = caches.iter().map(|c| c.device_ptr(&stream).0).collect();
            let mut conv_raw = stream.alloc_zeros::<f32>(n * conv_dim).expect("conv_raw");
            let mut silu = stream.alloc_zeros::<f32>(n * conv_dim).expect("silu");
            let mut q = stream.alloc_zeros::<f32>(n * key_dim).expect("q");
            let mut k = stream.alloc_zeros::<f32>(n * key_dim).expect("k");
            let mut v = stream.alloc_zeros::<f32>(n * value_dim).expect("v");
            if fused {
                // SAFETY: `caches` are live, distinct allocations of
                // `cache_len` floats, held for the call.
                unsafe {
                    block
                        .conv_silu_split_step_batch_raw(
                            &stream,
                            &x,
                            &weights.conv1d,
                            &ptrs,
                            &mut conv_raw,
                            &mut silu,
                            &mut q,
                            &mut k,
                            &mut v,
                        )
                        .expect("fused launch");
                }
            } else {
                // SAFETY: as above.
                unsafe {
                    block
                        .layer_ops()
                        .conv1d_step_batch_raw(
                            &stream,
                            &x,
                            &weights.conv1d,
                            &ptrs,
                            &mut conv_raw,
                            conv_dim,
                            geometry.conv_kernel,
                        )
                        .expect("conv launch");
                }
                block
                    .silu_split_qkv(&stream, &conv_raw, &mut silu, &mut q, &mut k, &mut v, n)
                    .expect("silu split launch");
            }
            let advanced = caches.iter_mut().map(|c| dtoh(&stream, c)).collect();
            (
                dtoh(&stream, &conv_raw),
                dtoh(&stream, &silu),
                dtoh(&stream, &q),
                dtoh(&stream, &k),
                dtoh(&stream, &v),
                advanced,
            )
        };

        let separate = run(false);
        let fused = run(true);
        assert_eq!(separate.0, fused.0, "n {n}: conv_raw differs");
        assert_eq!(separate.1, fused.1, "n {n}: conv_silu differs");
        assert_eq!(separate.2, fused.2, "n {n}: q differs");
        assert_eq!(separate.3, fused.3, "n {n}: k differs");
        assert_eq!(separate.4, fused.4, "n {n}: v differs");
        assert_eq!(separate.5, fused.5, "n {n}: advanced conv caches differ");
        // Sanity: the run did something to the caches (they hold the new
        // input) and the gate is not comparing two untouched buffers.
        for (seq, before) in caches_host.iter().enumerate() {
            assert_ne!(
                &fused.5[seq], before,
                "n {n}: sequence {seq}'s cache did not advance"
            );
        }
        println!("n {n}: conv_raw, conv_silu, q, k, v and {n} caches bit-identical");
    }
}
