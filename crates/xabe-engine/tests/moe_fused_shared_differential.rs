//! The shared expert under the routed expert's grid
//! (`MoeKernels::grouped_forward_partial_with_shared`) against the two
//! separate paths it replaces (`grouped_forward_partial` then
//! `shared_expert`), on real layer weights, at every decode width the fused
//! entries cover.
//!
//! Bit-exact, no tolerance: the fused entry runs the separate launches'
//! bodies on the same block shapes, so `partial`, `shared_inter` and the
//! shared expert's output must be identical floats. A nonzero difference
//! means a re-based block index is wrong, or the two-rows-per-block shared
//! gate/up body has drifted from the one-row kernel it copies.
//!
//! Widths 1..=4 are every width the fused path serves — the one-token GEMV
//! bodies at 1, the direct flat bodies at 2, 3 and 4 — and each is run
//! with every token slot live *and* with one live token in a wider pass, so
//! the `valid_tokens` gate on the shared rows is exercised as well as the
//! seam.
//!
//! SKIPS — reporting that it skipped — without a driver, a supported device,
//! or the model file.

use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_cuda::kernels::moe::{
    ExpertQuant, MoeGeometry, MoeKernels, QuantTensor, to_device_layout,
};
use xabe_gguf::{GgmlType, GgufFile};
use xabe_kernels::rng::Xorshift64Star;
use xabe_model::config::ModelConfig;
use xabe_model::weights::{Role, WeightSchema};

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llmxabe/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

/// Layer 0: Q6_K routed gate/up, Q8_0 routed down, Q8_0 shared expert in
/// the shipped file — the mixed case the fused entries exist for.
const LAYER: u32 = 0;

const BLOCK_SIZE: usize = 16;

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
        println!(
            "SKIPPED: model file not found at {}; set LLMXABE_MODEL to override",
            path.display(),
        );
        return None;
    }
    Some((ctx, GgufFile::open(&path).expect("valid GGUF v3")))
}

fn quant_of(ty: GgmlType) -> ExpertQuant {
    match ty {
        GgmlType::Q6K => ExpertQuant::Q6K,
        GgmlType::Q8_0 => ExpertQuant::Q8_0,
        other => panic!("unexpected expert tensor type {}", other.name()),
    }
}

fn dtoh(stream: &Arc<CudaStream>, buf: &CudaSlice<f32>) -> Vec<f32> {
    let v = stream.clone_dtoh(buf).expect("device read-back");
    stream.synchronize().expect("sync");
    v
}

struct Uploaded {
    bytes: CudaSlice<u8>,
    quant: ExpertQuant,
}

impl Uploaded {
    fn tensor(&self) -> QuantTensor<'_> {
        QuantTensor {
            bytes: &self.bytes,
            quant: self.quant,
        }
    }
}

fn upload(
    stream: &Arc<CudaStream>,
    file: &GgufFile,
    directory: &xabe_model::weights::Directory<'_>,
    role: Role,
) -> Uploaded {
    let entry = directory
        .find(role, Some(LAYER))
        .unwrap_or_else(|| panic!("{role} on layer {LAYER} missing"));
    let bytes = file
        .tensor_bytes(&entry.spec.name)
        .expect("tensor readable");
    let quant = quant_of(entry.info.ggml_type);
    let bytes = stream
        .clone_htod(&*to_device_layout(quant, bytes))
        .expect("upload");
    Uploaded { bytes, quant }
}

#[test]
fn fused_shared_expert_agrees_with_the_separate_launches() {
    let Some((ctx, file)) = setup() else {
        return;
    };
    let config = ModelConfig::qwen3_6_35b_a3b();
    let moe_cfg = config.moe().expect("routed model");
    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).expect("schema resolves");
    let stream = ctx.default_stream();

    let gate = upload(&stream, &file, &directory, Role::MoeGateExps);
    let up = upload(&stream, &file, &directory, Role::MoeUpExps);
    let down = upload(&stream, &file, &directory, Role::MoeDownExps);
    let s_gate = upload(&stream, &file, &directory, Role::MoeSharedGate);
    let s_up = upload(&stream, &file, &directory, Role::MoeSharedUp);
    let s_down = upload(&stream, &file, &directory, Role::MoeSharedDown);
    stream.synchronize().expect("sync");
    println!(
        "layer {LAYER}: routed {:?}/{:?}/{:?}, shared {:?}/{:?}/{:?}",
        gate.quant, up.quant, down.quant, s_gate.quant, s_up.quant, s_down.quant
    );

    let hidden = config.hidden_size as usize;
    let num_experts = moe_cfg.num_experts as usize;
    let mut rng = Xorshift64Star::new(0x_5EED_F05E);

    // (max_tokens, live tokens): every slot live, and one live token in a
    // wider pass so the `valid_tokens` gate on the shared rows is exercised.
    for (max_tokens, live) in [(1, 1), (2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 1)] {
        let g = MoeGeometry {
            num_experts,
            experts_per_token: moe_cfg.experts_per_token as usize,
            hidden,
            intermediate: moe_cfg.expert_intermediate as usize,
            block_size: BLOCK_SIZE,
            max_tokens,
        };
        let rows = rng.vec_f32(max_tokens * hidden, -1.0, 1.0);
        let d_hidden = stream.clone_htod(&rows).expect("upload hidden");
        let mut logits = rng.vec_f32(live * num_experts, -8.0, 8.0);
        logits.resize(max_tokens * num_experts, -1.0e30);
        let d_logits = stream.clone_htod(&logits).expect("upload logits");

        // Two independent kernel/buffer sets so nothing carries between the
        // arms but the inputs.
        let run = |fused: bool| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
            let kernels = MoeKernels::new(&ctx, g).expect("compiles");
            let mut buffers = kernels.buffers(&stream).expect("buffers");
            kernels
                .set_valid_tokens(&stream, &mut buffers, live)
                .expect("valid_tokens");
            kernels
                .route_and_dispatch(&stream, &mut buffers, &d_logits)
                .expect("route");
            let mut out = stream
                .alloc_zeros::<f32>(max_tokens * hidden)
                .expect("out allocates");
            if fused {
                let took = kernels
                    .grouped_forward_partial_with_shared(
                        &stream,
                        &mut buffers,
                        gate.tensor(),
                        up.tensor(),
                        down.tensor(),
                        s_gate.tensor(),
                        s_up.tensor(),
                        s_down.tensor(),
                        &d_hidden,
                        &mut out,
                    )
                    .expect("fused launches");
                assert!(
                    took,
                    "max_tokens {max_tokens}: the fused path declined a covered shape"
                );
            } else {
                kernels
                    .grouped_forward_partial(
                        &stream,
                        &mut buffers,
                        gate.tensor(),
                        up.tensor(),
                        down.tensor(),
                        &d_hidden,
                    )
                    .expect("routed launches");
                kernels
                    .shared_expert(
                        &stream,
                        &mut buffers,
                        s_gate.tensor(),
                        s_up.tensor(),
                        s_down.tensor(),
                        &d_hidden,
                        &mut out,
                    )
                    .expect("shared launches");
            }
            stream.synchronize().expect("sync");
            (
                dtoh(&stream, buffers.partial()),
                dtoh(&stream, buffers.shared_inter()),
                dtoh(&stream, &out),
            )
        };

        let (p_sep, si_sep, out_sep) = run(false);
        let (p_fused, si_fused, out_fused) = run(true);

        let finite = |v: &[f32]| v.iter().all(|x| x.is_finite());
        assert!(
            finite(&p_sep) && finite(&si_sep) && finite(&out_sep),
            "max_tokens {max_tokens}, live {live}: the separate launches produced a non-finite value"
        );
        let signal = out_sep[..live * hidden]
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            signal > 0.0,
            "max_tokens {max_tokens}: the shared expert produced all zeros"
        );
        println!(
            "max_tokens {max_tokens}, live {live}: partial {} floats, shared_inter {} floats, out {} floats, max |out| {signal:.3e}, all bit-identical: {}",
            p_sep.len(),
            si_sep.len(),
            out_sep.len(),
            p_sep == p_fused && si_sep == si_fused && out_sep == out_fused
        );
        assert_eq!(
            p_sep, p_fused,
            "max_tokens {max_tokens}, live {live}: routed `partial` differs under the fused grid"
        );
        assert_eq!(
            si_sep, si_fused,
            "max_tokens {max_tokens}, live {live}: `shared_inter` differs under the fused grid"
        );
        assert_eq!(
            out_sep, out_fused,
            "max_tokens {max_tokens}, live {live}: the shared expert's output differs under the fused grid"
        );
    }
}
