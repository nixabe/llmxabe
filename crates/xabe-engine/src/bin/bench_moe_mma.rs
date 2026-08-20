//! `moe_expert_ffn_mma` and `moe_expert_down_mma` alone, at the shapes a
//! chunked prefill actually launches them at.
//!
//! An nsys profile puts these two kernels at 17.3% and 10.2% of an
//! 8,192-token chunked pass —
//! the single biggest named cost after attention. The routed-expert MMA
//! kernel measured 16.1% of the card's 198 TOP/s int8 peak at its own real
//! shape and 31% of bandwidth peak, so it is latency-bound rather than
//! traffic-bound, and the same section names a Marlin-style shared-memory
//! staging pipeline as what would reach higher.
//!
//! This runs `MoeKernels::grouped_forward` — routing, dispatch, both MMA
//! GEMMs and the reduction — against one real layer's quantized expert
//! stacks (Q6_K gate/up, Q8_0 down, exactly as `moe_differential.rs` loads
//! them), so a kernel change to the staging loop can be measured in seconds
//! and interleaved against its predecessor instead of paid for with a full
//! model load and a `bench_forward` run.
//!
//! Not a duplicate of `xabe-cuda`'s own `bench_moe`: that one covers all four
//! public entry points at synthetic weights and batches up to 512, which is
//! the right tool for a broad sweep. This one is narrower and heavier —
//! real GGUF weights, up to 8,192 tokens, the two MMA kernels a chunked
//! prefill actually spends time in — which is what an isolated A/B on their
//! staging loop needs. The name is distinct because a cargo workspace cannot
//! link two same-named bins from different crates in one build.
//!
//! ```text
//! CUDA_VISIBLE_DEVICES=0 cargo run --release -p xabe-engine --bin bench_moe_mma
//! ```
//!
//! `LLMXABE_MODEL` overrides the model path.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use cudarc::driver::CudaContext;
use tracing::{error, info, warn};
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_cuda::kernels::moe::{
    ExpertQuant, MoeGeometry, MoeKernels, QuantTensor, to_device_layout,
};
use xabe_gguf::{GgmlType, GgufFile};
use xabe_model::config::ModelConfig;
use xabe_model::weights::{Role, WeightSchema};

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llama.cpp/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

/// The layer whose expert stacks are measured. Mixed quant, exactly as
/// `moe_differential.rs` relies on: gate/up are Q6_K, down is Q8_0, so both
/// `moe_expert_ffn_mma` and `moe_expert_down_mma` are exercised, not just one.
const LAYER: u32 = 0;

/// Chunk widths a real prefill actually launches at 8,192-token chunking,
/// plus the 512 row bench_attention and bench_mma already use. Each gets its
/// own dispatch tile width from `xabe_engine::forward::moe_block_size` --
/// the engine's own choice, not a fixed constant, since 512 and 8,192 now
/// take different compiled kernels. See "Two compiled widths instead of
/// one" in docs/BENCHMARKS.md.
const TOKEN_COUNTS: [usize; 2] = [512, 8192];

const WARMUP: usize = 3;
const REPS: usize = 10;

/// Xorshift64*, inline rather than through `xabe_kernels::rng` — that crate is
/// a dev-dependency, reachable from `tests/` and not from a binary. Nothing
/// here checks a value; correctness belongs to `moe_differential.rs`.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f32(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next() >> 40) as f32 / 16_777_216.0; // [0, 1)
        lo + u * (hi - lo)
    }
}

fn quant_of(ty: GgmlType) -> ExpertQuant {
    match ty {
        GgmlType::Q6K => ExpertQuant::Q6K,
        GgmlType::Q8_0 => ExpertQuant::Q8_0,
        other => panic!("unexpected expert stack type {}", other.name()),
    }
}

fn main() -> ExitCode {
    xabe_log::init_from_args();

    if !driver_available() {
        warn!("SKIPPED: no CUDA driver present");
        return ExitCode::SUCCESS;
    }
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            error!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let info = DeviceInfo::from_context(0, &ctx)?;
    if !info.is_supported() {
        warn!("SKIPPED: device 0 is below the sm_75 minimum");
        return Ok(());
    }
    let stream = ctx.default_stream();

    let path = std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH));
    if !path.exists() {
        warn!("SKIPPED: model file not found at {}", path.display());
        return Ok(());
    }
    let file = GgufFile::open(&path)?;
    let config = ModelConfig::qwen3_6_35b_a3b();
    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).expect("schema must resolve");

    let stacks: Vec<_> = [Role::MoeGateExps, Role::MoeUpExps, Role::MoeDownExps]
        .iter()
        .map(|&role| {
            directory
                .find(role, Some(LAYER))
                .unwrap_or_else(|| panic!("{role} on layer {LAYER} missing"))
        })
        .collect();
    info!(
        "device 0: {} -- layer {LAYER} expert stacks: {}",
        info.name,
        stacks
            .iter()
            .map(|e| format!("{}={}", e.spec.role, e.info.ggml_type.name()))
            .collect::<Vec<_>>()
            .join("  "),
    );

    let bytes: Vec<&[u8]> = stacks
        .iter()
        .map(|e| file.tensor_bytes(&e.spec.name).expect("tensor readable"))
        .collect();
    let quants: Vec<ExpertQuant> = stacks.iter().map(|e| quant_of(e.info.ggml_type)).collect();

    let t0 = Instant::now();
    let d_gate = stream.clone_htod(&*to_device_layout(quants[0], bytes[0]))?;
    let d_up = stream.clone_htod(&*to_device_layout(quants[1], bytes[1]))?;
    let d_down = stream.clone_htod(&*to_device_layout(quants[2], bytes[2]))?;
    stream.synchronize()?;
    info!(
        "uploaded {:.1} MiB of quantized expert weights in {:.2?}\n",
        bytes.iter().map(|b| b.len()).sum::<usize>() as f64 / (1024.0 * 1024.0),
        t0.elapsed(),
    );

    let gate = QuantTensor {
        bytes: &d_gate,
        quant: quants[0],
    };
    let up = QuantTensor {
        bytes: &d_up,
        quant: quants[1],
    };
    let down = QuantTensor {
        bytes: &d_down,
        quant: quants[2],
    };

    info!("{:>7}  {:>10}  {:>10}", "tokens", "ms", "TOP/s");
    info!("{:->7}--{:->10}--{:->10}", "", "", "");

    for &tokens in &TOKEN_COUNTS {
        let g = MoeGeometry {
            num_experts: config.moe.num_experts as usize,
            experts_per_token: config.moe.experts_per_token as usize,
            hidden: config.hidden_size as usize,
            intermediate: config.moe.expert_intermediate as usize,
            block_size: xabe_engine::forward::moe_block_size(tokens),
            max_tokens: tokens,
        };
        let kernels = MoeKernels::new(&ctx, g)?;
        if !kernels.tensor_cores_enabled() {
            warn!("SKIPPED: integer tensor cores unavailable on this device");
            return Ok(());
        }
        let mut buffers = kernels.buffers(&stream)?;
        kernels.set_valid_tokens(&stream, &mut buffers, tokens)?;

        // Router logits with a realistic spread, one seed per token count so
        // routing differs but is reproducible.
        let mut rng = Rng(0x_5EED_0B0E ^ tokens as u64);
        let logits: Vec<f32> = (0..tokens * g.num_experts)
            .map(|_| rng.f32(-4.0, 4.0))
            .collect();
        let mut flat_logits = logits;
        flat_logits.resize(g.max_tokens * g.num_experts, f32::NEG_INFINITY);
        let d_logits = stream.clone_htod(&flat_logits)?;

        let hidden: Vec<f32> = (0..tokens * g.hidden).map(|_| rng.f32(-1.0, 1.0)).collect();
        let mut flat_hidden = hidden;
        flat_hidden.resize(g.max_tokens * g.hidden, 0.0);
        let d_hidden = stream.clone_htod(&flat_hidden)?;

        let mut d_out = stream.alloc_zeros::<f32>(g.max_tokens * g.hidden)?;

        kernels.route(&stream, &mut buffers, &d_logits)?;
        kernels.build_dispatch(&stream, &mut buffers)?;

        let once = |buffers: &mut _, out: &mut _| {
            kernels
                .grouped_forward(&stream, buffers, gate, up, down, &d_hidden, out)
                .expect("grouped forward");
        };
        for _ in 0..WARMUP {
            once(&mut buffers, &mut d_out);
        }
        stream.synchronize()?;

        let t = Instant::now();
        for _ in 0..REPS {
            once(&mut buffers, &mut d_out);
        }
        stream.synchronize()?;
        let ms = t.elapsed().as_secs_f64() * 1e3 / REPS as f64;

        // Two flops (multiply-add) per weight element, over both the gate/up
        // GEMM and the down GEMM, at exactly the tokens routed (top_k of them
        // per token, not `max_tokens`) -- the same accounting bench_mma uses.
        let ops = 2.0
            * tokens as f64
            * g.experts_per_token as f64
            * (2.0 * g.hidden as f64 * g.intermediate as f64 // gate + up
                + g.intermediate as f64 * g.hidden as f64); // down
        let tops = ops / (ms / 1e3) / 1e12;
        info!("{tokens:>7}  {ms:>10.3}  {tops:>10.2}");
    }

    info!("");
    info!("int8 tensor-core peak measured at ~198 TOP/s on this card.");
    Ok(())
}
