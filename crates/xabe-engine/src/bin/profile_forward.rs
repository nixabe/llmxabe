//! Per-stage breakdown of the forward pass: where the milliseconds go.
//!
//! `bench_forward` says the pass costs 38 ms at one token and 6.8 s at 512.
//! It does not say *what* costs that, and every explanation offered so far has
//! been a suspicion rather than a measurement. This binary partitions the pass
//! into consecutive, non-overlapping spans — reset, embedding, then a mixer
//! and a MoE for each of the 40 blocks, then the final norm and the LM head —
//! and reports each one's share.
//!
//! # How the time is measured
//!
//! CUDA events, not `Instant`. Every launch in the pass is asynchronous, so a
//! host clock read between two stages measures enqueue latency, not work.
//! Timing the stages with `Instant` would need a `cuStreamSynchronize` per
//! stage — 84 per pass — which drains the pipeline every time and inflates the
//! total by more than the smaller stages cost. `cuEventRecord` is a
//! stream-ordered marker: the enqueue is cheap and the host reads the whole
//! timeline once, after the pass.
//!
//! The instrumentation is off by default in [`Forward`] and switched on here
//! with `enable_profiling`, so `bench_forward`'s figure remains the
//! uninstrumented one. This binary measures the difference itself: it times
//! `reps` passes with the events off and `reps` with them on, and prints both
//! totals. Read that difference as the instrumentation's cost.
//!
//! # The roofline column
//!
//! For the dominant stages the report divides the weight bytes the stage must
//! read — taken from the GGUF directory, so it is the stored quantized size,
//! not a dequantized one — by the measured time. The card is 672 GB/s, so the
//! last column is the fraction of streaming peak a perfect implementation of
//! the same arithmetic could not exceed.
//!
//! For the MoE the necessary traffic depends on routing: at one token exactly
//! `experts_per_token` of the 256 experts are read, at 512 tokens essentially
//! all of them are. Both bounds are printed, and the report says which applies.
//!
//! ```text
//! CUDA_VISIBLE_DEVICES=2 cargo run --release -p xabe-engine --bin profile_forward
//! ```
//!
//! Environment: `LLMXABE_MODEL` overrides the model path, `LLMXABE_PROFILE_N`
//! the comma-separated batch sizes, `LLMXABE_PROFILE_REPS` the repetitions.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use cudarc::driver::CudaContext;
use xabe_cuda::arena::memory_info;
use xabe_cuda::device::{DeviceInfo, driver_available};
use xabe_engine::block::moe::MoeBlock;
use xabe_engine::forward::{Forward, Stage, arena_holds};
use xabe_engine::weights::DeviceWeights;
use xabe_gguf::GgufFile;
use xabe_model::config::{LayerKind, ModelConfig};
use xabe_model::weights::{Directory, Role, WeightSchema};

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llama.cpp/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

/// Quadro RTX 8000, sm_75: HBM-equivalent GDDR6 streaming peak.
const PEAK_GB_S: f64 = 672.0;

/// Same card, fp32 FMA peak.
const PEAK_TFLOP_S: f64 = 16.3;

/// Discarded passes before timing, matching `bench_forward`.
const WARMUP: usize = 2;

/// Grouped-GEMM tile width, the same value `Forward` builds the MoE with.
const MOE_BLOCK_SIZE: usize = 16;

/// The buckets the report aggregates into.
///
/// Ordered so the table reads in execution order rather than alphabetically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Bucket {
    Reset,
    Embed,
    GdnMixer,
    AttnMixer,
    MoeOnGdn,
    MoeOnAttn,
    FinalNorm,
    LmHead,
}

impl Bucket {
    fn of(stage: Stage, config: &ModelConfig) -> Self {
        match stage {
            Stage::Reset => Self::Reset,
            Stage::Embed => Self::Embed,
            Stage::Mixer { kind, .. } => match kind {
                LayerKind::GatedDeltaNet => Self::GdnMixer,
                LayerKind::GatedAttention => Self::AttnMixer,
            },
            Stage::Moe { layer } => match config.layer_kind(layer) {
                LayerKind::GatedDeltaNet => Self::MoeOnGdn,
                LayerKind::GatedAttention => Self::MoeOnAttn,
            },
            Stage::FinalNorm => Self::FinalNorm,
            Stage::LmHead => Self::LmHead,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Reset => "reset (30 GDN state memsets + id upload)",
            Self::Embed => "embedding gather",
            Self::GdnMixer => "GDN mixer          x30",
            Self::AttnMixer => "Gated Attn mixer   x10",
            Self::MoeOnGdn => "MoE on GDN layers  x30",
            Self::MoeOnAttn => "MoE on attn layers x10",
            Self::FinalNorm => "final RMSNorm",
            Self::LmHead => "LM head (1 position)",
        }
    }
}

/// Mean and sample standard deviation.
fn stats(samples: &[f64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    if samples.len() < 2 {
        return (mean, 0.0);
    }
    let var = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt())
}

fn env_usize_list(key: &str, default: &[usize]) -> Vec<usize> {
    match std::env::var(key) {
        Ok(v) => v
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .filter(|n| *n > 0)
            .collect(),
        Err(_) => default.to_vec(),
    }
}

/// Stored bytes and parameter count of one role on one layer.
///
/// `(0, 0)` when the role is not present, which is the right answer for a role
/// that belongs to the other layer kind.
///
/// Two numbers because the two rooflines need different ones. Bandwidth wants
/// the *stored* size, which for a Q6_K tensor is well under four bytes per
/// parameter; arithmetic wants the parameter count, because a matvec against a
/// weight matrix is exactly `2 * elements` FLOP per row of activation
/// regardless of how the matrix is stored. Both come from the GGUF directory,
/// so neither is a guess about a tensor's shape.
fn role_cost(dir: &Directory<'_>, role: Role, layer: Option<u32>) -> (u64, u64) {
    dir.find(role, layer)
        .map_or((0, 0), |e| (e.info.n_bytes, e.info.n_elements))
}

/// What one layer of each shape costs, in bytes read and in FLOP per token.
///
/// The FLOP figures cover the **projections only** — every dense matmul in the
/// pass. The delta rule, flash attention, the norms and the elementwise glue
/// are excluded, so these are lower bounds on the work. That is deliberate and
/// it is the conservative direction: `nsys` puts the projection kernels at 97%
/// of the pass, so a bound that ignores the other 3% understates the achieved
/// FLOP/s by at most that much, and every conclusion drawn from "the
/// projections are far off roofline" only gets stronger.
struct WeightBytes {
    gdn_layer: u64,
    gdn_layer_params: u64,
    attn_layer: u64,
    attn_layer_params: u64,
    /// Router, both norms, and the shared expert — read on every token.
    moe_fixed: u64,
    moe_fixed_params: u64,
    /// One routed expert's gate, up, and down slices.
    moe_per_expert: u64,
    moe_per_expert_params: u64,
    /// All 256 routed experts on one layer.
    moe_all_experts: u64,
    embedding: u64,
    lm_head: u64,
    lm_head_params: u64,
}

impl WeightBytes {
    fn measure(dir: &Directory<'_>, config: &ModelConfig) -> Self {
        const GDN_ROLES: [Role; 10] = [
            Role::InputNorm,
            Role::GdnQkv,
            Role::GdnGate,
            Role::GdnConv1d,
            Role::GdnAlpha,
            Role::GdnBeta,
            Role::GdnDtBias,
            Role::GdnA,
            Role::GdnNorm,
            Role::GdnOut,
        ];
        const ATTN_ROLES: [Role; 7] = [
            Role::InputNorm,
            Role::AttnQNorm,
            Role::AttnKNorm,
            Role::AttnQGate,
            Role::AttnK,
            Role::AttnV,
            Role::AttnOut,
        ];
        const MOE_FIXED_ROLES: [Role; 6] = [
            Role::PostMixerNorm,
            Role::MoeRouter,
            Role::MoeSharedGateInp,
            Role::MoeSharedGate,
            Role::MoeSharedUp,
            Role::MoeSharedDown,
        ];
        const MOE_EXPERT_ROLES: [Role; 3] = [Role::MoeGateExps, Role::MoeUpExps, Role::MoeDownExps];

        // Layer 0 is a Gated DeltaNet block and layer 3 a Gated Attention one;
        // every layer of a kind has the same shapes, so one of each is
        // representative. Asserted rather than assumed.
        assert_eq!(config.layer_kind(0), LayerKind::GatedDeltaNet);
        assert_eq!(config.layer_kind(3), LayerKind::GatedAttention);

        // The projections, and only the projections — the roles whose cost is
        // a matmul rather than a per-element scale.
        const GDN_PROJ: [Role; 3] = [Role::GdnQkv, Role::GdnGate, Role::GdnOut];
        const ATTN_PROJ: [Role; 4] = [Role::AttnQGate, Role::AttnK, Role::AttnV, Role::AttnOut];
        const MOE_FIXED_PROJ: [Role; 4] = [
            Role::MoeRouter,
            Role::MoeSharedGate,
            Role::MoeSharedUp,
            Role::MoeSharedDown,
        ];

        let sum = |roles: &[Role], layer: u32| -> (u64, u64) {
            roles.iter().fold((0, 0), |(b, e), &r| {
                let (rb, re) = role_cost(dir, r, Some(layer));
                (b + rb, e + re)
            })
        };
        let (all_experts, all_expert_params) = sum(&MOE_EXPERT_ROLES, 0);
        let n_experts = config.moe.num_experts as u64;
        let (embed_bytes, _) = role_cost(dir, Role::TokenEmbedding, None);
        let (head_bytes, head_params) = role_cost(dir, Role::LmHead, None);
        Self {
            gdn_layer: sum(&GDN_ROLES, 0).0,
            gdn_layer_params: sum(&GDN_PROJ, 0).1,
            attn_layer: sum(&ATTN_ROLES, 3).0,
            attn_layer_params: sum(&ATTN_PROJ, 3).1,
            moe_fixed: sum(&MOE_FIXED_ROLES, 0).0,
            moe_fixed_params: sum(&MOE_FIXED_PROJ, 0).1,
            moe_per_expert: all_experts / n_experts,
            moe_per_expert_params: all_expert_params / n_experts,
            moe_all_experts: all_experts,
            // One row per token, not the whole table.
            embedding: embed_bytes / config.vocab_size as u64,
            lm_head: head_bytes,
            lm_head_params: head_params,
        }
    }

    /// Bytes one MoE layer must read for `tokens` positions.
    ///
    /// Two bounds, because the routed traffic depends on how many distinct
    /// experts the batch selects. The low bound assumes maximum sharing (every
    /// token picks the same 8), the high bound assumes no sharing up to the
    /// 256 available.
    fn moe_layer(&self, tokens: usize, config: &ModelConfig) -> (u64, u64) {
        let per_token = config.moe.experts_per_token as u64;
        let low = self.moe_fixed + self.moe_per_expert * per_token;
        let touched = (per_token * tokens as u64).min(config.moe.num_experts as u64);
        let high = self.moe_fixed + self.moe_per_expert * touched;
        (low, high)
    }
}

/// Achieved GB/s for `bytes` moved in `ms` milliseconds.
fn gb_s(bytes: u64, ms: f64) -> f64 {
    if ms <= 0.0 {
        return 0.0;
    }
    bytes as f64 / (ms / 1e3) / 1e9
}

fn main() {
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
    let path = std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH));
    if !path.exists() {
        println!("SKIPPED: model file not found at {}", path.display());
        return;
    }

    let batches = env_usize_list("LLMXABE_PROFILE_N", &[1, 512]);
    let reps: usize = std::env::var("LLMXABE_PROFILE_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    let file = GgufFile::open(&path).expect("valid GGUF v3");
    let config = ModelConfig::qwen3_6_35b_a3b();
    let stream = ctx.default_stream();
    let (free_at_start, total) = memory_info(&ctx).expect("memory info");

    println!("device: {} ({})", info.name, info.compute_capability);
    println!(
        "model:  {} ({:.2} GiB free of {:.2} GiB)",
        path.display(),
        free_at_start as f64 / (1u64 << 30) as f64,
        total as f64 / (1u64 << 30) as f64,
    );
    println!("peak:   {PEAK_GB_S} GB/s, {PEAK_TFLOP_S} TFLOP/s fp32\n");

    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).expect("schema resolves");
    let bytes = WeightBytes::measure(&directory, &config);
    let (weights, _report) =
        DeviceWeights::load_where(&ctx, &stream, &file, &directory, arena_holds)
            .expect("weight load");

    // The byte and FLOP models below are only as good as this table, so it is
    // printed rather than trusted: every figure downstream is a sum over these
    // rows and can be checked against the file.
    println!("the projections, as the GGUF stores them:");
    println!(
        "{:<22} | {:<7} | {:>12} | {:>12} | {:>6}",
        "tensor", "type", "elements", "bytes", "B/elt"
    );
    println!(
        "{:-<22}-+-{:-<7}-+-{:-<12}-+-{:-<12}-+-{:-<6}",
        "", "", "", "", ""
    );
    for (role, layer) in [
        (Role::GdnQkv, Some(0)),
        (Role::GdnGate, Some(0)),
        (Role::GdnOut, Some(0)),
        (Role::AttnQGate, Some(3)),
        (Role::AttnK, Some(3)),
        (Role::AttnV, Some(3)),
        (Role::AttnOut, Some(3)),
        (Role::MoeRouter, Some(0)),
        (Role::MoeGateExps, Some(0)),
        (Role::MoeUpExps, Some(0)),
        (Role::MoeDownExps, Some(0)),
        (Role::MoeSharedGate, Some(0)),
        (Role::MoeSharedUp, Some(0)),
        (Role::MoeSharedDown, Some(0)),
        (Role::TokenEmbedding, None),
        (Role::LmHead, None),
    ] {
        if let Some(e) = directory.find(role, layer) {
            println!(
                "{:<22} | {:<7} | {:>12} | {:>12} | {:>6.3}",
                e.spec.name,
                e.info.ggml_type.name(),
                e.info.n_elements,
                e.info.n_bytes,
                e.info.n_bytes as f64 / e.info.n_elements as f64,
            );
        }
    }

    println!("\nweight bytes as stored in the GGUF, per layer of each shape:");
    println!("  GDN mixer         {:>12} B", bytes.gdn_layer);
    println!("  Gated Attn mixer  {:>12} B", bytes.attn_layer);
    println!(
        "  MoE fixed         {:>12} B (norms, router, shared expert)",
        bytes.moe_fixed
    );
    println!(
        "  MoE per expert    {:>12} B ({} experts = {} B)",
        bytes.moe_per_expert, config.moe.num_experts, bytes.moe_all_experts
    );
    println!("  LM head           {:>12} B\n", bytes.lm_head);

    for &n in &batches {
        let mut forward = match Forward::new(
            &ctx,
            &stream,
            &file,
            &directory,
            &weights,
            config.clone(),
            n,
        ) {
            Ok(f) => f,
            Err(e) => {
                println!("n = {n}: FAILED to build: {e}");
                continue;
            }
        };

        // Same id generator as `bench_forward`: a degenerate all-zero prompt
        // would route every position to one expert and understate the MoE.
        let ids: Vec<i32> = (0..n)
            .map(|i| ((i * 7919 + 1234) % config.vocab_size as usize) as i32)
            .collect();

        for _ in 0..WARMUP {
            forward.run(&stream, &ids, |_, _| {}).expect("warmup");
        }
        stream.synchronize().expect("sync");

        // --- uninstrumented wall clock, for the overhead figure -------------
        //
        // Measured twice, before and after the instrumented run, because at
        // one token the pass is short enough that host-side drift between two
        // adjacent measurement windows is comparable to the thing being
        // measured. Two brackets around the instrumented block make that
        // drift visible instead of letting it masquerade as overhead.
        let timed_plain = |f: &mut Forward| {
            let mut plain = Vec::with_capacity(reps);
            for _ in 0..reps {
                let t = Instant::now();
                f.run(&stream, &ids, |_, _| {}).expect("pass");
                stream.synchronize().expect("sync");
                plain.push(t.elapsed().as_secs_f64() * 1e3);
            }
            stats(&plain)
        };
        let (plain_mean, plain_sd) = timed_plain(&mut forward);

        // --- the same passes with the event markers on ----------------------
        forward
            .enable_profiling(&ctx)
            .expect("profiling events allocate");
        let mut instrumented = Vec::with_capacity(reps);
        let mut totals: BTreeMap<Bucket, Vec<f64>> = BTreeMap::new();
        let mut worst: BTreeMap<Bucket, f64> = BTreeMap::new();
        for _ in 0..reps {
            let t = Instant::now();
            forward.run(&stream, &ids, |_, _| {}).expect("pass");
            stream.synchronize().expect("sync");
            instrumented.push(t.elapsed().as_secs_f64() * 1e3);

            let spans = forward
                .profile()
                .expect("profiling on")
                .spans()
                .expect("event timeline readable");
            let mut per_bucket: BTreeMap<Bucket, f64> = BTreeMap::new();
            for (stage, ms) in spans {
                let b = Bucket::of(stage, &config);
                *per_bucket.entry(b).or_default() += ms;
                let slot = worst.entry(b).or_default();
                *slot = slot.max(ms);
            }
            for (b, ms) in per_bucket {
                totals.entry(b).or_default().push(ms);
            }
        }
        let (inst_mean, inst_sd) = stats(&instrumented);
        forward.disable_profiling();
        let (plain2_mean, plain2_sd) = timed_plain(&mut forward);
        let plain_best = plain_mean.min(plain2_mean);

        let gpu_total: f64 = totals.values().map(|v| stats(v).0).sum();

        println!("================ n = {n} tokens ================");
        println!(
            "wall clock, events off : {plain_mean:8.2} ± {plain_sd:5.2} ms  ({:.2} tok/s)",
            n as f64 / (plain_mean / 1e3),
        );
        println!(
            "wall clock, events on  : {inst_mean:8.2} ± {inst_sd:5.2} ms  \
             (instrumentation {:+.2} ms, {:+.2}%)",
            inst_mean - plain_best,
            (inst_mean - plain_best) / plain_best * 100.0,
        );
        println!(
            "wall clock, events off : {plain2_mean:8.2} ± {plain2_sd:5.2} ms  (repeated after)",
        );
        println!("GPU time, summed spans : {gpu_total:8.2} ms\n");

        println!(
            "{:<40} | {:>10} | {:>9} | {:>7} | {:>10}",
            "stage", "ms total", "ms each", "% pass", "sd ms"
        );
        println!(
            "{:-<40}-+-{:-<10}-+-{:-<9}-+-{:-<7}-+-{:-<10}",
            "", "", "", "", ""
        );
        for (&b, samples) in &totals {
            let (mean, sd) = stats(samples);
            let count = match b {
                Bucket::GdnMixer | Bucket::MoeOnGdn => 30.0,
                Bucket::AttnMixer | Bucket::MoeOnAttn => 10.0,
                _ => 1.0,
            };
            println!(
                "{:<40} | {mean:>10.3} | {:>9.3} | {:>6.2}% | {sd:>10.3}",
                b.label(),
                mean / count,
                mean / gpu_total * 100.0,
            );
        }
        println!(
            "{:-<40}-+-{:-<10}-+-{:-<9}-+-{:-<7}-+-{:-<10}",
            "", "", "", "", ""
        );
        println!(
            "{:<40} | {gpu_total:>10.3} | {:>9} | 100.00% |",
            "total", ""
        );

        // --- both rooflines, dominant stages only ---------------------------
        //
        // Every projection in this pass is a matvec against a dequantized
        // weight matrix, so its arithmetic is exactly `2 * elements` FLOP per
        // activation row and its traffic is exactly the tensor's stored size
        // per *distinct* matrix touched. Which of the two binds is the
        // question the table answers: at one token the pass reads a lot and
        // computes almost nothing, at 512 the reverse.
        let mean_of = |b: Bucket| totals.get(&b).map(|v| stats(v).0).unwrap_or(0.0);
        let (moe_low, moe_high) = bytes.moe_layer(n, &config);
        let rows = n as u64;
        let pairs = rows * config.moe.experts_per_token as u64;
        let moe_ms = mean_of(Bucket::MoeOnGdn) + mean_of(Bucket::MoeOnAttn);

        println!("\nagainst both rooflines ({PEAK_GB_S} GB/s, {PEAK_TFLOP_S} TFLOP/s fp32):");
        println!(
            "{:<34} | {:>13} | {:>8} | {:>7} | {:>13} | {:>9} | {:>7}",
            "stage", "bytes", "GB/s", "% peak", "FLOP", "GFLOP/s", "% peak",
        );
        println!(
            "{:-<34}-+-{:-<13}-+-{:-<8}-+-{:-<7}-+-{:-<13}-+-{:-<9}-+-{:-<7}",
            "", "", "", "", "", "", "",
        );
        let row = |label: &str, total_bytes: u64, flop: u64, ms: f64| {
            let achieved = gb_s(total_bytes, ms);
            let gflops = if ms > 0.0 {
                flop as f64 / (ms / 1e3) / 1e9
            } else {
                0.0
            };
            println!(
                "{label:<34} | {total_bytes:>13} | {achieved:>8.2} | {:>6.2}% | {flop:>13} | \
                 {gflops:>9.2} | {:>6.2}%",
                achieved / PEAK_GB_S * 100.0,
                gflops / (PEAK_TFLOP_S * 1e3) * 100.0,
            );
        };
        row(
            "GDN projections x30",
            bytes.gdn_layer * 30,
            2 * bytes.gdn_layer_params * rows * 30,
            mean_of(Bucket::GdnMixer),
        );
        row(
            "Gated Attn projections x10",
            bytes.attn_layer * 10,
            2 * bytes.attn_layer_params * rows * 10,
            mean_of(Bucket::AttnMixer),
        );
        row(
            "MoE x40, max expert sharing",
            moe_low * 40,
            2 * (bytes.moe_per_expert_params * pairs + bytes.moe_fixed_params * rows) * 40,
            moe_ms,
        );
        row(
            "MoE x40, no expert sharing",
            moe_high * 40,
            2 * (bytes.moe_per_expert_params * pairs + bytes.moe_fixed_params * rows) * 40,
            moe_ms,
        );
        row(
            "LM head",
            bytes.lm_head,
            2 * bytes.lm_head_params,
            mean_of(Bucket::LmHead),
        );
        row(
            "embedding gather",
            bytes.embedding * rows,
            0,
            mean_of(Bucket::Embed),
        );

        // --- how much of the grouped-GEMM grid does real work ---------------
        //
        // `moe_expert_ffn` and `moe_expert_down` launch `sorted_capacity`
        // blocks in `grid.y` — a compile-time bound, so the launch shape stays
        // replayable from a captured graph (AGENTS.md rule 5). Only the slots
        // holding a live `(token, expert)` pair do any work; the rest read two
        // integers and return. This is the fraction that matters.
        let geometry = MoeBlock::geometry_for(&config, MOE_BLOCK_SIZE, n);
        let capacity = geometry.sorted_capacity();
        println!(
            "\nMoE grouped-GEMM grid.y = sorted_capacity = {capacity} slots, of which \
             {pairs} carry a live (token, expert) pair",
        );
        println!(
            "  {:.2}% of blocks do work, {:.2}% early-out; a perfectly packed grid \
             would be {:.2}x smaller",
            pairs as f64 / capacity as f64 * 100.0,
            (capacity as f64 - pairs as f64) / capacity as f64 * 100.0,
            capacity as f64 / pairs as f64,
        );

        // --- worst single instance, to expose per-layer spread --------------
        println!("\nslowest single instance of each repeated stage:");
        for b in [
            Bucket::GdnMixer,
            Bucket::AttnMixer,
            Bucket::MoeOnGdn,
            Bucket::MoeOnAttn,
        ] {
            if let Some(&ms) = worst.get(&b) {
                println!("  {:<38} {ms:>8.3} ms", b.label());
            }
        }
        println!();
    }

    println!(
        "{WARMUP} warmup passes discarded, {reps} timed repetitions. Stage times are CUDA \
         event deltas on the pass's own stream; the pass total is host wall clock with the \
         stream synchronized inside the timed region."
    );
}
