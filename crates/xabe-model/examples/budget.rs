//! Prints the VRAM segmentation table and the per-context bandwidth
//! roofline table for the target model and hardware.
//!
//! `cargo run -p xabe-model --example budget`
//!
//! Every number here is either a closed-form derivation from
//! [`xabe_model::config::ModelConfig`] or an explicitly labeled estimate —
//! see `crates/xabe-model/src/budget.rs`'s module doc for which is which,
//! and for where this diverges from `qwen36-rust-engine-plan.md`'s own
//! reference tables (and why).

use xabe_model::budget::{self, WeightBytesPerParam};
use xabe_model::config::ModelConfig;

/// 1 GiB in bytes.
const GIB: u64 = 1024 * 1024 * 1024;

/// Quadro RTX 8000: 48 GiB total, ~47.5 GiB usable after driver/display
/// reservation (`qwen36-rust-engine-plan.md` §03).
const USABLE_VRAM_BYTES: u64 = (475 * GIB) / 10;

/// Measured on-disk size of `Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf`'s tensor data
/// (`qwen36-rust-engine-plan.md` §03's "measured"-confidence figure).
const WEIGHTS_BYTES: u64 = (296 * GIB) / 10;

/// `-c 393216 -np 3` from the plan's llama.cpp reference invocation (§03).
const REFERENCE_CONTEXT_TOKENS: u64 = 393_216;
const REFERENCE_SLOTS: u32 = 3;

/// f16 KV cache (`-ctk f16 -ctv f16`).
const KV_ELEM_BYTES_F16: u64 = 2;

/// Quadro RTX 8000 memory bandwidth.
const RTX_8000_BANDWIDTH_BYTES_PER_SEC: f64 = 672e9;

fn gib(bytes: u64) -> f64 {
    bytes as f64 / GIB as f64
}

fn main() {
    let cfg = ModelConfig::qwen3_6_35b_a3b();

    println!("=== VRAM budget per card ({}) ===", cfg.name);
    println!(
        "context={REFERENCE_CONTEXT_TOKENS} slots={REFERENCE_SLOTS} kv=f16 weights={:.1} GiB\n",
        gib(WEIGHTS_BYTES)
    );

    let vram = budget::vram_budget(
        &cfg,
        REFERENCE_CONTEXT_TOKENS,
        REFERENCE_SLOTS,
        KV_ELEM_BYTES_F16,
        WEIGHTS_BYTES,
    );

    println!("{:<32} {:>10}", "Segment", "GiB");
    println!("{:<32} {:>10.2}", "Weights", gib(vram.weights_bytes));
    println!("{:<32} {:>10.2}", "KV pool", gib(vram.kv_pool_bytes));
    println!(
        "{:<32} {:>10.2}",
        "GDN recurrent state (x slots)",
        gib(vram.gdn_state_bytes)
    );
    println!(
        "{:<32} {:>10.2}",
        "Compute buffers (estimate)",
        gib(vram.compute_buffer_bytes)
    );
    println!(
        "{:<32} {:>10.2}",
        "CUDA context overhead (estimate)",
        gib(vram.cuda_context_overhead_bytes)
    );
    println!("{:<32} {:>10.2}", "Total", gib(vram.total_bytes()));
    println!(
        "{:<32} {:>10.2}",
        "Headroom (of usable)",
        vram.headroom_bytes(USABLE_VRAM_BYTES) as f64 / GIB as f64
    );
    println!(
        "\nNote: excludes the vision encoder (mmproj) — this engine is \
         text-only per AGENTS.md, so the total below runs lower than the \
         planning document's own ~41.3 GiB figure by roughly the vision \
         encoder's ~1.5 GiB, not by error.\n"
    );

    println!("=== Per-token decode bandwidth roofline ===");
    println!(
        "quantization: expert weights Q6_K, LM head + projections Q8_0 \
         (observed directly in the real GGUF file, tensor by tensor)\n"
    );
    println!(
        "{:<8} {:>12} {:>12} {:>12} {:>14}",
        "Context", "KV GB/tok", "Wt GB/tok", "Total GB/tok", "Roofline tok/s"
    );

    let bpp = WeightBytesPerParam::observed_in_target_file();
    for (label, ctx) in [
        ("4K", 4_096u64),
        ("32K", 32_768),
        ("128K", 131_072),
        ("256K", 262_144),
    ] {
        let bw = budget::decode_bandwidth(
            &cfg,
            ctx,
            KV_ELEM_BYTES_F16,
            bpp,
            RTX_8000_BANDWIDTH_BYTES_PER_SEC,
        );
        println!(
            "{:<8} {:>12.3} {:>12.3} {:>12.3} {:>14.1}",
            label,
            bw.kv_bytes as f64 / 1e9,
            bw.weight_bytes as f64 / 1e9,
            bw.total_bytes as f64 / 1e9,
            bw.roofline_tokens_per_sec
        );
    }

    let weight_bytes = {
        let bw = budget::decode_bandwidth(
            &cfg,
            0,
            KV_ELEM_BYTES_F16,
            bpp,
            RTX_8000_BANDWIDTH_BYTES_PER_SEC,
        );
        bw.weight_bytes
    };
    let crossover = budget::kv_bound_crossover_tokens(&cfg, KV_ELEM_BYTES_F16, weight_bytes);
    println!(
        "\nKV-bound crossover: {crossover} tokens (weight bytes/token exceeded \
         by KV bytes/token past this point)."
    );
    println!(
        "This differs from qwen36-rust-engine-plan.md \u{a7}04's ~109K crossover \
         estimate — see WeightBytesPerParam::observed_in_target_file's doc \
         comment for the derivation gap (projection_params is structurally \
         larger, and observed as Q8_0 rather than Q6_K, versus that plan's \
         rougher estimate)."
    );
}
