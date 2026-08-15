//! `llmxabe` — engine preflight.
//!
//! ```sh
//! cargo run -p xabe-server
//! ```
//!
//! There is no HTTP surface yet, and this binary does not serve requests. What
//! it does is run the startup path the engine will need regardless: validate
//! the model configuration, check the device fleet against the sm_75 gate,
//! derive the VRAM and bandwidth budgets, construct the two-group cache
//! geometry, and construct the scheduler — which is where the token-budget
//! rule is enforced.
//!
//! That last part is the point. Three of the project's design rules are
//! enforced by construction (`SchedulerConfig::new` rejects a budget at or
//! below the block size; `CacheConfig::new` rejects a misaligned retention
//! interval; the device gate rejects a heterogeneous fleet), so a preflight
//! that successfully builds these types has checked them.

use tracing::{error, info, warn};
use xabe_cache::config::CacheConfig;
use xabe_cuda::{check_gate, device};
use xabe_engine::{Engine, RouterConfig};
use xabe_model::budget;
use xabe_model::{ModelConfig, verify};
use xabe_sched::config::SchedulerConfig;

/// Per-step token budget. Must exceed `block_size + max_concurrent_decodes`.
const TOKEN_BUDGET: u32 = 4096;
/// Concurrent slots per worker, matching the llama.cpp baseline's `-np 3`.
const SLOTS_PER_WORKER: u32 = 3;
/// Total context across slots, matching the baseline's `-c 393216`.
const TOTAL_CONTEXT: u32 = 393_216;
/// f16 KV cache, matching the baseline's `-ctk f16 -ctv f16`.
const KV_ELEM_BYTES_F16: u64 = 2;
/// Measured tensor-data size of `Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf`.
const WEIGHTS_BYTES: u64 = (296 * (1024 * 1024 * 1024)) / 10;

/// Bytes as GiB, for display.
fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn main() -> std::process::ExitCode {
    xabe_log::init_from_args();

    info!("llmxabe preflight\n");

    // 1. Model configuration.
    let model = ModelConfig::qwen3_6_35b_a3b();
    match verify::check_config(&model) {
        Ok(()) => info!("model            {} — config self-consistent", model.name),
        Err(e) => {
            error!("model            FAIL — {e}");
            return std::process::ExitCode::FAILURE;
        }
    }
    info!(
        "                 {} layers ({} attention, {} GDN), {} experts, vocab {}",
        model.num_layers,
        model.num_attention_layers(),
        model.num_gdn_layers(),
        model.moe.num_experts,
        model.vocab_size
    );

    // 2. Cache geometry. Construction enforces that the retention interval is
    //    block-aligned; the two page sizes are never unified.
    let cache = match CacheConfig::with_defaults(model.clone()) {
        Ok(c) => c,
        Err(e) => {
            error!("cache            FAIL — {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    info!(
        "\ncache            attention block {} tokens → {:.2} MiB/page",
        cache.attention_block_size(),
        cache.attention_page_bytes() as f64 / (1024.0 * 1024.0)
    );
    info!(
        "                 GDN retention R = {} tokens → {:.2} MiB/snapshot",
        cache.gdn_retention_interval(),
        cache.gdn_page_bytes() as f64 / (1024.0 * 1024.0)
    );
    info!(
        "                 snapshot:KV ratio {:.2} over one retention interval",
        cache.snapshot_to_kv_ratio()
    );
    info!("                 (capacity is reported per group and never summed)");

    // 3. Scheduler. Construction rejects the budget-versus-block trap.
    let sched = match SchedulerConfig::with_defaults(
        TOKEN_BUDGET,
        cache.attention_block_size(),
        SLOTS_PER_WORKER,
    ) {
        Ok(s) => s,
        Err(e) => {
            error!("\nscheduler        FAIL — {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    info!(
        "\nscheduler        token budget {} > block {} + decodes {} — accepted",
        sched.token_budget(),
        sched.block_size(),
        sched.max_concurrent_decodes()
    );
    info!(
        "                 {} tokens charged per decode step ({} MTP drafts)",
        sched.tokens_per_decode_step(),
        sched.draft_tokens_per_step()
    );

    // 4. Devices. The only part that can be skipped — and it must be checked
    //    before the VRAM budget, so headroom is computed against this card's
    //    measured memory rather than a hardcoded assumption.
    info!("");
    if !device::driver_available() {
        warn!("devices          SKIPPED — no CUDA driver on this host");
        warn!("\nPreflight incomplete: host-side configuration is valid, but the");
        info!("device fleet was not checked. This is not a passing preflight.");
        return std::process::ExitCode::FAILURE;
    }

    let devices = match device::probe_all() {
        Ok(d) => d,
        Err(e) => {
            error!("devices          FAIL — {e:?}");
            return std::process::ExitCode::FAILURE;
        }
    };
    if let Err(e) = check_gate(&devices) {
        error!("devices          FAIL — {e}");
        return std::process::ExitCode::FAILURE;
    }
    info!(
        "devices          {} × {} (compute {}, {:.0} GB/s each)",
        devices.len(),
        devices[0].name,
        devices[0].compute_capability,
        devices[0].peak_bandwidth_gb_s()
    );

    // 5. VRAM budget, against this card's measured memory.
    let vram = budget::vram_budget(
        &model,
        u64::from(TOTAL_CONTEXT),
        SLOTS_PER_WORKER,
        KV_ELEM_BYTES_F16,
        WEIGHTS_BYTES,
    );
    let usable = devices[0].total_memory;
    let headroom = vram.headroom_bytes(usable);
    info!(
        "\nvram             {:.2} GiB of {:.2} GiB measured — {:.2} GiB headroom",
        gib(vram.total_bytes()),
        gib(usable),
        headroom as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    info!("                 (text-only; excludes the ~1.5 GiB vision encoder)");
    if headroom < 0 {
        error!("\nPreflight failed: this configuration does not fit in VRAM.");
        return std::process::ExitCode::FAILURE;
    }

    // 6. Engine. One worker per device, sharing one prefix tree.
    let attention_blocks = TOTAL_CONTEXT / cache.attention_block_size();
    let ordinals: Vec<usize> = devices.iter().map(|d| d.ordinal).collect();
    let engine = Engine::new(
        &ordinals,
        cache,
        sched,
        attention_blocks,
        SLOTS_PER_WORKER,
        RouterConfig::balanced(),
    );
    info!(
        "\nengine           {} workers, {} attention blocks each, one shared prefix tree",
        engine.worker_count(),
        attention_blocks
    );

    info!("\nPreflight passed. No HTTP surface yet — this engine cannot serve");
    info!("requests. See README.md for what is and is not implemented.");
    std::process::ExitCode::SUCCESS
}
