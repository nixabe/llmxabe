//! `llmxabe` — engine preflight and HTTP server.
//!
//! ```sh
//! cargo run -p xabe-server -- --help
//! ```
//!
//! Startup first runs the preflight: validate the model configuration, check
//! the device fleet against the sm_75 gate, derive the VRAM and bandwidth
//! budgets, construct the two-group cache geometry, and construct the
//! scheduler — which is where the token-budget rule is enforced. Only then
//! does the HTTP surface come up.
//!
//! That ordering is the point. Three of the project's design rules are
//! enforced by construction (`SchedulerConfig::new` rejects a budget at or
//! below the block size; `CacheConfig::new` rejects a misaligned retention
//! interval; the device gate rejects a heterogeneous fleet), so command-line
//! arguments feed those constructors directly and an invalid combination
//! fails preflight instead of serving. The arguments are documented in
//! `docs/CLI.md`.

mod http;
mod size;
mod tokenizer;

use clap::Parser;
use std::path::PathBuf;
use tracing::{error, info, warn};
use xabe_cache::config::CacheConfig;
use xabe_cuda::{check_gate, device};
use xabe_engine::{
    DEFAULT_SNAPSHOT_SLOTS_PER_WORKER, Engine, RouterConfig, ServingConfig, snapshot_bytes_per_slot,
};
use xabe_model::budget;
use xabe_model::{ModelConfig, verify};
use xabe_sched::config::SchedulerConfig;

/// f16 KV cache, matching the baseline's `-ctk f16 -ctv f16`.
const KV_ELEM_BYTES_F16: u64 = 2;
/// Measured tensor-data size of `Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf`.
const WEIGHTS_BYTES: u64 = (296 * (1024 * 1024 * 1024)) / 10;
const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llama.cpp/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

/// `--help` addendum for the flag clap never sees (see [`Args`] docs).
fn log_flag_help() -> String {
    format!("Logging:\n{}", xabe_log::FLAG_HELP)
}

/// Three-worker CUDA inference engine for Qwen3.6-35B-A3B.
//
// `--log-level` is absent on purpose: `xabe_log::init_from_args` strips it
// from the argument list before clap runs, so every binary in the workspace
// parses it identically. It is appended to `--help` via `log_flag_help`.
#[derive(Parser)]
#[command(
    name = "llmxabe",
    version,
    no_binary_name = true,
    after_help = log_flag_help(),
)]
struct Args {
    /// Path to the GGUF model file
    #[arg(short, long, env = "LLMXABE_MODEL", default_value = DEFAULT_MODEL_PATH)]
    model: PathBuf,

    /// Host the HTTP server binds
    #[arg(long, env = "LLMXABE_HOST", default_value = "127.0.0.1")]
    host: String,

    /// Port the HTTP server binds
    #[arg(long, env = "LLMXABE_PORT", default_value_t = 8000)]
    port: u16,

    /// Per-step token budget (also -tb); must exceed block_size + max_concurrent_decodes
    #[arg(long, default_value_t = 4096)]
    token_budget: u32,

    /// Concurrent slots per worker, matching the llama.cpp baseline's -np 3
    #[arg(short, long, default_value_t = 3)]
    slots_per_worker: u32,

    /// Total context across slots, matching the baseline's -c 393216
    #[arg(short = 'c', long, default_value_t = 393_216)]
    total_context: u32,

    /// Prefill chunk size in tokens (also -pc)
    #[arg(long, default_value_t = 4096)]
    prefill_chunk: usize,

    /// Host RAM for the prefix cache's pinned snapshots, across all workers
    /// (e.g. 8GiB); 0 disables snapshot retention and prefix sharing
    #[arg(long, env = "LLMXABE_CACHE_RAM", value_parser = size::parse_bytes)]
    cache_ram: Option<u64>,

    /// Fraction of the KV pool held back as admission headroom, in [0, 1)
    #[arg(long, default_value_t = xabe_sched::config::DEFAULT_WATERMARK_FRACTION)]
    watermark: f64,

    /// API key callers must present; with none set, every caller is accepted
    #[arg(long, env = "LLMXABE_API_KEY", hide_env_values = true)]
    api_key: Option<String>,

    /// Model name reported by /v1/models and echoed in responses
    #[arg(long, env = "LLMXABE_SERVED_MODEL_NAME", default_value = http::DEFAULT_MODEL)]
    served_model_name: String,

    /// Output token limit for requests that do not set one
    #[arg(long, default_value_t = 16)]
    default_max_tokens: u32,

    /// Answer without extended thinking unless a request asks for it
    #[arg(long)]
    no_reasoning: bool,
}

/// Rewrite the two-letter shorts clap cannot express (`-pc`, `-tb`) into
/// their long forms before parsing. Both bare and `=value` forms are handled.
fn expand_two_letter_shorts(args: Vec<String>) -> Vec<String> {
    args.into_iter()
        .map(|arg| {
            for (short, long) in [("-pc", "--prefill-chunk"), ("-tb", "--token-budget")] {
                if arg == short {
                    return long.to_owned();
                }
                if let Some(value) = arg.strip_prefix(short)
                    && let Some(value) = value.strip_prefix('=')
                {
                    return format!("{long}={value}");
                }
            }
            arg
        })
        .collect()
}

fn main() -> std::process::ExitCode {
    let rest = xabe_log::init_from_args();
    let args = Args::parse_from(expand_two_letter_shorts(rest));

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
    let sched = match SchedulerConfig::new(
        args.token_budget,
        cache.attention_block_size(),
        args.slots_per_worker,
        args.watermark,
        xabe_sched::config::DEFAULT_DRAFT_TOKENS_PER_STEP,
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
        "                 {} tokens charged per decode step ({} n-gram drafts)",
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
        u64::from(args.total_context),
        args.slots_per_worker,
        KV_ELEM_BYTES_F16,
        WEIGHTS_BYTES,
    );
    let usable = devices[0].total_memory;
    let headroom = vram.headroom_bytes(usable);
    info!(
        "\nvram             {:.2} GiB of {:.2} GiB measured — {:.2} GiB headroom",
        size::gib(vram.total_bytes()),
        size::gib(usable),
        headroom as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    info!("                 (text-only; excludes the ~1.5 GiB vision encoder)");
    if headroom < 0 {
        error!("\nPreflight failed: this configuration does not fit in VRAM.");
        return std::process::ExitCode::FAILURE;
    }

    // 6. Engine. One worker per device, sharing one prefix tree.
    let attention_blocks = args.total_context / cache.attention_block_size();
    let retention_interval = cache.gdn_retention_interval() as usize;
    let ordinals: Vec<usize> = devices.iter().map(|d| d.ordinal).collect();
    let mut engine = Engine::new(
        &ordinals,
        cache,
        sched,
        attention_blocks,
        args.slots_per_worker,
        RouterConfig::balanced(),
    );
    info!(
        "\nengine           {} workers, {} attention blocks each, one shared prefix tree",
        engine.worker_count(),
        attention_blocks
    );

    // 7. Prefix-cache RAM. Sized here, before anything is allocated, so the
    //    preflight can say what the budget actually bought — a slot count is
    //    the quantity that matters and a byte figure is what was asked for.
    let bytes_per_slot = snapshot_bytes_per_slot(&model, retention_interval) as u64;
    let slots_per_worker = match args.cache_ram {
        Some(budget) => budget / engine.worker_count().max(1) as u64 / bytes_per_slot.max(1),
        None => DEFAULT_SNAPSHOT_SLOTS_PER_WORKER as u64,
    } as usize;
    let cache_ram = slots_per_worker as u64 * bytes_per_slot * engine.worker_count() as u64;
    info!(
        "\ncache ram        {:.2} GiB pinned — {} snapshots per worker at {:.2} MiB each",
        size::gib(cache_ram),
        slots_per_worker,
        bytes_per_slot as f64 / (1024.0 * 1024.0)
    );
    if slots_per_worker == 0 {
        warn!("                 no snapshots retained — prefix sharing is off");
    } else if slots_per_worker < args.slots_per_worker as usize {
        // The engine keeps one slot per concurrent sequence free before it
        // will publish anything, so below that line it publishes nothing at
        // all — and whichever sequence loses the race for the remaining slots
        // stops retaining for the rest of its life.
        warn!(
            "                 fewer snapshots than the {} concurrent sequences per worker — \
             nothing will be shared, and retention will switch off for whichever sequence \
             loses the race",
            args.slots_per_worker
        );
    }

    let model_path = args.model;
    info!("\nloading           {}", model_path.display());
    let serving = ServingConfig {
        prefill_chunk: args.prefill_chunk,
        snapshot_slots: slots_per_worker,
    };
    if let Err((worker, failure)) = engine.bind_devices(&model_path, model, serving) {
        error!("worker {worker} failed to load: {failure}");
        return std::process::ExitCode::FAILURE;
    }
    let tokenizer = match tokenizer::from_gguf(&model_path) {
        Ok(tokenizer) => tokenizer,
        Err(failure) => {
            error!("tokenizer        FAIL — {failure}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let address = format!("{}:{}", args.host, args.port);
    let runtime = match tokio::runtime::Runtime::new() {
        Ok(runtime) => runtime,
        Err(failure) => {
            error!("runtime          FAIL — {failure}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let server = http::ServerConfig {
        api_key: args.api_key,
        model: args.served_model_name,
        default_max_tokens: args.default_max_tokens,
        default_reasoning: !args.no_reasoning,
    };
    match runtime.block_on(http::serve(engine, tokenizer, &address, server)) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(failure) => {
            error!("server           FAIL — {failure}");
            std::process::ExitCode::FAILURE
        }
    }
}
