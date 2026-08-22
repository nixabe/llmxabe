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

use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use tracing::{error, info, warn};
use xabe_cache::config::CacheConfig;
use xabe_cuda::{check_gate, device};
use xabe_engine::{
    DEFAULT_SNAPSHOT_SLOTS_PER_WORKER, Engine, RouterConfig, ServingConfig, Speculation,
    snapshot_bytes_per_slot,
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

/// Which speculative decoder to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SpecType {
    /// One token per decode step.
    None,
    /// Suffix lookup over the sequence's own prompt and output.
    Ngram,
    /// llama.cpp's ngram-simple: backward scan for a fixed-length tail.
    NgramSimple,
    /// llama.cpp's ngram-mod: worker-shared n-gram-to-next-token table.
    NgramMod,
    /// llama.cpp's ngram-map-k: key-n-gram map, drafts the newest match.
    NgramMapK,
    /// llama.cpp's ngram-map-k4v: as ngram-map-k, tracking up to four
    /// continuations per key and drafting only a dominant one.
    NgramMapK4v,
    /// The model's own multi-token-prediction head.
    DraftMtp,
    /// A trained DFlash drafter (block in-fill; needs --spec-dflash).
    SpecDflash,
}

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

    /// Path to the multimodal projector GGUF (the `mmproj-*.gguf` shipped
    /// beside the model); enables image input
    #[arg(long, env = "LLMXABE_MMPROJ")]
    mmproj: Option<PathBuf>,

    /// Path to the DFlash drafter GGUF; required by --spec-type spec-dflash
    #[arg(long, env = "LLMXABE_DFLASH")]
    spec_dflash: Option<PathBuf>,

    /// Most language-model tokens one image may occupy; larger images are
    /// resized down to fit
    #[arg(long, default_value_t = 1024)]
    image_max_tokens: u32,

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

    /// Speculative decoder to run
    #[arg(long, value_enum, default_value_t = SpecType::None)]
    spec_type: SpecType,

    /// draft-mtp/spec-dflash: maximum tokens the drafter proposes per step
    #[arg(long, default_value_t = xabe_sched::config::DEFAULT_DRAFT_TOKENS_PER_STEP)]
    spec_draft_n_max: u32,

    /// Drop any draft that comes out shorter than this (0 keeps every draft)
    #[arg(long, default_value_t = 0)]
    spec_draft_n_min: u32,

    /// draft-mtp/spec-dflash: stop drafting at the first token whose
    /// probability under the drafter's own head falls below this (0 disables)
    #[arg(long, default_value_t = 0.0)]
    spec_draft_p_min: f32,

    /// Accepted for llama.cpp flag compatibility; no current speculative
    /// decoder uses a split probability (llama.cpp's ignore it too)
    #[arg(long, default_value_t = 0.1)]
    spec_draft_p_split: f32,

    /// ngram: maximum tokens proposed from a suffix match per step
    #[arg(long, default_value_t = xabe_sched::config::DEFAULT_DRAFT_TOKENS_PER_STEP)]
    spec_ngram_n_max: u32,

    /// ngram: shortest suffix worth matching on
    #[arg(long, default_value_t = 2)]
    spec_ngram_min: usize,

    /// ngram: longest suffix matched before giving up
    #[arg(long, default_value_t = 4)]
    spec_ngram_max: usize,

    /// ngram-simple: ngram size N, length of the lookup n-gram
    #[arg(long, default_value_t = 12)]
    spec_ngram_simple_size_n: u32,

    /// ngram-simple: ngram size M, length of the draft m-gram
    #[arg(long, default_value_t = 48)]
    spec_ngram_simple_size_m: u32,

    /// Accepted for llama.cpp flag compatibility; ngram-simple keeps no
    /// hit statistics (llama.cpp's ignores it too)
    #[arg(long, default_value_t = 1)]
    spec_ngram_simple_min_hits: u32,

    /// ngram-mod: lookup n-gram length
    #[arg(long, default_value_t = 24)]
    spec_ngram_mod_n_match: u32,

    /// ngram-mod: minimum number of ngram tokens to draft; a chain that
    /// breaks earlier is dropped
    #[arg(long, default_value_t = 48)]
    spec_ngram_mod_n_min: u32,

    /// ngram-mod: maximum number of ngram tokens to draft per step
    #[arg(long, default_value_t = 64)]
    spec_ngram_mod_n_max: u32,

    /// ngram-map-k: ngram size N, length of the lookup n-gram
    #[arg(long, default_value_t = 12)]
    spec_ngram_map_k_size_n: u32,

    /// ngram-map-k: ngram size M, length of the draft m-gram
    #[arg(long, default_value_t = 48)]
    spec_ngram_map_k_size_m: u32,

    /// ngram-map-k: minimum hits at ngram lookup for a draft; kept for
    /// llama.cpp flag parity (its key-only draft path ignores it too)
    #[arg(long, default_value_t = 1)]
    spec_ngram_map_k_min_hits: u32,

    /// ngram-map-k4v: ngram size N, length of the lookup n-gram
    #[arg(long, default_value_t = 12)]
    spec_ngram_map_k4v_size_n: u32,

    /// ngram-map-k4v: ngram size M, length of the draft m-gram
    #[arg(long, default_value_t = 48)]
    spec_ngram_map_k4v_size_m: u32,

    /// ngram-map-k4v: minimum hits at ngram lookup for a draft
    #[arg(long, default_value_t = 1)]
    spec_ngram_map_k4v_min_hits: u32,

    /// Fraction of the KV pool held back as admission headroom, in [0, 1)
    #[arg(long, default_value_t = xabe_sched::config::DEFAULT_WATERMARK_FRACTION)]
    watermark: f64,

    /// API key callers must present; with none set, every caller is accepted
    #[arg(long, env = "LLMXABE_API_KEY", hide_env_values = true)]
    api_key: Option<String>,

    /// Model name reported by /v1/models and echoed in responses
    #[arg(short, long, env = "LLMXABE_SERVED_MODEL_NAME", default_value = http::DEFAULT_MODEL)]
    alias: String,

    /// Output token limit for requests that do not set one
    #[arg(long, default_value_t = 16)]
    max_tokens: u32,

    /// Answer without extended thinking unless a request asks for it
    #[arg(long)]
    no_reasoning: bool,

    /// Sampling temperature for requests that do not set one; 0 is greedy
    #[arg(long, visible_alias = "temp", default_value_t = 1.0)]
    temperature: f32,

    /// Nucleus cutoff for requests that do not set one; 1 disables it
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,

    /// Keep only tokens at least this likely relative to the most likely
    /// token, for requests that do not set it; 0 disables it
    #[arg(long, default_value_t = 0.0)]
    min_p: f32,
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

/// llama.cpp's range for every `--spec-ngram-*` size argument.
fn ngram_size_in_range(value: u32, flag: &str) -> Result<(), String> {
    if value == 0 || value > 1024 {
        return Err(format!(
            "{flag} {value} must be between 1 and 1024 inclusive"
        ));
    }
    Ok(())
}

/// llama.cpp's bound for every `--spec-ngram-*-min-hits` argument.
fn min_hits_at_least_one(value: u32, flag: &str) -> Result<(), String> {
    if value == 0 || value > u32::from(u16::MAX) {
        return Err(format!("{flag} {value} must be between 1 and 65535"));
    }
    Ok(())
}

/// Resolve `--spec-type` and its family into the draft count the scheduler
/// must budget for and the decoder a worker will run.
fn resolve_speculation(args: &Args) -> Result<(u32, Speculation), String> {
    if args.spec_draft_n_min > args.spec_draft_n_max {
        return Err(format!(
            "--spec-draft-n-min {} exceeds --spec-draft-n-max {}: every draft would be dropped",
            args.spec_draft_n_min, args.spec_draft_n_max
        ));
    }
    if !(0.0..1.0).contains(&args.spec_draft_p_min) {
        return Err(format!(
            "--spec-draft-p-min {} must be in [0, 1)",
            args.spec_draft_p_min
        ));
    }
    if !(0.0..=1.0).contains(&args.spec_draft_p_split) {
        return Err(format!(
            "--spec-draft-p-split {} must be in [0, 1]",
            args.spec_draft_p_split
        ));
    }
    match args.spec_type {
        SpecType::None => Ok((0, Speculation::None)),
        SpecType::Ngram => {
            if args.spec_ngram_min == 0 || args.spec_ngram_min > args.spec_ngram_max {
                return Err(format!(
                    "--spec-ngram-min {} and --spec-ngram-max {} must satisfy 0 < min <= max",
                    args.spec_ngram_min, args.spec_ngram_max
                ));
            }
            if args.spec_ngram_max >= args.total_context as usize {
                return Err(format!(
                    "--spec-ngram-max {} must be shorter than the {} token context it \
                     matches within",
                    args.spec_ngram_max, args.total_context
                ));
            }
            Ok((
                args.spec_ngram_n_max,
                Speculation::Ngram {
                    min: args.spec_ngram_min,
                    max: args.spec_ngram_max,
                },
            ))
        }
        SpecType::NgramSimple => {
            ngram_size_in_range(args.spec_ngram_simple_size_n, "--spec-ngram-simple-size-n")?;
            ngram_size_in_range(args.spec_ngram_simple_size_m, "--spec-ngram-simple-size-m")?;
            min_hits_at_least_one(
                args.spec_ngram_simple_min_hits,
                "--spec-ngram-simple-min-hits",
            )?;
            Ok((
                args.spec_ngram_simple_size_m,
                Speculation::NgramSimple {
                    size_n: args.spec_ngram_simple_size_n as usize,
                },
            ))
        }
        SpecType::NgramMod => {
            ngram_size_in_range(args.spec_ngram_mod_n_match, "--spec-ngram-mod-n-match")?;
            // llama.cpp allows 0 for both (and n-min > n-max): n-max 0 simply
            // never drafts, which --spec-type none states outright.
            if args.spec_ngram_mod_n_min > 1024 {
                return Err(format!(
                    "--spec-ngram-mod-n-min {} must be between 0 and 1024 inclusive",
                    args.spec_ngram_mod_n_min
                ));
            }
            if args.spec_ngram_mod_n_max == 0 || args.spec_ngram_mod_n_max > 1024 {
                return Err(format!(
                    "--spec-ngram-mod-n-max {} must be between 1 and 1024 inclusive \
                     (0 drafts nothing; use --spec-type none instead)",
                    args.spec_ngram_mod_n_max
                ));
            }
            Ok((
                args.spec_ngram_mod_n_max,
                Speculation::NgramMod {
                    n_match: args.spec_ngram_mod_n_match as usize,
                    n_min: args.spec_ngram_mod_n_min as usize,
                },
            ))
        }
        SpecType::NgramMapK => {
            ngram_size_in_range(args.spec_ngram_map_k_size_n, "--spec-ngram-map-k-size-n")?;
            ngram_size_in_range(args.spec_ngram_map_k_size_m, "--spec-ngram-map-k-size-m")?;
            min_hits_at_least_one(
                args.spec_ngram_map_k_min_hits,
                "--spec-ngram-map-k-min-hits",
            )?;
            Ok((
                args.spec_ngram_map_k_size_m,
                Speculation::NgramMapK {
                    size_n: args.spec_ngram_map_k_size_n as usize,
                    min_hits: args.spec_ngram_map_k_min_hits as u16,
                },
            ))
        }
        SpecType::NgramMapK4v => {
            ngram_size_in_range(
                args.spec_ngram_map_k4v_size_n,
                "--spec-ngram-map-k4v-size-n",
            )?;
            ngram_size_in_range(
                args.spec_ngram_map_k4v_size_m,
                "--spec-ngram-map-k4v-size-m",
            )?;
            min_hits_at_least_one(
                args.spec_ngram_map_k4v_min_hits,
                "--spec-ngram-map-k4v-min-hits",
            )?;
            Ok((
                args.spec_ngram_map_k4v_size_m,
                Speculation::NgramMapK4v {
                    size_n: args.spec_ngram_map_k4v_size_n as usize,
                    min_hits: args.spec_ngram_map_k4v_min_hits as u16,
                },
            ))
        }
        SpecType::DraftMtp => {
            if args.spec_draft_n_max == 0 {
                return Err(
                    "--spec-draft-n-max 0 asks the draft head to draft nothing; \
                     use --spec-type none instead"
                        .to_owned(),
                );
            }
            Ok((args.spec_draft_n_max, Speculation::Mtp))
        }
        SpecType::SpecDflash => {
            if args.spec_draft_n_max == 0 {
                return Err("--spec-draft-n-max 0 asks the drafter to draft nothing; \
                     use --spec-type none instead"
                    .to_owned());
            }
            let Some(path) = &args.spec_dflash else {
                return Err(
                    "--spec-type dflash needs --spec-dflash <drafter.gguf>                      (the trained DFlash checkpoint)"
                        .to_owned(),
                );
            };
            if !path.is_file() {
                return Err(format!("--spec-dflash {} is not a file", path.display()));
            }
            Ok((args.spec_draft_n_max, Speculation::DFlash))
        }
    }
}

fn main() -> std::process::ExitCode {
    let rest = xabe_log::init_from_args();
    let args = Args::parse_from(expand_two_letter_shorts(rest));

    info!("llmxabe preflight\n");

    let (draft_tokens, speculation) = match resolve_speculation(&args) {
        Ok(resolved) => resolved,
        Err(failure) => {
            error!("speculation      FAIL — {failure}");
            return std::process::ExitCode::FAILURE;
        }
    };

    // The sampling defaults stand in for per-request values, which are
    // range-checked at request time; a default outside that range would turn
    // every silent request into a 400 at serve time. Fail preflight instead.
    for (flag, value, high) in [
        ("--temperature", args.temperature, 2.0f32),
        ("--top-p", args.top_p, 1.0),
        ("--min-p", args.min_p, 1.0),
    ] {
        if !value.is_finite() || !(0.0..=high).contains(&value) {
            error!("sampling         FAIL — {flag} {value} must be between 0 and {high}");
            return std::process::ExitCode::FAILURE;
        }
    }

    // Vision serving. The projector file is only opened at worker bind; what
    // preflight can check is that the path exists and the token ceiling is
    // inside the model's own budget, so a typo fails here with the flag's
    // name on it.
    use xabe_kernels::vision::preprocess::{MAX_IMAGE_TOKENS, MIN_IMAGE_TOKENS};
    if let Some(mmproj) = &args.mmproj
        && !mmproj.is_file()
    {
        error!(
            "vision           FAIL — --mmproj {} is not a file",
            mmproj.display()
        );
        return std::process::ExitCode::FAILURE;
    }
    if !(MIN_IMAGE_TOKENS..=MAX_IMAGE_TOKENS).contains(&args.image_max_tokens) {
        error!(
            "vision           FAIL — --image-max-tokens {} must be between {MIN_IMAGE_TOKENS} \
             and {MAX_IMAGE_TOKENS}",
            args.image_max_tokens
        );
        return std::process::ExitCode::FAILURE;
    }
    let vision_config = xabe_model::VisionConfig::qwen3_6_35b_a3b();

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
        draft_tokens,
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
        "                 {} tokens charged per decode step ({})",
        sched.tokens_per_decode_step(),
        match args.spec_type {
            SpecType::None => "no drafting".to_owned(),
            SpecType::Ngram => format!(
                "{} n-gram drafts from {}..={} token suffixes",
                sched.draft_tokens_per_step(),
                args.spec_ngram_min,
                args.spec_ngram_max
            ),
            SpecType::NgramSimple => format!(
                "{} ngram-simple drafts from {}-token lookups",
                sched.draft_tokens_per_step(),
                args.spec_ngram_simple_size_n
            ),
            SpecType::NgramMod => format!(
                "{} ngram-mod drafts from {}-token lookups (n-min {})",
                sched.draft_tokens_per_step(),
                args.spec_ngram_mod_n_match,
                args.spec_ngram_mod_n_min
            ),
            SpecType::NgramMapK => format!(
                "{} ngram-map-k drafts from {}-token keys",
                sched.draft_tokens_per_step(),
                args.spec_ngram_map_k_size_n
            ),
            SpecType::NgramMapK4v => format!(
                "{} ngram-map-k4v drafts from {}-token keys (min hits {})",
                sched.draft_tokens_per_step(),
                args.spec_ngram_map_k4v_size_n,
                args.spec_ngram_map_k4v_min_hits
            ),
            SpecType::DraftMtp => format!(
                "{} tokens per step from the trained MTP head",
                sched.draft_tokens_per_step(),
            ),
            SpecType::SpecDflash => format!(
                "{} tokens per step from the DFlash drafter",
                sched.draft_tokens_per_step(),
            ),
        }
    );

    // Design rule 3 again, this time against the draft count actually asked
    // for. `SchedulerConfig` charges one token per decode, which is only true
    // with drafting off; with a draft count raised, a full house of decodes
    // can consume the whole step budget and starve prefill exactly as the rule
    // describes, while passing the constructor's check.
    let decode_tokens = sched
        .tokens_per_decode_step()
        .saturating_mul(sched.max_concurrent_decodes());
    if decode_tokens.saturating_add(sched.block_size()) >= sched.token_budget() {
        error!(
            "\nscheduler        FAIL — {} decodes drafting {} tokens each consume {} of a {} \
             token budget, leaving less than one {}-token block for prefill",
            sched.max_concurrent_decodes(),
            sched.draft_tokens_per_step(),
            decode_tokens,
            sched.token_budget(),
            sched.block_size()
        );
        error!("                 raise --token-budget, or lower the draft count for --spec-type");
        return std::process::ExitCode::FAILURE;
    }

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
    if args.mmproj.is_some() {
        info!("                 (excludes the mmproj vision tower, loaded per worker)");
    } else {
        info!("                 (text-only; excludes the ~1.5 GiB vision encoder)");
    }
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
        speculation,
        dflash_gguf: args.spec_dflash.clone(),
        draft_n_min: args.spec_draft_n_min as usize,
        draft_p_min: args.spec_draft_p_min,
        mmproj: args.mmproj.clone(),
        // The runtime's encode buffers are sized in pre-merge patches; the
        // serving ceiling is in post-merge tokens.
        max_image_patches: (args.image_max_tokens * vision_config.merge_factor()) as usize,
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
        model: args.alias,
        default_max_tokens: args.max_tokens,
        default_reasoning: !args.no_reasoning,
        sampling_defaults: http::SamplingDefaults {
            temperature: args.temperature,
            top_p: args.top_p,
            min_p: args.min_p,
        },
        vision: args.mmproj.is_some().then_some(http::VisionServingConfig {
            config: vision_config,
            max_tokens: args.image_max_tokens,
        }),
    };
    match runtime.block_on(http::serve(engine, tokenizer, &address, server)) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(failure) => {
            error!("server           FAIL — {failure}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(spec: SpecType) -> Args {
        Args::parse_from([
            "--spec-type",
            match spec {
                SpecType::None => "none",
                SpecType::Ngram => "ngram",
                SpecType::NgramSimple => "ngram-simple",
                SpecType::NgramMod => "ngram-mod",
                SpecType::NgramMapK => "ngram-map-k",
                SpecType::NgramMapK4v => "ngram-map-k4v",
                SpecType::DraftMtp => "draft-mtp",
                SpecType::SpecDflash => "spec-dflash",
            },
        ])
    }

    #[test]
    fn the_renamed_serving_flags_parse_in_both_spellings() {
        // `--temp` is a clap alias of `--temperature`, `-a` a real short;
        // both must land on the same fields as the long forms.
        let long = Args::parse_from(["--temperature", "0.5", "--alias", "m", "--max-tokens", "64"]);
        assert_eq!(long.temperature, 0.5);
        assert_eq!(long.alias, "m");
        assert_eq!(long.max_tokens, 64);

        let aliased = Args::parse_from(["--temp", "0.5", "-a", "m"]);
        assert_eq!(aliased.temperature, 0.5);
        assert_eq!(aliased.alias, "m");

        let joined = Args::parse_from(["--temp=0"]);
        assert_eq!(joined.temperature, 0.0);
    }

    #[test]
    fn the_sampling_default_flags_parse() {
        let sampled = Args::parse_from(["--top-p", "0.95", "--min-p", "0.05"]);
        assert_eq!(sampled.top_p, 0.95);
        assert_eq!(sampled.min_p, 0.05);
        let silent = Args::parse_from([] as [&str; 0]);
        assert_eq!(silent.top_p, 1.0);
        assert_eq!(silent.min_p, 0.0);
        assert_eq!(silent.temperature, 1.0);
    }

    #[test]
    fn no_speculation_charges_one_token_per_decode() {
        assert_eq!(
            resolve_speculation(&args(SpecType::None)),
            Ok((0, Speculation::None))
        );
    }

    #[test]
    fn ngram_carries_its_window_and_its_draft_count() {
        assert_eq!(
            resolve_speculation(&args(SpecType::Ngram)),
            Ok((3, Speculation::Ngram { min: 2, max: 4 }))
        );
    }

    #[test]
    fn an_inverted_ngram_window_is_refused() {
        let mut inverted = args(SpecType::Ngram);
        inverted.spec_ngram_min = 5;
        inverted.spec_ngram_max = 3;
        assert!(resolve_speculation(&inverted).is_err());

        inverted.spec_ngram_min = 0;
        inverted.spec_ngram_max = 4;
        assert!(resolve_speculation(&inverted).is_err());
    }

    #[test]
    fn an_ngram_window_longer_than_the_context_is_refused() {
        // It could never match, and `NgramConfig` would reject it further
        // down with a message that does not name the flag.
        let mut wide = args(SpecType::Ngram);
        wide.spec_ngram_max = wide.total_context as usize;
        assert!(resolve_speculation(&wide).is_err());
    }

    #[test]
    fn the_llama_cpp_ngram_variants_resolve_with_their_defaults() {
        // The tuple's first element is what the scheduler budgets per step:
        // the drafter's own cap (size-m or n-max), exactly as llama.cpp
        // defaults them.
        assert_eq!(
            resolve_speculation(&args(SpecType::NgramSimple)),
            Ok((48, Speculation::NgramSimple { size_n: 12 }))
        );
        assert_eq!(
            resolve_speculation(&args(SpecType::NgramMod)),
            Ok((
                64,
                Speculation::NgramMod {
                    n_match: 24,
                    n_min: 48
                }
            ))
        );
        assert_eq!(
            resolve_speculation(&args(SpecType::NgramMapK)),
            Ok((
                48,
                Speculation::NgramMapK {
                    size_n: 12,
                    min_hits: 1
                }
            ))
        );
        assert_eq!(
            resolve_speculation(&args(SpecType::NgramMapK4v)),
            Ok((
                48,
                Speculation::NgramMapK4v {
                    size_n: 12,
                    min_hits: 1
                }
            ))
        );
    }

    #[test]
    fn the_ngram_variant_flags_enforce_llama_cpp_ranges() {
        let mut bad = args(SpecType::NgramSimple);
        bad.spec_ngram_simple_size_n = 0;
        assert!(resolve_speculation(&bad).is_err());
        let mut bad = args(SpecType::NgramSimple);
        bad.spec_ngram_simple_size_m = 1025;
        assert!(resolve_speculation(&bad).is_err());
        let mut bad = args(SpecType::NgramMod);
        bad.spec_ngram_mod_n_match = 0;
        assert!(resolve_speculation(&bad).is_err());
        let mut bad = args(SpecType::NgramMod);
        bad.spec_ngram_mod_n_min = 1025;
        assert!(resolve_speculation(&bad).is_err());
        let mut bad = args(SpecType::NgramMod);
        bad.spec_ngram_mod_n_max = 0;
        assert!(resolve_speculation(&bad).is_err());
        let mut bad = args(SpecType::NgramMapK);
        bad.spec_ngram_map_k_min_hits = 0;
        assert!(resolve_speculation(&bad).is_err());
        let mut bad = args(SpecType::NgramMapK4v);
        bad.spec_ngram_map_k4v_size_n = 2000;
        assert!(resolve_speculation(&bad).is_err());
    }

    #[test]
    fn the_ngram_variant_flags_spell_exactly_like_llama_cpp() {
        let parsed = Args::parse_from([
            "--spec-type",
            "ngram-mod",
            "--spec-ngram-mod-n-match",
            "16",
            "--spec-ngram-mod-n-min",
            "8",
            "--spec-ngram-mod-n-max",
            "32",
        ]);
        assert_eq!(
            resolve_speculation(&parsed),
            Ok((
                32,
                Speculation::NgramMod {
                    n_match: 16,
                    n_min: 8
                }
            ))
        );
        let parsed = Args::parse_from([
            "--spec-type",
            "ngram-map-k4v",
            "--spec-ngram-map-k4v-size-n",
            "8",
            "--spec-ngram-map-k4v-size-m",
            "24",
            "--spec-ngram-map-k4v-min-hits",
            "2",
        ]);
        assert_eq!(
            resolve_speculation(&parsed),
            Ok((
                24,
                Speculation::NgramMapK4v {
                    size_n: 8,
                    min_hits: 2
                }
            ))
        );
        let parsed = Args::parse_from([
            "--spec-type",
            "ngram-simple",
            "--spec-ngram-simple-size-n",
            "6",
            "--spec-ngram-simple-size-m",
            "12",
            "--spec-ngram-simple-min-hits",
            "3",
        ]);
        assert_eq!(
            resolve_speculation(&parsed),
            Ok((12, Speculation::NgramSimple { size_n: 6 }))
        );
    }

    #[test]
    fn draft_mtp_resolves_to_the_mtp_speculation() {
        let (drafts, speculation) =
            resolve_speculation(&args(SpecType::DraftMtp)).expect("draft-mtp serves");
        assert_eq!(speculation, Speculation::Mtp);
        assert_eq!(
            drafts,
            xabe_sched::config::DEFAULT_DRAFT_TOKENS_PER_STEP,
            "the draft count must come from --spec-draft-n-max's default",
        );

        let mut zero = args(SpecType::DraftMtp);
        zero.spec_draft_n_max = 0;
        assert!(resolve_speculation(&zero).is_err());
    }
}
