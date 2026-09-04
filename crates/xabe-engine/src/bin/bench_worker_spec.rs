//! Scheduler-driven serving throughput with a speculative decoder on.
//!
//! `bench_worker_decode` asserts exactly `width` tokens per step — the
//! plain-decode contract — so it cannot time speculation, whose whole point
//! is a variable number of tokens per step. This binary times a fixed
//! *emitted-token* window instead: after every request is decode-ready, it
//! counts wall time until each sequence has emitted its quota, and reports
//! aggregate tokens per second plus tokens per scheduler step (the
//! end-to-end acceptance signal).
//!
//! ```sh
//! CUDA_VISIBLE_DEVICES=2 LLMXABE_SPEC=dflash LLMXABE_BATCH_N=1,3 \
//!   cargo run --release -p xabe-engine --bin bench_worker_spec -- 256 512
//! ```
//!
//! Args: `[context] [tokens_per_seq]` (defaults 256, 512). `LLMXABE_SPEC`
//! picks the decoder (`none`, `ngram`, `ngram-simple`, `ngram-mod`,
//! `ngram-map-k`, `ngram-map-k4v`, `draft-mtp`, `dflash`), each with
//! llama.cpp's own defaults; `LLMXABE_SPEC_DRAFTS` overrides the per-step
//! draft cap, which is what a matched-budget A/B across decoders needs
//! (note that `ngram-simple` never drafts with a cap below its 12-token
//! lookup — upstream drops any draft shorter than `size-n`);
//! `LLMXABE_DRAFT_N_MIN` / `LLMXABE_DRAFT_P_MIN` set the corresponding
//! serving gates (defaults 0, off);
//! `LLMXABE_DFLASH` the drafter GGUF; `LLMXABE_MODEL` the model;
//! `LLMXABE_BATCH_N` the widths. The synthetic prompt only seeds
//! generation — past the first few tokens the timed window is the model's
//! own text, which is the workload acceptance rates are about.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use tracing::{error, info};
use xabe_cache::CacheConfig;
use xabe_engine::worker::Speculation;
use xabe_engine::{SamplingParams, ServingConfig, Worker, WorkerId};
use xabe_gguf::GgufFile;
use xabe_model::ModelConfig;
use xabe_sched::config::{
    DEFAULT_DRAFT_TOKENS_PER_STEP, DEFAULT_WATERMARK_FRACTION, SchedulerConfig,
};
use xabe_sched::request::{NewRequest, RequestId};

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llmxabe/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";
const DEFAULT_DFLASH_PATH: &str =
    "/home/nixabe/llmxabe/models/Qwen3.6-35B-A3B-GGUF/qwen36-35b-a3b-dflash-Q8_0.gguf";
const DEFAULT_CONTEXT: usize = 256;
const DEFAULT_TOKENS: u32 = 512;
const PREFILL_CHUNK: usize = 2_048;
const TOKEN_BUDGET: u32 = 4_096;
const TOTAL_CONTEXT: u32 = 393_216;
/// Emitted per sequence before the timed window opens: enough to leave the
/// prompt's influence and reach steady generation.
const WARMUP_TOKENS: u32 = 32;

fn model_path() -> PathBuf {
    std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
}

fn dflash_path() -> PathBuf {
    std::env::var_os("LLMXABE_DFLASH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_DFLASH_PATH))
}

fn widths() -> Result<Vec<usize>, &'static str> {
    let raw = std::env::var("LLMXABE_BATCH_N").unwrap_or_else(|_| "1,3".to_owned());
    let parsed: Vec<usize> = raw
        .split(',')
        .filter_map(|part| part.trim().parse().ok())
        .collect();
    if parsed.is_empty() || parsed.iter().any(|width| !(1..=3).contains(width)) {
        return Err("LLMXABE_BATCH_N must contain comma-separated widths in 1..=3");
    }
    Ok(parsed)
}

/// llama.cpp's own default lookup length for the three variants that take
/// one (`--spec-ngram-{simple,map-k,map-k4v}-size-n`).
const LLAMA_CPP_SIZE_N: usize = 12;
/// llama.cpp's default draft cap for those three (`*-size-m`).
const LLAMA_CPP_SIZE_M: u32 = 48;
/// llama.cpp's `--spec-ngram-mod-{n-match,n-min,n-max}` defaults.
const LLAMA_CPP_MOD_N_MATCH: usize = 24;
const LLAMA_CPP_MOD_N_MIN: usize = 48;
const LLAMA_CPP_MOD_N_MAX: u32 = 64;

fn speculation() -> Result<(Speculation, u32), String> {
    let raw = std::env::var("LLMXABE_SPEC").unwrap_or_else(|_| "none".to_owned());
    // The draft cap is the scheduler's per-step count, so a matched-budget
    // A/B across decoders overrides it here rather than per variant.
    let cap = |default: u32| -> Result<u32, String> {
        match std::env::var("LLMXABE_SPEC_DRAFTS") {
            Ok(value) => value
                .parse()
                .map_err(|_| format!("LLMXABE_SPEC_DRAFTS={value} is not a token count")),
            Err(_) => Ok(default),
        }
    };
    match raw.as_str() {
        "none" => Ok((Speculation::None, 0)),
        "ngram" => Ok((
            Speculation::Ngram { min: 2, max: 4 },
            cap(DEFAULT_DRAFT_TOKENS_PER_STEP)?,
        )),
        "ngram-simple" => Ok((
            Speculation::NgramSimple {
                size_n: LLAMA_CPP_SIZE_N,
            },
            cap(LLAMA_CPP_SIZE_M)?,
        )),
        "ngram-mod" => Ok((
            Speculation::NgramMod {
                n_match: LLAMA_CPP_MOD_N_MATCH,
                n_min: LLAMA_CPP_MOD_N_MIN,
            },
            cap(LLAMA_CPP_MOD_N_MAX)?,
        )),
        "ngram-map-k" => Ok((
            Speculation::NgramMapK {
                size_n: LLAMA_CPP_SIZE_N,
                min_hits: 1,
            },
            cap(LLAMA_CPP_SIZE_M)?,
        )),
        "ngram-map-k4v" => Ok((
            Speculation::NgramMapK4v {
                size_n: LLAMA_CPP_SIZE_N,
                min_hits: 1,
            },
            cap(LLAMA_CPP_SIZE_M)?,
        )),
        "draft-mtp" => Ok((Speculation::Mtp, cap(DEFAULT_DRAFT_TOKENS_PER_STEP)?)),
        "dflash" => Ok((Speculation::DFlash, cap(DEFAULT_DRAFT_TOKENS_PER_STEP)?)),
        other => Err(format!(
            "LLMXABE_SPEC={other} is not none|ngram|ngram-simple|ngram-mod|ngram-map-k|\
             ngram-map-k4v|draft-mtp|dflash"
        )),
    }
}

fn synthetic_prompt(sequence: usize, len: usize, vocab: usize) -> Vec<i32> {
    (0..len)
        .map(|position| (((sequence + 1) * 104_729 + position * 7919 + 1234) % vocab) as i32)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_width(
    path: &std::path::Path,
    model: &ModelConfig,
    context: usize,
    tokens_per_seq: u32,
    width: usize,
    speculation: Speculation,
    drafts: u32,
) -> Result<WidthResult, String> {
    let cache = CacheConfig::with_defaults(model.clone()).map_err(|error| error.to_string())?;
    let scheduler = SchedulerConfig::new(
        TOKEN_BUDGET,
        cache.attention_block_size(),
        width as u32,
        DEFAULT_WATERMARK_FRACTION,
        drafts,
    )
    .map_err(|error| error.to_string())?;
    let attention_blocks = TOTAL_CONTEXT / cache.attention_block_size();
    let mut worker = Worker::new(
        WorkerId(0),
        0,
        cache,
        scheduler,
        attention_blocks,
        width as u32,
    );
    let draft_n_min = std::env::var("LLMXABE_DRAFT_N_MIN")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(0);
    let draft_p_min = std::env::var("LLMXABE_DRAFT_P_MIN")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(0.0);
    let serving = ServingConfig {
        speculation,
        dflash_gguf: matches!(speculation, Speculation::DFlash).then(dflash_path),
        draft_n_min,
        draft_p_min,
        ..ServingConfig::new(PREFILL_CHUNK.min(context).max(1))
    };
    worker
        .bind_device_for_benchmark(path, model.clone(), serving)
        .map_err(|error| error.to_string())?;

    let quota = WARMUP_TOKENS + tokens_per_seq;
    // Speculation lets sequences advance unevenly: while the slowest
    // sequence works through its quota, the fastest can emit up to a whole
    // verify window per step. The output budget must cover that divergence,
    // or a fast sequence retires at its cap before the window closes.
    // Deep-context runs need this set explicitly: prefilling a long prompt
    // makes sequences decode-ready at very different times, so a budget
    // derived from the draft count alone retires the earliest starter before
    // the latest one has caught up. An explicit value is also the fair one —
    // it reserves identical KV for every drafter.
    let max_output = std::env::var("LLMXABE_MAX_OUTPUT")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(quota * (drafts + 1) + 8);
    for sequence in 0..width {
        let request = NewRequest {
            id: RequestId(sequence as u64 + 1),
            prompt_tokens: context as u32,
            max_output_tokens: max_output,
        };
        worker
            .admit_tokens(
                request,
                synthetic_prompt(sequence, context, model.vocab_size as usize),
                Vec::new(),
                SamplingParams::GREEDY,
                None,
            )
            .map_err(|error| error.to_string())?;
    }

    // Prefill, timed separately: a sequence emits its first token from the
    // last chunk of its own prefill, so "every sequence has emitted one" is
    // the moment chunked prefill has finished for all of them. Speculation
    // never drafts here, but the trained drafters do extra per-chunk work
    // (the MTP head's catch-up pass, DFlash's feature taps), so this is the
    // number that says what a drafter costs before it has helped at all.
    let mut emitted = vec![0u32; width];
    let step_ceiling = 4 * (quota as usize + context.div_ceil(PREFILL_CHUNK)) * width + 64;
    let mut steps_taken = 0usize;
    let prefill_started = Instant::now();
    while emitted.contains(&0) {
        let step = worker.step_device().map_err(|error| error.to_string())?;
        for (id, _) in &step.generated {
            emitted[(id.0 - 1) as usize] += 1;
        }
        steps_taken += 1;
        if steps_taken > step_ceiling {
            return Err(format!(
                "prefill made no progress after {steps_taken} steps (ceiling {step_ceiling}), emitted {emitted:?}"
            ));
        }
    }
    let prefill_elapsed = prefill_started.elapsed().as_secs_f64();
    let prefill_tps = (context * width) as f64 / prefill_elapsed;

    // The rest of the warm-up, untimed: the decode window opens once every
    // sequence has cleared the warm-up quota.
    while emitted.iter().any(|&count| count < WARMUP_TOKENS) {
        let step = worker.step_device().map_err(|error| error.to_string())?;
        for (id, _) in &step.generated {
            emitted[(id.0 - 1) as usize] += 1;
        }
        steps_taken += 1;
        if steps_taken > step_ceiling {
            return Err(format!(
                "warm-up made no progress after {steps_taken} steps (ceiling {step_ceiling}), emitted {emitted:?}"
            ));
        }
    }

    // The timed window: wall clock and scheduler steps until every sequence
    // has its quota. Speculation emits a variable number per step, so both
    // the token count and the step count are measured, not assumed.
    let window_start: Vec<u32> = emitted.clone();
    let mut timed_steps = 0u64;
    // A fixed number of scheduler steps is the honest window once drafts get
    // long or prefill is deep: "until every sequence gains N" is gated by the
    // slowest sequence while the fastest runs arbitrarily far ahead, which
    // both skews the token count and grows the racers' contexts mid-window.
    let fixed_steps: Option<u64> = std::env::var("LLMXABE_TIMED_STEPS")
        .ok()
        .and_then(|raw| raw.parse().ok());
    let started = Instant::now();
    while match fixed_steps {
        Some(target) => timed_steps < target,
        None => emitted
            .iter()
            .zip(&window_start)
            .any(|(&count, &start)| count < start + tokens_per_seq),
    } {
        let step = worker.step_device().map_err(|error| error.to_string())?;
        for (id, _) in &step.generated {
            emitted[(id.0 - 1) as usize] += 1;
        }
        timed_steps += 1;
        if steps_taken + timed_steps as usize > step_ceiling {
            return Err(format!(
                "timed window stalled: {steps_taken} pre + {timed_steps} timed vs ceiling {step_ceiling}, emitted {emitted:?}, start {window_start:?}"
            ));
        }
    }
    let elapsed = started.elapsed().as_secs_f64();

    let window_tokens: u32 = emitted
        .iter()
        .zip(&window_start)
        .map(|(&count, &start)| count - start)
        .sum();
    let tps = f64::from(window_tokens) / elapsed;
    let tokens_per_step = f64::from(window_tokens) / timed_steps as f64;
    Ok(WidthResult {
        prefill_tps,
        tps,
        tokens_per_step,
        timed_steps,
    })
}

/// One width's measurement: prefill and decode are separate regimes and a
/// drafter can move them in opposite directions, so neither is folded away.
struct WidthResult {
    prefill_tps: f64,
    tps: f64,
    tokens_per_step: f64,
    timed_steps: u64,
}

fn main() -> ExitCode {
    xabe_log::init_from_args();
    let mut args = std::env::args().skip(1);
    let context: usize = args
        .next()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(DEFAULT_CONTEXT);
    let tokens_per_seq: u32 = args
        .next()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(DEFAULT_TOKENS);
    let widths = match widths() {
        Ok(widths) => widths,
        Err(error) => {
            error!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let (spec, drafts) = match speculation() {
        Ok(pair) => pair,
        Err(error) => {
            error!("{error}");
            return ExitCode::FAILURE;
        }
    };

    let path = model_path();
    // The architecture comes from the file, not from a constant: this
    // binary serves `qwen35` as well as `qwen35moe`, and a hardcoded config
    // fails a dense file with a wall of shape mismatches instead of running
    // it.
    let model = match GgufFile::open(&path)
        .map_err(|e| e.to_string())
        .and_then(|f| ModelConfig::from_gguf(&f).map_err(|e| e.to_string()))
    {
        Ok(model) => model,
        Err(error) => {
            error!("{}: {error}", path.display());
            return ExitCode::FAILURE;
        }
    };
    info!(
        "bench_worker_spec: context={context} tokens/seq={tokens_per_seq} spec={spec:?} drafts={drafts}",
    );
    info!("N     prefill tok/s   decode tok/s   tokens/step   steps");
    for &width in &widths {
        match run_width(&path, &model, context, tokens_per_seq, width, spec, drafts) {
            Ok(r) => {
                info!(
                    "N={width}   {:12.1}   {:12.1}       {:7.3}      {}",
                    r.prefill_tps, r.tps, r.tokens_per_step, r.timed_steps
                );
            }
            Err(error) => {
                error!("N={width}: {error}");
                return ExitCode::FAILURE;
            }
        }
    }
    ExitCode::SUCCESS
}
