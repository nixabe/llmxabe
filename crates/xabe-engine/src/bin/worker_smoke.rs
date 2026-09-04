//! Scheduler-to-CUDA worker smoke test for one or three concurrent sequences.
//!
//! This is intentionally below HTTP/tokenization: it proves the live worker
//! path using token ids and reports emitted ids through tracing.

use std::path::PathBuf;
use std::process::ExitCode;

use tracing::{error, info};
use xabe_cache::CacheConfig;
use xabe_engine::{SamplingParams, ServingConfig, Worker, WorkerId};
use xabe_model::ModelConfig;
use xabe_sched::config::SchedulerConfig;
use xabe_sched::request::{NewRequest, RequestId};

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llmxabe/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";
const PROMPT: usize = 128;
const OUTPUT: u32 = 4;

fn main() -> ExitCode {
    xabe_log::init_from_args();
    let width = std::env::var("LLMXABE_BATCH_N")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(3);
    if !(1..=3).contains(&width) {
        error!("LLMXABE_BATCH_N must be in 1..=3");
        return ExitCode::FAILURE;
    }
    let path = std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH));
    if !path.exists() {
        error!("model not found at {}", path.display());
        return ExitCode::FAILURE;
    }

    let model = ModelConfig::qwen3_6_35b_a3b();
    let cache = CacheConfig::with_defaults(model.clone()).expect("shipped cache config is valid");
    let sched = SchedulerConfig::with_defaults(4096, cache.attention_block_size(), width as u32)
        .expect("serving budget satisfies the starvation gate");
    let attention_blocks = 393_216 / cache.attention_block_size();
    let mut worker = Worker::new(WorkerId(0), 0, cache, sched, attention_blocks, width as u32);
    info!("binding worker0 to visible CUDA device 0");
    if let Err(error) = worker.bind_device(&path, model.clone(), ServingConfig::new(PROMPT)) {
        error!("worker binding failed: {error}");
        return ExitCode::FAILURE;
    }

    for sequence in 0..width {
        let prompt: Vec<i32> = (0..PROMPT)
            .map(|position| {
                ((sequence * 104_729 + position * 7919 + 1234) % model.vocab_size as usize) as i32
            })
            .collect();
        let req = NewRequest {
            id: RequestId(sequence as u64 + 1),
            prompt_tokens: PROMPT as u32,
            max_output_tokens: OUTPUT,
        };
        if let Err(error) =
            worker.admit_tokens(req, prompt, Vec::new(), SamplingParams::GREEDY, None)
        {
            error!("admission failed: {error}");
            return ExitCode::FAILURE;
        }
    }

    while worker.scheduler().running_len() + worker.scheduler().waiting_len() > 0 {
        match worker.step_device() {
            Ok(step) => {
                for (request, token) in step.generated {
                    info!(request = request.0, token, "generated");
                }
            }
            Err(error) => {
                error!("device step failed: {error}");
                return ExitCode::FAILURE;
            }
        }
    }
    info!("worker smoke completed for N={width}");
    ExitCode::SUCCESS
}
