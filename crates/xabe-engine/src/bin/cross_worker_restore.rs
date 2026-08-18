//! Cross-context pinned-prefix restore equality smoke for two CUDA workers.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use tracing::{error, info};
use xabe_cache::CacheConfig;
use xabe_engine::{SequenceSnapshot, Worker, WorkerId};
use xabe_model::ModelConfig;
use xabe_sched::SchedulerConfig;
use xabe_sched::request::{NewRequest, RequestId};

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llama.cpp/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";
const PROMPT: usize = 2048;
const OUTPUT: u32 = 4;

fn run(worker: &mut Worker) -> Result<(Vec<i32>, Option<Arc<SequenceSnapshot>>), String> {
    let mut output = Vec::new();
    let mut snapshot = None;
    loop {
        let step = worker
            .step_device()
            .map_err(|failure| failure.to_string())?;
        output.extend(step.generated.into_iter().map(|(_, token)| token));
        if let Some((_, retained)) = step.retained.into_iter().last() {
            snapshot = Some(retained);
        }
        if !step.completed.is_empty() {
            return Ok((output, snapshot));
        }
    }
}

fn main() -> ExitCode {
    xabe_log::init_from_args();
    let path = std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH));
    let model = ModelConfig::qwen3_6_35b_a3b();
    let cache = CacheConfig::with_defaults(model.clone()).expect("shipped cache config is valid");
    let sched = SchedulerConfig::with_defaults(4096, cache.attention_block_size(), 1)
        .expect("serving scheduler configuration is valid");
    let blocks = 131_072 / cache.attention_block_size();
    let mut cold = Worker::new(WorkerId(0), 0, cache.clone(), sched, blocks, 1);
    let mut restored = Worker::new(WorkerId(1), 1, cache, sched, blocks, 1);
    if let Err(failure) = cold.bind_device(&path, model.clone(), PROMPT) {
        error!("cold worker failed to bind: {failure}");
        return ExitCode::FAILURE;
    }
    if let Err(failure) = restored.bind_device(&path, model.clone(), PROMPT) {
        error!("restore worker failed to bind: {failure}");
        return ExitCode::FAILURE;
    }
    let prompt = (0..PROMPT)
        .map(|position| ((position * 7919 + 1234) % model.vocab_size as usize) as i32)
        .collect::<Vec<_>>();
    let request = NewRequest {
        id: RequestId(1),
        prompt_tokens: PROMPT as u32,
        max_output_tokens: OUTPUT,
    };
    if let Err(failure) = cold.admit_tokens(request, prompt.clone()) {
        error!("cold admission failed: {failure}");
        return ExitCode::FAILURE;
    }
    let (cold_output, snapshot) = match run(&mut cold) {
        Ok(result) => result,
        Err(failure) => {
            error!("cold run failed: {failure}");
            return ExitCode::FAILURE;
        }
    };
    let Some(snapshot) = snapshot else {
        error!("cold worker did not retain the 2048-token boundary");
        return ExitCode::FAILURE;
    };
    if let Err(failure) = restored.admit_tokens_restored(request, prompt, snapshot) {
        error!("restored admission failed: {failure}");
        return ExitCode::FAILURE;
    }
    let (restored_output, _) = match run(&mut restored) {
        Ok(result) => result,
        Err(failure) => {
            error!("restored run failed: {failure}");
            return ExitCode::FAILURE;
        }
    };
    if cold_output != restored_output {
        error!(
            ?cold_output,
            ?restored_output,
            "restored tokens differ from cold prefill"
        );
        return ExitCode::FAILURE;
    }
    info!(
        ?cold_output,
        "cross-worker restored output matches cold prefill"
    );
    ExitCode::SUCCESS
}
