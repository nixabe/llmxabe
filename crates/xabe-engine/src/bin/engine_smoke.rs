//! Three-worker, nine-sequence scheduler/runtime acceptance smoke.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

use tracing::{error, info};
use xabe_cache::CacheConfig;
use xabe_cache::radix::{BlockHash, ROOT_HASH, hash_block};
use xabe_engine::{Engine, RouterConfig, WorkerId};
use xabe_model::ModelConfig;
use xabe_sched::SchedulerConfig;
use xabe_sched::request::{NewRequest, RequestId};

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llama.cpp/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";
const OUTPUT: u32 = 4;

fn hashes(tokens: &[i32], block_size: usize) -> Vec<BlockHash> {
    let mut parent = ROOT_HASH;
    tokens
        .chunks(block_size)
        .map(|chunk| {
            let chunk = chunk.iter().map(|&token| token as u32).collect::<Vec<_>>();
            parent = hash_block(parent, &chunk);
            parent
        })
        .collect()
}

fn prompt(id: u64, tokens: usize, vocab: u32) -> Vec<i32> {
    (0..tokens)
        .map(|position| ((id as usize * 104_729 + position * 7919 + 1234) % vocab as usize) as i32)
        .collect()
}

fn admit(
    engine: &mut Engine,
    id: u64,
    tokens: usize,
    model: &ModelConfig,
    block_size: usize,
) -> Result<WorkerId, String> {
    let prompt = prompt(id, tokens, model.vocab_size);
    let hashes = hashes(&prompt, block_size);
    engine
        .place_tokens(
            NewRequest {
                id: RequestId(id),
                prompt_tokens: tokens as u32,
                max_output_tokens: OUTPUT,
            },
            prompt,
            &hashes,
        )
        .map(|placement| placement.worker)
        .map_err(|failure| failure.to_string())
}

fn main() -> ExitCode {
    xabe_log::init_from_args();
    let path = std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH));
    let model = ModelConfig::qwen3_6_35b_a3b();
    let cache = CacheConfig::with_defaults(model.clone()).expect("shipped cache config is valid");
    let block_size = cache.attention_block_size();
    let sched = SchedulerConfig::with_defaults(4096, block_size, 3)
        .expect("serving scheduler configuration is valid");
    let mut engine = Engine::new(
        &[0, 1, 2],
        cache,
        sched,
        393_216 / block_size,
        3,
        RouterConfig::balanced(),
    );
    if let Err((worker, failure)) = engine.bind_devices(&path, model.clone(), 128) {
        error!("{worker} failed to bind: {failure}");
        return ExitCode::FAILURE;
    }

    let mut placements = BTreeMap::<WorkerId, usize>::new();
    for id in 1..=6 {
        match admit(&mut engine, id, 128, &model, block_size as usize) {
            Ok(worker) => *placements.entry(worker).or_default() += 1,
            Err(failure) => {
                error!("initial admission failed: {failure}");
                return ExitCode::FAILURE;
            }
        }
    }
    let first = match engine.step_devices() {
        Ok(steps) => steps,
        Err(failure) => {
            error!("initial prefill failed: {failure}");
            return ExitCode::FAILURE;
        }
    };
    let mut completed = first
        .iter()
        .map(|(_, step)| step.completed.len())
        .sum::<usize>();

    for id in 7..=9 {
        match admit(&mut engine, id, 512, &model, block_size as usize) {
            Ok(worker) => *placements.entry(worker).or_default() += 1,
            Err(failure) => {
                error!("mixed admission failed: {failure}");
                return ExitCode::FAILURE;
            }
        }
    }
    if placements.values().copied().collect::<Vec<_>>() != [3, 3, 3] {
        error!(
            ?placements,
            "router did not distribute nine requests three per worker"
        );
        return ExitCode::FAILURE;
    }
    let mixed = match engine.step_devices() {
        Ok(steps) => steps,
        Err(failure) => {
            error!("mixed step failed: {failure}");
            return ExitCode::FAILURE;
        }
    };
    if mixed
        .iter()
        .any(|(_, step)| step.decode_items != 2 || step.prefill_items != 1)
    {
        error!("each worker must execute two decodes plus one prefill in the mixed step");
        return ExitCode::FAILURE;
    }
    completed += mixed
        .iter()
        .map(|(_, step)| step.completed.len())
        .sum::<usize>();
    while completed < 9 {
        match engine.step_devices() {
            Ok(steps) => {
                completed += steps
                    .iter()
                    .map(|(_, step)| step.completed.len())
                    .sum::<usize>()
            }
            Err(failure) => {
                error!("serving step failed: {failure}");
                return ExitCode::FAILURE;
            }
        }
    }
    info!("three workers completed nine sequences and the 2-decode + 1-prefill mixed step");
    ExitCode::SUCCESS
}
