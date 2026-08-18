//! OpenAI-compatible HTTP admission in front of the shared engine.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tracing::{error, info};
use xabe_cache::radix::{BlockHash, ROOT_HASH, hash_block};
use xabe_engine::Engine;
use xabe_sched::request::{NewRequest, RequestId};

enum ClientEvent {
    Token(i32),
    Done(FinishReason),
    Error(String),
}

#[derive(Clone, Copy)]
enum FinishReason {
    Stop,
    Length,
}

impl FinishReason {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

type ClientSender = mpsc::UnboundedSender<ClientEvent>;
type ClientMap = Arc<Mutex<HashMap<RequestId, ClientSender>>>;

#[derive(Clone)]
struct AppState {
    engine: Arc<Mutex<Engine>>,
    tokenizer: Arc<Tokenizer>,
    clients: ClientMap,
    next_id: Arc<AtomicU64>,
    block_size: usize,
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    model: Option<String>,
}

const fn default_max_tokens() -> u32 {
    16
}

#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    object: &'static str,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Serialize)]
struct Choice {
    text: String,
    index: u32,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.0,
            Json(serde_json::json!({ "error": { "message": self.1 } })),
        )
            .into_response()
    }
}

fn prompt_hashes(tokens: &[u32], block_size: usize) -> Vec<BlockHash> {
    let mut parent = ROOT_HASH;
    tokens
        .chunks(block_size)
        .map(|block| {
            parent = hash_block(parent, block);
            parent
        })
        .collect()
}

async fn health() -> &'static str {
    "ok"
}

async fn completions(
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, ApiError> {
    if request.stream {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            "stream=true is not implemented; use a non-streaming completion".to_owned(),
        ));
    }
    if request.max_tokens == 0 {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            "max_tokens must be positive".to_owned(),
        ));
    }
    let encoding = state
        .tokenizer
        .encode(request.prompt, false)
        .map_err(|error| ApiError(StatusCode::BAD_REQUEST, error.to_string()))?;
    let prompt_u32 = encoding.get_ids().to_vec();
    if prompt_u32.is_empty() {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            "prompt must not be empty".to_owned(),
        ));
    }
    let prompt = prompt_u32
        .iter()
        .map(|&token| token as i32)
        .collect::<Vec<_>>();
    let prompt_tokens = u32::try_from(prompt.len()).map_err(|_| {
        ApiError(
            StatusCode::PAYLOAD_TOO_LARGE,
            "tokenized prompt exceeds the request length representation".to_owned(),
        )
    })?;
    let hashes = prompt_hashes(&prompt_u32, state.block_size);
    let id = RequestId(state.next_id.fetch_add(1, Ordering::Relaxed));
    let (sender, mut receiver) = mpsc::unbounded_channel();
    state
        .clients
        .lock()
        .expect("client map poisoned")
        .insert(id, sender);
    let placement = state.engine.lock().expect("engine poisoned").place_tokens(
        NewRequest {
            id,
            prompt_tokens,
            max_output_tokens: request.max_tokens,
        },
        prompt,
        &hashes,
    );
    if let Err(error) = placement {
        state
            .clients
            .lock()
            .expect("client map poisoned")
            .remove(&id);
        return Err(ApiError(StatusCode::SERVICE_UNAVAILABLE, error.to_string()));
    }

    let mut output = Vec::with_capacity(request.max_tokens as usize);
    let finish_reason = loop {
        match receiver.recv().await {
            Some(ClientEvent::Token(token)) => output.push(token as u32),
            Some(ClientEvent::Done(reason)) => break reason,
            Some(ClientEvent::Error(message)) => {
                return Err(ApiError(StatusCode::INTERNAL_SERVER_ERROR, message));
            }
            None => {
                return Err(ApiError(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "scheduler closed the request without a completion status".to_owned(),
                ));
            }
        }
    };
    let text = state
        .tokenizer
        .decode(&output, false)
        .map_err(|error| ApiError(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
    let completion_tokens = output.len();
    let prompt_tokens = prompt_u32.len();
    Ok(Json(CompletionResponse {
        id: format!("cmpl-{}", id.0),
        object: "text_completion",
        model: request
            .model
            .unwrap_or_else(|| "Qwen3.6-35B-A3B".to_owned()),
        choices: vec![Choice {
            text,
            index: 0,
            finish_reason: finish_reason.as_str(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }))
}

fn scheduler_loop(state: AppState) {
    loop {
        if state
            .clients
            .lock()
            .expect("client map poisoned")
            .is_empty()
        {
            std::thread::sleep(Duration::from_millis(1));
            continue;
        }
        let result = state.engine.lock().expect("engine poisoned").step_devices();
        match result {
            Ok(steps) => {
                let mut clients = state.clients.lock().expect("client map poisoned");
                for (_, step) in steps {
                    let stopped = &step.stopped;
                    for (id, token) in step.generated {
                        if let Some(client) = clients.get(&id) {
                            let _ = client.send(ClientEvent::Token(token));
                        }
                    }
                    for id in step.completed {
                        if let Some(client) = clients.remove(&id) {
                            let reason = if stopped.contains(&id) {
                                FinishReason::Stop
                            } else {
                                FinishReason::Length
                            };
                            let _ = client.send(ClientEvent::Done(reason));
                        }
                    }
                }
            }
            Err(failure) => {
                error!("device scheduler failed: {failure}");
                let mut clients = state.clients.lock().expect("client map poisoned");
                for (_, client) in clients.drain() {
                    let _ = client.send(ClientEvent::Error(failure.to_string()));
                }
            }
        }
    }
}

pub async fn serve(
    engine: Engine,
    tokenizer: Tokenizer,
    block_size: usize,
    address: &str,
) -> Result<(), String> {
    let state = AppState {
        engine: Arc::new(Mutex::new(engine)),
        tokenizer: Arc::new(tokenizer),
        clients: Arc::new(Mutex::new(HashMap::new())),
        next_id: Arc::new(AtomicU64::new(1)),
        block_size,
    };
    let scheduler_state = state.clone();
    std::thread::Builder::new()
        .name("xabe-scheduler".to_owned())
        .spawn(move || scheduler_loop(scheduler_state))
        .map_err(|error| error.to_string())?;
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/completions", post(completions))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .map_err(|error| error.to_string())?;
    info!("serving OpenAI-compatible completions at http://{address}");
    axum::serve(listener, app)
        .await
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashes_are_chained_at_natural_attention_blocks() {
        let tokens = (0..10).collect::<Vec<_>>();
        let hashes = prompt_hashes(&tokens, 4);
        assert_eq!(hashes.len(), 3);
        let first = hash_block(ROOT_HASH, &tokens[..4]);
        assert_eq!(hashes[0], first);
        assert_eq!(hashes[1], hash_block(first, &tokens[4..8]));
        assert_eq!(hashes[2], hash_block(hashes[1], &tokens[8..]));
    }
}
