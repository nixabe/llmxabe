//! The HTTP surface: three request dialects, one engine.
//!
//! `/v1/completions` and `/v1/chat/completions` are OpenAI's, `/v1/responses`
//! is OpenAI's newer Responses API, and `/v1/messages` is Anthropic's. They
//! differ only in how a request is unwrapped and how output is wrapped again;
//! all four funnel through [`generate::Generation`], which owns admission,
//! detokenization, stop sequences, and cancellation.
//!
//! What the engine below does *not* offer is worth stating here, because the
//! wire formats imply it: decoding is greedy argmax, so `temperature`,
//! `top_p`, `top_k` and `seed` are accepted for client compatibility and have
//! no effect. Anything that would change the *shape* of a response rather
//! than its content — `n > 1`, tool definitions — is refused instead, so a
//! caller is never handed a reply that silently answers a different question
//! than the one it asked. See `docs/API.md`.

mod anthropic;
mod auth;
mod chat;
mod error;
mod generate;
mod openai;
mod responses;

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::response::sse::Event;
use axum::routing::{get, post};
use serde::Serialize;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tracing::{error, info};
use xabe_engine::Engine;
use xabe_sched::request::RequestId;

use generate::{ClientEvent, EngineFinish};

type ClientSender = mpsc::UnboundedSender<ClientEvent>;
type ClientMap = Arc<Mutex<HashMap<RequestId, ClientSender>>>;

/// The model identifier reported to clients that do not name one.
pub const DEFAULT_MODEL: &str = "Qwen3.6-35B-A3B";

/// The vocabulary entry that closes the model's reasoning span.
const THINK_CLOSE: &str = "</think>";

#[derive(Clone)]
struct AppState {
    engine: Arc<Mutex<Engine>>,
    tokenizer: Arc<Tokenizer>,
    clients: ClientMap,
    next_id: Arc<AtomicU64>,
    block_size: usize,
    /// `None` leaves the server open, which is what it was before an API key
    /// could be configured.
    api_key: Option<Arc<str>>,
    /// The id of `</think>`, looked up once so the generation path can close
    /// the reasoning span on a token comparison.
    think_close_token: Option<i32>,
    model: Arc<str>,
}

impl AppState {
    /// The model name to echo back: whatever the caller asked for, or this
    /// server's own name.
    fn model_name(&self, requested: Option<String>) -> String {
        requested.unwrap_or_else(|| self.model.to_string())
    }
}

/// Seconds since the epoch, for the `created` fields both OpenAI shapes
/// carry.
fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
}

/// An SSE event carrying a JSON payload.
fn sse_json<T: Serialize>(value: &T) -> Event {
    Event::default()
        .json_data(value)
        .expect("server-owned response types serialize")
}

/// A named SSE event carrying a JSON payload, as the Anthropic and Responses
/// streams use.
fn sse_named<T: Serialize>(name: &str, value: &T) -> Event {
    sse_json(value).event(name)
}

async fn health() -> &'static str {
    "ok"
}

#[derive(Serialize)]
struct ModelCard {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

#[derive(Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelCard>,
}

/// The model list OpenAI clients probe before their first request.
async fn models(
    axum::extract::State(state): axum::extract::State<AppState>,
) -> axum::Json<ModelList> {
    axum::Json(ModelList {
        object: "list",
        data: vec![ModelCard {
            id: state.model.to_string(),
            object: "model",
            created: 0,
            owned_by: "llmxabe",
        }],
    })
}

/// Drive the engine and fan its output out to waiting clients.
///
/// This runs on its own thread rather than a task because a step is a
/// blocking, GPU-bound call that holds the engine lock for its whole
/// duration.
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
                let mut disconnected_ids = Vec::new();
                for (_, step) in steps {
                    let stopped = &step.stopped;
                    for (id, token) in step.generated {
                        let disconnected = clients
                            .get(&id)
                            .is_some_and(|client| client.send(ClientEvent::Token(token)).is_err());
                        if disconnected {
                            clients.remove(&id);
                            disconnected_ids.push(id);
                        }
                    }
                    for id in step.completed {
                        if let Some(client) = clients.remove(&id) {
                            let reason = if stopped.contains(&id) {
                                EngineFinish::Eos
                            } else {
                                EngineFinish::Length
                            };
                            let _ = client.send(ClientEvent::Done(reason));
                        }
                    }
                }
                drop(clients);
                let mut engine = state.engine.lock().expect("engine poisoned");
                for id in disconnected_ids {
                    engine.cancel(id);
                }
            }
            Err(failure) => {
                error!("device scheduler failed: {failure}");
                let mut clients = state.clients.lock().expect("client map poisoned");
                let failed = clients
                    .drain()
                    .map(|(id, client)| {
                        let _ = client.send(ClientEvent::Error(failure.to_string()));
                        id
                    })
                    .collect::<Vec<_>>();
                drop(clients);
                let mut engine = state.engine.lock().expect("engine poisoned");
                for id in failed {
                    engine.cancel(id);
                }
            }
        }
    }
}

/// Bring up the HTTP surface over an already-loaded engine.
pub async fn serve(
    engine: Engine,
    tokenizer: Tokenizer,
    block_size: usize,
    address: &str,
    api_key: Option<String>,
) -> Result<(), String> {
    let think_close_token = tokenizer
        .token_to_id(THINK_CLOSE)
        .and_then(|id| i32::try_from(id).ok());
    if think_close_token.is_none() {
        // Without it, everything the model emits is reported as answer text,
        // reasoning included. That is a visible quality difference, so it is
        // said out loud rather than discovered in the output.
        error!(
            "tokenizer has no `{THINK_CLOSE}` token — reasoning will not be separated from answers"
        );
    }
    let state = AppState {
        engine: Arc::new(Mutex::new(engine)),
        tokenizer: Arc::new(tokenizer),
        clients: Arc::new(Mutex::new(HashMap::new())),
        next_id: Arc::new(AtomicU64::new(1)),
        block_size,
        api_key: api_key.map(Arc::from),
        think_close_token,
        model: Arc::from(DEFAULT_MODEL),
    };
    let scheduler_state = state.clone();
    std::thread::Builder::new()
        .name("xabe-scheduler".to_owned())
        .spawn(move || scheduler_loop(scheduler_state))
        .map_err(|error| error.to_string())?;

    // `/health` stays outside the authenticated routes so a load balancer can
    // probe the server without holding a key.
    let api = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/completions", post(openai::completions))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/responses", post(responses::create))
        .route("/v1/messages", post(anthropic::messages))
        .route("/v1/messages/count_tokens", post(anthropic::count_tokens))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth::require_api_key,
        ));
    let app = Router::new()
        .route("/health", get(health))
        .merge(api)
        .with_state(state.clone());

    let listener = tokio::net::TcpListener::bind(address)
        .await
        .map_err(|error| error.to_string())?;
    info!("serving OpenAI and Anthropic endpoints at http://{address}");
    info!("                 /v1/completions  /v1/chat/completions  /v1/responses  /v1/messages");
    if state.api_key.is_some() {
        info!("                 API key required (Authorization: Bearer, or x-api-key)");
    } else {
        info!("                 no API key configured — every caller is accepted");
    }
    axum::serve(listener, app)
        .await
        .map_err(|error| error.to_string())
}
