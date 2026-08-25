//! The HTTP surface: three request dialects, one engine.
//!
//! `/v1/completions` and `/v1/chat/completions` are OpenAI's, `/v1/responses`
//! is OpenAI's newer Responses API, and `/v1/messages` is Anthropic's. They
//! differ only in how a request is unwrapped and how output is wrapped again;
//! all four funnel through [`generate::Generation`], which owns admission,
//! detokenization, stop sequences, and cancellation.
//!
//! `temperature`, `top_p`, `top_k` and `seed` are honoured — decoding
//! samples through the engine's host sampler, and `temperature: 0` is greedy
//! argmax on the device. Tool calling is honoured too: definitions render
//! into the model's own template section and `<tool_call>` blocks are parsed
//! back out of the output. What is still refused, loudly, is anything that
//! would need machinery this server does not have — `n > 1`, a forced
//! `tool_choice` — so a caller is never handed a reply that silently answers
//! a different question than the one it asked. See `docs/API.md`.

mod anthropic;
mod auth;
mod chat;
mod error;
mod generate;
mod openai;
mod responses;
mod tools;
mod vision;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::response::sse::Event;
use axum::routing::{get, post};
use serde::Serialize;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tracing::{error, info, warn};
use xabe_engine::{Engine, WorkerId};
use xabe_sched::request::RequestId;

pub use generate::SamplingDefaults;
pub use vision::VisionServingConfig;

use generate::{ClientEvent, EngineFinish};

type ClientSender = mpsc::UnboundedSender<ClientEvent>;
type ClientMap = Arc<Mutex<HashMap<RequestId, ClientSender>>>;

/// The model identifier reported to clients that do not name one.
pub const DEFAULT_MODEL: &str = "Qwen3.6-35B-A3B";

/// The vocabulary entry that closes the model's reasoning span.
const THINK_CLOSE: &str = "</think>";

/// The serving defaults a request may override.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// `None` leaves the server open, which is what it was before an API key
    /// could be configured.
    pub api_key: Option<String>,
    /// The model name reported by `/v1/models` and echoed in responses.
    pub model: String,
    /// The output token limit for a request that does not set one.
    pub default_max_tokens: u32,
    /// Whether a chat request that says nothing about reasoning gets it.
    /// The model's own template defaults this on.
    pub default_reasoning: bool,
    /// What a request that does not set its own sampling fields samples
    /// with. Both dialects document temperature 1.0; a temperature of `0`
    /// makes silent requests greedy, which is what this server always did
    /// before it had a sampler.
    pub sampling_defaults: SamplingDefaults,
    /// `Some` when `--mmproj` loaded a vision tower; `None` serves text
    /// only and refuses image parts with a 400 that names the flag.
    pub vision: Option<VisionServingConfig>,
}

#[derive(Clone)]
struct AppState {
    engine: Arc<Engine>,
    tokenizer: Arc<Tokenizer>,
    clients: ClientMap,
    next_id: Arc<AtomicU64>,
    api_key: Option<Arc<str>>,
    /// The id of `</think>`, looked up once so the generation path can close
    /// the reasoning span on a token comparison.
    think_close_token: Option<i32>,
    model: Arc<str>,
    default_max_tokens: u32,
    default_reasoning: bool,
    sampling_defaults: SamplingDefaults,
    /// The vision serving state, with the pad token already resolved.
    vision: Option<Arc<vision::VisionServing>>,
    /// Handlers currently blocked trying to take a worker lock to submit a
    /// request. A driver loop holds its worker's lock for a whole GPU step
    /// and would otherwise reacquire it immediately; this is how it learns
    /// to stand aside. See `worker_loop`.
    submit_waiters: Arc<AtomicUsize>,
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
    // `event:` first, then `data:`. The SSE grammar accumulates both until
    // the blank line, so either order dispatches correctly through a
    // conforming parser — but every published example of these two streams
    // puts the name first, and clients that scan for it rather than parse
    // are common enough that matching the wire format costs nothing.
    Event::default()
        .event(name)
        .json_data(value)
        .expect("server-owned response types serialize")
}

/// Report tools this server dropped because it cannot execute them.
///
/// At `warn` so it appears by default. The request still serves — refusing it
/// outright would take the caller's own function tools down with the hosted
/// one — but a harness that actually needed the dropped tool would otherwise
/// only discover it as a worse answer, with nothing to point at.
fn warn_unsupported(offered: &crate::http::tools::OfferedTools) {
    if !offered.unsupported.is_empty() {
        warn!(
            "dropped {} tool(s) this server cannot execute: {}; serving the {} function tool(s) \
             offered alongside them",
            offered.unsupported.len(),
            offered.unsupported.join(", "),
            offered.definitions.len(),
        );
    }
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

/// Drive one worker and fan its output out to waiting clients.
///
/// One of these runs per worker, on its own thread rather than a task
/// because a step is a blocking, GPU-bound call.
///
/// The three loops are deliberately not synchronized. The engine used to be
/// stepped by a single thread that spawned all three workers and joined them
/// every step, which made every card run at the speed of the slowest one in
/// that step: a decode had to wait out whatever prefill chunk another card
/// happened to be grinding through. Each loop now takes only its own
/// worker's lock, so the cards interleave at whatever rate their own work
/// allows. See `docs/BENCHMARKS.md`.
fn worker_loop(state: AppState, worker: WorkerId) {
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
        match state.engine.step_worker(worker) {
            // No device bound: nothing for this loop to drive, ever. Sleep
            // rather than spin — the worker count comes from the engine, so
            // this is only reachable in a partially bound fleet.
            Ok(None) => std::thread::sleep(Duration::from_millis(1)),
            Ok(Some(step)) => {
                // A client is registered *before* its request reaches the
                // engine (see `generate.rs`), so a non-empty client map does
                // not mean there is work to do. Between those two moments
                // this loop would otherwise spin on the worker lock — and
                // `parking_lot::Mutex` hands off eagerly, but a loop that
                // reacquires immediately still crowds out the handler trying
                // to submit the next request.
                //
                // Measured on three concurrent sessions: 20,636 steps that
                // scheduled nothing in a 27 s run, against 15 in the run that
                // happened to win the lock race. The sessions behind the
                // first one could not be admitted until the prompt in flight
                // finished, which read as a scheduler that refused to share.
                // A 1 ms back-off on an empty step costs nothing when there
                // is work — the branch is never taken then — and hands the
                // lock over when there is not.
                if step.decode_items == 0 && step.prefill_items == 0 {
                    std::thread::sleep(Duration::from_millis(1));
                }
                // Hand the lock to anybody waiting to submit.
                //
                // The idle back-off above is not enough on its own: while a
                // long prompt is prefilling, every step is productive, so
                // the loop takes the lock, holds it for the whole GPU step —
                // over a second at 64K — releases it and takes it straight
                // back. A handler blocked in `place_tokens` scores every
                // worker, so it can be made to wait out an entire prompt.
                //
                // Measured at 64K with three sessions dispatched 2 ms apart:
                // `running=1, waiting=0` for the first 22 steps, the other
                // two sessions not merely unscheduled but never submitted.
                // It reads as a scheduler that will not share; it is a lock
                // that will not yield. One millisecond against a step of a
                // second or more is not a throughput cost, and it only
                // applies when somebody is actually waiting.
                if state.submit_waiters.load(Ordering::Acquire) > 0 {
                    std::thread::sleep(Duration::from_millis(1));
                }
                let mut clients = state.clients.lock().expect("client map poisoned");
                let mut disconnected_ids = Vec::new();
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
                drop(clients);
                for id in disconnected_ids {
                    state.engine.cancel(id);
                }
            }
            Err(failure) => {
                // A step failure is not attributable to one client, so every
                // client this worker could have been serving is told. The
                // other workers' loops are untouched: their sessions are on
                // different cards and are still being served.
                error!(worker = worker.0, "device scheduler failed: {failure}");
                let mut clients = state.clients.lock().expect("client map poisoned");
                let failed = clients
                    .drain()
                    .map(|(id, client)| {
                        let _ = client.send(ClientEvent::Error(failure.to_string()));
                        id
                    })
                    .collect::<Vec<_>>();
                drop(clients);
                for id in failed {
                    state.engine.cancel(id);
                }
            }
        }
    }
}

/// Bring up the HTTP surface over an already-loaded engine.
pub async fn serve(
    engine: Engine,
    tokenizer: Tokenizer,
    address: &str,
    config: ServerConfig,
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
    // Image expansion pivots on the pad token, and the rendered markers must
    // tokenize as the single ids the model was trained on — a vocabulary
    // without them cannot serve images correctly, so that fails startup
    // rather than every request.
    let vision = config
        .vision
        .map(|serving| {
            for spelling in [vision::VISION_START, vision::VISION_END] {
                if tokenizer.token_to_id(spelling).is_none() {
                    return Err(format!(
                        "--mmproj was given, but the tokenizer has no `{spelling}` token"
                    ));
                }
            }
            let image_pad = tokenizer.token_to_id(vision::IMAGE_PAD).ok_or_else(|| {
                format!(
                    "--mmproj was given, but the tokenizer has no `{}` token",
                    vision::IMAGE_PAD
                )
            })?;
            Ok(Arc::new(vision::VisionServing {
                config: serving.config,
                max_tokens: serving.max_tokens,
                image_pad,
            }))
        })
        .transpose()?;
    let state = AppState {
        engine: Arc::new(engine),
        tokenizer: Arc::new(tokenizer),
        clients: Arc::new(Mutex::new(HashMap::new())),
        next_id: Arc::new(AtomicU64::new(1)),
        submit_waiters: Arc::new(AtomicUsize::new(0)),
        api_key: config.api_key.map(Arc::from),
        think_close_token,
        model: Arc::from(config.model.as_str()),
        default_max_tokens: config.default_max_tokens,
        default_reasoning: config.default_reasoning,
        sampling_defaults: config.sampling_defaults,
        vision,
    };
    // One driver thread per worker. They share nothing but the client map
    // and the engine's shared prefix cache, so a card that is prefilling no
    // longer holds up a card that only has a decode token to emit.
    for index in 0..state.engine.worker_count() {
        let worker_state = state.clone();
        let worker = WorkerId(index as u32);
        std::thread::Builder::new()
            .name(format!("xabe-worker-{index}"))
            .spawn(move || worker_loop(worker_state, worker))
            .map_err(|error| error.to_string())?;
    }

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
        ))
        // axum's default 2 MB body cap is plenty for text but not for
        // base64 images; admission still bounds what a prompt can cost.
        .layer(axum::extract::DefaultBodyLimit::max(64 * 1024 * 1024));
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
    match &state.vision {
        Some(vision) => info!(
            "                 image input enabled — up to {} tokens per image",
            vision.max_tokens
        ),
        None => info!("                 text only — image parts are refused (no --mmproj)"),
    }
    axum::serve(listener, app)
        .await
        .map_err(|error| error.to_string())
}
