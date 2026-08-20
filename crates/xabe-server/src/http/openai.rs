//! OpenAI's `/v1/completions` and `/v1/chat/completions`.

use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

use super::chat::{Content, Conversation, Turn, unsupported_role, unsupported_tools};
use super::error::{ApiError, Dialect, parse_body};
use super::generate::{Chunk, Finish, Generation, GenerationSpec};
use super::{AppState, sse_json, unix_now};

const DIALECT: Dialect = Dialect::OpenAi;

const fn default_max_tokens() -> u32 {
    16
}

/// `stop` is a string or a list of strings in both OpenAI shapes.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Stop {
    One(String),
    Many(Vec<String>),
}

impl Stop {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(one) => vec![one],
            Self::Many(many) => many,
        }
    }
}

fn stop_sequences(stop: Option<Stop>) -> Vec<String> {
    stop.map(Stop::into_vec)
        .unwrap_or_default()
        .into_iter()
        .filter(|sequence| !sequence.is_empty())
        .collect()
}

#[derive(Debug, Deserialize, Clone, Copy)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

/// The finish reason string OpenAI uses. A stop sequence and an end-of-turn
/// token are both `"stop"` here; the Anthropic surface distinguishes them.
fn finish_reason(finish: &Finish) -> &'static str {
    match finish {
        Finish::EndOfTurn | Finish::StopSequence(_) => "stop",
        Finish::Length => "length",
    }
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

impl Usage {
    fn of(generation: &Generation) -> Self {
        Self {
            prompt_tokens: generation.prompt_tokens(),
            completion_tokens: generation.completion_tokens(),
            total_tokens: generation.prompt_tokens() + generation.completion_tokens(),
        }
    }
}

/// Refuse the parameters that would change the response's shape.
///
/// Parameters that only change sampling are accepted and ignored — decoding
/// is greedy argmax — but a caller that asked for four completions and got one
/// has been answered wrongly, not approximately.
fn reject_shape_changing(n: Option<u32>, best_of: Option<u32>) -> Result<(), ApiError> {
    for (name, value) in [("n", n), ("best_of", best_of)] {
        if value.is_some_and(|value| value > 1) {
            return Err(ApiError::bad_request(
                DIALECT,
                format!(
                    "`{name}` above 1 is not supported: decoding is greedy, so every completion \
                     would be identical"
                ),
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- completions

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Prompt {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    prompt: Prompt,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
    #[serde(default)]
    stop: Option<Stop>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    best_of: Option<u32>,
}

#[derive(Serialize)]
struct CompletionChoice {
    text: String,
    index: u32,
    finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
}

pub(crate) async fn completions(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<Response, ApiError> {
    let request: CompletionRequest = parse_body(&body, DIALECT)?;
    reject_shape_changing(request.n, request.best_of)?;
    let prompt = match request.prompt {
        Prompt::One(prompt) => prompt,
        Prompt::Many(mut prompts) if prompts.len() == 1 => prompts.remove(0),
        Prompt::Many(_) => {
            return Err(ApiError::bad_request(
                DIALECT,
                "a batch of prompts is not supported; send one prompt per request",
            ));
        }
    };
    let encoding = state
        .tokenizer
        .encode(prompt, false)
        .map_err(|error| ApiError::bad_request(DIALECT, error.to_string()))?;
    // A raw completion is not a chat turn, so there is no `<think>` block open
    // and everything the model emits is answer text.
    let spec = GenerationSpec {
        prompt: encoding.get_ids().to_vec(),
        max_tokens: request.max_tokens,
        stop: stop_sequences(request.stop),
        thinking: false,
        trim_spans: false,
    };
    let mut generation = Generation::start(&state, spec, DIALECT)?;
    let id = format!("cmpl-{}", generation.request_id());
    let model = state.model_name(request.model);
    let created = unix_now();

    if !request.stream {
        let (_, text) = generation.collect().await?;
        return Ok(axum::Json(CompletionResponse {
            id,
            object: "text_completion",
            created,
            model,
            choices: vec![CompletionChoice {
                text,
                index: 0,
                finish_reason: Some(finish_reason(&generation.finish())),
            }],
            usage: Some(Usage::of(&generation)),
        })
        .into_response());
    }

    let include_usage = request.stream_options.is_some_and(|o| o.include_usage);
    let stream = async_stream::stream! {
        loop {
            match generation.next().await {
                Ok(Some(Chunk::Reasoning(text) | Chunk::Text(text))) => {
                    yield Ok::<_, std::convert::Infallible>(sse_json(&CompletionResponse {
                        id: id.clone(),
                        object: "text_completion",
                        created,
                        model: model.clone(),
                        choices: vec![CompletionChoice { text, index: 0, finish_reason: None }],
                        usage: None,
                    }));
                }
                Ok(None) => {
                    yield Ok(sse_json(&CompletionResponse {
                        id: id.clone(),
                        object: "text_completion",
                        created,
                        model: model.clone(),
                        choices: vec![CompletionChoice {
                            text: String::new(),
                            index: 0,
                            finish_reason: Some(finish_reason(&generation.finish())),
                        }],
                        usage: include_usage.then(|| Usage::of(&generation)),
                    }));
                    break;
                }
                Err(failure) => {
                    yield Ok(sse_json(&failure.payload()));
                    break;
                }
            }
        }
        yield Ok(axum::response::sse::Event::default().data("[DONE]"));
    };
    Ok(Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}

// ----------------------------------------------------------- chat completions

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    #[serde(default)]
    content: Option<Content>,
    /// Reasoning replayed from a previous turn, as this server emits it.
    #[serde(default)]
    reasoning_content: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct ChatTemplateKwargs {
    /// vLLM's spelling of the model template's own `enable_thinking`.
    #[serde(default)]
    enable_thinking: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<u32>,
    /// OpenAI's newer name for `max_tokens`.
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
    #[serde(default)]
    stop: Option<Stop>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    tools: Option<serde_json::Value>,
    #[serde(default)]
    chat_template_kwargs: Option<ChatTemplateKwargs>,
}

/// Fold OpenAI chat messages into a conversation.
fn conversation(messages: Vec<ChatMessage>) -> Result<Conversation, ApiError> {
    let mut conversation = Conversation::default();
    let mut system = Vec::new();
    for message in messages {
        let (text, thinking) = message
            .content
            .as_ref()
            .map(Content::split)
            .transpose()
            .map_err(|failure| ApiError::bad_request(DIALECT, failure))?
            .unwrap_or_default();
        match message.role.as_str() {
            "system" | "developer" => {
                if !conversation.turns.is_empty() {
                    return Err(ApiError::bad_request(
                        DIALECT,
                        "a system message must come before the first user or assistant message",
                    ));
                }
                system.push(text);
            }
            "user" => conversation.turns.push(Turn::User(text)),
            "assistant" => conversation.turns.push(Turn::Assistant {
                reasoning: message.reasoning_content.unwrap_or(thinking),
                content: text,
            }),
            role => return Err(unsupported_role(DIALECT, role)),
        }
    }
    if !system.is_empty() {
        conversation.system = Some(system.join("\n"));
    }
    if conversation.turns.is_empty() {
        return Err(ApiError::bad_request(
            DIALECT,
            "`messages` must contain at least one user or assistant message",
        ));
    }
    Ok(conversation)
}

#[derive(Serialize, Default)]
struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
}

#[derive(Serialize)]
struct ChatChoice {
    index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<ChatDelta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    delta: Option<ChatDelta>,
    finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
struct ChatResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
}

pub(crate) async fn chat_completions(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<Response, ApiError> {
    let request: ChatRequest = parse_body(&body, DIALECT)?;
    if request.tools.is_some() {
        return Err(unsupported_tools(DIALECT));
    }
    reject_shape_changing(request.n, None)?;
    let thinking = request
        .chat_template_kwargs
        .as_ref()
        .and_then(|kwargs| kwargs.enable_thinking)
        .unwrap_or(true);
    let conversation = conversation(request.messages)?;
    let prompt = conversation.render(thinking);
    let encoding = state
        .tokenizer
        .encode(prompt, false)
        .map_err(|error| ApiError::bad_request(DIALECT, error.to_string()))?;
    let spec = GenerationSpec {
        prompt: encoding.get_ids().to_vec(),
        max_tokens: request
            .max_completion_tokens
            .or(request.max_tokens)
            .unwrap_or(default_max_tokens()),
        stop: stop_sequences(request.stop),
        thinking,
        trim_spans: true,
    };
    let mut generation = Generation::start(&state, spec, DIALECT)?;
    let id = format!("chatcmpl-{}", generation.request_id());
    let model = state.model_name(request.model);
    let created = unix_now();

    if !request.stream {
        let (reasoning, content) = generation.collect().await?;
        return Ok(axum::Json(ChatResponse {
            id,
            object: "chat.completion",
            created,
            model,
            choices: vec![ChatChoice {
                index: 0,
                message: Some(ChatDelta {
                    role: Some("assistant"),
                    content: Some(content),
                    reasoning_content: (!reasoning.is_empty()).then_some(reasoning),
                }),
                delta: None,
                finish_reason: Some(finish_reason(&generation.finish())),
            }],
            usage: Some(Usage::of(&generation)),
        })
        .into_response());
    }

    let include_usage = request.stream_options.is_some_and(|o| o.include_usage);
    let stream = async_stream::stream! {
        let chunk = |delta: ChatDelta, finish: Option<&'static str>, usage: Option<Usage>| {
            sse_json(&ChatResponse {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model.clone(),
                choices: vec![ChatChoice { index: 0, message: None, delta: Some(delta), finish_reason: finish }],
                usage,
            })
        };
        // OpenAI's first chunk announces the role and carries no content.
        yield Ok::<_, std::convert::Infallible>(chunk(
            ChatDelta { role: Some("assistant"), content: Some(String::new()), reasoning_content: None },
            None,
            None,
        ));
        loop {
            match generation.next().await {
                Ok(Some(Chunk::Text(text))) => {
                    yield Ok(chunk(ChatDelta { content: Some(text), ..ChatDelta::default() }, None, None));
                }
                Ok(Some(Chunk::Reasoning(text))) => {
                    yield Ok(chunk(
                        ChatDelta { reasoning_content: Some(text), ..ChatDelta::default() },
                        None,
                        None,
                    ));
                }
                Ok(None) => {
                    yield Ok(chunk(
                        ChatDelta::default(),
                        Some(finish_reason(&generation.finish())),
                        include_usage.then(|| Usage::of(&generation)),
                    ));
                    break;
                }
                Err(failure) => {
                    yield Ok(sse_json(&failure.payload()));
                    break;
                }
            }
        }
        yield Ok(axum::response::sse::Event::default().data("[DONE]"));
    };
    Ok(Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_owned(),
            content: Some(Content::Text(content.to_owned())),
            reasoning_content: None,
        }
    }

    #[test]
    fn leading_system_messages_merge_and_the_rest_become_turns() {
        let conversation = conversation(vec![
            message("system", "Be terse."),
            message("developer", "Answer in English."),
            message("user", "Hi"),
        ])
        .expect("a system-led conversation should fold");
        assert_eq!(
            conversation.system.as_deref(),
            Some("Be terse.\nAnswer in English.")
        );
        assert_eq!(conversation.turns.len(), 1);
    }

    #[test]
    fn a_late_system_message_is_refused_rather_than_dropped() {
        // The model's own template silently discards it; silently discarding
        // an instruction the caller wrote is worse than refusing it.
        let error = conversation(vec![message("user", "Hi"), message("system", "Be terse.")])
            .expect_err("a system message after a turn should be refused");
        assert_eq!(
            error.payload()["error"]["message"],
            serde_json::json!(
                "a system message must come before the first user or assistant message"
            )
        );
    }

    #[test]
    fn a_tool_message_is_refused() {
        assert!(conversation(vec![message("tool", "{}")]).is_err());
    }

    #[test]
    fn a_conversation_with_no_turns_is_refused() {
        assert!(conversation(vec![message("system", "Be terse.")]).is_err());
    }

    #[test]
    fn a_stop_string_and_a_stop_list_are_the_same_thing() {
        let one: Option<Stop> = serde_json::from_str(r#""END""#).ok();
        let many: Option<Stop> = serde_json::from_str(r#"["END",""]"#).ok();
        assert_eq!(stop_sequences(one), vec!["END".to_owned()]);
        assert_eq!(stop_sequences(many), vec!["END".to_owned()]);
        assert!(stop_sequences(None).is_empty());
    }

    #[test]
    fn asking_for_more_than_one_completion_is_refused() {
        assert!(reject_shape_changing(Some(2), None).is_err());
        assert!(reject_shape_changing(None, Some(4)).is_err());
        assert!(reject_shape_changing(Some(1), Some(1)).is_ok());
        assert!(reject_shape_changing(None, None).is_ok());
    }
}
