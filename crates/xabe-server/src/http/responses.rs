//! OpenAI's `/v1/responses`.
//!
//! The Responses API models a reply as a list of *output items* rather than a
//! single message, which is what lets the model's reasoning span and its
//! answer be two separate things instead of one string with a marker in it.
//! That maps onto this engine's output directly.
//!
//! Responses are not stored: `store` is reported as `false` and
//! `previous_response_id` is refused, because a server that accepted a
//! conversation reference it cannot resolve would answer without the context
//! the caller believed it had sent.

use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};

use super::chat::{Content, Conversation, Turn, unsupported_role, unsupported_tools};
use super::error::{ApiError, Dialect, parse_body};
use super::generate::{Chunk, Finish, Generation, GenerationSpec};
use super::{AppState, sse_named, unix_now};

const DIALECT: Dialect = Dialect::OpenAi;

const fn default_max_output_tokens() -> u32 {
    16
}

/// One entry of a structured `input` list.
#[derive(Debug, Deserialize)]
struct InputItem {
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Option<Content>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Input {
    Text(String),
    Items(Vec<InputItem>),
}

#[derive(Debug, Deserialize)]
struct ReasoningOptions {
    /// `"none"` and `"minimal"` are the Responses API's way of asking the
    /// model not to think.
    #[serde(default)]
    effort: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponsesRequest {
    input: Input,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default = "default_max_output_tokens")]
    max_output_tokens: u32,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    reasoning: Option<ReasoningOptions>,
    #[serde(default)]
    tools: Option<Value>,
    #[serde(default)]
    previous_response_id: Option<String>,
}

impl ResponsesRequest {
    fn thinking_enabled(&self) -> bool {
        !matches!(
            self.reasoning
                .as_ref()
                .and_then(|reasoning| reasoning.effort.as_deref()),
            Some("none" | "minimal")
        )
    }

    fn conversation(&self) -> Result<Conversation, ApiError> {
        let mut conversation = Conversation {
            system: self.instructions.clone(),
            turns: Vec::new(),
        };
        match &self.input {
            Input::Text(text) => conversation.turns.push(Turn::User(text.clone())),
            Input::Items(items) => {
                for item in items {
                    if item.kind.as_deref().is_some_and(|kind| kind != "message") {
                        return Err(ApiError::bad_request(
                            DIALECT,
                            format!(
                                "input items of type `{}` are not supported; send `message` items",
                                item.kind.as_deref().unwrap_or_default()
                            ),
                        ));
                    }
                    let (text, thinking) = item
                        .content
                        .as_ref()
                        .map(Content::split)
                        .transpose()
                        .map_err(|failure| ApiError::bad_request(DIALECT, failure))?
                        .unwrap_or_default();
                    match item.role.as_deref() {
                        Some("system" | "developer") => {
                            if !conversation.turns.is_empty() {
                                return Err(ApiError::bad_request(
                                    DIALECT,
                                    "a system input item must come before the first user item",
                                ));
                            }
                            let system = conversation.system.get_or_insert_with(String::new);
                            if !system.is_empty() {
                                system.push('\n');
                            }
                            system.push_str(&text);
                        }
                        Some("user") => conversation.turns.push(Turn::User(text)),
                        Some("assistant") => conversation.turns.push(Turn::Assistant {
                            reasoning: thinking,
                            content: text,
                        }),
                        Some(role) => return Err(unsupported_role(DIALECT, role)),
                        None => {
                            return Err(ApiError::bad_request(
                                DIALECT,
                                "every input item needs a `role`",
                            ));
                        }
                    }
                }
            }
        }
        if conversation.turns.is_empty() {
            return Err(ApiError::bad_request(
                DIALECT,
                "`input` must contain at least one user or assistant item",
            ));
        }
        Ok(conversation)
    }
}

/// The status a response carries, and the detail that explains a truncated
/// one.
fn status_of(finish: &Finish) -> (&'static str, Value) {
    match finish {
        Finish::Length => ("incomplete", json!({ "reason": "max_output_tokens" })),
        _ => ("completed", Value::Null),
    }
}

struct Envelope {
    id: String,
    created: u64,
    model: String,
}

impl Envelope {
    fn reasoning_item_id(&self) -> String {
        format!("rs_{}", self.id.trim_start_matches("resp_"))
    }

    fn message_item_id(&self) -> String {
        format!("msg_{}", self.id.trim_start_matches("resp_"))
    }

    fn reasoning_item(&self, text: &str) -> Value {
        json!({
            "id": self.reasoning_item_id(),
            "type": "reasoning",
            "summary": [],
            "content": [{ "type": "reasoning_text", "text": text }],
        })
    }

    fn message_item(&self, text: &str) -> Value {
        json!({
            "id": self.message_item_id(),
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": text, "annotations": [] }],
        })
    }

    fn object(&self, status: &str, incomplete: Value, output: Vec<Value>, usage: Value) -> Value {
        json!({
            "id": self.id,
            "object": "response",
            "created_at": self.created,
            "status": status,
            "model": self.model,
            "output": output,
            "incomplete_details": incomplete,
            "instructions": Value::Null,
            "store": false,
            "usage": usage,
        })
    }
}

fn usage_of(generation: &Generation) -> Value {
    json!({
        "input_tokens": generation.prompt_tokens(),
        "output_tokens": generation.completion_tokens(),
        "total_tokens": generation.prompt_tokens() + generation.completion_tokens(),
    })
}

pub(crate) async fn create(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<Response, ApiError> {
    let request: ResponsesRequest = parse_body(&body, DIALECT)?;
    if request.tools.is_some() {
        return Err(unsupported_tools(DIALECT));
    }
    if request.previous_response_id.is_some() {
        return Err(ApiError::bad_request(
            DIALECT,
            "`previous_response_id` is not supported: this server does not store responses, \
             so send the whole conversation in `input`",
        ));
    }
    let thinking = request.thinking_enabled();
    let prompt = request.conversation()?.render(thinking);
    let encoding = state
        .tokenizer
        .encode(prompt, false)
        .map_err(|error| ApiError::bad_request(DIALECT, error.to_string()))?;
    let spec = GenerationSpec {
        prompt: encoding.get_ids().to_vec(),
        max_tokens: request.max_output_tokens,
        stop: Vec::new(),
        thinking,
        trim_spans: true,
    };
    let mut generation = Generation::start(&state, spec, DIALECT)?;
    let envelope = Envelope {
        id: format!("resp_{}", generation.request_id()),
        created: unix_now(),
        model: state.model_name(request.model),
    };

    if !request.stream {
        let (reasoning, text) = generation.collect().await?;
        let (status, incomplete) = status_of(&generation.finish());
        let mut output = Vec::with_capacity(2);
        if !reasoning.is_empty() {
            output.push(envelope.reasoning_item(&reasoning));
        }
        output.push(envelope.message_item(&text));
        return Ok(
            axum::Json(envelope.object(status, incomplete, output, usage_of(&generation)))
                .into_response(),
        );
    }

    let stream = async_stream::stream! {
        // Every event carries a monotonic sequence number so a client can tell
        // a dropped event from a delayed one.
        let sequence = std::cell::Cell::new(0u64);
        macro_rules! emit {
            ($name:expr, $($body:tt)*) => {{
                let mut value = json!($($body)*);
                value["sequence_number"] = json!(sequence.replace(sequence.get() + 1));
                sse_named($name, &value)
            }};
        }

        yield Ok::<_, std::convert::Infallible>(emit!("response.created", {
            "type": "response.created",
            "response": envelope.object("in_progress", Value::Null, vec![], Value::Null),
        }));
        yield Ok(emit!("response.in_progress", {
            "type": "response.in_progress",
            "response": envelope.object("in_progress", Value::Null, vec![], Value::Null),
        }));

        let (mut reasoning, mut text) = (String::new(), String::new());
        // Output items open on their first delta, so a response with no
        // reasoning carries no reasoning item.
        let mut output_index = 0usize;
        let mut open: Option<&'static str> = None;

        loop {
            let chunk = match generation.next().await {
                Ok(Some(chunk)) => Some(chunk),
                Ok(None) => None,
                Err(failure) => {
                    yield Ok(emit!("error", failure.payload()));
                    return;
                }
            };
            let kind = match &chunk {
                Some(Chunk::Reasoning(_)) => Some("reasoning"),
                Some(Chunk::Text(_)) => Some("message"),
                None => None,
            };
            if open != kind {
                // Close whatever is open before opening the next item.
                match open {
                    Some("reasoning") => {
                        yield Ok(emit!("response.reasoning_text.done", {
                            "type": "response.reasoning_text.done",
                            "item_id": envelope.reasoning_item_id(),
                            "output_index": output_index,
                            "content_index": 0,
                            "text": reasoning,
                        }));
                        yield Ok(emit!("response.output_item.done", {
                            "type": "response.output_item.done",
                            "output_index": output_index,
                            "item": envelope.reasoning_item(&reasoning),
                        }));
                        output_index += 1;
                    }
                    Some(_) => {
                        yield Ok(emit!("response.output_text.done", {
                            "type": "response.output_text.done",
                            "item_id": envelope.message_item_id(),
                            "output_index": output_index,
                            "content_index": 0,
                            "text": text,
                        }));
                        yield Ok(emit!("response.content_part.done", {
                            "type": "response.content_part.done",
                            "item_id": envelope.message_item_id(),
                            "output_index": output_index,
                            "content_index": 0,
                            "part": { "type": "output_text", "text": text, "annotations": [] },
                        }));
                        yield Ok(emit!("response.output_item.done", {
                            "type": "response.output_item.done",
                            "output_index": output_index,
                            "item": envelope.message_item(&text),
                        }));
                        output_index += 1;
                    }
                    None => {}
                }
                match kind {
                    Some("reasoning") => {
                        yield Ok(emit!("response.output_item.added", {
                            "type": "response.output_item.added",
                            "output_index": output_index,
                            "item": {
                                "id": envelope.reasoning_item_id(),
                                "type": "reasoning",
                                "summary": [],
                                "content": [],
                            },
                        }));
                    }
                    Some(_) => {
                        yield Ok(emit!("response.output_item.added", {
                            "type": "response.output_item.added",
                            "output_index": output_index,
                            "item": {
                                "id": envelope.message_item_id(),
                                "type": "message",
                                "status": "in_progress",
                                "role": "assistant",
                                "content": [],
                            },
                        }));
                        yield Ok(emit!("response.content_part.added", {
                            "type": "response.content_part.added",
                            "item_id": envelope.message_item_id(),
                            "output_index": output_index,
                            "content_index": 0,
                            "part": { "type": "output_text", "text": "", "annotations": [] },
                        }));
                    }
                    None => {}
                }
                open = kind;
            }
            match chunk {
                Some(Chunk::Reasoning(delta)) => {
                    reasoning.push_str(&delta);
                    yield Ok(emit!("response.reasoning_text.delta", {
                        "type": "response.reasoning_text.delta",
                        "item_id": envelope.reasoning_item_id(),
                        "output_index": output_index,
                        "content_index": 0,
                        "delta": delta,
                    }));
                }
                Some(Chunk::Text(delta)) => {
                    text.push_str(&delta);
                    yield Ok(emit!("response.output_text.delta", {
                        "type": "response.output_text.delta",
                        "item_id": envelope.message_item_id(),
                        "output_index": output_index,
                        "content_index": 0,
                        "delta": delta,
                    }));
                }
                None => break,
            }
        }

        let (status, incomplete) = status_of(&generation.finish());
        let mut output = Vec::with_capacity(2);
        if !reasoning.is_empty() {
            output.push(envelope.reasoning_item(&reasoning));
        }
        output.push(envelope.message_item(&text));
        yield Ok(emit!("response.completed", {
            "type": "response.completed",
            "response": envelope.object(status, incomplete, output, usage_of(&generation)),
        }));
    };
    Ok(Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(body: &str) -> ResponsesRequest {
        serde_json::from_str(body).expect("test request should parse")
    }

    #[test]
    fn a_bare_string_input_is_one_user_turn() {
        let rendered = request(r#"{"input":"Hi"}"#)
            .conversation()
            .expect("a string input should fold")
            .render(true);
        assert_eq!(
            rendered,
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn instructions_become_the_system_turn() {
        let rendered = request(
            r#"{"instructions":"Be terse.","input":[{"role":"user","content":[{"type":"input_text","text":"Hi"}]}]}"#,
        )
        .conversation()
        .expect("an item input should fold")
        .render(true);
        assert!(rendered.starts_with("<|im_start|>system\nBe terse.<|im_end|>\n"));
    }

    #[test]
    fn minimal_reasoning_effort_turns_thinking_off() {
        assert!(request(r#"{"input":"Hi"}"#).thinking_enabled());
        assert!(request(r#"{"input":"Hi","reasoning":{"effort":"high"}}"#).thinking_enabled());
        assert!(!request(r#"{"input":"Hi","reasoning":{"effort":"none"}}"#).thinking_enabled());
        assert!(!request(r#"{"input":"Hi","reasoning":{"effort":"minimal"}}"#).thinking_enabled());
    }

    #[test]
    fn a_non_message_input_item_is_refused() {
        assert!(
            request(r#"{"input":[{"type":"function_call","role":"user","content":"x"}]}"#)
                .conversation()
                .is_err()
        );
    }

    #[test]
    fn a_truncated_response_says_so() {
        assert_eq!(status_of(&Finish::Length).0, "incomplete");
        assert_eq!(status_of(&Finish::EndOfTurn).0, "completed");
    }
}
