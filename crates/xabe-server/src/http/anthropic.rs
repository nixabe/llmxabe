//! Anthropic's `/v1/messages` and `/v1/messages/count_tokens`.

use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::chat::{Content, Conversation, Turn, unsupported_role, unsupported_tools};
use super::error::{ApiError, Dialect, parse_body};
use super::generate::{Chunk, Finish, Generation, GenerationSpec};
use super::{AppState, sse_named};

const DIALECT: Dialect = Dialect::Anthropic;

/// Whether the caller asked for extended thinking.
#[derive(Debug, Deserialize)]
struct Thinking {
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Debug, Deserialize)]
struct Message {
    role: String,
    content: Content,
}

#[derive(Debug, Deserialize)]
struct MessagesRequest {
    messages: Vec<Message>,
    max_tokens: u32,
    #[serde(default)]
    system: Option<Content>,
    #[serde(default)]
    stop_sequences: Vec<String>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    thinking: Option<Thinking>,
    #[serde(default)]
    tools: Option<serde_json::Value>,
}

impl MessagesRequest {
    /// Extended thinking follows the server default unless the caller says
    /// otherwise. The shipped default is on, matching the model's own
    /// template.
    fn thinking_enabled(&self, default: bool) -> bool {
        self.thinking
            .as_ref()
            .map_or(default, |thinking| thinking.kind != "disabled")
    }

    fn conversation(&self) -> Result<Conversation, ApiError> {
        let mut conversation = Conversation {
            system: self
                .system
                .as_ref()
                .map(Content::text)
                .transpose()
                .map_err(|failure| ApiError::bad_request(DIALECT, failure))?,
            turns: Vec::with_capacity(self.messages.len()),
        };
        for message in &self.messages {
            let (text, thinking) = message
                .content
                .split()
                .map_err(|failure| ApiError::bad_request(DIALECT, failure))?;
            match message.role.as_str() {
                "user" => conversation.turns.push(Turn::User(text)),
                "assistant" => conversation.turns.push(Turn::Assistant {
                    reasoning: thinking,
                    content: text,
                }),
                role => return Err(unsupported_role(DIALECT, role)),
            }
        }
        if conversation.turns.is_empty() {
            return Err(ApiError::bad_request(
                DIALECT,
                "`messages` must contain at least one message",
            ));
        }
        Ok(conversation)
    }
}

fn stop_reason(finish: &Finish) -> &'static str {
    match finish {
        Finish::EndOfTurn => "end_turn",
        Finish::Length => "max_tokens",
        Finish::StopSequence(_) => "stop_sequence",
    }
}

fn stop_sequence(finish: &Finish) -> serde_json::Value {
    match finish {
        Finish::StopSequence(sequence) => json!(sequence),
        _ => serde_json::Value::Null,
    }
}

/// Turn a request into an admitted generation, shared by the streaming and
/// non-streaming paths so they cannot drift.
fn start(state: &AppState, request: &MessagesRequest) -> Result<Generation, ApiError> {
    if request.tools.is_some() {
        return Err(unsupported_tools(DIALECT));
    }
    let thinking = request.thinking_enabled(state.default_reasoning);
    let prompt = request.conversation()?.render(thinking);
    let encoding = state
        .tokenizer
        .encode(prompt, false)
        .map_err(|error| ApiError::bad_request(DIALECT, error.to_string()))?;
    let spec = GenerationSpec {
        prompt: encoding.get_ids().to_vec(),
        max_tokens: request.max_tokens,
        stop: request
            .stop_sequences
            .iter()
            .filter(|sequence| !sequence.is_empty())
            .cloned()
            .collect(),
        thinking,
        trim_spans: true,
    };
    Generation::start(state, spec, DIALECT)
}

pub(crate) async fn messages(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<Response, ApiError> {
    let request: MessagesRequest = parse_body(&body, DIALECT)?;
    let mut generation = start(&state, &request)?;
    let id = format!("msg_{}", generation.request_id());
    let model = state.model_name(request.model.clone());

    if !request.stream {
        let (reasoning, text) = generation.collect().await?;
        let finish = generation.finish();
        let mut content = Vec::with_capacity(2);
        if !reasoning.is_empty() {
            // The real API signs thinking blocks so they can be replayed;
            // there is nothing to verify here, and clients that round-trip a
            // block still need the field present.
            content.push(json!({ "type": "thinking", "thinking": reasoning, "signature": "" }));
        }
        content.push(json!({ "type": "text", "text": text }));
        return Ok(axum::Json(json!({
            "id": id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason(&finish),
            "stop_sequence": stop_sequence(&finish),
            "usage": {
                "input_tokens": generation.prompt_tokens(),
                "output_tokens": generation.completion_tokens(),
            },
        }))
        .into_response());
    }

    let stream = async_stream::stream! {
        yield Ok::<_, std::convert::Infallible>(sse_named("message_start", &json!({
            "type": "message_start",
            "message": {
                "id": id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": serde_json::Value::Null,
                "stop_sequence": serde_json::Value::Null,
                "usage": { "input_tokens": generation.prompt_tokens(), "output_tokens": 0 },
            },
        })));

        // Content blocks open on their first delta, so a response with no
        // reasoning never emits an empty thinking block.
        let mut index = 0usize;
        let mut open: Option<&'static str> = None;
        loop {
            let (kind, delta_kind, field, text) = match generation.next().await {
                Ok(Some(Chunk::Reasoning(text))) => ("thinking", "thinking_delta", "thinking", text),
                Ok(Some(Chunk::Text(text))) => ("text", "text_delta", "text", text),
                Ok(None) => break,
                Err(failure) => {
                    yield Ok(sse_named("error", &failure.payload()));
                    return;
                }
            };
            if open != Some(kind) {
                if open.is_some() {
                    yield Ok(sse_named("content_block_stop", &json!({
                        "type": "content_block_stop", "index": index,
                    })));
                    index += 1;
                }
                let block = if kind == "thinking" {
                    json!({ "type": "thinking", "thinking": "", "signature": "" })
                } else {
                    json!({ "type": "text", "text": "" })
                };
                yield Ok(sse_named("content_block_start", &json!({
                    "type": "content_block_start", "index": index, "content_block": block,
                })));
                open = Some(kind);
            }
            yield Ok(sse_named("content_block_delta", &json!({
                "type": "content_block_delta",
                "index": index,
                "delta": { "type": delta_kind, field: text },
            })));
        }

        // A message always carries at least one content block, even when the
        // model produced nothing.
        if open.is_none() {
            yield Ok(sse_named("content_block_start", &json!({
                "type": "content_block_start",
                "index": index,
                "content_block": { "type": "text", "text": "" },
            })));
        }
        yield Ok(sse_named("content_block_stop", &json!({
            "type": "content_block_stop", "index": index,
        })));

        let finish = generation.finish();
        yield Ok(sse_named("message_delta", &json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason(&finish),
                "stop_sequence": stop_sequence(&finish),
            },
            "usage": { "output_tokens": generation.completion_tokens() },
        })));
        yield Ok(sse_named("message_stop", &json!({ "type": "message_stop" })));
    };
    Ok(Sse::new(stream)
        .keep_alive(KeepAlive::default().event(Event::default().event("ping").data("{}")))
        .into_response())
}

#[derive(Serialize)]
pub(crate) struct TokenCount {
    input_tokens: usize,
}

/// Price a prompt without running it — the same templating and tokenization
/// the request itself would do, and nothing else.
pub(crate) async fn count_tokens(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<axum::Json<TokenCount>, ApiError> {
    let request: MessagesRequest = parse_body(&body, DIALECT)?;
    if request.tools.is_some() {
        return Err(unsupported_tools(DIALECT));
    }
    let prompt = request
        .conversation()?
        .render(request.thinking_enabled(state.default_reasoning));
    let encoding = state
        .tokenizer
        .encode(prompt, false)
        .map_err(|error| ApiError::bad_request(DIALECT, error.to_string()))?;
    Ok(axum::Json(TokenCount {
        input_tokens: encoding.get_ids().len(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(body: &str) -> MessagesRequest {
        serde_json::from_str(body).expect("test request should parse")
    }

    #[test]
    fn a_system_string_and_a_system_block_list_are_the_same_thing() {
        // `render` takes the resolved flag, so pass it explicitly here.
        let plain = request(
            r#"{"max_tokens":16,"system":"Be terse.","messages":[{"role":"user","content":"Hi"}]}"#,
        );
        let blocks = request(
            r#"{"max_tokens":16,"system":[{"type":"text","text":"Be terse."}],
                "messages":[{"role":"user","content":[{"type":"text","text":"Hi"}]}]}"#,
        );
        assert_eq!(
            plain.conversation().unwrap().render(true),
            blocks.conversation().unwrap().render(true)
        );
    }

    #[test]
    fn a_silent_request_follows_the_server_default() {
        let silent = request(r#"{"max_tokens":1,"messages":[]}"#);
        assert!(silent.thinking_enabled(true));
        assert!(!silent.thinking_enabled(false));
    }

    #[test]
    fn a_request_that_names_a_mode_overrides_the_server_default() {
        // Either way round: `--no-reasoning` must not silently ignore a
        // caller that asked for thinking, and neither must the reverse.
        let on = request(r#"{"max_tokens":1,"messages":[],"thinking":{"type":"enabled"}}"#);
        let off = request(r#"{"max_tokens":1,"messages":[],"thinking":{"type":"disabled"}}"#);
        assert!(on.thinking_enabled(false));
        assert!(!off.thinking_enabled(true));
    }

    #[test]
    fn an_empty_conversation_is_refused() {
        assert!(
            request(r#"{"max_tokens":1,"messages":[]}"#)
                .conversation()
                .is_err()
        );
    }

    #[test]
    fn a_stop_sequence_is_reported_as_its_own_reason() {
        let finish = Finish::StopSequence("END".to_owned());
        assert_eq!(stop_reason(&finish), "stop_sequence");
        assert_eq!(stop_sequence(&finish), json!("END"));
        assert_eq!(stop_reason(&Finish::EndOfTurn), "end_turn");
        assert_eq!(stop_sequence(&Finish::EndOfTurn), serde_json::Value::Null);
        assert_eq!(stop_reason(&Finish::Length), "max_tokens");
    }
}
