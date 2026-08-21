//! Anthropic's `/v1/messages` and `/v1/messages/count_tokens`.

use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::chat::{Content, Conversation, Turn, unsupported_role, unsupported_tool_choice};
use super::error::{ApiError, Dialect, parse_body};
use super::generate::{Chunk, Finish, Generation, GenerationSpec, resolve_sampling};
use super::tools::{ToolCallParser, ToolDefinition};
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
    tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<u32>,
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

    /// Whether the caller's `tool_choice` lets the tools be offered at all.
    fn tools_offered(&self) -> Result<bool, ApiError> {
        match self
            .tool_choice
            .as_ref()
            .and_then(|choice| choice.get("type"))
            .and_then(serde_json::Value::as_str)
        {
            None | Some("auto") => Ok(true),
            Some("none") => Ok(false),
            Some(choice) => Err(unsupported_tool_choice(DIALECT, choice)),
        }
    }

    fn tool_definitions(&self) -> Result<Vec<ToolDefinition>, ApiError> {
        self.tools
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(ToolDefinition::from_anthropic)
            .collect::<Result<_, _>>()
            .map_err(|failure| ApiError::bad_request(DIALECT, failure))
    }

    fn sampling(&self, state: &AppState) -> Result<xabe_engine::SamplingParams, ApiError> {
        resolve_sampling(
            DIALECT,
            state.default_temperature,
            self.temperature,
            self.top_p,
            self.top_k,
            None,
        )
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
            tools: Vec::new(),
        };
        for message in &self.messages {
            let folded = message
                .content
                .fold()
                .map_err(|failure| ApiError::bad_request(DIALECT, failure))?;
            match message.role.as_str() {
                "user" => {
                    // Tool results render first, as `tool` turns; any user
                    // text alongside them becomes its own turn after. A
                    // message that was only tool results adds no text turn.
                    let only_results = !folded.tool_results.is_empty() && folded.text.is_empty();
                    for result in folded.tool_results {
                        conversation.push_tool_result(result);
                    }
                    if !only_results {
                        conversation.turns.push(Turn::User(folded.text));
                    }
                }
                "assistant" => {
                    if !folded.tool_results.is_empty() {
                        return Err(ApiError::bad_request(
                            DIALECT,
                            "`tool_result` blocks belong in user messages",
                        ));
                    }
                    conversation.turns.push(Turn::Assistant {
                        reasoning: folded.thinking,
                        content: folded.text,
                        tool_calls: folded.tool_calls,
                    });
                }
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

fn stop_reason(finish: &Finish, tool_calls: usize) -> &'static str {
    match finish {
        Finish::EndOfTurn if tool_calls > 0 => "tool_use",
        Finish::EndOfTurn => "end_turn",
        Finish::Length => "max_tokens",
        Finish::StopSequence(_) => "stop_sequence",
    }
}

/// The id a served `tool_use` block carries.
fn tool_use_id(request_id: u64, index: usize) -> String {
    format!("toolu_{request_id}_{index}")
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
    let thinking = request.thinking_enabled(state.default_reasoning);
    let mut conversation = request.conversation()?;
    if request.tools_offered()? {
        conversation.tools = request.tool_definitions()?;
    }
    let tool_parser =
        (!conversation.tools.is_empty()).then(|| ToolCallParser::new(&conversation.tools));
    let prompt = conversation.render(thinking);
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
        sampling: request.sampling(state)?,
        tool_parser,
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
        let collected = generation.collect().await?;
        let finish = generation.finish();
        let mut content = Vec::with_capacity(2 + collected.tool_calls.len());
        if !collected.reasoning.is_empty() {
            // The real API signs thinking blocks so they can be replayed;
            // there is nothing to verify here, and clients that round-trip a
            // block still need the field present.
            content.push(
                json!({ "type": "thinking", "thinking": collected.reasoning, "signature": "" }),
            );
        }
        if !collected.text.is_empty() || collected.tool_calls.is_empty() {
            content.push(json!({ "type": "text", "text": collected.text }));
        }
        let request_id = generation.request_id();
        for (index, call) in collected.tool_calls.iter().enumerate() {
            content.push(json!({
                "type": "tool_use",
                "id": tool_use_id(request_id, index),
                "name": call.name,
                "input": call.arguments,
            }));
        }
        return Ok(axum::Json(json!({
            "id": id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason(&finish, generation.tool_call_count()),
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
        let request_id = generation.request_id();
        let mut index = 0usize;
        let mut open: Option<&'static str> = None;
        let mut emitted_calls = 0usize;
        loop {
            let chunk = match generation.next().await {
                Ok(Some(chunk)) => chunk,
                Ok(None) => break,
                Err(failure) => {
                    yield Ok(sse_named("error", &failure.payload()));
                    return;
                }
            };
            if let Chunk::ToolCall(call) = chunk {
                // A call parses only once its block is complete, so it
                // streams as one self-contained tool_use block: start, one
                // input_json_delta with the whole input, stop.
                if open.take().is_some() {
                    yield Ok(sse_named("content_block_stop", &json!({
                        "type": "content_block_stop", "index": index,
                    })));
                    index += 1;
                }
                yield Ok(sse_named("content_block_start", &json!({
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tool_use_id(request_id, emitted_calls),
                        "name": call.name,
                        "input": {},
                    },
                })));
                yield Ok(sse_named("content_block_delta", &json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": { "type": "input_json_delta", "partial_json": call.arguments_json() },
                })));
                yield Ok(sse_named("content_block_stop", &json!({
                    "type": "content_block_stop", "index": index,
                })));
                index += 1;
                emitted_calls += 1;
                continue;
            }
            let (kind, delta_kind, field, text) = match chunk {
                Chunk::Reasoning(text) => ("thinking", "thinking_delta", "thinking", text),
                Chunk::Text(text) => ("text", "text_delta", "text", text),
                Chunk::ToolCall(_) => unreachable!("handled above"),
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
        // model produced nothing at all.
        if open.is_none() && index == 0 {
            yield Ok(sse_named("content_block_start", &json!({
                "type": "content_block_start",
                "index": index,
                "content_block": { "type": "text", "text": "" },
            })));
            open = Some("text");
        }
        if open.is_some() {
            yield Ok(sse_named("content_block_stop", &json!({
                "type": "content_block_stop", "index": index,
            })));
        }

        let finish = generation.finish();
        yield Ok(sse_named("message_delta", &json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason(&finish, generation.tool_call_count()),
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
    let mut conversation = request.conversation()?;
    if request.tools_offered()? {
        conversation.tools = request.tool_definitions()?;
    }
    let prompt = conversation.render(request.thinking_enabled(state.default_reasoning));
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
        assert_eq!(stop_reason(&finish, 0), "stop_sequence");
        assert_eq!(stop_sequence(&finish), json!("END"));
        assert_eq!(stop_reason(&Finish::EndOfTurn, 0), "end_turn");
        assert_eq!(stop_sequence(&Finish::EndOfTurn), serde_json::Value::Null);
        assert_eq!(stop_reason(&Finish::Length, 0), "max_tokens");
    }

    #[test]
    fn a_turn_that_called_tools_stops_with_tool_use() {
        assert_eq!(stop_reason(&Finish::EndOfTurn, 1), "tool_use");
        // A truncated call list is a truncation, not a tool_use turn.
        assert_eq!(stop_reason(&Finish::Length, 1), "max_tokens");
    }

    #[test]
    fn tool_use_and_tool_result_blocks_fold_into_the_conversation() {
        let request = request(
            r#"{"max_tokens":16,"messages":[
                {"role":"user","content":"Weather?"},
                {"role":"assistant","content":[
                    {"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Paris"}}]},
                {"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"toolu_1","content":"Sunny"}]}
            ]}"#,
        );
        let conversation = request.conversation().expect("tool turns fold");
        assert_eq!(conversation.turns.len(), 3);
        assert!(matches!(
            &conversation.turns[1],
            Turn::Assistant { tool_calls, .. } if tool_calls.len() == 1
        ));
        assert!(matches!(
            &conversation.turns[2],
            Turn::ToolResults(results) if results == &["Sunny".to_owned()]
        ));
    }

    #[test]
    fn forcing_a_tool_is_refused_but_auto_and_none_are_honoured() {
        let auto = request(r#"{"max_tokens":1,"messages":[],"tool_choice":{"type":"auto"}}"#);
        let none = request(r#"{"max_tokens":1,"messages":[],"tool_choice":{"type":"none"}}"#);
        let any = request(r#"{"max_tokens":1,"messages":[],"tool_choice":{"type":"any"}}"#);
        assert!(auto.tools_offered().expect("auto offers"));
        assert!(!none.tools_offered().expect("none withholds"));
        assert!(any.tools_offered().is_err());
    }
}
