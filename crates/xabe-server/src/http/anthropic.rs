//! Anthropic's `/v1/messages` and `/v1/messages/count_tokens`.

use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::chat::{
    Content, Conversation, Turn, image_misplaced, unsupported_role, unsupported_tool_choice,
};
use super::error::{ApiError, Dialect, parse_body};
use super::generate::{Chunk, Collected, Finish, Generation, GenerationSpec, resolve_sampling};
use super::tools::{OfferedTools, ToolCallParser};
use super::warn_unsupported;
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
    /// Not Anthropic's field, but llama.cpp-aimed clients send it here too.
    #[serde(default)]
    min_p: Option<f32>,
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
    ///
    /// Anthropic spells the choice as an object, but a harness pointing an
    /// OpenAI-shaped client at this endpoint sends the bare string — the same
    /// reason `min_p` is accepted here. Reading only an object's `type` meant
    /// a string fell through to the absent arm and was silently taken as
    /// `auto`: `"none"` offered the tools it asked this server not to offer,
    /// and `"required"` was accepted as a guarantee nothing here keeps.
    fn tools_offered(&self) -> Result<bool, ApiError> {
        let choice = match self.tool_choice.as_ref() {
            None => return Ok(true),
            Some(Value::String(choice)) => choice.as_str(),
            Some(choice) => choice.get("type").and_then(Value::as_str).unwrap_or(""),
        };
        match choice {
            "" | "auto" => Ok(true),
            "none" => Ok(false),
            choice => Err(unsupported_tool_choice(DIALECT, choice)),
        }
    }

    fn tool_definitions(&self) -> Result<OfferedTools, ApiError> {
        OfferedTools::from_anthropic(self.tools.as_deref().unwrap_or_default())
            .map_err(|failure| ApiError::bad_request(DIALECT, failure))
    }

    fn sampling(&self, state: &AppState) -> Result<xabe_engine::SamplingParams, ApiError> {
        resolve_sampling(
            DIALECT,
            state.sampling_defaults,
            self.temperature,
            self.top_p,
            self.top_k,
            self.min_p,
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
            images: Vec::new(),
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
                        conversation.images.extend(folded.images);
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
                    if !folded.images.is_empty() {
                        return Err(ApiError::bad_request(DIALECT, image_misplaced()));
                    }
                    conversation.turns.push(Turn::Assistant {
                        reasoning: folded.thinking,
                        content: folded.text,
                        tool_calls: folded.tool_calls,
                    });
                }
                "tool" => {
                    for result in folded.tool_results {
                        conversation.push_tool_result(result);
                    }
                    if !folded.text.is_empty() {
                        conversation.push_tool_result(folded.text);
                    }
                }
                "system" | "developer" => {
                    if conversation.turns.is_empty() {
                        let sys = conversation.system.get_or_insert_with(String::new);
                        if !sys.is_empty() {
                            sys.push('\n');
                        }
                        sys.push_str(&folded.text);
                    } else {
                        conversation.turns.push(Turn::System(folded.text));
                    }
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

/// The `content` array of a non-streaming message.
///
/// Split out from the handler so the block-shaping rules are testable without
/// a live generation; they are the rules clients actually break on.
fn content_blocks(collected: &Collected, request_id: u64) -> Vec<Value> {
    let mut content = Vec::with_capacity(2 + collected.tool_calls.len());
    if !collected.reasoning.is_empty() {
        content.push(json!({
            "type": "thinking",
            "thinking": collected.reasoning,
            "signature": thinking_signature(request_id),
        }));
    }
    // Only when it has something in it. An empty text block is not something
    // the real API emits, and a client that replays this message into its
    // next turn gets a 400 for it — text blocks must be non-empty on the way
    // in. Cutting off mid-thought produced exactly that: a thinking block
    // followed by an empty text block.
    if !collected.text.is_empty() {
        content.push(json!({ "type": "text", "text": collected.text }));
    }
    for (index, call) in collected.tool_calls.iter().enumerate() {
        content.push(json!({
            "type": "tool_use",
            "id": tool_use_id(request_id, index),
            "name": call.name,
            "input": call.arguments,
        }));
    }
    // Nothing at all is degenerate, but `content: []` is worse than one empty
    // block for clients that index the first one.
    if content.is_empty() {
        content.push(json!({ "type": "text", "text": "" }));
    }
    content
}

/// The signature a served `thinking` block carries.
///
/// The real API signs these so a replayed block can be verified as its own.
/// There is nothing to verify here, but the field may not be empty: the spec
/// requires a non-empty signature, clients replay the block unmodified, and a
/// validating one rejects `""`. So it is opaque, stable for a message, and
/// ignored on the way back in.
fn thinking_signature(request_id: u64) -> String {
    format!("xabe-unverified-{request_id}")
}

/// The id a served `tool_use` block carries.
fn tool_use_id(request_id: u64, index: usize) -> String {
    format!("toolu_{request_id}_{index}")
}

/// The events that close whatever content block is open, if any.
///
/// Anthropic signs a thinking block on the way out, as a `signature_delta`
/// just before the block closes. Clients accumulate it and replay the block
/// with it attached, so a block that closes without one arrives back
/// unsigned — and a harness that drops unsigned thinking drops the model's
/// chain exactly where an agentic loop needs it, across a tool call.
///
/// One function rather than a copy at each of the three places a block
/// closes: the copy on the tool-call path consumed `open` before testing it,
/// so that path — a thinking block closed by a tool call, which is every
/// reasoning turn that calls a tool — was the one case that never signed.
fn close_block(open: Option<&str>, index: usize, request_id: u64) -> Vec<Value> {
    let Some(kind) = open else {
        return Vec::new();
    };
    let mut events = Vec::with_capacity(2);
    if kind == "thinking" {
        events.push(json!({
            "type": "content_block_delta",
            "index": index,
            "delta": {
                "type": "signature_delta",
                "signature": thinking_signature(request_id),
            },
        }));
    }
    events.push(json!({ "type": "content_block_stop", "index": index }));
    events
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
        let offered = request.tool_definitions()?;
        warn_unsupported(&offered);
        conversation.tools = offered.definitions;
    }
    let tool_parser =
        (!conversation.tools.is_empty()).then(|| ToolCallParser::new(&conversation.tools));
    let prompt = conversation.render(thinking);
    let encoding = state
        .tokenizer
        .encode(prompt, false)
        .map_err(|error| ApiError::bad_request(DIALECT, error.to_string()))?;
    let (prompt, images) = super::vision::expand_images(
        state.vision.as_deref(),
        DIALECT,
        encoding.get_ids().to_vec(),
        &conversation.images,
    )?;
    let spec = GenerationSpec {
        prompt,
        images,
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
        let content = content_blocks(&collected, generation.request_id());
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
                if open.is_some() {
                    for event in close_block(open.take(), index, request_id) {
                        yield Ok(sse_named(
                            event["type"].as_str().expect("close_block sets `type`"),
                            &event,
                        ));
                    }
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
                    for event in close_block(open.take(), index, request_id) {
                        yield Ok(sse_named(
                            event["type"].as_str().expect("close_block sets `type`"),
                            &event,
                        ));
                    }
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
        for event in close_block(open.take(), index, request_id) {
            yield Ok(sse_named(
                event["type"].as_str().expect("close_block sets `type`"),
                &event,
            ));
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
        let offered = request.tool_definitions()?;
        warn_unsupported(&offered);
        conversation.tools = offered.definitions;
    }
    let prompt = conversation.render(request.thinking_enabled(state.default_reasoning));
    let encoding = state
        .tokenizer
        .encode(prompt, false)
        .map_err(|error| ApiError::bad_request(DIALECT, error.to_string()))?;
    Ok(axum::Json(TokenCount {
        input_tokens: super::vision::expanded_token_count(
            state.vision.as_deref(),
            DIALECT,
            encoding.get_ids().len(),
            &conversation.images,
        )?,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::tools::ParsedToolCall;

    fn collected(reasoning: &str, text: &str, calls: Vec<ParsedToolCall>) -> Collected {
        Collected {
            reasoning: reasoning.to_owned(),
            text: text.to_owned(),
            tool_calls: calls,
        }
    }

    fn call(name: &str) -> ParsedToolCall {
        let mut arguments = serde_json::Map::new();
        arguments.insert("city".to_owned(), Value::String("Paris".to_owned()));
        ParsedToolCall {
            name: name.to_owned(),
            arguments,
        }
    }

    #[test]
    fn a_reply_cut_off_mid_thought_carries_no_empty_text_block() {
        // The regression. Hitting max_tokens while still thinking used to
        // emit a thinking block plus `{"type":"text","text":""}`. A client
        // that replays that assistant turn is sending an empty text block
        // back, which the API rejects — so the harness breaks on its *next*
        // request, not this one.
        let blocks = content_blocks(&collected("still thinking", "", vec![]), 7);
        assert_eq!(
            blocks.len(),
            1,
            "expected only the thinking block: {blocks:?}"
        );
        assert_eq!(blocks[0]["type"], "thinking");
    }

    #[test]
    fn a_thinking_block_is_signed_and_the_signature_is_never_empty() {
        // The spec requires a non-empty signature and clients replay the
        // block unmodified; a validating one rejects "".
        let blocks = content_blocks(&collected("thought", "answer", vec![]), 7);
        let signature = blocks[0]["signature"]
            .as_str()
            .expect("signature is a string");
        assert!(
            !signature.is_empty(),
            "thinking blocks must carry a signature"
        );
    }

    #[test]
    fn tool_calls_follow_the_text_and_do_not_drag_an_empty_block_with_them() {
        let blocks = content_blocks(&collected("", "", vec![call("get_weather")]), 7);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "tool_use");
        assert_eq!(blocks[0]["name"], "get_weather");
        assert_eq!(blocks[0]["input"]["city"], "Paris");
    }

    #[test]
    fn a_reply_with_nothing_in_it_still_has_one_block() {
        // `content: []` breaks clients that index the first block, so the
        // degenerate case keeps one.
        let blocks = content_blocks(&collected("", "", vec![]), 7);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "text");
    }

    #[test]
    fn stop_reason_is_tool_use_only_when_a_call_was_made() {
        assert_eq!(stop_reason(&Finish::EndOfTurn, 1), "tool_use");
        assert_eq!(stop_reason(&Finish::EndOfTurn, 0), "end_turn");
        assert_eq!(stop_reason(&Finish::Length, 1), "max_tokens");
    }

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
    fn a_single_system_object_in_json_and_role_system_in_messages_are_both_accepted() {
        let single_obj = request(
            r#"{"max_tokens":16,"system":{"type":"text","text":"Be terse."},"messages":[{"role":"user","content":"Hi"}]}"#,
        );
        let in_messages = request(
            r#"{"max_tokens":16,"messages":[{"role":"system","content":"Be terse."},{"role":"user","content":"Hi"}]}"#,
        );
        let rendered_single = single_obj.conversation().unwrap().render(true);
        let rendered_in_msg = in_messages.conversation().unwrap().render(true);
        assert!(rendered_single.contains("<|im_start|>system\nBe terse.<|im_end|>\n"));
        assert!(rendered_in_msg.contains("<|im_start|>system\nBe terse.<|im_end|>\n"));
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

    #[test]
    fn a_bare_string_tool_choice_is_read_rather_than_ignored() {
        // The regression. Only an object's `type` was read, so the string
        // form an OpenAI-shaped client sends fell through to the absent arm
        // and became `auto`: `"none"` offered the tools it asked this server
        // to withhold, and `"required"` was answered 200 as a guarantee
        // nothing here keeps.
        let auto = request(r#"{"max_tokens":1,"messages":[],"tool_choice":"auto"}"#);
        let none = request(r#"{"max_tokens":1,"messages":[],"tool_choice":"none"}"#);
        let required = request(r#"{"max_tokens":1,"messages":[],"tool_choice":"required"}"#);
        assert!(auto.tools_offered().expect("auto offers"));
        assert!(!none.tools_offered().expect("none withholds"));
        assert!(
            required.tools_offered().is_err(),
            "forcing is still refused"
        );
    }

    #[test]
    fn a_thinking_block_closed_by_a_tool_call_is_still_signed() {
        // The regression, and the reason this is one function rather than a
        // copy at each of the three places a block closes: the copy on the
        // tool-call path had already consumed `open` when it tested it, so
        // the signature was skipped on exactly the path that carries it —
        // every reasoning turn that ends in a tool call. A client that keeps
        // only signed thinking then replays the turn without it, and the
        // model loses its chain across the tool boundary.
        let events = close_block(Some("thinking"), 0, 7);
        assert_eq!(events.len(), 2, "sign, then stop: {events:?}");
        assert_eq!(events[0]["delta"]["type"], "signature_delta");
        assert_eq!(events[0]["delta"]["signature"], thinking_signature(7));
        assert_eq!(events[0]["index"], 0);
        assert_eq!(events[1]["type"], "content_block_stop");
        assert_eq!(events[1]["index"], 0);
    }

    #[test]
    fn only_a_thinking_block_is_signed_and_a_closed_one_emits_nothing() {
        let text = close_block(Some("text"), 3, 7);
        assert_eq!(text.len(), 1, "a text block only stops: {text:?}");
        assert_eq!(text[0]["type"], "content_block_stop");
        assert_eq!(text[0]["index"], 3);
        assert!(
            close_block(None, 0, 7).is_empty(),
            "nothing open, nothing to close"
        );
    }

    #[test]
    fn every_close_block_event_names_itself_in_type() {
        // The stream names each SSE event after this field, so a payload
        // without one would be yielded under a panicking `expect`.
        for open in [Some("thinking"), Some("text")] {
            for event in close_block(open, 0, 1) {
                assert!(
                    event.get("type").and_then(Value::as_str).is_some(),
                    "no `type` in {event:?}"
                );
            }
        }
    }
}
