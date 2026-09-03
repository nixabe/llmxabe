//! OpenAI's `/v1/completions` and `/v1/chat/completions`.

use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use super::chat::{
    Content, Conversation, Turn, image_misplaced, unsupported_role, unsupported_tool_choice,
};
use super::error::{ApiError, Dialect, parse_body};
use super::generate::{Chunk, Finish, Generation, GenerationSpec, resolve_sampling};
use super::tools::{OfferedTools, ParsedToolCall, ToolCallParser};
use super::warn_unsupported;
use super::{AppState, sse_json, unix_now};

const DIALECT: Dialect = Dialect::OpenAi;

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

/// As [`finish_reason`], for a chat response that may have called tools: a
/// turn the model ended after calling tools is `"tool_calls"`, while a
/// truncated one stays `"length"` — the caller must know the call list may
/// be incomplete.
fn chat_finish_reason(finish: &Finish, tool_calls: usize) -> &'static str {
    match finish {
        Finish::EndOfTurn if tool_calls > 0 => "tool_calls",
        _ => finish_reason(finish),
    }
}

/// The wire shape of one emitted call, shared by the message and the delta.
fn tool_call_value(request_id: u64, index: usize, call: &ParsedToolCall) -> Value {
    json!({
        "id": format!("call_{request_id}_{index}"),
        "type": "function",
        "function": { "name": call.name, "arguments": call.arguments_json() },
    })
}

/// Fold a replayed `tool_calls` array back into parsed calls for the
/// template. OpenAI carries `arguments` as a JSON-encoded string.
fn replayed_tool_calls(raw: Option<Vec<Value>>) -> Result<Vec<ParsedToolCall>, ApiError> {
    let mut calls = Vec::new();
    for value in raw.unwrap_or_default() {
        let function = value.get("function").unwrap_or(&value);
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| {
                ApiError::bad_request(DIALECT, "every replayed tool call needs a function `name`")
            })?;
        let arguments = match function.get("arguments") {
            None | Some(Value::Null) => Map::new(),
            Some(Value::String(text)) if text.trim().is_empty() => Map::new(),
            Some(Value::String(text)) => serde_json::from_str::<Value>(text)
                .ok()
                .and_then(|parsed| parsed.as_object().cloned())
                .ok_or_else(|| {
                    ApiError::bad_request(
                        DIALECT,
                        format!("tool call `{name}` has `arguments` that are not a JSON object"),
                    )
                })?,
            Some(Value::Object(object)) => object.clone(),
            Some(_) => {
                return Err(ApiError::bad_request(
                    DIALECT,
                    format!("tool call `{name}` has `arguments` that are not a JSON object"),
                ));
            }
        };
        calls.push(ParsedToolCall {
            name: name.to_owned(),
            arguments,
        });
    }
    Ok(calls)
}

/// Whether to offer the tools to the model at all.
fn tools_offered(choice: Option<&Value>) -> Result<bool, ApiError> {
    match choice {
        None => Ok(true),
        Some(Value::String(choice)) if choice == "auto" => Ok(true),
        Some(Value::String(choice)) if choice == "none" => Ok(false),
        Some(Value::String(choice)) => Err(unsupported_tool_choice(DIALECT, choice)),
        Some(_) => Err(unsupported_tool_choice(DIALECT, "a named function")),
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
/// A caller that asked for four completions and got one has been answered
/// wrongly, not approximately.
fn reject_shape_changing(n: Option<u32>, best_of: Option<u32>) -> Result<(), ApiError> {
    for (name, value) in [("n", n), ("best_of", best_of)] {
        if value.is_some_and(|value| value > 1) {
            return Err(ApiError::bad_request(
                DIALECT,
                format!(
                    "`{name}` above 1 is not supported: this server generates one completion \
                     per request"
                ),
            ));
        }
    }
    Ok(())
}

/// The sampling fields both OpenAI request shapes carry. `top_k` and
/// `min_p` are not OpenAI's, but clients aimed at llama.cpp and vLLM send
/// them and both honour them, so refusing them would break those clients for
/// no gain.
#[derive(Debug, Default, Deserialize)]
struct SamplingFields {
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    seed: Option<i64>,
}

impl SamplingFields {
    fn resolve(&self, state: &AppState) -> Result<xabe_engine::SamplingParams, ApiError> {
        resolve_sampling(
            DIALECT,
            state.sampling_defaults,
            self.temperature,
            self.top_p,
            self.top_k,
            self.min_p,
            self.seed,
        )
    }
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
    #[serde(default)]
    max_tokens: Option<u32>,
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
    #[serde(flatten)]
    sampling: SamplingFields,
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
        // A raw completion has no content parts to carry an image in.
        images: Vec::new(),
        max_tokens: request.max_tokens.unwrap_or(state.default_max_tokens),
        stop: stop_sequences(request.stop),
        thinking: false,
        trim_spans: false,
        sampling: request.sampling.resolve(&state)?,
        tool_parser: None,
    };
    let mut generation = Generation::start(&state, spec, DIALECT)?;
    let id = format!("cmpl-{}", generation.request_id());
    let model = state.model_name(request.model);
    let created = unix_now();

    if !request.stream {
        let text = generation.collect().await?.text;
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
                // No tool parser is attached to a raw completion, so a
                // `ToolCall` chunk cannot arrive here.
                Ok(Some(Chunk::ToolCall(_))) => unreachable!("completions attach no tool parser"),
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
    /// Calls replayed from a previous assistant turn.
    #[serde(default)]
    tool_calls: Option<Vec<Value>>,
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
    tools: Option<Vec<Value>>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    chat_template_kwargs: Option<ChatTemplateKwargs>,
    #[serde(flatten)]
    sampling: SamplingFields,
}

/// Fold OpenAI chat messages into a conversation.
fn conversation(messages: Vec<ChatMessage>) -> Result<Conversation, ApiError> {
    let mut conversation = Conversation::default();
    let mut system = Vec::new();
    for message in messages {
        let folded = message
            .content
            .as_ref()
            .map(Content::fold)
            .transpose()
            .map_err(|failure| ApiError::bad_request(DIALECT, failure))?
            .unwrap_or_default();
        // Replayed tool calls and results may arrive either via dedicated
        // fields/roles or folded into content blocks.
        for result in folded.tool_results {
            conversation.push_tool_result(result);
        }
        // Images render as user-turn markup and nothing else.
        if !folded.images.is_empty() && message.role != "user" {
            return Err(ApiError::bad_request(DIALECT, image_misplaced()));
        }
        let (text, thinking) = (folded.text, folded.thinking);
        match message.role.as_str() {
            "system" | "developer" => {
                if conversation.turns.is_empty() {
                    system.push(text);
                } else {
                    conversation.turns.push(Turn::System(text));
                }
            }
            "user" => {
                conversation.images.extend(folded.images);
                if !text.is_empty() {
                    conversation.turns.push(Turn::User(text));
                }
            }
            "assistant" => {
                let mut tool_calls = replayed_tool_calls(message.tool_calls)?;
                tool_calls.extend(folded.tool_calls);
                conversation.turns.push(Turn::Assistant {
                    reasoning: message.reasoning_content.unwrap_or(thinking),
                    content: text,
                    tool_calls,
                });
            }
            "tool" => conversation.push_tool_result(text),
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
    /// Two levels because the field has three states on the wire and they
    /// are not interchangeable: absent (a streaming delta that carries no
    /// text), `null` (a message whose whole turn was tool calls — what
    /// OpenAI sends, and what a client deserializing `content` as a required
    /// nullable field needs to see), and a string.
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<Option<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<Value>>,
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
    reject_shape_changing(request.n, None)?;
    let thinking = request
        .chat_template_kwargs
        .as_ref()
        .and_then(|kwargs| kwargs.enable_thinking)
        .unwrap_or(state.default_reasoning);
    let mut conversation = conversation(request.messages)?;
    if tools_offered(request.tool_choice.as_ref())? {
        let offered = OfferedTools::from_openai(request.tools.as_deref().unwrap_or_default())
            .map_err(|failure| ApiError::bad_request(DIALECT, failure))?;
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
        max_tokens: request
            .max_completion_tokens
            .or(request.max_tokens)
            .unwrap_or(state.default_max_tokens),
        stop: stop_sequences(request.stop),
        thinking,
        trim_spans: true,
        sampling: request.sampling.resolve(&state)?,
        tool_parser,
    };
    let mut generation = Generation::start(&state, spec, DIALECT)?;
    let id = format!("chatcmpl-{}", generation.request_id());
    let model = state.model_name(request.model);
    let created = unix_now();

    if !request.stream {
        let collected = generation.collect().await?;
        let request_id = generation.request_id();
        let tool_calls = (!collected.tool_calls.is_empty()).then(|| {
            collected
                .tool_calls
                .iter()
                .enumerate()
                .map(|(index, call)| tool_call_value(request_id, index, call))
                .collect()
        });
        return Ok(axum::Json(ChatResponse {
            id,
            object: "chat.completion",
            created,
            model,
            choices: vec![ChatChoice {
                index: 0,
                message: Some(ChatDelta {
                    role: Some("assistant"),
                    // `null` content alongside tool calls is OpenAI's shape
                    // for a turn that only called tools.
                    content: Some(
                        (tool_calls.is_none() || !collected.text.is_empty())
                            .then_some(collected.text),
                    ),
                    reasoning_content: (!collected.reasoning.is_empty())
                        .then_some(collected.reasoning),
                    tool_calls,
                }),
                delta: None,
                finish_reason: Some(chat_finish_reason(
                    &generation.finish(),
                    generation.tool_call_count(),
                )),
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
            ChatDelta { role: Some("assistant"), content: Some(Some(String::new())), ..ChatDelta::default() },
            None,
            None,
        ));
        let request_id = generation.request_id();
        let mut emitted_calls = 0usize;
        loop {
            match generation.next().await {
                Ok(Some(Chunk::Text(text))) => {
                    yield Ok(chunk(ChatDelta { content: Some(Some(text)), ..ChatDelta::default() }, None, None));
                }
                Ok(Some(Chunk::Reasoning(text))) => {
                    yield Ok(chunk(
                        ChatDelta { reasoning_content: Some(text), ..ChatDelta::default() },
                        None,
                        None,
                    ));
                }
                Ok(Some(Chunk::ToolCall(call))) => {
                    // A call parses only once its block is complete, so it
                    // streams as one delta: the entry, name and arguments
                    // together, at the index clients accumulate by.
                    let mut entry = tool_call_value(request_id, emitted_calls, &call);
                    entry["index"] = json!(emitted_calls);
                    emitted_calls += 1;
                    yield Ok(chunk(
                        ChatDelta { tool_calls: Some(vec![entry]), ..ChatDelta::default() },
                        None,
                        None,
                    ));
                }
                Ok(None) => {
                    yield Ok(chunk(
                        ChatDelta::default(),
                        Some(chat_finish_reason(&generation.finish(), generation.tool_call_count())),
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
            tool_calls: None,
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
    fn a_late_system_message_becomes_a_system_turn() {
        let conv = conversation(vec![message("user", "Hi"), message("system", "Be terse.")])
            .expect("a system message after a turn should become a system turn");
        assert_eq!(conv.turns.len(), 2);
        assert!(
            conv.render(true)
                .contains("<|im_start|>system\nBe terse.<|im_end|>\n")
        );
    }

    #[test]
    fn a_tool_message_becomes_a_tool_response_turn() {
        let folded = conversation(vec![
            message("user", "Weather?"),
            message("assistant", "checking"),
            message("tool", "Sunny"),
            message("tool", "Windy"),
        ])
        .expect("tool messages fold");
        // Consecutive tool messages share one turn, as the template merges
        // them into one user turn of <tool_response> blocks.
        assert_eq!(folded.turns.len(), 3);
        assert!(matches!(
            &folded.turns[2],
            Turn::ToolResults(results) if results == &["Sunny".to_owned(), "Windy".to_owned()]
        ));
    }

    #[test]
    fn replayed_tool_calls_parse_string_and_object_arguments() {
        let calls = replayed_tool_calls(Some(vec![
            json!({ "id": "call_1", "type": "function",
                    "function": { "name": "f", "arguments": r#"{"city":"Paris"}"# } }),
            json!({ "function": { "name": "g", "arguments": { "days": 3 } } }),
        ]))
        .expect("well-formed replays parse");
        assert_eq!(calls[0].name, "f");
        assert_eq!(calls[0].arguments.get("city"), Some(&json!("Paris")));
        assert_eq!(calls[1].arguments.get("days"), Some(&json!(3)));

        assert!(replayed_tool_calls(Some(vec![json!({ "function": {} })])).is_err());
        assert!(
            replayed_tool_calls(Some(vec![
                json!({ "function": { "name": "f", "arguments": "not json" } })
            ]))
            .is_err()
        );
    }

    #[test]
    fn tool_choice_auto_and_none_work_and_forcing_is_refused() {
        assert!(tools_offered(None).expect("default is auto"));
        assert!(tools_offered(Some(&json!("auto"))).expect("auto offers"));
        assert!(!tools_offered(Some(&json!("none"))).expect("none withholds"));
        assert!(tools_offered(Some(&json!("required"))).is_err());
        assert!(
            tools_offered(Some(
                &json!({ "type": "function", "function": { "name": "f" } })
            ))
            .is_err()
        );
    }

    #[test]
    fn a_finished_turn_with_calls_reports_tool_calls_but_a_truncated_one_length() {
        assert_eq!(chat_finish_reason(&Finish::EndOfTurn, 2), "tool_calls");
        assert_eq!(chat_finish_reason(&Finish::EndOfTurn, 0), "stop");
        assert_eq!(chat_finish_reason(&Finish::Length, 2), "length");
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

    #[test]
    fn a_turn_that_only_called_tools_carries_a_null_content_not_a_missing_one() {
        // The regression: one `Option` served both the streaming delta, where
        // the field is absent, and the message, where OpenAI writes `null`.
        // `skip_serializing_if` won, so a client that reads `content` as a
        // required nullable field saw no key at all.
        let message = ChatDelta {
            role: Some("assistant"),
            content: Some(None),
            reasoning_content: None,
            tool_calls: Some(vec![json!({"id": "call_1_0"})]),
        };
        let wire = serde_json::to_value(&message).expect("serializes");
        assert_eq!(wire["content"], Value::Null);
        assert!(wire.as_object().expect("object").contains_key("content"));
    }

    #[test]
    fn a_delta_that_carries_no_text_omits_content_entirely() {
        // The other half of the same field: a streaming delta announcing a
        // tool call has no `content` key, which is what clients accumulate by.
        let delta = ChatDelta {
            tool_calls: Some(vec![json!({"index": 0})]),
            ..ChatDelta::default()
        };
        let wire = serde_json::to_value(&delta).expect("serializes");
        assert!(!wire.as_object().expect("object").contains_key("content"));
    }
}
