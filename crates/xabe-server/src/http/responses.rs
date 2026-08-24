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

use std::collections::HashSet;

use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};

use super::chat::{
    Content, Conversation, Turn, image_misplaced, unsupported_role, unsupported_tool_choice,
};
use super::error::{ApiError, Dialect, parse_body};
use super::generate::{Chunk, Finish, Generation, GenerationSpec, resolve_sampling};
use super::tools::{OfferedTools, ParsedToolCall, ToolCallParser};
use super::warn_unsupported;
use super::{AppState, sse_named, unix_now};

const DIALECT: Dialect = Dialect::OpenAi;

/// One entry of a structured `input` list: a `message`, or a replayed
/// `function_call` / `function_call_output` pair.
#[derive(Debug, Deserialize)]
struct InputItem {
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Option<Content>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
    #[serde(default)]
    output: Option<Value>,
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
    #[serde(default)]
    max_output_tokens: Option<u32>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    reasoning: Option<ReasoningOptions>,
    #[serde(default)]
    tools: Option<Vec<Value>>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    parallel_tool_calls: Option<bool>,
    #[serde(default)]
    previous_response_id: Option<String>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    /// Not OpenAI's field, but llama.cpp-aimed clients send it here too.
    #[serde(default)]
    min_p: Option<f32>,
}

impl ResponsesRequest {
    /// Whether the caller's `tool_choice` lets the tools be offered at all.
    fn tools_offered(&self) -> Result<bool, ApiError> {
        match &self.tool_choice {
            None => Ok(true),
            Some(Value::String(choice)) if choice == "auto" => Ok(true),
            Some(Value::String(choice)) if choice == "none" => Ok(false),
            Some(Value::String(choice)) => Err(unsupported_tool_choice(DIALECT, choice)),
            Some(_) => Err(unsupported_tool_choice(DIALECT, "a named function")),
        }
    }

    fn tool_definitions(&self) -> Result<OfferedTools, ApiError> {
        OfferedTools::from_responses(self.tools.as_deref().unwrap_or_default())
            .map_err(|failure| ApiError::bad_request(DIALECT, failure))
    }

    /// The names of the `namespace` groups the caller offered.
    ///
    /// Kept so a served call can be split back into name and namespace; the
    /// wire format carries them as separate fields.
    fn namespaces(&self) -> HashSet<String> {
        self.tools
            .as_deref()
            .unwrap_or_default()
            .iter()
            .filter(|tool| tool.get("type").and_then(Value::as_str) == Some("namespace"))
            .filter_map(|tool| tool.get("name").and_then(Value::as_str))
            .map(str::to_owned)
            .collect()
    }

    fn thinking_enabled(&self, default: bool) -> bool {
        match self
            .reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort.as_deref())
        {
            Some("none" | "minimal") => false,
            Some(_) => true,
            None => default,
        }
    }

    fn conversation(&self) -> Result<Conversation, ApiError> {
        let mut conversation = Conversation {
            system: self.instructions.clone(),
            turns: Vec::new(),
            tools: Vec::new(),
            images: Vec::new(),
        };
        match &self.input {
            Input::Text(text) => conversation.turns.push(Turn::User(text.clone())),
            Input::Items(items) => {
                for item in items {
                    match item.kind.as_deref() {
                        None | Some("message") => {}
                        // A call this server made on a previous turn, being
                        // replayed. It belongs to the assistant turn before
                        // it, which is also where the template renders it.
                        Some("function_call") => {
                            let name = item
                                .name
                                .clone()
                                .filter(|name| !name.is_empty())
                                .ok_or_else(|| {
                                    ApiError::bad_request(
                                        DIALECT,
                                        "a `function_call` item needs a `name`",
                                    )
                                })?;
                            let arguments = match item.arguments.as_deref() {
                                None | Some("") => serde_json::Map::new(),
                                Some(text) => serde_json::from_str::<Value>(text)
                                    .ok()
                                    .and_then(|parsed| parsed.as_object().cloned())
                                    .ok_or_else(|| {
                                        ApiError::bad_request(
                                            DIALECT,
                                            format!(
                                                "function_call `{name}` has `arguments` that \
                                                 are not a JSON object"
                                            ),
                                        )
                                    })?,
                            };
                            let call = ParsedToolCall { name, arguments };
                            if let Some(Turn::Assistant { tool_calls, .. }) =
                                conversation.turns.last_mut()
                            {
                                tool_calls.push(call);
                            } else {
                                conversation.turns.push(Turn::Assistant {
                                    reasoning: String::new(),
                                    content: String::new(),
                                    tool_calls: vec![call],
                                });
                            }
                            continue;
                        }
                        Some("function_call_output") => {
                            let output = match &item.output {
                                None => String::new(),
                                Some(Value::String(text)) => text.clone(),
                                Some(other) => {
                                    serde_json::to_string(other).expect("JSON values serialize")
                                }
                            };
                            conversation.push_tool_result(output);
                            continue;
                        }
                        Some(kind) => {
                            return Err(ApiError::bad_request(
                                DIALECT,
                                format!(
                                    "input items of type `{kind}` are not supported; send \
                                     `message`, `function_call`, or `function_call_output` items"
                                ),
                            ));
                        }
                    }
                    let folded = item
                        .content
                        .as_ref()
                        .map(Content::fold)
                        .transpose()
                        .map_err(|failure| ApiError::bad_request(DIALECT, failure))?
                        .unwrap_or_default();
                    // Tool traffic travels as `function_call` items in this
                    // dialect, never as content blocks.
                    if !folded.tool_calls.is_empty() || !folded.tool_results.is_empty() {
                        return Err(ApiError::bad_request(
                            DIALECT,
                            "`tool_use` and `tool_result` content blocks are not valid here",
                        ));
                    }
                    if !folded.images.is_empty() && item.role.as_deref() != Some("user") {
                        return Err(ApiError::bad_request(DIALECT, image_misplaced()));
                    }
                    let (text, thinking) = (folded.text, folded.thinking);
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
                        Some("user") => {
                            conversation.images.extend(folded.images);
                            conversation.turns.push(Turn::User(text));
                        }
                        Some("assistant") => conversation
                            .turns
                            .push(Turn::assistant_text(thinking, text)),
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
    /// Echoed back verbatim. `tools`, `tool_choice` and `parallel_tool_calls`
    /// are *required* members of the Responses object — the official SDK
    /// models them as non-optional, so a response without them fails
    /// validation before a caller ever sees the output. They were missing,
    /// which made this endpoint unusable from `openai-python` regardless of
    /// what the model produced.
    tools: Value,
    tool_choice: Value,
    parallel_tool_calls: bool,
    namespaces: HashSet<String>,
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

    fn function_call_item_id(&self, index: usize) -> String {
        format!("fc_{}_{index}", self.id.trim_start_matches("resp_"))
    }

    fn function_call_id(&self, index: usize) -> String {
        format!("call_{}_{index}", self.id.trim_start_matches("resp_"))
    }

    /// Split a served call name into its tool name and namespace.
    ///
    /// Namespaced tools are offered to the model as `group.member` so their
    /// names cannot collide, but the wire format wants the two apart.
    fn split_namespace<'a>(&self, name: &'a str) -> (&'a str, Option<&'a str>) {
        match name.split_once('.') {
            Some((group, member)) if self.namespaces.contains(group) => (member, Some(group)),
            _ => (name, None),
        }
    }

    fn function_call_item(&self, index: usize, call: &ParsedToolCall, status: &str) -> Value {
        let (name, namespace) = self.split_namespace(&call.name);
        let mut item = json!({
            "id": self.function_call_item_id(index),
            "type": "function_call",
            "status": status,
            "call_id": self.function_call_id(index),
            "name": name,
            "arguments": if status == "in_progress" {
                String::new()
            } else {
                call.arguments_json()
            },
        });
        if let Some(namespace) = namespace {
            item["namespace"] = Value::String(namespace.to_owned());
        }
        item
    }

    fn object(&self, status: &str, incomplete: Value, output: Vec<Value>, usage: Value) -> Value {
        json!({
            "id": self.id,
            "object": "response",
            "created_at": self.created,
            "status": status,
            "model": self.model,
            "output": output,
            "parallel_tool_calls": self.parallel_tool_calls,
            "tool_choice": self.tool_choice,
            "tools": self.tools,
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
    if request.previous_response_id.is_some() {
        return Err(ApiError::bad_request(
            DIALECT,
            "`previous_response_id` is not supported: this server does not store responses, \
             so send the whole conversation in `input`",
        ));
    }
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
        max_tokens: request
            .max_output_tokens
            .unwrap_or(state.default_max_tokens),
        stop: Vec::new(),
        thinking,
        trim_spans: true,
        sampling: resolve_sampling(
            DIALECT,
            state.sampling_defaults,
            request.temperature,
            request.top_p,
            None,
            request.min_p,
            None,
        )?,
        tool_parser,
    };
    let mut generation = Generation::start(&state, spec, DIALECT)?;
    // Read off the request before `model` is moved out of it.
    let namespaces = request.namespaces();
    let offered_tools = Value::Array(request.tools.clone().unwrap_or_default());
    let envelope = Envelope {
        id: format!("resp_{}", generation.request_id()),
        created: unix_now(),
        model: state.model_name(request.model),
        // Echoed as sent. `tool_choice` defaults to "auto" and
        // `parallel_tool_calls` to true, matching what the API does when the
        // caller omits them — several tool calls in one turn is a shape this
        // server does serve.
        namespaces,
        tools: offered_tools,
        tool_choice: request
            .tool_choice
            .unwrap_or_else(|| Value::String("auto".to_owned())),
        parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
    };

    if !request.stream {
        let collected = generation.collect().await?;
        let (status, incomplete) = status_of(&generation.finish());
        let mut output = Vec::with_capacity(2 + collected.tool_calls.len());
        if !collected.reasoning.is_empty() {
            output.push(envelope.reasoning_item(&collected.reasoning));
        }
        if !collected.text.is_empty() || collected.tool_calls.is_empty() {
            output.push(envelope.message_item(&collected.text));
        }
        for (index, call) in collected.tool_calls.iter().enumerate() {
            output.push(envelope.function_call_item(index, call, "completed"));
        }
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
        let mut calls: Vec<super::tools::ParsedToolCall> = Vec::new();
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
            // A tool call closes whatever item is open and emits a complete
            // item of its own, so it maps to "nothing open" here.
            let kind = match &chunk {
                Some(Chunk::Reasoning(_)) => Some("reasoning"),
                Some(Chunk::Text(_)) => Some("message"),
                Some(Chunk::ToolCall(_)) | None => None,
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
                Some(Chunk::ToolCall(call)) => {
                    // A call parses only once its block is complete, so its
                    // item streams as one added/delta/done/done group.
                    let index = calls.len();
                    let item_id = envelope.function_call_item_id(index);
                    yield Ok(emit!("response.output_item.added", {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": envelope.function_call_item(index, &call, "in_progress"),
                    }));
                    yield Ok(emit!("response.function_call_arguments.delta", {
                        "type": "response.function_call_arguments.delta",
                        "item_id": item_id,
                        "output_index": output_index,
                        "delta": call.arguments_json(),
                    }));
                    yield Ok(emit!("response.function_call_arguments.done", {
                        "type": "response.function_call_arguments.done",
                        "item_id": item_id,
                        "output_index": output_index,
                        "arguments": call.arguments_json(),
                    }));
                    yield Ok(emit!("response.output_item.done", {
                        "type": "response.output_item.done",
                        "output_index": output_index,
                        "item": envelope.function_call_item(index, &call, "completed"),
                    }));
                    output_index += 1;
                    calls.push(call);
                }
                None => break,
            }
        }

        let (status, incomplete) = status_of(&generation.finish());
        let mut output = Vec::with_capacity(2 + calls.len());
        if !reasoning.is_empty() {
            output.push(envelope.reasoning_item(&reasoning));
        }
        if !text.is_empty() || calls.is_empty() {
            output.push(envelope.message_item(&text));
        }
        for (index, call) in calls.iter().enumerate() {
            output.push(envelope.function_call_item(index, call, "completed"));
        }
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

    #[test]
    fn a_hosted_tool_is_dropped_and_the_function_tools_beside_it_still_serve() {
        // The regression. `web_search` is run by the provider, and there is
        // no provider here — but refusing the request over it took the
        // caller's own tools down too, so nothing served at all.
        let request: ResponsesRequest = serde_json::from_value(json!({
            "input": "hi",
            "tools": [
                {"type": "web_search"},
                {"type": "function", "name": "get_customer", "parameters": {"type": "object"}},
                {"type": "mcp", "server_label": "x"},
            ],
        }))
        .expect("request parses");
        let offered = request
            .tool_definitions()
            .expect("hosted tools do not fail the request");
        assert_eq!(
            offered.definitions.len(),
            1,
            "the function tool still serves"
        );
        assert_eq!(offered.unsupported, vec!["web_search", "mcp"]);
    }

    #[test]
    fn a_malformed_function_tool_is_still_an_error() {
        // Dropping what cannot be executed must not start swallowing the
        // caller's own mistakes: a nameless function is a bug to report.
        let request: ResponsesRequest = serde_json::from_value(json!({
            "input": "hi",
            "tools": [{"type": "function", "parameters": {"type": "object"}}],
        }))
        .expect("request parses");
        assert!(
            request.tool_definitions().is_err(),
            "a nameless tool is the caller's bug"
        );
    }

    fn namespaced_envelope() -> Envelope {
        Envelope {
            id: "resp_1".to_owned(),
            created: 0,
            model: "m".to_owned(),
            tools: Value::Array(vec![]),
            tool_choice: Value::String("auto".to_owned()),
            parallel_tool_calls: true,
            namespaces: HashSet::from(["crm".to_owned()]),
        }
    }

    fn parsed(name: &str) -> ParsedToolCall {
        ParsedToolCall {
            name: name.to_owned(),
            arguments: serde_json::Map::new(),
        }
    }

    #[test]
    fn a_namespace_group_is_flattened_into_its_member_tools() {
        // Harnesses group tools to keep a big surface from spending its whole
        // budget on schemas. Rejecting the group outright took every tool in
        // it with them.
        let request: ResponsesRequest = serde_json::from_value(json!({
            "input": "hi",
            "tools": [{
                "type": "namespace",
                "name": "crm",
                "description": "CRM tools.",
                "tools": [
                    {"type": "function", "name": "get_customer", "parameters": {"type": "object"}},
                    {"type": "function", "name": "list_orders", "defer_loading": true,
                     "parameters": {"type": "object"}},
                ],
            }],
        }))
        .expect("request parses");
        let tools = request.tool_definitions().expect("namespace flattens");
        assert_eq!(tools.definitions.len(), 2, "both members must be offered");
        assert_eq!(request.namespaces(), HashSet::from(["crm".to_owned()]));
    }

    #[test]
    fn a_namespaced_call_is_served_with_the_namespace_in_its_own_field() {
        // The wire format carries the namespace beside the name, not as a
        // prefix on it. A client matching on `name` never sees `crm.`.
        let item =
            namespaced_envelope().function_call_item(0, &parsed("crm.list_orders"), "completed");
        assert_eq!(item["name"], "list_orders");
        assert_eq!(item["namespace"], "crm");
    }

    #[test]
    fn a_plain_call_carries_no_namespace_field_at_all() {
        let item = namespaced_envelope().function_call_item(0, &parsed("list_orders"), "completed");
        assert_eq!(item["name"], "list_orders");
        assert!(
            item.get("namespace").is_none(),
            "unnamespaced calls stay unnamespaced"
        );
    }

    #[test]
    fn a_dot_that_is_not_a_known_namespace_stays_part_of_the_name() {
        // Tool names may legitimately contain a dot. Only a prefix matching a
        // group the caller actually offered is treated as a namespace.
        let item = namespaced_envelope().function_call_item(0, &parsed("v1.search"), "completed");
        assert_eq!(item["name"], "v1.search");
        assert!(item.get("namespace").is_none());
    }

    #[test]
    fn two_namespaces_may_each_hold_the_same_tool_name() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "input": "hi",
            "tools": [
                {"type": "namespace", "name": "crm", "tools": [
                    {"type": "function", "name": "search", "parameters": {"type": "object"}}]},
                {"type": "namespace", "name": "docs", "tools": [
                    {"type": "function", "name": "search", "parameters": {"type": "object"}}]},
            ],
        }))
        .expect("request parses");
        let tools = request.tool_definitions().expect("both namespaces flatten");
        assert_eq!(
            tools.definitions.len(),
            2,
            "prompt names are `group.member`, so they do not collide"
        );
    }

    #[test]
    fn a_namespace_holding_a_non_function_tool_is_refused_by_name() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "input": "hi",
            "tools": [{"type": "namespace", "name": "crm", "tools": [{"type": "mcp"}]}],
        }))
        .expect("request parses");
        assert!(request.tool_definitions().is_err());
    }

    /// The three fields the official SDK models as non-optional.
    #[test]
    fn the_response_object_carries_every_field_the_sdk_requires() {
        // These were missing, and their absence is not a soft failure: the
        // SDK validates the object before a caller sees any output, so every
        // call through `openai-python` failed regardless of what was served.
        let envelope = Envelope {
            id: "resp_1".to_owned(),
            created: 0,
            model: "m".to_owned(),
            tools: Value::Array(vec![]),
            tool_choice: Value::String("auto".to_owned()),
            parallel_tool_calls: true,
            namespaces: HashSet::new(),
        };
        let object = envelope.object("completed", Value::Null, vec![], Value::Null);
        for field in [
            "id",
            "created_at",
            "model",
            "object",
            "output",
            "parallel_tool_calls",
            "tool_choice",
            "tools",
        ] {
            assert!(
                object.get(field).is_some(),
                "missing required field `{field}`"
            );
        }
        assert_eq!(object["object"], "response");
    }

    #[test]
    fn an_incomplete_response_says_why_it_is_incomplete() {
        let (status, incomplete) = status_of(&Finish::Length);
        assert_eq!(status, "incomplete");
        assert_eq!(incomplete["reason"], "max_output_tokens");
    }

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
        assert!(!request(r#"{"input":"Hi","reasoning":{"effort":"none"}}"#).thinking_enabled(true));
        assert!(
            !request(r#"{"input":"Hi","reasoning":{"effort":"minimal"}}"#).thinking_enabled(true)
        );
    }

    #[test]
    fn a_named_effort_overrides_the_server_default_either_way() {
        assert!(request(r#"{"input":"Hi","reasoning":{"effort":"high"}}"#).thinking_enabled(false));
        assert!(!request(r#"{"input":"Hi","reasoning":{"effort":"none"}}"#).thinking_enabled(true));
    }

    #[test]
    fn a_silent_request_follows_the_server_default() {
        assert!(request(r#"{"input":"Hi"}"#).thinking_enabled(true));
        assert!(!request(r#"{"input":"Hi"}"#).thinking_enabled(false));
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
