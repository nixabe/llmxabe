//! Tool calling: definitions in three dialects, one prompt shape, and the
//! parser that turns the model's `<tool_call>` markup back into structured
//! calls.
//!
//! Qwen3.6's template (the GGUF's `tokenizer.chat_template`) speaks the
//! XML-parameter format Qwen3-Coder introduced:
//!
//! ```text
//! <tool_call>
//! <function=get_weather>
//! <parameter=city>
//! Paris
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! The parsing rules follow llama.cpp's parser for this family
//! (`common/chat.cpp`, `common_chat_params_init_qwen3_coder`): a parameter
//! whose schema type is `string` takes the raw text between its tags, any
//! other parameter is parsed as JSON, and a call may open with a bare
//! `<function=name>` — for a *known* name only — because the model
//! occasionally omits the `<tool_call>` line. Matching only known names is
//! what keeps prose like `#include <functional>` from being eaten.
//!
//! A block that does not parse is returned to the client as plain text
//! rather than dropped: the caller can at least see what the model said.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

/// One tool the caller offered, in the shape the prompt and the parser need.
#[derive(Debug, Clone)]
pub(crate) struct ToolDefinition {
    pub(crate) name: String,
    /// The `{"type":"function","function":{...}}` object rendered into the
    /// prompt's `<tools>` block, exactly as the template's `tool | tojson`
    /// renders whatever the caller passed.
    pub(crate) wrapper: Value,
    /// Parameter name → whether its declared schema type is `string`, which
    /// decides raw-text versus JSON parsing of the value.
    string_params: HashMap<String, bool>,
}

/// Whether a JSON schema fragment declares a plain string.
fn schema_is_string(schema: &Value) -> bool {
    match schema.get("type") {
        Some(Value::String(kind)) => kind == "string",
        Some(Value::Array(kinds)) => kinds
            .iter()
            .all(|kind| matches!(kind, Value::String(k) if k == "string" || k == "null")),
        _ => false,
    }
}

/// The tools a request offered, after dropping the ones this server cannot
/// serve.
///
/// A hosted tool — `web_search`, `file_search`, `code_interpreter`, an `mcp`
/// server, Anthropic's versioned `bash_*` and `text_editor_*` — is executed by
/// the provider, and there is no provider here. Refusing the whole request
/// over one was the wrong failure: a harness that offers `web_search`
/// alongside six function tools lost all seven and got nothing served at all.
///
/// So they are dropped and the rest are served. The drop is reported rather
/// than silent: the caller's own tools still work, and a `warn!` names what
/// went missing so a harness that genuinely needed it can be found out from
/// the log rather than from a wrong answer. Anything unrecognized is dropped
/// the same way, which keeps a tool type invented next year from taking a
/// working request down with it.
pub(crate) struct OfferedTools {
    pub(crate) definitions: Vec<ToolDefinition>,
    /// Distinct tool types dropped, in the order first seen.
    pub(crate) unsupported: Vec<String>,
}

impl OfferedTools {
    fn drop_kind(&mut self, kind: &str) {
        let kind = if kind.is_empty() { "(untyped)" } else { kind };
        if !self.unsupported.iter().any(|seen| seen == kind) {
            self.unsupported.push(kind.to_owned());
        }
    }

    fn kind_of(value: &Value) -> &str {
        value.get("type").and_then(Value::as_str).unwrap_or("")
    }

    /// Responses API: flat functions, and `namespace` groups of them.
    pub(crate) fn from_responses(values: &[Value]) -> Result<Self, String> {
        let mut offered = Self {
            definitions: Vec::new(),
            unsupported: Vec::new(),
        };
        for value in values {
            match Self::kind_of(value) {
                "function" | "namespace" => {
                    offered
                        .definitions
                        .extend(ToolDefinition::from_responses_entry(value)?);
                }
                kind => offered.drop_kind(kind),
            }
        }
        Ok(offered)
    }

    /// OpenAI chat completions: `{"type":"function","function":{…}}`.
    pub(crate) fn from_openai(values: &[Value]) -> Result<Self, String> {
        let mut offered = Self {
            definitions: Vec::new(),
            unsupported: Vec::new(),
        };
        for value in values {
            match Self::kind_of(value) {
                "function" => offered
                    .definitions
                    .push(ToolDefinition::from_openai(value)?),
                kind => offered.drop_kind(kind),
            }
        }
        Ok(offered)
    }

    /// Anthropic: a client tool carries `input_schema` and either no `type` or
    /// a `custom` one. A versioned `type` is one of Anthropic's own server
    /// tools.
    pub(crate) fn from_anthropic(values: &[Value]) -> Result<Self, String> {
        let mut offered = Self {
            definitions: Vec::new(),
            unsupported: Vec::new(),
        };
        for value in values {
            match Self::kind_of(value) {
                "" | "custom" => {
                    offered
                        .definitions
                        .push(ToolDefinition::from_anthropic(value)?);
                }
                kind if kind.starts_with("custom") => {
                    offered
                        .definitions
                        .push(ToolDefinition::from_anthropic(value)?);
                }
                kind => offered.drop_kind(kind),
            }
        }
        Ok(offered)
    }
}

impl ToolDefinition {
    fn build(
        name: &str,
        description: Option<&Value>,
        parameters: Option<&Value>,
    ) -> Result<Self, String> {
        if name.is_empty() {
            return Err("a tool needs a non-empty `name`".to_owned());
        }
        if name.contains(['>', '<', '\n']) {
            return Err(format!(
                "tool name `{name}` cannot be rendered into the prompt's `<function=...>` markup"
            ));
        }
        let mut function = Map::new();
        function.insert("name".to_owned(), json!(name));
        if let Some(description) = description {
            function.insert("description".to_owned(), description.clone());
        }
        if let Some(parameters) = parameters {
            function.insert("parameters".to_owned(), parameters.clone());
        }
        let string_params = parameters
            .and_then(|parameters| parameters.get("properties"))
            .and_then(Value::as_object)
            .map(|properties| {
                properties
                    .iter()
                    .map(|(key, schema)| (key.clone(), schema_is_string(schema)))
                    .collect()
            })
            .unwrap_or_default();
        Ok(Self {
            name: name.to_owned(),
            wrapper: json!({ "type": "function", "function": Value::Object(function) }),
            string_params,
        })
    }

    /// An OpenAI chat tool: `{"type":"function","function":{...}}`.
    pub(crate) fn from_openai(value: &Value) -> Result<Self, String> {
        if value.get("type").and_then(Value::as_str) != Some("function") {
            return Err("every tool must have `\"type\": \"function\"`".to_owned());
        }
        let function = value
            .get("function")
            .and_then(Value::as_object)
            .ok_or("a tool needs a `function` object")?;
        Self::build(
            function.get("name").and_then(Value::as_str).unwrap_or(""),
            function.get("description"),
            function.get("parameters"),
        )
    }

    /// An Anthropic tool: `{"name": ..., "input_schema": ...}`.
    pub(crate) fn from_anthropic(value: &Value) -> Result<Self, String> {
        if value
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| kind != "custom" && !kind.starts_with("custom"))
        {
            return Err(format!(
                "server tools of type `{}` are not supported; send client tools \
                 (`name` + `input_schema`)",
                value.get("type").and_then(Value::as_str).unwrap_or(""),
            ));
        }
        Self::build(
            value.get("name").and_then(Value::as_str).unwrap_or(""),
            value.get("description"),
            value.get("input_schema"),
        )
    }

    /// One entry of a Responses API `tools` array.
    ///
    /// Either a flat function, or a `namespace` group holding several — the
    /// shape harnesses use to keep a large tool surface from spending its
    /// whole budget on schemas. A group is flattened into its members, whose
    /// prompt names become `group.member` so two namespaces may each hold a
    /// `search` without colliding. The dotted name is split back apart when
    /// the call is served, because the wire format carries the namespace as
    /// its own field rather than as a prefix.
    ///
    /// `defer_loading` on a member is accepted and ignored: it exists so a
    /// schema can be fetched later by tool search, which this server does not
    /// implement. Every schema is offered up front instead, which costs
    /// prompt tokens and leaves the tool callable.
    pub(crate) fn from_responses_entry(value: &Value) -> Result<Vec<Self>, String> {
        match value.get("type").and_then(Value::as_str) {
            Some("namespace") => {
                let group = value
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|name| !name.is_empty())
                    .ok_or("a `namespace` tool needs a non-empty `name`")?;
                if group.contains('.') {
                    return Err(format!(
                        "namespace `{group}` cannot contain `.`; it separates the \
                         namespace from the tool name"
                    ));
                }
                value
                    .get("tools")
                    .and_then(Value::as_array)
                    .ok_or_else(|| format!("namespace `{group}` needs a `tools` array"))?
                    .iter()
                    .map(|tool| {
                        if tool.get("type").and_then(Value::as_str) != Some("function") {
                            return Err(format!(
                                "namespace `{group}` may only hold `function` tools"
                            ));
                        }
                        let name = tool.get("name").and_then(Value::as_str).unwrap_or("");
                        Self::build(
                            &format!("{group}.{name}"),
                            tool.get("description"),
                            tool.get("parameters"),
                        )
                    })
                    .collect()
            }
            _ => Ok(vec![Self::from_responses(value)?]),
        }
    }

    /// A Responses API tool: flat `{"type":"function","name":...}`.
    pub(crate) fn from_responses(value: &Value) -> Result<Self, String> {
        if value.get("type").and_then(Value::as_str) != Some("function") {
            return Err(format!(
                "tools of type `{}` are not supported; this server executes nothing itself, \
                 so only `function` tools, and `namespace` groups of them, make sense",
                value.get("type").and_then(Value::as_str).unwrap_or(""),
            ));
        }
        Self::build(
            value.get("name").and_then(Value::as_str).unwrap_or(""),
            value.get("description"),
            value.get("parameters"),
        )
    }
}

/// One call parsed out of the model's output.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ParsedToolCall {
    pub(crate) name: String,
    pub(crate) arguments: Map<String, Value>,
}

impl ParsedToolCall {
    /// The arguments as the JSON string OpenAI's wire format carries.
    pub(crate) fn arguments_json(&self) -> String {
        serde_json::to_string(&self.arguments).expect("JSON maps serialize")
    }
}

/// What the parser hands back for a piece of decoded output.
#[derive(Debug, PartialEq)]
pub(crate) enum ToolEvent {
    Text(String),
    Call(ParsedToolCall),
}

/// A call block the model left unclosed grows without bound if the model
/// rambles; past this the block is handed back as text.
const MAX_CALL_BYTES: usize = 256 * 1024;

const CALL_OPEN: &str = "<tool_call>";
const CALL_CLOSE: &str = "</tool_call>";

enum ParserState {
    /// Emitting text, watching for a call opener.
    Text,
    /// Between an opener and `</tool_call>`; `raw` holds everything consumed
    /// since the opener so an unparseable block can be returned verbatim.
    InCall { raw: String },
}

/// Incremental scanner over the answer span's text.
///
/// Text that could still turn out to be the beginning of a call opener is
/// held back, exactly as the generation path holds back potential stop
/// sequences, so an opener split across two chunks is never missed.
pub(crate) struct ToolCallParser {
    /// `<tool_call>`, plus one `<function=name>` opener per known tool.
    openers: Vec<String>,
    /// Tool name → its parameters' string-ness, for value typing.
    schemas: HashMap<String, HashMap<String, bool>>,
    pending: String,
    state: ParserState,
    /// Whitespace between consecutive calls is markup, not content.
    swallow_whitespace: bool,
}

impl ToolCallParser {
    pub(crate) fn new(tools: &[ToolDefinition]) -> Self {
        let mut openers = vec![CALL_OPEN.to_owned()];
        openers.extend(tools.iter().map(|tool| format!("<function={}>", tool.name)));
        Self {
            openers,
            schemas: tools
                .iter()
                .map(|tool| (tool.name.clone(), tool.string_params.clone()))
                .collect(),
            pending: String::new(),
            state: ParserState::Text,
            swallow_whitespace: false,
        }
    }

    /// Feed decoded text; parsed events append to `events`.
    pub(crate) fn push(&mut self, text: &str, events: &mut Vec<ToolEvent>) {
        self.pending.push_str(text);
        loop {
            match &mut self.state {
                ParserState::Text => {
                    let opener = self
                        .openers
                        .iter()
                        .filter_map(|opener| {
                            self.pending.find(opener.as_str()).map(|at| (at, opener))
                        })
                        .min_by_key(|&(at, _)| at);
                    match opener {
                        Some((at, opener)) => {
                            let opener = opener.clone();
                            let text: String = self.pending.drain(..at).collect();
                            self.emit_text(text, events);
                            self.pending.drain(..opener.len());
                            // A bare `<function=...>` opener is part of the
                            // call body; the `<tool_call>` line is markup.
                            let raw = if opener == CALL_OPEN {
                                String::new()
                            } else {
                                opener
                            };
                            self.state = ParserState::InCall { raw };
                        }
                        None => {
                            let held = held_back_len(&self.pending, &self.openers);
                            let emit = self.pending.len() - held;
                            if emit > 0 {
                                let text: String = self.pending.drain(..emit).collect();
                                self.emit_text(text, events);
                            }
                            return;
                        }
                    }
                }
                ParserState::InCall { raw } => {
                    match self.pending.find(CALL_CLOSE) {
                        Some(at) => {
                            raw.push_str(&self.pending[..at]);
                            self.pending.drain(..at + CALL_CLOSE.len());
                            let raw = std::mem::take(raw);
                            self.state = ParserState::Text;
                            match parse_call(&raw, &self.schemas) {
                                Some(call) => {
                                    events.push(ToolEvent::Call(call));
                                    self.swallow_whitespace = true;
                                }
                                None => {
                                    // Reassemble what was consumed so the
                                    // caller sees the model's actual output.
                                    self.emit_text(format!("{CALL_OPEN}{raw}{CALL_CLOSE}"), events);
                                }
                            }
                        }
                        None => {
                            // No closer yet: move everything but a possible
                            // closer prefix into the call buffer and wait.
                            let held = held_back_len(&self.pending, &[CALL_CLOSE.to_owned()]);
                            let take = self.pending.len() - held;
                            raw.push_str(&self.pending[..take]);
                            self.pending.drain(..take);
                            if raw.len() > MAX_CALL_BYTES {
                                let raw = std::mem::take(raw);
                                self.state = ParserState::Text;
                                self.emit_text(format!("{CALL_OPEN}{raw}"), events);
                            }
                            return;
                        }
                    }
                }
            }
        }
    }

    /// End of output: whatever is held back was never going to become a call.
    pub(crate) fn finish(&mut self, events: &mut Vec<ToolEvent>) {
        let tail = match std::mem::replace(&mut self.state, ParserState::Text) {
            ParserState::Text => std::mem::take(&mut self.pending),
            ParserState::InCall { raw } => {
                format!("{CALL_OPEN}{raw}{}", std::mem::take(&mut self.pending))
            }
        };
        self.emit_text(tail, events);
    }

    fn emit_text(&mut self, text: String, events: &mut Vec<ToolEvent>) {
        let text = if self.swallow_whitespace {
            let trimmed = text.trim_start();
            if trimmed.is_empty() {
                return;
            }
            self.swallow_whitespace = false;
            trimmed.to_owned()
        } else {
            text
        };
        if text.is_empty() {
            return;
        }
        events.push(ToolEvent::Text(text));
    }
}

/// The longest suffix of `text` that is a proper prefix of some needle.
///
/// The same scheme `generate.rs` uses for stop sequences, over byte-safe char
/// boundaries.
fn held_back_len(text: &str, needles: &[String]) -> usize {
    let mut held = 0;
    for needle in needles {
        for (end, _) in needle.char_indices().skip(1) {
            if text.len() >= end && text.is_char_boundary(text.len() - end) {
                let candidate = &text[text.len() - end..];
                if candidate == &needle[..end] {
                    held = held.max(end);
                }
            }
        }
    }
    held
}

/// Parse the inside of one `<tool_call>` block, `<function=...>` included.
///
/// `None` means the block does not follow the format and should be shown to
/// the caller as text.
fn parse_call(
    raw: &str,
    schemas: &HashMap<String, HashMap<String, bool>>,
) -> Option<ParsedToolCall> {
    let mut rest = raw.trim();
    rest = rest.strip_prefix("<function=")?;
    let name_end = rest.find('>')?;
    let name = &rest[..name_end];
    if name.is_empty() || name.contains(['<', '\n']) {
        return None;
    }
    rest = &rest[name_end + 1..];
    let string_params = schemas.get(name);

    let mut arguments = Map::new();
    loop {
        rest = rest.trim_start();
        if let Some(after) = rest.strip_prefix("</function>") {
            if !after.trim().is_empty() {
                return None;
            }
            return Some(ParsedToolCall {
                name: name.to_owned(),
                arguments,
            });
        }
        rest = rest.strip_prefix("<parameter=")?;
        let key_end = rest.find('>')?;
        let key = rest[..key_end].to_owned();
        if key.is_empty() || key.contains(['<', '\n']) {
            return None;
        }
        rest = &rest[key_end + 1..];
        // The template writes `<parameter=k>\n` value `\n</parameter>`; the
        // newlines belong to the markup, not the value.
        rest = rest.strip_prefix('\n').unwrap_or(rest);
        let value_end = rest.find("</parameter>")?;
        let value = rest[..value_end]
            .strip_suffix('\n')
            .unwrap_or(&rest[..value_end]);
        rest = &rest[value_end + "</parameter>".len()..];

        // Typing follows llama.cpp (`common/chat.cpp`,
        // `common_chat_params_init_qwen3_coder`): a schema that says
        // `string` takes the text verbatim; anything else is JSON, with the
        // raw text kept when it does not parse.
        let is_string = string_params
            .and_then(|params| params.get(&key).copied())
            .unwrap_or(false);
        let parsed = if is_string {
            Value::String(value.to_owned())
        } else {
            serde_json::from_str(value.trim()).unwrap_or_else(|_| Value::String(value.to_owned()))
        };
        arguments.insert(key, parsed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_server_tools_are_dropped_and_client_tools_beside_them_serve() {
        // Anthropic's own hosted tools carry a dated type. The server runs
        // them; this one cannot, but the caller's client tools are fine.
        let offered = OfferedTools::from_anthropic(&[
            json!({"type": "web_search_20250305", "name": "web_search"}),
            json!({"type": "bash_20250124", "name": "bash"}),
            json!({"name": "get_weather", "input_schema": {"type": "object"}}),
            json!({"type": "custom", "name": "lookup", "input_schema": {"type": "object"}}),
        ])
        .expect("hosted tools do not fail the request");
        assert_eq!(offered.definitions.len(), 2, "both client tools serve");
        assert_eq!(
            offered.unsupported,
            vec!["web_search_20250305", "bash_20250124"]
        );
    }

    #[test]
    fn openai_chat_drops_what_it_cannot_run_too() {
        let offered = OfferedTools::from_openai(&[
            json!({"type": "web_search"}),
            json!({"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}),
        ])
        .expect("hosted tools do not fail the request");
        assert_eq!(offered.definitions.len(), 1);
        assert_eq!(offered.unsupported, vec!["web_search"]);
    }

    #[test]
    fn the_same_dropped_type_is_reported_once_however_often_it_appears() {
        let offered = OfferedTools::from_openai(&[
            json!({"type": "web_search"}),
            json!({"type": "web_search"}),
            json!({}),
        ])
        .expect("parses");
        assert_eq!(offered.unsupported, vec!["web_search", "(untyped)"]);
        assert!(offered.definitions.is_empty());
    }

    fn weather_tool() -> ToolDefinition {
        ToolDefinition::from_openai(&json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up the weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": { "type": "string" },
                        "days": { "type": "integer" },
                    },
                    "required": ["city"],
                },
            },
        }))
        .expect("a well-formed tool parses")
    }

    fn events(parser: &mut ToolCallParser, pieces: &[&str]) -> Vec<ToolEvent> {
        let mut out = Vec::new();
        for piece in pieces {
            parser.push(piece, &mut out);
        }
        parser.finish(&mut out);
        out
    }

    fn call(name: &str, arguments: Value) -> ToolEvent {
        ToolEvent::Call(ParsedToolCall {
            name: name.to_owned(),
            arguments: arguments.as_object().expect("object").clone(),
        })
    }

    #[test]
    fn a_simple_call_parses_with_schema_typed_arguments() {
        let mut parser = ToolCallParser::new(&[weather_tool()]);
        let got = events(
            &mut parser,
            &[
                "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>",
            ],
        );
        // `city` is schema-typed string, so it stays text; `days` is an
        // integer and is parsed as JSON.
        assert_eq!(
            got,
            vec![call("get_weather", json!({ "city": "Paris", "days": 3 }))]
        );
    }

    #[test]
    fn text_before_the_call_is_text_and_whitespace_between_calls_is_markup() {
        let mut parser = ToolCallParser::new(&[weather_tool()]);
        let got = events(
            &mut parser,
            &[
                "Let me check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nLyon\n</parameter>\n</function>\n</tool_call>",
            ],
        );
        assert_eq!(
            got,
            vec![
                ToolEvent::Text("Let me check.\n\n".to_owned()),
                call("get_weather", json!({ "city": "Paris" })),
                call("get_weather", json!({ "city": "Lyon" })),
            ]
        );
    }

    #[test]
    fn an_opener_split_across_chunks_is_still_one_call() {
        let mut parser = ToolCallParser::new(&[weather_tool()]);
        let got = events(
            &mut parser,
            &[
                "Sure. <tool",
                "_call>\n<function=get_w",
                "eather>\n<parameter=city>\nOslo",
                "\n</parameter>\n</function>\n</tool_call>",
            ],
        );
        assert_eq!(
            got,
            vec![
                ToolEvent::Text("Sure. ".to_owned()),
                call("get_weather", json!({ "city": "Oslo" })),
            ]
        );
    }

    #[test]
    fn a_bare_function_opener_for_a_known_tool_is_accepted() {
        // llama.cpp's parser accepts this because the model occasionally
        // omits the `<tool_call>` line; only known names qualify, which is
        // what protects prose like `#include <functional>`.
        let mut parser = ToolCallParser::new(&[weather_tool()]);
        let got = events(
            &mut parser,
            &[
                "<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
            ],
        );
        assert_eq!(got, vec![call("get_weather", json!({ "city": "Paris" }))]);
    }

    #[test]
    fn an_unknown_function_opener_stays_text() {
        let mut parser = ToolCallParser::new(&[weather_tool()]);
        let got = events(&mut parser, &["use <function=std::mem::take> here"]);
        assert_eq!(
            got,
            vec![ToolEvent::Text(
                "use <function=std::mem::take> here".to_owned()
            )]
        );
    }

    #[test]
    fn a_malformed_block_is_returned_verbatim() {
        let mut parser = ToolCallParser::new(&[weather_tool()]);
        let raw = "<tool_call>\nnot a function block\n</tool_call>";
        let got = events(&mut parser, &[raw]);
        assert_eq!(got, vec![ToolEvent::Text(raw.to_owned())]);
    }

    #[test]
    fn an_unclosed_block_is_flushed_as_text_at_the_end() {
        let mut parser = ToolCallParser::new(&[weather_tool()]);
        let raw = "<tool_call>\n<function=get_weather>\n<parameter=city>\nPar";
        let got = events(&mut parser, &[raw]);
        assert_eq!(got, vec![ToolEvent::Text(raw.to_owned())]);
    }

    #[test]
    fn a_multiline_value_keeps_its_interior_newlines() {
        let mut parser = ToolCallParser::new(&[weather_tool()]);
        let got = events(
            &mut parser,
            &[
                "<tool_call>\n<function=get_weather>\n<parameter=city>\nline one\nline two\n</parameter>\n</function>\n</tool_call>",
            ],
        );
        assert_eq!(
            got,
            vec![call("get_weather", json!({ "city": "line one\nline two" }))]
        );
    }

    #[test]
    fn an_untyped_parameter_parses_as_json_when_it_can() {
        let tool = ToolDefinition::from_openai(&json!({
            "type": "function",
            "function": { "name": "f" },
        }))
        .expect("a tool without parameters is fine");
        let mut parser = ToolCallParser::new(&[tool]);
        let got = events(
            &mut parser,
            &[
                "<tool_call>\n<function=f>\n<parameter=flag>\ntrue\n</parameter>\n<parameter=note>\nplain words\n</parameter>\n</function>\n</tool_call>",
            ],
        );
        assert_eq!(
            got,
            vec![call("f", json!({ "flag": true, "note": "plain words" }))]
        );
    }

    #[test]
    fn the_three_dialect_shapes_normalize_to_one_wrapper() {
        let openai = weather_tool();
        let anthropic = ToolDefinition::from_anthropic(&json!({
            "name": "get_weather",
            "description": "Look up the weather",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": { "type": "string" },
                    "days": { "type": "integer" },
                },
                "required": ["city"],
            },
        }))
        .expect("an anthropic tool parses");
        let responses = ToolDefinition::from_responses(&json!({
            "type": "function",
            "name": "get_weather",
            "description": "Look up the weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": { "type": "string" },
                    "days": { "type": "integer" },
                },
                "required": ["city"],
            },
        }))
        .expect("a responses tool parses");
        assert_eq!(openai.wrapper, anthropic.wrapper);
        assert_eq!(openai.wrapper, responses.wrapper);
    }

    #[test]
    fn hostile_tool_shapes_are_refused_with_a_reason() {
        assert!(ToolDefinition::from_openai(&json!({ "type": "web_search" })).is_err());
        assert!(
            ToolDefinition::from_openai(&json!({ "type": "function", "function": {} })).is_err()
        );
        assert!(
            ToolDefinition::from_anthropic(&json!({ "type": "bash_20250124", "name": "bash" }))
                .is_err()
        );
        assert!(
            ToolDefinition::from_openai(&json!({
                "type": "function",
                "function": { "name": "bad>name" },
            }))
            .is_err()
        );
    }

    #[test]
    fn arguments_json_round_trips() {
        let call = ParsedToolCall {
            name: "f".to_owned(),
            arguments: json!({ "a": 1, "b": "x" })
                .as_object()
                .expect("object")
                .clone(),
        };
        assert_eq!(call.arguments_json(), r#"{"a":1,"b":"x"}"#);
    }
}
