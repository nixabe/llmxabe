//! Conversations, and the ChatML prompt Qwen3.6 expects.
//!
//! The GGUF carries a Jinja chat template. Rendering it would mean shipping a
//! Jinja engine plus a compatibility layer for the Python string methods the
//! template calls (`startswith`, `split`, `rstrip`), so this module writes the
//! same markup directly and pins the result with tests instead. The parts of
//! the template that are reproduced here are: the merged leading system
//! message, the per-role turn markers, the rule that a reasoning span is
//! replayed only for assistant turns after the last user query, and the
//! generation prompt's open — or pre-closed — `<think>` block.
//!
//! The tool-calling section is reproduced too: the `# Tools` system block,
//! the `<tool_call>`/`<function=`/`<parameter=` markup for replayed calls,
//! and tool results as `<tool_response>` blocks inside user turns. The one
//! part that is *not* reproduced is vision, and callers that ask for it are
//! refused rather than served a prompt that quietly drops what they sent.

use std::fmt::Write as _;

use serde::Deserialize;
use serde_json::Value;

use super::error::{ApiError, Dialect};
use super::tools::{ParsedToolCall, ToolDefinition};

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

/// Message content in either dialect: a bare string, or a list of typed
/// parts.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum Content {
    Text(String),
    Parts(Vec<Part>),
}

/// One content part this server knows how to render. `text` is OpenAI chat
/// and Anthropic; `input_text` and `output_text` are the Responses API;
/// `thinking` is a reasoning block being replayed; `tool_use` and
/// `tool_result` are Anthropic's tool blocks being replayed.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum KnownPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "output_text")]
    OutputText { text: String },
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        name: String,
        #[serde(default)]
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        #[serde(default)]
        content: Option<Value>,
    },
}

/// A content part.
///
/// Anything else — an image, an audio clip, a file reference — parses as
/// `Other` and is refused by name when the conversation is folded. Letting
/// serde reject it instead would report only that no variant of an untagged
/// enum matched, which tells the caller nothing about what this server cannot
/// do.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum Part {
    Known(KnownPart),
    Other(serde_json::Value),
}

/// Content folded down to what the prompt renders.
#[derive(Debug, Default)]
pub(crate) struct FoldedContent {
    pub(crate) text: String,
    pub(crate) thinking: String,
    pub(crate) tool_calls: Vec<ParsedToolCall>,
    pub(crate) tool_results: Vec<String>,
}

/// A `tool_result` block's content: a bare string, or text blocks joined.
fn tool_result_text(content: Option<&Value>) -> Result<String, String> {
    match content {
        None => Ok(String::new()),
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(blocks)) => {
            let mut out = String::new();
            for block in blocks {
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if !out.is_empty() {
                            out.push('\n');
                        }
                        out.push_str(block.get("text").and_then(Value::as_str).unwrap_or(""));
                    }
                    kind => {
                        return Err(format!(
                            "tool_result content of type `{}` is not supported: this engine \
                             is text-only",
                            kind.unwrap_or("(untyped)"),
                        ));
                    }
                }
            }
            Ok(out)
        }
        Some(_) => Err("tool_result `content` must be a string or a list of blocks".to_owned()),
    }
}

impl Content {
    /// Fold content into its text, reasoning, and replayed tool blocks.
    pub(crate) fn fold(&self) -> Result<FoldedContent, String> {
        let mut folded = FoldedContent::default();
        let parts = match self {
            Self::Text(text) => {
                folded.text = text.clone();
                return Ok(folded);
            }
            Self::Parts(parts) => parts,
        };
        for part in parts {
            let (target, value) = match part {
                Part::Known(
                    KnownPart::Text { text: value }
                    | KnownPart::InputText { text: value }
                    | KnownPart::OutputText { text: value },
                ) => (&mut folded.text, value),
                Part::Known(KnownPart::Thinking { thinking: value }) => {
                    (&mut folded.thinking, value)
                }
                Part::Known(KnownPart::ToolUse { name, input }) => {
                    folded.tool_calls.push(ParsedToolCall {
                        name: name.clone(),
                        arguments: input.as_object().cloned().unwrap_or_default(),
                    });
                    continue;
                }
                Part::Known(KnownPart::ToolResult { content }) => {
                    folded
                        .tool_results
                        .push(tool_result_text(content.as_ref())?);
                    continue;
                }
                Part::Other(value) => {
                    let kind = value
                        .get("type")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("(untyped)");
                    return Err(format!(
                        "content parts of type `{kind}` are not supported: this engine is \
                         text-only and does not load the model's vision encoder"
                    ));
                }
            };
            if !target.is_empty() {
                target.push('\n');
            }
            target.push_str(value);
        }
        Ok(folded)
    }

    /// Split content into its answer text and its reasoning text, for the
    /// places where tool blocks have no meaning.
    pub(crate) fn split(&self) -> Result<(String, String), String> {
        let folded = self.fold()?;
        if !folded.tool_calls.is_empty() || !folded.tool_results.is_empty() {
            return Err(
                "`tool_use` and `tool_result` content blocks are not valid here".to_owned(),
            );
        }
        Ok((folded.text, folded.thinking))
    }

    pub(crate) fn text(&self) -> Result<String, String> {
        self.split().map(|(text, _)| text)
    }
}

/// One conversation turn, after the dialect-specific wrapper is stripped off.
#[derive(Debug)]
pub(crate) enum Turn {
    User(String),
    Assistant {
        reasoning: String,
        content: String,
        tool_calls: Vec<ParsedToolCall>,
    },
    /// Consecutive tool results, which the template renders as one user turn
    /// of `<tool_response>` blocks.
    ToolResults(Vec<String>),
}

impl Turn {
    pub(crate) fn assistant_text(reasoning: String, content: String) -> Self {
        Self::Assistant {
            reasoning,
            content,
            tool_calls: Vec::new(),
        }
    }
}

/// A conversation ready to render.
#[derive(Debug, Default)]
pub(crate) struct Conversation {
    pub(crate) system: Option<String>,
    pub(crate) turns: Vec<Turn>,
    /// Tools offered to the model. Non-empty adds the template's `# Tools`
    /// system section; the caller's `tool_choice: "none"` simply leaves this
    /// empty while tool history still renders.
    pub(crate) tools: Vec<ToolDefinition>,
}

/// The template's tool-format instructions, verbatim from the GGUF's
/// `tokenizer.chat_template`.
const TOOL_INSTRUCTIONS: &str = "\n\nIf you choose to call a function ONLY reply in the \
following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n\
<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\n\
This is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n\
</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the \
specified format: an inner <function=...></function> block must be nested within \
<tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may \
provide optional reasoning for your function call in natural language BEFORE the function \
call, but NOT after\n- If there is no function call available, answer the question like \
normal with your current knowledge and do not tell the user about function calls\n\
</IMPORTANT>";

/// How a replayed call argument appears between its `<parameter>` tags.
///
/// Strings verbatim, everything else as JSON. The template renders scalars
/// through Python's `str()` (`True`, `None`), but JSON spellings are what the
/// values were before the wire carried them; the difference is cosmetic and
/// this one is pinned by tests.
fn parameter_value(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => serde_json::to_string(other).expect("JSON values serialize"),
    }
}

impl Conversation {
    /// Render the ChatML prompt, ending with the assistant's generation
    /// prompt.
    ///
    /// `thinking` decides whether that generation prompt leaves the `<think>`
    /// block open for the model to fill, or hands it back already closed and
    /// empty — which is how this model is told to answer without reasoning.
    pub(crate) fn render(&self, thinking: bool) -> String {
        let mut out = String::new();
        let system = self
            .system
            .as_deref()
            .map(str::trim)
            .filter(|system| !system.is_empty());
        if self.tools.is_empty() {
            if let Some(system) = system {
                let _ = write!(out, "{IM_START}system\n{system}{IM_END}\n");
            }
        } else {
            // With tools, the system turn always exists and the caller's own
            // system text follows the tool section.
            let _ = write!(
                out,
                "{IM_START}system\n# Tools\n\nYou have access to the following functions:\n\n<tools>"
            );
            for tool in &self.tools {
                out.push('\n');
                out.push_str(
                    &serde_json::to_string(&tool.wrapper).expect("tool wrappers serialize"),
                );
            }
            out.push_str("\n</tools>");
            out.push_str(TOOL_INSTRUCTIONS);
            if let Some(system) = system {
                let _ = write!(out, "\n\n{system}");
            }
            let _ = writeln!(out, "{IM_END}");
        }
        // An assistant turn replays its reasoning only if it came after the
        // caller's last *question*; earlier reasoning is dropped, exactly as
        // the template drops it. Tool results are not questions — neither a
        // `tool` turn nor a user turn that is one `<tool_response>` block —
        // which is what keeps reasoning replayed across an agentic loop's
        // intermediate steps.
        let last_query = self
            .turns
            .iter()
            .rposition(|turn| match turn {
                Turn::User(content) => {
                    let trimmed = content.trim();
                    !(trimmed.starts_with("<tool_response>")
                        && trimmed.ends_with("</tool_response>"))
                }
                _ => false,
            })
            .unwrap_or(self.turns.len().saturating_sub(1));
        for (index, turn) in self.turns.iter().enumerate() {
            match turn {
                Turn::User(content) => {
                    let _ = write!(out, "{IM_START}user\n{}{IM_END}\n", content.trim());
                }
                Turn::Assistant {
                    reasoning,
                    content,
                    tool_calls,
                } => {
                    let content = content.trim();
                    if index > last_query {
                        let _ = write!(
                            out,
                            "{IM_START}assistant\n<think>\n{}\n</think>\n\n{content}",
                            reasoning.trim()
                        );
                    } else {
                        let _ = write!(out, "{IM_START}assistant\n{content}");
                    }
                    for (call_index, call) in tool_calls.iter().enumerate() {
                        if call_index > 0 {
                            out.push('\n');
                        } else if !content.is_empty() {
                            out.push_str("\n\n");
                        }
                        let _ = write!(out, "<tool_call>\n<function={}>\n", call.name);
                        for (key, value) in &call.arguments {
                            let _ = write!(
                                out,
                                "<parameter={key}>\n{}\n</parameter>\n",
                                parameter_value(value)
                            );
                        }
                        out.push_str("</function>\n</tool_call>");
                    }
                    let _ = writeln!(out, "{IM_END}");
                }
                Turn::ToolResults(results) => {
                    let _ = write!(out, "{IM_START}user");
                    for result in results {
                        let _ = write!(
                            out,
                            "\n<tool_response>\n{}\n</tool_response>",
                            result.trim()
                        );
                    }
                    let _ = writeln!(out, "{IM_END}");
                }
            }
        }
        let _ = writeln!(out, "{IM_START}assistant");
        out.push_str(if thinking {
            "<think>\n"
        } else {
            "<think>\n\n</think>\n\n"
        });
        out
    }

    /// Append a tool result, merging into a preceding results turn the way
    /// the template merges consecutive `tool` messages into one user turn.
    pub(crate) fn push_tool_result(&mut self, result: String) {
        if let Some(Turn::ToolResults(results)) = self.turns.last_mut() {
            results.push(result);
        } else {
            self.turns.push(Turn::ToolResults(vec![result]));
        }
    }
}

/// Reject a role this server has no prompt markup for.
pub(crate) fn unsupported_role(dialect: Dialect, role: &str) -> ApiError {
    ApiError::bad_request(dialect, format!("the `{role}` role is not supported"))
}

/// Reject a `tool_choice` that would require constrained decoding.
///
/// `"auto"` and `"none"` cost nothing to honour. Forcing a call — `required`,
/// `any`, or a named function — is a *guarantee* about the output, and
/// nothing here constrains decoding to keep it; a request answered with prose
/// where a call was guaranteed is a wrong answer, not an approximate one.
pub(crate) fn unsupported_tool_choice(dialect: Dialect, choice: &str) -> ApiError {
    ApiError::bad_request(
        dialect,
        format!(
            "`tool_choice` `{choice}` is not supported: forcing a call would require \
             constrained decoding; use `auto` or `none`"
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(text: &str) -> Turn {
        Turn::User(text.to_owned())
    }

    fn assistant(reasoning: &str, content: &str) -> Turn {
        Turn::assistant_text(reasoning.to_owned(), content.to_owned())
    }

    #[test]
    fn a_single_question_renders_the_documented_chatml() {
        let conversation = Conversation {
            system: Some("You are terse.".to_owned()),
            turns: vec![user("Hi")],
            tools: Vec::new(),
        };
        assert_eq!(
            conversation.render(true),
            "<|im_start|>system\nYou are terse.<|im_end|>\n\
             <|im_start|>user\nHi<|im_end|>\n\
             <|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn thinking_off_hands_back_a_closed_empty_think_block() {
        let conversation = Conversation {
            system: None,
            turns: vec![user("Hi")],
            tools: Vec::new(),
        };
        assert_eq!(
            conversation.render(false),
            "<|im_start|>user\nHi<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn reasoning_before_the_last_question_is_dropped() {
        // Replaying every past reasoning span would grow the prompt without
        // bound and is not what the model was trained to read back.
        let conversation = Conversation {
            system: None,
            turns: vec![user("First"), assistant("pondering", "One"), user("Second")],
            tools: Vec::new(),
        };
        let rendered = conversation.render(true);
        assert!(
            !rendered.contains("pondering"),
            "an assistant turn before the last user query must not replay its reasoning"
        );
        assert!(rendered.contains("<|im_start|>assistant\nOne<|im_end|>"));
    }

    #[test]
    fn reasoning_after_the_last_question_is_replayed() {
        let conversation = Conversation {
            system: None,
            turns: vec![user("First"), assistant("pondering", "One")],
            tools: Vec::new(),
        };
        assert!(
            conversation
                .render(true)
                .contains("<|im_start|>assistant\n<think>\npondering\n</think>\n\nOne<|im_end|>")
        );
    }

    #[test]
    fn an_empty_system_message_adds_no_turn() {
        let conversation = Conversation {
            system: Some("   ".to_owned()),
            turns: vec![user("Hi")],
            tools: Vec::new(),
        };
        assert!(!conversation.render(true).contains("system"));
    }

    #[test]
    fn content_parts_are_joined_and_reasoning_kept_apart() {
        let content: Content = serde_json::from_str(
            r#"[{"type":"thinking","thinking":"why"},{"type":"text","text":"a"},{"type":"text","text":"b"}]"#,
        )
        .expect("content parts should parse");
        assert_eq!(
            content.split().expect("known parts should fold"),
            ("a\nb".to_owned(), "why".to_owned())
        );
    }

    #[test]
    fn a_bare_string_is_content() {
        let content: Content = serde_json::from_str(r#""hello""#).expect("string should parse");
        assert_eq!(
            content.split().expect("a string should fold"),
            ("hello".to_owned(), String::new())
        );
    }

    fn weather_tool() -> ToolDefinition {
        ToolDefinition::from_openai(&serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                },
            },
        }))
        .expect("a well-formed tool parses")
    }

    fn weather_call() -> ParsedToolCall {
        ParsedToolCall {
            name: "get_weather".to_owned(),
            arguments: serde_json::json!({ "city": "Paris", "days": 3 })
                .as_object()
                .expect("object")
                .clone(),
        }
    }

    #[test]
    fn tools_render_the_templates_system_section_with_the_system_text_after() {
        let conversation = Conversation {
            system: Some("Be terse.".to_owned()),
            turns: vec![user("Hi")],
            tools: vec![weather_tool()],
        };
        let rendered = conversation.render(true);
        let expected_open = "<|im_start|>system\n# Tools\n\nYou have access to the following \
             functions:\n\n<tools>\n{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\
             \"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}}}\
             \n</tools>";
        assert!(rendered.starts_with(expected_open), "{rendered}");
        assert!(rendered.contains("ONLY reply in the following format with NO suffix"));
        assert!(
            rendered.contains("</IMPORTANT>\n\nBe terse.<|im_end|>\n"),
            "the caller's system text must follow the tool section: {rendered}"
        );
    }

    #[test]
    fn a_replayed_tool_call_renders_the_function_parameter_markup() {
        let conversation = Conversation {
            system: None,
            turns: vec![
                user("Weather?"),
                Turn::Assistant {
                    reasoning: String::new(),
                    content: String::new(),
                    tool_calls: vec![weather_call()],
                },
                Turn::ToolResults(vec!["Sunny".to_owned()]),
            ],
            tools: vec![weather_tool()],
        };
        let rendered = conversation.render(true);
        // Empty content: the call follows the header with no blank line. The
        // assistant turn is after the last real query, so its (empty)
        // reasoning is replayed, exactly as the template does.
        assert!(
            rendered.contains(
                "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n\
                 <function=get_weather>\n<parameter=city>\nParis\n</parameter>\n\
                 <parameter=days>\n3\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
            ),
            "{rendered}"
        );
        assert!(
            rendered
                .contains("<|im_start|>user\n<tool_response>\nSunny\n</tool_response><|im_end|>\n"),
            "{rendered}"
        );
    }

    #[test]
    fn a_call_after_content_gets_a_blank_line_and_parallel_calls_one_newline() {
        let conversation = Conversation {
            system: None,
            turns: vec![
                user("Weather?"),
                Turn::Assistant {
                    reasoning: String::new(),
                    content: "Checking two cities.".to_owned(),
                    tool_calls: vec![weather_call(), weather_call()],
                },
            ],
            tools: Vec::new(),
        };
        let rendered = conversation.render(true);
        assert!(
            rendered.contains("Checking two cities.\n\n<tool_call>\n"),
            "{rendered}"
        );
        assert!(
            rendered.contains("</tool_call>\n<tool_call>\n"),
            "{rendered}"
        );
    }

    #[test]
    fn consecutive_tool_results_share_one_user_turn() {
        let conversation = Conversation {
            system: None,
            turns: vec![
                user("Go"),
                Turn::assistant_text(String::new(), "ok".to_owned()),
                Turn::ToolResults(vec!["one".to_owned(), "two".to_owned()]),
            ],
            tools: Vec::new(),
        };
        let rendered = conversation.render(true);
        assert!(
            rendered.contains(
                "<|im_start|>user\n<tool_response>\none\n</tool_response>\n\
                 <tool_response>\ntwo\n</tool_response><|im_end|>\n"
            ),
            "{rendered}"
        );
    }

    #[test]
    fn tool_results_do_not_count_as_the_last_query() {
        // The template's multi_step_tool rule: reasoning stays replayed
        // across an agentic loop's tool steps, because a tool response is not
        // a user question.
        let conversation = Conversation {
            system: None,
            turns: vec![
                user("Weather?"),
                Turn::Assistant {
                    reasoning: "let me check".to_owned(),
                    content: String::new(),
                    tool_calls: vec![weather_call()],
                },
                Turn::ToolResults(vec!["Sunny".to_owned()]),
            ],
            tools: Vec::new(),
        };
        assert!(
            conversation.render(true).contains("let me check"),
            "reasoning before a tool response must be replayed"
        );
    }

    #[test]
    fn anthropic_tool_blocks_fold_into_calls_and_results() {
        let content: Content = serde_json::from_str(
            r#"[{"type":"tool_use","id":"tu_1","name":"get_weather","input":{"city":"Paris"}},
                {"type":"text","text":"done"}]"#,
        )
        .expect("tool_use blocks parse");
        let folded = content.fold().expect("tool blocks fold");
        assert_eq!(folded.tool_calls.len(), 1);
        assert_eq!(folded.tool_calls[0].name, "get_weather");
        assert_eq!(folded.text, "done");

        let result: Content = serde_json::from_str(
            r#"[{"type":"tool_result","tool_use_id":"tu_1","content":[{"type":"text","text":"Sunny"}]}]"#,
        )
        .expect("tool_result blocks parse");
        assert_eq!(
            result.fold().expect("results fold").tool_results,
            vec!["Sunny".to_owned()]
        );
    }

    #[test]
    fn split_refuses_tool_blocks_where_they_have_no_meaning() {
        let content: Content =
            serde_json::from_str(r#"[{"type":"tool_use","id":"tu_1","name":"f","input":{}}]"#)
                .expect("tool_use blocks parse");
        assert!(content.split().is_err());
    }

    #[test]
    fn non_string_parameter_values_render_as_json() {
        assert_eq!(parameter_value(&serde_json::json!("plain")), "plain");
        assert_eq!(parameter_value(&serde_json::json!(true)), "true");
        assert_eq!(
            parameter_value(&serde_json::json!({ "a": 1 })),
            r#"{"a":1}"#
        );
    }

    #[test]
    fn an_image_part_is_refused_by_name() {
        // Serde would report only that no variant of an untagged enum matched,
        // which does not tell the caller this engine is text-only.
        let content: Content =
            serde_json::from_str(r#"[{"type":"image_url","image_url":{"url":"x"}}]"#)
                .expect("an unknown part should still parse");
        let failure = content
            .split()
            .expect_err("an image part should be refused");
        assert!(failure.contains("`image_url`"), "{failure}");
        assert!(failure.contains("text-only"), "{failure}");
    }
}
