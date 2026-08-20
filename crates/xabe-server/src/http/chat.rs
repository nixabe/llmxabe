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
//! The parts that are *not* reproduced are the tool-calling and vision
//! sections, and callers that ask for them are refused rather than served a
//! prompt that quietly drops what they sent.

use std::fmt::Write as _;

use serde::Deserialize;

use super::error::{ApiError, Dialect};

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
/// `thinking` is a reasoning block being replayed.
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

impl Content {
    /// Split content into its answer text and its reasoning text.
    pub(crate) fn split(&self) -> Result<(String, String), String> {
        let parts = match self {
            Self::Text(text) => return Ok((text.clone(), String::new())),
            Self::Parts(parts) => parts,
        };
        let (mut text, mut thinking) = (String::new(), String::new());
        for part in parts {
            let (target, value) = match part {
                Part::Known(
                    KnownPart::Text { text: value }
                    | KnownPart::InputText { text: value }
                    | KnownPart::OutputText { text: value },
                ) => (&mut text, value),
                Part::Known(KnownPart::Thinking { thinking: value }) => (&mut thinking, value),
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
        Ok((text, thinking))
    }

    pub(crate) fn text(&self) -> Result<String, String> {
        self.split().map(|(text, _)| text)
    }
}

/// One conversation turn, after the dialect-specific wrapper is stripped off.
#[derive(Debug)]
pub(crate) enum Turn {
    User(String),
    Assistant { reasoning: String, content: String },
}

/// A conversation ready to render.
#[derive(Debug, Default)]
pub(crate) struct Conversation {
    pub(crate) system: Option<String>,
    pub(crate) turns: Vec<Turn>,
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
        if let Some(system) = &self.system {
            let system = system.trim();
            if !system.is_empty() {
                let _ = write!(out, "{IM_START}system\n{system}{IM_END}\n");
            }
        }
        // An assistant turn replays its reasoning only if it came after the
        // caller's last question; earlier reasoning is dropped, exactly as the
        // template drops it.
        let last_query = self
            .turns
            .iter()
            .rposition(|turn| matches!(turn, Turn::User(_)))
            .unwrap_or(self.turns.len().saturating_sub(1));
        for (index, turn) in self.turns.iter().enumerate() {
            match turn {
                Turn::User(content) => {
                    let _ = write!(out, "{IM_START}user\n{}{IM_END}\n", content.trim());
                }
                Turn::Assistant { reasoning, content } => {
                    let content = content.trim();
                    if index > last_query {
                        let _ = write!(
                            out,
                            "{IM_START}assistant\n<think>\n{}\n</think>\n\n{content}{IM_END}\n",
                            reasoning.trim()
                        );
                    } else {
                        let _ = write!(out, "{IM_START}assistant\n{content}{IM_END}\n");
                    }
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
}

/// Reject a role this server has no prompt markup for.
pub(crate) fn unsupported_role(dialect: Dialect, role: &str) -> ApiError {
    ApiError::bad_request(
        dialect,
        format!(
            "the `{role}` role is not supported: this server does not implement tool calling, \
             so a tool result has nowhere to go"
        ),
    )
}

/// Reject tool definitions rather than drop them.
///
/// The model's template has a tool section and the model is trained for it,
/// but nothing here parses a `<tool_call>` block back out of the output, so a
/// caller that sent tools would get prose describing a call it cannot
/// execute. Saying so is more useful than that.
pub(crate) fn unsupported_tools(dialect: Dialect) -> ApiError {
    ApiError::bad_request(
        dialect,
        "tool calling is not implemented; send a request without `tools`",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(text: &str) -> Turn {
        Turn::User(text.to_owned())
    }

    fn assistant(reasoning: &str, content: &str) -> Turn {
        Turn::Assistant {
            reasoning: reasoning.to_owned(),
            content: content.to_owned(),
        }
    }

    #[test]
    fn a_single_question_renders_the_documented_chatml() {
        let conversation = Conversation {
            system: Some("You are terse.".to_owned()),
            turns: vec![user("Hi")],
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
