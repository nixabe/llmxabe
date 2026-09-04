//! Rendering the prompt with the model's own `tokenizer.chat_template`,
//! behind `--jinja`.
//!
//! # Why there is a choice at all
//!
//! [`Conversation::render`](super::chat::Conversation::render) writes this
//! model's ChatML by hand. That is fast, has no dependencies, and every test
//! in this crate pins its output — but it is a *copy* of the template, and a
//! copy can drift from the original without anything saying so. A GGUF
//! re-quantized against a newer template, or the dense sibling shipping a
//! different one, would be rendered by the wrong rules and look like a dumber
//! model rather than a bug.
//!
//! `--jinja` removes the copy from the loop: the template in the file is
//! evaluated, as llama.cpp does with its bundled `minja`. It costs a template
//! evaluation per request and depends on the template being one this engine
//! can run.
//!
//! # The Python that Jinja does not have
//!
//! Hugging Face chat templates are written against Python's Jinja, and call
//! Python's own string methods — `startswith`, `endswith`, `split`, `strip`.
//! Jinja proper has none of those, so they are supplied here through
//! minijinja's unknown-method hook. This is the same set llama.cpp's `minja`
//! implements, and for the same reason.
//!
//! # What the template is given
//!
//! The same conversation the hand-renderer walks, converted back to the
//! OpenAI message shape the template expects — including two normalizations
//! llama.cpp applies before rendering (`common/chat.cpp`, `workaround::`):
//! tool-call `arguments` are objects rather than JSON strings
//! (`func_args_not_string`), and an assistant turn that only called tools
//! still carries a `content` (`requires_non_null_content`).

use std::sync::Arc;

use minijinja::value::{Value as JinjaValue, ValueKind};
use minijinja::{Environment, Error, ErrorKind, State, context};
use serde_json::{Map, Value, json};
use tracing::warn;

use super::chat::{Conversation, Turn};

/// A compiled `tokenizer.chat_template`, ready to render prompts.
pub(crate) struct ChatTemplate {
    env: Environment<'static>,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ChatTemplate")
    }
}

impl ChatTemplate {
    /// Compile a template source. A source this engine cannot parse fails
    /// here, at startup, rather than on the first request that needs it.
    pub(crate) fn compile(source: String) -> Result<Arc<Self>, String> {
        let mut env = Environment::new();
        env.set_unknown_method_callback(python_method);
        // Templates call this to reject inputs they cannot render — a system
        // message holding an image, say. It has to fail the render, not
        // return a string.
        env.add_function(
            "raise_exception",
            |message: String| -> Result<String, Error> {
                Err(Error::new(ErrorKind::InvalidOperation, message))
            },
        );
        env.add_template_owned("chat", source)
            .map_err(|error| format!("chat template does not parse: {error:#}"))?;
        Ok(Arc::new(Self { env }))
    }

    /// Render a conversation, ending with the assistant's generation prompt.
    ///
    /// `thinking` is the template's own `enable_thinking`, which is what
    /// decides whether the `<think>` block is left open for the model or
    /// handed back already closed.
    pub(crate) fn render(
        &self,
        conversation: &Conversation,
        thinking: bool,
    ) -> Result<String, String> {
        // A system turn that is not one of the leading ones is dropped by
        // this family of templates — their message loop skips every
        // `system`/`developer` past the first two. llama.cpp inherits that
        // silently, because it renders what the template says; the loss is
        // said out loud here instead, because the caller wrote something that
        // is not going to reach the model.
        if conversation
            .turns
            .iter()
            .any(|turn| matches!(turn, Turn::System(_)))
        {
            warn!(
                "a system message after the start of the conversation is dropped by this \
                 model's own chat template; --no-jinja renders it"
            );
        }
        let messages = messages(conversation);
        let tools: Vec<&Value> = conversation
            .tools
            .iter()
            .map(|tool| &tool.wrapper)
            .collect();
        let template = self
            .env
            .get_template("chat")
            .expect("the template was added at construction");
        template
            .render(context! {
                messages => JinjaValue::from_serialize(&messages),
                // An empty list would still take the template's `# Tools`
                // branch on some templates; absent is what "no tools" means.
                tools => (!tools.is_empty()).then(|| JinjaValue::from_serialize(&tools)),
                add_generation_prompt => true,
                enable_thinking => thinking,
            })
            .map_err(|error| format!("chat template failed to render: {error:#}"))
    }
}

/// The conversation as the OpenAI message list a chat template expects.
fn messages(conversation: &Conversation) -> Vec<Value> {
    let mut messages = Vec::new();
    if let Some(system) = conversation
        .system
        .as_deref()
        .map(str::trim)
        .filter(|system| !system.is_empty())
    {
        messages.push(json!({ "role": "system", "content": system }));
    }
    for turn in &conversation.turns {
        match turn {
            Turn::User(content) => {
                messages.push(json!({ "role": "user", "content": content.trim() }));
            }
            Turn::System(content) => {
                messages.push(json!({ "role": "system", "content": content.trim() }));
            }
            Turn::Assistant {
                reasoning,
                content,
                tool_calls,
            } => {
                let mut message = Map::new();
                message.insert("role".to_owned(), json!("assistant"));
                // Never null: a template that concatenates `content` would
                // fail on one. llama.cpp's `requires_non_null_content`.
                message.insert("content".to_owned(), json!(content.trim()));
                if !reasoning.trim().is_empty() {
                    message.insert("reasoning_content".to_owned(), json!(reasoning.trim()));
                }
                if !tool_calls.is_empty() {
                    // Arguments as an object, not as the JSON string the wire
                    // carries: the template tests `is mapping` and would
                    // silently render no parameters at all for a string.
                    // llama.cpp's `func_args_not_string`.
                    message.insert(
                        "tool_calls".to_owned(),
                        Value::Array(
                            tool_calls
                                .iter()
                                .map(|call| {
                                    json!({
                                        "type": "function",
                                        "function": {
                                            "name": call.name,
                                            "arguments": Value::Object(call.arguments.clone()),
                                        },
                                    })
                                })
                                .collect(),
                        ),
                    );
                }
                messages.push(Value::Object(message));
            }
            Turn::ToolResults(results) => {
                for result in results {
                    messages.push(json!({ "role": "tool", "content": result.trim() }));
                }
            }
        }
    }
    messages
}

/// Python's string and mapping methods, which Jinja itself does not define.
fn python_method(
    _state: &State,
    object: &JinjaValue,
    name: &str,
    args: &[JinjaValue],
) -> Result<JinjaValue, Error> {
    let text = || -> Result<&str, Error> {
        object.as_str().ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidOperation,
                format!("`{name}` called on a value that is not a string"),
            )
        })
    };
    let cut = |index: usize| args.get(index).and_then(JinjaValue::as_str);
    let needle = |index: usize| -> Result<&str, Error> {
        cut(index).ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidOperation,
                format!("`{name}` needs a string argument"),
            )
        })
    };
    match name {
        "startswith" => Ok(JinjaValue::from(text()?.starts_with(needle(0)?))),
        "endswith" => Ok(JinjaValue::from(text()?.ends_with(needle(0)?))),
        "strip" => Ok(JinjaValue::from(trim(text()?, cut(0), true, true))),
        "lstrip" => Ok(JinjaValue::from(trim(text()?, cut(0), true, false))),
        "rstrip" => Ok(JinjaValue::from(trim(text()?, cut(0), false, true))),
        "upper" => Ok(JinjaValue::from(text()?.to_uppercase())),
        "lower" => Ok(JinjaValue::from(text()?.to_lowercase())),
        "split" => {
            let parts: Vec<JinjaValue> = match cut(0) {
                Some(separator) => text()?
                    .split(separator)
                    .map(|part| JinjaValue::from(part.to_owned()))
                    .collect(),
                None => text()?
                    .split_whitespace()
                    .map(|part| JinjaValue::from(part.to_owned()))
                    .collect(),
            };
            Ok(JinjaValue::from(parts))
        }
        "items" if object.kind() == ValueKind::Map => {
            let mut pairs = Vec::new();
            for key in object.try_iter()? {
                let value = object.get_item(&key)?;
                pairs.push(JinjaValue::from(vec![key, value]));
            }
            Ok(JinjaValue::from(pairs))
        }
        "get" if object.kind() == ValueKind::Map => {
            let key = args.first().cloned().unwrap_or(JinjaValue::UNDEFINED);
            let found = object.get_item(&key).unwrap_or(JinjaValue::UNDEFINED);
            Ok(if found.is_undefined() {
                args.get(1).cloned().unwrap_or(JinjaValue::from(()))
            } else {
                found
            })
        }
        other => Err(Error::new(
            ErrorKind::UnknownMethod,
            format!("this engine has no `{other}` method for chat templates"),
        )),
    }
}

/// Python's `strip` family: a set of characters to remove, or whitespace.
fn trim(text: &str, chars: Option<&str>, start: bool, end: bool) -> String {
    let mut slice = text;
    match chars {
        Some(chars) => {
            if start {
                slice = slice.trim_start_matches(|ch| chars.contains(ch));
            }
            if end {
                slice = slice.trim_end_matches(|ch| chars.contains(ch));
            }
        }
        None => {
            if start {
                slice = slice.trim_start();
            }
            if end {
                slice = slice.trim_end();
            }
        }
    }
    slice.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::tools::{OfferedTools, ParsedToolCall};

    /// A stand-in for the shape of the real template, small enough to read.
    const TEMPLATE: &str = r#"{%- if tools %}
{{- '<|im_start|>system\n<tools>' }}
{%- for tool in tools %}
{{- '\n' + (tool | tojson) }}
{%- endfor %}
{{- '\n</tools><|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
{%- set content = message.content | trim %}
{%- if message.role == 'user' %}
{{- '<|im_start|>user\n' + content + '<|im_end|>\n' }}
{%- if content.startswith('<tool_response>') %}{{- '' }}{%- endif %}
{%- elif message.role == 'assistant' %}
{{- '<|im_start|>assistant\n' + content }}
{%- for call in message.tool_calls %}
{{- '\n<tool_call>\n<function=' + call.function.name + '>\n' }}
{%- if call.function.arguments is mapping %}
{%- for key in call.function.arguments %}
{{- '<parameter=' + key + '>\n' + (call.function.arguments[key] | string) + '\n</parameter>\n' }}
{%- endfor %}
{%- endif %}
{{- '</function>\n</tool_call>' }}
{%- endfor %}
{{- '<|im_end|>\n' }}
{%- elif message.role == 'tool' %}
{{- '<|im_start|>user\n<tool_response>\n' + content + '\n</tool_response><|im_end|>\n' }}
{%- endif %}
{%- endfor %}
{{- '<|im_start|>assistant\n' }}
{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\n\n</think>\n\n' }}{%- endif %}"#;

    fn conversation() -> Conversation {
        let mut conversation = Conversation {
            system: Some("  be terse  ".to_owned()),
            ..Conversation::default()
        };
        conversation.tools = OfferedTools::from_openai(&[serde_json::json!({
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {
                    "type": "object",
                    "properties": { "filePath": { "type": "string" } },
                },
            },
        })])
        .expect("tools parse")
        .definitions;
        conversation.turns.push(Turn::User("read it".to_owned()));
        conversation.turns.push(Turn::Assistant {
            reasoning: "thinking".to_owned(),
            content: String::new(),
            tool_calls: vec![ParsedToolCall {
                name: "read".to_owned(),
                arguments: serde_json::json!({ "filePath": "a.rs" })
                    .as_object()
                    .expect("object")
                    .clone(),
            }],
        });
        conversation
            .turns
            .push(Turn::ToolResults(vec!["contents".to_owned()]));
        conversation
    }

    #[test]
    fn a_conversation_renders_through_the_template() {
        let template = ChatTemplate::compile(TEMPLATE.to_owned()).expect("compiles");
        let rendered = template
            .render(&conversation(), true)
            .expect("the conversation renders");
        // The tool schema reached the template, the replayed call kept its
        // parameter markup, and the result became a `<tool_response>` turn.
        assert!(rendered.contains(r#"{"type":"function","function":{"name":"read""#));
        assert!(rendered.contains("<tool_call>\n<function=read>\n<parameter=filePath>\na.rs\n"));
        assert!(rendered.contains("<tool_response>\ncontents\n</tool_response>"));
        assert!(rendered.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn thinking_off_closes_the_block_the_template_opens() {
        let template = ChatTemplate::compile(TEMPLATE.to_owned()).expect("compiles");
        let rendered = template.render(&conversation(), false).expect("renders");
        assert!(rendered.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn a_tool_calls_arguments_reach_the_template_as_a_mapping() {
        // A JSON *string* here renders no parameters at all, silently. The
        // conversion is what keeps a replayed call from losing its arguments.
        let template = ChatTemplate::compile(TEMPLATE.to_owned()).expect("compiles");
        let rendered = template.render(&conversation(), true).expect("renders");
        assert!(
            rendered.contains("<parameter=filePath>"),
            "arguments must arrive as an object: {rendered}"
        );
    }

    #[test]
    fn python_string_methods_the_templates_use_are_available() {
        let template = ChatTemplate::compile(
            "{{ 'ab' if ' x '.strip() == 'x' and 'abc'.startswith('a') \
             and 'abc'.endswith('c') and 'a,b'.split(',')[1] == 'b' else 'no' }}"
                .to_owned(),
        )
        .expect("compiles");
        assert_eq!(
            template
                .render(&Conversation::default(), true)
                .expect("renders"),
            "ab"
        );
    }

    /// The hand-written ChatML and the model's own template, on the same
    /// conversation, byte for byte.
    ///
    /// This is the gate the hand-renderer never had. It is a *copy* of the
    /// template, and nothing else in this crate would notice it drifting —
    /// a re-quantized GGUF carrying an updated template would just make the
    /// model look worse. Here the two are compared directly.
    #[test]
    fn the_hand_written_chatml_matches_the_models_own_template() {
        let path = std::env::var_os("LLMXABE_MODEL")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(crate::tokenizer::DEFAULT_MODEL_PATH));
        if !path.exists() {
            eprintln!("SKIP: model GGUF is not available at {}", path.display());
            return;
        }
        let source = crate::tokenizer::chat_template_from_gguf(&path)
            .expect("the file reads")
            .expect("the model carries a chat template");
        let template = ChatTemplate::compile(source).expect("the model's template compiles");
        for (name, conversation) in fixtures() {
            for thinking in [true, false] {
                assert_eq!(
                    template
                        .render(&conversation, thinking)
                        .expect("the model's template renders"),
                    conversation.render(thinking),
                    "hand-written ChatML and `tokenizer.chat_template` disagree on \
                     `{name}` (enable_thinking = {thinking})"
                );
            }
        }
    }

    /// The conversations the two renderers are held to agree on.
    ///
    /// Every rule the hand-renderer implements is here, because with the
    /// template serving by default the rest of this crate's rendering tests
    /// pin a path that is no longer the one requests take — this is what ties
    /// the two together.
    fn fixtures() -> Vec<(&'static str, Conversation)> {
        let mut all = Vec::new();

        let mut plain = Conversation::default();
        plain.turns.push(Turn::User("hello".to_owned()));
        all.push(("a bare user turn", plain));

        let mut system_only = Conversation {
            system: Some("be terse".to_owned()),
            ..Conversation::default()
        };
        system_only.turns.push(Turn::User("hello".to_owned()));
        all.push(("a system prompt without tools", system_only));

        // Reasoning is replayed only for assistant turns after the caller's
        // last question, and dropped for the ones before it.
        let mut across = Conversation::default();
        across.turns.push(Turn::User("first".to_owned()));
        across.turns.push(Turn::Assistant {
            reasoning: "early thinking".to_owned(),
            content: "an answer".to_owned(),
            tool_calls: Vec::new(),
        });
        across.turns.push(Turn::User("second".to_owned()));
        across.turns.push(Turn::Assistant {
            reasoning: "late thinking".to_owned(),
            content: "another".to_owned(),
            tool_calls: Vec::new(),
        });
        all.push(("reasoning before and after the last question", across));

        all.push(("tools, a call and its result", conversation()));

        // Two calls in one turn, after some content, then two results that
        // share a single user turn.
        let mut parallel = Conversation {
            tools: conversation().tools,
            ..Conversation::default()
        };
        parallel.turns.push(Turn::User("read both".to_owned()));
        parallel.turns.push(Turn::Assistant {
            reasoning: "two files".to_owned(),
            content: "Checking both.".to_owned(),
            tool_calls: vec![call("a.rs"), call("b.rs")],
        });
        parallel
            .turns
            .push(Turn::ToolResults(vec!["one".to_owned(), "two".to_owned()]));
        parallel.turns.push(Turn::User("thanks".to_owned()));
        all.push(("parallel calls and merged results", parallel));

        // A tool loop: the reasoning of an assistant turn that follows a tool
        // result is still replayed, because a tool response is not a question.
        let mut loop_ = Conversation {
            tools: conversation().tools,
            ..Conversation::default()
        };
        loop_.turns.push(Turn::User("go".to_owned()));
        loop_.turns.push(Turn::Assistant {
            reasoning: "step one".to_owned(),
            content: String::new(),
            tool_calls: vec![call("a.rs")],
        });
        loop_
            .turns
            .push(Turn::ToolResults(vec!["contents".to_owned()]));
        loop_.turns.push(Turn::Assistant {
            reasoning: "step two".to_owned(),
            content: String::new(),
            tool_calls: vec![call("b.rs")],
        });
        loop_.turns.push(Turn::ToolResults(vec!["more".to_owned()]));
        all.push(("an agentic loop", loop_));

        // An integer argument, and a value spanning lines.
        let mut widths = Conversation {
            tools: conversation().tools,
            ..Conversation::default()
        };
        widths.turns.push(Turn::User("write it".to_owned()));
        widths.turns.push(Turn::Assistant {
            reasoning: String::new(),
            content: String::new(),
            tool_calls: vec![ParsedToolCall {
                name: "read".to_owned(),
                arguments: serde_json::json!({
                    "filePath": "a\nb\nc",
                    "limit": 7,
                })
                .as_object()
                .expect("object")
                .clone(),
            }],
        });
        all.push(("a multi-line value and an integer", widths));

        all
    }

    fn call(path: &str) -> ParsedToolCall {
        ParsedToolCall {
            name: "read".to_owned(),
            arguments: serde_json::json!({ "filePath": path })
                .as_object()
                .expect("object")
                .clone(),
        }
    }

    /// The one conversation the two renderers deliberately disagree on.
    ///
    /// This model's template drops a `system` message that is not one of the
    /// leading ones; the hand-renderer keeps it. Under the template — the
    /// default — the caller's text does not reach the model, which is what
    /// `render` warns about.
    #[test]
    fn a_late_system_turn_is_dropped_by_the_template_and_kept_by_the_hand_renderer() {
        let path = std::env::var_os("LLMXABE_MODEL")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(crate::tokenizer::DEFAULT_MODEL_PATH));
        if !path.exists() {
            eprintln!("SKIP: model GGUF is not available at {}", path.display());
            return;
        }
        let source = crate::tokenizer::chat_template_from_gguf(&path)
            .expect("the file reads")
            .expect("the model carries a chat template");
        let template = ChatTemplate::compile(source).expect("compiles");
        let mut conversation = Conversation::default();
        conversation.turns.push(Turn::User("hello".to_owned()));
        conversation
            .turns
            .push(Turn::System("injected later".to_owned()));
        conversation.turns.push(Turn::User("again".to_owned()));
        assert!(conversation.render(true).contains("injected later"));
        assert!(
            !template
                .render(&conversation, true)
                .expect("renders")
                .contains("injected later")
        );
    }

    /// The one divergence, pinned so it is a decision and not a surprise.
    ///
    /// The template renders a non-string argument through Python's `str()`,
    /// so a boolean comes out `True`; the hand-renderer writes the JSON
    /// spelling the value arrived as, `true`. Both parse back to the same
    /// argument, and the JSON spelling is what the model is asked to emit in
    /// the first place, so the hand-renderer keeps it.
    #[test]
    fn a_boolean_argument_is_the_one_place_the_two_renderings_differ() {
        let path = std::env::var_os("LLMXABE_MODEL")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(crate::tokenizer::DEFAULT_MODEL_PATH));
        if !path.exists() {
            eprintln!("SKIP: model GGUF is not available at {}", path.display());
            return;
        }
        let source = crate::tokenizer::chat_template_from_gguf(&path)
            .expect("the file reads")
            .expect("the model carries a chat template");
        let template = ChatTemplate::compile(source).expect("compiles");
        let mut conversation = Conversation::default();
        conversation.turns.push(Turn::User("go".to_owned()));
        conversation.turns.push(Turn::Assistant {
            reasoning: String::new(),
            content: String::new(),
            tool_calls: vec![ParsedToolCall {
                name: "edit".to_owned(),
                arguments: serde_json::json!({ "replaceAll": true })
                    .as_object()
                    .expect("object")
                    .clone(),
            }],
        });
        assert!(conversation.render(true).contains("\ntrue\n</parameter>"));
        let through_template = template.render(&conversation, true).expect("renders");
        assert!(
            through_template.contains("\ntrue\n</parameter>")
                || through_template.contains("\nTrue\n</parameter>"),
            "{through_template}"
        );
    }

    #[test]
    fn a_template_that_does_not_parse_fails_at_construction() {
        assert!(ChatTemplate::compile("{% for x in %}".to_owned()).is_err());
    }

    #[test]
    fn a_template_that_raises_reports_its_own_message() {
        let template = ChatTemplate::compile("{{ raise_exception('no images here') }}".to_owned())
            .expect("compiles");
        let failure = template
            .render(&Conversation::default(), true)
            .expect_err("the template raises");
        assert!(failure.contains("no images here"), "{failure}");
    }
}
