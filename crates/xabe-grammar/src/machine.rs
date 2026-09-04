//! The tool-call grammar, compiled to a byte-level nondeterministic machine.
//!
//! # What it accepts
//!
//! The shape is llama.cpp's, where it is built as a PEG and lowered to GBNF.
//! It is the grammar for the *format*, not for a model: llama.cpp picks the
//! format by what the template contains, not by what the model is called
//! (`common/chat.cpp`, the dispatcher — a template holding `<tool_call>`,
//! `<function=` and `<parameter=` takes this path). Qwen3.6-35B-A3B's own
//! template holds all three, so this is its grammar; the builder is still
//! named `common_chat_params_init_qwen3_coder` after the first model to ship
//! the format, and it carries Qwen3.6 too.
//!
//! ```text
//! root        := "<tool_call>\n"? body ( "<tool_call>\n" body )*
//! body        := tool "</tool_call>" space
//! tool        := "<function=" NAME ">\n" args "</function>\n"
//! args        := <required parameters, in any order> <optional parameters>*
//! parameter   := "<parameter=" PNAME ">\n" value
//! value       := <raw text up to the first "\n</parameter>\n">
//!              | <literal> "\n</parameter>\n"
//! ```
//!
//! `NAME` and `PNAME` are literals drawn from the tools the caller offered,
//! which is the whole point: the model cannot name a tool that was not
//! offered, cannot name a parameter the schema does not declare, cannot
//! repeat one, and cannot close a call before every required parameter has
//! been written. llama.cpp accepts the required ones in any order — "as Qwen
//! does not always adhere to the order provided" — and so does this.
//!
//! The leading `<tool_call>` is optional on the *first* call only, matching
//! llama.cpp's `tool-call-first` rule, which exists because the model
//! occasionally omits it.
//!
//! # How it runs
//!
//! States are bytes, not tokens: a cursor is a position in this machine plus
//! the tool it committed to and the bitset of parameters it has written. A
//! set of cursors advances one byte at a time, and a token is admissible
//! exactly when feeding its bytes leaves at least one cursor alive. Turning
//! that into a vocabulary mask is [`crate::vocab`]'s job.

use smallvec::SmallVec;

use crate::spec::{ToolSpec, ValueSpec};

/// The delimiter that closes every parameter value.
///
/// llama.cpp's `arg_close`, `p.literal("\n</parameter>\n")`. For a raw-text
/// value it is also the terminator of the Aho-Corasick "including" grammar
/// (`gbnf_including_grammar`), which stops at its *first* occurrence — so a
/// text value cannot contain one.
pub(crate) const CLOSE: &[u8] = b"\n</parameter>\n";

/// Parameters past this index lose their no-repeat and required-present
/// guards, because the cursor tracks written parameters in one word.
///
/// No tool this server has been offered comes close; the cap is here so a
/// pathological schema degrades to "structure enforced, bookkeeping
/// dropped" rather than being refused or mis-tracked.
const TRACKED_PARAMS: usize = 64;

#[derive(Debug, Clone)]
enum State {
    /// Consume exactly this byte.
    Byte { byte: u8, next: u32 },
    /// Epsilon: any of these.
    Split { alts: SmallVec<[u32; 4]> },
    /// Epsilon: commit the cursor to a tool.
    EnterTool { tool: u16, next: u32 },
    /// Epsilon: write parameter `param`, if it has not been written yet.
    Mark { param: u8, next: u32 },
    /// Epsilon: pass only once every required parameter has been written.
    RequireAll { next: u32 },
    /// Free bytes, `k` of [`CLOSE`] matched so far, leaving on the last one.
    Until { k: u8, next: u32 },
    /// The grammar is satisfied; only end-of-generation may follow.
    Accept,
}

/// One live position in the machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Cursor {
    state: u32,
    /// The tool this cursor committed to, once past `EnterTool`.
    tool: u16,
    /// Parameters already written, by index into that tool's list.
    written: u64,
}

/// A compiled tool-call grammar. Immutable, and shared by every request that
/// offered the same tools.
#[derive(Debug)]
pub struct Machine {
    states: Vec<State>,
    root: u32,
    /// Per tool, the parameters that must be written before `</function>`.
    required: Vec<u64>,
    /// KMP failure function over [`CLOSE`].
    fallback: Vec<u8>,
    /// `<tool_call>`, and `<function=NAME>` for each tool: the strings whose
    /// appearance in the output arms the constraint. llama.cpp's
    /// `grammar_triggers` for this format, and for the same reason — matching
    /// a bare `<function` would eat prose like `#include <functional>`.
    triggers: Vec<Vec<u8>>,
}

/// Scratch the stepping functions reuse, so advancing a cursor set allocates
/// nothing on the decode path.
#[derive(Debug, Default)]
pub(crate) struct StepScratch {
    work: Vec<Cursor>,
    out: Vec<Cursor>,
}

impl Machine {
    /// Compile the grammar for the tools a request offered.
    ///
    /// `parallel` allows more than one call in a turn, which is llama.cpp's
    /// `parallel_tool_calls` — with it off, the grammar accepts exactly one.
    pub fn compile(tools: &[ToolSpec], parallel: bool) -> Self {
        let mut builder = Builder {
            states: Vec::new(),
            required: Vec::new(),
        };
        let root = builder.build(tools, parallel);
        let mut triggers = vec![b"<tool_call>".to_vec()];
        triggers.extend(
            tools
                .iter()
                .map(|tool| format!("<function={}>", tool.name).into_bytes()),
        );
        Self {
            states: builder.states,
            root,
            required: builder.required,
            fallback: kmp_fallback(CLOSE),
            triggers,
        }
    }

    /// The strings whose appearance in the output arms the constraint.
    pub fn triggers(&self) -> &[Vec<u8>] {
        &self.triggers
    }

    /// The cursor set at the start of a call, epsilon-closed.
    pub(crate) fn start(&self, scratch: &mut StepScratch) -> Vec<Cursor> {
        let mut cursors = vec![Cursor {
            state: self.root,
            tool: u16::MAX,
            written: 0,
        }];
        self.close(&mut cursors, scratch);
        cursors
    }

    /// Advance every cursor over one byte. Returns whether any survived.
    pub(crate) fn step(
        &self,
        cursors: &mut Vec<Cursor>,
        byte: u8,
        scratch: &mut StepScratch,
    ) -> bool {
        scratch.out.clear();
        for cursor in cursors.iter() {
            match &self.states[cursor.state as usize] {
                State::Byte { byte: want, next } if *want == byte => {
                    push_unique(
                        &mut scratch.out,
                        Cursor {
                            state: *next,
                            ..*cursor
                        },
                    );
                }
                State::Until { k, next } => {
                    let k = self.advance_close(*k, byte);
                    let state = if k as usize == CLOSE.len() {
                        *next
                    } else {
                        // `Until` states are laid out consecutively from
                        // `k == 0`, so the one for `k` is a fixed offset back.
                        cursor.state - u32::from(self.until_k(cursor.state)) + u32::from(k)
                    };
                    push_unique(&mut scratch.out, Cursor { state, ..*cursor });
                }
                _ => {}
            }
        }
        cursors.clear();
        cursors.append(&mut scratch.out);
        self.close(cursors, scratch);
        !cursors.is_empty()
    }

    /// Whether the machine is satisfied here, so the model may stop.
    pub(crate) fn accepts_end(&self, cursors: &[Cursor]) -> bool {
        cursors
            .iter()
            .any(|cursor| matches!(self.states[cursor.state as usize], State::Accept))
    }

    /// Whether every cursor sits in free text with nothing of [`CLOSE`]
    /// matched, which is the state a long parameter value spends most of its
    /// bytes in and the one worth a shortcut in the mask.
    pub(crate) fn is_free_text(&self, cursors: &[Cursor]) -> bool {
        cursors.iter().all(|cursor| {
            matches!(
                self.states[cursor.state as usize],
                State::Until { k: 0, .. }
            )
        })
    }

    /// The `k` of an `Until` state.
    fn until_k(&self, state: u32) -> u8 {
        match self.states[state as usize] {
            State::Until { k, .. } => k,
            _ => unreachable!("only called on an Until state"),
        }
    }

    /// One byte of the Aho-Corasick scan for [`CLOSE`].
    fn advance_close(&self, mut k: u8, byte: u8) -> u8 {
        loop {
            if byte == CLOSE[k as usize] {
                return k + 1;
            }
            if k == 0 {
                return 0;
            }
            k = self.fallback[k as usize];
        }
    }

    /// Expand epsilon states in place, leaving only byte-consuming ones.
    fn close(&self, cursors: &mut Vec<Cursor>, scratch: &mut StepScratch) {
        scratch.work.clear();
        scratch.work.append(cursors);
        while let Some(cursor) = scratch.work.pop() {
            match &self.states[cursor.state as usize] {
                State::Byte { .. } | State::Until { .. } | State::Accept => {
                    push_unique(cursors, cursor);
                }
                State::Split { alts } => {
                    for &alt in alts {
                        scratch.work.push(Cursor {
                            state: alt,
                            ..cursor
                        });
                    }
                }
                State::EnterTool { tool, next } => scratch.work.push(Cursor {
                    state: *next,
                    tool: *tool,
                    written: 0,
                }),
                State::Mark { param, next } => {
                    let bit = param_bit(*param);
                    if cursor.written & bit == 0 {
                        scratch.work.push(Cursor {
                            state: *next,
                            written: cursor.written | bit,
                            ..cursor
                        });
                    }
                }
                State::RequireAll { next } => {
                    let required = self.required[cursor.tool as usize];
                    if cursor.written & required == required {
                        scratch.work.push(Cursor {
                            state: *next,
                            ..cursor
                        });
                    }
                }
            }
        }
    }
}

/// The bit a parameter index occupies, or none once past [`TRACKED_PARAMS`].
fn param_bit(param: u8) -> u64 {
    if (param as usize) < TRACKED_PARAMS {
        1 << param
    } else {
        0
    }
}

fn push_unique(cursors: &mut Vec<Cursor>, cursor: Cursor) {
    if !cursors.contains(&cursor) {
        cursors.push(cursor);
    }
}

/// The KMP failure function: for each prefix length, the longest proper
/// border, so a mismatch can fall back without rescanning.
fn kmp_fallback(needle: &[u8]) -> Vec<u8> {
    let mut fallback = vec![0u8; needle.len() + 1];
    let mut border = 0usize;
    for at in 1..needle.len() {
        while border > 0 && needle[at] != needle[border] {
            border = fallback[border] as usize;
        }
        if needle[at] == needle[border] {
            border += 1;
        }
        fallback[at + 1] = border as u8;
    }
    fallback
}

struct Builder {
    states: Vec<State>,
    required: Vec<u64>,
}

impl Builder {
    fn add(&mut self, state: State) -> u32 {
        self.states.push(state);
        (self.states.len() - 1) as u32
    }

    /// A chain consuming `bytes`, ending at `next`. Built back to front.
    fn literal(&mut self, bytes: &[u8], next: u32) -> u32 {
        bytes
            .iter()
            .rev()
            .fold(next, |next, &byte| self.add(State::Byte { byte, next }))
    }

    fn split(&mut self, alts: impl IntoIterator<Item = u32>) -> u32 {
        self.add(State::Split {
            alts: alts.into_iter().collect(),
        })
    }

    /// The `Until` chain for a raw-text value, ending at `next`.
    ///
    /// One state per matched prefix of [`CLOSE`], laid out consecutively from
    /// `k == 0` so `step` can address them by offset.
    fn until(&mut self, next: u32) -> u32 {
        let first = self.states.len() as u32;
        for k in 0..CLOSE.len() {
            self.add(State::Until { k: k as u8, next });
        }
        first
    }

    /// One JSON digit.
    fn digit(&mut self, next: u32) -> u32 {
        let alts: SmallVec<[u32; 4]> = (b'0'..=b'9')
            .map(|byte| self.add(State::Byte { byte, next }))
            .collect();
        self.add(State::Split { alts })
    }

    /// One or more digits.
    fn digits(&mut self, next: u32) -> u32 {
        // `more` loops back through itself, so it is allocated first as a
        // placeholder and patched once its successors exist.
        let more = self.add(State::Split {
            alts: SmallVec::new(),
        });
        let again = self.digit(more);
        self.states[more as usize] = State::Split {
            alts: SmallVec::from_slice(&[again, next]),
        };
        self.digit(more)
    }

    /// A JSON number: `-`? ( `0` | [1-9] digit* ) ( `.` digit+ )? exponent?
    ///
    /// `fractional` off leaves the integer grammar, which is what a schema
    /// saying `integer` admits.
    fn number(&mut self, next: u32, fractional: bool) -> u32 {
        // Built back to front: exponent, then fraction, then magnitude.
        let exponent_digits = self.digits(next);
        let plus = self.add(State::Byte {
            byte: b'+',
            next: exponent_digits,
        });
        let minus_exponent = self.add(State::Byte {
            byte: b'-',
            next: exponent_digits,
        });
        let signed = self.split([plus, minus_exponent, exponent_digits]);
        let lower = self.add(State::Byte {
            byte: b'e',
            next: signed,
        });
        let upper = self.add(State::Byte {
            byte: b'E',
            next: signed,
        });
        let exponent = self.split([lower, upper]);
        let mut tail = self.split([next, exponent]);
        if fractional {
            let after_point = self.digits(tail);
            let point = self.add(State::Byte {
                byte: b'.',
                next: after_point,
            });
            tail = self.split([tail, point]);
        }
        let zero = self.add(State::Byte {
            byte: b'0',
            next: tail,
        });
        let more = self.digits(tail);
        let rest = self.split([tail, more]);
        let leading: Vec<u32> = (b'1'..=b'9')
            .map(|byte| self.add(State::Byte { byte, next: rest }))
            .collect();
        let magnitude = self.split(std::iter::once(zero).chain(leading));
        let minus = self.add(State::Byte {
            byte: b'-',
            next: magnitude,
        });
        self.split([magnitude, minus])
    }

    /// The value of one parameter, ending back at its tool's argument loop.
    fn value(&mut self, spec: &ValueSpec, args: u32) -> u32 {
        match spec {
            ValueSpec::Text => self.until(args),
            other => {
                let close = self.literal(CLOSE, args);
                match other {
                    ValueSpec::Literals(literals) => {
                        let alts: Vec<u32> = literals
                            .iter()
                            .map(|literal| self.literal(literal.as_bytes(), close))
                            .collect();
                        self.split(alts)
                    }
                    ValueSpec::Boolean => {
                        let yes = self.literal(b"true", close);
                        let no = self.literal(b"false", close);
                        self.split([yes, no])
                    }
                    ValueSpec::Null => self.literal(b"null", close),
                    ValueSpec::Integer => self.number(close, false),
                    ValueSpec::Number => self.number(close, true),
                    ValueSpec::Text => unreachable!("handled above"),
                }
            }
        }
    }

    fn build(&mut self, tools: &[ToolSpec], parallel: bool) -> u32 {
        let accept = self.add(State::Accept);
        // `after` and `entry` are mutually recursive when parallel calls are
        // allowed, so both start as placeholders.
        let after = self.add(State::Split {
            alts: SmallVec::new(),
        });
        let entry = self.add(State::Split {
            alts: SmallVec::new(),
        });
        let call_close = self.literal(b"</tool_call>", after);

        let mut openers = SmallVec::<[u32; 4]>::new();
        for (index, tool) in tools.iter().enumerate() {
            let mut required = 0u64;
            for (param, spec) in tool.params.iter().enumerate() {
                if spec.required {
                    required |= param_bit(param as u8);
                }
            }
            self.required.push(required);

            let args = self.add(State::Split {
                alts: SmallVec::new(),
            });
            let mut alts = SmallVec::<[u32; 4]>::new();
            for (param, spec) in tool.params.iter().enumerate() {
                let value = self.value(&spec.value, args);
                let open = self.literal(format!("<parameter={}>\n", spec.name).as_bytes(), value);
                alts.push(self.add(State::Mark {
                    param: param as u8,
                    next: open,
                }));
            }
            let closing = self.literal(b"</function>\n", call_close);
            alts.push(self.add(State::RequireAll { next: closing }));
            self.states[args as usize] = State::Split { alts };

            let named = self.literal(format!("<function={}>\n", tool.name).as_bytes(), args);
            openers.push(self.add(State::EnterTool {
                tool: index as u16,
                next: named,
            }));
        }
        self.states[entry as usize] = State::Split { alts: openers };

        // `p.space()` after `</tool_call>`: whitespace is markup between
        // calls, and trailing whitespace does not un-satisfy the grammar.
        let mut tail = SmallVec::<[u32; 4]>::from_slice(&[accept]);
        for &byte in b" \t\n\r" {
            tail.push(self.add(State::Byte { byte, next: after }));
        }
        if parallel {
            let again = self.literal(b"<tool_call>\n", entry);
            tail.push(again);
        }
        self.states[after as usize] = State::Split { alts: tail };

        // The first call may omit its `<tool_call>` line; llama.cpp's
        // `tool-call-first`.
        let opened = self.literal(b"<tool_call>\n", entry);
        self.split([opened, entry])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::ParamSpec;

    fn read_tool() -> ToolSpec {
        ToolSpec {
            name: "read".to_owned(),
            params: vec![
                ParamSpec {
                    name: "filePath".to_owned(),
                    required: true,
                    value: ValueSpec::Text,
                },
                ParamSpec {
                    name: "limit".to_owned(),
                    required: false,
                    value: ValueSpec::Integer,
                },
            ],
        }
    }

    /// Feed `text` and report whether the machine survived it, and whether it
    /// would let the model stop there.
    fn run(machine: &Machine, text: &str) -> Option<bool> {
        let mut scratch = StepScratch::default();
        let mut cursors = machine.start(&mut scratch);
        for &byte in text.as_bytes() {
            if !machine.step(&mut cursors, byte, &mut scratch) {
                return None;
            }
        }
        Some(machine.accepts_end(&cursors))
    }

    #[test]
    fn a_well_formed_call_runs_to_a_stopping_point() {
        let machine = Machine::compile(&[read_tool()], true);
        assert_eq!(
            run(
                &machine,
                "<tool_call>\n<function=read>\n<parameter=filePath>\nE:/a.js\n</parameter>\n</function>\n</tool_call>"
            ),
            Some(true)
        );
    }

    #[test]
    fn a_parameter_the_schema_does_not_declare_is_refused_at_its_first_wrong_byte() {
        let machine = Machine::compile(&[read_tool()], true);
        // The reported failure: the model wrote `file_path` for a schema that
        // declares `filePath`. It cannot get past the `_`.
        assert_eq!(
            run(&machine, "<tool_call>\n<function=read>\n<parameter=file_"),
            None
        );
        assert!(
            run(&machine, "<tool_call>\n<function=read>\n<parameter=file").is_some(),
            "the shared prefix is still open"
        );
    }

    #[test]
    fn a_tool_that_was_not_offered_cannot_be_named() {
        let machine = Machine::compile(&[read_tool()], true);
        assert_eq!(run(&machine, "<tool_call>\n<function=list_mcp"), None);
    }

    #[test]
    fn a_call_cannot_close_before_its_required_parameters_are_written() {
        let machine = Machine::compile(&[read_tool()], true);
        // `</function>` is unreachable until `filePath` has been written.
        assert_eq!(run(&machine, "<tool_call>\n<function=read>\n</f"), None);
        assert_eq!(
            run(
                &machine,
                "<tool_call>\n<function=read>\n<parameter=limit>\n7\n</parameter>\n</f"
            ),
            None
        );
    }

    #[test]
    fn a_parameter_cannot_be_written_twice() {
        let machine = Machine::compile(&[read_tool()], true);
        assert_eq!(
            run(
                &machine,
                "<tool_call>\n<function=read>\n<parameter=filePath>\na\n</parameter>\n<parameter=f"
            ),
            None
        );
    }

    #[test]
    fn required_parameters_may_arrive_in_any_order() {
        let machine = Machine::compile(&[read_tool()], true);
        assert_eq!(
            run(
                &machine,
                "<tool_call>\n<function=read>\n<parameter=limit>\n7\n</parameter>\n\
                 <parameter=filePath>\na\n</parameter>\n</function>\n</tool_call>"
            ),
            Some(true)
        );
    }

    #[test]
    fn the_opening_line_is_optional_on_the_first_call_only() {
        let machine = Machine::compile(&[read_tool()], true);
        let body =
            "<function=read>\n<parameter=filePath>\na\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(run(&machine, body), Some(true));
        // A second call must open properly.
        assert_eq!(run(&machine, &format!("{body}<function=read>")), None);
        assert_eq!(
            run(&machine, &format!("{body}\n<tool_call>\n<function=read>")),
            Some(false)
        );
    }

    #[test]
    fn one_call_is_all_a_serial_grammar_admits() {
        let machine = Machine::compile(&[read_tool()], false);
        let body =
            "<function=read>\n<parameter=filePath>\na\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(run(&machine, body), Some(true));
        assert_eq!(run(&machine, &format!("{body}\n<tool_call>\n")), None);
    }

    #[test]
    fn a_typed_value_takes_its_own_shape() {
        let machine = Machine::compile(&[read_tool()], true);
        let head = "<tool_call>\n<function=read>\n<parameter=filePath>\na\n</parameter>\n\
                    <parameter=limit>\n";
        assert!(run(&machine, &format!("{head}-12\n</parameter>\n")).is_some());
        assert_eq!(run(&machine, &format!("{head}seven")), None);
        // An integer parameter takes no fractional part.
        assert_eq!(run(&machine, &format!("{head}1.")), None);
    }

    #[test]
    fn a_boolean_and_an_enum_take_only_their_own_spellings() {
        let tool = ToolSpec {
            name: "edit".to_owned(),
            params: vec![
                ParamSpec {
                    name: "mode".to_owned(),
                    required: true,
                    value: ValueSpec::Literals(vec!["1".to_owned(), "2".to_owned()]),
                },
                ParamSpec {
                    name: "replaceAll".to_owned(),
                    required: false,
                    value: ValueSpec::Boolean,
                },
            ],
        };
        let machine = Machine::compile(&[tool], true);
        let head = "<tool_call>\n<function=edit>\n<parameter=mode>\n";
        assert!(run(&machine, &format!("{head}1\n</parameter>\n")).is_some());
        assert_eq!(run(&machine, &format!("{head}3")), None);
        let with_mode = format!("{head}2\n</parameter>\n<parameter=replaceAll>\n");
        assert!(run(&machine, &format!("{with_mode}true\n</parameter>\n")).is_some());
        assert!(run(&machine, &format!("{with_mode}false\n</parameter>\n")).is_some());
        // Python's spelling, which the model sees in its own replayed
        // history under some templates, is not JSON and is not admitted.
        assert_eq!(run(&machine, &format!("{with_mode}True")), None);
    }

    #[test]
    fn a_text_value_ends_at_its_first_delimiter() {
        let machine = Machine::compile(&[read_tool()], true);
        let head = "<tool_call>\n<function=read>\n<parameter=filePath>\n";
        // Anything at all inside the value, including markup that is not the
        // delimiter.
        assert!(
            run(
                &machine,
                &format!("{head}a </parameter> b\n</parameter>\n</function>\n</tool_call>")
            )
            .is_some()
        );
        // But once the delimiter has been written the value is over, so more
        // value bytes have nowhere to go.
        assert_eq!(run(&machine, &format!("{head}a\n</parameter>\nmore")), None);
    }

    #[test]
    fn the_triggers_are_the_opener_and_every_offered_function() {
        let machine = Machine::compile(&[read_tool()], true);
        assert_eq!(
            machine.triggers(),
            [b"<tool_call>".to_vec(), b"<function=read>".to_vec()]
        );
    }
}
