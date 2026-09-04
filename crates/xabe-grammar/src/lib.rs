//! Grammar-constrained tool calling.
//!
//! When a request offers tools, llama.cpp does not merely parse whatever the
//! model produced — it constrains what the model may produce. Its server
//! attaches a lazy GBNF grammar to every `/v1/chat/completions` request that
//! carries `tools` (`tools/server/server-common.cpp`, `grammar_type =
//! "tool_calls"`), built by the initializer its template dispatches to
//! (`common/chat.cpp`). Until a trigger appears the model writes freely; from
//! the trigger to the end of the call every byte is checked against the
//! tools the caller actually offered.
//!
//! Which initializer that is comes from the *template*, never from the model
//! name: one holding `<tool_call>`, `<function=` and `<parameter=` takes the
//! XML-parameter path, and Qwen3.6-35B-A3B's template holds all three. Its
//! `<think>` block is handled on the same path — llama.cpp reads reasoning
//! support off the template too, and ends the reasoning span at `</think>`
//! or `<tool_call>`, which is what `generate.rs` does here.
//!
//! Without it the model is free to invent, and does: a parameter named
//! `file_path` where the schema says `filePath`, a tool that was never
//! offered, a `</function>` before a required parameter has been written, a
//! missing `</tool_call>` that turns the whole call back into prose. Each of
//! those reaches the client as a broken call or as no call at all.
//!
//! The pieces:
//!
//! - [`ToolSpec`] reads the offered schemas — which parameters exist, which
//!   are required, and which are written as raw text.
//! - [`Machine`] compiles them into the byte-level grammar and runs it.
//! - [`Vocab`] holds the token pieces and turns a grammar position into a
//!   logit mask.
//! - [`ToolConstraint`] is what a sequence carries: idle until a trigger
//!   appears in the output, masking every step after that.
//!
//! # Where this stops short of llama.cpp
//!
//! A parameter whose schema is an object or an array is constrained to its
//! surrounding markup but not inside its own value; llama.cpp runs the
//! parameter's full schema through `json_schema_to_grammar` and constrains
//! the value too. Scalars — strings, enums, booleans, numbers, `null` — are
//! constrained here as they are there. See [`spec::ValueSpec`].

mod machine;
mod spec;
mod vocab;

pub use machine::Machine;
pub use spec::{ParamSpec, ToolSpec, ValueSpec};
pub use vocab::Vocab;

use std::sync::Arc;

use machine::{Cursor, StepScratch};
use vocab::MaskScratch;

/// The tool-call grammar for one request, and the vocabulary to mask it
/// against.
///
/// Compiled once per request and shared by the sequence that carries it; the
/// vocabulary is built once at startup and shared by every request.
#[derive(Debug)]
pub struct ToolGrammar {
    machine: Machine,
    vocab: Arc<Vocab>,
}

impl ToolGrammar {
    /// Compile the grammar for `tools`. `parallel` admits more than one call
    /// in a turn, llama.cpp's `parallel_tool_calls`.
    ///
    /// Returns `None` when there is nothing to constrain — no tools, so no
    /// call the model could get wrong.
    pub fn new(tools: &[ToolSpec], parallel: bool, vocab: Arc<Vocab>) -> Option<Self> {
        if tools.is_empty() {
            return None;
        }
        Some(Self {
            machine: Machine::compile(tools, parallel),
            vocab,
        })
    }
}

/// One sequence's position in its request's tool-call grammar.
///
/// Idle — and free — until one of the grammar's triggers appears in the
/// output. From there every step is masked, until the call ends and the
/// model stops.
#[derive(Debug)]
pub struct ToolConstraint {
    grammar: Arc<ToolGrammar>,
    /// The tail of the output, long enough to hold the longest trigger.
    tail: Vec<u8>,
    cursors: Vec<Cursor>,
    armed: bool,
    /// Set when a byte the machine could not take slipped through anyway,
    /// which can only happen if a token was chosen without consulting the
    /// mask. The constraint stands down rather than deadlocking the request.
    broken: bool,
    step: StepScratch,
    mask: MaskScratch,
    /// The emitted token's bytes, reused so [`Self::observe`] allocates
    /// nothing.
    piece: Vec<u8>,
    longest_trigger: usize,
}

impl ToolConstraint {
    pub fn new(grammar: Arc<ToolGrammar>) -> Self {
        let longest_trigger = grammar
            .machine
            .triggers()
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(0);
        Self {
            grammar,
            tail: Vec::with_capacity(longest_trigger),
            cursors: Vec::new(),
            armed: false,
            broken: false,
            step: StepScratch::default(),
            mask: MaskScratch::default(),
            piece: Vec::new(),
            longest_trigger,
        }
    }

    /// Whether the grammar is currently constraining output, and so whether
    /// this step needs its logits on the host at all.
    pub fn is_masking(&self) -> bool {
        self.armed && !self.broken
    }

    /// How many tokens the mask covers, which is the vocabulary width the
    /// logits row must have.
    pub fn vocab_len(&self) -> usize {
        self.grammar.vocab.len()
    }

    /// Exclude every token the grammar cannot take next.
    ///
    /// A no-op while the constraint is idle. Callers should gate on
    /// [`Self::is_masking`] rather than pay for the logits read-back.
    pub fn mask(&mut self, logits: &mut [f32]) {
        if !self.is_masking() {
            return;
        }
        self.grammar
            .vocab
            .mask(&self.grammar.machine, &self.cursors, &mut self.mask, logits);
    }

    /// Feed back the bytes of the token that was actually emitted.
    ///
    /// Before the constraint is armed this watches for a trigger; after it,
    /// this is what advances the grammar. The trigger's own bytes are
    /// replayed into the machine, exactly as llama.cpp applies the matched
    /// trigger word to a lazy grammar before constraining the next token.
    pub fn observe(&mut self, token: i32) {
        if token < 0 {
            return;
        }
        // The piece is looked up rather than passed in: the decode loop
        // holds token ids and no tokenizer, and the vocabulary is right
        // here. Through a buffer the constraint owns, so a step on the
        // decode path allocates nothing.
        let mut piece = std::mem::take(&mut self.piece);
        piece.clear();
        piece.extend_from_slice(self.grammar.vocab.piece(token as usize));
        self.observe_bytes(&piece);
        self.piece = piece;
    }

    fn observe_bytes(&mut self, piece: &[u8]) {
        if self.broken {
            return;
        }
        if self.armed {
            for &byte in piece {
                if !self
                    .grammar
                    .machine
                    .step(&mut self.cursors, byte, &mut self.step)
                {
                    self.broken = true;
                    return;
                }
            }
            return;
        }
        // Idle: keep just enough of the tail that a trigger split across
        // tokens is still seen whole.
        self.tail.extend_from_slice(piece);
        if let Some(at) = self.trigger_at() {
            let matched = self.tail[at..].to_vec();
            self.cursors = self.grammar.machine.start(&mut self.step);
            self.armed = true;
            self.tail.clear();
            self.observe_bytes(&matched);
            return;
        }
        let keep = self.longest_trigger.saturating_sub(1);
        if self.tail.len() > keep {
            self.tail.drain(..self.tail.len() - keep);
        }
    }

    /// Where in the tail the earliest trigger starts.
    fn trigger_at(&self) -> Option<usize> {
        self.grammar
            .machine
            .triggers()
            .iter()
            .filter_map(|trigger| find(&self.tail, trigger))
            .min()
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    (0..=haystack.len() - needle.len()).find(|&at| &haystack[at..at + needle.len()] == needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn vocab() -> Arc<Vocab> {
        // A tiny byte-level vocabulary: every ASCII byte, plus the two
        // multi-byte pieces that matter here, plus one end-of-generation
        // token at the end.
        let mut pieces: Vec<Vec<u8>> = (0u8..=127).map(|byte| vec![byte]).collect();
        pieces.push(b"<tool_call>".to_vec());
        pieces.push(b"</tool_call>".to_vec());
        pieces.push(b"filePath".to_vec());
        pieces.push(b"file_path".to_vec());
        pieces.push(b"<|im_end|>".to_vec());
        Arc::new(Vocab::new(&pieces, &[132]))
    }

    fn grammar() -> Arc<ToolGrammar> {
        let tool = ToolSpec::from_wrapper(&json!({
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {
                    "type": "object",
                    "properties": { "filePath": { "type": "string" } },
                    "required": ["filePath"],
                },
            },
        }))
        .expect("a named function parses");
        Arc::new(ToolGrammar::new(&[tool], true, vocab()).expect("one tool compiles"))
    }

    /// The token ids of the single-byte pieces spelling `text`.
    fn bytes(text: &str) -> Vec<u8> {
        text.as_bytes().to_vec()
    }

    fn allowed(constraint: &mut ToolConstraint) -> Vec<usize> {
        let mut logits = vec![0.0f32; constraint.vocab_len()];
        constraint.mask(&mut logits);
        logits
            .iter()
            .enumerate()
            .filter(|(_, logit)| logit.is_finite())
            .map(|(index, _)| index)
            .collect()
    }

    #[test]
    fn nothing_is_masked_until_a_trigger_appears() {
        let mut constraint = ToolConstraint::new(grammar());
        constraint.observe_bytes(&bytes("Let me look at that file."));
        assert!(!constraint.is_masking());
        let mut logits = vec![0.0f32; constraint.vocab_len()];
        constraint.mask(&mut logits);
        assert!(logits.iter().all(|logit| logit.is_finite()));
    }

    #[test]
    fn the_trigger_arms_the_constraint_and_its_own_bytes_advance_it() {
        let mut constraint = ToolConstraint::new(grammar());
        constraint.observe_bytes(&bytes("sure <tool_call>"));
        assert!(constraint.is_masking());
        // The grammar is now inside `<tool_call>`, one byte short of the
        // newline the template writes next.
        assert_eq!(allowed(&mut constraint), vec![b'\n' as usize]);
    }

    #[test]
    fn a_trigger_split_across_tokens_is_still_seen() {
        let mut constraint = ToolConstraint::new(grammar());
        constraint.observe_bytes(&bytes("<tool"));
        assert!(!constraint.is_masking());
        constraint.observe_bytes(&bytes("_call>"));
        assert!(constraint.is_masking());
    }

    #[test]
    fn only_the_offered_parameter_name_survives_the_mask() {
        let mut constraint = ToolConstraint::new(grammar());
        constraint.observe_bytes(&bytes("<tool_call>\n<function=read>\n<parameter="));
        let allowed = allowed(&mut constraint);
        // `filePath` as one token is admissible; `file_path` is not, and
        // neither is the `_` that would start it.
        assert!(allowed.contains(&130), "the schema's own name is allowed");
        assert!(!allowed.contains(&131), "the model's misspelling is not");
        assert!(allowed.contains(&(b'f' as usize)));
        assert!(!allowed.contains(&(b'_' as usize)));
    }

    #[test]
    fn end_of_generation_is_admissible_only_once_the_call_is_closed() {
        let mut constraint = ToolConstraint::new(grammar());
        constraint.observe_bytes(&bytes(
            "<tool_call>\n<function=read>\n<parameter=filePath>\na\n</parameter>\n</function>\n",
        ));
        assert!(!allowed(&mut constraint).contains(&132));
        constraint.observe_bytes(&bytes("</tool_call>"));
        assert!(allowed(&mut constraint).contains(&132));
    }

    #[test]
    fn a_free_text_value_admits_everything_but_end_of_generation() {
        let mut constraint = ToolConstraint::new(grammar());
        constraint.observe_bytes(&bytes(
            "<tool_call>\n<function=read>\n<parameter=filePath>\n",
        ));
        let allowed = allowed(&mut constraint);
        assert!(allowed.contains(&(b'E' as usize)));
        assert!(allowed.contains(&(b'<' as usize)));
        assert!(allowed.contains(&130), "a multi-byte piece too");
        assert!(!allowed.contains(&132), "but not the end of the turn");
    }

    #[test]
    fn a_constraint_that_is_forced_off_its_grammar_stands_down() {
        let mut constraint = ToolConstraint::new(grammar());
        constraint.observe_bytes(&bytes("<tool_call>\n<function=read>\n<parameter=file"));
        assert!(constraint.is_masking());
        // A byte the mask excluded, emitted anyway.
        constraint.observe_bytes(&bytes("_"));
        assert!(!constraint.is_masking(), "no deadlock, just no constraint");
    }
}
