//! The one generation path all three dialects go through.
//!
//! Everything above this file is a wire format. This is where a tokenized
//! prompt becomes an admitted request, and where the scheduler's token stream
//! becomes text: incremental detokenization, stop sequences, and the split
//! between the model's reasoning span and its answer.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use axum::http::StatusCode;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use xabe_cache::radix::{BlockHash, ROOT_HASH, hash_block};
use xabe_engine::Engine;
use xabe_sched::request::{NewRequest, RequestId};

use super::AppState;
use super::error::{ApiError, Dialect};

/// What the scheduler thread sends to a waiting client.
pub(crate) enum ClientEvent {
    Token(i32),
    Done(EngineFinish),
    Error(String),
}

/// Why the engine stopped generating. The engine knows about two reasons; a
/// stop sequence is recognized here, above it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EngineFinish {
    Eos,
    Length,
}

/// Why generation ended, as reported to the client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Finish {
    /// The model emitted its end-of-turn token.
    EndOfTurn,
    /// `max_tokens` was reached first.
    Length,
    /// One of the caller's stop sequences appeared in the output.
    StopSequence(String),
}

/// A piece of decoded output, routed to the span it belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Chunk {
    Reasoning(String),
    Text(String),
}

/// A prompt that has already been templated and tokenized.
pub(crate) struct GenerationSpec {
    pub(crate) prompt: Vec<u32>,
    pub(crate) max_tokens: u32,
    pub(crate) stop: Vec<String>,
    /// Whether the prompt leaves the model inside an open `<think>` span, so
    /// output up to `</think>` is reasoning rather than answer.
    pub(crate) thinking: bool,
    /// Whether to drop the whitespace that opens the reasoning span and the
    /// answer. A chat reply should not begin with the newlines that separate
    /// it from the markup; a raw completion should be returned untouched.
    pub(crate) trim_spans: bool,
}

/// Chain the prompt's block hashes so the shared prefix tree can match them.
fn prompt_hashes(tokens: &[u32], block_size: usize) -> Vec<BlockHash> {
    let mut parent = ROOT_HASH;
    tokens
        .chunks(block_size)
        .map(|block| {
            parent = hash_block(parent, block);
            parent
        })
        .collect()
}

/// Cancels its request unless generation reached a terminal state first.
///
/// This is what makes a dropped HTTP connection release engine capacity: the
/// stream body owns the [`Generation`], which owns this.
struct RequestGuard {
    id: RequestId,
    engine: Arc<Mutex<Engine>>,
    armed: bool,
}

impl RequestGuard {
    fn new(id: RequestId, engine: Arc<Mutex<Engine>>) -> Self {
        Self {
            id,
            engine,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }

    /// Release the request now rather than at drop, for a stop sequence that
    /// ends the response while the engine would happily keep going.
    fn cancel_now(&mut self) {
        if self.armed {
            self.engine.lock().expect("engine poisoned").cancel(self.id);
            self.armed = false;
        }
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        if self.armed {
            self.engine.lock().expect("engine poisoned").cancel(self.id);
        }
    }
}

/// Turns a token stream into a text stream, one token at a time.
///
/// A byte-level BPE token is not a string: a multi-byte character can span
/// two tokens, and decoding each token alone yields replacement characters at
/// the seam. So each step decodes a short window ending at the new token and
/// emits only what that window added, which is the standard incremental
/// scheme. The window bounds the work, so the cost per token does not grow
/// with the length of the response.
struct Detokenizer {
    tokenizer: Arc<Tokenizer>,
    tokens: Vec<u32>,
    prefix_offset: usize,
    read_offset: usize,
}

impl Detokenizer {
    fn new(tokenizer: Arc<Tokenizer>, capacity: usize) -> Self {
        Self {
            tokenizer,
            tokens: Vec::with_capacity(capacity),
            prefix_offset: 0,
            read_offset: 0,
        }
    }

    fn decode(&self, range: std::ops::Range<usize>) -> String {
        self.tokenizer
            .decode(&self.tokens[range], false)
            .unwrap_or_default()
    }

    fn push(&mut self, token: u32) -> String {
        self.tokens.push(token);
        let prefix = self.decode(self.prefix_offset..self.read_offset);
        let whole = self.decode(self.prefix_offset..self.tokens.len());
        // A trailing replacement character means the window ends mid-character
        // and the next token will complete it; hold everything back until it
        // does.
        if whole.len() <= prefix.len() || whole.ends_with('\u{fffd}') {
            return String::new();
        }
        let delta = whole
            .strip_prefix(prefix.as_str())
            .map_or(whole.clone(), str::to_owned);
        self.prefix_offset = self.read_offset;
        self.read_offset = self.tokens.len();
        delta
    }
}

/// The longest suffix of `text` that is a proper prefix of some stop
/// sequence, and so cannot be emitted yet without risking a stop sequence
/// being split across two chunks.
fn held_back_len(text: &str, stop: &[String]) -> usize {
    let mut held = 0;
    for sequence in stop {
        for (end, _) in sequence.char_indices().skip(1) {
            if text.len() >= end && text.is_char_boundary(text.len() - end) {
                let candidate = &text[text.len() - end..];
                if candidate == &sequence[..end] {
                    held = held.max(end);
                }
            }
        }
    }
    held
}

/// Where the earliest stop sequence begins in `text`, and which one it was.
fn earliest_stop<'a>(text: &str, stop: &'a [String]) -> Option<(usize, &'a str)> {
    stop.iter()
        .filter_map(|sequence| {
            text.find(sequence.as_str())
                .map(|at| (at, sequence.as_str()))
        })
        .min_by_key(|&(at, _)| at)
}

/// One in-flight response.
pub(crate) struct Generation {
    id: RequestId,
    guard: RequestGuard,
    receiver: mpsc::UnboundedReceiver<ClientEvent>,
    detokenizer: Detokenizer,
    dialect: Dialect,
    stop: Vec<String>,
    /// Text decoded but not yet emitted, because it might turn out to be the
    /// beginning of a stop sequence.
    pending: String,
    /// The token that closes the model's reasoning span, if the prompt opened
    /// one.
    think_close: Option<i32>,
    reasoning_open: bool,
    trim_spans: bool,
    /// Set at the start of each span; cleared once that span has emitted
    /// something other than whitespace.
    trim_span_start: bool,
    prompt_tokens: usize,
    completion_tokens: usize,
    finish: Option<Finish>,
    drained: bool,
}

impl Generation {
    /// Admit a request and take ownership of its client channel.
    ///
    /// Fails before any response has been written, which is what lets a
    /// streaming handler report admission failures with a real status code
    /// rather than an event.
    pub(crate) fn start(
        state: &AppState,
        spec: GenerationSpec,
        dialect: Dialect,
    ) -> Result<Self, ApiError> {
        if spec.max_tokens == 0 {
            return Err(ApiError::bad_request(
                dialect,
                "the output token limit must be positive",
            ));
        }
        if spec.prompt.is_empty() {
            return Err(ApiError::bad_request(
                dialect,
                "the prompt must not be empty",
            ));
        }
        let prompt_tokens = u32::try_from(spec.prompt.len()).map_err(|_| {
            ApiError::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                dialect,
                "the tokenized prompt exceeds the request length representation",
            )
        })?;
        let tokens = spec.prompt.iter().map(|&token| token as i32).collect();
        let hashes = prompt_hashes(&spec.prompt, state.block_size);

        let id = RequestId(state.next_id.fetch_add(1, Ordering::Relaxed));
        let (sender, receiver) = mpsc::unbounded_channel();
        state
            .clients
            .lock()
            .expect("client map poisoned")
            .insert(id, sender);
        let placement = state.engine.lock().expect("engine poisoned").place_tokens(
            NewRequest {
                id,
                prompt_tokens,
                max_output_tokens: spec.max_tokens,
            },
            tokens,
            &hashes,
        );
        if let Err(failure) = placement {
            state
                .clients
                .lock()
                .expect("client map poisoned")
                .remove(&id);
            return Err(ApiError::unavailable(dialect, failure.to_string()));
        }

        Ok(Self {
            id,
            guard: RequestGuard::new(id, Arc::clone(&state.engine)),
            receiver,
            detokenizer: Detokenizer::new(
                Arc::clone(&state.tokenizer),
                spec.max_tokens.min(4096) as usize,
            ),
            dialect,
            stop: spec.stop,
            pending: String::new(),
            think_close: state.think_close_token,
            reasoning_open: spec.thinking && state.think_close_token.is_some(),
            trim_spans: spec.trim_spans,
            trim_span_start: spec.trim_spans,
            prompt_tokens: spec.prompt.len(),
            completion_tokens: 0,
            finish: None,
            drained: false,
        })
    }

    pub(crate) fn request_id(&self) -> u64 {
        self.id.0
    }

    pub(crate) fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    pub(crate) fn completion_tokens(&self) -> usize {
        self.completion_tokens
    }

    /// How generation ended. Only meaningful once [`Self::next`] has returned
    /// `None`; before that it reports `Length`, the reason a response that was
    /// cut short here would carry.
    pub(crate) fn finish(&self) -> Finish {
        self.finish.clone().unwrap_or(Finish::Length)
    }

    /// Route decoded text to the span it belongs to, dropping the whitespace
    /// that opens the span.
    ///
    /// `</think>` is followed by a blank line in this model's markup, so
    /// without this every answer would begin with two newlines the caller did
    /// not ask for. A raw completion opts out: there, the model's output is
    /// the response, verbatim.
    fn chunk(&mut self, text: String) -> Option<Chunk> {
        let text = if self.trim_span_start {
            text.trim_start().to_owned()
        } else {
            text
        };
        if text.is_empty() {
            return None;
        }
        self.trim_span_start = false;
        Some(if self.reasoning_open {
            Chunk::Reasoning(text)
        } else {
            Chunk::Text(text)
        })
    }

    /// The next piece of output, or `None` once the response is complete.
    pub(crate) async fn next(&mut self) -> Result<Option<Chunk>, ApiError> {
        if self.drained {
            return Ok(None);
        }
        loop {
            let event = self.receiver.recv().await;
            let token = match event {
                Some(ClientEvent::Token(token)) => token,
                Some(ClientEvent::Done(reason)) => {
                    self.guard.disarm();
                    self.drained = true;
                    self.finish = Some(match reason {
                        EngineFinish::Eos => Finish::EndOfTurn,
                        EngineFinish::Length => Finish::Length,
                    });
                    let tail = std::mem::take(&mut self.pending);
                    return Ok(self.chunk(tail));
                }
                Some(ClientEvent::Error(message)) => {
                    self.drained = true;
                    return Err(ApiError::internal(self.dialect, message));
                }
                None => {
                    self.drained = true;
                    return Err(ApiError::internal(
                        self.dialect,
                        "the scheduler closed the request without a completion status",
                    ));
                }
            };
            self.completion_tokens += 1;

            // The reasoning span closes on a token, not on a substring: the
            // model emits `</think>` as the single vocabulary entry it was
            // trained on, so there is nothing to scan for and no chance of the
            // marker being split across a chunk boundary.
            if self.reasoning_open && Some(token) == self.think_close {
                let tail = std::mem::take(&mut self.pending);
                let flushed = self.chunk(tail);
                self.reasoning_open = false;
                self.trim_span_start = self.trim_spans;
                if flushed.is_some() {
                    return Ok(flushed);
                }
                continue;
            }

            // The engine carries token ids as `i32`; every one of them is a
            // vocabulary index, so the cast is exact.
            let delta = self.detokenizer.push(token as u32);
            if delta.is_empty() {
                continue;
            }
            self.pending.push_str(&delta);

            if let Some((at, sequence)) = earliest_stop(&self.pending, &self.stop) {
                let sequence = sequence.to_owned();
                self.pending.truncate(at);
                let text = std::mem::take(&mut self.pending);
                self.guard.cancel_now();
                self.drained = true;
                self.finish = Some(Finish::StopSequence(sequence));
                return Ok(self.chunk(text));
            }

            let held = held_back_len(&self.pending, &self.stop);
            let emit = self.pending.len() - held;
            if emit > 0 {
                let text: String = self.pending.drain(..emit).collect();
                if let Some(chunk) = self.chunk(text) {
                    return Ok(Some(chunk));
                }
            }
        }
    }

    /// Run to completion, accumulating the reasoning span and the answer.
    pub(crate) async fn collect(&mut self) -> Result<(String, String), ApiError> {
        let (mut reasoning, mut text) = (String::new(), String::new());
        while let Some(chunk) = self.next().await? {
            match chunk {
                Chunk::Reasoning(part) => reasoning.push_str(&part),
                Chunk::Text(part) => text.push_str(&part),
            }
        }
        Ok((reasoning, text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashes_are_chained_at_natural_attention_blocks() {
        let tokens = (0..10).collect::<Vec<_>>();
        let hashes = prompt_hashes(&tokens, 4);
        assert_eq!(hashes.len(), 3);
        let first = hash_block(ROOT_HASH, &tokens[..4]);
        assert_eq!(hashes[0], first);
        assert_eq!(hashes[1], hash_block(first, &tokens[4..8]));
        assert_eq!(hashes[2], hash_block(hashes[1], &tokens[8..]));
    }

    #[test]
    fn a_partial_stop_sequence_is_held_back() {
        let stop = vec!["<|end|>".to_owned()];
        assert_eq!(held_back_len("hello <|en", &stop), 4);
        assert_eq!(held_back_len("hello", &stop), 0);
        // A complete match is not "held back"; `earliest_stop` claims it.
        assert_eq!(
            earliest_stop("hello <|end|> more", &stop),
            Some((6, "<|end|>"))
        );
    }

    #[test]
    fn holding_back_never_splits_a_character() {
        // The held-back suffix is measured in bytes, so a multi-byte character
        // adjacent to a stop-sequence prefix must not be sliced through.
        let stop = vec!["世界".to_owned()];
        assert_eq!(held_back_len("你好世", &stop), 3);
        assert_eq!(held_back_len("你好", &stop), 0);
    }

    #[test]
    fn the_earliest_of_several_stop_sequences_wins() {
        let stop = vec!["END".to_owned(), "STOP".to_owned()];
        assert_eq!(earliest_stop("a STOP b END", &stop), Some((2, "STOP")));
        assert_eq!(earliest_stop("nothing here", &stop), None);
    }
}
