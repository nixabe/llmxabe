//! Per-sequence n-gram drafting for speculative decoding.
//!
//! Unlike MTP, n-gram drafting has no model pass and no shared mutable model
//! state.  Each decoding sequence owns one [`NgramSpeculator`], so a batch of
//! independent streams can draft in parallel and the drafts are then verified
//! by the same batched target-model pass.  This module owns only the draft and
//! acceptance policy; executing the target-model verification belongs to the
//! engine.

/// Bounds for prompt/history lookup and the number of tokens drafted at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramConfig {
    pub min_ngram: usize,
    pub max_ngram: usize,
    pub max_draft_tokens: usize,
    pub history_capacity: usize,
}

impl NgramConfig {
    pub fn new(
        min_ngram: usize,
        max_ngram: usize,
        max_draft_tokens: usize,
        history_capacity: usize,
    ) -> Result<Self, NgramConfigError> {
        if min_ngram == 0 || min_ngram > max_ngram {
            return Err(NgramConfigError::InvalidNgramRange);
        }
        if max_draft_tokens == 0 {
            return Err(NgramConfigError::ZeroDraftTokens);
        }
        if history_capacity <= max_ngram {
            return Err(NgramConfigError::HistoryTooShort);
        }
        Ok(Self {
            min_ngram,
            max_ngram,
            max_draft_tokens,
            history_capacity,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NgramConfigError {
    InvalidNgramRange,
    ZeroDraftTokens,
    HistoryTooShort,
}

impl core::fmt::Display for NgramConfigError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidNgramRange => write!(f, "n-gram range must satisfy 0 < min <= max"),
            Self::ZeroDraftTokens => write!(f, "an n-gram draft must allow at least one token"),
            Self::HistoryTooShort => write!(f, "history capacity must exceed max n-gram length"),
        }
    }
}

impl core::error::Error for NgramConfigError {}

/// A fixed-capacity token history and deterministic suffix matcher.
///
/// Storage is allocated once at construction. [`Self::observe`] and
/// [`Self::propose_into`] do not allocate; callers pre-allocate the output
/// buffer to `config.max_draft_tokens`, matching the engine's no-allocation
/// hot-path rule.
#[derive(Debug, Clone)]
pub struct NgramSpeculator {
    config: NgramConfig,
    history: Vec<i32>,
    start: usize,
    len: usize,
}

impl NgramSpeculator {
    pub fn new(config: NgramConfig) -> Self {
        Self {
            config,
            history: vec![0; config.history_capacity],
            start: 0,
            len: 0,
        }
    }

    pub fn config(&self) -> NgramConfig {
        self.config
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Add one committed token, evicting only the oldest token when full.
    pub fn observe(&mut self, token: i32) {
        if self.len < self.history.len() {
            let index = (self.start + self.len) % self.history.len();
            self.history[index] = token;
            self.len += 1;
        } else {
            self.history[self.start] = token;
            self.start = (self.start + 1) % self.history.len();
        }
    }

    pub fn observe_all(&mut self, tokens: &[i32]) {
        for &token in tokens {
            self.observe(token);
        }
    }

    /// Draft from the newest earlier occurrence of the longest suffix.
    ///
    /// Returns the number appended to `out`. If `out` lacks the capacity
    /// reserved by the caller, the draft is truncated instead of allocating.
    pub fn propose_into(&self, out: &mut Vec<i32>) -> usize {
        let room = out
            .capacity()
            .saturating_sub(out.len())
            .min(self.config.max_draft_tokens);
        if room == 0 {
            return 0;
        }

        let max_n = self.config.max_ngram.min(self.len);
        for n in (self.config.min_ngram..=max_n).rev() {
            // A match must have at least one following token to draft.
            if self.len <= n {
                continue;
            }
            for candidate in (0..=self.len - n - 1).rev() {
                let suffix = self.len - n;
                if (0..n).all(|i| self.at(candidate + i) == self.at(suffix + i)) {
                    let available = self.len - (candidate + n);
                    let count = room.min(available);
                    for i in 0..count {
                        out.push(self.at(candidate + n + i));
                    }
                    return count;
                }
            }
        }
        0
    }

    fn at(&self, logical: usize) -> i32 {
        debug_assert!(logical < self.len);
        self.history[(self.start + logical) % self.history.len()]
    }
}

/// Result of comparing a draft with target-model tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Verification {
    /// Consecutive draft tokens accepted before the first mismatch.
    pub accepted: usize,
    /// Target token at the mismatch, or the bonus token after full acceptance.
    pub next_token: i32,
}

/// Apply greedy speculative verification to one sequence.
///
/// `target` must contain at least `draft.len() + 1` tokens: one target result
/// for every draft position and the bonus token used when the whole draft is
/// accepted. Sampling-aware verification can be added above this primitive;
/// the engine currently uses greedy argmax sampling.
pub fn verify_greedy(draft: &[i32], target: &[i32]) -> Option<Verification> {
    if target.len() < draft.len() + 1 {
        return None;
    }
    let accepted = draft
        .iter()
        .zip(target)
        .take_while(|(drafted, actual)| drafted == actual)
        .count();
    Some(Verification {
        accepted,
        next_token: target[accepted],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(capacity: usize) -> NgramConfig {
        NgramConfig::new(2, 4, 3, capacity).unwrap()
    }

    #[test]
    fn longest_suffix_uses_the_newest_matching_continuation() {
        let mut ngram = NgramSpeculator::new(config(32));
        ngram.observe_all(&[1, 2, 7, 1, 2, 8, 1, 2]);
        let mut draft = Vec::with_capacity(3);
        assert_eq!(ngram.propose_into(&mut draft), 3);
        assert_eq!(draft, [8, 1, 2]);
    }

    #[test]
    fn circular_history_keeps_matching_after_eviction() {
        let mut ngram = NgramSpeculator::new(config(6));
        ngram.observe_all(&[9, 9, 1, 2, 3, 1, 2]);
        let mut draft = Vec::with_capacity(3);
        ngram.propose_into(&mut draft);
        assert_eq!(draft, [3, 1, 2]);
        assert_eq!(ngram.len(), 6);
    }

    #[test]
    fn proposal_never_grows_the_callers_buffer() {
        let mut ngram = NgramSpeculator::new(config(16));
        ngram.observe_all(&[1, 2, 3, 4, 1, 2]);
        let mut draft = Vec::with_capacity(1);
        let capacity = draft.capacity();
        assert_eq!(ngram.propose_into(&mut draft), 1);
        assert_eq!(draft.capacity(), capacity);
        assert_eq!(draft, [3]);
    }

    #[test]
    fn concurrent_sequences_have_independent_draft_histories() {
        let mut a = NgramSpeculator::new(config(16));
        let mut b = NgramSpeculator::new(config(16));
        a.observe_all(&[1, 2, 3, 1, 2]);
        b.observe_all(&[1, 2, 9, 1, 2]);
        let (mut da, mut db) = (Vec::with_capacity(3), Vec::with_capacity(3));
        a.propose_into(&mut da);
        b.propose_into(&mut db);
        assert_eq!(da, [3, 1, 2]);
        assert_eq!(db, [9, 1, 2]);
    }

    #[test]
    fn greedy_verification_returns_mismatch_or_bonus_token() {
        assert_eq!(
            verify_greedy(&[3, 4, 5], &[3, 4, 8, 9]),
            Some(Verification {
                accepted: 2,
                next_token: 8
            })
        );
        assert_eq!(
            verify_greedy(&[3, 4, 5], &[3, 4, 5, 9]),
            Some(Verification {
                accepted: 3,
                next_token: 9
            })
        );
        assert_eq!(verify_greedy(&[3], &[3]), None);
    }
}
