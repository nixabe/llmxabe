//! Per-sequence n-gram drafting for speculative decoding.
//!
//! Unlike MTP, n-gram drafting has no model pass and no shared mutable model
//! state.  Each decoding sequence owns one [`NgramSpeculator`], so a batch of
//! independent streams can draft in parallel and the drafts are then verified
//! by the same batched target-model pass.  This module owns only the draft and
//! acceptance policy; executing the target-model verification belongs to the
//! engine.
//!
//! # Matching is a hash lookup, not a scan
//!
//! The worker sizes `history_capacity` to the whole context pool, so a
//! backward suffix scan is O(history) per proposal per sequence — host work
//! that sits between GPU launches and grows with context length.  Instead,
//! [`NgramSpeculator::observe`] indexes every n-gram *that has a
//! continuation* into a fixed-capacity direct-mapped table per n (newest
//! occurrence wins), and [`NgramSpeculator::propose_into`] is one lookup per
//! n.  A candidate from the table is verified token-by-token against the
//! history before anything is drafted, so a hash collision can only cost a
//! missed draft, never a wrong one — and a wrong one would anyway be caught
//! by target-model verification, which is the actual acceptance gate.
//!
//! Entries are keyed by the n-gram's *absolute* stream offset (tokens
//! observed since construction), so a stale entry whose window has been
//! evicted from the ring fails a cheap range check instead of matching
//! garbage.

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

/// Cap on each per-n index table, in entries. 2^20 entries is 8 MiB per
/// table; beyond that, extra capacity buys collision reduction on histories
/// long enough that the drafts themselves have stopped mattering.
const MAX_TABLE_ENTRIES: usize = 1 << 20;

/// FxHash-style mix of one token into a running hash. Multiplicative mixing
/// is enough here: a collision is verified away before use.
#[inline]
fn mix(hash: u64, token: i32) -> u64 {
    (hash.rotate_left(5) ^ (token as u32 as u64)).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95)
}

/// A fixed-capacity token history with a per-n hash index over its n-grams.
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
    /// Tokens observed since construction; logical index `i` corresponds to
    /// absolute offset `total - len + i`.
    total: u64,
    /// One direct-mapped table per n in `min_ngram..=max_ngram`, all the same
    /// power-of-two size. `tables[j][slot]` holds the absolute *end* offset
    /// of the newest n-gram (n = min + j) hashing to `slot` that has at
    /// least one following token; `0` is empty (a real end offset is
    /// always >= min_ngram >= 1).
    tables: Vec<Vec<u64>>,
    table_mask: u64,
}

impl NgramSpeculator {
    pub fn new(config: NgramConfig) -> Self {
        let entries = (config.history_capacity * 2)
            .next_power_of_two()
            .min(MAX_TABLE_ENTRIES);
        let tables = (config.min_ngram..=config.max_ngram)
            .map(|_| vec![0u64; entries])
            .collect();
        Self {
            config,
            history: vec![0; config.history_capacity],
            start: 0,
            len: 0,
            total: 0,
            tables,
            table_mask: (entries - 1) as u64,
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

    /// Add one committed token, evicting only the oldest token when full,
    /// and index the n-grams that just gained a continuation.
    pub fn observe(&mut self, token: i32) {
        if self.len < self.history.len() {
            let index = (self.start + self.len) % self.history.len();
            self.history[index] = token;
            self.len += 1;
        } else {
            self.history[self.start] = token;
            self.start = (self.start + 1) % self.history.len();
        }
        self.total += 1;
        // The n-grams ending at the *previous* position now have `token` as
        // a continuation; the suffix ending at the new position has none yet
        // and is deliberately not indexed — every stored entry can draft at
        // least one token by construction.
        let end = self.len - 1; // logical end of the just-completed n-grams
        for (j, n) in (self.config.min_ngram..=self.config.max_ngram).enumerate() {
            if end < n {
                break;
            }
            let mut hash = 0u64;
            for i in (end - n)..end {
                hash = mix(hash, self.at(i));
            }
            let slot = (hash & self.table_mask) as usize;
            self.tables[j][slot] = self.total - 1;
        }
    }

    pub fn observe_all(&mut self, tokens: &[i32]) {
        for &token in tokens {
            self.observe(token);
        }
    }

    /// Draft the continuation of the newest indexed occurrence of the
    /// longest suffix, verified against the history before use.
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
            let j = n - self.config.min_ngram;
            let mut hash = 0u64;
            for i in (self.len - n)..self.len {
                hash = mix(hash, self.at(i));
            }
            let end_abs = self.tables[j][(hash & self.table_mask) as usize];
            if end_abs == 0 {
                continue;
            }
            let window_base = self.total - self.len as u64;
            // Stale if the n-gram has (partially) left the ring, or if it is
            // somehow not older than the current suffix.
            if end_abs >= self.total || end_abs < window_base + n as u64 {
                continue;
            }
            let end = (end_abs - window_base) as usize;
            let suffix = self.len - n;
            if !(0..n).all(|i| self.at(end - n + i) == self.at(suffix + i)) {
                continue; // hash collision — skip, never draft unverified
            }
            let available = self.len - end;
            let count = room.min(available);
            for i in 0..count {
                out.push(self.at(end + i));
            }
            return count;
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

    /// The pre-index behavior: newest earlier occurrence of the longest
    /// suffix, by direct backward scan. The hash index must agree whenever
    /// it drafts at all.
    fn scan_propose(history: &[i32], config: NgramConfig, room: usize) -> Vec<i32> {
        let len = history.len();
        let max_n = config.max_ngram.min(len);
        for n in (config.min_ngram..=max_n).rev() {
            if len <= n {
                continue;
            }
            for candidate in (0..=len - n - 1).rev() {
                let suffix = len - n;
                if (0..n).all(|i| history[candidate + i] == history[suffix + i]) {
                    let available = len - (candidate + n);
                    let count = room.min(available).min(config.max_draft_tokens);
                    return history[candidate + n..candidate + n + count].to_vec();
                }
            }
        }
        Vec::new()
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

    #[test]
    fn a_stale_entry_whose_window_was_evicted_never_drafts() {
        // Capacity 5 with min_ngram 2: [1,2,3] indexes (1,2)->3; then enough
        // unrelated tokens evict 1 and 2 from the ring while the table entry
        // survives. Re-observing the suffix (1,2) must not draft from the
        // evicted occurrence.
        let ngram_config = NgramConfig::new(2, 2, 3, 5).unwrap();
        let mut ngram = NgramSpeculator::new(ngram_config);
        ngram.observe_all(&[1, 2, 3, 7, 8, 9, 1, 2]);
        // History ring now holds [9, 1, 2] of the original occurrence's era —
        // the (1,2)->3 continuation is gone; only the fresh (1,2) at the tail
        // remains, and it has no continuation yet.
        let mut draft = Vec::with_capacity(3);
        // Whatever happens, it must not fabricate tokens: any draft must be a
        // verified continuation of a real in-ring occurrence.
        let drafted = ngram.propose_into(&mut draft);
        if drafted > 0 {
            // The only legal source would be an in-ring occurrence of (1,2)
            // older than the suffix; there is none besides the suffix itself.
            panic!("drafted {draft:?} from an evicted occurrence");
        }
    }

    #[test]
    fn every_draft_is_a_verified_continuation_of_the_suffix() {
        // Pseudo-random streams over a small alphabet: whenever the index
        // drafts, the draft must equal what the direct backward scan would
        // have produced from *some* real occurrence — specifically, the
        // drafted tokens must follow an in-history match of the suffix.
        let mut rng = 0x243f_6a88_85a3_08d3u64;
        for round in 0..64 {
            let ngram_config = NgramConfig::new(2, 4, 3, 64).unwrap();
            let mut ngram = NgramSpeculator::new(ngram_config);
            let mut stream_tokens = Vec::new();
            for _ in 0..(96 + round) {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let token = ((rng >> 33) % 5) as i32;
                stream_tokens.push(token);
                ngram.observe(token);

                let mut draft = Vec::with_capacity(3);
                if ngram.propose_into(&mut draft) == 0 {
                    continue;
                }
                // Reconstruct the retained window and check the draft is a
                // genuine continuation of some occurrence of the suffix.
                let window: Vec<i32> =
                    stream_tokens[stream_tokens.len().saturating_sub(64)..].to_vec();
                let mut legal = false;
                'outer: for n in ngram_config.min_ngram..=ngram_config.max_ngram.min(window.len()) {
                    let suffix = &window[window.len() - n..];
                    for end in n..window.len() {
                        if &window[end - n..end] == suffix && window[end..].starts_with(&draft) {
                            legal = true;
                            break 'outer;
                        }
                    }
                }
                assert!(legal, "draft {draft:?} has no supporting occurrence");
            }
        }
    }

    #[test]
    fn the_index_agrees_with_the_scan_on_collision_free_histories() {
        // On short histories over a tiny alphabet the direct-mapped tables
        // are far from full, so the index should reproduce the scan's answer
        // token for token (the scan is the documented policy: newest earlier
        // occurrence of the longest suffix).
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut agreements = 0usize;
        let mut proposals = 0usize;
        for _ in 0..32 {
            let ngram_config = config(64);
            let mut ngram = NgramSpeculator::new(ngram_config);
            let mut tokens = Vec::new();
            for _ in 0..80 {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let token = ((rng >> 33) % 4) as i32;
                tokens.push(token);
                ngram.observe(token);
            }
            let mut draft = Vec::with_capacity(3);
            ngram.propose_into(&mut draft);
            let expected =
                scan_propose(&tokens[tokens.len().saturating_sub(64)..], ngram_config, 3);
            if !expected.is_empty() {
                proposals += 1;
                if draft == expected {
                    agreements += 1;
                }
            }
        }
        // The index may miss a draft the scan finds (collision), but on this
        // scale it should agree almost always; a systematic disagreement is
        // a logic bug, not a collision.
        assert!(
            agreements * 10 >= proposals * 9,
            "index agreed on only {agreements}/{proposals} scan proposals"
        );
    }
}
