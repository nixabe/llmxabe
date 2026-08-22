//! `ngram-mod` self-speculative drafting, ported from llama.cpp
//! `common/ngram-mod.{h,cpp}` (the `common_ngram_mod` hasher) and
//! `common/speculative.cpp` (`common_speculative_impl_ngram_mod`, which owns
//! the drafting and reset policy; upstream PR ggml-org/llama.cpp#19164).
//!
//! Unlike the other n-gram drafters, the hash table is **shared by every
//! sequence a worker serves**, as upstream shares one `common_ngram_mod`
//! across all of a context's sequences: what one request taught the table,
//! a concurrent request drafts from. The table is host state mutated only
//! on the single runtime thread; [`crate::spec::SelfSpecFactory`] hands each
//! sequence an `Rc<RefCell>` handle. Collisions are never verified against
//! history — a colliding entry drafts a wrong-context token, which the
//! target model's verify pass then rejects; the occupancy and
//! low-acceptance resets below are upstream's defense against the table
//! souring that way.
//!
//! The same `(tokens, sampled)`-to-owned-history index mapping as
//! [`crate::ngram_map`] applies: upstream's `cur_len` is `history.len() - 1`
//! here, and the lookup n-gram is the history's `n_match`-token tail.

use std::cell::RefCell;
use std::rc::Rc;

use crate::ngram::NgramConfigError;

/// Table slots, matching upstream's fixed `4*1024*1024` (16 MiB of i32).
pub const NGRAM_MOD_TABLE_ENTRIES: usize = 4 * 1024 * 1024;

/// The empty-slot marker (upstream `common_ngram_mod::EMPTY`).
pub const NGRAM_MOD_EMPTY: i32 = -1;

/// Reset the shared table when more than this fraction of it is occupied
/// at a sequence's start (upstream `f_thold` in the `begin` handler).
const OCCUPANCY_RESET_FRACTION: f64 = 0.25;

/// An accept round with less than this fraction of the draft accepted
/// counts toward the low-acceptance streak.
const LOW_ACCEPTANCE_FRACTION: f64 = 0.25;

/// Consecutive low-acceptance rounds that trigger a table reset.
const LOW_ACCEPTANCE_STREAK: u32 = 5;

/// Direct-mapped map from an `n`-gram's hash to the token that followed it,
/// newest writer wins. Upstream `common_ngram_mod`.
#[derive(Debug, Clone)]
pub struct NgramModTable {
    n: usize,
    used: usize,
    entries: Vec<i32>,
}

impl NgramModTable {
    pub fn new(n: usize, size: usize) -> Self {
        Self {
            n,
            used: 0,
            entries: vec![NGRAM_MOD_EMPTY; size],
        }
    }

    /// Upstream's LCG: `res = res * 6364136223846793005 + token` in 64 bits,
    /// reduced modulo the table size. Tokens sign-extend, matching C's
    /// `int` → `size_t` conversion.
    fn idx(&self, tokens: &[i32]) -> usize {
        let mut res = 0u64;
        for &token in &tokens[..self.n] {
            res = res
                .wrapping_mul(6364136223846793005)
                .wrapping_add(token as i64 as u64);
        }
        (res % self.entries.len() as u64) as usize
    }

    /// `tokens` holds the `n`-gram key followed by its continuation token.
    pub fn add(&mut self, tokens: &[i32]) {
        let i = self.idx(tokens);
        if self.entries[i] == NGRAM_MOD_EMPTY {
            self.used += 1;
        }
        self.entries[i] = tokens[self.n];
    }

    /// The continuation recorded for this `n`-gram, or [`NGRAM_MOD_EMPTY`].
    pub fn get(&self, tokens: &[i32]) -> i32 {
        self.entries[self.idx(tokens)]
    }

    pub fn reset(&mut self) {
        self.entries.fill(NGRAM_MOD_EMPTY);
        self.used = 0;
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn used(&self) -> usize {
        self.used
    }

    pub fn size(&self) -> usize {
        self.entries.len()
    }
}

/// Bounds for `ngram-mod`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramModConfig {
    /// Lookup n-gram length (llama.cpp `--spec-ngram-mod-n-match`).
    pub n_match: usize,
    /// Drop any draft shorter than this (`--spec-ngram-mod-n-min`).
    pub n_min: usize,
    /// Longest chain drafted per step (`--spec-ngram-mod-n-max`).
    pub n_max: usize,
    /// Tokens of history the speculator pre-allocates for.
    pub history_capacity: usize,
}

impl NgramModConfig {
    /// `n_min` may be zero and upstream does not require `n_min <= n_max`;
    /// a draft is dropped when the chain breaks before `n_min` tokens,
    /// whatever `n_max` allows.
    pub fn new(
        n_match: usize,
        n_min: usize,
        n_max: usize,
        history_capacity: usize,
    ) -> Result<Self, NgramConfigError> {
        if n_match == 0 {
            return Err(NgramConfigError::InvalidNgramRange);
        }
        if n_max == 0 {
            return Err(NgramConfigError::ZeroDraftTokens);
        }
        if history_capacity <= n_match {
            return Err(NgramConfigError::HistoryTooShort);
        }
        Ok(Self {
            n_match,
            n_min,
            n_max,
            history_capacity,
        })
    }
}

/// Per-sequence half of `ngram-mod`: the owned history and upstream's
/// `seq_info` (indexing watermark, last draft length, low-acceptance
/// streak), drafting against the worker-shared [`NgramModTable`].
#[derive(Debug, Clone)]
pub struct NgramModSpeculator {
    config: NgramModConfig,
    table: Rc<RefCell<NgramModTable>>,
    history: Vec<i32>,
    generation_started: bool,
    /// The last history position whose n-gram was added to the table.
    i_last: usize,
    /// Length of the last draft, for acceptance-fraction bookkeeping.
    n_draft_last: usize,
    /// Consecutive accept rounds below [`LOW_ACCEPTANCE_FRACTION`].
    n_low: u32,
    /// Reused chain buffer (`n_match + n_max`), so drafting allocates
    /// nothing per step.
    scratch: Vec<i32>,
}

impl NgramModSpeculator {
    pub fn new(config: NgramModConfig, table: Rc<RefCell<NgramModTable>>) -> Self {
        debug_assert_eq!(table.borrow().n(), config.n_match);
        Self {
            config,
            table,
            history: Vec::with_capacity(config.history_capacity),
            generation_started: false,
            i_last: 0,
            n_draft_last: 0,
            n_low: 0,
            scratch: Vec::with_capacity(config.n_match + config.n_max),
        }
    }

    pub fn config(&self) -> NgramModConfig {
        self.config
    }

    /// Add one generated token. The first call marks the prompt/generation
    /// boundary and runs upstream's `begin`: index every prompt n-gram into
    /// the shared table, then reset the table if the whole thing has grown
    /// past [`OCCUPANCY_RESET_FRACTION`] — a saturating table is mostly
    /// collisions, and its drafts stop earning their verify slots.
    pub fn observe(&mut self, token: i32) {
        if !self.generation_started {
            self.generation_started = true;
            self.begin();
        }
        push_bounded(&mut self.history, token);
    }

    /// Add prompt tokens (chunked prefill delivers them in pieces).
    pub fn observe_all(&mut self, tokens: &[i32]) {
        for &token in tokens {
            push_bounded(&mut self.history, token);
        }
    }

    fn begin(&mut self) {
        self.i_last = 0;
        self.n_draft_last = 0;
        let n = self.config.n_match;
        if self.history.len() < n {
            return;
        }
        let mut table = self.table.borrow_mut();
        for i in 0..self.history.len() - n {
            table.add(&self.history[i..=i + n]);
        }
        // Upstream keeps the watermark even when the reset below empties
        // the table: this prompt's n-grams are not re-added.
        self.i_last = self.history.len() - n;
        let occupancy = table.used() as f64 / table.size() as f64;
        if occupancy > OCCUPANCY_RESET_FRACTION {
            table.reset();
        }
    }

    /// Upstream's `draft_one`: index the n-grams observed since the last
    /// draft (in chunks of at least 32), then chain table lookups from the
    /// history's tail until the chain breaks or `n_max` is reached. Returns
    /// the number appended to `out`, truncated to the capacity the caller
    /// reserved.
    pub fn propose_into(&mut self, out: &mut Vec<i32>) -> usize {
        self.n_draft_last = 0;
        let room = out
            .capacity()
            .saturating_sub(out.len())
            .min(self.config.n_max);
        if room == 0 || self.history.is_empty() {
            return 0;
        }
        let n = self.config.n_match;
        let cur_len = self.history.len() - 1;
        if cur_len < n {
            return 0;
        }

        let mut table = self.table.borrow_mut();
        if self.i_last + 32 < cur_len {
            for i in self.i_last..cur_len - n {
                table.add(&self.history[i..=i + n]);
            }
            self.i_last = cur_len - n;
        }

        // The chain seed is the history's n-token tail (upstream: the last
        // `n - 1` committed tokens plus the sampled one).
        self.scratch.clear();
        self.scratch
            .extend_from_slice(&self.history[self.history.len() - n..]);
        for i in 0..self.config.n_max {
            let token = table.get(&self.scratch[i..i + n]);
            if token == NGRAM_MOD_EMPTY {
                if i < self.config.n_min {
                    return 0;
                }
                break;
            }
            self.scratch.push(token);
        }

        let draft = &self.scratch[n..];
        let count = draft.len().min(room);
        out.extend_from_slice(&draft[..count]);
        self.n_draft_last = count;
        count
    }

    /// Upstream's accept handler: five consecutive rounds with under a
    /// quarter of the draft accepted reset the shared table — it has
    /// soured (collisions or a topic shift), and a fresh table re-learns
    /// from the histories still being observed.
    pub fn accept(&mut self, n_accepted: usize) {
        if self.n_draft_last == 0 {
            return;
        }
        let f_acc = n_accepted as f64 / self.n_draft_last as f64;
        if f_acc < LOW_ACCEPTANCE_FRACTION {
            self.n_low += 1;
            if self.n_low >= LOW_ACCEPTANCE_STREAK {
                self.table.borrow_mut().reset();
                self.n_low = 0;
                self.i_last = 0;
            }
        } else {
            self.n_low = 0;
        }
    }
}

/// Append within the pre-reserved capacity; see `ngram_map::push_bounded`.
fn push_bounded(history: &mut Vec<i32>, token: i32) {
    debug_assert!(
        history.len() < history.capacity(),
        "history outgrew the capacity it was sized for"
    );
    if history.len() < history.capacity() {
        history.push(token);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn speculator(n_match: usize, n_min: usize, n_max: usize) -> NgramModSpeculator {
        let table = Rc::new(RefCell::new(NgramModTable::new(n_match, 1 << 16)));
        NgramModSpeculator::new(
            NgramModConfig::new(n_match, n_min, n_max, 512).unwrap(),
            table,
        )
    }

    fn propose(spec: &mut NgramModSpeculator, cap: usize) -> Vec<i32> {
        let mut out = Vec::with_capacity(cap);
        spec.propose_into(&mut out);
        out
    }

    #[test]
    fn table_records_the_newest_continuation() {
        let mut table = NgramModTable::new(2, 64);
        table.add(&[1, 2, 3]);
        assert_eq!(table.get(&[1, 2]), 3);
        assert_eq!(table.used(), 1);
        table.add(&[1, 2, 7]);
        assert_eq!(table.get(&[1, 2]), 7);
        assert_eq!(table.used(), 1, "overwrites do not grow occupancy");
        table.reset();
        assert_eq!(table.get(&[1, 2]), NGRAM_MOD_EMPTY);
    }

    #[test]
    fn drafts_chain_through_the_table_up_to_n_max() {
        let mut spec = speculator(2, 0, 3);
        // Prompt teaches (1,2)->3, (2,3)->4, (3,4)->5; generation reaches
        // the (1, 2) tail again.
        spec.observe_all(&[1, 2, 3, 4, 5, 9, 1]);
        spec.observe(2);
        assert_eq!(propose(&mut spec, 3), [3, 4, 5]);
    }

    #[test]
    fn a_chain_shorter_than_n_min_is_dropped() {
        // The prompt keys only up to (2,3)->4 — (3,4) is never a key, so
        // the chain from the (1,2) tail breaks after two tokens.
        let mut spec = speculator(2, 3, 4);
        spec.observe_all(&[9, 8, 1, 2, 3, 4]);
        spec.observe(1);
        spec.observe(2);
        assert_eq!(propose(&mut spec, 4), Vec::<i32>::new());
        // The same chain passes once n_min allows it.
        let mut spec = speculator(2, 2, 4);
        spec.observe_all(&[9, 8, 1, 2, 3, 4]);
        spec.observe(1);
        spec.observe(2);
        assert_eq!(propose(&mut spec, 4), [3, 4]);
    }

    #[test]
    fn generated_tokens_are_indexed_in_chunks() {
        let mut spec = speculator(2, 0, 2);
        spec.observe_all(&[9, 9]);
        // Feed a repeating cycle as "generated" tokens; after the 32-token
        // chunk threshold the table has seen the cycle and drafts it.
        for round in 0..64 {
            spec.observe([1, 2, 3][round % 3]);
        }
        let draft = propose(&mut spec, 2);
        assert!(!draft.is_empty(), "cycle was indexed and should draft");
        // Whatever the chain drafted came from the observed cycle.
        for &token in &draft {
            assert!([1, 2, 3].contains(&token));
        }
    }

    #[test]
    fn five_low_acceptance_rounds_reset_the_shared_table() {
        let mut spec = speculator(2, 0, 4);
        spec.observe_all(&[1, 2, 3, 4, 5, 9, 1]);
        spec.observe(2);
        assert!(!propose(&mut spec, 4).is_empty());
        for _ in 0..4 {
            spec.accept(0);
            assert!(spec.table.borrow().used() > 0, "streak not reached yet");
            assert!(!propose(&mut spec, 4).is_empty());
        }
        spec.accept(0);
        assert_eq!(spec.table.borrow().used(), 0, "fifth low round resets");
        assert_eq!(spec.i_last, 0, "history is re-indexed from scratch");
    }

    #[test]
    fn a_good_round_clears_the_low_streak() {
        let mut spec = speculator(2, 0, 4);
        spec.observe_all(&[1, 2, 3, 4, 5, 9, 1]);
        spec.observe(2);
        for _ in 0..4 {
            assert!(!propose(&mut spec, 4).is_empty());
            spec.accept(0);
        }
        assert!(!propose(&mut spec, 4).is_empty());
        spec.accept(3); // 3 of the 4-token draft accepted — the streak resets
        assert_eq!(spec.n_low, 0);
        assert!(spec.table.borrow().used() > 0);
    }

    #[test]
    fn sequences_share_one_table_through_the_factory_handle() {
        let table = Rc::new(RefCell::new(NgramModTable::new(2, 1 << 16)));
        let config = NgramModConfig::new(2, 0, 3, 512).unwrap();
        let mut a = NgramModSpeculator::new(config, Rc::clone(&table));
        let mut b = NgramModSpeculator::new(config, Rc::clone(&table));
        // Sequence A's prompt teaches the chain; B has never seen those
        // tokens in its own history but drafts from the shared table.
        a.observe_all(&[1, 2, 3, 4, 5]);
        a.observe(9);
        b.observe_all(&[7, 7, 7, 1]);
        b.observe(2);
        assert_eq!(propose(&mut b, 3), [3, 4, 5]);
    }

    #[test]
    fn an_over_occupied_table_is_reset_at_generation_start() {
        let table = Rc::new(RefCell::new(NgramModTable::new(1, 8)));
        let config = NgramModConfig::new(1, 0, 2, 512).unwrap();
        let mut spec = NgramModSpeculator::new(config, Rc::clone(&table));
        // Seven distinct unigram keys into eight slots: > 25% occupied.
        spec.observe_all(&[1, 2, 3, 4, 5, 6, 7, 8]);
        spec.observe(9);
        assert_eq!(table.borrow().used(), 0, "begin reset the table");
        // And the watermark still advanced: the prompt is not re-indexed.
        assert_eq!(spec.i_last, 7);
    }

    #[test]
    fn config_validation_rejects_degenerate_shapes() {
        assert!(NgramModConfig::new(0, 0, 4, 64).is_err());
        assert!(NgramModConfig::new(2, 0, 0, 64).is_err());
        assert!(NgramModConfig::new(24, 48, 64, 24).is_err());
        // Upstream allows n_min > n_max and n_min = 0; so do we.
        assert!(NgramModConfig::new(24, 100, 64, 512).is_ok());
    }
}
