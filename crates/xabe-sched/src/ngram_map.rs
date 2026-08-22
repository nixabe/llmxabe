//! `ngram-simple`, `ngram-map-k` and `ngram-map-k4v` self-speculative
//! drafting, ported from llama.cpp `common/ngram-map.{h,cpp}`
//! (`common_ngram_simple_draft`, `common_ngram_map_draft`,
//! `common_ngram_map_accept`; upstream PR ggml-org/llama.cpp#18471).
//!
//! Like [`crate::ngram`], these are host-side drafting policies: they own no
//! model state, one speculator per decoding sequence, and every draft is
//! checked by the target model's batched verify pass, so a bad draft can
//! only cost speed, never tokens.
//!
//! # Index mapping against the upstream code
//!
//! llama.cpp passes `(tokens, sampled)` — the committed history plus the
//! token just sampled, kept separate. Here the speculator owns one
//! append-only `history` whose tail *is* the just-sampled token, so
//! upstream's `cur_len == tokens.size()` becomes `history.len() - 1` and
//! every `tokens[x]` with `x < cur_len` is `history[x]` unchanged. The
//! history is append-only (a serving sequence never loses tokens), which
//! keeps upstream's convention that index 0 doubles as "no match" and makes
//! `common_ngram_map_begin`'s shrink-cleanup branches dead code; only its
//! bookkeeping (`size_last_begin`, `idx_last_check`) is ported, marked at
//! the generation boundary by the first single-token [`observe`].
//!
//! [`observe`]: NgramMapSpeculator::observe

use crate::ngram::NgramConfigError;

/// Maximum number of m-gram values tracked per key n-gram
/// (llama.cpp `COMMON_NGRAM_MAX_VALUES`).
pub const NGRAM_MAP_MAX_VALUES: usize = 4;

/// Entries in the hash map from n-gram hash to n-gram history index
/// (llama.cpp `COMMON_NGRAM_HASH_MAP_SIZE`).
pub const NGRAM_MAP_HASH_MAP_SIZE: usize = 262_144;

/// Occurrence-counter saturation (llama.cpp `COMMON_NGRAM_MAX_VALUE_COUNT`).
const MAX_VALUE_COUNT: u32 = 16_380;

/// Key entries pre-allocated per sequence (~72 B each, so ~290 KiB): a
/// generation adds at most one key per decode step, and this covers a long
/// one without reallocating between GPU launches (AGENTS.md rule 6).
/// Upstream caps nothing, and neither does this — past the reserve the
/// vector simply grows.
const KEYS_RESERVED: usize = 4096;

/// Prime near `(sqrt(5) - 1)/2 * 2^32`, llama.cpp's `LCG_FACTOR`.
const LCG_FACTOR: u32 = 2_654_435_761;

/// llama.cpp `common_ngram_map_hash`: 32-bit LCG over the n-gram. Token ids
/// enter as their two's-complement bits, matching C's int → uint32 wrap.
fn map_hash(tokens: &[i32]) -> u32 {
    let mut hash = 0u32;
    for &token in tokens {
        hash = hash.wrapping_mul(LCG_FACTOR).wrapping_add(token as u32);
    }
    hash
}

// n-gram simple
//

/// Bounds for `ngram-simple`: fixed lookup length, fixed draft length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramSimpleConfig {
    /// Length of the lookup n-gram (llama.cpp `--spec-ngram-simple-size-n`).
    pub size_n: usize,
    /// Longest m-gram drafted from a match (`--spec-ngram-simple-size-m`).
    pub size_m: usize,
    /// Tokens of history the speculator pre-allocates for.
    pub history_capacity: usize,
}

impl NgramSimpleConfig {
    pub fn new(
        size_n: usize,
        size_m: usize,
        history_capacity: usize,
    ) -> Result<Self, NgramConfigError> {
        if size_n == 0 {
            return Err(NgramConfigError::InvalidNgramRange);
        }
        if size_m == 0 {
            return Err(NgramConfigError::ZeroDraftTokens);
        }
        // The draft needs a match plus its continuation plus the current
        // suffix to ever fire (`cur_len > n + m + 1` upstream).
        if history_capacity <= size_n + size_m + 1 {
            return Err(NgramConfigError::HistoryTooShort);
        }
        Ok(Self {
            size_n,
            size_m,
            history_capacity,
        })
    }
}

/// llama.cpp's `common_ngram_simple_draft` over an owned history: backward
/// scan for the newest earlier occurrence of the current n-token tail,
/// drafting the m-gram that followed it.
#[derive(Debug, Clone)]
pub struct NgramSimpleSpeculator {
    config: NgramSimpleConfig,
    history: Vec<i32>,
}

impl NgramSimpleSpeculator {
    pub fn new(config: NgramSimpleConfig) -> Self {
        Self {
            config,
            history: Vec::with_capacity(config.history_capacity),
        }
    }

    pub fn config(&self) -> NgramSimpleConfig {
        self.config
    }

    pub fn observe(&mut self, token: i32) {
        push_bounded(&mut self.history, token);
    }

    pub fn observe_all(&mut self, tokens: &[i32]) {
        for &token in tokens {
            push_bounded(&mut self.history, token);
        }
    }

    /// Draft the continuation of the newest earlier occurrence of the
    /// history's `size_n`-token tail. Returns the number appended to `out`;
    /// the draft is truncated to the capacity the caller reserved rather
    /// than allocating.
    pub fn propose_into(&self, out: &mut Vec<i32>) -> usize {
        let room = out
            .capacity()
            .saturating_sub(out.len())
            .min(self.config.size_m);
        if room == 0 || self.history.is_empty() {
            return 0;
        }
        let h = &self.history;
        let n = self.config.size_n;
        let m = self.config.size_m;
        // Upstream's `(tokens, sampled)` split: everything before the tail
        // token, plus the tail token.
        let cur_len = h.len() - 1;
        if cur_len <= n + m + 1 {
            return 0;
        }

        // The pattern is upstream's last `n - 1` committed tokens plus the
        // sampled one — exactly the history's n-token tail.
        let pattern = &h[h.len() - n..];
        let mut match_pos = 0;
        // Search backwards, skipping the current match; position 0 means
        // "no match", exactly as upstream.
        let mut j = cur_len - n - 1;
        while j > 0 {
            if h[j..j + n] == *pattern {
                match_pos = j;
                break;
            }
            j -= 1;
        }
        if match_pos == 0 {
            return 0;
        }

        let copy_max = m.min(cur_len - (match_pos + n));
        if copy_max < n {
            // Upstream drops any draft shorter than the lookup n-gram.
            return 0;
        }
        let count = copy_max.min(room);
        out.extend_from_slice(&h[match_pos + n..match_pos + n + count]);
        count
    }
}

// n-gram map (ngram-map-k when key-only, ngram-map-k4v with values)
//

/// Bounds for the map-based drafters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramMapConfig {
    /// Length of the key n-gram (`--spec-ngram-map-k*-size-n`).
    pub size_n: usize,
    /// Length of the value m-gram drafted (`--spec-ngram-map-k*-size-m`).
    pub size_m: usize,
    /// `true` is `ngram-map-k`: draft straight from the newest key match.
    /// `false` is `ngram-map-k4v`: track up to [`NGRAM_MAP_MAX_VALUES`]
    /// distinct continuations per key and draft only a dominant one.
    pub key_only: bool,
    /// Minimum key hits before a k4v draft (`--spec-ngram-map-k*-min-hits`).
    /// The key-only path drafts on the first hit, as upstream does.
    pub min_hits: u16,
    /// Tokens of history the speculator pre-allocates for.
    pub history_capacity: usize,
}

impl NgramMapConfig {
    pub fn new(
        size_n: usize,
        size_m: usize,
        key_only: bool,
        min_hits: u16,
        history_capacity: usize,
    ) -> Result<Self, NgramConfigError> {
        if size_n == 0 {
            return Err(NgramConfigError::InvalidNgramRange);
        }
        if size_m == 0 {
            return Err(NgramConfigError::ZeroDraftTokens);
        }
        if min_hits == 0 {
            return Err(NgramConfigError::ZeroMinHits);
        }
        // The draft path needs `cur_len >= 2n + m` to ever fire.
        if history_capacity < 2 * size_n + size_m + 1 {
            return Err(NgramConfigError::HistoryTooShort);
        }
        Ok(Self {
            size_n,
            size_m,
            key_only,
            min_hits,
            history_capacity,
        })
    }
}

/// One tracked continuation of a key n-gram
/// (llama.cpp `common_ngram_map_value`).
#[derive(Debug, Clone, Copy)]
struct MapValue {
    /// Index of the value m-gram in the history; 0 marks an unused slot.
    value_idx: usize,
    /// Occurrences of this m-gram after the key, saturating.
    value_num: u16,
    /// Tokens accepted the last time this value was drafted; caps the next
    /// draft's length. Initialized to `size_m`.
    n_accepted: i16,
}

/// Statistics of one key n-gram (llama.cpp `common_ngram_map_key`).
#[derive(Debug, Clone)]
struct MapKey {
    /// Index of the key n-gram in the history.
    key_idx: usize,
    /// History index the value statistics have been computed up to.
    stat_idx: usize,
    /// Draft-time hits of this key, saturating.
    key_num: u16,
    values: [MapValue; NGRAM_MAP_MAX_VALUES],
}

/// llama.cpp's `common_ngram_map` over an owned history.
#[derive(Debug, Clone)]
pub struct NgramMapSpeculator {
    config: NgramMapConfig,
    history: Vec<i32>,
    /// Key n-grams that have matched at draft time, searched linearly (as
    /// upstream searches its `keys` vector). At most one key is added per
    /// decode step, and [`KEYS_RESERVED`] of them are pre-allocated, so the
    /// hot path reallocates at most a handful of times per sequence.
    keys: Vec<MapKey>,
    /// Hash → history index of a key n-gram; 0 marks an empty slot,
    /// collisions keep the first writer (upstream `key_map`).
    key_map: Vec<u32>,
    /// Highest history index whose n-gram hash has been inserted; linear
    /// scans only cover positions above it.
    key_map_last_idx: u32,
    /// History length when generation started (upstream `size_last_begin`,
    /// set by `common_ngram_map_begin`).
    size_last_begin: usize,
    generation_started: bool,
    last_draft_created: bool,
    last_draft_key_idx: usize,
    last_draft_value_idx: usize,
    /// Highest `cur_len` a draft call has seen (upstream `idx_last_check`).
    idx_last_check: usize,
}

impl NgramMapSpeculator {
    pub fn new(config: NgramMapConfig) -> Self {
        Self {
            config,
            history: Vec::with_capacity(config.history_capacity),
            keys: Vec::with_capacity(KEYS_RESERVED),
            key_map: vec![0; NGRAM_MAP_HASH_MAP_SIZE],
            key_map_last_idx: 0,
            size_last_begin: 0,
            generation_started: false,
            last_draft_created: false,
            last_draft_key_idx: 0,
            last_draft_value_idx: 0,
            idx_last_check: 0,
        }
    }

    pub fn config(&self) -> NgramMapConfig {
        self.config
    }

    /// Add one generated token. The first call marks the prompt/generation
    /// boundary — llama.cpp's `common_ngram_map_begin`, whose cleanup
    /// branches are dead on an append-only history.
    pub fn observe(&mut self, token: i32) {
        if !self.generation_started {
            self.generation_started = true;
            self.size_last_begin = self.history.len();
            self.idx_last_check = self.history.len();
        }
        push_bounded(&mut self.history, token);
    }

    /// Add prompt tokens (chunked prefill delivers them in pieces).
    pub fn observe_all(&mut self, tokens: &[i32]) {
        for &token in tokens {
            push_bounded(&mut self.history, token);
        }
    }

    /// llama.cpp's `common_ngram_map_draft`. Returns the number appended to
    /// `out`, truncated to the capacity the caller reserved.
    pub fn propose_into(&mut self, out: &mut Vec<i32>) -> usize {
        self.last_draft_created = false;
        self.last_draft_key_idx = 0;
        self.last_draft_value_idx = 0;

        let room = out
            .capacity()
            .saturating_sub(out.len())
            .min(self.config.size_m);
        if room == 0 || self.history.is_empty() {
            return 0;
        }
        let n = self.config.size_n;
        let m = self.config.size_m;
        let cur_len = self.history.len() - 1;
        if cur_len < 2 * n + m {
            return 0;
        }
        debug_assert!(cur_len < u32::MAX as usize, "key_map stores u32 indices");
        debug_assert!(
            self.idx_last_check <= cur_len && self.size_last_begin <= cur_len,
            "history is append-only"
        );
        self.idx_last_check = cur_len;

        let h = &self.history;
        // The key n-gram: the last `n - 1` committed tokens plus the
        // sampled one — the history's n-token tail.
        let key = &h[h.len() - n..];

        // Newest match first via the hash map, then linear scans of what
        // the map has not indexed yet: the pre-generation history, then the
        // generated region. Each scan runs newest-first and stops above
        // `key_map_last_idx`, below which every n-gram is already hashed.
        let mut match_pos = 0;
        if !self.key_map.is_empty() {
            let idx_hash = map_hash(key) as usize % self.key_map.len();
            let idx_key = self.key_map[idx_hash] as usize;
            if idx_key != 0 && idx_key < cur_len - n - m - 1 && h[idx_key..idx_key + n] == *key {
                match_pos = idx_key;
            }
        }
        if match_pos == 0 && self.size_last_begin > n + m + 1 {
            let mut j = self.size_last_begin - n - m - 1;
            while j > self.key_map_last_idx as usize {
                if h[j..j + n] == *key {
                    match_pos = j;
                    break;
                }
                j -= 1;
            }
        }
        if match_pos == 0 {
            let mut j = cur_len - n - m - 1;
            while j > self.size_last_begin && j > self.key_map_last_idx as usize {
                if h[j..j + n] == *key {
                    match_pos = j;
                    break;
                }
                j -= 1;
            }
        }

        // Index the n-grams the scans above just walked, same order.
        if !self.key_map.is_empty() {
            let entries = self.key_map.len();
            let indexed = self.key_map_last_idx as usize;
            if self.size_last_begin > n + m + 1 {
                let mut j = self.size_last_begin - n - m - 1;
                while j > indexed {
                    let idx_hash = map_hash(&h[j..j + n]) as usize % entries;
                    if self.key_map[idx_hash] == 0 {
                        self.key_map[idx_hash] = j as u32;
                    }
                    j -= 1;
                }
            }
            let mut j = cur_len - n - m - 1;
            while j > self.size_last_begin && j > indexed {
                let idx_hash = map_hash(&h[j..j + n]) as usize % entries;
                if self.key_map[idx_hash] == 0 {
                    self.key_map[idx_hash] = j as u32;
                }
                j -= 1;
            }
            self.key_map_last_idx = self.key_map_last_idx.max((cur_len - n - m - 1) as u32);
        }

        if match_pos == 0 {
            return 0;
        }

        // Find or create the key's statistics entry.
        let mut key_offset = self.keys.len();
        for (i, k) in self.keys.iter().enumerate() {
            if h[k.key_idx..k.key_idx + n] == *key {
                key_offset = i;
                break;
            }
        }
        if key_offset == self.keys.len() {
            self.keys.push(MapKey {
                key_idx: match_pos,
                stat_idx: 0,
                key_num: 0,
                values: [MapValue {
                    value_idx: 0,
                    value_num: 0,
                    n_accepted: m as i16,
                }; NGRAM_MAP_MAX_VALUES],
            });
        }

        let key_num = (u32::from(self.keys[key_offset].key_num) + 1).min(MAX_VALUE_COUNT) as u16;
        self.keys[key_offset].key_num = key_num;

        if self.config.key_only {
            // ngram-map-k: draft the m tokens after the newest match,
            // capped by what the last draft from this key got accepted.
            // Upstream drafts here without consulting min_hits.
            let n_accepted = self.keys[key_offset].values[0].n_accepted;
            let count = (m as i64).min(i64::from(n_accepted)).max(0) as usize;
            let count = count.min(room);
            out.extend_from_slice(&h[match_pos + n..match_pos + n + count]);
            self.last_draft_created = true;
            self.last_draft_key_idx = key_offset;
            self.last_draft_value_idx = 0;
            return count;
        }

        if key_num < self.config.min_hits {
            return 0;
        }

        // ngram-map-k4v: fold every key occurrence in
        // `[stat_idx, match_pos]` into the (at most four) value slots.
        let stat_start = self.keys[key_offset].stat_idx;
        for i in stat_start..=match_pos {
            if h[i..i + n] != *key {
                continue;
            }
            let value_start = i + n;
            let mut idx_value = None;
            for (v, value) in self.keys[key_offset].values.iter_mut().enumerate() {
                if value.value_idx == 0 {
                    // An empty slot: a value m-gram not seen before.
                    value.value_idx = value_start;
                    value.value_num = 0;
                    value.n_accepted = m as i16;
                    idx_value = Some(v);
                    break;
                }
                if h[value_start..value_start + m] == h[value.value_idx..value.value_idx + m] {
                    idx_value = Some(v);
                    break;
                }
            }
            if let Some(v) = idx_value {
                let value = &mut self.keys[key_offset].values[v];
                value.value_num = (u32::from(value.value_num) + 1).min(MAX_VALUE_COUNT) as u16;
            }
        }
        self.keys[key_offset].stat_idx = match_pos;

        // Draft only when one value dominates: its count must be at least
        // twice the other slots' combined.
        let values = &self.keys[key_offset].values;
        let mut slot_max = 0;
        let mut max_occur = 0u16;
        for (v, value) in values.iter().enumerate() {
            if value.value_num > max_occur {
                max_occur = value.value_num;
                slot_max = v;
            }
        }
        let sum_occur: u32 = values
            .iter()
            .enumerate()
            .filter(|&(v, _)| v != slot_max)
            .map(|(_, value)| u32::from(value.value_num))
            .sum();
        if sum_occur > 0 && u32::from(max_occur) < 2 * sum_occur {
            return 0;
        }

        let n_accepted = values[slot_max].n_accepted;
        let count = (m as i64).min(i64::from(n_accepted)).max(0) as usize;
        let count = count.min(room);
        out.extend_from_slice(&h[match_pos + n..match_pos + n + count]);
        self.last_draft_created = true;
        self.last_draft_key_idx = key_offset;
        self.last_draft_value_idx = slot_max;
        count
    }

    /// llama.cpp's `common_ngram_map_accept`: record how much of the last
    /// draft the target accepted, capping that value's next draft length.
    pub fn accept(&mut self, n_accepted: usize) {
        if !self.last_draft_created {
            return;
        }
        let value = &mut self.keys[self.last_draft_key_idx].values[self.last_draft_value_idx];
        value.n_accepted = n_accepted.min(i16::MAX as usize) as i16;
    }
}

/// Append within the pre-reserved capacity. A serving sequence cannot
/// outlive the pool the capacity was sized from, so hitting the cap is a
/// bug upstream of here; in release the token is dropped, which can only
/// cost draft quality — every draft is verified by the target model.
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

    fn simple(size_n: usize, size_m: usize) -> NgramSimpleSpeculator {
        NgramSimpleSpeculator::new(NgramSimpleConfig::new(size_n, size_m, 512).unwrap())
    }

    fn map(size_n: usize, size_m: usize, key_only: bool, min_hits: u16) -> NgramMapSpeculator {
        NgramMapSpeculator::new(
            NgramMapConfig::new(size_n, size_m, key_only, min_hits, 512).unwrap(),
        )
    }

    fn propose(spec: &mut NgramMapSpeculator, cap: usize) -> Vec<i32> {
        let mut out = Vec::with_capacity(cap);
        spec.propose_into(&mut out);
        out
    }

    #[test]
    fn simple_drafts_the_continuation_of_the_newest_match() {
        let mut spec = simple(2, 3);
        // History long enough (`cur_len > n + m + 1`), with (7, 8) seen
        // earlier followed by 9, 10, 11.
        spec.observe_all(&[1, 2, 3, 7, 8, 9, 10, 11, 5, 7]);
        spec.observe(8);
        let mut out = Vec::with_capacity(3);
        assert_eq!(spec.propose_into(&mut out), 3);
        assert_eq!(out, [9, 10, 11]);
    }

    #[test]
    fn simple_rejects_a_draft_shorter_than_the_lookup() {
        // The (7, 8, 9) match is found but only two continuation tokens
        // exist before the current suffix; upstream requires at least
        // `size_n` = 3 draft tokens.
        let mut spec = simple(3, 4);
        spec.observe_all(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 8]);
        spec.observe(9);
        let mut out = Vec::with_capacity(4);
        assert_eq!(spec.propose_into(&mut out), 0);
    }

    #[test]
    fn simple_needs_more_history_than_n_plus_m_plus_one() {
        let mut spec = simple(2, 3);
        spec.observe_all(&[7, 8, 9, 7]); // cur_len = 3 <= 2 + 3 + 1
        spec.observe(8);
        let mut out = Vec::with_capacity(3);
        assert_eq!(spec.propose_into(&mut out), 0);
    }

    #[test]
    fn simple_never_grows_the_callers_buffer() {
        let mut spec = simple(2, 4);
        spec.observe_all(&[1, 2, 3, 7, 8, 9, 10, 11, 12, 5, 7]);
        spec.observe(8);
        let mut out = Vec::with_capacity(2);
        let capacity = out.capacity();
        assert_eq!(spec.propose_into(&mut out), 2);
        assert_eq!(out.capacity(), capacity);
        assert_eq!(out, [9, 10]);
    }

    #[test]
    fn map_k_drafts_from_the_first_hit() {
        let mut spec = map(2, 2, true, 1);
        // Key (7, 8) at index 1 (index 0 must stay unused: it means "no
        // match" upstream), followed by (9, 10).
        spec.observe_all(&[0, 7, 8, 9, 10, 1, 2, 3, 4, 7]);
        spec.observe(8);
        assert_eq!(propose(&mut spec, 2), [9, 10]);
    }

    #[test]
    fn map_k_shrinks_the_next_draft_to_what_was_accepted() {
        let mut spec = map(2, 2, true, 1);
        spec.observe_all(&[0, 7, 8, 9, 10, 1, 2, 3, 4, 7]);
        spec.observe(8);
        assert_eq!(propose(&mut spec, 2), [9, 10]);
        spec.accept(1);
        // Same key again later: the draft is capped at the accepted length.
        spec.observe_all(&[9, 5, 6, 7]);
        spec.observe(8);
        assert_eq!(propose(&mut spec, 2), [9]);
    }

    #[test]
    fn map_k4v_needs_min_hits_before_drafting() {
        let mut spec = map(2, 2, false, 2);
        spec.observe_all(&[0, 7, 8, 9, 10, 1, 2, 3, 4, 7]);
        spec.observe(8);
        // First hit: key_num = 1 < min_hits = 2, no draft.
        assert_eq!(propose(&mut spec, 2), Vec::<i32>::new());
        spec.observe_all(&[9, 5, 6, 7]);
        spec.observe(8);
        // Second hit drafts the (9, 10) continuation.
        assert_eq!(propose(&mut spec, 2), [9, 10]);
    }

    #[test]
    fn map_k4v_withholds_a_draft_when_no_value_dominates() {
        let mut spec = map(2, 2, false, 1);
        // (7, 8) continues as (1, 2) once and (3, 4) once: neither value
        // reaches twice the other's count, so no draft.
        spec.observe_all(&[0, 7, 8, 1, 2, 5, 7, 8, 3, 4, 5, 6, 7]);
        spec.observe(8);
        assert_eq!(propose(&mut spec, 2), Vec::<i32>::new());
    }

    #[test]
    fn map_k4v_drafts_when_the_newest_continuation_dominates() {
        let mut spec = map(2, 2, false, 1);
        // (7, 8) continues as (1, 2) twice and (3, 4) once: the dominance
        // check passes (2 >= 2 * 1) and the draft is the continuation of
        // the newest match — upstream copies from `match_pos`, not from the
        // dominant value's own position.
        spec.observe_all(&[0, 7, 8, 1, 2, 7, 8, 3, 4, 7, 8, 1, 2, 5, 6, 7]);
        spec.observe(8);
        assert_eq!(propose(&mut spec, 2), [1, 2]);
    }

    #[test]
    fn map_every_draft_continues_a_real_key_occurrence() {
        // The invariant target-model verification relies on for cheapness:
        // whatever the hash map or the scans matched, the drafted tokens
        // literally follow an in-history occurrence of the key n-gram.
        for key_only in [true, false] {
            let mut spec = map(2, 3, key_only, 1);
            let mut rng = 0x243f_6a88_85a3_08d3u64;
            let mut stream = Vec::new();
            for round in 0..192 {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let token = ((rng >> 33) % 5) as i32;
                stream.push(token);
                if round < 24 {
                    spec.observe_all(&[token]);
                    continue;
                }
                spec.observe(token);
                let draft = propose(&mut spec, 3);
                if draft.is_empty() {
                    continue;
                }
                let n = 2;
                let key = &stream[stream.len() - n..];
                let legal = (1..stream.len() - n)
                    .any(|j| stream[j..j + n] == *key && stream[j + n..].starts_with(&draft));
                assert!(legal, "draft {draft:?} continues no occurrence of {key:?}");
                spec.accept(draft.len().saturating_sub(1));
            }
        }
    }

    #[test]
    fn map_draft_indices_stay_in_bounds_on_random_streams() {
        // The port carries a lot of index arithmetic; drive it with random
        // small-alphabet streams (maximum repetition) so any out-of-bounds
        // slice panics here rather than in serving.
        for key_only in [true, false] {
            let mut spec = map(3, 4, key_only, 1);
            let mut rng = 0x9e37_79b9_7f4a_7c15u64;
            for round in 0..256 {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let token = ((rng >> 33) % 3) as i32;
                if round < 32 {
                    spec.observe_all(&[token]);
                } else {
                    spec.observe(token);
                    let drafted = propose(&mut spec, 4);
                    spec.accept(drafted.len().saturating_sub(1));
                }
            }
        }
    }

    #[test]
    fn config_validation_rejects_degenerate_shapes() {
        assert!(NgramSimpleConfig::new(0, 4, 64).is_err());
        assert!(NgramSimpleConfig::new(2, 0, 64).is_err());
        assert!(NgramSimpleConfig::new(2, 4, 7).is_err());
        assert!(NgramMapConfig::new(0, 4, true, 1, 64).is_err());
        assert!(NgramMapConfig::new(2, 0, true, 1, 64).is_err());
        assert!(NgramMapConfig::new(2, 4, true, 0, 64).is_err());
        assert!(NgramMapConfig::new(12, 48, false, 1, 64).is_err());
    }
}
