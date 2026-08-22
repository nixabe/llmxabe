//! One handle over every self-speculative (model-free) drafting policy.
//!
//! The engine's runtime holds a [`SelfSpecFactory`] per worker and one
//! [`SelfSpeculator`] per decoding sequence, and drives them through four
//! calls whatever the variant: `observe_all` for prompt chunks, `observe`
//! for generated tokens, `propose_into` before a decode step, and `accept`
//! after the batched verify. The variants match llama.cpp's `--spec-type`
//! names: `ngram` (this project's own indexed longest-suffix matcher, see
//! [`crate::ngram`]), `ngram-simple`, `ngram-mod`, `ngram-map-k` and
//! `ngram-map-k4v`.
//!
//! The factory exists for `ngram-mod`, whose hash table is shared by every
//! sequence a worker serves (as llama.cpp shares one `common_ngram_mod`
//! across a context's sequences). The table lives behind `Rc<RefCell>`:
//! all speculators for one worker are created and driven on that worker's
//! single runtime thread.

use std::cell::RefCell;
use std::rc::Rc;

use crate::ngram::{NgramConfig, NgramSpeculator};
use crate::ngram_map::{
    NgramMapConfig, NgramMapSpeculator, NgramSimpleConfig, NgramSimpleSpeculator,
};
use crate::ngram_mod::{
    NGRAM_MOD_TABLE_ENTRIES, NgramModConfig, NgramModSpeculator, NgramModTable,
};

/// Which self-speculative drafter to run, with its bounds. Plain data —
/// safe to send to the runtime thread that builds the factory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelfSpecConfig {
    /// Indexed longest-suffix matching over the sequence's own history.
    Ngram(NgramConfig),
    /// llama.cpp `ngram-simple`: backward scan for a fixed-length tail.
    Simple(NgramSimpleConfig),
    /// llama.cpp `ngram-mod`: worker-shared n-gram → next-token table.
    Mod(NgramModConfig),
    /// llama.cpp `ngram-map-k`/`ngram-map-k4v`, split by `key_only`.
    Map(NgramMapConfig),
}

impl SelfSpecConfig {
    /// The most tokens one draft may propose — what sizes the engine's
    /// verify window and what the scheduler charges per step.
    pub fn max_draft_tokens(&self) -> usize {
        match self {
            Self::Ngram(config) => config.max_draft_tokens,
            Self::Simple(config) => config.size_m,
            Self::Mod(config) => config.n_max,
            Self::Map(config) => config.size_m,
        }
    }
}

/// Per-worker state behind sequence speculators: the config, plus the
/// `ngram-mod` table every sequence of this worker shares.
#[derive(Debug)]
pub struct SelfSpecFactory {
    config: SelfSpecConfig,
    mod_table: Option<Rc<RefCell<NgramModTable>>>,
}

impl SelfSpecFactory {
    pub fn new(config: SelfSpecConfig) -> Self {
        let mod_table = match &config {
            SelfSpecConfig::Mod(mod_config) => Some(Rc::new(RefCell::new(NgramModTable::new(
                mod_config.n_match,
                NGRAM_MOD_TABLE_ENTRIES,
            )))),
            _ => None,
        };
        Self { config, mod_table }
    }

    pub fn config(&self) -> SelfSpecConfig {
        self.config
    }

    pub fn max_draft_tokens(&self) -> usize {
        self.config.max_draft_tokens()
    }

    /// A fresh speculator for one admitted sequence.
    pub fn new_speculator(&self) -> SelfSpeculator {
        match self.config {
            SelfSpecConfig::Ngram(config) => SelfSpeculator::Ngram(NgramSpeculator::new(config)),
            SelfSpecConfig::Simple(config) => {
                SelfSpeculator::Simple(NgramSimpleSpeculator::new(config))
            }
            SelfSpecConfig::Mod(config) => SelfSpeculator::Mod(NgramModSpeculator::new(
                config,
                Rc::clone(
                    self.mod_table
                        .as_ref()
                        .expect("a Mod factory builds its table at construction"),
                ),
            )),
            SelfSpecConfig::Map(config) => SelfSpeculator::Map(NgramMapSpeculator::new(config)),
        }
    }
}

/// One sequence's drafting state, any variant.
#[derive(Debug, Clone)]
pub enum SelfSpeculator {
    Ngram(NgramSpeculator),
    Simple(NgramSimpleSpeculator),
    Mod(NgramModSpeculator),
    Map(NgramMapSpeculator),
}

impl SelfSpeculator {
    /// Add one generated token. For the variants that distinguish prompt
    /// from generation (`ngram-mod`'s table indexing and occupancy check,
    /// `ngram-map-*`'s begin bookkeeping), the first call marks that
    /// boundary.
    pub fn observe(&mut self, token: i32) {
        match self {
            Self::Ngram(spec) => spec.observe(token),
            Self::Simple(spec) => spec.observe(token),
            Self::Mod(spec) => spec.observe(token),
            Self::Map(spec) => spec.observe(token),
        }
    }

    /// Add prompt tokens, as chunked prefill delivers them.
    pub fn observe_all(&mut self, tokens: &[i32]) {
        match self {
            Self::Ngram(spec) => spec.observe_all(tokens),
            Self::Simple(spec) => spec.observe_all(tokens),
            Self::Mod(spec) => spec.observe_all(tokens),
            Self::Map(spec) => spec.observe_all(tokens),
        }
    }

    /// Draft into `out`, never past the capacity the caller reserved.
    /// Returns the number of tokens appended.
    pub fn propose_into(&mut self, out: &mut Vec<i32>) -> usize {
        match self {
            Self::Ngram(spec) => spec.propose_into(out),
            Self::Simple(spec) => spec.propose_into(out),
            Self::Mod(spec) => spec.propose_into(out),
            Self::Map(spec) => spec.propose_into(out),
        }
    }

    /// How many tokens of the last non-empty draft the target model
    /// accepted. `ngram` and `ngram-simple` keep no acceptance state.
    pub fn accept(&mut self, n_accepted: usize) {
        match self {
            Self::Ngram(_) | Self::Simple(_) => {}
            Self::Mod(spec) => spec.accept(n_accepted),
            Self::Map(spec) => spec.accept(n_accepted),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factory_shares_the_mod_table_and_nothing_else() {
        let factory = SelfSpecFactory::new(SelfSpecConfig::Mod(
            NgramModConfig::new(2, 0, 3, 512).unwrap(),
        ));
        let (mut a, mut b) = (factory.new_speculator(), factory.new_speculator());
        a.observe_all(&[1, 2, 3, 4, 5]);
        a.observe(9);
        b.observe_all(&[7, 7, 7, 1]);
        b.observe(2);
        let mut draft = Vec::with_capacity(3);
        assert_eq!(b.propose_into(&mut draft), 3);
        assert_eq!(draft, [3, 4, 5]);

        // The map variants share nothing: what A saw, B cannot draft from.
        let factory = SelfSpecFactory::new(SelfSpecConfig::Map(
            NgramMapConfig::new(2, 2, true, 1, 512).unwrap(),
        ));
        let (mut a, mut b) = (factory.new_speculator(), factory.new_speculator());
        a.observe_all(&[0, 7, 8, 9, 10, 1, 2, 3, 4, 7]);
        a.observe(8);
        b.observe_all(&[0, 5, 5, 5, 5, 5, 5, 5, 5, 7]);
        b.observe(8);
        let mut draft = Vec::with_capacity(2);
        assert_eq!(a.propose_into(&mut draft), 2);
        draft.clear();
        assert_eq!(b.propose_into(&mut draft), 0);
    }

    #[test]
    fn max_draft_tokens_matches_each_variant_cap() {
        let ngram = SelfSpecConfig::Ngram(NgramConfig::new(2, 4, 3, 64).unwrap());
        assert_eq!(ngram.max_draft_tokens(), 3);
        let simple = SelfSpecConfig::Simple(NgramSimpleConfig::new(12, 48, 512).unwrap());
        assert_eq!(simple.max_draft_tokens(), 48);
        let ngram_mod = SelfSpecConfig::Mod(NgramModConfig::new(24, 48, 64, 512).unwrap());
        assert_eq!(ngram_mod.max_draft_tokens(), 64);
        let map = SelfSpecConfig::Map(NgramMapConfig::new(12, 48, false, 1, 512).unwrap());
        assert_eq!(map.max_draft_tokens(), 48);
    }
}
