//! Host-side bookkeeping for the shared prefix cache.
//!
//! Two things live here: what *names* a sequence's blocks, and which
//! snapshots are held for other requests to find. Both exist to serve
//! [`crate::Engine`]'s one structural advantage over three separate
//! `llama-server` processes, and both are places where being wrong is
//! silent rather than loud.

use std::collections::HashMap;
use std::sync::Arc;

use xabe_cache::radix::{BlockHash, ROOT_HASH, hash_block};

use crate::image::ImagePlacement;
use crate::state::SequenceSnapshot;
use crate::worker::WorkerId;

/// The chained block hashes naming a sequence's complete blocks, extended as
/// the sequence generates.
///
/// # The promise this type has to keep
///
/// A snapshot filed in the shared tree under a chain at position `p` is a
/// claim: *any* sequence whose own first `p / block_size` hashes match may
/// skip prefilling its first `p` tokens and resume from this recurrent state
/// instead. If the chain and the tokens ever disagree, a request resumes from
/// a prefix it was never given and answers the wrong prompt — fluently, and
/// with nothing in the output to say so.
///
/// So a chain is only ever assembled from the sequence's own tokens, in
/// order: the prompt it was admitted with, then every token the runtime
/// reported generating, appended as it was reported. There is no other way to
/// build one, which is why this type owns the hashing rather than accepting
/// hashes from a caller.
///
/// Only *complete* blocks are hashed. A partial trailing block has no stable
/// name — the same tokens hash differently once the block fills — and
/// [`xabe_cache::radix::RadixTree::insert`] charges every entry a full
/// `block_size` of positions, so one partial entry would misplace every
/// block after it.
///
/// # Images
///
/// Every token of an image span is the same `<|image_pad|>` id, so two
/// prompts differing only in their image *bytes* would hash identically —
/// and one request would silently resume from the other's image, the exact
/// wrong-prompt failure described above. So blocks are named over a
/// substituted stream: an image slot contributes its
/// [`ImagePlacement::lane`] — a per-slot digest of the image's content
/// hash — instead of the token id. Same content, same name (cross-request
/// prefix sharing keeps working); different content, a different name in
/// every overlapping block.
pub(crate) struct SequenceChain {
    block_size: usize,
    hashes: Vec<BlockHash>,
    /// Tokens of the block being filled. Always fewer than `block_size`.
    pending: Vec<u32>,
    /// Where the prompt ended and generation began.
    prompt_len: usize,
    /// The prompt's image spans, sorted and validated by the engine.
    /// Generated tokens are always past `prompt_len` and never in a span.
    images: Vec<ImagePlacement>,
    /// Absolute index of the next token [`Self::push`] will see.
    next_index: usize,
}

impl SequenceChain {
    /// Start a chain over a request's prompt and its image placements.
    pub(crate) fn new(block_size: u32, prompt: &[i32], images: &[ImagePlacement]) -> Self {
        let block_size = block_size.max(1) as usize;
        let mut chain = Self {
            block_size,
            hashes: Vec::with_capacity(prompt.len() / block_size + 1),
            pending: Vec::with_capacity(block_size),
            prompt_len: prompt.len(),
            images: images.to_vec(),
            next_index: 0,
        };
        chain.extend(prompt.iter().copied());
        chain
    }

    /// Append one token, closing a block if that filled it.
    ///
    /// Image slots contribute their content lane instead of the token id —
    /// see the type docs.
    pub(crate) fn push(&mut self, token: i32) {
        let index = self.next_index;
        self.next_index += 1;
        let value = self
            .images
            .iter()
            .find(|img| index >= img.start && index < img.end())
            .map_or(token as u32, |img| img.lane(index - img.start));
        self.pending.push(value);
        if self.pending.len() == self.block_size {
            let parent = self.hashes.last().copied().unwrap_or(ROOT_HASH);
            self.hashes.push(hash_block(parent, &self.pending));
            self.pending.clear();
        }
    }

    pub(crate) fn extend(&mut self, tokens: impl Iterator<Item = i32>) {
        for token in tokens {
            self.push(token);
        }
    }

    /// The chain over every complete block so far.
    pub(crate) fn hashes(&self) -> &[BlockHash] {
        &self.hashes
    }

    /// Tokens accounted for: hashed blocks plus the block being filled.
    pub(crate) fn position(&self) -> usize {
        self.hashes.len() * self.block_size + self.pending.len()
    }

    /// The hashes naming exactly the blocks a snapshot at `position` covers,
    /// or `None` if this chain cannot name them.
    ///
    /// It cannot when the sequence has not reached that position yet, or when
    /// `position` falls inside a block rather than on its edge — a snapshot
    /// covering half a block could not be described by any chain, so it must
    /// not be shared under one.
    pub(crate) fn hashes_for(&self, position: usize) -> Option<&[BlockHash]> {
        if position == 0 || !position.is_multiple_of(self.block_size) {
            return None;
        }
        let blocks = position / self.block_size;
        (blocks <= self.hashes.len()).then(|| &self.hashes[..blocks])
    }

    /// The token the *model* produced at `position`, for checking against a
    /// snapshot's own record of what it predicted there.
    ///
    /// `None` unless two things hold. The position must be at or past the end
    /// of the prompt: a snapshot taken mid-prompt records the model's
    /// prediction for that point, which is not the prompt's next token and
    /// has no reason to equal it — comparing them would report drift on every
    /// prefill. And the position must still be inside the block being filled,
    /// since a block's tokens are dropped once it is hashed. The one caller
    /// checks a snapshot in the step that produced it, so that second
    /// condition holds whenever the first does.
    pub(crate) fn generated_token_at(&self, position: usize) -> Option<u32> {
        if position < self.prompt_len {
            return None;
        }
        let hashed = self.hashes.len() * self.block_size;
        position
            .checked_sub(hashed)
            .and_then(|offset| self.pending.get(offset).copied())
    }
}

/// Snapshots published for other requests to resume from, keyed by the hash
/// of the block they end on.
///
/// # Why this is bounded
///
/// Every entry pins at least one of its worker's pinned snapshot slots, and
/// often several: a snapshot stores only the KV of its own retention interval
/// and holds an `Arc` to its parent for everything before, so publishing a
/// deep snapshot pins the whole chain behind it.
///
/// Those are the same slots live sequences take their own snapshots from, and
/// running out is not a soft failure: the runtime answers
/// `SnapshotArenaExhausted` by setting `retention_disabled` on the live
/// sequence, permanently, so that sequence stops retaining for the rest of
/// its life. Hoarding a snapshot against a future hit therefore costs a
/// present one, and costs it silently.
///
/// So the cache yields under pressure rather than holding: before a worker
/// publishes, its least recently *used* entries are dropped until that
/// worker's arena has room to spare again. Recency counts actual reuse, not
/// insertion, because the case this project exists to serve — one system
/// prompt shared by three workers — is exactly the entry that is old and
/// still worth keeping.
///
/// Generic over what is held so the eviction policy can be tested without a
/// device: a `SequenceSnapshot` owns a lease on pinned device-adjacent
/// memory and cannot be built host-side.
pub(crate) struct SharedSnapshots<T = Arc<SequenceSnapshot>> {
    entries: HashMap<BlockHash, Held<T>>,
    clock: u64,
}

impl<T> Default for SharedSnapshots<T> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            clock: 0,
        }
    }
}

struct Held<T> {
    worker: WorkerId,
    value: T,
    last_used: u64,
}

impl<T: Clone> SharedSnapshots<T> {
    /// Whether a snapshot is available under this hash. Does not count as
    /// use: routing asks this about every candidate worker, and only one of
    /// them will actually resume.
    pub(crate) fn contains(&self, hash: BlockHash) -> bool {
        self.entries.contains_key(&hash)
    }

    /// Take a snapshot to resume from, marking it as used.
    pub(crate) fn take_for_reuse(&mut self, hash: BlockHash) -> Option<T> {
        self.clock += 1;
        let clock = self.clock;
        let held = self.entries.get_mut(&hash)?;
        held.last_used = clock;
        Some(held.value.clone())
    }

    pub(crate) fn publish(&mut self, hash: BlockHash, worker: WorkerId, value: T) {
        self.clock += 1;
        self.entries.insert(
            hash,
            Held {
                worker,
                value,
                last_used: self.clock,
            },
        );
    }

    pub(crate) fn remove(&mut self, hash: BlockHash) {
        self.entries.remove(&hash);
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Drop `worker`'s least recently used entries until its arena reports at
    /// least `reserve` free slots, or until this cache holds nothing more of
    /// that worker's.
    ///
    /// `available` is re-read after each drop rather than predicted: an entry
    /// whose parent chain a live sequence still holds frees only its own
    /// tail, so how much a drop actually recovers is not knowable from here.
    ///
    /// Returns whether there is now room to spare.
    pub(crate) fn reclaim_for(
        &mut self,
        worker: WorkerId,
        reserve: usize,
        mut available: impl FnMut() -> usize,
    ) -> bool {
        while available() < reserve {
            let victim = self
                .entries
                .iter()
                .filter(|(_, held)| held.worker == worker)
                .min_by_key(|(_, held)| held.last_used)
                .map(|(&hash, _)| hash);
            let Some(hash) = victim else {
                return false;
            };
            self.entries.remove(&hash);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chain_hashes_complete_blocks_and_holds_the_rest() {
        let tokens: Vec<i32> = (0..10).collect();
        let chain = SequenceChain::new(4, &tokens, &[]);
        assert_eq!(chain.position(), 10);
        assert_eq!(
            chain.hashes().len(),
            2,
            "8 of 10 tokens complete two blocks"
        );

        let first = hash_block(ROOT_HASH, &[0, 1, 2, 3]);
        assert_eq!(chain.hashes()[0], first);
        assert_eq!(chain.hashes()[1], hash_block(first, &[4, 5, 6, 7]));
    }

    #[test]
    fn generating_a_block_extends_the_chain_exactly_as_a_prompt_would() {
        // The property the whole fix rests on: a sequence's name does not
        // depend on which of its tokens were prompt and which were generated.
        // A second turn that sends the first turn's reply back as prompt must
        // produce the hashes the first turn published while generating it.
        let prompt: Vec<i32> = (0..6).collect();
        let generated: Vec<i32> = (6..12).collect();

        let mut live = SequenceChain::new(4, &prompt, &[]);
        live.extend(generated.iter().copied());

        let whole: Vec<i32> = prompt.iter().chain(&generated).copied().collect();
        let replayed = SequenceChain::new(4, &whole, &[]);

        assert_eq!(live.hashes(), replayed.hashes());
        assert_eq!(live.position(), replayed.position());
    }

    #[test]
    fn a_chain_names_only_positions_it_has_reached_and_lands_on() {
        let chain = SequenceChain::new(4, &(0..10).collect::<Vec<i32>>(), &[]);
        assert_eq!(chain.hashes_for(8).map(<[u64]>::len), Some(2));
        assert_eq!(chain.hashes_for(4).map(<[u64]>::len), Some(1));
        // Past the complete blocks, even though the chain has seen 10 tokens.
        assert_eq!(chain.hashes_for(12), None);
        // Inside a block rather than on its edge.
        assert_eq!(chain.hashes_for(6), None);
        // The empty prefix names nothing.
        assert_eq!(chain.hashes_for(0), None);
    }

    #[test]
    fn a_chain_answers_only_for_generated_positions_it_still_holds() {
        let mut chain = SequenceChain::new(4, &(0..6).collect::<Vec<i32>>(), &[]);
        chain.extend(6..10);
        // Two blocks hashed, so positions 8 and 9 are still pending.
        assert_eq!(chain.generated_token_at(8), Some(8));
        assert_eq!(chain.generated_token_at(9), Some(9));
        // Position 10 has not been reached; 4 is inside a hashed block.
        assert_eq!(chain.generated_token_at(10), None);
        assert_eq!(chain.generated_token_at(4), None);
    }

    #[test]
    fn a_chain_will_not_answer_for_a_position_inside_the_prompt() {
        // A snapshot taken mid-prompt records what the model *would* have
        // said next, which is not the prompt's next token. Answering here
        // would report drift on every prefill snapshot and decline to share
        // a perfectly good one.
        let chain = SequenceChain::new(4, &(0..10).collect::<Vec<i32>>(), &[]);
        assert_eq!(chain.generated_token_at(8), None);
        assert_eq!(chain.generated_token_at(9), None);
    }

    #[test]
    fn a_short_prompt_names_nothing() {
        let chain = SequenceChain::new(256, &[1, 2, 3], &[]);
        assert!(chain.hashes().is_empty());
        assert_eq!(chain.hashes_for(256), None);
    }

    #[test]
    fn identical_image_pads_with_different_content_hash_differently() {
        use crate::image::ImagePlacement;
        // Two prompts, byte-identical token streams (the pad id repeated),
        // different image content. Their chains must diverge in the first
        // block the image touches — this is the silent-wrong-answer case.
        let pad = 248_056i32;
        let prompt: Vec<i32> = vec![100, 101, pad, pad, pad, pad, pad, pad, 102, 103];
        let img = |hash| ImagePlacement {
            start: 2,
            grid_h: 2,
            grid_w: 3,
            content_hash: hash,
        };
        let a = SequenceChain::new(4, &prompt, &[img(1)]);
        let b = SequenceChain::new(4, &prompt, &[img(2)]);
        assert_eq!(a.hashes().len(), 2);
        assert_ne!(a.hashes()[0], b.hashes()[0], "first overlapping block");
        assert_ne!(a.hashes()[1], b.hashes()[1]);

        // Same content: identical names, so cross-request sharing works.
        let c = SequenceChain::new(4, &prompt, &[img(1)]);
        assert_eq!(a.hashes(), c.hashes());
    }

    #[test]
    fn image_lanes_do_not_disturb_text_only_blocks() {
        use crate::image::ImagePlacement;
        // A block entirely before the image span hashes exactly as a
        // text-only chain would — prefix sharing up to the image survives.
        let pad = 248_056i32;
        let prompt: Vec<i32> = vec![1, 2, 3, 4, pad, pad, pad, pad];
        let img = ImagePlacement {
            start: 4,
            grid_h: 2,
            grid_w: 2,
            content_hash: 7,
        };
        let with_image = SequenceChain::new(4, &prompt, &[img]);
        let text_only = SequenceChain::new(4, &[1, 2, 3, 4], &[]);
        assert_eq!(with_image.hashes()[0], text_only.hashes()[0]);
    }

    fn shared_with_three() -> SharedSnapshots<u32> {
        let mut shared = SharedSnapshots::<u32>::default();
        shared.publish(1, WorkerId(0), 10);
        shared.publish(2, WorkerId(0), 20);
        shared.publish(3, WorkerId(0), 30);
        shared
    }

    /// A stand-in arena that recovers exactly one slot per drop: the nth call
    /// reports n free.
    fn one_slot_per_drop() -> impl FnMut() -> usize {
        let mut recovered = 0;
        move || {
            let free = recovered;
            recovered += 1;
            free
        }
    }

    #[test]
    fn reclaim_drops_the_least_recently_used_first() {
        // Recency has to count reuse rather than insertion, or the shared
        // system prompt — old by construction, and the whole point of the
        // engine — is the first thing evicted.
        let mut shared = shared_with_three();
        shared.take_for_reuse(1).expect("entry 1 is held");

        assert!(shared.reclaim_for(WorkerId(0), 1, one_slot_per_drop()));

        assert_eq!(shared.len(), 2, "one drop was enough to reach the reserve");
        assert!(
            shared.contains(1),
            "the reused entry must outlive the idle ones"
        );
        assert!(
            !shared.contains(2),
            "the least recently used entry goes first"
        );
        assert!(shared.contains(3));
    }

    #[test]
    fn reclaim_gives_up_rather_than_looping_when_it_holds_nothing_of_this_workers() {
        // Slots are per worker: dropping worker 1's entries would not free a
        // single one of worker 0's, so there is nothing to be done and the
        // caller must be told so rather than spun.
        let mut shared = SharedSnapshots::<u32>::default();
        shared.publish(1, WorkerId(1), 10);

        assert!(!shared.reclaim_for(WorkerId(0), 1, || 0));
        assert!(shared.contains(1));
    }

    #[test]
    fn reclaim_does_nothing_when_there_is_already_room() {
        let mut shared = shared_with_three();
        assert!(shared.reclaim_for(WorkerId(0), 2, || 5));
        assert_eq!(shared.len(), 3);
    }
}
