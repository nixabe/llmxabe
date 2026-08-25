//! Chunked prefill scheduler with admission control.
//!
//! Ported design, not code, from vLLM's `Scheduler.schedule()`
//! (`vllm/v1/core/sched/scheduler.py`): each step first batches every
//! decode-ready running request, then spends whatever token budget remains
//! on prefill — continuing any request already mid-chunk before admitting
//! new ones from the waiting queue — chunking the last request that
//! doesn't fit and leaving the rest for a later step. This pairs
//! compute-bound prefill with memory-bound decode in a single batch.
//!
//! Two things are simplified relative to a full production scheduler and
//! are called out where they matter: KV capacity is reserved for a
//! request's full worst-case lifetime (`prompt + max_output_tokens`) the
//! moment it is pulled from the waiting queue into the running batch,
//! rather than growing incrementally block-by-block as generation
//! proceeds; and speculative draft-token rejection is modeled only at the
//! token-budget level ([`crate::config::SchedulerConfig::tokens_per_decode_step`]),
//! not at the level of discarding individual speculative KV blocks. Both
//! are noted again on the relevant methods below.

use std::collections::VecDeque;

use crate::config::SchedulerConfig;
use crate::error::AdmissionError;
use crate::request::{BatchDescription, DecodeItem, NewRequest, PrefillItem, RequestId};

/// Scheduler-internal bookkeeping for one request. Not exposed directly —
/// callers see [`RequestId`] and the [`BatchDescription`] a step produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TrackedRequest {
    id: RequestId,
    prompt_tokens: u32,
    max_output_tokens: u32,
    /// Tokens computed so far (prefill progress, then decode progress once
    /// this reaches `prompt_tokens`).
    computed_tokens: u32,
    /// Attention blocks reserved for this request's full potential
    /// lifetime. Zero while the request sits in the waiting queue; set the
    /// moment it is pulled into the running batch.
    blocks_reserved: u32,
}

impl TrackedRequest {
    fn full_seq_len(&self) -> u32 {
        self.prompt_tokens + self.max_output_tokens
    }

    fn is_decode_ready(&self) -> bool {
        self.computed_tokens >= self.prompt_tokens
    }
}

/// Chunked prefill scheduler.
///
/// Owns the waiting/running queues and a plain count of free attention
/// blocks (this crate's caller is expected to keep this in step with the
/// real [`xabe_cache::pool::BlockPool`]; the scheduler doesn't hold a live
/// pool itself so it stays testable without a GPU or a real cache).
pub struct Scheduler {
    config: SchedulerConfig,
    total_attention_blocks: u32,
    free_attention_blocks: u32,
    waiting: VecDeque<TrackedRequest>,
    running: Vec<TrackedRequest>,
    /// Rotating start for phase 2, so a step that cannot cover every
    /// prefilling session shorts a different one each time.
    prefill_cursor: usize,
}

/// How much of a step's token budget is held back for admission while a long
/// prefill is in flight, as a divisor: half of the budget.
///
/// A half rather than some smaller slice because of what the *runtime* does
/// with a grant. A prefill pass may not straddle a snapshot boundary, so with
/// a retention interval of 2048 the widest pass the engine ever issues is
/// 2048 — `-pc 4096` decomposes into two of them, never one wide pass. A
/// grant is therefore only as good as its alignment to that interval: 2048
/// lands as one full-width pass, while 3072 lands as one full-width pass plus
/// a 1024 remainder chopped into four 256-wide passes, because the prebuilt
/// tail shapes stop at 256.
///
/// So halving the budget is not the conservative choice between a quarter and
/// a half — at a 4096 budget it is the choice that keeps *both* the
/// continuing prefill and the newcomer on one aligned 2048-wide pass each,
/// where a quarter misaligns both. It applies only when the waiting queue is
/// non-empty, so it costs nothing in the throughput case.
const ADMISSION_RESERVE_FRACTION: u32 = 2;

impl Scheduler {
    pub fn new(config: SchedulerConfig, total_attention_blocks: u32) -> Self {
        Self {
            config,
            total_attention_blocks,
            free_attention_blocks: total_attention_blocks,
            waiting: VecDeque::new(),
            running: Vec::new(),
            prefill_cursor: 0,
        }
    }

    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    pub fn total_attention_blocks(&self) -> u32 {
        self.total_attention_blocks
    }

    pub fn free_attention_blocks(&self) -> u32 {
        self.free_attention_blocks
    }

    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    pub fn running_len(&self) -> usize {
        self.running.len()
    }

    pub fn is_running(&self, id: RequestId) -> bool {
        self.running.iter().any(|r| r.id == id)
    }

    pub fn is_waiting(&self, id: RequestId) -> bool {
        self.waiting.iter().any(|r| r.id == id)
    }

    /// Attention blocks held for a running request's full lifetime.
    ///
    /// This is the single source of truth the worker uses to mirror the
    /// scheduler reservation into its physical attention-block pool.
    pub fn reserved_attention_blocks(&self, id: RequestId) -> Option<u32> {
        self.running
            .iter()
            .find(|request| request.id == id)
            .map(|request| request.blocks_reserved)
    }

    /// Account for additional target tokens accepted by one speculative
    /// decode step beyond the one token charged by [`Self::step`].
    pub fn advance_speculative(&mut self, id: RequestId, additional: u32) -> bool {
        let Some(req) = self.running.iter_mut().find(|req| req.id == id) else {
            return false;
        };
        req.computed_tokens = req
            .computed_tokens
            .saturating_add(additional)
            .min(req.full_seq_len());
        true
    }

    fn attention_blocks_needed(&self, full_seq_len: u32) -> u32 {
        // Speculative decode's verify pass writes its whole fixed window of
        // K/V rows before acceptance truncates anything, so a drafting
        // engine's true per-sequence footprint is `full_seq_len` plus one
        // draft window of scratch. Reserving it here keeps rule 4 honest:
        // the blocks a sequence will actually touch are the blocks reserved.
        (full_seq_len + self.config.draft_tokens_per_step()).div_ceil(self.config.block_size())
    }

    /// Whether phase 3 of a step could actually admit somebody, checked
    /// before phases 1 and 2 spend the budget.
    ///
    /// The admission reserve is only worth holding if it can be spent. Phase
    /// 3 needs three things at once — a queued request, a free decode slot,
    /// and enough block headroom to take it without breaching the watermark —
    /// and this asks the same three questions in the same order, against the
    /// same state. Phases 1 and 2 move none of it: a decode step and a
    /// prefill continuation neither admit a request nor reserve a block, so
    /// the answer here is the answer phase 3 reaches.
    ///
    /// Gating the reserve on a non-empty queue alone was not enough. A worker
    /// running its full complement of sequences with a further request queued
    /// held back a slice of every step that phase 3 then refused to spend,
    /// because `running.len() < max_concurrent_decodes` was already false —
    /// dead budget on every step until a slot freed, which is exactly when
    /// the queue is longest.
    fn admission_can_proceed(&self) -> bool {
        if self.running.len() >= self.config.max_concurrent_decodes() as usize {
            return false;
        }
        let Some(candidate) = self.waiting.front() else {
            return false;
        };
        let needed_blocks = self.attention_blocks_needed(candidate.full_seq_len());
        let watermark = self.config.watermark_blocks(self.total_attention_blocks);
        self.free_attention_blocks >= needed_blocks + watermark
    }

    /// Whether [`Self::admit`] would accept this request, without enqueueing
    /// it.
    ///
    /// The router needs this to score a worker before committing a request to
    /// it, and it must not be a reimplementation of the rule: admission
    /// feasibility is AGENTS.md rule 4, and rule 4 lives here. Both this and
    /// [`Self::admit`] derive from
    /// `Self::attention_blocks_needed` against total capacity, so they
    /// cannot disagree.
    pub fn can_admit(&self, req: &NewRequest) -> bool {
        (self.waiting.len() as u32) < self.config.max_waiting_requests()
            && self.attention_blocks_needed(req.full_seq_len()) <= self.total_attention_blocks
    }

    /// Admit a request into the waiting queue.
    ///
    /// AGENTS.md rule 4: rejects the request if its *full* potential
    /// lifetime (`prompt_tokens + max_output_tokens`) could never fit in
    /// the pool's total capacity, regardless of how small its first chunk
    /// is. Chunked prefill splits compute, not memory — a request that will
    /// eventually need more blocks than exist must never be admitted, or it
    /// over-commits capacity the moment later chunks (or its own decode
    /// phase) need room that was never actually available.
    ///
    /// This is a feasibility check against total capacity, not a
    /// reservation — blocks are actually reserved (and checked against
    /// current free capacity plus watermark headroom) when the request is
    /// pulled from the waiting queue into the running batch in [`Self::step`].
    pub fn admit(&mut self, req: NewRequest) -> Result<RequestId, AdmissionError> {
        self.admit_with_prefix(req, 0)
    }

    /// Admit a request whose first `computed_prefix` prompt tokens have been
    /// restored from a shared cache snapshot.
    pub fn admit_with_prefix(
        &mut self,
        req: NewRequest,
        computed_prefix: u32,
    ) -> Result<RequestId, AdmissionError> {
        if computed_prefix > req.prompt_tokens {
            return Err(AdmissionError::InvalidReusablePrefix {
                prefix_tokens: computed_prefix,
                prompt_tokens: req.prompt_tokens,
            });
        }
        let full_seq_len = req.full_seq_len();
        let needed_blocks = self.attention_blocks_needed(full_seq_len);
        if needed_blocks > self.total_attention_blocks {
            return Err(AdmissionError::ExceedsTotalCapacity {
                full_seq_len,
                needed_blocks,
                total_blocks: self.total_attention_blocks,
            });
        }
        if self.waiting.len() as u32 >= self.config.max_waiting_requests() {
            return Err(AdmissionError::WaitingQueueFull {
                capacity: self.config.max_waiting_requests(),
            });
        }
        self.waiting.push_back(TrackedRequest {
            id: req.id,
            prompt_tokens: req.prompt_tokens,
            max_output_tokens: req.max_output_tokens,
            computed_tokens: computed_prefix,
            blocks_reserved: 0,
        });
        Ok(req.id)
    }

    /// Mark a request as finished, freeing its reserved blocks.
    ///
    /// Returns `false` if `id` was not a running request.
    pub fn finish_request(&mut self, id: RequestId) -> bool {
        if let Some(pos) = self.running.iter().position(|r| r.id == id) {
            let req = self.running.remove(pos);
            self.free_attention_blocks += req.blocks_reserved;
            true
        } else {
            false
        }
    }

    /// Cancel a request regardless of whether it is waiting or running.
    ///
    /// Running requests release their full-lifetime reservation. Waiting
    /// requests hold no blocks yet, but still need to be removed so a dropped
    /// client cannot consume a future serving slot.
    pub fn cancel_request(&mut self, id: RequestId) -> bool {
        if self.finish_request(id) {
            return true;
        }
        if let Some(pos) = self.waiting.iter().position(|request| request.id == id) {
            self.waiting.remove(pos);
            true
        } else {
            false
        }
    }

    /// Preempt a running request by recompute.
    ///
    /// Frees its reserved blocks, discards its progress
    /// (`computed_tokens = 0`), and re-queues it at the *front* of the
    /// waiting queue so it is the first candidate re-admitted once capacity
    /// allows. Recompute rather than swap-to-host: this project's shared
    /// prefix radix cache (`xabe_cache::radix::RadixTree`) already gives
    /// swap's main benefit — not redoing prefill work another request left
    /// cached — at lower overhead than moving KV to and from host memory.
    ///
    /// Returns `false` if `id` was not a running request.
    pub fn preempt(&mut self, id: RequestId) -> bool {
        if let Some(pos) = self.running.iter().position(|r| r.id == id) {
            let mut req = self.running.remove(pos);
            self.free_attention_blocks += req.blocks_reserved;
            req.blocks_reserved = 0;
            req.computed_tokens = 0;
            self.waiting.push_front(req);
            true
        } else {
            false
        }
    }

    /// Run one scheduling step, returning a description of the batch.
    ///
    /// Three phases, in order, sharing one token budget:
    ///
    /// 1. Every decode-ready running request is scheduled unconditionally.
    ///    Each consumes [`SchedulerConfig::tokens_per_decode_step`] tokens
    ///    of budget (the drafted case, not the accepted case — see
    ///    AGENTS.md's speculative-decode note).
    /// 2. Running requests still mid-prefill (chunked from an earlier step)
    ///    continue with whatever budget remains, before any new admission.
    /// 3. Waiting requests are pulled into the running batch and given
    ///    their first chunk, gated by a watermark: a request is only
    ///    admitted from the waiting queue if enough free blocks remain
    ///    *after* reserving its full worst-case lifetime and leaving the
    ///    configured watermark headroom free. This is what stops admission
    ///    from immediately re-triggering eviction/preemption of the very
    ///    requests already running.
    ///
    /// In phases 2 and 3, the first request whose remaining work exceeds
    /// the remaining budget is chunked (scheduled partially) and nothing
    /// further is scheduled that step — this is the "chunking the last
    /// request that does not fit" behavior AGENTS.md describes.
    pub fn step(&mut self) -> BatchDescription {
        let mut batch = BatchDescription::default();
        self.step_into(&mut batch);
        batch
    }

    /// Allocation-free scheduler step when `batch` was reserved at startup.
    pub fn step_into(&mut self, batch: &mut BatchDescription) {
        batch.decodes.clear();
        batch.prefills.clear();
        let mut budget = self.config.token_budget();

        // Phase 1: every decode-ready running request, unconditionally.
        for req in self
            .running
            .iter_mut()
            .filter(|r| r.is_decode_ready() && r.computed_tokens < r.full_seq_len())
        {
            let cost = self.config.tokens_per_decode_step();
            if budget < cost {
                break;
            }
            batch.decodes.push(DecodeItem {
                id: req.id,
                tokens: cost,
            });
            req.computed_tokens += 1;
            budget -= cost;
        }

        // How much of this step to hold back so a waiting request can start.
        //
        // Zero unless phase 3 can actually spend it, and that is the whole
        // point: with nobody admittable there is nobody to hold it for, the
        // chunk below takes the entire budget exactly as it always did, and
        // peak prefill throughput is untouched. The reserve only exists in
        // the state that was pathological — a long prompt mid-flight and
        // somebody queued behind it who can be let in.
        //
        // Without it, a prefill longer than one step took the whole budget
        // every step until it finished, so phase 3 never ran and a newcomer's
        // time to first token became the *remaining* duration of whatever
        // large prompt happened to be in flight. Measured on a 63K-token
        // prompt, that was 0.0015 s idle against 29 s behind it.
        let admission_reserve = if self.admission_can_proceed() {
            self.config.token_budget() / ADMISSION_RESERVE_FRACTION
        } else {
            0
        };

        // Phase 2: every request still mid-prefill shares the step.
        //
        // "Shares" is the whole point, and it used to say "first". The old
        // loop stopped at the first request whose prompt did not fit in one
        // step, so that request took the entire budget every step until it
        // finished and every other admitted session got nothing. On one card
        // with four sessions that produced first tokens at 34, 71, 136 and
        // 203 seconds: one prompt at a time wearing three slots. A session
        // that is merely *admitted* is not a session that is running.
        //
        // Two things make sharing free rather than a trade. The grant is
        // capped at `prefill_slice`, which the server sets to the snapshot
        // retention interval — the widest pass the engine can issue anyway,
        // since a pass may not straddle a boundary — so the same tokens move
        // at the same width, merely spread across sessions. And the start
        // rotates, so when the budget covers fewer slices than there are
        // sessions, the shortfall lands on a different session each step
        // instead of always the last one.
        //
        // What it does change is the shape of latency: mean time-to-first-
        // token rises and the worst case falls. That is the right trade for
        // an interactive harness, where a session frozen for three minutes
        // reads as a hung engine.
        // Rotating, allocation-free walk over the running set. Indices
        // rather than a collected list because `step_into` is the
        // allocation-free path (rule 6) and three workers contending on the
        // host allocator produce latency spikes that read like GPU stalls.
        if !self.running.is_empty() {
            let slice = match self.config.prefill_slice() {
                0 => u32::MAX,
                slice => slice,
            };
            let count = self.running.len();
            let start = self.prefill_cursor % count;
            let mut granted = false;
            for offset in 0..count {
                if budget == 0 {
                    break;
                }
                let spendable = budget.saturating_sub(admission_reserve);
                if spendable == 0 {
                    // The reserve is all that is left. Leave it for phase 3
                    // rather than spend it here; these requests continue next
                    // step, one step later than they would have.
                    break;
                }
                let req = &mut self.running[(start + offset) % count];
                if req.is_decode_ready() {
                    continue;
                }
                let remaining = req.prompt_tokens - req.computed_tokens;
                let chunk = remaining.min(slice).min(spendable);
                if chunk == 0 {
                    continue;
                }
                batch.prefills.push(PrefillItem {
                    id: req.id,
                    tokens: chunk,
                });
                req.computed_tokens += chunk;
                budget -= chunk;
                granted = true;
            }
            if granted {
                // Advance by one so the session that went first this step
                // goes last next step.
                self.prefill_cursor = self.prefill_cursor.wrapping_add(1);
            }
        }

        // Phase 3: admit from the waiting queue, watermark-gated.
        //
        // Reached whatever phase 2 left, which under the sharing loop above
        // is the admission reserve plus anything the running sessions could
        // not use. It is deliberately not gated on whether phase 2 truncated
        // somebody: that gate is what kept the reserve from ever being
        // spent, which was the starvation this reserve exists to prevent.
        while budget > 0 && self.running.len() < self.config.max_concurrent_decodes() as usize {
            let Some(candidate) = self.waiting.front() else {
                break;
            };
            let needed_blocks = self.attention_blocks_needed(candidate.full_seq_len());
            let watermark = self.config.watermark_blocks(self.total_attention_blocks);
            if self.free_attention_blocks < needed_blocks + watermark {
                // Not enough headroom to admit without breaching the
                // watermark reserve. Leave it queued for a later step
                // rather than thrash by admitting anyway.
                break;
            }

            let mut req = self
                .waiting
                .pop_front()
                .expect("front() just returned Some");
            self.free_attention_blocks -= needed_blocks;
            req.blocks_reserved = needed_blocks;

            let remaining = req.prompt_tokens - req.computed_tokens;
            // Capped by the same slice the running sessions get, so a
            // newcomer's first chunk is one aligned pass rather than a grant
            // whose remainder the runtime has to spend 256 tokens at a time.
            let slice = match self.config.prefill_slice() {
                0 => u32::MAX,
                slice => slice,
            };
            let chunk = remaining.min(slice).min(budget);
            batch.prefills.push(PrefillItem {
                id: req.id,
                tokens: chunk,
            });
            req.computed_tokens += chunk;
            budget -= chunk;
            let truncated = chunk < remaining;
            self.running.push(req);
            if truncated {
                // One partially-prefilled newcomer per step. Admitting a
                // second would split the step between two prompts that both
                // then need several more, which is slower for both than
                // finishing one.
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A long prefill with nothing queued behind it must be unchanged.
    #[test]
    fn a_lone_long_prefill_still_takes_the_whole_step() {
        // The throughput case, and the reason the reserve is conditional. No
        // waiting request means no reserve, so the chunk is the full budget
        // exactly as before this existed.
        let mut s = sched(4096, 256, 3, 100_000);
        s.admit(NewRequest {
            id: RequestId(1),
            prompt_tokens: 60_000,
            max_output_tokens: 16,
        })
        .expect("admitted");
        let mut batch = BatchDescription::default();
        s.step_into(&mut batch); // first step admits it
        for _ in 0..3 {
            batch.prefills.clear();
            s.step_into(&mut batch);
            let chunk: u32 = batch.prefills.iter().map(|p| p.tokens).sum();
            assert_eq!(chunk, 4096, "a lone prefill must keep the entire budget");
        }
    }

    /// The regression: a newcomer must not wait out a 60K prompt.
    #[test]
    fn a_waiting_request_starts_while_a_long_prefill_is_still_running() {
        let mut s = sched(4096, 256, 3, 100_000);
        s.admit(NewRequest {
            id: RequestId(1),
            prompt_tokens: 60_000,
            max_output_tokens: 16,
        })
        .expect("admitted");
        let mut batch = BatchDescription::default();
        s.step_into(&mut batch); // the long prompt starts

        // Somebody queues behind it.
        s.admit(NewRequest {
            id: RequestId(2),
            prompt_tokens: 100,
            max_output_tokens: 16,
        })
        .expect("admitted");
        batch.prefills.clear();
        s.step_into(&mut batch);

        let served: Vec<RequestId> = batch.prefills.iter().map(|p| p.id).collect();
        assert!(
            served.contains(&RequestId(2)),
            "the newcomer must get tokens in the same step, not after 60K: {served:?}"
        );
        assert!(
            served.contains(&RequestId(1)),
            "the long prompt must keep progressing too"
        );
        let total: u32 = batch.prefills.iter().map(|p| p.tokens).sum();
        assert!(
            total <= 4096,
            "the step must not overspend its budget: {total}"
        );
    }

    #[test]
    fn a_shared_step_hands_out_retention_aligned_grants() {
        // Sharing must not become starvation in the other direction — and it
        // must not hand out a *misaligned* half either.
        //
        // This crate cannot see the retention interval; the runtime owns it.
        // But the two are coupled, because `choose_prefill_width` refuses a
        // pass that would straddle a snapshot boundary: a grant that is not a
        // whole multiple of the interval leaves a remainder the engine can
        // only spend through its prebuilt tail shapes, which stop at 256. At
        // the shipped defaults that turned a 1024 remainder into four 256-wide
        // passes on both sides of the split. So the property under test is
        // alignment, not a fraction.
        const RETENTION_INTERVAL: u32 = 2048;
        let mut s = sched(4096, 256, 3, 100_000);
        s.admit(NewRequest {
            id: RequestId(1),
            prompt_tokens: 60_000,
            max_output_tokens: 16,
        })
        .expect("admitted");
        let mut batch = BatchDescription::default();
        s.step_into(&mut batch);
        s.admit(NewRequest {
            id: RequestId(2),
            prompt_tokens: 60_000,
            max_output_tokens: 16,
        })
        .expect("admitted");
        batch.prefills.clear();
        s.step_into(&mut batch);
        let first = batch
            .prefills
            .iter()
            .find(|p| p.id == RequestId(1))
            .expect("still running");
        assert_eq!(
            first.tokens,
            4096 - 4096 / 2,
            "half the step stays with the prompt in flight"
        );
        for item in &batch.prefills {
            assert_eq!(
                item.tokens % RETENTION_INTERVAL,
                0,
                "grant for {:?} is {} — not a whole number of {RETENTION_INTERVAL}-token \
                 passes, so the runtime must spend the remainder 256 tokens at a time",
                item.id,
                item.tokens,
            );
        }
    }

    fn sched(
        token_budget: u32,
        block_size: u32,
        max_concurrent_decodes: u32,
        total_blocks: u32,
    ) -> Scheduler {
        let config =
            SchedulerConfig::new(token_budget, block_size, max_concurrent_decodes, 0.0, 0).unwrap();
        Scheduler::new(config, total_blocks)
    }

    fn req(id: u64, prompt: u32, max_out: u32) -> NewRequest {
        NewRequest {
            id: RequestId(id),
            prompt_tokens: prompt,
            max_output_tokens: max_out,
        }
    }

    #[test]
    fn cancellation_removes_waiting_and_running_requests_and_releases_capacity() {
        let mut scheduler = sched(4096, 256, 3, 100);
        scheduler.admit(req(1, 512, 256)).unwrap();
        scheduler.admit(req(2, 512, 256)).unwrap();

        assert!(scheduler.cancel_request(RequestId(2)));
        assert!(!scheduler.is_waiting(RequestId(2)));
        assert_eq!(scheduler.waiting_len(), 1);

        scheduler.step();
        assert!(scheduler.is_running(RequestId(1)));
        assert!(scheduler.free_attention_blocks() < scheduler.total_attention_blocks());
        assert!(scheduler.cancel_request(RequestId(1)));
        assert_eq!(
            scheduler.free_attention_blocks(),
            scheduler.total_attention_blocks()
        );
        assert!(!scheduler.cancel_request(RequestId(1)));
    }

    /// AGENTS.md rule 4, the named regression: chunked prefill splits
    /// compute, not memory. A request whose full lifetime exceeds capacity
    /// must be rejected at admission even though its first chunk (bounded
    /// by the step token budget) would easily fit.
    #[test]
    fn admission_rejects_full_sequence_exceeding_capacity_even_when_first_chunk_fits_regression() {
        // block_size 256, 4 total blocks -> 1024 tokens of total capacity.
        let mut s = sched(512, 256, 4, 4);
        // Prompt is 4000 tokens: far beyond the 1024-token pool, but its
        // first chunk (capped by the 512-token step budget) would fit
        // trivially if admission only looked at the first chunk.
        let err = s.admit(req(1, 4000, 0)).unwrap_err();
        match err {
            AdmissionError::ExceedsTotalCapacity {
                full_seq_len,
                needed_blocks,
                total_blocks,
            } => {
                assert_eq!(full_seq_len, 4000);
                assert_eq!(needed_blocks, 16); // ceil(4000/256)
                assert_eq!(total_blocks, 4);
            }
            other => panic!("expected ExceedsTotalCapacity, got {other:?}"),
        }
        assert_eq!(s.waiting_len(), 0, "rejected request must not be queued");
    }

    /// The full sequence length must include max_output_tokens, not just
    /// the prompt — otherwise a short prompt with a long generation budget
    /// would slip past the same check this rule requires.
    #[test]
    fn admission_capacity_check_counts_max_output_tokens_too() {
        let mut s = sched(512, 256, 4, 4); // 1024 tokens total capacity
        // Prompt alone (100 tokens) fits easily, but + max_output (10000)
        // does not.
        let err = s.admit(req(1, 100, 10_000)).unwrap_err();
        assert!(matches!(err, AdmissionError::ExceedsTotalCapacity { .. }));
    }

    #[test]
    fn sequence_length_overflow_saturates_and_is_rejected() {
        let mut s = sched(512, 256, 4, 4);
        let request = req(1, u32::MAX - 10, 100);
        assert_eq!(request.full_seq_len(), u32::MAX);
        assert!(matches!(
            s.admit(request),
            Err(AdmissionError::ExceedsTotalCapacity { .. })
        ));
    }

    #[test]
    fn admission_accepts_a_request_that_fits() {
        let mut s = sched(512, 256, 4, 4);
        assert!(s.admit(req(1, 500, 100)).is_ok());
        assert_eq!(s.waiting_len(), 1);
    }

    #[test]
    fn decode_requests_are_always_scheduled_before_prefill() {
        // Enough budget for one decode step (1 token here, no drafts) plus
        // some prefill.
        let mut s = sched(300, 256, 2, 8);
        s.admit(req(1, 200, 50)).unwrap();
        let batch1 = s.step();
        assert_eq!(batch1.decodes.len(), 0, "nothing running yet");
        assert_eq!(batch1.prefills.len(), 1);
        assert_eq!(batch1.prefills[0].tokens, 200, "prompt fits in one chunk");

        // Now request 1 is decode-ready (computed == prompt_tokens).
        s.admit(req(2, 200, 50)).unwrap();
        let batch2 = s.step();
        assert_eq!(batch2.decodes.len(), 1, "request 1 should now decode");
        assert_eq!(batch2.decodes[0].id, RequestId(1));
        assert!(
            batch2.has_prefill(),
            "leftover budget should prefill request 2"
        );
    }

    #[test]
    fn waiting_requests_never_overfill_the_workers_decode_slots() {
        let mut s = sched(4096, 256, 3, 64);
        for id in 1..=4 {
            s.admit(req(id, 32, 8)).unwrap();
        }
        let first = s.step();
        assert_eq!(first.prefills.len(), 3);
        assert_eq!(s.running_len(), 3);
        assert_eq!(s.waiting_len(), 1);

        let second = s.step();
        assert_eq!(second.decodes.len(), 3);
        assert!(second.prefills.is_empty());
        assert_eq!(s.running_len(), 3);
        assert_eq!(s.waiting_len(), 1);

        assert!(s.finish_request(RequestId(1)));
        let third = s.step();
        assert_eq!(third.prefills.len(), 1);
        assert_eq!(third.prefills[0].id, RequestId(4));
        assert_eq!(s.running_len(), 3);
        assert_eq!(s.waiting_len(), 0);
    }

    #[test]
    fn bounded_waiting_queue_rejects_before_host_growth() {
        let config = SchedulerConfig::new(4096, 256, 1, 0.01, 0).unwrap();
        let mut scheduler = Scheduler::new(config, 128);
        for id in 1..=config.max_waiting_requests() {
            scheduler.admit(req(u64::from(id), 32, 8)).unwrap();
        }
        assert!(!scheduler.can_admit(&req(999, 32, 8)));
        let error = scheduler.admit(req(999, 32, 8)).unwrap_err();
        assert_eq!(
            error,
            AdmissionError::WaitingQueueFull {
                capacity: config.max_waiting_requests()
            }
        );
        assert_eq!(
            scheduler.waiting_len() as u32,
            config.max_waiting_requests()
        );
        assert!(matches!(
            scheduler.admit_with_prefix(req(1000, 32, 8), 33),
            Err(AdmissionError::InvalidReusablePrefix { .. })
        ));
    }

    #[test]
    fn step_into_reuses_reserved_batch_storage() {
        let mut s = sched(4096, 256, 3, 64);
        for id in 1..=3 {
            s.admit(req(id, 32, 8)).unwrap();
        }
        let mut batch = BatchDescription::with_capacity(3, 3);
        let decode_ptr = batch.decodes.as_ptr();
        let prefill_ptr = batch.prefills.as_ptr();
        s.step_into(&mut batch);
        assert_eq!(batch.prefills.len(), 3);
        assert_eq!(batch.decodes.as_ptr(), decode_ptr);
        assert_eq!(batch.prefills.as_ptr(), prefill_ptr);
        s.step_into(&mut batch);
        assert_eq!(batch.decodes.len(), 3);
        assert_eq!(batch.decodes.as_ptr(), decode_ptr);
        assert_eq!(batch.prefills.as_ptr(), prefill_ptr);
    }

    #[test]
    fn a_prompt_longer_than_the_budget_is_chunked_and_continues_next_step() {
        let mut s = sched(300, 256, 1, 8);
        s.admit(req(1, 1000, 0)).unwrap();

        let batch1 = s.step();
        assert_eq!(batch1.prefills.len(), 1);
        assert_eq!(batch1.prefills[0].tokens, 300, "chunked to the step budget");
        assert!(s.is_running(RequestId(1)));

        let batch2 = s.step();
        assert_eq!(batch2.prefills.len(), 1);
        assert_eq!(batch2.prefills[0].tokens, 300);

        let batch3 = s.step();
        assert_eq!(batch3.prefills[0].tokens, 300);

        let batch4 = s.step();
        // 1000 - 300*3 = 100 tokens left.
        assert_eq!(batch4.prefills[0].tokens, 100);
    }

    #[test]
    fn a_chunked_request_blocks_further_admission_in_the_same_step() {
        let mut s = sched(300, 256, 1, 8);
        s.admit(req(1, 1000, 0)).unwrap(); // will be chunked
        s.admit(req(2, 50, 0)).unwrap(); // would otherwise fit in leftover budget

        let batch = s.step();
        assert_eq!(
            batch.prefills.len(),
            1,
            "chunking the first request stops further scheduling"
        );
        assert!(s.is_waiting(RequestId(2)));
    }

    #[test]
    fn watermark_blocks_admission_until_capacity_frees_up() {
        // block_size 256, 4 total blocks. Watermark 25% -> 1 block reserved.
        let config = SchedulerConfig::new(1024, 256, 4, 0.25, 0).unwrap();
        let mut s = Scheduler::new(config, 4);

        // Request A needs ceil(768/256) = 3 blocks. Free=4, watermark=1:
        // 4 - 3 = 1 >= 1, so it's admitted.
        s.admit(req(1, 768, 0)).unwrap();
        let batch1 = s.step();
        assert_eq!(batch1.prefills.len(), 1);
        assert_eq!(s.free_attention_blocks(), 1);

        // Request B needs 1 block. Free=1, watermark=1: 1 - 1 = 0 < 1, so
        // it must NOT be admitted this step.
        s.admit(req(2, 100, 0)).unwrap();
        let batch2 = s.step();
        assert!(
            batch2.prefills.iter().all(|p| p.id != RequestId(2)),
            "watermark should block admission of request 2 while headroom is insufficient"
        );
        assert!(s.is_waiting(RequestId(2)));

        // Once request A finishes and frees its blocks, request B can be
        // admitted.
        s.finish_request(RequestId(1));
        assert_eq!(s.free_attention_blocks(), 4);
        let batch3 = s.step();
        assert!(batch3.prefills.iter().any(|p| p.id == RequestId(2)));
    }

    /// A queue that cannot be admitted from must not cost the step anything.
    #[test]
    fn a_full_worker_does_not_reserve_for_an_admission_it_would_refuse() {
        // Every slot busy and a further request queued behind them. Phase 3
        // cannot take it — `running.len() < max_concurrent_decodes` is
        // already false — so holding a slice back for it is budget nothing
        // in the step can spend, and this is exactly when the queue is
        // longest.
        let mut s = sched(4096, 256, 3, 100_000);
        for id in 1..=3 {
            s.admit(NewRequest {
                id: RequestId(id),
                prompt_tokens: 60_000,
                max_output_tokens: 16,
            })
            .expect("admitted");
        }
        let mut batch = BatchDescription::default();
        // Three steps: one admission each, since chunking the first request
        // that does not fit ends the step.
        for _ in 0..3 {
            batch.prefills.clear();
            s.step_into(&mut batch);
        }
        assert_eq!(s.running_len(), 3, "all three slots should be occupied");

        s.admit(NewRequest {
            id: RequestId(4),
            prompt_tokens: 60_000,
            max_output_tokens: 16,
        })
        .expect("admitted");
        assert!(s.is_waiting(RequestId(4)));

        batch.prefills.clear();
        s.step_into(&mut batch);
        let spent: u32 = batch.prefills.iter().map(|p| p.tokens).sum();
        assert_eq!(
            spent, 4096,
            "the step must spend its whole budget when nobody can be admitted, \
             not hold a slice back for a request phase 3 will refuse"
        );
        assert!(
            batch.prefills.iter().all(|p| p.id != RequestId(4)),
            "and the queued request is still not admitted"
        );
    }

    /// The same, when it is the watermark rather than the slots that refuses.
    #[test]
    fn a_candidate_the_watermark_refuses_does_not_earn_a_reserve() {
        // block_size 256, 40 total blocks, watermark 25% -> 10 held back.
        // A's 6000-token prompt needs ceil(6000/256) = 24 blocks and fits
        // (40 >= 24 + 10), leaving 16 free. B then needs 8, and 16 < 8 + 10,
        // so the watermark refuses it however long it waits. Decode slots are
        // deliberately plentiful so that the watermark is the only thing
        // saying no — this is the case slot-counting alone would miss.
        //
        // A's prompt spans six steps at a 1024-token budget, so there is a
        // mid-prefill step to observe. Its chunk must stay the whole budget:
        // gated on a non-empty queue alone this was 1024 - 1024/2 = 512.
        let config = SchedulerConfig::new(1024, 256, 4, 0.25, 0).unwrap();
        let mut s = Scheduler::new(config, 40);
        s.admit(req(1, 6000, 0)).unwrap();
        s.step();
        s.admit(req(2, 2000, 0)).unwrap();
        let batch = s.step();
        assert!(s.is_waiting(RequestId(2)), "the watermark still refuses it");
        let spent: u32 = batch.prefills.iter().map(|p| p.tokens).sum();
        assert_eq!(
            spent, 1024,
            "a request the watermark will refuse must not reserve budget"
        );
    }

    /// The point of the whole engine: sessions admitted together must run
    /// together.
    #[test]
    fn several_prefilling_sessions_share_a_step_and_take_turns() {
        // Three long prompts, a 4096 budget and a 2048 slice: two sessions
        // fit in a step, so the third is shorted — but a *different* third
        // each step, which is what the rotating cursor is for.
        let config = SchedulerConfig::new(4096, 256, 3, 0.0, 0)
            .unwrap()
            .with_prefill_slice(2048);
        let mut s = Scheduler::new(config, 100_000);
        for id in 1..=3 {
            s.admit(NewRequest {
                id: RequestId(id),
                prompt_tokens: 60_000,
                max_output_tokens: 16,
            })
            .expect("admitted");
        }
        let mut batch = BatchDescription::default();
        // One truncated admission per step, so three steps to get them all
        // running.
        for _ in 0..3 {
            batch.prefills.clear();
            s.step_into(&mut batch);
        }
        assert_eq!(s.running_len(), 3, "all three sessions running");

        let mut served: [u32; 4] = [0; 4];
        for _ in 0..3 {
            batch.prefills.clear();
            s.step_into(&mut batch);
            assert!(
                batch.prefills.len() > 1,
                "a step must be shared, not handed to one session: {:?}",
                batch.prefills
            );
            for item in &batch.prefills {
                assert!(
                    item.tokens <= 2048,
                    "no session may exceed its slice: {item:?}"
                );
                served[item.id.0 as usize] += item.tokens;
            }
        }
        for id in 1..=3 {
            assert!(
                served[id] > 0,
                "session {id} got nothing across three steps: {served:?}"
            );
        }
    }

    /// The regression this replaced: one long prompt must not own every step.
    #[test]
    fn a_long_prompt_no_longer_starves_the_sessions_beside_it() {
        let config = SchedulerConfig::new(4096, 256, 3, 0.0, 0)
            .unwrap()
            .with_prefill_slice(2048);
        let mut s = Scheduler::new(config, 100_000);
        s.admit(NewRequest {
            id: RequestId(1),
            prompt_tokens: 200_000,
            max_output_tokens: 16,
        })
        .expect("admitted");
        let mut batch = BatchDescription::default();
        s.step_into(&mut batch);
        s.admit(NewRequest {
            id: RequestId(2),
            prompt_tokens: 60_000,
            max_output_tokens: 16,
        })
        .expect("admitted");

        // Step once to admit the newcomer. Admission alone proves nothing —
        // the old serial loop admitted it too, then never served it again —
        // so the assertion is about the steps *after* it is running.
        batch.prefills.clear();
        s.step_into(&mut batch);
        assert!(s.is_running(RequestId(2)), "newcomer admitted");

        let mut served_while_running = 0;
        for _ in 0..4 {
            batch.prefills.clear();
            s.step_into(&mut batch);
            if batch.prefills.iter().any(|p| p.id == RequestId(2)) {
                served_while_running += 1;
            }
        }
        assert_eq!(
            served_while_running, 4,
            "an admitted session must keep progressing while the 200K prompt \
             beside it is still prefilling, not merely be admitted and parked"
        );
    }

    #[test]
    fn preemption_resets_progress_and_frees_blocks_then_requeues_at_the_front() {
        let mut s = sched(1024, 256, 4, 8);
        s.admit(req(1, 200, 0)).unwrap();
        s.admit(req(2, 200, 0)).unwrap();
        s.step(); // both admitted and fully prefilled (small prompts, big budget)

        assert!(s.is_running(RequestId(1)));
        let free_before = s.free_attention_blocks();

        assert!(s.preempt(RequestId(1)));
        assert!(!s.is_running(RequestId(1)));
        assert!(s.is_waiting(RequestId(1)));
        assert!(
            s.free_attention_blocks() > free_before,
            "preemption must free the preempted request's reserved blocks"
        );

        // Re-admitted request 1 must recompute from scratch.
        let batch = s.step();
        let re_admitted = batch
            .prefills
            .iter()
            .find(|p| p.id == RequestId(1))
            .expect("request 1 should be re-admitted first (front of queue)");
        assert_eq!(
            re_admitted.tokens, 200,
            "recompute starts from token 0 again"
        );
    }

    /// Speculative decode draft tokens (whether MTP or n-gram) must enter the per-step budget
    /// calculation, sized for the drafted case rather than the accepted
    /// case, or admission oscillates as the acceptance rate varies.
    #[test]
    fn draft_tokens_are_charged_against_the_step_budget_for_every_decode() {
        let config = SchedulerConfig::new(20, 4, 3, 0.0, 3).unwrap(); // 1 + 3 draft = 4 tokens/decode
        let mut s = Scheduler::new(config, 100);
        s.admit(req(1, 4, 10)).unwrap();
        s.admit(req(2, 4, 10)).unwrap();
        s.step(); // both prefill fully (4 tokens each, budget 20)

        // Now both are decode-ready: 2 requests * 4 tokens/decode = 8
        // tokens of the 20-token budget, leaving 12 for prefill.
        s.admit(req(3, 12, 0)).unwrap();
        let batch = s.step();
        assert_eq!(batch.decodes.len(), 2);
        assert!(batch.decodes.iter().all(|d| d.tokens == 4));
        let prefill_tokens: u32 = batch.prefills.iter().map(|p| p.tokens).sum();
        assert_eq!(
            prefill_tokens, 12,
            "exactly the budget left after both decodes"
        );
    }

    /// The verify pass writes a full draft window of K/V before acceptance
    /// truncates anything, so a drafting scheduler must reserve that window
    /// as scratch beyond `full_seq_len` — a sequence one emission short of
    /// its cap still verifies a full window.
    #[test]
    fn draft_scratch_is_reserved_beyond_the_full_sequence_length() {
        // block_size 256, 4 total blocks. A 1024-token request exactly fills
        // capacity with drafting off...
        let plain = SchedulerConfig::new(2048, 256, 1, 0.0, 0).unwrap();
        let mut s = Scheduler::new(plain, 4);
        s.admit(req(1, 512, 512)).unwrap();
        assert!(s.step().prefills.iter().any(|p| p.id == RequestId(1)));
        assert_eq!(s.free_attention_blocks(), 0);

        // ...but with 3 draft tokens per step, the same request needs a 5th
        // block for the verify window's scratch rows and must be rejected.
        let drafting = SchedulerConfig::new(2048, 256, 1, 0.0, 3).unwrap();
        let mut s = Scheduler::new(drafting, 4);
        assert!(!s.can_admit(&req(2, 512, 512)));
        assert!(matches!(
            s.admit(req(2, 512, 512)),
            Err(AdmissionError::ExceedsTotalCapacity { .. })
        ));

        // A request whose scratch fits inside its last block's padding is
        // unaffected: 1021 + 3 still rounds to 4 blocks.
        s.admit(req(3, 512, 509)).unwrap();
        assert!(s.step().prefills.iter().any(|p| p.id == RequestId(3)));
        assert_eq!(s.free_attention_blocks(), 0);
    }

    #[test]
    fn a_request_with_no_output_budget_is_not_scheduled_for_decode() {
        let mut s = sched(300, 256, 1, 8);
        s.admit(req(1, 10, 0)).unwrap();
        s.step();
        assert!(s.step().decodes.is_empty());
    }

    #[test]
    fn accepted_speculative_tokens_advance_without_exceeding_the_output_limit() {
        let mut s = sched(300, 256, 1, 8);
        s.admit(req(1, 10, 3)).unwrap();
        s.step();
        assert_eq!(s.step().decodes.len(), 1);
        assert!(s.advance_speculative(RequestId(1), 99));
        assert!(s.step().decodes.is_empty());
    }

    #[test]
    fn restored_prefix_skips_only_the_computed_prompt_tokens() {
        let mut s = sched(300, 256, 1, 8);
        s.admit_with_prefix(req(1, 600, 10), 512).unwrap();
        let batch = s.step();
        assert_eq!(batch.prefills[0].tokens, 88);
        assert!(s.step().decodes.len() == 1);
    }

    #[test]
    fn restored_prefix_cannot_extend_beyond_the_prompt() {
        let mut s = sched(300, 256, 1, 8);
        assert!(matches!(
            s.admit_with_prefix(req(1, 10, 1), 11),
            Err(AdmissionError::InvalidReusablePrefix { .. })
        ));
    }
}
