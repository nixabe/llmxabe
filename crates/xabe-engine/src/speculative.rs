//! MTP speculative decode — the draft/verify/accept/rollback driver.
//!
//! `docs/OPTIMIZATION.md` §R6. This wires the three pieces built for it
//! together: [`crate::block::mtp::MtpBlock`] drafts, [`crate::forward::
//! Forward::run_verify`] verifies `1 + d` positions in one weight-read pass,
//! and [`crate::block::gdn_verify`]'s window-local snapshot ring commits the
//! Gated DeltaNet state to exactly the accepted prefix at zero extra
//! weight-read cost.
//!
//! # Correctness, not just speed
//!
//! Under greedy decoding, a drafted token is accepted only if it equals the
//! target model's own argmax at that position, and a step that rejects a
//! draft early still emits the target's own argmax as a "bonus" token in its
//! place. So the emitted sequence is **exactly** what plain, non-speculative
//! greedy decode over the same prompt would emit — this file changes only
//! how many weight reads it costs to get there, never what comes out.
//! `tests/speculative_identity.rs` is the test that holds this file to that.
//!
//! # The two hidden-state chains
//!
//! Two different `h` chains feed the draft head, and mixing them up drafts
//! from the wrong context without ever erroring — see `crate::block::mtp`'s
//! module docs:
//!
//! - **Prompt catch-up** ([`SpeculativeSession::new`]): the draft head's own
//!   key/value cache must be filled over the whole prompt before it can
//!   draft anything, using the *target's* hidden state at each position,
//!   shifted right by one (position `p`'s input pairs token `p` with the
//!   target's hidden state from position `p - 1`, zero for `p = 0`) — this
//!   is what makes an untouched draft cache a silent quality bug rather than
//!   an error.
//! - **The draft chain** ([`Self::draft`]): every position after the first
//!   in a window feeds back the draft head's *own* previous emitted hidden
//!   state, because the target has not run that position yet.
//!
//! # What "accepted" commits, and what stays free
//!
//! A verify step's window is `[id_last, draft_1, .., draft_d]`. Row `i` of
//! [`crate::forward::Forward::run_verify`]'s output is the target's argmax
//! *after* `window[i]`; comparing rows `0..d` against `window[1..=d]` finds
//! the accepted prefix `k` (`0 <= k <= d`) and row `k` is always a valid
//! "bonus" token — the target's own choice, never a draft. `k + 1`
//! positions of the window (`id_last` plus `k` accepted drafts) are now
//! real: [`GdnSnapshotRing::commit`] at slot `k + 1` on every layer, and
//! [`SequenceState`]'s position advances by `k + 1`. The attention
//! key/value caches and the draft head's own cache need no explicit
//! truncation — see `crate::state`'s module docs on why a reset does not
//! clear them; the same reasoning is why simply not advancing past `k + 1`
//! positions is already correct.

use std::sync::Arc;

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaContext, CudaEvent, CudaSlice, CudaStream, DriverError};

use xabe_gguf::GgufFile;
use xabe_model::config::ModelConfig;
use xabe_model::weights::Directory;

use crate::block::attention::{AttentionBlockError, KvCache};
use crate::block::gdn::GdnGeometry;
use crate::block::gdn_verify::GdnSnapshotRing;
use crate::block::mtp::{MtpBlock, MtpBlockError};
use crate::forward::{Forward, ForwardError};
use crate::state::{SequenceState, StateError};
use crate::weights::DeviceWeights;

/// Something went wrong building or running a speculative session.
#[derive(Debug)]
pub enum SpeculativeError {
    Forward(ForwardError),
    Mtp(MtpBlockError),
    Attention(AttentionBlockError),
    State(StateError),
    Driver(DriverError),
    /// The prompt was empty — there is no `id_last` to draft from.
    EmptyPrompt,
}

impl std::fmt::Display for SpeculativeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Forward(e) => write!(f, "{e}"),
            Self::Mtp(e) => write!(f, "{e}"),
            Self::Attention(e) => write!(f, "{e}"),
            Self::State(e) => write!(f, "{e}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::EmptyPrompt => {
                write!(f, "cannot start a speculative session from an empty prompt")
            }
        }
    }
}

impl std::error::Error for SpeculativeError {}

macro_rules! from_error {
    ($src:ty, $variant:ident) => {
        impl From<$src> for SpeculativeError {
            fn from(e: $src) -> Self {
                Self::$variant(e)
            }
        }
    };
}
from_error!(ForwardError, Forward);
from_error!(MtpBlockError, Mtp);
from_error!(AttentionBlockError, Attention);
from_error!(StateError, State);
from_error!(DriverError, Driver);

/// GPU-timeline split of one [`SpeculativeSession::step`], from CUDA events
/// recorded on the same stream as the work. Host `Instant` around a step
/// measures enqueue plus the DtoH syncs that already sit inside draft and
/// verify (both return sampled ids); these numbers are the GPU spans
/// between those markers, which is what `AGENTS.md` wants for anything
/// inside a pass.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepTimes {
    /// Three sequential one-token block-40 forwards, plus the host-to-device
    /// position writes that seed each of them.
    pub draft_ms: f64,
    /// One `1 + d`-token target pass, including GDN snapshot DtoD and the
    /// per-row argmax that `run_verify` returns.
    pub verify_ms: f64,
    /// Per-layer snapshot restore plus the seed-`h` copy for the next draft.
    pub commit_ms: f64,
}

impl StepTimes {
    /// Sum of the three spans. Compare against host wall to see launch tax.
    pub fn gpu_ms(self) -> f64 {
        self.draft_ms + self.verify_ms + self.commit_ms
    }
}

/// One draft-verify-accept-commit round's outcome, for the caller to record
/// acceptance statistics with (see `docs/BENCHMARKS.md`'s honest-throughput
/// discipline: "accepted tokens per wall-clock second", not drafted).
pub struct StepOutcome {
    /// Newly emitted tokens this step, oldest first: `k` accepted drafts
    /// plus the bonus token, `1 <= len() <= d + 1`.
    pub emitted: Vec<i32>,
    /// How many of the `d` drafted tokens were accepted.
    pub accepted: usize,
    /// How many tokens were drafted (`d`).
    pub drafted: usize,
    /// GPU-timeline split of this step. Always populated; the events live
    /// on the session and are reused (`AGENTS.md` rule 6).
    pub times: StepTimes,
}

/// A single speculatively-decoding sequence: the target model, its state,
/// the draft head, and the per-layer snapshot rings that make partial
/// acceptance exact.
pub struct SpeculativeSession {
    verify: Forward,
    target_state: SequenceState,
    rings: Vec<GdnSnapshotRing>,

    draft: MtpBlock,
    draft_cache: KvCache,
    /// Positions the draft head's own KV cache has committed to — distinct
    /// from the target's position because a rejected draft tail leaves the
    /// draft cache ahead until the next step's draft call overwrites it; see
    /// the module docs.
    draft_pos: usize,
    /// Reused across every draft-head and target call: one absolute
    /// position, refreshed by an host-to-device copy before each launch
    /// rather than reallocated (`AGENTS.md` rule 6).
    positions: CudaSlice<i32>,
    /// The next draft call's seed hidden state: the target's `h_nextn` at
    /// the position that produced the current `id_last`. Reused in place —
    /// [`Self::draft`] overwrites it with the draft head's own emitted `h`
    /// after the first chained position.
    pending_h: CudaSlice<f32>,

    d: usize,
    hidden: usize,
    id_last: i32,

    /// Four reused stream markers: start, after draft, after verify, after
    /// commit. Created once in [`Self::new`] — recording on the hot path
    /// is an enqueue, not an allocation.
    ev_start: CudaEvent,
    ev_draft: CudaEvent,
    ev_verify: CudaEvent,
    ev_commit: CudaEvent,
}

impl SpeculativeSession {
    /// Prefill `prompt_ids` through the target model, catch the draft head's
    /// own key/value cache up over the same prompt, and return the session
    /// ready to draft — plus the one token prefill itself sampled
    /// (`id_last`), which the caller must record as the first emitted token.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        file: &GgufFile,
        directory: &Directory<'_>,
        weights: &DeviceWeights,
        config: &ModelConfig,
        rms_eps: f32,
        rope_theta: f32,
        prompt_ids: &[i32],
        max_seq: usize,
        d: usize,
    ) -> Result<(Self, i32), SpeculativeError> {
        let hidden = config.hidden_size as usize;
        if prompt_ids.is_empty() {
            return Err(SpeculativeError::EmptyPrompt);
        }
        let prompt_len = prompt_ids.len();

        // --- target: prefill, then reshape to the verify window ---------
        let mut prefill = Forward::new(
            ctx,
            stream,
            file,
            directory,
            weights,
            config.clone(),
            prompt_len,
        )?;
        let mut target_state = prefill.new_state(stream, max_seq)?;
        prefill.run(stream, &mut target_state, prompt_ids, |_, _| {})?;
        let id_last = prefill.sample_argmax(stream)?;

        // The seed `h` for the first draft call: the target's own hidden
        // state at the position that produced `id_last` — the prompt's last.
        let mut pending_h = stream.alloc_zeros::<f32>(hidden)?;
        {
            let last = (prompt_len - 1) * hidden;
            let row = prefill.final_norm().slice(last..last + hidden);
            stream.memcpy_dtod(&row, &mut pending_h)?;
        }

        let tokens = 1 + d;
        let mut verify = prefill.reshape(ctx, stream, file, directory, weights, tokens)?;
        verify.enable_batch_decode(ctx, stream)?;
        verify.enable_verify(stream)?;

        let gdn_layers = config.num_gdn_layers() as usize;
        let gdn_geometry = GdnGeometry::from_config(config, tokens, rms_eps);
        let mut rings = Vec::with_capacity(gdn_layers);
        for _ in 0..gdn_layers {
            rings.push(GdnSnapshotRing::new(stream, &gdn_geometry, tokens + 1)?);
        }

        // --- draft head: catch its own KV up over the whole prompt -------
        let mut catchup = MtpBlock::new(
            ctx, stream, file, directory, weights, config, prompt_len, rms_eps, rope_theta, false,
        )?;
        let mut draft_cache = KvCache::new(stream, config, max_seq)?;
        let mut positions = stream.alloc_zeros::<i32>(1)?;
        stream.memcpy_htod(&[0i32], &mut positions)?;

        // Shift the target's own hidden states right by one: position `p`'s
        // input pairs `prompt_ids[p]` with `final_norm[p - 1]` (zero at
        // `p = 0`) — see the module docs.
        let mut catchup_h = stream.alloc_zeros::<f32>(prompt_len * hidden)?;
        if prompt_len > 1 {
            let src = prefill.final_norm().slice(0..(prompt_len - 1) * hidden);
            let mut dst = catchup_h.slice_mut(hidden..prompt_len * hidden);
            stream.memcpy_dtod(&src, &mut dst)?;
        }
        catchup.forward(
            stream,
            prompt_ids,
            &catchup_h,
            &mut draft_cache,
            0,
            &positions,
        )?;

        let draft = catchup.reshape(ctx, stream, weights, config, 1, rms_eps, rope_theta, true)?;

        // Timing-enabled events: `new_event(None)` would set
        // `CU_EVENT_DISABLE_TIMING` and `elapsed_ms` would then fail. Same
        // flag `StageProfile` uses in `forward.rs`.
        let ev_start = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        let ev_draft = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        let ev_verify = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        let ev_commit = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;

        Ok((
            Self {
                verify,
                target_state,
                rings,
                draft,
                draft_cache,
                draft_pos: prompt_len,
                positions,
                pending_h,
                d,
                hidden,
                id_last,
                ev_start,
                ev_draft,
                ev_verify,
                ev_commit,
            },
            id_last,
        ))
    }

    /// Draft `d` tokens, greedily, chaining the draft head's own emitted
    /// hidden state after the first position (see the module docs). The
    /// draft's own sampled ids are never trusted directly — verification
    /// recomputes everything — this just proposes them.
    fn draft(&mut self, stream: &Arc<CudaStream>) -> Result<Vec<i32>, SpeculativeError> {
        let mut ids = Vec::with_capacity(self.d);
        let mut tok = self.id_last;
        for i in 0..self.d {
            stream.memcpy_htod(&[(self.draft_pos + i) as i32], &mut self.positions)?;
            let sampled = self.draft.forward(
                stream,
                &[tok],
                &self.pending_h,
                &mut self.draft_cache,
                self.draft_pos + i,
                &self.positions,
            )?;
            let id = sampled.expect("draft instance built with a LM head")[0];
            // Chain: the next position's seed `h` is this call's own
            // emitted `h_nextn`, not the target's (which has not run this
            // position yet).
            stream.memcpy_dtod(self.draft.h_nextn(), &mut self.pending_h)?;
            ids.push(id);
            tok = id;
        }
        self.draft_pos += self.d;
        Ok(ids)
    }

    /// One draft-verify-accept-commit round. Returns the newly emitted
    /// tokens (`1..=d+1` of them) and advances the session so the next call
    /// continues from where this one left off.
    pub fn step(&mut self, stream: &Arc<CudaStream>) -> Result<StepOutcome, SpeculativeError> {
        self.ev_start.record(stream)?;
        let drafts = self.draft(stream)?;
        self.ev_draft.record(stream)?;

        let mut window = Vec::with_capacity(1 + self.d);
        window.push(self.id_last);
        window.extend_from_slice(&drafts);

        let predicted =
            self.verify
                .run_verify(stream, &mut self.target_state, &mut self.rings, &window)?;
        self.ev_verify.record(stream)?;

        let mut accepted = 0usize;
        while accepted < self.d && predicted[accepted] == drafts[accepted] {
            accepted += 1;
        }
        let bonus = predicted[accepted];

        for (slot, ring) in self.rings.iter().enumerate() {
            ring.commit(stream, accepted + 1, self.target_state.gdn_mut(slot))?;
        }
        self.target_state.advance(accepted + 1);

        // The next step's seed `h`: the target's own hidden state at the
        // window row that produced `bonus` — see the module docs.
        {
            let last = accepted * self.hidden;
            let row = self.verify.final_norm().slice(last..last + self.hidden);
            stream.memcpy_dtod(&row, &mut self.pending_h)?;
        }
        self.ev_commit.record(stream)?;

        // The draft head's own cache is left wherever drafting advanced it
        // (`draft_pos + d`); a rejected tail's entries are simply
        // unreachable until the next `draft()` call overwrites them
        // starting at the real `draft_pos` — see the module docs. Roll
        // `draft_pos` back to match what was actually accepted so the next
        // call's positions are correct.
        self.draft_pos = self.draft_pos - self.d + accepted + 1;

        self.id_last = bonus;

        // `elapsed_ms` synchronizes on the later event, so it is safe to
        // read immediately: `run_verify` already DtoH'd the argmax ids, and
        // the commit copies are stream-ordered before `ev_commit`.
        let times = StepTimes {
            draft_ms: self.ev_start.elapsed_ms(&self.ev_draft)? as f64,
            verify_ms: self.ev_draft.elapsed_ms(&self.ev_verify)? as f64,
            commit_ms: self.ev_verify.elapsed_ms(&self.ev_commit)? as f64,
        };

        let mut emitted = drafts[..accepted].to_vec();
        emitted.push(bonus);
        Ok(StepOutcome {
            emitted,
            accepted,
            drafted: self.d,
            times,
        })
    }
}
