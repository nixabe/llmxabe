//! A single worker: one GPU, one cache pool, one scheduler.
//!
//! Workers are symmetric. Each holds a complete copy of the model, so any
//! request can be served by any worker and the answer is identical. What
//! differs between them is only cost — what they already have cached and how
//! loaded they are — which is what [`crate::router`] scores.
//!
//! Workers never touch each other's device memory. No P2P, no NCCL, nothing
//! crosses PCIe on the decode path. What makes them one engine is the shared
//! prefix tree above them, not any device-level coupling.
//!
//! A worker is initially host-only so routing and admission tests need no GPU.
//! [`Worker::bind_device`] installs the CUDA half: context, stream, resident
//! weights, fixed serving shapes and per-request sequence states.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use xabe_cache::config::{CacheConfig, CapacityReport};
use xabe_cache::error::PoolExhausted;
use xabe_cache::pool::{BlockId, BlockPool};
use xabe_sched::config::SchedulerConfig;
use xabe_sched::error::AdmissionError;
use xabe_sched::ngram::NgramConfig;
use xabe_sched::ngram_map::{NgramMapConfig, NgramSimpleConfig};
use xabe_sched::ngram_mod::NgramModConfig;
use xabe_sched::spec::SelfSpecConfig;

use xabe_model::ModelConfig;
use xabe_sched::request::{BatchDescription, NewRequest, RequestId};
use xabe_sched::scheduler::Scheduler;

use crate::router::WorkerLoad;
use crate::runtime::{DeviceRuntimeHandle, DeviceStep, RuntimeConfig, RuntimeError};
use crate::sampling::SamplingParams;
use crate::state::{SequenceSnapshot, SnapshotSlots};

/// Which speculative decoder a worker runs.
///
/// The draft *count* is deliberately absent: it comes from the scheduler,
/// which has to charge those tokens against its step budget whether or not
/// they are later accepted. Carrying it here too would let the two disagree.
///
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Speculation {
    /// One token per decode step, drafted by nothing.
    None,
    /// Suffix lookup over the sequence's own prompt and output.
    Ngram {
        /// Shortest suffix worth matching on.
        min: usize,
        /// Longest suffix matched before giving up.
        max: usize,
    },
    /// llama.cpp's `ngram-simple`: backward scan for the newest earlier
    /// occurrence of a fixed-length tail. The draft length (its `size_m`)
    /// is the scheduler's draft count, per the rule above.
    NgramSimple {
        /// Lookup n-gram length (llama.cpp `--spec-ngram-simple-size-n`).
        size_n: usize,
    },
    /// llama.cpp's `ngram-mod`: a hash table from n-gram to next token,
    /// shared by every sequence this worker serves. Its `n_max` is the
    /// scheduler's draft count.
    NgramMod {
        /// Lookup n-gram length (llama.cpp `--spec-ngram-mod-n-match`).
        n_match: usize,
        /// Drop any draft shorter than this (`--spec-ngram-mod-n-min`).
        n_min: usize,
    },
    /// llama.cpp's `ngram-map-k`: per-sequence key-n-gram map, drafting
    /// straight from the newest key match. Its `size_m` is the scheduler's
    /// draft count.
    NgramMapK {
        /// Key n-gram length (llama.cpp `--spec-ngram-map-k-size-n`).
        size_n: usize,
        /// Minimum key hits kept for llama.cpp parity (`--spec-ngram-map-k-min-hits`);
        /// the key-only draft path does not consult it, as upstream's does not.
        min_hits: u16,
    },
    /// llama.cpp's `ngram-map-k4v`: as `ngram-map-k`, but tracking up to
    /// four continuations per key and drafting only a dominant one.
    NgramMapK4v {
        /// Key n-gram length (llama.cpp `--spec-ngram-map-k4v-size-n`).
        size_n: usize,
        /// Minimum key hits before drafting (`--spec-ngram-map-k4v-min-hits`).
        min_hits: u16,
    },
    /// The model's own trained next-token-prediction head (GGUF block 40)
    /// drafts; the same batched verify pass the n-gram path uses accepts.
    /// Costs one extra layer's weights on the device and one draft-head
    /// KV cache per resident sequence.
    Mtp,
    /// A trained DFlash drafter (a separate GGUF, named in
    /// [`ServingConfig::dflash_gguf`]) in-fills a block of masked positions
    /// in one drafter pass per step; the same batched verify accepts.
    /// Costs the drafter's weights on the device and one six-layer draft
    /// KV cache per resident sequence.
    DFlash,
}

/// The bind-time knobs a worker cannot derive for itself.
///
/// Everything else the runtime needs — batch width, draft count, retention
/// interval — the worker reads off its own scheduler and cache
/// configuration, so those are not repeated here where they could disagree.
#[derive(Debug, Clone, PartialEq)]
pub struct ServingConfig {
    /// Tokens per chunked-prefill step.
    pub prefill_chunk: usize,
    /// Pinned host snapshot slots for this worker. Zero disables retention,
    /// and with it prefix sharing.
    pub snapshot_slots: usize,
    /// Which speculative decoder to run.
    pub speculation: Speculation,
    /// DFlash drafter GGUF; required by [`Speculation::DFlash`], ignored
    /// otherwise.
    pub dflash_gguf: Option<std::path::PathBuf>,
    /// Drop any draft shorter than this. Zero keeps every draft.
    pub draft_n_min: usize,
    /// Greedy confidence gate for the trained drafters: stop drafting at
    /// the first token whose probability under the drafter's own head
    /// falls below this. Zero disables the gate.
    pub draft_p_min: f32,
    /// Vision tower (mmproj) GGUF, or `None` for text-only serving.
    pub mmproj: Option<std::path::PathBuf>,
    /// Patch budget the vision tower is pre-allocated for.
    pub max_image_patches: usize,
}

impl ServingConfig {
    /// A configuration with the shipped snapshot budget and no drafting.
    pub fn new(prefill_chunk: usize) -> Self {
        Self {
            prefill_chunk,
            snapshot_slots: crate::runtime::DEFAULT_SNAPSHOT_SLOTS_PER_WORKER,
            speculation: Speculation::None,
            dflash_gguf: None,
            draft_n_min: 0,
            draft_p_min: 0.0,
            mmproj: None,
            max_image_patches: crate::runtime::DEFAULT_MAX_IMAGE_PATCHES,
        }
    }
}

#[derive(Debug)]
pub enum WorkerExecutionError {
    Runtime(RuntimeError),
    Admission(AdmissionError),
    Cache(PoolExhausted),
    NotBound,
}

impl core::fmt::Display for WorkerExecutionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Runtime(e) => write!(f, "{e}"),
            Self::Admission(e) => write!(f, "{e}"),
            Self::Cache(e) => write!(f, "{e}"),
            Self::NotBound => write!(f, "worker has not been bound to its CUDA device"),
        }
    }
}

impl core::error::Error for WorkerExecutionError {}

impl From<RuntimeError> for WorkerExecutionError {
    fn from(value: RuntimeError) -> Self {
        Self::Runtime(value)
    }
}

/// Identifies a worker within the engine.
///
/// Distinct from a CUDA device ordinal, though in the normal configuration
/// they happen to coincide. Keeping them separate means the engine can run
/// fewer workers than there are GPUs, or be tested with no GPUs at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WorkerId(pub u32);

impl core::fmt::Display for WorkerId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "worker{}", self.0)
    }
}

/// One worker's host-side state.
pub struct Worker {
    id: WorkerId,
    device_ordinal: usize,
    cache: CacheConfig,
    /// Attention KV blocks — the group that grows with sequence position.
    attention_pool: BlockPool,
    /// Gated DeltaNet state slots — fixed size per sequence, never padded to
    /// the attention page size. See `docs/CACHE.md`.
    gdn_pool: BlockPool,
    scheduler: Scheduler,
    runtime: Option<DeviceRuntimeHandle>,
    batch_scratch: BatchDescription,
    pending: HashMap<RequestId, PendingSequence>,
    reservations: Vec<CacheReservation>,
    vocab: Option<u32>,
}

struct CacheReservation {
    request: Option<RequestId>,
    attention: Vec<BlockId>,
    gdn: Option<BlockId>,
}

struct PendingSequence {
    request: NewRequest,
    prompt: Vec<i32>,
    images: Vec<crate::image::SequenceImage>,
    snapshot: Option<Arc<SequenceSnapshot>>,
    sampling: SamplingParams,
}

impl Worker {
    /// Build a worker's host-side state.
    ///
    /// The two pools are constructed with their own natural page sizes, taken
    /// from `cache`. They are deliberately not derived from one another.
    pub fn new(
        id: WorkerId,
        device_ordinal: usize,
        cache: CacheConfig,
        sched: SchedulerConfig,
        attention_blocks: u32,
        gdn_slots: u32,
    ) -> Self {
        let attention_pool =
            BlockPool::new("attention", cache.attention_page_bytes(), attention_blocks);
        let gdn_pool = BlockPool::new("gdn", cache.gdn_page_bytes(), gdn_slots);
        let scheduler = Scheduler::new(sched, attention_blocks);

        let width = scheduler.config().max_concurrent_decodes() as usize;
        let reservations = (0..width)
            .map(|_| CacheReservation {
                request: None,
                attention: Vec::with_capacity(attention_blocks as usize),
                gdn: None,
            })
            .collect();
        Self {
            id,
            device_ordinal,
            cache,
            attention_pool,
            gdn_pool,
            scheduler,
            runtime: None,
            batch_scratch: BatchDescription::with_capacity(width, width),
            pending: HashMap::new(),
            reservations,
            vocab: None,
        }
    }

    /// This worker's identifier.
    pub fn id(&self) -> WorkerId {
        self.id
    }

    /// The CUDA device this worker is for.
    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    /// The cache geometry this worker was built with.
    pub fn cache_config(&self) -> &CacheConfig {
        &self.cache
    }

    /// Free and total capacity, reported per group.
    ///
    /// Deliberately two independent figures. Summing attention blocks and GDN
    /// slots into one "capacity" number is the mistake `docs/CACHE.md`
    /// rule 1 exists to prevent.
    pub fn capacity(&self) -> CapacityReport {
        CapacityReport {
            attention_free_blocks: self.attention_pool.free_count(),
            attention_total_blocks: self.attention_pool.total(),
            attention_block_size: self.cache.attention_block_size(),
            gdn_free_slots: self.gdn_pool.free_count(),
            gdn_total_slots: self.gdn_pool.total(),
            gdn_bytes_per_slot: self.cache.gdn_page_bytes(),
        }
    }

    /// Fraction of the attention block pool in use, in `[0, 1]`.
    ///
    /// Counted from the attention group alone, because only it grows with
    /// sequence position. The GDN group is a fixed per-slot cost and mixing
    /// it in would make utilization mean nothing.
    pub fn kv_utilization(&self) -> f64 {
        let total = self.attention_pool.total();
        if total == 0 {
            return 1.0;
        }
        let used = total - self.attention_pool.free_count();
        f64::from(used) / f64::from(total)
    }

    /// Sequences assigned to this worker and not yet finished.
    ///
    /// Waiting and running both, because both will occupy a decode slot. This
    /// is what the router balances on: [`Self::queued_tokens`] charges a
    /// waiting request a whole token budget but a resident decoding one only
    /// `tokens_per_decode_step`, so on its own it is nearly blind to a card
    /// that is already busy generating.
    pub fn resident_sequences(&self) -> u32 {
        (self.scheduler.waiting_len() + self.scheduler.running_len()) as u32
    }

    /// Tokens scheduled but not yet computed.
    ///
    /// Approximated as one step's worth of budget per running request, which
    /// is what the router needs: a relative measure of backlog, not an exact
    /// token count.
    pub fn queued_tokens(&self) -> u32 {
        let waiting = self.scheduler.waiting_len() as u32;
        let running = self.scheduler.running_len() as u32;
        waiting.saturating_mul(self.scheduler.config().token_budget())
            + running.saturating_mul(self.scheduler.config().tokens_per_decode_step())
    }

    /// Describe this worker to the router for a specific incoming request.
    ///
    /// `matched_tokens` comes from the shared prefix tree and must already be
    /// truncated to a GDN retention boundary — see `docs/CACHE.md`. This
    /// method does not truncate it, because the tree is the only thing that
    /// knows where snapshots were actually retained.
    pub fn load_for(&self, matched_tokens: u32, req: &NewRequest) -> WorkerLoad {
        WorkerLoad {
            id: self.id,
            matched_tokens,
            queued_tokens: self.queued_tokens(),
            kv_utilization: self.kv_utilization(),
            resident_sequences: self.resident_sequences(),
            can_admit: self.scheduler.can_admit(req),
        }
    }

    /// Admit a request onto this worker.
    pub fn admit(&mut self, req: NewRequest) -> Result<RequestId, AdmissionError> {
        self.scheduler.admit(req)
    }

    /// Create the CUDA half of this worker and prebuild every decode width.
    pub fn bind_device(
        &mut self,
        model_path: &Path,
        model: ModelConfig,
        serving: ServingConfig,
    ) -> Result<(), RuntimeError> {
        self.bind_device_inner(model_path, model, serving, true)
    }

    /// Bind the real scheduler/runtime path while leaving EOS as an ordinary
    /// token, so a fixed-width throughput benchmark cannot silently shrink.
    pub fn bind_device_for_benchmark(
        &mut self,
        model_path: &Path,
        model: ModelConfig,
        serving: ServingConfig,
    ) -> Result<(), RuntimeError> {
        self.bind_device_inner(model_path, model, serving, false)
    }

    fn bind_device_inner(
        &mut self,
        model_path: &Path,
        model: ModelConfig,
        serving: ServingConfig,
        stop_on_eos: bool,
    ) -> Result<(), RuntimeError> {
        let max_batch = self.scheduler.config().max_concurrent_decodes() as usize;
        let drafts = self.scheduler.config().draft_tokens_per_step() as usize;
        let history_capacity =
            (self.attention_pool.total() * self.cache.attention_block_size()) as usize;
        // The self-speculative drafters' draft caps (`ngram`'s window,
        // `ngram-simple`/`ngram-map-*`'s `size_m`, `ngram-mod`'s `n_max`)
        // are the scheduler's draft count: the scheduler charges those
        // tokens against its step budget, and a second copy here could
        // disagree.
        let spec = match serving.speculation {
            _ if drafts == 0 => None,
            Speculation::None | Speculation::Mtp | Speculation::DFlash => None,
            Speculation::Ngram { min, max } => Some(SelfSpecConfig::Ngram(
                NgramConfig::new(min, max, drafts, history_capacity)
                    .map_err(RuntimeError::Speculation)?,
            )),
            Speculation::NgramSimple { size_n } => Some(SelfSpecConfig::Simple(
                NgramSimpleConfig::new(size_n, drafts, history_capacity)
                    .map_err(RuntimeError::Speculation)?,
            )),
            Speculation::NgramMod { n_match, n_min } => Some(SelfSpecConfig::Mod(
                NgramModConfig::new(n_match, n_min, drafts, history_capacity)
                    .map_err(RuntimeError::Speculation)?,
            )),
            Speculation::NgramMapK { size_n, min_hits } => Some(SelfSpecConfig::Map(
                NgramMapConfig::new(size_n, drafts, true, min_hits, history_capacity)
                    .map_err(RuntimeError::Speculation)?,
            )),
            Speculation::NgramMapK4v { size_n, min_hits } => Some(SelfSpecConfig::Map(
                NgramMapConfig::new(size_n, drafts, false, min_hits, history_capacity)
                    .map_err(RuntimeError::Speculation)?,
            )),
        };
        let mtp_drafts = match serving.speculation {
            Speculation::Mtp => drafts,
            _ => 0,
        };
        let dflash = match serving.speculation {
            Speculation::DFlash if drafts > 0 => Some(crate::runtime::DFlashServing {
                gguf: serving.dflash_gguf.clone().ok_or_else(|| {
                    RuntimeError::Schema(
                        "--spec-type dflash needs a drafter GGUF (dflash_gguf)".into(),
                    )
                })?,
                drafts,
            }),
            _ => None,
        };
        let vocab = model.vocab_size;
        self.runtime = Some(DeviceRuntimeHandle::spawn(
            self.device_ordinal,
            model_path.to_path_buf(),
            model,
            RuntimeConfig {
                prefill_chunk: serving.prefill_chunk,
                max_batch,
                spec,
                mtp_drafts,
                dflash,
                draft_n_min: serving.draft_n_min,
                draft_p_min: serving.draft_p_min,
                retention_interval: self.cache.gdn_retention_interval() as usize,
                snapshot_slots: serving.snapshot_slots,
                stop_on_eos,
                mmproj: serving.mmproj.clone(),
                max_image_patches: serving.max_image_patches,
            },
        )?);
        self.vocab = Some(vocab);
        Ok(())
    }

    pub fn is_device_bound(&self) -> bool {
        self.runtime.is_some()
    }

    /// How much room this worker's pinned snapshot arena has left, or `None`
    /// for a host-only worker, which has no arena to run out of.
    pub fn snapshot_slots(&self) -> Option<&SnapshotSlots> {
        self.runtime
            .as_ref()
            .map(crate::runtime::DeviceRuntimeHandle::snapshot_slots)
    }

    /// Admit both scheduler metadata and the actual prompt token ids.
    ///
    /// `sampling` is how this request's tokens are chosen;
    /// [`SamplingParams::GREEDY`] keeps the on-device argmax path.
    pub fn admit_tokens(
        &mut self,
        req: NewRequest,
        prompt: Vec<i32>,
        images: Vec<crate::image::SequenceImage>,
        sampling: SamplingParams,
    ) -> Result<RequestId, WorkerExecutionError> {
        self.validate_pending(req, &prompt, None)?;
        let id = self
            .scheduler
            .admit(req)
            .map_err(WorkerExecutionError::Admission)?;
        self.pending.insert(
            id,
            PendingSequence {
                request: req,
                prompt,
                images,
                snapshot: None,
                sampling,
            },
        );
        Ok(id)
    }

    pub fn admit_tokens_restored(
        &mut self,
        req: NewRequest,
        prompt: Vec<i32>,
        images: Vec<crate::image::SequenceImage>,
        snapshot: Arc<SequenceSnapshot>,
        sampling: SamplingParams,
    ) -> Result<RequestId, WorkerExecutionError> {
        self.validate_pending(req, &prompt, Some(&snapshot))?;
        let prefix = snapshot.position() as u32;
        let id = self
            .scheduler
            .admit_with_prefix(req, prefix)
            .map_err(WorkerExecutionError::Admission)?;
        self.pending.insert(
            id,
            PendingSequence {
                request: req,
                prompt,
                images,
                snapshot: Some(snapshot),
                sampling,
            },
        );
        Ok(id)
    }

    pub fn snapshot(&self, id: RequestId) -> Result<Arc<SequenceSnapshot>, WorkerExecutionError> {
        self.runtime
            .as_ref()
            .ok_or(WorkerExecutionError::NotBound)?
            .snapshot(id)
            .map_err(WorkerExecutionError::Runtime)
    }

    /// Advance one scheduling step.
    pub fn step(&mut self) -> BatchDescription {
        self.scheduler.step()
    }

    /// Schedule and execute one live device step.
    pub fn step_device(&mut self) -> Result<DeviceStep, WorkerExecutionError> {
        self.scheduler.step_into(&mut self.batch_scratch);
        if self.runtime.is_none() {
            return Err(WorkerExecutionError::NotBound);
        }
        for index in 0..self.batch_scratch.prefills.len() {
            let id = self.batch_scratch.prefills[index].id;
            if self.pending.contains_key(&id) {
                self.reserve_cache(id)?;
                let pending = self
                    .pending
                    .remove(&id)
                    .expect("pending request was checked above");
                let runtime = self.runtime.as_ref().expect("runtime was checked above");
                match pending.snapshot {
                    Some(snapshot) => runtime.admit_restored(
                        pending.request,
                        pending.prompt,
                        pending.images,
                        snapshot,
                        pending.sampling,
                    )?,
                    None => runtime.admit(
                        pending.request,
                        pending.prompt,
                        pending.images,
                        pending.sampling,
                    )?,
                }
            }
        }
        let batch = std::mem::take(&mut self.batch_scratch);
        let mut step = self
            .runtime
            .as_ref()
            .expect("runtime was checked above")
            .execute(batch)
            .map_err(WorkerExecutionError::Runtime)?;
        for decode in &step.scheduled.decodes {
            let emitted = step
                .generated
                .iter()
                .filter(|(id, _)| *id == decode.id)
                .count() as u32;
            if emitted > 1 {
                self.scheduler.advance_speculative(decode.id, emitted - 1);
            }
        }
        for &id in &step.completed {
            self.scheduler.finish_request(id);
            self.release_cache(id);
        }
        let decode_items = step.scheduled.decodes.len();
        let prefill_items = step.scheduled.prefills.len();
        self.batch_scratch = std::mem::take(&mut step.scheduled);
        Ok(DeviceStep {
            decode_items,
            prefill_items,
            generated: step.generated,
            completed: step.completed,
            stopped: step.stopped,
            retained: step.retained,
        })
    }

    fn validate_pending(
        &self,
        req: NewRequest,
        prompt: &[i32],
        snapshot: Option<&SequenceSnapshot>,
    ) -> Result<(), WorkerExecutionError> {
        let vocab = self.vocab.ok_or(WorkerExecutionError::NotBound)?;
        if self.pending.contains_key(&req.id)
            || self.scheduler.is_waiting(req.id)
            || self.scheduler.is_running(req.id)
        {
            return Err(RuntimeError::DuplicateRequest(req.id).into());
        }
        if prompt.len() != req.prompt_tokens as usize {
            return Err(RuntimeError::PromptLength {
                declared: req.prompt_tokens,
                actual: prompt.len(),
            }
            .into());
        }
        if let Some(&token) = prompt
            .iter()
            .find(|&&token| token < 0 || token as u32 >= vocab)
        {
            return Err(RuntimeError::TokenOutOfRange { token, vocab }.into());
        }
        if let Some(snapshot) = snapshot
            && snapshot.position() > prompt.len()
        {
            return Err(RuntimeError::SnapshotPrefix {
                snapshot: snapshot.position(),
                prompt: prompt.len(),
            }
            .into());
        }
        Ok(())
    }

    /// Cancel a waiting or running request and release all of its state.
    pub fn cancel(&mut self, id: RequestId) -> bool {
        let pending = self.pending.remove(&id).is_some();
        let running = self.scheduler.is_running(id);
        if running && let Some(runtime) = &self.runtime {
            let _ = runtime.remove(id);
        }
        let cancelled = self.scheduler.cancel_request(id) || pending;
        if cancelled {
            self.release_cache(id);
        }
        cancelled
    }

    fn reserve_cache(&mut self, id: RequestId) -> Result<(), WorkerExecutionError> {
        let blocks = self
            .scheduler
            .reserved_attention_blocks(id)
            .expect("a first prefill is running and has a scheduler reservation");
        let slot = self
            .reservations
            .iter_mut()
            .find(|slot| slot.request.is_none())
            .expect("scheduler width and reservation-slot count agree");
        self.attention_pool
            .alloc_into(blocks, &mut slot.attention)
            .map_err(WorkerExecutionError::Cache)?;
        match self.gdn_pool.alloc() {
            Ok(gdn) => {
                slot.request = Some(id);
                slot.gdn = Some(gdn);
                Ok(())
            }
            Err(error) => {
                self.attention_pool.free_blocks(slot.attention.drain(..));
                Err(WorkerExecutionError::Cache(error))
            }
        }
    }

    fn release_cache(&mut self, id: RequestId) {
        let Some(slot) = self
            .reservations
            .iter_mut()
            .find(|slot| slot.request == Some(id))
        else {
            return;
        };
        self.attention_pool.free_blocks(slot.attention.drain(..));
        if let Some(gdn) = slot.gdn.take() {
            self.gdn_pool.free_block(gdn);
        }
        slot.request = None;
    }

    /// Read-only access to the scheduler.
    pub fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }
}

impl core::fmt::Debug for Worker {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Worker")
            .field("id", &self.id)
            .field("device_ordinal", &self.device_ordinal)
            .field("kv_utilization", &self.kv_utilization())
            .field("waiting", &self.scheduler.waiting_len())
            .field("running", &self.scheduler.running_len())
            .field("device_bound", &self.is_device_bound())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_reservations_move_both_natural_cache_groups_together() {
        let model = ModelConfig::qwen3_6_35b_a3b();
        let cache = CacheConfig::with_defaults(model).unwrap();
        let sched = SchedulerConfig::with_defaults(4096, cache.attention_block_size(), 3).unwrap();
        let mut worker = Worker::new(WorkerId(0), 0, cache, sched, 32, 3);
        let id = RequestId(7);
        worker
            .admit(NewRequest {
                id,
                prompt_tokens: 512,
                max_output_tokens: 256,
            })
            .unwrap();
        let batch = worker.step();
        assert_eq!(batch.prefills[0].id, id);
        worker.reserve_cache(id).unwrap();

        // 512 + 256 tokens plus the default 3-token draft window's verify
        // scratch: 771 tokens -> 4 blocks of 256.
        let live = worker.capacity();
        assert_eq!(live.attention_free_blocks, 28);
        assert_eq!(live.gdn_free_slots, 2);
        assert_eq!(worker.kv_utilization(), 4.0 / 32.0);

        assert!(worker.cancel(id));
        let released = worker.capacity();
        assert_eq!(released.attention_free_blocks, 32);
        assert_eq!(released.gdn_free_slots, 3);
    }
}
