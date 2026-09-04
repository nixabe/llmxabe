//! The engine: three workers, one router, one shared prefix tree.
//!
//! This is the type that makes three GPUs one engine rather than three
//! independent servers. The whole of that difference lives here and in
//! [`crate::router`] — below this layer, workers are entirely independent and
//! never touch each other's device memory.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use parking_lot::{Mutex, MutexGuard, RwLock};
use smallvec::SmallVec;
use tracing::{debug, warn};

use xabe_cache::config::CacheConfig;
use xabe_cache::pool::BlockId;
use xabe_cache::radix::{BlockHash, PrefixBlock, RadixTree};
use xabe_model::ModelConfig;
use xabe_sched::config::SchedulerConfig;
use xabe_sched::error::AdmissionError;
use xabe_sched::request::{NewRequest, RequestId};

use crate::prefix::{SequenceChain, SharedSnapshots};
use crate::router::{Routed, RouterConfig, RoutingError, WorkerLoad, route};
use crate::runtime::{DeviceStep, RuntimeError};
use crate::sampling::SamplingParams;
use crate::state::SequenceSnapshot;
use crate::worker::{ServingConfig, Worker, WorkerExecutionError, WorkerId};
use xabe_grammar::ToolConstraint;

#[derive(Debug)]
pub enum EngineExecutionError {
    Placement(PlacementError),
    Worker {
        worker: WorkerId,
        source: WorkerExecutionError,
    },
    WorkerPanicked(WorkerId),
    SnapshotNotRetained {
        position: usize,
        interval: u32,
    },
    MissingWorker(WorkerId),
}

impl core::fmt::Display for EngineExecutionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Placement(e) => write!(f, "{e}"),
            Self::Worker { worker, source } => write!(f, "{worker}: {source}"),
            Self::WorkerPanicked(worker) => write!(f, "{worker} execution thread panicked"),
            Self::SnapshotNotRetained { position, interval } => write!(
                f,
                "snapshot position {position} is not a {interval}-token retention boundary"
            ),
            Self::MissingWorker(worker) => write!(f, "{worker} does not exist"),
        }
    }
}

impl core::error::Error for EngineExecutionError {}

/// Why a request could not be placed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlacementError {
    /// No worker was available.
    Routing(RoutingError),
    /// The chosen worker refused the request.
    ///
    /// Should be rare: routing already excludes workers that cannot admit.
    /// Reaching this means worker state changed between scoring and
    /// admission, which is worth surfacing rather than retrying silently.
    Admission(AdmissionError),
}

impl core::fmt::Display for PlacementError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Routing(e) => write!(f, "routing failed: {e}"),
            Self::Admission(e) => write!(f, "admission failed after routing: {e}"),
        }
    }
}

impl core::error::Error for PlacementError {}

/// A placed request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Placement {
    /// Where it went.
    pub worker: WorkerId,
    /// Its scheduler-side identifier.
    pub request: RequestId,
    /// Prefix tokens this worker already held, so prefill can skip them.
    ///
    /// Already truncated to a GDN retention boundary by the prefix tree.
    pub reusable_prefix_tokens: u32,
    /// Router score, retained so a placement can be explained rather than
    /// merely observed.
    pub score: f64,
}

/// Three workers, one router, one shared prefix cache.
pub struct Engine {
    /// One lock per worker, which is what lets the three of them run at
    /// their own pace.
    ///
    /// The engine used to be one `Mutex<Engine>` stepped by one driver
    /// thread that joined all three workers every step, so every card ran at
    /// the speed of the slowest. Measured on a decode beside two prefills on
    /// *other* cards: mean inter-token latency 12.1 ms → 146.0 ms, with 21 of
    /// 199 tokens absorbing 26.9 s of the 29.0 s the session took. The median
    /// was unchanged at 12.1 ms, which is why this went unnoticed for so
    /// long — the cost is entirely in the tail. See `docs/BENCHMARKS.md`.
    workers: Vec<Mutex<Worker>>,
    /// The shared prefix cache.
    ///
    /// This is the project's one structural advantage over three separate
    /// `llama-server` processes: a radix tree in one address space, behind an
    /// `RwLock`, with no serialization format, no filenames, and no tmpfs.
    /// See `docs/ARCHITECTURE.md`.
    prefix_tree: Arc<RadixTree>,
    router: RouterConfig,
    snapshots: RwLock<SharedSnapshots>,
    /// What names each live sequence's blocks, extended as it generates.
    ///
    /// Keyed by worker as well as request, so a driver thread only ever
    /// touches its own rows; the lock is held for a map operation, never
    /// across a GPU step.
    request_chains: Mutex<HashMap<(WorkerId, RequestId), SequenceChain>>,
    request_refs: Mutex<HashMap<(WorkerId, RequestId), Vec<BlockHash>>>,
    /// Serializes the route-and-admit decision.
    ///
    /// Scoring reads every worker's load and then admits to the cheapest
    /// one, which is only correct if no other handler admits in between.
    /// One `Mutex<Engine>` used to provide that for free; per-worker locks
    /// do not, and concurrent scorers all see the same least-loaded card and
    /// all pile onto it. Measured with nine simultaneous 64K sessions on
    /// three cards: placements came out 4/2/3 rather than 3/3/3, and the
    /// fourth session on the oversubscribed card waited for a slot — a
    /// 150 s first token against 113 s for its neighbours, and 269 s on a
    /// worse split.
    ///
    /// This is held across scoring and admission only. It never covers a GPU
    /// step, so the driver loops never wait on it.
    placement: Mutex<()>,
    max_prefix_nodes: usize,
    block_size: u32,
    /// Snapshot slots each worker must keep free for its own live sequences.
    ///
    /// One per concurrent sequence, which is exactly enough that every live
    /// sequence can take its next snapshot the moment it reaches a retention
    /// boundary. Below that, the shared cache is holding slots a live
    /// sequence is about to need.
    slot_reserve: usize,
}

impl Engine {
    /// Build an engine over `device_ordinals`, one worker per device.
    ///
    /// All workers share one prefix tree. They share nothing else.
    pub fn new(
        device_ordinals: &[usize],
        cache: CacheConfig,
        sched: SchedulerConfig,
        attention_blocks_per_worker: u32,
        gdn_slots_per_worker: u32,
        router: RouterConfig,
    ) -> Self {
        let prefix_tree = Arc::new(RadixTree::new(cache.attention_block_size()));

        let workers = device_ordinals
            .iter()
            .enumerate()
            .map(|(i, &ordinal)| {
                Mutex::new(Worker::new(
                    WorkerId(i as u32),
                    ordinal,
                    cache.clone(),
                    sched,
                    attention_blocks_per_worker,
                    gdn_slots_per_worker,
                ))
            })
            .collect();

        Self {
            workers,
            prefix_tree,
            router,
            snapshots: RwLock::new(SharedSnapshots::default()),
            request_chains: Mutex::new(HashMap::new()),
            request_refs: Mutex::new(HashMap::new()),
            placement: Mutex::new(()),
            max_prefix_nodes: attention_blocks_per_worker as usize,
            block_size: cache.attention_block_size(),
            slot_reserve: gdn_slots_per_worker as usize,
        }
    }

    /// Number of workers.
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// The shared prefix cache.
    pub fn prefix_tree(&self) -> &Arc<RadixTree> {
        &self.prefix_tree
    }

    /// Lock one worker.
    ///
    /// Hold the guard for as short a span as the work allows: a driver
    /// thread takes it for a whole GPU step, so anything that blocks on it
    /// waits out that step.
    pub fn worker(&self, id: WorkerId) -> Option<MutexGuard<'_, Worker>> {
        self.workers.get(id.0 as usize).map(|worker| worker.lock())
    }

    /// Score every worker against an incoming request.
    ///
    /// Exposed separately from [`Self::place`] so routing decisions can be
    /// inspected and logged. The prefix match is looked up once, in the shared
    /// tree, and is already truncated to a retention boundary.
    pub fn score_workers(&self, req: &NewRequest, block_hashes: &[BlockHash]) -> Vec<WorkerLoad> {
        let matched = self.prefix_tree.match_prefix(block_hashes);
        // Credit `gdn_matched_tokens`, not `matched_tokens`. The attention
        // match may run further, but recurrent state is only resumable at a
        // retained snapshot boundary, and 30 of 40 layers hold recurrent
        // state — so prefill genuinely restarts at the snapshot. Scoring the
        // longer attention match would overstate the saving and route toward
        // a worker that cannot actually deliver it. See `docs/CACHE.md`.
        let usable = matched
            .gdn_snapshot_hash
            .filter(|&hash| self.snapshots.read().contains(hash))
            .map_or(0, |_| matched.gdn_matched_tokens);
        self.workers
            .iter()
            .map(|worker| worker.lock().load_for(usable, req))
            .collect()
    }

    /// Cap a caller's speculative output ceiling at one slot's share.
    ///
    /// Returns what `max_output_tokens` was, when it had to be lowered.
    ///
    /// `max_output_tokens` is reserved in full, up front, and backed by real
    /// pages — so an agent that asks for 65,536 tokens and emits fifty holds
    /// 1.28 GiB of this card hostage for its whole life, and two of them are
    /// enough to make the third session queue behind them. The cap is on the
    /// reservation only: a sequence that genuinely runs that long stops on
    /// it and reports `length`, which is what llama.cpp does when a slot's
    /// context fills — it checks the prompt against the slot and treats
    /// `n_predict` purely as a stopping condition
    /// (`tools/server/server-context.cpp`, `slot.task->n_tokens() >=
    /// slot.n_ctx` and `server_slot::n_remaining`).
    ///
    /// A prompt that is already longer than the ceiling keeps a floor of one
    /// block, so this only ever lowers a speculative ceiling and never turns
    /// an admissible request into one that cannot generate at all.
    fn cap_output_reservation(&self, req: &mut NewRequest) -> Option<u32> {
        let worker = self.workers.first()?;
        let (ceiling, floor) = {
            let worker = worker.lock();
            (
                worker.scheduler().per_slot_token_ceiling(),
                worker.scheduler().config().block_size(),
            )
        };
        let headroom = ceiling.saturating_sub(req.prompt_tokens).max(floor);
        (req.max_output_tokens > headroom).then(|| {
            let requested = req.max_output_tokens;
            req.max_output_tokens = headroom;
            requested
        })
    }

    /// Turn a routing refusal into the reason a worker actually gave.
    ///
    /// `AllWorkersSaturated` says only that every worker declined, which is
    /// what a caller was left to guess from. The workers share a
    /// configuration, so the first one's refusal is the answer for all of
    /// them.
    fn routing_failure(&self, error: RoutingError, req: &NewRequest) -> PlacementError {
        if matches!(error, RoutingError::AllWorkersSaturated)
            && let Some(worker) = self.workers.first()
            && let Some(refusal) = worker.lock().scheduler().refusal_for(req)
        {
            return PlacementError::Admission(refusal);
        }
        PlacementError::Routing(error)
    }

    /// Route a request and admit it onto the chosen worker.
    ///
    /// `block_hashes` are the chained block hashes of the request's prompt,
    /// as produced by [`xabe_cache::radix::hash_block`].
    pub fn place(
        &self,
        mut req: NewRequest,
        block_hashes: &[BlockHash],
    ) -> Result<Placement, PlacementError> {
        self.cap_output_reservation(&mut req);
        let budget = self
            .workers
            .first()
            .map(|worker| worker.lock().scheduler().config().token_budget())
            .unwrap_or(0);

        let _routing = self.placement.lock();
        let loads = self.score_workers(&req, block_hashes);
        let Routed {
            worker,
            score,
            matched_tokens,
        } = route(&self.router, &loads, req.prompt_tokens, budget)
            .map_err(|error| self.routing_failure(error, &req))?;

        let request = self
            .worker(worker)
            .expect("router returned a worker that exists")
            .admit(req)
            .map_err(PlacementError::Admission)?;

        Ok(Placement {
            worker,
            request,
            reusable_prefix_tokens: matched_tokens,
            score,
        })
    }

    /// Load one model replica on every worker's configured device.
    ///
    /// All of them at once. The devices are independent — separate contexts,
    /// separate arenas, separate PCIe paths — so loading them one after
    /// another simply multiplied startup by the worker count.
    ///
    /// The larger win is on the host side, and it is why this matters even
    /// though a single upload is not PCIe-bound: every worker reads the *same*
    /// mapped file. Serially, each read raced the page cache and lost whenever
    /// the model did not fit in what was left of it, so the file was pulled
    /// off disk once per worker. Concurrently, the first fault brings a page
    /// in and the others find it already there — one pass over the file
    /// instead of three, which on a slow disk is the whole startup.
    ///
    /// A failure is reported after every worker has finished rather than at
    /// the first error, because the scope must join them all regardless; the
    /// lowest-numbered failing worker is the one named.
    pub fn bind_devices(
        &self,
        model_path: &Path,
        model: ModelConfig,
        serving: ServingConfig,
    ) -> Result<(), (WorkerId, RuntimeError)> {
        let outcomes: Vec<(WorkerId, Result<(), RuntimeError>)> = std::thread::scope(|scope| {
            let threads: Vec<_> = self
                .workers
                .iter()
                .map(|worker| {
                    let model = model.clone();
                    let serving = serving.clone();
                    scope.spawn(move || {
                        let mut worker = worker.lock();
                        let id = worker.id();
                        (id, worker.bind_device(model_path, model, serving))
                    })
                })
                .collect();
            threads
                .into_iter()
                .map(|thread| thread.join().expect("a worker bind thread panicked"))
                .collect()
        });
        for (id, outcome) in outcomes {
            outcome.map_err(|error| (id, error))?;
        }
        Ok(())
    }

    /// Route and admit a request together with its tokenized prompt.
    ///
    /// The block hashes are derived here, from these tokens, rather than
    /// supplied: they are what the engine will later file this sequence's
    /// snapshots under, and a caller-supplied chain that disagreed with the
    /// tokens would publish a snapshot under a prefix it does not describe.
    /// See `SequenceChain`.
    pub fn place_tokens(
        &self,
        mut req: NewRequest,
        prompt: Vec<i32>,
        images: Vec<crate::image::SequenceImage>,
        sampling: SamplingParams,
        constraint: Option<Box<ToolConstraint>>,
    ) -> Result<Placement, EngineExecutionError> {
        let placements: Vec<crate::image::ImagePlacement> =
            images.iter().map(|i| i.placement).collect();
        debug_assert!(
            crate::image::validate_placements(&placements, prompt.len()).is_ok(),
            "image placements must be validated at the server boundary"
        );
        let budget = self
            .workers
            .first()
            .map(|worker| worker.lock().scheduler().config().token_budget())
            .unwrap_or(0);
        if let Some(requested) = self.cap_output_reservation(&mut req) {
            debug!(
                request = req.id.0,
                requested,
                capped_to = req.max_output_tokens,
                prompt_tokens = req.prompt_tokens,
                "output ceiling exceeds one slot's share; capping the reservation"
            );
        }
        let chain = SequenceChain::new(self.block_size, &prompt, &placements);
        // Claiming a snapshot for reuse is part of the same decision as
        // choosing a worker, so it is inside the routing lock too: two
        // handlers that both matched the same prefix must not both take it.
        let _routing = self.placement.lock();
        let matched = self.prefix_tree.match_prefix(chain.hashes());
        let snapshot = matched
            .gdn_snapshot_hash
            .and_then(|hash| self.snapshots.write().take_for_reuse(hash))
            // A snapshot covering the *whole* prompt supplies the first
            // output token through its recorded `next_token`. That inherited
            // token is only right when this request is greedy and the
            // recording sequence was too (a sampled producer records none) —
            // otherwise fall back to prefilling, which recomputes the final
            // logits so the token can be chosen properly.
            .filter(|snapshot| {
                snapshot.position() < prompt.len()
                    || (sampling.is_greedy() && snapshot.next_token().is_some())
            });
        let loads = self.score_workers(&req, chain.hashes());
        let Routed {
            worker,
            score,
            matched_tokens,
        } = route(&self.router, &loads, req.prompt_tokens, budget)
            .map_err(|error| EngineExecutionError::Placement(self.routing_failure(error, &req)))?;
        let request = {
            let mut target = self
                .worker(worker)
                .expect("router returned an existing worker");
            if let Some(snapshot) = snapshot {
                target.admit_tokens_restored(req, prompt, images, snapshot, sampling, constraint)
            } else {
                target.admit_tokens(req, prompt, images, sampling, constraint)
            }
        }
        .map_err(|source| EngineExecutionError::Worker { worker, source })?;
        let referenced = matched_tokens as usize / self.block_size as usize;
        if referenced > 0 {
            let hashes = chain.hashes()[..referenced].to_vec();
            self.prefix_tree.incr_ref_chain(&hashes);
            self.request_refs.lock().insert((worker, request), hashes);
        }
        self.request_chains.lock().insert((worker, request), chain);
        Ok(Placement {
            worker,
            request,
            reusable_prefix_tokens: matched_tokens,
            score,
        })
    }

    /// Cancel a live request and release its scheduler, runtime, and cache
    /// bookkeeping regardless of whether it is waiting or running.
    pub fn cancel(&self, request: RequestId) -> bool {
        let Some(worker) = self.workers.iter().find_map(|worker| {
            let worker = worker.lock();
            (worker.scheduler().is_waiting(request) || worker.scheduler().is_running(request))
                .then_some(worker.id())
        }) else {
            return false;
        };
        self.request_chains.lock().remove(&(worker, request));
        if let Some(hashes) = self.request_refs.lock().remove(&(worker, request)) {
            self.prefix_tree.decr_ref_chain(&hashes);
        }
        self.worker(worker)
            .is_some_and(|mut worker| worker.cancel(request))
    }

    /// Publish a snapshot into the shared prefix tree, named by the chain of
    /// the sequence that produced it.
    ///
    /// Returns `false` when the snapshot was not shared. That is a normal
    /// outcome, not a failure — the sequence keeps using it locally either
    /// way — and it happens when the chain cannot name the snapshot's
    /// position, when the chain and the runtime disagree about what token
    /// followed it, or when the worker's snapshot arena has no room to spare.
    fn install_snapshot(
        &self,
        worker: WorkerId,
        snapshot: Arc<SequenceSnapshot>,
        chain: &SequenceChain,
    ) -> Result<bool, EngineExecutionError> {
        // Take what this needs from the worker and drop the guard, rather
        // than reading through it further down. Everything below touches the
        // shared prefix tree and snapshot map, and holding a worker lock
        // across those would be the one place in the engine where two locks
        // are held at once — an ordering constraint to get right later for
        // no gain, since both values are `Copy` or cheaply cloned.
        let (interval, slots) = {
            let source = self
                .worker(worker)
                .ok_or(EngineExecutionError::MissingWorker(worker))?;
            (
                source.cache_config().gdn_retention_interval(),
                source.snapshot_slots().cloned(),
            )
        };
        let position = snapshot.position();
        if position == 0 || !(position as u32).is_multiple_of(interval) {
            return Err(EngineExecutionError::SnapshotNotRetained { position, interval });
        }
        let Some(hashes) = chain.hashes_for(position) else {
            debug!(
                position,
                named = chain.position(),
                "sequence has not named this snapshot's position yet; not sharing it"
            );
            return Ok(false);
        };

        // The chain is assembled here, from the tokens the runtime reported;
        // the snapshot's `next_token` was recorded there, by the runtime, at
        // the same position. They are two independent records of the same
        // fact, so comparing them catches a chain that has drifted out of
        // step with the sequence it names — the one failure that would
        // otherwise be silent, and would resume a later request from a prefix
        // it never sent. It answers only for snapshots past the prompt, which
        // are exactly the ones the chain had to grow to reach.
        if let (Some(followed), Some(predicted)) =
            (chain.generated_token_at(position), snapshot.next_token())
            && followed != predicted as u32
        {
            warn!(
                position,
                followed,
                predicted,
                "prefix chain disagrees with the runtime about this sequence; not sharing it"
            );
            return Ok(false);
        }

        // Every published snapshot pins at least one of this worker's arena
        // slots until it is dropped. Yield before publishing rather than let
        // a live sequence hit `SnapshotArenaExhausted`, which would switch
        // that sequence's retention off permanently.
        if let Some(slots) = slots {
            let reserve = self.slot_reserve;
            if !self
                .snapshots
                .write()
                .reclaim_for(worker, reserve, || slots.available())
            {
                return Ok(false);
            }
        }

        let blocks = hashes.len();
        let prefix: Vec<PrefixBlock> = hashes
            .iter()
            .enumerate()
            .map(|(index, &hash)| PrefixBlock {
                hash,
                block: BlockId(index as u32),
                gdn_snapshot: (index + 1 == blocks).then_some(BlockId(0)),
            })
            .collect();
        let leaf = prefix.last().expect("non-zero snapshot has a leaf").hash;
        self.prefix_tree.insert(&prefix);
        self.snapshots.write().publish(leaf, worker, snapshot);
        let excess = self.prefix_tree.len().saturating_sub(self.max_prefix_nodes);
        if excess > 0 {
            let evicted = self.prefix_tree.evict_unreferenced_entries(excess);
            let mut snapshots = self.snapshots.write();
            for entry in evicted {
                snapshots.remove(entry.hash);
            }
        }
        Ok(true)
    }

    /// Execute one scheduler step on a single worker.
    ///
    /// This is the unit the server drives: one thread per worker, each
    /// looping at its own pace. Only that worker's lock is taken, and it is
    /// released before the step's cache bookkeeping runs, so a card that is
    /// prefilling holds nothing another card's decode needs.
    ///
    /// `Ok(None)` means the worker has no device bound and there was nothing
    /// to step.
    pub fn step_worker(
        &self,
        worker: WorkerId,
    ) -> Result<Option<DeviceStep>, EngineExecutionError> {
        let mut step = {
            let Some(mut guard) = self.worker(worker) else {
                return Err(EngineExecutionError::MissingWorker(worker));
            };
            if !guard.is_device_bound() {
                return Ok(None);
            }
            guard
                .step_device()
                .map_err(|source| EngineExecutionError::Worker { worker, source })?
        };
        self.apply_step(worker, &mut step)?;
        Ok(Some(step))
    }

    /// Fold one worker's completed step into the shared cache bookkeeping.
    ///
    /// Runs with no worker lock held. The maps are keyed by worker, so
    /// concurrent callers for different workers never touch the same rows.
    fn apply_step(
        &self,
        worker: WorkerId,
        step: &mut DeviceStep,
    ) -> Result<(), EngineExecutionError> {
        // Extend the chains *before* installing this step's snapshots,
        // and the order is load-bearing rather than incidental.
        //
        // Within a step the runtime interleaves the two: a speculative
        // round retains at position p and then emits the token at p, and
        // the next round does the same at p+1. Taking the generated
        // tokens first means every chain has reached at least the deepest
        // position this step retained, and `hashes_for` reads only the
        // blocks below that — so no snapshot is ever named by a chain
        // that has not yet caught up to it.
        {
            let mut chains = self.request_chains.lock();
            for (request, token) in &step.generated {
                if let Some(chain) = chains.get_mut(&(worker, *request)) {
                    chain.push(*token);
                }
            }
        }
        for (request, snapshot) in std::mem::take(&mut step.retained) {
            // Clone the chain rather than install through the map guard:
            // `install_snapshot` reaches for the prefix tree and the shared
            // snapshot map, and holding the chain lock across that would
            // stall every other worker's bookkeeping behind this one.
            let chain = self.request_chains.lock().get(&(worker, request)).cloned();
            if let Some(chain) = chain {
                self.install_snapshot(worker, snapshot, &chain)?;
            }
        }
        for request in &step.completed {
            self.request_chains.lock().remove(&(worker, *request));
            let hashes = self.request_refs.lock().remove(&(worker, *request));
            if let Some(hashes) = hashes {
                self.prefix_tree.decr_ref_chain(&hashes);
            }
        }
        Ok(())
    }

    /// Step every device-bound worker once, concurrently, and join them.
    ///
    /// Retained for the smoke binary and the tests, which want one bounded
    /// unit of fleet-wide progress. The server does *not* drive the engine
    /// this way — the join is exactly the barrier that made every card run
    /// at the speed of the slowest. See [`Self::step_worker`].
    pub fn step_devices(
        &self,
    ) -> Result<SmallVec<[(WorkerId, DeviceStep); 3]>, EngineExecutionError> {
        std::thread::scope(|scope| {
            let handles: SmallVec<[_; 3]> = (0..self.workers.len())
                .map(|index| {
                    let worker = WorkerId(index as u32);
                    (worker, scope.spawn(move || self.step_worker(worker)))
                })
                .collect();
            handles
                .into_iter()
                .filter_map(|(worker, handle)| match handle.join() {
                    Ok(Ok(Some(step))) => Some(Ok((worker, step))),
                    Ok(Ok(None)) => None,
                    Ok(Err(error)) => Some(Err(error)),
                    Err(_) => Some(Err(EngineExecutionError::WorkerPanicked(worker))),
                })
                .collect()
        })
    }
}

impl core::fmt::Debug for Engine {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Engine")
            .field("workers", &self.workers.len())
            .field("prefix_tree_nodes", &self.prefix_tree.len())
            .field("shared_snapshots", &self.snapshots.read().len())
            .field("router", &self.router)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xabe_model::ModelConfig;

    /// A small pool, so capacity limits are reachable in a test.
    fn engine(attention_blocks: u32) -> Engine {
        let cache = CacheConfig::with_defaults(ModelConfig::qwen3_6_35b_a3b()).unwrap();
        // Budget comfortably above block_size + max_concurrent_decodes, as
        // rule 3 requires.
        let sched = SchedulerConfig::with_defaults(4096, cache.attention_block_size(), 3).unwrap();
        Engine::new(
            &[0, 1, 2],
            cache,
            sched,
            attention_blocks,
            3,
            RouterConfig::balanced(),
        )
    }

    fn req(id: u64, prompt: u32, output: u32) -> NewRequest {
        NewRequest {
            id: RequestId(id),
            prompt_tokens: prompt,
            max_output_tokens: output,
        }
    }

    #[test]
    fn an_engine_has_one_worker_per_device_and_one_shared_tree() {
        let e = engine(1024);
        assert_eq!(e.worker_count(), 3);
        for i in 0..3u32 {
            assert_eq!(e.worker(WorkerId(i)).unwrap().device_ordinal(), i as usize);
        }
        assert!(e.prefix_tree().is_empty());
    }

    #[test]
    fn nine_sessions_arriving_at_once_still_spread_evenly() {
        // Routing reads every worker's load and then admits to the cheapest
        // one. Once the engine stopped being one lock, nothing made that
        // pair atomic, and concurrent handlers all scored the same idle
        // fleet and all chose the same card. On hardware that showed up as
        // 4/2/3 placements for nine simultaneous 64K sessions, the fourth
        // session on the oversubscribed card waiting for a free slot: a
        // 150 s first token against 113 s for its neighbours, and 269 s on a
        // worse split.
        //
        // The barrier is what gives this teeth. Nine threads spawned in a
        // loop finish microseconds apart and never overlap, so the race
        // needs them released together; repeating the round then turns a
        // possible interleaving into a near-certain one. Without the
        // routing lock this fails in the first round or two.
        for round in 0..256 {
            let e = engine(4096);
            let gate = std::sync::Barrier::new(9);
            std::thread::scope(|scope| {
                for id in 0..9u64 {
                    let (e, gate) = (&e, &gate);
                    scope.spawn(move || {
                        gate.wait();
                        e.place(req(id, 256, 16), &[])
                            .expect("nine small sessions fit three workers")
                    });
                }
            });
            let placed: Vec<usize> = (0..3)
                .map(|i| e.worker(WorkerId(i)).unwrap().scheduler().waiting_len())
                .collect();
            assert_eq!(placed, vec![3, 3, 3], "round {round}: placements piled up");
        }
    }

    #[test]
    fn workers_are_independent_apart_from_the_shared_tree() {
        // Admitting on one worker must not consume another's capacity. This
        // is the property that makes routing a cost decision rather than a
        // correctness one.
        let e = engine(1024);
        let before: Vec<_> = (0..3)
            .map(|i| e.worker(WorkerId(i)).unwrap().kv_utilization())
            .collect();

        e.worker(WorkerId(0))
            .unwrap()
            .admit(req(1, 2048, 256))
            .unwrap();

        let after: Vec<_> = (0..3)
            .map(|i| e.worker(WorkerId(i)).unwrap().kv_utilization())
            .collect();
        assert_eq!(before[1], after[1]);
        assert_eq!(before[2], after[2]);
    }

    #[test]
    fn a_request_larger_than_any_pool_is_refused_rather_than_placed() {
        // Every worker refuses, and the caller is told which of the
        // scheduler's reasons applied. `AllWorkersSaturated` on its own sent
        // a 503 saying only "every worker refused admission", which is what
        // a person then has to reverse-engineer from the flags.
        let e = engine(4); // 4 blocks * 256 = 1024 tokens total
        let err = e.place(req(1, 100_000, 1024), &[]).unwrap_err();
        match err {
            PlacementError::Admission(AdmissionError::ExceedsTotalCapacity {
                total_blocks,
                needed_blocks,
                ..
            }) => {
                assert_eq!(total_blocks, 4);
                assert!(needed_blocks > total_blocks);
            }
            other => panic!("a refusal must name its reason, got {other:?}"),
        }
    }

    #[test]
    fn a_speculative_output_ceiling_is_capped_rather_than_refused() {
        // 1024 blocks across 3 slots is 341 blocks each, so one slot may
        // reserve 87,296 tokens. A caller asking for 300,000 output tokens
        // on a 1,000-token prompt used to be refused outright, because
        // admission reserves prompt + max_output in full and 301,000 tokens
        // needs 1,176 blocks of a 1,024-block pool. It is now capped to the
        // slot's share and admitted; if it really generates that far it
        // stops there and reports `length`.
        let e = engine(1024);
        assert!(
            e.place(req(1, 1_000, 300_000), &[]).is_ok(),
            "a speculative output ceiling must be capped, not refused"
        );
    }

    #[test]
    fn placement_reports_where_it_went_and_why() {
        let e = engine(1024);
        let p = e.place(req(1, 2048, 256), &[]).unwrap();
        assert_eq!(p.request, RequestId(1));
        assert!(p.score.is_finite());
        // Empty tree, so nothing was reusable.
        assert_eq!(p.reusable_prefix_tokens, 0);
        assert!(
            e.worker(p.worker)
                .unwrap()
                .scheduler()
                .is_waiting(RequestId(1))
        );
    }

    #[test]
    fn an_empty_prefix_tree_still_routes() {
        // Cold start must work: with no cache anywhere, the load terms decide.
        let e = engine(1024);
        for i in 0..3 {
            assert!(e.place(req(i, 512, 128), &[]).is_ok());
        }
    }

    #[test]
    fn capacity_is_reported_per_group_never_summed() {
        let e = engine(1024);
        let cap = e.worker(WorkerId(0)).unwrap().capacity();
        assert_eq!(cap.attention_total_blocks, 1024);
        assert_eq!(cap.gdn_total_slots, 3);
        // The GDN page must not have been padded to the attention page.
        assert_ne!(cap.gdn_bytes_per_slot, 0);
        assert!(
            cap.gdn_bytes_per_slot > u64::from(cap.attention_block_size) * 20_480 / 4,
            "GDN slot must remain its own natural size"
        );
    }
    #[test]
    fn a_generating_sequence_grows_a_name_for_the_snapshots_it_reaches() {
        // The limitation this replaced: a 20-token prompt yields no complete
        // block, so the snapshot at 2048 had nothing to be filed under and
        // the reply was never shareable. Now the chain grows with the reply,
        // so the boundary the sequence generates through is named by the time
        // it gets there.
        let cache = CacheConfig::with_defaults(ModelConfig::qwen3_6_35b_a3b()).unwrap();
        let block = cache.attention_block_size();
        let interval = cache.gdn_retention_interval() as usize;

        let mut chain = SequenceChain::new(block, &[7; 20], &[]);
        assert_eq!(chain.hashes_for(interval), None, "the prompt is 20 tokens");

        chain.extend((0..interval as i32).map(|token| token % 1000));
        assert_eq!(
            chain.hashes_for(interval).map(<[u64]>::len),
            Some(interval / block as usize),
            "generating past the boundary names every block below it"
        );
    }

    #[test]
    fn cancellation_removes_a_waiting_request_from_its_worker() {
        let e = engine(1024);
        let hash = 11;
        e.prefix_tree.insert(&[PrefixBlock {
            hash,
            block: BlockId(0),
            gdn_snapshot: None,
        }]);
        let placement = e.place(req(77, 2048, 16), &[]).unwrap();
        e.prefix_tree.incr_ref_chain(&[hash]);
        e.request_refs
            .lock()
            .insert((placement.worker, placement.request), vec![hash]);
        assert_eq!(e.prefix_tree.ref_count(hash), 1);
        assert!(
            e.worker(placement.worker)
                .unwrap()
                .scheduler()
                .is_waiting(placement.request)
        );
        assert!(e.cancel(placement.request));
        assert_eq!(e.prefix_tree.ref_count(hash), 0);
        assert!(
            !e.worker(placement.worker)
                .unwrap()
                .scheduler()
                .is_waiting(placement.request)
        );
        assert!(!e.cancel(placement.request));
    }
}
