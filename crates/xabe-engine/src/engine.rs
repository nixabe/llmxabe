//! The engine: three workers, one router, one shared prefix tree.
//!
//! This is the type that makes three GPUs one engine rather than three
//! independent servers. The whole of that difference lives here and in
//! [`crate::router`] — below this layer, workers are entirely independent and
//! never touch each other's device memory.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use parking_lot::RwLock;
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
    workers: Vec<Worker>,
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
    request_chains: HashMap<(WorkerId, RequestId), SequenceChain>,
    request_refs: HashMap<(WorkerId, RequestId), Vec<BlockHash>>,
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
                Worker::new(
                    WorkerId(i as u32),
                    ordinal,
                    cache.clone(),
                    sched,
                    attention_blocks_per_worker,
                    gdn_slots_per_worker,
                )
            })
            .collect();

        Self {
            workers,
            prefix_tree,
            router,
            snapshots: RwLock::new(SharedSnapshots::default()),
            request_chains: HashMap::new(),
            request_refs: HashMap::new(),
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

    /// Read-only access to a worker.
    pub fn worker(&self, id: WorkerId) -> Option<&Worker> {
        self.workers.get(id.0 as usize)
    }

    /// Mutable access to a worker.
    pub fn worker_mut(&mut self, id: WorkerId) -> Option<&mut Worker> {
        self.workers.get_mut(id.0 as usize)
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
            .map(|w| w.load_for(usable, req))
            .collect()
    }

    /// Route a request and admit it onto the chosen worker.
    ///
    /// `block_hashes` are the chained block hashes of the request's prompt,
    /// as produced by [`xabe_cache::radix::hash_block`].
    pub fn place(
        &mut self,
        req: NewRequest,
        block_hashes: &[BlockHash],
    ) -> Result<Placement, PlacementError> {
        let budget = self
            .workers
            .first()
            .map(|w| w.scheduler().config().token_budget())
            .unwrap_or(0);

        let loads = self.score_workers(&req, block_hashes);
        let Routed {
            worker,
            score,
            matched_tokens,
        } = route(&self.router, &loads, req.prompt_tokens, budget)
            .map_err(PlacementError::Routing)?;

        let request = self
            .worker_mut(worker)
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
    pub fn bind_devices(
        &mut self,
        model_path: &Path,
        model: ModelConfig,
        serving: ServingConfig,
    ) -> Result<(), (WorkerId, RuntimeError)> {
        for worker in &mut self.workers {
            if let Err(error) = worker.bind_device(model_path, model.clone(), serving) {
                return Err((worker.id(), error));
            }
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
        &mut self,
        req: NewRequest,
        prompt: Vec<i32>,
        sampling: SamplingParams,
    ) -> Result<Placement, EngineExecutionError> {
        let budget = self
            .workers
            .first()
            .map(|worker| worker.scheduler().config().token_budget())
            .unwrap_or(0);
        let chain = SequenceChain::new(self.block_size, &prompt);
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
            .map_err(|error| EngineExecutionError::Placement(PlacementError::Routing(error)))?;
        let request = if let Some(snapshot) = snapshot {
            self.worker_mut(worker)
                .expect("router returned an existing worker")
                .admit_tokens_restored(req, prompt, snapshot, sampling)
        } else {
            self.worker_mut(worker)
                .expect("router returned an existing worker")
                .admit_tokens(req, prompt, sampling)
        }
        .map_err(|source| EngineExecutionError::Worker { worker, source })?;
        let referenced = matched_tokens as usize / self.block_size as usize;
        if referenced > 0 {
            let hashes = chain.hashes()[..referenced].to_vec();
            self.prefix_tree.incr_ref_chain(&hashes);
            self.request_refs.insert((worker, request), hashes);
        }
        self.request_chains.insert((worker, request), chain);
        Ok(Placement {
            worker,
            request,
            reusable_prefix_tokens: matched_tokens,
            score,
        })
    }

    /// Cancel a live request and release its scheduler, runtime, and cache
    /// bookkeeping regardless of whether it is waiting or running.
    pub fn cancel(&mut self, request: RequestId) -> bool {
        let Some(worker) = self.workers.iter().find_map(|worker| {
            (worker.scheduler().is_waiting(request) || worker.scheduler().is_running(request))
                .then_some(worker.id())
        }) else {
            return false;
        };
        self.request_chains.remove(&(worker, request));
        if let Some(hashes) = self.request_refs.remove(&(worker, request)) {
            self.prefix_tree.decr_ref_chain(&hashes);
        }
        self.worker_mut(worker)
            .is_some_and(|worker| worker.cancel(request))
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
        let source = self
            .worker(worker)
            .ok_or(EngineExecutionError::MissingWorker(worker))?;
        let position = snapshot.position();
        let interval = source.cache_config().gdn_retention_interval();
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
        if let Some(slots) = source.snapshot_slots() {
            let slots = slots.clone();
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

    /// Execute one scheduler step on every device-bound worker concurrently.
    pub fn step_devices(
        &mut self,
    ) -> Result<SmallVec<[(WorkerId, DeviceStep); 3]>, EngineExecutionError> {
        let mut steps: SmallVec<[(WorkerId, DeviceStep); 3]> = std::thread::scope(|scope| {
            let handles: SmallVec<[_; 3]> = self
                .workers
                .iter_mut()
                .filter(|worker| worker.is_device_bound())
                .map(|worker| {
                    let id = worker.id();
                    (id, scope.spawn(move || worker.step_device()))
                })
                .collect();
            handles
                .into_iter()
                .map(|(worker, handle)| match handle.join() {
                    Ok(Ok(step)) => Ok((worker, step)),
                    Ok(Err(source)) => Err(EngineExecutionError::Worker { worker, source }),
                    Err(_) => Err(EngineExecutionError::WorkerPanicked(worker)),
                })
                .collect::<Result<SmallVec<[_; 3]>, _>>()
        })?;
        for (worker, step) in &mut steps {
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
            for (request, token) in &step.generated {
                if let Some(chain) = self.request_chains.get_mut(&(*worker, *request)) {
                    chain.push(*token);
                }
            }
            for (request, snapshot) in std::mem::take(&mut step.retained) {
                if let Some(chain) = self.request_chains.get(&(*worker, request)) {
                    self.install_snapshot(*worker, snapshot, chain)?;
                }
            }
            for request in &step.completed {
                self.request_chains.remove(&(*worker, *request));
                if let Some(hashes) = self.request_refs.remove(&(*worker, *request)) {
                    self.prefix_tree.decr_ref_chain(&hashes);
                }
            }
        }
        Ok(steps)
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
    fn workers_are_independent_apart_from_the_shared_tree() {
        // Admitting on one worker must not consume another's capacity. This
        // is the property that makes routing a cost decision rather than a
        // correctness one.
        let mut e = engine(1024);
        let before: Vec<_> = (0..3)
            .map(|i| e.worker(WorkerId(i)).unwrap().kv_utilization())
            .collect();

        e.worker_mut(WorkerId(0))
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
        // Every worker refuses, so this is saturation, not a routing bug.
        let mut e = engine(4); // 4 blocks * 256 = 1024 tokens total
        let err = e.place(req(1, 100_000, 1024), &[]).unwrap_err();
        assert_eq!(
            err,
            PlacementError::Routing(RoutingError::AllWorkersSaturated),
            "an over-large request must be reported as saturation so the \
             caller queues rather than crashes"
        );
    }

    #[test]
    fn placement_reports_where_it_went_and_why() {
        let mut e = engine(1024);
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
        let mut e = engine(1024);
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

        let mut chain = SequenceChain::new(block, &[7; 20]);
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
        let mut e = engine(1024);
        let hash = 11;
        e.prefix_tree.insert(&[PrefixBlock {
            hash,
            block: BlockId(0),
            gdn_snapshot: None,
        }]);
        let placement = e.place(req(77, 2048, 16), &[]).unwrap();
        e.prefix_tree.incr_ref_chain(&[hash]);
        e.request_refs
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
