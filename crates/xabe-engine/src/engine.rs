//! The engine: three workers, one router, one shared prefix tree.
//!
//! This is the type that makes three GPUs one engine rather than three
//! independent servers. The whole of that difference lives here and in
//! [`crate::router`] — below this layer, workers are entirely independent and
//! never touch each other's device memory.

use std::sync::Arc;

use xabe_cache::config::CacheConfig;
use xabe_cache::radix::{BlockHash, RadixTree};
use xabe_sched::config::SchedulerConfig;
use xabe_sched::error::AdmissionError;
use xabe_sched::request::{NewRequest, RequestId};

use crate::router::{Routed, RouterConfig, RoutingError, WorkerLoad, route};
use crate::worker::{Worker, WorkerId};

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
        let usable = matched.gdn_matched_tokens;
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
}

impl core::fmt::Debug for Engine {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Engine")
            .field("workers", &self.workers.len())
            .field("prefix_tree_nodes", &self.prefix_tree.len())
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
}
