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
//! # Not yet bound to a device
//!
//! This type currently owns the host-side half of a worker: the two-group
//! cache pools and the scheduler. It records which GPU it is *for*
//! ([`Worker::device_ordinal`]) but does not create a CUDA context, load
//! weights, or launch anything. The device half arrives with the kernels.

use xabe_cache::config::{CacheConfig, CapacityReport};
use xabe_cache::pool::BlockPool;
use xabe_sched::config::SchedulerConfig;
use xabe_sched::error::AdmissionError;
use xabe_sched::request::{BatchDescription, NewRequest, RequestId};
use xabe_sched::scheduler::Scheduler;

use crate::router::WorkerLoad;

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

        Self {
            id,
            device_ordinal,
            cache,
            attention_pool,
            gdn_pool,
            scheduler,
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
            can_admit: self.scheduler.can_admit(req),
        }
    }

    /// Admit a request onto this worker.
    pub fn admit(&mut self, req: NewRequest) -> Result<RequestId, AdmissionError> {
        self.scheduler.admit(req)
    }

    /// Advance one scheduling step.
    pub fn step(&mut self) -> BatchDescription {
        self.scheduler.step()
    }

    /// Release a completed request's reservations.
    pub fn finish(&mut self, id: RequestId) -> bool {
        self.scheduler.finish_request(id)
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
            .finish()
    }
}
