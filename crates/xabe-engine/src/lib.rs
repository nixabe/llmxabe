//! The llmxabe engine: three workers, one router, one shared prefix cache.
//!
//! Three GPUs each hold a complete copy of the model, so any request can be
//! served by any worker and the answer is identical. Workers never touch each
//! other's device memory — no P2P, no NCCL, nothing crosses PCIe on the decode
//! path.
//!
//! What makes them one engine rather than three servers lives entirely in this
//! crate: one admission path ([`Engine::place`]), one scoring function
//! ([`router`]), and one prefix radix tree shared across all three
//! ([`Engine::prefix_tree`]). That last one is the project's only unqualified
//! architectural claim — see `docs/ARCHITECTURE.md`.
//!
//! Workers retain a host-only construction path so scheduling and routing stay
//! testable without CUDA. [`Worker::bind_device`] adds the serving runtime and
//! [`Worker::step_device`] executes scheduler batches on that worker's card.

pub mod block;
pub mod engine;
pub mod forward;
pub(crate) mod prefix;
pub mod router;
pub mod runtime;
pub mod sampling;
pub mod speculative;
pub mod state;
pub(crate) mod viewslice;
pub mod vision;
pub mod weights;
pub mod worker;

pub use engine::{Engine, EngineExecutionError, Placement, PlacementError};
pub use router::{Routed, RouterConfig, RoutingError, WorkerLoad, route};
pub use runtime::{
    DEFAULT_SNAPSHOT_SLOTS_PER_WORKER, DeviceRuntime, DeviceStep, RuntimeConfig, RuntimeError,
};
pub use sampling::{Sampler, SamplingParams};
pub use state::{
    SequenceSnapshot, SequenceState, SnapshotSlots, StateError, snapshot_bytes_per_slot,
};
pub use weights::{DeviceWeights, LoadError, LoadReport, TensorPlacement};
pub use worker::{ServingConfig, Speculation, Worker, WorkerExecutionError, WorkerId};
