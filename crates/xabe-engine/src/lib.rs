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
//! # Status
//!
//! Host-side only. Workers record which GPU they are for but do not yet create
//! a CUDA context, load weights, or launch kernels. Everything here is
//! deterministic and testable without a device, which is deliberate:
//! scheduling and routing policy should be debuggable on a laptop.

pub mod engine;
pub mod router;
pub mod worker;

pub use engine::{Engine, Placement, PlacementError};
pub use router::{Routed, RouterConfig, RoutingError, WorkerLoad, route};
pub use worker::{Worker, WorkerId};
