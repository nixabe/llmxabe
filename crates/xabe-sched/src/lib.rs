//! Chunked prefill scheduler and admission control.
//!
//! This crate exists to enforce AGENTS.md rules 3 and 4, both paid for in
//! someone else's production incident:
//!
//! 3. [`config::SchedulerConfig::new`] rejects, as a typed
//!    [`error::SchedulerConfigError`], any configuration where
//!    `token_budget <= block_size + max_concurrent_decodes` — the boundary
//!    at which a single decoding request can starve all prefill admission
//!    and execution serializes to batch 1.
//! 4. [`scheduler::Scheduler::admit`] checks a request's full potential
//!    lifetime (`prompt_tokens + max_output_tokens`) against total
//!    attention capacity, not just what would fit in one step's chunk.
//!
//! [`scheduler::Scheduler::step`] is a pure function of scheduler state —
//! no device handles — so scheduling policy (decode-first batching, chunked
//! prefill, watermark-gated admission, recompute preemption, speculative
//! draft-token budgeting) is fully testable without a GPU.
//!
//! Start at [`config::SchedulerConfig`] for tunables and
//! [`scheduler::Scheduler`] for the scheduler itself.

pub mod config;
pub mod error;
pub mod request;
pub mod scheduler;

pub use config::SchedulerConfig;
pub use error::{AdmissionError, SchedulerConfigError};
pub use request::{BatchDescription, DecodeItem, NewRequest, PrefillItem, RequestId};
pub use scheduler::Scheduler;
