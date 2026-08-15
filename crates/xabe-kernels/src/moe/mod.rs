//! Mixture-of-experts reference kernels: router top-k, block-aligned
//! dispatch, and the grouped-GEMM forward pass those tables drive.
//!
//! Read [`dispatch`] first — it's the piece that turns Qwen3.6's 1,080
//! scattered per-token GEMVs (`MoeConfig::naive_gemvs_per_token` in
//! `xabe-model`) into one grouped-GEMM launch, and everything else in this
//! module exists to produce its inputs or consume its outputs.

pub mod dispatch;
pub mod gemm;
pub mod router;

pub use dispatch::{MoeDispatch, moe_align_block_size};
pub use gemm::{ExpertWeights, grouped_forward, naive_forward};
pub use router::{RoutingDecision, route_batch, route_token};
