//! One transformer block, assembled from the kernels in `xabe-cuda`.
//!
//! Qwen3.6 has three block shapes and this module has one file per shape:
//!
//! - [`gdn`] — Gated DeltaNet, 30 of the 40 layers (every layer whose index
//!   mod 4 is not 3).
//! - [`attention`] — Gated Attention, the other 10, plus the MTP head at
//!   block 40.
//! - [`moe`] — the 256-expert mixture, which sits on **every** block of both
//!   kinds, so it is a separate module rather than a branch inside them.
//!
//! Each module owns its whole shape: it constructs the kernels it needs,
//! resolves its own weights out of [`crate::DeviceWeights`] by
//! [`xabe_model::weights::Role`], and exposes a `forward` that takes a hidden
//! state and returns one. Nothing is shared between them but that convention,
//! which is deliberate — the three were written concurrently, and a shared
//! context struct would have been a merge conflict with no compensating
//! benefit at three call sites.
//!
//! # Gating
//!
//! Every block here is checked against real llama.cpp intermediates rather
//! than against a hand-derived expectation. `docs/ORACLE.md` describes the
//! capture; `crates/xabe-engine/tests/golden.rs` loads it. The waypoints
//! `attn_norm-N`, `attn_residual-N`, `attn_post_norm-N`, `ffn_out-N` and
//! `l_out-N` exist for all 40 blocks, and the mixer internals of blocks
//! 0/4/20 (GDN) and 3/39 (attention) are captured in full.
//!
//! That is what makes a wrong forward pass bisectable: a block whose output
//! diverges from `l_out-N` while its input matched `l_out-(N-1)` is the block
//! at fault, and its captured internals say which step inside it.

pub mod attention;
pub mod gdn;
pub mod moe;
