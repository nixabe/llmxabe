//! CPU reference kernels and the differential-testing harness that checks
//! everything else in this project against them.
//!
//! # Why this crate exists
//!
//! Per `AGENTS.md`: numerics drift is the highest-likelihood risk in this
//! project — the engine stays fluent while getting quietly worse, and no
//! throughput benchmark catches it. Every CUDA kernel written for
//! `xabe-cuda` gets validated against the reference here. A kernel without
//! a passing differential test against this crate is not done, regardless
//! of how fast it runs.
//!
//! # What is (and isn't) here
//!
//! These are **references, not production kernels**: every implementation
//! in this crate is scalar fp32, optimized for being obviously correct and
//! easy to read against its cited math, never for speed. In particular:
//!
//! - [`gdn`] — Gated DeltaNet, both the recurrent (decode) and chunked
//!   (prefill) forms, and the equivalence test between them. This is the
//!   single most important module in the crate: 30 of Qwen3.6's 40 layers
//!   are Gated DeltaNet, and it has no flash-attention-style reference
//!   implementation anywhere else to lean on. Start here.
//! - [`gemv`] — row-major scalar GEMV, the oracle for the LM head's
//!   248,320 x 2,048 matrix-vector product, plus the `argmax` the sampled
//!   token actually comes from.
//! - [`moe`] — router top-k, block-aligned dispatch
//!   (`moe_align_block_size`), and the grouped-GEMM forward pass.
//! - [`attention`] — causal GQA softmax attention, naive and
//!   online-softmax forms, cross-checked against each other.
//! - [`rope`] — partial rotary embedding (rotates 64 of 256 head
//!   dimensions), with a test that the untouched tail is byte-identical.
//! - [`norm`] — RMSNorm, SwiGLU, residual add.
//! - [`quant`] — Q6_K / Q8_0 dequantization, ported from llama.cpp's
//!   bit-unpacking, plus a round-trip quantizer for testing.
//! - [`compare`] — the differential-testing harness itself: metrics,
//!   named tolerance presets with documented rationale, and
//!   [`compare::assert_matches`], the API a future GPU differential test
//!   calls.
//! - [`rng`] — a small dependency-free deterministic RNG for reproducible
//!   test inputs.

pub mod attention;
pub mod compare;
pub mod conv;
pub mod f16;
pub mod gdn;
pub mod gemv;
pub mod mma;
pub mod moe;
pub mod mrope;
pub mod norm;
pub mod quant;
pub mod rng;
pub mod rope;
pub mod vision;
