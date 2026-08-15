//! Structural description of Qwen3.6-35B-A3B and the resource budgets that
//! follow from it.
//!
//! This crate holds no state and touches no device. It answers questions of
//! the form "given this architecture, how many bytes does X cost" — VRAM
//! segmentation, per-token bandwidth, cache page geometry — so that every
//! other crate derives those numbers from one place rather than embedding
//! constants of its own.
//!
//! Start at [`ModelConfig`].

pub mod config;
pub mod verify;

pub use config::{AttentionConfig, GdnConfig, LayerKind, ModelConfig, MoeConfig};
