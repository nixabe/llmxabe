//! Structural description of the Qwen3.5-family models this engine serves,
//! and the resource budgets that follow from it.
//!
//! Two architectures are described: `qwen35moe` (Qwen3.6-35B-A3B, a routed
//! 256-expert MoE) and `qwen35` (Qwen3.8-27B, the dense sibling). They share
//! the hybrid Gated-DeltaNet/Gated-Attention layer pattern and differ in the
//! feed-forward block and in every width; see [`FfnConfig`].
//!
//! This crate holds no state and touches no device. It answers questions of
//! the form "given this architecture, how many bytes does X cost" — VRAM
//! segmentation, per-token bandwidth, cache page geometry — so that every
//! other crate derives those numbers from one place rather than embedding
//! constants of its own.
//!
//! Start at [`ModelConfig`].

pub mod budget;
pub mod config;
pub mod dflash;
pub mod gguf;
mod metadata;
pub mod verify;
pub mod vision;
pub mod weights;

pub use config::{
    AttentionConfig, DenseFfnConfig, FfnConfig, GdnConfig, LayerKind, ModelConfig, MoeConfig,
    UnknownArchitecture,
};
pub use dflash::DFlashConfig;
pub use vision::VisionConfig;
pub use weights::{Directory, Role, Section, TensorSpec, WeightError, WeightSchema};

pub use gguf::ConfigLoadError;
