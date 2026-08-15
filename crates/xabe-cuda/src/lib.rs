//! CUDA device access for the llmxabe engine.
//!
//! This crate owns the boundary between Rust and the CUDA driver: device
//! discovery, the sm_75 capability gate, kernel compilation through NVRTC, and
//! graph capture. It holds no model knowledge — it does not know what a layer
//! or an expert is, and it should stay that way.
//!
//! ## Toolchain
//!
//! The engine uses **cudarc + NVRTC**, with CUDA C++ kernel sources compiled at
//! load time. That choice was gated on the spike in [`spike`], which runs on
//! the target hardware and verifies sm_75 PTX emission, inline-PTX
//! availability, warp intrinsic lowering, and CUDA graph capture. See
//! `docs/TOOLCHAIN.md` for the reasoning and the alternative that was
//! rejected.
//!
//! ## Testing without a GPU
//!
//! Everything here degrades to a reported skip when no driver or device is
//! present, so the workspace stays testable on a laptop. A skip is reported
//! distinctly from a pass — see [`spike::CheckOutcome`]. Do not read a green
//! test run on a GPU-less machine as validation of device work.

pub mod arena;
pub mod device;
pub mod kernels;
pub mod spike;

pub use arena::{ALIGNMENT, Allocation, ArenaError, DeviceArena, memory_info};
pub use device::{
    ComputeCapability, DeviceInfo, GateFailure, MIN_COMPUTE_CAPABILITY, check_gate,
    driver_available, probe_all,
};
pub use kernels::dequant::{DequantError, Dequantizer};
pub use kernels::gdn::{GdnError, GdnKernels, GdnScratch};
pub use spike::{CheckOutcome, SpikeReport, TOOLCHAIN_DECISION};
