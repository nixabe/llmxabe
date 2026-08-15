//! CUDA kernels, compiled from CUDA C++ at load time through NVRTC.
//!
//! Every kernel here has a scalar fp32 counterpart in `xabe-kernels` and a
//! differential test that checks one against the other. Per `AGENTS.md`, a
//! kernel without a passing differential test is not done regardless of how
//! fast it runs, so the module layout mirrors `xabe-kernels`' own.
//!
//! Sources are `&'static str` constants rather than `.cu` files so that the
//! kernel and the Rust that launches it cannot drift out of the same commit,
//! and so a build needs no CUDA toolkit — only a driver at runtime.

pub mod attention;
pub mod dequant;
pub mod gdn;
pub mod gdn_chunked;
pub mod layer_ops;
pub mod lm_head;
pub mod moe;

use cudarc::nvrtc::{CompileOptions, Ptx};

/// The only architecture this project targets.
///
/// Quadro RTX 8000 is Turing. Compiling for anything newer produces PTX the
/// driver will refuse, and compiling for something older silently gives up
/// the `m16n8k8` tensor-core path — see `docs/TOOLCHAIN.md`.
pub const TARGET_ARCH: &str = "compute_75";

/// Compile a CUDA C++ source to PTX for [`TARGET_ARCH`].
///
/// `name` appears in NVRTC's diagnostics, so it should identify the kernel
/// rather than the caller.
pub fn compile(src: &str, name: &str) -> Result<Ptx, String> {
    cudarc::nvrtc::compile_ptx_with_opts(
        src,
        CompileOptions {
            arch: Some(TARGET_ARCH),
            name: Some(name.to_string()),
            ..Default::default()
        },
    )
    .map_err(|e| format!("{name}: {e:?}"))
}
