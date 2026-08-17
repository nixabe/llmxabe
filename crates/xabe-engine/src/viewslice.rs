//! A zero-copy, zero-allocation sub-range view into an existing device
//! buffer.
//!
//! # Why this exists
//!
//! Batched decode packs `N` sequences' one-token activations into a single
//! `[N][width]` scratch buffer so the weight-bound projections can read that
//! buffer once and amortize the weight fetch across all `N` sequences — see
//! `crate::forward::Forward::run_batch_decode`. Two steps in the Gated
//! DeltaNet mixer do not amortize at all: the causal convolution and the
//! delta-rule recurrent update both read and write **one sequence's own
//! state**, so they must run once per sequence rather than once over the
//! batch. Those calls still need a `&CudaSlice<f32>` naming just that
//! sequence's row of the batch buffer, and every kernel wrapper in this
//! workspace takes `&CudaSlice<T>` by reference to a whole allocation, not a
//! byte range within one — the same obstacle
//! [`crate::weights::ResidentTensor`] exists to work around for the weight
//! arena.
//!
//! This is that same technique, applied to scratch instead of weights:
//! [`CudaSlice::device_ptr`] gets the allocation's own base address, and
//! [`CudaStream::upgrade_device_ptr`] wraps an offset into it back into a
//! plain `CudaSlice`, sealed in a [`ManuallyDrop`] so it is never freed for
//! real. Nothing is copied and nothing is allocated.
//!
//! # Why a view rather than widening every wrapper to take a range
//!
//! The alternative is a prefix-tolerant `offset` parameter threaded through
//! every kernel wrapper this file's callers use — `conv1d`, `GdnKernels::step`,
//! `GatedAttentionBlock::forward`'s eleven internal calls, and so on. That
//! would touch two crates' worth of already-gated kernel signatures for a
//! capability only the batched-decode path needs. A view is the same
//! zero-cost pointer arithmetic without widening anyone else's contract.

use std::mem::ManuallyDrop;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};

/// A view of `[offset, offset + len)` elements of `base`, usable anywhere a
/// kernel wrapper wants `&CudaSlice<T>` or `&mut CudaSlice<T>`.
///
/// Deref/DerefMut to `CudaSlice<T>`, and drops without freeing anything: the
/// inner slice is sealed in a [`ManuallyDrop`].
///
/// # Safety
///
/// `offset + len` must be at most `base.len()`, and the view must not outlive
/// `base`.
pub(crate) unsafe fn subslice<T>(
    stream: &Arc<CudaStream>,
    base: &CudaSlice<T>,
    offset: usize,
    len: usize,
) -> ManuallyDrop<CudaSlice<T>> {
    debug_assert!(
        offset + len <= base.len(),
        "subslice [{offset}, {}) is out of bounds for a {}-element buffer",
        offset + len,
        base.len(),
    );
    let (ptr, _sync) = base.device_ptr(stream);
    let byte_offset = (offset * size_of::<T>()) as u64;
    // SAFETY: the caller guarantees `offset + len <= base.len()`, so
    // `[ptr + byte_offset, ptr + byte_offset + len * size_of::<T>())` is
    // inside `base`'s own allocation. The result is immediately sealed in a
    // `ManuallyDrop`, so dropping it never asks the driver to free a pointer
    // into the middle of (or the whole of) `base`'s allocation.
    ManuallyDrop::new(unsafe { stream.upgrade_device_ptr::<T>(ptr + byte_offset, len) })
}

#[cfg(test)]
mod tests {
    // Exercised end-to-end by `tests/batch_decode.rs`, which needs a device;
    // there is nothing to check about pointer arithmetic without one.
}
