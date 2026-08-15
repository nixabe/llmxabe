//! A single contiguous device allocation, sub-allocated by bumping.
//!
//! Model weights are ~29.6 GiB across 733 tensors. Allocating each tensor
//! separately would work, but one arena is preferable for three reasons:
//!
//! 1. **Fragmentation.** The KV pool and the recurrent-state pool are
//!    allocated after the weights and are far larger than any single tensor.
//!    733 independent allocations leave the driver's allocator in a state
//!    where a later 7.5 GiB request can fail on a card that has 7.5 GiB free.
//! 2. **Address stability.** Graph capture records device pointers. Weights
//!    are immutable for the process lifetime, so their addresses should be
//!    fixed once and never revisited.
//! 3. **One measurable number.** `mem_get_info` before and after a single
//!    allocation attributes VRAM to the arena exactly, with no per-allocation
//!    driver padding to reason about.
//!
//! This module holds no model knowledge — it does not know what a layer or an
//! expert is. See `xabe_engine::weights` for what puts tensors into it.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DriverError};

/// Alignment applied to every sub-allocation, in bytes.
///
/// 256 is CUDA's natural alignment: `cuMemAlloc` returns 256-byte-aligned
/// base pointers, and coalesced 128-byte global loads want tensor rows to
/// start on that boundary. Quantized block sizes do not divide it — a Q6_K
/// superblock is 210 bytes — so without explicit padding every tensor after
/// the first would start at an arbitrary offset.
pub const ALIGNMENT: usize = 256;

/// Something that went wrong allocating or filling the arena.
#[derive(Debug)]
pub enum ArenaError {
    /// The driver refused the allocation, or a copy failed.
    Driver(DriverError),
    /// A sub-allocation did not fit in the remaining capacity.
    ///
    /// Carries enough to say by how much, because "out of memory" without a
    /// number is not actionable.
    Exhausted {
        requested: usize,
        remaining: usize,
        capacity: usize,
    },
    /// An upload's byte count did not match its reservation.
    ///
    /// A short write would leave stale bytes in a tensor and produce plausible
    /// but wrong output, so it is rejected rather than truncated.
    LengthMismatch { reserved: usize, provided: usize },
}

impl std::fmt::Display for ArenaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::Exhausted {
                requested,
                remaining,
                capacity,
            } => write!(
                f,
                "arena exhausted: requested {requested} B with {remaining} B left of {capacity} B",
            ),
            Self::LengthMismatch { reserved, provided } => {
                write!(f, "upload of {provided} B into a {reserved} B reservation",)
            }
        }
    }
}

impl std::error::Error for ArenaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Driver(e) => Some(e),
            _ => None,
        }
    }
}

impl From<DriverError> for ArenaError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

/// A reserved byte range within an arena.
///
/// Copyable and cheap: it is an offset and a length, not a handle. The arena
/// owns the memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Allocation {
    /// Byte offset from the arena base. Always a multiple of [`ALIGNMENT`].
    pub offset: usize,
    /// Usable length in bytes, excluding trailing alignment padding.
    pub len: usize,
}

impl Allocation {
    /// The half-open byte range this allocation covers.
    pub fn range(&self) -> std::ops::Range<usize> {
        self.offset..self.offset + self.len
    }
}

/// One contiguous device allocation with bump sub-allocation.
pub struct DeviceArena {
    slab: CudaSlice<u8>,
    used: usize,
    capacity: usize,
}

impl DeviceArena {
    /// Reserve `capacity` bytes on the device backing `stream`.
    ///
    /// The allocation is not zeroed. Zeroing 29.6 GiB costs a full pass over
    /// device memory for no benefit, since every byte is overwritten by an
    /// upload before it is read — and a tensor that is *not* overwritten is a
    /// bug that zeroing would hide behind plausible-looking output.
    pub fn new(stream: &Arc<CudaStream>, capacity: usize) -> Result<Self, ArenaError> {
        // SAFETY: the contents are uninitialised, which is sound for `u8` —
        // it has no invalid bit patterns. Reading before writing yields
        // arbitrary bytes, not undefined behaviour.
        let slab = unsafe { stream.alloc::<u8>(capacity) }?;
        Ok(Self {
            slab,
            used: 0,
            capacity,
        })
    }

    /// Total capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes handed out so far, alignment padding included.
    pub fn used(&self) -> usize {
        self.used
    }

    /// Bytes still available.
    pub fn remaining(&self) -> usize {
        self.capacity - self.used
    }

    /// Reserve `len` bytes, aligned to [`ALIGNMENT`].
    pub fn reserve(&mut self, len: usize) -> Result<Allocation, ArenaError> {
        let offset = self.used;
        let padded = len.next_multiple_of(ALIGNMENT);
        if padded > self.remaining() {
            return Err(ArenaError::Exhausted {
                requested: padded,
                remaining: self.remaining(),
                capacity: self.capacity,
            });
        }
        self.used += padded;
        Ok(Allocation { offset, len })
    }

    /// Reserve space for `bytes` and copy it to the device.
    pub fn push(
        &mut self,
        stream: &Arc<CudaStream>,
        bytes: &[u8],
    ) -> Result<Allocation, ArenaError> {
        let alloc = self.reserve(bytes.len())?;
        self.write(stream, &alloc, bytes)?;
        Ok(alloc)
    }

    /// Copy `bytes` into an existing reservation.
    ///
    /// The length must match exactly — see [`ArenaError::LengthMismatch`].
    pub fn write(
        &mut self,
        stream: &Arc<CudaStream>,
        alloc: &Allocation,
        bytes: &[u8],
    ) -> Result<(), ArenaError> {
        if bytes.len() != alloc.len {
            return Err(ArenaError::LengthMismatch {
                reserved: alloc.len,
                provided: bytes.len(),
            });
        }
        let mut view = self.slab.slice_mut(alloc.range());
        stream.memcpy_htod(bytes, &mut view)?;
        Ok(())
    }

    /// Read an allocation back from the device.
    ///
    /// This is a diagnostic and verification path, not a hot path: it
    /// synchronises and allocates on the host.
    pub fn read(
        &self,
        stream: &Arc<CudaStream>,
        alloc: &Allocation,
    ) -> Result<Vec<u8>, ArenaError> {
        let view = self.slab.slice(alloc.range());
        let mut out = vec![0u8; alloc.len];
        stream.memcpy_dtoh(&view, out.as_mut_slice())?;
        stream.synchronize()?;
        Ok(out)
    }

    /// The whole slab, for handing device pointers to kernels.
    pub fn slab(&self) -> &CudaSlice<u8> {
        &self.slab
    }
}

/// Free and total device memory, in bytes, for the device backing `ctx`.
///
/// This is the only honest way to measure what an allocation actually cost:
/// the driver's own accounting, including its context overhead and any
/// rounding it applies.
pub fn memory_info(ctx: &Arc<CudaContext>) -> Result<(u64, u64), DriverError> {
    ctx.bind_to_thread()?;
    let (free, total) = ctx.mem_get_info()?;
    Ok((free as u64, total as u64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservations_are_aligned_and_do_not_overlap() {
        // Exercised without a device: bump arithmetic is the part that can be
        // wrong in a way a GPU would not catch, because an overlapping
        // reservation silently corrupts a neighbouring tensor.
        let mut used = 0usize;
        let mut reserve = |len: usize| {
            let offset = used;
            used += len.next_multiple_of(ALIGNMENT);
            Allocation { offset, len }
        };

        // 210 bytes is one Q6_K superblock — deliberately coprime to 256.
        let a = reserve(210);
        let b = reserve(1);
        let c = reserve(256);

        for alloc in [a, b, c] {
            assert_eq!(alloc.offset % ALIGNMENT, 0, "{alloc:?} is misaligned");
        }
        assert!(a.range().end <= b.offset, "{a:?} overlaps {b:?}");
        assert!(b.range().end <= c.offset, "{b:?} overlaps {c:?}");
        assert_eq!(c.offset, 512);
    }

    #[test]
    fn exhaustion_reports_the_shortfall() {
        // The arena cannot be built without a device, so this checks the
        // arithmetic the error carries rather than the allocation itself.
        let capacity = 1024usize;
        let used = 900usize;
        let requested = 256usize;
        let err = ArenaError::Exhausted {
            requested,
            remaining: capacity - used,
            capacity,
        };
        let text = err.to_string();
        assert!(text.contains("256"), "{text}");
        assert!(text.contains("124"), "{text}");
    }

    #[test]
    fn a_short_upload_is_rejected_rather_than_truncated() {
        let err = ArenaError::LengthMismatch {
            reserved: 210,
            provided: 200,
        };
        assert!(err.to_string().contains("210"));
    }
}
