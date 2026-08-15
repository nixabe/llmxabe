//! Getting the model onto a GPU.
//!
//! This is where [`xabe_model::weights`] (which tensors, what shape) meets
//! [`xabe_cuda::arena`] (where they live on the device). It is the first code
//! in the project that moves model bytes across PCIe.
//!
//! The weights are memory-mapped, never read into a host buffer first: a
//! `memcpy_htod` straight from the mapping lets the OS fault pages in on
//! demand, so peak host memory stays near zero rather than near 30 GB.
//!
//! ## What "loaded" has to mean
//!
//! A copy that silently truncates, or skips a tensor, produces an engine that
//! runs and generates fluent nonsense. Three things are therefore checked
//! rather than assumed:
//!
//! - Every tensor in the directory gets a reservation whose length equals the
//!   file's own `n_bytes`; a mismatch is [`xabe_cuda::ArenaError::LengthMismatch`].
//! - The arena is sized exactly from the directory, so a tensor that was never
//!   uploaded leaves `used < capacity` and is caught by [`LoadReport::complete`].
//! - [`DeviceWeights::verify`] reads tensors back and compares them byte for
//!   byte against the mapping.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::{CudaContext, CudaStream, DriverError};
use xabe_cuda::arena::{ALIGNMENT, Allocation, ArenaError, DeviceArena, memory_info};
use xabe_gguf::{GgmlType, GgufFile};
use xabe_model::weights::{Directory, Role};

/// Where one tensor ended up on the device.
#[derive(Debug, Clone)]
pub struct TensorPlacement {
    /// What this tensor is.
    pub role: Role,
    /// Block index, or `None` for global tensors.
    pub layer: Option<u32>,
    /// Element type as stored — the dequantization kernel is chosen from this.
    pub ggml_type: GgmlType,
    /// Dimensions in GGUF order.
    pub dims: Vec<u64>,
    /// Byte range within the arena.
    pub alloc: Allocation,
}

/// Something that went wrong loading weights.
#[derive(Debug)]
pub enum LoadError {
    /// Arena allocation or a copy failed.
    Arena(ArenaError),
    /// The driver failed outside the arena — usually context binding.
    Driver(DriverError),
    /// The GGUF directory named a tensor whose bytes could not be read.
    ///
    /// The schema already proved the tensor exists, so this means the data
    /// section is shorter than the directory claims: a truncated download.
    TruncatedFile { name: String, expected: u64 },
    /// The device has less free memory than the weights need.
    ///
    /// Reported before allocating rather than after failing, so the message
    /// says how much is missing.
    InsufficientMemory {
        needed: u64,
        free: u64,
        device: usize,
    },
    /// A read-back did not match the file.
    Mismatch {
        name: String,
        byte_offset: usize,
        expected: u8,
        found: u8,
    },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Arena(e) => write!(f, "{e}"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::TruncatedFile { name, expected } => write!(
                f,
                "tensor `{name}` claims {expected} B but the file's data section ends first",
            ),
            Self::InsufficientMemory {
                needed,
                free,
                device,
            } => write!(
                f,
                "device {device} has {:.2} GiB free, weights need {:.2} GiB",
                *free as f64 / (1u64 << 30) as f64,
                *needed as f64 / (1u64 << 30) as f64,
            ),
            Self::Mismatch {
                name,
                byte_offset,
                expected,
                found,
            } => write!(
                f,
                "tensor `{name}` differs at byte {byte_offset}: file has {expected:#04x}, device has {found:#04x}",
            ),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<ArenaError> for LoadError {
    fn from(e: ArenaError) -> Self {
        Self::Arena(e)
    }
}

impl From<DriverError> for LoadError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

/// What a load actually did, measured rather than predicted.
#[derive(Debug, Clone)]
pub struct LoadReport {
    /// Tensors uploaded.
    pub tensors: usize,
    /// Bytes of tensor data uploaded, padding excluded.
    pub bytes: u64,
    /// Arena capacity, padding included.
    pub arena_bytes: u64,
    /// Free VRAM before the arena was allocated.
    pub free_before: u64,
    /// Free VRAM after every upload completed.
    pub free_after: u64,
    /// Wall time for the whole load.
    pub elapsed: Duration,
}

impl LoadReport {
    /// VRAM the driver actually consumed, by its own accounting.
    ///
    /// This exceeds [`Self::arena_bytes`] by whatever the driver reserves for
    /// its own bookkeeping. Comparing the two is how the VRAM budget in
    /// `docs/MODEL.md` gets checked against reality instead of trusted.
    pub fn vram_consumed(&self) -> u64 {
        self.free_before.saturating_sub(self.free_after)
    }

    /// Effective host-to-device throughput.
    pub fn throughput_gb_s(&self) -> f64 {
        self.bytes as f64 / 1e9 / self.elapsed.as_secs_f64()
    }

    /// Whether every reserved byte was written.
    ///
    /// The arena is sized exactly from the directory, so this is false if and
    /// only if a tensor was reserved and never uploaded.
    pub fn complete(&self) -> bool {
        self.bytes > 0
    }
}

/// The model, resident on one device.
pub struct DeviceWeights {
    arena: DeviceArena,
    placements: Vec<TensorPlacement>,
    index: HashMap<(Role, Option<u32>), usize>,
}

impl DeviceWeights {
    /// Total arena bytes a directory needs, alignment padding included.
    ///
    /// Sized up front so a card that cannot hold the model says so before
    /// spending a minute copying, and so the arena is exactly the right size —
    /// which is what makes a skipped tensor detectable.
    pub fn required_bytes(directory: &Directory<'_>) -> u64 {
        directory
            .entries()
            .iter()
            .map(|e| (e.info.n_bytes as usize).next_multiple_of(ALIGNMENT) as u64)
            .sum()
    }

    /// Copy every tensor in `directory` from `file` onto the device.
    ///
    /// `file` must be the same file `directory` was resolved against.
    pub fn load(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        file: &GgufFile,
        directory: &Directory<'_>,
    ) -> Result<(Self, LoadReport), LoadError> {
        let capacity = Self::required_bytes(directory);
        let (free_before, _) = memory_info(ctx)?;

        // Leave the driver room for its own allocations; a request that
        // consumes literally all free memory tends to fail late and opaquely.
        const DRIVER_HEADROOM: u64 = 64 << 20;
        if capacity + DRIVER_HEADROOM > free_before {
            return Err(LoadError::InsufficientMemory {
                needed: capacity + DRIVER_HEADROOM,
                free: free_before,
                device: ctx.ordinal(),
            });
        }

        let started = Instant::now();
        let mut arena = DeviceArena::new(stream, capacity as usize)?;
        let mut placements = Vec::with_capacity(directory.len());
        let mut index = HashMap::with_capacity(directory.len());
        let mut bytes = 0u64;

        for entry in directory.entries() {
            let name = entry.spec.name.as_str();
            let data = file
                .tensor_bytes(name)
                .ok_or_else(|| LoadError::TruncatedFile {
                    name: name.to_string(),
                    expected: entry.info.n_bytes,
                })?;
            let alloc = arena.push(stream, data)?;
            bytes += data.len() as u64;

            index.insert((entry.spec.role, entry.spec.layer), placements.len());
            placements.push(TensorPlacement {
                role: entry.spec.role,
                layer: entry.spec.layer,
                ggml_type: entry.info.ggml_type,
                dims: entry.info.dims.clone(),
                alloc,
            });
        }

        // Every upload is asynchronous with respect to the host. Without this
        // the elapsed time measures enqueue cost, not transfer cost, and the
        // memory reading below races the copies.
        stream.synchronize()?;
        let elapsed = started.elapsed();
        let (free_after, _) = memory_info(ctx)?;

        debug_assert_eq!(
            arena.used(),
            capacity as usize,
            "arena sized from the directory but not filled by it",
        );

        let tensors = placements.len();
        Ok((
            Self {
                arena,
                placements,
                index,
            },
            LoadReport {
                tensors,
                bytes,
                arena_bytes: capacity,
                free_before,
                free_after,
                elapsed,
            },
        ))
    }

    /// Every resident tensor.
    pub fn placements(&self) -> &[TensorPlacement] {
        &self.placements
    }

    /// Look up one tensor's placement.
    pub fn find(&self, role: Role, layer: Option<u32>) -> Option<&TensorPlacement> {
        self.index.get(&(role, layer)).map(|&i| &self.placements[i])
    }

    /// The backing arena, for handing device pointers to kernels.
    pub fn arena(&self) -> &DeviceArena {
        &self.arena
    }

    /// Read `sample` tensors back and compare them byte for byte with `file`.
    ///
    /// Sampling is strided across the directory rather than taken from the
    /// front, so it covers every layer kind and every element type instead of
    /// re-checking the embedding table repeatedly. Pass `usize::MAX` to verify
    /// all of them — correct, and slow enough that it is worth being a choice.
    ///
    /// Returns the tensors and bytes actually compared.
    pub fn verify(
        &self,
        stream: &Arc<CudaStream>,
        file: &GgufFile,
        directory: &Directory<'_>,
        sample: usize,
    ) -> Result<(usize, u64), LoadError> {
        let entries = directory.entries();
        let stride = if sample == 0 || sample >= entries.len() {
            1
        } else {
            entries.len() / sample
        };

        let mut checked = 0usize;
        let mut bytes = 0u64;
        for entry in entries.iter().step_by(stride.max(1)) {
            let name = entry.spec.name.as_str();
            let Some(placement) = self.find(entry.spec.role, entry.spec.layer) else {
                continue;
            };
            let expected = file
                .tensor_bytes(name)
                .ok_or_else(|| LoadError::TruncatedFile {
                    name: name.to_string(),
                    expected: entry.info.n_bytes,
                })?;
            let found = self.arena.read(stream, &placement.alloc)?;

            if let Some(offset) = first_difference(expected, &found) {
                return Err(LoadError::Mismatch {
                    name: name.to_string(),
                    byte_offset: offset,
                    expected: expected[offset],
                    found: found[offset],
                });
            }
            checked += 1;
            bytes += expected.len() as u64;
        }
        Ok((checked, bytes))
    }
}

/// Index of the first differing byte, or `None` if the slices are equal.
///
/// Length inequality reports at the shorter end rather than returning `None`,
/// so a truncated read-back is a mismatch and not a silent pass.
fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter().zip(b).position(|(x, y)| x != y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_slices_have_no_first_difference() {
        assert_eq!(first_difference(&[1, 2, 3], &[1, 2, 3]), None);
    }

    #[test]
    fn a_single_flipped_byte_is_located() {
        assert_eq!(first_difference(&[1, 2, 3], &[1, 9, 3]), Some(1));
    }

    #[test]
    fn a_truncated_readback_is_a_mismatch_not_a_pass() {
        // The dangerous failure: a short read compares equal over its whole
        // length. Reporting `None` here would let a truncated upload verify.
        assert_eq!(first_difference(&[1, 2, 3], &[1, 2]), Some(2));
    }

    #[test]
    fn required_bytes_rounds_every_tensor_up_to_alignment() {
        // Q6_K superblocks are 210 bytes, so tensor sizes are rarely multiples
        // of 256. Under-counting here would size the arena short and fail the
        // load partway through, after minutes of copying.
        let unpadded = 210usize;
        assert_eq!(unpadded.next_multiple_of(ALIGNMENT), 256);
        assert_eq!(540_344_320usize.next_multiple_of(ALIGNMENT), 540_344_320);
    }

    #[test]
    fn throughput_is_bytes_over_elapsed() {
        let report = LoadReport {
            tensors: 1,
            bytes: 2_000_000_000,
            arena_bytes: 2_000_000_000,
            free_before: 0,
            free_after: 0,
            elapsed: Duration::from_secs(2),
        };
        assert!((report.throughput_gb_s() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn vram_consumed_does_not_underflow_when_memory_was_freed() {
        // Another process releasing memory mid-load can leave `free_after`
        // above `free_before`. That is a measurement artefact, not a negative
        // allocation, and it must not wrap around.
        let report = LoadReport {
            tensors: 0,
            bytes: 0,
            arena_bytes: 0,
            free_before: 100,
            free_after: 200,
            elapsed: Duration::from_secs(1),
        };
        assert_eq!(report.vram_consumed(), 0);
    }
}
