//! Device discovery and the sm_75 capability gate.
//!
//! Everything this project does is shaped by running on Turing. Turing has no
//! `cp.async`, so double-buffering is by hand; its tensor cores are the
//! `m16n8k8` fp16 MMA family; and its L2 is small enough that MoE block launch
//! ordering measurably changes hit rate.
//!
//! Rather than discover that a machine is not Turing halfway through a kernel
//! launch, the engine probes once at startup and refuses to run on anything it
//! was not designed for.

use cudarc::driver::{CudaContext, DriverError, sys};
use std::sync::Arc;

/// Minimum compute capability this engine targets.
///
/// Turing. The kernels assume its instruction set and its absence of
/// `cp.async`; running them on something older is not a performance question
/// but a correctness one.
pub const MIN_COMPUTE_CAPABILITY: ComputeCapability = ComputeCapability { major: 7, minor: 5 };

/// A CUDA compute capability, ordered so that comparisons mean what they look
/// like they mean.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ComputeCapability {
    /// Major version.
    pub major: i32,
    /// Minor version.
    pub minor: i32,
}

impl ComputeCapability {
    /// The `sm_XX` string NVRTC and PTX expect.
    pub fn sm_arch(&self) -> String {
        format!("sm_{}{}", self.major, self.minor)
    }

    /// Whether this device meets the engine's minimum.
    pub fn is_supported(&self) -> bool {
        *self >= MIN_COMPUTE_CAPABILITY
    }

    /// Whether this is Turing specifically, as opposed to merely new enough.
    ///
    /// Some tuning decisions — hand double-buffering, MoE launch ordering for
    /// a small L2 — are pessimizations on later architectures rather than
    /// merely unnecessary.
    pub fn is_turing(&self) -> bool {
        self.major == 7 && self.minor == 5
    }
}

impl core::fmt::Display for ComputeCapability {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// Everything the engine wants to know about one GPU before it commits work to
/// it.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// CUDA device ordinal.
    pub ordinal: usize,
    /// Marketing name, e.g. "Quadro RTX 8000".
    pub name: String,
    /// Compute capability.
    pub compute_capability: ComputeCapability,
    /// Total device memory in bytes.
    pub total_memory: u64,
    /// Streaming multiprocessor count. Sets the natural grid size for kernels
    /// that stride over tiles rather than sizing the grid to the worst case.
    pub sm_count: i32,
    /// L2 cache size in bytes. Small on Turing, which is why MoE block launch
    /// ordering matters here more than on later parts.
    pub l2_cache_bytes: i32,
    /// Memory bus width in bits.
    pub memory_bus_width_bits: i32,
    /// Peak memory clock in kHz.
    pub memory_clock_khz: i32,
    /// Maximum shared memory per block, in bytes.
    pub max_shared_memory_per_block: i32,
    /// Maximum registers available to a single block.
    pub max_registers_per_block: i32,
    /// Warp size. Assumed to be 32 throughout; checked rather than trusted.
    pub warp_size: i32,
}

impl DeviceInfo {
    /// Theoretical peak memory bandwidth in GB/s.
    ///
    /// Every roofline in `docs/MODEL.md` is stated against this number, and
    /// this workload is bandwidth-bound, so it is the single most predictive
    /// figure the device reports. For the RTX 8000 it should come out near
    /// 672 GB/s.
    pub fn peak_bandwidth_gb_s(&self) -> f64 {
        // Clock is in kHz; GDDR6 transfers on both edges, hence the factor 2.
        let bits_per_second =
            f64::from(self.memory_clock_khz) * 1e3 * 2.0 * f64::from(self.memory_bus_width_bits);
        bits_per_second / 8.0 / 1e9
    }

    /// Whether this device meets the engine's minimum capability.
    pub fn is_supported(&self) -> bool {
        self.compute_capability.is_supported()
    }

    /// Probe a single device by ordinal.
    pub fn probe(ordinal: usize) -> Result<Self, DriverError> {
        let ctx = CudaContext::new(ordinal)?;
        Self::from_context(ordinal, &ctx)
    }

    /// Read device properties from an already-created context.
    pub fn from_context(ordinal: usize, ctx: &Arc<CudaContext>) -> Result<Self, DriverError> {
        use sys::CUdevice_attribute as A;

        Ok(Self {
            ordinal,
            name: ctx.name()?,
            compute_capability: ComputeCapability {
                major: ctx.attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
                minor: ctx.attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
            },
            total_memory: ctx.total_mem()? as u64,
            sm_count: ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
            l2_cache_bytes: ctx.attribute(A::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)?,
            memory_bus_width_bits: ctx.attribute(A::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)?,
            memory_clock_khz: ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)?,
            max_shared_memory_per_block: ctx
                .attribute(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?,
            max_registers_per_block: ctx
                .attribute(A::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK)?,
            warp_size: ctx.attribute(A::CU_DEVICE_ATTRIBUTE_WARP_SIZE)?,
        })
    }
}

/// Probe every visible CUDA device.
///
/// Returns an empty vector when the driver is present but no devices are
/// visible. An `Err` means the driver itself could not be reached, which is a
/// different situation and should be reported differently.
pub fn probe_all() -> Result<Vec<DeviceInfo>, DriverError> {
    let count = CudaContext::device_count()?;
    (0..count as usize).map(DeviceInfo::probe).collect()
}

/// Whether a CUDA driver is reachable at all.
///
/// Used by tests to decide between skipping and failing. A test that cannot
/// find a GPU has not passed — it has not run, and it must say so.
///
/// The `catch_unwind` is load-bearing, not defensive. `cudarc`'s
/// `fallback-dynamic-loading` resolves `libcuda` on first use and *panics*
/// from `panic_no_lib_found` when it finds none, so `device_count().is_ok()`
/// never gets to answer `false` on a machine with no driver installed — the
/// one machine this function exists to recognize. Every GPU-gated test in the
/// workspace calls through here, so without this they all abort instead of
/// skipping, which is how CI on a GPU-less runner failed.
///
/// A driver that is present but unhappy still returns `Err` normally; only
/// the library-not-found case unwinds. The panic message is written to
/// stderr, which the test harness captures and discards for a test that goes
/// on to pass.
pub fn driver_available() -> bool {
    std::panic::catch_unwind(|| CudaContext::device_count().is_ok()).unwrap_or(false)
}

/// Reasons a set of devices is unsuitable for this engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateFailure {
    /// No CUDA devices were visible.
    NoDevices,
    /// At least one device is below the minimum compute capability.
    BelowMinimum {
        /// Ordinal of the offending device.
        ordinal: usize,
        /// What it reported.
        found: ComputeCapability,
    },
    /// Devices differ in compute capability.
    ///
    /// The engine compiles one set of kernels and captures one graph shape per
    /// worker. A mixed fleet would silently get the lowest common denominator,
    /// so it is rejected rather than accommodated.
    Heterogeneous {
        /// Distinct capabilities observed.
        found: Vec<ComputeCapability>,
    },
}

impl core::fmt::Display for GateFailure {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoDevices => write!(f, "no CUDA devices visible"),
            Self::BelowMinimum { ordinal, found } => write!(
                f,
                "device {ordinal} has compute capability {found}, below the required \
                 {MIN_COMPUTE_CAPABILITY}"
            ),
            Self::Heterogeneous { found } => {
                write!(f, "mixed compute capabilities across devices: ")?;
                for (i, cc) in found.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{cc}")?;
                }
                Ok(())
            }
        }
    }
}

impl core::error::Error for GateFailure {}

/// Check that a device set is one this engine can run on.
pub fn check_gate(devices: &[DeviceInfo]) -> Result<(), GateFailure> {
    if devices.is_empty() {
        return Err(GateFailure::NoDevices);
    }

    for d in devices {
        if !d.is_supported() {
            return Err(GateFailure::BelowMinimum {
                ordinal: d.ordinal,
                found: d.compute_capability,
            });
        }
    }

    let mut caps: Vec<_> = devices.iter().map(|d| d.compute_capability).collect();
    caps.sort();
    caps.dedup();
    if caps.len() > 1 {
        return Err(GateFailure::Heterogeneous { found: caps });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(ordinal: usize, major: i32, minor: i32) -> DeviceInfo {
        DeviceInfo {
            ordinal,
            name: "test".into(),
            compute_capability: ComputeCapability { major, minor },
            total_memory: 48 * 1024 * 1024 * 1024,
            sm_count: 72,
            l2_cache_bytes: 6 * 1024 * 1024,
            memory_bus_width_bits: 384,
            memory_clock_khz: 7_000_000,
            max_shared_memory_per_block: 49_152,
            max_registers_per_block: 65_536,
            warp_size: 32,
        }
    }

    #[test]
    fn compute_capability_orders_by_major_then_minor() {
        let sm70 = ComputeCapability { major: 7, minor: 0 };
        let sm75 = ComputeCapability { major: 7, minor: 5 };
        let sm80 = ComputeCapability { major: 8, minor: 0 };
        assert!(sm70 < sm75);
        assert!(sm75 < sm80);
        assert!(!sm70.is_supported(), "Volta is below the Turing minimum");
        assert!(sm75.is_supported());
        assert!(sm80.is_supported());
    }

    #[test]
    fn only_seventy_five_counts_as_turing() {
        assert!(ComputeCapability { major: 7, minor: 5 }.is_turing());
        assert!(!ComputeCapability { major: 8, minor: 0 }.is_turing());
        assert!(!ComputeCapability { major: 7, minor: 0 }.is_turing());
    }

    #[test]
    fn sm_arch_string_is_what_nvrtc_expects() {
        assert_eq!(
            ComputeCapability { major: 7, minor: 5 }.sm_arch(),
            "sm_75",
            "NVRTC rejects any other spelling"
        );
    }

    #[test]
    fn rtx_8000_bandwidth_derives_to_about_672_gb_s() {
        // 384-bit bus at 7000 MHz effective GDDR6.
        let d = dev(0, 7, 5);
        let bw = d.peak_bandwidth_gb_s();
        assert!(
            (600.0..=700.0).contains(&bw),
            "expected ~672 GB/s for a 384-bit GDDR6 bus, derived {bw}"
        );
    }

    #[test]
    fn an_empty_fleet_is_rejected() {
        assert_eq!(check_gate(&[]), Err(GateFailure::NoDevices));
    }

    #[test]
    fn a_homogeneous_turing_fleet_passes() {
        let fleet = [dev(0, 7, 5), dev(1, 7, 5), dev(2, 7, 5)];
        assert_eq!(check_gate(&fleet), Ok(()));
    }

    #[test]
    fn a_pre_turing_device_is_rejected() {
        let fleet = [dev(0, 7, 5), dev(1, 6, 1)];
        assert!(matches!(
            check_gate(&fleet),
            Err(GateFailure::BelowMinimum { ordinal: 1, .. })
        ));
    }

    #[test]
    fn a_mixed_fleet_is_rejected_rather_than_accommodated() {
        // One graph shape and one kernel set are compiled per worker; a mixed
        // fleet would silently run at the lowest common denominator.
        let fleet = [dev(0, 7, 5), dev(1, 8, 6)];
        assert!(matches!(
            check_gate(&fleet),
            Err(GateFailure::Heterogeneous { .. })
        ));
    }
}
