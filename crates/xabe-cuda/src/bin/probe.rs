//! Device inventory and the milestone-00 toolchain gate.
//!
//! ```sh
//! cargo run -p xabe-cuda --bin probe
//! ```
//!
//! Prints what the host offers and then runs the four gating checks from the
//! design plan against device 0. Exits non-zero if the fleet fails the gate or
//! a check that ran did not pass.

use tracing::{error, info};
use xabe_cuda::{SpikeReport, check_gate, device, spike};

fn main() -> std::process::ExitCode {
    xabe_log::init_from_args();

    info!("llmxabe device probe\n");

    if !device::driver_available() {
        error!("No CUDA driver reachable on this host.");
        error!("Host-side crates still build and test; device work cannot be verified here.");
        return std::process::ExitCode::FAILURE;
    }

    let devices = match device::probe_all() {
        Ok(d) => d,
        Err(e) => {
            error!("Failed to probe devices: {e:?}");
            return std::process::ExitCode::FAILURE;
        }
    };

    if devices.is_empty() {
        error!("Driver present, but no CUDA devices are visible.");
        return std::process::ExitCode::FAILURE;
    }

    for d in &devices {
        info!("Device {} — {}", d.ordinal, d.name);
        info!(
            "  compute capability   {} ({}){}",
            d.compute_capability,
            d.compute_capability.sm_arch(),
            if d.compute_capability.is_turing() {
                "  [Turing — the target architecture]"
            } else {
                ""
            }
        );
        info!(
            "  total memory         {:.1} GiB",
            d.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        info!("  SMs                  {}", d.sm_count);
        info!(
            "  L2 cache             {:.1} MiB",
            f64::from(d.l2_cache_bytes) / (1024.0 * 1024.0)
        );
        info!(
            "  memory bus           {}-bit @ {:.0} MHz effective",
            d.memory_bus_width_bits,
            f64::from(d.memory_clock_khz) / 1000.0
        );
        info!(
            "  peak bandwidth       {:.0} GB/s   (every roofline is stated against this)",
            d.peak_bandwidth_gb_s()
        );
        info!(
            "  shared mem / block   {} KiB",
            d.max_shared_memory_per_block / 1024
        );
        info!("  warp size            {}", d.warp_size);
        info!("");
    }

    match check_gate(&devices) {
        Ok(()) => info!(
            "Capability gate: PASS — {} device(s), all compute {}\n",
            devices.len(),
            devices[0].compute_capability
        ),
        Err(e) => {
            error!("Capability gate: FAIL — {e}\n");
            return std::process::ExitCode::FAILURE;
        }
    }

    info!("Milestone 00 — toolchain gating spike (device 0)\n");
    let report = spike::run(0);
    for (label, outcome) in SpikeReport::LABELS.iter().zip(report.checks()) {
        info!("  {label:<34} {outcome}");
    }
    info!("");

    if !report.all_ran() {
        error!("Spike incomplete: at least one check did not run.");
        return std::process::ExitCode::FAILURE;
    }

    if report.all_ran_checks_passed() {
        info!("Toolchain decision: {}", xabe_cuda::TOOLCHAIN_DECISION);
        info!("See docs/TOOLCHAIN.md for the reasoning.");
        std::process::ExitCode::SUCCESS
    } else {
        error!("A gating check failed. This changes the toolchain decision — see");
        info!("docs/TOOLCHAIN.md before proceeding with kernel work.");
        std::process::ExitCode::FAILURE
    }
}
