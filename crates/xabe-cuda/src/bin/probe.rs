//! Device inventory and the milestone-00 toolchain gate.
//!
//! ```sh
//! cargo run -p xabe-cuda --bin probe
//! ```
//!
//! Prints what the host offers and then runs the four gating checks from the
//! design plan against device 0. Exits non-zero if the fleet fails the gate or
//! a check that ran did not pass.

use xabe_cuda::{SpikeReport, check_gate, device, spike};

fn main() -> std::process::ExitCode {
    println!("llmxabe device probe\n");

    if !device::driver_available() {
        println!("No CUDA driver reachable on this host.");
        println!("Host-side crates still build and test; device work cannot be verified here.");
        return std::process::ExitCode::FAILURE;
    }

    let devices = match device::probe_all() {
        Ok(d) => d,
        Err(e) => {
            println!("Failed to probe devices: {e:?}");
            return std::process::ExitCode::FAILURE;
        }
    };

    if devices.is_empty() {
        println!("Driver present, but no CUDA devices are visible.");
        return std::process::ExitCode::FAILURE;
    }

    for d in &devices {
        println!("Device {} — {}", d.ordinal, d.name);
        println!(
            "  compute capability   {} ({}){}",
            d.compute_capability,
            d.compute_capability.sm_arch(),
            if d.compute_capability.is_turing() {
                "  [Turing — the target architecture]"
            } else {
                ""
            }
        );
        println!(
            "  total memory         {:.1} GiB",
            d.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        println!("  SMs                  {}", d.sm_count);
        println!(
            "  L2 cache             {:.1} MiB",
            f64::from(d.l2_cache_bytes) / (1024.0 * 1024.0)
        );
        println!(
            "  memory bus           {}-bit @ {:.0} MHz effective",
            d.memory_bus_width_bits,
            f64::from(d.memory_clock_khz) / 1000.0
        );
        println!(
            "  peak bandwidth       {:.0} GB/s   (every roofline is stated against this)",
            d.peak_bandwidth_gb_s()
        );
        println!(
            "  shared mem / block   {} KiB",
            d.max_shared_memory_per_block / 1024
        );
        println!("  warp size            {}", d.warp_size);
        println!();
    }

    match check_gate(&devices) {
        Ok(()) => println!(
            "Capability gate: PASS — {} device(s), all compute {}\n",
            devices.len(),
            devices[0].compute_capability
        ),
        Err(e) => {
            println!("Capability gate: FAIL — {e}\n");
            return std::process::ExitCode::FAILURE;
        }
    }

    println!("Milestone 00 — toolchain gating spike (device 0)\n");
    let report = spike::run(0);
    for (label, outcome) in SpikeReport::LABELS.iter().zip(report.checks()) {
        println!("  {label:<34} {outcome}");
    }
    println!();

    if !report.all_ran() {
        println!("Spike incomplete: at least one check did not run.");
        return std::process::ExitCode::FAILURE;
    }

    if report.all_ran_checks_passed() {
        println!("Toolchain decision: {}", xabe_cuda::TOOLCHAIN_DECISION);
        println!("See docs/TOOLCHAIN.md for the reasoning.");
        std::process::ExitCode::SUCCESS
    } else {
        println!("A gating check failed. This changes the toolchain decision — see");
        println!("docs/TOOLCHAIN.md before proceeding with kernel work.");
        std::process::ExitCode::FAILURE
    }
}
