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

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use cudarc::nvrtc::{CompileOptions, Ptx};
use tracing::debug;

/// The only architecture this project targets.
///
/// Quadro RTX 8000 is Turing. Compiling for anything newer produces PTX the
/// driver will refuse, and compiling for something older silently gives up
/// the `m16n8k8` tensor-core path — see `docs/TOOLCHAIN.md`.
pub const TARGET_ARCH: &str = "compute_75";

/// Compiled PTX, keyed by the source that produced it.
///
/// # Why this exists
///
/// The same source is compiled several times per model build, because the
/// kernel sets are constructed independently by the code that needs them
/// rather than threaded down from one owner. `LayerOpsKernels::new` takes only
/// a context — it has no geometry to specialize on, so all four of its call
/// sites (`forward.rs`, and the GDN, attention and MoE blocks) produce
/// byte-identical PTX. That was ~147 ms each, paid four times, and
/// `Forward::reshape` doubled it by building a second shape over the same
/// weights.
///
/// Caching here rather than at each call site fixes every such case at once,
/// including ones nobody has looked for, and it needs no cache threaded
/// through four block constructors — which is what made this not worth fixing
/// before the source-level instrumentation showed how much it cost.
///
/// # Why the source is a sound key
///
/// PTX is a pure function of (source, compile options), and the only option
/// that varies is [`TARGET_ARCH`], which is a constant. `name` affects
/// diagnostics only, so two calls differing solely by name must produce the
/// same PTX and are correctly served from one entry.
///
/// This caches *PTX*, not modules. Loading PTX into a context is per-context
/// and still happens every time, which is what keeps this safe across the
/// three devices: a `CudaModule` belongs to the context that loaded it, and
/// nothing here holds one.
static PTX_CACHE: LazyLock<Mutex<HashMap<String, Ptx>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// How many times NVRTC has actually been invoked this process.
///
/// Distinct from how many times [`compile`] was called, which is the point:
/// the gap between the two is what the cache saved. Counted rather than
/// inferred from timings so a test can assert on it without being flaky.
static NVRTC_INVOCATIONS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Number of real NVRTC compilations so far, cache hits excluded.
pub fn nvrtc_invocations() -> usize {
    NVRTC_INVOCATIONS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether `src` already has compiled PTX cached.
///
/// Exists so a test can assert memoization directly. The obvious alternative —
/// a delta on [`nvrtc_invocations`] — is a global counter and races with any
/// other test compiling in the same binary, which is exactly how the first
/// version of that test failed.
#[cfg(test)]
fn is_cached(src: &str) -> bool {
    PTX_CACHE
        .lock()
        .expect("the PTX cache mutex is never held across a panic")
        .contains_key(src)
}

/// Compile a CUDA C++ source to PTX for [`TARGET_ARCH`], memoized on the
/// source.
///
/// `name` appears in NVRTC's diagnostics, so it should identify the kernel
/// rather than the caller.
///
/// Every kernel in the workspace compiles through here, so the `debug!` below
/// is the whole NVRTC bill for a run: which modules were built, how large
/// each source was, and what each cost. Cache hits are logged too, and as
/// hits rather than as suspiciously fast compiles — an instrumentation point
/// that lied about which work actually happened would be worse than none.
pub fn compile(src: &str, name: &str) -> Result<Ptx, String> {
    // The lock is released before compiling: holding it across NVRTC would
    // serialize compilation of *different* kernels behind one another. Two
    // threads racing on the same source both compile and the second overwrites
    // the first with an identical value, which wastes one compile and breaks
    // nothing.
    if let Some(hit) = PTX_CACHE
        .lock()
        .expect("the PTX cache mutex is never held across a panic")
        .get(src)
        .cloned()
    {
        debug!("nvrtc {name}: {} source bytes, cache hit", src.len());
        return Ok(hit);
    }

    NVRTC_INVOCATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let started = std::time::Instant::now();
    let result = cudarc::nvrtc::compile_ptx_with_opts(
        src,
        CompileOptions {
            arch: Some(TARGET_ARCH),
            name: Some(name.to_string()),
            ..Default::default()
        },
    )
    .map_err(|e| format!("{name}: {e:?}"));

    debug!(
        "nvrtc {name}: {} source bytes for {TARGET_ARCH} in {:.0} ms{}",
        src.len(),
        started.elapsed().as_secs_f64() * 1e3,
        if result.is_ok() { "" } else { " — FAILED" },
    );

    // Failures are not cached. A compile failure is deterministic for a given
    // source, so caching it would be sound — but it would also mean the error
    // is reported once with its diagnostics and thereafter from a cache, and
    // NVRTC's diagnostics are the whole value of the failure.
    if let Ok(ptx) = &result {
        PTX_CACHE
            .lock()
            .expect("the PTX cache mutex is never held across a panic")
            .insert(src.to_string(), ptx.clone());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A source unique to this test, so it cannot be served by an entry some
    /// other test in the same binary put there.
    const TRIVIAL: &str = r#"
extern "C" __global__ void ptx_cache_probe(float* out) { out[threadIdx.x] = 1.0f; }
"#;

    #[test]
    fn compiling_the_same_source_twice_invokes_nvrtc_once() {
        assert!(
            !is_cached(TRIVIAL),
            "this source is unique to this test and must start uncached",
        );
        let Ok(first) = compile(TRIVIAL, "ptx_cache_probe") else {
            // NVRTC is a host library, not a device, so this is unexpected —
            // but a skip is honest and a false pass is not.
            println!("SKIPPED: NVRTC unavailable in this environment");
            return;
        };
        assert!(
            is_cached(TRIVIAL),
            "a successful compile must populate the cache, or nothing is memoized",
        );

        // A different `name` for the same source: `name` is diagnostics only,
        // so this must still be a hit rather than a second compile.
        let second = compile(TRIVIAL, "a_different_name").expect("cached compile cannot fail");
        assert_eq!(
            first.to_src(),
            second.to_src(),
            "a cache hit must return what the compile returned",
        );

        // The counter is process-global and other tests compile concurrently,
        // so it can only be asserted on as a lower bound.
        assert!(nvrtc_invocations() >= 1);
    }

    #[test]
    fn a_different_source_is_not_served_from_another_sources_entry() {
        let a = "extern \"C\" __global__ void probe_a(float* o) { o[0] = 1.0f; }";
        let b = "extern \"C\" __global__ void probe_b(float* o) { o[0] = 2.0f; }";
        let (Ok(pa), Ok(pb)) = (compile(a, "probe_a"), compile(b, "probe_b")) else {
            println!("SKIPPED: NVRTC unavailable in this environment");
            return;
        };
        let (sa, sb) = (pa.to_src(), pb.to_src());
        assert_ne!(
            sa, sb,
            "distinct sources must not collide in the cache; if they do, the \
             key is wrong and every kernel is a coin flip",
        );
        assert!(sa.contains("probe_a") && sb.contains("probe_b"));
    }
}
