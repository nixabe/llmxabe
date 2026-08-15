//! Structural checks on how the workspace uses `tracing`.
//!
//! These are source scans, not behaviour tests, because the properties they
//! guard are about *where* code appears rather than what it computes. Both
//! failure modes they catch are silent: a library that installs a subscriber
//! steals the decision from its caller and only shows up as "my logging
//! configuration is ignored" in some downstream program, and a `println!`
//! that creeps back into a binary ignores `--log-level` while looking
//! completely normal at the default level.

use std::fs;
use std::path::{Path, PathBuf};

/// The workspace root, from this crate's manifest directory.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/xabe-log is two levels below the workspace root")
        .to_path_buf()
}

/// Every `.rs` file under `crates/`, with its path relative to the root.
fn workspace_sources() -> Vec<(PathBuf, String)> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }

    let root = workspace_root();
    let mut paths = Vec::new();
    walk(&root.join("crates"), &mut paths);
    assert!(
        paths.len() > 20,
        "found only {} source files; the walk is broken, not the workspace",
        paths.len()
    );

    paths
        .into_iter()
        .map(|p| {
            let text = fs::read_to_string(&p).expect("workspace sources are UTF-8");
            let rel = p.strip_prefix(&root).unwrap_or(&p).to_path_buf();
            (rel, text)
        })
        .collect()
}

/// True for the files that are allowed to own process-global logging setup:
/// binary entry points and examples.
fn is_entry_point(path: &Path) -> bool {
    let s = path.to_string_lossy().replace('\\', "/");
    s.contains("/src/bin/") || s.contains("/examples/") || s.ends_with("/src/main.rs")
}

/// True for this crate's own sources, which of course mention `xabe_log`.
fn is_xabe_log_itself(path: &Path) -> bool {
    path.to_string_lossy()
        .replace('\\', "/")
        .contains("crates/xabe-log/")
}

#[test]
fn crates_do_not_install_a_subscriber() {
    // A library that calls `init` decides logging for every program that links
    // it. That decision belongs to the program.
    let offenders: Vec<String> = workspace_sources()
        .into_iter()
        .filter(|(p, _)| !is_entry_point(p) && !is_xabe_log_itself(p))
        .filter(|(_, src)| {
            src.contains("xabe_log::init")
                || src.contains("tracing_subscriber")
                || src.contains("set_global_default")
        })
        .map(|(p, _)| p.display().to_string())
        .collect();

    assert!(
        offenders.is_empty(),
        "library code must emit `tracing` events and leave subscriber \
         installation to the binary; these files install one: {offenders:?}"
    );
}

#[test]
fn entry_points_do_not_print_directly() {
    // `println!` in a binary bypasses `--log-level` entirely, and at the
    // default level it renders identically to `info!` — so this is invisible
    // in review and invisible in the output. Only a scan catches it.
    let offenders: Vec<String> = workspace_sources()
        .into_iter()
        .filter(|(p, _)| is_entry_point(p))
        .filter_map(|(p, src)| {
            let hits = src
                .lines()
                .enumerate()
                .filter(|(_, l)| {
                    let l = l.trim_start();
                    !l.starts_with("//")
                        && (l.contains("println!")
                            || l.contains("eprintln!")
                            || l.contains("eprint!("))
                })
                .map(|(i, _)| i + 1)
                .collect::<Vec<_>>();
            (!hits.is_empty()).then(|| format!("{} at lines {hits:?}", p.display()))
        })
        .collect();

    assert!(
        offenders.is_empty(),
        "binaries and examples must log through `tracing`, not print directly, \
         or their output ignores --log-level: {offenders:?}"
    );
}

/// Test code is exempt, and that exemption is deliberate rather than an
/// oversight — so assert that it is still being exercised. If every `println!`
/// in the workspace's tests were ever migrated, this test should be deleted
/// along with the exemption, not left asserting nothing.
#[test]
fn test_code_keeps_using_libtests_own_capture() {
    let test_prints = workspace_sources()
        .into_iter()
        .filter(|(p, _)| !is_entry_point(p))
        .filter(|(_, src)| src.contains("#[cfg(test)]") || src.contains("#[test]"))
        .filter(|(_, src)| src.contains("println!") || src.contains("eprintln!"))
        .count();

    assert!(
        test_prints > 0,
        "no test still prints directly; if that is intentional, delete this \
         test and the exemption in `is_entry_point`'s callers rather than \
         leaving a vacuous assertion behind"
    );
}
