//! `gguf-info <path>` — dump header, selected metadata, and a tensor-type
//! histogram for a GGUF file. Uses only the public `xabe_gguf` API, so it
//! doubles as a smoke test that the API surface is actually usable.

use std::collections::BTreeMap;
use std::env;
use std::process::ExitCode;

use xabe_gguf::GgufFile;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: gguf-info <path-to-gguf>");
        return ExitCode::from(2);
    };

    let file = match GgufFile::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("failed to load `{path}`: {e}");
            return ExitCode::FAILURE;
        }
    };

    println!("file: {path}");
    println!("version: {}", file.version());
    println!("alignment: {} bytes", file.alignment());
    println!("tensors: {}", file.n_tensors());
    println!("metadata keys: {}", file.n_kv());

    if let Some(arch) = file.get_str("general.architecture") {
        println!("architecture: {arch}");
    }
    if let Some(name) = file.get_str("general.name") {
        println!("name: {name}");
    }

    println!("\ntensor type histogram:");
    let mut hist: BTreeMap<&'static str, (u64, u64)> = BTreeMap::new();
    for t in file.tensors() {
        let entry = hist.entry(t.ggml_type.name()).or_default();
        entry.0 += 1;
        entry.1 += t.n_bytes;
    }
    let total_bytes: u64 = hist.values().map(|(_, bytes)| bytes).sum();
    for (ty, (count, bytes)) in &hist {
        println!(
            "  {ty:<6} count={count:<6} bytes={bytes:<14} ({:.2} GiB)",
            *bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }
    println!(
        "  total tensor bytes: {total_bytes} ({:.2} GiB)",
        total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    ExitCode::SUCCESS
}
