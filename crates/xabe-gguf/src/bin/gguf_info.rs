//! `gguf-info <path> [--tensors [substring]] [--meta [substring]]` — dump
//! header, selected metadata, and a tensor-type histogram for a GGUF file.
//! Uses only the public `xabe_gguf` API, so it doubles as a smoke test that
//! the API surface is actually usable.

use std::collections::BTreeMap;
use std::process::ExitCode;

use tracing::{error, info};
use xabe_gguf::{GgufArray, GgufFile, GgufValue};

/// One-line rendering of a metadata value.
///
/// Arrays are summarised rather than printed: `tokenizer.ggml.tokens` alone
/// is 248,320 strings, and dumping it would bury every other key.
fn render(value: &GgufValue) -> String {
    fn array(kind: &str, len: usize, head: String) -> String {
        format!("{kind}[{len}] {head}")
    }
    match value {
        GgufValue::U8(v) => v.to_string(),
        GgufValue::I8(v) => v.to_string(),
        GgufValue::U16(v) => v.to_string(),
        GgufValue::I16(v) => v.to_string(),
        GgufValue::U32(v) => v.to_string(),
        GgufValue::I32(v) => v.to_string(),
        GgufValue::F32(v) => v.to_string(),
        GgufValue::Bool(v) => v.to_string(),
        GgufValue::U64(v) => v.to_string(),
        GgufValue::I64(v) => v.to_string(),
        GgufValue::F64(v) => v.to_string(),
        GgufValue::String(v) if v.len() > 72 => format!("{:?}…", &v[..72]),
        GgufValue::String(v) => format!("{v:?}"),
        GgufValue::Array(a) => match a {
            GgufArray::U8(v) => array("u8", v.len(), format!("{:?}", head(v))),
            GgufArray::I8(v) => array("i8", v.len(), format!("{:?}", head(v))),
            GgufArray::U16(v) => array("u16", v.len(), format!("{:?}", head(v))),
            GgufArray::I16(v) => array("i16", v.len(), format!("{:?}", head(v))),
            GgufArray::U32(v) => array("u32", v.len(), format!("{:?}", head(v))),
            GgufArray::I32(v) => array("i32", v.len(), format!("{:?}", head(v))),
            GgufArray::F32(v) => array("f32", v.len(), format!("{:?}", head(v))),
            GgufArray::Bool(v) => array("bool", v.len(), format!("{:?}", head(v))),
            GgufArray::String(v) => array("str", v.len(), format!("{:?}", head(v))),
            GgufArray::U64(v) => array("u64", v.len(), format!("{:?}", head(v))),
            GgufArray::I64(v) => array("i64", v.len(), format!("{:?}", head(v))),
            GgufArray::F64(v) => array("f64", v.len(), format!("{:?}", head(v))),
        },
    }
}

fn head<T: Clone>(v: &[T]) -> Vec<T> {
    v.iter().take(8).cloned().collect()
}

fn main() -> ExitCode {
    let mut args = xabe_log::init_from_args().into_iter();
    let Some(path) = args.next() else {
        error!("usage: gguf-info <path-to-gguf> [--tensors [substring]] [--meta [substring]]");
        error!("{}", xabe_log::FLAG_HELP);
        return ExitCode::from(2);
    };
    let rest: Vec<String> = args.collect();
    let filter_after = |flag: &str| -> Option<Option<String>> {
        let i = rest.iter().position(|a| a == flag)?;
        Some(rest.get(i + 1).filter(|a| !a.starts_with("--")).cloned())
    };
    let tensor_filter = filter_after("--tensors");
    let meta_filter = filter_after("--meta");

    let file = match GgufFile::open(&path) {
        Ok(f) => f,
        Err(e) => {
            error!("failed to load `{path}`: {e}");
            return ExitCode::FAILURE;
        }
    };

    info!("file: {path}");
    info!("version: {}", file.version());
    info!("alignment: {} bytes", file.alignment());
    info!("tensors: {}", file.n_tensors());
    info!("metadata keys: {}", file.n_kv());

    if let Some(arch) = file.get_str("general.architecture") {
        info!("architecture: {arch}");
    }
    if let Some(name) = file.get_str("general.name") {
        info!("name: {name}");
    }

    info!("\ntensor type histogram:");
    let mut hist: BTreeMap<&'static str, (u64, u64)> = BTreeMap::new();
    for t in file.tensors() {
        let entry = hist.entry(t.ggml_type.name()).or_default();
        entry.0 += 1;
        entry.1 += t.n_bytes;
    }
    let total_bytes: u64 = hist.values().map(|(_, bytes)| bytes).sum();
    for (ty, (count, bytes)) in &hist {
        info!(
            "  {ty:<6} count={count:<6} bytes={bytes:<14} ({:.2} GiB)",
            *bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }
    info!(
        "  total tensor bytes: {total_bytes} ({:.2} GiB)",
        total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    if let Some(filter) = meta_filter {
        info!("\nmetadata:");
        let mut keys: Vec<&str> = file.metadata_keys().collect();
        keys.sort_unstable();
        for key in keys {
            if filter.as_ref().is_some_and(|f| !key.contains(f.as_str())) {
                continue;
            }
            let Some(value) = file.get(key) else { continue };
            info!("  {key:<44} {}", render(value));
        }
    }

    if let Some(filter) = tensor_filter {
        info!("\ntensors:");
        for t in file.tensors() {
            if filter
                .as_ref()
                .is_some_and(|f| !t.name.contains(f.as_str()))
            {
                continue;
            }
            info!(
                "  {:<40} {:<6} {:?} offset={} bytes={}",
                t.name,
                t.ggml_type.name(),
                t.dims,
                t.offset,
                t.n_bytes
            );
        }
    }

    ExitCode::SUCCESS
}
