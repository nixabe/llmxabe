//! Integration test against the real target model file.
//!
//! Honors `LLMXABE_MODEL` as an override; otherwise falls back to the path
//! this project's plan documents. The file is 32 GB — [`GgufFile::open`]
//! memory-maps it, so this test does not read it into process memory. If the
//! file is absent (e.g. CI, or a machine without the model downloaded), the
//! test SKIPS with a printed message rather than failing: per `AGENTS.md`,
//! "a skipped test is not a passing test," but it is also not evidence of a
//! bug in this crate.

use std::path::PathBuf;

use xabe_gguf::GgufFile;

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llama.cpp/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

fn model_path() -> PathBuf {
    std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
}

#[test]
fn loads_real_model_and_reports_plausible_structure() {
    let path = model_path();
    if !path.exists() {
        println!(
            "SKIPPED: real model file not found at {}; set LLMXABE_MODEL to override",
            path.display()
        );
        return;
    }

    let file = GgufFile::open(&path).expect("real model file must parse as valid GGUF v3");

    println!("loaded {}", path.display());
    println!("version={} alignment={}", file.version(), file.alignment());
    println!("n_tensors={} n_kv={}", file.n_tensors(), file.n_kv());

    // Plausible tensor count: this model has ~750 tensors in the UD-Q6_K_XL
    // quant (41 transformer blocks incl. the MTP block, each with ~18
    // weights, plus a handful of top-level tensors).
    assert!(
        (500..2000).contains(&file.n_tensors()),
        "tensor count {} outside plausible range",
        file.n_tensors()
    );

    let arch = file
        .get_str("general.architecture")
        .expect("general.architecture metadata key must be present");
    println!("architecture={arch}");
    assert!(!arch.is_empty());

    // Expert tensors: MoE weight matrices carry "_exps" in their name in
    // llama.cpp's GGUF tensor naming convention (ffn_gate_exps,
    // ffn_up_exps, ffn_down_exps).
    let expert_tensor_count = file
        .tensors()
        .iter()
        .filter(|t| t.name.contains("_exps"))
        .count();
    println!("expert tensor count={expert_tensor_count}");
    assert!(
        expert_tensor_count > 0,
        "expected at least one MoE expert tensor (name containing `_exps`)"
    );

    // Type histogram: count and total bytes per ggml type actually present.
    use std::collections::BTreeMap;
    let mut hist: BTreeMap<&'static str, (u64, u64)> = BTreeMap::new();
    for t in file.tensors() {
        let entry = hist.entry(t.ggml_type.name()).or_default();
        entry.0 += 1;
        entry.1 += t.n_bytes;
    }
    println!("type histogram:");
    for (ty, (count, bytes)) in &hist {
        println!(
            "  {ty:<6} count={count:<6} bytes={bytes:<14} ({:.2} GiB)",
            *bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }
    assert!(!hist.is_empty());

    // Spot-check zero-copy tensor byte access on whatever the first tensor
    // in the directory is, without assuming a specific name.
    let first = &file.tensors()[0];
    let bytes = file
        .tensor_bytes(&first.name)
        .expect("directory-listed tensor must resolve to a byte slice");
    assert_eq!(bytes.len(), first.n_bytes as usize);
}
