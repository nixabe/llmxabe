//! Resolve the weight schema against the real target model file.
//!
//! [`WeightSchema`] derives every expected tensor name and dimension from
//! [`ModelConfig`] alone. This test is what closes the loop: it asserts that
//! the derivation matches a 32 GB file byte for byte in structure — every
//! tensor present, every shape as predicted, nothing left over.
//!
//! If the file is absent the test SKIPS with a printed message. Per
//! `AGENTS.md`, a skipped test is reported as skipped and is not evidence the
//! schema is right.

use std::path::PathBuf;

use xabe_gguf::GgufFile;
use xabe_model::config::ModelConfig;
use xabe_model::weights::{Role, Section, WeightSchema};

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llama.cpp/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

fn model_path() -> PathBuf {
    std::env::var_os("LLMXABE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
}

fn open() -> Option<GgufFile> {
    let path = model_path();
    if !path.exists() {
        println!(
            "SKIPPED: real model file not found at {}; set LLMXABE_MODEL to override",
            path.display()
        );
        return None;
    }
    Some(GgufFile::open(&path).expect("real model file must parse as valid GGUF v3"))
}

#[test]
fn schema_resolves_against_the_real_file_with_no_mismatches() {
    let Some(file) = open() else { return };
    let config = ModelConfig::qwen3_6_35b_a3b();
    let schema = WeightSchema::with_mtp(&config);

    let directory = match schema.resolve(&file) {
        Ok(d) => d,
        Err(errors) => {
            for e in &errors {
                eprintln!("  {e}");
            }
            panic!("{} tensor(s) did not match the schema", errors.len());
        }
    };

    // The schema is exhaustive: it accounts for every tensor in the file, with
    // none left unclaimed. This is the assertion that would catch a tensor the
    // engine silently never loads.
    assert_eq!(
        directory.len(),
        file.n_tensors(),
        "schema resolved {} tensors but the file holds {}",
        directory.len(),
        file.n_tensors(),
    );

    println!("resolved {} tensors", directory.len());
    println!(
        "total {:.2} GiB across sections:",
        directory.total_bytes() as f64 / (1 << 30) as f64
    );
    for section in [
        Section::Embedding,
        Section::LmHead,
        Section::Experts,
        Section::Projections,
        Section::Norms,
        Section::Mtp,
    ] {
        println!(
            "  {section:<12?} {:>10.3} GiB  {:>12} params",
            directory.section_bytes(section) as f64 / (1 << 30) as f64,
            directory.section_elements(section),
        );
    }
    println!("types:");
    for (ty, count, bytes) in directory.type_histogram() {
        println!(
            "  {:<6} count={count:<4} {:>8.3} GiB",
            ty.name(),
            bytes as f64 / (1 << 30) as f64
        );
    }
}

#[test]
fn text_path_excludes_the_mtp_block_and_is_smaller_for_it() {
    let Some(file) = open() else { return };
    let config = ModelConfig::qwen3_6_35b_a3b();

    let text_schema = WeightSchema::new(&config);
    let full_schema = WeightSchema::with_mtp(&config);
    let text = text_schema
        .resolve(&file)
        .expect("text-only schema must resolve");
    let full = full_schema
        .resolve(&file)
        .expect("full schema must resolve");

    assert_eq!(text.section_bytes(Section::Mtp), 0);
    assert!(full.section_bytes(Section::Mtp) > 0);

    let saved = full.total_bytes() - text.total_bytes();
    println!(
        "text path {:.2} GiB, full {:.2} GiB — skipping MTP saves {:.2} GiB",
        text.total_bytes() as f64 / (1 << 30) as f64,
        full.total_bytes() as f64 / (1 << 30) as f64,
        saved as f64 / (1 << 30) as f64,
    );
    assert_eq!(full.len() - text.len(), 20, "MTP block is 20 tensors");
}

#[test]
fn resolved_parameter_count_lands_where_the_model_name_claims() {
    let Some(file) = open() else { return };
    let config = ModelConfig::qwen3_6_35b_a3b();
    let schema = WeightSchema::new(&config);
    let text = schema
        .resolve(&file)
        .expect("text-only schema must resolve");

    // "35B" in the model name, counted over the text path only. This is a
    // ground-truth sum over the file's own tensor directory, not a derivation
    // — it is the number `verify::check_config` is checked against.
    let billions = text.total_elements() as f64 / 1e9;
    println!("text-path parameters: {billions:.4} B");
    assert!(
        (34.0..36.0).contains(&billions),
        "expected ~35B parameters on the text path, summed {billions:.4} B",
    );
}

#[test]
fn derived_parameter_count_agrees_with_the_file_section_by_section() {
    // `ModelConfig::total_params()` derives a parameter count from head
    // counts and expert geometry. This test compares that derivation against a
    // sum over the file's own tensor directory, per section, so a config error
    // reports *which* part of the architecture is wrong rather than only that
    // the total drifted.
    //
    // Both sides cover the text path only. The MTP block is a full extra
    // attention layer plus its own 256 experts — 0.84 B parameters — and
    // including it on one side but not the other accounts for exactly the
    // 2.4% gap an earlier hand-rolled comparison reported.
    let Some(file) = open() else { return };
    let config = ModelConfig::qwen3_6_35b_a3b();
    let schema = WeightSchema::new(&config);
    let d = schema.resolve(&file).expect("schema must resolve");

    let cases: [(&str, u64, u64); 4] = [
        (
            "embedding",
            u64::from(config.hidden_size) * u64::from(config.vocab_size),
            d.section_elements(Section::Embedding),
        ),
        (
            "lm_head",
            u64::from(config.hidden_size) * u64::from(config.vocab_size),
            d.section_elements(Section::LmHead),
        ),
        (
            "experts",
            config.total_ffn_params(),
            d.section_elements(Section::Experts),
        ),
        ("total", config.total_params(), d.total_elements()),
    ];

    let mut worst = 0.0f64;
    for (name, derived, actual) in cases {
        let error = (derived as f64 - actual as f64) / actual as f64;
        println!(
            "  {name:<10} derived={derived:>13} file={actual:>13} {:+.3}%",
            error * 100.0
        );
        worst = worst.max(error.abs());
    }
    assert!(
        worst < 0.01,
        "derived parameter counts drift from the file by up to {:.2}%",
        worst * 100.0,
    );
}

#[test]
fn predicted_weight_vram_matches_what_the_file_actually_needs() {
    // The VRAM budget in `docs/MODEL.md` is stated against a weights figure.
    // Resolving the schema gives the real number, so the prediction can be
    // checked rather than trusted.
    let Some(file) = open() else { return };
    let config = ModelConfig::qwen3_6_35b_a3b();
    let schema = WeightSchema::new(&config);
    let d = schema.resolve(&file).expect("schema must resolve");

    const DOCUMENTED_WEIGHTS_GIB: f64 = 29.6;
    let actual_gib = d.total_bytes() as f64 / (1 << 30) as f64;
    let error = (DOCUMENTED_WEIGHTS_GIB - actual_gib) / actual_gib;
    println!(
        "documented {DOCUMENTED_WEIGHTS_GIB:.2} GiB, file {actual_gib:.2} GiB ({:+.2}%)",
        error * 100.0
    );
    assert!(
        error.abs() < 0.02,
        "documented weight footprint is off by {:.2}%",
        error * 100.0,
    );
}

#[test]
fn the_lm_head_is_the_single_largest_tensor() {
    let Some(file) = open() else { return };
    let config = ModelConfig::qwen3_6_35b_a3b();
    let schema = WeightSchema::new(&config);
    let directory = schema.resolve(&file).expect("schema must resolve");

    let lm_head = directory.find(Role::LmHead, None).expect("LM head present");
    // Stacked expert tensors are larger as a block, but per *tensor* the
    // untied LM head is the biggest single read on the decode path, which is
    // why `docs/MODEL.md` gives it its own line.
    let largest_non_expert = directory
        .entries()
        .iter()
        .filter(|e| e.spec.role.section() != Section::Experts)
        .max_by_key(|e| e.info.n_bytes)
        .unwrap();
    assert_eq!(largest_non_expert.spec.role, Role::LmHead);
    println!(
        "LM head: {} {:?} {:.1} MB",
        lm_head.info.ggml_type.name(),
        lm_head.info.dims,
        lm_head.info.n_bytes as f64 / 1e6,
    );
}
