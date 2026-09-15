//! Resolve the dense schema against the real `qwen35` model file.
//!
//! The `qwen35moe` counterpart of this test lives in `real_model_weights.rs`
//! and does the same job for Qwen3.6-35B-A3B. Both matter for the same
//! reason: [`WeightSchema`] derives every expected name and dimension from
//! [`ModelConfig`] alone, so resolving cleanly against a real file is what
//! says the transcription is right — and the dense config's GDN head split
//! (48 value heads of 128, 16 q/k heads) is *derived* from `ssm.inner_size`
//! and `ssm.group_count` rather than stated, which makes it exactly the kind
//! of number that wants checking against bytes.
//!
//! If the file is absent the test SKIPS with a printed message. Per
//! `AGENTS.md`, a skipped test is reported as skipped and is not evidence the
//! schema is right.

use std::path::PathBuf;

use xabe_gguf::GgufFile;
use xabe_model::config::ModelConfig;
use xabe_model::weights::{Role, Section, WeightSchema};

const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llmxabe/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf";

fn model_path() -> PathBuf {
    std::env::var_os("LLMXABE_DENSE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH))
}

fn open() -> Option<GgufFile> {
    let path = model_path();
    if !path.exists() {
        println!(
            "SKIPPED: dense model file not found at {}; set LLMXABE_DENSE_MODEL to override",
            path.display()
        );
        return None;
    }
    Some(GgufFile::open(&path).expect("dense model file must parse as valid GGUF v3"))
}

#[test]
fn the_dense_schema_resolves_against_the_real_file_with_no_mismatches() {
    let Some(file) = open() else { return };
    let config = ModelConfig::from_gguf(&file).expect("GGUF geometry");
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
        Section::DenseFfn,
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
}

#[test]
fn the_file_declares_the_hyperparameters_the_config_was_transcribed_from() {
    let Some(file) = open() else { return };
    let c = ModelConfig::qwen3_8_27b();
    let key = |s: &str| c.hparam_key(s);

    // Stated directly in the file.
    assert_eq!(file.get_str("general.architecture"), Some("qwen35"));
    assert_eq!(file.get_u32(&key("block_count")), Some(c.num_blocks()));
    assert_eq!(file.get_u32(&key("embedding_length")), Some(c.hidden_size));
    assert_eq!(
        file.get_u32(&key("feed_forward_length")),
        Some(c.ffn.intermediate())
    );
    assert_eq!(
        file.get_u32(&key("attention.head_count")),
        Some(c.attention.q_heads)
    );
    assert_eq!(
        file.get_u32(&key("attention.head_count_kv")),
        Some(c.attention.kv_heads)
    );
    assert_eq!(
        file.get_u32(&key("attention.key_length")),
        Some(c.attention.head_dim)
    );
    assert_eq!(
        file.get_u32(&key("rope.dimension_count")),
        Some(c.attention.rope_dim)
    );
    assert_eq!(
        file.get_u32(&key("full_attention_interval")),
        Some(c.pattern_period)
    );
    assert_eq!(
        file.get_u32(&key("ssm.conv_kernel")),
        Some(c.gdn.conv_kernel)
    );

    // Derived, not stated. `ssm.inner_size` is the value width and
    // `ssm.time_step_rank` the value-head count, so their quotient is the head
    // dimension; `ssm.group_count` is the q/k head count and `ssm.state_size`
    // states the head dimension independently. All four must agree, and this
    // is what says they do.
    let inner = file
        .get_u32(&key("ssm.inner_size"))
        .expect("ssm.inner_size");
    let rank = file
        .get_u32(&key("ssm.time_step_rank"))
        .expect("ssm.time_step_rank");
    let groups = file
        .get_u32(&key("ssm.group_count"))
        .expect("ssm.group_count");
    let state = file
        .get_u32(&key("ssm.state_size"))
        .expect("ssm.state_size");
    assert_eq!(rank, c.gdn.value_heads);
    assert_eq!(groups, c.gdn.qk_heads);
    assert_eq!(state, c.gdn.head_dim);
    assert_eq!(inner, c.gdn.value_heads * c.gdn.head_dim);
}

#[test]
fn the_dense_file_carries_no_routed_tensor_at_all() {
    let Some(file) = open() else { return };
    // The check that would catch the dense config being pointed at a MoE
    // file, or a schema that kept a router "just in case".
    for role in [
        Role::MoeRouter,
        Role::MoeGateExps,
        Role::MoeUpExps,
        Role::MoeDownExps,
        Role::MoeSharedGateInp,
    ] {
        let name = format!("blk.0.{}", role.suffix());
        assert!(
            file.tensor(&name).is_none(),
            "a dense file should not hold `{name}`"
        );
    }
    assert!(file.tensor("blk.0.ffn_gate.weight").is_some());
}

#[test]
fn the_moe_schema_refuses_the_dense_file_by_architecture() {
    let Some(file) = open() else { return };
    let schema = WeightSchema::new(&ModelConfig::qwen3_6_35b_a3b());
    let errors = schema
        .resolve(&file)
        .expect_err("the routed schema must not resolve against a dense file");
    assert!(
        errors.iter().any(|e| matches!(
            e,
            xabe_model::WeightError::ArchitectureMismatch { expected, found }
                if expected == "qwen35moe" && found == "qwen35"
        )),
        "the first thing reported should be the architecture, not a shape: {errors:?}"
    );
}
