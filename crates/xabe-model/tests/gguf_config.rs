//! Metadata-only files exercise loading without weights or a CUDA device.
use xabe_gguf::GgufFile;
use xabe_model::{FfnConfig, ModelConfig};

#[derive(Clone)]
enum Value {
    Uint(u32),
    Wide(u64),
    Bools(Vec<bool>),
    Text(String),
    Tokens(u32),
}

fn fixture(c: &ModelConfig) -> Vec<(String, Value)> {
    let mut values = vec![
        (
            "general.architecture".into(),
            Value::Text(c.architecture.into()),
        ),
        ("tokenizer.ggml.tokens".into(), Value::Tokens(c.vocab_size)),
    ];
    let mut add = |key: &str, n| values.push((c.hparam_key(key), Value::Uint(n)));
    for (key, n) in [
        ("block_count", c.num_blocks()),
        ("nextn_predict_layers", c.mtp_layers()),
        ("embedding_length", c.hidden_size),
        ("full_attention_interval", c.pattern_period),
        ("context_length", c.native_context),
        ("attention.head_count", c.attention.q_heads),
        ("attention.head_count_kv", c.attention.kv_heads),
        ("attention.key_length", c.attention.head_dim),
        ("attention.value_length", c.attention.head_dim),
        ("rope.dimension_count", c.attention.rope_dim),
        ("ssm.time_step_rank", c.gdn.value_heads),
        ("ssm.group_count", c.gdn.qk_heads),
        ("ssm.state_size", c.gdn.head_dim),
        ("ssm.inner_size", c.gdn.head_dim * c.gdn.value_heads),
        ("ssm.conv_kernel", c.gdn.conv_kernel),
    ] {
        add(key, n);
    }
    match c.ffn {
        FfnConfig::Moe(m) => {
            add("expert_count", m.num_experts);
            add("expert_used_count", m.experts_per_token);
            add("expert_feed_forward_length", m.expert_intermediate);
            add("expert_shared_feed_forward_length", m.expert_intermediate);
        }
        FfnConfig::Dense(d) => add("feed_forward_length", d.intermediate),
    }
    values
}

fn file(values: &[(String, Value)]) -> GgufFile {
    file_with_tensors(values, &[])
}

fn file_with_tensors(values: &[(String, Value)], tensors: &[String]) -> GgufFile {
    fn string(bytes: &mut Vec<u8>, text: &str) {
        bytes.extend((text.len() as u64).to_le_bytes());
        bytes.extend(text.as_bytes());
    }
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend((tensors.len() as u64).to_le_bytes());
    bytes.extend((values.len() as u64).to_le_bytes());
    for (key, value) in values {
        string(&mut bytes, key);
        match value {
            Value::Uint(n) => {
                bytes.extend(4u32.to_le_bytes());
                bytes.extend(n.to_le_bytes());
            }
            Value::Wide(n) => {
                bytes.extend(10u32.to_le_bytes());
                bytes.extend(n.to_le_bytes());
            }
            Value::Bools(v) => {
                bytes.extend(9u32.to_le_bytes());
                bytes.extend(7u32.to_le_bytes());
                bytes.extend((v.len() as u64).to_le_bytes());
                bytes.extend(v.iter().map(|&b| u8::from(b)));
            }
            Value::Text(s) => {
                bytes.extend(8u32.to_le_bytes());
                string(&mut bytes, s);
            }
            Value::Tokens(n) => {
                bytes.extend(9u32.to_le_bytes());
                bytes.extend(8u32.to_le_bytes());
                bytes.extend(u64::from(*n).to_le_bytes());
                for _ in 0..*n {
                    string(&mut bytes, "t");
                }
            }
        }
    }
    // Presence-only fixtures: shapes are checked separately by WeightSchema.
    for (i, name) in tensors.iter().enumerate() {
        string(&mut bytes, name);
        bytes.extend(1u32.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.extend(0u32.to_le_bytes()); // F32
        bytes.extend((i as u64 * 32).to_le_bytes());
    }
    bytes.resize(bytes.len().next_multiple_of(32), 0);
    bytes.resize(bytes.len() + tensors.len() * 32, 0);
    GgufFile::from_bytes(bytes).unwrap()
}

fn expected(mut c: ModelConfig) -> ModelConfig {
    c.name = c.architecture;
    c.advertised_params = 0;
    c.yarn_context = c.native_context;
    c
}

#[test]
fn both_reference_geometries_are_reconstructed_without_tensor_shapes() {
    for preset in ModelConfig::KNOWN {
        let c = preset();
        assert_eq!(
            ModelConfig::from_gguf(&file(&fixture(&c))).unwrap(),
            expected(c)
        );
    }
}

#[test]
fn different_sizes_of_each_architecture_do_not_inherit_preset_geometry() {
    for preset in ModelConfig::KNOWN {
        let mut c = preset();
        c.num_layers = 12;
        c.pattern_period = 3;
        c.attention_offset = 2;
        c.hidden_size = 1024;
        c.vocab_size = 256;
        c.native_context = 8192;
        c.has_mtp = false;
        c.gdn.value_heads = 16;
        c.gdn.qk_heads = 8;
        c.attention.q_heads = 8;
        c.attention.kv_heads = 4;
        match &mut c.ffn {
            FfnConfig::Moe(m) => {
                m.num_experts = 32;
                m.experts_per_token = 4;
                m.expert_intermediate = 256;
            }
            FfnConfig::Dense(d) => d.intermediate = 2048,
        }
        let mut values = fixture(&c);
        values.retain(|(k, _)| !k.ends_with("nextn_predict_layers"));
        let loaded = ModelConfig::from_gguf(&file(&values)).unwrap();
        assert_eq!(loaded.num_attention_layers(), 4);
        assert_eq!(loaded, expected(c));
    }
}

#[test]
fn malformed_or_unrepresentable_geometry_is_rejected() {
    let mut c = ModelConfig::qwen3_6_35b_a3b();
    c.vocab_size = 32;
    for (suffix, value) in [
        ("embedding_length", Value::Text("2048".into())),
        ("block_count", Value::Uint(0)),
        ("full_attention_interval", Value::Uint(0)),
        ("nextn_predict_layers", Value::Uint(2)),
        ("ssm.inner_size", Value::Uint(17)),
        ("ssm.group_count", Value::Uint(3)),
        ("attention.value_length", Value::Uint(128)),
        ("attention.head_count_kv", Value::Uint(3)),
        ("expert_shared_feed_forward_length", Value::Uint(1024)),
        ("expert_used_count", Value::Uint(257)),
        ("rope.dimension_count", Value::Uint(65)),
    ] {
        let mut values = fixture(&c);
        values
            .iter_mut()
            .find(|(k, _)| *k == c.hparam_key(suffix))
            .unwrap()
            .1 = value;
        assert!(ModelConfig::from_gguf(&file(&values)).is_err(), "{suffix}");
    }
    let mut values = fixture(&c);
    values.retain(|(k, _)| *k != c.hparam_key("embedding_length"));
    assert!(
        ModelConfig::from_gguf(&file(&values))
            .unwrap_err()
            .to_string()
            .contains("embedding_length")
    );
    values[0].1 = Value::Text("llama".into());
    assert!(
        ModelConfig::from_gguf(&file(&values))
            .unwrap_err()
            .to_string()
            .contains("llama")
    );
}

#[test]
fn optional_metadata_defaults_and_explicit_layout_are_checked() {
    let mut c = ModelConfig::qwen3_6_35b_a3b();
    c.vocab_size = 32;
    let mut values = fixture(&c);
    values.retain(|(k, _)| !k.ends_with("full_attention_interval"));
    values
        .iter_mut()
        .find(|(k, _)| k.ends_with("embedding_length"))
        .unwrap()
        .1 = Value::Wide(2048);
    let recurrent: Vec<bool> = (0..c.num_blocks())
        .map(|i| i < c.num_layers && i % 4 != 3)
        .collect();
    values.push((
        c.hparam_key("attention.recurrent_layers"),
        Value::Bools(recurrent.clone()),
    ));
    assert_eq!(
        ModelConfig::from_gguf(&file(&values)).unwrap(),
        expected(c.clone())
    );
    let mut wrong = recurrent;
    wrong[0] = false;
    values.last_mut().unwrap().1 = Value::Bools(wrong);
    assert!(
        ModelConfig::from_gguf(&file(&values))
            .unwrap_err()
            .to_string()
            .contains("recurrent_layers")
    );
    values.pop();
    values.push((c.hparam_key("vocab_size"), Value::Uint(33)));
    assert!(
        ModelConfig::from_gguf(&file(&values))
            .unwrap_err()
            .to_string()
            .contains("vocab_size")
    );
    values.pop();
    values
        .iter_mut()
        .find(|(k, _)| k.ends_with("embedding_length"))
        .unwrap()
        .1 = Value::Wide(u64::MAX);
    assert!(
        ModelConfig::from_gguf(&file(&values))
            .unwrap_err()
            .to_string()
            .contains("exceeds u32")
    );
}

#[test]
fn mtp_availability_requires_every_head_tensor_and_preserves_trunk_geometry() {
    for preset in ModelConfig::KNOWN {
        let mut c = preset();
        c.vocab_size = 32;
        let values = fixture(&c);
        let schema = xabe_model::WeightSchema::with_mtp(&c);
        let names: Vec<String> = schema
            .specs()
            .iter()
            .filter(|s| s.layer == Some(c.num_layers))
            .map(|s| s.name.clone())
            .collect();
        assert!(!names.is_empty());
        let complete = file_with_tensors(&values, &names);
        let loaded = ModelConfig::from_gguf(&complete).unwrap();
        assert!(loaded.mtp_available(&complete));
        let stripped = file(&values);
        let stripped_config = ModelConfig::from_gguf(&stripped).unwrap();
        assert_eq!(stripped_config.num_layers, c.num_layers);
        assert_eq!(stripped_config.num_blocks(), c.num_blocks());
        assert!(!stripped_config.mtp_available(&stripped));
        for missing in &names {
            let partial: Vec<String> = names.iter().filter(|n| *n != missing).cloned().collect();
            assert!(
                !loaded.mtp_available(&file_with_tensors(&values, &partial)),
                "{missing}"
            );
        }
        let mut undeclared = loaded;
        undeclared.has_mtp = false;
        assert!(!undeclared.mtp_available(&complete));
    }
}
