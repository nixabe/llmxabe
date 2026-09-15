//! GGUF geometry for the supported Qwen3.5-family text architectures.
//!
//! Metadata interpretation follows llama.cpp `src/models/qwen35{,moe}.cpp`,
//! `load_arch_hparams` and `load_arch_tensors`. Architecture selects semantics;
//! widths come from metadata. Kernel tuning (GDN chunk length) stays local.

use core::fmt;
use xabe_gguf::{GgufFile, GgufValue};

use crate::{AttentionConfig, DenseFfnConfig, FfnConfig, GdnConfig, ModelConfig, MoeConfig};

/// Missing, malformed, or unsupported model metadata, reported before allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigLoadError(pub String);

impl fmt::Display for ConfigLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl core::error::Error for ConfigLoadError {}

fn invalid(key: &str, reason: &str) -> ConfigLoadError {
    ConfigLoadError(format!("GGUF `{key}`: {reason}"))
}

pub(crate) fn load(file: &GgufFile) -> Result<ModelConfig, ConfigLoadError> {
    let architecture = match file.get_str("general.architecture") {
        Some("qwen35moe") => "qwen35moe",
        Some("qwen35") => "qwen35",
        declared => {
            return Err(ConfigLoadError(
                crate::UnknownArchitecture {
                    declared: declared.unwrap_or("<absent>").into(),
                }
                .to_string(),
            ));
        }
    };
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let optional = |suffix: &str| -> Result<Option<u32>, ConfigLoadError> {
        let k = key(suffix);
        match file.get(&k) {
            None => Ok(None),
            Some(GgufValue::U32(v)) => Ok(Some(*v)),
            Some(GgufValue::U64(v)) => u32::try_from(*v)
                .map(Some)
                .map_err(|_| invalid(&k, "integer exceeds u32")),
            _ => Err(invalid(&k, "expected an unsigned integer")),
        }
    };
    let required = |suffix: &str| -> Result<u32, ConfigLoadError> {
        match optional(suffix)? {
            Some(v) if v > 0 => Ok(v),
            Some(_) => Err(invalid(&key(suffix), "must be non-zero")),
            None => Err(invalid(&key(suffix), "required metadata is missing")),
        }
    };
    let blocks = required("block_count")?;
    let mtp = optional("nextn_predict_layers")?.unwrap_or(0);
    if mtp > 1 || mtp >= blocks {
        return Err(invalid(
            &key("nextn_predict_layers"),
            "only zero or one MTP block after a non-empty trunk is supported",
        ));
    }
    // Same family default as llama.cpp; never a model-size preset.
    let period = optional("full_attention_interval")?.unwrap_or(4);
    if period == 0 {
        return Err(invalid(&key("full_attention_interval"), "must be non-zero"));
    }
    let head_dim = required("attention.key_length")?;
    if required("attention.value_length")? != head_dim {
        return Err(invalid(
            &key("attention.value_length"),
            "unequal key/value head dimensions are unsupported",
        ));
    }
    let gdn = GdnConfig {
        value_heads: required("ssm.time_step_rank")?,
        qk_heads: required("ssm.group_count")?,
        head_dim: required("ssm.state_size")?,
        conv_kernel: required("ssm.conv_kernel")?,
        chunk_len: 64,
    };
    if gdn.value_heads.checked_mul(gdn.head_dim) != Some(required("ssm.inner_size")?) {
        return Err(invalid(
            &key("ssm.inner_size"),
            "must equal time_step_rank * state_size",
        ));
    }
    if !gdn.value_heads.is_multiple_of(gdn.qk_heads) {
        return Err(invalid(
            &key("ssm.time_step_rank"),
            "must be a multiple of group_count",
        ));
    }
    let ffn = if architecture == "qwen35moe" {
        let intermediate = required("expert_feed_forward_length")?;
        if required("expert_shared_feed_forward_length")? != intermediate {
            return Err(invalid(
                &key("expert_shared_feed_forward_length"),
                "only one shared expert with the routed expert width is supported",
            ));
        }
        FfnConfig::Moe(MoeConfig {
            num_experts: required("expert_count")?,
            experts_per_token: required("expert_used_count")?,
            shared_experts: 1,
            expert_intermediate: intermediate,
        })
    } else {
        FfnConfig::Dense(DenseFfnConfig {
            intermediate: required("feed_forward_length")?,
        })
    };
    let tokens = file
        .get_string_array("tokenizer.ggml.tokens")
        .ok_or_else(|| {
            invalid(
                "tokenizer.ggml.tokens",
                "required string array is missing or malformed",
            )
        })?;
    let vocab_size = u32::try_from(tokens.len())
        .map_err(|_| invalid("tokenizer.ggml.tokens", "vocabulary exceeds u32"))?;
    if let Some(n) = optional("vocab_size")?
        && n != vocab_size
    {
        return Err(invalid(
            &key("vocab_size"),
            "does not match tokenizer.ggml.tokens",
        ));
    }
    let native_context = required("context_length")?;
    let config = ModelConfig {
        name: architecture,
        architecture,
        // Marketing size is not geometry and is not required GGUF metadata.
        advertised_params: 0,
        num_layers: blocks - mtp,
        hidden_size: required("embedding_length")?,
        vocab_size,
        pattern_period: period,
        attention_offset: period - 1,
        gdn,
        attention: AttentionConfig {
            q_heads: required("attention.head_count")?,
            kv_heads: required("attention.head_count_kv")?,
            head_dim,
            rope_dim: required("rope.dimension_count")?,
        },
        ffn,
        native_context,
        yarn_context: native_context,
        has_mtp: mtp == 1,
    };
    let recurrent_key = key("attention.recurrent_layers");
    if file.get(&recurrent_key).is_some() {
        let recurrent = file
            .get_bool_array(&recurrent_key)
            .ok_or_else(|| invalid(&recurrent_key, "expected a bool array"))?;
        if recurrent.len() != blocks as usize
            || recurrent.iter().enumerate().any(|(i, &v)| {
                v != (i < config.num_layers as usize
                    && config.layer_kind(i as u32) == crate::LayerKind::GatedDeltaNet)
            })
        {
            return Err(invalid(
                &recurrent_key,
                "only the declared periodic hybrid layout is supported",
            ));
        }
    }
    if !config.attention.rope_dim.is_multiple_of(2) {
        return Err(invalid(&key("rope.dimension_count"), "must be even"));
    }
    crate::verify::check_config(&config).map_err(|e| ConfigLoadError(e.to_string()))?;
    Ok(config)
}
