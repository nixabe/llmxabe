//! Self-consistency checks on [`ModelConfig`].
//!
//! Every resource budget in this project is derived from a handful of integers
//! transcribed from a `config.json`. A single mistyped field propagates
//! silently into VRAM planning, cache capacity, and bandwidth rooflines — and
//! shows up much later as an out-of-memory error or a wrong performance
//! conclusion.
//!
//! These checks re-derive quantities that are independently known (the
//! published parameter count, the layer pattern) and reject configurations
//! that disagree.

use crate::config::{LayerKind, ModelConfig};

/// A structural inconsistency found in a [`ModelConfig`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// A field that must be non-zero was zero.
    Zero(&'static str),
    /// The layer count is not a whole number of pattern periods, so the
    /// repeating hybrid pattern would be truncated mid-cycle.
    LayerCountNotWholePeriods { layers: u32, period: u32 },
    /// The attention layer's position within the period is outside the period.
    AttentionOffsetOutOfRange { offset: u32, period: u32 },
    /// Query heads are not a whole multiple of KV heads, so grouped-query
    /// attention has no consistent grouping.
    GqaNotDivisible { q_heads: u32, kv_heads: u32 },
    /// The rotary dimension exceeds the head dimension.
    RopeDimExceedsHeadDim { rope_dim: u32, head_dim: u32 },
    /// More experts activated per token than exist.
    MoreExpertsActiveThanExist { active: u32, total: u32 },
    /// The derived parameter count is outside the plausible band for the
    /// advertised model size — the signature of a transcription error.
    ParamCountImplausible { derived: u64 },
}

impl core::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Zero(field) => write!(f, "config field `{field}` must be non-zero"),
            Self::LayerCountNotWholePeriods { layers, period } => write!(
                f,
                "{layers} layers is not a whole number of {period}-layer periods; \
                 the hybrid pattern would be truncated"
            ),
            Self::AttentionOffsetOutOfRange { offset, period } => write!(
                f,
                "attention offset {offset} is outside the {period}-layer period, \
                 so no layer would ever be an attention layer"
            ),
            Self::GqaNotDivisible { q_heads, kv_heads } => write!(
                f,
                "{q_heads} query heads do not divide evenly into {kv_heads} KV heads"
            ),
            Self::RopeDimExceedsHeadDim { rope_dim, head_dim } => write!(
                f,
                "rotary dimension {rope_dim} exceeds head dimension {head_dim}"
            ),
            Self::MoreExpertsActiveThanExist { active, total } => write!(
                f,
                "{active} experts activated per token but only {total} routed experts exist"
            ),
            Self::ParamCountImplausible { derived } => write!(
                f,
                "derived parameter count {derived} is outside the plausible band; \
                 a config field is likely mistyped"
            ),
        }
    }
}

impl core::error::Error for ConfigError {}

/// Lower bound of the plausible parameter band, in parameters.
const MIN_PLAUSIBLE_PARAMS: u64 = 33_500_000_000;
/// Upper bound of the plausible parameter band, in parameters.
const MAX_PLAUSIBLE_PARAMS: u64 = 35_500_000_000;

/// Check a configuration for internal consistency.
///
/// Returns the first inconsistency found, or `Ok(())` if the configuration
/// describes a coherent model.
pub fn check_config(c: &ModelConfig) -> Result<(), ConfigError> {
    let nonzero = [
        ("num_layers", c.num_layers),
        ("hidden_size", c.hidden_size),
        ("vocab_size", c.vocab_size),
        ("pattern_period", c.pattern_period),
        ("gdn.value_heads", c.gdn.value_heads),
        ("gdn.qk_heads", c.gdn.qk_heads),
        ("gdn.head_dim", c.gdn.head_dim),
        ("gdn.chunk_len", c.gdn.chunk_len),
        ("attention.q_heads", c.attention.q_heads),
        ("attention.kv_heads", c.attention.kv_heads),
        ("attention.head_dim", c.attention.head_dim),
        ("moe.num_experts", c.moe.num_experts),
        ("moe.experts_per_token", c.moe.experts_per_token),
        ("moe.expert_intermediate", c.moe.expert_intermediate),
    ];
    for (name, value) in nonzero {
        if value == 0 {
            return Err(ConfigError::Zero(name));
        }
    }

    if !c.num_layers.is_multiple_of(c.pattern_period) {
        return Err(ConfigError::LayerCountNotWholePeriods {
            layers: c.num_layers,
            period: c.pattern_period,
        });
    }

    if c.attention_offset >= c.pattern_period {
        return Err(ConfigError::AttentionOffsetOutOfRange {
            offset: c.attention_offset,
            period: c.pattern_period,
        });
    }

    if !c.attention.q_heads.is_multiple_of(c.attention.kv_heads) {
        return Err(ConfigError::GqaNotDivisible {
            q_heads: c.attention.q_heads,
            kv_heads: c.attention.kv_heads,
        });
    }

    if c.attention.rope_dim > c.attention.head_dim {
        return Err(ConfigError::RopeDimExceedsHeadDim {
            rope_dim: c.attention.rope_dim,
            head_dim: c.attention.head_dim,
        });
    }

    if c.moe.experts_per_token > c.moe.num_experts {
        return Err(ConfigError::MoreExpertsActiveThanExist {
            active: c.moe.experts_per_token,
            total: c.moe.num_experts,
        });
    }

    let derived = c.total_params();
    if !(MIN_PLAUSIBLE_PARAMS..=MAX_PLAUSIBLE_PARAMS).contains(&derived) {
        return Err(ConfigError::ParamCountImplausible { derived });
    }

    Ok(())
}

/// Expand the layer pattern into an explicit per-layer listing.
///
/// Useful for constructing cache groups and for eyeballing that the pattern is
/// what the architecture description claims.
pub fn layer_layout(c: &ModelConfig) -> Vec<LayerKind> {
    (0..c.num_layers).map(|i| c.layer_kind(i)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shipped_config_is_self_consistent() {
        let c = ModelConfig::qwen3_6_35b_a3b();
        assert_eq!(check_config(&c), Ok(()));
    }

    #[test]
    fn layout_has_one_entry_per_layer() {
        let c = ModelConfig::qwen3_6_35b_a3b();
        let layout = layer_layout(&c);
        assert_eq!(layout.len(), c.num_layers as usize);
        assert_eq!(
            layout
                .iter()
                .filter(|k| **k == LayerKind::GatedAttention)
                .count(),
            10
        );
    }

    #[test]
    fn a_zeroed_field_is_rejected() {
        let mut c = ModelConfig::qwen3_6_35b_a3b();
        c.moe.num_experts = 0;
        assert_eq!(
            check_config(&c),
            Err(ConfigError::Zero("moe.num_experts")),
            "a zeroed field must not slip through into budget arithmetic"
        );
    }

    #[test]
    fn a_truncated_layer_pattern_is_rejected() {
        let mut c = ModelConfig::qwen3_6_35b_a3b();
        c.num_layers = 41;
        assert!(matches!(
            check_config(&c),
            Err(ConfigError::LayerCountNotWholePeriods { .. })
        ));
    }

    #[test]
    fn an_unreachable_attention_offset_is_rejected() {
        let mut c = ModelConfig::qwen3_6_35b_a3b();
        c.attention_offset = 4;
        assert!(matches!(
            check_config(&c),
            Err(ConfigError::AttentionOffsetOutOfRange { .. })
        ));
    }

    #[test]
    fn non_divisible_gqa_grouping_is_rejected() {
        let mut c = ModelConfig::qwen3_6_35b_a3b();
        c.attention.kv_heads = 3;
        assert!(matches!(
            check_config(&c),
            Err(ConfigError::GqaNotDivisible { .. })
        ));
    }

    #[test]
    fn a_mistyped_hidden_size_shows_up_as_an_implausible_param_count() {
        // This is the check's real purpose: catching a transcription error in
        // a field that is individually plausible.
        let mut c = ModelConfig::qwen3_6_35b_a3b();
        c.hidden_size = 4096;
        assert!(matches!(
            check_config(&c),
            Err(ConfigError::ParamCountImplausible { .. })
        ));
    }
}
