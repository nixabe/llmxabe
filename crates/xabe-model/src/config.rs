//! Qwen3.6-35B-A3B structural configuration.
//!
//! Every derived quantity in this crate — VRAM budgets, bandwidth rooflines,
//! cache page geometry — comes from [`ModelConfig`]. It is the single place
//! the architecture is described, so that a wrong number is wrong in exactly
//! one location.
//!
//! The defaults in [`ModelConfig::qwen3_6_35b_a3b`] are transcribed from
//! Qwen's published `config.json`. [`crate::verify`] re-derives the total
//! parameter count from them as a transcription check.

use core::fmt;

/// Which kind of token mixer a layer uses.
///
/// The two variants have fundamentally different cache behaviour, and keeping
/// them distinct in the type system is what stops the page-geometry mistake
/// described in `AGENTS.md`: attention state grows with position, recurrent
/// state does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerKind {
    /// Gated DeltaNet — fixed-size recurrent state, O(1) in sequence length.
    GatedDeltaNet,
    /// Gated Attention — paged KV, O(n) in sequence length.
    GatedAttention,
}

impl LayerKind {
    /// Whether this layer's cache footprint grows with sequence position.
    ///
    /// Capacity accounting counts only layers for which this is true; see
    /// `docs/CACHE.md`.
    pub const fn is_positional(self) -> bool {
        matches!(self, Self::GatedAttention)
    }
}

impl fmt::Display for LayerKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::GatedDeltaNet => "gdn",
            Self::GatedAttention => "attn",
        })
    }
}

/// Gated DeltaNet layer geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnConfig {
    /// Number of value heads.
    pub value_heads: u32,
    /// Number of query/key heads.
    pub qk_heads: u32,
    /// Dimension of each head.
    pub head_dim: u32,
    /// Short convolution kernel width applied before the delta rule.
    pub conv_kernel: u32,
    /// Chunk length used by the chunked parallel form during prefill.
    ///
    /// The chunked delta rule inverts a `chunk_len x chunk_len` triangular
    /// matrix per chunk, so this trades parallelism against that cost.
    pub chunk_len: u32,
}

impl GdnConfig {
    /// Bytes of recurrent state held per layer, per sequence.
    ///
    /// The state is the delta-rule matrix `S` of shape
    /// `[value_heads, head_dim, head_dim]`, kept in fp32 because it is
    /// accumulated across the whole sequence and fp16 drifts.
    ///
    /// This is constant in sequence length — that is the entire point of the
    /// hybrid architecture, and the reason capacity accounting must not treat
    /// these layers as per-token cost.
    pub const fn state_bytes_per_layer(&self) -> u64 {
        const FP32: u64 = 4;
        self.value_heads as u64 * self.head_dim as u64 * self.head_dim as u64 * FP32
    }
}

/// Gated Attention layer geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionConfig {
    /// Number of query heads.
    pub q_heads: u32,
    /// Number of key/value heads (grouped-query attention).
    pub kv_heads: u32,
    /// Dimension of each head.
    pub head_dim: u32,
    /// Number of leading dimensions that receive rotary embedding.
    ///
    /// Qwen3.6 rotates only 64 of 256 dimensions. Partial rotary is unusual
    /// and is a documented source of subtle correctness bugs — the untouched
    /// tail must pass through byte-identical.
    pub rope_dim: u32,
}

impl AttentionConfig {
    /// Ratio of query heads to key/value heads.
    pub const fn gqa_ratio(&self) -> u32 {
        self.q_heads / self.kv_heads
    }

    /// Bytes of KV cache per token, per layer, at the given element size.
    ///
    /// Both K and V are stored, hence the factor of two.
    pub const fn kv_bytes_per_token_per_layer(&self, elem_size: u64) -> u64 {
        2 * self.kv_heads as u64 * self.head_dim as u64 * elem_size
    }

    /// Fraction of each head's dimensions that receive rotary embedding.
    pub fn rope_fraction(&self) -> f64 {
        f64::from(self.rope_dim) / f64::from(self.head_dim)
    }
}

/// Mixture-of-experts block geometry.
///
/// Present on *every* layer of this model, including all Gated DeltaNet
/// layers. There is no dense-layer shortcut.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeConfig {
    /// Total number of routed experts to choose from.
    pub num_experts: u32,
    /// Number of routed experts activated per token.
    pub experts_per_token: u32,
    /// Number of always-active shared experts.
    ///
    /// The shared expert needs no routing, sorting, or indirection, so it is
    /// hoisted out of the routed path entirely.
    pub shared_experts: u32,
    /// Intermediate width of a single expert.
    pub expert_intermediate: u32,
}

impl MoeConfig {
    /// Experts evaluated per token, routed plus shared.
    pub const fn active_experts(&self) -> u32 {
        self.experts_per_token + self.shared_experts
    }

    /// Matrices per expert: gate, up, down.
    pub const MATS_PER_EXPERT: u32 = 3;

    /// Number of separate GEMVs a naive per-expert dispatch would launch for a
    /// single token across `layers` layers.
    ///
    /// For this model that is 1,080 — the number the fused MoE path exists to
    /// eliminate. See `docs/KERNELS.md`.
    pub const fn naive_gemvs_per_token(&self, layers: u32) -> u32 {
        self.active_experts() * Self::MATS_PER_EXPERT * layers
    }
}

/// Complete structural description of the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelConfig {
    /// Human-readable model identifier.
    pub name: &'static str,
    /// Total transformer layers.
    pub num_layers: u32,
    /// Residual stream width.
    pub hidden_size: u32,
    /// Vocabulary size. Input and output embeddings are untied in this model,
    /// so this counts toward parameters twice.
    pub vocab_size: u32,
    /// Length of the repeating layer pattern.
    pub pattern_period: u32,
    /// Position of the attention layer within each pattern period.
    ///
    /// The pattern is `period - 1` Gated DeltaNet layers followed by one
    /// Gated Attention layer.
    pub attention_offset: u32,
    /// Gated DeltaNet geometry.
    pub gdn: GdnConfig,
    /// Gated Attention geometry.
    pub attention: AttentionConfig,
    /// MoE geometry, applied on every layer.
    pub moe: MoeConfig,
    /// Native trained context length in tokens.
    pub native_context: u32,
    /// Extended context length reachable with YaRN scaling.
    pub yarn_context: u32,
    /// Whether the model ships a trained multi-token-prediction head.
    pub has_mtp: bool,
}

impl ModelConfig {
    /// The target model: `unsloth/Qwen3.6-35B-A3B-GGUF`.
    ///
    /// Transcribed from Qwen's published `config.json`. Verified against the
    /// published 35B total parameter count by [`crate::verify::check_config`].
    pub const fn qwen3_6_35b_a3b() -> Self {
        Self {
            name: "Qwen3.6-35B-A3B",
            num_layers: 40,
            hidden_size: 2048,
            vocab_size: 248_320,
            pattern_period: 4,
            attention_offset: 3,
            gdn: GdnConfig {
                value_heads: 32,
                qk_heads: 16,
                head_dim: 128,
                conv_kernel: 4,
                chunk_len: 64,
            },
            attention: AttentionConfig {
                q_heads: 16,
                kv_heads: 2,
                head_dim: 256,
                rope_dim: 64,
            },
            moe: MoeConfig {
                num_experts: 256,
                experts_per_token: 8,
                shared_experts: 1,
                expert_intermediate: 512,
            },
            native_context: 262_144,
            yarn_context: 1_010_000,
            has_mtp: true,
        }
    }

    /// Number of multi-token-prediction blocks that follow the transformer
    /// layers.
    ///
    /// The GGUF file reports `block_count = 41` for a 40-layer model: block 40
    /// is the MTP head, a complete dense-attention block with its own MoE.
    /// Anything iterating "all blocks in the file" must account for it, and
    /// anything iterating "the layers that produce the next token" must not.
    pub const fn mtp_layers(&self) -> u32 {
        if self.has_mtp { 1 } else { 0 }
    }

    /// Total blocks present in the GGUF file, MTP head included.
    pub const fn num_blocks(&self) -> u32 {
        self.num_layers + self.mtp_layers()
    }

    /// Which mixer the layer at `index` uses.
    ///
    /// The pattern repeats with period [`Self::pattern_period`]: three Gated
    /// DeltaNet layers, then one Gated Attention layer.
    pub const fn layer_kind(&self, index: u32) -> LayerKind {
        if index % self.pattern_period == self.attention_offset {
            LayerKind::GatedAttention
        } else {
            LayerKind::GatedDeltaNet
        }
    }

    /// Number of layers holding growing KV cache.
    ///
    /// Only these count toward per-token cache capacity.
    pub const fn num_attention_layers(&self) -> u32 {
        self.num_layers / self.pattern_period
    }

    /// Number of layers holding fixed-size recurrent state.
    pub const fn num_gdn_layers(&self) -> u32 {
        self.num_layers - self.num_attention_layers()
    }

    /// Bytes of KV cache consumed per token across all attention layers.
    ///
    /// At f16 this is 20,480 B/token — the term that comes to dominate decode
    /// bandwidth past roughly 100K context. See `docs/MODEL.md`.
    pub const fn kv_bytes_per_token(&self, elem_size: u64) -> u64 {
        self.attention.kv_bytes_per_token_per_layer(elem_size) * self.num_attention_layers() as u64
    }

    /// Bytes of recurrent state held per sequence across all GDN layers.
    ///
    /// Constant in sequence length: about 60 MiB per slot.
    pub const fn gdn_state_bytes_per_sequence(&self) -> u64 {
        self.gdn.state_bytes_per_layer() * self.num_gdn_layers() as u64
    }

    /// Parameters in one expert's three matrices.
    pub const fn params_per_expert(&self) -> u64 {
        MoeConfig::MATS_PER_EXPERT as u64
            * self.hidden_size as u64
            * self.moe.expert_intermediate as u64
    }

    /// Total parameters held in MoE expert weights across all layers.
    ///
    /// This is the bulk of the model: about 32.2B of 34.2B.
    pub const fn total_expert_params(&self) -> u64 {
        let experts = (self.moe.num_experts + self.moe.shared_experts) as u64;
        experts * self.params_per_expert() * self.num_layers as u64
    }

    /// Expert parameters actually read for a single token.
    ///
    /// Nine of 257 experts per layer — the sparsity the whole design exploits.
    pub const fn active_expert_params(&self) -> u64 {
        self.moe.active_experts() as u64 * self.params_per_expert() * self.num_layers as u64
    }

    /// Parameters in the input and output embedding matrices combined.
    ///
    /// Untied in this model, so the vocabulary is paid for twice.
    pub const fn embedding_params(&self) -> u64 {
        2 * self.vocab_size as u64 * self.hidden_size as u64
    }

    /// Parameters in the LM head alone.
    ///
    /// A single matrix that dominates per-token weight traffic more than any
    /// other component; see `docs/MODEL.md`.
    pub const fn lm_head_params(&self) -> u64 {
        self.vocab_size as u64 * self.hidden_size as u64
    }

    /// Parameters in per-layer projections, excluding experts and embeddings.
    ///
    /// Covers GDN q/k/v/gate/beta/output projections, attention q/k/v/output
    /// projections, the MoE router, and the short convolution.
    pub fn projection_params(&self) -> u64 {
        let h = u64::from(self.hidden_size);
        let mut total = 0u64;

        for layer in 0..self.num_layers {
            total += match self.layer_kind(layer) {
                LayerKind::GatedDeltaNet => {
                    let g = &self.gdn;
                    let v_dim = u64::from(g.value_heads) * u64::from(g.head_dim);
                    let qk_dim = u64::from(g.qk_heads) * u64::from(g.head_dim);
                    // q, k projections; v and the gate at value width; the
                    // output projection back to the residual stream.
                    let proj = h * (2 * qk_dim + 2 * v_dim) + v_dim * h;
                    // Per-head scalar decay and beta gates.
                    let gates = h * u64::from(g.value_heads) * 2;
                    // Depthwise short convolution over q, k, v.
                    let conv = (2 * qk_dim + v_dim) * u64::from(g.conv_kernel);
                    proj + gates + conv
                }
                LayerKind::GatedAttention => {
                    let a = &self.attention;
                    let q_dim = u64::from(a.q_heads) * u64::from(a.head_dim);
                    let kv_dim = u64::from(a.kv_heads) * u64::from(a.head_dim);
                    // q, k, v, output projections, plus the output gate.
                    h * (q_dim + 2 * kv_dim) + q_dim * h + h * q_dim
                }
            };
            // MoE router: hidden -> num_experts, on every layer.
            total += h * u64::from(self.moe.num_experts);
        }

        total
    }

    /// Total parameter count across the whole model.
    pub fn total_params(&self) -> u64 {
        self.total_expert_params() + self.embedding_params() + self.projection_params()
    }

    /// Parameters read to decode a single token.
    ///
    /// Active experts, the LM head, and all projections — the numerator of the
    /// weight-bandwidth term in `docs/MODEL.md`.
    pub fn active_params_per_token(&self) -> u64 {
        self.active_expert_params() + self.lm_head_params() + self.projection_params()
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::qwen3_6_35b_a3b()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ModelConfig {
        ModelConfig::qwen3_6_35b_a3b()
    }

    #[test]
    fn layer_pattern_is_three_gdn_then_one_attention() {
        let c = cfg();
        let kinds: Vec<_> = (0..8).map(|i| c.layer_kind(i)).collect();
        assert_eq!(
            kinds,
            [
                LayerKind::GatedDeltaNet,
                LayerKind::GatedDeltaNet,
                LayerKind::GatedDeltaNet,
                LayerKind::GatedAttention,
                LayerKind::GatedDeltaNet,
                LayerKind::GatedDeltaNet,
                LayerKind::GatedDeltaNet,
                LayerKind::GatedAttention,
            ]
        );
    }

    #[test]
    fn layer_counts_match_published_architecture() {
        let c = cfg();
        assert_eq!(c.num_attention_layers(), 10);
        assert_eq!(c.num_gdn_layers(), 30);
        assert_eq!(
            c.num_attention_layers() + c.num_gdn_layers(),
            c.num_layers,
            "every layer must be classified exactly once"
        );
    }

    #[test]
    fn counting_layer_kinds_directly_agrees_with_the_closed_form() {
        let c = cfg();
        let attn = (0..c.num_layers)
            .filter(|&i| c.layer_kind(i).is_positional())
            .count() as u32;
        assert_eq!(attn, c.num_attention_layers());
    }

    #[test]
    fn kv_per_token_is_20480_bytes_at_f16() {
        // 10 layers x 2 KV heads x 256 head dim x 2 (K and V) x 2 bytes.
        assert_eq!(cfg().kv_bytes_per_token(2), 20_480);
    }

    #[test]
    fn gdn_state_is_about_60_mib_per_sequence() {
        let c = cfg();
        // 32 heads x 128 x 128 x fp32 = 2 MiB per layer.
        assert_eq!(c.gdn.state_bytes_per_layer(), 2 * 1024 * 1024);
        let mib = c.gdn_state_bytes_per_sequence() / (1024 * 1024);
        assert_eq!(mib, 60);
    }

    #[test]
    fn total_params_match_the_published_35b_figure() {
        let total = cfg().total_params();
        // Published as 35B; the structural decomposition gives ~34.2B. A
        // transcription error in any field moves this well outside the band.
        assert!(
            (33_500_000_000..=35_500_000_000).contains(&total),
            "total params {total} outside expected band; a config field is likely wrong"
        );
    }

    #[test]
    fn expert_weights_dominate_the_parameter_count() {
        let c = cfg();
        let share = c.total_expert_params() as f64 / c.total_params() as f64;
        assert!(share > 0.9, "experts should be >90% of params, got {share}");
    }

    #[test]
    fn active_params_are_roughly_three_billion() {
        let c = cfg();
        let active = c.active_params_per_token();
        assert!(
            (2_000_000_000..=3_500_000_000).contains(&active),
            "active params {active} should be near the advertised 3B"
        );
    }

    #[test]
    fn naive_dispatch_would_launch_1080_gemvs_per_token() {
        let c = cfg();
        assert_eq!(c.moe.naive_gemvs_per_token(c.num_layers), 1080);
    }

    #[test]
    fn attention_is_16_to_2_grouped_query() {
        let c = cfg();
        assert_eq!(c.attention.gqa_ratio(), 8);
        // Partial rotary: only a quarter of each head is rotated.
        assert!((c.attention.rope_fraction() - 0.25).abs() < 1e-12);
    }
}
