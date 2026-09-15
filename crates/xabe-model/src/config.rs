//! Structural configuration for Qwen3.5-family text models.
//!
//! Every derived quantity in this crate — VRAM budgets, bandwidth rooflines,
//! cache page geometry — comes from [`ModelConfig`]. It is the single place
//! the architecture is described, so that a wrong number is wrong in exactly
//! one location.
//!
//! The reference values in [`ModelConfig::qwen3_6_35b_a3b`] are transcribed from
//! Qwen's published `config.json`. [`crate::verify`] re-derives the total
//! parameter count from them as a transcription check. Runtime geometry comes
//! from [`ModelConfig::from_gguf`].

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
/// Present on *every* layer of a `qwen35moe` model, including all Gated
/// DeltaNet layers. There is no dense-layer shortcut *within* such a model —
/// but the sibling `qwen35` architecture replaces the whole block with a
/// single [`DenseFfnConfig`] MLP, which is why this hangs off [`FfnConfig`]
/// rather than off [`ModelConfig`] directly.
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

/// Dense feed-forward block geometry.
///
/// One SwiGLU MLP per layer — `down(silu(gate . x) * (up . x))` — with no
/// router, no expert stack, and no shared-expert gate. This is what the
/// `qwen35` architecture (Qwen3.8-27B) carries where `qwen35moe` carries
/// [`MoeConfig`]; everything else about the two models — the 3:1 hybrid layer
/// pattern, partial rotary at `head_dim` 256, the recurrent state geometry —
/// is the same shape at different widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseFfnConfig {
    /// Intermediate width of the MLP.
    pub intermediate: u32,
}

/// Which feed-forward block a model's layers carry.
///
/// The two variants differ in more than a width: the MoE block routes, sums
/// eight experts and gates a shared one, while the dense block is a single
/// MLP whose output goes straight to the residual. Keeping them apart in the
/// type system is what stops a dense model from silently acquiring a router,
/// and what makes "which architecture is this file" a question with one
/// answer rather than a scatter of `if num_experts > 0` tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnConfig {
    /// A 256-expert routed block plus one shared expert.
    Moe(MoeConfig),
    /// A single dense SwiGLU MLP.
    Dense(DenseFfnConfig),
}

impl FfnConfig {
    /// The MoE geometry, if this is a routed block.
    pub const fn moe(&self) -> Option<MoeConfig> {
        match self {
            Self::Moe(m) => Some(*m),
            Self::Dense(_) => None,
        }
    }

    /// The dense geometry, if this is a dense block.
    pub const fn dense(&self) -> Option<DenseFfnConfig> {
        match self {
            Self::Dense(d) => Some(*d),
            Self::Moe(_) => None,
        }
    }

    /// Width of one FFN unit's intermediate dimension.
    ///
    /// For MoE that is a *single expert's* width, not the routed total.
    pub const fn intermediate(&self) -> u32 {
        match self {
            Self::Moe(m) => m.expert_intermediate,
            Self::Dense(d) => d.intermediate,
        }
    }

    /// Number of FFN units whose weights exist in one layer.
    ///
    /// Routed experts plus shared experts, or one for a dense block.
    pub const fn units_per_layer(&self) -> u32 {
        match self {
            Self::Moe(m) => m.num_experts + m.shared_experts,
            Self::Dense(_) => 1,
        }
    }

    /// Number of FFN units a single token actually reads.
    pub const fn active_units(&self) -> u32 {
        match self {
            Self::Moe(m) => m.active_experts(),
            Self::Dense(_) => 1,
        }
    }
}

/// A GGUF whose `general.architecture` no [`ModelConfig`] describes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownArchitecture {
    /// What the file declared.
    pub declared: String,
}

impl fmt::Display for UnknownArchitecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let known: Vec<&str> = ModelConfig::KNOWN
            .iter()
            .map(|b| b().architecture)
            .collect();
        write!(
            f,
            "architecture `{}` is not implemented; this engine serves {}",
            self.declared,
            known.join(" and "),
        )
    }
}

impl core::error::Error for UnknownArchitecture {}

/// Complete structural description of the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelConfig {
    /// Human-readable model identifier.
    pub name: &'static str,
    /// The GGUF `general.architecture` string a file must declare to be this
    /// model.
    ///
    /// Also the prefix every hyperparameter key in the file carries, so it is
    /// what [`Self::hparam_key`] builds metadata lookups from rather than any
    /// module hard-coding `"qwen35moe."`.
    pub architecture: &'static str,
    /// Parameter count the model is published under (zero when unknown), used only as the centre
    /// of [`crate::verify::check_config`]'s plausibility band.
    pub advertised_params: u64,
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
    /// Feed-forward geometry, applied on every layer.
    pub ffn: FfnConfig,
    /// Native trained context length in tokens.
    pub native_context: u32,
    /// Extended context budget for reference presets; loaded files use native context.
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
            architecture: "qwen35moe",
            advertised_params: 35_000_000_000,
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
            ffn: FfnConfig::Moe(MoeConfig {
                num_experts: 256,
                experts_per_token: 8,
                shared_experts: 1,
                expert_intermediate: 512,
            }),
            native_context: 262_144,
            yarn_context: 1_010_000,
            has_mtp: true,
        }
    }

    /// The dense sibling: `unsloth/Qwen3.8-27B-GGUF`.
    ///
    /// Transcribed from the GGUF's own `qwen35.*` hyperparameters, which are
    /// what `dump the file` reports and what `llama.cpp`'s `qwen35` loader
    /// reads:
    ///
    /// ```text
    ///   block_count 65 (64 layers + one MTP head)   embedding_length 5120
    ///   attention.head_count 24 / head_count_kv 4   key/value_length 256
    ///   rope.dimension_count 64                     full_attention_interval 4
    ///   ssm.inner_size 6144  state_size 128         time_step_rank 48
    ///   ssm.group_count 16   conv_kernel 4          feed_forward_length 17408
    /// ```
    ///
    /// The GDN head split is not stated directly: `ssm.inner_size` is the
    /// value width and `ssm.group_count` the number of q/k heads, so with
    /// `ssm_norm.weight` of length 128 the head dimension is 128, giving 48
    /// value heads and 16 q/k heads. That reading is checked against the
    /// file: `attn_qkv.weight` is `[5120, 10240]` and `10240 = 2*16*128 +
    /// 48*128`.
    ///
    /// Verified against the published 27B figure by
    /// [`crate::verify::check_config`].
    pub const fn qwen3_8_27b() -> Self {
        Self {
            name: "Qwen3.8-27B",
            architecture: "qwen35",
            advertised_params: 27_000_000_000,
            num_layers: 64,
            hidden_size: 5120,
            vocab_size: 248_320,
            pattern_period: 4,
            attention_offset: 3,
            gdn: GdnConfig {
                value_heads: 48,
                qk_heads: 16,
                head_dim: 128,
                conv_kernel: 4,
                chunk_len: 64,
            },
            attention: AttentionConfig {
                q_heads: 24,
                kv_heads: 4,
                head_dim: 256,
                rope_dim: 64,
            },
            ffn: FfnConfig::Dense(DenseFfnConfig {
                intermediate: 17_408,
            }),
            native_context: 262_144,
            yarn_context: 1_010_000,
            has_mtp: true,
        }
    }

    /// Every model this engine has a transcribed configuration for.
    pub const KNOWN: [fn() -> Self; 2] = [Self::qwen3_6_35b_a3b, Self::qwen3_8_27b];

    /// The reference preset for an architecture, for examples and tests.
    /// Runtime callers must use [`Self::from_gguf`] to read actual geometry.
    ///
    /// Returns `None` rather than guessing: a file whose architecture is not
    /// listed here has hyperparameters nobody has transcribed, and inferring
    /// them from the tensor shapes would produce a model that loads and is
    /// wrong.
    pub fn for_architecture(architecture: &str) -> Option<Self> {
        Self::KNOWN
            .iter()
            .map(|f| f())
            .find(|c| c.architecture == architecture)
    }

    /// Derive geometry from a supported GGUF's architecture-scoped metadata.
    ///
    /// Named constructors remain reference presets for tests and budget examples;
    /// loading never uses their dimensions as defaults.
    pub fn from_gguf(file: &xabe_gguf::GgufFile) -> Result<Self, crate::gguf::ConfigLoadError> {
        crate::gguf::load(file)
    }

    /// The full GGUF metadata key for an architecture-scoped hyperparameter.
    ///
    /// `cfg.hparam_key("rope.freq_base")` is `"qwen35moe.rope.freq_base"` for
    /// the MoE model and `"qwen35.rope.freq_base"` for the dense one. Every
    /// hyperparameter in a GGUF is prefixed this way, so a module that spells
    /// one architecture's prefix into a literal reads nothing at all on the
    /// other — silently, since these lookups all have defaults.
    pub fn hparam_key(&self, suffix: &str) -> String {
        format!("{}.{suffix}", self.architecture)
    }

    /// The MoE geometry, if this model has one.
    pub const fn moe(&self) -> Option<MoeConfig> {
        self.ffn.moe()
    }

    /// The dense FFN geometry, if this model has one.
    pub const fn dense_ffn(&self) -> Option<DenseFfnConfig> {
        self.ffn.dense()
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

    /// Parameters in one FFN unit's three matrices.
    ///
    /// A unit is one expert on the MoE model and the whole MLP on the dense
    /// one; `gate`, `up` and `down` are each `hidden x intermediate`.
    pub const fn params_per_ffn_unit(&self) -> u64 {
        MoeConfig::MATS_PER_EXPERT as u64 * self.hidden_size as u64 * self.ffn.intermediate() as u64
    }

    /// Total parameters held in feed-forward weights across all layers.
    ///
    /// The bulk of either model: about 32.2B of 34.2B on Qwen3.6-35B-A3B,
    /// about 17.1B of 26.9B on Qwen3.8-27B.
    pub const fn total_ffn_params(&self) -> u64 {
        self.ffn.units_per_layer() as u64 * self.params_per_ffn_unit() * self.num_layers as u64
    }

    /// Feed-forward parameters actually read for a single token.
    ///
    /// Nine of 257 experts per layer on the MoE model — the sparsity the
    /// whole design exploits — and the entire MLP on the dense one, which is
    /// why the dense model reads roughly 6x the FFN weight per token despite
    /// being the smaller file.
    pub const fn active_ffn_params(&self) -> u64 {
        self.ffn.active_units() as u64 * self.params_per_ffn_unit() * self.num_layers as u64
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
            // MoE router: hidden -> num_experts, on every layer. The dense
            // model has no router at all, not a router of width one.
            if let Some(m) = self.moe() {
                total += h * u64::from(m.num_experts);
            }
        }

        total
    }

    /// Total parameter count across the whole model.
    pub fn total_params(&self) -> u64 {
        self.total_ffn_params() + self.embedding_params() + self.projection_params()
    }

    /// Parameters read to decode a single token.
    ///
    /// Active experts, the LM head, and all projections — the numerator of the
    /// weight-bandwidth term in `docs/MODEL.md`.
    pub fn active_params_per_token(&self) -> u64 {
        self.active_ffn_params() + self.lm_head_params() + self.projection_params()
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
        let share = c.total_ffn_params() as f64 / c.total_params() as f64;
        assert!(share > 0.9, "experts should be >90% of params, got {share}");
    }

    fn dense() -> ModelConfig {
        ModelConfig::qwen3_8_27b()
    }

    #[test]
    fn the_dense_config_matches_the_published_27b_figure() {
        let total = dense().total_params();
        assert!(
            (26_000_000_000..=28_000_000_000).contains(&total),
            "total params {total} outside expected band; a config field is likely wrong"
        );
    }

    #[test]
    fn the_dense_model_reads_its_whole_ffn_per_token() {
        let d = dense();
        assert_eq!(d.total_ffn_params(), d.active_ffn_params());
        // And the MoE model, emphatically, does not: 9 units of 257.
        let m = cfg();
        assert!(m.active_ffn_params() * 25 < m.total_ffn_params());
    }

    #[test]
    fn the_smaller_file_reads_far_more_ffn_weight_per_token() {
        // The dense model's whole cost story, and the reason a decode
        // expectation transplanted from the MoE model reads wrong: 27B dense
        // touches 17.1B of FFN weight per token where 35B-A3B touches 1.13B.
        // The file is *smaller*; the per-token weight traffic is 15x larger.
        let ratio = dense().active_ffn_params() as f64 / cfg().active_ffn_params() as f64;
        assert!((14.0..16.0).contains(&ratio), "ratio {ratio}");
    }

    #[test]
    fn the_dense_gdn_head_split_reproduces_the_files_qkv_width() {
        // `blk.0.attn_qkv.weight` is `[5120, 10240]` in the GGUF. The head
        // split is derived, not stated, so this is the check that it was
        // derived right.
        let g = dense().gdn;
        let qkv = 2 * g.qk_heads * g.head_dim + g.value_heads * g.head_dim;
        assert_eq!(qkv, 10_240);
        assert_eq!(g.value_heads * g.head_dim, 6_144, "ssm.inner_size");
    }

    #[test]
    fn the_dense_attention_reproduces_the_files_projection_widths() {
        // `attn_q` is `[5120, 12288]` (query and gate interleaved),
        // `attn_k`/`attn_v` `[5120, 1024]`, `attn_output` `[6144, 5120]`.
        let a = dense().attention;
        assert_eq!(a.q_heads * a.head_dim, 6_144);
        assert_eq!(a.q_heads * a.head_dim * 2, 12_288);
        assert_eq!(a.kv_heads * a.head_dim, 1_024);
        assert_eq!(a.gqa_ratio(), 6);
    }

    #[test]
    fn architectures_resolve_to_exactly_one_config_each() {
        for build in ModelConfig::KNOWN {
            let c = build();
            assert_eq!(
                ModelConfig::for_architecture(c.architecture).map(|f| f.name),
                Some(c.name)
            );
        }
        assert!(ModelConfig::for_architecture("llama").is_none());
        assert_eq!(
            cfg().hparam_key("rope.freq_base"),
            "qwen35moe.rope.freq_base"
        );
        assert_eq!(
            dense().hparam_key("rope.freq_base"),
            "qwen35.rope.freq_base"
        );
    }

    #[test]
    fn the_dense_model_holds_more_kv_and_more_recurrent_state_per_slot() {
        // 16 attention layers x 4 KV heads against 10 x 2, and 48 GDN layers
        // of 48 heads against 30 of 32. Both terms grow: 3.2x the KV per
        // token and 2.4x the recurrent state per slot, which is what makes
        // the smaller model the more expensive one to serve at depth.
        let d = dense();
        assert_eq!(d.num_attention_layers(), 16);
        assert_eq!(d.num_gdn_layers(), 48);
        assert_eq!(d.kv_bytes_per_token(2), 65_536);
        assert_eq!(d.gdn_state_bytes_per_sequence() / (1024 * 1024), 144);
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
        assert_eq!(
            c.moe()
                .expect("the MoE model has a routed block")
                .naive_gemvs_per_token(c.num_layers),
            1080
        );
    }

    #[test]
    fn attention_is_16_to_2_grouped_query() {
        let c = cfg();
        assert_eq!(c.attention.gqa_ratio(), 8);
        // Partial rotary: only a quarter of each head is rotated.
        assert!((c.attention.rope_fraction() - 0.25).abs() < 1e-12);
    }
}
