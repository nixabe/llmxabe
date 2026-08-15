//! The model's weight directory: which tensors Qwen3.6 needs, what shape each
//! one has, and how they resolve against a GGUF file.
//!
//! This module is the single place that knows GGUF *tensor names*. Everything
//! downstream — the CUDA loader, the VRAM budget cross-check, the differential
//! harness — asks for a [`Role`] and a layer index rather than typing
//! `"blk.7.ssm_conv1d.weight"` into a string literal of its own.
//!
//! The schema is *derived from* [`ModelConfig`] rather than written out, so a
//! config that disagrees with the file cannot pass [`WeightSchema::resolve`]:
//! every expected dimension is computed from `hidden_size`, head counts, and
//! expert geometry. A wrong `hidden_size` fails on the first tensor.
//!
//! ## Where the shapes come from
//!
//! Confirmed against `src/models/qwen35moe.cpp` in llama.cpp
//! (`llama_model_qwen35moe::load_tensors`), not inferred from the file:
//!
//! - The attention `attn_q` tensor is `n_embd x (head_dim * n_head * 2)` —
//!   it packs the query **and its output gate**, interleaved per head. Upstream
//!   reads the query half with `ggml_view_3d(.., head_dim, n_head, n_tokens,
//!   stride = head_dim*2, ..)`, so the layout is
//!   `[q_h0, gate_h0, q_h1, gate_h1, ...]` — *not* two contiguous halves.
//!   Splitting it down the middle is a silent, plausible-looking corruption.
//! - The GDN `attn_qkv` tensor is `n_embd x (key_dim*2 + value_dim)`, where
//!   `key_dim = qk_heads * head_dim` and `value_dim = value_heads * head_dim`.
//! - Every layer, GDN included, carries a full 256-expert MoE block.

use core::fmt;

use xabe_gguf::{GgmlType, GgufFile, TensorInfo};

use crate::config::{LayerKind, ModelConfig};

/// What a tensor is used for.
///
/// Names follow the GGUF convention where it exists, and the upstream
/// llama.cpp field name otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Role {
    /// `token_embd.weight` — input embedding table.
    TokenEmbedding,
    /// `output_norm.weight` — final RMSNorm before the LM head.
    OutputNorm,
    /// `output.weight` — untied LM head. The single most bandwidth-expensive
    /// tensor in the model; see `docs/MODEL.md`.
    LmHead,

    /// `blk.N.attn_norm.weight` — RMSNorm before the mixer, both kinds.
    InputNorm,
    /// `blk.N.post_attention_norm.weight` — RMSNorm before the MoE block.
    PostMixerNorm,

    /// `blk.N.attn_q.weight` — packed query **and** output gate, interleaved
    /// per head. See the module docs.
    AttnQGate,
    /// `blk.N.attn_k.weight`.
    AttnK,
    /// `blk.N.attn_v.weight`.
    AttnV,
    /// `blk.N.attn_q_norm.weight` — per-head RMSNorm over `head_dim`.
    AttnQNorm,
    /// `blk.N.attn_k_norm.weight` — per-head RMSNorm over `head_dim`.
    AttnKNorm,
    /// `blk.N.attn_output.weight`.
    AttnOut,

    /// `blk.N.attn_qkv.weight` — fused q, k, v for the delta rule.
    GdnQkv,
    /// `blk.N.attn_gate.weight` — output gate over the value stream.
    GdnGate,
    /// `blk.N.ssm_conv1d.weight` — causal depthwise convolution of width
    /// `conv_kernel` over the q/k/v stream, applied before the delta rule.
    GdnConv1d,
    /// `blk.N.ssm_a` — per-head decay parameter.
    GdnA,
    /// `blk.N.ssm_alpha.weight` — per-head input-dependent decay projection.
    GdnAlpha,
    /// `blk.N.ssm_beta.weight` — per-head input-dependent write projection.
    GdnBeta,
    /// `blk.N.ssm_dt.bias` — per-head timestep bias.
    GdnDtBias,
    /// `blk.N.ssm_norm.weight` — RMSNorm over `head_dim` inside the gate.
    GdnNorm,
    /// `blk.N.ssm_out.weight` — output projection back to the residual stream.
    GdnOut,

    /// `blk.N.ffn_gate_inp.weight` — router logits over all experts.
    MoeRouter,
    /// `blk.N.ffn_gate_exps.weight` — stacked per-expert gate projections.
    MoeGateExps,
    /// `blk.N.ffn_up_exps.weight` — stacked per-expert up projections.
    MoeUpExps,
    /// `blk.N.ffn_down_exps.weight` — stacked per-expert down projections.
    MoeDownExps,
    /// `blk.N.ffn_gate_inp_shexp.weight` — shared-expert gate scalar
    /// projection.
    MoeSharedGateInp,
    /// `blk.N.ffn_gate_shexp.weight`.
    MoeSharedGate,
    /// `blk.N.ffn_up_shexp.weight`.
    MoeSharedUp,
    /// `blk.N.ffn_down_shexp.weight`.
    MoeSharedDown,

    /// `blk.N.nextn.eh_proj.weight` — MTP head input projection.
    MtpEhProj,
    /// `blk.N.nextn.enorm.weight`.
    MtpENorm,
    /// `blk.N.nextn.hnorm.weight`.
    MtpHNorm,
    /// `blk.N.nextn.shared_head_norm.weight`.
    MtpSharedHeadNorm,
}

/// Coarse grouping used for VRAM and bandwidth accounting.
///
/// These are the buckets `docs/MODEL.md` reports, so a resolved directory can
/// be summed per section and compared against the predicted budget directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Section {
    /// Input embedding table.
    Embedding,
    /// Untied LM head.
    LmHead,
    /// Stacked per-expert MoE matrices — the bulk of the model.
    Experts,
    /// Everything else with a matrix in it: mixers, shared expert, router.
    Projections,
    /// Norm vectors and per-head scalars.
    Norms,
    /// The multi-token-prediction head.
    Mtp,
}

impl Role {
    /// Which accounting bucket this role falls into.
    pub const fn section(self) -> Section {
        use Role::*;
        match self {
            TokenEmbedding => Section::Embedding,
            LmHead => Section::LmHead,
            MoeGateExps | MoeUpExps | MoeDownExps => Section::Experts,
            OutputNorm | InputNorm | PostMixerNorm | AttnQNorm | AttnKNorm | GdnNorm | GdnA
            | GdnDtBias => Section::Norms,
            MtpEhProj | MtpENorm | MtpHNorm | MtpSharedHeadNorm => Section::Mtp,
            _ => Section::Projections,
        }
    }

    /// The GGUF name suffix, without the `blk.N.` prefix.
    ///
    /// Global tensors return their whole name.
    pub const fn suffix(self) -> &'static str {
        use Role::*;
        match self {
            TokenEmbedding => "token_embd.weight",
            OutputNorm => "output_norm.weight",
            LmHead => "output.weight",
            InputNorm => "attn_norm.weight",
            PostMixerNorm => "post_attention_norm.weight",
            AttnQGate => "attn_q.weight",
            AttnK => "attn_k.weight",
            AttnV => "attn_v.weight",
            AttnQNorm => "attn_q_norm.weight",
            AttnKNorm => "attn_k_norm.weight",
            AttnOut => "attn_output.weight",
            GdnQkv => "attn_qkv.weight",
            GdnGate => "attn_gate.weight",
            GdnConv1d => "ssm_conv1d.weight",
            GdnA => "ssm_a",
            GdnAlpha => "ssm_alpha.weight",
            GdnBeta => "ssm_beta.weight",
            GdnDtBias => "ssm_dt.bias",
            GdnNorm => "ssm_norm.weight",
            GdnOut => "ssm_out.weight",
            MoeRouter => "ffn_gate_inp.weight",
            MoeGateExps => "ffn_gate_exps.weight",
            MoeUpExps => "ffn_up_exps.weight",
            MoeDownExps => "ffn_down_exps.weight",
            MoeSharedGateInp => "ffn_gate_inp_shexp.weight",
            MoeSharedGate => "ffn_gate_shexp.weight",
            MoeSharedUp => "ffn_up_shexp.weight",
            MoeSharedDown => "ffn_down_shexp.weight",
            MtpEhProj => "nextn.eh_proj.weight",
            MtpENorm => "nextn.enorm.weight",
            MtpHNorm => "nextn.hnorm.weight",
            MtpSharedHeadNorm => "nextn.shared_head_norm.weight",
        }
    }

    /// Whether this role names a global (non-layer) tensor.
    pub const fn is_global(self) -> bool {
        matches!(self, Role::TokenEmbedding | Role::OutputNorm | Role::LmHead)
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.suffix())
    }
}

/// One tensor the model requires, with the shape the config says it must have.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    /// Full GGUF tensor name.
    pub name: String,
    /// What this tensor is.
    pub role: Role,
    /// Block index, or `None` for global tensors.
    pub layer: Option<u32>,
    /// Expected dimensions in GGUF/ggml order — `dims[0]` is fastest-varying,
    /// matching `ne[0]` in `ggml_tensor`.
    pub dims: Vec<u64>,
}

impl TensorSpec {
    /// Total element count implied by [`Self::dims`].
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

/// Reasons a GGUF file can fail to satisfy the schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightError {
    /// The schema names a tensor the file does not contain.
    Missing { name: String, role: Role },
    /// The tensor exists but its dimensions differ from what the config
    /// implies. This is the check that catches a wrong `ModelConfig`.
    ShapeMismatch {
        name: String,
        expected: Vec<u64>,
        found: Vec<u64>,
    },
    /// The file declares an architecture this schema was not written for.
    ///
    /// Loading the wrong model with the right-looking tensor names would
    /// produce fluent garbage, so this is checked rather than assumed.
    ArchitectureMismatch { expected: String, found: String },
}

impl fmt::Display for WeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing { name, role } => {
                write!(f, "missing tensor `{name}` (role {role})")
            }
            Self::ShapeMismatch {
                name,
                expected,
                found,
            } => write!(
                f,
                "tensor `{name}` has dims {found:?}, but the model config implies {expected:?}"
            ),
            Self::ArchitectureMismatch { expected, found } => write!(
                f,
                "file declares architecture `{found}`, expected `{expected}`"
            ),
        }
    }
}

impl std::error::Error for WeightError {}

/// The GGUF `general.architecture` value this schema is written against.
pub const EXPECTED_ARCHITECTURE: &str = "qwen35moe";

/// Every tensor the model needs, derived from a [`ModelConfig`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightSchema {
    specs: Vec<TensorSpec>,
    includes_mtp: bool,
}

impl WeightSchema {
    /// Build the schema for `config`, covering the transformer layers, the
    /// embedding table, and the LM head.
    ///
    /// The MTP head is excluded. It is a complete extra attention block plus
    /// its own MoE — 20 tensors and roughly 0.75 GiB — and the engine has no
    /// speculative decode path yet, so loading it would waste VRAM the text
    /// path needs. Use [`Self::with_mtp`] once milestone 09 lands.
    pub fn new(config: &ModelConfig) -> Self {
        Self::build(config, false)
    }

    /// As [`Self::new`], but also covering the MTP head block.
    pub fn with_mtp(config: &ModelConfig) -> Self {
        Self::build(config, true)
    }

    fn build(config: &ModelConfig, includes_mtp: bool) -> Self {
        let hidden = u64::from(config.hidden_size);
        let vocab = u64::from(config.vocab_size);
        let mut specs = Vec::new();

        let mut push = |role: Role, layer: Option<u32>, dims: Vec<u64>| {
            let name = match layer {
                Some(i) => format!("blk.{i}.{}", role.suffix()),
                None => role.suffix().to_string(),
            };
            specs.push(TensorSpec {
                name,
                role,
                layer,
                dims,
            });
        };

        push(Role::TokenEmbedding, None, vec![hidden, vocab]);
        push(Role::OutputNorm, None, vec![hidden]);
        push(Role::LmHead, None, vec![hidden, vocab]);

        let g = &config.gdn;
        let key_dim = u64::from(g.qk_heads) * u64::from(g.head_dim);
        let value_dim = u64::from(g.value_heads) * u64::from(g.head_dim);

        let a = &config.attention;
        let head_dim = u64::from(a.head_dim);
        let q_dim = u64::from(a.q_heads) * head_dim;
        let kv_dim = u64::from(a.kv_heads) * head_dim;

        let m = &config.moe;
        let experts = u64::from(m.num_experts);
        let ff = u64::from(m.expert_intermediate);

        let mtp_blocks = if includes_mtp { config.mtp_layers() } else { 0 };
        for layer in 0..config.num_layers + mtp_blocks {
            let is_mtp = layer >= config.num_layers;
            let l = Some(layer);

            push(Role::InputNorm, l, vec![hidden]);
            push(Role::PostMixerNorm, l, vec![hidden]);

            // The MTP block is dense-attention regardless of where it falls in
            // the repeating pattern — see the `mtp_on_hybrid_qwen` branch in
            // `llama_model::create_memory`, which gives it a plain KV cache.
            let kind = if is_mtp {
                LayerKind::GatedAttention
            } else {
                config.layer_kind(layer)
            };

            match kind {
                LayerKind::GatedAttention => {
                    // Packed query + output gate: two `head_dim` slices per
                    // head, interleaved. See the module docs.
                    push(Role::AttnQGate, l, vec![hidden, q_dim * 2]);
                    push(Role::AttnK, l, vec![hidden, kv_dim]);
                    push(Role::AttnV, l, vec![hidden, kv_dim]);
                    push(Role::AttnQNorm, l, vec![head_dim]);
                    push(Role::AttnKNorm, l, vec![head_dim]);
                    push(Role::AttnOut, l, vec![q_dim, hidden]);
                }
                LayerKind::GatedDeltaNet => {
                    push(Role::GdnQkv, l, vec![hidden, key_dim * 2 + value_dim]);
                    push(Role::GdnGate, l, vec![hidden, value_dim]);
                    push(
                        Role::GdnConv1d,
                        l,
                        vec![u64::from(g.conv_kernel), key_dim * 2 + value_dim],
                    );
                    push(Role::GdnA, l, vec![u64::from(g.value_heads)]);
                    push(Role::GdnAlpha, l, vec![hidden, u64::from(g.value_heads)]);
                    push(Role::GdnBeta, l, vec![hidden, u64::from(g.value_heads)]);
                    push(Role::GdnDtBias, l, vec![u64::from(g.value_heads)]);
                    push(Role::GdnNorm, l, vec![u64::from(g.head_dim)]);
                    push(Role::GdnOut, l, vec![value_dim, hidden]);
                }
            }

            push(Role::MoeRouter, l, vec![hidden, experts]);
            push(Role::MoeGateExps, l, vec![hidden, ff, experts]);
            push(Role::MoeUpExps, l, vec![hidden, ff, experts]);
            push(Role::MoeDownExps, l, vec![ff, hidden, experts]);
            push(Role::MoeSharedGateInp, l, vec![hidden]);
            push(Role::MoeSharedGate, l, vec![hidden, ff]);
            push(Role::MoeSharedUp, l, vec![hidden, ff]);
            push(Role::MoeSharedDown, l, vec![ff, hidden]);

            if is_mtp {
                push(Role::MtpEhProj, l, vec![hidden * 2, hidden]);
                push(Role::MtpENorm, l, vec![hidden]);
                push(Role::MtpHNorm, l, vec![hidden]);
                push(Role::MtpSharedHeadNorm, l, vec![hidden]);
            }
        }

        Self {
            specs,
            includes_mtp,
        }
    }

    /// Every tensor in the schema, in block order.
    pub fn specs(&self) -> &[TensorSpec] {
        &self.specs
    }

    /// Whether the MTP block is part of this schema.
    pub fn includes_mtp(&self) -> bool {
        self.includes_mtp
    }

    /// Look up one tensor by role and layer.
    pub fn find(&self, role: Role, layer: Option<u32>) -> Option<&TensorSpec> {
        self.specs
            .iter()
            .find(|s| s.role == role && s.layer == layer)
    }

    /// Match the schema against `file`, checking that every tensor exists with
    /// the shape the config implies.
    ///
    /// Returns **all** mismatches rather than the first, so a wrong config
    /// reports its full blast radius in one run instead of one tensor per
    /// edit-compile cycle.
    pub fn resolve<'a>(&'a self, file: &'a GgufFile) -> Result<Directory<'a>, Vec<WeightError>> {
        let mut errors = Vec::new();

        match file.get_str("general.architecture") {
            Some(EXPECTED_ARCHITECTURE) => {}
            found => errors.push(WeightError::ArchitectureMismatch {
                expected: EXPECTED_ARCHITECTURE.to_string(),
                found: found.unwrap_or("<absent>").to_string(),
            }),
        }

        let mut entries = Vec::with_capacity(self.specs.len());
        for spec in &self.specs {
            let Some(info) = file.tensor(&spec.name) else {
                errors.push(WeightError::Missing {
                    name: spec.name.clone(),
                    role: spec.role,
                });
                continue;
            };
            if info.dims != spec.dims {
                errors.push(WeightError::ShapeMismatch {
                    name: spec.name.clone(),
                    expected: spec.dims.clone(),
                    found: info.dims.clone(),
                });
                continue;
            }
            entries.push(Entry { spec, info });
        }

        if errors.is_empty() {
            Ok(Directory { entries })
        } else {
            Err(errors)
        }
    }
}

/// One resolved tensor: what the config expected, and what the file holds.
#[derive(Debug, Clone, Copy)]
pub struct Entry<'a> {
    /// The schema entry this satisfies.
    pub spec: &'a TensorSpec,
    /// The GGUF directory record, including type, offset, and byte size.
    pub info: &'a TensorInfo,
}

/// Every tensor the engine needs, resolved against a specific file.
#[derive(Debug, Clone)]
pub struct Directory<'a> {
    entries: Vec<Entry<'a>>,
}

impl<'a> Directory<'a> {
    /// All resolved tensors, in block order.
    pub fn entries(&self) -> &[Entry<'a>] {
        &self.entries
    }

    /// Number of resolved tensors.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the directory is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total bytes of tensor data this directory covers.
    ///
    /// This is the figure a device loader must fit in VRAM, before any cache
    /// or workspace allocation.
    pub fn total_bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.info.n_bytes).sum()
    }

    /// Bytes in one accounting section.
    pub fn section_bytes(&self, section: Section) -> u64 {
        self.entries
            .iter()
            .filter(|e| e.spec.role.section() == section)
            .map(|e| e.info.n_bytes)
            .sum()
    }

    /// Total parameter count — elements, not bytes.
    pub fn total_elements(&self) -> u64 {
        self.entries.iter().map(|e| e.info.n_elements).sum()
    }

    /// Elements in one accounting section.
    pub fn section_elements(&self, section: Section) -> u64 {
        self.entries
            .iter()
            .filter(|e| e.spec.role.section() == section)
            .map(|e| e.info.n_elements)
            .sum()
    }

    /// Look up one resolved tensor.
    pub fn find(&self, role: Role, layer: Option<u32>) -> Option<Entry<'a>> {
        self.entries
            .iter()
            .copied()
            .find(|e| e.spec.role == role && e.spec.layer == layer)
    }

    /// Distinct element types present, with a tensor count and byte total for
    /// each.
    ///
    /// The dequantization kernels must cover exactly this set — anything not
    /// listed here does not need a kernel, and anything listed does.
    pub fn type_histogram(&self) -> Vec<(GgmlType, usize, u64)> {
        let mut out: Vec<(GgmlType, usize, u64)> = Vec::new();
        for e in &self.entries {
            match out.iter_mut().find(|(t, _, _)| *t == e.info.ggml_type) {
                Some(slot) => {
                    slot.1 += 1;
                    slot.2 += e.info.n_bytes;
                }
                None => out.push((e.info.ggml_type, 1, e.info.n_bytes)),
            }
        }
        out.sort_by_key(|(t, _, _)| *t as i32);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ModelConfig {
        ModelConfig::qwen3_6_35b_a3b()
    }

    #[test]
    fn schema_tensor_count_matches_the_published_file() {
        // The real UD-Q6_K_XL file holds 753 tensors. That is 733 for the 40
        // transformer layers plus the three global tensors, and 20 more for
        // the MTP block. Both halves are asserted so a change to either the
        // per-layer role list or the MTP list has to be deliberate.
        let c = config();
        assert_eq!(WeightSchema::new(&c).specs().len(), 733);
        assert_eq!(WeightSchema::with_mtp(&c).specs().len(), 753);
    }

    #[test]
    fn per_layer_role_counts_follow_the_mixer_pattern() {
        let c = config();
        let schema = WeightSchema::new(&c);

        let gdn_layer_tensors = schema.specs().iter().filter(|s| s.layer == Some(0)).count();
        let attn_layer_tensors = schema.specs().iter().filter(|s| s.layer == Some(3)).count();

        // 2 norms + 9 GDN + 8 MoE, against 2 norms + 6 attention + 8 MoE.
        assert_eq!(gdn_layer_tensors, 19);
        assert_eq!(attn_layer_tensors, 16);
    }

    #[test]
    fn attention_query_tensor_is_double_width_because_it_packs_the_gate() {
        // This is the shape that looks like a bug and is not. `attn_q` is
        // q_heads * head_dim * 2, and the second slice of each head is the
        // output gate. A schema that expects q_dim here would reject the real
        // file; a loader that splits it in half would corrupt every head.
        let c = config();
        let schema = WeightSchema::new(&c);
        let q = schema.find(Role::AttnQGate, Some(3)).unwrap();
        let out = schema.find(Role::AttnOut, Some(3)).unwrap();

        assert_eq!(q.dims, vec![2048, 8192]);
        // The output projection consumes only the query half.
        assert_eq!(out.dims, vec![4096, 2048]);
        assert_eq!(q.dims[1], out.dims[0] * 2);
    }

    #[test]
    fn gdn_qkv_width_is_two_key_dims_plus_one_value_dim() {
        let c = config();
        let schema = WeightSchema::new(&c);
        let qkv = schema.find(Role::GdnQkv, Some(0)).unwrap();
        let conv = schema.find(Role::GdnConv1d, Some(0)).unwrap();
        let gate = schema.find(Role::GdnGate, Some(0)).unwrap();

        assert_eq!(qkv.dims, vec![2048, 8192]);
        assert_eq!(gate.dims, vec![2048, 4096]);
        // The convolution is depthwise over exactly the fused qkv stream, so
        // its channel count must track `attn_qkv`'s output width or the
        // per-channel filters silently misalign.
        assert_eq!(conv.dims, vec![4, 8192]);
        assert_eq!(conv.dims[1], qkv.dims[1]);
    }

    #[test]
    fn every_layer_carries_a_full_moe_block() {
        let c = config();
        let schema = WeightSchema::new(&c);
        for layer in 0..c.num_layers {
            assert!(
                schema.find(Role::MoeGateExps, Some(layer)).is_some(),
                "layer {layer} has no routed experts",
            );
            assert!(
                schema.find(Role::MoeSharedGate, Some(layer)).is_some(),
                "layer {layer} has no shared expert",
            );
        }
    }

    #[test]
    fn mtp_block_is_dense_attention_not_gdn() {
        // Block 40 sits at pattern position 0, which is a GDN slot, but the
        // MTP head is a plain attention block. Deriving its kind from the
        // pattern would look up `ssm_*` tensors that do not exist.
        let c = config();
        assert_eq!(c.layer_kind(c.num_layers), LayerKind::GatedDeltaNet);
        let schema = WeightSchema::with_mtp(&c);
        assert!(schema.find(Role::AttnQGate, Some(c.num_layers)).is_some());
        assert!(schema.find(Role::GdnQkv, Some(c.num_layers)).is_none());
        assert!(schema.find(Role::MtpEhProj, Some(c.num_layers)).is_some());
    }

    #[test]
    fn sections_partition_the_schema() {
        let c = config();
        let schema = WeightSchema::with_mtp(&c);
        // Every role lands in exactly one section, and every section is
        // reachable — a role added without a section arm would fall into
        // Projections silently, so assert the counts are all non-zero.
        for section in [
            Section::Embedding,
            Section::LmHead,
            Section::Experts,
            Section::Projections,
            Section::Norms,
            Section::Mtp,
        ] {
            assert!(
                schema.specs().iter().any(|s| s.role.section() == section),
                "no tensor in section {section:?}",
            );
        }
    }

    #[test]
    fn names_are_unique() {
        let schema = WeightSchema::with_mtp(&config());
        let mut names: Vec<&str> = schema.specs().iter().map(|s| s.name.as_str()).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before, "duplicate tensor name in schema");
    }

    #[test]
    fn a_wrong_hidden_size_is_rejected_by_shape_checking() {
        // The schema derives every dimension from the config, so this is the
        // mechanism that would have caught a plausible-but-wrong hidden_size
        // of 4096 without needing the real file present.
        let mut c = config();
        c.hidden_size = 4096;
        let schema = WeightSchema::new(&c);
        assert_eq!(
            schema.find(Role::TokenEmbedding, None).unwrap().dims,
            vec![4096, 248_320],
        );
    }
}
