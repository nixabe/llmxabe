//! The DFlash draft model: a small dense transformer that in-fills a block
//! of masked positions in one forward pass, drafting for the big model.
//!
//! DFlash (arXiv 2602.06036) drafts differently from both n-gram lookup and
//! the MTP head: the drafter never runs autoregressively. Its key/value
//! cache over the *context* is not computed from tokens at all — it is
//! projected from the target model's own hidden states (the residual
//! stream entering a fixed set of target layers, concatenated and fused
//! through one `fc` matrix). One query pass over
//! `[last_token, MASK × n]` then predicts every masked position at once:
//! one drafter NFE per accept/verify round, regardless of `n`.
//!
//! This module owns the *shape* of that drafter — configuration transcribed
//! from the checkpoint and the tensor directory it must contain — the same
//! split [`crate::vision`] uses for the mmproj tower. The execution lives
//! in `xabe-engine`.
//!
//! Two upstream implementations of this exact architecture exist and were
//! both read before this was written: llama.cpp (`src/models/dflash.cpp`,
//! `common/speculative.cpp`'s `common_speculative_impl_draft_dflash`) and
//! vLLM (`vllm/model_executor/models/qwen3_dflash.py`,
//! `vllm/v1/spec_decode/dflash.py`). Where the two disagree — notably on
//! which residual-stream tap `target_layers` names, and on causal masking
//! for sliding-window layers — this module follows llama.cpp, because the
//! local checkpoint is a llama.cpp-ecosystem GGUF (`general.architecture =
//! dflash`) and llama.cpp is what its conversion targeted.

use core::fmt;

use xabe_gguf::GgufFile;

use crate::weights::WeightError;

/// Configuration of a DFlash draft model.
#[derive(Debug, Clone, PartialEq)]
pub struct DFlashConfig {
    /// Dense transformer blocks in the drafter.
    pub num_layers: u32,
    /// Hidden width — equal to the target's, because the context features
    /// are projected into it and the token embeddings are shared.
    pub hidden_size: u32,
    /// Query heads per block.
    pub num_q_heads: u32,
    /// Key/value heads per block (GQA).
    pub num_kv_heads: u32,
    /// Head dimension, shared by q and k and v.
    pub head_dim: u32,
    /// SwiGLU intermediate width.
    pub ffn_size: u32,
    /// RMSNorm epsilon.
    pub rms_eps: f32,
    /// Rotary base. The drafter has its own, distinct from the target's.
    pub rope_theta: f32,
    /// Sliding-window width for the layers `swa_pattern` marks.
    pub sliding_window: u32,
    /// Per-layer flag: `true` = sliding-window attention, `false` = full.
    /// The whole drafter runs **non-causally** (llama.cpp sets
    /// `llama_set_causal_attn(ctx_dft, false)`); a sliding layer masks only
    /// keys more than `sliding_window - 1` positions *behind* the query.
    pub swa_pattern: Vec<bool>,
    /// Target-model layers whose *input* residual stream is captured,
    /// concatenated in this order, and fused through `fc` into the context
    /// features. llama.cpp semantics: `t_layer_inp[il]` is the stream
    /// entering layer `il`, i.e. the state after layers `0..il` ran.
    /// (vLLM shifts these ids by one and taps one layer later; see the
    /// module docs on why llama.cpp's reading is followed.)
    pub target_layers: Vec<u32>,
    /// The block width the drafter was trained at: one committed token
    /// plus at most `block_size - 1` masked positions per query pass.
    pub block_size: u32,
    /// The vocabulary id embedded into every masked query slot.
    pub mask_token_id: u32,
}

impl DFlashConfig {
    /// The drafter shipped as `qwen36-35b-a3b-dflash-Q8_0.gguf` for
    /// `Qwen3.6-35B-A3B`, transcribed from that file's `dflash.*` metadata.
    pub fn qwen3_6_35b_a3b() -> Self {
        Self {
            num_layers: 6,
            hidden_size: 2048,
            num_q_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            ffn_size: 6144,
            rms_eps: 1e-6,
            rope_theta: 1e7,
            sliding_window: 4096,
            swa_pattern: vec![true, true, true, true, true, false],
            target_layers: vec![2, 7, 12, 17, 23, 28, 33, 38],
            block_size: 16,
            mask_token_id: 248_077,
        }
    }

    /// Width of one query projection output: `num_q_heads * head_dim`.
    pub const fn q_dim(&self) -> u32 {
        self.num_q_heads * self.head_dim
    }

    /// Width of one key or value projection output.
    pub const fn kv_dim(&self) -> u32 {
        self.num_kv_heads * self.head_dim
    }

    /// Width of the concatenated target features `fc` consumes:
    /// `target_layers.len() * hidden_size`.
    pub fn fc_input_dim(&self) -> u32 {
        self.target_layers.len() as u32 * self.hidden_size
    }

    /// Most tokens one query pass may draft: the trained block minus the
    /// committed anchor slot.
    pub const fn max_draft_tokens(&self) -> u32 {
        self.block_size - 1
    }
}

impl Default for DFlashConfig {
    fn default() -> Self {
        Self::qwen3_6_35b_a3b()
    }
}

/// What a DFlash tensor is used for. Names follow the GGUF written by the
/// llama.cpp conversion (`general.architecture = dflash`).
///
/// Deliberately absent: `token_embd.weight` and `output.weight`. This
/// checkpoint ships neither — the drafter embeds query tokens through the
/// *target's* token embedding table and scores through the target's LM
/// head, exactly as llama.cpp's decoder graph does when the draft GGUF
/// lacks its own copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DFlashRole {
    /// `fc.weight` — fuses the concatenated target features into the
    /// drafter's hidden width.
    Fc,
    /// `enc.output_norm.weight` — RMSNorm applied *after* `fc` (llama.cpp
    /// names it `output_norm_enc`, "encoder hidden_norm (after fc)").
    EncNorm,
    /// `output_norm.weight` — final RMSNorm before the (shared) LM head.
    OutputNorm,

    /// `blk.N.attn_norm.weight`.
    AttnNorm,
    /// `blk.N.attn_q.weight`.
    AttnQ,
    /// `blk.N.attn_k.weight`.
    AttnK,
    /// `blk.N.attn_v.weight`.
    AttnV,
    /// `blk.N.attn_output.weight`.
    AttnOut,
    /// `blk.N.attn_q_norm.weight` — per-head RMSNorm on q.
    AttnQNorm,
    /// `blk.N.attn_k_norm.weight` — per-head RMSNorm on k.
    AttnKNorm,
    /// `blk.N.ffn_norm.weight`.
    FfnNorm,
    /// `blk.N.ffn_gate.weight`.
    FfnGate,
    /// `blk.N.ffn_up.weight`.
    FfnUp,
    /// `blk.N.ffn_down.weight`.
    FfnDown,
}

impl DFlashRole {
    /// The GGUF name suffix, without the `blk.N.` prefix. Global tensors
    /// return their whole name.
    pub const fn suffix(self) -> &'static str {
        use DFlashRole::*;
        match self {
            Fc => "fc.weight",
            EncNorm => "enc.output_norm.weight",
            OutputNorm => "output_norm.weight",
            AttnNorm => "attn_norm.weight",
            AttnQ => "attn_q.weight",
            AttnK => "attn_k.weight",
            AttnV => "attn_v.weight",
            AttnOut => "attn_output.weight",
            AttnQNorm => "attn_q_norm.weight",
            AttnKNorm => "attn_k_norm.weight",
            FfnNorm => "ffn_norm.weight",
            FfnGate => "ffn_gate.weight",
            FfnUp => "ffn_up.weight",
            FfnDown => "ffn_down.weight",
        }
    }

    /// Whether this role names a global (non-block) tensor.
    pub const fn is_global(self) -> bool {
        use DFlashRole::*;
        matches!(self, Fc | EncNorm | OutputNorm)
    }
}

impl fmt::Display for DFlashRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.suffix())
    }
}

/// One DFlash tensor with its expected shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DFlashTensorSpec {
    /// Full GGUF tensor name.
    pub name: String,
    /// What this tensor is.
    pub role: DFlashRole,
    /// Block index, or `None` for global tensors.
    pub layer: Option<u32>,
    /// Expected dimensions in GGUF/ggml order (`dims[0]` fastest-varying).
    pub dims: Vec<u64>,
}

/// The `general.architecture` value a DFlash GGUF declares.
pub const DFLASH_ARCHITECTURE: &str = "dflash";

/// Every tensor the drafter needs, derived from a [`DFlashConfig`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DFlashWeightSchema {
    specs: Vec<DFlashTensorSpec>,
}

impl DFlashWeightSchema {
    /// Build the schema for `config`.
    pub fn new(config: &DFlashConfig) -> Self {
        let h = u64::from(config.hidden_size);
        let q = u64::from(config.q_dim());
        let kv = u64::from(config.kv_dim());
        let hd = u64::from(config.head_dim);
        let ffn = u64::from(config.ffn_size);
        let fc_in = u64::from(config.fc_input_dim());

        let mut specs = Vec::new();
        let mut push = |role: DFlashRole, layer: Option<u32>, dims: Vec<u64>| {
            let name = match layer {
                Some(i) => format!("blk.{i}.{}", role.suffix()),
                None => role.suffix().to_string(),
            };
            specs.push(DFlashTensorSpec {
                name,
                role,
                layer,
                dims,
            });
        };

        use DFlashRole::*;
        push(Fc, None, vec![fc_in, h]);
        push(EncNorm, None, vec![h]);
        push(OutputNorm, None, vec![h]);

        for layer in 0..config.num_layers {
            let l = Some(layer);
            push(AttnNorm, l, vec![h]);
            push(AttnQ, l, vec![h, q]);
            push(AttnK, l, vec![h, kv]);
            push(AttnV, l, vec![h, kv]);
            push(AttnOut, l, vec![q, h]);
            push(AttnQNorm, l, vec![hd]);
            push(AttnKNorm, l, vec![hd]);
            push(FfnNorm, l, vec![h]);
            push(FfnGate, l, vec![h, ffn]);
            push(FfnUp, l, vec![h, ffn]);
            push(FfnDown, l, vec![ffn, h]);
        }

        Self { specs }
    }

    /// Every tensor in the schema.
    pub fn specs(&self) -> &[DFlashTensorSpec] {
        &self.specs
    }

    /// Look up one tensor by role and layer.
    pub fn find(&self, role: DFlashRole, layer: Option<u32>) -> Option<&DFlashTensorSpec> {
        self.specs
            .iter()
            .find(|s| s.role == role && s.layer == layer)
    }

    /// Match the schema and `config` against a drafter `file`.
    ///
    /// Checks the declared architecture, the metadata this config was
    /// transcribed from (layer count, widths, the target-layer list, the
    /// mask token), and every tensor's existence and shape. Returns all
    /// mismatches rather than the first, matching
    /// [`crate::weights::WeightSchema::resolve`].
    pub fn resolve(&self, config: &DFlashConfig, file: &GgufFile) -> Result<(), Vec<WeightError>> {
        let mut errors = Vec::new();

        match file.get_str("general.architecture") {
            Some(DFLASH_ARCHITECTURE) => {}
            found => errors.push(WeightError::ArchitectureMismatch {
                expected: DFLASH_ARCHITECTURE.to_string(),
                found: found.unwrap_or("<absent>").to_string(),
            }),
        }

        let mut check_u32 = |key: &str, expected: u32| match file.get_u32(key) {
            Some(found) if found == expected => {}
            found => errors.push(WeightError::ArchitectureMismatch {
                expected: format!("{key} = {expected}"),
                found: found.map_or_else(|| "<absent>".to_string(), |v| v.to_string()),
            }),
        };
        check_u32("dflash.block_count", config.num_layers);
        check_u32("dflash.embedding_length", config.hidden_size);
        check_u32("dflash.attention.head_count", config.num_q_heads);
        check_u32("dflash.attention.head_count_kv", config.num_kv_heads);
        check_u32("dflash.attention.key_length", config.head_dim);
        check_u32("dflash.feed_forward_length", config.ffn_size);
        check_u32("dflash.block_size", config.block_size);
        check_u32("dflash.attention.sliding_window", config.sliding_window);
        check_u32("tokenizer.ggml.mask_token_id", config.mask_token_id);

        match file.get_i32_array("dflash.target_layers") {
            Some(found)
                if found.len() == config.target_layers.len()
                    && found
                        .iter()
                        .zip(&config.target_layers)
                        .all(|(&a, &b)| a == b as i32) => {}
            found => errors.push(WeightError::ArchitectureMismatch {
                expected: format!("dflash.target_layers = {:?}", config.target_layers),
                found: found.map_or_else(|| "<absent>".to_string(), |v| format!("{v:?}")),
            }),
        }

        for spec in &self.specs {
            let Some(info) = file.tensor(&spec.name) else {
                errors.push(WeightError::Missing {
                    name: spec.name.clone(),
                });
                continue;
            };
            if info.dims != spec.dims {
                errors.push(WeightError::ShapeMismatch {
                    name: spec.name.clone(),
                    expected: spec.dims.clone(),
                    found: info.dims.clone(),
                });
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> DFlashConfig {
        DFlashConfig::qwen3_6_35b_a3b()
    }

    #[test]
    fn the_derived_widths_match_the_checkpoint() {
        let c = cfg();
        assert_eq!(c.q_dim(), 4096);
        assert_eq!(c.kv_dim(), 1024);
        assert_eq!(c.fc_input_dim(), 16384);
        assert_eq!(c.max_draft_tokens(), 15);
        assert_eq!(c.swa_pattern.len(), c.num_layers as usize);
    }

    #[test]
    fn the_schema_covers_three_globals_and_eleven_tensors_per_block() {
        let schema = DFlashWeightSchema::new(&cfg());
        assert_eq!(schema.specs().len(), 3 + 6 * 11);
        assert_eq!(
            schema.find(DFlashRole::Fc, None).unwrap().dims,
            vec![16384, 2048],
        );
        assert_eq!(
            schema.find(DFlashRole::AttnQ, Some(5)).unwrap().name,
            "blk.5.attn_q.weight",
        );
        assert_eq!(
            schema.find(DFlashRole::AttnOut, Some(0)).unwrap().dims,
            vec![4096, 2048],
        );
    }

    #[test]
    fn every_role_appears_exactly_where_its_kind_says() {
        let schema = DFlashWeightSchema::new(&cfg());
        for spec in schema.specs() {
            assert_eq!(
                spec.layer.is_none(),
                spec.role.is_global(),
                "{} placement disagrees with its role",
                spec.name,
            );
        }
    }
}
