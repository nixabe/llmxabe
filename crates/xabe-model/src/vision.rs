//! Qwen3.6-35B-A3B vision tower (mmproj) structural configuration.
//!
//! The vision weights ship as a *separate* GGUF (`mmproj-*.gguf`, arch
//! `clip`, `general.type = mmproj`) sitting next to the language model. The
//! server takes its path explicitly (`--mmproj`); nothing here assumes the
//! two files travel together.
//!
//! The tower is a SigLIP-style ViT with a `qwen3vl_merger` projector: a
//! temporal-pair patch conv, a learned 48×48 position grid resized to the
//! actual patch grid, 27 pre-LayerNorm bidirectional-attention blocks, a
//! post-LayerNorm, and a 2×2 spatial merge feeding a two-layer GELU MLP
//! into the language model's 2048-wide embedding space.
//!
//! Runtime configuration is read from the mmproj GGUF metadata; the
//! tensor schema checks those dimensions against the weight directory. This model ships **no deepstack layers** — the
//! projector output is spliced into the token embedding stream at exactly
//! one point, which is what makes vision support orthogonal to the decode
//! hot path.

use core::fmt;

use xabe_gguf::GgufFile;

use crate::weights::WeightError;

/// Structural description of the vision tower and projector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisionConfig {
    /// Transformer blocks in the tower.
    pub num_layers: u32,
    /// Hidden width of the tower.
    pub hidden_size: u32,
    /// Attention heads per block. Bidirectional, no GQA.
    pub num_heads: u32,
    /// MLP intermediate width (GELU, up/down only — no gate).
    pub ffn_size: u32,
    /// Native square input the position grid was trained at, in pixels.
    pub image_size: u32,
    /// Square patch edge in pixels.
    pub patch_size: u32,
    /// Frames folded into one patch embedding. Still images duplicate
    /// their single frame across the pair.
    pub temporal_patch_size: u32,
    /// Edge of the spatial merge window; `merge^2` adjacent patches are
    /// concatenated before the projector.
    pub spatial_merge: u32,
    /// Output width of the projector — must equal the LLM hidden size.
    pub projection_dim: u32,
    /// LayerNorm epsilon (full LayerNorm with bias, not RMSNorm).
    pub ln_eps: f32,
    /// Rotary base for the vision M-RoPE applied inside each block.
    pub rope_theta: f32,
    /// Per-channel normalization: `(pixel - mean) / std`, applied after
    /// scaling bytes to `[0, 1]`.
    pub image_mean: [f32; 3],
    /// See [`Self::image_mean`].
    pub image_std: [f32; 3],
}

impl VisionConfig {
    /// The tower shipped in `mmproj-F16.gguf` for `Qwen3.6-35B-A3B`.
    ///
    /// Transcribed from that file's `clip.vision.*` metadata.
    pub const fn qwen3_6_35b_a3b() -> Self {
        Self {
            num_layers: 27,
            hidden_size: 1152,
            num_heads: 16,
            ffn_size: 4304,
            image_size: 768,
            patch_size: 16,
            temporal_patch_size: 2,
            spatial_merge: 2,
            projection_dim: 2048,
            ln_eps: 1e-6,
            rope_theta: 10_000.0,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
        }
    }

    /// The tower shipped beside `config`'s weights.
    ///
    /// The two `mmproj-F16.gguf` files this engine serves are the same tower:
    /// dumping both and diffing their `clip.*` metadata leaves exactly one
    /// differing key, `clip.vision.projection_dim`, which is 2048 for
    /// Qwen3.6-35B-A3B and 5120 for Qwen3.8-27B — the language model's hidden
    /// size in each case, since the projector's job is to land in it. Every
    /// other field (27 layers, 1152 wide, 16 heads, 4304 FFN, patch 16,
    /// spatial merge 2) is identical, and both files hold 334 tensors.
    ///
    /// So this is the shared transcription with the one dependent field
    /// derived, rather than a second near-copy that could drift from it.
    pub const fn for_model(config: &crate::config::ModelConfig) -> Self {
        Self {
            projection_dim: config.hidden_size,
            ..Self::qwen3_6_35b_a3b()
        }
    }

    /// Read the implemented qwen3vl_merger tower from its own mmproj metadata.
    /// Temporal pairing and rotary base are projector-family semantics, matching
    /// llama.cpp tools/mtmd/clip.cpp, clip_model_loader's PROJECTOR_TYPE_QWEN3VL branch.
    pub fn from_gguf(
        file: &GgufFile,
        target: &crate::ModelConfig,
    ) -> Result<Self, crate::ConfigLoadError> {
        use crate::metadata::Metadata;
        let m = Metadata(file);
        m.text("general.architecture", "clip")?;
        m.text("clip.projector_type", "qwen3vl_merger")?;
        if file.get("clip.use_gelu") != Some(&xabe_gguf::GgufValue::Bool(true)) {
            return Err(Metadata::error(
                "clip.use_gelu",
                "only GELU towers are implemented",
            ));
        }
        let c = Self {
            num_layers: m.positive("clip.vision.block_count")?,
            hidden_size: m.positive("clip.vision.embedding_length")?,
            num_heads: m.positive("clip.vision.attention.head_count")?,
            ffn_size: m.positive("clip.vision.feed_forward_length")?,
            image_size: m.positive("clip.vision.image_size")?,
            patch_size: m.positive("clip.vision.patch_size")?,
            spatial_merge: m.positive("clip.vision.spatial_merge_size")?,
            projection_dim: m.positive("clip.vision.projection_dim")?,
            ln_eps: m.float("clip.vision.attention.layer_norm_epsilon")?,
            image_mean: m.rgb("clip.vision.image_mean", false)?,
            image_std: m.rgb("clip.vision.image_std", true)?,
            temporal_patch_size: 2,
            rope_theta: 10_000.0,
        };
        if c.projection_dim != target.hidden_size {
            return Err(Metadata::error(
                "clip.vision.projection_dim",
                "must match the target hidden width",
            ));
        }
        if !c.hidden_size.is_multiple_of(c.num_heads) || !c.head_dim().is_multiple_of(4) {
            return Err(Metadata::error(
                "clip.vision.attention.head_count",
                "must divide hidden width into heads divisible by four",
            ));
        }
        if !c.image_size.is_multiple_of(c.patch_size) || c.spatial_merge != 2 {
            return Err(Metadata::error(
                "clip.vision.spatial_merge_size",
                "requires a two-by-two merger and an integral native patch grid",
            ));
        }
        if file.get("clip.vision.is_deepstack_layers").is_some() {
            let layers = file
                .get_bool_array("clip.vision.is_deepstack_layers")
                .ok_or_else(|| {
                    Metadata::error("clip.vision.is_deepstack_layers", "expected bool array")
                })?;
            if layers.len() != c.num_layers as usize || layers.iter().any(|v| *v) {
                return Err(Metadata::error(
                    "clip.vision.is_deepstack_layers",
                    "deepstack is unsupported; expected one false per layer",
                ));
            }
        }
        Ok(c)
    }

    /// Dimension of each attention head.
    pub const fn head_dim(&self) -> u32 {
        self.hidden_size / self.num_heads
    }

    /// Patches concatenated by one spatial merge window.
    pub const fn merge_factor(&self) -> u32 {
        self.spatial_merge * self.spatial_merge
    }

    /// Width of a merged patch vector entering the projector.
    pub const fn merger_input_dim(&self) -> u32 {
        self.hidden_size * self.merge_factor()
    }

    /// Edge of the learned position grid (`image_size / patch_size`).
    pub const fn pos_grid_edge(&self) -> u32 {
        self.image_size / self.patch_size
    }

    /// Elements of one flattened patch fed to the patch conv:
    /// `channels × temporal × patch × patch`.
    pub const fn patch_elems(&self) -> u32 {
        3 * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    /// Language-model tokens produced by a `grid_h × grid_w` patch grid.
    ///
    /// Both edges must already be even multiples of [`Self::spatial_merge`];
    /// preprocessing guarantees that by construction.
    pub const fn output_tokens(&self, grid_h: u32, grid_w: u32) -> u32 {
        (grid_h * grid_w) / self.merge_factor()
    }
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self::qwen3_6_35b_a3b()
    }
}

/// What a vision tensor is used for.
///
/// Names follow the mmproj GGUF convention (`v.` prefix for the tower,
/// `mm.` for the projector), as written by llama.cpp's conversion scripts
/// and read back by `tools/mtmd/clip.cpp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum VisionRole {
    /// `v.patch_embd.weight` — first temporal slice of the patch conv.
    PatchEmbed0,
    /// `v.patch_embd.weight.1` — second temporal slice.
    PatchEmbed1,
    /// `v.patch_embd.bias`.
    PatchBias,
    /// `v.position_embd.weight` — learned 48×48 position grid.
    PositionEmbed,
    /// `v.post_ln.weight` — LayerNorm after the last block.
    PostLnWeight,
    /// `v.post_ln.bias`.
    PostLnBias,

    /// `v.blk.N.ln1.weight` — LayerNorm before attention.
    Ln1Weight,
    /// `v.blk.N.ln1.bias`.
    Ln1Bias,
    /// `v.blk.N.attn_qkv.weight` — fused q, k, v.
    AttnQkvWeight,
    /// `v.blk.N.attn_qkv.bias`.
    AttnQkvBias,
    /// `v.blk.N.attn_out.weight`.
    AttnOutWeight,
    /// `v.blk.N.attn_out.bias`.
    AttnOutBias,
    /// `v.blk.N.ln2.weight` — LayerNorm before the MLP.
    Ln2Weight,
    /// `v.blk.N.ln2.bias`.
    Ln2Bias,
    /// `v.blk.N.ffn_up.weight`.
    FfnUpWeight,
    /// `v.blk.N.ffn_up.bias`.
    FfnUpBias,
    /// `v.blk.N.ffn_down.weight`.
    FfnDownWeight,
    /// `v.blk.N.ffn_down.bias`.
    FfnDownBias,

    /// `mm.0.weight` — first projector matrix over the merged patch vector.
    MergerFc1Weight,
    /// `mm.0.bias`.
    MergerFc1Bias,
    /// `mm.2.weight` — projection into the LLM embedding width.
    MergerFc2Weight,
    /// `mm.2.bias`.
    MergerFc2Bias,
}

impl VisionRole {
    /// The GGUF name suffix, without the `v.blk.N.` prefix.
    ///
    /// Global tensors return their whole name.
    pub const fn suffix(self) -> &'static str {
        use VisionRole::*;
        match self {
            PatchEmbed0 => "v.patch_embd.weight",
            PatchEmbed1 => "v.patch_embd.weight.1",
            PatchBias => "v.patch_embd.bias",
            PositionEmbed => "v.position_embd.weight",
            PostLnWeight => "v.post_ln.weight",
            PostLnBias => "v.post_ln.bias",
            Ln1Weight => "ln1.weight",
            Ln1Bias => "ln1.bias",
            AttnQkvWeight => "attn_qkv.weight",
            AttnQkvBias => "attn_qkv.bias",
            AttnOutWeight => "attn_out.weight",
            AttnOutBias => "attn_out.bias",
            Ln2Weight => "ln2.weight",
            Ln2Bias => "ln2.bias",
            FfnUpWeight => "ffn_up.weight",
            FfnUpBias => "ffn_up.bias",
            FfnDownWeight => "ffn_down.weight",
            FfnDownBias => "ffn_down.bias",
            MergerFc1Weight => "mm.0.weight",
            MergerFc1Bias => "mm.0.bias",
            MergerFc2Weight => "mm.2.weight",
            MergerFc2Bias => "mm.2.bias",
        }
    }

    /// Whether this role names a global (non-block) tensor.
    pub const fn is_global(self) -> bool {
        use VisionRole::*;
        matches!(
            self,
            PatchEmbed0
                | PatchEmbed1
                | PatchBias
                | PositionEmbed
                | PostLnWeight
                | PostLnBias
                | MergerFc1Weight
                | MergerFc1Bias
                | MergerFc2Weight
                | MergerFc2Bias
        )
    }
}

impl fmt::Display for VisionRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.suffix())
    }
}

/// One vision tensor with its expected shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionTensorSpec {
    /// Full GGUF tensor name.
    pub name: String,
    /// What this tensor is.
    pub role: VisionRole,
    /// Block index, or `None` for global tensors.
    pub layer: Option<u32>,
    /// Expected dimensions in GGUF/ggml order (`dims[0]` fastest-varying).
    pub dims: Vec<u64>,
}

/// The `general.architecture` value an mmproj GGUF declares.
pub const MMPROJ_ARCHITECTURE: &str = "clip";

/// Every tensor the vision tower needs, derived from a [`VisionConfig`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionWeightSchema {
    specs: Vec<VisionTensorSpec>,
}

impl VisionWeightSchema {
    /// Build the schema for `config`.
    pub fn new(config: &VisionConfig) -> Self {
        let h = u64::from(config.hidden_size);
        let ffn = u64::from(config.ffn_size);
        let patch = u64::from(config.patch_size);
        let grid = u64::from(config.pos_grid_edge());
        let merged = u64::from(config.merger_input_dim());
        let proj = u64::from(config.projection_dim);

        let mut specs = Vec::new();
        let mut push = |role: VisionRole, layer: Option<u32>, dims: Vec<u64>| {
            let name = match layer {
                Some(i) => format!("v.blk.{i}.{}", role.suffix()),
                None => role.suffix().to_string(),
            };
            specs.push(VisionTensorSpec {
                name,
                role,
                layer,
                dims,
            });
        };

        use VisionRole::*;
        push(PatchEmbed0, None, vec![patch, patch, 3, h]);
        push(PatchEmbed1, None, vec![patch, patch, 3, h]);
        push(PatchBias, None, vec![h]);
        push(PositionEmbed, None, vec![h, grid * grid]);
        push(PostLnWeight, None, vec![h]);
        push(PostLnBias, None, vec![h]);
        push(MergerFc1Weight, None, vec![merged, merged]);
        push(MergerFc1Bias, None, vec![merged]);
        push(MergerFc2Weight, None, vec![merged, proj]);
        push(MergerFc2Bias, None, vec![proj]);

        for layer in 0..config.num_layers {
            let l = Some(layer);
            push(Ln1Weight, l, vec![h]);
            push(Ln1Bias, l, vec![h]);
            push(AttnQkvWeight, l, vec![h, 3 * h]);
            push(AttnQkvBias, l, vec![3 * h]);
            push(AttnOutWeight, l, vec![h, h]);
            push(AttnOutBias, l, vec![h]);
            push(Ln2Weight, l, vec![h]);
            push(Ln2Bias, l, vec![h]);
            push(FfnUpWeight, l, vec![h, ffn]);
            push(FfnUpBias, l, vec![ffn]);
            push(FfnDownWeight, l, vec![ffn, h]);
            push(FfnDownBias, l, vec![h]);
        }

        Self { specs }
    }

    /// Every tensor in the schema.
    pub fn specs(&self) -> &[VisionTensorSpec] {
        &self.specs
    }

    /// Look up one tensor by role and layer.
    pub fn find(&self, role: VisionRole, layer: Option<u32>) -> Option<&VisionTensorSpec> {
        self.specs
            .iter()
            .find(|s| s.role == role && s.layer == layer)
    }

    /// Match the schema against an mmproj `file`.
    ///
    /// Checks the declared architecture (`clip`), the projector type
    /// (`qwen3vl_merger`), and every tensor's existence and shape. Returns
    /// all mismatches rather than the first, matching
    /// [`crate::weights::WeightSchema::resolve`].
    pub fn resolve(&self, file: &GgufFile) -> Result<(), Vec<WeightError>> {
        let mut errors = Vec::new();

        match file.get_str("general.architecture") {
            Some(MMPROJ_ARCHITECTURE) => {}
            found => errors.push(WeightError::ArchitectureMismatch {
                expected: MMPROJ_ARCHITECTURE.to_string(),
                found: found.unwrap_or("<absent>").to_string(),
            }),
        }
        match file.get_str("clip.projector_type") {
            Some("qwen3vl_merger") => {}
            found => errors.push(WeightError::ArchitectureMismatch {
                expected: "qwen3vl_merger".to_string(),
                found: found.unwrap_or("<absent>").to_string(),
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

    fn cfg() -> VisionConfig {
        VisionConfig::qwen3_6_35b_a3b()
    }

    #[test]
    fn head_dim_divides_evenly() {
        assert_eq!(cfg().head_dim(), 72);
        assert_eq!(cfg().head_dim() * cfg().num_heads, cfg().hidden_size);
    }

    #[test]
    fn merger_matches_the_gguf_tensor_shapes() {
        let c = cfg();
        // mm.0 is 4608x4608, mm.2 is 4608x2048 in the file.
        assert_eq!(c.merger_input_dim(), 4608);
        assert_eq!(c.projection_dim, 2048);
    }

    #[test]
    fn position_grid_is_48_by_48() {
        let c = cfg();
        assert_eq!(c.pos_grid_edge(), 48);
        // v.position_embd.weight is 2304 rows in the file.
        assert_eq!(c.pos_grid_edge() * c.pos_grid_edge(), 2304);
    }

    #[test]
    fn patch_vector_matches_the_conv_weights() {
        // Two 16x16x3 conv slices: 1536 elements together.
        assert_eq!(cfg().patch_elems(), 1536);
    }

    #[test]
    fn a_full_native_image_yields_576_tokens() {
        let c = cfg();
        let edge = c.pos_grid_edge();
        assert_eq!(c.output_tokens(edge, edge), 576);
    }

    #[test]
    fn projection_lands_in_the_llm_embedding_width() {
        use crate::config::ModelConfig;
        assert_eq!(
            cfg().projection_dim,
            ModelConfig::qwen3_6_35b_a3b().hidden_size
        );
    }

    #[test]
    fn the_tower_differs_between_the_two_models_only_in_its_projection_width() {
        use crate::ModelConfig;
        let moe = VisionConfig::for_model(&ModelConfig::qwen3_6_35b_a3b());
        let dense = VisionConfig::for_model(&ModelConfig::qwen3_8_27b());
        assert_eq!(moe, VisionConfig::qwen3_6_35b_a3b());
        assert_eq!(dense.projection_dim, 5120);
        assert_eq!(
            VisionConfig {
                projection_dim: moe.projection_dim,
                ..dense
            },
            moe,
            "the two mmproj files agree on every `clip.*` key but projection_dim"
        );
    }

    #[test]
    fn schema_tensor_count_matches_the_published_mmproj() {
        // The real mmproj-F16.gguf holds 334 tensors: 12 per block across 27
        // blocks, plus 10 globals. Asserted so a role added or dropped has to
        // be deliberate.
        let schema = VisionWeightSchema::new(&cfg());
        assert_eq!(schema.specs().len(), 334);
    }

    #[test]
    fn schema_names_are_unique() {
        let schema = VisionWeightSchema::new(&cfg());
        let mut names: Vec<&str> = schema.specs().iter().map(|s| s.name.as_str()).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(
            names.len(),
            before,
            "duplicate tensor name in vision schema"
        );
    }

    #[test]
    fn global_roles_are_exactly_the_non_block_tensors() {
        let schema = VisionWeightSchema::new(&cfg());
        for spec in schema.specs() {
            assert_eq!(
                spec.role.is_global(),
                spec.layer.is_none(),
                "{} classified inconsistently",
                spec.name
            );
        }
    }

    #[test]
    fn qkv_is_fused_three_way() {
        let schema = VisionWeightSchema::new(&cfg());
        let qkv = schema.find(VisionRole::AttnQkvWeight, Some(0)).unwrap();
        assert_eq!(qkv.dims, vec![1152, 3456]);
    }
}
