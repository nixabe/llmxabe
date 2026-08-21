//! Device-resident vision tower: encodes preprocessed images into
//! language-model embedding rows.
//!
//! Orchestrates [`xabe_cuda::kernels::vision::VisionKernels`] into the full
//! `qwen3vl_merger` pipeline whose reference and provenance live in
//! [`xabe_kernels::vision`]. One instance per worker device, constructed at
//! startup with every device and host buffer pre-allocated for
//! [`VisionForward::max_patches`] (AGENTS.md rule 6) — encoding allocates
//! nothing.
//!
//! The pass runs on the worker's stream *between* engine steps, at
//! admission time. It is never captured: prefill itself is not captured
//! either, and the decode graphs never see any of these buffers, which is
//! what keeps this feature structurally off the text hot path.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::f16;

use xabe_cuda::kernels::vision::{VisionError, VisionKernels};
use xabe_gguf::GgufFile;
use xabe_kernels::vision::tower::{VisionBlockWeights, VisionWeights, resized_pos_embed};
use xabe_kernels::vision::{PreprocessedImage, cell_order_index};
use xabe_model::VisionConfig;
use xabe_model::vision::{VisionRole, VisionWeightSchema};

/// Load and widen the mmproj's tensors into the reference weight layout.
///
/// Everything becomes f32 host-side (the file mixes f16 matrices with f32
/// vectors); [`VisionForward::new`] rounds back to f16 on upload, so the
/// device sees exactly the file's f16 matrix values. The two temporal
/// patch-conv slices are summed here — still images feed both the same
/// frame (see `xabe_kernels::vision::tower::VisionWeights`).
pub fn load_vision_weights(file: &GgufFile, cfg: &VisionConfig) -> Result<VisionWeights, String> {
    let schema = VisionWeightSchema::new(cfg);
    if let Err(errors) = schema.resolve(file) {
        let mut msg = format!(
            "mmproj does not match the vision schema ({}):",
            errors.len()
        );
        for e in errors.iter().take(8) {
            msg.push_str(&format!("\n  {e}"));
        }
        return Err(msg);
    }

    let tensor_f32 = |role: VisionRole, layer: Option<u32>| -> Result<Vec<f32>, String> {
        let spec = schema
            .find(role, layer)
            .expect("schema covers every role it was built from");
        let info = file
            .tensor(&spec.name)
            .expect("resolve() checked existence");
        let bytes = file
            .tensor_bytes(&spec.name)
            .expect("resolve() checked existence");
        match info.ggml_type {
            xabe_gguf::GgmlType::F32 => Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()),
            xabe_gguf::GgmlType::F16 => Ok(bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()),
            other => Err(format!("{}: unsupported tensor type {other:?}", spec.name)),
        }
    };

    use VisionRole::*;
    let mut patch_embed = tensor_f32(PatchEmbed0, None)?;
    for (a, b) in patch_embed.iter_mut().zip(tensor_f32(PatchEmbed1, None)?) {
        *a += b;
    }
    Ok(VisionWeights {
        patch_embed,
        patch_bias: tensor_f32(PatchBias, None)?,
        pos_embed: tensor_f32(PositionEmbed, None)?,
        blocks: (0..cfg.num_layers)
            .map(|l| {
                Ok(VisionBlockWeights {
                    ln1_w: tensor_f32(Ln1Weight, Some(l))?,
                    ln1_b: tensor_f32(Ln1Bias, Some(l))?,
                    qkv_w: tensor_f32(AttnQkvWeight, Some(l))?,
                    qkv_b: tensor_f32(AttnQkvBias, Some(l))?,
                    out_w: tensor_f32(AttnOutWeight, Some(l))?,
                    out_b: tensor_f32(AttnOutBias, Some(l))?,
                    ln2_w: tensor_f32(Ln2Weight, Some(l))?,
                    ln2_b: tensor_f32(Ln2Bias, Some(l))?,
                    up_w: tensor_f32(FfnUpWeight, Some(l))?,
                    up_b: tensor_f32(FfnUpBias, Some(l))?,
                    down_w: tensor_f32(FfnDownWeight, Some(l))?,
                    down_b: tensor_f32(FfnDownBias, Some(l))?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
        post_ln_w: tensor_f32(PostLnWeight, None)?,
        post_ln_b: tensor_f32(PostLnBias, None)?,
        fc1_w: tensor_f32(MergerFc1Weight, None)?,
        fc1_b: tensor_f32(MergerFc1Bias, None)?,
        fc2_w: tensor_f32(MergerFc2Weight, None)?,
        fc2_b: tensor_f32(MergerFc2Bias, None)?,
    })
}

/// Errors constructing or running the device vision tower.
#[derive(Debug)]
pub enum VisionForwardError {
    /// The kernels or a launch failed.
    Kernels(VisionError),
    /// The driver failed.
    Driver(cudarc::driver::DriverError),
    /// An image exceeds the pre-allocated patch budget.
    TooManyPatches { patches: usize, max: usize },
    /// The weights disagree with the config's shapes.
    WrongWeights { what: &'static str },
}

impl std::fmt::Display for VisionForwardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Kernels(e) => write!(f, "vision kernels: {e}"),
            Self::Driver(e) => write!(f, "driver: {e}"),
            Self::TooManyPatches { patches, max } => write!(
                f,
                "image has {patches} patches, above the pre-allocated budget of {max}"
            ),
            Self::WrongWeights { what } => write!(f, "vision weights: bad {what}"),
        }
    }
}

impl std::error::Error for VisionForwardError {}

impl From<VisionError> for VisionForwardError {
    fn from(e: VisionError) -> Self {
        Self::Kernels(e)
    }
}

impl From<cudarc::driver::DriverError> for VisionForwardError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        Self::Driver(e)
    }
}

/// One block's device weights, f16.
struct DeviceBlock {
    ln1_w: CudaSlice<f16>,
    ln1_b: CudaSlice<f16>,
    qkv_w: CudaSlice<f16>,
    qkv_b: CudaSlice<f16>,
    out_w: CudaSlice<f16>,
    out_b: CudaSlice<f16>,
    ln2_w: CudaSlice<f16>,
    ln2_b: CudaSlice<f16>,
    up_w: CudaSlice<f16>,
    up_b: CudaSlice<f16>,
    down_w: CudaSlice<f16>,
    down_b: CudaSlice<f16>,
}

/// The vision tower resident on one device.
pub struct VisionForward {
    kernels: VisionKernels,
    stream: Arc<CudaStream>,
    cfg: VisionConfig,
    max_patches: usize,
    /// Heads whose score matrices are materialized at once; bounds the
    /// `heads_chunk * max_patches^2` score workspace.
    head_chunk: usize,

    // Weights, f16.
    patch_embed: CudaSlice<f16>,
    patch_bias: CudaSlice<f16>,
    blocks: Vec<DeviceBlock>,
    post_ln_w: CudaSlice<f16>,
    post_ln_b: CudaSlice<f16>,
    fc1_w: CudaSlice<f16>,
    fc1_b: CudaSlice<f16>,
    fc2_w: CudaSlice<f16>,
    fc2_b: CudaSlice<f16>,
    /// Raw 48×48 grid kept on the host for per-image bilinear resize.
    pos_embed_host: Vec<f32>,

    // Activations, pre-sized to `max_patches`.
    patches_in: CudaSlice<f16>,
    x: CudaSlice<f16>,
    normed: CudaSlice<f16>,
    qkv: CudaSlice<f16>,
    attn: CudaSlice<f16>,
    ffn: CudaSlice<f16>,
    scores: CudaSlice<f16>,
    pos_embed_img: CudaSlice<f16>,
    positions: CudaSlice<i32>,
    merged_mid: CudaSlice<f16>,
    out_f16: CudaSlice<f16>,
    out_f32: CudaSlice<f32>,

    // Reused host staging (rule 6: no per-image allocation).
    stage_f16: Vec<f16>,
    stage_pos: Vec<i32>,
    stage_f32: Vec<f32>,
}

fn to_f16(v: &[f32]) -> Vec<f16> {
    v.iter().map(|&x| f16::from_f32(x)).collect()
}

impl VisionForward {
    /// Upload `weights` and pre-allocate every buffer for images up to
    /// `max_patches` patches (`max_patches / 4` output tokens).
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
        cfg: &VisionConfig,
        weights: &VisionWeights,
        max_patches: usize,
    ) -> Result<Self, VisionForwardError> {
        let hidden = cfg.hidden_size as usize;
        let ffn = cfg.ffn_size as usize;
        let patch_len = (cfg.patch_elems() / cfg.temporal_patch_size) as usize;
        let merged = cfg.merger_input_dim() as usize;
        let proj = cfg.projection_dim as usize;
        let heads = cfg.num_heads as usize;
        let merge = cfg.merge_factor() as usize;
        assert!(
            max_patches.is_multiple_of(merge) && max_patches > 0,
            "max_patches must be a positive multiple of the merge factor"
        );

        let check = |what: &'static str, got: usize, want: usize| {
            if got == want {
                Ok(())
            } else {
                Err(VisionForwardError::WrongWeights { what })
            }
        };
        check("patch_embed", weights.patch_embed.len(), patch_len * hidden)?;
        check("pos_embed", weights.pos_embed.len(), {
            let edge = cfg.pos_grid_edge() as usize;
            edge * edge * hidden
        })?;
        check("blocks", weights.blocks.len(), cfg.num_layers as usize)?;

        let kernels = VisionKernels::new(ctx, stream.clone())?;
        let up = |v: &[f32]| -> Result<CudaSlice<f16>, VisionForwardError> {
            Ok(stream.clone_htod(&to_f16(v))?)
        };

        let blocks = weights
            .blocks
            .iter()
            .map(|b| {
                Ok(DeviceBlock {
                    ln1_w: up(&b.ln1_w)?,
                    ln1_b: up(&b.ln1_b)?,
                    qkv_w: up(&b.qkv_w)?,
                    qkv_b: up(&b.qkv_b)?,
                    out_w: up(&b.out_w)?,
                    out_b: up(&b.out_b)?,
                    ln2_w: up(&b.ln2_w)?,
                    ln2_b: up(&b.ln2_b)?,
                    up_w: up(&b.up_w)?,
                    up_b: up(&b.up_b)?,
                    down_w: up(&b.down_w)?,
                    down_b: up(&b.down_b)?,
                })
            })
            .collect::<Result<Vec<_>, VisionForwardError>>()?;

        // 4 heads of f16 scores at the largest grid; 4 × max_patches² × 2 B.
        let head_chunk = heads.min(4);
        let max_tokens = max_patches / merge;

        Ok(Self {
            patch_embed: up(&weights.patch_embed)?,
            patch_bias: up(&weights.patch_bias)?,
            blocks,
            post_ln_w: up(&weights.post_ln_w)?,
            post_ln_b: up(&weights.post_ln_b)?,
            fc1_w: up(&weights.fc1_w)?,
            fc1_b: up(&weights.fc1_b)?,
            fc2_w: up(&weights.fc2_w)?,
            fc2_b: up(&weights.fc2_b)?,
            pos_embed_host: weights.pos_embed.clone(),
            patches_in: stream.alloc_zeros(max_patches * patch_len)?,
            x: stream.alloc_zeros(max_patches * hidden)?,
            normed: stream.alloc_zeros(max_patches * hidden)?,
            qkv: stream.alloc_zeros(max_patches * 3 * hidden)?,
            attn: stream.alloc_zeros(max_patches * hidden)?,
            ffn: stream.alloc_zeros(max_patches * ffn)?,
            scores: stream.alloc_zeros(head_chunk * max_patches * max_patches)?,
            pos_embed_img: stream.alloc_zeros(max_patches * hidden)?,
            positions: stream.alloc_zeros(2 * max_patches)?,
            merged_mid: stream.alloc_zeros(max_tokens * merged)?,
            out_f16: stream.alloc_zeros(max_tokens * proj)?,
            out_f32: stream.alloc_zeros(max_tokens * proj)?,
            stage_f16: Vec::with_capacity(max_patches * patch_len),
            stage_pos: Vec::with_capacity(2 * max_patches),
            stage_f32: Vec::with_capacity(max_tokens * proj),
            kernels,
            stream,
            cfg: *cfg,
            max_patches,
            head_chunk,
        })
    }

    /// The patch budget this instance was allocated for.
    pub fn max_patches(&self) -> usize {
        self.max_patches
    }

    /// The tower's structural config.
    pub fn config(&self) -> &VisionConfig {
        &self.cfg
    }

    /// Encode one preprocessed image; the embeddings land in the returned
    /// device buffer's first `output_tokens * projection_dim` floats.
    ///
    /// Synchronizes the stream before returning, so the buffer is
    /// immediately consumable (and re-entrant use of the shared activation
    /// buffers is safe).
    pub fn encode(
        &mut self,
        image: &PreprocessedImage,
    ) -> Result<&CudaSlice<f32>, VisionForwardError> {
        let cfg = self.cfg;
        let hidden = cfg.hidden_size as usize;
        let heads = cfg.num_heads as usize;
        let d_head = cfg.head_dim() as usize;
        let ffn = cfg.ffn_size as usize;
        let patch_len = (cfg.patch_elems() / cfg.temporal_patch_size) as usize;
        let (gh, gw) = (image.grid_h, image.grid_w);
        let n = (gh * gw) as usize;
        if n > self.max_patches {
            return Err(VisionForwardError::TooManyPatches {
                patches: n,
                max: self.max_patches,
            });
        }
        assert_eq!(image.patches.len(), n * patch_len, "patch buffer size");
        let stream = &self.stream;

        // Host-side per-image inputs: patches, resized position grid, and
        // (row, col) rope positions, all in cell order.
        self.stage_f16.clear();
        self.stage_f16
            .extend(image.patches.iter().map(|&v| f16::from_f32(v)));
        stream.memcpy_htod(
            &self.stage_f16,
            &mut self.patches_in.slice_mut(..n * patch_len),
        )?;

        let pos = resized_pos_embed(
            &self.pos_embed_host,
            hidden,
            cfg.pos_grid_edge() as usize,
            gh,
            gw,
        );
        self.stage_f16.clear();
        self.stage_f16.extend(pos.iter().map(|&v| f16::from_f32(v)));
        stream.memcpy_htod(
            &self.stage_f16,
            &mut self.pos_embed_img.slice_mut(..n * hidden),
        )?;

        self.stage_pos.clear();
        self.stage_pos.resize(2 * n, 0);
        for y in 0..gh {
            for x in 0..gw {
                let i = cell_order_index(x, y, gw);
                self.stage_pos[2 * i] = y as i32;
                self.stage_pos[2 * i + 1] = x as i32;
            }
        }
        stream.memcpy_htod(&self.stage_pos, &mut self.positions.slice_mut(..2 * n))?;

        // Patch embedding + position embedding.
        self.kernels.proj(
            &self.patches_in,
            &self.patch_embed,
            Some(&self.patch_bias),
            &mut self.x,
            n,
            patch_len,
            hidden,
            0.0,
        )?;
        self.kernels
            .add_assign(stream, &mut self.x, &self.pos_embed_img, n * hidden)?;

        let scale = 1.0 / (d_head as f32).sqrt();
        for blk in &self.blocks {
            // Attention.
            self.kernels.layer_norm(
                stream,
                &self.x,
                &mut self.normed,
                &blk.ln1_w,
                &blk.ln1_b,
                n,
                hidden,
                cfg.ln_eps,
            )?;
            self.kernels.proj(
                &self.normed,
                &blk.qkv_w,
                Some(&blk.qkv_b),
                &mut self.qkv,
                n,
                hidden,
                3 * hidden,
                0.0,
            )?;
            self.kernels.rope_qk(
                stream,
                &mut self.qkv,
                &self.positions,
                n,
                hidden,
                d_head,
                cfg.rope_theta,
            )?;
            let mut h0 = 0usize;
            while h0 < heads {
                let chunk = self.head_chunk.min(heads - h0);
                self.kernels.attn_scores(
                    &self.qkv,
                    h0 * d_head,
                    hidden + h0 * d_head,
                    &mut self.scores,
                    n,
                    3 * hidden,
                    d_head,
                    chunk,
                    scale,
                )?;
                self.kernels
                    .softmax_rows(stream, &mut self.scores, chunk * n, n)?;
                self.kernels.attn_output(
                    &self.scores,
                    &self.qkv,
                    2 * hidden + h0 * d_head,
                    &mut self.attn,
                    h0 * d_head,
                    n,
                    3 * hidden,
                    hidden,
                    d_head,
                    chunk,
                )?;
                h0 += chunk;
            }
            // Output projection accumulates onto the residual stream.
            self.kernels.proj(
                &self.attn,
                &blk.out_w,
                Some(&blk.out_b),
                &mut self.x,
                n,
                hidden,
                hidden,
                1.0,
            )?;

            // MLP.
            self.kernels.layer_norm(
                stream,
                &self.x,
                &mut self.normed,
                &blk.ln2_w,
                &blk.ln2_b,
                n,
                hidden,
                cfg.ln_eps,
            )?;
            self.kernels.proj(
                &self.normed,
                &blk.up_w,
                Some(&blk.up_b),
                &mut self.ffn,
                n,
                hidden,
                ffn,
                0.0,
            )?;
            self.kernels.gelu(stream, &mut self.ffn, n * ffn)?;
            self.kernels.proj(
                &self.ffn,
                &blk.down_w,
                Some(&blk.down_b),
                &mut self.x,
                n,
                ffn,
                hidden,
                1.0,
            )?;
        }

        // Post-LN, then the 2×2 merge is a free reinterpretation: four
        // consecutive `hidden`-wide rows are one `4*hidden`-wide row.
        self.kernels.layer_norm(
            stream,
            &self.x,
            &mut self.normed,
            &self.post_ln_w,
            &self.post_ln_b,
            n,
            hidden,
            cfg.ln_eps,
        )?;
        let merge = cfg.merge_factor() as usize;
        let merged_dim = cfg.merger_input_dim() as usize;
        let proj_dim = cfg.projection_dim as usize;
        let tokens = n / merge;
        self.kernels.proj(
            &self.normed,
            &self.fc1_w,
            Some(&self.fc1_b),
            &mut self.merged_mid,
            tokens,
            merged_dim,
            merged_dim,
            0.0,
        )?;
        self.kernels
            .gelu(stream, &mut self.merged_mid, tokens * merged_dim)?;
        self.kernels.proj(
            &self.merged_mid,
            &self.fc2_w,
            Some(&self.fc2_b),
            &mut self.out_f16,
            tokens,
            merged_dim,
            proj_dim,
            0.0,
        )?;
        self.kernels
            .to_f32(stream, &self.out_f16, &mut self.out_f32, tokens * proj_dim)?;
        stream.synchronize()?;
        Ok(&self.out_f32)
    }

    /// [`Self::encode`], read back to the host — the differential tests'
    /// entry point.
    pub fn encode_to_host(
        &mut self,
        image: &PreprocessedImage,
    ) -> Result<Vec<f32>, VisionForwardError> {
        let tokens = image.output_tokens(&self.cfg) as usize;
        let proj = self.cfg.projection_dim as usize;
        self.encode(image)?;
        self.stage_f32.clear();
        self.stage_f32.resize(tokens * proj, 0.0);
        self.stream
            .memcpy_dtoh(&self.out_f32.slice(..tokens * proj), &mut self.stage_f32)?;
        self.stream.synchronize()?;
        Ok(self.stage_f32.clone())
    }
}
