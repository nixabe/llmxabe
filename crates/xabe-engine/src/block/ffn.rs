//! Which feed-forward block a layer runs, chosen once from the model config.
//!
//! [`crate::forward::Forward`] holds one of these per pass and one
//! [`FfnLayerWeights`] per block, and calls [`FfnBlock::forward`] in exactly
//! the places it used to call `MoeBlock::forward`. The dispatch is a match on
//! a two-variant enum rather than a trait object because there are two
//! architectures, both known at build time, and the alternative — a `dyn`
//! call per layer per step — would put a vtable indirection inside the one
//! loop that runs 40 or 64 times a token.
//!
//! Pairing is checked, not assumed: handing a [`MoeBlock`] a dense layer's
//! weights is a mismatch this returns an error for rather than a shape check
//! failing three launches later with a message about element counts.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use xabe_cuda::kernels::moe::MoeGeometry;
use xabe_gguf::GgufFile;
use xabe_model::config::{FfnConfig, ModelConfig};
use xabe_model::weights::Directory;

use crate::block::dense_ffn::{DENSE_REPACK_INT8, DenseFfnBlock, DenseFfnLayerWeights};
use crate::block::moe::{MoeBlock, MoeBlockError, MoeLayerWeights};

/// The feed-forward block for one pass shape.
pub enum FfnBlock {
    /// `qwen35moe`: 256 routed experts plus a gated shared expert.
    Moe(MoeBlock),
    /// `qwen35`: one dense SwiGLU MLP.
    Dense(DenseFfnBlock),
}

/// One layer's feed-forward weights, resident on the device.
///
/// The variants differ by ~460 bytes and clippy would rather the larger were
/// boxed. It stays unboxed: a model is one architecture, so every entry of
/// the `Vec<FfnLayerWeights>` is the same variant and the padding is only
/// ever paid on the variant that is not there — zero, in both real cases.
/// Boxing would buy nothing and add an indirection per layer per pass.
#[allow(clippy::large_enum_variant)]
pub enum FfnLayerWeights {
    /// See [`MoeLayerWeights`].
    Moe(MoeLayerWeights),
    /// See [`DenseFfnLayerWeights`].
    Dense(DenseFfnLayerWeights),
}

impl FfnLayerWeights {
    /// Upload every tensor block `layer`'s feed-forward block needs.
    pub fn upload(
        stream: &Arc<CudaStream>,
        file: &GgufFile,
        directory: &Directory<'_>,
        layer: u32,
        config: &ModelConfig,
        geometry: &MoeGeometry,
    ) -> Result<Self, MoeBlockError> {
        Ok(match config.ffn {
            FfnConfig::Moe(_) => Self::Moe(MoeLayerWeights::upload(
                stream, file, directory, layer, geometry,
            )?),
            FfnConfig::Dense(_) => Self::Dense(DenseFfnLayerWeights::upload(
                stream,
                file,
                directory,
                layer,
                geometry,
                DENSE_REPACK_INT8,
            )?),
        })
    }

    /// Which block these weights came from.
    pub fn layer(&self) -> u32 {
        match self {
            Self::Moe(w) => w.layer(),
            Self::Dense(w) => w.layer(),
        }
    }

    /// Total device bytes held.
    pub fn bytes(&self) -> usize {
        match self {
            Self::Moe(w) => w.bytes(),
            Self::Dense(w) => w.bytes(),
        }
    }
}

impl FfnBlock {
    /// The geometry `config`'s feed-forward block wants for `max_tokens` per
    /// step.
    ///
    /// One function for both architectures so a caller cannot pick the wrong
    /// one: the dense variant reuses [`MoeGeometry`] because it runs on the
    /// same shared-expert kernels, at a different intermediate width.
    pub fn geometry_for(
        config: &ModelConfig,
        block_size: usize,
        max_tokens: usize,
    ) -> Option<MoeGeometry> {
        match config.ffn {
            FfnConfig::Moe(_) => MoeBlock::geometry_for(config, block_size, max_tokens),
            FfnConfig::Dense(_) => DenseFfnBlock::geometry_for(config, block_size, max_tokens),
        }
    }

    /// Compile every kernel the block needs and allocate every buffer, once.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        config: &ModelConfig,
        geometry: MoeGeometry,
        eps: f32,
    ) -> Result<Self, MoeBlockError> {
        Ok(match config.ffn {
            FfnConfig::Moe(_) => Self::Moe(MoeBlock::new(ctx, stream, geometry, eps)?),
            FfnConfig::Dense(_) => Self::Dense(DenseFfnBlock::new(
                ctx,
                stream,
                geometry,
                eps,
                DENSE_REPACK_INT8,
            )?),
        })
    }

    /// Force every feed-forward GEMM back onto its fp32 kernel.
    pub fn disable_tensor_cores(&mut self) {
        match self {
            Self::Moe(b) => b.disable_tensor_cores(),
            Self::Dense(b) => b.disable_tensor_cores(),
        }
    }

    /// Whether the feed-forward GEMMs will take the integer tensor-core path.
    pub fn tensor_cores_enabled(&self) -> bool {
        match self {
            Self::Moe(b) => b.tensor_cores_enabled(),
            Self::Dense(b) => b.tensor_cores_enabled(),
        }
    }

    /// Pin the routed-expert kernels to the flat decode regime. A no-op on
    /// the dense block, which has no routed kernels and so no regime to pin:
    /// its one GEMM path is selected by token width alone.
    pub fn set_exact_decode_regime(&mut self, on: bool) {
        if let Self::Moe(b) = self {
            b.set_exact_decode_regime(on);
        }
    }

    /// The geometry this block was compiled for.
    pub fn geometry(&self) -> MoeGeometry {
        match self {
            Self::Moe(b) => b.geometry(),
            Self::Dense(b) => b.geometry(),
        }
    }

    /// `attn_post_norm-N`: the post-mixer RMSNorm output.
    pub fn normed(&self) -> &CudaSlice<f32> {
        match self {
            Self::Moe(b) => b.normed(),
            Self::Dense(b) => b.normed(),
        }
    }

    /// The routed block, if this is one. `None` on the dense path, which has
    /// no router logits, no dispatch tables and no shared-expert gate to
    /// inspect.
    pub fn as_moe(&self) -> Option<&MoeBlock> {
        match self {
            Self::Moe(b) => Some(b),
            Self::Dense(_) => None,
        }
    }

    /// Publish this pass's token count into the device scalar the kernels
    /// gate on, ahead of the pass.
    pub fn publish_tokens(
        &mut self,
        stream: &Arc<CudaStream>,
        tokens: usize,
    ) -> Result<(), MoeBlockError> {
        match self {
            Self::Moe(b) => b.publish_tokens(stream, tokens),
            Self::Dense(b) => b.publish_tokens(stream, tokens),
        }
    }

    /// Run the whole block over `tokens` tokens.
    ///
    /// See [`MoeBlock::forward`] and [`DenseFfnBlock::forward`]; the contract
    /// is the same in both, down to `ffn_out` being captured before the
    /// residual add and `l_out` after.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        stream: &Arc<CudaStream>,
        w: &FfnLayerWeights,
        residual: &CudaSlice<f32>,
        tokens: usize,
        ffn_out: &mut CudaSlice<f32>,
        l_out: &mut CudaSlice<f32>,
    ) -> Result<(), MoeBlockError> {
        match (self, w) {
            (Self::Moe(b), FfnLayerWeights::Moe(w)) => {
                b.forward(stream, w, residual, tokens, ffn_out, l_out)
            }
            (Self::Dense(b), FfnLayerWeights::Dense(w)) => {
                b.forward(stream, w, residual, tokens, ffn_out, l_out)
            }
            (block, w) => Err(MoeBlockError::FfnKindMismatch {
                block: block.kind_name(),
                weights: w.kind_name(),
            }),
        }
    }

    fn kind_name(&self) -> &'static str {
        match self {
            Self::Moe(_) => "moe",
            Self::Dense(_) => "dense",
        }
    }
}

impl FfnLayerWeights {
    fn kind_name(&self) -> &'static str {
        match self {
            Self::Moe(_) => "moe",
            Self::Dense(_) => "dense",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_architecture_gets_a_geometry_and_they_are_not_the_same_one() {
        let moe = FfnBlock::geometry_for(&ModelConfig::qwen3_6_35b_a3b(), 32, 128).unwrap();
        let dense = FfnBlock::geometry_for(&ModelConfig::qwen3_8_27b(), 32, 128).unwrap();
        assert_eq!(
            (moe.hidden, moe.intermediate, moe.num_experts),
            (2048, 512, 256)
        );
        assert_eq!(
            (dense.hidden, dense.intermediate, dense.num_experts),
            (5120, 17_408, 1)
        );
    }
}
