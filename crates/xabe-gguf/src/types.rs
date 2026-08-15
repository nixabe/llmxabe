//! `ggml_type` block layout for the tensor types this crate loads.
//!
//! Block size and byte size per block are transcribed from the
//! `type_traits` table in `ggml/src/ggml.c` (upstream llama.cpp,
//! `/home/nixabe/llama.cpp`), cross-checked against the `static_assert`s next
//! to each `block_*` struct definition in `ggml/src/ggml-common.h`:
//!
//! - `block_q4_0`  = `sizeof(ggml_half) + QK4_0/2`               = 2 + 16  = 18  bytes / 32  elements
//! - `block_q8_0`  = `sizeof(ggml_half) + QK8_0`                 = 2 + 32  = 34  bytes / 32  elements
//! - `block_q4_K`  = `2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2`        = 4 + 12 + 128 = 144 bytes / 256 elements
//! - `block_q5_K`  = `2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2 + QK_K/8` = 4 + 12 + 128 + 32 = 176 bytes / 256 elements
//! - `block_q6_K`  = `sizeof(ggml_half) + QK_K/16 + 3*QK_K/4`             = 2 + 16 + 192 = 210 bytes / 256 elements
//!
//! `ggml_half` is `uint16_t` (`ggml-common.h`), `K_SCALE_SIZE` is `12`, and
//! `QK_K` is `256` — all `#define`s in the same header. `ggml_type` numeric
//! ids come from `ggml/include/ggml.h`'s `enum ggml_type`.
//!
//! This is deliberately not the full ggml type table. A histogram taken with
//! `gguf-py` against the real target file
//! (`Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf`) shows exactly four types in use —
//! F32, Q8_0, Q6_K, BF16 — all covered here. Q4_0, Q4_K, Q5_K, F16 are
//! included because the model spec calls for them as a minimum support set;
//! anything else fails to load with [`crate::GgufError::UnsupportedGgmlType`]
//! rather than being silently mis-sized.

use crate::error::GgufError;

/// A ggml tensor element type this crate knows how to size and dequantize
/// offsets for.
///
/// The discriminant values match `enum ggml_type` in `ggml/include/ggml.h`
/// exactly, so `GgmlType as i32` and [`GgmlType::try_from`] round-trip
/// against the raw id stored in a GGUF tensor info record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum GgmlType {
    /// 32-bit float. Block size 1.
    F32 = 0,
    /// IEEE-754 half precision. Block size 1.
    F16 = 1,
    /// 4-bit, block-quantized with one fp16 scale per 32-element block.
    Q4_0 = 2,
    /// 8-bit, block-quantized with one fp16 scale per 32-element block.
    Q8_0 = 8,
    /// "K-quant" 4-bit: per-256-element superblock with 12 bytes of packed
    /// 6-bit sub-block scales/mins plus two fp16 superblock scales.
    Q4K = 12,
    /// "K-quant" 5-bit: as `Q4_K` plus one extra high bit per element,
    /// packed separately (the trailing `QK_K/8` bytes).
    Q5K = 13,
    /// "K-quant" 6-bit: per-256-element superblock, no separate min (6-bit
    /// values are signed), one fp16 superblock scale.
    Q6K = 14,
    /// Brain float 16: truncated fp32 mantissa. Block size 1.
    Bf16 = 30,
}

impl GgmlType {
    /// Elements per quantization block. `1` for unquantized types.
    pub const fn block_size(self) -> u64 {
        match self {
            Self::F32 | Self::F16 | Self::Bf16 => 1,
            Self::Q4_0 | Self::Q8_0 => 32,
            Self::Q4K | Self::Q5K | Self::Q6K => 256,
        }
    }

    /// Bytes occupied by one full block (scales included, for quantized
    /// types).
    pub const fn type_size(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::Q4_0 => 18,
            Self::Q8_0 => 34,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
        }
    }

    /// Short name matching ggml's own `type_name` field, for display.
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Q4_0 => "q4_0",
            Self::Q8_0 => "q8_0",
            Self::Q4K => "q4_K",
            Self::Q5K => "q5_K",
            Self::Q6K => "q6_K",
            Self::Bf16 => "bf16",
        }
    }

    /// Bytes needed to store `n` elements of this type.
    ///
    /// Rejects element counts that are not a whole number of blocks —
    /// ggml itself refuses to define a row size in that case (see the
    /// `blck_size == 0 || ne[0] % blck_size != 0` check in
    /// `gguf_init_from_reader`, `ggml/src/gguf.cpp`).
    pub fn bytes_for_elements(self, n: u64) -> Result<u64, GgufError> {
        let block = self.block_size();
        if !n.is_multiple_of(block) {
            return Err(GgufError::UnalignedElementCount {
                name: String::new(),
                ty: self,
                elements: n,
                block_size: block,
            });
        }
        Ok((n / block) * self.type_size())
    }
}

impl TryFrom<i32> for GgmlType {
    type Error = GgufError;

    fn try_from(raw: i32) -> Result<Self, Self::Error> {
        match raw {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            8 => Ok(Self::Q8_0),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            30 => Ok(Self::Bf16),
            other => Err(GgufError::UnsupportedGgmlType(other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_sizes_match_ggml_type_traits() {
        assert_eq!(GgmlType::F32.block_size(), 1);
        assert_eq!(GgmlType::F32.type_size(), 4);
        assert_eq!(GgmlType::F16.block_size(), 1);
        assert_eq!(GgmlType::F16.type_size(), 2);
        assert_eq!(GgmlType::Bf16.block_size(), 1);
        assert_eq!(GgmlType::Bf16.type_size(), 2);
        assert_eq!(GgmlType::Q4_0.block_size(), 32);
        assert_eq!(GgmlType::Q4_0.type_size(), 18);
        assert_eq!(GgmlType::Q8_0.block_size(), 32);
        assert_eq!(GgmlType::Q8_0.type_size(), 34);
        assert_eq!(GgmlType::Q4K.block_size(), 256);
        assert_eq!(GgmlType::Q4K.type_size(), 144);
        assert_eq!(GgmlType::Q5K.block_size(), 256);
        assert_eq!(GgmlType::Q5K.type_size(), 176);
        assert_eq!(GgmlType::Q6K.block_size(), 256);
        assert_eq!(GgmlType::Q6K.type_size(), 210);
    }

    #[test]
    fn bytes_for_elements_rejects_partial_blocks() {
        assert!(GgmlType::Q6K.bytes_for_elements(255).is_err());
        assert!(GgmlType::Q6K.bytes_for_elements(256).is_ok());
        assert_eq!(GgmlType::Q6K.bytes_for_elements(512).unwrap(), 420);
    }

    #[test]
    fn try_from_round_trips_ggml_type_ids() {
        for (id, ty) in [
            (0, GgmlType::F32),
            (1, GgmlType::F16),
            (2, GgmlType::Q4_0),
            (8, GgmlType::Q8_0),
            (12, GgmlType::Q4K),
            (13, GgmlType::Q5K),
            (14, GgmlType::Q6K),
            (30, GgmlType::Bf16),
        ] {
            assert_eq!(GgmlType::try_from(id).unwrap(), ty);
        }
        assert!(GgmlType::try_from(99).is_err());
    }
}
