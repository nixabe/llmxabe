//! Tensor directory entries.

use crate::types::GgmlType;

/// One entry from a GGUF file's tensor directory.
///
/// Everything here except `offset` and `n_bytes` is exactly what the file
/// declared; `offset` is still relative to the (aligned) data section start
/// as stored on disk — [`crate::GgufFile::tensor_bytes`] is what resolves it
/// to an absolute file position.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    /// Tensor name, e.g. `blk.3.attn_q.weight`.
    pub name: String,
    /// Dimensions in GGUF/ggml order: `dims[0]` is the fastest-varying
    /// (innermost / row) dimension, matching `ne[0]` in `ggml_tensor`.
    pub dims: Vec<u64>,
    /// Element type.
    pub ggml_type: GgmlType,
    /// Byte offset of this tensor's data, relative to the start of the
    /// (alignment-padded) data section — not an absolute file offset.
    pub offset: u64,
    /// Total element count (product of `dims`).
    pub n_elements: u64,
    /// Total byte size of this tensor's data, as returned by
    /// [`GgmlType::bytes_for_elements`].
    pub n_bytes: u64,
}
