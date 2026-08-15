//! Typed failure modes for GGUF parsing.
//!
//! GGUF files are untrusted input: they come from the filesystem, are often
//! tens of gigabytes, and a truncated download or a hand-edited quant recipe
//! is a realistic way to end up with a malformed one. Every failure here is a
//! `Result`, never a panic — a 32 GB mmap with a bad tensor offset must fail
//! with a message, not a segfault three layers away in a dequant kernel.

/// A malformed or inconsistent GGUF file.
#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    /// The first four bytes were not `GGUF`.
    #[error("bad magic bytes: expected `GGUF`, found {0:02x?}")]
    BadMagic([u8; 4]),

    /// Only GGUF v3 is supported; see `ggml/include/gguf.h` (`GGUF_VERSION`).
    #[error("unsupported GGUF version {0}; only version 3 is supported")]
    UnsupportedVersion(u32),

    /// The file ended before a value that the header or a preceding field
    /// said should be present.
    #[error("unexpected end of file while reading {context} at byte offset {offset}")]
    UnexpectedEof { context: &'static str, offset: u64 },

    /// A string's declared byte length is not valid UTF-8.
    #[error("metadata string is not valid UTF-8: {0}")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),

    /// The header declared a negative tensor or KV count.
    ///
    /// GGUF stores these as `int64_t` (see `gguf.cpp`); a negative count can
    /// only come from a corrupt or adversarial file.
    #[error("{field} count {value} is negative")]
    NegativeCount { field: &'static str, value: i64 },

    /// Two metadata keys with the same name.
    #[error("duplicate metadata key `{0}`")]
    DuplicateKey(String),

    /// A GGUF value type tag outside the 0..=12 range defined by `gguf_type`.
    #[error("unknown GGUF value type tag {0}")]
    UnknownValueType(i32),

    /// An array of arrays; not representable, and not produced by any writer.
    #[error("nested arrays are not supported (key had an array-of-array value)")]
    NestedArray,

    /// `general.alignment` was present but not a `u32`.
    #[error("`general.alignment` must be of type u32")]
    BadAlignmentType,

    /// `general.alignment` was present but not a power of two.
    #[error("alignment {0} is not a power of two")]
    BadAlignment(u32),

    /// A tensor declared more than `GGML_MAX_DIMS` (4) dimensions.
    #[error("tensor `{name}` has {n_dims} dimensions, more than the maximum of 4")]
    TooManyDimensions { name: String, n_dims: u32 },

    /// A tensor dimension was zero.
    #[error("tensor `{name}` has a zero-sized dimension")]
    ZeroDimension { name: String },

    /// Two tensors with the same name.
    #[error("duplicate tensor name `{0}`")]
    DuplicateTensor(String),

    /// A `ggml_type` id outside the set this crate knows how to size.
    ///
    /// This crate supports the types actually used by the target model (see
    /// `GgmlType`), not the full ggml type table. A file using an
    /// unsupported quantization fails to load with this error rather than
    /// guessing at a block layout.
    #[error("unsupported ggml tensor type id {0}")]
    UnsupportedGgmlType(i32),

    /// A tensor's element count is not a whole number of quantization
    /// blocks, so its byte size cannot be computed.
    #[error(
        "tensor `{name}` has {elements} elements of type {ty:?}, \
         not a whole number of {block_size}-element blocks"
    )]
    UnalignedElementCount {
        name: String,
        ty: crate::types::GgmlType,
        elements: u64,
        block_size: u64,
    },

    /// A tensor's offset or size arithmetic overflowed `u64`.
    #[error("tensor `{0}` offset/size arithmetic overflowed")]
    OffsetOverflow(String),

    /// A tensor's byte range extends past the end of the file.
    #[error(
        "tensor `{name}` byte range [{start}, {end}) exceeds the file size of {file_len} bytes"
    )]
    TensorOutOfBounds {
        name: String,
        start: u64,
        end: u64,
        file_len: u64,
    },

    /// Underlying I/O failure opening or mapping the file.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
