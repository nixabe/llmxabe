//! GGUF v3 container parsing: header, metadata key-value store, tensor
//! directory, and zero-copy tensor byte access over a memory-mapped file.
//!
//! This crate is the only place that understands the GGUF *byte layout*.
//! Everything downstream — [`xabe_model`](../xabe_model/index.html)'s
//! structural description of the model, and eventually the CUDA loader —
//! reads tensors and metadata through [`GgufFile`] rather than re-parsing
//! bytes, so a layout bug shows up in exactly one place.
//!
//! The format itself is documented in the header comment of
//! `ggml/src/gguf.cpp` in upstream llama.cpp
//! (`/home/nixabe/llama.cpp`); [`file`] transcribes it field by field.
//!
//! Start at [`GgufFile::open`].

mod error;
mod file;
mod reader;
mod tensor;
mod types;
mod value;

pub use error::GgufError;
pub use file::GgufFile;
pub use tensor::TensorInfo;
pub use types::GgmlType;
pub use value::{GgufArray, GgufValue};
