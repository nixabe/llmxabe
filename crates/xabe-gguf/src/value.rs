//! GGUF metadata value representation.
//!
//! GGUF's key-value store carries 12 scalar types plus one level of array
//! nesting (`gguf_type` in `ggml/include/gguf.h`); arrays of arrays are not
//! part of the format. [`GgufValue`] mirrors that shape directly rather than
//! collapsing everything to a dynamic/JSON-like representation, so a caller
//! asking for `get_u32` on a key that is actually a `string` gets `None`
//! instead of a silent truncating cast.

/// One GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(GgufArray),
}

/// A homogeneous GGUF metadata array.
///
/// GGUF arrays are typed uniformly (one `gguf_type` tag covers every
/// element), so this is a `Vec` per element type rather than
/// `Vec<GgufValue>` — it can't represent a mixed array because the format
/// can't either.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufArray {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
    String(Vec<String>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F64(Vec<f64>),
}
