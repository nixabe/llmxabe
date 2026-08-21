//! GGUF v3 container: header, metadata key-value store, and tensor
//! directory, backed by a memory map.
//!
//! Layout transcribed from the format comment at the top of
//! `ggml/src/gguf.cpp` (upstream llama.cpp, `/home/nixabe/llama.cpp`) and
//! cross-checked against the field-by-field reader in
//! `gguf_init_from_reader` in that same file:
//!
//! 1. Magic `"GGUF"` (4 bytes, no nul).
//! 2. Version (`u32`). Only `3` is accepted here — v1 is long dead upstream
//!    and there is no v2 in the wild for this model family.
//! 3. Tensor count, KV count (`i64` each — GGUF stores counts signed even
//!    though they can never legitimately be negative; a negative value here
//!    is itself a corruption signal, so it is rejected rather than cast).
//! 4. `n_kv` key-value pairs: `string` key, `i32` type tag, then either one
//!    value of that type or (if the tag is `ARRAY`) an element type tag, a
//!    `u64` count, then that many elements.
//! 5. `n_tensors` tensor info records: `string` name, `u32` dimension count
//!    (≤ 4), that many `i64` dimensions, `i32` ggml type, `u64` offset
//!    relative to the data section.
//! 6. The data section, starting at the metadata end padded up to
//!    `general.alignment` (a `u32` KV, default 32 — `GGUF_DEFAULT_ALIGNMENT`
//!    in `gguf.h`). Each tensor's absolute file offset is
//!    `data_section_start + info.offset`.

use std::collections::HashMap;
use std::path::Path;

use memmap2::Mmap;

use crate::error::GgufError;
use crate::reader::Cursor;
use crate::tensor::TensorInfo;
use crate::types::GgmlType;
use crate::value::{GgufArray, GgufValue};

/// Default alignment when `general.alignment` is absent, per
/// `GGUF_DEFAULT_ALIGNMENT` in `ggml/include/gguf.h`.
const DEFAULT_ALIGNMENT: u32 = 32;

const GGUF_VERSION: u32 = 3;

/// The byte source backing a parsed file: a memory map for real files, or an
/// owned buffer for in-memory round-trip tests. Parsing is identical either
/// way — only how the bytes got there differs.
enum Backing {
    Mmap(Mmap),
    Owned(Vec<u8>),
}

impl std::ops::Deref for Backing {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Self::Mmap(m) => m,
            Self::Owned(v) => v,
        }
    }
}

/// A parsed GGUF file: header, full metadata store, tensor directory, and
/// zero-copy access to each tensor's raw bytes.
///
/// Holds no owned copy of tensor data — [`Self::open`] memory-maps the file,
/// so loading a 32 GB checkpoint touches only the pages the OS needs for the
/// metadata scan plus whatever tensors are later read.
pub struct GgufFile {
    backing: Backing,
    version: u32,
    alignment: u32,
    metadata: HashMap<String, GgufValue>,
    tensors: Vec<TensorInfo>,
    tensor_index: HashMap<String, usize>,
    /// Absolute file offset where the tensor data section begins (metadata
    /// end, padded up to `alignment`).
    data_offset: u64,
}

impl GgufFile {
    /// Memory-map and parse the GGUF file at `path`.
    ///
    /// # Safety of the underlying mmap
    ///
    /// `memmap2::Mmap::map` is unsafe because the OS cannot guarantee the
    /// file isn't truncated or mutated by another process while mapped; a
    /// concurrent truncation can raise `SIGBUS` on access. This is the
    /// standard tradeoff for loading multi-gigabyte model weights without
    /// copying them into process memory first, and matches how llama.cpp
    /// itself maps GGUF files.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_backing(Backing::Mmap(mmap))
    }

    /// Parse a GGUF file already fully in memory.
    ///
    /// Used by synthetic round-trip tests, which build a small buffer by
    /// hand rather than writing and mapping a temp file.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, GgufError> {
        Self::from_backing(Backing::Owned(bytes))
    }

    fn from_backing(backing: Backing) -> Result<Self, GgufError> {
        let parsed = parse(&backing)?;

        let file_len = backing.len() as u64;
        for t in &parsed.tensors {
            let start = parsed
                .data_offset
                .checked_add(t.offset)
                .ok_or_else(|| GgufError::OffsetOverflow(t.name.clone()))?;
            let end = start
                .checked_add(t.n_bytes)
                .ok_or_else(|| GgufError::OffsetOverflow(t.name.clone()))?;
            if end > file_len {
                return Err(GgufError::TensorOutOfBounds {
                    name: t.name.clone(),
                    start,
                    end,
                    file_len,
                });
            }
        }

        let tensor_index = parsed
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();

        Ok(Self {
            backing,
            version: parsed.version,
            alignment: parsed.alignment,
            metadata: parsed.metadata,
            tensors: parsed.tensors,
            tensor_index,
            data_offset: parsed.data_offset,
        })
    }

    /// GGUF format version. Always `3` — see `GGUF_VERSION`.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Data-section alignment in bytes (`general.alignment`, default 32).
    pub fn alignment(&self) -> u32 {
        self.alignment
    }

    /// Number of metadata key-value pairs.
    pub fn n_kv(&self) -> usize {
        self.metadata.len()
    }

    /// Number of tensors in the directory.
    pub fn n_tensors(&self) -> usize {
        self.tensors.len()
    }

    /// All tensor directory entries, in file order.
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// Look up a tensor's directory entry by name.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_index.get(name).map(|&i| &self.tensors[i])
    }

    /// Zero-copy view of a tensor's raw (still-quantized) bytes.
    ///
    /// The returned slice borrows directly from the memory map (or owned
    /// buffer); no data is copied. Bounds were already validated at load
    /// time in `Self::from_backing`, so this indexing cannot panic.
    pub fn tensor_bytes(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensor(name)?;
        let start = (self.data_offset + info.offset) as usize;
        let end = start + info.n_bytes as usize;
        Some(&self.backing[start..end])
    }

    /// Iterate all metadata keys.
    pub fn metadata_keys(&self) -> impl Iterator<Item = &str> {
        self.metadata.keys().map(String::as_str)
    }

    /// Fetch a metadata value without asserting its type.
    ///
    /// The typed accessors below are the ergonomic path when the caller
    /// knows the GGUF key schema. This one exists for callers that must
    /// handle whatever type the file actually carries — model loaders
    /// tolerating `u32`/`u64` drift across converter versions, and
    /// diagnostics that dump keys they have no schema for.
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// Fetch a `u32`-typed metadata value.
    ///
    /// Returns `None` both when the key is absent and when it is present
    /// with a different type — this is a convenience accessor for callers
    /// who know the expected type from the GGUF key schema, not a general
    /// coercion.
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key)? {
            GgufValue::U32(v) => Some(*v),
            _ => None,
        }
    }

    /// Fetch a `u64`-typed metadata value.
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        match self.metadata.get(key)? {
            GgufValue::U64(v) => Some(*v),
            _ => None,
        }
    }

    /// Fetch an `f32`-typed metadata value.
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key)? {
            GgufValue::F32(v) => Some(*v),
            _ => None,
        }
    }

    /// Fetch a `string`-typed metadata value.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key)? {
            GgufValue::String(v) => Some(v.as_str()),
            _ => None,
        }
    }

    /// Fetch a `string[]`-typed metadata value.
    pub fn get_string_array(&self, key: &str) -> Option<&[String]> {
        match self.metadata.get(key)? {
            GgufValue::Array(GgufArray::String(v)) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// Fetch a `bool[]`-typed metadata value.
    pub fn get_bool_array(&self, key: &str) -> Option<&[bool]> {
        match self.metadata.get(key)? {
            GgufValue::Array(GgufArray::Bool(v)) => Some(v.as_slice()),
            _ => None,
        }
    }
}

struct Parsed {
    version: u32,
    alignment: u32,
    metadata: HashMap<String, GgufValue>,
    tensors: Vec<TensorInfo>,
    data_offset: u64,
}

fn parse(bytes: &[u8]) -> Result<Parsed, GgufError> {
    let mut cur = Cursor::new(bytes);

    let magic = cur.bytes4("magic")?;
    if &magic != b"GGUF" {
        return Err(GgufError::BadMagic(magic));
    }

    let version = cur.u32()?;
    if version != GGUF_VERSION {
        return Err(GgufError::UnsupportedVersion(version));
    }

    let n_tensors = cur.i64()?;
    if n_tensors < 0 {
        return Err(GgufError::NegativeCount {
            field: "tensor",
            value: n_tensors,
        });
    }
    let n_kv = cur.i64()?;
    if n_kv < 0 {
        return Err(GgufError::NegativeCount {
            field: "kv",
            value: n_kv,
        });
    }

    let mut metadata = HashMap::with_capacity(cur.safe_capacity_hint(n_kv as u64));
    for _ in 0..n_kv {
        let key = cur.string()?;
        let value = read_value(&mut cur)?;
        if metadata.insert(key.clone(), value).is_some() {
            return Err(GgufError::DuplicateKey(key));
        }
    }

    let alignment = match metadata.get("general.alignment") {
        Some(GgufValue::U32(a)) => *a,
        Some(_) => return Err(GgufError::BadAlignmentType),
        None => DEFAULT_ALIGNMENT,
    };
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(GgufError::BadAlignment(alignment));
    }

    let mut tensors = Vec::with_capacity(cur.safe_capacity_hint(n_tensors as u64));
    let mut seen_names = HashMap::with_capacity(tensors.capacity());
    for _ in 0..n_tensors {
        let name = cur.string()?;
        if seen_names.insert(name.clone(), ()).is_some() {
            return Err(GgufError::DuplicateTensor(name));
        }

        let n_dims = cur.u32()?;
        if n_dims > 4 {
            return Err(GgufError::TooManyDimensions { name, n_dims });
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            let d = cur.i64()?;
            if d <= 0 {
                return Err(GgufError::ZeroDimension { name });
            }
            dims.push(d as u64);
        }

        let raw_type = cur.i32()?;
        let ggml_type = GgmlType::try_from(raw_type)?;

        let offset = cur.u64()?;

        let n_elements: u64 = dims.iter().product();
        let n_bytes = ggml_type.bytes_for_elements(n_elements).map_err(|_| {
            GgufError::UnalignedElementCount {
                name: name.clone(),
                ty: ggml_type,
                elements: n_elements,
                block_size: ggml_type.block_size(),
            }
        })?;

        tensors.push(TensorInfo {
            name,
            dims,
            ggml_type,
            offset,
            n_elements,
            n_bytes,
        });
    }

    let meta_end = cur.pos();
    let data_offset = align_up(meta_end, u64::from(alignment));

    Ok(Parsed {
        version,
        alignment,
        metadata,
        tensors,
        data_offset,
    })
}

fn align_up(value: u64, alignment: u64) -> u64 {
    debug_assert!(alignment.is_power_of_two());
    (value + alignment - 1) & !(alignment - 1)
}

fn read_value(cur: &mut Cursor) -> Result<GgufValue, GgufError> {
    let type_tag = cur.i32()?;
    read_typed_value(cur, type_tag)
}

fn read_typed_value(cur: &mut Cursor, type_tag: i32) -> Result<GgufValue, GgufError> {
    Ok(match type_tag {
        0 => GgufValue::U8(cur.u8()?),
        1 => GgufValue::I8(cur.i8()?),
        2 => GgufValue::U16(cur.u16()?),
        3 => GgufValue::I16(cur.i16()?),
        4 => GgufValue::U32(cur.u32()?),
        5 => GgufValue::I32(cur.i32()?),
        6 => GgufValue::F32(cur.f32()?),
        7 => GgufValue::Bool(cur.bool_()?),
        8 => GgufValue::String(cur.string()?),
        9 => {
            let elem_tag = cur.i32()?;
            let n = cur.u64()?;
            read_array(cur, elem_tag, n)?
        }
        10 => GgufValue::U64(cur.u64()?),
        11 => GgufValue::I64(cur.i64()?),
        12 => GgufValue::F64(cur.f64()?),
        other => return Err(GgufError::UnknownValueType(other)),
    })
}

macro_rules! read_array_of {
    ($cur:expr, $n:expr, $read:ident, $variant:path) => {{
        let cap = $cur.safe_capacity_hint($n);
        let mut v = Vec::with_capacity(cap);
        for _ in 0..$n {
            v.push($cur.$read()?);
        }
        $variant(v)
    }};
}

fn read_array(cur: &mut Cursor, elem_tag: i32, n: u64) -> Result<GgufValue, GgufError> {
    let arr = match elem_tag {
        0 => read_array_of!(cur, n, u8, GgufArray::U8),
        1 => read_array_of!(cur, n, i8, GgufArray::I8),
        2 => read_array_of!(cur, n, u16, GgufArray::U16),
        3 => read_array_of!(cur, n, i16, GgufArray::I16),
        4 => read_array_of!(cur, n, u32, GgufArray::U32),
        5 => read_array_of!(cur, n, i32, GgufArray::I32),
        6 => read_array_of!(cur, n, f32, GgufArray::F32),
        7 => read_array_of!(cur, n, bool_, GgufArray::Bool),
        8 => read_array_of!(cur, n, string, GgufArray::String),
        9 => return Err(GgufError::NestedArray),
        10 => read_array_of!(cur, n, u64, GgufArray::U64),
        11 => read_array_of!(cur, n, i64, GgufArray::I64),
        12 => read_array_of!(cur, n, f64, GgufArray::F64),
        other => return Err(GgufError::UnknownValueType(other)),
    };
    Ok(GgufValue::Array(arr))
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Synthetic GGUF byte-buffer builder -------------------------------
    //
    // Builds a minimal but structurally real GGUF v3 buffer by hand, per the
    // layout documented at the top of this file, so parsing can be
    // round-tripped without needing a file on disk.

    struct Builder {
        buf: Vec<u8>,
        n_kv: u64,
        n_tensors: u64,
        kv: Vec<u8>,
        tensor_info: Vec<u8>,
        tensor_data: Vec<u8>,
        alignment: u32,
    }

    impl Builder {
        fn new() -> Self {
            Self {
                buf: Vec::new(),
                n_kv: 0,
                n_tensors: 0,
                kv: Vec::new(),
                tensor_info: Vec::new(),
                tensor_data: Vec::new(),
                alignment: DEFAULT_ALIGNMENT,
            }
        }

        fn push_string(buf: &mut Vec<u8>, s: &str) {
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }

        fn kv_u32(mut self, key: &str, val: u32) -> Self {
            Self::push_string(&mut self.kv, key);
            self.kv.extend_from_slice(&4i32.to_le_bytes());
            self.kv.extend_from_slice(&val.to_le_bytes());
            self.n_kv += 1;
            self
        }

        fn kv_f32(mut self, key: &str, val: f32) -> Self {
            Self::push_string(&mut self.kv, key);
            self.kv.extend_from_slice(&6i32.to_le_bytes());
            self.kv.extend_from_slice(&val.to_le_bytes());
            self.n_kv += 1;
            self
        }

        fn kv_str(mut self, key: &str, val: &str) -> Self {
            Self::push_string(&mut self.kv, key);
            self.kv.extend_from_slice(&8i32.to_le_bytes());
            Self::push_string(&mut self.kv, val);
            self.n_kv += 1;
            self
        }

        fn kv_string_array(mut self, key: &str, vals: &[&str]) -> Self {
            Self::push_string(&mut self.kv, key);
            self.kv.extend_from_slice(&9i32.to_le_bytes()); // ARRAY
            self.kv.extend_from_slice(&8i32.to_le_bytes()); // elem type: STRING
            self.kv
                .extend_from_slice(&(vals.len() as u64).to_le_bytes());
            for v in vals {
                Self::push_string(&mut self.kv, v);
            }
            self.n_kv += 1;
            self
        }

        fn kv_u32_array(mut self, key: &str, vals: &[u32]) -> Self {
            Self::push_string(&mut self.kv, key);
            self.kv.extend_from_slice(&9i32.to_le_bytes());
            self.kv.extend_from_slice(&4i32.to_le_bytes());
            self.kv
                .extend_from_slice(&(vals.len() as u64).to_le_bytes());
            for v in vals {
                self.kv.extend_from_slice(&v.to_le_bytes());
            }
            self.n_kv += 1;
            self
        }

        /// Add a tensor with `n_elements` of `ty`, filled with zero bytes,
        /// aligned per `self.alignment`.
        fn tensor(mut self, name: &str, dims: &[u64], ty: GgmlType) -> Self {
            Self::push_string(&mut self.tensor_info, name);
            self.tensor_info
                .extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for &d in dims {
                self.tensor_info
                    .extend_from_slice(&(d as i64).to_le_bytes());
            }
            self.tensor_info
                .extend_from_slice(&(ty as i32).to_le_bytes());

            let n_elements: u64 = dims.iter().product();
            // Some negative tests deliberately pass an element count that
            // isn't a whole number of blocks; the parser is what's supposed
            // to reject that; the builder just needs *some* number of bytes
            // to write so the file has a well-formed data section.
            let n_bytes = ty.bytes_for_elements(n_elements).unwrap_or(0);

            // pad tensor_data up to alignment before recording this offset
            let pad = (self.alignment as u64
                - (self.tensor_data.len() as u64 % self.alignment as u64))
                % self.alignment as u64;
            self.tensor_data
                .extend(std::iter::repeat_n(0u8, pad as usize));

            let offset = self.tensor_data.len() as u64;
            self.tensor_info.extend_from_slice(&offset.to_le_bytes());
            self.tensor_data
                .extend(std::iter::repeat_n(0xABu8, n_bytes as usize));

            self.n_tensors += 1;
            self
        }

        fn build(mut self) -> Vec<u8> {
            self.buf.extend_from_slice(b"GGUF");
            self.buf.extend_from_slice(&GGUF_VERSION.to_le_bytes());
            self.buf
                .extend_from_slice(&(self.n_tensors as i64).to_le_bytes());
            self.buf
                .extend_from_slice(&(self.n_kv as i64).to_le_bytes());
            self.buf.extend_from_slice(&self.kv);
            self.buf.extend_from_slice(&self.tensor_info);

            let meta_end = self.buf.len() as u64;
            let padded = align_up(meta_end, self.alignment as u64);
            self.buf
                .extend(std::iter::repeat_n(0u8, (padded - meta_end) as usize));
            self.buf.extend_from_slice(&self.tensor_data);
            self.buf
        }
    }

    #[test]
    fn round_trips_header_metadata_and_tensors() {
        let bytes = Builder::new()
            .kv_u32("general.alignment", 32)
            .kv_str("general.architecture", "qwen35moe")
            .kv_u32("qwen35moe.block_count", 40)
            .kv_f32("qwen35moe.attention.layer_norm_rms_epsilon", 1e-6)
            .kv_string_array("tokenizer.ggml.tokens", &["a", "b", "c"])
            .kv_u32_array("some.array", &[1, 2, 3, 4])
            .tensor("blk.0.attn_norm.weight", &[2048], GgmlType::F32)
            .tensor(
                "blk.0.ffn_gate_exps.weight",
                &[2048, 512, 256],
                GgmlType::Q6K,
            )
            .build();

        let f = GgufFile::from_bytes(bytes).expect("parses");
        assert_eq!(f.version(), 3);
        assert_eq!(f.alignment(), 32);
        assert_eq!(f.n_kv(), 6);
        assert_eq!(f.n_tensors(), 2);

        assert_eq!(f.get_u32("general.alignment"), Some(32));
        assert_eq!(f.get_str("general.architecture"), Some("qwen35moe"));
        assert_eq!(f.get_u32("qwen35moe.block_count"), Some(40));
        assert!(
            (f.get_f32("qwen35moe.attention.layer_norm_rms_epsilon")
                .unwrap()
                - 1e-6)
                .abs()
                < 1e-12
        );
        assert_eq!(
            f.get_string_array("tokenizer.ggml.tokens"),
            Some(["a".to_string(), "b".to_string(), "c".to_string()].as_slice())
        );
        // type mismatch -> None, not a panic or a coerced value
        assert_eq!(f.get_u64("general.alignment"), None);
        assert_eq!(f.get_str("nonexistent.key"), None);

        let t0 = f.tensor("blk.0.attn_norm.weight").unwrap();
        assert_eq!(t0.dims, vec![2048]);
        assert_eq!(t0.ggml_type, GgmlType::F32);
        assert_eq!(t0.n_bytes, 2048 * 4);

        let t1 = f.tensor("blk.0.ffn_gate_exps.weight").unwrap();
        assert_eq!(t1.n_elements, 2048 * 512 * 256);
        assert_eq!(
            t1.n_bytes,
            (2048u64 * 512 * 256 / 256) * 210 // Q6_K: 256-elem blocks, 210 B/block
        );

        assert_eq!(
            f.tensor_bytes("blk.0.attn_norm.weight").unwrap().len(),
            t0.n_bytes as usize
        );
        assert_eq!(f.tensor("does.not.exist"), None);
        assert_eq!(f.tensor_bytes("does.not.exist"), None);
    }

    /// `GgufFile` deliberately doesn't derive `Debug` (its `Mmap` field
    /// doesn't either), so tests assert on the error via a match instead of
    /// `Result::unwrap_err`, which requires `T: Debug`.
    fn expect_err(bytes: Vec<u8>) -> GgufError {
        match GgufFile::from_bytes(bytes) {
            Ok(_) => panic!("expected parsing to fail, but it succeeded"),
            Err(e) => e,
        }
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = Builder::new().build();
        bytes[0] = b'X';
        assert!(matches!(expect_err(bytes), GgufError::BadMagic(_)));
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut bytes = Builder::new().build();
        bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
        assert!(matches!(
            expect_err(bytes),
            GgufError::UnsupportedVersion(2)
        ));
    }

    #[test]
    fn rejects_truncated_file() {
        let bytes = Builder::new()
            .kv_u32("general.alignment", 32)
            .tensor("t", &[32], GgmlType::F32)
            .build();
        // cut off partway through the tensor data blob
        let truncated = bytes[..bytes.len() - 16].to_vec();
        assert!(matches!(
            expect_err(truncated),
            GgufError::TensorOutOfBounds { .. }
        ));
    }

    #[test]
    fn rejects_truncated_header() {
        // not even a full header present
        let bytes = b"GGUF".to_vec();
        assert!(matches!(expect_err(bytes), GgufError::UnexpectedEof { .. }));
    }

    #[test]
    fn rejects_tensor_extending_past_eof() {
        // Build a valid file, then truncate the data section far short of
        // what the tensor directory promises, so the recorded offset
        // resolves past EOF.
        let mut bytes = Builder::new().tensor("t", &[32], GgmlType::F32).build();
        let cut = bytes.len() - 4;
        bytes.truncate(cut);
        assert!(matches!(
            expect_err(bytes),
            GgufError::TensorOutOfBounds { .. }
        ));
    }

    #[test]
    fn rejects_unaligned_element_count() {
        // Q6_K has a 256-element block; 100 elements is not a whole block.
        let bytes = Builder::new().tensor("t", &[100], GgmlType::Q6K).build();
        assert!(matches!(
            expect_err(bytes),
            GgufError::UnalignedElementCount { .. }
        ));
    }

    #[test]
    fn rejects_zero_dimension() {
        let bytes = Builder::new()
            .tensor("t", &[2048, 0], GgmlType::F32)
            .build();
        assert!(matches!(expect_err(bytes), GgufError::ZeroDimension { .. }));
    }

    #[test]
    fn rejects_duplicate_tensor_names() {
        let bytes = Builder::new()
            .tensor("t", &[32], GgmlType::F32)
            .tensor("t", &[32], GgmlType::F32)
            .build();
        assert!(matches!(expect_err(bytes), GgufError::DuplicateTensor(_)));
    }

    #[test]
    fn rejects_duplicate_metadata_keys() {
        let bytes = Builder::new().kv_u32("dup", 1).kv_u32("dup", 2).build();
        assert!(matches!(expect_err(bytes), GgufError::DuplicateKey(_)));
    }

    #[test]
    fn rejects_non_power_of_two_alignment() {
        let bytes = Builder::new().kv_u32("general.alignment", 3).build();
        assert!(matches!(expect_err(bytes), GgufError::BadAlignment(3)));
    }

    #[test]
    fn default_alignment_is_32_when_key_absent() {
        let bytes = Builder::new().tensor("t", &[32], GgmlType::F32).build();
        let f = GgufFile::from_bytes(bytes).unwrap();
        assert_eq!(f.alignment(), 32);
    }
}
