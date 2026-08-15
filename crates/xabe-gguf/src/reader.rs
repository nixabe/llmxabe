//! A bounds-checked byte cursor for parsing GGUF's flat binary layout.
//!
//! Every read here returns [`GgufError::UnexpectedEof`] instead of
//! panicking or reading past the buffer. That is the entire reason this
//! exists instead of transmuting the mmap directly: a 32 GB model file
//! truncated mid-download must fail with a message, not a segfault deep
//! inside a `&[u8]` slice op.

use crate::error::GgufError;

pub(crate) struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub(crate) fn pos(&self) -> u64 {
        self.pos as u64
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    /// A capacity hint that never exceeds the bytes actually left in the
    /// buffer, so a corrupt `n` (e.g. a truncated file claiming a
    /// billion-element array) cannot drive an oversized allocation before
    /// the per-element bounds check below has a chance to fail.
    pub(crate) fn safe_capacity_hint(&self, n: u64) -> usize {
        (n as u128).min(self.remaining() as u128) as usize
    }

    fn take(&mut self, n: usize, context: &'static str) -> Result<&'a [u8], GgufError> {
        if n > self.remaining() {
            return Err(GgufError::UnexpectedEof {
                context,
                offset: self.pos(),
            });
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    pub(crate) fn bytes4(&mut self, context: &'static str) -> Result<[u8; 4], GgufError> {
        let s = self.take(4, context)?;
        Ok([s[0], s[1], s[2], s[3]])
    }

    pub(crate) fn u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.take(1, "u8")?[0])
    }

    pub(crate) fn i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.take(1, "i8")?[0] as i8)
    }

    pub(crate) fn u16(&mut self) -> Result<u16, GgufError> {
        Ok(u16::from_le_bytes(self.take(2, "u16")?.try_into().unwrap()))
    }

    pub(crate) fn i16(&mut self) -> Result<i16, GgufError> {
        Ok(i16::from_le_bytes(self.take(2, "i16")?.try_into().unwrap()))
    }

    pub(crate) fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.take(4, "u32")?.try_into().unwrap()))
    }

    pub(crate) fn i32(&mut self) -> Result<i32, GgufError> {
        Ok(i32::from_le_bytes(self.take(4, "i32")?.try_into().unwrap()))
    }

    pub(crate) fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.take(8, "u64")?.try_into().unwrap()))
    }

    pub(crate) fn i64(&mut self) -> Result<i64, GgufError> {
        Ok(i64::from_le_bytes(self.take(8, "i64")?.try_into().unwrap()))
    }

    pub(crate) fn f32(&mut self) -> Result<f32, GgufError> {
        Ok(f32::from_le_bytes(self.take(4, "f32")?.try_into().unwrap()))
    }

    pub(crate) fn f64(&mut self) -> Result<f64, GgufError> {
        Ok(f64::from_le_bytes(self.take(8, "f64")?.try_into().unwrap()))
    }

    /// GGUF stores `bool` as a single `int8_t` (0 = false); see the format
    /// comment in `ggml/include/gguf.h`.
    pub(crate) fn bool_(&mut self) -> Result<bool, GgufError> {
        Ok(self.i8()? != 0)
    }

    /// GGUF strings are a `u64` byte length followed by raw (non-nul-terminated)
    /// UTF-8 bytes.
    pub(crate) fn string(&mut self) -> Result<String, GgufError> {
        let len = self.u64()?;
        let len = usize::try_from(len).map_err(|_| GgufError::UnexpectedEof {
            context: "string length",
            offset: self.pos(),
        })?;
        let bytes = self.take(len, "string bytes")?;
        Ok(String::from_utf8(bytes.to_vec())?)
    }
}
