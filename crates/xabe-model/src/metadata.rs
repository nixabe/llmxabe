//! Strict metadata access shared by auxiliary model loaders.
use crate::ConfigLoadError;
use xabe_gguf::{GgufArray, GgufFile, GgufValue};

pub(crate) struct Metadata<'a>(pub &'a GgufFile);
impl Metadata<'_> {
    pub fn error(key: &str, why: &str) -> ConfigLoadError {
        ConfigLoadError(format!("GGUF `{key}`: {why}"))
    }
    pub fn text(&self, key: &str, expected: &str) -> Result<(), ConfigLoadError> {
        if self.0.get_str(key) == Some(expected) {
            Ok(())
        } else {
            Err(Self::error(key, &format!("expected {expected}")))
        }
    }
    pub fn uint(&self, key: &str) -> Result<u32, ConfigLoadError> {
        match self.0.get(key) {
            Some(GgufValue::U32(v)) => Ok(*v),
            Some(GgufValue::U64(v)) => {
                u32::try_from(*v).map_err(|_| Self::error(key, "exceeds u32"))
            }
            _ => Err(Self::error(
                key,
                "required unsigned integer missing or malformed",
            )),
        }
    }
    pub fn positive(&self, key: &str) -> Result<u32, ConfigLoadError> {
        let n = self.uint(key)?;
        if n == 0 {
            Err(Self::error(key, "must be positive"))
        } else {
            Ok(n)
        }
    }
    pub fn float(&self, key: &str) -> Result<f32, ConfigLoadError> {
        let n = self
            .0
            .get_f32(key)
            .ok_or_else(|| Self::error(key, "required f32 missing or malformed"))?;
        if !n.is_finite() || n <= 0.0 {
            Err(Self::error(key, "must be finite and positive"))
        } else {
            Ok(n)
        }
    }
    pub fn rgb(&self, key: &str, positive: bool) -> Result<[f32; 3], ConfigLoadError> {
        match self.0.get(key) {
            Some(GgufValue::Array(GgufArray::F32(v)))
                if v.len() == 3 && v.iter().all(|v| v.is_finite() && (!positive || *v > 0.0)) =>
            {
                Ok([v[0], v[1], v[2]])
            }
            _ => Err(Self::error(
                key,
                "expected three finite channel values (positive for std)",
            )),
        }
    }
}
