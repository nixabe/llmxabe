//! Qwen tokenizer construction from the vocabulary embedded in GGUF.

use std::path::Path;

use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};
use xabe_gguf::GgufFile;

const QWEN35_PATTERN: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[derive(Debug)]
pub enum TokenizerError {
    Gguf(xabe_gguf::GgufError),
    Missing(&'static str),
    InvalidMerge(String),
    Build(tokenizers::Error),
}

impl core::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Gguf(error) => write!(f, "could not read GGUF tokenizer: {error}"),
            Self::Missing(key) => write!(f, "GGUF tokenizer metadata `{key}` is missing"),
            Self::InvalidMerge(merge) => write!(f, "invalid tokenizer merge `{merge}`"),
            Self::Build(error) => write!(f, "could not construct tokenizer: {error}"),
        }
    }
}

impl core::error::Error for TokenizerError {}

impl From<xabe_gguf::GgufError> for TokenizerError {
    fn from(value: xabe_gguf::GgufError) -> Self {
        Self::Gguf(value)
    }
}

/// Load the GPT-2 byte-level BPE used by Qwen3.5/Qwen3.6 from a GGUF file.
pub fn from_gguf(path: &Path) -> Result<Tokenizer, TokenizerError> {
    let gguf = GgufFile::open(path)?;
    let tokens = gguf
        .get_string_array("tokenizer.ggml.tokens")
        .ok_or(TokenizerError::Missing("tokenizer.ggml.tokens"))?;
    let merge_strings = gguf
        .get_string_array("tokenizer.ggml.merges")
        .ok_or(TokenizerError::Missing("tokenizer.ggml.merges"))?;

    let vocab: Vocab = tokens
        .iter()
        .enumerate()
        .map(|(id, token)| (token.clone(), id as u32))
        .collect();
    let merges = merge_strings
        .iter()
        .map(|merge| {
            merge
                .split_once(' ')
                .map(|(left, right)| (left.to_owned(), right.to_owned()))
                .ok_or_else(|| TokenizerError::InvalidMerge(merge.clone()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(TokenizerError::Build)?;

    let split = Split::new(
        SplitPattern::Regex(QWEN35_PATTERN.to_owned()),
        SplitDelimiterBehavior::Isolated,
        true,
    )
    .map_err(TokenizerError::Build)?;
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(Sequence::new(vec![
        split.into(),
        ByteLevel::new(false, false, false).into(),
    ])));
    tokenizer.with_decoder(Some(ByteLevelDecoder::new(false, false, false)));

    // Special tokens must remain indivisible when they occur in prompts.
    let special_tokens = tokens
        .iter()
        .enumerate()
        .filter(|(id, _)| *id >= 248_000)
        .map(|(_, token)| AddedToken::from(token.clone(), true))
        .collect::<Vec<_>>();
    tokenizer.add_special_tokens(&special_tokens);
    Ok(tokenizer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const DEFAULT_MODEL_PATH: &str =
        "/home/nixabe/llmxabe/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

    #[test]
    fn real_qwen_tokenizer_matches_llama_cpp_oracle() {
        let path = std::env::var_os("LLMXABE_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH));
        if !path.exists() {
            eprintln!("SKIP: model GGUF is not available at {}", path.display());
            return;
        }
        let tokenizer = from_gguf(&path).expect("tokenizer metadata should be valid");
        let encoding = tokenizer
            .encode("Hello, world!", false)
            .expect("test prompt should encode");
        assert_eq!(encoding.get_ids(), &[9419, 11, 1814, 0]);
        assert_eq!(
            tokenizer
                .decode(encoding.get_ids(), false)
                .expect("test tokens should decode"),
            "Hello, world!"
        );
    }
}
