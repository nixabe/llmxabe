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

/// The model a test falls back to when `LLMXABE_MODEL` is unset.
#[cfg(test)]
pub const DEFAULT_MODEL_PATH: &str =
    "/home/nixabe/llmxabe/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

/// The id at and above which this vocabulary's entries are special tokens
/// carried verbatim rather than byte-level encoded.
const FIRST_SPECIAL: usize = 248_000;

/// The GPT-2 byte-level alphabet, as a char-to-byte map.
///
/// A byte-level BPE writes each of the 256 bytes as one printable character,
/// so a token's stored spelling is not its bytes. Reversing that is what lets
/// a grammar be matched against the vocabulary
/// ([`crate::http::tools`] hands the result to `xabe-grammar`).
fn byte_of_char() -> std::collections::HashMap<char, u8> {
    let mut direct: Vec<u8> = (b'!'..=b'~')
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut points: Vec<u32> = direct.iter().map(|&byte| u32::from(byte)).collect();
    let mut extra = 0u32;
    for byte in 0u8..=255 {
        if !direct.contains(&byte) {
            direct.push(byte);
            points.push(256 + extra);
            extra += 1;
        }
    }
    direct
        .into_iter()
        .zip(points)
        .filter_map(|(byte, point)| char::from_u32(point).map(|ch| (ch, byte)))
        .collect()
}

/// The byte spelling of every token in the vocabulary, in id order, and the
/// ids that end a turn.
///
/// Constrained decoding matches a byte grammar against candidate tokens, so
/// it needs the bytes each token actually contributes to the output — which
/// for a byte-level BPE is not the token's stored spelling. Special tokens
/// are the exception: they are stored, and emitted, verbatim.
pub fn pieces_from_gguf(path: &Path) -> Result<(Vec<Vec<u8>>, Vec<u32>), TokenizerError> {
    let gguf = GgufFile::open(path)?;
    let tokens = gguf
        .get_string_array("tokenizer.ggml.tokens")
        .ok_or(TokenizerError::Missing("tokenizer.ggml.tokens"))?;
    let map = byte_of_char();
    let mut pieces = Vec::with_capacity(tokens.len());
    let mut eog = Vec::new();
    for (id, token) in tokens.iter().enumerate() {
        if id >= FIRST_SPECIAL {
            // `<|im_end|>` ends a turn; `<tool_call>` and `</tool_call>` are
            // markup the grammar itself writes, and reach it as their own
            // bytes like any other piece.
            if token == "<|im_end|>" || token == "<|endoftext|>" {
                eog.push(id as u32);
            }
            pieces.push(token.as_bytes().to_vec());
            continue;
        }
        pieces.push(
            token
                .chars()
                .filter_map(|ch| map.get(&ch).copied())
                .collect(),
        );
    }
    Ok((pieces, eog))
}

/// The model's own `tokenizer.chat_template`, if the file carries one.
///
/// What `--jinja` renders prompts with instead of the hand-written ChatML in
/// `http::chat`. A GGUF without the key cannot serve that mode, which is a
/// startup failure rather than a per-request one.
pub fn chat_template_from_gguf(path: &Path) -> Result<Option<String>, TokenizerError> {
    Ok(GgufFile::open(path)?
        .get_str("tokenizer.chat_template")
        .map(str::to_owned))
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
        .filter(|(id, _)| *id >= FIRST_SPECIAL)
        .map(|(_, token)| AddedToken::from(token.clone(), true))
        .collect::<Vec<_>>();
    tokenizer.add_special_tokens(&special_tokens);
    Ok(tokenizer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tracing::info;

    /// The grammar, against the real vocabulary, on the call that failed.
    ///
    /// OpenCode offers a `read` tool whose only required parameter is
    /// `filePath`; the model wrote `file_path` and the call came back
    /// "Missing key at [filePath]". With the grammar attached that spelling
    /// is not reachable.
    #[test]
    fn the_tool_grammar_masks_the_real_vocabulary() {
        use std::sync::Arc;
        use xabe_grammar::{ToolConstraint, ToolGrammar, ToolSpec, Vocab};

        let path = std::env::var_os("LLMXABE_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH));
        if !path.exists() {
            eprintln!("SKIP: model GGUF is not available at {}", path.display());
            return;
        }
        let tokenizer = from_gguf(&path).expect("tokenizer metadata should be valid");
        let (pieces, eog) = pieces_from_gguf(&path).expect("piece table");
        assert!(
            !eog.is_empty(),
            "the vocabulary must name an end-of-turn token"
        );
        // The piece table must agree with the tokenizer it was read beside,
        // or the mask would be over the wrong bytes.
        for text in ["Hello, world!", "E:/dfans-plugin/manifest.minimal.json"] {
            let ids = tokenizer.encode(text, false).expect("encodes");
            let joined: Vec<u8> = ids
                .get_ids()
                .iter()
                .flat_map(|&id| pieces[id as usize].clone())
                .collect();
            assert_eq!(
                String::from_utf8(joined).expect("byte-level pieces rejoin"),
                text
            );
        }

        let tool = ToolSpec::from_wrapper(&serde_json::json!({
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filePath": { "type": "string" },
                        "limit": { "type": "number" },
                    },
                    "required": ["filePath"],
                },
            },
        }))
        .expect("a named function parses");
        let vocab = Arc::new(Vocab::new(&pieces, &eog));
        let grammar =
            Arc::new(ToolGrammar::new(&[tool], true, Arc::clone(&vocab)).expect("compiles"));

        let mut constraint = ToolConstraint::new(Arc::clone(&grammar));
        let opening = "<tool_call>\n<function=read>\n<parameter=";
        for id in tokenizer.encode(opening, false).expect("encodes").get_ids() {
            constraint.observe(*id as i32);
        }
        assert!(constraint.is_masking(), "the opener arms the constraint");

        let mut logits = vec![0.0f32; pieces.len()];
        let started = std::time::Instant::now();
        constraint.mask(&mut logits);
        let elapsed = started.elapsed();
        let admitted = |text: &str| {
            let ids = tokenizer.encode(text, false).expect("encodes");
            let first = ids.get_ids()[0] as usize;
            logits[first].is_finite()
        };
        assert!(
            admitted("filePath"),
            "the schema's own spelling is reachable"
        );
        assert!(
            !admitted("_path"),
            "the misspelling the model actually wrote is not"
        );
        assert!(!admitted("limits"), "and neither is a name near a real one");
        let live = logits.iter().filter(|logit| logit.is_finite()).count();
        info!(
            tokens = pieces.len(),
            admitted = live,
            micros = elapsed.as_micros() as u64,
            "tool grammar mask over the real vocabulary"
        );
        assert!(
            live < 64,
            "only a parameter name may start here, got {live}"
        );
    }

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
