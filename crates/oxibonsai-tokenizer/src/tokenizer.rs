//! High-level OxiBonsai tokenizer: BPE + char-level fallback.
//!
//! [`OxiTokenizer`] ties together a [`Vocabulary`], a [`BpeMerges`] table, and
//! a [`TokenizerConfig`] into a complete encode/decode API that is
//! `no_std`-friendly and WASM-compatible.

use std::collections::HashSet;

use tracing::debug;

use crate::{
    bpe::{bpe_encode, byte_fallback_id, pretokenize, BpeMerges},
    error::{TokenizerError, TokenizerResult},
    vocab::Vocabulary,
};

// ── TokenizerConfig ───────────────────────────────────────────────────────────

/// Configuration knobs for an [`OxiTokenizer`].
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// Whether to prepend a BOS (beginning-of-sequence) token.
    pub add_bos: bool,
    /// Whether to append an EOS (end-of-sequence) token.
    pub add_eos: bool,
    /// Token ID used for BOS.
    pub bos_token_id: u32,
    /// Token ID used for EOS.
    pub eos_token_id: u32,
    /// Token ID used for unknown tokens (fallback).
    pub unk_token_id: u32,
    /// Token ID used for padding.
    pub pad_token_id: u32,
    /// Optional maximum output length (tokens are truncated, not padded).
    pub max_length: Option<usize>,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            add_bos: false,
            add_eos: false,
            bos_token_id: 1,
            eos_token_id: 2,
            unk_token_id: 0,
            pad_token_id: 3,
            max_length: None,
        }
    }
}

// ── OxiTokenizer ─────────────────────────────────────────────────────────────

/// Pure Rust BPE tokenizer compatible with MeCrab and the WASM target.
///
/// The tokenizer supports:
/// - Standard BPE encoding via a merge table
/// - Optional BOS/EOS injection
/// - Byte-fallback for out-of-vocabulary bytes
/// - Character-level stub mode (no vocab file needed — useful in tests)
pub struct OxiTokenizer {
    vocab: Vocabulary,
    merges: BpeMerges,
    config: TokenizerConfig,
    /// The set of special token IDs for quick membership tests.
    special_ids: HashSet<u32>,
}

impl OxiTokenizer {
    /// Construct a tokenizer from pre-built components.
    pub fn new(vocab: Vocabulary, merges: BpeMerges, config: TokenizerConfig) -> Self {
        let special_ids = build_special_ids(&config);
        Self {
            vocab,
            merges,
            config,
            special_ids,
        }
    }

    /// Encode a single text string into a sequence of token IDs.
    ///
    /// Steps:
    /// 1. Pre-tokenize into words.
    /// 2. BPE-encode each word.
    /// 3. Optionally prepend BOS and append EOS.
    /// 4. Optionally truncate to `config.max_length`.
    pub fn encode(&self, text: &str) -> TokenizerResult<Vec<u32>> {
        debug!(text_len = text.len(), "encoding text");

        let mut ids: Vec<u32> = Vec::new();

        if self.config.add_bos {
            ids.push(self.config.bos_token_id);
        }

        let words = pretokenize(text);
        for word in &words {
            let word_ids = bpe_encode(word, &self.vocab, &self.merges);
            if word_ids.is_empty() {
                // Byte-fallback path: encode each UTF-8 byte explicitly.
                for byte in word.as_bytes() {
                    let fallback = byte_fallback_id(*byte);
                    let fallback_id = self.vocab.get_id(&fallback);
                    ids.push(fallback_id.unwrap_or(self.config.unk_token_id));
                }
            } else {
                ids.extend_from_slice(&word_ids);
            }
        }

        if self.config.add_eos {
            ids.push(self.config.eos_token_id);
        }

        // Truncate if configured.
        if let Some(max) = self.config.max_length {
            ids.truncate(max);
        }

        Ok(ids)
    }

    /// Encode a batch of texts in sequence (returns one Vec<u32> per input).
    pub fn encode_batch(&self, texts: &[&str]) -> TokenizerResult<Vec<Vec<u32>>> {
        texts.iter().map(|t| self.encode(t)).collect()
    }

    /// Decode a sequence of token IDs back into a string.
    ///
    /// Special tokens (BOS, EOS, PAD, UNK) are silently skipped.
    /// Byte-fallback tokens (`<0xHH>`) are decoded back to their original byte.
    /// Unknown IDs that are not in the vocabulary produce `\u{FFFD}` (replacement
    /// character) rather than an error, to be maximally robust.
    pub fn decode(&self, ids: &[u32]) -> TokenizerResult<String> {
        let mut bytes: Vec<u8> = Vec::with_capacity(ids.len() * 2);

        for &id in ids {
            // Skip special tokens.
            if self.special_ids.contains(&id) {
                continue;
            }

            let token = match self.vocab.get_token(id) {
                Some(t) => t,
                None => {
                    // Unknown ID → replacement character bytes.
                    bytes.extend_from_slice("\u{FFFD}".as_bytes());
                    continue;
                }
            };

            // Decode byte-fallback tokens: `<0xHH>` → one raw byte.
            if let Some(byte) = parse_byte_fallback(token) {
                bytes.push(byte);
            } else {
                // Strip the GPT-2 space prefix (Ġ = U+0120) and push UTF-8.
                let stripped = token.trim_start_matches('\u{0120}');
                if token.starts_with('\u{0120}') && !bytes.is_empty() {
                    bytes.push(b' ');
                }
                bytes.extend_from_slice(stripped.as_bytes());
            }
        }

        String::from_utf8(bytes).map_err(|e| TokenizerError::DecodeFailed(e.to_string()))
    }

    /// Decode a single token ID to its string representation.
    pub fn decode_token(&self, id: u32) -> TokenizerResult<String> {
        self.vocab
            .get_token(id)
            .map(|s| s.to_owned())
            .ok_or_else(|| TokenizerError::DecodeFailed(format!("unknown token id {id}")))
    }

    /// Return the total vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.size()
    }

    /// Construct a tokenizer from JSON-encoded vocabulary and merge lists.
    ///
    /// `vocab_json`: `{ "token": id, ... }`
    /// `merges_json`: `[["a", "b"], ...]` (ordered from highest to lowest priority)
    pub fn from_json(
        vocab_json: &str,
        merges_json: &str,
        config: TokenizerConfig,
    ) -> TokenizerResult<Self> {
        let vocab = Vocabulary::from_json(vocab_json)?;

        let raw_merges: Vec<(String, String)> = serde_json::from_str(merges_json)
            .map_err(|e| TokenizerError::InvalidJson(e.to_string()))?;

        let mut merges = BpeMerges::new();
        for (a, b) in &raw_merges {
            // The merged token name is the concatenation.
            let merged = format!("{a}{b}");
            let result_id = vocab.get_id(&merged).ok_or_else(|| {
                TokenizerError::InvalidVocab(format!("merged token {merged:?} not in vocabulary"))
            })?;
            merges.add_merge(a, b, result_id);
        }

        Ok(Self::new(vocab, merges, config))
    }

    /// Create a character-level stub tokenizer for testing.
    ///
    /// Assigns IDs 4..vocab_size to printable ASCII characters (space = 4,
    /// '!' = 5, ...) with IDs 0-3 reserved for UNK/BOS/EOS/PAD.
    ///
    /// This tokenizer has no BPE merges: each character is its own token.
    pub fn char_level_stub(vocab_size: usize) -> Self {
        assert!(
            vocab_size >= 4,
            "char_level_stub requires vocab_size >= 4 for special tokens"
        );

        let mut vocab = Vocabulary::new();
        vocab.add_special("<unk>", 0);
        vocab.add_special("<bos>", 1);
        vocab.add_special("<eos>", 2);
        vocab.add_special("<pad>", 3);

        // Fill remaining slots with printable ASCII characters.
        let mut next_id = 4u32;
        for byte in 0x20u8..=0x7Eu8 {
            if next_id as usize >= vocab_size {
                break;
            }
            let ch = char::from(byte);
            vocab.insert(&ch.to_string(), next_id);
            next_id += 1;
        }

        // Also populate byte-fallback tokens for any remaining slots.
        for byte in 0u8..=255u8 {
            if next_id as usize >= vocab_size {
                break;
            }
            let fallback = byte_fallback_id(byte);
            if vocab.get_id(&fallback).is_none() {
                vocab.insert(&fallback, next_id);
                next_id += 1;
            }
        }

        let config = TokenizerConfig {
            add_bos: false,
            add_eos: false,
            bos_token_id: 1,
            eos_token_id: 2,
            unk_token_id: 0,
            pad_token_id: 3,
            max_length: None,
        };

        let merges = BpeMerges::new();
        Self::new(vocab, merges, config)
    }

    // ── Special token helpers ─────────────────────────────────────────────

    /// Return the BOS token ID from the configuration.
    pub fn bos_id(&self) -> u32 {
        self.config.bos_token_id
    }

    /// Return the EOS token ID from the configuration.
    pub fn eos_id(&self) -> u32 {
        self.config.eos_token_id
    }

    /// Return `true` if `id` is one of the configured special token IDs.
    pub fn is_special(&self, id: u32) -> bool {
        self.special_ids.contains(&id)
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Build the set of special token IDs from a config.
fn build_special_ids(config: &TokenizerConfig) -> HashSet<u32> {
    let mut set = HashSet::new();
    set.insert(config.bos_token_id);
    set.insert(config.eos_token_id);
    set.insert(config.unk_token_id);
    set.insert(config.pad_token_id);
    set
}

/// Parse a byte-fallback token like `<0x41>` and return the byte value.
///
/// Returns `None` if the token is not in the `<0xHH>` format.
fn parse_byte_fallback(token: &str) -> Option<u8> {
    let inner = token.strip_prefix("<0x")?.strip_suffix('>')?;
    if inner.len() != 2 {
        return None;
    }
    u8::from_str_radix(inner, 16).ok()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_level_stub_encode_ascii() {
        let tok = OxiTokenizer::char_level_stub(200);
        let ids = tok.encode("ab").expect("encode should succeed");
        // Each char should map to a consistent non-zero ID.
        assert_eq!(ids.len(), 2);
        assert_ne!(ids[0], 0); // not UNK
        assert_ne!(ids[1], 0);
        assert_ne!(ids[0], ids[1]); // 'a' ≠ 'b'
    }

    #[test]
    fn char_level_stub_bos_eos() {
        let mut tok = OxiTokenizer::char_level_stub(200);
        tok.config.add_bos = true;
        tok.config.add_eos = true;
        tok.special_ids = build_special_ids(&tok.config);
        let ids = tok.encode("hi").expect("encode should succeed");
        assert_eq!(ids[0], 1); // BOS
        assert_eq!(*ids.last().expect("must have last element"), 2); // EOS
    }

    #[test]
    fn char_level_stub_vocab_size() {
        let tok = OxiTokenizer::char_level_stub(50);
        assert!(tok.vocab_size() <= 50);
        assert!(tok.vocab_size() >= 4); // at least special tokens
    }

    #[test]
    fn special_token_detection() {
        let tok = OxiTokenizer::char_level_stub(200);
        assert!(tok.is_special(0)); // UNK
        assert!(tok.is_special(1)); // BOS
        assert!(tok.is_special(2)); // EOS
        assert!(tok.is_special(3)); // PAD
        assert!(!tok.is_special(4)); // first real token
    }

    #[test]
    fn bos_eos_ids_match_config() {
        let tok = OxiTokenizer::char_level_stub(200);
        assert_eq!(tok.bos_id(), 1);
        assert_eq!(tok.eos_id(), 2);
    }

    #[test]
    fn decode_token_roundtrip() {
        let tok = OxiTokenizer::char_level_stub(200);
        // 'a' should map to some ID; we can look it up.
        let ids = tok.encode("a").expect("should encode");
        if let Some(&id) = ids.first() {
            let s = tok.decode_token(id).expect("decode_token should succeed");
            assert_eq!(s, "a");
        }
    }

    #[test]
    fn decode_unknown_id_returns_error() {
        let tok = OxiTokenizer::char_level_stub(50);
        let result = tok.decode_token(99_999);
        assert!(result.is_err());
    }

    #[test]
    fn max_length_truncates() {
        let mut tok = OxiTokenizer::char_level_stub(200);
        tok.config.max_length = Some(3);
        tok.special_ids = build_special_ids(&tok.config);
        let ids = tok.encode("hello world").expect("encode should succeed");
        assert!(ids.len() <= 3);
    }

    #[test]
    fn encode_batch_consistency() {
        let tok = OxiTokenizer::char_level_stub(200);
        let texts = ["ab", "cd", "ef"];
        let batch = tok
            .encode_batch(&texts)
            .expect("batch encode should succeed");
        assert_eq!(batch.len(), 3);
        for (i, ids) in batch.iter().enumerate() {
            let single = tok.encode(texts[i]).expect("single encode should succeed");
            assert_eq!(*ids, single);
        }
    }

    #[test]
    fn parse_byte_fallback_valid() {
        assert_eq!(parse_byte_fallback("<0x41>"), Some(0x41));
        assert_eq!(parse_byte_fallback("<0x00>"), Some(0x00));
        assert_eq!(parse_byte_fallback("<0xFF>"), Some(0xFF));
    }

    #[test]
    fn parse_byte_fallback_invalid() {
        assert_eq!(parse_byte_fallback("hello"), None);
        assert_eq!(parse_byte_fallback("<0x>"), None);
        assert_eq!(parse_byte_fallback("<0x1>"), None);
    }

    #[test]
    fn from_json_roundtrip() {
        let vocab_json = r#"{"a":10,"b":11,"ab":20,"<unk>":0,"<bos>":1,"<eos>":2,"<pad>":3}"#;
        let merges_json = r#"[["a","b"]]"#;
        let config = TokenizerConfig::default();
        let tok = OxiTokenizer::from_json(vocab_json, merges_json, config)
            .expect("from_json should succeed");
        assert_eq!(tok.vocab_size(), 7);
        // Encoding "ab" should produce a single merged token 20.
        let ids = tok.encode("ab").expect("encode should succeed");
        assert!(ids.contains(&20));
    }
}
