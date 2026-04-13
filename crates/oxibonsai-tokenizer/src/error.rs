//! Error types for the OxiBonsai tokenizer.

use thiserror::Error;

/// All errors that can occur during tokenization operations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    /// A token string was not found in the vocabulary.
    #[error("unknown token: {0:?}")]
    UnknownToken(String),

    /// The vocabulary data is malformed or inconsistent.
    #[error("invalid vocabulary: {0}")]
    InvalidVocab(String),

    /// Encoding of input text failed.
    #[error("encode failed: {0}")]
    EncodeFailed(String),

    /// Decoding of token IDs failed.
    #[error("decode failed: {0}")]
    DecodeFailed(String),

    /// JSON deserialization failed.
    #[error("invalid JSON: {0}")]
    InvalidJson(String),
}

/// Convenience result alias for tokenizer operations.
pub type TokenizerResult<T> = Result<T, TokenizerError>;
