//! # oxibonsai-tokenizer
//!
//! Pure Rust BPE tokenizer for OxiBonsai — MeCrab-compatible, WASM-safe.
//!
//! This crate is a **stub implementation** that will eventually replace the
//! HuggingFace `tokenizers` dependency in [`oxibonsai-runtime`].  It provides:
//!
//! - [`OxiTokenizer`] — high-level encode/decode API
//! - [`Vocabulary`] — bidirectional token ↔ ID mapping with special-token support
//! - [`BpeMerges`] — ordered BPE merge table
//! - [`bpe_encode`] / [`pretokenize`] — core BPE primitives
//! - [`byte_fallback_id`] — `<0xHH>` byte-fallback helper
//! - [`TokenizerError`] / [`TokenizerResult`] — error types
//!
//! ## Quick start (char-level stub — no vocab file needed)
//!
//! ```rust
//! use oxibonsai_tokenizer::OxiTokenizer;
//!
//! let tok = OxiTokenizer::char_level_stub(256);
//! let ids = tok.encode("Hello!").expect("encode should succeed");
//! assert!(!ids.is_empty());
//! ```
//!
//! ## Loading from JSON vocab + merges
//!
//! ```rust
//! use oxibonsai_tokenizer::{OxiTokenizer, TokenizerConfig};
//!
//! let vocab_json = r#"{"a":10,"b":11,"ab":20,"<unk>":0,"<bos>":1,"<eos>":2,"<pad>":3}"#;
//! let merges_json = r#"[["a","b"]]"#;
//! let tok = OxiTokenizer::from_json(vocab_json, merges_json, TokenizerConfig::default())
//!     .expect("loading should succeed");
//! assert_eq!(tok.vocab_size(), 7);
//! ```

pub mod bpe;
pub mod error;
pub mod serialization;
pub mod tests;
pub mod tokenizer;
pub mod trainer;
pub mod utils;
pub mod vocab;

// Re-export the most commonly used types at the crate root.
pub use bpe::{bpe_encode, byte_fallback_id, pretokenize, BpeMerges};
pub use error::{TokenizerError, TokenizerResult};
pub use serialization::{
    base64_decode, base64_encode, SerializationError, TokenizerState, FORMAT_MAGIC,
};
pub use tokenizer::{OxiTokenizer, TokenizerConfig};
pub use trainer::{
    BpeTrainer, MergeRule, SymbolPair, TrainedTokenizer, TrainerConfig, TrainerError, TrainingStats,
};
pub use vocab::Vocabulary;
