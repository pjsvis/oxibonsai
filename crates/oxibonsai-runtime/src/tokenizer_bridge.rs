//! Tokenizer bridge wrapping HuggingFace tokenizers.
//!
//! On WASM targets, the `tokenizers` crate is unavailable (requires native C extensions).
//! A stub implementation is provided that returns errors for all operations.

use crate::error::{RuntimeError, RuntimeResult};

/// Thin wrapper around `tokenizers::Tokenizer`.
///
/// On non-WASM targets, delegates to the full HuggingFace tokenizers library.
/// On WASM targets, all methods return a `RuntimeError::Tokenizer` error.
pub struct TokenizerBridge {
    #[cfg(not(target_arch = "wasm32"))]
    inner: tokenizers::Tokenizer,
    #[cfg(target_arch = "wasm32")]
    _phantom: (),
}

impl TokenizerBridge {
    /// Load a tokenizer from a JSON file.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_file(path: &str) -> RuntimeResult<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| RuntimeError::Tokenizer(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Load a tokenizer from a JSON file.
    ///
    /// On WASM targets, always returns an error since the tokenizers library
    /// requires native code not available in WebAssembly.
    #[cfg(target_arch = "wasm32")]
    pub fn from_file(_path: &str) -> RuntimeResult<Self> {
        Err(RuntimeError::Tokenizer(
            "tokenizers library is not available on wasm32 targets".to_string(),
        ))
    }

    /// Encode text to token IDs.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn encode(&self, text: &str) -> RuntimeResult<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| RuntimeError::Tokenizer(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Encode text to token IDs.
    ///
    /// On WASM targets, always returns an error.
    #[cfg(target_arch = "wasm32")]
    pub fn encode(&self, _text: &str) -> RuntimeResult<Vec<u32>> {
        Err(RuntimeError::Tokenizer(
            "tokenizers library is not available on wasm32 targets".to_string(),
        ))
    }

    /// Decode token IDs to text.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn decode(&self, ids: &[u32]) -> RuntimeResult<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| RuntimeError::Tokenizer(e.to_string()))
    }

    /// Decode token IDs to text.
    ///
    /// On WASM targets, always returns an error.
    #[cfg(target_arch = "wasm32")]
    pub fn decode(&self, _ids: &[u32]) -> RuntimeResult<String> {
        Err(RuntimeError::Tokenizer(
            "tokenizers library is not available on wasm32 targets".to_string(),
        ))
    }

    /// Get the vocabulary size.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Get the vocabulary size.
    ///
    /// On WASM targets, returns 0 since no tokenizer is available.
    #[cfg(target_arch = "wasm32")]
    pub fn vocab_size(&self) -> usize {
        0
    }

    /// Get the internal tokenizer reference.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }
}
