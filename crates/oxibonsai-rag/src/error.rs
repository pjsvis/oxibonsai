//! Error types for the OxiBonsai RAG pipeline.

use thiserror::Error;

/// Errors that can occur in the RAG pipeline.
#[derive(Debug, Error)]
pub enum RagError {
    /// Input document was empty (no text to index).
    #[error("document is empty")]
    EmptyDocument,

    /// Query string was empty.
    #[error("query is empty")]
    EmptyQuery,

    /// Retrieval was attempted before any documents were indexed.
    #[error("no documents have been indexed yet")]
    NoDocumentsIndexed,

    /// The embedding backend failed to produce a vector.
    #[error("embedding failed: {0}")]
    EmbeddingFailed(String),

    /// A vector was inserted with a dimensionality that does not match the store's dimension.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// The dimensionality the store was configured with.
        expected: usize,
        /// The dimensionality of the offending vector.
        got: usize,
    },
}
