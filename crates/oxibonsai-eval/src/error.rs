//! Error types for the evaluation harness.

use thiserror::Error;

/// Errors that can occur during model evaluation.
#[derive(Debug, Error)]
pub enum EvalError {
    /// The provided dataset contains no examples.
    #[error("dataset is empty")]
    DatasetEmpty,

    /// Input data has an unexpected or malformed format.
    #[error("invalid format: {0}")]
    InvalidFormat(String),

    /// The model inference step failed.
    #[error("inference failed: {0}")]
    InferenceFailed(String),

    /// An I/O error occurred while reading or writing evaluation data.
    #[error("I/O error: {0}")]
    IoError(String),

    /// Parsing of a value (e.g. JSON field, integer) failed.
    #[error("parse error: {0}")]
    ParseError(String),
}
