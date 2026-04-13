//! # oxibonsai-eval
//!
//! Model evaluation harness for OxiBonsai.
//!
//! Provides utilities for:
//!
//! - **Perplexity** — measures how well a model predicts held-out text.
//! - **MMLU-style multiple choice** — accuracy on four-option questions.
//! - **Exact match** — token-level accuracy for text-generation tasks.
//! - **Throughput benchmarking** — tokens-per-second and latency statistics.
//! - **Dataset loading** — JSONL-based [`EvalDataset`] and [`McDataset`].
//! - **Report generation** — JSON and Markdown evaluation reports.
//!
//! ## Quick start
//!
//! ```rust
//! use oxibonsai_eval::perplexity::PerplexityEvaluator;
//!
//! let eval = PerplexityEvaluator::new();
//! // Perfect predictions → PPL ≈ 1.0
//! let ppl = eval.compute(&[0.0f32; 10]);
//! assert!((ppl - 1.0).abs() < 1e-5);
//! ```

pub mod accuracy;
pub mod dataset;
pub mod error;
pub mod perplexity;
pub mod report;
pub mod rouge;
pub mod throughput;

#[cfg(test)]
mod tests;

// ──────────────────────────────────────────────────────────────────────────────
// Public re-exports
// ──────────────────────────────────────────────────────────────────────────────

pub use accuracy::{AccuracyResult, ExactMatchEvaluator, McEvaluator};
pub use dataset::{EvalDataset, EvalExample, McDataset, MultipleChoiceQuestion};
pub use error::EvalError;
pub use perplexity::{PerplexityEvaluator, PerplexityResult};
pub use report::{EvalReport, EvalResultEntry};
pub use rouge::{
    ngram_counts, tokenize, CorpusRouge, RougeLScore, RougeNScore, RougeSScore, TokenSeq,
};
pub use throughput::{percentile, time_fn, ThroughputBenchmark, ThroughputResult};
