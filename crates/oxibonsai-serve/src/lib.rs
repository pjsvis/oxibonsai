//! Library interface for oxibonsai-serve.
//!
//! Exposes the argument-parsing and banner modules so they can be exercised
//! from integration tests without going through `main`.

pub mod args;
pub mod banner;

pub use args::{ParseError, ServerArgs};
