//! Reranker providers for the mem0 Rust port.
//!
//! Each provider implements [`mem0_core::traits::Reranker`], returning
//! `(original_index, score)` pairs sorted by descending relevance. Construct
//! providers by name via [`factory::build_reranker`].

pub mod cohere;
pub mod config;
pub mod factory;
pub mod llm;
pub mod util;
pub mod zero_entropy;

pub use config::RerankerSettings;
pub use factory::build_reranker;

use mem0_core::error::Mem0Error;

pub(crate) fn http_error(context: &str, err: reqwest::Error) -> Mem0Error {
    Mem0Error::llm(format!("{context}: {err}"))
}
