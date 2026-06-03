//! Embedding providers for the mem0 Rust port.
//!
//! Each provider implements [`mem0_core::traits::Embedder`] over `reqwest`
//! (rustls-TLS, so the resulting binary links statically and builds cleanly on
//! macOS and Linux). Construct providers by name via [`factory::build_embedder`].

pub mod azure;
pub mod config;
pub mod factory;
pub mod gemini;
pub mod mock;
pub mod ollama;
pub mod openai;

pub use config::EmbedderSettings;
pub use factory::build_embedder;

use mem0_core::error::Mem0Error;

/// Map a transport-level reqwest error into a `Mem0Error::Embedding`.
pub(crate) fn http_error(context: &str, err: reqwest::Error) -> Mem0Error {
    Mem0Error::embedding(format!("{context}: {err}"))
}
