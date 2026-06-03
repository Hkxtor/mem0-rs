//! LLM providers for the mem0 Rust port.
//!
//! Each provider implements [`mem0_core::traits::Llm`] over `reqwest`
//! (rustls-TLS). Construct providers by name via [`factory::build_llm`].

pub mod anthropic;
pub mod azure;
pub mod config;
pub mod factory;
pub mod gemini;
pub mod mock;
pub mod ollama;
pub mod openai;

pub use config::LlmSettings;
pub use factory::build_llm;

use mem0_core::error::Mem0Error;
use mem0_core::types::Message;
use serde_json::{json, Value};

/// Map a transport-level reqwest error into a `Mem0Error::Llm`.
pub(crate) fn http_error(context: &str, err: reqwest::Error) -> Mem0Error {
    Mem0Error::llm(format!("{context}: {err}"))
}

/// Convert messages into `[{role, content}]` wire form (drops `name` for
/// providers that reject it).
pub(crate) fn to_wire_messages(messages: &[Message]) -> Vec<Value> {
    messages
        .iter()
        .map(|m| json!({ "role": m.role, "content": m.content }))
        .collect()
}
