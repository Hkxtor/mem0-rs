//! Reranker configuration parsed from the `config` object of a
//! `{ provider, config }` reranker spec.

use mem0_core::error::Result;
use serde::Deserialize;
use serde_json::Value;

/// Provider-agnostic reranker settings. Unknown fields are ignored.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RerankerSettings {
    /// Model identifier.
    pub model: Option<String>,
    /// API key.
    pub api_key: Option<String>,
    /// Default number of results to keep.
    pub top_k: Option<usize>,
    /// API base URL override (testability / self-hosted).
    pub base_url: Option<String>,
    /// Whether to return documents (Cohere). Defaults to false.
    pub return_documents: Option<bool>,

    // LLM reranker
    /// Nested LLM spec `{ provider, config }` for the LLM reranker.
    pub llm: Option<Value>,
    /// LLM provider name (when not using a nested `llm`).
    pub provider: Option<String>,
    /// LLM sampling temperature (default 0.0 for reranking).
    pub temperature: Option<f64>,
    /// LLM max tokens (default 100 for reranking).
    pub max_tokens: Option<u32>,
}

impl RerankerSettings {
    /// Parse settings from a JSON config object.
    pub fn from_value(value: &Value) -> Result<Self> {
        if value.is_null() {
            return Ok(Self::default());
        }
        Ok(serde_json::from_value(value.clone())?)
    }
}
