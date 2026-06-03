//! Embedder configuration parsed from the `config` object of an
//! `{ provider, config }` embedder spec.

use mem0_core::error::Result;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;

/// Azure-specific connection settings (mirrors `AzureConfig`).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct AzureKwargs {
    /// Azure API key.
    pub api_key: Option<String>,
    /// Azure deployment name (used as the model).
    pub azure_deployment: Option<String>,
    /// Azure endpoint base URL.
    pub azure_endpoint: Option<String>,
    /// Azure API version.
    pub api_version: Option<String>,
    /// Extra default headers.
    #[serde(default)]
    pub default_headers: Option<HashMap<String, String>>,
}

/// Provider-agnostic embedder settings. Unknown fields are ignored.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct EmbedderSettings {
    /// Model identifier.
    pub model: Option<String>,
    /// API key.
    pub api_key: Option<String>,
    /// Embedding dimensionality (when set, requested explicitly from the API).
    pub embedding_dims: Option<usize>,
    /// OpenAI / OpenAI-compatible base URL.
    pub openai_base_url: Option<String>,
    /// Ollama base URL.
    pub ollama_base_url: Option<String>,
    /// Generic base URL override (used for testability and compatible backends).
    pub base_url: Option<String>,
    /// Gemini output dimensionality.
    pub output_dimensionality: Option<usize>,
    /// Azure connection settings.
    #[serde(default)]
    pub azure_kwargs: AzureKwargs,
}

impl EmbedderSettings {
    /// Parse settings from a JSON config object.
    pub fn from_value(value: &Value) -> Result<Self> {
        if value.is_null() {
            return Ok(Self::default());
        }
        Ok(serde_json::from_value(value.clone())?)
    }
}
