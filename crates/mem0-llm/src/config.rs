//! LLM configuration parsed from the `config` object of an `{ provider, config }`
//! LLM spec. Defaults mirror `BaseLlmConfig`.

use mem0_core::error::Result;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;

/// Azure-specific connection settings.
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

/// Provider-agnostic LLM settings. Unknown fields are ignored.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct LlmSettings {
    /// Model identifier.
    pub model: Option<String>,
    /// API key.
    pub api_key: Option<String>,
    /// Sampling temperature (default 0.1).
    pub temperature: Option<f64>,
    /// Max tokens to generate (default 2000).
    pub max_tokens: Option<u32>,
    /// Nucleus sampling (default 0.1).
    pub top_p: Option<f64>,
    /// Top-k sampling.
    pub top_k: Option<u32>,
    /// Reasoning effort for reasoning models.
    pub reasoning_effort: Option<String>,
    /// OpenAI / OpenAI-compatible base URL.
    pub openai_base_url: Option<String>,
    /// Ollama base URL.
    pub ollama_base_url: Option<String>,
    /// Generic base URL override (Anthropic/Gemini and testability).
    pub base_url: Option<String>,
    /// Azure connection settings.
    #[serde(default)]
    pub azure_kwargs: AzureKwargs,
}

impl LlmSettings {
    /// Parse settings from a JSON config object.
    pub fn from_value(value: &Value) -> Result<Self> {
        if value.is_null() {
            return Ok(Self::default());
        }
        Ok(serde_json::from_value(value.clone())?)
    }

    /// Effective temperature (default 0.1).
    pub fn temperature(&self) -> f64 {
        self.temperature.unwrap_or(0.1)
    }
    /// Effective max tokens (default 2000).
    pub fn max_tokens(&self) -> u32 {
        self.max_tokens.unwrap_or(2000)
    }
    /// Effective top_p (default 0.1).
    pub fn top_p(&self) -> f64 {
        self.top_p.unwrap_or(0.1)
    }
}

/// Whether a model is a reasoning / GPT-5 model that rejects sampling params.
/// Port of `LLMBase._is_reasoning_model`.
pub fn is_reasoning_model(model: &str) -> bool {
    let lower = model.to_lowercase();
    let base = lower.rsplit('/').next().unwrap_or(&lower);
    const REASONING: &[&str] = &[
        "o1",
        "o1-preview",
        "o3-mini",
        "o3",
        "gpt-5",
        "gpt-5o",
        "gpt-5o-mini",
        "gpt-5o-micro",
    ];
    if REASONING.contains(&base) {
        return true;
    }
    ["o1-", "o1.", "o3-", "o3."]
        .iter()
        .any(|p| base.starts_with(p))
}
