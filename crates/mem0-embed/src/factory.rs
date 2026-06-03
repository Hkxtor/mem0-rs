//! Embedder factory. Port of `EmbedderFactory`.

use crate::azure::AzureEmbedder;
use crate::config::EmbedderSettings;
use crate::gemini::GeminiEmbedder;
use crate::mock::MockEmbedder;
use crate::ollama::OllamaEmbedder;
use crate::openai::OpenAiEmbedder;
use mem0_core::error::{Mem0Error, Result};
use mem0_core::traits::Embedder;
use serde_json::Value;

/// Build an [`Embedder`] for the given provider name and config object.
pub fn build_embedder(provider: &str, config: &Value) -> Result<Box<dyn Embedder>> {
    let settings = EmbedderSettings::from_value(config)?;
    match provider {
        "openai" => Ok(Box::new(OpenAiEmbedder::new(settings)?)),
        "openai_like" | "openai_compatible" | "lmstudio" | "together" | "vllm" => {
            Ok(Box::new(OpenAiEmbedder::new_compatible(settings)?))
        }
        "azure_openai" | "azure" => Ok(Box::new(AzureEmbedder::new(settings)?)),
        "gemini" => Ok(Box::new(GeminiEmbedder::new(settings)?)),
        "ollama" => Ok(Box::new(OllamaEmbedder::new(settings)?)),
        "mock" => Ok(Box::new(MockEmbedder)),
        other => Err(Mem0Error::configuration(format!(
            "Unsupported embedder provider: {other}"
        ))),
    }
}
