//! LLM factory. Port of `LlmFactory`.

use crate::anthropic::AnthropicLlm;
use crate::azure::AzureLlm;
use crate::config::LlmSettings;
use crate::gemini::GeminiLlm;
use crate::mock::MockLlm;
use crate::ollama::OllamaLlm;
use crate::openai::OpenAiLlm;
use mem0_core::error::{Mem0Error, Result};
use mem0_core::traits::Llm;
use serde_json::Value;

/// Build an [`Llm`] for the given provider name and config object.
pub fn build_llm(provider: &str, config: &Value) -> Result<Box<dyn Llm>> {
    let settings = LlmSettings::from_value(config)?;
    match provider {
        "openai" | "openai_structured" => Ok(Box::new(OpenAiLlm::new(settings)?)),
        "openai_like" | "openai_compatible" | "lmstudio" | "vllm" | "together" | "groq" | "xai"
        | "deepseek" | "sarvam" | "minimax" => Ok(Box::new(OpenAiLlm::new_compatible(settings)?)),
        "azure_openai" | "azure_openai_structured" | "azure" => {
            Ok(Box::new(AzureLlm::new(settings)?))
        }
        "anthropic" => Ok(Box::new(AnthropicLlm::new(settings)?)),
        "gemini" => Ok(Box::new(GeminiLlm::new(settings)?)),
        "ollama" => Ok(Box::new(OllamaLlm::new(settings)?)),
        "mock" => Ok(Box::new(MockLlm::new(&settings))),
        other => Err(Mem0Error::configuration(format!(
            "Unsupported LLM provider: {other}"
        ))),
    }
}
