//! Mock LLM returning a canned response (default: an empty additive payload).
//! Useful for a runnable default facade without API keys, and for tests.

use crate::config::LlmSettings;
use async_trait::async_trait;
use mem0_core::error::Result;
use mem0_core::traits::{GenerateOptions, Llm};
use mem0_core::types::Message;

/// A mock LLM that always returns a fixed response.
pub struct MockLlm {
    response: String,
}

impl MockLlm {
    /// Build from settings (`config.response`, else an empty additive payload).
    pub fn new(settings: &LlmSettings) -> Self {
        // `response` is not a typed field; default to empty additive output.
        let _ = settings;
        Self {
            response: "{\"memory\": []}".to_string(),
        }
    }
}

#[async_trait]
impl Llm for MockLlm {
    async fn generate(&self, _messages: &[Message], _options: &GenerateOptions) -> Result<String> {
        Ok(self.response.clone())
    }
}
