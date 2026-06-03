//! Anthropic chat LLM. Port of `llms/anthropic.py` (Messages API).

use crate::config::LlmSettings;
use crate::{http_error, to_wire_messages};
use async_trait::async_trait;
use mem0_core::error::{Mem0Error, Result};
use mem0_core::traits::{GenerateOptions, Llm};
use mem0_core::types::Message;
use serde_json::{json, Value};

/// Anthropic chat LLM over `…/v1/messages`.
pub struct AnthropicLlm {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    temperature: f64,
    max_tokens: u32,
    top_p: Option<f64>,
}

impl AnthropicLlm {
    /// Construct an Anthropic LLM from settings.
    pub fn new(settings: LlmSettings) -> Result<Self> {
        let api_key = settings
            .api_key
            .clone()
            .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
            .unwrap_or_default();
        let base = settings
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.anthropic.com".to_string());
        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
            base_url: base.trim_end_matches('/').to_string(),
            model: settings
                .model
                .clone()
                .unwrap_or_else(|| "claude-3-5-sonnet-20240620".to_string()),
            temperature: settings.temperature(),
            max_tokens: settings.max_tokens(),
            // Anthropic forbids temperature + top_p together; keep temperature.
            top_p: settings.top_p,
        })
    }
}

#[async_trait]
impl Llm for AnthropicLlm {
    async fn generate(&self, messages: &[Message], options: &GenerateOptions) -> Result<String> {
        let mut system = String::new();
        let mut filtered: Vec<&Message> = Vec::new();
        for m in messages {
            if m.role == "system" {
                system = m.content.clone();
            } else {
                filtered.push(m);
            }
        }
        let wire: Vec<Value> =
            to_wire_messages(&filtered.iter().map(|m| (*m).clone()).collect::<Vec<_>>());

        let mut body = json!({
            "model": self.model,
            "messages": wire,
            "system": system,
            "max_tokens": options.max_tokens.unwrap_or(self.max_tokens),
        });
        let obj = body.as_object_mut().unwrap();
        // Prefer temperature; only send top_p if temperature is absent.
        let temperature = options
            .temperature
            .map(|t| t as f64)
            .unwrap_or(self.temperature);
        obj.insert("temperature".into(), json!(temperature));
        let _ = self.top_p; // retained for parity; not sent alongside temperature

        let url = format!("{}/v1/messages", self.base_url);
        let resp = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .map_err(|e| http_error("Anthropic request failed", e))?;
        if !resp.status().is_success() {
            let code = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Mem0Error::llm(format!("Anthropic HTTP {code}: {text}")));
        }
        let value: Value = resp
            .json()
            .await
            .map_err(|e| http_error("Anthropic decode failed", e))?;
        let content = value
            .get("content")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string();
        Ok(content)
    }
}
