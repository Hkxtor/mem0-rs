//! OpenAI / OpenAI-compatible chat LLM. Port of `llms/openai.py`.

use crate::config::{is_reasoning_model, LlmSettings};
use crate::{http_error, to_wire_messages};
use async_trait::async_trait;
use mem0_core::error::{Mem0Error, Result};
use mem0_core::traits::{GenerateOptions, Llm};
use mem0_core::types::Message;
use serde_json::{json, Value};

/// OpenAI / OpenAI-compatible chat-completions LLM.
pub struct OpenAiLlm {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    temperature: f64,
    max_tokens: u32,
    top_p: f64,
    reasoning_effort: Option<String>,
}

impl OpenAiLlm {
    /// Construct an OpenAI LLM (defaults to the public OpenAI base URL).
    pub fn new(settings: LlmSettings) -> Result<Self> {
        let api_key = settings
            .api_key
            .clone()
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .unwrap_or_default();
        let base = settings
            .openai_base_url
            .clone()
            .or_else(|| std::env::var("OPENAI_BASE_URL").ok())
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        Ok(Self::build(settings, api_key, base, "gpt-5-mini"))
    }

    /// Construct an OpenAI-compatible LLM (requires an explicit base URL).
    pub fn new_compatible(settings: LlmSettings) -> Result<Self> {
        let base = settings
            .openai_base_url
            .clone()
            .or_else(|| settings.base_url.clone())
            .ok_or_else(|| {
                Mem0Error::configuration(
                    "OpenAI-compatible LLM requires 'openai_base_url' in config",
                )
            })?;
        let api_key = settings
            .api_key
            .clone()
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .unwrap_or_default();
        Ok(Self::build(settings, api_key, base, "gpt-5-mini"))
    }

    fn build(settings: LlmSettings, api_key: String, base: String, default_model: &str) -> Self {
        let model = settings
            .model
            .clone()
            .unwrap_or_else(|| default_model.to_string());
        Self {
            client: reqwest::Client::new(),
            api_key,
            base_url: base.trim_end_matches('/').to_string(),
            model,
            temperature: settings.temperature(),
            max_tokens: settings.max_tokens(),
            top_p: settings.top_p(),
            reasoning_effort: settings.reasoning_effort.clone(),
        }
    }

    fn build_body(&self, messages: &[Message], options: &GenerateOptions) -> Value {
        let mut body = json!({
            "model": self.model,
            "messages": to_wire_messages(messages),
        });
        let obj = body.as_object_mut().unwrap();
        if is_reasoning_model(&self.model) {
            if let Some(re) = &self.reasoning_effort {
                obj.insert("reasoning_effort".into(), json!(re));
            }
        } else {
            obj.insert(
                "temperature".into(),
                json!(options
                    .temperature
                    .map(|t| t as f64)
                    .unwrap_or(self.temperature)),
            );
            obj.insert(
                "max_tokens".into(),
                json!(options.max_tokens.unwrap_or(self.max_tokens)),
            );
            obj.insert(
                "top_p".into(),
                json!(options.top_p.map(|t| t as f64).unwrap_or(self.top_p)),
            );
        }
        if options.response_format_json {
            obj.insert("response_format".into(), json!({ "type": "json_object" }));
        }
        body
    }
}

#[async_trait]
impl Llm for OpenAiLlm {
    async fn generate(&self, messages: &[Message], options: &GenerateOptions) -> Result<String> {
        let url = format!("{}/chat/completions", self.base_url);
        let body = self.build_body(messages, options);
        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| http_error("OpenAI chat request failed", e))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let retry_after = retry_after(&resp);
            let text = resp.text().await.unwrap_or_default();
            return Err(Mem0Error::llm_http(
                status.as_u16(),
                retry_after,
                format!("OpenAI chat HTTP {status}: {text}"),
            ));
        }
        let value: Value = resp
            .json()
            .await
            .map_err(|e| http_error("OpenAI chat decode failed", e))?;
        Ok(extract_chat_content(&value))
    }
}

/// Extract `choices[0].message.content` from an OpenAI-style chat response.
pub(crate) fn extract_chat_content(value: &Value) -> String {
    value
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string()
}

/// The upstream's `Retry-After`, in seconds, when it sent one.
fn retry_after(resp: &reqwest::Response) -> Option<u64> {
    mem0_core::error::parse_retry_after(
        resp.headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok()),
    )
}
