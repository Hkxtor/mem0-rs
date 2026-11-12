//! Ollama chat LLM. Port of `llms/ollama.py` (REST `/api/chat`).

use crate::config::LlmSettings;
use crate::{http_error, to_wire_messages};
use async_trait::async_trait;
use mem0_core::error::{Mem0Error, Result};
use mem0_core::traits::{GenerateOptions, Llm};
use mem0_core::types::Message;
use serde_json::{json, Value};

/// Ollama chat LLM over `…/api/chat`.
pub struct OllamaLlm {
    client: reqwest::Client,
    base_url: String,
    model: String,
    temperature: f64,
    max_tokens: u32,
    top_p: f64,
}

impl OllamaLlm {
    /// Construct an Ollama LLM from settings.
    pub fn new(settings: LlmSettings) -> Result<Self> {
        let base = settings
            .ollama_base_url
            .clone()
            .or_else(|| settings.base_url.clone())
            .unwrap_or_else(|| "http://localhost:11434".to_string());
        Ok(Self {
            client: reqwest::Client::new(),
            base_url: base.trim_end_matches('/').to_string(),
            model: settings
                .model
                .clone()
                .unwrap_or_else(|| "llama3.1:70b".to_string()),
            temperature: settings.temperature(),
            max_tokens: settings.max_tokens(),
            top_p: settings.top_p(),
        })
    }
}

#[async_trait]
impl Llm for OllamaLlm {
    async fn generate(&self, messages: &[Message], options: &GenerateOptions) -> Result<String> {
        let mut wire = to_wire_messages(messages);
        if options.response_format_json {
            // Append a JSON instruction to the last user message (or add one).
            let appended = wire
                .last()
                .and_then(|m| m.get("role"))
                .and_then(|r| r.as_str())
                == Some("user");
            if appended {
                if let Some(last) = wire.last_mut() {
                    if let Some(c) = last.get("content").and_then(|c| c.as_str()) {
                        last["content"] =
                            json!(format!("{c}\n\nPlease respond with valid JSON only."));
                    }
                }
            } else {
                wire.push(
                    json!({ "role": "user", "content": "Please respond with valid JSON only." }),
                );
            }
        }

        let mut body = json!({
            "model": self.model,
            "messages": wire,
            "stream": false,
            "options": {
                "temperature": options.temperature.map(|t| t as f64).unwrap_or(self.temperature),
                "num_predict": options.max_tokens.unwrap_or(self.max_tokens),
                "top_p": options.top_p.map(|t| t as f64).unwrap_or(self.top_p),
            }
        });
        if options.response_format_json {
            body["format"] = json!("json");
        }

        let url = format!("{}/api/chat", self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| http_error("Ollama request failed", e))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let retry_after = retry_after(&resp);
            let text = resp.text().await.unwrap_or_default();
            return Err(Mem0Error::llm_http(
                status.as_u16(),
                retry_after,
                format!("Ollama HTTP {status}: {text}"),
            ));
        }
        let value: Value = resp
            .json()
            .await
            .map_err(|e| http_error("Ollama decode failed", e))?;
        let content = value
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();
        Ok(content)
    }
}

/// The upstream's `Retry-After`, in seconds, when it sent one.
fn retry_after(resp: &reqwest::Response) -> Option<u64> {
    mem0_core::error::parse_retry_after(
        resp.headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok()),
    )
}
