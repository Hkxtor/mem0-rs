//! Gemini chat LLM. Port of `llms/gemini.py` (REST generateContent).

use crate::config::LlmSettings;
use crate::http_error;
use async_trait::async_trait;
use mem0_core::error::{Mem0Error, Result};
use mem0_core::traits::{GenerateOptions, Llm};
use mem0_core::types::Message;
use serde_json::{json, Value};

/// Gemini chat LLM over `…/v1beta/{model}:generateContent`.
pub struct GeminiLlm {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    temperature: f64,
    max_tokens: u32,
    top_p: f64,
}

impl GeminiLlm {
    /// Construct a Gemini LLM from settings.
    pub fn new(settings: LlmSettings) -> Result<Self> {
        let api_key = settings
            .api_key
            .clone()
            .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
            .unwrap_or_default();
        let base = settings
            .gemini_base_url
            .clone()
            .or_else(|| settings.base_url.clone())
            .unwrap_or_else(|| "https://generativelanguage.googleapis.com".to_string());
        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
            base_url: base.trim_end_matches('/').to_string(),
            model: settings
                .model
                .clone()
                .unwrap_or_else(|| "gemini-2.0-flash".to_string()),
            temperature: settings.temperature(),
            max_tokens: settings.max_tokens(),
            top_p: settings.top_p(),
        })
    }
}

#[async_trait]
impl Llm for GeminiLlm {
    async fn generate(&self, messages: &[Message], options: &GenerateOptions) -> Result<String> {
        let mut system_instruction: Option<String> = None;
        let mut contents: Vec<Value> = Vec::new();
        for m in messages {
            if m.role == "system" {
                system_instruction = Some(m.content.clone());
            } else {
                let role = if m.role == "assistant" {
                    "model"
                } else {
                    "user"
                };
                contents.push(json!({
                    "role": role,
                    "parts": [ { "text": m.content } ]
                }));
            }
        }

        let mut generation_config = json!({
            "temperature": options.temperature.map(|t| t as f64).unwrap_or(self.temperature),
            "maxOutputTokens": options.max_tokens.unwrap_or(self.max_tokens),
            "topP": options.top_p.map(|t| t as f64).unwrap_or(self.top_p),
        });
        if options.response_format_json {
            generation_config["responseMimeType"] = json!("application/json");
        }

        let mut body = json!({
            "contents": contents,
            "generationConfig": generation_config,
        });
        if let Some(si) = system_instruction {
            body["systemInstruction"] = json!({ "parts": [ { "text": si } ] });
        }

        let url = format!(
            "{}/v1beta/{}:generateContent?key={}",
            self.base_url, self.model, self.api_key
        );
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| http_error("Gemini request failed", e))?;
        if !resp.status().is_success() {
            let code = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Mem0Error::llm(format!("Gemini HTTP {code}: {text}")));
        }
        let value: Value = resp
            .json()
            .await
            .map_err(|e| http_error("Gemini decode failed", e))?;
        let text = value
            .get("candidates")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("content"))
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.as_array())
            .and_then(|parts| {
                parts
                    .iter()
                    .find_map(|p| p.get("text").and_then(|t| t.as_str()))
            })
            .unwrap_or("")
            .to_string();
        Ok(text)
    }
}
