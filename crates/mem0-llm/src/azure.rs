//! Azure OpenAI chat LLM. Port of `llms/azure_openai.py` (API-key auth).

use crate::config::{is_reasoning_model, LlmSettings};
use crate::openai::extract_chat_content;
use crate::{http_error, to_wire_messages};
use async_trait::async_trait;
use mem0_core::error::{Mem0Error, Result};
use mem0_core::traits::{GenerateOptions, Llm};
use mem0_core::types::Message;
use serde_json::{json, Value};

/// Azure OpenAI chat LLM over `…/openai/deployments/{deployment}/chat/completions`.
pub struct AzureLlm {
    client: reqwest::Client,
    api_key: String,
    endpoint: String,
    deployment: String,
    api_version: String,
    model: String,
    temperature: f64,
    max_tokens: u32,
    top_p: f64,
    reasoning_effort: Option<String>,
    default_headers: Vec<(String, String)>,
}

impl AzureLlm {
    /// Construct an Azure chat LLM (requires endpoint + deployment).
    pub fn new(settings: LlmSettings) -> Result<Self> {
        let az = &settings.azure_kwargs;
        let api_key = az
            .api_key
            .clone()
            .or_else(|| std::env::var("LLM_AZURE_OPENAI_API_KEY").ok())
            .unwrap_or_default();
        let endpoint = az
            .azure_endpoint
            .clone()
            .or_else(|| settings.base_url.clone())
            .or_else(|| std::env::var("LLM_AZURE_ENDPOINT").ok())
            .ok_or_else(|| Mem0Error::configuration("Azure LLM requires 'azure_endpoint'"))?;
        let deployment = az
            .azure_deployment
            .clone()
            .or_else(|| std::env::var("LLM_AZURE_DEPLOYMENT").ok())
            .ok_or_else(|| Mem0Error::configuration("Azure LLM requires 'azure_deployment'"))?;
        let api_version = az
            .api_version
            .clone()
            .or_else(|| std::env::var("LLM_AZURE_API_VERSION").ok())
            .unwrap_or_else(|| "2024-02-01".to_string());
        let default_headers = az
            .default_headers
            .clone()
            .map(|m| m.into_iter().collect())
            .unwrap_or_default();
        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
            endpoint: endpoint.trim_end_matches('/').to_string(),
            deployment,
            api_version,
            model: settings
                .model
                .clone()
                .unwrap_or_else(|| "gpt-5-mini".to_string()),
            temperature: settings.temperature(),
            max_tokens: settings.max_tokens(),
            top_p: settings.top_p(),
            reasoning_effort: settings.reasoning_effort.clone(),
            default_headers,
        })
    }
}

#[async_trait]
impl Llm for AzureLlm {
    async fn generate(&self, messages: &[Message], options: &GenerateOptions) -> Result<String> {
        // Port of the quirk: the last message has "assistant" replaced with "ai".
        let mut wire = to_wire_messages(messages);
        if let Some(last) = wire.last_mut() {
            if let Some(content) = last.get("content").and_then(|c| c.as_str()) {
                let replaced = content.replace("assistant", "ai");
                last["content"] = json!(replaced);
            }
        }

        let mut body = json!({ "model": self.model, "messages": wire });
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

        let url = format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            self.endpoint, self.deployment, self.api_version
        );
        let mut request = self
            .client
            .post(&url)
            .header("api-key", &self.api_key)
            .json(&body);
        for (k, v) in &self.default_headers {
            request = request.header(k, v);
        }
        let resp = request
            .send()
            .await
            .map_err(|e| http_error("Azure chat request failed", e))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let retry_after = retry_after(&resp);
            let text = resp.text().await.unwrap_or_default();
            return Err(Mem0Error::llm_http(
                status.as_u16(),
                retry_after,
                format!("Azure chat HTTP {status}: {text}"),
            ));
        }
        let value: Value = resp
            .json()
            .await
            .map_err(|e| http_error("Azure chat decode failed", e))?;
        Ok(extract_chat_content(&value))
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
