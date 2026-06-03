//! OpenAI and OpenAI-compatible embedder. Port of `embeddings/openai.py`.

use crate::config::EmbedderSettings;
use crate::http_error;
use async_trait::async_trait;
use mem0_core::error::{Mem0Error, Result};
use mem0_core::traits::{Embedder, MemoryAction};
use serde::{Deserialize, Serialize};

/// OpenAI / OpenAI-compatible embedder over the `/embeddings` endpoint.
pub struct OpenAiEmbedder {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    dims: usize,
    pass_dims: bool,
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    input: &'a [String],
    model: &'a str,
    encoding_format: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

#[derive(Deserialize)]
struct EmbedItem {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedItem>,
}

impl OpenAiEmbedder {
    /// Construct an OpenAI embedder (defaults to the public OpenAI base URL).
    pub fn new(settings: EmbedderSettings) -> Result<Self> {
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
        Self::build(settings, api_key, base)
    }

    /// Construct an OpenAI-compatible embedder (requires an explicit base URL).
    pub fn new_compatible(settings: EmbedderSettings) -> Result<Self> {
        let base = settings
            .openai_base_url
            .clone()
            .or_else(|| settings.base_url.clone())
            .ok_or_else(|| {
                Mem0Error::configuration(
                    "OpenAI-compatible embedder requires 'openai_base_url' in config",
                )
            })?;
        let api_key = settings
            .api_key
            .clone()
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .unwrap_or_default();
        Self::build(settings, api_key, base)
    }

    fn build(settings: EmbedderSettings, api_key: String, base: String) -> Result<Self> {
        let model = settings
            .model
            .clone()
            .unwrap_or_else(|| "text-embedding-3-small".to_string());
        let pass_dims = settings.embedding_dims.is_some();
        let dims = settings.embedding_dims.unwrap_or(1536);
        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
            base_url: base.trim_end_matches('/').to_string(),
            model,
            dims,
            pass_dims,
        })
    }

    async fn embed_request(&self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let req = EmbedRequest {
            input: &inputs,
            model: &self.model,
            encoding_format: "float",
            dimensions: if self.pass_dims {
                Some(self.dims)
            } else {
                None
            },
        };
        let url = format!("{}/embeddings", self.base_url);
        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&req)
            .send()
            .await
            .map_err(|e| http_error("OpenAI embeddings request failed", e))?;
        if !resp.status().is_success() {
            let code = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(Mem0Error::embedding(format!(
                "OpenAI embeddings HTTP {code}: {body}"
            )));
        }
        let mut parsed: EmbedResponse = resp
            .json()
            .await
            .map_err(|e| http_error("OpenAI embeddings decode failed", e))?;
        parsed.data.sort_by_key(|i| i.index);
        Ok(parsed.data.into_iter().map(|i| i.embedding).collect())
    }
}

#[async_trait]
impl Embedder for OpenAiEmbedder {
    async fn embed(&self, text: &str, _action: MemoryAction) -> Result<Vec<f32>> {
        let cleaned = text.replace('\n', " ");
        let mut v = self.embed_request(vec![cleaned]).await?;
        v.pop()
            .ok_or_else(|| Mem0Error::embedding("OpenAI returned no embedding"))
    }

    async fn embed_batch(&self, texts: &[String], _action: MemoryAction) -> Result<Vec<Vec<f32>>> {
        const MAX_BATCH: usize = 100;
        let cleaned: Vec<String> = texts.iter().map(|t| t.replace('\n', " ")).collect();
        let mut out = Vec::with_capacity(cleaned.len());
        for chunk in cleaned.chunks(MAX_BATCH) {
            out.extend(self.embed_request(chunk.to_vec()).await?);
        }
        Ok(out)
    }

    fn dims(&self) -> usize {
        self.dims
    }
}
