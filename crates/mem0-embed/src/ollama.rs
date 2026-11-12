//! Ollama embedder. Port of `embeddings/ollama.py` (REST `/api/embed`).

use crate::config::EmbedderSettings;
use crate::http_error;
use async_trait::async_trait;
use mem0_core::error::{Mem0Error, Result};
use mem0_core::traits::{Embedder, MemoryAction};
use serde::{Deserialize, Serialize};

/// Ollama embedder over `…/api/embed`.
pub struct OllamaEmbedder {
    client: reqwest::Client,
    base_url: String,
    model: String,
    dims: usize,
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

impl OllamaEmbedder {
    /// Construct an Ollama embedder from settings.
    pub fn new(settings: EmbedderSettings) -> Result<Self> {
        let base_url = settings
            .ollama_base_url
            .clone()
            .or_else(|| settings.base_url.clone())
            .unwrap_or_else(|| "http://localhost:11434".to_string());
        let model = settings
            .model
            .clone()
            .unwrap_or_else(|| "nomic-embed-text".to_string());
        let dims = settings.embedding_dims.unwrap_or(512);
        Ok(Self {
            client: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            model,
            dims,
        })
    }
}

#[async_trait]
impl Embedder for OllamaEmbedder {
    async fn embed(&self, text: &str, _action: MemoryAction) -> Result<Vec<f32>> {
        let url = format!("{}/api/embed", self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(&EmbedRequest {
                model: &self.model,
                input: text,
            })
            .send()
            .await
            .map_err(|e| http_error("Ollama embeddings request failed", e))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let retry_after = retry_after(&resp);
            let body = resp.text().await.unwrap_or_default();
            return Err(Mem0Error::embedding_http(
                status.as_u16(),
                retry_after,
                format!("Ollama embeddings HTTP {status}: {body}"),
            ));
        }
        let parsed: EmbedResponse = resp
            .json()
            .await
            .map_err(|e| http_error("Ollama embeddings decode failed", e))?;
        parsed
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| Mem0Error::embedding("Ollama returned no embeddings"))
    }

    fn dims(&self) -> usize {
        self.dims
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
