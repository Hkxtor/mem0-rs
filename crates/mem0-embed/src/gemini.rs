//! Google Gemini embedder. Port of `embeddings/gemini.py` (REST embedContent).

use crate::config::EmbedderSettings;
use crate::http_error;
use async_trait::async_trait;
use mem0_core::error::{Mem0Error, Result};
use mem0_core::traits::{Embedder, MemoryAction};
use mem0_core::util::qualify_gemini_model;
use serde::{Deserialize, Serialize};

/// Gemini embedder over `…/v1beta/{model}:embedContent`.
pub struct GeminiEmbedder {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    dims: usize,
}

#[derive(Serialize)]
struct Part<'a> {
    text: &'a str,
}

#[derive(Serialize)]
struct Content<'a> {
    parts: Vec<Part<'a>>,
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    content: Content<'a>,
    #[serde(rename = "outputDimensionality")]
    output_dimensionality: usize,
}

#[derive(Deserialize)]
struct Embedding {
    values: Vec<f32>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embedding: Embedding,
}

impl GeminiEmbedder {
    /// Construct a Gemini embedder from settings.
    pub fn new(settings: EmbedderSettings) -> Result<Self> {
        let api_key = settings
            .api_key
            .clone()
            .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
            .unwrap_or_default();
        let base_url = settings
            .gemini_base_url
            .clone()
            .or_else(|| settings.base_url.clone())
            .unwrap_or_else(|| "https://generativelanguage.googleapis.com".to_string());
        // A bare `gemini-embedding-001` is what an operator naturally types, and the
        // SDK would qualify it; building the URL from it unqualified is an empty 404.
        let model = qualify_gemini_model(
            &settings
                .model
                .clone()
                .unwrap_or_else(|| "models/gemini-embedding-001".to_string()),
        );
        let dims = settings
            .embedding_dims
            .or(settings.output_dimensionality)
            .unwrap_or(768);
        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
            base_url: base_url.trim_end_matches('/').to_string(),
            model,
            dims,
        })
    }
}

#[async_trait]
impl Embedder for GeminiEmbedder {
    async fn embed(&self, text: &str, _action: MemoryAction) -> Result<Vec<f32>> {
        let cleaned = text.replace('\n', " ");
        let url = format!(
            "{}/v1beta/{}:embedContent?key={}",
            self.base_url, self.model, self.api_key
        );
        let req = EmbedRequest {
            content: Content {
                parts: vec![Part { text: &cleaned }],
            },
            output_dimensionality: self.dims,
        };
        let resp = self
            .client
            .post(&url)
            .json(&req)
            .send()
            .await
            .map_err(|e| http_error("Gemini embeddings request failed", e))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            // An empty body means we never reached the API's error handling — usually a
            // malformed path or the wrong host. Name the request (never the key) so the
            // 404 is actionable instead of `HTTP 404 Not Found: `.
            let detail = if body.trim().is_empty() {
                format!(
                    "(empty response body) for POST {}/v1beta/{}:embedContent",
                    self.base_url, self.model
                )
            } else {
                body
            };
            return Err(Mem0Error::embedding(format!(
                "Gemini embeddings HTTP {status}: {detail}"
            )));
        }
        let parsed: EmbedResponse = resp
            .json()
            .await
            .map_err(|e| http_error("Gemini embeddings decode failed", e))?;
        Ok(parsed.embedding.values)
    }

    fn dims(&self) -> usize {
        self.dims
    }
}
