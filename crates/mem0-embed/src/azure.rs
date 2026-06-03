//! Azure OpenAI embedder. Port of `embeddings/azure_openai.py` (API-key auth).

use crate::config::EmbedderSettings;
use crate::http_error;
use async_trait::async_trait;
use mem0_core::error::{Mem0Error, Result};
use mem0_core::traits::{Embedder, MemoryAction};
use serde::{Deserialize, Serialize};

/// Azure OpenAI embedder over `…/openai/deployments/{deployment}/embeddings`.
pub struct AzureEmbedder {
    client: reqwest::Client,
    api_key: String,
    endpoint: String,
    deployment: String,
    api_version: String,
    dims: usize,
    default_headers: Vec<(String, String)>,
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    input: &'a [String],
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

impl AzureEmbedder {
    /// Construct an Azure embedder from settings (requires endpoint + deployment).
    pub fn new(settings: EmbedderSettings) -> Result<Self> {
        let az = &settings.azure_kwargs;
        let api_key = az
            .api_key
            .clone()
            .or_else(|| std::env::var("EMBEDDING_AZURE_OPENAI_API_KEY").ok())
            .unwrap_or_default();
        let endpoint = az
            .azure_endpoint
            .clone()
            .or_else(|| settings.base_url.clone())
            .or_else(|| std::env::var("EMBEDDING_AZURE_ENDPOINT").ok())
            .ok_or_else(|| Mem0Error::configuration("Azure embedder requires 'azure_endpoint'"))?;
        let deployment = az
            .azure_deployment
            .clone()
            .or_else(|| std::env::var("EMBEDDING_AZURE_DEPLOYMENT").ok())
            .ok_or_else(|| {
                Mem0Error::configuration("Azure embedder requires 'azure_deployment'")
            })?;
        let api_version = az
            .api_version
            .clone()
            .or_else(|| std::env::var("EMBEDDING_AZURE_API_VERSION").ok())
            .unwrap_or_else(|| "2024-02-01".to_string());
        let dims = settings.embedding_dims.unwrap_or(1536);
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
            dims,
            default_headers,
        })
    }

    async fn embed_request(&self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let url = format!(
            "{}/openai/deployments/{}/embeddings?api-version={}",
            self.endpoint, self.deployment, self.api_version
        );
        let mut request = self
            .client
            .post(&url)
            .header("api-key", &self.api_key)
            .json(&EmbedRequest { input: &inputs });
        for (k, v) in &self.default_headers {
            request = request.header(k, v);
        }
        let resp = request
            .send()
            .await
            .map_err(|e| http_error("Azure embeddings request failed", e))?;
        if !resp.status().is_success() {
            let code = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(Mem0Error::embedding(format!(
                "Azure embeddings HTTP {code}: {body}"
            )));
        }
        let mut parsed: EmbedResponse = resp
            .json()
            .await
            .map_err(|e| http_error("Azure embeddings decode failed", e))?;
        parsed.data.sort_by_key(|i| i.index);
        Ok(parsed.data.into_iter().map(|i| i.embedding).collect())
    }
}

#[async_trait]
impl Embedder for AzureEmbedder {
    async fn embed(&self, text: &str, _action: MemoryAction) -> Result<Vec<f32>> {
        let cleaned = text.replace('\n', " ");
        let mut v = self.embed_request(vec![cleaned]).await?;
        v.pop()
            .ok_or_else(|| Mem0Error::embedding("Azure returned no embedding"))
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
