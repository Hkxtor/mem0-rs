//! ZeroEntropy reranker. Port of `reranker/zero_entropy_reranker.py`.

use crate::config::RerankerSettings;
use crate::http_error;
use async_trait::async_trait;
use mem0_core::error::{Mem0Error, Result};
use mem0_core::traits::Reranker;
use serde::Deserialize;
use serde_json::json;

/// ZeroEntropy reranker.
pub struct ZeroEntropyReranker {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    top_k: Option<usize>,
}

#[derive(Deserialize)]
struct RerankResult {
    index: usize,
    relevance_score: f32,
}

#[derive(Deserialize)]
struct RerankResponse {
    results: Vec<RerankResult>,
}

impl ZeroEntropyReranker {
    /// Construct a ZeroEntropy reranker from settings.
    pub fn new(settings: RerankerSettings) -> Result<Self> {
        let api_key = settings
            .api_key
            .clone()
            .or_else(|| std::env::var("ZERO_ENTROPY_API_KEY").ok())
            .ok_or_else(|| Mem0Error::configuration("ZeroEntropy reranker requires an API key"))?;
        let base_url = settings
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.zeroentropy.dev".to_string());
        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
            base_url: base_url.trim_end_matches('/').to_string(),
            model: settings
                .model
                .clone()
                .unwrap_or_else(|| "zerank-1".to_string()),
            top_k: settings.top_k,
        })
    }
}

#[async_trait]
impl Reranker for ZeroEntropyReranker {
    async fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_n: usize,
    ) -> Result<Vec<(usize, f32)>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }
        let body = json!({
            "model": self.model,
            "query": query,
            "documents": documents,
        });
        let url = format!("{}/v1/models/rerank", self.base_url);
        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| http_error("ZeroEntropy rerank request failed", e))?;
        if !resp.status().is_success() {
            let code = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Mem0Error::llm(format!(
                "ZeroEntropy rerank HTTP {code}: {text}"
            )));
        }
        let parsed: RerankResponse = resp
            .json()
            .await
            .map_err(|e| http_error("ZeroEntropy rerank decode failed", e))?;
        let mut out: Vec<(usize, f32)> = parsed
            .results
            .into_iter()
            .map(|r| (r.index, r.relevance_score))
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let n = if top_n == 0 {
            self.top_k.unwrap_or(out.len())
        } else {
            top_n
        };
        out.truncate(n);
        Ok(out)
    }
}
