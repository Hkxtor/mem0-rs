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
    input_type: InputType,
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    input: &'a [String],
    model: &'a str,
    encoding_format: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_type: Option<&'static str>,
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

/// What to send as `input_type`, decided once at build time.
///
/// Providers that do not know the field reject the whole request when it appears, so the
/// default is to leave it out entirely rather than to guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputType {
    /// Not sent. Every provider accepts a body without it.
    None,
    /// `passage` when storing, `query` when searching — the asymmetric contract.
    Auto,
    /// Always this value, whatever the call is doing.
    Fixed(&'static str),
}

impl InputType {
    fn parse(raw: Option<&str>) -> Result<Self> {
        let Some(raw) = raw.map(str::trim).filter(|v| !v.is_empty()) else {
            return Ok(Self::None);
        };
        match raw.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "query" => Ok(Self::Fixed("query")),
            "passage" | "document" => Ok(Self::Fixed("passage")),
            other => Err(Mem0Error::configuration(format!(
                "embedder input_type '{other}' is not one of: auto, query, passage"
            ))),
        }
    }

    fn for_action(self, action: MemoryAction) -> Option<&'static str> {
        match self {
            Self::None => None,
            Self::Fixed(v) => Some(v),
            // Storing is a passage; looking something up is a query. `Update` rewrites stored
            // text, so it is a passage too.
            Self::Auto => Some(match action {
                MemoryAction::Search => "query",
                MemoryAction::Add | MemoryAction::Update => "passage",
            }),
        }
    }
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
        let input_type = InputType::parse(settings.input_type.as_deref())?;
        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
            base_url: base.trim_end_matches('/').to_string(),
            model,
            dims,
            pass_dims,
            input_type,
        })
    }

    async fn embed_request(
        &self,
        inputs: Vec<String>,
        action: MemoryAction,
    ) -> Result<Vec<Vec<f32>>> {
        let req = EmbedRequest {
            input: &inputs,
            model: &self.model,
            encoding_format: "float",
            dimensions: if self.pass_dims {
                Some(self.dims)
            } else {
                None
            },
            input_type: self.input_type.for_action(action),
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
            let status = resp.status();
            let retry_after = retry_after(&resp);
            let body = resp.text().await.unwrap_or_default();
            return Err(Mem0Error::embedding_http(
                status.as_u16(),
                retry_after,
                format!("OpenAI embeddings HTTP {status}: {body}"),
            ));
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
    async fn embed(&self, text: &str, action: MemoryAction) -> Result<Vec<f32>> {
        let cleaned = text.replace('\n', " ");
        let mut v = self.embed_request(vec![cleaned], action).await?;
        v.pop()
            .ok_or_else(|| Mem0Error::embedding("OpenAI returned no embedding"))
    }

    async fn embed_batch(&self, texts: &[String], action: MemoryAction) -> Result<Vec<Vec<f32>>> {
        const MAX_BATCH: usize = 100;
        let cleaned: Vec<String> = texts.iter().map(|t| t.replace('\n', " ")).collect();
        let mut out = Vec::with_capacity(cleaned.len());
        for chunk in cleaned.chunks(MAX_BATCH) {
            out.extend(self.embed_request(chunk.to_vec(), action).await?);
        }
        Ok(out)
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
