//! LLM-based reranker. Port of `reranker/llm_reranker.py`.

use crate::config::RerankerSettings;
use crate::util::extract_score;
use async_trait::async_trait;
use mem0_core::error::{Mem0Error, Result};
use mem0_core::traits::{GenerateOptions, Llm, Reranker};
use mem0_core::types::Message;
use serde_json::{json, Value};

const SYSTEM_PROMPT: &str = "You are a relevance scoring assistant. \
Given a query and a document, score how relevant the document is to the query.\n\n\
Score the relevance on a scale from 0.0 to 1.0, where:\n\
- 1.0 = Perfectly relevant and directly answers the query\n\
- 0.8-0.9 = Highly relevant with good information\n\
- 0.6-0.7 = Moderately relevant with some useful information\n\
- 0.4-0.5 = Slightly relevant with limited useful information\n\
- 0.0-0.3 = Not relevant or no useful information\n\n\
Respond with only a single numerical score between 0.0 and 1.0. \
Do not include any explanation or additional text.";

const MAX_INPUT_LEN: usize = 4000;

/// LLM-based reranker that scores each document with an injected LLM.
pub struct LlmReranker {
    llm: Box<dyn Llm>,
    top_k: Option<usize>,
}

/// Resolve the inner LLM `(provider, config)` from reranker settings.
///
/// A nested `llm` spec overrides the provider and supplies provider-specific
/// config, while top-level `model`/`temperature`/`max_tokens`/`api_key` fill
/// in defaults without overriding values already set in the nested config.
/// A nested `config` of `null` (or absent) is treated as an empty object.
pub fn resolve_inner_llm(settings: &RerankerSettings) -> Result<(String, Value)> {
    if let Some(llm) = &settings.llm {
        let provider = llm
            .get("provider")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| settings.provider.clone())
            .unwrap_or_else(|| "openai".to_string());
        // Treat `config: null` (or missing) as an empty object.
        let mut config = llm
            .get("config")
            .cloned()
            .filter(|v| !v.is_null())
            .unwrap_or_else(|| json!({}));
        let obj = config.as_object_mut().ok_or_else(|| {
            Mem0Error::configuration("reranker 'llm.config' must be a JSON object")
        })?;
        obj.entry("model").or_insert_with(|| {
            json!(settings
                .model
                .clone()
                .unwrap_or_else(|| "gpt-4o-mini".into()))
        });
        obj.entry("temperature")
            .or_insert_with(|| json!(settings.temperature.unwrap_or(0.0)));
        obj.entry("max_tokens")
            .or_insert_with(|| json!(settings.max_tokens.unwrap_or(100)));
        if let Some(key) = &settings.api_key {
            obj.entry("api_key").or_insert_with(|| json!(key));
        }
        Ok((provider, config))
    } else {
        let provider = settings.provider.clone().unwrap_or_else(|| "openai".into());
        let mut config = json!({
            "model": settings.model.clone().unwrap_or_else(|| "gpt-4o-mini".into()),
            "temperature": settings.temperature.unwrap_or(0.0),
            "max_tokens": settings.max_tokens.unwrap_or(100),
        });
        if let Some(key) = &settings.api_key {
            config["api_key"] = json!(key);
        }
        Ok((provider, config))
    }
}

impl LlmReranker {
    /// Construct an LLM reranker, building the inner LLM from settings.
    pub fn new(settings: RerankerSettings) -> Result<Self> {
        let (provider, config) = resolve_inner_llm(&settings)?;
        let llm = mem0_llm::build_llm(&provider, &config)?;
        Ok(Self {
            llm,
            top_k: settings.top_k,
        })
    }

    /// Construct directly from an injected LLM (used for testing).
    pub fn with_llm(llm: Box<dyn Llm>, top_k: Option<usize>) -> Self {
        Self { llm, top_k }
    }
}

fn truncate(s: &str, limit: usize) -> String {
    s.chars().take(limit).collect()
}

#[async_trait]
impl Reranker for LlmReranker {
    async fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_n: usize,
    ) -> Result<Vec<(usize, f32)>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }
        let opts = GenerateOptions::default();
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(documents.len());
        for (idx, doc) in documents.iter().enumerate() {
            let user_message = format!(
                "Query: {}\n\nDocument: {}",
                truncate(query, MAX_INPUT_LEN),
                truncate(doc, MAX_INPUT_LEN)
            );
            let messages = vec![Message::system(SYSTEM_PROMPT), Message::user(user_message)];
            let score = match self.llm.generate(&messages, &opts).await {
                Ok(resp) => extract_score(&resp),
                Err(_) => 0.5,
            };
            scored.push((idx, score));
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let n = if top_n == 0 {
            self.top_k.unwrap_or(scored.len())
        } else {
            top_n
        };
        scored.truncate(n);
        Ok(scored)
    }
}
