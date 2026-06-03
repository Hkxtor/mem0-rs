//! Reranker factory. Port of `RerankerFactory`.

use crate::cohere::CohereReranker;
use crate::config::RerankerSettings;
use crate::llm::LlmReranker;
use crate::zero_entropy::ZeroEntropyReranker;
use mem0_core::error::{Mem0Error, Result};
use mem0_core::traits::Reranker;
use serde_json::Value;

/// Build a [`Reranker`] for the given provider name and config object.
pub fn build_reranker(provider: &str, config: &Value) -> Result<Box<dyn Reranker>> {
    let settings = RerankerSettings::from_value(config)?;
    match provider {
        "cohere" => Ok(Box::new(CohereReranker::new(settings)?)),
        "zero_entropy" => Ok(Box::new(ZeroEntropyReranker::new(settings)?)),
        "llm" | "llm_reranker" => Ok(Box::new(LlmReranker::new(settings)?)),
        other => Err(Mem0Error::configuration(format!(
            "Unsupported reranker provider: {other}"
        ))),
    }
}
