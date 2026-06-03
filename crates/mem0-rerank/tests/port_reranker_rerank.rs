//! Ported from mem0 `tests/rerankers/test_llm_reranker_rerank.py`.
//!
//! Adapted: the Rust `Reranker` trait operates on pre-extracted document
//! strings (the orchestrator extracts `memory`/`text`/`content` fields before
//! calling the reranker), so the Python field-extraction tests are covered at
//! the `Memory::search` layer. Sorting / top_k / error-fallback / score
//! extraction are ported here using injected LLMs.

use async_trait::async_trait;
use mem0_core::error::{Mem0Error, Result};
use mem0_core::testing::MockLlm;
use mem0_core::traits::{GenerateOptions, Llm, Reranker};
use mem0_core::types::Message;
use mem0_rerank::llm::LlmReranker;
use mem0_rerank::util::extract_score;

// --- extract_score ---

#[test]
fn extract_score_valid() {
    assert!((extract_score("0.85") - 0.85).abs() < 1e-6);
    assert!((extract_score("0.0") - 0.0).abs() < 1e-6);
    assert!((extract_score("1.0") - 1.0).abs() < 1e-6);
    assert!((extract_score("The score is 0.72.") - 0.72).abs() < 1e-6);
    assert!((extract_score("Score: 0.9 out of 1.0") - 0.9).abs() < 1e-6);
}

#[test]
fn extract_score_fallback() {
    assert!((extract_score("no numbers here") - 0.5).abs() < 1e-6);
}

// --- rerank ---

fn docs(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("doc{i}")).collect()
}

#[tokio::test]
async fn empty_documents() {
    let r = LlmReranker::with_llm(Box::new(MockLlm::new("0.5")), None);
    assert!(r.rerank("query", &[], 10).await.unwrap().is_empty());
}

#[tokio::test]
async fn documents_sorted_by_score_descending() {
    let llm = MockLlm::with_responses(vec!["0.3".into(), "0.9".into(), "0.6".into()]);
    let r = LlmReranker::with_llm(Box::new(llm), None);
    let out = r.rerank("test query", &docs(3), 0).await.unwrap();
    assert_eq!(out.len(), 3);
    assert!((out[0].1 - 0.9).abs() < 1e-6);
    assert!((out[1].1 - 0.6).abs() < 1e-6);
    assert!((out[2].1 - 0.3).abs() < 1e-6);
    // The highest score corresponds to the second document (index 1).
    assert_eq!(out[0].0, 1);
}

#[tokio::test]
async fn top_k_limits_results() {
    let llm = MockLlm::with_responses(vec!["0.9".into(), "0.5".into(), "0.1".into()]);
    let r = LlmReranker::with_llm(Box::new(llm), None);
    let out = r.rerank("query", &docs(3), 2).await.unwrap();
    assert_eq!(out.len(), 2);
}

#[tokio::test]
async fn config_top_k_used_when_arg_not_provided() {
    let llm = MockLlm::with_responses(vec!["0.9".into(), "0.5".into(), "0.1".into()]);
    let r = LlmReranker::with_llm(Box::new(llm), Some(1));
    // top_n = 0 means "use the configured top_k".
    let out = r.rerank("query", &docs(3), 0).await.unwrap();
    assert_eq!(out.len(), 1);
}

/// An LLM that always errors, to exercise the fallback-score path.
struct ErrLlm;

#[async_trait]
impl Llm for ErrLlm {
    async fn generate(&self, _messages: &[Message], _options: &GenerateOptions) -> Result<String> {
        Err(Mem0Error::llm("simulated API error"))
    }
}

#[tokio::test]
async fn fallback_score_on_llm_error() {
    let r = LlmReranker::with_llm(Box::new(ErrLlm), None);
    let out = r.rerank("query", &["doc".to_string()], 10).await.unwrap();
    assert_eq!(out.len(), 1);
    assert!((out[0].1 - 0.5).abs() < 1e-6);
}
