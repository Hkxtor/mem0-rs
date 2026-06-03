//! M5 tests for mem0-rerank.

use mem0_rerank::build_reranker;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn docs() -> Vec<String> {
    vec!["doc about cats".to_string(), "doc about dogs".to_string()]
}

#[tokio::test]
async fn cohere_rerank_returns_ranked_indices() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/rerank"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [
                { "index": 1, "relevance_score": 0.91 },
                { "index": 0, "relevance_score": 0.12 }
            ]
        })))
        .mount(&server)
        .await;

    let r = build_reranker(
        "cohere",
        &json!({ "api_key": "k", "base_url": server.uri() }),
    )
    .unwrap();
    let out = r.rerank("query", &docs(), 2).await.unwrap();
    assert_eq!(out, vec![(1, 0.91), (0, 0.12)]);
}

#[tokio::test]
async fn zero_entropy_rerank_sorts_desc() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/models/rerank"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [
                { "index": 0, "relevance_score": 0.3 },
                { "index": 1, "relevance_score": 0.8 }
            ]
        })))
        .mount(&server)
        .await;

    let r = build_reranker(
        "zero_entropy",
        &json!({ "api_key": "k", "base_url": server.uri() }),
    )
    .unwrap();
    let out = r.rerank("query", &docs(), 2).await.unwrap();
    assert_eq!(out[0].0, 1);
    assert!(out[0].1 > out[1].1);
}

#[tokio::test]
async fn llm_reranker_scores_each_document() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [ { "message": { "content": "0.87" } } ]
        })))
        .mount(&server)
        .await;

    let cfg = json!({
        "llm": {
            "provider": "openai",
            "config": { "api_key": "k", "openai_base_url": server.uri(), "model": "gpt-4o-mini" }
        }
    });
    let r = build_reranker("llm", &cfg).unwrap();
    let out = r
        .rerank("query", &["only doc".to_string()], 1)
        .await
        .unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].0, 0);
    assert!((out[0].1 - 0.87).abs() < 1e-5);
}

#[tokio::test]
async fn unknown_provider_errors() {
    assert!(build_reranker("nope", &json!({})).is_err());
}
