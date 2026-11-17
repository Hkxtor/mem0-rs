//! M2 tests for mem0-embed: request shaping and response parsing via wiremock.

use mem0_core::traits::MemoryAction;
use mem0_embed::build_embedder;
use serde_json::json;
use wiremock::matchers::{body_partial_json, header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn openai_embed_single() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(header("authorization", "Bearer test-key"))
        .and(body_partial_json(
            json!({ "model": "text-embedding-3-small", "encoding_format": "float" }),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [ { "embedding": [0.1, 0.2, 0.3], "index": 0 } ]
        })))
        .mount(&server)
        .await;

    let cfg = json!({ "api_key": "test-key", "openai_base_url": server.uri() });
    let embedder = build_embedder("openai", &cfg).unwrap();
    let v = embedder
        .embed("hello world", MemoryAction::Add)
        .await
        .unwrap();
    assert_eq!(v, vec![0.1, 0.2, 0.3]);
    assert_eq!(embedder.dims(), 1536);
}

#[tokio::test]
async fn openai_embed_batch_sorts_by_index() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                { "embedding": [2.0], "index": 1 },
                { "embedding": [1.0], "index": 0 }
            ]
        })))
        .mount(&server)
        .await;

    let cfg = json!({ "api_key": "k", "openai_base_url": server.uri(), "embedding_dims": 1 });
    let embedder = build_embedder("openai", &cfg).unwrap();
    let vs = embedder
        .embed_batch(&["a".to_string(), "b".to_string()], MemoryAction::Add)
        .await
        .unwrap();
    assert_eq!(vs, vec![vec![1.0], vec![2.0]]);
    assert_eq!(embedder.dims(), 1);
}

#[tokio::test]
async fn openai_compatible_requires_base_url() {
    let cfg = json!({ "api_key": "k" });
    assert!(build_embedder("openai_like", &cfg).is_err());
}

#[tokio::test]
async fn azure_embed_uses_deployment_and_api_key_header() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openai/deployments/my-deploy/embeddings"))
        .and(query_param("api-version", "2024-02-01"))
        .and(header("api-key", "az-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [ { "embedding": [0.5, 0.6], "index": 0 } ]
        })))
        .mount(&server)
        .await;

    let cfg = json!({
        "azure_kwargs": {
            "api_key": "az-key",
            "azure_deployment": "my-deploy",
            "azure_endpoint": server.uri()
        }
    });
    let embedder = build_embedder("azure_openai", &cfg).unwrap();
    let v = embedder.embed("hi", MemoryAction::Search).await.unwrap();
    assert_eq!(v, vec![0.5, 0.6]);
}

#[tokio::test]
async fn ollama_embed_parses_embeddings() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .and(body_partial_json(
            json!({ "model": "nomic-embed-text", "input": "hello" }),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embeddings": [ [0.7, 0.8, 0.9] ]
        })))
        .mount(&server)
        .await;

    let cfg = json!({ "ollama_base_url": server.uri() });
    let embedder = build_embedder("ollama", &cfg).unwrap();
    let v = embedder.embed("hello", MemoryAction::Add).await.unwrap();
    assert_eq!(v, vec![0.7, 0.8, 0.9]);
    assert_eq!(embedder.dims(), 512);
}

#[tokio::test]
async fn gemini_embed_parses_values() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-embedding-001:embedContent"))
        .and(query_param("key", "g-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embedding": { "values": [0.11, 0.22] }
        })))
        .mount(&server)
        .await;

    let cfg = json!({ "api_key": "g-key", "base_url": server.uri() });
    let embedder = build_embedder("gemini", &cfg).unwrap();
    let v = embedder.embed("hi", MemoryAction::Add).await.unwrap();
    assert_eq!(v, vec![0.11, 0.22]);
}

#[tokio::test]
async fn mock_embed_fixed_vector() {
    let embedder = build_embedder("mock", &json!({})).unwrap();
    let v = embedder.embed("anything", MemoryAction::Add).await.unwrap();
    assert_eq!(v.len(), 10);
    assert_eq!(embedder.dims(), 10);
}

#[tokio::test]
async fn unknown_provider_errors() {
    assert!(build_embedder("does-not-exist", &json!({})).is_err());
}

#[tokio::test]
async fn gemini_embed_qualifies_a_bare_model_name() {
    // Live API check: POST /v1beta/gemini-embedding-001:embedContent?key=… answers
    // 404 with an empty body, while /v1beta/models/gemini-embedding-001:… reaches the
    // real handler. The Python SDK qualifies the name for callers; we must too.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-embedding-001:embedContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embedding": { "values": [0.4, 0.5] }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let cfg = json!({
        "api_key": "test-key",
        "model": "gemini-embedding-001",
        "gemini_base_url": server.uri(),
        "embedding_dims": 768,
    });
    let embedder = build_embedder("gemini", &cfg).unwrap();
    let v = embedder.embed("hi", MemoryAction::Add).await.unwrap();
    assert_eq!(v, vec![0.4, 0.5]);
}

#[tokio::test]
async fn gemini_embed_already_qualified_model_is_untouched() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-embedding-001:embedContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embedding": { "values": [1.0] }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let cfg = json!({
        "api_key": "k",
        "model": "models/gemini-embedding-001",
        "gemini_base_url": server.uri(),
    });
    let embedder = build_embedder("gemini", &cfg).unwrap();
    assert_eq!(
        embedder.embed("hi", MemoryAction::Add).await.unwrap(),
        vec![1.0]
    );
}

/// The embedder is the section that hits quota first (a single import fans one text per
/// chunk), so a rate-limited embedding has to keep its status and `Retry-After` too.
#[tokio::test]
async fn a_rate_limited_embedding_keeps_its_status_and_retry_after() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "3")
                .set_body_string(r#"{"error":{"message":"quota"}}"#),
        )
        .mount(&server)
        .await;

    let embedder = build_embedder(
        "openai",
        &json!({"api_key": "k", "openai_base_url": server.uri(), "model": "text-embedding-3-small"}),
    )
    .expect("build");
    let err = embedder
        .embed("hello", MemoryAction::Add)
        .await
        .unwrap_err();
    assert_eq!(err.http_status(), Some(429));
    assert_eq!(err.retry_after_secs(), Some(3));
    assert_eq!(err.code(), Some("EMBED_001"));
}

// ---------- input_type: the asymmetric-model handshake ----------

/// The body the embedder actually sent, as JSON.
async fn sent_body(server: &MockServer) -> serde_json::Value {
    let requests = server.received_requests().await.unwrap();
    serde_json::from_slice(&requests.last().unwrap().body).unwrap()
}

async fn embedder_with(
    server: &MockServer,
    extra: serde_json::Value,
) -> Box<dyn mem0_core::traits::Embedder> {
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [ { "embedding": [0.1], "index": 0 } ]
        })))
        .mount(server)
        .await;
    let mut cfg = json!({ "api_key": "k", "openai_base_url": server.uri() });
    if let (Some(c), Some(e)) = (cfg.as_object_mut(), extra.as_object()) {
        for (k, v) in e {
            c.insert(k.clone(), v.clone());
        }
    }
    build_embedder("openai", &cfg).unwrap()
}

#[tokio::test]
async fn input_type_is_absent_unless_configured() {
    let server = MockServer::start().await;
    let embedder = embedder_with(&server, json!({})).await;
    embedder.embed("hi", MemoryAction::Add).await.unwrap();
    let body = sent_body(&server).await;
    assert!(
        body.get("input_type").is_none(),
        "a provider that does not know the field must never see it: {body}"
    );
}

#[tokio::test]
async fn auto_input_type_follows_the_call() {
    let server = MockServer::start().await;
    let embedder = embedder_with(&server, json!({ "input_type": "auto" })).await;

    embedder.embed("a memory", MemoryAction::Add).await.unwrap();
    assert_eq!(sent_body(&server).await["input_type"], "passage");

    embedder
        .embed("a question", MemoryAction::Search)
        .await
        .unwrap();
    assert_eq!(sent_body(&server).await["input_type"], "query");

    embedder
        .embed("rewritten", MemoryAction::Update)
        .await
        .unwrap();
    assert_eq!(sent_body(&server).await["input_type"], "passage");
}

#[tokio::test]
async fn a_fixed_input_type_ignores_the_call() {
    let server = MockServer::start().await;
    let embedder = embedder_with(&server, json!({ "input_type": "passage" })).await;
    embedder
        .embed("a question", MemoryAction::Search)
        .await
        .unwrap();
    assert_eq!(sent_body(&server).await["input_type"], "passage");
}

#[tokio::test]
async fn a_batch_carries_the_same_input_type() {
    let server = MockServer::start().await;
    let embedder = embedder_with(&server, json!({ "input_type": "auto" })).await;
    let texts = vec!["one".to_string(), "two".to_string()];
    embedder
        .embed_batch(&texts, MemoryAction::Add)
        .await
        .unwrap();
    assert_eq!(sent_body(&server).await["input_type"], "passage");
}

#[test]
fn an_unknown_input_type_is_a_configuration_error() {
    let cfg = json!({ "api_key": "k", "openai_base_url": "http://127.0.0.1:9/v1", "input_type": "sideways" });
    let err = match build_embedder("openai", &cfg) {
        Ok(_) => panic!("an unknown input_type must not build"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("not one of: auto, query, passage"), "{err}");
}

#[test]
fn an_empty_input_type_means_absent() {
    let cfg =
        json!({ "api_key": "k", "openai_base_url": "http://127.0.0.1:9/v1", "input_type": "  " });
    assert!(build_embedder("openai", &cfg).is_ok());
}
