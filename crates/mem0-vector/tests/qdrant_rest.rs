//! M4 qdrant REST tests (run with `--features qdrant`).
#![cfg(feature = "qdrant")]

use mem0_core::types::{JsonMap, VectorRecord};
use mem0_vector::build_vector_store;
use serde_json::json;
use wiremock::matchers::{method, path, path_regex};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn qdrant_insert_and_search() {
    let server = MockServer::start().await;

    // Collection existence check → 200 (already exists).
    Mock::given(method("GET"))
        .and(path("/collections/mem0"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "result": {} })))
        .mount(&server)
        .await;

    // Insert points.
    Mock::given(method("PUT"))
        .and(path_regex(r"^/collections/mem0/points$"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({ "result": {}, "status": "ok" })),
        )
        .mount(&server)
        .await;

    // Query.
    Mock::given(method("POST"))
        .and(path("/collections/mem0/points/query"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": { "points": [
                { "id": "a", "score": 0.92, "payload": { "data": "hello", "user_id": "u1" } }
            ] }
        })))
        .mount(&server)
        .await;

    let store = build_vector_store("qdrant", &json!({ "url": server.uri() })).unwrap();

    let mut payload = JsonMap::new();
    payload.insert("data".into(), json!("hello"));
    payload.insert("user_id".into(), json!("u1"));
    store
        .insert(vec![VectorRecord {
            id: "a".into(),
            vector: vec![0.1, 0.2],
            payload,
        }])
        .await
        .unwrap();

    let mut filters = JsonMap::new();
    filters.insert("user_id".into(), json!("u1"));
    let hits = store.search("", &[0.1, 0.2], 5, &filters).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "a");
    assert!((hits[0].score - 0.92).abs() < 1e-5);
}
