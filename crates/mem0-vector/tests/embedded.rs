//! M4 tests for the embedded vector store.

use mem0_core::traits::VectorStore;
use mem0_core::types::{JsonMap, VectorRecord};
use mem0_vector::build_vector_store;
use mem0_vector::EmbeddedVectorStore;
use serde_json::json;

fn rec(id: &str, vector: Vec<f32>, data: &str, user: &str) -> VectorRecord {
    let mut payload = JsonMap::new();
    payload.insert("data".into(), json!(data));
    payload.insert(
        "text_lemmatized".into(),
        json!(mem0_core::nlp::lemmatize_for_bm25(data)),
    );
    payload.insert("user_id".into(), json!(user));
    payload.insert("created_at".into(), json!("2026-01-01T00:00:00Z"));
    VectorRecord {
        id: id.to_string(),
        vector,
        payload,
    }
}

fn umap(user: &str) -> JsonMap {
    let mut m = JsonMap::new();
    m.insert("user_id".into(), json!(user));
    m
}

#[tokio::test]
async fn cosine_search_and_filter() {
    let store = EmbeddedVectorStore::new("test", None).unwrap();
    store
        .insert(vec![
            rec("a", vec![1.0, 0.0], "I like cats", "u1"),
            rec("b", vec![0.0, 1.0], "I like dogs", "u1"),
            rec("c", vec![1.0, 0.0], "other user", "u2"),
        ])
        .await
        .unwrap();

    // Query close to "a"/"c" direction, but filtered to u1 → only "a".
    let hits = store
        .search("", &[1.0, 0.0], 10, &umap("u1"))
        .await
        .unwrap();
    assert_eq!(hits[0].id, "a");
    assert!(hits
        .iter()
        .all(|h| h.payload.get("user_id").unwrap() == "u1"));
}

#[tokio::test]
async fn keyword_search_bm25() {
    let store = EmbeddedVectorStore::new("test", None).unwrap();
    store
        .insert(vec![
            rec("a", vec![1.0, 0.0], "I enjoy hiking mountains", "u1"),
            rec("b", vec![0.0, 1.0], "I enjoy cooking pasta", "u1"),
        ])
        .await
        .unwrap();
    let hits = store
        .keyword_search("hiking", 10, &umap("u1"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "a");
}

#[tokio::test]
async fn advanced_filter_operators() {
    let store = EmbeddedVectorStore::new("test", None).unwrap();
    let mut p = rec("a", vec![1.0, 0.0], "food", "u1");
    p.payload.insert("category".into(), json!("food"));
    let mut q = rec("b", vec![1.0, 0.0], "travel", "u1");
    q.payload.insert("category".into(), json!("travel"));
    store.insert(vec![p, q]).await.unwrap();

    let mut filters = umap("u1");
    filters.insert("category".into(), json!({ "in": ["food"] }));
    let hits = store.list(&filters, None).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "a");
}

#[tokio::test]
async fn persistence_across_reload() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_string_lossy().to_string();

    {
        let store = build_vector_store(
            "embedded",
            &json!({ "collection_name": "mem", "path": path }),
        )
        .unwrap();
        store
            .insert(vec![rec("a", vec![1.0, 0.0], "persisted memory", "u1")])
            .await
            .unwrap();
    }

    // New instance from the same path must load the persisted record.
    let store2 = build_vector_store(
        "embedded",
        &json!({ "collection_name": "mem", "path": path }),
    )
    .unwrap();
    let got = store2.get("a").await.unwrap();
    assert!(got.is_some());
    assert_eq!(
        got.unwrap().payload.get("data").unwrap(),
        "persisted memory"
    );
}

#[tokio::test]
async fn unsupported_backends_report_clearly() {
    assert!(build_vector_store("nope", &json!({})).is_err());

    #[cfg(not(feature = "qdrant"))]
    assert!(build_vector_store("qdrant", &json!({ "url": "http://localhost:6333" })).is_err());
    #[cfg(not(feature = "pgvector"))]
    assert!(build_vector_store(
        "pgvector",
        &json!({ "connection_string": "postgresql://localhost/db" })
    )
    .is_err());
}
