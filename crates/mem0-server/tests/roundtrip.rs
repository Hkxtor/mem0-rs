//! M7 integration test: server + client round-trip with mock providers.

use std::sync::Arc;

use mem0_client::Mem0Client;
use serde_json::json;

async fn start_server() -> String {
    let config = mem0::MemoryConfig::from_json(
        r#"{
            "embedder": { "provider": "mock" },
            "llm": { "provider": "mock" },
            "vector_store": { "provider": "embedded" },
            "history_db_path": ":memory:"
        }"#,
    )
    .unwrap();
    let memory = Arc::new(mem0::from_config(config).unwrap());
    let app = mem0_server::app(memory);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn server_client_roundtrip() {
    let base = start_server().await;
    let client = Mem0Client::new(base);

    // Add (non-inferred).
    let added = client
        .add(&json!({
            "messages": "I love hiking",
            "user_id": "u1",
            "infer": false
        }))
        .await
        .unwrap();
    let results = added["results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    let id = results[0]["id"].as_str().unwrap().to_string();

    // Get one.
    let got = client.get(&id).await.unwrap();
    assert_eq!(got["memory"], "I love hiking");

    // Get all.
    let all = client
        .get_all(Some("u1"), None, None, Some(10))
        .await
        .unwrap();
    assert_eq!(all["results"].as_array().unwrap().len(), 1);

    // Search.
    let search = client
        .search(&json!({ "query": "hiking", "user_id": "u1" }))
        .await
        .unwrap();
    assert!(!search["results"].as_array().unwrap().is_empty());

    // Update.
    client
        .update(&id, "I love trail running", None)
        .await
        .unwrap();
    let got = client.get(&id).await.unwrap();
    assert_eq!(got["memory"], "I love trail running");

    // History has ADD + UPDATE.
    let history = client.history(&id).await.unwrap();
    assert!(history["history"].as_array().unwrap().len() >= 2);

    // Delete.
    client.delete(&id).await.unwrap();
    let all = client
        .get_all(Some("u1"), None, None, Some(10))
        .await
        .unwrap();
    assert_eq!(all["results"].as_array().unwrap().len(), 0);

    // Reset.
    client.reset().await.unwrap();
}

#[tokio::test]
async fn health_check() {
    let base = start_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{base}/health"))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(resp["status"], "ok");
}
