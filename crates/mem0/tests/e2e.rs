//! M6 end-to-end facade tests (mock providers, no network).

use mem0::{AddOptions, JsonMap, MemoryConfig, SearchOptions};
use serde_json::json;

fn mock_config() -> MemoryConfig {
    MemoryConfig::from_json(
        r#"{
            "embedder": { "provider": "mock" },
            "llm": { "provider": "mock" },
            "vector_store": { "provider": "embedded" },
            "history_db_path": ":memory:"
        }"#,
    )
    .unwrap()
}

fn umap(user: &str) -> JsonMap {
    let mut m = JsonMap::new();
    m.insert("user_id".into(), json!(user));
    m
}

#[tokio::test]
async fn from_config_wires_providers_and_runs() {
    let mem = mem0::from_config(mock_config()).unwrap();

    let added = mem
        .add(
            "I love hiking",
            AddOptions {
                user_id: Some("u1".into()),
                infer: Some(false),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(added.len(), 1);
    let id = added[0].id.clone();

    let got = mem.get(&id).await.unwrap().unwrap();
    assert_eq!(got.get("memory").unwrap(), "I love hiking");

    let all = mem.get_all(&umap("u1"), 10).await.unwrap();
    assert_eq!(all["results"].as_array().unwrap().len(), 1);

    let search = mem
        .search("hiking", &umap("u1"), SearchOptions::default())
        .await
        .unwrap();
    assert!(!search["results"].as_array().unwrap().is_empty());

    // Mock LLM returns an empty additive payload → inferred add stores nothing.
    let inferred = mem
        .add(
            "some chatter",
            AddOptions {
                user_id: Some("u1".into()),
                infer: Some(true),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(inferred.len(), 0);
}

#[test]
fn blocking_wrapper_works() {
    let mem = mem0::blocking::Memory::from_config(mock_config()).unwrap();
    let added = mem
        .add(
            "blocking memory",
            AddOptions {
                user_id: Some("b1".into()),
                infer: Some(false),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(added.len(), 1);

    let all = mem.get_all(&umap("b1"), 10).unwrap();
    assert_eq!(all["results"].as_array().unwrap().len(), 1);

    mem.reset().unwrap();
    let all = mem.get_all(&umap("b1"), 10).unwrap();
    assert_eq!(all["results"].as_array().unwrap().len(), 0);
}
