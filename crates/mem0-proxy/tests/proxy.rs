//! M8 tests for mem0-proxy.

use mem0::{AddOptions, MemoryConfig};
use mem0_core::types::Message;
use mem0_proxy::MemoryProxy;

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

#[tokio::test]
async fn proxy_chat_augments_and_responds() {
    let proxy = MemoryProxy::from_config(mock_config()).unwrap();

    // Seed a memory directly.
    proxy
        .memory()
        .add(
            "I love hiking in the mountains",
            AddOptions {
                user_id: Some("u1".into()),
                infer: Some(false),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let result = proxy
        .chat(
            vec![Message::user("hiking plans?")],
            Some("u1"),
            None,
            None,
            5,
        )
        .await
        .unwrap();

    // Mock LLM returns the empty additive payload as its chat response.
    assert!(result.response.contains("memory"));
    // The seeded memory shares the "hiking" token, so retrieval surfaces it.
    assert!(result.memories_used.iter().any(|m| m.contains("hiking")));
}

#[tokio::test]
async fn proxy_chat_without_memories_still_responds() {
    let proxy = MemoryProxy::from_config(mock_config()).unwrap();
    let result = proxy
        .chat(vec![Message::user("hello")], Some("u2"), None, None, 5)
        .await
        .unwrap();
    assert!(!result.response.is_empty());
}
