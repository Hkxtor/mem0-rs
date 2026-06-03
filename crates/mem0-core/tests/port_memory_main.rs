//! Ported behavioral cases from mem0 `tests/memory/test_main.py` (sync).
//!
//! The Python suite is heavily mock-internal (asserting MagicMock call
//! patterns) and duplicates each case for `AsyncMemory`. The Rust port is
//! async-native, so we port the observable behaviors once: UTC timestamps,
//! created_at preservation + updated_at bump on update, payload shape, metadata
//! immutability, role/actor in history, actor preservation on update, and the
//! validation errors. Prompt-vs-custom-instructions selection is exercised via
//! the additive prompt builder (see `m1_pipeline.rs`).

use mem0_core::testing::{InMemoryVectorStore, MockEmbedder, MockLlm};
use mem0_core::types::{JsonMap, Message};
use mem0_core::{AddOptions, Memory, MemoryConfig, SearchOptions};
use serde_json::json;

fn memory() -> Memory {
    let cfg = MemoryConfig {
        history_db_path: ":memory:".into(),
        ..Default::default()
    };
    Memory::new(
        cfg,
        Box::new(MockEmbedder::new(32)),
        Box::new(MockLlm::new("{\"memory\": []}")),
        Box::new(InMemoryVectorStore::new()),
        None,
    )
    .unwrap()
}

fn umap(user: &str) -> JsonMap {
    let mut m = JsonMap::new();
    m.insert("user_id".into(), json!(user));
    m
}

fn raw(user: &str) -> AddOptions {
    AddOptions {
        user_id: Some(user.into()),
        infer: Some(false),
        ..Default::default()
    }
}

#[tokio::test]
async fn create_memory_uses_utc_timestamps() {
    let mem = memory();
    let r = mem.add("hello", raw("u1")).await.unwrap();
    let got = mem.get(&r[0].id).await.unwrap().unwrap();
    let created = got["created_at"].as_str().unwrap();
    assert!(created.ends_with("+00:00") || created.ends_with('Z'));
}

#[tokio::test]
async fn create_memory_sets_updated_at_equal_to_created() {
    let mem = memory();
    let r = mem.add("hello", raw("u1")).await.unwrap();
    let got = mem.get(&r[0].id).await.unwrap().unwrap();
    assert_eq!(got["created_at"], got["updated_at"]);
}

#[tokio::test]
async fn search_and_get_all_return_same_timestamps() {
    let mem = memory();
    let r = mem.add("I love hiking", raw("u1")).await.unwrap();
    let id = r[0].id.clone();
    let all = mem.get_all(&umap("u1"), 10).await.unwrap();
    let a = &all["results"][0];
    let s = mem
        .search("hiking", &umap("u1"), SearchOptions::default())
        .await
        .unwrap();
    let sr = &s["results"][0];
    assert_eq!(a["created_at"], sr["created_at"]);
    assert_eq!(a["updated_at"], sr["updated_at"]);
    assert_eq!(a["id"], json!(id));
}

#[tokio::test]
async fn update_preserves_created_at_and_bumps_updated_at() {
    let mem = memory();
    let r = mem.add("v1", raw("u1")).await.unwrap();
    let id = r[0].id.clone();
    let before = mem.get(&id).await.unwrap().unwrap();
    let created = before["created_at"].as_str().unwrap().to_string();

    mem.update(&id, "v2", None).await.unwrap();
    let after = mem.get(&id).await.unwrap().unwrap();
    assert_eq!(after["created_at"].as_str().unwrap(), created);
    assert!(after["updated_at"].as_str().unwrap() >= created.as_str());
    assert_eq!(after["memory"], "v2");
}

#[tokio::test]
async fn create_memory_stores_correct_payload() {
    let mem = memory();
    let mut meta = JsonMap::new();
    meta.insert("foo".into(), json!("bar"));
    let r = mem
        .add(
            vec![Message::user("hello")],
            AddOptions {
                user_id: Some("u1".into()),
                infer: Some(false),
                metadata: Some(meta),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let got = mem.get(&r[0].id).await.unwrap().unwrap();
    assert_eq!(got["memory"], "hello");
    assert_eq!(got["metadata"]["foo"], "bar");
    assert_eq!(got["role"], "user");
    assert!(got["hash"].is_string());
}

#[tokio::test]
async fn create_memory_with_none_metadata() {
    let mem = memory();
    let r = mem.add("hi", raw("u1")).await.unwrap();
    assert_eq!(r.len(), 1);
}

#[tokio::test]
async fn shared_metadata_across_calls_not_mutated() {
    let mem = memory();
    let mut meta = JsonMap::new();
    meta.insert("k".into(), json!("v"));
    for content in ["a", "b"] {
        mem.add(
            content,
            AddOptions {
                user_id: Some("u1".into()),
                infer: Some(false),
                metadata: Some(meta.clone()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    }
    // The caller's metadata map is untouched.
    assert_eq!(meta.len(), 1);
    assert_eq!(meta["k"], "v");
    let all = mem.get_all(&umap("u1"), 10).await.unwrap();
    assert_eq!(all["results"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn preserves_role_and_actor_id_in_history() {
    let mem = memory();
    let msg = Message {
        role: "user".into(),
        content: "hi".into(),
        name: Some("alice".into()),
    };
    let r = mem.add(vec![msg], raw("u1")).await.unwrap();
    let hist = mem.history(&r[0].id).await.unwrap();
    assert_eq!(hist.len(), 1);
    assert_eq!(hist[0].actor_id.as_deref(), Some("alice"));
    assert_eq!(hist[0].role.as_deref(), Some("user"));
}

#[tokio::test]
async fn update_preserves_actor_id_when_different_actor_updates() {
    let mem = memory();
    let msg = Message {
        role: "user".into(),
        content: "hi".into(),
        name: Some("alice".into()),
    };
    let r = mem.add(vec![msg], raw("u1")).await.unwrap();
    let id = r[0].id.clone();
    mem.update(&id, "updated", None).await.unwrap();
    let got = mem.get(&id).await.unwrap().unwrap();
    assert_eq!(got["actor_id"], "alice");
}

#[tokio::test]
async fn invalid_memory_type_errors() {
    let mem = memory();
    let e = mem
        .add(
            "x",
            AddOptions {
                user_id: Some("u1".into()),
                memory_type: Some("bogus".into()),
                ..Default::default()
            },
        )
        .await;
    assert!(e.is_err());
}

#[tokio::test]
async fn requires_at_least_one_session_id() {
    let mem = memory();
    let e = mem
        .add(
            "x",
            AddOptions {
                infer: Some(false),
                ..Default::default()
            },
        )
        .await;
    assert!(e.is_err());
}

#[tokio::test]
async fn empty_llm_response_returns_nothing() {
    let mem = memory();
    let r = mem
        .add(
            "just chatter",
            AddOptions {
                user_id: Some("u1".into()),
                infer: Some(true),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(r.is_empty());
}
