//! Import-fidelity tests: an importer-provided id / created_at / updated_at must survive
//! into the stored record, and ordinary adds (no identity given) must behave exactly as
//! before the fields existed.

use mem0_core::testing::{InMemoryVectorStore, MockEmbedder, MockLlm};
use mem0_core::types::Message;
use mem0_core::{AddOptions, Memory, MemoryConfig};
use serde_json::json;

fn memory() -> Memory {
    let cfg = MemoryConfig {
        history_db_path: ":memory:".to_string(),
        ..Default::default()
    };
    Memory::new(
        cfg,
        Box::new(MockEmbedder::new(64)),
        Box::new(MockLlm::with_responses(vec![])),
        Box::new(InMemoryVectorStore::new()),
        None,
    )
    .expect("construct memory")
}

fn raw_opts(user: &str) -> AddOptions {
    AddOptions {
        user_id: Some(user.into()),
        infer: Some(false),
        ..Default::default()
    }
}

#[tokio::test]
async fn raw_add_without_identity_generates_id_and_stamps_now() {
    let mem = memory();
    let res = mem
        .add(
            vec![Message {
                role: "user".into(),
                content: "I like hiking".into(),
                name: None,
            }],
            raw_opts("alice"),
        )
        .await
        .unwrap();
    assert_eq!(res.len(), 1);
    // A fresh UUID was generated, not empty.
    assert_eq!(res[0].id.len(), 36);

    let all = mem
        .get_all(&json!({"user_id": "alice"}).as_object().unwrap().clone(), 10)
        .await
        .unwrap();
    let item = &all["results"][0];
    assert_eq!(item["id"], json!(res[0].id));
    // updated_at defaults to created_at, as before.
    assert_eq!(item["created_at"], item["updated_at"]);
}

#[tokio::test]
async fn raw_add_with_identity_preserves_id_and_timestamps() {
    let mem = memory();
    let opts = AddOptions {
        id: Some("aac45e70-56d5-4d79-9d0b-df736d7ef129".into()),
        created_at: Some("2026-11-06T18:08:56-08:00".into()),
        updated_at: Some("2026-11-06T18:09:20.014941-08:00".into()),
        ..raw_opts("alice")
    };
    let res = mem
        .add(
            vec![Message {
                role: "user".into(),
                content: "User asked how many memories are stored".into(),
                name: None,
            }],
            opts,
        )
        .await
        .unwrap();
    assert_eq!(res[0].id, "aac45e70-56d5-4d79-9d0b-df736d7ef129");

    let all = mem
        .get_all(&json!({"user_id": "alice"}).as_object().unwrap().clone(), 10)
        .await
        .unwrap();
    let item = &all["results"][0];
    assert_eq!(item["id"], json!("aac45e70-56d5-4d79-9d0b-df736d7ef129"));
    assert_eq!(item["created_at"], json!("2026-11-06T18:08:56-08:00"));
    assert_eq!(
        item["updated_at"],
        json!("2026-11-06T18:09:20.014941-08:00")
    );
}

#[tokio::test]
async fn infer_add_ignores_identity_for_multiple_records_but_keeps_timestamps() {
    let llm = r#"{"memory":[{"text":"User's name is John"},{"text":"User loves hiking"}]}"#;
    let cfg = MemoryConfig {
        history_db_path: ":memory:".to_string(),
        ..Default::default()
    };
    let mem = Memory::new(
        cfg,
        Box::new(MockEmbedder::new(64)),
        Box::new(MockLlm::with_responses(vec![llm.to_string()])),
        Box::new(InMemoryVectorStore::new()),
        None,
    )
    .unwrap();
    let opts = AddOptions {
        user_id: Some("u1".into()),
        infer: Some(true),
        id: Some("should-not-collide".into()),
        created_at: Some("2026-01-01T00:00:00+00:00".into()),
        ..Default::default()
    };
    let res = mem.add("Hi, I'm John and I love hiking", opts).await.unwrap();
    assert_eq!(res.len(), 2);
    // Two records: the imported id may only land on one of them, the other gets a UUID.
    let ids: Vec<&str> = res.iter().map(|r| r.id.as_str()).collect();
    assert!(ids.contains(&"should-not-collide"));
    let other = ids.iter().find(|i| **i != "should-not-collide").unwrap();
    assert_eq!(other.len(), 36, "second record must get a fresh UUID");
}

#[tokio::test]
async fn infer_add_single_record_uses_imported_id() {
    let llm = r#"{"memory":[{"text":"User lives in Berlin"}]}"#;
    let cfg = MemoryConfig {
        history_db_path: ":memory:".to_string(),
        ..Default::default()
    };
    let mem = Memory::new(
        cfg,
        Box::new(MockEmbedder::new(64)),
        Box::new(MockLlm::with_responses(vec![llm.to_string()])),
        Box::new(InMemoryVectorStore::new()),
        None,
    )
    .unwrap();
    let opts = AddOptions {
        user_id: Some("u1".into()),
        infer: Some(true),
        id: Some("fixed-id-1".into()),
        ..Default::default()
    };
    let res = mem.add("I live in Berlin", opts).await.unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].id, "fixed-id-1");
}
