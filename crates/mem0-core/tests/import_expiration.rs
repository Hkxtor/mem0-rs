//! Expiration-date behaviour: stored on add, emitted on reads, hidden from
//! get_all/list_all/search once expired (mirroring OSS `_payload_is_expired`).

use mem0_core::testing::{InMemoryVectorStore, MockEmbedder, MockLlm};
use mem0_core::types::{JsonMap, Message};
use mem0_core::{AddOptions, Memory, MemoryConfig, SearchOptions};
use serde_json::{json, Value};

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
    .unwrap()
}

fn raw(user: &str, exp: Option<&str>) -> AddOptions {
    AddOptions {
        user_id: Some(user.into()),
        infer: Some(false),
        expiration_date: exp.map(|s| s.to_string()),
        ..Default::default()
    }
}

fn msg(text: &str) -> Vec<Message> {
    vec![Message {
        role: "user".into(),
        content: text.into(),
        name: None,
    }]
}

fn scope(user: &str) -> JsonMap {
    json!({"user_id": user}).as_object().unwrap().clone()
}

#[tokio::test]
async fn invalid_expiration_date_is_rejected_on_add() {
    let mem = memory();
    let err = mem
        .add(msg("hello"), raw("alice", Some("not-a-date")))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("YYYY-MM-DD"), "{err}");
}

#[tokio::test]
async fn expired_memory_is_hidden_from_list_and_search_but_stays_in_the_store() {
    let mem = memory();
    let yesterday = (chrono::Utc::now().date_naive() - chrono::Duration::days(1)).to_string();
    let tomorrow = (chrono::Utc::now().date_naive() + chrono::Duration::days(1)).to_string();

    mem.add(msg("old fact"), raw("alice", Some(&yesterday)))
        .await
        .unwrap();
    mem.add(msg("fresh fact"), raw("alice", Some(&tomorrow)))
        .await
        .unwrap();
    mem.add(msg("timeless fact"), raw("alice", None))
        .await
        .unwrap();

    // list/get_all hide the expired one
    let all = mem.get_all(&scope("alice"), 100).await.unwrap();
    let texts: Vec<&str> = all["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["memory"].as_str().unwrap())
        .collect();
    assert_eq!(texts.len(), 2, "{all}");
    assert!(!texts.contains(&"old fact"), "{texts:?}");

    // search hides it too
    let found = mem
        .search(
            "fact",
            &scope("alice"),
            SearchOptions {
                top_k: 10,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let found_texts: Vec<&str> = found["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["memory"].as_str().unwrap())
        .collect();
    assert!(!found_texts.contains(&"old fact"), "{found_texts:?}");

    // ...but the raw store still holds it (export reads list_all which also hides it;
    // the store itself is the source of truth for export-with-expired — see app tests).
    let raw_hits = mem
        .vector_store()
        .list(&JsonMap::new(), Some(100))
        .await
        .unwrap();
    assert_eq!(raw_hits.len(), 3, "store keeps the expired row");
}

#[tokio::test]
async fn formatted_items_always_carry_expiration_date() {
    let mem = memory();
    let tomorrow = (chrono::Utc::now().date_naive() + chrono::Duration::days(1)).to_string();
    mem.add(msg("dated fact"), raw("alice", Some(&tomorrow)))
        .await
        .unwrap();
    mem.add(msg("undated fact"), raw("alice", None))
        .await
        .unwrap();

    let all = mem.get_all(&scope("alice"), 100).await.unwrap();
    let rows = all["results"].as_array().unwrap();
    for row in rows {
        assert!(
            row.as_object().unwrap().contains_key("expiration_date"),
            "{row}"
        );
        // and it must not leak into the metadata bucket
        if let Some(md) = row.get("metadata") {
            assert!(
                !md.as_object().unwrap().contains_key("expiration_date"),
                "{row}"
            );
        }
    }
    let dated = rows
        .iter()
        .find(|r| r["memory"] == json!("dated fact"))
        .unwrap();
    assert_eq!(dated["expiration_date"], json!(tomorrow));
    let undated = rows
        .iter()
        .find(|r| r["memory"] == json!("undated fact"))
        .unwrap();
    assert_eq!(undated["expiration_date"], Value::Null);
}

#[tokio::test]
async fn expiring_today_is_still_valid() {
    let mem = memory();
    let today = chrono::Utc::now().date_naive().to_string();
    mem.add(msg("today fact"), raw("alice", Some(&today)))
        .await
        .unwrap();
    let all = mem.get_all(&scope("alice"), 100).await.unwrap();
    assert_eq!(all["results"].as_array().unwrap().len(), 1, "{all}");
}
