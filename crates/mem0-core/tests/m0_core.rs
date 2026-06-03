//! Milestone M0 tests for `mem0-core`.

use mem0_core::filters::{build_filters_and_metadata, build_session_scope};
use mem0_core::history::HistoryStore;
use mem0_core::nlp::{extract_entities, lemmatize_for_bm25};
use mem0_core::scoring::{bm25_scores, get_bm25_params, normalize_bm25, score_and_rank};
use mem0_core::testing::{InMemoryVectorStore, MockEmbedder, MockLlm};
use mem0_core::types::{JsonMap, Message, SearchHit};
use mem0_core::{AddOptions, Memory, MemoryConfig};
use serde_json::{json, Value};
use std::collections::HashMap;

fn test_memory() -> Memory {
    let cfg = MemoryConfig {
        history_db_path: ":memory:".to_string(),
        ..Default::default()
    };
    Memory::new(
        cfg,
        Box::new(MockEmbedder::new(32)),
        Box::new(MockLlm::new("{\"memory\": []}")),
        Box::new(InMemoryVectorStore::new()),
        None,
    )
    .expect("construct memory")
}

fn umap(user: &str) -> JsonMap {
    let mut m = JsonMap::new();
    m.insert("user_id".into(), Value::String(user.into()));
    m
}

#[test]
fn history_store_roundtrip() {
    let db = HistoryStore::new(":memory:").unwrap();
    db.add_history(
        "m1",
        None,
        Some("hello"),
        "ADD",
        Some("t0"),
        Some("t0"),
        0,
        None,
        None,
    )
    .unwrap();
    db.add_history(
        "m1",
        Some("hello"),
        Some("hi"),
        "UPDATE",
        Some("t0"),
        Some("t1"),
        0,
        None,
        None,
    )
    .unwrap();
    let hist = db.get_history("m1").unwrap();
    assert_eq!(hist.len(), 2);
    assert_eq!(hist[0].event, "ADD");
    assert_eq!(hist[1].event, "UPDATE");
    assert_eq!(hist[1].old_memory.as_deref(), Some("hello"));
}

#[test]
fn history_messages_eviction() {
    let db = HistoryStore::new(":memory:").unwrap();
    let msgs: Vec<Message> = (0..15).map(|i| Message::user(format!("m{i}"))).collect();
    db.save_messages(&msgs, "user_id=u1").unwrap();
    let last = db.get_last_messages("user_id=u1", 10).unwrap();
    assert_eq!(last.len(), 10);
}

#[test]
fn filters_require_session_id() {
    let err = build_filters_and_metadata(None, None, None, None, None, None);
    assert!(err.is_err());
    let (meta, filters) =
        build_filters_and_metadata(Some("u1"), None, Some("r1"), None, None, None).unwrap();
    assert_eq!(meta.get("user_id").unwrap(), "u1");
    assert_eq!(filters.get("run_id").unwrap(), "r1");
}

#[test]
fn session_scope_is_deterministic() {
    let mut f = JsonMap::new();
    f.insert("user_id".into(), json!("u1"));
    f.insert("agent_id".into(), json!("a1"));
    assert_eq!(build_session_scope(&f), "agent_id=a1&user_id=u1");
}

#[test]
fn lemmatize_is_symmetric_and_drops_stopwords() {
    let a = lemmatize_for_bm25("The cats are running quickly");
    // stopwords (the, are) dropped; plurals/-ing normalized; -ing original kept.
    assert!(a.contains("cat"));
    assert!(a.contains("run"));
    assert!(!a.split_whitespace().any(|t| t == "the"));
}

#[test]
fn entities_proper_and_quoted() {
    let ents = extract_entities("Marcus visited Osteria Francescana and read \"The Nightingale\"");
    let texts: Vec<String> = ents.iter().map(|(_, t)| t.clone()).collect();
    assert!(texts.iter().any(|t| t.contains("Osteria Francescana")));
    assert!(texts.iter().any(|t| t == "The Nightingale"));
}

#[test]
fn scoring_params_and_rank() {
    let (mid, steep) = get_bm25_params("one two three");
    assert_eq!((mid, steep), (5.0, 0.7));
    let n = normalize_bm25(5.0, 5.0, 0.7);
    assert!((n - 0.5).abs() < 1e-9);

    let hits = vec![
        SearchHit {
            id: "a".into(),
            score: 0.9,
            payload: JsonMap::new(),
        },
        SearchHit {
            id: "b".into(),
            score: 0.05,
            payload: JsonMap::new(),
        },
    ];
    let ranked = score_and_rank(&hits, &HashMap::new(), &HashMap::new(), 0.1, 10);
    assert_eq!(ranked.len(), 1);
    assert_eq!(ranked[0].id, "a");
}

#[test]
fn bm25_scores_rank_relevant_doc() {
    let docs = vec![
        ("d1".to_string(), "cat dog run".to_string()),
        ("d2".to_string(), "airplane sky cloud".to_string()),
    ];
    let scores = bm25_scores("cat", &docs);
    assert!(scores.contains_key("d1"));
    assert!(!scores.contains_key("d2"));
}

#[tokio::test]
async fn raw_add_get_getall() {
    let mem = test_memory();
    let res = mem
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
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].event, "ADD");
    let id = res[0].id.clone();

    let got = mem.get(&id).await.unwrap().unwrap();
    assert_eq!(got.get("memory").unwrap(), "I love hiking in the mountains");
    assert_eq!(got.get("user_id").unwrap(), "u1");

    let all = mem.get_all(&umap("u1"), 20).await.unwrap();
    assert_eq!(all.get("results").unwrap().as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn raw_add_skips_system_messages() {
    let mem = test_memory();
    let msgs = vec![
        Message::system("you are helpful"),
        Message::user("my name is John"),
    ];
    let res = mem
        .add(
            msgs,
            AddOptions {
                user_id: Some("u2".into()),
                infer: Some(false),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].memory, "my name is John");
}

#[tokio::test]
async fn update_delete_history() {
    let mem = test_memory();
    let res = mem
        .add(
            "I like tea",
            AddOptions {
                user_id: Some("u3".into()),
                infer: Some(false),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let id = res[0].id.clone();

    mem.update(&id, "I like coffee", None).await.unwrap();
    let got = mem.get(&id).await.unwrap().unwrap();
    assert_eq!(got.get("memory").unwrap(), "I like coffee");

    let hist = mem.history(&id).await.unwrap();
    assert_eq!(hist.len(), 2);
    assert_eq!(hist[1].event, "UPDATE");

    mem.delete(&id).await.unwrap();
    assert!(mem.get(&id).await.unwrap().is_none());
    let hist = mem.history(&id).await.unwrap();
    assert_eq!(hist.last().unwrap().event, "DELETE");
}

#[tokio::test]
async fn delete_all_and_reset() {
    let mem = test_memory();
    for i in 0..3 {
        mem.add(
            format!("fact {i}"),
            AddOptions {
                user_id: Some("u4".into()),
                infer: Some(false),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    }
    let all = mem.get_all(&umap("u4"), 20).await.unwrap();
    assert_eq!(all.get("results").unwrap().as_array().unwrap().len(), 3);

    mem.delete_all(Some("u4"), None, None).await.unwrap();
    let all = mem.get_all(&umap("u4"), 20).await.unwrap();
    assert_eq!(all.get("results").unwrap().as_array().unwrap().len(), 0);

    mem.add(
        "another",
        AddOptions {
            user_id: Some("u4".into()),
            infer: Some(false),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    mem.reset().await.unwrap();
    let all = mem.get_all(&umap("u4"), 20).await.unwrap();
    assert_eq!(all.get("results").unwrap().as_array().unwrap().len(), 0);
}
