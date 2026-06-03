//! Milestone M1 tests: inferred pipeline + search.

use mem0_core::testing::{InMemoryVectorStore, MockEmbedder, MockLlm};
use mem0_core::types::{JsonMap, Message};
use mem0_core::{AddOptions, Memory, MemoryConfig, SearchOptions};
use serde_json::{json, Value};

fn memory_with_llm(responses: Vec<String>) -> Memory {
    let cfg = MemoryConfig {
        history_db_path: ":memory:".to_string(),
        ..Default::default()
    };
    Memory::new(
        cfg,
        Box::new(MockEmbedder::new(64)),
        Box::new(MockLlm::with_responses(responses)),
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

fn u(user: &str) -> AddOptions {
    AddOptions {
        user_id: Some(user.into()),
        infer: Some(true),
        ..Default::default()
    }
}

#[tokio::test]
async fn infer_add_extracts_memories() {
    let llm = r#"{"memory":[{"text":"User's name is John","attributed_to":"user"},{"text":"User loves hiking","attributed_to":"user"}]}"#;
    let mem = memory_with_llm(vec![llm.to_string()]);
    let res = mem
        .add("Hi, I'm John and I love hiking", u("u1"))
        .await
        .unwrap();
    assert_eq!(res.len(), 2);
    assert!(res.iter().all(|r| r.event == "ADD"));

    let all = mem.get_all(&umap("u1"), 20).await.unwrap();
    assert_eq!(all["results"].as_array().unwrap().len(), 2);
    // payload carries the lemmatized field used for BM25.
    let first = &all["results"][0];
    assert!(first
        .get("memory")
        .unwrap()
        .as_str()
        .unwrap()
        .contains("User"));
}

#[tokio::test]
async fn infer_add_dedups_by_hash() {
    let llm = r#"{"memory":[{"text":"User likes tea"},{"text":"User likes tea"}]}"#;
    let mem = memory_with_llm(vec![llm.to_string()]);
    let res = mem.add("I like tea", u("u2")).await.unwrap();
    assert_eq!(
        res.len(),
        1,
        "identical extracted texts must dedup by md5 hash"
    );
}

#[tokio::test]
async fn infer_add_empty_extraction_returns_nothing() {
    let mem = memory_with_llm(vec![r#"{"memory":[]}"#.to_string()]);
    let res = mem.add("just chitchat, hello!", u("u3")).await.unwrap();
    assert_eq!(res.len(), 0);
    let all = mem.get_all(&umap("u3"), 20).await.unwrap();
    assert_eq!(all["results"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn search_ranks_relevant_first() {
    let mem = memory_with_llm(vec![]);
    for text in [
        "I love hiking in the mountains",
        "I enjoy cooking pasta",
        "My favorite color is blue",
    ] {
        mem.add(
            text,
            AddOptions {
                user_id: Some("u4".into()),
                infer: Some(false),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    }
    let res = mem
        .search("hiking trails", &umap("u4"), SearchOptions::default())
        .await
        .unwrap();
    let results = res["results"].as_array().unwrap();
    assert!(!results.is_empty());
    assert!(results[0]["memory"].as_str().unwrap().contains("hiking"));
    assert!(results[0].get("score").is_some());
}

#[tokio::test]
async fn search_honors_advanced_metadata_filters() {
    let mem = memory_with_llm(vec![]);
    let mut meta_a = JsonMap::new();
    meta_a.insert("category".into(), json!("food"));
    let mut meta_b = JsonMap::new();
    meta_b.insert("category".into(), json!("travel"));

    mem.add(
        "I love sushi",
        AddOptions {
            user_id: Some("u5".into()),
            infer: Some(false),
            metadata: Some(meta_a),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    mem.add(
        "I love Tokyo",
        AddOptions {
            user_id: Some("u5".into()),
            infer: Some(false),
            metadata: Some(meta_b),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    // Filter category == food via operator form.
    let mut filters = umap("u5");
    filters.insert("category".into(), json!({ "eq": "food" }));
    let res = mem
        .search("love", &filters, SearchOptions::default())
        .await
        .unwrap();
    let results = res["results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0]["memory"].as_str().unwrap().contains("sushi"));
}

#[tokio::test]
async fn procedural_memory() {
    let mem = memory_with_llm(vec!["Step 1: open file. Step 2: edit it.".to_string()]);
    let res = mem
        .add(
            vec![Message::user("I opened the file then edited it")],
            AddOptions {
                agent_id: Some("a1".into()),
                memory_type: Some("procedural_memory".into()),
                infer: Some(true),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(res.len(), 1);
    assert!(res[0].memory.contains("Step 1"));

    let mut filters = JsonMap::new();
    filters.insert("agent_id".into(), json!("a1"));
    let all = mem.get_all(&filters, 20).await.unwrap();
    assert_eq!(all["results"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn infer_add_with_entity_store_is_nonfatal() {
    let cfg = MemoryConfig {
        history_db_path: ":memory:".to_string(),
        ..Default::default()
    };
    let llm = r#"{"memory":[{"text":"Marcus works at Shopify in Toronto"}]}"#;
    let mem = Memory::new(
        cfg,
        Box::new(MockEmbedder::new(64)),
        Box::new(MockLlm::with_responses(vec![llm.to_string()])),
        Box::new(InMemoryVectorStore::new()),
        None,
    )
    .unwrap()
    .with_entity_store(Box::new(InMemoryVectorStore::new()));

    let res = mem
        .add("Marcus works at Shopify in Toronto", u("u6"))
        .await
        .unwrap();
    assert_eq!(res.len(), 1);
    let all = mem.get_all(&umap("u6"), 20).await.unwrap();
    assert_eq!(all["results"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn agent_scoped_infer_uses_agent_context() {
    // Agent-scoped (agent_id, no user_id) add should succeed through the
    // additive pipeline with the agent-context suffix appended.
    let llm = r#"{"memory":[{"text":"Agent specializes in travel planning","attributed_to":"assistant"}]}"#;
    let mem = memory_with_llm(vec![llm.to_string()]);
    let res = mem
        .add(
            vec![Message::assistant("I specialize in travel planning")],
            AddOptions {
                agent_id: Some("a2".into()),
                infer: Some(true),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(res.len(), 1);
    let mut filters = JsonMap::new();
    filters.insert("agent_id".into(), json!("a2"));
    let all = mem.get_all(&filters, 20).await.unwrap();
    assert_eq!(all["results"].as_array().unwrap().len(), 1);
}
