//! `format_memory_item` must surface `categories` as a top-level array (default []),
//! matching the official export shape, and keep it out of `metadata`.

use mem0_core::memory::format_memory_item_for_test;
use mem0_core::types::{JsonMap, SearchHit};
use serde_json::{json, Value};

fn hit(id: &str, payload: Value) -> SearchHit {
    SearchHit {
        id: id.to_string(),
        score: 0.0,
        payload: payload.as_object().unwrap().clone(),
    }
}

#[test]
fn categories_is_top_level_array_and_not_in_metadata() {
    let mut payload = JsonMap::new();
    payload.insert("data".into(), json!("likes hiking"));
    payload.insert("user_id".into(), json!("alice"));
    payload.insert("categories".into(), json!(["hobbies", "sports"]));
    payload.insert("channel".into(), json!("weixin"));
    let item = format_memory_item_for_test(&hit("m1", json!(payload)), false);
    assert_eq!(item["categories"], json!(["hobbies", "sports"]), "{item}");
    assert_eq!(item["metadata"]["channel"], json!("weixin"), "{item}");
    assert!(item["metadata"].get("categories").is_none(), "{item}");
}

#[test]
fn categories_defaults_to_empty_array() {
    let mut payload = JsonMap::new();
    payload.insert("data".into(), json!("plain fact"));
    let item = format_memory_item_for_test(&hit("m2", json!(payload)), false);
    assert_eq!(item["categories"], json!([]), "{item}");
    assert!(item.get("metadata").is_none(), "{item}");
}

#[tokio::test]
async fn list_all_filtered_applies_categories_without_scope() {
    use mem0_core::testing::MockEmbedder;
    use mem0_core::testing::MockLlm;
    use mem0_core::Memory;
    use mem0_core::config::MemoryConfig;
    use mem0_core::types::JsonMap;
    use serde_json::json;

    let store = Box::new(mem0_core::testing::InMemoryVectorStore::new());
    let mem = Memory::new(
        MemoryConfig::default(),
        Box::new(MockEmbedder::new(4)),
        Box::new(MockLlm::new("{\"memory\": []}")),
        store,
        None,
    )
    .unwrap();

    let mk = |id: &str, cats: Value| {
        let mut p = JsonMap::new();
        p.insert("data".into(), json!(format!("text {id}")));
        p.insert("categories".into(), cats);
        p
    };
    let hits = vec![
        mem0_core::types::SearchHit { id: "a".into(), score: 0.0, payload: mk("a", json!(["food"])) },
        mem0_core::types::SearchHit { id: "b".into(), score: 0.0, payload: mk("b", json!(["food", "travel"])) },
    ];
    // insert via the store trait directly
    for h in hits {
        mem.vector_store()
            .insert(vec![mem0_core::types::VectorRecord {
                id: h.id.clone(),
                vector: vec![0.1; 4],
                payload: h.payload.clone(),
            }])
            .await
            .unwrap();
    }

    let mut filters = JsonMap::new();
    filters.insert("categories".into(), json!(["food", "travel"]));
    let out = mem.list_all_filtered(&filters, 10).await.unwrap();
    let ids: Vec<&str> = out["results"].as_array().unwrap().iter().map(|r| r["id"].as_str().unwrap()).collect();
    assert_eq!(ids, vec!["b"], "AND filter without scope: {ids:?}");

    // no filters at all still works (regression for plain list)
    let all = mem.list_all_filtered(&JsonMap::new(), 10).await.unwrap();
    assert_eq!(all["results"].as_array().unwrap().len(), 2);
}
