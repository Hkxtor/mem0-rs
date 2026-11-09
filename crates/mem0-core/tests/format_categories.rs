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
