//! Ported from mem0 `tests/configs/test_prompts.py`.
//!
//! Adapted: the non-empty-memory assertion checks the JSON rendering (Rust)
//! rather than Python's `str(list_of_dicts)` repr.

use mem0_core::prompts::{get_update_memory_messages, DEFAULT_UPDATE_MEMORY_PROMPT};
use serde_json::json;

#[test]
fn custom_prompt_prefix() {
    let old = vec![json!({ "id": "1", "text": "old memory 1" })];
    let resp = vec![json!("new fact")];
    let custom = "custom prompt determining memory update";
    let result = get_update_memory_messages(&old, &resp, Some(custom));
    assert!(result.starts_with(custom));
}

#[test]
fn default_prompt_prefix() {
    let old = vec![json!({ "id": "1", "text": "old memory 1" })];
    let resp = vec![json!("new fact")];
    let result = get_update_memory_messages(&old, &resp, None);
    assert!(result.starts_with(DEFAULT_UPDATE_MEMORY_PROMPT));
}

#[test]
fn empty_memory_message() {
    let result = get_update_memory_messages(&[], &[json!("new fact")], None);
    assert!(result.contains("Current memory is empty"));
}

#[test]
fn non_empty_memory() {
    let memory = vec![json!({ "id": "1", "text": "existing memory" })];
    let result = get_update_memory_messages(&memory, &[json!("new fact")], None);
    assert!(result.contains("existing memory"));
    assert!(result.contains("\"id\":\"1\""));
    assert!(result.contains("current content of my memory"));
}
