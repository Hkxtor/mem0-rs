//! Ported from mem0 `tests/test_chatty_llm_parsing.py` (faithful).

use mem0_core::text::{extract_json, remove_code_blocks};
use serde_json::Value;

fn parse(text: &str) -> Value {
    serde_json::from_str(text).expect("valid json")
}

// --- extract_json ---

#[test]
fn pure_json() {
    let text = r#"{"memory": [{"id": "0", "text": "likes basketball", "event": "ADD"}]}"#;
    let parsed = parse(&extract_json(text));
    assert_eq!(parsed["memory"][0]["text"], "likes basketball");
}

#[test]
fn json_in_markdown_code_block() {
    let text = "```json\n{\"memory\": [{\"id\": \"0\", \"text\": \"likes basketball\"}]}\n```";
    let parsed = parse(&extract_json(text));
    assert_eq!(parsed["memory"][0]["text"], "likes basketball");
}

#[test]
fn json_in_plain_code_block() {
    let text = "```\n{\"memory\": [{\"id\": \"0\", \"text\": \"likes basketball\"}]}\n```";
    let parsed = parse(&extract_json(text));
    assert_eq!(parsed["memory"][0]["text"], "likes basketball");
}

#[test]
fn chatty_with_markdown() {
    let text = "Here is the extracted memory:\n```json\n{\"memory\": [{\"id\": \"0\", \"text\": \"likes basketball\"}]}\n```\nI hope this helps!";
    let parsed = parse(&extract_json(text));
    assert_eq!(parsed["memory"][0]["text"], "likes basketball");
}

#[test]
fn chatty_without_markdown() {
    let text = "Here is the memory update you requested:\n{\"memory\": [{\"id\": \"0\", \"text\": \"likes gaming\"}]}\nThat's the result.";
    let parsed = parse(&extract_json(text));
    assert_eq!(parsed["memory"][0]["text"], "likes gaming");
}

#[test]
fn chatty_multiline_json_without_markdown() {
    let text = "Here is the memory update:\n{\n  \"memory\": [\n    {\n      \"id\": \"0\",\n      \"text\": \"User likes gaming\"\n    }\n  ]\n}\n";
    let parsed = parse(&extract_json(text));
    assert_eq!(parsed["memory"][0]["text"], "User likes gaming");
}

#[test]
fn no_json_at_all() {
    let text = "I don't have any memory updates.";
    assert_eq!(extract_json(text), text);
}

#[test]
fn whitespace_padding() {
    let text = "  \n  {\"memory\": []}  \n  ";
    let parsed = parse(&extract_json(text));
    assert_eq!(parsed, serde_json::json!({ "memory": [] }));
}

#[test]
fn nested_json_objects() {
    let text = r#"Sure! {"memory": [{"id": "0", "text": "test", "event": "ADD"}]} Done."#;
    let parsed = parse(&extract_json(text));
    assert_eq!(parsed["memory"][0]["id"], "0");
}

// --- remove_code_blocks ---

#[test]
fn clean_code_block() {
    let text = "```json\n{\"memory\": []}\n```";
    let parsed = parse(&remove_code_blocks(text));
    assert_eq!(parsed, serde_json::json!({ "memory": [] }));
}

#[test]
fn no_code_block_not_json() {
    let text = "Here is the result: {\"memory\": []}";
    let result = remove_code_blocks(text);
    assert!(serde_json::from_str::<Value>(&result).is_err());
}

#[test]
fn think_tags_removed_inside_code_block() {
    let text = "```json\n<think>reasoning here</think>\n{\"memory\": []}\n```";
    let parsed = parse(&remove_code_blocks(text));
    assert_eq!(parsed, serde_json::json!({ "memory": [] }));
}

#[test]
fn think_tags_before_code_block_not_handled() {
    let text = "<think>reasoning here</think>\n```json\n{\"memory\": []}\n```";
    let result = remove_code_blocks(text);
    assert!(serde_json::from_str::<Value>(&result).is_err());
}

// --- full fallback chain (remove_code_blocks -> extract_json) ---

fn parse_with_fallback(response: &str) -> Option<Value> {
    let cleaned = remove_code_blocks(response);
    if let Ok(v) = serde_json::from_str::<Value>(&cleaned) {
        return Some(v);
    }
    serde_json::from_str::<Value>(&extract_json(response)).ok()
}

#[test]
fn fallback_clean_json() {
    let r = r#"{"memory": [{"id": "0", "text": "Name is Alex", "event": "ADD"}]}"#;
    assert_eq!(
        parse_with_fallback(r).unwrap()["memory"][0]["text"],
        "Name is Alex"
    );
}

#[test]
fn fallback_markdown_wrapped() {
    let r = "```json\n{\"memory\": [{\"id\": \"0\", \"text\": \"Name is Alex\"}]}\n```";
    assert_eq!(
        parse_with_fallback(r).unwrap()["memory"][0]["text"],
        "Name is Alex"
    );
}

#[test]
fn fallback_chatty_markdown() {
    let r = "Here is the extracted memory:\n```json\n{\n  \"memory\": [\n    {\"id\": \"0\", \"text\": \"Name is Alex\"},\n    {\"id\": \"1\", \"text\": \"Love basketball\"},\n    {\"id\": \"2\", \"text\": \"Love gaming\"}\n  ]\n}\n```\nI hope this helps!";
    let v = parse_with_fallback(r).unwrap();
    assert_eq!(v["memory"].as_array().unwrap().len(), 3);
    assert_eq!(v["memory"][1]["text"], "Love basketball");
}

#[test]
fn fallback_think_tags_with_json() {
    let r = "<think>Let me process this...</think>\n```json\n{\"memory\": [{\"id\": \"0\", \"text\": \"test\"}]}\n```";
    assert_eq!(parse_with_fallback(r).unwrap()["memory"][0]["text"], "test");
}

#[test]
fn fallback_completely_invalid() {
    assert!(parse_with_fallback("I don't understand the question").is_none());
}

#[test]
fn fallback_facts_extraction() {
    let r = "Sure! Here are the facts:\n{\"facts\": [\"Name is Alex\", \"Loves basketball\"]}\nHope that helps!";
    let v = parse_with_fallback(r).unwrap();
    assert_eq!(v["facts"][0], "Name is Alex");
    assert_eq!(v["facts"][1], "Loves basketball");
}
