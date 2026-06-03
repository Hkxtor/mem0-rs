//! Text utilities ported from `mem0/mem0/memory/utils.py`.

use crate::types::Message;
use regex::Regex;
use serde_json::Value;

/// Remove enclosing ```` ```lang ... ``` ```` fences and `<think>...</think>` blocks.
///
/// Port of `remove_code_blocks`.
pub fn remove_code_blocks(content: &str) -> String {
    let trimmed = content.trim();
    // (?s) = DOTALL so `.` matches newlines.
    let fence = Regex::new(r"(?s)^```[a-zA-Z0-9]*\n(.*?)\n```$").unwrap();
    let inner = if let Some(caps) = fence.captures(trimmed) {
        caps.get(1).map(|m| m.as_str().trim()).unwrap_or(trimmed)
    } else {
        trimmed
    };
    let think = Regex::new(r"(?s)<think>.*?</think>").unwrap();
    think.replace_all(inner, "").trim().to_string()
}

/// Extract a JSON substring from `text`, stripping code fences or falling back to
/// the first `{` .. last `}` slice. Port of `extract_json`.
pub fn extract_json(text: &str) -> String {
    let text = text.trim();
    let fenced = Regex::new(r"(?s)```(?:json)?\s*(.*?)\s*```").unwrap();
    if let Some(caps) = fenced.captures(text) {
        if let Some(m) = caps.get(1) {
            return m.as_str().to_string();
        }
    }
    let start = text.find('{');
    let end = text.rfind('}');
    if let (Some(s), Some(e)) = (start, end) {
        if e > s {
            return text[s..=e].to_string();
        }
    }
    text.to_string()
}

/// Flatten messages into a `role: content` transcript. Port of `parse_messages`.
pub fn parse_messages(messages: &[Message]) -> String {
    let mut response = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => response.push_str(&format!("system: {}\n", msg.content)),
            "user" => response.push_str(&format!("user: {}\n", msg.content)),
            "assistant" => response.push_str(&format!("assistant: {}\n", msg.content)),
            _ => {}
        }
    }
    response
}

/// Normalize raw LLM-extracted facts (strings or `{fact|text: ...}` objects) into
/// a list of strings. Port of `normalize_facts`.
pub fn normalize_facts(raw_facts: &[Value]) -> Vec<String> {
    let mut normalized = Vec::new();
    for item in raw_facts {
        let fact = match item {
            Value::String(s) => Some(s.clone()),
            Value::Object(map) => map
                .get("fact")
                .or_else(|| map.get("text"))
                .and_then(|v| v.as_str().map(|s| s.to_string())),
            other => Some(other.to_string()),
        };
        if let Some(f) = fact {
            if !f.is_empty() {
                normalized.push(f);
            }
        }
    }
    normalized
}
