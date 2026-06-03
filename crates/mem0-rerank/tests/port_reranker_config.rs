//! Ported from mem0 `tests/rerankers/test_llm_reranker_config.py` and
//! `test_llm_reranker_nested_config.py`.
//!
//! Adapted: the Python tests mock `LlmFactory.create` and assert the
//! `(provider, config)` it is called with. The Rust equivalent is the pure
//! `resolve_inner_llm`, which we assert directly.

use mem0_rerank::config::RerankerSettings;
use mem0_rerank::llm::resolve_inner_llm;
use serde_json::{json, Value};

fn resolve(cfg: Value) -> (String, Value) {
    let settings = RerankerSettings::from_value(&cfg).unwrap();
    resolve_inner_llm(&settings).unwrap()
}

#[test]
fn default_config_defaults() {
    let (provider, config) = resolve(json!({ "provider": "openai" }));
    assert_eq!(provider, "openai");
    assert_eq!(config["model"], "gpt-4o-mini");
    assert_eq!(config["temperature"], 0.0);
    assert_eq!(config["max_tokens"], 100);
}

#[test]
fn dict_config_passes_through() {
    let (provider, config) =
        resolve(json!({ "provider": "openai", "model": "gpt-4o", "api_key": "sk-test" }));
    assert_eq!(provider, "openai");
    assert_eq!(config["model"], "gpt-4o");
    assert_eq!(config["temperature"], 0.0);
    assert_eq!(config["max_tokens"], 100);
    assert_eq!(config["api_key"], "sk-test");
}

#[test]
fn without_api_key_omits_it() {
    let (_provider, config) = resolve(json!({ "provider": "openai", "model": "gpt-4o-mini" }));
    assert!(config.get("api_key").is_none());
}

#[test]
fn nested_llm_overrides_provider() {
    let (provider, _config) = resolve(json!({
        "provider": "openai",
        "model": "gpt-4o-mini",
        "llm": { "provider": "ollama", "config": { "model": "llama3", "ollama_base_url": "http://localhost:11434" } }
    }));
    assert_eq!(provider, "ollama");
}

#[test]
fn nested_llm_passes_provider_specific_config() {
    let (_provider, config) = resolve(json!({
        "provider": "openai",
        "llm": { "provider": "ollama", "config": { "model": "llama3", "ollama_base_url": "http://localhost:11434" } }
    }));
    assert_eq!(config["ollama_base_url"], "http://localhost:11434");
    assert_eq!(config["model"], "llama3");
}

#[test]
fn nested_llm_inherits_top_level_defaults() {
    let (_provider, config) = resolve(json!({
        "provider": "openai",
        "temperature": 0.0,
        "max_tokens": 100,
        "llm": { "provider": "ollama", "config": { "model": "llama3" } }
    }));
    assert_eq!(config["temperature"], 0.0);
    assert_eq!(config["max_tokens"], 100);
}

#[test]
fn nested_llm_config_values_take_precedence() {
    let (_provider, config) = resolve(json!({
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "max_tokens": 100,
        "llm": { "provider": "ollama", "config": { "model": "custom-model", "temperature": 0.5, "max_tokens": 200 } }
    }));
    assert_eq!(config["model"], "custom-model");
    assert_eq!(config["temperature"], 0.5);
    assert_eq!(config["max_tokens"], 200);
}

#[test]
fn nested_llm_falls_back_to_top_level_provider() {
    let (provider, config) = resolve(json!({
        "provider": "anthropic",
        "model": "claude-3-haiku",
        "llm": { "config": { "model": "claude-3-sonnet" } }
    }));
    assert_eq!(provider, "anthropic");
    assert_eq!(config["model"], "claude-3-sonnet");
}

#[test]
fn nested_llm_with_empty_config() {
    let (provider, config) = resolve(json!({
        "provider": "openai",
        "model": "gpt-4o-mini",
        "llm": { "provider": "ollama" }
    }));
    assert_eq!(provider, "ollama");
    assert_eq!(config["model"], "gpt-4o-mini");
    assert_eq!(config["temperature"], 0.0);
    assert_eq!(config["max_tokens"], 100);
}

#[test]
fn nested_llm_with_none_config() {
    let (provider, config) = resolve(json!({
        "provider": "openai",
        "model": "gpt-4o-mini",
        "llm": { "provider": "ollama", "config": null }
    }));
    assert_eq!(provider, "ollama");
    assert_eq!(config["model"], "gpt-4o-mini");
}

#[test]
fn nested_llm_inherits_top_level_api_key() {
    let (_provider, config) = resolve(json!({
        "provider": "openai",
        "api_key": "sk-top-level",
        "llm": { "provider": "openai", "config": { "model": "gpt-4o" } }
    }));
    assert_eq!(config["api_key"], "sk-top-level");
}

#[test]
fn nested_llm_config_api_key_not_overridden() {
    let (_provider, config) = resolve(json!({
        "provider": "openai",
        "api_key": "sk-top-level",
        "llm": { "provider": "openai", "config": { "model": "gpt-4o", "api_key": "sk-nested" } }
    }));
    assert_eq!(config["api_key"], "sk-nested");
}
