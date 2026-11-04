//! Configuration model ported from `mem0/mem0/configs/base.py` and the
//! per-component config modules.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::Result;

/// Default mem0 home directory (`$MEM0_DIR` or `~/.mem0`).
pub fn mem0_dir() -> String {
    if let Ok(dir) = std::env::var("MEM0_DIR") {
        return dir;
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    format!("{home}/.mem0")
}

fn default_history_db_path() -> String {
    format!("{}/history.db", mem0_dir())
}

/// Embedder configuration: `{ provider, config }`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbedderConfig {
    /// Provider name (e.g. `openai`, `ollama`, `mock`).
    pub provider: String,
    /// Provider-specific configuration object.
    pub config: Value,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            provider: "openai".to_string(),
            config: json!({}),
        }
    }
}

/// LLM configuration: `{ provider, config }`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    /// Provider name (e.g. `openai`, `anthropic`, `ollama`, `mock`).
    pub provider: String,
    /// Provider-specific configuration object.
    pub config: Value,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: "openai".to_string(),
            config: json!({}),
        }
    }
}

/// Vector store configuration: `{ provider, config }`.
///
/// Deviation from Python (which defaults to `qdrant`): the Rust port defaults to
/// the in-process `embedded` store so a default `Memory` is runnable with no
/// external services.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VectorStoreConfig {
    /// Provider name (e.g. `embedded`, `qdrant`, `pgvector`).
    pub provider: String,
    /// Provider-specific configuration object.
    pub config: Value,
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        Self {
            provider: "embedded".to_string(),
            config: json!({}),
        }
    }
}

/// Reranker configuration: `{ provider, config }`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RerankerConfig {
    /// Provider name (e.g. `llm`, `cohere`, `zero_entropy`).
    pub provider: String,
    /// Provider-specific configuration object.
    pub config: Value,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            provider: "llm".to_string(),
            config: json!({}),
        }
    }
}

/// Top-level memory configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    /// Vector store configuration.
    pub vector_store: VectorStoreConfig,
    /// LLM configuration.
    pub llm: LlmConfig,
    /// Embedder configuration.
    pub embedder: EmbedderConfig,
    /// Optional reranker configuration.
    pub reranker: Option<RerankerConfig>,
    /// Path to the SQLite history database.
    pub history_db_path: String,
    /// API version string (default `v1.1`).
    pub version: String,
    /// Optional custom fact-extraction instructions.
    pub custom_instructions: Option<String>,
    /// When true, the extraction prompt carries the same-language requirement block.
    /// Python mem0 defaults this to False but exposes it; the desktop product needs True.
    pub use_input_language: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            vector_store: VectorStoreConfig::default(),
            llm: LlmConfig::default(),
            embedder: EmbedderConfig::default(),
            reranker: None,
            history_db_path: default_history_db_path(),
            version: "v1.1".to_string(),
            custom_instructions: None,
            use_input_language: false,
        }
    }
}

impl MemoryConfig {
    /// Parse a [`MemoryConfig`] from a JSON string.
    pub fn from_json(s: &str) -> Result<Self> {
        Ok(serde_json::from_str(s)?)
    }
}
