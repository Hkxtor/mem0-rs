//! mem0 — long-term memory for AI agents, in Rust.
//!
//! This facade crate wires the provider factories ([`mem0_embed`],
//! [`mem0_llm`], [`mem0_vector`], [`mem0_rerank`]) into the [`mem0_core::Memory`]
//! orchestrator. Construct a fully-configured memory from JSON with
//! [`from_json`], or from a [`MemoryConfig`] with [`from_config`]. A synchronous
//! [`blocking::Memory`] wrapper is provided for non-async callers.
//!
//! # Example
//! ```no_run
//! # async fn run() -> mem0::Result<()> {
//! let mem = mem0::from_json(r#"{
//!     "embedder": { "provider": "openai", "config": { "api_key": "sk-..." } },
//!     "llm": { "provider": "openai", "config": { "api_key": "sk-..." } },
//!     "vector_store": { "provider": "embedded", "config": { "path": "/tmp/mem0" } }
//! }"#)?;
//! let added = mem
//!     .add("I love hiking", mem0::AddOptions { user_id: Some("alice".into()), ..Default::default() })
//!     .await?;
//! # Ok(()) }
//! ```

pub use mem0_core::config::LlmConfig;
pub use mem0_core::history::HistoryRecord;
pub use mem0_core::{
    AddOptions, AddResult, Embedder, EmbedderConfig, Event, GenerateOptions, JsonMap, Llm,
    Mem0Error, Memory, MemoryAction, MemoryConfig, MemoryType, Message, MessagesInput, Reranker,
    RerankerConfig, Result, SearchHit, SearchOptions, VectorRecord, VectorStore, VectorStoreConfig,
};

pub use mem0_embed::build_embedder;
pub use mem0_llm::build_llm;
pub use mem0_rerank::build_reranker;
pub use mem0_vector::build_vector_store;

use serde_json::{json, Value};

/// Build a fully-wired [`Memory`] from a [`MemoryConfig`].
///
/// The embedder is constructed first and its dimensionality is injected into the
/// vector store config (so fixed-dimension backends like qdrant/pgvector match).
/// A best-effort entity store (a sibling collection) is attached to enable entity
/// linking and entity boosts.
pub fn from_config(mut config: MemoryConfig) -> Result<Memory> {
    let embedder = build_embedder(&config.embedder.provider, &config.embedder.config)?;
    let dims = embedder.dims();
    inject_dims(&mut config.vector_store.config, dims);

    let llm = build_llm(&config.llm.provider, &config.llm.config)?;
    let vector = build_vector_store(&config.vector_store.provider, &config.vector_store.config)?;
    let reranker = match &config.reranker {
        Some(r) => Some(build_reranker(&r.provider, &r.config)?),
        None => None,
    };
    let entity_store = build_entity_store(&config);

    let memory = Memory::new(config, embedder, llm, vector, reranker)?;
    Ok(match entity_store {
        Some(es) => memory.with_entity_store(es),
        None => memory,
    })
}

/// Build a fully-wired [`Memory`] from a JSON config string.
pub fn from_json(s: &str) -> Result<Memory> {
    from_config(MemoryConfig::from_json(s)?)
}

fn inject_dims(config: &mut Value, dims: usize) {
    // The embedder's dimensionality is the single source of truth and the vector
    // store follows it (Python mem0's _sync_embedding_dims overwrites, not or_insert):
    // an update that only changes the LLM must not reset the column width, but an
    // update that changes the embedder model must win over a stale stored value.
    match config {
        Value::Object(map) => {
            map.insert("embedding_model_dims".to_string(), json!(dims));
        }
        Value::Null => {
            *config = json!({ "embedding_model_dims": dims });
        }
        _ => {}
    }
}

fn build_entity_store(config: &MemoryConfig) -> Option<Box<dyn VectorStore>> {
    let base = config
        .vector_store
        .config
        .get("collection_name")
        .and_then(|v| v.as_str())
        .unwrap_or("mem0")
        .to_string();
    let mut entity_cfg = config.vector_store.config.clone();
    let entity_collection = json!(format!("{base}_entities"));
    match &mut entity_cfg {
        Value::Object(map) => {
            map.insert("collection_name".into(), entity_collection);
        }
        _ => {
            entity_cfg = json!({ "collection_name": format!("{base}_entities") });
        }
    }
    build_vector_store(&config.vector_store.provider, &entity_cfg).ok()
}

/// Synchronous (blocking) wrapper around the async [`Memory`] API.
pub mod blocking {
    use super::*;
    use tokio::runtime::Runtime;

    /// A blocking handle around [`mem0_core::Memory`] backed by its own Tokio runtime.
    pub struct Memory {
        rt: Runtime,
        inner: super::Memory,
    }

    impl Memory {
        /// Build a blocking memory from a [`MemoryConfig`].
        pub fn from_config(config: MemoryConfig) -> Result<Self> {
            let rt = Runtime::new().map_err(|e| {
                Mem0Error::configuration(format!("failed to create Tokio runtime: {e}"))
            })?;
            let inner = super::from_config(config)?;
            Ok(Self { rt, inner })
        }

        /// Build a blocking memory from a JSON config string.
        pub fn from_json(s: &str) -> Result<Self> {
            Self::from_config(MemoryConfig::from_json(s)?)
        }

        /// Add memories. See [`mem0_core::Memory::add`].
        pub fn add(
            &self,
            messages: impl Into<MessagesInput>,
            opts: AddOptions,
        ) -> Result<Vec<AddResult>> {
            self.rt.block_on(self.inner.add(messages, opts))
        }

        /// Retrieve a memory by id.
        pub fn get(&self, memory_id: &str) -> Result<Option<Value>> {
            self.rt.block_on(self.inner.get(memory_id))
        }

        /// List memories matching filters.
        pub fn get_all(&self, filters: &JsonMap, top_k: usize) -> Result<Value> {
            self.rt.block_on(self.inner.get_all(filters, top_k))
        }

        /// Search memories.
        pub fn search(
            &self,
            query: &str,
            filters: &JsonMap,
            options: SearchOptions,
        ) -> Result<Value> {
            self.rt.block_on(self.inner.search(query, filters, options))
        }

        /// Update a memory's content.
        pub fn update(
            &self,
            memory_id: &str,
            data: &str,
            metadata: Option<JsonMap>,
        ) -> Result<Value> {
            self.rt
                .block_on(self.inner.update(memory_id, data, metadata))
        }

        /// Delete a memory by id.
        pub fn delete(&self, memory_id: &str) -> Result<Value> {
            self.rt.block_on(self.inner.delete(memory_id))
        }

        /// Delete all memories in a scope.
        pub fn delete_all(
            &self,
            user_id: Option<&str>,
            agent_id: Option<&str>,
            run_id: Option<&str>,
        ) -> Result<Value> {
            self.rt
                .block_on(self.inner.delete_all(user_id, agent_id, run_id))
        }

        /// Return the change history of a memory.
        pub fn history(&self, memory_id: &str) -> Result<Vec<HistoryRecord>> {
            self.rt.block_on(self.inner.history(memory_id))
        }

        /// Reset all memories.
        pub fn reset(&self) -> Result<()> {
            self.rt.block_on(self.inner.reset())
        }
    }
}

#[cfg(test)]
mod inject_dims_tests {
    use super::*;

    #[test]
    fn embedder_dims_overwrite_stale_stored_value() {
        let mut cfg = json!({"embedding_model_dims": 1536});
        inject_dims(&mut cfg, 768);
        assert_eq!(cfg["embedding_model_dims"], json!(768));
    }

    #[test]
    fn embedder_dims_fill_missing_value() {
        let mut cfg = json!({"collection_name": "mem0"});
        inject_dims(&mut cfg, 768);
        assert_eq!(cfg["embedding_model_dims"], json!(768));
        assert_eq!(cfg["collection_name"], json!("mem0"));
    }
}
