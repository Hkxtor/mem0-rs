//! Provider traits. Concrete implementations live in sibling crates and are
//! injected into [`crate::Memory`]. Ported from the Python base classes.

use crate::error::Result;
use crate::types::{JsonMap, Message, SearchHit, VectorRecord};
use async_trait::async_trait;
use serde_json::Value;

/// The memory action context for embeddings (`add` | `search` | `update`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryAction {
    /// Embedding for storage.
    Add,
    /// Embedding for a query.
    Search,
    /// Embedding for an update.
    Update,
}

impl MemoryAction {
    /// The wire string value.
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryAction::Add => "add",
            MemoryAction::Search => "search",
            MemoryAction::Update => "update",
        }
    }
}

/// Text → vector embedder. Port of `EmbeddingBase`.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Embed a single text.
    async fn embed(&self, text: &str, action: MemoryAction) -> Result<Vec<f32>>;

    /// Embed many texts. Default loops over [`Embedder::embed`].
    async fn embed_batch(&self, texts: &[String], action: MemoryAction) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            out.push(self.embed(t, action).await?);
        }
        Ok(out)
    }

    /// The embedding dimensionality.
    fn dims(&self) -> usize;
}

/// Options for an LLM generation call.
#[derive(Debug, Clone, Default)]
pub struct GenerateOptions {
    /// Request a JSON object response (`response_format={"type":"json_object"}`).
    pub response_format_json: bool,
    /// Sampling temperature override.
    pub temperature: Option<f32>,
    /// Max tokens override.
    pub max_tokens: Option<u32>,
    /// Nucleus sampling override.
    pub top_p: Option<f32>,
    /// Optional tool definitions (passthrough).
    pub tools: Option<Value>,
}

/// Chat-completion LLM. Port of `LLMBase`.
#[async_trait]
pub trait Llm: Send + Sync {
    /// Generate a response for the given messages.
    async fn generate(&self, messages: &[Message], options: &GenerateOptions) -> Result<String>;
}

/// Vector store. Port of `VectorStoreBase`.
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Insert records.
    async fn insert(&self, records: Vec<VectorRecord>) -> Result<()>;

    /// Semantic search with optional metadata filters.
    async fn search(
        &self,
        query: &str,
        vector: &[f32],
        top_k: usize,
        filters: &JsonMap,
    ) -> Result<Vec<SearchHit>>;

    /// Retrieve a single record by id.
    async fn get(&self, id: &str) -> Result<Option<SearchHit>>;

    /// Update a record's vector and/or payload.
    async fn update(
        &self,
        id: &str,
        vector: Option<Vec<f32>>,
        payload: Option<JsonMap>,
    ) -> Result<()>;

    /// Delete a record by id.
    async fn delete(&self, id: &str) -> Result<()>;

    /// List records matching filters (no query vector).
    async fn list(&self, filters: &JsonMap, limit: Option<usize>) -> Result<Vec<SearchHit>>;

    /// Count records by their `categories` payload array, and count the ones with none.
    ///
    /// The dashboard's category histogram needs aggregates, not rows: listing the collection to
    /// count in the caller reads every vector to produce a dozen numbers. Backends that can group
    /// in the store should override this; the default keeps every other backend correct by
    /// falling back to a listing, with the same expiration rule `list_all` applies.
    ///
    /// Returns `(counts_by_category, unclassified)`.
    async fn category_counts(&self) -> Result<(std::collections::BTreeMap<String, u64>, u64)> {
        let hits = self.list(&JsonMap::new(), Some(10_000)).await?;
        let mut counts = std::collections::BTreeMap::new();
        let mut unclassified = 0u64;
        for hit in hits {
            let cats = hit
                .payload
                .get("categories")
                .and_then(|c| c.as_array())
                .cloned()
                .unwrap_or_default();
            if cats.is_empty() {
                // Unclassified counts only what the user can see: an expired memory is hidden
                // from every list and search, so tagging it would spend tokens on invisible data.
                if !crate::util::payload_is_expired(&hit.payload) {
                    unclassified += 1;
                }
                continue;
            }
            for c in cats {
                if let Some(name) = c.as_str() {
                    *counts.entry(name.to_string()).or_insert(0) += 1;
                }
            }
        }
        Ok((counts, unclassified))
    }

    /// Delete the entire collection.
    async fn delete_col(&self) -> Result<()>;

    /// Reset: drop and recreate the collection.
    async fn reset(&self) -> Result<()>;

    /// Optional keyword/BM25 search. Returns `None` if unsupported.
    async fn keyword_search(
        &self,
        _query: &str,
        _top_k: usize,
        _filters: &JsonMap,
    ) -> Result<Option<Vec<SearchHit>>> {
        Ok(None)
    }

    /// Batch search. Default loops over [`VectorStore::search`].
    async fn search_batch(
        &self,
        queries: &[String],
        vectors: &[Vec<f32>],
        top_k: usize,
        filters: &JsonMap,
    ) -> Result<Vec<Vec<SearchHit>>> {
        let mut out = Vec::with_capacity(queries.len());
        for (q, v) in queries.iter().zip(vectors.iter()) {
            out.push(self.search(q, v, top_k, filters).await?);
        }
        Ok(out)
    }
}

/// Reranker. Port of the reranker base.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Rerank `documents` for `query`, returning `(original_index, score)` pairs.
    async fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_n: usize,
    ) -> Result<Vec<(usize, f32)>>;
}
