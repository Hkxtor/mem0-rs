//! Deterministic mock providers and an in-memory vector store.
//!
//! These back the core unit tests and provide a dependency-free default vector
//! store (the `embedded` provider is implemented on top of this in the
//! `mem0-vector` crate; here it is used directly for testing).

use crate::error::Result;
use crate::filters::matches_filters;
use crate::nlp::lemmatize_for_bm25;
use crate::scoring::{bm25_scores, corpus_from_hits};
use crate::traits::{Embedder, GenerateOptions, Llm, MemoryAction, VectorStore};
use crate::types::{JsonMap, Message, SearchHit, VectorRecord};
use async_trait::async_trait;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Mutex;

/// Deterministic hash-based embedder: similar text → similar vectors.
pub struct MockEmbedder {
    dims: usize,
}

impl MockEmbedder {
    /// Construct a mock embedder with `dims` dimensions.
    pub fn new(dims: usize) -> Self {
        Self { dims }
    }

    fn hash_embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dims];
        for tok in text
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
        {
            // FNV-1a hash → bucket.
            let mut h: u64 = 0xcbf29ce484222325;
            for b in tok.bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            let idx = (h % self.dims as u64) as usize;
            v[idx] += 1.0;
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }
}

#[async_trait]
impl Embedder for MockEmbedder {
    async fn embed(&self, text: &str, _action: MemoryAction) -> Result<Vec<f32>> {
        Ok(self.hash_embed(text))
    }
    fn dims(&self) -> usize {
        self.dims
    }
}

/// Mock LLM returning queued responses (or a default when the queue is empty).
pub struct MockLlm {
    responses: Mutex<VecDeque<String>>,
    default: String,
}

impl MockLlm {
    /// Construct a mock LLM that always returns `default`.
    pub fn new(default: impl Into<String>) -> Self {
        Self {
            responses: Mutex::new(VecDeque::new()),
            default: default.into(),
        }
    }

    /// Construct a mock LLM that returns the given responses in order, then the
    /// empty-additive default.
    pub fn with_responses(responses: Vec<String>) -> Self {
        Self {
            responses: Mutex::new(responses.into_iter().collect()),
            default: "{\"memory\": []}".to_string(),
        }
    }
}

#[async_trait]
impl Llm for MockLlm {
    async fn generate(&self, _messages: &[Message], _options: &GenerateOptions) -> Result<String> {
        let mut q = self.responses.lock().unwrap();
        Ok(q.pop_front().unwrap_or_else(|| self.default.clone()))
    }
}

/// In-memory vector store backed by a `HashMap`, with cosine similarity search
/// and BM25 keyword search over the `text_lemmatized` payload field.
#[derive(Default)]
pub struct InMemoryVectorStore {
    inner: Mutex<HashMap<String, VectorRecord>>,
}

impl InMemoryVectorStore {
    /// Construct an empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn insert(&self, records: Vec<VectorRecord>) -> Result<()> {
        let mut map = self.inner.lock().unwrap();
        for r in records {
            map.insert(r.id.clone(), r);
        }
        Ok(())
    }

    async fn search(
        &self,
        _query: &str,
        vector: &[f32],
        top_k: usize,
        filters: &JsonMap,
    ) -> Result<Vec<SearchHit>> {
        let map = self.inner.lock().unwrap();
        let mut hits: Vec<SearchHit> = map
            .values()
            .filter(|r| matches_filters(&r.payload, filters))
            .map(|r| SearchHit {
                id: r.id.clone(),
                score: cosine(vector, &r.vector).max(0.0),
                payload: r.payload.clone(),
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(top_k);
        Ok(hits)
    }

    async fn get(&self, id: &str) -> Result<Option<SearchHit>> {
        let map = self.inner.lock().unwrap();
        Ok(map.get(id).map(|r| SearchHit {
            id: r.id.clone(),
            score: 0.0,
            payload: r.payload.clone(),
        }))
    }

    async fn update(
        &self,
        id: &str,
        vector: Option<Vec<f32>>,
        payload: Option<JsonMap>,
    ) -> Result<()> {
        let mut map = self.inner.lock().unwrap();
        if let Some(rec) = map.get_mut(id) {
            if let Some(v) = vector {
                rec.vector = v;
            }
            if let Some(p) = payload {
                rec.payload = p;
            }
        }
        Ok(())
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let mut map = self.inner.lock().unwrap();
        map.remove(id);
        Ok(())
    }

    async fn list(&self, filters: &JsonMap, limit: Option<usize>) -> Result<Vec<SearchHit>> {
        let map = self.inner.lock().unwrap();
        let mut hits: Vec<SearchHit> = map
            .values()
            .filter(|r| matches_filters(&r.payload, filters))
            .map(|r| SearchHit {
                id: r.id.clone(),
                score: 0.0,
                payload: r.payload.clone(),
            })
            .collect();
        hits.sort_by(|a, b| {
            let ca = a
                .payload
                .get("created_at")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let cb = b
                .payload
                .get("created_at")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            cb.cmp(ca)
        });
        if let Some(n) = limit {
            hits.truncate(n);
        }
        Ok(hits)
    }

    async fn delete_col(&self) -> Result<()> {
        self.inner.lock().unwrap().clear();
        Ok(())
    }

    async fn reset(&self) -> Result<()> {
        self.inner.lock().unwrap().clear();
        Ok(())
    }

    async fn keyword_search(
        &self,
        query: &str,
        top_k: usize,
        filters: &JsonMap,
    ) -> Result<Option<Vec<SearchHit>>> {
        let filtered: Vec<SearchHit> = {
            let map = self.inner.lock().unwrap();
            map.values()
                .filter(|r| matches_filters(&r.payload, filters))
                .map(|r| SearchHit {
                    id: r.id.clone(),
                    score: 0.0,
                    payload: r.payload.clone(),
                })
                .collect()
        };

        let query_lemma = lemmatize_for_bm25(query);
        let corpus = corpus_from_hits(&filtered);
        let scores = bm25_scores(&query_lemma, &corpus);

        let mut hits: Vec<SearchHit> = filtered
            .into_iter()
            .filter_map(|mut h| {
                let s = scores.get(&h.id).copied().unwrap_or(0.0);
                if s > 0.0 {
                    h.score = s as f32;
                    Some(h)
                } else {
                    None
                }
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(top_k);
        Ok(Some(hits))
    }
}
