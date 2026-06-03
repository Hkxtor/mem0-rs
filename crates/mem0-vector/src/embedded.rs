//! Embedded in-process vector store with optional JSON-file persistence.
//!
//! Provides cosine semantic search, operator-aware metadata filtering, and
//! native BM25 keyword search over the `text_lemmatized` payload field. With no
//! `path` configured it is purely in-memory; with a `path` it persists the
//! collection to `{path}/{collection}.json` after each mutation.

use async_trait::async_trait;
use mem0_core::error::{Mem0Error, Result};
use mem0_core::filters::matches_filters;
use mem0_core::scoring::{bm25_scores, corpus_from_hits};
use mem0_core::traits::VectorStore;
use mem0_core::types::{JsonMap, SearchHit, VectorRecord};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

#[derive(Serialize, Deserialize)]
struct PersistRecord {
    id: String,
    vector: Vec<f32>,
    payload: JsonMap,
}

/// In-process vector store, optionally persisted to a JSON file.
pub struct EmbeddedVectorStore {
    file: Option<PathBuf>,
    inner: Mutex<HashMap<String, VectorRecord>>,
}

impl EmbeddedVectorStore {
    /// Construct a store for `collection_name`. If `path` is given, the store is
    /// loaded from (and persisted to) `{path}/{collection_name}.json`.
    pub fn new(collection_name: &str, path: Option<String>) -> Result<Self> {
        let file = path.map(|p| PathBuf::from(p).join(format!("{collection_name}.json")));
        let map = match &file {
            Some(f) if f.exists() => load(f)?,
            _ => HashMap::new(),
        };
        Ok(Self {
            file,
            inner: Mutex::new(map),
        })
    }

    fn persist(&self, map: &HashMap<String, VectorRecord>) {
        let Some(f) = &self.file else { return };
        if let Some(parent) = f.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let recs: Vec<PersistRecord> = map
            .values()
            .map(|r| PersistRecord {
                id: r.id.clone(),
                vector: r.vector.clone(),
                payload: r.payload.clone(),
            })
            .collect();
        match serde_json::to_string(&recs) {
            Ok(json) => {
                if let Err(e) = std::fs::write(f, json) {
                    tracing::warn!("Failed to persist embedded store to {f:?}: {e}");
                }
            }
            Err(e) => tracing::warn!("Failed to serialize embedded store: {e}"),
        }
    }
}

fn load(f: &PathBuf) -> Result<HashMap<String, VectorRecord>> {
    let data = std::fs::read_to_string(f)
        .map_err(|e| Mem0Error::vector_store(format!("failed to read {f:?}: {e}")))?;
    let recs: Vec<PersistRecord> = serde_json::from_str(&data)?;
    Ok(recs
        .into_iter()
        .map(|r| {
            (
                r.id.clone(),
                VectorRecord {
                    id: r.id,
                    vector: r.vector,
                    payload: r.payload,
                },
            )
        })
        .collect())
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
impl VectorStore for EmbeddedVectorStore {
    async fn insert(&self, records: Vec<VectorRecord>) -> Result<()> {
        let mut map = self.inner.lock().unwrap();
        for r in records {
            map.insert(r.id.clone(), r);
        }
        self.persist(&map);
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
        let mut scored: Vec<(&VectorRecord, f32)> = map
            .values()
            .filter(|r| matches_filters(&r.payload, filters))
            .map(|r| (r, cosine(vector, &r.vector).max(0.0)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        Ok(scored
            .into_iter()
            .map(|(r, score)| SearchHit {
                id: r.id.clone(),
                score,
                payload: r.payload.clone(),
            })
            .collect())
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
        self.persist(&map);
        Ok(())
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let mut map = self.inner.lock().unwrap();
        map.remove(id);
        self.persist(&map);
        Ok(())
    }

    async fn list(&self, filters: &JsonMap, limit: Option<usize>) -> Result<Vec<SearchHit>> {
        let map = self.inner.lock().unwrap();
        let mut refs: Vec<&VectorRecord> = map
            .values()
            .filter(|r| matches_filters(&r.payload, filters))
            .collect();
        refs.sort_by(|a, b| {
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
            refs.truncate(n);
        }
        Ok(refs
            .into_iter()
            .map(|r| SearchHit {
                id: r.id.clone(),
                score: 0.0,
                payload: r.payload.clone(),
            })
            .collect())
    }

    async fn delete_col(&self) -> Result<()> {
        let mut map = self.inner.lock().unwrap();
        map.clear();
        self.persist(&map);
        Ok(())
    }

    async fn reset(&self) -> Result<()> {
        let mut map = self.inner.lock().unwrap();
        map.clear();
        self.persist(&map);
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

        let corpus = corpus_from_hits(&filtered);
        let scores = bm25_scores(query, &corpus);
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
