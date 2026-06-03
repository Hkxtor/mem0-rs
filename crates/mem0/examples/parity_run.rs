//! Rust side of the quality-parity harness.
//!
//! Executes `bench/parity_scenarios.json` through a [`Memory`] built from the
//! deterministic hash-based [`MockEmbedder`], a scripted [`MockLlm`], and a
//! *semantic-only* in-process store (BM25 disabled, no entity store) — matching
//! Python mem0 driven with the same deterministic embedder, an in-process stub
//! store, and spaCy absent (entity linking/boosts are no-ops on both sides).
//! Emits a JSON array of per-op results to stdout for `parity_compare.py`.

use async_trait::async_trait;
use mem0_core::testing::{InMemoryVectorStore, MockEmbedder, MockLlm};
use mem0_core::traits::VectorStore;
use mem0_core::types::{JsonMap, SearchHit, VectorRecord};
use mem0_core::Result as M0Result;
use mem0_core::{AddOptions, Memory, MemoryConfig, SearchOptions};
use serde_json::{json, Map, Value};

/// Wraps the in-process store but disables BM25 keyword search, so retrieval is
/// pure semantic (cosine) — directly comparable to the Python stub store.
struct SemanticOnly(InMemoryVectorStore);

#[async_trait]
impl VectorStore for SemanticOnly {
    async fn insert(&self, records: Vec<VectorRecord>) -> M0Result<()> {
        self.0.insert(records).await
    }
    async fn search(&self, q: &str, v: &[f32], k: usize, f: &JsonMap) -> M0Result<Vec<SearchHit>> {
        self.0.search(q, v, k, f).await
    }
    async fn get(&self, id: &str) -> M0Result<Option<SearchHit>> {
        self.0.get(id).await
    }
    async fn update(&self, id: &str, vec: Option<Vec<f32>>, pay: Option<JsonMap>) -> M0Result<()> {
        self.0.update(id, vec, pay).await
    }
    async fn delete(&self, id: &str) -> M0Result<()> {
        self.0.delete(id).await
    }
    async fn list(&self, f: &JsonMap, l: Option<usize>) -> M0Result<Vec<SearchHit>> {
        self.0.list(f, l).await
    }
    async fn delete_col(&self) -> M0Result<()> {
        self.0.delete_col().await
    }
    async fn reset(&self) -> M0Result<()> {
        self.0.reset().await
    }
    async fn keyword_search(
        &self,
        _q: &str,
        _k: usize,
        _f: &JsonMap,
    ) -> M0Result<Option<Vec<SearchHit>>> {
        Ok(None)
    }
}

fn round4(x: f64) -> f64 {
    (x * 10000.0).round() / 10000.0
}

#[tokio::main]
async fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "bench/parity_scenarios.json".to_string());
    let data = std::fs::read_to_string(&path).expect("read scenarios");
    let scenarios: Vec<Value> = serde_json::from_str(&data).expect("parse scenarios");

    // Build the scripted LLM queue from inferred adds, in execution order.
    let mut llm_queue: Vec<String> = Vec::new();
    for op in &scenarios {
        if op["op"] == "add" && op["infer"].as_bool().unwrap_or(false) {
            llm_queue.push(op["llm"].to_string());
        }
    }

    let config = MemoryConfig {
        history_db_path: ":memory:".to_string(),
        ..Default::default()
    };
    let mem = Memory::new(
        config,
        Box::new(MockEmbedder::new(32)),
        Box::new(MockLlm::with_responses(llm_queue)),
        Box::new(SemanticOnly(InMemoryVectorStore::new())),
        None,
    )
    .expect("build memory");

    // ids returned by each add op, in add order (for update/history/delete refs).
    let mut added_ids: Vec<Vec<String>> = Vec::new();
    let mut out: Vec<Value> = Vec::new();

    for op in &scenarios {
        match op["op"].as_str().unwrap() {
            "add" => {
                let user = op["user"].as_str().unwrap();
                let text = op["text"].as_str().unwrap();
                let infer = op["infer"].as_bool().unwrap_or(false);
                let meta = op.get("meta").and_then(|m| m.as_object()).cloned();
                let opts = AddOptions {
                    user_id: Some(user.to_string()),
                    infer: Some(infer),
                    metadata: meta,
                    ..Default::default()
                };
                let res = mem.add(text.to_string(), opts).await.expect("add");
                added_ids.push(res.iter().map(|r| r.id.clone()).collect());
                let mut texts: Vec<String> = res.iter().map(|r| r.memory.clone()).collect();
                texts.sort();
                out.push(json!({ "op": "add", "count": res.len(), "texts": texts }));
            }
            "search" => {
                let user = op["user"].as_str().unwrap();
                let query = op["query"].as_str().unwrap();
                let top_k = op["top_k"].as_u64().unwrap_or(5) as usize;
                let mut filters = Map::new();
                filters.insert("user_id".into(), json!(user));
                if let Some(f) = op.get("filter").and_then(|f| f.as_object()) {
                    for (k, v) in f {
                        filters.insert(k.clone(), v.clone());
                    }
                }
                let res = mem
                    .search(
                        query,
                        &filters,
                        SearchOptions {
                            top_k,
                            threshold: 0.1,
                            rerank: false,
                        },
                    )
                    .await
                    .expect("search");
                let arr = res["results"].as_array().cloned().unwrap_or_default();
                let texts: Vec<String> = arr
                    .iter()
                    .map(|m| m["memory"].as_str().unwrap_or("").to_string())
                    .collect();
                let scores: Vec<f64> = arr
                    .iter()
                    .map(|m| round4(m.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0)))
                    .collect();
                out.push(json!({ "op": "search", "texts": texts, "scores": scores }));
            }
            "get_all" => {
                let user = op["user"].as_str().unwrap();
                let mut filters = Map::new();
                filters.insert("user_id".into(), json!(user));
                let res = mem.get_all(&filters, 100).await.expect("get_all");
                let mut texts: Vec<String> = res["results"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|m| m["memory"].as_str().unwrap_or("").to_string())
                    .collect();
                texts.sort();
                out.push(json!({ "op": "get_all", "texts": texts }));
            }
            "update" => {
                let idx = op["add_index"].as_u64().unwrap() as usize;
                let id = added_ids[idx][0].clone();
                let text = op["text"].as_str().unwrap();
                mem.update(&id, text, None).await.expect("update");
                out.push(json!({ "op": "update", "ok": true }));
            }
            "history" => {
                let idx = op["add_index"].as_u64().unwrap() as usize;
                let id = added_ids[idx][0].clone();
                let hist = mem.history(&id).await.expect("history");
                let events: Vec<String> = hist.iter().map(|h| h.event.clone()).collect();
                out.push(json!({ "op": "history", "events": events }));
            }
            "delete" => {
                let idx = op["add_index"].as_u64().unwrap() as usize;
                let id = added_ids[idx][0].clone();
                mem.delete(&id).await.expect("delete");
                out.push(json!({ "op": "delete", "ok": true }));
            }
            other => {
                out.push(json!({ "op": other, "error": "unknown op" }));
            }
        }
    }

    println!("{}", serde_json::to_string(&out).unwrap());
}
