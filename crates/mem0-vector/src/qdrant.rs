//! Qdrant vector store over the REST API. Port of `vector_stores/qdrant.py`
//! (dense semantic search + metadata filtering). BM25 sparse search (fastembed)
//! is out of scope; `keyword_search` returns `None` so the orchestrator relies
//! on semantic scoring for this backend.

use crate::config::VectorStoreSettings;
use async_trait::async_trait;
use mem0_core::error::{Mem0Error, Result};
use mem0_core::traits::VectorStore;
use mem0_core::types::{JsonMap, SearchHit, VectorRecord};
use reqwest::RequestBuilder;
use serde_json::{json, Value};
use tokio::sync::OnceCell;

/// Qdrant REST-backed vector store.
pub struct QdrantStore {
    client: reqwest::Client,
    base: String,
    api_key: Option<String>,
    collection: String,
    dims: usize,
    init: OnceCell<()>,
}

impl QdrantStore {
    /// Construct a Qdrant store (requires `url`, or `host` + `port`).
    pub fn new(settings: VectorStoreSettings) -> Result<Self> {
        let base = if let Some(url) = &settings.url {
            url.trim_end_matches('/').to_string()
        } else if let (Some(h), Some(p)) = (&settings.host, settings.port) {
            format!("http://{h}:{p}")
        } else {
            return Err(Mem0Error::configuration(
                "qdrant requires 'url' or 'host' + 'port'",
            ));
        };
        Ok(Self {
            client: reqwest::Client::new(),
            base,
            api_key: settings.api_key.clone(),
            collection: settings.collection_name(),
            dims: settings.dims(),
            init: OnceCell::new(),
        })
    }

    fn with_key(&self, rb: RequestBuilder) -> RequestBuilder {
        match &self.api_key {
            Some(k) => rb.header("api-key", k),
            None => rb,
        }
    }

    async fn ensure(&self) -> Result<()> {
        self.init
            .get_or_try_init(|| self.create_collection())
            .await
            .map(|_| ())
    }

    async fn create_collection(&self) -> Result<()> {
        let url = format!("{}/collections/{}", self.base, self.collection);
        let resp = self
            .with_key(self.client.get(&url))
            .send()
            .await
            .map_err(net)?;
        if resp.status().is_success() {
            return Ok(());
        }
        let body = json!({
            "vectors": { "size": self.dims, "distance": "Cosine" }
        });
        let resp = self
            .with_key(self.client.put(&url))
            .json(&body)
            .send()
            .await
            .map_err(net)?;
        ok_or_err(resp, "qdrant create collection").await
    }
}

fn net(e: reqwest::Error) -> Mem0Error {
    Mem0Error::vector_store(format!("qdrant request failed: {e}"))
}

async fn ok_or_err(resp: reqwest::Response, ctx: &str) -> Result<()> {
    if resp.status().is_success() {
        Ok(())
    } else {
        let code = resp.status();
        let body = resp.text().await.unwrap_or_default();
        Err(Mem0Error::vector_store(format!(
            "{ctx} HTTP {code}: {body}"
        )))
    }
}

fn hit_from(point: &Value) -> Option<SearchHit> {
    let id = point.get("id").map(value_to_id)?;
    let score = point.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0) as f32;
    let payload = point
        .get("payload")
        .and_then(|p| p.as_object())
        .cloned()
        .unwrap_or_default();
    Some(SearchHit { id, score, payload })
}

fn value_to_id(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Translate mem0 (normalized) filters into a Qdrant filter object.
/// Port of `_create_filter` / `_build_field_condition`.
pub(crate) fn qdrant_filter(filters: &JsonMap) -> Option<Value> {
    if filters.is_empty() {
        return None;
    }
    // Normalize $or/$not/$and → OR/NOT/AND, dedup by normalized key (keep first).
    let mut normalized: Vec<(String, &Value)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (k, v) in filters {
        let nk = match k.as_str() {
            "$or" => "OR",
            "$not" => "NOT",
            "$and" => "AND",
            other => other,
        }
        .to_string();
        if seen.insert(nk.clone()) {
            normalized.push((nk, v));
        }
    }

    let mut must: Vec<Value> = Vec::new();
    let mut should: Vec<Value> = Vec::new();
    let mut must_not: Vec<Value> = Vec::new();

    for (key, value) in normalized {
        match key.as_str() {
            "AND" | "OR" | "NOT" => {
                if let Some(arr) = value.as_array() {
                    for sub in arr {
                        if let Some(obj) = sub.as_object() {
                            if let Some(f) = qdrant_filter(obj) {
                                match key.as_str() {
                                    "AND" => must.push(f),
                                    "OR" => should.push(f),
                                    _ => must_not.push(f),
                                }
                            }
                        }
                    }
                }
            }
            _ => {
                if let Some(cond) = field_condition(&key, value) {
                    must.push(cond);
                }
            }
        }
    }

    if must.is_empty() && should.is_empty() && must_not.is_empty() {
        return None;
    }
    let mut obj = serde_json::Map::new();
    if !must.is_empty() {
        obj.insert("must".into(), Value::Array(must));
    }
    if !should.is_empty() {
        obj.insert("should".into(), Value::Array(should));
    }
    if !must_not.is_empty() {
        obj.insert("must_not".into(), Value::Array(must_not));
    }
    Some(Value::Object(obj))
}

fn field_condition(key: &str, value: &Value) -> Option<Value> {
    match value {
        Value::Object(ops) => {
            let range_keys = ["gt", "gte", "lt", "lte"];
            if ops.keys().any(|k| range_keys.contains(&k.as_str())) {
                let mut range = serde_json::Map::new();
                for rk in range_keys {
                    if let Some(v) = ops.get(rk) {
                        range.insert(rk.into(), v.clone());
                    }
                }
                return Some(json!({ "key": key, "range": range }));
            }
            if let Some(v) = ops.get("eq") {
                return Some(json!({ "key": key, "match": { "value": v } }));
            }
            if let Some(v) = ops.get("ne") {
                return Some(json!({ "key": key, "match": { "except": [v] } }));
            }
            if let Some(v) = ops.get("in") {
                return Some(json!({ "key": key, "match": { "any": v } }));
            }
            if let Some(v) = ops.get("nin") {
                return Some(json!({ "key": key, "match": { "except": v } }));
            }
            if let Some(v) = ops.get("contains").or_else(|| ops.get("icontains")) {
                return Some(json!({ "key": key, "match": { "text": v } }));
            }
            None
        }
        Value::String(s) if s == "*" => None,
        Value::Array(_) => Some(json!({ "key": key, "match": { "any": value } })),
        _ => Some(json!({ "key": key, "match": { "value": value } })),
    }
}

#[async_trait]
impl VectorStore for QdrantStore {
    async fn insert(&self, records: Vec<VectorRecord>) -> Result<()> {
        self.ensure().await?;
        let points: Vec<Value> = records
            .iter()
            .map(|r| json!({ "id": r.id, "vector": r.vector, "payload": Value::Object(r.payload.clone()) }))
            .collect();
        let url = format!(
            "{}/collections/{}/points?wait=true",
            self.base, self.collection
        );
        let resp = self
            .with_key(self.client.put(&url))
            .json(&json!({ "points": points }))
            .send()
            .await
            .map_err(net)?;
        ok_or_err(resp, "qdrant insert").await
    }

    async fn search(
        &self,
        _query: &str,
        vector: &[f32],
        top_k: usize,
        filters: &JsonMap,
    ) -> Result<Vec<SearchHit>> {
        self.ensure().await?;
        let mut body = json!({ "query": vector, "limit": top_k, "with_payload": true });
        if let Some(f) = qdrant_filter(filters) {
            body["filter"] = f;
        }
        let url = format!("{}/collections/{}/points/query", self.base, self.collection);
        let resp = self
            .with_key(self.client.post(&url))
            .json(&body)
            .send()
            .await
            .map_err(net)?;
        if !resp.status().is_success() {
            return Err(Mem0Error::vector_store(format!(
                "qdrant search HTTP {}",
                resp.status()
            )));
        }
        let value: Value = resp.json().await.map_err(net)?;
        let points = value
            .get("result")
            .and_then(|r| r.get("points"))
            .and_then(|p| p.as_array())
            .cloned()
            .unwrap_or_default();
        Ok(points.iter().filter_map(hit_from).collect())
    }

    async fn get(&self, id: &str) -> Result<Option<SearchHit>> {
        self.ensure().await?;
        let url = format!(
            "{}/collections/{}/points/{}",
            self.base, self.collection, id
        );
        let resp = self
            .with_key(self.client.get(&url))
            .send()
            .await
            .map_err(net)?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !resp.status().is_success() {
            return Err(Mem0Error::vector_store(format!(
                "qdrant get HTTP {}",
                resp.status()
            )));
        }
        let value: Value = resp.json().await.map_err(net)?;
        Ok(value
            .get("result")
            .filter(|r| !r.is_null())
            .and_then(hit_from))
    }

    async fn update(
        &self,
        id: &str,
        vector: Option<Vec<f32>>,
        payload: Option<JsonMap>,
    ) -> Result<()> {
        self.ensure().await?;
        if let Some(p) = &payload {
            let url = format!(
                "{}/collections/{}/points/payload?wait=true",
                self.base, self.collection
            );
            let resp = self
                .with_key(self.client.post(&url))
                .json(&json!({ "payload": Value::Object(p.clone()), "points": [id] }))
                .send()
                .await
                .map_err(net)?;
            ok_or_err(resp, "qdrant set payload").await?;
        }
        if let Some(v) = &vector {
            let url = format!(
                "{}/collections/{}/points/vectors?wait=true",
                self.base, self.collection
            );
            let resp = self
                .with_key(self.client.put(&url))
                .json(&json!({ "points": [ { "id": id, "vector": v } ] }))
                .send()
                .await
                .map_err(net)?;
            ok_or_err(resp, "qdrant update vectors").await?;
        }
        Ok(())
    }

    async fn delete(&self, id: &str) -> Result<()> {
        self.ensure().await?;
        let url = format!(
            "{}/collections/{}/points/delete?wait=true",
            self.base, self.collection
        );
        let resp = self
            .with_key(self.client.post(&url))
            .json(&json!({ "points": [id] }))
            .send()
            .await
            .map_err(net)?;
        ok_or_err(resp, "qdrant delete").await
    }

    async fn list(&self, filters: &JsonMap, limit: Option<usize>) -> Result<Vec<SearchHit>> {
        self.ensure().await?;
        let mut body = json!({ "limit": limit.unwrap_or(100), "with_payload": true });
        if let Some(f) = qdrant_filter(filters) {
            body["filter"] = f;
        }
        let url = format!(
            "{}/collections/{}/points/scroll",
            self.base, self.collection
        );
        let resp = self
            .with_key(self.client.post(&url))
            .json(&body)
            .send()
            .await
            .map_err(net)?;
        if !resp.status().is_success() {
            return Err(Mem0Error::vector_store(format!(
                "qdrant scroll HTTP {}",
                resp.status()
            )));
        }
        let value: Value = resp.json().await.map_err(net)?;
        let points = value
            .get("result")
            .and_then(|r| r.get("points"))
            .and_then(|p| p.as_array())
            .cloned()
            .unwrap_or_default();
        Ok(points.iter().filter_map(hit_from).collect())
    }

    async fn delete_col(&self) -> Result<()> {
        let url = format!("{}/collections/{}", self.base, self.collection);
        let resp = self
            .with_key(self.client.delete(&url))
            .send()
            .await
            .map_err(net)?;
        ok_or_err(resp, "qdrant delete collection").await
    }

    async fn reset(&self) -> Result<()> {
        self.delete_col().await?;
        self.create_collection().await
    }
}
