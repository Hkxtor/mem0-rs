//! HTTP client for the mem0 REST server (`mem0-server`).
//!
//! All methods return the server's JSON response as [`serde_json::Value`], or a
//! [`mem0_core::Mem0Error`] on transport / non-2xx errors.

use mem0_core::error::{Mem0Error, Result};
use serde_json::{json, Value};

/// A client for a running mem0 server.
#[derive(Clone)]
pub struct Mem0Client {
    http: reqwest::Client,
    base_url: String,
}

impl Mem0Client {
    /// Construct a client targeting `base_url` (e.g. `http://localhost:8080`).
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
        }
    }

    async fn handle(resp: reqwest::Response, ctx: &str) -> Result<Value> {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if status.is_success() {
            if body.is_empty() {
                return Ok(Value::Null);
            }
            serde_json::from_str(&body).map_err(Mem0Error::from)
        } else if status.as_u16() == 404 {
            Err(Mem0Error::not_found(format!("{ctx}: {body}")))
        } else if status.as_u16() == 400 {
            Err(Mem0Error::validation(format!("{ctx}: {body}")))
        } else {
            Err(Mem0Error::configuration(format!(
                "{ctx}: HTTP {status}: {body}"
            )))
        }
    }

    /// Add memories. `request` is the JSON body (`messages`, `user_id`, …).
    pub async fn add(&self, request: &Value) -> Result<Value> {
        let url = format!("{}/v1/memories", self.base_url);
        let resp = self
            .http
            .post(&url)
            .json(request)
            .send()
            .await
            .map_err(|e| Mem0Error::configuration(format!("add request failed: {e}")))?;
        Self::handle(resp, "add").await
    }

    /// Retrieve a memory by id.
    pub async fn get(&self, id: &str) -> Result<Value> {
        let url = format!("{}/v1/memories/{id}", self.base_url);
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| Mem0Error::configuration(format!("get request failed: {e}")))?;
        Self::handle(resp, "get").await
    }

    /// List memories for a scope.
    pub async fn get_all(
        &self,
        user_id: Option<&str>,
        agent_id: Option<&str>,
        run_id: Option<&str>,
        top_k: Option<usize>,
    ) -> Result<Value> {
        let url = format!("{}/v1/memories", self.base_url);
        let mut query: Vec<(String, String)> = Vec::new();
        if let Some(v) = user_id {
            query.push(("user_id".into(), v.into()));
        }
        if let Some(v) = agent_id {
            query.push(("agent_id".into(), v.into()));
        }
        if let Some(v) = run_id {
            query.push(("run_id".into(), v.into()));
        }
        if let Some(k) = top_k {
            query.push(("top_k".into(), k.to_string()));
        }
        let resp = self
            .http
            .get(&url)
            .query(&query)
            .send()
            .await
            .map_err(|e| Mem0Error::configuration(format!("get_all request failed: {e}")))?;
        Self::handle(resp, "get_all").await
    }

    /// Search memories. `request` is the JSON body (`query`, `user_id`, …).
    pub async fn search(&self, request: &Value) -> Result<Value> {
        let url = format!("{}/v1/memories/search", self.base_url);
        let resp = self
            .http
            .post(&url)
            .json(request)
            .send()
            .await
            .map_err(|e| Mem0Error::configuration(format!("search request failed: {e}")))?;
        Self::handle(resp, "search").await
    }

    /// Update a memory.
    pub async fn update(&self, id: &str, data: &str, metadata: Option<Value>) -> Result<Value> {
        let url = format!("{}/v1/memories/{id}", self.base_url);
        let body = json!({ "data": data, "metadata": metadata });
        let resp = self
            .http
            .put(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| Mem0Error::configuration(format!("update request failed: {e}")))?;
        Self::handle(resp, "update").await
    }

    /// Delete a memory by id.
    pub async fn delete(&self, id: &str) -> Result<Value> {
        let url = format!("{}/v1/memories/{id}", self.base_url);
        let resp = self
            .http
            .delete(&url)
            .send()
            .await
            .map_err(|e| Mem0Error::configuration(format!("delete request failed: {e}")))?;
        Self::handle(resp, "delete").await
    }

    /// Get the change history of a memory.
    pub async fn history(&self, id: &str) -> Result<Value> {
        let url = format!("{}/v1/memories/{id}/history", self.base_url);
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| Mem0Error::configuration(format!("history request failed: {e}")))?;
        Self::handle(resp, "history").await
    }

    /// Reset all memories.
    pub async fn reset(&self) -> Result<Value> {
        let url = format!("{}/v1/reset", self.base_url);
        let resp = self
            .http
            .post(&url)
            .send()
            .await
            .map_err(|e| Mem0Error::configuration(format!("reset request failed: {e}")))?;
        Self::handle(resp, "reset").await
    }
}
