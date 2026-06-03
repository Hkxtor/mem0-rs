//! axum REST API for mem0.
//!
//! [`app`] builds the router around an `Arc<Memory>`; [`build_memory_from_env`]
//! constructs the memory from `MEM0_CONFIG` / `MEM0_CONFIG_FILE` (or a sensible
//! default). The binary in `main.rs` ties them together into a single
//! self-contained server.

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use mem0::{AddOptions, JsonMap, Memory, MemoryConfig, Message, MessagesInput, SearchOptions};
use mem0_core::Mem0Error;
use serde::Deserialize;
use serde_json::{json, Value};

/// Shared application state.
#[derive(Clone)]
pub struct AppState {
    /// The wired memory instance.
    pub memory: Arc<Memory>,
}

/// Build the axum router for the given memory instance.
pub fn app(memory: Arc<Memory>) -> Router {
    let state = AppState { memory };
    Router::new()
        .route("/health", get(health))
        .route("/v1/memories", post(add).get(get_all).delete(delete_all))
        .route("/v1/memories/search", post(search))
        .route(
            "/v1/memories/:id",
            get(get_one).put(update).delete(delete_one),
        )
        .route("/v1/memories/:id/history", get(history))
        .route("/v1/reset", post(reset))
        .with_state(state)
}

/// Build a [`Memory`] from environment configuration.
///
/// Resolution order: `MEM0_CONFIG_FILE` (path to JSON) → `MEM0_CONFIG` (inline
/// JSON) → [`MemoryConfig::default`] (embedded vector store + OpenAI providers).
pub fn build_memory_from_env() -> mem0_core::Result<Memory> {
    let config = if let Ok(path) = std::env::var("MEM0_CONFIG_FILE") {
        let s = std::fs::read_to_string(&path)
            .map_err(|e| Mem0Error::configuration(format!("failed to read {path}: {e}")))?;
        MemoryConfig::from_json(&s)?
    } else if let Ok(inline) = std::env::var("MEM0_CONFIG") {
        MemoryConfig::from_json(&inline)?
    } else {
        MemoryConfig::default()
    };
    mem0::from_config(config)
}

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

struct ApiError(Mem0Error);

impl From<Mem0Error> for ApiError {
    fn from(e: Mem0Error) -> Self {
        ApiError(e)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match &self.0 {
            Mem0Error::Validation { .. } => StatusCode::BAD_REQUEST,
            Mem0Error::NotFound { .. } => StatusCode::NOT_FOUND,
            Mem0Error::Authentication { .. } => StatusCode::UNAUTHORIZED,
            Mem0Error::RateLimit { .. } => StatusCode::TOO_MANY_REQUESTS,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        let body = json!({
            "error": self.0.to_string(),
            "code": self.0.code(),
        });
        (status, Json(body)).into_response()
    }
}

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct AddRequest {
    messages: Value,
    #[serde(default)]
    user_id: Option<String>,
    #[serde(default)]
    agent_id: Option<String>,
    #[serde(default)]
    run_id: Option<String>,
    #[serde(default)]
    metadata: Option<JsonMap>,
    #[serde(default)]
    infer: Option<bool>,
    #[serde(default)]
    memory_type: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
}

#[derive(Deserialize)]
struct SearchRequest {
    query: String,
    #[serde(default)]
    user_id: Option<String>,
    #[serde(default)]
    agent_id: Option<String>,
    #[serde(default)]
    run_id: Option<String>,
    #[serde(default)]
    filters: Option<JsonMap>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    threshold: Option<f64>,
    #[serde(default)]
    rerank: Option<bool>,
}

#[derive(Deserialize)]
struct UpdateRequest {
    data: String,
    #[serde(default)]
    metadata: Option<JsonMap>,
}

#[derive(Deserialize)]
struct ScopeParams {
    #[serde(default)]
    user_id: Option<String>,
    #[serde(default)]
    agent_id: Option<String>,
    #[serde(default)]
    run_id: Option<String>,
    #[serde(default)]
    top_k: Option<usize>,
}

fn to_messages_input(v: Value) -> Result<MessagesInput, Mem0Error> {
    match v {
        Value::String(s) => Ok(MessagesInput::Text(s)),
        Value::Array(_) => {
            let msgs: Vec<Message> = serde_json::from_value(v).map_err(Mem0Error::from)?;
            Ok(MessagesInput::Many(msgs))
        }
        Value::Object(_) => {
            let m: Message = serde_json::from_value(v).map_err(Mem0Error::from)?;
            Ok(MessagesInput::One(m))
        }
        _ => Err(Mem0Error::validation(
            "messages must be a string, object, or array of messages",
        )),
    }
}

fn scope_filters(
    user_id: &Option<String>,
    agent_id: &Option<String>,
    run_id: &Option<String>,
    extra: Option<JsonMap>,
) -> JsonMap {
    let mut filters = extra.unwrap_or_default();
    if let Some(v) = user_id {
        filters.insert("user_id".into(), json!(v));
    }
    if let Some(v) = agent_id {
        filters.insert("agent_id".into(), json!(v));
    }
    if let Some(v) = run_id {
        filters.insert("run_id".into(), json!(v));
    }
    filters
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

async fn add(
    State(st): State<AppState>,
    Json(req): Json<AddRequest>,
) -> Result<Json<Value>, ApiError> {
    let messages = to_messages_input(req.messages)?;
    let opts = AddOptions {
        user_id: req.user_id,
        agent_id: req.agent_id,
        run_id: req.run_id,
        metadata: req.metadata,
        infer: req.infer,
        memory_type: req.memory_type,
        prompt: req.prompt,
    };
    let results = st.memory.add(messages, opts).await?;
    Ok(Json(json!({ "results": results })))
}

async fn get_one(
    State(st): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    match st.memory.get(&id).await? {
        Some(v) => Ok(Json(v)),
        None => Err(ApiError(Mem0Error::not_found(format!(
            "Memory with id {id} not found"
        )))),
    }
}

async fn get_all(
    State(st): State<AppState>,
    Query(params): Query<ScopeParams>,
) -> Result<Json<Value>, ApiError> {
    let filters = scope_filters(&params.user_id, &params.agent_id, &params.run_id, None);
    let result = st
        .memory
        .get_all(&filters, params.top_k.unwrap_or(20))
        .await?;
    Ok(Json(result))
}

async fn search(
    State(st): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<Value>, ApiError> {
    let filters = scope_filters(&req.user_id, &req.agent_id, &req.run_id, req.filters);
    let options = SearchOptions {
        top_k: req.top_k.unwrap_or(20),
        threshold: req.threshold.unwrap_or(0.1),
        rerank: req.rerank.unwrap_or(false),
    };
    let result = st.memory.search(&req.query, &filters, options).await?;
    Ok(Json(result))
}

async fn update(
    State(st): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<UpdateRequest>,
) -> Result<Json<Value>, ApiError> {
    let result = st.memory.update(&id, &req.data, req.metadata).await?;
    Ok(Json(result))
}

async fn delete_one(
    State(st): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let result = st.memory.delete(&id).await?;
    Ok(Json(result))
}

async fn delete_all(
    State(st): State<AppState>,
    Query(params): Query<ScopeParams>,
) -> Result<Json<Value>, ApiError> {
    let result = st
        .memory
        .delete_all(
            params.user_id.as_deref(),
            params.agent_id.as_deref(),
            params.run_id.as_deref(),
        )
        .await?;
    Ok(Json(result))
}

async fn history(
    State(st): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let records = st.memory.history(&id).await?;
    Ok(Json(json!({ "history": records })))
}

async fn reset(State(st): State<AppState>) -> Result<Json<Value>, ApiError> {
    st.memory.reset().await?;
    Ok(Json(json!({ "message": "All memories reset" })))
}

// Re-export for the `Query<HashMap>` ergonomics if needed by downstream users.
#[allow(dead_code)]
type RawQuery = HashMap<String, String>;
