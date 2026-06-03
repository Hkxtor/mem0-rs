# Rust library API reference

Everything here is re-exported from the `mem0` crate. Types originate in
`mem0-core`.

## Construction

```rust
pub fn mem0::from_json(s: &str) -> Result<Memory>;
pub fn mem0::from_config(config: MemoryConfig) -> Result<Memory>;

// lower-level (mem0_core) — inject your own providers:
Memory::new(config: MemoryConfig,
            embedder: Box<dyn Embedder>,
            llm: Box<dyn Llm>,
            vector_store: Box<dyn VectorStore>,
            reranker: Option<Box<dyn Reranker>>) -> Result<Memory>;
Memory::with_entity_store(self, store: Box<dyn VectorStore>) -> Memory;

// provider factories (re-exported):
mem0::build_embedder(provider: &str, config: &Value) -> Result<Box<dyn Embedder>>;
mem0::build_llm(provider: &str, config: &Value) -> Result<Box<dyn Llm>>;
mem0::build_vector_store(provider: &str, config: &Value) -> Result<Box<dyn VectorStore>>;
mem0::build_reranker(provider: &str, config: &Value) -> Result<Box<dyn Reranker>>;
```

## `Memory` methods

```rust
async fn add(&self, messages: impl Into<MessagesInput>, opts: AddOptions)
    -> Result<Vec<AddResult>>;

async fn search(&self, query: &str, filters: &JsonMap, options: SearchOptions)
    -> Result<Value>;                       // { "results": [ memory_item, … ] }

async fn get(&self, memory_id: &str) -> Result<Option<Value>>;     // memory_item | None
async fn get_all(&self, filters: &JsonMap, top_k: usize) -> Result<Value>; // { "results": [...] }

async fn update(&self, memory_id: &str, data: &str, metadata: Option<JsonMap>)
    -> Result<Value>;                       // { "message": "…" }
async fn delete(&self, memory_id: &str) -> Result<Value>;          // { "message": "…" }
async fn delete_all(&self, user_id: Option<&str>, agent_id: Option<&str>, run_id: Option<&str>)
    -> Result<Value>;                       // { "message": "…" }

async fn history(&self, memory_id: &str) -> Result<Vec<HistoryRecord>>;
async fn reset(&self) -> Result<()>;
```

All methods require at least one scope id where applicable (`add`, `get_all`,
`search`, `delete_all`).

## `AddOptions`

```rust
pub struct AddOptions {
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub run_id: Option<String>,
    pub metadata: Option<JsonMap>,
    pub infer: Option<bool>,            // None => true
    pub memory_type: Option<String>,   // "procedural_memory"
    pub prompt: Option<String>,        // per-call extraction instructions
}
```
`AddOptions::default()` has all fields `None` (so `infer` defaults to `true`).

## `SearchOptions`

```rust
pub struct SearchOptions { pub top_k: usize, pub threshold: f64, pub rerank: bool }
// Default: top_k = 20, threshold = 0.1, rerank = false
```

## Core types

```rust
pub type JsonMap = serde_json::Map<String, serde_json::Value>;

pub struct Message { pub role: String, pub content: String, pub name: Option<String> }
impl Message { fn user(c)->Message; fn system(c)->Message; fn assistant(c)->Message;
               fn new(role, content)->Message; }

pub enum MessagesInput { Text(String), One(Message), Many(Vec<Message>) }
// From<&str>, From<String>, From<Message>, From<Vec<Message>>

pub struct AddResult { pub id: String, pub memory: String, pub event: String,
                       pub actor_id: Option<String>, pub role: Option<String> }

pub struct SearchHit { pub id: String, pub score: f32, pub payload: JsonMap }
pub struct VectorRecord { pub id: String, pub vector: Vec<f32>, pub payload: JsonMap }

pub enum Event { Add, Update, Delete, None }      // serializes as ADD/UPDATE/DELETE/NONE
pub enum MemoryType { Semantic, Episodic, Procedural }
pub enum MemoryAction { Add, Search, Update }      // embedding context
```

### `HistoryRecord`

Returned by `history()` (and serialized under `"history"` by the server):

```rust
pub struct HistoryRecord {
    pub id: String,
    pub memory_id: String,
    pub old_memory: Option<String>,
    pub new_memory: Option<String>,
    pub event: String,             // ADD | UPDATE | DELETE
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
    pub is_deleted: bool,
    pub actor_id: Option<String>,
    pub role: Option<String>,
}
```

### Memory item shape

`get` / `get_all` / `search` return memory items as JSON:

```json
{
  "id": "…",
  "memory": "User is vegetarian",
  "hash": "…",
  "created_at": "…", "updated_at": "…",
  "user_id": "alex",          // promoted scope keys when present
  "score": 0.83,              // search results only
  "metadata": { "category": "diet" }   // any non-core payload keys
}
```

## Configuration types

```rust
pub struct MemoryConfig {
    pub vector_store: VectorStoreConfig,
    pub llm: LlmConfig,
    pub embedder: EmbedderConfig,
    pub reranker: Option<RerankerConfig>,
    pub history_db_path: String,         // default ~/.mem0/history.db
    pub version: String,                 // "v1.1"
    pub custom_instructions: Option<String>,
}
impl MemoryConfig { fn from_json(s: &str) -> Result<MemoryConfig>; }

// each of EmbedderConfig/LlmConfig/VectorStoreConfig/RerankerConfig is:
pub struct XConfig { pub provider: String, pub config: serde_json::Value }
```

## Traits (for custom providers)

```rust
#[async_trait] pub trait Embedder {
    async fn embed(&self, text: &str, action: MemoryAction) -> Result<Vec<f32>>;
    async fn embed_batch(&self, texts: &[String], action: MemoryAction) -> Result<Vec<Vec<f32>>>; // default loops
    fn dims(&self) -> usize;
}
#[async_trait] pub trait Llm {
    async fn generate(&self, messages: &[Message], options: &GenerateOptions) -> Result<String>;
}
#[async_trait] pub trait VectorStore { /* insert, search, get, update, delete, list,
    delete_col, reset, keyword_search, search_batch */ }
#[async_trait] pub trait Reranker {
    async fn rerank(&self, query: &str, documents: &[String], top_n: usize)
        -> Result<Vec<(usize, f32)>>;
}

pub struct GenerateOptions {
    pub response_format_json: bool,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f32>,
    pub tools: Option<serde_json::Value>,
}
```

## Blocking API

`mem0::blocking::Memory` mirrors the async API synchronously (it owns a Tokio
runtime):

```rust
impl blocking::Memory {
    fn from_config(MemoryConfig) -> Result<Self>;
    fn from_json(&str) -> Result<Self>;
    fn add(&self, impl Into<MessagesInput>, AddOptions) -> Result<Vec<AddResult>>;
    fn get(&self, &str) -> Result<Option<Value>>;
    fn get_all(&self, &JsonMap, usize) -> Result<Value>;
    fn search(&self, &str, &JsonMap, SearchOptions) -> Result<Value>;
    fn update(&self, &str, &str, Option<JsonMap>) -> Result<Value>;
    fn delete(&self, &str) -> Result<Value>;
    fn delete_all(&self, Option<&str>, Option<&str>, Option<&str>) -> Result<Value>;
    fn history(&self, &str) -> Result<Vec<HistoryRecord>>;
    fn reset(&self) -> Result<()>;
}
```

## Errors

```rust
pub type Result<T> = std::result::Result<T, Mem0Error>;

pub enum Mem0Error {
    Validation { code, message, suggestion },
    Authentication { … }, RateLimit { … }, NotFound { … }, Network { … },
    Configuration { … }, VectorStore { … }, Embedding { … }, Llm { … },
    Database { … }, Dependency { … },
    Json(serde_json::Error), Sqlite(rusqlite::Error),
}
impl Mem0Error { fn code(&self) -> Option<&str>; }
```

Representative codes: `VALIDATION_001` (missing scope id), `VALIDATION_004`
(bad `top_k`), `VALIDATION_005` (bad `threshold`), `MEM_404` (not found),
`CFG_001`, `VECTOR_001`, `EMBED_001`, `LLM_001`, `DB_001`.

See also: [REST API](rest.md) · [HTTP client](client.md).