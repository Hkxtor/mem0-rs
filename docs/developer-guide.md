# Developer guide (Rust library)

This guide covers using mem0-rs as a Rust library end to end. For the HTTP
service, see [server.md](server.md); for concepts, see [concepts.md](concepts.md).

## Add the dependency

mem0-rs is a Cargo workspace. Depend on the `mem0` facade crate (path or git),
and on `serde_json` for building filters/metadata. Enable a scale-out backend
feature if you need it:

```toml
[dependencies]
mem0 = { path = "../mem0-rs/crates/mem0" }          # or a git dependency
serde_json = "1"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }

# optional external vector backends
# mem0 = { path = "…", features = ["qdrant"] }
# mem0 = { path = "…", features = ["pgvector"] }
```

The facade re-exports everything you need: `Memory`, `AddOptions`,
`SearchOptions`, `Message`, `JsonMap`, `Mem0Error`, `Result`, and the provider
traits.

## Construct a Memory

```rust
// From an inline JSON config:
let mem = mem0::from_json(r#"{
    "embedder": { "provider": "openai", "config": { "api_key": "sk-..." } },
    "llm":      { "provider": "openai", "config": { "api_key": "sk-..." } },
    "vector_store": { "provider": "embedded", "config": { "path": "/var/lib/mem0" } }
}"#)?;

// …or from a typed MemoryConfig:
use mem0::MemoryConfig;
let cfg = MemoryConfig::from_json(std::fs::read_to_string("mem0.json")?.as_str())?;
let mem = mem0::from_config(cfg)?;
```

`Memory` is `Send + Sync`; wrap it in `Arc` to share across tasks.

## Adding memories

`add` accepts a `&str`, a `String`, a `Message`, or a `Vec<Message>` (anything
`Into<MessagesInput>`), plus `AddOptions`.

```rust
use mem0::{AddOptions, Message};

// Inferred (LLM extracts durable facts) — the default.
let added = mem.add(
    vec![
        Message::user("I'm Alex, I'm vegetarian and I love trail running"),
        Message::assistant("Great to know, Alex!"),
    ],
    AddOptions { user_id: Some("alex".into()), ..Default::default() },
).await?;
for a in &added {
    println!("ADDED {}: {}", a.id, a.memory);
}

// Raw (store text verbatim, no LLM).
mem.add("User prefers window seats", AddOptions {
    user_id: Some("alex".into()), infer: Some(false), ..Default::default() }).await?;

// With metadata for later filtering.
let mut meta = mem0::JsonMap::new();
meta.insert("category".into(), serde_json::json!("travel"));
mem.add("User is planning a trip to Japan", AddOptions {
    user_id: Some("alex".into()), infer: Some(false), metadata: Some(meta),
    ..Default::default() }).await?;
```

Scope with `user_id`, `agent_id`, and/or `run_id` (at least one required). A
`Message`'s optional `name` becomes the `actor_id` for raw adds.

## Searching

```rust
use mem0::{SearchOptions, JsonMap};
use serde_json::json;

let mut filters = JsonMap::new();
filters.insert("user_id".into(), json!("alex"));
filters.insert("category".into(), json!({ "in": ["food", "travel"] }));

let results = mem.search("what does the user like to eat?", &filters,
    SearchOptions { top_k: 5, threshold: 0.1, rerank: false }).await?;

for m in results["results"].as_array().unwrap() {
    println!("{} (score {})", m["memory"], m["score"]);
}
```

`search` returns `{"results": [ {id, memory, score, created_at, …, metadata} ]}`.

## Reading, updating, deleting

```rust
// One memory by id.
let one = mem.get(&id).await?;                       // Option<Value>

// All memories in a scope.
let mut f = JsonMap::new(); f.insert("user_id".into(), json!("alex"));
let all = mem.get_all(&f, 50).await?;                // {"results":[...]}

// Update text (and optionally metadata).
mem.update(&id, "User is now vegan", None).await?;

// Delete one / a whole scope.
mem.delete(&id).await?;
mem.delete_all(Some("alex"), None, None).await?;

// Audit trail.
let history = mem.history(&id).await?;               // Vec<HistoryRecord>

// Wipe everything (vectors + history).
mem.reset().await?;
```

## Procedural memory

```rust
mem.add(conversation, AddOptions {
    agent_id: Some("coding-agent".into()),
    memory_type: Some("procedural_memory".into()),
    ..Default::default()
}).await?;
```

## Blocking (synchronous) API

```rust
let mem = mem0::blocking::Memory::from_json(config_json)?;
mem.add("I like aisle seats", mem0::AddOptions {
    user_id: Some("alex".into()), infer: Some(false), ..Default::default() })?;
let all = mem.get_all(&filters, 20)?;
```

`blocking::Memory` owns a Tokio runtime and mirrors the async method set.

## Error handling

All operations return `Result<T, mem0::Mem0Error>`. `Mem0Error` carries a
machine-readable `code()` and an optional suggestion:

```rust
match mem.get_all(&JsonMap::new(), 10).await {
    Ok(v) => { /* … */ }
    Err(e) => eprintln!("error {:?}: {e}", e.code()),  // e.g. Some("VALIDATION_001")
}
```

Common codes: `VALIDATION_001` (missing scope), `VALIDATION_004/005` (bad
`top_k`/`threshold`), `MEM_404` (not found), `LLM_001`, `EMBED_001`,
`VECTOR_001`, `CFG_001`. See [api/library.md](api/library.md#errors).

## Extending with a custom provider

Because the core depends only on traits, you can implement your own embedder,
LLM, vector store, or reranker and pass it directly to `Memory::new` (the
lower-level constructor in `mem0_core`):

```rust
use mem0_core::{Memory, MemoryConfig};
use mem0_core::traits::{Embedder, MemoryAction};
use async_trait::async_trait;

struct MyEmbedder;
#[async_trait]
impl Embedder for MyEmbedder {
    async fn embed(&self, text: &str, _a: MemoryAction) -> mem0_core::Result<Vec<f32>> {
        Ok(/* your vector */ vec![0.0; 384])
    }
    fn dims(&self) -> usize { 384 }
}

let mem = Memory::new(
    MemoryConfig { history_db_path: ":memory:".into(), ..Default::default() },
    Box::new(MyEmbedder),
    mem0::build_llm("mock", &serde_json::json!({}))?,
    mem0::build_vector_store("embedded", &serde_json::json!({}))?,
    None,
)?;
```

The built-in `build_embedder` / `build_llm` / `build_vector_store` /
`build_reranker` factories are also re-exported from `mem0` if you want to
construct individual providers by name.

## Testing your integration

Use the `mock` LLM + `mock` embedder + `embedded` (`:memory:`) store for fast,
deterministic, network-free tests — the same setup mem0-rs uses internally
(see the `quickstart` and `parity_run` examples).

Next: [Rust library API](api/library.md) · [REST API](api/rest.md).