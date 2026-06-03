# Getting started

This guide gets you from zero to a working memory in a few minutes, both as a
Rust library and as a standalone server.

## Prerequisites

- **Rust** (stable, 1.80+; the workspace is developed on current stable). Install
  via [rustup](https://rustup.rs).
- A C toolchain (for the bundled SQLite via `rusqlite`) — `cc`/`clang` on macOS,
  `gcc` on Linux. Usually already present.
- Optional: API keys for your chosen provider (OpenAI, Anthropic, Gemini, …) —
  not needed for the in-process `mock`/`embedded` quickstart.

No external services are required for the default configuration.

## Get the code

mem0-rs lives in the `mem0-rs/` Cargo workspace. From the repository root:

```bash
cd mem0-rs
cargo build --release          # builds all crates
```

To build just the server binary:

```bash
cargo build --release -p mem0-server
# -> target/release/mem0-server   (~12 MB, no external runtime)
```

Optional external vector backends are feature-gated:

```bash
cargo build --release -p mem0-server --features qdrant
cargo build --release -p mem0-server --features pgvector
```

## Your first program (library)

Add the crate to your `Cargo.toml` (path or git dependency on the workspace),
then:

```rust
use mem0::{AddOptions, SearchOptions, JsonMap};
use serde_json::json;

#[tokio::main]
async fn main() -> mem0::Result<()> {
    // In-process: mock providers + embedded store. No API keys, no services.
    let mem = mem0::from_json(r#"{
        "embedder":     { "provider": "mock" },
        "llm":          { "provider": "mock" },
        "vector_store": { "provider": "embedded", "config": { "path": "/tmp/mem0" } },
        "history_db_path": ":memory:"
    }"#)?;

    // Store a raw memory (infer=false skips the LLM extraction step).
    mem.add("I love hiking in the mountains", AddOptions {
        user_id: Some("alice".into()),
        infer: Some(false),
        ..Default::default()
    }).await?;

    // Search within alice's scope.
    let mut filters = JsonMap::new();
    filters.insert("user_id".into(), json!("alice"));
    let results = mem.search("hiking", &filters, SearchOptions::default()).await?;
    println!("{results}");

    Ok(())
}
```

Run the bundled version:

```bash
cargo run -p mem0 --example quickstart
```

### Using a real provider

Swap the `mock` providers for real ones (here, OpenAI for both extraction and
embeddings) and turn on inference so the LLM extracts durable facts:

```rust
let mem = mem0::from_json(r#"{
    "embedder": { "provider": "openai", "config": { "api_key": "sk-..." } },
    "llm":      { "provider": "openai", "config": { "api_key": "sk-..." } },
    "vector_store": { "provider": "embedded", "config": { "path": "/var/lib/mem0" } }
}"#)?;

mem.add(
    vec![
        mem0::Message::user("Hi, I'm Alex. I'm vegetarian and I love trail running."),
        mem0::Message::assistant("Nice to meet you, Alex!"),
    ],
    mem0::AddOptions { user_id: Some("alex".into()), ..Default::default() }, // infer defaults to true
).await?;
```

See [providers/llms.md](providers/llms.md) and
[providers/embedders.md](providers/embedders.md) for every provider and option,
and [concepts.md](concepts.md) for what "inferred" vs "raw" means.

## Synchronous (blocking) API

If you are not in an async context, use `mem0::blocking::Memory`, which owns its
own Tokio runtime:

```rust
let mem = mem0::blocking::Memory::from_json(r#"{ "embedder": {"provider":"mock"},
    "llm": {"provider":"mock"}, "vector_store": {"provider":"embedded"},
    "history_db_path": ":memory:" }"#)?;

mem.add("I prefer aisle seats", mem0::AddOptions {
    user_id: Some("alice".into()), infer: Some(false), ..Default::default() })?;
let all = mem.get_all(&{ let mut f = mem0::JsonMap::new();
    f.insert("user_id".into(), serde_json::json!("alice")); f }, 10)?;
```

## Run the server

```bash
export MEM0_CONFIG='{
  "embedder":     {"provider":"openai","config":{"api_key":"sk-..."}},
  "llm":          {"provider":"openai","config":{"api_key":"sk-..."}},
  "vector_store": {"provider":"embedded","config":{"path":"/var/lib/mem0"}}
}'
./target/release/mem0-server               # 0.0.0.0:8080
```

```bash
curl -s localhost:8080/health
curl -s localhost:8080/v1/memories -H 'content-type: application/json' \
  -d '{"messages":"My name is Alex and I love trail running","user_id":"alex"}'
curl -s "localhost:8080/v1/memories?user_id=alex"
```

Full server docs: [server.md](server.md). Full REST reference:
[api/rest.md](api/rest.md).

## Where to next

- [Core concepts](concepts.md) — the mental model.
- [Developer guide](developer-guide.md) — the complete library workflow.
- [Configuration reference](configuration.md) — every knob.
- [Agent integrations](agents/overview.md) — give an existing agent memory.