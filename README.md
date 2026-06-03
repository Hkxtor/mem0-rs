# mem0-rs

A Rust port of [mem0](https://github.com/mem0ai/mem0) — long-term memory for AI
agents — designed to compile to a **single, dependency-light, cross-platform
binary** (macOS + Linux). It replaces the Python implementation's runtime
footprint with a memory-safe, statically-linked alternative while preserving the
core behavior of mem0 v2.0.4.

> Why Rust? Lower memory use, no Python runtime, compile-time safety, and a
> single self-contained binary (the default `embedded` vector store needs no
> external services). HTTP uses `rustls` (no system OpenSSL), so the binary
> builds and links cleanly on macOS and Linux.

## What's implemented

The additive-extraction memory pipeline of mem0 v2.0.4:

- `add` (raw and LLM-inferred additive extraction with md5 dedup),
- `search` (semantic + BM25 hybrid + entity boosts + advanced metadata filters + optional reranking),
- `get` / `get_all` / `update` / `delete` / `delete_all` / `history` / `reset`,
- procedural memory, best-effort entity linking,
- a SQLite history/messages store, and the verbatim mem0 prompts (codegen'd from the Python source).

See [`../mem0-rust-port/PORTING_ANALYSIS.md`](../mem0-rust-port/PORTING_ANALYSIS.md)
for the architecture mapping and [`REVIEW_LOG.md`](REVIEW_LOG.md) for the
per-milestone adversarial reviews.

## Workspace layout

| Crate | Purpose |
|-------|---------|
| `mem0-core` | Config, types, prompts, NLP, scoring, filters, SQLite history, provider **traits**, and the `Memory` orchestrator |
| `mem0-embed` | Embedders: OpenAI, OpenAI-compatible, Azure, Gemini, Ollama, mock |
| `mem0-llm` | LLMs: OpenAI, OpenAI-compatible, Azure, Anthropic, Gemini, Ollama, mock |
| `mem0-vector` | Vector stores: `embedded` (default), `qdrant` (feature), `pgvector` (feature) |
| `mem0-rerank` | Rerankers: Cohere, ZeroEntropy, LLM |
| `mem0` | Facade: `from_config` / `from_json` wiring + `blocking::Memory` |
| `mem0-server` | axum REST API (library + `mem0-server` binary) |
| `mem0-client` | reqwest HTTP client for the server |
| `mem0-proxy` | Memory-augmented chat proxy |
| `xtask` | Workspace automation (codegen, CI helpers) |

## Quickstart (library)

```rust
use mem0::{AddOptions, SearchOptions, JsonMap};
use serde_json::json;

#[tokio::main]
async fn main() -> mem0::Result<()> {
    // Fully in-process: mock providers + embedded store, no API keys.
    let mem = mem0::from_json(r#"{
        "embedder": { "provider": "mock" },
        "llm": { "provider": "mock" },
        "vector_store": { "provider": "embedded", "config": { "path": "/tmp/mem0" } }
    }"#)?;

    mem.add("I love hiking", AddOptions { user_id: Some("alice".into()),
        infer: Some(false), ..Default::default() }).await?;

    let mut filters = JsonMap::new();
    filters.insert("user_id".into(), json!("alice"));
    let hits = mem.search("hiking", &filters, SearchOptions::default()).await?;
    println!("{hits}");
    Ok(())
}
```

Run the bundled example:

```bash
cargo run -p mem0 --example quickstart
```

A synchronous API is available via `mem0::blocking::Memory`.

## Server (single binary)

```bash
# Build the self-contained server binary
cargo build --release -p mem0-server

# Configure providers (inline JSON or a file), then run
export MEM0_CONFIG='{"embedder":{"provider":"openai","config":{"api_key":"sk-..."}},
                     "llm":{"provider":"openai","config":{"api_key":"sk-..."}},
                     "vector_store":{"provider":"embedded","config":{"path":"/var/lib/mem0"}}}'
./target/release/mem0-server      # listens on 0.0.0.0:8080 (override with $PORT)
```

REST endpoints: `POST/GET/DELETE /v1/memories`, `POST /v1/memories/search`,
`GET/PUT/DELETE /v1/memories/:id`, `GET /v1/memories/:id/history`,
`POST /v1/reset`, `GET /health`. The `mem0-client` crate mirrors these.

## Configuration

```json
{
  "embedder":     { "provider": "openai",   "config": { "api_key": "..." } },
  "llm":          { "provider": "openai",   "config": { "api_key": "..." } },
  "vector_store": { "provider": "embedded", "config": { "path": "/var/lib/mem0" } },
  "reranker":     { "provider": "llm",      "config": {} },
  "history_db_path": "/var/lib/mem0/history.db"
}
```

Optional external vector backends are feature-gated:

```bash
cargo build -p mem0-server --features qdrant     # Qdrant (REST)
cargo build -p mem0-server --features pgvector   # Postgres + pgvector
```

## Development

```bash
cargo test --workspace
cargo clippy --all-targets -- -D warnings
cargo fmt --all --check
cargo run -p xtask -- ci          # runs all three
cargo run -p xtask -- gen-prompts # regenerate prompt constants from the Python source
```

## Fidelity notes

Ported faithfully: pipeline control flow, prompts (verbatim), md5 hash dedup,
payload schema, SQLite schema/queries, BM25 scoring math, filter/scope
construction, and the error taxonomy. Adapted for Rust: a dependency-free NLP
layer (deterministic lemmatizer + PROPER/QUOTED entity extraction in place of
spaCy), async-native provider traits (with a blocking wrapper), `rustls` HTTP,
and a curated provider set. Telemetry (PostHog) is intentionally omitted.