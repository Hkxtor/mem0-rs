# mem0-rs — long-term memory for AI agents, in Rust

**mem0-rs** gives your AI agents durable, searchable long-term memory — the
ability to remember user preferences, facts, and past interactions across
sessions — as a **single, self-contained, memory-safe binary** that runs
anywhere (macOS, Linux) with **no Python runtime and no external services
required**.

It is a faithful Rust port of [mem0](https://github.com/mem0ai/mem0) v2.0.4: the
same extraction prompts (byte-for-byte), the same hybrid retrieval and
deduplication behavior, the same REST surface — re-implemented for speed,
footprint, and operational simplicity.

```text
   user/agent conversation ──▶  mem0-rs  ──▶  extract facts ──▶ store (vectors + SQLite)
                                    ▲                                     │
   next conversation ──────────────┴──────── search + inject ◀───────────┘
```

---

## Why mem0-rs

- **Speed.** On an equal, network-free workload it is **~3–3.5× faster per
  operation** than Python mem0 (no interpreter, no GIL, no per-call pydantic
  validation or `deepcopy`, monomorphized scoring math). See
  [docs/performance.md](docs/performance.md).
- **Low latency & footprint.** **~6.5× lower peak memory** (~9–15 MB RSS vs
  ~100 MB) and a single **~12 MB binary**. Cheap to run at density, instant to
  deploy.
- **Security & safety.** Memory-safe Rust; `rustls` TLS (no system OpenSSL);
  parameterized SQL; no telemetry/phone-home; secrets only ever read from the
  environment or your config. See [docs/security.md](docs/security.md).
- **Scalability.** Start fully in-process with the built-in `embedded` store
  (zero dependencies), then scale out to **Qdrant** or **Postgres/pgvector**
  with a one-line config change — no code changes. See
  [docs/providers/vector-stores.md](docs/providers/vector-stores.md).
- **Drop-in fidelity.** Same memory *quality* as Python mem0 — proven by
  byte-identical prompts and a 100% deterministic equivalence harness. See
  [docs/quality.md](docs/quality.md).

## What it does

mem0-rs sits between your agent and an LLM/embedding provider and turns raw
conversations into a curated, queryable memory:

- **`add`** — extract durable facts from messages (LLM-inferred) or store raw
  text, with automatic md5 **deduplication**.
- **`search`** — hybrid retrieval (semantic vectors + BM25 keyword + entity
  boosts) with advanced metadata filtering and optional reranking.
- **`get` / `get_all` / `update` / `delete` / `delete_all` / `history` /
  `reset`** — full lifecycle management, with an audit trail.
- **Scoping** by `user_id`, `agent_id`, and `run_id`; **procedural memory** for
  agent workflows; best-effort **entity linking**.

It ships three ways to use it: a **Rust library**, a **REST server** (single
binary) with a matching **HTTP client**, and a **memory-augmented chat proxy**.

## How it works (in one paragraph)

On an inferred `add`, mem0-rs retrieves existing related memories, sends them
plus the new messages to your LLM with mem0's verbatim additive-extraction
prompt, parses the returned facts, deduplicates them by md5 against existing
memories, embeds them, and writes them to the vector store with a lemmatized
field for BM25 — recording every change in a SQLite history table. On `search`,
it embeds the query, runs semantic + BM25 + entity-boost scoring, gates by a
threshold, and returns the ranked memories. Full internals:
[docs/how-it-works.md](docs/how-it-works.md).

---

## Quickstart (library, no API keys)

```rust
use mem0::{AddOptions, SearchOptions, JsonMap};
use serde_json::json;

#[tokio::main]
async fn main() -> mem0::Result<()> {
    // Fully in-process: mock providers + embedded store.
    let mem = mem0::from_json(r#"{
        "embedder":     { "provider": "mock" },
        "llm":          { "provider": "mock" },
        "vector_store": { "provider": "embedded", "config": { "path": "/tmp/mem0" } },
        "history_db_path": ":memory:"
    }"#)?;

    mem.add("I love hiking", AddOptions {
        user_id: Some("alice".into()), infer: Some(false), ..Default::default()
    }).await?;

    let mut filters = JsonMap::new();
    filters.insert("user_id".into(), json!("alice"));
    let hits = mem.search("hiking", &filters, SearchOptions::default()).await?;
    println!("{hits}");
    Ok(())
}
```

```bash
cargo run -p mem0 --example quickstart
```

A synchronous API is available via `mem0::blocking::Memory`. Full guide:
[docs/getting-started.md](docs/getting-started.md).

## Quickstart (server, single binary)

```bash
cargo build --release -p mem0-server

export MEM0_CONFIG='{
  "embedder":     {"provider":"openai","config":{"api_key":"sk-..."}},
  "llm":          {"provider":"openai","config":{"api_key":"sk-..."}},
  "vector_store": {"provider":"embedded","config":{"path":"/var/lib/mem0"}}
}'
./target/release/mem0-server          # listens on 0.0.0.0:8080 (override with $PORT)
```

```bash
# add a memory
curl -s localhost:8080/v1/memories -H 'content-type: application/json' -d '{
  "messages":"My name is Alex and I love trail running","user_id":"alex"}'
# search
curl -s localhost:8080/v1/memories/search -H 'content-type: application/json' -d '{
  "query":"what sports does the user like?","user_id":"alex"}'
```

Full REST reference: [docs/api/rest.md](docs/api/rest.md).

## Give an agent memory

mem0-rs works with any agent built on OpenAI/Codex, Anthropic/Claude, Gemini, or
local models — configure that model family as the memory LLM/embedder and wire
`search` (before the turn) + `add` (after the turn) into your agent loop, or use
the proxy to do it automatically.

- [Agent integration overview](docs/agents/overview.md)
- [OpenAI / Codex](docs/agents/openai-codex.md) ·
  [Anthropic / Claude](docs/agents/claude.md) ·
  [Gemini](docs/agents/gemini.md) ·
  [Local / Ollama](docs/agents/local-ollama.md)

---

## Documentation

| Area | Docs |
|------|------|
| **Start here** | [Why mem0-rs](docs/why.md) · [Getting started](docs/getting-started.md) · [Core concepts](docs/concepts.md) |
| **Using it** | [Developer guide (library)](docs/developer-guide.md) · [Server](docs/server.md) · [Proxy](docs/proxy.md) |
| **APIs** | [Rust library API](docs/api/library.md) · [REST API](docs/api/rest.md) · [HTTP client](docs/api/client.md) |
| **Configuration** | [Configuration reference](docs/configuration.md) · Embedders ([↗](docs/providers/embedders.md)) · LLMs ([↗](docs/providers/llms.md)) · Vector stores ([↗](docs/providers/vector-stores.md)) · Rerankers ([↗](docs/providers/rerankers.md)) |
| **Internals** | [Architecture](docs/architecture.md) · [How it works](docs/how-it-works.md) |
| **Agents** | [Overview](docs/agents/overview.md) · [OpenAI/Codex](docs/agents/openai-codex.md) · [Claude](docs/agents/claude.md) · [Gemini](docs/agents/gemini.md) · [Local/Ollama](docs/agents/local-ollama.md) |
| **Operations** | [DevOps & deployment](docs/devops.md) · [Security](docs/security.md) |
| **Reference** | [Performance](docs/performance.md) · [Quality parity](docs/quality.md) · [Migration from Python mem0](docs/migration-from-python.md) · [Comparison matrix](docs/comparison.md) · [Troubleshooting](docs/troubleshooting.md) · [FAQ](docs/faq.md) |

The complete index lives in [docs/README.md](docs/README.md).

## Project layout

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

## Development

```bash
cargo test --workspace                       # 148 tests
cargo clippy --all-targets -- -D warnings
cargo fmt --all --check
cargo run -p xtask -- ci                     # runs all three
```

See [docs/developer-guide.md](docs/developer-guide.md) and
[docs/architecture.md](docs/architecture.md).

## Fidelity & license

Ported faithfully from mem0 v2.0.4: pipeline control flow, prompts (verbatim),
md5 dedup, payload schema, SQLite schema/queries, BM25 scoring math, filter/scope
construction, and the error taxonomy. Adapted for Rust: a dependency-free NLP
layer (deterministic lemmatizer + PROPER/QUOTED entity extraction in place of
spaCy), async-native provider traits (with a blocking wrapper), `rustls` HTTP,
and a curated provider set. Telemetry (PostHog) is intentionally omitted. See
[docs/comparison.md](docs/comparison.md) and
[docs/quality.md](docs/quality.md).

Licensed under Apache-2.0, matching upstream mem0.