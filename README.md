# mem0-rs

mem0-rs gives AI agents long-term memory without a Python runtime. It is a Rust port of [mem0](https://github.com/mem0ai/mem0) v2.0.4 that distills conversations into short factual memories, stores them per user, agent or run, and finds the relevant ones again before the next model call, so an assistant can still know that someone is vegetarian weeks after they mentioned it. It is for people building agents and chat products who want what mem0 does but would rather link a Rust library or deploy one native binary than keep a Python process running beside their application.

The port keeps the parts that decide what gets remembered and replaces the parts that tie it to Python. mem0's extraction prompts are copied byte for byte and its pipeline, hybrid scoring and history schema are carried across, while spaCy gives way to a small built-in lemmatizer and entity extractor and the default vector store becomes an in-process one instead of Qdrant. What it does not replace is the models: extraction calls an LLM and every add and search calls an embedding model, which can be a hosted API or a local server such as Ollama. It covers mem0's core memory operations with a smaller set of providers, and [the differences](#how-closely-it-follows-mem0) are set out below.

## How it works

When you add a conversation, mem0-rs embeds it, retrieves the ten most similar memories in that scope along with the ten most recent stored messages, and sends all of that to your LLM with mem0's extraction prompt. Each fact the model returns is hashed with MD5 so that exact repeats of those ten memories, or of each other, are dropped, and the rest are embedded, written to the vector store and recorded in a SQLite history table. Extraction only ever adds, so if someone moves from Paris to Berlin both facts stay stored until you update or delete the old one. Passing `infer: false` skips the model and the hash check and stores each message as it is.

Search embeds the query and fetches a generous set of candidates from the vector store. A candidate must clear the semantic threshold, 0.1 by default, to be considered at all; BM25 keyword scores and entity matches then raise its rank rather than bringing in new results, and a configured reranker can reorder the final list on request.

## Run the server

You need current stable Rust and a C compiler, because SQLite is compiled into the binary, as well as access to an LLM and an embedding model. This example uses OpenAI, and [the Ollama configuration below](#providers-and-storage) keeps everything on your machine.

```bash
git clone https://github.com/chrischabot/mem0-rs
cd mem0-rs
cargo build --release -p mem0-server

mkdir -p mem0-data
export OPENAI_API_KEY=...
export MEM0_CONFIG='{
  "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
  "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
  "vector_store": {"provider": "embedded", "config": {"path": "mem0-data"}},
  "history_db_path": "mem0-data/history.db"
}'
./target/release/mem0-server
```

The data directory has to exist first, because SQLite will not create it and the server cannot start without its history database. The server listens on `0.0.0.0:8080`, and `PORT` changes the port. You can also put the JSON in a file and point `MEM0_CONFIG_FILE` at it.

Add a memory, then search for it:

```bash
curl -s localhost:8080/v1/memories -H 'content-type: application/json' -d '{
  "messages": "My name is Alex and I love trail running", "user_id": "alex"}'

curl -s localhost:8080/v1/memories/search -H 'content-type: application/json' -d '{
  "query": "what sports does the user like?", "user_id": "alex"}'
```

An empty `results` list from an inferred add can mean the model found nothing new worth keeping, but it can also mean the LLM call failed, because that failure is logged rather than returned.

The server has no authentication, TLS or rate limiting, and `POST /v1/reset` erases everything, so keep it on a private network or behind a gateway. [docs/security.md](docs/security.md) covers hardening.

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Liveness check |
| `POST /v1/memories` | Add from a string, one message or a list of messages |
| `GET /v1/memories?user_id=…` | List memories in a scope |
| `DELETE /v1/memories?user_id=…` | Delete all memories in a scope |
| `POST /v1/memories/search` | Search within a scope |
| `GET`, `PUT` or `DELETE /v1/memories/:id` | Read, update or delete one memory |
| `GET /v1/memories/:id/history` | Read a memory's change history |
| `POST /v1/reset` | Erase all memories and history |

Adding, listing and searching all require at least one of `user_id`, `agent_id` or `run_id`. Search also takes `top_k`, `threshold`, `rerank` and metadata `filters`, which support `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`, `nin`, `contains`, `icontains`, the `*` wildcard and `AND`, `OR` and `NOT` groups. The [REST reference](docs/api/rest.md) has the full request and response shapes.

## Use it as a Rust library

The crates are at version 0.1.0 and not on crates.io, and the `mem0` crate you will find there comes from a different repository, so depend on this one through Git:

```toml
[dependencies]
mem0 = { git = "https://github.com/chrischabot/mem0-rs" }
serde_json = "1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust
use mem0::{AddOptions, JsonMap, Message, SearchOptions};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all("mem0-data")?;
    // The OpenAI providers read OPENAI_API_KEY when the config carries no key.
    let mem = mem0::from_json(r#"{
        "llm": { "provider": "openai", "config": { "model": "gpt-4o-mini" } },
        "embedder": { "provider": "openai", "config": { "model": "text-embedding-3-small" } },
        "vector_store": { "provider": "embedded", "config": { "path": "mem0-data" } },
        "history_db_path": "mem0-data/history.db"
    }"#)?;

    mem.add(
        vec![
            Message::user("I'm Alex. I'm vegetarian and I love trail running."),
            Message::assistant("Nice to meet you, Alex!"),
        ],
        AddOptions { user_id: Some("alex".into()), ..Default::default() },
    )
    .await?;

    let mut filters = JsonMap::new();
    filters.insert("user_id".into(), json!("alex"));
    let memories = mem
        .search("what should I cook for Alex?", &filters, SearchOptions::default())
        .await?;
    println!("{memories}");
    Ok(())
}
```

`mem0::blocking::Memory` offers the same operations to synchronous code. `cargo run -p mem0 --example quickstart` runs without any keys by using mock providers, but the mock embedder returns the same vector for every input and the mock LLM extracts nothing, so it proves that your build works and not that recall does.

## Providers and storage

| Role | Providers |
|------|-----------|
| LLM | `openai`, `anthropic`, `gemini`, `azure_openai`, `ollama`, and OpenAI-compatible endpoints such as `groq`, `deepseek`, `xai`, `together` and `lmstudio` |
| Embedder | `openai`, `gemini`, `azure_openai`, `ollama`, and OpenAI-compatible endpoints such as `together` and `lmstudio` |
| Reranker | `llm`, `cohere`, `zero_entropy` |
| Vector store | `embedded`, `qdrant`, `pgvector` |

The [provider pages](docs/README.md#configuration) list their options, and OpenAI-compatible providers need an `openai_base_url`. With Ollama and the embedded store, nothing leaves your machine:

```json
{
  "llm": {"provider": "ollama", "config": {"model": "llama3.1:8b"}},
  "embedder": {"provider": "ollama", "config": {"model": "nomic-embed-text", "embedding_dims": 768}},
  "vector_store": {"provider": "embedded", "config": {"path": "mem0-data"}},
  "history_db_path": "mem0-data/history.db"
}
```

The choice of vector store affects both ranking and operations:

| Store | Enable with | Keyword signal | What to keep in mind |
|-------|-------------|----------------|----------------------|
| `embedded` | Default | BM25, computed in process | Holds every memory in process and persists only when `path` is set. Each search scans all records in scope and each change rewrites the whole JSON file, and a failed write is only logged, so it suits one process with a modest number of memories. |
| `qdrant` | `--features qdrant` | None | Ranking relies on semantic similarity and entity matches. |
| `pgvector` | `--features pgvector` | Postgres full-text rank | Connects without TLS, so it cannot reach a server that requires encrypted connections. |

The feature flags work on both `mem0-server` and the `mem0` crate. Qdrant and pgvector size their collection from the embedder's `embedding_dims`, which defaults to 1536 for OpenAI and 512 for Ollama, so set it whenever your model differs. Several server replicas can share one of these backends, although each keeps its own SQLite history.

If you need to honour deletion requests, note that deleting a memory leaves its text in the SQLite history, and any entity names drawn from it in the entity collection, until `reset`. On Qdrant and pgvector, `delete_all` also handles at most 100 memories per call, so repeat it until the scope is empty.

## Wiring it into an agent

The usual loop is to call `search` before each model call, inject what comes back, and call `add` with the exchange afterwards. Any language can do that over REST, and there are guides for [OpenAI and Codex](docs/agents/openai-codex.md), [Claude](docs/agents/claude.md), [Gemini](docs/agents/gemini.md) and [Ollama](docs/agents/local-ollama.md). mem0-rs ships no MCP server or editor plugin, so the integration point is the API, usually exposed to the agent as a pair of tools you define yourself.

In Rust, `mem0-client` is an HTTP client for the server and `mem0-proxy` wraps the whole loop in one call: `MemoryProxy::chat` retrieves memories for the latest user message, prepends them to it, calls the LLM and stores the exchange. Despite its name it is a library type, with no binary or network listener of its own.

## How closely it follows mem0

All five prompt constants are byte-identical to mem0 v2.0.4, and `bench/verify_prompts.py` checks them against an installed copy of the Python package. `bench/run_parity.sh` then drives both implementations through the same 17-step script of adds, searches, an operator filter, an update, a history read and a delete, using a deterministic hash embedder, scripted LLM replies and stub in-process stores, with BM25 and entity linking switched off on both sides. [docs/quality.md](docs/quality.md) records agreement on all 17 steps, which is good evidence that the orchestration matches but says nothing about real models, BM25 ranking, entity boosts, reranking or real vector stores. Treat mem0-rs as a close port of the core rather than a guaranteed drop-in, and test it against your own data if you are migrating.

The differences you are most likely to meet are these:

- BM25 tokens come from a suffix-stripping lemmatizer rather than spaCy, and entity extraction covers only capitalized and quoted phrases, so keyword and entity signals can rank results differently from a mem0 install with spaCy.
- REST paths differ, with `/v1/memories` and `/v1/memories/search` in place of `/memories` and `/search`, and there is no `/configure` endpoint, so clients of the Python server need changes.
- There are three vector stores and shorter lists of LLMs and embedders, with no graph memory, hosted platform client or multimodal input. [docs/comparison.md](docs/comparison.md) goes through them feature by feature.
- There is no importer for an existing Python mem0 store, so migrating means ingesting your data again. [docs/migration-from-python.md](docs/migration-from-python.md) maps the calls and config.
- mem0's PostHog telemetry is gone, and the only outbound calls are to the providers you configure.

## Performance

`bench/run_compare.sh` takes models and the network out of both implementations and times raw adds and searches against in-process stores. The run recorded in [docs/performance.md](docs/performance.md), with 2,000 adds and 500 searches on a 2 vCPU Linux machine, measured about 19 µs against Python's 65 µs per add, about 6 ms against 18 ms per search, and a peak of about 15 MB against 101 MB of memory, with a release binary of about 12 MB. The two sides use different stub embedders and stores, so read those ratios as a rough measure of framework overhead. In a real agent the model calls dominate latency, so the gain you will notice is the smaller and simpler deployment rather than faster responses.

## Development

```bash
cargo test --workspace
cargo clippy --all-targets -- -D warnings
cargo fmt --all --check
cargo run -p xtask -- ci                      # runs the three commands above
cargo test -p mem0-vector --features qdrant   # Qdrant client against a mock server
cargo bench -p mem0-core                      # also: cargo bench -p mem0
```

The test suite runs against mock HTTP servers and in-process stores rather than live services, and the pgvector backend has no tests of its own, so try your provider and store combination before depending on it.

The comparison harnesses need the Python package at the version this port tracks:

```bash
pip install mem0ai==2.0.4
bench/run_parity.sh
bench/run_compare.sh 2000 500
```

`cargo run -p xtask -- gen-prompts` regenerates the prompt constants from an upstream checkout, which it expects at `../mem0` beside this repository. [REVIEW_LOG.md](REVIEW_LOG.md) records what the review at each milestone of the port found and how it was resolved.

| Crate | Purpose |
|-------|---------|
| `mem0` | Entry point: `from_json`, `from_config` and `blocking::Memory` |
| `mem0-core` | The `Memory` orchestrator, prompts, scoring, filters, SQLite history and provider traits |
| `mem0-embed`, `mem0-llm`, `mem0-rerank`, `mem0-vector` | Provider implementations |
| `mem0-server` | The axum REST API, as a library and as the `mem0-server` binary |
| `mem0-client` | HTTP client for the server |
| `mem0-proxy` | The `MemoryProxy` chat wrapper |
| `xtask` | The `ci` and `gen-prompts` tasks |

The [documentation index](docs/README.md) links everything else, including the [architecture](docs/architecture.md), the [configuration reference](docs/configuration.md), the [Rust API](docs/api/library.md) and [deployment notes](docs/devops.md).

## Project background

- [mem0-rs project overview](https://chabot.dev/projects/#mem0-ports) on Chris Chabot’s site.
- [The prompt I was not allowed to improve](https://chabot.dev/blog/porting-agent-memory-means-porting-behavior/): preserving memory behaviour across Rust and TypeScript ports.

## Attribution and license

mem0-rs is an independent port of [mem0](https://github.com/mem0ai/mem0) from mem0ai, based on v2.0.4 of the Python package `mem0ai`. The memory pipeline, retrieval design and prompts are theirs, and the prompts ship here as verbatim copies generated from upstream's source; this repository contributes the Rust implementation, the in-process store and the built-in text processing.

Licensed under Apache-2.0 as declared in [Cargo.toml](Cargo.toml), the same license as upstream.
