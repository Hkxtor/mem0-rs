# Architecture

mem0-rs is a Cargo **workspace** of focused crates. A provider-agnostic core
(`mem0-core`) defines the data model, the memory pipeline, and a set of
**traits**; sibling crates implement those traits for concrete providers; a
facade crate (`mem0`) wires everything together; and thin crates expose the
result as a server, client, and proxy.

## Crate map

```
                ┌─────────────────────────────────────────────┐
                │                  mem0-core                   │
                │  config · types · prompts · nlp · scoring ·  │
                │  filters · history(SQLite) · Memory          │
                │  traits: Embedder · Llm · VectorStore ·      │
                │          Reranker                            │
                └───────────────▲───────────────▲─────────────┘
                                │ implements      │ uses
       ┌──────────────┬─────────┴────┬───────────┴───┬───────────────┐
       │ mem0-embed   │ mem0-llm     │ mem0-vector   │ mem0-rerank    │
       │ OpenAI/Azure │ OpenAI/Azure │ embedded /    │ Cohere /       │
       │ /Gemini/     │ /Anthropic/  │ qdrant /      │ ZeroEntropy /  │
       │ Ollama/mock  │ Gemini/      │ pgvector      │ LLM            │
       │              │ Ollama/mock  │               │                │
       └──────┬───────┴──────┬───────┴──────┬────────┴───────┬────────┘
              │              │              │                │
              └──────────────┴───────┬──────┴────────────────┘
                                     │ wired by factories
                              ┌──────▼───────┐
                              │     mem0     │  facade: from_config / from_json
                              │  (+ blocking)│  + blocking::Memory
                              └──────▲───────┘
                  ┌──────────────────┼───────────────────┐
            ┌─────┴──────┐    ┌──────┴───────┐     ┌──────┴───────┐
            │ mem0-server│    │ mem0-client  │     │ mem0-proxy   │
            │ axum REST  │    │ reqwest HTTP │     │ chat + memory│
            └────────────┘    └──────────────┘     └──────────────┘
```

| Crate | Responsibility |
|-------|----------------|
| `mem0-core` | The orchestrator and everything provider-agnostic: config model, types, verbatim prompts, dependency-free NLP, BM25 scoring, filter/scope logic, the SQLite history store, and the four provider **traits**. Contains in-crate mocks + an in-memory store for tests. |
| `mem0-embed` | `Embedder` implementations (OpenAI, OpenAI-compatible, Azure, Gemini, Ollama, mock) + `build_embedder` factory. |
| `mem0-llm` | `Llm` implementations (OpenAI, OpenAI-compatible, Azure, Anthropic, Gemini, Ollama, mock) + reasoning-model parameter handling + `build_llm`. |
| `mem0-vector` | `VectorStore` implementations: `embedded` (default), `qdrant` (feature), `pgvector` (feature) + `build_vector_store`. |
| `mem0-rerank` | `Reranker` implementations (Cohere, ZeroEntropy, LLM) + `build_reranker`. |
| `mem0` | Facade: builds providers from config via the factories and constructs `Memory`; provides `blocking::Memory`. |
| `mem0-server` | axum router + handlers + env-config bootstrap + the `mem0-server` binary. |
| `mem0-client` | reqwest HTTP client mirroring the REST API. |
| `mem0-proxy` | Memory-augmented chat proxy (retrieve → augment → generate → persist). |
| `xtask` | Workspace automation: `ci` (fmt+clippy+test) and `gen-prompts` (codegen prompt constants from the Python source). |

## The trait abstraction

`mem0-core` depends on no provider. It talks to four async traits
(`mem0_core::traits`):

```rust
#[async_trait] pub trait Embedder {
    async fn embed(&self, text: &str, action: MemoryAction) -> Result<Vec<f32>>;
    async fn embed_batch(&self, texts: &[String], action: MemoryAction) -> Result<Vec<Vec<f32>>>;
    fn dims(&self) -> usize;
}
#[async_trait] pub trait Llm {
    async fn generate(&self, messages: &[Message], options: &GenerateOptions) -> Result<String>;
}
#[async_trait] pub trait VectorStore {
    async fn insert(&self, records: Vec<VectorRecord>) -> Result<()>;
    async fn search(&self, query: &str, vector: &[f32], top_k: usize, filters: &JsonMap) -> Result<Vec<SearchHit>>;
    async fn get(&self, id: &str) -> Result<Option<SearchHit>>;
    async fn update(&self, id: &str, vector: Option<Vec<f32>>, payload: Option<JsonMap>) -> Result<()>;
    async fn delete(&self, id: &str) -> Result<()>;
    async fn list(&self, filters: &JsonMap, limit: Option<usize>) -> Result<Vec<SearchHit>>;
    async fn delete_col(&self) -> Result<()>;
    async fn reset(&self) -> Result<()>;
    async fn keyword_search(&self, query: &str, top_k: usize, filters: &JsonMap) -> Result<Option<Vec<SearchHit>>>; // default: None
}
#[async_trait] pub trait Reranker {
    async fn rerank(&self, query: &str, documents: &[String], top_n: usize) -> Result<Vec<(usize, f32)>>;
}
```

`Memory` holds `Box<dyn Embedder>`, `Box<dyn Llm>`, `Box<dyn VectorStore>`, an
optional `Box<dyn Reranker>`, an optional entity `VectorStore`, and a
`HistoryStore` (rusqlite). Because the dependency points *inward* (providers
depend on `mem0-core`, not vice-versa), you can add a provider without touching
the core, and you can unit-test the entire pipeline with the in-crate mocks.

## How the facade wires it

`mem0::from_config(config)`:

1. builds the embedder and reads its `dims()`;
2. injects `embedding_model_dims` into the vector-store config (so fixed-dim
   backends like qdrant/pgvector match the embedder);
3. builds the LLM, the vector store, and (optionally) the reranker;
4. builds a best-effort **entity store** as a sibling collection
   (`<collection>_entities`);
5. constructs `Memory` and attaches the entity store.

`from_json(&str)` is `from_config(MemoryConfig::from_json(s)?)`.

## Data flow

- **add (inferred):** messages → existing-memory search → LLM extraction
  (verbatim prompt) → parse → batch-embed → md5 dedup → vector insert → SQLite
  history (`ADD`) → best-effort entity linking → save recent messages.
- **add (raw):** each message → embed → vector insert → SQLite history (`ADD`).
- **search:** query → embed → vector `search` + `keyword_search` (BM25) +
  entity boosts → `score_and_rank` → ranked memories (+ optional rerank).
- **update / delete:** vector `update`/`delete` + SQLite history (`UPDATE`/`DELETE`).

Step-by-step internals with the exact scoring math: [how-it-works.md](how-it-works.md).

## Storage model

- **Vectors + payload** live in the vector store. The payload carries `data`
  (the memory text), `hash`, `created_at`/`updated_at`, scope ids,
  `text_lemmatized` (for BM25), and any user metadata.
- **History + recent messages** live in a SQLite database (`history_db_path`,
  default `~/.mem0/history.db`, or `:memory:`). The history table records every
  `ADD`/`UPDATE`/`DELETE`; a messages table keeps a small recent-message buffer
  per scope.

## Design decisions

- **`embedded` vector store is the default** (Python mem0 defaults to Qdrant).
  This makes a default `Memory` runnable with zero external services and enables
  the single-binary story. Scale-out backends are feature-gated.
- **Dependency-free NLP.** Instead of spaCy, mem0-core ships a deterministic
  lemmatizer (for BM25) and PROPER/QUOTED entity extraction. This keeps the
  binary self-contained; the trade-off (no `COMPOUND`/`NOUN` entities) is
  documented in [comparison.md](comparison.md).
- **Async-native traits with a blocking wrapper.** Python mem0 duplicates
  sync/async classes; Rust is async-first, with `blocking::Memory` owning a
  Tokio runtime for sync callers.
- **`rustls` everywhere.** No system OpenSSL dependency; clean static builds on
  macOS and Linux.
- **Verbatim prompts via codegen.** The extraction/update prompts are generated
  byte-for-byte from the Python source by `xtask gen-prompts`, guaranteeing
  extraction-quality parity (see [quality.md](quality.md)).
- **No telemetry.** PostHog is intentionally omitted.

## Build & feature flags

| Flag | Crate(s) | Effect |
|------|----------|--------|
| `qdrant` | `mem0-vector`, `mem0-server`, `mem0` | Enable the Qdrant REST backend |
| `pgvector` | `mem0-vector`, `mem0-server`, `mem0` | Enable the Postgres/pgvector backend |

```bash
cargo build --release -p mem0-server                      # embedded only
cargo build --release -p mem0-server --features qdrant
cargo build --release -p mem0-server --features pgvector
```

Next: [How it works](how-it-works.md).