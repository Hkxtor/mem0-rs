# Configuration reference

mem0-rs is configured with a single JSON object (the `MemoryConfig`). The same
schema is used by the library (`from_json` / `from_config`), the server
(`MEM0_CONFIG` / `MEM0_CONFIG_FILE`), and the proxy.

## Top-level schema

```json
{
  "embedder":      { "provider": "openai",   "config": { /* … */ } },
  "llm":           { "provider": "openai",   "config": { /* … */ } },
  "vector_store":  { "provider": "embedded", "config": { /* … */ } },
  "reranker":      { "provider": "llm",      "config": { /* … */ } },
  "history_db_path": "/var/lib/mem0/history.db",
  "version": "v1.1",
  "custom_instructions": null
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `embedder` | `{provider, config}` | `{ "openai", {} }` | Text→vector provider. |
| `llm` | `{provider, config}` | `{ "openai", {} }` | Fact-extraction / chat model. |
| `vector_store` | `{provider, config}` | `{ "embedded", {} }` | Where vectors + payloads live. |
| `reranker` | `{provider, config}` or omitted | omitted | Optional result reranker. |
| `history_db_path` | string | `~/.mem0/history.db` | SQLite file (or `:memory:`). |
| `version` | string | `"v1.1"` | API version marker. |
| `custom_instructions` | string or null | null | Extra extraction instructions. |

Every provider block is `{ "provider": "<name>", "config": { … } }`. Unknown
keys inside `config` are ignored, so configs are forward-compatible.

### Defaults & minimal configs

A bare `{}` is valid and yields: OpenAI embedder + OpenAI LLM + `embedded` store
+ no reranker + `~/.mem0/history.db`. The smallest fully-offline config:

```json
{ "embedder": {"provider":"mock"}, "llm": {"provider":"mock"},
  "vector_store": {"provider":"embedded"}, "history_db_path": ":memory:" }
```

## Providers at a glance

Full per-provider options are in the provider pages:

- [Embedders](providers/embedders.md) — `openai`, `openai_compatible`/`lmstudio`/`together`/`vllm`, `azure_openai`, `gemini`, `ollama`, `mock`.
- [LLMs](providers/llms.md) — `openai`, `openai_compatible`/`groq`/`xai`/`deepseek`/`together`/`vllm`/`lmstudio`/`minimax`/`sarvam`, `azure_openai`, `anthropic`, `gemini`, `ollama`, `mock`.
- [Vector stores](providers/vector-stores.md) — `embedded` (default), `qdrant`, `pgvector`.
- [Rerankers](providers/rerankers.md) — `llm`, `cohere`, `zero_entropy`.

## Environment variables

### Server bootstrap

| Variable | Purpose |
|----------|---------|
| `MEM0_CONFIG_FILE` | Path to a JSON config file (highest priority). |
| `MEM0_CONFIG` | Inline JSON config string. |
| `PORT` | Server listen port (default `8080`). |
| `RUST_LOG` | `tracing` filter (e.g. `info`, `mem0_server=debug`). |
| `MEM0_DIR` | Overrides the `~/.mem0` home used for the default history path. |

Resolution order for the server: `MEM0_CONFIG_FILE` → `MEM0_CONFIG` →
built-in default.

### Provider credentials (read when not set in `config`)

| Variable | Used by |
|----------|---------|
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | OpenAI embedder & LLM |
| `ANTHROPIC_API_KEY` | Anthropic LLM |
| `GOOGLE_API_KEY` | Gemini embedder & LLM |
| `LLM_AZURE_OPENAI_API_KEY`, `LLM_AZURE_ENDPOINT`, `LLM_AZURE_DEPLOYMENT`, `LLM_AZURE_API_VERSION` | Azure LLM |
| `EMBEDDING_AZURE_OPENAI_API_KEY`, `EMBEDDING_AZURE_ENDPOINT`, `EMBEDDING_AZURE_DEPLOYMENT`, `EMBEDDING_AZURE_API_VERSION` | Azure embedder |
| `COHERE_API_KEY` | Cohere reranker |
| `ZERO_ENTROPY_API_KEY` | ZeroEntropy reranker |

Explicit `config` values always take precedence over environment variables.
Prefer environment variables (or a file with restricted permissions) for secrets
rather than inline JSON in process listings — see [security.md](security.md).

## Per-operation options

These are passed at call time, not in the config.

### `AddOptions`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `user_id` / `agent_id` / `run_id` | `Option<String>` | none | Scope (≥1 required). |
| `metadata` | `Option<JsonMap>` | none | Stored with each memory; filterable. |
| `infer` | `Option<bool>` | `true` | `true` = LLM extraction; `false` = raw. |
| `memory_type` | `Option<String>` | none | `"procedural_memory"` for procedural. |
| `prompt` | `Option<String>` | none | Override extraction instructions for this call. |

### `SearchOptions`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `top_k` | `usize` | `20` | Max results. |
| `threshold` | `f64` | `0.1` | Min semantic score (0–1). |
| `rerank` | `bool` | `false` | Apply the configured reranker. |

## Metadata filtering

Filters are JSON objects passed to `search`/`get_all` (and as the `filters`
field in the REST search body). Scope ids are ordinary filters.

| Form | Example | Meaning |
|------|---------|---------|
| equality | `{"category": "food"}` | exact match |
| `eq` / `ne` | `{"category": {"eq": "food"}}` | equal / not equal |
| `gt` / `gte` / `lt` / `lte` | `{"score": {"gte": 0.5}}` | numeric comparison |
| `in` / `nin` | `{"category": {"in": ["food","travel"]}}` | membership / exclusion |
| `contains` / `icontains` | `{"data": {"contains": "pizza"}}` | substring (case-insensitive variant) |
| wildcard | `{"category": "*"}` | field exists |
| `AND` / `OR` / `NOT` | `{"AND": [{"user_id":"u1"}, {"category":"food"}]}` | logical groups (lists of filter objects) |

Example combining a scope with an operator filter:

```json
{ "user_id": "alex", "category": { "in": ["food", "travel"] } }
```

> Operator support across backends: the `embedded` store and `qdrant` implement
> the full operator set; `pgvector` implements equality, comparison, membership,
> substring, existence, and `OR`/`NOT` groups. See
> [providers/vector-stores.md](providers/vector-stores.md).

## Config recipes

**Local-only (Ollama + embedded), no cloud:**
```json
{
  "embedder": {"provider":"ollama","config":{"model":"nomic-embed-text"}},
  "llm":      {"provider":"ollama","config":{"model":"llama3.1:8b"}},
  "vector_store": {"provider":"embedded","config":{"path":"/var/lib/mem0"}}
}
```

**OpenAI + Qdrant (scaled), with LLM reranking:**
```json
{
  "embedder": {"provider":"openai","config":{"model":"text-embedding-3-small"}},
  "llm":      {"provider":"openai","config":{"model":"gpt-4o-mini"}},
  "vector_store": {"provider":"qdrant","config":{"url":"http://qdrant:6333","collection_name":"mem0"}},
  "reranker": {"provider":"llm","config":{}}
}
```

**Postgres/pgvector:**
```json
{
  "embedder": {"provider":"openai","config":{}},
  "llm":      {"provider":"openai","config":{}},
  "vector_store": {"provider":"pgvector","config":{
    "connection_string":"postgresql://user:pass@db:5432/mem0","collection_name":"mem0"}}
}
```

Next: provider details ([Embedders](providers/embedders.md) ·
[LLMs](providers/llms.md) · [Vector stores](providers/vector-stores.md) ·
[Rerankers](providers/rerankers.md)).