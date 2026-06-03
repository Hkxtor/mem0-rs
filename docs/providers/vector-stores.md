# Vector stores

The vector store holds memory embeddings and payloads and serves semantic (and,
where supported, BM25 keyword) search. Configure with
`"vector_store": { "provider": "<name>", "config": { … } }`.

Common config fields:

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `collection_name` | string | `mem0` | Collection / table name. |
| `embedding_model_dims` | int | `1536` | Auto-injected from the embedder. |
| `path` | string | — | `embedded` JSON persistence directory. |

## `embedded` (default) — also `memory` / `in_memory`

In-process store with **native BM25 keyword search** over the lemmatized text and
the full metadata-operator filter set. No external services.

- **In-memory** when no `path` is set (great for tests / stateless workers).
- **File-persisted** when `path` is set: the collection is written to
  `{path}/{collection_name}.json` after each mutation and reloaded on startup.

```json
{ "provider": "embedded", "config": { "path": "/var/lib/mem0", "collection_name": "mem0" } }
```

Capabilities: cosine semantic search, BM25 keyword search, all metadata
operators (`eq/ne/gt/gte/lt/lte/in/nin/contains/icontains/*`, `AND/OR/NOT`).
This is the backend behind the single-binary deployment. For very large
collections or multi-instance sharing, use Qdrant or pgvector.

## Qdrant — `qdrant` (feature)

Build with `--features qdrant`. Talks to Qdrant over its REST API; the collection
is created lazily (Cosine distance, dimension from the embedder).

```json
{ "provider": "qdrant", "config": {
    "url": "http://localhost:6333", "api_key": "optional", "collection_name": "mem0" } }
```

| Field | Notes |
|-------|-------|
| `url` | Full Qdrant URL, **or**… |
| `host` + `port` | …host/port pair. |
| `api_key` | Sent as the `api-key` header when present. |
| `collection_name` | Default `mem0`. |
| `on_disk` | Reserved. |

Capabilities: cosine semantic search + the full metadata-operator filter set
(translated to Qdrant `must`/`should`/`must_not`). BM25 sparse search
(fastembed) is **out of scope** for this backend, so retrieval relies on semantic
scoring (plus entity boosts).

```bash
cargo build --release -p mem0-server --features qdrant
```

## Postgres / pgvector — `pgvector` (feature)

Build with `--features pgvector`. Stores vectors in a `vector(dims)` column and
provides Postgres full-text keyword search over the lemmatized field.

```json
{ "provider": "pgvector", "config": {
    "connection_string": "postgresql://user:pass@localhost:5432/mem0",
    "collection_name": "mem0", "hnsw": true } }
```

| Field | Default | Notes |
|-------|---------|-------|
| `connection_string` | — | Full DSN (overrides the parts below). |
| `dbname` / `user` / `password` / `host` / `port` | `postgres` / — / — / `localhost` / `5432` | Connection parts. |
| `collection_name` | `mem0` | Table name (validated `[A-Za-z_][A-Za-z0-9_]*`). |
| `hnsw` | `true` | Create an HNSW index. |
| `diskann` | `false` | Reserved. |

Capabilities: cosine semantic search (distance converted to similarity), Postgres
full-text keyword search, and metadata operators
(`eq/ne/gt/gte/lt/lte/in/nin/contains/icontains/*`, `OR`/`NOT`). Collection names
are validated and all filter values are bound parameters — see
[security.md](../security.md).

```bash
cargo build --release -p mem0-server --features pgvector
```

## Choosing a backend

| Need | Backend |
|------|---------|
| Zero infra, single node, edge | `embedded` |
| Shared store, horizontal scale, ANN at scale | `qdrant` |
| Memories alongside relational data / existing Postgres | `pgvector` |

Switching backends is a config change only — no application code changes. Note
that moving between backends does not migrate existing data; re-ingest or export
as needed.

Back to [Configuration](../configuration.md).