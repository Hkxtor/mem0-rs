# Migration from Python mem0

mem0-rs mirrors mem0 v2.0.4's behavior, config shape, and operations, so
migrating is mostly mechanical. This page maps the concepts and calls.

## Config mapping

The config shape is the same nested `{ provider, config }` structure. Key
differences:

| Aspect | Python mem0 | mem0-rs |
|--------|-------------|---------|
| Default vector store | `qdrant` | **`embedded`** (in-process, zero deps) |
| Config input | Python dict / `MemoryConfig` | JSON (`from_json`) or `MemoryConfig` |
| Telemetry | PostHog (opt-out) | **none** |
| NLP | spaCy (optional) | built-in (dependency-free) |

A Python config translates directly to JSON:

```python
# Python
config = {
  "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
  "llm":      {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
  "vector_store": {"provider": "qdrant", "config": {"url": "http://localhost:6333"}},
}
m = Memory.from_config(config)
```

```rust
// mem0-rs
let mem = mem0::from_json(r#"{
  "embedder": {"provider":"openai","config":{"model":"text-embedding-3-small"}},
  "llm":      {"provider":"openai","config":{"model":"gpt-4o-mini"}},
  "vector_store": {"provider":"qdrant","config":{"url":"http://localhost:6333"}}
}"#)?;   // build with --features qdrant
```

## API mapping

| Python mem0 | mem0-rs (library) | REST |
|-------------|-------------------|------|
| `m.add(messages, user_id=…, infer=…, metadata=…)` | `mem.add(messages, AddOptions{ user_id, infer, metadata, … })` | `POST /v1/memories` |
| `m.search(query, user_id=…, filters=…, limit=…)` | `mem.search(query, &filters, SearchOptions{ top_k, threshold, rerank })` | `POST /v1/memories/search` |
| `m.get(id)` | `mem.get(id)` | `GET /v1/memories/:id` |
| `m.get_all(user_id=…)` | `mem.get_all(&filters, top_k)` | `GET /v1/memories?user_id=…` |
| `m.update(id, data=…)` | `mem.update(id, data, metadata)` | `PUT /v1/memories/:id` |
| `m.delete(id)` | `mem.delete(id)` | `DELETE /v1/memories/:id` |
| `m.delete_all(user_id=…)` | `mem.delete_all(user, agent, run)` | `DELETE /v1/memories?user_id=…` |
| `m.history(id)` | `mem.history(id)` | `GET /v1/memories/:id/history` |
| `m.reset()` | `mem.reset()` | `POST /v1/reset` |

Notes:

- In Rust, scope ids and per-call options are passed via `AddOptions` /
  `SearchOptions` and an explicit `filters` map rather than keyword arguments.
- `limit`/`top_k` is `SearchOptions.top_k` (default 20); `threshold` defaults to
  0.1; `rerank` is opt-in per search.
- Result shapes match: `add` returns `{results:[{id,memory,event,…}]}`; `search`
  / `get_all` return `{results:[…]}`.

## Behavioral parity

The extraction pipeline, prompts, md5 deduplication, hybrid scoring, filtering,
and history semantics are ported faithfully and validated as identical — see
[quality.md](quality.md). You should observe the same memories and rankings given
the same model.

## Differences to know

1. **Default store is `embedded`.** If you relied on Qdrant by default, set
   `vector_store.provider` explicitly and build with `--features qdrant`.
2. **Provider set is curated.** Embedders: OpenAI/compatible, Azure, Gemini,
   Ollama, mock. LLMs: OpenAI/compatible, Azure, Anthropic, Gemini, Ollama, mock.
   Vector stores: embedded, qdrant, pgvector. Rerankers: llm, cohere,
   zero_entropy. Other Python backends are not ported — see
   [comparison.md](comparison.md).
3. **Entity extraction is PROPER/QUOTED only** (no spaCy `COMPOUND`/`NOUN`); a
   secondary-signal divergence documented in [quality.md](quality.md).
4. **No telemetry.** Nothing to opt out of.
5. **Async-first.** Use the async API or `mem0::blocking::Memory` for sync code.
6. **No data migration.** mem0-rs does not import an existing Python mem0 store;
   re-ingest your source data, or point both at the same external Qdrant/pgvector
   collection (same embedding model/dimensions) to share vectors.
7. **Platform/cloud client and OpenMemory MCP server are out of scope** — this is
   an OSS-only engine + server. Integrate agents via the REST API
   ([agents/overview.md](agents/overview.md)).

## Migration checklist

- [ ] Translate the config dict to JSON; set `vector_store` explicitly.
- [ ] Choose providers from the supported set; build with needed features.
- [ ] Replace keyword args with `AddOptions`/`SearchOptions` + `filters`.
- [ ] Re-ingest data, or share an external vector backend with matching dims.
- [ ] Validate with `/health`, a sample `add`, and a `search`.

See also: [Comparison matrix](comparison.md) · [Getting started](getting-started.md).