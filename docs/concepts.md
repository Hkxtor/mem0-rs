# Core concepts

This page explains the mental model behind mem0-rs. It applies equally to the
library, the server, and the proxy.

## A memory

A **memory** is a short, self-contained factual statement derived from a
conversation — e.g. *"User is vegetarian"* or *"User's favorite restaurant is
Osteria Francescana"*. Each stored memory has:

- a stable **id** (UUID),
- the **memory** text (`data` in the payload),
- a **hash** (md5 of the text, used for deduplication),
- **created_at** / **updated_at** timestamps (UTC, RFC 3339),
- the **scope** ids it belongs to (`user_id` / `agent_id` / `run_id`),
- optional **actor_id** / **role**, **metadata**, and an internal
  **text_lemmatized** field used for BM25 keyword search.

When returned by the API, a memory looks like:

```json
{
  "id": "f1c2…",
  "memory": "User is vegetarian",
  "hash": "…",
  "created_at": "2026-06-03T10:00:00+00:00",
  "updated_at": "2026-06-03T10:00:00+00:00",
  "user_id": "alex",
  "score": 0.83,                // only on search results
  "metadata": { "category": "diet" }
}
```

## Scoping: user, agent, run

Every memory belongs to one or more scopes:

- **`user_id`** — the end user the memory is about.
- **`agent_id`** — the agent/assistant identity (useful for multi-agent systems
  and agent-self memories).
- **`run_id`** — a single session/run/thread.

At least one scope id is **required** for `add`, `get_all`, `search`, and
`delete_all`. Scopes are applied both as stored metadata and as query filters, so
memories never leak across users. See
[Filtering](#filtering-and-metadata) for combining scopes with metadata.

## Inferred vs raw add

`add` has two modes, controlled by `infer`:

- **`infer: true`** (default) — the **additive-extraction pipeline**. mem0-rs
  sends the messages (plus related existing memories) to your LLM with mem0's
  verbatim extraction prompt, which returns a list of durable facts. Those facts
  — not the raw text — are stored.
- **`infer: false`** — **raw** mode. Each non-system message's content is stored
  verbatim as a memory. No LLM call. Useful for importing known facts, logs, or
  pre-computed memories, and for deterministic tests.

```rust
// Inferred: the LLM extracts facts like "User is vegetarian".
mem.add(messages, AddOptions { user_id: Some("alex".into()), ..Default::default() }).await?;

// Raw: stores the string as-is.
mem.add("User is vegetarian", AddOptions {
    user_id: Some("alex".into()), infer: Some(false), ..Default::default() }).await?;
```

## Additive extraction & deduplication

The inferred pipeline is **additive** (ADD-only): it extracts new facts and adds
them, rather than rewriting the whole memory set on every turn. To avoid
duplicates it computes the **md5 hash** of each extracted fact and skips any that
match an existing memory in scope or a duplicate within the same batch. So
re-stating a known fact ("My name is still Alex") yields **zero** new memories.

Explicit `update` and `delete` operations exist for changing or removing
specific memories by id.

## Hybrid retrieval (search)

`search` combines three signals and ranks the result:

1. **Semantic** — cosine similarity between the query embedding and memory
   embeddings.
2. **BM25 keyword** — lexical match over the lemmatized memory text (sigmoid-
   normalized).
3. **Entity boost** — a best-effort bump for memories linked to entities found
   in the query.

The combined score uses an adaptive divisor (semantic = 1.0; +1.0 if BM25 is
available; +0.5 if entity boosts are present) and a **threshold** that gates the
semantic score *before* combining. The internals and exact math are in
[how-it-works.md](how-it-works.md).

`SearchOptions` controls retrieval:

- `top_k` (default 20) — maximum results.
- `threshold` (default 0.1) — minimum semantic score.
- `rerank` (default false) — apply the configured reranker to reorder results.

## Filtering and metadata

You can attach arbitrary **metadata** to memories and filter on it at query time
with operators:

- equality (`{"category": "food"}`),
- comparison (`{"score": {"gte": 0.5}}` — `eq/ne/gt/gte/lt/lte`),
- membership (`{"category": {"in": ["food","travel"]}}` — `in/nin`),
- substring (`{"text": {"contains": "pizza"}}` — `contains/icontains`),
- existence (`{"category": "*"}`),
- logical groups (`AND` / `OR` / `NOT`).

Scope ids (`user_id`/`agent_id`/`run_id`) are just well-known filters. Full
syntax: [Configuration → filtering](configuration.md#metadata-filtering) and
[api/library.md](api/library.md).

## Memory types

- **Default (semantic) memory** — the facts produced by the extraction pipeline.
- **Procedural memory** — a single summarized record of *how* an agent
  accomplished a task, produced by passing `memory_type: "procedural_memory"` to
  `add` with an `agent_id`. It uses a dedicated summarization prompt.

## Entity linking

During an inferred `add`, mem0-rs extracts named entities (proper nouns and
quoted titles) from the new memories and links related memories through a
sibling entity collection. This is **best-effort and never fatal** — if it
fails, the memories are still stored. Entity links also feed the entity-boost
signal in search. (mem0-rs implements the dependency-free PROPER/QUOTED subset of
mem0's spaCy entity extraction; see [comparison.md](comparison.md).)

## History & audit

Every `ADD`, `UPDATE`, and `DELETE` is recorded in a SQLite history table with
old/new text, timestamps, and actor/role. Retrieve it with `history(id)`:

```json
{ "history": [
  { "event": "ADD",    "new_memory": "User is vegetarian", "created_at": "…" },
  { "event": "UPDATE", "old_memory": "User is vegetarian",
    "new_memory": "User is vegan", "updated_at": "…" }
]}
```

A recent-messages buffer is also kept per scope to give the extraction pipeline
short-term conversational context.

## Lifecycle summary

| Operation | Purpose |
|-----------|---------|
| `add` | Extract+store (inferred) or store raw memories |
| `search` | Hybrid retrieve ranked memories |
| `get` | Fetch one memory by id |
| `get_all` | List memories in a scope |
| `update` | Replace a memory's text/metadata |
| `delete` | Remove one memory |
| `delete_all` | Remove all memories in a scope |
| `history` | Audit trail for a memory |
| `reset` | Wipe all memories and history |

Next: [Developer guide](developer-guide.md) or [How it works](how-it-works.md).