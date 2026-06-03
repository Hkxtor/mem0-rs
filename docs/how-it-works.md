# How it works (internals)

This is the deep-dive companion to [architecture.md](architecture.md). It traces
the actual code paths in `mem0-core::Memory` and the helper modules.

## The `add` pipeline

`Memory::add(messages, opts)` first builds scope metadata and effective query
filters from `user_id`/`agent_id`/`run_id` (at least one is required), validates
`memory_type`, then branches:

### Raw (`infer = false`)

For each non-`system` message, store the content verbatim:

1. attach `role` (and `actor_id` from the message `name`) to the payload;
2. `create_memory`: embed the text, compute `hash = md5(text)`, set
   `created_at = updated_at` (UTC RFC 3339), compute `text_lemmatized`, insert
   into the vector store, and write an `ADD` history row.

### Inferred (`infer = true`) — additive extraction

This is the V3 ADD-only pipeline, in phases:

1. **Context.** Load the recent-message buffer for the scope from SQLite
   (`get_last_messages`, key derived from the sorted scope ids) and flatten the
   new messages into a transcript.
2. **Existing-memory retrieval.** Embed the transcript and semantic-search the
   vector store (top 10, scoped) to gather related existing memories; map them to
   integer ids (`{id, text}`) so the LLM cannot hallucinate UUIDs.
3. **LLM extraction.** Build the system prompt (`ADDITIVE_EXTRACTION_PROMPT`,
   plus `AGENT_CONTEXT_SUFFIX` for agent-scoped adds) and a user prompt from
   `generate_additive_extraction_prompt(...)`, then call the LLM with
   `response_format = json_object`. Custom instructions (from `opts.prompt` or
   `config.custom_instructions`) are threaded in.
4. **Parse.** Parse `{"memory": [{text, attributed_to?, linked_memory_ids?}]}`,
   falling back through `remove_code_blocks` → `extract_json` for chatty/
   markdown-wrapped responses. LLM errors are **non-fatal** (logged → returns no
   new memories).
5. **Batch embed.** `embed_batch` the extracted texts (per-item fallback).
6. **Deduplicate.** Compute `md5(text)` and skip any that match an existing
   payload hash (gathered in phase 2) or a duplicate within the same batch.
7. **Persist.** Build payloads (`data`, `text_lemmatized`, `hash`, timestamps,
   scope ids, `attributed_to`) and batch-insert; the pipeline tracks which
   records *actually* persisted and bases history/links/return on those.
8. **History.** Batch-write `ADD` rows (per-item fallback).
9. **Entity linking.** Best-effort: extract PROPER/QUOTED entities from the new
   memories, embed them, and link them through the entity collection
   (≥0.95 similarity merges into an existing entity, else inserts a new one).
   Never fatal.
10. **Persist messages.** Save the new messages to the recent-message buffer and
    return `[{id, memory, event: "ADD"}]`.

### Procedural memory

`add(..., memory_type = "procedural_memory")` with an `agent_id` runs a single
LLM summarization (`PROCEDURAL_MEMORY_SYSTEM_PROMPT`) over the conversation and
stores one summary memory.

## The `search` pipeline

`Memory::search(query, filters, options)`:

1. Validate `threshold ∈ [0,1]` and `top_k > 0`; require a scope id; normalize
   any advanced operator filters (`process_metadata_filters`).
2. `search_vector_store`:
   - lemmatize the query and extract query entities;
   - embed the query; semantic-search the store with an internal over-fetch
     (`max(top_k*4, 60)`);
   - if the store supports it, run `keyword_search` over `text_lemmatized` to get
     **BM25** scores, normalized by a query-length-adaptive sigmoid;
   - compute **entity boosts** from the entity store (if present);
   - combine and rank with `score_and_rank`, dropping memories whose `data` is
     empty.
3. If `rerank = true` and a reranker is configured, reorder the top results with
   it.

### Hybrid scoring math

**BM25 normalization** (`scoring::normalize_bm25`): a logistic sigmoid maps the
unbounded raw BM25 score to `[0,1]`:

```text
norm = 1 / (1 + exp(-steepness * (raw - midpoint)))
```

`(midpoint, steepness)` are chosen by lemmatized query length
(`get_bm25_params`): `≤3 terms → (5.0, 0.7)`, `≤6 → (7.0, 0.6)`,
`≤9 → (9.0, 0.5)`, `≤15 → (10.0, 0.5)`, else `(12.0, 0.5)`.

**Combination** (`scoring::score_and_rank`): the `threshold` gates the **semantic**
score *before* combining. The combined score uses an adaptive divisor so the
result stays in `[0,1]` regardless of which signals are active:

```text
max_possible = 1.0
             + 1.0   (if any BM25 scores present)
             + 0.5   (ENTITY_BOOST_WEIGHT, if any entity boosts present)

combined = min( (semantic + bm25 + entity_boost) / max_possible , 1.0 )
```

Results are sorted by `combined` descending and truncated to `top_k`. This is a
verbatim port of mem0's `score_and_rank`; the exact numbers are covered by
`port_scoring.rs` (mirroring mem0's `tests/utils/test_scoring.py`).

## The NLP layer (dependency-free)

`mem0-core::nlp` replaces spaCy with deterministic, dependency-free functions:

- **`lemmatize_for_bm25`** — lowercases, drops a curated stopword set and
  punctuation, applies light suffix normalization (plurals, `-ing`, `-ed`), and
  keeps original `-ing` forms alongside their lemmas. Used **symmetrically** at
  add-time (stored in `text_lemmatized`) and search-time so BM25 matching is
  consistent. Always returns a value.
- **`extract_entities`** — extracts PROPER (capitalized sequences) and QUOTED
  (single/double-quoted spans) entities, deduplicated case-insensitively
  (PROPER preferred over QUOTED). The spaCy `COMPOUND`/`NOUN` cases (which need a
  dependency parser) are intentionally omitted; entity linking is best-effort and
  non-fatal, matching mem0's behavior when spaCy is absent.

## Deduplication

Every stored memory carries `hash = md5(data)`. The inferred pipeline dedups new
extractions against existing in-scope hashes and within the batch; raw adds store
as-is. This is the same md5 mechanism as Python mem0 (covered by
`m1_pipeline::infer_add_dedups_by_hash`).

## History & messages store (SQLite)

`mem0-core::history::HistoryStore` (rusqlite, bundled SQLite):

- **history** table — `id, memory_id, old_memory, new_memory, event,
  created_at, updated_at, is_deleted, actor_id, role`. Reads are ordered
  `created_at ASC, DATETIME(updated_at) ASC` (verbatim from mem0). A schema
  migration upgrades legacy tables in place.
- **messages** table — a per-scope recent-message buffer (most recent 10 kept),
  feeding short-term context into the extraction pipeline.

Set `history_db_path` to `:memory:` for an ephemeral store (tests, stateless
workers) or a file path for persistence.

## The prompt system

The extraction/update/procedural prompts are **verbatim** copies of mem0's,
generated into `mem0-core/src/prompts/constants.rs` by
`xtask gen-prompts` (which loads `mem0/mem0/configs/prompts.py` and emits raw
Rust string constants). `verify_prompts.py` confirms byte-equality. The dynamic
builders (`generate_additive_extraction_prompt`, `get_update_memory_messages`)
are ported in `prompts/mod.rs`. Because the prompts are identical, extraction
quality is identical given the same LLM — see [quality.md](quality.md).

## Error model

All operations return `Result<T, Mem0Error>`. `Mem0Error` carries a
machine-readable `code` (e.g. `VALIDATION_001`, `MEM_404`, `LLM_001`) and an
optional suggestion. The REST server maps these to HTTP status codes (see
[api/rest.md](api/rest.md)).

Next: [Configuration reference](configuration.md).