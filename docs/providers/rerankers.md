# Rerankers

A reranker is **optional**. When configured and enabled per-search
(`rerank: true`), it reorders the hybrid-retrieved results using a dedicated
relevance model. Configure with `"reranker": { "provider": "<name>", "config": { … } }`.

Rerankers return `(original_index, score)` pairs; mem0-rs applies them to reorder
the top results from `search`.

## LLM reranker — `llm` / `llm_reranker`

Scores each candidate with an LLM prompt (relevance `0.0`–`1.0`) and sorts by
score. Reuses any configured LLM provider via a nested `llm` block, or a
top-level provider.

```json
{ "provider": "llm", "config": {
    "llm": { "provider": "openai", "config": { "model": "gpt-4o-mini", "api_key": "sk-..." } } } }
```

| Field | Default | Notes |
|-------|---------|-------|
| `llm` | — | Nested `{provider, config}` for the scoring model. |
| `provider` | `openai` | Used when no nested `llm`. |
| `model` | `gpt-4o-mini` | Scoring model. |
| `temperature` | `0.0` | Deterministic scoring. |
| `max_tokens` | `100` | Short numeric replies. |
| `top_k` | — | Default result cap. |
| `api_key` | env | Inherited by the nested config if not set there. |

`config: null` is treated as defaults. Inputs are truncated to keep prompts
bounded; on a scoring error the candidate gets a neutral `0.5`.

## Cohere — `cohere`

Calls Cohere's `/v1/rerank`.

```json
{ "provider": "cohere", "config": { "model": "rerank-english-v3.0", "api_key": "..." } }
```

| Field | Default | Notes |
|-------|---------|-------|
| `model` | `rerank-english-v3.0` | Rerank model. |
| `api_key` | `COHERE_API_KEY` | Required. |
| `base_url` | `https://api.cohere.com` | Endpoint. |
| `top_k` | — | Default `top_n`. |

## ZeroEntropy — `zero_entropy`

Calls ZeroEntropy's `/v1/models/rerank` and sorts by relevance.

```json
{ "provider": "zero_entropy", "config": { "model": "zerank-1", "api_key": "..." } }
```

| Field | Default | Notes |
|-------|---------|-------|
| `model` | `zerank-1` | Rerank model. |
| `api_key` | `ZERO_ENTROPY_API_KEY` | Required. |
| `base_url` | `https://api.zeroentropy.dev` | Endpoint. |
| `top_k` | — | Default cap. |

## Using reranking

Reranking is opt-in **per search**:

```rust
let results = mem.search("best italian place I mentioned", &filters,
    SearchOptions { top_k: 10, rerank: true, ..Default::default() }).await?;
```

Over REST, set `"rerank": true` in the search body. If `rerank` is true but no
reranker is configured, search returns the hybrid-ranked results unchanged.

> Reranking adds an extra model call per search (and latency). Use it when
> precision at the very top matters more than latency; the hybrid
> semantic+BM25 ranking is already strong on its own.

Back to [Configuration](../configuration.md).