# Embedders

The embedder turns text into vectors for semantic search. Configure it with
`"embedder": { "provider": "<name>", "config": { … } }`. All HTTP providers use
`rustls` TLS.

Common config fields (provider-agnostic; unknown fields ignored):

| Field | Type | Notes |
|-------|------|-------|
| `model` | string | Embedding model id. |
| `api_key` | string | Falls back to a provider env var. |
| `embedding_dims` | int | When set, requested explicitly from the API and used as the store dimension. |
| `openai_base_url` | string | OpenAI / OpenAI-compatible endpoint. |
| `ollama_base_url` | string | Ollama endpoint. |
| `base_url` | string | Generic endpoint override. |
| `output_dimensionality` | int | Gemini output dims (alias of `embedding_dims`). |
| `azure_kwargs` | object | Azure connection (see Azure below). |

> The embedder's dimensionality is injected into the vector-store config
> automatically by the facade, so fixed-dimension backends (qdrant/pgvector)
> match without you repeating it.

## OpenAI — `openai`

- Default model: `text-embedding-3-small`; default dims `1536` (only sent to the
  API when you set `embedding_dims`).
- Endpoint: `https://api.openai.com/v1` (or `openai_base_url` / `OPENAI_BASE_URL`).
- Key: `api_key` or `OPENAI_API_KEY`.

```json
{ "provider": "openai", "config": { "model": "text-embedding-3-small", "api_key": "sk-..." } }
```

## OpenAI-compatible — `openai_compatible` / `openai_like` / `lmstudio` / `together` / `vllm`

For any server that speaks the OpenAI `/embeddings` API (LM Studio, vLLM,
Together, etc.). **Requires** `openai_base_url` (or `base_url`).

```json
{ "provider": "vllm", "config": { "openai_base_url": "http://localhost:8000/v1", "model": "BAAI/bge-small-en" } }
```

## Azure OpenAI — `azure_openai` / `azure`

Uses `azure_kwargs` (or `EMBEDDING_AZURE_*` env vars). API-key auth.

```json
{ "provider": "azure_openai", "config": { "azure_kwargs": {
    "api_key": "...", "azure_deployment": "my-embed-deploy",
    "azure_endpoint": "https://my.openai.azure.com", "api_version": "2024-02-01" } } }
```

| `azure_kwargs` field | Env fallback |
|----------------------|--------------|
| `api_key` | `EMBEDDING_AZURE_OPENAI_API_KEY` |
| `azure_deployment` | `EMBEDDING_AZURE_DEPLOYMENT` |
| `azure_endpoint` | `EMBEDDING_AZURE_ENDPOINT` |
| `api_version` | `EMBEDDING_AZURE_API_VERSION` (default `2024-02-01`) |
| `default_headers` | — |

## Google Gemini — `gemini`

- Default model: `models/gemini-embedding-001`; default dims `768` (or
  `embedding_dims` / `output_dimensionality`).
- Endpoint: `https://generativelanguage.googleapis.com` (or `base_url`).
- Key: `api_key` or `GOOGLE_API_KEY`.

```json
{ "provider": "gemini", "config": { "api_key": "...", "output_dimensionality": 768 } }
```

## Ollama (local) — `ollama`

- Default model: `nomic-embed-text`; default dims `512`.
- Endpoint: `http://localhost:11434` (or `ollama_base_url` / `base_url`),
  `/api/embed`.

```json
{ "provider": "ollama", "config": { "model": "nomic-embed-text", "ollama_base_url": "http://localhost:11434" } }
```

## Mock — `mock`

A deterministic, fixed-size embedder for tests and offline demos (no network).
Use with the `embedded` store for a fully in-process setup.

```json
{ "provider": "mock" }
```

## Choosing dimensions

If you switch embedding models (different dimensionality), use a fresh
collection / store path — vectors of different dimensions cannot share a
collection. Set `embedding_dims` to match your model when using non-default
sizes.

Back to [Configuration](../configuration.md).