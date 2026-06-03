# Fully local memory with Ollama

Run the entire memory stack — extraction LLM, embeddings, and vector store —
**locally with no cloud calls**, ideal for privacy-sensitive or air-gapped
deployments. This pairs [Ollama](https://ollama.com) with the in-process
`embedded` store.

## 1. Install models in Ollama

```bash
ollama pull llama3.1:8b           # extraction LLM
ollama pull nomic-embed-text      # embeddings
# ollama serve    # if not already running on :11434
```

## 2. Configure mem0-rs fully local

```json
{
  "llm":      { "provider": "ollama", "config": { "model": "llama3.1:8b",
                "ollama_base_url": "http://localhost:11434" } },
  "embedder": { "provider": "ollama", "config": { "model": "nomic-embed-text",
                "ollama_base_url": "http://localhost:11434" } },
  "vector_store": { "provider": "embedded", "config": { "path": "/var/lib/mem0" } },
  "history_db_path": "/var/lib/mem0/history.db"
}
```

No API keys, no outbound internet. The default Ollama endpoint is
`http://localhost:11434`; the embedder uses `/api/embed` and the LLM `/api/chat`.

## 3. Run the service

```bash
export MEM0_CONFIG_FILE=/etc/mem0/local.json
./target/release/mem0-server
```

## 4. Use it

The agent loop is identical to the cloud guides — just point your agent at the
local memory service:

```bash
curl -s localhost:8080/v1/memories -H 'content-type: application/json' \
  -d '{"messages":"I am vegetarian and allergic to peanuts","user_id":"alex"}'
curl -s localhost:8080/v1/memories/search -H 'content-type: application/json' \
  -d '{"query":"dietary restrictions","user_id":"alex"}'
```

Or, in Rust, use the [proxy](../proxy.md) with `llm.provider = "ollama"`.

## Privacy & operational notes

- **No data leaves the host.** Extraction, embeddings, vectors, and history all
  stay local (Ollama on localhost + the `embedded` store + local SQLite).
- **Footprint.** The `embedded` store keeps memories in process (optionally
  persisted to `path`); combine with mem0-rs's small RSS for a lightweight,
  self-contained memory node.
- **Model choice.** Smaller instruct models (`llama3.1:8b`, `qwen2.5:7b`) extract
  facts well and keep latency low; embeddings with `nomic-embed-text` (768-dim)
  or any Ollama embedding model work out of the box (set `embedding_dims` if your
  model differs and use a fresh store path).
- **Scaling.** For multi-node local clusters, swap the `embedded` store for a
  self-hosted `qdrant` (still no cloud) — see
  [providers/vector-stores.md](../providers/vector-stores.md).

This is the most private configuration: a single local binary plus a local model
server, with memory that never touches a third-party API.

See also: [Agent overview](overview.md) · [Security](../security.md).