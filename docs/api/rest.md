# REST API reference

The `mem0-server` binary exposes the memory operations over HTTP (axum). All
request and response bodies are JSON. See [server.md](../server.md) for running and
configuring it.

Base URL in examples: `http://localhost:8080`.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness check |
| `POST` | `/v1/memories` | Add memories |
| `GET` | `/v1/memories` | List memories in a scope |
| `DELETE` | `/v1/memories` | Delete all memories in a scope |
| `POST` | `/v1/memories/search` | Search memories |
| `GET` | `/v1/memories/:id` | Get one memory |
| `PUT` | `/v1/memories/:id` | Update a memory |
| `DELETE` | `/v1/memories/:id` | Delete a memory |
| `GET` | `/v1/memories/:id/history` | Memory change history |
| `POST` | `/v1/reset` | Reset all memories |

## `GET /health`

```bash
curl -s localhost:8080/health
# {"status":"ok"}
```

## `POST /v1/memories` — add

Body:

| Field | Type | Notes |
|-------|------|-------|
| `messages` | string \| object \| array | A string, one `{role,content,name?}`, or an array of them. |
| `user_id` / `agent_id` / `run_id` | string | Scope (≥1 required). |
| `metadata` | object | Stored with each memory. |
| `infer` | bool | Default `true` (LLM extraction); `false` = raw. |
| `memory_type` | string | `"procedural_memory"` for procedural. |
| `prompt` | string | Per-call extraction instructions. |

```bash
curl -s localhost:8080/v1/memories -H 'content-type: application/json' -d '{
  "messages": [
    {"role":"user","content":"I am Alex and I love trail running"},
    {"role":"assistant","content":"Nice to meet you, Alex!"}
  ],
  "user_id": "alex"
}'
```

Response: `{ "results": [ { "id": "…", "memory": "…", "event": "ADD", … } ] }`.

## `GET /v1/memories` — list

Query params: `user_id`, `agent_id`, `run_id` (≥1 required), `top_k`
(default 20).

```bash
curl -s "localhost:8080/v1/memories?user_id=alex&top_k=50"
# { "results": [ { "id":"…","memory":"…", … } ] }
```

## `POST /v1/memories/search` — search

Body:

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `query` | string | — | Required. |
| `user_id` / `agent_id` / `run_id` | string | — | Scope (≥1 required). |
| `filters` | object | — | Additional metadata filters (merged with scope). |
| `top_k` | int | 20 | Max results. |
| `threshold` | float | 0.1 | Min semantic score. |
| `rerank` | bool | false | Apply configured reranker. |

```bash
curl -s localhost:8080/v1/memories/search -H 'content-type: application/json' -d '{
  "query": "what sports does the user like?",
  "user_id": "alex",
  "filters": { "category": { "in": ["sports","hobby"] } },
  "top_k": 5
}'
# { "results": [ { "id":"…","memory":"…","score":0.83, … } ] }
```

## `GET /v1/memories/:id` — get one

```bash
curl -s localhost:8080/v1/memories/<id>
# { "id":"…","memory":"…","created_at":"…", … }   (404 if not found)
```

## `PUT /v1/memories/:id` — update

Body: `{ "data": "new text", "metadata": { … } }` (`metadata` optional).

```bash
curl -s -X PUT localhost:8080/v1/memories/<id> -H 'content-type: application/json' \
  -d '{"data":"User is now vegan"}'
# { "message": "Memory updated successfully!" }
```

## `DELETE /v1/memories/:id` — delete one

```bash
curl -s -X DELETE localhost:8080/v1/memories/<id>
# { "message": "Memory deleted successfully!" }
```

## `DELETE /v1/memories` — delete all in scope

Query params: `user_id`, `agent_id`, `run_id` (≥1 required).

```bash
curl -s -X DELETE "localhost:8080/v1/memories?user_id=alex"
# { "message": "Memories deleted successfully!" }
```

## `GET /v1/memories/:id/history`

```bash
curl -s localhost:8080/v1/memories/<id>/history
# { "history": [ { "event":"ADD","new_memory":"…","created_at":"…" },
#                { "event":"UPDATE","old_memory":"…","new_memory":"…", … } ] }
```

## `POST /v1/reset`

```bash
curl -s -X POST localhost:8080/v1/reset
# { "message": "All memories reset" }
```

## Errors

Errors return a JSON body and an HTTP status mapped from the error type:

```json
{ "error": "[VALIDATION_001] At least one of 'user_id', 'agent_id', or 'run_id' must be provided.",
  "code": "VALIDATION_001" }
```

| Error type | HTTP status |
|------------|-------------|
| Validation | `400 Bad Request` |
| Authentication | `401 Unauthorized` |
| Not found | `404 Not Found` |
| Rate limit | `429 Too Many Requests` |
| Everything else | `500 Internal Server Error` |

## Notes

- The server is stateless beyond its configured stores; run multiple instances
  behind a load balancer pointing at a shared Qdrant/pgvector for horizontal
  scale (see [devops.md](../devops.md)).
- There is no built-in authentication; terminate auth/TLS at a gateway in front
  of the server (see [security.md](../security.md)).

See also: [HTTP client](client.md) · [Server](../server.md).