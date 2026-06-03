# HTTP client (`mem0-client`)

`mem0-client` is a thin reqwest client that mirrors the [REST API](rest.md). Every
method returns the server's JSON response as `serde_json::Value`, or a
`mem0_core::Mem0Error` on transport / non-2xx responses (404 → `NotFound`,
400 → `Validation`, else `Configuration`).

## Construct

```rust
use mem0_client::Mem0Client;
let client = Mem0Client::new("http://localhost:8080");   // trailing slash trimmed
```

## Methods

```rust
// Add — request is the raw JSON body.
client.add(&serde_json::json!({
    "messages": "I love trail running",
    "user_id": "alex"
})).await?;

// Get one by id.
let memory = client.get("<id>").await?;

// List a scope (user_id, agent_id, run_id, top_k — all optional but ≥1 scope needed server-side).
let all = client.get_all(Some("alex"), None, None, Some(50)).await?;

// Search — request is the raw JSON body.
let results = client.search(&serde_json::json!({
    "query": "what does the user like?",
    "user_id": "alex",
    "top_k": 5
})).await?;

// Update text (+ optional metadata).
client.update("<id>", "User is now vegan", None).await?;

// Delete one.
client.delete("<id>").await?;

// History.
let history = client.history("<id>").await?;   // { "history": [ … ] }

// Reset everything.
client.reset().await?;
```

## Method reference

| Method | Maps to |
|--------|---------|
| `add(&Value)` | `POST /v1/memories` |
| `get(&str)` | `GET /v1/memories/:id` |
| `get_all(user, agent, run, top_k)` | `GET /v1/memories` |
| `search(&Value)` | `POST /v1/memories/search` |
| `update(id, data, Option<Value>)` | `PUT /v1/memories/:id` |
| `delete(&str)` | `DELETE /v1/memories/:id` |
| `history(&str)` | `GET /v1/memories/:id/history` |
| `reset()` | `POST /v1/reset` |

## Example: end-to-end

```rust
use mem0_client::Mem0Client;
use serde_json::json;

#[tokio::main]
async fn main() -> mem0_core::Result<()> {
    let client = Mem0Client::new("http://localhost:8080");
    client.add(&json!({"messages":"I'm vegetarian","user_id":"alex","infer":false})).await?;
    let hits = client.search(&json!({"query":"diet","user_id":"alex"})).await?;
    println!("{hits}");
    Ok(())
}
```

The request/response shapes are exactly those documented in the
[REST API reference](rest.md). For non-Rust clients, call the REST endpoints
directly with any HTTP library.