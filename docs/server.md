# Server

`mem0-server` is a single, self-contained HTTP binary that exposes the
[REST API](api/rest.md). It is the language-agnostic way to give any
application or agent long-term memory.

## Build

```bash
cargo build --release -p mem0-server
# -> target/release/mem0-server   (~12 MB; no Python, no external runtime)

# with a scale-out vector backend:
cargo build --release -p mem0-server --features qdrant
cargo build --release -p mem0-server --features pgvector
```

## Configure

The server reads its `MemoryConfig` from the environment, in this order:

1. **`MEM0_CONFIG_FILE`** — path to a JSON config file (recommended for secrets).
2. **`MEM0_CONFIG`** — inline JSON config string.
3. **Default** — OpenAI embedder + OpenAI LLM + `embedded` store + history at
   `~/.mem0/history.db`.

```bash
# Option A: a file (keep it chmod 600; avoids secrets in process listings)
cat > /etc/mem0/config.json <<'JSON'
{
  "embedder":     {"provider":"openai","config":{"api_key":"sk-..."}},
  "llm":          {"provider":"openai","config":{"api_key":"sk-..."}},
  "vector_store": {"provider":"embedded","config":{"path":"/var/lib/mem0"}}
}
JSON
export MEM0_CONFIG_FILE=/etc/mem0/config.json

# Option B: inline
export MEM0_CONFIG='{"embedder":{"provider":"mock"},"llm":{"provider":"mock"},
                     "vector_store":{"provider":"embedded"},"history_db_path":":memory:"}'

./target/release/mem0-server
```

Provider credentials can also come from their standard env vars
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, …) instead of inline
config — see [configuration.md](configuration.md#environment-variables).

## Runtime environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `PORT` | `8080` | Listen port (binds `0.0.0.0:$PORT`). |
| `RUST_LOG` | `info` | `tracing` filter, e.g. `RUST_LOG=mem0_server=debug,info`. |
| `MEM0_CONFIG_FILE` / `MEM0_CONFIG` | — | Config source (above). |
| `MEM0_DIR` | `~/.mem0` | Home dir for the default history path. |

## Verify

```bash
curl -s localhost:8080/health                       # {"status":"ok"}
curl -s localhost:8080/v1/memories -H 'content-type: application/json' \
  -d '{"messages":"My name is Alex","user_id":"alex"}'
curl -s "localhost:8080/v1/memories?user_id=alex"
```

## Embedding in your own server

`mem0-server` is also a library — build the router around your own `Arc<Memory>`
and mount it in a larger axum app:

```rust
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let memory = Arc::new(mem0::from_json(/* config */)?);
    let app = mem0_server::app(memory);          // axum::Router
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    axum::serve(listener, app).await?;
    Ok(())
}
```

`mem0_server::build_memory_from_env()` reproduces the binary's env-based config
resolution if you want it.

## Operational notes

- **Statelessness.** The server keeps no state beyond the configured vector
  store and SQLite history. With a shared `qdrant`/`pgvector` backend you can run
  multiple replicas behind a load balancer.
- **Persistence.** Use a file-backed `embedded` store (`path`) or an external
  backend; with `history_db_path` pointing at a real file, history survives
  restarts. `:memory:` is ephemeral.
- **Auth/TLS.** There is no built-in auth; put a reverse proxy / API gateway in
  front for TLS termination, authentication, and rate limiting. See
  [security.md](security.md) and [devops.md](devops.md).

Next: [DevOps & deployment](devops.md) · [REST API](api/rest.md).