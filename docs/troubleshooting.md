# Troubleshooting

Common issues and how to resolve them. Errors carry a machine-readable `code`
(see [api/library.md](api/library.md#errors)).

## "At least one of 'user_id', 'agent_id', or 'run_id' must be provided" (`VALIDATION_001`)

Every `add`, `get_all`, `search`, and `delete_all` needs a scope. Pass at least
one id:

```rust
AddOptions { user_id: Some("alex".into()), ..Default::default() }
```
```bash
curl ... -d '{"messages":"…","user_id":"alex"}'
```

## "top_k must be a positive integer" / "threshold must be between 0.0 and 1.0" (`VALIDATION_004/005`)

Use `top_k > 0` and `0.0 ≤ threshold ≤ 1.0` in `SearchOptions` / the search body.

## "Memory with id … not found" (`MEM_404`)

The id doesn't exist (wrong id, different scope, or already deleted). List with
`get_all` to find valid ids. Over REST this is a `404`.

## "Unsupported … provider: X" (`CFG_001`)

The provider name is misspelled or not in the supported set. Check the exact
names in [configuration.md](configuration.md) and the provider pages. Aliases
exist (e.g. `azure` = `azure_openai`, `memory` = `embedded`).

## "OpenAI-compatible … requires 'openai_base_url'"

The `openai_compatible` / `groq` / `vllm` / `lmstudio` / `together` (and similar)
providers need an explicit endpoint:

```json
{ "provider": "vllm", "config": { "openai_base_url": "http://localhost:8000/v1" } }
```

## "qdrant/pgvector backend not enabled"

These backends are compile-time features. Rebuild with the flag:

```bash
cargo build --release -p mem0-server --features qdrant     # or --features pgvector
```

## Empty or wrong search results

- **Threshold too high.** Lower `threshold` (default 0.1) — memories below the
  semantic cutoff are dropped.
- **Wrong scope.** Ensure the `user_id`/`agent_id`/`run_id` matches what you
  stored.
- **Dimension mismatch / model change.** If you switched embedding models, the
  stored vectors have a different dimensionality; use a fresh collection / store
  `path`. The facade injects `embedding_model_dims` from the embedder, so don't
  mix models in one collection.
- **Inferred add returned nothing.** The LLM extracted no durable facts from
  chit-chat (this is expected), or the LLM call failed (check logs — extraction
  errors are non-fatal and logged). Verify the LLM `api_key`/endpoint.

## Authentication / provider errors (`LLM_001`, `EMBED_001`)

The provider rejected the request — usually a missing/invalid API key or wrong
endpoint. Set the key in `config.api_key` or the standard env var
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, …). Confirm the model
name is valid for your account.

## Server uses the wrong config

Resolution order is `MEM0_CONFIG_FILE` → `MEM0_CONFIG` → default. If a file is
set, it wins. Print effective behavior by hitting `/health` then a sample
`add`/`search`, and run with `RUST_LOG=mem0_server=debug,info` for detail.

## Server not reachable / wrong port

It binds `0.0.0.0:$PORT` (default 8080). Set `PORT` to change it; check that
nothing else owns the port and that your firewall/gateway forwards to it.

## Memories don't persist across restarts

- `embedded` store: set a `path` (it persists to `{path}/{collection}.json`);
  in-memory only when `path` is unset.
- History: set `history_db_path` to a file (not `:memory:`).
- Ensure the process can write to those paths (permissions / volume mounts).

## Ollama: connection refused

Ensure `ollama serve` is running and reachable at `ollama_base_url`
(default `http://localhost:11434`), and that the model is pulled
(`ollama pull llama3.1:8b`, `ollama pull nomic-embed-text`).

## Reasoning model rejects parameters

mem0-rs already omits `temperature`/`top_p`/`max_tokens` for detected reasoning
models (`o1`/`o3`/`gpt-5*`). If you use such a model under a custom name,
detection is by name — name it conventionally or set `reasoning_effort`.

## Build issues

- Ensure a C toolchain is present (bundled SQLite compiles via `rusqlite`).
- Use current stable Rust (`rustup update`).
- For clippy/fmt locally: `rustup component add clippy rustfmt`.

Still stuck? Capture `RUST_LOG=debug` output (redacting secrets and user data —
see [security.md](security.md)) and the exact config/operation.

See also: [FAQ](faq.md) · [Configuration](configuration.md).