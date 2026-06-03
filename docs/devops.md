# DevOps & deployment

mem0-rs deploys as a single static-friendly binary with a tiny footprint, which
makes operations simple. This page covers building, packaging, running, scaling,
and observing the service.

## Build artifacts

```bash
cargo build --release -p mem0-server
# target/release/mem0-server  (~12 MB; rustls TLS, bundled SQLite — no system OpenSSL)

# scale-out backends are compile-time features:
cargo build --release -p mem0-server --features qdrant
cargo build --release -p mem0-server --features pgvector
```

The binary needs no Python, no virtualenv, and (for the default `embedded`
store) no external services. Cross-compile with the standard Rust toolchain for
your target triple.

## Docker

A minimal multi-stage image (the runtime stage can be `debian:slim` or even
`gcr.io/distroless/cc` since there is no Python):

```dockerfile
FROM rust:1-slim AS build
WORKDIR /src
COPY mem0-rs/ ./mem0-rs/
RUN cd mem0-rs && cargo build --release -p mem0-server

FROM gcr.io/distroless/cc-debian12
COPY --from=build /src/mem0-rs/target/release/mem0-server /usr/local/bin/mem0-server
ENV PORT=8080
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/mem0-server"]
```

```bash
docker build -t mem0-rs .
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=sk-... \
  -e MEM0_CONFIG='{"embedder":{"provider":"openai","config":{}},"llm":{"provider":"openai","config":{}},"vector_store":{"provider":"embedded","config":{"path":"/data"}}}' \
  -v mem0data:/data \
  mem0-rs
```

> For Qdrant/pgvector deployments, build the image with the matching
> `--features` flag and point `vector_store` at the external service.

## systemd

```ini
# /etc/systemd/system/mem0.service
[Unit]
Description=mem0-rs memory server
After=network.target

[Service]
ExecStart=/usr/local/bin/mem0-server
Environment=PORT=8080
Environment=RUST_LOG=info
Environment=MEM0_CONFIG_FILE=/etc/mem0/config.json
DynamicUser=yes
StateDirectory=mem0
Restart=on-failure
# Hardening
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
```

Keep `/etc/mem0/config.json` at `chmod 600` (it may hold API keys), or supply
keys via `Environment=`/secret managers instead.

## Configuration management

- Prefer `MEM0_CONFIG_FILE` over inline `MEM0_CONFIG` so secrets do not appear in
  process listings.
- Inject provider keys via the standard env vars
  (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, …) from your secret
  manager; the config can omit `api_key` entirely.
- Pin the model names in config so behavior is reproducible across deploys.

## Scaling

The server is **stateless** beyond its configured stores:

- **Vertical:** the default `embedded` store keeps everything in process — one
  node, no dependencies, very low RSS. Persist with a `path` volume.
- **Horizontal:** point `vector_store` at a shared **Qdrant** or **pgvector**
  and run N replicas behind a load balancer. Each replica is identical and holds
  no session state. Use a shared/managed SQLite alternative or accept per-replica
  history if you only need vectors shared (history is local to each instance's
  `history_db_path`).

```text
        ┌────────────┐
client ─┤  load bal. ├─┬─ mem0-server #1 ─┐
        └────────────┘ ├─ mem0-server #2 ─┼─▶ Qdrant / Postgres+pgvector (shared)
                       └─ mem0-server #N ─┘
```

## Persistence & backups

| State | Where | Backup |
|-------|-------|--------|
| Memories (vectors+payload) | `embedded` JSON at `{path}/{collection}.json`, or Qdrant/Postgres | Snapshot the volume / DB |
| History + recent messages | SQLite at `history_db_path` | Snapshot the file (or use `:memory:` if you don't need it) |

For the `embedded` store, back up the `path` directory; for Qdrant/pgvector use
their native snapshot/backup tooling.

## Observability

- **Logs:** structured `tracing` to stdout; control verbosity with `RUST_LOG`
  (e.g. `RUST_LOG=mem0_server=debug,info`). Ship stdout to your log pipeline.
- **Health:** `GET /health` returns `{"status":"ok"}` — use it for liveness and
  readiness probes.
- **Metrics:** not built in; measure at the gateway/load balancer (latency,
  status codes) or wrap the binary with your APM sidecar. The error responses
  carry a machine-readable `code` for alerting.

## Resource sizing

Resident memory is dominated by your working set in the `embedded` store; the
runtime overhead is small (~9–15 MB in benchmarks). Start with modest limits
(e.g. 128–256 MB) for `embedded` single-node and size up with your memory
volume; offload to Qdrant/pgvector for large corpora so the server stays light.
See [performance.md](performance.md).

## Upgrades

Replace the binary and restart; the SQLite history schema self-migrates on
startup. Validate a new build against your config in staging (run the
`/health` + a sample `add`/`search`) before rolling out.

See also: [Server](server.md) · [Security](security.md).