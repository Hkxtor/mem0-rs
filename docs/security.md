# Security

This page describes mem0-rs's security properties, what it does and does not
protect, and how to harden a deployment.

## Properties

- **Memory safety.** Written in safe Rust; no buffer overflows, use-after-free,
  or data races in safe code. The codebase avoids `unsafe`.
- **TLS via rustls.** Outbound HTTPS to providers uses `rustls`, not a system
  OpenSSL — fewer CVE surfaces and clean static builds.
- **Parameterized SQL.** All SQLite access uses bound parameters. The optional
  `pgvector` backend additionally validates collection identifiers against a
  strict `[A-Za-z_][A-Za-z0-9_]*` allow-list and binds all filter values, so
  metadata filters cannot inject SQL.
- **No telemetry / phone-home.** Upstream mem0's PostHog telemetry is removed.
  mem0-rs makes outbound network calls **only** to the providers you configure
  (LLM, embedder, vector DB, reranker).
- **Small, auditable supply chain.** A curated dependency set rather than a large
  Python tree; review with `cargo tree` / `cargo audit`.

## Threat model & responsibilities

mem0-rs is a memory **engine and service**. It deliberately does **not** include:

- **Authentication / authorization** on the REST server.
- **TLS termination** for inbound traffic.
- **Rate limiting** or quota enforcement.
- **Multi-tenant isolation** beyond scope filters (`user_id`/`agent_id`/`run_id`).

These are the deployer's responsibility — put the server behind a reverse proxy
or API gateway that provides them. Treat the server as an internal service.

## Network exposure

The server binds `0.0.0.0:$PORT`. In production:

- Bind it to a private network / localhost and front it with a gateway, or run it
  as a sidecar reachable only by your app.
- Terminate TLS and authenticate at the gateway (mTLS, API keys, JWT, …).
- Apply rate limiting and request size limits at the gateway.

## Secrets handling

- Provider API keys are read from the `config` you supply **or** from standard
  environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`,
  `COHERE_API_KEY`, `ZERO_ENTROPY_API_KEY`, the Azure `*_AZURE_*` vars).
- Prefer `MEM0_CONFIG_FILE` (chmod 600) or environment injection from a secret
  manager over inline `MEM0_CONFIG`, so keys don't show up in `ps`/process
  listings or shell history.
- Keys are never logged. Avoid `RUST_LOG=trace` with custom code that might echo
  request bodies.

## Data privacy & PII

Memories are derived from user conversations and can contain personal data. You
control where they live:

- **Local-only option.** With Ollama + the `embedded` store, no data leaves the
  host (see [agents/local-ollama.md](agents/local-ollama.md)).
- **Cloud providers.** When using OpenAI/Anthropic/Gemini/Cohere/etc., message
  content and extracted memories are sent to those providers per their data
  policies — review and configure accordingly.
- **Deletion / right to be forgotten.** Use `delete` (one memory),
  `delete_all` (a whole `user_id` scope), or `reset` (everything). History rows
  are retained as an audit trail until `reset`; if your policy requires purging
  history too, use `reset` for the store or rotate the `history_db_path`.
- **Scoping.** Always pass `user_id` (and `run_id` where relevant) so memories
  are partitioned per user; never query across users without a scope.

## At-rest data

- `embedded` store: a JSON file at `{path}/{collection}.json` — protect the
  volume with filesystem permissions and disk encryption.
- SQLite history: a file at `history_db_path` — same protections; use `:memory:`
  if you don't need persistence.
- Qdrant/pgvector: rely on the backend's encryption/access controls.

## Hardening checklist

- [ ] Server bound to a private interface or behind a gateway.
- [ ] TLS + authentication + rate limiting enforced at the gateway.
- [ ] Secrets via env/secret manager or a `chmod 600` config file (not inline).
- [ ] Per-user `user_id` scoping enforced by your application.
- [ ] Volumes for the `embedded` store / SQLite encrypted and permission-locked.
- [ ] Run as a non-root user (e.g. systemd `DynamicUser=yes`, distroless image).
- [ ] `cargo audit` in CI; dependencies reviewed.
- [ ] A documented deletion path for user data (`delete_all` / `reset`).

## Reporting

Treat memory content as sensitive in logs and error reports. When filing issues,
redact API keys and any user data from request/response examples.

See also: [DevOps](devops.md) · [Configuration](configuration.md).