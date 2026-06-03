# mem0-rs documentation

Comprehensive, self-contained documentation for **mem0-rs** — the Rust port of
[mem0](https://github.com/mem0ai/mem0) that provides long-term memory for AI
agents as a single binary.

New here? Read [Why mem0-rs](why.md) → [Getting started](getting-started.md) →
[Core concepts](concepts.md).

## Start here

- [Why mem0-rs](why.md) — motivation: speed, latency, security, scalability.
- [Getting started](getting-started.md) — install, build, your first program, run the server.
- [Core concepts](concepts.md) — memories, scoping, inferred vs raw, dedup, hybrid retrieval, entity linking, memory types.

## Using mem0-rs

- [Developer guide](developer-guide.md) — using the Rust library end to end.
- [Server](server.md) — running the single-binary REST server.
- [Proxy](proxy.md) — the memory-augmented chat proxy.

## API reference

- [Rust library API](api/library.md) — `Memory`, `AddOptions`, `SearchOptions`, types, errors.
- [REST API](api/rest.md) — endpoints, request/response shapes, status codes, curl.
- [HTTP client](api/client.md) — the `mem0-client` crate.

## Configuration

- [Configuration reference](configuration.md) — the full config schema, defaults, environment variables.
- [Embedders](providers/embedders.md)
- [LLMs](providers/llms.md)
- [Vector stores](providers/vector-stores.md)
- [Rerankers](providers/rerankers.md)

## Internals

- [Architecture](architecture.md) — crate map, data flow, design decisions.
- [How it works](how-it-works.md) — the add/search pipelines, scoring math, NLP, storage.

## Agent integrations

- [Overview](agents/overview.md) — the universal retrieve → augment → store pattern.
- [OpenAI / Codex](agents/openai-codex.md)
- [Anthropic / Claude](agents/claude.md)
- [Gemini](agents/gemini.md)
- [Local / Ollama](agents/local-ollama.md)

## Operations

- [DevOps & deployment](devops.md) — building, Docker, systemd, scaling, observability.
- [Security](security.md) — threat model, secrets, data handling, hardening.

## Reference & comparisons

- [Performance](performance.md) — benchmark methodology and results vs Python mem0.
- [Quality parity](quality.md) — how memory quality is validated as identical to mem0.
- [Migration from Python mem0](migration-from-python.md) — API/config mapping and differences.
- [Comparison matrix](comparison.md) — feature-by-feature vs Python mem0.
- [Troubleshooting](troubleshooting.md)
- [FAQ](faq.md)