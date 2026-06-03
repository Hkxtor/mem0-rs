# Why mem0-rs

AI agents are only as good as what they remember. [mem0](https://github.com/mem0ai/mem0)
is an excellent memory layer, but its Python runtime carries costs that matter
in production: high resident memory, interpreter overhead on every call, a heavy
dependency tree, and a deployment surface that includes a Python environment and
(by default) an external vector database.

**mem0-rs** keeps mem0's behavior and prompts while removing those costs. It is
designed around four goals.

## 1. Speed

The orchestration path — message parsing, scope/filter construction, md5
hashing, BM25 lemmatization, SQLite history writes, payload construction, and
cosine/BM25 scoring — runs as compiled native code with static dispatch and
monomorphized math. There is no interpreter, no GIL, and no per-call pydantic
validation or `deepcopy`.

On an equal, network-free workload (same N adds + M searches, mock embedder,
in-process store), mem0-rs is **~3–3.5× faster per operation** than Python mem0
v2.0.4. The biggest wins are on the hot orchestration paths (`add` ~3.5×) where
the framework overhead — not the network — dominates.

> The unavoidable cost of any memory system is the network round-trip to the LLM
> and embedding APIs, which is identical for both implementations. mem0-rs makes
> *everything else* faster. Methodology and full numbers:
> [performance.md](performance.md).

## 2. Latency & footprint

- **~6.5× lower peak memory** (~9–15 MB RSS vs ~100 MB for the Python process).
- A single **~12 MB binary** — no CPython runtime and no heavyweight resident
  modules (pydantic, openai, qdrant-client, posthog).
- Cold start is a process exec, not an interpreter + import graph.

Lower, predictable memory means you can pack many more memory services per host
and run comfortably on small instances and at the edge.

## 3. Security & safety

- **Memory-safe by construction** (safe Rust; no buffer overflows, use-after-free,
  or data races in safe code).
- **`rustls` TLS** — no dependency on a system OpenSSL; the binary links cleanly
  and statically on macOS and Linux.
- **Parameterized SQL everywhere**; the optional pgvector backend additionally
  validates collection identifiers against a strict allow-list.
- **No telemetry / phone-home.** Upstream mem0's PostHog telemetry is
  intentionally omitted. mem0-rs makes no outbound calls except to the providers
  you configure.
- **Smaller supply chain.** A curated, auditable dependency set versus a large
  Python tree.

Details and a threat model: [security.md](security.md).

## 4. Scalability

- **Zero-dependency default.** The built-in `embedded` vector store runs
  in-process with optional JSON persistence and native BM25 — perfect for
  single-node, embedded, and edge deployments.
- **Scale out without code changes.** Switch `vector_store.provider` to
  `qdrant` or `pgvector` (feature-gated builds) when you need a shared,
  horizontally-scalable backend. The application code and API are unchanged.
- **Stateless server.** The REST server holds no session state beyond the
  configured stores, so you can run multiple instances behind a load balancer
  pointing at the same Qdrant/Postgres.

## What you do not give up

Memory **quality** is identical to Python mem0 — same extraction prompts
(byte-for-byte), same deduplication, same hybrid ranking, same filtering and
history semantics — validated by a deterministic equivalence harness at 100%
agreement. See [quality.md](quality.md). The one documented divergence is the
spaCy-derived `COMPOUND`/`NOUN` entity-boost signal (a secondary signal that
does not affect the primary retrieval path); see [comparison.md](comparison.md).

## When to use what

| You want… | Use |
|-----------|-----|
| Memory inside a Rust app | the [`mem0` library](developer-guide.md) |
| A language-agnostic memory service | the [`mem0-server`](server.md) + your HTTP client |
| Drop-in memory for a chat loop | the [`mem0-proxy`](proxy.md) |
| Single-node / edge, no infra | `embedded` vector store (default) |
| Shared, horizontally scaled store | `qdrant` or `pgvector` |

Next: [Getting started](getting-started.md).