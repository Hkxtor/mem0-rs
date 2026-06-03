# FAQ

## Is this the same as mem0?

It is a faithful Rust port of mem0's OSS core (v2.0.4): the same extraction
prompts (byte-for-byte), the same hybrid retrieval, dedup, filtering, and history
semantics. Memory quality is validated as identical ([quality.md](quality.md)),
while it is faster and far lighter ([performance.md](performance.md)). Some
backends and the hosted platform are out of scope ([comparison.md](comparison.md)).

## Do I need to write Rust to use it?

No. Run the [`mem0-server`](server.md) binary and call its [REST API](api/rest.md)
from any language (Python, TypeScript, Go, …). If you *are* in Rust, use the
[library](developer-guide.md).

## Can it run fully offline / on-prem?

Yes. With Ollama (local LLM + embeddings) and the `embedded` store, **no data
leaves the host** — see [agents/local-ollama.md](agents/local-ollama.md).

## Does it need a vector database?

No. The default `embedded` store is in-process (optionally file-persisted) with
native BM25. Switch to Qdrant or pgvector only when you need a shared, scaled
backend — a config change, no code changes
([providers/vector-stores.md](providers/vector-stores.md)).

## How do I give an existing agent memory?

Wire `search` before the model call and `add` after, via the library, the REST
API, or the [proxy](proxy.md). Per-stack guides:
[OpenAI/Codex](agents/openai-codex.md), [Claude](agents/claude.md),
[Gemini](agents/gemini.md), [Local/Ollama](agents/local-ollama.md).

## Which providers are supported?

Embedders: OpenAI, OpenAI-compatible, Azure, Gemini, Ollama, mock. LLMs: OpenAI,
OpenAI-compatible (Groq/xAI/DeepSeek/Together/vLLM/LM Studio/MiniMax/Sarvam),
Azure, Anthropic, Gemini, Ollama, mock. Vector stores: embedded, qdrant,
pgvector. Rerankers: llm, cohere, zero_entropy. Full list:
[configuration.md](configuration.md).

## Is there a synchronous API?

Yes — `mem0::blocking::Memory` mirrors the async API and owns its own runtime
([developer-guide.md](developer-guide.md#blocking-synchronous-api)).

## How much faster / lighter is it?

~3–3.5× faster per operation and ~6.5× lower peak memory than Python mem0 on an
equal, network-free workload; a single ~12 MB binary vs a ~100 MB process. The
unavoidable LLM/embedding network cost is identical for both
([performance.md](performance.md)).

## Where is my data stored?

Memories live in the configured vector store (`embedded` JSON file, Qdrant, or
Postgres); change history + a recent-message buffer live in SQLite
(`history_db_path`). You control deletion via `delete` / `delete_all` / `reset`.
See [security.md](security.md).

## Does it support images / multimodal or graph memory?

Not in this port — text messages only, and graph memory (Neo4j/Kuzu/etc.) is out
of scope ([comparison.md](comparison.md)).

## How do I delete a user's data (GDPR-style)?

`delete_all` with that `user_id` removes their memories; `reset` wipes everything
including history. See [security.md](security.md#data-privacy--pii).

## Can I add a provider or backend that isn't included?

Yes. Implement the relevant trait (`Embedder` / `Llm` / `VectorStore` /
`Reranker`) and pass it to `Memory::new`, or register it in the factory. See
[developer-guide.md](developer-guide.md#extending-with-a-custom-provider) and
[architecture.md](architecture.md).

## Does it phone home / send telemetry?

No. PostHog telemetry is intentionally omitted; the only outbound calls go to the
providers you configure.

## How do I verify the parity and performance claims myself?

```bash
pip install -e mem0
mem0-rs/bench/run_parity.sh      # prompt fidelity + 17/17 equivalence
mem0-rs/bench/run_compare.sh     # speed + memory comparison
```

## What license is it under?

Apache-2.0, matching upstream mem0.

More: [Troubleshooting](troubleshooting.md) · [docs index](README.md).