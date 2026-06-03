# Comparison with Python mem0

mem0-rs ports the **OSS core** of mem0 v2.0.4 faithfully. This page is the
honest, feature-by-feature accounting of what is identical, adapted, and
out-of-scope. For the *evidence* that behavior matches, see
[quality.md](quality.md).

Legend: ✅ ported/identical · 🟦 adapted (documented difference) · ⬜ out of scope.

## Memory operations

| Operation | Status | Notes |
|-----------|:------:|-------|
| `add` (raw, `infer=false`) | ✅ | Identical. |
| `add` (inferred additive extraction) | ✅ | Verbatim prompt; md5 dedup. |
| `search` (semantic) | ✅ | Cosine. |
| `search` (BM25 hybrid) | ✅ | Native in `embedded`/`pgvector`; semantic-only on `qdrant`. |
| Entity boosts in search | 🟦 | PROPER/QUOTED entities only (no spaCy `COMPOUND`/`NOUN`). |
| Reranking | ✅ | LLM / Cohere / ZeroEntropy. |
| Metadata filtering (operators, AND/OR/NOT) | ✅ | Full set on `embedded`/`qdrant`; broad set on `pgvector`. |
| `get` / `get_all` | ✅ | Identical shapes. |
| `update` / `delete` / `delete_all` | ✅ | Identical. |
| `history` | ✅ | Same SQLite schema + ordering. |
| `reset` | ✅ | Clears vectors + history (+ entity store). |
| Procedural memory | ✅ | `memory_type: "procedural_memory"`. |
| Scoping (user/agent/run) | ✅ | Identical. |
| Custom extraction instructions | ✅ | `custom_instructions` / per-call `prompt`. |

## Embedders

| Provider | Status |
|----------|:------:|
| OpenAI | ✅ |
| OpenAI-compatible (LM Studio / vLLM / Together / …) | ✅ |
| Azure OpenAI | ✅ |
| Gemini | ✅ |
| Ollama | ✅ |
| mock | ✅ |
| HuggingFace, VertexAI, FastEmbed, LangChain, AWS Bedrock | ⬜ Not ported |

## LLMs

| Provider | Status |
|----------|:------:|
| OpenAI (+ structured) | ✅ |
| OpenAI-compatible (Groq / xAI / DeepSeek / Together / vLLM / LM Studio / MiniMax / Sarvam) | ✅ |
| Azure OpenAI | ✅ |
| Anthropic | ✅ |
| Gemini | ✅ |
| Ollama | ✅ |
| mock | ✅ |
| AWS Bedrock, LiteLLM, LangChain (native) | ⬜ Not ported (many reachable via the OpenAI-compatible provider) |

## Vector stores

| Backend | Status |
|---------|:------:|
| `embedded` (in-process, BM25, persistable) | ✅ Default (new in the Rust port) |
| Qdrant | ✅ (feature `qdrant`) |
| pgvector | ✅ (feature `pgvector`) |
| Chroma, Pinecone, Milvus, Weaviate, Redis, Valkey, MongoDB, Elasticsearch, OpenSearch, FAISS, Azure AI Search, Supabase, and others | ⬜ Not ported |

## Rerankers

| Provider | Status |
|----------|:------:|
| LLM reranker | ✅ |
| Cohere | ✅ |
| ZeroEntropy | ✅ |
| SentenceTransformer, HuggingFace cross-encoder | ⬜ Not ported |

## NLP

| Capability | Status | Notes |
|------------|:------:|-------|
| BM25 lemmatization | 🟦 | Deterministic dependency-free lemmatizer (no spaCy); used symmetrically. |
| Entity extraction (PROPER, QUOTED) | ✅ | Always-on, dependency-free. |
| Entity extraction (COMPOUND, NOUN) | ⬜ | Needs a dependency parser; omitted. Secondary signal only. |

## Interfaces

| Interface | Status | Notes |
|-----------|:------:|-------|
| Library API | ✅ | Async + `blocking::Memory`. |
| REST server | ✅ | axum, single binary. |
| HTTP client | ✅ | `mem0-client`. |
| Memory-augmented proxy | ✅ | `mem0-proxy`. |
| Graph memory (Neo4j / Kuzu / Memgraph / Neptune) | ⬜ | Not ported. |
| Hosted/platform cloud client | ⬜ | OSS-only port. |
| OpenMemory MCP server / editor plugins | ⬜ | Integrate via REST instead. |
| Telemetry (PostHog) | ⬜ | Intentionally removed. |
| Vision / multimodal inputs | ⬜ | Text messages only. |

## Behavioral fidelity

Ported byte-for-byte / exactly: extraction & update prompts, md5 dedup, payload
schema, SQLite schema + query ordering, BM25 scoring math, filter/scope
construction, error taxonomy. Validated identical by the deterministic
equivalence harness ([quality.md](quality.md)).

Adapted (documented): dependency-free NLP, async-native traits (+ blocking
wrapper), `rustls` HTTP, `embedded` default store, curated provider set, no
telemetry.

## Summary

If you use mem0's core memory operations with OpenAI/Anthropic/Gemini/Ollama and
Qdrant/pgvector or in-process storage, mem0-rs is a behavior-compatible,
faster, lighter drop-in. If you depend on graph memory, an unported
vector/LLM/embedder backend, the hosted platform, or the MCP plugins, those are
out of scope for this port — see the rows above.

See also: [Migration guide](migration-from-python.md) · [Quality](quality.md).