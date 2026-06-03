# Agent integrations — overview

mem0-rs gives **any** agent long-term memory, regardless of which model powers
it. There are two independent choices:

1. **Which model extracts/embeds memories** — set the `llm` and `embedder`
   providers (OpenAI, Anthropic/Claude, Gemini, a local model, …). This is
   independent of whatever model your agent uses to chat.
2. **How memory plugs into your agent loop** — via the Rust library, the REST
   server, or the proxy.

## The universal pattern: retrieve → augment → store

Every integration follows the same loop:

```text
on each user turn:
  1. RETRIEVE  memories = mem.search(user_message, {user_id})        # before the model call
  2. AUGMENT   prompt   = system + memories + conversation
  3. GENERATE  reply    = your_agent_model(prompt)
  4. STORE     mem.add([user_message, reply], {user_id})            # after the model call
```

- **Retrieve** with `search` (scoped by `user_id`/`agent_id`/`run_id`).
- **Augment** by injecting the retrieved memory texts into your system prompt or
  the latest user message.
- **Store** the exchange with `add` (inferred), which extracts durable facts and
  deduplicates them.

The [`mem0-proxy`](../proxy.md) implements exactly this loop for you; the
per-model guides show both the proxy and the manual version.

## Three integration surfaces

| Surface | Best for | Doc |
|---------|----------|-----|
| **Rust library** | Agents/tools written in Rust | [developer-guide.md](../developer-guide.md) |
| **REST server** | Agents in any language; a shared memory service | [api/rest.md](../api/rest.md) |
| **Proxy** | Drop-in memory-augmented chat | [proxy.md](../proxy.md) |

Any agent framework that can call an HTTP endpoint (Python, TypeScript, Go, …)
can use the [REST server](../server.md) — `search` before the model call and
`add` after.

## Pointing memory at a model family

The memory `llm` only needs to be good at extracting concise facts (a small
instruct model is ideal and cheap). Pick the family that matches your stack:

- [OpenAI / Codex](openai-codex.md) — `openai` provider (and `openai_compatible`
  for OpenAI-API-compatible gateways).
- [Anthropic / Claude](claude.md) — `anthropic` provider.
- [Gemini](gemini.md) — `gemini` provider.
- [Local / Ollama](local-ollama.md) — `ollama` (fully offline).

You can mix: e.g. embed with OpenAI and extract with Claude, or run everything
locally on Ollama.

## Scope & honesty

mem0-rs provides the **memory layer and a service**; it does not ship a
turnkey editor/CLI plugin (e.g. an MCP server) for specific agent products. The
integration point is the REST API (or the library/proxy), which is intentionally
small and works with any agent. If your agent runtime supports custom tools or
HTTP calls, wiring `search`/`add` as two tools (`recall_memory` /
`save_memory`) is usually all you need — see the per-model guides for concrete
tool definitions.

## A minimal "memory tool" pair

Most agent frameworks let you register tools/functions. Define two backed by the
REST API or library:

```text
recall_memory(query, user_id)   -> POST /v1/memories/search  -> list of memory texts
save_memory(text, user_id)      -> POST /v1/memories          -> stored
```

Call `recall_memory` before answering and `save_memory` after, or let the agent
call them autonomously. The per-model guides include ready-to-adapt snippets.

Continue to your stack: [OpenAI/Codex](openai-codex.md) ·
[Claude](claude.md) · [Gemini](gemini.md) · [Local/Ollama](local-ollama.md).