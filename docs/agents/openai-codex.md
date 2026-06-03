# Give an OpenAI / Codex agent memory

This guide adds long-term memory to an agent built on OpenAI models (GPT-4o,
GPT-4o-mini, the Codex coding agents, or any OpenAI-API-compatible gateway).

## 1. Configure mem0-rs with OpenAI

```json
{
  "embedder": { "provider": "openai", "config": { "model": "text-embedding-3-small" } },
  "llm":      { "provider": "openai", "config": { "model": "gpt-4o-mini" } },
  "vector_store": { "provider": "embedded", "config": { "path": "/var/lib/mem0" } }
}
```

Set `OPENAI_API_KEY` in the environment (or put `api_key` in each `config`). For
an OpenAI-compatible gateway (Azure-style proxies, vLLM, LiteLLM, etc.), use the
`openai_compatible` provider with `openai_base_url`. See
[providers/llms.md](../providers/llms.md).

> The memory LLM (`gpt-4o-mini` here) is only used to extract facts; it is
> independent of the model your agent uses to think/code.

## 2. Run the memory service

```bash
export OPENAI_API_KEY=sk-...
export MEM0_CONFIG_FILE=/etc/mem0/openai.json
./target/release/mem0-server          # http://localhost:8080
```

## 3a. Wire it into the agent loop (manual)

Pseudocode for any OpenAI-driven agent (Python shown; the calls are plain HTTP):

```python
import requests, openai
MEM0 = "http://localhost:8080"
USER = "alex"

def recall(query):
    r = requests.post(f"{MEM0}/v1/memories/search",
                      json={"query": query, "user_id": USER, "top_k": 5})
    return [m["memory"] for m in r.json()["results"]]

def remember(messages):
    requests.post(f"{MEM0}/v1/memories", json={"messages": messages, "user_id": USER})

def turn(user_msg):
    memories = recall(user_msg)
    system = "You are a helpful assistant.\nRelevant memories:\n" + "\n".join(f"- {m}" for m in memories)
    reply = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"system","content":system},
                  {"role":"user","content":user_msg}]).choices[0].message.content
    remember([{"role":"user","content":user_msg},
              {"role":"assistant","content":reply}])
    return reply
```

## 3b. Expose memory as agent tools (function calling)

If your agent uses OpenAI tool/function calling, register two tools backed by the
REST API so the model can recall and save memory autonomously:

```json
[
  { "type": "function", "function": {
      "name": "recall_memory",
      "description": "Search the user's long-term memory for relevant facts.",
      "parameters": { "type": "object",
        "properties": { "query": { "type": "string" } }, "required": ["query"] } } },
  { "type": "function", "function": {
      "name": "save_memory",
      "description": "Save new durable facts about the user.",
      "parameters": { "type": "object",
        "properties": { "text": { "type": "string" } }, "required": ["text"] } } }
]
```

Map `recall_memory` → `POST /v1/memories/search` and `save_memory` →
`POST /v1/memories` (with the user's `user_id`). For coding agents (Codex-style),
the same two tools let the agent remember project conventions, past decisions,
and user preferences across sessions.

## 4. Or use the proxy (Rust)

If your orchestration is in Rust, the [proxy](../proxy.md) does retrieve→augment→
generate→store in one call:

```rust
let proxy = mem0_proxy::MemoryProxy::from_config(cfg /* llm.provider = openai */)?;
let out = proxy.chat(vec![mem0_core::types::Message::user(user_msg)],
                     Some("alex"), None, None, 5).await?;
```

## Tips

- Use `gpt-4o-mini` (or smaller) for extraction — fast and cheap; your agent can
  still answer with a larger model.
- Scope per end user with `user_id`; add `run_id` to isolate sessions/threads.
- Tag memories with `metadata` (e.g. `{"project":"acme"}`) and filter on it at
  recall time for project-scoped coding agents.

See also: [Agent overview](overview.md) · [REST API](../api/rest.md).