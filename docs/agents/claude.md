# Give a Claude (Anthropic) agent memory

This guide adds long-term memory to an agent built on Anthropic's Claude models
(including Claude Code-style coding agents).

## 1. Configure the memory LLM as Anthropic

Anthropic provides chat models but not embeddings, so pair the `anthropic` LLM
with an embedding provider (OpenAI, Gemini, or local Ollama):

```json
{
  "llm":      { "provider": "anthropic", "config": { "model": "claude-3-5-sonnet-20240620" } },
  "embedder": { "provider": "openai",    "config": { "model": "text-embedding-3-small" } },
  "vector_store": { "provider": "embedded", "config": { "path": "/var/lib/mem0" } }
}
```

Set `ANTHROPIC_API_KEY` (and `OPENAI_API_KEY` for the embedder), or put `api_key`
in each `config`. To stay fully cloud-free, swap the embedder to
`{"provider":"ollama","config":{"model":"nomic-embed-text"}}` — see
[Local/Ollama](local-ollama.md).

> Claude here only extracts durable facts during `add`; your agent can chat with
> any Claude model independently.

## 2. Run the memory service

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export MEM0_CONFIG_FILE=/etc/mem0/claude.json
./target/release/mem0-server
```

## 3. Wire it into the agent loop

```python
import requests, anthropic
MEM0, USER = "http://localhost:8080", "alex"
client = anthropic.Anthropic()

def recall(q):
    r = requests.post(f"{MEM0}/v1/memories/search", json={"query": q, "user_id": USER, "top_k": 5})
    return [m["memory"] for m in r.json()["results"]]

def turn(user_msg):
    memories = recall(user_msg)
    system = "You are a helpful assistant.\nRelevant memories:\n" + "\n".join(f"- {m}" for m in memories)
    reply = client.messages.create(
        model="claude-3-5-sonnet-20240620", max_tokens=1024,
        system=system, messages=[{"role":"user","content":user_msg}]
    ).content[0].text
    requests.post(f"{MEM0}/v1/memories",
                  json={"messages":[{"role":"user","content":user_msg},
                                    {"role":"assistant","content":reply}], "user_id": USER})
    return reply
```

## 4. Expose memory via Claude tool use

Claude's tool-use API can call memory tools autonomously. Define two tools:

```json
[
  { "name": "recall_memory",
    "description": "Search the user's long-term memory.",
    "input_schema": { "type": "object",
      "properties": { "query": { "type": "string" } }, "required": ["query"] } },
  { "name": "save_memory",
    "description": "Save durable facts about the user.",
    "input_schema": { "type": "object",
      "properties": { "text": { "type": "string" } }, "required": ["text"] } }
]
```

Handle `recall_memory` with `POST /v1/memories/search` and `save_memory` with
`POST /v1/memories` (passing the session's `user_id`). For Claude Code-style
agents, these let Claude remember repo conventions and prior decisions across
sessions.

## 5. Or use the proxy (Rust)

```rust
let mem = mem0::from_json(claude_config)?;          // anthropic llm + openai embedder
let chat_llm = mem0::build_llm("anthropic",
    &serde_json::json!({ "model": "claude-3-5-sonnet-20240620" }))?;
let proxy = mem0_proxy::MemoryProxy::new(mem, chat_llm);
let out = proxy.chat(vec![mem0_core::types::Message::user(msg)], Some("alex"), None, None, 5).await?;
```

See also: [Agent overview](overview.md) · [LLM providers](../providers/llms.md).