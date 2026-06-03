# Give a Gemini agent memory

This guide adds long-term memory to an agent built on Google's Gemini models.
Gemini can power **both** the memory extractor and the embedder.

## 1. Configure memory with Gemini

```json
{
  "llm":      { "provider": "gemini", "config": { "model": "gemini-2.0-flash" } },
  "embedder": { "provider": "gemini", "config": { "model": "models/gemini-embedding-001" } },
  "vector_store": { "provider": "embedded", "config": { "path": "/var/lib/mem0" } }
}
```

Set `GOOGLE_API_KEY` (or put `api_key` in each `config`). See
[providers/llms.md](../providers/llms.md) and
[providers/embedders.md](../providers/embedders.md).

> `gemini-2.0-flash` is a great, cheap extractor; your agent can still answer
> with a larger Gemini model.

## 2. Run the memory service

```bash
export GOOGLE_API_KEY=...
export MEM0_CONFIG_FILE=/etc/mem0/gemini.json
./target/release/mem0-server
```

## 3. Wire it into the agent loop

```python
import requests
import google.generativeai as genai
MEM0, USER = "http://localhost:8080", "alex"

def recall(q):
    r = requests.post(f"{MEM0}/v1/memories/search", json={"query": q, "user_id": USER, "top_k": 5})
    return [m["memory"] for m in r.json()["results"]]

def turn(user_msg):
    memories = recall(user_msg)
    context = "Relevant memories:\n" + "\n".join(f"- {m}" for m in memories)
    model = genai.GenerativeModel("gemini-2.0-flash",
                                  system_instruction="You are a helpful assistant.\n" + context)
    reply = model.generate_content(user_msg).text
    requests.post(f"{MEM0}/v1/memories",
                  json={"messages":[{"role":"user","content":user_msg},
                                    {"role":"assistant","content":reply}], "user_id": USER})
    return reply
```

## 4. Expose memory via Gemini function calling

Declare two functions the model can call; route them to the REST API:

```python
recall_memory = {
  "name": "recall_memory",
  "description": "Search the user's long-term memory.",
  "parameters": {"type":"object","properties":{"query":{"type":"string"}},"required":["query"]},
}
save_memory = {
  "name": "save_memory",
  "description": "Save durable facts about the user.",
  "parameters": {"type":"object","properties":{"text":{"type":"string"}},"required":["text"]},
}
# recall_memory -> POST /v1/memories/search ; save_memory -> POST /v1/memories
```

## 5. Or use the proxy (Rust)

```rust
let proxy = mem0_proxy::MemoryProxy::from_config(
    mem0::MemoryConfig::from_json(gemini_config)?)?;   // llm.provider = gemini
let out = proxy.chat(vec![mem0_core::types::Message::user(msg)],
                     Some("alex"), None, None, 5).await?;
```

## Notes

- Gemini embeddings default to 768 dimensions; set `output_dimensionality` /
  `embedding_dims` to change it (use a fresh collection if you do).
- The Gemini LLM provider maps `assistant`→`model`, lifts the system prompt to
  `systemInstruction`, and requests JSON output for extraction automatically.

See also: [Agent overview](overview.md).