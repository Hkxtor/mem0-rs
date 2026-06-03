# Memory-augmented chat proxy

`mem0-proxy` is the simplest way to give a chat loop long-term memory. It wraps a
`Memory` and a chat `Llm` and, on each turn:

1. **retrieves** memories relevant to the latest user message,
2. **augments** that message with them,
3. **generates** the assistant reply with your LLM, and
4. **persists** the exchange back into memory (best-effort, non-fatal) so future
   turns can recall it.

This is the behavioral port of mem0's proxy.

## Construct

```rust
use mem0_proxy::MemoryProxy;
use mem0::MemoryConfig;

// Reuses config.llm as the chat model AND the extraction model.
let proxy = MemoryProxy::from_config(MemoryConfig::from_json(r#"{
    "embedder": { "provider": "openai", "config": { "api_key": "sk-..." } },
    "llm":      { "provider": "openai", "config": { "api_key": "sk-...", "model": "gpt-4o-mini" } },
    "vector_store": { "provider": "embedded", "config": { "path": "/var/lib/mem0" } }
}"#)?)?;
```

Or build it from an existing memory + a separate chat LLM:

```rust
use mem0_proxy::MemoryProxy;
let proxy = MemoryProxy::new(memory, mem0::build_llm("anthropic",
    &serde_json::json!({ "api_key": "sk-ant-...", "model": "claude-3-5-sonnet-20240620" }))?);
```

## Chat

```rust
use mem0_core::types::Message;

let result = proxy.chat(
    vec![Message::user("Where should I eat tonight?")],
    Some("alex"),   // user_id
    None,           // agent_id
    None,           // run_id
    5,              // top_k memories to retrieve
).await?;

println!("assistant: {}", result.response);
println!("grounded on {} memories: {:?}", result.memories_used.len(), result.memories_used);
```

`ChatResult` contains:

- `response` — the assistant's reply text.
- `memories_used` — the memory texts injected as context (for transparency /
  debugging).

The latest `user` message drives retrieval; retrieved memories are prepended to
that message as a `Relevant memories:` block before the LLM call. The full
exchange (your messages + the assistant reply) is then stored with `infer: true`
so durable facts are extracted for next time.

## Inspect the underlying memory

```rust
let mem = proxy.memory();                 // &Memory
let all = mem.get_all(&filters, 50).await?;
```

## When to use the proxy vs. direct calls

| Use the proxy when… | Use `Memory` directly when… |
|---------------------|-----------------------------|
| You want a drop-in memory-augmented chat turn | You control prompt assembly yourself |
| The default retrieve→augment→store flow fits | You need custom retrieval/injection logic |
| You want minimal glue code | You only need storage or only retrieval |

For agents built on a specific model family, see the
[agent integration guides](agents/overview.md), which use this same pattern
(directly or via the proxy).