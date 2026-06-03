# LLMs

The LLM performs fact extraction during inferred `add` (and is reused as the chat
model by the proxy and the LLM reranker). Configure it with
`"llm": { "provider": "<name>", "config": { … } }`.

Common config fields:

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `model` | string | provider-specific | Model id. |
| `api_key` | string | env fallback | API key. |
| `temperature` | float | `0.1` | Sampling temperature. |
| `max_tokens` | int | `2000` | Max generated tokens. |
| `top_p` | float | `0.1` | Nucleus sampling. |
| `top_k` | int | — | Where supported. |
| `reasoning_effort` | string | — | For reasoning models (`low`/`medium`/`high`). |
| `openai_base_url` | string | — | OpenAI / compatible endpoint. |
| `ollama_base_url` | string | — | Ollama endpoint. |
| `base_url` | string | — | Generic endpoint override (Anthropic/Gemini/testing). |
| `azure_kwargs` | object | — | Azure connection. |

Extraction always requests a JSON-object response so the additive prompt's output
parses reliably.

### Reasoning models

For reasoning / GPT-5-class models (`o1`, `o3`, `gpt-5*`, …), mem0-rs
automatically **omits** `temperature` / `top_p` / `max_tokens` (which those APIs
reject) and forwards `reasoning_effort` when set. Detection is by model name.

## OpenAI — `openai` / `openai_structured`

- Default model: `gpt-5-mini`. Endpoint `https://api.openai.com/v1` (or
  `openai_base_url` / `OPENAI_BASE_URL`). Key `api_key` or `OPENAI_API_KEY`.

```json
{ "provider": "openai", "config": { "model": "gpt-4o-mini", "api_key": "sk-..." } }
```

## OpenAI-compatible — `openai_compatible` / `openai_like` / `groq` / `xai` / `deepseek` / `together` / `vllm` / `lmstudio` / `minimax` / `sarvam`

Any OpenAI `/chat/completions`-compatible endpoint. **Requires**
`openai_base_url` (or `base_url`). The provider alias is for readability; they all
use the OpenAI-compatible client.

```json
{ "provider": "groq", "config": { "openai_base_url": "https://api.groq.com/openai/v1",
    "model": "llama-3.1-70b-versatile", "api_key": "gsk_..." } }
```

## Azure OpenAI — `azure_openai` / `azure_openai_structured` / `azure`

Uses `azure_kwargs` (or `LLM_AZURE_*` env). The last message's literal word
`assistant` is rewritten to `ai` to match upstream behavior.

```json
{ "provider": "azure_openai", "config": { "model": "gpt-4o", "azure_kwargs": {
    "api_key": "...", "azure_deployment": "gpt4o-deploy",
    "azure_endpoint": "https://my.openai.azure.com", "api_version": "2024-02-01" } } }
```

| `azure_kwargs` field | Env fallback |
|----------------------|--------------|
| `api_key` | `LLM_AZURE_OPENAI_API_KEY` |
| `azure_deployment` | `LLM_AZURE_DEPLOYMENT` |
| `azure_endpoint` | `LLM_AZURE_ENDPOINT` |
| `api_version` | `LLM_AZURE_API_VERSION` (default `2024-02-01`) |

## Anthropic (Claude) — `anthropic`

- Default model: `claude-3-5-sonnet-20240620`. Endpoint
  `https://api.anthropic.com` (or `base_url`). Key `api_key` or
  `ANTHROPIC_API_KEY`. System messages are lifted into the top-level `system`
  field; only `temperature` is sent (Anthropic forbids `temperature`+`top_p`
  together).

```json
{ "provider": "anthropic", "config": { "model": "claude-3-5-sonnet-20240620", "api_key": "sk-ant-..." } }
```

## Google Gemini — `gemini`

- Default model: `gemini-2.0-flash`. Endpoint
  `https://generativelanguage.googleapis.com` (or `base_url`). Key `api_key` or
  `GOOGLE_API_KEY`. `assistant` is mapped to `model`; system → `systemInstruction`;
  JSON requests set `responseMimeType: application/json`.

```json
{ "provider": "gemini", "config": { "model": "gemini-2.0-flash", "api_key": "..." } }
```

## Ollama (local) — `ollama`

- Default model: `llama3.1:70b`. Endpoint `http://localhost:11434` (or
  `ollama_base_url`/`base_url`), `/api/chat`. JSON requests set `format: "json"`
  and append a JSON instruction to the last user message.

```json
{ "provider": "ollama", "config": { "model": "llama3.1:8b", "ollama_base_url": "http://localhost:11434" } }
```

## Mock — `mock`

Returns an empty additive payload (`{"memory": []}`). Pairs with the `mock`
embedder + `embedded` store for offline runs and tests.

```json
{ "provider": "mock" }
```

## Model recommendations

- A small, fast instruct model is plenty for extraction — `gpt-4o-mini`,
  `claude-3-5-haiku`, `gemini-2.0-flash`, or `llama3.1:8b` locally.
- Lower `temperature` (the default `0.1`) yields more consistent extractions.

Back to [Configuration](../configuration.md).