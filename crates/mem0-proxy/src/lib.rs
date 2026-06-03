//! Memory-augmented chat proxy. Port of `mem0/mem0/proxy/main.py` (behavioral).
//!
//! [`MemoryProxy::chat`] retrieves relevant memories for the latest user
//! message, augments the prompt with them, calls the chat LLM, and persists the
//! exchange back into memory for future recall.

use mem0::{AddOptions, JsonMap, Memory, MemoryConfig, SearchOptions};
use mem0_core::error::Result;
use mem0_core::traits::{GenerateOptions, Llm};
use mem0_core::types::Message;
use serde_json::json;

/// Result of a [`MemoryProxy::chat`] call.
#[derive(Debug, Clone)]
pub struct ChatResult {
    /// The assistant response text.
    pub response: String,
    /// The memory texts that were injected as context.
    pub memories_used: Vec<String>,
}

/// A memory-augmented chat proxy wrapping a [`Memory`] and a chat [`Llm`].
pub struct MemoryProxy {
    memory: Memory,
    llm: Box<dyn Llm>,
}

impl MemoryProxy {
    /// Construct a proxy from an existing memory and chat LLM.
    pub fn new(memory: Memory, llm: Box<dyn Llm>) -> Self {
        Self { memory, llm }
    }

    /// Build a proxy from a [`MemoryConfig`], reusing `config.llm` as the chat model.
    pub fn from_config(config: MemoryConfig) -> Result<Self> {
        let llm = mem0_llm::build_llm(&config.llm.provider, &config.llm.config)?;
        let memory = mem0::from_config(config)?;
        Ok(Self { memory, llm })
    }

    /// Access the underlying memory (e.g. for seeding or inspection).
    pub fn memory(&self) -> &Memory {
        &self.memory
    }

    /// Run a memory-augmented chat turn.
    ///
    /// Retrieves up to `top_k` memories relevant to the latest user message,
    /// augments that message with them, calls the chat LLM, and persists the
    /// full exchange (best-effort) for future recall.
    pub async fn chat(
        &self,
        messages: Vec<Message>,
        user_id: Option<&str>,
        agent_id: Option<&str>,
        run_id: Option<&str>,
        top_k: usize,
    ) -> Result<ChatResult> {
        let filters = scope_filters(user_id, agent_id, run_id);

        // Latest user message drives retrieval.
        let query = messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.clone())
            .unwrap_or_default();

        let mut memories_used = Vec::new();
        if !query.is_empty() && !filters.is_empty() {
            let search = self
                .memory
                .search(
                    &query,
                    &filters,
                    SearchOptions {
                        top_k,
                        ..Default::default()
                    },
                )
                .await?;
            if let Some(arr) = search.get("results").and_then(|r| r.as_array()) {
                for item in arr {
                    if let Some(s) = item.get("memory").and_then(|m| m.as_str()) {
                        memories_used.push(s.to_string());
                    }
                }
            }
        }

        let augmented = augment(&messages, &memories_used);
        let response = self
            .llm
            .generate(&augmented, &GenerateOptions::default())
            .await?;

        // Persist the exchange for future recall (best-effort, non-fatal).
        if !filters.is_empty() {
            let mut to_store = messages.clone();
            to_store.push(Message::assistant(response.clone()));
            let opts = AddOptions {
                user_id: user_id.map(|s| s.to_string()),
                agent_id: agent_id.map(|s| s.to_string()),
                run_id: run_id.map(|s| s.to_string()),
                infer: Some(true),
                ..Default::default()
            };
            if let Err(e) = self.memory.add(to_store, opts).await {
                tracing::warn!("Proxy failed to persist exchange: {e}");
            }
        }

        Ok(ChatResult {
            response,
            memories_used,
        })
    }
}

fn scope_filters(user_id: Option<&str>, agent_id: Option<&str>, run_id: Option<&str>) -> JsonMap {
    let mut filters = JsonMap::new();
    if let Some(v) = user_id {
        filters.insert("user_id".into(), json!(v));
    }
    if let Some(v) = agent_id {
        filters.insert("agent_id".into(), json!(v));
    }
    if let Some(v) = run_id {
        filters.insert("run_id".into(), json!(v));
    }
    filters
}

/// Augment the latest user message with retrieved memories.
fn augment(messages: &[Message], memories: &[String]) -> Vec<Message> {
    let mut out = messages.to_vec();
    if memories.is_empty() {
        return out;
    }
    if let Some(last) = out.last_mut() {
        if last.role == "user" {
            let bullets = memories
                .iter()
                .map(|m| format!("- {m}"))
                .collect::<Vec<_>>()
                .join("\n");
            let original = last.content.clone();
            last.content = format!("Relevant memories:\n{bullets}\n\nUser query: {original}");
        }
    }
    out
}
