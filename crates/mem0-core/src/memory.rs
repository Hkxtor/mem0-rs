//! The [`Memory`] orchestrator. Ported from `mem0/mem0/memory/main.py`.
//!
//! Implements construction, `add` (both non-inferred raw and the inferred
//! additive-extraction pipeline), `search` (semantic + BM25 + entity boosts +
//! advanced metadata filters + optional reranking), reads (`get`/`get_all`),
//! mutations (`update`/`delete`/`delete_all`), procedural memory, `history`,
//! and `reset`. Entity linking and entity boosts are best-effort and require an
//! injected entity store.

use crate::config::MemoryConfig;
use crate::enums::MemoryType;
use crate::error::{Mem0Error, Result};
use crate::filters::{
    build_filters_and_metadata, build_session_scope, has_advanced_operators,
    process_metadata_filters, session_filters, validate_search_params,
};
use crate::history::{HistoryStore, NewHistory};
use crate::nlp::{extract_entities, extract_entities_batch, lemmatize_for_bm25};
use crate::prompts::{
    generate_additive_extraction_prompt, AdditivePromptArgs, ADDITIVE_EXTRACTION_PROMPT,
    AGENT_CONTEXT_SUFFIX, PROCEDURAL_MEMORY_SYSTEM_PROMPT,
};
use crate::scoring::{get_bm25_params, normalize_bm25, score_and_rank, ENTITY_BOOST_WEIGHT};
use crate::text::{extract_json, parse_messages, remove_code_blocks};
use crate::traits::{Embedder, GenerateOptions, Llm, MemoryAction, Reranker, VectorStore};
use crate::types::{AddResult, JsonMap, Message, MessagesInput, SearchHit, VectorRecord};
use crate::util::{md5_hex, now_utc_rfc3339};
use serde_json::{json, Map, Value};
use std::collections::{BTreeSet, HashMap, HashSet};
use uuid::Uuid;

/// Options for [`Memory::add`].
#[derive(Debug, Clone, Default)]
pub struct AddOptions {
    /// User scope id.
    pub user_id: Option<String>,
    /// Agent scope id.
    pub agent_id: Option<String>,
    /// Run scope id.
    pub run_id: Option<String>,
    /// Extra metadata stored with each memory.
    pub metadata: Option<JsonMap>,
    /// Whether to run the LLM extraction pipeline (default `true`).
    pub infer: Option<bool>,
    /// Memory type (`procedural_memory` is the only special case).
    pub memory_type: Option<String>,
    /// Custom extraction prompt / instructions.
    pub prompt: Option<String>,
    /// Preserve an existing memory id (import path). `None` generates a fresh UUID,
    /// exactly as before.
    pub id: Option<String>,
    /// Preserve the original creation timestamp (import path). `None` stamps now.
    pub created_at: Option<String>,
    /// Preserve the original update timestamp (import path). `None` falls back to
    /// `created_at`, exactly as before.
    pub updated_at: Option<String>,
    /// Expiration date (YYYY-MM-DD). Stored in the payload; expired memories are hidden
    /// from reads unless explicitly requested. Validation happens on use.
    pub expiration_date: Option<String>,
}

/// Identity fields an importer wants preserved on the created record (id, timestamps).
/// `None` anywhere means "generate/fill as before", so ordinary adds are unaffected.
#[derive(Debug, Clone, Default)]
pub(crate) struct RecordIdentity {
    pub id: Option<String>,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
    pub expiration_date: Option<String>,
}

impl RecordIdentity {
    pub(crate) fn from_opts(opts: &AddOptions, expiration: Option<String>) -> Self {
        RecordIdentity {
            id: opts.id.clone(),
            created_at: opts.created_at.clone(),
            updated_at: opts.updated_at.clone(),
            expiration_date: expiration,
        }
    }
}

/// Options for [`Memory::search`].
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// Maximum number of results.
    pub top_k: usize,
    /// Minimum semantic score to include.
    pub threshold: f64,
    /// Whether to apply reranking (requires a reranker).
    pub rerank: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            top_k: 20,
            threshold: 0.1,
            rerank: false,
        }
    }
}

/// Promoted payload keys surfaced as top-level fields in read results.
const PROMOTED_KEYS: &[&str] = &["user_id", "agent_id", "run_id", "actor_id", "role"];

/// The memory orchestrator. Holds injected providers and the history store.
pub struct Memory {
    /// Effective configuration.
    pub config: MemoryConfig,
    embedder: Box<dyn Embedder>,
    llm: Box<dyn Llm>,
    vector_store: Box<dyn VectorStore>,
    reranker: Option<Box<dyn Reranker>>,
    entity_store: Option<Box<dyn VectorStore>>,
    db: HistoryStore,
    custom_instructions: Option<String>,
    use_input_language: bool,
}

impl Memory {
    /// Construct a [`Memory`] from explicit providers.
    pub fn new(
        config: MemoryConfig,
        embedder: Box<dyn Embedder>,
        llm: Box<dyn Llm>,
        vector_store: Box<dyn VectorStore>,
        reranker: Option<Box<dyn Reranker>>,
    ) -> Result<Self> {
        let db = HistoryStore::new(&config.history_db_path)?;
        let custom_instructions = config.custom_instructions.clone();
        let use_input_language = config.use_input_language;
        Ok(Self {
            config,
            embedder,
            llm,
            vector_store,
            reranker,
            entity_store: None,
            db,
            custom_instructions,
            use_input_language,
        })
    }

    /// Attach an entity store, enabling best-effort entity linking and boosts.
    pub fn with_entity_store(mut self, store: Box<dyn VectorStore>) -> Self {
        self.entity_store = Some(store);
        self
    }

    /// Add memories. Mirrors Python `Memory.add`.
    pub async fn add(
        &self,
        messages: impl Into<MessagesInput>,
        opts: AddOptions,
    ) -> Result<Vec<AddResult>> {
        let (processed_metadata, effective_filters) = build_filters_and_metadata(
            opts.user_id.as_deref(),
            opts.agent_id.as_deref(),
            opts.run_id.as_deref(),
            None,
            opts.metadata.as_ref(),
            None,
        )?;

        // Normalize/validate expiration_date up front so both record paths just store it.
        // Mirrors OSS `_normalize_expiration_date` (raises ValueError on bad input).
        let expiration = match &opts.expiration_date {
            Some(raw) => crate::util::normalize_expiration_date(&Value::String(raw.clone()))?,
            None => None,
        };

        if let Some(mt) = &opts.memory_type {
            if mt != MemoryType::Procedural.as_str() {
                return Err(Mem0Error::validation_code(
                    "VALIDATION_002",
                    format!(
                        "Invalid 'memory_type'. Please pass {} to create procedural memories.",
                        MemoryType::Procedural.as_str()
                    ),
                    Some(format!(
                        "Use '{}' to create procedural memories.",
                        MemoryType::Procedural.as_str()
                    )),
                ));
            }
        }

        let messages = messages.into().into_messages();

        if opts.memory_type.as_deref() == Some(MemoryType::Procedural.as_str()) {
            return self
                .create_procedural_memory(&messages, &processed_metadata, opts.prompt.as_deref())
                .await;
        }

        let infer = opts.infer.unwrap_or(true);
        let identity = RecordIdentity::from_opts(&opts, expiration);
        if !infer {
            return self
                .add_to_vector_store_raw(&messages, &processed_metadata, &identity)
                .await;
        }

        self.add_to_vector_store_infer(
            &messages,
            &processed_metadata,
            &effective_filters,
            opts.prompt.as_deref(),
            &identity,
        )
        .await
    }

    /// Non-inferred add: store each non-system message as a raw memory.
    async fn add_to_vector_store_raw(
        &self,
        messages: &[Message],
        metadata: &JsonMap,
        identity: &RecordIdentity,
    ) -> Result<Vec<AddResult>> {
        let mut returned = Vec::new();
        for message in messages {
            if message.role == "system" {
                continue;
            }
            let mut per_msg_meta = metadata.clone();
            per_msg_meta.insert("role".into(), Value::String(message.role.clone()));
            if let Some(name) = &message.name {
                per_msg_meta.insert("actor_id".into(), Value::String(name.clone()));
            }

            let mem_id = self
                .create_memory(&message.content, None, per_msg_meta, Some(identity))
                .await?;

            returned.push(AddResult {
                id: mem_id,
                memory: message.content.clone(),
                event: "ADD".to_string(),
                actor_id: message.name.clone(),
                role: Some(message.role.clone()),
            });
        }
        Ok(returned)
    }

    /// Inferred additive-extraction pipeline. Port of the V3 phased pipeline in
    /// `_add_to_vector_store` (infer=true).
    async fn add_to_vector_store_infer(
        &self,
        messages: &[Message],
        metadata: &JsonMap,
        filters: &JsonMap,
        prompt: Option<&str>,
        identity: &RecordIdentity,
    ) -> Result<Vec<AddResult>> {
        // Phase 0: context.
        let session_scope = build_session_scope(filters);
        let last = self.db.get_last_messages(&session_scope, 10)?;
        let last_k: Vec<(String, String)> = last
            .into_iter()
            .map(|m| (m.role.unwrap_or_default(), m.content.unwrap_or_default()))
            .collect();
        let parsed_messages = parse_messages(messages);

        // Phase 1: existing-memory retrieval.
        let search_filters = session_filters(filters);
        let query_embedding = self
            .embedder
            .embed(&parsed_messages, MemoryAction::Search)
            .await?;
        let existing = self
            .vector_store
            .search(&parsed_messages, &query_embedding, 10, &search_filters)
            .await
            .unwrap_or_default();

        let mut existing_memories: Vec<Value> = Vec::new();
        for (idx, mem) in existing.iter().enumerate() {
            let text = mem
                .payload
                .get("data")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            existing_memories.push(json!({ "id": idx.to_string(), "text": text }));
        }

        // Phase 2: LLM extraction.
        let is_agent_scoped = filters.get("agent_id").is_some() && filters.get("user_id").is_none();
        let system_prompt = if is_agent_scoped {
            format!("{ADDITIVE_EXTRACTION_PROMPT}{AGENT_CONTEXT_SUFFIX}")
        } else {
            ADDITIVE_EXTRACTION_PROMPT.to_string()
        };
        let custom_instr = prompt.or(self.custom_instructions.as_deref());
        let user_prompt = generate_additive_extraction_prompt(&AdditivePromptArgs {
            summary: "",
            recently_extracted_memories: &[],
            existing_memories: &existing_memories,
            new_messages: &parsed_messages,
            last_k_messages: &last_k,
            current_date: None,
            observation_date: None,
            custom_instructions: custom_instr,
            use_input_language: self.use_input_language,
        });

        let llm_messages = vec![Message::system(system_prompt), Message::user(user_prompt)];
        let opts = GenerateOptions {
            response_format_json: true,
            ..Default::default()
        };
        let response = match self.llm.generate(&llm_messages, &opts).await {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("LLM extraction failed: {e}");
                // Swallowed on purpose: a failed extraction must not fail the add. The
                // reason is recorded so the caller can explain the empty result.
                crate::diagnostics::record(
                    crate::diagnostics::Stage::LlmError,
                    "",
                    0,
                    Some(e.to_string()),
                );
                return Ok(vec![]);
            }
        };

        // Parse the extraction response. Three outcomes, and the caller is told which:
        // the model answered "nothing to remember", the answer could not be parsed, or
        // there was an answer at all.
        let cleaned = remove_code_blocks(&response);
        let extracted: Vec<Value> = if cleaned.trim().is_empty() {
            crate::diagnostics::record(crate::diagnostics::Stage::EmptyAnswer, &response, 0, None);
            let _ = self.db.save_messages(messages, &session_scope);
            return Ok(vec![]);
        } else {
            match parse_memory_array(&cleaned) {
                Some(arr) if arr.is_empty() => {
                    crate::diagnostics::record(
                        crate::diagnostics::Stage::EmptyAnswer,
                        &response,
                        0,
                        None,
                    );
                    let _ = self.db.save_messages(messages, &session_scope);
                    return Ok(vec![]);
                }
                Some(arr) => arr,
                None => match salvage_memory_array(&cleaned) {
                    // The answer was cut off. Keep the entries that were complete: losing
                    // them would discard memories the model already wrote and paid for.
                    Some(arr) => {
                        crate::diagnostics::record(
                            crate::diagnostics::Stage::Salvaged,
                            &response,
                            arr.len(),
                            None,
                        );
                        arr
                    }
                    None => {
                        crate::diagnostics::record(
                            crate::diagnostics::Stage::Unparsed,
                            &response,
                            0,
                            None,
                        );
                        let _ = self.db.save_messages(messages, &session_scope);
                        return Ok(vec![]);
                    }
                },
            }
        };

        // Phase 3: batch embed extracted texts.
        let mem_texts: Vec<String> = extracted
            .iter()
            .filter_map(|m| {
                m.get("text")
                    .and_then(|t| t.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
            })
            .collect();
        let mut embed_map: HashMap<String, Vec<f32>> = HashMap::new();
        match self
            .embedder
            .embed_batch(&mem_texts, MemoryAction::Add)
            .await
        {
            Ok(vs) => {
                for (t, v) in mem_texts.iter().zip(vs) {
                    embed_map.insert(t.clone(), v);
                }
            }
            Err(_) => {
                for t in &mem_texts {
                    if let Ok(v) = self.embedder.embed(t, MemoryAction::Add).await {
                        embed_map.insert(t.clone(), v);
                    }
                }
            }
        }

        // Phases 4–5: build records with hash dedup.
        let mut existing_hashes: HashSet<String> = HashSet::new();
        for mem in &existing {
            if let Some(h) = mem.payload.get("hash").and_then(|v| v.as_str()) {
                existing_hashes.insert(h.to_string());
            }
        }

        let mut records: Vec<(String, String, Vec<f32>, JsonMap)> = Vec::new();
        let mut seen_hashes: HashSet<String> = HashSet::new();
        for mem in &extracted {
            let text = match mem.get("text").and_then(|t| t.as_str()) {
                Some(t) if !t.is_empty() => t,
                _ => continue,
            };
            let embedding = match embed_map.get(text) {
                Some(e) => e.clone(),
                None => continue,
            };
            let mem_hash = md5_hex(text);
            if existing_hashes.contains(&mem_hash) || seen_hashes.contains(&mem_hash) {
                continue;
            }
            seen_hashes.insert(mem_hash.clone());

            let mut meta = metadata.clone();
            meta.insert("data".into(), json!(text));
            meta.insert("text_lemmatized".into(), json!(lemmatize_for_bm25(text)));
            meta.insert("hash".into(), json!(mem_hash));
            // An imported id is only meaningful for a single produced record; with several
            // extracted memories it would collide, so it is applied to the first record only.
            if records.is_empty() {
                if let Some(id) = &identity.id {
                    meta.insert("__imported_id".into(), json!(id));
                }
            }
            if let Some(ts) = &identity.created_at {
                meta.insert("created_at".into(), json!(ts));
            } else {
                meta.entry("created_at").or_insert(json!(now_utc_rfc3339()));
            }
            let created_at = meta
                .get("created_at")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(now_utc_rfc3339);
            let updated_at = identity
                .updated_at
                .clone()
                .unwrap_or_else(|| created_at.clone());
            meta.insert("updated_at".into(), json!(updated_at));
            if let Some(exp) = &identity.expiration_date {
                meta.insert("expiration_date".into(), json!(exp));
            }
            if let Some(att) = mem.get("attributed_to").and_then(|v| v.as_str()) {
                meta.insert("attributed_to".into(), json!(att));
            }

            let id = meta
                .remove("__imported_id")
                .and_then(|v| v.as_str().map(|s| s.to_string()))
                .unwrap_or_else(|| Uuid::new_v4().to_string());
            records.push((id, text.to_string(), embedding, meta));
        }

        if records.is_empty() {
            let _ = self.db.save_messages(messages, &session_scope);
            return Ok(vec![]);
        }

        // Phase 6: persist (batch, with per-item fallback).
        let vrecords: Vec<VectorRecord> = records
            .iter()
            .map(|r| VectorRecord {
                id: r.0.clone(),
                vector: r.2.clone(),
                payload: r.3.clone(),
            })
            .collect();
        let persisted: Vec<(String, String, Vec<f32>, JsonMap)> =
            match self.vector_store.insert(vrecords).await {
                Ok(()) => records.clone(),
                Err(e) => {
                    tracing::warn!("Batch insert failed ({e}); falling back to per-item inserts");
                    let mut ok = Vec::new();
                    for r in &records {
                        match self
                            .vector_store
                            .insert(vec![VectorRecord {
                                id: r.0.clone(),
                                vector: r.2.clone(),
                                payload: r.3.clone(),
                            }])
                            .await
                        {
                            Ok(()) => ok.push(r.clone()),
                            Err(e) => tracing::error!("Failed to insert memory {}: {e}", r.0),
                        }
                    }
                    ok
                }
            };

        if persisted.is_empty() {
            let _ = self.db.save_messages(messages, &session_scope);
            return Ok(vec![]);
        }

        let hrecords: Vec<NewHistory> = persisted
            .iter()
            .map(|r| {
                let created_at =
                    r.3.get("created_at")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                NewHistory {
                    memory_id: r.0.clone(),
                    old_memory: None,
                    new_memory: Some(r.1.clone()),
                    event: "ADD".into(),
                    created_at: created_at.clone(),
                    updated_at: created_at,
                    is_deleted: 0,
                    actor_id: None,
                    role: None,
                }
            })
            .collect();
        if self.db.batch_add_history(&hrecords).is_err() {
            for h in &hrecords {
                let _ = self.db.add_history(
                    &h.memory_id,
                    None,
                    h.new_memory.as_deref(),
                    "ADD",
                    h.created_at.as_deref(),
                    h.updated_at.as_deref(),
                    0,
                    None,
                    None,
                );
            }
        }

        // Phase 7: best-effort entity linking.
        self.link_entities(&persisted, &search_filters).await;

        // Phase 8: persist messages and return.
        let _ = self.db.save_messages(messages, &session_scope);

        Ok(persisted
            .iter()
            .map(|r| AddResult {
                id: r.0.clone(),
                memory: r.1.clone(),
                event: "ADD".into(),
                actor_id: None,
                role: None,
            })
            .collect())
    }

    /// Create a procedural memory. Port of `_create_procedural_memory`.
    async fn create_procedural_memory(
        &self,
        messages: &[Message],
        metadata: &JsonMap,
        prompt: Option<&str>,
    ) -> Result<Vec<AddResult>> {
        let mut parsed = Vec::with_capacity(messages.len() + 2);
        parsed.push(Message::system(
            prompt.unwrap_or(PROCEDURAL_MEMORY_SYSTEM_PROMPT),
        ));
        parsed.extend_from_slice(messages);
        parsed.push(Message::user(
            "Create procedural memory of the above conversation.",
        ));

        let response = self
            .llm
            .generate(&parsed, &GenerateOptions::default())
            .await?;
        let procedural = remove_code_blocks(&response);

        let mut meta = metadata.clone();
        meta.insert(
            "memory_type".into(),
            Value::String(MemoryType::Procedural.as_str().into()),
        );
        let embedding = self.embedder.embed(&procedural, MemoryAction::Add).await?;
        let id = self
            .create_memory(&procedural, Some(embedding), meta, None)
            .await?;

        Ok(vec![AddResult {
            id,
            memory: procedural,
            event: "ADD".into(),
            actor_id: None,
            role: None,
        }])
    }

    /// Create a single memory and record ADD history. Port of `_create_memory`.
    pub(crate) async fn create_memory(
        &self,
        data: &str,
        precomputed: Option<Vec<f32>>,
        mut metadata: JsonMap,
        identity: Option<&RecordIdentity>,
    ) -> Result<String> {
        let embeddings = match precomputed {
            Some(v) => v,
            None => self.embedder.embed(data, MemoryAction::Add).await?,
        };
        let memory_id = identity
            .and_then(|i| i.id.clone())
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        metadata.insert("data".into(), Value::String(data.to_string()));
        metadata.insert("hash".into(), Value::String(md5_hex(data)));
        if let Some(ts) = identity.and_then(|i| i.created_at.as_ref()) {
            metadata.insert("created_at".into(), Value::String(ts.clone()));
        } else if !metadata.contains_key("created_at") {
            metadata.insert("created_at".into(), Value::String(now_utc_rfc3339()));
        }
        let created_at = metadata
            .get("created_at")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let updated_at = identity
            .and_then(|i| i.updated_at.clone())
            .unwrap_or_else(|| created_at.clone());
        metadata.insert("updated_at".into(), Value::String(updated_at));
        if let Some(exp) = identity.and_then(|i| i.expiration_date.as_ref()) {
            metadata.insert("expiration_date".into(), Value::String(exp.clone()));
        }
        metadata.insert(
            "text_lemmatized".into(),
            Value::String(lemmatize_for_bm25(data)),
        );

        let actor_id = metadata
            .get("actor_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let role = metadata
            .get("role")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        self.vector_store
            .insert(vec![VectorRecord {
                id: memory_id.clone(),
                vector: embeddings,
                payload: metadata,
            }])
            .await?;

        self.db.add_history(
            &memory_id,
            None,
            Some(data),
            "ADD",
            Some(&created_at),
            Some(&created_at),
            0,
            actor_id.as_deref(),
            role.as_deref(),
        )?;
        Ok(memory_id)
    }

    /// Retrieve a memory by id. Port of `get`.
    pub async fn get(&self, memory_id: &str) -> Result<Option<Value>> {
        let memory = self.vector_store.get(memory_id).await?;
        Ok(memory.map(|hit| format_memory_item(&hit, false)))
    }

    /// List memories matching filters. Port of `get_all`.
    pub async fn get_all(&self, filters: &JsonMap, top_k: usize) -> Result<Value> {
        if !["user_id", "agent_id", "run_id"]
            .iter()
            .any(|k| filters.contains_key(*k))
        {
            return Err(Mem0Error::validation_code(
                "VALIDATION_001",
                "filters must contain at least one of: user_id, agent_id, run_id.",
                Some("Example: filters={\"user_id\": \"u1\"}".into()),
            ));
        }
        // OSS fetches max(limit*4, 60) when hiding expired memories, filters, then truncates
        // to the requested limit — mirror that so a few expired rows don't shrink the page.
        let fetch_limit = std::cmp::max(top_k * 4, 60);
        let hits = self.vector_store.list(filters, Some(fetch_limit)).await?;
        let results: Vec<Value> = hits
            .iter()
            .filter(|h| !crate::util::payload_is_expired(&h.payload))
            .take(top_k)
            .map(|h| format_memory_item(h, true))
            .collect();
        Ok(json!({ "results": results }))
    }

    /// List every stored memory, unfiltered. Port of the server helper
    /// `list_all_memories` (`vector_store.list(top_k=limit)`), which is what the
    /// dashboard's Memories and Entities pages and the export endpoint read.
    ///
    /// `get_all` refuses to run without a scope key — correctly, it answers "what does
    /// this user have". A dashboard listing everything is a different question, and it
    /// is answered from the store directly in Python too.
    pub async fn list_all(&self, top_k: usize) -> Result<Value> {
        let fetch_limit = std::cmp::max(top_k * 4, 60);
        let hits = self
            .vector_store
            .list(&JsonMap::new(), Some(fetch_limit))
            .await?;
        let results: Vec<Value> = hits
            .iter()
            .filter(|h| !crate::util::payload_is_expired(&h.payload))
            .take(top_k)
            .map(|h| format_memory_item(h, false))
            .collect();
        Ok(json!({ "results": results }))
    }

    /// Like [`Self::list_all`] but applies `filters` without requiring a scope key
    /// (`get_all` demands user_id/agent_id/run_id; the dashboard's category-only filter
    /// has none). Same expiration semantics: expired rows are hidden.
    pub async fn list_all_filtered(&self, filters: &JsonMap, top_k: usize) -> Result<Value> {
        let fetch_limit = std::cmp::max(top_k * 4, 60);
        let hits = self.vector_store.list(filters, Some(fetch_limit)).await?;
        let results: Vec<Value> = hits
            .iter()
            .filter(|h| !crate::util::payload_is_expired(&h.payload))
            .take(top_k)
            .map(|h| format_memory_item(h, false))
            .collect();
        Ok(json!({ "results": results }))
    }

    /// List every stored memory including expired ones. The server's export uses this
    /// (Python's `list_all_memories` reads `vector_store.list` directly and never applies
    /// the expiration filter — backups must not silently drop rows).
    pub async fn list_all_include_expired(&self, top_k: usize) -> Result<Value> {
        let hits = self.vector_store.list(&JsonMap::new(), Some(top_k)).await?;
        let results: Vec<Value> = hits.iter().map(|h| format_memory_item(h, false)).collect();
        Ok(json!({ "results": results }))
    }

    /// The vector store, for callers that need a raw listing (entities, export).
    ///
    /// Python's `Memory` exposes `self.vector_store`; this is the same handle.
    pub fn vector_store(&self) -> &dyn VectorStore {
        self.vector_store.as_ref()
    }

    /// Search memories. Port of `search` + `_search_vector_store`.
    pub async fn search(
        &self,
        query: &str,
        filters: &JsonMap,
        options: SearchOptions,
    ) -> Result<Value> {
        validate_search_params(Some(options.threshold), Some(options.top_k as i64))?;

        let mut effective: JsonMap = filters.clone();
        if !["user_id", "agent_id", "run_id"]
            .iter()
            .any(|k| effective.contains_key(*k))
        {
            return Err(Mem0Error::validation_code(
                "VALIDATION_001",
                "filters must contain at least one of: user_id, agent_id, run_id.",
                Some("Example: filters={\"user_id\": \"u1\"}".into()),
            ));
        }

        if has_advanced_operators(&effective) {
            let processed = process_metadata_filters(&effective)?;
            for lk in ["AND", "OR", "NOT"] {
                effective.remove(lk);
            }
            let keys: Vec<String> = effective.keys().cloned().collect();
            for fk in keys {
                if !["AND", "OR", "NOT", "user_id", "agent_id", "run_id"].contains(&fk.as_str())
                    && effective.get(&fk).map(|v| v.is_object()).unwrap_or(false)
                {
                    effective.remove(&fk);
                }
            }
            for (k, v) in processed {
                effective.insert(k, v);
            }
        }

        let mut results = self
            .search_vector_store(query, &effective, options.top_k, options.threshold)
            .await?;

        if options.rerank {
            if let Some(reranker) = &self.reranker {
                if !results.is_empty() {
                    let docs: Vec<String> = results
                        .iter()
                        .map(|v| {
                            v.get("memory")
                                .and_then(|m| m.as_str())
                                .unwrap_or("")
                                .to_string()
                        })
                        .collect();
                    match reranker.rerank(query, &docs, options.top_k).await {
                        Ok(order) => {
                            let mut reordered = Vec::with_capacity(order.len());
                            for (idx, score) in order {
                                if let Some(item) = results.get(idx) {
                                    let mut it = item.clone();
                                    it["score"] = json!(score);
                                    reordered.push(it);
                                }
                            }
                            results = reordered;
                        }
                        Err(e) => tracing::warn!("Reranking failed, using original results: {e}"),
                    }
                }
            }
        }

        Ok(json!({ "results": results }))
    }

    /// Port of `_search_vector_store`.
    async fn search_vector_store(
        &self,
        query: &str,
        filters: &JsonMap,
        limit: usize,
        threshold: f64,
    ) -> Result<Vec<Value>> {
        let query_lemmatized = lemmatize_for_bm25(query);
        let query_entities = extract_entities(query);

        let embeddings = self.embedder.embed(query, MemoryAction::Search).await?;
        let internal_limit = (limit * 4).max(60);

        let semantic = self
            .vector_store
            .search(query, &embeddings, internal_limit, filters)
            .await?;

        // BM25 from keyword search, if supported.
        let mut bm25: HashMap<String, f64> = HashMap::new();
        if let Some(keyword) = self
            .vector_store
            .keyword_search(&query_lemmatized, internal_limit, filters)
            .await?
        {
            let (midpoint, steepness) = get_bm25_params(&query_lemmatized);
            for mem in keyword {
                let raw = mem.score as f64;
                if raw > 0.0 {
                    bm25.insert(mem.id.clone(), normalize_bm25(raw, midpoint, steepness));
                }
            }
        }

        // Entity boosts.
        let entity_boosts = if !query_entities.is_empty() {
            self.compute_entity_boosts(&query_entities, filters).await
        } else {
            HashMap::new()
        };

        let scored = score_and_rank(&semantic, &bm25, &entity_boosts, threshold, limit);

        let mut out = Vec::new();
        for hit in scored {
            let has_data = hit
                .payload
                .get("data")
                .and_then(|v| v.as_str())
                .map(|s| !s.is_empty())
                .unwrap_or(false);
            if !has_data {
                continue;
            }
            // OSS drops expired memories from search candidates before scoring.
            if crate::util::payload_is_expired(&hit.payload) {
                continue;
            }
            out.push(format_memory_item(&hit, false));
        }
        Ok(out)
    }

    /// Best-effort entity linking (Phase 7). Non-fatal. Requires an entity store.
    async fn link_entities(
        &self,
        records: &[(String, String, Vec<f32>, JsonMap)],
        search_filters: &JsonMap,
    ) {
        let store = match &self.entity_store {
            Some(s) => s.as_ref(),
            None => return,
        };
        if let Err(e) = self
            .link_entities_inner(store, records, search_filters)
            .await
        {
            tracing::warn!("Batch entity linking failed: {e}");
        }
    }

    async fn link_entities_inner(
        &self,
        store: &dyn VectorStore,
        records: &[(String, String, Vec<f32>, JsonMap)],
        search_filters: &JsonMap,
    ) -> Result<()> {
        let texts: Vec<String> = records.iter().map(|r| r.1.clone()).collect();
        let all_entities = extract_entities_batch(&texts);

        let mut order: Vec<String> = Vec::new();
        let mut global: HashMap<String, (String, String, BTreeSet<String>)> = HashMap::new();
        for (idx, rec) in records.iter().enumerate() {
            let mid = &rec.0;
            if let Some(ents) = all_entities.get(idx) {
                for (etype, etext) in ents {
                    let key = etext.trim().to_lowercase();
                    if key.is_empty() {
                        continue;
                    }
                    global
                        .entry(key.clone())
                        .and_modify(|e| {
                            e.2.insert(mid.clone());
                        })
                        .or_insert_with(|| {
                            order.push(key.clone());
                            let mut s = BTreeSet::new();
                            s.insert(mid.clone());
                            (etype.clone(), etext.clone(), s)
                        });
                }
            }
        }
        if order.is_empty() {
            return Ok(());
        }

        let entity_texts: Vec<String> = order.iter().map(|k| global[k].1.clone()).collect();
        let embeddings: Vec<Option<Vec<f32>>> = match self
            .embedder
            .embed_batch(&entity_texts, MemoryAction::Add)
            .await
        {
            Ok(vs) => vs.into_iter().map(Some).collect(),
            Err(_) => {
                let mut out = Vec::new();
                for t in &entity_texts {
                    out.push(self.embedder.embed(t, MemoryAction::Add).await.ok());
                }
                out
            }
        };

        let mut to_insert: Vec<VectorRecord> = Vec::new();
        for (i, key) in order.iter().enumerate() {
            let emb = match &embeddings[i] {
                Some(e) => e.clone(),
                None => continue,
            };
            let (etype, etext, mids) = &global[key];
            let matches = store
                .search(etext, &emb, 1, search_filters)
                .await
                .unwrap_or_default();
            if let Some(m) = matches.first() {
                if m.score >= 0.95 {
                    let mut payload = m.payload.clone();
                    let mut linked: BTreeSet<String> = payload
                        .get("linked_memory_ids")
                        .and_then(|v| v.as_array())
                        .map(|a| {
                            a.iter()
                                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                                .collect()
                        })
                        .unwrap_or_default();
                    for mid in mids {
                        linked.insert(mid.clone());
                    }
                    payload.insert(
                        "linked_memory_ids".into(),
                        Value::Array(linked.into_iter().map(Value::String).collect()),
                    );
                    let _ = store.update(&m.id, None, Some(payload)).await;
                    continue;
                }
            }
            let mut payload = JsonMap::new();
            payload.insert("data".into(), Value::String(etext.clone()));
            payload.insert("entity_type".into(), Value::String(etype.clone()));
            payload.insert(
                "linked_memory_ids".into(),
                Value::Array(mids.iter().cloned().map(Value::String).collect()),
            );
            for (k, v) in search_filters {
                payload.insert(k.clone(), v.clone());
            }
            to_insert.push(VectorRecord {
                id: Uuid::new_v4().to_string(),
                vector: emb,
                payload,
            });
        }
        if !to_insert.is_empty() {
            let _ = store.insert(to_insert).await;
        }
        Ok(())
    }

    /// Best-effort per-memory entity boosts. Port of `_compute_entity_boosts`.
    async fn compute_entity_boosts(
        &self,
        query_entities: &[(String, String)],
        filters: &JsonMap,
    ) -> HashMap<String, f64> {
        let store = match &self.entity_store {
            Some(s) => s.as_ref(),
            None => return HashMap::new(),
        };

        let mut seen: HashSet<String> = HashSet::new();
        let mut deduped: Vec<String> = Vec::new();
        for (_, etext) in query_entities.iter().take(8) {
            let key = etext.trim().to_lowercase();
            if !key.is_empty() && seen.insert(key) {
                deduped.push(etext.clone());
            }
        }
        if deduped.is_empty() {
            return HashMap::new();
        }

        let search_filters = session_filters(filters);
        let mut boosts: HashMap<String, f64> = HashMap::new();
        for etext in deduped {
            let emb = match self.embedder.embed(&etext, MemoryAction::Search).await {
                Ok(e) => e,
                Err(_) => continue,
            };
            let matches = store
                .search(&etext, &emb, 500, &search_filters)
                .await
                .unwrap_or_default();
            for m in matches {
                let similarity = m.score as f64;
                if similarity < 0.5 {
                    continue;
                }
                let linked: Vec<String> = m
                    .payload
                    .get("linked_memory_ids")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();
                let num_linked = linked.len().max(1) as f64;
                let weight = 1.0 / (1.0 + 0.001 * (num_linked - 1.0).powi(2));
                let boost = similarity * ENTITY_BOOST_WEIGHT * weight;
                for mid in linked {
                    let entry = boosts.entry(mid).or_insert(0.0);
                    if boost > *entry {
                        *entry = boost;
                    }
                }
            }
        }
        boosts
    }

    /// Update a memory's content. Port of `update`.
    pub async fn update(
        &self,
        memory_id: &str,
        data: &str,
        metadata: Option<JsonMap>,
    ) -> Result<Value> {
        let embeddings = self.embedder.embed(data, MemoryAction::Update).await?;
        self.update_memory(memory_id, data, Some(embeddings), metadata)
            .await?;
        Ok(json!({ "message": "Memory updated successfully!" }))
    }

    /// Port of `_update_memory`.
    pub(crate) async fn update_memory(
        &self,
        memory_id: &str,
        data: &str,
        precomputed: Option<Vec<f32>>,
        metadata: Option<JsonMap>,
    ) -> Result<String> {
        let existing =
            self.vector_store.get(memory_id).await?.ok_or_else(|| {
                Mem0Error::not_found(format!("Memory with id {memory_id} not found"))
            })?;
        let prev_value = existing
            .payload
            .get("data")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let mut new_metadata: JsonMap = metadata.unwrap_or_default();
        new_metadata.insert("data".into(), Value::String(data.to_string()));
        new_metadata.insert("hash".into(), Value::String(md5_hex(data)));
        new_metadata.insert(
            "text_lemmatized".into(),
            Value::String(lemmatize_for_bm25(data)),
        );
        let created_at = existing
            .payload
            .get("created_at")
            .cloned()
            .unwrap_or(Value::Null);
        new_metadata.insert("created_at".into(), created_at.clone());
        let updated_at = now_utc_rfc3339();
        new_metadata.insert("updated_at".into(), Value::String(updated_at.clone()));

        for key in ["user_id", "agent_id", "run_id"] {
            if !new_metadata.contains_key(key) {
                if let Some(v) = existing.payload.get(key) {
                    new_metadata.insert(key.into(), v.clone());
                }
            }
        }
        if let Some(v) = existing.payload.get("actor_id") {
            new_metadata.insert("actor_id".into(), v.clone());
        }
        if !new_metadata.contains_key("role") {
            if let Some(v) = existing.payload.get("role") {
                new_metadata.insert("role".into(), v.clone());
            }
        }

        let actor_id = new_metadata
            .get("actor_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let role = new_metadata
            .get("role")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        self.vector_store
            .update(memory_id, precomputed, Some(new_metadata))
            .await?;

        self.db.add_history(
            memory_id,
            prev_value.as_deref(),
            Some(data),
            "UPDATE",
            created_at.as_str(),
            Some(&updated_at),
            0,
            actor_id.as_deref(),
            role.as_deref(),
        )?;
        Ok(memory_id.to_string())
    }

    /// Delete a memory by id. Port of `delete`.
    pub async fn delete(&self, memory_id: &str) -> Result<Value> {
        let existing = self.vector_store.get(memory_id).await?;
        if existing.is_none() {
            return Err(Mem0Error::not_found(format!(
                "Memory with id {memory_id} not found"
            )));
        }
        self.delete_memory(memory_id, existing).await?;
        Ok(json!({ "message": "Memory deleted successfully!" }))
    }

    /// Port of `_delete_memory`.
    pub(crate) async fn delete_memory(
        &self,
        memory_id: &str,
        existing: Option<SearchHit>,
    ) -> Result<String> {
        let existing = match existing {
            Some(e) => e,
            None => self.vector_store.get(memory_id).await?.ok_or_else(|| {
                Mem0Error::not_found(format!("Memory with id {memory_id} not found"))
            })?,
        };
        let prev_value = existing
            .payload
            .get("data")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let created_at = existing
            .payload
            .get("created_at")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let updated_at = now_utc_rfc3339();
        let actor_id = existing
            .payload
            .get("actor_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let role = existing
            .payload
            .get("role")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        self.vector_store.delete(memory_id).await?;
        self.db.add_history(
            memory_id,
            Some(&prev_value),
            None,
            "DELETE",
            created_at.as_deref(),
            Some(&updated_at),
            1,
            actor_id.as_deref(),
            role.as_deref(),
        )?;
        Ok(memory_id.to_string())
    }

    /// Delete all memories in a scope. Port of `delete_all`.
    pub async fn delete_all(
        &self,
        user_id: Option<&str>,
        agent_id: Option<&str>,
        run_id: Option<&str>,
    ) -> Result<Value> {
        let mut filters = Map::new();
        if let Some(v) = user_id {
            filters.insert("user_id".into(), Value::String(v.to_string()));
        }
        if let Some(v) = agent_id {
            filters.insert("agent_id".into(), Value::String(v.to_string()));
        }
        if let Some(v) = run_id {
            filters.insert("run_id".into(), Value::String(v.to_string()));
        }
        if filters.is_empty() {
            return Err(Mem0Error::validation_code(
                "VALIDATION_006",
                "At least one filter is required to delete all memories. Use reset() to delete everything.",
                None,
            ));
        }
        let hits = self.vector_store.list(&filters, None).await?;
        let count = hits.len();
        for hit in hits {
            self.delete_memory(&hit.id, Some(hit.clone())).await?;
        }
        tracing::info!("Deleted {count} memories");
        Ok(json!({ "message": "Memories deleted successfully!" }))
    }

    /// Return the change history of a memory. Port of `history`.
    pub async fn history(&self, memory_id: &str) -> Result<Vec<crate::history::HistoryRecord>> {
        self.db.get_history(memory_id)
    }

    /// Reset: clear the vector store, entity store, and history db. Port of `reset`.
    pub async fn reset(&self) -> Result<()> {
        self.db.reset()?;
        self.vector_store.reset().await?;
        if let Some(store) = &self.entity_store {
            let _ = store.reset().await;
        }
        Ok(())
    }
}

/// Parse a `{"memory": [...]}` array from cleaned LLM output, with an
/// `extract_json` fallback.
///
/// `None` means no `memory` array could be found in the answer — a parse failure, which is
/// a different event from a model that legitimately answered `{"memory": []}`.
fn parse_memory_array(cleaned: &str) -> Option<Vec<Value>> {
    if let Ok(v) = serde_json::from_str::<Value>(cleaned) {
        if let Some(arr) = v.get("memory").and_then(|m| m.as_array()) {
            return Some(arr.clone());
        }
    }
    let ej = extract_json(cleaned);
    if let Ok(v) = serde_json::from_str::<Value>(&ej) {
        if let Some(arr) = v.get("memory").and_then(|m| m.as_array()) {
            return Some(arr.clone());
        }
    }
    None
}

/// Recover the complete entries from an answer that is not valid JSON.
///
/// The usual cause is the model running out of output budget mid-array: everything before
/// the cut is well-formed, the tail is not. Discarding the whole answer throws away
/// memories the model already wrote, so instead walk the `"memory": [` array and keep every
/// object that parses on its own, stopping at the first one that does not.
///
/// Only called when the strict parse failed, and only when an actual `"memory"` key is
/// present — prose with braces in it must not become memories.
fn salvage_memory_array(text: &str) -> Option<Vec<Value>> {
    let key = text.find("\"memory\"")?;
    let open = text[key..].find('[')? + key;
    let mut items = Vec::new();
    let mut pos = open + 1;
    while let Some(rel) = text[pos..].find('{') {
        let start = pos + rel;
        // Parse one value from the slice: `StreamDeserializer::next` stops after the first
        // complete value and does not care that the rest of the tail is broken.
        let mut stream = serde_json::Deserializer::from_str(&text[start..]).into_iter::<Value>();
        match stream.next() {
            Some(Ok(v)) if v.is_object() => {
                let consumed = stream.byte_offset();
                items.push(v);
                if consumed == 0 {
                    break;
                }
                pos = start + consumed;
            }
            // A complete non-object entry means the array was not what we assumed, and an
            // error means the tail is the part that got cut off.
            _ => break,
        }
        // The array may have ended between the entry we just took and the next `{`.
        let next_brace = text[pos..].find('{');
        if let Some(bracket) = text[pos..].find(']') {
            if next_brace.is_none_or(|b| bracket < b) {
                break;
            }
        }
    }
    if items.is_empty() {
        None
    } else {
        Some(items)
    }
}

/// Format a [`SearchHit`] into the public memory-item shape.
/// Port of the `MemoryItem(...).model_dump()` + promotion logic.
/// Public wrapper for integration tests: format a hit into the API item shape.
#[doc(hidden)]
pub fn format_memory_item_for_test(hit: &SearchHit, exclude_score: bool) -> Value {
    format_memory_item(hit, exclude_score)
}

pub(crate) fn format_memory_item(hit: &SearchHit, exclude_score: bool) -> Value {
    let payload = &hit.payload;
    let mut item = Map::new();
    item.insert("id".into(), Value::String(hit.id.clone()));
    item.insert(
        "memory".into(),
        payload
            .get("data")
            .cloned()
            .unwrap_or(Value::String(String::new())),
    );
    if let Some(h) = payload.get("hash") {
        item.insert("hash".into(), h.clone());
    }
    if let Some(c) = payload.get("created_at") {
        item.insert("created_at".into(), c.clone());
    }
    if let Some(u) = payload.get("updated_at") {
        item.insert("updated_at".into(), u.clone());
    }
    // Official exports always carry this key (null when unset); mirror that shape so an
    // export from either side diffs cleanly. Kept out of `metadata` below.
    item.insert(
        "expiration_date".into(),
        payload
            .get("expiration_date")
            .cloned()
            .unwrap_or(Value::Null),
    );
    // Categories are a top-level array on official exports too; default to [] so every
    // item has a stable shape. Kept out of `metadata` below.
    item.insert(
        "categories".into(),
        payload
            .get("categories")
            .cloned()
            .unwrap_or_else(|| json!([])),
    );
    if !exclude_score && hit.score != 0.0 {
        item.insert("score".into(), json!(hit.score));
    }

    for key in PROMOTED_KEYS {
        if let Some(v) = payload.get(*key) {
            item.insert((*key).to_string(), v.clone());
        }
    }

    let core_and_promoted: [&str; 10] = [
        "data",
        "hash",
        "created_at",
        "updated_at",
        "id",
        "text_lemmatized",
        "attributed_to",
        "score",
        "expiration_date",
        "categories",
    ];
    let mut additional = Map::new();
    for (k, v) in payload {
        if core_and_promoted.contains(&k.as_str()) || PROMOTED_KEYS.contains(&k.as_str()) {
            continue;
        }
        additional.insert(k.clone(), v.clone());
    }
    if !additional.is_empty() {
        item.insert("metadata".into(), Value::Object(additional));
    }

    Value::Object(item)
}

#[cfg(test)]
mod salvage_tests {
    use super::salvage_memory_array;

    #[test]
    fn keeps_the_entries_that_were_complete() {
        // Exactly what a cut-off answer looks like: two whole objects, then the budget ends.
        let cut = r#"{"memory": [{"id": "0", "text": "moved to Hangzhou"}, {"id": "1", "text": "rides 15 minutes"#;
        assert!(super::parse_memory_array(cut).is_none(), "precondition");
        let kept = salvage_memory_array(cut).expect("salvaged");
        assert_eq!(kept.len(), 1, "only the complete entry survives: {kept:?}");
        assert_eq!(kept[0]["text"], "moved to Hangzhou");
    }

    #[test]
    fn keeps_everything_before_a_cut_in_a_long_answer() {
        let cut = r#"{"memory": [{"id": "0", "text": "a"}, {"id": "1", "text": "b"}, {"id": "2", "text": "c"#;
        assert_eq!(salvage_memory_array(cut).unwrap().len(), 2);
    }

    #[test]
    fn refuses_prose_that_merely_contains_braces() {
        let prose = r#"I would store {"id": "0"} but the model was cut"#;
        assert!(salvage_memory_array(prose).is_none());
    }

    #[test]
    fn refuses_an_array_of_non_objects() {
        assert!(salvage_memory_array(r#"{"memory": ["a", "b"]}"#).is_none());
    }

    #[test]
    fn a_complete_answer_is_not_affected() {
        let whole = r#"{"memory": [{"id": "0", "text": "x"}]}"#;
        assert_eq!(super::parse_memory_array(whole).unwrap().len(), 1);
        assert_eq!(salvage_memory_array(whole).unwrap().len(), 1);
    }

    #[test]
    fn an_empty_array_salvages_nothing() {
        assert!(salvage_memory_array(r#"{"memory": []"#).is_none());
    }
}
