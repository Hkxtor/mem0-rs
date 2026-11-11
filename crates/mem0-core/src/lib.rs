//! Core of the mem0 Rust port.
//!
//! This crate is provider-agnostic: it defines the configuration model, the
//! memory data types, the verbatim-ported prompts, the NLP/scoring helpers, the
//! SQLite history store, the provider traits ([`Embedder`], [`Llm`],
//! [`VectorStore`], [`Reranker`]), and the [`Memory`] orchestrator. Concrete
//! providers live in sibling crates and are injected into [`Memory`].

pub mod config;
pub mod diagnostics;
pub mod enums;
pub mod error;
pub mod filters;
pub mod history;
pub mod memory;
pub mod nlp;
pub mod prompts;
pub mod scoring;
pub mod testing;
pub mod text;
pub mod traits;
pub mod types;
pub mod util;

pub use config::{EmbedderConfig, LlmConfig, MemoryConfig, RerankerConfig, VectorStoreConfig};
pub use enums::MemoryType;
pub use error::{Mem0Error, Result};
pub use memory::{AddOptions, Memory, SearchOptions};
pub use traits::{Embedder, GenerateOptions, Llm, MemoryAction, Reranker, VectorStore};
pub use types::{AddResult, Event, JsonMap, Message, MessagesInput, SearchHit, VectorRecord};
