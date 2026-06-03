//! Vector store backends for the mem0 Rust port.
//!
//! The default `embedded` backend is an in-process, optionally file-persisted
//! store with native BM25 keyword search — it requires no external services and
//! underpins the single-binary deployment. The `qdrant` and `pgvector` backends
//! are feature-gated for users who want external scale.

pub mod config;
pub mod embedded;
pub mod factory;

#[cfg(feature = "qdrant")]
pub mod qdrant;

#[cfg(feature = "pgvector")]
pub mod pgvector;

pub use config::VectorStoreSettings;
pub use embedded::EmbeddedVectorStore;
pub use factory::build_vector_store;
