//! Vector store factory. Port of `VectorStoreFactory`.

use crate::config::VectorStoreSettings;
use crate::embedded::EmbeddedVectorStore;
use mem0_core::error::{Mem0Error, Result};
use mem0_core::traits::VectorStore;
use serde_json::Value;

/// Build a [`VectorStore`] for the given provider name and config object.
pub fn build_vector_store(provider: &str, config: &Value) -> Result<Box<dyn VectorStore>> {
    let settings = VectorStoreSettings::from_value(config)?;
    match provider {
        "embedded" | "memory" | "in_memory" => Ok(Box::new(EmbeddedVectorStore::new(
            &settings.collection_name(),
            settings.path.clone(),
        )?)),
        "qdrant" => build_qdrant(settings),
        "pgvector" => build_pgvector(settings),
        other => Err(Mem0Error::configuration(format!(
            "Unsupported vector store provider: {other}"
        ))),
    }
}

#[cfg(feature = "qdrant")]
fn build_qdrant(settings: VectorStoreSettings) -> Result<Box<dyn VectorStore>> {
    Ok(Box::new(crate::qdrant::QdrantStore::new(settings)?))
}

#[cfg(not(feature = "qdrant"))]
fn build_qdrant(_settings: VectorStoreSettings) -> Result<Box<dyn VectorStore>> {
    Err(Mem0Error::configuration(
        "qdrant backend not enabled; rebuild mem0-vector with `--features qdrant`",
    ))
}

#[cfg(feature = "pgvector")]
fn build_pgvector(settings: VectorStoreSettings) -> Result<Box<dyn VectorStore>> {
    Ok(Box::new(crate::pgvector::PgVectorStore::new(settings)?))
}

#[cfg(not(feature = "pgvector"))]
fn build_pgvector(_settings: VectorStoreSettings) -> Result<Box<dyn VectorStore>> {
    Err(Mem0Error::configuration(
        "pgvector backend not enabled; rebuild mem0-vector with `--features pgvector`",
    ))
}
