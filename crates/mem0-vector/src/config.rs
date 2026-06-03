//! Vector store configuration parsed from the `config` object of a
//! `{ provider, config }` vector-store spec.

use mem0_core::error::Result;
use serde::Deserialize;
use serde_json::Value;

/// Provider-agnostic vector store settings. Unknown fields are ignored.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct VectorStoreSettings {
    /// Collection / table name.
    pub collection_name: Option<String>,
    /// Embedding dimensionality.
    pub embedding_model_dims: Option<usize>,
    /// Filesystem path for the embedded store's JSON persistence (directory).
    pub path: Option<String>,

    // Qdrant
    /// Full Qdrant URL (e.g. `http://localhost:6333`).
    pub url: Option<String>,
    /// Qdrant host (with `port`).
    pub host: Option<String>,
    /// Qdrant port (with `host`).
    pub port: Option<u16>,
    /// Qdrant API key.
    pub api_key: Option<String>,
    /// Qdrant on-disk vectors flag.
    pub on_disk: Option<bool>,

    // pgvector
    /// Postgres database name.
    pub dbname: Option<String>,
    /// Postgres user.
    pub user: Option<String>,
    /// Postgres password.
    pub password: Option<String>,
    /// Full Postgres connection string (overrides individual params).
    pub connection_string: Option<String>,
    /// Use HNSW index (pgvector).
    pub hnsw: Option<bool>,
    /// Use DiskANN index (pgvector).
    pub diskann: Option<bool>,
}

impl VectorStoreSettings {
    /// Parse settings from a JSON config object.
    pub fn from_value(value: &Value) -> Result<Self> {
        if value.is_null() {
            return Ok(Self::default());
        }
        Ok(serde_json::from_value(value.clone())?)
    }

    /// Effective collection name (default `mem0`).
    pub fn collection_name(&self) -> String {
        self.collection_name
            .clone()
            .unwrap_or_else(|| "mem0".to_string())
    }

    /// Effective embedding dimensionality (default 1536).
    pub fn dims(&self) -> usize {
        self.embedding_model_dims.unwrap_or(1536)
    }
}
