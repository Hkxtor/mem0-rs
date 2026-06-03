//! Error taxonomy ported from `mem0/mem0/exceptions.py`.
//!
//! Each variant carries a machine-readable `code` and an optional user-facing
//! `suggestion`, mirroring the structured `MemoryError` hierarchy in Python.

use thiserror::Error;

/// Convenience result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Mem0Error>;

/// Structured error type for all mem0 operations.
#[derive(Debug, Error)]
pub enum Mem0Error {
    /// Input validation failed (`VALIDATION_*`).
    #[error("[{code}] {message}")]
    Validation {
        code: String,
        message: String,
        suggestion: Option<String>,
    },
    /// Authentication failure.
    #[error("[{code}] {message}")]
    Authentication {
        code: String,
        message: String,
        suggestion: Option<String>,
    },
    /// Rate limit exceeded.
    #[error("[{code}] {message}")]
    RateLimit {
        code: String,
        message: String,
        suggestion: Option<String>,
    },
    /// Requested memory not found.
    #[error("[{code}] {message}")]
    NotFound {
        code: String,
        message: String,
        suggestion: Option<String>,
    },
    /// Network / connectivity failure.
    #[error("[{code}] {message}")]
    Network {
        code: String,
        message: String,
        suggestion: Option<String>,
    },
    /// Invalid client configuration.
    #[error("[{code}] {message}")]
    Configuration {
        code: String,
        message: String,
        suggestion: Option<String>,
    },
    /// Vector store operation failed.
    #[error("[{code}] {message}")]
    VectorStore {
        code: String,
        message: String,
        suggestion: Option<String>,
    },
    /// Embedding generation failed.
    #[error("[{code}] {message}")]
    Embedding {
        code: String,
        message: String,
        suggestion: Option<String>,
    },
    /// LLM operation failed.
    #[error("[{code}] {message}")]
    Llm {
        code: String,
        message: String,
        suggestion: Option<String>,
    },
    /// Database operation failed.
    #[error("[{code}] {message}")]
    Database {
        code: String,
        message: String,
        suggestion: Option<String>,
    },
    /// Required dependency missing.
    #[error("[{code}] {message}")]
    Dependency {
        code: String,
        message: String,
        suggestion: Option<String>,
    },
    /// JSON (de)serialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// SQLite error.
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
}

impl Mem0Error {
    /// Returns the machine-readable error code, if any.
    pub fn code(&self) -> Option<&str> {
        match self {
            Mem0Error::Validation { code, .. }
            | Mem0Error::Authentication { code, .. }
            | Mem0Error::RateLimit { code, .. }
            | Mem0Error::NotFound { code, .. }
            | Mem0Error::Network { code, .. }
            | Mem0Error::Configuration { code, .. }
            | Mem0Error::VectorStore { code, .. }
            | Mem0Error::Embedding { code, .. }
            | Mem0Error::Llm { code, .. }
            | Mem0Error::Database { code, .. }
            | Mem0Error::Dependency { code, .. } => Some(code),
            _ => None,
        }
    }

    /// Build a validation error with an explicit code and suggestion.
    pub fn validation_code(
        code: impl Into<String>,
        message: impl Into<String>,
        suggestion: Option<String>,
    ) -> Self {
        Mem0Error::Validation {
            code: code.into(),
            message: message.into(),
            suggestion,
        }
    }

    /// Build a validation error with a default code.
    pub fn validation(message: impl Into<String>) -> Self {
        Mem0Error::validation_code("VALIDATION_000", message, None)
    }

    /// Build a not-found error.
    pub fn not_found(message: impl Into<String>) -> Self {
        Mem0Error::NotFound {
            code: "MEM_404".into(),
            message: message.into(),
            suggestion: Some("Please check the memory ID and ensure it exists".into()),
        }
    }

    /// Build a configuration error.
    pub fn configuration(message: impl Into<String>) -> Self {
        Mem0Error::Configuration {
            code: "CFG_001".into(),
            message: message.into(),
            suggestion: None,
        }
    }

    /// Build a vector store error.
    pub fn vector_store(message: impl Into<String>) -> Self {
        Mem0Error::VectorStore {
            code: "VECTOR_001".into(),
            message: message.into(),
            suggestion: Some("Please check your vector store configuration and connection".into()),
        }
    }

    /// Build an embedding error.
    pub fn embedding(message: impl Into<String>) -> Self {
        Mem0Error::Embedding {
            code: "EMBED_001".into(),
            message: message.into(),
            suggestion: Some("Please check your embedding model configuration".into()),
        }
    }

    /// Build an LLM error.
    pub fn llm(message: impl Into<String>) -> Self {
        Mem0Error::Llm {
            code: "LLM_001".into(),
            message: message.into(),
            suggestion: Some("Please check your LLM configuration and API key".into()),
        }
    }

    /// Build a database error.
    pub fn database(message: impl Into<String>) -> Self {
        Mem0Error::Database {
            code: "DB_001".into(),
            message: message.into(),
            suggestion: Some("Please check your database configuration and connection".into()),
        }
    }
}
