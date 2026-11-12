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
    /// An upstream answered with a status that was not a success.
    ///
    /// Exists so a caller that pools credentials can act on the status — park a key that
    /// was rate limited, retire one whose credential was rejected — without parsing the
    /// message, which also means `Retry-After` survives instead of being dropped with the
    /// response. `Display` matches [`Mem0Error::Llm`] / [`Mem0Error::Embedding`] exactly,
    /// so nothing that only reads the text sees a difference.
    #[error("[{code}] {message}")]
    Http {
        code: String,
        message: String,
        suggestion: Option<String>,
        /// The upstream HTTP status, e.g. `429`.
        status: u16,
        /// The upstream's `Retry-After` in seconds, when it sent a usable one.
        retry_after: Option<u64>,
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
            | Mem0Error::Dependency { code, .. }
            | Mem0Error::Http { code, .. } => Some(code),
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

    /// Build an LLM error carrying the upstream status.
    ///
    /// Same code and message shape as [`Mem0Error::llm`], so the text an operator already
    /// sees does not change; the status and `Retry-After` ride along for callers that need
    /// them.
    pub fn llm_http(status: u16, retry_after: Option<u64>, message: impl Into<String>) -> Self {
        Mem0Error::Http {
            code: "LLM_001".into(),
            message: message.into(),
            suggestion: Some("Please check your LLM configuration and API key".into()),
            status,
            retry_after,
        }
    }

    /// Build an embedding error carrying the upstream status. See [`Mem0Error::llm_http`].
    pub fn embedding_http(
        status: u16,
        retry_after: Option<u64>,
        message: impl Into<String>,
    ) -> Self {
        Mem0Error::Http {
            code: "EMBED_001".into(),
            message: message.into(),
            suggestion: Some("Please check your embedding model configuration".into()),
            status,
            retry_after,
        }
    }

    /// The upstream status, when this error came from an unsuccessful HTTP response.
    pub fn http_status(&self) -> Option<u16> {
        match self {
            Mem0Error::Http { status, .. } => Some(*status),
            _ => None,
        }
    }

    /// The upstream's requested wait, in seconds, when it sent one.
    pub fn retry_after_secs(&self) -> Option<u64> {
        match self {
            Mem0Error::Http { retry_after, .. } => *retry_after,
            _ => None,
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

/// Seconds from a `Retry-After` header value.
///
/// The header allows either a delay in seconds or an HTTP date. Only the numeric form is
/// honoured: a date would mean carrying an HTTP-date parser for a value that is almost
/// always a number, and the caller has a sane default when this returns `None`.
pub fn parse_retry_after(value: Option<&str>) -> Option<u64> {
    value?.trim().parse::<u64>().ok()
}

#[cfg(test)]
mod http_status_tests {
    use super::*;

    #[test]
    fn a_status_survives_the_conversion_and_display_is_unchanged() {
        let e = Mem0Error::llm_http(
            429,
            Some(7),
            "OpenAI chat HTTP 429 Too Many Requests: slow down",
        );
        assert_eq!(e.http_status(), Some(429));
        assert_eq!(e.retry_after_secs(), Some(7));
        assert_eq!(e.code(), Some("LLM_001"));
        assert_eq!(
            e.to_string(),
            "[LLM_001] OpenAI chat HTTP 429 Too Many Requests: slow down"
        );
        // A plain llm() error carries no status: it did not come from a response.
        assert_eq!(Mem0Error::llm("boom").http_status(), None);
        assert_eq!(
            Mem0Error::embedding_http(401, None, "nope").code(),
            Some("EMBED_001")
        );
    }

    #[test]
    fn retry_after_reads_seconds_and_ignores_everything_else() {
        assert_eq!(parse_retry_after(Some("30")), Some(30));
        assert_eq!(parse_retry_after(Some(" 0 ")), Some(0));
        // An HTTP date is not a delay we can use without a date parser.
        assert_eq!(
            parse_retry_after(Some("Wed, 21 Oct 2015 07:28:00 GMT")),
            None
        );
        assert_eq!(parse_retry_after(Some("-5")), None);
        assert_eq!(parse_retry_after(None), None);
    }
}
