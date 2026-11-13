//! Diagnostics for the extraction step.
//!
//! `Memory::add` swallows two failures by design: an LLM error becomes an empty result list
//! (matching the Python SDK, where a failed extraction must not fail the whole add), and an
//! unparseable answer is parsed to `[]`. Both look identical from the outside: `results: []`
//! with HTTP 200. That is correct behaviour and terrible for diagnosis — the operator cannot
//! tell "the model said nothing worth remembering" from "the model returned a 401" or "the
//! answer was JSON in an unexpected shape".
//!
//! This module keeps a small, bounded record of the last extraction attempts so a caller can
//! explain an empty result without changing `add`'s signature or its return type.
//!
//! Callers bracket an `add` with [`current_seq`] / [`since`]:
//!
//! ```ignore
//! let before = mem0_core::diagnostics::current_seq();
//! let results = mem.add(messages, opts).await?;
//! for d in mem0_core::diagnostics::since(before) { /* log it */ }
//! ```
//!
//! Bracketing by sequence number rather than by timestamp means concurrent adds cannot read
//! each other's records: each one only sees what was recorded after its own starting point.

use std::sync::{Mutex, OnceLock};

use serde::Serialize;

/// Which point of the extraction the record describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// The LLM call itself failed. `add` turned this into an empty result list.
    LlmError,
    /// The LLM answered with nothing (empty or whitespace-only content).
    EmptyAnswer,
    /// The LLM answered, but no `{"memory": [...]}` array could be found in it.
    Unparsed,
    /// The LLM answered and parsed, but every entry lacked usable text.
    NoUsableEntries,
    /// The answer was not valid JSON (an answer cut off at the output-token budget looks
    /// exactly like this), but complete entries could be recovered from it. The add does
    /// store them, so this is a warning, not a failure.
    Salvaged,
}

impl Stage {
    pub fn as_str(self) -> &'static str {
        match self {
            Stage::LlmError => "llm_error",
            Stage::EmptyAnswer => "empty_answer",
            Stage::Unparsed => "unparsed",
            Stage::NoUsableEntries => "no_usable_entries",
            Stage::Salvaged => "salvaged",
        }
    }
}

/// One extraction attempt, as seen from inside `add`.
#[derive(Debug, Clone, Serialize)]
pub struct ExtractionDiagnostic {
    /// Monotonic sequence number, assigned on record.
    pub seq: u64,
    /// RFC 3339 timestamp of the attempt.
    pub at: String,
    pub stage: Stage,
    /// The model's answer, verbatim. Empty when there was none.
    pub answer: String,
    /// How many entries survived parsing.
    pub parsed_count: usize,
    /// The provider's error, when `stage` is [`Stage::LlmError`].
    pub error: Option<String>,
}

/// How many records to keep. An `add` produces at most one, and the caller drains them
/// immediately, so this only needs to absorb bursts.
const CAPACITY: usize = 32;

struct Store {
    seq: u64,
    records: Vec<ExtractionDiagnostic>,
}

fn store() -> &'static Mutex<Store> {
    static STORE: OnceLock<Mutex<Store>> = OnceLock::new();
    STORE.get_or_init(|| {
        Mutex::new(Store {
            seq: 0,
            records: Vec::new(),
        })
    })
}

/// Record one attempt. Called from inside the extraction path.
pub fn record(stage: Stage, answer: &str, parsed_count: usize, error: Option<String>) {
    let Ok(mut s) = store().lock() else {
        // A poisoned lock must not take down an add: diagnostics are optional.
        return;
    };
    s.seq += 1;
    let seq = s.seq;
    if s.records.len() >= CAPACITY {
        s.records.remove(0);
    }
    s.records.push(ExtractionDiagnostic {
        seq,
        at: chrono::Utc::now().to_rfc3339(),
        stage,
        answer: answer.to_string(),
        parsed_count,
        error,
    });
}

/// The sequence number to bracket an `add` with.
pub fn current_seq() -> u64 {
    store().lock().map(|s| s.seq).unwrap_or(0)
}

/// Every record written after `seq`, oldest first.
pub fn since(seq: u64) -> Vec<ExtractionDiagnostic> {
    store()
        .lock()
        .map(|s| s.records.iter().filter(|r| r.seq > seq).cloned().collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bracketing_returns_only_records_after_the_marker() {
        let before = current_seq();
        record(Stage::EmptyAnswer, "", 0, None);
        let seen = since(before);
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].stage, Stage::EmptyAnswer);
        assert!(since(seen[0].seq).is_empty());
    }

    #[test]
    fn the_buffer_is_bounded() {
        let before = current_seq();
        for _ in 0..CAPACITY + 5 {
            record(Stage::Unparsed, "garbage", 0, None);
        }
        assert!(since(before).len() <= CAPACITY);
    }

    #[test]
    fn llm_errors_keep_their_message() {
        let before = current_seq();
        record(Stage::LlmError, "", 0, Some("401 unauthorized".into()));
        let seen = since(before);
        assert_eq!(seen[0].error.as_deref(), Some("401 unauthorized"));
    }
}
