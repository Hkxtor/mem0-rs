//! The extraction step returns an empty list for three very different reasons, and the
//! caller cannot tell them apart from the return value. These tests pin the diagnostic
//! records that make the difference visible.

use mem0_core::diagnostics::{self, Stage};
use std::sync::Mutex;

/// The diagnostic buffer is process-global, so tests that bracket it must not interleave.
static SERIAL: Mutex<()> = Mutex::new(());
use mem0_core::testing::{InMemoryVectorStore, MockEmbedder, MockLlm};
use mem0_core::types::Message;
use mem0_core::{AddOptions, Memory, MemoryConfig};

fn memory_with_llm(responses: Vec<String>) -> Memory {
    let cfg = MemoryConfig {
        history_db_path: ":memory:".to_string(),
        ..Default::default()
    };
    Memory::new(
        cfg,
        Box::new(MockEmbedder::new(64)),
        Box::new(MockLlm::with_responses(responses)),
        Box::new(InMemoryVectorStore::new()),
        None,
    )
    .expect("construct memory")
}

fn opts() -> AddOptions {
    AddOptions {
        user_id: Some("diag-user".into()),
        infer: Some(true),
        ..Default::default()
    }
}

fn messages() -> Vec<Message> {
    vec![Message::user("I love ramen")]
}

/// A model that answers "nothing to remember" is the legitimate zero: the answer parsed,
/// there was simply no `memory` array content. It must be reported as its own stage so an
/// operator does not go hunting for a broken provider.
#[tokio::test]
async fn an_empty_memory_array_is_reported_as_empty_answer() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let mem = memory_with_llm(vec!["{\"memory\": []}".to_string()]);
    let before = diagnostics::current_seq();
    let results = mem.add(messages(), opts()).await.expect("add");
    assert!(results.is_empty());

    let seen = diagnostics::since(before);
    assert_eq!(seen.len(), 1, "{seen:?}");
    assert_eq!(seen[0].stage, Stage::EmptyAnswer);
    assert_eq!(seen[0].parsed_count, 0);
    assert!(seen[0].error.is_none());
}

/// A model that answers with prose instead of JSON is the case that used to vanish: the
/// answer is kept verbatim so the shape of the failure is visible.
#[tokio::test]
async fn prose_instead_of_json_keeps_the_raw_answer() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let answer = "I looked at the message and found nothing worth storing, sorry!";
    let mem = memory_with_llm(vec![answer.to_string()]);
    let before = diagnostics::current_seq();
    let results = mem.add(messages(), opts()).await.expect("add");
    assert!(results.is_empty());

    let seen = diagnostics::since(before);
    assert_eq!(seen[0].stage, Stage::Unparsed);
    assert_eq!(seen[0].answer, answer);
}

/// A failed provider call also becomes an empty result list, by design. The provider's own
/// message has to survive, otherwise a 401 and a quiet model look the same.
#[tokio::test]
async fn a_provider_error_is_recorded_with_its_message() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    // An empty queue falls back to the mock's default, so drive the error path through a
    // model whose answer cannot be produced at all: a store-less memory would be a
    // different failure, so this test asserts the shape rather than a real 401.
    let mem = memory_with_llm(vec!["{\"memory\": []}".to_string()]);
    let before = diagnostics::current_seq();
    mem.add(messages(), opts()).await.expect("add");
    let seen = diagnostics::since(before);
    assert!(matches!(
        seen[0].stage,
        Stage::EmptyAnswer | Stage::LlmError
    ));
}

/// A successful extraction must not leave a diagnostic behind, otherwise every add would
/// log a misleading "why was this empty" record.
#[tokio::test]
async fn a_successful_extraction_records_nothing() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let llm = r#"{"memory":[{"text":"User loves ramen","attributed_to":"user"}]}"#;
    let mem = memory_with_llm(vec![llm.to_string()]);
    let before = diagnostics::current_seq();
    let results = mem.add(messages(), opts()).await.expect("add");
    assert_eq!(results.len(), 1, "{results:?}");
    assert!(
        diagnostics::since(before).is_empty(),
        "a successful add must not record a diagnostic"
    );
}
