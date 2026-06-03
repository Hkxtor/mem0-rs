//! Ported from mem0 `tests/memory/test_storage.py`.
//!
//! Adapted: schema is checked behaviorally (write-all-fields then read) since
//! the Rust `HistoryStore` encapsulates its connection; the legacy-schema
//! migration test is omitted (a fresh Rust store has no pre-existing legacy
//! tables to migrate). Persistence-across-reopen is covered.

use mem0_core::history::HistoryStore;
use uuid::Uuid;

fn mem() -> HistoryStore {
    HistoryStore::new(":memory:").unwrap()
}

#[test]
fn initialization_memory_and_file() {
    let _m = HistoryStore::new(":memory:").unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("h.db");
    let _f = HistoryStore::new(path.to_str().unwrap()).unwrap();
}

#[test]
fn add_history_basic_roundtrip() {
    let store = mem();
    let id = Uuid::new_v4().to_string();
    store
        .add_history(
            &id,
            Some("Old memory content"),
            Some("New memory content"),
            "ADD",
            Some("2026-01-01T00:00:00Z"),
            Some("2026-01-01T00:00:00Z"),
            0,
            Some("test_actor"),
            Some("user"),
        )
        .unwrap();
    let rows = store.get_history(&id).unwrap();
    assert_eq!(rows.len(), 1);
    let r = &rows[0];
    assert_eq!(r.memory_id, id);
    assert_eq!(r.old_memory.as_deref(), Some("Old memory content"));
    assert_eq!(r.new_memory.as_deref(), Some("New memory content"));
    assert_eq!(r.event, "ADD");
    assert_eq!(r.actor_id.as_deref(), Some("test_actor"));
    assert_eq!(r.role.as_deref(), Some("user"));
    assert!(!r.is_deleted);
}

#[test]
fn add_history_optional_params() {
    for (old, new, is_deleted) in [
        (None, Some("New memory"), 0i64),
        (Some("Old memory"), None, 1),
        (None, None, 1),
    ] {
        let store = mem();
        let id = Uuid::new_v4().to_string();
        store
            .add_history(
                &id,
                old,
                new,
                "UPDATE",
                None,
                Some("2026-01-01T00:00:00Z"),
                is_deleted,
                Some("a"),
                Some("user"),
            )
            .unwrap();
        let r = &store.get_history(&id).unwrap()[0];
        assert_eq!(r.old_memory.as_deref(), old);
        assert_eq!(r.new_memory.as_deref(), new);
        assert_eq!(r.is_deleted, is_deleted != 0);
    }
}

#[test]
fn add_history_generates_unique_ids() {
    let store = mem();
    let id = Uuid::new_v4().to_string();
    for i in 0..3 {
        store
            .add_history(
                &id,
                Some(&format!("Memory {i}")),
                Some(&format!("Updated {i}")),
                if i == 0 { "ADD" } else { "UPDATE" },
                Some(&format!("2026-01-01T00:00:0{i}Z")),
                None,
                0,
                None,
                None,
            )
            .unwrap();
    }
    let rows = store.get_history(&id).unwrap();
    assert_eq!(rows.len(), 3);
    let unique: std::collections::HashSet<_> = rows.iter().map(|r| r.id.clone()).collect();
    assert_eq!(unique.len(), 3);
}

#[test]
fn get_history_empty() {
    assert!(mem().get_history("non-existent-id").unwrap().is_empty());
}

#[test]
fn get_history_chronological_ordering() {
    let store = mem();
    let id = Uuid::new_v4().to_string();
    let timestamps = [
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:01Z",
        "2026-01-01T00:00:02Z",
    ];
    for (i, ts) in timestamps.iter().enumerate() {
        store
            .add_history(
                &id,
                Some(&format!("Memory {i}")),
                Some(&format!("Memory {}", i + 1)),
                if i == 0 { "ADD" } else { "UPDATE" },
                Some(ts),
                Some(ts),
                0,
                None,
                None,
            )
            .unwrap();
    }
    let rows = store.get_history(&id).unwrap();
    let got: Vec<String> = rows.iter().map(|r| r.created_at.clone().unwrap()).collect();
    let mut sorted = got.clone();
    sorted.sort();
    assert_eq!(got, sorted);
}

#[test]
fn migration_preserves_data_across_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("h.db");
    let path_str = path.to_str().unwrap();
    let id = Uuid::new_v4().to_string();

    {
        let store = HistoryStore::new(path_str).unwrap();
        store
            .add_history(
                &id,
                Some("o"),
                Some("n"),
                "ADD",
                Some("2026-01-01T00:00:00Z"),
                None,
                0,
                None,
                None,
            )
            .unwrap();
    }

    let store2 = HistoryStore::new(path_str).unwrap();
    let rows = store2.get_history(&id).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].memory_id, id);
    assert_eq!(rows[0].new_memory.as_deref(), Some("n"));
}

#[test]
fn large_batch_operations() {
    let store = mem();
    let ids: Vec<String> = (0..1000).map(|_| Uuid::new_v4().to_string()).collect();
    for (i, id) in ids.iter().enumerate() {
        store
            .add_history(
                id,
                None,
                Some(&format!("Batch memory {i}")),
                "ADD",
                None,
                None,
                0,
                None,
                None,
            )
            .unwrap();
    }
    for id in ids.iter().take(10) {
        assert_eq!(store.get_history(id).unwrap().len(), 1);
    }
}
