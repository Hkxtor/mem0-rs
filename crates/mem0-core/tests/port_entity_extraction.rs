//! Ported from mem0 `tests/utils/test_entity_extraction.py`.
//!
//! Adapted: the Rust extractor implements the dependency-free PROPER + QUOTED
//! subset (the spaCy POS/dependency-driven COMPOUND/NOUN cases are out of
//! scope), so `test_compound_nouns` is intentionally omitted. All other cases
//! hold against the Rust implementation.

use mem0_core::nlp::{extract_entities, extract_entities_batch};
use std::collections::BTreeSet;

#[test]
fn proper_nouns() {
    let entities = extract_entities("John Smith works at Google on machine learning projects");
    let texts: Vec<String> = entities.iter().map(|(_, t)| t.clone()).collect();
    assert!(texts
        .iter()
        .any(|t| t.contains("John") || t.contains("Google")));
}

#[test]
fn quoted_text() {
    let entities = extract_entities("She is reading \"The Great Gatsby\" this week");
    let texts: Vec<String> = entities.iter().map(|(_, t)| t.clone()).collect();
    assert!(texts.iter().any(|t| t.contains("Great Gatsby")));
}

#[test]
fn empty_string() {
    assert!(extract_entities("").is_empty());
}

#[test]
fn no_entities() {
    let entities = extract_entities("I like things and stuff");
    let texts: Vec<String> = entities.iter().map(|(_, t)| t.to_lowercase()).collect();
    assert!(!texts.contains(&"things".to_string()));
    assert!(!texts.contains(&"stuff".to_string()));
}

#[test]
fn deduplication() {
    let entities = extract_entities("Google is great. I love working at Google.");
    let count = entities
        .iter()
        .filter(|(_, t)| t.contains("Google"))
        .count();
    assert!(count <= 1);
}

#[test]
fn returns_tuples() {
    let entities = extract_entities("John Smith lives in New York City");
    for (etype, etext) in &entities {
        assert!(["PROPER", "QUOTED", "COMPOUND", "NOUN"].contains(&etype.as_str()));
        assert!(!etext.is_empty());
    }
}

#[test]
fn batch_processing() {
    let texts = vec![
        "John works at Google".to_string(),
        "Mary lives in Paris".to_string(),
        "The cat sat on the mat".to_string(),
    ];
    let results = extract_entities_batch(&texts);
    assert_eq!(results.len(), 3);
}

#[test]
fn batch_empty_input() {
    assert!(extract_entities_batch(&[]).is_empty());
}

#[test]
fn batch_consistency_with_single() {
    let text = "John Smith works at Google headquarters";
    let single: BTreeSet<String> = extract_entities(text).into_iter().map(|(_, t)| t).collect();
    let batch = extract_entities_batch(&[text.to_string()]);
    assert_eq!(batch.len(), 1);
    let batch_set: BTreeSet<String> = batch[0].iter().map(|(_, t)| t.clone()).collect();
    assert_eq!(single, batch_set);
}
