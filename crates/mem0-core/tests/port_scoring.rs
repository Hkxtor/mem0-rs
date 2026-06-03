//! Ported from mem0 `tests/utils/test_scoring.py` (faithful — exact math).

use mem0_core::scoring::{get_bm25_params, normalize_bm25, score_and_rank, ENTITY_BOOST_WEIGHT};
use mem0_core::types::{JsonMap, SearchHit};
use serde_json::json;
use std::collections::HashMap;

fn hit(id: &str, score: f32, data: &str) -> SearchHit {
    let mut payload = JsonMap::new();
    if !data.is_empty() {
        payload.insert("data".into(), json!(data));
    }
    SearchHit {
        id: id.into(),
        score,
        payload,
    }
}

fn bm(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
    pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
}

// --- get_bm25_params ---

#[test]
fn short_query() {
    assert_eq!(get_bm25_params("hello world"), (5.0, 0.7));
}

#[test]
fn medium_query() {
    assert_eq!(get_bm25_params("one two three four five"), (7.0, 0.6));
}

#[test]
fn long_query() {
    let words = (0..20)
        .map(|i| format!("word{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    assert_eq!(get_bm25_params(&words), (12.0, 0.5));
}

#[test]
fn empty_lemmatized() {
    assert_eq!(get_bm25_params("").0, 5.0);
}

// --- normalize_bm25 ---

#[test]
fn at_midpoint() {
    assert!((normalize_bm25(5.0, 5.0, 0.7) - 0.5).abs() < 0.01);
}

#[test]
fn high_score() {
    assert!(normalize_bm25(20.0, 5.0, 0.7) > 0.99);
}

#[test]
fn low_score() {
    assert!(normalize_bm25(0.0, 5.0, 0.7) < 0.05);
}

#[test]
fn range() {
    for raw in [0.0, 1.0, 5.0, 10.0, 20.0, 50.0] {
        let s = normalize_bm25(raw, 5.0, 0.7);
        assert!((0.0..=1.0).contains(&s));
    }
}

// --- score_and_rank ---

#[test]
fn semantic_only() {
    let results = vec![hit("a", 0.9, "mem a"), hit("b", 0.5, "mem b")];
    let scored = score_and_rank(&results, &HashMap::new(), &HashMap::new(), 0.1, 10);
    assert_eq!(scored.len(), 2);
    assert!((scored[0].score - 0.9).abs() < 1e-4);
    assert!((scored[1].score - 0.5).abs() < 1e-4);
}

#[test]
fn semantic_plus_bm25() {
    let results = vec![hit("a", 0.8, "mem a"), hit("b", 0.6, "mem b")];
    let scored = score_and_rank(
        &results,
        &bm(&[("a", 0.3), ("b", 0.9)]),
        &HashMap::new(),
        0.1,
        10,
    );
    assert_eq!(scored[0].id, "b");
    assert!((scored[0].score - 0.75).abs() < 1e-4);
    assert_eq!(scored[1].id, "a");
    assert!((scored[1].score - 0.55).abs() < 1e-4);
}

#[test]
fn all_three_signals() {
    let results = vec![hit("a", 0.8, "mem a")];
    let scored = score_and_rank(&results, &bm(&[("a", 0.6)]), &bm(&[("a", 0.3)]), 0.1, 10);
    let expected = ((0.8 + 0.6 + 0.3) / 2.5) as f32;
    assert!((scored[0].score - expected).abs() < 1e-4);
}

#[test]
fn threshold_gates_on_semantic() {
    let results = vec![hit("a", 0.05, "mem a"), hit("b", 0.5, "mem b")];
    let scored = score_and_rank(&results, &bm(&[("a", 0.99)]), &HashMap::new(), 0.1, 10);
    assert_eq!(scored.len(), 1);
    assert_eq!(scored[0].id, "b");
}

#[test]
fn top_k_limit() {
    let results: Vec<SearchHit> = (0..20).map(|i| hit(&i.to_string(), 0.5, "")).collect();
    let scored = score_and_rank(&results, &HashMap::new(), &HashMap::new(), 0.1, 5);
    assert_eq!(scored.len(), 5);
}

#[test]
fn adaptive_divisor_semantic_only() {
    let results = vec![hit("a", 0.8, "")];
    let scored = score_and_rank(&results, &HashMap::new(), &HashMap::new(), 0.1, 10);
    assert!((scored[0].score - 0.8).abs() < 1e-4);
}

#[test]
fn adaptive_divisor_semantic_plus_entity() {
    let results = vec![hit("a", 0.8, "")];
    let scored = score_and_rank(&results, &HashMap::new(), &bm(&[("a", 0.3)]), 0.1, 10);
    let expected = ((0.8 + 0.3) / 1.5) as f32;
    assert!((scored[0].score - expected).abs() < 1e-4);
}

#[test]
fn empty_results() {
    let scored = score_and_rank(&[], &HashMap::new(), &HashMap::new(), 0.1, 10);
    assert!(scored.is_empty());
}

#[test]
fn score_clamped_to_1() {
    let results = vec![hit("a", 1.0, "")];
    let scored = score_and_rank(&results, &bm(&[("a", 1.0)]), &bm(&[("a", 0.5)]), 0.1, 10);
    assert!(scored[0].score <= 1.0);
}

#[test]
fn entity_boost_weight_value() {
    assert_eq!(ENTITY_BOOST_WEIGHT, 0.5);
}
