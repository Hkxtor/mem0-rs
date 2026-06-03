//! Hybrid-retrieval scoring ported from `mem0/mem0/utils/scoring.py`, plus a
//! BM25 corpus scorer used by the in-process vector store's keyword search.

use crate::types::{JsonMap, SearchHit};
use std::collections::HashMap;

/// Weight applied to entity-boost contributions. Matches Python `ENTITY_BOOST_WEIGHT`.
pub const ENTITY_BOOST_WEIGHT: f64 = 0.5;

/// Query-length-adaptive sigmoid `(midpoint, steepness)` for BM25 normalization.
/// Port of `get_bm25_params`.
pub fn get_bm25_params(lemmatized: &str) -> (f64, f64) {
    let num_terms = if lemmatized.is_empty() {
        1
    } else {
        lemmatized.split_whitespace().count()
    };
    if num_terms <= 3 {
        (5.0, 0.7)
    } else if num_terms <= 6 {
        (7.0, 0.6)
    } else if num_terms <= 9 {
        (9.0, 0.5)
    } else if num_terms <= 15 {
        (10.0, 0.5)
    } else {
        (12.0, 0.5)
    }
}

/// Logistic-sigmoid normalization of a raw BM25 score into `[0, 1]`.
/// Port of `normalize_bm25`.
pub fn normalize_bm25(raw_score: f64, midpoint: f64, steepness: f64) -> f64 {
    1.0 / (1.0 + (-steepness * (raw_score - midpoint)).exp())
}

/// Additively score and rank candidates. Port of `score_and_rank`.
///
/// `threshold` gates the semantic score *before* combining. The divisor adapts:
/// semantic only = 1.0; +1.0 if any BM25; +`ENTITY_BOOST_WEIGHT` if any entity.
pub fn score_and_rank(
    semantic_results: &[SearchHit],
    bm25_scores: &HashMap<String, f64>,
    entity_boosts: &HashMap<String, f64>,
    threshold: f64,
    top_k: usize,
) -> Vec<SearchHit> {
    let has_bm25 = !bm25_scores.is_empty();
    let has_entity = !entity_boosts.is_empty();

    let mut max_possible = 1.0;
    if has_bm25 {
        max_possible += 1.0;
    }
    if has_entity {
        max_possible += ENTITY_BOOST_WEIGHT;
    }

    let mut scored: Vec<SearchHit> = Vec::new();
    for result in semantic_results {
        let semantic_score = result.score as f64;
        if semantic_score < threshold {
            continue;
        }
        let bm25_score = bm25_scores.get(&result.id).copied().unwrap_or(0.0);
        let entity_boost = entity_boosts.get(&result.id).copied().unwrap_or(0.0);
        let raw_combined = semantic_score + bm25_score + entity_boost;
        let combined = (raw_combined / max_possible).min(1.0);
        scored.push(SearchHit {
            id: result.id.clone(),
            score: combined as f32,
            payload: result.payload.clone(),
        });
    }

    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(top_k);
    scored
}

/// Compute raw Okapi BM25 scores for `query_lemmatized` over a corpus of
/// `(id, lemmatized_text)` documents. Used by the in-process store's keyword
/// search; the Python relies on the backing store's native full-text search.
pub fn bm25_scores(query_lemmatized: &str, docs: &[(String, String)]) -> HashMap<String, f64> {
    const K1: f64 = 1.5;
    const B: f64 = 0.75;

    let query_terms: Vec<&str> = query_lemmatized.split_whitespace().collect();
    let mut scores: HashMap<String, f64> = HashMap::new();
    if query_terms.is_empty() || docs.is_empty() {
        return scores;
    }

    // Tokenize documents.
    let tokenized: Vec<(String, Vec<String>)> = docs
        .iter()
        .map(|(id, text)| {
            (
                id.clone(),
                text.split_whitespace().map(|s| s.to_string()).collect(),
            )
        })
        .collect();

    let n = tokenized.len() as f64;
    let avgdl: f64 = tokenized.iter().map(|(_, t)| t.len() as f64).sum::<f64>() / n.max(1.0);

    // Document frequency per unique query term.
    let mut df: HashMap<&str, f64> = HashMap::new();
    for term in &query_terms {
        let count = tokenized
            .iter()
            .filter(|(_, toks)| toks.iter().any(|w| w == term))
            .count() as f64;
        df.insert(term, count);
    }

    for (id, toks) in &tokenized {
        let dl = toks.len() as f64;
        let mut score = 0.0;
        for term in &query_terms {
            let f = toks.iter().filter(|w| w.as_str() == *term).count() as f64;
            if f == 0.0 {
                continue;
            }
            let n_q = *df.get(term).unwrap_or(&0.0);
            // Okapi BM25 idf with +1 to keep it non-negative.
            let idf = (((n - n_q + 0.5) / (n_q + 0.5)) + 1.0).ln();
            let denom = f + K1 * (1.0 - B + B * dl / avgdl.max(1e-9));
            score += idf * (f * (K1 + 1.0)) / denom;
        }
        if score > 0.0 {
            scores.insert(id.clone(), score);
        }
    }
    scores
}

/// Build a `(id, lemmatized_text)` corpus from search hits' payloads, reading the
/// `text_lemmatized` field (falling back to lemmatizing `data`).
pub fn corpus_from_hits(hits: &[SearchHit]) -> Vec<(String, String)> {
    hits.iter()
        .map(|h| {
            let lemma = payload_lemmatized(&h.payload);
            (h.id.clone(), lemma)
        })
        .collect()
}

fn payload_lemmatized(payload: &JsonMap) -> String {
    if let Some(v) = payload.get("text_lemmatized").and_then(|v| v.as_str()) {
        return v.to_string();
    }
    if let Some(v) = payload.get("data").and_then(|v| v.as_str()) {
        return crate::nlp::lemmatize_for_bm25(v);
    }
    String::new()
}
