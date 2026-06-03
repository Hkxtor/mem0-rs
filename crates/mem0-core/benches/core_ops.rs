//! Criterion microbenchmarks for the CPU-bound core operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mem0_core::filters::matches_filters;
use mem0_core::nlp::{extract_entities, lemmatize_for_bm25};
use mem0_core::scoring::{bm25_scores, score_and_rank};
use mem0_core::types::{JsonMap, SearchHit};
use serde_json::json;
use std::collections::HashMap;

const SAMPLE: &str = "Marcus was promoted to Senior Engineer at Shopify last week \
after working toward it for two years, and celebrated with dinner at Osteria Francescana.";

fn bench_lemmatize(c: &mut Criterion) {
    c.bench_function("lemmatize_for_bm25", |b| {
        b.iter(|| lemmatize_for_bm25(black_box(SAMPLE)))
    });
}

fn bench_entities(c: &mut Criterion) {
    c.bench_function("extract_entities", |b| {
        b.iter(|| extract_entities(black_box(SAMPLE)))
    });
}

fn bench_bm25(c: &mut Criterion) {
    let corpus: Vec<(String, String)> = (0..1000)
        .map(|i| {
            (
                format!("id{i}"),
                format!("memory number {i} about hiking and cooking and travel"),
            )
        })
        .collect();
    let query = lemmatize_for_bm25("hiking travel");
    c.bench_function("bm25_scores_1000docs", |b| {
        b.iter(|| bm25_scores(black_box(&query), black_box(&corpus)))
    });
}

fn bench_score_and_rank(c: &mut Criterion) {
    let hits: Vec<SearchHit> = (0..1000)
        .map(|i| {
            let mut p = JsonMap::new();
            p.insert("data".into(), json!(format!("mem {i}")));
            SearchHit {
                id: format!("id{i}"),
                score: (i % 100) as f32 / 100.0,
                payload: p,
            }
        })
        .collect();
    let bm25: HashMap<String, f64> = (0..1000).map(|i| (format!("id{i}"), 0.5)).collect();
    c.bench_function("score_and_rank_1000_top20", |b| {
        b.iter(|| score_and_rank(black_box(&hits), black_box(&bm25), &HashMap::new(), 0.1, 20))
    });
}

fn bench_matches_filters(c: &mut Criterion) {
    let mut payload = JsonMap::new();
    payload.insert("user_id".into(), json!("u1"));
    payload.insert("category".into(), json!("food"));
    let mut filters = JsonMap::new();
    filters.insert("user_id".into(), json!("u1"));
    filters.insert("category".into(), json!({ "in": ["food", "travel"] }));
    c.bench_function("matches_filters_operator", |b| {
        b.iter(|| matches_filters(black_box(&payload), black_box(&filters)))
    });
}

criterion_group!(
    benches,
    bench_lemmatize,
    bench_entities,
    bench_bm25,
    bench_score_and_rank,
    bench_matches_filters
);
criterion_main!(benches);
