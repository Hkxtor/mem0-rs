//! End-to-end memory-operation benchmarks (framework overhead, no network).
//!
//! Uses mock embedder + mock LLM + embedded store, so the measured cost is the
//! orchestration path (message parsing, hashing, lemmatization, SQLite history,
//! payload build, vector insert/cosine + BM25 search) — the part that differs
//! between the Rust port and Python mem0.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mem0::{AddOptions, JsonMap, Memory, MemoryConfig, SearchOptions};
use serde_json::json;
use tokio::runtime::Runtime;

fn build() -> (Runtime, Memory) {
    let rt = Runtime::new().unwrap();
    let config = MemoryConfig::from_json(
        r#"{
            "embedder": { "provider": "mock" },
            "llm": { "provider": "mock" },
            "vector_store": { "provider": "embedded" },
            "history_db_path": ":memory:"
        }"#,
    )
    .unwrap();
    let mem = mem0::from_config(config).unwrap();
    (rt, mem)
}

fn umap(user: &str) -> JsonMap {
    let mut m = JsonMap::new();
    m.insert("user_id".into(), json!(user));
    m
}

fn raw(user: &str) -> AddOptions {
    AddOptions {
        user_id: Some(user.into()),
        infer: Some(false),
        ..Default::default()
    }
}

fn bench_add(c: &mut Criterion) {
    let (rt, mem) = build();
    let mut i = 0u64;
    c.bench_function("memory_add_raw", |b| {
        b.iter(|| {
            i += 1;
            rt.block_on(mem.add(
                black_box(format!("memory item {i} about hiking")),
                raw("u1"),
            ))
            .unwrap()
        })
    });
}

fn bench_search(c: &mut Criterion) {
    let (rt, mem) = build();
    // Seed 1000 memories.
    rt.block_on(async {
        for i in 0..1000 {
            mem.add(
                format!("memory {i} about hiking cooking travel and music"),
                raw("u1"),
            )
            .await
            .unwrap();
        }
    });
    c.bench_function("memory_search_over_1000", |b| {
        b.iter(|| {
            rt.block_on(mem.search(
                black_box("hiking travel"),
                &umap("u1"),
                SearchOptions::default(),
            ))
            .unwrap()
        })
    });
}

criterion_group!(benches, bench_add, bench_search);
criterion_main!(benches);
