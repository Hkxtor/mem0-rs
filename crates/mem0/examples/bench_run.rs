//! Standalone timing driver for the Python-vs-Rust comparison.
//!
//! Runs `N` raw adds + `M` searches + one `get_all` through the mock embedder +
//! embedded store (no network, `infer=false`), and prints timings + peak RSS as
//! JSON. Usage: `bench_run [N=2000] [M=500]`.

use mem0::{AddOptions, JsonMap, SearchOptions};
use serde_json::json;
use std::time::Instant;

fn peak_rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0)
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let m: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(500);

    let mem = mem0::from_json(
        r#"{
            "embedder": { "provider": "mock" },
            "llm": { "provider": "mock" },
            "vector_store": { "provider": "embedded" },
            "history_db_path": ":memory:"
        }"#,
    )
    .expect("build memory");

    let opts = || AddOptions {
        user_id: Some("u1".into()),
        infer: Some(false),
        ..Default::default()
    };
    let mut filters = JsonMap::new();
    filters.insert("user_id".into(), json!("u1"));

    let t0 = Instant::now();
    for i in 0..n {
        mem.add(
            format!("memory item {i} about hiking cooking travel"),
            opts(),
        )
        .await
        .unwrap();
    }
    let add_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    for _ in 0..m {
        mem.search("hiking travel", &filters, SearchOptions::default())
            .await
            .unwrap();
    }
    let search_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let t2 = Instant::now();
    let _ = mem.get_all(&filters, 20).await.unwrap();
    let get_all_ms = t2.elapsed().as_secs_f64() * 1000.0;

    println!(
        "{}",
        json!({
            "impl": "rust",
            "n_add": n,
            "add_ms": add_ms,
            "n_search": m,
            "search_ms": search_ms,
            "get_all_ms": get_all_ms,
            "peak_rss_kb": peak_rss_kb(),
        })
    );
}
