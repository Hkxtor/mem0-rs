//! Runnable quickstart using mock/embedded providers (no API keys required).
//!
//! ```text
//! cargo run -p mem0 --example quickstart
//! ```

use mem0::{AddOptions, JsonMap, SearchOptions};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A fully in-process configuration: mock embedder + mock LLM + embedded store.
    let mem = mem0::from_json(
        r#"{
            "embedder": { "provider": "mock" },
            "llm": { "provider": "mock" },
            "vector_store": { "provider": "embedded" },
            "history_db_path": ":memory:"
        }"#,
    )?;

    let added = mem
        .add(
            "I love hiking in the mountains",
            AddOptions {
                user_id: Some("alice".into()),
                infer: Some(false),
                ..Default::default()
            },
        )
        .await?;
    println!("Added {} memory/memories: {:?}", added.len(), added);

    let mut filters = JsonMap::new();
    filters.insert("user_id".into(), json!("alice"));

    let all = mem.get_all(&filters, 10).await?;
    println!("All memories: {all}");

    let results = mem
        .search("hiking", &filters, SearchOptions::default())
        .await?;
    println!("Search results: {results}");

    Ok(())
}
