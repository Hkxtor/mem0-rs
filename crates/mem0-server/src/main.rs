//! mem0 server binary.
//!
//! Configure via `MEM0_CONFIG_FILE` (path to JSON) or `MEM0_CONFIG` (inline
//! JSON); otherwise a default embedded vector store + OpenAI providers is used.
//! Binds `0.0.0.0:$PORT` (default 8080).

use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let memory = Arc::new(mem0_server::build_memory_from_env()?);
    let app = mem0_server::app(memory);

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);
    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("mem0-server listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}
