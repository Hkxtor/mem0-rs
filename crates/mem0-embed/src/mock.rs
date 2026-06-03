//! Mock embedder. Port of `embeddings/mock.py` (fixed 10-dim vector).

use async_trait::async_trait;
use mem0_core::error::Result;
use mem0_core::traits::{Embedder, MemoryAction};

/// A deterministic mock embedder returning a fixed 10-dimensional vector.
#[derive(Debug, Default)]
pub struct MockEmbedder;

#[async_trait]
impl Embedder for MockEmbedder {
    async fn embed(&self, _text: &str, _action: MemoryAction) -> Result<Vec<f32>> {
        Ok(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    }

    fn dims(&self) -> usize {
        10
    }
}
