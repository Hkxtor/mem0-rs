//! Enumerations ported from `mem0/mem0/configs/enums.py`.

use serde::{Deserialize, Serialize};

/// The kind of memory being stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryType {
    /// `semantic_memory`
    #[serde(rename = "semantic_memory")]
    Semantic,
    /// `episodic_memory`
    #[serde(rename = "episodic_memory")]
    Episodic,
    /// `procedural_memory`
    #[serde(rename = "procedural_memory")]
    Procedural,
}

impl MemoryType {
    /// The wire string value (matches the Python `.value`).
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryType::Semantic => "semantic_memory",
            MemoryType::Episodic => "episodic_memory",
            MemoryType::Procedural => "procedural_memory",
        }
    }
}
