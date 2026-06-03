//! Core data types shared across the crate.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// A JSON object used for payloads, metadata, and filters.
pub type JsonMap = Map<String, Value>;

/// A single chat message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// `system` | `user` | `assistant`.
    pub role: String,
    /// The message content.
    pub content: String,
    /// Optional speaker name (maps to `actor_id` for non-inferred adds).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    /// Construct a message with the given role and content.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            name: None,
        }
    }
    /// Construct a `user` message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }
    /// Construct a `system` message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }
    /// Construct an `assistant` message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

/// A memory-management event, serialized as `ADD`/`UPDATE`/`DELETE`/`NONE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Event {
    /// New memory added.
    #[serde(rename = "ADD")]
    Add,
    /// Existing memory updated.
    #[serde(rename = "UPDATE")]
    Update,
    /// Existing memory deleted.
    #[serde(rename = "DELETE")]
    Delete,
    /// No change.
    #[serde(rename = "NONE")]
    None,
}

impl Event {
    /// The wire string value.
    pub fn as_str(&self) -> &'static str {
        match self {
            Event::Add => "ADD",
            Event::Update => "UPDATE",
            Event::Delete => "DELETE",
            Event::None => "NONE",
        }
    }
}

/// A record stored in the vector store: id, embedding, and payload metadata.
#[derive(Debug, Clone)]
pub struct VectorRecord {
    /// Stable UUID for the memory.
    pub id: String,
    /// The embedding vector.
    pub vector: Vec<f32>,
    /// Arbitrary payload (includes `data`, `hash`, timestamps, session ids, ...).
    pub payload: JsonMap,
}

/// A search/list result hit.
#[derive(Debug, Clone)]
pub struct SearchHit {
    /// The memory id.
    pub id: String,
    /// Similarity / combined score.
    pub score: f32,
    /// The stored payload.
    pub payload: JsonMap,
}

/// The result of adding a memory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AddResult {
    /// The memory id.
    pub id: String,
    /// The memory text.
    pub memory: String,
    /// The event (`ADD` for the additive pipeline).
    pub event: String,
    /// Actor id, when present (non-inferred adds with a message `name`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actor_id: Option<String>,
    /// Message role, when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
}

/// Accepted input shapes for [`crate::Memory::add`], mirroring Python's
/// `str | dict | list[dict]`.
pub enum MessagesInput {
    /// A single string becomes one `user` message.
    Text(String),
    /// A single message.
    One(Message),
    /// A list of messages.
    Many(Vec<Message>),
}

impl MessagesInput {
    /// Normalize into a vector of messages.
    pub fn into_messages(self) -> Vec<Message> {
        match self {
            MessagesInput::Text(s) => vec![Message::user(s)],
            MessagesInput::One(m) => vec![m],
            MessagesInput::Many(v) => v,
        }
    }
}

impl From<&str> for MessagesInput {
    fn from(s: &str) -> Self {
        MessagesInput::Text(s.to_string())
    }
}
impl From<String> for MessagesInput {
    fn from(s: String) -> Self {
        MessagesInput::Text(s)
    }
}
impl From<Message> for MessagesInput {
    fn from(m: Message) -> Self {
        MessagesInput::One(m)
    }
}
impl From<Vec<Message>> for MessagesInput {
    fn from(v: Vec<Message>) -> Self {
        MessagesInput::Many(v)
    }
}
