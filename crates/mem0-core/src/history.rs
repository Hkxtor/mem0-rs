//! SQLite history + messages store ported from `mem0/mem0/memory/storage.py`.

use crate::error::Result;
use crate::types::Message;
use crate::util::now_utc_rfc3339;
use rusqlite::{params, Connection};
use serde::Serialize;
use std::sync::Mutex;
use uuid::Uuid;

/// A history row returned by [`HistoryStore::get_history`].
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct HistoryRecord {
    /// Row id (UUID).
    pub id: String,
    /// The memory this row concerns.
    pub memory_id: String,
    /// Previous memory text (for UPDATE/DELETE).
    pub old_memory: Option<String>,
    /// New memory text (for ADD/UPDATE).
    pub new_memory: Option<String>,
    /// Event name (`ADD`/`UPDATE`/`DELETE`).
    pub event: String,
    /// Creation timestamp.
    pub created_at: Option<String>,
    /// Update timestamp.
    pub updated_at: Option<String>,
    /// Whether the memory was deleted.
    pub is_deleted: bool,
    /// Actor id, if any.
    pub actor_id: Option<String>,
    /// Role, if any.
    pub role: Option<String>,
}

/// A stored message row returned by [`HistoryStore::get_last_messages`].
#[derive(Debug, Clone, PartialEq)]
pub struct StoredMessage {
    /// Message role.
    pub role: Option<String>,
    /// Message content.
    pub content: Option<String>,
    /// Speaker name.
    pub name: Option<String>,
    /// Creation timestamp.
    pub created_at: Option<String>,
}

/// Parameters for inserting a single history row.
#[derive(Debug, Clone, Default)]
pub struct NewHistory {
    /// The memory id.
    pub memory_id: String,
    /// Previous memory text.
    pub old_memory: Option<String>,
    /// New memory text.
    pub new_memory: Option<String>,
    /// Event name.
    pub event: String,
    /// Creation timestamp.
    pub created_at: Option<String>,
    /// Update timestamp.
    pub updated_at: Option<String>,
    /// Soft-delete flag.
    pub is_deleted: i64,
    /// Actor id.
    pub actor_id: Option<String>,
    /// Role.
    pub role: Option<String>,
}

/// SQLite-backed history and recent-message store.
pub struct HistoryStore {
    conn: Mutex<Connection>,
}

impl HistoryStore {
    /// Open (or create) the store at `db_path` (use `:memory:` for an in-memory db).
    pub fn new(db_path: &str) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.create_history_table()?;
        store.create_messages_table()?;
        Ok(store)
    }

    fn create_history_table(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "CREATE TABLE IF NOT EXISTS history (
                id           TEXT PRIMARY KEY,
                memory_id    TEXT,
                old_memory   TEXT,
                new_memory   TEXT,
                event        TEXT,
                created_at   DATETIME,
                updated_at   DATETIME,
                is_deleted   INTEGER,
                actor_id     TEXT,
                role         TEXT
            )",
            [],
        )?;
        Ok(())
    }

    fn create_messages_table(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_scope TEXT,
                role TEXT,
                content TEXT,
                name TEXT,
                created_at DATETIME
            )",
            [],
        )?;
        Ok(())
    }

    /// Insert a single history row. Port of `add_history`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_history(
        &self,
        memory_id: &str,
        old_memory: Option<&str>,
        new_memory: Option<&str>,
        event: &str,
        created_at: Option<&str>,
        updated_at: Option<&str>,
        is_deleted: i64,
        actor_id: Option<&str>,
        role: Option<&str>,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO history (id, memory_id, old_memory, new_memory, event,
                created_at, updated_at, is_deleted, actor_id, role)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                Uuid::new_v4().to_string(),
                memory_id,
                old_memory,
                new_memory,
                event,
                created_at,
                updated_at,
                is_deleted,
                actor_id,
                role,
            ],
        )?;
        Ok(())
    }

    /// Insert many history rows in one transaction. Port of `batch_add_history`.
    pub fn batch_add_history(&self, records: &[NewHistory]) -> Result<()> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO history (id, memory_id, old_memory, new_memory, event,
                    created_at, updated_at, is_deleted, actor_id, role)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            )?;
            for r in records {
                stmt.execute(params![
                    Uuid::new_v4().to_string(),
                    r.memory_id,
                    r.old_memory,
                    r.new_memory,
                    r.event,
                    r.created_at,
                    r.updated_at,
                    r.is_deleted,
                    r.actor_id,
                    r.role,
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Return all history rows for a memory, chronologically. Port of `get_history`.
    pub fn get_history(&self, memory_id: &str) -> Result<Vec<HistoryRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, memory_id, old_memory, new_memory, event,
                    created_at, updated_at, is_deleted, actor_id, role
             FROM history
             WHERE memory_id = ?1
             ORDER BY created_at ASC, DATETIME(updated_at) ASC",
        )?;
        let rows = stmt.query_map(params![memory_id], |row| {
            Ok(HistoryRecord {
                id: row.get(0)?,
                memory_id: row.get(1)?,
                old_memory: row.get(2)?,
                new_memory: row.get(3)?,
                event: row.get(4)?,
                created_at: row.get(5)?,
                updated_at: row.get(6)?,
                is_deleted: row.get::<_, i64>(7)? != 0,
                actor_id: row.get(8)?,
                role: row.get(9)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Persist messages for a scope, evicting all but the most recent 10.
    /// Port of `save_messages`.
    pub fn save_messages(&self, messages: &[Message], session_scope: &str) -> Result<()> {
        if messages.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        let now = now_utc_rfc3339();
        {
            let mut stmt = tx.prepare(
                "INSERT INTO messages (id, session_scope, role, content, name, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            )?;
            for m in messages {
                stmt.execute(params![
                    Uuid::new_v4().to_string(),
                    session_scope,
                    m.role,
                    m.content,
                    m.name,
                    now,
                ])?;
            }
        }
        tx.execute(
            "DELETE FROM messages WHERE session_scope = ?1 AND id NOT IN (
                SELECT id FROM (
                    SELECT id FROM messages WHERE session_scope = ?2
                    ORDER BY created_at DESC LIMIT 10
                )
            )",
            params![session_scope, session_scope],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Return up to `limit` most-recent messages for a scope, oldest-first.
    /// Port of `get_last_messages`.
    pub fn get_last_messages(&self, session_scope: &str, limit: i64) -> Result<Vec<StoredMessage>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT role, content, name, created_at FROM (
                SELECT role, content, name, created_at
                FROM messages
                WHERE session_scope = ?1
                ORDER BY created_at DESC
                LIMIT ?2
            ) ORDER BY created_at ASC",
        )?;
        let rows = stmt.query_map(params![session_scope, limit], |row| {
            Ok(StoredMessage {
                role: row.get(0)?,
                content: row.get(1)?,
                name: row.get(2)?,
                created_at: row.get(3)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Drop and recreate both tables. Port of `reset`.
    pub fn reset(&self) -> Result<()> {
        {
            let conn = self.conn.lock().unwrap();
            conn.execute("DROP TABLE IF EXISTS history", [])?;
            conn.execute("DROP TABLE IF EXISTS messages", [])?;
        }
        self.create_history_table()?;
        self.create_messages_table()?;
        Ok(())
    }
}
