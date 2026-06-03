//! pgvector store over `tokio-postgres`. Port of `vector_stores/pgvector.py`.
//!
//! Deviations (documented): point ids are stored as `TEXT` (the orchestrator
//! uses UUID strings) rather than the Python `UUID` column, and the cosine
//! distance returned by `<=>` is converted to a similarity (`1 - distance`) so
//! the score is consistent with the embedded/qdrant backends and the additive
//! `score_and_rank` (higher = better).
//!
//! Security: the collection name is interpolated into SQL identifiers, so it is
//! validated against a strict `[A-Za-z_][A-Za-z0-9_]*` pattern at construction.
//! Filter *values* are always passed as bound parameters.

use crate::config::VectorStoreSettings;
use async_trait::async_trait;
use mem0_core::error::{Mem0Error, Result};
use mem0_core::traits::VectorStore;
use mem0_core::types::{JsonMap, SearchHit, VectorRecord};
use pgvector::Vector;
use serde_json::Value;
use tokio::sync::{Mutex, OnceCell};
use tokio_postgres::types::ToSql;
use tokio_postgres::{Client, NoTls};

/// Validate that a string is a safe SQL identifier (no quoting tricks possible).
fn validate_identifier(name: &str) -> Result<()> {
    let mut chars = name.chars();
    let valid_start = chars
        .next()
        .map(|c| c.is_ascii_alphabetic() || c == '_')
        .unwrap_or(false);
    let valid_rest = name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if name.is_empty() || !valid_start || !valid_rest {
        return Err(Mem0Error::configuration(format!(
            "Invalid collection name '{name}': must match [A-Za-z_][A-Za-z0-9_]*"
        )));
    }
    Ok(())
}

/// pgvector-backed vector store.
pub struct PgVectorStore {
    conninfo: String,
    collection: String,
    dims: usize,
    hnsw: bool,
    client: OnceCell<Mutex<Client>>,
}

impl PgVectorStore {
    /// Construct a pgvector store from settings.
    pub fn new(settings: VectorStoreSettings) -> Result<Self> {
        let collection = settings.collection_name();
        validate_identifier(&collection)?;
        let conninfo = if let Some(cs) = &settings.connection_string {
            cs.clone()
        } else {
            let user = settings.user.clone().unwrap_or_else(|| "postgres".into());
            let password = settings.password.clone().unwrap_or_default();
            let host = settings.host.clone().unwrap_or_else(|| "localhost".into());
            let port = settings.port.unwrap_or(5432);
            let dbname = settings.dbname.clone().unwrap_or_else(|| "postgres".into());
            format!("postgresql://{user}:{password}@{host}:{port}/{dbname}")
        };
        Ok(Self {
            conninfo,
            collection,
            dims: settings.dims(),
            hnsw: settings.hnsw.unwrap_or(true),
            client: OnceCell::new(),
        })
    }

    async fn client(&self) -> Result<&Mutex<Client>> {
        self.client
            .get_or_try_init(|| async {
                let (client, connection) = tokio_postgres::connect(&self.conninfo, NoTls)
                    .await
                    .map_err(|e| Mem0Error::vector_store(format!("pg connect failed: {e}")))?;
                tokio::spawn(async move {
                    if let Err(e) = connection.await {
                        tracing::error!("pg connection error: {e}");
                    }
                });
                self.create_col(&client).await?;
                Ok::<Mutex<Client>, Mem0Error>(Mutex::new(client))
            })
            .await
    }

    async fn create_col(&self, client: &Client) -> Result<()> {
        client
            .batch_execute("CREATE EXTENSION IF NOT EXISTS vector")
            .await
            .map_err(pg)?;
        let create = format!(
            "CREATE TABLE IF NOT EXISTS \"{}\" (id TEXT PRIMARY KEY, vector vector({}), payload JSONB)",
            self.collection, self.dims
        );
        client.batch_execute(&create).await.map_err(pg)?;
        if self.hnsw {
            let idx = format!(
                "CREATE INDEX IF NOT EXISTS \"{c}_hnsw_idx\" ON \"{c}\" USING hnsw (vector vector_cosine_ops)",
                c = self.collection
            );
            let _ = client.batch_execute(&idx).await;
        }
        let gin = format!(
            "CREATE INDEX IF NOT EXISTS \"{c}_text_idx\" ON \"{c}\" USING gin(to_tsvector('simple', payload->>'text_lemmatized'))",
            c = self.collection
        );
        let _ = client.batch_execute(&gin).await;
        Ok(())
    }
}

fn pg(e: tokio_postgres::Error) -> Mem0Error {
    Mem0Error::vector_store(format!("pg error: {e}"))
}

/// A SQL parameter value with a concrete Postgres type.
enum Param {
    Text(String),
    Num(f64),
    TextArray(Vec<String>),
}

impl Param {
    fn as_sql(&self) -> &(dyn ToSql + Sync) {
        match self {
            Param::Text(s) => s,
            Param::Num(n) => n,
            Param::TextArray(a) => a,
        }
    }
}

/// Build a parameterized WHERE fragment (without the `WHERE` keyword) and its
/// params, starting numbering at `start_idx`. Port of `_build_filter_conditions`.
fn build_conditions(
    filters: &JsonMap,
    start_idx: &mut usize,
    params: &mut Vec<Param>,
) -> Vec<String> {
    let mut conditions = Vec::new();
    for (key, value) in filters {
        match key.as_str() {
            "$or" => {
                if let Some(arr) = value.as_array() {
                    let mut groups = Vec::new();
                    for sub in arr {
                        if let Some(obj) = sub.as_object() {
                            let sub_conds = build_conditions(obj, start_idx, params);
                            if !sub_conds.is_empty() {
                                groups.push(format!("({})", sub_conds.join(" AND ")));
                            }
                        }
                    }
                    if !groups.is_empty() {
                        conditions.push(format!("({})", groups.join(" OR ")));
                    }
                }
            }
            "$not" => {
                if let Some(arr) = value.as_array() {
                    let mut groups = Vec::new();
                    for sub in arr {
                        if let Some(obj) = sub.as_object() {
                            let sub_conds = build_conditions(obj, start_idx, params);
                            if !sub_conds.is_empty() {
                                groups.push(format!("({})", sub_conds.join(" AND ")));
                            }
                        }
                    }
                    if !groups.is_empty() {
                        conditions.push(format!("NOT ({})", groups.join(" OR ")));
                    }
                }
            }
            _ => {
                if value.as_str() == Some("*") {
                    let i = next(start_idx);
                    conditions.push(format!("payload ? ${i}"));
                    params.push(Param::Text(key.clone()));
                    continue;
                }
                if let Some(ops) = value.as_object() {
                    for (op, op_value) in ops {
                        push_op(key, op, op_value, start_idx, params, &mut conditions);
                    }
                } else if let Some(arr) = value.as_array() {
                    let i_key = next(start_idx);
                    let i_arr = next(start_idx);
                    conditions.push(format!("payload->>${i_key} = ANY(${i_arr})"));
                    params.push(Param::Text(key.clone()));
                    params.push(Param::TextArray(arr.iter().map(scalar_to_string).collect()));
                } else {
                    let i_key = next(start_idx);
                    let i_val = next(start_idx);
                    conditions.push(format!("payload->>${i_key} = ${i_val}"));
                    params.push(Param::Text(key.clone()));
                    params.push(Param::Text(scalar_to_string(value)));
                }
            }
        }
    }
    conditions
}

fn push_op(
    key: &str,
    op: &str,
    op_value: &Value,
    start_idx: &mut usize,
    params: &mut Vec<Param>,
    conditions: &mut Vec<String>,
) {
    let i_key = next(start_idx);
    match op {
        "eq" | "ne" => {
            let i_val = next(start_idx);
            let cmp = if op == "eq" { "=" } else { "!=" };
            conditions.push(format!("payload->>${i_key} {cmp} ${i_val}"));
            params.push(Param::Text(key.to_string()));
            params.push(Param::Text(scalar_to_string(op_value)));
        }
        "gt" | "gte" | "lt" | "lte" => {
            let i_val = next(start_idx);
            let cmp = match op {
                "gt" => ">",
                "gte" => ">=",
                "lt" => "<",
                _ => "<=",
            };
            conditions.push(format!("(payload->>${i_key})::numeric {cmp} ${i_val}"));
            params.push(Param::Text(key.to_string()));
            params.push(Param::Num(op_value.as_f64().unwrap_or(0.0)));
        }
        "in" | "nin" => {
            let i_arr = next(start_idx);
            let arr: Vec<String> = op_value
                .as_array()
                .map(|a| a.iter().map(scalar_to_string).collect())
                .unwrap_or_default();
            if op == "in" {
                conditions.push(format!("payload->>${i_key} = ANY(${i_arr})"));
            } else {
                conditions.push(format!("NOT (payload->>${i_key} = ANY(${i_arr}))"));
            }
            params.push(Param::Text(key.to_string()));
            params.push(Param::TextArray(arr));
        }
        "contains" | "icontains" => {
            let i_val = next(start_idx);
            let like = if op == "contains" { "LIKE" } else { "ILIKE" };
            conditions.push(format!("payload->>${i_key} {like} ${i_val}"));
            params.push(Param::Text(key.to_string()));
            params.push(Param::Text(format!("%{}%", scalar_to_string(op_value))));
        }
        _ => {}
    }
}

fn next(idx: &mut usize) -> usize {
    let v = *idx;
    *idx += 1;
    v
}

fn scalar_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn row_to_hit(id: String, score: f32, payload: Value) -> SearchHit {
    let payload = payload.as_object().cloned().unwrap_or_default();
    SearchHit { id, score, payload }
}

#[async_trait]
impl VectorStore for PgVectorStore {
    async fn insert(&self, records: Vec<VectorRecord>) -> Result<()> {
        let lock = self.client().await?;
        let client = lock.lock().await;
        let sql = format!(
            "INSERT INTO \"{c}\" (id, vector, payload) VALUES ($1, $2, $3) \
             ON CONFLICT (id) DO UPDATE SET vector = EXCLUDED.vector, payload = EXCLUDED.payload",
            c = self.collection
        );
        for r in records {
            let vector = Vector::from(r.vector.clone());
            let payload = Value::Object(r.payload.clone());
            client
                .execute(sql.as_str(), &[&r.id, &vector, &payload])
                .await
                .map_err(pg)?;
        }
        Ok(())
    }

    async fn search(
        &self,
        _query: &str,
        vector: &[f32],
        top_k: usize,
        filters: &JsonMap,
    ) -> Result<Vec<SearchHit>> {
        let lock = self.client().await?;
        let client = lock.lock().await;

        let qvec = Vector::from(vector.to_vec());
        let mut idx = 2usize; // $1 is the query vector
        let mut params: Vec<Param> = Vec::new();
        let conds = build_conditions(filters, &mut idx, &mut params);
        let where_clause = if conds.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conds.join(" AND "))
        };
        let limit_idx = idx;
        let sql = format!(
            "SELECT id, vector <=> $1::vector AS distance, payload FROM \"{c}\" {where_clause} \
             ORDER BY distance LIMIT ${limit_idx}",
            c = self.collection
        );

        let mut bind: Vec<&(dyn ToSql + Sync)> = Vec::new();
        bind.push(&qvec);
        for p in &params {
            bind.push(p.as_sql());
        }
        let limit = top_k as i64;
        bind.push(&limit);

        let rows = client.query(sql.as_str(), &bind).await.map_err(pg)?;
        Ok(rows
            .iter()
            .map(|row| {
                let id: String = row.get(0);
                let distance: f64 = row.get(1);
                let payload: Value = row.get(2);
                row_to_hit(id, (1.0 - distance) as f32, payload)
            })
            .collect())
    }

    async fn get(&self, id: &str) -> Result<Option<SearchHit>> {
        let lock = self.client().await?;
        let client = lock.lock().await;
        let sql = format!(
            "SELECT id, payload FROM \"{}\" WHERE id = $1",
            self.collection
        );
        let row = client.query_opt(sql.as_str(), &[&id]).await.map_err(pg)?;
        Ok(row.map(|r| {
            let id: String = r.get(0);
            let payload: Value = r.get(1);
            row_to_hit(id, 0.0, payload)
        }))
    }

    async fn update(
        &self,
        id: &str,
        vector: Option<Vec<f32>>,
        payload: Option<JsonMap>,
    ) -> Result<()> {
        let lock = self.client().await?;
        let client = lock.lock().await;
        if let Some(v) = vector {
            let qvec = Vector::from(v);
            let sql = format!(
                "UPDATE \"{}\" SET vector = $1 WHERE id = $2",
                self.collection
            );
            client
                .execute(sql.as_str(), &[&qvec, &id])
                .await
                .map_err(pg)?;
        }
        if let Some(p) = payload {
            let payload = Value::Object(p);
            let sql = format!(
                "UPDATE \"{}\" SET payload = $1 WHERE id = $2",
                self.collection
            );
            client
                .execute(sql.as_str(), &[&payload, &id])
                .await
                .map_err(pg)?;
        }
        Ok(())
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let lock = self.client().await?;
        let client = lock.lock().await;
        let sql = format!("DELETE FROM \"{}\" WHERE id = $1", self.collection);
        client.execute(sql.as_str(), &[&id]).await.map_err(pg)?;
        Ok(())
    }

    async fn list(&self, filters: &JsonMap, limit: Option<usize>) -> Result<Vec<SearchHit>> {
        let lock = self.client().await?;
        let client = lock.lock().await;
        let mut idx = 1usize;
        let mut params: Vec<Param> = Vec::new();
        let conds = build_conditions(filters, &mut idx, &mut params);
        let where_clause = if conds.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conds.join(" AND "))
        };
        let limit_idx = idx;
        let sql = format!(
            "SELECT id, payload FROM \"{c}\" {where_clause} LIMIT ${limit_idx}",
            c = self.collection
        );
        let mut bind: Vec<&(dyn ToSql + Sync)> = Vec::new();
        for p in &params {
            bind.push(p.as_sql());
        }
        let lim = limit.unwrap_or(100) as i64;
        bind.push(&lim);
        let rows = client.query(sql.as_str(), &bind).await.map_err(pg)?;
        Ok(rows
            .iter()
            .map(|row| {
                let id: String = row.get(0);
                let payload: Value = row.get(1);
                row_to_hit(id, 0.0, payload)
            })
            .collect())
    }

    async fn delete_col(&self) -> Result<()> {
        let lock = self.client().await?;
        let client = lock.lock().await;
        let sql = format!("DROP TABLE IF EXISTS \"{}\"", self.collection);
        client.batch_execute(&sql).await.map_err(pg)?;
        Ok(())
    }

    async fn reset(&self) -> Result<()> {
        self.delete_col().await?;
        let lock = self.client().await?;
        let client = lock.lock().await;
        self.create_col(&client).await
    }

    async fn keyword_search(
        &self,
        query: &str,
        top_k: usize,
        filters: &JsonMap,
    ) -> Result<Option<Vec<SearchHit>>> {
        let lock = self.client().await?;
        let client = lock.lock().await;
        // $1, $2 = query (rank + match); filters start at $3.
        let mut idx = 3usize;
        let mut params: Vec<Param> = Vec::new();
        let conds = build_conditions(filters, &mut idx, &mut params);
        let extra = if conds.is_empty() {
            String::new()
        } else {
            format!("AND {}", conds.join(" AND "))
        };
        let limit_idx = idx;
        let sql = format!(
            "SELECT id, ts_rank_cd(to_tsvector('simple', payload->>'text_lemmatized'), \
             plainto_tsquery('simple', $1)) AS score, payload FROM \"{c}\" \
             WHERE to_tsvector('simple', payload->>'text_lemmatized') @@ plainto_tsquery('simple', $2) \
             {extra} ORDER BY score DESC LIMIT ${limit_idx}",
            c = self.collection
        );
        let q = query.to_string();
        let mut bind: Vec<&(dyn ToSql + Sync)> = Vec::new();
        bind.push(&q);
        bind.push(&q);
        for p in &params {
            bind.push(p.as_sql());
        }
        let limit = top_k as i64;
        bind.push(&limit);
        match client.query(sql.as_str(), &bind).await {
            Ok(rows) => Ok(Some(
                rows.iter()
                    .map(|row| {
                        let id: String = row.get(0);
                        let score: f32 = row.get(1);
                        let payload: Value = row.get(2);
                        row_to_hit(id, score, payload)
                    })
                    .collect(),
            )),
            Err(e) => {
                tracing::debug!("pg keyword search failed: {e}");
                Ok(None)
            }
        }
    }
}
