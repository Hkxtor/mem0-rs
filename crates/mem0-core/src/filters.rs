//! Filter / metadata construction and evaluation ported from
//! `mem0/mem0/memory/main.py` (`_build_filters_and_metadata`,
//! `_build_session_scope`, `_has_advanced_operators`,
//! `_process_metadata_filters`, validators).

use crate::error::{Mem0Error, Result};
use crate::types::JsonMap;
use serde_json::Value;

/// Validate and trim an optional entity id. Port of `_validate_and_trim_entity_id`.
pub fn validate_and_trim_entity_id(value: Option<&str>, name: &str) -> Result<Option<String>> {
    match value {
        None => Ok(None),
        Some(v) => {
            let trimmed = v.trim();
            if trimmed.is_empty() {
                return Err(Mem0Error::validation_code(
                    "VALIDATION_003",
                    format!("'{name}' must be a non-empty string when provided."),
                    Some(format!(
                        "Provide a non-empty value for '{name}' or omit it."
                    )),
                ));
            }
            Ok(Some(trimmed.to_string()))
        }
    }
}

/// Validate search parameters. Port of `_validate_search_params`.
pub fn validate_search_params(threshold: Option<f64>, top_k: Option<i64>) -> Result<()> {
    if let Some(t) = top_k {
        if t <= 0 {
            return Err(Mem0Error::validation_code(
                "VALIDATION_004",
                "top_k must be a positive integer.",
                Some("Pass a top_k greater than 0.".into()),
            ));
        }
    }
    if let Some(th) = threshold {
        if !(0.0..=1.0).contains(&th) {
            return Err(Mem0Error::validation_code(
                "VALIDATION_005",
                "threshold must be between 0.0 and 1.0.",
                Some("Pass a threshold in the [0.0, 1.0] range.".into()),
            ));
        }
    }
    Ok(())
}

/// Build `(base_metadata_template, effective_query_filters)`.
/// Port of `_build_filters_and_metadata`.
pub fn build_filters_and_metadata(
    user_id: Option<&str>,
    agent_id: Option<&str>,
    run_id: Option<&str>,
    actor_id: Option<&str>,
    input_metadata: Option<&JsonMap>,
    input_filters: Option<&JsonMap>,
) -> Result<(JsonMap, JsonMap)> {
    let mut base_metadata: JsonMap = input_metadata.cloned().unwrap_or_default();
    let mut filters: JsonMap = input_filters.cloned().unwrap_or_default();

    let user_id = validate_and_trim_entity_id(user_id, "user_id")?;
    let agent_id = validate_and_trim_entity_id(agent_id, "agent_id")?;
    let run_id = validate_and_trim_entity_id(run_id, "run_id")?;

    let mut provided = false;
    if let Some(v) = user_id {
        base_metadata.insert("user_id".into(), Value::String(v.clone()));
        filters.insert("user_id".into(), Value::String(v));
        provided = true;
    }
    if let Some(v) = agent_id {
        base_metadata.insert("agent_id".into(), Value::String(v.clone()));
        filters.insert("agent_id".into(), Value::String(v));
        provided = true;
    }
    if let Some(v) = run_id {
        base_metadata.insert("run_id".into(), Value::String(v.clone()));
        filters.insert("run_id".into(), Value::String(v));
        provided = true;
    }

    if !provided {
        return Err(Mem0Error::validation_code(
            "VALIDATION_001",
            "At least one of 'user_id', 'agent_id', or 'run_id' must be provided.",
            Some("Please provide at least one identifier to scope the memory operation.".into()),
        ));
    }

    let resolved_actor = actor_id.map(|s| s.to_string()).or_else(|| {
        filters
            .get("actor_id")
            .and_then(|v| v.as_str().map(|s| s.to_string()))
    });
    if let Some(a) = resolved_actor {
        filters.insert("actor_id".into(), Value::String(a));
    }

    Ok((base_metadata, filters))
}

/// Build a deterministic session-scope key from entity ids.
/// Port of `_build_session_scope`.
pub fn build_session_scope(filters: &JsonMap) -> String {
    let mut parts: Vec<String> = Vec::new();
    // Python sorts the key list: ["agent_id", "run_id", "user_id"].
    for key in ["agent_id", "run_id", "user_id"] {
        if let Some(val) = filters.get(key).and_then(|v| v.as_str()) {
            if !val.is_empty() {
                parts.push(format!("{key}={val}"));
            }
        }
    }
    parts.join("&")
}

/// Extract only the session-scoping ids (`user_id`/`agent_id`/`run_id`) from filters.
pub fn session_filters(filters: &JsonMap) -> JsonMap {
    let mut out = JsonMap::new();
    for key in ["user_id", "agent_id", "run_id"] {
        if let Some(v) = filters.get(key) {
            if v.as_str().map(|s| !s.is_empty()).unwrap_or(false) {
                out.insert(key.to_string(), v.clone());
            }
        }
    }
    out
}

const OPERATORS: &[&str] = &[
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "in",
    "nin",
    "contains",
    "icontains",
];

/// Whether filters contain advanced operators. Port of `_has_advanced_operators`.
pub fn has_advanced_operators(filters: &JsonMap) -> bool {
    for (key, value) in filters {
        if matches!(key.as_str(), "AND" | "OR" | "NOT") {
            return true;
        }
        if let Some(obj) = value.as_object() {
            for op in obj.keys() {
                if OPERATORS.contains(&op.as_str()) {
                    return true;
                }
            }
        }
        if value.as_str() == Some("*") {
            return true;
        }
    }
    false
}

fn process_condition(key: &str, condition: &Value) -> Result<JsonMap> {
    let mut result = JsonMap::new();
    match condition {
        Value::Object(map) => {
            let mut opmap = JsonMap::new();
            for (op, val) in map {
                if OPERATORS.contains(&op.as_str()) {
                    opmap.insert(op.clone(), val.clone());
                } else {
                    return Err(Mem0Error::validation(format!(
                        "Unsupported metadata filter operator: {op}"
                    )));
                }
            }
            result.insert(key.to_string(), Value::Object(opmap));
        }
        Value::String(s) if s == "*" => {
            result.insert(key.to_string(), Value::String("*".into()));
        }
        other => {
            result.insert(key.to_string(), other.clone());
        }
    }
    Ok(result)
}

fn merge_filters(target: &mut JsonMap, source: &JsonMap) {
    for (k, v) in source {
        let deep = matches!(
            (target.get(k), v),
            (Some(Value::Object(_)), Value::Object(_))
        );
        if deep {
            if let (Some(Value::Object(tv)), Value::Object(sv)) = (target.get_mut(k), v) {
                for (kk, vv) in sv {
                    tv.insert(kk.clone(), vv.clone());
                }
            }
        } else {
            target.insert(k.clone(), v.clone());
        }
    }
}

/// Normalize advanced filters into vector-store-compatible form.
/// Port of `_process_metadata_filters`.
pub fn process_metadata_filters(metadata_filters: &JsonMap) -> Result<JsonMap> {
    let mut processed = JsonMap::new();
    for (key, value) in metadata_filters {
        match key.as_str() {
            "AND" => {
                let arr = value.as_array().ok_or_else(|| {
                    Mem0Error::validation("AND operator requires a list of conditions")
                })?;
                for condition in arr {
                    if let Some(obj) = condition.as_object() {
                        for (sk, sv) in obj {
                            let pc = process_condition(sk, sv)?;
                            merge_filters(&mut processed, &pc);
                        }
                    }
                }
            }
            "OR" => {
                let arr = value.as_array().filter(|a| !a.is_empty()).ok_or_else(|| {
                    Mem0Error::validation("OR operator requires a non-empty list of conditions")
                })?;
                let mut or_list = Vec::new();
                for condition in arr {
                    let mut or_cond = JsonMap::new();
                    if let Some(obj) = condition.as_object() {
                        for (sk, sv) in obj {
                            let pc = process_condition(sk, sv)?;
                            merge_filters(&mut or_cond, &pc);
                        }
                    }
                    or_list.push(Value::Object(or_cond));
                }
                processed.insert("$or".into(), Value::Array(or_list));
            }
            "NOT" => {
                let arr = value.as_array().filter(|a| !a.is_empty()).ok_or_else(|| {
                    Mem0Error::validation("NOT operator requires a non-empty list of conditions")
                })?;
                let mut not_list = Vec::new();
                for condition in arr {
                    let mut not_cond = JsonMap::new();
                    if let Some(obj) = condition.as_object() {
                        for (sk, sv) in obj {
                            let pc = process_condition(sk, sv)?;
                            merge_filters(&mut not_cond, &pc);
                        }
                    }
                    not_list.push(Value::Object(not_cond));
                }
                processed.insert("$not".into(), Value::Array(not_list));
            }
            _ => {
                let pc = process_condition(key, value)?;
                merge_filters(&mut processed, &pc);
            }
        }
    }
    Ok(processed)
}

/// Evaluate whether a payload satisfies (possibly operator-laden) filters.
///
/// Supports plain equality, the normalized operator dicts produced by
/// [`process_metadata_filters`] (`{key: {op: val}}`), wildcard (`"*"`), and the
/// `$or` / `$not` logical groups. Null filter values are ignored.
pub fn matches_filters(payload: &JsonMap, filters: &JsonMap) -> bool {
    for (key, cond) in filters {
        match key.as_str() {
            "$or" => {
                let ok = cond.as_array().is_some_and(|arr| {
                    arr.iter()
                        .any(|f| f.as_object().is_some_and(|o| matches_filters(payload, o)))
                });
                if !ok {
                    return false;
                }
            }
            "$not" => {
                let any = cond.as_array().is_some_and(|arr| {
                    arr.iter()
                        .any(|f| f.as_object().is_some_and(|o| matches_filters(payload, o)))
                });
                if any {
                    return false;
                }
            }
            _ => {
                if cond.is_null() {
                    continue;
                }
                if !matches_condition(payload.get(key), cond) {
                    return false;
                }
            }
        }
    }
    true
}

fn matches_condition(pv: Option<&Value>, cond: &Value) -> bool {
    match cond {
        Value::Object(ops) => ops.iter().all(|(op, val)| eval_op(pv, op, val)),
        Value::String(s) if s == "*" => pv.is_some(),
        _ => pv == Some(cond),
    }
}

fn eval_op(pv: Option<&Value>, op: &str, val: &Value) -> bool {
    match op {
        "eq" => pv == Some(val),
        "ne" => pv != Some(val),
        "in" => val
            .as_array()
            .is_some_and(|arr| pv.is_some_and(|p| arr.contains(p))),
        "nin" => val
            .as_array()
            .is_none_or(|arr| pv.is_none_or(|p| !arr.contains(p))),
        "gt" | "gte" | "lt" | "lte" => match (pv.and_then(|v| v.as_f64()), val.as_f64()) {
            (Some(a), Some(b)) => match op {
                "gt" => a > b,
                "gte" => a >= b,
                "lt" => a < b,
                _ => a <= b,
            },
            _ => false,
        },
        "contains" => match (pv.and_then(|v| v.as_str()), val.as_str()) {
            (Some(a), Some(b)) => a.contains(b),
            _ => false,
        },
        "icontains" => match (pv.and_then(|v| v.as_str()), val.as_str()) {
            (Some(a), Some(b)) => a.to_lowercase().contains(&b.to_lowercase()),
            _ => false,
        },
        _ => false,
    }
}
