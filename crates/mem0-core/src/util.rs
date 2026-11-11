//! Small shared utilities.

use md5::{Digest, Md5};

/// Return the lowercase hex md5 digest of `s` (matches Python `hashlib.md5(...).hexdigest()`).
pub fn md5_hex(s: &str) -> String {
    let mut hasher = Md5::new();
    hasher.update(s.as_bytes());
    let digest = hasher.finalize();
    let mut out = String::with_capacity(32);
    for byte in digest {
        out.push_str(&format!("{:02x}", byte));
    }
    out
}

/// Current UTC time as an RFC3339 string (matches Python `datetime.now(timezone.utc).isoformat()`).
pub fn now_utc_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339()
}

/// Normalize an `expiration_date` input to `YYYY-MM-DD`, mirroring OSS
/// `_normalize_expiration_date`: strings must parse as an ISO date; anything else
/// is a validation error with the same message the Python SDK raises.
pub fn normalize_expiration_date(
    value: &serde_json::Value,
) -> Result<Option<String>, crate::error::Mem0Error> {
    match value {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::String(s) => match chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
            Ok(d) => Ok(Some(d.to_string())),
            Err(_) => Err(crate::error::Mem0Error::validation_code(
                "VALIDATION_003",
                "expiration_date must be a valid date in YYYY-MM-DD format.".to_string(),
                Some("Pass a date string like \"2026-12-31\".".to_string()),
            )),
        },
        _ => Err(crate::error::Mem0Error::validation_code(
            "VALIDATION_003",
            "expiration_date must be a date string in YYYY-MM-DD format.".to_string(),
            None,
        )),
    }
}

/// Whether a stored payload is expired, mirroring OSS `_payload_is_expired`: expired means
/// the expiration date is strictly before today (UTC) — expiring today still counts as
/// valid. An unparseable stored value is treated as not expired, exactly like Python.
pub fn payload_is_expired(payload: &crate::types::JsonMap) -> bool {
    let Some(raw) = payload.get("expiration_date").and_then(|v| v.as_str()) else {
        return false;
    };
    match chrono::NaiveDate::parse_from_str(raw, "%Y-%m-%d") {
        Ok(d) => d < chrono::Utc::now().date_naive(),
        Err(_) => false,
    }
}

/// Qualify a Gemini model name the way the `google-genai` SDK does.
///
/// The REST path is `…/v1beta/{model}:{method}`, and a model is identified as
/// `<publisher>/<id>` — so `gemini-embedding-001` becomes `models/gemini-embedding-001`.
/// The Python SDK adds that publisher for us; a hand-built URL that skips it gets an
/// HTTP 404 with an empty body (verified against the live API), which reads like an
/// outage rather than a malformed path. Names that already carry a publisher
/// (`models/…`, `tunedModels/…`, `publishers/google/models/…`) are left alone.
pub fn qualify_gemini_model(model: &str) -> String {
    let trimmed = model.trim();
    if trimmed.is_empty() || trimmed.contains('/') {
        return trimmed.to_string();
    }
    format!("models/{trimmed}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expiration_date_normalizes_to_iso_date() {
        assert_eq!(
            normalize_expiration_date(&serde_json::Value::Null).unwrap(),
            None
        );
        assert_eq!(
            normalize_expiration_date(&serde_json::json!("2026-12-31")).unwrap(),
            Some("2026-12-31".to_string())
        );
        // Full datetimes are not accepted as strings — same as OSS (str must be a bare date).
        assert!(normalize_expiration_date(&serde_json::json!("2026-12-31T10:00:00")).is_err());
        assert!(normalize_expiration_date(&serde_json::json!("not-a-date")).is_err());
        assert!(normalize_expiration_date(&serde_json::json!("2026-13-40")).is_err());
        assert!(normalize_expiration_date(&serde_json::json!(20261231)).is_err());
    }

    #[test]
    fn expiration_boundary_is_strictly_before_today_utc() {
        let today = chrono::Utc::now().date_naive();
        let yesterday = today - chrono::Duration::days(1);
        let tomorrow = today + chrono::Duration::days(1);
        let mut payload = crate::types::JsonMap::new();

        payload.insert(
            "expiration_date".into(),
            serde_json::json!(yesterday.to_string()),
        );
        assert!(payload_is_expired(&payload), "yesterday must be expired");

        payload.insert(
            "expiration_date".into(),
            serde_json::json!(today.to_string()),
        );
        assert!(
            !payload_is_expired(&payload),
            "expiring today is still valid"
        );

        payload.insert(
            "expiration_date".into(),
            serde_json::json!(tomorrow.to_string()),
        );
        assert!(!payload_is_expired(&payload), "tomorrow is valid");

        payload.insert("expiration_date".into(), serde_json::json!("garbage"));
        assert!(
            !payload_is_expired(&payload),
            "unparseable stored value is not expired (OSS parity)"
        );

        payload.remove("expiration_date");
        assert!(!payload_is_expired(&payload));
    }

    #[test]
    fn gemini_model_names_gain_a_publisher_only_when_they_lack_one() {
        assert_eq!(
            qualify_gemini_model("gemini-embedding-001"),
            "models/gemini-embedding-001"
        );
        assert_eq!(
            qualify_gemini_model("models/gemini-embedding-001"),
            "models/gemini-embedding-001"
        );
        assert_eq!(
            qualify_gemini_model("publishers/google/models/gemini-embedding-001"),
            "publishers/google/models/gemini-embedding-001"
        );
        assert_eq!(qualify_gemini_model("tunedModels/mine"), "tunedModels/mine");
        // A trailing newline from a paste must not end up in the URL.
        assert_eq!(
            qualify_gemini_model(" gemini-2.0-flash\n"),
            "models/gemini-2.0-flash"
        );
    }
}
