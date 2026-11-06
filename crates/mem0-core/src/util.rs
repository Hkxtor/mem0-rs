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
