//! Shared reranker utilities.

use regex::Regex;
use std::sync::OnceLock;

fn score_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b([01](?:\.\d+)?)\b").unwrap())
}

/// Extract a `[0.0, 1.0]` relevance score from LLM text. Port of `_extract_score`.
/// Returns 0.5 when no valid score is found.
pub fn extract_score(response_text: &str) -> f32 {
    if let Some(caps) = score_re().captures(response_text) {
        if let Some(m) = caps.get(1) {
            if let Ok(score) = m.as_str().parse::<f32>() {
                return score.clamp(0.0, 1.0);
            }
        }
    }
    0.5
}
