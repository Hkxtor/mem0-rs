//! Dependency-free NLP layer.
//!
//! mem0's Python uses spaCy for lemmatization and entity extraction, but both
//! paths degrade gracefully when spaCy is unavailable (lemmatize returns the raw
//! text; entity extraction returns `[]`, and entity linking is best-effort).
//! This module provides a deterministic, dependency-free implementation that
//! preserves the *contracts*: a stable lemmatizer used symmetrically at
//! add-time and search-time (so BM25 matching is consistent), and PROPER/QUOTED
//! entity extraction (the POS/dependency-driven COMPOUND/NOUN cases are omitted).

use regex::Regex;
use std::sync::OnceLock;

/// A compact English stopword set (subset of spaCy's list).
const STOPWORDS: &[&str] = &[
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "cannot",
    "could",
    "couldn't",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "shan't",
    "she",
    "should",
    "shouldn't",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasn't",
    "we",
    "were",
    "weren't",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "won't",
    "would",
    "wouldn't",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "s",
    "t",
    "can't",
    "i'm",
    "i've",
    "it's",
];

fn is_stopword(word: &str) -> bool {
    STOPWORDS.contains(&word)
}

/// Lightweight, deterministic lemmatizer used only for BM25 normalization.
fn simple_lemma(s: &str) -> String {
    if !s.is_ascii() {
        return s.to_string();
    }
    if s.len() > 4 && s.ends_with("ies") {
        return format!("{}y", &s[..s.len() - 3]);
    }
    if s.len() > 3 && s.ends_with("es") {
        return s[..s.len() - 2].to_string();
    }
    if s.len() > 3 && s.ends_with('s') && !s.ends_with("ss") {
        return s[..s.len() - 1].to_string();
    }
    if s.len() > 4 && s.ends_with("ing") {
        return s[..s.len() - 3].to_string();
    }
    if s.len() > 3 && s.ends_with("ed") {
        return s[..s.len() - 2].to_string();
    }
    s.to_string()
}

/// Lemmatize text for BM25 matching. Port of `lemmatize_for_bm25` (behavioral).
///
/// Lowercases, drops stopwords and punctuation, applies light suffix
/// normalization, and—matching the Python—also keeps the original `-ing` form
/// alongside its lemma. Always returns a value.
pub fn lemmatize_for_bm25(text: &str) -> String {
    let lower = text.to_lowercase();
    let mut tokens: Vec<String> = Vec::new();
    for tok in lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
    {
        if is_stopword(tok) {
            continue;
        }
        let lemma = simple_lemma(tok);
        if lemma.chars().all(|c| c.is_alphanumeric()) && !lemma.is_empty() {
            tokens.push(lemma.clone());
        }
        if tok.ends_with("ing") && tok != lemma {
            tokens.push(tok.to_string());
        }
    }
    tokens.join(" ")
}

fn proper_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*").unwrap())
}

fn quoted_double_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r#""([^"]+)""#).unwrap())
}

fn quoted_single_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"'([^']+)'").unwrap())
}

/// Extract `(entity_type, entity_text)` pairs from `text`.
///
/// Implements the dependency-free subset of `extract_entities`: PROPER
/// (capitalized word sequences) and QUOTED (single/double quoted spans). Results
/// are deduplicated case-insensitively, preferring the first occurrence and
/// PROPER over QUOTED.
pub fn extract_entities(text: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    let push = |etype: &str,
                etext: &str,
                out: &mut Vec<(String, String)>,
                seen: &mut std::collections::HashMap<String, usize>| {
        let t = etext.trim();
        if t.len() <= 2 {
            return;
        }
        let key = t.to_lowercase();
        let priority = if etype == "PROPER" { 0 } else { 1 };
        if let Some(&idx) = seen.get(&key) {
            // Upgrade type if higher priority (PROPER beats QUOTED).
            let existing_priority = if out[idx].0 == "PROPER" { 0 } else { 1 };
            if priority < existing_priority {
                out[idx] = (etype.to_string(), t.to_string());
            }
        } else {
            seen.insert(key, out.len());
            out.push((etype.to_string(), t.to_string()));
        }
    };

    for m in proper_re().find_iter(text) {
        push("PROPER", m.as_str(), &mut out, &mut seen);
    }
    for caps in quoted_double_re().captures_iter(text) {
        push("QUOTED", &caps[1], &mut out, &mut seen);
    }
    for caps in quoted_single_re().captures_iter(text) {
        push("QUOTED", &caps[1], &mut out, &mut seen);
    }

    out
}

/// Batch entity extraction. Port of `extract_entities_batch`.
pub fn extract_entities_batch(texts: &[String]) -> Vec<Vec<(String, String)>> {
    texts.iter().map(|t| extract_entities(t)).collect()
}
