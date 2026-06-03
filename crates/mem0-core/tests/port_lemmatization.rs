//! Ported from mem0 `tests/utils/test_lemmatization.py`.
//!
//! Adapted: the Rust lemmatizer is dependency-free (no spaCy), so the Python
//! `_ensure_spacy` skip is dropped. The Python assertions are intentionally
//! loose (lemma presence, stopword/punctuation removal, lowercasing, `-ing`
//! preservation) and are all satisfied by the Rust implementation.

use mem0_core::nlp::lemmatize_for_bm25;

#[test]
fn basic_lemmatization() {
    let r = lemmatize_for_bm25("The cats are running quickly");
    assert!(r.contains("cat"));
    assert!(r.contains("run") || r.contains("running"));
    assert!(!r.split_whitespace().any(|t| t == "the"));
}

#[test]
fn verb_forms_normalized() {
    let r = lemmatize_for_bm25("she attended multiple meetings yesterday");
    assert!(r.contains("attend") || r.contains("attended"));
    assert!(r.contains("meeting"));
}

#[test]
fn ing_preservation() {
    let r = lemmatize_for_bm25("attending the morning meeting");
    let tokens: Vec<&str> = r.split_whitespace().collect();
    assert!(tokens.contains(&"attending") || tokens.contains(&"attend"));
}

#[test]
fn empty_string() {
    assert_eq!(lemmatize_for_bm25(""), "");
}

#[test]
fn punctuation_removed() {
    let r = lemmatize_for_bm25("Hello, world! How are you?");
    assert!(!r.contains(','));
    assert!(!r.contains('!'));
    assert!(!r.contains('?'));
}

#[test]
fn lowercased() {
    let r = lemmatize_for_bm25("PYTHON Programming LANGUAGE");
    for token in r.split_whitespace() {
        assert_eq!(token, token.to_lowercase());
    }
}

#[test]
fn stop_words_removed() {
    let r = lemmatize_for_bm25("this is a very simple test of the system");
    let tokens: Vec<&str> = r.split_whitespace().collect();
    for stop in ["this", "is", "a", "very", "of", "the"] {
        assert!(!tokens.contains(&stop));
    }
}
