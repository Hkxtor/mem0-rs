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
