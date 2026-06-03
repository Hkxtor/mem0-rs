#!/usr/bin/env python3
"""Verify the Rust prompt constants are byte-identical to Python mem0's.

The additive-extraction prompt is the single biggest determinant of what gets
stored (and thus of memory quality), so byte-equality guarantees that, given the
same LLM, both implementations extract the same memories.

Usage: verify_prompts.py [path/to/constants.rs]
"""
import re
import sys

from mem0.configs import prompts as P

CONSTS = sys.argv[1] if len(sys.argv) > 1 else "crates/mem0-core/src/prompts/constants.rs"

CHECKS = [
    ("ADDITIVE_EXTRACTION_PROMPT", P.ADDITIVE_EXTRACTION_PROMPT),
    ("DEFAULT_UPDATE_MEMORY_PROMPT", P.DEFAULT_UPDATE_MEMORY_PROMPT),
    ("AGENT_CONTEXT_SUFFIX", P.AGENT_CONTEXT_SUFFIX),
    ("PROCEDURAL_MEMORY_SYSTEM_PROMPT", P.PROCEDURAL_MEMORY_SYSTEM_PROMPT),
    ("MEMORY_ANSWER_PROMPT", P.MEMORY_ANSWER_PROMPT),
]


def extract(rust_src, name):
    m = re.search(r'pub const ' + name + r': &str = r#"(.*?)"#;', rust_src, re.DOTALL)
    return m.group(1) if m else None


def main():
    with open(CONSTS) as f:
        rust_src = f.read()

    print("\nPrompt fidelity (Rust constants vs Python mem0 source)\n")
    all_ok = True
    for name, py_text in CHECKS:
        rs_text = extract(rust_src, name)
        ok = rs_text is not None and rs_text == py_text
        all_ok &= ok
        mark = "IDENTICAL" if ok else "DIFFERS"
        rs_len = len(rs_text) if rs_text else 0
        print(f"  [{mark}] {name}  (python {len(py_text)} chars, rust {rs_len} chars)")

    print(f"\nPROMPT_FIDELITY: {'PASS' if all_ok else 'FAIL'}\n")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()