#!/usr/bin/env python3
"""Compare Rust vs Python parity outputs and report agreement.

Usage: parity_compare.py '<rust_json>' '<python_json>'
"""
import json
import sys

SCORE_TOL = 0.01


def check(rust, py):
    results = []
    if len(rust) != len(py):
        results.append((False, f"op-count mismatch: rust={len(rust)} python={len(py)}"))
        return results

    for i, (r, p) in enumerate(zip(rust, py)):
        op = r.get("op")
        label = f"#{i} {op}"
        if op != p.get("op"):
            results.append((False, f"{label}: op kind mismatch ({op} vs {p.get('op')})"))
            continue

        if op == "add":
            ok = r["count"] == p["count"] and r["texts"] == p["texts"]
            results.append((ok, f"{label}: count={r['count']}/{p['count']} texts_match={r['texts'] == p['texts']}"))
        elif op == "get_all":
            ok = sorted(r["texts"]) == sorted(p["texts"])
            results.append((ok, f"{label}: {len(r['texts'])} vs {len(p['texts'])} memories, set_match={ok}"))
        elif op == "search":
            order_ok = r["texts"] == p["texts"]
            scores_ok = len(r["scores"]) == len(p["scores"]) and all(
                abs(a - b) <= SCORE_TOL for a, b in zip(r["scores"], p["scores"])
            )
            ok = order_ok and scores_ok
            results.append((ok, f"{label}: order_match={order_ok} scores_within_tol={scores_ok} -> {r['texts']}"))
        elif op == "history":
            ok = r["events"] == p["events"]
            results.append((ok, f"{label}: events {r['events']} vs {p['events']}"))
        elif op in ("update", "delete"):
            ok = r.get("ok") and p.get("ok")
            results.append((ok, f"{label}: ok={ok}"))
        else:
            results.append((False, f"{label}: unknown op"))
    return results


def main():
    rust = json.loads(sys.argv[1])
    py = json.loads(sys.argv[2])
    results = check(rust, py)

    passed = sum(1 for ok, _ in results if ok)
    total = len(results)
    pct = (100.0 * passed / total) if total else 0.0

    print("\nMemory-quality parity (Rust port vs Python mem0) — deterministic equivalence\n")
    for ok, msg in results:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {msg}")
    print(f"\nAgreement: {passed}/{total} checks passed ({pct:.1f}%)\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()