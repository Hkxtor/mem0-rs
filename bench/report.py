#!/usr/bin/env python3
"""Render a comparison table from the Rust and Python bench JSON blobs.

Usage: report.py '<rust_json>' '<python_json>'
"""
import json
import sys


def per_op_us(total_ms, count):
    return (total_ms * 1000.0) / count if count else 0.0


def main():
    rust = json.loads(sys.argv[1])
    py = json.loads(sys.argv[2])

    n_add = rust["n_add"]
    n_search = rust["n_search"]

    rows = [
        ("add (us/op)", per_op_us(rust["add_ms"], n_add), per_op_us(py["add_ms"], n_add)),
        ("search (us/op)", per_op_us(rust["search_ms"], n_search), per_op_us(py["search_ms"], n_search)),
        ("get_all (ms)", rust["get_all_ms"], py["get_all_ms"]),
        ("peak RSS (MB)", rust["peak_rss_kb"] / 1024.0, py["peak_rss_kb"] / 1024.0),
    ]

    print(f"\nEqual-workload comparison (N_add={n_add}, M_search={n_search}, infer=false, mock embedder)\n")
    print(f"{'metric':<18}{'rust':>14}{'python':>14}{'python/rust':>14}")
    print("-" * 60)
    for name, r, p in rows:
        ratio = (p / r) if r else float("inf")
        print(f"{name:<18}{r:>14.2f}{p:>14.2f}{ratio:>13.2f}x")
    print()


if __name__ == "__main__":
    main()