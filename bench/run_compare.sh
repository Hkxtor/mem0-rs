#!/usr/bin/env bash
# Equal-workload Python-vs-Rust comparison. Requires the Python `mem0` package
# importable (e.g. `pip install -e mem0` from the repo root).
#
# Usage: bench/run_compare.sh [N_add=2000] [M_search=500]
set -euo pipefail

N="${1:-2000}"
M="${2:-500}"

# cd to the mem0-rs workspace root (this script lives in mem0-rs/bench).
cd "$(dirname "$0")/.."

echo "Building Rust bench driver (release)..."
cargo build --release -p mem0 --example bench_run >/dev/null

echo "Running Rust..."
RUST_JSON="$(./target/release/examples/bench_run "$N" "$M")"
echo "  $RUST_JSON"

echo "Running Python (mem0)..."
PY_JSON="$(python3 bench/compare.py "$N" "$M" 2>/dev/null)"
echo "  $PY_JSON"

python3 bench/report.py "$RUST_JSON" "$PY_JSON"