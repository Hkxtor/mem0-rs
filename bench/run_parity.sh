#!/usr/bin/env bash
# Memory-quality parity validation (Rust port vs Python mem0).
# Requires the Python `mem0` package importable (e.g. `pip install -e mem0`).
#
# Usage: bench/run_parity.sh
set -euo pipefail

cd "$(dirname "$0")/.."   # -> mem0-rs workspace root

echo "=== Prompt fidelity ==="
python3 bench/verify_prompts.py crates/mem0-core/src/prompts/constants.rs

echo "=== Building Rust parity runner (release) ==="
cargo build --release -p mem0 --example parity_run >/dev/null

echo "=== Running Rust ==="
RUST_JSON="$(./target/release/examples/parity_run bench/parity_scenarios.json)"

echo "=== Running Python (mem0) ==="
PY_JSON="$(python3 bench/parity.py bench/parity_scenarios.json)"

echo "=== Comparing ==="
python3 bench/parity_compare.py "$RUST_JSON" "$PY_JSON"