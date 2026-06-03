# Quality parity with mem0

mem0-rs delivers its speed and footprint advantages **without changing memory
quality**. What gets stored, how it is deduplicated, retrieved, ranked, filtered,
and how update/history/delete behave are **identical** to Python mem0 v2.0.4.
The full write-up is in
[`mem0-rust-port/QUALITY.md`](../../mem0-rust-port/QUALITY.md).

Quality is determined by the **algorithms and prompts**, not the language — so
given the same model and inputs, the same algorithm yields the same memories.
This is validated three ways.

## 1. Prompt fidelity — byte-identical

The extraction prompt is the single biggest determinant of what gets stored. All
five prompt constants are byte-for-byte identical to the Python source
(verified by `bench/verify_prompts.py`):

```
[IDENTICAL] ADDITIVE_EXTRACTION_PROMPT      (33,653 chars)
[IDENTICAL] DEFAULT_UPDATE_MEMORY_PROMPT    (5,310)
[IDENTICAL] AGENT_CONTEXT_SUFFIX            (563)
[IDENTICAL] PROCEDURAL_MEMORY_SYSTEM_PROMPT (5,100)
[IDENTICAL] MEMORY_ANSWER_PROMPT            (547)
PROMPT_FIDELITY: PASS
```

Consequence: given the same LLM, both implementations extract the same facts.

## 2. Deterministic cross-implementation equivalence — 100%

`bench/run_parity.sh` drives **both** stacks through the same scenario script
with the model and embedding variables removed (a bit-identical FNV-1a embedder
and the same scripted LLM outputs on both sides), so any difference would be a
true port-fidelity bug. Result: **17/17 checks pass (100%)** across:

| Dimension | Validation | Result |
|-----------|-----------|--------|
| Storage | raw + inferred adds; `get_all` sets | identical |
| Deduplication | re-mention of an existing fact (md5) | identical (0 new on both) |
| Retrieval / search | ranked texts + scores per query | identical order, scores within 0.01 |
| Filtering | operator filter (`category eq food`) | identical result set |
| Update + history | update then history events | identical (`[ADD, UPDATE]`) |
| Delete | delete + post-delete `get_all` | identical |

## 3. Algorithm-fidelity tests — exact

The quality-determining math is covered by exact-match ports of mem0's own tests:

- `port_scoring.rs` — BM25 params, sigmoid normalization, `score_and_rank`.
- `port_text_parsing.rs` — `remove_code_blocks` / `extract_json` / fallback chain.
- `port_storage.rs` — history schema, event sequencing, ordering.
- `m1_pipeline.rs::infer_add_dedups_by_hash` — md5 dedup.
- `m1_pipeline.rs::search_honors_advanced_metadata_filters` — operator filters.
- `port_prompts.rs` — prompt builders.

## Entity linking — honest characterization

mem0's entity linking/boosting is **best-effort and spaCy-gated**: when spaCy is
absent it does nothing. mem0-rs implements the dependency-free **PROPER + QUOTED**
subset always-on, and linking remains best-effort and non-fatal. Relative to
mem0:

- **vs mem0 without spaCy:** mem0-rs provides *more* linking — a strict superset.
- **vs mem0 with spaCy:** mem0 additionally derives `COMPOUND`/`NOUN` entities
  from a dependency parse, which the dependency-free extractor does not. This is
  the one intentional, documented divergence; it affects only the secondary
  entity-boost signal, never the primary semantic+BM25 ranking or what is stored.

See [comparison.md](comparison.md) for the full feature matrix.

## Reproduce

```bash
pip install -e mem0
mem0-rs/bench/run_parity.sh          # prompt fidelity + 17/17 equivalence
```

## Conclusion

For storage, dedup, retrieval, ranking, filtering, and update/history/delete,
mem0-rs is **behaviorally identical** to Python mem0 — proven by byte-identical
prompts, 100% deterministic equivalence, and exact algorithm tests — while
delivering its ~3–6.5× speed/memory advantage.

See also: [Performance](performance.md).