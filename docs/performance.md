# Performance

mem0-rs is substantially faster and lighter than Python mem0 while preserving
identical memory quality. This page summarizes the measured results and how to
reproduce them; the full write-up is in
[`mem0-rust-port/PERFORMANCE.md`](../../mem0-rust-port/PERFORMANCE.md).

## Headline results

Equal-workload comparison — the **same** N adds + M searches with the network
removed on both sides (mock embedder, `infer=false`, in-process store) — so it
measures **framework overhead**, the part that differs between the two
implementations.

**N = 2000 adds, M = 500 searches**

| metric | Rust | Python | Python ÷ Rust |
|--------|-----:|-------:|--------------:|
| add (µs/op) | 18.7 | 64.8 | **3.47×** |
| search (µs/op) | 6066 | 18353 | **3.03×** |
| get_all (ms) | 0.86 | 1.98 | 2.31× |
| peak RSS (MB) | **15.4** | 100.9 | **6.55×** |

**N = 500 adds, M = 100 searches**

| metric | Rust | Python | Python ÷ Rust |
|--------|-----:|-------:|--------------:|
| add (µs/op) | 15.9 | 78.9 | 4.95× |
| search (µs/op) | 1631 | 4547 | 2.79× |
| peak RSS (MB) | **9.5** | 96.9 | **10.2×** |

So **~3–3.5× faster per operation** and **~6.5× lower peak memory** (≈10× at
smaller working sets), in a single **~12 MB binary** versus a ~100 MB Python
process. Measured on a 2 vCPU / 8 GB Linux box (release build).

## Why "equal tests" means framework overhead

A memory system's real-world latency is dominated by the **network round-trip to
the LLM and embedding APIs**, which is identical for both implementations. To
compare the *implementations* rather than the network, the harness removes that
shared cost:

- a trivial fixed-vector embedder on both sides (no network);
- `infer=false` (no LLM call);
- an in-process vector store on both sides (Rust `embedded`; Python an injected
  stub with the same cosine + equality-filter semantics);
- the identical workload; wall-clock per op + peak RSS.

This isolates message parsing, scope/filter construction, md5 hashing, BM25
lemmatization, SQLite history writes, payload construction, and cosine/BM25
search.

> The comparison is **conservative to Python**: the Python run used the
> spaCy-absent fast path for lemmatization/entity extraction. With spaCy
> installed, Python would be slower, so the real gap is wider.

## Why Rust is faster

1. No interpreter, no GIL — compiled native orchestration.
2. Far lower allocation pressure — stack values, `&str` slices, typed structs vs
   Python boxing everything.
3. No pydantic validation or per-memory `deepcopy` on the hot path.
4. Static dispatch + monomorphized math (cosine, BM25, sigmoid).
5. `rusqlite` with prepared statements + batched transactions.
6. ~6.5× lower RSS — no CPython runtime and no heavyweight resident modules.

## Optimization applied

The `embedded` store's `search`/`list` clone payloads only for the retained
`top_k` results (O(N) → O(top_k) allocations per query), instead of cloning every
candidate before truncating.

## In production

With a real vector backend (Qdrant/pgvector), the vector scan is offloaded to the
database, so the **per-call framework overhead** (the add path, ~3.5×) and
**resident memory** (~6.5×) become the dominant differentiators — on top of the
identical, unavoidable network cost to the providers.

## Reproduce

```bash
pip install -e mem0                  # the Python package
mem0-rs/bench/run_compare.sh 2000 500
```

There are also criterion microbenchmarks for the CPU-bound core ops:

```bash
cargo bench -p mem0-core            # lemmatize, bm25, score_and_rank, entities, filters
cargo bench -p mem0                 # end-to-end add/search
```

See also: [Quality parity](quality.md) — speed without changing behavior.