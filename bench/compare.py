#!/usr/bin/env python3
"""Equal-workload driver for Python mem0 (framework-overhead comparison).

Runs the SAME workload as the Rust `bench_run` example: N raw adds (infer=False)
+ M searches + one get_all, using:
  - a fixed hash-based stub embedder (no network),
  - an in-process stub vector store (cosine + equality filters), and
  - in-memory SQLite history.

This isolates the *framework overhead* (message parsing, pydantic/deepcopy,
md5 hashing, lemmatization, SQLite history, payload construction, cosine search)
— the part that differs between Python mem0 and the Rust port. The LLM and
network embedder are removed from both sides so the comparison is apples-to-apples.

Usage: compare.py [N=2000] [M=500]
"""
import hashlib
import json
import math
import os
import resource
import sys
import time

os.environ.setdefault("MEM0_TELEMETRY", "False")

# --- Disable telemetry before importing the package ---
import mem0.memory.telemetry as _tele  # noqa: E402

_tele.capture_event = lambda *a, **k: None

from mem0.utils import factory  # noqa: E402


class StubEmbedder:
    def __init__(self, dims=32):
        self.dims = dims

    def embed(self, text, memory_action=None):
        v = [0.0] * self.dims
        for tok in text.lower().split():
            idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dims
            v[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm else v

    def embed_batch(self, texts, memory_action="add"):
        return [self.embed(t, memory_action) for t in texts]


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class _Hit:
    def __init__(self, id, score, payload):
        self.id = id
        self.score = score
        self.payload = payload


class StubStore:
    embedding_model_dims = 32

    def __init__(self, *args, **kwargs):
        self.data = {}

    def insert(self, vectors, payloads=None, ids=None):
        for i, vec in enumerate(vectors):
            self.data[ids[i]] = (vec, payloads[i])

    def _match(self, payload, filters):
        if not filters:
            return True
        return all(payload.get(k) == v for k, v in filters.items() if v is not None)

    def search(self, query, vectors, top_k=5, filters=None):
        hits = [
            _Hit(i, max(_cosine(vectors, vec), 0.0), pay)
            for i, (vec, pay) in self.data.items()
            if self._match(pay, filters)
        ]
        hits.sort(key=lambda h: -h.score)
        return hits[:top_k]

    def keyword_search(self, query, top_k=5, filters=None):
        return None

    def list(self, filters=None, top_k=100):
        out = [_Hit(i, 0.0, pay) for i, (vec, pay) in self.data.items() if self._match(pay, filters)]
        return [out[:top_k]]

    def get(self, vector_id):
        if vector_id in self.data:
            _vec, pay = self.data[vector_id]
            return _Hit(vector_id, 0.0, pay)
        return None

    def delete(self, vector_id):
        self.data.pop(vector_id, None)

    def update(self, vector_id, vector=None, payload=None):
        if vector_id in self.data:
            v, p = self.data[vector_id]
            self.data[vector_id] = (vector if vector is not None else v, payload if payload is not None else p)

    def reset(self):
        self.data = {}

    def delete_col(self):
        self.data = {}

    def col_info(self):
        return {}


# Patch the factories so Memory.__init__ never touches qdrant/openai.
factory.EmbedderFactory.create = staticmethod(lambda *a, **k: StubEmbedder())
factory.VectorStoreFactory.create = staticmethod(lambda *a, **k: StubStore())
factory.LlmFactory.create = staticmethod(lambda *a, **k: None)

from mem0.configs.base import MemoryConfig  # noqa: E402
from mem0.memory.main import Memory  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    config = MemoryConfig()
    config.history_db_path = ":memory:"
    mem = Memory(config)

    filters = {"user_id": "u1"}

    t0 = time.perf_counter()
    for i in range(n):
        mem.add(f"memory item {i} about hiking cooking travel", user_id="u1", infer=False)
    t1 = time.perf_counter()
    for _ in range(m):
        mem.search("hiking travel", filters=filters)
    t2 = time.perf_counter()
    mem.get_all(filters=filters, top_k=20)
    t3 = time.perf_counter()

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux
    print(json.dumps({
        "impl": "python",
        "n_add": n,
        "add_ms": (t1 - t0) * 1000.0,
        "n_search": m,
        "search_ms": (t2 - t1) * 1000.0,
        "get_all_ms": (t3 - t2) * 1000.0,
        "peak_rss_kb": rss,
    }))


if __name__ == "__main__":
    main()