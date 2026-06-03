#!/usr/bin/env python3
"""Python side of the cross-implementation quality-parity harness.

Drives `mem0.Memory` with:
  - a deterministic FNV-1a hash embedder that is *bit-identical* to the Rust
    `MockEmbedder` (same tokenization, same 64-bit FNV-1a, same L2 norm), so
    embeddings — and therefore cosine rankings — match the Rust port exactly;
  - an in-process stub vector store with cosine search, operator-aware metadata
    filtering, and BM25 disabled (keyword_search -> None) to match the Rust
    `SemanticOnly` store;
  - a scripted stub LLM that returns the same extraction JSON the Rust
    `MockLlm` is given.

Executes `bench/parity_scenarios.json` and prints a JSON array of per-op results
for `parity_compare.py`. Telemetry and spaCy are absent, so entity linking and
boosts are no-ops on both sides (the comparable primary path is isolated).

Usage: parity.py [scenarios.json]
"""
import json
import math
import os
import sys

os.environ.setdefault("MEM0_TELEMETRY", "False")

import mem0.memory.telemetry as _tele  # noqa: E402

_tele.capture_event = lambda *a, **k: None

from mem0.utils import factory  # noqa: E402

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = 0xFFFFFFFFFFFFFFFF


def _tokens(text):
    tok = []
    cur = []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            tok.append("".join(cur))
            cur = []
    if cur:
        tok.append("".join(cur))
    return tok


class FnvEmbedder:
    """Bit-identical to mem0_core::testing::MockEmbedder (FNV-1a, dims=32)."""

    def __init__(self, dims=32):
        self.dims = dims

    def embed(self, text, memory_action=None):
        v = [0.0] * self.dims
        for tok in _tokens(text):
            h = FNV_OFFSET
            for b in tok.encode("utf-8"):
                h = (h ^ b) & MASK64
                h = (h * FNV_PRIME) & MASK64
            v[h % self.dims] += 1.0
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


def _match_ops(pv, ops):
    for op, val in ops.items():
        if op == "eq":
            if pv != val:
                return False
        elif op == "ne":
            if pv == val:
                return False
        elif op == "in":
            if pv not in val:
                return False
        elif op == "nin":
            if pv in val:
                return False
        elif op in ("gt", "gte", "lt", "lte"):
            try:
                a, b = float(pv), float(val)
            except (TypeError, ValueError):
                return False
            if op == "gt" and not a > b:
                return False
            if op == "gte" and not a >= b:
                return False
            if op == "lt" and not a < b:
                return False
            if op == "lte" and not a <= b:
                return False
        elif op == "contains":
            if not (isinstance(pv, str) and val in pv):
                return False
        elif op == "icontains":
            if not (isinstance(pv, str) and val.lower() in pv.lower()):
                return False
        else:
            return False
    return True


def _matches(payload, filters):
    if not filters:
        return True
    for k, cond in filters.items():
        if k == "$or":
            if not any(_matches(payload, sub) for sub in cond):
                return False
        elif k == "$not":
            if any(_matches(payload, sub) for sub in cond):
                return False
        elif isinstance(cond, dict):
            if not _match_ops(payload.get(k), cond):
                return False
        elif cond == "*":
            if payload.get(k) is None:
                return False
        elif cond is None:
            continue
        else:
            if payload.get(k) != cond:
                return False
    return True


class StubStore:
    """Semantic-only in-process store (cosine + operator filters, no BM25)."""

    embedding_model_dims = 32

    def __init__(self, *args, **kwargs):
        self.data = {}

    def insert(self, vectors, payloads=None, ids=None, **kw):
        for i, vec in enumerate(vectors):
            self.data[ids[i]] = (vec, payloads[i])

    def search(self, query=None, vectors=None, top_k=5, filters=None, **kw):
        hits = [
            _Hit(i, max(_cosine(vectors, vec), 0.0), pay)
            for i, (vec, pay) in self.data.items()
            if _matches(pay, filters)
        ]
        hits.sort(key=lambda h: -h.score)
        return hits[:top_k]

    def keyword_search(self, query=None, top_k=5, filters=None, **kw):
        return None

    def list(self, filters=None, top_k=100, **kw):
        out = [_Hit(i, 0.0, pay) for i, (vec, pay) in self.data.items() if _matches(pay, filters)]
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


class StubLLM:
    """Returns scripted extraction JSON strings in call order."""

    def __init__(self, queue):
        self.queue = list(queue)

    def generate_response(self, messages, response_format=None, tools=None, tool_choice="auto", **kwargs):
        if self.queue:
            return self.queue.pop(0)
        return json.dumps({"memory": []})


def _memories(res):
    """Normalize add/get_all/search return shape to a list of dicts."""
    if isinstance(res, dict):
        return res.get("results", [])
    return res


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "bench/parity_scenarios.json"
    with open(path) as f:
        scenarios = json.load(f)

    queue = [json.dumps(op["llm"]) for op in scenarios if op["op"] == "add" and op.get("infer")]

    factory.EmbedderFactory.create = staticmethod(lambda *a, **k: FnvEmbedder())
    factory.VectorStoreFactory.create = staticmethod(lambda *a, **k: StubStore())
    factory.LlmFactory.create = staticmethod(lambda *a, **k: StubLLM(queue))

    from mem0.configs.base import MemoryConfig
    from mem0.memory.main import Memory

    config = MemoryConfig()
    config.history_db_path = ":memory:"
    mem = Memory(config)

    added_ids = []
    out = []

    for op in scenarios:
        kind = op["op"]
        if kind == "add":
            res = mem.add(
                op["text"],
                user_id=op["user"],
                infer=bool(op.get("infer", False)),
                metadata=op.get("meta"),
            )
            items = _memories(res)
            added_ids.append([m["id"] for m in items])
            texts = sorted(m["memory"] for m in items)
            out.append({"op": "add", "count": len(items), "texts": texts})
        elif kind == "search":
            filters = {"user_id": op["user"]}
            if op.get("filter"):
                filters.update(op["filter"])
            res = mem.search(op["query"], filters=filters, top_k=op.get("top_k", 5), threshold=0.1)
            items = _memories(res)
            texts = [m["memory"] for m in items]
            scores = [round(m.get("score", 0.0), 4) for m in items]
            out.append({"op": "search", "texts": texts, "scores": scores})
        elif kind == "get_all":
            res = mem.get_all(filters={"user_id": op["user"]}, top_k=100)
            items = _memories(res)
            texts = sorted(m["memory"] for m in items)
            out.append({"op": "get_all", "texts": texts})
        elif kind == "update":
            mem_id = added_ids[op["add_index"]][0]
            mem.update(mem_id, op["text"])
            out.append({"op": "update", "ok": True})
        elif kind == "history":
            mem_id = added_ids[op["add_index"]][0]
            hist = mem.history(mem_id)
            out.append({"op": "history", "events": [h["event"] for h in hist]})
        elif kind == "delete":
            mem_id = added_ids[op["add_index"]][0]
            mem.delete(mem_id)
            out.append({"op": "delete", "ok": True})
        else:
            out.append({"op": kind, "error": "unknown op"})

    print(json.dumps(out))


if __name__ == "__main__":
    main()