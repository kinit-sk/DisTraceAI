"""Chroma Context-1 backend (README §5, contribution 3).

Drives Chroma Context-1 as an agentic search subagent (multi-turn query
decomposition + pruning) over the narrative-cluster corpus, then re-scores the
gathered clusters by cosine similarity to the query — so the returned score is a
real [0,1] semantic similarity (not a synthetic 1/(rank+1)), comparable to the
assignment threshold. Context-1 supplies ranking *quality*; cosine supplies the
*score*.
"""
from __future__ import annotations

import numpy as np

from core.hierarchy.backends.base import RetrievalBackend
from core.hierarchy.harness import AgenticSearchHarness

_DISCOVERY_PROMPT = """\
You are a retrieval subagent finding existing narrative clusters whose central
claims match a query disinformation claim.

Available tools (return JSON, one object or an array for parallel calls):
  {"tool": "search_corpus", "query": "<text>"}
  {"tool": "grep_corpus",   "pattern": "<regex>"}
  {"tool": "read_document", "chunk_id": "<cluster_id>"}
  {"tool": "prune_chunks",  "chunk_ids": ["<cluster_id>", ...]}
  {"tool": "done",          "reasoning": "<why done>"}

Decompose the claim into 2-3 sub-topics and search for each. You MUST call
search_corpus at least twice before calling done. /no_think
"""


class _ClusterSearchTools:
    """Adapts a FactCheckCorpus to the harness tool protocol (cluster = document)."""

    def __init__(self, corpus, k: int = 5) -> None:
        self.corpus = corpus
        self.k = k

    def search(self, query: str, seen: set, k: int) -> list[tuple[str, str]]:
        out = []
        for cid, _score in self.corpus.search(query, k=k + len(seen)):
            if cid not in seen:
                out.append((cid, self.corpus.cluster_document(cid)))
            if len(out) >= k:
                break
        return out

    def grep(self, pattern: str, seen: set) -> list[tuple[str, str]]:
        return [(cid, doc) for cid, doc in self.corpus.grep(pattern) if cid not in seen]

    def get(self, cluster_id: str) -> str | None:
        doc = self.corpus.cluster_document(cluster_id)
        return doc or None


class Context1Backend(RetrievalBackend):
    name = "context1"

    def __init__(self, generate, embedder, *, max_turns: int = 8,
                 token_budget: int = 8192, top_k: int = 5,
                 min_searches: int = 2) -> None:
        self.generate = generate
        self.embedder = embedder
        self.max_turns = max_turns
        self.token_budget = token_budget
        self.top_k = top_k
        # Floor on agentic search turns: forces the harness to run a genuine
        # multi-turn loop rather than terminating after a single pass even if the
        # model emits an early `done`.
        self.min_searches = max(1, min_searches)

    def rank(self, query: str, corpus, k: int = 10) -> list[tuple[str, float]]:
        tools = _ClusterSearchTools(corpus, k=self.top_k)
        harness = AgenticSearchHarness(
            tools, self.generate, _DISCOVERY_PROMPT,
            token_budget=self.token_budget, top_k=self.top_k,
            max_turns=self.max_turns, min_searches=self.min_searches)
        gathered = harness.search(query)            # [(cluster_id, document)]
        if not gathered:
            return []
        ids = [cid for cid, _ in gathered]
        docs = [doc for _, doc in gathered]
        q = corpus.encode_query(query)
        emb = np.asarray(self.embedder.encode(docs), dtype=np.float32)
        emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        scores = emb @ q
        ranked = sorted(zip(ids, scores.tolist()), key=lambda x: x[1], reverse=True)
        return [(cid, float(s)) for cid, s in ranked[:k]]
