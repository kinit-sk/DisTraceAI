"""BM25 + dense + RRF baseline backend (README §5).

Candidate clusters are selected by the corpus's hybrid BM25+dense RRF search;
the returned score is the cosine similarity of the query to each candidate
cluster's pooled embedding, so it lives on a [0,1] scale the assignment
threshold can be compared against directly. No LLM, no generation.
"""
from __future__ import annotations


from core.hierarchy.backends.base import RetrievalBackend


class BM25RagBackend(RetrievalBackend):
    name = "bm25_rag"

    def rank(self, query: str, corpus, k: int = 10) -> list[tuple[str, float]]:
        hybrid = corpus.search(query, k=max(k, 20))   # RRF candidate generation
        if not hybrid:
            return []
        ids, mat = corpus.cluster_matrix()
        row = {cid: i for i, cid in enumerate(ids)}
        q = corpus.encode_query(query)
        scored = [(cid, float(mat[row[cid]] @ q)) for cid, _ in hybrid if cid in row]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:k]
