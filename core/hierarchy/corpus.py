"""Dense + BM25 retrieval corpus over narrative clusters (README §5, §6).

Each narrative cluster contributes one or more text entries (its central claim
plus member sub-narrative claims). Retrieval fuses BM25 and dense rankings with
Reciprocal Rank Fusion (k=60) at the entry level, then aggregates to the cluster
by best fused score.

`remove_cluster` is a first-class operation (row-mask on the dense matrix + BM25
rebuild from cached tokens — no re-encoding, no private-internals surgery), which
is what lets the re-clustering sweep (§6) stay clean.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from core.models import encode_with_backoff

logger = logging.getLogger(__name__)
_RRF_K = 60


def _tokenize(text: str) -> list[str]:
    import re
    return re.findall(r"\w+", text.lower())


class FactCheckCorpus:
    def __init__(self, embedder, device: str = "cpu") -> None:
        self.embedder = embedder
        self.device = device
        self._cluster_ids: list[str] = []   # parallel to entries
        self._texts: list[str] = []
        self._tokens: list[list[str]] = []
        self._emb: np.ndarray | None = None  # (N, d) per-entry, L2-normalised
        self._bm25 = None
        self._dirty = False
        self._cluster_mat: tuple | None = None   # memorised cluster_matrix() result

    # ---- mutation --------------------------------------------------------
    def add_cluster(self, cluster_id: str, texts: list[str]) -> None:
        """Append entries for a cluster (idempotent-append: re-calling extends it)."""
        new = [t for t in texts if t and t.strip()]
        if not new:
            return
        embs = np.asarray(encode_with_backoff(self.embedder, new), dtype=np.float32)
        embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        self._cluster_ids.extend([cluster_id] * len(new))
        self._texts.extend(new)
        self._tokens.extend(_tokenize(t) for t in new)
        self._emb = embs if self._emb is None else np.vstack([self._emb, embs])
        self._dirty = True
        self._cluster_mat = None

    def add_many(self, items: list[tuple[str, str]], *, batch_size: int = 32,
                 show_progress: bool = False) -> None:
        """Append many (cluster_id, text) pairs with a SINGLE batched encode and a
        single matrix allocation — avoids the per-document GPU call and O(n^2)
        vstack of repeated add_cluster(). Encoding goes through the OOM back-off
        helper so embedding very large corpora (e.g. 435k fact-checks) degrades
        gracefully instead of crashing."""
        pairs = [(cid, t) for cid, t in items if t and t.strip()]
        if not pairs:
            return
        texts = [t for _, t in pairs]
        embs = np.asarray(encode_with_backoff(self.embedder, texts,
                                              initial_batch_size=batch_size,
                                              show_progress=show_progress),
                          dtype=np.float32)
        embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        self.add_precomputed([cid for cid, _ in pairs], texts, embs)

    def add_precomputed(self, cluster_ids: list[str], texts: list[str],
                        embs: np.ndarray) -> None:
        """Append entries with already-computed (and L2-normalised) embeddings."""
        self._cluster_ids.extend(cluster_ids)
        self._texts.extend(texts)
        self._tokens.extend(_tokenize(t) for t in texts)
        self._emb = embs if self._emb is None else np.vstack([self._emb, embs])
        self._dirty = True
        self._cluster_mat = None

    def remove_cluster(self, cluster_id: str) -> None:
        """Drop every entry of a cluster. Clean replacement for the old in-place surgery."""
        keep = [i for i, c in enumerate(self._cluster_ids) if c != cluster_id]
        if len(keep) == len(self._cluster_ids):
            return
        self._cluster_ids = [self._cluster_ids[i] for i in keep]
        self._texts = [self._texts[i] for i in keep]
        self._tokens = [self._tokens[i] for i in keep]
        self._emb = self._emb[keep] if self._emb is not None and keep else None
        self._dirty = True
        self._cluster_mat = None

    # ---- introspection (used by SpecFi-C / Context-1) ---------------------
    def encode_query(self, query: str) -> np.ndarray:
        q = np.asarray(self.embedder.encode([query]), dtype=np.float32)[0]
        return q / (np.linalg.norm(q) + 1e-12)

    def cluster_ids(self) -> list[str]:
        return sorted(set(self._cluster_ids))

    def cluster_document(self, cluster_id: str) -> str:
        """Concatenated text of a cluster's entries (the 'document' agents read)."""
        return " ".join(t for c, t in zip(self._cluster_ids, self._texts) if c == cluster_id)

    def grep(self, pattern: str) -> list[tuple[str, str]]:
        """Regex over cluster documents. Returns [(cluster_id, document)]."""
        import re
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return []
        out = []
        for cid in self.cluster_ids():
            doc = self.cluster_document(cid)
            if rx.search(doc):
                out.append((cid, doc))
        return out

    def cluster_matrix(self) -> tuple[list[str], np.ndarray]:
        """Mean-pooled, L2-normalised embedding per cluster.

        Memoised: recomputed only after a mutation, not on every call. Callers
        (SpecFi-C, DenseBackend) invoke this once per query, so without the memo a
        doc corpus paid an O(entries x clusters) regroup on every single query."""
        if self._cluster_mat is not None:
            return self._cluster_mat
        ids = self.cluster_ids()
        if not ids or self._emb is None:
            self._cluster_mat = (ids, np.zeros((len(ids), 0), dtype=np.float32))
            return self._cluster_mat
        groups: dict[str, list[int]] = defaultdict(list)      # one pass over entries
        for i, c in enumerate(self._cluster_ids):
            groups[c].append(i)
        rows = []
        for cid in ids:
            v = self._emb[groups[cid]].mean(axis=0)
            rows.append(v / (np.linalg.norm(v) + 1e-12))
        self._cluster_mat = (ids, np.vstack(rows))
        return self._cluster_mat

    # ---- search ----------------------------------------------------------
    def _ensure_built(self) -> None:
        if not self._dirty:
            return
        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi(self._tokens) if self._tokens else None
        self._dirty = False

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Hybrid BM25+dense RRF, aggregated to clusters. Returns [(cluster_id, score)]."""
        if not self._cluster_ids:
            return []
        self._ensure_built()
        n = len(self._texts)
        cand = min(max(k, 50), n)   # expand the fusion pool when a large k is requested

        ranked_lists: list[list[int]] = []
        if self._bm25 is not None:
            bm25_scores = self._bm25.get_scores(_tokenize(query))
            # Only use BM25 when it actually discriminates; on tiny corpora the IDF
            # can collapse to zero (term in 1 of 2 docs) and the reversed argsort of
            # all-equal scores would inject a spurious ordering that ties the fusion.
            if float(np.ptp(bm25_scores)) > 1e-9:
                ranked_lists.append(list(np.argsort(bm25_scores)[::-1][:cand]))
        q = np.asarray(self.embedder.encode([query]), dtype=np.float32)[0]
        q /= (np.linalg.norm(q) + 1e-12)
        dense = self._emb @ q
        ranked_lists.append(list(np.argsort(dense)[::-1][:cand]))

        fused: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, idx in enumerate(ranked):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (_RRF_K + rank + 1)

        per_cluster: dict[str, float] = {}
        for idx, score in fused.items():
            cid = self._cluster_ids[idx]
            if score > per_cluster.get(cid, -1.0):
                per_cluster[cid] = score
        return sorted(per_cluster.items(), key=lambda x: x[1], reverse=True)[:k]
