"""Retrieval llm_backends interface (README §5).

A llm_backends ranks the existing narrative clusters (held in a FactCheckCorpus) for a
query claim. The assigner uses the top result to decide merge-or-create.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class RetrievalBackend(ABC):
    name: str

    @abstractmethod
    def rank(self, query: str, corpus, k: int = 10) -> list[tuple[str, float]]:
        """Return [(cluster_id, score)] sorted descending."""
        ...
