from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Article:
    id: str
    url: str
    title: str
    content: str                      # original-language full text
    source_domain: str
    source_language: str = "und"      # ISO code; used by the N4 coordination signal
    published_at: str | None = None   # ISO timestamp (a field, never part of an id)
    author: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Article":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


@dataclass
class CheckWorthyClaim:
    """A single check-worthy sentence extracted from an article."""
    sentence: str
    sentence_index: int               # position in the original article's sentence list

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CheckWorthyClaim":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class ArticleClaims:
    """KB record: check-worthy claims extracted from one article.

    Carries the article's own metadata (title, author, source language and any
    extra fields under ``metadata``) so the per-article JSON is self-contained.
    Downstream steps fill in ``canonized_claims`` (canonization) and
    ``verdicts`` / ``verified`` (veracity estimation).

    Index alignment: ``canonized_claims[i]`` and ``verdicts[i]`` both correspond
    to ``claims[i]``.
    """
    source_path: str                  # relative path to original article in data/
    detector: str                     # e.g. 'xlm-multicw' or 'mdb-multicw'
    dataset: str                      # 'polynarrative' or 'fake-cti'
    article_name: str                 # stem used as the filename
    title: str = ""
    author: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    claims: list[CheckWorthyClaim] = field(default_factory=list)
    canonized_claims: list[str] = field(default_factory=list)
    verdicts: list[str] = field(default_factory=list)   # per-claim: True/False/Disputed
    verified: bool = False            # True once claims have been verified

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArticleClaims":
        raw = d.copy()
        raw["claims"] = [CheckWorthyClaim.from_dict(c) for c in raw.get("claims", [])]
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})


@dataclass
class SubNarrative:
    """A cluster of thematically related canonized claims from one article.

    ``central_claim`` is a synthesized English sentence that summarises the
    cluster.  ``claims`` are the canonized claim strings that were assigned to
    this cluster (cosine-similarity ≥ threshold).  ``article_name`` references
    the ``ArticleClaims`` record so full article metadata is always retrievable.
    ``veracity`` / ``veracity_confidence`` are filled by the veracity step.
    """
    id: str
    article_name: str                 # FK → ArticleClaims.article_name
    dataset: str                      # 'polynarrative' or 'fake-cti'
    detector: str                     # e.g. 'xlm-multicw'
    central_claim: str                # synthesized English summary
    claims: list[str] = field(default_factory=list)   # canonized claim strings
    veracity: float | None = None     # 0..1, lower = more likely false
    veracity_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SubNarrative":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})


@dataclass
class Narrative:
    id: str
    backend: str                      # which retrieval backend produced it
    central_claim: str                # English
    sub_narratives: list[str] = field(default_factory=list)   # membership lives here
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Narrative":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})


@dataclass
class Campaign:
    id: str
    backend: str
    label: str                        # Disinformation Campaign / Information Campaign / Organic Trend
    narratives: list[str] = field(default_factory=list)
    coordination: dict[str, float] = field(default_factory=dict)  # per-signal scores (N1–N4)
    confidence: float = 1.0           # propagated campaign-level confidence (N6)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Campaign":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})
