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
    """KB record: check-worthy claims extracted from one article."""
    source_path: str                  # relative path to original article in data/
    detector: str                     # e.g. 'xlm-multicw' or 'mdb-multicw'
    dataset: str                      # 'polynarrative' or 'fake-cti'
    article_name: str                 # stem used as the filename
    claims: list[CheckWorthyClaim] = field(default_factory=list)

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
    id: str
    article_id: str                   # plain field; replaces the old `article_ref` hash
    central_claim: str                # English
    related_claims: list[str] = field(default_factory=list)
    veracity: float | None = None     # 0..1, lower = more likely false
    veracity_confidence: float | None = None
    confidence: float = 1.0           # clustering confidence (N6)

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
