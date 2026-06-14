"""Filesystem-backed knowledge base.

Layout
------
knowledge/
  # Check-worthy claims (per dataset / detector / article)
  <dataset>/<detector>/articles/<article-name>.json
      {source_path, detector, dataset, article_name, claims: [{sentence, sentence_index}]}

  # Downstream pipeline structures (namespaced by retrieval backend)
  sub_narratives/<sn_id>.json
  narratives/<backend>/<nar_id>.json
  campaigns/<backend>/<camp_id>.json

<dataset>  : 'polynarrative' | 'fake-cti'
<detector> : 'xlm-multicw'  | 'mdb-multicw'
"""
from __future__ import annotations

import json
from pathlib import Path

from core.structures import (
    Article, ArticleClaims, SubNarrative, Narrative, Campaign,
)

# Canonical dataset slug names used as top-level KB folders.
DATASET_POLYNARRATIVE = "polynarrative"
DATASET_FAKECTI       = "fake-cti"


class KnowledgeBase:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Check-worthy claims
    # ------------------------------------------------------------------ #

    def _claims_path(self, dataset: str, detector: str, article_name: str) -> Path:
        """knowledge/<dataset>/<detector>/articles/<article-name>.json"""
        return self.root / dataset / detector / "articles" / f"{article_name}.json"

    def save_article_claims(self, ac: ArticleClaims) -> None:
        path = self._claims_path(ac.dataset, ac.detector, ac.article_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write(path, ac.to_dict())

    def load_article_claims(self, dataset: str, detector: str,
                            article_name: str) -> ArticleClaims | None:
        path = self._claims_path(dataset, detector, article_name)
        return ArticleClaims.from_dict(self._read(path)) if path.exists() else None

    def all_article_claims(self, dataset: str, detector: str) -> list[ArticleClaims]:
        d = self.root / dataset / detector / "articles"
        if not d.is_dir():
            return []
        return [ArticleClaims.from_dict(self._read(p)) for p in sorted(d.glob("*.json"))]

    def claims_exist(self, dataset: str, detector: str, article_name: str) -> bool:
        return self._claims_path(dataset, detector, article_name).exists()

    # ------------------------------------------------------------------ #
    # PolyNarrative raw article store (used by the converter)
    # ------------------------------------------------------------------ #

    def _poly_articles_dir(self) -> Path:
        return self.root / "polynarrative" / "_articles"

    def save_article(self, a: Article) -> None:
        d = self._poly_articles_dir()
        d.mkdir(parents=True, exist_ok=True)
        self._write(d / f"{a.source_domain}_{a.id}.json", a.to_dict())

    def articles(self) -> list[Article]:
        d = self._poly_articles_dir()
        return [Article.from_dict(self._read(p)) for p in d.glob("*.json")] if d.is_dir() else []

    # ------------------------------------------------------------------ #
    # Sub-narratives
    # ------------------------------------------------------------------ #

    def save_sub_narrative(self, sn: SubNarrative) -> None:
        d = self.root / "sub_narratives"
        d.mkdir(parents=True, exist_ok=True)
        self._write(d / f"{sn.id}.json", sn.to_dict())

    def sub_narratives(self) -> list[SubNarrative]:
        d = self.root / "sub_narratives"
        return [SubNarrative.from_dict(self._read(p))
                for p in d.glob("*.json")] if d.is_dir() else []

    def get_sub_narrative(self, sn_id: str) -> SubNarrative | None:
        path = self.root / "sub_narratives" / f"{sn_id}.json"
        return SubNarrative.from_dict(self._read(path)) if path.exists() else None

    # ------------------------------------------------------------------ #
    # Narratives
    # ------------------------------------------------------------------ #

    def save_narrative(self, n: Narrative) -> None:
        d = self.root / "narratives" / n.backend
        d.mkdir(parents=True, exist_ok=True)
        self._write(d / f"{n.id}.json", n.to_dict())

    def narratives(self, backend: str) -> list[Narrative]:
        d = self.root / "narratives" / backend
        return [Narrative.from_dict(self._read(p))
                for p in d.glob("*.json")] if d.is_dir() else []

    def delete_narrative(self, backend: str, narrative_id: str) -> None:
        path = self.root / "narratives" / backend / f"{narrative_id}.json"
        if path.exists():
            path.unlink()

    # ------------------------------------------------------------------ #
    # Campaigns
    # ------------------------------------------------------------------ #

    def save_campaign(self, c: Campaign) -> None:
        d = self.root / "campaigns" / c.backend
        d.mkdir(parents=True, exist_ok=True)
        self._write(d / f"{c.id}.json", c.to_dict())

    def campaigns(self, backend: str) -> list[Campaign]:
        d = self.root / "campaigns" / backend
        return [Campaign.from_dict(self._read(p))
                for p in d.glob("*.json")] if d.is_dir() else []

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _write(path: Path, obj: dict) -> None:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _read(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))
