"""Filesystem-backed knowledge base.

Layout
------
knowledge/
  # Check-worthy claims (per dataset / detector / article)
  <dataset>/<detector>/articles/<article-name>.json
      {source_path, detector, dataset, article_name, claims: [{sentence, sentence_index}]}

  # Sub-narratives (per dataset / detector / article)
  <dataset>/<detector>/sub-narratives/<sn-id>.json

  # Downstream pipeline structures (namespaced by retrieval backend)
  narratives/<dataset>/<backend>/<nar_id>.json
  campaigns/<dataset>/<backend>/<camp_id>.json
  veracity/cache/<claim_hash>.json
  veracity/multiclaim_test_paraphrases.json
  veracity/multiclaim_embs.npz          (pre-cached MultiClaim embeddings)
  veracity/multiclaim_embs_meta.json    (cache key: embedder name + record hash)

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

    def _articles_dir(self, dataset: str | None = None) -> Path:
        """Dataset-generic article store: <dataset>/_articles/"""
        if dataset:
            return self.root / dataset / "_articles"
        return self.root / "polynarrative" / "_articles"   # legacy default

    def save_article(self, a: Article, dataset: str | None = None) -> None:
        d = self._articles_dir(dataset)
        d.mkdir(parents=True, exist_ok=True)
        self._write(d / f"{a.id}.json", a.to_dict())

    def load_article(self, article_id: str,
                     dataset: str | None = None) -> "Article | None":
        # Try dataset-specific path first, then scan known datasets.
        if dataset:
            path = self._articles_dir(dataset) / f"{article_id}.json"
            if path.exists():
                return Article.from_dict(self._read(path))
            return None
        for ds in ("polynarrative", "fake-cti", "massivesumm"):
            path = self._articles_dir(ds) / f"{article_id}.json"
            if path.exists():
                return Article.from_dict(self._read(path))
        return None

    def articles(self, dataset: str | None = None) -> "list[Article]":
        d = self._articles_dir(dataset)
        return [Article.from_dict(self._read(p))
                for p in d.glob("*.json")] if d.is_dir() else []

    # ------------------------------------------------------------------ #
    # Sub-narratives  (detector-scoped: downstream of a specific detector run)
    # ------------------------------------------------------------------ #

    def _sub_narratives_dir(self, dataset: str, detector: str) -> Path:
        """knowledge/<dataset>/<detector>/sub-narratives/"""
        return self.root / dataset / detector / "sub-narratives"

    def save_sub_narrative(self, sn: SubNarrative) -> None:
        d = self._sub_narratives_dir(sn.dataset, sn.detector)
        d.mkdir(parents=True, exist_ok=True)
        self._write(d / f"{sn.id}.json", sn.to_dict())

    def sub_narratives(self, dataset: str, detector: str) -> list[SubNarrative]:
        d = self._sub_narratives_dir(dataset, detector)
        if not d.is_dir():
            return []
        return [SubNarrative.from_dict(self._read(p)) for p in sorted(d.glob("*.json"))]

    def get_sub_narrative(self, dataset: str, detector: str,
                          sn_id: str) -> SubNarrative | None:
        path = self._sub_narratives_dir(dataset, detector) / f"{sn_id}.json"
        return SubNarrative.from_dict(self._read(path)) if path.exists() else None

    def sub_narratives_exist(self, dataset: str, detector: str,
                             article_name: str) -> bool:
        """True if any sub-narrative has already been extracted for this article."""
        d = self._sub_narratives_dir(dataset, detector)
        if not d.is_dir():
            return False
        # Sub-narrative IDs are prefixed with article_name (see gen_sub_narratives).
        return any(True for _ in d.glob(f"{article_name}_sn*.json"))

    # ------------------------------------------------------------------ #
    # Narratives
    # ------------------------------------------------------------------ #

    def save_narrative(self, n: Narrative) -> None:
        d = self.root / "narratives" / n.dataset / n.backend
        d.mkdir(parents=True, exist_ok=True)
        self._write(d / f"{n.id}.json", n.to_dict())

    def narratives(self, dataset: str, backend: str) -> list[Narrative]:
        d = self.root / "narratives" / dataset / backend
        return [Narrative.from_dict(self._read(p))
                for p in d.glob("*.json")] if d.is_dir() else []

    def delete_narrative(self, dataset: str, backend: str, narrative_id: str) -> None:
        path = self.root / "narratives" / dataset / backend / f"{narrative_id}.json"
        if path.exists():
            path.unlink()

    # ------------------------------------------------------------------ #
    # Campaigns
    # ------------------------------------------------------------------ #

    def save_campaign(self, c: Campaign) -> None:
        d = self.root / "campaigns" / c.dataset / c.backend
        d.mkdir(parents=True, exist_ok=True)
        self._write(d / f"{c.id}.json", c.to_dict())

    def campaigns(self, dataset: str, backend: str) -> list[Campaign]:
        d = self.root / "campaigns" / dataset / backend
        return [Campaign.from_dict(self._read(p))
                for p in d.glob("*.json")] if d.is_dir() else []

    def delete_campaign(self, dataset: str, backend: str, cid: str) -> None:
        path = self.root / "campaigns" / dataset / backend / f"{cid}.json"
        if path.exists():
            path.unlink()

    # ------------------------------------------------------------------ #
    # Veracity cache
    # ------------------------------------------------------------------ #

    def _veracity_cache_dir(self) -> Path:
        return self.root / "veracity" / "cache"

    def save_veracity_cache(self, claim_hash: str, record: dict) -> None:
        d = self._veracity_cache_dir()
        d.mkdir(parents=True, exist_ok=True)
        self._write(d / f"{claim_hash}.json", record)

    def load_veracity_cache(self, claim_hash: str) -> dict | None:
        path = self._veracity_cache_dir() / f"{claim_hash}.json"
        return self._read(path) if path.exists() else None

    def save_paraphrase_test(self, records: list, generator: str) -> None:
        d = self.root / "veracity"
        d.mkdir(parents=True, exist_ok=True)
        self._write(d / "multiclaim_test_paraphrases.json",
                    {"generator": generator, "records": records})

    def load_paraphrase_test(self, generator: str) -> list | None:
        path = self.root / "veracity" / "multiclaim_test_paraphrases.json"
        if not path.exists():
            return None
        data = self._read(path)
        if data.get("generator") != generator:
            return None   # stale cache — different generator
        return data.get("records", [])

    def save_multiclaim_embs(self, ids: list[str], embs: "np.ndarray",
                             embedder_name: str, record_hash: str) -> None:
        """Persist the pre-computed MultiClaim embedding matrix to disk.

        Stored as a compressed .npz (ids + float32 matrix) alongside a small
        JSON metadata file. Invalidated automatically when the embedder or the
        MultiClaim records change (detected via record_hash).
        """
        import numpy as np
        d = self.root / "veracity"
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(d / "multiclaim_embs.npz"),
                            ids=np.array(ids, dtype=object),
                            embs=embs.astype(np.float32))
        self._write(d / "multiclaim_embs_meta.json",
                    {"embedder": embedder_name, "record_hash": record_hash,
                     "n": len(ids)})

    def load_multiclaim_embs(self, embedder_name: str,
                             record_hash: str) -> "tuple | None":
        """Load cached MultiClaim embeddings if the cache is still valid.

        Returns (ids: list[str], embs: np.ndarray) or None if stale/absent.
        The cache is valid iff both the embedder name and the record content
        hash match — so changing the embedder OR the MultiClaim CSV invalidates
        it automatically.
        """
        import numpy as np
        meta_path = self.root / "veracity" / "multiclaim_embs_meta.json"
        npz_path  = self.root / "veracity" / "multiclaim_embs.npz"
        if not meta_path.exists() or not npz_path.exists():
            return None
        meta = self._read(meta_path)
        if meta.get("embedder") != embedder_name or \
           meta.get("record_hash") != record_hash:
            return None   # stale: different embedder or CSV changed
        data = np.load(str(npz_path), allow_pickle=True)
        ids  = list(data["ids"])
        embs = data["embs"].astype(np.float32)
        return ids, embs

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _write(path: Path, obj: dict) -> None:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _read(path: Path) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt KB record at {path}: {exc}") from exc
