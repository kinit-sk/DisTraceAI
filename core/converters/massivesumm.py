"""MassiveSumm → KB converter for dataset compilation (README §10).

Source: data/MassiveSumm/ — expected layout:
  data/MassiveSumm/<lang>/articles.jsonl   (one article per line)
  data/MassiveSumm/<lang>/<lang>.jsonl     (alternative single-file layout)

Each JSONL line is expected to have at least:
  {"id": "...", "text": "...", "title": "...", "domain": "...",
   "date": "YYYY-MM-DD", "lang": "sk"}

Language subset defaults to SK and CZ (Slovak + Czech), as per the project
scope for the dataset publication KPI.

The converter emits Article records into the KB.  Unlike FakeCTI, MassiveSumm
does not carry ground-truth campaign labels — it is used solely for dataset
generation (full pipeline → CSV export).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from core.structures import Article
from core.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

_SUPPORTED_LANGS = ("sk", "cs", "cz")   # Slovak + Czech
_DATASET_SLUG    = "massivesumm"

_LANG_NORM = {"cz": "cs"}   # normalize common alternative spellings


def _iter_articles(lang_dir: Path) -> "Generator[dict, None, None]":
    """Yield raw dicts from any JSONL file found in a language directory."""
    for path in sorted(lang_dir.rglob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.debug("[massivesumm] JSON parse error in %s: %s", path, exc)


def convert(src: Path, out_root: Path, *,
            langs: list[str] | None = None) -> int:
    """Convert MassiveSumm SK/CZ articles to KB Article records.

    Args:
        src:      Path to data/MassiveSumm/ directory.
        out_root: KB root (knowledge/).
        langs:    Language codes to include. Defaults to ['sk', 'cs'].

    Returns:
        Total number of articles written.
    """
    src, out_root = Path(src), Path(out_root)
    if not src.exists():
        raise FileNotFoundError(
            f"MassiveSumm data not found at {src}. "
            "Place the SK/CZ subset under data/MassiveSumm/.")

    target_langs = [_LANG_NORM.get(l, l) for l in (langs or ["sk", "cs"])]
    kb = KnowledgeBase(out_root)

    total = 0
    for lang_dir in sorted(src.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang = _LANG_NORM.get(lang_dir.name.lower(), lang_dir.name.lower())
        if lang not in target_langs:
            continue

        n_lang = 0
        for _idx, raw in enumerate(_iter_articles(lang_dir)):
            text = str(raw.get("text", raw.get("body", ""))).strip()
            if not text:
                continue
            url   = str(raw.get("url", raw.get("link", f"massivesumm://{lang}/{raw.get('id','')}")))
            title = str(raw.get("title", text[:80]))
            date  = str(raw.get("date", raw.get("published_date", "")))[:10] or None
            domain = str(raw.get("domain", raw.get("source", "")))

            # Use the JSONL record's native id field for a human-readable,
            # stable article name. Fall back to a positional counter if absent.
            raw_id = raw.get("id") or raw.get("article_id") or str(_idx)
            aid = f"article_{raw_id}"
            kb.save_article(Article(
                id=aid,
                url=url,
                title=title,
                content=text,
                source_domain=domain,
                source_language=lang.upper(),
                published_at=date,
                author=str(raw.get("author", "Unknown")),
            ), dataset="massivesumm")
            n_lang += 1

        logger.info("[massivesumm] %s: %d articles converted", lang, n_lang)
        total += n_lang

    print(f"MassiveSumm [{','.join(target_langs)}]: {total} articles → {out_root}")
    return total
