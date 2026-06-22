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

import hashlib
import json
import logging
from pathlib import Path

from core.structures import Article
from core.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

_SUPPORTED_LANGS = ("sk", "cs", "cz")   # Slovak + Czech

_DATASET_SLUG    = "massivesumm"

# Normalize the various spellings/ISO-639 codes seen in the wild to 2-letter.
# MassiveSumm ships 3-letter ISO-639-3 codes (slk, ces) in both the filenames
# (slk.all.jsonl / ces.all.jsonl) and the per-record "language" field.
_LANG_NORM = {
    "cz": "cs", "cze": "cs", "ces": "cs", "czech": "cs",
    "slk": "sk", "slo": "sk", "slovak": "sk",
}


def _iter_jsonl(path: Path) -> "Generator[dict, None, None]":
    """Yield raw dicts from one JSONL file."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.debug("[massivesumm] JSON parse error in %s: %s", path, exc)


def _norm_lang(code: str) -> str:
    code = (code or "").strip().lower()
    return _LANG_NORM.get(code, code)


def convert(src: Path, out_root: Path, *,
            langs: list[str] | None = None) -> int:
    """Convert MassiveSumm SK/CZ articles to KB Article records.

    Handles both layouts: flat top-level files (``slk.all.jsonl`` /
    ``ces.all.jsonl``) and per-language subdirectories. Language is taken from
    each record's ``language``/``lang`` field (normalized from ISO-639-3), with
    the filename stem as a fallback.

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

    target_langs = {_norm_lang(l) for l in (langs or ["sk", "cs"])}
    kb = KnowledgeBase(out_root)

    per_lang: dict[str, int] = {}
    seen_ids: set[str] = set()
    for path in sorted(src.rglob("*.jsonl")):
        # Filename-stem language fallback: "slk.all.jsonl" -> "slk", and the
        # subdirectory name for the legacy <lang>/articles.jsonl layout.
        file_lang = _norm_lang(path.name.split(".", 1)[0])
        if file_lang not in target_langs:
            file_lang = _norm_lang(path.parent.name)

        for _idx, raw in enumerate(_iter_jsonl(path)):
            lang = _norm_lang(raw.get("language") or raw.get("lang") or file_lang)
            if lang not in target_langs:
                continue
            text = str(raw.get("text", raw.get("body", ""))).strip()
            if not text:
                continue
            url   = str(raw.get("url", raw.get("link", "")))
            title = str(raw.get("title") or text[:80])
            date  = (str(raw.get("date") or raw.get("published_date") or "")[:10]
                     or None)
            domain = str(raw.get("domain", raw.get("source", "")))

            # No native id in MassiveSumm records — derive a stable one from the
            # URL (or the text if the URL is missing), so re-runs are idempotent.
            raw_id = raw.get("id") or raw.get("article_id")
            if not raw_id:
                basis = url or text
                raw_id = hashlib.md5(basis.encode("utf-8")).hexdigest()[:12]
            aid = f"article_{raw_id}"
            if aid in seen_ids:
                continue
            seen_ids.add(aid)

            kb.save_article(Article(
                id=aid,
                url=url or f"massivesumm://{lang}/{raw_id}",
                title=title,
                content=text,
                source_domain=domain,
                source_language=lang.upper(),
                published_at=date,
                author=str(raw.get("author", "Unknown")),
            ), dataset="massivesumm")
            per_lang[lang] = per_lang.get(lang, 0) + 1

    total = sum(per_lang.values())
    for lang, n in sorted(per_lang.items()):
        logger.info("[massivesumm] %s: %d articles converted", lang, n)
    print(f"MassiveSumm [{','.join(sorted(target_langs))}]: {total} articles "
          f"→ {out_root}")
    return total
