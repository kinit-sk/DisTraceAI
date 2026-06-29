"""EUvsDisinfo → KB converter for dataset compilation.

Source: data/EUvsDisinfo/ — a CSV from the EUvsDisinfo dataset (Leite et al.,
CIKM '24; Zenodo 10.5281/zenodo.10514307). The dataset is a multilingual corpus
of pro-Kremlin disinformation articles paired with trustworthy articles, sourced
from EUvsDisinfo debunks.

IMPORTANT — two CSV variants exist, and only one carries article text:
  * ``euvsdisinfo_base.csv`` (public Zenodo release): METADATA + URLs ONLY, no
    article body. The authors do not redistribute the full text for copyright
    reasons. This converter CANNOT run the pipeline on it — there is nothing to
    extract claims from. Run their reconstruction software (Zenodo 10492913) or
    obtain the full dataset from the authors first.
  * the RECONSTRUCTED csv: adds an article-body column (one of ``article_text``
    / ``text`` / ``content`` / ``body``). This converter reads that and emits
    Article records into the KB.

Base-CSV columns (all variants):
  debunk_id, keywords, article_id, article_publisher, article_domain,
  article_url, article_language, debunk_date (dd-mm-yyyy),
  class (trustworthy | disinformation)

Field mapping into the KB Article:
  article_id        → id (prefixed ``article_``)
  body text         → content                 (REQUIRED; row skipped if empty)
  article_url       → url
  article_language  → source_language          (ISO code, upper-cased)
  article_domain    → source_domain
  debunk_date       → published_at (YYYY-MM-DD) — real dates, unlike MassiveSumm,
                      so the N1 burst coordination signal is meaningful here.
  article_publisher → author (best-effort provenance)

EUvsDisinfo is multilingual (40+ languages). By default we keep ALL languages —
that is the dataset's purpose — but ``langs`` can restrict to a subset (e.g.
["sk", "cs"]) if you want to mirror the earlier SK/CZ scope.

Note: debunk_id (which groups articles into a shared debunked narrative) and
class (disinformation/trustworthy) are NOT stored on the Article (the record has
no slot for them). They remain available in the CSV and could later be wired in
as campaign/veracity ground truth — see the offer in the converter docstring.
"""
from __future__ import annotations

import csv
import hashlib
import logging
from pathlib import Path

from core.structures import Article
from core.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

_DATASET_SLUG = "euvsdisinfo"

# Candidate column names for the article body in the reconstructed CSV, in
# priority order. The public base CSV has none of these (URLs only).
_TEXT_COLS = ("article_text", "text", "content", "body", "article_body")

# Normalize a few common language spellings to 2-letter ISO codes. EUvsDisinfo
# mostly already uses 2-letter codes; this is a light safety net.
_LANG_NORM = {
    "english": "en", "russian": "ru", "czech": "cs", "cze": "cs", "ces": "cs",
    "slovak": "sk", "slo": "sk", "slk": "sk", "german": "de", "french": "fr",
    "spanish": "es", "polish": "pl", "ukrainian": "uk", "bulgarian": "bg",
}


def _norm_lang(code: str) -> str:
    code = (code or "").strip().lower()
    return _LANG_NORM.get(code, code)


def _norm_date(raw: str) -> str | None:
    """EUvsDisinfo debunk_date is dd-mm-yyyy; emit ISO YYYY-MM-DD."""
    raw = (raw or "").strip()
    if not raw:
        return None
    parts = raw.replace("/", "-").split("-")
    if len(parts) == 3 and len(parts[0]) == 2:        # dd-mm-yyyy
        d, m, y = parts
        return f"{y}-{m.zfill(2)}-{d.zfill(2)}"
    if len(parts) == 3 and len(parts[0]) == 4:        # already yyyy-mm-dd
        return raw[:10]
    return raw[:10] or None


def _first_text(row: dict) -> str:
    for col in _TEXT_COLS:
        v = row.get(col)
        if v and str(v).strip():
            return str(v).strip()
    return ""


def convert(src: Path, out_root: Path, *,
            langs: list[str] | None = None,
            limit: int | None = None) -> int:
    """Convert EUvsDisinfo articles to KB Article records.

    Args:
        src:      Path to data/EUvsDisinfo/ (any *.csv inside is read).
        out_root: KB root (knowledge/).
        langs:    ISO language codes to keep. ``None`` (default) keeps all
                  languages; pass e.g. ["sk", "cs"] to restrict.
        limit:    Maximum number of articles to write. ``None`` (default) or
                  0 = no limit; positive values stop ingestion once that many
                  records have been persisted. Used by Generate Dataset to
                  honour ``cfg.camp_sample_size`` at conversion time instead
                  of paying for ingestion of records the pipeline will drop.

    Returns:
        Total number of articles written.
    """
    src, out_root = Path(src), Path(out_root)
    if not src.exists():
        raise FileNotFoundError(
            f"EUvsDisinfo data not found at {src}. Place a EUvsDisinfo CSV "
            "(reconstructed, with article text) under data/EUvsDisinfo/.")

    csvs = sorted(src.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No CSV found under {src}. Expected e.g. euvsdisinfo_base.csv "
            "(reconstructed with article text).")

    target = {_norm_lang(l) for l in langs} if langs is not None else None
    kb = KnowledgeBase(out_root)

    # The default csv.field_size_limit (131072 bytes) is too small for articles
    # with long body text. Raise it to the largest value the platform allows
    # (sys.maxsize overflows on some builds; binary-search down to a safe cap).
    import sys as _sys
    _field_limit = min(int(2**31 - 1), _sys.maxsize)
    while True:
        try:
            csv.field_size_limit(_field_limit)
            break
        except OverflowError:
            _field_limit //= 2

    per_lang: dict[str, int] = {}
    seen_ids: set[str] = set()
    rows_total = textless = 0
    # Plan §4.6 + cfg.camp_sample_size: cap articles converted at `limit`.
    # `limit=None` or 0 means unbounded.
    eff_limit = int(limit or 0) or None
    n_written = 0

    for path in csvs:
        if eff_limit is not None and n_written >= eff_limit:
            break
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if eff_limit is not None and n_written >= eff_limit:
                    break
                rows_total += 1
                lang = _norm_lang(row.get("article_language", ""))
                if target is not None and lang not in target:
                    continue

                text = _first_text(row)
                if not text:
                    textless += 1
                    continue

                url    = str(row.get("article_url", "")).strip()
                domain = str(row.get("article_domain", "")).strip()
                date   = _norm_date(row.get("debunk_date", ""))
                publisher = str(row.get("article_publisher", "") or "Unknown").strip()

                raw_id = str(row.get("article_id", "")).strip()
                if not raw_id:
                    raw_id = hashlib.md5((url or text).encode("utf-8")).hexdigest()[:12]
                aid = f"article_{raw_id}"
                if aid in seen_ids:
                    continue
                seen_ids.add(aid)

                title = str(row.get("title") or text[:80]).strip()

                kb.save_article(Article(
                    id=aid,
                    url=url or f"euvsdisinfo://{lang}/{raw_id}",
                    title=title,
                    content=text,
                    source_domain=domain,
                    source_language=(lang or "und").upper(),
                    published_at=date,
                    author=publisher,
                ), dataset=_DATASET_SLUG)
                per_lang[lang] = per_lang.get(lang, 0) + 1
                n_written += 1

    total = sum(per_lang.values())
    for lang, n in sorted(per_lang.items()):
        logger.info("[euvsdisinfo] %s: %d articles converted", lang, n)

    if total == 0 and textless and textless == rows_total - 0:
        # Every row lacked body text → almost certainly the base (URL-only) CSV.
        raise RuntimeError(
            f"EUvsDisinfo CSV at {src} has no article-body column "
            f"({textless}/{rows_total} rows text-less). The public "
            "euvsdisinfo_base.csv contains URLs only; run the authors' "
            "reconstruction software (Zenodo 10492913) to fetch article text, "
            "then re-run. Expected a body column among: "
            f"{', '.join(_TEXT_COLS)}.")
    if textless:
        logger.warning("[euvsdisinfo] %d/%d rows had no article text (skipped)",
                       textless, rows_total)

    scope = "all langs" if target is None else ",".join(sorted(target))
    print(f"EUvsDisinfo [{scope}]: {total} articles → {out_root}")
    return total
