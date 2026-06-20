"""Generate step: extract check-worthy claims from PolyNarrative and FakeCTI.

For each dataset article the detector scores every sentence. Only check-worthy
sentences (label == 1) are kept. Results are stored in the knowledge base as:

    knowledge/<dataset>/<detector>/articles/<article-name>.json

Articles already present in the KB are skipped (idempotent).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from core.claims.cw_detector import CheckWorthinessDetector
from core.converters.polynarrative import metadata_for as _poly_metadata_for
from core.converters.polynarrative import parse_train_annotations, parse_dev_annotations
from core.ids import article_name_from_relpath
from core.knowledge_base import KnowledgeBase, DATASET_POLYNARRATIVE, DATASET_FAKECTI
from core.structures import ArticleClaims, CheckWorthyClaim

logger = logging.getLogger(__name__)
console = Console()

POLYNARRATIVE_DATA = Path("data/PolyNarrative")
FAKECTI_CSV        = Path("data/FakeCTI/FakeCTI.csv")
KNOWLEDGE_ROOT     = Path("knowledge")

# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

# Matches sentence-ending punctuation (.!?।) optionally followed by closing
# quote/bracket characters, then a lookahead requiring an uppercase letter to
# start the next sentence.  This handles:
#   - Normal periods:   "He said. She replied."
#   - Closing quotes:   '"Done." Next sentence.'  (curly or straight)
#   - Always-safe !?:   "Alert! The system. Restart."
#   - Hindi danda ।:    split on newlines (one sentence per line)
#   - Cyrillic capital: "Россия напала. Украина устояла."
#   - Abbreviations:    "U.S. forces" NOT split ("forces" starts lowercase)
#
# Note: two-letter titles like "Dr." and "Mr." still produce false splits when
# followed by a capitalised surname ("Dr. Smith" → ["Dr", "Smith went home"]).
# This was equally true of the original regex and is an acceptable trade-off
# for a multilingual corpus without a language-specific abbreviation allowlist.
_SENT_BOUNDARY = re.compile(
    r'(?<=[.!?।])'                                      # after terminating punctuation
    r'["\u2018\u2019\u201c\u201d)]*'                # consume any trailing quotes/brackets
    r'(?=\s+[A-Z\u0400-\u042F\"\u201c\u0900-\u0939])'  # uppercase/Devanagari continuation
)


def split_sentences(text: str) -> list[str]:
    """Multilingual sentence splitter for news articles and CTI reports.

    Strategy:
    1. Split on every newline so that line-per-sentence news formatting and
       paragraph breaks are handled correctly.
    2. Within each line, split on sentence boundaries: .!?। optionally
       followed by closing quotes/brackets, then an uppercase-letter start.

    Handles Latin, Cyrillic, Devanagari (Hindi) scripts and quoted sentences.
    Avoids false splits on common abbreviations like "U.S." (lowercase follows).
    """
    if not text or not text.strip():
        return []

    sentences: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = _SENT_BOUNDARY.split(line)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)

    return sentences


# ---------------------------------------------------------------------------
# PolyNarrative article iterator
# ---------------------------------------------------------------------------

def _poly_annotations(split_lang_dir: Path, split: str) -> dict:
    """Load the annotation map for one <split>/<lang> directory (best effort)."""
    if split == "train":
        return parse_train_annotations(split_lang_dir / "subtask-3-annotations.txt")
    return parse_dev_annotations(split_lang_dir / "subtask-3-dominant-narratives.txt")


def _polynarrative_articles(data_root: Path):
    """Yield (article_name, text, source_path, meta) for each PolyNarrative document.

    Expects the raw PolyNarrative layout:
        data/PolyNarrative/<split>/<lang>/raw-documents/*.txt   (train)
        data/PolyNarrative/<split>/<lang>/subtask-*-documents/*.txt  (dev/test)
    Falls back to any *.txt found recursively under data_root.

    ``meta`` carries the synthetic title/author/metadata produced by the
    PolyNarrative converter, keyed off the same document, so the per-article KB
    record is self-contained.
    """
    txt_files = sorted(data_root.rglob("*.txt"))
    if not txt_files:
        logger.warning("[cw_generate] No .txt files found under %s", data_root)
        return

    # Annotation / label files share the .txt extension but are not documents.
    _NON_DOCUMENT = {
        "subtask-3-annotations.txt",
        "subtask-3-dominant-narratives.txt",
    }
    txt_files = [p for p in txt_files if p.name not in _NON_DOCUMENT]

    _ann_cache: dict[Path, dict] = {}

    for path in txt_files:
        try:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="latin-1")
            if not text.strip():
                continue
            # article_name: path-safe and unique via the canonical helper.
            rel = path.relative_to(data_root)
            article_name = article_name_from_relpath(rel)
            source_path = str(Path("data/PolyNarrative") / rel)

            # Derive <split>/<lang> from the relative path: <split>/<lang>/<subdir>/<file>
            parts = rel.parts
            split = parts[0] if len(parts) >= 1 else ""
            lang  = parts[1] if len(parts) >= 2 else "EN"
            doc_id = path.name

            ann_dir = data_root / split / lang
            if ann_dir not in _ann_cache:
                _ann_cache[ann_dir] = _poly_annotations(ann_dir, split) if ann_dir.is_dir() else {}
            entry = _ann_cache[ann_dir].get(doc_id) or _ann_cache[ann_dir].get(Path(doc_id).stem)
            narratives = entry.get("narratives") if entry else None

            meta = _poly_metadata_for(doc_id, text, lang, narratives)
            yield article_name, text, source_path, meta
        except Exception as exc:
            logger.warning("[cw_generate] skipping %s: %s", path, exc)


# ---------------------------------------------------------------------------
# FakeCTI article iterator
# ---------------------------------------------------------------------------

def _fakecti_articles(csv_path: Path):
    """Yield (article_name, text, source_path, meta) for usable FakeCTI rows.

    FakeCTI ships ~12k rows, but most are unusable for this pipeline: non-WEB
    media types, rows with no extractable date, and campaigns too small to
    exhibit coordination (the corpus is ~94% one campaign). Processing the whole
    file wastes hours of CW detection / canonization on rows the campaign step
    will later discard. We therefore apply the SAME scope filter the FakeCTI
    converter uses — TYPE==WEB, non-empty text, an extractable date, and
    campaign size >= FAKECTI_MIN_CAMPAIGN — so every downstream step (which read
    from the KB, not the CSV) inherits the filtered set automatically.

    Filter knobs match the converter and are env-overridable:
      FAKECTI_TYPE          (default "WEB"; "ALL" disables the type filter)
      FAKECTI_MIN_CAMPAIGN  (default 5)
    """
    if not csv_path.exists():
        logger.warning("[cw_generate] FakeCTI CSV not found: %s", csv_path)
        return

    # Reuse the converter's date/domain logic so the scope is identical and
    # there is a single source of truth for "usable FakeCTI row".
    import os
    from collections import defaultdict
    from core.converters.fakecti import extract_date, source_domain

    type_filter = os.getenv("FAKECTI_TYPE", "WEB").upper()
    min_campaign = int(os.getenv("FAKECTI_MIN_CAMPAIGN", "5"))

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().upper() for c in df.columns]
    if "TEXT" not in df.columns:
        logger.error("[cw_generate] FakeCTI CSV has no 'TEXT' column — found: %s",
                     list(df.columns))
        return
    if "ID" not in df.columns:
        df["ID"] = df.index + 1

    df["TEXT"] = df["TEXT"].fillna("").astype(str)
    for col in ("URL", "TITLE", "SOURCE", "CAMPAIGN", "TYPE"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    # 1) media-type scope.
    df["TYPE"] = df["TYPE"].str.strip().str.upper()
    if type_filter and type_filter != "ALL":
        df = df[df["TYPE"] == type_filter]

    # 2) build candidate records with non-empty text + an extractable date.
    extra_cols = [c for c in df.columns if c not in ("TEXT", "ID")]
    candidates = []
    for _, row in df.iterrows():
        text = str(row["TEXT"]).strip()
        url  = str(row["URL"]).strip()
        if not text or not url:
            continue
        date = extract_date(url)
        if not date:
            continue
        campaign = str(row["CAMPAIGN"]).strip()
        if not campaign:
            continue
        candidates.append((row, text, url, date, campaign, extra_cols))

    # 3) drop campaigns too small to exhibit coordination.
    per_campaign: dict[str, int] = defaultdict(int)
    for _row, _text, _url, _date, campaign, _ex in candidates:
        per_campaign[campaign] += 1
    kept = {c for c, n in per_campaign.items() if n >= min_campaign}

    n_total = len(df)
    n_kept = 0
    for row, text, url, date, campaign, ex in candidates:
        if campaign not in kept:
            continue
        n_kept += 1
        article_id   = str(row["ID"])
        article_name = f"article_{article_id}"
        source_path  = str(csv_path)
        first_line   = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        # Fold the real article metadata into the stored metadata dict (the only
        # free-form field ArticleClaims persists) so the campaign step's N1-N4
        # signals can read genuine date/domain/language structure later.
        extra_meta = {c: (None if pd.isna(row[c]) else row[c]) for c in ex}
        extra_meta.update({
            "published_at": date,
            "source_domain": source_domain(url, str(row.get("SOURCE", ""))),
            "source_language": "EN",
            "campaign": campaign,
        })
        meta = {
            "title":  (str(row["TITLE"]).strip() or first_line)[:200],
            "author": None,
            "metadata": extra_meta,
        }
        yield article_name, text, source_path, meta

    logger.info("[cw_generate] FakeCTI scope: %d usable rows kept from %d "
                "(TYPE=%s, dated, campaign>=%d, %d campaigns)",
                n_kept, n_total, type_filter, min_campaign, len(kept))


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------

def _process_dataset(dataset_slug: str,
                     article_iter,
                     detector: CheckWorthinessDetector,
                     kb: KnowledgeBase) -> tuple[int, int, int]:
    """Process one dataset; return (articles_processed, articles_skipped, total_claims)."""
    processed = skipped = total_claims = 0

    articles = list(article_iter)
    if not articles:
        return 0, 0, 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} batches"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        art_task = progress.add_task(
            f"[cyan]{dataset_slug}[/cyan] · {detector.slug}", total=len(articles))

        for article_name, text, source_path, meta in articles:
            if kb.claims_exist(dataset_slug, detector.slug, article_name):
                skipped += 1
                progress.advance(art_task)
                continue

            sentences = split_sentences(text)
            if not sentences:
                progress.advance(art_task)
                continue

            # Single batching point inside predict(); the callback advances the
            # per-article progress task once per completed batch.
            n_batches  = detector.num_batches(len(sentences))
            batch_task = progress.add_task(
                f"  [dim]{article_name[:40]}[/dim]", total=n_batches)

            labels = detector.predict(
                sentences, progress_callback=lambda: progress.advance(batch_task))

            progress.remove_task(batch_task)

            cw_claims = [
                CheckWorthyClaim(sentence=s, sentence_index=i)
                for i, (s, lbl) in enumerate(zip(sentences, labels))
                if lbl == 1
            ]

            ac = ArticleClaims(
                source_path=source_path,
                detector=detector.slug,
                dataset=dataset_slug,
                article_name=article_name,
                title=meta.get("title", "") if meta else "",
                author=meta.get("author") if meta else None,
                metadata=meta.get("metadata", {}) if meta else {},
                claims=cw_claims,
            )
            kb.save_article_claims(ac)
            processed   += 1
            total_claims += len(cw_claims)
            progress.advance(art_task)

    skip_note = (f"  [dim]{skipped} already processed (skipped)[/dim]"
                 if skipped else "")
    total_in_kb = kb.all_article_claims(dataset_slug, detector.slug)
    console.print(
        f"  new={total_claims} claims  skipped={skipped}  total_articles_in_kb={len(total_in_kb)}"
        + (f"\n{skip_note}" if skipped else "")
    )
    return processed, skipped, total_claims, len(total_in_kb)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate(detector: CheckWorthinessDetector,
             kb: KnowledgeBase | None = None) -> dict:
    """Extract CW claims from PolyNarrative and FakeCTI into the KB.

    Returns a summary dict with counts per dataset.
    """
    if kb is None:
        kb = KnowledgeBase(KNOWLEDGE_ROOT)

    summary: dict = {}

    # PolyNarrative
    if POLYNARRATIVE_DATA.exists():
        console.print(f"\n[bold]PolyNarrative[/bold] — {POLYNARRATIVE_DATA}")
        proc, skip, claims, total = _process_dataset(
            DATASET_POLYNARRATIVE,
            _polynarrative_articles(POLYNARRATIVE_DATA),
            detector, kb)
        summary[DATASET_POLYNARRATIVE] = {
            "processed": proc, "skipped": skip,
            "new_cw_claims": claims, "total_in_kb": total}
    else:
        console.print(f"[yellow]PolyNarrative not found at {POLYNARRATIVE_DATA} — skipping.[/yellow]")

    # FakeCTI
    if FAKECTI_CSV.exists():
        console.print(f"\n[bold]FakeCTI[/bold] — {FAKECTI_CSV}")
        proc, skip, claims, total = _process_dataset(
            DATASET_FAKECTI,
            _fakecti_articles(FAKECTI_CSV),
            detector, kb)
        summary[DATASET_FAKECTI] = {
            "processed": proc, "skipped": skip,
            "new_cw_claims": claims, "total_in_kb": total}
    else:
        console.print(f"[yellow]FakeCTI not found at {FAKECTI_CSV} — skipping.[/yellow]")

    return summary
