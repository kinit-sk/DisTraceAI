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
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Minimal sentence splitter — splits on .!? followed by whitespace."""
    sentences = _SENT_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# PolyNarrative article iterator
# ---------------------------------------------------------------------------

def _polynarrative_articles(data_root: Path):
    """Yield (article_name, text, source_path) for each PolyNarrative document.

    Expects the raw PolyNarrative layout:
        data/PolyNarrative/<split>/<lang>/raw-documents/*.txt   (train)
        data/PolyNarrative/<split>/<lang>/subtask-*-documents/*.txt  (dev/test)
    Falls back to any *.txt found recursively under data_root.
    """
    txt_files = sorted(data_root.rglob("*.txt"))
    if not txt_files:
        logger.warning("[cw_generate] No .txt files found under %s", data_root)
        return

    for path in txt_files:
        try:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="latin-1")
            if not text.strip():
                continue
            # article_name: make it path-safe and unique by using relative path stem
            rel = path.relative_to(data_root)
            article_name = str(rel).replace("/", "_").replace("\\", "_")
            if article_name.endswith(".txt"):
                article_name = article_name[:-4]
            source_path = str(Path("data/PolyNarrative") / rel)
            yield article_name, text, source_path
        except Exception as exc:
            logger.warning("[cw_generate] skipping %s: %s", path, exc)


# ---------------------------------------------------------------------------
# FakeCTI article iterator
# ---------------------------------------------------------------------------

def _fakecti_articles(csv_path: Path):
    """Yield (article_name, text, source_path) for each FakeCTI row."""
    if not csv_path.exists():
        logger.warning("[cw_generate] FakeCTI CSV not found: %s", csv_path)
        return

    df = pd.read_csv(csv_path)
    if "TEXT" not in df.columns:
        logger.error("[cw_generate] FakeCTI CSV has no 'TEXT' column — found: %s",
                     list(df.columns))
        return
    if "ID" not in df.columns:
        df["ID"] = df.index + 1

    df["TEXT"] = df["TEXT"].fillna("").astype(str)
    df = df[df["TEXT"].str.strip() != ""].reset_index(drop=True)

    for _, row in df.iterrows():
        article_id   = str(row["ID"])
        text         = row["TEXT"]
        article_name = f"article_{article_id}"
        source_path  = str(Path("data/FakeCTI/FakeCTI.csv"))
        yield article_name, text, source_path


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
        TextColumn("{task.completed}/{task.total} articles"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        art_task = progress.add_task(
            f"[cyan]{dataset_slug}[/cyan] · {detector.slug}", total=len(articles))

        for article_name, text, source_path in articles:
            if kb.claims_exist(dataset_slug, detector.slug, article_name):
                skipped += 1
                progress.advance(art_task)
                continue

            sentences = split_sentences(text)
            if not sentences:
                progress.advance(art_task)
                continue

            # Batch inference with a nested progress task
            batch_size = detector.batch_size
            batches    = list(range(0, len(sentences), batch_size))
            batch_task = progress.add_task(
                f"  [dim]{article_name[:40]}[/dim]", total=len(batches))

            labels: list[int] = []
            for start in batches:
                labels.extend(detector.predict(sentences[start : start + batch_size]))
                progress.advance(batch_task)

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
                claims=cw_claims,
            )
            kb.save_article_claims(ac)
            processed   += 1
            total_claims += len(cw_claims)
            progress.advance(art_task)

    return processed, skipped, total_claims


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
        proc, skip, claims = _process_dataset(
            DATASET_POLYNARRATIVE,
            _polynarrative_articles(POLYNARRATIVE_DATA),
            detector, kb)
        summary[DATASET_POLYNARRATIVE] = {
            "processed": proc, "skipped": skip, "cw_claims": claims}
        console.print(f"  processed={proc}  skipped={skip}  cw_claims={claims}")
    else:
        console.print(f"[yellow]PolyNarrative not found at {POLYNARRATIVE_DATA} — skipping.[/yellow]")

    # FakeCTI
    if FAKECTI_CSV.exists():
        console.print(f"\n[bold]FakeCTI[/bold] — {FAKECTI_CSV}")
        proc, skip, claims = _process_dataset(
            DATASET_FAKECTI,
            _fakecti_articles(FAKECTI_CSV),
            detector, kb)
        summary[DATASET_FAKECTI] = {
            "processed": proc, "skipped": skip, "cw_claims": claims}
        console.print(f"  processed={proc}  skipped={skip}  cw_claims={claims}")
    else:
        console.print(f"[yellow]FakeCTI not found at {FAKECTI_CSV} — skipping.[/yellow]")

    return summary
