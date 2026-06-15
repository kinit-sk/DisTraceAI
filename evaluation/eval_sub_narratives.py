"""Evaluate sub-narrative extraction against PolyNarrative ground truth.

Method
------
For each article in the PolyNarrative test/dev set that has at least one
extracted sub-narrative:

1. Take the central claim of each extracted sub-narrative.
2. Generate ``n_hypotheticals`` hypothetical sub-narrative descriptions for it
   (HyDE style) using the configured LLM.
3. Embed the central claim and all hypotheticals.
4. For each embedding, retrieve the most similar ground-truth sub-narrative
   label from the taxonomy index (built from representative claim texts of
   each distinct label).
5. Vote across the (1 + n_hypotheticals) retrievals.  The winning label is the
   one with the most votes; ties are broken in favour of any label that matches
   ground truth.
6. Compare the winning label to the ground-truth label(s) for that article.
   A match is counted if the winning label equals any ground-truth label.

Metrics reported
----------------
- Per-language: precision, recall, F1, accuracy (over sub-narrative predictions).
- Overall aggregate.
- Results saved to ``results/eval_sub_narratives_<detector>.csv``.
- HTML report saved to ``evaluation/eval_sub_narratives.html``.

Ground-truth source
-------------------
``knowledge/ground_truth/annotations.json`` (written by the PolyNarrative
converter).  Each entry carries ``sub_narratives: [label, ...]``.  Because
PolyNarrative has one dominant sub-narrative per article, the label list is
typically length 1.

Taxonomy index
--------------
Built from the representative texts of each distinct sub-narrative label.  For
each label, the representative text is the concatenation of a sample of central
claims from all extracted sub-narratives of articles carrying that label in the
ground truth.  This gives semantically rich embeddings rather than embedding the
short label string directly.
"""
from __future__ import annotations

import csv
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from rich.console import Console
from rich import box
from rich.table import Table
from rich.terminal_theme import MONOKAI
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    MofNCompleteColumn, TimeElapsedColumn,
)

from core.knowledge_base import KnowledgeBase, DATASET_POLYNARRATIVE
from core.models import make_embedder, make_generator, encode_with_backoff

logger  = logging.getLogger(__name__)
console = Console(record=True)

# ---------------------------------------------------------------------------
# HyDE prompt
# ---------------------------------------------------------------------------
_SYSTEM_HYDE = (
    "You are an expert analyst describing disinformation sub-narratives. /no_think"
)

_USER_HYDE_TMPL = (
    "Given the following central claim of a sub-narrative, write a short "
    "description (≤ 30 words) of a disinformation sub-narrative that this "
    "claim might belong to.\n\n"
    "STRICT RULES:\n"
    "- Output ONLY the sub-narrative description. No explanation.\n"
    "- Write in ENGLISH.\n\n"
    "Central claim:\n{claim}"
)


def _generate_hypotheticals(llm, central_claim: str, n: int) -> list[str]:
    """Generate n hypothetical sub-narrative descriptions for a central claim."""
    hyps = []
    for _ in range(n):
        user   = _USER_HYDE_TMPL.format(claim=central_claim.strip())
        result = llm(_SYSTEM_HYDE, user, max_tokens=64, temperature=0.7)
        if result and result.strip():
            hyps.append(result.strip())
    return hyps


# ---------------------------------------------------------------------------
# Taxonomy index construction
# ---------------------------------------------------------------------------

def _build_taxonomy_index(
    annotations: dict,
    sns_by_article: dict[str, list],
    embedder,
) -> tuple[list[str], np.ndarray]:
    """Build a retrieval index over distinct ground-truth sub-narrative labels.

    Representative text for each label = concatenation of up to 10 central
    claims from extracted sub-narratives of articles carrying that label.

    Returns
    -------
    labels : list[str]
        Ordered list of distinct sub-narrative label strings.
    index_embs : np.ndarray
        L2-normalised embeddings, shape ``(len(labels), dim)``.
    """
    label_texts: dict[str, list[str]] = defaultdict(list)
    for article_name, ann in annotations.items():
        gt_labels = ann.get("sub_narratives", [])
        sns = sns_by_article.get(article_name, [])
        for label in gt_labels:
            for sn in sns:
                if sn.central_claim:
                    label_texts[label].append(sn.central_claim)

    if not label_texts:
        return [], np.empty((0, 0))

    labels = sorted(label_texts.keys())
    representative_texts = [
        " | ".join(label_texts[lbl][:10]) or lbl
        for lbl in labels
    ]

    embs = encode_with_backoff(embedder, representative_texts)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    return labels, embs / norms


def _retrieve_label(
    query_emb: np.ndarray,
    index_labels: list[str],
    index_embs: np.ndarray,
) -> str:
    """Return the label most similar to query_emb (cosine)."""
    norm = np.linalg.norm(query_emb)
    q    = query_emb / (norm if norm > 1e-10 else 1.0)
    return index_labels[int(np.argmax(index_embs @ q))]


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------

def _vote(
    central_claim: str,
    hypotheticals: list[str],
    embedder,
    index_labels: list[str],
    index_embs: np.ndarray,
    gt_labels: list[str],
) -> tuple[str, bool]:
    """Embed all queries, retrieve a label per query, majority-vote for winner.

    Tie-break: if multiple labels share the top vote count and one of them
    matches any ground-truth label, it is selected (counted as correct).

    Returns ``(predicted_label, is_correct)``.
    """
    queries = [central_claim] + hypotheticals
    embs    = encode_with_backoff(embedder, queries)
    norms   = np.linalg.norm(embs, axis=1, keepdims=True)
    norms   = np.where(norms == 0, 1e-10, norms)
    embs    = embs / norms

    votes: dict[str, int] = defaultdict(int)
    for emb in embs:
        votes[_retrieve_label(emb, index_labels, index_embs)] += 1

    max_v      = max(votes.values())
    top_labels = [lbl for lbl, v in votes.items() if v == max_v]
    gt_set     = set(gt_labels)

    for lbl in top_labels:
        if lbl in gt_set:
            return lbl, True

    return top_labels[0], False


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    tp      = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp      = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn      = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    prec    = tp / (tp + fp) if (tp + fp) else 0.0
    rec     = tp / (tp + fn) if (tp + fn) else 0.0
    f1      = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc     = correct / len(y_true) if y_true else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "n": len(y_true)}


def _per_language_metrics(
    langs: list[str], y_true: list[bool], y_pred: list[bool]
) -> dict[str, dict]:
    buckets: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for lang, t, p in zip(langs, y_true, y_pred):
        buckets[lang][0].append(t)
        buckets[lang][1].append(p)
    return {lang: _metrics(yt, yp) for lang, (yt, yp) in sorted(buckets.items())}


# ---------------------------------------------------------------------------
# Rich display
# ---------------------------------------------------------------------------

def _score_style(v: float) -> str:
    return "bold green" if v >= 0.70 else ("yellow" if v >= 0.40 else "red")


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def _print_results(
    detector_slug: str, overall: dict, per_lang: dict[str, dict]
) -> None:
    console.print()
    console.rule(
        f"[bold cyan]Sub-narrative Evaluation — {detector_slug}[/bold cyan]"
    )
    console.print()

    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold white")
    t.add_column("Scope",  style="bold", min_width=10)
    t.add_column("P",      justify="right", min_width=7)
    t.add_column("R",      justify="right", min_width=7)
    t.add_column("F1",     justify="right", min_width=7)
    t.add_column("Acc",    justify="right", min_width=7)
    t.add_column("N",      justify="right", min_width=6)

    def _row(label, m, style=""):
        t.add_row(
            label,
            f"[{_score_style(m['precision'])}]{_fmt(m['precision'])}[/]",
            f"[{_score_style(m['recall'])}]{_fmt(m['recall'])}[/]",
            f"[{_score_style(m['f1'])}]{_fmt(m['f1'])}[/]",
            f"[{_score_style(m['accuracy'])}]{_fmt(m['accuracy'])}[/]",
            str(m["n"]),
            style=style,
        )

    _row("OVERALL", overall, style="bold")
    t.add_section()
    for lang, m in per_lang.items():
        _row(lang, m)

    console.print(t)
    console.print()


# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

def _save_csv(
    detector_slug: str, overall: dict, per_lang: dict[str, dict]
) -> None:
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"eval_sub_narratives_{detector_slug}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope", "precision", "recall", "f1", "accuracy", "n"])
        w.writerow(["overall",
                    overall["precision"], overall["recall"],
                    overall["f1"], overall["accuracy"], overall["n"]])
        for lang, m in per_lang.items():
            w.writerow([lang, m["precision"], m["recall"],
                        m["f1"], m["accuracy"], m["n"]])
    logger.info("[eval_sub_nar] results saved to %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(cfg=None) -> None:
    from config import Config
    cfg = cfg if cfg is not None else Config.load()

    kb_root = Path("knowledge")
    kb      = KnowledgeBase(kb_root)

    gt_path = kb_root / "ground_truth" / "annotations.json"
    if not gt_path.exists():
        poly_src = Path("data/PolyNarrative")
        if not poly_src.exists():
            console.print(
                f"[red]Ground truth not found:[/red] {gt_path}\n"
                f"[red]PolyNarrative data not found:[/red] {poly_src}\n"
                "Place the dataset under data/PolyNarrative/ and re-run."
            )
            return
        console.print(
            f"[yellow]Ground truth not found.[/yellow] "
            f"Running PolyNarrative converter on {poly_src}\u2026"
        )
        from core.converters.polynarrative import convert
        convert(poly_src, kb_root)
        if not gt_path.exists():
            console.print(
                f"[red]Converter ran but {gt_path} was not produced. "
                "Check the PolyNarrative dataset layout.[/red]"
            )
            return
        console.print("[green]Converter finished.[/green]")

    annotations: dict = json.loads(gt_path.read_text(encoding="utf-8"))

    detector_path = cfg.subnar_detector
    if detector_path == "both":
        detector_slugs = ["xlm-multicw", "mdb-multicw"]
    else:
        detector_slugs = [os.path.basename(detector_path.rstrip("/\\"))]

    console.print(
        f"\n[bold]Loading embedder[/bold] [cyan]{cfg.subnar_embedder}[/cyan]…"
    )
    embedder = make_embedder(cfg.subnar_embedder)

    console.print(
        f"[bold]Loading generator[/bold] [cyan]{cfg.subnar_generator}[/cyan] "
        f"([dim]{cfg.subnar_quantization}[/dim])…"
    )
    llm = make_generator(cfg.subnar_generator, cfg.subnar_quantization)

    html_out = Path("evaluation") / "eval_sub_narratives.html"

    for detector_slug in detector_slugs:
        console.print(
            f"\n[bold cyan]Evaluating — {detector_slug}[/bold cyan]"
        )

        all_sns = kb.sub_narratives(DATASET_POLYNARRATIVE, detector_slug)
        if not all_sns:
            console.print(
                f"  [yellow]No sub-narratives found for "
                f"polynarrative/{detector_slug}. Run Generate first.[/yellow]"
            )
            continue

        sns_by_article: dict[str, list] = defaultdict(list)
        for sn in all_sns:
            sns_by_article[sn.article_name].append(sn)

        console.print("  Building taxonomy index…")
        index_labels, index_embs = _build_taxonomy_index(
            annotations, sns_by_article, embedder
        )
        if not index_labels:
            console.print(
                "  [yellow]Taxonomy index is empty — no overlap between ground "
                "truth and extracted sub-narratives. Check that article_names "
                "match annotations.json.[/yellow]"
            )
            continue

        console.print(
            f"  Index: {len(index_labels)} distinct sub-narrative labels."
        )

        articles_eval = [
            name for name in sns_by_article
            if name in annotations and annotations[name].get("sub_narratives")
        ]
        if not articles_eval:
            console.print(
                "  [yellow]No annotated articles with extracted sub-narratives.[/yellow]"
            )
            continue

        console.print(f"  Evaluating {len(articles_eval)} annotated articles…")

        y_true: list[bool] = []
        y_pred: list[bool] = []
        langs:  list[str]  = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]HyDE retrieval[/cyan] {detector_slug}",
                total=len(articles_eval),
            )
            for article_name in articles_eval:
                ann     = annotations[article_name]
                gt_labs = ann.get("sub_narratives", [])
                lang    = ann.get("language", "??")

                for sn in sns_by_article[article_name]:
                    hyps = _generate_hypotheticals(
                        llm, sn.central_claim, cfg.subnar_hypotheticals
                    )
                    _, correct = _vote(
                        sn.central_claim, hyps,
                        embedder, index_labels, index_embs,
                        gt_labs,
                    )
                    y_true.append(True)
                    y_pred.append(correct)
                    langs.append(lang)

                progress.advance(task)

        overall  = _metrics(y_true, y_pred)
        per_lang = _per_language_metrics(langs, y_true, y_pred)

        _print_results(detector_slug, overall, per_lang)
        _save_csv(detector_slug, overall, per_lang)

    del llm

    html_out.parent.mkdir(parents=True, exist_ok=True)
    console.save_html(str(html_out), theme=MONOKAI, clear=False)
    console.print(f"[dim]HTML report saved to {html_out}[/dim]")
