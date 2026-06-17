"""Evaluate check-worthiness detection against MultiCW test set.

Loads data/MultiCW/multicw-test.csv, runs the configured detector, and
prints a Rich table with overall metrics plus a per-language breakdown.
Results are also saved to results/eval_cw_<detector>.csv.
"""
from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.table import Table
from rich import box
from rich.terminal_theme import MONOKAI

logger = logging.getLogger(__name__)
console = Console(record=True)

DATA_PATH    = Path("data/MultiCW/multicw-test.csv")
EVAL_HTML_OUT = Path("evaluation/eval_claim_detection.html")


# ---------------------------------------------------------------------------
# Pure metric helpers
# ---------------------------------------------------------------------------

def cw_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Binary metrics for the check-worthy (label == 1) class."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    accuracy  = correct / len(y_true) if y_true else 0.0
    return {"precision": precision, "recall": recall,
            "f1": f1, "accuracy": accuracy, "n": len(y_true)}


def per_language_metrics(langs: list[str],
                         y_true: list[int],
                         y_pred: list[int]) -> dict[str, dict]:
    """Full binary metrics per language code."""
    buckets: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for lang, t, p in zip(langs, y_true, y_pred):
        buckets[lang][0].append(t)
        buckets[lang][1].append(p)
    return {lang: cw_metrics(yt, yp) for lang, (yt, yp) in sorted(buckets.items())}


# ---------------------------------------------------------------------------
# Load & run
# ---------------------------------------------------------------------------

def _load_test_rows() -> list[dict]:
    df = pd.read_csv(DATA_PATH)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(np.int32)
    df["text"]  = df["text"].fillna("").astype(str)
    df["lang"]  = df["lang"].fillna("??").astype(str)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    return df[["text", "label", "lang"]].to_dict(orient="records")


def evaluate(detector, rows: list[dict]) -> tuple[dict, dict[str, dict]]:
    """Run detector over rows; return (overall_metrics, per_lang_metrics).

    Displays a Rich progress bar at the batch level so long test sets show
    incremental progress rather than a frozen terminal.
    """
    texts  = [r["text"]  for r in rows]
    y_true = [int(r["label"]) for r in rows]
    langs  = [r["lang"]  for r in rows]

    n_batches = detector.num_batches(len(texts))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Evaluating[/cyan] {detector.slug}", total=n_batches)
        y_pred = detector.predict(
            texts, progress_callback=lambda: progress.advance(task))

    return cw_metrics(y_true, y_pred), per_language_metrics(langs, y_true, y_pred)


# ---------------------------------------------------------------------------
# Rich display
# ---------------------------------------------------------------------------

def _score_style(v: float) -> str:
    if v >= 0.70:
        return "bold green"
    if v >= 0.40:
        return "yellow"
    return "red"


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def print_results(detector_slug: str,
                  overall: dict,
                  per_lang: dict[str, dict]) -> None:
    console.print()
    console.rule(f"[bold cyan]Check-Worthiness Detection — {detector_slug}[/bold cyan]")
    console.print()

    # ---- overall row ----
    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold white")
    t.add_column("Scope",     style="bold", min_width=10)
    t.add_column("P",         justify="right", min_width=7)
    t.add_column("R",         justify="right", min_width=7)
    t.add_column("F1",        justify="right", min_width=7)
    t.add_column("Acc",       justify="right", min_width=7)
    t.add_column("N",         justify="right", min_width=6)

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

def save_csv(detector_slug: str, overall: dict, per_lang: dict[str, dict]) -> None:
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"eval_cw_{detector_slug}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope", "precision", "recall", "f1", "accuracy", "n"])
        w.writerow(["overall",
                    overall["precision"], overall["recall"],
                    overall["f1"], overall["accuracy"], overall["n"]])
        for lang, m in per_lang.items():
            w.writerow([lang, m["precision"], m["recall"],
                        m["f1"], m["accuracy"], m["n"]])
    logger.info("[eval_cw] results saved to %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(cfg=None) -> None:
    from config import Config
    from core.claims.cw_detector import CheckWorthinessDetector

    cfg = cfg if cfg is not None else Config.load()

    if not DATA_PATH.exists():
        console.print(f"[red]MultiCW test set not found:[/red] {DATA_PATH}")
        return

    console.print(f"[dim]Loading MultiCW test set from {DATA_PATH}…[/dim]")
    rows = _load_test_rows()
    console.print(f"[dim]{len(rows)} sentences loaded.[/dim]")

    console.print(f"[dim]Loading detector: {cfg.detector}…[/dim]")
    detector = CheckWorthinessDetector(cfg.detector)

    overall, per_lang = evaluate(detector, rows)
    print_results(detector.slug, overall, per_lang)
    save_csv(detector.slug, overall, per_lang)

    try:
        from core.ui.stats import save_eval_stats
        save_eval_stats(
            "claim-detection",
            param_key=detector.slug,
            params={"detector": detector.slug},
            scores={"f1": overall["f1"], "acc": overall["accuracy"],
                    "n": overall["n"]},
            det_slug=detector.slug,
        )
    except Exception:
        pass

    from evaluation.report_paths import report_path
    html_out = report_path("claim-detection", detector=detector.slug)
    console.save_html(str(html_out), theme=MONOKAI, clear=False)
    console.print(f"[dim]HTML report saved to {html_out}[/dim]")


if __name__ == "__main__":
    main()
