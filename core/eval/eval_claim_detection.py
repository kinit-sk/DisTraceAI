"""Evaluate check-worthiness detection against MultiCW test set.

Loads data/MultiCW/multicw-test.csv, runs the configured detector, and
prints a Rich table with overall metrics plus a per-language breakdown.
Results are also saved to results/eval_cw_<detector>.csv.

Metric math (binary precision / recall / F1 / accuracy + per-language
breakdown) lives in ``core.eval.metrics`` and is shared with
``eval_sub_narratives``.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import (Progress, SpinnerColumn, BarColumn, TextColumn,
                           TimeElapsedColumn, MofNCompleteColumn)
from rich.terminal_theme import MONOKAI

from core.eval.metrics import (
    binary_metrics, per_language_binary, print_prf_table,
)

logger = logging.getLogger(__name__)
console = Console(record=True)

DATA_PATH    = Path("data/MultiCW/multicw-test.csv")


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
    """Run detector over rows; return (overall_metrics, per_lang_metrics)."""
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

    return binary_metrics(y_true, y_pred), per_language_binary(langs, y_true, y_pred)


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
    print_prf_table(
        console, f"Check-Worthiness Detection — {detector.slug}",
        overall, per_lang)
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
    except Exception as exc:
        # We don't fail the eval just because the stats sidecar couldn't be
        # written, but we want to know it happened (per code-review note).
        logger.warning("[eval_cw] save_eval_stats failed: %s", exc)

    from core.eval.report_paths import report_path
    html_out = report_path("claim-detection", detector=detector.slug)
    console.save_html(str(html_out), theme=MONOKAI, clear=False)
    console.print(f"[dim]HTML report saved to {html_out}[/dim]")


if __name__ == "__main__":
    main()
