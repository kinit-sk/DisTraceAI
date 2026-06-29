"""Shared evaluation metrics.

Both ``eval_claim_detection`` and ``eval_sub_narratives`` compute binary
precision / recall / F1 / accuracy and a per-language breakdown of the same
metrics. Before this module each had its own copy with slightly different
defaults. Putting the math in one place removes the duplication and makes the
two evaluations directly comparable.

Two flavours are exposed:

* ``binary_metrics(y_true, y_pred)`` — when both arrays are 0/1 ints
  (claim-detection's check-worthy class).
* ``correctness_metrics(y_true, y_pred)`` — same shape but ``y_true`` is the
  ground-truth booleans and ``y_pred`` is whether the system's prediction was
  correct (sub-narratives, after the HyDE vote: the "positive class" is
  "predicted correctly"). Mathematically identical to ``binary_metrics`` once
  both arrays are coerced to 0/1; kept as a separate name so call sites read
  more clearly at the point of use.

Per-language wrappers ``per_language_binary`` / ``per_language_correctness``
bucket by an aligned ``langs`` list and call the corresponding scalar metric.
"""
from __future__ import annotations

from collections import defaultdict


def _prf_acc(tp: int, fp: int, fn: int, total: int, correct: int) -> dict:
    """Pure scalar metric: P / R / F1 / accuracy / N from the four counts."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    accuracy  = correct / total if total else 0.0
    return {"precision": precision, "recall": recall,
            "f1": f1, "accuracy": accuracy, "n": total}


def binary_metrics(y_true, y_pred) -> dict:
    """Binary metrics where the positive class is ``1``.

    ``y_true`` / ``y_pred`` must be aligned int (or coerceable) sequences of
    length N. Used by check-worthy detection: positive class = check-worthy
    sentence.
    """
    tp = fp = fn = correct = 0
    for t, p in zip(y_true, y_pred):
        t_i, p_i = int(t), int(p)
        if t_i == 1 and p_i == 1:
            tp += 1
        elif t_i == 0 and p_i == 1:
            fp += 1
        elif t_i == 1 and p_i == 0:
            fn += 1
        if t_i == p_i:
            correct += 1
    n = len(y_true) if hasattr(y_true, "__len__") else sum(1 for _ in y_true)
    return _prf_acc(tp, fp, fn, n, correct)


def correctness_metrics(y_true, y_pred) -> dict:
    """Metrics where the positive class is ``True`` (== a correct prediction).

    Used by sub-narrative HyDE-vote eval: ``y_true`` is always all-True (every
    annotated sub-narrative is an evaluation target), and ``y_pred`` is True
    iff the winning label matched any ground-truth label for that article.
    Mathematically identical to ``binary_metrics`` after bool→int coercion.
    """
    tp = fp = fn = correct = 0
    for t, p in zip(y_true, y_pred):
        if t and p:        tp += 1
        elif (not t) and p: fp += 1
        elif t and (not p): fn += 1
        if t == p:          correct += 1
    n = len(y_true) if hasattr(y_true, "__len__") else sum(1 for _ in y_true)
    return _prf_acc(tp, fp, fn, n, correct)


def _per_language(metric_fn, langs, y_true, y_pred) -> dict[str, dict]:
    """Bucket aligned (lang, y_true, y_pred) triples and apply ``metric_fn``."""
    buckets: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for lang, t, p in zip(langs, y_true, y_pred):
        buckets[lang][0].append(t)
        buckets[lang][1].append(p)
    return {lang: metric_fn(yt, yp) for lang, (yt, yp) in sorted(buckets.items())}


def per_language_binary(langs, y_true, y_pred) -> dict[str, dict]:
    return _per_language(binary_metrics, langs, y_true, y_pred)


def per_language_correctness(langs, y_true, y_pred) -> dict[str, dict]:
    return _per_language(correctness_metrics, langs, y_true, y_pred)


# ---------------------------------------------------------------------------
# Rich display helpers (also shared) — same P/R/F1/Acc table layout used by
# both consumers. score_style + fmt are inlined here so the consumers don't
# each have to redefine them.
# ---------------------------------------------------------------------------

def score_style(v: float) -> str:
    if v >= 0.70:
        return "bold green"
    if v >= 0.40:
        return "yellow"
    return "red"


def fmt_score(v: float) -> str:
    return f"{v:.3f}"


def print_prf_table(console, title: str, overall: dict,
                    per_lang: dict[str, dict]) -> None:
    """Render the standard P / R / F1 / Acc / N table with one row per language.

    Both eval modules render the same table; centralising it keeps formatting
    consistent and removes ~30 lines of duplicate Rich boilerplate per module.
    """
    from rich import box
    from rich.table import Table

    console.print()
    console.rule(f"[bold cyan]{title}[/bold cyan]")
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
            f"[{score_style(m['precision'])}]{fmt_score(m['precision'])}[/]",
            f"[{score_style(m['recall'])}]{fmt_score(m['recall'])}[/]",
            f"[{score_style(m['f1'])}]{fmt_score(m['f1'])}[/]",
            f"[{score_style(m['accuracy'])}]{fmt_score(m['accuracy'])}[/]",
            str(m["n"]),
            style=style,
        )

    _row("OVERALL", overall, style="bold")
    t.add_section()
    for lang, m in per_lang.items():
        _row(lang, m)

    console.print(t)
    console.print()
