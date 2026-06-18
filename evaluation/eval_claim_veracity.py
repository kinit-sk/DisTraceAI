"""Veracity evaluation benchmark.

Method: paraphrase-based evaluation with leave-one-out.

1. Load MultiClaim CSV, filter to True/False/Disputed entries.
2. Generate N paraphrases per claim using Gemma4-12b (cached to
   ``knowledge/veracity/multiclaim_test_paraphrases.json`` for reproducibility).
3. For each paraphrase query:
     - Exclude its source claim from the MultiClaim evidence corpus
       (leave-one-out: prevents trivial retrieval of the exact source).
     - Run the Context-1 agentic harness to gather evidence.
     - Synthesize a verdict (True/False/Disputed) via Gemma4-e2b.
     - Compare against the source claim's gold label.
4. Report 3-class accuracy + macro-F1.

Why paraphrase-based?  Human-assigned gold labels from MultiClaim are genuine
ground truth.  Paraphrasing varies surface form while preserving semantics, so
the eval measures robustness to wording variation — exactly the real-world
scenario.  A disclosed limitation: LLM paraphrases may be easier than natural
reformulations; results are a mild upper bound on real performance.
"""
from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich import box
from rich.table import Table
from rich.terminal_theme import MONOKAI
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    MofNCompleteColumn, TimeElapsedColumn,
)

from core.knowledge_base import KnowledgeBase
from core.models import make_embedder, make_generator
from core.claims.gen_veracity import (
    _load_multiclaim, build_evidence_tools,
    synthesize_verdict, _VERACITY_SYSTEM,
)

logger  = logging.getLogger(__name__)
console = Console(record=True)

_LABELS = ["True", "False", "Disputed"]
_MULTICLAIM_PATH = Path("data/MultiClaim/fact_checks.csv")


# ---------------------------------------------------------------------------
# Paraphrase generation + caching
# ---------------------------------------------------------------------------

_PARAPHRASE_SYSTEM = """\
Rewrite the given claim in different words while preserving its exact meaning.
Output ONLY the rewritten claim. No explanation, no quotes. /no_think
"""


def _generate_paraphrase(claim: str, llm) -> str:
    try:
        raw = (llm(_PARAPHRASE_SYSTEM, f"Claim: {claim}", max_tokens=120) or "").strip()
    except TypeError:
        raw = (llm(_PARAPHRASE_SYSTEM, f"Claim: {claim}") or "").strip()
    return raw or claim


def _build_test_set(records: list[dict], cfg, kb: KnowledgeBase) -> list[dict]:
    """Generate paraphrases or load from cache.

    Returns list of {source_id, paraphrase, gold_label, generator, quant}.
    """
    cached = kb.load_paraphrase_test(cfg.ver_paraphrase_generator,
                                     cfg.ver_quantization)
    if cached:
        console.print(
            f"[dim]Loaded {len(cached)} cached paraphrase test entries.[/dim]")
        return cached

    console.print(
        f"\n[bold]Generating paraphrases[/bold] "
        f"({cfg.ver_n_paraphrases}× each, "
        f"{len(records)} claims → "
        f"~{len(records) * cfg.ver_n_paraphrases} queries)…")
    console.print(
        f"[dim]Generator: {cfg.ver_paraphrase_generator} "
        f"({cfg.ver_quantization})[/dim]\n")

    llm = make_generator(cfg.ver_paraphrase_generator, cfg.ver_quantization)
    test_records = []
    with Progress(SpinnerColumn(),
                  TextColumn("[progress.description]{task.description}"),
                  BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                  console=console) as prog:
        task = prog.add_task("[cyan]Paraphrasing…[/cyan]", total=len(records))
        for rec in records:
            for _ in range(max(1, cfg.ver_n_paraphrases)):
                para = _generate_paraphrase(rec["text"], llm)
                if para and para.strip() != rec["text"].strip():
                    test_records.append({
                        "source_id":   rec["id"],
                        "paraphrase":  para,
                        "gold_label":  rec["label"],
                        "generator":   cfg.ver_paraphrase_generator,
                        "quant":       cfg.ver_quantization,
                    })
            prog.advance(task)
    del llm

    kb.save_paraphrase_test(test_records,
                            cfg.ver_paraphrase_generator, cfg.ver_quantization)
    console.print(
        f"[dim]{len(test_records)} paraphrase queries cached to "
        f"knowledge/veracity/multiclaim_test_paraphrases.json[/dim]")
    return test_records


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _macro_f1(confusion: dict[str, dict[str, int]]) -> dict[str, float]:
    """Per-class precision/recall/F1 and macro-F1 from a confusion dict."""
    result: dict[str, float] = {}
    f1s = []
    for label in _LABELS:
        tp = confusion[label].get(label, 0)
        fp = sum(confusion[other].get(label, 0)
                 for other in _LABELS if other != label)
        fn = sum(confusion[label].get(pred, 0)
                 for pred in _LABELS if pred != label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        result[f"prec_{label}"] = prec
        result[f"rec_{label}"]  = rec
        result[f"f1_{label}"]   = f1
        f1s.append(f1)
    result["macro_f1"] = sum(f1s) / len(f1s)
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(cfg=None) -> None:
    from config import Config
    cfg = cfg or Config.load()

    kb_root = Path("knowledge")
    kb = KnowledgeBase(kb_root)

    # Load MultiClaim
    mc_path = _MULTICLAIM_PATH
    if not mc_path.exists():
        for p in (Path("data/MultiClaim").glob("*.csv")
                  if Path("data/MultiClaim").exists() else []):
            mc_path = p; break
    records = _load_multiclaim(
        mc_path, cfg.ver_multiclaim_text_col, cfg.ver_multiclaim_label_col)
    if not records:
        console.print(
            "[red]MultiClaim data not found or empty.[/red] "
            "Place fact_checks.csv under data/MultiClaim/.")
        return

    console.print(
        f"\n[bold cyan]Claim veracity evaluation[/bold cyan]\n"
        f"[dim]{len(records)} MultiClaim records (True/False/Disputed)[/dim]")

    # Build / load cached test set
    test_set = _build_test_set(records, cfg, kb)
    if not test_set:
        console.print("[red]No test paraphrases generated.[/red]")
        return

    # Build evidence embedder and verdict generator
    console.print(f"\n[bold]Loading embedder[/bold] [cyan]{cfg.camp_embedder}[/cyan]…")
    embedder = make_embedder(cfg.camp_embedder)
    console.print(
        f"[bold]Loading verdict generator[/bold] [cyan]{cfg.ver_generator}[/cyan]…")
    llm = make_generator(cfg.ver_generator, cfg.ver_quantization)

    # Pre-build the MultiClaim embedding index ONCE (cached to disk) so the
    # leave-one-out loop does not re-encode the corpus for every query.
    # We instantiate a no-exclude instance purely to trigger index building/loading.
    console.print("[bold]Preparing MultiClaim evidence index…[/bold]")
    _seed_tools = build_evidence_tools(
        cfg, embedder,
        kb=kb,
        embedder_name=cfg.camp_embedder,
    )
    # Force the index to build now (warm the cache) before the eval loop.
    # The loop will make new instances but they'll all load from the .npz cache.
    for src in _seed_tools._sources:
        if hasattr(src, "_ensure_index"):
            src._ensure_index()
    del _seed_tools

    # Index of source_id → record (for leave-one-out)
    id_to_record = {r["id"]: r for r in records}

    confusion: dict[str, dict[str, int]] = {l: defaultdict(int) for l in _LABELS}
    per_label_n: dict[str, int] = defaultdict(int)

    from core.hierarchy.harness import AgenticSearchHarness

    with Progress(SpinnerColumn(),
                  TextColumn("[progress.description]{task.description}"),
                  BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                  console=console) as prog:
        task = prog.add_task("[cyan]Evaluating…[/cyan]", total=len(test_set))

        for entry in test_set:
            source_id  = entry["source_id"]
            paraphrase = entry["paraphrase"]
            gold       = entry["gold_label"]

            # Build evidence tools with this source claim excluded (leave-one-out).
            # The embedding index is already on disk from the warm-up above, so
            # this call loads from cache in ~milliseconds.
            tools = build_evidence_tools(
                cfg, embedder,
                exclude_ids={source_id},
                kb=kb,
                embedder_name=cfg.camp_embedder,
            )

            harness = AgenticSearchHarness(
                tools, llm, _VERACITY_SYSTEM,
                token_budget=cfg.ver_token_budget,
                top_k=5,
                max_turns=cfg.ver_max_turns,
            )
            evidence = harness.search(paraphrase)
            predicted, _ = synthesize_verdict(paraphrase, evidence, llm)

            # Normalize to title-case
            gold_norm = gold.title() if gold.title() in _LABELS else "Disputed"
            pred_norm = predicted.title() if predicted.title() in _LABELS else "Disputed"

            confusion[gold_norm][pred_norm] += 1
            per_label_n[gold_norm] += 1
            prog.advance(task)

    del llm

    # Compute metrics
    total = sum(confusion[l][l] for l in _LABELS)
    n_all = sum(per_label_n.values())
    accuracy = total / n_all if n_all else 0.0
    metrics  = _macro_f1(confusion)

    # Display
    console.print()
    console.rule("[bold cyan]Veracity Evaluation Results[/bold cyan]")
    console.print(
        f"  [dim]queries={n_all}  "
        f"paraphrase_generator={cfg.ver_paraphrase_generator}  "
        f"verdict_generator={cfg.ver_generator}[/dim]")
    console.print()

    # Overall table
    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold white")
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")
    t.add_row("Accuracy",  f"{accuracy:.3f}")
    t.add_row("Macro-F1",  f"{metrics['macro_f1']:.3f}")
    for label in _LABELS:
        t.add_row(f"  F1 / {label}",
                  f"{metrics[f'f1_{label}']:.3f}  "
                  f"(P={metrics[f'prec_{label}']:.2f} "
                  f"R={metrics[f'rec_{label}']:.2f})")
    console.print(t)

    # Confusion matrix
    console.print("[dim]Confusion matrix (rows=gold, cols=predicted):[/dim]")
    cm = Table(box=box.SIMPLE, show_header=True, header_style="dim")
    cm.add_column("Gold\\Pred")
    for p in _LABELS:
        cm.add_column(p, justify="right")
    for g in _LABELS:
        cm.add_row(g, *[str(confusion[g][p]) for p in _LABELS])
    console.print(cm)

    # CSV
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "eval_claim_veracity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["accuracy", accuracy])
        w.writerow(["macro_f1", metrics["macro_f1"]])
        for label in _LABELS:
            w.writerow([f"f1_{label}", metrics[f"f1_{label}"]])

    # Stats
    try:
        from core.ui.stats import save_eval_stats
        save_eval_stats(
            "claim-veracity",
            param_key=f"{cfg.ver_generator}__{cfg.ver_quantization}",
            params={"generator": cfg.ver_generator,
                    "quant": cfg.ver_quantization,
                    "sources": cfg.ver_sources},
            scores={"accuracy": accuracy, "macro_f1": metrics["macro_f1"],
                    "n": n_all},
        )
    except Exception:
        pass

    from evaluation.report_paths import report_path
    html_out = report_path(
        "claim-veracity",
        extra=f"{cfg.ver_generator}__{cfg.ver_quantization}")
    console.save_html(str(html_out), theme=MONOKAI, clear=False)
    console.print(f"[dim]Results → {csv_path}  HTML → {html_out}[/dim]")
