"""Campaign extraction evaluation against FakeCTI ground truth.

Ground truth: ``knowledge/ground_truth/annotations_by_campaign.json``
(written by the FakeCTI converter) maps campaign name → list of article IDs.

Evaluation unit: per-article campaign assignment. For each article in the
intersection of (extracted campaigns) and (FakeCTI ground truth), we build:
  * true_labels  — gold campaign name per article
  * pred_labels  — extracted campaign ID per article

Metrics: Adjusted Rand Index (ARI), Normalised Mutual Information (NMI),
V-measure (homogeneity/completeness/V). These are standard cluster-quality
metrics that do not require label alignment between predicted and gold clusters.

The article→campaign mapping is traced via:
    campaign.narratives → narrative.sub_narratives → sub_narrative.article_name
"""
from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich import box
from rich.table import Table
from rich.terminal_theme import MONOKAI

from core.knowledge_base import KnowledgeBase, DATASET_FAKECTI

logger  = logging.getLogger(__name__)
console = Console(record=True)


# ---------------------------------------------------------------------------
# Article→campaign tracing
# ---------------------------------------------------------------------------

def _trace_campaign_articles(camp, kb: KnowledgeBase,
                              dataset: str) -> list[str]:
    """Return all article_names reachable from a campaign via the hierarchy."""
    articles = []
    for nar_id in camp.narratives:
        nar = None
        for backend in ("dense", "bm25-rag", "bm25_rag",
                        "specfi-cs", "cspecfi", "context-1"):
            for n in kb.narratives(dataset, backend):
                if n.id == nar_id:
                    nar = n; break
            if nar:
                break
        if nar is None:
            continue
        for sn_id in nar.sub_narratives:
            for det in ("xlm-multicw", "mdb-multicw"):
                sns = kb.sub_narratives(dataset, det)
                for sn in sns:
                    if sn.id == sn_id:
                        articles.append(sn.article_name)
                        break
    return articles


def _build_assignment_arrays(kb: KnowledgeBase, dataset: str,
                              gt_by_article: dict[str, str],
                              backends: list[str]) -> tuple[list, list]:
    """Build aligned (true_labels, pred_labels) for all traceable articles."""
    # Collect campaigns across all backends
    all_camps = []
    for backend in backends:
        all_camps += kb.campaigns(dataset, backend)

    # Build article → predicted campaign mapping
    pred_map: dict[str, str] = {}
    for camp in all_camps:
        for art in _trace_campaign_articles(camp, kb, dataset):
            pred_map[art] = camp.id

    # Intersect with ground truth
    true_labels = []
    pred_labels = []
    for art_id, gold_campaign in gt_by_article.items():
        if art_id in pred_map:
            true_labels.append(gold_campaign)
            pred_labels.append(pred_map[art_id])

    return true_labels, pred_labels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _clustering_metrics(true_labels: list, pred_labels: list) -> dict[str, float]:
    try:
        from sklearn.metrics import (
            adjusted_rand_score,
            normalized_mutual_info_score,
            v_measure_score,
            homogeneity_score,
            completeness_score,
        )
        return {
            "ari":          float(adjusted_rand_score(true_labels, pred_labels)),
            "nmi":          float(normalized_mutual_info_score(true_labels, pred_labels)),
            "v_measure":    float(v_measure_score(true_labels, pred_labels)),
            "homogeneity":  float(homogeneity_score(true_labels, pred_labels)),
            "completeness": float(completeness_score(true_labels, pred_labels)),
        }
    except ImportError:
        logger.error("[eval_camp] sklearn not installed; cannot compute metrics")
        return {}


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _style(v: float) -> str:
    return "bold green" if v >= 0.50 else ("yellow" if v >= 0.25 else "red")


def _print_results(metrics: dict, n_articles: int, n_gold: int,
                   n_pred: int) -> None:
    console.print()
    console.rule("[bold cyan]Campaign Extraction — Clustering Metrics[/bold cyan]")
    console.print(
        f"  [dim]articles evaluated={n_articles}  "
        f"gold campaigns={n_gold}  "
        f"predicted campaigns={n_pred}[/dim]")
    console.print()
    if not metrics:
        console.print("[red]No metrics computed (sklearn missing?).[/red]")
        return
    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold white")
    t.add_column("Metric", style="bold", min_width=16)
    t.add_column("Value", justify="right", min_width=10)
    t.add_row("ARI",
              f"[{_style(metrics['ari'])}]{metrics['ari']:.4f}[/]")
    t.add_row("NMI",
              f"[{_style(metrics['nmi'])}]{metrics['nmi']:.4f}[/]")
    t.add_row("V-measure",
              f"[{_style(metrics['v_measure'])}]{metrics['v_measure']:.4f}[/]")
    t.add_row("  Homogeneity",
              f"[dim]{metrics['homogeneity']:.4f}[/dim]")
    t.add_row("  Completeness",
              f"[dim]{metrics['completeness']:.4f}[/dim]")
    console.print(t)
    console.print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(cfg=None) -> None:
    from config import Config
    cfg = cfg or Config.load()

    kb_root = Path("knowledge")
    kb = KnowledgeBase(kb_root)

    # FakeCTI ground truth (per-article campaign label)
    gt_path = kb_root / "ground_truth" / "annotations.json"
    if not gt_path.exists():
        # Try FakeCTI-specific ground truth path
        from core.knowledge_base import DATASET_FAKECTI as FAKECTI_SLUG
        fakecti_gt = kb_root / "fakecti" / "ground_truth" / "annotations.json"
        if fakecti_gt.exists():
            gt_path = fakecti_gt
        else:
            console.print(
                f"[red]FakeCTI ground truth not found.[/red] "
                "Run the FakeCTI converter first:\n"
                "  python core/converters/fakecti.py")
            return

    gt_by_article: dict[str, str] = json.loads(
        gt_path.read_text(encoding="utf-8"))

    # Determine which backends to check
    camp_det = getattr(cfg, "camp_detector", "models/xlm-multicw")
    backends_to_check = [
        "dense", "bm25-rag", "bm25_rag", "specfi-cs", "cspecfi", "context-1",
        getattr(cfg, "camp_extractor", "dense"),
    ]
    backends_to_check = list(dict.fromkeys(backends_to_check))  # unique, ordered

    true_labels, pred_labels = _build_assignment_arrays(
        kb, DATASET_FAKECTI, gt_by_article, backends_to_check)

    if not true_labels:
        console.print(
            "[yellow]No articles found in both ground truth and extracted "
            "campaigns.[/yellow]\n"
            "Run Campaigns → Generate (with FakeCTI data) first.")
        return

    n_gold = len(set(true_labels))
    n_pred = len(set(pred_labels))
    metrics = _clustering_metrics(true_labels, pred_labels)

    _print_results(metrics, len(true_labels), n_gold, n_pred)

    # Per-gold-campaign coverage report
    coverage: dict[str, dict] = {}
    for art, gold, pred in zip(
        [None] * len(true_labels), true_labels, pred_labels
    ):
        cov = coverage.setdefault(gold, {"pred_camps": set(), "count": 0})
        cov["pred_camps"].add(pred)
        cov["count"] += 1
    console.print("[dim]Per-campaign coverage (gold → # matched articles):[/dim]")
    for gold_name in sorted(coverage, key=lambda x: -coverage[x]["count"]):
        cov = coverage[gold_name]
        console.print(
            f"  {gold_name[:50]}  n={cov['count']}  "
            f"pred_camps={len(cov['pred_camps'])}")

    # Save CSV
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "eval_campaigns.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, v])
        w.writerow(["n_articles", len(true_labels)])
        w.writerow(["n_gold_campaigns", n_gold])
        w.writerow(["n_pred_campaigns", n_pred])

    # Stats
    try:
        from core.ui.stats import save_eval_stats
        save_eval_stats(
            "campaigns",
            param_key=getattr(cfg, "camp_extractor", "dense"),
            params={"extractor": getattr(cfg, "camp_extractor", "dense")},
            scores={"ari":       metrics.get("ari", 0),
                    "nmi":       metrics.get("nmi", 0),
                    "v_measure": metrics.get("v_measure", 0),
                    "n":         len(true_labels)},
        )
    except Exception:
        pass

    from evaluation.report_paths import report_path
    html_out = report_path("campaigns", dataset="fake-cti",
                           method=getattr(cfg, "camp_extractor", None))
    console.save_html(str(html_out), theme=MONOKAI, clear=False)
    console.print(f"[dim]Results → {csv_path}  HTML → {html_out}[/dim]")
