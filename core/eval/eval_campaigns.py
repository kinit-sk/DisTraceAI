"""Campaign extraction evaluation against FakeCTI ground truth.

Ground truth: ``knowledge/fakecti/ground_truth/annotations.json``
(written by the FakeCTI converter) maps each article ID to its gold
campaign name.

Evaluation unit: per-article campaign assignment. For each article in the
intersection of (extracted campaigns) and (FakeCTI ground truth), we build:
  * true_labels  — gold campaign name per article
  * pred_labels  — extracted campaign ID per article

Metrics: Adjusted Rand Index (ARI), Normalised Mutual Information (NMI),
V-measure (homogeneity/completeness/V). These are standard cluster-quality
metrics that do not require label alignment between predicted and gold clusters.

The article→campaign mapping is traced via:
    campaign.narratives → narrative.sub_narratives → sub_narrative.article_name

Auto-bootstrap: if any upstream artefact is missing (raw FakeCTI not yet
converted, no CW claims, no canonized claims, no sub-narratives, no
narratives, no campaigns), the evaluation entry point runs the missing
stages itself. Every stage is idempotent at the dataset level — already-
processed datasets are skipped automatically — so the bootstrap is cheap
when only a tail of the pipeline is missing.
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

# Backend names that may have produced narratives/campaigns. Kept in one
# place so the trace/index/scan loops below stay in sync. (`bm25-rag` and
# `bm25_rag` are both listed because earlier runs alternated spellings.)
_KNOWN_BACKENDS = (
    "dense", "bm25-rag", "bm25_rag",
    "specfi-cs", "specfi-ccs", "cspecfi", "context-1",
)
_KNOWN_DETECTORS = ("xlm-multicw", "mdb-multicw")


def _trace_campaign_articles(camp, kb: KnowledgeBase, dataset: str,
                             *, sn_to_article: dict[str, str]) -> list[str]:
    """Return all article_names reachable from a campaign via the hierarchy.

    Uses the precomputed ``sn_to_article`` map (built once per evaluation
    run, see ``_build_assignment_arrays``) so the per-sub-narrative lookup
    is O(1) instead of O(M·detectors). Narrative lookup uses
    ``kb.narrative_by_id`` to avoid scanning the whole narratives directory
    once per backend per call.
    """
    articles: list[str] = []
    for nar_id in camp.narratives:
        nar = None
        for backend in _KNOWN_BACKENDS:
            nar = kb.narrative_by_id(dataset, backend, nar_id)
            if nar is not None:
                break
        if nar is None:
            continue
        for sn_id in nar.sub_narratives:
            art = sn_to_article.get(sn_id)
            if art is not None:
                articles.append(art)
    return articles


def _build_assignment_arrays(kb: KnowledgeBase, dataset: str,
                              gt_by_article: dict[str, str],
                              backends: list[str]) -> tuple[list, list]:
    """Build aligned (true_labels, pred_labels) for all traceable articles."""
    # Build the sub-narrative → article map ONCE across all detectors. Before
    # this refactor, _trace_campaign_articles re-loaded sub_narratives(...)
    # for every (campaign, narrative, sn_id) triple — O(N·M·K) JSON reads.
    sn_to_article: dict[str, str] = {}
    for det in _KNOWN_DETECTORS:
        for sn in kb.sub_narratives(dataset, det):
            sn_to_article.setdefault(sn.id, sn.article_name)

    # Collect campaigns across all backends.
    all_camps = []
    for backend in backends:
        all_camps += kb.campaigns(dataset, backend)

    # Build article → predicted campaign mapping.
    pred_map: dict[str, str] = {}
    for camp in all_camps:
        for art in _trace_campaign_articles(camp, kb, dataset,
                                            sn_to_article=sn_to_article):
            pred_map[art] = camp.id

    # Intersect with ground truth.
    true_labels, pred_labels = [], []
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
# Auto-bootstrap
# ---------------------------------------------------------------------------

def _detector_slug(detector_path: str) -> str:
    """Map cfg.*_detector ('models/mdb-multicw') to the on-disk slug ('mdb-multicw')."""
    return Path(detector_path).name


def _has_canonized(kb, dataset: str, det_slug: str) -> bool:
    """True iff at least one ArticleClaims for (dataset, detector) has at
    least one canonized claim."""
    try:
        records = kb.all_article_claims(dataset, det_slug)
    except Exception:
        return False
    return any(getattr(r, "canonized_claims", None) for r in records)


def _has_cw_claims(kb, dataset: str, det_slug: str) -> bool:
    """True iff at least one ArticleClaims for (dataset, detector) exists."""
    try:
        return bool(kb.all_article_claims(dataset, det_slug))
    except Exception:
        return False


def _ensure_fakecti_ready(cfg, console) -> "tuple[bool, object]":
    """Run any pipeline stage whose FakeCTI output is still missing.

    Returns (ready, kb). ``ready=False`` means the raw FakeCTI CSV could not
    be located and no further work is possible; the caller should surface
    that to the user. Otherwise ``kb`` is a fresh ``KnowledgeBase`` pointing
    at the populated tree.
    """
    from core.knowledge_base import KnowledgeBase, DATASET_FAKECTI

    kb_root = Path("knowledge")
    kb = KnowledgeBase(kb_root)

    # ---- 0. FakeCTI ground truth (cheap — pure CSV transform, no LLM) ----
    gt_path = kb_root / "fakecti" / "ground_truth" / "annotations.json"
    if not gt_path.exists():
        console.print(
            "[yellow]FakeCTI ground truth not found — running the converter "
            "automatically…[/yellow]")
        try:
            from core.converters.fakecti import convert as _fakecti_convert
            # Pass the canonical layout; the converter falls back to the flat
            # layout when needed (see _resolve_src in core/converters/fakecti.py).
            _fakecti_convert(Path("data/FakeCTI/FakeCTI.csv"),
                             kb_root / "fakecti")
        except FileNotFoundError as exc:
            console.print(f"[red]Cannot bootstrap FakeCTI:[/red] {exc}")
            return False, None
        except Exception as exc:
            console.print(f"[red]FakeCTI converter failed: {exc}[/red]")
            return False, None
        # Reload — the converter wrote new files.
        kb = KnowledgeBase(kb_root)

    # ---- 1. Articles + CW claims (claim-detection step) -------------------
    # gen_cw_detect.generate ingests FakeCTI articles AND extracts CW claims
    # in one pass, so we use the CW-claims presence as the readiness signal.
    det_slug = _detector_slug(cfg.detector)
    if not _has_cw_claims(kb, DATASET_FAKECTI, det_slug):
        console.print(
            "[yellow]No FakeCTI check-worthy claims yet — running "
            "claim-detection automatically…[/yellow]")
        from main import run_generate
        run_generate("claim-detection", cfg)
        kb = KnowledgeBase(kb_root)

    # ---- 2. Canonized claims ---------------------------------------------
    canon_det_slug = _detector_slug(cfg.canon_detector)
    if not _has_canonized(kb, DATASET_FAKECTI, canon_det_slug):
        console.print(
            "[yellow]No FakeCTI canonized claims yet — running canonization "
            "automatically…[/yellow]")
        from main import run_generate
        run_generate("claim-canonization", cfg)
        kb = KnowledgeBase(kb_root)

    # ---- 3. Sub-narratives -----------------------------------------------
    subnar_det_slug = _detector_slug(cfg.subnar_detector)
    if not kb.sub_narratives(DATASET_FAKECTI, subnar_det_slug):
        console.print(
            "[yellow]No FakeCTI sub-narratives yet — running sub-narrative "
            "extraction automatically…[/yellow]")
        from main import run_generate
        run_generate("sub-narratives", cfg)
        kb = KnowledgeBase(kb_root)

    # ---- 4. Narratives ---------------------------------------------------
    nar_backend = cfg.nar_extractor.replace("-", "_")  # e.g. bm25-rag → bm25_rag
    if not kb.narratives(DATASET_FAKECTI, nar_backend):
        console.print(
            "[yellow]No FakeCTI narratives yet — running narrative "
            "extraction automatically…[/yellow]")
        from main import run_generate
        run_generate("narratives", cfg)
        kb = KnowledgeBase(kb_root)

    # ---- 5. Campaigns ----------------------------------------------------
    # The evaluation checks every known backend below; we only need ONE of
    # them populated to score the run. Use the user's active method as the
    # bootstrap target.
    camp_backend = cfg.camp_extractor.replace("-", "_")
    if not kb.campaigns(DATASET_FAKECTI, camp_backend):
        console.print(
            "[yellow]No FakeCTI campaigns yet — running campaign extraction "
            "automatically…[/yellow]")
        from main import run_generate
        run_generate("campaigns", cfg)
        kb = KnowledgeBase(kb_root)

    return True, kb


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(cfg=None) -> None:
    from config import Config
    cfg = cfg or Config.load()

    # Auto-bootstrap every missing FakeCTI pipeline output. Each stage is
    # idempotent and only runs when its dataset-level output is absent, so
    # repeated evaluations after a successful first bootstrap return here
    # within milliseconds.
    ready, kb = _ensure_fakecti_ready(cfg, console)
    if not ready:
        console.print(
            "[red]Auto-bootstrap aborted — the raw FakeCTI CSV is missing.[/red]\n"
            "Place FakeCTI.csv at data/FakeCTI/FakeCTI.csv (or data/FakeCTI.csv "
            "as a fallback) and re-run.")
        return

    kb_root = Path("knowledge")

    # FakeCTI ground truth (per-article campaign label)
    gt_path = kb_root / "ground_truth" / "annotations.json"
    if not gt_path.exists():
        # Try FakeCTI-specific ground truth path (written by the converter)
        from core.knowledge_base import DATASET_FAKECTI as FAKECTI_SLUG
        fakecti_gt = kb_root / "fakecti" / "ground_truth" / "annotations.json"
        if fakecti_gt.exists():
            gt_path = fakecti_gt
        else:
            # Should not happen after _ensure_fakecti_ready, but kept as a
            # last-line guard so the failure mode is explicit if some other
            # path-resolution issue is at play.
            console.print(
                "[red]FakeCTI ground truth still not found after auto-bootstrap.[/red]\n"
                f"Expected at: {fakecti_gt}\n"
                "This usually means the converter ran but produced an empty "
                "ground-truth file (the WEB subset of FakeCTI is empty or the "
                "min_campaign_size filter dropped every campaign).")
            return

    gt_by_article: dict[str, str] = json.loads(
        gt_path.read_text(encoding="utf-8"))

    # Determine which backends to check. Keep this in sync with
    # _KNOWN_BACKENDS so a campaign produced by any retrieval method is
    # evaluated. Active method goes last so dict.fromkeys preserves it but
    # never duplicates it.
    backends_to_check = list(_KNOWN_BACKENDS) + [
        getattr(cfg, "camp_extractor", "dense")
    ]
    backends_to_check = list(dict.fromkeys(backends_to_check))  # unique, ordered

    true_labels, pred_labels = _build_assignment_arrays(
        kb, DATASET_FAKECTI, gt_by_article, backends_to_check)

    if not true_labels:
        console.print(
            "[yellow]No articles overlap between FakeCTI ground truth and the "
            "extracted campaigns.[/yellow]\n"
            "The full pipeline ran via auto-bootstrap, but the article→campaign "
            "trace produced an empty intersection. Likely causes:\n"
            "  • the active retrieval backend produced campaigns under a name "
            f"that isn't in {backends_to_check}\n"
            "  • the FakeCTI WEB subset is empty for the configured "
            "min_campaign_size filter (see FAKECTI_MIN_CAMPAIGN env var)\n"
            "  • article IDs were normalised differently between the converter "
            "and the cw_detect ingestion path")
        return

    n_gold = len(set(true_labels))
    n_pred = len(set(pred_labels))
    metrics = _clustering_metrics(true_labels, pred_labels)

    _print_results(metrics, len(true_labels), n_gold, n_pred)

    # Per-gold-campaign coverage report.
    # (Previous version zipped a synthetic `[None] * len(true_labels)` as the
    # article slot — the variable was never used; dropped entirely.)
    coverage: dict[str, dict] = {}
    for gold, pred in zip(true_labels, pred_labels):
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
    except Exception as exc:
        logger.warning("[eval_camp] save_eval_stats failed: %s", exc)

    from core.eval.report_paths import report_path
    html_out = report_path("campaigns", dataset="fake-cti",
                           method=getattr(cfg, "camp_extractor", None))
    console.save_html(str(html_out), theme=MONOKAI, clear=False)
    console.print(f"[dim]Results → {csv_path}  HTML → {html_out}[/dim]")
