"""Dataset compilation: full pipeline over MassiveSumm SK/CZ → CSV export.

Runs the complete DisTraceAI pipeline on the Slovak and Czech subset of
MassiveSumm and exports the resulting hierarchy as three CSV files under
``knowledge/dataset/``:

  subnarratives.csv   — id, narrative_id, campaign_id, central_claim,
                        claims (pipe-separated), detector, language,
                        veracity, veracity_confidence
  narratives.csv      — id, campaign_id, central_claim, backend, dataset,
                        languages, veracity, veracity_confidence, member_count
  campaigns.csv       — id, label, central_claim, backend, dataset,
                        languages, veracity, veracity_confidence,
                        coordination_score, n1_burst, n2_coamp, n3_reuse,
                        n4_crosslingual, member_count

The original MassiveSumm article texts are NOT included — only the
extracted/synthesised hierarchy content.  Referential integrity is maintained
via FK columns (sub-narrative → narrative → campaign).
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

from rich.console import Console

from core.knowledge_base import KnowledgeBase

logger  = logging.getLogger(__name__)
console = Console()

_MASSIVESUMM_DATA = Path("data/MassiveSumm")
_DATASET_SLUG     = "massivesumm"
_OUTPUT_DIR       = Path("knowledge/dataset")


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(kb: KnowledgeBase, cfg) -> dict:
    """Run each pipeline step over the MassiveSumm dataset, skipping
    already-processed articles (all steps are idempotent)."""

    console.print("[bold]Step 1/5:[/bold] Claim detection…")
    from generation.gen_cw_detect import generate as gen_cw
    gen_cw(cfg.detector, kb)

    console.print("[bold]Step 2/5:[/bold] Canonization…")
    from generation.gen_canonize import canonize
    canonize(cfg.canon_detector, cfg.canon_generator, kb)

    console.print("[bold]Step 3/5:[/bold] Sub-narrative extraction…")
    from generation.gen_sub_narratives import generate as gen_sn
    gen_sn(
        detector_path=cfg.subnar_detector,
        embedder_name=cfg.subnar_embedder,
        generator_key=cfg.subnar_generator,
        kb=kb,
        min_similarity=cfg.subnar_min_similarity,
        min_claims=cfg.subnar_min_claims,
    )

    console.print("[bold]Step 4/5:[/bold] Narrative extraction…")
    from generation.gen_narratives import generate as gen_nar
    gen_nar(
        detector_path=cfg.nar_detector,
        extractor=cfg.nar_extractor,
        embedder_name=cfg.nar_embedder,
        generator_key=cfg.nar_generator,
        kb=kb,
        cfg=cfg,
    )

    console.print("[bold]Step 5/5:[/bold] Campaign extraction…")
    from generation.gen_campaigns import generate as gen_camp
    gen_camp(
        dataset=_DATASET_SLUG,
        detector_path=cfg.camp_detector,
        extractor=cfg.camp_extractor,
        embedder_name=cfg.camp_embedder,
        generator_key=cfg.camp_generator,
        kb=kb,
        cfg=cfg,
    )
    return {}


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def _build_campaign_lookup(kb: KnowledgeBase) -> dict[str, str]:
    """Build narrative_id → campaign_id lookup across all backends."""
    nar_to_camp: dict[str, str] = {}
    for backend in ("dense", "bm25-rag", "bm25_rag",
                    "specfi-cs", "cspecfi", "context-1"):
        for camp in kb.campaigns(_DATASET_SLUG, backend):
            for nar_id in camp.narratives:
                nar_to_camp[nar_id] = camp.id
    return nar_to_camp


def _build_sub_to_nar_lookup(kb: KnowledgeBase, det_slug: str) -> dict[str, str]:
    """Build sub_narrative_id → narrative_id lookup."""
    sn_to_nar: dict[str, str] = {}
    for backend in ("dense", "bm25-rag", "bm25_rag",
                    "specfi-cs", "cspecfi", "context-1"):
        for nar in kb.narratives(_DATASET_SLUG, backend):
            for sn_id in nar.sub_narratives:
                sn_to_nar[sn_id] = nar.id
    return sn_to_nar


def export_csvs(kb: KnowledgeBase, cfg, out_dir: Path) -> dict:
    """Export the hierarchy to three CSV files; return row counts."""
    out_dir.mkdir(parents=True, exist_ok=True)

    import os
    det_path = getattr(cfg, "camp_detector", getattr(cfg, "nar_detector",
                                                      "models/xlm-multicw"))
    if det_path == "both":
        det_slugs = ["xlm-multicw", "mdb-multicw"]
    else:
        det_slugs = [os.path.basename(det_path.rstrip("/\\"))]

    nar_to_camp = _build_campaign_lookup(kb)
    counts = {}

    # ---- sub-narratives ----
    sn_path = out_dir / "subnarratives.csv"
    n_sn = 0
    with sn_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "narrative_id", "campaign_id", "central_claim",
                    "claims", "detector", "language", "veracity",
                    "veracity_confidence"])
        for det_slug in det_slugs:
            sn_to_nar = _build_sub_to_nar_lookup(kb, det_slug)
            for sn in kb.sub_narratives(_DATASET_SLUG, det_slug):
                nar_id  = sn_to_nar.get(sn.id, "")
                camp_id = nar_to_camp.get(nar_id, "")
                # Infer language from article claims metadata
                lang = ""
                ac = kb.load_article_claims(_DATASET_SLUG, det_slug,
                                            sn.article_name)
                if ac:
                    lang = (ac.metadata or {}).get("source_language", "")
                w.writerow([sn.id, nar_id, camp_id, sn.central_claim,
                            " | ".join(sn.claims), det_slug, lang,
                            sn.veracity, sn.veracity_confidence])
                n_sn += 1
    counts["sub_narratives"] = n_sn
    console.print(f"  subnarratives.csv: {n_sn} rows")

    # ---- narratives ----
    nar_path = out_dir / "narratives.csv"
    n_nar = 0
    with nar_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "campaign_id", "central_claim", "backend", "dataset",
                    "languages", "veracity", "veracity_confidence",
                    "member_count"])
        for backend in ("dense", "bm25-rag", "bm25_rag",
                        "specfi-cs", "cspecfi", "context-1"):
            for nar in kb.narratives(_DATASET_SLUG, backend):
                camp_id = nar_to_camp.get(nar.id, "")
                w.writerow([nar.id, camp_id, nar.central_claim,
                            nar.backend, nar.dataset,
                            "|".join(nar.languages),
                            nar.veracity, nar.veracity_confidence,
                            nar.member_count])
                n_nar += 1
    # deduplicate (same nar may be written by multiple backend passes)
    counts["narratives"] = n_nar
    console.print(f"  narratives.csv: {n_nar} rows")

    # ---- campaigns ----
    camp_path = out_dir / "campaigns.csv"
    n_camp = 0
    with camp_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "central_claim", "backend", "dataset",
                    "languages", "veracity", "veracity_confidence",
                    "coordination_score", "n1_burst", "n2_coamp",
                    "n3_reuse", "n4_crosslingual", "member_count"])
        for backend in ("dense", "bm25-rag", "bm25_rag",
                        "specfi-cs", "cspecfi", "context-1"):
            for camp in kb.campaigns(_DATASET_SLUG, backend):
                coord = camp.coordination or {}
                w.writerow([
                    camp.id, camp.label, camp.central_claim,
                    camp.backend, camp.dataset,
                    "|".join(camp.languages),
                    camp.veracity, camp.veracity_confidence,
                    camp.coordination_score,
                    coord.get("n1_burst", ""),
                    coord.get("n2_coamp", ""),
                    coord.get("n3_reuse", ""),
                    coord.get("n4_crosslingual", ""),
                    camp.member_count,
                ])
                n_camp += 1
    counts["campaigns"] = n_camp
    console.print(f"  campaigns.csv: {n_camp} rows")

    console.print(
        f"\n[bold]Dataset saved to[/bold] [cyan]{out_dir}[/cyan]")
    return counts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_dataset(cfg=None) -> dict:
    """Convert MassiveSumm SK/CZ, run the full pipeline, export CSVs."""
    from config import Config
    cfg = cfg or Config.load()

    kb_root = Path("knowledge")
    kb = KnowledgeBase(kb_root)

    # 1) Convert MassiveSumm → KB articles
    if not _MASSIVESUMM_DATA.exists():
        console.print(
            f"[red]MassiveSumm data not found at {_MASSIVESUMM_DATA}.[/red]\n"
            "Place the SK/CZ subset under data/MassiveSumm/.")
        return {}

    console.print(f"\n[bold cyan]Generate Dataset — MassiveSumm SK/CZ[/bold cyan]")
    console.print(
        f"[bold]Converting MassiveSumm[/bold] [cyan]{_MASSIVESUMM_DATA}[/cyan]…")
    from core.converters.massivesumm import convert
    n_articles = convert(_MASSIVESUMM_DATA, kb_root)
    console.print(f"  {n_articles} articles loaded into KB")

    # 2) Run pipeline
    console.print("\n[bold]Running pipeline…[/bold]")
    run_pipeline(kb, cfg)

    # 3) Export CSVs
    console.print("\n[bold]Exporting dataset CSVs…[/bold]")
    counts = export_csvs(kb, cfg, _OUTPUT_DIR)

    return {"articles": n_articles, **counts}
