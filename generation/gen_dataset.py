"""Dataset compilation: full pipeline over EUvsDisinfo → CSV export.

Runs the complete DisTraceAI pipeline on the EUvsDisinfo dataset (Leite et al.,
CIKM '24) and exports the resulting hierarchy as three CSV files under
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

The original EUvsDisinfo article texts are NOT included — only the
extracted/synthesised hierarchy content.  Referential integrity is maintained
via FK columns (sub-narrative → narrative → campaign).

Pipeline orchestration
----------------------
This module reads EUvsDisinfo articles via the EUvsDisinfo converter, then runs
the same five pipeline steps as the regular Generate flow — but TARGETED at the
``euvsdisinfo`` dataset slug only (the per-step ``generate()`` entry points
hard-code PolyNarrative + FakeCTI iteration, so we bypass them and call each
step's ``_process_dataset`` helper directly with the EUvsDisinfo slug).

The narrative step uses the SAME backend as Narrative detection
(``cfg.nar_extractor`` / ``cfg.nar_embedder`` / ``cfg.nar_generator``), so the
hierarchy produced here is methodologically consistent with the Narrative
evaluation run.
"""
from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from core.knowledge_base import KnowledgeBase

logger  = logging.getLogger(__name__)
console = Console()

# Dataset slug + on-disk source. EUvsDisinfo replaces the earlier MassiveSumm
# SK/CZ scope: it ships real publication dates and a multilingual corpus, both
# of which make the N1 (burst) and N4 (cross-lingual) coordination signals
# meaningful here.
_EUVSDISINFO_DATA = Path("data/EUvsDisinfo")
_DATASET_SLUG     = "euvsdisinfo"
_OUTPUT_DIR       = Path("knowledge/dataset")


def _resolve_data_dir(canonical: Path) -> Path | None:
    """Return the EUvsDisinfo data dir, tolerating filesystem case variants.

    Linux is case-sensitive, so a directory named ``data/Euvsdisinfo/`` (or any
    other casing of ``EUvsDisinfo``) will not match ``Path("data/EUvsDisinfo")``
    directly. We try the canonical path first; on miss we scan the parent dir
    for any entry whose lowercased name matches the canonical name, so common
    casings (``EUvsDisinfo``, ``Euvsdisinfo``, ``euvsdisinfo``, …) all resolve.
    Returns None if nothing matches.
    """
    if canonical.is_dir():
        return canonical
    parent = canonical.parent
    if not parent.is_dir():
        return None
    target = canonical.name.lower()
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == target:
            return child
    return None


# ---------------------------------------------------------------------------
# KB-backed article iterator (consumed by gen_cw_detect._process_dataset)
# ---------------------------------------------------------------------------

def _articles_from_kb(kb: KnowledgeBase, dataset_slug: str = _DATASET_SLUG):
    """Yield ``(article_name, text, source_path, meta)`` for each KB article.

    Built once the EUvsDisinfo converter has populated
    ``knowledge/<dataset_slug>/_articles/``.  Mirrors the tuple shape produced
    by ``_polynarrative_articles`` / ``_fakecti_articles`` in
    ``gen_cw_detect`` so the same ``_process_dataset`` consumer works
    unchanged.

    The article's KB id (already prefixed ``article_``) is reused as the
    per-article filename so canonization / sub-narratives / narratives can
    re-find the same record across pipeline steps.
    """
    for art in kb.articles(dataset_slug):
        text = (art.content or "").strip()
        if not text:
            continue
        article_name = art.id   # already prefixed 'article_' by the converter
        source_path  = art.url or f"{dataset_slug}://{art.id}"
        meta = {
            "title":  art.title or "",
            "author": art.author,
            "metadata": {
                "source_domain":   art.source_domain or "",
                "source_language": art.source_language or "UND",
                "published_at":    art.published_at,
            },
        }
        yield article_name, text, source_path, meta


# ---------------------------------------------------------------------------
# Pipeline runner — orchestrates per-step processing for euvsdisinfo only
# ---------------------------------------------------------------------------

def run_pipeline(kb: KnowledgeBase, cfg) -> dict:
    """Run each pipeline step over EUvsDisinfo, skipping already-processed
    articles (all steps are idempotent).

    We bypass the public ``generate()`` entry points (which hard-code
    PolyNarrative+FakeCTI iteration) and call each step's ``_process_dataset``
    helper directly, scoped to ``_DATASET_SLUG``.

    All LLM generators are loaded at bf16 regardless of the config precision,
    because AWQ-quantised weights are not required for the dataset pipeline and
    may not be present on every deployment.
    """
    from core.claims.cw_detector import CheckWorthinessDetector
    from core.claims.gen_cw_detect import _process_dataset as cw_process
    from core.claims.gen_canonize import _process_dataset as canon_process
    from core.claims.gen_canonize import _detector_slug as _canon_det_slug
    from core.claims.gen_sub_narratives import _process_dataset as sn_process
    from core.claims.gen_narratives import _process_dataset as nar_process
    from core.claims.gen_campaigns import generate as gen_camp_entry
    from core.models import make_embedder, make_generator, close_generator

    # AWQ weights may not be present; always use bf16 for this pipeline.
    _PREC = "bf16"

    # Downstream helpers (gen_campaigns, gen_narratives) pull precision
    # directly from cfg for NodeRAG build paths. Shadow all precision fields
    # on a lightweight wrapper so bf16 is used consistently.
    import copy as _copy
    cfg = _copy.copy(cfg)
    cfg.canon_precision  = _PREC
    cfg.subnar_precision = _PREC
    cfg.nar_precision    = _PREC
    cfg.camp_precision   = _PREC

    # ---- Step 1: check-worthy claim detection -----------------------------
    console.print("[bold]Step 1/5:[/bold] Claim detection…")
    # IMPORTANT: gen_cw_detect.generate() expects a CheckWorthinessDetector
    # instance, not a path string. The earlier wiring passed cfg.detector
    # (a str) directly and tripped detector.slug → AttributeError.
    detector = CheckWorthinessDetector(cfg.detector)
    cw_process(
        _DATASET_SLUG,
        _articles_from_kb(kb, _DATASET_SLUG),
        detector, kb,
    )

    # ---- Step 2: canonization --------------------------------------------
    console.print("[bold]Step 2/5:[/bold] Canonization…")
    canon_llm = make_generator(cfg.canon_generator, _PREC)
    canon_process(_DATASET_SLUG, _canon_det_slug(cfg.canon_detector),
                  canon_llm, kb)
    close_generator(canon_llm)

    # ---- Step 3: sub-narrative extraction --------------------------------
    console.print("[bold]Step 3/5:[/bold] Sub-narrative extraction…")
    sn_embedder = make_embedder(cfg.subnar_embedder)
    sn_llm      = make_generator(cfg.subnar_generator, _PREC)
    sn_process(
        _DATASET_SLUG, _canon_det_slug(cfg.subnar_detector),
        sn_embedder, sn_llm, kb,
        cfg.subnar_min_similarity, cfg.subnar_min_claims,
    )
    close_generator(sn_llm)

    # ---- Step 4: narrative extraction (uses the Narrative-detection backend)
    console.print(
        f"[bold]Step 4/5:[/bold] Narrative extraction "
        f"(backend={cfg.nar_extractor})…")
    nar_embedder = make_embedder(cfg.nar_embedder)
    nar_llm = None
    if os.getenv("DISTRACE_NAR_NO_LLM") != "1":
        nar_llm = make_generator(cfg.nar_generator, _PREC)
    elif cfg.nar_extractor != "dense":
        raise RuntimeError(
            f"nar_extractor={cfg.nar_extractor!r} requires a generator, but "
            "DISTRACE_NAR_NO_LLM=1 is set.")
    index_root = os.getenv(
        "DISTRACE_NAR_NODERAG_INDEX_ROOT",
        str(Path("knowledge") / "noderag"))
    nar_process(
        _DATASET_SLUG, _canon_det_slug(cfg.nar_detector), cfg.nar_extractor,
        nar_embedder, nar_llm, kb, cfg, index_root,
    )
    close_generator(nar_llm)

    # ---- Step 5: campaign extraction -------------------------------------
    console.print("[bold]Step 5/5:[/bold] Campaign extraction…")
    gen_camp_entry(
        dataset=_DATASET_SLUG,
        detector_path=cfg.camp_detector,
        extractor=cfg.camp_extractor,
        embedder_name=cfg.camp_embedder,
        generator_key=cfg.camp_generator,
        precision=_PREC,
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
    """Convert EUvsDisinfo, run the full pipeline, export CSVs."""
    from config import Config
    cfg = cfg or Config.load()

    kb_root = Path("knowledge")
    kb = KnowledgeBase(kb_root)

    # 1) Convert EUvsDisinfo → KB articles
    data_dir = _resolve_data_dir(_EUVSDISINFO_DATA)
    if data_dir is None:
        console.print(
            f"[red]EUvsDisinfo data not found at {_EUVSDISINFO_DATA} "
            f"(or any case variant under {_EUVSDISINFO_DATA.parent}/).[/red]\n"
            "Place a EUvsDisinfo CSV (reconstructed with article text) "
            "under data/EUvsDisinfo/.")
        return {}

    console.print(f"\n[bold cyan]Generate Dataset — EUvsDisinfo[/bold cyan]")
    console.print(
        f"[bold]Converting EUvsDisinfo[/bold] [cyan]{data_dir}[/cyan]…")
    from core.converters.euvsdisinfo import convert
    n_articles = convert(data_dir, kb_root)
    console.print(f"  {n_articles} articles loaded into KB")

    if n_articles == 0:
        console.print(
            "[red]No EUvsDisinfo articles converted — aborting pipeline.[/red]\n"
            "Check that the CSV under data/EUvsDisinfo/ has a body column "
            "(article_text / text / content / body) — the public "
            "euvsdisinfo_base.csv is URL-only and must be reconstructed first.")
        return {"articles": 0}

    # 2) Run pipeline
    console.print("\n[bold]Running pipeline…[/bold]")
    run_pipeline(kb, cfg)

    # 3) Export CSVs
    console.print("\n[bold]Exporting dataset CSVs…[/bold]")
    counts = export_csvs(kb, cfg, _OUTPUT_DIR)

    return {"articles": n_articles, **counts}