"""Dataset compilation: full pipeline over EUvsDisinfo → CSV export.

Runs the complete DisTraceAI pipeline on the EUvsDisinfo dataset (Leite et al.,
CIKM '24) and exports the resulting hierarchy as three CSV files under
``knowledge/dataset/``:

  subnarratives.csv   — id, narrative_id, campaign_id, central_claim,
                        claims (pipe-separated), detector, language,
                        veracity, veracity_confidence
  narratives.csv      — id, campaign_id, central_claim, llm_backends, dataset,
                        languages, veracity, veracity_confidence, member_count
  campaigns.csv       — id, label, central_claim, llm_backends, dataset,
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

The narrative step uses the SAME llm_backends as Narrative detection
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
# Plan §4.6: "stored in results/EUDisinfoAtlas in campaigns.csv,
# narratives.csv and sub-narratives.csv". Previously knowledge/dataset/ —
# functionally fine but off the plan-mandated path.
_OUTPUT_DIR       = Path("results/EUDisinfoAtlas")
# CSV files produced by export_csvs (subnarratives.csv keeps the no-hyphen
# spelling matched in export_csvs below). The full triple must exist for
# camp_re_extract='off' to short-circuit Generate Dataset.
_OUTPUT_FILES     = ("campaigns.csv", "narratives.csv", "subnarratives.csv")


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

def _articles_from_kb(kb: KnowledgeBase, dataset_slug: str = _DATASET_SLUG,
                      *, limit: int | None = None):
    """Yield ``(article_name, text, source_path, meta)`` for each KB article.

    Built once the EUvsDisinfo converter has populated
    ``knowledge/<dataset_slug>/_articles/``.  Mirrors the tuple shape produced
    by ``_polynarrative_articles`` / ``_fakecti_articles`` in
    ``gen_cw_detect`` so the same ``_process_dataset`` consumer works
    unchanged.

    The article's KB id (already prefixed ``article_``) is reused as the
    per-article filename so canonization / sub-narratives / narratives can
    re-find the same record across pipeline steps.

    ``limit``: when truthy, caps the number of yielded articles. Used by
    Generate Dataset to honour ``camp_sample_size`` post-conversion when the
    converter itself didn't accept a limit (older signature).
    """
    yielded = 0
    for art in kb.articles(dataset_slug):
        if limit is not None and limit > 0 and yielded >= limit:
            break
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
        yielded += 1
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
    """
    from core.claims.cw_detector import CheckWorthinessDetector
    from core.gen.gen_cw_detect import _process_dataset as cw_process
    from core.gen.gen_canonize import _process_dataset as canon_process
    from core.gen.gen_canonize import _detector_slug as _canon_det_slug
    from core.gen.gen_sub_narratives import _process_dataset as sn_process
    from core.gen.gen_narratives import _process_dataset as nar_process
    from core.gen.gen_campaigns import generate as gen_camp_entry
    from core.models import make_embedder, make_generator, close_generator

    # Shadow cfg so the downstream helpers (gen_campaigns, gen_narratives)
    # cannot mutate the caller's config by accident. Precision shadowing —
    # which the old code did here — is gone: precision is now fixed by the
    # active backend (BF16 vLLM / Q8_0 llama-cpp) at the backend layer.
    import copy as _copy
    cfg = _copy.copy(cfg)

    # ---- Step 1: check-worthy claim detection -----------------------------
    console.print("[bold]Step 1/5:[/bold] Claim detection…")
    # IMPORTANT: gen_cw_detect.generate() expects a CheckWorthinessDetector
    # instance, not a path string. The earlier wiring passed cfg.detector
    # (a str) directly and tripped detector.slug → AttributeError.
    detector = CheckWorthinessDetector(cfg.detector)
    # camp_sample_size caps the article count post-conversion (plan §4.6).
    sample_limit = int(getattr(cfg, "camp_sample_size", 0) or 0) or None
    cw_process(
        _DATASET_SLUG,
        _articles_from_kb(kb, _DATASET_SLUG, limit=sample_limit),
        detector, kb,
    )

    # ---- Step 2: canonization --------------------------------------------
    console.print("[bold]Step 2/5:[/bold] Canonization…")
    canon_llm = make_generator(cfg.canon_generator)
    canon_process(_DATASET_SLUG, _canon_det_slug(cfg.canon_detector),
                  canon_llm, kb)
    close_generator(canon_llm)

    # ---- Step 3: sub-narrative extraction --------------------------------
    console.print("[bold]Step 3/5:[/bold] Sub-narrative extraction…")
    sn_embedder = make_embedder(cfg.subnar_embedder)
    sn_llm      = make_generator(cfg.subnar_generator)
    sn_process(
        _DATASET_SLUG, _canon_det_slug(cfg.subnar_detector),
        sn_embedder, sn_llm, kb,
        cfg.subnar_min_similarity, cfg.subnar_min_claims,
    )
    close_generator(sn_llm)
    close_generator(sn_embedder)  # free VRAM before Step 4 loads embedder+generator

    # ---- Step 4: narrative extraction (uses the Narrative-detection llm_backends)
    console.print(
        f"[bold]Step 4/5:[/bold] Narrative extraction "
        f"(llm_backends={cfg.nar_extractor})…")
    nar_embedder = make_embedder(cfg.nar_embedder)
    nar_llm = None
    if os.getenv("DISTRACE_NAR_NO_LLM") != "1":
        nar_llm = make_generator(cfg.nar_generator)
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
    close_generator(nar_embedder)  # free VRAM before Step 5

    # ---- Step 5: campaign extraction -------------------------------------
    console.print("[bold]Step 5/5:[/bold] Campaign extraction…")
    gen_camp_entry(
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

# Backend names whose narratives/campaigns should appear in the exported
# CSVs. Kept as a single tuple so adding a new retrieval method only requires
# one edit. `bm25-rag` and `bm25_rag` are both listed to cover earlier runs
# that used the underscored spelling.
_EXPORT_BACKENDS = (
    "dense", "bm25-rag", "bm25_rag",
    "specfi-cs", "specfi-ccs", "cspecfi", "context-1",
)


def _build_campaign_lookup(kb: KnowledgeBase) -> dict[str, str]:
    """Build narrative_id → campaign_id lookup across all backends."""
    nar_to_camp: dict[str, str] = {}
    for backend in _EXPORT_BACKENDS:
        for camp in kb.campaigns(_DATASET_SLUG, backend):
            for nar_id in camp.narratives:
                nar_to_camp[nar_id] = camp.id
    return nar_to_camp


def _build_sub_to_nar_lookup(kb: KnowledgeBase, det_slug: str) -> dict[str, str]:
    """Build sub_narrative_id → narrative_id lookup."""
    sn_to_nar: dict[str, str] = {}
    for backend in _EXPORT_BACKENDS:
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
        w.writerow(["id", "campaign_id", "central_claim", "llm_backends", "dataset",
                    "languages", "veracity", "veracity_confidence",
                    "member_count"])
        for backend in _EXPORT_BACKENDS:
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
        w.writerow(["id", "label", "central_claim", "llm_backends", "dataset",
                    "languages", "veracity", "veracity_confidence",
                    "coordination_score", "n1_burst", "n2_coamp",
                    "n3_reuse", "n4_crosslingual", "member_count"])
        for backend in _EXPORT_BACKENDS:
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

def _outputs_already_exist() -> bool:
    """True iff all three plan §4.6 CSVs already exist under _OUTPUT_DIR."""
    return all((_OUTPUT_DIR / fname).is_file() for fname in _OUTPUT_FILES)


def generate_dataset(cfg=None) -> dict:
    """Convert EUvsDisinfo, run the full pipeline, export CSVs.

    Honours the four plan §4.6 workflow parameters set in cfg:

    * ``camp_sample_size``        — cap on EUvsDisinfo articles to ingest
                                    (0 = all).
    * ``camp_re_extract``         — 'on' forces re-extraction even when the
                                    output CSVs already exist; 'off' skips
                                    the extraction pipeline and re-uses them.
    * ``camp_apply_coordination`` — 'off' skips the coordination-signal pass
                                    after extraction. (The pass is run by
                                    gen_campaigns; we forward the flag to it
                                    via cfg.)
    * ``camp_veracity_mode``      — 'off' / 'verify_hierarchy' / 'deep_verify'.
                                    On any non-'off' value, ``verify_hierarchy``
                                    is invoked once the hierarchy exists, with
                                    ``deep=True`` iff mode is 'deep_verify'.
    """
    from config import Config
    cfg = cfg or Config.load()

    re_extract        = str(getattr(cfg, "camp_re_extract", "off")).lower()
    veracity_mode     = str(getattr(cfg, "camp_veracity_mode", "off")).lower()
    apply_coord       = str(getattr(cfg, "camp_apply_coordination", "on")).lower()
    sample_size       = int(getattr(cfg, "camp_sample_size", 100) or 0)

    kb_root = Path("knowledge")
    kb = KnowledgeBase(kb_root)

    # ---- Plan §4.6 short-circuit: skip extraction if outputs already exist
    if re_extract == "off" and _outputs_already_exist():
        console.print(
            f"\n[bold cyan]Generate Dataset — EUvsDisinfo[/bold cyan]")
        console.print(
            f"[yellow]Existing CSVs found under {_OUTPUT_DIR}:[/yellow] "
            + ", ".join(_OUTPUT_FILES) + "\n"
            f"[dim]camp_re_extract={re_extract!r} → skipping extraction "
            f"pipeline. Set 'Re-Extract campaign candidates' to 'on' to "
            f"force a full re-run.[/dim]")
        # Still honour camp_veracity_mode against the existing hierarchy.
        summary: dict = {"articles": 0, "reused_csvs": True}
        if veracity_mode != "off":
            summary["veracity"] = _run_veracity_chain(kb, cfg, veracity_mode)
        return summary

    # ---- Convert EUvsDisinfo → KB articles
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
        f"[dim]sample_size={sample_size or 'all'}  "
        f"re_extract={re_extract}  apply_coordination={apply_coord}  "
        f"veracity_mode={veracity_mode}[/dim]")
    console.print(
        f"[bold]Converting EUvsDisinfo[/bold] [cyan]{data_dir}[/cyan]…")
    from core.converters.euvsdisinfo import convert
    # The converter takes an optional ``limit`` arg in newer revisions; pass
    # it via kwargs to stay backward-compatible with older converter sigs.
    try:
        n_articles = convert(data_dir, kb_root, limit=sample_size or None)
    except TypeError:
        # Older converter signature without limit support → convert all, then
        # warn that sample_size was ignored at the source layer. Pipeline
        # still respects it via _articles_from_kb (see run_pipeline).
        n_articles = convert(data_dir, kb_root)
        if sample_size:
            console.print(
                "[yellow]Note: EUvsDisinfo converter does not support an "
                "article-count limit; sample_size will be applied "
                "post-conversion when reading from the KB.[/yellow]")
    console.print(f"  {n_articles} articles loaded into KB")

    if n_articles == 0:
        console.print(
            "[red]No EUvsDisinfo articles converted — aborting pipeline.[/red]\n"
            "Check that the CSV under data/EUvsDisinfo/ has a body column "
            "(article_text / text / content / body) — the public "
            "euvsdisinfo_base.csv is URL-only and must be reconstructed first.")
        return {"articles": 0}

    # ---- Run pipeline
    console.print("\n[bold]Running pipeline…[/bold]")
    run_pipeline(kb, cfg)

    # ---- Export CSVs
    console.print("\n[bold]Exporting dataset CSVs…[/bold]")
    counts = export_csvs(kb, cfg, _OUTPUT_DIR)

    summary = {"articles": n_articles, "reused_csvs": False, **counts}

    # ---- Optional veracity chain
    if veracity_mode != "off":
        summary["veracity"] = _run_veracity_chain(kb, cfg, veracity_mode)

    return summary


def _run_veracity_chain(kb: KnowledgeBase, cfg, mode: str) -> dict:
    """Plan §4.6: chain ``verify_hierarchy`` after extraction.

    ``mode`` is one of 'verify_hierarchy' / 'deep_verify' (any 'off' value is
    filtered upstream). Returns the verifier's per-level summary so the caller
    can persist it into the run-stats sidecar.
    """
    from core.gen.gen_veracity import verify_hierarchy
    deep = (mode == "deep_verify")
    console.print(
        f"\n[bold]Veracity chain[/bold]  "
        f"[dim](camp_veracity_mode={mode!r}, deep={deep})[/dim]")
    return verify_hierarchy(kb, cfg, deep=deep)