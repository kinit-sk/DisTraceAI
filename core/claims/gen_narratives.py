"""Generate step: build the narrative hierarchy from sub-narratives (step 5).

Application of the retrieval component benchmarked in
``evaluation/eval_narratives.py``. The retrieval *method* is selected by
``nar_extractor`` (dense / specfi-cs / cspecfi / context-1); this module applies
it as an incremental assign-or-cluster loop:

  * Sub-narratives are streamed per source article (KB order). For each, the
    backend ranks existing narratives; a top match above ``nar_assign_threshold``
    merges. Non-matches go to an unassigned pool; a new narrative is seeded once
    ``nar_min_new_size`` mutually-similar pooled sub-narratives accumulate
    (``nar_new_threshold``).
  * Remaining pooled sub-narratives persist across articles — each new article
    adds to the pool, and new narratives may later form from it.

Periodic sweep / rebuild (``nar_recluster_cadence`` = N, 0 disables)
-------------------------------------------------------------------
Incremental assignment is path-dependent. Every N processed articles a
``ReclusteringSweep`` runs (splits low-cohesion narratives and re-assigns their
members). For ``cspecfi`` the same cadence ALSO rebuilds the NodeRAG conditioning
graph over the narratives-so-far — this is the operational difference from the
static ``specfi-cs`` baseline: cSpecFi's graph tracks the evolving hierarchy
rather than a fixed corpus. (NodeRAG 0.1.0 cannot do incremental updates, so the
cadence bounds the cost of full rebuilds instead of rebuilding every article.)

This step has no automatic metric (the hierarchy has no ground truth); its output
is the narrative KB under ``narratives/<dataset>/<backend>/``.
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    MofNCompleteColumn, TimeElapsedColumn,
)

from core.knowledge_base import KnowledgeBase, DATASET_POLYNARRATIVE, DATASET_FAKECTI
from core.models import make_embedder, make_generator, close_generator
from core.hierarchy.corpus import FactCheckCorpus
from core.hierarchy.assigner import RetrievalAssigner
from core.hierarchy.reclustering import ReclusteringSweep

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Backend construction
# ---------------------------------------------------------------------------

def _detector_slugs(detector_path: str) -> list[str]:
    if detector_path == "both":
        return ["xlm-multicw", "mdb-multicw"]
    return [os.path.basename(detector_path.rstrip("/\\"))]


def _build_backend(method, embedder, llm, cfg, *, dataset, detector, index_root):
    """Construct the retrieval backend for ``nar_extractor``.

    Returns (backend, noderag_or_None). The NodeRAG handle is returned so the
    caller can rebuild it on cadence (cspecfi only).
    """
    if method == "dense":
        from core.hierarchy.backends.bm25_rag import BM25RagBackend
        return BM25RagBackend(), None

    if method == "context-1":
        from core.hierarchy.backends.context1 import Context1Backend
        return Context1Backend(
            llm, embedder,
            max_turns=cfg.nar_context1_max_turns,
            token_budget=cfg.nar_context1_token_budget), None

    if method in ("specfi-cs", "specfi-ccs", "cspecfi"):
        from core.hierarchy.noderag import NodeRagGraph
        from core.hierarchy.backends.specfi_c import SpecFiCBackend
        mode = {"specfi-cs": "static", "specfi-ccs": "static-ccs",
                "cspecfi": "continuous"}[method]
        index_path = str(Path(index_root) / dataset / detector / method)
        # The static variants build a NodeRAG graph at eval/generate time; pass
        # model info so build() fills spare VRAM with an auto-sized worker pool
        # (torn down after the build). cspecfi also keeps a graph handle here so
        # the continuous rebuild (_rebuild_graph_from_narratives) can drive it.
        graph = NodeRagGraph(
            index_path, generate=llm, embedder=embedder,
            build_model_key=cfg.nar_generator, build_quant=cfg.nar_quantization,
            build_context_size=getattr(cfg, "nar_context1_token_budget", 16384),
            build_repr=("canonized" if method == "specfi-ccs" else "text"),
        )
        # static/static-ccs pass the graph to the backend (NodeRAG conditioning);
        # continuous passes noderag=None (conditions on the sub-narrative's
        # claims) but still returns the graph handle for the rebuild machinery.
        backend = SpecFiCBackend(
            embedder, llm,
            graph if mode != "continuous" else None,
            k=cfg.nar_specfi_hypotheticals, mode=mode)
        return backend, graph

    raise ValueError(f"Unknown nar_extractor: {method!r}")


def _rebuild_graph_from_narratives(graph, assigner) -> None:
    """Rebuild a cSpecFi NodeRAG graph over the narratives gathered so far.

    Writes each current narrative's central claim (+ member claims) as an input
    document, clears the built index, and triggers a fresh build. No-op if there
    are no narratives yet.
    """
    inp = Path(graph.index_path) / "input"
    inp.mkdir(parents=True, exist_ok=True)
    # Clear previous inputs so removed/merged narratives do not linger.
    for old in inp.glob("*.txt"):
        old.unlink()
    n = 0
    for nar in assigner.narratives.values():
        members = assigner._member_records(nar.sub_narratives)
        text = " | ".join([nar.central_claim] + [m.central_claim for m in members])
        (inp / f"{nar.id}.txt").write_text(text, encoding="utf-8")
        n += 1
    if n == 0:
        return
    graph._search = None      # force reload after rebuild
    try:
        graph.build()
        logger.info("[gen_nar] rebuilt cSpecFi NodeRAG graph over %d narratives", n)
    except Exception as exc:                       # pragma: no cover - runtime
        logger.warning("[gen_nar] cSpecFi graph rebuild failed (%s); "
                       "continuing with stale graph", exc)


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

def _process_dataset(dataset, detector, method, embedder, llm, kb, cfg,
                     index_root) -> dict:
    sns = kb.sub_narratives(dataset, detector)
    if not sns:
        console.print(
            f"  [dim]No sub-narratives for {dataset}/{detector} — run the "
            f"sub-narrative Generate first; skipping.[/dim]")
        return {}

    # Group sub-narratives by source article so processing streams article-by-
    # article (the cadence is defined in processed-article units).
    by_article: dict[str, list] = defaultdict(list)
    for sn in sns:
        by_article[sn.article_name].append(sn)
    articles = sorted(by_article)

    corpus = FactCheckCorpus(embedder)
    backend, graph = _build_backend(
        method, embedder, llm, cfg,
        dataset=dataset, detector=detector, index_root=index_root)

    assigner = RetrievalAssigner(
        backend, corpus, kb, llm,
        dataset=dataset, detector=detector,
        threshold=cfg.nar_assign_threshold,
        min_new_narrative_size=cfg.nar_min_new_size,
        new_narrative_threshold=cfg.nar_new_threshold)
    sweep = ReclusteringSweep(
        kb, assigner, embedder,
        cohesion_threshold=cfg.nar_assign_threshold)

    cadence = max(0, int(cfg.nar_recluster_cadence))
    is_cspecfi = (method == "cspecfi")

    # cSpecFi needs an initial graph before the first ranking; build it from any
    # pre-existing narratives (empty on a fresh run → first cadence builds it,
    # but we also build once up front so early articles get conditioning).
    if is_cspecfi and graph is not None and assigner.narratives:
        _rebuild_graph_from_narratives(graph, assigner)

    assigned = sweeps = 0
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]{dataset}/{detector} ({method})[/cyan]",
                                 total=len(articles))
        for i, article_name in enumerate(articles, start=1):
            for sn in by_article[article_name]:
                if assigner.assign(sn) is not None:
                    assigned += 1
            progress.advance(task)

            if cadence and (i % cadence == 0):
                if is_cspecfi and graph is not None:
                    _rebuild_graph_from_narratives(graph, assigner)
                result = sweep.run()
                sweeps += 1
                logger.info("[gen_nar] sweep after %d articles: %s", i, result)

    result = {
        "articles": len(articles),
        "sub_narratives": len(sns),
        "narratives": len(assigner.narratives),
        "unassigned_pool": assigner.unassigned_count,
        "sweeps": sweeps,
    }
    console.print(
        f"  {detector}: articles={result['articles']}  "
        f"sub_narratives={result['sub_narratives']}  "
        f"narratives={result['narratives']}  "
        f"pool={result['unassigned_pool']}  sweeps={sweeps}")
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate(
    detector_path: str,
    extractor: str,
    embedder_name: str,
    generator_key: str,
    quantization: str,
    kb: KnowledgeBase | None = None,
    *,
    cfg=None,
) -> dict:
    """Build narratives from sub-narratives across PolyNarrative and FakeCTI.

    Returns ``{dataset: {detector: {...counts...}}}``.
    """
    from config import Config
    cfg = cfg if cfg is not None else Config.load()
    if kb is None:
        kb = KnowledgeBase(Path("knowledge"))

    detector_slugs = _detector_slugs(detector_path)
    index_root = os.getenv("DISTRACE_NAR_NODERAG_INDEX_ROOT",
                           str(Path("knowledge") / "noderag"))

    console.print(f"\n[bold]Loading embedder[/bold] [cyan]{embedder_name}[/cyan]…")
    embedder = make_embedder(embedder_name)

    # All methods synthesize narrative central claims via the LLM. dense uses no
    # LLM for *ranking*, but synthesis still improves narrative quality, so the
    # generator is loaded unless DISTRACE_NAR_NO_LLM=1 (then the assigner falls
    # back to a representative member claim).
    llm = None
    if os.getenv("DISTRACE_NAR_NO_LLM") != "1":
        console.print(
            f"[bold]Loading generator[/bold] [cyan]{generator_key}[/cyan] "
            f"([dim]{quantization}[/dim])…")
        llm = make_generator(generator_key, quantization)
    elif extractor != "dense":
        raise RuntimeError(
            f"nar_extractor={extractor!r} requires a generator, but "
            "DISTRACE_NAR_NO_LLM=1 is set.")

    summary: dict = {}
    for dataset in [DATASET_POLYNARRATIVE, DATASET_FAKECTI]:
        for detector in detector_slugs:
            console.print(
                f"\n[bold]{dataset}[/bold]  [dim](detector: {detector}, "
                f"method: {extractor})[/dim]")
            result = _process_dataset(dataset, detector, extractor, embedder,
                                      llm, kb, cfg, index_root)
            if result:
                summary.setdefault(dataset, {})[detector] = result

    close_generator(llm)
    return summary
