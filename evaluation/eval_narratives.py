"""Evaluate narrative *retrieval* against the PolyNarrative taxonomy.

What this measures (and what it does NOT)
-----------------------------------------
Narrative extraction (step 5) is two things:

  * a RETRIEVAL component — given a query sub-narrative, rank previously-observed
    sub-narratives by relevance; and
  * its APPLICATION — an incremental assign-or-cluster loop (gen_narratives.py).

Only the retrieval component has ground truth, so this module benchmarks that
component in isolation.

Benchmark
---------
PolyNarrative's two-level taxonomy is shared across train and test. A query
and a corpus item are a correct match iff they carry the same fine-grained
sub-narrative label.

  Corpus  : sub-narratives from PolyNarrative train articles.
  Queries : sub-narratives from a held-out split (test or dev).

Gold labels are inherited from the source article. Query side uses the dominant
label; corpus side uses the full multilabel set.

Methods (nar_extractor)
-----------------------
  dense     : embedding cosine. nar_dense_repr selects the text representation:
                article   — raw article content (unit = ARTICLE)
                canonized — joined canonized claims (unit = ARTICLE)
                subnar    — sub-narrative central claim (unit = SUB-NARRATIVE)
  bm25-rag  : BM25 + dense RRF candidate selection, re-scored by cosine.
              Strictly stronger than dense/subnar; no LLM needed.
  specfi-cs : reproduced original SpecFi-CS (Upravitelev et al.).
              NodeRAG graph built ONCE over raw train ARTICLE TEXTS (matching
              the paper's setup: noderag_pn_training is built from article text).
              Community-level findings become few-shot examples; LLM generates
              n=10 hypothetical texts per query; max-cosine ranks against
              the sub-narrative corpus.
  cspecfi   : our continuous variant. NO NodeRAG. Instead, the query sub-
              narrative's own canonized claims are used as conditioning context
              for HyDE generation. No graph build step, no NodeRAG dependency.
              Continuous in the sense that the conditioning naturally updates
              as sub-narrative extraction proceeds.
  context-1 : agentic multi-turn search harness; gathered items re-scored by
              cosine to the query.

Corpus for ranking: all methods rank against sub-narrative central claims
(unit = SUB-NARRATIVE), except dense/article and dense/canonized which rank
articles.

Metrics
-------
Acc@1, Acc@3, Acc@5, MAP (recall-aware AP over the full ranking), reported
overall and per query-language. Unanswerable queries (no label match in corpus)
are excluded. Saved to results/eval_narratives_<detector>_<method>.csv.
"""
from __future__ import annotations

import csv
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from rich.console import Console
from rich import box
from rich.table import Table
from rich.terminal_theme import MONOKAI
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    MofNCompleteColumn, TimeElapsedColumn,
)

from core.knowledge_base import KnowledgeBase, DATASET_POLYNARRATIVE
from core.models import make_embedder, encode_with_backoff

logger  = logging.getLogger(__name__)
console = Console(record=True)

_CUTOFFS = (1, 3, 5)


# ---------------------------------------------------------------------------
# Item assembly
# ---------------------------------------------------------------------------

class _Item:
    """One retrievable item: text for embedding, labels for scoring,
    and (for query items) the underlying canonized claims for cSpecFi."""
    __slots__ = ("key", "text", "labels", "language", "unit", "claims")

    def __init__(self, key, text, labels, language, unit, claims=()):
        self.key      = key
        self.text     = text
        self.labels   = labels
        self.language = language
        self.unit     = unit
        self.claims   = list(claims)  # canonized claims — populated for query items


def _gold_labels(annotations, article_name):
    ann = annotations.get(article_name)
    if not ann:
        return set()
    return {s for s in ann.get("sub_narratives", []) if s and s.lower() != "none"}


def _split_of(annotations, article_name):
    ann = annotations.get(article_name)
    return ann.get("split") if ann else None


def _domain_of(article_name: str) -> str | None:
    """Infer the PolyNarrative domain from an article name.

    PolyNarrative names embed the domain as a token, e.g.
    ``test_PT_subtask-3-documents_PT_CC_TEST_523`` (CC = climate change) or
    ``..._PT_URW_TEST_486`` (URW = Ukraine-Russia war). Returns "CC", "URW",
    or None when no domain token is present.
    """
    if not article_name:
        return None
    tokens = {t.upper() for t in article_name.replace("-", "_").split("_")}
    if "CC" in tokens:
        return "CC"
    if "URW" in tokens:
        return "URW"
    return None


def _domain_ok(article_name: str, domain: str | None) -> bool:
    """True if the article belongs to the requested domain (or domain is all/None)."""
    if not domain or domain.lower() == "all":
        return True
    return _domain_of(article_name) == domain


def _dataset_seg(domain: str | None) -> str:
    """Dataset path segment for HTML reports; appends the domain when subset."""
    if not domain or domain.lower() == "all":
        return "polynarrative"
    return f"polynarrative-{domain}"


def _print_benchmark_table(rows, domain=None):
    """Print a side-by-side benchmark table for the 'all' method run.

    rows: list of (detector_slug, method, overall_dict, n_scored).
    Highlights the best method per metric and prints simple spread statistics.
    """
    from rich.table import Table
    from rich import box

    dom = "" if domain in (None, "all") else f"  ·  domain={domain}"
    console.rule(f"[bold cyan]Narrative retrieval benchmark{dom}[/bold cyan]")

    metrics = [("Acc@1", 1), ("Acc@3", 3), ("Acc@5", 5), ("MAP", "map")]
    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold white")
    t.add_column("Detector", style="dim")
    t.add_column("Method", style="bold")
    for label, _ in metrics:
        t.add_column(label, justify="right")
    t.add_column("N", justify="right", style="dim")

    # Best value per (detector, metric) for highlighting.
    best: dict[tuple, float] = {}
    for det, _method, overall, _n in rows:
        for _, key in metrics:
            v = overall.get(key, 0.0)
            best[(det, key)] = max(best.get((det, key), float("-inf")), v)

    for det, method, overall, n in rows:
        cells = [det, method]
        for _, key in metrics:
            v = overall.get(key, 0.0)
            s = f"{v:.3f}"
            if abs(v - best.get((det, key), v)) < 1e-9:
                s = f"[bold green]{s}[/bold green]"
            cells.append(s)
        cells.append(str(n))
        t.add_row(*cells)
    console.print(t)

    # Spread statistics across methods (per detector) for the headline metric.
    import statistics as _stats
    by_det: dict[str, list[float]] = defaultdict(list)
    for det, _method, overall, _n in rows:
        by_det[det].append(overall.get("map", 0.0))
    for det, maps in by_det.items():
        if len(maps) >= 2:
            console.print(
                f"  [dim]{det}: MAP min={min(maps):.3f}  max={max(maps):.3f}  "
                f"mean={_stats.mean(maps):.3f}  stdev={_stats.pstdev(maps):.3f}[/dim]")


def _build_items(kb, detector_slug, annotations, repr_mode, want_split,
                 is_query=False, domain=None):
    """Assemble retrievable items for one split.

    repr_mode in {subnar, article, canonized}.
    When is_query=True and repr_mode=subnar, the item's .claims list is
    populated from the sub-narrative (used by cSpecFi for conditioning).
    domain: optional "CC"/"URW" filter (None or "all" keeps both).
    """
    sns = kb.sub_narratives(DATASET_POLYNARRATIVE, detector_slug)
    by_article = defaultdict(list)
    for sn in sns:
        by_article[sn.article_name].append(sn)

    items = []

    if repr_mode == "subnar":
        for sn in sns:
            if _split_of(annotations, sn.article_name) != want_split:
                continue
            if not _domain_ok(sn.article_name, domain):
                continue
            labels = _gold_labels(annotations, sn.article_name)
            if not labels:
                continue
            ann = annotations[sn.article_name]
            claims = sn.claims if is_query else []
            items.append(_Item(
                key=sn.id,
                text=sn.central_claim or "",
                labels=labels,
                language=ann.get("language", "??"),
                unit="subnar",
                claims=claims,
            ))
        return [it for it in items if it.text.strip()]

    # article / canonized
    for article_name in sorted(by_article):
        if _split_of(annotations, article_name) != want_split:
            continue
        if not _domain_ok(article_name, domain):
            continue
        labels = _gold_labels(annotations, article_name)
        if not labels:
            continue
        ac = kb.load_article_claims(DATASET_POLYNARRATIVE, detector_slug, article_name)
        if ac is None:
            continue
        if repr_mode == "article":
            text = " ".join(c.sentence for c in ac.claims) or ac.title or ""
        else:  # canonized
            text = " | ".join(c for c in ac.canonized_claims if c and c.strip())
        if not text.strip():
            continue
        ann = annotations[article_name]
        items.append(_Item(
            key=article_name, text=text, labels=labels,
            language=ann.get("language", "??"), unit="article"))
    return items


def _build_article_texts(kb, detector_slug, annotations, want_split="train",
                         domain=None):
    """Raw article texts for the SpecFi-CS NodeRAG graph.

    The published SpecFi paper (Upravitelev et al.) builds NodeRAG over the
    raw article texts of the training set — not over sub-narrative central
    claims. This matches noderag_pn_training in the reference implementation.
    Returns {article_name: raw_text} for articles in want_split with labels.
    domain: optional "CC"/"URW" filter (None or "all" keeps both).
    """
    sns = kb.sub_narratives(DATASET_POLYNARRATIVE, detector_slug)
    article_names = {sn.article_name for sn in sns
                     if _split_of(annotations, sn.article_name) == want_split
                     and _domain_ok(sn.article_name, domain)
                     and _gold_labels(annotations, sn.article_name)}
    out = {}
    for name in sorted(article_names):
        ac = kb.load_article_claims(DATASET_POLYNARRATIVE, detector_slug, name)
        if ac is None:
            continue
        text = " ".join(c.sentence for c in ac.claims) or ac.title or ""
        if text.strip():
            out[name] = text
    return out


def _build_article_canonized(kb, detector_slug, annotations, want_split="train",
                             domain=None):
    """Per-article canonized claims for the SpecFi-CCS NodeRAG graph.

    SpecFi-CCS builds the graph from the CANONIZED CLAIMS extracted per article
    rather than the raw article text. One input document per article: that
    article's canonized claims joined together (newline-separated) so NodeRAG
    still forms communities ACROSS articles while seeing decontextualised,
    English-normalised claim text instead of raw multilingual prose.
    Returns {article_name: joined_canonized_claims} for articles in want_split.
    domain: optional "CC"/"URW" filter (None or "all" keeps both).
    """
    sns = kb.sub_narratives(DATASET_POLYNARRATIVE, detector_slug)
    article_names = {sn.article_name for sn in sns
                     if _split_of(annotations, sn.article_name) == want_split
                     and _domain_ok(sn.article_name, domain)
                     and _gold_labels(annotations, sn.article_name)}
    out = {}
    for name in sorted(article_names):
        ac = kb.load_article_claims(DATASET_POLYNARRATIVE, detector_slug, name)
        if ac is None:
            continue
        claims = [c.strip() for c in ac.canonized_claims if c and c.strip()]
        if claims:
            out[name] = "\n".join(claims)
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _relevant_mask(corpus_items, query_labels):
    return [i for i, it in enumerate(corpus_items) if it.labels & query_labels]


def _average_precision(ranking, relevant_set):
    if not relevant_set:
        return None
    hits = 0
    precisions = []
    for rank, ci in enumerate(ranking, start=1):
        if ci in relevant_set:
            hits += 1
            precisions.append(hits / rank)
    return sum(precisions) / len(relevant_set)


def _score_rankings(rankings, query_items, corpus_items):
    hits = {c: [] for c in _CUTOFFS}
    aps  = []
    langs = []
    n_skipped = 0
    for qi, q in enumerate(query_items):
        relevant = set(_relevant_mask(corpus_items, q.labels))
        if not relevant:
            n_skipped += 1
            continue
        ranking = list(rankings[qi])
        first_hit = next((r for r, ci in enumerate(ranking) if ci in relevant), None)
        for c in _CUTOFFS:
            hits[c].append(first_hit is not None and first_hit < c)
        aps.append(_average_precision(ranking, relevant))
        langs.append(q.language)
    return hits, aps, langs, n_skipped


def _acc(flags):
    return sum(flags) / len(flags) if flags else 0.0


def _per_language(hits, aps, langs):
    acc_buckets = defaultdict(lambda: {c: [] for c in _CUTOFFS})
    ap_buckets  = defaultdict(list)
    for i, lang in enumerate(langs):
        for c in _CUTOFFS:
            acc_buckets[lang][c].append(hits[c][i])
        ap_buckets[lang].append(aps[i])
    counts = {lang: len(v[_CUTOFFS[0]]) for lang, v in acc_buckets.items()}
    out = {}
    for lang in sorted(acc_buckets):
        out[lang] = {c: _acc(acc_buckets[lang][c]) for c in _CUTOFFS}
        out[lang]["map"] = (sum(ap_buckets[lang]) / len(ap_buckets[lang])
                            if ap_buckets[lang] else 0.0)
    return out, counts


# ---------------------------------------------------------------------------
# Display / save
# ---------------------------------------------------------------------------

def _style(v):
    return "bold green" if v >= 0.70 else ("yellow" if v >= 0.40 else "red")


def _print_results(detector_slug, method, unit, overall, per_lang,
                   lang_counts, n_query, n_corpus):
    console.print()
    console.rule(
        f"[bold cyan]Narrative Retrieval — {detector_slug} / {method}"
        f"{f' ({unit})' if unit else ''}[/bold cyan]")
    console.print(
        f"  [dim]queries={n_query}  corpus={n_corpus}  retrieval unit={unit}[/dim]")
    console.print()
    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold white")
    t.add_column("Scope", style="bold", min_width=10)
    for c in _CUTOFFS:
        t.add_column(f"Acc@{c}", justify="right", min_width=8)
    t.add_column("MAP", justify="right", min_width=8)
    t.add_column("N", justify="right", min_width=6)
    t.add_row("OVERALL",
              *[f"[{_style(overall[c])}]{overall[c]:.3f}[/]" for c in _CUTOFFS],
              f"[{_style(overall['map'])}]{overall['map']:.3f}[/]",
              str(n_query), style="bold")
    t.add_section()
    for lang, accs in per_lang.items():
        t.add_row(lang,
                  *[f"[{_style(accs[c])}]{accs[c]:.3f}[/]" for c in _CUTOFFS],
                  f"[{_style(accs['map'])}]{accs['map']:.3f}[/]",
                  str(lang_counts.get(lang, "")))
    console.print(t)
    console.print()


def _save_csv(detector_slug, method, unit, overall, per_lang, lang_counts, n_query):
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"eval_narratives_{detector_slug}_{method}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope", "unit", *[f"acc@{c}" for c in _CUTOFFS], "map", "n"])
        w.writerow(["overall", unit, *[overall[c] for c in _CUTOFFS],
                    overall["map"], n_query])
        for lang, accs in per_lang.items():
            w.writerow([lang, unit, *[accs[c] for c in _CUTOFFS], accs["map"],
                        lang_counts.get(lang, "")])
    logger.info("[eval_nar] results saved to %s", path)


# ---------------------------------------------------------------------------
# Dense helpers
# ---------------------------------------------------------------------------

def _l2(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.where(norms == 0, 1e-10, norms)


def _rank_dense_full(query_embs, corpus_embs):
    sims = query_embs @ corpus_embs.T
    return np.argsort(-sims, axis=1)


# ---------------------------------------------------------------------------
# Progress helper
# ---------------------------------------------------------------------------

class _Progress:
    def __init__(self, desc):
        self.desc = desc
    def __enter__(self):
        self.p = Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console)
        self.p.__enter__()
        self.task = self.p.add_task(f"[cyan]{self.desc}[/cyan]", total=0)
        return self
    def start(self, total):
        self.p.update(self.task, total=total)
    def advance(self):
        self.p.advance(self.task)
    def __exit__(self, *a):
        self.p.__exit__(*a)


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------

def _run_dense(query_items, corpus_items, embedder):
    corpus_embs = _l2(np.asarray(
        encode_with_backoff(embedder, [it.text for it in corpus_items]),
        dtype=np.float32))
    query_embs = _l2(np.asarray(
        encode_with_backoff(embedder, [it.text for it in query_items]),
        dtype=np.float32))
    return _rank_dense_full(query_embs, corpus_embs)


def _run_bm25rag(query_items, corpus_items, embedder):
    """BM25 + dense RRF candidate selection, re-scored by cosine.

    Strictly stronger than pure dense: BM25 catches keyword overlap that
    embedding similarity misses, and the two signals are fused via RRF before
    the final cosine re-score. No LLM needed.
    """
    from core.hierarchy.corpus import FactCheckCorpus
    from core.hierarchy.backends.bm25_rag import BM25RagBackend

    fc = FactCheckCorpus(embedder)
    for it in corpus_items:
        fc.add_cluster(it.key, [it.text])

    backend = BM25RagBackend()
    key_to_idx = {it.key: i for i, it in enumerate(corpus_items)}
    n = len(corpus_items)
    rankings = []
    with _Progress("BM25-RAG retrieval") as prog:
        prog.start(len(query_items))
        for q in query_items:
            ranked = backend.rank(q.text, fc, k=n)
            idxs = [key_to_idx[cid] for cid, _ in ranked if cid in key_to_idx]
            seen = set(idxs)
            idxs += [i for i in range(n) if i not in seen]
            rankings.append(idxs)
            prog.advance()
    return rankings


def _build_noderag_on_articles(article_texts, embedder, llm, index_path,
                                *, model_key=None, quant=None, ctx=16384):
    """Build a NodeRAG graph over the supplied input documents.

    Used by both SpecFi-CS (raw article texts) and SpecFi-CCS (per-article
    canonized claims). Each entry of ``article_texts`` becomes one input
    document. The graph is built once and reused across all queries.

    ``model_key``/``quant`` enable an auto-sized parallel build pool inside
    NodeRagGraph.build(): the graph construction fills spare VRAM with extra
    worker contexts and tears them down afterwards, dramatically cutting build
    time versus the single-context default.
    """
    from core.hierarchy.noderag import NodeRagGraph

    inp = Path(index_path) / "input"
    inp.mkdir(parents=True, exist_ok=True)
    for name, text in article_texts.items():
        (inp / f"{name}.txt").write_text(text, encoding="utf-8")

    graph = NodeRagGraph(index_path, generate=llm, embedder=embedder,
                         build_model_key=model_key, build_precision=quant,
                         build_context_size=ctx)
    graph.ensure_loaded()   # builds once if HNSW.bin absent, reuses if present
    return graph


def _rank_via_specfi_backend(query_items, corpus_items, embedder, backend, desc,
                              get_claims=None):
    """Shared retrieval loop for SpecFi-CS and cSpecFi.

    get_claims: optional callable(query_item) -> list[str] that supplies
    per-query conditioning claims to the backend (cSpecFi only).
    Returns full-length rankings (list of index lists).
    """
    from core.hierarchy.corpus import FactCheckCorpus
    fc = FactCheckCorpus(embedder)
    for it in corpus_items:
        fc.add_cluster(it.key, [it.text])

    n = len(corpus_items)
    key_to_idx = {it.key: i for i, it in enumerate(corpus_items)}
    rankings = []
    with _Progress(desc) as prog:
        prog.start(len(query_items))
        for q in query_items:
            if get_claims is not None:
                ranked = backend.rank(q.text, fc, k=n, claims=get_claims(q))
            else:
                ranked = backend.rank(q.text, fc, k=n)
            idxs = [key_to_idx[cid] for cid, _ in ranked if cid in key_to_idx]
            seen = set(idxs)
            idxs += [i for i in range(n) if i not in seen]
            rankings.append(idxs)
            prog.advance()
    return rankings


def _run_specfi_cs(query_items, corpus_items, embedder, cfg, kb, detector_slug,
                   annotations, domain=None):
    """Reproduced original SpecFi-CS.

    NodeRAG graph is built over raw train article texts (matching the paper's
    noderag_pn_training setup). Corpus for ranking = sub-narrative central
    claims. n=10 hypotheticals per query (matching paper's generate_hypotheticals
    n=10 call). Graph is built once and reused across queries.
    domain: optional "CC"/"URW" subset filter (separate index cache per domain).
    """
    from core.models import make_generator, close_generator
    from core.hierarchy.backends.specfi_c import SpecFiCBackend

    llm = make_generator(cfg.nar_generator, cfg.nar_precision)

    # Build NodeRAG over article texts, not sub-narrative claims.
    article_texts = _build_article_texts(kb, detector_slug, annotations,
                                         want_split="train", domain=domain)
    if not article_texts:
        console.print(
            "  [yellow]No article texts found for SpecFi-CS NodeRAG build.[/yellow]")
        close_generator(llm)
        return None

    dom_seg = "" if domain in (None, "all") else f"_{domain}"
    index_path = os.getenv(
        "DISTRACE_NAR_NODERAG_INDEX",
        str(Path("knowledge") / "noderag" / "specfi_cs" / f"{detector_slug}{dom_seg}"))
    graph = _build_noderag_on_articles(article_texts, embedder, llm, index_path,
                                       model_key=cfg.nar_generator,
                                       quant=cfg.nar_precision,
                                       ctx=getattr(cfg, "nar_context1_token_budget", 16384))

    backend = SpecFiCBackend(embedder, llm, graph,
                             k=cfg.nar_specfi_hypotheticals,
                             mode="static")
    out = _rank_via_specfi_backend(query_items, corpus_items, embedder, backend,
                                   "SpecFi-CS retrieval")
    close_generator(llm)
    return out


def _run_specfi_ccs(query_items, corpus_items, embedder, cfg, kb, detector_slug,
                    annotations, domain=None):
    """SpecFi-CCS — Canonized Community Summaries.

    Identical to SpecFi-CS EXCEPT the NodeRAG graph is built from the per-article
    CANONIZED CLAIMS rather than raw article text. The graph still forms
    communities across articles, but over decontextualised English claims, so
    the community summaries are claim-centric. Conditioning and ranking are
    otherwise the same as SpecFi-CS. Uses a separate index cache directory so it
    never collides with the CS graph.
    domain: optional "CC"/"URW" subset filter (separate index cache per domain).
    """
    from core.models import make_generator, close_generator
    from core.hierarchy.backends.specfi_c import SpecFiCBackend

    llm = make_generator(cfg.nar_generator, cfg.nar_precision)

    canonized = _build_article_canonized(kb, detector_slug, annotations,
                                          want_split="train", domain=domain)
    if not canonized:
        console.print(
            "  [yellow]No canonized claims found for SpecFi-CCS NodeRAG build.[/yellow]")
        close_generator(llm)
        return None

    dom_seg = "" if domain in (None, "all") else f"_{domain}"
    index_path = os.getenv(
        "DISTRACE_NAR_NODERAG_CCS_INDEX",
        str(Path("knowledge") / "noderag" / "specfi_ccs" / f"{detector_slug}{dom_seg}"))
    graph = _build_noderag_on_articles(canonized, embedder, llm, index_path,
                                       model_key=cfg.nar_generator,
                                       quant=cfg.nar_precision,
                                       ctx=getattr(cfg, "nar_context1_token_budget", 16384))

    backend = SpecFiCBackend(embedder, llm, graph,
                             k=cfg.nar_specfi_hypotheticals,
                             mode="static-ccs")
    out = _rank_via_specfi_backend(query_items, corpus_items, embedder, backend,
                                   "SpecFi-CCS retrieval")
    close_generator(llm)
    return out


def _run_cspecfi(query_items, corpus_items, embedder, cfg):
    """cSpecFi — our continuous variant.

    No NodeRAG. The query sub-narrative's own canonized claims provide the
    conditioning context for HyDE generation instead of NodeRAG community
    summaries. This is 'continuous' because the conditioning naturally updates
    as sub-narrative extraction proceeds — no graph build or rebuild needed.
    The specfi-cs vs cspecfi comparison isolates the conditioning source:
    graph community summaries vs. claim-level evidence in the sub-narrative.
    """
    from core.models import make_generator, close_generator
    from core.hierarchy.backends.specfi_c import SpecFiCBackend

    llm = make_generator(cfg.nar_generator, cfg.nar_precision)
    # cSpecFi: noderag=None signals the backend to skip _conditioning()
    # and receive claims directly via rank(..., claims=[...]).
    backend = SpecFiCBackend(embedder, llm, noderag=None,
                             k=cfg.nar_specfi_hypotheticals,
                             mode="continuous")
    out = _rank_via_specfi_backend(
        query_items, corpus_items, embedder, backend,
        "cSpecFi retrieval",
        get_claims=lambda q: q.claims)
    close_generator(llm)
    return out


def _run_context1(query_items, corpus_items, embedder, cfg):
    from core.models import make_generator, close_generator
    from core.hierarchy.backends.context1 import Context1Backend

    llm = make_generator(cfg.nar_generator, cfg.nar_precision)
    from core.hierarchy.corpus import FactCheckCorpus
    fc = FactCheckCorpus(embedder)
    for it in corpus_items:
        fc.add_cluster(it.key, [it.text])

    backend = Context1Backend(
        llm, embedder,
        max_turns=cfg.nar_context1_max_turns,
        token_budget=cfg.nar_context1_token_budget)
    n = len(corpus_items)
    key_to_idx = {it.key: i for i, it in enumerate(corpus_items)}
    rankings = []
    with _Progress("Context-1 retrieval") as prog:
        prog.start(len(query_items))
        for q in query_items:
            ranked = backend.rank(q.text, fc, k=n)
            idxs = [key_to_idx[cid] for cid, _ in ranked if cid in key_to_idx]
            seen = set(idxs)
            idxs += [i for i in range(n) if i not in seen]
            rankings.append(idxs)
            prog.advance()
    close_generator(llm)
    return rankings


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(cfg=None):
    from config import Config
    cfg = cfg if cfg is not None else Config.load()

    kb_root = Path("knowledge")
    kb = KnowledgeBase(kb_root)

    gt_path = kb_root / "ground_truth" / "annotations.json"
    if not gt_path.exists():
        poly_src = Path("data/PolyNarrative")
        if not poly_src.exists():
            console.print(
                f"[red]Ground truth not found:[/red] {gt_path}\n"
                f"[red]PolyNarrative data not found:[/red] {poly_src}")
            return
        console.print(
            f"[yellow]Ground truth not found — running converter on {poly_src}…[/yellow]")
        from core.converters.polynarrative import convert
        convert(poly_src, kb_root)
        if not gt_path.exists():
            console.print(f"[red]Converter ran but {gt_path} not produced.[/red]")
            return

    annotations = json.loads(gt_path.read_text(encoding="utf-8"))

    detector_path = cfg.nar_detector
    detector_slugs = (["xlm-multicw", "mdb-multicw"] if detector_path == "both"
                      else [os.path.basename(detector_path.rstrip("/\\"))])

    configured = cfg.nar_extractor
    query_split = cfg.nar_eval_split
    domain = getattr(cfg, "nar_eval_domain", "all")
    dom_label = "" if domain in (None, "all") else f"  domain={domain}"

    # "all" runs the full benchmark across every method; otherwise a single one.
    ALL_METHODS = ["dense", "bm25-rag", "specfi-cs", "specfi-ccs",
                   "cspecfi", "context-1"]
    methods = ALL_METHODS if configured == "all" else [configured]
    benchmark = configured == "all"

    console.print(f"\n[bold]Loading embedder[/bold] [cyan]{cfg.nar_embedder}[/cyan]…")
    embedder = make_embedder(cfg.nar_embedder)

    from evaluation.report_paths import report_path

    # Collected for the benchmark summary table: (detector, method, overall, n).
    bench_rows: list[tuple[str, str, dict, int]] = []

    for detector_slug in detector_slugs:
        for method in methods:
            repr_mode = cfg.nar_dense_repr if method == "dense" else "subnar"
            console.print(
                f"\n[bold cyan]Evaluating — {detector_slug} / {method}"
                f"{f' / repr={repr_mode}' if method == 'dense' else ''}"
                f"{dom_label}[/bold cyan]")

            corpus_items = _build_items(kb, detector_slug, annotations,
                                        repr_mode, want_split="train",
                                        is_query=False, domain=domain)
            query_items  = _build_items(kb, detector_slug, annotations,
                                        repr_mode, want_split=query_split,
                                        is_query=True, domain=domain)

            if not corpus_items:
                console.print(
                    f"  [yellow]No train sub-narratives for polynarrative/"
                    f"{detector_slug}{dom_label}. Run sub-narrative Generate "
                    f"first.[/yellow]")
                continue
            if not query_items:
                console.print(
                    f"  [yellow]No '{query_split}' sub-narratives to query"
                    f"{dom_label}.[/yellow]")
                continue

            unit = corpus_items[0].unit
            console.print(
                f"  Corpus: {len(corpus_items)} items  Queries: {len(query_items)} "
                f"(unit: {unit})")

            rankings = None
            if method == "dense":
                rankings = _run_dense(query_items, corpus_items, embedder)
            elif method == "bm25-rag":
                rankings = _run_bm25rag(query_items, corpus_items, embedder)
            elif method == "specfi-cs":
                rankings = _run_specfi_cs(query_items, corpus_items, embedder, cfg,
                                          kb, detector_slug, annotations,
                                          domain=domain)
            elif method == "specfi-ccs":
                rankings = _run_specfi_ccs(query_items, corpus_items, embedder, cfg,
                                           kb, detector_slug, annotations,
                                           domain=domain)
            elif method == "cspecfi":
                missing = sum(1 for q in query_items if not q.claims)
                if missing:
                    console.print(
                        f"  [yellow]{missing}/{len(query_items)} query sub-narratives "
                        f"have no canonized claims stored. cSpecFi conditioning will "
                        f"fall back to the central claim for those items.[/yellow]")
                rankings = _run_cspecfi(query_items, corpus_items, embedder, cfg)
            elif method == "context-1":
                rankings = _run_context1(query_items, corpus_items, embedder, cfg)
            else:
                console.print(f"  [red]Unknown nar_extractor: {method!r}[/red]")
                continue

            if rankings is None:
                continue

            hits, aps, langs, n_skipped = _score_rankings(
                rankings, query_items, corpus_items)
            if n_skipped:
                console.print(
                    f"  [dim]{n_skipped} query(ies) skipped "
                    f"(no label match in corpus).[/dim]")
            n_scored = len(langs)
            if n_scored == 0:
                console.print("  [yellow]No answerable queries; nothing to score.[/yellow]")
                continue

            overall = {c: _acc(hits[c]) for c in _CUTOFFS}
            overall["map"] = sum(aps) / len(aps)
            per_lang, lang_counts = _per_language(hits, aps, langs)

            _print_results(detector_slug, method, unit, overall, per_lang,
                           lang_counts, n_scored, len(corpus_items))
            _save_csv(detector_slug, method, unit, overall, per_lang,
                      lang_counts, n_scored)
            bench_rows.append((detector_slug, method, dict(overall), n_scored))

            try:
                from core.ui.stats import save_eval_stats
                dom_suffix = "" if domain in (None, "all") else f"__{domain}"
                save_eval_stats(
                    "narratives",
                    param_key=f"{detector_slug}__{method}{dom_suffix}",
                    params={"detector": detector_slug, "method": method,
                            "domain": domain},
                    scores={"acc@1": overall[1], "acc@3": overall[3],
                            "acc@5": overall[5], "map": overall["map"],
                            "n": n_scored},
                    det_slug=detector_slug,
                )
            except Exception:
                pass

            if not benchmark:
                # Single-method run: one structured HTML report per detector × method.
                html_out = report_path(
                    "narratives", dataset=_dataset_seg(domain),
                    detector=detector_slug, method=method)
                console.save_html(str(html_out), theme=MONOKAI, clear=False)
                console.print(f"[dim]HTML report → {html_out}[/dim]")

    # ── Benchmark summary ─────────────────────────────────────────────────
    if benchmark and bench_rows:
        _print_benchmark_table(bench_rows, domain)
        html_out = report_path(
            "narratives", dataset=_dataset_seg(domain),
            detector=(detector_slugs[0] if len(detector_slugs) == 1 else "both"),
            method="all")
        console.save_html(str(html_out), theme=MONOKAI, clear=False)
        console.print(f"[dim]Benchmark HTML report → {html_out}[/dim]")
