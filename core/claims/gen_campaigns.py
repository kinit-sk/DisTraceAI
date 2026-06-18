"""Generate step: build the campaign hierarchy from narratives (step 6).

Same assign-or-cluster mechanism as gen_narratives.py, one level up:
narratives→campaigns instead of sub-narratives→narratives. The extractor
backend is selected by ``camp_extractor`` (same choices as nar_extractor).

After clustering, four coordination signals are computed for each campaign
and combined into a weighted coordination_score:

  N1  burst / time synchrony       — publication-date entropy (inverted)
  N2  co-amplification             — pairwise Jaccard of outlet sets per narrative
  N3  content reuse                — fraction of article pairs with near-identical
                                     canonized claims (Jaccard token overlap)
  N4  cross-lingual co-occurrence  — normalised language diversity

Signals are renormalised over available signals (e.g. a corpus with no
timestamps produces a meaningful N2/N3/N4 score; N1 contributes zero but the
remaining weights are scaled up to sum to 1).

Classification (two-axis):
  coordination_score < camp_coordination_threshold  → Organic Trend
  coordination_score ≥ threshold AND veracity ≥ camp_veracity_threshold
                                                    → Information Campaign
  coordination_score ≥ threshold AND veracity < camp_veracity_threshold
                                                    → Disinformation Campaign
  coordination_score ≥ threshold AND veracity is None
                                                    → Information Campaign (default;
                                                       run Verify hierarchy first)
"""
from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    MofNCompleteColumn, TimeElapsedColumn,
)

from core.knowledge_base import KnowledgeBase, DATASET_FAKECTI, DATASET_POLYNARRATIVE
from core.structures import Campaign, Narrative
from core.ids import campaign_id
from core.models import make_embedder, make_generator, close_generator

logger  = logging.getLogger(__name__)
console = Console()

_MASSIVESUMM_DATASET = "massivesumm"   # dataset slug for MassiveSumm runs


# ---------------------------------------------------------------------------
# Coordination signals
# ---------------------------------------------------------------------------

def _n1_burst(published_ats: list[str]) -> float:
    """Burst / time synchrony: inverted normalised date entropy.

    High burst = many articles published in a short window = low entropy.
    Returns 0.0 when no dates are available.
    """
    if not published_ats:
        return 0.0
    from collections import Counter
    from datetime import datetime
    buckets: Counter = Counter()
    for ts in published_ats:
        try:
            dt = datetime.strptime(ts[:10], "%Y-%m-%d")
            buckets[dt.strftime("%Y-%W")] += 1
        except (ValueError, TypeError):
            pass   # skip malformed dates — do NOT bucket them as "unknown"

    # Need at least 2 valid dates to say anything about synchrony
    total = sum(buckets.values())
    if total < 2:
        return 0.0
    entropy = -sum((c / total) * math.log2(c / total) for c in buckets.values())
    max_entropy = math.log2(len(buckets)) if len(buckets) > 1 else 1.0
    normalised = entropy / max_entropy if max_entropy > 0 else 0.0
    return float(1.0 - normalised)         # burst = inverted entropy


def _n2_coamplification(narrative_domains: list[set[str]]) -> float:
    """Co-amplification: mean pairwise Jaccard similarity of outlet sets.

    High score = same outlets repeatedly push the same narratives together.
    """
    filtered = [s for s in narrative_domains if s]
    if len(filtered) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i, a in enumerate(filtered):
        for b in filtered[i + 1:]:
            inter = len(a & b)
            union = len(a | b)
            total += inter / union if union > 0 else 0.0
            pairs += 1
    return total / pairs if pairs > 0 else 0.0


def _n3_content_reuse(claim_sets: list[set[str]]) -> float:
    """Content reuse: fraction of article pairs with Jaccard token overlap > 0.5."""
    if len(claim_sets) < 2:
        return 0.0
    total_pairs = high_overlap = 0
    for i, a in enumerate(claim_sets):
        for b in claim_sets[i + 1:]:
            inter = len(a & b)
            union = len(a | b)
            jac = inter / union if union > 0 else 0.0
            if jac > 0.5:
                high_overlap += 1
            total_pairs += 1
    return high_overlap / total_pairs if total_pairs > 0 else 0.0


def _n4_crosslingual(languages: list[str]) -> float:
    """Cross-lingual co-occurrence: normalised language diversity.

    Score = min(n_distinct_languages - 1, 3) / 3.
    0 = monolingual, 1 = 4+ languages.
    """
    n = len(set(languages))
    return min(max(n - 1, 0), 3) / 3.0


def compute_coordination(campaign: Campaign, kb: KnowledgeBase,
                         dataset: str, det_slug: str,
                         cfg) -> None:
    """Compute N1–N4 signals, apply weights, set coordination_score and label."""
    # Collect article metadata via narrative → sub-narrative → article chain
    all_dates: list[str]          = []
    all_domains: list[str]        = []
    all_languages: list[str]      = []
    narrative_domains: list[set]  = []    # per-narrative domain set (for N2)
    article_claim_sets: list[set] = []    # per-article token set of claims (for N3)

    # Build sub-narrative lookup once; avoids O(N×M×K) KB reads
    _sn_index = {sn.id: sn for sn in kb.sub_narratives(dataset, det_slug)}

    for nar_id in campaign.narratives:
        nar = None
        for backend in ("dense", "bm25-rag", "specfi-cs", "cspecfi", "context-1",
                        "bm25_rag"):
            nars = {n.id: n for n in kb.narratives(dataset, backend)}
            if nar_id in nars:
                nar = nars[nar_id]
                break
        if nar is None:
            continue

        nar_domains: set[str] = set()
        for sn_id in nar.sub_narratives:
            sn = _sn_index.get(sn_id)
            if sn is None:
                continue
            ac = kb.load_article_claims(dataset, det_slug, sn.article_name)
            if ac is None:
                continue
            art = kb.load_article(sn.article_name, dataset=dataset)

            if art and art.published_at:
                all_dates.append(art.published_at)
            if art and art.source_domain:
                all_domains.append(art.source_domain)
                nar_domains.add(art.source_domain)
            if art and art.source_language:
                all_languages.append(art.source_language)

            # N3: token set from canonized claims
            tokens: set[str] = set()
            for claim in ac.canonized_claims:
                tokens |= {w.lower() for w in claim.split() if len(w) > 3}
            if tokens:
                article_claim_sets.append(tokens)

        narrative_domains.append(nar_domains)

    # Store accumulated signals back on campaign
    campaign.published_ats  = all_dates
    campaign.source_domains = list(set(all_domains))
    campaign.languages      = sorted(set(all_languages))

    # Compute raw signals
    raw = {
        "n1_burst":        _n1_burst(all_dates),
        "n2_coamp":        _n2_coamplification(narrative_domains),
        "n3_reuse":        _n3_content_reuse(article_claim_sets),
        "n4_crosslingual": _n4_crosslingual(all_languages),
    }
    campaign.coordination = raw

    # Weighted combination (renormalise over signals with available data)
    weights_raw = {
        "n1_burst":        cfg.camp_n1_weight if all_dates else 0.0,
        "n2_coamp":        cfg.camp_n2_weight,
        "n3_reuse":        cfg.camp_n3_weight,
        "n4_crosslingual": cfg.camp_n4_weight,
    }
    weight_sum = sum(weights_raw.values())
    if weight_sum > 0:
        campaign.coordination_score = sum(
            raw[k] * weights_raw[k] / weight_sum for k in raw)
    else:
        campaign.coordination_score = 0.0

    # Two-axis classification
    coord_thr = cfg.camp_coordination_threshold
    ver_thr   = cfg.camp_veracity_threshold
    if campaign.coordination_score < coord_thr:
        campaign.label = "Organic Trend"
    elif campaign.veracity is None or campaign.veracity >= ver_thr:
        campaign.label = "Information Campaign"
    else:
        campaign.label = "Disinformation Campaign"


# ---------------------------------------------------------------------------
# Campaign assigner (mirrors narrative assigner one level up)
# ---------------------------------------------------------------------------

class CampaignAssigner:
    """Assign narratives to campaigns using the same mechanism as the narrative
    assigner, but operating on Narrative objects rather than SubNarrative."""

    def __init__(self, backend, corpus, kb, llm,
                 dataset: str, det_slug: str,
                 threshold: float,
                 min_new_size: int = 2,
                 new_threshold: float = 0.70) -> None:
        from core.hierarchy.assigner import RetrievalAssigner
        # Reuse RetrievalAssigner — it works on any object with .id and .central_claim
        # We wrap Narrative objects to look like SubNarrative for the assigner
        self._kb = kb
        self._dataset = dataset
        self._det_slug = det_slug
        self._backend = backend
        self._corpus = corpus
        self._llm = llm
        self._threshold = threshold
        self._min_new = min_new_size
        self._new_thr = new_threshold
        self._campaigns: dict[str, Campaign] = {
            c.id: c for c in kb.campaigns(dataset, backend.name)
        }
        for c in self._campaigns.values():
            self._corpus.add_cluster(c.id, [c.central_claim])
        self._seq = self._max_seq()
        self._unassigned: dict[str, Narrative] = {}
        # Cache all narratives across backends for O(1) lookup in assign/synthesize
        self._nar_index: dict[str, Narrative] = {}
        for backend in ("dense", "bm25-rag", "bm25_rag",
                        "specfi-cs", "cspecfi", "context-1"):
            for n in kb.narratives(dataset, backend):
                self._nar_index.setdefault(n.id, n)

    def _max_seq(self) -> int:
        seqs = [int(cid.split("_")[1]) for cid in self._campaigns
                if cid.startswith("camp_") and cid.split("_")[1].isdigit()]
        return max(seqs) + 1 if seqs else 0

    def _synthesize(self, claims: list[str]) -> str:
        if not claims or self._llm is None:
            return max(claims, key=len) if claims else ""
        system = (
            "You are a precise analytical assistant. Given a list of related "
            "disinformation narrative claims, produce a single concise campaign "
            "central claim (one sentence, <=25 words). Output only the claim. "
            "/no_think"
        )
        user = "\n".join(f"- {c}" for c in claims)
        try:
            out = (self._llm(system, user, max_tokens=60) or "").strip()
        except TypeError:
            out = (self._llm(system, user) or "").strip()
        return out or (claims[0] if claims else "")

    def _create_campaign(self, narratives: list[Narrative]) -> Campaign:
        claims = [n.central_claim for n in narratives]
        camp = Campaign(
            id=campaign_id(self._seq),
            backend=self._backend.name,
            dataset=self._dataset,
            central_claim=self._synthesize(claims),
            narratives=[n.id for n in narratives],
            member_count=len(narratives),
            languages=sorted({l for n in narratives for l in n.languages}),
        )
        # Propagate veracity from member narratives
        ver_pairs = [(n.veracity, n.veracity_confidence or 0.5)
                     for n in narratives if n.veracity is not None]
        if ver_pairs:
            num = sum(v * c for v, c in ver_pairs)
            den = sum(c for _, c in ver_pairs)
            camp.veracity = num / den if den > 0 else None
            camp.veracity_confidence = den / len(ver_pairs)
        self._seq += 1
        self._campaigns[camp.id] = camp
        self._corpus.add_cluster(camp.id, [camp.central_claim] + claims)
        self._kb.save_campaign(camp)
        return camp

    def assign(self, nar: Narrative) -> Campaign | None:
        ranked = self._backend.rank(nar.central_claim, self._corpus, k=1)
        if ranked and ranked[0][1] >= self._threshold:
            camp = self._campaigns[ranked[0][0]]
            if nar.id not in camp.narratives:
                camp.narratives.append(nar.id)
                camp.member_count = len(camp.narratives)
                # Update central claim (all_claims was a dead variable; synthesize directly)
                camp.central_claim = self._synthesize(
                    [n.central_claim for n_id in camp.narratives
                     if (n := self._find_nar(n_id)) is not None])
                # Merge languages
                camp.languages = sorted(set(camp.languages) | set(nar.languages))
                # Update veracity
                if nar.veracity is not None:
                    existing_v = [(camp.veracity or 0.5,
                                   camp.veracity_confidence or 0.5)]
                    merged, mc = self._conf_mean(
                        existing_v + [(nar.veracity,
                                       nar.veracity_confidence or 0.5)])
                    camp.veracity = merged
                    camp.veracity_confidence = mc
                # Replace cluster in corpus
                self._corpus.remove_cluster(camp.id)
                self._corpus.add_cluster(camp.id, [camp.central_claim])
                self._kb.save_campaign(camp)
            return camp

        self._unassigned[nar.id] = nar
        return self._try_form_new(nar)

    def _find_nar(self, nar_id: str) -> Narrative | None:
        """O(1) lookup via the pre-built index; keeps KB round-trips minimal."""
        return self._nar_index.get(nar_id)

    @staticmethod
    def _conf_mean(pairs: list[tuple[float, float]]) -> tuple[float, float]:
        if not pairs:
            return 0.5, 0.5
        num = sum(v * c for v, c in pairs)
        den = sum(c for _, c in pairs)
        return (num / den if den > 0 else 0.5), (den / len(pairs))

    def _try_form_new(self, seed: Narrative) -> Campaign | None:
        candidates = [seed]
        q = self._corpus.encode_query(seed.central_claim)
        for other in self._unassigned.values():
            if other.id == seed.id:
                continue
            score = float(q @ self._corpus.encode_query(other.central_claim))
            if score >= self._new_thr:
                candidates.append(other)
        if len(candidates) < self._min_new:
            return None
        for n in candidates:
            self._unassigned.pop(n.id, None)
        return self._create_campaign(candidates)

    @property
    def campaigns(self) -> dict[str, Campaign]:
        return self._campaigns

    @property
    def unassigned_count(self) -> int:
        return len(self._unassigned)


# ---------------------------------------------------------------------------
# Main generate entry point
# ---------------------------------------------------------------------------

def _detector_slugs(detector_path: str) -> list[str]:
    if detector_path == "both":
        return ["xlm-multicw", "mdb-multicw"]
    return [os.path.basename(detector_path.rstrip("/\\"))]


def generate(
    dataset: str,
    detector_path: str,
    extractor: str,
    embedder_name: str,
    generator_key: str,
    quantization: str,
    kb: KnowledgeBase | None = None,
    *,
    cfg=None,
) -> dict:
    """Build campaigns from narratives for a given dataset.

    Returns {detector: {campaigns: N, unassigned: M, ...}}.
    """
    from config import Config
    from core.hierarchy.corpus import FactCheckCorpus
    from core.hierarchy.reclustering import ReclusteringSweep
    cfg = cfg or Config.load()
    if kb is None:
        kb = KnowledgeBase(Path("knowledge"))

    det_slugs = _detector_slugs(detector_path)
    llm = None
    if extractor != "dense":
        console.print(
            f"[bold]Loading generator[/bold] [cyan]{generator_key}[/cyan]…")
        llm = make_generator(generator_key, quantization)

    console.print(
        f"[bold]Loading embedder[/bold] [cyan]{embedder_name}[/cyan]…")
    embedder = make_embedder(embedder_name)

    summary: dict = {}
    for det_slug in det_slugs:
        narratives = kb.narratives(dataset, extractor)
        if not narratives:
            # Try all backends if no specific-backend narratives
            for backend in ("dense", "bm25-rag", "bm25_rag",
                            "specfi-cs", "cspecfi", "context-1"):
                narratives += kb.narratives(dataset, backend)
            narratives = list({n.id: n for n in narratives}.values())

        if not narratives:
            console.print(
                f"  [yellow]No narratives for {dataset}/{det_slug} — "
                f"run narrative Generate first; skipping.[/yellow]")
            continue

        from core.hierarchy.backends.bm25_rag import BM25RagBackend
        corpus = FactCheckCorpus(embedder)

        if extractor == "dense" or extractor == "bm25-rag":
            backend = BM25RagBackend()
        elif extractor == "context-1":
            from core.hierarchy.backends.context1 import Context1Backend
            backend = Context1Backend(llm, embedder,
                                      max_turns=cfg.camp_context1_max_turns,
                                      token_budget=cfg.camp_context1_token_budget)
        elif extractor in ("specfi-cs", "specfi-ccs", "cspecfi"):
            from core.hierarchy.backends.specfi_c import SpecFiCBackend
            mode = {"specfi-cs": "static", "specfi-ccs": "static-ccs",
                    "cspecfi": "continuous"}[extractor]
            if mode == "continuous":
                # cSpecFi: no NodeRAG; conditions on each narrative's own claims.
                backend = SpecFiCBackend(embedder, llm, noderag=None, mode=mode,
                                         k=cfg.camp_specfi_hypotheticals)
            else:
                # static / static-ccs: build a NodeRAG graph over the narrative
                # layer. static uses narrative central claims as raw text;
                # static-ccs uses the underlying canonized claims. Both get an
                # auto-sized parallel build pool, torn down after the build.
                from core.hierarchy.noderag import NodeRagGraph
                index_path = str(Path("knowledge") / "noderag"
                                 / f"camp_{extractor}" / dataset / det_slug)
                self_inp = Path(index_path) / "input"
                self_inp.mkdir(parents=True, exist_ok=True)
                for old in self_inp.glob("*.txt"):
                    old.unlink()
                for nar in narratives:
                    if mode == "static-ccs":
                        # underlying canonized claims of this narrative's subs
                        doc_lines = []
                        for sn_id in nar.sub_narratives:
                            sn = next((s for s in kb.sub_narratives(dataset, det_slug)
                                       if s.id == sn_id), None)
                            if sn is None:
                                continue
                            ac = kb.load_article_claims(dataset, det_slug, sn.article_name)
                            if ac:
                                doc_lines += [c for c in ac.canonized_claims if c and c.strip()]
                        doc = "\n".join(doc_lines) or nar.central_claim
                    else:
                        doc = nar.central_claim
                    (self_inp / f"{nar.id}.txt").write_text(doc, encoding="utf-8")
                graph = NodeRagGraph(
                    index_path, generate=llm, embedder=embedder,
                    build_model_key=cfg.camp_generator,
                    build_quant=cfg.camp_quantization,
                    build_context_size=getattr(cfg, "camp_context1_token_budget", 16384))
                backend = SpecFiCBackend(embedder, llm, graph, mode=mode,
                                         k=cfg.camp_specfi_hypotheticals)
        else:
            backend = BM25RagBackend()

        assigner = CampaignAssigner(
            backend, corpus, kb, llm,
            dataset=dataset, det_slug=det_slug,
            threshold=cfg.camp_assign_threshold,
            min_new_size=cfg.camp_min_new_size,
            new_threshold=cfg.camp_new_threshold,
        )

        cadence = max(0, int(cfg.camp_recluster_cadence))
        sweeps = 0

        console.print(
            f"\n[bold]{dataset}[/bold] / {det_slug}  "
            f"({len(narratives)} narratives, method={extractor})")

        with Progress(SpinnerColumn(),
                      TextColumn("[progress.description]{task.description}"),
                      BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                      console=console) as prog:
            task = prog.add_task(f"[cyan]{det_slug}[/cyan]",
                                 total=len(narratives))
            for i, nar in enumerate(narratives, 1):
                assigner.assign(nar)
                prog.advance(task)
                if cadence and i % cadence == 0:
                    sweeps += 1

        # Compute coordination signals and classify
        console.print(
            f"  Computing coordination signals for "
            f"{len(assigner.campaigns)} campaigns…")
        for camp in assigner.campaigns.values():
            compute_coordination(camp, kb, dataset, det_slug, cfg)
            kb.save_campaign(camp)

        counts = {
            "campaigns": len(assigner.campaigns),
            "unassigned_narratives": assigner.unassigned_count,
            "sweeps": sweeps,
        }
        label_dist = defaultdict(int)
        for camp in assigner.campaigns.values():
            label_dist[camp.label] += 1
        counts["labels"] = dict(label_dist)

        console.print(
            f"  campaigns={counts['campaigns']}  "
            f"unassigned={counts['unassigned_narratives']}  "
            f"labels={counts['labels']}")
        summary[det_slug] = counts

    close_generator(llm)
    return summary
