"""Merge-or-cluster narrative assignment via sklearn agglomerative clustering.

This replaces the earlier ``RetrievalAssigner`` whose ``_try_form_new_narrative``
demanded ``min_new_size`` mutually-similar items above a high cosine threshold
to seed a new narrative. On EUvsDisinfo (and any corpus where each article
covers a distinct topic) that condition almost never fired, producing zero
narratives → zero campaigns.

Plan §4.3 / §4.5: *"Matched against existing → merged if sufficiently similar
→ otherwise added to an unassigned pool. The unassigned pool is periodically
clustered to discover new narratives."* — this module implements that
literally with sklearn ``AgglomerativeClustering``:

1. ``assign(sn)`` ranks the existing narratives via the configured retrieval
   backend; on a hit above ``threshold`` the sub-narrative is merged in
   exactly as before.
2. On a miss the sub-narrative joins the unassigned pool. The pool is then
   re-clustered: any cluster with ≥ ``min_new_size`` members above the
   ``new_threshold`` distance bound becomes a new narrative; remaining
   members stay in the pool.

The "periodic" framing is satisfied by re-clustering on every miss (cheap:
pool is bounded by the not-yet-assigned tail of the stream, typically small).
Callers can additionally invoke ``flush_pool()`` at the end of a run to drop
any straggler clusters that only meet the size threshold at the very end.

Same Narrative metadata accumulation as the old assigner: languages,
veracity-conf-weighted mean, member_count, synthesized central claim.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from core.structures import SubNarrative, Narrative
from core.ids import narrative_id

logger = logging.getLogger(__name__)


class AgglomerativeGrouper:
    """Streaming narrative assigner: merge-or-pool, with sklearn clustering of the pool.

    Constructor parameters mirror the old ``RetrievalAssigner`` so callers
    (``gen_narratives._process_dataset`` etc.) can swap one for the other
    without a wider refactor. The two additions are ``linkage`` (forwarded to
    sklearn) and ``recluster_on_pool_growth`` (set False if you only want the
    explicit ``flush_pool()`` to trigger clustering, e.g. in a unit test).
    """

    def __init__(
        self,
        backend,
        corpus,
        kb,
        llm,
        dataset: str,
        detector: str,
        threshold: float,
        min_new_narrative_size: int = 2,
        new_narrative_threshold: float = 0.55,
        linkage: str = "average",
        recluster_on_pool_growth: bool = True,
    ) -> None:
        self.backend = backend
        self.corpus = corpus
        self.kb = kb
        self.generate = llm

        # Scope: one (dataset, detector) pair per instance.
        self.dataset = dataset
        self.detector = detector

        # Existing-narrative merge threshold (cosine).
        self.threshold = threshold
        # Minimum members for the pool to spawn a new narrative.
        self.min_new_narrative_size = min_new_narrative_size
        # Pool-clustering threshold (cosine). Converted to a distance below.
        self.new_narrative_threshold = new_narrative_threshold
        self.linkage = linkage
        self._recluster_on_pool_growth = recluster_on_pool_growth

        # Load existing narratives into the corpus so subsequent ranks see them.
        self.narratives: dict[str, Narrative] = {
            n.id: n for n in kb.narratives(dataset, backend.name)
        }
        for n in self.narratives.values():
            self.corpus.add_cluster(n.id, [n.central_claim])
        self._seq = self._max_seq()

        # In-memory caches for the metadata-aggregation step.
        self._sn_index: dict[str, SubNarrative] = {
            sn.id: sn for sn in kb.sub_narratives(dataset, detector)
        }
        # Pool of unassigned sub-narratives keyed by id.
        self._unassigned: dict[str, SubNarrative] = {}

    # ---- id sequencing ---------------------------------------------------
    def _max_seq(self) -> int:
        seqs = [
            int(nid.split("_")[1])
            for nid in self.narratives
            if nid.startswith("nar_") and nid.split("_")[1].isdigit()
        ]
        return max(seqs) + 1 if seqs else 0

    # ---- metadata aggregation -------------------------------------------
    def _member_records(self, sn_ids: list[str]) -> list[SubNarrative]:
        out = []
        for sid in sn_ids:
            sn = self._sn_index.get(sid)
            if sn is None:
                sn = self.kb.get_sub_narrative(self.dataset, self.detector, sid)
                if sn is not None:
                    self._sn_index[sid] = sn
            if sn is not None:
                out.append(sn)
        return out

    def _article_language(self, article_name: str) -> str | None:
        """Cached lookup of a member article's source_language."""
        cache = getattr(self, "_lang_cache", None)
        if cache is None:
            cache = self._lang_cache = {}
        if article_name in cache:
            return cache[article_name]
        lang = None
        ac = self.kb.load_article_claims(self.dataset, self.detector, article_name)
        if ac is not None:
            md = getattr(ac, "metadata", None) or {}
            lang = md.get("source_language")
        if lang is None:
            amap = getattr(self, "_article_lang_map", None)
            if amap is None:
                amap = self._article_lang_map = {
                    a.id: a.source_language for a in self.kb.articles(self.dataset)
                }
            lang = amap.get(article_name)
        cache[article_name] = lang
        return lang

    def _accumulate_metadata(self, nar: Narrative) -> None:
        members = self._member_records(nar.sub_narratives)
        nar.member_count = len(nar.sub_narratives)
        langs: set[str] = set()
        for sn in members:
            lang = self._article_language(sn.article_name)
            if lang:
                langs.add(lang)
        nar.languages = sorted(langs)

        # Confidence-weighted veracity mean over members that have a verdict.
        num = den = 0.0
        confs: list[float] = []
        for sn in members:
            if sn.veracity is not None:
                w = sn.veracity_confidence if sn.veracity_confidence is not None else 1.0
                num += sn.veracity * w
                den += w
                if sn.veracity_confidence is not None:
                    confs.append(sn.veracity_confidence)
        nar.veracity = (num / den) if den > 0 else None
        nar.veracity_confidence = (sum(confs) / len(confs)) if confs else None

    # ---- central-claim synthesis ----------------------------------------
    # Defensive bound: the joined ``user`` prompt is ``"\n".join("- " + claim)``
    # over every member's central claim. On corpora where assignment merges
    # aggressively (e.g. polynarrative under cspecfi / dense), a narrative can
    # accrete hundreds of members; at ~30 tokens/claim that pushes the prompt
    # past the 16K context window of typical local models (this was the real
    # 1250-th-sample crash: VLLMValidationError 16385>16384). We cap the input
    # at ``MAX_CLAIMS_FOR_SYNTH`` and prefer the LONGEST member claims, on the
    # assumption that they carry the most truth-conditional content.
    MAX_CLAIMS_FOR_SYNTH = 40

    def _synthesize_central_claim(self, claims: list[str]) -> str:
        if not claims:
            return ""
        if self.generate is None:
            return max(claims, key=len)

        sample = claims
        if len(sample) > self.MAX_CLAIMS_FOR_SYNTH:
            # Deterministic sample: longest claims first, capped. Order doesn't
            # matter to the LLM's summary task; determinism does, so the same
            # cluster always produces the same synthesized claim across runs.
            sample = sorted(claims, key=len, reverse=True)[: self.MAX_CLAIMS_FOR_SYNTH]

        system = (
            "You are a precise analytical assistant. Given a list of related "
            "misinformation sub-claims, produce a single concise central claim "
            "(one sentence, <=25 words) that best captures the shared narrative "
            "across all of them. Output only the central claim. /no_think"
        )
        user = "\n".join(f"- {c}" for c in sample)
        out = self.generate(system, user, max_tokens=60)
        return (out or "").strip() or sample[0]

    # ---- narrative creation ---------------------------------------------
    def _create_narrative(self, sub_narratives: list[SubNarrative]) -> Narrative:
        claims = [sn.central_claim for sn in sub_narratives]
        nar = Narrative(
            id=narrative_id(self._seq),
            backend=self.backend.name,
            dataset=self.dataset,
            central_claim=self._synthesize_central_claim(claims),
            sub_narratives=[sn.id for sn in sub_narratives],
        )
        self._seq += 1
        self.narratives[nar.id] = nar
        for sn in sub_narratives:
            self._sn_index.setdefault(sn.id, sn)
        self.corpus.add_cluster(nar.id, [nar.central_claim] + claims)
        self._accumulate_metadata(nar)
        self.kb.save_narrative(nar)
        return nar

    # ---- pool re-clustering ---------------------------------------------
    def _cluster_pool(self, *, force: bool = False) -> list[Narrative]:
        """Run agglomerative clustering over the unassigned pool.

        Promotes any resulting cluster with at least ``min_new_narrative_size``
        members into a new narrative. Members of promoted clusters leave the
        pool; the rest stay.

        ``force`` forces a single pass even when the pool would otherwise be
        skipped (used by ``flush_pool``).
        """
        pool = list(self._unassigned.values())
        if len(pool) < self.min_new_narrative_size:
            return []

        # Encode pool central claims once. Re-using the corpus's query
        # encoder keeps the embedding space consistent with assign-time ranks.
        # ``encode_query`` returns L2-normalised vectors, so squared euclidean
        # distance is 2 (1 - cosine) — i.e. monotone with cosine distance.
        vecs = np.stack([self.corpus.encode_query(sn.central_claim) for sn in pool])

        # Convert cosine THRESHOLD (similarity) → cosine DISTANCE.
        # sklearn AgglomerativeClustering with metric='cosine' treats pairs
        # with distance < `distance_threshold` as mergeable; we want to merge
        # pairs with similarity >= new_narrative_threshold.
        distance_threshold = max(0.0, 1.0 - float(self.new_narrative_threshold))

        try:
            from sklearn.cluster import AgglomerativeClustering
        except ImportError as exc:                                  # pragma: no cover
            logger.error("[grouper] sklearn missing: %s", exc)
            return []

        try:
            labels = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage=self.linkage,
                distance_threshold=distance_threshold,
            ).fit_predict(vecs)
        except Exception as exc:                                    # pragma: no cover - runtime
            logger.warning("[grouper] agglomerative clustering failed (%s); "
                           "skipping new-narrative seeding this round", exc)
            return []

        # Group pool members by cluster label.
        groups: dict[int, list[SubNarrative]] = {}
        for sn, lbl in zip(pool, labels):
            groups.setdefault(int(lbl), []).append(sn)

        created: list[Narrative] = []
        for members in groups.values():
            if len(members) < self.min_new_narrative_size and not force:
                continue
            # Reject clusters that meet the size floor only by including
            # members below the new-threshold neighbourhood. (sklearn's
            # distance_threshold guarantees the per-cluster maximum link
            # distance is ≤ threshold under 'single' linkage, and tighter
            # under 'complete'/'average'.) We additionally drop singletons
            # in ``force`` mode to avoid promoting every leftover to its own
            # one-member narrative.
            if len(members) < max(2, self.min_new_narrative_size) and force:
                continue
            nar = self._create_narrative(members)
            created.append(nar)
            for sn in members:
                self._unassigned.pop(sn.id, None)
        return created

    def flush_pool(self) -> list[Narrative]:
        """Final pass: promote any remaining pool cluster meeting the size floor."""
        return self._cluster_pool(force=True)

    # Member-count milestones at which the central claim is re-synthesized
    # when a sub-narrative MERGES into an existing narrative. Below 5 we
    # re-summarize on every merge (the summary is still volatile); past that
    # we only re-summarize when the cluster has roughly doubled, since at
    # 100+ members one more sub-narrative is statistically a no-op on the
    # summary's wording.
    _RESYNTH_MILESTONES = frozenset({2, 3, 4, 5, 10, 25, 50, 100, 250, 500,
                                     1000, 2500, 5000, 10000})

    # ---- assignment ------------------------------------------------------
    def assign(self, sn: SubNarrative) -> Narrative | None:
        """Merge ``sn`` into the best existing narrative if above threshold;
        otherwise pool it and run a re-clustering pass over the pool."""
        self._sn_index.setdefault(sn.id, sn)
        ranked = self.backend.rank(sn.central_claim, self.corpus, k=1)

        if ranked and ranked[0][1] >= self.threshold:
            nar = self.narratives[ranked[0][0]]
            if sn.id not in nar.sub_narratives:
                nar.sub_narratives.append(sn.id)
                new_size = len(nar.sub_narratives)

                # Only re-synthesize at milestones (see _RESYNTH_MILESTONES).
                # Skipping the re-synth still keeps the cluster's pooled
                # embedding fresh below, so retrieval quality doesn't degrade.
                if new_size in self._RESYNTH_MILESTONES:
                    claims = [m.central_claim
                              for m in self._member_records(nar.sub_narratives)]
                    nar.central_claim = self._synthesize_central_claim(claims)
                    # Refresh the cluster's pooled embedding with the new
                    # central claim + all member claims.
                    self.corpus.remove_cluster(nar.id)
                    self.corpus.add_cluster(nar.id, [nar.central_claim] + claims)

                self._accumulate_metadata(nar)
            self.kb.save_narrative(nar)
            return nar

        # No match → pool, then re-cluster.
        self._unassigned[sn.id] = sn
        if self._recluster_on_pool_growth:
            created = self._cluster_pool()
            if created:
                # Return the narrative the seed ended up in, if any (so the
                # caller's stats are accurate).
                for nar in created:
                    if sn.id in nar.sub_narratives:
                        return nar
        return None

    def remove_narrative(self, nar_id: str) -> None:
        """Delete a narrative everywhere: in-memory cache, corpus, and KB."""
        self.narratives.pop(nar_id, None)
        self.corpus.remove_cluster(nar_id)
        self.kb.delete_narrative(self.dataset, self.backend.name, nar_id)

    @property
    def unassigned_count(self) -> int:
        return len(self._unassigned)


# Back-compat alias — old code that imported `RetrievalAssigner` directly
# from this module path still works during the transition. New code should
# import `AgglomerativeGrouper` explicitly.
RetrievalAssigner = AgglomerativeGrouper
