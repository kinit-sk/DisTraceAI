"""Merge-or-create narrative assignment (narrative extraction, step 5).

For each sub-narrative the configured retrieval backend ranks existing
narratives; the top match above the assignment threshold merges. Sub-narratives
that do not match are placed into an unassigned pool; a new narrative is created
only when enough mutually-similar pooled sub-narratives accumulate.

Scope
-----
An assigner instance is bound to one ``(dataset, detector)`` pair: it only ever
sees that pair's sub-narratives, and writes narratives under
``narratives/<dataset>/<backend>/``. This keeps polynarrative and fake-cti pools
separate and lets ``kb.get_sub_narrative`` / ``kb.sub_narratives`` be resolved
with the correct scope (both require dataset + detector).

Metadata accumulation
---------------------
A narrative is "a central claim anchored in relevant sub-narratives enriched with
metadata accumulated from underlying sub-narratives." On every merge / creation
the assigner recomputes, from the member SubNarrative records:
  * the synthesized central claim,
  * the set of member source languages (feeds the later N4 coordination signal),
  * a confidence-weighted mean veracity (+ mean confidence), ignoring members
    without a verdict,
  * the member count.
"""
from __future__ import annotations

import logging

from core.structures import SubNarrative, Narrative
from core.ids import narrative_id

logger = logging.getLogger(__name__)


class RetrievalAssigner:
    def __init__(
        self,
        backend,
        corpus,
        kb,
        llm,
        dataset: str,
        detector: str,
        threshold: float,
        min_new_narrative_size: int = 3,
        new_narrative_threshold: float = 0.75,
    ) -> None:
        self.backend = backend
        self.corpus = corpus
        self.kb = kb
        self.generate = llm

        # Scope: this assigner only touches one dataset/detector pair.
        self.dataset = dataset
        self.detector = detector

        # Existing-narrative assignment threshold.
        self.threshold = threshold
        # Min mutually-similar pooled sub-narratives required to seed a new one.
        self.min_new_narrative_size = min_new_narrative_size
        # Cosine threshold used while clustering the unassigned pool.
        self.new_narrative_threshold = new_narrative_threshold

        # Load existing narratives for THIS dataset/backend into the corpus.
        self.narratives: dict[str, Narrative] = {
            n.id: n for n in kb.narratives(dataset, backend.name)
        }
        for n in self.narratives.values():
            self.corpus.add_cluster(n.id, [n.central_claim])

        self._seq = self._max_seq()

        # Fast lookup of member SubNarrative records (for metadata aggregation).
        # Scoped to this dataset/detector.
        self._sn_index: dict[str, SubNarrative] = {
            sn.id: sn for sn in kb.sub_narratives(dataset, detector)
        }

        # Buffer of sub-narratives that do not yet belong to any narrative.
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
        """Resolve member IDs to SubNarrative records, preferring the in-memory
        index and falling back to the KB (then caching)."""
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
        """Resolve a member article's source language, cached.

        Prefers ArticleClaims.metadata['source_language'] (dataset/detector
        scoped, cheap); falls back to the Article record by id. Results are
        memoised in ``self._lang_cache``.
        """
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
            # Fall back to the Article record (article_name == article id).
            amap = getattr(self, "_article_lang_map", None)
            if amap is None:
                # Scope to this dataset only — avoids loading all articles
                amap = self._article_lang_map = {
                    a.id: a.source_language
                    for a in self.kb.articles(self.dataset)}
            lang = amap.get(article_name)
        cache[article_name] = lang
        return lang

    def _accumulate_metadata(self, nar: Narrative) -> None:
        """Recompute languages / veracity / member_count from current members."""
        members = self._member_records(nar.sub_narratives)
        nar.member_count = len(nar.sub_narratives)

        # Languages: the source article's language for each member. The language
        # lives on the Article record (source_language) and, for converter-built
        # data, also in ArticleClaims.metadata["source_language"]; try both.
        langs = set()
        for sn in members:
            lang = self._article_language(sn.article_name)
            if lang:
                langs.add(lang)
        nar.languages = sorted(langs)

        # Veracity: confidence-weighted mean over members that have a verdict.
        num = den = 0.0
        confs = []
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
    def _synthesize_central_claim(self, claims: list[str]) -> str:
        if not claims:
            return ""
        if self.generate is None:
            # No generator (e.g. a pure-dense run without an LLM loaded): use the
            # longest member claim as a representative central claim. Synthesis
            # improves quality, so callers normally pass a generator.
            return max(claims, key=len)
        system = (
            "You are a precise analytical assistant. Given a list of related "
            "misinformation sub-claims, produce a single concise central claim "
            "(one sentence, <=25 words) that best captures the shared narrative "
            "across all of them. Output only the central claim. /no_think"
        )
        user = "\n".join(f"- {c}" for c in claims)
        out = self.generate(system, user, max_tokens=60)
        return (out or "").strip() or claims[0]

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
        for sn in sub_narratives:        # keep the index warm
            self._sn_index.setdefault(sn.id, sn)
        # Anchor the cluster with the synthesized claim, then the member claims,
        # so BM25 / mean-pooled dense retrieval retain the full signal.
        self.corpus.add_cluster(nar.id, [nar.central_claim] + claims)
        self._accumulate_metadata(nar)
        self.kb.save_narrative(nar)
        return nar

    def _try_form_new_narrative(self, seed: SubNarrative) -> Narrative | None:
        """If enough pooled sub-narratives are similar to the seed, promote them."""
        candidates = [seed]
        seed_vec = self.corpus.encode_query(seed.central_claim)
        for other in self._unassigned.values():
            if other.id == seed.id:
                continue
            score = float(seed_vec @ self.corpus.encode_query(other.central_claim))
            if score >= self.new_narrative_threshold:
                candidates.append(other)

        if len(candidates) < self.min_new_narrative_size:
            return None
        for sn in candidates:
            self._unassigned.pop(sn.id, None)
        return self._create_narrative(candidates)

    # ---- assignment ------------------------------------------------------
    def assign(self, sn: SubNarrative) -> Narrative | None:
        """Merge ``sn`` into the top-ranked existing narrative if above
        threshold; otherwise pool it and possibly seed a new narrative."""
        self._sn_index.setdefault(sn.id, sn)
        ranked = self.backend.rank(sn.central_claim, self.corpus, k=1)

        if ranked and ranked[0][1] >= self.threshold:
            nar = self.narratives[ranked[0][0]]
            if sn.id not in nar.sub_narratives:
                nar.sub_narratives.append(sn.id)
                claims = [m.central_claim
                          for m in self._member_records(nar.sub_narratives)]
                nar.central_claim = self._synthesize_central_claim(claims)
                # Replace the cluster's entries so future ranking uses the
                # updated centroid (synthesized claim anchors, members follow).
                self.corpus.remove_cluster(nar.id)
                self.corpus.add_cluster(nar.id, [nar.central_claim] + claims)
                self._accumulate_metadata(nar)
            self.kb.save_narrative(nar)
            return nar

        # No match → pool, then try to seed a new narrative.
        self._unassigned[sn.id] = sn
        return self._try_form_new_narrative(sn)

    def remove_narrative(self, nar_id: str) -> None:
        """Delete a narrative everywhere: in-memory cache, corpus, and KB."""
        self.narratives.pop(nar_id, None)
        self.corpus.remove_cluster(nar_id)
        self.kb.delete_narrative(self.dataset, self.backend.name, nar_id)

    @property
    def unassigned_count(self) -> int:
        return len(self._unassigned)
