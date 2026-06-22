"""SpecFi backend — shared machinery for the SpecFi variants.

`mode` selects the variant:
  * "static"     → reproduced ORIGINAL SpecFi-CS (Upravitelev et al.):
                   NodeRAG graph built ONCE over raw article texts; community-
                   level findings become few-shot examples for HyDE generation.
                   n=10 hypotheticals (matching the paper). noderag must not be None.
  * "static-ccs" → SpecFi-CCS (Canonized Community Summaries), our variant:
                   identical to "static" EXCEPT the NodeRAG graph is built from
                   the CANONIZED CLAIMS extracted per article rather than the raw
                   article text. The conditioning source (NodeRAG community
                   findings) and ranking are the same as static; only the graph's
                   input documents differ, which is handled by the eval/gen layer.
                   noderag must not be None.
  * "continuous" → cSpecFi, our continuous variant:
                   NO NodeRAG. The query sub-narrative's own canonized claims
                   are used as few-shot conditioning context instead of NodeRAG
                   community summaries. Pass noderag=None and supply claims=[...]
                   to rank(). Continuous because conditioning updates naturally
                   as sub-narrative extraction proceeds — no graph build needed.

All variants share the same HyDE generation + max-cosine ranking. static and
static-ccs differ only in the corpus the graph is built from; continuous differs
in the conditioning source for the few-shot prompt.
"""
from __future__ import annotations

import logging

import numpy as np

from core.hierarchy.backends.base import RetrievalBackend
from core.hierarchy.noderag import NodeRagGraph

logger = logging.getLogger(__name__)

# Generic HyDE prompt:
# HyDE generation prompts, matching the original SpecFi implementation
# (XplaiNLP/SpecFi-Narrative-Retrieval, specfi_pn.py::generate_hypotheticals).
# The generated text is not shown to the user; it is only used to improve
# retrieval quality through embedding similarity.
_GEN_SYSTEM = (
    "You are a disinformation investigator. Your first step is to generate "
    "short disinformation texts that sound like actual ones. You get a "
    "disinformation narrative and return a disinformation text that aligns "
    "with that narrative. Return only 1 single text!"
)

# User prompt prefix used to build the final generation request. Few-shot
# examples (from NodeRAG for SpecFi-CS/CCS, or the sub-narrative's own claims
# for cSpecFi) are appended after this prefix as in-context conditioning.
_GEN_USER_PREFIX = (
    "You are a disinformation investigator. Given a disinformation narrative, "
    "generate a short, realistic text (such as a news excerpt, blog post, or "
    "social media post) that supports or aligns with that narrative. The text "
    "should sound plausible and could be found in the wild."
)

# Instruction prefix recommended for Qwen3-Embedding query encoding.
# Documents remain unmodified; only queries/hypotheticals are wrapped.
_INSTRUCT_TASK = (
    "Given a text as a query retrieve relevant passages that align "
    "with narratives similar to the query"
)


def _instruct_wrap(text: str) -> str:
    """Qwen3-Embedding query-side instruction prefix (documents are left plain)."""
    return f"Instruct: {_INSTRUCT_TASK}\nQuery: {text}"


class SpecFiCBackend(RetrievalBackend):
    name = "specfi"

    def __init__(
        self,
        embedder,
        generate,
        noderag: NodeRagGraph,
        k: int = 5,
        temperature: float = 1.0,
        use_instruct: bool = True,
        mode: str = "continuous",
    ) -> None:
        # Variant selector.
        if mode not in ("static", "static-ccs", "continuous"):
            raise ValueError(
                f"mode must be 'static', 'static-ccs', or 'continuous', got {mode!r}")
        self.mode = mode
        self.name = {"static": "specfi-cs",
                     "static-ccs": "specfi-ccs",
                     "continuous": "cspecfi"}[mode]

        if mode in ("static", "static-ccs") and noderag is None:
            raise ValueError(
                f"SpecFi {self.name} (mode={mode!r}) requires a NodeRagGraph instance.")
        # cSpecFi (mode='continuous') does not use NodeRAG at all; noderag=None is correct.

        self.embedder = embedder

        # Callable LLM interface:
        # generate(system_prompt, user_prompt, **kwargs)
        self.generate = generate

        # Graph-based retrieval component providing community summaries that
        # serve as retrieval conditioning context.
        self.noderag = noderag

        # Number of hypothetical documents to generate per query.
        self.k = k

        # Sampling temperature for generation diversity.
        self.temperature = temperature

        # Whether to prepend the Qwen3 embedding instruction template.
        self.use_instruct = use_instruct

        # Cache generated hypotheticals to avoid repeated LLM calls for the
        # same query during a session.
        self._cache: dict[str, list[str]] = {}

    # ---- conditioning + generation --------------------------------------

    def _conditioning(self, claim: str, claims: list[str] | None = None) -> str:
        """Return the few-shot conditioning context for HyDE generation.

        SpecFi-CS (mode='static') and SpecFi-CCS (mode='static-ccs') both query
        NodeRAG for community-level findings that serve as few-shot examples.
        They differ only in what the graph was built from (raw text vs canonized
        claims), which is decided at graph-build time, not here.

        cSpecFi (mode='continuous'): uses the query sub-narrative's own
        canonized claims as conditioning context. No NodeRAG required.
        If claims is empty, falls back to the central claim alone.
        """
        if self.mode == "continuous":
            # cSpecFi: condition on the sub-narrative's canonized claims.
            context_items = claims if claims else [claim]
            if not context_items:
                return ""
            lines = ["\n\nHere are the underlying claims of this sub-narrative:\n"]
            for c in context_items:
                lines.append(
                    f"    Narrative: {claim}\n"
                    f"    Claim: {c.replace(chr(10), ' ').strip()}\n"
                )
            return "\n".join(lines)

        # SpecFi-CS: use NodeRAG community context.
        bullets = self.noderag.community_context(claim)
        if not bullets:
            return ""
        lines = ["\n\nHere are some examples:\n"]
        for b in bullets:
            lines.append(
                f"    Narrative: {claim}\n"
                f"    Text: {b.replace(chr(10), ' ').strip()}\n"
            )
        return "\n".join(lines)

    def _hypotheticals(self, claim: str,
                       claims: list[str] | None = None) -> list[str]:
        """Generate hypothetical documents for a query claim.

        SpecFi-CS: conditioning comes from NodeRAG community context.
        cSpecFi: conditioning comes from the sub-narrative's canonized claims.
        Cache key includes both the claim and the claims list.
        """
        cache_key = (claim, tuple(claims or []))
        if cache_key in self._cache:
            return self._cache[cache_key]

        ctx = self._conditioning(claim, claims=claims).rstrip()
        user = f"{_GEN_USER_PREFIX}{ctx}\n\nNarrative: {claim}\nText:"
        out: list[str] = []

        for _ in range(self.k):
            try:
                text = (self.generate(
                    _GEN_SYSTEM, user,
                    temperature=self.temperature, max_tokens=256,
                ) or "").strip()
            except TypeError:
                text = (self.generate(_GEN_SYSTEM, user) or "").strip()
            except Exception as exc:
                logger.warning("[specfi] generation failed: %s", exc)
                text = ""
            out.append(text if len(text) > 20 else claim)

        seen: set[str] = set()
        uniq = [h for h in out if not (h in seen or seen.add(h))]
        self._cache[cache_key] = uniq or [claim]
        return self._cache[cache_key]

    # ---- ranking ---------------------------------------------------------

    def rank(self, query: str, corpus, k: int = 10,
             claims: list[str] | None = None) -> list[tuple[str, float]]:
        """Rank corpus clusters for a query.

        claims: canonized claims of the query sub-narrative (cSpecFi only).
        For SpecFi-CS this is ignored; conditioning comes from NodeRAG.
        """
        ids, mat = corpus.cluster_matrix()
        if not ids or mat.size == 0:
            return []

        hyps = self._hypotheticals(query, claims=claims)
        wrapped = [_instruct_wrap(h) if self.use_instruct else h for h in hyps]
        hyp_emb = np.asarray(self.embedder.encode(wrapped), dtype=np.float32)
        hyp_emb /= (np.linalg.norm(hyp_emb, axis=1, keepdims=True) + 1e-12)

        sims = hyp_emb @ mat.T                # (n_hyps, n_clusters)
        cluster_scores = sims.max(axis=0)     # max over hypotheticals per cluster
        order = np.argsort(cluster_scores)[::-1][:k]
        return [(ids[i], float(cluster_scores[i])) for i in order]

    def reset_cache(self) -> None:
        """Clear cached hypothetical generations."""
        self._cache.clear()