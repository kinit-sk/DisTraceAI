"""Generate step: extract sub-narratives from canonized claims.

Algorithm (per article)
-----------------------
1. Build the claim pool: article title + all non-empty canonized claims.
2. Embed the entire pool in one batch.
3. Seed selection: pick the claim with the highest sum of cosine similarities
   to every other claim in the pool (most-connected node).
4. Synthesize a central claim for the current seed via the LLM.
5. Assign every pool claim whose cosine similarity to the seed embedding is
   ≥ ``min_similarity`` to this sub-narrative (including the seed itself).
6. Remove assigned claims from the pool.
7. If the pool has fewer than ``min_claims`` remaining, stop.  Otherwise go
   to step 3 with the reduced pool.
8. Persist each sub-narrative to
   ``knowledge/<dataset>/<detector>/sub-narratives/<article_name>_sn<n>.json``.

Articles that already have sub-narratives in the KB are skipped (idempotent).
Articles without any canonized claims are skipped with a warning.

KB layout
---------
knowledge/<dataset>/<detector>/sub-narratives/<article_name>_sn<n>.json
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    MofNCompleteColumn, TimeElapsedColumn,
)

from core.knowledge_base import KnowledgeBase, DATASET_POLYNARRATIVE, DATASET_FAKECTI
from core.models import make_embedder, make_generator, encode_with_backoff, close_generator
from core.structures import SubNarrative

logger  = logging.getLogger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_SYSTEM_CENTRAL = (
    "You are an expert analyst summarising disinformation narratives. /no_think"
)

_USER_CENTRAL_TMPL = (
    "The following claims all relate to the same sub-narrative topic.\n"
    "Write a single concise English sentence (≤ 25 words) that captures "
    "the shared central claim across all of them.\n\n"
    "STRICT RULES:\n"
    "- Output ONLY the central claim sentence. No explanation, no commentary.\n"
    "- Write in ENGLISH regardless of the source language.\n\n"
    "Claims:\n{claims}"
)


def _synthesize_central_claim(llm, claims: list[str]) -> str:
    """Ask the LLM to synthesize a single central claim from a list of claims."""
    bullet_list = "\n".join(f"- {c}" for c in claims)
    user   = _USER_CENTRAL_TMPL.format(claims=bullet_list)
    result = llm(_SYSTEM_CENTRAL, user, max_tokens=80)
    return result.strip() if result else claims[0]


# ---------------------------------------------------------------------------
# Greedy sub-narrative extraction for a single article
# ---------------------------------------------------------------------------

def _cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return the full N×N cosine-similarity matrix for a batch of embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normed = embeddings / norms
    return normed @ normed.T


def extract_sub_narratives(
    claims: list[str],
    embeddings: np.ndarray,
    llm,
    article_name: str,
    dataset: str,
    detector: str,
    min_similarity: float,
    min_claims: int,
) -> list[SubNarrative]:
    """Greedy, seed-first extraction of sub-narratives from a claim pool.

    Parameters
    ----------
    claims:
        Ordered list of canonized claim strings (title first, then claims).
    embeddings:
        Corresponding L2-normalised embeddings, shape ``(len(claims), dim)``.
    llm:
        Generator callable used to synthesize central claims.
    article_name, dataset, detector:
        Metadata written into each ``SubNarrative`` record.
    min_similarity:
        Cosine threshold for assigning a claim to the current seed cluster.
    min_claims:
        Minimum pool size to attempt another sub-narrative.  When the remaining
        pool drops below this, extraction stops and the remainder is discarded.

    Returns
    -------
    list[SubNarrative]
        Sub-narratives in extraction order, not yet persisted.
    """
    n = len(claims)
    if n < min_claims:
        return []

    sim = _cosine_matrix(embeddings)          # (N, N)
    active = list(range(n))                   # indices of claims still in pool
    sub_narratives: list[SubNarrative] = []
    sn_counter = 0

    while len(active) >= min_claims:
        # --- seed: most-connected claim in the current pool ---
        sub_sim = sim[np.ix_(active, active)]  # pool × pool sub-matrix
        row_sums = sub_sim.sum(axis=1)
        seed_local = int(np.argmax(row_sums))
        seed_global = active[seed_local]

        # --- assign all pool claims above threshold to this cluster ---
        seed_sims = sim[seed_global, active]   # similarity of seed to every pool claim
        assigned_local  = [i for i, s in enumerate(seed_sims) if s >= min_similarity]
        assigned_global = [active[i] for i in assigned_local]

        if len(assigned_global) < min_claims:
            # Seed is too isolated; no valid cluster can form — stop.
            break

        cluster_claims = [claims[i] for i in assigned_global]

        # --- synthesize central claim ---
        central = _synthesize_central_claim(llm, cluster_claims)

        sn_id = f"{article_name}_sn{sn_counter}"
        sub_narratives.append(SubNarrative(
            id=sn_id,
            article_name=article_name,
            dataset=dataset,
            detector=detector,
            central_claim=central,
            claims=cluster_claims,
        ))
        sn_counter += 1

        # --- remove assigned claims from pool ---
        assigned_set = set(assigned_global)
        active = [i for i in active if i not in assigned_set]

    return sub_narratives


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

def _detector_slugs(detector_path: str) -> list[str]:
    """Expand 'both' into the two canonical slugs; otherwise derive the slug."""
    if detector_path == "both":
        return ["xlm-multicw", "mdb-multicw"]
    return [__import__("os").path.basename(detector_path.rstrip("/\\"))]


def _process_dataset(
    dataset_slug: str,
    detector_slug: str,
    embedder,
    llm,
    kb: KnowledgeBase,
    min_similarity: float,
    min_claims: int,
) -> dict:
    """Extract sub-narratives for all articles in one dataset/detector pair.

    Returns ``{"articles": int, "skipped": int, "sub_narratives": int}``.
    """
    all_acs = kb.all_article_claims(dataset_slug, detector_slug)
    if not all_acs:
        console.print(
            f"  [dim]No articles found for {dataset_slug}/{detector_slug} — skipping.[/dim]"
        )
        return {}

    processed = skipped = total_sns = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]{dataset_slug}/{detector_slug}[/cyan]",
            total=len(all_acs),
        )

        for ac in all_acs:
            if kb.sub_narratives_exist(dataset_slug, detector_slug, ac.article_name):
                skipped += 1
                progress.advance(task)
                continue

            # Build claim pool: title first, then non-empty canonized claims.
            title_entry = ac.title.strip() if ac.title else ""
            raw_claims  = [c for c in ac.canonized_claims if c and c.strip()]
            if not raw_claims:
                logger.warning("[sub_nar] %s has no canonized claims — skipping", ac.article_name)
                progress.advance(task)
                continue

            pool = ([title_entry] if title_entry else []) + raw_claims

            embeddings = encode_with_backoff(embedder, pool)   # (N, dim)

            sns = extract_sub_narratives(
                claims=pool,
                embeddings=embeddings,
                llm=llm,
                article_name=ac.article_name,
                dataset=dataset_slug,
                detector=detector_slug,
                min_similarity=min_similarity,
                min_claims=min_claims,
            )

            for sn in sns:
                kb.save_sub_narrative(sn)

            total_sns += len(sns)
            processed += 1
            progress.advance(task)

    total_in_kb = len(kb.sub_narratives(dataset_slug, detector_slug))
    console.print(
        f"  {detector_slug}: new={total_sns}  skipped={skipped}"
        f"  total_in_kb={total_in_kb}"
    )
    return {"articles_processed": processed, "skipped": skipped,
            "new_sub_narratives": total_sns, "total_in_kb": total_in_kb}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate(
    detector_path: str,
    embedder_name: str,
    generator_key: str,
    kb: KnowledgeBase | None = None,
    min_similarity: float = 0.45,
    min_claims: int = 2,
) -> dict:
    """Extract sub-narratives from canonized claims in PolyNarrative and FakeCTI.

    Parameters
    ----------
    detector_path:
        Path of the CW detector whose canonized output to process, e.g.
        ``"models/xlm-multicw"``, ``"models/mdb-multicw"``, or ``"both"``.
    embedder_name:
        HuggingFace model name for the SentenceTransformer embedder.
    generator_key:
        Key from ``_CATALOGUE`` for the LLM used to synthesize central claims.
    kb:
        Knowledge-base instance; defaults to ``KnowledgeBase("knowledge")``.
    min_similarity:
        Cosine threshold for claim assignment (default 0.45).
    min_claims:
        Minimum pool size to form a sub-narrative (default 2).

    Returns
    -------
    dict
        ``{dataset_slug: {detector_slug: {"articles", "skipped", "sub_narratives"}}}``.
    """
    if kb is None:
        kb = KnowledgeBase(Path("knowledge"))

    detector_slugs = _detector_slugs(detector_path)

    console.print(
        f"\n[bold]Loading embedder[/bold] [cyan]{embedder_name}[/cyan]…"
    )
    embedder = make_embedder(embedder_name)

    console.print(
        f"[bold]Loading generator[/bold] [cyan]{generator_key}[/cyan] "
    )
    llm = make_generator(generator_key)

    summary: dict = {}
    for dataset_slug in [DATASET_POLYNARRATIVE, DATASET_FAKECTI]:
        for detector_slug in detector_slugs:
            console.print(
                f"\n[bold]{dataset_slug}[/bold]  [dim](detector: {detector_slug})[/dim]"
            )
            result = _process_dataset(
                dataset_slug, detector_slug, embedder, llm, kb,
                min_similarity, min_claims,
            )
            if result:
                summary.setdefault(dataset_slug, {})[detector_slug] = result

    close_generator(llm)
    return summary
