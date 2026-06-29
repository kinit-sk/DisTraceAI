"""Generate step: canonize check-worthy claims from PolysNarrative and FakeCTI.

For each article that already has extracted CW claims the selected LLM
decontextualizes and translates every claim into English in a single prompt.
The result is stored in the same article JSON file alongside the original
claims list as a new ``canonized_claims`` list (always overwritten):

    knowledge/<dataset>/<cw-detector>/articles/<article-name>.json
    {
        ...,
        "claims": [{sentence, sentence_index}, ...],
        "canonized_claims": ["<English decontextualized claim>", ...]
    }

``canonized_claims[i]`` corresponds to ``claims[i]``.  If the LLM returns
nothing for a claim the slot is left as an empty string so indices stay
aligned.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    MofNCompleteColumn, TimeElapsedColumn,
)

from core.knowledge_base import (
    KnowledgeBase, DATASET_POLYNARRATIVE, DATASET_FAKECTI, DATASET_EUVSDISINFO,
)
from core.models import make_generator, close_generator

logger  = logging.getLogger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# Canonization prompt
# ---------------------------------------------------------------------------
_SYSTEM = (
    "You are a linguistics expert performing claim decontextualization "
    "and translation. /no_think"
)

_USER_TMPL = (
    "Decontextualize the following claim and translate it to English.\n"
    "Guidelines:\n"
    "- Replace unclear pronouns or references with explicit entities.\n"
    "- Reformat so the claim requires no other context to be understood.\n"
    "- Example: 'It started on Monday.' → 'The elections started on Monday.'\n\n"
    "STRICT RULES:\n"
    "- Output ONLY the decontextualized English claim. No explanation, no commentary.\n"
    "- Always write in ENGLISH regardless of the source language.\n"
    "- Maximum 40 words.\n\n"
    "Claim:\n{claim}"
)


def _canonize_claim(llm, claim: str) -> str:
    """Run the canonization prompt on a single claim; return stripped output."""
    user   = _USER_TMPL.format(claim=claim.strip())
    result = llm(_SYSTEM, user, max_tokens=128)
    return result.strip() if result else ""


def _detector_slug(detector_path: str) -> str:
    """Derive the KB detector slug from a detector path.

    Mirrors ``CheckWorthinessDetector.slug``:
    ``"Models/xlm-multicw"`` → ``"xlm-multicw"``.
    """
    return os.path.basename(detector_path.rstrip("/\\"))


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

def _process_dataset(
    dataset_slug: str,
    detector_slug: str,
    llm,
    kb: KnowledgeBase,
) -> dict:
    """Canonize all claims for *detector_slug* under *dataset_slug*.

    Returns a summary dict::

        {"articles": int, "claims": int}
    """
    all_acs = kb.all_article_claims(dataset_slug, detector_slug)
    if not all_acs:
        console.print(
            f"  [dim]No articles found for {dataset_slug}/{detector_slug} — skipping.[/dim]"
        )
        return {}

    total_claims = 0

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
            canonized: list[str] = []
            for cw in ac.claims:
                canon = _canonize_claim(llm, cw.sentence)
                canonized.append(canon)
                total_claims += 1

            # Read-modify-write the existing JSON so other fields are
            # preserved and only canonized_claims is updated.
            path = kb._claims_path(dataset_slug, detector_slug, ac.article_name)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[canon] could not read %s: %s", path, exc)
                progress.advance(task)
                continue

            data["canonized_claims"] = canonized
            try:
                path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                logger.warning("[canon] could not write %s: %s", path, exc)

            progress.advance(task)

    console.print(
        f"  {detector_slug}: articles={len(all_acs)}  canonized_claims={total_claims}"
    )
    return {"articles": len(all_acs), "claims": total_claims}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def canonize(
    detector_path: str,
    generator_key: str,
    kb: KnowledgeBase | None = None,
) -> dict:
    """Decontextualize and translate CW claims produced by *detector_path*.

    Parameters
    ----------
    detector_path:
        Path value of the detector whose KB output to canonize, e.g.
        ``"Models/xlm-multicw"``, ``"Models/mdb-multicw"``, or ``"both"``
        to process claims from both detectors in a single LLM session.
        The slug is derived as ``os.path.basename(detector_path)``.
    generator_key:
        One of the six supported model keys:
        ``qwen3.5-2b`` / ``qwen3.5-4b`` / ``qwen3.5-9b`` /
        ``gemma4-e2b`` / ``gemma4-e4b`` / ``gemma4-12b``.

        Precision is fixed by the active backend (BF16 on vLLM, Q8_0 on
        llama-cpp) per plan §4 — no precision argument is exposed.
    kb:
        Knowledge-base instance.  Defaults to ``KnowledgeBase("knowledge")``.

    Returns
    -------
    dict
        ``{dataset_slug: {detector_slug: {"articles": int, "claims": int}}}``
    """
    if kb is None:
        kb = KnowledgeBase(Path("knowledge"))

    # "both" expands to all supported detectors; otherwise derive slug normally.
    _ALL_SLUGS = ["xlm-multicw", "mdb-multicw"]
    if detector_path == "both":
        detector_slugs = _ALL_SLUGS
    else:
        detector_slugs = [_detector_slug(detector_path)]

    console.print(
        f"\n[bold]Loading[/bold] [cyan]{generator_key}[/cyan]"
    )
    # Backend picks precision: vLLM ignores quant; llama-cpp defaults to Q8_0.
    llm = make_generator(generator_key)

    summary: dict = {}
    # Plan: each per-step Generate runs over every dataset present in the KB,
    # so the user can see interim EUvsDisinfo results without going through
    # Generate Dataset's full pipeline. _process_dataset is idempotent and
    # a no-op when the dataset+detector pair has nothing to canonize.
    dataset_slugs = [DATASET_POLYNARRATIVE, DATASET_FAKECTI]
    if kb.articles(DATASET_EUVSDISINFO):
        dataset_slugs.append(DATASET_EUVSDISINFO)
    for dataset_slug in dataset_slugs:
        for detector_slug in detector_slugs:
            console.print(f"\n[bold]{dataset_slug}[/bold]  [dim](detector: {detector_slug})[/dim]")
            ds_summary = _process_dataset(dataset_slug, detector_slug, llm, kb)
            if ds_summary:
                summary.setdefault(dataset_slug, {})[detector_slug] = ds_summary

    close_generator(llm)
    return summary
