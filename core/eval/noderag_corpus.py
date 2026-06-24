"""Export KB articles as NodeRAG input documents.

Used by ``NodeRagGraph.build()`` to auto-populate an empty ``input/`` directory
from the DisTraceAI knowledge base when a SpecFi graph is built in *generate*
mode (the eval path pre-populates ``input/`` itself and never calls this).

The ``repr`` parameter selects what each per-article document contains:

  * "text"      → the article's check-worthy claim sentences joined (SpecFi-CS):
                  the closest available proxy for the raw article text.
  * "canonized" → the article's canonized claims joined (SpecFi-CCS): one doc
                  per article built from decontextualised English claims, so
                  NodeRAG communities form over claim-level content.

One file per article is written as ``<article_name>.txt``. Returns the number
of files emitted.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def export(kb_root: Path, out_dir: Path, lang: str | None = None,
           detector: str | None = None, *, repr: str = "text",
           dataset: str | None = None) -> int:
    """Emit NodeRAG input documents from the KB.

    kb_root:  the knowledge/ root (the KB is opened on this path).
    out_dir:  destination input/ directory (created if absent).
    lang:     optional ISO language filter (e.g. "EN"); None keeps all.
    detector: detector slug to read claims for; defaults to the first detector
              found with article claims.
    repr:     "text" (CW claim sentences) or "canonized" (canonized claims).
    dataset:  dataset slug; defaults to polynarrative.
    """
    from core.knowledge_base import (KnowledgeBase, DATASET_POLYNARRATIVE,
                                     DATASET_FAKECTI)

    if repr not in ("text", "canonized"):
        raise ValueError(f"repr must be 'text' or 'canonized', got {repr!r}")

    kb = KnowledgeBase(Path(kb_root) if str(kb_root).endswith("knowledge")
                       else Path("knowledge"))
    ds = dataset or DATASET_POLYNARRATIVE

    # Resolve a detector slug: use the requested one, else probe known slugs.
    detectors = ([detector] if detector
                 else ["xlm-multicw", "mdb-multicw"])
    acs = []
    for det in detectors:
        acs = kb.all_article_claims(ds, det)
        if acs:
            break
    if not acs:
        logger.warning("[noderag_corpus] no article claims found for %s "
                       "(detectors tried: %s)", ds, detectors)
        return 0

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lang_norm = (lang or "").strip().upper()
    n = 0
    for ac in acs:
        # Language filter (best-effort: source language lives in metadata).
        if lang_norm:
            ac_lang = str((ac.metadata or {}).get("source_language", "")).upper()
            if ac_lang and ac_lang != lang_norm:
                continue

        if repr == "canonized":
            parts = [c.strip() for c in ac.canonized_claims if c and c.strip()]
            doc = "\n".join(parts)
        else:
            parts = [c.sentence.strip() for c in ac.claims
                     if getattr(c, "sentence", "").strip()]
            doc = " ".join(parts) or ac.title or ""

        if not doc.strip():
            continue
        (out_dir / f"{ac.article_name}.txt").write_text(doc, encoding="utf-8")
        n += 1

    logger.info("[noderag_corpus] emitted %d docs (repr=%s, dataset=%s, lang=%s)",
                n, repr, ds, lang_norm or "all")
    return n
