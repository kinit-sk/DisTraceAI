"""Structured output paths for evaluation/generation HTML reports.

Replaces the old flat ``evaluation/eval_<step>.html`` naming (which collided
across detectors and datasets) with a directory layout that keeps every run's
report distinct:

    results/<step>/<dataset>/<detector>__<method>.html

Segments that don't apply to a given step are omitted gracefully. Examples:

    results/claim-detection/polynarrative/xlm-multicw.html
    results/narratives/polynarrative/xlm-multicw__specfi-ccs.html
    results/campaigns/fake-cti/xlm-multicw__dense.html
    results/claim-canonization/benchmark.html          (no single detector)
    results/claim-veracity/gemma4-e2b.html      (no dataset/detector)

All segments are slugified so model paths like ``models/xlm-multicw`` become
``xlm-multicw`` strings stay intact.
"""
from __future__ import annotations

import re
from pathlib import Path

_RESULTS_ROOT = Path("results")


def _slug(text: str) -> str:
    """Filesystem-safe slug: keep alnum, dot, dash, underscore; drop the rest.

    Path-like values (``models/xlm-multicw``) collapse to their basename so the
    directory layout stays flat and predictable.
    """
    if not text:
        return ""
    text = str(text).strip().rstrip("/\\")
    if "/" in text or "\\" in text:
        text = re.split(r"[/\\]", text)[-1]
    text = text.replace(" ", "-")
    return re.sub(r"[^A-Za-z0-9._-]", "", text)


def report_path(step: str, *, dataset: str | None = None,
                detector: str | None = None, method: str | None = None,
                extra: str | None = None, action: str = "eval") -> Path:
    """Build a structured HTML report path and ensure its parent dir exists.

    step:     pipeline step slug (e.g. "narratives", "claim-detection").
    dataset:  dataset slug (e.g. "polynarrative"); omitted if None.
    detector: detector path/slug; slugified to its basename.
    method:   retrieval method / variant (e.g. "specfi-ccs"); joined to the
              detector with "__".
    extra:    fallback stem when there is no detector/method (e.g. a model+quant
              combo for the canonization benchmark or veracity eval).
    action:   "eval" or "generate" — only affects the filename when no other
              stem is available.

    The filename is ``<detector>__<method>.html`` when both are present, or
    whichever single component exists, or ``extra``/``action`` as a last resort.
    """
    parts = [_RESULTS_ROOT, _slug(step)]
    if dataset:
        parts.append(_slug(dataset))
    out_dir = Path(*parts)

    det = _slug(detector) if detector else ""
    meth = _slug(method) if method else ""
    if det and meth:
        stem = f"{det}__{meth}"
    elif det:
        stem = det
    elif meth:
        stem = meth
    elif extra:
        stem = _slug(extra)
    else:
        stem = _slug(action) or "report"

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}.html"
