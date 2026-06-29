"""FakeCTI -> KB converter for the campaign-detection evaluation (README §9).

Source: data/FakeCTI/FakeCTI.csv (Cotroneo et al. 2025, arXiv:2505.03345),
columns ID, URL, TITLE, SOURCE, TEXT, CAMPAIGN, THREAT ACTOR, TYPE.
A flat data/FakeCTI.csv layout is also accepted as a fallback.

Unlike the earlier converter, this one uses REAL article metadata so the N1-N4
coordination signals read genuine structure rather than synthetic noise:

  * published_at  — parsed from the URL. Most FakeCTI WEB rows are Wayback
                    captures (``web.archive.org/web/<YYYYMMDDhhmmss>/<inner>``),
                    whose timestamp gives a real capture date; a ``/YYYY/MM/DD/``
                    path segment in the inner URL is used as a fallback.
  * source_domain — the REAL publisher host, taken from the inner (archived) URL
                    (not ``web.archive.org``), so burst/co-amplification/content-
                    reuse see distinct outlets.
  * source_language — FakeCTI is English; set to EN (so N4 cross-lingual is
                    correctly inapplicable on this monolingual corpus).

Scope (matches the agreed design): keep only TYPE==WEB rows that have BOTH text
and an extractable date, then drop campaigns with fewer than ``min_campaign_size``
such articles (singleton/sparse campaigns cannot exhibit coordination and only
add label noise). Emits Article records + campaign ground truth.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from core.structures import Article
from core.knowledge_base import KnowledgeBase

# Default scope knobs (overridable via env so the same converter serves the
# `convert(src, out_root)` signature every eval/_ensure_converted call expects).
TYPE_FILTER = os.getenv("FAKECTI_TYPE", "WEB").upper()
MIN_CAMPAIGN_SIZE = int(os.getenv("FAKECTI_MIN_CAMPAIGN", "5"))
SOURCE_LANGUAGE = "EN"

_WAYBACK_TS = re.compile(r"web\.archive\.org/web/(\d{4})(\d{2})(\d{2})\d*/(.*)$", re.I)
_PATH_DATE = re.compile(r"/((?:19|20)\d{2})/(\d{1,2})/(\d{1,2})/")


def _inner_url(url: str) -> str:
    """Strip a Wayback ``/web/<ts>/`` prefix to recover the original URL."""
    m = _WAYBACK_TS.search(url)
    if m:
        inner = m.group(4)
        # Wayback sometimes double-encodes the scheme (http:/ -> http://).
        return re.sub(r"^(https?):/(?!/)", r"\1://", inner)
    return url


def extract_date(url: str) -> str | None:
    """Return an ISO date (YYYY-MM-DD) parsed from a FakeCTI URL, or None.

    Prefers the Wayback capture timestamp; falls back to a /YYYY/MM/DD/ segment
    in the (possibly archived) URL path.
    """
    m = _WAYBACK_TS.search(url)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = _PATH_DATE.search(url)
    if m:
        year, mon, day = m.group(1), int(m.group(2)), int(m.group(3))
        if 1 <= mon <= 12 and 1 <= day <= 31:
            return f"{year}-{mon:02d}-{day:02d}"
    return None


def source_domain(url: str, fallback: str = "") -> str:
    """Real publisher host from the (de-archived) URL; SOURCE column as fallback."""
    host = (urlparse(_inner_url(url)).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host and host != "web.archive.org":
        return host
    fb = (fallback or "").strip().lower()
    return fb or "unknown"


def _resolve_src(path: Path) -> Path:
    """Resolve the FakeCTI CSV path, trying the canonical layout first.

    The pipeline expects ``data/FakeCTI/FakeCTI.csv`` (matches
    ``gen_cw_detect.FAKECTI_CSV``); historical drops sometimes used a flat
    ``data/FakeCTI.csv``. Accept both so users don't have to know which
    layout is canonical.
    """
    candidates = [path]
    # If the user passed the flat layout but the subfolder layout exists,
    # prefer the subfolder one (matches what gen_cw_detect reads).
    if path == Path("data/FakeCTI.csv"):
        candidates.insert(0, Path("data/FakeCTI/FakeCTI.csv"))
    elif path == Path("data/FakeCTI/FakeCTI.csv"):
        candidates.append(Path("data/FakeCTI.csv"))
    for c in candidates:
        if c.exists():
            return c
    return path  # let the caller raise with the original (most-specific) path


def convert(src: Path, out_root: Path, *,
            type_filter: str | None = None, min_campaign_size: int | None = None) -> int:
    src, out_root = Path(src), Path(out_root)
    src = _resolve_src(src)
    type_filter = (type_filter or TYPE_FILTER).upper()
    min_campaign_size = MIN_CAMPAIGN_SIZE if min_campaign_size is None else int(min_campaign_size)
    if not src.exists():
        raise FileNotFoundError(
            f"FakeCTI not found. Tried data/FakeCTI/FakeCTI.csv and "
            f"data/FakeCTI.csv. Place FakeCTI.csv (columns ID, URL, TITLE, "
            f"SOURCE, TEXT, CAMPAIGN, THREAT ACTOR, TYPE) at either path — "
            f"see Cotroneo et al. 2025, arXiv:2505.03345.")
    df = pd.read_csv(src)
    df.columns = [c.strip().upper() for c in df.columns]
    missing = {"URL", "TEXT", "CAMPAIGN", "TYPE"} - set(df.columns)
    if missing:
        raise ValueError(f"FakeCTI CSV missing columns: {missing}")

    # 1) scope by media type and presence of text.
    df["TYPE"] = df["TYPE"].fillna("").astype(str).str.strip().str.upper()
    if type_filter and type_filter != "ALL":
        df = df[df["TYPE"] == type_filter]
    df = df.dropna(subset=["TEXT", "CAMPAIGN"])
    for col in ("TITLE", "SOURCE", "URL"):
        df[col] = df.get(col, "").fillna("").astype(str)

    # 2) keep only rows with an extractable date, and pre-compute date/domain.
    records = []
    for _, row in df.iterrows():
        text = str(row["TEXT"]).strip()
        url = str(row["URL"]).strip()
        if not text or not url:
            continue
        date = extract_date(url)
        if not date:
            continue
        records.append({
            "csv_id": str(row["ID"]),
            "url": url, "campaign": str(row["CAMPAIGN"]).strip(),
            "title": str(row["TITLE"]).strip(), "text": text,
            "date": date, "domain": source_domain(url, str(row.get("SOURCE", ""))),
        })

    # 3) drop campaigns too small to exhibit coordination.
    per_campaign = defaultdict(int)
    for r in records:
        per_campaign[r["campaign"]] += 1
    kept_campaigns = {c for c, n in per_campaign.items() if n >= min_campaign_size}
    records = [r for r in records if r["campaign"] in kept_campaigns]

    # 4) emit Article records + ground truth.
    kb = KnowledgeBase(out_root)
    gt_dir = out_root / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    ground_truth: dict[str, str] = {}
    by_campaign: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for r in records:
        aid = f"article_{r['csv_id']}"
        if aid in seen:                      # de-dup identical archived URLs
            continue
        seen.add(aid)
        kb.save_article(Article(
            id=aid, url=r["url"], source_domain=r["domain"],
            title=(r["title"] or r["text"][:80]),
            content=r["text"], source_language=SOURCE_LANGUAGE,
            published_at=r["date"], author="Unknown"),
            dataset="fake-cti")
        ground_truth[aid] = r["campaign"]
        by_campaign[r["campaign"]].append(aid)

    total = len(seen)
    (gt_dir / "annotations.json").write_text(
        json.dumps(ground_truth, ensure_ascii=False, indent=2), encoding="utf-8")
    (gt_dir / "annotations_by_campaign.json").write_text(
        json.dumps({c: sorted(ids) for c, ids in sorted(by_campaign.items())},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (gt_dir / "_scope.json").write_text(
        json.dumps({"type": type_filter, "min_campaign": min_campaign_size,
                    "n_articles": total, "n_campaigns": len(by_campaign)}, indent=2),
        encoding="utf-8")
    print(f"FakeCTI[{type_filter}]: {total} dated articles across "
          f"{len(by_campaign)} campaigns (>= {min_campaign_size} articles each) -> {out_root}")
    return total


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Convert FakeCTI.csv → KB articles + campaign ground truth.")
    p.add_argument("--src", type=Path, default=Path("data/FakeCTI/FakeCTI.csv"),
                   help="Path to FakeCTI.csv. Falls back to data/FakeCTI.csv "
                        "if the default subfolder layout is absent.")
    p.add_argument("--out", type=Path, default=Path("knowledge/fakecti"))
    a = p.parse_args()
    convert(a.src, a.out)
