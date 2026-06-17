"""Claim veracity estimation (step 3).

Automatically verifies the True / False / Disputed status of central claims
from sub-narratives and narratives using a Context-1 agentic evidence-gathering
harness over three evidence sources, followed by a Gemma4-e2b verdict synthesis.

Evidence sources (controlled by ``ver_sources`` in Config):
  * multiclaim  — local MultiClaim CSV (claims + human ratings). Always available.
  * wikipedia   — online Wikipedia REST API. Degrades gracefully when offline.
  * web         — online web search (DuckDuckGo). Optional, degrades gracefully.

The harness is shared with the Context-1 narrative retrieval backend
(``core.hierarchy.harness.AgenticSearchHarness``) — the only difference is the
evidence-tools adapter and system prompt.

Verify modes:
  verify_hierarchy(kb, cfg, deep=False)
      deep=False: verify only sub-narrative and narrative CENTRAL CLAIMS.
      deep=True:  also verify all underlying canonized claims in each sub-
                  narrative.  Results propagate up: sub-narrative veracity is
                  the confidence-weighted mean of its claim verdicts; narrative
                  veracity is the mean of its sub-narrative veracity scores.

Veracity is cached in ``knowledge/veracity/cache/`` so subsequent runs skip
already-verified claims.

Verdict scale: True=1.0, Disputed=0.5, False=0.0.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    MofNCompleteColumn, TimeElapsedColumn,
)

from core.knowledge_base import KnowledgeBase, DATASET_FAKECTI

logger  = logging.getLogger(__name__)
console = Console()

# Verdict label → numeric score
_VERDICTS = {"true": 1.0, "false": 0.0, "disputed": 0.5}

_VERACITY_SYSTEM = """\
You are an evidence-gathering assistant for fact-checking.

Given a claim to verify, your job is to search the available corpus for
supporting or contradicting evidence. Use several different search queries
to find diverse evidence. When you have gathered enough snippets, call done.

Available tools (respond with JSON):
  {"tool": "search_corpus", "query": "<text>"}
  {"tool": "grep_corpus",   "pattern": "<text>"}
  {"tool": "read_document", "chunk_id": "<id>"}
  {"tool": "prune_chunks",  "chunk_ids": ["<id>", ...]}
  {"tool": "done",          "reasoning": "<why done>"}

Search at least 3 different queries before calling done. /no_think
"""

_VERDICT_SYSTEM = """\
You are a fact-checking assistant. Given a claim and evidence snippets, output
EXACTLY one of these verdicts: True, False, Disputed.

Rules:
- True: evidence clearly supports the claim.
- False: evidence clearly contradicts the claim.
- Disputed: evidence is mixed, absent, or inconclusive.

After the verdict, give a confidence percentage (0-100).
Output format (two lines only):
Verdict: <True|False|Disputed>
Confidence: <0-100>
/no_think
"""


# ---------------------------------------------------------------------------
# Evidence tools adapters
# ---------------------------------------------------------------------------

class _MultiClaimTools:
    """Evidence tool adapter over a local MultiClaim CSV.

    Embedding index is built lazily on the first ``search()`` call and
    persisted to ``knowledge/veracity/multiclaim_embs.npz`` so subsequent
    eval/verify runs skip the expensive encode step entirely.

    Cache invalidation is automatic: the cache key combines the embedder model
    name with a SHA-256 fingerprint of the (sorted) record IDs + texts, so any
    change to the MultiClaim CSV or the embedder model busts the cache.
    """

    def __init__(self, records: list[dict], embedder, *,
                 exclude_ids: set | None = None,
                 kb: KnowledgeBase | None = None,
                 embedder_name: str = "") -> None:
        """
        records      : [{"id": str, "text": str, "label": str}, ...]
        embedder     : SentenceTransformer-compatible model
        exclude_ids  : claim IDs to skip during search (leave-one-out)
        kb           : KnowledgeBase instance for reading/writing the cache
        embedder_name: human-readable model name used as the cache key
        """
        self._records     = records
        self._exclude     = exclude_ids or set()
        self._id_to_text  = {r["id"]: f"{r['text']} [{r['label']}]"
                             for r in records}
        self._embedder      = embedder
        self._embedder_name = embedder_name
        self._kb            = kb
        self._ids: list[str]      = []
        self._embs: "np.ndarray | None" = None

    # ---- cache key -------------------------------------------------------

    def _record_hash(self) -> str:
        """Content fingerprint: SHA-256 over sorted (id, text) pairs.

        Changing any record text or adding/removing rows changes this hash and
        therefore invalidates the cached embeddings automatically.
        """
        h = hashlib.sha256()
        for r in sorted(self._records, key=lambda x: x["id"]):
            h.update(r["id"].encode())
            h.update(r["text"].encode())
        return h.hexdigest()[:24]

    # ---- index build / load ----------------------------------------------

    def _ensure_index(self) -> None:
        import numpy as np

        if self._embs is not None:
            return

        rec_hash = self._record_hash()

        # Try loading from disk cache first
        if self._kb is not None and self._embedder_name:
            cached = self._kb.load_multiclaim_embs(self._embedder_name, rec_hash)
            if cached is not None:
                self._ids, self._embs = cached
                logger.info("[ver] MultiClaim embeddings loaded from cache "
                            "(%d records)", len(self._ids))
                return

        # Cache miss — build the index, then persist
        logger.info("[ver] Building MultiClaim embedding index "
                    "(%d records)…", len(self._records))
        texts      = [f"{r['text']} [{r['label']}]" for r in self._records]
        self._ids  = [r["id"] for r in self._records]

        # Encode in one batched call with a Rich progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
            console=console,
        ) as prog:
            task = prog.add_task(
                f"[cyan]Encoding MultiClaim corpus "
                f"({len(texts)} claims)…[/cyan]",
                total=1,
            )
            raw        = self._embedder.encode(texts)
            prog.advance(task)

        norms      = np.linalg.norm(raw, axis=1, keepdims=True)
        self._embs = raw / np.where(norms == 0, 1e-10, norms)
        self._embs = self._embs.astype(np.float32)

        # Persist to disk so the next run skips encoding
        if self._kb is not None and self._embedder_name:
            self._kb.save_multiclaim_embs(
                self._ids, self._embs, self._embedder_name, rec_hash)
            logger.info("[ver] MultiClaim embeddings cached to "
                        "knowledge/veracity/multiclaim_embs.npz")
        else:
            logger.debug("[ver] No KB provided; MultiClaim embeddings not cached")

    # ---- search interface ------------------------------------------------

    def search(self, query: str, seen: set, k: int) -> list[tuple[str, str]]:
        import numpy as np
        self._ensure_index()
        q     = self._embedder.encode([query])
        q     = q / (np.linalg.norm(q) + 1e-10)
        sims  = (q @ self._embs.T)[0]
        order = np.argsort(-sims)
        out   = []
        for i in order:
            cid = self._ids[i]
            if cid in seen or cid in self._exclude:
                continue
            out.append((cid, self._id_to_text[cid]))
            if len(out) >= k:
                break
        return out

    def grep(self, pattern: str, seen: set) -> list[tuple[str, str]]:
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            rx = re.compile(re.escape(pattern), re.IGNORECASE)
        return [(r["id"], self._id_to_text[r["id"]])
                for r in self._records
                if r["id"] not in seen and r["id"] not in self._exclude
                and rx.search(r["text"])]

    def get(self, doc_id: str) -> str | None:
        return self._id_to_text.get(doc_id)


class _WikipediaTools:
    """Evidence tool adapter over the Wikipedia REST API."""

    _BASE = "https://en.wikipedia.org"
    _SEARCH = _BASE + "/w/api.php"
    _SUMMARY = _BASE + "/api/rest_v1/page/summary/{title}"

    def __init__(self, lang: str = "en") -> None:
        self._lang = lang
        self._cache: dict[str, str] = {}

    def _safe_get(self, url: str, params: dict | None = None) -> dict:
        try:
            import requests
            r = requests.get(url, params=params, timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.debug("[wiki] request failed: %s", exc)
            return {}

    def search(self, query: str, seen: set, k: int) -> list[tuple[str, str]]:
        data = self._safe_get(self._SEARCH, {
            "action": "query", "list": "search",
            "srsearch": query, "srlimit": k * 2, "format": "json",
        })
        results = data.get("query", {}).get("search", [])
        out = []
        for item in results:
            title = item.get("title", "")
            doc_id = f"wiki:{title}"
            if doc_id in seen:
                continue
            snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
            text = f"{title}: {snippet}"
            self._cache[doc_id] = text
            out.append((doc_id, text))
            if len(out) >= k:
                break
        return out

    def grep(self, pattern: str, seen: set) -> list[tuple[str, str]]:
        return []  # Wikipedia doesn't support grep; search handles it

    def get(self, doc_id: str) -> str | None:
        if doc_id in self._cache:
            return self._cache[doc_id]
        if not doc_id.startswith("wiki:"):
            return None
        title = doc_id[5:]
        data = self._safe_get(self._SUMMARY.format(title=title.replace(" ", "_")))
        text = data.get("extract", "")
        if text:
            self._cache[doc_id] = text
        return text or None


class _WebSearchTools:
    """Evidence tool adapter over DuckDuckGo instant answer API (best-effort)."""

    _URL = "https://api.duckduckgo.com/"

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}

    def search(self, query: str, seen: set, k: int) -> list[tuple[str, str]]:
        try:
            import requests
            r = requests.get(self._URL, params={"q": query, "format": "json",
                                                "no_redirect": 1}, timeout=5)
            data = r.json()
        except Exception as exc:
            logger.debug("[web] search failed: %s", exc)
            return []
        out = []
        for item in data.get("Results", data.get("RelatedTopics", []))[:k * 2]:
            url  = item.get("FirstURL", "")
            text = item.get("Text", "")
            if not text or not url:
                continue
            doc_id = f"web:{url}"
            if doc_id in seen:
                continue
            self._cache[doc_id] = text
            out.append((doc_id, text))
            if len(out) >= k:
                break
        return out

    def grep(self, _p, _s) -> list:
        return []

    def get(self, doc_id: str) -> str | None:
        return self._cache.get(doc_id)


class _CompositeEvidenceTools:
    """Merges multiple tool adapters into the single interface AgenticSearchHarness
    expects: search(query, seen, k) / grep(pattern, seen) / get(doc_id).
    """

    def __init__(self, sources: list) -> None:
        self._sources = sources

    def search(self, query: str, seen: set, k: int) -> list[tuple[str, str]]:
        out, added = [], set()
        per = max(1, k // max(len(self._sources), 1))
        for src in self._sources:
            for did, text in src.search(query, seen | added, per + 2):
                if did not in added:
                    out.append((did, text))
                    added.add(did)
                if len(out) >= k:
                    return out
        return out[:k]

    def grep(self, pattern: str, seen: set) -> list[tuple[str, str]]:
        out, added = [], set()
        for src in self._sources:
            for did, text in src.grep(pattern, seen | added):
                if did not in added:
                    out.append((did, text))
                    added.add(did)
        return out

    def get(self, doc_id: str) -> str | None:
        for src in self._sources:
            val = src.get(doc_id)
            if val is not None:
                return val
        return None


# ---------------------------------------------------------------------------
# Build evidence tools from config + MultiClaim data
# ---------------------------------------------------------------------------

def _load_multiclaim(path: Path, text_col: str, label_col: str) -> list[dict]:
    """Load and filter MultiClaim CSV to True/False/Disputed entries."""
    if not path.exists():
        logger.warning("[ver] MultiClaim not found at %s", path)
        return []
    try:
        import pandas as pd
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        tc = text_col.lower()
        lc = label_col.lower()
        if tc not in df.columns or lc not in df.columns:
            available = list(df.columns)
            logger.warning("[ver] MultiClaim missing columns %r/%r; have %s",
                           tc, lc, available)
            # Try common alternatives
            for alt_t in ("claim", "text", "statement"):
                if alt_t in df.columns:
                    tc = alt_t; break
            for alt_l in ("label", "verdict", "rating"):
                if alt_l in df.columns:
                    lc = alt_l; break
        _label_map = {
            "true": "True", "mostly true": "True", "correct": "True",
            "false": "False", "mostly false": "False", "incorrect": "False",
            "pants on fire": "False",
            "disputed": "Disputed", "half-true": "Disputed",
            "mixed": "Disputed", "unverified": "Disputed",
        }
        df["_label"] = df[lc].astype(str).str.strip().str.lower().map(_label_map)
        df["_text"]  = df[tc].astype(str).str.strip()
        df = df.dropna(subset=["_label"])
        df = df[df["_text"] != ""]
        id_col = "id" if "id" in df.columns else None
        records = [
            {
                "id":    str(row[id_col]) if id_col else str(idx),
                "text":  row["_text"],
                "label": row["_label"],
            }
            for idx, row in df.iterrows()
        ]
        logger.info("[ver] MultiClaim: %d usable records loaded", len(records))
        return records
    except Exception as exc:
        logger.error("[ver] failed to load MultiClaim: %s", exc)
        return []


def build_evidence_tools(cfg, embedder, *,
                         exclude_ids: set | None = None,
                         kb: KnowledgeBase | None = None,
                         embedder_name: str = "") -> _CompositeEvidenceTools:
    """Build a CompositeEvidenceTools from the configured sources.

    ``kb`` and ``embedder_name`` are forwarded to ``_MultiClaimTools`` so it
    can read/write the pre-computed embedding cache in
    ``knowledge/veracity/multiclaim_embs.npz``.
    """
    sources_str = getattr(cfg, "ver_sources", "multiclaim,wikipedia,web")
    enabled = {s.strip().lower() for s in sources_str.split(",")}
    sources = []

    if "multiclaim" in enabled:
        multiclaim_path = Path("data") / "MultiClaim" / "multiclaim.csv"
        if not multiclaim_path.exists():
            for p in Path("data/MultiClaim").glob("*.csv") if Path("data/MultiClaim").exists() else []:
                multiclaim_path = p
                break
        records = _load_multiclaim(
            multiclaim_path,
            getattr(cfg, "ver_multiclaim_text_col", "claim"),
            getattr(cfg, "ver_multiclaim_label_col", "label"),
        )
        if records:
            sources.append(_MultiClaimTools(
                records, embedder,
                exclude_ids=exclude_ids,
                kb=kb,
                embedder_name=embedder_name,
            ))

    if "wikipedia" in enabled:
        sources.append(_WikipediaTools())

    if "web" in enabled:
        sources.append(_WebSearchTools())

    return _CompositeEvidenceTools(sources)


# ---------------------------------------------------------------------------
# Verdict synthesis
# ---------------------------------------------------------------------------

def _claim_hash(claim: str) -> str:
    return hashlib.sha256(claim.encode()).hexdigest()[:16]


def _parse_verdict_response(raw: str) -> tuple[str, float]:
    """Parse 'Verdict: X\\nConfidence: Y' into (label, score 0..1)."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    verdict = "Disputed"
    confidence = 0.5
    for line in raw.splitlines():
        line = line.strip()
        if line.lower().startswith("verdict:"):
            v = line.split(":", 1)[1].strip().lower()
            if "true" in v:
                verdict = "True"
            elif "false" in v:
                verdict = "False"
            else:
                verdict = "Disputed"
        elif line.lower().startswith("confidence:"):
            try:
                pct = float(re.search(r"[\d.]+", line)[0])
                confidence = min(max(pct / 100.0, 0.0), 1.0)
            except (TypeError, ValueError):
                pass
    return verdict, confidence


def synthesize_verdict(claim: str, evidence: list[tuple[str, str]],
                       llm) -> tuple[str, float]:
    """Call the LLM to produce a final verdict from gathered evidence."""
    if not evidence:
        return "Disputed", 0.3

    snippets = "\n".join(
        f"[{did}] {text[:300]}" for did, text in evidence[:10]
    )
    user = (
        f"Claim: {claim}\n\n"
        f"Evidence snippets:\n{snippets}\n\n"
        "Based on this evidence, what is your verdict?"
    )
    try:
        raw = (llm(_VERDICT_SYSTEM, user, max_tokens=80) or "").strip()
    except TypeError:
        raw = (llm(_VERDICT_SYSTEM, user) or "").strip()
    return _parse_verdict_response(raw)


def verify_claim_cached(claim: str, tools, llm, kb: KnowledgeBase) -> tuple[str, float]:
    """Verify a claim, using the KB cache to skip already-verified claims."""
    from core.hierarchy.harness import AgenticSearchHarness

    h = _claim_hash(claim)
    cached = kb.load_veracity_cache(h)
    if cached:
        return cached["verdict"], cached["confidence"]

    harness = AgenticSearchHarness(
        tools, llm, _VERACITY_SYSTEM,
        token_budget=4096, top_k=5, max_turns=6,
    )
    evidence = harness.search(claim)
    verdict, confidence = synthesize_verdict(claim, evidence, llm)

    kb.save_veracity_cache(h, {"claim": claim, "verdict": verdict,
                               "confidence": confidence})
    return verdict, confidence


# ---------------------------------------------------------------------------
# Propagation helpers
# ---------------------------------------------------------------------------

_VERDICT_SCORE = {"True": 1.0, "False": 0.0, "Disputed": 0.5}


def _conf_weighted_mean(verdicts_with_conf: list[tuple[str, float]]) -> tuple[float | None, float | None]:
    """Confidence-weighted mean veracity score from (verdict, confidence) pairs."""
    if not verdicts_with_conf:
        return None, None
    num = sum(_VERDICT_SCORE.get(v, 0.5) * c for v, c in verdicts_with_conf)
    den = sum(c for _, c in verdicts_with_conf)
    mean_conf = den / len(verdicts_with_conf)
    return (num / den if den > 0 else 0.5), mean_conf


# ---------------------------------------------------------------------------
# Main verify_hierarchy entry point
# ---------------------------------------------------------------------------

def verify_hierarchy(kb: KnowledgeBase, cfg, *, deep: bool = False) -> dict:
    """Verify central claims (and optionally canonized claims) for FakeCTI.

    deep=False: only central claims of sub-narratives and narratives.
    deep=True:  also all canonized claims within each sub-narrative; propagate
                up to sub-narrative → narrative veracity.

    Returns summary statistics.
    """
    from core.models import make_embedder, make_generator, close_generator

    console.print(
        f"\n[bold cyan]{'Deep v' if deep else 'V'}erify hierarchy "
        f"— FakeCTI[/bold cyan]")
    console.print(
        f"[dim]Sources: {cfg.ver_sources}  "
        f"Generator: {cfg.ver_generator} ({cfg.ver_quantization})[/dim]\n")

    # camp_embedder is intentionally reused here: sharing the embedding model
    # with the campaign step means the MultiClaim .npz cache is shared too.
    console.print(f"[bold]Loading embedder[/bold] [cyan]{cfg.camp_embedder}[/cyan]…")
    embedder = make_embedder(cfg.camp_embedder)

    console.print(
        f"[bold]Loading generator[/bold] [cyan]{cfg.ver_generator}[/cyan] "
        f"([dim]{cfg.ver_quantization}[/dim])…")
    llm = make_generator(cfg.ver_generator, cfg.ver_quantization)

    tools = build_evidence_tools(cfg, embedder,
                                 kb=kb,
                                 embedder_name=cfg.camp_embedder)

    # ---- sub-narratives ----
    sns = kb.sub_narratives(DATASET_FAKECTI, "xlm-multicw") + \
          kb.sub_narratives(DATASET_FAKECTI, "mdb-multicw")
    # deduplicate across detectors by id
    sns_by_id = {sn.id: sn for sn in sns}
    sns = list(sns_by_id.values())

    verified_sn = skipped_sn = 0
    with Progress(SpinnerColumn(),
                  TextColumn("[progress.description]{task.description}"),
                  BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                  console=console) as prog:
        task = prog.add_task("[cyan]Verifying sub-narratives…[/cyan]",
                             total=len(sns))
        for sn in sns:
            claims_to_verify: list[tuple[str, bool]] = [
                (sn.central_claim, False)]  # (text, is_canon)
            if deep:
                claims_to_verify += [(c, True) for c in sn.claims if c.strip()]

            verdicts = []
            for text, _ in claims_to_verify:
                if not text.strip():
                    continue
                v, c = verify_claim_cached(text, tools, llm, kb)
                verdicts.append((v, c))

            if verdicts:
                # Central claim result drives the primary veracity
                sn.veracity, sn.veracity_confidence = (
                    _VERDICT_SCORE.get(verdicts[0][0], 0.5), verdicts[0][1]
                ) if not deep else _conf_weighted_mean(verdicts)
                kb.save_sub_narrative(sn)
                verified_sn += 1
            else:
                skipped_sn += 1
            prog.advance(task)

    console.print(f"  sub-narratives: verified={verified_sn}  skipped={skipped_sn}")

    # ---- narratives ----
    # Collect narratives across all backends
    narratives = []
    for backend in ("dense", "bm25-rag", "specfi-cs", "cspecfi", "context-1"):
        narratives += kb.narratives(DATASET_FAKECTI, backend)
    nar_by_id = {n.id: n for n in narratives}
    narratives = list(nar_by_id.values())

    verified_nar = skipped_nar = 0
    with Progress(SpinnerColumn(),
                  TextColumn("[progress.description]{task.description}"),
                  BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                  console=console) as prog:
        task = prog.add_task("[cyan]Verifying narratives…[/cyan]",
                             total=len(narratives))
        for nar in narratives:
            if not nar.central_claim.strip():
                skipped_nar += 1
                prog.advance(task)
                continue
            v, c = verify_claim_cached(nar.central_claim, tools, llm, kb)
            nar.veracity = _VERDICT_SCORE.get(v, 0.5)
            nar.veracity_confidence = c
            # Blend central-claim verdict with member sub-narrative verdicts.
            # sn.veracity is already a numeric score; use it directly.
            member_v = [
                (sn.veracity, sn.veracity_confidence or 0.5)
                for sn_id in nar.sub_narratives
                if (sn := sns_by_id.get(sn_id)) and sn.veracity is not None
            ]
            if member_v:
                mean_v, mean_c = _conf_weighted_mean(member_v)
                if mean_v is not None and mean_c is not None:
                    # 60% central claim verdict, 40% member-aggregate
                    nar.veracity = 0.6 * _VERDICT_SCORE.get(v, 0.5) + 0.4 * mean_v
                    nar.veracity_confidence = 0.6 * c + 0.4 * mean_c
            kb.save_narrative(nar)
            verified_nar += 1
            prog.advance(task)

    console.print(f"  narratives: verified={verified_nar}  skipped={skipped_nar}")
    close_generator(llm)

    summary = {
        "mode": "deep" if deep else "central",
        "sub_narratives_verified": verified_sn,
        "sub_narratives_skipped": skipped_sn,
        "narratives_verified": verified_nar,
        "narratives_skipped": skipped_nar,
    }
    console.print(f"\n[bold]Summary:[/bold] {summary}")
    return summary
