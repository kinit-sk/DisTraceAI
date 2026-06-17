"""Pipeline step statistics — persistence and TUI display.

Storage
-------
All files live under ``knowledge/stats/`` so they are deleted together with the
knowledge base and never outlive it.

  knowledge/stats/eval_<step>.json
  knowledge/stats/generate_<step>.json

Each file is a dict keyed by a *parameter-combo string* that uniquely identifies
one configuration (detector, method, …). Writing a run with the same key
overwrites the previous result; a new key appends. This gives one row per
distinct parameter combination in the TUI panel.

Eval record   : {"params": {...}, "ts": "...", <score fields>, "kb": {...}}
Generate record: {"params": {...}, "ts": "...", "kb": {...}}

TUI display contract
--------------------
- Eval Launch row   : eval scores (overall only, per-parameter-combo) +
                      KB counts from the same record.
- Generate Launch row: KB counts only (total articles/claims/subs in KB).
- Per-language breakdown is intentionally omitted — keep it concise.
- "Not computed yet" is shown when no file or no matching record exists.

Parameter-combo keys
--------------------
step                 key fields
----                 ----------
claim-detection      detector
claim-canonization   model_key + quant  (eval); detector (generate)
sub-narratives       detector
narratives           detector + method
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _kb_stats_dir() -> Path:
    """Always relative to cwd so it follows the knowledge/ folder."""
    return Path("knowledge") / "stats"


def _eval_path(step: str) -> Path:
    return _kb_stats_dir() / f"eval_{step.replace('-', '_')}.json"


def _gen_path(step: str) -> Path:
    return _kb_stats_dir() / f"generate_{step.replace('-', '_')}.json"


_NOT_YET = "[dim]Not computed yet.[/dim]"


# ---------------------------------------------------------------------------
# Low-level read / write
# ---------------------------------------------------------------------------

def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    except Exception:
        pass   # stats are display-only; never crash the main flow


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------------------
# KB count helpers
# ---------------------------------------------------------------------------

def _kb_counts(step: str, det_slug: str) -> dict:
    """Count KB records for this step/detector across datasets.

    Returns a dict of {dataset: human-readable string} for the TUI panel.
    For claim-detection and downstream steps, counts articles, CW claims,
    and canonized claims directly from ArticleClaims records.
    """
    from core.knowledge_base import (KnowledgeBase, DATASET_POLYNARRATIVE,
                                     DATASET_FAKECTI)
    kb = KnowledgeBase(Path("knowledge"))
    counts: dict = {}

    if step in ("claim-detection", "claim-canonization"):
        for ds in (DATASET_POLYNARRATIVE, DATASET_FAKECTI):
            acs = kb.all_article_claims(ds, det_slug)
            if not acs:
                continue
            n_art    = len(acs)
            n_claims = sum(len(ac.claims) for ac in acs)
            n_canon  = sum(1 for ac in acs if any(ac.canonized_claims))
            if step == "claim-detection":
                counts[ds] = f"{n_art} articles  {n_claims} CW claims"
            else:
                counts[ds] = f"{n_art} articles  {n_claims} CW claims  {n_canon} canonized"

    elif step == "sub-narratives":
        for ds in (DATASET_POLYNARRATIVE, DATASET_FAKECTI):
            acs = kb.all_article_claims(ds, det_slug)
            sns = kb.sub_narratives(ds, det_slug)
            if not acs and not sns:
                continue
            n_art = len(acs)
            n_sn  = len(sns)
            counts[ds] = f"{n_art} articles  {n_sn} sub-narratives"

    elif step == "narratives":
        for ds in (DATASET_POLYNARRATIVE, DATASET_FAKECTI):
            n = sum(
                len(kb.narratives(ds, b))
                for b in ("dense", "bm25_rag", "bm25-rag",
                          "specfi-cs", "cspecfi", "context-1", "context1")
            )
            if n:
                counts[ds] = f"{n} narratives"

    elif step == "claim-veracity":
        # Show how many sub-narratives have a veracity verdict.
        for ds in (DATASET_POLYNARRATIVE, DATASET_FAKECTI):
            all_sns = []
            for d in ("xlm-multicw", "mdb-multicw"):
                all_sns += kb.sub_narratives(ds, d)
            verified = sum(1 for sn in all_sns if sn.veracity is not None)
            if all_sns:
                counts[ds] = f"{verified}/{len(all_sns)} sub-narratives verified"

    elif step == "campaigns":
        for ds in (DATASET_POLYNARRATIVE, DATASET_FAKECTI):
            n = sum(
                len(kb.campaigns(ds, b))
                for b in ("dense", "bm25_rag", "bm25-rag",
                          "specfi-cs", "cspecfi", "context-1", "context1")
            )
            if n:
                counts[ds] = f"{n} campaigns"

    return counts


# ---------------------------------------------------------------------------
# Save helpers — called from main.run_generate and eval modules
# ---------------------------------------------------------------------------

def save_eval_stats(step: str, param_key: str, params: dict,
                    scores: dict, det_slug: str = "") -> None:
    """Upsert one eval result keyed by param_key."""
    data = _read(_eval_path(step))
    kb = {}
    try:
        kb = _kb_counts(step, det_slug or params.get("detector", ""))
    except Exception:
        pass
    data[param_key] = {"params": params, "ts": _ts(), "scores": scores, "kb": kb}
    _write(_eval_path(step), data)


def save_generate_stats(step: str, summary: dict) -> None:
    """Upsert generate stats for each detector found in summary.

    summary shape: {dataset: {detector: counts_dict}} or {dataset: counts_dict}.
    The full counts_dict is preserved verbatim so the TUI can display
    articles/skipped/new_*/total_in_kb exactly as printed by the generator.
    Writing the same detector key again overwrites the previous record.
    """
    # Flatten to per-detector records: {det_slug: {dataset: counts_dict}}
    det_records: dict[str, dict] = {}

    for dataset, val in summary.items():
        if not isinstance(val, dict):
            continue
        first = next(iter(val.values()), None)
        if isinstance(first, dict):
            # nested: {dataset: {detector: counts_dict}}
            for det, counts in val.items():
                det_records.setdefault(det, {})[dataset] = (
                    counts if isinstance(counts, dict) else {"total": counts})
        else:
            # flat: {dataset: {processed, skipped, total_in_kb, ...}}
            det_records.setdefault("_all", {})[dataset] = val

    path = _gen_path(step)
    data = _read(path)
    for det, ds_map in det_records.items():
        data[det] = {"params": {"detector": det}, "ts": _ts(), "kb": ds_map}
    _write(path, data)


# ---------------------------------------------------------------------------
# Formatters — return Rich markup strings
# ---------------------------------------------------------------------------

def _fmt_scores_row(scores: dict, step: str) -> str:
    """One-line summary of the most important scores for a step."""
    if step in ("claim-detection", "sub-narratives"):
        f1  = scores.get("f1", "?")
        acc = scores.get("acc", "?")
        try:
            return f"F1={float(f1):.1%}  Acc={float(acc):.1%}"
        except (ValueError, TypeError):
            return f"F1={f1}  Acc={acc}"
    if step == "claim-canonization":
        ok = scores.get("english_ok", "?")
        lat = scores.get("median_lat_s", "?")
        try:
            s = f"EN-OK={float(ok):.1%}"
        except (ValueError, TypeError):
            s = f"EN-OK={ok}"
        try:
            s += f"  lat={float(lat):.2f}s"
        except (ValueError, TypeError):
            pass
        return s
    if step == "narratives":
        a1  = scores.get("acc@1", "?")
        mp  = scores.get("map",   "?")
        try:
            return f"Acc@1={float(a1):.3f}  MAP={float(mp):.3f}"
        except (ValueError, TypeError):
            return f"Acc@1={a1}  MAP={mp}"
    if step == "claim-veracity":
        acc = scores.get("accuracy", "?")
        f1  = scores.get("macro_f1", "?")
        try:
            return f"Acc={float(acc):.1%}  MacroF1={float(f1):.3f}"
        except (ValueError, TypeError):
            return f"Acc={acc}  MacroF1={f1}"
    if step == "campaigns":
        ari = scores.get("ari", "?")
        nmi = scores.get("nmi", "?")
        vm  = scores.get("v_measure", "?")
        try:
            return f"ARI={float(ari):.3f}  NMI={float(nmi):.3f}  V={float(vm):.3f}"
        except (ValueError, TypeError):
            return f"ARI={ari}  NMI={nmi}  V={vm}"
    return str(scores)


def _fmt_kb(kb: dict) -> str:
    """Format a {dataset: counts} dict for one-line display.

    counts may be a plain scalar (legacy) or a dict with keys like
    total_in_kb, new_cw_claims, articles_processed, skipped, etc.
    Only the most informative fields are shown to keep the line short.
    """
    if not kb:
        return ""
    parts = []
    for ds, v in kb.items():
        if isinstance(v, dict):
            # Pick the most informative fields in priority order.
            total   = v.get("total_in_kb")
            new_c   = v.get("new_cw_claims") or v.get("new_sub_narratives")
            skipped = v.get("skipped")
            proc    = v.get("articles_processed")
            frag = ""
            if total is not None:
                frag = f"{total} total"
            if new_c is not None:
                frag += f"  +{new_c} new"
            if skipped is not None and (proc or 0) == 0 and total:
                frag += f"  ({skipped} skipped)"
            parts.append(f"{ds}: {frag.strip()}" if frag else f"{ds}: {v}")
        else:
            parts.append(f"{ds}: {v}")
    return "  |  ".join(parts)


def _eval_panel(step: str, cfg) -> str:
    data = _read(_eval_path(step))
    if not data:
        return _NOT_YET

    lines = ["[bold]Eval results[/bold]"]
    for key, rec in sorted(data.items(), key=lambda x: x[1].get("ts", "")):
        params = rec.get("params", {})
        scores = rec.get("scores", {})
        kb     = rec.get("kb", {})
        ts     = rec.get("ts", "")

        score_str = _fmt_scores_row(scores, step)
        kb_str    = _fmt_kb(kb)
        param_str = "  ".join(f"{k}={v}" for k, v in params.items()
                               if k != "detector" or len(params) > 1)
        # For single-param configs just show the value, not the key
        if not param_str and params:
            param_str = next(iter(params.values()), key)

        lines.append(f"[dim]{param_str or key}[/dim]  {score_str}")
        if kb_str:
            lines.append(f"  [dim]kb: {kb_str}[/dim]")
        if ts:
            lines.append(f"  [dim]{ts}[/dim]")

    return "\n".join(lines)


def _generate_panel(step: str, cfg) -> str:
    data = _read(_gen_path(step))
    if not data:
        return _NOT_YET

    lines = ["[bold]Last Generate[/bold]"]
    for key, rec in sorted(data.items(), key=lambda x: x[1].get("ts", "")):
        kb  = rec.get("kb", {})
        ts  = rec.get("ts", "")
        det = rec.get("params", {}).get("detector", key)
        det_label = f"[dim]{det}[/dim]" if det != "_all" else ""
        if det_label:
            lines.append(det_label)
        for ds, counts in kb.items():
            if isinstance(counts, dict):
                total   = counts.get("total_in_kb")
                new_c   = counts.get("new_cw_claims") or counts.get("new_sub_narratives")
                skipped = counts.get("skipped")
                frags = []
                if total is not None:
                    frags.append(f"{total} in KB")
                if new_c is not None:
                    frags.append(f"+{new_c} new")
                if skipped:
                    frags.append(f"{skipped} skipped")
                lines.append(f"  [dim]{ds}:[/dim] {'  '.join(frags) or str(counts)}")
            else:
                lines.append(f"  [dim]{ds}:[/dim] {counts}")
        if ts:
            lines.append(f"  [dim]{ts}[/dim]")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Eval stats — claim veracity
# ---------------------------------------------------------------------------

def _load_eval_veracity() -> str:
    path = _eval_path("claim-veracity")
    data = _read(path)
    if not data:
        return _NOT_YET
    lines = ["[bold]Veracity eval results[/bold]"]
    for key, rec in sorted(data.items(), key=lambda x: x[1].get("ts", "")):
        scores = rec.get("scores", {})
        acc = scores.get("accuracy", "?")
        f1  = scores.get("macro_f1", "?")
        n   = scores.get("n", "?")
        params = rec.get("params", {})
        label = params.get("generator", key)
        try:
            lines.append(f"[dim]{label}[/dim]  Acc={float(acc):.1%}  MacroF1={float(f1):.3f}  n={n}")
        except (ValueError, TypeError):
            lines.append(f"[dim]{label}[/dim]  Acc={acc}  MacroF1={f1}")
        lines.append(f"  [dim]{rec.get('ts','')}[/dim]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Eval stats — campaigns
# ---------------------------------------------------------------------------

def _load_eval_campaigns() -> str:
    path = _eval_path("campaigns")
    data = _read(path)
    if not data:
        return _NOT_YET
    lines = ["[bold]Campaign eval results[/bold]"]
    for key, rec in sorted(data.items(), key=lambda x: x[1].get("ts", "")):
        scores = rec.get("scores", {})
        ari  = scores.get("ari", "?")
        nmi  = scores.get("nmi", "?")
        vm   = scores.get("v_measure", "?")
        n    = scores.get("n", "?")
        label = rec.get("params", {}).get("extractor", key)
        try:
            lines.append(f"[dim]{label}[/dim]  ARI={float(ari):.3f}  NMI={float(nmi):.3f}  V={float(vm):.3f}  n={n}")
        except (ValueError, TypeError):
            lines.append(f"[dim]{label}[/dim]  ARI={ari}  NMI={nmi}  V={vm}")
        lines.append(f"  [dim]{rec.get('ts','')}[/dim]")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_stats(step: str, action: str, cfg) -> str:
    """Return Rich-markup stats for the TUI Launch-row description.

    Eval   → eval scores (overall, per param-combo) + KB counts.
    Generate → KB counts only.
    """
    if action == "eval":
        if step == "claim-veracity":
            return _load_eval_veracity()
        if step == "campaigns":
            return _load_eval_campaigns()
        eval_part = _eval_panel(step, cfg)
        gen_data  = _read(_gen_path(step))
        if gen_data:
            kb_lines = ["[bold]KB contents[/bold]"]
            for key, rec in sorted(gen_data.items(),
                                   key=lambda x: x[1].get("ts", "")):
                kb_str = _fmt_kb(rec.get("kb", {}))
                det = rec.get("params", {}).get("detector", key)
                if det != "_all":
                    kb_lines.append(f"[dim]{det}:[/dim]  {kb_str or '(empty)'}")
                else:
                    kb_lines.append(kb_str or "(empty)")
            sep = "\n\n"
            return eval_part + sep + "\n".join(kb_lines)
        return eval_part

    if action == "generate":
        return _generate_panel(step, cfg)

    return _NOT_YET
