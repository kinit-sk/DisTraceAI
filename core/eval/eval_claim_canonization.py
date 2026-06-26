# ---------------------------------------------------------------------------
# Claim Canonization Benchmark
# ---------------------------------------------------------------------------
#
# Samples 2 ground-truth check-worthy claims (label=1) per language from the
# MultiCW dataset test set (up to 20 languages × 2 = up to 40 claims) and
# runs each of the six benchmark LLMs through the canonization prompt.
#
# For each model × quantization combination we record:
#
#   • English success rate — heuristic: ≥ 75 % of characters in the Latin
#     Unicode block or common punctuation/digits.  Detects the most common
#     failure mode (model returns text in the source language).
#   • Median and mean latency per claim (seconds).
#
# Output
# ------
#   • Three per-quantization tables showing all 6 models with sampled outputs.
#   • An overall statistics table (English rate + latency across all quants).
#   • CSV export: results/eval_claim_canonization.csv
# ---------------------------------------------------------------------------

from __future__ import annotations

import csv
import os
import statistics
import time as _time
from pathlib import Path

from rich.terminal_theme import MONOKAI
from core.models import make_generator, close_generator

# ── English-detection heuristic ─────────────────────────────────────────────
_LATIN_CHARS   = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
_ALLOWED_EXTRA = set("0123456789 \t\n.,!?;:'\"-–—()[]{}/@#%&*+=/<>\\")


def _is_english(text: str, threshold: float = 0.75) -> bool:
    """Return True when the text appears to be English / Latin-script output.

    Counts the fraction of characters that are ASCII letters, digits, spaces,
    or common punctuation.  A threshold of 0.75 catches the common failure
    mode (output entirely in Arabic, Chinese, Cyrillic, etc.) while tolerating
    proper nouns and entity names in other scripts.
    """
    text = (text or "").strip()
    if not text:
        return False
    allowed = sum(1 for c in text if c in _LATIN_CHARS or c in _ALLOWED_EXTRA)
    return (allowed / len(text)) >= threshold


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

BENCH_MODEL_KEYS: list[str] = [
    "qwen3.5-2b",
    "qwen3.5-4b",
    "qwen3.5-9b",
    "gemma4-e2b",
    "gemma4-e4b",
    "gemma4-12b",
]

_BENCH_DISPLAY: dict[str, str] = {
    "qwen3.5-2b":  "Qwen3.5-2B",
    "qwen3.5-4b":  "Qwen3.5-4B",
    "qwen3.5-9b":  "Qwen3.5-9B",
    "gemma4-e2b":  "Gemma-4-E2B-IT",
    "gemma4-e4b":  "Gemma-4-E4B-IT",
    "gemma4-12b":  "Gemma-4-12B-IT",
}

_BENCH_PARAMS: dict[str, str] = {
    "qwen3.5-2b":  "2B",
    "qwen3.5-4b":  "4B",
    "qwen3.5-9b":  "9B",
    "gemma4-e2b":  "2B",
    "gemma4-e4b":  "4B",
    "gemma4-12b":  "12B",
}

# Max languages to sample from (dataset may have fewer)
_MAX_LANGS        = 20
_SAMPLES_PER_LANG = 2

# ---------------------------------------------------------------------------
# Canonization prompt  (identical to canon_generate._canonize_claim)
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
    "- Maximum 20 words.\n\n"
    "Claim:\n{claim}"
)


def _canonize(llm, claim: str) -> tuple[str, float]:
    """Run the canonization prompt; return (output_text, elapsed_seconds)."""
    user    = _USER_TMPL.format(claim=claim.strip())
    t0      = _time.perf_counter()
    result  = llm(_SYSTEM, user, max_tokens=64)
    elapsed = _time.perf_counter() - t0
    return (result.strip() if result else ""), elapsed


# ---------------------------------------------------------------------------
# Main benchmark function
# ---------------------------------------------------------------------------

def eval_claim_canonization_benchmark(project_root: Path, console=None) -> None:
    """Benchmark the model catalogue on multilingual claim canonization.

    Dataset
    -------
    MultiCW test set — 2 ground-truth CW claims (label=1) per language,
    up to 20 languages, sampled deterministically (random_state=42).

    Output
    ------
    - A Rich table with sampled claim outputs for all 6 models.
    - An overall statistics table (English rate, latency) across all models.
    - CSV export to results/eval_claim_canonization.csv.
    """
    import pandas as pd
    from rich.console import Console as RichConsole
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn,
        BarColumn, MofNCompleteColumn, TaskProgressColumn,
    )
    from rich.table import Table
    from rich.rule import Rule
    from rich import box as _box

    _console = console if console is not None else RichConsole(record=True)

    multicw_path  = project_root / "data" / "MultiCW" / "multicw-test.csv"
    results_dir   = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_out       = results_dir / "eval_claim_canonization.csv"

    def _print(msg):
        _console.print(msg)

    # ── Load and validate MultiCW ────────────────────────────────────────────
    if not multicw_path.exists():
        _print(f"[red]MultiCW dataset not found at {multicw_path}[/red]")
        return

    df = pd.read_csv(multicw_path)
    required = {"text", "label", "lang"}
    missing  = required - set(df.columns)
    if missing:
        _print(f"[red]MultiCW CSV missing columns: {missing}[/red]")
        return

    df = df.dropna(subset=["text", "label", "lang"])
    df["text"]  = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    cw_df        = df[df["label"] == 1].copy()
    all_langs    = sorted(cw_df["lang"].unique())
    target_langs = all_langs[:_MAX_LANGS]

    sampled = pd.concat(
        [
            cw_df[cw_df["lang"] == lang].sample(
                n=min(_SAMPLES_PER_LANG, len(cw_df[cw_df["lang"] == lang])),
                random_state=42,
            )
            for lang in target_langs
        ],
        ignore_index=True,
    )

    n_claims = len(sampled)
    n_langs  = sampled["lang"].nunique()
    _print(f"[cyan]Loaded {n_claims} ground-truth CW claims across {n_langs} languages.[/cyan]")
    _print(f"[dim]Languages: {', '.join(sorted(sampled['lang'].unique()))}[/dim]\n")

    # ── Run benchmark: model → claims ────────────────────────────────
    all_results: dict[str, list[dict]] = {}

    for model_key in BENCH_MODEL_KEYS:
        display_name = _BENCH_DISPLAY[model_key]
        _print(Rule(f"[bold cyan]{display_name}[/bold cyan]", style="cyan"))
        _print(f"[dim]Loading {display_name}…[/dim]")

        try:
            llm = make_generator(model_key)
        except Exception as exc:
            _print(f"[red]Failed to load {display_name}: {exc}[/red]")
            all_results[model_key] = []
            continue

        rows: list[dict] = []
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                console=_console,
                transient=True,
            ) as prog:
                t = prog.add_task(
                    f"Canonizing [{display_name}]…",
                    total=n_claims,
                )
                for _, row in sampled.iterrows():
                    claim   = str(row["text"])
                    lang    = str(row["lang"])
                    output, elapsed = _canonize(llm, claim)
                    en_ok   = _is_english(output)
                    rows.append({
                        "model_key":    model_key,
                        "display_name": display_name,
                        "language":     lang,
                        "original":     claim,
                        "output":       output,
                        "english_ok":   en_ok,
                        "latency_s":    round(elapsed, 3),
                    })
                    prog.advance(t, 1)
                prog.update(t, description=f"[green]✔ {display_name} done[/green]")
        finally:
            close_generator(llm)

        all_results[model_key] = rows

        if rows:
            en_rate = sum(r["english_ok"] for r in rows) / len(rows)
            lats    = [r["latency_s"] for r in rows]
            _print(
                f"  English rate: [bold]{en_rate:.1%}[/bold]  "
                f"median latency: {statistics.median(lats):.2f}s  "
                f"mean: {statistics.mean(lats):.2f}s\n"
            )

    # ── Output table — all models × all claims ────────────────────────────────
    _print("")
    _print(Rule("[bold]Canonized Claims[/bold]"))
    _print(
        f"[dim]Original claim + model outputs for each of the {n_claims} claims. "
        f"Review for meaning preservation.[/dim]\n"
    )

    # Build per-claim map: (lang, original) → {model_key: output}
    claim_map: dict = {}
    for model_key, model_rows in all_results.items():
        for r in model_rows:
            ck = (r["language"], r["original"])
            if ck not in claim_map:
                claim_map[ck] = {"language": r["language"], "original": r["original"]}
            claim_map[ck][model_key] = r["output"]

    tbl = Table(
        show_header  = True,
        header_style = "bold magenta",
        border_style = "dim",
        box          = _box.SIMPLE_HEAVY,
        show_lines   = True,
    )
    tbl.add_column("Lang",     style="cyan",  width=5,  no_wrap=True)
    tbl.add_column("Original", style="white", width=38, overflow="fold")
    for key in BENCH_MODEL_KEYS:
        tbl.add_column(
            _BENCH_DISPLAY[key],
            style="green",
            width=28,
            overflow="fold",
        )

    for (lang, original), entry in sorted(claim_map.items(), key=lambda x: x[0][0]):
        row_cells = [lang, original]
        for key in BENCH_MODEL_KEYS:
            out  = entry.get(key, "[dim]—[/dim]")
            en   = _is_english(out)
            cell = out if en else f"[red]{out}[/red]"
            row_cells.append(cell)
        tbl.add_row(*row_cells)

    _print(tbl)

    # ── Overall statistics table ──────────────────────────────────────────────
    _print("")
    _print(Rule("[bold]Overall Benchmark Statistics[/bold]"))
    summary_tbl = Table(
        title        = f"Claim Canonization Benchmark — {len(BENCH_MODEL_KEYS)} models",
        border_style = "blue",
        box          = _box.ROUNDED,
    )
    summary_tbl.add_column("Model",          style="cyan",   min_width=18)
    summary_tbl.add_column("Params",         style="dim",    justify="right")
    summary_tbl.add_column("English rate",   style="green",  justify="right")
    summary_tbl.add_column("Median lat (s)", style="yellow", justify="right")
    summary_tbl.add_column("Mean lat (s)",   style="yellow", justify="right")
    summary_tbl.add_column("Claims run",     style="dim",    justify="right")

    for model_key in BENCH_MODEL_KEYS:
        display_name = _BENCH_DISPLAY[model_key]
        param_count  = _BENCH_PARAMS[model_key]
        rows = all_results.get(model_key, [])
        if not rows:
            summary_tbl.add_row(
                display_name, param_count,
                "[red]FAILED[/red]", "—", "—", "0",
            )
        else:
            en_rate = sum(r["english_ok"] for r in rows) / len(rows)
            lats    = [r["latency_s"] for r in rows]
            summary_tbl.add_row(
                display_name,
                param_count,
                f"{en_rate:.1%}",
                f"{statistics.median(lats):.2f}",
                f"{statistics.mean(lats):.2f}",
                str(len(rows)),
            )
        # Visual separator between models
        summary_tbl.add_section()

    _print(summary_tbl)

    # ── HTML export ───────────────────────────────────────────────────────────
    from core.eval.report_paths import report_path
    html_out = report_path("claim-canonization", extra="benchmark")
    _console.save_html(str(html_out), theme=MONOKAI, clear=False)
    _print(f"[dim]HTML report saved to {html_out}[/dim]")

    # ── CSV export ────────────────────────────────────────────────────────────
    flat_rows: list[dict] = []
    for model_rows in all_results.values():
        flat_rows.extend(model_rows)

    if flat_rows:
        with open(csv_out, "w", newline="", encoding="utf-8") as fh:
            fieldnames = [
                "model_key", "display_name",
                "language", "original", "output", "english_ok", "latency_s",
            ]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)
        _print(f"\n[green]Full results saved to[/green] [cyan]{csv_out}[/cyan]")

    # Persist per-model stats to the KB stats dir.
    try:
        from core.ui.stats import save_eval_stats
        for model_key in BENCH_MODEL_KEYS:
            rows_m = all_results.get(model_key, [])
            if not rows_m:
                continue
            ok_vals  = [r["english_ok"] for r in rows_m]
            lat_vals = [r["latency_s"]  for r in rows_m]
            save_eval_stats(
                "claim-canonization",
                param_key=model_key,
                params={"model": _BENCH_DISPLAY.get(model_key, model_key)},
                scores={"english_ok":    sum(ok_vals)  / len(ok_vals),
                        "median_lat_s":  statistics.median(lat_vals),
                        "n":             len(rows_m)},
            )
    except Exception:
        pass


def main(cfg=None) -> None:
    eval_claim_canonization_benchmark(Path(os.getcwd()))


if __name__ == "__main__":
    main()
