"""Veracity evaluation benchmark.

Method: paraphrase-based, two-class (True / False), with stratified sampling
and a three-stage evidence fallback.

1. Load MultiClaim CSV, filter to True/False entries (Disputed is excluded).
2. Stratify pre-paraphrase: equal numbers of True and False source records.
3. Generate N paraphrases per claim using the configured paraphrase generator
   (cached to ``knowledge/veracity/multiclaim_test_paraphrases.json``). The
   paraphrase prompt asks the model to produce a MORE GENERAL, self-contained
   ENGLISH query — overly specific details (dates, named witnesses, exact
   numbers, quoted attributions) are dropped, and pronouns are expanded into
   the entities they refer to. This keeps the query retrievable when the
   matching source claim has been removed by the leave-one-out filter.
4. For each paraphrase query, run an evidence-source fallback ladder:
     a. MultiClaim only — agentic harness → synthesize verdict.
        If the verdict is True or False, accept it.
     b. If MultiClaim returned Disputed (typically: not enough evidence),
        re-query against Wikipedia (en.wikipedia.org).
     c. If Wikipedia still returns Disputed, re-query against the web.
     d. If web also returns Disputed, flag the query as a MISS and count it
        against accuracy. Misses are reported separately as a coverage stat.
   Leave-one-out is applied to the MultiClaim source corpus at step (a) so the
   exact source claim cannot be retrieved trivially.
5. Report two-class accuracy + macro-F1 over {True, False}, plus a coverage
   stat (queries that produced a non-Disputed verdict somewhere in the ladder).

Why two classes?  Including Disputed as a class conflates "evidence says
mixed" with "we could not find enough evidence" — the latter is a coverage
problem, not a classification one. By restricting to True/False gold labels
and routing Disputed predictions into the fallback ladder + miss counter, the
benchmark separates classification quality from retrieval coverage.

Why paraphrase-based?  Human-assigned gold labels from MultiClaim are genuine
ground truth.  Paraphrasing varies surface form while preserving semantics, so
the eval measures robustness to wording variation — exactly the real-world
scenario.  A disclosed limitation: LLM paraphrases may be easier than natural
reformulations; results are a mild upper bound on real performance.
"""
from __future__ import annotations

import csv
import logging
import random
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich import box
from rich.table import Table
from rich.terminal_theme import MONOKAI
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    MofNCompleteColumn, TimeElapsedColumn,
)

from core.knowledge_base import KnowledgeBase
from core.models import make_embedder, make_generator
from core.gen.gen_veracity import (
    _load_multiclaim, build_single_source_tools,
    synthesize_verdict, _VERACITY_SYSTEM,
)

logger  = logging.getLogger(__name__)
console = Console(record=True)

# Two-class evaluation: Disputed is a fallback signal, not a target class.
_LABELS = ["True", "False"]
_MULTICLAIM_PATH = Path("data/MultiClaim/fact_checks.csv")

# Precision is fixed to bf16 (awq4 removed); no longer a config parameter.
_PRECISION = "bf16"

# Deterministic stratification + paraphrase shuffling
_RNG_SEED = 1605


# ---------------------------------------------------------------------------
# Paraphrase generation + caching
# ---------------------------------------------------------------------------

# Updated prompt: ask the model for a generalised, self-contained English
# query. The previous "rewrite while preserving exact meaning" produced
# paraphrases that were as specific as the source claim — when the leave-one-
# out filter then removed that source claim from MultiClaim, retrieval often
# returned nothing relevant. Asking for a more general formulation that drops
# overly specific surface details (dates, named witnesses, quoted attributions,
# exact figures) and expands pronouns makes the query retrievable against
# OTHER MultiClaim entries / Wikipedia / web pages that cover the same topic.
_PARAPHRASE_SYSTEM = """\
Rewrite the given claim as a more GENERAL, SELF-CONTAINED ENGLISH query that
preserves the core factual statement while making it easier to retrieve
evidence about the same topic from other sources.

Guidelines:
- ALWAYS write the output in ENGLISH, regardless of the source language.
- Drop overly specific surface details when the claim is too narrow:
  exact dates, named individual witnesses, quoted attributions, very precise
  numbers ("a small number" instead of "exactly 327"). Keep the core entities
  (countries, organisations, well-known figures, events).
- Replace pronouns and unclear references with the entities they refer to.
- The result must stand on its own — no external context needed to interpret.
- Preserve the truth-conditional meaning: the rewritten claim must be true
  if and only if the original was true.
- Output a single declarative sentence, no longer than the original.

STRICT RULES:
- Output ONLY the rewritten English claim. No explanation, no quotes,
  no commentary, no leading/trailing punctuation other than a final period.
/no_think
"""


def _generate_paraphrase(claim: str, llm) -> str:
    try:
        raw = (llm(_PARAPHRASE_SYSTEM, f"Claim: {claim}", max_tokens=120) or "").strip()
    except TypeError:
        raw = (llm(_PARAPHRASE_SYSTEM, f"Claim: {claim}") or "").strip()
    return raw or claim


def _random_sample(records: list[dict], n: int) -> list[dict]:
    """Randomly draw up to *n* records from the full MultiClaim list.

    Uses the deterministic seed so repeated runs with the same ``n`` always
    draw the same subset.  If ``n`` is 0 or >= len(records) the full list is
    returned unchanged.
    """
    if n <= 0 or n >= len(records):
        return records
    rng = random.Random(_RNG_SEED)
    pool = list(records)
    rng.shuffle(pool)
    return pool[:n]


def _stratify_records(records: list[dict]) -> list[dict]:
    """Filter to True/False only and balance the two classes by subsampling.

    Pre-paraphrase stratification: equal counts of True and False source
    records, sampled without replacement using a deterministic seed.
    Disputed records are dropped entirely.
    """
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        lbl = r.get("label", "")
        if lbl in _LABELS:
            by_label[lbl].append(r)

    n_true  = len(by_label.get("True",  []))
    n_false = len(by_label.get("False", []))
    n_take  = min(n_true, n_false)

    rng = random.Random(_RNG_SEED)
    balanced: list[dict] = []
    for lbl in _LABELS:
        pool = list(by_label.get(lbl, []))
        rng.shuffle(pool)
        balanced.extend(pool[:n_take])
    rng.shuffle(balanced)

    logger.info("[ver-eval] stratified: True=%d, False=%d → %d each "
                "(Disputed dropped)", n_true, n_false, n_take)
    return balanced


def _build_test_set(records: list[dict], cfg, kb: KnowledgeBase) -> list[dict]:
    """Generate paraphrases or load from cache.

    Returns list of {source_id, paraphrase, gold_label, generator, quant}.
    Caller is responsible for stratifying ``records`` first.
    """
    cached = kb.load_paraphrase_test(cfg.ver_paraphrase_generator,
                                     _PRECISION,
                                     cfg.ver_n_samples)
    if cached:
        # Cached test set may have been built with a different filter (e.g.
        # previously including Disputed). Discard cached entries whose gold
        # label is no longer in scope so the eval stays consistent with the
        # current True/False scoping.
        kept = [c for c in cached if c.get("gold_label") in _LABELS]
        if kept:
            console.print(
                f"[dim]Loaded {len(kept)} cached paraphrase test entries "
                f"({len(cached) - len(kept)} out-of-scope entries skipped).[/dim]")
            return kept

    console.print(
        f"\n[bold]Generating paraphrases[/bold] "
        f"({cfg.ver_n_paraphrases}× each, "
        f"{len(records)} claims → "
        f"~{len(records) * cfg.ver_n_paraphrases} queries)…")
    console.print(
        f"[dim]Generator: {cfg.ver_paraphrase_generator} "
        f"({_PRECISION})[/dim]\n")

    llm = make_generator(cfg.ver_paraphrase_generator)
    test_records = []
    with Progress(SpinnerColumn(),
                  TextColumn("[progress.description]{task.description}"),
                  BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                  console=console) as prog:
        task = prog.add_task("[cyan]Paraphrasing…[/cyan]", total=len(records))
        for rec in records:
            for _ in range(max(1, cfg.ver_n_paraphrases)):
                para = _generate_paraphrase(rec["text"], llm)
                if para and para.strip() != rec["text"].strip():
                    test_records.append({
                        "source_id":   rec["id"],
                        "paraphrase":  para,
                        "gold_label":  rec["label"],
                        "generator":   cfg.ver_paraphrase_generator,
                        "quant":       _PRECISION,
                    })
            prog.advance(task)
    del llm

    kb.save_paraphrase_test(test_records,
                            cfg.ver_paraphrase_generator, _PRECISION,
                            cfg.ver_n_samples)
    console.print(
        f"[dim]{len(test_records)} paraphrase queries cached to "
        f"knowledge/veracity/multiclaim_test_paraphrases.json[/dim]")
    return test_records


# ---------------------------------------------------------------------------
# Sequential evidence-source fallback
# ---------------------------------------------------------------------------

def _verdict_with_fallback(paraphrase: str, source_id: str, *,
                           cfg, embedder, llm, kb,
                           embedder_name: str) -> tuple[str, str]:
    """Run the three-stage MultiClaim → Wikipedia → Web fallback ladder.

    Returns ``(predicted_label, stage)``:
      * ``predicted_label`` ∈ {"True", "False", "Disputed"}.
      * ``stage`` is one of ``"multiclaim"``, ``"wikipedia"``, ``"web"``,
        ``"miss"`` — the stage at which a non-Disputed verdict was produced,
        or ``"miss"`` if every stage returned Disputed.

    The MultiClaim stage applies leave-one-out (the source claim is excluded
    from the evidence corpus).  Wikipedia / Web stages do not, since the
    public web cannot be filtered.
    """
    from core.hierarchy.harness import AgenticSearchHarness

    stage_specs = [
        ("multiclaim", {source_id}),  # leave-one-out only on the local corpus
        ("wikipedia",  None),
        ("web",        None),
    ]
    for stage, exclude_ids in stage_specs:
        tools = build_single_source_tools(
            stage, cfg, embedder,
            exclude_ids=exclude_ids,
            kb=kb, embedder_name=embedder_name,
        )
        # If the stage has no usable llm_backends (e.g. MultiClaim CSV missing),
        # treat it as Disputed and fall through to the next stage.
        if not tools._sources:
            continue
        harness = AgenticSearchHarness(
            tools, llm, _VERACITY_SYSTEM,
            token_budget=cfg.ver_token_budget,
            top_k=5,
            max_turns=cfg.ver_max_turns,
        )
        evidence = harness.search(paraphrase)
        predicted, _ = synthesize_verdict(paraphrase, evidence, llm)
        pred_norm = predicted.title() if predicted.title() in (*_LABELS, "Disputed") else "Disputed"
        if pred_norm in _LABELS:
            return pred_norm, stage
        # else: Disputed → fall through to next stage

    # Every stage returned Disputed (or had no llm_backends) → miss.
    return "Disputed", "miss"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _macro_f1(confusion: dict[str, dict[str, int]]) -> dict[str, float]:
    """Per-class precision/recall/F1 and macro-F1 over the two-class confusion.

    The confusion is indexed by gold and pred labels in ``_LABELS`` only;
    misses (Disputed predictions) are accounted for as false negatives against
    the gold label (the off-diagonal column is implicit since the prediction
    is not in ``_LABELS``).
    """
    result: dict[str, float] = {}
    f1s = []
    for label in _LABELS:
        tp = confusion[label].get(label, 0)
        fp = sum(confusion[other].get(label, 0)
                 for other in _LABELS if other != label)
        # False negatives include any non-tp prediction for this gold label,
        # including misses tracked under the "Disputed" pred bucket below.
        gold_total = sum(confusion[label].values())
        fn = gold_total - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        result[f"prec_{label}"] = prec
        result[f"rec_{label}"]  = rec
        result[f"f1_{label}"]   = f1
        f1s.append(f1)
    result["macro_f1"] = sum(f1s) / len(f1s) if f1s else 0.0
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(cfg=None) -> None:
    from config import Config
    cfg = cfg or Config.load()

    kb_root = Path("knowledge")
    kb = KnowledgeBase(kb_root)

    # Load MultiClaim
    mc_path = _MULTICLAIM_PATH
    if not mc_path.exists():
        for p in (Path("data/MultiClaim").glob("*.csv")
                  if Path("data/MultiClaim").exists() else []):
            mc_path = p; break
    records = _load_multiclaim(
        mc_path, cfg.ver_multiclaim_text_col, cfg.ver_multiclaim_label_col)
    if not records:
        console.print(
            "[red]MultiClaim data not found or empty.[/red] "
            "Place fact_checks.csv under data/MultiClaim/.")
        return

    # Random sub-sample before stratification (controlled by ver_n_samples).
    all_records = records
    records = _random_sample(records, cfg.ver_n_samples)
    if len(records) < len(all_records):
        console.print(
            f"[dim]Randomly sampled {len(records)} / {len(all_records)} "
            f"MultiClaim records (ver_n_samples={cfg.ver_n_samples}).[/dim]")

    # Stratify pre-paraphrase: True/False only, equal counts.
    records = _stratify_records(records)
    if not records:
        console.print(
            "[red]No True/False records after stratification.[/red] "
            "MultiClaim must contain at least one True and one False entry.")
        return

    n_true_recs  = sum(1 for r in records if r["label"] == "True")
    n_false_recs = sum(1 for r in records if r["label"] == "False")
    console.print(
        f"\n[bold cyan]Claim veracity evaluation[/bold cyan]\n"
        f"[dim]{len(records)} stratified MultiClaim records "
        f"(True={n_true_recs}, False={n_false_recs}; Disputed excluded)[/dim]")

    # Build / load cached test set
    test_set = _build_test_set(records, cfg, kb)
    if not test_set:
        console.print("[red]No test paraphrases generated.[/red]")
        return

    # Build evidence embedder and verdict generator
    console.print(f"\n[bold]Loading embedder[/bold] [cyan]{cfg.camp_embedder}[/cyan]…")
    embedder = make_embedder(cfg.camp_embedder)
    console.print(
        f"[bold]Loading verdict generator[/bold] [cyan]{cfg.ver_generator}[/cyan]…")
    llm = make_generator(cfg.ver_generator)

    # Pre-build the MultiClaim embedding index ONCE (cached to disk) so the
    # leave-one-out loop does not re-encode the corpus for every query.
    console.print("[bold]Preparing MultiClaim evidence index…[/bold]")
    _seed_tools = build_single_source_tools(
        "multiclaim", cfg, embedder, kb=kb, embedder_name=cfg.camp_embedder,
    )
    for src in _seed_tools._sources:
        if hasattr(src, "_ensure_index"):
            src._ensure_index()
    del _seed_tools

    # Confusion is indexed only by labels in scope. Misses are counted both
    # against the gold label (so accuracy correctly degrades) and in a
    # separate stage_counts dict for the coverage report.
    confusion: dict[str, dict[str, int]] = {l: defaultdict(int) for l in _LABELS}
    per_label_n: dict[str, int] = defaultdict(int)
    stage_counts: dict[str, int] = defaultdict(int)
    misses_by_label: dict[str, int] = defaultdict(int)

    with Progress(SpinnerColumn(),
                  TextColumn("[progress.description]{task.description}"),
                  BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                  console=console) as prog:
        task = prog.add_task("[cyan]Evaluating…[/cyan]", total=len(test_set))

        for entry in test_set:
            source_id  = entry["source_id"]
            paraphrase = entry["paraphrase"]
            gold       = entry["gold_label"]

            gold_norm = gold.title() if gold.title() in _LABELS else None
            if gold_norm is None:
                # Defensive: cached test sets may carry an out-of-scope label.
                prog.advance(task)
                continue

            pred_norm, stage = _verdict_with_fallback(
                paraphrase, source_id,
                cfg=cfg, embedder=embedder, llm=llm, kb=kb,
                embedder_name=cfg.camp_embedder,
            )

            stage_counts[stage] += 1
            if stage == "miss":
                # Miss counts as a wrong prediction for accuracy (numerator
                # excludes it). We record it under the gold-label row but
                # against a synthetic "Disputed" pred bucket so the confusion
                # row total equals per_label_n[gold_norm].
                confusion[gold_norm]["Disputed"] += 1
                misses_by_label[gold_norm] += 1
            else:
                confusion[gold_norm][pred_norm] += 1
            per_label_n[gold_norm] += 1
            prog.advance(task)

    del llm

    # Compute metrics
    total_correct = sum(confusion[l][l] for l in _LABELS)
    n_all   = sum(per_label_n.values())
    n_miss  = sum(misses_by_label.values())
    accuracy = total_correct / n_all if n_all else 0.0
    coverage = (n_all - n_miss) / n_all if n_all else 0.0
    metrics  = _macro_f1(confusion)

    # Display
    console.print()
    console.rule("[bold cyan]Veracity Evaluation Results[/bold cyan]")
    console.print(
        f"  [dim]queries={n_all}  "
        f"paraphrase_generator={cfg.ver_paraphrase_generator}  "
        f"verdict_generator={cfg.ver_generator}[/dim]")
    console.print()

    # Overall table
    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold white")
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")
    t.add_row("Accuracy",  f"{accuracy:.3f}")
    t.add_row("Macro-F1",  f"{metrics['macro_f1']:.3f}")
    t.add_row("Coverage",  f"{coverage:.3f}  (miss={n_miss}/{n_all})")
    for label in _LABELS:
        t.add_row(f"  F1 / {label}",
                  f"{metrics[f'f1_{label}']:.3f}  "
                  f"(P={metrics[f'prec_{label}']:.2f} "
                  f"R={metrics[f'rec_{label}']:.2f})")
    console.print(t)

    # Coverage breakdown by stage where the verdict was produced.
    stage_t = Table(box=box.SIMPLE, show_header=True, header_style="dim")
    stage_t.add_column("Resolution stage")
    stage_t.add_column("Queries", justify="right")
    stage_t.add_column("Share",   justify="right")
    for stage in ("multiclaim", "wikipedia", "web", "miss"):
        n = stage_counts.get(stage, 0)
        share = (n / n_all) if n_all else 0.0
        stage_t.add_row(stage, str(n), f"{share:.1%}")
    console.print("[dim]Evidence-source fallback breakdown:[/dim]")
    console.print(stage_t)

    # Confusion matrix (gold rows × pred cols, with a Miss column for visibility)
    console.print("[dim]Confusion matrix (rows=gold, cols=predicted; "
                  "Miss = Disputed after web fallback):[/dim]")
    cm = Table(box=box.SIMPLE, show_header=True, header_style="dim")
    cm.add_column("Gold\\Pred")
    for p in _LABELS:
        cm.add_column(p, justify="right")
    cm.add_column("Miss", justify="right")
    for g in _LABELS:
        row = [g] + [str(confusion[g][p]) for p in _LABELS]
        row.append(str(misses_by_label.get(g, 0)))
        cm.add_row(*row)
    console.print(cm)

    # CSV
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "eval_claim_veracity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["accuracy", accuracy])
        w.writerow(["coverage", coverage])
        w.writerow(["macro_f1", metrics["macro_f1"]])
        w.writerow(["misses",   n_miss])
        for label in _LABELS:
            w.writerow([f"f1_{label}", metrics[f"f1_{label}"]])
        for stage in ("multiclaim", "wikipedia", "web", "miss"):
            w.writerow([f"stage_{stage}", stage_counts.get(stage, 0)])

    # Stats
    try:
        from core.ui.stats import save_eval_stats
        save_eval_stats(
            "claim-veracity",
            param_key=f"{cfg.ver_generator}__{_PRECISION}",
            params={"generator": cfg.ver_generator,
                    "quant": _PRECISION,
                    "sources": "multiclaim→wikipedia→web (sequential)"},
            scores={"accuracy": accuracy,
                    "macro_f1": metrics["macro_f1"],
                    "coverage": coverage,
                    "misses":   n_miss,
                    "n": n_all},
        )
    except Exception:
        pass

    from core.eval.report_paths import report_path
    html_out = report_path(
        "claim-veracity",
        extra=f"{cfg.ver_generator}__{_PRECISION}")
    console.save_html(str(html_out), theme=MONOKAI, clear=False)
    console.print(f"[dim]Results → {csv_path}  HTML → {html_out}[/dim]")
