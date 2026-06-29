"""Veracity evaluation benchmark.

Method: paraphrase-based, two-class (True / False), with stratified sampling
and a CONFIDENCE-GATED three-stage evidence fallback (plan §4.4).

1. Load MultiClaim CSV, filter to True/False entries (Disputed excluded).
2. Stratify pre-paraphrase: equal numbers of True and False source records.
3. Generate N paraphrases per claim using the configured paraphrase generator
   (cached to ``knowledge/veracity/multiclaim_test_paraphrases.json``). The
   paraphrase prompt is meaning-preserving (rewrites surface form, KEEPS the
   truth-conditional content) — earlier versions deliberately generalised the
   claim, which weakened the truth-preservation guarantee and risked
   invalidating gold labels.
4. For each paraphrase query, run the confidence-gated fallback ladder:
     a. MultiClaim — agentic harness over the local MultiClaim corpus.
        Accepts the verdict iff it is True or False AND confidence is at or
        above ``cfg.ver_confidence_threshold``.
     b. Otherwise: Wikipedia. Same acceptance rule.
     c. Otherwise: Web (duckduckgo-search). Same acceptance rule.
     d. If even the web stage cannot produce a confident True/False, count
        the query as a MISS (counts against accuracy + reported as coverage).
   The previous implementation only escalated on a LITERAL "Disputed"
   verdict — so weak/under-confident True/False predictions on shallow
   MultiClaim evidence never reached Wikipedia or the web, which was exactly
   the observed ~30% accuracy failure mode (DisTraceAI review note B).
5. Leave-one-out is NOT applied on the MultiClaim corpus (per design decision
   B.6): the test queries are paraphrases, not exact copies; allowing the
   source record to appear in retrieval keeps the eval a realistic test of
   "given a paraphrased claim, can the retrieval+verdict pipeline find the
   correct answer in the corpus we have".
6. Report two-class accuracy + macro-F1 over {True, False}, plus a coverage
   stat (queries that produced a confident non-Disputed verdict somewhere in
   the ladder).
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

# Meaning-preserving paraphrase prompt (per design decision D.15). The
# previous prompt deliberately generalised the claim (dropped dates / named
# witnesses / exact numbers) to make it retrievable against leave-one-out.
# That weakens the "paraphrase ≈ same gold label" guarantee — a generalised
# version of a False claim might be True (or vice-versa), invalidating the
# ground-truth label. We now keep semantics intact and have removed
# leave-one-out (B.6) so retrieval can still hit the source record itself.
_PARAPHRASE_SYSTEM = """\
Rewrite the given claim in ENGLISH while PRESERVING its exact factual content.

Guidelines:
- ALWAYS write the output in ENGLISH, regardless of the source language.
- Vary the surface form (word choice, sentence structure, voice), but keep
  ALL truth-conditional content: dates, numbers, named entities, attributions,
  quantifiers ("all", "some", "no"), and negations must survive intact.
- Replace pronouns and unclear references with the entities they refer to,
  so the result stands on its own without external context.
- The rewritten claim must be TRUE if and only if the original was TRUE.
- Output a single declarative sentence.

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


def _filter_and_sample(records: list[dict], target_n: int) -> list[dict]:
    """Filter to True/False, then sample to ``target_n`` (balanced).

    Pipeline:
      1. Drop every record whose gold label is not in ``_LABELS``
         (Disputed and any other label is excluded *before* sampling).
      2. Compute ``per_class = target_n // 2`` (or the smaller class's size
         if either class has fewer than that available — the result is then
         balanced at the smaller pool's size).
      3. Sample ``per_class`` records without replacement from each class
         using a deterministic seed, concatenate, then shuffle once more so
         True/False rows are interleaved for the progress bar.

    Returns at most ``target_n`` records, balanced 50/50 between True and
    False. Filtering BEFORE sampling guarantees the final test-set size is
    deterministic (= 2 × per_class) instead of dependent on how many
    True/False rows happened to land in a pre-stratification random subset.
    """
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        lbl = r.get("label", "")
        if lbl in _LABELS:
            by_label[lbl].append(r)

    n_true  = len(by_label.get("True",  []))
    n_false = len(by_label.get("False", []))

    # If target_n <= 0, take everything available balanced at min(class).
    cap = (target_n // 2) if target_n > 0 else min(n_true, n_false)
    per_class = min(cap, n_true, n_false)

    rng = random.Random(_RNG_SEED)
    balanced: list[dict] = []
    for lbl in _LABELS:
        pool = list(by_label.get(lbl, []))
        rng.shuffle(pool)
        balanced.extend(pool[:per_class])
    rng.shuffle(balanced)

    logger.info(
        "[ver-eval] filtered → True=%d, False=%d (Disputed dropped); "
        "balanced sample = %d each (target_n=%d)",
        n_true, n_false, per_class, target_n)
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
# Confidence-gated sequential evidence-source fallback
# ---------------------------------------------------------------------------

def _verdict_with_fallback(paraphrase: str, source_id: str, *,
                           cfg, embedder, llm, kb,
                           embedder_name: str) -> tuple[str, float, str]:
    """Run the confidence-gated MultiClaim → Wikipedia → Web fallback ladder.

    Returns ``(predicted_label, confidence, stage)``:
      * ``predicted_label`` ∈ {"True", "False", "Disputed"}.
      * ``confidence``      ∈ [0, 1] from the final verdict.
      * ``stage``           ∈ {"multiclaim", "wikipedia", "web", "miss"} —
                              the stage at which an accepted verdict was
                              produced, or ``"miss"`` if every stage returned
                              either Disputed or a confident-enough True/False
                              that we still didn't accept.

    Acceptance rule (plan §4.4):
        accept if  pred ∈ {True, False}  AND  confidence ≥ threshold
    where the threshold is ``cfg.ver_confidence_threshold``. Otherwise we
    proceed to the next stage. This is the fix for the observed ~30%
    accuracy: under the old "Disputed-only" escalation, weak True/False on
    shallow MultiClaim evidence was accepted, so Wikipedia/web never ran.

    Leave-one-out is NOT applied (B.6): the test queries are paraphrases, not
    the source claim verbatim; the source record is a legitimate retrieval
    target. The ``source_id`` parameter is retained for forward compatibility
    (e.g. an opt-in leave-one-out flag) but currently unused.
    """
    from core.hierarchy.harness import AgenticSearchHarness

    threshold = float(getattr(cfg, "ver_confidence_threshold", 0.65))
    stages = ("multiclaim", "wikipedia", "web")
    last_pred = "Disputed"
    last_conf = 0.0

    for stage in stages:
        tools = build_single_source_tools(
            stage, cfg, embedder,
            exclude_ids=None,   # B.6 — no leave-one-out
            kb=kb, embedder_name=embedder_name,
        )
        # If the stage has no usable evidence source (e.g. MultiClaim CSV
        # missing, web stage couldn't reach the internet), treat it as a
        # silent skip and fall through.
        if not tools._sources:
            logger.info("[ver-eval] stage %r has no usable source; skipping", stage)
            continue

        harness = AgenticSearchHarness(
            tools, llm, _VERACITY_SYSTEM,
            token_budget=cfg.ver_token_budget,
            top_k=5,
            max_turns=cfg.ver_max_turns,
        )
        evidence = harness.search(paraphrase)
        predicted, confidence = synthesize_verdict(paraphrase, evidence, llm)
        pred_norm = predicted.title() if predicted.title() in (*_LABELS, "Disputed") else "Disputed"

        last_pred, last_conf = pred_norm, confidence

        if pred_norm in _LABELS and confidence >= threshold:
            return pred_norm, confidence, stage
        # else: fall through to next stage — either Disputed, or True/False
        # below the confidence threshold.

    # Every stage returned either Disputed or under-confident True/False.
    # We surface the last (web-stage) prediction for diagnostic value but
    # record this as a "miss" for the accuracy / coverage report.
    return last_pred, last_conf, "miss"


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
    records = _load_multiclaim(mc_path)
    if not records:
        console.print(
            "[red]MultiClaim data not found or empty.[/red] "
            "Place fact_checks.csv under data/MultiClaim/.")
        return

    # Filter True/False (drop Disputed) and sample to ver_n_samples,
    # balanced 50/50. Filter-then-sample (was sample-then-filter) guarantees
    # the final test-set size is deterministic.
    all_records = records
    records = _filter_and_sample(records, cfg.ver_n_samples)
    if not records:
        console.print(
            "[red]No True/False records after filtering.[/red] "
            "MultiClaim must contain at least one True and one False entry.")
        return
    n_true_recs  = sum(1 for r in records if r["label"] == "True")
    n_false_recs = sum(1 for r in records if r["label"] == "False")
    if len(records) < cfg.ver_n_samples:
        console.print(
            f"[dim]Only {len(records)} balanced records available "
            f"(requested {cfg.ver_n_samples}; limited by the smaller class "
            f"in MultiClaim).[/dim]")
    console.print(
        f"\n[bold cyan]Claim veracity evaluation[/bold cyan]\n"
        f"[dim]{len(records)} balanced MultiClaim records "
        f"(True={n_true_recs}, False={n_false_recs}; "
        f"Disputed pre-filtered from {len(all_records)} total)[/dim]")

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
    # Mean confidence per accepted stage — surfaced in the breakdown table so
    # the user can see whether MultiClaim was over-confident (a known failure
    # mode the confidence-gated cascade is designed to mitigate).
    stage_conf_sum: dict[str, float] = defaultdict(float)

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

            pred_norm, pred_conf, stage = _verdict_with_fallback(
                paraphrase, source_id,
                cfg=cfg, embedder=embedder, llm=llm, kb=kb,
                embedder_name=cfg.camp_embedder,
            )

            stage_counts[stage] += 1
            if stage == "miss":
                # Miss counts as a wrong prediction for accuracy. We log under
                # the gold-label row but against a synthetic "Disputed" pred
                # bucket so the confusion row total equals
                # per_label_n[gold_norm].
                confusion[gold_norm]["Disputed"] += 1
                misses_by_label[gold_norm] += 1
            else:
                confusion[gold_norm][pred_norm] += 1
                # Track the per-stage confidence distribution for the report.
                stage_conf_sum[stage] += pred_conf
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
    stage_t.add_column("Mean conf", justify="right")
    for stage in ("multiclaim", "wikipedia", "web", "miss"):
        n = stage_counts.get(stage, 0)
        share = (n / n_all) if n_all else 0.0
        mean_conf = (stage_conf_sum.get(stage, 0.0) / n) if n and stage != "miss" else 0.0
        mean_conf_cell = f"{mean_conf:.3f}" if (stage != "miss" and n) else "—"
        stage_t.add_row(stage, str(n), f"{share:.1%}", mean_conf_cell)
    console.print(
        f"[dim]Evidence-source cascade breakdown "
        f"(accept iff verdict ∈ True/False AND confidence ≥ "
        f"{cfg.ver_confidence_threshold:.2f}):[/dim]")
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
                    "sources": "multiclaim→wikipedia→web (confidence-gated)",
                    "conf_threshold": cfg.ver_confidence_threshold},
            scores={"accuracy": accuracy,
                    "macro_f1": metrics["macro_f1"],
                    "coverage": coverage,
                    "misses":   n_miss,
                    "n": n_all},
        )
    except Exception as exc:
        logger.warning("[ver-eval] save_eval_stats failed: %s", exc)

    from core.eval.report_paths import report_path
    html_out = report_path(
        "claim-veracity",
        extra=f"{cfg.ver_generator}__{_PRECISION}")
    console.save_html(str(html_out), theme=MONOKAI, clear=False)
    console.print(f"[dim]Results → {csv_path}  HTML → {html_out}[/dim]")
