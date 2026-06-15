"""DisTraceAI — entry point.

Main menu
---------
1. Claim detection
2. Claim canonization
3. Claim veracity estimation
4. Sub-narratives extraction
5. Narrative extraction
6. Campaigns extraction
7. Full pipeline - Dataset compilation

Items 1–6 expose two sub-menu actions:
  • Evaluation  — run the evaluation module for that step
  • Generate    — run the generation / extraction module for that step

Item 7 runs the full end-to-end pipeline.
"""
from __future__ import annotations

import argparse
import logging

from config import Config

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

STEPS = [
    "claim-detection",
    "claim-canonization",
    "claim-veracity",
    "sub-narratives",
    "narratives",
    "campaigns",
]

STEP_LABELS = {
    "claim-detection":    "Claim detection",
    "claim-canonization": "Claim canonization",
    "claim-veracity":     "Claim veracity estimation",
    "sub-narratives":     "Sub-narratives extraction",
    "narratives":         "Narrative extraction",
    "campaigns":          "Campaigns extraction",
}

# Config fields shown in the pre-launch review for each step
STEP_PARAMS: dict[str, list[str]] = {
    "claim-detection":    ["detector"],
    "claim-canonization": ["canon_detector", "canon_generator", "canon_quantization"],
    "claim-veracity":     [],
    "sub-narratives":     ["subnar_detector", "subnar_embedder", "subnar_generator",
                           "subnar_quantization", "subnar_min_similarity",
                           "subnar_min_claims"],
    "narratives":         [],
    "campaigns":          [],
}

# Evaluation-specific param lists (when they differ from generate params).
# Steps not listed here reuse STEP_PARAMS for both actions.
STEP_EVAL_PARAMS: dict[str, list[str]] = {
    "sub-narratives": ["subnar_detector", "subnar_embedder", "subnar_generator",
                       "subnar_quantization", "subnar_min_similarity",
                       "subnar_min_claims", "subnar_hypotheticals"],
}

# Evaluation module for each step
EVAL_MODULES: dict[str, str] = {
    "claim-detection":    "evaluation.eval_claim_detection",
    "claim-canonization": "evaluation.eval_claim_canonization",
    "claim-veracity":     "evaluation.eval_claim_veracity",
    "sub-narratives":     "evaluation.eval_sub_narratives",
    "narratives":         "evaluation.eval_narratives",
    "campaigns":          "evaluation.eval_campaigns",
}


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------

def run_eval(step: str, cfg: Config) -> None:
    import importlib
    mod = EVAL_MODULES.get(step)
    if not mod:
        print(f"[error] No evaluation module registered for '{step}'.")
        return
    importlib.import_module(mod).main(cfg)


def run_generate(step: str, cfg: Config) -> None:
    from pathlib import Path
    from core.knowledge_base import KnowledgeBase
    from rich.console import Console
    console = Console()

    kb = KnowledgeBase(Path("knowledge"))

    if step == "claim-detection":
        from core.claims.cw_detector import CheckWorthinessDetector
        from core.claims.gen_cw_detect import generate
        console.print(f"\n[bold cyan]Claim detection — Generate[/bold cyan]")
        console.print(f"[dim]Detector: {cfg.detector}[/dim]\n")
        detector = CheckWorthinessDetector(cfg.detector)
        summary = generate(detector, kb)
        console.print("\n[bold]Summary:[/bold]")
        for dataset, counts in summary.items():
            console.print(f"  {dataset}: {counts}")
    elif step == "claim-canonization":
        from core.claims.gen_canonize import canonize
        console.print(f"\n[bold cyan]Claim canonization — Generate[/bold cyan]")
        console.print(f"[dim]Detector: {cfg.canon_detector}  Generator: {cfg.canon_generator}  Quant: {cfg.canon_quantization}[/dim]\n")
        summary = canonize(cfg.canon_detector, cfg.canon_generator, cfg.canon_quantization, kb)
        console.print("\n[bold]Summary:[/bold]")
        for dataset, counts in summary.items():
            console.print(f"  {dataset}: {counts}")
    elif step == "sub-narratives":
        from core.claims.gen_sub_narratives import generate as generate_sub_narratives
        console.print(f"\n[bold cyan]Sub-narratives — Generate[/bold cyan]")
        console.print(
            f"[dim]Detector: {cfg.subnar_detector}  Embedder: {cfg.subnar_embedder}  "
            f"Generator: {cfg.subnar_generator}  Quant: {cfg.subnar_quantization}  "
            f"MinSim: {cfg.subnar_min_similarity}  MinClaims: {cfg.subnar_min_claims}[/dim]\n"
        )
        summary = generate_sub_narratives(
            detector_path=cfg.subnar_detector,
            embedder_name=cfg.subnar_embedder,
            generator_key=cfg.subnar_generator,
            quantization=cfg.subnar_quantization,
            kb=kb,
            min_similarity=cfg.subnar_min_similarity,
            min_claims=cfg.subnar_min_claims,
        )
        console.print("\n[bold]Summary:[/bold]")
        for dataset, det_map in summary.items():
            for detector, counts in det_map.items():
                console.print(f"  {dataset}/{detector}: {counts}")
    else:
        print(f"Generate not yet implemented for step '{step}'.")


def run_full_pipeline(cfg: Config) -> None:
    from rich.console import Console
    Console().print("[yellow]Full pipeline not yet implemented.[/yellow]")


# ---------------------------------------------------------------------------
# TUI
# ---------------------------------------------------------------------------

def _step_submenu(step: str, cfg: Config) -> None:
    """Arrow-key sub-menu for a single pipeline step."""
    from core.ui import tui as ui

    gen_params  = STEP_PARAMS.get(step, [])
    eval_params = STEP_EVAL_PARAMS.get(step, gen_params)
    menu_items  = ["Evaluation", "Generate", "← Back"]

    # Canonization Evaluation is a fixed full benchmark with its own key.
    eval_key = f"{step}-eval" if step == "claim-canonization" else step

    while True:
        choice = ui.arrow_menu(STEP_LABELS[step], menu_items)

        if choice < 0 or choice == 2:
            return

        if choice == 0:
            review_needed = (step == "claim-canonization") or bool(eval_params)
            if review_needed and not ui.prelaunch_review(cfg, eval_key):
                continue
            run_eval(step, cfg)
            input("\n[done] press Enter to continue…")

        elif choice == 1:
            if gen_params and not ui.prelaunch_review(cfg, step):
                continue
            run_generate(step, cfg)
            input("\n[done] press Enter to continue…")


def tui(cfg: Config) -> None:
    from core.ui import tui as ui

    main_items = [STEP_LABELS[s] for s in STEPS] + \
                 ["Full pipeline - Dataset compilation", "Quit"]
    full_pipeline_idx = len(STEPS)
    quit_idx          = len(STEPS) + 1

    while True:
        choice = ui.arrow_menu("DisTraceAI", main_items,
                               subtitle="Campaign detection pipeline")
        if choice < 0 or choice == quit_idx:
            return

        if choice == full_pipeline_idx:
            if ui.prelaunch_review(cfg, "pipeline"):
                run_full_pipeline(cfg)
                input("\n[done] press Enter to continue…")
        elif 0 <= choice < len(STEPS):
            _step_submenu(STEPS[choice], cfg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(prog="distrace",
                                     description="DisTraceAI — campaign detection pipeline")
    parser.add_argument("--eval",     choices=STEPS, metavar="STEP",
                        help="run evaluation for a step non-interactively")
    parser.add_argument("--generate", choices=STEPS, metavar="STEP",
                        help="run generation for a step non-interactively")
    parser.add_argument("--pipeline", action="store_true",
                        help="run the full pipeline non-interactively")
    Config.add_cli_arguments(parser)
    args = parser.parse_args()

    cfg = Config.load()
    cfg.apply_cli(args)

    if args.eval:
        run_eval(args.eval, cfg)
    elif args.generate:
        run_generate(args.generate, cfg)
    elif args.pipeline:
        run_full_pipeline(cfg)
    else:
        tui(cfg)


if __name__ == "__main__":
    main()
