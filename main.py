"""DisTraceAI — entry point.

Main menu
---------
1. Claim detection
2. Claim canonization
3. Claim veracity estimation
4. Sub-narratives extraction
5. Narrative extraction
6. Campaigns extraction
7. Settings

Steps 1-5 expose two sub-menu actions:
  • Evaluation  — run the evaluation module for that step
  • Generate    — run the generation / extraction module for that step

Campaigns extraction (6) has a four-item sub-menu instead:
  • Verify hierarchy  — verify central claims of the existing hierarchy
  • Deep verify       — verify central + underlying claims
  • Evaluation        — clustering metrics against FakeCTI ground truth
  • Generate Dataset  — run the full pipeline on EUvsDisinfo and export CSVs

Settings (7) exposes three sub-menus:
  • LLM Backend       — switch between vLLM and llama-cpp + GGUF quant
  • Embedder & Memory — embedder device, precision, batching, sequence length
  • Advanced / Env    — all remaining OS environment-variable overrides
"""
from __future__ import annotations

import os
import argparse
import logging

from config import Config

# Environment defaults are now managed by Config.load() / Config._apply_env_fields().
# We keep the two most critical ones here so they fire before any heavy import
# (vllm / transformers) that might be triggered by top-level module imports on
# some code paths — but Config will also set them correctly once loaded.
os.environ.setdefault("DISABLE_KERNEL_MAPPING", "1")
os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

STEPS = [
    "claim-detection",
    "claim-canonization",
    "sub-narratives",
    "claim-veracity",
    "narratives",
    "campaigns",
]

STEP_LABELS = {
    "claim-detection":    "Claim detection",
    "claim-canonization": "Claim canonization",
    "sub-narratives":     "Sub-narratives extraction",
    "claim-veracity":     "Claim veracity estimation",
    "narratives":         "Narrative extraction",
    "campaigns":          "Campaigns extraction",
}

STEP_PARAMS: dict[str, list[str]] = {
    "claim-detection":    ["detector"],
    "claim-canonization": ["canon_detector", "canon_generator", "canon_precision"],
    "sub-narratives":     ["subnar_detector", "subnar_embedder", "subnar_generator",
                           "subnar_precision", "subnar_min_similarity",
                           "subnar_min_claims"],
    "claim-veracity":     [],
    "narratives":         ["nar_detector", "nar_extractor", "nar_dense_repr",
                           "nar_embedder", "nar_generator", "nar_precision",
                           "nar_assign_threshold", "nar_min_new_size",
                           "nar_new_threshold", "nar_recluster_cadence",
                           "nar_specfi_hypotheticals", "nar_context1_max_turns",
                           "nar_context1_token_budget"],
    "campaigns":          [],
}

STEP_EVAL_PARAMS: dict[str, list[str]] = {
    "sub-narratives": ["subnar_detector", "subnar_embedder", "subnar_generator",
                       "subnar_precision", "subnar_min_similarity",
                       "subnar_min_claims", "subnar_hypotheticals"],
    "narratives":     ["nar_detector", "nar_extractor", "nar_dense_repr",
                       "nar_embedder", "nar_generator", "nar_precision",
                       "nar_specfi_hypotheticals", "nar_context1_max_turns",
                       "nar_context1_token_budget", "nar_eval_split"],
}

EVAL_MODULES: dict[str, str] = {
    "claim-detection":    "core.eval.eval_claim_detection",
    "claim-canonization": "core.eval.eval_claim_canonization",
    "sub-narratives":     "core.eval.eval_sub_narratives",
    "claim-veracity":     "core.eval.eval_claim_veracity",
    "narratives":         "core.eval.eval_narratives",
    "campaigns":          "core.eval.eval_campaigns",
}

# ---------------------------------------------------------------------------
# Settings field groups
# ---------------------------------------------------------------------------

# Fields shown in the "LLM Backend" settings sub-menu
_SETTINGS_BACKEND_KEYS = [
    "llm_backend",
]

# Fields shown in the "Embedder & Memory" settings sub-menu
_SETTINGS_EMBEDDER_KEYS = [
    "env_distrace_embed_maxlen",
    "env_distrace_embed_gpu_util",   # vLLM
    "env_distrace_gen_gpu_util",     # vLLM
    "env_distrace_embedder_device",  # llama-cpp
    "env_distrace_embed_fp32",       # llama-cpp
    "env_distrace_encode_batch",     # llama-cpp
    "env_distrace_noderag_workers",  # llama-cpp
    "env_distrace_cw_cpu",
]

# Fields shown in the "Advanced / Env" settings sub-menu
_SETTINGS_ADVANCED_KEYS = [
    "env_vllm_deep_gemm_warmup",
    "env_vllm_use_deep_gemm",
    "env_use_flashinfer_sampler",
    "env_disable_kernel_mapping",
    "env_distrace_show_tqdm",
    "env_hf_hub_download_timeout",
    "env_hf_token",
    "env_distrace_noderag_maxtok",
    "env_distrace_noderag_dim",
    "env_distrace_noderag_chunk",
    "env_distrace_noderag_lang",
    "env_distrace_noderag_rate",
    "env_distrace_noderag_lang_filter",
    "env_distrace_nar_no_llm",
    "env_distrace_nar_noderag_index_root",
]

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
    from core.ui.stats import save_generate_stats
    from rich.console import Console
    console = Console()

    kb = KnowledgeBase(Path("knowledge"))

    if step == "claim-detection":
        from core.claims.cw_detector import CheckWorthinessDetector
        from core.gen.gen_cw_detect import generate
        console.print(f"\n[bold cyan]Claim detection — Generate[/bold cyan]")
        console.print(f"[dim]Detector: {cfg.detector}[/dim]\n")
        detector = CheckWorthinessDetector(cfg.detector)
        summary = generate(detector, kb)
        console.print("\n[bold]Summary:[/bold]")
        for dataset, counts in summary.items():
            console.print(f"  {dataset}: {counts}")
        save_generate_stats(step, summary)
    elif step == "claim-canonization":
        from core.gen.gen_canonize import canonize
        console.print(f"\n[bold cyan]Claim canonization — Generate[/bold cyan]")
        console.print(f"[dim]Detector: {cfg.canon_detector}  Generator: {cfg.canon_generator}  Precision: {cfg.canon_precision}[/dim]\n")
        summary = canonize(cfg.canon_detector, cfg.canon_generator, cfg.canon_precision, kb)
        console.print("\n[bold]Summary:[/bold]")
        for dataset, counts in summary.items():
            console.print(f"  {dataset}: {counts}")
        save_generate_stats(step, summary)
    elif step == "sub-narratives":
        from core.gen.gen_sub_narratives import generate as generate_sub_narratives
        console.print(f"\n[bold cyan]Sub-narratives — Generate[/bold cyan]")
        console.print(
            f"[dim]Detector: {cfg.subnar_detector}  Embedder: {cfg.subnar_embedder}  "
            f"Generator: {cfg.subnar_generator} "
            f"MinSim: {cfg.subnar_min_similarity}  MinClaims: {cfg.subnar_min_claims}[/dim]\n"
        )
        summary = generate_sub_narratives(
            detector_path=cfg.subnar_detector,
            embedder_name=cfg.subnar_embedder,
            generator_key=cfg.subnar_generator,
            kb=kb,
            min_similarity=cfg.subnar_min_similarity,
            min_claims=cfg.subnar_min_claims,
        )
        console.print("\n[bold]Summary:[/bold]")
        for dataset, det_map in summary.items():
            for detector, counts in det_map.items():
                console.print(f"  {dataset}/{detector}: {counts}")
        save_generate_stats(step, summary)
    elif step == "narratives":
        from core.gen.gen_narratives import generate as generate_narratives
        if cfg.nar_extractor == "all":
            console.print(
                "[red]nar_extractor='all' is an Evaluation-only benchmark mode "
                "and cannot be used for Generate.[/red] "
                "Pick a concrete method (dense / bm25-rag / specfi-cs / "
                "specfi-ccs / cspecfi / context-1) before generating.")
            return
        console.print(f"\n[bold cyan]Narratives — Generate[/bold cyan]")
        console.print(
            f"[dim]Detector: {cfg.nar_detector}  Method: {cfg.nar_extractor}  "
            f"Embedder: {cfg.nar_embedder}  Generator: {cfg.nar_generator}  "
            f"AssignThr: {cfg.nar_assign_threshold}  "
            f"MinNew: {cfg.nar_min_new_size}  NewThr: {cfg.nar_new_threshold}  "
            f"Cadence: {cfg.nar_recluster_cadence}[/dim]\n")
        summary = generate_narratives(
            detector_path=cfg.nar_detector,
            extractor=cfg.nar_extractor,
            embedder_name=cfg.nar_embedder,
            generator_key=cfg.nar_generator,
            kb=kb,
            cfg=cfg,
        )
        console.print("\n[bold]Summary:[/bold]")
        for dataset, det_map in summary.items():
            for detector, counts in det_map.items():
                console.print(f"  {dataset}/{detector}: {counts}")
        save_generate_stats(step, summary)
    elif step == "claim-veracity":
        from core.gen.gen_veracity import verify_hierarchy
        console.print(f"\n[bold cyan]Claim veracity — Verify hierarchy[/bold cyan]")
        console.print(
            f"[dim]Sources: {cfg.ver_sources}  Generator: {cfg.ver_generator}[/dim]\n")
        summary = verify_hierarchy(kb, cfg, deep=False)
        save_generate_stats(step, summary)
    elif step == "campaigns":
        from core.gen.gen_campaigns import generate as gen_camp
        from core.knowledge_base import DATASET_FAKECTI, DATASET_POLYNARRATIVE
        console.print(f"\n[bold cyan]Campaigns — Generate[/bold cyan]")
        console.print(
            f"[dim]Detector: {cfg.camp_detector}  Extractor: {cfg.camp_extractor}  "
            f"Embedder: {cfg.camp_embedder}[/dim]\n")
        summary = {}
        for dataset in (DATASET_FAKECTI, DATASET_POLYNARRATIVE):
            result = gen_camp(
                dataset=dataset,
                detector_path=cfg.camp_detector,
                extractor=cfg.camp_extractor,
                embedder_name=cfg.camp_embedder,
                generator_key=cfg.camp_generator,
                kb=kb, cfg=cfg,
            )
            if result:
                summary[dataset] = result
        save_generate_stats(step, summary)
    else:
        print(f"Generate not yet implemented for step '{step}'.")


def _campaigns_submenu(cfg: Config) -> None:
    """4-item campaigns submenu: Verify / Deep Verify / Evaluation / Generate Dataset."""
    from core.ui import tui as ui
    from pathlib import Path
    from core.knowledge_base import KnowledgeBase

    items = ["Verify hierarchy", "Deep verify", "Evaluation",
             "Generate Dataset", "← Back"]
    while True:
        choice = ui.arrow_menu("Campaigns extraction", items)
        if choice < 0 or choice == 4:
            return

        if choice == 0:
            if ui.prelaunch_review(cfg, "campaigns-verify"):
                from core.gen.gen_veracity import verify_hierarchy
                kb = KnowledgeBase(Path("knowledge"))
                summary = verify_hierarchy(kb, cfg, deep=False)
                from core.ui.stats import save_generate_stats
                save_generate_stats("claim-veracity", summary)
                input("\n[done] press Enter to continue…")

        elif choice == 1:
            if ui.prelaunch_review(cfg, "campaigns-deep-verify"):
                from core.gen.gen_veracity import verify_hierarchy
                kb = KnowledgeBase(Path("knowledge"))
                summary = verify_hierarchy(kb, cfg, deep=True)
                from core.ui.stats import save_generate_stats
                save_generate_stats("claim-veracity", {"deep": summary})
                input("\n[done] press Enter to continue…")

        elif choice == 2:
            if ui.prelaunch_review(cfg, "campaigns-eval"):
                from core.eval.eval_campaigns import main as eval_camp
                eval_camp(cfg)
                input("\n[done] press Enter to continue…")

        elif choice == 3:
            if ui.prelaunch_review(cfg, "campaigns-generate"):
                from core.gen.gen_dataset import generate_dataset
                summary = generate_dataset(cfg)
                from core.ui.stats import save_generate_stats
                save_generate_stats("campaigns", summary)
                input("\n[done] press Enter to continue…")


# ---------------------------------------------------------------------------
# Settings menu
# ---------------------------------------------------------------------------

def _settings_menu(cfg: Config) -> None:
    """Three-panel Settings menu: Backend / Embedder & Memory / Advanced."""
    from core.ui import tui as ui

    items = [
        "LLM Backend",
        "Embedder & Memory",
        "Advanced / Environment",
        "← Back",
    ]

    while True:
        # Show current backend in the subtitle for quick orientation
        subtitle = f"Active backend: [bold]{cfg.llm_backend}[/bold]"
        choice = ui.arrow_menu("Settings", items, subtitle=subtitle)
        if choice < 0 or choice == 3:
            return

        if choice == 0:
            _settings_backend(cfg, ui)
        elif choice == 1:
            _settings_embedder(cfg, ui)
        elif choice == 2:
            _settings_advanced(cfg, ui)


def _settings_backend(cfg: Config, ui) -> None:
    """Backend selector sub-menu."""
    keys = [k for k in _SETTINGS_BACKEND_KEYS if k in cfg.field_names()]
    ui.edit_settings(
        cfg, keys,
        "Settings › LLM Backend",
        allow_launch=False,
        save_on_exit=True,
    )
    # Inform the user they may need to switch conda envs
    from rich.console import Console
    from rich.panel import Panel
    Console().print(Panel(
        f"[bold]Active backend set to:[/bold] [cyan]{cfg.llm_backend}[/cyan]\n\n"
        "If you changed the backend, remember to activate the matching conda env "
        "before running any pipeline step:\n\n"
        "  [dim]vLLM:[/dim]      [green]conda activate distrace-vllm[/green]\n"
        "  [dim]llama-cpp:[/dim] [green]conda activate distrace-llama[/green]",
        title="[bold yellow]⚠  Conda environment reminder[/bold yellow]",
        border_style="yellow",
    ))
    input("\nPress Enter to continue…")


def _settings_embedder(cfg: Config, ui) -> None:
    """Embedder & memory settings sub-menu."""
    keys = [k for k in _SETTINGS_EMBEDDER_KEYS if k in cfg.field_names()]
    ui.edit_settings(
        cfg, keys,
        "Settings › Embedder & Memory",
        allow_launch=False,
        save_on_exit=True,
    )


def _settings_advanced(cfg: Config, ui) -> None:
    """Advanced / environment-variable settings sub-menu."""
    keys = [k for k in _SETTINGS_ADVANCED_KEYS if k in cfg.field_names()]
    ui.edit_settings(
        cfg, keys,
        "Settings › Advanced / Environment",
        allow_launch=False,
        save_on_exit=True,
    )


# ---------------------------------------------------------------------------
# TUI
# ---------------------------------------------------------------------------

def _step_submenu(step: str, cfg: Config) -> None:
    """Arrow-key sub-menu for a single pipeline step."""
    from core.ui import tui as ui

    if step == "campaigns":
        _campaigns_submenu(cfg)
        return

    gen_params  = STEP_PARAMS.get(step, [])
    eval_params = STEP_EVAL_PARAMS.get(step, gen_params)

    if step == "claim-veracity":
        eval_key = f"{step}-eval" if f"{step}-eval" in ui.RELEVANT else step
        review_needed = (f"{step}-eval" in ui.RELEVANT
                         or eval_key in ui.RELEVANT
                         or bool(eval_params))
        if review_needed and not ui.prelaunch_review(cfg, eval_key):
            return
        run_eval(step, cfg)
        input("\n[done] press Enter to continue…")
        return

    menu_items = ["Evaluation", "Generate", "← Back"]
    eval_key = f"{step}-eval" if f"{step}-eval" in ui.RELEVANT else step
    gen_key  = f"{step}-generate" if f"{step}-generate" in ui.RELEVANT else step

    while True:
        choice = ui.arrow_menu(STEP_LABELS[step], menu_items)

        if choice < 0 or choice == 2:
            return

        if choice == 0:
            review_needed = (f"{step}-eval" in ui.RELEVANT
                             or eval_key in ui.RELEVANT
                             or bool(eval_params))
            if review_needed and not ui.prelaunch_review(cfg, eval_key):
                continue
            run_eval(step, cfg)
            input("\n[done] press Enter to continue…")

        elif choice == 1:
            review_needed = (f"{step}-generate" in ui.RELEVANT
                             or gen_key in ui.RELEVANT
                             or bool(gen_params))
            if review_needed and not ui.prelaunch_review(cfg, gen_key):
                continue
            run_generate(step, cfg)
            input("\n[done] press Enter to continue…")


def tui(cfg: Config) -> None:
    from core.ui import tui as ui

    main_items = [STEP_LABELS[s] for s in STEPS] + ["Settings", "Quit"]
    settings_idx = len(STEPS)
    quit_idx     = len(STEPS) + 1

    while True:
        choice = ui.arrow_menu(
            "DisTraceAI", main_items,
            subtitle=f"Campaign detection pipeline  ·  backend: {cfg.llm_backend}")
        if choice < 0 or choice == quit_idx:
            return
        if choice == settings_idx:
            _settings_menu(cfg)
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
    parser.add_argument("--generate-dataset", action="store_true",
                        help="run full pipeline on EUvsDisinfo and export CSVs")
    Config.add_cli_arguments(parser)
    args = parser.parse_args()

    cfg = Config.load()
    cfg.apply_cli(args)

    if args.eval:
        run_eval(args.eval, cfg)
    elif args.generate:
        run_generate(args.generate, cfg)
    elif args.generate_dataset:
        from core.gen.gen_dataset import generate_dataset
        generate_dataset(cfg)
    else:
        tui(cfg)


if __name__ == "__main__":
    main()
