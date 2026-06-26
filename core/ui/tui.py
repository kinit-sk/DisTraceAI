"""Rich terminal UI: arrow/Enter navigation, per-setting side descriptions, and
an editable pre-launch review screen.

Raw-key primitives are cross-platform (termios / msvcrt). The settings editor
and the pre-launch screen share one component; the pre-launch screen restricts
the visible settings to those relevant to the action and offers a Launch row.

Dynamic parameter groups
------------------------
Some steps (narrative retrieval) have method-dependent parameters. The
``DYNAMIC_FOLLOWERS`` dict maps a "selector" field name to a callable that
returns the list of dependent fields given the current selector value. When the
selector is changed in the editor, the visible key list is rebuilt immediately.
This means the user first sees common params + the method selector, picks a
method, and the screen refreshes to show only the relevant extra params.
"""
from __future__ import annotations

import os
import sys
import textwrap

from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text

console = Console()

# Maps action keys back to (step, action) for stats lookup.
_STATS_KEY: dict[str, tuple[str, str]] = {
    "claim-detection":         ("claim-detection",   "eval"),
    "claim-canonization-eval": ("claim-canonization","eval"),
    "claim-canonization":      ("claim-canonization","generate"),
    "sub-narratives":          ("sub-narratives",    "eval"),
    "narratives-eval":         ("narratives",        "eval"),
    "narratives-generate":     ("narratives",        "generate"),
    "claim-veracity-eval":     ("claim-veracity",    "eval"),
    "campaigns-verify":        ("claim-veracity",    "generate"),
    "campaigns-deep-verify":   ("claim-veracity",    "generate"),
    "campaigns-eval":          ("campaigns",         "eval"),
    "campaigns-generate":      ("campaigns",         "generate"),
}

# ---------------------------------------------------------------------------
# Parameter lists per action
# ---------------------------------------------------------------------------

_NAR_COMMON = [
    "nar_detector",
    "nar_embedder",
    "nar_extractor",
]

_NAR_METHOD_PARAMS: dict[str, list[str]] = {
    "dense":     ["nar_dense_repr"],
    "bm25-rag":  [],
    "specfi-cs": ["nar_generator", "nar_specfi_hypotheticals"],
    "specfi-ccs": ["nar_generator", "nar_specfi_hypotheticals"],
    "cspecfi":   ["nar_generator", "nar_specfi_hypotheticals"],
    "context-1": ["nar_generator", "nar_context1_context_size",
                  "nar_context1_max_turns", "nar_context1_token_budget"],
}

_NAR_GENERATE_EXTRA = [
    "nar_assign_threshold",
    "nar_min_new_size",
    "nar_new_threshold",
    "nar_recluster_cadence",
]

_CAT_COLOR = {
    "detector":  "bright_cyan",
    "canon":     "blue",
    "subnar":    "magenta",
    "nar":       "green",
    "camp":      "yellow",
    "ver":       "cyan",
    # Settings categories
    "llm":       "bright_white",
    "env_vllm":  "bright_magenta",
    "env_distr": "bright_green",
    "env_hf":    "bright_blue",
    "env_":      "dark_orange",   # fallback for all env_ keys
}

_CAMP_METHOD_PARAMS: dict[str, list[str]] = {
    "dense":     ["camp_dense_repr"],
    "bm25-rag":  [],
    "specfi-cs": ["camp_generator", "camp_specfi_hypotheticals"],
    "specfi-ccs": ["camp_generator", "camp_specfi_hypotheticals"],
    "cspecfi":   ["camp_generator", "camp_specfi_hypotheticals"],
    "context-1": ["camp_generator", "camp_context1_max_turns",
                  "camp_context1_token_budget"],
}

_CAMP_COMMON = ["camp_detector", "camp_embedder", "camp_extractor"]

_CAMP_GENERATE_EXTRA = [
    "camp_assign_threshold", "camp_min_new_size", "camp_new_threshold",
    "camp_recluster_cadence", "camp_coordination_threshold",
    "camp_veracity_threshold", "camp_n1_weight", "camp_n2_weight",
    "camp_n3_weight", "camp_n4_weight",
]

_VER_COMMON = [
    "ver_sources", "ver_generator",
    "ver_max_turns", "ver_token_budget",
    "ver_multiclaim_text_col", "ver_multiclaim_label_col",
]

RELEVANT: dict[str, list[str]] = {
    "claim-detection":         ["detector"],
    "claim-canonization":      ["canon_detector", "canon_generator", "canon_precision"],
    "claim-canonization-eval": [],
    "sub-narratives-eval":     ["subnar_detector", "subnar_embedder", "subnar_generator",
                                "subnar_precision", "subnar_min_similarity",
                                "subnar_min_claims", "subnar_hypotheticals"],
    "sub-narratives-generate": ["subnar_detector", "subnar_embedder", "subnar_generator",
                                "subnar_precision", "subnar_min_similarity",
                                "subnar_min_claims"],
    "narratives-eval":         _NAR_COMMON + ["nar_eval_split", "nar_eval_domain"],
    "narratives-generate":     _NAR_COMMON + _NAR_GENERATE_EXTRA,
    "campaigns-verify":        _VER_COMMON,
    "campaigns-deep-verify":   _VER_COMMON,
    "claim-veracity-eval":     _VER_COMMON + ["ver_n_samples", "ver_n_paraphrases",
                                              "ver_paraphrase_generator"],
    "campaigns-eval":          _CAMP_COMMON,
    "campaigns-generate":      _CAMP_COMMON + _CAMP_GENERATE_EXTRA,
}

DYNAMIC_FOLLOWERS: dict[str, object] = {
    "nar_extractor":  lambda val: _method_followers(val, _NAR_METHOD_PARAMS),
    "camp_extractor": lambda val: _method_followers(val, _CAMP_METHOD_PARAMS),
}


def _method_followers(value: str, method_params: dict[str, list[str]]) -> list[str]:
    if value == "all":
        seen: set[str] = set()
        out: list[str] = []
        for key, params in method_params.items():
            if key == "all":
                continue
            for p in params:
                if p not in seen:
                    seen.add(p)
                    out.append(p)
        return out
    return method_params.get(value, [])


# ---------------------------------------------------------------------------
# Raw key reading
# ---------------------------------------------------------------------------

def _flush_stdin() -> None:
    if os.name == "nt":
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getwch()
    else:
        try:
            import termios
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        except Exception:
            pass


def readkey() -> str:
    """Return 'up'|'down'|'left'|'right'|'enter'|'esc' or the raw char."""
    _flush_stdin()
    if os.name == "nt":
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            return {"H": "up", "P": "down", "K": "left", "M": "right"}.get(
                msvcrt.getwch(), "")
        if ch == "\r":
            return "enter"
        if ch == "\x1b":
            return "esc"
        return ch
    import tty, termios
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            if sys.stdin.read(1) == "[":
                return {"A": "up", "B": "down", "C": "right", "D": "left"}.get(
                    sys.stdin.read(1), "")
            return "esc"
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def arrow_menu(title: str, items: list[str], subtitle: str = "") -> int:
    """Highlighted arrow-key menu; returns the chosen index, or -1 on Esc.

    Also accepts number keys (1-9) as direct shortcuts.
    """
    idx = 0
    needs_redraw = True
    while True:
        if needs_redraw:
            console.clear()
            head = (f"[bold cyan]{title}[/bold cyan]"
                    + (f"\n[dim]{subtitle}[/dim]" if subtitle else ""))
            console.print(Panel(head, border_style="blue"))
            console.print()
            for i, item in enumerate(items):
                num = f"[dim]{i + 1}.[/dim] " if i < 9 else "   "
                if i == idx:
                    console.print(f"  [bold white on blue] › {item} [/bold white on blue]")
                else:
                    console.print(f"  {num}[dim]{item}[/dim]")
            console.print("\n[dim]↑ ↓ navigate  ·  1-9 jump  ·  Enter select  ·  Esc back[/dim]")
            needs_redraw = False

        key = readkey()
        if key == "up":
            idx = (idx - 1) % len(items); needs_redraw = True
        elif key == "down":
            idx = (idx + 1) % len(items); needs_redraw = True
        elif key == "enter":
            return idx
        elif key == "esc":
            return -1
        elif key.isdigit() and key != "0":
            n = int(key) - 1
            if n < len(items):
                return n


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_value(val) -> str:
    if isinstance(val, bool):
        return "[green]ON[/green]" if val else "[red]OFF[/red]"
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def _plain_value(val) -> str:
    if isinstance(val, bool):
        return "ON" if val else "OFF"
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def _cat_color(key: str) -> str:
    """Return a border colour for the right-panel based on the field prefix."""
    # Exact prefix matches first (longest prefix wins)
    for prefix in sorted(_CAT_COLOR, key=len, reverse=True):
        if key.startswith(prefix):
            return _CAT_COLOR[prefix]
    return "white"


def _edit_value(cfg, key: str) -> str:
    """Cooked-mode prompt for a free (non-choice) field."""
    if cfg.is_locked(key):
        return "Locked by a CLI argument for this run — not editable."
    console.print(f"\n[bold]{cfg.label(key)}[/bold] — current: {cfg.get(key)}")
    try:
        raw = input("  new value (blank to keep): ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""
    if not raw:
        return ""
    try:
        cfg.set(key, raw)
        return f"✓ {cfg.label(key)} updated."
    except (ValueError, TypeError):
        return f"Invalid value for {cfg.label(key)} — keeping previous."


# ---------------------------------------------------------------------------
# Dynamic key list builder
# ---------------------------------------------------------------------------

def _build_keys(base_keys: list[str], cfg) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for k in base_keys:
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
        if k in DYNAMIC_FOLLOWERS:
            for dep in DYNAMIC_FOLLOWERS[k](cfg.get(k)):
                if dep not in seen:
                    seen.add(dep)
                    out.append(dep)
    return out


# ---------------------------------------------------------------------------
# Settings editor
# ---------------------------------------------------------------------------

def edit_settings(cfg, base_keys: list[str], title: str, *,
                  allow_launch: bool = False,
                  save_on_exit: bool = True,
                  launch_desc: str | None = None,
                  stats_text: str | None = None) -> bool:
    """Two-panel editor. Returns True only if the user chose Launch."""
    idx = 0
    message = ""

    while True:
        keys = [k for k in _build_keys(base_keys, cfg) if k in cfg.field_names()]
        rows = keys + (["__launch__"] if allow_launch else [])
        idx = min(idx, len(rows) - 1)

        label_w = max((len(cfg.label(k)) for k in keys), default=10)

        console.clear()
        hint = ("↑ ↓ navigate  ·  ← →  cycle  ·  Enter  "
                + ("launch / " if allow_launch else "") + "select / edit  ·  "
                + ("R reset  ·  " if not allow_launch else "") + "Esc "
                + ("cancel (keeps edits)" if allow_launch else "save & back"))
        console.print(Panel(
            f"[bold cyan]{title}[/bold cyan]\n[dim]{hint}[/dim]",
            border_style="blue"))

        raw_rows = []
        for i, key in enumerate(rows):
            if key == "__launch__":
                raw_rows.append((i, "  ▶ Launch", "launch"))
                continue
            val = _plain_value(cfg.get(key))
            ch = cfg.choices(key)
            hint_n = (f"  ({ch.index(cfg.get(key)) + 1}/{len(ch)})"
                      if (ch and cfg.get(key) in ch) else "")
            lock = "  [locked]" if cfg.is_locked(key) else ""
            is_follower = (key not in base_keys)
            caret = "›" if i == idx else " "
            prefix = "  " if is_follower else ""
            raw_rows.append((
                i,
                f" {caret} {prefix}{cfg.label(key).ljust(label_w)}   {val}{hint_n}{lock}",
                "locked" if cfg.is_locked(key) else "follower" if is_follower else "normal",
            ))

        term_w   = console.width or 120
        left_max = min(max((len(s) for _, s, _ in raw_rows), default=20) + 1,
                       int(term_w * 0.60), 100)
        right_w  = max(30, term_w - left_max - 8)
        desc_w   = right_w - 4

        body = Text()
        for i, s, kind in raw_rows:
            line = s[:left_max].ljust(left_max)
            if i == idx:
                body.append(line,
                             style="bold black on cyan" if kind == "launch"
                             else "bold white on blue")
            elif kind == "launch":
                body.append(line, style="bold green")
            elif kind == "locked":
                body.append(line, style="dim")
            elif kind == "follower":
                body.append(line, style="dim cyan")
            else:
                body.append(line)
            body.append("\n")

        cur = rows[idx]
        if cur == "__launch__":
            if launch_desc and stats_text:
                sep = f"\n[dim]{'─' * (desc_w - 2)}[/dim]\n"
                desc = launch_desc + sep + stats_text
            elif launch_desc:
                desc = launch_desc
            else:
                stats_section = (
                    f"\n[dim]{'─' * (desc_w - 2)}[/dim]\n{stats_text}"
                    if stats_text else ""
                )
                desc = (
                    "[bold]Launch[/bold]\n"
                    "Start this run with the settings above."
                    + stats_section
                )
            border = "green"
        else:
            wrapped = textwrap.fill(cfg.desc(cur), width=desc_w) or "(no description)"
            ch = cfg.choices(cur)
            if isinstance(cfg.get(cur), bool):
                extra = "\n[dim]Enter / ← → toggles[/dim]"
            elif ch:
                extra = "\n[dim]← → cycles · Enter to pick from list[/dim]"
            else:
                extra = "\n[dim]Enter to edit value[/dim]"
            badge = ("[dim italic]method-specific parameter[/dim italic]\n"
                     if cur not in base_keys else "")
            # Special badge for env-var fields
            env_badge = ""
            if cur.startswith("env_"):
                from main import _SETTINGS_ADVANCED_KEYS, _SETTINGS_EMBEDDER_KEYS
                from config import Config as _Cfg
                env_var = _Cfg._ENV_FIELD_MAP.get(cur, "")
                env_badge = (f"[dim]env var:[/dim] [bold yellow]{env_var}[/bold yellow]\n"
                             if env_var else "")
            desc = (f"[bold]{cfg.label(cur)}[/bold]\n"
                    f"[dim]{'─' * (desc_w - 2)}[/dim]\n"
                    f"{badge}{env_badge}{wrapped}\n\n"
                    f"[dim]Current:[/dim] {_fmt_value(cfg.get(cur))}{extra}")
            border = _cat_color(cur)

        left  = Panel(body, title="[bold]Settings[/bold]", border_style="blue",
                      width=left_max + 4, padding=(0, 0))
        right = Panel(desc, title="[bold]Description[/bold]", border_style=border,
                      width=right_w, padding=(0, 1))
        console.print(Columns([left, right], equal=False, expand=False))
        if message:
            console.print(f"[yellow]{message}[/yellow]")

        key = readkey()
        cur = rows[idx]
        if key == "up":
            idx = (idx - 1) % len(rows); message = ""
        elif key == "down":
            idx = (idx + 1) % len(rows); message = ""
        elif key == "esc":
            if save_on_exit:
                cfg.save()
            return False
        elif cur == "__launch__":
            if key == "enter":
                if save_on_exit:
                    cfg.save()
                return True
        elif cfg.is_locked(cur):
            message = "🔒 Controlled by a CLI argument for this run."
        else:
            val = cfg.get(cur)
            ch = cfg.choices(cur)
            if isinstance(val, bool):
                if key in ("enter", "right", "left", " "):
                    cfg.cycle(cur, +1); message = ""
            elif ch:
                if key == "right" or key == " ":
                    cfg.cycle(cur, +1); message = ""
                elif key == "left":
                    cfg.cycle(cur, -1); message = ""
                elif key == "enter":
                    sel = arrow_menu(f"Select — {cfg.label(cur)}", ch,
                                     subtitle=cfg.desc(cur))
                    if sel >= 0:
                        cfg.set(cur, ch[sel])
                    message = ""
            elif key == "enter":
                message = _edit_value(cfg, cur)
            elif key == "r" and not allow_launch:
                if _confirm("Reset ALL settings to defaults?"):
                    cfg.reset()
                    message = "✓ Settings reset to defaults."
                else:
                    message = "Reset cancelled."


def _confirm(question: str) -> bool:
    """Inline y/N confirmation for destructive actions. Defaults to No."""
    try:
        ans = input(f"\n  {question} [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return ans in ("y", "yes")


# ---------------------------------------------------------------------------
# Launch-row descriptions for parameterless actions
# ---------------------------------------------------------------------------

_CANON_BENCH_DESC = (
    "[bold]Launch — Canonization benchmark[/bold]\n"
    f"[dim]{'─' * 44}[/dim]\n"
    "Runs the full canonization benchmark: every one of the 6 models "
    "is evaluated at bf16 precision.\n\n"
    "[bold]Models[/bold]\n"
    "Qwen3.5-2B, Qwen3.5-4B, Qwen3.5-9B,\n"
    "Gemma-4-E2B-IT, Gemma-4-E4B-IT, Gemma-4-12B-IT\n\n"
    "[bold]Precision[/bold]\n"
    "bf16 (16-bit)\n\n"
    "[dim]6 models × 1 precision = 6 combinations.\n"
    "Results → evaluation/eval_claim_canonization.html[/dim]"
)

LAUNCH_DESC: dict[str, str] = {
    "claim-canonization-eval": _CANON_BENCH_DESC,
}


# ---------------------------------------------------------------------------
# Top-level screens
# ---------------------------------------------------------------------------

def prelaunch_review(cfg, action: str) -> bool:
    """Pre-launch parameter review for an action."""
    base_keys = RELEVANT.get(action, cfg.field_names())
    base_keys = [k for k in base_keys if k in cfg.field_names()]

    stats_text: str | None = None
    if action in _STATS_KEY:
        try:
            from core.ui.stats import get_stats
            step, act = _STATS_KEY[action]
            stats_text = get_stats(step, act, cfg)
        except Exception:
            pass

    return edit_settings(
        cfg, base_keys,
        f"Review settings — {action}",
        allow_launch=True,
        save_on_exit=True,
        launch_desc=LAUNCH_DESC.get(action),
        stats_text=stats_text,
    )
