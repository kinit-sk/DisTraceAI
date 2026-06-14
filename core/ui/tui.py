"""Rich terminal UI: arrow/Enter navigation, per-setting side descriptions, and
an editable pre-launch review screen (README §4).

Raw-key primitives are cross-platform (termios / msvcrt). The settings editor
and the pre-launch screen share one component; the pre-launch screen restricts
the visible settings to those relevant to the action and offers a Launch row.
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

# Settings shown on the pre-launch review screen, per action.
RELEVANT = {
    "pipeline":  ["enable_veracity", "retrieval_backend",
                  "generator_model", "assignment_threshold", "enable_reclustering",
                  "coord_threshold", "campaign_min_narratives"],
    "dataset":   ["massivesumm_src", "massivesumm_source", "massivesumm_hf_repo",
                  "massivesumm_languages", "massivesumm_rebuild", "dataset_out",
                  "enable_veracity", "retrieval_backend",
                  "embedder_model", "generator_model", "generator_quant",
                  "generator_workers", "generator_server_url", "cw_model",
                  "assignment_threshold", "new_narrative_threshold",
                  "min_sns_per_narrative", "enable_reclustering",
                  "campaign_grouping", "campaign_linkage", "campaign_group_threshold",
                  "campaign_min_narratives", "coord_burst_window_hours",
                  "coord_neardup_threshold", "coord_threshold"],
    "claim-extraction": ["detector"],
    "retrieval": ["noderag_index", "polynm_data_dir",
                  "retrieval_domain", "retrieval_languages", "retrieval_eval_split",
                  "retrieval_query_source", "retrieval_query_min_claims",
                  "retrieval_query_rebuild",
                  "retrieval_rebuild_index", "sfc_n_hypotheticals", "sfc_temperature",
                  "generator_context_size", "context1_context_size",
                  "embedder_model", "generator_model", "context1_quant",
                  "cw_model", "generator_workers", "generator_server_url"],
    "veracity":  ["enable_veracity", "context1_model", "context1_quant", "context1_context_size",
                  "generator_model", "embedder_model", "generator_workers", "generator_server_url"],
    "campaigns": ["campaign_method",
                  "fakecti_type", "fakecti_min_campaign", "fakecti_rebuild", "fakecti_src",
                  "embedder_model", "generator_model", "generator_quant",
                  "generator_workers", "generator_server_url", "generator_context_size",
                  "cw_model",
                  "noderag_index", "sfc_n_hypotheticals", "sfc_temperature",
                  "assignment_threshold", "new_narrative_threshold", "min_sns_per_narrative",
                  "campaign_grouping", "campaign_linkage", "campaign_group_threshold", "campaign_min_narratives",
                  "coord_burst_window_hours", "coord_neardup_threshold", "coord_threshold",
                  "campaign_min_total_narratives"],
    "cw":        ["cw_model"],
}

_CAT_COLOR = {
    "detector": "bright_cyan",
    "retrieval": "green", "assignment": "green", "noderag": "bright_blue",
    "sfc": "bright_blue", "bench": "white", "coord": "magenta",
    "campaign": "magenta", "recluster": "yellow", "fakecti": "cyan",
    "embedder": "blue", "generator": "blue", "cw": "blue", "context1": "blue",
}


# ── raw key reading ──────────────────────────────────────────────────────────
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
            return {"H": "up", "P": "down", "K": "left", "M": "right"}.get(msvcrt.getwch(), "")
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
                return {"A": "up", "B": "down", "C": "right", "D": "left"}.get(sys.stdin.read(1), "")
            return "esc"
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def arrow_menu(title: str, items: list[str], subtitle: str = "") -> int:
    """Highlighted arrow-key menu; returns the chosen index, or -1 on Esc."""
    idx = 0
    while True:
        console.clear()
        head = f"[bold cyan]{title}[/bold cyan]" + (f"\n[dim]{subtitle}[/dim]" if subtitle else "")
        console.print(Panel(head, border_style="blue"))
        console.print()
        for i, item in enumerate(items):
            if i == idx:
                console.print(f"  [bold white on blue] › {item} [/bold white on blue]")
            else:
                console.print(f"  [dim]  {item}[/dim]")
        console.print("\n[dim]↑ ↓ navigate  ·  Enter select  ·  Esc back[/dim]")
        key = readkey()
        if key == "up":
            idx = (idx - 1) % len(items)
        elif key == "down":
            idx = (idx + 1) % len(items)
        elif key == "enter":
            return idx
        elif key == "esc":
            return -1


# ── settings editor (shared by Settings menu + pre-launch review) ────────────
def _fmt_value(val) -> str:
    if isinstance(val, bool):
        return "[green]ON[/green]" if val else "[red]OFF[/red]"
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def _plain_value(val) -> str:
    """Style-free value for the settings list (colour comes from the row style)."""
    if isinstance(val, bool):
        return "ON" if val else "OFF"
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def _cat_color(key: str) -> str:
    for prefix, col in _CAT_COLOR.items():
        if key.startswith(prefix):
            return col
    return "white"


def _edit_value(cfg, key: str) -> str:
    """Cooked-mode prompt for a free (non-choice) field. Returns a status msg."""
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
        return f"[red]Invalid value for {cfg.label(key)}.[/red]"


def edit_settings(cfg, keys: list[str], title: str, *,
                  allow_launch: bool = False, save_on_exit: bool = True) -> bool:
    """Two-panel editor over `keys`. Returns True only if the user chose Launch."""
    idx = 0
    rows = list(keys) + (["__launch__"] if allow_launch else [])
    label_w = max((len(cfg.label(k)) for k in keys), default=10)
    message = ""
    while True:
        console.clear()
        hint = ("↑ ↓ navigate  ·  ← →  cycle  ·  Enter  "
                + ("launch / " if allow_launch else "") + "select / edit  ·  "
                + ("R reset  ·  " if not allow_launch else "") + "Esc "
                + ("cancel" if allow_launch else "save & back"))
        console.print(Panel(f"[bold cyan]{title}[/bold cyan]\n[dim]{hint}[/dim]", border_style="blue"))

        # build each row as a plain, fixed-width string so the highlight spans
        # the whole line and long values never wrap
        raw_rows = []
        for i, key in enumerate(rows):
            if key == "__launch__":
                raw_rows.append((i, "  ▶ Launch", "launch"))
                continue
            val = _plain_value(cfg.get(key))
            ch = cfg.choices(key)
            hint_n = f"  ({ch.index(cfg.get(key)) + 1}/{len(ch)})" if (ch and cfg.get(key) in ch) else ""
            lock = "  [locked]" if cfg.is_locked(key) else ""
            caret = "›" if i == idx else " "
            raw_rows.append((i, f" {caret} {cfg.label(key).ljust(label_w)}   {val}{hint_n}{lock}",
                             "locked" if cfg.is_locked(key) else "normal"))
        inner = min(max(len(s) for _, s, _ in raw_rows) + 1, 110)

        body = Text()
        for i, s, kind in raw_rows:
            line = s[:inner].ljust(inner)
            if i == idx:
                body.append(line, style="bold black on cyan" if kind == "launch" else "bold white on blue")
            elif kind == "launch":
                body.append(line, style="bold green")
            elif kind == "locked":
                body.append(line, style="dim")
            else:
                body.append(line)
            body.append("\n")

        cur = rows[idx]
        if cur == "__launch__":
            desc = "[bold]Launch[/bold]\nStart this run with the settings above."
            border = "green"
        else:
            wrapped = textwrap.fill(cfg.desc(cur), width=46) or "(no description)"
            ch = cfg.choices(cur)
            if isinstance(cfg.get(cur), bool):
                extra = "\n[dim]Enter / ← → toggles[/dim]"
            elif ch:
                extra = "\n[dim]← → cycles · Enter to pick from list[/dim]"
            else:
                extra = "\n[dim]Enter to edit value[/dim]"
            desc = (f"[bold]{cfg.label(cur)}[/bold]\n[dim]{'─' * 44}[/dim]\n{wrapped}\n\n"
                    f"[dim]Current:[/dim] {_fmt_value(cfg.get(cur))}{extra}")
            border = _cat_color(cur)

        left = Panel(body, title="[bold]Settings[/bold]", border_style="blue",
                     width=inner + 4, padding=(0, 0))
        right = Panel(desc, title="[bold]Description[/bold]", border_style=border,
                      width=50, padding=(0, 1))
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
                elif key == "enter":               # pick from the viable options
                    sel = arrow_menu(f"Select — {cfg.label(cur)}", ch, subtitle=cfg.desc(cur))
                    if sel >= 0:
                        cfg.set(cur, ch[sel])
                    message = ""
            elif key == "enter":
                message = _edit_value(cfg, cur)
            elif key == "r" and not allow_launch:
                cfg.reset(); message = "✓ Settings reset to defaults."


# ── top-level screens ────────────────────────────────────────────────────────
def settings_menu(cfg) -> None:
    edit_settings(cfg, cfg.field_names(), "Settings", allow_launch=False, save_on_exit=True)


def prelaunch_review(cfg, action: str) -> bool:
    keys = RELEVANT.get(action, cfg.field_names())
    keys = [k for k in keys if k in cfg.field_names()]
    return edit_settings(cfg, keys, f"Review settings — {action}",
                         allow_launch=True, save_on_exit=True)