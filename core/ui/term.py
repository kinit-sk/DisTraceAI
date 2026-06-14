"""Tiny ANSI colour helper for evaluation reports.

Colour is emitted only when stdout is a TTY (so piped output and SLURM logs stay
clean), unless forced. Honours NO_COLOR (https://no-color.org) and
FORCE_COLOR. Because it no-ops off-TTY, report text is identical under
pytest, keeping substring assertions stable.
"""
from __future__ import annotations

import os
import sys

_CODES = {"red": "31", "green": "32", "yellow": "33", "blue": "34",
          "magenta": "35", "cyan": "36", "bold": "1", "dim": "2"}


def enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def color(text, *styles: str) -> str:
    if not styles or not enabled():
        return str(text)
    codes = ";".join(_CODES[s] for s in styles if s in _CODES)
    return f"\033[{codes}m{text}\033[0m" if codes else str(text)


def red(t):    return color(t, "red")
def green(t):  return color(t, "green")
def yellow(t): return color(t, "yellow")
def cyan(t):   return color(t, "cyan")
def bold(t):   return color(t, "bold")
def dim(t):    return color(t, "dim")


def ok(passed: bool) -> str:
    """Green ✓ / red ✗."""
    return green("✓") if passed else red("✗")


def metric(value: float, good: float = 0.7, mid: float = 0.4, width: int = 0) -> str:
    """Colour a 0–1 score: green (≥good) / yellow (≥mid) / red. `width` right-pads."""
    s = f"{value:.3f}"
    if width:
        s = f"{s:>{width}}"
    style = "green" if value >= good else ("yellow" if value >= mid else "red")
    return color(s, style)


def rule(text: str = "", width: int = 70) -> str:
    """A dim horizontal rule, optionally titled."""
    return dim("═" * width if not text else f"══ {text} " + "═" * max(0, width - len(text) - 4))
