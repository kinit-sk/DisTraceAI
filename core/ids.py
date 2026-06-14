"""Simple ID generation for KB records.

IDs are derived from the original source name (filename stem, row ID, etc.)
and made unique with a counter only when a collision occurs.
"""
from __future__ import annotations

from pathlib import Path


def make_id(name: str) -> str:
    """Sanitise *name* into a filesystem-safe ID string.

    Strips the file extension if present, replaces path separators and
    whitespace with underscores, and strips leading/trailing underscores.
    """
    stem = Path(name).stem if "." in Path(name).name else name
    safe = stem.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return safe.strip("_") or "article"


class IdRegistry:
    """Tracks already-assigned IDs and appends a counter on collision."""

    def __init__(self) -> None:
        self._seen: dict[str, int] = {}

    def assign(self, name: str) -> str:
        base = make_id(name)
        if base not in self._seen:
            self._seen[base] = 0
            return base
        self._seen[base] += 1
        return f"{base}_{self._seen[base]}"

    def reset(self) -> None:
        self._seen.clear()


# Module-level registry used by the PolyNarrative converter so that a single
# convert() call gets consistent IDs without callers needing to manage state.
_default_registry = IdRegistry()


def article_id(name: str, *, registry: IdRegistry | None = None) -> str:
    """Return a unique, human-readable ID for an article named *name*."""
    return (registry or _default_registry).assign(name)


def reset_default_registry() -> None:
    """Clear the module-level registry (useful between test runs)."""
    _default_registry.reset()
