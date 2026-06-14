"""Single-source configuration.

One settings object, edited by either the TUI or the CLI. Each field carries its
own label / description / choice-list metadata (the single source the TUI reads),
plus get/set/cycle/lock helpers so the editor logic stays testable. CLI flags
override the saved file for the run and are surfaced as locked (read-only) in the
TUI.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path

CONFIG_PATH = Path("config.json")


def _f(default, label, desc, choices=None):
    return field(default=default, metadata={"label": label, "desc": desc, "choices": choices})


@dataclass
class Config:
    detector: str = _f(
        "Models/xlm-multicw",
        "Check-worthiness classifier",
        "Fine-tuned check-worthiness classifier (mDeBERTa or XLM-R), under Models/.",
        choices=["Models/xlm-multicw", "Models/mdb-multicw"],
    )

    # ------------------------------------------------------------------ #
    def __post_init__(self):
        self._locked: set = set()

    # ---- persistence / CLI ----
    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Config":
        cfg = cls()
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for f in fields(cls):
                if f.name in data:
                    setattr(cfg, f.name, data[f.name])
        return cfg

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def apply_cli(self, args: argparse.Namespace) -> list[str]:
        overridden = []
        for f in fields(type(self)):
            val = getattr(args, f.name, None)
            if val is not None:
                self.set(f.name, val)
                overridden.append(f.name)
        self._locked = set(overridden)
        return overridden

    @staticmethod
    def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
        for f in fields(Config):
            argname = "--" + f.name.replace("_", "-")
            if f.metadata.get("choices"):
                parser.add_argument(argname, dest=f.name,
                                    choices=f.metadata["choices"], default=None)
            else:
                parser.add_argument(argname, dest=f.name, default=None)

    # ---- TUI introspection / mutation helpers ----
    def field_names(self) -> list[str]:
        return [f.name for f in fields(self)]

    def _meta(self, name: str) -> dict:
        return {f.name: f.metadata for f in fields(self)}[name]

    def label(self, name: str) -> str:
        return self._meta(name).get("label") or name

    def desc(self, name: str) -> str:
        return self._meta(name).get("desc", "")

    def choices(self, name: str):
        return self._meta(name).get("choices")

    def is_locked(self, name: str) -> bool:
        return name in getattr(self, "_locked", set())

    def get(self, name: str):
        return getattr(self, name)

    def set(self, name: str, raw) -> None:
        cur = getattr(self, name)
        if isinstance(cur, bool):
            val = raw if isinstance(raw, bool) else str(raw).strip().lower() in ("1", "true", "on", "yes")
        elif isinstance(cur, int) and not isinstance(cur, bool):
            val = int(raw)
        elif isinstance(cur, float):
            val = float(raw)
        else:
            val = str(raw)
        setattr(self, name, val)

    def cycle(self, name: str, direction: int) -> None:
        """Toggle a bool, or advance a choice list (wraps). No-op for free fields."""
        cur = getattr(self, name)
        if isinstance(cur, bool):
            setattr(self, name, not cur)
            return
        ch = self.choices(name)
        if ch:
            i = ch.index(cur) if cur in ch else 0
            setattr(self, name, ch[(i + direction) % len(ch)])

    def reset(self) -> None:
        for f in fields(type(self)):
            if f.name in getattr(self, "_locked", set()):
                continue
            setattr(self, f.name, f.default)
