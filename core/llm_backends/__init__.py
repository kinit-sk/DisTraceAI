"""LLM backend dispatcher for DisTraceAI.

The active backend is determined by Config.llm_backend ("vllm" or "llama-cpp").
All pipeline code imports from core.models, which proxies here at call-time —
so switching the backend in the TUI takes effect on the next model load without
restarting the process.

Do not import heavy backend modules at package level; imports happen lazily
inside get_backend() so that the TUI starts instantly regardless of which
packages are installed.
"""
from __future__ import annotations

_BACKEND_NAMES = ("vllm", "llama-cpp")


def get_backend(name: str):
    """Return the backend module for *name* ('vllm' or 'llama-cpp')."""
    if name == "vllm":
        from core.llm_backends import vllm as mod
        return mod
    if name in ("llama-cpp", "llama_cpp"):
        from core.llm_backends import llama_cpp as mod
        return mod
    raise ValueError(
        f"Unknown LLM backend {name!r}. Valid options: {_BACKEND_NAMES}")
