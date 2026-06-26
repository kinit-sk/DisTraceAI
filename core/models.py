"""DisTraceAI model façade.

This module is the single import point for all model-related functions across
the pipeline.  It dispatches every call to the active LLM backend, which is
chosen by Config.llm_backend ("vllm" or "llama-cpp") and can be changed at
runtime via the TUI Settings without restarting the process.

Usage (unchanged from the old single-backend models.py):
    from core.models import make_embedder, make_generator, close_generator, ...

Adding a new backend: implement a module in core/llm_backends/ that exposes the
functions listed in _REQUIRED_SYMBOLS, then register its name in
core/llm_backends/__init__.py.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------

_DEFAULT_BACKEND = "vllm"

# Symbols every backend module must provide (duck-typed, not enforced at import).
_REQUIRED_SYMBOLS = (
    "make_embedder",
    "encode_with_backoff",
    "make_generator",
    "close_generator",
    "generator_is_concurrent",
    "parallel_map",
    "VLLMGenerator",       # canonical generator class (may be an alias)
)


def _active_backend():
    """Return the active backend module, resolved lazily each call.

    Reads Config.llm_backend if a Config is importable, otherwise falls back
    to the DISTRACE_LLM_BACKEND environment variable, then to "vllm".
    The late-binding means a TUI backend switch is effective on the next
    model-load call with no process restart required.
    """
    name = os.environ.get("DISTRACE_LLM_BACKEND", _DEFAULT_BACKEND)
    # Try the live Config first (available after config.py is imported by main.py)
    try:
        from config import Config
        cfg = Config.load()
        name = getattr(cfg, "llm_backend", name)
    except Exception:
        pass
    from core.llm_backends import get_backend
    return get_backend(name)


# ---------------------------------------------------------------------------
# Public API — thin wrappers that proxy to the active backend
# ---------------------------------------------------------------------------

def make_embedder(model_name: str, **kw):
    """Load the embedder for the active backend (SentenceTransformer or vLLM)."""
    return _active_backend().make_embedder(model_name, **kw)


def encode_with_backoff(embedder, texts: Sequence[str], **kw):
    """Encode texts to embeddings using the active backend's strategy."""
    return _active_backend().encode_with_backoff(embedder, texts, **kw)


def make_generator(model_key: str, quant: str = "", **kw):
    """Build a generator using the active backend.

    quant is only meaningful for llama-cpp (GGUF quantization level, e.g. "Q4_K_M").
    It is silently ignored by the vLLM backend.
    """
    return _active_backend().make_generator(model_key, quant, **kw)


def close_generator(generate) -> None:
    """Safely close any generator and free GPU VRAM."""
    if generate is None:
        return
    # close() is defined on the generator object itself; no backend dispatch needed.
    close = getattr(generate, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:
            logger.debug("[models] close_generator raised (ignored): %s", exc)


def generator_is_concurrent(generate) -> bool:
    """True if the generator supports concurrent calls (server-backed)."""
    return _active_backend().generator_is_concurrent(generate)


def parallel_map(fn, items, max_workers: int = 1, description: str | None = None):
    """Apply fn over items up to max_workers at a time, in order."""
    return _active_backend().parallel_map(fn, items, max_workers=max_workers,
                                          description=description)


# ---------------------------------------------------------------------------
# Re-export resolve_generator for the few call sites that use it directly
# (gen_narratives, gen_campaigns).  Each backend exposes its own version.
# ---------------------------------------------------------------------------

def resolve_generator(model_key: str, quant: str = ""):
    """Resolve (model_key, quant) → backend-specific handle.

    For vLLM: returns the HuggingFace repo id (quant ignored).
    For llama-cpp: returns (repo_id, gguf_filename).
    """
    bk = _active_backend()
    fn = getattr(bk, "resolve_generator", None)
    if fn is None:
        raise NotImplementedError(
            "Active backend does not expose resolve_generator()")
    try:
        return fn(model_key, quant)
    except TypeError:
        return fn(model_key)  # vLLM backend takes only model_key


# ---------------------------------------------------------------------------
# Convenience: VLLMGenerator class import  (duck-typed alias on each backend)
# ---------------------------------------------------------------------------
# Pipeline code that type-hints VLLMGenerator can still import it from here.
# Each backend exposes VLLMGenerator (or an alias) so this always resolves.

def __getattr__(name: str):
    if name == "VLLMGenerator":
        return getattr(_active_backend(), "VLLMGenerator")
    if name == "LlamaGenerator":
        return getattr(_active_backend(), "LlamaGenerator",
                       getattr(_active_backend(), "VLLMGenerator"))
    raise AttributeError(f"module 'core.models' has no attribute {name!r}")
