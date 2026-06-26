"""Backend registry.

Backends are imported lazily via ``get_backend`` so that importing this package
never eagerly pulls heavy / optional dependencies (SpecFi requires the patched
NodeRAG). Live code paths import the concrete submodule they need directly.
"""

_BACKEND_MODULES = {
    "bm25_rag": ("core.methods.bm25_rag", "BM25RagBackend"),
    "dense":    ("core.methods.bm25_rag", "BM25RagBackend"),
    "context1": ("core.methods.context1", "Context1Backend"),
    "context-1": ("core.methods.context1", "Context1Backend"),
    "specfi":   ("core.methods.specfi_c", "SpecFiCBackend"),
    "specfi-cs": ("core.methods.specfi_c", "SpecFiCBackend"),
    "cspecfi":  ("core.methods.specfi_c", "SpecFiCBackend"),
}


def get_backend(name: str):
    """Return the llm_backends CLASS registered under *name* (imported on demand)."""
    import importlib
    if name not in _BACKEND_MODULES:
        raise KeyError(f"Unknown llm_backends {name!r}. Known: {sorted(_BACKEND_MODULES)}")
    module_path, cls_name = _BACKEND_MODULES[name]
    return getattr(importlib.import_module(module_path), cls_name)
