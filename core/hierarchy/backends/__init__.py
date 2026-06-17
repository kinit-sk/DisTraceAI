"""Backend registry.

Backends are imported lazily via ``get_backend`` so that importing this package
never eagerly pulls heavy / optional dependencies (SpecFi requires the patched
NodeRAG). Live code paths import the concrete submodule they need directly.
"""

_BACKEND_MODULES = {
    "bm25_rag": ("core.hierarchy.backends.bm25_rag", "BM25RagBackend"),
    "dense":    ("core.hierarchy.backends.bm25_rag", "BM25RagBackend"),
    "context1": ("core.hierarchy.backends.context1", "Context1Backend"),
    "context-1": ("core.hierarchy.backends.context1", "Context1Backend"),
    "specfi":   ("core.hierarchy.backends.specfi_c", "SpecFiCBackend"),
    "specfi-cs": ("core.hierarchy.backends.specfi_c", "SpecFiCBackend"),
    "cspecfi":  ("core.hierarchy.backends.specfi_c", "SpecFiCBackend"),
}


def get_backend(name: str):
    """Return the backend CLASS registered under *name* (imported on demand)."""
    import importlib
    if name not in _BACKEND_MODULES:
        raise KeyError(f"Unknown backend {name!r}. Known: {sorted(_BACKEND_MODULES)}")
    module_path, cls_name = _BACKEND_MODULES[name]
    return getattr(importlib.import_module(module_path), cls_name)
