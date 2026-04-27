from __future__ import annotations

import importlib
import inspect
from typing import Any

from .exceptions import BackendLoadError
from .vector_backends.base import VectorBackend

BUILTIN_VECTOR_BACKENDS: dict[str, str] = {
    "sqlite_vec": "raggen.core.store.vector_backends.sqlite_vec:SQLiteVecBackend",
    "pgvector": "raggen.core.store.vector_backends.pgvector:PgVectorBackend",
}


def resolve_vector_backend_import(backend_key: str, override: str = "") -> str:
    """Return the import path for a vector backend.

    If *override* is non-empty it is used as-is (custom plugin path).
    Otherwise *backend_key* is looked up in BUILTIN_VECTOR_BACKENDS.
    Raises BackendLoadError when neither resolves to a known path.
    """
    if override:
        return override
    path = BUILTIN_VECTOR_BACKENDS.get(backend_key)
    if path is None:
        known = ", ".join(sorted(BUILTIN_VECTOR_BACKENDS))
        raise BackendLoadError(
            f"Unknown backend_key '{backend_key}'. "
            f"Built-in backends: {known}. "
            f"Set vector_backend_import in [storage] to use a custom backend."
        )
    return path


def load_object(import_path: str) -> Any:
    """Load an object given an import path in form 'module.sub:AttrName'."""
    if not isinstance(import_path, str) or ":" not in import_path:
        raise BackendLoadError(
            f"Import path must be a string 'module:Attr', got: {import_path!r}"
        )
    module_path, attr = import_path.split(":", 1)
    if not module_path or not attr:
        raise BackendLoadError(
            f"Invalid import path '{import_path}'. Expected format 'module:ClassName'."
        )
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        raise BackendLoadError(
            f"Failed to import module '{module_path}': {exc}"
        ) from exc
    try:
        obj = getattr(module, attr)
    except AttributeError as exc:
        raise BackendLoadError(
            f"Module '{module_path}' has no attribute '{attr}'"
        ) from exc
    return obj


def load_vector_backend(import_path: str) -> VectorBackend:
    obj = load_object(import_path)
    # if it's a class, instantiate
    if inspect.isclass(obj):
        try:
            inst = obj()
        except TypeError as exc:
            raise BackendLoadError(
                f"Failed to instantiate backend class '{import_path}': {exc}"
            ) from exc
    else:
        # allow passing an already-instantiated object
        inst = obj
    # duck-type check: ensure required methods exist
    required = ("supports", "create_schema", "drop_schema",
                "upsert_vectors", "delete_vectors", "search")
    for name in required:
        if not hasattr(inst, name) or not callable(getattr(inst, name)):
            raise BackendLoadError(
                f"Loaded object from '{import_path}' does not implement required method '{name}'"
            )
    return inst
