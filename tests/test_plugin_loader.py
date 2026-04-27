import sys
import types

import pytest

from raggen.core.store.plugin_loader import (
    BUILTIN_VECTOR_BACKENDS,
    load_vector_backend,
    resolve_vector_backend_import,
)
from raggen.core.store.exceptions import BackendLoadError


# ---------------------------------------------------------------------------
# resolve_vector_backend_import
# ---------------------------------------------------------------------------

def test_resolve_known_key_sqlite_vec():
    path = resolve_vector_backend_import("sqlite_vec")
    assert path == BUILTIN_VECTOR_BACKENDS["sqlite_vec"]


def test_resolve_known_key_pgvector():
    path = resolve_vector_backend_import("pgvector")
    assert path == BUILTIN_VECTOR_BACKENDS["pgvector"]


def test_resolve_override_takes_precedence_over_known_key():
    custom = "my_pkg.backends:MyBackend"
    path = resolve_vector_backend_import("sqlite_vec", override=custom)
    assert path == custom


def test_resolve_override_used_for_unknown_key():
    custom = "my_pkg.backends:MyBackend"
    path = resolve_vector_backend_import("totally_unknown", override=custom)
    assert path == custom


def test_resolve_unknown_key_no_override_raises():
    with pytest.raises(BackendLoadError) as exc_info:
        resolve_vector_backend_import("no_such_backend")
    msg = str(exc_info.value)
    assert "no_such_backend" in msg
    # error message should name the known backends
    assert "sqlite_vec" in msg
    assert "pgvector" in msg


def test_resolve_empty_override_falls_back_to_registry():
    # explicit empty string must behave the same as no override
    path = resolve_vector_backend_import("sqlite_vec", override="")
    assert path == BUILTIN_VECTOR_BACKENDS["sqlite_vec"]


# ---------------------------------------------------------------------------
# load_vector_backend — error paths
# ---------------------------------------------------------------------------

def test_load_vector_backend_invalid_format_no_colon():
    with pytest.raises(BackendLoadError):
        load_vector_backend("no_colon_here")


def test_load_vector_backend_nonexistent_module():
    with pytest.raises(BackendLoadError) as exc_info:
        load_vector_backend("nonexistent.module.xyz:SomeClass")
    assert "nonexistent.module.xyz" in str(exc_info.value)


def test_load_vector_backend_missing_required_method():
    mod = types.ModuleType("tests._incomplete_backend_mod")

    class IncompleteBackend:
        key = "incomplete"
        def supports(self, engine): return True
        def create_schema(self, engine, dim): pass
        # missing: drop_schema, upsert_vectors, delete_vectors, search

    mod.IncompleteBackend = IncompleteBackend
    sys.modules["tests._incomplete_backend_mod"] = mod

    with pytest.raises(BackendLoadError) as exc_info:
        load_vector_backend("tests._incomplete_backend_mod:IncompleteBackend")
    assert "drop_schema" in str(exc_info.value)


def test_load_vector_backend_accepts_instance_not_just_class():
    """load_vector_backend should accept a pre-instantiated object."""
    mod = types.ModuleType("tests._instance_backend_mod")

    class FullBackend:
        key = "full"
        def supports(self, engine): return True
        def create_schema(self, engine, dim): pass
        def drop_schema(self, engine): pass
        def upsert_vectors(self, conn, **kwargs): pass
        def delete_vectors(self, conn, **kwargs): pass
        def search(self, engine, **kwargs): return []

    mod.instance = FullBackend()
    sys.modules["tests._instance_backend_mod"] = mod

    result = load_vector_backend("tests._instance_backend_mod:instance")
    assert isinstance(result, FullBackend)
