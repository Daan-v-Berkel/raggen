"""
Tests for Step 2: ModelSpecsCache + MissingModelSpecsError.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from raggen.core.embeddings.capabilities import ModelCapabilities
from raggen.core.embeddings.model_specs_cache import (
    MissingModelSpecsError,
    ModelSpecsCache,
    _model_id_to_filename,
)


# ---------------------------------------------------------------------------
# File-name sanitisation
# ---------------------------------------------------------------------------


class TestModelIdToFilename:
    def test_slash_becomes_double_underscore(self):
        assert _model_id_to_filename("sentence-transformers/all-MiniLM-L6-v2") == (
            "sentence-transformers__all-MiniLM-L6-v2"
        )

    def test_baai_model(self):
        assert _model_id_to_filename("BAAI/bge-small-en-v1.5") == "BAAI__bge-small-en-v1.5"

    def test_no_slashes_unchanged(self):
        assert _model_id_to_filename("simple-model") == "simple-model"

    def test_special_chars_replaced_with_underscore(self):
        result = _model_id_to_filename("org/model@v2")
        assert "/" not in result
        assert "@" not in result

    def test_no_json_extension(self):
        result = _model_id_to_filename("org/model")
        assert not result.endswith(".json")


# ---------------------------------------------------------------------------
# ModelSpecsCache
# ---------------------------------------------------------------------------


_CAPS = ModelCapabilities(
    model_id="sentence-transformers/all-MiniLM-L6-v2",
    actual_dim=384,
    max_seq_length=256,
    max_batch_size=None,
)


class TestModelSpecsCache:
    def test_get_returns_none_on_miss(self, tmp_path):
        cache = ModelSpecsCache(tmp_path / "specs")
        assert cache.get("some/unknown-model") is None

    def test_exists_returns_false_on_miss(self, tmp_path):
        cache = ModelSpecsCache(tmp_path / "specs")
        assert cache.exists("some/unknown-model") is False

    def test_put_creates_directory(self, tmp_path):
        specs_dir = tmp_path / "nested" / "specs"
        cache = ModelSpecsCache(specs_dir)
        cache.put(_CAPS)
        assert specs_dir.exists()

    def test_put_writes_file_without_json_extension(self, tmp_path):
        cache = ModelSpecsCache(tmp_path)
        cache.put(_CAPS)
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert not files[0].name.endswith(".json")

    def test_roundtrip_get_after_put(self, tmp_path):
        cache = ModelSpecsCache(tmp_path)
        cache.put(_CAPS)
        result = cache.get(_CAPS.model_id)

        assert result is not None
        assert result.model_id == _CAPS.model_id
        assert result.actual_dim == _CAPS.actual_dim
        assert result.max_seq_length == _CAPS.max_seq_length
        assert result.max_batch_size is None

    def test_exists_returns_true_after_put(self, tmp_path):
        cache = ModelSpecsCache(tmp_path)
        cache.put(_CAPS)
        assert cache.exists(_CAPS.model_id) is True

    def test_put_writes_valid_json(self, tmp_path):
        cache = ModelSpecsCache(tmp_path)
        cache.put(_CAPS)
        # Find the written file and parse it as JSON
        files = list(tmp_path.iterdir())
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        assert payload["model_id"] == _CAPS.model_id
        assert payload["actual_dim"] == 384
        assert payload["max_seq_length"] == 256
        assert payload["max_batch_size"] is None
        assert "cached_at" in payload

    def test_put_overwrites_existing_entry(self, tmp_path):
        cache = ModelSpecsCache(tmp_path)
        caps_v1 = ModelCapabilities(
            model_id="org/model", actual_dim=384, max_seq_length=256, max_batch_size=None
        )
        caps_v2 = ModelCapabilities(
            model_id="org/model", actual_dim=768, max_seq_length=512, max_batch_size=None
        )
        cache.put(caps_v1)
        cache.put(caps_v2)
        result = cache.get("org/model")
        assert result is not None
        assert result.actual_dim == 768

    def test_multiple_models_get_separate_files(self, tmp_path):
        cache = ModelSpecsCache(tmp_path)
        caps_a = ModelCapabilities(
            model_id="org/model-a", actual_dim=384, max_seq_length=256, max_batch_size=None
        )
        caps_b = ModelCapabilities(
            model_id="org/model-b", actual_dim=768, max_seq_length=512, max_batch_size=None
        )
        cache.put(caps_a)
        cache.put(caps_b)
        assert len(list(tmp_path.iterdir())) == 2

        got_a = cache.get("org/model-a")
        got_b = cache.get("org/model-b")
        assert got_a is not None and got_a.actual_dim == 384
        assert got_b is not None and got_b.actual_dim == 768


# ---------------------------------------------------------------------------
# MissingModelSpecsError
# ---------------------------------------------------------------------------


class TestMissingModelSpecsError:
    def test_message_contains_model_id(self):
        err = MissingModelSpecsError("sentence-transformers/all-MiniLM-L6-v2")
        assert "sentence-transformers/all-MiniLM-L6-v2" in str(err)

    def test_message_mentions_build(self):
        err = MissingModelSpecsError("any/model")
        assert "build" in str(err).lower()

    def test_model_id_attribute(self):
        err = MissingModelSpecsError("my/model")
        assert err.model_id == "my/model"

    def test_is_runtime_error_subclass(self):
        assert isinstance(MissingModelSpecsError("m"), RuntimeError)
