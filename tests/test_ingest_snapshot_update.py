"""
Tests for Fix 1: do_ingest() updates project_state.json after a clean --force run.

Rules:
  - force=True  + no errors  → snapshot IS updated (chunking hashes reflect current config)
  - force=True  + errors     → snapshot is NOT updated (ingest was partial)
  - force=False              → snapshot is NOT updated (skip-based run; not all files re-ingested)
"""
from __future__ import annotations

import numpy as np
import pytest

from raggen.core.config.project import GroupChunkingConfig, default_project_config
from raggen.core.embeddings.capabilities import ModelCapabilities
from raggen.core.embeddings.model_specs_cache import ModelSpecsCache
from raggen.core.ingest import do_ingest
from raggen.core.metadata.models import ProjectLifecycleState
from raggen.core.metadata.store import (
    compute_chunking_hash,
    create_project_state,
    load_project_state,
    save_project_state,
)

_DUMMY_DIM = 4
_DUMMY_MAX_SEQ = 512


# ---------------------------------------------------------------------------
# Shared helpers (duplicated from test_ingest_chunking_drift.py by design —
# each test module is self-contained)
# ---------------------------------------------------------------------------


def _seed_model_specs_cache(root, model_id):
    caps = ModelCapabilities(
        model_id=model_id,
        actual_dim=_DUMMY_DIM,
        max_seq_length=_DUMMY_MAX_SEQ,
        max_batch_size=None,
    )
    specs_dir = root / ".rag" / "metadata" / "model_specs"
    ModelSpecsCache(specs_dir).put(caps)


def _dummy_embedder_factory(*args, **kwargs):
    class _Dummy:
        model_id = "dummy"

        @property
        def dim(self):
            return _DUMMY_DIM

        @property
        def max_seq_length(self):
            return _DUMMY_MAX_SEQ

        def embed_texts(self, texts, batch_size=32, normalize=True):
            return np.zeros((len(texts), _DUMMY_DIM), dtype=np.float32)

        def get_length_function(self):
            return len

    return _Dummy()


def _bootstrap_project(tmp_path, monkeypatch):
    root = tmp_path
    monkeypatch.chdir(root)

    cfg = default_project_config(root)
    cfg.storage.database_url = (
        f"sqlite:///{(root / '.rag' / 'rag.db').resolve().as_posix()}"
    )

    cfg_dir = root / ".rag"
    cfg_dir.mkdir(exist_ok=True)
    cfg_file = cfg_dir / "config.toml"
    cfg_file.write_text(
        f"[storage]\n"
        f"backend_key = \"{cfg.storage.backend_key}\"\n"
        f"database_url = \"{cfg.storage.database_url}\"\n"
        f"vector_backend_import = \"{cfg.storage.vector_backend_import}\"\n"
        f"[embedding]\n"
        f"model_id = \"{cfg.embedding.model_id}\"\n"
        f"dim = {_DUMMY_DIM}\n"
        f"normalize = {str(cfg.embedding.normalize).lower()}\n"
    )

    from raggen.core.bootstrap import bootstrap
    cfg = bootstrap(cfg_file)

    from raggen.core.store.initializer import init_database
    init_database(cfg)

    _seed_model_specs_cache(root, cfg.embedding.model_id)

    state = create_project_state(cfg=cfg, state=ProjectLifecycleState.SET_UP)
    save_project_state(state)

    return cfg


# ---------------------------------------------------------------------------
# Snapshot-update tests
# ---------------------------------------------------------------------------


class TestIngestSnapshotUpdate:
    def test_force_clean_ingest_updates_snapshot(self, tmp_path, monkeypatch):
        """After a clean --force ingest, project_state.json reflects current config."""
        cfg = _bootstrap_project(tmp_path, monkeypatch)

        # Record the old hash for the fallback group
        old_hash = compute_chunking_hash(cfg.chunking["fallback"])

        # Drift the config so current hash ≠ stored hash
        cfg.chunking["fallback"].chunk_size += 500
        new_hash = compute_chunking_hash(cfg.chunking["fallback"])
        assert new_hash != old_hash

        monkeypatch.setattr(
            "raggen.core.ingest.ingest_service.LocalSentenceTransformerEmbedder",
            _dummy_embedder_factory,
        )

        result = do_ingest(force=True)
        assert result.success
        assert not result.errors

        # Load the saved state and check that the hash was updated
        updated_state = load_project_state(cfg.project_root)
        assert updated_state is not None
        stored = updated_state.foundation.chunking_hashes.get("fallback")
        assert stored == new_hash, (
            f"Expected stored hash {new_hash!r}, got {stored!r}"
        )

    def test_force_clean_ingest_clears_drift_on_subsequent_run(self, tmp_path, monkeypatch):
        """After --force, a normal ingest must not emit chunking_drift warnings."""
        cfg = _bootstrap_project(tmp_path, monkeypatch)
        cfg.chunking["fallback"].chunk_size += 500

        monkeypatch.setattr(
            "raggen.core.ingest.ingest_service.LocalSentenceTransformerEmbedder",
            _dummy_embedder_factory,
        )

        # First run: force (clears drift)
        result1 = do_ingest(force=True)
        assert result1.success

        # Second run: normal (no force)
        result2 = do_ingest()
        drift = [w for w in result2.warnings if w.code == "chunking_drift"]
        assert drift == [], "drift warning should be gone after force re-ingest"

    def test_force_with_errors_does_not_update_snapshot(self, tmp_path, monkeypatch):
        """If --force ingest has errors, the snapshot must NOT be updated."""
        cfg = _bootstrap_project(tmp_path, monkeypatch)

        # Drift the config
        cfg.chunking["fallback"].chunk_size += 500
        old_stored_hash = load_project_state(cfg.project_root).foundation.chunking_hashes["fallback"]

        # Patch embedder to raise on every call
        class _FailEmbedder:
            model_id = "dummy"

            @property
            def dim(self):
                return _DUMMY_DIM

            @property
            def max_seq_length(self):
                return _DUMMY_MAX_SEQ

            def embed_texts(self, texts, batch_size=32, normalize=True):
                raise RuntimeError("embedding exploded")

            def get_length_function(self):
                return len

        monkeypatch.setattr(
            "raggen.core.ingest.ingest_service.LocalSentenceTransformerEmbedder",
            lambda *a, **kw: _FailEmbedder(),
        )

        # Write a real file so there's something to try ingesting
        (tmp_path / "note.txt").write_text("hello world")

        result = do_ingest(force=True)
        # Ingest still succeeds (errors are non-fatal per-file), but errors list is non-empty
        assert result.errors

        updated_state = load_project_state(cfg.project_root)
        stored = updated_state.foundation.chunking_hashes.get("fallback")
        assert stored == old_stored_hash, (
            "Snapshot must NOT be updated when ingest has errors"
        )

    def test_normal_ingest_without_force_does_not_update_snapshot(self, tmp_path, monkeypatch):
        """A regular ingest (no --force) must never update the snapshot."""
        cfg = _bootstrap_project(tmp_path, monkeypatch)

        # Drift the config
        cfg.chunking["fallback"].chunk_size += 500
        old_stored_hash = load_project_state(cfg.project_root).foundation.chunking_hashes["fallback"]

        monkeypatch.setattr(
            "raggen.core.ingest.ingest_service.LocalSentenceTransformerEmbedder",
            _dummy_embedder_factory,
        )

        result = do_ingest(force=False)
        assert result.success

        updated_state = load_project_state(cfg.project_root)
        stored = updated_state.foundation.chunking_hashes.get("fallback")
        assert stored == old_stored_hash, (
            "Snapshot must NOT be updated on a normal (non-force) ingest"
        )

    def test_force_ingest_updates_all_groups(self, tmp_path, monkeypatch):
        """After --force, all groups' hashes in the snapshot reflect current config."""
        cfg = _bootstrap_project(tmp_path, monkeypatch)

        # Add a second group and save it in state
        cfg.chunking["docs"] = GroupChunkingConfig(
            strategy="fixed", unit="chars", chunk_size=800, overlap=80
        )
        state = create_project_state(cfg=cfg, state=ProjectLifecycleState.SET_UP)
        save_project_state(state)

        # Drift both groups
        cfg.chunking["fallback"].chunk_size += 500
        cfg.chunking["docs"].overlap += 200

        expected_fallback = compute_chunking_hash(cfg.chunking["fallback"])
        expected_docs = compute_chunking_hash(cfg.chunking["docs"])

        monkeypatch.setattr(
            "raggen.core.ingest.ingest_service.LocalSentenceTransformerEmbedder",
            _dummy_embedder_factory,
        )

        result = do_ingest(force=True)
        assert result.success
        assert not result.errors

        updated_state = load_project_state(cfg.project_root)
        assert updated_state.foundation.chunking_hashes["fallback"] == expected_fallback
        assert updated_state.foundation.chunking_hashes["docs"] == expected_docs
