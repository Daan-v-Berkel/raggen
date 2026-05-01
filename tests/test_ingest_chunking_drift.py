"""
Tests for Step 6: chunking drift detection in do_ingest().

Strategy:
  1. Bootstrap a minimal project and save a project_state.json that contains
     chunking hashes computed from the current config.
  2. Modify the in-memory config (or a second group's config) to simulate drift.
  3. Call do_ingest() and check that the expected warnings appear.

No real model is loaded — the embedder is monkeypatched to a dummy.
"""
from __future__ import annotations

import numpy as np

from raggen.core.config.project import GroupChunkingConfig, default_project_config
from raggen.core.embeddings.capabilities import ModelCapabilities
from raggen.core.embeddings.model_specs_cache import ModelSpecsCache
from raggen.core.ingest import do_ingest
from raggen.core.metadata.models import ProjectLifecycleState
from raggen.core.metadata.store import (
    compute_chunking_hash,
    create_project_state,
    save_project_state,
)

_DUMMY_DIM = 4
_DUMMY_MAX_SEQ = 512


# ---------------------------------------------------------------------------
# Shared helpers
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
    """
    Create a minimal project and return the bootstrapped config.

    ``monkeypatch.chdir(tmp_path)`` is called here so every relative-path
    resolution (cfg.project_root = Path(".")) anchors to tmp_path for both
    the save_project_state call and the later do_ingest() call.

    After this call:
    - CWD == tmp_path
    - .rag/config.toml is written
    - The DB is initialised
    - Model specs cache is seeded
    - A SET_UP project_state.json is saved (with chunking hashes)
    """
    root = tmp_path
    # chdir first so Path(".").resolve() == tmp_path everywhere below.
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

    # Save a SET_UP project state — this writes chunking_hashes derived from
    # the current cfg.chunking (the default "fallback" group).
    # CWD is tmp_path so Path(".").resolve() == tmp_path and the JSON lands in
    # tmp_path/.rag/metadata/project_state.json as expected.
    state = create_project_state(cfg=cfg, state=ProjectLifecycleState.SET_UP)
    save_project_state(state)

    return cfg


# ---------------------------------------------------------------------------
# compute_chunking_hash unit tests
# ---------------------------------------------------------------------------


class TestComputeChunkingHash:
    def test_same_config_produces_same_hash(self):
        conf = GroupChunkingConfig(strategy="fixed", unit="chars", chunk_size=1000, overlap=100)
        assert compute_chunking_hash(conf) == compute_chunking_hash(conf)

    def test_different_chunk_size_produces_different_hash(self):
        a = GroupChunkingConfig(strategy="fixed", unit="chars", chunk_size=500, overlap=100)
        b = GroupChunkingConfig(strategy="fixed", unit="chars", chunk_size=1000, overlap=100)
        assert compute_chunking_hash(a) != compute_chunking_hash(b)

    def test_different_strategy_produces_different_hash(self):
        a = GroupChunkingConfig(strategy="fixed", unit="chars", chunk_size=1000, overlap=100)
        b = GroupChunkingConfig(strategy="paragraphMerge", unit="chars", chunk_size=1000, overlap=100)
        assert compute_chunking_hash(a) != compute_chunking_hash(b)

    def test_different_overlap_produces_different_hash(self):
        a = GroupChunkingConfig(strategy="fixed", unit="chars", chunk_size=1000, overlap=0)
        b = GroupChunkingConfig(strategy="fixed", unit="chars", chunk_size=1000, overlap=200)
        assert compute_chunking_hash(a) != compute_chunking_hash(b)

    def test_different_unit_produces_different_hash(self):
        a = GroupChunkingConfig(strategy="fixed", unit="chars", chunk_size=1000, overlap=100)
        b = GroupChunkingConfig(strategy="fixed", unit="tokens", chunk_size=1000, overlap=100)
        assert compute_chunking_hash(a) != compute_chunking_hash(b)

    def test_returns_64_char_hex_string(self):
        conf = GroupChunkingConfig()
        h = compute_chunking_hash(conf)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_is_deterministic_across_calls(self):
        conf = GroupChunkingConfig(strategy="fixed", unit="chars", chunk_size=800, overlap=50)
        assert compute_chunking_hash(conf) == compute_chunking_hash(conf)


# ---------------------------------------------------------------------------
# snapshot_foundational_config includes chunking_hashes
# ---------------------------------------------------------------------------


class TestSnapshotIncludesChunkingHashes:
    def test_hashes_present_for_all_groups(self, tmp_path):
        cfg = default_project_config(tmp_path)
        cfg.chunking["extra"] = GroupChunkingConfig(
            strategy="fixed", unit="chars", chunk_size=500, overlap=50
        )
        from raggen.core.metadata.store import snapshot_foundational_config
        snap = snapshot_foundational_config(cfg)
        assert "fallback" in snap.chunking_hashes
        assert "extra" in snap.chunking_hashes

    def test_hash_matches_compute_chunking_hash(self, tmp_path):
        cfg = default_project_config(tmp_path)
        from raggen.core.metadata.store import snapshot_foundational_config
        snap = snapshot_foundational_config(cfg)
        expected = compute_chunking_hash(cfg.chunking["fallback"])
        assert snap.chunking_hashes["fallback"] == expected

    def test_empty_chunking_gives_empty_hashes(self, tmp_path):
        cfg = default_project_config(tmp_path)
        cfg.chunking.clear()
        from raggen.core.metadata.store import snapshot_foundational_config
        snap = snapshot_foundational_config(cfg)
        assert snap.chunking_hashes == {}


# ---------------------------------------------------------------------------
# do_ingest() drift detection integration tests
# ---------------------------------------------------------------------------


class TestIngestChunkingDrift:
    def test_no_drift_emits_no_chunking_warning(self, tmp_path, monkeypatch):
        """Unchanged chunking config must produce no chunking_drift warnings."""
        cfg = _bootstrap_project(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "raggen.core.ingest.ingest_service.LocalSentenceTransformerEmbedder",
            lambda *a, **kw: _dummy_embedder_factory(),
        )

        result = do_ingest()
        drift = [w for w in result.warnings if w.code == "chunking_drift"]
        assert drift == []

    def test_changed_chunk_size_emits_warning(self, tmp_path, monkeypatch):
        """Changing chunk_size after saving state must produce a chunking_drift warning."""
        cfg = _bootstrap_project(tmp_path, monkeypatch)

        # Simulate drift: change chunk_size in the live config (state still has old hash)
        cfg.chunking["fallback"].chunk_size += 500

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "raggen.core.ingest.ingest_service.LocalSentenceTransformerEmbedder",
            lambda *a, **kw: _dummy_embedder_factory(),
        )

        result = do_ingest()
        drift = [w for w in result.warnings if w.code == "chunking_drift"]
        assert len(drift) == 1

    def test_warning_names_the_group(self, tmp_path, monkeypatch):
        """The drift warning must mention the group name."""
        cfg = _bootstrap_project(tmp_path, monkeypatch)
        cfg.chunking["fallback"].chunk_size += 500

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "raggen.core.ingest.ingest_service.LocalSentenceTransformerEmbedder",
            lambda *a, **kw: _dummy_embedder_factory(),
        )

        result = do_ingest()
        w = next(w for w in result.warnings if w.code == "chunking_drift")
        assert "fallback" in w.message

    def test_warning_contains_force_ingest_command(self, tmp_path, monkeypatch):
        """The drift warning must include 'rag ingest --force'."""
        cfg = _bootstrap_project(tmp_path, monkeypatch)
        cfg.chunking["fallback"].overlap += 100

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "raggen.core.ingest.ingest_service.LocalSentenceTransformerEmbedder",
            lambda *a, **kw: _dummy_embedder_factory(),
        )

        result = do_ingest()
        w = next(w for w in result.warnings if w.code == "chunking_drift")
        assert "rag ingest --force" in w.message

    def test_warning_mentions_current_settings(self, tmp_path, monkeypatch):
        """The drift warning must mention re-indexing with current settings."""
        cfg = _bootstrap_project(tmp_path, monkeypatch)
        cfg.chunking["fallback"].strategy = "paragraphMerge"

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "raggen.core.ingest.ingest_service.LocalSentenceTransformerEmbedder",
            lambda *a, **kw: _dummy_embedder_factory(),
        )

        result = do_ingest()
        w = next(w for w in result.warnings if w.code == "chunking_drift")
        assert "current settings" in w.message

    def test_drift_does_not_abort_ingest(self, tmp_path, monkeypatch):
        """A chunking drift warning must not prevent ingest from succeeding."""
        root = tmp_path
        (root / "note.txt").write_text("hello world")

        cfg = _bootstrap_project(tmp_path, monkeypatch)
        cfg.chunking["fallback"].chunk_size += 500

        monkeypatch.chdir(root)
        monkeypatch.setattr(
            "raggen.core.ingest.ingest_service.LocalSentenceTransformerEmbedder",
            lambda *a, **kw: _dummy_embedder_factory(),
        )

        result = do_ingest()
        assert result.success
        drift = [w for w in result.warnings if w.code == "chunking_drift"]
        assert len(drift) == 1

    def test_two_groups_with_drift_emit_two_warnings(self, tmp_path, monkeypatch):
        """Each group with drift gets its own warning."""
        cfg = _bootstrap_project(tmp_path, monkeypatch)

        # Add a second group to the live config AND to the saved state
        cfg.chunking["docs"] = GroupChunkingConfig(
            strategy="fixed", unit="chars", chunk_size=800, overlap=80
        )
        # Re-save state so both groups have stored hashes
        state = create_project_state(cfg=cfg, state=ProjectLifecycleState.SET_UP)
        save_project_state(state)

        # Now drift both groups
        cfg.chunking["fallback"].chunk_size += 500
        cfg.chunking["docs"].overlap += 200

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "raggen.core.ingest.ingest_service.LocalSentenceTransformerEmbedder",
            lambda *a, **kw: _dummy_embedder_factory(),
        )

        result = do_ingest()
        drift = [w for w in result.warnings if w.code == "chunking_drift"]
        assert len(drift) == 2
        group_names = {w.message.split("'")[1] for w in drift}
        assert group_names == {"fallback", "docs"}

    def test_only_changed_group_warns(self, tmp_path, monkeypatch):
        """Only the group whose config changed must emit a warning."""
        cfg = _bootstrap_project(tmp_path, monkeypatch)

        # Add a second group with its own stored hash
        cfg.chunking["docs"] = GroupChunkingConfig(
            strategy="fixed", unit="chars", chunk_size=800, overlap=80
        )
        state = create_project_state(cfg=cfg, state=ProjectLifecycleState.SET_UP)
        save_project_state(state)

        # Only drift the "docs" group; leave "fallback" unchanged
        cfg.chunking["docs"].chunk_size += 500

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "raggen.core.ingest.ingest_service.LocalSentenceTransformerEmbedder",
            lambda *a, **kw: _dummy_embedder_factory(),
        )

        result = do_ingest()
        drift = [w for w in result.warnings if w.code == "chunking_drift"]
        assert len(drift) == 1
        assert "docs" in drift[0].message

    def test_no_project_state_emits_no_drift_warning(self, tmp_path, monkeypatch):
        """
        If there is no project_state.json (e.g. built before state tracking),
        chunking drift detection must be skipped silently.
        """
        cfg = _bootstrap_project(tmp_path, monkeypatch)

        # Remove the project_state.json — use the same path helper so we
        # delete the exact file that was written (resolves via cfg.project_root).
        from raggen.core.metadata.store import project_state_path
        project_state_path(cfg.project_root).unlink()

        cfg.chunking["fallback"].chunk_size += 999

        monkeypatch.setattr(
            "raggen.core.ingest.ingest_service.LocalSentenceTransformerEmbedder",
            lambda *a, **kw: _dummy_embedder_factory(),
        )

        result = do_ingest()
        drift = [w for w in result.warnings if w.code == "chunking_drift"]
        assert drift == []

    def test_new_group_without_stored_hash_emits_no_warning(self, tmp_path, monkeypatch):
        """
        A group that didn't exist at build time (no stored hash) must not warn.
        """
        cfg = _bootstrap_project(tmp_path, monkeypatch)

        # Add a brand-new group that was never in the stored state
        cfg.chunking["brand_new"] = GroupChunkingConfig(
            strategy="fixed", unit="chars", chunk_size=600, overlap=60
        )

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "raggen.core.ingest.ingest_service.LocalSentenceTransformerEmbedder",
            lambda *a, **kw: _dummy_embedder_factory(),
        )

        result = do_ingest()
        drift = [w for w in result.warnings if w.code == "chunking_drift"]
        assert drift == []
