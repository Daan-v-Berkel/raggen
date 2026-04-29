"""
Tests for Step 5: tier-aware SchemaMismatchError formatting and
validate_existing_project() raising only on BREAKING changes.
"""
from __future__ import annotations

import pytest

from raggen.core.config.drift_tiers import DriftTier
from raggen.core.store.exceptions import FieldChange, SchemaMismatchError


# ---------------------------------------------------------------------------
# Helpers — build FieldChange lists for common scenarios
# ---------------------------------------------------------------------------

def _model_change() -> list[FieldChange]:
    from raggen.core.config.drift_tiers import field_tier_info
    info = field_tier_info("embedding.model_id")
    return [FieldChange(
        field="embedding.model_id",
        old_value="all-MiniLM-L6-v2",
        new_value="all-mpnet-base-v2",
        tier=info.tier,
        reason=info.reason,
    )]


def _dim_change() -> list[FieldChange]:
    from raggen.core.config.drift_tiers import field_tier_info
    info = field_tier_info("embedding.dim")
    return [FieldChange(
        field="embedding.dim",
        old_value=384,
        new_value=768,
        tier=info.tier,
        reason=info.reason,
    )]


def _multi_change() -> list[FieldChange]:
    from raggen.core.config.drift_tiers import field_tier_info
    changes = []
    for field, old, new in [
        ("embedding.model_id", "all-MiniLM-L6-v2", "all-mpnet-base-v2"),
        ("embedding.dim", 384, 768),
    ]:
        info = field_tier_info(field)
        changes.append(FieldChange(field=field, old_value=old, new_value=new,
                                   tier=info.tier, reason=info.reason))
    return changes


# ---------------------------------------------------------------------------
# SchemaMismatchError basic contract
# ---------------------------------------------------------------------------


class TestSchemaMismatchErrorBasics:
    def test_is_exception_subclass(self):
        err = SchemaMismatchError(_model_change())
        assert isinstance(err, Exception)

    def test_changes_attribute_stored(self):
        changes = _model_change()
        err = SchemaMismatchError(changes)
        assert err.changes == changes

    def test_str_is_idempotent(self):
        """str(err) must return the same string on every call."""
        err = SchemaMismatchError(_model_change())
        assert str(err) == str(err)
        assert str(err) == err.args[0]

    def test_empty_changes_does_not_raise(self):
        """Constructing with an empty list must not crash."""
        err = SchemaMismatchError([])
        assert "rebuild" in str(err).lower()


# ---------------------------------------------------------------------------
# Message content
# ---------------------------------------------------------------------------


class TestMessageContent:
    def test_starts_with_configuration_change(self):
        msg = str(SchemaMismatchError(_model_change()))
        assert msg.startswith("Configuration change requires a full rebuild.")

    def test_contains_what_changed_header(self):
        msg = str(SchemaMismatchError(_model_change()))
        assert "What changed" in msg

    def test_contains_why_it_matters_header(self):
        msg = str(SchemaMismatchError(_model_change()))
        assert "Why this matters" in msg

    def test_contains_to_proceed_header(self):
        msg = str(SchemaMismatchError(_model_change()))
        assert "To proceed" in msg

    def test_ends_with_destructive_command(self):
        msg = str(SchemaMismatchError(_model_change()))
        assert "rag build --destructive" in msg

    def test_contains_warning_about_data_loss(self):
        msg = str(SchemaMismatchError(_model_change()))
        assert "--destructive" in msg
        assert "re-ingest" in msg.lower() or "indexed data" in msg.lower()

    def test_model_change_shows_old_and_new_model(self):
        msg = str(SchemaMismatchError(_model_change()))
        assert "all-MiniLM-L6-v2" in msg
        assert "all-mpnet-base-v2" in msg

    def test_model_change_shows_field_name(self):
        msg = str(SchemaMismatchError(_model_change()))
        assert "embedding.model_id" in msg

    def test_dim_change_shows_old_and_new_dim(self):
        msg = str(SchemaMismatchError(_dim_change()))
        assert "384" in msg
        assert "768" in msg

    def test_dim_change_shows_field_name(self):
        msg = str(SchemaMismatchError(_dim_change()))
        assert "embedding.dim" in msg

    def test_string_values_are_quoted(self):
        """String old/new values must appear in quotes in 'What changed'."""
        msg = str(SchemaMismatchError(_model_change()))
        assert '"all-MiniLM-L6-v2"' in msg
        assert '"all-mpnet-base-v2"' in msg

    def test_int_values_are_not_quoted(self):
        """Integer values must not be wrapped in quotes."""
        msg = str(SchemaMismatchError(_dim_change()))
        assert '"384"' not in msg
        assert '"768"' not in msg

    def test_arrow_separator_present(self):
        msg = str(SchemaMismatchError(_model_change()))
        assert "→" in msg

    def test_multi_change_lists_all_fields(self):
        msg = str(SchemaMismatchError(_multi_change()))
        assert "embedding.model_id" in msg
        assert "embedding.dim" in msg

    def test_reason_text_included(self):
        """The reason string must appear in the 'Why this matters' section."""
        msg = str(SchemaMismatchError(_model_change()))
        # The model_id reason mentions vectors/model
        assert "vector" in msg.lower() or "model" in msg.lower()


# ---------------------------------------------------------------------------
# validate_existing_project — integration via init_database
# ---------------------------------------------------------------------------


class TestValidateExistingProject:
    """
    These tests call init_database twice: once to create the schema, once to
    detect drift.  They rely on cfg_factory and write_cfg from conftest.
    """

    def test_breaking_change_raises_schema_mismatch(
        self, tmp_path, cfg_factory, write_cfg
    ):
        """Changing embedding_dim (BREAKING) must raise SchemaMismatchError."""
        cfg1 = cfg_factory(tmp_path, embedding_dim=1234)
        cfg_path = write_cfg(cfg1, tmp_path)
        from raggen.core.bootstrap import bootstrap
        from raggen.core.store import init_database
        cfg1 = bootstrap(cfg_path)
        init_database(cfg1)

        cfg2 = cfg_factory(tmp_path, embedding_dim=9999)
        with pytest.raises(SchemaMismatchError):
            init_database(cfg2)

    def test_schema_mismatch_error_carries_changes(
        self, tmp_path, cfg_factory, write_cfg
    ):
        """SchemaMismatchError.changes must contain the differing field."""
        cfg1 = cfg_factory(tmp_path, embedding_dim=1234)
        cfg_path = write_cfg(cfg1, tmp_path)
        from raggen.core.bootstrap import bootstrap
        from raggen.core.store import init_database
        cfg1 = bootstrap(cfg_path)
        init_database(cfg1)

        cfg2 = cfg_factory(tmp_path, embedding_dim=9999)
        with pytest.raises(SchemaMismatchError) as exc_info:
            init_database(cfg2)

        fields = [c.field for c in exc_info.value.changes]
        assert "embedding.dim" in fields

    def test_schema_mismatch_error_message_ends_with_destructive(
        self, tmp_path, cfg_factory, write_cfg
    ):
        """The error message must mention 'rag build --destructive'."""
        cfg1 = cfg_factory(tmp_path, embedding_dim=1234)
        cfg_path = write_cfg(cfg1, tmp_path)
        from raggen.core.bootstrap import bootstrap
        from raggen.core.store import init_database
        cfg1 = bootstrap(cfg_path)
        init_database(cfg1)

        cfg2 = cfg_factory(tmp_path, embedding_dim=9999)
        with pytest.raises(SchemaMismatchError) as exc_info:
            init_database(cfg2)

        assert "rag build --destructive" in str(exc_info.value)

    def test_runtime_only_change_does_not_raise(
        self, tmp_path, cfg_factory, write_cfg
    ):
        """Changing query_model_id (RUNTIME tier) must not raise SchemaMismatchError."""
        cfg1 = cfg_factory(tmp_path)
        cfg_path = write_cfg(cfg1, tmp_path)
        from raggen.core.bootstrap import bootstrap
        from raggen.core.store import init_database
        cfg1 = bootstrap(cfg_path)
        init_database(cfg1)

        # query_model_id is RUNTIME — changing it must be silently ignored
        cfg2 = cfg_factory(tmp_path)
        cfg2.query.model_id = "some-different-query-model"
        # Should not raise
        init_database(cfg2)

    def test_same_config_does_not_raise(self, tmp_path, cfg_factory, write_cfg):
        """A second init_database call with the same config must not raise."""
        cfg = cfg_factory(tmp_path)
        cfg_path = write_cfg(cfg, tmp_path)
        from raggen.core.bootstrap import bootstrap
        from raggen.core.store import init_database
        cfg = bootstrap(cfg_path)
        init_database(cfg)
        init_database(cfg)  # must not raise

    def test_error_message_no_raw_dict_dump(
        self, tmp_path, cfg_factory, write_cfg
    ):
        """The error must not contain a raw JSON dict dump (old format)."""
        cfg1 = cfg_factory(tmp_path, embedding_dim=1234)
        cfg_path = write_cfg(cfg1, tmp_path)
        from raggen.core.bootstrap import bootstrap
        from raggen.core.store import init_database
        cfg1 = bootstrap(cfg_path)
        init_database(cfg1)

        cfg2 = cfg_factory(tmp_path, embedding_dim=9999)
        with pytest.raises(SchemaMismatchError) as exc_info:
            init_database(cfg2)

        msg = str(exc_info.value)
        # Old format had "Stored project configuration differs:" and a JSON blob
        assert "Stored project configuration differs" not in msg
        assert '"stored"' not in msg
