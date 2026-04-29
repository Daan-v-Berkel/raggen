"""
Tests for Step 4: DriftTier enum and classify_field().

Verifies every field in the mapping table, prefix matching for
chunking.<group>.* and query.*, the unknown-field fallback, and the
no-I/O guarantee (just import checks).
"""
from __future__ import annotations

import pytest

from raggen.core.config.drift_tiers import (
    DriftTier,
    TierInfo,
    classify_field,
    field_tier_info,
)


# ---------------------------------------------------------------------------
# Enum sanity
# ---------------------------------------------------------------------------


class TestDriftTierEnum:
    def test_has_breaking(self):
        assert DriftTier.BREAKING.value == "breaking"

    def test_has_stale(self):
        assert DriftTier.STALE.value == "stale"

    def test_has_runtime(self):
        assert DriftTier.RUNTIME.value == "runtime"

    def test_three_tiers(self):
        assert len(DriftTier) == 3


# ---------------------------------------------------------------------------
# BREAKING fields — exact matches
# ---------------------------------------------------------------------------


class TestBreakingFields:
    @pytest.mark.parametrize(
        "field",
        [
            "embedding.model_id",
            "embedding.dim",
            "embedding.normalize",
            "storage.backend_key",
            "storage.database_url",
        ],
    )
    def test_breaking_exact(self, field):
        assert classify_field(field) == DriftTier.BREAKING

    def test_embedding_model_id(self):
        assert classify_field("embedding.model_id") == DriftTier.BREAKING

    def test_embedding_dim(self):
        assert classify_field("embedding.dim") == DriftTier.BREAKING

    def test_embedding_normalize(self):
        assert classify_field("embedding.normalize") == DriftTier.BREAKING

    def test_storage_backend_key(self):
        assert classify_field("storage.backend_key") == DriftTier.BREAKING

    def test_storage_database_url(self):
        assert classify_field("storage.database_url") == DriftTier.BREAKING


# ---------------------------------------------------------------------------
# STALE fields — chunking.<group>.<suffix> prefix matching
# ---------------------------------------------------------------------------


class TestStaleFields:
    def test_chunking_fallback_strategy(self):
        assert classify_field("chunking.fallback.strategy") == DriftTier.STALE

    def test_chunking_fallback_chunk_size(self):
        assert classify_field("chunking.fallback.chunk_size") == DriftTier.STALE

    def test_chunking_fallback_overlap(self):
        assert classify_field("chunking.fallback.overlap") == DriftTier.STALE

    def test_chunking_code_chunk_size(self):
        """The plan's explicit example: chunking.code.chunk_size → STALE."""
        assert classify_field("chunking.code.chunk_size") == DriftTier.STALE

    def test_chunking_docs_strategy(self):
        assert classify_field("chunking.docs.strategy") == DriftTier.STALE

    def test_chunking_any_group_overlap(self):
        assert classify_field("chunking.my_custom_group.overlap") == DriftTier.STALE

    @pytest.mark.parametrize("suffix", ["strategy", "chunk_size", "overlap"])
    def test_all_stale_suffixes(self, suffix):
        assert classify_field(f"chunking.some_group.{suffix}") == DriftTier.STALE


# ---------------------------------------------------------------------------
# RUNTIME fields — exact and prefix
# ---------------------------------------------------------------------------


class TestRuntimeFields:
    def test_embedding_batch_size(self):
        """The plan's explicit example: embedding.batch_size → RUNTIME."""
        assert classify_field("embedding.batch_size") == DriftTier.RUNTIME

    def test_embedding_model_cache_dir(self):
        assert classify_field("embedding.model_cache_dir") == DriftTier.RUNTIME

    def test_query_model_id(self):
        assert classify_field("query.model_id") == DriftTier.RUNTIME

    def test_query_top_k(self):
        assert classify_field("query.top_k") == DriftTier.RUNTIME

    def test_query_anything(self):
        assert classify_field("query.some_future_field") == DriftTier.RUNTIME


# ---------------------------------------------------------------------------
# Unknown fields → RUNTIME (safe default)
# ---------------------------------------------------------------------------


class TestUnknownFields:
    def test_completely_unknown_field(self):
        assert classify_field("totally.unknown.field") == DriftTier.RUNTIME

    def test_empty_string_does_not_raise(self):
        assert classify_field("") == DriftTier.RUNTIME

    def test_single_word_does_not_raise(self):
        assert classify_field("anything") == DriftTier.RUNTIME

    def test_chunking_with_unknown_suffix(self):
        """chunking.<group>.<unknown> should fall back to RUNTIME."""
        assert classify_field("chunking.fallback.unknown_option") == DriftTier.RUNTIME

    def test_near_miss_not_breaking(self):
        """Slightly wrong field names must not accidentally match BREAKING."""
        assert classify_field("embedding.model") == DriftTier.RUNTIME
        assert classify_field("embedding.dimensions") == DriftTier.RUNTIME


# ---------------------------------------------------------------------------
# field_tier_info — tier + reason string
# ---------------------------------------------------------------------------


class TestFieldTierInfo:
    def test_returns_tier_info_namedtuple(self):
        info = field_tier_info("embedding.model_id")
        assert isinstance(info, TierInfo)

    def test_breaking_field_has_non_empty_reason(self):
        info = field_tier_info("embedding.model_id")
        assert info.tier == DriftTier.BREAKING
        assert len(info.reason) > 0

    def test_stale_field_has_non_empty_reason(self):
        info = field_tier_info("chunking.docs.chunk_size")
        assert info.tier == DriftTier.STALE
        assert len(info.reason) > 0

    def test_runtime_field_has_non_empty_reason(self):
        info = field_tier_info("embedding.batch_size")
        assert info.tier == DriftTier.RUNTIME
        assert len(info.reason) > 0

    def test_unknown_field_returns_runtime_tier_info(self):
        info = field_tier_info("no.such.field")
        assert info.tier == DriftTier.RUNTIME
        assert len(info.reason) > 0

    def test_reason_mentions_model_for_model_id(self):
        info = field_tier_info("embedding.model_id")
        assert "model" in info.reason.lower()

    def test_reason_mentions_dimension_for_dim(self):
        info = field_tier_info("embedding.dim")
        assert "dimension" in info.reason.lower() or "dim" in info.reason.lower()

    def test_classify_field_consistent_with_tier_info(self):
        """classify_field must return the same tier as field_tier_info."""
        for field in [
            "embedding.model_id",
            "embedding.dim",
            "embedding.normalize",
            "storage.backend_key",
            "storage.database_url",
            "embedding.batch_size",
            "chunking.fallback.chunk_size",
            "query.top_k",
        ]:
            assert classify_field(field) == field_tier_info(field).tier
