"""
Tier classification for foundational config fields.

Every field that can appear in a ``FoundationalConfigSnapshot`` is assigned to
exactly one of three tiers:

    BREAKING — schema incompatible; ``rag build --destructive`` required
    STALE    — schema OK, but indexed data may be stale (advisory warning)
    RUNTIME  — safe to change between runs; no action required

``classify_field(field_path)`` is the public API.  ``field_tier_info`` returns
the tier *and* the human-readable reason string that Step 5 renders in error /
warning messages.

No I/O anywhere in this module.
"""
from __future__ import annotations

from enum import Enum
from typing import NamedTuple


class DriftTier(Enum):
    BREAKING = "breaking"  # schema incompatible, --destructive required
    STALE = "stale"        # schema OK, but indexed data is stale
    RUNTIME = "runtime"    # safe to change between runs


class TierInfo(NamedTuple):
    tier: DriftTier
    reason: str


# ---------------------------------------------------------------------------
# Exact-match field table
# ---------------------------------------------------------------------------

_EXACT: dict[str, TierInfo] = {
    # --- BREAKING ---
    "embedding.model_id": TierInfo(
        DriftTier.BREAKING,
        "Changing the embedding model invalidates all stored vectors — the old "
        "and new models embed text into incompatible vector spaces. Similarity "
        "search would return meaningless results.",
    ),
    "embedding.dim": TierInfo(
        DriftTier.BREAKING,
        "The vector table schema is fixed at creation time. Changing the "
        "embedding dimension requires rebuilding the schema from scratch.",
    ),
    "embedding.normalize": TierInfo(
        DriftTier.BREAKING,
        "Normalisation changes the geometry of all stored vectors. Mixing "
        "normalised and un-normalised vectors corrupts similarity search results.",
    ),
    "storage.backend_key": TierInfo(
        DriftTier.BREAKING,
        "Changing the storage backend means the existing database cannot be "
        "read by the new engine.",
    ),
    "storage.database_url": TierInfo(
        DriftTier.BREAKING,
        "Changing the database URL points to a different database. All "
        "existing indexed data is unreachable.",
    ),
    # --- RUNTIME ---
    "embedding.batch_size": TierInfo(
        DriftTier.RUNTIME,
        "Batch size is a performance tuning parameter and does not affect "
        "stored data or search results.",
    ),
    "embedding.model_cache_dir": TierInfo(
        DriftTier.RUNTIME,
        "The model cache directory controls where model weights are stored on "
        "disk; it does not affect indexed data.",
    ),
}

# ---------------------------------------------------------------------------
# Suffix-based matching for chunking.<group>.<suffix>
# ---------------------------------------------------------------------------

_CHUNKING_SUFFIX: dict[str, TierInfo] = {
    "strategy": TierInfo(
        DriftTier.STALE,
        "The chunking strategy determines how text is split. Existing indexed "
        "chunks used a different split method and may not reflect current settings.",
    ),
    "chunk_size": TierInfo(
        DriftTier.STALE,
        "Changing chunk_size shifts chunk boundaries. Existing indexed chunks "
        "were created with a different size and may not reflect current settings.",
    ),
    "overlap": TierInfo(
        DriftTier.STALE,
        "Changing overlap shifts chunk boundaries. Existing indexed chunks "
        "were created with a different overlap and may not reflect current settings.",
    ),
}

# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

_RUNTIME_FALLBACK = TierInfo(
    DriftTier.RUNTIME,
    "This setting does not affect stored data or the search index.",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def field_tier_info(field_path: str) -> TierInfo:
    """
    Return the ``TierInfo`` (tier + reason) for *field_path*.

    Matching rules, in order:
    1. Exact match in ``_EXACT``.
    2. Three-part path ``chunking.<group>.<suffix>`` where ``<suffix>`` is in
       ``_CHUNKING_SUFFIX``.
    3. Any path starting with ``query.`` → RUNTIME.
    4. Everything else → RUNTIME.
    """
    # 1. Exact match
    info = _EXACT.get(field_path)
    if info is not None:
        return info

    # 2. chunking.<group>.<suffix>
    parts = field_path.split(".")
    if len(parts) >= 3 and parts[0] == "chunking":
        suffix_info = _CHUNKING_SUFFIX.get(parts[2])
        if suffix_info is not None:
            return suffix_info

    # 3. query.*
    if field_path.startswith("query."):
        return _RUNTIME_FALLBACK

    # 4. Safe default
    return _RUNTIME_FALLBACK


def classify_field(field_path: str) -> DriftTier:
    """
    Return the ``DriftTier`` for *field_path*.

    Returns ``DriftTier.RUNTIME`` for any field not explicitly mapped — the
    safe default that does not block builds or warn unnecessarily.
    """
    return field_tier_info(field_path).tier
