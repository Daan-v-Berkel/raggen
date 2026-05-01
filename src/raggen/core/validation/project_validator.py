"""
ProjectValidator — unified validation entry point for build and ingest.

No new validation logic lives here.  This module orchestrates the checks
defined in Steps 1–6 into two consistent sequences with a defined output
ordering:

  validate_for_build:
    1. EmbeddingConfigValidator (Step 3) — dim mismatch → ModelCapabilityError
       Includes dim resolution as a side effect on cfg.embedding.dim.
    2. validate_existing_project (Step 5) — BREAKING schema change → SchemaMismatchError
       Skipped when destructive=True (schema will be dropped and recreated).

  validate_for_ingest:
    1. validate_existing_project (Step 5) — BREAKING schema change → SchemaMismatchError
    2. Chunking drift check (Step 6)   — advisory warnings, does not raise
    3. Max-seq-length check (Step 2)   — tokens: ConfigError; chars: advisory warning

Advisory warnings are returned as a list[ResultMessage] so the caller can add
them to the result envelope.  Hard errors are raised as exceptions.

Output contract
---------------
- Errors  : raised; CLI catches and prints str(e) without a stack trace
- Warnings: emitted via logger.warning() AND returned for the result envelope
- All output precedes any progress output from the command itself
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from raggen.core.config.project import ConfigError, ProjectConfig
from raggen.core.embeddings.capabilities import ModelCapabilities
from raggen.core.embeddings.config_validator import EmbeddingConfigValidator
from raggen.core.metadata.store import compute_chunking_hash
from raggen.core.results.envelope import ResultMessage
from raggen.core.store.initializer import validate_existing_project

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine
    from raggen.core.metadata.models import ProjectState

logger = logging.getLogger("raggen")

_CHARS_PER_TOKEN_FLOOR = 3  # conservative floor: code / CJK worst-case


class ProjectValidator:

    @staticmethod
    def validate_for_build(
        cfg: ProjectConfig,
        caps: ModelCapabilities,
        engine: "Engine",
        *,
        destructive: bool = False,
    ) -> None:
        """
        Run all build-time validations.

        Side effect: resolves ``cfg.embedding.dim`` to a concrete integer when
        it was ``None`` in the config.  This must happen before
        ``validate_existing_project`` so the schema comparison has a real dim.

        Raises:
          ModelCapabilityError: pinned dim contradicts what the model produces
          SchemaMismatchError: BREAKING config change vs stored schema
                               (not raised when ``destructive=True``)
        """
        # Step 3 — raises ModelCapabilityError if pinned dim is wrong
        EmbeddingConfigValidator.validate(cfg.embedding, caps)

        # Resolve dim so every downstream call (snapshot, DB insert, schema
        # check) sees a concrete integer rather than None.
        cfg.embedding.dim = cfg.embedding.dim or caps.actual_dim

        # Step 5 — raises SchemaMismatchError on BREAKING changes.
        # Skipped for destructive builds: the schema is about to be dropped and
        # recreated, so a mismatch against the old schema is expected and fine.
        if not destructive:
            validate_existing_project(engine, cfg)

    @staticmethod
    def validate_for_ingest(
        cfg: ProjectConfig,
        engine: "Engine",
        state: "ProjectState | None",
        caps: ModelCapabilities,
    ) -> list[ResultMessage]:
        """
        Run all ingest-time validations.

        Returns a list of advisory ``ResultMessage`` objects (chunking drift,
        chunk-size estimate) to be added to the ingest result envelope.
        Hard errors are raised as exceptions.

        Raises:
          SchemaMismatchError: BREAKING config change vs stored schema
          ConfigError: chunk_size (tokens unit) exceeds model capacity
        """
        warnings: list[ResultMessage] = []

        # ------------------------------------------------------------------ #
        # 1. Schema check — catch BREAKING changes (e.g. model_id changed     #
        #    since last build) before touching any files.                      #
        # ------------------------------------------------------------------ #
        validate_existing_project(engine, cfg)

        # ------------------------------------------------------------------ #
        # 2. Chunking drift — advisory; ingest continues after warnings        #
        # ------------------------------------------------------------------ #
        if state is not None:
            for group, group_conf in cfg.chunking.items():
                stored_hash = state.foundation.chunking_hashes.get(group)
                if stored_hash is not None:
                    current_hash = compute_chunking_hash(group_conf)
                    if current_hash != stored_hash:
                        m = (
                            f"Warning: chunking config for group '{group}' "
                            f"has changed since last build.\n"
                            f"\n"
                            f"Existing indexed chunks for this group may not "
                            f"reflect current settings.\n"
                            f"Run 'rag ingest --force' to re-index all files "
                            f"with the current settings."
                        )
                        logger.warning(m)
                        warnings.append(
                            ResultMessage(code="chunking_drift", message=m)
                        )

        # ------------------------------------------------------------------ #
        # 3. Max-seq-length check — uses cached caps, no model load required  #
        # ------------------------------------------------------------------ #
        _max_seq = caps.max_seq_length
        _usable_tokens = _max_seq - 2  # reserve CLS + SEP

        for group, group_conf in cfg.chunking.items():
            if group_conf.unit == "tokens":
                if group_conf.chunk_size > _usable_tokens:
                    raise ConfigError(
                        f"Chunking group '{group}': "
                        f"chunk_size={group_conf.chunk_size} tokens "
                        f"exceeds the model's usable capacity "
                        f"({_max_seq} max − 2 special tokens = {_usable_tokens}). "
                        f"Reduce chunk_size to {_usable_tokens} or below."
                    )
            elif group_conf.unit == "chars":
                estimated_tokens = group_conf.chunk_size // _CHARS_PER_TOKEN_FLOOR
                if estimated_tokens > _usable_tokens:
                    m = (
                        f"Chunking group '{group}': "
                        f"chunk_size={group_conf.chunk_size} chars "
                        f"may produce chunks up to ~{estimated_tokens} tokens "
                        f"(estimated at {
                            _CHARS_PER_TOKEN_FLOOR} chars/token — "
                        f"actual varies by content). "
                        f"The model supports {_usable_tokens} content tokens. "
                        f"Chunks that exceed this will be silently truncated "
                        f"during embedding. "
                        f"Consider reducing chunk_size or switching to "
                        f"unit='tokens'."
                    )
                    logger.warning(m)
                    warnings.append(
                        ResultMessage(
                            code="chunk_size_estimate_warning", message=m)
                    )

        return warnings
