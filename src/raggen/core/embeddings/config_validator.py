"""
EmbeddingConfigValidator — validates that a pinned embedding.dim matches what
the model actually produces, and resolves a missing dim from the model cache.

Raised exception: ModelCapabilityError (RuntimeError subclass).
"""
from __future__ import annotations

from raggen.core.config.project import EmbeddingConfig
from raggen.core.embeddings.capabilities import ModelCapabilities


class ModelCapabilityError(RuntimeError):
    """
    Raised when a config value contradicts the actual model capabilities.

    Typical trigger: ``embedding.dim`` in config.toml does not match the
    dimension the model returns when inspected.
    """


class EmbeddingConfigValidator:
    @staticmethod
    def validate(cfg: EmbeddingConfig, caps: ModelCapabilities) -> None:
        """
        Validate ``cfg.dim`` against the cached model capabilities.

        - ``cfg.dim is None``           → no error; dim will be resolved from ``caps.actual_dim``
        - ``cfg.dim == caps.actual_dim`` → no error
        - ``cfg.dim != caps.actual_dim`` → raise ``ModelCapabilityError`` with a clear fix message
        """
        if cfg.dim is None:
            return

        if cfg.dim == caps.actual_dim:
            return

        raise ModelCapabilityError(
            f"Embedding dimension mismatch for model '{caps.model_id}'.\n"
            f"\n"
            f"  Config specifies:  embedding.dim = {cfg.dim}\n"
            f"  Model produces:    {caps.actual_dim} dimensions\n"
            f"\n"
            f"Fix: update .rag/config.toml —\n"
            f"  Set   embedding.dim = {caps.actual_dim}\n"
            f"  or remove the line entirely to let it be detected automatically."
        )
