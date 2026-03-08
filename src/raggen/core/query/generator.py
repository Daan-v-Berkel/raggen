from __future__ import annotations

from raggen.core.config.project import ProjectConfig
from raggen.core.query.generation_models import GenerationResult
from raggen.core.query.models import RetrievedChunk


class GenerationNotImplementedError(NotImplementedError):
    """Raised when generation is requested but no provider is implemented."""
    pass


def generate_answer(
    *,
    query: str,
    chunks: list[RetrievedChunk],
    cfg: ProjectConfig,
) -> GenerationResult:
    """
    Stable generation entrypoint.

    This defines the seam where future generation providers will plug in,
    but does not implement any real provider calls yet.
    """
    if not cfg.generation.enabled:
        return GenerationResult(
            text="",
            model_id="",
            provider="",
            usage=None,
        )

    provider = cfg.generation.provider.strip()
    model_id = cfg.generation.model_id.strip()

    if not provider:
        raise ValueError(
            "Generation is enabled, but no generation provider is configured."
        )

    if not model_id:
        raise ValueError(
            "Generation is enabled, but no generation model_id is configured."
        )

    raise GenerationNotImplementedError(
        f"Generation provider '{provider}' is not implemented yet."
    )
