from __future__ import annotations

from raggen.core.query.models import RetrievedChunk


class GenerationNotImplementedError(NotImplementedError):
    pass


def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    model_id: str,
) -> str:
    """
    Placeholder generation interface.

    Future implementations may route to different providers/models,
    but query service should depend only on this function-level contract.
    """
    raise GenerationNotImplementedError(
        f"Answer generation is not implemented yet (requested model: {model_id!r})."
    )
