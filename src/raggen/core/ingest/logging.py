from __future__ import annotations

import logging

logger = logging.getLogger("raggen")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


def log_stage(stage: str, info: dict | None = None) -> None:
    logger.info(f"stage={stage} info={info}")


def log_error(doc_id: str | None, stage: str, exc: Exception) -> None:
    logger.exception(f"error in stage={stage} doc_id={doc_id}: {exc}")
