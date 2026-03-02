from .chunks import ChunkConfig, Chunk
import hashlib
import json
from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ConfigError(ValueError):
    """
    Raised when a ChunkConfig is structurally valid (Pydantic),
    but semantically invalid for the Chunker (impossible combinations,
    unsupported strategy/structure mismatches, etc.).
    """

    def __init__(self, message: str, *, config: ChunkConfig | None = None):
        self.config = config
        super().__init__(message)

    def __str__(self) -> str:
        base = super().__str__()
        if self.config is not None:
            return f"{base}\nConfig: {self.config.model_dump()}"
        return base


class Chunker:

    def __init__(self, doc: Any):
        """Chunker operates on a simple Document-like object with attributes:
        - doc_id: str
        - text: str
        - source: object (e.g., SourceRef)
        """
        self.STRATEGIES = {
            "fixed": self._chunk_fixed,
            "headingAware": self._chunk_heading,
            "paragraphMerge": self._chunk_paragraph,
            "tokenAware": self._chunk_token,
        }
        self.document = doc

    def validate_config(self, conf: ChunkConfig) -> None:
        """
        Validate ChunkConfig in two phases:
          1) Pydantic validation (raises pydantic.ValidationError)
          2) "Impossible combo" validation (raises ConfigError with actionable messages)

        Returns: None (validation passes)
        """
        try:
            conf = ChunkConfig.model_validate(conf)
        except Exception:
            # Let Pydantic's own ValidationError bubble up unchanged
            raise

        errors: List[str] = []

        # Relationship constraints
        if conf.chunk_size > 0 and conf.overlap >= conf.chunk_size:
            errors.append(
                f"overlap ({conf.overlap}) must be smaller than chunk_size ({conf.chunk_size})."
            )

        if conf.min_chunk_size > 0 and conf.chunk_size > 0 and conf.min_chunk_size > conf.chunk_size:
            errors.append(
                f"min_chunk_size ({conf.min_chunk_size}) cannot be larger than chunk_size ({conf.chunk_size})."
            )

        # Unit/Tokenizer constraints
        if conf.unit == "tokens" and conf.tokenizer.name == "":
            errors.append(
                "unit='tokens' requires a tokenizer configuration.")

        if errors:
            raise ConfigError("Invalid ChunkConfig:\n- " + "\n- ".join(errors))

        return None

    def _stable_config_hash(self, conf: ChunkConfig) -> str:
        conf_json = json.dumps(
            conf.model_dump(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(conf_json.encode("utf-8")).hexdigest()

    def _enrich_chunks(self, conf: ChunkConfig, pieces: list[str]) -> list[Chunk]:
        """Create Chunk objects from piece strings without relying on character offsets."""
        config_hash = self._stable_config_hash(conf)

        out: List[Chunk] = []
        for idx, piece in enumerate(pieces):
            chunk_id = f"{getattr(self.document, 'doc_id', 'unknown')}:{config_hash}:{idx}"

            meta = {
                "page_start": None,
                "page_end": None,
                "heading": None,
                "section_path": None,
                "source": getattr(self.document, "source", None),
            }

            stats = {
                "char_count": len(piece) if piece is not None else None,
                "token_count": None,
            }

            out.append(
                Chunk(
                    doc_id=getattr(self.document, "doc_id", "unknown"),
                    chunk_index=idx,
                    text=piece,
                    start_char=None,
                    end_char=None,
                    metadata=meta,
                    stats=stats,
                    config_hash=config_hash,
                    chunk_id=chunk_id,
                )
            )

        return out

    def chunk(self, conf: ChunkConfig) -> list[Chunk]:
        pieces = self.STRATEGIES[conf.strategy](conf)
        return self._enrich_chunks(conf, pieces)

    def _chunk_fixed(self, conf: ChunkConfig) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=conf.chunk_size,
            chunk_overlap=conf.overlap,
            separators=conf.separators,
            keep_separator=conf.preserve_newlines,
        )
        return splitter.split_text(getattr(self.document, "text", ""))

    def _chunk_heading(self, conf: ChunkConfig) -> list[str]:
        # Not implemented: fallback to fixed
        return self._chunk_fixed(conf)

    def _chunk_paragraph(self, conf: ChunkConfig) -> list[str]:
        # Simple paragraph-based chunking: split on double-newline and then apply merging
        text = getattr(self.document, "text", "")
        paras = text.split("\n\n") if text else []
        out: List[str] = []
        for p in paras:
            if p == "":
                continue
            if len(p) <= conf.chunk_size:
                out.append(p)
            else:
                # fallback to fixed-size slicing within paragraph
                start = 0
                while start < len(p):
                    end = min(len(p), start + conf.chunk_size)
                    out.append(p[start:end])
                    start = max(0, end - conf.overlap)
        return out

    def _chunk_token(self, conf: ChunkConfig) -> list[str]:
        # Token-aware chunking not implemented yet; fallback to fixed
        return self._chunk_fixed(conf)
