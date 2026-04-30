from __future__ import annotations

import json
from abc import ABC, abstractmethod

from raggen.core.results.envelope import ResultEnvelope
from raggen.core.results.formats import OutputFormat

_SNIPPET_WIDTH = 120


class OptionalDependencyError(RuntimeError):
    pass


class Renderer(ABC):
    @abstractmethod
    def render(self, result: ResultEnvelope) -> str:
        raise NotImplementedError


class JsonRenderer(Renderer):
    def __init__(self, indent: int = 2):
        self.indent = indent

    def render(self, result: ResultEnvelope) -> str:
        return json.dumps(
            result.to_plain(),
            indent=self.indent,
            ensure_ascii=False,
        )


class XmlRenderer(Renderer):
    def render(self, result: ResultEnvelope) -> str:
        try:
            from .xml_adapter import to_xml_string
        except ImportError as exc:
            raise OptionalDependencyError(
                "XML output requires the optional dependency 'pydantic-xml'. "
                "Install it with: pip install pydantic-xml"
            ) from exc

        return to_xml_string(result)


class TextRenderer(Renderer):
    def render(self, result: ResultEnvelope) -> str:
        op = result.operation
        if op == "ingest":
            lines = self._render_ingest(result)
        elif op == "query":
            lines = self._render_query(result)
        elif op == "init":
            lines = self._render_init(result)
        elif op == "build":
            lines = self._render_build(result)
        else:
            lines = self._render_fallback(result)

        if result.warnings:
            lines.append("")
            for w in result.warnings:
                lines.append(f"! {w.message}")

        if result.errors:
            lines.append("")
            for e in result.errors:
                lines.append(f"ERROR [{e.code}]: {e.message}")

        return "\n".join(lines)

    def _render_ingest(self, result: ResultEnvelope) -> list[str]:
        d = result.data or {}
        if not result.success:
            return ["Ingest failed."]

        docs = d.get("docs_parsed", 0)
        chunks = d.get("chunks", 0)
        skipped = d.get("docs_skipped", 0)
        removed = d.get("docs_removed", 0)

        lines = [f"{docs} docs ingested  ({chunks} chunks)"]
        parts = []
        if skipped:
            parts.append(f"{skipped} unchanged")
        if removed:
            parts.append(f"{removed} removed")
        if parts:
            lines.append("  " + ", ".join(parts))
        return lines

    def _render_query(self, result: ResultEnvelope) -> list[str]:
        from raggen.core.query.models import QueryResponse

        if not result.success or result.data is None:
            return ["Query failed."]

        data = result.data

        if not isinstance(data, QueryResponse):
            # summary dict — chunk details not available
            lines = [f'Query: "{data.get("query", "")}"']
            lines.append(f"  {data.get('matches', 0)
                              } matches  (use --detailed to see results)")
            if data.get("answer"):
                lines.append(f"  Answer: {data['answer']}")
            return lines

        lines = [f'Query: "{data.query}"', ""]

        if not data.matches:
            lines.append("  No matches found.")
            return lines

        for i, chunk in enumerate(data.matches, 1):
            lines.append(f"[{i}] {chunk.doc_id}  (score: {chunk.score:.2f})")
            lines.append(f"    {_snippet(chunk.text)}")
            lines.append("")

        if data.answer:
            lines.append(f"Answer: {data.answer}")

        return lines

    def _render_init(self, result: ResultEnvelope) -> list[str]:
        d = result.data or {}
        if not result.success:
            return ["Init failed."]

        lines = [f"Initialized {d.get('project_root', '')}"]
        if d.get("config_path"):
            lines.append(f"  Config:  {d['config_path']}")
        if d.get("state"):
            lines.append(f"  State:   {d['state']}")
        return lines

    def _render_build(self, result: ResultEnvelope) -> list[str]:
        d = result.data
        if not result.success:
            lines = ["Build failed."]
            return lines

        lines = ["Database initialized"]
        before, after = d.get("state_before", ""), d.get("state_after", "")
        if before and after:
            lines.append(f"  {before} -> {after}")
        changed = d.get("changed_foundation_fields", [])
        if changed:
            lines.append(f"  Changed: {', '.join(changed)}")
        return lines

    def _render_fallback(self, result: ResultEnvelope) -> list[str]:
        lines = [f"{result.operation}  success={result.success}"]
        if isinstance(result.data, dict):
            for k, v in result.data.items():
                lines.append(f"  {k}: {v}")
        return lines


def _snippet(text: str) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= _SNIPPET_WIDTH:
        return cleaned
    return cleaned[:_SNIPPET_WIDTH - 3] + "..."


def get_renderer(fmt: OutputFormat) -> Renderer:
    if fmt == OutputFormat.TEXT:
        return TextRenderer()
    if fmt == OutputFormat.JSON:
        return JsonRenderer()
    if fmt == OutputFormat.XML:
        return XmlRenderer()
    raise ValueError(f"Unsupported output format: {fmt}")
