from __future__ import annotations

import json
from abc import ABC, abstractmethod

from raggen.core.results.envelope import ResultEnvelope
from raggen.core.results.formats import OutputFormat


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


def get_renderer(fmt: OutputFormat) -> Renderer:
    if fmt == OutputFormat.JSON:
        return JsonRenderer()
    if fmt == OutputFormat.XML:
        return XmlRenderer()
    raise ValueError(f"Unsupported output format: {fmt}")
