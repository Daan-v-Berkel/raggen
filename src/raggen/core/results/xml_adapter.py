from __future__ import annotations

import json
from typing import Any

from raggen.core.results.envelope import ResultEnvelope
from .xml_models import (
    XmlDataContainer,
    XmlKeyValueItem,
    XmlResultEnvelope,
    XmlResultMessage,
    XmlResultMeta,
)


def _stringify_data_items(data: Any) -> list[XmlKeyValueItem]:
    """
    Minimal v1 strategy:
    flatten only top-level dict keys into XML items,
    storing nested values as JSON strings.

    This keeps the XML contract stable without requiring dynamic XML model
    generation for arbitrary nested payloads.
    """
    if data is None:
        return []

    if isinstance(data, dict):
        items: list[XmlKeyValueItem] = []
        for key, value in data.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                items.append(XmlKeyValueItem(
                    key=key, value="" if value is None else str(value)))
            else:
                items.append(
                    XmlKeyValueItem(
                        key=key,
                        value=json.dumps(value, ensure_ascii=False),
                    )
                )
        return items

    return [XmlKeyValueItem(key="value", value=str(data))]


def to_xml_model(result: ResultEnvelope) -> XmlResultEnvelope:
    data_model = None
    if result.data is not None:
        data_model = XmlDataContainer(items=_stringify_data_items(result.data))

    meta_model = None
    if result.meta is not None:
        meta_model = XmlResultMeta(
            trace_id=result.meta.trace_id,
            duration_ms=result.meta.duration_ms,
        )

    return XmlResultEnvelope(
        schema_version=result.schema_version,
        operation=result.operation,
        success=result.success,
        data=data_model,
        warnings=[
            XmlResultMessage(code=item.code, message=item.message)
            for item in result.warnings
        ],
        errors=[
            XmlResultMessage(code=item.code, message=item.message)
            for item in result.errors
        ],
        meta=meta_model,
    )


def to_xml_string(result: ResultEnvelope) -> str:
    xml_model = to_xml_model(result)
    return xml_model.to_xml(
        encoding="unicode",
        exclude_none=True,
    )
