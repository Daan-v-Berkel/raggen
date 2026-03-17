from __future__ import annotations

from pydantic_xml import BaseXmlModel, element


class XmlResultMessage(BaseXmlModel, tag="message"):
    code: str = element()
    message: str = element()


class XmlResultMeta(BaseXmlModel, tag="meta"):
    trace_id: str | None = element(default=None)
    duration_ms: int | None = element(default=None)


class XmlScalarItem(BaseXmlModel, tag="item"):
    value: str = element()


class XmlKeyValueItem(BaseXmlModel, tag="item"):
    key: str = element()
    value: str = element()


class XmlDataContainer(BaseXmlModel, tag="data"):
    """
    Minimal starter model.

    This keeps the first version simple by flattening data into repeated
    key/value items. Later, if you want richer nested XML, you can replace
    this with per-operation XML models.
    """
    items: list[XmlKeyValueItem] = element(tag="item", default_factory=list)


class XmlResultEnvelope(BaseXmlModel, tag="result"):
    schema_version: str = element()
    operation: str = element()
    success: bool = element()

    data: XmlDataContainer | None = element(default=None)
    warnings: list[XmlResultMessage] = element(
        tag="message", default_factory=list)
    errors: list[XmlResultMessage] = element(
        tag="message", default_factory=list)
    meta: XmlResultMeta | None = element(default=None)
