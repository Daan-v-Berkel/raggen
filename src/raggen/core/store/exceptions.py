from __future__ import annotations


class SchemaMismatchError(Exception):
    pass


class AlreadyInitializedError(Exception):
    pass


class InvalidConfigError(Exception):
    pass


class BackendLoadError(Exception):
    pass


class BackendNotSupportedError(Exception):
    pass


class VectorSchemaError(Exception):
    pass
