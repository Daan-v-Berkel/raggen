from __future__ import annotations


class SchemaMismatchError(Exception):
    pass


class AlreadyInitializedError(Exception):
    pass


class InvalidConfigError(Exception):
    pass
