from __future__ import annotations

from functools import wraps
from typing import Callable, TypeVar, ParamSpec

from raggen.core.results.envelope import ResultEnvelope
from raggen.core.runs.interface import RunStore


P = ParamSpec("P")
R = TypeVar("R", bound=ResultEnvelope)


def persist_result(run_store_factory: Callable[[], RunStore]):
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            result = func(*args, **kwargs)

            if not isinstance(result, ResultEnvelope):
                raise TypeError(
                    f"{func.__name__} must return ResultEnvelope, "
                    f"got {type(result).__name__}"
                )

            store = run_store_factory()
            store.save_result(result)
            return result

        return wrapper

    return decorator
