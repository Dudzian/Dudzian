"""Polityka wyjątków dla usług core."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator, TypeVar

from KryptoLowca.logging_utils import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])
@contextmanager
def exception_guard(component: str, *, re_raise: bool = False) -> Generator[None, None, None]:
    """Chroni bloki kodu przed nieobsłużonymi wyjątkami."""

    try:
        yield
    except Exception as exc:  # pragma: no cover - defensywna warstwa
        logger.exception("Błąd w komponencie %s", component)
        if re_raise:
            raise


def guard_exceptions(component: str, *, re_raise: bool = False) -> Callable[[F], F]:
    """Dekorator obsługujący wyjątki dla funkcji synchronicznych i asynchronicznych."""

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:  # pragma: no cover
                    logger.exception("Błąd w komponencie %s", component)
                    if re_raise:
                        raise
                    return None

            return async_wrapper  # type: ignore[return-value]

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # pragma: no cover
                logger.exception("Błąd w komponencie %s", component)
                if re_raise:
                    raise
                return None

        return sync_wrapper  # type: ignore[return-value]

    return decorator


__all__ = ["exception_guard", "guard_exceptions"]
