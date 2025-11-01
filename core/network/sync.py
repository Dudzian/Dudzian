"""Narzędzia do bezpiecznego wykonywania asynchronicznych wywołań w kontekście synchronicznym."""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import threading
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from anyio.from_thread import BlockingPortal, start_blocking_portal


__all__ = ["run_sync"]


_ResultT = TypeVar("_ResultT")

_PORTAL_CM: contextlib.AbstractContextManager[BlockingPortal] | None = None
_PORTAL: BlockingPortal | None = None
_PORTAL_LOCK = threading.Lock()


def _shutdown_portal() -> None:
    global _PORTAL_CM, _PORTAL  # noqa: PLW0603 - aktualizujemy modułowy stan współdzielony
    with _PORTAL_LOCK:
        if _PORTAL_CM is None:
            return
        cm = _PORTAL_CM
        _PORTAL_CM = None
        _PORTAL = None
    cm.__exit__(None, None, None)


def _ensure_portal() -> BlockingPortal:
    global _PORTAL_CM, _PORTAL  # noqa: PLW0603 - aktualizujemy modułowy stan współdzielony
    with _PORTAL_LOCK:
        if _PORTAL is None:
            cm = start_blocking_portal()
            portal = cm.__enter__()
            _PORTAL_CM = cm
            _PORTAL = portal
            atexit.register(_shutdown_portal)
        assert _PORTAL is not None  # pragma: no cover - ochrona przed mypy
        return _PORTAL


def run_sync(
    async_fn: Callable[..., Awaitable[_ResultT]],
    *args,
    **kwargs,
) -> _ResultT:
    """Uruchamia funkcję asynchroniczną w kontekście synchronicznym.

    Funkcja deleguje wykonanie do współdzielonego portalu AnyIO, dzięki czemu
    można bezpiecznie korzystać z adapterów HTTP w wątkach interfejsu
    użytkownika lub testach integracyjnych bez ręcznego tworzenia pętli
    ``asyncio``.
    """

    portal = _ensure_portal()
    ctx = contextvars.copy_context()
    kwargs_copy = dict(kwargs)
    return portal.call(_call_with_context, ctx, async_fn, args, kwargs_copy)


async def _call_with_context(
    ctx: contextvars.Context,
    async_fn: Callable[..., Awaitable[_ResultT]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> _ResultT:
    coroutine = ctx.run(_create_coroutine, async_fn, args, kwargs)
    return await coroutine


def _create_coroutine(
    async_fn: Callable[..., Awaitable[_ResultT]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Awaitable[_ResultT]:
    return async_fn(*args, **kwargs)

