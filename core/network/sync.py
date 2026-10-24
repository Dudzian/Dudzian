"""Narzędzia do bezpiecznego wykonywania asynchronicznych wywołań w kontekście synchronicznym."""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import threading
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar, cast

from anyio.from_thread import BlockingPortal, start_blocking_portal


__all__ = ["run_sync"]


_ResultT = TypeVar("_ResultT")

_PORTAL_CM: contextlib.AbstractContextManager[BlockingPortal] | None = None
_PORTAL: BlockingPortal | None = None
_PORTAL_CONDITION = threading.Condition()
_PORTAL_CALLS = 0
_PORTAL_SHUTTING_DOWN = False
_PORTAL_THREAD_ID: int | None = None
_PORTAL_CALL_CONTEXT = contextvars.ContextVar("network_sync_portal_call", default=False)


def _shutdown_portal() -> None:
    """Drain accepted calls and close the portal before its dependencies finalize."""

    global _PORTAL_CM, _PORTAL  # noqa: PLW0603 - aktualizujemy modułowy stan współdzielony
    global _PORTAL_SHUTTING_DOWN, _PORTAL_THREAD_ID
    with _PORTAL_CONDITION:
        if _PORTAL_CALL_CONTEXT.get() or threading.get_ident() == _PORTAL_THREAD_ID:
            raise RuntimeError("Cannot shut down the portal from an active portal call")
        while _PORTAL_SHUTTING_DOWN:
            _PORTAL_CONDITION.wait()
        if _PORTAL_CM is None:
            return
        _PORTAL_SHUTTING_DOWN = True
        _PORTAL_CONDITION.notify_all()
    try:
        with _PORTAL_CONDITION:
            while _PORTAL_CALLS:
                _PORTAL_CONDITION.wait()
            cm = _PORTAL_CM
            _PORTAL_CM = None
            _PORTAL = None
        cm.__exit__(None, None, None)
    finally:
        with _PORTAL_CONDITION:
            if _PORTAL is None:
                _PORTAL_THREAD_ID = None
            _PORTAL_SHUTTING_DOWN = False
            _PORTAL_CONDITION.notify_all()


def _ensure_portal() -> BlockingPortal:
    global _PORTAL_CM, _PORTAL  # noqa: PLW0603 - aktualizujemy modułowy stan współdzielony
    global _PORTAL_THREAD_ID
    with _PORTAL_CONDITION:
        if threading.get_ident() == _PORTAL_THREAD_ID:
            raise RuntimeError("Cannot call run_sync from the portal's event loop thread")
        if _PORTAL_SHUTTING_DOWN:
            raise RuntimeError("The portal is shutting down; new calls are not accepted")
        if _PORTAL is None:
            cm = start_blocking_portal()
            portal = cm.__enter__()
            _PORTAL_THREAD_ID = portal.call(threading.get_ident)
            _PORTAL_CM = cm
            _PORTAL = portal
        assert _PORTAL is not None  # pragma: no cover - ochrona przed mypy
        return _PORTAL


# Register once, before clients register their own cleanup. Registering while an
# atexit callback recreates the portal is too late for the current exit pass.
atexit.register(_shutdown_portal)


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

    Rozpoczęte wywołania kończą się przed zamknięciem portalu. Nowe wywołanie
    w trakcie zamykania zgłasza ``RuntimeError``; nie ponawiamy automatycznie
    operacji, które mogłyby mieć skutki uboczne.
    """

    global _PORTAL_CALLS
    with _PORTAL_CONDITION:
        portal = _ensure_portal()
        _PORTAL_CALLS += 1
    try:
        ctx = contextvars.copy_context()
        kwargs_copy = dict(kwargs)
        return cast(_ResultT, portal.call(_call_with_context, ctx, async_fn, args, kwargs_copy))
    finally:
        with _PORTAL_CONDITION:
            _PORTAL_CALLS -= 1
            _PORTAL_CONDITION.notify_all()


async def _call_with_context(
    ctx: contextvars.Context,
    async_fn: Callable[..., Awaitable[_ResultT]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> _ResultT:
    token = _PORTAL_CALL_CONTEXT.set(True)
    try:
        coroutine: Awaitable[_ResultT] = ctx.run(_create_coroutine, async_fn, args, kwargs)
        return await coroutine
    finally:
        _PORTAL_CALL_CONTEXT.reset(token)


def _create_coroutine(
    async_fn: Callable[..., Awaitable[_ResultT]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Awaitable[_ResultT]:
    return async_fn(*args, **kwargs)
