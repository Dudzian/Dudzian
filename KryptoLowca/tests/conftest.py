from __future__ import annotations

import sys
from pathlib import Path
import asyncio
import inspect
from contextlib import suppress
from typing import Any, Callable, Dict, Tuple

import pytest

# Dostosuj sys.path, aby testy mogły importować pakiet i root projektu
ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(PACKAGE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _run_coroutine(coro: Any) -> Any:
    """Run *coro* in a temporary event loop and return its result."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        with suppress(Exception):
            loop.run_until_complete(loop.shutdown_asyncgens())
        asyncio.set_event_loop(None)
        loop.close()


def _resolve_fixture_arguments(fixturedef, request) -> Dict[str, Any]:
    """Collect already-evaluated dependency fixtures for ``fixturedef``."""
    kwargs: Dict[str, Any] = {}
    for name in getattr(fixturedef, "argnames", ()):
        kwargs[name] = request.getfixturevalue(name)
    return kwargs


@pytest.hookimpl(tryfirst=True)
def pytest_addoption(parser):
    """Register legacy asyncio config option expected by the test-suite."""
    parser.addini(
        "asyncio_mode",
        "Compatibility placeholder so pytest accepts the asyncio_mode option",
        default="auto",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "asyncio: test or fixture using asyncio without the pytest-asyncio plugin",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(fixturedef, request):
    """Pozwala na używanie async fixture’ów bez pytest-asyncio."""
    func = fixturedef.func
    if inspect.iscoroutinefunction(func):

        async def _call() -> Any:
            kwargs = _resolve_fixture_arguments(fixturedef, request)
            return await func(**kwargs)

        result = _run_coroutine(_call())
        fixturedef.cached_result = (result, fixturedef.cache_key(request), None)
        return result

    if inspect.isasyncgenfunction(func):

        async def _call_gen() -> Tuple[Any, Any]:
            kwargs = _resolve_fixture_arguments(fixturedef, request)
            agen = func(**kwargs)
            value = await agen.__anext__()
            return value, agen

        value, agen = _run_coroutine(_call_gen())

        def _finalizer() -> None:
            async def _close() -> None:
                with suppress(StopAsyncIteration):
                    await agen.__anext__()
                await agen.aclose()

            _run_coroutine(_close())

        request.addfinalizer(_finalizer)
        fixturedef.cached_result = (value, fixturedef.cache_key(request), None)
        return value


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """Uruchamiaj funkcje testowe będące coroutine bez zewnętrznego pluginu."""
    if inspect.iscoroutinefunction(pyfuncitem.obj):

        async def _resolve_args():
            resolved: Dict[str, Any] = {}
            closers: list[Any] = []
            for name in pyfuncitem._fixtureinfo.argnames:
                if name not in pyfuncitem.funcargs:
                    continue
                value = pyfuncitem.funcargs[name]
                if inspect.iscoroutine(value):
                    value = await value
                elif inspect.isasyncgen(value):
                    agen = value
                    # pobierz pierwszy (i jedyny) element z async generatora
                    value = await agen.__anext__()

                    async def _close(gen):
                        try:
                            await gen.aclose()
                        except Exception:
                            # nie przerywaj zamykania przy błędach cleanupu
                            pass

                    closers.append(_close(agen))
                resolved[name] = value
            return resolved, closers

        async def _invoke():
            kwargs, closers = await _resolve_args()
            try:
                await pyfuncitem.obj(**kwargs)
            finally:
                for closer in reversed(closers):
                    # closers są coroutine – trzeba je odczekać
                    await closer

        # własna pętla event loop dla testu
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_invoke())
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                asyncio.set_event_loop(None)
                loop.close()
        return True
