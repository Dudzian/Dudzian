from __future__ import annotations

import sys
from pathlib import Path
import asyncio
import inspect
import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(PACKAGE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.hookimpl
def pytest_pyfunc_call(pyfuncitem):
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        async def _resolve_args():
            resolved = {}
            closers: list = []
            for name in pyfuncitem._fixtureinfo.argnames:
                if name not in pyfuncitem.funcargs:
                    continue
                value = pyfuncitem.funcargs[name]
                if inspect.iscoroutine(value):
                    value = await value
                elif inspect.isasyncgen(value):
                    agen = value
                    value = await agen.__anext__()

                    async def _close(gen):
                        await gen.aclose()

                    closers.append(_close(agen))
                resolved[name] = value
            return resolved, closers

        async def _invoke():
            kwargs, closers = await _resolve_args()
            try:
                await pyfuncitem.obj(**kwargs)
            finally:
                for closer in reversed(closers):
                    await closer

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
