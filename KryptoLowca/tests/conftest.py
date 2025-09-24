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
            for name in pyfuncitem._fixtureinfo.argnames:
                if name not in pyfuncitem.funcargs:
                    continue
                value = pyfuncitem.funcargs[name]
                if inspect.iscoroutine(value):
                    value = await value
                resolved[name] = value
            return resolved

        async def _invoke():
            kwargs = await _resolve_args()
            await pyfuncitem.obj(**kwargs)

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
