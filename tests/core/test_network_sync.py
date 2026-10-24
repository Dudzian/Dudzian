from __future__ import annotations

import contextvars
import json
import subprocess
import sys
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anyio
import pytest

from core.network import sync


@pytest.fixture(autouse=True)
def close_portal():
    sync._shutdown_portal()
    yield
    sync._shutdown_portal()


def test_run_sync_preserves_context_arguments_and_propagates_failure() -> None:
    value = contextvars.ContextVar("network_sync_value", default="missing")
    value.set("caller")

    async def read(number: int, *, increment: int) -> tuple[str, int]:
        return value.get(), number + increment

    async def fail() -> None:
        raise ValueError("request failed")

    assert sync.run_sync(read, 2, increment=3) == ("caller", 5)
    with pytest.raises(ValueError, match="request failed"):
        sync.run_sync(fail)
    assert sync._PORTAL_CALLS == 0
    assert sync.run_sync(read, 4, increment=1) == ("caller", 5)


def test_portal_can_be_initialized_and_closed_repeatedly() -> None:
    async def identify_thread() -> threading.Thread:
        return threading.current_thread()

    for _ in range(3):
        portal_thread = sync.run_sync(identify_thread)
        assert portal_thread.is_alive()
        sync._shutdown_portal()
        sync._shutdown_portal()
        assert not portal_thread.is_alive()
        assert sync._PORTAL is None
        assert sync._PORTAL_CM is None


def test_lifecycle_calls_from_the_portal_thread_fail_without_deadlocking() -> None:
    async def answer() -> int:
        return 42

    async def misuse() -> None:
        with pytest.raises(RuntimeError, match="active portal call"):
            sync._shutdown_portal()
        with pytest.raises(RuntimeError, match="event loop thread"):
            sync.run_sync(answer)

    sync.run_sync(misuse)
    assert sync.run_sync(answer) == 42


def test_shutdown_waits_for_a_call_that_has_not_yet_been_submitted(monkeypatch) -> None:
    async def answer() -> int:
        return 42

    assert sync.run_sync(answer) == 42
    portal = sync._PORTAL
    assert portal is not None
    original_call = portal.call
    acquired = threading.Event()
    submit = threading.Event()

    def delayed_call(*args, **kwargs):
        acquired.set()
        assert submit.wait(5)
        return original_call(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(portal, "call", delayed_call)
        with ThreadPoolExecutor(max_workers=2) as executor:
            request = executor.submit(sync.run_sync, answer)
            assert acquired.wait(5)
            shutdown = executor.submit(sync._shutdown_portal)
            with sync._PORTAL_CONDITION:
                assert sync._PORTAL_CONDITION.wait_for(
                    lambda: sync._PORTAL_SHUTTING_DOWN, timeout=5
                )
            assert not shutdown.done()
            submit.set()
            assert request.result(timeout=5) == 42
            shutdown.result(timeout=5)
    assert sync._PORTAL is None


def test_shutdown_drains_active_calls_without_cancelling_them() -> None:
    started = threading.Event()
    finish = threading.Event()

    async def request() -> str:
        started.set()
        while not finish.is_set():
            await anyio.sleep(0.01)
        return "completed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        result = executor.submit(sync.run_sync, request)
        assert started.wait(5)
        shutdown = executor.submit(sync._shutdown_portal)
        with sync._PORTAL_CONDITION:
            assert sync._PORTAL_CONDITION.wait_for(lambda: sync._PORTAL_SHUTTING_DOWN, 5)
        assert not shutdown.done()
        finish.set()
        assert result.result(timeout=5) == "completed"
        shutdown.result(timeout=5)
    assert sync._PORTAL is None


def test_new_calls_are_rejected_until_concurrent_shutdown_finishes(monkeypatch) -> None:
    original_start = sync.start_blocking_portal
    closing = threading.Event()
    finish_close = threading.Event()

    class DelayedClose:
        def __init__(self):
            self.cm = original_start()

        def __enter__(self):
            return self.cm.__enter__()

        def __exit__(self, *args):
            closing.set()
            assert finish_close.wait(5)
            return self.cm.__exit__(*args)

    monkeypatch.setattr(sync, "start_blocking_portal", DelayedClose)

    async def identify_thread() -> threading.Thread:
        return threading.current_thread()

    old_thread = sync.run_sync(identify_thread)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first_shutdown = executor.submit(sync._shutdown_portal)
        assert closing.wait(5)
        second_shutdown = executor.submit(sync._shutdown_portal)
        assert not first_shutdown.done()
        assert not second_shutdown.done()
        with pytest.raises(RuntimeError, match="shutting down"):
            sync.run_sync(identify_thread)
        finish_close.set()
        first_shutdown.result(timeout=5)
        second_shutdown.result(timeout=5)
    new_thread = sync.run_sync(identify_thread)
    assert not old_thread.is_alive()
    assert new_thread is not old_thread


@pytest.mark.parametrize("shutdown_from_worker", [False, True])
def test_worker_reentry_during_shutdown_fails_without_deadlocking(
    shutdown_from_worker: bool,
) -> None:
    code = textwrap.dedent(
        f"""
        import faulthandler
        import threading
        from concurrent.futures import ThreadPoolExecutor
        import anyio
        from core.network import sync

        faulthandler.dump_traceback_later(15, exit=True)
        started = threading.Event()
        proceed = threading.Event()
        async def inner():
            return 42
        def worker():
            started.set()
            assert proceed.wait(5)
            try:
                if {shutdown_from_worker!r}:
                    sync._shutdown_portal()
                else:
                    sync.run_sync(inner)
            except RuntimeError:
                return "rejected"
            raise AssertionError("recursive operation was accepted during shutdown")
        async def outer():
            return await anyio.to_thread.run_sync(worker)
        with ThreadPoolExecutor(max_workers=2) as pool:
            call = pool.submit(sync.run_sync, outer)
            assert started.wait(5)
            shutdown = pool.submit(sync._shutdown_portal)
            with sync._PORTAL_CONDITION:
                assert sync._PORTAL_CONDITION.wait_for(
                    lambda: sync._PORTAL_SHUTTING_DOWN, 5
                )
            proceed.set()
            assert call.result(5) == "rejected"
            shutdown.result(5)
        assert sync.run_sync(inner) == 42
        sync._shutdown_portal()
        faulthandler.cancel_dump_traceback_later()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""


@pytest.mark.parametrize("http_first", [True, False])
@pytest.mark.parametrize("restart", [True, False])
def test_process_exit_closes_http_clients_before_portal(http_first: bool, restart: bool) -> None:
    code = textwrap.dedent(
        f"""
        import atexit
        import json
        import threading

        def inspect_cleanup():
            print(json.dumps({{
                "client_closed": client.client.is_closed,
                "portal_closed": sync._PORTAL is None,
                "portal_thread_closed": not portal_thread.is_alive(),
            }}), flush=True)
            # Keep a failing regression bounded after recording the leak.
            sync._shutdown_portal()
        # Observe after both real cleanup callbacks, before module finalization.
        atexit.register(inspect_cleanup)
        if {http_first!r}:
            from bot_core.exchanges import http_client
            from core.network import sync
        else:
            from core.network import sync
            from bot_core.exchanges import http_client

        client = http_client._resolve_client("https://example.test", 1.0)
        async def identify_thread():
            return threading.current_thread()
        portal_thread = sync.run_sync(identify_thread)
        if {restart!r}:
            sync._shutdown_portal()
            assert not portal_thread.is_alive()
            portal_thread = sync.run_sync(identify_thread)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert json.loads(result.stdout) == {
        "client_closed": True,
        "portal_closed": True,
        "portal_thread_closed": True,
    }
