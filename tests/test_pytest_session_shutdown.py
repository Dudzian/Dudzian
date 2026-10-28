"""Prove that the real session finalizers permit the pytest process to exit."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock, call

import pytest


@pytest.mark.parametrize(
    ("test_fails", "wait_times_out"),
    [(False, False), (True, False), (False, True)],
    ids=["passing", "failing", "teardown-wait-timeout"],
)
def test_pytest_exits_after_database_background_loop(
    tmp_path: Path, test_fails: bool, wait_times_out: bool
) -> None:
    repository = Path(__file__).resolve().parents[1]
    fixture_source = repository / "tests" / "conftest.py"
    (tmp_path / "conftest.py").write_text(
        "import importlib.util\n"
        "spec = importlib.util.spec_from_file_location(\n"
        f"    'repository_cleanup', {str(fixture_source)!r}\n"
        ")\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "shutdown_background_components = module.shutdown_background_components\n"
        "shutdown_db_background_loop_at_session_end = (\n"
        "    module.shutdown_db_background_loop_at_session_end\n"
        ")\n",
        encoding="utf-8",
    )
    (tmp_path / "test_owned_background_loop.py").write_text(
        "import sys\n"
        "from bot_core.database import manager\n"
        "from bot_core.exchanges.streaming import LocalLongPollStream\n"
        "def test_owned_background_loop():\n"
        "    loop = manager._ensure_background_loop()\n"
        "    if sys.platform != 'win32':\n"
        "        assert loop is not None and loop.is_running()\n"
        "        assert manager._background_thread.daemon is False\n"
        # The database must also outlive dependent component finalizers.
        "    LocalLongPollStream.close_all_active = staticmethod(manager._ensure_background_loop)\n"
        f"    if {wait_times_out!r}:\n"
        "        manager.DatabaseManager.wait_for_aiosqlite_threads = (\n"
        "            staticmethod(lambda **kwargs: False)\n"
        "        )\n"
        f"    assert {not test_fails!r}, 'intentional child-test failure'\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.update(
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        DUDZIAN_TEST_MODE="1",
        DUDZIAN_ALLOW_LONG_POLL="0",
        BOT_CORE_DISABLE_DB_BACKGROUND_LOOP="0",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "asyncio",
            f"--confcutdir={tmp_path}",
            "-o",
            "addopts=",
            str(tmp_path / "test_owned_background_loop.py"),
        ],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == (1 if test_fails or wait_times_out else 0), (
        completed.stdout + completed.stderr
    )
    assert ("1 failed" if test_fails else "1 passed") in completed.stdout
    if wait_times_out:
        assert "1 error" in completed.stdout
        assert "aiosqlite threads did not stop within 5 seconds" in completed.stdout


@pytest.fixture
def session_cleanup() -> Iterator[None]:
    source = Path(__file__).resolve().with_name("conftest.py")
    spec = importlib.util.spec_from_file_location("repository_cleanup_result", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cleanup = module.shutdown_db_background_loop_at_session_end.__wrapped__()
    next(cleanup)
    return cleanup


@pytest.fixture
def cleanup_calls(monkeypatch: pytest.MonkeyPatch) -> Mock:
    from bot_core.database import DatabaseManager

    calls = Mock()
    calls.wait_for_aiosqlite_threads.return_value = True
    for name in (
        "close_all_active",
        "wait_for_aiosqlite_threads",
        "shutdown_background_loop",
    ):
        monkeypatch.setattr(DatabaseManager, name, getattr(calls, name))
    return calls


def test_session_cleanup_succeeds_when_threads_stop(
    session_cleanup: Iterator[None], cleanup_calls: Mock
) -> None:
    with pytest.raises(StopIteration):
        next(session_cleanup)
    assert cleanup_calls.mock_calls == [
        call.close_all_active(blocking=True, timeout=5.0),
        call.wait_for_aiosqlite_threads(timeout=5.0, poll_interval=0.05),
        call.shutdown_background_loop(timeout=5.0),
    ]


def test_session_cleanup_wait_timeout_fails_and_still_shuts_down_loop(
    session_cleanup: Iterator[None], cleanup_calls: Mock
) -> None:
    cleanup_calls.wait_for_aiosqlite_threads.return_value = False
    with pytest.raises(RuntimeError, match="aiosqlite threads did not stop within 5 seconds"):
        next(session_cleanup)
    assert cleanup_calls.mock_calls == [
        call.close_all_active(blocking=True, timeout=5.0),
        call.wait_for_aiosqlite_threads(timeout=5.0, poll_interval=0.05),
        call.shutdown_background_loop(timeout=5.0),
    ]


@pytest.mark.parametrize(
    "failed_operation",
    ["close_all_active", "wait_for_aiosqlite_threads", "shutdown_background_loop"],
)
def test_session_cleanup_exception_is_not_suppressed(
    session_cleanup: Iterator[None], cleanup_calls: Mock, failed_operation: str
) -> None:
    failure = RuntimeError(f"synthetic {failed_operation} failure")
    getattr(cleanup_calls, failed_operation).side_effect = failure
    with pytest.raises(RuntimeError) as refused:
        next(session_cleanup)
    assert refused.value is failure
    cleanup_calls.shutdown_background_loop.assert_called_once_with(timeout=5.0)
    if failed_operation == "close_all_active":
        cleanup_calls.wait_for_aiosqlite_threads.assert_not_called()


@pytest.mark.parametrize("wait_raises", [False, True], ids=["wait-false", "wait-exception"])
def test_session_cleanup_preserves_wait_error_when_shutdown_also_fails(
    session_cleanup: Iterator[None], cleanup_calls: Mock, wait_raises: bool
) -> None:
    wait_failure = RuntimeError("synthetic wait failure")
    if wait_raises:
        cleanup_calls.wait_for_aiosqlite_threads.side_effect = wait_failure
    else:
        cleanup_calls.wait_for_aiosqlite_threads.return_value = False
    shutdown_failure = RuntimeError("synthetic shutdown failure")
    cleanup_calls.shutdown_background_loop.side_effect = shutdown_failure
    with pytest.raises(RuntimeError) as refused:
        next(session_cleanup)
    assert refused.value is shutdown_failure
    if wait_raises:
        assert refused.value.__context__ is wait_failure
    else:
        assert isinstance(refused.value.__context__, RuntimeError)
        assert "aiosqlite threads did not stop within 5 seconds" in str(refused.value.__context__)
    cleanup_calls.shutdown_background_loop.assert_called_once_with(timeout=5.0)
