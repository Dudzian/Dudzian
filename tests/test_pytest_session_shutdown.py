"""Prove that the real session finalizers permit the pytest process to exit."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("test_fails", [False, True], ids=["passing", "failing"])
def test_pytest_exits_after_database_background_loop(tmp_path: Path, test_fails: bool) -> None:
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
    assert completed.returncode == (1 if test_fails else 0), completed.stdout + completed.stderr
    assert ("1 failed" if test_fails else "1 passed") in completed.stdout


def test_session_cleanup_failure_is_not_suppressed(monkeypatch: pytest.MonkeyPatch) -> None:
    from bot_core.database import DatabaseManager

    source = Path(__file__).resolve().with_name("conftest.py")
    spec = importlib.util.spec_from_file_location("repository_cleanup_failure", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def failed_close(**kwargs: object) -> None:
        raise RuntimeError("synthetic cleanup failure")

    monkeypatch.setattr(DatabaseManager, "close_all_active", failed_close)
    cleanup = module.shutdown_db_background_loop_at_session_end.__wrapped__()
    next(cleanup)
    with pytest.raises(RuntimeError, match="synthetic cleanup failure"):
        next(cleanup)
