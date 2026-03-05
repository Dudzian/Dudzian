from __future__ import annotations

import asyncio
import logging
import os
import socket
import threading
import sys
from pathlib import Path
from contextlib import contextmanager
from typing import Generator, cast

import pytest

from bot_core.backtest.simulation import SimulationScenario

logger = logging.getLogger(__name__)

# Testy/CI często działają w VM (np. Hyper-V na runnerach), a build_capability_guard
# blokuje VM i rzuca FingerprintError. W testach nie weryfikujemy anty-tamper,
# tylko logikę licencji/workflow.
os.environ.setdefault("DUDZIAN_SECURITY_SKIP", "1")

if sys.platform == "win32":
    os.environ.setdefault("BOT_CORE_DISABLE_DB_BACKGROUND_LOOP", "1")

UNSTABLE_WINDOWS_REASON = (
    "Niestabilny test na Windows (self-hosted runner); pomijany na win32, wykonywany na "
    "pozostałych platformach."
)


def is_windows() -> bool:
    return sys.platform.startswith("win")


unstable_windows = pytest.mark.skipif(
    is_windows(),
    reason=UNSTABLE_WINDOWS_REASON,
)

pytest.mark.unstable_windows = unstable_windows  # type: ignore[attr-defined]

_UI_QML_DIRS = pytest.StashKey[tuple[Path, Path]]()


def _force_windows_selector_event_loop_policy() -> None:
    if sys.platform != "win32":
        return
    try:
        policy = asyncio.WindowsSelectorEventLoopPolicy()
    except Exception:
        logger.debug(
            "teardown/windows_selector_policy_failed: unable to create WindowsSelectorEventLoopPolicy.",
            exc_info=True,
        )
        return
    try:
        asyncio.set_event_loop_policy(policy)
    except Exception:
        logger.debug(
            "teardown/windows_selector_policy_failed: unable to set WindowsSelectorEventLoopPolicy.",
            exc_info=True,
        )
        return


_force_windows_selector_event_loop_policy()


@pytest.fixture(scope="session", autouse=True)
def enforce_windows_selector_event_loop_policy() -> Generator[None, None, None]:
    if sys.platform != "win32":
        yield
        return
    _force_windows_selector_event_loop_policy()
    yield


@pytest.fixture(scope="session", autouse=True)
def disable_db_background_loop_on_windows() -> Generator[None, None, None]:
    if sys.platform == "win32":
        logger.debug(
            "teardown/db_manager_background_loop_disabled: env=%s (platform=%s pid=%s threads=%s)",
            os.environ.get("BOT_CORE_DISABLE_DB_BACKGROUND_LOOP"),
            sys.platform,
            os.getpid(),
            [thread.name for thread in threading.enumerate()],
        )
    yield


@pytest.fixture(autouse=True)
def windows_per_test_db_cleanup() -> Generator[None, None, None]:
    yield
    if sys.platform != "win32":
        return
    anyio_loaded = "anyio" in sys.modules
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    logger.debug(
        "teardown/db_manager_per_test_begin: "
        "platform=%s pid=%s thread=%s ident=%s anyio_loaded=%s running_loop=%s",
        sys.platform,
        os.getpid(),
        threading.current_thread().name,
        threading.current_thread().ident,
        anyio_loaded,
        bool(running_loop and running_loop.is_running()),
    )
    try:
        from bot_core.database.manager import DatabaseManager
    except Exception:
        logger.debug(
            "teardown/db_manager_per_test_import_failed: "
            "DatabaseManager per-test cleanup import failed. (platform=%s pid=%s)",
            sys.platform,
            os.getpid(),
            exc_info=True,
        )
        return
    try:
        DatabaseManager.close_all_active(blocking=False, timeout=2.0)
        logger.debug(
            "teardown/db_manager_per_test_done: DatabaseManager per-test cleanup done. "
            "(platform=%s pid=%s thread=%s ident=%s anyio_loaded=%s running_loop=%s)",
            sys.platform,
            os.getpid(),
            threading.current_thread().name,
            threading.current_thread().ident,
            anyio_loaded,
            bool(running_loop and running_loop.is_running()),
        )
    except Exception:
        logger.debug(
            "teardown/db_manager_per_test_failed: DatabaseManager per-test cleanup failed. "
            "(platform=%s pid=%s thread=%s ident=%s anyio_loaded=%s running_loop=%s)",
            sys.platform,
            os.getpid(),
            threading.current_thread().name,
            threading.current_thread().ident,
            anyio_loaded,
            bool(running_loop and running_loop.is_running()),
            exc_info=True,
        )


@pytest.fixture(scope="session", autouse=True)
def windows_session_db_cleanup() -> Generator[None, None, None]:
    yield
    if sys.platform != "win32":
        return
    anyio_loaded = "anyio" in sys.modules
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    running_loop_active = bool(running_loop and running_loop.is_running())
    if running_loop_active:
        logger.debug(
            "teardown/db_manager_session_cleanup_skipped_running_loop: "
            "platform=%s pid=%s thread=%s ident=%s anyio_loaded=%s running_loop=%s",
            sys.platform,
            os.getpid(),
            threading.current_thread().name,
            threading.current_thread().ident,
            anyio_loaded,
            running_loop_active,
        )
        try:
            from bot_core.database.manager import DatabaseManager
        except Exception:
            logger.debug(
                "teardown/db_manager_session_cleanup_skipped_running_loop_failed: "
                "session cleanup import failed. "
                "(platform=%s pid=%s)",
                sys.platform,
                os.getpid(),
                exc_info=True,
            )
            return
        try:
            DatabaseManager.close_all_active(blocking=False, timeout=2.0)
        except Exception:
            logger.debug(
                "teardown/db_manager_session_cleanup_skipped_running_loop_failed: "
                "session cleanup failed. "
                "(platform=%s pid=%s thread=%s ident=%s anyio_loaded=%s running_loop=%s)",
                sys.platform,
                os.getpid(),
                threading.current_thread().name,
                threading.current_thread().ident,
                anyio_loaded,
                running_loop_active,
                exc_info=True,
            )
        return
    logger.debug(
        "teardown/db_manager_session_cleanup_begin: "
        "platform=%s pid=%s thread=%s ident=%s anyio_loaded=%s running_loop=%s",
        sys.platform,
        os.getpid(),
        threading.current_thread().name,
        threading.current_thread().ident,
        anyio_loaded,
        running_loop_active,
    )
    try:
        from bot_core.database.manager import DatabaseManager
    except Exception:
        logger.debug(
            "teardown/db_manager_session_cleanup_failed: session cleanup import failed. "
            "(platform=%s pid=%s)",
            sys.platform,
            os.getpid(),
            exc_info=True,
        )
        return
    try:
        DatabaseManager.close_all_active(blocking=True, timeout=5.0)
        DatabaseManager.wait_for_aiosqlite_threads(timeout=5.0, poll_interval=0.05)
        logger.debug(
            "teardown/db_manager_session_cleanup_done: session cleanup done. "
            "(platform=%s pid=%s thread=%s ident=%s anyio_loaded=%s running_loop=%s)",
            sys.platform,
            os.getpid(),
            threading.current_thread().name,
            threading.current_thread().ident,
            anyio_loaded,
            running_loop_active,
        )
    except Exception:
        logger.debug(
            "teardown/db_manager_session_cleanup_failed: session cleanup failed. "
            "(platform=%s pid=%s thread=%s ident=%s anyio_loaded=%s running_loop=%s)",
            sys.platform,
            os.getpid(),
            threading.current_thread().name,
            threading.current_thread().ident,
            anyio_loaded,
            running_loop_active,
            exc_info=True,
        )


@pytest.fixture
def latency_one_no_cost_scenario() -> SimulationScenario:
    return SimulationScenario(
        name="latency_one_no_cost",
        latency_bars=1,
        slippage_bps=0.0,
        fee_bps=0.0,
        liquidity_share=1.0,
    )


@pytest.fixture
def instant_no_cost_scenario() -> SimulationScenario:
    return SimulationScenario(
        name="instant_no_cost",
        latency_bars=0,
        slippage_bps=0.0,
        fee_bps=0.0,
        liquidity_share=1.0,
    )


@pytest.fixture
def partial_fill_scenario() -> SimulationScenario:
    return SimulationScenario(
        name="partial_fill_half_liquidity",
        latency_bars=0,
        slippage_bps=0.0,
        fee_bps=0.0,
        liquidity_share=0.5,
    )


@pytest.fixture
def slippage_fee_scenario() -> SimulationScenario:
    return SimulationScenario(
        name="slippage_and_fee",
        latency_bars=0,
        slippage_bps=10.0,
        fee_bps=25.0,
        liquidity_share=1.0,
    )


@pytest.fixture
def heavy_slippage_scenario() -> SimulationScenario:
    return SimulationScenario(
        name="heavy_slippage_no_fee",
        latency_bars=0,
        slippage_bps=100.0,
        fee_bps=0.0,
        liquidity_share=1.0,
    )


@contextmanager
def _guard_network_connections() -> None:
    """Block outbound network during tests to keep the suite hermetic."""

    allowed_hosts = {"127.0.0.1", "::1", "localhost"}
    real_create_connection = socket.create_connection
    real_connect = socket.socket.connect

    def _is_unix_socket(address: tuple[object, ...] | object) -> bool:
        if isinstance(address, str):
            return address.startswith("/")
        if isinstance(address, tuple) and address and isinstance(address[0], str):
            return address[0].startswith("/")
        return False

    def _check_address(address: tuple[object, ...] | object) -> None:
        host = address[0] if isinstance(address, tuple) and address else address
        if _is_unix_socket(address):
            # Unix domain sockets are local-only; allow them.
            return
        if host is None:
            raise RuntimeError(
                "External network access is blocked during tests; mock with responses/respx "
                "or set ALLOW_NETWORK_TESTS=1 (host=None)."
            )
        if isinstance(host, str):
            if host in allowed_hosts or host.startswith("127."):
                return
            raise RuntimeError(
                "External network access is blocked during tests; mock with responses/respx "
                f"or set ALLOW_NETWORK_TESTS=1 (host={host!r})."
            )
        raise RuntimeError(
            "External network access is blocked during tests; mock with responses/respx "
            f"or set ALLOW_NETWORK_TESTS=1 (host={host!r})."
        )

    def _guarded_create_connection(address, *args, **kwargs):  # type: ignore[override]
        _check_address(address)
        return real_create_connection(address, *args, **kwargs)

    def _guarded_connect(self, address):  # type: ignore[override]
        _check_address(address)
        return real_connect(self, address)

    socket.create_connection = _guarded_create_connection  # type: ignore[assignment]
    socket.socket.connect = _guarded_connect  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.create_connection = real_create_connection  # type: ignore[assignment]
        socket.socket.connect = real_connect  # type: ignore[assignment]


@pytest.fixture(autouse=True)
def block_external_network() -> None:
    if os.environ.get("ALLOW_NETWORK_TESTS") == "1" or os.environ.get("RUN_NETWORK_TESTS") == "1":
        # Explicit opt-in for integration runs.
        yield
        return

    with _guard_network_connections():
        yield


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-network-tests",
        action="store_true",
        default=False,
        help="Run tests marked as requiring external network access",
    )
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help=(
            "Run a quick subset of tests (skips integration/qml/performance/"
            "soak/retraining/e2e/long_poll markers). "
            "You can also set PYTEST_FAST=1."
        ),
    )


def _resolve_item_path(item: pytest.Item) -> Path:
    return (getattr(item, "path", None) or Path(str(item.fspath))).resolve()


def _get_ui_qml_dirs(config: pytest.Config) -> tuple[Path, Path]:
    dirs = config.stash.get(_UI_QML_DIRS, None)
    if dirs is None:
        root = Path(str(config.rootpath)).resolve()
        dirs = (root / "tests/ui_pyside", root / "tests/ui_backend")
        config.stash[_UI_QML_DIRS] = dirs
    return cast(tuple[Path, Path], dirs)


def _is_ui_qml_test_path(config: pytest.Config, item_path: Path) -> bool:
    return any(item_path.is_relative_to(directory) for directory in _get_ui_qml_dirs(config))


@pytest.hookimpl(tryfirst=True)
def pytest_itemcollected(item: pytest.Item) -> None:
    item_path = _resolve_item_path(item)
    if _is_ui_qml_test_path(item.config, item_path):
        item.add_marker(pytest.mark.qml)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_network = (
        config.getoption("--run-network-tests")
        or os.environ.get("RUN_NETWORK_TESTS") == "1"
        or os.environ.get("ALLOW_NETWORK_TESTS") == "1"
    )
    if not run_network:
        skip_network = pytest.mark.skip(
            reason="Network tests require RUN_NETWORK_TESTS=1 or --run-network-tests",
        )
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_network)

    fast_mode = bool(config.getoption("--fast")) or os.environ.get("PYTEST_FAST") == "1"
    if not fast_mode:
        return

    skip_fast = pytest.mark.skip(reason="Skipped in --fast / PYTEST_FAST=1 mode")
    fast_excluded = {
        "integration",
        "qml",
        "performance",
        "soak",
        "retraining",
        "e2e_demo_paper",
        "e2e_retraining",
        "long_poll",
    }
    for item in items:
        if any(marker in item.keywords for marker in fast_excluded):
            item.add_marker(skip_fast)


@pytest.fixture(autouse=True, scope="session")
def shutdown_background_components() -> None:
    yield
    try:
        from bot_core.exchanges.streaming import LocalLongPollStream
        from bot_core.execution.live_router import LiveExecutionRouter
    except Exception:
        logger.debug(
            "teardown/background_components_import_failed: "
            "Background component shutdown skipped. (platform=%s pid=%s)",
            sys.platform,
            os.getpid(),
            exc_info=True,
        )
        return
    LocalLongPollStream.close_all_active()
    LiveExecutionRouter.close_all_active()


@pytest.fixture(scope="session", autouse=True)
def shutdown_db_background_loop_at_session_end() -> Generator[None, None, None]:
    yield
    if sys.platform != "win32":
        return
    logger.debug(
        "teardown/db_manager_shutdown_begin: "
        "DatabaseManager session shutdown begin. (platform=%s pid=%s)",
        sys.platform,
        os.getpid(),
    )
    try:
        from bot_core.database.manager import DatabaseManager
    except Exception:
        logger.debug(
            "teardown/db_manager_import_failed: "
            "Skipping DatabaseManager session shutdown fixture (import failed). "
            "(platform=%s pid=%s)",
            sys.platform,
            os.getpid(),
            exc_info=True,
        )
        return

    try:
        DatabaseManager.close_all_active(blocking=True, timeout=5.0)
        DatabaseManager.wait_for_aiosqlite_threads(timeout=5.0, poll_interval=0.05)
        DatabaseManager.shutdown_background_loop(timeout=5.0)
        logger.debug(
            "teardown/db_manager_shutdown_done: "
            "DatabaseManager session shutdown done. (platform=%s pid=%s)",
            sys.platform,
            os.getpid(),
        )
    except Exception:
        logger.debug(
            "teardown/db_manager_shutdown_failed: "
            "DatabaseManager session shutdown failed. (platform=%s pid=%s)",
            sys.platform,
            os.getpid(),
            exc_info=True,
        )
