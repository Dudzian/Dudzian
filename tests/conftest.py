import os
import socket
from contextlib import contextmanager

import pytest

from bot_core.backtest.simulation import SimulationScenario


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
    if os.environ.get("ALLOW_NETWORK_TESTS") == "1":
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


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_network = (
        config.getoption("--run-network-tests")
        or os.environ.get("RUN_NETWORK_TESTS") == "1"
        or os.environ.get("ALLOW_NETWORK_TESTS") == "1"
    )
    if run_network:
        return

    skip_network = pytest.mark.skip(
        reason="Network tests require RUN_NETWORK_TESTS=1 or --run-network-tests",
    )
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)
