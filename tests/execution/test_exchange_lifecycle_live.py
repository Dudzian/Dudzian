import types

import pytest

from bot_core.config.loader import load_core_config
from bot_core.config.models import RuntimeExecutionLiveSettings, RuntimeExecutionSettings
from bot_core.execution.execution_service import build_live_execution_service
from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)


class _DummyAdapter(ExchangeAdapter):
    def __init__(self, name: str) -> None:
        super().__init__(ExchangeCredentials(key_id=name))
        self.name = name

    def configure_network(self, *, ip_allowlist=None) -> None:
        return None

    def fetch_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            balances={}, total_equity=0.0, available_margin=0.0, maintenance_margin=0.0
        )

    def fetch_symbols(self):
        return []

    def fetch_ohlcv(self, symbol, interval, start=None, end=None, limit=None):
        return []

    def place_order(self, request: OrderRequest) -> OrderResult:
        return OrderResult(
            order_id="1",
            status="filled",
            filled_quantity=request.quantity,
            avg_price=request.price,
            raw_response={},
        )

    def cancel_order(self, order_id: str, *, symbol=None) -> None:
        return None

    def stream_public_data(self, *, channels):
        return types.SimpleNamespace()

    def stream_private_data(self, *, channels):
        return types.SimpleNamespace()


class _EnvLive:
    exchange = "binance_spot"
    environment = Environment.LIVE


class _Bootstrap:
    def __init__(self, adapters, lifecycle):
        self.adapters = adapters
        self.core_config = types.SimpleNamespace(exchange_lifecycle=lifecycle)


def test_disabled_exchange_is_rejected_before_order_execution() -> None:
    settings = RuntimeExecutionSettings(
        live=RuntimeExecutionLiveSettings(enabled=True, default_route=("zonda_spot",))
    )
    bootstrap = _Bootstrap(
        adapters={"zonda_spot": _DummyAdapter("zonda_spot")},
        lifecycle={"zonda_spot": types.SimpleNamespace(status="disabled")},
    )

    with pytest.raises(RuntimeError, match="disabled/deprecated"):
        build_live_execution_service(
            bootstrap_ctx=bootstrap,
            environment=_EnvLive(),
            runtime_settings=settings,
        )


def test_deprecated_exchange_is_rejected_for_live_default_route() -> None:
    settings = RuntimeExecutionSettings(
        live=RuntimeExecutionLiveSettings(enabled=True, default_route=("zonda_spot",))
    )
    bootstrap = _Bootstrap(
        adapters={"zonda_spot": _DummyAdapter("zonda_spot")},
        lifecycle={"zonda_spot": types.SimpleNamespace(status="deprecated")},
    )

    with pytest.raises(RuntimeError, match="disabled/deprecated"):
        build_live_execution_service(
            bootstrap_ctx=bootstrap,
            environment=_EnvLive(),
            runtime_settings=settings,
        )


def test_disabled_exchange_cannot_be_selected_as_failover_venue() -> None:
    settings = RuntimeExecutionSettings(
        live=RuntimeExecutionLiveSettings(
            enabled=True,
            default_route=("binance_spot",),
            route_overrides={"BTCUSDT": ("binance_spot", "zonda_spot")},
        )
    )
    bootstrap = _Bootstrap(
        adapters={
            "binance_spot": _DummyAdapter("binance_spot"),
            "zonda_spot": _DummyAdapter("zonda_spot"),
        },
        lifecycle={"zonda_spot": types.SimpleNamespace(status="disabled")},
    )

    with pytest.raises(RuntimeError, match="disabled/deprecated"):
        build_live_execution_service(
            bootstrap_ctx=bootstrap,
            environment=_EnvLive(),
            runtime_settings=settings,
        )


def test_deprecated_exchange_cannot_be_selected_as_failover_venue() -> None:
    settings = RuntimeExecutionSettings(
        live=RuntimeExecutionLiveSettings(
            enabled=True,
            default_route=("binance_spot",),
            route_overrides={"BTCUSDT": ("binance_spot", "zonda_spot")},
        )
    )
    bootstrap = _Bootstrap(
        adapters={
            "binance_spot": _DummyAdapter("binance_spot"),
            "zonda_spot": _DummyAdapter("zonda_spot"),
        },
        lifecycle={"zonda_spot": types.SimpleNamespace(status="deprecated")},
    )

    with pytest.raises(RuntimeError, match="disabled/deprecated|niedozwolone"):
        build_live_execution_service(
            bootstrap_ctx=bootstrap,
            environment=_EnvLive(),
            runtime_settings=settings,
        )


def test_invalid_lifecycle_status_is_fail_closed() -> None:
    settings = RuntimeExecutionSettings(
        live=RuntimeExecutionLiveSettings(enabled=True, default_route=("binance_spot",))
    )
    bootstrap = _Bootstrap(
        adapters={"binance_spot": _DummyAdapter("binance_spot")},
        lifecycle={"zonda_spot": types.SimpleNamespace(status="disbaled")},
    )

    with pytest.raises(RuntimeError, match="Nieznany status lifecycle"):
        build_live_execution_service(
            bootstrap_ctx=bootstrap,
            environment=_EnvLive(),
            runtime_settings=settings,
        )


def test_zonda_reported_as_disabled_in_exchange_registry() -> None:
    settings = RuntimeExecutionSettings(
        live=RuntimeExecutionLiveSettings(enabled=True, default_route=("binance_spot",))
    )
    lifecycle = {"zonda_spot": types.SimpleNamespace(status="disabled")}
    bootstrap = _Bootstrap(
        adapters={"binance_spot": _DummyAdapter("binance_spot")}, lifecycle=lifecycle
    )

    router = build_live_execution_service(
        bootstrap_ctx=bootstrap,
        environment=_EnvLive(),
        runtime_settings=settings,
    )

    # Snapshot lifecycle jest udostępniany na routerze do introspekcji matrix/registry.
    assert getattr(router, "exchange_lifecycle", {}).get("zonda_spot") == "disabled"


def test_loader_reads_exchange_lifecycle_from_real_core_config() -> None:
    config = load_core_config("config/core.yaml")

    assert config.exchange_lifecycle["zonda_spot"].status == "disabled"


def test_loader_rejects_invalid_exchange_lifecycle_status(tmp_path) -> None:
    config_file = tmp_path / "invalid_lifecycle.yaml"
    config_file.write_text(
        """
environments: {}
risk_profiles: {}
exchange_lifecycle:
  zonda_spot:
    status: disbaled
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Nieznany status lifecycle"):
        load_core_config(config_file)
