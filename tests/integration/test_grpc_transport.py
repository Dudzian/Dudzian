from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from bot_core.api.server import LocalRuntimeContext, LocalRuntimeGateway
from bot_core.exchanges import streaming as core_streaming
from bot_core.exchanges.base import AccountSnapshot
from bot_core.execution.paper import MarketMetadata
from KryptoLowca.exchanges import interfaces as exchange_interfaces
from KryptoLowca.exchanges import streaming as exchange_streaming


def _build_stub_context():
    markets = {
        "BTC/USDT": MarketMetadata(
            base_asset="BTC",
            quote_asset="USDT",
            min_quantity=0.001,
            min_notional=10.0,
            tick_size=0.1,
            step_size=0.001,
        )
    }

    class _DummyExecutionService:
        def __init__(self) -> None:
            self._markets = markets

        def execute(self, *_args, **_kwargs):
            return SimpleNamespace(order_id="demo-order", raw_response={"exchange_order_id": "demo"})

        def cancel(self, *_args, **_kwargs) -> None:  # pragma: no cover - not used in test
            return None

    class _DummyController:
        def __init__(self) -> None:
            self.symbols = ["BTC/USDT"]
            self.interval = "1h"
            self.execution_context = SimpleNamespace()

        def account_loader(self) -> AccountSnapshot:
            return AccountSnapshot(
                balances={"USDT": 1000.0},
                total_equity=1000.0,
                available_margin=1000.0,
                maintenance_margin=0.0,
            )

    class _DummyDataSource:
        def fetch_ohlcv(self, request):
            rows = [
                [request.start + index * 60_000, 100.0, 101.0, 99.5, 100.5, 1.0]
                for index in range(5)
            ]
            return SimpleNamespace(columns=("timestamp", "open", "high", "low", "close", "volume"), rows=rows)

    class _DummyAutoTrader:
        def configure_controller_runner(self, *_args, **_kwargs) -> None:
            return None

        def start(self) -> None:  # pragma: no cover - no-op
            return None

        def confirm_auto_trade(self, *_args, **_kwargs) -> None:  # pragma: no cover - no-op
            return None

        def stop(self) -> None:  # pragma: no cover - no-op
            return None

    pipeline = SimpleNamespace(
        execution_service=_DummyExecutionService(),
        controller=_DummyController(),
        bootstrap=SimpleNamespace(environment=SimpleNamespace(exchange="paper")),
        data_source=_DummyDataSource(),
    )

    context = LocalRuntimeContext(
        config=SimpleNamespace(trading=SimpleNamespace(), core=SimpleNamespace()),
        entrypoint=SimpleNamespace(trusted_auto_confirm=False),
        config_path=Path("."),
        pipeline=pipeline,
        trading_controller=pipeline.controller,
        runner=SimpleNamespace(),
        auto_trader=_DummyAutoTrader(),
        secret_manager=SimpleNamespace(),
    )
    context.metrics_registry = None
    context.risk_store = None
    context.risk_builder = None
    context.risk_publisher = None
    context.marketplace_repository = None
    context.marketplace_enabled = False
    return context


@pytest.mark.integration
def test_local_runtime_gateway_exposes_health_data() -> None:
    context = _build_stub_context()
    gateway = LocalRuntimeGateway(context)
    payload = gateway.dispatch("health.check", {})
    assert payload["version"]
    assert int(payload["started_at_ms"]) > 0


def test_local_runtime_gateway_serves_market_data() -> None:
    context = _build_stub_context()
    gateway = LocalRuntimeGateway(context)
    response = gateway.dispatch(
        "market_data.get_ohlcv_history",
        {"symbol": "BTC/USDT", "limit": 3},
    )
    assert len(response["candles"]) == 3
    assert response["candles"][0]["close"] > 0


def test_ui_transport_configuration_defaults_to_grpc() -> None:
    config_path = Path("ui/config/example.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["transport"]["mode"] == "grpc"
    serialized = config_path.read_text(encoding="utf-8").lower()
    assert "websocket" not in serialized


def test_streaming_layer_exposes_long_poll_only() -> None:
    exported = {name for name in dir(exchange_interfaces) if "WebSocket" in name}
    assert not exported
    assert hasattr(exchange_interfaces, "MarketStreamHandle")
    assert "websocket" not in (exchange_streaming.LongPollSubscription.__doc__ or "").lower()
    assert not hasattr(core_streaming, "LocalWebSocketBridge")
    assert not hasattr(core_streaming.LocalLongPollStream, "websocket_bridge")
