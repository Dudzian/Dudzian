from __future__ import annotations

import logging
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from bot_core.api.server import LocalRuntimeContext, LocalRuntimeGateway
from bot_core.exchanges import interfaces as exchange_interfaces
from bot_core.exchanges import streaming as exchange_streaming
from bot_core.exchanges.base import AccountSnapshot
from bot_core.execution.paper import MarketMetadata
from bot_core.testing import TradingStubServer, build_default_dataset
from ui.backend.runtime_service import RuntimeService


def _build_stub_context(*, preset_dir: Path | None = None):
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
    if preset_dir is not None:
        context.strategy_presets_dir = preset_dir
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


def test_local_runtime_gateway_saves_strategy_preset(tmp_path: Path) -> None:
    context = _build_stub_context(preset_dir=tmp_path)
    gateway = LocalRuntimeGateway(context)
    response = gateway.dispatch(
        "autotrader.save_strategy_preset",
        {
            "preset": {
                "name": "Alpha Momentum",
                "blocks": [
                    {"type": "data_feed", "label": "Feed", "params": {"symbol": "BTC/USDT"}},
                    {"type": "allocator", "label": "Fixed", "params": {"fraction": 0.1}},
                ],
            }
        },
    )
    assert response["ok"] is True
    path = Path(response["path"])
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["name"] == "Alpha Momentum"
    assert payload["blocks"][0]["type"] == "data_feed"


def test_local_runtime_gateway_lists_and_loads_strategy_presets(tmp_path: Path) -> None:
    context = _build_stub_context(preset_dir=tmp_path)
    gateway = LocalRuntimeGateway(context)

    gateway.dispatch(
        "autotrader.save_strategy_preset",
        {
            "preset": {
                "name": "Beta Trend",
                "blocks": [
                    {"type": "data_feed", "label": "Feed", "params": {"symbol": "ETH/USDT"}},
                    {"type": "signal", "label": "Signal", "params": {"window": 14}},
                ],
            }
        },
    )

    gateway.dispatch(
        "autotrader.save_strategy_preset",
        {
            "preset": {
                "name": "Gamma Mean Reversion",
                "blocks": [
                    {"type": "data_feed", "label": "Feed", "params": {"symbol": "BTC/USDT"}},
                    {"type": "filter", "label": "Risk", "params": {"max_drawdown": 0.1}},
                ],
            }
        },
    )

    listing = gateway.dispatch("autotrader.list_strategy_presets", {})
    presets = listing["presets"]
    assert len(presets) == 2
    assert all(entry["block_count"] >= 2 for entry in presets)

    first = presets[0]
    loaded = gateway.dispatch("autotrader.load_strategy_preset", {"slug": first["slug"]})
    assert loaded["name"]
    assert len(loaded["blocks"]) == first["block_count"]

    second = presets[1]
    loaded_by_path = gateway.dispatch("autotrader.load_strategy_preset", {"path": second["path"]})
    assert loaded_by_path["path"].endswith(".json")
    assert loaded_by_path["blocks"][0]["type"] in {"data_feed", "filter", "signal"}

    removed = gateway.dispatch("autotrader.delete_strategy_preset", {"slug": first["slug"]})
    assert removed["ok"] is True
    removed_path = Path(removed["path"])
    assert not removed_path.exists()

    listing_after = gateway.dispatch("autotrader.list_strategy_presets", {})
    assert len(listing_after["presets"]) == 1
    assert listing_after["presets"][0]["slug"] == second["slug"]


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
    assert "websocket" not in (
        exchange_streaming.LocalLongPollStream.__doc__ or ""
    ).lower()
    assert not hasattr(exchange_streaming, "LocalWebSocketBridge")
    assert not hasattr(exchange_streaming.LocalLongPollStream, "websocket_bridge")


@pytest.mark.integration
def test_runtime_service_consumes_grpc_stream(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication  # type: ignore[attr-defined]

    dataset = build_default_dataset()
    metrics_path = tmp_path / "latency.json"

    with TradingStubServer(dataset, port=0, stream_repeat=True, stream_interval=0.0) as server:
        os.environ["BOT_CORE_UI_GRPC_ENDPOINT"] = server.address
        os.environ["BOT_CORE_UI_FEED_LATENCY_PATH"] = str(metrics_path)
        app = QCoreApplication.instance() or QCoreApplication([])
        service = RuntimeService(default_limit=5)
        try:
            assert service.attachToLiveDecisionLog("") is True
            deadline = time.time() + 5.0
            while time.time() < deadline and not service.decisions:
                app.processEvents()
                time.sleep(0.05)
            assert service.decisions, "Brak decyzji z gRPC"

            deadline = time.time() + 5.0
            while time.time() < deadline and not metrics_path.exists():
                app.processEvents()
                time.sleep(0.05)
            assert metrics_path.exists()
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            assert payload["count"] >= 1
            assert payload["max_ms"] >= payload["min_ms"] >= 0.0
        finally:
            service._stop_grpc_stream()
            app.quit()
        os.environ.pop("BOT_CORE_UI_GRPC_ENDPOINT", None)
        os.environ.pop("BOT_CORE_UI_FEED_LATENCY_PATH", None)
