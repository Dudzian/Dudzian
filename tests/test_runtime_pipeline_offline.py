from __future__ import annotations

import contextlib
import json
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from urllib.parse import parse_qs, urlparse

import pytest
import yaml

from bot_core.data.base import OHLCVRequest
from bot_core.data.ohlcv.cache import OfflineOnlyDataSource
from bot_core.runtime.pipeline import build_daily_trend_pipeline, build_multi_strategy_runtime
from bot_core.execution.live_router import LiveExecutionRouter
from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
)
from bot_core.observability.metrics import MetricsRegistry, summarize_live_execution_metrics
from bot_core.config.loader import load_runtime_app_config
from bot_core.exchanges.testing.loopback import LoopbackExchangeAdapter

from tests.test_runtime_bootstrap import _BASE_CONFIG, _prepare_manager


def _write_offline_pipeline_config(
    tmp_path: Path,
    *,
    include_suspension: bool = False,
    include_signal_limits: bool = False,
) -> Path:
    data = yaml.safe_load(_BASE_CONFIG)

    environments = data.setdefault("environments", {})
    offline_env = environments.get("coinbase_offline")
    if offline_env is None:  # pragma: no cover - defensywne zabezpieczenie konfiguracji testowej
        raise AssertionError("Konfiguracja bazowa nie zawiera środowiska coinbase_offline")

    data["environments"] = {"coinbase_offline": offline_env}

    offline_env["instrument_universe"] = "offline_universe"
    offline_env["default_strategy"] = "offline_trend"
    offline_env["default_controller"] = "offline_controller"
    offline_env["data_source"] = {
        "enable_snapshots": True,
        "cache_namespace": "offline_cache",
    }
    paper_settings = {
        "valuation_asset": "USDT",
        "portfolio_id": "offline-portfolio",
        "position_size": 0.25,
        "default_leverage": 1.0,
        "quote_assets": ["USDT"],
        "initial_balances": {"USDT": 250_000.0},
        "maker_fee": 0.0004,
        "taker_fee": 0.0006,
        "slippage_bps": 5.0,
        "default_market": {
            "min_notional": 10.0,
            "step_size": 0.001,
            "tick_size": 0.01,
        },
        "markets": {
            "BTC/USDT": {
                "min_notional": 10.0,
                "step_size": 0.001,
                "tick_size": 0.01,
            }
        },
    }
    offline_env["adapter_settings"] = {"paper_trading": paper_settings}

    data["strategies"] = {
        "offline_trend": {
            "engine": "daily_trend_momentum",
            "parameters": {
                "fast_ma": 5,
                "slow_ma": 20,
                "breakout_lookback": 10,
                "momentum_window": 10,
                "atr_window": 14,
                "atr_multiplier": 2.0,
                "min_trend_strength": 0.5,
                "min_momentum": 0.5,
            },
        }
    }
    runtime_section = data.setdefault("runtime", {})
    runtime_section.setdefault("controllers", {})["offline_controller"] = {
        "tick_seconds": 60,
        "interval": "1d",
    }
    data["instrument_universes"] = {
        "offline_universe": {
            "name": "offline_universe",
            "description": "Offline smoke-test universe",
            "instruments": {
                "BTC_USDT": {
                    "name": "BTC_USDT",
                    "base_asset": "BTC",
                    "quote_asset": "USDT",
                    "categories": ["core"],
                    "exchanges": {"coinbase_spot": "BTC/USDT"},
                    "backfill": [
                        {"interval": "1d", "lookback_days": 30},
                        {"interval": "1h", "lookback_days": 7},
                    ],
                }
            },
        }
    }
    scheduler_entry = {
        "name": "offline_scheduler",
        "telemetry_namespace": "offline",
        "schedules": {
            "offline_daily_trend": {
                "name": "offline_daily_trend",
                "strategy": "offline_trend",
                "cadence_seconds": 3600,
                "max_drift_seconds": 120,
                "warmup_bars": 5,
                "risk_profile": "balanced",
                "max_signals": 5,
                "interval": "1d",
            }
        },
    }
    if include_suspension:
        scheduler_entry["initial_suspensions"] = [
            {
                "schedule": "offline_daily_trend",
                "reason": "maintenance",
                "duration_seconds": 1200,
            },
            {
                "tag": "offline",
                "reason": "compliance",
                "until": "2030-01-02T00:00:00+00:00",
            },
        ]
    if include_signal_limits:
        scheduler_entry["initial_signal_limits"] = {
            "offline_trend": {
                "balanced": {
                    "limit": 3,
                    "reason": "bootstrap",
                    "duration_seconds": 1800,
                }
            }
        }
    data["multi_strategy_schedulers"] = {"offline_scheduler": scheduler_entry}

    for name, env_cfg in data.get("environments", {}).items():
        env_cfg["data_cache_path"] = str(tmp_path / "cache" / name)

    config_path = tmp_path / "core_offline_pipeline.yaml"
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return config_path


class _OfflineAdapter(ExchangeAdapter):
    name = "coinbase_spot"

    def __init__(
        self,
        credentials: ExchangeCredentials,
        *,
        environment: Environment | None = None,
        settings=None,
        client=None,
    ) -> None:
        super().__init__(credentials)
        self.environment = environment
        self.settings = settings
        self.client = client
        self.calls: list[tuple[str, tuple]] = []

    def configure_network(self, *, ip_allowlist=None) -> None:  # noqa: D401 - interfejs bazowy
        entries = tuple(ip_allowlist or ())
        self.calls.append(("configure_network", entries))

    def fetch_account_snapshot(self) -> AccountSnapshot:  # pragma: no cover - tryb offline
        return AccountSnapshot(balances={}, total_equity=0.0, available_margin=0.0, maintenance_margin=0.0)

    def fetch_symbols(self):  # pragma: no cover - nieużywane w testach
        return []

    def fetch_ohlcv(self, symbol, interval, start=None, end=None, limit=None):  # pragma: no cover - kontrola offline
        self.calls.append(("fetch_ohlcv", (symbol, interval, start, end, limit)))
        raise RuntimeError("Adapter nie powinien wykonywać zapytań sieciowych w trybie offline")

    def place_order(self, request):  # pragma: no cover - brak egzekucji w testach
        raise NotImplementedError

    def cancel_order(self, order_id, *, symbol=None):  # pragma: no cover - brak egzekucji w testach
        raise NotImplementedError

    def stream_public_data(self, *, channels):  # pragma: no cover - brak streamingu
        raise NotImplementedError

    def stream_private_data(self, *, channels):  # pragma: no cover - brak streamingu
        raise NotImplementedError


def _offline_adapter_factory(credentials: ExchangeCredentials, **kwargs) -> _OfflineAdapter:
    return _OfflineAdapter(credentials, **kwargs)


class _StubMarketIntelAggregator:
    def __init__(self, storage) -> None:
        self.storage = storage


@dataclass(slots=True)
class _LoopbackExchangeState:
    port: int
    symbols: tuple[str, ...]
    ohlcv: dict[str, list[list[float]]]
    account: dict[str, Any]
    orders: list[dict[str, Any]] = field(default_factory=list)
    cancellations: list[dict[str, Any]] = field(default_factory=list)
    streams: list[dict[str, Any]] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)


class _LoopbackHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass, state: _LoopbackExchangeState):
        super().__init__(server_address, RequestHandlerClass)
        self.state = state


class _LoopbackRequestHandler(BaseHTTPRequestHandler):
    server: _LoopbackHTTPServer  # type: ignore[assignment]
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return None

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        payload = self.rfile.read(length)
        try:
            return json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        state = self.server.state
        state.requests.append({"method": "GET", "path": parsed.path, "query": parsed.query})
        if parsed.path == "/account":
            self._send_json(state.account)
            return
        if parsed.path == "/symbols":
            self._send_json({"symbols": list(state.symbols)})
            return
        if parsed.path == "/ohlcv":
            params = parse_qs(parsed.query)
            symbol = params.get("symbol", [state.symbols[0]])[0]
            rows = state.ohlcv.get(symbol, [])
            self._send_json({"rows": rows})
            return
        self._send_json({"error": "not_found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        state = self.server.state
        payload = self._read_json_body()
        state.requests.append({"method": "POST", "path": parsed.path, "payload": payload})
        if parsed.path == "/orders":
            order = {
                "symbol": payload.get("symbol"),
                "side": payload.get("side"),
                "quantity": payload.get("quantity"),
                "price": payload.get("price"),
            }
            state.orders.append(order)
            order_id = f"loop-{len(state.orders)}"
            response = {
                "order_id": order_id,
                "status": "filled",
                "filled_quantity": payload.get("quantity", 0.0),
                "avg_price": payload.get("price", 100.0),
            }
            self._send_json(response)
            return
        if parsed.path in {"/stream/public", "/stream/private"}:
            entry = {"endpoint": parsed.path, **payload}
            state.streams.append(entry)
            self._send_json({"status": "ok"})
            return
        self._send_json({"error": "not_found"}, status=404)

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        state = self.server.state
        state.requests.append({"method": "DELETE", "path": parsed.path})
        if parsed.path.startswith("/orders/"):
            order_id = parsed.path.split("/", 2)[-1]
            payload = self._read_json_body()
            state.cancellations.append({"order_id": order_id, **payload})
            self._send_json({"status": "cancelled"})
            return
        self._send_json({"error": "not_found"}, status=404)


def _wait_for_loopback(port: int, *, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with contextlib.closing(
                socket.create_connection(("127.0.0.1", port), timeout=0.2)
            ):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"Serwer loopback nie nasłuchuje na porcie {port} w zadanym czasie")


@pytest.fixture()
def loopback_exchange_server() -> _LoopbackExchangeState:
    state = _LoopbackExchangeState(
        port=0,
        symbols=("BTCUSDT",),
        ohlcv={
            "BTCUSDT": [
                [1_700_000_000_000.0, 10.0, 11.0, 9.5, 10.5, 42.0],
                [1_700_003_600_000.0, 10.5, 11.5, 10.0, 11.2, 39.0],
            ]
        },
        account={
            "balances": {"USDT": 50_000.0},
            "total_equity": 50_000.0,
            "available_margin": 50_000.0,
            "maintenance_margin": 0.0,
        },
    )
    server = _LoopbackHTTPServer(("127.0.0.1", 0), _LoopbackRequestHandler, state)
    state.port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _wait_for_loopback(state.port)
        yield state
    finally:
        server.shutdown()
        thread.join(timeout=5.0)


def test_summarize_live_execution_metrics_filters_by_labels() -> None:
    registry = MetricsRegistry()

    success = registry.counter("live_orders_success_total", "desc")
    success.inc(labels={"exchange": "loopback_spot", "route": "primary"})
    success.inc(labels={"exchange": "loopback_spot", "route": "fallback"})
    success.inc(labels={"exchange": "other", "route": "primary"})
    success.inc()  # brak etykiet powinien być ignorowany przy filtrach

    failed = registry.counter("live_orders_failed_total", "desc")
    failed.inc(labels={"exchange": "other", "route": "primary"})

    routed = registry.counter("live_orders_total", "desc")
    routed.inc(labels={"exchange": "loopback_spot", "route": "primary"})
    routed.inc(labels={"exchange": "loopback_spot", "route": "fallback"})
    routed.inc(labels={"exchange": "other", "route": "primary"})
    routed.inc()

    summary_filtered = summarize_live_execution_metrics(
        registry,
        exchange="loopback_spot",
        route="primary",
    )
    assert summary_filtered["orders_success_total"] == 1
    assert summary_filtered["orders_failed_total"] == 0
    assert summary_filtered["orders_total"] == 1
    assert summary_filtered["orders_routed_total"] == 1

    summary_route_all = summarize_live_execution_metrics(
        registry,
        exchange="loopback_spot",
    )
    assert summary_route_all["orders_success_total"] == 2
    assert summary_route_all["orders_total"] == 2

    summary_all = summarize_live_execution_metrics(registry)
    assert summary_all["orders_success_total"] == 4
    assert summary_all["orders_failed_total"] == 1
    assert summary_all["orders_total"] == 5
    assert summary_all["orders_routed_total"] == 4


def _write_sample_cache(storage, rows):
    key = "BTC/USDT::1d"
    storage.write(
        key,
        {
            "columns": ["open_time", "open", "high", "low", "close", "volume"],
            "rows": rows,
        },
    )


def test_daily_trend_pipeline_offline_uses_cached_source(tmp_path: Path) -> None:
    config_path = _write_offline_pipeline_config(tmp_path)
    _, secret_manager = _prepare_manager()

    pipeline = build_daily_trend_pipeline(
        environment_name="coinbase_offline",
        strategy_name=None,
        controller_name=None,
        config_path=config_path,
        secret_manager=secret_manager,
        adapter_factories={"coinbase_spot": _offline_adapter_factory},
    )

    adapter = pipeline.bootstrap.adapter
    assert isinstance(adapter, _OfflineAdapter)
    assert pipeline.bootstrap.environment.offline_mode is True
    assert isinstance(pipeline.data_source.upstream, OfflineOnlyDataSource)
    assert pipeline.data_source.snapshot_fetcher is None
    assert pipeline.data_source.storage._primary._namespace == "offline_cache"  # type: ignore[attr-defined]

    sample_rows = [
        [1_700_000_000_000.0, 10.0, 11.0, 9.5, 10.5, 42.0],
        [1_700_086_400_000.0, 10.5, 11.5, 10.0, 11.2, 39.0],
    ]
    _write_sample_cache(pipeline.data_source.storage, sample_rows)

    request = OHLCVRequest(symbol="BTC/USDT", interval="1d", start=0, end=1_800_000_000_000, limit=10)
    response = pipeline.data_source.fetch_ohlcv(request)

    assert response.rows == sample_rows
    assert adapter.calls
    assert adapter.calls[0][0] == "configure_network"
    assert all(call[0] != "fetch_ohlcv" for call in adapter.calls)


def test_multi_strategy_runtime_offline_reuses_cached_feed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "bot_core.runtime.pipeline.MarketIntelAggregator",
        lambda storage: _StubMarketIntelAggregator(storage),
    )
    config_path = _write_offline_pipeline_config(tmp_path)
    _, secret_manager = _prepare_manager()

    runtime = build_multi_strategy_runtime(
        environment_name="coinbase_offline",
        scheduler_name=None,
        config_path=config_path,
        secret_manager=secret_manager,
        adapter_factories={"coinbase_spot": _offline_adapter_factory},
    )

    adapter = runtime.bootstrap.adapter
    assert isinstance(adapter, _OfflineAdapter)
    assert runtime.bootstrap.environment.offline_mode is True

    data_source = runtime.data_feed._data_source  # type: ignore[attr-defined]
    assert isinstance(data_source.upstream, OfflineOnlyDataSource)
    assert data_source.snapshot_fetcher is None
    assert data_source.storage._primary._namespace == "offline_cache"  # type: ignore[attr-defined]
    assert runtime.data_feed._symbols_map.get("offline_trend")  # type: ignore[attr-defined]
    assert runtime.data_feed._interval_map.get("offline_trend") == "1d"  # type: ignore[attr-defined]

    sample_rows = [
        [1_700_000_000_000.0, 10.0, 11.0, 9.5, 10.5, 42.0],
        [1_700_086_400_000.0, 10.5, 11.5, 10.0, 11.2, 39.0],
    ]
    _write_sample_cache(data_source.storage, sample_rows)

    probe = OHLCVRequest(symbol="BTC/USDT", interval="1d", start=0, end=2_000_000_000_000, limit=10)
    probe_response = data_source.fetch_ohlcv(probe)
    assert probe_response.rows

    snapshots = runtime.data_feed.fetch_latest("offline_trend")
    assert snapshots
    assert adapter.calls
    assert adapter.calls[0][0] == "configure_network"
    assert all(call[0] != "fetch_ohlcv" for call in adapter.calls)


def test_multi_strategy_runtime_applies_initial_suspensions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "bot_core.runtime.pipeline.MarketIntelAggregator",
        lambda storage: _StubMarketIntelAggregator(storage),
    )
    config_path = _write_offline_pipeline_config(tmp_path, include_suspension=True)
    _, secret_manager = _prepare_manager()

    runtime = build_multi_strategy_runtime(
        environment_name="coinbase_offline",
        scheduler_name=None,
        config_path=config_path,
        secret_manager=secret_manager,
        adapter_factories={"coinbase_spot": _offline_adapter_factory},
    )

    snapshot = runtime.scheduler.suspension_snapshot()
    schedule_entry = snapshot["schedules"].get("offline_daily_trend")
    assert schedule_entry is not None
    assert schedule_entry["reason"] == "maintenance"
    tag_entry = snapshot["tags"].get("offline")
    assert tag_entry is not None
    assert tag_entry["reason"] == "compliance"
    until_iso = tag_entry.get("until")
    assert until_iso is not None
    assert datetime.fromisoformat(str(until_iso)) == datetime(2030, 1, 2, tzinfo=timezone.utc)


def test_multi_strategy_runtime_applies_initial_signal_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "bot_core.runtime.pipeline.MarketIntelAggregator",
        lambda storage: _StubMarketIntelAggregator(storage),
    )
    config_path = _write_offline_pipeline_config(
        tmp_path,
        include_signal_limits=True,
    )
    _, secret_manager = _prepare_manager()

    runtime = build_multi_strategy_runtime(
        environment_name="coinbase_offline",
        scheduler_name=None,
        config_path=config_path,
        secret_manager=secret_manager,
        adapter_factories={"coinbase_spot": _offline_adapter_factory},
    )

    snapshot = runtime.scheduler.signal_limit_snapshot()
    strategy_limits = snapshot.get("offline_trend")
    assert strategy_limits is not None
    profile_entry = strategy_limits.get("balanced")
    assert profile_entry is not None
    assert profile_entry.get("limit") == 3
    assert profile_entry.get("reason") == "bootstrap"
    assert profile_entry.get("active") is True
    expires_at = profile_entry.get("expires_at")
    assert isinstance(expires_at, str)


def _materialize_loopback_configs(tmp_path: Path, *, port: int) -> tuple[Path, Path]:
    core_template = Path("config/e2e/core_loopback.yaml")
    runtime_template = Path("config/e2e/live_loopback.yaml")
    data = yaml.safe_load(core_template.read_text(encoding="utf-8"))
    environments = data.get("environments", {})
    for env_cfg in environments.values():
        env_cfg["data_cache_path"] = str(tmp_path / env_cfg.get("environment", "loopback"))
        adapter_settings = env_cfg.setdefault("adapter_settings", {})
        live_settings = adapter_settings.get("live_trading", {})
        live_settings["base_url"] = f"http://127.0.0.1:{port}"
        adapter_settings["live_trading"] = live_settings
    core_path = tmp_path / "core_loopback.yaml"
    core_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    runtime_data = yaml.safe_load(runtime_template.read_text(encoding="utf-8"))
    runtime_data.setdefault("core", {})["path"] = core_path.name
    runtime_path = tmp_path / "runtime_loopback.yaml"
    runtime_path.write_text(yaml.safe_dump(runtime_data, sort_keys=False), encoding="utf-8")
    return core_path, runtime_path


def test_live_pipeline_uses_loopback_adapter(
    tmp_path: Path,
    loopback_exchange_server: _LoopbackExchangeState,
) -> None:
    core_path, runtime_path = _materialize_loopback_configs(tmp_path, port=loopback_exchange_server.port)
    _, secret_manager = _prepare_manager()
    runtime_cfg = load_runtime_app_config(runtime_path)

    pipeline = build_daily_trend_pipeline(
        environment_name="loopback_testnet",
        strategy_name=None,
        controller_name=None,
        config_path=core_path,
        secret_manager=secret_manager,
        runtime_config=runtime_cfg,
    )

    adapter = pipeline.bootstrap.adapter
    assert isinstance(adapter, LoopbackExchangeAdapter)
    assert isinstance(pipeline.execution_service, LiveExecutionRouter)

    stream = adapter.stream_public_data(channels=["ticker"])
    stream.close()
    assert any(entry.get("action") == "open" for entry in loopback_exchange_server.streams)

    request = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        order_type="market",
        quantity=0.1,
        price=10.5,
    )
    execution_context = pipeline.controller.execution_context
    result = pipeline.execution_service.execute(request, execution_context)
    assert result.order_id.startswith("loop-")
    assert loopback_exchange_server.orders

    metrics = summarize_live_execution_metrics(
        pipeline.execution_service._metrics,  # type: ignore[attr-defined]
        exchange="loopback_spot",
        symbol="BTCUSDT",
        portfolio=execution_context.portfolio_id,
    )
    assert metrics["fill_ratio_avg"] == pytest.approx(1.0)
    assert metrics["fill_ratio_count"] == 1
    assert metrics["fill_ratio_sum"] == pytest.approx(1.0)
    assert metrics["fill_ratio_p50"] == pytest.approx(1.0)
    assert metrics["fill_ratio_p95"] == pytest.approx(1.0)
    assert metrics["fill_ratio_min"] == pytest.approx(1.0)
    assert metrics["fill_ratio_max"] == pytest.approx(1.0)
    assert metrics["fill_ratio_stddev"] == pytest.approx(0.0)
    assert metrics["errors_total"] == 0
    assert metrics["latency_count"] == 1
    assert metrics["latency_sum"] >= 0.0
    assert metrics["latency_avg"] is not None
    assert metrics["latency_p50"] is not None
    assert metrics["latency_p95"] is not None
    assert metrics["latency_p99"] is not None
    assert metrics["latency_p99"] >= metrics["latency_p95"] >= metrics["latency_p50"] >= 0.0
    assert metrics["latency_min"] is not None
    assert metrics["latency_max"] is not None
    assert metrics["latency_max"] >= metrics["latency_min"] >= 0.0
    assert metrics["latency_stddev"] is not None
    assert metrics["latency_stddev"] >= 0.0
    assert metrics["orders_success_total"] == 1
    assert metrics["orders_failed_total"] == 0
    assert metrics["orders_total"] == 1
    assert metrics["orders_routed_total"] == 1
    assert metrics["orders_attempts_total"] == 1
    assert metrics["orders_attempts_success"] == 1
    assert metrics["orders_attempts_error"] == 0
    assert metrics["orders_attempts_api_error"] == 0
    assert metrics["orders_attempts_auth_error"] == 0
    assert metrics["orders_attempts_exception"] == 0
    assert metrics["orders_attempts_success_rate"] == pytest.approx(1.0)
    assert metrics["orders_attempts_error_rate"] == pytest.approx(0.0)
    assert metrics["orders_attempts_api_error_rate"] == pytest.approx(0.0)
    assert metrics["orders_attempts_auth_error_rate"] == pytest.approx(0.0)
    assert metrics["orders_attempts_exception_rate"] == pytest.approx(0.0)
    assert metrics["orders_fallback_total"] == 0
    assert metrics["orders_success_rate"] == pytest.approx(1.0)
    assert metrics["orders_failure_rate"] == pytest.approx(0.0)
    assert metrics["orders_fallback_rate"] == pytest.approx(0.0)
