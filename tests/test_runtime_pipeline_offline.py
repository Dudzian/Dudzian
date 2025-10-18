from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bot_core.data.base import OHLCVRequest
from bot_core.data.ohlcv.cache import OfflineOnlyDataSource
from bot_core.runtime.pipeline import build_daily_trend_pipeline, build_multi_strategy_runtime
from bot_core.exchanges.base import AccountSnapshot, Environment, ExchangeAdapter, ExchangeCredentials

from tests.test_runtime_bootstrap import _BASE_CONFIG, _prepare_manager


def _write_offline_pipeline_config(tmp_path: Path) -> Path:
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
    data["multi_strategy_schedulers"] = {
        "offline_scheduler": {
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
    }

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
