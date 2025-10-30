from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import pytest
import yaml

from bot_core.execution.live_router import LiveExecutionRouter
from bot_core.execution.base import ExecutionContext
from bot_core.exchanges.base import AccountSnapshot, Environment, ExchangeAdapter, ExchangeCredentials, OrderRequest, OrderResult
from bot_core.runtime.pipeline import build_daily_trend_pipeline
from bot_core.config.models import RuntimeExecutionLiveSettings, RuntimeExecutionSettings
from bot_core.security import SecretManager, SecretStorage


class _InMemorySecretStorage(SecretStorage):
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get_secret(self, key: str) -> str | None:
        return self._store.get(key)

    def set_secret(self, key: str, value: str) -> None:
        self._store[key] = value

    def delete_secret(self, key: str) -> None:
        self._store.pop(key, None)


@dataclass(slots=True)
class _OhlcvFixture:
    symbol: str
    rows: Sequence[Sequence[float]]


class _StreamStub:
    def close(self) -> None:  # pragma: no cover - brak logiki w stubie
        return None


@dataclass(slots=True)
class _RuntimeConfigStub:
    execution: RuntimeExecutionSettings


class RecordingLiveAdapter(ExchangeAdapter):
    name = "fake_exchange"

    def __init__(self, credentials: ExchangeCredentials, fixtures: Sequence[_OhlcvFixture]) -> None:
        super().__init__(credentials)
        self._fixtures = {fixture.symbol: list(fixture.rows) for fixture in fixtures}
        self.orders: list[OrderRequest] = []
        self.cancelled: list[str] = []

    def configure_network(self, *, ip_allowlist: Sequence[str] | None = None) -> None:  # noqa: D401, ARG002
        return None

    def fetch_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            balances={"USDT": 25_000.0},
            total_equity=25_000.0,
            available_margin=25_000.0,
            maintenance_margin=0.0,
        )

    def fetch_symbols(self) -> Sequence[str]:
        return tuple(self._fixtures.keys())

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        limit: int | None = None,
    ) -> Sequence[Sequence[float]]:
        del interval, start, end, limit
        return list(self._fixtures.get(symbol, ()))

    def place_order(self, request: OrderRequest) -> OrderResult:
        self.orders.append(request)
        return OrderResult(
            order_id=f"ord-{len(self.orders)}",
            status="filled",
            filled_quantity=request.quantity,
            avg_price=request.price or 100.0,
            raw_response={"exchange_order_id": f"ext-{len(self.orders)}"},
        )

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:
        del symbol
        self.cancelled.append(order_id)

    def stream_public_data(self, *, channels: Sequence[str]) -> _StreamStub:  # noqa: D401, ARG002
        del channels
        return _StreamStub()

    def stream_private_data(self, *, channels: Sequence[str]) -> _StreamStub:  # noqa: D401, ARG002
        del channels
        return _StreamStub()


@pytest.fixture()
def live_runtime_fixture(tmp_path: Path) -> tuple[Path, RecordingLiveAdapter, SecretManager, _RuntimeConfigStub]:
    candles = [
        [1_600_000_000_000, 100.0, 105.0, 95.0, 102.0, 12.0],
        [1_600_086_400_000, 102.0, 107.0, 101.0, 104.0, 11.0],
    ]
    adapter = RecordingLiveAdapter(
        ExchangeCredentials(key_id="public", environment=Environment.LIVE),
        fixtures=(_OhlcvFixture(symbol="BTCUSDT", rows=candles),),
    )

    config = {
        "risk_profiles": {
            "balanced": {
                "max_daily_loss_pct": 0.02,
                "max_position_pct": 0.1,
                "target_volatility": 0.12,
                "max_leverage": 2.0,
                "stop_loss_atr_multiple": 1.5,
                "max_open_positions": 4,
                "hard_drawdown_pct": 0.2,
            }
        },
        "runtime": {
            "controllers": {"daily_trend_core": {"tick_seconds": 3600, "interval": "1h"}}
        },
        "strategies": {
            "core_daily_trend": {
                "engine": "daily_trend_momentum",
                "parameters": {
                    "fast_ma": 3,
                    "slow_ma": 5,
                    "breakout_lookback": 4,
                    "momentum_window": 3,
                    "atr_window": 3,
                    "atr_multiplier": 1.5,
                    "min_trend_strength": 0.0,
                    "min_momentum": 0.0,
                },
            }
        },
        "instrument_universes": {
            "core_universe": {
                "description": "fixture",
                "instruments": {
                    "BTC_USDT": {
                        "base_asset": "BTC",
                        "quote_asset": "USDT",
                        "categories": ["core"],
                        "exchanges": {"fake_exchange": "BTCUSDT"},
                        "backfill": [{"interval": "1h", "lookback_days": 5}],
                    }
                },
            }
        },
        "environments": {
            "fake_live": {
                "exchange": "fake_exchange",
                "environment": "live",
                "offline_mode": True,
                "keychain_key": "fake_key",
                "data_cache_path": str(tmp_path / "data"),
                "risk_profile": "balanced",
                "default_strategy": "core_daily_trend",
                "default_controller": "daily_trend_core",
                "alert_channels": [],
                "instrument_universe": "core_universe",
                "adapter_settings": {
                    "live_trading": {
                        "valuation_asset": "USDT",
                        "maker_fee": 0.0004,
                        "taker_fee": 0.0006,
                    }
                },
            }
        },
        "alerts": {},
    }

    core_path = tmp_path / "core.yaml"
    core_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    storage = _InMemorySecretStorage()
    secret_manager = SecretManager(storage)
    secret_manager.store_exchange_credentials(
        "fake_key",
        ExchangeCredentials(key_id="public", secret="secret", environment=Environment.LIVE),
    )

    execution_settings = RuntimeExecutionSettings(
        default_mode="live",
        force_paper_when_offline=False,
        auth_token="local-token",
        live=RuntimeExecutionLiveSettings(
            enabled=True,
            default_route=("fake_exchange",),
        ),
    )

    runtime_config = _RuntimeConfigStub(execution=execution_settings)

    # W środowiskach testowych _ensure_local_market_data_availability może nie istnieć.
    import bot_core.runtime.pipeline as pipeline_module

    if not hasattr(pipeline_module, "_ensure_local_market_data_availability"):
        pipeline_module._ensure_local_market_data_availability = lambda *args, **kwargs: None

    return core_path, adapter, secret_manager, runtime_config


def test_live_execution_flow(live_runtime_fixture: tuple[Path, RecordingLiveAdapter, SecretManager, _RuntimeConfigStub]) -> None:
    config_path, adapter, secret_manager, runtime_config = live_runtime_fixture

    pipeline = build_daily_trend_pipeline(
        environment_name="fake_live",
        strategy_name="core_daily_trend",
        controller_name="daily_trend_core",
        config_path=config_path,
        secret_manager=secret_manager,
        adapter_factories={"fake_exchange": lambda credentials, **_: adapter},
        runtime_config=runtime_config,
    )

    assert isinstance(pipeline.execution_service, LiveExecutionRouter)

    execution_context: ExecutionContext = pipeline.controller.execution_context
    order_request = OrderRequest(symbol="BTCUSDT", side="buy", quantity=0.1, order_type="market")

    result = pipeline.execution_service.execute(order_request, execution_context)

    assert adapter.orders, "Adapter powinien otrzymać zlecenie w trybie live"
    assert result.order_id

    pipeline.execution_service.cancel(result.order_id, execution_context)
    assert result.order_id in adapter.cancelled

    snapshot = pipeline.controller.account_loader()
    assert isinstance(snapshot, AccountSnapshot)
    assert snapshot.total_equity == pytest.approx(25_000.0)
