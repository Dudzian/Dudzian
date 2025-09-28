from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Protocol, Sequence

import pytest

from bot_core.alerts import DefaultAlertRouter
from bot_core.alerts.audit import InMemoryAlertAuditLog
from bot_core.alerts.base import AlertChannel, AlertMessage
from bot_core.execution.base import ExecutionContext
from bot_core.execution.paper import PaperTradingExecutionService
from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)
from bot_core.runtime import build_daily_trend_pipeline, create_trading_controller
from bot_core.security import SecretManager, SecretStorage
from bot_core.strategies import StrategySignal
from bot_core.strategies.daily_trend import DailyTrendMomentumStrategy


class _InMemorySecretStorage(SecretStorage):
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get_secret(self, key: str) -> Optional[str]:
        return self._store.get(key)

    def set_secret(self, key: str, value: str) -> None:
        self._store[key] = value

    def delete_secret(self, key: str) -> None:
        self._store.pop(key, None)


@dataclass(slots=True)
class _OhlcvFixture:
    symbol: str
    rows: Sequence[Sequence[float]]


class _FakeStream(Protocol):
    def close(self) -> None:  # pragma: no cover - wymagane przez interfejs
        ...


class FakeExchangeAdapter(ExchangeAdapter):
    name = "fake_exchange"

    def __init__(self, credentials: ExchangeCredentials, *, fixtures: Sequence[_OhlcvFixture]) -> None:
        super().__init__(credentials)
        self._fixtures = {fixture.symbol: fixture.rows for fixture in fixtures}

    def configure_network(self, *, ip_allowlist: Optional[Sequence[str]] = None) -> None:  # noqa: D401, ARG002
        return None

    def fetch_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            balances={"USDT": 0.0},
            total_equity=0.0,
            available_margin=0.0,
            maintenance_margin=0.0,
        )

    def fetch_symbols(self) -> Iterable[str]:  # pragma: no cover - nieużywane w teście
        return tuple(self._fixtures.keys())

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Sequence[Sequence[float]]:
        rows = list(self._fixtures.get(symbol, ()))
        if start is not None:
            rows = [row for row in rows if float(row[0]) >= start]
        if end is not None:
            rows = [row for row in rows if float(row[0]) <= end]
        if limit is not None:
            rows = rows[:limit]
        return rows

    def place_order(self, request: OrderRequest) -> OrderResult:  # pragma: no cover - pipeline używa paper tradingu
        raise NotImplementedError

    def cancel_order(self, order_id: str, *, symbol: Optional[str] = None) -> None:  # pragma: no cover - nieużywane
        raise NotImplementedError

    def stream_public_data(self, *, channels: Sequence[str]) -> _FakeStream:  # pragma: no cover - nieużywane
        raise NotImplementedError

    def stream_private_data(self, *, channels: Sequence[str]) -> _FakeStream:  # pragma: no cover - nieużywane
        raise NotImplementedError


class CollectingChannel(AlertChannel):
    name = "collector"

    def __init__(self) -> None:
        self.messages: list[AlertMessage] = []

    def send(self, message: AlertMessage) -> None:
        self.messages.append(message)

    def health_check(self) -> Mapping[str, str]:
        return {"status": "ok"}


@pytest.fixture()
def pipeline_fixture(tmp_path: Path) -> tuple[Path, FakeExchangeAdapter, SecretManager]:
    candles = [
        [1_600_000_000_000, 100.0, 105.0, 95.0, 102.0, 12.0],
        [1_600_086_400_000, 102.0, 107.0, 101.0, 104.0, 11.0],
    ]
    adapter = FakeExchangeAdapter(
        ExchangeCredentials(key_id="public", environment=Environment.PAPER),
        fixtures=(_OhlcvFixture(symbol="BTCUSDT", rows=candles),),
    )

    config = {
        "risk_profiles": {
            "balanced": {
                "max_daily_loss_pct": 0.015,
                "max_position_pct": 0.05,
                "target_volatility": 0.11,
                "max_leverage": 3.0,
                "stop_loss_atr_multiple": 1.5,
                "max_open_positions": 5,
                "hard_drawdown_pct": 0.1,
            }
        },
        "runtime": {
            "controllers": {"daily_trend_core": {"tick_seconds": 86400, "interval": "1d"}}
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
                        "categories": [],
                        "exchanges": {"fake_exchange": "BTCUSDT"},
                        "backfill": [{"interval": "1d", "lookback_days": 10}],
                    }
                },
            }
        },
        "environments": {
            "fake_paper": {
                "exchange": "fake_exchange",
                "environment": "paper",
                "keychain_key": "fake_key",
                "data_cache_path": str(tmp_path / "data"),
                "risk_profile": "balanced",
                "alert_channels": [],
                "instrument_universe": "core_universe",
                "adapter_settings": {
                    "paper_trading": {
                        "valuation_asset": "USDT",
                        "position_size": 0.1,
                        "initial_balances": {"USDT": 10_000},
                        "default_market": {"min_quantity": 0.001, "min_notional": 10.0},
                    }
                },
            }
        },
        "alerts": {},
    }

    config_path = tmp_path / "core.yaml"

    # Zamieniamy strukturę na YAML kompatybilny z loaderem.
    import yaml  # type: ignore

    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    storage = _InMemorySecretStorage()
    manager = SecretManager(storage)
    manager.store_exchange_credentials(
        "fake_key",
        ExchangeCredentials(key_id="public", secret="sekret", environment=Environment.PAPER),
    )
    return config_path, adapter, manager


def test_build_daily_trend_pipeline(pipeline_fixture: tuple[Path, FakeExchangeAdapter, SecretManager]) -> None:
    config_path, adapter, manager = pipeline_fixture

    pipeline = build_daily_trend_pipeline(
        environment_name="fake_paper",
        strategy_name="core_daily_trend",
        controller_name="daily_trend_core",
        config_path=config_path,
        secret_manager=manager,
        adapter_factories={"fake_exchange": lambda credentials, **_: adapter},
    )

    assert pipeline.controller.symbols == ("BTCUSDT",)
    snapshot = pipeline.controller.account_loader()
    assert snapshot.total_equity == pytest.approx(10_000.0)
    assert snapshot.available_margin == pytest.approx(10_000.0)
    assert isinstance(pipeline.execution_service, PaperTradingExecutionService)
    assert isinstance(pipeline.strategy, DailyTrendMomentumStrategy)

    balances = pipeline.execution_service.balances()
    assert balances["USDT"] == 10_000.0

    data_root = config_path.parent / "data"
    parquet_root = data_root / "ohlcv_parquet" / "fake_exchange"
    manifest_path = data_root / "ohlcv_manifest.sqlite"
    assert manifest_path.exists()
    parquet_files = list(parquet_root.rglob("data.parquet"))
    assert parquet_files, "Backfill pipeline powinien zapisać dane OHLCV w Parquet"


def test_create_trading_controller_executes_signal(
    pipeline_fixture: tuple[Path, FakeExchangeAdapter, SecretManager]
) -> None:
    config_path, adapter, manager = pipeline_fixture

    pipeline = build_daily_trend_pipeline(
        environment_name="fake_paper",
        strategy_name="core_daily_trend",
        controller_name="daily_trend_core",
        config_path=config_path,
        secret_manager=manager,
        adapter_factories={"fake_exchange": lambda credentials, **_: adapter},
    )

    router = DefaultAlertRouter(audit_log=InMemoryAlertAuditLog())
    channel = CollectingChannel()
    router.register(channel)

    trading_controller = create_trading_controller(
        pipeline,
        router,
        health_check_interval=0.0,
    )

    signal = StrategySignal(
        symbol="BTCUSDT",
        side="BUY",
        confidence=0.9,
        metadata={"quantity": 0.2, "price": 102.0},
    )

    results = trading_controller.process_signals([signal])

    assert len(results) == 1
    assert results[0].status == "filled"
    assert channel.messages
    assert channel.messages[0].category == "strategy"


def test_account_loader_handles_multi_currency_and_shorts(tmp_path: Path) -> None:
    candles_btc_usdt = [[1_600_000_000_000, 20_000.0, 20_000.0, 20_000.0, 20_000.0, 15.0]]
    candles_eth_usdt = [[1_600_000_000_000, 1_500.0, 1_500.0, 1_500.0, 1_500.0, 12.0]]
    candles_btc_eur = [[1_600_000_000_000, 18_000.0, 18_000.0, 18_000.0, 18_000.0, 11.0]]

    adapter = FakeExchangeAdapter(
        ExchangeCredentials(key_id="public", environment=Environment.PAPER),
        fixtures=(
            _OhlcvFixture(symbol="BTCUSDT", rows=candles_btc_usdt),
            _OhlcvFixture(symbol="ETHUSDT", rows=candles_eth_usdt),
            _OhlcvFixture(symbol="BTCEUR", rows=candles_btc_eur),
        ),
    )

    config = {
        "risk_profiles": {
            "balanced": {
                "max_daily_loss_pct": 0.015,
                "max_position_pct": 0.05,
                "target_volatility": 0.11,
                "max_leverage": 3.0,
                "stop_loss_atr_multiple": 1.5,
                "max_open_positions": 5,
                "hard_drawdown_pct": 0.1,
            }
        },
        "runtime": {"controllers": {"daily_trend_core": {"tick_seconds": 86400, "interval": "1d"}}},
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
            "multi_quote": {
                "description": "fixture",
                "instruments": {
                    "BTC_USDT": {
                        "base_asset": "BTC",
                        "quote_asset": "USDT",
                        "categories": [],
                        "exchanges": {"fake_exchange": "BTCUSDT"},
                        "backfill": [{"interval": "1d", "lookback_days": 10}],
                    },
                    "ETH_USDT": {
                        "base_asset": "ETH",
                        "quote_asset": "USDT",
                        "categories": [],
                        "exchanges": {"fake_exchange": "ETHUSDT"},
                        "backfill": [{"interval": "1d", "lookback_days": 10}],
                    },
                    "BTC_EUR": {
                        "base_asset": "BTC",
                        "quote_asset": "EUR",
                        "categories": [],
                        "exchanges": {"fake_exchange": "BTCEUR"},
                        "backfill": [{"interval": "1d", "lookback_days": 10}],
                    },
                },
            }
        },
        "environments": {
            "fake_multi": {
                "exchange": "fake_exchange",
                "environment": "paper",
                "keychain_key": "fake_key",
                "data_cache_path": str(tmp_path / "data"),
                "risk_profile": "balanced",
                "alert_channels": [],
                "instrument_universe": "multi_quote",
                "adapter_settings": {
                    "paper_trading": {
                        "valuation_asset": "USDT",
                        "quote_assets": ["USDT", "EUR"],
                        "position_size": 0.1,
                        "initial_balances": {"USDT": 0.0, "EUR": 0.0, "BTC": 0.0},
                        "default_market": {"min_quantity": 0.001, "min_notional": 10.0},
                    }
                },
            }
        },
        "alerts": {},
    }

    config_path = tmp_path / "core_multi.yaml"
    import yaml  # type: ignore

    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    storage = _InMemorySecretStorage()
    manager = SecretManager(storage)
    manager.store_exchange_credentials(
        "fake_key",
        ExchangeCredentials(key_id="public", secret="sekret", environment=Environment.PAPER),
    )

    pipeline = build_daily_trend_pipeline(
        environment_name="fake_multi",
        strategy_name="core_daily_trend",
        controller_name="daily_trend_core",
        config_path=config_path,
        secret_manager=manager,
        adapter_factories={"fake_exchange": lambda credentials, **_: adapter},
    )

    # Ustawiamy stany konta paper tradingu.
    execution_service = pipeline.execution_service
    assert isinstance(execution_service, PaperTradingExecutionService)
    execution_service._balances.clear()  # type: ignore[attr-defined]
    execution_service._balances.update({  # type: ignore[attr-defined]
        "USDT": 8_000.0,
        "EUR": 5_000.0,
        "BTC": 0.1,
    })

    snapshot = pipeline.controller.account_loader()
    btc_usdt_close = candles_btc_usdt[-1][4]
    btc_eur_close = candles_btc_eur[-1][4]
    eur_to_usdt = btc_usdt_close / btc_eur_close
    expected_initial = (
        execution_service._balances["USDT"]  # type: ignore[attr-defined]
        + execution_service._balances["BTC"] * btc_usdt_close  # type: ignore[attr-defined]
        + execution_service._balances["EUR"] * eur_to_usdt  # type: ignore[attr-defined]
    )
    assert snapshot.total_equity == pytest.approx(expected_initial, rel=1e-4)
    assert snapshot.available_margin == pytest.approx(execution_service._balances["USDT"])  # type: ignore[attr-defined]

    context = ExecutionContext(
        portfolio_id="test",
        risk_profile="balanced",
        environment="paper",
        metadata={"leverage": "3"},
    )
    execution_service.execute(
        OrderRequest(symbol="ETHUSDT", side="sell", quantity=1.0, order_type="market", price=1_500.0),
        context,
    )

    snapshot_after = pipeline.controller.account_loader()
    usdt_after = execution_service._balances["USDT"]  # type: ignore[attr-defined]
    short_state = execution_service._short_positions["ETHUSDT"]  # type: ignore[attr-defined]
    converted_balances = (
        usdt_after
        + execution_service._balances["BTC"] * btc_usdt_close  # type: ignore[attr-defined]
        + execution_service._balances["EUR"] * eur_to_usdt  # type: ignore[attr-defined]
    )
    expected_after = (
        converted_balances
        + short_state.margin
        - candles_eth_usdt[-1][4] * short_state.quantity
    )
    assert snapshot_after.total_equity == pytest.approx(expected_after, rel=1e-4)
    assert snapshot_after.available_margin == pytest.approx(usdt_after)
