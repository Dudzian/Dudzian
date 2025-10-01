from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.data.ohlcv import ParquetCacheStorage, SQLiteCacheStorage
from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeCredentials,
)
from bot_core.runtime.pipeline import build_daily_trend_pipeline
from bot_core.security import SecretManager, SecretStorage


class _InMemorySecretStorage(SecretStorage):
    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def get_secret(self, key: str) -> str | None:
        return self._data.get(key)

    def set_secret(self, key: str, value: str) -> None:
        self._data[key] = value

    def delete_secret(self, key: str) -> None:
        self._data.pop(key, None)


_SYMBOL = "BTCUSDT"
_INTERVAL = "1d"
_NAMESPACE = "binance_spot"
_COLUMNS = ("open_time", "open", "high", "low", "close", "volume")


def _fixture_rows() -> list[list[float]]:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    day_ms = 24 * 60 * 60 * 1000
    prices = (
        (42000.0, 42500.0, 41800.0, 42400.0, 128.5),
        (42400.0, 43000.0, 42300.0, 42800.0, 131.2),
        (42800.0, 43500.0, 42600.0, 43300.0, 135.4),
        (43300.0, 44000.0, 43200.0, 43800.0, 142.1),
        (43800.0, 44600.0, 43700.0, 44400.0, 147.8),
        (44400.0, 45200.0, 44300.0, 45000.0, 152.6),
        (45000.0, 45900.0, 44900.0, 45650.0, 158.9),
        (45650.0, 46500.0, 45500.0, 46300.0, 163.5),
    )
    rows: list[list[float]] = []
    for index, (open_, high, low, close, volume) in enumerate(prices):
        timestamp = int((base.timestamp() * 1000) + index * day_ms)
        rows.append([timestamp, open_, high, low, close, volume])
    return rows


class _FixtureExchangeAdapter(ExchangeAdapter):
    name = "binance_spot"

    def __init__(self, credentials: ExchangeCredentials, **_: object) -> None:
        super().__init__(credentials)

    def configure_network(self, *, ip_allowlist: tuple[str, ...] | None = None) -> None:  # noqa: D401, ARG002
        return None

    def fetch_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            balances={"USDT": 100_000.0},
            total_equity=100_000.0,
            available_margin=100_000.0,
            maintenance_margin=0.0,
        )

    def fetch_symbols(self):  # pragma: no cover - nieużywane w teście
        return ("BTCUSDT",)

    def fetch_ohlcv(self, symbol: str, interval: str, start: int | None = None, end: int | None = None, limit: int | None = None):  # noqa: D401, ARG002
        return []

    def place_order(self, request):  # pragma: no cover - nieużywane w teście
        raise NotImplementedError

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:  # pragma: no cover - nieużywane w teście
        raise NotImplementedError

    def stream_public_data(self, *, channels):  # pragma: no cover - nieużywane w teście
        raise NotImplementedError

    def stream_private_data(self, *, channels):  # pragma: no cover - nieużywane w teście
        raise NotImplementedError


@pytest.fixture()
def _fixture_cache(tmp_path: Path) -> Path:
    cache_dir = tmp_path / "cache"
    payload = {"columns": _COLUMNS, "rows": _fixture_rows()}
    parquet_storage = ParquetCacheStorage(cache_dir / "ohlcv_parquet", namespace=_NAMESPACE)
    parquet_storage.write(f"{_SYMBOL}::{_INTERVAL}", payload)
    metadata = parquet_storage.metadata()
    metadata[f"row_count::{_SYMBOL}::{_INTERVAL}"] = str(len(payload["rows"]))
    metadata[f"last_timestamp::{_SYMBOL}::{_INTERVAL}"] = str(int(payload["rows"][-1][0]))

    manifest_storage = SQLiteCacheStorage(cache_dir / "ohlcv_manifest.sqlite", store_rows=False)
    manifest_storage.write(f"{_SYMBOL}::{_INTERVAL}", payload)
    return cache_dir


def _write_config(path: Path, *, cache_dir: Path, ledger_dir: Path) -> None:
    config = {
        "risk_profiles": {
            "binance_smoke_profile": {
                "max_daily_loss_pct": 0.5,
                "max_position_pct": 1.0,
                "target_volatility": 0.5,
                "max_leverage": 3.0,
                "stop_loss_atr_multiple": 1.5,
                "max_open_positions": 10,
                "hard_drawdown_pct": 0.9,
            }
        },
        "runtime": {"controllers": {"daily_trend_smoke": {"tick_seconds": 86400, "interval": "1d"}}},
        "strategies": {
            "smoke_daily_trend": {
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
            "binance_smoke_universe": {
                "description": "smoke test fixtures",
                "instruments": {
                    "BTC_USDT": {
                        "base_asset": "BTC",
                        "quote_asset": "USDT",
                        "categories": ["smoke"],
                        "exchanges": {"binance_spot": "BTCUSDT"},
                        "backfill": [{"interval": "1d", "lookback_days": 30}],
                    }
                },
            }
        },
        "environments": {
            "binance_smoke": {
                "exchange": "binance_spot",
                "environment": "paper",
                "keychain_key": "binance_fixture_key",
                "credential_purpose": "trading",
                "data_cache_path": str(cache_dir),
                "risk_profile": "binance_smoke_profile",
                "alert_channels": [],
                "instrument_universe": "binance_smoke_universe",
                "ip_allowlist": [],
                "required_permissions": ["read", "trade"],
                "forbidden_permissions": [],
                "adapter_settings": {
                    "paper_trading": {
                        "valuation_asset": "USDT",
                        "position_size": 0.25,
                        "initial_balances": {"USDT": 100_000.0},
                        "default_market": {"min_quantity": 0.001, "min_notional": 10.0},
                        "ledger_directory": str(ledger_dir),
                        "ledger_filename_pattern": "ledger-%Y%m%d.jsonl",
                        "ledger_retention_days": 7,
                        "ledger_fsync": False,
                    }
                },
                "alert_throttle": {
                    "window_seconds": 60,
                    "exclude_severities": [],
                    "exclude_categories": [],
                    "max_entries": 32,
                },
                "alert_audit": {
                    "backend": "file",
                    "directory": str(cache_dir / "alerts"),
                    "filename_pattern": "alerts-%Y%m%d.jsonl",
                    "retention_days": 7,
                    "fsync": False,
                },
                "decision_journal": {
                    "backend": "file",
                    "directory": str(cache_dir / "decisions"),
                    "filename_pattern": "decisions-%Y%m%d.jsonl",
                    "retention_days": 7,
                    "fsync": False,
                },
            }
        },
        "alerts": {},
        "reporting": {},
        "sms_providers": {},
        "telegram_channels": {},
        "email_channels": {},
        "signal_channels": {},
        "whatsapp_channels": {},
        "messenger_channels": {},
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _build_secret_manager() -> SecretManager:
    storage = _InMemorySecretStorage()
    manager = SecretManager(storage, namespace="testsuite")
    manager.store_exchange_credentials(
        "binance_fixture_key",
        ExchangeCredentials(
            key_id="fixture",
            secret="secret",
            environment=Environment.PAPER,
            permissions=("read", "trade"),
        ),
    )
    return manager


def _date_bounds() -> tuple[int, int]:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 8, tzinfo=timezone.utc)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def test_daily_trend_pipeline_smoke(tmp_path: Path, _fixture_cache: Path) -> None:
    ledger_dir = tmp_path / "ledger"
    config_path = tmp_path / "core_smoke.yaml"
    _write_config(config_path, cache_dir=_fixture_cache, ledger_dir=ledger_dir)
    secret_manager = _build_secret_manager()

    # Pierwsze uruchomienie: weryfikacja, że generowane są sygnały strategii.
    pipeline_for_signals = build_daily_trend_pipeline(
        environment_name="binance_smoke",
        strategy_name="smoke_daily_trend",
        controller_name="daily_trend_smoke",
        config_path=config_path,
        secret_manager=secret_manager,
        adapter_factories={"binance_spot": lambda credentials, **kwargs: _FixtureExchangeAdapter(credentials, **kwargs)},
    )
    start_ms, end_ms = _date_bounds()
    collected = pipeline_for_signals.controller.collect_signals(start=start_ms, end=end_ms)
    assert collected, "Strategia powinna wygenerować co najmniej jeden sygnał."

    # Drugie uruchomienie: pełny cykl z egzekucją i ledgerem.
    pipeline = build_daily_trend_pipeline(
        environment_name="binance_smoke",
        strategy_name="smoke_daily_trend",
        controller_name="daily_trend_smoke",
        config_path=config_path,
        secret_manager=secret_manager,
        adapter_factories={"binance_spot": lambda credentials, **kwargs: _FixtureExchangeAdapter(credentials, **kwargs)},
    )

    results = pipeline.controller.run_cycle(start=start_ms, end=end_ms)
    assert results, "Kontroler powinien zwrócić zrealizowane zlecenia."
    for result in results:
        assert result.status.lower() == "filled"

    ledger_entries = list(pipeline.execution_service.ledger())
    assert ledger_entries, "W ledgerze powinny znaleźć się wpisy z symulacji."
    ledger_files = list(pipeline.execution_service.ledger_files())
    assert ledger_files and ledger_files[0].exists()

    profile_name = pipeline.bootstrap.environment.risk_profile
    risk_snapshot = pipeline.bootstrap.risk_engine.snapshot_state(profile_name)
    assert risk_snapshot is not None
    assert not bool(risk_snapshot.get("force_liquidation")), (
        "Profil ryzyka nie powinien przejść w tryb awaryjny."
    )
