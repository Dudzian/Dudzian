"""Testy skryptu check_data_coverage oraz współdzielone helpery CLI."""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import check_data_coverage as cli  # noqa: E402  - import po modyfikacji sys.path


def _last_row_iso(rows: Sequence[Sequence[float]]) -> str:
    timestamp_ms = int(float(rows[-1][0]))
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


def test_check_data_coverage_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache_dir = tmp_path / "cache_success"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 40)
    _write_cache(cache_dir, rows)
    config_path = _write_config(tmp_path, cache_dir)

    exit_code = cli.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_smoke",
            "--as-of",
            _last_row_iso(rows),
            "--json",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "ok"
    summary = payload["summary"]
    assert summary["status"] == "ok"
    assert summary["total"] == 1
    assert summary["ok"] == 1
    assert summary["issue_counts"] == {}
    assert summary["issue_examples"] == {}
    assert summary["stale_entries"] == 0


def test_check_data_coverage_insufficient_rows(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache_dir = tmp_path / "cache_failure"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 5)
    _write_cache(cache_dir, rows)
    config_path = _write_config(
        tmp_path,
        cache_dir,
        backfill={"BTC_USDT": [{"interval": "1d", "lookback_days": 60}]},
    )

    exit_code = cli.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_smoke",
            "--as-of",
            _last_row_iso(rows),
            "--json",
        ]
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "error"
    summary = payload["summary"]
    assert summary["status"] == "error"
    assert summary["total"] == 1
    assert summary["error"] == 1
    assert "insufficient_rows" in summary["issue_counts"]
    assert summary["issue_counts"]["insufficient_rows"] == 1
    assert "insufficient_rows" in summary["issue_examples"]
    issues = payload["issues"]
    assert any("insufficient_rows" in issue for issue in issues)


def test_check_data_coverage_interval_and_symbol_filters(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_dir = tmp_path / "cache_filters"
    rows_daily = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 45, interval="1d")
    rows_hourly = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 12, interval="1h")
    _write_cache(cache_dir, rows_daily, interval="1d")
    _write_cache(cache_dir, rows_hourly, interval="1h")
    config_path = _write_config(
        tmp_path,
        cache_dir,
        backfill={
            "BTC_USDT": [
                {"interval": "1d", "lookback_days": 30},
                {"interval": "1h", "lookback_days": 24},
            ]
        },
    )

    as_of_iso = _last_row_iso(rows_daily if rows_daily[-1][0] >= rows_hourly[-1][0] else rows_hourly)
    exit_code = cli.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_smoke",
            "--symbol",
            "BTC_USDT",
            "--interval",
            "1d",
            "--as-of",
            as_of_iso,
            "--json",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "ok"
    entries = payload["entries"]
    assert len(entries) == 1
    assert entries[0]["symbol"] == "BTCUSDT"
    assert entries[0]["interval"] == "1d"
    summary = payload["summary"]
    assert summary["total"] == 1
    assert summary["status"] == "ok"


# --- Współdzielone helpery dla scenariuszy CLI ---


def _generate_rows(start: datetime, count: int, *, interval: str = "1d") -> list[list[float]]:
    """Buduje syntetyczne świece OHLCV dla testów CLI."""

    if count < 0:
        raise ValueError("count musi być nieujemny")

    step_map: Mapping[str, timedelta] = {
        "1d": timedelta(days=1),
        "1h": timedelta(hours=1),
        "15m": timedelta(minutes=15),
    }
    try:
        step = step_map[interval]
    except KeyError as exc:  # pragma: no cover - nie używane w obecnych testach
        raise ValueError("Nieobsługiwany interwał testowy: {interval}".format(interval=interval)) from exc

    rows: list[list[float]] = []
    current = start.astimezone(timezone.utc)
    price = 10_000.0
    for _ in range(count):
        timestamp = int(current.timestamp() * 1000)
        open_price = price
        close_price = max(0.0001, open_price * 1.001)
        high_price = max(open_price, close_price) * 1.001
        low_price = min(open_price, close_price) * 0.999
        volume = 100.0 + len(rows)
        rows.append(
            [
                float(timestamp),
                float(round(open_price, 6)),
                float(round(high_price, 6)),
                float(round(low_price, 6)),
                float(round(close_price, 6)),
                float(round(volume, 6)),
            ]
        )
        price = close_price
        current += step
    return rows


def _write_cache(
    cache_dir: Path,
    rows: Sequence[Sequence[float]],
    *,
    symbol: str = "BTCUSDT",
    interval: str = "1d",
) -> Path:
    """Zapisuje manifest SQLite z metadanymi wykorzystanymi w testach."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "ohlcv_manifest.sqlite"

    with sqlite3.connect(manifest_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

        if rows:
            last_ts = int(float(rows[-1][0]))
            row_count = len(rows)
            entries = {
                f"last_timestamp::{symbol}::{interval}": str(last_ts),
                f"row_count::{symbol}::{interval}": str(row_count),
            }
            for key, value in entries.items():
                connection.execute(
                    """
                    INSERT INTO metadata(key, value) VALUES(?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (key, value),
                )

    return manifest_path


def _write_config(
    tmp_path: Path,
    cache_dir: Path,
    *,
    environment_name: str = "binance_smoke",
    exchange_name: str = "binance_spot",
    instrument_name: str = "BTC_USDT",
    symbol: str = "BTCUSDT",
    backfill: Mapping[str, Sequence[Mapping[str, int]]] | None = None,
) -> Path:
    """Generuje minimalną konfigurację CoreConfig dla testów CLI."""

    universe_name = "smoke_universe"
    instrument_backfill = None
    if backfill:
        instrument_backfill = backfill.get(instrument_name)
    if instrument_backfill is None:
        instrument_backfill = (
            {"interval": "1d", "lookback_days": 30},
        )

    base_asset, _, quote_asset = instrument_name.partition("_")
    if not quote_asset:
        quote_asset = "USDT"

    config_payload = {
        "risk_profiles": {
            "balanced": {
                "max_daily_loss_pct": 0.02,
                "max_position_pct": 0.05,
                "target_volatility": 0.1,
                "max_leverage": 3.0,
                "stop_loss_atr_multiple": 1.5,
                "max_open_positions": 5,
                "hard_drawdown_pct": 0.1,
            }
        },
        "instrument_universes": {
            universe_name: {
                "description": "Testowe uniwersum dla scenariuszy CLI",
                "instruments": {
                    instrument_name: {
                        "base_asset": base_asset,
                        "quote_asset": quote_asset,
                        "categories": ["core"],
                        "exchanges": {exchange_name: symbol},
                        "backfill": list(instrument_backfill),
                    }
                },
            }
        },
        "environments": {
            environment_name: {
                "exchange": exchange_name,
                "environment": "paper",
                "keychain_key": f"{exchange_name}_paper",  # fikcyjny wpis dla walidacji
                "data_cache_path": str(cache_dir),
                "risk_profile": "balanced",
                "alert_channels": [],
                "instrument_universe": universe_name,
            }
        },
    }

    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        yaml.safe_dump(config_payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return config_path
