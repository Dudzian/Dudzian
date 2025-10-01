from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.data.ohlcv import ParquetCacheStorage, SQLiteCacheStorage
from scripts import check_data_coverage


_COLUMNS = ("open_time", "open", "high", "low", "close", "volume")


def _generate_rows(start: datetime, count: int) -> list[list[float]]:
    base_ts = int(start.timestamp() * 1000)
    day_ms = 24 * 60 * 60 * 1000
    rows: list[list[float]] = []
    price = 40_000.0
    for index in range(count):
        timestamp = base_ts + index * day_ms
        rows.append([
            float(timestamp),
            price + index * 10,
            price + index * 15,
            price + index * 5,
            price + index * 12,
            100 + index,
        ])
    return rows


def _write_cache(cache_dir: Path, rows: list[list[float]], *, symbol: str = "BTCUSDT") -> None:
    payload = {"columns": _COLUMNS, "rows": rows}
    parquet = ParquetCacheStorage(cache_dir / "ohlcv_parquet", namespace="binance_spot")
    parquet.write(f"{symbol}::1d", payload)
    manifest = SQLiteCacheStorage(cache_dir / "ohlcv_manifest.sqlite", store_rows=False)
    manifest.write(f"{symbol}::1d", payload)


def _write_config(
    tmp_path: Path,
    cache_dir: Path,
    instruments: dict[str, str] | None = None,
) -> Path:
    instruments = instruments or {"BTC_USDT": "BTCUSDT"}

    universe_instruments: dict[str, dict[str, object]] = {}
    for name, symbol in instruments.items():
        base, quote = name.split("_", 1)
        universe_instruments[name] = {
            "base_asset": base,
            "quote_asset": quote,
            "categories": ["smoke"],
            "exchanges": {"binance_spot": symbol},
            "backfill": [
                {"interval": "1d", "lookback_days": 30},
            ],
        }

    content = {
        "risk_profiles": {
            "test_profile": {
                "max_daily_loss_pct": 0.5,
                "max_position_pct": 1.0,
                "target_volatility": 0.5,
                "max_leverage": 2.0,
                "stop_loss_atr_multiple": 1.5,
                "max_open_positions": 10,
                "hard_drawdown_pct": 0.8,
            }
        },
        "instrument_universes": {
            "test_universe": {
                "description": "smoke",
                "instruments": universe_instruments,
            }
        },
        "environments": {
            "binance_smoke": {
                "exchange": "binance_spot",
                "environment": "paper",
                "keychain_key": "smoke_key",
                "data_cache_path": str(cache_dir),
                "risk_profile": "test_profile",
                "alert_channels": [],
                "instrument_universe": "test_universe",
                "required_permissions": ["read"],
                "forbidden_permissions": [],
            }
        },
    }

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(content, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_check_data_coverage_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache_dir = tmp_path / "cache_success"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 40)
    _write_cache(cache_dir, rows)
    config_path = _write_config(tmp_path, cache_dir)

    as_of = datetime.fromtimestamp(rows[-1][0] / 1000, tz=timezone.utc).isoformat()
    exit_code = check_data_coverage.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_smoke",
            "--as-of",
            as_of,
            "--json",
        ]
    )
    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "ok"
    assert output["entries"][0]["required_rows"] >= 30


def test_check_data_coverage_writes_output_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache_dir = tmp_path / "cache_output"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 40)
    _write_cache(cache_dir, rows)
    config_path = _write_config(tmp_path, cache_dir)

    output_path = tmp_path / "reports" / "coverage.json"
    as_of = datetime.fromtimestamp(rows[-1][0] / 1000, tz=timezone.utc).isoformat()
    exit_code = check_data_coverage.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_smoke",
            "--as-of",
            as_of,
            "--output",
            str(output_path),
        ]
    )
    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Manifest:" in stdout  # tryb tekstowy nadal działa
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["manifest_path"].endswith("ohlcv_manifest.sqlite")


def test_check_data_coverage_detects_insufficient_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_dir = tmp_path / "cache_failure"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 5)
    _write_cache(cache_dir, rows)
    config_path = _write_config(tmp_path, cache_dir)

    as_of = datetime.fromtimestamp(rows[-1][0] / 1000, tz=timezone.utc).isoformat()
    exit_code = check_data_coverage.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_smoke",
            "--as-of",
            as_of,
            "--json",
        ]
    )
    assert exit_code == 1
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "error"
    assert any("insufficient_rows" in issue for issue in output["issues"])


def test_check_data_coverage_filters_symbols(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_dir = tmp_path / "cache_filter"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 40)
    _write_cache(cache_dir, rows, symbol="BTCUSDT")
    _write_cache(cache_dir, rows, symbol="ETHUSDT")
    instruments = {"BTC_USDT": "BTCUSDT", "ETH_USDT": "ETHUSDT"}
    config_path = _write_config(tmp_path, cache_dir, instruments=instruments)

    as_of = datetime.fromtimestamp(rows[-1][0] / 1000, tz=timezone.utc).isoformat()
    exit_code = check_data_coverage.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_smoke",
            "--as-of",
            as_of,
            "--json",
            "--symbol",
            "BTC_USDT",
        ]
    )
    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert {entry["symbol"] for entry in output["entries"]} == {"BTCUSDT"}


def test_check_data_coverage_unknown_symbol(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_dir = tmp_path / "cache_unknown"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 40)
    _write_cache(cache_dir, rows, symbol="BTCUSDT")
    config_path = _write_config(tmp_path, cache_dir)

    exit_code = check_data_coverage.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_smoke",
            "--symbol",
            "DOGE_USDT",
        ]
    )
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Nieznane symbole" in captured.err
