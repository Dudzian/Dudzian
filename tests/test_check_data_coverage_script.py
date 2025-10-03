from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.data.ohlcv import ParquetCacheStorage, SQLiteCacheStorage
from scripts import check_data_coverage


_COLUMNS = ("open_time", "open", "high", "low", "close", "volume")


def _generate_rows(start: datetime, count: int, *, step_minutes: int = 24 * 60) -> list[list[float]]:
    base_ts = int(start.timestamp() * 1000)
    step_ms = step_minutes * 60 * 1000
    rows: list[list[float]] = []
    price = 40_000.0
    for index in range(count):
        timestamp = base_ts + index * step_ms
        rows.append([
            float(timestamp),
            price + index * 10,
            price + index * 15,
            price + index * 5,
            price + index * 12,
            100 + index,
        ])
    return rows


def _write_cache(
    cache_dir: Path,
    rows: list[list[float]],
    *,
    symbol: str = "BTCUSDT",
    interval: str = "1d",
) -> None:
    payload = {"columns": _COLUMNS, "rows": rows}
    parquet = ParquetCacheStorage(cache_dir / "ohlcv_parquet", namespace="binance_spot")
    parquet.write(f"{symbol}::{interval}", payload)
    manifest = SQLiteCacheStorage(cache_dir / "ohlcv_manifest.sqlite", store_rows=False)
    manifest.write(f"{symbol}::{interval}", payload)


def _write_config(
    tmp_path: Path,
    cache_dir: Path,
    instruments: dict[str, str] | None = None,
    *,
    backfill: dict[str, list[dict[str, object]]] | None = None,
    data_quality: dict[str, object] | None = None,
) -> Path:
    instruments = instruments or {"BTC_USDT": "BTCUSDT"}
    backfill = backfill or {}

    universe_instruments: dict[str, dict[str, object]] = {}
    for name, symbol in instruments.items():
        base, quote = name.split("_", 1)
        universe_instruments[name] = {
            "base_asset": base,
            "quote_asset": quote,
            "categories": ["smoke"],
            "exchanges": {"binance_spot": symbol},
            "backfill": backfill.get(
                name,
                [
                    {"interval": "1d", "lookback_days": 30},
                ],
            ),
        }

    environment_entry: dict[str, object] = {
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
    if data_quality:
        environment_entry["data_quality"] = data_quality

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
            "binance_smoke": environment_entry,
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
    summary = output["summary"]
    assert summary["total"] >= 1
    assert summary["status"] == "ok"
    assert summary.get("ok_ratio") == pytest.approx(1.0)
    interval_bucket = summary["by_interval"]["1d"]
    assert interval_bucket["ok"] >= 1
    assert interval_bucket["status"] == "ok"
    assert interval_bucket["manifest_status_counts"]["ok"] >= 1
    symbol_bucket = summary["by_symbol"]["BTCUSDT"]
    assert symbol_bucket["ok"] >= 1
    assert symbol_bucket["status"] == "ok"
    assert symbol_bucket["manifest_status_counts"]["ok"] >= 1


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
    summary = payload["summary"]
    assert summary["total"] == len(payload["entries"])
    assert summary["status"] == "ok"
    interval_bucket = summary["by_interval"]["1d"]
    assert interval_bucket["ok"] >= 1
    assert interval_bucket["status"] == "ok"
    assert interval_bucket["manifest_status_counts"]["ok"] >= 1
    symbol_bucket = summary["by_symbol"]["BTCUSDT"]
    assert symbol_bucket["ok"] >= 1
    assert symbol_bucket["status"] == "ok"
    assert symbol_bucket["manifest_status_counts"]["ok"] >= 1


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
    summary = output["summary"]
    assert summary["error"] >= 1
    assert summary["status"] == "error"
    assert summary.get("ok_ratio", 0.0) < 1.0


def test_check_data_coverage_max_gap_threshold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_dir = tmp_path / "cache_gap"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 40)
    _write_cache(cache_dir, rows)
    config_path = _write_config(tmp_path, cache_dir)

    last_dt = datetime.fromtimestamp(rows[-1][0] / 1000, tz=timezone.utc)
    as_of = (last_dt + timedelta(minutes=30)).isoformat()

    exit_code = check_data_coverage.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_smoke",
            "--as-of",
            as_of,
            "--max-gap-minutes",
            "10",
            "--json",
        ]
    )
    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert any("max_gap_exceeded" in issue for issue in payload["issues"])
    worst_gap = payload["summary"].get("worst_gap")
    assert isinstance(worst_gap, dict)
    assert worst_gap.get("gap_minutes", 0) > 10
    thresholds = payload.get("thresholds", {})
    assert pytest.approx(thresholds.get("max_gap_minutes", 0.0), rel=1e-6) == 10.0


def test_check_data_coverage_min_ok_ratio_threshold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_dir = tmp_path / "cache_ratio"
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
            "--min-ok-ratio",
            "0.95",
            "--json",
        ]
    )
    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert any("ok_ratio_below_threshold" in issue for issue in payload["issues"])
    thresholds = payload.get("thresholds", {})
    assert pytest.approx(thresholds.get("min_ok_ratio", 0.0), rel=1e-6) == 0.95


def test_check_data_coverage_min_ok_ratio_environment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_dir = tmp_path / "cache_ratio_env"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 5)
    _write_cache(cache_dir, rows)
    config_path = _write_config(
        tmp_path,
        cache_dir,
        data_quality={"min_ok_ratio": 0.9},
    )

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
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    thresholds = payload.get("thresholds", {})
    assert pytest.approx(thresholds.get("min_ok_ratio", 0.0), rel=1e-6) == 0.9


def test_check_data_coverage_invalid_min_ok_ratio(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache_invalid_ratio"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 40)
    _write_cache(cache_dir, rows)
    config_path = _write_config(tmp_path, cache_dir)

    exit_code = check_data_coverage.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_smoke",
            "--min-ok-ratio",
            "1.5",
        ]
    )
    assert exit_code == 2


def test_check_data_coverage_uses_environment_threshold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_dir = tmp_path / "cache_env_gap"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 40)
    _write_cache(cache_dir, rows)
    config_path = _write_config(
        tmp_path,
        cache_dir,
        data_quality={"max_gap_minutes": 60, "min_ok_ratio": 0.8},
    )

    last_dt = datetime.fromtimestamp(rows[-1][0] / 1000, tz=timezone.utc)
    as_of = (last_dt + timedelta(days=2)).isoformat()

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
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    thresholds = payload.get("thresholds", {})
    assert pytest.approx(thresholds.get("max_gap_minutes", 0.0), rel=1e-6) == 60.0
    assert pytest.approx(thresholds.get("min_ok_ratio", 0.0), rel=1e-6) == 0.8
    summary_thresholds = payload["summary"].get("thresholds", {})
    assert pytest.approx(summary_thresholds.get("max_gap_minutes", 0.0), rel=1e-6) == 60.0
    assert pytest.approx(summary_thresholds.get("min_ok_ratio", 0.0), rel=1e-6) == 0.8
    assert any(issue.startswith("max_gap_exceeded") for issue in payload["issues"])


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


def test_check_data_coverage_filters_intervals(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_dir = tmp_path / "cache_interval"
    daily_rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 40)
    hourly_rows = _generate_rows(
        datetime(2024, 1, 5, tzinfo=timezone.utc),
        120,
        step_minutes=60,
    )
    _write_cache(cache_dir, daily_rows, symbol="BTCUSDT", interval="1d")
    _write_cache(cache_dir, hourly_rows, symbol="BTCUSDT", interval="1h")

    config_path = _write_config(
        tmp_path,
        cache_dir,
        backfill={
            "BTC_USDT": [
                {"interval": "1d", "lookback_days": 30},
                {"interval": "1h", "lookback_days": 2},
            ]
        },
    )

    latest_hourly = hourly_rows[-1][0]
    as_of = datetime.fromtimestamp(latest_hourly / 1000, tz=timezone.utc).isoformat()
    exit_code = check_data_coverage.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_smoke",
            "--as-of",
            as_of,
            "--json",
            "--interval",
            "1h",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert {entry["interval"] for entry in output["entries"]} == {"1h"}


def test_check_data_coverage_interval_alias(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_dir = tmp_path / "cache_interval_alias"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 40)
    _write_cache(cache_dir, rows, interval="1d")
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
            "--interval",
            "D1",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert {entry["interval"] for entry in output["entries"]} == {"1d"}


def test_check_data_coverage_unknown_interval(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_dir = tmp_path / "cache_interval_unknown"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 40)
    _write_cache(cache_dir, rows, interval="1d")
    config_path = _write_config(tmp_path, cache_dir)

    exit_code = check_data_coverage.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_smoke",
            "--interval",
            "5m",
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Brak wpisów w manifeście" in captured.err


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
