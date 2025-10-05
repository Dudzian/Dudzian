import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import coverage_gap_report as cli  # noqa: E402 - import po modyfikacji sys.path
from tests.test_check_data_coverage_script import (  # noqa: E402 - wspólne helpery
    _generate_rows,
    _last_row_iso,
    _write_cache,
    _write_config,
)


def _run_cli(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, object]]:
    exit_code = cli.main(argv)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    return exit_code, payload


def test_gap_report_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache_dir = tmp_path / "gap_report_cache"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 45)
    _write_cache(cache_dir, rows)
    config_path = _write_config(tmp_path, cache_dir)

    exit_code, payload = _run_cli(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_smoke",
            "--as-of",
            _last_row_iso(rows),
            "--json",
        ],
        capsys,
    )

    assert exit_code == 0
    assert "reports" in payload
    reports = payload["reports"]
    assert len(reports) == 1
    stats = reports[0]["gap_statistics"]
    assert stats["total_entries"] == 1
    assert stats["with_gap_measurement"] == 1
    assert stats["max_gap_minutes"] >= 0


def test_gap_report_interval_breakdown(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache_dir = tmp_path / "gap_report_intervals"
    rows_daily = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 40, interval="1d")
    rows_hourly = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 24, interval="1h")
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

    exit_code, payload = _run_cli(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_smoke",
            "--as-of",
            _last_row_iso(rows_daily if rows_daily[-1][0] >= rows_hourly[-1][0] else rows_hourly),
            "--group-by-interval",
            "--json",
        ],
        capsys,
    )

    assert exit_code == 0
    reports = payload["reports"]
    interval_stats = reports[0]["gap_statistics_by_interval"]
    assert set(interval_stats.keys()) == {"1d", "1h"}
    assert interval_stats["1d"]["with_gap_measurement"] == 1


def test_gap_report_all_configured_uses_monitoring(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache_dir = tmp_path / "gap_report_all"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 30)
    _write_cache(cache_dir, rows)
    config_path = _write_config(
        tmp_path,
        cache_dir,
        coverage_monitoring={
            "targets": [
                {"environment": "binance_smoke", "dispatch": False},
            ]
        },
    )

    exit_code, payload = _run_cli(
        [
            "--config",
            str(config_path),
            "--all-configured",
            "--json",
        ],
        capsys,
    )

    assert exit_code == 0
    assert len(payload["reports"]) == 1

