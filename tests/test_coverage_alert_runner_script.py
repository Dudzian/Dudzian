"""Testy skryptu coverage_alert_runner wykorzystującego runner alertów."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import coverage_alert_runner as cli  # noqa: E402 - import po modyfikacji sys.path
from tests.test_check_data_coverage_script import (  # noqa: E402 - współdzielone helpery
    _generate_rows,
    _last_row_iso,
    _write_cache,
    _write_config,
)

SAMPLE_ROOT = Path(__file__).resolve().parent / "assets" / "coverage_sample"
SAMPLE_CONFIG = SAMPLE_ROOT / "core.yaml"


def _prepare_sample_configuration(tmp_path: Path) -> Path:
    payload = yaml.safe_load(SAMPLE_CONFIG.read_text(encoding="utf-8"))
    cache_dir = tmp_path / "coverage_sample_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = _generate_rows(datetime(2023, 12, 22, tzinfo=timezone.utc), 40)
    _write_cache(cache_dir, rows, symbol="BTCUSDT", interval="1d")
    _write_cache(cache_dir, rows, symbol="ETHUSDT", interval="1d")

    payload["environments"]["binance_sample"]["data_cache_path"] = str(cache_dir)

    config_path = tmp_path / "coverage_sample.yaml"
    config_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def _run_cli(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, object]]:
    exit_code = cli.main(argv)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    return exit_code, payload


def test_runner_without_dispatch_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache_dir = tmp_path / "cache_runner_success"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 40)
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
    assert payload["status"] == "ok"
    assert payload["dispatch_requested"] is False
    assert payload["dispatch_enabled"] is False
    assert payload["alert_dispatched"] is False
    summary = payload["summary"]
    assert summary["status"] == "ok"
    assert summary["ok"] == 1
    assert payload["gap_statistics"]["total_entries"] == 1


def test_runner_threshold_violation_sets_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache_dir = tmp_path / "cache_runner_threshold"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 40)
    _write_cache(cache_dir, rows)
    config_path = _write_config(
        tmp_path,
        cache_dir,
        data_quality={"max_gap_minutes": 60.0},
    )

    last_iso = _last_row_iso(rows)
    future_dt = datetime.fromisoformat(last_iso) + timedelta(hours=4)

    exit_code, payload = _run_cli(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_smoke",
            "--as-of",
            future_dt.isoformat(),
            "--json",
        ],
        capsys,
    )

    assert exit_code == 1
    assert payload["status"] == "error"
    assert payload["threshold_issues"]
    assert payload["dispatch_requested"] is False
    assert payload["alert_dispatched"] is False


def test_runner_uses_risk_profile_threshold(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache_dir = tmp_path / "cache_runner_profile_threshold"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 40)
    _write_cache(cache_dir, rows)
    config_path = _write_config(
        tmp_path,
        cache_dir,
        risk_profile_data_quality={"max_gap_minutes": 45.0},
    )

    future_dt = datetime.fromisoformat(_last_row_iso(rows)) + timedelta(hours=2)

    exit_code, payload = _run_cli(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_smoke",
            "--as-of",
            future_dt.isoformat(),
            "--json",
        ],
        capsys,
    )

    assert exit_code == 1
    assert payload["status"] == "error"
    assert payload["threshold_issues"]
    thresholds = payload["threshold_evaluation"]["thresholds"]
    assert thresholds["max_gap_minutes"] == pytest.approx(45.0)


def test_runner_dispatch_flag_uses_router(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache_runner_dispatch"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 40)
    _write_cache(cache_dir, rows)
    config_path = _write_config(
        tmp_path,
        cache_dir,
        data_quality={"max_gap_minutes": 60.0},
    )

    fake_secret = object()
    monkeypatch.setattr(cli, "_create_secret_manager", lambda *_: fake_secret)

    routers: list[str] = []

    def _fake_initialize_router(*, config, environment, secret_manager):
        assert secret_manager is fake_secret
        routers.append(environment.name)
        return object()

    dispatched_payloads: list[dict[str, Any]] = []

    def _fake_dispatch(router, payload, *, severity_override=None, category="data.ohlcv") -> bool:
        assert category == "data.ohlcv"
        dispatched_payloads.append(payload)
        return True

    monkeypatch.setattr(cli, "_initialize_router", _fake_initialize_router)
    monkeypatch.setattr(cli, "dispatch_coverage_alert", _fake_dispatch)

    exit_code, payload = _run_cli(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_smoke",
            "--as-of",
            (datetime.fromisoformat(_last_row_iso(rows)) + timedelta(hours=4)).isoformat(),
            "--dispatch",
            "--json",
        ],
        capsys,
    )

    assert exit_code == 1
    assert payload["dispatch_requested"] is True
    assert payload["dispatch_enabled"] is True
    assert payload["alert_dispatched"] is True
    assert routers == ["binance_smoke"]
    assert dispatched_payloads and dispatched_payloads[0]["environment"] == "binance_smoke"


def test_runner_all_configured_targets(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache_ok = tmp_path / "cache_ok"
    cache_gap = tmp_path / "cache_gap"
    rows_ok = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 40)
    rows_gap = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 10)
    _write_cache(cache_ok, rows_ok)
    _write_cache(cache_gap, rows_gap)

    config_path = _write_config(
        tmp_path,
        cache_ok,
        extra_environments={
            "kraken_smoke": {
                "exchange": "kraken_spot",
                "environment": "paper",
                "keychain_key": "kraken_spot_paper",
                "data_cache_path": str(cache_gap),
                "risk_profile": "balanced",
                "alert_channels": [],
                "instrument_universe": "smoke_universe",
                "data_quality": {"max_gap_minutes": 60.0},
            }
        },
    )

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["instrument_universes"]["smoke_universe"]["instruments"]["BTC_USDT"]["exchanges"][
        "kraken_spot"
    ] = "BTCUSDT"
    payload["coverage_monitoring"] = {
        "enabled": True,
        "default_dispatch": False,
        "targets": [
            {"environment": "binance_smoke"},
            {
                "environment": "kraken_smoke",
                "dispatch": True,
                "category": "data.kraken",
                "severity_override": "warning",
            },
        ],
    }
    config_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")

    future_dt = datetime.fromisoformat(_last_row_iso(rows_gap)) + timedelta(hours=4)

    exit_code, output = _run_cli(
        [
            "--config",
            str(config_path),
            "--all-configured",
            "--as-of",
            future_dt.isoformat(),
            "--json",
        ],
        capsys,
    )

    assert exit_code == 1
    assert output["overall_status"] == "error"
    assert output["total_runs"] == 2
    environments = {run["environment"] for run in output["runs"]}
    assert environments == {"binance_smoke", "kraken_smoke"}
    failing = [run for run in output["runs"] if run["environment"] == "kraken_smoke"][0]
    assert failing["dispatch_requested"] is True
    assert failing["dispatch_enabled"] is False
    assert failing["threshold_issues"]
    assert output.get("suppressed_dispatch_environments") == ["kraken_smoke"]


def test_runner_all_configured_respects_dispatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache_ok = tmp_path / "cache_ok_multi"
    cache_gap = tmp_path / "cache_gap_multi"
    rows_ok = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 40)
    rows_gap = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 10)
    _write_cache(cache_ok, rows_ok)
    _write_cache(cache_gap, rows_gap)

    config_path = _write_config(
        tmp_path,
        cache_ok,
        extra_environments={
            "kraken_smoke": {
                "exchange": "kraken_spot",
                "environment": "paper",
                "keychain_key": "kraken_spot_paper",
                "data_cache_path": str(cache_gap),
                "risk_profile": "balanced",
                "alert_channels": [],
                "instrument_universe": "smoke_universe",
            }
        },
        coverage_monitoring={
            "enabled": True,
            "default_dispatch": False,
            "targets": [
                {"environment": "binance_smoke"},
                {"environment": "kraken_smoke", "dispatch": True},
            ],
        },
    )

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["instrument_universes"]["smoke_universe"]["instruments"]["BTC_USDT"]["exchanges"][
        "kraken_spot"
    ] = "BTCUSDT"
    config_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")

    fake_secret = object()
    monkeypatch.setattr(cli, "_create_secret_manager", lambda *_: fake_secret)

    routers: list[str] = []
    dispatched_envs: list[str] = []

    def _fake_initialize_router(*, config, environment, secret_manager):
        assert secret_manager is fake_secret
        routers.append(environment.name)
        return object()

    def _fake_dispatch(router, payload, *, severity_override=None, category="data.ohlcv") -> bool:
        dispatched_envs.append(payload["environment"])
        return True

    monkeypatch.setattr(cli, "_initialize_router", _fake_initialize_router)
    monkeypatch.setattr(cli, "dispatch_coverage_alert", _fake_dispatch)

    exit_code, output = _run_cli(
        [
            "--config",
            str(config_path),
            "--all-configured",
            "--dispatch",
            "--as-of",
            (datetime.fromisoformat(_last_row_iso(rows_gap)) + timedelta(hours=2)).isoformat(),
            "--json",
        ],
        capsys,
    )

    assert exit_code == 1
    assert output["overall_status"] == "error"
    assert output["total_runs"] == 2
    dispatch_matrix = {run["environment"]: run["dispatch_enabled"] for run in output["runs"]}
    assert dispatch_matrix["binance_smoke"] is False
    assert dispatch_matrix["kraken_smoke"] is True
    assert routers == ["kraken_smoke"]
    assert dispatched_envs == ["kraken_smoke"]


def test_runner_sample_assets_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = _prepare_sample_configuration(tmp_path)

    exit_code, payload = _run_cli(
        [
            "--config",
            str(config_path),
            "--all-configured",
            "--as-of",
            "2024-01-22T00:00:00+00:00",
            "--json",
        ],
        capsys,
    )

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["summary"]["status"] == "ok"
    assert payload["summary"]["ok_ratio"] == 1.0
    assert payload["dispatch_requested"] is False


def test_runner_sample_assets_threshold_violation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = _prepare_sample_configuration(tmp_path)

    exit_code, payload = _run_cli(
        [
            "--config",
            str(config_path),
            "--all-configured",
            "--as-of",
            "2024-02-05T00:00:00+00:00",
            "--json",
        ],
        capsys,
    )

    assert exit_code == 1
    assert payload["status"] == "error"
    assert payload["threshold_issues"]
    assert any(issue.startswith("max_gap_exceeded") for issue in payload["threshold_issues"])
    assert payload["summary"]["status"] in {"warning", "error"}

