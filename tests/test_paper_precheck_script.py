from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from scripts import paper_precheck
from tests.test_check_data_coverage_script import (
    _generate_rows,
    _write_cache,
    _write_config,
)


def test_paper_precheck_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache_dir = tmp_path / "cache_success"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 40)
    _write_cache(cache_dir, rows)
    config_path = _write_config(tmp_path, cache_dir)

    exit_code = paper_precheck.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_smoke",
            "--as-of",
            datetime.fromtimestamp(rows[-1][0] / 1000, tz=timezone.utc).isoformat(),
            "--json",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["config"]["valid"] is True
    assert payload["coverage_status"] == "ok"
    assert payload["risk_status"] == "ok"
    summary = payload["coverage"]["summary"]
    assert summary["status"] == "ok"
    assert summary.get("ok_ratio") == pytest.approx(1.0)
    assert summary["stale_entries"] == 0
    assert summary["issue_counts"] == {}
    assert summary["issue_examples"] == {}
    risk = payload["risk"]
    assert risk["status"] == "ok"
    assert risk["issues"] == []
    assert risk["warnings"] == []
    checks = risk["checks"]
    assert checks["tight_stop_rejected"] is True
    assert checks["wide_stop_allowed"] is True
    assert checks["oversized_blocked"] is True
    observations = risk["observations"]
    assert observations["allowed_quantity"] > 0


def test_paper_precheck_invalid_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache_dir = tmp_path / "cache_invalid_config"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 40)
    _write_cache(cache_dir, rows)
    config_path = _write_config(tmp_path, cache_dir)

    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_data["environments"]["binance_smoke"]["risk_profile"] = "missing_profile"
    config_path.write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    exit_code = paper_precheck.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_smoke",
            "--json",
        ]
    )
    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert payload["config"]["valid"] is False
    assert payload["risk_status"] == "error"
    assert "profile_not_defined" in payload["risk"]["issues"]


def test_paper_precheck_coverage_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache_dir = tmp_path / "cache_failure"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 5)
    _write_cache(cache_dir, rows)
    config_path = _write_config(
        tmp_path,
        cache_dir,
        backfill={"BTC_USDT": [{"interval": "1d", "lookback_days": 60}]},
    )

    exit_code = paper_precheck.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_smoke",
            "--as-of",
            datetime.fromtimestamp(rows[-1][0] / 1000, tz=timezone.utc).isoformat(),
            "--json",
        ]
    )
    assert exit_code == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert payload["coverage_status"] == "error"
    assert payload["risk_status"] == "ok"
    issues = payload["coverage"]["issues"]
    assert any("insufficient_rows" in issue for issue in issues)


def test_paper_precheck_manifest_missing_warning(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache_dir = tmp_path / "cache_missing"
    cache_dir.mkdir()
    config_path = _write_config(tmp_path, cache_dir)

    exit_code = paper_precheck.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_smoke",
            "--json",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "warning"
    assert payload["coverage_status"] == "skipped"
    assert "manifest_missing" in payload["coverage_warnings"]
    assert payload["risk_status"] == "ok"


def test_paper_precheck_fail_on_warnings(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache_dir = tmp_path / "cache_missing_fail"
    cache_dir.mkdir()
    config_path = _write_config(tmp_path, cache_dir)

    exit_code = paper_precheck.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_smoke",
            "--fail-on-warnings",
            "--json",
        ]
    )
    assert exit_code == 4
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert payload["coverage_status"] == "skipped"
    assert payload["risk_status"] == "ok"


def test_paper_precheck_risk_warning_when_target_vol_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_dir = tmp_path / "cache_risk_warning"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 40)
    _write_cache(cache_dir, rows)
    risk_profiles = {
        "manual_custom": {
            "max_daily_loss_pct": 0.01,
            "max_position_pct": 0.05,
            "target_volatility": 0.0,
            "max_leverage": 2.0,
            "stop_loss_atr_multiple": 1.2,
            "max_open_positions": 3,
            "hard_drawdown_pct": 0.1,
        }
    }
    config_path = _write_config(
        tmp_path,
        cache_dir,
        risk_profiles=risk_profiles,
        risk_profile_name="manual_custom",
    )

    exit_code = paper_precheck.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_smoke",
            "--as-of",
            datetime.fromtimestamp(rows[-1][0] / 1000, tz=timezone.utc).isoformat(),
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "warning"
    assert payload["risk_status"] == "warning"
    warnings = payload["risk"]["warnings"]
    assert "target_volatility_not_positive" in warnings


def test_run_precheck_returns_payload(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache_run_precheck"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 40)
    _write_cache(cache_dir, rows)
    config_path = _write_config(tmp_path, cache_dir)

    payload, exit_code = paper_precheck.run_precheck(
        environment_name="binance_smoke",
        config_path=config_path,
        as_of=datetime.fromtimestamp(rows[-1][0] / 1000, tz=timezone.utc),
    )

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["coverage_status"] == "ok"
    assert payload["risk_status"] == "ok"


def test_paper_precheck_risk_warning_when_target_vol_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_dir = tmp_path / "cache_risk_warning"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 40)
    _write_cache(cache_dir, rows)
    risk_profiles = {
        "manual_custom": {
            "max_daily_loss_pct": 0.01,
            "max_position_pct": 0.05,
            "target_volatility": 0.0,
            "max_leverage": 2.0,
            "stop_loss_atr_multiple": 1.2,
            "max_open_positions": 3,
            "hard_drawdown_pct": 0.1,
        }
    }
    config_path = _write_config(
        tmp_path,
        cache_dir,
        risk_profiles=risk_profiles,
        risk_profile_name="manual_custom",
    )

    exit_code = paper_precheck.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_smoke",
            "--as-of",
            datetime.fromtimestamp(rows[-1][0] / 1000, tz=timezone.utc).isoformat(),
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "warning"
    assert payload["risk_status"] == "warning"
    warnings = payload["risk"]["warnings"]
    assert "target_volatility_not_positive" in warnings


def test_run_precheck_returns_payload(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache_run_precheck"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 40)
    _write_cache(cache_dir, rows)
    config_path = _write_config(tmp_path, cache_dir)

    payload, exit_code = paper_precheck.run_precheck(
        environment_name="binance_smoke",
        config_path=config_path,
        as_of=datetime.fromtimestamp(rows[-1][0] / 1000, tz=timezone.utc),
    )

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["coverage_status"] == "ok"
    assert payload["risk_status"] == "ok"
