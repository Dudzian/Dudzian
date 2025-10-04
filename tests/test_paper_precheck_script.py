import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

sys_path_added = False


def _ensure_sys_path() -> None:
    global sys_path_added  # noqa: PLW0603 - modyfikujemy cache w module testowym
    if sys_path_added:
        return
    sys_path_added = True
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))


_ensure_sys_path()

from scripts import paper_precheck  # noqa: E402  - import po modyfikacji sys.path
from tests.test_check_data_coverage_script import (  # noqa: E402
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
    summary = payload["coverage"]["summary"]
    assert summary["status"] == "ok"
    assert summary.get("ok_ratio") == pytest.approx(1.0)
    assert summary["stale_entries"] == 0
    assert summary["issue_counts"] == {}
    assert summary["issue_examples"] == {}


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

