import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

sys_path_added = False

if not sys_path_added:
    import sys

    
    sys_path_added = True

from scripts import publish_paper_smoke_artifacts as publish_script  # noqa: E402


def _write_summary(report_dir: Path, *, start: str = "2024-01-01", end: str = "2024-01-31") -> None:
    summary = {
        "environment": "binance_paper",
        "window": {"start": start, "end": end},
        "orders": [],
        "ledger_entries": 0,
        "metrics": {},
        "alert_snapshot": {},
    }
    summary_path = report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")


def _build_config(tmp_path: Path, *, reporting: dict | None = None) -> Path:
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
        "environments": {
            "binance_paper": {
                "exchange": "binance_spot",
                "environment": "paper",
                "keychain_key": "binance_paper_key",
                "data_cache_path": str(tmp_path / "cache"),
                "risk_profile": "balanced",
                "alert_channels": [],
            }
        },
    }
    if reporting is not None:
        config_payload["reporting"] = reporting
    config_path = tmp_path / "core.yaml"
    config_path.write_text(yaml.safe_dump(config_payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return config_path


def _write_json_log(json_log_path: Path, *, summary_sha: str, timestamp: str = "2025-01-02T03:04:05+00:00") -> None:
    record = {
        "record_id": "J-20250102T030405-deadbeef",
        "timestamp": timestamp,
        "environment": "binance_paper",
        "summary_sha256": summary_sha,
        "summary_path": "report/summary.json",
    }
    json_log_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_structured_summary(
    path: Path,
    *,
    summary_sha: str,
    report_dir: Path,
    json_log_path: Path,
    record_id: str,
    archive_path: Path | None = None,
) -> None:
    payload: dict[str, object] = {
        "environment": "binance_paper",
        "timestamp": "2025-01-02T03:04:05+00:00",
        "operator": "CI Agent",
        "severity": "INFO",
        "window": {"start": "2024-01-01", "end": "2024-01-31"},
        "report": {
            "directory": str(report_dir),
            "summary_path": str(report_dir / "summary.json"),
            "summary_sha256": summary_sha,
        },
        "precheck": {
            "status": "ok",
            "coverage_status": "ok",
            "risk_status": "ok",
        },
        "json_log": {
            "path": str(json_log_path),
            "record_id": record_id,
            "record": {
                "record_id": record_id,
                "environment": "binance_paper",
            },
        },
    }
    if archive_path is not None:
        payload["archive"] = {"path": str(archive_path)}

    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_publish_success_local_backends(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    _write_summary(report_dir)
    summary_bytes = (report_dir / "summary.json").read_bytes()
    summary_sha = hashlib.sha256(summary_bytes).hexdigest()

    json_log_path = tmp_path / "paper_trading_log.jsonl"
    _write_json_log(json_log_path, summary_sha=summary_sha)

    archive_path = Path(shutil.make_archive(str(report_dir), "zip", root_dir=report_dir))

    json_target = tmp_path / "json_sync"
    archive_target = tmp_path / "archive_sync"
    reporting_cfg = {
        "paper_smoke_json_sync": {
            "backend": "local",
            "local": {
                "directory": str(json_target),
                "filename_pattern": "{environment}_{date}.jsonl",
                "fsync": False,
            },
        },
        "smoke_archive_upload": {
            "backend": "local",
            "local": {
                "directory": str(archive_target),
                "filename_pattern": "{environment}_{hash}.zip",
                "fsync": False,
            },
        },
    }

    config_path = _build_config(tmp_path, reporting=reporting_cfg)

    exit_code = publish_script.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_paper",
            "--report-dir",
            str(report_dir),
            "--json-log",
            str(json_log_path),
            "--archive",
            str(archive_path),
            "--json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "ok"
    assert output["json_sync"]["status"] == "ok"
    assert output["archive_upload"]["status"] == "ok"

    synced_files = list(json_target.glob("*.jsonl"))
    assert len(synced_files) == 1
    assert synced_files[0].read_text(encoding="utf-8").strip() != ""

    archived_files = list(archive_target.glob("*.zip"))
    assert len(archived_files) == 1
    assert archived_files[0].is_file()


def test_missing_reporting_config_skips_operations(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    _write_summary(report_dir)
    summary_bytes = (report_dir / "summary.json").read_bytes()
    summary_sha = hashlib.sha256(summary_bytes).hexdigest()

    json_log_path = tmp_path / "paper_trading_log.jsonl"
    _write_json_log(json_log_path, summary_sha=summary_sha)

    config_path = _build_config(tmp_path, reporting=None)

    exit_code = publish_script.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_paper",
            "--report-dir",
            str(report_dir),
            "--json-log",
            str(json_log_path),
            "--json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "skipped"
    assert output["json_sync"]["status"] == "skipped"
    assert output["archive_upload"]["status"] == "skipped"


def test_missing_record_returns_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    _write_summary(report_dir)

    json_log_path = tmp_path / "paper_trading_log.jsonl"
    json_log_path.write_text("", encoding="utf-8")

    json_target = tmp_path / "json_sync"
    archive_target = tmp_path / "archive_sync"
    reporting_cfg = {
        "paper_smoke_json_sync": {
            "backend": "local",
            "local": {
                "directory": str(json_target),
                "filename_pattern": "{environment}_{date}.jsonl",
            },
        },
        "smoke_archive_upload": {
            "backend": "local",
            "local": {
                "directory": str(archive_target),
                "filename_pattern": "{environment}_{hash}.zip",
            },
        },
    }
    config_path = _build_config(tmp_path, reporting=reporting_cfg)

    exit_code = publish_script.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_paper",
            "--report-dir",
            str(report_dir),
            "--json-log",
            str(json_log_path),
            "--json",
        ]
    )

    assert exit_code != 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "error"
    assert output["json_sync"]["status"] == "error"
    assert output["json_sync"]["reason"] == "missing_record"


def test_publish_uses_structured_summary_defaults(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    _write_summary(report_dir)
    summary_bytes = (report_dir / "summary.json").read_bytes()
    summary_sha = hashlib.sha256(summary_bytes).hexdigest()

    json_log_path = tmp_path / "paper_trading_log.jsonl"
    _write_json_log(json_log_path, summary_sha=summary_sha)

    summary_payload_path = tmp_path / "smoke_summary.json"
    _write_structured_summary(
        summary_payload_path,
        summary_sha=summary_sha,
        report_dir=report_dir,
        json_log_path=json_log_path,
        record_id="J-20250102T030405-deadbeef",
    )

    json_target = tmp_path / "json_sync"
    archive_target = tmp_path / "archive_sync"
    reporting_cfg = {
        "paper_smoke_json_sync": {
            "backend": "local",
            "local": {
                "directory": str(json_target),
                "filename_pattern": "{environment}_{date}.jsonl",
                "fsync": False,
            },
        },
        "smoke_archive_upload": {
            "backend": "local",
            "local": {
                "directory": str(archive_target),
                "filename_pattern": "{environment}_{hash}.zip",
                "fsync": False,
            },
        },
    }

    config_path = _build_config(tmp_path, reporting=reporting_cfg)

    exit_code = publish_script.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_paper",
            "--report-dir",
            str(report_dir),
            "--json-log",
            str(tmp_path / "unused.jsonl"),
            "--summary-json",
            str(summary_payload_path),
            "--json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "ok"
    assert output["json_sync"]["status"] == "ok"
    assert output["json_sync"].get("record_id") == "J-20250102T030405-deadbeef"


def test_publish_errors_on_summary_hash_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    _write_summary(report_dir)
    summary_bytes = (report_dir / "summary.json").read_bytes()
    summary_sha = hashlib.sha256(summary_bytes).hexdigest()

    json_log_path = tmp_path / "paper_trading_log.jsonl"
    _write_json_log(json_log_path, summary_sha=summary_sha)

    summary_payload_path = tmp_path / "smoke_summary.json"
    _write_structured_summary(
        summary_payload_path,
        summary_sha="deadbeef",
        report_dir=report_dir,
        json_log_path=json_log_path,
        record_id="J-20250102T030405-deadbeef",
    )

    config_path = _build_config(tmp_path, reporting=None)

    exit_code = publish_script.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_paper",
            "--report-dir",
            str(report_dir),
            "--json-log",
            str(json_log_path),
            "--summary-json",
            str(summary_payload_path),
            "--json",
        ]
    )

    assert exit_code != 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "error"
    assert output["json_sync"]["reason"] == "summary_sha_mismatch"
    assert output["archive_upload"]["reason"] == "summary_sha_mismatch"
