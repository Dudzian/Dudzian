from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bot_core.config.models import EnvironmentConfig, EnvironmentReportStorageConfig
from bot_core.exchanges.base import Environment
from bot_core.reporting.environment_storage import store_environment_report


@pytest.fixture
def environment_cfg(tmp_path: Path) -> EnvironmentConfig:
    data_dir = tmp_path / "cache"
    data_dir.mkdir()
    return EnvironmentConfig(
        name="coinbase_offline",
        exchange="coinbase_spot",
        environment=Environment.PAPER,
        keychain_key="offline",
        data_cache_path=str(data_dir),
        risk_profile="manual",
        alert_channels=(),
    )


def test_store_environment_report_creates_copy(environment_cfg: EnvironmentConfig, tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{\n  \"ok\": true\n}\n", encoding="utf-8")

    environment_cfg.report_storage = EnvironmentReportStorageConfig(
        backend="file",
        directory="reports",
        filename_pattern="report-%Y%m%d.json",
        retention_days=30,
        fsync=True,
    )

    timestamp = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    stored = store_environment_report(summary_path, environment_cfg, now=timestamp)

    expected_path = Path(environment_cfg.data_cache_path) / "reports" / "report-20240102.json"
    assert stored == expected_path
    assert expected_path.exists()
    assert expected_path.read_text(encoding="utf-8") == summary_path.read_text(encoding="utf-8")


def test_store_environment_report_prunes_old_files(environment_cfg: EnvironmentConfig, tmp_path: Path) -> None:
    target_dir = Path(environment_cfg.data_cache_path) / "reports"
    target_dir.mkdir(parents=True, exist_ok=True)
    old_report = target_dir / "report-20231231.json"
    old_report.write_text("{}\n", encoding="utf-8")
    old_time = datetime(2023, 12, 31, tzinfo=timezone.utc)
    os.utime(old_report, (old_time.timestamp(), old_time.timestamp()))

    environment_cfg.report_storage = EnvironmentReportStorageConfig(
        backend="file",
        directory="reports",
        filename_pattern="report-%Y%m%d.json",
        retention_days=5,
        fsync=False,
    )

    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{\n  \"ok\": true\n}\n", encoding="utf-8")
    timestamp = datetime(2024, 1, 10, tzinfo=timezone.utc)

    stored = store_environment_report(summary_path, environment_cfg, now=timestamp)

    assert stored is not None
    assert not old_report.exists()


def test_store_environment_report_unsupported_backend(environment_cfg: EnvironmentConfig, tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{}\n", encoding="utf-8")
    environment_cfg.report_storage = EnvironmentReportStorageConfig(backend="s3")

    with pytest.raises(ValueError):
        store_environment_report(summary_path, environment_cfg, now=datetime.now(timezone.utc))


def test_store_environment_report_no_config_returns_none(environment_cfg: EnvironmentConfig, tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{}\n", encoding="utf-8")

    assert store_environment_report(summary_path, environment_cfg, now=datetime.now(timezone.utc)) is None
