from __future__ import annotations

from pathlib import Path

import pytest

from bot_core.config.loader import load_core_config
from scripts import list_exchange_adapters as report


def test_exchange_report_schedule_is_strict_and_fetches_real_snapshots() -> None:
    workflow = Path(".github/workflows/exchange-report.yml").read_text(encoding="utf-8")

    assert "schedule)" in workflow
    assert 'echo "LONG_POLL_MODE=strict"' in workflow
    assert "if: github.event_name == 'schedule'" in workflow
    assert "python scripts/check_required_adapter_factories.py" in workflow
    assert "python scripts/fetch_long_poll_snapshots.py" in workflow
    assert "cleanup_gateway()" in workflow
    assert "trap 'cleanup_gateway \"$gateway_pid\"'" in workflow
    assert "kill -TERM" in workflow
    assert "kill -KILL" in workflow
    assert 'wait "${pid}"' in workflow
    assert "timeout 90s python scripts/fetch_long_poll_snapshots.py" in workflow
    assert "--skip-freshness" in workflow


def test_exchange_report_dispatch_remains_fixture_mode() -> None:
    workflow = Path(".github/workflows/exchange-report.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch)" in workflow
    assert 'echo "LONG_POLL_MODE=fixture"' in workflow
    assert "--allow-stale-long-poll" in workflow

def test_retired_bitmex_stays_historical_without_live_metrics(tmp_path, monkeypatch):
    config = load_core_config(Path("config/core.yaml"))

    def long_poll(*, adapter_id, **_kwargs):
        return {
            "long_poll_metrics_status": (
                "fresh" if str(adapter_id).startswith("deribit") else "missing"
            ),
        }

    def signal_quality(*, adapter_id, **_kwargs):
        return {
            "signal_quality_snapshot_status": (
                "fresh" if str(adapter_id).startswith("deribit") else "missing"
            ),
        }

    monkeypatch.setattr(report, "_summarize_long_poll_metrics", long_poll)
    monkeypatch.setattr(report, "_summarize_signal_quality", signal_quality)
    rows = report.build_rows(config, signal_quality_dir=tmp_path, active_only=True)
    historical = [row for row in rows if row["exchange"] == "bitmex"]
    assert historical, "historical BitMEX rows must not disappear"
    assert all(row["long_poll_metrics_status"] == "retired" for row in historical)
    assert all(row["signal_quality_snapshot_status"] == "retired" for row in historical)
    assert all(row["long_poll_snapshot_age_minutes"] is None for row in historical)
    assert all(row["signal_quality_snapshot_path"] is None for row in historical)

    with pytest.raises(RuntimeError, match="bitmex"):
        report.build_rows(config, signal_quality_dir=tmp_path, active_only=False)


def test_active_deribit_still_fails_closed_when_metrics_missing(tmp_path, monkeypatch):
    config = load_core_config(Path("config/core.yaml"))
    monkeypatch.setattr(
        report,
        "_summarize_long_poll_metrics",
        lambda **_kwargs: {"long_poll_metrics_status": "missing"},
    )
    monkeypatch.setattr(
        report,
        "_summarize_signal_quality",
        lambda **_kwargs: {"signal_quality_snapshot_status": "missing"},
    )
    with pytest.raises(RuntimeError, match="deribit"):
        report.build_rows(config, signal_quality_dir=tmp_path, active_only=True)
