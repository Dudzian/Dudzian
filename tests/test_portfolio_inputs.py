from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bot_core.runtime.portfolio_inputs import (
    build_slo_status_provider,
    build_stress_override_provider,
    load_slo_statuses,
    load_stress_overrides,
)


def _sample_slo_report(now: datetime) -> dict[str, object]:
    return {
        "results": {
            "latency": {
                "indicator": "router_latency_ms",
                "value": 275.5,
                "target": 250.0,
                "comparison": "<=",
                "status": "breach",
                "severity": "critical",
                "warning_threshold": 220.0,
                "sample_size": 5000,
                "window_start": (now - timedelta(minutes=30)).isoformat(),
                "window_end": now.isoformat(),
                "reason": "latency spike",
                "metadata": {"latency_p95": 275.5},
            },
            "availability": {
                "indicator": "router_availability_pct",
                "value": 99.2,
                "target": 99.0,
                "comparison": ">=",
                "status": "ok",
                "severity": "info",
                "warning_threshold": 99.5,
                "sample_size": 86400,
            },
        }
    }


def _sample_stress_report() -> dict[str, object]:
    return {
        "schema": "stage6.risk.stress_lab.report",
        "overrides": [
            {
                "severity": "critical",
                "reason": "latency_spike",
                "symbol": "BTC_USDT",
                "weight_multiplier": 0.4,
                "force_rebalance": True,
                "tags": ["stage6", "latency"],
            },
            {
                "severity": "warning",
                "reason": "global_budget",
                "risk_budget": "balanced",
                "min_weight": 0.05,
            },
        ],
    }


def test_load_slo_statuses_parses_payload(tmp_path: Path) -> None:
    now = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    report_path = tmp_path / "slo_report.json"
    report_path.write_text(json.dumps(_sample_slo_report(now)), encoding="utf-8")

    statuses = load_slo_statuses(report_path)
    assert set(statuses) == {"latency", "availability"}
    latency = statuses["latency"]
    assert latency.status == "breach"
    assert latency.severity == "critical"
    assert latency.metadata["latency_p95"] == pytest.approx(275.5)
    assert latency.window_end == datetime.fromisoformat(now.isoformat())


def test_load_slo_statuses_ignores_stale(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING")
    now = datetime.now(timezone.utc)
    report_path = tmp_path / "slo_report.json"
    report_path.write_text(json.dumps(_sample_slo_report(now)), encoding="utf-8")
    old_ts = (now - timedelta(hours=6)).timestamp()
    os.utime(report_path, (old_ts, old_ts))

    statuses = load_slo_statuses(report_path, max_age=timedelta(minutes=30))
    assert statuses == {}
    assert "przestarzały" in caplog.text


def test_load_stress_overrides_parses_entries(tmp_path: Path) -> None:
    report_path = tmp_path / "stress_lab.json"
    report_path.write_text(json.dumps(_sample_stress_report()), encoding="utf-8")

    overrides = load_stress_overrides(report_path)
    assert len(overrides) == 2
    critical = overrides[0]
    assert critical.symbol == "BTC_USDT"
    assert critical.force_rebalance is True
    assert critical.tags == ("stage6", "latency")


def test_build_slo_status_provider_uses_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fallback_dir = tmp_path / "fallback"
    fallback_dir.mkdir()
    now = datetime(2024, 1, 5, tzinfo=timezone.utc)
    (fallback_dir / "slo_report.json").write_text(
        json.dumps(_sample_slo_report(now)),
        encoding="utf-8",
    )

    provider = build_slo_status_provider(
        "slo_report.json",
        fallback_directories=[fallback_dir],
    )

    monkeypatch.chdir(tmp_path)
    statuses = provider()
    assert statuses and "latency" in statuses


def test_build_stress_override_provider_returns_latest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fallback_dir = tmp_path / "stage6"
    fallback_dir.mkdir()
    (fallback_dir / "stress.json").write_text(
        json.dumps(_sample_stress_report()),
        encoding="utf-8",
    )

    provider = build_stress_override_provider(
        "stress.json",
        fallback_directories=[fallback_dir],
        max_age=timedelta(hours=1),
    )

    monkeypatch.chdir(tmp_path)
    overrides = provider()
    assert overrides and overrides[0].reason == "latency_spike"

