from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bot_core.ai import (
    FeatureDataset,
    FeatureVector,
    WalkForwardResult,
    collect_pipeline_compliance_summary,
    save_walk_forward_report,
)
from bot_core.ai.audit import load_recent_walk_forward_reports, save_scheduler_state
from bot_core.ai.data_monitoring import (
    export_data_quality_report,
    export_drift_alert_report,
    load_recent_data_quality_reports,
    load_recent_drift_reports,
    update_sign_off,
)


def _dataset() -> FeatureDataset:
    vectors = (
        FeatureVector(timestamp=1.0, symbol="BTCUSDT", features={"x": 1.0}, target_bps=10.0),
        FeatureVector(timestamp=2.0, symbol="BTCUSDT", features={"x": 1.5}, target_bps=-4.0),
        FeatureVector(timestamp=3.0, symbol="BTCUSDT", features={"x": 2.0}, target_bps=6.0),
    )
    return FeatureDataset(vectors=vectors, metadata={"symbol": "BTCUSDT"})


def test_collect_pipeline_compliance_summary_flags_missing_sign_offs(tmp_path, monkeypatch) -> None:
    audit_root = tmp_path / "audit"
    monkeypatch.setenv("AI_DECISION_AUDIT_ROOT", str(audit_root))
    dataset = _dataset()

    export_data_quality_report(
        {
            "category": "completeness",
            "status": "alert",
            "policy": {"enforce": True},
            "issues": [
                {"code": "missing_bars", "message": "Brak danych", "severity": "critical"}
            ],
        }
    )
    export_drift_alert_report(
        {
            "model_name": "btc-usdt",
            "drift_score": 0.9,
            "threshold": 0.6,
            "window": {"start": "2024-01-01", "end": "2024-01-02"},
            "backend": "monitor",
        }
    )
    walk_result = WalkForwardResult(
        windows=(
            {"mae": 1.1, "directional_accuracy": 0.62},
            {"mae": 0.9, "directional_accuracy": 0.68},
        ),
        average_mae=1.0,
        average_directional_accuracy=0.65,
    )
    save_walk_forward_report(
        walk_result,
        job_name="btc-usdt",
        dataset=dataset,
        audit_root=audit_root,
    )

    summary = collect_pipeline_compliance_summary(audit_root=audit_root, include_scheduler=False)

    assert summary["ready"] is False
    issues = set(summary["issues"])
    assert "data_quality_alerts" in issues
    assert "drift_alerts" in issues
    assert "missing_sign_offs" in issues
    pending = summary["pending_sign_off"]
    assert len(pending["risk"]) == 3
    assert len(pending["compliance"]) == 3


def test_collect_pipeline_compliance_summary_when_ready(tmp_path, monkeypatch) -> None:
    audit_root = tmp_path / "audit"
    monkeypatch.setenv("AI_DECISION_AUDIT_ROOT", str(audit_root))
    dataset = _dataset()

    export_data_quality_report(
        {
            "category": "completeness",
            "status": "ok",
            "policy": {"enforce": False},
            "issues": [],
        }
    )
    export_drift_alert_report(
        {
            "model_name": "btc-usdt",
            "drift_score": 0.1,
            "threshold": 0.6,
            "window": {"start": "2024-01-01", "end": "2024-01-02"},
            "backend": "monitor",
        }
    )
    walk_result = WalkForwardResult(
        windows=({"mae": 0.8, "directional_accuracy": 0.7},),
        average_mae=0.8,
        average_directional_accuracy=0.7,
    )
    save_walk_forward_report(
        walk_result,
        job_name="btc-usdt",
        dataset=dataset,
        audit_root=audit_root,
    )

    dq_report = load_recent_data_quality_reports(limit=1, audit_root=audit_root)[0]
    drift_report = load_recent_drift_reports(limit=1, audit_root=audit_root)[0]
    walk_report = load_recent_walk_forward_reports(limit=1, audit_root=audit_root)[0]
    for report in (dq_report, drift_report, walk_report):
        update_sign_off(report, role="risk", status="approved", signed_by="RiskOps")
        update_sign_off(report, role="compliance", status="approved", signed_by="CompOps")

    now = datetime.now(timezone.utc)
    save_scheduler_state(
        {
            "version": 5,
            "interval": 1800,
            "last_run": (now - timedelta(minutes=10)).isoformat(),
            "next_run": (now + timedelta(minutes=20)).isoformat(),
            "updated_at": (now - timedelta(minutes=10)).isoformat(),
            "failure_streak": 0,
            "cooldown_until": None,
            "paused_until": None,
        },
        audit_root=audit_root,
    )

    summary = collect_pipeline_compliance_summary(audit_root=audit_root)

    assert summary["ready"] is True
    assert summary["issues"] == ()
    pending = summary["pending_sign_off"]
    assert pending["risk"] == ()
    assert pending["compliance"] == ()
    scheduler = summary["scheduler"]
    assert scheduler is not None
    assert scheduler["is_overdue"] is False
    assert scheduler["cooldown_active"] is False
    assert scheduler["paused"] is False
