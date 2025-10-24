from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from bot_core.ai import (
    AIManager,
    DecisionModelInference,
    ComplianceSignOffError,
    DataQualityException,
    InferenceDataCompletenessWatcher,
    InferenceFeatureBoundsValidator,
    ModelArtifact,
    ModelRepository,
    ensure_compliance_sign_offs,
    export_drift_alert_report,
    export_data_quality_report,
    load_recent_data_quality_reports,
    load_recent_drift_reports,
    score_with_data_monitoring,
    summarize_data_quality_reports,
    summarize_drift_reports,
    update_sign_off,
)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_latest_category_report(root: Path, category: str) -> dict:
    candidates = sorted(root.glob(f"*_{category}.json"))
    if not candidates:
        pytest.fail(f"no reports found for category {category}")
    return _read_json(candidates[-1])


def _make_artifact(
    *,
    feature_names: tuple[str, ...] = ("alpha", "beta"),
    model_state: dict | None = None,
    metrics: dict | None = None,
    trained_at: datetime | None = None,
    data_quality: dict | None = None,
    drift_monitor: dict | None = None,
    feature_stats: dict | None = None,
    metadata_feature_scalers: dict | None = None,
    feature_scaler_summary: dict | None = None,
    metadata_extra: dict | None = None,
    decision_journal_entry_id: str | None = None,
    target_scale: float = 1.0,
    training_rows: int = 10,
    validation_rows: int = 5,
    test_rows: int = 5,
) -> ModelArtifact:
    feature_names = tuple(feature_names)
    metadata_scalers = metadata_feature_scalers or {
        name: {"mean": 0.0, "stdev": 1.0} for name in feature_names
    }
    stats = feature_stats or {
        name: {"mean": 0.0, "stdev": 1.0, "min": -1.0, "max": 1.0}
        for name in feature_names
    }
    scaler_summary = feature_scaler_summary or {
        name: (0.0, 1.0) for name in feature_names
    }
    metadata = {
        "feature_names": list(feature_names),
        "feature_scalers": metadata_scalers,
        "feature_stats": stats,
    }
    if data_quality is None:
        metadata["data_quality"] = {"enforce": True}
    else:
        metadata["data_quality"] = data_quality
    if drift_monitor is not None:
        metadata["drift_monitor"] = drift_monitor
    if metadata_extra:
        metadata.update(metadata_extra)
    model_defaults = {
        "learning_rate": 0.1,
        "n_estimators": 0,
        "initial_prediction": 0.0,
        "feature_names": list(feature_names),
        "feature_scalers": metadata_scalers,
        "stumps": [],
    }

    return ModelArtifact(
        feature_names=feature_names,
        model_state=model_state or model_defaults,
        trained_at=trained_at or datetime.now(timezone.utc),
        metrics=metrics or {"mae": 0.0},
        metadata=metadata,
        target_scale=target_scale,
        training_rows=training_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
        feature_scalers=scaler_summary,
        decision_journal_entry_id=decision_journal_entry_id,
        backend="builtin",
    )


def test_data_completeness_watcher_detects_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_DECISION_AUDIT_ROOT", str(tmp_path / "audit"))
    watcher = InferenceDataCompletenessWatcher()
    watcher.configure(["alpha", "beta"])

    report = watcher.observe({"alpha": 1.0, "beta": None}, context={"pipeline": "unit"})

    assert report["status"] == "alert"
    assert report["missing_features"] == ["beta"]
    exported = Path(report["report_path"])
    assert exported.exists()
    payload = _read_json(exported)
    assert "sign_off" in payload
    assert sorted(payload["missing_features"]) == ["beta"]


def test_feature_bounds_validator_detects_outliers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_DECISION_AUDIT_ROOT", str(tmp_path / "audit"))
    validator = InferenceFeatureBoundsValidator()
    validator.configure({"ratio": {"min": 0.0, "max": 1.0}})

    report = validator.observe({"ratio": 10.0}, context={"pipeline": "unit"})

    assert report["status"] == "alert"
    assert report["violations"]
    violation = report["violations"][0]
    assert violation["feature"] == "ratio"
    exported = Path(report["report_path"])
    assert exported.exists()
    payload = _read_json(exported)
    assert payload["status"] == "alert"
    assert payload["violations"][0]["reason"] == "out_of_bounds"


def test_inference_monitoring_exports_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audit_root = tmp_path / "audit"
    monkeypatch.setenv("AI_DECISION_AUDIT_ROOT", str(audit_root))
    repository = ModelRepository(tmp_path / "repo")
    artifact = _make_artifact(
        drift_monitor={
            "threshold": 0.5,
            "window": 1,
            "min_observations": 1,
            "cooldown": 1,
        }
    )
    model_path = repository.save(artifact, "test_model.json")

    inference = DecisionModelInference(repository)
    inference.model_label = "unit-model"
    inference.load_weights(model_path)

    with pytest.raises(DataQualityException) as excinfo:
        score_with_data_monitoring(
            inference,
            {"alpha": None, "beta": 10.0},
            context={"run": "unit"},
        )

    reports = excinfo.value.reports
    assert "completeness" in reports
    assert reports["completeness"]["status"] == "alert"
    assert reports["completeness"]["policy"]["enforce"] is True
    assert "bounds" in reports
    assert reports["bounds"]["status"] == "alert"
    assert reports["bounds"]["policy"]["enforce"] is True
    assert inference.last_data_quality_report == reports

    data_quality_dir = audit_root / "data_quality"
    drift_dir = audit_root / "drift"
    quality_reports = list(data_quality_dir.glob("*.json"))
    drift_reports = list(drift_dir.glob("*.json"))
    assert quality_reports, "no data quality reports were generated"
    assert drift_reports, "no drift reports were generated"

    sample_quality = _read_json(quality_reports[-1])
    assert sample_quality["sign_off"]["risk"]["status"] == "pending"
    assert sample_quality["policy"]["enforce"] is True
    sample_drift = _read_json(drift_reports[-1])
    assert sample_drift["drift_score"] >= sample_drift["threshold"]
    assert "sign_off" in sample_drift


def test_score_with_data_monitoring_passes_when_data_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_root = tmp_path / "audit"
    monkeypatch.setenv("AI_DECISION_AUDIT_ROOT", str(audit_root))
    repository = ModelRepository(tmp_path / "repo")
    artifact = _make_artifact(data_quality={"enforce": True})
    model_path = repository.save(artifact, "test_model_ok.json")

    inference = DecisionModelInference(repository)
    inference.model_label = "unit-model"
    inference.load_weights(model_path)

    score, report = score_with_data_monitoring(
        inference,
        {"alpha": 0.1, "beta": -0.2},
        context={"run": "unit-ok"},
    )

    assert math.isfinite(score.expected_return_bps)
    assert 0.0 <= score.success_probability <= 1.0
    assert report
    assert report["completeness"]["status"] == "ok"
    assert report["completeness"]["policy"]["enforce"] is True
    assert report["bounds"]["status"] == "ok"
    assert report["bounds"]["policy"]["enforce"] is True
    assert inference.last_data_quality_report == report

    quality_dir = audit_root / "data_quality"
    latest_completeness = _read_latest_category_report(quality_dir, "completeness")
    latest_bounds = _read_latest_category_report(quality_dir, "bounds")
    assert latest_completeness["policy"]["enforce"] is True
    assert latest_bounds["policy"]["enforce"] is True


def test_score_with_data_monitoring_respects_category_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_root = tmp_path / "audit"
    monkeypatch.setenv("AI_DECISION_AUDIT_ROOT", str(audit_root))
    repository = ModelRepository(tmp_path / "repo")
    artifact = _make_artifact(
        feature_names=("alpha",),
        data_quality={
            "enforce": True,
            "categories": {
                "bounds": {"enforce": False},
            },
        },
    )
    model_path = repository.save(artifact, "policy_model.json")

    inference = DecisionModelInference(repository)
    inference.model_label = "policy-model"
    inference.load_weights(model_path)

    score, report = score_with_data_monitoring(
        inference,
        {"alpha": 5.0},
        context={"run": "policy"},
    )

    assert isinstance(score.expected_return_bps, float)
    assert report["bounds"]["status"] == "alert"
    assert report["bounds"]["policy"]["enforce"] is False
    assert inference.last_data_quality_report["bounds"]["policy"]["enforce"] is False

    quality_dir = audit_root / "data_quality"
    latest_bounds = _read_latest_category_report(quality_dir, "bounds")
    assert latest_bounds["policy"]["enforce"] is False


def test_update_sign_off_updates_exported_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_root = tmp_path / "audit"
    monkeypatch.setenv("AI_DECISION_AUDIT_ROOT", str(audit_root))
    repository = ModelRepository(tmp_path / "repo")
    artifact = _make_artifact(feature_names=("alpha",))
    model_path = repository.save(artifact, "signoff_model.json")

    inference = DecisionModelInference(repository)
    inference.model_label = "signoff-model"
    inference.load_weights(model_path)

    score_with_data_monitoring(
        inference,
        {"alpha": 0.3},
        context={"run": "signoff"},
    )

    report = inference.last_data_quality_report
    assert report is not None
    completeness = report["completeness"]

    updated = update_sign_off(
        completeness,
        role="risk",
        status="approved",
        signed_by="Risk Officer",
        notes="Dane zweryfikowane",
    )

    assert updated["sign_off"]["risk"]["status"] == "approved"
    assert updated["sign_off"]["risk"]["signed_by"] == "Risk Officer"
    assert updated["sign_off"]["risk"]["notes"] == "Dane zweryfikowane"
    assert isinstance(updated["sign_off"]["risk"].get("timestamp"), str)

    payload = _read_json(Path(updated["report_path"]))
    assert payload["sign_off"]["risk"]["status"] == "approved"
    assert payload["sign_off"]["risk"]["signed_by"] == "Risk Officer"
    assert payload["sign_off"]["risk"]["notes"] == "Dane zweryfikowane"


def test_load_recent_reports_filters_and_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_root = tmp_path / "audit"
    monkeypatch.setenv("AI_DECISION_AUDIT_ROOT", str(audit_root))
    repository = ModelRepository(tmp_path / "repo")
    artifact = _make_artifact(
        drift_monitor={
            "threshold": 0.5,
            "window": 1,
            "min_observations": 1,
            "cooldown": 1,
        }
    )
    model_path = repository.save(artifact, "history_model.json")

    inference = DecisionModelInference(repository)
    inference.model_label = "history-model"
    inference.load_weights(model_path)

    score_with_data_monitoring(
        inference,
        {"alpha": 0.0, "beta": 0.0},
        context={"run": "baseline"},
    )

    with pytest.raises(DataQualityException):
        score_with_data_monitoring(
            inference,
            {"alpha": None, "beta": 10.0},
            context={"run": "alert"},
        )

    reports_all = load_recent_data_quality_reports(limit=5)
    assert reports_all
    assert Path(reports_all[0]["report_path"]).exists()
    assert reports_all[0]["context"]["run"] == "alert"
    assert any(report["status"] == "alert" for report in reports_all)

    completeness_reports = load_recent_data_quality_reports(
        category="completeness", limit=5
    )
    assert completeness_reports
    assert completeness_reports[0]["context"]["run"] == "alert"
    assert all(
        report.get("category") == "completeness" for report in completeness_reports
    )

    with pytest.raises(ValueError):
        load_recent_data_quality_reports(limit=0)

    drift_reports = load_recent_drift_reports(limit=5)
    assert drift_reports
    assert Path(drift_reports[0]["report_path"]).exists()


def test_update_sign_off_validates_inputs() -> None:
    with pytest.raises(ValueError):
        update_sign_off({}, role="finance", status="approved")
    with pytest.raises(ValueError):
        update_sign_off({}, role="risk", status="done")


def test_summarize_data_quality_reports_flags_pending_sign_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_root = tmp_path / "audit"
    monkeypatch.setenv("AI_DECISION_AUDIT_ROOT", str(audit_root))
    repository = ModelRepository(tmp_path / "repo")
    artifact = _make_artifact()
    model_path = repository.save(artifact, "summary_model.json")

    inference = DecisionModelInference(repository)
    inference.model_label = "summary-model"
    inference.load_weights(model_path)

    score_with_data_monitoring(
        inference,
        {"alpha": 0.0, "beta": 0.0},
        context={"run": "healthy"},
    )

    with pytest.raises(DataQualityException):
        score_with_data_monitoring(
            inference,
            {"alpha": None, "beta": 10.0},
            context={"run": "breach"},
        )

    reports = load_recent_data_quality_reports(limit=10)
    summary = summarize_data_quality_reports(reports)

    assert summary["total"] >= 4
    assert summary["alerts"] >= 2
    assert summary["enforced_alerts"] >= 2
    by_category = summary["by_category"]
    assert "completeness" in by_category
    assert by_category["completeness"]["alerts"] >= 1
    assert by_category["completeness"]["latest_status"] == "alert"
    pending_risk = summary["pending_sign_off"]["risk"]
    pending_compliance = summary["pending_sign_off"]["compliance"]
    assert pending_risk
    assert pending_compliance
    assert any(item["category"] == "completeness" for item in pending_risk)
    assert all(item["status"] != "approved" for item in pending_compliance)


def test_summarize_data_quality_reports_preserves_additional_roles() -> None:
    reports = (
        {
            "category": "completeness",
            "status": "alert",
            "policy": {"enforce": True},
            "sign_off": {
                "risk": {"status": "approved"},
                "compliance": {"status": "approved"},
                "aml": {"status": "pending"},
            },
            "timestamp": "2024-01-01T00:00:00Z",
        },
    )

    summary = summarize_data_quality_reports(reports)

    pending = summary["pending_sign_off"]
    assert pending["risk"] == ()
    assert "aml" in pending
    aml_pending = pending["aml"]
    assert aml_pending
    assert aml_pending[0]["status"] == "pending"
    assert aml_pending[0]["category"] == "completeness"


def test_summarize_drift_reports_marks_threshold_excess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_root = tmp_path / "audit"
    monkeypatch.setenv("AI_DECISION_AUDIT_ROOT", str(audit_root))

    export_drift_alert_report(
        {
            "model_name": "baseline",
            "drift_score": 0.4,
            "threshold": 0.5,
            "window": 5,
            "backend": "unit",
        }
    )
    export_drift_alert_report(
        {
            "model_name": "baseline",
            "drift_score": 0.8,
            "threshold": 0.5,
            "window": 5,
            "backend": "unit",
        }
    )

    reports = load_recent_drift_reports(limit=5)
    summary = summarize_drift_reports(reports)

    assert summary["total"] == 2
    assert summary["exceeds_threshold"] == 1
    assert summary["latest_report_path"]
    assert summary["latest_exceeding_report_path"]
    pending_risk = summary["pending_sign_off"]["risk"]
    assert pending_risk
    assert pending_risk[0]["status"] == "pending"


def test_ensure_compliance_sign_offs_detects_pending(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="bot_core.ai.data_monitoring")
    reports = (
        {
            "category": "completeness",
            "status": "alert",
            "policy": {"enforce": True},
            "sign_off": {
                "risk": {"status": "pending"},
                "compliance": {"status": "approved"},
            },
        },
    )

    with pytest.raises(ComplianceSignOffError) as excinfo:
        ensure_compliance_sign_offs(data_quality_reports=reports)
    assert excinfo.value.pending["risk"]
    assert not excinfo.value.pending["compliance"]
    assert "awaiting risk sign-off" in caplog.text


def test_ensure_compliance_sign_offs_accepts_approved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AI_DECISION_AUDIT_ROOT", str(tmp_path / "audit"))
    report_path = export_data_quality_report(
        {
            "category": "completeness",
            "status": "alert",
            "policy": {"enforce": True},
            "sign_off": {
                "risk": {"status": "approved", "signed_by": "risk-user"},
                "compliance": {"status": "approved", "signed_by": "comp-user"},
            },
        },
        category="completeness",
    )
    report = _read_json(report_path)
    report["report_path"] = str(report_path)

    result = ensure_compliance_sign_offs(data_quality_reports=(report,))
    assert dict(result) == {"risk": (), "compliance": ()}


def test_ensure_compliance_sign_offs_respects_custom_roles(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="bot_core.ai.data_monitoring")
    reports = (
        {
            "category": "completeness",
            "status": "alert",
            "policy": {"enforce": True},
            "sign_off": {
                "risk": {"status": "approved"},
                "compliance": {"status": "pending"},
            },
        },
    )

    result = ensure_compliance_sign_offs(
        data_quality_reports=reports,
        roles=("risk",),
    )

    assert dict(result) == {"risk": ()}
    assert "Missing" not in caplog.text


def test_ensure_compliance_sign_offs_validates_roles() -> None:
    with pytest.raises(ValueError):
        ensure_compliance_sign_offs(roles=("unknown",))


def test_ensure_compliance_sign_offs_warns_on_empty_roles(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="bot_core.ai.data_monitoring")

    with pytest.raises(ValueError):
        ensure_compliance_sign_offs(roles=())

    assert "No supported compliance sign-off roles configured" in caplog.text


def test_summarize_reports_mark_missing_sign_offs() -> None:
    reports = (
        {
            "category": "bounds",
            "status": "alert",
            "policy": {"enforce": True},
        },
    )

    dq_summary = summarize_data_quality_reports(reports)
    for role in ("risk", "compliance"):
        pending = dq_summary["pending_sign_off"][role]
        assert pending
        assert pending[0]["status"] == "pending"


def test_summarize_reports_require_sign_off_raises() -> None:
    reports = (
        {
            "category": "completeness",
            "status": "alert",
            "policy": {"enforce": True},
            "sign_off": {"risk": {"status": "pending"}},
        },
    )

    with pytest.raises(ComplianceSignOffError) as excinfo:
        summarize_data_quality_reports(reports, require_sign_off=True)
    assert excinfo.value.pending["risk"]
    assert excinfo.value.pending["compliance"]


def test_summarize_drift_reports_require_sign_off_raises() -> None:
    reports = (
        {
            "category": "drift_alert",
            "drift_score": 1.0,
            "threshold": 0.5,
            "sign_off": {"risk": {"status": "investigating"}},
        },
    )

    with pytest.raises(ComplianceSignOffError) as excinfo:
        summarize_drift_reports(reports, require_sign_off=True)
    assert excinfo.value.pending["risk"]


def test_summarize_reports_supports_custom_roles() -> None:
    reports = (
        {
            "category": "bounds",
            "status": "alert",
            "policy": {"enforce": True},
            "sign_off": {"risk": {"status": "pending"}},
        },
    )

    summary = summarize_data_quality_reports(reports, roles=("risk",))
    assert tuple(summary["pending_sign_off"].keys()) == ("risk",)
    assert summary["pending_sign_off"]["risk"]


def test_ai_manager_validates_custom_sign_off_roles(tmp_path: Path) -> None:
    manager = AIManager(model_dir=tmp_path / "cache")

    with pytest.raises(ValueError):
        manager.set_compliance_sign_off_roles(("unknown",))


def test_ai_manager_passes_configured_sign_off_roles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured_roles: list[Sequence[str] | None] = []

    def _fake_ensure(**kwargs):
        roles = kwargs.get("roles")
        captured_roles.append(roles)
        if roles is None:
            roles = ("risk", "compliance")
        return {role: () for role in roles}

    monkeypatch.setattr("bot_core.ai.manager.ensure_compliance_sign_offs", _fake_ensure)

    manager = AIManager(model_dir=tmp_path / "cache")
    manager.set_compliance_sign_off_requirement(True)
    manager.set_compliance_sign_off_roles(("risk",))
    manager._ensure_compliance_activation_gate()
    assert captured_roles[-1] == ("risk",)

    manager.set_compliance_sign_off_roles(None)
    manager._ensure_compliance_activation_gate()
    assert captured_roles[-1] is None


def test_ai_manager_skips_sign_off_gate_when_not_required(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[Sequence[str] | None] = []

    def _record_call(**kwargs):
        roles = kwargs.get("roles")
        calls.append(roles)
        effective_roles = roles or ("risk", "compliance")
        return {role: () for role in effective_roles}

    monkeypatch.setattr("bot_core.ai.manager.ensure_compliance_sign_offs", _record_call)

    manager = AIManager(model_dir=tmp_path / "cache")
    manager.set_compliance_sign_off_roles(("risk",))

    assert calls == [("risk",)]

    # Domyślnie bramka jest wyłączona, więc helper nie powinien zostać wywołany ponownie.
    manager._ensure_compliance_activation_gate()
    assert calls == [("risk",)]

    manager.set_compliance_sign_off_requirement(True)
    manager._ensure_compliance_activation_gate()

    assert calls[-1] == ("risk",)
    assert len(calls) == 2
