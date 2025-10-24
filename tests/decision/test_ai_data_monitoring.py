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
    collect_pending_compliance_sign_offs,
    ensure_compliance_sign_offs,
    export_drift_alert_report,
    export_data_quality_report,
    filter_audit_reports_by_tags,
    filter_audit_reports_by_sign_off_status,
    filter_audit_reports_by_status,
    filter_audit_reports_by_source,
    filter_audit_reports_by_schedule,
    filter_audit_reports_by_category,
    filter_audit_reports_by_job_name,
    filter_audit_reports_by_run,
    filter_audit_reports_by_symbol,
    filter_audit_reports_by_pipeline,
    filter_audit_reports_by_strategy,
    filter_audit_reports_by_dataset,
    filter_audit_reports_by_model,
    filter_audit_reports_by_model_version,
    filter_audit_reports_by_exchange,
    filter_audit_reports_by_license_tier,
    filter_audit_reports_by_risk_class,
    filter_audit_reports_by_required_data,
    filter_audit_reports_by_capability,
    filter_audit_reports_by_profile,
    filter_audit_reports_by_environment,
    filter_audit_reports_by_portfolio,
    filter_audit_reports_by_policy_enforcement,
    filter_audit_reports_since,
    load_recent_data_quality_reports,
    load_recent_drift_reports,
    normalize_report_status,
    normalize_report_source,
    normalize_report_schedule,
    normalize_report_category,
    normalize_report_pipeline,
    normalize_report_license_tier,
    normalize_report_risk_class,
    normalize_report_required_data,
    normalize_report_exchange,
    normalize_report_capability,
    normalize_report_environment,
    normalize_report_portfolio,
    normalize_report_profile,
    normalize_report_dataset,
    normalize_report_strategy,
    normalize_report_model,
    normalize_report_model_version,
    normalize_policy_enforcement,
    normalize_report_symbol,
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


def test_collect_pending_compliance_sign_offs_merges_sources() -> None:
    data_quality_reports = (
        {
            "category": "completeness",
            "status": "alert",
            "policy": {"enforce": True},
            "sign_off": {
                "risk": {"status": "pending"},
                "compliance": {"status": "approved"},
            },
            "report_path": "/tmp/completeness.json",
            "timestamp": "2024-01-01T00:00:00Z",
        },
    )
    drift_reports = (
        {
            "category": "drift_alert",
            "drift_score": 0.7,
            "threshold": 0.5,
            "sign_off": {
                "risk": {"status": "approved"},
                "compliance": {"status": "investigating"},
            },
            "report_path": "/tmp/drift.json",
            "timestamp": "2024-01-02T00:00:00Z",
        },
    )

    pending = collect_pending_compliance_sign_offs(
        data_quality_reports=data_quality_reports,
        drift_reports=drift_reports,
    )

    assert pending["risk"]
    assert pending["risk"][0]["category"] == "completeness"
    assert pending["compliance"]
    assert pending["compliance"][0]["category"] == "drift_alert"


def test_collect_pending_compliance_sign_offs_respects_roles() -> None:
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

    pending = collect_pending_compliance_sign_offs(
        data_quality_reports=reports,
        roles=("risk",),
    )

    assert tuple(pending.keys()) == ("risk",)
    assert pending["risk"] == ()


def test_filter_audit_reports_since_filters_old_entries() -> None:
    reports = (
        {"timestamp": "2024-01-10T00:00:00Z", "category": "old"},
        {"timestamp": "2024-01-20T00:00:00Z", "category": "new"},
        {"category": "missing"},
    )

    filtered = filter_audit_reports_since(
        reports, since=datetime(2024, 1, 15, tzinfo=timezone.utc)
    )

    assert filtered[0]["category"] == "new"
    assert filtered[1]["category"] == "missing"
    assert len(filtered) == 2


def test_filter_audit_reports_since_requires_timezone() -> None:
    with pytest.raises(ValueError):
        filter_audit_reports_since((), since=datetime(2024, 1, 1))


def test_filter_audit_reports_by_tags_filters_include_and_exclude() -> None:
    reports = (
        {"tags": ["pipeline", "nightly"], "category": "keep"},
        {"tags": ["Legacy"], "category": "drop-exclude"},
        {"tags": [], "category": "no-tags"},
        {"category": "missing"},
    )

    filtered_include = filter_audit_reports_by_tags(reports, include=("pipeline",))
    assert len(filtered_include) == 1
    assert filtered_include[0]["category"] == "keep"

    filtered_exclude = filter_audit_reports_by_tags(reports, exclude=("legacy",))
    categories = {entry["category"] for entry in filtered_exclude}
    assert "drop-exclude" not in categories
    assert categories == {"keep", "no-tags", "missing"}

    filtered_both = filter_audit_reports_by_tags(
        reports, include=("pipeline",), exclude=("nightly",)
    )
    assert filtered_both == ()


def test_normalize_report_status_handles_inputs() -> None:
    assert normalize_report_status(" Alert ") == "alert"
    assert normalize_report_status("ok") == "ok"
    assert normalize_report_status(" ") is None
    assert normalize_report_status(None) is None


def test_normalize_report_source_handles_inputs() -> None:
    assert normalize_report_source(" Pipeline ") == "pipeline"
    assert normalize_report_source("") is None
    assert normalize_report_source(None) is None
    assert normalize_report_source("legacy") == "legacy"


def test_normalize_report_schedule_handles_inputs() -> None:
    assert normalize_report_schedule(" Nightly ") == "nightly"
    assert normalize_report_schedule(" ") is None
    assert normalize_report_schedule(None) is None
    assert normalize_report_schedule("eu-open") == "eu-open"


def test_normalize_report_exchange_handles_inputs() -> None:
    assert normalize_report_exchange(" Binance ") == "binance"
    assert normalize_report_exchange("KRAKEN") == "kraken"
    assert normalize_report_exchange(" ") is None
    assert normalize_report_exchange(None) is None


def test_normalize_report_category_handles_inputs() -> None:
    assert normalize_report_category(" Completeness ") == "completeness"
    assert normalize_report_category(" ") is None
    assert normalize_report_category(None) is None
    assert normalize_report_category("drift") == "drift"


def test_normalize_report_pipeline_handles_inputs() -> None:
    assert normalize_report_pipeline(" Nightly ") == "nightly"
    assert normalize_report_pipeline(" ") is None
    assert normalize_report_pipeline(None) is None
    assert normalize_report_pipeline("Retrain") == "retrain"


def test_normalize_report_license_tier_handles_inputs() -> None:
    assert normalize_report_license_tier(" Standard ") == "standard"
    assert normalize_report_license_tier("enterprise") == "enterprise"
    assert normalize_report_license_tier(" ") is None
    assert normalize_report_license_tier(None) is None


def test_normalize_report_risk_class_handles_inputs() -> None:
    assert normalize_report_risk_class(" Directional ") == "directional"
    assert normalize_report_risk_class("market_making") == "market_making"
    assert normalize_report_risk_class(" ") is None
    assert normalize_report_risk_class(None) is None


def test_normalize_report_required_data_handles_inputs() -> None:
    assert normalize_report_required_data(" OHLCV ") == "ohlcv"
    assert normalize_report_required_data("technical_indicators") == "technical_indicators"
    assert normalize_report_required_data(" ") is None
    assert normalize_report_required_data(None) is None


def test_normalize_report_profile_handles_inputs() -> None:
    assert normalize_report_profile(" Conservative ") == "conservative"
    assert normalize_report_profile("AGGRESSIVE") == "aggressive"
    assert normalize_report_profile(" ") is None
    assert normalize_report_profile(None) is None


def test_normalize_report_dataset_handles_inputs() -> None:
    assert normalize_report_dataset(" Primary ") == "primary"
    assert normalize_report_dataset("feature_store") == "feature_store"
    assert normalize_report_dataset(" ") is None
    assert normalize_report_dataset(None) is None


def test_normalize_report_model_handles_inputs() -> None:
    assert normalize_report_model(" Trend_V2 ") == "trend_v2"
    assert normalize_report_model(" ") is None
    assert normalize_report_model(None) is None


def test_normalize_report_model_version_handles_inputs() -> None:
    assert normalize_report_model_version(" 1.2.0 ") == "1.2.0"
    assert normalize_report_model_version("RC1") == "rc1"
    assert normalize_report_model_version(42) == "42"
    assert normalize_report_model_version(" ") is None


def test_normalize_report_strategy_handles_inputs() -> None:
    assert normalize_report_strategy(" Mean_Reversion ") == "mean_reversion"
    assert normalize_report_strategy("TREND-FOLLOW") == "trend-follow"
    assert normalize_report_strategy(" ") is None
    assert normalize_report_strategy(None) is None


def test_normalize_policy_enforcement_handles_inputs() -> None:
    assert normalize_policy_enforcement(True) is True
    assert normalize_policy_enforcement(False) is False
    assert normalize_policy_enforcement(" enforced ") is True
    assert normalize_policy_enforcement("not-enforced") is False
    assert normalize_policy_enforcement("0") is False
    assert normalize_policy_enforcement("unknown") is None
    assert normalize_policy_enforcement(1) is True


def test_filter_audit_reports_by_status_filters_values() -> None:
    reports = (
        {"status": "Alert", "category": "keep-include"},
        {"status": "ok", "category": "drop-exclude"},
        {"category": "no-status"},
    )

    filtered_include = filter_audit_reports_by_status(reports, include=("alert",))
    assert len(filtered_include) == 1
    assert filtered_include[0]["category"] == "keep-include"

    filtered_exclude = filter_audit_reports_by_status(reports, exclude=("ok",))
    categories = {entry.get("category") for entry in filtered_exclude}
    assert "drop-exclude" not in categories
    assert "keep-include" in categories

    filtered_both = filter_audit_reports_by_status(
        reports, include=("alert",), exclude=("alert",)
    )
    assert filtered_both == ()


def test_filter_audit_reports_by_category_filters_values() -> None:
    reports = (
        {"category": "Completeness", "status": "alert"},
        {"category": "drift", "status": "warning"},
        {"category": "legacy", "status": "ok"},
        {"status": "alert"},
    )

    filtered_include = filter_audit_reports_by_category(
        reports, include=("completeness", "drift")
    )
    categories = {entry.get("category") for entry in filtered_include}
    assert categories == {"Completeness", "drift"}

    filtered_exclude = filter_audit_reports_by_category(reports, exclude=("legacy",))
    categories = {entry.get("category") for entry in filtered_exclude}
    assert "legacy" not in categories
    assert "Completeness" in categories

    filtered_both = filter_audit_reports_by_category(
        reports, include=("drift",), exclude=("drift",)
    )
    assert filtered_both == ()


def test_filter_audit_reports_by_symbol() -> None:
    reports = (
        {"symbol": "btcusdt", "category": "keep-direct"},
        {"symbols": ("ethusdt", "ltcusdt"), "category": "keep-multi"},
        {"dataset": {"metadata": {"symbol": "xrpusdt"}}, "category": "keep-dataset"},
        {"context": {"symbols": ["adausdt", "maticusdt"]}, "category": "keep-context"},
        {"category": "missing"},
    )

    filtered_include = filter_audit_reports_by_symbol(reports, include=("BTCUSDT", "ADAUSDT"))
    categories = {entry.get("category") for entry in filtered_include}
    assert categories == {"keep-direct", "keep-context"}

    filtered_exclude = filter_audit_reports_by_symbol(reports, exclude=("LTCUSDT", "XRPUSDT"))
    categories = {entry.get("category") for entry in filtered_exclude}
    assert "keep-multi" not in categories
    assert "keep-dataset" not in categories
    assert "missing" in categories

    filtered_both = filter_audit_reports_by_symbol(
        reports, include=("ETHUSDT",), exclude=("ETHUSDT",)
    )
    assert filtered_both == ()


def test_filter_audit_reports_by_pipeline() -> None:
    reports = (
        {"context": {"pipeline": "Nightly"}, "category": "keep-context"},
        {"dataset": {"metadata": {"pipeline": "Retrain"}}, "category": "keep-dataset"},
        {"pipeline": "LEGACY", "category": "drop"},
        {"category": "missing"},
    )

    filtered_include = filter_audit_reports_by_pipeline(reports, include=("nightly",))
    categories = {entry.get("category") for entry in filtered_include}
    assert categories == {"keep-context"}

    filtered_exclude = filter_audit_reports_by_pipeline(reports, exclude=("legacy",))
    categories = {entry.get("category") for entry in filtered_exclude}
    assert "drop" not in categories
    assert "keep-context" in categories
    assert "keep-dataset" in categories

    filtered_both = filter_audit_reports_by_pipeline(
        reports, include=("nightly", "retrain"), exclude=("legacy",)
    )
    categories = {entry.get("category") for entry in filtered_both}
    assert categories == {"keep-context", "keep-dataset"}


def test_filter_audit_reports_by_capability() -> None:
    reports = (
        {"capability": "TREND_D1", "category": "keep-direct"},
        {"context": {"capability": "legacy"}, "category": "drop-context"},
        {"dataset": {"metadata": {"capability": "trend_d1"}}, "category": "keep-dataset"},
        {"metadata": {"capability": "legacy"}, "category": "drop-metadata"},
        {"strategy": {"capability": "trend_d1"}, "category": "keep-strategy"},
        {"category": "missing"},
    )

    filtered_include = filter_audit_reports_by_capability(
        reports, include=("trend_d1",)
    )
    categories = {entry.get("category") for entry in filtered_include}
    assert categories == {"keep-direct", "keep-dataset", "keep-strategy"}

    filtered_exclude = filter_audit_reports_by_capability(
        reports, exclude=("legacy",)
    )
    categories = {entry.get("category") for entry in filtered_exclude}
    assert "drop-context" not in categories
    assert "drop-metadata" not in categories
    assert "keep-direct" in categories

    filtered_both = filter_audit_reports_by_capability(
        reports, include=("trend_d1",), exclude=("trend_d1",)
    )
    assert filtered_both == ()


def test_filter_audit_reports_by_license_tier() -> None:
    reports = (
        {"license_tier": "Standard", "category": "keep-direct"},
        {"context": {"license_tier": "legacy"}, "category": "drop-context"},
        {"strategy": {"metadata": {"license_tier": "standard"}}, "category": "keep-strategy"},
        {"metadata": {"license_tier": "legacy"}, "category": "drop-metadata"},
        {"category": "missing"},
    )

    filtered_include = filter_audit_reports_by_license_tier(
        reports, include=("standard",)
    )
    categories = {entry.get("category") for entry in filtered_include}
    assert categories == {"keep-direct", "keep-strategy"}

    filtered_exclude = filter_audit_reports_by_license_tier(
        reports, exclude=("legacy",)
    )
    categories = {entry.get("category") for entry in filtered_exclude}
    assert "drop-context" not in categories
    assert "drop-metadata" not in categories
    assert "keep-direct" in categories

    filtered_both = filter_audit_reports_by_license_tier(
        reports, include=("standard",), exclude=("standard",)
    )
    assert filtered_both == ()


def test_filter_audit_reports_by_risk_class() -> None:
    reports = (
        {"risk_classes": ["Directional"], "category": "keep-direct"},
        {"context": {"risk_class": "legacy"}, "category": "drop-context"},
        {
            "dataset": {"metadata": {"risk_classes": ["momentum", "directional"]}},
            "category": "keep-dataset",
        },
        {"strategy": {"metadata": {"risk_classes": ["legacy"]}}, "category": "drop-metadata"},
        {"category": "missing"},
    )

    filtered_include = filter_audit_reports_by_risk_class(
        reports, include=("directional",)
    )
    categories = {entry.get("category") for entry in filtered_include}
    assert categories == {"keep-direct", "keep-dataset"}

    filtered_exclude = filter_audit_reports_by_risk_class(
        reports, exclude=("legacy",)
    )
    categories = {entry.get("category") for entry in filtered_exclude}
    assert "drop-context" not in categories
    assert "drop-metadata" not in categories
    assert "keep-dataset" in categories

    filtered_both = filter_audit_reports_by_risk_class(
        reports, include=("directional",), exclude=("directional",)
    )
    assert filtered_both == ()


def test_filter_audit_reports_by_required_data() -> None:
    reports = (
        {"required_data": ["OHLCV"], "category": "keep-direct"},
        {"context": {"required_data": "legacy"}, "category": "drop-context"},
        {
            "dataset": {"metadata": {"required_data": ["technical_indicators", "ohlcv"]}},
            "category": "keep-dataset",
        },
        {"metadata": {"required_data": ["legacy"]}, "category": "drop-metadata"},
        {"category": "missing"},
    )

    filtered_include = filter_audit_reports_by_required_data(
        reports, include=("ohlcv",)
    )
    categories = {entry.get("category") for entry in filtered_include}
    assert categories == {"keep-direct", "keep-dataset"}

    filtered_exclude = filter_audit_reports_by_required_data(
        reports, exclude=("legacy",)
    )
    categories = {entry.get("category") for entry in filtered_exclude}
    assert "drop-context" not in categories
    assert "drop-metadata" not in categories
    assert "keep-direct" in categories

    filtered_both = filter_audit_reports_by_required_data(
        reports, include=("ohlcv",), exclude=("ohlcv",)
    )
    assert filtered_both == ()


def test_filter_audit_reports_by_environment() -> None:
    reports = (
        {"context": {"environment": "Prod"}, "category": "keep-context"},
        {"dataset": {"metadata": {"environment": "paper"}}, "category": "keep-dataset"},
        {"environment": "LEGACY", "category": "drop"},
        {"category": "missing"},
    )

    filtered_include = filter_audit_reports_by_environment(reports, include=("prod",))
    categories = {entry.get("category") for entry in filtered_include}
    assert categories == {"keep-context"}

    filtered_exclude = filter_audit_reports_by_environment(reports, exclude=("legacy",))
    categories = {entry.get("category") for entry in filtered_exclude}
    assert "drop" not in categories
    assert "keep-context" in categories
    assert "keep-dataset" in categories

    filtered_both = filter_audit_reports_by_environment(
        reports, include=("prod", "paper"), exclude=("legacy",)
    )
    categories = {entry.get("category") for entry in filtered_both}
    assert categories == {"keep-context", "keep-dataset"}


def test_filter_audit_reports_by_exchange() -> None:
    reports = (
        {"context": {"exchange": "Binance"}, "category": "keep-context"},
        {"dataset": {"metadata": {"exchange": "kraken"}}, "category": "keep-dataset"},
        {"metadata": {"exchange": "legacy"}, "category": "drop"},
        {"category": "missing"},
    )

    filtered_include = filter_audit_reports_by_exchange(reports, include=("binance",))
    categories = {entry.get("category") for entry in filtered_include}
    assert categories == {"keep-context"}

    filtered_exclude = filter_audit_reports_by_exchange(reports, exclude=("legacy",))
    categories = {entry.get("category") for entry in filtered_exclude}
    assert "drop" not in categories
    assert "keep-context" in categories
    assert "keep-dataset" in categories

    filtered_both = filter_audit_reports_by_exchange(
        reports, include=("binance", "kraken"), exclude=("legacy",)
    )
    categories = {entry.get("category") for entry in filtered_both}
    assert categories == {"keep-context", "keep-dataset"}


def test_filter_audit_reports_by_portfolio() -> None:
    reports = (
        {"context": {"portfolio": "Core"}, "category": "keep-context"},
        {"dataset": {"metadata": {"portfolio": "hf"}}, "category": "keep-dataset"},
        {"portfolio": "LEGACY", "category": "drop"},
        {"category": "missing"},
    )

    filtered_include = filter_audit_reports_by_portfolio(reports, include=("core",))
    categories = {entry.get("category") for entry in filtered_include}
    assert categories == {"keep-context"}

    filtered_exclude = filter_audit_reports_by_portfolio(reports, exclude=("legacy",))
    categories = {entry.get("category") for entry in filtered_exclude}
    assert "drop" not in categories
    assert "keep-context" in categories
    assert "keep-dataset" in categories

    filtered_both = filter_audit_reports_by_portfolio(
        reports, include=("core", "hf"), exclude=("legacy",)
    )
    categories = {entry.get("category") for entry in filtered_both}
    assert categories == {"keep-context", "keep-dataset"}


def test_filter_audit_reports_by_profile() -> None:
    reports = (
        {"strategy": {"profile": "Conservative"}, "category": "keep-strategy"},
        {"context": {"risk_profile": "Balanced"}, "category": "keep-context"},
        {"metadata": {"profile": "legacy"}, "category": "drop"},
        {"category": "missing"},
    )

    filtered_include = filter_audit_reports_by_profile(reports, include=("conservative",))
    categories = {entry.get("category") for entry in filtered_include}
    assert categories == {"keep-strategy"}

    filtered_exclude = filter_audit_reports_by_profile(reports, exclude=("legacy",))
    categories = {entry.get("category") for entry in filtered_exclude}
    assert "drop" not in categories
    assert "keep-strategy" in categories
    assert "keep-context" in categories

    filtered_both = filter_audit_reports_by_profile(
        reports, include=("conservative", "balanced"), exclude=("legacy",)
    )
    categories = {entry.get("category") for entry in filtered_both}
    assert categories == {"keep-strategy", "keep-context"}


def test_filter_audit_reports_by_strategy() -> None:
    reports = (
        {"strategy": "mean_reversion", "category": "keep-direct"},
        {"context": {"strategy": {"name": "trend-follow"}}, "category": "keep-context"},
        {
            "dataset": {
                "metadata": {"strategy": {"identifier": "scalping", "name": "scalping"}}
            },
            "category": "keep-dataset",
        },
        {"metadata": {"strategy_name": "legacy"}, "category": "drop"},
        {"category": "missing"},
    )

    filtered_include = filter_audit_reports_by_strategy(
        reports, include=("mean_reversion", "trend-follow")
    )
    categories = {entry.get("category") for entry in filtered_include}
    assert categories == {"keep-direct", "keep-context"}

    filtered_exclude = filter_audit_reports_by_strategy(reports, exclude=("legacy", "scalping"))
    categories = {entry.get("category") for entry in filtered_exclude}
    assert "drop" not in categories
    assert "keep-dataset" not in categories
    assert "keep-direct" in categories
    assert "keep-context" in categories

    filtered_both = filter_audit_reports_by_strategy(
        reports,
        include=("mean_reversion", "scalping"),
        exclude=("legacy", "scalping"),
    )
    categories = {entry.get("category") for entry in filtered_both}
    assert categories == {"keep-direct"}


def test_filter_audit_reports_by_dataset() -> None:
    reports = (
        {"dataset": "primary", "category": "keep-direct"},
        {"context": {"dataset": {"name": "shadow"}}, "category": "keep-context"},
        {"metadata": {"dataset_name": "nightly-features"}, "category": "keep-metadata"},
        {
            "dataset": {
                "metadata": {"dataset_identifier": "legacy", "dataset_id": "legacy"}
            },
            "category": "drop",
        },
        {"category": "missing"},
    )

    filtered_include = filter_audit_reports_by_dataset(
        reports, include=("primary", "shadow")
    )
    categories = {entry.get("category") for entry in filtered_include}
    assert categories == {"keep-direct", "keep-context"}

    filtered_exclude = filter_audit_reports_by_dataset(reports, exclude=("legacy",))
    categories = {entry.get("category") for entry in filtered_exclude}
    assert "drop" not in categories
    assert "keep-direct" in categories
    assert "keep-context" in categories
    assert "keep-metadata" in categories

    filtered_both = filter_audit_reports_by_dataset(
        reports, include=("nightly-features", "legacy"), exclude=("legacy",)
    )
    categories = {entry.get("category") for entry in filtered_both}
    assert categories == {"keep-metadata"}


def test_filter_audit_reports_by_model() -> None:
    reports = (
        {"model": "trend_v2", "category": "keep-direct"},
        {"context": {"model": {"name": "mean_v1"}}, "category": "keep-context"},
        {"dataset": {"metadata": {"model_name": "aux"}}, "category": "keep-dataset"},
        {"strategy": {"metadata": {"model": "legacy"}}, "category": "drop"},
        {"category": "missing"},
    )

    filtered_include = filter_audit_reports_by_model(
        reports, include=("trend_v2", "mean_v1")
    )
    categories = {entry.get("category") for entry in filtered_include}
    assert categories == {"keep-direct", "keep-context"}

    filtered_exclude = filter_audit_reports_by_model(reports, exclude=("legacy",))
    categories = {entry.get("category") for entry in filtered_exclude}
    assert "drop" not in categories
    assert {"keep-direct", "keep-context", "keep-dataset"}.issubset(categories)

    filtered_both = filter_audit_reports_by_model(
        reports, include=("aux", "legacy"), exclude=("legacy",)
    )
    categories = {entry.get("category") for entry in filtered_both}
    assert categories == {"keep-dataset"}


def test_filter_audit_reports_by_model_version() -> None:
    reports = (
        {"model_version": "1.2.0", "category": "keep-direct"},
        {"context": {"model_version": "2.0.0"}, "category": "keep-context"},
        {"dataset": {"metadata": {"version": "1.2.0"}}, "category": "keep-dataset"},
        {"strategy": {"metadata": {"model_version": "legacy"}}, "category": "drop"},
        {"metadata": {"version": "3.0.0"}, "category": "keep-metadata"},
        {"category": "missing"},
    )

    filtered_include = filter_audit_reports_by_model_version(
        reports, include=("1.2.0", "2.0.0")
    )
    categories = {entry.get("category") for entry in filtered_include}
    assert categories == {"keep-direct", "keep-context", "keep-dataset"}

    filtered_exclude = filter_audit_reports_by_model_version(
        reports, exclude=("legacy",)
    )
    categories = {entry.get("category") for entry in filtered_exclude}
    assert "drop" not in categories
    assert {"keep-direct", "keep-context", "keep-dataset", "keep-metadata"}.issubset(
        categories
    )

    filtered_both = filter_audit_reports_by_model_version(
        reports, include=("3.0.0", "legacy"), exclude=("legacy",)
    )
    categories = {entry.get("category") for entry in filtered_both}
    assert categories == {"keep-metadata"}


def test_filter_audit_reports_by_run() -> None:
    reports = (
        {"context": {"run": "Alert"}, "category": "keep-context"},
        {"dataset": {"metadata": {"run": "Baseline"}}, "category": "keep-dataset"},
        {"run": "LEGACY", "category": "drop"},
        {"category": "missing"},
    )

    filtered_include = filter_audit_reports_by_run(reports, include=("alert",))
    categories = {entry.get("category") for entry in filtered_include}
    assert categories == {"keep-context"}

    filtered_exclude = filter_audit_reports_by_run(reports, exclude=("legacy",))
    categories = {entry.get("category") for entry in filtered_exclude}
    assert "drop" not in categories
    assert "keep-context" in categories
    assert "keep-dataset" in categories

    filtered_both = filter_audit_reports_by_run(
        reports, include=("alert", "baseline"), exclude=("legacy",)
    )
    categories = {entry.get("category") for entry in filtered_both}
    assert categories == {"keep-context", "keep-dataset"}


def test_filter_audit_reports_by_job_name_filters_values() -> None:
    reports = (
        {"job_name": "Pipeline:BTCUSDT", "category": "keep-include"},
        {"job_name": "LEGACY", "category": "drop-exclude"},
        {"job": "Pipeline:ETHUSDT", "category": "keep-fallback"},
        {"category": "missing"},
    )

    filtered_include = filter_audit_reports_by_job_name(
        reports, include=("pipeline:btcusdt", "pipeline:ethusdt")
    )
    categories = {entry.get("category") for entry in filtered_include}
    assert categories == {"keep-include", "keep-fallback"}

    filtered_exclude = filter_audit_reports_by_job_name(
        reports, exclude=("legacy", "pipeline:ethusdt")
    )
    categories = {entry.get("category") for entry in filtered_exclude}
    assert "drop-exclude" not in categories
    assert "keep-fallback" not in categories
    assert "keep-include" in categories
    assert "missing" in categories

    filtered_both = filter_audit_reports_by_job_name(
        reports, include=("pipeline:btcusdt",), exclude=("pipeline:btcusdt",)
    )
    assert filtered_both == ()


def test_filter_audit_reports_by_sign_off_status_supports_include_and_roles() -> None:
    reports = (
        {
            "category": "keep",
            "sign_off": {
                "risk": {"status": "Pending"},
                "compliance": {"status": "approved"},
            },
        },
        {
            "category": "drop-include",
            "sign_off": {
                "risk": {"status": "waived"},
                "compliance": {"status": "approved"},
            },
        },
        {
            "category": "keep-role",
            "sign_off": {
                "risk": {"status": "approved"},
                "compliance": {"status": "Investigating"},
            },
        },
    )

    filtered_include = filter_audit_reports_by_sign_off_status(
        reports, include=("pending",)
    )
    assert len(filtered_include) == 1
    assert filtered_include[0]["category"] == "keep"

    filtered_exclude = filter_audit_reports_by_sign_off_status(
        reports, exclude=("waived",)
    )
    categories = {entry["category"] for entry in filtered_exclude}
    assert "drop-include" not in categories
    assert categories == {"keep", "keep-role"}

    filtered_roles = filter_audit_reports_by_sign_off_status(
        reports, include=("investigating",), roles=("compliance",)
    )
    assert len(filtered_roles) == 1
    assert filtered_roles[0]["category"] == "keep-role"


def test_filter_audit_reports_by_source() -> None:
    reports = (
        {"source": "pipeline", "category": "keep"},
        {"source": "Legacy", "category": "drop"},
        {"category": "missing"},
    )

    filtered_include = filter_audit_reports_by_source(reports, include=("pipeline",))
    assert len(filtered_include) == 1
    assert filtered_include[0]["category"] == "keep"

    filtered_exclude = filter_audit_reports_by_source(reports, exclude=("legacy",))
    categories = {entry.get("category") for entry in filtered_exclude}
    assert "drop" not in categories
    assert "keep" in categories
    assert "missing" in categories

    filtered_both = filter_audit_reports_by_source(
        reports, include=("pipeline",), exclude=("pipeline",)
    )
    assert filtered_both == ()


def test_filter_audit_reports_by_schedule() -> None:
    reports = (
        {"schedule": "Nightly", "category": "keep"},
        {"schedule": "legacy", "category": "drop"},
        {"category": "missing"},
    )

    filtered_include = filter_audit_reports_by_schedule(reports, include=("nightly",))
    assert len(filtered_include) == 1
    assert filtered_include[0]["category"] == "keep"

    filtered_exclude = filter_audit_reports_by_schedule(reports, exclude=("legacy",))
    categories = {entry.get("category") for entry in filtered_exclude}
    assert "drop" not in categories
    assert "keep" in categories
    assert "missing" in categories

    filtered_both = filter_audit_reports_by_schedule(
        reports, include=("nightly",), exclude=("nightly",)
    )
    assert filtered_both == ()


def test_filter_audit_reports_by_policy_enforcement() -> None:
    reports = (
        {"policy": {"enforce": True}, "category": "keep-enforced"},
        {"policy": {"enforce": False}, "category": "drop-exclude"},
        {"category": "missing"},
    )

    filtered_include = filter_audit_reports_by_policy_enforcement(
        reports, include=(True,)
    )
    assert len(filtered_include) == 1
    assert filtered_include[0]["category"] == "keep-enforced"

    filtered_exclude = filter_audit_reports_by_policy_enforcement(
        reports, exclude=(False,)
    )
    categories = {entry.get("category") for entry in filtered_exclude}
    assert "drop-exclude" not in categories
    assert "keep-enforced" in categories
    assert "missing" in categories

    filtered_both = filter_audit_reports_by_policy_enforcement(
        reports, include=(True,), exclude=(True,)
    )
    assert filtered_both == ()


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
    assert captured_roles == []
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

    assert calls == []

    # Domyślnie bramka jest wyłączona, więc helper nie powinien zostać wywołany.
    manager._ensure_compliance_activation_gate()
    assert calls == []

    manager.set_compliance_sign_off_requirement(True)
    manager._ensure_compliance_activation_gate()

    assert calls[-1] == ("risk",)
    assert len(calls) == 1
