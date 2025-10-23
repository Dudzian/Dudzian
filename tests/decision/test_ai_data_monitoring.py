from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from bot_core.ai import (
    DataCompletenessWatcher,
    DecisionModelInference,
    DataQualityException,
    FeatureBoundsValidator,
    ModelArtifact,
    ModelRepository,
    score_with_data_monitoring,
    update_sign_off,
    load_recent_data_quality_reports,
    load_recent_drift_reports,
    export_drift_alert_report,
    summarize_data_quality_reports,
    summarize_drift_reports,
)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_latest_category_report(root: Path, category: str) -> dict:
    candidates = sorted(root.glob(f"*_{category}.json"))
    if not candidates:
        pytest.fail(f"no reports found for category {category}")
    return _read_json(candidates[-1])


def _normalize_feature_scalers(
    scalers: Mapping[str, object] | None,
) -> dict[str, tuple[float, float]]:
    if not isinstance(scalers, Mapping):
        return {}
    normalized: dict[str, tuple[float, float]] = {}
    for name, raw in scalers.items():
        if isinstance(raw, Mapping):
            mean = float(raw.get("mean", 0.0))
            stdev = float(raw.get("stdev", 1.0))
            normalized[str(name)] = (mean, stdev)
        elif isinstance(raw, Sequence) and len(raw) >= 2:
            normalized[str(name)] = (float(raw[0]), float(raw[1]))
    return normalized


def _merge_metrics(blocks: Mapping[str, object] | None) -> Mapping[str, object]:
    merged: dict[str, object] = {
        "summary": {},
        "train": {},
        "validation": {},
        "test": {},
    }
    if not isinstance(blocks, Mapping):
        return merged
    for split, payload in blocks.items():
        if isinstance(payload, Mapping):
            merged[str(split)] = dict(payload)
    return merged


def _make_artifact(
    *,
    feature_names: Sequence[str],
    model_state: Mapping[str, object],
    metadata: Mapping[str, object],
    metrics: Mapping[str, object] | None = None,
    backend: str = "builtin",
    trained_at: datetime | None = None,
    target_scale: float | None = None,
    training_rows: int | None = None,
    validation_rows: int | None = None,
    test_rows: int | None = None,
    feature_scalers: Mapping[str, tuple[float, float]] | None = None,
    decision_journal_entry_id: str | None = None,
) -> ModelArtifact:
    target_scale = (
        float(target_scale)
        if target_scale is not None
        else float(metadata.get("target_scale", 1.0))
    )
    training_rows = (
        int(training_rows)
        if training_rows is not None
        else int(metadata.get("training_rows", 0))
    )
    validation_rows = (
        int(validation_rows)
        if validation_rows is not None
        else int(metadata.get("validation_rows", 0))
    )
    test_rows = (
        int(test_rows)
        if test_rows is not None
        else int(metadata.get("test_rows", 0))
    )
    computed_metrics = _merge_metrics(metrics)

    scalers_source: Mapping[str, object] | None
    if feature_scalers is not None:
        scalers_source = feature_scalers
    else:
        scalers_source = metadata.get("feature_scalers") or model_state.get(
            "feature_scalers"
        )
    computed_feature_scalers = _normalize_feature_scalers(scalers_source)
    if not computed_feature_scalers:
        computed_feature_scalers = {
            str(name): (0.0, 1.0) for name in feature_names
        }

    if decision_journal_entry_id is None:
        raw_decision_id = metadata.get("decision_journal_entry_id") or metadata.get(
            "decision_journal_entry"
        )
        if raw_decision_id is not None:
            decision_journal_entry_id = str(raw_decision_id)

    trained_at = trained_at or datetime.now(timezone.utc)
    return ModelArtifact(
        feature_names=tuple(feature_names),
        model_state=model_state,
        trained_at=trained_at,
        metrics=computed_metrics,
        metadata=metadata,
        target_scale=target_scale,
        training_rows=training_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
        feature_scalers=computed_feature_scalers,
        decision_journal_entry_id=decision_journal_entry_id,
        backend=backend,
    )


def test_data_completeness_watcher_detects_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_DECISION_AUDIT_ROOT", str(tmp_path / "audit"))
    watcher = DataCompletenessWatcher()
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
    validator = FeatureBoundsValidator()
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
    metadata = {
        "feature_scalers": {
            "alpha": {"mean": 0.0, "stdev": 1.0},
            "beta": {"mean": 0.0, "stdev": 1.0},
        },
        "feature_stats": {
            "alpha": {"mean": 0.0, "stdev": 1.0, "min": -1.0, "max": 1.0},
            "beta": {"mean": 0.0, "stdev": 1.0, "min": -1.0, "max": 1.0},
        },
        "drift_monitor": {
            "threshold": 0.5,
            "window": 1,
            "min_observations": 1,
            "cooldown": 1,
        },
    }
    artifact = _make_artifact(
        feature_names=("alpha", "beta"),
        model_state={
            "learning_rate": 0.1,
            "n_estimators": 0,
            "initial_prediction": 0.0,
            "feature_names": ["alpha", "beta"],
            "feature_scalers": {
                "alpha": {"mean": 0.0, "stdev": 1.0},
                "beta": {"mean": 0.0, "stdev": 1.0},
            },
            "stumps": [],
        },
        metadata=metadata,
        metrics={"mae": 0.0},
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
    metadata = {
        "feature_scalers": {
            "alpha": {"mean": 0.0, "stdev": 1.0},
            "beta": {"mean": 0.0, "stdev": 1.0},
        },
        "feature_stats": {
            "alpha": {"mean": 0.0, "stdev": 1.0, "min": -1.0, "max": 1.0},
            "beta": {"mean": 0.0, "stdev": 1.0, "min": -1.0, "max": 1.0},
        },
        "data_quality": {"enforce": True},
    }
    artifact = _make_artifact(
        feature_names=("alpha", "beta"),
        model_state={
            "learning_rate": 0.1,
            "n_estimators": 0,
            "initial_prediction": 0.0,
            "feature_names": ["alpha", "beta"],
            "feature_scalers": {
                "alpha": {"mean": 0.0, "stdev": 1.0},
                "beta": {"mean": 0.0, "stdev": 1.0},
            },
            "stumps": [],
        },
        metadata=metadata,
        metrics={"mae": 0.0},
    )
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
    metadata = {
        "feature_stats": {
            "alpha": {"mean": 0.0, "stdev": 1.0, "min": -1.0, "max": 1.0},
        },
        "data_quality": {
            "enforce": True,
            "categories": {
                "bounds": {"enforce": False},
            },
        },
    }
    artifact = _make_artifact(
        feature_names=("alpha",),
        model_state={
            "learning_rate": 0.1,
            "n_estimators": 0,
            "initial_prediction": 0.0,
            "feature_names": ["alpha"],
            "feature_scalers": {"alpha": {"mean": 0.0, "stdev": 1.0}},
            "stumps": [],
        },
        metadata=metadata,
        metrics={"mae": 0.0},
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
    metadata = {
        "feature_stats": {
            "alpha": {"mean": 0.0, "stdev": 1.0, "min": -1.0, "max": 1.0},
        },
    }
    artifact = _make_artifact(
        feature_names=("alpha",),
        model_state={
            "learning_rate": 0.1,
            "n_estimators": 0,
            "initial_prediction": 0.0,
            "feature_names": ["alpha"],
            "feature_scalers": {"alpha": {"mean": 0.0, "stdev": 1.0}},
            "stumps": [],
        },
        metadata=metadata,
        metrics={"mae": 0.0},
    )
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
    metadata = {
        "feature_stats": {
            "alpha": {"mean": 0.0, "stdev": 1.0, "min": -1.0, "max": 1.0},
            "beta": {"mean": 0.0, "stdev": 1.0, "min": -1.0, "max": 1.0},
        },
        "drift_monitor": {
            "threshold": 0.5,
            "window": 1,
            "min_observations": 1,
            "cooldown": 1,
        },
    }
    artifact = _make_artifact(
        feature_names=("alpha", "beta"),
        model_state={
            "learning_rate": 0.1,
            "n_estimators": 0,
            "initial_prediction": 0.0,
            "feature_names": ["alpha", "beta"],
            "feature_scalers": {
                "alpha": {"mean": 0.0, "stdev": 1.0},
                "beta": {"mean": 0.0, "stdev": 1.0},
            },
            "stumps": [],
        },
        metadata=metadata,
        metrics={"mae": 0.0},
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
    metadata = {
        "feature_stats": {
            "alpha": {"mean": 0.0, "stdev": 1.0, "min": -1.0, "max": 1.0},
            "beta": {"mean": 0.0, "stdev": 1.0, "min": -1.0, "max": 1.0},
        },
    }
    artifact = _make_artifact(
        feature_names=("alpha", "beta"),
        model_state={
            "learning_rate": 0.1,
            "n_estimators": 0,
            "initial_prediction": 0.0,
            "feature_names": ["alpha", "beta"],
            "feature_scalers": {
                "alpha": {"mean": 0.0, "stdev": 1.0},
                "beta": {"mean": 0.0, "stdev": 1.0},
            },
            "stumps": [],
        },
        metadata=metadata,
        metrics={"mae": 0.0},
    )
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
