"""Pakiet bot_core.ai dostarcza pipeline trenowania i inference modeli decyzyjnych."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    # inference / repository
    "DecisionModelInference": (".inference", "DecisionModelInference"),
    "ModelRepository": (".inference", "ModelRepository"),
    # modele i serializacja
    "ModelArtifact": (".models", "ModelArtifact"),
    "ModelMetrics": (".models", "ModelMetrics"),
    "ModelScore": (".models", "ModelScore"),
    "ModelArtifactValidationError": (".validation", "ModelArtifactValidationError"),
    "validate_model_artifact_schema": (".validation", "validate_model_artifact_schema"),
    # training pipeline
    "ModelTrainer": (".training", "ModelTrainer"),
    "SimpleGradientBoostingModel": (".training", "SimpleGradientBoostingModel"),
    "ExternalModelAdapter": (".training", "ExternalModelAdapter"),
    "get_external_model_adapter": (".training", "get_external_model_adapter"),
    "register_external_model_adapter": (".training", "register_external_model_adapter"),
    # audyt
    "AUDIT_SUBDIRECTORIES": (".audit", "AUDIT_SUBDIRECTORIES"),
    "DEFAULT_AUDIT_ROOT": (".audit", "DEFAULT_AUDIT_ROOT"),
    "ensure_audit_structure": (".audit", "ensure_audit_structure"),
    "scheduler_state_path": (".audit", "scheduler_state_path"),
    "save_scheduler_state": (".audit", "save_scheduler_state"),
    "load_scheduler_state": (".audit", "load_scheduler_state"),
    "save_walk_forward_report": (".audit", "save_walk_forward_report"),
    "load_recent_walk_forward_reports": (".audit", "load_recent_walk_forward_reports"),
    "summarize_walk_forward_reports": (".audit", "summarize_walk_forward_reports"),
    "save_data_quality_report": (".audit", "save_data_quality_report"),
    "save_drift_report": (".audit", "save_drift_report"),
    "list_audit_reports": (".audit", "list_audit_reports"),
    "load_audit_report": (".audit", "load_audit_report"),
    "load_latest_walk_forward_report": (".audit", "load_latest_walk_forward_report"),
    "load_latest_data_quality_report": (".audit", "load_latest_data_quality_report"),
    "load_latest_drift_report": (".audit", "load_latest_drift_report"),
    # monitoring danych (pipeline)
    "DataQualityAssessment": (".monitoring", "DataQualityAssessment"),
    "DataQualityIssue": (".monitoring", "DataQualityIssue"),
    "FeatureDriftAnalyzer": (".monitoring", "FeatureDriftAnalyzer"),
    "FeatureDriftAssessment": (".monitoring", "FeatureDriftAssessment"),
    # pipeline helpers
    "train_gradient_boosting_model": (".pipeline", "train_gradient_boosting_model"),
    "register_model_artifact": (".pipeline", "register_model_artifact"),
    "score_with_data_monitoring": (".pipeline", "score_with_data_monitoring"),
    # harmonogram treningów
    "RetrainingScheduler": (".scheduler", "RetrainingScheduler"),
    "ScheduledTrainingJob": (".scheduler", "ScheduledTrainingJob"),
    "TrainingRunRecord": (".scheduler", "TrainingRunRecord"),
    "TrainingScheduler": (".scheduler", "TrainingScheduler"),
    "WalkForwardResult": (".scheduler", "WalkForwardResult"),
    "WalkForwardValidator": (".scheduler", "WalkForwardValidator"),
    # reżimy rynku
    "MarketRegime": (".regime", "MarketRegime"),
    "MarketRegimeAssessment": (".regime", "MarketRegimeAssessment"),
    "MarketRegimeClassifier": (".regime", "MarketRegimeClassifier"),
    "RegimeHistory": (".regime", "RegimeHistory"),
    "RegimeSnapshot": (".regime", "RegimeSnapshot"),
    "RegimeSummary": (".regime", "RegimeSummary"),
    "RegimeStrategyWeights": (".regime", "RegimeStrategyWeights"),
    "RiskLevel": (".regime", "RiskLevel"),
    # sequential AI
    "BUILTIN_HEURISTICS": (".sequential", "BUILTIN_HEURISTICS"),
    "HistoricalFeatureRepository": (".sequential", "HistoricalFeatureRepository"),
    "OnlineScoringResult": (".sequential", "OnlineScoringResult"),
    "SequentialOnlineScorer": (".sequential", "SequentialOnlineScorer"),
    "SequentialTrainingPipeline": (".sequential", "SequentialTrainingPipeline"),
    "SequentialTrainingReport": (".sequential", "SequentialTrainingReport"),
    "TemporalDifferencePolicy": (".sequential", "TemporalDifferencePolicy"),
    "WalkForwardMetrics": (".sequential", "WalkForwardMetrics"),
    # feature engineering
    "FeatureDataset": (".feature_engineering", "FeatureDataset"),
    "FeatureEngineer": (".feature_engineering", "FeatureEngineer"),
    "FeatureVector": (".feature_engineering", "FeatureVector"),
    # monitoring danych (inference)
    "export_data_quality_report": (".data_monitoring", "export_data_quality_report"),
    "export_drift_alert_report": (".data_monitoring", "export_drift_alert_report"),
    "DataQualityException": (".data_monitoring", "DataQualityException"),
    "apply_policy_to_report": (".data_monitoring", "apply_policy_to_report"),
    "update_sign_off": (".data_monitoring", "update_sign_off"),
    "ComplianceSignOffError": (".data_monitoring", "ComplianceSignOffError"),
    "collect_pending_compliance_sign_offs": (
        ".data_monitoring",
        "collect_pending_compliance_sign_offs",
    ),
    "filter_audit_reports_since": (
        ".data_monitoring",
        "filter_audit_reports_since",
    ),
    "filter_audit_reports_by_tags": (
        ".data_monitoring",
        "filter_audit_reports_by_tags",
    ),
    "filter_audit_reports_by_sign_off_status": (
        ".data_monitoring",
        "filter_audit_reports_by_sign_off_status",
    ),
    "filter_audit_reports_by_status": (
        ".data_monitoring",
        "filter_audit_reports_by_status",
    ),
    "filter_audit_reports_by_source": (
        ".data_monitoring",
        "filter_audit_reports_by_source",
    ),
    "filter_audit_reports_by_schedule": (
        ".data_monitoring",
        "filter_audit_reports_by_schedule",
    ),
    "filter_audit_reports_by_category": (
        ".data_monitoring",
        "filter_audit_reports_by_category",
    ),
    "filter_audit_reports_by_job_name": (
        ".data_monitoring",
        "filter_audit_reports_by_job_name",
    ),
    "filter_audit_reports_by_run": (
        ".data_monitoring",
        "filter_audit_reports_by_run",
    ),
    "filter_audit_reports_by_symbol": (
        ".data_monitoring",
        "filter_audit_reports_by_symbol",
    ),
    "filter_audit_reports_by_pipeline": (
        ".data_monitoring",
        "filter_audit_reports_by_pipeline",
    ),
    "filter_audit_reports_by_environment": (
        ".data_monitoring",
        "filter_audit_reports_by_environment",
    ),
    "filter_audit_reports_by_exchange": (
        ".data_monitoring",
        "filter_audit_reports_by_exchange",
    ),
    "filter_audit_reports_by_portfolio": (
        ".data_monitoring",
        "filter_audit_reports_by_portfolio",
    ),
    "filter_audit_reports_by_profile": (
        ".data_monitoring",
        "filter_audit_reports_by_profile",
    ),
    "filter_audit_reports_by_strategy": (
        ".data_monitoring",
        "filter_audit_reports_by_strategy",
    ),
    "filter_audit_reports_by_engine": (
        ".data_monitoring",
        "filter_audit_reports_by_engine",
    ),
    "filter_audit_reports_by_issue_code": (
        ".data_monitoring",
        "filter_audit_reports_by_issue_code",
    ),
    "filter_audit_reports_by_dataset": (
        ".data_monitoring",
        "filter_audit_reports_by_dataset",
    ),
    "filter_audit_reports_by_model": (
        ".data_monitoring",
        "filter_audit_reports_by_model",
    ),
    "filter_audit_reports_by_model_version": (
        ".data_monitoring",
        "filter_audit_reports_by_model_version",
    ),
    "filter_audit_reports_by_license_tier": (
        ".data_monitoring",
        "filter_audit_reports_by_license_tier",
    ),
    "filter_audit_reports_by_risk_class": (
        ".data_monitoring",
        "filter_audit_reports_by_risk_class",
    ),
    "filter_audit_reports_by_required_data": (
        ".data_monitoring",
        "filter_audit_reports_by_required_data",
    ),
    "filter_audit_reports_by_capability": (
        ".data_monitoring",
        "filter_audit_reports_by_capability",
    ),
    "filter_audit_reports_by_policy_enforcement": (
        ".data_monitoring",
        "filter_audit_reports_by_policy_enforcement",
    ),
    "ensure_compliance_sign_offs": (".data_monitoring", "ensure_compliance_sign_offs"),
    "normalize_compliance_sign_off_roles": (
        ".data_monitoring",
        "normalize_compliance_sign_off_roles",
    ),
    "normalize_sign_off_status": (
        ".data_monitoring",
        "normalize_sign_off_status",
    ),
    "normalize_report_status": (
        ".data_monitoring",
        "normalize_report_status",
    ),
    "normalize_report_source": (
        ".data_monitoring",
        "normalize_report_source",
    ),
    "normalize_report_schedule": (
        ".data_monitoring",
        "normalize_report_schedule",
    ),
    "normalize_report_category": (
        ".data_monitoring",
        "normalize_report_category",
    ),
    "normalize_report_job_name": (
        ".data_monitoring",
        "normalize_report_job_name",
    ),
    "normalize_report_run": (
        ".data_monitoring",
        "normalize_report_run",
    ),
    "normalize_report_symbol": (
        ".data_monitoring",
        "normalize_report_symbol",
    ),
    "normalize_report_pipeline": (
        ".data_monitoring",
        "normalize_report_pipeline",
    ),
    "normalize_report_environment": (
        ".data_monitoring",
        "normalize_report_environment",
    ),
    "normalize_report_exchange": (
        ".data_monitoring",
        "normalize_report_exchange",
    ),
    "normalize_report_portfolio": (
        ".data_monitoring",
        "normalize_report_portfolio",
    ),
    "normalize_report_profile": (
        ".data_monitoring",
        "normalize_report_profile",
    ),
    "normalize_report_dataset": (
        ".data_monitoring",
        "normalize_report_dataset",
    ),
    "normalize_report_strategy": (
        ".data_monitoring",
        "normalize_report_strategy",
    ),
    "normalize_report_engine": (
        ".data_monitoring",
        "normalize_report_engine",
    ),
    "normalize_report_issue_code": (
        ".data_monitoring",
        "normalize_report_issue_code",
    ),
    "normalize_report_model": (
        ".data_monitoring",
        "normalize_report_model",
    ),
    "normalize_report_model_version": (
        ".data_monitoring",
        "normalize_report_model_version",
    ),
    "normalize_report_license_tier": (
        ".data_monitoring",
        "normalize_report_license_tier",
    ),
    "normalize_report_risk_class": (
        ".data_monitoring",
        "normalize_report_risk_class",
    ),
    "normalize_report_required_data": (
        ".data_monitoring",
        "normalize_report_required_data",
    ),
    "normalize_report_capability": (
        ".data_monitoring",
        "normalize_report_capability",
    ),
    "normalize_policy_enforcement": (
        ".data_monitoring",
        "normalize_policy_enforcement",
    ),
    "get_supported_sign_off_statuses": (
        ".data_monitoring",
        "get_supported_sign_off_statuses",
    ),
    "load_recent_data_quality_reports": (
        ".data_monitoring",
        "load_recent_data_quality_reports",
    ),
    "load_recent_drift_reports": (".data_monitoring", "load_recent_drift_reports"),
    "summarize_data_quality_reports": (
        ".data_monitoring",
        "summarize_data_quality_reports",
    ),
    "summarize_drift_reports": (
        ".data_monitoring",
        "summarize_drift_reports",
    ),
    # compliance snapshot
    "collect_pipeline_compliance_summary": (
        ".compliance",
        "collect_pipeline_compliance_summary",
    ),
    # manager & historię pipeline'u
    "AIManager": (".manager", "AIManager"),
    "EnsembleDefinition": (".manager", "EnsembleDefinition"),
    "EnsembleRegistryDiff": (".manager", "EnsembleRegistryDiff"),
    "EnsembleRegistrySnapshot": (".manager", "EnsembleRegistrySnapshot"),
    "ModelEvaluation": (".manager", "ModelEvaluation"),
    "PipelineExecutionRecord": (".manager", "PipelineExecutionRecord"),
    "PipelineHistoryDiff": (".manager", "PipelineHistoryDiff"),
    "PipelineHistorySnapshot": (".manager", "PipelineHistorySnapshot"),
    "DataQualityCheck": (".manager", "DataQualityCheck"),
}

_PIPELINE_MONITORING_EXPORTS: Dict[str, Tuple[str, str]] = {
    "DataCompletenessWatcher": (".monitoring", "DataCompletenessWatcher"),
    "FeatureBoundsValidator": (".monitoring", "FeatureBoundsValidator"),
}

_INFERENCE_MONITORING_EXPORTS: Dict[str, Tuple[str, str]] = {
    "InferenceDataCompletenessWatcher": (".data_monitoring", "DataCompletenessWatcher"),
    "InferenceFeatureBoundsValidator": (".data_monitoring", "FeatureBoundsValidator"),
}

_EXPORTS.update(_PIPELINE_MONITORING_EXPORTS)
_EXPORTS.update(_INFERENCE_MONITORING_EXPORTS)

__all__ = sorted(_EXPORTS)


if TYPE_CHECKING:  # pragma: no cover - tylko dla statycznych analizatorów typu mypy
    from .data_monitoring import (
        DataCompletenessWatcher as InferenceDataCompletenessWatcher,
        FeatureBoundsValidator as InferenceFeatureBoundsValidator,
    )
    from .monitoring import DataCompletenessWatcher, FeatureBoundsValidator


def __getattr__(name: str) -> Any:  # pragma: no cover - prosty mechanizm leniwy
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:  # pragma: no cover - delegujemy do standardowego mechanizmu
        raise AttributeError(f"module 'bot_core.ai' has no attribute {name!r}") from exc
    module = importlib.import_module(module_name, package=__name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover - mała funkcja użytkowa
    return sorted(list(globals().keys()) + list(__all__))

