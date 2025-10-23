"""Pakiet bot_core.ai dostarcza pipeline trenowania i inference modeli decyzyjnych."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    # inference / repository
    "DecisionModelInference": (".inference", "DecisionModelInference"),
    "ModelRepository": (".inference", "ModelRepository"),
    # modele i serializacja
    "ModelArtifact": (".models", "ModelArtifact"),
    "ModelScore": (".models", "ModelScore"),
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
    "save_walk_forward_report": (".audit", "save_walk_forward_report"),
    "save_data_quality_report": (".audit", "save_data_quality_report"),
    "save_drift_report": (".audit", "save_drift_report"),
    "list_audit_reports": (".audit", "list_audit_reports"),
    "load_audit_report": (".audit", "load_audit_report"),
    "load_latest_walk_forward_report": (".audit", "load_latest_walk_forward_report"),
    "load_latest_data_quality_report": (".audit", "load_latest_data_quality_report"),
    "load_latest_drift_report": (".audit", "load_latest_drift_report"),
    # monitoring danych
    "DataCompletenessWatcher": (".monitoring", "DataCompletenessWatcher"),
    "DataQualityAssessment": (".monitoring", "DataQualityAssessment"),
    "DataQualityIssue": (".monitoring", "DataQualityIssue"),
    "FeatureBoundsValidator": (".monitoring", "FeatureBoundsValidator"),
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
    # monitoring danych
    "DataCompletenessWatcher": (".data_monitoring", "DataCompletenessWatcher"),
    "FeatureBoundsValidator": (".data_monitoring", "FeatureBoundsValidator"),
    "export_data_quality_report": (".data_monitoring", "export_data_quality_report"),
    "export_drift_alert_report": (".data_monitoring", "export_drift_alert_report"),
    "DataQualityException": (".data_monitoring", "DataQualityException"),
    "apply_policy_to_report": (".data_monitoring", "apply_policy_to_report"),
    "update_sign_off": (".data_monitoring", "update_sign_off"),
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

__all__ = sorted(_EXPORTS)


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

