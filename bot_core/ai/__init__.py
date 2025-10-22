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
    # pipeline helpers
    "train_gradient_boosting_model": (".pipeline", "train_gradient_boosting_model"),
    "register_model_artifact": (".pipeline", "register_model_artifact"),
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
    # manager & historię pipeline'u
    "AIManager": (".manager", "AIManager"),
    "EnsembleDefinition": (".manager", "EnsembleDefinition"),
    "EnsembleRegistryDiff": (".manager", "EnsembleRegistryDiff"),
    "EnsembleRegistrySnapshot": (".manager", "EnsembleRegistrySnapshot"),
    "ModelEvaluation": (".manager", "ModelEvaluation"),
    "PipelineExecutionRecord": (".manager", "PipelineExecutionRecord"),
    "PipelineHistoryDiff": (".manager", "PipelineHistoryDiff"),
    "PipelineHistorySnapshot": (".manager", "PipelineHistorySnapshot"),
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

