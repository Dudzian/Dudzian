"""Pakiet bot_core.ai dostarcza pipeline trenowania i inference modeli decyzyjnych."""

from .feature_engineering import FeatureDataset, FeatureEngineer, FeatureVector
from .inference import DecisionModelInference, ModelRepository
from .manager import (
    AIManager,
    EnsembleDefinition,
    EnsembleRegistryDiff,
    EnsembleRegistrySnapshot,
    ModelEvaluation,
    PipelineExecutionRecord,
    PipelineHistoryDiff,
    PipelineHistorySnapshot,
)
from .models import ModelArtifact, ModelScore
from .sequential import (
    BUILTIN_HEURISTICS,
    HistoricalFeatureRepository,
    OnlineScoringResult,
    SequentialOnlineScorer,
    SequentialTrainingPipeline,
    SequentialTrainingReport,
    TemporalDifferencePolicy,
    WalkForwardMetrics,
)
from .regime import (
    MarketRegime,
    MarketRegimeAssessment,
    MarketRegimeClassifier,
    RegimeHistory,
    RegimeSnapshot,
    RegimeSummary,
    RegimeStrategyWeights,
    RiskLevel,
)
from .pipeline import register_model_artifact, train_gradient_boosting_model
from .scheduler import (
    RetrainingScheduler,
    ScheduledTrainingJob,
    TrainingRunRecord,
    TrainingScheduler,
    WalkForwardResult,
    WalkForwardValidator,
)
from .training import (
    ExternalModelAdapter,
    ModelTrainer,
    SimpleGradientBoostingModel,
    get_external_model_adapter,
    register_external_model_adapter,
)

__all__ = [
    "AIManager",
    "BUILTIN_HEURISTICS",
    "DecisionModelInference",
    "FeatureDataset",
    "FeatureEngineer",
    "FeatureVector",
    "HistoricalFeatureRepository",
    "EnsembleDefinition",
    "EnsembleRegistryDiff",
    "EnsembleRegistrySnapshot",
    "ModelArtifact",
    "ModelEvaluation",
    "ModelRepository",
    "ModelScore",
    "OnlineScoringResult",
    "MarketRegime",
    "MarketRegimeAssessment",
    "MarketRegimeClassifier",
    "RegimeHistory",
    "RegimeSnapshot",
    "RegimeSummary",
    "RegimeStrategyWeights",
    "RiskLevel",
    "ExternalModelAdapter",
    "ModelTrainer",
    "RetrainingScheduler",
    "ScheduledTrainingJob",
    "SequentialOnlineScorer",
    "SequentialTrainingPipeline",
    "SequentialTrainingReport",
    "SimpleGradientBoostingModel",
    "TrainingRunRecord",
    "TrainingScheduler",
    "TemporalDifferencePolicy",
    "WalkForwardResult",
    "WalkForwardMetrics",
    "WalkForwardValidator",
    "get_external_model_adapter",
    "PipelineExecutionRecord",
    "PipelineHistoryDiff",
    "PipelineHistorySnapshot",
    "register_external_model_adapter",
    "register_model_artifact",
    "train_gradient_boosting_model",
]
