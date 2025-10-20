"""Pakiet bot_core.ai dostarcza pipeline trenowania i inference modeli decyzyjnych."""

from .feature_engineering import FeatureEngineer, FeatureDataset
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
from .regime import (
    MarketRegime,
    MarketRegimeAssessment,
    MarketRegimeClassifier,
    RegimeHistory,
    RegimeSnapshot,
    RegimeSummary,
    RiskLevel,
)
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
    "DecisionModelInference",
    "FeatureDataset",
    "FeatureEngineer",
    "EnsembleDefinition",
    "EnsembleRegistryDiff",
    "EnsembleRegistrySnapshot",
    "ModelArtifact",
    "ModelEvaluation",
    "ModelRepository",
    "ModelScore",
    "MarketRegime",
    "MarketRegimeAssessment",
    "MarketRegimeClassifier",
    "RegimeHistory",
    "RegimeSnapshot",
    "RegimeSummary",
    "RiskLevel",
    "ExternalModelAdapter",
    "ModelTrainer",
    "RetrainingScheduler",
    "ScheduledTrainingJob",
    "SimpleGradientBoostingModel",
    "TrainingRunRecord",
    "TrainingScheduler",
    "WalkForwardResult",
    "WalkForwardValidator",
    "get_external_model_adapter",
    "PipelineExecutionRecord",
    "PipelineHistoryDiff",
    "PipelineHistorySnapshot",
    "register_external_model_adapter",
]
