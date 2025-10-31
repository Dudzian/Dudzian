"""Pakiet narzędzi ML niezależny od zewnętrznych zależności."""

from .factory import (
    BackendEntry,
    BackendFactory,
    build_backend,
    list_available_backends,
    register_backend,
)
from .model_registry import ModelMetadata, ModelRegistry, ModelRegistryError
from .training_pipeline import TrainingPipeline, TrainingPipelineResult

__all__ = [
    "BackendEntry",
    "BackendFactory",
    "ModelMetadata",
    "ModelRegistry",
    "ModelRegistryError",
    "TrainingPipeline",
    "TrainingPipelineResult",
    "register_backend",
    "build_backend",
    "list_available_backends",
]
