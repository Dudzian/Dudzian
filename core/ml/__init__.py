"""Pakiet narzędzi ML niezależny od zewnętrznych zależności."""

from __future__ import annotations

from .model_registry import ModelMetadata, ModelRegistry, ModelRegistryError

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


def __getattr__(name: str):
    if name in {
        "BackendEntry",
        "BackendFactory",
        "register_backend",
        "build_backend",
        "list_available_backends",
    }:
        from .factory import (
            BackendEntry,
            BackendFactory,
            build_backend,
            list_available_backends,
            register_backend,
        )

        return {
            "BackendEntry": BackendEntry,
            "BackendFactory": BackendFactory,
            "register_backend": register_backend,
            "build_backend": build_backend,
            "list_available_backends": list_available_backends,
        }[name]

    if name in {"TrainingPipeline", "TrainingPipelineResult"}:
        from .training_pipeline import TrainingPipeline, TrainingPipelineResult

        return {
            "TrainingPipeline": TrainingPipeline,
            "TrainingPipelineResult": TrainingPipelineResult,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
