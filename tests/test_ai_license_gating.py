"""Testy wymuszające moduł AI Signals dla komponentów bot_core.ai."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from bot_core.ai import AIManager
from bot_core.ai.inference import DecisionModelInference, ModelRepository
from bot_core.ai.scheduler import RetrainingScheduler, ScheduledTrainingJob, TrainingScheduler
from bot_core.ai.training import (
    ExternalModelAdapter,
    ExternalTrainingResult,
    ModelTrainer,
    register_external_model_adapter,
)
from bot_core.security.capabilities import build_capabilities_from_payload
from bot_core.security.guards import (
    LicenseCapabilityError,
    install_capability_guard,
    reset_capability_guard,
)


@pytest.fixture(autouse=True)
def _reset_guard() -> None:
    reset_capability_guard()
    yield
    reset_capability_guard()


def _install_capabilities(ai_enabled: bool) -> None:
    payload = {
        "edition": "pro",
        "environments": ["demo", "paper", "live"],
        "exchanges": {"binance_spot": True},
        "strategies": {"trend_d1": True},
        "runtime": {"multi_strategy_scheduler": True},
        "modules": {"ai_signals": ai_enabled},
        "limits": {},
    }
    capabilities = build_capabilities_from_payload(payload, effective_date=date(2025, 1, 1))
    install_capability_guard(capabilities)


def test_ai_manager_requires_ai_module(tmp_path: Path) -> None:
    _install_capabilities(ai_enabled=False)
    with pytest.raises(LicenseCapabilityError):
        AIManager(model_dir=tmp_path)


def test_inference_requires_ai_module(tmp_path: Path) -> None:
    _install_capabilities(ai_enabled=False)
    repository = ModelRepository(tmp_path)
    with pytest.raises(LicenseCapabilityError):
        DecisionModelInference(repository)


def test_training_scheduler_requires_ai_module() -> None:
    _install_capabilities(ai_enabled=False)
    with pytest.raises(LicenseCapabilityError):
        TrainingScheduler()


def test_retraining_scheduler_requires_ai_module() -> None:
    _install_capabilities(ai_enabled=False)
    with pytest.raises(LicenseCapabilityError):
        RetrainingScheduler(interval=timedelta(hours=1))


def test_model_trainer_requires_ai_module() -> None:
    _install_capabilities(ai_enabled=False)
    with pytest.raises(LicenseCapabilityError):
        ModelTrainer()


def test_register_external_adapter_requires_ai_module() -> None:
    _install_capabilities(ai_enabled=False)

    adapter = ExternalModelAdapter(
        backend="xgboost",
        train=lambda _ctx: ExternalTrainingResult(state={}),
        load=lambda _state, _features, _metadata: None,
    )

    with pytest.raises(LicenseCapabilityError):
        register_external_model_adapter(adapter)


def test_ai_components_allowed_when_module_enabled(tmp_path: Path) -> None:
    _install_capabilities(ai_enabled=True)

    manager = AIManager(model_dir=tmp_path)
    repository = ModelRepository(tmp_path)
    DecisionModelInference(repository)
    scheduler = TrainingScheduler()
    job = ScheduledTrainingJob(
        name="test",
        scheduler=RetrainingScheduler(interval=timedelta(hours=1)),
        trainer_factory=lambda: ModelTrainer(),
        dataset_provider=lambda: pytest.skip("dataset provider nie powinien zostać wywołany"),
    )
    scheduler.register(job)

