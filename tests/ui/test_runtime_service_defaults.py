from datetime import datetime, timezone
from pathlib import Path

from bot_core.ai import ModelRepository
from bot_core.ai.models import ModelArtifact
from ui.backend.runtime_service import RuntimeService


def test_runtime_service_uses_demo_loader_when_no_journal() -> None:
    service = RuntimeService()

    result = service.loadRecentDecisions(5)

    assert result, "Oczekiwano wpisów demonstracyjnych przy pustej konfiguracji"
    assert service.errorMessage == ""


def test_runtime_service_refreshes_runtime_metadata(tmp_path: Path) -> None:
    registry_dir = tmp_path / "models"
    registry_dir.mkdir()
    runtime_template = Path("config/runtime.yaml").read_text(encoding="utf-8")
    runtime_config_path = tmp_path / "runtime.yaml"
    runtime_config_path.write_text(
        runtime_template.replace("model_registry_path: models", f"model_registry_path: {registry_dir}"),
        encoding="utf-8",
    )

    repository = ModelRepository(registry_dir)
    now = datetime.now(timezone.utc)
    artifact = ModelArtifact(
        feature_names=("regime", "strategy"),
        model_state={
            "policies": {
                "trend": {
                    "regime": "trend",
                    "total_plays": 5,
                    "strategies": [
                        {
                            "name": "trend_following",
                            "plays": 5,
                            "total_reward": 3.5,
                            "total_squared_reward": 2.1,
                            "last_reward": 0.8,
                            "updated_at": now.isoformat(),
                        }
                    ],
                }
            }
        },
        trained_at=now,
        metrics={"summary": {"total_plays": 5}},
        metadata={"updated_at": now.isoformat()},
        target_scale=1.0,
        training_rows=5,
        validation_rows=0,
        test_rows=0,
        feature_scalers={},
    )
    repository.save(
        artifact,
        "adaptive_strategy_policy.json",
        version=now.strftime("%Y%m%dT%H%M%S"),
        aliases=("latest",),
        activate=True,
    )

    service = RuntimeService(
        decision_loader=lambda limit: [],
        runtime_config_path=runtime_config_path,
    )

    service.refreshRuntimeMetadata()

    assert service.retrainNextRun, "Oczekiwano obliczenia harmonogramu retrainingu"
    summary = service.adaptiveStrategySummary
    assert "trend" in summary
    assert "trend_following" in summary
