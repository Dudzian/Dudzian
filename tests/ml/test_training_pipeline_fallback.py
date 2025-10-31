from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from bot_core.ai import backends
from core.ml.training_pipeline import TrainingPipeline

from tests.ml.fixtures import synthetic_feature_dataset


@pytest.fixture(autouse=True)
def _clear_backend_caches() -> None:
    backends.clear_backend_caches()
    yield
    backends.clear_backend_caches()


def test_training_pipeline_falls_back_to_reference_backend(
    synthetic_feature_dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "backends.yml"
    config_path.write_text(
        """
priority:
  - lightgbm
  - reference
backends:
  lightgbm:
    module: lightgbm
    install_hint: pip install lightgbm
""".strip(),
        encoding="utf-8",
    )

    original_import = importlib.import_module

    def _missing(name: str, package: str | None = None):  # pragma: no cover - monkeypatch
        if name == "lightgbm":
            raise ModuleNotFoundError(name)
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _missing)
    monkeypatch.setattr(backends, "_DEFAULT_CONFIG_PATH", config_path)

    pipeline = TrainingPipeline(
        preferred_backends=("lightgbm", "reference"),
        config_path=config_path,
        fallback_log_dir=tmp_path / "fallback",
    )

    result = pipeline.train(synthetic_feature_dataset)

    assert result.backend == "reference"
    assert result.fallback_chain, "Fallback chain should capture missing primary backend"
    failure = result.fallback_chain[0]
    assert failure["backend"] == "lightgbm"
    assert "lightgbm" in failure["message"]
    assert "pip install lightgbm" in failure.get("install_hint", "")
    assert result.log_path is not None
    payload = json.loads(result.log_path.read_text(encoding="utf-8"))
    assert payload["selected_backend"] == "reference"
    assert payload["errors"][0]["backend"] == "lightgbm"
    assert result.validation_log_path is not None
    validation_payload = json.loads(
        result.validation_log_path.read_text(encoding="utf-8")
    )
    assert validation_payload["status"] == "passed"

