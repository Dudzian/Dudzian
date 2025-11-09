from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bot_core.ai.inference import ModelRepository
from bot_core.ai.models import ModelArtifact
from bot_core.ai.validation import ModelQualityReport, record_model_quality_report
from bot_core.reporting.model_quality import load_champion_overview
from bot_core.runtime.ui_bridge import build_auto_mode_snapshot


def _make_artifact(metadata: dict[str, object] | None = None) -> ModelArtifact:
    payload = dict(metadata or {})
    metrics = {
        "summary": {"mae": 1.25, "directional_accuracy": 0.6},
        "train": {},
        "validation": {},
        "test": {},
    }
    return ModelArtifact(
        feature_names=("f1", "f2"),
        model_state={"weights": [0.1, 0.2], "bias": 0.0},
        trained_at=datetime.now(timezone.utc),
        metrics=metrics,
        metadata=payload,
        target_scale=1.0,
        training_rows=128,
        validation_rows=64,
        test_rows=64,
        feature_scalers={"f1": (0.0, 1.0), "f2": (0.0, 1.0)},
        decision_journal_entry_id=None,
        backend="builtin",
    )


def _make_report(version: str, directional: float, mae: float, status: str = "improved") -> ModelQualityReport:
    metrics = {"summary": {"directional_accuracy": directional, "mae": mae}}
    return ModelQualityReport(
        model_name="decision_engine",
        version=version,
        evaluated_at=datetime.now(timezone.utc),
        metrics=metrics,
        status=status,
    )


@pytest.mark.timeout(10)
def test_offline_snapshot_matches_champion_registry(tmp_path: Path) -> None:
    repo_root = tmp_path / "var" / "models"
    model_dir = repo_root / "decision_engine"
    repository = ModelRepository(model_dir)
    repository.publish(_make_artifact(), version="v1", filename="model-v1.json", aliases=("latest",), activate=True)

    quality_dir = repo_root / "quality"
    first_decision = record_model_quality_report(
        _make_report("v1", directional=0.62, mae=14.0),
        history_root=quality_dir,
    )
    second_decision = record_model_quality_report(
        _make_report("v2", directional=0.55, mae=18.0, status="degraded"),
        history_root=quality_dir,
    )
    assert first_decision.decision == "champion"
    assert second_decision.decision == "challenger"

    snapshot = build_auto_mode_snapshot(
        model_name="decision_engine",
        repository=model_dir,
        quality_dir=quality_dir,
    )

    overview = load_champion_overview("decision_engine", base_dir=quality_dir)
    assert overview is not None
    champion = overview["champion"]
    metadata = overview["champion_metadata"]

    decision_summary = snapshot["decision_summary"]
    assert decision_summary["version"] == champion["version"]
    assert decision_summary["reason"] == metadata["reason"]
    assert decision_summary["active_version"] == repository.get_active_version()

    history = snapshot["controller_history"]
    assert history and history[0]["event"] == "champion"
    assert history[0]["model"] == champion["version"]
    assert any(entry["event"] == "challenger" and entry["model"] == second_decision.candidate.get("version") for entry in history)

    guardrail_summary = snapshot["guardrail_summary"]
    assert guardrail_summary["status"] == champion["status"]

    recommendations = snapshot["recommendations"]
    assert isinstance(recommendations, list) and recommendations
    assert "reason" in recommendations[0]

    guard = snapshot["performance_guard"]
    assert guard["fps_target"] == 60
    assert "reduce_motion_after_seconds" in guard

    command = [
        sys.executable,
        "-m",
        "bot_core.runtime.ui_bridge",
        "auto-mode-snapshot",
        "--model",
        "decision_engine",
        "--repository",
        str(model_dir),
        "--quality-dir",
        str(quality_dir),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    parsed = json.loads(result.stdout)
    assert parsed["decision_summary"]["version"] == champion["version"]
    assert parsed["controller_history"][0]["event"] == "champion"
