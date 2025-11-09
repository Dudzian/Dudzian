from __future__ import annotations

import json
from pathlib import Path

import pytest

from bot_core.reporting import model_quality
from scripts.audit import champion_diff


def _write_champion(
    path: Path,
    *,
    decided_at: str,
    metrics: dict[str, object],
    parameters: dict[str, object],
) -> None:
    payload = {
        "model_name": path.parent.name,
        "decided_at": decided_at,
        "reason": "test",
        "report": {
            "status": "active",
            "metrics": metrics,
            "parameters": parameters,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_generate_diff_creates_summary(tmp_path: Path) -> None:
    lhs_root = tmp_path / "lhs"
    rhs_root = tmp_path / "rhs"
    (lhs_root / "alpha").mkdir(parents=True)
    (rhs_root / "alpha").mkdir(parents=True)

    _write_champion(
        lhs_root / "alpha" / model_quality.CHAMPION_FILENAME,
        decided_at="2024-01-01T00:00:00Z",
        metrics={"directional_accuracy": 0.55, "nested": {"mae": 0.42}},
        parameters={"window": 24},
    )
    _write_champion(
        rhs_root / "alpha" / model_quality.CHAMPION_FILENAME,
        decided_at="2024-02-01T00:00:00Z",
        metrics={"directional_accuracy": 0.6, "nested": {"mae": 0.35}},
        parameters={"window": 36},
    )

    output_dir = tmp_path / "out"
    report_path = champion_diff.generate_diff(
        lhs_root=lhs_root,
        rhs_root=rhs_root,
        models=["alpha"],
        output_dir=output_dir,
        tag="ci",
    )

    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["models"] == ["alpha"]
    comparison = data["comparisons"][0]
    assert comparison["differences"]["status_change"] is None
    metrics_delta = comparison["differences"]["metrics_delta"]
    assert pytest.approx(metrics_delta["directional_accuracy"]["delta"]) == 0.05
    assert pytest.approx(metrics_delta["nested.mae"]["delta"]) == -0.07
    parameter_changes = comparison["differences"]["parameter_changes"]
    assert parameter_changes == {"window": {"from": 24, "to": 36}}


def test_generate_diff_handles_missing_models(tmp_path: Path) -> None:
    lhs_root = tmp_path / "lhs"
    rhs_root = tmp_path / "rhs"
    (lhs_root / "beta").mkdir(parents=True)
    _write_champion(
        lhs_root / "beta" / model_quality.CHAMPION_FILENAME,
        decided_at="2024-03-01T00:00:00Z",
        metrics={"expected_pnl": 1.2},
        parameters={},
    )

    output_dir = tmp_path / "out"
    report_path = champion_diff.generate_diff(
        lhs_root=lhs_root,
        rhs_root=rhs_root,
        models=None,
        output_dir=output_dir,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["models"] == ["beta"]
    comparison = payload["comparisons"][0]
    assert comparison["rhs"]["exists"] is False
    assert comparison["differences"]["status_change"] == {"from": "active", "to": None}
