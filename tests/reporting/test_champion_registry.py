from __future__ import annotations

from datetime import datetime, timezone

from bot_core.ai.validation import ModelQualityReport, record_model_quality_report
from bot_core.reporting.model_quality import load_champion_overview, promote_challenger


def _build_report(version: str, directional: float, mae: float, status: str = "improved") -> ModelQualityReport:
    metrics = {
        "summary": {
            "directional_accuracy": directional,
            "mae": mae,
        }
    }
    return ModelQualityReport(
        model_name="demo",
        version=version,
        evaluated_at=datetime.now(timezone.utc),
        metrics=metrics,
        status=status,
    )


def test_champion_registry_promotes_and_tracks_challengers(tmp_path) -> None:
    history_root = tmp_path / "quality"

    first = _build_report("v1", directional=0.62, mae=14.0)
    decision_first = record_model_quality_report(first, history_root=history_root)
    assert decision_first.decision == "champion"

    overview = load_champion_overview("demo", base_dir=history_root)
    assert overview is not None
    assert overview["champion"].get("version") == "v1"
    assert overview["challengers"] == []

    second = _build_report("v2", directional=0.55, mae=16.5, status="ok")
    decision_second = record_model_quality_report(second, history_root=history_root)
    assert decision_second.decision == "challenger"

    overview = load_champion_overview("demo", base_dir=history_root)
    assert overview is not None
    assert overview["champion"].get("version") == "v1"
    challengers = overview["challengers"]
    assert len(challengers) == 1
    assert challengers[0]["report"].get("version") == "v2"

    third = _build_report("v3", directional=0.68, mae=12.0)
    decision_third = record_model_quality_report(third, history_root=history_root)
    assert decision_third.decision == "champion"

    overview = load_champion_overview("demo", base_dir=history_root)
    assert overview is not None
    assert overview["champion"].get("version") == "v3"
    challengers = overview["challengers"]
    assert len(challengers) == 2
    assert challengers[0]["report"].get("version") == "v1"
    assert challengers[1]["report"].get("version") == "v2"


def test_promote_challenger_replaces_champion(tmp_path) -> None:
    history_root = tmp_path / "quality"

    first = _build_report("v1", directional=0.6, mae=12.0)
    decision_first = record_model_quality_report(first, history_root=history_root)
    assert decision_first.decision == "champion"

    second = _build_report("v2", directional=0.58, mae=11.5, status="ok")
    decision_second = record_model_quality_report(second, history_root=history_root)
    assert decision_second.decision == "challenger"

    decision_promote = promote_challenger("demo", "v2", base_dir=history_root, reason="Manual override")
    assert decision_promote.decision == "champion"
    assert decision_promote.reason == "Manual override"

    overview = load_champion_overview("demo", base_dir=history_root)
    assert overview is not None
    assert overview["champion"].get("version") == "v2"
    challengers = overview["challengers"]
    assert challengers
    assert challengers[0]["report"].get("version") == "v1"
