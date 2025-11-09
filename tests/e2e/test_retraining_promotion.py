from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bot_core.ai.validation import ModelQualityReport, record_model_quality_report
import scripts.run_retraining_cycle as run_retraining_cycle


def _quality_report(version: str, directional: float, mae: float) -> ModelQualityReport:
    return ModelQualityReport(
        model_name="decision_engine",
        version=version,
        evaluated_at=datetime.now(timezone.utc),
        metrics={"summary": {"directional_accuracy": directional, "mae": mae}},
        status="improved",
    )


@pytest.mark.e2e_retraining
def test_retraining_cycle_promotes_best_challenger(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    quality_dir = tmp_path / "quality"
    champion_report = _quality_report("v1", directional=0.55, mae=15.0)
    record_model_quality_report(champion_report, history_root=quality_dir)

    challenger_report = {
        "model_name": "decision_engine",
        "version": "v2",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "status": "improved",
        "metrics": {"summary": {"directional_accuracy": 0.66, "mae": 13.2}},
    }
    challengers_path = quality_dir / "decision_engine" / "challengers.json"
    challengers_payload = {
        "model_name": "decision_engine",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "entries": [
            {
                "decided_at": datetime.now(timezone.utc).isoformat(),
                "reason": "Oczekiwanie na auto-promocję",
                "report": challenger_report,
            }
        ],
    }
    challengers_path.write_text(json.dumps(challengers_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    dataset_path = tmp_path / "dataset.json"
    dataset_payload = {
        "symbol": "SYNTH",
        "start_timestamp": 1_700_000_000.0,
        "features": [
            {"momentum": 0.1, "volatility": 0.25, "spread": 0.01},
            {"momentum": 0.3, "volatility": 0.35, "spread": 0.02},
            {"momentum": -0.2, "volatility": 0.28, "spread": 0.015},
            {"momentum": 0.4, "volatility": 0.32, "spread": 0.018},
        ],
        "targets": [0.01, 0.015, -0.005, 0.02],
    }
    dataset_path.write_text(json.dumps(dataset_payload), encoding="utf-8")

    report_dir = tmp_path / "reports"
    snapshot_dir = tmp_path / "snapshots"
    e2e_log_dir = tmp_path / "logs"
    fallback_dir = tmp_path / "fallback"
    validation_dir = tmp_path / "validation"

    exit_code = run_retraining_cycle.main(
        [
            "--dataset",
            str(dataset_path),
            "--preferred-backend",
            "reference",
            "--quality-dir",
            str(quality_dir),
            "--auto-promote-model",
            "decision_engine",
            "--report-dir",
            str(report_dir),
            "--kpi-snapshot-dir",
            str(snapshot_dir),
            "--e2e-log-dir",
            str(e2e_log_dir),
            "--fallback-log-dir",
            str(fallback_dir),
            "--validation-log-dir",
            str(validation_dir),
        ]
    )

    assert exit_code == 0

    stdout = capsys.readouterr().out.strip()
    assert stdout
    payload = json.loads(stdout)
    promotion = payload["promotion"]
    assert promotion["status"] == "executed"
    assert any(entry["decision"] == "champion" for entry in promotion["decisions"])

    champion_path = quality_dir / "decision_engine" / "champion.json"
    champion_data = json.loads(champion_path.read_text(encoding="utf-8"))
    assert champion_data["report"]["version"] == "v2"

    json_report = next(report_dir.glob("retraining_*.json"))
    report_data = json.loads(json_report.read_text(encoding="utf-8"))
    assert report_data["promotion"]["status"] == "executed"
    assert any(entry["decision"] == "champion" for entry in report_data["promotion"]["decisions"])

    log_file = next(e2e_log_dir.glob("retraining_run_*.json"))
    log_data = json.loads(log_file.read_text(encoding="utf-8"))
    assert log_data["promotion"]["status"] == "executed"
