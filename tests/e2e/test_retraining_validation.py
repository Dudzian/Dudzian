from __future__ import annotations

import json
from pathlib import Path

import pytest

from bot_core.ai import backends
import scripts.run_retraining_cycle as run_retraining_cycle


@pytest.fixture(autouse=True)
def _force_missing_lightgbm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zapewnia, że backend lightgbm jest traktowany jako niedostępny."""

    original_require = backends.require_backend
    original_is_available = backends.is_backend_available

    def _fake_require(name: str, *, config_path: Path | None = None):
        if name.strip().lower() == "lightgbm":
            raise backends.BackendUnavailableError(
                "lightgbm", "lightgbm", install_hint="pip install lightgbm"
            )
        return original_require(name, config_path=config_path)

    def _fake_is_available(name: str, *, config_path: Path | None = None) -> bool:
        if name.strip().lower() == "lightgbm":
            return False
        return original_is_available(name, config_path=config_path)

    backends.clear_backend_caches()
    monkeypatch.setattr(backends, "require_backend", _fake_require)
    monkeypatch.setattr(backends, "is_backend_available", _fake_is_available)
    backends.clear_backend_caches()
    yield
    backends.clear_backend_caches()


@pytest.mark.e2e_retraining
def test_retraining_validation_cycle(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_payload = {
        "symbol": "SYNTH",
        "start_timestamp": 1_700_000_000.0,
        "features": [
            {"momentum": 0.1, "volatility": 0.25, "spread": 0.01},
            {"momentum": 0.3, "volatility": 0.35, "spread": 0.02},
            {"momentum": -0.2, "volatility": 0.28, "spread": 0.015},
        ],
        "targets": [0.01, 0.015, -0.005],
    }
    dataset_path.write_text(json.dumps(dataset_payload), encoding="utf-8")

    config_path = tmp_path / "retraining.yml"
    config_path.write_text(
        """
interval_minutes: 15
chaos:
  enabled: false
""".strip(),
        encoding="utf-8",
    )

    report_dir = tmp_path / "reports"
    snapshot_dir = tmp_path / "snapshots"
    e2e_log_dir = tmp_path / "logs"
    fallback_dir = tmp_path / "fallback"
    validation_dir = tmp_path / "validation"

    exit_code = run_retraining_cycle.main(
        [
            "--config",
            str(config_path),
            "--dataset",
            str(dataset_path),
            "--preferred-backend",
            "lightgbm",
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
    assert stdout, "CLI powinno zwrócić JSON z raportem"
    report_payload = json.loads(stdout)
    assert report_payload["status"] == "completed"
    assert report_payload["backend"] == "reference"
    assert report_payload["kpi"]["fallback_count"] >= 1
    assert "validation_log_path" in report_payload["kpi"]
    promotion_payload = report_payload.get("promotion")
    assert promotion_payload is not None
    assert promotion_payload["status"] == "skipped"
    assert promotion_payload["reason"] in {"alerts", "fallback"}
    details = promotion_payload.get("details", {})
    if "fallback_count" in details:
        assert details["fallback_count"] >= 1

    json_report = next(report_dir.glob("retraining_*.json"))
    report_data = json.loads(json_report.read_text(encoding="utf-8"))
    assert report_data["backend"] == "reference"
    assert report_data["kpi"]["fallback_count"] >= 1
    assert report_data["promotion"]["status"] == "skipped"

    markdown_report = next(report_dir.glob("retraining_*.md"))
    assert "Raport cyklu retreningu" in markdown_report.read_text(encoding="utf-8")

    snapshot_file = next(snapshot_dir.glob("kpi_*.json"))
    snapshot_data = json.loads(snapshot_file.read_text(encoding="utf-8"))
    assert snapshot_data["backend"] == "reference"
    assert snapshot_data["kpi"]["fallback_count"] >= 1

    log_file = next(e2e_log_dir.glob("retraining_run_*.json"))
    log_data = json.loads(log_file.read_text(encoding="utf-8"))
    assert log_data["status"] == "completed"
    assert log_data["kpi_snapshot"] == str(snapshot_file)
    assert log_data["report_json"] == str(json_report)
    assert log_data["report_markdown"] == str(markdown_report)
    assert log_data["promotion"]["status"] == "skipped"

    fallback_files = list(fallback_dir.glob("fallback_*.json"))
    assert fallback_files, "Powinien powstać log fallbacku backendu"
    fallback_payload = json.loads(fallback_files[0].read_text(encoding="utf-8"))
    assert fallback_payload["selected_backend"] == "reference"
    assert any(entry.get("backend") == "lightgbm" for entry in fallback_payload["errors"])

    validation_files = list(validation_dir.glob("dataset_validation_*.json"))
    assert validation_files, "Powinien powstać raport walidacji datasetu"
