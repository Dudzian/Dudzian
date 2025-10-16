import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.run_stage6_observability_cycle import run as run_cycle


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_run_stage6_observability_cycle_script(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    definitions = {
        "definitions": [
            {
                "name": "latency",
                "indicator": "latency_ms",
                "target": 500,
                "comparison": "<=",
                "warning_threshold": 400,
                "severity": "critical",
            }
        ],
    }
    measurements = {
        "indicator": "latency_ms",
        "value": 650,
        "window_start": (now - timedelta(minutes=30)).isoformat(),
        "window_end": now.isoformat(),
        "sample_size": 120,
    }

    definitions_path = tmp_path / "definitions.json"
    measurements_path = tmp_path / "measurements.json"
    _write_json(definitions_path, definitions)
    _write_json(measurements_path, measurements)

    dashboard_dir = tmp_path / "dashboards"
    alerts_dir = tmp_path / "alerts"
    dashboard_dir.mkdir()
    alerts_dir.mkdir()
    dashboard_path = dashboard_dir / "stage6_dashboard.json"
    dashboard_path.write_text(json.dumps({"uid": "stage6"}), encoding="utf-8")
    (alerts_dir / "stage6_alerts.yml").write_text("groups: []\n", encoding="utf-8")

    slo_json = tmp_path / "audit" / "slo_report.json"
    slo_csv = tmp_path / "audit" / "slo_report.csv"
    overrides_json = tmp_path / "audit" / "alert_overrides.json"
    annotations_json = tmp_path / "audit" / "dashboard_annotations.json"
    bundle_dir = tmp_path / "bundle"

    exit_code = run_cycle(
        [
            "--definitions",
            str(definitions_path),
            "--metrics",
            str(measurements_path),
            "--slo-json",
            str(slo_json),
            "--slo-csv",
            str(slo_csv),
            "--overrides-json",
            str(overrides_json),
            "--dashboard",
            str(dashboard_path),
            "--annotations-output",
            str(annotations_json),
            "--annotations-panel-id",
            "3",
            "--bundle-output-dir",
            str(bundle_dir),
            "--bundle-source",
            f"dashboards={dashboard_dir}",
            "--bundle-source",
            f"alerts={alerts_dir}",
            "--tag",
            "stage6",
            "--signing-key",
            "supersecretkey12345",
            "--signing-key-id",
            "ops",
        ]
    )

    assert exit_code == 0
    assert slo_json.exists()
    assert slo_csv.exists()
    assert overrides_json.exists()
    assert annotations_json.exists()

    manifest_files = list(bundle_dir.glob("*.manifest.json"))
    assert manifest_files, "Manifest paczki nie został wygenerowany"
    manifest = json.loads(manifest_files[0].read_text(encoding="utf-8"))
    assert manifest["metadata"]["slo_report"]["json"] == slo_json.as_posix()
