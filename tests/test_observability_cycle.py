import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bot_core.observability.hypercare import (
    BundleConfig,
    DashboardSyncConfig,
    ObservabilityCycleConfig,
    ObservabilityHypercareCycle,
    OverridesOutputConfig,
    SLOOutputConfig,
)
from bot_core.observability.bundle import AssetSource


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sample_definitions(now: datetime) -> dict[str, object]:
    return {
        "definitions": [
            {
                "name": "latency",
                "indicator": "latency_ms",
                "target": 500,
                "comparison": "<=",
                "warning_threshold": 400,
                "severity": "critical",
                "tags": ["stage6"],
            }
        ],
        "composites": [
            {
                "name": "core_stack",
                "objectives": ["latency"],
                "max_breaches": 0,
                "severity": "critical",
            }
        ],
    }


def _sample_measurements(now: datetime) -> dict[str, object]:
    window_start = (now - timedelta(minutes=30)).isoformat()
    window_end = now.isoformat()
    return {
        "indicator": "latency_ms",
        "value": 650,
        "window_start": window_start,
        "window_end": window_end,
        "sample_size": 120,
        "metadata": {"latency_ms": 650},
    }


def test_observability_cycle_end_to_end(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    definitions_path = tmp_path / "definitions.json"
    measurements_path = tmp_path / "measurements.json"
    _write_json(definitions_path, _sample_definitions(now))
    _write_json(measurements_path, _sample_measurements(now))

    dashboard_dir = tmp_path / "dashboards"
    alerts_dir = tmp_path / "alerts"
    dashboard_dir.mkdir()
    alerts_dir.mkdir()
    dashboard_path = dashboard_dir / "stage6_dashboard.json"
    dashboard_path.write_text(json.dumps({"uid": "stage6", "panels": []}), encoding="utf-8")
    (alerts_dir / "stage6_alerts.yml").write_text("groups: []\n", encoding="utf-8")

    slo_json = tmp_path / "audit" / "slo_report.json"
    slo_csv = tmp_path / "audit" / "slo_report.csv"
    overrides_json = tmp_path / "audit" / "alert_overrides.json"
    annotations_json = tmp_path / "audit" / "dashboard_annotations.json"

    cycle = ObservabilityHypercareCycle(
        ObservabilityCycleConfig(
            definitions_path=definitions_path,
            metrics_path=measurements_path,
            slo=SLOOutputConfig(json_path=slo_json, csv_path=slo_csv, pretty_json=True),
            overrides=OverridesOutputConfig(
                json_path=overrides_json,
                include_warning=True,
                ttl=timedelta(minutes=60),
                requested_by="NOC",
                source="slo_monitor",
                tags=("stage6",),
            ),
            dashboard=DashboardSyncConfig(
                dashboard_path=dashboard_path,
                output_path=annotations_json,
                panel_id=1,
                pretty=True,
            ),
            bundle=BundleConfig(
                output_dir=tmp_path / "bundle",
                sources=(
                    AssetSource(category="dashboards", root=dashboard_dir),
                    AssetSource(category="alerts", root=alerts_dir),
                ),
            ),
            signing_key=b"supersecretkey12345",
            signing_key_id="stage6",
        )
    )

    result = cycle.run()

    assert result.slo_report_path.exists()
    assert result.slo_signature_path and result.slo_signature_path.exists()
    assert result.slo_csv_path and result.slo_csv_path.exists()

    overrides_data = json.loads(result.overrides_path.read_text(encoding="utf-8"))
    assert overrides_data["schema"] == "stage6.observability.alert_overrides"
    assert overrides_data["summary"]["active"] >= 1

    annotations_data = json.loads(result.dashboard_annotations_path.read_text(encoding="utf-8"))
    assert annotations_data["schema"] == "stage6.observability.dashboard_annotations"

    assert result.bundle_path and result.bundle_path.exists()
    assert result.bundle_manifest_path and result.bundle_manifest_path.exists()
    manifest = json.loads(result.bundle_manifest_path.read_text(encoding="utf-8"))
    assert manifest["metadata"]["slo_report"]["json"] == slo_json.as_posix()
    assert result.bundle_signature_path and result.bundle_signature_path.exists()

    verification = result.bundle_verification
    assert verification and verification["verified_files"] == manifest["file_count"]
    assert verification["signature_verified"] is True
