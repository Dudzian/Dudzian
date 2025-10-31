from __future__ import annotations

import json
import os
from pathlib import Path


from datetime import datetime, timedelta, timezone

from bot_core.observability.alert_overrides import AlertOverride, AlertOverrideManager

from scripts.export_observability_bundle import run as export_bundle


ROOT = Path(__file__).resolve().parents[1]
def _write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")


def test_export_observability_bundle_cli(tmp_path: Path) -> None:
    dashboards = tmp_path / "dashboards"
    alerts = tmp_path / "alerts"
    _write(dashboards / "stage6_dashboard.json", json.dumps({"title": "Stage6"}))
    _write(alerts / "stage6_alerts.yaml", "groups: []\n")

    override = AlertOverride(
        alert="latency",
        status="breach",
        severity="critical",
        reason="Test",
        indicator="latency_p95",
        created_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
        metadata={"error_budget_pct": 0.5},
    )
    overrides_manager = AlertOverrideManager([override])
    overrides_payload = overrides_manager.to_payload()
    overrides_path = tmp_path / "alert_overrides.json"
    overrides_path.write_text(json.dumps(overrides_payload), encoding="utf-8")

    key_path = tmp_path / "keys" / "bundle.key"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key = os.urandom(32)
    key_path.write_bytes(key)
    if os.name != "nt":
        key_path.chmod(0o600)

    output_dir = tmp_path / "out"

    exit_code = export_bundle(
        [
            "--source",
            f"dashboards={dashboards}",
            "--source",
            f"alerts={alerts}",
            "--include",
            "stage6*",
            "--include",
            "**/stage6*",
            "--output-dir",
            str(output_dir),
            "--bundle-name",
            "stage6-observability-test",
            "--metadata",
            "env=ci",
            "--overrides",
            str(overrides_path),
            "--hmac-key-file",
            str(key_path),
            "--hmac-key-id",
            "ops-stage6",
        ]
    )

    assert exit_code == 0

    bundles = list(output_dir.glob("stage6-observability-test-*.zip"))
    assert len(bundles) == 1
    bundle = bundles[0]
    manifest = bundle.with_suffix(".manifest.json")
    signature = bundle.with_suffix(".manifest.sig")

    assert manifest.exists()
    assert signature.exists()

    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    overrides_metadata = manifest_data["metadata"].get("alert_overrides")
    assert overrides_metadata
    assert overrides_metadata["summary"]["active"] == 1
