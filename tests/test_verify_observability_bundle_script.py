import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import datetime, timedelta, timezone

from bot_core.observability.alert_overrides import AlertOverride, AlertOverrideManager

from scripts.export_observability_bundle import run as export_bundle
from scripts.verify_observability_bundle import run as verify_bundle


def _prepare_sources(base: Path) -> tuple[Path, Path]:
    dashboards = base / "dashboards"
    alerts = base / "alerts"
    dashboards.mkdir(parents=True, exist_ok=True)
    alerts.mkdir(parents=True, exist_ok=True)
    (dashboards / "stage6_dashboard.json").write_text(
        json.dumps({"title": "Stage6"}),
        encoding="utf-8",
    )
    (alerts / "stage6_alerts.yml").write_text("groups: []\n", encoding="utf-8")
    return dashboards, alerts


def _write_overrides(path: Path) -> None:
    override = AlertOverride(
        alert="latency",
        status="breach",
        severity="critical",
        reason="Test",
        indicator="latency_p95",
        created_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        metadata={"error_budget_pct": 0.5},
    )
    payload = AlertOverrideManager([override]).to_payload()
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_verify_observability_bundle_success(tmp_path: Path) -> None:
    dashboards, alerts = _prepare_sources(tmp_path)
    overrides_path = tmp_path / "overrides.json"
    _write_overrides(overrides_path)

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

    bundle = next(output_dir.glob("stage6-observability-test-*.zip"))

    verify_code = verify_bundle(
        [
            str(bundle),
            "--hmac-key-file",
            str(key_path),
        ]
    )

    assert verify_code == 0


def test_verify_observability_bundle_detects_tamper(tmp_path: Path) -> None:
    dashboards, alerts = _prepare_sources(tmp_path)
    output_dir = tmp_path / "out"

    export_bundle(
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
        ]
    )

    bundle = next(output_dir.glob("stage6-observability-test-*.zip"))
    manifest = bundle.with_suffix(".manifest.json")

    import zipfile

    with zipfile.ZipFile(bundle, "a") as archive:
        archive.writestr("alerts/extra.yml", "tamper")

    exit_code = verify_bundle(
        [
            str(bundle),
            "--manifest",
            str(manifest),
        ]
    )

    assert exit_code == 2

