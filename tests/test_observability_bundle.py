from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot_core.observability.bundle import (
    AssetSource,
    ObservabilityBundleBuilder,
    ObservabilityBundleVerifier,
    load_manifest,
    load_signature,
    verify_signature,
)


def _write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")


def test_build_and_verify_observability_bundle(tmp_path: Path) -> None:
    dashboards = tmp_path / "grafana"
    alerts = tmp_path / "prom"
    _write(dashboards / "stage6_resilience_operations.json", json.dumps({"title": "Stage6"}))
    _write(alerts / "rules" / "stage6_alerts.yml", "groups: []\n")

    key = os.urandom(32)

    builder = ObservabilityBundleBuilder(
        [
            AssetSource(category="dashboards", root=dashboards),
            AssetSource(category="alerts", root=alerts),
        ],
        include=("stage6*", "**/stage6*"),
    )
    artifacts = builder.build(
        bundle_name="stage6-observability-test",
        output_dir=tmp_path / "out",
        metadata={"environment": "ci"},
        signing_key=key,
        signing_key_id="test-key",
    )

    manifest = load_manifest(artifacts.manifest_path)
    verifier = ObservabilityBundleVerifier(artifacts.bundle_path, manifest)
    assert verifier.verify_files() == []

    signature = load_signature(artifacts.signature_path)
    assert signature is not None
    assert verify_signature(manifest, signature, key=key) == []


def test_observability_bundle_detects_tamper(tmp_path: Path) -> None:
    dashboards = tmp_path / "grafana"
    _write(dashboards / "stage6_dashboard.json", "{}")

    builder = ObservabilityBundleBuilder(
        [AssetSource(category="dashboards", root=dashboards)],
        include=("stage6*", "**/stage6*"),
    )
    artifacts = builder.build(
        bundle_name="stage6-observability-test",
        output_dir=tmp_path / "out",
    )

    manifest = load_manifest(artifacts.manifest_path)

    import zipfile

    with zipfile.ZipFile(artifacts.bundle_path, "a") as archive:
        archive.writestr("dashboards/extra.json", "{}")

    verifier = ObservabilityBundleVerifier(artifacts.bundle_path, manifest)
    errors = verifier.verify_files()
    assert any("nieoczekiwane" in message for message in errors)

