from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from bot_core.marketplace.assignments import PresetAssignmentStore
from bot_core.security.signing import build_hmac_signature
from scripts import ui_marketplace_bridge as bridge


def _write_preset(
    path,
    *,
    preset_id: str,
    fingerprint: str,
    signing_key: bytes,
    version: str = "1.0.0",
    dependencies: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    payload = {
        "name": "Automation Pack",
        "strategies": [
            {
                "name": "mean-pack",
                "engine": "mean_reversion",
                "parameters": {"lookback": 50},
                "license_tier": "professional",
                "risk_classes": ["statistical"],
                "required_data": ["ohlcv"],
            }
        ],
        "metadata": {
            "id": preset_id,
            "profile": "ai",
            "license": {
                "module_id": f"module::{preset_id}",
                "fingerprint": fingerprint,
                "expires_at": datetime(2099, 1, 1, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
            },
        },
    }
    payload["metadata"]["version"] = version
    if dependencies is not None:
        payload["metadata"]["dependencies"] = list(dependencies)
    signature = build_hmac_signature(payload, key=signing_key, key_id="catalog")
    path.write_text(json.dumps({"preset": payload, "signature": signature}, ensure_ascii=False, indent=2), encoding="utf-8")


@pytest.fixture()
def preset_dir(tmp_path, monkeypatch):
    presets = tmp_path / "presets"
    presets.mkdir()
    monkeypatch.chdir(tmp_path)
    return presets


@pytest.fixture()
def signing_key() -> bytes:
    return b"catalog-secret"


def test_list_activate_deactivate_flow(capsys, preset_dir, signing_key):
    preset_file = preset_dir / "automation.json"
    _write_preset(preset_file, preset_id="automation-ai", fingerprint="original", signing_key=signing_key)

    licenses_path = preset_dir.parent / "licenses.json"

    key_file = preset_dir.parent / "keys.json"
    key_file.write_text(json.dumps({"catalog": signing_key.hex()}), encoding="utf-8")

    list_args = [
        f"--presets-dir={preset_dir}",
        f"--licenses-path={licenses_path}",
        "--fingerprint",
        "local-override",
        f"--signing-key=catalog={signing_key.hex()}",
        f"--signing-key-file={key_file}",
        "list",
    ]
    bridge.main(list_args)
    output = json.loads(capsys.readouterr().out)
    assert output["presets"][0]["license"]["status"] in {"pending", "fingerprint_mismatch"}
    license_meta = output["presets"][0]["license"].get("metadata")
    assert isinstance(license_meta, dict)
    assert "seat_summary" in license_meta
    assert "subscription_summary" in license_meta
    assert "warning_messages" in output["presets"][0]
    assert "warnings" in output["presets"][0]

    license_payload = preset_dir.parent / "license_payload.json"
    license_payload.write_text(
        json.dumps({"fingerprint": "local-override", "expires_at": "2099-01-01T00:00:00Z"}, ensure_ascii=False),
        encoding="utf-8",
    )

    bridge.main(
        [
            f"--presets-dir={preset_dir}",
            f"--licenses-path={licenses_path}",
            "--fingerprint",
            "local-override",
            f"--signing-key=catalog={signing_key.hex()}",
            f"--signing-key-file={key_file}",
            "activate",
            "--preset-id",
            "automation-ai",
            f"--license-json={license_payload}",
        ]
    )
    activation_output = json.loads(capsys.readouterr().out)
    assert activation_output["preset"]["license"]["status"] == "active"

    bridge.main(list_args)
    refreshed = json.loads(capsys.readouterr().out)
    assert refreshed["presets"][0]["license"]["status"] == "active"

    bridge.main(
        [
            f"--presets-dir={preset_dir}",
            f"--licenses-path={licenses_path}",
            "--fingerprint",
            "local-override",
            f"--signing-key=catalog={signing_key.hex()}",
            f"--signing-key-file={key_file}",
            "deactivate",
            "--preset-id",
            "automation-ai",
        ]
    )
    deactivate_output = json.loads(capsys.readouterr().out)
    assert deactivate_output["preset"]["license"]["status"] in {"pending", "fingerprint_mismatch"}


def test_plan_command_outputs_installation_plan(capsys, preset_dir, signing_key):
    dependency_file = preset_dir / "dependency.json"
    _write_preset(
        dependency_file,
        preset_id="dependency",
        fingerprint="stack",
        signing_key=signing_key,
        version="1.0.0",
    )
    primary_file = preset_dir / "primary.json"
    _write_preset(
        primary_file,
        preset_id="primary",
        fingerprint="stack",
        signing_key=signing_key,
        version="2.0.0",
        dependencies=[{"preset_id": "dependency"}],
    )

    assignments_store = PresetAssignmentStore(preset_dir / ".meta" / "assignments.json")
    assignments_store.assign("primary", "portfolio-a")

    args = [
        f"--presets-dir={preset_dir}",
        f"--licenses-path={preset_dir.parent / 'licenses.json'}",
        "--fingerprint",
        "stack",
        f"--signing-key=catalog={signing_key.hex()}",
        "plan",
        "--preset-id",
        "primary",
    ]
    bridge.main(args)
    output = json.loads(capsys.readouterr().out)

    assert output["selection"] == ["primary"]
    assert output["installOrder"] == ["dependency", "primary"]
    dependencies_payload = output["requiredDependencies"].get("primary")
    assert dependencies_payload and dependencies_payload[0]["presetId"] == "dependency"
    license_summaries = output["licenseSummaries"]
    assert license_summaries["primary"]["licenseMissing"] is True
    assignment_summaries = output["assignmentSummaries"]
    assert assignment_summaries["primary"]["assignedPortfolios"] == ["portfolio-a"]
    assert "pendingAssignments" not in assignment_summaries["primary"]

    portfolio_summaries = output["portfolioSummaries"]
    assert portfolio_summaries["portfolio-a"]["assignedPresets"] == ["primary"]
    assert portfolio_summaries["portfolio-a"]["unlicensedPresets"] == ["primary"]
    assert "portfolio-assignment-unlicensed" in portfolio_summaries["portfolio-a"]["warningCodes"]


def test_install_workflow_assigns_and_stores_preferences(capsys, preset_dir, signing_key):
    preset_file = preset_dir / "automation.json"
    _write_preset(
        preset_file,
        preset_id="automation-ai",
        fingerprint="device-001",
        signing_key=signing_key,
    )

    catalog_dir = preset_dir.parent / "catalog"
    catalog_dir.mkdir()
    catalog_manifest = catalog_dir / "catalog.yaml"
    catalog_manifest.write_text(
        """
schema_version: "1.0"
generated_at: "2024-01-01T00:00:00Z"
presets:
  - id: automation-ai
    name: Automation AI
    version: "1.0.0"
    author:
      name: QA Bot
    artifact: "../presets/automation.json"
""".strip(),
        encoding="utf-8",
    )

    licenses_path = preset_dir.parent / "licenses_index.json"
    license_payload_path = preset_dir.parent / "license_payload.json"
    license_payload_path.write_text(
        json.dumps(
            {
                "preset_id": "automation-ai",
                "allowed_fingerprints": ["device-001"],
                "expires_at": "2099-01-01T00:00:00Z",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    preferences_path = preset_dir.parent / "preferences.json"
    preferences_path.write_text(
        json.dumps(
            {"risk_target": "balanced", "budget": 1250, "max_positions": 4},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    licenses_dir = preset_dir.parent / "licenses"

    args = [
        f"--presets-dir={preset_dir}",
        f"--licenses-path={licenses_path}",
        "--fingerprint",
        "device-001",
        f"--signing-key=catalog={signing_key.hex()}",
        "install",
        "--preset-id",
        "automation-ai",
        "--portfolio-id",
        "master-1",
        "--license-json",
        str(license_payload_path),
        "--licenses-dir",
        str(licenses_dir),
        "--catalog-path",
        str(catalog_dir),
        "--preferences-json",
        str(preferences_path),
    ]

    bridge.main(args)
    output = json.loads(capsys.readouterr().out)

    assert output["install"]["success"] is True
    assert output["assignments"]["master-1"] == ["master-1"]

    license_file = Path(output["licenseFile"])
    assert license_file.exists()
    stored_license = json.loads(license_file.read_text(encoding="utf-8"))
    assert stored_license["allowed_fingerprints"] == ["device-001"]

    preferences_store = Path(output["stores"]["preferences"])
    stored_preferences = json.loads(preferences_store.read_text(encoding="utf-8"))
    pref_entry = stored_preferences["preferences"]["automation-ai"]["master-1"]
    assert pref_entry["preferences"]["budget"] == 1250

    license_index = json.loads(licenses_path.read_text(encoding="utf-8"))
    stored_entry = license_index["licenses"]["automation-ai"]
    assert stored_entry["allowed_fingerprints"] == ["device-001"]
