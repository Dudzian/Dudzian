import json
from datetime import datetime, timezone
from pathlib import Path

from bot_core.marketplace import PresetRepository
from bot_core.marketplace.assignments import PresetAssignmentStore
from bot_core.marketplace.preferences import PresetPreferenceStore
from bot_core.security.hwid import HwIdProvider
from bot_core.security.signing import build_hmac_signature
from bot_core.strategies.catalog import StrategyCatalog
from bot_core.strategies.installer import MarketplacePresetInstaller
from bot_core.ui.api import MarketplaceService


def _write_signed_preset(path: Path, *, preset_id: str, signing_key: bytes) -> None:
    payload = {
        "name": preset_id,
        "metadata": {
            "id": preset_id,
            "version": "1.0.0",
            "license": {
                "module_id": f"module::{preset_id}",
                "fingerprint": "device-xyz",
                "expires_at": datetime(2099, 1, 1, tzinfo=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            },
        },
        "strategies": [
            {
                "name": f"{preset_id}-strategy",
                "engine": "mean_reversion",
                "parameters": {
                    "budget": 1000,
                    "risk_multiplier": 1.0,
                    "leverage": 2.0,
                    "max_positions": 3,
                },
            }
        ],
    }
    signature = build_hmac_signature(payload, key=signing_key, key_id="catalog")
    document = {"preset": payload, "signature": signature}
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_catalog(path: Path, preset_id: str) -> None:
    manifest = f"""
schema_version: "1.0"
generated_at: "2024-01-01T00:00:00Z"
presets:
  - id: {preset_id}
    name: {preset_id}
    version: "1.0.0"
    author:
      name: QA Bot
    artifact: "../artifacts/{preset_id}.json"
""".strip()
    path.write_text(manifest, encoding="utf-8")


def test_marketplace_install_workflow_end_to_end(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    license_dir = tmp_path / "licenses"
    license_dir.mkdir()

    signing_key = b"catalog-secret"
    preset_id = "automation-ai"
    _write_signed_preset(artifacts_dir / f"{preset_id}.json", preset_id=preset_id, signing_key=signing_key)
    _write_catalog(catalog_dir / "catalog.yaml", preset_id)

    license_payload = {
        "preset_id": preset_id,
        "allowed_fingerprints": ["device-xyz"],
        "expires_at": "2099-01-01T00:00:00Z",
    }
    (license_dir / f"{preset_id}.json").write_text(
        json.dumps(license_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    repository = PresetRepository(tmp_path / "installed")
    installer = MarketplacePresetInstaller(
        repository,
        catalog_path=catalog_dir,
        licenses_dir=license_dir,
        signing_keys={"catalog": signing_key},
        hwid_provider=HwIdProvider(fingerprint_reader=lambda: "device-xyz"),
    )
    service = MarketplaceService(installer, repository)

    workflow_result = service.install_preset_workflow(
        preset_id,
        ["master-portfolio"],
        user_preferences={"risk_target": "balanced", "budget": 2000, "max_positions": 5},
    )

    assert workflow_result["install"]["success"] is True
    assignments_store = PresetAssignmentStore(repository.root / ".meta" / "assignments.json")
    assert "master-portfolio" in assignments_store.assigned_portfolios(preset_id)

    preferences_store = PresetPreferenceStore(repository.root / ".meta" / "preferences.json")
    preference_entry = preferences_store.entry(preset_id, "master-portfolio")
    assert preference_entry is not None
    assert preference_entry["preferences"]["budget"] == 2000

    catalog = StrategyCatalog(hwid_provider=HwIdProvider(fingerprint_reader=lambda: "device-xyz"))
    catalog.load_presets_from_directory(repository.root, signing_keys={"catalog": signing_key})
    catalog.install_license_override(preset_id, license_payload, hwid_provider=HwIdProvider(fingerprint_reader=lambda: "device-xyz"))
    descriptor = catalog.preset(preset_id)
    assert descriptor.preset_id == preset_id
    assert descriptor.license_status.status.value.lower() not in {"invalid", "revoked"}

    preferences_payload = service.preferences_payload()
    assert preset_id in preferences_payload
    assert "master-portfolio" in preferences_payload[preset_id]
