from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from bot_core.security.hwid import HwIdProvider
from bot_core.security.signing import build_hmac_signature
from bot_core.strategies.catalog import (
    PresetLicenseState,
    StrategyCatalog,
    StrategyPresetProfile,
)


def _build_preset_document(
    *,
    preset_id: str,
    profile: str,
    fingerprint: str,
    expires_at: datetime,
    signing_key: bytes,
) -> dict[str, object]:
    preset_payload = {
        "name": "Premium Grid",
        "strategies": [
            {
                "name": "grid-entry",
                "engine": "grid_trading",
                "parameters": {"grid_size": 5, "grid_spacing": 0.01},
                "license_tier": "professional",
                "risk_classes": ["market_making"],
                "required_data": ["order_book"],
            }
        ],
        "metadata": {
            "id": preset_id,
            "profile": profile,
            "license": {
                "module_id": f"module::{preset_id}",
                "fingerprint": fingerprint,
                "expires_at": expires_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                "edition": "enterprise",
            },
        },
    }
    signature = build_hmac_signature(preset_payload, key=signing_key, key_id="catalog")
    return {"preset": preset_payload, "signature": signature}


@pytest.fixture()
def signing_key() -> bytes:
    return b"catalog-secret"


def test_load_preset_with_valid_signature(tmp_path, signing_key):
    preset_path = tmp_path / "grid.json"
    expires = datetime(2099, 1, 1, tzinfo=timezone.utc)
    document = _build_preset_document(
        preset_id="grid-pro",
        profile="dca",
        fingerprint="hwid-123",
        expires_at=expires,
        signing_key=signing_key,
    )
    preset_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")

    provider = HwIdProvider(fingerprint_reader=lambda: "hwid-123")
    catalog = StrategyCatalog(hwid_provider=provider)
    catalog.load_presets_from_directory(
        tmp_path,
        signing_keys={"catalog": signing_key},
        hwid_provider=provider,
    )

    descriptor = catalog.preset("grid-pro", hwid_provider=provider)
    assert descriptor.profile == StrategyPresetProfile.DCA
    assert descriptor.signature_verified is True
    assert descriptor.license_status.status is PresetLicenseState.ACTIVE
    assert descriptor.license_status.fingerprint_verified is True
    assert descriptor.required_parameters["grid-entry"] == ("grid_size", "grid_spacing")


def test_license_override_activation_cycle(tmp_path, signing_key):
    preset_path = tmp_path / "trend.json"
    expires = datetime(2099, 1, 1, tzinfo=timezone.utc)
    document = _build_preset_document(
        preset_id="trend-ai",
        profile="ai",
        fingerprint="original-hwid",
        expires_at=expires,
        signing_key=signing_key,
    )
    preset_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")

    provider = HwIdProvider(fingerprint_reader=lambda: "override-hwid")
    catalog = StrategyCatalog(hwid_provider=provider)
    catalog.load_presets_from_directory(
        tmp_path,
        signing_keys={"catalog": signing_key},
        hwid_provider=provider,
    )

    descriptor = catalog.preset("trend-ai", hwid_provider=provider)
    assert descriptor.license_status.status is PresetLicenseState.FINGERPRINT_MISMATCH

    override = {"fingerprint": "override-hwid", "expires_at": "2099-01-01T00:00:00Z"}
    activated = catalog.install_license_override("trend-ai", override, hwid_provider=provider)
    assert activated.license_status.status is PresetLicenseState.ACTIVE

    reverted = catalog.clear_license_override("trend-ai", hwid_provider=provider)
    assert reverted.license_status.status is PresetLicenseState.FINGERPRINT_MISMATCH


def test_describe_presets_filters_profile(tmp_path, signing_key):
    preset_path = tmp_path / "combo.json"
    expires = datetime(2099, 1, 1, tzinfo=timezone.utc)
    document = _build_preset_document(
        preset_id="combo-pack",
        profile="grid",
        fingerprint="combo-hwid",
        expires_at=expires,
        signing_key=signing_key,
    )
    preset_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")

    provider = HwIdProvider(fingerprint_reader=lambda: "combo-hwid")
    catalog = StrategyCatalog(hwid_provider=provider)
    catalog.load_presets_from_directory(
        tmp_path,
        signing_keys={"catalog": signing_key},
        hwid_provider=provider,
    )

    summaries = catalog.describe_presets(profile="grid", include_strategies=False, hwid_provider=provider)
    assert len(summaries) == 1
    assert summaries[0]["preset_id"] == "combo-pack"
    assert summaries[0]["profile"] == "grid"

    empty = catalog.describe_presets(profile="ai", include_strategies=False, hwid_provider=provider)
    assert empty == []
