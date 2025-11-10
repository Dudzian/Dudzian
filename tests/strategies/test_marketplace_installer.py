from datetime import datetime, timedelta, timezone
from pathlib import Path

from bot_core.marketplace import PresetRepository
from bot_core.security.hwid import HwIdProvider
from bot_core.strategies.installer import MarketplacePresetInstaller
from bot_core.strategies.marketplace import load_catalog

MARKETPLACE_DIR = Path(__file__).resolve().parents[2] / "bot_core" / "strategies" / "marketplace"
LICENSES_DIR = MARKETPLACE_DIR / "licenses"


def test_load_catalog_contains_demo_preset() -> None:
    catalog = load_catalog(MARKETPLACE_DIR)
    preset = catalog.find("mean_reversion_demo")
    assert preset is not None
    assert preset.name == "Mean Reversion Demo"
    assert "BINANCE" in preset.required_exchanges
    assert preset.signature is not None


def test_installer_validates_signature_and_fingerprint(tmp_path: Path) -> None:
    repository = PresetRepository(tmp_path / "presets")
    installer = MarketplacePresetInstaller(
        repository,
        catalog_path=MARKETPLACE_DIR,
        licenses_dir=LICENSES_DIR,
        hwid_provider=HwIdProvider(fingerprint_reader=lambda: "OEM-TEST-12345"),
    )

    result = installer.install_from_catalog("mean_reversion_demo")

    assert result.success is True
    assert result.signature_verified is True
    assert result.fingerprint_verified is True
    assert result.installed_path is not None
    installed_docs = repository.load_all()
    assert installed_docs and installed_docs[0].preset_id == "mean_reversion_demo"


def test_installer_reports_missing_license(tmp_path: Path) -> None:
    repository = PresetRepository(tmp_path / "presets")
    installer = MarketplacePresetInstaller(
        repository,
        catalog_path=MARKETPLACE_DIR,
        licenses_dir=tmp_path / "licenses",  # brak licencji
        hwid_provider=HwIdProvider(fingerprint_reader=lambda: "OEM-TEST-00001"),
    )

    preview = installer.preview_installation("mean_reversion_demo")

    assert preview.success is False
    assert "license-missing" in preview.issues
    assert preview.signature_verified is True
    assert preview.fingerprint_verified is None


def test_validate_license_payload_includes_seat_and_subscription(tmp_path: Path) -> None:
    repository = PresetRepository(tmp_path / "presets")

    class _DummyCatalog:
        presets: tuple = ()

        def find(self, preset_id: str) -> None:
            return None

    installer = MarketplacePresetInstaller(
        repository,
        catalog=_DummyCatalog(),
        licenses_dir=None,
        hwid_provider=HwIdProvider(fingerprint_reader=lambda: "OEM-DEVICE-001"),
    )

    payload = {
        "preset_id": "demo",
        "allowed_versions": ["1.0.0"],
        "allowed_fingerprints": ["OEM-DEVICE-001"],
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=10)).isoformat().replace("+00:00", "Z"),
        "seat_policy": {
            "total": 1,
            "assignments": [],
            "enforcement": "hard",
            "auto_assign": False,
        },
        "subscription": {
            "status": "paused",
            "current_period": {
                "start": "2025-01-01T00:00:00Z",
                "end": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat().replace("+00:00", "Z"),
            },
            "grace_period_days": 1,
        },
    }

    success, fingerprint_ok, issues, warnings, normalized = installer._validate_license(
        "demo",
        "1.0.0",
        payload,
    )

    assert success is False
    assert fingerprint_ok is True
    assert "license.seats.fingerprint_not_assigned" in issues
    assert "license.subscription.status" in warnings
    assert "license-expiring-soon" in warnings
    assert normalized is not None
    assert normalized["seat_summary"]["total"] == 1
    assert normalized["seat_summary"]["enforcement"] == "hard"
    assert normalized["subscription_summary"]["status"] == "paused"
    validation = normalized.get("validation")
    assert isinstance(validation, dict)
    assert "warning_messages" in validation
    assert any("subskry" in message.lower() for message in validation["warning_messages"])
    assert "warning_codes" in validation
    assert "license.subscription.status" in validation["warning_codes"]
