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
