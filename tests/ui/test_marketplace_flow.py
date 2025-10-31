from __future__ import annotations

from pathlib import Path

from bot_core.marketplace import PresetRepository
from bot_core.security.hwid import HwIdProvider
from bot_core.strategies.installer import MarketplacePresetInstaller
from bot_core.ui.api import MarketplaceService

REPO_ROOT = Path(__file__).resolve().parents[2]
MARKETPLACE_DIR = REPO_ROOT / "bot_core" / "strategies" / "marketplace"
LICENSES_DIR = MARKETPLACE_DIR / "licenses"


def _build_service(tmp_path: Path) -> MarketplaceService:
    repository = PresetRepository(tmp_path / "presets")
    installer = MarketplacePresetInstaller(
        repository,
        catalog_path=MARKETPLACE_DIR,
        licenses_dir=LICENSES_DIR,
        hwid_provider=HwIdProvider(fingerprint_reader=lambda: "OEM-TEST-DEVICE"),
    )
    return MarketplaceService(installer, repository)


def test_marketplace_service_lists_and_installs(tmp_path: Path) -> None:
    service = _build_service(tmp_path)

    listings = service.list_presets()
    assert listings, "Katalog marketplace powinien zawierać co najmniej jeden preset"
    entry = listings[0]
    assert entry.signature_verified is True
    assert "BINANCE" in entry.required_exchanges

    result = service.install_from_catalog(entry.preset_id)
    assert result.success is True

    payloads = service.list_presets_payload()
    selected = next(item for item in payloads if item["presetId"] == entry.preset_id)
    assert selected["installed"] is True
    assert selected["installedVersion"] == entry.version

    exported_payload, blob = service.export_preset(entry.preset_id)
    assert exported_payload["metadata"]["id"] == entry.preset_id
    assert len(blob) > 0

    assert service.remove_preset(entry.preset_id) is True

    copy_path = tmp_path / "local_copy.json"
    copy_path.write_bytes((MARKETPLACE_DIR / "presets" / "mean_reversion_demo.json").read_bytes())
    reinstall = service.install_from_file(copy_path)
    assert reinstall.success is True
