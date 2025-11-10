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
    assert "warnings" in selected
    assert "warningMessages" in selected
    license_payload = selected.get("license")
    assert isinstance(license_payload, dict)
    assert "seat_summary" in license_payload
    validation_payload = license_payload.get("validation")
    if validation_payload is not None:
        assert "warning_messages" in validation_payload

    exported_payload, blob = service.export_preset(entry.preset_id)
    assert exported_payload["metadata"]["id"] == entry.preset_id
    assert len(blob) > 0

    assert service.remove_preset(entry.preset_id) is True

    copy_path = tmp_path / "local_copy.json"
    copy_path.write_bytes((MARKETPLACE_DIR / "presets" / "mean_reversion_demo.json").read_bytes())
    reinstall = service.install_from_file(copy_path)
    assert reinstall.success is True

    assigned_portfolio = "TEST-PORTFOLIO"
    service.assign_to_portfolio(entry.preset_id, assigned_portfolio)

    plan = service.plan_installation([entry.preset_id])
    assert plan.install_order
    plan_payload = service.plan_installation_payload([entry.preset_id])
    assert plan_payload["selection"] == [entry.preset_id]
    assert plan_payload["installOrder"]
    license_summaries = plan_payload["licenseSummaries"]
    assert entry.preset_id in license_summaries
    plan_summary = license_summaries[entry.preset_id]
    assert plan_summary["presetId"] == entry.preset_id
    assert plan_summary.get("licenseMissing") is not True
    assert plan_summary["signatureVerified"] is True
    assert isinstance(plan_summary.get("license"), dict)
    assert isinstance(plan_summary.get("warningMessages"), list)

    assignment_summaries = plan_payload["assignmentSummaries"]
    assert entry.preset_id in assignment_summaries
    assignment_summary = assignment_summaries[entry.preset_id]
    assert assignment_summary["presetId"] == entry.preset_id
    assert assigned_portfolio in assignment_summary["assignedPortfolios"]
    assert assignment_summary["assignedCount"] >= 1
    portfolio_summaries = plan_payload["portfolioSummaries"]
    assert assigned_portfolio in portfolio_summaries
    portfolio_summary = portfolio_summaries[assigned_portfolio]
    assert entry.preset_id in portfolio_summary.get("assignedPresets", [])
