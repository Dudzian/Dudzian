from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pytest

from bot_core.marketplace import PresetDocument, PresetSignatureVerification
from bot_core.strategies.installer import MarketplaceInstallResult
from bot_core.ui.api import MarketplaceService


def _document(preset_id: str, version: str) -> PresetDocument:
    payload = {
        "name": preset_id,
        "metadata": {
            "id": preset_id,
            "version": version,
        },
    }
    return PresetDocument(
        payload=payload,
        signature=None,
        verification=PresetSignatureVerification(True, ()),
        fmt="json",
        path=None,
        issues=tuple(),
    )


@dataclass
class _Descriptor:
    preset_id: str
    name: str = "alpha"
    version: str = "1.0.0"
    summary: str | None = None
    required_exchanges: Sequence[str] = ()
    tags: Sequence[str] = ()
    license_tier: str | None = None
    artifact_path: Path = Path("/tmp/alpha.json")


class _InstallerStub:
    def __init__(self, document: PresetDocument, preview: MarketplaceInstallResult) -> None:
        self._document = document
        self._preview = preview

    def list_available(self) -> Sequence[_Descriptor]:
        return (_Descriptor(preset_id=self._document.preset_id, version=self._document.version or "1.0.0"),)

    def preview_installation(self, preset_id: str) -> MarketplaceInstallResult:
        assert preset_id == self._document.preset_id
        return self._preview

    def load_catalog_document(self, preset_id: str) -> PresetDocument:
        assert preset_id == self._document.preset_id
        return self._document


class _RepositoryStub:
    def __init__(self, root: Path) -> None:
        self.root = root

    def load_all(self, *_, **__) -> Sequence[PresetDocument]:  # type: ignore[override]
        return ()

    def remove(self, preset_id: str) -> bool:
        return False

    def export_preset(self, preset_id: str, *, format: str = "json") -> tuple[dict[str, Any], bytes]:
        return {"metadata": {"id": preset_id}}, b"{}"


@pytest.fixture()
def service(tmp_path: Path) -> MarketplaceService:
    document = _document("alpha", "1.0.0")
    license_payload = {
        "status": "active",
        "seat_summary": {
            "total": 2,
            "in_use": 1,
            "assignments": ["portfolio-a"],
            "pending": ["portfolio-b"],
        },
        "validation": {"warning_messages": ["Istniejące ostrzeżenie"]},
    }
    preview = MarketplaceInstallResult(
        preset_id="alpha",
        version="1.0.0",
        success=True,
        installed_path=None,
        signature_verified=True,
        fingerprint_verified=True,
        issues=(),
        warnings=(),
        license_payload=license_payload,
    )
    installer = _InstallerStub(document, preview)
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    repository = _RepositoryStub(repository_root)
    return MarketplaceService(installer, repository)  # type: ignore[arg-type]


def test_plan_payload_merges_assignment_warnings(service: MarketplaceService) -> None:
    service.assign_to_portfolio("alpha", "portfolio-a")
    service.assign_to_portfolio("alpha", "portfolio-b")
    service.assign_to_portfolio("alpha", "portfolio-c")

    payload = service.plan_installation_payload(["alpha"])

    license_summary = payload["licenseSummaries"]["alpha"]
    assignment_summary = payload["assignmentSummaries"]["alpha"]

    assert assignment_summary["licensedAssignments"] == ["portfolio-a"]
    assert assignment_summary["unlicensedAssignments"] == ["portfolio-b", "portfolio-c"]
    assert assignment_summary["pendingAssignments"] == ["portfolio-b"]
    assert assignment_summary["seatLimit"] == 2
    assert assignment_summary["seatShortfall"] == 1
    assert "assignment-unlicensed" in assignment_summary["warningCodes"]
    assert "assignment-seat-shortfall" in assignment_summary["warningCodes"]
    assert "assignment-pending" in assignment_summary["warningCodes"]
    assert any("Brak licencji" in msg for msg in assignment_summary["warningMessages"])
    assert any("Brakuje" in msg for msg in assignment_summary["warningMessages"])
    assert any("oczekują" in msg for msg in assignment_summary["warningMessages"])  # type: ignore[str-bytes-safe]

    assert "Istniejące ostrzeżenie" in license_summary["warningMessages"]
    assert any("Brak licencji" in msg for msg in license_summary["warningMessages"])
    assert any("Brakuje" in msg for msg in license_summary["warningMessages"])
    assert "assignment-unlicensed" in license_summary["warningCodes"]
    assert "assignment-seat-shortfall" in license_summary["warningCodes"]
    assert "assignment-pending" in license_summary["warningCodes"]

    portfolio_summaries = payload["portfolioSummaries"]
    summary_a = portfolio_summaries["portfolio-a"]
    assert summary_a["assignedPresets"] == ["alpha"]
    assert summary_a["licensedPresets"] == ["alpha"]
    assert "portfolio-assignment-unlicensed" not in summary_a.get("warningCodes", [])
    assert "portfolio-seat-shortfall" in summary_a.get("warningCodes", [])

    summary_b = portfolio_summaries["portfolio-b"]
    assert summary_b["assignedPresets"] == ["alpha"]
    assert summary_b["unlicensedPresets"] == ["alpha"]
    assert summary_b["pendingPresets"] == ["alpha"]
    assert "portfolio-assignment-unlicensed" in summary_b["warningCodes"]
    assert "portfolio-assignment-pending" in summary_b["warningCodes"]

    summary_c = portfolio_summaries["portfolio-c"]
    assert summary_c["assignedPresets"] == ["alpha"]
    assert summary_c["unlicensedPresets"] == ["alpha"]
    assert "portfolio-assignment-unlicensed" in summary_c["warningCodes"]
