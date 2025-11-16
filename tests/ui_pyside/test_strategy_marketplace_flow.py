from __future__ import annotations

from typing import Any

import json

from bot_core.strategies.installer import MarketplaceInstallResult
from ui.backend.runtime_service import RuntimeService
from ui.pyside_app.controllers.strategy import StrategyManagementController


class _StubMarketplaceService:
    def __init__(self) -> None:
        self._presets: list[dict[str, Any]] = [
            {
                "presetId": "grid_classic",
                "name": "Grid Classic",
                "version": "1.0",
                "summary": "Klasyczny grid na rynek spot.",
                "tags": ["scalping", "grid"],
                "license": {"status": "pending"},
                "assignedPortfolios": [],
            }
        ]
        self.install_calls: list[str] = []
        self.assign_calls: list[tuple[str, str]] = []
        self._assignments: dict[str, set[str]] = {}

    def list_presets_payload(self) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for preset in self._presets:
            entry = dict(preset)
            entry["assignedPortfolios"] = sorted(self._assignments.get(entry["presetId"], set()))
            payload.append(entry)
        return payload

    def install_from_catalog(self, preset_id: str) -> MarketplaceInstallResult:
        self.install_calls.append(preset_id)
        return MarketplaceInstallResult(
            preset_id=preset_id,
            version="1.0",
            success=True,
            installed_path=None,
            signature_verified=True,
            fingerprint_verified=True,
            issues=(),
            warnings=(),
            license_payload={"status": "active"},
        )

    def assign_to_portfolio(self, preset_id: str, portfolio_id: str) -> tuple[str, ...]:
        self.assign_calls.append((preset_id, portfolio_id))
        bucket = self._assignments.setdefault(preset_id, set())
        if portfolio_id:
            bucket.add(portfolio_id)
        return tuple(sorted(bucket))


def test_strategy_management_controller_applies_and_refreshes() -> None:
    service = _StubMarketplaceService()
    controller = StrategyManagementController(marketplace_service=service)
    controller.refreshMarketplace()
    assert controller.presets, "Brak presetów początkowych"

    result = controller.activateAndAssign("grid_classic", "portfolio-alpha")
    assert result["success"] is True
    assert service.install_calls == ["grid_classic"]
    assert service.assign_calls[-1] == ("grid_classic", "portfolio-alpha")

    updated_presets = controller.presets
    assert updated_presets[0]["assignedPortfolios"] == ["portfolio-alpha"]
    assert "Zainstalowano" in controller.statusMessage


def test_runtime_service_exposes_activation_summary() -> None:
    service = RuntimeService(decision_loader=lambda limit: [])
    service._decisions = [  # noqa: SLF001 - ustawienie stanu testowego
        {
            "event": "decision_ready",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "status": "approved",
            "metadata": {
                "activation": json.dumps({
                    "preset_name": "grid_classic",
                    "preset_hash": "hash-grid",
                    "regime": "trend",
                    "used_fallback": False,
                }),
                "guardrail_transition": json.dumps({
                    "timestamp": "2024-01-01T00:00:00+00:00",
                    "active": True,
                    "previous_active": False,
                    "reasons": ["latency"],
                }),
            },
        }
    ]
    service._refresh_activation_summary()  # noqa: SLF001

    summary = service.regimeActivationSummary
    assert summary, "Oczekiwano podsumowania aktywacji"
    payload = json.loads(summary)
    assert payload["activePreset"]["preset_name"] == "grid_classic"
    assert payload["guardrailTrace"][0]["active"] is True
