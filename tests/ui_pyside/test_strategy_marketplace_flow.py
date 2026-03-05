from __future__ import annotations

from typing import Any

import json
from pathlib import Path

from bot_core.strategies.installer import MarketplaceInstallResult
from ui.backend.runtime_service import RuntimeService
from ui.pyside_app.controllers.strategy import StrategyManagementController
import yaml


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
                "activation": json.dumps(
                    {
                        "preset_name": "grid_classic",
                        "preset_hash": "hash-grid",
                        "regime": "trend",
                        "used_fallback": False,
                    }
                ),
                "guardrail_transition": json.dumps(
                    {
                        "timestamp": "2024-01-01T00:00:00+00:00",
                        "active": True,
                        "previous_active": False,
                        "reasons": ["latency"],
                    }
                ),
            },
        }
    ]
    service._refresh_activation_summary()  # noqa: SLF001

    summary = service.regimeActivationSummary
    assert summary, "Oczekiwano podsumowania aktywacji"
    payload = json.loads(summary)
    assert payload["activePreset"]["preset_name"] == "grid_classic"
    assert payload["guardrailTrace"][0]["active"] is True


def test_strategy_controller_bundle_export_and_cloud_toggle(tmp_path: Path) -> None:
    service = _StubMarketplaceService()
    runtime_config = tmp_path / "config" / "runtime.yaml"
    runtime_config.parent.mkdir(parents=True, exist_ok=True)
    runtime_config.write_text("cloud:\n  enabled_signed: false\n", encoding="utf-8")
    controller = StrategyManagementController(
        marketplace_service=service, runtime_config_path=runtime_config
    )

    selection = [
        {"presetId": "grid_classic", "label": "Grid Classic", "order": 2, "mode": "live"},
        {"presetId": "grid_pro", "label": "Grid Pro", "order": 1, "mode": "paper"},
    ]
    result = controller.createPresetBundle(
        "Test Bundle", selection, {"bundleMode": "sequential", "cloudEnabled": True}
    )
    assert result["success"] is True
    bundle_path = Path(result["path"])
    assert bundle_path.exists()
    data = yaml.safe_load(bundle_path.read_text(encoding="utf-8"))
    assert data["presets"][0]["presetId"] == "grid_pro"
    assert data["cloudEnabled"] is True

    assert controller.cloudRuntimeEnabled is False
    toggle_result = controller.setCloudRuntimeEnabled(True)
    assert toggle_result["success"] is True
    assert controller.cloudRuntimeEnabled is True
    config_payload = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
    assert config_payload["cloud"]["enabled_signed"] is True
