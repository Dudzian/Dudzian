from __future__ import annotations

from types import SimpleNamespace

from datetime import datetime, timezone

from bot_core.runtime.pipeline import (
    build_multi_portfolio_scheduler_from_config,
    describe_multi_portfolio_state,
)
from bot_core.security.hwid import HwIdProvider
from bot_core.strategies.catalog import (
    PresetLicenseState,
    PresetLicenseStatus,
    StrategyCatalog,
    StrategyPresetDescriptor,
    StrategyPresetProfile,
)


def build_catalog() -> StrategyCatalog:
    provider = HwIdProvider(fingerprint_reader=lambda: "test-hwid")
    catalog = StrategyCatalog(hwid_provider=provider)
    status = PresetLicenseStatus(
        preset_id="preset-main",
        module_id="module-main",
        status=PresetLicenseState.ACTIVE,
        fingerprint=None,
        fingerprint_candidates=(),
        fingerprint_verified=True,
        activated_at=datetime.now(timezone.utc),
        expires_at=None,
        edition="pro",
        capability="grid",
        signature_verified=True,
        issues=(),
        metadata={},
    )
    descriptor = StrategyPresetDescriptor(
        preset_id="preset-main",
        name="Preset Main",
        profile=StrategyPresetProfile.GRID,
        strategies=({"name": "grid", "engine": "GridStrategy", "parameters": {}},),
        required_parameters={"grid": ()},
        license_status=status,
        signature_verified=True,
        metadata={},
    )
    catalog._presets[descriptor.preset_id] = descriptor  # type: ignore[attr-defined]
    catalog.install_license_override("preset-main", {"fingerprint": "test-hwid"})
    return catalog


def test_build_multi_portfolio_scheduler_from_config() -> None:
    catalog = build_catalog()
    config = SimpleNamespace(
        multi_portfolio=[
            {
                "portfolio_id": "master-a",
                "primary_preset": "preset-main",
                "fallback_presets": ["preset-main"],
                "followers": [
                    {"portfolio_id": "follower-1", "scaling": 0.5},
                    {"portfolio_id": "follower-2", "scaling": 0.75, "risk_multiplier": 1.2},
                ],
            }
        ]
    )

    scheduler = build_multi_portfolio_scheduler_from_config(
        core_config=config,
        catalog=catalog,
    )
    state = describe_multi_portfolio_state(scheduler)
    assert state[0]["portfolio_id"] == "master-a"
    assert "follower-1" in state[0]["followers"]
