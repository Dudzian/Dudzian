from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bot_core.exchanges.manager import ExchangeManager
from bot_core.strategies.catalog import (
    PresetLicenseState,
    PresetLicenseStatus,
    StrategyCatalog,
    StrategyPresetDescriptor,
    StrategyPresetProfile,
)


def _build_descriptor(preset_id: str, *, profile: StrategyPresetProfile) -> StrategyPresetDescriptor:
    license_status = PresetLicenseStatus(
        preset_id=preset_id,
        module_id=f"module-{preset_id}",
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
    return StrategyPresetDescriptor(
        preset_id=preset_id,
        name=f"Preset {preset_id}",
        profile=profile,
        strategies=(
            {
                "name": f"{preset_id}-grid",
                "engine": "GridTradingStrategy",
                "parameters": {"step": 0.1, "levels": 3},
            },
        ),
        required_parameters={f"{preset_id}-grid": ("step",)},
        license_status=license_status,
        signature_verified=True,
        metadata={"notes": f"demo-{preset_id}"},
    )


def test_strategy_context_sync_and_assignment() -> None:
    catalog = StrategyCatalog()
    primary = _build_descriptor("alpha", profile=StrategyPresetProfile.GRID)
    fallback = _build_descriptor("beta", profile=StrategyPresetProfile.AI)
    catalog._presets[primary.preset_id] = primary  # type: ignore[attr-defined]
    catalog._presets[fallback.preset_id] = fallback  # type: ignore[attr-defined]

    manager = ExchangeManager()
    manager.set_strategy_catalog(catalog)

    snapshot = manager.describe_strategy_contexts()
    assert snapshot["mode"] == "paper"
    assert set(snapshot["contexts"]) >= {"paper", "spot", "margin", "futures", "live"}
    assert snapshot["contexts"]["paper"]["presets"][0]["preset_id"] == "alpha"

    binding = manager.activate_strategy_preset(
        "paper",
        "alpha",
        parameter_overrides={"alpha-grid": {"step": 0.25}},
    )
    assert binding["preset_id"] == "alpha"
    assert binding["strategies"]["alpha-grid"]["step"] == pytest.approx(0.25)

    active = manager.strategy_assignment("paper")
    assert active is not None
    assert active["metadata"]["notes"] == "demo-alpha"

    # Alias live -> spot uses to mirror assignments
    live_snapshot = manager.describe_strategy_contexts()["contexts"]["live"]
    assert live_snapshot["mode"] == "spot"

    assert manager.clear_strategy_assignment("paper") is True
    assert manager.strategy_assignment("paper") is None

    with pytest.raises(ValueError):
        manager.activate_strategy_preset("unknown", "alpha")

