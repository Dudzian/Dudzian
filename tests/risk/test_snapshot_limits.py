from __future__ import annotations

from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.risk.profiles.balanced import BalancedProfile


def test_snapshot_limits_are_ui_safe() -> None:
    engine = ThresholdRiskEngine()
    profile = BalancedProfile()
    engine.register_profile(profile)

    snapshot = engine.snapshot_state(profile.name)
    assert snapshot is not None

    limits = snapshot.get("limits")
    assert isinstance(limits, dict)

    values = list(limits.values())
    assert values, "Snapshot should expose normalized risk limits for UI"
    assert all(isinstance(value, float) for value in values)
    assert not any(isinstance(value, (tuple, list)) for value in values)

    assert "trade_risk_pct_range_min" in limits
    assert "trade_risk_pct_range_max" in limits
    assert "trade_risk_pct_range" not in limits
