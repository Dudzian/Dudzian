from __future__ import annotations

import math

import pandas as pd
import pytest
from hypothesis import given, strategies as st

from bot_core.risk.portfolio import RiskManagement


@pytest.mark.security
@given(st.sampled_from([math.nan, math.inf, -math.inf]))
def test_position_limit_rejects_non_finite_size(size: float) -> None:
    manager = RiskManagement({"max_portfolio_risk": 0.25, "max_risk_per_trade": 0.05})

    allowed, reason = manager.check_position_limits(
        {"symbol": "BTC/USDT", "size": size},
        {},
    )

    assert allowed is False
    assert "finite" in reason.lower()


@pytest.mark.security
@given(
    size=st.floats(min_value=0.000001, max_value=0.2, allow_nan=False, allow_infinity=False),
    existing=st.floats(min_value=0.0, max_value=0.24, allow_nan=False, allow_infinity=False),
)
def test_position_limit_never_accepts_excess_risk(size: float, existing: float) -> None:
    manager = RiskManagement({"max_portfolio_risk": 0.25, "max_risk_per_trade": 0.05})
    existing_volatility = 0.2
    portfolio = {"ETH/USDT": {"size": existing, "volatility": existing_volatility}}

    allowed, _ = manager.check_position_limits(
        {"symbol": "BTC/USDT", "size": size},
        portfolio,
    )

    existing_heat = existing * existing_volatility
    if size > manager.max_risk_per_trade or existing_heat + size > manager.max_portfolio_risk:
        assert allowed is False


@pytest.mark.security
@given(
    strength=st.one_of(
        st.sampled_from([math.nan, math.inf, -math.inf]),
        st.floats(max_value=-0.000001, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.000001, allow_nan=False, allow_infinity=False),
    ),
)
def test_invalid_signal_fails_closed_without_position(strength: float) -> None:
    manager = RiskManagement({"max_portfolio_risk": 0.25, "max_risk_per_trade": 0.05})
    market = pd.DataFrame({"close": [100.0, 101.0, 99.0]})

    result = manager.calculate_position_size(
        "BTC/USDT",
        {"strength": strength, "confidence": 0.8},
        market,
        {},
    )

    assert result.recommended_size == 0.0
    assert result.max_allowed_size == 0.0
    assert math.isfinite(result.risk_adjusted_size)
