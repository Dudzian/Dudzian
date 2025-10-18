import pytest

from bot_core.runtime.metadata import RiskManagerSettings

import KryptoLowca.run_trading_gui_paper as paper_launcher
from KryptoLowca.ui.trading.risk_helpers import RiskSnapshot


def test_derive_risk_defaults_respects_runtime_settings() -> None:
    snapshot = RiskSnapshot(
        paper_balance=50_000.0,
        profile_name="balanced",
        settings=RiskManagerSettings(
            max_risk_per_trade=0.04,
            max_daily_loss_pct=0.12,
            max_portfolio_risk=0.25,
            max_positions=6,
            emergency_stop_drawdown=0.2,
        ),
    )

    capital, risk_pct, portfolio_pct, notional = paper_launcher._derive_risk_defaults(snapshot)

    assert capital == pytest.approx(50_000.0)
    assert risk_pct == pytest.approx(4.0)
    assert portfolio_pct == pytest.approx(25.0)
    assert notional == pytest.approx(2_000.0)


def test_derive_risk_defaults_fallback_to_defaults() -> None:
    snapshot = RiskSnapshot(paper_balance=0.0, profile_name=None, settings=None)

    capital, risk_pct, portfolio_pct, notional = paper_launcher._derive_risk_defaults(snapshot)

    assert capital == pytest.approx(paper_launcher.DEFAULT_CAPITAL_USDT)
    assert risk_pct == pytest.approx(paper_launcher.DEFAULT_RISK_PCT)
    assert portfolio_pct == pytest.approx(paper_launcher.DEFAULT_PORTFOLIO_PCT)
    assert notional == pytest.approx(paper_launcher.DEFAULT_NOTIONAL_USDT)
