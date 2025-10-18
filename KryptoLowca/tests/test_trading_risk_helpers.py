from types import SimpleNamespace

import pytest

from bot_core.runtime.metadata import RiskManagerSettings

from KryptoLowca.ui.trading.risk_helpers import (
    RiskSnapshot,
    build_risk_limits_summary,
    build_risk_profile_hint,
    compute_default_notional,
    format_decimal,
    format_notional,
    snapshot_from_app,
)


def test_snapshot_from_app_extracts_fields() -> None:
    app = SimpleNamespace(
        paper_balance="25000",
        risk_profile_name="balanced",
        risk_manager_settings=RiskManagerSettings(
            max_risk_per_trade=0.04,
            max_daily_loss_pct=0.12,
            max_portfolio_risk=0.2,
            max_positions=6,
            emergency_stop_drawdown=0.18,
        ),
    )

    snapshot = snapshot_from_app(app)
    assert isinstance(snapshot, RiskSnapshot)
    assert snapshot.paper_balance == pytest.approx(25_000.0)
    assert snapshot.profile_name == "balanced"
    assert snapshot.settings is app.risk_manager_settings


def test_compute_default_notional_caps_to_runtime_limit() -> None:
    settings = RiskManagerSettings(
        max_risk_per_trade=0.05,
        max_daily_loss_pct=0.15,
        max_portfolio_risk=0.3,
        max_positions=4,
        emergency_stop_drawdown=0.2,
    )
    snapshot = RiskSnapshot(paper_balance=12_000.0, settings=settings, profile_name="aggressive")

    assert compute_default_notional(snapshot, default_notional=1_200.0) == pytest.approx(600.0)


def test_compute_default_notional_caps_to_portfolio_limit() -> None:
    settings = RiskManagerSettings(
        max_risk_per_trade=0.12,
        max_daily_loss_pct=0.15,
        max_portfolio_risk=0.04,
        max_positions=4,
        emergency_stop_drawdown=0.2,
    )
    snapshot = RiskSnapshot(paper_balance=10_000.0, settings=settings, profile_name="balanced")

    assert compute_default_notional(snapshot, default_notional=2_500.0) == pytest.approx(400.0)


def test_compute_default_notional_respects_smaller_fraction() -> None:
    settings = RiskManagerSettings(
        max_risk_per_trade=0.05,
        max_daily_loss_pct=0.12,
        max_portfolio_risk=0.2,
        max_positions=3,
        emergency_stop_drawdown=0.15,
    )
    snapshot = RiskSnapshot(paper_balance=8_000.0, settings=settings, profile_name="custom")

    assert compute_default_notional(snapshot, default_notional=320.0) == pytest.approx(320.0)


def test_compute_default_notional_uses_risk_when_fallback_zero() -> None:
    settings = RiskManagerSettings(
        max_risk_per_trade=0.02,
        max_daily_loss_pct=0.1,
        max_portfolio_risk=0.15,
        max_positions=5,
        emergency_stop_drawdown=0.2,
    )
    snapshot = RiskSnapshot(paper_balance=5_000.0, settings=settings, profile_name="balanced")

    assert compute_default_notional(snapshot, default_notional=0.0) == pytest.approx(100.0)


def test_compute_default_notional_fallback_when_missing() -> None:
    snapshot = RiskSnapshot(paper_balance=0.0, settings=None, profile_name=None)
    assert compute_default_notional(snapshot, default_notional=25.0) == pytest.approx(25.0)


def test_build_risk_profile_hint_formats_percentage() -> None:
    snapshot = RiskSnapshot(
        paper_balance=0.0,
        profile_name="balanced",
        settings=RiskManagerSettings(
            max_risk_per_trade=0.025,
            max_daily_loss_pct=0.10,
            max_portfolio_risk=0.15,
            max_positions=5,
            emergency_stop_drawdown=0.18,
        ),
    )

    hint = build_risk_profile_hint(snapshot)
    assert hint == "Profil ryzyka: balanced (≈2.50% / trade)"


def test_build_risk_limits_summary_lists_constraints() -> None:
    snapshot = RiskSnapshot(
        paper_balance=0.0,
        profile_name="balanced",
        settings=RiskManagerSettings(
            max_risk_per_trade=0.025,
            max_daily_loss_pct=0.10,
            max_portfolio_risk=0.25,
            max_positions=6,
            emergency_stop_drawdown=0.18,
        ),
    )

    summary = build_risk_limits_summary(snapshot)
    assert summary == "Limity ryzyka: Ekspozycja: 25.0% | Dzienna strata: 10.0% | Stop awaryjny: 18.0% | Pozycje: 6"


def test_format_helpers_strip_trailing_zeroes() -> None:
    assert format_notional(123.400) == "123.4"
    assert format_decimal(25000.0, decimals=2, fallback="0") == "25000"
