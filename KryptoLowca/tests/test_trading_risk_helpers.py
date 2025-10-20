import logging
from types import SimpleNamespace

import pytest

from bot_core.runtime.metadata import RiskManagerSettings

from KryptoLowca.ui.trading.risk_helpers import (
    apply_runtime_risk_context,
    refresh_runtime_risk_context,
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


def test_apply_runtime_risk_context_updates_gui(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = RiskManagerSettings(
        max_risk_per_trade=0.05,
        max_daily_loss_pct=0.12,
        max_portfolio_risk=0.3,
        max_positions=4,
        emergency_stop_drawdown=0.18,
    )

    captured: dict[str, object] = {}

    def fake_loader(entrypoint: str, **kwargs: object) -> tuple[str, object, RiskManagerSettings]:
        captured["entrypoint"] = entrypoint
        captured["kwargs"] = kwargs
        return "balanced", {"max_position_pct": 0.05}, settings

    monkeypatch.setattr(
        "KryptoLowca.ui.trading.risk_helpers.load_risk_manager_settings",
        fake_loader,
    )

    class DummyRoot:
        def __init__(self) -> None:
            self._title = "Trading GUI"
            self.history: list[str] = []

        def title(self, new_value: str | None = None) -> str:
            if new_value is None:
                return self._title
            self._title = new_value
            self.history.append(new_value)
            return self._title

    gui = SimpleNamespace(
        root=DummyRoot(),
        paper_balance=15_000.0,
        risk_profile_name=None,
        risk_profile_config=None,
        risk_manager_settings=None,
        _update_risk_banner=lambda: None,
    )

    snapshot = apply_runtime_risk_context(
        gui,
        entrypoint="trading_gui",
        config_path="/tmp/core.yaml",
        default_notional=100.0,
        logger=logging.getLogger("test"),
    )

    assert captured["entrypoint"] == "trading_gui"
    assert gui.risk_profile_name == "balanced"
    assert gui.risk_profile_config == {"max_position_pct": 0.05}
    assert gui.risk_manager_settings is settings
    assert gui.default_paper_notional == pytest.approx(750.0)
    assert snapshot.settings is settings
    assert gui.root.history and "Profil ryzyka" in gui.root.history[-1]


def test_refresh_runtime_risk_context_updates_notional() -> None:
    settings = RiskManagerSettings(
        max_risk_per_trade=0.05,
        max_daily_loss_pct=0.1,
        max_portfolio_risk=0.3,
        max_positions=4,
        emergency_stop_drawdown=0.2,
    )

    class DummyRoot:
        def __init__(self) -> None:
            self._title = "Trading GUI"
            self.titles: list[str] = []

        def title(self, new_value: str | None = None) -> str:
            if new_value is None:
                return self._title
            self._title = new_value
            self.titles.append(new_value)
            return self._title

    banner_calls: list[None] = []

    gui = SimpleNamespace(
        root=DummyRoot(),
        paper_balance=15_000.0,
        risk_profile_name="balanced",
        risk_profile_config={"source": "runtime"},
        risk_manager_settings=settings,
        _update_risk_banner=lambda: banner_calls.append(None),
    )

    snapshot = refresh_runtime_risk_context(
        gui,
        default_notional=100.0,
        logger=logging.getLogger("test"),
    )

    assert snapshot.settings is settings
    assert gui.default_paper_notional == pytest.approx(750.0)
    assert banner_calls  # baner został odświeżony
    assert gui.root.titles and "Profil ryzyka" in gui.root.titles[-1]
