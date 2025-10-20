from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

import KryptoLowca.run_trading_gui_paper_emitter as emitter
import KryptoLowca.ui.trading.risk_helpers as risk_helpers
from bot_core.runtime.metadata import RiskManagerSettings


class _DummyRoot:
    def __init__(self) -> None:
        self._title = "KryptoŁowca — Paper (Event Emitter)"

    def title(self, new_title: str | None = None) -> str:
        if new_title is None:
            return self._title
        self._title = new_title
        return self._title


def _make_gui_stub(balance: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        root=_DummyRoot(),
        paper_balance=balance,
        risk_manager_settings=RiskManagerSettings(
            max_risk_per_trade=0.01,
            max_daily_loss_pct=0.1,
            max_portfolio_risk=0.1,
            max_positions=5,
            emergency_stop_drawdown=0.15,
        ),
        risk_profile_name=None,
        risk_profile_config=None,
    )


def test_configure_runtime_risk_updates_gui(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    settings = RiskManagerSettings(
        max_risk_per_trade=0.02,
        max_daily_loss_pct=0.1,
        max_portfolio_risk=0.25,
        max_positions=8,
        emergency_stop_drawdown=0.2,
    )

    captured_entrypoint: dict[str, str | None] = {}

    def fake_loader(entrypoint: str, **_: object) -> tuple[str, object, RiskManagerSettings]:
        captured_entrypoint["value"] = entrypoint
        return "balanced", {"max_position_pct": 0.02}, settings

    monkeypatch.setattr(risk_helpers, "load_risk_manager_settings", fake_loader)
    gui = _make_gui_stub(balance=20_000.0)

    caplog.set_level(logging.INFO, emitter.logger.name)

    emitter._configure_runtime_risk(gui)

    assert captured_entrypoint["value"] == "trading_gui"
    assert gui.risk_profile_name == "balanced"
    assert gui.risk_profile_config == {"max_position_pct": 0.02}
    assert gui.risk_manager_settings is settings
    assert gui.default_paper_notional >= 399.99
    assert "Profil ryzyka: balanced" in caplog.text
    assert "Domyślny notional (paper):" in caplog.text
    assert "Profil ryzyka" in gui.root.title()


def test_configure_runtime_risk_handles_loader_failure(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def boom(*_: object, **__: object) -> tuple[None, None, RiskManagerSettings]:
        raise RuntimeError("boom")

    monkeypatch.setattr(risk_helpers, "load_risk_manager_settings", boom)
    gui = _make_gui_stub(balance=0.0)

    caplog.set_level(logging.INFO, emitter.logger.name)

    emitter._configure_runtime_risk(gui)

    assert gui.default_paper_notional == pytest.approx(emitter.DEFAULT_PAPER_ORDER_NOTIONAL)
    assert "Domyślny notional (paper):" in caplog.text
