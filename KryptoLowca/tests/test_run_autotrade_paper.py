from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from KryptoLowca.paper_auto_trade_app import HeadlessTradingStub, PaperAutoTradeApp


def _write_core_config(path: Path, *, max_daily_loss: float) -> None:
    path.write_text(
        dedent(
            f"""
            risk_profiles:
              balanced:
                max_daily_loss_pct: {max_daily_loss}
                max_position_pct: 0.05
                target_volatility: 0.1
                max_leverage: 3.0
                stop_loss_atr_multiple: 1.5
                max_open_positions: 5
                hard_drawdown_pct: 0.08
            environments:
              paper_env:
                name: paper_env
                exchange: binance_spot
                environment: paper
                keychain_key: dummy
                data_cache_path: ./cache
                risk_profile: balanced
                alert_channels: []
                required_permissions: []
                forbidden_permissions: []
            """
        ).strip()
    )


def test_reload_updates_headless_stub(tmp_path: Path) -> None:
    core_path = tmp_path / "core.yaml"
    _write_core_config(core_path, max_daily_loss=0.02)

    stub = HeadlessTradingStub()
    app = PaperAutoTradeApp(gui=None, headless_stub=stub, core_config_path=core_path, core_environment="paper_env")

    profile, settings, _ = app.reload_risk_settings()
    assert profile == "balanced"
    assert settings["max_daily_loss_pct"] == pytest.approx(0.02)
    assert stub.last_risk_settings["max_daily_loss_pct"] == pytest.approx(0.02)
    first_update = stub.update_count

    _write_core_config(core_path, max_daily_loss=0.03)
    assert app.handle_cli_command("reload-risk")
    assert stub.last_risk_settings["max_daily_loss_pct"] == pytest.approx(0.03)
    assert stub.update_count == first_update + 1


def test_app_uses_gui_reload() -> None:
    responses = [
        ("balanced", {"max_daily_loss_pct": 0.02, "max_portfolio_risk": 0.08}, object()),
    ]

    class _DummyGUI:
        def __init__(self) -> None:
            self.calls: list[str | None] = []

        def reload_risk_manager_settings(self, *, environment: str | None = None):
            self.calls.append(environment)
            return responses.pop(0)

    gui = _DummyGUI()
    stub = HeadlessTradingStub()
    app = PaperAutoTradeApp(gui=gui, headless_stub=stub, core_environment="paper_env")

    profile, settings, _ = app.reload_risk_settings()
    assert profile == "balanced"
    assert settings["max_daily_loss_pct"] == pytest.approx(0.02)
    assert gui.calls == ["paper_env"]
    assert stub.last_risk_settings["max_daily_loss_pct"] == pytest.approx(0.02)
