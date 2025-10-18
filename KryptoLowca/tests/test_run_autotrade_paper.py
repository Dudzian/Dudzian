from __future__ import annotations

import types
from pathlib import Path

import pytest

import KryptoLowca.run_autotrade_paper as launcher
from KryptoLowca.auto_trader import paper as paper_launcher
from KryptoLowca.auto_trader.paper import PaperAutoTradeOptions, parse_cli_args
from bot_core.runtime.metadata import RiskManagerSettings


def test_headless_stub_executes_trades() -> None:
    settings = RiskManagerSettings(
        max_risk_per_trade=0.1,
        max_daily_loss_pct=0.15,
        max_portfolio_risk=0.25,
        max_positions=5,
        emergency_stop_drawdown=0.2,
    )
    stub = launcher.HeadlessTradingStub(
        symbol="ETH/USDT",
        paper_balance=1_000.0,
        risk_manager_settings=settings,
    )
    stub._bridge_execute_trade("ETH/USDT", "buy", 2000.0)
    assert "ETH/USDT" in stub._open_positions
    position = stub._open_positions["ETH/USDT"]
    expected_qty = (stub.paper_balance * settings.max_risk_per_trade) / 2000.0
    assert position["qty"] == pytest.approx(expected_qty)

    stub._bridge_execute_trade("ETH/USDT", "sell", 2100.0)
    assert "ETH/USDT" not in stub._open_positions
    assert stub.paper_balance > 1_000.0


def test_paper_app_uses_headless_mode_when_gui_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, object] = {}
    dummy_settings = RiskManagerSettings(
        max_risk_per_trade=0.07,
        max_daily_loss_pct=0.18,
        max_portfolio_risk=0.22,
        max_positions=8,
        emergency_stop_drawdown=0.25,
    )

    class FakeTrader:
        def __init__(self, emitter, gui, symbol_getter, **kwargs):  # type: ignore[no-untyped-def]
            created["gui"] = gui
            created["symbol"] = symbol_getter()
            created["kwargs"] = kwargs
            self.stopped = False

        def start(self) -> None:  # pragma: no cover - prosty marker
            created["started"] = True

        def stop(self) -> None:  # pragma: no cover - prosty marker
            self.stopped = True
            created["stopped"] = True

    monkeypatch.setattr(paper_launcher, "AutoTrader", FakeTrader)
    monkeypatch.setattr(
        paper_launcher,
        "DummyMarketFeed",
        lambda *a, **kw: types.SimpleNamespace(start=lambda: None, stop=lambda: None, join=lambda timeout=None: None),
    )
    monkeypatch.setattr(
        paper_launcher,
        "load_risk_manager_settings",
        lambda *a, **kw: ("balanced", {"max_position_pct": 0.07}, dummy_settings),
    )
    monkeypatch.setattr(paper_launcher, "resolve_core_config_path", lambda: Path("/fake/core.yaml"))

    app = launcher.PaperAutoTradeApp(symbol="SOL/USDT", enable_gui=False, use_dummy_feed=False)
    assert isinstance(app.gui, launcher.HeadlessTradingStub)
    assert created["symbol"] == "SOL/USDT"
    assert app.risk_profile_name == "balanced"
    assert app.gui.risk_profile_name == "balanced"
    assert app.gui.risk_manager_settings.max_risk_per_trade == dummy_settings.max_risk_per_trade

    app.start()
    assert created.get("started") is True

    app.stop()
    assert created.get("stopped") is True
    app.stop()  # idempotencja


def test_parse_cli_args_supports_flags() -> None:
    options = parse_cli_args(["--nogui", "--no-feed", "--symbol=ETH/BTC", "--paper-balance", "2500"])
    assert options == PaperAutoTradeOptions(
        enable_gui=False,
        use_dummy_feed=False,
        symbol="ETH/BTC",
        paper_balance=2500.0,
        core_config_path=None,
        risk_profile=None,
    )


def test_parse_cli_args_handles_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        parse_cli_args(["--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "Usage:" in captured.out


def test_parse_cli_args_supports_risk_and_config() -> None:
    options = parse_cli_args(["--core-config", "/tmp/core.yaml", "--risk-profile=aggressive"])
    assert options.core_config_path == "/tmp/core.yaml"
    assert options.risk_profile == "aggressive"


def test_paper_app_reload_risk_profile_updates_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    initial_settings = RiskManagerSettings(
        max_risk_per_trade=0.05,
        max_daily_loss_pct=0.2,
        max_portfolio_risk=0.25,
        max_positions=5,
        emergency_stop_drawdown=0.3,
    )
    updated_settings = RiskManagerSettings(
        max_risk_per_trade=0.08,
        max_daily_loss_pct=0.25,
        max_portfolio_risk=0.4,
        max_positions=7,
        emergency_stop_drawdown=0.35,
    )

    responses = [
        ("balanced", {"max_position_pct": 0.05}, initial_settings),
        ("growth", {"max_position_pct": 0.08}, updated_settings),
    ]

    def fake_loader(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(paper_launcher, "load_risk_manager_settings", fake_loader)
    monkeypatch.setattr(paper_launcher, "resolve_core_config_path", lambda: Path("/fake/core.yaml"))

    app = launcher.PaperAutoTradeApp(symbol="BTC/USDT", enable_gui=False)

    assert app.risk_profile_name == "balanced"
    assert app.gui.risk_manager_settings.max_risk_per_trade == pytest.approx(0.05)

    app.reload_risk_profile("growth")

    assert app.risk_profile_name == "growth"
    assert app.gui.risk_manager_settings.max_risk_per_trade == pytest.approx(0.08)
    assert app.risk_manager_settings.max_portfolio_risk == pytest.approx(0.4)


def test_paper_app_propagates_risk_updates_to_trader(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_settings = RiskManagerSettings(
        max_risk_per_trade=0.05,
        max_daily_loss_pct=0.2,
        max_portfolio_risk=0.25,
        max_positions=5,
        emergency_stop_drawdown=0.3,
    )
    updated_settings = RiskManagerSettings(
        max_risk_per_trade=0.1,
        max_daily_loss_pct=0.25,
        max_portfolio_risk=0.4,
        max_positions=8,
        emergency_stop_drawdown=0.35,
    )

    responses = [
        ("balanced", {"max_position_pct": 0.05}, dummy_settings),
        ("growth", {"max_position_pct": 0.1}, updated_settings),
    ]

    class RecordingTrader:
        def __init__(self, *_: object, **__: object) -> None:
            self.updated: list[tuple[str | None, RiskManagerSettings, object | None]] = []

        def start(self) -> None:  # pragma: no cover - prosty stub
            return None

        def stop(self) -> None:  # pragma: no cover - prosty stub
            return None

        def update_risk_manager_settings(
            self,
            settings: RiskManagerSettings,
            *,
            profile_name: str | None = None,
            profile_config: object | None = None,
        ) -> None:
            self.updated.append((profile_name, settings, profile_config))

    monkeypatch.setattr(paper_launcher, "AutoTrader", RecordingTrader)
    monkeypatch.setattr(paper_launcher, "resolve_core_config_path", lambda: Path("/fake/core.yaml"))
    monkeypatch.setattr(paper_launcher, "load_risk_manager_settings", lambda *_, **__: responses.pop(0))

    app = launcher.PaperAutoTradeApp(symbol="BTC/USDT", enable_gui=False)
    trader = app.trader  # type: ignore[assignment]

    assert isinstance(trader, RecordingTrader)
    assert trader.updated == []

    app.reload_risk_profile("growth")

    assert trader.updated
    name, settings, payload = trader.updated[-1]
    assert name == "growth"
    assert settings.max_risk_per_trade == pytest.approx(0.1)
    assert payload == {"max_position_pct": 0.1}


def test_paper_app_check_risk_change_triggers_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paper_launcher, "resolve_core_config_path", lambda: Path("/fake/core.yaml"))
    monkeypatch.setattr(
        paper_launcher,
        "load_risk_manager_settings",
        lambda *_, **__: (
            "balanced",
            {"max_position_pct": 0.05},
            RiskManagerSettings(
                max_risk_per_trade=0.05,
                max_daily_loss_pct=0.2,
                max_portfolio_risk=0.25,
                max_positions=5,
                emergency_stop_drawdown=0.3,
            ),
        ),
    )

    app = launcher.PaperAutoTradeApp(symbol="BTC/USDT", enable_gui=False)
    app.core_config_path = Path("/tmp/core.yaml")
    app._risk_config_mtime = 1.0

    calls: list[str | None] = []

    monkeypatch.setattr(app, "_get_risk_config_mtime", lambda: 2.0)

    def fake_reload(profile: str | None = None):
        calls.append(profile)
        return app.risk_manager_settings

    app.reload_risk_profile = fake_reload  # type: ignore[assignment]

    changed = app._check_risk_config_change()

    assert changed is True
    assert calls == [None]
