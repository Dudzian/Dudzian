from __future__ import annotations

import types
from pathlib import Path
from datetime import datetime

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


def test_start_auto_reload_accepts_positional_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[float] = []

    def _fake_start(self: paper_launcher.PaperAutoTradeApp) -> None:
        recorded.append(self._risk_watch_interval)

    monkeypatch.setattr(paper_launcher.PaperAutoTradeApp, "_start_risk_watcher", _fake_start)

    app = launcher.PaperAutoTradeApp(symbol="BTC/USDT", enable_gui=False)
    app.start_auto_reload(0.25)
    assert recorded == [pytest.approx(0.25)]


def test_start_auto_reload_rejects_non_positive_interval() -> None:
    app = launcher.PaperAutoTradeApp(symbol="BTC/USDT", enable_gui=False)
    with pytest.raises(ValueError):
        app.start_auto_reload(0.0)


def test_stop_auto_reload_exposes_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, float | None] = {}

    def _fake_stop(self: paper_launcher.PaperAutoTradeApp, *, timeout: float | None = None) -> None:
        captured["timeout"] = timeout

    monkeypatch.setattr(paper_launcher.PaperAutoTradeApp, "_stop_risk_watcher", _fake_stop)

    app = launcher.PaperAutoTradeApp(symbol="BTC/USDT", enable_gui=False)
    app.stop_auto_reload(timeout=3.5)
    assert captured["timeout"] == pytest.approx(3.5)


def test_cli_command_supports_positional_tokens(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base = tmp_path
    primary = (base / "core.yaml").resolve()
    alternate = (base / "alt-core.yaml").resolve()
    alternate.touch()

    primary_settings = RiskManagerSettings(
        max_risk_per_trade=0.05,
        max_daily_loss_pct=0.02,
        max_portfolio_risk=0.1,
        max_positions=5,
        emergency_stop_drawdown=0.1,
    )
    alternate_settings = RiskManagerSettings(
        max_risk_per_trade=0.07,
        max_daily_loss_pct=0.03,
        max_portfolio_risk=0.12,
        max_positions=6,
        emergency_stop_drawdown=0.12,
    )

    def _fake_loader(*args, **kwargs):
        cfg_path = Path(kwargs.get("config_path", primary)).resolve()
        if cfg_path == primary:
            return "balanced", {"source": "primary"}, primary_settings
        if cfg_path == alternate:
            return "growth", {"source": "alternate"}, alternate_settings
        raise AssertionError(f"unexpected config path: {cfg_path}")

    monkeypatch.setattr(paper_launcher, "load_risk_manager_settings", _fake_loader)

    app = launcher.PaperAutoTradeApp(symbol="BTC/USDT", enable_gui=False, core_config_path=primary)
    assert app.risk_profile_name == "balanced"
    assert app.headless_stub is not None
    assert app.headless_stub.risk_manager_settings.max_daily_loss_pct == pytest.approx(0.02)

    assert app.handle_cli_command(f"reload-risk paper_env {alternate}")
    assert app.core_environment == "paper_env"
    assert Path(app.core_config_path).resolve() == alternate
    assert app.risk_profile_name == "growth"
    assert app.headless_stub.risk_manager_settings.max_daily_loss_pct == pytest.approx(0.03)


def test_paper_app_prefers_bootstrap_risk_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    bootstrap_settings = RiskManagerSettings(
        max_risk_per_trade=0.03,
        max_daily_loss_pct=0.12,
        max_portfolio_risk=0.18,
        max_positions=7,
        emergency_stop_drawdown=0.22,
    )
    bootstrap_ctx = types.SimpleNamespace(
        risk_profile_name="paper-profile",
        risk_profile_config={"source": "bootstrap"},
        risk_manager_settings=bootstrap_settings,
    )

    def _failing_loader(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("load_risk_manager_settings should not be called")

    monkeypatch.setattr(paper_launcher, "load_risk_manager_settings", _failing_loader)

    app = launcher.PaperAutoTradeApp(
        symbol="BTC/USDT",
        enable_gui=False,
        bootstrap_context=bootstrap_ctx,
    )

    assert app.risk_profile_name == "paper-profile"
    assert app.risk_profile_config == {"source": "bootstrap"}
    assert app.risk_manager_settings is bootstrap_settings


def test_paper_app_reload_uses_loader_for_different_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap_settings = RiskManagerSettings(
        max_risk_per_trade=0.02,
        max_daily_loss_pct=0.1,
        max_portfolio_risk=0.15,
        max_positions=6,
        emergency_stop_drawdown=0.2,
    )
    bootstrap_ctx = types.SimpleNamespace(
        risk_profile_name="paper-profile",
        risk_profile_config={"source": "bootstrap"},
        risk_manager_settings=bootstrap_settings,
    )

    loader_calls: list[str | None] = []
    new_settings = RiskManagerSettings(
        max_risk_per_trade=0.05,
        max_daily_loss_pct=0.25,
        max_portfolio_risk=0.3,
        max_positions=4,
        emergency_stop_drawdown=0.35,
    )

    def _loader(
        entrypoint: str,
        *,
        profile_name: str | None = None,
        config_path=None,
        logger=None,
    ) -> tuple[str | None, object | None, RiskManagerSettings]:
        loader_calls.append(profile_name)
        return "aggressive", {"source": "file"}, new_settings

    monkeypatch.setattr(paper_launcher, "load_risk_manager_settings", _loader)

    app = launcher.PaperAutoTradeApp(
        symbol="BTC/USDT",
        enable_gui=False,
        bootstrap_context=bootstrap_ctx,
    )

    result = app.reload_risk_profile("aggressive")

    assert loader_calls == ["aggressive"]
    assert result is new_settings
    assert app.risk_profile_name == "aggressive"
    assert app.risk_profile_config == {"source": "file"}
    assert app.bootstrap_context.risk_profile_name == "aggressive"
    assert app.bootstrap_context.risk_manager_settings is new_settings


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


def test_paper_app_reload_risk_settings_updates_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dummy_settings = RiskManagerSettings(
        max_risk_per_trade=0.05,
        max_daily_loss_pct=0.2,
        max_portfolio_risk=0.25,
        max_positions=5,
        emergency_stop_drawdown=0.3,
    )

    monkeypatch.setattr(paper_launcher, "resolve_core_config_path", lambda: tmp_path / "base.yaml")
    monkeypatch.setattr(
        paper_launcher,
        "load_risk_manager_settings",
        lambda *_, **__: ("balanced", {"max_position_pct": 0.05}, dummy_settings),
    )

    app = launcher.PaperAutoTradeApp(symbol="BTC/USDT", enable_gui=False)

    new_core = tmp_path / "alt.yaml"
    profile, settings, payload = app.reload_risk_settings(
        config_path=new_core,
        environment="paper_env",
    )

    assert Path(app.core_config_path) == new_core
    assert app.core_environment == "paper_env"
    assert profile == app.risk_profile_name
    assert settings is app.risk_manager_settings
    assert payload is app.risk_profile_config


def test_paper_app_listeners_receive_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_settings = RiskManagerSettings(
        max_risk_per_trade=0.05,
        max_daily_loss_pct=0.2,
        max_portfolio_risk=0.25,
        max_positions=5,
        emergency_stop_drawdown=0.3,
    )

    monkeypatch.setattr(paper_launcher, "resolve_core_config_path", lambda: Path("/fake/core.yaml"))
    monkeypatch.setattr(
        paper_launcher,
        "load_risk_manager_settings",
        lambda *_, **__: ("balanced", {"max_position_pct": 0.05}, dummy_settings),
    )

    app = launcher.PaperAutoTradeApp(symbol="BTC/USDT", enable_gui=False)
    received: list[tuple[RiskManagerSettings, str | None, object | None]] = []

    def listener(settings: RiskManagerSettings, profile: str | None, payload: object | None) -> None:
        received.append((settings, profile, payload))

    app.add_listener(listener)
    assert received, "Listener powinien zostać powiadomiony o stanie początkowym"
    initial_settings, initial_profile, initial_payload = received[-1]
    assert initial_settings is app.risk_manager_settings
    assert initial_profile == app.risk_profile_name
    assert initial_payload is app.risk_profile_config

    received.clear()
    app.reload_risk_profile("balanced")

    assert received
    captured_settings, captured_profile, captured_payload = received[-1]
    assert captured_settings is app.risk_manager_settings
    assert captured_profile == app.risk_profile_name
    assert captured_payload is app.risk_profile_config


def test_paper_app_handle_cli_command_invokes_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_settings = RiskManagerSettings(
        max_risk_per_trade=0.05,
        max_daily_loss_pct=0.2,
        max_portfolio_risk=0.25,
        max_positions=5,
        emergency_stop_drawdown=0.3,
    )

    monkeypatch.setattr(paper_launcher, "resolve_core_config_path", lambda: Path("/fake/core.yaml"))
    monkeypatch.setattr(
        paper_launcher,
        "load_risk_manager_settings",
        lambda *_, **__: ("balanced", {"max_position_pct": 0.05}, dummy_settings),
    )

    app = launcher.PaperAutoTradeApp(symbol="BTC/USDT", enable_gui=False)
    calls: dict[str, object] = {}

    def fake_reload(*, config_path=None, environment=None):
        calls["config_path"] = config_path
        calls["environment"] = environment
        return app.risk_profile_name, app.risk_manager_settings, app.risk_profile_config

    monkeypatch.setattr(app, "reload_risk_settings", fake_reload)

    assert app.handle_cli_command("reload-risk --config /tmp/core.yaml --env paper_env") is True
    assert calls["config_path"] == "/tmp/core.yaml"
    assert calls["environment"] == "paper_env"


def test_paper_app_auto_reload_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_settings = RiskManagerSettings(
        max_risk_per_trade=0.05,
        max_daily_loss_pct=0.2,
        max_portfolio_risk=0.25,
        max_positions=5,
        emergency_stop_drawdown=0.3,
    )

    monkeypatch.setattr(paper_launcher, "resolve_core_config_path", lambda: Path("/fake/core.yaml"))
    monkeypatch.setattr(
        paper_launcher,
        "load_risk_manager_settings",
        lambda *_, **__: ("balanced", {"max_position_pct": 0.05}, dummy_settings),
    )

    app = launcher.PaperAutoTradeApp(symbol="BTC/USDT", enable_gui=False)
    events: list[str] = []

    monkeypatch.setattr(app, "_start_risk_watcher", lambda: events.append("start"))
    monkeypatch.setattr(app, "_stop_risk_watcher", lambda: events.append("stop"))

    app.start_auto_reload(interval=0.25)
    assert "start" in events
    assert app._risk_watch_interval == pytest.approx(0.25)

    app.stop_auto_reload()
    assert "stop" in events


def test_paper_app_prefers_bootstrap_execution_service(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_settings = RiskManagerSettings(
        max_risk_per_trade=0.05,
        max_daily_loss_pct=0.2,
        max_portfolio_risk=0.25,
        max_positions=5,
        emergency_stop_drawdown=0.3,
    )

    class _DummyService:
        def execute(self, *_: object, **__: object) -> object:  # pragma: no cover - never called
            return object()

        def cancel(self, *_: object, **__: object) -> None:  # pragma: no cover - never called
            return None

        def flush(self) -> None:  # pragma: no cover - never called
            return None

    bootstrap_calls: dict[str, object] = {}

    class RecordingTrader:
        def __init__(self, *_: object, **kwargs: object) -> None:
            bootstrap_calls.update(kwargs)

        def start(self) -> None:  # pragma: no cover - prosty stub
            return None

        def stop(self) -> None:  # pragma: no cover - prosty stub
            return None

    service = _DummyService()
    bootstrap = types.SimpleNamespace(execution_service=service, environment=types.SimpleNamespace(name="paper"))

    monkeypatch.setattr(paper_launcher, "AutoTrader", RecordingTrader)
    monkeypatch.setattr(paper_launcher, "resolve_core_config_path", lambda: Path("/fake/core.yaml"))
    monkeypatch.setattr(
        paper_launcher,
        "load_risk_manager_settings",
        lambda *_, **__: ("balanced", {"max_position_pct": 0.05}, dummy_settings),
    )

    app = paper_launcher.PaperAutoTradeApp(
        symbol="BTC/USDT",
        enable_gui=False,
        bootstrap_context=bootstrap,
    )

    assert bootstrap_calls["bootstrap_context"] is bootstrap
    assert bootstrap_calls["execution_service"] is service
    assert app.bootstrap_context is bootstrap


def test_paper_app_explicit_execution_service_overrides_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_settings = RiskManagerSettings(
        max_risk_per_trade=0.05,
        max_daily_loss_pct=0.2,
        max_portfolio_risk=0.25,
        max_positions=5,
        emergency_stop_drawdown=0.3,
    )

    class _DummyService:
        def execute(self, *_: object, **__: object) -> object:  # pragma: no cover - never called
            return object()

        def cancel(self, *_: object, **__: object) -> None:  # pragma: no cover - never called
            return None

        def flush(self) -> None:  # pragma: no cover - never called
            return None

    captured: dict[str, object] = {}

    class RecordingTrader:
        def __init__(self, *_: object, **kwargs: object) -> None:
            captured.update(kwargs)

        def start(self) -> None:  # pragma: no cover - prosty stub
            return None

        def stop(self) -> None:  # pragma: no cover - prosty stub
            return None

    bootstrap = types.SimpleNamespace(execution_service=_DummyService())
    explicit = _DummyService()

    monkeypatch.setattr(paper_launcher, "AutoTrader", RecordingTrader)
    monkeypatch.setattr(paper_launcher, "resolve_core_config_path", lambda: Path("/fake/core.yaml"))
    monkeypatch.setattr(
        paper_launcher,
        "load_risk_manager_settings",
        lambda *_, **__: ("balanced", {"max_position_pct": 0.05}, dummy_settings),
    )

    paper_launcher.PaperAutoTradeApp(
        symbol="BTC/USDT",
        enable_gui=False,
        bootstrap_context=bootstrap,
        execution_service=explicit,
    )

    assert captured["bootstrap_context"] is bootstrap
    assert captured["execution_service"] is explicit


def test_paper_app_records_reload_stats_and_deduplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    initial_settings = RiskManagerSettings(
        max_risk_per_trade=0.05,
        max_daily_loss_pct=0.2,
        max_portfolio_risk=0.25,
        max_positions=5,
        emergency_stop_drawdown=0.3,
    )
    duplicate_settings = RiskManagerSettings(
        max_risk_per_trade=0.05,
        max_daily_loss_pct=0.2,
        max_portfolio_risk=0.25,
        max_positions=5,
        emergency_stop_drawdown=0.3,
    )

    responses = [
        ("balanced", {"max_position_pct": 0.05}, initial_settings),
        ("balanced", {"max_position_pct": 0.05}, initial_settings),
        ("balanced", {"max_position_pct": 0.05}, duplicate_settings),
    ]

    monkeypatch.setattr(paper_launcher, "resolve_core_config_path", lambda: Path("/fake/core.yaml"))
    monkeypatch.setattr(paper_launcher, "load_risk_manager_settings", lambda *_, **__: responses.pop(0))

    app = launcher.PaperAutoTradeApp(symbol="BTC/USDT", enable_gui=False)
    notifications: list[str] = []
    app._notify_trader_of_risk_update = lambda *_: notifications.append("trader")  # type: ignore[assignment]
    app._notify_listeners = lambda *_: notifications.append("listener")  # type: ignore[assignment]

    app.reload_risk_profile("balanced")
    app.reload_risk_profile("balanced")

    assert app.headless_stub is app.gui
    assert app.reload_count == 2
    assert isinstance(app.last_reload_at, datetime)
    assert notifications == ["trader", "listener"]


def test_paper_app_infers_environment_from_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_settings = RiskManagerSettings(
        max_risk_per_trade=0.05,
        max_daily_loss_pct=0.2,
        max_portfolio_risk=0.25,
        max_positions=5,
        emergency_stop_drawdown=0.3,
    )

    bootstrap = types.SimpleNamespace(
        environment=types.SimpleNamespace(name="paper-env"),
        execution_service=None,
    )

    monkeypatch.setattr(paper_launcher, "resolve_core_config_path", lambda: Path("/fake/core.yaml"))
    monkeypatch.setattr(
        paper_launcher,
        "load_risk_manager_settings",
        lambda *_, **__: ("balanced", {"max_position_pct": 0.05}, dummy_settings),
    )

    app = paper_launcher.PaperAutoTradeApp(
        symbol="BTC/USDT",
        enable_gui=False,
        bootstrap_context=bootstrap,
    )

    assert app.core_environment == "paper-env"
    assert app._watch_interval == app._risk_watch_interval
    assert app._watch_last_mtime == app._risk_config_mtime


def test_paper_app_updates_bootstrap_metadata_on_start(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_settings = RiskManagerSettings(
        max_risk_per_trade=0.05,
        max_daily_loss_pct=0.2,
        max_portfolio_risk=0.25,
        max_positions=5,
        emergency_stop_drawdown=0.3,
    )

    bootstrap = types.SimpleNamespace(
        risk_profile_name=None,
        risk_profile_config=None,
        risk_manager_settings=None,
    )

    monkeypatch.setattr(paper_launcher, "resolve_core_config_path", lambda: Path("/fake/core.yaml"))
    monkeypatch.setattr(
        paper_launcher,
        "load_risk_manager_settings",
        lambda *_, **__: ("balanced", {"max_position_pct": 0.05}, dummy_settings),
    )

    app = paper_launcher.PaperAutoTradeApp(
        symbol="BTC/USDT",
        enable_gui=False,
        bootstrap_context=bootstrap,
    )

    assert bootstrap.risk_profile_name == "balanced"
    assert bootstrap.risk_profile_config == {"max_position_pct": 0.05}
    assert bootstrap.risk_manager_settings is app.risk_manager_settings


def test_paper_app_accepts_custom_headless_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    settings_one = RiskManagerSettings(
        max_risk_per_trade=0.05,
        max_daily_loss_pct=0.2,
        max_portfolio_risk=0.25,
        max_positions=5,
        emergency_stop_drawdown=0.3,
    )
    settings_two = RiskManagerSettings(
        max_risk_per_trade=0.07,
        max_daily_loss_pct=0.22,
        max_portfolio_risk=0.3,
        max_positions=6,
        emergency_stop_drawdown=0.28,
    )

    responses = [
        ("balanced", {"source": "primary"}, settings_one),
        ("growth", {"source": "secondary"}, settings_two),
    ]

    class DummyTrader:
        def __init__(self, *_: object, **__: object) -> None:  # pragma: no cover - prosty stub
            return

        def start(self) -> None:  # pragma: no cover - prosty stub
            return

        def stop(self) -> None:  # pragma: no cover - prosty stub
            return

    class CustomStub:
        def __init__(self) -> None:
            self.symbol = "ADA/USDT"
            self.paper_balance = 5_000.0
            self.calls: list[tuple[str | None, RiskManagerSettings]] = []

        def apply_risk_profile(self, name: str | None, settings: RiskManagerSettings) -> RiskManagerSettings:
            self.calls.append((name, settings))
            self.risk_manager_settings = settings
            return settings

        def get_symbol(self) -> str:
            return self.symbol

    monkeypatch.setattr(paper_launcher, "AutoTrader", DummyTrader)
    monkeypatch.setattr(paper_launcher, "resolve_core_config_path", lambda: Path("/fake/core.yaml"))
    monkeypatch.setattr(paper_launcher, "load_risk_manager_settings", lambda *_, **__: responses.pop(0))

    stub = CustomStub()
    app = paper_launcher.PaperAutoTradeApp(
        symbol="ADA/USDT",
        enable_gui=False,
        headless_stub=stub,
    )

    assert app.gui is stub
    assert app.headless_stub is stub
    assert stub.calls[0][0] == "balanced"
    assert stub.calls[0][1].max_daily_loss_pct == pytest.approx(0.2)

    app.reload_risk_profile("growth")

    assert stub.calls[-1][0] == "growth"
    assert stub.calls[-1][1].max_daily_loss_pct == pytest.approx(0.22)


def test_paper_app_uses_external_gui_and_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    settings_one = RiskManagerSettings(
        max_risk_per_trade=0.04,
        max_daily_loss_pct=0.18,
        max_portfolio_risk=0.2,
        max_positions=4,
        emergency_stop_drawdown=0.25,
    )
    settings_two = RiskManagerSettings(
        max_risk_per_trade=0.06,
        max_daily_loss_pct=0.24,
        max_portfolio_risk=0.32,
        max_positions=7,
        emergency_stop_drawdown=0.3,
    )

    responses = [
        ("balanced", {"source": "primary"}, settings_one),
        ("growth", {"source": "secondary"}, settings_two),
    ]

    class DummyTrader:
        def __init__(self, *_: object, **__: object) -> None:  # pragma: no cover - prosty stub
            return

        def start(self) -> None:  # pragma: no cover - prosty stub
            return

        def stop(self) -> None:  # pragma: no cover - prosty stub
            return

    class CustomGUI:
        def __init__(self) -> None:
            self.paper_balance = 0.0
            self.symbol_var = types.SimpleNamespace(get=lambda: "DOGE/USDT")
            self.calls: list[str | None] = []
            self.current_settings = settings_one

        def reload_risk_profile(self, profile_name: str | None) -> RiskManagerSettings:
            self.calls.append(profile_name)
            return self.current_settings

    class CustomStub:
        def __init__(self) -> None:
            self.symbol = "DOGE/USDT"
            self.paper_balance = 1_000.0
            self.applied: list[tuple[str | None, RiskManagerSettings]] = []

        def apply_risk_profile(self, name: str | None, settings: RiskManagerSettings) -> RiskManagerSettings:
            self.applied.append((name, settings))
            self.risk_manager_settings = settings
            return settings

        def get_symbol(self) -> str:
            return self.symbol

    monkeypatch.setattr(paper_launcher, "AutoTrader", DummyTrader)
    monkeypatch.setattr(paper_launcher, "resolve_core_config_path", lambda: Path("/fake/core.yaml"))
    monkeypatch.setattr(paper_launcher, "load_risk_manager_settings", lambda *_, **__: responses.pop(0))

    gui = CustomGUI()
    stub = CustomStub()
    app = paper_launcher.PaperAutoTradeApp(
        symbol="DOGE/USDT",
        enable_gui=True,
        gui=gui,
        headless_stub=stub,
    )

    assert app.gui is gui
    assert app.headless_stub is stub
    assert gui.paper_balance == pytest.approx(10_000.0)
    assert app.symbol_getter() == "DOGE/USDT"
    assert stub.applied[0][0] == "balanced"
    assert stub.applied[0][1].max_daily_loss_pct == pytest.approx(0.18)

    gui.current_settings = settings_two
    app.reload_risk_profile("growth")

    assert gui.calls == ["growth"]
    assert stub.applied[-1][0] == "growth"
    assert stub.applied[-1][1].max_daily_loss_pct == pytest.approx(0.24)


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
    app._risk_config_mtime = 1

    calls: list[str | None] = []

    monkeypatch.setattr(app, "_get_risk_config_mtime", lambda: 2)

    def fake_reload(profile: str | None = None):
        calls.append(profile)
        app._risk_config_mtime = app._get_risk_config_mtime()
        return app.risk_manager_settings

    app.reload_risk_profile = fake_reload  # type: ignore[assignment]

    changed = app._check_risk_config_change()

    assert changed is True
    assert calls == [None]


def test_paper_app_risk_watch_retries_after_failure(monkeypatch: pytest.MonkeyPatch) -> None:
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
    app._risk_config_mtime = 1

    monkeypatch.setattr(app, "_get_risk_config_mtime", lambda: 2)

    calls: list[str] = []

    def _failing_reload(profile: str | None = None):
        calls.append("fail")
        raise RuntimeError("boom")

    app.reload_risk_profile = _failing_reload  # type: ignore[assignment]

    first = app._check_risk_config_change()

    assert first is False
    assert calls == ["fail"]
    assert app._risk_config_mtime == 1

    def _successful_reload(profile: str | None = None):
        calls.append("ok")
        app._risk_config_mtime = app._get_risk_config_mtime()
        return app.risk_manager_settings

    app.reload_risk_profile = _successful_reload  # type: ignore[assignment]

    second = app._check_risk_config_change()

    assert second is True
    assert calls == ["fail", "ok"]
    assert app._risk_config_mtime == 2


def test_paper_app_risk_watch_handles_unknown_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
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
    app._risk_config_mtime = None

    monkeypatch.setattr(app, "_get_risk_config_mtime", lambda: 5)

    calls: list[str | None] = []

    def _successful_reload(profile: str | None = None):
        calls.append(profile)
        app._risk_config_mtime = app._get_risk_config_mtime()
        return app.risk_manager_settings

    app.reload_risk_profile = _successful_reload  # type: ignore[assignment]

    changed = app._check_risk_config_change()

    assert changed is True
    assert calls == [None]
    assert app._risk_config_mtime == 5

from pathlib import Path
import time
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


def _write_multi_profile_config(path: Path) -> None:
    path.write_text(
        dedent(
            """
            risk_profiles:
              balanced:
                max_daily_loss_pct: 0.02
                max_position_pct: 0.05
                max_leverage: 3.0
                stop_loss_atr_multiple: 1.5
                max_open_positions: 5
                hard_drawdown_pct: 0.08
              aggressive:
                max_daily_loss_pct: 0.05
                max_position_pct: 0.1
                max_leverage: 4.0
                stop_loss_atr_multiple: 2.0
                max_open_positions: 7
                hard_drawdown_pct: 0.12
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
    assert app.handle_cli_command(f"reload-risk config={core_path} env=paper_env")
    assert stub.last_risk_settings["max_daily_loss_pct"] == pytest.approx(0.03)
    assert stub.update_count == first_update + 1


def test_app_uses_gui_reload() -> None:
    responses = [
        ("balanced", {"max_daily_loss_pct": 0.02, "max_portfolio_risk": 0.08}, object()),
    ]

    class _DummyGUI:
        def __init__(self) -> None:
            self.calls: list[tuple[object, str | None]] = []
            self.core_config_path: Path | None = None
            self.core_environment: str | None = None

        def reload_risk_manager_settings(self, *, config_path=None, environment: str | None = None):
            self.calls.append((config_path, environment))
            if config_path is not None:
                self.core_config_path = Path(config_path).expanduser().resolve()
            if environment:
                self.core_environment = environment
            elif self.core_environment is None:
                self.core_environment = "paper_env"
            return responses.pop(0)

    gui = _DummyGUI()
    stub = HeadlessTradingStub()
    app = PaperAutoTradeApp(gui=gui, headless_stub=stub, core_environment="paper_env")

    profile, settings, _ = app.reload_risk_settings()
    assert profile == "balanced"
    assert settings["max_daily_loss_pct"] == pytest.approx(0.02)
    assert len(gui.calls) == 1
    cfg_path, env_name = gui.calls[0]
    assert env_name == "paper_env"
    assert Path(cfg_path) == app.core_config_path
    assert app.core_environment == "paper_env"
    assert stub.last_risk_settings["max_daily_loss_pct"] == pytest.approx(0.02)


def test_listeners_receive_updates(tmp_path: Path) -> None:
    core_path = tmp_path / "core.yaml"
    _write_core_config(core_path, max_daily_loss=0.02)

    app = PaperAutoTradeApp(core_config_path=core_path, core_environment="paper_env")
    received: list[tuple[dict[str, float], str | None]] = []

    def _listener(settings, profile_name, _profile_cfg) -> None:
        received.append((dict(settings), profile_name))

    app.add_listener(_listener)

    profile, settings, _ = app.reload_risk_settings()
    assert profile == "balanced"
    assert settings["max_daily_loss_pct"] == pytest.approx(0.02)
    assert received
    stored_settings, stored_profile = received[-1]
    assert stored_profile == "balanced"
    assert stored_settings["max_daily_loss_pct"] == pytest.approx(0.02)


def test_auto_reload_detects_changes(tmp_path: Path) -> None:
    core_path = tmp_path / "core.yaml"
    _write_core_config(core_path, max_daily_loss=0.02)

    stub = HeadlessTradingStub()
    app = PaperAutoTradeApp(core_config_path=core_path, core_environment="paper_env", headless_stub=stub)

    profile, settings, _ = app.reload_risk_settings()
    assert profile == "balanced"
    assert settings["max_daily_loss_pct"] == pytest.approx(0.02)
    first_updates = stub.update_count

    app.start_auto_reload(interval=0.05)
    try:
        _write_core_config(core_path, max_daily_loss=0.03)

        deadline = time.time() + 2.0
        while time.time() < deadline and stub.update_count < first_updates + 1:
            time.sleep(0.05)

        assert stub.update_count >= first_updates + 1
        assert stub.last_risk_settings["max_daily_loss_pct"] == pytest.approx(0.03)
    finally:
        app.stop_auto_reload()


def test_auto_reload_recovers_after_missing_file(tmp_path: Path) -> None:
    core_path = tmp_path / "core.yaml"
    _write_core_config(core_path, max_daily_loss=0.02)

    stub = HeadlessTradingStub()
    app = PaperAutoTradeApp(core_config_path=core_path, core_environment="paper_env", headless_stub=stub)

    profile, settings, _ = app.reload_risk_settings()
    assert profile == "balanced"
    assert settings["max_daily_loss_pct"] == pytest.approx(0.02)
    first_updates = stub.update_count

    app.start_auto_reload(interval=0.05)
    try:
        try:
            core_path.unlink()
        except FileNotFoundError:
            pass

        deadline = time.time() + 2.0
        while time.time() < deadline and app._watch_last_mtime is not None:
            time.sleep(0.05)

        _write_core_config(core_path, max_daily_loss=0.04)

        deadline = time.time() + 2.0
        while time.time() < deadline and stub.update_count < first_updates + 1:
            time.sleep(0.05)

        assert stub.update_count >= first_updates + 1
        assert stub.last_risk_settings["max_daily_loss_pct"] == pytest.approx(0.04)
    finally:
        app.stop_auto_reload()


def test_reload_profile_updates_core_risk_profile(tmp_path: Path) -> None:
    core_path = tmp_path / "core.yaml"
    _write_multi_profile_config(core_path)

    bootstrap_ctx = types.SimpleNamespace(risk_profile_name="balanced")
    app = launcher.PaperAutoTradeApp(
        core_config_path=core_path,
        core_environment="paper_env",
        bootstrap_context=bootstrap_ctx,
        enable_gui=False,
        use_dummy_feed=False,
    )

    app.trader._core_risk_profile = "balanced"

    updated_settings = app.reload_risk_profile("aggressive")

    assert app.risk_profile_name == "aggressive"
    assert app.trader._core_risk_profile == "aggressive"
    assert bootstrap_ctx.risk_profile_name == "aggressive"
    assert app.risk_manager_settings is updated_settings
    assert bootstrap_ctx.risk_manager_settings is updated_settings


def test_cli_parsing_with_flags_and_quotes(tmp_path: Path) -> None:
    core_path = tmp_path / "core.yaml"
    _write_core_config(core_path, max_daily_loss=0.02)

    stub = HeadlessTradingStub()
    app = PaperAutoTradeApp(core_config_path=core_path, core_environment="paper_env", headless_stub=stub)
    app.reload_risk_settings()

    fancy_dir = tmp_path / "alt config"
    fancy_dir.mkdir()
    alt_path = fancy_dir / "core.yaml"
    _write_core_config(alt_path, max_daily_loss=0.05)

    quoted_command = f'reload-risk --config "{alt_path}" --env paper_env'
    assert app.handle_cli_command(quoted_command)
    assert app.core_config_path == alt_path.resolve()
    assert stub.last_risk_settings["max_daily_loss_pct"] == pytest.approx(0.05)

    _write_core_config(alt_path, max_daily_loss=0.06)
    flag_command = f'reload-risk -c "{alt_path}" -e paper_env'
    assert app.handle_cli_command(flag_command)
    assert stub.last_risk_settings["max_daily_loss_pct"] == pytest.approx(0.06)


def test_reload_skips_duplicate_updates(tmp_path: Path) -> None:
    core_path = tmp_path / "core.yaml"
    _write_core_config(core_path, max_daily_loss=0.02)

    stub = HeadlessTradingStub()
    app = PaperAutoTradeApp(core_config_path=core_path, core_environment="paper_env", headless_stub=stub)

    app.reload_risk_settings()
    first_count = stub.update_count
    first_signature = app._last_signature
    first_timestamp = app.last_reload_at

    assert app.handle_cli_command(f"reload-risk --config {core_path} --env paper_env")

    assert stub.update_count == first_count
    assert app._last_signature == first_signature
    assert app.last_reload_at is not None
    assert app.last_reload_at >= first_timestamp
    assert app.reload_count == 2


def test_auto_reload_ignores_duplicate_content(tmp_path: Path) -> None:
    core_path = tmp_path / "core.yaml"
    _write_core_config(core_path, max_daily_loss=0.02)

    stub = HeadlessTradingStub()
    app = PaperAutoTradeApp(core_config_path=core_path, core_environment="paper_env", headless_stub=stub)

    app.reload_risk_settings()
    first_count = stub.update_count

    app.start_auto_reload(interval=0.05)
    try:
        # Zapisz tę samą konfigurację jeszcze raz, aby wywołać zmianę mtime bez zmiany ustawień.
        _write_core_config(core_path, max_daily_loss=0.02)

        deadline = time.time() + 1.5
        while time.time() < deadline and stub.update_count == first_count:
            time.sleep(0.05)

        assert stub.update_count == first_count
        assert app.reload_count >= 2
    finally:
        app.stop_auto_reload()
