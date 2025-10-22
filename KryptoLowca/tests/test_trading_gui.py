"""Testy funkcjonalne nowego Trading GUI."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from textwrap import dedent

import pytest
import tkinter as tk

import importlib

import KryptoLowca.trading_gui as trading_gui
from KryptoLowca.trading_gui import AppState, TradingGUI, TradingSessionController
from KryptoLowca.ui import trading as trading_package
from KryptoLowca.ui.trading import app as trading_app_module
from bot_core.runtime.metadata import RiskManagerSettings, derive_risk_manager_settings
from KryptoLowca.exchanges import AdapterError


@pytest.fixture
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError:  # pragma: no cover - brak wyświetlacza
        pytest.skip("Tkinter display not available")
    yield root
    try:
        root.destroy()
    except Exception:  # pragma: no cover - defensywne
        pass


@pytest.mark.asyncio
async def test_worker_one_iteration(app):
    await app._load_markets()
    k = list(app.symbol_vars.keys())[0]
    app.symbol_vars[k].set(True)
    app._apply_symbol_selection()
    await app._process_symbol(k)
    assert k in app.paper_positions
    assert app.engine.last_live_tick[0] == k

@pytest.mark.asyncio
async def test_backtest_and_report_export(app, tmp_path):
    await app._load_markets()
    k = list(app.symbol_vars.keys())[0]
    app.symbol_vars[k].set(True)
    app._apply_symbol_selection()
    await app._run_backtest()
    out = tmp_path / "report.pdf"
    app.reporter.export_pdf = lambda fn: out.write_bytes(b"%PDF-1.4 test")
    await app._export_pdf_report()
    assert out.exists()

@pytest.mark.asyncio
async def test_presets_roundtrip(app):
    data = app._gather_settings()
    await app.config_mgr.save_user_config(1, "unit", data)
    got = await app.config_mgr.load_config(preset_name="unit", user_id=1)
    assert got and got["ai"]["enable"] == data["ai"]["enable"]

@pytest.mark.asyncio
async def test_keys_roundtrip(app):
    app.password_var.set("s3cret")
    app.testnet_key.set("A" * 16)
    app.testnet_secret.set("B" * 16)
    app.live_key.set("C" * 16)
    app.live_secret.set("D" * 16)
    await app._save_keys()
    app.testnet_key.set("")
    app.testnet_secret.set("")
    app.live_key.set("")
    app.live_secret.set("")
    await app._load_keys()
    assert app.testnet_key.get().startswith("A") and app.live_secret.get().startswith("D")

@pytest.mark.asyncio
async def test_dashboard_update(app):
    await app._update_dashboard()
    assert "PnL: 100.0" in app.pnl_var.get()
    assert "Positions: 0" in app.positions_var.get()

@pytest.mark.asyncio
async def test_risk_profile_section_updates(app):
    app.set_risk_profile_context("balanced")
    app.risk_manager_settings = {
        "risk_per_trade": 0.015,
        "portfolio_risk": 0.25,
    }
    await asyncio.sleep(0)

    assert app.risk_profile_display_var.get() == "balanced"
    limits = app.risk_limits_display_var.get()
    assert "1.5% per trade" in limits
    assert "25.0% exposure cap" in limits


@pytest.mark.asyncio
async def test_invalid_symbol_selection(app):
    await app._load_markets()
    app._apply_symbol_selection()
    assert not app.selected_symbols
    # Should not crash
    await app._run_backtest()


def _write_core_config(
    path: Path,
    *,
    max_daily_loss: float,
    max_position: float,
    hard_drawdown: float,
    stop_loss: float = 1.5,
) -> None:
    path.write_text(
        dedent(
            f"""
            risk_profiles:
              balanced:
                max_daily_loss_pct: {max_daily_loss}
                max_position_pct: {max_position}
                target_volatility: 0.1
                max_leverage: 3.0
                stop_loss_atr_multiple: {stop_loss}
                max_open_positions: 5
                hard_drawdown_pct: {hard_drawdown}
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


@pytest.mark.asyncio
async def test_reload_risk_manager_settings_updates_gui(app, tmp_path):
    core_path = tmp_path / "core.yaml"
    _write_core_config(core_path, max_daily_loss=0.02, max_position=0.05, hard_drawdown=0.08, stop_loss=1.5)

    app.core_config_path = core_path
    app.core_environment = "paper_env"

    old_mgr = app.risk_mgr
    profile_name, settings, _ = app.reload_risk_manager_settings()

    assert profile_name == "balanced"
    assert settings["max_daily_loss_pct"] == pytest.approx(0.02)
    assert app.max_daily_loss_pct == pytest.approx(0.02)
    assert app.risk_per_trade.get() == pytest.approx(0.05)
    assert app.portfolio_risk.get() == pytest.approx(0.08)
    assert app.risk_profile_var.get() == "balanced"
    assert app.trail_atr_mult_var.get() == pytest.approx(1.5)
    assert app.risk_mgr is not old_mgr

    _write_core_config(core_path, max_daily_loss=0.03, max_position=0.07, hard_drawdown=0.1, stop_loss=2.25)

    profile_name2, settings2, _ = app.reload_risk_manager_settings()
    assert profile_name2 == "balanced"
    assert settings2["max_daily_loss_pct"] == pytest.approx(0.03)
    assert app.max_daily_loss_pct == pytest.approx(0.03)
    assert app.trail_atr_mult_var.get() == pytest.approx(2.25)


def test_sync_positions_from_service_spot(tmp_path, monkeypatch):
    db_path = tmp_path / "spot_positions.db"
    db_url = f"sqlite+aiosqlite:///{db_path.as_posix()}"
    ex_mgr = ExchangeManager(exchange_id="binance", db_url=db_url)
    ex_mgr.set_mode(spot=True)
    db = ex_mgr._ensure_db()
    assert db is not None
    db.sync.upsert_position({
        "symbol": "BTC/USDT",
        "side": "LONG",
        "quantity": 0.5,
        "avg_price": 25_000.0,
        "unrealized_pnl": 123.45,
        "mode": "live",
    })

    def start(self) -> None:  # pragma: no cover - prosta logika
        self.started = True
        self.state.running = True
        self.state.status.set("Sesja start")

    def stop(self) -> None:  # pragma: no cover - prosta logika
        self.stopped = True
        self.state.running = False
        self.state.status.set("Sesja stop")

    def ensure_exchange(self):  # pragma: no cover - prosty stub
        return self._exchange

    def get_exchange(self):  # pragma: no cover - prosty stub
        return self._exchange

    def update_risk_settings(self, settings):  # pragma: no cover - prosty stub
        self.updated_settings = settings
        self.state.risk_manager_settings = settings


def test_trading_gui_initializes_with_injected_controller(tk_root):
    created = {}

    def controller_factory(state: AppState) -> DummyController:
        dummy = DummyController(state)
        created["controller"] = dummy
        return dummy

    gui = TradingGUI(tk_root, session_controller_factory=controller_factory)
    assert gui.root is tk_root
    assert isinstance(gui.controller, DummyController)
    assert gui.state.status.get() == "Kontroler testowy zainicjalizowany"
    assert gui.view is not None
    assert created["controller"].state is gui.state
    assert isinstance(gui.risk_manager_settings, RiskManagerSettings)
    assert gui.state.risk_manager_settings is gui.risk_manager_settings
    assert gui.state.risk_profile_label.get().startswith("Profil ryzyka")
    assert gui.state.risk_limits_label.get().startswith("Limity ryzyka")
    assert gui.state.fraction.get() == pytest.approx(0.02)
    notional_label = gui.state.risk_notional_label.get()
    assert notional_label.startswith("Domyślna kwota")
    assert "200" in notional_label


def test_trading_gui_uses_env_default_symbol(monkeypatch, tk_root):
    monkeypatch.setenv("TRADING_GUI_DEFAULT_SYMBOL", "ltc/usd")

    gui = TradingGUI(tk_root, session_controller_factory=DummyController)

    assert gui.state.symbol.get() == "LTC-USD"
    assert gui.state.market_symbol.get() == "LTC-USD"
def test_trading_gui_reuses_injected_frontend_services(monkeypatch, tk_root):
    class _StubServices:
        def __init__(self) -> None:
            self.exchange_manager = object()
            self.market_intel = object()
            self.execution_service = object()
            self.account_manager = object()
            self.router = object()

    services = _StubServices()
    called: dict[str, bool] = {"bootstrap": False}

    def _fail_bootstrap(*args, **kwargs):  # pragma: no cover - defensywne
        called["bootstrap"] = True
        raise AssertionError("bootstrap_frontend_services should not be invoked")

    monkeypatch.setattr(trading_app_module, "bootstrap_frontend_services", _fail_bootstrap)

    class StubController:
        def __init__(self, state: AppState) -> None:
            self.state = state
            self.market_intel = None
            self._exchange = services.exchange_manager
            self.updated_settings = None

        def start(self) -> None:  # pragma: no cover - prosty stub
            self.state.running = True

        def stop(self) -> None:  # pragma: no cover - prosty stub
            self.state.running = False

        def ensure_exchange(self):  # pragma: no cover - prosty stub
            return self._exchange

        def get_exchange(self):  # pragma: no cover - prosty stub
            return self._exchange

        def update_risk_settings(self, settings) -> None:
            self.updated_settings = settings

    gui = TradingGUI(
        tk_root,
        session_controller_factory=StubController,
        frontend_services=services,
    )

    assert gui.frontend_services is services
    assert gui.market_intel is services.market_intel
    assert gui.ex_mgr is services.exchange_manager
    assert isinstance(gui.controller, StubController)
    assert gui.controller.updated_settings is not None
    assert called["bootstrap"] is False


def test_trading_gui_start_stop_updates_state(tk_root):
    gui = TradingGUI(tk_root, session_controller_factory=DummyController)
    gui.view._start_clicked()
    assert gui.state.running is True
    assert gui.controller.started is True
    gui.view._stop_clicked()
    assert gui.state.running is False
    assert gui.controller.stopped is True


def test_logging_handler_appends_to_view(tk_root):
    gui = TradingGUI(tk_root, session_controller_factory=DummyController)
    logger = logging.getLogger("test_gui_logger")
    logger.setLevel(logging.INFO)
    logger.info("Przykladowa wiadomosc")
    contents = gui.view.log_text.get("1.0", "end").strip()
    assert "Przykladowa wiadomosc" in contents


def test_gui_compatibility_surface_exposes_legacy_fields(tk_root):
    gui = TradingGUI(tk_root, session_controller_factory=DummyController)
    assert gui.network_var is gui.state.network
    assert gui.timeframe_var is gui.state.timeframe
    assert hasattr(gui, "symbol_var")
    assert isinstance(gui.paper_balance, float)
    assert isinstance(gui._open_positions, dict)
    assert isinstance(gui.risk_manager_settings, RiskManagerSettings)


def test_gui_bridge_execute_trade_simulates_positions(tk_root):
    gui = TradingGUI(tk_root, session_controller_factory=DummyController)
    gui._bridge_execute_trade("BTC/USDT", "buy", 10000.0)
    assert "BTC/USDT" in gui._open_positions
    before_balance = gui.paper_balance
    qty = gui._open_positions["BTC/USDT"]["qty"]
    gui._bridge_execute_trade("BTC/USDT", "sell", 11000.0)
    assert "BTC/USDT" not in gui._open_positions
    expected_balance = before_balance + (11000.0 - 10000.0) * qty
    assert gui.paper_balance == pytest.approx(expected_balance)
    display_value = gui.state.paper_balance.get().replace(" ", "").replace(",", "")
    assert float(display_value) == pytest.approx(expected_balance)
    label_value = gui.state.risk_notional_label.get().split(":", 1)[1].strip().split(" ")[0]
    assert float(label_value) == pytest.approx(expected_balance * gui.state.fraction.get())


def test_gui_accepts_custom_trade_executor(tk_root):
    calls = []

    def custom_executor(gui: TradingGUI, symbol: str, side: str, price: float) -> None:
        calls.append((symbol, side, price))
        gui.default_trade_executor(symbol, side, price)

    gui = TradingGUI(
        tk_root,
        session_controller_factory=DummyController,
        trade_executor=custom_executor,
    )
    gui._bridge_execute_trade("ETH/USDT", "buy", 1500.0)
    assert calls == [("ETH/USDT", "buy", 1500.0)]
    assert "ETH/USDT" in gui._open_positions

    # reset executor to domyślny i zamknij pozycję
    gui.set_trade_executor(None)
    gui._bridge_execute_trade("ETH/USDT", "sell", 1400.0)
    assert "ETH/USDT" not in gui._open_positions


def test_public_reexports_align_with_new_package():
    assert trading_package.TradingGUI is trading_gui.TradingGUI
    assert trading_package.TradingSessionController is TradingSessionController
    assert trading_package.AppState is AppState


def test_emitter_launcher_imports_modular_gui(monkeypatch):
    module = importlib.reload(
        importlib.import_module("KryptoLowca.run_trading_gui_paper_emitter")
    )
    assert module.TradingGUI is trading_package.TradingGUI


def test_trading_gui_uses_runtime_risk_profile(monkeypatch, tk_root):
    profile_payload = {
        "max_position_pct": 0.07,
        "max_daily_loss_pct": 0.12,
        "max_open_positions": 7,
        "hard_drawdown_pct": 0.21,
        "target_volatility": 0.18,
    }

    monkeypatch.setattr(
        trading_app_module,
        "load_risk_manager_settings",
        lambda *_, **__: (
            "balanced",
            profile_payload,
            derive_risk_manager_settings(profile_payload, profile_name="balanced"),
        ),
    )

    gui = TradingGUI(tk_root, session_controller_factory=DummyController)

    assert gui.risk_profile_name == "balanced"
    assert gui.state.risk_profile_name == "balanced"
    assert gui.risk_profile_config == profile_payload
    assert gui.risk_manager_config["max_risk_per_trade"] == pytest.approx(0.07)
    assert gui.risk_manager_config["max_portfolio_risk"] > gui.risk_manager_config["max_risk_per_trade"]
    assert gui.risk_manager_config["max_positions"] == 7
    assert gui.risk_manager_settings.max_daily_loss_pct == pytest.approx(0.12)
    assert gui.risk_manager_settings.confidence_level is not None
    assert gui.state.risk_manager_settings is gui.risk_manager_settings
    assert "Domyślna kwota" in gui.state.risk_notional_label.get()
    assert "700" in gui.state.risk_notional_label.get()


def test_trading_gui_reload_risk_profile_updates_labels(monkeypatch, tk_root):
    initial_settings = RiskManagerSettings(
        max_risk_per_trade=0.02,
        max_daily_loss_pct=0.1,
        max_portfolio_risk=0.2,
        max_positions=5,
        emergency_stop_drawdown=0.2,
    )
    updated_settings = RiskManagerSettings(
        max_risk_per_trade=0.06,
        max_daily_loss_pct=0.15,
        max_portfolio_risk=0.35,
        max_positions=8,
        emergency_stop_drawdown=0.25,
    )

    responses = [
        ("balanced", {"max_position_pct": 0.02}, initial_settings),
        ("aggressive", {"max_position_pct": 0.06}, updated_settings),
    ]

    def fake_loader(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(trading_app_module, "load_risk_manager_settings", fake_loader)

    gui = TradingGUI(tk_root, session_controller_factory=DummyController)

    assert "balanced" in gui.state.risk_profile_label.get()
    assert "Domyślna kwota" in gui.state.risk_notional_label.get()
    assert "200" in gui.state.risk_notional_label.get()

    gui.reload_risk_profile("aggressive")

    assert gui.risk_profile_name == "aggressive"
    assert "aggressive" in gui.state.risk_profile_label.get()
    assert "35.0%" in gui.state.risk_limits_label.get()
    assert gui.controller.updated_settings is gui.risk_manager_settings
    assert gui.state.fraction.get() == pytest.approx(0.06)
    assert "600" in gui.state.risk_notional_label.get()


def test_trading_gui_fraction_spinbox_respects_risk_limits(monkeypatch, tk_root):
    responses = [
        (
            "balanced",
            {"max_position_pct": 0.012},
            RiskManagerSettings(
                max_risk_per_trade=0.012,
                max_daily_loss_pct=0.08,
                max_portfolio_risk=0.18,
                max_positions=6,
                emergency_stop_drawdown=0.2,
            ),
        ),
        (
            "aggressive",
            {"max_position_pct": 0.08},
            RiskManagerSettings(
                max_risk_per_trade=0.08,
                max_daily_loss_pct=0.20,
                max_portfolio_risk=0.35,
                max_positions=8,
                emergency_stop_drawdown=0.25,
            ),
        ),
    ]

    monkeypatch.setattr(
        trading_app_module, "load_risk_manager_settings", lambda *_, **__: responses.pop(0)
    )

    gui = TradingGUI(tk_root, session_controller_factory=DummyController)

    minimum, maximum, increment = gui.view.get_fraction_limits()
    assert minimum == pytest.approx(0.0)
    assert maximum == pytest.approx(0.012)
    assert increment == pytest.approx(0.0024)
    assert gui.state.fraction.get() == pytest.approx(0.012)
    assert "120" in gui.state.risk_notional_label.get()

    gui.state.fraction.set(0.5)
    gui.reload_risk_profile("aggressive")

    minimum2, maximum2, increment2 = gui.view.get_fraction_limits()
    assert minimum2 == pytest.approx(0.0)
    assert maximum2 == pytest.approx(0.08)
    assert increment2 == pytest.approx(0.01)
    assert gui.state.fraction.get() == pytest.approx(0.08)
    assert "800" in gui.state.risk_notional_label.get()


def test_trading_gui_refresh_button_invokes_reload(monkeypatch, tk_root):
    monkeypatch.setattr(
        trading_app_module,
        "load_risk_manager_settings",
        lambda *_, **__: (None, None, RiskManagerSettings(
            max_risk_per_trade=0.02,
            max_daily_loss_pct=0.1,
            max_portfolio_risk=0.2,
            max_positions=5,
            emergency_stop_drawdown=0.2,
        )),
    )

    gui = TradingGUI(tk_root, session_controller_factory=DummyController)

    calls: list[object] = []

    def mark_reload(profile: str | None = None):
        calls.append(profile)
        return gui.risk_manager_settings

    gui.reload_risk_profile = mark_reload  # type: ignore[assignment]
    gui.view._refresh_risk_clicked()

    assert calls == [None]


def test_trading_gui_notifies_risk_reload_listeners(monkeypatch, tk_root):
    responses = [
        (
            "balanced",
            {"max_position_pct": 0.02},
            RiskManagerSettings(
                max_risk_per_trade=0.02,
                max_daily_loss_pct=0.1,
                max_portfolio_risk=0.2,
                max_positions=5,
                emergency_stop_drawdown=0.2,
            ),
        ),
        (
            "aggressive",
            {"max_position_pct": 0.06},
            RiskManagerSettings(
                max_risk_per_trade=0.06,
                max_daily_loss_pct=0.18,
                max_portfolio_risk=0.32,
                max_positions=7,
                emergency_stop_drawdown=0.25,
            ),
        ),
    ]

    monkeypatch.setattr(trading_app_module, "load_risk_manager_settings", lambda *_, **__: responses.pop(0))

    gui = TradingGUI(tk_root, session_controller_factory=DummyController)
    captured: list[tuple[str | None, RiskManagerSettings, object | None]] = []

    gui.add_risk_reload_listener(lambda name, settings, payload: captured.append((name, settings, payload)))

    gui.reload_risk_profile("aggressive")

    assert captured
    name, settings, payload = captured[-1]
    assert name == "aggressive"
    assert settings.max_risk_per_trade == pytest.approx(0.06)
    assert payload == {"max_position_pct": 0.06}


def test_trading_gui_watchdog_triggers_reload(monkeypatch, tk_root, tmp_path: Path):
    gui = TradingGUI(tk_root, session_controller_factory=DummyController)
    fake_config = tmp_path / "core.yaml"
    fake_config.write_text("runtime: {}", encoding="utf-8")
    gui._core_config_path = fake_config
    gui._risk_config_mtime = 1.0

    calls: list[str | None] = []

    monkeypatch.setattr(gui, "_get_risk_config_timestamp", lambda: 2.0)

    def fake_reload(profile: str | None = None):
        calls.append(profile)
        return gui.risk_manager_settings

    gui.reload_risk_profile = fake_reload  # type: ignore[assignment]
    monkeypatch.setattr(gui.root, "after", lambda *args, **kwargs: None)

    changed = gui._risk_watchdog_tick()

    assert changed is True
    assert calls == [None]


@pytest.mark.asyncio
async def test_market_data_worker_reports_factory_error(tk_root):
    def failing_factory(*, demo_mode: bool):
        raise RuntimeError("boom")

    gui = TradingGUI(
        tk_root,
        session_controller_factory=DummyController,
        market_data_adapter_factory=failing_factory,
    )

    await gui._market_data_worker(["BTC-PLN"], True)

    assert gui.state.status.get() == "Błąd uruchamiania adaptera rynku"


@pytest.mark.asyncio
async def test_market_data_worker_reports_connection_error(tk_root):
    class FailingAdapter:
        async def connect(self) -> None:
            raise RuntimeError("connect failed")

        async def close(self) -> None:
            pass

    gui = TradingGUI(
        tk_root,
        session_controller_factory=DummyController,
        market_data_adapter_factory=lambda *, demo_mode: FailingAdapter(),
    )

    await gui._market_data_worker(["BTC-PLN"], True)

    assert gui.state.status.get() == "Nie udało się połączyć z rynkiem (REST)"


@pytest.mark.asyncio
async def test_market_data_worker_sets_success_status(monkeypatch, tk_root):
    class SuccessfulAdapter:
        async def connect(self) -> None:
            return None

        async def close(self) -> None:
            return None

    captured: list[tuple[str, dict[str, float]]] = []

    class StubPoller:
        def __init__(self, adapter, *, symbols, interval, callback, error_callback=None):
            self._callback = callback
            self._error_callback = error_callback

        async def start(self) -> None:
            await self._callback("BTC-PLN", {"last": 101.0})

        async def stop(self) -> None:
            return None

    monkeypatch.setattr(trading_app_module, "MarketDataPoller", StubPoller)

    gui = TradingGUI(
        tk_root,
        session_controller_factory=DummyController,
        market_data_adapter_factory=lambda *, demo_mode: SuccessfulAdapter(),
        market_data_interval=0.01,
    )
    gui._market_data_stop.clear()
    original_callback = gui._market_data_callback

    async def record_callback(symbol: str, payload: dict[str, float]) -> None:
        captured.append((symbol, payload))
        await original_callback(symbol, payload)
        gui._market_data_stop.set()

    monkeypatch.setattr(gui, "_market_data_callback", record_callback)

    await gui._market_data_worker(["BTC-PLN"], True)
    gui._drain_market_data_queue()

    assert gui.state.status.get() == "Ticker REST aktywny"
    assert captured and captured[0][0] == "BTC-PLN"
    assert gui.state.market_price.get() == "101.00"


def test_stop_market_data_sets_status(tk_root):
    gui = TradingGUI(tk_root, session_controller_factory=DummyController)

    gui._stop_market_data()

    assert gui.state.status.get() == "Ticker zatrzymany"


@pytest.mark.asyncio
async def test_market_data_error_sets_status(tk_root):
    gui = TradingGUI(tk_root, session_controller_factory=DummyController)

    await gui._market_data_error("btc-pln", RuntimeError("timeout"))

    assert "Błąd REST tickera BTC-PLN" in gui.state.status.get()


def test_market_data_interval_from_env(monkeypatch, tk_root):
    monkeypatch.setenv("TRADING_GUI_MARKET_INTERVAL", "4.25")

    gui = TradingGUI(tk_root, session_controller_factory=DummyController)

    assert gui._market_data_interval == pytest.approx(4.25)


def test_market_data_interval_invalid_env_falls_back(monkeypatch, tk_root, caplog):
    caplog.set_level("WARNING")
    monkeypatch.setenv("TRADING_GUI_MARKET_INTERVAL", "0")

    gui = TradingGUI(tk_root, session_controller_factory=DummyController)

    assert gui._market_data_interval == pytest.approx(2.0)
    assert any(
        "TRADING_GUI_MARKET_INTERVAL" in record.message for record in caplog.records
    )


def test_market_data_adapter_from_env(monkeypatch, tk_root):
    created: dict[str, object] = {}

    class StubAdapter:
        async def connect(self) -> None:
            return None

        async def close(self) -> None:
            return None

    def fake_create(name: str, **options):
        created["name"] = name
        created["options"] = options
        return StubAdapter()

    monkeypatch.setenv("TRADING_GUI_MARKET_ADAPTER", "bitstamp")
    monkeypatch.setenv("TRADING_GUI_MARKET_ADAPTER_OPTIONS", json.dumps({"timeout": 3}))
    monkeypatch.setenv(
        "TRADING_GUI_MARKET_ADAPTER_KWARGS", json.dumps({"enable_streaming": False})
    )
    monkeypatch.setenv("TRADING_GUI_MARKET_COMPLIANCE_ACK", "true")
    monkeypatch.setattr(trading_app_module, "create_exchange_adapter", fake_create)

    gui = TradingGUI(tk_root, session_controller_factory=DummyController)

    adapter = gui._market_data_adapter_factory(demo_mode=False)

    assert isinstance(adapter, StubAdapter)
    assert created["name"] == "bitstamp"
    options = created["options"]
    assert options["demo_mode"] is False
    assert options["sandbox"] is False
    assert options["testnet"] is False
    assert options["timeout"] == 3
    assert options["compliance_ack"] is True
    assert options["adapter_kwargs"]["enable_streaming"] is False


def test_market_data_adapter_env_fallback_on_error(monkeypatch, tk_root):
    monkeypatch.setenv("TRADING_GUI_MARKET_ADAPTER", "bitstamp")

    def fake_create(name: str, **options):
        raise AdapterError("boom")

    monkeypatch.setattr(trading_app_module, "create_exchange_adapter", fake_create)

    gui = TradingGUI(tk_root, session_controller_factory=DummyController)

    adapter = gui._market_data_adapter_factory(demo_mode=True)

    assert isinstance(adapter, trading_app_module.ZondaAdapter)
