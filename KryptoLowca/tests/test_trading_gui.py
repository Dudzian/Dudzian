"""Testy funkcjonalne nowego Trading GUI."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import tkinter as tk

import importlib

import KryptoLowca.trading_gui as trading_gui
from KryptoLowca.trading_gui import AppState, TradingGUI, TradingSessionController
from KryptoLowca.ui import trading as trading_package
from KryptoLowca.ui.trading import app as trading_app_module
from bot_core.runtime.metadata import RiskManagerSettings, derive_risk_manager_settings


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


@pytest.fixture(autouse=True)
def stub_runtime_metadata(monkeypatch):
    monkeypatch.setattr(
        trading_app_module, "load_runtime_entrypoint_metadata", lambda *_, **__: None
    )
    monkeypatch.setattr(
        trading_app_module,
        "load_risk_manager_settings",
        lambda *_, **__: (
            None,
            None,
            RiskManagerSettings(
                max_risk_per_trade=0.02,
                max_daily_loss_pct=0.10,
                max_portfolio_risk=0.10,
                max_positions=10,
                emergency_stop_drawdown=0.15,
            ),
        ),
    )


class DummyController(TradingSessionController):
    def __init__(self, state: AppState) -> None:
        self.state = state
        self.started = False
        self.stopped = False
        self.state.status.set("Kontroler testowy zainicjalizowany")
        self._exchange = object()
        self.updated_settings = None

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
