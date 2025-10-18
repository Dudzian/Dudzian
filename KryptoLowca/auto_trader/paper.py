"""Komponenty launchera AutoTradera w trybie papierowym."""

from __future__ import annotations

import logging
import signal
import threading
from dataclasses import dataclass, field
from types import SimpleNamespace
from pathlib import Path
from typing import Callable, Iterable, Optional

from KryptoLowca.event_emitter_adapter import (
    DummyMarketFeed,
    DummyMarketFeedConfig,
    EmitterAdapter,
    wire_gui_logs_to_adapter,
)
from KryptoLowca.logging_utils import get_logger

from .app import AutoTrader
from bot_core.runtime.metadata import (
    RiskManagerSettings,
    load_risk_manager_settings,
)
from bot_core.runtime.paths import resolve_core_config_path


logger = get_logger("paper-autotrade")

DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_PAPER_BALANCE = 10_000.0


def _default_risk_settings() -> RiskManagerSettings:
    """Buduje konserwatywny zestaw ustawień ryzyka dla trybu headless."""

    return RiskManagerSettings(
        max_risk_per_trade=0.05,
        max_daily_loss_pct=0.20,
        max_portfolio_risk=0.20,
        max_positions=10,
        emergency_stop_drawdown=0.20,
    )


@dataclass(frozen=True, slots=True)
class PaperAutoTradeOptions:
    """Zbiór flag CLI dla launchera papierowego."""

    enable_gui: bool = True
    use_dummy_feed: bool = True
    symbol: str = DEFAULT_SYMBOL
    paper_balance: float = DEFAULT_PAPER_BALANCE
    core_config_path: str | None = None
    risk_profile: str | None = None


@dataclass
class HeadlessTradingStub:
    """Minimalny obiekt zgodny z API ``TradingGUI`` wykorzystywany w trybie headless."""

    symbol: str = DEFAULT_SYMBOL
    paper_balance: float = DEFAULT_PAPER_BALANCE
    risk_profile_name: str | None = None
    risk_manager_settings: RiskManagerSettings = field(default_factory=_default_risk_settings)

    def __post_init__(self) -> None:
        self.account_balance = float(self.paper_balance)
        self.network_var = SimpleNamespace(get=lambda: "demo")
        self.timeframe_var = SimpleNamespace(get=lambda: "1m")
        self.symbol_var = SimpleNamespace(get=self.get_symbol)
        self._open_positions: dict[str, dict[str, float]] = {}
        self._logs: list[str] = []

    def get_symbol(self) -> str:
        return self.symbol

    def is_demo_mode_active(self) -> bool:
        return True

    def is_live_trading_allowed(self) -> bool:
        return True

    def get_portfolio_snapshot(self, symbol: str) -> dict[str, float | dict[str, dict[str, float]]]:
        position = self._open_positions.get(symbol.upper(), {})
        position_notional = position.get("qty", 0.0)
        if position.get("side") == "sell":
            position_notional *= -1.0
        return {
            "symbol": symbol,
            "portfolio_value": float(self.paper_balance),
            "position": position_notional,
            "positions": dict(self._open_positions),
        }

    def apply_risk_profile(
        self,
        name: str | None,
        settings: RiskManagerSettings | None,
    ) -> None:
        """Aktualizuje profil oraz ustawienia ryzyka wykorzystywane przez stub."""

        if settings is None:
            settings = _default_risk_settings()
        self.risk_profile_name = name
        self.risk_manager_settings = settings
        self.account_balance = float(self.paper_balance)

    def _bridge_execute_trade(self, symbol: str, side: str, price: float) -> None:
        """Symuluje wykonanie transakcji na potrzeby AutoTradera."""

        try:
            price_f = float(price)
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.warning("Headless stub otrzymał niepoprawną cenę: %r", price)
            return

        side_norm = (side or "").lower()
        symbol_key = (symbol or "").upper() or self.symbol.upper()
        if side_norm not in {"buy", "sell"}:
            logger.warning("Headless stub otrzymał nieobsługiwany kierunek: %s", side)
            return

        position = self._open_positions.get(symbol_key)
        if side_norm == "buy":
            fraction = getattr(self.risk_manager_settings, "max_risk_per_trade", 0.05) or 0.05
            if fraction <= 0:
                fraction = 0.05
            notional = max(self.paper_balance * fraction, 0.0)
            qty = notional / price_f if price_f > 0 else 0.0
            if qty <= 0:
                logger.warning("Headless stub nie mógł obliczyć wielkości pozycji dla %s", symbol_key)
                return
            self._open_positions[symbol_key] = {"side": "buy", "qty": qty, "entry": price_f}
            logger.info("Headless stub BUY %s qty=%.6f @ %.2f", symbol_key, qty, price_f)
            return

        if not position:
            logger.warning("Headless stub SELL %s – brak otwartej pozycji", symbol_key)
            return

        qty = float(position.get("qty", 0.0))
        entry = float(position.get("entry", price_f))
        pnl = (price_f - entry) * qty
        self.paper_balance += pnl
        self.account_balance = self.paper_balance
        logger.info(
            "Headless stub SELL %s qty=%.6f @ %.2f pnl=%.2f",
            symbol_key,
            qty,
            price_f,
            pnl,
        )
        self._open_positions.pop(symbol_key, None)


class PaperAutoTradeApp:
    """Zarządza cyklem życia AutoTradera i opcjonalnym feedem w trybie papierowym."""

    def __init__(
        self,
        *,
        symbol: str = DEFAULT_SYMBOL,
        enable_gui: bool = True,
        use_dummy_feed: bool = True,
        paper_balance: float = DEFAULT_PAPER_BALANCE,
        core_config_path: str | Path | None = None,
        risk_profile: str | None = None,
    ) -> None:
        self.symbol = symbol or DEFAULT_SYMBOL
        self.enable_gui = enable_gui
        self.use_dummy_feed = use_dummy_feed
        self.paper_balance = paper_balance

        self.core_config_path = self._resolve_core_config_path(core_config_path)
        (
            self.risk_profile_name,
            self.risk_profile_config,
            loaded_settings,
        ) = self._load_risk_settings(risk_profile)
        self.risk_manager_settings = loaded_settings or _default_risk_settings()
        if self.risk_profile_name:
            logger.info(
                "Profil ryzyka dla launchera paper: %s", self.risk_profile_name
            )
        else:
            logger.info("Launcher paper korzysta z domyślnego profilu ryzyka")

        self.adapter = EmitterAdapter()
        wire_gui_logs_to_adapter(self.adapter)

        self._gui_risk_listener_active = False
        self.gui, self.symbol_getter = self._build_gui()
        self.trader = AutoTrader(
            self.adapter.emitter,
            self.gui,
            self.symbol_getter,
            walkforward_interval_s=None,
        )
        self.feed = self._build_feed()
        self._stop_event = threading.Event()
        self._stopped = True
        self._risk_watch_stop = threading.Event()
        self._risk_watch_thread: Optional[threading.Thread] = None
        self._risk_watch_interval = 5.0
        self._risk_config_mtime = self._get_risk_config_mtime()

    def _resolve_core_config_path(self, explicit: str | Path | None) -> Optional[Path]:
        if explicit is None:
            try:
                return resolve_core_config_path()
            except Exception:  # pragma: no cover - środowiska bez runtime
                logger.debug("Nie udało się ustalić ścieżki konfiguracji core", exc_info=True)
                return None
        return Path(explicit)

    def _load_risk_settings(
        self, profile: str | None
    ) -> tuple[str | None, object | None, Optional[RiskManagerSettings]]:
        try:
            return load_risk_manager_settings(
                "auto_trader",
                profile_name=profile,
                config_path=self.core_config_path,
                logger=logger,
            )
        except Exception:  # pragma: no cover - diagnostyka runtime
            logger.exception("Nie udało się wczytać ustawień risk managera")
            return profile, None, None

    def reload_risk_profile(self, profile: str | None = None) -> RiskManagerSettings:
        """Przeładowuje profil ryzyka i propaguje go do GUI lub stuba."""

        requested = profile or self.risk_profile_name
        name, payload, settings = self._load_risk_settings(requested)
        profile_payload = payload

        if name:
            self.risk_profile_name = name
        if payload is not None:
            self.risk_profile_config = payload

        if settings is None:
            settings = _default_risk_settings()

        self.risk_manager_settings = settings

        gui_reload = getattr(self.gui, "reload_risk_profile", None)
        notified_via_gui = False
        if callable(gui_reload):
            try:
                settings = gui_reload(self.risk_profile_name)
                self.risk_manager_settings = settings
                notified_via_gui = self._gui_risk_listener_active
            except Exception:  # pragma: no cover - diagnostyka runtime
                logger.exception("GUI nie zaktualizowało profilu ryzyka")
        else:
            apply_profile = getattr(self.gui, "apply_risk_profile", None)
            if callable(apply_profile):
                try:
                    apply_profile(self.risk_profile_name, settings)
                except Exception:  # pragma: no cover - diagnostyka runtime
                    logger.exception("Stub GUI nie przyjął nowych ustawień ryzyka")

        if hasattr(self.gui, "risk_manager_settings"):
            try:
                setattr(self.gui, "risk_manager_settings", settings)
            except Exception:
                logger.debug("Nie udało się zaktualizować risk_manager_settings na GUI")

        self.risk_profile_config = profile_payload
        if not notified_via_gui:
            self._notify_trader_of_risk_update(settings, profile_payload)
        self._risk_config_mtime = self._get_risk_config_mtime()
        logger.info("Zaktualizowano profil ryzyka na: %s", self.risk_profile_name)
        return settings

    def _build_gui(self) -> tuple[object, Callable[[], str]]:
        if self.enable_gui:
            try:
                import tkinter as tk

                from KryptoLowca.ui.trading import TradingGUI

                root = tk.Tk()
                gui = TradingGUI(root)
                gui.paper_balance = self.paper_balance
                register_listener = getattr(gui, "add_risk_reload_listener", None)
                if callable(register_listener):
                    register_listener(self._handle_gui_risk_reload)
                    self._gui_risk_listener_active = True
                try:
                    root.wm_title("KryptoLowca AutoTrader (paper)")
                except Exception:  # pragma: no cover - brak wsparcia tytułów
                    pass

                def getter() -> str:
                    var = getattr(gui, "symbol_var", None)
                    if var is not None and hasattr(var, "get"):
                        try:
                            value = var.get()
                            if value:
                                return value
                        except Exception:  # pragma: no cover - defensywne
                            logger.debug("Nie udało się pobrać symbolu z GUI", exc_info=True)
                    return self.symbol

                if hasattr(root, "protocol"):
                    root.protocol("WM_DELETE_WINDOW", self.stop)
                return gui, getter
            except Exception:  # pragma: no cover - środowiska bez wyświetlacza
                logger.exception("Nie udało się uruchomić Trading GUI – przełączam na tryb headless")
                self.enable_gui = False

        stub = HeadlessTradingStub(symbol=self.symbol, paper_balance=self.paper_balance)
        stub.apply_risk_profile(self.risk_profile_name, self.risk_manager_settings)
        self._gui_risk_listener_active = False

        def getter() -> str:
            return stub.get_symbol()

        return stub, getter

    def _build_feed(self) -> Optional[DummyMarketFeed]:
        if not self.use_dummy_feed:
            return None
        cfg = DummyMarketFeedConfig(symbol=self.symbol.replace("/", ""), start_price=30_000.0, interval_sec=1.0)
        return DummyMarketFeed(self.adapter, cfg=cfg)

    def start(self) -> None:
        if not self._stopped:
            return
        self._stopped = False
        self.trader.start()
        if self.feed is not None:
            self.feed.start()
        self._start_risk_watcher()
        logger.info("AutoTrader paper app started (symbol=%s, gui=%s)", self.symbol, self.enable_gui)

    def stop(self, *_: object) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._stop_event.set()
        self._stop_risk_watcher()
        try:
            self.trader.stop()
        except Exception:  # pragma: no cover - defensywne
            logger.exception("Nie udało się zatrzymać AutoTradera")
        if self.feed is not None:
            try:
                self.feed.stop()
                self.feed.join(timeout=2.0)
            except Exception:  # pragma: no cover - defensywne
                logger.debug("Problem z zatrzymaniem DummyMarketFeed", exc_info=True)
        try:
            self.adapter.bus.close()
        except Exception:  # pragma: no cover - defensywne
            logger.debug("Problem z zamknięciem EventBus", exc_info=True)
        logger.info("AutoTrader paper app stopped")

    def run(self) -> None:
        self.start()

        def _shutdown_handler(sig: int, _frame: Optional[object]) -> None:
            logger.info("Odebrano sygnał %s – zatrzymuję AutoTradera", sig)
            self.stop()

        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _shutdown_handler)
        hup_installed = False
        original_hup: object | None = None
        try:
            original_hup = signal.getsignal(signal.SIGHUP)  # type: ignore[attr-defined]
            signal.signal(signal.SIGHUP, self._handle_reload_signal)  # type: ignore[attr-defined]
            hup_installed = True
        except AttributeError:  # pragma: no cover - platformy bez SIGHUP
            original_hup = None
        try:
            if self.enable_gui and hasattr(self.gui, "run"):
                try:
                    self.gui.run()
                finally:
                    self.stop()
            else:
                while not self._stop_event.wait(0.5):
                    pass
        finally:
            signal.signal(signal.SIGINT, original_handler)
            if hup_installed:
                signal.signal(signal.SIGHUP, original_hup)  # type: ignore[attr-defined]

    def _handle_reload_signal(self, sig: int, _frame: Optional[object]) -> None:
        logger.info("Odebrano sygnał %s – przeładowuję profil ryzyka", sig)
        try:
            self.reload_risk_profile()
        except Exception:  # pragma: no cover - diagnostyka runtime
            logger.exception("Nie udało się przeładować profilu ryzyka po sygnale")

    def _get_risk_config_mtime(self) -> Optional[float]:
        if not self.core_config_path:
            return None
        try:
            return Path(self.core_config_path).stat().st_mtime
        except FileNotFoundError:
            return None
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.debug("Nie udało się pobrać mtime konfiguracji core", exc_info=True)
            return None

    def _start_risk_watcher(self) -> None:
        if self.enable_gui or self.core_config_path is None:
            return
        if self._risk_watch_thread and self._risk_watch_thread.is_alive():
            return
        self._risk_watch_stop.clear()

        def _loop() -> None:
            while not self._risk_watch_stop.wait(self._risk_watch_interval):
                try:
                    self._check_risk_config_change()
                except Exception:  # pragma: no cover - defensywne
                    logger.exception("Watcher profilu ryzyka zgłosił wyjątek")

        self._risk_watch_thread = threading.Thread(
            target=_loop,
            name="paper-risk-watch",
            daemon=True,
        )
        self._risk_watch_thread.start()

    def _stop_risk_watcher(self) -> None:
        self._risk_watch_stop.set()
        thread = self._risk_watch_thread
        if thread and thread.is_alive():
            thread.join(timeout=1.5)
        self._risk_watch_thread = None
        self._risk_watch_stop = threading.Event()

    def _check_risk_config_change(self) -> bool:
        new_mtime = self._get_risk_config_mtime()
        if new_mtime is None:
            self._risk_config_mtime = None
            return False
        if self._risk_config_mtime is None:
            self._risk_config_mtime = new_mtime
            return False
        if new_mtime <= self._risk_config_mtime:
            return False
        self._risk_config_mtime = new_mtime
        try:
            self.reload_risk_profile()
        except Exception:  # pragma: no cover - diagnostyka runtime
            logger.exception("Automatyczne przeładowanie profilu ryzyka nie powiodło się")
            return False
        return True

    def _notify_trader_of_risk_update(
        self,
        settings: RiskManagerSettings,
        profile_payload: object | None,
    ) -> None:
        update_method = getattr(self.trader, "update_risk_manager_settings", None)
        if callable(update_method):
            try:
                update_method(
                    settings,
                    profile_name=self.risk_profile_name,
                    profile_config=profile_payload,
                )
            except Exception:  # pragma: no cover - defensywne
                logger.exception("Nie udało się zaktualizować ustawień ryzyka w AutoTraderze")

    def _handle_gui_risk_reload(
        self,
        profile_name: str | None,
        settings: RiskManagerSettings,
        profile_payload: object | None,
    ) -> None:
        if profile_name:
            self.risk_profile_name = profile_name
        self.risk_manager_settings = settings
        self.risk_profile_config = profile_payload
        self._notify_trader_of_risk_update(settings, profile_payload)
        self._risk_config_mtime = self._get_risk_config_mtime()


def parse_cli_args(argv: Iterable[str]) -> PaperAutoTradeOptions:
    """Zamienia argumenty CLI na strukturalne opcje."""

    enable_gui = True
    use_dummy_feed = True
    symbol = DEFAULT_SYMBOL
    paper_balance = DEFAULT_PAPER_BALANCE
    core_config_path: str | None = None
    risk_profile: str | None = None

    args = list(argv)
    skip_next = False
    for idx, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        key = arg.lower()
        if key in {"nogui", "--nogui", "-nogui"}:
            enable_gui = False
            continue
        if key.startswith("real") or key in {"--real"}:
            use_dummy_feed = False
            continue
        if key in {"--no-feed", "nofeed", "--nofeed"}:
            use_dummy_feed = False
            continue
        if key.startswith("--symbol="):
            symbol = arg.split("=", 1)[1] or symbol
            continue
        if key == "--symbol" and idx + 1 < len(args):
            symbol = args[idx + 1]
            skip_next = True
            continue
        if key.startswith("--paper-balance="):
            try:
                paper_balance = float(arg.split("=", 1)[1])
            except Exception:
                logger.warning("Nie udało się sparsować wartości paper balance: %s", arg)
            continue
        if key == "--paper-balance" and idx + 1 < len(args):
            try:
                paper_balance = float(args[idx + 1])
            except Exception:
                logger.warning("Nie udało się sparsować wartości paper balance: %s", args[idx + 1])
            skip_next = True
            continue
        if key.startswith("--core-config="):
            core_config_path = arg.split("=", 1)[1] or core_config_path
            continue
        if key == "--core-config" and idx + 1 < len(args):
            core_config_path = args[idx + 1]
            skip_next = True
            continue
        if key.startswith("--risk-profile="):
            risk_profile = arg.split("=", 1)[1] or risk_profile
            continue
        if key == "--risk-profile" and idx + 1 < len(args):
            risk_profile = args[idx + 1]
            skip_next = True
            continue
        if key in {"-h", "--help"}:
            print(
                "Usage: python -m KryptoLowca.run_autotrade_paper [--nogui] [--no-feed] "
                "[--symbol=PAIR] [--paper-balance=N] [--core-config PATH] [--risk-profile=NAME]"
            )
            raise SystemExit(0)

    return PaperAutoTradeOptions(
        enable_gui=enable_gui,
        use_dummy_feed=use_dummy_feed,
        symbol=symbol,
        paper_balance=paper_balance,
        core_config_path=core_config_path,
        risk_profile=risk_profile,
    )


def main(argv: Optional[Iterable[str]] = None) -> None:
    """Wejście CLI kompatybilne ze starym skryptem."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    effective_argv = list(argv if argv is not None else [])
    options = parse_cli_args(effective_argv)
    app = PaperAutoTradeApp(
        symbol=options.symbol,
        enable_gui=options.enable_gui,
        use_dummy_feed=options.use_dummy_feed,
        paper_balance=options.paper_balance,
        core_config_path=options.core_config_path,
        risk_profile=options.risk_profile,
    )
    app.run()


__all__ = [
    "DEFAULT_SYMBOL",
    "DEFAULT_PAPER_BALANCE",
    "PaperAutoTradeOptions",
    "HeadlessTradingStub",
    "PaperAutoTradeApp",
    "parse_cli_args",
    "main",
]
