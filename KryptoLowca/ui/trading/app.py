"""Główne API aplikacji Trading GUI."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Protocol

import tkinter as tk

from collections import defaultdict

try:  # pragma: no cover - zależność opcjonalna
    from bot_core.market_intel import MarketIntelAggregator
except Exception:  # pragma: no cover - fallback gdy moduł nie istnieje
    MarketIntelAggregator = None  # type: ignore[assignment]

from bot_core.runtime.paths import (
    DesktopAppPaths,
    build_desktop_app_paths,
    resolve_core_config_path,
)
from bot_core.runtime.metadata import (
    RuntimeEntrypointMetadata,
    RiskManagerSettings,
    derive_risk_manager_settings,
    load_risk_manager_settings,
    load_runtime_entrypoint_metadata,
)

from KryptoLowca.logging_utils import (
    DEFAULT_LOG_FILE,
    LOGS_DIR as GLOBAL_LOGS_DIR,
    get_logger,
    setup_app_logging,
)
from KryptoLowca.database_manager import DatabaseManager
from KryptoLowca.managers.security_manager import SecurityManager
from KryptoLowca.managers.config_manager import ConfigManager
from KryptoLowca.managers.report_manager import ReportManager
from KryptoLowca.managers.risk_manager_adapter import RiskManager
from KryptoLowca.managers.ai_manager import AIManager
from KryptoLowca.managers.exchange_manager import ExchangeManager

from .state import AppState
from .controller import TradingSessionController
from .view import TradingView
from .risk_helpers import (
    RiskSnapshot,
    build_risk_limits_summary,
    build_risk_profile_hint,
    compute_default_notional,
    format_notional,
    snapshot_from_app,
)


_DEFAULT_FRACTION = 0.05


def _ensure_repo_root() -> None:
    current_dir = Path(__file__).resolve().parent.parent
    for candidate in (current_dir, *current_dir.parents):
        package_init = candidate / "KryptoLowca" / "__init__.py"
        if package_init.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return


if __package__ in (None, ""):
    _ensure_repo_root()


setup_app_logging()
logger = get_logger(__name__)


class TradeExecutor(Protocol):
    """Callable odpowiedzialny za egzekucję transakcji w GUI."""

    def __call__(self, gui: "TradingGUI", symbol: str, side: str, price: float) -> None:
        """Execute trade for ``symbol`` with direction ``side`` at ``price``."""


class TradingGUI:
    """Klasa spinająca widok i logikę."""

    def __init__(
        self,
        root: Optional[tk.Tk] = None,
        *,
        paths: Optional[DesktopAppPaths] = None,
        session_controller_factory: Optional[
            Callable[[AppState], TradingSessionController]
        ] = None,
        trade_executor: Optional[TradeExecutor] = None,
        market_intel: Optional["MarketIntelAggregator"] = None,
    ) -> None:
        self.root = root or tk.Tk()
        self.paths = paths or build_desktop_app_paths(
            __file__,
            logs_dir=GLOBAL_LOGS_DIR,
            text_log_file=DEFAULT_LOG_FILE,
        )
        self.market_intel = market_intel or self._build_default_market_intel()
        self._core_config_path = self._resolve_core_config_path()
        self.runtime_metadata = self._load_metadata(self._core_config_path)
        (
            self._risk_profile_name,
            self._risk_profile_config,
            self.risk_manager_settings,
        ) = self._load_risk_profile(self.runtime_metadata, self._core_config_path)
        self.risk_manager_config = self.risk_manager_settings.to_dict()
        self._risk_config_mtime = self._get_risk_config_timestamp()
        self._risk_watchdog_after: Optional[str] = None
        self._risk_watch_interval_ms = 5_000
        self._risk_reload_listeners: list[
            Callable[[str | None, RiskManagerSettings, Any | None], None]
        ] = []
        self.state = self._create_state()
        controller_factory = session_controller_factory or self._default_controller_factory
        self.controller: TradingSessionController = controller_factory(self.state)
        if getattr(self.controller, "market_intel", None) is None and self.market_intel is not None:
            try:
                self.controller.market_intel = self.market_intel  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensywnie
                logger.debug("Nie udało się wstrzyknąć MarketIntelAggregator do kontrolera", exc_info=True)
        self.view = TradingView(
            self.root,
            self.state,
            self.controller,
            on_refresh_risk=self.reload_risk_profile,
        )
        self._configure_fraction_widget(self.risk_manager_settings)
        self._configure_logging_handler()
        self.ex_mgr = None
        self.network_var = self.state.network
        self.timeframe_var = self.state.timeframe
        self.symbol_var = tk.StringVar(value="BTC/USDT")
        self.paper_balance = self._parse_float(self.state.paper_balance.get())
        self.account_balance = 0.0
        self._open_positions = self.state.open_positions
        self._view_logs: Dict[str, str] = {}
        self.default_trade_executor = self._default_trade_executor
        self._trade_executor_callable = self._wrap_trade_executor(trade_executor)
        self.view.sync_positions()
        self.risk_profile_name = self.state.risk_profile_name
        self.risk_profile_config = self.state.risk_profile_config
        self.risk_manager_settings = self.state.risk_manager_settings or self.risk_manager_settings
        self._update_risk_banner()
        self._start_risk_watchdog()
        try:
            self.root.bind("<Destroy>", self._on_root_destroy, add="+")
        except Exception:  # pragma: no cover - środowiska bez pełnego Tk
            logger.debug("Nie udało się zarejestrować obserwatora Destroy", exc_info=True)
        if hasattr(self.root, "protocol"):
            try:
                self.root.protocol("WM_DELETE_WINDOW", self._handle_window_close)
            except Exception:  # pragma: no cover - środowiska bez WM
                logger.debug("Nie udało się ustawić handlera WM_DELETE_WINDOW", exc_info=True)

    # ------------------------------------------------------------------
    def _default_controller_factory(self, state: AppState) -> TradingSessionController:
        return TradingSessionController(
            state,
            DatabaseManager(),
            SecurityManager(self.paths.keys_file, self.paths.salt_file),
            ConfigManager(self.paths.presets_dir),
            ReportManager(str(self.paths.db_file)),
            RiskManager(config=self.risk_manager_config),
            self._build_ai_manager(),
            exchange_manager=None,
            market_intel=self.market_intel,
        )

    def _build_default_market_intel(self) -> Optional["MarketIntelAggregator"]:
        if MarketIntelAggregator is None:  # pragma: no cover - zależność opcjonalna
            return None

        class _InMemoryCacheStorage:
            def __init__(self) -> None:
                self._data: Dict[str, Dict[str, Iterable[Iterable[float]]]] = defaultdict(dict)

            def read(self, key: str) -> Dict[str, Iterable[Iterable[float]]]:
                return self._data.get(
                    key,
                    {
                        "columns": ("open_time", "close", "volume"),
                        "rows": (
                            (0.0, 26_500.0, 120.0),
                            (60.0, 26_750.0, 118.0),
                            (120.0, 26_900.0, 130.0),
                        ),
                    },
                )

            def write(self, key: str, payload: Dict[str, Iterable[Iterable[float]]]) -> None:
                self._data[key] = payload

            def metadata(self) -> Dict[str, str]:
                return {}

            def latest_timestamp(self, key: str) -> float | None:  # noqa: ARG002
                return 120.0

        try:
            storage = _InMemoryCacheStorage()
            return MarketIntelAggregator(storage)  # type: ignore[call-arg]
        except Exception:  # pragma: no cover - defensywnie
            logger.debug("Nie udało się utworzyć domyślnego MarketIntelAggregator", exc_info=True)
            return None

    # ------------------------------------------------------------------
    def _load_metadata(
        self, config_path: Optional[Path]
    ) -> Optional[RuntimeEntrypointMetadata]:
        try:
            metadata = load_runtime_entrypoint_metadata(
                "trading_gui",
                config_path=config_path,
                logger=logger,
            )
            if metadata:
                logger.info("Runtime metadata: %s", metadata.to_dict())
            else:
                logger.warning("Brak metadanych runtime dla trading_gui")
            return metadata
        except Exception:  # pragma: no cover - środowisko bez konfiguracji
            logger.exception("Nie udało się wczytać metadanych runtime")
            return None

    # ------------------------------------------------------------------
    def _create_state(self) -> AppState:
        profile_label, limits_label = self._initial_risk_labels()
        fraction_value = self._compute_fraction_value(self.risk_manager_settings)
        notional_label = self._initial_default_notional_label(fraction_value)
        return AppState(
            paths=self.paths,
            runtime_metadata=self.runtime_metadata,
            risk_profile_name=self._risk_profile_name,
            risk_profile_config=self._risk_profile_config,
            risk_manager_config=self.risk_manager_config,
            risk_manager_settings=self.risk_manager_settings,
            risk_profile_label=tk.StringVar(value=profile_label),
            risk_limits_label=tk.StringVar(value=limits_label),
            risk_notional_label=tk.StringVar(value=notional_label),
            network=tk.StringVar(value="Testnet"),
            mode=tk.StringVar(value="Spot"),
            timeframe=tk.StringVar(value="1m"),
            fraction=tk.DoubleVar(value=fraction_value),
            paper_balance=tk.StringVar(value="10 000.00"),
            account_balance=tk.StringVar(value="—"),
            status=tk.StringVar(value="Oczekiwanie na start"),
            market_intel_label=tk.StringVar(value="Market intel: —"),
            market_intel_summary="Market intel: —",
            market_intel_auto_save=tk.BooleanVar(value=False),
        )

    # ------------------------------------------------------------------
    def _build_ai_manager(self) -> AIManager:
        try:
            return AIManager(models_dir=self.paths.models_dir, logger_=logger)
        except TypeError:
            try:
                return AIManager(self.paths.models_dir, logger)
            except TypeError:
                return AIManager(self.paths.models_dir)

    # ------------------------------------------------------------------
    def _configure_logging_handler(self) -> None:
        handler = _TkLogHandler(self.view)
        logging.getLogger().addHandler(handler)

    # ------------------------------------------------------------------
    def _resolve_core_config_path(self) -> Optional[Path]:
        try:
            return resolve_core_config_path()
        except Exception:  # pragma: no cover - środowisko bez konfiguracji
            logger.debug(
                "Nie udało się ustalić ścieżki konfiguracji core dla Trading GUI",
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    def _load_risk_profile(
        self,
        metadata: Optional[RuntimeEntrypointMetadata],
        config_path: Optional[Path],
    ) -> tuple[str | None, Any | None, RiskManagerSettings]:
        candidate_name: str | None = None
        if metadata is not None:
            candidate_name = getattr(metadata, "risk_profile", None)
        try:
            (
                resolved_name,
                profile,
                settings,
            ) = load_risk_manager_settings(
                "trading_gui",
                profile_name=candidate_name,
                config_path=config_path,
                logger=logger,
            )
        except Exception:  # pragma: no cover - środowisko bez konfiguracji
            logger.exception("Nie udało się wczytać profilu ryzyka Trading GUI")
            fallback_settings = derive_risk_manager_settings(
                None,
                profile_name=candidate_name,
            )
            return candidate_name, None, fallback_settings
        if resolved_name:
            candidate_name = resolved_name
        if profile is not None:
            logger.info("Zastosowano profil ryzyka %s dla Trading GUI", candidate_name)
        return candidate_name, profile, settings

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def run(self) -> None:
        logger.info("Uruchamiam Trading GUI")
        try:
            self.root.mainloop()
        finally:
            logger.info("Zamykam Trading GUI")

    # ------------------------------------------------------------------
    def _parse_float(self, value: str) -> float:
        try:
            normalised = value.replace(" ", "").replace(",", "")
            return float(normalised)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    def _ensure_exchange(self) -> ExchangeManager:
        """Zachowuje zgodność z dawnym API TradingGUI."""

        ensure = getattr(self.controller, "ensure_exchange", None)
        if ensure is None:
            raise RuntimeError("Brak kontrolera zapewniającego ExchangeManager")
        self.ex_mgr = ensure()
        return self.ex_mgr

    # ------------------------------------------------------------------
    def get_exchange_manager(self) -> Optional[ExchangeManager]:
        """Zwraca bieżący ExchangeManager (dla zgodności)."""

        if self.ex_mgr is not None:
            return self.ex_mgr
        if hasattr(self.controller, "get_exchange"):
            self.ex_mgr = self.controller.get_exchange()
        return self.ex_mgr

    # ------------------------------------------------------------------
    def reload_risk_profile(self, profile_name: str | None = None) -> RiskManagerSettings:
        """Ponownie wczytuje ustawienia profilu ryzyka i aktualizuje GUI."""

        candidate = profile_name or self.risk_profile_name or self._risk_profile_name
        resolved_name = candidate
        profile_payload: Any | None = None
        settings: RiskManagerSettings

        try:
            (
                loaded_name,
                profile_payload,
                loaded_settings,
            ) = load_risk_manager_settings(
                "trading_gui",
                profile_name=candidate,
                config_path=self._core_config_path,
                logger=logger,
            )
        except Exception:
            logger.exception("Błąd podczas ponownego wczytywania profilu ryzyka")
            loaded_name = candidate
            profile_payload = None
            loaded_settings = None

        if loaded_name:
            resolved_name = loaded_name

        if loaded_settings is None:
            settings = derive_risk_manager_settings(
                profile_payload,
                profile_name=resolved_name,
            )
        else:
            settings = loaded_settings

        self.risk_profile_name = resolved_name
        self.risk_profile_config = profile_payload
        self.risk_manager_settings = settings
        self.risk_manager_config = settings.to_dict()
        self._risk_config_mtime = self._get_risk_config_timestamp()

        self.state.risk_profile_name = resolved_name
        self.state.risk_profile_config = profile_payload
        self.state.risk_manager_settings = settings
        self.state.risk_manager_config = self.risk_manager_config

        self._apply_fraction_from_settings(settings)

        controller_update = getattr(self.controller, "update_risk_settings", None)
        if callable(controller_update):
            try:
                controller_update(settings)
            except Exception:
                logger.exception("Kontroler nie przyjął nowych ustawień ryzyka")

        self._update_risk_banner()
        self._notify_risk_reload_listeners(resolved_name, settings, profile_payload)
        return settings

    # ------------------------------------------------------------------
    def _wrap_trade_executor(
        self, trade_executor: Optional[TradeExecutor]
    ) -> Callable[[str, str, float], None]:
        if trade_executor is None:
            return self._default_trade_executor

        def executor(symbol: str, side: str, price: float) -> None:
            trade_executor(self, symbol, side, price)

        return executor

    # ------------------------------------------------------------------
    def _log(self, message: str, level: str = "INFO") -> None:
        """Loguje zdarzenie i dopisuje je do panelu tekstowego GUI."""

        level_norm = (level or "INFO").upper()
        log_fn = getattr(logger, level_norm.lower(), logger.info)
        log_fn(message)
        try:
            formatted = f"[{level_norm}] {message}"
            self.view.append_log(formatted)
            self._view_logs[level_norm] = message
        except Exception:  # pragma: no cover - obrona przed błędami UI
            logger.debug("Nie udało się dopisać logu do widoku", exc_info=True)

    # ------------------------------------------------------------------
    def set_trade_executor(self, trade_executor: Optional[TradeExecutor]) -> None:
        """Ustawia zewnętrzny executor transakcji."""

        self._trade_executor_callable = self._wrap_trade_executor(trade_executor)

    # ------------------------------------------------------------------
    def _bridge_execute_trade(self, symbol: str, side: str, mkt_price: float) -> None:
        """Minimalna symulacja egzekucji wykorzystywana przez moduły paper."""

        self._trade_executor_callable(symbol, side, mkt_price)

    # ------------------------------------------------------------------
    def _default_trade_executor(self, symbol: str, side: str, mkt_price: float) -> None:
        """Domyślna implementacja symulacji transakcji."""

        try:
            price = float(mkt_price)
        except Exception:
            self._log(f"Nieprawidłowa cena: {mkt_price}", "ERROR")
            return

        symbol_key = (symbol or "").upper() or "UNKNOWN"
        side_norm = (side or "").lower()

        if side_norm not in {"buy", "sell"}:
            self._log(f"Nieobsługiwany kierunek transakcji: {side}", "ERROR")
            return

        position = self._open_positions.get(symbol_key)

        if side_norm == "buy":
            fraction = self._get_fraction_from_state()
            notional = max(self.paper_balance * fraction, 0.0)
            qty = notional / price if price > 0 else 0.0
            if qty <= 0:
                self._log("Nie udało się obliczyć wielkości pozycji", "WARNING")
                return
            self._open_positions[symbol_key] = {
                "side": "buy",
                "qty": qty,
                "entry": price,
            }
            self._log(
                f"Symulowany zakup {symbol_key} qty={qty:.6f} @ {price:.2f}",
                "INFO",
            )
            self.view.sync_positions()
            self._update_risk_banner()
            return

        if not position or position.get("side") != "buy":
            self._log(f"Brak pozycji do zamknięcia dla {symbol_key}", "WARNING")
            return

        qty = float(position.get("qty", 0.0) or 0.0)
        entry = float(position.get("entry", price) or price)
        pnl = (price - entry) * qty
        self.paper_balance += pnl
        self.state.paper_balance.set(f"{self.paper_balance:,.2f}")
        self._open_positions.pop(symbol_key, None)
        self._log(
            f"Symulowana sprzedaż {symbol_key} qty={qty:.6f} @ {price:.2f} (PnL={pnl:.2f})",
            "INFO",
        )

        self.view.sync_positions()
        self._update_risk_banner()

    # ------------------------------------------------------------------
    def _initial_risk_labels(self) -> tuple[str, str]:
        snapshot = RiskSnapshot(
            paper_balance=0.0,
            settings=self.risk_manager_settings,
            profile_name=self._risk_profile_name,
        )
        profile_text = build_risk_profile_hint(snapshot) or "Profil ryzyka: —"
        limits_text = build_risk_limits_summary(snapshot) or "Limity ryzyka: —"
        return profile_text, limits_text

    # ------------------------------------------------------------------
    def _initial_default_notional_label(self, fraction_value: float) -> str:
        fallback = self._fallback_default_notional(fraction_value)
        snapshot = RiskSnapshot(
            paper_balance=max(self.paper_balance, 0.0),
            settings=self.risk_manager_settings,
            profile_name=self._risk_profile_name,
        )
        amount = compute_default_notional(snapshot, default_notional=fallback)
        return self._format_default_notional(amount)

    # ------------------------------------------------------------------
    def _fallback_default_notional(self, fraction_value: float | None = None) -> float:
        balance = max(self.paper_balance, 0.0)
        if fraction_value is None:
            fraction_value = self._get_fraction_from_state()
        if fraction_value is None:
            fraction_value = _DEFAULT_FRACTION
        try:
            fraction = float(fraction_value)
        except Exception:
            fraction = _DEFAULT_FRACTION
        fraction = max(0.0, fraction)
        if fraction == 0.0 or balance == 0.0:
            return 0.0
        return balance * fraction

    # ------------------------------------------------------------------
    def _get_fraction_from_state(self) -> float:
        state_obj = getattr(self, "state", None)
        fraction_var = getattr(state_obj, "fraction", None)
        if hasattr(fraction_var, "get"):
            try:
                value = float(fraction_var.get())
            except Exception:
                value = _DEFAULT_FRACTION
        else:
            value = _DEFAULT_FRACTION
        return max(0.0, value)

    # ------------------------------------------------------------------
    def _calculate_default_notional(self) -> float:
        snapshot = snapshot_from_app(self)
        fallback = self._fallback_default_notional()
        return compute_default_notional(snapshot, default_notional=fallback)

    # ------------------------------------------------------------------
    def _format_default_notional(self, amount: float) -> str:
        if amount <= 0:
            return "Domyślna kwota: —"
        return f"Domyślna kwota: {format_notional(amount)} USDT"

    # ------------------------------------------------------------------
    def _compute_fraction_value(
        self, settings: RiskManagerSettings | None
    ) -> float:
        state_fraction: float | None = None
        state_obj = getattr(self, "state", None)
        if state_obj is not None:
            fraction_var = getattr(state_obj, "fraction", None)
            if hasattr(fraction_var, "get"):
                try:
                    state_fraction = float(fraction_var.get())
                except Exception:
                    state_fraction = None
        if isinstance(settings, RiskManagerSettings):
            try:
                value = float(settings.max_risk_per_trade)
            except Exception:
                value = 0.0
            value = max(0.0, min(1.0, value))
            if value > 0:
                return value
        if state_fraction is not None and state_fraction > 0:
            return state_fraction
        return _DEFAULT_FRACTION

    # ------------------------------------------------------------------
    def _configure_fraction_widget(
        self, settings: RiskManagerSettings | None
    ) -> float:
        max_fraction = 1.0
        if isinstance(settings, RiskManagerSettings):
            try:
                candidate = float(settings.max_risk_per_trade)
            except Exception:
                candidate = None
            if candidate is not None and candidate > 0:
                max_fraction = min(1.0, candidate)
        increment = 0.01
        if max_fraction > 0:
            increment = max(0.001, min(0.01, max_fraction / 5))
        view = getattr(self, "view", None)
        if hasattr(view, "configure_fraction_input"):
            try:
                view.configure_fraction_input(maximum=max_fraction, increment=increment)
            except Exception:  # pragma: no cover - defensywne logowanie
                logger.debug(
                    "Nie udało się skonfigurować pola frakcji na podstawie profilu ryzyka",
                    exc_info=True,
                )
        return max_fraction

    # ------------------------------------------------------------------
    def _apply_fraction_from_settings(self, settings: RiskManagerSettings | None) -> None:
        max_fraction = self._configure_fraction_widget(settings)
        fraction_value = self._compute_fraction_value(settings)
        if max_fraction > 0:
            fraction_value = min(fraction_value, max_fraction)
        if fraction_value <= 0:
            fraction_value = max_fraction if max_fraction > 0 else _DEFAULT_FRACTION
        fraction_var = getattr(self.state, "fraction", None)
        if hasattr(fraction_var, "set"):
            try:
                fraction_var.set(fraction_value)
            except Exception:  # pragma: no cover - defensywne logowanie
                logger.debug(
                    "Nie udało się ustawić frakcji transakcji na podstawie profilu ryzyka",
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    def _update_risk_banner(self) -> None:
        snapshot = snapshot_from_app(self)
        profile_text = build_risk_profile_hint(snapshot) or "Profil ryzyka: —"
        limits_text = build_risk_limits_summary(snapshot) or "Limity ryzyka: —"
        notional_text = self._format_default_notional(self._calculate_default_notional())
        if self.state.risk_profile_label is not None:
            self.state.risk_profile_label.set(profile_text)
        if self.state.risk_limits_label is not None:
            self.state.risk_limits_label.set(limits_text)
        if self.state.risk_notional_label is not None:
            self.state.risk_notional_label.set(notional_text)

    # ------------------------------------------------------------------
    def add_risk_reload_listener(
        self,
        callback: Callable[[str | None, RiskManagerSettings, Any | None], None],
    ) -> None:
        if not callable(callback):
            raise TypeError("Oczekiwano wywoływalnego callbacku")
        self._risk_reload_listeners.append(callback)

    # ------------------------------------------------------------------
    def remove_risk_reload_listener(
        self,
        callback: Callable[[str | None, RiskManagerSettings, Any | None], None],
    ) -> None:
        try:
            self._risk_reload_listeners.remove(callback)
        except ValueError:  # pragma: no cover - defensywne
            logger.debug("Próba usunięcia niezarejestrowanego callbacku", exc_info=True)

    # ------------------------------------------------------------------
    def _notify_risk_reload_listeners(
        self,
        profile_name: str | None,
        settings: RiskManagerSettings,
        profile_payload: Any | None,
    ) -> None:
        listeners: Iterable[
            Callable[[str | None, RiskManagerSettings, Any | None], None]
        ] = tuple(self._risk_reload_listeners)
        for callback in listeners:
            try:
                callback(profile_name, settings, profile_payload)
            except Exception:  # pragma: no cover - defensywne logowanie
                logger.exception("Callback przeładowania profilu ryzyka zgłosił wyjątek")

    # ------------------------------------------------------------------
    def _handle_window_close(self) -> None:
        self._stop_risk_watchdog()
        try:
            self.root.quit()
        except Exception:  # pragma: no cover - defensywne
            logger.debug("Nie udało się zatrzymać pętli głównej Tk", exc_info=True)
        try:
            self.root.destroy()
        except Exception:  # pragma: no cover - defensywne
            logger.debug("Nie udało się zamknąć okna Trading GUI", exc_info=True)

    # ------------------------------------------------------------------
    def _on_root_destroy(self, event: tk.Event) -> None:  # type: ignore[name-defined]
        widget = getattr(event, "widget", None)
        if widget is self.root:
            self._stop_risk_watchdog()

    # ------------------------------------------------------------------
    def _get_risk_config_timestamp(self) -> Optional[float]:
        if not self._core_config_path:
            return None
        try:
            return self._core_config_path.stat().st_mtime
        except FileNotFoundError:
            return None
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.debug(
                "Nie udało się pobrać stempla czasowego konfiguracji core", exc_info=True
            )
            return None

    # ------------------------------------------------------------------
    def _start_risk_watchdog(self) -> None:
        if not self._core_config_path or not hasattr(self.root, "after"):
            return
        self._stop_risk_watchdog()
        try:
            self._risk_watchdog_after = self.root.after(
                self._risk_watch_interval_ms,
                self._risk_watchdog_tick,
            )
        except Exception:  # pragma: no cover - środowiska bez after()
            logger.debug("Nie udało się uruchomić watchdog profilu ryzyka", exc_info=True)

    # ------------------------------------------------------------------
    def _stop_risk_watchdog(self) -> None:
        token = self._risk_watchdog_after
        if token is None:
            return
        try:
            self.root.after_cancel(token)
        except Exception:  # pragma: no cover - środowiska bez cancel
            logger.debug("Nie udało się zatrzymać watchdog profilu ryzyka", exc_info=True)
        finally:
            self._risk_watchdog_after = None

    # ------------------------------------------------------------------
    def _risk_watchdog_tick(self) -> bool:
        self._risk_watchdog_after = None
        changed = self._check_risk_config_change()
        if hasattr(self.root, "after"):
            try:
                self._risk_watchdog_after = self.root.after(
                    self._risk_watch_interval_ms,
                    self._risk_watchdog_tick,
                )
            except Exception:  # pragma: no cover - środowiska bez after()
                logger.debug("Nie udało się ponownie zaplanować watchdog", exc_info=True)
        return changed

    # ------------------------------------------------------------------
    def _check_risk_config_change(self) -> bool:
        new_mtime = self._get_risk_config_timestamp()
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
        except Exception:  # pragma: no cover - defensywne
            logger.exception("Automatyczne przeładowanie profilu ryzyka nie powiodło się")
            return False
        return True

    # ------------------------------------------------------------------
    def get_portfolio_snapshot(self, symbol: str) -> Dict[str, float]:
        """Udostępnia uproszczony stan portfela dla modułów papierowych."""

        entry = self._open_positions.get(symbol.upper()) if symbol else None
        qty = float(entry.get("qty", 0.0)) if isinstance(entry, dict) else 0.0
        entry_price = float(entry.get("entry", 0.0)) if isinstance(entry, dict) else 0.0
        return {
            "portfolio_value": float(self.paper_balance),
            "position": qty,
            "entry": entry_price,
            "symbol": symbol,
        }


class _TkLogHandler(logging.Handler):
    """Logger wysyłający wpisy do panelu tekstowego GUI."""

    def __init__(self, view: TradingView) -> None:
        super().__init__()
        self.view = view

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - UI
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        self.view.append_log(msg)


def main() -> None:
    TradingGUI().run()


__all__ = ["TradingGUI", "TradeExecutor", "main"]
