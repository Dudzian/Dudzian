"""Logika kontrolera sesji tradingowej."""

from __future__ import annotations

import logging
import os
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from tkinter import messagebox

try:  # pragma: no cover - zależność opcjonalna
    from bot_core.market_intel import MarketIntelAggregator
except Exception:  # pragma: no cover - fallback gdy moduł niedostępny
    MarketIntelAggregator = None  # type: ignore[assignment]

from KryptoLowca.database_manager import DatabaseManager
from KryptoLowca.managers.security_manager import SecurityManager
from KryptoLowca.managers.config_manager import ConfigManager
from KryptoLowca.managers.exchange_manager import ExchangeManager
from KryptoLowca.managers.report_manager import ReportManager
from KryptoLowca.managers.risk_manager_adapter import RiskManager
from bot_core.runtime.metadata import RiskManagerSettings
from KryptoLowca.managers.ai_manager import AIManager
from KryptoLowca.core.trading_engine import TradingEngine
from KryptoLowca.runtime.bootstrap import bootstrap_exchange_manager

from .state import AppState


logger = logging.getLogger(__name__)


class TradingSessionController:
    """Skapsułkowana logika zarządzania sesją tradingową."""

    def __init__(
        self,
        state: AppState,
        db: DatabaseManager,
        security: SecurityManager,
        config: ConfigManager,
        reporter: ReportManager,
        risk: RiskManager,
        ai_manager: AIManager,
        *,
        exchange_manager: Optional[ExchangeManager] = None,
        market_intel: Optional["MarketIntelAggregator"] = None,
    ) -> None:
        self.state = state
        self.db = db
        self.security = security
        self.config = config
        self.reporter = reporter
        self.risk = risk
        self.ai_manager = ai_manager
        self.exchange: Optional[ExchangeManager] = exchange_manager
        self.market_intel: Optional["MarketIntelAggregator"] = market_intel
        self.engine = TradingEngine(market_intel=market_intel)
        if hasattr(self.engine, "set_market_intel"):
            try:
                self.engine.set_market_intel(market_intel)
            except Exception:  # pragma: no cover - defensywne logowanie
                logger.debug("Nie udało się ustawić MarketIntelAggregator w TradingEngine", exc_info=True)
        initial_destination = self._parse_optional_path(
            getattr(self.state, "market_intel_history_destination", None)
        )
        if initial_destination is not None and self._is_default_history_path(initial_destination):
            initial_destination = None
        self._remember_history_destination(initial_destination)
        self._attach_engine_callbacks()
        self._autoload_market_intel_history()

    # ------------------------------------------------------------------
    def _attach_engine_callbacks(self) -> None:
        if hasattr(self.engine, "set_report_manager"):
            try:
                self.engine.set_report_manager(self.reporter)
            except Exception:  # pragma: no cover - defensywne
                logger.exception("Nie udało się przypisać reportera do silnika")
        if hasattr(self.engine, "on_event"):
            try:
                self.engine.on_event(self._handle_engine_event)
            except Exception:  # pragma: no cover - defensywne
                logger.exception("Nie udało się podpiąć handlera zdarzeń")

    # ------------------------------------------------------------------
    def _handle_engine_event(self, event: Dict[str, Any]) -> None:
        """Reaguj na zdarzenia TradingEngine."""

        if not event:
            return
        event_type = event.get("type", "unknown")
        logger.info("Zdarzenie TradingEngine: %s", event)
        if event_type == "position_opened":
            symbol = event.get("symbol", "?")
            self.state.open_positions[symbol] = event
        elif event_type == "position_closed":
            symbol = event.get("symbol")
            if symbol and symbol in self.state.open_positions:
                self.state.open_positions.pop(symbol, None)
        elif event_type == "plan_created":
            plan = event.get("plan") or {}
            intel_payload = plan.get("market_intel") if isinstance(plan, dict) else None
            summary = self._format_market_intel(intel_payload)
            self._update_market_intel_summary(summary)
        self.state.status.set(f"Odebrano zdarzenie: {event_type}")

    # ------------------------------------------------------------------
    def clear_market_intel_history(self) -> None:
        """Czyści historię market intel w stanie i, jeśli to możliwe, w GUI."""

        self._sync_market_intel_history([])
        logger.info("Historia market intel została wyczyszczona")

    # ------------------------------------------------------------------
    def get_market_intel_history_text(self) -> str:
        """Zwraca historię market intel jako tekst gotowy do skopiowania."""

        history = getattr(self.state, "market_intel_history", None)
        if not isinstance(history, list) or not history:
            return "Brak historii market intel"

        normalised: list[str] = []
        for entry in history:
            if isinstance(entry, str):
                normalised.append(entry)
            else:
                normalised.append(str(entry))
        return "\n".join(normalised)

    # ------------------------------------------------------------------
    def export_market_intel_history(
        self,
        destination: Optional[Path | str] = None,
        *,
        silent: bool = False,
    ) -> Path:
        """Zapisuje historię market intel do pliku i zwraca docelową ścieżkę."""

        history_text = self.get_market_intel_history_text()
        target_path = self._resolve_history_export_path(destination)
        custom_destination = False
        if destination is not None:
            custom_destination = not self._is_default_history_path(target_path)
        else:
            custom_destination = self._has_custom_history_destination()
        self._remember_history_destination(target_path if custom_destination else None)

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if not history_text.endswith("\n"):
                history_text += "\n"
            target_path.write_text(history_text, encoding="utf-8")
        except Exception as exc:
            logger.exception("Nie udało się zapisać historii market intel do pliku")
            raise RuntimeError("Nie udało się zapisać historii market intel") from exc

        message = f"Zapisano historię market intel do {target_path}"
        if silent:
            logger.debug(message)
        else:
            self._set_status(message)
            logger.info(message)
        return target_path

    # ------------------------------------------------------------------
    def load_market_intel_history(self, source: Optional[Path | str] = None) -> list[str]:
        """Wczytuje historię market intel z pliku i aktualizuje stan GUI."""

        target_path = self._resolve_history_export_path(source)
        custom_destination = False
        if source is not None:
            custom_destination = not self._is_default_history_path(target_path)
        else:
            custom_destination = self._has_custom_history_destination()
        self._remember_history_destination(target_path if custom_destination else None)
        try:
            content = target_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            message = f"Brak zapisanej historii market intel w {target_path}"
            logger.info(message)
            self._set_status("Brak zapisanej historii market intel")
            self._sync_market_intel_history([])
            return []
        except Exception as exc:
            logger.exception("Nie udało się wczytać historii market intel z pliku")
            raise RuntimeError("Nie udało się wczytać historii market intel") from exc

        entries = [line.strip() for line in content.splitlines() if line.strip()]
        if not entries:
            self._sync_market_intel_history([])
            self._set_status("Wczytano pustą historię market intel")
            return []

        history = entries[-10:]
        self._sync_market_intel_history(history)
        summary_entry = history[-1]
        summary_text = self._extract_summary_from_entry(summary_entry)
        self._apply_market_intel_summary(summary_text)

        message = f"Wczytano {len(history)} wpisów historii market intel z {target_path}"
        logger.info(message)
        self._set_status(message)
        return history

    # ------------------------------------------------------------------
    def reveal_market_intel_history(
        self,
        *,
        opener: Optional[Callable[[Path], bool]] = None,
    ) -> Path:
        """Otwiera plik historii market intel w zewnętrznej aplikacji."""

        try:
            target_path = self._resolve_history_export_path(None)
        except RuntimeError as exc:
            logger.info("Nie można ustalić domyślnej ścieżki historii market intel")
            self._set_status("Nie udało się otworzyć pliku historii market intel")
            raise RuntimeError("Nie udało się otworzyć pliku historii market intel") from exc

        if not target_path.exists():
            message = f"Brak zapisanej historii market intel w {target_path}"
            logger.info(message)
            self._set_status("Brak zapisanej historii market intel")
            raise FileNotFoundError(message)

        def _default_opener(path: Path) -> bool:
            try:
                return bool(webbrowser.open(path.as_uri()))
            except Exception:  # pragma: no cover - defensywne logowanie
                logger.debug("Domyślny opener nie otworzył pliku historii", exc_info=True)
                return False

        open_callable = opener or _default_opener

        try:
            opened = bool(open_callable(target_path))
        except Exception as exc:
            logger.exception("Nie udało się otworzyć pliku historii market intel")
            self._set_status("Nie udało się otworzyć pliku historii market intel")
            raise RuntimeError("Nie udało się otworzyć pliku historii market intel") from exc

        if not opened:
            logger.info("Opener zwrócił False przy otwieraniu %s", target_path)
            self._set_status("Nie udało się otworzyć pliku historii market intel")
            raise RuntimeError("Nie udało się otworzyć pliku historii market intel")

        message = f"Otwarto plik historii market intel: {target_path}"
        logger.info(message)
        self._set_status(message)
        return target_path

    # ------------------------------------------------------------------
    def set_market_intel_auto_save(self, enabled: bool) -> None:
        """Włącza lub wyłącza automatyczny zapis historii market intel."""

        var = getattr(self.state, "market_intel_auto_save", None)
        if hasattr(var, "set"):
            try:
                var.set(bool(enabled))
            except Exception:  # pragma: no cover - defensywne logowanie
                logger.debug(
                    "Nie udało się zaktualizować market_intel_auto_save", exc_info=True
                )
        else:
            setattr(self.state, "market_intel_auto_save", bool(enabled))

        message = (
            "Auto-zapis historii market intel włączony"
            if enabled
            else "Auto-zapis historii market intel wyłączony"
        )
        self._set_status(message)
        logger.info(message)

        if enabled:
            history = getattr(self.state, "market_intel_history", None) or []
            if history:
                self._auto_save_market_intel_history()

    # ------------------------------------------------------------------
    def set_market_intel_history_destination(
        self, destination: Optional[Path | str]
    ) -> Optional[Path]:
        """Aktualizuje ścieżkę pliku historii market intel wykorzystywaną przy eksporcie."""

        if destination in (None, "", False):
            self._remember_history_destination(None)
            message = "Przywrócono domyślny plik historii market intel"
            self._set_status(message)
            logger.info(message)
            return None

        path = self._parse_optional_path(destination)
        if path is None:
            raise ValueError("Niepoprawna ścieżka docelowa historii market intel")

        self._remember_history_destination(path)
        message = f"Ustawiono plik historii market intel: {path}"
        self._set_status(message)
        logger.info(message)
        return path

    # ------------------------------------------------------------------
    def get_market_intel_history_destination(self) -> Path:
        """Zwraca aktualnie ustawioną ścieżkę pliku historii market intel."""

        stored = self._parse_optional_path(
            getattr(self.state, "market_intel_history_destination", None)
        )
        if stored is not None:
            return stored
        return self._default_history_path()

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.state.running:
            return
        try:
            self.ensure_exchange()
            self.state.running = True
            self.state.status.set("Sesja handlowa uruchomiona")
            logger.info("Trading session started")
        except Exception as exc:  # pragma: no cover - reakcja UI
            logger.exception("Nie udało się uruchomić sesji")
            self.state.status.set("Błąd przy uruchamianiu sesji")
            messagebox.showerror("Trading GUI", str(exc))

    # ------------------------------------------------------------------
    def stop(self) -> None:
        if not self.state.running:
            return
        self.state.running = False
        self.state.status.set("Sesja zatrzymana")
        logger.info("Trading session stopped")

    # ------------------------------------------------------------------
    def ensure_exchange(self) -> ExchangeManager:
        """Zapewnia instancję ``ExchangeManager`` zgodną z aktualnym stanem GUI."""

        if self.exchange is None:
            logger.debug("Brak instancji ExchangeManager – tworzę nową")
            paths = getattr(self.state, "paths", None)
            manager = bootstrap_exchange_manager(paths=paths)
            if manager is None:
                raise RuntimeError("ExchangeManager nie jest dostępny w tej dystrybucji")
            self._configure_exchange_manager(manager)
            self.exchange = manager
        else:
            logger.debug("Ponownie wykorzystuję wstrzyknięty ExchangeManager")
            self._configure_exchange_manager(self.exchange)
        return self.exchange

    # ------------------------------------------------------------------
    def _configure_exchange_manager(self, manager: ExchangeManager) -> None:
        """Ustawia tryb działania i dane logowania ExchangeManagera."""

        network = self._safe_get(getattr(self.state, "network", None), default="testnet").lower()
        mode = self._safe_get(getattr(self.state, "mode", None), default="spot").lower()
        futures = mode == "futures"

        if network == "live":
            if self._configure_live_credentials(manager, futures=futures):
                return
            logger.warning(
                "Nie znaleziono kluczy LIVE – przełączam ExchangeManager w tryb papierowy"
            )
        elif network in {"testnet", "demo"}:
            if self._configure_testnet_credentials(manager, futures=futures):
                return
            logger.info(
                "Brak kluczy TESTNET – korzystam z trybu papierowego ExchangeManagera"
            )

        self._configure_paper_mode(manager)

    # ------------------------------------------------------------------
    def _configure_live_credentials(
        self, manager: ExchangeManager, *, futures: bool
    ) -> bool:
        api_key, api_secret = self._resolve_api_credentials("live")
        if not api_key or not api_secret:
            return False
        manager.set_mode(futures=futures, spot=not futures, testnet=False)
        manager.set_credentials(api_key, api_secret)
        logger.info("ExchangeManager skonfigurowany dla trybu LIVE (%s)", "futures" if futures else "spot")
        return True

    # ------------------------------------------------------------------
    def _configure_testnet_credentials(
        self, manager: ExchangeManager, *, futures: bool
    ) -> bool:
        api_key, api_secret = self._resolve_api_credentials("testnet")
        if not api_key or not api_secret:
            return False
        manager.set_mode(futures=futures, spot=not futures, testnet=True)
        manager.set_credentials(api_key, api_secret)
        logger.info(
            "ExchangeManager skonfigurowany dla trybu TESTNET (%s)",
            "futures" if futures else "spot",
        )
        return True

    # ------------------------------------------------------------------
    def _configure_paper_mode(self, manager: ExchangeManager) -> None:
        manager.set_mode(paper=True)
        balance = self._resolve_paper_balance()
        manager.set_paper_balance(balance, asset="USDT")
        logger.info("ExchangeManager działa w trybie papierowym z saldem %.2f USDT", balance)

    # ------------------------------------------------------------------
    def _resolve_paper_balance(self) -> float:
        raw = self._safe_get(getattr(self.state, "paper_balance", None), default="10000")
        try:
            normalised = raw.replace(" ", "").replace(",", "")
            return max(0.0, float(normalised))
        except Exception:
            return 10_000.0

    # ------------------------------------------------------------------
    def _resolve_api_credentials(self, variant: str) -> Tuple[Optional[str], Optional[str]]:
        loader = getattr(self.security, "load_encrypted_keys", None)
        if not callable(loader):
            return (None, None)

        password = self._resolve_credentials_password()
        if not password:
            return (None, None)
        try:
            payload = loader(password) or {}
        except Exception:
            logger.exception("Nie udało się wczytać kluczy API dla wariantu %s", variant)
            return (None, None)

        key = payload.get(f"{variant}_key") or payload.get(f"{variant}_api_key")
        secret = payload.get(f"{variant}_secret") or payload.get(f"{variant}_api_secret")
        return (key or None, secret or None)

    # ------------------------------------------------------------------
    def _update_market_intel_summary(self, summary: str) -> None:
        self._apply_market_intel_summary(summary)
        self._record_market_intel_history(summary)

    # ------------------------------------------------------------------
    def _record_market_intel_history(self, summary: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        entry = f"{timestamp} | {summary}"
        history = getattr(self.state, "market_intel_history", None)
        if not isinstance(history, list):
            history = []
        history.append(entry)
        max_length = 10
        if len(history) > max_length:
            del history[: len(history) - max_length]

        self._sync_market_intel_history(history)
        self._auto_save_market_intel_history()

    # ------------------------------------------------------------------
    @staticmethod
    def _format_market_intel(payload: Optional[Dict[str, Any]]) -> str:
        prefix = "Market intel: "
        if not payload:
            return prefix + "—"

        def _coerce_float(value: Any) -> Optional[float]:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _format_price(value: Optional[float]) -> Optional[str]:
            if value is None or value <= 0:
                return None
            if value >= 1000:
                return f"price≈{value:,.2f}"  # noqa: P101 - przecinki zwiększają czytelność
            if value >= 10:
                return f"price≈{value:,.2f}"
            return f"price≈{value:,.4f}"

        def _format_liquidity(value: Optional[float]) -> Optional[str]:
            if value is None or value <= 0:
                return None
            abs_value = abs(value)
            if abs_value >= 1_000_000_000:
                return f"liq≈{value / 1_000_000_000:.2f}B USD"
            if abs_value >= 1_000_000:
                return f"liq≈{value / 1_000_000:.2f}M USD"
            if abs_value >= 1_000:
                return f"liq≈{value / 1_000:.2f}K USD"
            if abs_value >= 100:
                return f"liq≈{value:,.0f} USD"
            return f"liq≈{value:.2f} USD"

        def _format_momentum(value: Optional[float]) -> Optional[str]:
            if value is None:
                return None
            return f"mom≈{value:+.2f}"

        def _format_volatility(value: Optional[float]) -> Optional[str]:
            if value is None:
                return None
            return f"vol≈{value:.2f}%"

        price_keys = ("mid_price", "price", "last", "close")
        liquidity_keys = ("liquidity_usd", "liquidity", "notional_usd")
        momentum_keys = ("momentum_score", "momentum", "momentum_zscore")
        volatility_keys = ("volatility_pct", "volatility", "atr_pct")

        price = next((payload.get(key) for key in price_keys if key in payload), None)
        liquidity = next((payload.get(key) for key in liquidity_keys if key in payload), None)
        momentum = next((payload.get(key) for key in momentum_keys if key in payload), None)
        volatility = next((payload.get(key) for key in volatility_keys if key in payload), None)

        parts: list[str] = []
        price_str = _format_price(_coerce_float(price))
        if price_str:
            parts.append(price_str)
        liquidity_str = _format_liquidity(_coerce_float(liquidity))
        if liquidity_str:
            parts.append(liquidity_str)
        momentum_str = _format_momentum(_coerce_float(momentum))
        if momentum_str:
            parts.append(momentum_str)
        volatility_str = _format_volatility(_coerce_float(volatility))
        if volatility_str:
            parts.append(volatility_str)

        if not parts:
            return prefix + "dane dostępne"
        return prefix + ", ".join(parts)

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_credentials_password() -> Optional[str]:
        candidates = (
            os.environ.get("KRYPTLOWCA_KEYS_PASSWORD"),
            os.environ.get("TRADING_GUI_KEYS_PASSWORD"),
        )
        for candidate in candidates:
            if candidate:
                return candidate
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _safe_get(variable: Any, *, default: str) -> str:
        if hasattr(variable, "get"):
            try:
                value = variable.get()
            except Exception:
                return default
            return str(value) if value is not None else default
        if variable is None:
            return default
        return str(variable)

    # ------------------------------------------------------------------
    def _resolve_history_export_path(self, destination: Optional[Path | str]) -> Path:
        parsed = self._parse_optional_path(destination)
        if parsed is not None:
            return parsed

        stored = self._parse_optional_path(
            getattr(self.state, "market_intel_history_destination", None)
        )
        if stored is not None:
            return stored

        return self._default_history_path()

    # ------------------------------------------------------------------
    def _default_history_path(self) -> Path:
        paths = getattr(self.state, "paths", None)
        logs_dir = getattr(paths, "logs_dir", None) if paths is not None else None
        if not logs_dir:
            raise RuntimeError("Brak katalogu logów do zapisania historii market intel")
        return Path(logs_dir) / "market_intel_history.txt"

    # ------------------------------------------------------------------
    def _autoload_market_intel_history(self) -> None:
        try:
            default_path = self._default_history_path()
        except RuntimeError:
            return

        if not default_path.exists():
            return

        try:
            self.load_market_intel_history(default_path)
        except RuntimeError:
            logger.debug("Nie udało się automatycznie wczytać historii market intel", exc_info=True)

    # ------------------------------------------------------------------
    def _apply_market_intel_summary(self, summary: str) -> None:
        setattr(self.state, "market_intel_summary", summary)
        label = getattr(self.state, "market_intel_label", None)
        if label is None:
            return
        setter = getattr(label, "set", None)
        if callable(setter):
            try:
                setter(summary)
            except Exception:  # pragma: no cover - defensywne logowanie
                logger.debug("Nie udało się zaktualizować market_intel_label", exc_info=True)
        else:  # pragma: no cover - fallback dla nietypowych struktur
            try:
                setattr(self.state, "market_intel_label", summary)
            except Exception:
                logger.debug("Nie udało się przypisać market_intel_label", exc_info=True)

    # ------------------------------------------------------------------
    def _sync_market_intel_history(self, history: list[str]) -> None:
        history_copy = list(history)
        setattr(self.state, "market_intel_history", history_copy)
        display = "\n".join(history_copy[-5:]) if history_copy else "Brak historii market intel"
        setattr(self.state, "market_intel_history_display", display)

        label = getattr(self.state, "market_intel_history_label", None)
        setter = getattr(label, "set", None)
        if callable(setter):
            try:
                setter(display)
            except Exception:  # pragma: no cover - defensywne logowanie
                logger.debug("Nie udało się zaktualizować historii market intel", exc_info=True)
        elif label is not None:  # pragma: no cover - fallback gdy brak metody set
            try:
                setattr(self.state, "market_intel_history_label", display)
            except Exception:
                logger.debug(
                    "Nie udało się zaktualizować referencji etykiety historii market intel",
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    def _sync_history_destination_label(self, path: Optional[Path]) -> None:
        display_text = self._format_history_destination(path)
        setattr(self.state, "market_intel_history_destination_display", display_text)

        label = getattr(self.state, "market_intel_history_path_label", None)
        setter = getattr(label, "set", None)
        if callable(setter):
            try:
                setter(display_text)
            except Exception:  # pragma: no cover - defensywne logowanie
                logger.debug(
                    "Nie udało się zaktualizować etykiety ścieżki historii market intel",
                    exc_info=True,
                )
        elif label is not None:  # pragma: no cover - fallback gdy brak metody set
            try:
                setattr(self.state, "market_intel_history_path_label", display_text)
            except Exception:
                logger.debug(
                    "Nie udało się przypisać etykiety ścieżki historii market intel",
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_summary_from_entry(entry: str) -> str:
        if "|" in entry:
            parts = entry.split("|", 1)
            return parts[1].strip()
        return entry.strip()

    # ------------------------------------------------------------------
    def _set_status(self, message: str) -> None:
        status = getattr(self.state, "status", None)
        setter = getattr(status, "set", None)
        if callable(setter):
            try:
                setter(message)
            except Exception:  # pragma: no cover - defensywne logowanie
                logger.debug("Nie udało się zaktualizować statusu GUI", exc_info=True)

    # ------------------------------------------------------------------
    def _should_auto_save_history(self) -> bool:
        toggle = getattr(self.state, "market_intel_auto_save", None)
        if hasattr(toggle, "get"):
            try:
                return bool(toggle.get())
            except Exception:  # pragma: no cover - defensywne logowanie
                logger.debug("Nie udało się pobrać market_intel_auto_save", exc_info=True)
                return False
        if isinstance(toggle, bool):
            return toggle
        return False

    # ------------------------------------------------------------------
    def _auto_save_market_intel_history(self) -> None:
        if not self._should_auto_save_history():
            return

        history = getattr(self.state, "market_intel_history", None) or []
        if not history:
            return

        try:
            self.export_market_intel_history(silent=True)
        except RuntimeError:  # pragma: no cover - brak katalogu logów itp.
            logger.debug(
                "Automatyczny zapis historii market intel nie powiódł się (brak katalogu)",
                exc_info=True,
            )
        except Exception:  # pragma: no cover - defensywnie logujemy
            logger.debug(
                "Nie udało się automatycznie zapisać historii market intel",
                exc_info=True,
            )

    # ------------------------------------------------------------------
    def _remember_history_destination(self, path: Optional[Path]) -> None:
        stored_value = str(path) if path is not None else None
        setattr(self.state, "market_intel_history_destination", stored_value)
        self._sync_history_destination_label(path)

    # ------------------------------------------------------------------
    def _has_custom_history_destination(self) -> bool:
        stored = self._parse_optional_path(
            getattr(self.state, "market_intel_history_destination", None)
        )
        if stored is None:
            return False
        return not self._is_default_history_path(stored)

    # ------------------------------------------------------------------
    def _parse_optional_path(self, value: Optional[Path | str | bool]) -> Optional[Path]:
        if value in (None, "", False):
            return None
        try:
            path = Path(value).expanduser()
        except (TypeError, ValueError, OSError):
            logger.debug("Nie udało się sparsować ścieżki historii market intel", exc_info=True)
            return None
        try:
            return path.resolve()
        except OSError:
            logger.debug("Nie udało się znormalizować ścieżki historii market intel", exc_info=True)
            return path

    # ------------------------------------------------------------------
    def _is_default_history_path(self, path: Path) -> bool:
        try:
            default_path = self._default_history_path()
        except RuntimeError:
            return False
        try:
            return path.resolve() == default_path.resolve()
        except OSError:
            try:
                return path == default_path.resolve()
            except OSError:
                return path == default_path

    # ------------------------------------------------------------------
    def _format_history_destination(self, path: Optional[Path]) -> str:
        if path is not None:
            return f"Plik historii: {path}"
        try:
            default_path = self._default_history_path()
        except RuntimeError:
            return "Plik historii: domyślny (brak katalogu logów)"
        return f"Plik historii: domyślny ({default_path})"

    # ------------------------------------------------------------------
    def get_exchange(self) -> Optional[ExchangeManager]:
        """Zwraca bieżący ``ExchangeManager`` jeśli został już utworzony."""

        return self.exchange

    # ------------------------------------------------------------------
    def update_risk_settings(self, settings: RiskManagerSettings) -> None:
        """Aktualizuje konfigurację adaptera ryzyka dla bieżącej sesji."""

        if not isinstance(settings, RiskManagerSettings):
            raise TypeError("Oczekiwano instancji RiskManagerSettings")

        config = settings.to_dict()
        self.state.risk_manager_settings = settings
        self.state.risk_manager_config = config
        mode = self.state.mode.get() if hasattr(self.state.mode, "get") else "paper"
        self.risk = RiskManager(config=config, db_manager=self.db, mode=str(mode or "paper"))


__all__ = ["TradingSessionController"]
