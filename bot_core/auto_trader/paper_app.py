"""Komponenty launchera AutoTradera w trybie papierowym."""

from __future__ import annotations

import json
import logging
import os
import shlex
import signal
import threading
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Optional

from bot_core.events import (
    DummyMarketFeed,
    DummyMarketFeedConfig,
    EmitterAdapter,
    wire_gui_logs_to_adapter,
)

try:  # pragma: no cover - opcjonalne zależności GUI
    _risk_helpers = import_module("KryptoLowca.ui.trading.risk_helpers")
except ModuleNotFoundError:  # pragma: no cover - fallback dla środowisk testowych

    def apply_runtime_risk_context(*_args: object, **_kwargs: object) -> None:
        return None


    def refresh_runtime_risk_context(*_args: object, **_kwargs: object) -> None:
        return None


else:
    apply_runtime_risk_context = _risk_helpers.apply_runtime_risk_context
    refresh_runtime_risk_context = _risk_helpers.refresh_runtime_risk_context


try:
    _runtime_bootstrap = import_module("KryptoLowca.runtime.bootstrap")
except ModuleNotFoundError as error:  # pragma: no cover - wymagane w środowiskach produkcyjnych

    def bootstrap_frontend_services(*_args: object, **_kwargs: object) -> object:
        raise ModuleNotFoundError(
            "Pakiet KryptoLowca.runtime.bootstrap jest wymagany do uruchomienia GUI"
        ) from error


else:
    bootstrap_frontend_services = _runtime_bootstrap.bootstrap_frontend_services

from .app import AutoTrader
from bot_core.alerts import AlertSeverity, emit_alert
from bot_core.execution import ExecutionService
from bot_core.security.capabilities import LicenseCapabilities
from bot_core.security.guards import (
    CapabilityGuard,
    LicenseCapabilityError,
    get_capability_guard,
    install_capability_guard,
)
from bot_core.security.license_service import LicenseService, LicenseServiceError
from bot_core.runtime.metadata import (
    RiskManagerSettings,
    load_risk_manager_settings,
)
from bot_core.runtime.paths import resolve_core_config_path


logger = logging.getLogger("paper-autotrade")

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

    _balance_listeners: list[Callable[[float], None]] = field(
        default_factory=list, init=False, repr=False
    )

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

    def add_balance_listener(self, callback: Callable[[float], None]) -> None:
        """Rejestruje obserwatora zmian salda symulowanego portfela."""

        if not callable(callback):
            raise TypeError("Oczekiwano wywoływalnego callbacku")
        self._balance_listeners.append(callback)

    def remove_balance_listener(self, callback: Callable[[float], None]) -> None:
        """Usuwa uprzednio zarejestrowanego obserwatora salda."""

        try:
            self._balance_listeners.remove(callback)
        except ValueError:  # pragma: no cover - defensywne logowanie
            logger.debug(
                "Próba usunięcia niezarejestrowanego listenera salda", exc_info=True
            )

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
        self._notify_balance_listeners()

    def _notify_balance_listeners(self) -> None:
        """Powiadamia zarejestrowanych słuchaczy o nowym saldzie."""

        if not self._balance_listeners:
            return
        balance = float(self.paper_balance)
        for callback in tuple(self._balance_listeners):
            try:
                callback(balance)
            except Exception:  # pragma: no cover - defensywne logowanie
                logger.exception("Listener zmian salda headless stuba zgłosił wyjątek")


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
        core_environment: str | None = None,
        risk_profile: str | None = None,
        bootstrap_context: Any | None = None,
        execution_service: ExecutionService | None = None,
        gui: object | None = None,
        headless_stub: HeadlessTradingStub | None = None,
    ) -> None:
        self.symbol = symbol or DEFAULT_SYMBOL
        self.enable_gui = enable_gui
        self.use_dummy_feed = use_dummy_feed
        self.paper_balance = paper_balance

        self.bootstrap_context = bootstrap_context
        self.core_environment = self._infer_environment_hint(
            core_environment,
            bootstrap_context,
        )
        (
            self.license_path,
            self.license_capabilities,
            self.capability_guard,
            self._license_notice,
        ) = self._load_license_context()
        self._reserved_slot_kind: str | None = None
        self._bot_slot_reserved = False
        self._license_slot_kind: str | None = None
        try:
            self._license_slot_kind = self._evaluate_license_requirements()
        except LicenseCapabilityError as exc:
            if self._license_notice:
                logger.warning(self._license_notice)
            self._emit_license_alert(str(exc), capability=exc.capability)
            raise
        if self._license_notice:
            logger.warning(self._license_notice)

        self.core_config_path = self._resolve_core_config_path(core_config_path)
        (
            self.risk_profile_name,
            self.risk_profile_config,
            loaded_settings,
        ) = self._load_risk_settings(risk_profile)
        initial_settings_loaded = loaded_settings is not None
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
        self._listeners: list[
            Callable[[RiskManagerSettings, str | None, object | None], None]
        ] = []
        self._provided_gui = gui
        self.headless_stub = headless_stub
        services = bootstrap_frontend_services(config_path=self.core_config_path)
        self.frontend_services = services
        self.gui, self.symbol_getter = self._build_gui()
        self._sync_headless_stub_settings()

        resolved_execution_service = self._resolve_execution_service(
            bootstrap_context,
            execution_service,
        )
        if resolved_execution_service is None and services.execution_service is not None:
            resolved_execution_service = services.execution_service

        autotrader_kwargs: dict[str, Any] = {"walkforward_interval_s": None}
        if resolved_execution_service is not None:
            autotrader_kwargs["execution_service"] = resolved_execution_service
        if bootstrap_context is not None:
            autotrader_kwargs["bootstrap_context"] = bootstrap_context
        if services.market_intel is not None and "market_intel" not in autotrader_kwargs:
            autotrader_kwargs["market_intel"] = services.market_intel

        self.trader = AutoTrader(
            self.adapter.emitter,
            self.gui,
            self.symbol_getter,
            **autotrader_kwargs,
        )
        self._update_trader_balance(self.paper_balance)
        self.feed = self._build_feed()
        self._stop_event = threading.Event()
        self._stopped = True
        self._risk_watch_stop = threading.Event()
        self._risk_watch_thread: Optional[threading.Thread] = None
        self._risk_watch_interval = 2.0
        initial_mtime = self._get_risk_config_mtime()
        self._risk_config_mtime: int | None = (
            int(initial_mtime) if initial_settings_loaded and initial_mtime is not None else None
        )
        self._watch_thread: Optional[threading.Thread] = None
        self._watch_stop_event: threading.Event | None = self._risk_watch_stop
        self._watch_interval: float = self._risk_watch_interval
        self._watch_last_mtime: int | None = self._risk_config_mtime
        self._last_signature: str | None = None
        self.last_reload_at: datetime | None = None
        self.reload_count: int = 0
        self._update_watch_aliases()
        self._update_bootstrap_context(
            self.risk_manager_settings,
            self.risk_profile_config,
        )

    def _load_license_context(
        self,
    ) -> tuple[Path | None, LicenseCapabilities | None, CapabilityGuard | None, str]:
        guard: CapabilityGuard | None = None
        capabilities: LicenseCapabilities | None = None
        notice = ""
        license_path: Path | None = None

        context = self.bootstrap_context
        if context is not None:
            ctx_guard = getattr(context, "capability_guard", None)
            if isinstance(ctx_guard, CapabilityGuard):
                guard = ctx_guard
                capabilities = ctx_guard.capabilities
            else:
                ctx_capabilities = getattr(context, "license_capabilities", None)
                if isinstance(ctx_capabilities, LicenseCapabilities):
                    capabilities = ctx_capabilities

        if guard is None:
            existing = get_capability_guard()
            if existing is not None:
                guard = existing
                capabilities = existing.capabilities

        if guard is None and capabilities is not None:
            guard = install_capability_guard(capabilities)
        elif guard is not None and capabilities is None:
            capabilities = guard.capabilities

        if guard is None:
            public_key = os.environ.get("BOT_CORE_LICENSE_PUBLIC_KEY")
            license_path_value = os.environ.get("BOT_CORE_LICENSE_PATH")
            if public_key and license_path_value:
                candidate_path = Path(license_path_value).expanduser()
                try:
                    service = LicenseService(verify_key_hex=public_key)
                    snapshot = service.load_from_file(candidate_path)
                except FileNotFoundError:
                    notice = (
                        f"Nie znaleziono pliku licencji: {candidate_path}. Skontaktuj się z opiekunem licencji."
                    )
                except LicenseServiceError as exc:
                    notice = f"Nie udało się zweryfikować licencji offline: {exc}"
                except Exception:
                    logger.exception(
                        "Nieoczekiwany błąd podczas ładowania licencji offline"
                    )
                    notice = (
                        "Wystąpił nieoczekiwany błąd podczas ładowania licencji offline."
                    )
                else:
                    capabilities = snapshot.capabilities
                    guard = install_capability_guard(capabilities)
                    license_path = snapshot.bundle_path
            else:
                notice = (
                    "Brak skonfigurowanej licencji offline. Skontaktuj się z opiekunem licencji."
                )

        if guard is not None and license_path is None:
            path_value = getattr(context, "license_path", None) if context is not None else None
            if isinstance(path_value, (str, Path)):
                license_path = Path(path_value)

        return license_path, capabilities, guard, notice

    def _evaluate_license_requirements(self) -> str:
        guard = self.capability_guard
        capabilities = self.license_capabilities
        if guard is None or capabilities is None:
            raise LicenseCapabilityError(
                "Licencja nie obejmuje modułu AutoTrader. Skontaktuj się z opiekunem licencji.",
                capability="auto_trader",
            )

        guard.require_runtime(
            "auto_trader",
            message="Licencja nie obejmuje modułu AutoTrader. Skontaktuj się z opiekunem licencji.",
        )
        environment_kind = self._determine_environment_kind()
        if environment_kind == "live":
            guard.require_environment(
                "live",
                message="Tryb live wymaga edycji Pro. Skontaktuj się z opiekunem licencji.",
            )
            guard.require_edition(
                "pro",
                message="Tryb live wymaga edycji Pro. Skontaktuj się z opiekunem licencji.",
            )
            slot_kind = "live_controller"
        else:
            aliases = ("paper", "demo", "testnet")
            if not any(capabilities.is_environment_allowed(alias) for alias in aliases):
                raise LicenseCapabilityError(
                    "Licencja nie obejmuje środowiska demo/testnet. Skontaktuj się z opiekunem licencji.",
                    capability="environment",
                )
            slot_kind = "paper_controller"

        if self._requires_futures_module():
            guard.require_module(
                "futures",
                message="Dodaj moduł Futures, aby aktywować handel kontraktami.",
            )

        return slot_kind

    def _determine_environment_kind(self) -> str:
        context = self.bootstrap_context
        if context is not None:
            environment = getattr(context, "environment", None)
            candidate = self._normalize_environment_value(environment)
            if candidate:
                return candidate

        hint = str(self.core_environment or "").lower()
        if "live" in hint:
            return "live"
        if any(alias in hint for alias in ("paper", "demo", "testnet", "sim")):
            return "paper"
        return "paper"

    @staticmethod
    def _normalize_environment_value(environment: object) -> str | None:
        if environment is None:
            return None
        values: list[str] = []
        candidate = getattr(environment, "environment", None)
        if isinstance(candidate, str):
            values.append(candidate)
        elif candidate is not None:
            values.append(str(getattr(candidate, "value", candidate)))
        if isinstance(environment, Mapping):
            raw = environment.get("environment") or environment.get("mode")  # type: ignore[index]
            if isinstance(raw, str):
                values.append(raw)
        for value in values:
            normalized = value.strip().lower()
            if normalized == "live":
                return "live"
            if normalized in {"paper", "demo", "testnet"}:
                return "paper"
        return None

    def _requires_futures_module(self) -> bool:
        context = self.bootstrap_context
        exchange_hint = ""
        if context is not None:
            environment = getattr(context, "environment", None)
            exchange_hint = str(getattr(environment, "exchange", "")) if environment is not None else ""
            if not exchange_hint and isinstance(environment, Mapping):
                raw = environment.get("exchange") or environment.get("adapter")  # type: ignore[index]
                if isinstance(raw, str):
                    exchange_hint = raw
        if not exchange_hint and isinstance(self.core_environment, str):
            exchange_hint = self.core_environment
        normalized = exchange_hint.lower()
        return any(token in normalized for token in ("futures", "perp"))

    def _reserve_license_slots(self) -> None:
        guard = self.capability_guard
        if guard is None or not self._license_slot_kind:
            return
        reserved: list[str] = []
        try:
            guard.reserve_slot(self._license_slot_kind)
            reserved.append(self._license_slot_kind)
            guard.reserve_slot("bot")
            reserved.append("bot")
        except LicenseCapabilityError as exc:
            for kind in reserved:
                with suppress(Exception):
                    guard.release_slot(kind)
            self._emit_license_alert(str(exc), capability=exc.capability)
            raise
        self._reserved_slot_kind = self._license_slot_kind
        self._bot_slot_reserved = True

    def _release_license_slots(self) -> None:
        guard = self.capability_guard
        if guard is None:
            return
        if self._reserved_slot_kind:
            with suppress(Exception):
                guard.release_slot(self._reserved_slot_kind)
            self._reserved_slot_kind = None
        if self._bot_slot_reserved:
            with suppress(Exception):
                guard.release_slot("bot")
            self._bot_slot_reserved = False

    def _emit_license_alert(
        self,
        message: str,
        *,
        capability: str | None = None,
        severity: AlertSeverity = AlertSeverity.ERROR,
    ) -> None:
        logger.warning("Blokada licencyjna AutoTradera: %s", message)
        context = {
            "component": "auto_trader.paper",
            "environment": str(self.core_environment or "unknown"),
            "capability": capability or "unknown",
        }
        try:
            emit_alert(
                message,
                severity=severity,
                source="license_restriction",
                context=context,
            )
        except Exception:
            logger.debug("Nie udało się wysłać alertu licencyjnego", exc_info=True)

    @staticmethod
    def _infer_environment_hint(
        explicit: str | None,
        bootstrap_context: Any | None,
    ) -> str | None:
        if explicit:
            return explicit
        if bootstrap_context is None:
            return None

        candidate = getattr(bootstrap_context, "environment", None)
        if isinstance(candidate, str) and candidate:
            return candidate
        if isinstance(candidate, Mapping):
            for key in ("name", "environment", "id", "slug"):
                value = candidate.get(key)  # type: ignore[index]
                if isinstance(value, str) and value:
                    return value
        elif candidate is not None:
            for attr in ("name", "environment", "id", "slug"):
                value = getattr(candidate, attr, None)
                if isinstance(value, str) and value:
                    return value

        fallback = getattr(bootstrap_context, "environment_name", None)
        if isinstance(fallback, str) and fallback:
            return fallback
        return None

    @staticmethod
    def _resolve_execution_service(
        bootstrap_context: Any | None,
        explicit_service: ExecutionService | None,
    ) -> ExecutionService | None:
        if explicit_service is not None:
            return explicit_service

        if bootstrap_context is None:
            return None

        candidate = getattr(bootstrap_context, "execution_service", None)
        if isinstance(candidate, ExecutionService):
            return candidate

        required = ("execute", "cancel", "flush")
        if candidate is not None and all(
            callable(getattr(candidate, attr, None)) for attr in required
        ):
            return candidate  # type: ignore[return-value]

        return None

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
        bootstrap_payload = self._load_risk_settings_from_bootstrap(profile)
        if bootstrap_payload is not None:
            return bootstrap_payload

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

    def _load_risk_settings_from_bootstrap(
        self, profile: str | None
    ) -> tuple[str | None, object | None, RiskManagerSettings] | None:
        context = getattr(self, "bootstrap_context", None)
        if context is None:
            return None

        candidate = getattr(context, "risk_manager_settings", None)
        if candidate is None:
            return None

        settings = self._coerce_risk_settings(candidate)
        if settings is None:
            return None

        context_profile = getattr(context, "risk_profile_name", None)
        requested = (profile or "").strip().lower() or None
        context_normalized = (
            str(context_profile).strip().lower() if isinstance(context_profile, str) else None
        )
        if requested and context_normalized and requested != context_normalized:
            return None

        effective_profile = context_profile or profile
        profile_payload = getattr(context, "risk_profile_config", None)
        return effective_profile, profile_payload, settings

    def _coerce_risk_settings(
        self, candidate: object
    ) -> RiskManagerSettings | None:
        if isinstance(candidate, RiskManagerSettings):
            return candidate

        defaults = _default_risk_settings()

        def _read_value(*keys: str) -> object | None:
            for key in keys:
                if isinstance(candidate, Mapping) and key in candidate:
                    return candidate[key]
                attr = key.replace("-", "_")
                if hasattr(candidate, attr):
                    return getattr(candidate, attr)
            return None

        def _coerce_float(value: object | None, fallback: float) -> float:
            if value is None:
                return fallback
            try:
                return float(value)
            except Exception:
                return fallback

        def _coerce_int(value: object | None, fallback: int) -> int:
            if value is None:
                return fallback
            try:
                return int(value)
            except Exception:
                return fallback

        max_risk = _coerce_float(
            _read_value("max_risk_per_trade", "max_position_pct", "max_position_notional_pct"),
            defaults.max_risk_per_trade,
        )
        daily_loss = _coerce_float(
            _read_value("max_daily_loss_pct", "max_daily_loss"),
            defaults.max_daily_loss_pct,
        )
        portfolio_risk = _coerce_float(
            _read_value("max_portfolio_risk", "max_portfolio_risk_pct"),
            defaults.max_portfolio_risk,
        )
        max_positions = _coerce_int(_read_value("max_positions"), defaults.max_positions)
        drawdown = _coerce_float(
            _read_value("emergency_stop_drawdown", "max_drawdown_pct"),
            defaults.emergency_stop_drawdown,
        )

        def _coerce_optional_float(value: object | None) -> float | None:
            if value is None:
                return None
            try:
                return float(value)
            except Exception:
                return None

        confidence = _coerce_optional_float(_read_value("confidence_level"))
        target_vol = _coerce_optional_float(_read_value("target_volatility"))
        profile_name = _read_value("profile_name")
        normalized_profile = str(profile_name) if isinstance(profile_name, str) else None

        try:
            return RiskManagerSettings(
                max_risk_per_trade=max_risk,
                max_daily_loss_pct=daily_loss,
                max_portfolio_risk=portfolio_risk,
                max_positions=max_positions,
                emergency_stop_drawdown=drawdown,
                confidence_level=confidence,
                target_volatility=target_vol,
                profile_name=normalized_profile,
            )
        except Exception:
            logger.debug(
                "Nie udało się znormalizować ustawień ryzyka z kontekstu bootstrap", exc_info=True
            )
            return None

    def add_listener(
        self,
        listener: Callable[[RiskManagerSettings, str | None, object | None], None],
    ) -> None:
        """Rejestruje callback informowany o aktualizacji limitów ryzyka."""

        self._listeners.append(listener)

        # Zapewniamy natychmiastową synchronizację nowego słuchacza z aktualnym
        # stanem profilu, aby komponenty dziedziczące z legacy API nie musiały
        # inicjować dodatkowego przeładowania tylko po to, by uzyskać bieżące
        # ustawienia limitów.
        try:
            listener(
                self.risk_manager_settings,
                self.risk_profile_name,
                self.risk_profile_config,
            )
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.exception("Słuchacz aktualizacji ryzyka zgłosił wyjątek przy rejestracji")

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
        self._sync_headless_stub_settings()

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

        self._refresh_runtime_risk_context()

        signature = self._make_signature(
            settings,
            self.risk_profile_name,
            profile_payload,
            environment=self.core_environment,
        )
        changed = signature != self._last_signature
        self._last_signature = signature
        self.reload_count += 1
        self.last_reload_at = datetime.utcnow()

        self.risk_profile_config = profile_payload
        if changed and not notified_via_gui:
            self._notify_trader_of_risk_update(settings, profile_payload)
            self._notify_listeners(settings, profile_payload)
        self._risk_config_mtime = self._get_risk_config_mtime()
        self._update_watch_aliases()
        logger.info("Zaktualizowano profil ryzyka na: %s", self.risk_profile_name)
        self._update_bootstrap_context(settings, profile_payload)
        return settings

    def reload_risk_settings(
        self,
        *,
        config_path: str | Path | None = None,
        environment: str | None = None,
    ) -> tuple[str | None, RiskManagerSettings, object | None]:
        """Zachowuje kompatybilność z historycznym API launchera."""

        if config_path is not None:
            self.core_config_path = self._resolve_core_config_path(config_path)
        if environment is not None:
            self.core_environment = environment

        settings = self.reload_risk_profile(self.risk_profile_name)
        return self.risk_profile_name, settings, self.risk_profile_config

    def _runtime_risk_default_notional(self) -> float:
        """Szacuje domyślny notional na podstawie salda i profilu ryzyka."""

        gui = getattr(self, "gui", None)
        balance_value: float | None = None
        if gui is not None and hasattr(gui, "paper_balance"):
            try:
                balance_attr = getattr(gui, "paper_balance")
                if balance_attr is not None:
                    balance_value = float(balance_attr)
            except Exception:
                logger.debug(
                    "Nie udało się odczytać salda z GUI podczas wyliczania notionala",
                    exc_info=True,
                )

        if balance_value is None:
            try:
                balance_value = float(getattr(self, "paper_balance", 0.0) or 0.0)
            except Exception:
                balance_value = 0.0

        balance = float(balance_value or 0.0)
        if balance > 0:
            self.paper_balance = balance
        if balance <= 0:
            return 0.0

        settings = getattr(self, "risk_manager_settings", None)
        fraction = 0.0
        if isinstance(settings, RiskManagerSettings):
            try:
                fraction = float(settings.max_risk_per_trade)
            except Exception:
                fraction = 0.0

        fraction = max(0.0, fraction)
        if fraction <= 0:
            return 0.0

        return balance * min(fraction, 1.0)

    def _handle_headless_balance_change(self, balance: float) -> None:
        """Synchronizuje saldo aplikacji headless i odświeża domyślny notional."""

        try:
            value = float(balance)
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.debug(
                "Headless stub przekazał niepoprawne saldo: %r", balance, exc_info=True
            )
            return

        self.paper_balance = value
        gui = getattr(self, "gui", None)
        if gui is not None and hasattr(gui, "paper_balance"):
            try:
                setattr(gui, "paper_balance", value)
            except Exception:  # pragma: no cover - defensywne logowanie
                logger.debug(
                    "Nie udało się zsynchronizować salda na GUI headless", exc_info=True
                )

        self._update_trader_balance(value)
        self._refresh_runtime_risk_context()

    def _update_trader_balance(self, balance: float) -> None:
        trader = getattr(self, "trader", None)
        if trader is None:
            return
        update_method = getattr(trader, "update_account_equity", None)
        if not callable(update_method):
            return
        try:
            update_method(balance)
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.debug(
                "Nie udało się zaktualizować salda AutoTradera",
                exc_info=True,
            )

    def _apply_runtime_risk_context(self, gui: object) -> None:
        """Aktualizuje GUI przy pomocy helpera runtime, ignorując błędy."""

        if gui is None:
            return

        try:
            config_path = (
                str(self.core_config_path) if getattr(self, "core_config_path", None) else None
            )
        except Exception:
            config_path = None

        snapshot = None
        try:
            snapshot = apply_runtime_risk_context(
                gui,
                entrypoint="auto_trader",
                config_path=config_path,
                default_notional=self._runtime_risk_default_notional(),
                logger=logger,
            )
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.debug(
                "Nie udało się zastosować runtime risk context dla GUI AutoTradera",
                exc_info=True,
            )
        else:
            self._consume_risk_snapshot(snapshot, target_gui=gui)

    def _refresh_runtime_risk_context(self) -> None:
        gui = self.gui
        if gui is None:
            return
        snapshot = None
        try:
            snapshot = refresh_runtime_risk_context(
                gui,
                default_notional=self._runtime_risk_default_notional(),
                logger=logger,
            )
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.debug(
                "Nie udało się odświeżyć runtime risk context dla GUI AutoTradera",
                exc_info=True,
            )
        else:
            self._consume_risk_snapshot(snapshot)

    def _consume_risk_snapshot(
        self,
        snapshot: object | None,
        *,
        target_gui: object | None = None,
    ) -> None:
        if snapshot is None:
            return

        gui = target_gui if target_gui is not None else getattr(self, "gui", None)

        try:
            balance_value = getattr(snapshot, "paper_balance", None)
        except Exception:
            balance_value = None

        if balance_value is not None:
            try:
                numeric_balance = float(balance_value)
            except Exception:
                numeric_balance = None
            if numeric_balance is not None:
                self.paper_balance = numeric_balance
                if gui is not None and hasattr(gui, "paper_balance"):
                    try:
                        setattr(gui, "paper_balance", numeric_balance)
                    except Exception:  # pragma: no cover - defensywne logowanie
                        logger.debug("GUI nie przyjęło zaktualizowanego salda", exc_info=True)
                self._update_trader_balance(numeric_balance)

        settings = getattr(snapshot, "settings", None)
        if isinstance(settings, RiskManagerSettings):
            self.risk_manager_settings = settings

        profile_name = getattr(snapshot, "profile_name", None)
        if isinstance(profile_name, str) and profile_name:
            self.risk_profile_name = profile_name

    def _build_gui(self) -> tuple[object, Callable[[], str]]:
        provided_gui = getattr(self, "_provided_gui", None)
        provided_stub = self.headless_stub

        if provided_gui is not None:
            self.enable_gui = True
            gui = provided_gui
            if self.frontend_services is not None:
                if hasattr(gui, "frontend_services"):
                    try:
                        gui.frontend_services = self.frontend_services  # type: ignore[attr-defined]
                    except Exception:  # pragma: no cover - defensywne
                        logger.debug("Nie udało się ustawić frontend_services na przekazanym GUI", exc_info=True)
                market_intel = getattr(self.frontend_services, "market_intel", None)
                if (
                    market_intel is not None
                    and getattr(gui, "market_intel", None) is None
                ):
                    try:
                        gui.market_intel = market_intel  # type: ignore[attr-defined]
                    except Exception:  # pragma: no cover - defensywne
                        logger.debug("Nie udało się wstrzyknąć market_intel do przekazanego GUI", exc_info=True)
            self._gui_risk_listener_active = False
            register_listener = getattr(gui, "add_risk_reload_listener", None)
            if callable(register_listener):
                try:
                    register_listener(self._handle_gui_risk_reload)
                except Exception:  # pragma: no cover - defensywne
                    logger.debug("Nie udało się zarejestrować listenera GUI", exc_info=True)
                else:
                    self._gui_risk_listener_active = True
            if hasattr(gui, "paper_balance"):
                try:
                    setattr(gui, "paper_balance", self.paper_balance)
                except Exception:  # pragma: no cover - defensywne
                    logger.debug("Nie udało się ustawić paper_balance na przekazanym GUI", exc_info=True)

            self._apply_runtime_risk_context(gui)

            def getter() -> str:
                symbol_var = getattr(gui, "symbol_var", None)
                if symbol_var is not None and hasattr(symbol_var, "get"):
                    try:
                        value = symbol_var.get()
                        if value:
                            return value
                    except Exception:  # pragma: no cover - defensywne
                        logger.debug("Nie udało się pobrać symbolu z przekazanego GUI", exc_info=True)
                getter_method = getattr(gui, "get_symbol", None)
                if callable(getter_method):
                    try:
                        value = getter_method()
                        if isinstance(value, str) and value:
                            return value
                    except Exception:  # pragma: no cover - defensywne
                        logger.debug("Przekazane GUI zgłosiło wyjątek get_symbol", exc_info=True)
                value = getattr(gui, "symbol", None)
                if isinstance(value, str) and value:
                    return value
                return self.symbol

            return gui, getter

        if self.enable_gui:
            try:
                import tkinter as tk

                TradingGUI = import_module("KryptoLowca.ui.trading").TradingGUI

                root = tk.Tk()
                gui = TradingGUI(root, frontend_services=self.frontend_services)
                gui.paper_balance = self.paper_balance
                register_listener = getattr(gui, "add_risk_reload_listener", None)
                if callable(register_listener):
                    register_listener(self._handle_gui_risk_reload)
                    self._gui_risk_listener_active = True
                try:
                    root.wm_title("KryptoLowca AutoTrader (paper)")
                except Exception:  # pragma: no cover - brak wsparcia tytułów
                    pass

                self._apply_runtime_risk_context(gui)

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

        stub = provided_stub or HeadlessTradingStub(symbol=self.symbol, paper_balance=self.paper_balance)
        if hasattr(stub, "symbol"):
            try:
                setattr(stub, "symbol", self.symbol)
            except Exception:  # pragma: no cover - defensywne
                logger.debug("Nie udało się ustawić symbolu na przekazanym stubie", exc_info=True)
        if hasattr(stub, "paper_balance"):
            try:
                setattr(stub, "paper_balance", self.paper_balance)
            except Exception:  # pragma: no cover - defensywne
                logger.debug("Nie udało się ustawić salda na przekazanym stubie", exc_info=True)
        self._gui_risk_listener_active = False
        self.headless_stub = stub
        initial_settings = self.risk_manager_settings
        try:
            apply_profile = getattr(stub, "apply_risk_profile", None)
            if callable(apply_profile):
                apply_profile(self.risk_profile_name, initial_settings)
            else:
                limits_updater = getattr(stub, "update_risk_limits", None)
                if callable(limits_updater):
                    limits_updater(
                        asdict(initial_settings),
                        profile_name=self.risk_profile_name,
                    )
        except Exception:  # pragma: no cover - defensywne
            logger.debug("Stub headless nie przyjął ustawień startowych", exc_info=True)

        register_balance_listener = getattr(stub, "add_balance_listener", None)
        if callable(register_balance_listener):
            try:
                register_balance_listener(self._handle_headless_balance_change)
            except Exception:  # pragma: no cover - defensywne logowanie
                logger.debug(
                    "Nie udało się podpiąć listenera salda headless stuba", exc_info=True
                )

        previous_gui = getattr(self, "gui", None)
        try:
            self.gui = stub
            self._refresh_runtime_risk_context()
        finally:
            self.gui = previous_gui

        def getter() -> str:
            getter_method = getattr(stub, "get_symbol", None)
            if callable(getter_method):
                try:
                    value = getter_method()
                    if isinstance(value, str) and value:
                        return value
                except Exception:  # pragma: no cover - defensywne
                    logger.debug("Stub headless zgłosił wyjątek get_symbol", exc_info=True)
            value = getattr(stub, "symbol", None)
            if isinstance(value, str) and value:
                return value
            return self.symbol

        return stub, getter

    def _build_feed(self) -> Optional[DummyMarketFeed]:
        if not self.use_dummy_feed:
            return None
        cfg = DummyMarketFeedConfig(symbol=self.symbol.replace("/", ""), start_price=30_000.0, interval_sec=1.0)
        return DummyMarketFeed(self.adapter, cfg=cfg)

    def start(self) -> None:
        if not self._stopped:
            return
        self._reserve_license_slots()
        self._stopped = False
        try:
            self.trader.start()
            if self.feed is not None:
                self.feed.start()
            self._start_risk_watcher()
        except Exception:
            self._stopped = True
            self._release_license_slots()
            raise
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
        self._release_license_slots()
        logger.info("AutoTrader paper app stopped")

    def start_auto_reload(self, interval: float | None = None) -> None:
        """Rozpoczyna monitorowanie pliku konfiguracji ryzyka w tle."""

        if interval is not None:
            try:
                value = float(interval)
            except Exception as exc:  # pragma: no cover - defensywne walidowanie
                raise ValueError("interval must be positive") from exc
            if value <= 0:
                raise ValueError("interval must be positive")
            self._risk_watch_interval = value
        self._start_risk_watcher()
        self._update_watch_aliases()

    def stop_auto_reload(self, timeout: float | None = 1.0) -> None:
        """Zatrzymuje monitorowanie konfiguracji ryzyka."""

        try:
            self._stop_risk_watcher(timeout=timeout)
        except TypeError:
            self._stop_risk_watcher()  # type: ignore[misc]
        self._update_watch_aliases()

    def handle_cli_command(self, command: str) -> bool:
        """Obsługuje uproszczone polecenia CLI kompatybilne z legacy."""

        raw = command.strip()
        if not raw:
            return False
        try:
            parts = shlex.split(raw)
        except ValueError:
            return False
        if not parts:
            return False
        key = parts[0].lower().replace("_", "-")
        if key not in {"reload-risk", "risk-reload", "reload-risk-config"}:
            return False

        env_override: str | None = None
        config_override: str | None = None
        tokens = iter(parts[1:])
        for token in tokens:
            lowered = token.lower()
            if lowered in {"--env", "-e"}:
                env_override = next(tokens, None)
                continue
            if lowered in {"--config", "-c", "--core"}:
                config_override = next(tokens, None)
                continue
            if lowered.startswith("env="):
                env_override = token.split("=", 1)[1] or env_override
                continue
            if lowered.startswith("config=") or lowered.startswith("core="):
                config_override = token.split("=", 1)[1] or config_override
                continue
            if env_override is None and token:
                env_override = token
                continue
            if config_override is None and token:
                config_override = token

        self.reload_risk_settings(
            config_path=config_override,
            environment=env_override,
        )
        return True

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

    def _get_risk_config_mtime(self) -> Optional[int]:
        if not self.core_config_path:
            return None
        try:
            stat_result = Path(self.core_config_path).stat()
        except FileNotFoundError:
            return None
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.debug("Nie udało się pobrać mtime konfiguracji core", exc_info=True)
            return None

        try:
            return int(stat_result.st_mtime_ns)
        except AttributeError:  # pragma: no cover - fallback dla systemów bez ns
            return int(stat_result.st_mtime * 1_000_000_000)

    def _start_risk_watcher(self) -> None:
        if self.core_config_path is None:
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
        self._update_watch_aliases()

    def _stop_risk_watcher(self, *, timeout: float | None = 1.0) -> None:
        self._risk_watch_stop.set()
        thread = self._risk_watch_thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        self._risk_watch_thread = None
        self._risk_watch_stop = threading.Event()
        self._update_watch_aliases()

    def _check_risk_config_change(self) -> bool:
        new_mtime = self._get_risk_config_mtime()
        if new_mtime is None:
            self._risk_config_mtime = None
            self._update_watch_aliases()
            return False
        last_known = self._risk_config_mtime
        if last_known is not None and new_mtime <= last_known:
            return False

        try:
            self.reload_risk_profile()
        except Exception:  # pragma: no cover - diagnostyka runtime
            logger.exception("Automatyczne przeładowanie profilu ryzyka nie powiodło się")
            return False

        if self._risk_config_mtime is None:
            # reload_risk_profile może nie zaktualizować znacznika w przypadku fallbacków
            self._risk_config_mtime = new_mtime
        self._update_watch_aliases()
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
        gui = self.gui
        if gui is not None:
            if hasattr(gui, "risk_manager_settings"):
                try:
                    setattr(gui, "risk_manager_settings", settings)
                except Exception:  # pragma: no cover - defensywne logowanie
                    logger.debug(
                        "GUI nie przyjęło nowych risk_manager_settings podczas reloadu",
                        exc_info=True,
                    )
            if profile_payload is not None and hasattr(gui, "risk_profile_config"):
                try:
                    setattr(gui, "risk_profile_config", profile_payload)
                except Exception:  # pragma: no cover - defensywne logowanie
                    logger.debug(
                        "GUI nie przyjęło nowego payloadu profilu podczas reloadu",
                        exc_info=True,
                    )
        self._notify_trader_of_risk_update(settings, profile_payload)
        self._notify_listeners(settings, profile_payload)
        self._risk_config_mtime = self._get_risk_config_mtime()
        self._sync_headless_stub_settings()
        self._update_bootstrap_context(settings, profile_payload)
        self._refresh_runtime_risk_context()

    def _notify_listeners(
        self,
        settings: RiskManagerSettings,
        profile_payload: object | None,
    ) -> None:
        if not self._listeners:
            return
        for listener in list(self._listeners):
            try:
                listener(settings, self.risk_profile_name, profile_payload)
            except Exception:  # pragma: no cover - defensywne logowanie
                logger.exception("Słuchacz aktualizacji ryzyka zgłosił wyjątek")

    @staticmethod
    def _make_signature(
        settings: RiskManagerSettings,
        profile_name: str | None,
        profile_payload: object | None,
        *,
        environment: str | None,
    ) -> str:
        try:
            settings_payload = json.dumps(asdict(settings), sort_keys=True, default=str)
        except Exception:  # pragma: no cover - defensywne serializowanie
            settings_payload = repr(settings)

        try:
            if isinstance(profile_payload, Mapping):
                payload_repr = json.dumps(profile_payload, sort_keys=True, default=str)
            else:
                payload_repr = json.dumps(profile_payload, sort_keys=True, default=str)
        except Exception:  # pragma: no cover - defensywne serializowanie
            payload_repr = repr(profile_payload)

        return "|".join(
            (
                profile_name or "",
                environment or "",
                settings_payload,
                payload_repr,
            )
        )

    def _update_watch_aliases(self) -> None:
        """Synchronizuje aliasy kompatybilne z legacy API."""

        self._watch_thread = self._risk_watch_thread
        self._watch_stop_event = self._risk_watch_stop
        self._watch_interval = self._risk_watch_interval
        self._watch_last_mtime = self._risk_config_mtime

    def _sync_headless_stub_settings(self) -> None:
        stub = self.headless_stub
        if stub is None or stub is self.gui:
            return
        settings = self.risk_manager_settings
        try:
            apply_profile = getattr(stub, "apply_risk_profile", None)
            if callable(apply_profile):
                apply_profile(self.risk_profile_name, settings)
                return
        except Exception:  # pragma: no cover - defensywne
            logger.exception("Przekazany stub nie przyjął ustawień ryzyka via apply_risk_profile")
            return

        limits_updater = getattr(stub, "update_risk_limits", None)
        if callable(limits_updater):
            try:
                limits_updater(
                    asdict(settings),
                    profile_name=self.risk_profile_name,
                )
            except Exception:  # pragma: no cover - defensywne
                logger.exception("Przekazany stub nie przyjął ustawień ryzyka via update_risk_limits")

    def _update_bootstrap_context(
        self,
        settings: RiskManagerSettings,
        profile_payload: object | None,
    ) -> None:
        if self.bootstrap_context is None:
            return
        try:
            setattr(self.bootstrap_context, "risk_profile_name", self.risk_profile_name)
        except Exception:  # pragma: no cover - kontekst może być tylko-do-odczytu
            logger.debug("Nie udało się ustawić risk_profile_name na BootstrapContext", exc_info=True)
        try:
            setattr(self.bootstrap_context, "risk_profile_config", profile_payload)
        except Exception:  # pragma: no cover - kontekst może być tylko-do-odczytu
            logger.debug("Nie udało się ustawić risk_profile_config na BootstrapContext", exc_info=True)
        try:
            setattr(self.bootstrap_context, "risk_manager_settings", settings)
        except Exception:  # pragma: no cover - kontekst może być tylko-do-odczytu
            logger.debug("Nie udało się ustawić risk_manager_settings na BootstrapContext", exc_info=True)


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
    try:
        app = PaperAutoTradeApp(
            symbol=options.symbol,
            enable_gui=options.enable_gui,
            use_dummy_feed=options.use_dummy_feed,
            paper_balance=options.paper_balance,
            core_config_path=options.core_config_path,
            risk_profile=options.risk_profile,
        )
    except LicenseCapabilityError as exc:
        logger.error("Uruchomienie AutoTradera zablokowane przez licencję: %s", exc)
        raise SystemExit(1) from exc
    try:
        app.run()
    except LicenseCapabilityError as exc:
        logger.error("AutoTrader zatrzymany z powodu ograniczeń licencyjnych: %s", exc)
        raise SystemExit(2) from exc


__all__ = [
    "DEFAULT_SYMBOL",
    "DEFAULT_PAPER_BALANCE",
    "PaperAutoTradeOptions",
    "HeadlessTradingStub",
    "PaperAutoTradeApp",
    "parse_cli_args",
    "main",
]
