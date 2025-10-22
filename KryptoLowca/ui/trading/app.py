"""Główne API aplikacji Trading GUI."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import queue
import threading
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Protocol

import tkinter as tk

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
from bot_core.runtime.preset_service import PresetConfigService
from bot_core.security.file_storage import EncryptedFileSecretStorage

from KryptoLowca.logging_utils import (
    DEFAULT_LOG_FILE,
    LOGS_DIR as GLOBAL_LOGS_DIR,
    get_logger,
    setup_app_logging,
)
from KryptoLowca.runtime.bootstrap import FrontendBootstrap, bootstrap_frontend_services
from KryptoLowca.database_manager import DatabaseManager
from KryptoLowca.security_manager import SecurityManager
from KryptoLowca.config_manager import ConfigManager
from KryptoLowca.report_manager import ReportManager
from KryptoLowca.risk_manager import RiskManager
from KryptoLowca.ai_manager import AIManager
from KryptoLowca.exchange_manager import ExchangeManager
from KryptoLowca.exchanges import AdapterError, MarketDataPoller, create_exchange_adapter
from KryptoLowca.exchanges.zonda import ZondaAdapter

from .state import AppState
from .controller import TradingSessionController
from .view import TradingView
from .license_context import LicenseUiContext, build_license_ui_context
from .risk_helpers import (
    RiskSnapshot,
    build_risk_limits_summary,
    build_risk_profile_hint,
    compute_default_notional,
    format_notional,
    snapshot_from_app,
)


_DEFAULT_FRACTION = 0.05
_DEFAULT_MARKET_SYMBOL = "BTC-PLN"
_DEFAULT_MARKET_INTERVAL = 2.0
_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}


def get_default_market_symbol() -> str:
    """Odczytuje domyślny symbol rynku z ENV lub korzysta z wartości wbudowanej."""

    env_symbol = os.getenv("TRADING_GUI_DEFAULT_SYMBOL")
    if env_symbol:
        cleaned = env_symbol.strip().replace("/", "-").upper()
        if cleaned:
            return cleaned
    return _DEFAULT_MARKET_SYMBOL


def get_default_market_interval() -> float:
    """Zwraca odstęp odpytywania REST z ENV (`TRADING_GUI_MARKET_INTERVAL`)."""

    env_interval = os.getenv("TRADING_GUI_MARKET_INTERVAL")
    if not env_interval:
        return _DEFAULT_MARKET_INTERVAL
    try:
        value = float(env_interval)
    except ValueError:
        logging.getLogger(__name__).warning(
            "Nieprawidłowa wartość TRADING_GUI_MARKET_INTERVAL=%s – używam %.2f s",
            env_interval,
            _DEFAULT_MARKET_INTERVAL,
        )
        return _DEFAULT_MARKET_INTERVAL
    if value <= 0:
        logging.getLogger(__name__).warning(
            "TRADING_GUI_MARKET_INTERVAL musi być dodatnie (otrzymano %s) – używam %.2f s",
            env_interval,
            _DEFAULT_MARKET_INTERVAL,
        )
        return _DEFAULT_MARKET_INTERVAL
    return value


def _parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
    return None


def _merge_adapter_kwargs(options: Dict[str, Any], payload: Dict[str, Any]) -> None:
    existing = options.get("adapter_kwargs")
    if existing is None:
        options["adapter_kwargs"] = dict(payload)
        return
    if not isinstance(existing, dict):
        logger.warning(
            "TRADING_GUI_MARKET_ADAPTER_KWARGS musi być słownikiem – pomijam nadpisanie"
        )
        return
    existing.update(payload)


def _resolve_market_adapter_env() -> Optional[tuple[str, Dict[str, Any]]]:
    adapter_name = os.getenv("TRADING_GUI_MARKET_ADAPTER")
    if not adapter_name:
        return None

    options: Dict[str, Any] = {}

    raw_options = os.getenv("TRADING_GUI_MARKET_ADAPTER_OPTIONS")
    if raw_options:
        try:
            parsed = json.loads(raw_options)
        except json.JSONDecodeError:
            logger.warning(
                "Nie udało się sparsować TRADING_GUI_MARKET_ADAPTER_OPTIONS jako JSON"
            )
        else:
            if isinstance(parsed, dict):
                options.update(parsed)
            else:
                logger.warning(
                    "TRADING_GUI_MARKET_ADAPTER_OPTIONS musi być obiektem JSON"
                )

    raw_kwargs = os.getenv("TRADING_GUI_MARKET_ADAPTER_KWARGS")
    if raw_kwargs:
        try:
            parsed_kwargs = json.loads(raw_kwargs)
        except json.JSONDecodeError:
            logger.warning(
                "Nie udało się sparsować TRADING_GUI_MARKET_ADAPTER_KWARGS jako JSON"
            )
        else:
            if isinstance(parsed_kwargs, dict):
                _merge_adapter_kwargs(options, parsed_kwargs)
            else:
                logger.warning(
                    "TRADING_GUI_MARKET_ADAPTER_KWARGS musi być obiektem JSON"
                )

    demo_override = os.getenv("TRADING_GUI_MARKET_DEMO_MODE")
    if demo_override:
        parsed_demo = _parse_bool(demo_override)
        if parsed_demo is not None:
            options["demo_mode"] = parsed_demo

    compliance_ack = os.getenv("TRADING_GUI_MARKET_COMPLIANCE_ACK")
    if compliance_ack:
        parsed_ack = _parse_bool(compliance_ack)
        if parsed_ack is not None:
            options["compliance_ack"] = parsed_ack

    for key in ("demo_mode", "sandbox", "testnet", "compliance_ack"):
        if key in options:
            parsed_value = _parse_bool(options[key])
            if parsed_value is not None:
                options[key] = parsed_value

    adapter_name = adapter_name.strip()
    if not adapter_name:
        return None

    adapter_kwargs = options.get("adapter_kwargs")
    if adapter_kwargs is not None and not isinstance(adapter_kwargs, dict):
        logger.warning("adapter_kwargs musi być słownikiem – pomijam wartości z ENV")
        options.pop("adapter_kwargs", None)

    return adapter_name, options


def normalize_market_symbol(
    symbol: str, *, default: Optional[str] = None
) -> str:
    """Zamienia zapis symbolu na format wymagany przez REST (np. ``BTC/PLN`` → ``BTC-PLN``)."""

    cleaned = (symbol or "").strip().replace("/", "-").upper()
    fallback = default if default is not None else get_default_market_symbol()
    return cleaned or fallback


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", "").strip())
        except ValueError:
            return None
    return None


def extract_market_price(payload: Mapping[str, Any] | None) -> Optional[float]:
    """Wydobywa cenę z odpowiedzi tickera niezależnie od wariantu struktury."""

    if payload is None:
        return None
    if isinstance(payload, Mapping):
        for key in ("last", "rate", "closing_rate", "closingRate", "price", "close", "sell", "buy"):
            if key in payload:
                value = _to_float(payload[key])
                if value is not None:
                    return value
        for nested_key in ("ticker", "data", "stats", "statistics"):
            nested = payload.get(nested_key)
            price = extract_market_price(nested)
            if price is not None:
                return price
        items = payload.get("items")
        if isinstance(items, Iterable):
            for item in items:
                price = extract_market_price(item)
                if price is not None:
                    return price
    elif isinstance(payload, (list, tuple)):
        for entry in payload:
            price = extract_market_price(entry)
            if price is not None:
                return price
    return None


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


logger = logging.getLogger(__name__)


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
        market_data_adapter_factory: Optional[Callable[..., Any]] = None,
        market_data_interval: Optional[float] = None,
    ) -> None:
        setup_app_logging()
        global logger
        logger = get_logger(__name__)

        self.root = root or tk.Tk()
        self.paths = paths or build_desktop_app_paths(
            __file__,
            logs_dir=GLOBAL_LOGS_DIR,
            text_log_file=DEFAULT_LOG_FILE,
        )
        core_config_path = self._resolve_core_config_path()
        if frontend_services is None:
            services = bootstrap_frontend_services(
                paths=self.paths,
                config_path=core_config_path,
            )
        else:
            services = frontend_services
        self.frontend_services = services
        self.market_intel = market_intel or self.frontend_services.market_intel
        self._core_config_path = core_config_path
        self.runtime_metadata = self._load_metadata(self._core_config_path)
        (
            self._risk_profile_name,
            self._risk_profile_config,
            self.risk_manager_settings,
        ) = self._load_risk_profile(self.runtime_metadata, self._core_config_path)
        self._risk_repository_dir = self.paths.logs_dir / "risk_state"
        self._risk_repository_dir.mkdir(parents=True, exist_ok=True)
        self._risk_repository = FileRiskRepository(self._risk_repository_dir)
        self._risk_decision_log = RiskDecisionLog(
            max_entries=500,
            jsonl_path=self.paths.logs_dir / "risk_decisions.jsonl",
        )
        self.risk_manager_config = self._settings_to_adapter_config(self.risk_manager_settings)
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
            on_start=self._handle_view_start,
            on_stop=self._handle_view_stop,
        )
        self._apply_license_restrictions()
        self._configure_fraction_widget(self.risk_manager_settings)
        self._configure_logging_handler()
        self.ex_mgr = self.frontend_services.exchange_manager
        self.network_var = self.state.network
        self.timeframe_var = self.state.timeframe
        self.symbol_var = self.state.symbol
        self.paper_balance = self._parse_float(self.state.paper_balance.get())
        self.account_balance = 0.0
        self._open_positions = self.state.open_positions
        self._view_logs: Dict[str, str] = {}
        self.default_trade_executor = self._default_trade_executor
        self._trade_executor_callable = self._wrap_trade_executor(trade_executor)
        self._market_data_queue: "queue.Queue[tuple[str, Dict[str, Any]]]" = queue.Queue()
        self._market_status_queue: "queue.Queue[str]" = queue.Queue()
        self._market_data_thread: Optional[threading.Thread] = None
        self._market_data_loop: Optional[asyncio.AbstractEventLoop] = None
        self._market_data_stop = threading.Event()
        self._market_data_adapter_factory = (
            market_data_adapter_factory or self._default_market_adapter_factory
        )
        interval = (
            market_data_interval
            if market_data_interval is not None
            else get_default_market_interval()
        )
        self._market_data_interval = max(0.1, float(interval))
        self._market_data_poller: Optional[MarketDataPoller] = None
        self._market_data_adapter: Any = None
        self._symbol_trace = self.symbol_var.trace_add("write", self._on_symbol_changed)
        self._schedule_market_data_drain()
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

    def _settings_to_adapter_config(
        self, settings: RiskManagerSettings | None
    ) -> Dict[str, Any]:
        if not isinstance(settings, RiskManagerSettings):
            return {}

        payload: Dict[str, Any] = dict(settings.to_dict())
        payload.setdefault("max_daily_loss_pct", float(settings.max_daily_loss_pct))
        payload.setdefault(
            "max_drawdown_pct", float(settings.emergency_stop_drawdown)
        )
        payload.setdefault(
            "hard_drawdown_pct", float(settings.emergency_stop_drawdown)
        )
        payload.setdefault("max_positions", int(settings.max_positions))
        payload.setdefault("max_risk_per_trade", float(settings.max_risk_per_trade))
        payload.setdefault("max_portfolio_risk", float(settings.max_portfolio_risk))
        if settings.profile_name:
            payload.setdefault("risk_profile_name", settings.profile_name)
        return payload

    # ------------------------------------------------------------------
    def _default_controller_factory(self, state: AppState) -> TradingSessionController:
        db_manager = DatabaseManager()
        risk_settings = self.state.risk_manager_settings or self.risk_manager_settings
        config_payload = self._settings_to_adapter_config(risk_settings)
        if not config_payload:
            config_payload = dict(self.risk_manager_config or {})

        risk_mode = "paper"
        try:
            network = (state.network.get() if hasattr(state.network, "get") else "testnet")
            risk_mode = "paper" if str(network).lower() != "live" else "live"
        except Exception:
            risk_mode = "paper"

        risk_manager = RiskManager(
            config=config_payload,
            db_manager=db_manager,
            mode=risk_mode,
            profile_name=config_payload.get("risk_profile_name"),
            decision_log=self._risk_decision_log,
            repository=self._risk_repository,
        )

        return TradingSessionController(
            state,
            db_manager,
            SecurityManager(self.paths.keys_file, self.paths.salt_file),
            ConfigManager(self.paths.presets_dir),
            ReportManager(str(self.paths.db_file)),
            risk_manager,
            self._build_ai_manager(),
            exchange_manager=self.frontend_services.exchange_manager,
            market_intel=self.market_intel,
        )

    # ------------------------------------------------------------------
    def _default_market_adapter_factory(self, *, demo_mode: bool) -> ZondaAdapter:
        env_config = _resolve_market_adapter_env()
        if env_config:
            adapter_name, env_options = env_config
            options = dict(env_options)

            adapter_kwargs = options.get("adapter_kwargs")
            if isinstance(adapter_kwargs, dict):
                options["adapter_kwargs"] = dict(adapter_kwargs)

            default_demo = options.get("demo_mode")
            if default_demo is None:
                options["demo_mode"] = demo_mode
                default_demo = demo_mode
            options.setdefault("sandbox", default_demo)
            options.setdefault("testnet", default_demo)

            try:
                return create_exchange_adapter(adapter_name, **options)
            except AdapterError as exc:
                logger.warning(
                    "Nie udało się utworzyć adaptera rynku %s: %s – używam ZondaAdapter",
                    adapter_name,
                    exc,
                )
            except Exception as exc:  # pragma: no cover - defensywnie
                logger.warning(
                    "Błąd podczas inicjalizacji adaptera rynku %s: %s – używam ZondaAdapter",
                    adapter_name,
                    exc,
                )

        return ZondaAdapter(demo_mode=demo_mode)

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
    def _load_license_context(
        self,
    ) -> tuple[Path | None, LicenseCapabilities | None, CapabilityGuard | None, LicenseUiContext]:
        license_path: Path | None = None
        capabilities: LicenseCapabilities | None = None
        guard: CapabilityGuard | None = None
        extra_notice = ""

        public_key = os.environ.get("BOT_CORE_LICENSE_PUBLIC_KEY")
        license_path_value = os.environ.get("BOT_CORE_LICENSE_PATH")

        if not public_key or not license_path_value:
            extra_notice = (
                "Brak skonfigurowanej licencji offline. Skontaktuj się z opiekunem licencji."
            )
            logger.warning(extra_notice)
        else:
            try:
                from bot_core.security.guards import install_capability_guard
                from bot_core.security.license_service import (
                    LicenseService,
                    LicenseServiceError,
                )
            except Exception:
                logger.exception("Nie udało się zaimportować modułów obsługi licencji")
                extra_notice = "Nie udało się zainicjalizować obsługi licencji offline."
            else:
                license_path = Path(license_path_value).expanduser()
                try:
                    service = LicenseService(verify_key_hex=public_key)
                    snapshot = service.load_from_file(license_path)
                except FileNotFoundError:
                    extra_notice = (
                        f"Nie znaleziono pliku licencji: {license_path}. Skontaktuj się z opiekunem licencji."
                    )
                    logger.error(extra_notice)
                except LicenseServiceError as exc:
                    extra_notice = f"Nie udało się zweryfikować licencji offline: {exc}"
                    logger.error(extra_notice)
                except Exception:
                    logger.exception("Nieoczekiwany błąd podczas ładowania licencji offline")
                    extra_notice = (
                        "Wystąpił nieoczekiwany błąd podczas ładowania licencji offline."
                    )
                else:
                    capabilities = snapshot.capabilities
                    guard = install_capability_guard(capabilities)

        context = build_license_ui_context(capabilities)
        if extra_notice:
            context = replace(
                context,
                notice=self._combine_notices(extra_notice, context.notice),
            )
        return license_path, capabilities, guard, context

    # ------------------------------------------------------------------
    @staticmethod
    def _combine_notices(*messages: str) -> str:
        merged: list[str] = []
        for message in messages:
            text = (message or "").strip()
            if not text:
                continue
            if text not in merged:
                merged.append(text)
        return " ".join(merged)

    # ------------------------------------------------------------------
    def _apply_license_restrictions(self) -> None:
        context = getattr(self, "_license_ui_context", None)
        if context is None:
            return

        self.state.capability_guard = self.license_guard
        self.state.license_capabilities = self.license_capabilities
        if self.state.license_summary is not None:
            self.state.license_summary.set(context.summary)
        if self.state.license_notice is not None:
            self.state.license_notice.set(context.notice)
        if hasattr(self.view, "configure_network_options"):
            self.view.configure_network_options(live_enabled=context.live_enabled)
        if hasattr(self.view, "configure_mode_options"):
            self.view.configure_mode_options(futures_enabled=context.futures_enabled)
        if hasattr(self.view, "set_start_enabled"):
            self.view.set_start_enabled(context.auto_trader_enabled)

    # ------------------------------------------------------------------
    def _create_state(self) -> AppState:
        profile_label, limits_label = self._initial_risk_labels()
        fraction_value = self._compute_fraction_value(self.risk_manager_settings)
        notional_label = self._initial_default_notional_label(fraction_value)
        default_symbol = get_default_market_symbol()
        symbol_var = tk.StringVar(value=default_symbol)
        market_symbol_var = tk.StringVar(
            value=normalize_market_symbol(symbol_var.get(), default=default_symbol)
        )
        market_price_var = tk.StringVar(value="—")

        return AppState(
            paths=self.paths,
            runtime_metadata=self.runtime_metadata,
            symbol=symbol_var,
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
            license_capabilities=self.license_capabilities,
            capability_guard=self.license_guard,
            license_summary=license_summary,
            license_notice=license_notice,
            license_path=str(self._license_path) if self._license_path else None,
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
    def _build_secret_storage(self) -> EncryptedFileSecretStorage:
        vault_path = getattr(self.paths, "secret_vault_file", None)
        if vault_path is None:
            vault_path = self.paths.keys_file.with_suffix(".vault")
        passphrase = os.environ.get("DUDZIAN_GUI_SECRET_PASSPHRASE")
        if not passphrase:
            logger.warning(
                "Brak zmiennej DUDZIAN_GUI_SECRET_PASSPHRASE – użyto domyślnego hasła magazynu."
            )
            passphrase = "trading-gui"
        return EncryptedFileSecretStorage(vault_path, passphrase)

    # ------------------------------------------------------------------
    def _build_preset_service(self) -> PresetConfigService:
        core_path = self._core_config_path or resolve_core_config_path()
        try:
            return PresetConfigService(core_path)
        except Exception:  # pragma: no cover - konfiguracja może być niepełna podczas developmentu
            logger.exception("Nie udało się utworzyć serwisu konfiguracji presetów")
            return PresetConfigService(core_path)

    # ------------------------------------------------------------------
    def _configure_logging_handler(self) -> None:
        handler = _TkLogHandler(self.view)
        logging.getLogger().addHandler(handler)

    # ------------------------------------------------------------------
    def _schedule_market_data_drain(self) -> None:
        if not hasattr(self.root, "after"):
            return
        try:
            self.root.after(300, self._drain_market_data_queue)
        except Exception:  # pragma: no cover - środowiska bez after()
            logger.debug("Nie udało się zaplanować odczytu kolejki rynku", exc_info=True)

    # ------------------------------------------------------------------
    def _drain_market_data_queue(self) -> None:
        self._flush_market_status_queue()
        while True:
            try:
                symbol, payload = self._market_data_queue.get_nowait()
            except queue.Empty:
                break
            self._process_market_payload(symbol, payload)
        if hasattr(self.root, "after"):
            try:
                self.root.after(500, self._drain_market_data_queue)
            except Exception:  # pragma: no cover - środowiska bez after()
                logger.debug("Nie udało się ponownie zaplanować odczytu rynku", exc_info=True)

    # ------------------------------------------------------------------
    def _process_market_payload(self, symbol: str, payload: Mapping[str, Any]) -> None:
        normalized = normalize_market_symbol(symbol)
        if self.state.market_symbol is not None:
            self.state.market_symbol.set(normalized)
        price = extract_market_price(payload)
        if price is not None and self.state.market_price is not None:
            self.state.market_price.set(f"{price:,.2f}")
            self._set_status_immediate(f"Ticker {normalized}: {price:,.2f}")
        else:
            self._set_status_immediate(f"Odświeżono dane {normalized}")

    # ------------------------------------------------------------------
    def _on_symbol_changed(self, *_: Any) -> None:
        normalized = normalize_market_symbol(self.symbol_var.get())
        if self.state.market_symbol is not None:
            self.state.market_symbol.set(normalized)
        if self._market_data_thread and self._market_data_thread.is_alive():
            self._restart_market_data()

    # ------------------------------------------------------------------
    def _handle_view_start(self) -> None:
        self._start_market_data()

    # ------------------------------------------------------------------
    def _handle_view_stop(self) -> None:
        self._stop_market_data()

    # ------------------------------------------------------------------
    def _determine_market_symbols(self) -> list[str]:
        symbol = self.state.market_symbol.get() if self.state.market_symbol else ""
        return [normalize_market_symbol(symbol)]

    # ------------------------------------------------------------------
    def _should_use_demo_mode(self) -> bool:
        try:
            network = self.state.network.get()
        except Exception:
            return True
        return str(network or "").strip().lower() != "live"

    # ------------------------------------------------------------------
    def _start_market_data(self) -> None:
        if self._market_data_thread and self._market_data_thread.is_alive():
            return
        symbols = self._determine_market_symbols()
        if not symbols:
            return
        self._market_data_stop.clear()
        demo_mode = self._should_use_demo_mode()
        self._set_status("Łączenie z rynkiem (REST)...")

        def runner() -> None:
            loop = asyncio.new_event_loop()
            self._market_data_loop = loop
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._market_data_worker(symbols, demo_mode))
            finally:
                with contextlib.suppress(Exception):
                    loop.run_until_complete(loop.shutdown_asyncgens())
                asyncio.set_event_loop(None)
                loop.close()
                self._market_data_loop = None

        self._market_data_thread = threading.Thread(
            target=runner, name="market-data-poller", daemon=True
        )
        self._market_data_thread.start()

    # ------------------------------------------------------------------
    def _restart_market_data(self) -> None:
        self._stop_market_data()
        if hasattr(self.root, "after"):
            try:
                self.root.after(200, self._start_market_data)
                return
            except Exception:  # pragma: no cover - środowiska bez after()
                logger.debug("Nie udało się przeplanować restartu pollera", exc_info=True)
        self._start_market_data()

    # ------------------------------------------------------------------
    def _stop_market_data(self) -> None:
        self._market_data_stop.set()
        loop = self._market_data_loop
        if loop is not None:
            if self._market_data_poller is not None:
                future = asyncio.run_coroutine_threadsafe(
                    self._market_data_poller.stop(), loop
                )
                with contextlib.suppress(Exception):
                    future.result(timeout=1.0)
            if self._market_data_adapter is not None:
                future = asyncio.run_coroutine_threadsafe(
                    self._market_data_adapter.close(), loop
                )
                with contextlib.suppress(Exception):
                    future.result(timeout=1.0)
            with contextlib.suppress(Exception):
                loop.call_soon_threadsafe(lambda: None)
        thread = self._market_data_thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        self._market_data_thread = None
        self._market_data_loop = None
        self._market_data_poller = None
        self._market_data_adapter = None
        self._market_data_stop.clear()
        if self.state.market_price is not None:
            self.state.market_price.set("—")
        self._set_status("Ticker zatrzymany")

    # ------------------------------------------------------------------
    async def _market_data_worker(self, symbols: list[str], demo_mode: bool) -> None:
        try:
            adapter = self._market_data_adapter_factory(demo_mode=demo_mode)
        except Exception as exc:
            logger.warning("Nie udało się utworzyć adaptera rynku: %s", exc)
            self._set_status("Błąd uruchamiania adaptera rynku")
            return
        self._market_data_adapter = adapter
        try:
            await adapter.connect()
        except Exception as exc:
            logger.warning("Połączenie adaptera rynku nie powiodło się: %s", exc)
            self._set_status("Nie udało się połączyć z rynkiem (REST)")
            try:
                await adapter.close()
            except Exception:  # pragma: no cover - najlepszy wysiłek podczas sprzątania
                logger.debug(
                    "Nie udało się zamknąć adaptera rynku po błędzie połączenia",
                    exc_info=True,
                )
            finally:
                self._market_data_adapter = None
            return
        poller = MarketDataPoller(
            adapter,
            symbols=symbols,
            interval=self._market_data_interval,
            callback=self._market_data_callback,
            error_callback=self._market_data_error,
        )
        self._market_data_poller = poller
        try:
            await poller.start()
            self._set_status("Ticker REST aktywny")
            while not self._market_data_stop.is_set():
                await asyncio.sleep(0.2)
        finally:
            with contextlib.suppress(Exception):
                await poller.stop()
            with contextlib.suppress(Exception):
                await adapter.close()
            self._market_data_poller = None
            self._market_data_adapter = None

    # ------------------------------------------------------------------
    async def _market_data_callback(self, symbol: str, payload: Dict[str, Any]) -> None:
        self._market_data_queue.put((symbol, payload))

    # ------------------------------------------------------------------
    async def _market_data_error(self, symbol: str, exc: Exception) -> None:
        normalized = normalize_market_symbol(symbol)
        logger.warning("Błąd REST tickera %s: %s", normalized, exc)
        self._set_status(f"Błąd REST tickera {normalized}: {exc}")

    # ------------------------------------------------------------------
    def _flush_market_status_queue(self) -> None:
        while True:
            try:
                message = self._market_status_queue.get_nowait()
            except queue.Empty:
                break
            self._set_status_immediate(message)

    # ------------------------------------------------------------------
    def _set_status(self, message: str) -> None:
        if threading.current_thread() is threading.main_thread():
            self._set_status_immediate(message)
        else:
            self._market_status_queue.put(message)

    # ------------------------------------------------------------------
    def _set_status_immediate(self, message: str) -> None:
        status_var = getattr(self.state, "status", None)
        if status_var is None:
            return
        try:
            status_var.set(message)
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.debug("Nie udało się ustawić statusu GUI", exc_info=True)

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
        self.risk_manager_config = self._settings_to_adapter_config(settings)
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
        self._stop_market_data()
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
            self._stop_market_data()

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


__all__ = [
    "TradingGUI",
    "TradeExecutor",
    "normalize_market_symbol",
    "get_default_market_symbol",
    "get_default_market_interval",
    "extract_market_price",
    "main",
]
