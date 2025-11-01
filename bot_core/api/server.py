from __future__ import annotations

import sys

import base64
import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import grpc
import pandas as pd
from google.protobuf import empty_pb2, timestamp_pb2
from google.protobuf.json_format import MessageToDict

from bot_core.alerts import DefaultAlertRouter
from bot_core.alerts.dispatcher import (
    AlertSeverity,
    ensure_offline_logging_sink,
    get_alert_dispatcher,
)
from bot_core.alerts.base import AlertMessage
from bot_core.config.loader import load_core_config, load_runtime_app_config
from bot_core.config.models import CoreConfig, RuntimeAppConfig, RuntimeEntrypointConfig
from bot_core.data.base import OHLCVRequest
from bot_core.data.intervals import interval_to_milliseconds
from bot_core.marketplace import PresetRepository, decode_key_material
from bot_core.execution.base import ExecutionContext, ExecutionService
from bot_core.execution.paper import MarketMetadata, PaperTradingExecutionService
from bot_core.generated import trading_pb2, trading_pb2_grpc
from bot_core.observability import get_global_metrics_registry
from bot_core.observability.exporters import LocalPrometheusExporter
from bot_core.security.base import SecretManager, SecretStorage
from bot_core.exchanges.base import (
    Environment as ExchangeEnvironment,
    ExchangeCredentials,
    OrderRequest,
    AccountSnapshot,
)


def _build_trading_engine_stub() -> ModuleType:
    module = ModuleType("bot_core.trading.engine")

    @dataclass(slots=True)
    class TradingParameters:  # type: ignore[override]
        values: Mapping[str, float] | None = None
        metadata: Mapping[str, Any] | None = None

    module.TradingParameters = TradingParameters
    module.__all__ = ["TradingParameters"]
    return module


def _ensure_trading_engine_available() -> None:
    try:
        import bot_core.trading.engine  # noqa: F401
    except SyntaxError as exc:  # pragma: no cover - degradacja do stubu
        logging.getLogger(__name__).warning(
            "Import trading.engine zakończył się błędem składni: %s", exc
        )
        sys.modules["bot_core.trading.engine"] = _build_trading_engine_stub()
    except Exception as exc:  # pragma: no cover - inne błędy importu
        logging.getLogger(__name__).warning(
            "Import trading.engine zgłosił wyjątek: %s", exc
        )


_ensure_trading_engine_available()


_LOGGER = logging.getLogger(__name__)

from bot_core.runtime.pipeline import (  # noqa: E402
    DailyTrendPipeline,
    build_daily_trend_pipeline,
    create_trading_controller,
)
from bot_core.runtime.realtime import DailyTrendRealtimeRunner  # noqa: E402
from bot_core.runtime.risk_service import (  # noqa: E402
    RiskSnapshotBuilder,
    RiskSnapshotPublisher,
    RiskSnapshotStore,
)
from bot_core.strategies.catalog import DEFAULT_STRATEGY_CATALOG

_ISO_TO_INTERVAL: Mapping[str, str] = {
    "PT1M": "1m",
    "PT3M": "3m",
    "PT5M": "5m",
    "PT15M": "15m",
    "PT30M": "30m",
    "PT1H": "1h",
    "PT2H": "2h",
    "PT4H": "4h",
    "PT6H": "6h",
    "PT8H": "8h",
    "PT12H": "12h",
    "P1D": "1d",
    "P3D": "3d",
    "P1W": "1w",
    "P1M": "1M",
}
_INTERVAL_TO_ISO: Mapping[str, str] = {value: key for key, value in _ISO_TO_INTERVAL.items()}
_DEFAULT_COLUMNS: Sequence[str] = ("timestamp", "open", "high", "low", "close", "volume")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp_from_ms(milliseconds: int) -> timestamp_pb2.Timestamp:
    seconds, remainder = divmod(int(milliseconds), 1000)
    timestamp = timestamp_pb2.Timestamp()
    timestamp.seconds = int(seconds)
    timestamp.nanos = int(remainder) * 1_000_000
    return timestamp


def _ms_from_timestamp(ts: Optional[timestamp_pb2.Timestamp]) -> Optional[int]:
    if ts is None:
        return None
    if ts.seconds == 0 and ts.nanos == 0:
        return None
    return int(ts.seconds) * 1000 + int(ts.nanos) // 1_000_000


def _iso_from_interval(interval: str) -> str:
    normalized = (interval or "").strip()
    if not normalized:
        return "PT1M"
    return _INTERVAL_TO_ISO.get(normalized, "PT1M")


def _interval_from_iso(granularity: Optional[trading_pb2.CandleGranularity], default: str) -> str:
    if granularity is None:
        return default
    iso = getattr(granularity, "iso8601_duration", "")
    normalized = (iso or "").strip() or None
    if normalized is None:
        return default
    return _ISO_TO_INTERVAL.get(normalized, default)


class _InMemorySecretStorage(SecretStorage):
    """Minimalny magazyn sekretów w pamięci na potrzeby lokalnego runtime."""

    def __init__(self) -> None:
        self._storage: Dict[str, str] = {}
        self._lock = threading.Lock()

    def get_secret(self, key: str) -> Optional[str]:
        with self._lock:
            return self._storage.get(key)

    def set_secret(self, key: str, value: str) -> None:
        with self._lock:
            self._storage[key] = value

    def delete_secret(self, key: str) -> None:
        with self._lock:
            self._storage.pop(key, None)


class _ValueHolder:
    def __init__(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        return self._value


class _Emitter:
    """Prosty emitter obsługujący metody wymagane przez AutoTrader."""

    def __init__(self) -> None:
        self._handlers: Dict[str, list[Callable[..., Any]]] = {}
        self._lock = threading.Lock()

    def on(self, event: str, callback: Callable[..., Any], *, tag: str | None = None) -> None:  # noqa: D401
        del tag
        with self._lock:
            self._handlers.setdefault(event, []).append(callback)

    def off(self, event: str, *, tag: str | None = None) -> None:
        del tag
        with self._lock:
            self._handlers.pop(event, None)

    def emit(self, event: str, **payload: Any) -> None:
        with self._lock:
            handlers = list(self._handlers.get(event, ()))
        for handler in handlers:
            try:
                handler(**payload)
            except Exception:  # pragma: no cover - diagnostyczne logowanie
                _LOGGER.exception("Emitter handler for %s failed", event)

    def log(self, message: str, *args: Any, level: int = logging.INFO, **kwargs: Any) -> None:
        numeric_level: int
        if isinstance(level, str):
            candidate = logging.getLevelName(level.upper())
            numeric_level = candidate if isinstance(candidate, int) else logging.INFO
        else:
            numeric_level = int(level)
        extras = kwargs.pop("extra", None)
        if extras is None:
            extras = {}
        else:
            extras = dict(extras)
        component = kwargs.pop("component", None)
        if component is not None:
            extras.setdefault("component", component)
        _LOGGER.log(numeric_level, message, *args, extra=extras or None, **kwargs)


class DefaultAlertRouterStub:
    """Minimalny router alertów używany, gdy bootstrap nie dostarczył instancji."""

    def register(self, channel: Any) -> None:  # pragma: no cover - brak kanałów w stubie
        del channel

    def dispatch(self, message: AlertMessage) -> None:  # pragma: no cover - stub
        del message


class _GuiStub:
    """Minimalna implementacja interfejsu GUI wykorzystywana przez AutoTrader."""

    def __init__(self, timeframe: str, ai_manager: Any | None, portfolio_manager: Any | None) -> None:
        self.timeframe_var = _ValueHolder(timeframe)
        self._demo = True
        self.ai_mgr = ai_manager
        self.portfolio_manager = portfolio_manager
        self.portfolio_mgr = portfolio_manager

    def is_demo_mode_active(self) -> bool:
        return self._demo


class _AutoTraderMarketDataProvider:
    """Dostarcza dane OHLCV dla AutoTradera korzystając z lokalnego źródła."""

    def __init__(self, data_source: Any) -> None:
        self._data_source = data_source

    def get_historical(self, symbol: str, timeframe: str, limit: int = 256) -> pd.DataFrame:
        interval = (timeframe or "1h").strip() or "1h"
        try:
            window_ms = interval_to_milliseconds(interval)
        except Exception:
            window_ms = interval_to_milliseconds("1h")
        end_ms = int(time.time() * 1000)
        start_ms = max(0, end_ms - window_ms * max(limit + 5, 10))
        request = OHLCVRequest(symbol=symbol, interval=interval, start=start_ms, end=end_ms, limit=limit)
        response = self._data_source.fetch_ohlcv(request)
        columns = tuple(response.columns or _DEFAULT_COLUMNS)
        if "timestamp" not in {name.lower() for name in columns}:
            columns = tuple(_DEFAULT_COLUMNS)
        frame = pd.DataFrame(response.rows, columns=columns)
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
            frame = frame.set_index("timestamp")
        return frame[[column for column in ("open", "high", "low", "close", "volume") if column in frame.columns]]


class _AutoTraderStub:
    """Minimalny substytut AutoTradera uruchamiający kontroler w tle."""

    def __init__(
        self,
        emitter: Any,
        gui: Any,
        symbol_getter: Callable[[], str],
        *_,
        controller_runner: Any | None = None,
        auto_trade_interval_s: float = 5.0,
        **__,
    ) -> None:
        self.emitter = emitter
        self.gui = gui
        self.symbol_getter = symbol_getter
        self._runner = controller_runner
        self._runner_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self.auto_trade_interval_s = float(auto_trade_interval_s)
        self.enable_auto_trade = True
        self._auto_trade_user_confirmed = False

    def configure_controller_runner(self, runner: Any | None = None, *, factory: Callable[[], Any] | None = None) -> None:
        if runner is not None:
            with self._runner_lock:
                self._runner = runner
        elif factory is not None:
            try:
                candidate = factory()
            except Exception:  # pragma: no cover - diagnostyka fabryki
                _LOGGER.debug("AutoTrader stub factory failed", exc_info=True)
                return
            with self._runner_lock:
                self._runner = candidate

    def start(self) -> None:
        self.enable_auto_trade = True

    def confirm_auto_trade(self, flag: bool) -> None:
        self._auto_trade_user_confirmed = bool(flag)
        if flag:
            self._start_thread()
        else:
            self._stop_thread()

    def stop(self) -> None:
        self._stop_thread()

    def _start_thread(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="autotrader-stub", daemon=True)
        self._thread.start()

    def _stop_thread(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None

    def _run_loop(self) -> None:
        interval = max(0.5, float(self.auto_trade_interval_s))
        while not self._stop_event.is_set():
            with self._runner_lock:
                runner = self._runner
            if runner is not None:
                try:
                    runner.run_once()
                except Exception:  # pragma: no cover - diagnostyka cyklu
                    _LOGGER.debug("AutoTrader stub cycle failed", exc_info=True)
            if self._stop_event.wait(interval):
                break


def _sanitize_core_config(
    core_config: CoreConfig,
    environment_name: str,
    *,
    strategy_override: str | None = None,
    controller_override: str | None = None,
) -> CoreConfig:
    sanitized = deepcopy(core_config)
    environment = sanitized.environments.get(environment_name)
    if environment is None:
        return sanitized

    sanitized.environments = {environment_name: environment}

    risk_profile_name = getattr(environment, "risk_profile", None)
    if risk_profile_name and risk_profile_name in sanitized.risk_profiles:
        sanitized.risk_profiles = {risk_profile_name: sanitized.risk_profiles[risk_profile_name]}
    else:
        sanitized.risk_profiles = dict(sanitized.risk_profiles)

    required_buckets: set[str] = set()
    for profile in sanitized.risk_profiles.values():
        required_buckets.update(getattr(profile, "instrument_buckets", ()) or ())

    default_universe = getattr(environment, "instrument_universe", None)
    required_universes: set[str] = set()
    if default_universe:
        required_universes.add(default_universe)

    for bucket_name in required_buckets:
        bucket = sanitized.instrument_buckets.get(bucket_name)
        if bucket is None:
            continue
        universe = getattr(bucket, "universe", None)
        if universe:
            required_universes.add(universe)

    sanitized.instrument_universes = {
        name: sanitized.instrument_universes[name]
        for name in required_universes
        if name in sanitized.instrument_universes
    }

    sanitized.instrument_buckets = {
        name: bucket
        for name, bucket in sanitized.instrument_buckets.items()
        if name in required_buckets
        or getattr(bucket, "universe", None) in sanitized.instrument_universes
        or getattr(bucket, "universe", None) is None
    }

    adapter_settings = getattr(environment, "adapter_settings", None)
    if isinstance(adapter_settings, MutableMapping):
        paper_settings = adapter_settings.get("paper_trading")
        if isinstance(paper_settings, MutableMapping):
            default_market = paper_settings.get("default_market")
            if isinstance(default_market, MutableMapping):
                default_market["min_notional"] = 0.0
            per_symbol = paper_settings.get("per_symbol")
            if isinstance(per_symbol, MutableMapping):
                for overrides in per_symbol.values():
                    if isinstance(overrides, MutableMapping):
                        overrides["min_notional"] = 0.0

    required_strategies: set[str] = set()
    default_strategy = getattr(environment, "default_strategy", None)
    if default_strategy:
        required_strategies.add(default_strategy)
    if strategy_override:
        required_strategies.add(strategy_override)

    if required_strategies:
        sanitized.strategies = {
            name: sanitized.strategies[name]
            for name in required_strategies
            if name in sanitized.strategies
        }
        sanitized.strategy_definitions = {
            name: definition
            for name, definition in sanitized.strategy_definitions.items()
            if name in required_strategies
        } or dict(sanitized.strategy_definitions)
    else:
        sanitized.strategies = dict(sanitized.strategies)
        sanitized.strategy_definitions = dict(sanitized.strategy_definitions)

    required_controllers: set[str] = set()
    default_controller = getattr(environment, "default_controller", None)
    if default_controller:
        required_controllers.add(default_controller)
    if controller_override:
        required_controllers.add(controller_override)
    if required_controllers:
        sanitized.runtime_controllers = {
            name: sanitized.runtime_controllers[name]
            for name in required_controllers
            if name in sanitized.runtime_controllers
        }
    else:
        sanitized.runtime_controllers = dict(sanitized.runtime_controllers)

    sanitized.metrics_service = None
    sanitized.live_routing = None
    sanitized.coverage_monitoring = None
    if getattr(sanitized, "license", None) is not None:
        try:
            sanitized.license.license_keys_path = None
            sanitized.license.fingerprint_keys_path = None
        except AttributeError:
            sanitized.license = None
    return sanitized


def _detect_version() -> str:
    try:
        import importlib.metadata as metadata  # type: ignore

        return metadata.version("bot-core")
    except Exception:
        pass
    try:
        import tomllib  # type: ignore

        content = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        project = content.get("project")
        if isinstance(project, Mapping):
            version = project.get("version")
            if isinstance(version, str) and version.strip():
                return version.strip()
    except Exception:
        pass
    return os.environ.get("BOTCORE_VERSION", "dev")


def _detect_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


@dataclass(slots=True)
class LocalRuntimeContext:
    """Przechowuje zainicjalizowane komponenty runtime potrzebne serwerowi gRPC."""

    config: RuntimeAppConfig
    entrypoint: RuntimeEntrypointConfig
    config_path: Path
    pipeline: DailyTrendPipeline
    trading_controller: Any
    runner: DailyTrendRealtimeRunner
    auto_trader: Any
    secret_manager: SecretManager
    alert_router: Any | None = None
    risk_store: RiskSnapshotStore | None = None
    risk_builder: RiskSnapshotBuilder | None = None
    risk_publisher: RiskSnapshotPublisher | None = None
    metrics_registry: Any | None = None
    version: str = field(default_factory=_detect_version)
    git_commit: str | None = field(default_factory=_detect_git_commit)
    started_at: datetime = field(default_factory=_now_utc)
    auth_token: str | None = None
    portfolio_snapshot: AccountSnapshot | None = None
    prometheus_exporter: LocalPrometheusExporter | None = None
    alert_sink_token: str | None = None
    marketplace_repository: PresetRepository | None = None
    marketplace_signing_keys: Mapping[str, bytes] = field(default_factory=dict)
    marketplace_allow_unsigned: bool = False
    marketplace_enabled: bool = True
    _started: bool = field(default=False, init=False, repr=False)

    def start(self, *, auto_confirm: bool = True) -> None:
        if self._started:
            return
        if self.prometheus_exporter is not None:
            try:
                self.prometheus_exporter.start()
            except OSError:
                _LOGGER.warning(
                    "Eksporter Prometheus nie został uruchomiony (port zajęty?)"
                )
                self.prometheus_exporter = None
        if self.risk_publisher is not None:
            try:
                snapshots = self.risk_publisher.publish_once()
            except Exception:  # pragma: no cover - diagnostyczne logowanie
                _LOGGER.exception("Initial risk snapshot publication failed")
                snapshots = ()
            if not snapshots and self.risk_store is not None:
                fallback = trading_pb2.RiskState(
                    profile=trading_pb2.RISK_PROFILE_BALANCED,
                    portfolio_value=0.0,
                    current_drawdown=0.0,
                    max_daily_loss=0.0,
                    used_leverage=0.0,
                )
                fallback.generated_at.CopyFrom(_timestamp_from_ms(int(time.time() * 1000)))
                self.risk_store.append(fallback, metadata={"source": "fallback"})
            try:
                self.risk_publisher.start()
            except Exception:  # pragma: no cover - diagnostyczne logowanie
                _LOGGER.exception("Unable to start RiskSnapshotPublisher thread")
        try:
            self.auto_trader.configure_controller_runner(self.runner)
        except Exception:  # pragma: no cover - diagnostyczne logowanie
            _LOGGER.exception("Failed to attach controller runner")
        self.auto_trader.start()
        if auto_confirm or getattr(self.entrypoint, "trusted_auto_confirm", False):
            self.auto_trader.confirm_auto_trade(True)
        self._started = True

    def stop(self) -> None:
        if self.risk_publisher is not None:
            try:
                self.risk_publisher.stop()
            except Exception:  # pragma: no cover
                _LOGGER.debug("RiskSnapshotPublisher stop failed", exc_info=True)
        try:
            self.auto_trader.stop()
        except Exception:  # pragma: no cover
            _LOGGER.debug("AutoTrader stop failed", exc_info=True)
        if self.prometheus_exporter is not None:
            try:
                self.prometheus_exporter.stop()
            except Exception:  # pragma: no cover - defensywne
                _LOGGER.debug(
                    "Błąd podczas zatrzymywania eksportera Prometheus", exc_info=True
                )
        if self.alert_sink_token:
            try:
                get_alert_dispatcher().unregister(self.alert_sink_token)
            except Exception:  # pragma: no cover - defensywne logowanie
                _LOGGER.debug(
                    "Nie udało się wyrejestrować offline alert sink", exc_info=True
                )
            finally:
                self.alert_sink_token = None
        self._started = False

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> "LocalRuntimeContext":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.stop()

    def authorize(self, rpc_context: grpc.ServicerContext | None) -> None:
        if self.auth_token is None or rpc_context is None:
            return
        metadata = {key.lower(): value for key, value in rpc_context.invocation_metadata()}
        token = metadata.get("authorization") or metadata.get("x-local-auth")
        if token:
            lowered = token.strip()
            if lowered.lower().startswith("bearer "):
                lowered = lowered[7:].strip()
            token = lowered
        if token != self.auth_token:
            rpc_context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid auth token")

    def refresh_portfolio(self) -> None:
        loader = getattr(self.pipeline.controller, "account_loader", None)
        if not callable(loader):
            return
        try:
            snapshot = loader()
        except Exception:  # pragma: no cover - diagnostyka portfela
            _LOGGER.debug("Nie udało się odświeżyć stanu portfela", exc_info=True)
            return
        if isinstance(snapshot, AccountSnapshot):
            self.portfolio_snapshot = snapshot

    def list_marketplace_presets(self) -> tuple["PresetDocument", ...]:
        if not self.marketplace_enabled or self.marketplace_repository is None:
            return ()
        try:
            return self.marketplace_repository.load_all(
                signing_keys=self.marketplace_signing_keys
            )
        except Exception:  # pragma: no cover - diagnostyka presetów
            _LOGGER.debug("Nie udało się odczytać presetów Marketplace", exc_info=True)
            return ()

    def reload_marketplace_presets(self) -> tuple["PresetDocument", ...]:
        documents = self.list_marketplace_presets()
        repo = self.marketplace_repository
        if not self.marketplace_enabled or repo is None:
            return documents
        try:
            DEFAULT_STRATEGY_CATALOG.load_presets_from_directory(
                repo.root,
                signing_keys=self.marketplace_signing_keys,
                hwid_provider=None,
            )
        except FileNotFoundError:
            return documents
        except Exception:  # pragma: no cover - diagnostyka presetów
            _LOGGER.debug(
                "Aktualizacja katalogu strategii z presetów Marketplace nie powiodła się",
                exc_info=True,
            )
        return documents

    def import_marketplace_preset(
        self, payload: bytes, filename: str | None = None
    ) -> "PresetDocument":
        if self.marketplace_repository is None:
            raise RuntimeError("Marketplace repository not configured")
        document = self.marketplace_repository.import_payload(
            payload,
            filename=filename,
            signing_keys=self.marketplace_signing_keys,
            require_signature=not self.marketplace_allow_unsigned,
        )
        self.reload_marketplace_presets()
        return document

    def export_marketplace_preset(
        self, preset_id: str, *, format: str = "json"
    ) -> tuple["PresetDocument", bytes]:
        if self.marketplace_repository is None:
            raise RuntimeError("Marketplace repository not configured")
        return self.marketplace_repository.export_preset(
            preset_id,
            format=format,
            signing_keys=self.marketplace_signing_keys,
        )

    def remove_marketplace_preset(self, preset_id: str) -> bool:
        repo = self.marketplace_repository
        if repo is None:
            return False
        removed = repo.remove(preset_id)
        if removed:
            self.reload_marketplace_presets()
        return removed

    def get_marketplace_preset(self, preset_id: str) -> "PresetDocument | None":
        repo = self.marketplace_repository
        if repo is None:
            return None
        try:
            document, _ = repo.export_preset(
                preset_id,
                format="json",
                signing_keys=self.marketplace_signing_keys,
            )
        except FileNotFoundError:
            return None
        except Exception:  # pragma: no cover - diagnostyka presetów
            _LOGGER.debug(
                "Nie udało się odczytać presetu Marketplace %s", preset_id, exc_info=True
            )
            return None
        return document

    def activate_marketplace_preset(self, preset_id: str) -> "PresetDocument | None":
        document = self.get_marketplace_preset(preset_id)
        if document is None:
            return None
        self.reload_marketplace_presets()
        return self.get_marketplace_preset(preset_id)

    def emit_alert(
        self,
        *,
        category: str,
        title: str,
        body: str,
        severity: str = "info",
        context: Mapping[str, str] | None = None,
    ) -> None:
        router = self.alert_router
        if router is None:
            return
        payload = {str(k): str(v) for k, v in (context or {}).items()}
        message = AlertMessage(
            category=category,
            title=title,
            body=body,
            severity=severity,
            context=payload,
        )
        try:
            router.dispatch(message)
        except Exception:  # pragma: no cover - alerty nie powinny blokować runtime
            _LOGGER.debug("Nie udało się wysłać alertu", exc_info=True)

    @property
    def execution_context(self) -> ExecutionContext:
        controller = self.pipeline.controller
        return getattr(controller, "execution_context")

    @property
    def exchange_name(self) -> str:
        return getattr(self.pipeline.bootstrap.environment, "exchange", "").upper()

    @property
    def primary_symbol(self) -> str:
        symbols = getattr(self.pipeline.controller, "symbols", ())
        if not symbols:
            raise RuntimeError("Pipeline controller nie posiada żadnych symboli")
        return str(symbols[0])

    @property
    def metrics_endpoint(self) -> str | None:
        exporter = self.prometheus_exporter
        if exporter is None:
            return None
        return exporter.metrics_url


class _MarketDataServicer(trading_pb2_grpc.MarketDataServiceServicer):
    def __init__(self, context: LocalRuntimeContext) -> None:
        self._context = context
        self._lock = threading.Lock()
        self._synthetic_state: Dict[str, float] = {}

    def _resolve_metadata(self, symbol: str) -> MarketMetadata | None:
        service = getattr(self._context.pipeline, "execution_service", None)
        markets: Mapping[str, MarketMetadata] | None = getattr(service, "_markets", None)
        if isinstance(markets, Mapping):
            return markets.get(symbol)
        return None

    def _build_instrument(self, symbol: str, metadata: MarketMetadata | None) -> trading_pb2.Instrument:
        exchange = self._context.exchange_name or "PAPER"
        base = metadata.base_asset if metadata is not None else symbol.split("/")[0]
        quote = metadata.quote_asset if metadata is not None else symbol.split("/")[-1]
        venue = symbol.replace("/", "").replace("-", "").replace(":", "")
        return trading_pb2.Instrument(
            exchange=exchange,
            symbol=symbol,
            venue_symbol=venue,
            quote_currency=str(quote).upper(),
            base_currency=str(base).upper(),
        )

    def _normalize_rows(
        self,
        rows: Sequence[Sequence[float]],
        symbol: str,
        interval: str,
        limit: int,
        metadata: MarketMetadata | None,
    ) -> Sequence[Sequence[float]]:
        if rows:
            return rows
        last_price = self._synthetic_state.get(symbol, 100.0)
        try:
            step_ms = interval_to_milliseconds(interval)
        except Exception:
            step_ms = interval_to_milliseconds("1m")
        generated: list[list[float]] = []
        current_time = int(time.time() * 1000) - step_ms * limit
        for _ in range(max(limit, 1)):
            current_time += step_ms
            open_price = last_price
            close_price = last_price * 1.0005
            high_price = max(open_price, close_price) * 1.0005
            low_price = min(open_price, close_price) * 0.9995
            volume = 1.0
            generated.append(
                [
                    float(current_time),
                    float(open_price),
                    float(high_price),
                    float(low_price),
                    float(close_price),
                    float(volume),
                ]
            )
            last_price = close_price
        self._synthetic_state[symbol] = last_price
        return generated

    def GetOhlcvHistory(self, request, context):  # noqa: N802
        symbol = request.instrument.symbol or request.instrument.venue_symbol or self._context.primary_symbol
        metadata = self._resolve_metadata(symbol)
        default_interval = getattr(self._context.pipeline.controller, "interval", "1h")
        interval = _interval_from_iso(
            request.granularity if request.HasField("granularity") else None,
            default_interval,
        )
        limit = request.limit or 250
        end_ms = _ms_from_timestamp(request.end_time if request.HasField("end_time") else None)
        if end_ms is None:
            end_ms = int(time.time() * 1000)
        start_ms = _ms_from_timestamp(request.start_time if request.HasField("start_time") else None)
        if start_ms is None:
            try:
                window_ms = interval_to_milliseconds(interval)
            except Exception:
                window_ms = interval_to_milliseconds("1h")
            start_ms = end_ms - window_ms * max(limit, 1)
        request_payload = OHLCVRequest(symbol=symbol, interval=interval, start=int(start_ms), end=int(end_ms), limit=int(limit))
        response = self._context.pipeline.data_source.fetch_ohlcv(request_payload)
        columns = tuple(response.columns or _DEFAULT_COLUMNS)
        indices: MutableMapping[str, int] = {name.lower(): idx for idx, name in enumerate(columns)}
        rows = self._normalize_rows(response.rows, symbol, interval, limit, metadata)
        granularity_iso = _iso_from_interval(interval)
        candles: list[trading_pb2.OhlcvCandle] = []
        sequence = 0
        for row in rows[-limit:]:
            sequence += 1
            timestamp_ms = float(row[indices.get("timestamp", 0)])
            candle = trading_pb2.OhlcvCandle(
                instrument=self._build_instrument(symbol, metadata),
                open=float(row[indices.get("open", 1)]),
                high=float(row[indices.get("high", 2)]),
                low=float(row[indices.get("low", 3)]),
                close=float(row[indices.get("close", 4)]),
                volume=float(row[indices.get("volume", 5)]),
                closed=True,
                sequence=sequence,
            )
            candle.open_time.CopyFrom(_timestamp_from_ms(int(timestamp_ms)))
            candle.granularity.iso8601_duration = granularity_iso
            candles.append(candle)
        return trading_pb2.GetOhlcvHistoryResponse(candles=candles, has_more=False)

    def StreamOhlcv(self, request, context):  # noqa: N802
        history = self.GetOhlcvHistory(request, context)
        if not history.candles:
            return
        snapshot = trading_pb2.StreamOhlcvSnapshot(candles=history.candles)
        update = trading_pb2.StreamOhlcvUpdate(snapshot=snapshot)
        yield update

    def ListTradableInstruments(self, request, context):  # noqa: N802
        requested_exchange = (request.exchange or "").strip().upper()
        context_exchange = (self._context.exchange_name or "PAPER").upper()
        if requested_exchange and requested_exchange != context_exchange:
            return trading_pb2.ListTradableInstrumentsResponse(instruments=[])
        service = getattr(self._context.pipeline, "execution_service", None)
        markets: Mapping[str, MarketMetadata] | None = getattr(service, "_markets", None)
        instruments: list[trading_pb2.TradableInstrumentMetadata] = []
        if isinstance(markets, Mapping):
            for symbol, metadata in markets.items():
                instrument = self._build_instrument(symbol, metadata)
                entry = trading_pb2.TradableInstrumentMetadata(
                    instrument=instrument,
                    price_step=float(metadata.tick_size or 0.0),
                    amount_step=float(metadata.step_size or 0.0),
                    min_notional=float(metadata.min_notional or 0.0),
                    min_amount=float(metadata.min_quantity or 0.0),
                )
                instruments.append(entry)
        return trading_pb2.ListTradableInstrumentsResponse(instruments=instruments)


class _OrderServicer(trading_pb2_grpc.OrderServiceServicer):
    def __init__(self, context: LocalRuntimeContext) -> None:
        self._context = context
        self._lock = threading.Lock()

    def _map_side(self, side: int) -> str:
        return {
            trading_pb2.ORDER_SIDE_BUY: "buy",
            trading_pb2.ORDER_SIDE_SELL: "sell",
        }.get(side, "buy")

    def _map_type(self, order_type: int) -> str:
        return {
            trading_pb2.ORDER_TYPE_MARKET: "market",
            trading_pb2.ORDER_TYPE_LIMIT: "limit",
            trading_pb2.ORDER_TYPE_STOP_MARKET: "stop_market",
            trading_pb2.ORDER_TYPE_STOP_LIMIT: "stop_limit",
        }.get(order_type, "market")

    def _map_tif(self, tif: int) -> str | None:
        return {
            trading_pb2.TIME_IN_FORCE_GTC: "GTC",
            trading_pb2.TIME_IN_FORCE_GTD: "GTD",
            trading_pb2.TIME_IN_FORCE_IOC: "IOC",
            trading_pb2.TIME_IN_FORCE_FOK: "FOK",
        }.get(tif)

    def SubmitOrder(self, request, context):  # noqa: N802
        self._context.authorize(context)
        symbol = request.instrument.symbol or request.instrument.venue_symbol or self._context.primary_symbol
        execution_service: ExecutionService = self._context.pipeline.execution_service
        order_request = OrderRequest(
            symbol=symbol,
            side=self._map_side(request.side),
            quantity=float(request.quantity or 0.0),
            order_type=self._map_type(request.type),
            price=float(request.price) if request.price else None,
            time_in_force=self._map_tif(request.time_in_force),
            client_order_id=request.client_order_id or None,
            metadata={"source": "grpc"},
        )
        context_model = self._context.execution_context
        try:
            with self._lock:
                result = execution_service.execute(order_request, context_model)
        except Exception as exc:
            violation = trading_pb2.OrderConstraintViolation(code="error", message=str(exc))
            self._context.emit_alert(
                category="execution.order",
                title="Zlecenie odrzucone",
                body=str(exc),
                severity="error",
                context={"symbol": symbol, "side": order_request.side},
            )
            return trading_pb2.SubmitOrderResponse(status=trading_pb2.ORDER_STATUS_REJECTED, violations=[violation])
        self._context.refresh_portfolio()
        return trading_pb2.SubmitOrderResponse(
            order_id=result.order_id,
            external_order_id=str(result.raw_response.get("exchange_order_id", ""))
            if isinstance(getattr(result, "raw_response", None), Mapping)
            else "",
            status=trading_pb2.ORDER_STATUS_ACCEPTED,
        )

    def CancelOrder(self, request, context):  # noqa: N802
        self._context.authorize(context)
        execution_service: ExecutionService = self._context.pipeline.execution_service
        try:
            with self._lock:
                execution_service.cancel(request.order_id, self._context.execution_context)
        except Exception as exc:
            self._context.emit_alert(
                category="execution.order",
                title="Anulowanie zlecenia nie powiodło się",
                body=str(exc),
                severity="warning",
                context={"order_id": request.order_id},
            )
            return trading_pb2.CancelOrderResponse(
                status=trading_pb2.ORDER_STATUS_REJECTED,
                message=str(exc),
            )
        return trading_pb2.CancelOrderResponse(status=trading_pb2.ORDER_STATUS_ACCEPTED)


class _RiskServicer(trading_pb2_grpc.RiskServiceServicer):
    def __init__(self, context: LocalRuntimeContext) -> None:
        if context.risk_store is None:
            raise RuntimeError("Risk store is not configured")
        self._store = context.risk_store

    def GetRiskState(self, request, context):  # noqa: N802
        return self._store.latest()

    def StreamRiskState(self, request, context):  # noqa: N802
        for snapshot in self._store.history():
            yield snapshot
        queue = self._store.register()
        try:
            while True:
                if context is not None and not getattr(context, "is_active", lambda: True)():
                    return
                try:
                    snapshot = queue.get(timeout=0.5)
                except Exception:
                    continue
                yield snapshot
        finally:
            self._store.unregister(queue)


class _MetricsServicer(trading_pb2_grpc.MetricsServiceServicer):
    def __init__(self, context: LocalRuntimeContext) -> None:
        self._context = context

    def StreamMetrics(self, request, context):  # noqa: N802
        snapshot = trading_pb2.MetricsSnapshot(
            generated_at=_timestamp_from_ms(int(time.time() * 1000)),
            event_to_frame_p95_ms=0.0,
            fps=0.0,
            cpu_utilization=0.0,
            gpu_utilization=0.0,
            ram_megabytes=0.0,
            dropped_frames=0,
            processed_messages_per_second=0,
            notes="local-runtime",
        )
        yield snapshot

    def PushMetrics(self, request, context):  # noqa: N802
        del request, context
        return trading_pb2.MetricsAck(accepted=True)


class _HealthServicer(trading_pb2_grpc.HealthServiceServicer):
    def __init__(self, context: LocalRuntimeContext) -> None:
        self._context = context

    def Check(self, request, context):  # noqa: N802
        del request, context
        response = trading_pb2.HealthCheckResponse(
            version=self._context.version,
            git_commit=self._context.git_commit or "unknown",
        )
        response.started_at.CopyFrom(_timestamp_from_ms(int(self._context.started_at.timestamp() * 1000)))
        return response


class _MarketplaceServicer(trading_pb2_grpc.MarketplaceServiceServicer):
    def __init__(self, context: LocalRuntimeContext) -> None:
        self._context = context

    def _build_summary(self, document) -> trading_pb2.MarketplacePresetSummary:
        metadata = document.metadata if isinstance(document.metadata, Mapping) else {}
        profile = str(metadata.get("profile") or metadata.get("risk_profile") or "").strip()
        summary = trading_pb2.MarketplacePresetSummary(
            preset_id=document.preset_id,
            name=str(document.payload.get("name") or document.preset_id or ""),
            version=document.version or "",
            profile=profile,
            signature_verified=document.verification.verified,
            source_path=str(document.path) if document.path else "",
        )
        summary.tags.extend(list(document.tags))
        summary.issues.extend(list(document.issues))
        return summary

    def ListPresets(self, request, context):  # noqa: N802
        del request
        self._context.authorize(context)
        response = trading_pb2.ListMarketplacePresetsResponse()
        for document in self._context.list_marketplace_presets():
            response.presets.append(self._build_summary(document))
        return response

    def ImportPreset(self, request, context):  # noqa: N802
        self._context.authorize(context)
        if not request.payload:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "payload is required")
        try:
            document = self._context.import_marketplace_preset(
                bytes(request.payload),
                request.filename or None,
            )
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:  # pragma: no cover - diagnostyka
            context.abort(grpc.StatusCode.INTERNAL, str(exc))
        return trading_pb2.ImportMarketplacePresetResponse(
            preset=self._build_summary(document)
        )

    def ExportPreset(self, request, context):  # noqa: N802
        self._context.authorize(context)
        format_value = request.format or "json"
        try:
            document, payload = self._context.export_marketplace_preset(
                request.preset_id,
                format=format_value,
            )
        except FileNotFoundError:
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"Preset {request.preset_id} not found",
            )
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:  # pragma: no cover - diagnostyka
            context.abort(grpc.StatusCode.INTERNAL, str(exc))
        extension = "yaml" if format_value.lower().strip() in {"yaml", "yml"} else "json"
        filename = request.preset_id or "preset"
        response = trading_pb2.ExportMarketplacePresetResponse(
            payload=payload,
            filename=f"{filename}.{extension}",
            preset=self._build_summary(document),
            format=format_value or extension,
        )
        return response

    def RemovePreset(self, request, context):  # noqa: N802
        self._context.authorize(context)
        removed = self._context.remove_marketplace_preset(request.preset_id)
        return trading_pb2.RemoveMarketplacePresetResponse(removed=removed)

    def ActivatePreset(self, request, context):  # noqa: N802
        self._context.authorize(context)
        document = self._context.activate_marketplace_preset(request.preset_id)
        if document is None:
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"Preset {request.preset_id} not found",
            )
        return trading_pb2.ActivateMarketplacePresetResponse(
            preset=self._build_summary(document)
        )


def _serialize_timestamp(timestamp: timestamp_pb2.Timestamp) -> int:
    return int(timestamp.seconds) * 1000 + int(timestamp.nanos) // 1_000_000


def _serialize_candle(candle: trading_pb2.OhlcvCandle) -> Mapping[str, Any]:
    return {
        "timestamp_ms": _serialize_timestamp(candle.open_time),
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        "volume": candle.volume,
        "sequence": candle.sequence,
    }


def _serialize_instrument_metadata(
    entry: trading_pb2.TradableInstrumentMetadata,
) -> Mapping[str, Any]:
    instrument = entry.instrument
    return {
        "exchange": instrument.exchange,
        "symbol": instrument.symbol,
        "venue_symbol": instrument.venue_symbol,
        "quote_currency": instrument.quote_currency,
        "base_currency": instrument.base_currency,
        "price_step": entry.price_step,
        "amount_step": entry.amount_step,
        "min_notional": entry.min_notional,
        "min_amount": entry.min_amount,
        "max_amount": entry.max_amount,
        "min_price": entry.min_price,
        "max_price": entry.max_price,
    }


def _serialize_risk_state(state: trading_pb2.RiskState) -> Mapping[str, Any]:
    payload = MessageToDict(state, preserving_proto_field_name=True)
    payload["generated_at_ms"] = _serialize_timestamp(state.generated_at)
    return payload


def _serialize_marketplace_summary(
    summary: trading_pb2.MarketplacePresetSummary,
) -> Mapping[str, Any]:
    return MessageToDict(summary, preserving_proto_field_name=True)


class LocalRuntimeGateway:
    """Small dispatch layer used by the desktop shell JSON bridge."""

    def __init__(self, context: LocalRuntimeContext) -> None:
        self._context = context
        self._market = _MarketDataServicer(context)
        self._order = _OrderServicer(context)
        self._metrics = _MetricsServicer(context)
        self._health = _HealthServicer(context)
        self._risk = _RiskServicer(context) if context.risk_store is not None else None
        self._marketplace = (
            _MarketplaceServicer(context)
            if context.marketplace_repository is not None and context.marketplace_enabled
            else None
        )

    def dispatch(self, method: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        handlers: Mapping[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]] = {
            "health.check": self._health_check,
            "market_data.get_ohlcv_history": self._get_ohlcv_history,
            "market_data.stream_ohlcv": self._stream_ohlcv,
            "market_data.list_tradable_instruments": self._list_instruments,
            "risk.get_state": self._get_risk_state,
            "marketplace.list_presets": self._list_marketplace_presets,
            "marketplace.import_preset": self._import_marketplace_preset,
            "marketplace.export_preset": self._export_marketplace_preset,
            "marketplace.remove_preset": self._remove_marketplace_preset,
            "marketplace.activate_preset": self._activate_marketplace_preset,
        }
        handler = handlers.get(method)
        if handler is None:
            raise KeyError(f"Unsupported method: {method}")
        return handler(params)

    def _health_check(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        del params
        response = self._health.Check(empty_pb2.Empty(), None)
        payload = MessageToDict(response, preserving_proto_field_name=True)
        payload["started_at_ms"] = _serialize_timestamp(response.started_at)
        return payload

    def _get_ohlcv_history(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        request = trading_pb2.GetOhlcvHistoryRequest()
        instrument = request.instrument
        symbol = str(params.get("symbol") or "").strip() or self._context.primary_symbol
        instrument.symbol = symbol
        instrument.venue_symbol = str(params.get("venue_symbol") or "")
        instrument.exchange = str(params.get("exchange") or "")
        if params.get("granularity"):
            request.granularity.iso8601_duration = str(params.get("granularity"))
        if params.get("limit"):
            request.limit = int(params.get("limit"))
        if params.get("start_ms"):
            request.start_time.CopyFrom(
                _timestamp_from_ms(int(params.get("start_ms")))
            )
        if params.get("end_ms"):
            request.end_time.CopyFrom(
                _timestamp_from_ms(int(params.get("end_ms")))
            )
        response = self._market.GetOhlcvHistory(request, None)
        candles = [_serialize_candle(candle) for candle in response.candles]
        return {"candles": candles, "has_more": response.has_more}

    def _stream_ohlcv(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        request = trading_pb2.StreamOhlcvRequest()
        instrument = request.instrument
        instrument.symbol = str(params.get("symbol") or "") or self._context.primary_symbol
        if params.get("limit"):
            request.limit = int(params.get("limit"))
        stream = self._market.StreamOhlcv(request, None)
        snapshot: list[Mapping[str, Any]] = []
        updates: list[Mapping[str, Any]] = []
        for update in stream:
            if update.HasField("snapshot"):
                snapshot = [_serialize_candle(c) for c in update.snapshot.candles]
            if update.HasField("increment"):
                updates.append(_serialize_candle(update.increment.candle))
        return {"snapshot": snapshot, "updates": updates}

    def _list_instruments(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        request = trading_pb2.ListTradableInstrumentsRequest()
        if params.get("exchange"):
            request.exchange = str(params["exchange"])
        response = self._market.ListTradableInstruments(request, None)
        instruments = [_serialize_instrument_metadata(entry) for entry in response.instruments]
        return {"instruments": instruments}

    def _get_risk_state(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        del params
        if self._risk is None:
            return {}
        response = self._risk.GetRiskState(trading_pb2.RiskStateRequest(), None)
        return _serialize_risk_state(response)

    def _ensure_marketplace(self) -> _MarketplaceServicer:
        if self._marketplace is None:
            raise RuntimeError("Marketplace is disabled")
        return self._marketplace

    def _list_marketplace_presets(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        del params
        service = self._ensure_marketplace()
        response = service.ListPresets(trading_pb2.ListMarketplacePresetsRequest(), None)
        presets = [_serialize_marketplace_summary(entry) for entry in response.presets]
        return {"presets": presets}

    def _import_marketplace_preset(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        service = self._ensure_marketplace()
        request = trading_pb2.ImportMarketplacePresetRequest()
        payload = params.get("payload", b"")
        if isinstance(payload, str):
            request.payload = base64.b64decode(payload.encode("ascii"))
        else:
            request.payload = payload or b""
        if params.get("filename"):
            request.filename = str(params["filename"])
        response = service.ImportPreset(request, None)
        return MessageToDict(response, preserving_proto_field_name=True)

    def _export_marketplace_preset(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        service = self._ensure_marketplace()
        request = trading_pb2.ExportMarketplacePresetRequest()
        request.preset_id = str(params.get("preset_id") or "")
        if params.get("format"):
            request.format = str(params["format"])
        response = service.ExportPreset(request, None)
        payload = base64.b64encode(response.payload).decode("ascii")
        return {
            "preset": MessageToDict(response.preset, preserving_proto_field_name=True),
            "payload_base64": payload,
            "filename": response.filename,
        }

    def _remove_marketplace_preset(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        service = self._ensure_marketplace()
        request = trading_pb2.RemoveMarketplacePresetRequest()
        request.preset_id = str(params.get("preset_id") or "")
        response = service.RemovePreset(request, None)
        return MessageToDict(response, preserving_proto_field_name=True)

    def _activate_marketplace_preset(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        service = self._ensure_marketplace()
        request = trading_pb2.ActivateMarketplacePresetRequest()
        request.preset_id = str(params.get("preset_id") or "")
        response = service.ActivatePreset(request, None)
        return MessageToDict(response, preserving_proto_field_name=True)


class LocalRuntimeServer:
    """Łączy serwisy gRPC i udostępnia je pod jednym adresem."""

    def __init__(
        self,
        context: LocalRuntimeContext,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        max_workers: int = 16,
    ) -> None:
        self._context = context
        self._server = grpc.server(ThreadPoolExecutor(max_workers=max_workers))
        trading_pb2_grpc.add_MarketDataServiceServicer_to_server(_MarketDataServicer(context), self._server)
        trading_pb2_grpc.add_OrderServiceServicer_to_server(_OrderServicer(context), self._server)
        if context.risk_store is not None:
            trading_pb2_grpc.add_RiskServiceServicer_to_server(_RiskServicer(context), self._server)
        trading_pb2_grpc.add_MetricsServiceServicer_to_server(_MetricsServicer(context), self._server)
        trading_pb2_grpc.add_HealthServiceServicer_to_server(_HealthServicer(context), self._server)
        if context.marketplace_repository is not None and context.marketplace_enabled:
            trading_pb2_grpc.add_MarketplaceServiceServicer_to_server(
                _MarketplaceServicer(context),
                self._server,
            )
        address = f"{host}:{port}"
        bound = self._server.add_insecure_port(address)
        if bound == 0:
            raise RuntimeError(f"Nie udało się zbindować adresu {address}")
        if port == 0:
            self._address = f"{host}:{bound}"
        else:
            self._address = address

    @property
    def address(self) -> str:
        return self._address

    def start(self) -> None:
        self._server.start()

    def stop(self, grace: float | None = 0.0) -> None:
        self._server.stop(grace).wait()

    def wait(self) -> None:
        self._server.wait_for_termination()


def build_local_runtime_context(
    *,
    config_path: str | Path = "config/runtime.yaml",
    entrypoint: str | None = None,
    secret_manager: SecretManager | None = None,
) -> LocalRuntimeContext:
    runtime_config = load_runtime_app_config(config_path)
    entrypoint_name = entrypoint or runtime_config.trading.default_entrypoint
    try:
        entrypoint_cfg = runtime_config.trading.entrypoints[entrypoint_name]
    except KeyError as exc:  # pragma: no cover - walidacja wejścia
        raise KeyError(f"Brak punktu wejścia '{entrypoint_name}' w runtime.yaml") from exc
    config_file = Path(config_path).expanduser().resolve()
    core_path = runtime_config.core.resolved_path or runtime_config.core.path
    core_config_path = (config_file.parent / core_path).resolve()
    core_config = load_core_config(core_config_path)
    environment_cfg = core_config.environments.get(entrypoint_cfg.environment)
    effective_strategy = entrypoint_cfg.strategy
    if effective_strategy and environment_cfg is not None:
        available = getattr(core_config, "strategies", {})
        if effective_strategy not in available:
            fallback_strategy = getattr(environment_cfg, "default_strategy", None)
            if fallback_strategy:
                effective_strategy = fallback_strategy
    effective_controller = entrypoint_cfg.controller
    if effective_controller and environment_cfg is not None:
        controllers = getattr(core_config, "runtime_controllers", {})
        if effective_controller not in controllers:
            fallback_controller = getattr(environment_cfg, "default_controller", None)
            if fallback_controller:
                effective_controller = fallback_controller
    sanitized_core_config = _sanitize_core_config(
        core_config,
        entrypoint_cfg.environment,
        strategy_override=effective_strategy,
        controller_override=effective_controller,
    )
    pipeline_module = sys.modules.get("bot_core.runtime.pipeline")
    if pipeline_module is not None and not hasattr(
        pipeline_module, "_ensure_local_market_data_availability"
    ):
        _LOGGER.warning(
            "Brak funkcji _ensure_local_market_data_availability – pomijam weryfikację danych OHLCV"
        )
        setattr(
            pipeline_module,
            "_ensure_local_market_data_availability",
            lambda *args, **kwargs: None,
        )
    if secret_manager is None:
        secret_manager = SecretManager(_InMemorySecretStorage())
    try:
        import bot_core.trading.engine as _trading_engine  # noqa: F401
    except SyntaxError as exc:  # pragma: no cover - degradacja do stubu
        _LOGGER.warning("Import trading.engine zakończył się błędem składni: %s", exc)
        stub = _build_trading_engine_stub()
        sys.modules["bot_core.trading.engine"] = stub
    except Exception as exc:  # pragma: no cover - inne błędy importu
        _LOGGER.warning("Import trading.engine zgłosił wyjątek: %s", exc)
    try:
        from bot_core.auto_trader.app import AutoTrader as _AutoTraderCls
    except Exception as exc:  # pragma: no cover - środowiska bez pełnego modułu trading
        _LOGGER.warning("Nie udało się załadować AutoTradera – używam implementacji zastępczej: %s", exc)
        _AutoTraderCls = _AutoTraderStub
    environment_cfg = sanitized_core_config.environments.get(entrypoint_cfg.environment)
    if isinstance(secret_manager._storage, _InMemorySecretStorage) and environment_cfg is not None:
        keychain_key = getattr(environment_cfg, "keychain_key", None)
        credential_purpose = getattr(environment_cfg, "credential_purpose", "trading")
        env_value = getattr(environment_cfg, "environment", ExchangeEnvironment.PAPER)
        try:
            env_enum = env_value if isinstance(env_value, ExchangeEnvironment) else ExchangeEnvironment(str(env_value))
        except Exception:
            env_enum = ExchangeEnvironment.PAPER
        if keychain_key:
            stub_credentials = ExchangeCredentials(
                key_id="stub-key",
                secret="stub-secret",
                passphrase=None,
                environment=env_enum,
                permissions=("read", "trade"),
            )
            try:
                secret_manager.store_exchange_credentials(
                    keychain_key,
                    stub_credentials,
                    purpose=str(credential_purpose or "trading"),
                )
            except Exception:  # pragma: no cover - diagnostyka środowisk niestandardowych
                _LOGGER.debug("Nie udało się zapisać stubowych poświadczeń API", exc_info=True)

    pipeline = build_daily_trend_pipeline(
        environment_name=entrypoint_cfg.environment,
        strategy_name=effective_strategy,
        controller_name=effective_controller,
        config_path=core_config_path,
        secret_manager=secret_manager,
        risk_profile_name=entrypoint_cfg.risk_profile,
        core_config=sanitized_core_config,
        runtime_config=runtime_config,
    )
    execution_service = getattr(pipeline, "execution_service", None)
    if execution_service is not None and getattr(execution_service, "_price_resolver", None) is None:
        execution_service._price_resolver = lambda symbol: 100.0  # type: ignore[attr-defined]
        markets = getattr(execution_service, "_markets", None)
        if isinstance(markets, dict):
            for symbol, market in list(markets.items()):
                markets[symbol] = MarketMetadata(
                    base_asset=market.base_asset,
                    quote_asset=market.quote_asset,
                    min_quantity=getattr(market, "min_quantity", 0.0),
                    min_notional=0.0,
                    step_size=getattr(market, "step_size", None),
                    tick_size=getattr(market, "tick_size", None),
                )
    controller_context = getattr(pipeline.controller, "execution_context", None)
    if controller_context is not None and getattr(controller_context, "price_resolver", None) is None:
        controller_context.price_resolver = lambda symbol: 100.0
    alert_router = getattr(pipeline.bootstrap, "alert_router", None)
    if alert_router is None:
        try:
            from bot_core.alerts.audit import InMemoryAlertAuditLog

            alert_router = DefaultAlertRouter(InMemoryAlertAuditLog())
        except Exception:  # pragma: no cover - fallback gdy brak modułu audytu
            alert_router = None
    active_alert_router = alert_router or DefaultAlertRouterStub()
    trading_controller = create_trading_controller(pipeline, active_alert_router)
    runner = DailyTrendRealtimeRunner(
        controller=pipeline.controller,
        trading_controller=trading_controller,
    )
    symbols = getattr(pipeline.controller, "symbols", ())
    if not symbols:
        raise RuntimeError("Zbudowany pipeline nie udostępnia żadnych symboli handlowych")
    gui_stub = _GuiStub(
        timeframe=getattr(pipeline.controller, "interval", "1h"),
        ai_manager=getattr(pipeline.bootstrap, "ai_manager", None),
        portfolio_manager=getattr(pipeline.bootstrap, "portfolio_governor", None),
    )
    market_data_provider = _AutoTraderMarketDataProvider(pipeline.data_source)
    auto_trader = _AutoTraderCls(
        _Emitter(),
        gui_stub,
        lambda: str(symbols[0]),
        market_data_provider=market_data_provider,
        risk_service=getattr(pipeline.bootstrap, "risk_engine", None),
        execution_service=pipeline.execution_service,
        bootstrap_context=pipeline.bootstrap,
        core_risk_engine=getattr(pipeline.bootstrap, "risk_engine", None),
        core_execution_service=pipeline.execution_service,
        ai_connector=getattr(pipeline.bootstrap, "ai_manager", None),
        decision_journal=getattr(pipeline.bootstrap, "decision_journal", None),
        decision_journal_context={"environment": pipeline.bootstrap.environment.name},
        controller_runner=runner,
        trusted_auto_confirm=True,
    )
    risk_store: RiskSnapshotStore | None = None
    risk_builder: RiskSnapshotBuilder | None = None
    risk_publisher: RiskSnapshotPublisher | None = None
    risk_engine = getattr(pipeline.bootstrap, "risk_engine", None)
    if risk_engine is not None:
        try:
            risk_store = RiskSnapshotStore(maxlen=512)
            risk_builder = RiskSnapshotBuilder(risk_engine)
            def _append_snapshot(snapshot: Any) -> None:
                try:
                    proto = snapshot.to_proto()
                except Exception:  # pragma: no cover - diagnostyka konwersji
                    _LOGGER.debug("Nie udało się przekształcić RiskSnapshot w protobuf", exc_info=True)
                    return
                risk_store.append(proto)
            risk_publisher = RiskSnapshotPublisher(
                risk_builder,
                profiles=(pipeline.risk_profile_name,),
                sinks=[_append_snapshot],
            )
        except Exception:  # pragma: no cover - diagnostyka
            _LOGGER.exception("Nie udało się zainicjalizować komponentów RiskService")
            risk_store = risk_builder = risk_publisher = None
    metrics_registry = get_global_metrics_registry()
    observability_cfg = getattr(runtime_config, "observability", None)
    prometheus_exporter: LocalPrometheusExporter | None = None
    alert_sink_token: str | None = None
    if observability_cfg is not None:
        metrics_cfg = getattr(observability_cfg, "prometheus", None)
        if metrics_cfg is not None and getattr(metrics_cfg, "enabled", True):
            prometheus_exporter = LocalPrometheusExporter(
                host=getattr(metrics_cfg, "host", "127.0.0.1"),
                port=int(getattr(metrics_cfg, "port", 0) or 0),
                metrics_path=getattr(metrics_cfg, "path", "/metrics"),
                registry=metrics_registry,
            )
        alerts_cfg = getattr(observability_cfg, "alerts", None)
        if alerts_cfg is not None:
            severity_value = getattr(alerts_cfg, "min_severity", "warning") or "warning"
            try:
                severity_enum = AlertSeverity(str(severity_value).lower())
            except Exception:
                severity_enum = AlertSeverity.WARNING
            try:
                alert_sink_token = ensure_offline_logging_sink(
                    min_severity=severity_enum
                )
            except Exception:  # pragma: no cover - rejestracja alertów nie powinna blokować
                _LOGGER.debug("Nie udało się zarejestrować sinka alertów offline", exc_info=True)
    marketplace_cfg = getattr(runtime_config, "marketplace", None)
    marketplace_repository: PresetRepository | None = None
    marketplace_signing_keys: dict[str, bytes] = {}
    marketplace_allow_unsigned = False
    marketplace_enabled = True
    if marketplace_cfg is not None:
        marketplace_enabled = bool(getattr(marketplace_cfg, "enabled", True))
        presets_location = getattr(marketplace_cfg, "presets_path", "config/marketplace/presets")
        presets_path = Path(presets_location)
        if not presets_path.is_absolute():
            presets_path = (config_file.parent / presets_path).resolve()
        marketplace_repository = PresetRepository(presets_path)
        marketplace_repository.root.mkdir(parents=True, exist_ok=True)
        raw_keys = getattr(marketplace_cfg, "signing_keys", {}) or {}
        if isinstance(raw_keys, Mapping):
            for key_id, raw_value in raw_keys.items():
                try:
                    marketplace_signing_keys[str(key_id)] = decode_key_material(raw_value)
                except Exception:  # pragma: no cover - defensywne logowanie
                    _LOGGER.debug(
                        "Nie udało się zdekodować klucza podpisu Marketplace %s", key_id,
                        exc_info=True,
                    )
        marketplace_allow_unsigned = bool(getattr(marketplace_cfg, "allow_unsigned", False))
        if marketplace_enabled:
            try:
                DEFAULT_STRATEGY_CATALOG.load_presets_from_directory(
                    presets_path,
                    signing_keys=marketplace_signing_keys,
                    hwid_provider=None,
                )
            except FileNotFoundError:
                pass
            except Exception:  # pragma: no cover - diagnostyka
                _LOGGER.debug(
                    "Nie udało się wczytać presetów Marketplace przy inicjalizacji",
                    exc_info=True,
                )
    context = LocalRuntimeContext(
        config=runtime_config,
        entrypoint=entrypoint_cfg,
        config_path=config_file,
        pipeline=pipeline,
        trading_controller=trading_controller,
        runner=runner,
        auto_trader=auto_trader,
        alert_router=active_alert_router,
        secret_manager=secret_manager,
        risk_store=risk_store,
        risk_builder=risk_builder,
        risk_publisher=risk_publisher,
        metrics_registry=metrics_registry,
        auth_token=getattr(runtime_config.execution, "auth_token", None),
        prometheus_exporter=prometheus_exporter,
        alert_sink_token=alert_sink_token,
        marketplace_repository=marketplace_repository,
        marketplace_signing_keys=marketplace_signing_keys,
        marketplace_allow_unsigned=marketplace_allow_unsigned,
        marketplace_enabled=marketplace_enabled,
    )
    if marketplace_repository is not None and marketplace_enabled:
        try:
            context.reload_marketplace_presets()
        except Exception:  # pragma: no cover - diagnostyka
            _LOGGER.debug(
                "Nie udało się zainicjalizować katalogu Marketplace w runtime",
                exc_info=True,
            )
    return context


__all__ = [
    "LocalRuntimeContext",
    "LocalRuntimeServer",
    "build_local_runtime_context",
]
