"""Serwis gRPC udostępniający aktualny stan profili ryzyka."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from concurrent import futures
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hmac
import logging
import threading
from queue import Empty, SimpleQueue
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.security.tokens import ServiceToken, ServiceTokenValidator

try:  # pragma: no cover - środowiska bez protobuf/gRPC
    from bot_core.generated import trading_pb2, trading_pb2_grpc  # type: ignore
except ImportError:  # pragma: no cover - brak wygenerowanych stubów
    trading_pb2 = None  # type: ignore
    trading_pb2_grpc = None  # type: ignore

try:  # pragma: no cover - zależność opcjonalna
    import grpc
except ImportError:  # pragma: no cover - brak biblioteki gRPC
    grpc = None  # type: ignore

try:  # pragma: no cover - protobuf jest opcjonalny w środowisku deweloperskim
    from google.protobuf import timestamp_pb2
except Exception:  # pragma: no cover
    timestamp_pb2 = None  # type: ignore

try:  # pragma: no cover - presety profili są opcjonalne w starszych gałęziach
    from bot_core.runtime.telemetry_risk_profiles import get_risk_profile_summary  # type: ignore
except Exception:  # pragma: no cover - brak presetów/telemetrii
    get_risk_profile_summary = None  # type: ignore

_LOGGER = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _warning_threshold(max_value: float | None, ratio: float = 0.85) -> float | None:
    if not max_value or max_value <= 0:
        return None
    return max_value * ratio


def _profile_enum(profile_name: str) -> int:
    if trading_pb2 is None:  # pragma: no cover - brak stubów
        return 0
    mapping = {
        "conservative": trading_pb2.RiskProfile.RISK_PROFILE_CONSERVATIVE,
        "balanced": trading_pb2.RiskProfile.RISK_PROFILE_BALANCED,
        "aggressive": trading_pb2.RiskProfile.RISK_PROFILE_AGGRESSIVE,
        "manual": trading_pb2.RiskProfile.RISK_PROFILE_MANUAL,
    }
    return mapping.get(profile_name.strip().lower(), trading_pb2.RiskProfile.RISK_PROFILE_UNSPECIFIED)


def _default_profile_summary_resolver(profile_name: str) -> Mapping[str, Any] | None:
    if get_risk_profile_summary is None:
        return None
    return get_risk_profile_summary(profile_name)


@dataclass(slots=True)
class RiskExposure:
    """Pojedynczy limit ekspozycji w strukturze raportowanej do UI."""

    code: str
    current: float
    maximum: float | None = None
    threshold: float | None = None

    def as_proto(self):  # pragma: no cover - zależne od protobuf
        if trading_pb2 is None:
            raise RuntimeError("Brak modułu trading_pb2 – wygeneruj stuby z proto/trading.proto")
        limit = trading_pb2.ExposureLimit()
        limit.code = self.code
        limit.current_value = float(self.current)
        if self.maximum is not None:
            limit.max_value = float(self.maximum)
        if self.threshold is not None:
            limit.threshold_value = float(self.threshold)
        return limit


@dataclass(slots=True)
class RiskSnapshot:
    """Ujednolicony snapshot stanu profilu ryzyka."""

    profile_name: str
    portfolio_value: float
    current_drawdown: float
    daily_loss: float
    used_leverage: float
    exposures: Sequence[RiskExposure]
    generated_at: datetime
    force_liquidation: bool = False
    metadata: Mapping[str, object] | None = None

    def profile_summary(self) -> Mapping[str, Any] | None:
        """Zwraca podsumowanie profilu ryzyka osadzone w metadanych."""

        if not isinstance(self.metadata, Mapping):
            return None
        summary = self.metadata.get("risk_profile_summary")
        if isinstance(summary, Mapping):
            return summary
        return None

    def profile_enum(self) -> int:
        return _profile_enum(self.profile_name)

    def to_proto(self):  # pragma: no cover - zależne od protobuf
        if trading_pb2 is None or timestamp_pb2 is None:
            raise RuntimeError(
                "Konwersja RiskSnapshot do protobuf wymaga pakietów trading_pb2 oraz google.protobuf"
            )
        message = trading_pb2.RiskState()
        message.profile = self.profile_enum()
        message.portfolio_value = float(self.portfolio_value)
        message.current_drawdown = float(self.current_drawdown)
        message.max_daily_loss = float(self.daily_loss)
        message.used_leverage = float(self.used_leverage)
        timestamp = timestamp_pb2.Timestamp()
        timestamp.FromDatetime(self.generated_at)
        message.generated_at.CopyFrom(timestamp)
        for exposure in self.exposures:
            message.limits.add().CopyFrom(exposure.as_proto())
        if self.force_liquidation:
            liquidation = trading_pb2.ExposureLimit()
            liquidation.code = "force_liquidation"
            liquidation.current_value = 1.0
            liquidation.max_value = 1.0
            liquidation.threshold_value = 1.0
            message.limits.add().CopyFrom(liquidation)
        return message


class RiskSnapshotBuilder:
    """Buduje RiskSnapshot z danych przechowywanych w ThresholdRiskEngine."""

    def __init__(
        self,
        risk_engine: ThresholdRiskEngine,
        *,
        clock: Callable[[], datetime] | None = None,
        profile_summary_resolver: Callable[[str], Mapping[str, Any]] | None = None,
    ) -> None:
        self._risk_engine = risk_engine
        self._clock = clock or _utc_now
        self._summary_resolver = profile_summary_resolver or _default_profile_summary_resolver

    def build(self, profile_name: str) -> RiskSnapshot | None:
        state = self._risk_engine.snapshot_state(profile_name)
        if state is None:
            return None

        limits_raw = state.get("limits", {}) if isinstance(state, Mapping) else {}
        limits: MutableMapping[str, float] = {}
        if isinstance(limits_raw, Mapping):
            limits = {str(key): _safe_float(value) for key, value in limits_raw.items()}

        equity = _safe_float(state.get("last_equity"), _safe_float(state.get("start_of_day_equity")))
        gross_notional = _safe_float(state.get("gross_notional"))
        used_leverage = gross_notional / equity if equity > 0 else 0.0
        drawdown = _safe_float(state.get("drawdown_pct"))
        daily_loss = _safe_float(state.get("daily_loss_pct"))
        active_positions = _safe_float(state.get("active_positions"))
        force_liquidation = bool(state.get("force_liquidation"))

        positions_raw = state.get("positions", {}) if isinstance(state, Mapping) else {}
        largest_position = 0.0
        if isinstance(positions_raw, Mapping):
            for entry in positions_raw.values():
                if isinstance(entry, Mapping):
                    largest_position = max(largest_position, _safe_float(entry.get("notional")))

        exposures: list[RiskExposure] = []

        max_positions = limits.get("max_positions")
        exposures.append(
            RiskExposure(
                code="active_positions",
                current=active_positions,
                maximum=max_positions,
                threshold=_warning_threshold(max_positions),
            )
        )

        max_leverage = limits.get("max_leverage")
        max_notional = max_leverage * equity if max_leverage and equity > 0 else None
        exposures.append(
            RiskExposure(
                code="portfolio_leverage",
                current=used_leverage,
                maximum=max_leverage,
                threshold=_warning_threshold(max_leverage),
            )
        )
        exposures.append(
            RiskExposure(
                code="gross_notional",
                current=gross_notional,
                maximum=max_notional,
                threshold=_warning_threshold(max_notional),
            )
        )

        daily_limit = limits.get("daily_loss_limit")
        exposures.append(
            RiskExposure(
                code="daily_loss_pct",
                current=daily_loss,
                maximum=daily_limit,
                threshold=_warning_threshold(daily_limit),
            )
        )

        drawdown_limit = limits.get("drawdown_limit")
        exposures.append(
            RiskExposure(
                code="drawdown_pct",
                current=drawdown,
                maximum=drawdown_limit,
                threshold=_warning_threshold(drawdown_limit),
            )
        )

        position_limit_pct = limits.get("max_position_pct")
        largest_ratio = largest_position / equity if equity > 0 else 0.0
        exposures.append(
            RiskExposure(
                code="largest_position_pct",
                current=largest_ratio,
                maximum=position_limit_pct,
                threshold=_warning_threshold(position_limit_pct),
            )
        )

        metadata: dict[str, object] = {
            "force_liquidation": force_liquidation,
            "limits": dict(limits),
            "positions": positions_raw if isinstance(positions_raw, Mapping) else {},
            "gross_notional": gross_notional,
            "active_positions": active_positions,
        }

        recent_decisions: Sequence[Mapping[str, object]] | None = None
        decisions_provider = getattr(self._risk_engine, "recent_decisions", None)
        if callable(decisions_provider):
            try:
                recent_decisions = decisions_provider(profile_name=profile_name, limit=10)
            except TypeError:
                try:  # pragma: no cover - defensywna obsługa starszych sygnatur
                    recent_decisions = decisions_provider(profile_name, 10)
                except TypeError:
                    recent_decisions = None
        if recent_decisions:
            metadata["recent_decisions"] = [dict(event) for event in recent_decisions]

        summary_payload = self._resolve_profile_summary(profile_name)
        if summary_payload is not None:
            metadata["risk_profile_summary"] = summary_payload

        snapshot = RiskSnapshot(
            profile_name=profile_name,
            portfolio_value=equity,
            current_drawdown=drawdown,
            daily_loss=daily_loss,
            used_leverage=used_leverage,
            exposures=tuple(exposures),
            generated_at=self._clock(),
            force_liquidation=force_liquidation,
            metadata=metadata,
        )
        metadata["profile"] = profile_name
        metadata["generated_at"] = snapshot.generated_at.isoformat()
        return snapshot

    def _resolve_profile_summary(self, profile_name: str) -> Mapping[str, Any] | None:
        resolver = self._summary_resolver
        if resolver is None:
            return None
        try:
            summary = resolver(profile_name)
        except Exception:  # pragma: no cover - diagnostyka resolvera
            _LOGGER.debug("Nie udało się pobrać podsumowania profilu ryzyka", exc_info=True)
            return None
        if not isinstance(summary, Mapping):
            return None
        return deepcopy(summary)

    def profile_names(self) -> Sequence[str]:
        """Zwraca listę profili dostępnych w powiązanym silniku ryzyka."""

        provider = getattr(self._risk_engine, "profile_names", None)
        names: Iterable[str] | None = None
        if callable(provider):
            try:
                names = provider()
            except TypeError:
                _LOGGER.debug(
                    "Resolver profili ryzyka wymaga nieobsługiwanych argumentów", exc_info=True
                )
                names = None
        if names is None:
            profiles_attr = getattr(self._risk_engine, "_profiles", None)
            if isinstance(profiles_attr, Mapping):
                names = profiles_attr.keys()
            else:
                return ()
        return tuple(str(name) for name in list(names))


class RiskSnapshotPublisher:
    """Publikuje snapshoty ryzyka do podłączonych sinków (np. RiskService)."""

    def __init__(
        self,
        builder: RiskSnapshotBuilder,
        *,
        profiles: Iterable[str] | None = None,
        sinks: Iterable[Callable[[RiskSnapshot], None]] | None = None,
        interval_seconds: float = 5.0,
        profile_provider: Callable[[], Iterable[str]] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._builder = builder
        self._profiles = tuple(str(name) for name in profiles) if profiles is not None else None
        self._profile_provider = profile_provider or builder.profile_names
        self._sinks: list[Callable[[RiskSnapshot], None]] = list(sinks or [])
        interval = float(interval_seconds)
        self._interval = 1.0 if interval <= 0 else interval
        self._logger = logger or _LOGGER
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def add_sink(self, sink: Callable[[RiskSnapshot], None]) -> None:
        with self._lock:
            self._sinks.append(sink)

    def publish_once(self) -> tuple[RiskSnapshot, ...]:
        snapshots: list[RiskSnapshot] = []
        profiles = self._profiles or self._normalize_profiles(self._profile_provider())
        for profile_name in profiles:
            try:
                snapshot = self._builder.build(profile_name)
            except Exception:  # pragma: no cover - diagnostyka buildera
                self._logger.exception("Nie udało się zbudować snapshotu ryzyka", extra={"profile": profile_name})
                continue
            if snapshot is None:
                continue
            snapshots.append(snapshot)
            sinks = self._current_sinks()
            for sink in sinks:
                try:
                    sink(snapshot)
                except Exception:  # pragma: no cover - defensywne logowanie
                    self._logger.exception(
                        "Sink snapshotu ryzyka zgłosił wyjątek",
                        extra={"profile": profile_name, "sink": getattr(sink, "__name__", sink.__class__.__name__)},
                    )
        return tuple(snapshots)

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._stop_event.clear()
            thread = threading.Thread(target=self._run, name="RiskSnapshotPublisher", daemon=True)
            self._thread = thread
            thread.start()

    def stop(self, timeout: float | None = None) -> None:
        thread: threading.Thread | None
        with self._lock:
            thread = self._thread
            if thread is None:
                return
            self._stop_event.set()
        thread.join(timeout)
        with self._lock:
            self._thread = None

    close = stop

    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None

    def __enter__(self) -> "RiskSnapshotPublisher":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _normalize_profiles(self, names: Iterable[str] | None) -> tuple[str, ...]:
        if names is None:
            return ()
        return tuple(str(name) for name in names)

    def _current_sinks(self) -> list[Callable[[RiskSnapshot], None]]:
        with self._lock:
            return list(self._sinks)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.publish_once()
            if self._stop_event.wait(self._interval):
                break


class RiskSnapshotStore:
    """Pamięć pierścieniowa + broker strumieniowy dla RiskState."""

    def __init__(self, *, maxlen: int = 256) -> None:
        if trading_pb2 is None:  # pragma: no cover - brak stubów
            raise RuntimeError(
                "Brak wygenerowanych modułów trading_pb2. Uruchom 'python scripts/generate_trading_stubs.py'."
            )
        self._history: deque = deque(maxlen=maxlen)
        self._metadata: deque = deque(maxlen=maxlen)
        self._subscribers: set[SimpleQueue] = set()
        self._lock = threading.Lock()

    def append(self, snapshot, *, metadata: Mapping[str, object] | None = None) -> None:
        clone = trading_pb2.RiskState()
        clone.CopyFrom(snapshot)
        metadata_payload = dict(metadata or {})
        with self._lock:
            self._history.append(clone)
            self._metadata.append(metadata_payload)
            subscribers = list(self._subscribers)
        for queue in subscribers:
            queue.put(clone)

    def latest(self):
        with self._lock:
            if not self._history:
                return trading_pb2.RiskState()
            return self._clone(self._history[-1])

    def history(self) -> Sequence:
        with self._lock:
            return [self._clone(item) for item in self._history]

    def metadata_history(self) -> Sequence[Mapping[str, object]]:
        with self._lock:
            return [dict(entry) for entry in self._metadata]

    def latest_metadata(self) -> Mapping[str, object] | None:
        with self._lock:
            if not self._metadata:
                return None
            return dict(self._metadata[-1])

    def register(self) -> SimpleQueue:
        queue: SimpleQueue = SimpleQueue()
        with self._lock:
            self._subscribers.add(queue)
        return queue

    def unregister(self, queue: SimpleQueue) -> None:
        with self._lock:
            self._subscribers.discard(queue)

    @staticmethod
    def _clone(snapshot) -> Any:
        clone = trading_pb2.RiskState()
        clone.CopyFrom(snapshot)
        return clone


class RiskServiceServicer(trading_pb2_grpc.RiskServiceServicer if trading_pb2_grpc else object):  # type: ignore[misc]
    """Implementacja serwisu gRPC udostępniającego RiskState."""

    def __init__(
        self,
        store: RiskSnapshotStore,
        *,
        auth_token: str | None = None,
        token_validator: ServiceTokenValidator | None = None,
    ) -> None:
        if trading_pb2_grpc is None or grpc is None:  # pragma: no cover - brak zależności
            raise RuntimeError("Uruchomienie RiskService wymaga wygenerowanych stubów oraz biblioteki grpcio")
        try:
            super().__init__()  # type: ignore[misc]
        except Exception:  # pragma: no cover - kompatybilność z różnymi wersjami gRPC
            pass
        self._store = store
        self._auth_token = auth_token
        self._token_validator = token_validator

    def _extract_token(self, context) -> str | None:
        if context is None:
            return None
        metadata = getattr(context, "invocation_metadata", lambda: ())()
        for key, value in metadata or ():
            key_lower = key.lower()
            if key_lower == "authorization":
                if value.lower().startswith("bearer "):
                    return value[7:]
                return value
            if key_lower == "x-bot-auth":
                return value
        return None

    def _attach_token_metadata(self, context, token: ServiceToken) -> None:
        if context is None or grpc is None:
            return
        try:
            context.set_trailing_metadata((("x-token-id", token.token_id),))
        except Exception:  # pragma: no cover - zależne od implementacji gRPC
            pass

    def _abort_unauthorized(self, context) -> None:
        if context is not None and grpc is not None:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Brak poprawnego tokenu risk service")
        raise RuntimeError("Brak poprawnego tokenu risk service")

    def _ensure_authorized(self, context, *, scope: str = "risk.read") -> None:
        requires_auth = self._auth_token is not None or (
            self._token_validator is not None and self._token_validator.requires_token
        )
        if not requires_auth:
            return
        token = self._extract_token(context)
        if self._token_validator is not None:
            matched = self._token_validator.validate(token, scope=scope)
            if matched is not None:
                self._attach_token_metadata(context, matched)
                return
        if self._auth_token is not None and token is not None:
            if hmac.compare_digest(token, self._auth_token):
                return
        self._abort_unauthorized(context)

    def GetRiskState(self, request, context):  # noqa: N802 - nazwa generowana przez gRPC
        self._ensure_authorized(context, scope="risk.read")
        return self._store.latest()

    def StreamRiskState(self, request, context):  # noqa: N802 - nazwa generowana przez gRPC
        self._ensure_authorized(context, scope="risk.read")
        for snapshot in self._store.history():
            yield snapshot
        queue = self._store.register()
        try:
            while True:
                if context and not getattr(context, "is_active", lambda: True)():
                    return
                try:
                    snapshot = queue.get(timeout=0.5)
                except Empty:
                    continue
                yield snapshot
        finally:
            self._store.unregister(queue)


def _read_tls_material(source: Any) -> bytes | None:
    if source is None:
        return None
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    return Path(source).read_bytes()


def _get_value(config: Mapping[str, Any] | Any, *candidates: str) -> Any:
    if config is None:
        return None
    if isinstance(config, Mapping):
        for name in candidates:
            if name in config and config[name] is not None:
                return config[name]
    for name in candidates:
        if hasattr(config, name):
            value = getattr(config, name)
            if value is not None:
                return value
    return None


def _build_server_credentials(tls_config: Mapping[str, Any] | Any) -> Any | None:
    if tls_config is None:
        return None
    enabled = bool(_get_value(tls_config, "enabled", "enable_tls") or True)
    if not enabled:
        return None
    if grpc is None:
        raise RuntimeError("Konfiguracja TLS wymaga biblioteki grpcio")

    certificate_source = _get_value(tls_config, "certificate_path", "certificate", "cert_path", "cert")
    key_source = _get_value(tls_config, "private_key_path", "private_key", "key_path", "key")
    if not certificate_source or not key_source:
        raise ValueError("Konfiguracja TLS wymaga certyfikatu serwera oraz klucza prywatnego")

    certificate = _read_tls_material(certificate_source)
    private_key = _read_tls_material(key_source)

    client_ca_source = _get_value(tls_config, "client_ca_path", "client_ca", "ca_path", "ca")
    client_ca = _read_tls_material(client_ca_source) if client_ca_source else None
    require_client_auth = bool(
        _get_value(
            tls_config,
            "require_client_auth",
            "require_client_certificate",
            "require_client_cert",
        )
        or False
    )
    return grpc.ssl_server_credentials(
        ((private_key, certificate),),
        root_certificates=client_ca,
        require_client_auth=require_client_auth,
    )


class RiskServer:
    """Pełny serwer gRPC publikujący aktualny RiskState."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        max_workers: int = 8,
        history_size: int = 256,
        auth_token: str | None = None,
        server_credentials: Any | None = None,
        token_validator: ServiceTokenValidator | None = None,
    ) -> None:
        if grpc is None or trading_pb2_grpc is None:
            raise RuntimeError("Uruchomienie RiskServer wymaga pakietów grpcio oraz trading_pb2*")
        self._store = RiskSnapshotStore(maxlen=history_size)
        self._servicer = RiskServiceServicer(
            self._store,
            auth_token=auth_token,
            token_validator=token_validator,
        )
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        trading_pb2_grpc.add_RiskServiceServicer_to_server(self._servicer, self._server)
        self._address = f"{host}:{port}"
        self._token_validator = token_validator
        if server_credentials is not None:
            bound_port = self._server.add_secure_port(self._address, server_credentials)
            self._tls_enabled = True
        else:
            bound_port = self._server.add_insecure_port(self._address)
            self._tls_enabled = False
        if bound_port == 0:
            raise RuntimeError("Nie udało się zbindować portu dla RiskServer")
        if port == 0:
            self._address = f"{host}:{bound_port}"

    @property
    def address(self) -> str:
        return self._address

    @property
    def store(self) -> RiskSnapshotStore:
        return self._store

    @property
    def token_validator(self) -> ServiceTokenValidator | None:
        return self._token_validator

    def start(self) -> None:
        self._server.start()

    def stop(self, grace: float | None = None) -> None:
        self._server.stop(grace).wait()

    def publish(self, snapshot: RiskSnapshot) -> None:
        proto = snapshot.to_proto()
        self._store.append(proto, metadata=snapshot.metadata)


def build_risk_server_from_config(
    config: Mapping[str, Any] | Any | None,
    *,
    auth_token: str | None = None,
    token_validator: ServiceTokenValidator | None = None,
) -> RiskServer | None:
    if config is None:
        return None
    enabled = bool(_get_value(config, "enabled", "enable", "active") or True)
    if not enabled:
        return None
    host = str(_get_value(config, "host") or "127.0.0.1")
    port = int(_get_value(config, "port") or 0)
    history_size = int(_get_value(config, "history_size", "history", "buffer_size") or 256)
    token = str(_get_value(config, "auth_token", "token") or auth_token or "") or None
    tls_config = _get_value(config, "tls", "server_credentials")
    credentials = _build_server_credentials(tls_config)
    return RiskServer(
        host=host,
        port=port,
        history_size=history_size,
        auth_token=token,
        server_credentials=credentials,
        token_validator=token_validator,
    )


__all__ = [
    "RiskExposure",
    "RiskSnapshot",
    "RiskSnapshotBuilder",
    "RiskSnapshotPublisher",
    "RiskSnapshotStore",
    "RiskServiceServicer",
    "RiskServer",
    "build_risk_server_from_config",
]
