"""Implementacja serwera MetricsService odbierającego telemetrię z powłoki Qt/QML.

Moduł zapewnia:

* lekkie przechowywanie ostatnich metryk w pamięci,
* możliwość strumieniowania `MetricsSnapshot` do wielu klientów,
* integrację z dodatkowymi "sinkami" (np. logowanie, JSONL, alerty),
* serwer gRPC, który można wpiąć w docelowy daemon `bot_core`.

Kod nie zakłada obecności WebSocketów – jedynie gRPC (`grpcio`).
"""

from __future__ import annotations

from collections import deque
from concurrent import futures
from datetime import datetime, timezone
import json
import logging
import os
import threading
from queue import SimpleQueue, Empty
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence, TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:  # pragma: no cover
    from bot_core.alerts.base import AlertRouter
    from bot_core.config.models import MetricsServiceConfig

# --- Stuby gRPC są opcjonalne podczas developmentu ---
try:  # pragma: no cover - import opcjonalny, gdy brak wygenerowanych stubów
    from bot_core.generated import trading_pb2, trading_pb2_grpc  # type: ignore
except ImportError:  # pragma: no cover - instrukcja dla developera
    trading_pb2 = None  # type: ignore
    trading_pb2_grpc = None  # type: ignore

if trading_pb2_grpc is not None:  # pragma: no cover - zależne od wygenerowanych stubów
    _MetricsServicerBase = trading_pb2_grpc.MetricsServiceServicer  # type: ignore
else:  # pragma: no cover - środowiska developerskie bez gRPC
    _MetricsServicerBase = object

try:  # pragma: no cover - środowiska bez grpcio
    import grpc
except ImportError:  # pragma: no cover - instrukcja dla developera
    grpc = None  # type: ignore

# Alerty mogą być nieobecne na starszych gałęziach
try:  # pragma: no cover
    from bot_core.alerts.base import AlertMessage
except Exception:  # pragma: no cover
    AlertMessage = None  # type: ignore

try:  # pragma: no cover - sink telemetrii UI jest opcjonalny
    from bot_core.runtime.metrics_alerts import (  # type: ignore
        DEFAULT_UI_ALERTS_JSONL_PATH,
        UiTelemetryAlertSink,
    )
except Exception:  # pragma: no cover - brak modułu telemetrii UI
    UiTelemetryAlertSink = None  # type: ignore
    DEFAULT_UI_ALERTS_JSONL_PATH = Path("logs/ui_telemetry_alerts.jsonl")

try:  # pragma: no cover - presety profili ryzyka mogą nie być dostępne
    from bot_core.runtime.telemetry_risk_profiles import (  # type: ignore
        MetricsRiskProfileResolver,
        load_risk_profiles_with_metadata,
        risk_profile_metadata,
    )
except Exception:  # pragma: no cover - starsze gałęzie bez presetów
    MetricsRiskProfileResolver = None  # type: ignore
    load_risk_profiles_with_metadata = None  # type: ignore
    risk_profile_metadata = None  # type: ignore

_LOGGER = logging.getLogger(__name__)


# =============================================================================
# Sinki metryk
# =============================================================================

class MetricsSink(Protocol):
    """Komponent przyjmujący pojedynczy `MetricsSnapshot`."""

    def handle_snapshot(self, snapshot) -> None:  # pragma: no cover - interfejs
        ...


class LoggingSink:
    """Prosty sink logujący przychodzące snapshoty."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or _LOGGER

    def handle_snapshot(self, snapshot) -> None:  # pragma: no cover - logowanie
        self._logger.info("MetricsSnapshot received", extra={"metrics_notes": snapshot.notes})


class JsonlSink:
    """Zapisuje snapshoty do pliku JSONL (bez blokowania pipeline'u)."""

    def __init__(self, path: str | Path, *, fsync: bool = False) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Tworzymy plik, aby kolejne kroki CI mogły odczytać artefakt nawet bez metryk.
        self._path.touch(exist_ok=True)
        self._fsync = fsync
        self._lock = threading.Lock()

    def handle_snapshot(self, snapshot) -> None:
        record = {
            "generated_at": _timestamp_to_iso(snapshot.generated_at)
            if getattr(snapshot, "HasField", None) and snapshot.HasField("generated_at")
            else None,
            "fps": snapshot.fps if getattr(snapshot, "HasField", None) and snapshot.HasField("fps") else None,
            "notes": snapshot.notes,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                if self._fsync:
                    handle.flush()
                    os.fsync(handle.fileno())


class ReduceMotionAlertSink:
    """Wysyła alerty, gdy UI zgłasza zdarzenia reduce-motion."""

    def __init__(
        self,
        router: "AlertRouter",
        *,
        category: str = "ui.performance",
        severity_active: str = "warning",
        severity_recovered: str = "info",
        logger: logging.Logger | None = None,
    ) -> None:
        if AlertMessage is None:  # pragma: no cover - brak modułu alertów
            raise RuntimeError("ReduceMotionAlertSink wymaga modułu bot_core.alerts.base")
        self._router = router
        self._category = category
        self._severity_active = severity_active
        self._severity_recovered = severity_recovered
        self._logger = logger or _LOGGER
        self._last_state: bool | None = None

    def handle_snapshot(self, snapshot) -> None:
        notes = getattr(snapshot, "notes", "")
        if not notes:
            return
        try:
            payload = json.loads(notes)
        except json.JSONDecodeError:
            self._logger.debug("Niepoprawny JSON w notes: %s", notes)
            return
        if payload.get("event") != "reduce_motion":
            return

        active = bool(payload.get("active", False))
        overlay_active = payload.get("overlay_active")
        overlay_allowed = payload.get("overlay_allowed")
        fps_target = payload.get("fps_target")
        window_count = payload.get("window_count")
        disable_secondary = payload.get("disable_secondary_fps")
        tag = payload.get("tag")

        # Duplikaty – pomijamy, by nie zalewać alertów
        if self._last_state is not None and self._last_state == active:
            return
        self._last_state = active

        severity = self._severity_active if active else self._severity_recovered
        title = "Reduce motion aktywny" if active else "Reduce motion przywrócony"
        body_lines = [
            f"Stan: {'aktywny' if active else 'wyłączony'} (okna={window_count or 'n/d'}).",
            f"Nakładki: {overlay_active if overlay_active is not None else 'n/d'} / "
            f"{overlay_allowed if overlay_allowed is not None else 'n/d'}; fps_target={fps_target if fps_target is not None else 'n/d'}.",
        ]
        if disable_secondary:
            body_lines.append(f"Próg wyłączania nakładek: {disable_secondary} FPS")
        if getattr(snapshot, "HasField", None) and snapshot.HasField("fps"):
            body_lines.append(f"Ostatnia próbka FPS={snapshot.fps:.2f}")
        body = "\n".join(body_lines)

        context = {
            "event": "reduce_motion",
            "active": str(active).lower(),
            "overlay_active": str(overlay_active) if overlay_active is not None else "",
            "overlay_allowed": str(overlay_allowed) if overlay_allowed is not None else "",
            "fps_target": str(fps_target) if fps_target is not None else "",
        }
        if window_count is not None:
            context["window_count"] = str(window_count)
        if disable_secondary is not None:
            context["disable_secondary_fps"] = str(disable_secondary)
        if getattr(snapshot, "HasField", None) and snapshot.HasField("fps"):
            context["fps"] = f"{snapshot.fps:.4f}"
        if tag:
            context["tag"] = str(tag)

        message = AlertMessage(
            category=self._category,
            title=title,
            body=body,
            severity=severity,
            context=context,
        )
        try:
            self._router.dispatch(message)
        except Exception:  # pragma: no cover - nie blokujemy telemetrii
            self._logger.exception("Nie udało się wysłać alertu reduce-motion")


class OverlayBudgetAlertSink:
    """Generuje alerty, gdy liczba nakładek przekracza przydzielony limit."""

    def __init__(
        self,
        router: "AlertRouter",
        *,
        category: str = "ui.performance",
        severity_exceeded: str = "warning",
        severity_recovered: str = "info",
        logger: logging.Logger | None = None,
    ) -> None:
        if AlertMessage is None:  # pragma: no cover - brak modułu alertów
            raise RuntimeError("OverlayBudgetAlertSink wymaga modułu bot_core.alerts.base")
        self._router = router
        self._category = category
        self._severity_exceeded = severity_exceeded
        self._severity_recovered = severity_recovered
        self._logger = logger or _LOGGER
        self._last_exceeded: bool | None = None

    def handle_snapshot(self, snapshot) -> None:
        notes = getattr(snapshot, "notes", "")
        if not notes:
            return
        try:
            payload = json.loads(notes)
        except json.JSONDecodeError:
            self._logger.debug("Niepoprawny JSON w notes: %s", notes)
            return
        if payload.get("event") != "overlay_budget":
            return

        try:
            active = int(payload.get("active_overlays"))
            allowed = int(payload.get("allowed_overlays"))
        except (TypeError, ValueError):
            self._logger.debug("Brak prawidłowych wartości overlay w notes: %s", notes)
            return
        if allowed < 0:
            return

        exceeded = active > allowed
        if self._last_exceeded is not None and self._last_exceeded == exceeded:
            return

        self._last_exceeded = exceeded
        reduce_motion = bool(payload.get("reduce_motion", False))
        disable_secondary = payload.get("disable_secondary_fps")
        fps_target = payload.get("fps_target")
        window_count = payload.get("window_count")

        severity = self._severity_exceeded if exceeded else self._severity_recovered
        title = "Limit nakładek przekroczony" if exceeded else "Limit nakładek wrócił do normy"
        body_lines = [
            f"Nakładki: {active} / {allowed} (reduce_motion={'tak' if reduce_motion else 'nie'}).",
        ]
        if disable_secondary:
            body_lines.append(f"Próg wyłączania nakładek: {disable_secondary} FPS")
        if fps_target is not None:
            body_lines.append(f"Docelowe FPS: {fps_target}")
        if getattr(snapshot, "HasField", None) and snapshot.HasField("fps"):
            body_lines.append(f"Ostatnia próbka FPS={snapshot.fps:.2f}")
        body = "\n".join(body_lines)

        context = {
            "event": "overlay_budget",
            "exceeded": str(exceeded).lower(),
            "active_overlays": str(active),
            "allowed_overlays": str(allowed),
            "reduce_motion": str(reduce_motion).lower(),
        }
        if disable_secondary is not None:
            context["disable_secondary_fps"] = str(disable_secondary)
        if fps_target is not None:
            context["fps_target"] = str(fps_target)
        if window_count is not None:
            context["window_count"] = str(window_count)
        if getattr(snapshot, "HasField", None) and snapshot.HasField("fps"):
            context["fps"] = f"{snapshot.fps:.4f}"

        message = AlertMessage(
            category=self._category,
            title=title,
            body=body,
            severity=severity,
            context=context,
        )
        try:
            self._router.dispatch(message)
        except Exception:  # pragma: no cover - nie blokujemy telemetrii
            self._logger.exception("Nie udało się wysłać alertu overlay budget")


# =============================================================================
# Pamięć + broker
# =============================================================================

def _timestamp_to_iso(ts) -> str | None:
    if ts is None:
        return None
    seconds = getattr(ts, "seconds", 0)
    nanos = getattr(ts, "nanos", 0)
    if seconds == 0 and nanos == 0:
        return None
    dt = datetime.fromtimestamp(seconds + nanos / 1_000_000_000, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


class MetricsSnapshotStore:
    """Pamięć pierścieniowa + broker strumieniowy dla metryk."""

    def __init__(self, *, maxlen: int = 1024) -> None:
        if trading_pb2 is None:  # pragma: no cover - brak stubów
            raise RuntimeError(
                "Brak wygenerowanych modułów trading_pb2. Uruchom "
                "'python scripts/generate_trading_stubs.py --skip-cpp'."
            )
        self._history: deque = deque(maxlen=maxlen)
        self._subscribers: set[SimpleQueue] = set()
        self._lock = threading.Lock()

    def append(self, snapshot) -> None:
        """Dodaje snapshot i publikuje go do wszystkich subskrybentów."""
        clone = trading_pb2.MetricsSnapshot()
        clone.CopyFrom(snapshot)
        with self._lock:
            self._history.append(clone)
            subscribers = list(self._subscribers)
        for queue in subscribers:
            queue.put(clone)

    def snapshot_history(self) -> Sequence:
        with self._lock:
            return list(self._history)

    def register(self) -> SimpleQueue:
        queue: SimpleQueue = SimpleQueue()
        with self._lock:
            self._subscribers.add(queue)
        return queue

    def unregister(self, queue: SimpleQueue) -> None:
        with self._lock:
            self._subscribers.discard(queue)


# =============================================================================
# Serwis gRPC
# =============================================================================

class MetricsServiceServicer(_MetricsServicerBase):
    """Implementacja serwisu gRPC `MetricsService`."""

    def __init__(
        self,
        store: MetricsSnapshotStore,
        sinks: Iterable[MetricsSink] | None = None,
        auth_token: str | None = None,
    ) -> None:
        if trading_pb2_grpc is None or grpc is None:  # pragma: no cover - brak stubów
            raise RuntimeError(
                "Do uruchomienia MetricsService wymagane są moduły trading_pb2*_grpc "
                "oraz biblioteka grpcio."
            )
        try:
            super().__init__()  # type: ignore[misc]
        except Exception:
            pass
        self._store = store
        self._sinks: tuple[MetricsSink, ...] = tuple(sinks or ())
        self._auth_token = auth_token

    # --- autoryzacja (opcjonalna) ---
    def _extract_token(self, context) -> str | None:
        if context is None or self._auth_token is None:
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

    def _ensure_authorized(self, context) -> None:
        if self._auth_token is None:
            return
        token = self._extract_token(context)
        if token == self._auth_token:
            return
        if context is not None and grpc is not None:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Brak poprawnego tokenu telemetrii")
        raise RuntimeError("Brak poprawnego tokenu telemetrii")

    @property
    def sinks(self) -> tuple[MetricsSink, ...]:
        """Zwraca aktywne sinki powiązane z serwisem."""
        return self._sinks

    def StreamMetrics(self, request, context):  # noqa: N802 - sygnatura gRPC
        self._ensure_authorized(context)
        for snapshot in self._store.snapshot_history():
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

    def PushMetrics(self, request, context):  # noqa: N802 - sygnatura gRPC
        self._ensure_authorized(context)
        self._store.append(request)
        for sink in self._sinks:
            try:
                sink.handle_snapshot(request)
            except Exception:  # pragma: no cover - nie blokujemy pipeline'u
                _LOGGER.exception("Sink %s zgłosił wyjątek", sink)
        ack = trading_pb2.MetricsAck()
        ack.accepted = True
        return ack


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


def _read_tls_material(source: Any) -> bytes | None:
    if source is None:
        return None
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    path = Path(source)
    return path.read_bytes()


def _build_server_credentials(tls_config: Mapping[str, Any] | Any) -> Any | None:
    if tls_config is None:
        return None
    if isinstance(tls_config, Mapping):
        enabled = bool(tls_config.get("enabled", True))
    else:
        enabled = bool(getattr(tls_config, "enabled", True))
    if not enabled:
        return None
    if grpc is None:
        raise RuntimeError("Konfiguracja TLS wymaga biblioteki grpcio")

    certificate_source = _get_value(
        tls_config, "certificate_path", "certificate", "cert_path", "cert"
    )
    key_source = _get_value(
        tls_config, "private_key_path", "private_key", "key_path", "key"
    )
    if not certificate_source or not key_source:
        raise ValueError("Konfiguracja TLS wymaga certyfikatu serwera i klucza prywatnego")

    certificate = _read_tls_material(certificate_source)
    private_key = _read_tls_material(key_source)
    client_ca_source = _get_value(
        tls_config, "client_ca_path", "client_ca", "ca_path", "ca"
    )
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


class MetricsServer:
    """Pełny serwer gRPC obsługujący `MetricsService`."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        max_workers: int = 4,
        history_size: int = 1024,
        sinks: Iterable[MetricsSink] | None = None,
        # obie gałęzie: wspieramy jednocześnie TLS/mTLS i auth token
        server_credentials: Any | None = None,
        runtime_metadata: Mapping[str, Any] | None = None,
        auth_token: str | None = None,
    ) -> None:
        if grpc is None or trading_pb2_grpc is None:  # pragma: no cover
            raise RuntimeError(
                "Uruchomienie MetricsServer wymaga pakietów grpcio oraz wygenerowanych stubów."
            )
        self._store = MetricsSnapshotStore(maxlen=history_size)
        self._servicer = MetricsServiceServicer(self._store, sinks=sinks, auth_token=auth_token)
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        trading_pb2_grpc.add_MetricsServiceServicer_to_server(self._servicer, self._server)
        self._address = f"{host}:{port}"
        self._history_size = history_size
        self._runtime_metadata: dict[str, Any] = dict(runtime_metadata or {})
        tls_meta = self._runtime_metadata.get("tls", {})
        self._tls_client_auth_required = bool(tls_meta.get("require_client_auth"))
        if server_credentials is not None:
            bound_port = self._server.add_secure_port(self._address, server_credentials)
            self._tls_enabled = True
        else:
            bound_port = self._server.add_insecure_port(self._address)
            self._tls_enabled = False
        if bound_port == 0:
            raise RuntimeError("Nie udało się zbindować portu dla MetricsServer")
        if port == 0:
            self._address = f"{host}:{bound_port}"

    @property
    def address(self) -> str:
        return self._address

    @property
    def store(self) -> MetricsSnapshotStore:
        return self._store

    @property
    def sinks(self) -> tuple[MetricsSink, ...]:
        return self._servicer.sinks

    @property
    def history_size(self) -> int:
        return self._history_size

    @property
    def tls_enabled(self) -> bool:
        return self._tls_enabled

    @property
    def tls_client_auth_required(self) -> bool:
        return self._tls_client_auth_required

    @property
    def runtime_metadata(self) -> Mapping[str, Any]:
        return self._runtime_metadata

    def start(self) -> None:
        self._server.start()

    def stop(self, grace: Optional[float] = None) -> None:
        self._server.stop(grace).wait()

    def wait_for_termination(self, timeout: Optional[float] = None) -> None:
        self._server.wait_for_termination(timeout)


# =============================================================================
# Fabryki
# =============================================================================

def create_server(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    sinks: Iterable[MetricsSink] | None = None,
    history_size: int = 1024,
    enable_logging_sink: bool = True,
    jsonl_path: str | Path | None = None,
    jsonl_fsync: bool = False,
    ui_alerts_jsonl_path: str | Path | None = None,
    ui_alerts_config: Mapping[str, Any] | None = None,
    # spójnie wspieramy oba warianty
    tls_config: Mapping[str, Any] | Any | None = None,
    auth_token: str | None = None,
):
    """Pomocnicza funkcja do budowy serwera z domyślnymi sinkami."""
    base_sinks: list[MetricsSink] = []
    if enable_logging_sink:
        base_sinks.append(LoggingSink())
    if jsonl_path:
        base_sinks.append(JsonlSink(jsonl_path, fsync=jsonl_fsync))
    additional_sinks: list[MetricsSink] = list(sinks or ())
    if additional_sinks:
        base_sinks.extend(additional_sinks)
    credentials = None
    tls_require_client_auth = False

    def _tls_value(name: str, default: Any = None) -> Any:
        if tls_config is None:
            return default
        if isinstance(tls_config, Mapping):
            return tls_config.get(name, default)
        return getattr(tls_config, name, default)

    if tls_config is not None:
        tls_require_client_auth = bool(_tls_value("require_client_auth", False))
        credentials = _build_server_credentials(tls_config)
    risk_profile_meta = None
    if ui_alerts_config and "risk_profile" in ui_alerts_config:
        profile_payload = ui_alerts_config["risk_profile"]
        if isinstance(profile_payload, Mapping):
            risk_profile_meta = dict(profile_payload)
        elif profile_payload is not None:
            risk_profile_meta = {"name": str(profile_payload)}

    runtime_metadata = {
        "history_size": history_size,
        "logging_sink_enabled": enable_logging_sink,
        "jsonl_sink": {
            "active": bool(jsonl_path),
            "path": str(Path(jsonl_path).expanduser()) if jsonl_path else None,
            "fsync": bool(jsonl_fsync) if jsonl_path else False,
        },
        "ui_alerts_sink": {
            "active": bool(ui_alerts_jsonl_path),
            "path": str(Path(ui_alerts_jsonl_path).expanduser()) if ui_alerts_jsonl_path else None,
            "config": dict(ui_alerts_config) if ui_alerts_config else None,
            "risk_profile": risk_profile_meta,
        },
        "sink_descriptions": [
            {"class": sink.__class__.__name__, "module": sink.__class__.__module__}
            for sink in base_sinks
        ],
        "additional_sink_descriptions": [
            {"class": sink.__class__.__name__, "module": sink.__class__.__module__}
            for sink in additional_sinks
        ],
        "tls": {
            "configured": credentials is not None,
            "require_client_auth": tls_require_client_auth,
        },
    }
    server = MetricsServer(
        host=host,
        port=port,
        history_size=history_size,
        sinks=base_sinks,
        server_credentials=credentials,
        runtime_metadata=runtime_metadata,
        auth_token=auth_token,
    )
    return server


def build_metrics_server_from_config(
    config: "MetricsServiceConfig" | None,
    *,
    sinks: Iterable[MetricsSink] | None = None,
    alerts_router: "AlertRouter" | None = None,
) -> MetricsServer | None:
    """Buduje `MetricsServer` na bazie konfiguracji – zwraca ``None`` gdy wyłączony."""
    if config is None or not config.enabled:
        return None

    sink_list: list[MetricsSink] = list(sinks or [])
    ui_alerts_path: str | None = None
    ui_alerts_settings: Mapping[str, Any] | None = None
    risk_profile_meta: Mapping[str, Any] | None = None
    risk_profiles_file_meta: Mapping[str, Any] | None = None
    resolver: "MetricsRiskProfileResolver" | None = None
    profiles_file_value = getattr(config, "ui_alerts_risk_profiles_file", None)
    if profiles_file_value:
        normalized_file = str(Path(profiles_file_value).expanduser())
        if load_risk_profiles_with_metadata is None:  # type: ignore[truthy-bool]
            risk_profiles_file_meta = {
                "path": normalized_file,
                "warning": "risk_profile_loader_unavailable",
            }
        else:
            try:
                _, risk_profiles_file_meta = load_risk_profiles_with_metadata(  # type: ignore[misc]
                    normalized_file,
                    origin_label=f"metrics_service_config:{normalized_file}",
                )
            except Exception:  # pragma: no cover - diagnostyka konfiguracji
                _LOGGER.exception(
                    "Nie udało się wczytać profili ryzyka telemetrii z %s", normalized_file
                )
                raise
    profile_name = getattr(config, "ui_alerts_risk_profile", None)
    if profile_name:
        normalized_profile = str(profile_name).strip().lower()
        if MetricsRiskProfileResolver is None:
            base_meta = {"name": normalized_profile}
            if callable(risk_profile_metadata):  # type: ignore[arg-type]
                try:
                    base_meta = dict(risk_profile_metadata(normalized_profile))  # type: ignore[misc]
                except Exception:  # pragma: no cover - diagnostyka
                    base_meta = {"name": normalized_profile}
            base_meta["warning"] = "resolver_unavailable"
            risk_profile_meta = base_meta
        else:
            try:
                resolver = MetricsRiskProfileResolver(normalized_profile, config)
            except KeyError:
                warning_meta = {"name": normalized_profile, "error": "unknown_profile"}
                risk_profile_meta = warning_meta
                resolver = None
                _LOGGER.warning(
                    "Nieznany profil ryzyka telemetrii UI: %s", normalized_profile
                )
            except Exception:  # pragma: no cover - diagnostyka
                resolver = None
                base_meta = {"name": normalized_profile}
                _LOGGER.exception("Nie udało się zastosować profilu ryzyka %s", normalized_profile)
                risk_profile_meta = base_meta

    def _resolve_config_value(field_name: str, default: Any) -> Any:
        value = getattr(config, field_name, default)
        if resolver is not None:
            value = resolver.override(field_name, value)
        return value
    # Integracja z alertami UI – gdy dostępny router i włączone flagi w configu
    def _normalize_mode(mode_value, *, fallback_attr: str) -> str:
        if mode_value is not None:
            normalized = str(mode_value).lower()
            if normalized in {"enable", "jsonl", "disable"}:
                return normalized
        else:
            normalized = None
        dispatch_enabled = bool(getattr(config, fallback_attr, False))
        if normalized is None:
            return "enable" if dispatch_enabled else "disable"
        return normalized

    reduce_mode = _normalize_mode(
        _resolve_config_value("reduce_motion_mode", None), fallback_attr="reduce_motion_alerts"
    )
    overlay_mode = _normalize_mode(
        _resolve_config_value("overlay_alert_mode", None), fallback_attr="overlay_alerts"
    )
    jank_mode = _normalize_mode(
        _resolve_config_value("jank_alert_mode", None), fallback_attr="jank_alerts"
    )
    reduce_dispatch = reduce_mode == "enable"
    overlay_dispatch = overlay_mode == "enable"
    jank_dispatch = jank_mode == "enable"
    reduce_logging = reduce_mode in {"enable", "jsonl"}
    overlay_logging = overlay_mode in {"enable", "jsonl"}
    jank_logging = jank_mode in {"enable", "jsonl"}
    ui_sink_attached = False
    if alerts_router is not None and UiTelemetryAlertSink is not None:
        try:
            configured_path = getattr(config, "ui_alerts_jsonl_path", None)
            path_value = configured_path or str(DEFAULT_UI_ALERTS_JSONL_PATH)
            reduce_category = _resolve_config_value("reduce_motion_category", "ui.performance")
            reduce_active = _resolve_config_value("reduce_motion_severity_active", "warning")
            reduce_recovered = _resolve_config_value("reduce_motion_severity_recovered", "info")
            overlay_category = _resolve_config_value("overlay_alert_category", "ui.performance")
            overlay_exceeded = _resolve_config_value("overlay_alert_severity_exceeded", "warning")
            overlay_recovered = _resolve_config_value("overlay_alert_severity_recovered", "info")
            overlay_critical = _resolve_config_value("overlay_alert_severity_critical", None)
            overlay_threshold_raw = _resolve_config_value("overlay_alert_critical_threshold", None)
            jank_category = _resolve_config_value("jank_alert_category", "ui.performance")
            jank_spike = _resolve_config_value("jank_alert_severity_spike", "warning")
            jank_critical = _resolve_config_value("jank_alert_severity_critical", None)
            jank_threshold_raw = _resolve_config_value("jank_alert_critical_over_ms", None)

            sink_kwargs = dict(
                jsonl_path=path_value,
                enable_reduce_motion_alerts=reduce_dispatch,
                enable_overlay_alerts=overlay_dispatch,
                log_reduce_motion_events=reduce_logging,
                log_overlay_events=overlay_logging,
                enable_jank_alerts=jank_dispatch,
                log_jank_events=jank_logging,
                reduce_motion_category=reduce_category,
                reduce_motion_severity_active=reduce_active,
                reduce_motion_severity_recovered=reduce_recovered,
                overlay_category=overlay_category,
                overlay_severity_exceeded=overlay_exceeded,
                overlay_severity_recovered=overlay_recovered,
                jank_category=jank_category,
                jank_severity_spike=jank_spike,
            )
            if overlay_critical is not None:
                sink_kwargs["overlay_severity_critical"] = overlay_critical
            overlay_threshold_value: int | None = None
            if overlay_threshold_raw is not None:
                try:
                    overlay_threshold_value = int(overlay_threshold_raw)
                except (TypeError, ValueError):
                    _LOGGER.debug(
                        "Nieprawidłowy próg overlay_alert_critical_threshold=%s", overlay_threshold_raw
                    )
                else:
                    sink_kwargs["overlay_critical_threshold"] = overlay_threshold_value
            jank_threshold_value: float | None = None
            if jank_threshold_raw is not None:
                try:
                    jank_threshold_value = float(jank_threshold_raw)
                except (TypeError, ValueError):
                    _LOGGER.debug(
                        "Nieprawidłowy próg jank_alert_critical_over_ms=%s", jank_threshold_raw
                    )
                else:
                    sink_kwargs["jank_critical_over_ms"] = jank_threshold_value
            if jank_critical is not None:
                sink_kwargs["jank_severity_critical"] = jank_critical
            if resolver is not None:
                risk_profile_meta = resolver.metadata()
            if risk_profile_meta is not None:
                sink_kwargs["risk_profile"] = dict(risk_profile_meta)
            sink_list.append(UiTelemetryAlertSink(alerts_router, **sink_kwargs))
            ui_alerts_path = path_value
            ui_sink_attached = True
            ui_alerts_settings = {
                "jsonl_path": path_value,
                "reduce_mode": reduce_mode,
                "overlay_mode": overlay_mode,
                "jank_mode": jank_mode,
                "reduce_motion_alerts": reduce_dispatch,
                "overlay_alerts": overlay_dispatch,
                "jank_alerts": jank_dispatch,
                "reduce_motion_logging": reduce_logging,
                "overlay_logging": overlay_logging,
                "jank_logging": jank_logging,
                "reduce_motion_category": sink_kwargs.get("reduce_motion_category"),
                "reduce_motion_severity_active": sink_kwargs.get("reduce_motion_severity_active"),
                "reduce_motion_severity_recovered": sink_kwargs.get("reduce_motion_severity_recovered"),
                "overlay_category": sink_kwargs.get("overlay_category"),
                "overlay_severity_exceeded": sink_kwargs.get("overlay_severity_exceeded"),
                "overlay_severity_recovered": sink_kwargs.get("overlay_severity_recovered"),
                "overlay_severity_critical": sink_kwargs.get("overlay_severity_critical"),
                "overlay_critical_threshold": sink_kwargs.get("overlay_critical_threshold"),
                "jank_category": sink_kwargs.get("jank_category"),
                "jank_severity_spike": sink_kwargs.get("jank_severity_spike"),
                "jank_severity_critical": sink_kwargs.get("jank_severity_critical"),
                "jank_critical_over_ms": sink_kwargs.get("jank_critical_over_ms"),
            }
            if risk_profile_meta is not None:
                ui_alerts_settings["risk_profile"] = dict(risk_profile_meta)
            if risk_profiles_file_meta is not None:
                ui_alerts_settings["risk_profiles_file"] = dict(risk_profiles_file_meta)
        except Exception:  # pragma: no cover - diagnostyka pomocnicza
            _LOGGER.exception("Nie udało się zainicjalizować UiTelemetryAlertSink")
    if not ui_sink_attached and alerts_router is not None:
        if reduce_dispatch:
            sink_list.append(
                ReduceMotionAlertSink(
                    alerts_router,
                    category=getattr(config, "reduce_motion_category", "ui.performance"),
                    severity_active=getattr(
                        config, "reduce_motion_severity_active", "warning"
                    ),
                    severity_recovered=getattr(
                        config, "reduce_motion_severity_recovered", "info"
                    ),
                )
            )
        if overlay_dispatch:
            sink_list.append(
                OverlayBudgetAlertSink(
                    alerts_router,
                    category=getattr(config, "overlay_alert_category", "ui.performance"),
                    severity_exceeded=getattr(
                        config, "overlay_alert_severity_exceeded", "warning"
                    ),
                    severity_recovered=getattr(
                        config, "overlay_alert_severity_recovered", "info"
                    ),
                )
            )
    if ui_alerts_path is None and UiTelemetryAlertSink is not None:
        for sink in sink_list:
            if isinstance(sink, UiTelemetryAlertSink):
                try:
                    ui_alerts_path = str(Path(sink.jsonl_path).expanduser())
                except Exception:  # pragma: no cover - diagnostyka pomocnicza
                    ui_alerts_path = str(sink.jsonl_path)
                break

    if resolver is not None and risk_profile_meta is None:
        risk_profile_meta = resolver.metadata()
    if ui_alerts_settings is None and risk_profile_meta is not None:
        ui_alerts_settings = {"risk_profile": dict(risk_profile_meta)}
    if risk_profiles_file_meta is not None:
        if ui_alerts_settings is None:
            ui_alerts_settings = {"risk_profiles_file": dict(risk_profiles_file_meta)}
        else:
            ui_alerts_settings["risk_profiles_file"] = dict(risk_profiles_file_meta)

    return create_server(
        host=config.host,
        port=config.port,
        history_size=config.history_size,
        sinks=sink_list,
        enable_logging_sink=getattr(config, "log_sink", True),
        jsonl_path=getattr(config, "jsonl_path", None),
        jsonl_fsync=getattr(config, "jsonl_fsync", False),
        ui_alerts_jsonl_path=ui_alerts_path,
        ui_alerts_config=ui_alerts_settings,
        tls_config=getattr(config, "tls", None),
        auth_token=getattr(config, "auth_token", None),
    )
