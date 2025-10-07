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
from typing import Iterable, Optional, Protocol, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from bot_core.alerts.base import AlertRouter
    from bot_core.config.models import MetricsServiceConfig

# --- Stuby gRPC są opcjonalne podczas developmentu ---
try:  # pragma: no cover - import opcjonalny, gdy brak wygenerowanych stubów
    from bot_core.generated import trading_pb2, trading_pb2_grpc  # type: ignore
except ImportError:  # pragma: no cover - instrukcja dla developera
    trading_pb2 = None  # type: ignore
    trading_pb2_grpc = None  # type: ignore

try:  # pragma: no cover - środowiska bez grpcio
    import grpc
except ImportError:  # pragma: no cover - instrukcja dla developera
    grpc = None  # type: ignore

# Alerty mogą być nieobecne na starszych gałęziach
try:  # pragma: no cover
    from bot_core.alerts.base import AlertMessage
except Exception:  # pragma: no cover
    AlertMessage = None  # type: ignore

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

class MetricsServiceServicer(trading_pb2_grpc.MetricsServiceServicer):
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
        super().__init__()
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
        bound_port = self._server.add_insecure_port(self._address)
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
    auth_token: str | None = None,
):
    """Pomocnicza funkcja do budowy serwera z domyślnymi sinkami."""
    base_sinks: list[MetricsSink] = []
    if enable_logging_sink:
        base_sinks.append(LoggingSink())
    if jsonl_path:
        base_sinks.append(JsonlSink(jsonl_path, fsync=jsonl_fsync))
    if sinks:
        base_sinks.extend(list(sinks))
    server = MetricsServer(
        host=host,
        port=port,
        history_size=history_size,
        sinks=base_sinks,
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
    # Integracja z alertami UI – gdy dostępny router i włączone flagi w configu
    if alerts_router is not None and getattr(config, "reduce_motion_alerts", False):
        sink_list.append(
            ReduceMotionAlertSink(
                alerts_router,
                category=getattr(config, "reduce_motion_category", "ui.performance"),
                severity_active=getattr(config, "reduce_motion_severity_active", "warning"),
                severity_recovered=getattr(config, "reduce_motion_severity_recovered", "info"),
            )
        )
    if alerts_router is not None and getattr(config, "overlay_alerts", False):
        sink_list.append(
            OverlayBudgetAlertSink(
                alerts_router,
                category=getattr(config, "overlay_alert_category", "ui.performance"),
                severity_exceeded=getattr(config, "overlay_alert_severity_exceeded", "warning"),
                severity_recovered=getattr(config, "overlay_alert_severity_recovered", "info"),
            )
        )

    return create_server(
        host=config.host,
        port=config.port,
        history_size=config.history_size,
        sinks=sink_list,
        enable_logging_sink=config.log_sink,
        jsonl_path=config.jsonl_path,
        jsonl_fsync=config.jsonl_fsync,
        auth_token=getattr(config, "auth_token", None),
    )
