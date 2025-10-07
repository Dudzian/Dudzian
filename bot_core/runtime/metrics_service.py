"""Implementacja serwera MetricsService odbierającego telemetrię z powłoki Qt/QML.

Moduł zapewnia:

* lekkie przechowywanie ostatnich metryk w pamięci,
* możliwość strumieniowania `MetricsSnapshot` do wielu klientów,
* integrację z dodatkowymi "sinkami" (np. logowanie, Prometheus),
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
    from bot_core.config.models import MetricsServiceConfig

try:  # pragma: no cover - import opcjonalny, gdy brak wygenerowanych stubów
    from bot_core.generated import trading_pb2, trading_pb2_grpc  # type: ignore
except ImportError:  # pragma: no cover - instrukcja dla developera
    trading_pb2 = None  # type: ignore
    trading_pb2_grpc = None  # type: ignore

try:  # pragma: no cover - środowiska bez grpcio
    import grpc
except ImportError:  # pragma: no cover - instrukcja dla developera
    grpc = None  # type: ignore

_LOGGER = logging.getLogger(__name__)


class MetricsSink(Protocol):
    """Komponent przyjmujący pojedynczy `MetricsSnapshot`.

    Sink może zapisywać dane do plików, Prometheusa lub wysyłać je dalej.
    """

    def handle_snapshot(self, snapshot) -> None:  # pragma: no cover - interfejs
        ...


class LoggingSink:
    """Prosty sink logujący przychodzące snapshoty."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or _LOGGER

    def handle_snapshot(self, snapshot) -> None:  # pragma: no cover - logowanie
        self._logger.info(
            "MetricsSnapshot received", extra={"metrics_notes": snapshot.notes}
        )


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
            if snapshot.HasField("generated_at")
            else None,
            "fps": snapshot.fps if snapshot.HasField("fps") else None,
            "notes": snapshot.notes,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                if self._fsync:
                    handle.flush()
                    os.fsync(handle.fileno())


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


class MetricsServiceServicer(trading_pb2_grpc.MetricsServiceServicer):
    """Implementacja serwisu gRPC `MetricsService`."""

    def __init__(
        self,
        store: MetricsSnapshotStore,
        sinks: Iterable[MetricsSink] | None = None,
    ) -> None:
        if trading_pb2_grpc is None or grpc is None:  # pragma: no cover - brak stubów
            raise RuntimeError(
                "Do uruchomienia MetricsService wymagane są moduły trading_pb2*_grpc "
                "oraz biblioteka grpcio."
            )
        super().__init__()
        self._store = store
        self._sinks: tuple[MetricsSink, ...] = tuple(sinks or ())

    def StreamMetrics(self, request, context):  # noqa: N802 - sygnatura gRPC
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
    ) -> None:
        if grpc is None or trading_pb2_grpc is None:  # pragma: no cover
            raise RuntimeError(
                "Uruchomienie MetricsServer wymaga pakietów grpcio oraz wygenerowanych stubów."
            )
        self._store = MetricsSnapshotStore(maxlen=history_size)
        self._servicer = MetricsServiceServicer(self._store, sinks=sinks)
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


def create_server(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    sinks: Iterable[MetricsSink] | None = None,
    history_size: int = 1024,
    enable_logging_sink: bool = True,
    jsonl_path: str | Path | None = None,
    jsonl_fsync: bool = False,
):
    """Pomocnicza funkcja do budowy serwera z domyślnym logging sinkiem."""

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
    )
    return server


def build_metrics_server_from_config(
    config: "MetricsServiceConfig" | None,
    *,
    sinks: Iterable[MetricsSink] | None = None,
) -> MetricsServer | None:
    """Buduje `MetricsServer` na bazie konfiguracji – zwraca ``None`` gdy wyłączony."""

    if config is None or not config.enabled:
        return None
    return create_server(
        host=config.host,
        port=config.port,
        history_size=config.history_size,
        sinks=sinks,
        enable_logging_sink=config.log_sink,
        jsonl_path=config.jsonl_path,
        jsonl_fsync=config.jsonl_fsync,
    )

