"""Klient odpowiedzialny za gRPC decision streaming dla RuntimeService."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

_LOGGER = logging.getLogger(__name__)


class GrpcDecisionStreamClient:
    """Obsługuje połączenie, odbiór streamu i reconnect dla decision feedu."""

    def __init__(
        self,
        *,
        target: str,
        metadata: tuple[tuple[str, str], ...],
        ssl_credentials: object,
        authority_override: str | None,
        limit: int,
        ready_timeout: float,
        retry_base: float,
        retry_multiplier: float,
        retry_max: float,
        cycle_metrics_serializer: Callable[[object], dict[str, float]],
        grpc_module: Any,
        stubs_loader: Callable[[], tuple[object, object]],
        queue_size: int = 64,
    ) -> None:
        self._target = target
        self._metadata = metadata
        self._ssl_credentials = ssl_credentials
        self._authority_override = authority_override
        self._limit = max(1, int(limit))
        self._ready_timeout = max(1.0, float(ready_timeout))
        self._retry_base = max(0.1, float(retry_base))
        self._retry_multiplier = max(1.0, float(retry_multiplier))
        self._retry_max = max(self._retry_base, float(retry_max))
        self._cycle_metrics_serializer = cycle_metrics_serializer
        self._grpc = grpc_module
        self._stubs_loader = stubs_loader

        self.events_queue: "queue.Queue[tuple[str, object]]" = queue.Queue(
            maxsize=max(8, queue_size)
        )
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self._lifecycle_lock = threading.Lock()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self.thread is not None and self.thread.is_alive():
                return
            self.stop_event.clear()
            self.thread = threading.Thread(
                target=self._worker,
                name="RuntimeServiceGrpc",
                daemon=True,
            )
            self.thread.start()

    def stop(self) -> None:
        with self._lifecycle_lock:
            worker = self.thread
            self.stop_event.set()
        if worker is not None and worker.is_alive():
            worker.join(timeout=1.5)
        with self._lifecycle_lock:
            self.thread = None

    def _worker(self) -> None:
        warn_rate_limit_seconds = 5.0
        dropped_updates = 0
        dropped_requeue_events = 0
        last_deadline_warning = 0.0
        last_requeue_warning = 0.0

        def _is_data_event(event_kind: object) -> bool:
            return str(event_kind) in {"snapshot", "increment"}

        def _track_dropped_update() -> None:
            nonlocal dropped_updates
            dropped_updates += 1
            if dropped_updates == 1 or dropped_updates % 100 == 0:
                _LOGGER.debug(
                    "Pominięto %s aktualizacji danych gRPC z powodu pełnej kolejki.",
                    dropped_updates,
                )

        def _track_dropped_requeue_event() -> None:
            nonlocal dropped_requeue_events
            dropped_requeue_events += 1
            if dropped_requeue_events == 1 or dropped_requeue_events % 25 == 0:
                _LOGGER.debug(
                    "Utracono %s zdarzeń sterujących gRPC podczas ponownego kolejkowania.",
                    dropped_requeue_events,
                )

        def _enqueue(
            kind: str,
            payload: Mapping[str, object] | None,
            *,
            drop_if_full: bool = False,
            deadline_seconds: float | None = 3.0,
            allow_when_stopped: bool = False,
        ) -> bool:
            nonlocal last_deadline_warning, last_requeue_warning
            deadline = None
            if deadline_seconds is not None:
                deadline = time.monotonic() + max(0.1, float(deadline_seconds))
            while True:
                if self.stop_event.is_set() and not allow_when_stopped:
                    return False
                try:
                    self.events_queue.put((kind, payload), timeout=0.1)
                    return True
                except queue.Full:
                    if drop_if_full:
                        _track_dropped_update()
                        return False
                    try:
                        queued_kind, queued_payload = self.events_queue.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        if _is_data_event(queued_kind):
                            self.events_queue.task_done()
                            _track_dropped_update()
                            continue
                        try:
                            self.events_queue.put_nowait((queued_kind, queued_payload))
                        except queue.Full:
                            self.events_queue.task_done()
                            _track_dropped_requeue_event()
                            now = time.monotonic()
                            if now - last_requeue_warning >= warn_rate_limit_seconds:
                                _LOGGER.warning(
                                    "Nie udało się odtworzyć zdarzenia gRPC %s po próbie priorytetyzacji; zdarzenie utracone.",
                                    queued_kind,
                                )
                                last_requeue_warning = now
                        else:
                            self.events_queue.task_done()
                    if deadline is not None and time.monotonic() >= deadline:
                        now = time.monotonic()
                        if now - last_deadline_warning >= warn_rate_limit_seconds:
                            _LOGGER.warning(
                                "Przekroczono deadline publikacji zdarzenia gRPC %s; pomijam event.",
                                kind,
                            )
                            last_deadline_warning = now
                        return False

        try:
            trading_pb2, trading_pb2_grpc = self._stubs_loader()
        except Exception as exc:
            _enqueue(
                "connection-error",
                {"attempt": 1, "message": str(exc)},
                allow_when_stopped=True,
            )
            _enqueue("done", None, allow_when_stopped=True)
            return

        request = trading_pb2.StreamDecisionsRequest(
            limit=max(0, int(self._limit)),
            skip_snapshot=False,
            poll_interval_seconds=1.0,
        )
        backoff = self._retry_base
        attempt = 0
        while not self.stop_event.is_set():
            attempt += 1
            channel = None
            try:
                if self._ssl_credentials is not None:
                    options = []
                    if self._authority_override:
                        options.append(("grpc.ssl_target_name_override", self._authority_override))
                    channel = self._grpc.secure_channel(
                        self._target,
                        self._ssl_credentials,
                        options=options or None,
                    )
                else:
                    channel = self._grpc.insecure_channel(self._target)
                ready_future = self._grpc.channel_ready_future(channel)
                ready_future.result(timeout=self._ready_timeout)
                stub = trading_pb2_grpc.RuntimeServiceStub(channel)
                if self._metadata:
                    stream = stub.StreamDecisions(request, metadata=self._metadata)
                else:
                    stream = stub.StreamDecisions(request)
                _enqueue("connected", {"attempt": attempt})
                backoff = self._retry_base
                terminated_normally = True
                for update in stream:
                    if self.stop_event.is_set():
                        terminated_normally = False
                        break
                    metrics_payload = self._cycle_metrics_serializer(
                        getattr(update, "cycle_metrics", None)
                    )
                    if update.HasField("snapshot"):
                        _enqueue(
                            "snapshot",
                            {
                                "records": [
                                    dict(entry.fields) for entry in update.snapshot.records
                                ],
                                "metrics": metrics_payload,
                            },
                            drop_if_full=True,
                            deadline_seconds=None,
                        )
                    elif update.HasField("increment"):
                        _enqueue(
                            "increment",
                            {
                                "record": dict(update.increment.record.fields),
                                "metrics": metrics_payload,
                            },
                            drop_if_full=True,
                            deadline_seconds=None,
                        )
                if terminated_normally and not self.stop_event.is_set():
                    _enqueue("stream-ended", {"attempt": attempt})
            except Exception as exc:  # pragma: no cover
                _enqueue("connection-error", {"attempt": attempt, "message": str(exc)})
            finally:
                if channel is not None:
                    try:
                        channel.close()
                    except Exception:
                        pass
            if self.stop_event.is_set():
                break
            sleep_seconds = min(self._retry_max, backoff)
            _enqueue("retrying", {"attempt": attempt, "sleep": float(sleep_seconds)})
            deadline = time.monotonic() + sleep_seconds
            while time.monotonic() < deadline:
                if self.stop_event.is_set():
                    break
                time.sleep(0.1)
            backoff = min(self._retry_max, backoff * self._retry_multiplier)
        _enqueue("done", None, allow_when_stopped=True)
