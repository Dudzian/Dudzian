"""Lightweight JSON-RPC gateway for the desktop shell.

This module bootstraps a :class:`~bot_core.api.server.LocalRuntimeContext`
and exposes a small set of blocking RPC-style calls using newline-delimited
JSON payloads over STDIN/STDOUT.  The protocol is intentionally simple so
that the Qt shell can communicate with the Python runtime without spinning up
an additional gRPC server instance.  Each line received on stdin must contain
valid JSON with at least ``id`` and ``method`` fields.  Responses mirror that
shape and include either ``result`` or ``error`` objects.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import queue
import signal
import sys
import selectors
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import yaml

from bot_core.api.server import LocalRuntimeGateway, build_local_runtime_context


_LOG = logging.getLogger(__name__)

_DEFAULT_POLL_INTERVAL = 0.05
_DEFAULT_MAX_WORKERS = 4
_DEFAULT_QUEUE_SIZE = 64
_SERVER_BUSY_MESSAGE = "server-busy"
_EOF = object()


@dataclass
class _PendingRequest:
    future: Future
    method: str
    submitted_at: float
    deadline: Optional[float]
    started_at: Optional[float] = None


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/runtime.yaml",
        help="Ścieżka do pliku konfiguracyjnego runtime",
    )
    parser.add_argument(
        "--entrypoint",
        help="Nazwa punktu wejścia z pliku runtime (domyślnie z konfiguracji)",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Poziom logowania",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maksymalna liczba wątków obsługujących żądania JSON-RPC",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        help="Maksymalna liczba równoczesnych żądań w kolejce JSON-RPC",
    )
    return parser.parse_args(argv)


def _load_gateway_settings(config_path: Path) -> Dict[str, Any]:
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _LOG.debug("Plik konfiguracji %s nie istnieje - używam wartości domyślnych", config_path)
        return {}
    except Exception as exc:  # pragma: no cover - diagnostyka IO/YAML
        _LOG.warning("Nie udało się odczytać konfiguracji runtime (%s): %s", config_path, exc)
        return {}

    if not isinstance(raw, dict):
        return {}

    candidates: list[Dict[str, Any]] = []
    direct = raw.get("local_gateway")
    if isinstance(direct, dict):
        candidates.append(direct)

    runtime_section = raw.get("runtime")
    if isinstance(runtime_section, dict):
        runtime_gateway = runtime_section.get("local_gateway")
        if isinstance(runtime_gateway, dict):
            candidates.append(runtime_gateway)

    core_section = raw.get("core")
    if isinstance(core_section, dict):
        core_gateway = core_section.get("local_gateway")
        if isinstance(core_gateway, dict):
            candidates.append(core_gateway)

    return candidates[0] if candidates else {}


def _coerce_positive_int(value: Any, default: int, name: str) -> int:
    if value is None:
        return default
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        _LOG.warning("Nieprawidłowa wartość %s=%r - używam %d", name, value, default)
        return default
    if coerced <= 0:
        _LOG.warning("Wartość %s musi być dodatnia (otrzymano %d) - używam %d", name, coerced, default)
        return default
    return coerced


def _emit(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


class _StdinPoller:
    """Cross-platform detektor danych dostępnych na STDIN."""

    def __init__(
        self,
        stdin: Any,
        poll_interval: float,
        stop_event: threading.Event,
    ) -> None:
        self._stdin = stdin
        self._poll_interval = poll_interval
        self._stop_event = stop_event
        self._backend = "selector"
        self._selector: selectors.BaseSelector | None = None
        self._queue: "queue.Queue[Any]" | None = None
        self._reader: threading.Thread | None = None

        try:
            selector = selectors.DefaultSelector()
            selector.register(stdin, selectors.EVENT_READ)
        except (OSError, ValueError):
            if "selector" in locals():
                with contextlib.suppress(Exception):
                    selector.close()
            self._backend = "thread"
            self._selector = None
            self._queue = queue.Queue()
            self._reader = threading.Thread(
                target=self._reader_loop,
                name="json-rpc-stdin-reader",
                daemon=True,
            )
            self._reader.start()
        else:
            self._selector = selector

    @property
    def backend(self) -> str:
        return self._backend

    def poll(self) -> Optional[str]:
        if self._selector is not None:
            try:
                events = self._selector.select(self._poll_interval)
            except (OSError, ValueError):
                return ""
            if not events:
                return None
            return self._stdin.readline()

        assert self._queue is not None
        try:
            item = self._queue.get(timeout=self._poll_interval)
        except queue.Empty:
            return None

        if item is _EOF:
            return ""

        return item

    def close(self) -> None:
        if self._selector is not None:
            with contextlib.suppress(Exception):
                self._selector.unregister(self._stdin)
            with contextlib.suppress(Exception):
                self._selector.close()
            self._selector = None
        if self._reader is not None and self._queue is not None:
            self._stop_event.set()
            self._queue.put(_EOF)
            self._reader.join(timeout=0.1)
            self._reader = None

    def _reader_loop(self) -> None:
        assert self._queue is not None
        while not self._stop_event.is_set():
            try:
                line = self._stdin.readline()
            except Exception as exc:  # pragma: no cover - defensywne logowanie
                _LOG.debug("Błąd odczytu STDIN: %s", exc)
                self._queue.put(_EOF)
                return
            if line == "":
                self._queue.put(_EOF)
                return
            self._queue.put(line)
        self._queue.put(_EOF)


class JsonRpcServer:
    """Koordynuje wykonywanie żądań JSON-RPC w tle."""

    def __init__(
        self,
        gateway: LocalRuntimeGateway,
        *,
        max_workers: int = _DEFAULT_MAX_WORKERS,
        max_queue_size: int = _DEFAULT_QUEUE_SIZE,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        emit: Callable[[Dict[str, Any]], None] = _emit,
    ) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")

        self._gateway = gateway
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._queue_size = max_queue_size
        self._results: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max_queue_size)
        self._overflow: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._pending: dict[Any, _PendingRequest] = {}
        self._pending_lock = threading.Lock()
        self._capacity = threading.BoundedSemaphore(max_queue_size)
        self._poll_interval = poll_interval
        self._stopping = threading.Event()
        self._stopping_notified = threading.Event()
        self._emit_callback = emit
        self._emit_lock = threading.Lock()
        self._flush_requested = threading.Event()
        self._stats_lock = threading.Lock()
        self._max_pending_seen = 0
        self._busy_rejections = 0
        self._queue_wait_total = 0.0
        self._queue_wait_samples = 0
        self._max_queue_wait = 0.0
        self._completed = 0

    def submit(self, request: Dict[str, Any]) -> None:
        identifier = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}
        timeout_ms = request.get("timeout_ms")
        if timeout_ms is None and "timeout" in request:
            timeout_ms = request.get("timeout")
        cancel_requested = bool(request.get("cancel"))

        if identifier is None:
            raise ValueError("missing-request-id")

        if cancel_requested:
            self.cancel(identifier)
            return

        if not isinstance(method, str):
            raise ValueError("missing-method")

        deadline = None
        if timeout_ms:
            try:
                timeout_ms = float(timeout_ms)
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid-timeout") from exc
            if timeout_ms < 0:
                timeout_ms = 0.0
            if timeout_ms > 0:
                deadline = time.monotonic() + timeout_ms / 1000.0

        _LOG.debug("Submitting request %s (%s) with timeout %s", identifier, method, timeout_ms)

        if not self._capacity.acquire(blocking=False):
            pending_now = self._pending_count()
            self._record_busy_rejection()
            _LOG.warning(
                "Rejecting request %s (%s) due to capacity limit (pending=%d, max=%d)",
                identifier,
                method,
                pending_now,
                self._queue_size,
            )
            self._deliver(
                {
                    "id": identifier,
                    "error": {"message": _SERVER_BUSY_MESSAGE},
                    "retry_after_ms": int(self._poll_interval * 1000),
                }
            )
            return

        submitted_at = time.monotonic()
        try:
            future = self._executor.submit(self._dispatch, identifier, method, params)
        except Exception:
            self._release_capacity()
            raise

        future.add_done_callback(lambda fut, req_id=identifier: self._on_complete(req_id, fut))
        with self._pending_lock:
            if identifier in self._pending:
                self._release_capacity()
                future.cancel()
                raise ValueError("duplicate-request")
            self._pending[identifier] = _PendingRequest(
                future=future,
                method=method,
                submitted_at=submitted_at,
                deadline=deadline,
            )
            pending_size = len(self._pending)
        self._record_pending_depth(pending_size)
        if future.done():
            self._on_complete(identifier, future)

    def cancel(self, identifier: Any) -> None:
        with self._pending_lock:
            state = self._pending.pop(identifier, None)
        if state is None:
            _LOG.debug("Cancel request for %s ignored - not found", identifier)
            self._deliver({"id": identifier, "error": {"message": "unknown-request"}})
            return
        cancelled = state.future.cancel()
        _LOG.info("Cancelled request %s (%s) - %s", identifier, state.method, cancelled)
        payload = {
            "id": identifier,
            "error": {"message": "cancelled"},
            "cancelled": True,
        }
        self._release_capacity()
        self._deliver(payload)
        self._record_completion()

    def _dispatch(self, identifier: Any, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        started_at = time.monotonic()
        method_name = method
        with self._pending_lock:
            state = self._pending.get(identifier)
            if state is not None:
                state.started_at = started_at
                method_name = state.method
                submitted_at = state.submitted_at
            else:
                submitted_at = started_at
        self._record_queue_wait(method_name, started_at - submitted_at)
        return self._gateway.dispatch(method, params)

    def _on_complete(self, identifier: Any, future: Future) -> None:
        with self._pending_lock:
            state = self._pending.pop(identifier, None)
        if state is None:
            return
        try:
            result = future.result()
        except Exception as exc:  # pragma: no cover - log details in emit
            _LOG.warning("Request %s (%s) failed: %s", identifier, state.method, exc, exc_info=True)
            payload = {"id": identifier, "error": {"message": str(exc)}}
        else:
            duration = time.monotonic() - state.submitted_at
            _LOG.debug("Request %s (%s) completed in %.3fs", identifier, state.method, duration)
            payload = {"id": identifier, "result": result}
        self._deliver(payload)
        self._release_capacity()
        self._record_completion()

    def _pending_count(self) -> int:
        with self._pending_lock:
            return len(self._pending)

    def _cancel_remaining(self, reason: str) -> None:
        with self._pending_lock:
            remaining = list(self._pending.items())
            self._pending.clear()

        for identifier, state in remaining:
            cancelled = state.future.cancel()
            _LOG.warning(
                "Request %s (%s) aborted during shutdown (cancelled=%s)",
                identifier,
                state.method,
                cancelled,
            )
            self._deliver(
                {
                    "id": identifier,
                    "error": {"message": reason},
                    "cancelled": True,
                }
            )
            self._release_capacity()
            self._record_completion()

    def _drain_pending_results(self) -> None:
        """Flushuj wszystkie wyniki do momentu wyczyszczenia `_pending`."""

        shutdown_deadline = time.monotonic() + 5.0
        while True:
            self.flush_ready()
            if self._pending_count() == 0:
                break
            try:
                payload = self._results.get(timeout=self._poll_interval)
            except queue.Empty:
                try:
                    payload = self._overflow.get_nowait()
                except queue.Empty:
                    if time.monotonic() >= shutdown_deadline:
                        self._cancel_remaining("shutdown")
                        self.flush_ready()
                        break
                    continue
            self._emit_payload(payload)

    def flush_ready(self) -> None:
        while True:
            drained = False

            try:
                payload = self._overflow.get_nowait()
            except queue.Empty:
                pass
            else:
                drained = True
                self._emit_payload(payload)

            try:
                payload = self._results.get_nowait()
            except queue.Empty:
                pass
            else:
                drained = True
                self._emit_payload(payload)

            if not drained:
                break

    def enforce_timeouts(self) -> None:
        now = time.monotonic()
        expired: list[Any] = []
        with self._pending_lock:
            for identifier, state in list(self._pending.items()):
                if state.deadline is not None and now >= state.deadline:
                    expired.append(identifier)
        for identifier in expired:
            with self._pending_lock:
                state = self._pending.pop(identifier, None)
            if state is None:
                continue
            cancelled = state.future.cancel()
            _LOG.warning(
                "Request %s (%s) timed out after %.3fs (cancelled=%s)",
                identifier,
                state.method,
                now - state.submitted_at,
                cancelled,
            )
            self._deliver(
                {
                    "id": identifier,
                    "error": {"message": "timeout"},
                    "timeout": True,
                }
            )
            self._release_capacity()
            self._record_completion()

    def run(self) -> None:
        stdin = sys.stdin
        poller = _StdinPoller(stdin, self._poll_interval, self._stopping)
        try:
            while not self._stopping.is_set():
                self.enforce_timeouts()
                self.flush_ready()
                if self._flush_requested.is_set():
                    self._flush_requested.clear()
                    continue
                line = poller.poll()
                if line is None:
                    continue
                if line == "":
                    break
                raw = line.strip()
                if not raw:
                    continue
                try:
                    request = json.loads(raw)
                except json.JSONDecodeError as exc:
                    self._emit_payload({"error": {"message": f"invalid-json: {exc}"}})
                    continue
                if not isinstance(request, dict):
                    self._emit_payload({"error": {"message": "invalid-request"}})
                    continue
                try:
                    self.submit(request)
                except Exception as exc:  # pragma: no cover - defensive
                    _LOG.exception("Błąd podczas dodawania żądania")
                    identifier = request.get("id")
                    self._emit_payload({"id": identifier, "error": {"message": str(exc)}})

        finally:
            poller.close()
            self._stopping.set()
            self._notify_runtime_stopping()
            self.enforce_timeouts()
            self._drain_pending_results()
            self._executor.shutdown(wait=True, cancel_futures=True)
            self.flush_ready()
            self._log_statistics()

    def stop(self) -> None:
        self._stopping.set()
        self._notify_runtime_stopping()

    def _notify_runtime_stopping(self) -> None:
        if self._stopping_notified.is_set():
            return
        self._stopping_notified.set()
        self._emit_payload({"event": "runtime-stopping"})

    def _deliver(self, payload: Dict[str, Any]) -> None:
        try:
            self._results.put_nowait(payload)
        except queue.Full:
            _LOG.warning("Result queue full - queueing payload for overflow flush")
            self._overflow.put(payload)
            self._flush_requested.set()

    def _release_capacity(self) -> None:
        with contextlib.suppress(ValueError):
            self._capacity.release()

    def _record_pending_depth(self, pending_size: int) -> None:
        with self._stats_lock:
            if pending_size > self._max_pending_seen:
                self._max_pending_seen = pending_size

    def _record_busy_rejection(self) -> None:
        with self._stats_lock:
            self._busy_rejections += 1

    def _record_queue_wait(self, method: str, wait_duration: float) -> None:
        if wait_duration <= 0:
            return
        with self._stats_lock:
            self._queue_wait_total += wait_duration
            self._queue_wait_samples += 1
            if wait_duration > self._max_queue_wait:
                self._max_queue_wait = wait_duration
        _LOG.debug("Request %s queued for %.3fs", method, wait_duration)

    def _record_completion(self) -> None:
        with self._stats_lock:
            self._completed += 1

    def _log_statistics(self) -> None:
        with self._stats_lock:
            completed = self._completed
            busy = self._busy_rejections
            max_pending = self._max_pending_seen
            queue_samples = self._queue_wait_samples
            queue_total = self._queue_wait_total
            max_queue_wait = self._max_queue_wait
        if completed or busy:
            average_wait = queue_total / queue_samples if queue_samples else 0.0
            _LOG.info(
                "JSON-RPC telemetry: completed=%d busy=%d max_pending=%d avg_queue_wait=%.4fs max_queue_wait=%.4fs",
                completed,
                busy,
                max_pending,
                average_wait,
                max_queue_wait,
            )

    def _emit_payload(self, payload: Dict[str, Any]) -> None:
        with self._emit_lock:
            self._emit_callback(payload)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    config_path = Path(args.config)
    gateway_settings = _load_gateway_settings(config_path)

    configured_max_workers = args.max_workers if args.max_workers is not None else gateway_settings.get("max_workers")
    configured_queue_size = args.queue_size if args.queue_size is not None else (
        gateway_settings.get("queue_size") or gateway_settings.get("max_queue_size")
    )

    max_workers = _coerce_positive_int(configured_max_workers, _DEFAULT_MAX_WORKERS, "max_workers")
    queue_size = _coerce_positive_int(configured_queue_size, _DEFAULT_QUEUE_SIZE, "queue_size")

    try:
        context = build_local_runtime_context(
            config_path=config_path,
            entrypoint=args.entrypoint,
        )
    except Exception as exc:  # pragma: no cover - bootstrap errors are rare
        _LOG.error("Nie udało się zbudować kontekstu runtime: %s", exc)
        return 2

    with context:
        gateway = LocalRuntimeGateway(context)
        _LOG.info(
            "Uruchamiam JsonRpcServer z max_workers=%d i queue_size=%d", max_workers, queue_size
        )
        server = JsonRpcServer(gateway, max_workers=max_workers, max_queue_size=queue_size)
        _emit({"event": "ready", "version": context.version})

        def _signal_handler(*_args: object) -> None:  # pragma: no cover - manual interrupt
            _LOG.info("Odebrano sygnał zatrzymania - kończę pętlę JSON RPC")
            server.stop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, _signal_handler)

        server.run()

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main(sys.argv[1:]))

