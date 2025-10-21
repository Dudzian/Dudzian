"""Single-host offline runtime daemon exposing a REST/IPC interface.

The Qt shell can launch this daemon to work entirely offline without a gRPC
backend.  The daemon reuses the in-memory dataset from
``bot_core.testing.trading_stub_server`` and exposes a minimal HTTP/JSON API
for market data history, risk snapshots as well as automation helpers.  The
protocol is intentionally tiny so it can be consumed from QML/Qt Network
stack without pulling additional dependencies.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterable, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from google.protobuf.timestamp_pb2 import Timestamp

from bot_core.testing.trading_stub_server import (
    InMemoryTradingDataset,
    build_default_dataset,
    load_dataset_from_yaml,
)

_LOG = logging.getLogger(__name__)

DEFAULT_BIND = ("127.0.0.1", 58081)


class OfflineRuntimeState:
    """Holds current runtime state shared across HTTP handlers."""

    def __init__(self, dataset: InMemoryTradingDataset) -> None:
        self._dataset = dataset
        self._strategy: Dict[str, Any] = {}
        self._automation_running = False
        self._automation_enabled = False
        self._automation_last_started: Optional[float] = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Market data helpers
    # ------------------------------------------------------------------
    def history(
        self, exchange: Optional[str], symbol: Optional[str], granularity: Optional[str], limit: int
    ) -> list[dict[str, Any]]:
        with self._lock:
            key = self._match_key(exchange, symbol, granularity)
            candles = list(self._dataset.history.get(key, []))
            if limit > 0:
                candles = candles[-limit:]
            return [self._candle_to_dict(item) for item in candles]

    def latest(
        self, exchange: Optional[str], symbol: Optional[str], granularity: Optional[str]
    ) -> Optional[dict[str, Any]]:
        history = self.history(exchange, symbol, granularity, limit=1)
        return history[-1] if history else None

    def risk_snapshot(self) -> Optional[dict[str, Any]]:
        with self._lock:
            snapshots = next(iter(self._dataset.risk_states.values()), [])
            if not snapshots:
                return None
            return self._risk_to_dict(snapshots[-1])

    def performance_guard(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._dataset.performance_guard)

    # ------------------------------------------------------------------
    # Automation helpers
    # ------------------------------------------------------------------
    def status_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": "offline",
                "automation": {
                    "running": self._automation_running,
                    "auto_enabled": self._automation_enabled,
                    "last_started": self._automation_last_started,
                },
                "strategy": dict(self._strategy),
            }

    def update_strategy(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._strategy = dict(payload)
            return {"status": "ok", "strategy": dict(self._strategy)}

    def set_automation(self, running: bool, *, auto: Optional[bool] = None) -> dict[str, Any]:
        with self._lock:
            self._automation_running = running
            if auto is not None:
                self._automation_enabled = auto
            if running:
                self._automation_last_started = time.time()
            return self.status_payload()["automation"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _match_key(
        self, exchange: Optional[str], symbol: Optional[str], granularity: Optional[str]
    ) -> Tuple[str, str, str]:
        if self._dataset.history:
            default_key = next(iter(self._dataset.history.keys()))
        else:
            default_key = (exchange or "*", symbol or "*", granularity or "*")
        return (
            (exchange or default_key[0] or "*").upper(),
            symbol or default_key[1],
            granularity or default_key[2],
        )

    @staticmethod
    def _timestamp_to_ms(ts: Timestamp) -> int:
        return int(ts.seconds * 1000 + ts.nanos / 1_000_000)

    @classmethod
    def _candle_to_dict(cls, candle: Any) -> dict[str, Any]:
        return {
            "timestamp_ms": cls._timestamp_to_ms(candle.open_time),
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
            "closed": bool(candle.closed),
            "sequence": int(getattr(candle, "sequence", 0)),
            "exchange": getattr(candle.instrument, "exchange", ""),
            "symbol": getattr(candle.instrument, "symbol", ""),
            "granularity": getattr(candle.granularity, "iso8601_duration", ""),
        }

    @classmethod
    def _risk_to_dict(cls, state: Any) -> dict[str, Any]:
        exposures = [
            {
                "code": limit.code,
                "max_value": getattr(limit, "max_value", 0.0),
                "current_value": getattr(limit, "current_value", 0.0),
                "threshold_value": getattr(limit, "threshold_value", 0.0),
            }
            for limit in getattr(state, "limits", [])
        ]
        generated = getattr(state, "generated_at", None)
        generated_ms = cls._timestamp_to_ms(generated) if generated else None
        profile_name = getattr(state, "profile", "")
        if hasattr(profile_name, "name"):
            profile_value = profile_name.name
        else:
            profile_value = int(profile_name)
        return {
            "profile": profile_value,
            "portfolio_value": getattr(state, "portfolio_value", 0.0),
            "current_drawdown": getattr(state, "current_drawdown", 0.0),
            "max_daily_loss": getattr(state, "max_daily_loss", 0.0),
            "used_leverage": getattr(state, "used_leverage", 0.0),
            "generated_at_ms": generated_ms,
            "exposures": exposures,
        }


class _RequestContext(BaseHTTPRequestHandler):
    server_version = "OfflineRuntimeHTTP/1.0"
    state: OfflineRuntimeState

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        if parsed.path == "/v1/status":
            return self._send_json(self.server.state.status_payload())
        if parsed.path == "/v1/market-data/history":
            return self._handle_history(parsed.query)
        if parsed.path == "/v1/market-data/latest":
            return self._handle_latest(parsed.query)
        if parsed.path == "/v1/risk/state":
            snapshot = self.server.state.risk_snapshot()
            if snapshot is None:
                return self._send_json({"snapshot": None}, status=HTTPStatus.NO_CONTENT)
            return self._send_json({"snapshot": snapshot})
        if parsed.path == "/v1/performance-guard":
            return self._send_json({"guard": self.server.state.performance_guard()})
        return self._send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        payload = self._read_json_body()
        if parsed.path == "/v1/strategy":
            if not isinstance(payload, dict):
                return self._send_error(HTTPStatus.BAD_REQUEST, "Invalid payload")
            result = self.server.state.update_strategy(payload)
            return self._send_json(result)
        if parsed.path == "/v1/automation/start":
            auto = bool(payload.get("auto", True)) if isinstance(payload, dict) else True
            result = self.server.state.set_automation(True, auto=auto)
            return self._send_json({"automation": result})
        if parsed.path == "/v1/automation/stop":
            result = self.server.state.set_automation(False)
            return self._send_json({"automation": result})
        return self._send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401 - silenced verbose HTTP handler logs
        _LOG.debug("HTTP: " + fmt, *args)

    def _handle_history(self, query: str) -> None:
        params = parse_qs(query)
        limit = int(params.get("limit", ["500"])[0])
        candles = self.server.state.history(
            params.get("exchange", [None])[0],
            params.get("symbol", [None])[0],
            params.get("granularity", [None])[0],
            limit,
        )
        self._send_json({"candles": candles})

    def _handle_latest(self, query: str) -> None:
        params = parse_qs(query)
        candle = self.server.state.latest(
            params.get("exchange", [None])[0],
            params.get("symbol", [None])[0],
            params.get("granularity", [None])[0],
        )
        if candle is None:
            self._send_json({"candle": None}, status=HTTPStatus.NO_CONTENT)
        else:
            self._send_json({"candle": candle})

    def _read_json_body(self) -> Any:
        content_length = self.headers.get("Content-Length")
        if not content_length:
            return {}
        try:
            length = int(content_length)
        except ValueError:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _send_json(self, payload: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        payload = {"error": message, "status": status.value}
        self._send_json(payload, status=status)


class _Server(ThreadingHTTPServer):
    state: OfflineRuntimeState


def _load_dataset(path: Optional[str]) -> InMemoryTradingDataset:
    if not path:
        _LOG.debug("Using default dataset for offline daemon")
        return build_default_dataset()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file {path} does not exist")
    return load_dataset_from_yaml(path)


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline runtime REST daemon")
    parser.add_argument("--host", default=DEFAULT_BIND[0])
    parser.add_argument("--port", type=int, default=DEFAULT_BIND[1])
    parser.add_argument("--dataset", help="Optional YAML dataset override")
    parser.add_argument("--ready-file", help="Write JSON with address to this file when ready")
    parser.add_argument(
        "--no-stdout-ready",
        action="store_true",
        help="Do not emit JSON ready payload on stdout",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--lifetime",
        type=float,
        default=0.0,
        help="Optional lifetime in seconds (0 = infinite)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _announce_ready(address: str, ready_file: Optional[str], emit_stdout: bool) -> None:
    payload = json.dumps({"event": "ready", "address": address, "pid": os.getpid()})
    if ready_file:
        with open(ready_file, "w", encoding="utf-8") as handle:
            handle.write(payload)
    if emit_stdout:
        print(payload, flush=True)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    try:
        dataset = _load_dataset(args.dataset)
    except Exception as exc:  # pragma: no cover - configuration error
        _LOG.error("Failed to load dataset: %s", exc)
        return 2

    state = OfflineRuntimeState(dataset)
    server = _Server((args.host, args.port), _RequestContext)
    server.state = state

    stop_event = threading.Event()

    def _handle_signal(_signum, _frame) -> None:
        stop_event.set()
        server.shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    actual_host, actual_port = server.server_address
    address = f"http://{actual_host}:{actual_port}"
    _LOG.info("Offline daemon listening on %s", address)
    try:
        _announce_ready(address, args.ready_file, not args.no_stdout_ready)
    except Exception as exc:  # pragma: no cover
        _LOG.warning("Unable to announce ready payload: %s", exc)

    lifetime_deadline = time.monotonic() + args.lifetime if args.lifetime > 0 else None

    thread = threading.Thread(target=server.serve_forever, name="offline-daemon", daemon=True)
    thread.start()

    try:
        while not stop_event.wait(timeout=0.5):
            if lifetime_deadline and time.monotonic() >= lifetime_deadline:
                _LOG.info("Lifetime expired – shutting down")
                break
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
