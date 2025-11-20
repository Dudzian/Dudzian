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
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable

from bot_core.api.server import LocalRuntimeGateway, build_local_runtime_context
from bot_core.runtime.cloud_profiles import (
    RuntimeCloudClientSelection,
    resolve_runtime_cloud_client,
)
from bot_core.security.cloud_flag import (
    CloudFlagValidationError,
    validate_runtime_cloud_flag,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class _GatewayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: Any | None = None
    method: str
    params: Dict[str, Any] = Field(default_factory=dict)


class _TokenBucket:
    """Prosty limiter tokenów do kontroli tempa przyjmowania żądań."""

    def __init__(self, capacity: int, refill_rate_per_sec: float) -> None:
        self.capacity = capacity
        self.tokens = float(capacity)
        self.refill_rate = float(refill_rate_per_sec)
        self._last_refill = time.monotonic()

    def consume(self, amount: float = 1.0) -> bool:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False


class _GatewayMetrics:
    def __init__(self) -> None:
        self.invalid_json = 0
        self.invalid_request = 0
        self.rate_limited = 0
        self.dispatch_errors = 0
        self.processed = 0
        self._last_log_at = time.monotonic()

    def _log_if_needed(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_log_at
        if elapsed < 10:
            return
        rate = self.invalid_json / elapsed if elapsed else 0.0
        _LOG.info(
            "Metryki gatewaya: invalid_json=%d invalid_request=%d rate_limited=%d dispatch_errors=%d processed=%d invalid_json_rate/s=%.3f",
            self.invalid_json,
            self.invalid_request,
            self.rate_limited,
            self.dispatch_errors,
            self.processed,
            rate,
        )
        self.invalid_json = 0
        self.invalid_request = 0
        self.rate_limited = 0
        self.dispatch_errors = 0
        self.processed = 0
        self._last_log_at = now

    def increment(self, attr: str) -> None:
        if hasattr(self, attr):
            setattr(self, attr, getattr(self, attr) + 1)
        self._log_if_needed()


def _error_payload(identifier: Any, code: str, message: str) -> Dict[str, Any]:
    return {"id": identifier, "error": {"code": code, "message": message}}


def _process_request(
    raw: str,
    gateway: LocalRuntimeGateway,
    metrics: _GatewayMetrics,
    rate_limiter: _TokenBucket,
) -> Dict[str, Any] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        metrics.increment("invalid_json")
        return {"error": {"code": "invalid-json", "message": str(exc)}}
    if not isinstance(parsed, dict):
        metrics.increment("invalid_request")
        return _error_payload(None, "invalid-request", "Payload musi być obiektem JSON")
    try:
        envelope = _GatewayRequest.model_validate(parsed)
    except ValidationError as exc:
        metrics.increment("invalid_request")
        return _error_payload(parsed.get("id"), "invalid-request", exc.errors()[0]["msg"])

    if not rate_limiter.consume():
        metrics.increment("rate_limited")
        return _error_payload(envelope.id, "rate-limit", "Przekroczono limit zapytań")

    try:
        result = gateway.dispatch(envelope.method, envelope.params)
    except Exception as exc:  # pragma: no cover - defensive fallback
        metrics.increment("dispatch_errors")
        _LOG.exception("Błąd podczas obsługi metody %s", envelope.method)
        return _error_payload(envelope.id, "dispatch-error", str(exc))

    metrics.increment("processed")
    return {"id": envelope.id, "result": result}


_LOG = logging.getLogger(__name__)


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
        "--enable-cloud-runtime",
        action="store_true",
        help="Przełącza UI na tryb cloudowy wskazany w runtime.yaml",
    )
    return parser.parse_args(argv)


def _emit(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _emit_exit(code: int, message: str, *, details: Dict[str, Any] | None = None) -> int:
    payload: Dict[str, Any] = {"event": "error", "code": code, "message": message}
    if details:
        payload["details"] = details
    _emit(payload)
    return code


def _cloud_ready_event(selection: RuntimeCloudClientSelection) -> Dict[str, Any]:
    entrypoint = selection.profile.entrypoint or selection.client.fallback_entrypoint
    return {
        "event": "ready",
        "version": "cloud",
        "cloud": {
            "profile": selection.profile_name,
            "mode": selection.profile.mode,
            "target": selection.client.address,
            "entrypoint": entrypoint,
        },
    }


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    cloud_selection: RuntimeCloudClientSelection | None = None
    if getattr(args, "enable_cloud_runtime", False):
        try:
            validate_runtime_cloud_flag(Path(args.config))
        except CloudFlagValidationError as exc:
            _LOG.error("Walidacja flagi cloudowej nie powiodła się: %s", exc)
            return _emit_exit(
                4,
                "Walidacja flagi cloudowej nie powiodła się",
                details={"reason": str(exc)},
            )
        try:
            cloud_selection = resolve_runtime_cloud_client(Path(args.config))
        except Exception as exc:  # pragma: no cover - diagnostyka konfiguracji
            _LOG.error("Nie udało się wczytać konfiguracji cloud: %s", exc)
            return _emit_exit(
                2,
                "Nie udało się wczytać konfiguracji cloud",
                details={"reason": str(exc)},
            )
        if cloud_selection is None:
            _LOG.error("Flaga cloud aktywna, ale runtime.yaml nie zawiera profilu remote")
            return _emit_exit(3, "Brak profilu remote w runtime.yaml", details={})
        _emit(_cloud_ready_event(cloud_selection))
        return 0

    try:
        context = build_local_runtime_context(
            config_path=Path(args.config),
            entrypoint=args.entrypoint,
        )
    except Exception as exc:  # pragma: no cover - bootstrap errors are rare
        _LOG.error("Nie udało się zbudować kontekstu runtime: %s", exc)
        return _emit_exit(
            2,
            "Nie udało się zbudować kontekstu runtime",
            details={"reason": str(exc)},
        )

    with context:
        gateway = LocalRuntimeGateway(context)
        metrics = _GatewayMetrics()
        rate_limiter = _TokenBucket(capacity=30, refill_rate_per_sec=10)
        _emit({"event": "ready", "version": context.version})

        stop = False

        def _signal_handler(*_args: object) -> None:  # pragma: no cover - manual interrupt
            nonlocal stop
            stop = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, _signal_handler)

        for line in sys.stdin:
            if stop:
                break
            response = _process_request(line.strip(), gateway, metrics, rate_limiter)
            if response is None:
                continue
            _emit(response)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main(sys.argv[1:]))

