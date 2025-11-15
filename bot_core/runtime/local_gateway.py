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
from pathlib import Path
from typing import Any, Dict, Iterable

from bot_core.api.server import LocalRuntimeGateway, build_local_runtime_context
from bot_core.runtime.cloud_profiles import (
    RuntimeCloudClientSelection,
    resolve_runtime_cloud_client,
)


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
            cloud_selection = resolve_runtime_cloud_client(Path(args.config))
        except Exception as exc:  # pragma: no cover - diagnostyka konfiguracji
            _LOG.error("Nie udało się wczytać konfiguracji cloud: %s", exc)
            return 2
        if cloud_selection is None:
            _LOG.error("Flaga cloud aktywna, ale runtime.yaml nie zawiera profilu remote")
            return 3
        _emit(_cloud_ready_event(cloud_selection))
        return 0

    try:
        context = build_local_runtime_context(
            config_path=Path(args.config),
            entrypoint=args.entrypoint,
        )
    except Exception as exc:  # pragma: no cover - bootstrap errors are rare
        _LOG.error("Nie udało się zbudować kontekstu runtime: %s", exc)
        return 2

    with context:
        gateway = LocalRuntimeGateway(context)
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
            raw = line.strip()
            if not raw:
                continue
            try:
                request = json.loads(raw)
            except json.JSONDecodeError as exc:
                _emit({"error": {"message": f"invalid-json: {exc}"}})
                continue
            if not isinstance(request, dict):
                _emit({"error": {"message": "invalid-request"}})
                continue
            identifier = request.get("id")
            method = request.get("method")
            params = request.get("params") or {}
            if not isinstance(method, str):
                _emit({"id": identifier, "error": {"message": "missing-method"}})
                continue
            try:
                result = gateway.dispatch(method, params)
            except Exception as exc:  # pragma: no cover - defensive fallback
                _LOG.exception("Błąd podczas obsługi metody %s", method)
                _emit({"id": identifier, "error": {"message": str(exc)}})
                continue
            _emit({"id": identifier, "result": result})

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main(sys.argv[1:]))

