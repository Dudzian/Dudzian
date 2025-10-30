"""Uruchamia lokalny backend bota (AutoTrader + serwer gRPC) na potrzeby UI."""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from bot_core.api.server import build_local_runtime_context, LocalRuntimeServer
from bot_core.logging.config import install_metrics_logging_handler

_LOGGER = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )


def _write_ready_payload(
    address: str,
    *,
    ready_file: Optional[str],
    emit_stdout: bool,
    metrics_url: str | None = None,
) -> None:
    payload = {"event": "ready", "address": address, "pid": os.getpid()}
    if metrics_url:
        payload["metrics_url"] = metrics_url
    serialized = json.dumps(payload, ensure_ascii=False)
    if ready_file:
        Path(ready_file).expanduser().write_text(serialized, encoding="utf-8")
    if emit_stdout:
        print(serialized, flush=True)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/runtime.yaml", help="Ścieżka do pliku runtime.yaml")
    parser.add_argument("--entrypoint", help="Nazwa punktu wejścia z sekcji trading.entrypoints")
    parser.add_argument("--host", default="127.0.0.1", help="Adres nasłuchiwania serwera gRPC")
    parser.add_argument("--port", default=0, type=int, help="Port serwera gRPC (0 = automatyczny)")
    parser.add_argument("--log-level", default="INFO", help="Poziom logowania")
    parser.add_argument("--ready-file", help="Opcjonalny plik, do którego zostanie zapisany adres serwera")
    parser.add_argument(
        "--no-ready-stdout",
        action="store_true",
        help="Nie wypisuj komunikatu gotowości na stdout (użyteczne przy uruchomieniach z QProcess)",
    )
    parser.add_argument(
        "--manual-confirm",
        action="store_true",
        help="Nie aktywuj automatycznie auto-tradingu (wymaga ręcznego potwierdzenia w UI)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    try:
        context = build_local_runtime_context(
            config_path=args.config,
            entrypoint=args.entrypoint,
        )
    except Exception as exc:  # pragma: no cover - walidacja konfiguracji
        _LOGGER.error("Nie udało się zainicjalizować runtime: %s", exc)
        return 2

    observability_cfg = getattr(context.config, "observability", None)
    enable_metrics_handler = True
    if observability_cfg is not None:
        enable_metrics_handler = getattr(observability_cfg, "enable_log_metrics", True)
    if enable_metrics_handler:
        install_metrics_logging_handler()

    context.start(auto_confirm=not args.manual_confirm)
    server = LocalRuntimeServer(context, host=args.host, port=args.port)
    server.start()
    metrics_url = context.metrics_endpoint
    _write_ready_payload(
        server.address,
        ready_file=args.ready_file,
        emit_stdout=not args.no_ready_stdout,
        metrics_url=metrics_url,
    )

    stop_event = threading.Event()

    def _handle_signal(signum, frame):  # noqa: D401, ANN001
        del signum, frame
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:  # pragma: no cover - alternatywna obsługa sygnałów
        stop_event.set()
    finally:
        _LOGGER.info("Zatrzymywanie lokalnego runtime")
        try:
            server.stop(0.5)
        except Exception:  # pragma: no cover - defensywne
            _LOGGER.debug("Błąd podczas zatrzymywania serwera gRPC", exc_info=True)
        try:
            context.stop()
        except Exception:  # pragma: no cover - defensywne
            _LOGGER.debug("Błąd podczas zatrzymywania kontekstu runtime", exc_info=True)

    return 0


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    sys.exit(main())
