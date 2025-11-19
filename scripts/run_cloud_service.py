"""Uruchamia usługę gRPC Stage6 w trybie cloud za flagą."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
script_entry = str(SCRIPT_DIR)
try:
    sys.path.remove(script_entry)
except ValueError:
    pass
if script_entry not in sys.path:
    sys.path.append(script_entry)
repo_entry = str(REPO_ROOT)
if repo_entry not in sys.path:
    sys.path.insert(0, repo_entry)

from bot_core.cloud import CloudRuntimeService, load_cloud_server_config
from bot_core.security.cloud_flag import (
    CloudFlagValidationError,
    validate_runtime_cloud_flag,
)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/cloud/server.yaml",
        help="Ścieżka do konfiguracji cloud (YAML)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Poziom logowania",
    )
    parser.add_argument(
        "--ready-file",
        help="Opcjonalna ścieżka do zapisania payloadu gotowości",
    )
    parser.add_argument(
        "--emit-stdout",
        action="store_true",
        help="Wyemituj komunikat ready na stdout",
    )
    parser.add_argument(
        "--health-file",
        help="Ścieżka do pliku health probe (aktualizowana cyklicznie)",
    )
    return parser.parse_args(argv)


def _emit_ready(payload: Mapping[str, object], *, ready_file: str | None, emit_stdout: bool) -> None:
    serialized = json.dumps(payload, ensure_ascii=False)
    if ready_file:
        Path(ready_file).expanduser().write_text(serialized, encoding="utf-8")
    if emit_stdout:
        print(serialized, flush=True)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)
    try:
        config = load_cloud_server_config(args.config)
    except Exception as exc:
        logging.getLogger(__name__).error("Nie udało się załadować konfiguracji cloud: %s", exc)
        return 2

    try:
        validate_runtime_cloud_flag(config.runtime.config_path)
    except CloudFlagValidationError as exc:
        logging.getLogger(__name__).error(
            "Walidacja podpisanej flagi cloudowej nie powiodła się: %s",
            exc,
        )
        return 4

    service = CloudRuntimeService(
        config,
        ready_hook=lambda payload: _emit_ready(
            payload,
            ready_file=args.ready_file,
            emit_stdout=args.emit_stdout,
        ),
        health_probe_path=args.health_file,
    )

    stop_requested = False

    def _handle_signal(_signum, _frame) -> None:  # pragma: no cover - sygnały manualne
        nonlocal stop_requested
        stop_requested = True
        service.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    try:
        service.start()
    except Exception as exc:  # pragma: no cover - bootstrap
        logging.getLogger(__name__).exception("Nie udało się uruchomić CloudRuntimeService: %s", exc)
        return 3

    try:
        service.wait()
    except KeyboardInterrupt:  # pragma: no cover - fallback
        service.stop()

    if not stop_requested:
        service.stop()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main(sys.argv[1:]))
