"""Lightweight local gRPC service exposing the bot_core trading API.

This module spins up the in-repo testing stub so that the Qt shell can talk to
an isolated backend without relying on external infrastructure.  It accepts a
YAML dataset definition compatible with ``bot_core.testing.trading_stub_server``
and publishes the bound address on stdout in a structured form that the UI can
parse.
"""
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

from bot_core.testing.trading_stub_server import (
    InMemoryTradingDataset,
    TradingStubServer,
    build_default_dataset,
    load_dataset_from_yaml,
)

_LOG = logging.getLogger(__name__)


class _StopEvent:
    """Signal handler helper that cooperates with threads."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def set(self, *_args) -> None:  # pragma: no cover - signature dictated by signal handler
        self._event.set()

    def wait(self, timeout: Optional[float]) -> bool:
        return self._event.wait(timeout)

    @property
    def is_set(self) -> bool:
        return self._event.is_set()


def _load_dataset(path: Optional[str]) -> InMemoryTradingDataset:
    if not path:
        _LOG.debug("Using default in-memory dataset for stub server")
        return build_default_dataset()
    yaml_path = Path(path).expanduser().resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset file {yaml_path!s} does not exist")
    _LOG.info("Loading dataset from %s", yaml_path)
    return load_dataset_from_yaml(yaml_path)


def _announce_ready(address: str, ready_file: Optional[str], emit_stdout: bool) -> None:
    payload = {"event": "ready", "address": address, "pid": os.getpid()}
    message = json.dumps(payload, separators=(",", ":"))
    if ready_file:
        ready_path = Path(ready_file).expanduser().resolve()
        ready_path.write_text(message, encoding="utf-8")
    if emit_stdout:
        print(message, flush=True)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Address to bind the stub service on")
    parser.add_argument("--port", default=0, type=int, help="Port to bind (0 = auto assign)")
    parser.add_argument(
        "--dataset",
        help="Path to YAML dataset definition used to seed the stub services",
    )
    parser.add_argument(
        "--stream-repeat",
        action="store_true",
        help="When set the OHLCV stream loops the dataset indefinitely",
    )
    parser.add_argument(
        "--stream-interval",
        default=0.0,
        type=float,
        help="Delay in seconds between streamed OHLCV updates",
    )
    parser.add_argument(
        "--ready-file",
        help="Optional file that receives a JSON payload with the bound address",
    )
    parser.add_argument(
        "--no-stdout-ready",
        action="store_true",
        help="Do not emit the ready payload on stdout (useful for parent process pipes)",
    )
    parser.add_argument(
        "--lifetime",
        type=float,
        default=0.0,
        help="Optional lifetime in seconds after which the service stops automatically",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


def _configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    try:
        dataset = _load_dataset(args.dataset)
    except Exception as exc:  # pragma: no cover - configuration error
        _LOG.error("Failed to load dataset: %s", exc)
        return 2

    server = TradingStubServer(
        dataset=dataset,
        host=args.host,
        port=args.port,
        stream_repeat=args.stream_repeat,
        stream_interval=args.stream_interval,
    )

    stop_event = _StopEvent()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop_event.set)

    server.start()
    _LOG.info("Stub bot_core trading service listening on %s", server.address)

    emit_stdout = not args.no_stdout_ready
    try:
        _announce_ready(server.address, args.ready_file, emit_stdout)
    except Exception as exc:  # pragma: no cover - IO error
        _LOG.warning("Unable to announce ready state: %s", exc)

    deadline = time.monotonic() + args.lifetime if args.lifetime > 0.0 else None
    try:
        while True:
            if stop_event.is_set:
                break
            if deadline is not None and time.monotonic() >= deadline:
                _LOG.info("Lifetime expired – stopping stub service")
                break
            time.sleep(0.25)
    finally:
        _LOG.info("Stopping stub service")
        server.stop()

    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution
    sys.exit(main())
