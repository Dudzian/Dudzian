"""Uruchamia lokalny gateway HTTP obsługujący streamy long-pollowe."""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from typing import Sequence

from bot_core.config.loader import load_core_config
from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.stream_gateway import StreamGateway, start_stream_gateway
from bot_core.runtime.bootstrap import get_registered_adapter_factories


def _configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Uruchamia lokalny gateway streamingu long-pollowego.")
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do pliku core.yaml")
    parser.add_argument(
        "--environment",
        dest="environments",
        action="append",
        required=True,
        help="Nazwa środowiska z pliku konfiguracyjnego (można podać wielokrotnie)",
    )
    parser.add_argument("--host", help="Adres hosta nasłuchu gateway'a")
    parser.add_argument("--port", type=int, help="Port nasłuchu gateway'a")
    parser.add_argument("--api-key", dest="api_key", help="Klucz API używany do inicjalizacji adapterów")
    parser.add_argument("--api-secret", dest="api_secret", help="Sekret API dla podpisanych endpointów")
    parser.add_argument("--log-level", default="INFO", help="Poziom logowania (np. INFO, DEBUG)")
    return parser.parse_args(list(argv))


def _build_adapter(
    *,
    env_config,
    api_key: str,
    api_secret: str | None,
):
    factories = get_registered_adapter_factories()
    try:
        factory = factories[env_config.exchange]
    except KeyError as exc:  # pragma: no cover - zależy od konfiguracji
        raise SystemExit(f"Brak fabryki adaptera '{env_config.exchange}'") from exc

    permissions = tuple(getattr(env_config, "required_permissions", ()) or ("read", "trade"))
    credentials = ExchangeCredentials(
        key_id=api_key,
        secret=api_secret,
        environment=Environment(env_config.environment),
        permissions=permissions,
    )
    settings = dict(getattr(env_config, "adapter_settings", {}) or {})
    adapter = factory(credentials, environment=env_config.environment, settings=settings)
    allowlist = tuple(getattr(env_config, "ip_allowlist", ()) or ())
    adapter.configure_network(ip_allowlist=allowlist or None)
    return adapter


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    _configure_logging(args.log_level)

    core_config = load_core_config(args.config)
    environments = getattr(core_config, "environments", {})
    selected: list[tuple[str, object]] = []
    for name in args.environments:
        env = environments.get(name)
        if env is None:
            raise SystemExit(f"Nie znaleziono środowiska '{name}' w konfiguracji")
        selected.append((name, env))

    if not selected:
        raise SystemExit("Nie wybrano żadnego środowiska do obsługi streamingu")

    api_key = args.api_key or "public"
    api_secret = args.api_secret

    gateway = StreamGateway()
    for name, env in selected:
        adapter = _build_adapter(env_config=env, api_key=api_key, api_secret=api_secret)
        gateway.register_adapter(env.exchange, environment=env.environment.value, adapter=adapter)

    # Domyślny host/port z konfiguracji streamu (pierwsze środowisko)
    stream_cfg = getattr(selected[0][1], "stream", None)
    host = args.host or getattr(stream_cfg, "host", None) or "127.0.0.1"
    port = args.port if args.port is not None else getattr(stream_cfg, "port", None) or 8765

    server, thread = start_stream_gateway(host, port, gateway=gateway)

    stop_event = threading.Event()

    def _handle_signal(signum, frame):  # noqa: D401, ARG001 - sygnatura wymagania signal
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    try:  # pragma: no cover - obsługa SIGINT nie jest testowana
        signal.signal(signal.SIGINT, _handle_signal)
    except ValueError:
        # W niektórych środowiskach (np. wątkach) nie można ustawiać handlerów
        pass

    try:
        while not stop_event.is_set():
            thread.join(timeout=1.0)
            if not thread.is_alive():
                break
    except KeyboardInterrupt:  # pragma: no cover - interakcja użytkownika
        stop_event.set()
    finally:
        server.shutdown()
        gateway.close()
        thread.join(timeout=5.0)

    return 0


if __name__ == "__main__":  # pragma: no cover - manualne uruchomienie
    raise SystemExit(main())
