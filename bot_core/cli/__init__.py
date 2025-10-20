"""Prosty interfejs wiersza poleceń dla modułów giełdowych."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 nie jest wspierany
    tomllib = None  # type: ignore[assignment]

from bot_core.exchanges.core import Mode
from bot_core.exchanges.health import HealthCheck, HealthMonitor, HealthStatus
from bot_core.exchanges.manager import ExchangeManager


DEFAULT_CREDENTIALS_PATH = Path("secrets/desktop.toml")


class CLIUsageError(RuntimeError):
    """Błąd walidacji danych wejściowych CLI."""


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bot_core.cli",
        description="Narzędzia diagnostyczne dla adapterów giełdowych.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    health = subparsers.add_parser(
        "health-check",
        help="Uruchamia testy zdrowia dla wskazanej giełdy",
    )
    health.add_argument("--exchange", required=True, help="Identyfikator giełdy (np. binance, kraken, zonda)")
    health.add_argument(
        "--mode",
        choices=("spot", "margin", "futures", "paper"),
        help="Tryb działania (domyślnie zgodny z konfiguracją w pliku).",
    )
    health.add_argument(
        "--testnet",
        action="store_true",
        help="Wymusza użycie środowiska testnet dla trybów spot/margin/futures.",
    )
    health.add_argument(
        "--credentials-file",
        default=str(DEFAULT_CREDENTIALS_PATH),
        help="Ścieżka do pliku TOML z poświadczeniami (domyślnie secrets/desktop.toml).",
    )
    health.add_argument("--key", help="Klucz API – jeżeli nie podano, zostanie odczytany z pliku")
    health.add_argument("--secret", help="Sekret API – jeżeli nie podano, zostanie odczytany z pliku")
    health.add_argument(
        "--skip-public",
        action="store_true",
        help="Pomija publiczny test (np. ładowanie rynków).",
    )
    health.add_argument(
        "--skip-private",
        action="store_true",
        help="Pomija prywatny test (np. pobieranie salda).",
    )

    return parser


def _load_exchange_profile(path: str | Path | None, exchange: str) -> dict[str, object]:
    if not path:
        return {}
    storage = Path(path).expanduser()
    if not storage.exists():
        return {}
    if tomllib is None:  # pragma: no cover - zabezpieczenie przed starszym Pythonem
        raise CLIUsageError("Uruchomienie CLI wymaga Pythona 3.11 lub nowszego (moduł tomllib).")
    try:
        with storage.open("rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:  # pragma: no cover - błędy IO raportujemy użytkownikowi
        raise CLIUsageError(f"Nie udało się odczytać pliku {storage}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise CLIUsageError(f"Plik {storage} zawiera niepoprawny TOML: {exc}") from exc
    section = data.get(exchange)
    if not isinstance(section, dict):
        return {}
    return dict(section)


def _extract_adapter_settings(profile: Mapping[str, object]) -> dict[str, object]:
    skip_keys = {
        "key",
        "secret",
        "mode",
        "passphrase",
        "testnet",
        "paper",
        "watchdog",
        "health_check",
    }
    return {key: value for key, value in profile.items() if key not in skip_keys}


def _configure_mode(manager: ExchangeManager, mode: str | None, *, testnet: bool) -> None:
    normalized = (mode or "spot").lower()
    if normalized == "paper":
        manager.set_mode(paper=True)
    elif normalized == "margin":
        manager.set_mode(margin=True, testnet=testnet)
    elif normalized == "futures":
        manager.set_mode(futures=True, testnet=testnet)
    else:
        manager.set_mode(spot=True, testnet=testnet)


def _configure_watchdog(manager: ExchangeManager, profile: Mapping[str, object]) -> None:
    watchdog = profile.get("watchdog")
    if not isinstance(watchdog, Mapping):
        return
    retry_policy = watchdog.get("retry_policy")
    circuit_breaker = watchdog.get("circuit_breaker")
    kwargs: dict[str, Mapping[str, object]] = {}
    if isinstance(retry_policy, Mapping):
        kwargs["retry_policy"] = retry_policy
    if isinstance(circuit_breaker, Mapping):
        kwargs["circuit_breaker"] = circuit_breaker
    if kwargs:
        manager.configure_watchdog(**kwargs)


def _build_health_checks(
    manager: ExchangeManager,
    *,
    include_public: bool,
    include_private: bool,
    public_symbol: str | None,
) -> list[HealthCheck]:
    checks: list[HealthCheck] = []
    if include_public:
        if public_symbol:
            checks.append(HealthCheck(name="public_api", check=lambda: manager.fetch_ticker(public_symbol)))
        else:
            checks.append(HealthCheck(name="public_api", check=lambda: manager.load_markets()))
    if include_private:
        checks.append(HealthCheck(name="private_api", check=lambda: manager.fetch_balance()))
    if not checks:
        raise CLIUsageError("Brak aktywnych testów zdrowia – włącz przynajmniej jeden check.")
    return checks


def _format_details(details: Mapping[str, object]) -> str:
    if not details:
        return ""
    parts: list[str] = []
    for key in sorted(details):
        value = details[key]
        parts.append(f"{key}={value}")
    return " [" + ", ".join(parts) + "]"


def run_health_check(
    args: argparse.Namespace,
    *,
    manager_factory: type[ExchangeManager] = ExchangeManager,
) -> int:
    profile = _load_exchange_profile(args.credentials_file, args.exchange)
    profile_mode = str(profile.get("mode", "spot") or "spot") if "mode" in profile else None
    testnet_flag = bool(args.testnet or profile.get("testnet"))

    manager = manager_factory(exchange_id=args.exchange)
    _configure_mode(manager, args.mode or profile_mode, testnet=testnet_flag)

    api_key = args.key or profile.get("key") if profile else args.key
    secret = args.secret or profile.get("secret") if profile else args.secret
    manager.set_credentials(api_key, secret)

    if profile:
        _configure_watchdog(manager, profile)

    settings = _extract_adapter_settings(profile)
    if settings and manager.mode in {Mode.MARGIN, Mode.FUTURES}:
        manager.configure_native_adapter(settings=settings)

    health_config = profile.get("health_check") if isinstance(profile.get("health_check"), Mapping) else {}
    public_symbol = None
    if isinstance(health_config, Mapping):
        symbol_raw = health_config.get("public_symbol")
        if isinstance(symbol_raw, str) and symbol_raw.strip():
            public_symbol = symbol_raw.strip()
    skip_public = bool(args.skip_public)
    skip_private = bool(args.skip_private)
    if isinstance(health_config, Mapping):
        skip_public = skip_public or bool(health_config.get("skip_public"))
        skip_private = skip_private or bool(health_config.get("skip_private"))

    include_public = not skip_public
    include_private = not skip_private
    notes: list[str] = []

    if include_private:
        if manager.mode is Mode.PAPER:
            include_private = False
            notes.append("Pomijam test private_api w trybie paper.")
        elif not (api_key and secret):
            include_private = False
            notes.append("Pomijam test private_api – brak kompletnych poświadczeń API.")

    checks = _build_health_checks(
        manager,
        include_public=include_public,
        include_private=include_private,
        public_symbol=public_symbol,
    )
    monitor = manager.create_health_monitor(checks)
    results = monitor.run()

    for note in notes:
        print(note, file=sys.stderr)

    for result in results:
        latency = f"{result.latency:.3f}s"
        details = _format_details(result.details)
        print(f"{result.name}: {result.status.value} (latency={latency}){details}")

    overall = HealthMonitor.overall_status(results)
    print(f"Overall status: {overall.value}")

    exit_code_map = {
        HealthStatus.HEALTHY: 0,
        HealthStatus.DEGRADED: 1,
        HealthStatus.UNAVAILABLE: 2,
    }
    return exit_code_map.get(overall, 2)


def main(
    argv: Sequence[str] | None = None,
    *,
    manager_factory: type[ExchangeManager] = ExchangeManager,
) -> int:
    parser = create_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "health-check":
            return run_health_check(args, manager_factory=manager_factory)
        raise CLIUsageError(f"Nieznana komenda: {args.command}")
    except CLIUsageError as exc:
        print(f"Błąd: {exc}", file=sys.stderr)
        return 2


__all__ = ["main", "create_parser", "run_health_check", "CLIUsageError"]
