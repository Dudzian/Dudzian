"""Prosty interfejs wiersza poleceń dla modułów giełdowych."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 nie jest wspierany
    tomllib = None  # type: ignore[assignment]

try:  # pragma: no cover - PyYAML jest wymagany tylko dla konfiguracji środowisk
    import yaml
except ModuleNotFoundError:  # pragma: no cover - opcjonalna zależność
    yaml = None  # type: ignore[assignment]

from bot_core.exchanges.core import Mode
from bot_core.exchanges.health import HealthCheck, HealthMonitor, HealthStatus
from bot_core.exchanges.manager import ExchangeManager


DEFAULT_CREDENTIALS_PATH = Path("secrets/desktop.toml")
DEFAULT_ENVIRONMENT_CONFIG_PATH = Path("config/environments/exchange_modes.yaml")

_ENV_PLACEHOLDER_RE = re.compile(r"^\$\{([A-Z0-9_]+)\}$")
_CREDENTIAL_ALIASES = {
    "key": ("key", "api_key", "apiKey"),
    "secret": ("secret", "api_secret", "apiSecret"),
    "passphrase": ("passphrase", "password", "passPhrase"),
}


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
    health.add_argument(
        "--exchange",
        help="Identyfikator giełdy (np. binance, kraken, zonda).",
    )
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
    health.add_argument(
        "--key-env",
        help="Nazwa zmiennej środowiskowej z kluczem API (gdy brak wartości bezpośredniej)",
    )
    health.add_argument("--secret", help="Sekret API – jeżeli nie podano, zostanie odczytany z pliku")
    health.add_argument(
        "--secret-env",
        help="Nazwa zmiennej środowiskowej z sekretem API (gdy brak wartości bezpośredniej)",
    )
    health.add_argument(
        "--passphrase",
        help="Passphrase API – jeżeli nie podano, zostanie odczytany z pliku",
    )
    health.add_argument(
        "--passphrase-env",
        help="Zmienna środowiskowa z passphrase API (gdy brak wartości bezpośredniej)",
    )
    health.add_argument(
        "--environment-config",
        help="Ścieżka do pliku YAML opisującego środowiska (domyślnie config/environments/exchange_modes.yaml).",
    )
    health.add_argument(
        "--environment",
        help="Nazwa środowiska z pliku YAML, które ma zostać załadowane.",
    )
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


def _merge_mappings(
    base: Mapping[str, object] | None,
    override: Mapping[str, object] | None,
) -> dict[str, object]:
    result: dict[str, object] = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge_mappings(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = value
    return result


def _coerce_credential_value(value: object, *, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    match = _ENV_PLACEHOLDER_RE.fullmatch(stripped)
    if match:
        env_name = match.group(1)
        env_value = os.environ.get(env_name)
        if env_value is None or not env_value.strip():
            raise CLIUsageError(
                f"Zmienna środowiskowa {env_name} nie zawiera wartości wymaganej dla '{key}'."
            )
        return env_value.strip()
    return stripped


def _load_environment_profile(path: str | Path, environment: str) -> dict[str, object]:
    if yaml is None:  # pragma: no cover - opcjonalna zależność
        raise CLIUsageError("Wczytanie konfiguracji środowisk wymaga pakietu PyYAML (pip install pyyaml).")

    storage = Path(path).expanduser()
    if not storage.exists():
        raise CLIUsageError(f"Plik konfiguracji środowisk {storage} nie istnieje.")

    try:
        with storage.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:  # type: ignore[attr-defined]
        raise CLIUsageError(f"Plik {storage} zawiera niepoprawny YAML: {exc}") from exc
    except OSError as exc:
        raise CLIUsageError(f"Nie udało się odczytać pliku {storage}: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise CLIUsageError(f"Plik {storage} musi zawierać słownik środowisk.")

    defaults_raw = payload.get("defaults")
    defaults: Mapping[str, object] | None = defaults_raw if isinstance(defaults_raw, Mapping) else None

    section = payload.get(environment)
    if not isinstance(section, Mapping):
        raise CLIUsageError(f"Środowisko '{environment}' nie istnieje w pliku {storage}.")

    merged = _merge_mappings(defaults, section)
    merged.setdefault("name", environment)
    merged.setdefault("__path__", str(storage))
    return merged


def _resolve_credential(
    *,
    inline: str | None,
    env_name: str | None,
    profile: Mapping[str, object] | None,
    environment: Mapping[str, object] | None,
    key: str,
) -> str | None:
    if inline is not None:
        return inline
    if env_name:
        value = os.environ.get(env_name)
        if value is None or not value.strip():
            raise CLIUsageError(
                f"Zmienna środowiskowa {env_name} nie zawiera wartości wymaganej dla '{key}'."
            )
        return value
    lookup_keys = _CREDENTIAL_ALIASES.get(key, (key,))
    for source in (environment, profile):
        if not isinstance(source, Mapping):
            continue
        for alias in lookup_keys:
            value = _coerce_credential_value(source.get(alias), key=key)
            if value is not None:
                return value
    return None


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
    if getattr(args, "environment_config", None) and not getattr(args, "environment", None):
        raise CLIUsageError("Argument --environment-config wymaga podania nazwy środowiska (--environment).")

    environment_profile: dict[str, object] = {}
    if getattr(args, "environment", None):
        config_path = getattr(args, "environment_config", None) or str(DEFAULT_ENVIRONMENT_CONFIG_PATH)
        environment_profile = _load_environment_profile(config_path, args.environment)

    exchange_id = args.exchange or environment_profile.get("exchange")
    if not exchange_id:
        raise CLIUsageError("Nie określono giełdy – podaj --exchange lub konfigurację środowiska.")
    exchange_id = str(exchange_id)

    profile = _load_exchange_profile(args.credentials_file, exchange_id)
    profile_mode = str(profile.get("mode", "spot") or "spot") if "mode" in profile else None

    env_manager_cfg = environment_profile.get("exchange_manager")
    env_manager_cfg = env_manager_cfg if isinstance(env_manager_cfg, Mapping) else {}

    env_mode_raw = env_manager_cfg.get("mode") if "mode" in env_manager_cfg else None
    env_mode = str(env_mode_raw) if isinstance(env_mode_raw, str) else None

    mode_choice = args.mode or env_mode or profile_mode

    profile_testnet = profile.get("testnet") if "testnet" in profile else None
    env_testnet = env_manager_cfg.get("testnet") if "testnet" in env_manager_cfg else None
    if args.testnet:
        testnet_flag = True
    elif env_testnet is not None:
        testnet_flag = bool(env_testnet)
    elif profile_testnet is not None:
        testnet_flag = bool(profile_testnet)
    else:
        testnet_flag = False

    manager = manager_factory(exchange_id=exchange_id)
    _configure_mode(manager, mode_choice or profile_mode, testnet=testnet_flag)

    profile_mapping = profile if profile else None
    environment_credentials = environment_profile.get("credentials") if isinstance(environment_profile.get("credentials"), Mapping) else None
    api_key = _resolve_credential(
        inline=getattr(args, "key", None),
        env_name=getattr(args, "key_env", None),
        profile=profile_mapping,
        environment=environment_credentials,
        key="key",
    )
    secret = _resolve_credential(
        inline=getattr(args, "secret", None),
        env_name=getattr(args, "secret_env", None),
        profile=profile_mapping,
        environment=environment_credentials,
        key="secret",
    )
    passphrase = _resolve_credential(
        inline=getattr(args, "passphrase", None),
        env_name=getattr(args, "passphrase_env", None),
        profile=profile_mapping,
        environment=environment_credentials,
        key="passphrase",
    )
    manager.set_credentials(api_key, secret, passphrase=passphrase)

    if environment_profile:
        paper_variant = env_manager_cfg.get("paper_variant")
        if isinstance(paper_variant, str) and paper_variant.strip():
            manager.set_paper_variant(paper_variant)

        if "paper_initial_cash" in env_manager_cfg:
            initial_cash = env_manager_cfg.get("paper_initial_cash")
            cash_asset = env_manager_cfg.get("paper_cash_asset")
            manager.set_paper_balance(
                float(initial_cash),
                asset=str(cash_asset) if isinstance(cash_asset, str) and cash_asset.strip() else None,
            )
        elif "paper_cash_asset" in env_manager_cfg:
            cash_asset_only = env_manager_cfg.get("paper_cash_asset")
            if isinstance(cash_asset_only, str) and cash_asset_only.strip():
                manager.set_paper_balance(manager.get_paper_initial_cash(), asset=cash_asset_only)

        if "paper_fee_rate" in env_manager_cfg:
            manager.set_paper_fee_rate(float(env_manager_cfg.get("paper_fee_rate")))

        simulator_settings = env_manager_cfg.get("simulator")
        if isinstance(simulator_settings, Mapping):
            manager.configure_paper_simulator(**simulator_settings)

    combined_watchdog = _merge_mappings(
        profile.get("watchdog") if isinstance(profile.get("watchdog"), Mapping) else None,
        env_manager_cfg.get("watchdog") if isinstance(env_manager_cfg.get("watchdog"), Mapping) else None,
    )
    if combined_watchdog:
        _configure_watchdog(manager, {"watchdog": combined_watchdog})

    settings = _extract_adapter_settings(profile)
    if settings and manager.mode in {Mode.MARGIN, Mode.FUTURES}:
        manager.configure_native_adapter(settings=settings)

    env_native = env_manager_cfg.get("native_adapter")
    if isinstance(env_native, Mapping):
        native_settings = env_native.get("settings")
        native_mode_raw = env_native.get("mode")
        target_mode = None
        if isinstance(native_mode_raw, str):
            normalized = native_mode_raw.strip().lower()
            if normalized == "margin":
                target_mode = Mode.MARGIN
            elif normalized == "futures":
                target_mode = Mode.FUTURES
            else:
                raise CLIUsageError(
                    "Konfiguracja natywnego adaptera wspiera tylko tryby 'margin' lub 'futures'."
                )
        if isinstance(native_settings, Mapping):
            manager.configure_native_adapter(settings=native_settings, mode=target_mode)

    health_profile = _merge_mappings(
        profile.get("health_check") if isinstance(profile.get("health_check"), Mapping) else None,
        environment_profile.get("health_check")
        if isinstance(environment_profile.get("health_check"), Mapping)
        else None,
    )
    health_config: Mapping[str, object] = health_profile
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
