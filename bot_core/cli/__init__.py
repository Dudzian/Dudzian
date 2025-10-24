"""Prosty interfejs wiersza poleceń dla modułów giełdowych."""

from __future__ import annotations

import argparse
import importlib
import json
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

from bot_core.config.loader import load_core_config
from bot_core.exchanges.core import Mode
from bot_core.exchanges.health import HealthCheck, HealthMonitor, HealthStatus
from bot_core.exchanges.manager import ExchangeManager
from bot_core.runtime.pipeline import (
    describe_multi_strategy_configuration,
    describe_strategy_definitions,
)
from bot_core.strategies.catalog import DEFAULT_STRATEGY_CATALOG


DEFAULT_CREDENTIALS_PATH = Path("secrets/desktop.toml")
DEFAULT_ENVIRONMENT_CONFIG_PATH = Path("config/environments/exchange_modes.yaml")

_ENV_PLACEHOLDER_RE = re.compile(r"^\$\{([A-Z0-9_]+)\}$")
_CREDENTIAL_ALIASES = {
    "key": ("key", "api_key", "apiKey"),
    "secret": ("secret", "api_secret", "apiSecret"),
    "passphrase": ("passphrase", "password", "passPhrase"),
}

_SUPPORTED_HEALTH_CHECKS = ("public_api", "private_api")


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
        help=(
            "Ścieżka do pliku YAML opisującego środowiska (domyślnie "
            "config/environments/exchange_modes.yaml). Gdy plik posiada sekcję "
            "defaults.environment, komenda użyje jej jako domyślnego środowiska."
        ),
    )
    health.add_argument(
        "--environment",
        help=(
            "Nazwa środowiska z pliku YAML, które ma zostać załadowane. "
            "Opcjonalna, jeśli plik definiuje defaults.environment."
        ),
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
    health.add_argument(
        "--check",
        dest="checks",
        action="append",
        help=(
            "Wykonuje tylko wskazane testy zdrowia (np. --check public_api). "
            "Argument można podawać wielokrotnie lub przekazać listę rozdzieloną przecinkami. "
            "Nazwy testów są nieczułe na wielkość liter."
        ),
    )
    health.add_argument(
        "--list-checks",
        action="store_true",
        help="Wyświetla listę dostępnych testów i kończy działanie komendy.",
    )
    health.add_argument(
        "--private-asset",
        help="Symbol waluty, której saldo ma być weryfikowane w teście private_api (np. USDT).",
    )
    health.add_argument(
        "--private-min-balance",
        type=float,
        help="Minimalne wymagane saldo dla wskazanej waluty w teście private_api.",
    )
    health.add_argument(
        "--public-symbol",
        help=(
            "Symbol giełdowy używany w teście public_api (np. BTC/USDT). "
            "Domyślnie wybierany jest ticker z konfiguracji środowiska lub ładowane są całe rynki."
        ),
    )
    health.add_argument(
        "--paper-variant",
        help="Wymusza wariant symulatora paper (np. spot, margin, futures).",
    )
    health.add_argument(
        "--paper-initial-cash",
        type=float,
        help="Ustawia początkowy kapitał w symulatorze paper (wartość dodatnia).",
    )
    health.add_argument(
        "--paper-cash-asset",
        help="Ustawia walutę gotówkową w symulatorze paper (np. USDT).",
    )
    health.add_argument(
        "--paper-fee-rate",
        type=float,
        help="Ustawia stawkę prowizji symulatora paper (wartość nieujemna).",
    )
    health.add_argument(
        "--paper-leverage-limit",
        type=float,
        help=(
            "Ustawia limit dźwigni w symulatorze margin/futures (wartość dodatnia)."
        ),
    )
    health.add_argument(
        "--paper-maintenance-margin",
        type=float,
        help=(
            "Ustawia współczynnik maintenance margin w symulatorze (wartość dodatnia)."
        ),
    )
    health.add_argument(
        "--paper-funding-rate",
        type=float,
        help=(
            "Ustawia dzienny funding rate w symulatorze (może być ujemny/zerowy)."
        ),
    )
    health.add_argument(
        "--paper-funding-interval",
        type=float,
        help=(
            "Ustawia odstęp między naliczeniami funding w sekundach (wartość dodatnia)."
        ),
    )
    health.add_argument(
        "--paper-simulator-setting",
        dest="paper_simulator_settings",
        action="append",
        metavar="KEY=VALUE",
        help=(
            "Nadpisuje dowolny parametr symulatora paper (np. --paper-simulator-setting maintenance_margin_ratio=0.12)."
        ),
    )
    health.add_argument(
        "--watchdog-max-attempts",
        type=int,
        help="Ustawia maksymalną liczbę prób w retry policy watchdog-a (wartość dodatnia).",
    )
    health.add_argument(
        "--watchdog-base-delay",
        type=float,
        help="Ustawia początkowe opóźnienie retry w sekundach (wartość dodatnia).",
    )
    health.add_argument(
        "--watchdog-max-delay",
        type=float,
        help="Ustawia maksymalne opóźnienie retry w sekundach (wartość dodatnia).",
    )
    health.add_argument(
        "--watchdog-jitter-min",
        type=float,
        help="Nadpisuje minimalny jitter retry (wartość nieujemna).",
    )
    health.add_argument(
        "--watchdog-jitter-max",
        type=float,
        help="Nadpisuje maksymalny jitter retry (wartość nieujemna).",
    )
    health.add_argument(
        "--watchdog-failure-threshold",
        type=int,
        help="Ustawia próg otwarcia circuit breakera (liczba dodatnia).",
    )
    health.add_argument(
        "--watchdog-recovery-timeout",
        type=float,
        help="Ustawia czas regeneracji circuit breakera w sekundach (wartość dodatnia).",
    )
    health.add_argument(
        "--watchdog-half-open-success",
        type=int,
        help="Ustawia liczbę udanych prób wymaganych do zamknięcia stanu half-open (wartość dodatnia).",
    )
    health.add_argument(
        "--watchdog-retry-exception",
        dest="watchdog_retry_exceptions",
        action="append",
        metavar="MODULE.Class",
        help=(
            "Dodaje klasę wyjątku do listy retry watchdog-a (np. builtins.TimeoutError)."
            " Argument można powtarzać wielokrotnie."
        ),
    )
    health.add_argument(
        "--native-setting",
        dest="native_settings",
        action="append",
        metavar="KEY=VALUE",
        help="Nadpisuje ustawienie natywnego adaptera (np. --native-setting margin_mode=cross).",
    )
    health.add_argument(
        "--native-mode",
        choices=("margin", "futures"),
        help="Wymusza tryb natywnego adaptera (domyślnie zgodny z bieżącym trybem managera).",
    )
    health.add_argument(
        "--output-format",
        choices=("text", "json", "json-pretty"),
        default="text",
        help=(
            "Format wyjścia komendy (domyślnie text). Użyj json-pretty, aby otrzymać czytelny JSON."
        ),
    )
    health.add_argument(
        "--output-path",
        help=(
            "Ścieżka do pliku, do którego zostanie zapisany wynik testów. Dostępne tylko "
            "w połączeniu z --output-format=json lub json-pretty."
        ),
    )

    list_envs = subparsers.add_parser(
        "list-environments",
        help="Wyświetla zdefiniowane środowiska z pliku YAML.",
    )
    list_envs.add_argument(
        "--environment-config",
        help="Ścieżka do pliku YAML opisującego środowiska (domyślnie config/environments/exchange_modes.yaml).",
    )

    show_env = subparsers.add_parser(
        "show-environment",
        help="Prezentuje pełną konfigurację środowiska po scaleniu z wartościami domyślnymi.",
    )
    show_env.add_argument(
        "--environment-config",
        help="Ścieżka do pliku YAML opisującego środowiska (domyślnie config/environments/exchange_modes.yaml).",
    )
    show_env.add_argument(
        "--environment",
        required=True,
        help="Nazwa środowiska, którego konfigurację należy wyświetlić.",
    )

    catalog = subparsers.add_parser(
        "strategy-catalog",
        help="Wypisuje katalog silników strategii oraz opcjonalnie definicje z konfiguracji.",
    )
    catalog.add_argument(
        "--config",
        help="Ścieżka do pliku konfiguracji core, z którego zostaną odczytane definicje strategii.",
    )
    catalog.add_argument(
        "--scheduler",
        help="Opcjonalnie zawęża listę definicji do wskazanego scheduler-a.",
    )
    catalog.add_argument(
        "--engine",
        dest="engines",
        action="append",
        help="Filtruje wyniki po nazwie silnika (można podać wielokrotnie).",
    )
    catalog.add_argument(
        "--capability",
        dest="capabilities",
        action="append",
        help="Filtruje wyniki po wymaganej licencji/capability silnika.",
    )
    catalog.add_argument(
        "--tag",
        dest="tags",
        action="append",
        help="Filtruje wyniki po tagach (domyślnych lub wynikających z definicji).",
    )
    catalog.add_argument(
        "--output-format",
        choices=("text", "json", "json-pretty"),
        default="text",
        help="Format wyjścia (text/json/json-pretty).",
    )
    catalog.add_argument(
        "--include-parameters",
        action="store_true",
        help="Dołącza parametry strategii przy wczytanej konfiguracji core.",
    )

    plan = subparsers.add_parser(
        "scheduler-plan",
        help="Buduje raport konfiguracyjny scheduler-a multi-strategy.",
    )
    plan.add_argument(
        "--config",
        required=True,
        help="Ścieżka do pliku konfiguracji core.",
    )
    plan.add_argument(
        "--scheduler",
        help="Nazwa scheduler-a multi-strategy do opisania (domyślnie pierwszy z konfiguracji).",
    )
    plan.add_argument(
        "--output-format",
        choices=("text", "json", "json-pretty"),
        default="text",
        help="Format wyjścia (text/json/json-pretty).",
    )
    plan.add_argument(
        "--filter-tag",
        dest="filter_tags",
        action="append",
        help="Zawęża listę harmonogramów do strategii posiadających wskazany tag.",
    )
    plan.add_argument(
        "--filter-strategy",
        dest="filter_strategies",
        action="append",
        help="Zawęża listę harmonogramów do wskazanych nazw strategii.",
    )
    plan.add_argument(
        "--no-definitions",
        dest="include_definitions",
        action="store_false",
        help="Pomija sekcję definicji strategii w raporcie JSON.",
    )
    plan.set_defaults(include_definitions=True)

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
        "native_adapter",
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


def _coerce_optional_float(value: object, *, key: str) -> float | None:
    """Konwertuje wartość na liczbę zmiennoprzecinkową lub zwraca ``None``."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError as exc:
            raise CLIUsageError(f"Wartość {key} musi być liczbą.") from exc
    raise CLIUsageError(f"Wartość {key} musi być liczbą.")


def _normalize_native_mode(value: str, *, context: str) -> Mode:
    normalized = value.strip().lower()
    if normalized == "margin":
        return Mode.MARGIN
    if normalized == "futures":
        return Mode.FUTURES
    raise CLIUsageError(f"{context} wspiera tylko tryby 'margin' lub 'futures'.")


def _parse_native_setting_argument(argument: str) -> tuple[str, object]:
    if not isinstance(argument, str):
        raise CLIUsageError("Opcja --native-setting wymaga formatu klucz=wartość.")
    if "=" not in argument:
        raise CLIUsageError("Opcja --native-setting wymaga formatu klucz=wartość.")
    key_raw, value_raw = argument.split("=", 1)
    key = key_raw.strip()
    if not key:
        raise CLIUsageError("Opcja --native-setting wymaga niepustego klucza.")
    value = value_raw.strip()
    if (value.startswith("\"") and value.endswith("\"")) or (
        value.startswith("'") and value.endswith("'")
    ):
        value = value[1:-1]
    if not value:
        return key, ""
    lowered = value.casefold()
    if lowered == "true":
        return key, True
    if lowered == "false":
        return key, False
    if lowered in {"null", "none"}:
        return key, None
    try:
        return key, int(value)
    except ValueError:
        try:
            return key, float(value)
        except ValueError:
            return key, value


def _parse_paper_simulator_setting_argument(argument: str) -> tuple[str, float]:
    """Parses ``KEY=VALUE`` pairs for paper simulator overrides."""

    key, value = _parse_native_setting_argument(argument)
    if isinstance(value, bool):
        raise CLIUsageError(
            "Wartość opcji --paper-simulator-setting musi być liczbą zmiennoprzecinkową."
        )
    if isinstance(value, (int, float)):
        return key, float(value)
    raise CLIUsageError(
        "Wartość opcji --paper-simulator-setting musi być liczbą zmiennoprzecinkową."
    )


def _extract_jitter_pair(value: object) -> tuple[float, float] | None:
    """Konwertuje sekwencję na parę ``(min, max)`` dla jittera."""

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _read_environment_payload(
    path: str | Path,
) -> tuple[Path, Mapping[str, object] | None, dict[str, Mapping[str, object]]]:
    if yaml is None:  # pragma: no cover - opcjonalna zależność
        raise CLIUsageError(
            "Wczytanie konfiguracji środowisk wymaga pakietu PyYAML (pip install pyyaml)."
        )

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
    defaults: Mapping[str, object] | None = (
        defaults_raw if isinstance(defaults_raw, Mapping) else None
    )

    environments: dict[str, Mapping[str, object]] = {}
    for key, value in payload.items():
        if key == "defaults":
            continue
        if not isinstance(key, str):
            raise CLIUsageError(
                f"Klucz środowiska musi być łańcuchem znaków – wykryto {key!r} w pliku {storage}."
            )
        if not isinstance(value, Mapping):
            raise CLIUsageError(
                f"Środowisko '{key}' w pliku {storage} musi być opisane słownikiem konfiguracji."
            )
        environments[key] = value

    return storage, defaults, environments


def _load_environment_profile(path: str | Path, environment: str) -> dict[str, object]:
    storage, defaults, environments = _read_environment_payload(path)

    section = environments.get(environment)
    if not isinstance(section, Mapping):
        raise CLIUsageError(f"Środowisko '{environment}' nie istnieje w pliku {storage}.")

    merged = _merge_mappings(defaults, section)
    merged.setdefault("name", environment)
    merged.setdefault("__path__", str(storage))
    return merged


def _summarize_environment(profile: Mapping[str, object]) -> str:
    name_raw = profile.get("name")
    name = (
        str(name_raw).strip()
        if isinstance(name_raw, str) and str(name_raw).strip()
        else "(nieznane)"
    )

    path_raw = profile.get("__path__")
    path = str(path_raw) if isinstance(path_raw, str) and path_raw else str(DEFAULT_ENVIRONMENT_CONFIG_PATH)

    summary_parts: list[str] = []

    exchange_raw = profile.get("exchange")
    if isinstance(exchange_raw, str) and exchange_raw.strip():
        summary_parts.append(f"exchange={exchange_raw.strip()}")

    portfolio_raw = profile.get("portfolio")
    if isinstance(portfolio_raw, str) and portfolio_raw.strip():
        summary_parts.append(f"portfolio={portfolio_raw.strip()}")

    manager_cfg = profile.get("exchange_manager") if isinstance(profile.get("exchange_manager"), Mapping) else None
    if isinstance(manager_cfg, Mapping):
        mode_raw = manager_cfg.get("mode")
        if isinstance(mode_raw, str) and mode_raw.strip():
            summary_parts.append(f"mode={mode_raw.strip()}")

        if "testnet" in manager_cfg:
            summary_parts.append(f"testnet={'true' if bool(manager_cfg.get('testnet')) else 'false'}")

        paper_variant = manager_cfg.get("paper_variant")
        if isinstance(paper_variant, str) and paper_variant.strip():
            summary_parts.append(f"paper_variant={paper_variant.strip()}")

    suffix = f" [{', '.join(summary_parts)}]" if summary_parts else ""
    return f"Aktywne środowisko: {name} ({path}){suffix}"


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


def _resolve_exception_class(path: str, *, context: str) -> type[Exception]:
    candidate = path.strip()
    if not candidate:
        raise CLIUsageError(f"{context} wymaga niepustej nazwy klasy wyjątku.")
    module_path, separator, class_name = candidate.replace(":", ".").rpartition(".")
    if not separator:
        raise CLIUsageError(
            f"{context} wymaga pełnej ścieżki modułowej (np. module.ExceptionClass)."
        )
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:  # pragma: no cover - zależne od środowiska uruchomienia
        raise CLIUsageError(f"Nie udało się zaimportować modułu {module_path!r}: {exc}") from exc
    try:
        attr = getattr(module, class_name)
    except AttributeError as exc:
        raise CLIUsageError(
            f"Moduł {module_path!r} nie zawiera klasy {class_name!r} wymaganej przez {context}."
        ) from exc
    if not isinstance(attr, type) or not issubclass(attr, Exception):
        raise CLIUsageError(
            f"{candidate!r} nie jest klasą wyjątku – wymagane są typy dziedziczące po Exception."
        )
    return attr


def _parse_retry_exceptions_config(value: object, *, context: str) -> tuple[type[Exception], ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        candidates: Sequence[object] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        candidates = value
    else:
        raise CLIUsageError(f"{context} musi być listą nazw klas wyjątków.")

    resolved: list[type[Exception]] = []
    for entry in candidates:
        if isinstance(entry, type) and issubclass(entry, Exception):
            resolved.append(entry)
            continue
        if not isinstance(entry, str):
            raise CLIUsageError(f"{context} może zawierać wyłącznie nazwy klas wyjątków.")
        resolved.append(_resolve_exception_class(entry, context=context))
    return tuple(resolved)


def _stringify_retry_exception(entry: object) -> str:
    if isinstance(entry, type) and issubclass(entry, Exception):
        return f"{entry.__module__}.{entry.__qualname__}"
    return str(entry)


def _serialize_watchdog_config(config: Mapping[str, object] | None) -> dict[str, object] | None:
    if not isinstance(config, Mapping) or not config:
        return None
    payload: dict[str, object] = {}
    retry_policy = config.get("retry_policy")
    if isinstance(retry_policy, Mapping):
        payload["retry_policy"] = dict(retry_policy)
    circuit_breaker = config.get("circuit_breaker")
    if isinstance(circuit_breaker, Mapping):
        payload["circuit_breaker"] = dict(circuit_breaker)
    if "retry_exceptions" in config:
        raw_exceptions = config.get("retry_exceptions")
        if isinstance(raw_exceptions, Sequence) and not isinstance(raw_exceptions, (bytes, bytearray, str)):
            items = list(raw_exceptions)
        elif raw_exceptions is None:
            items = []
        else:
            items = [raw_exceptions]
        payload["retry_exceptions"] = [
            _stringify_retry_exception(item) for item in items if item is not None
        ]
    return payload or None


def _configure_watchdog(manager: ExchangeManager, profile: Mapping[str, object]) -> None:
    watchdog = profile.get("watchdog")
    if not isinstance(watchdog, Mapping):
        return
    retry_policy = watchdog.get("retry_policy")
    circuit_breaker = watchdog.get("circuit_breaker")
    retry_exceptions = watchdog.get("retry_exceptions")
    kwargs: dict[str, object] = {}
    if isinstance(retry_policy, Mapping):
        kwargs["retry_policy"] = retry_policy
    if isinstance(circuit_breaker, Mapping):
        kwargs["circuit_breaker"] = circuit_breaker
    if retry_exceptions is not None:
        kwargs["retry_exceptions"] = _parse_retry_exceptions_config(
            retry_exceptions, context="Konfiguracja watchdog.retry_exceptions"
        )
    if kwargs:
        manager.configure_watchdog(**kwargs)


def _normalize_requested_checks(raw_checks: Sequence[str] | None) -> list[str]:
    """Zamienia argumenty CLI na listę unikalnych nazw testów."""

    if not raw_checks:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for entry in raw_checks:
        if not entry:
            continue
        pieces = (segment.strip() for segment in entry.split(","))
        for piece in pieces:
            if not piece:
                continue
            canonical = piece.casefold()
            if canonical not in seen:
                normalized.append(canonical)
                seen.add(canonical)
    return normalized


def _build_health_checks(
    manager: ExchangeManager,
    *,
    include_public: bool,
    include_private: bool,
    public_symbol: str | None,
    private_asset: str | None,
    private_min_balance: float | None,
) -> list[HealthCheck]:
    checks: list[HealthCheck] = []
    if include_public:
        if public_symbol:
            checks.append(HealthCheck(name="public_api", check=lambda: manager.fetch_ticker(public_symbol)))
        else:
            checks.append(HealthCheck(name="public_api", check=lambda: manager.load_markets()))
    if include_private:
        asset_label = private_asset

        def _validate_private() -> object:
            balance = manager.fetch_balance()
            if not asset_label:
                return balance
            if not isinstance(balance, Mapping):
                raise RuntimeError(
                    "Odpowiedź salda powinna być mapą z sekcjami total/free – otrzymano inny typ."
                )

            normalized = asset_label.upper()

            def _extract(section: Mapping[str, object] | None) -> float | None:
                if not isinstance(section, Mapping):
                    return None
                for key, raw_value in section.items():
                    if isinstance(key, str) and key.upper() == normalized:
                        try:
                            return float(raw_value) if raw_value is not None else None
                        except (TypeError, ValueError) as exc:  # pragma: no cover - defensywne logowanie
                            raise RuntimeError(
                                f"Nie udało się sparsować salda {asset_label} do liczby: {raw_value!r}"
                            ) from exc
                return None

            candidates: list[Mapping[str, object] | None] = [
                balance.get("total"),
                balance.get("free"),
                balance.get("used"),
            ]
            amount = None
            for section in candidates:
                amount = _extract(section)
                if amount is not None:
                    break
            if amount is None:
                amount = _extract(balance)

            if amount is None:
                raise RuntimeError(f"Saldo nie zawiera waluty {asset_label} w sekcjach total/free.")

            if private_min_balance is not None and amount < private_min_balance:
                raise RuntimeError(
                    f"Saldo waluty {asset_label} ({amount:.8f}) jest niższe niż wymagane minimum {private_min_balance:.8f}."
                )

            return balance

        checks.append(HealthCheck(name="private_api", check=_validate_private))
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
    output_format = getattr(args, "output_format", "text") or "text"
    if getattr(args, "list_checks", False):
        print("Dostępne testy health-check:")
        for name in _SUPPORTED_HEALTH_CHECKS:
            print(f"  * {name}")
        return 0
    environment_profile: dict[str, object] = {}
    environment_name: str | None = getattr(args, "environment", None)
    env_config_arg = getattr(args, "environment_config", None)
    environment_summary: str | None = None
    native_adapter_payload: dict[str, object] | None = None
    watchdog_payload: dict[str, object] | None = None
    if environment_name or env_config_arg:
        config_path = env_config_arg or str(DEFAULT_ENVIRONMENT_CONFIG_PATH)
        selected_environment = environment_name
        if not selected_environment:
            _, defaults, _ = _read_environment_payload(config_path)
            default_env = None
            if isinstance(defaults, Mapping):
                raw_default = defaults.get("environment")
                if isinstance(raw_default, str) and raw_default.strip():
                    default_env = raw_default.strip()
            if not default_env:
                raise CLIUsageError(
                    "Argument --environment-config wymaga podania nazwy środowiska (--environment) "
                    "lub zdefiniowania defaults.environment w pliku."
                )
            selected_environment = default_env
        environment_profile = _load_environment_profile(config_path, selected_environment)
        environment_summary = _summarize_environment(environment_profile)
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

    if environment_summary and output_format == "text":
        print(environment_summary)

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

        simulator_settings = env_manager_cfg.get("simulator")
        if isinstance(simulator_settings, Mapping):
            manager.configure_paper_simulator(**simulator_settings)

    cli_paper_variant = getattr(args, "paper_variant", None)
    if isinstance(cli_paper_variant, str) and cli_paper_variant.strip():
        manager.set_paper_variant(cli_paper_variant)

    cli_paper_cash_asset = getattr(args, "paper_cash_asset", None)
    if cli_paper_cash_asset is not None:
        normalized_asset = str(cli_paper_cash_asset).strip()
        if not normalized_asset:
            raise CLIUsageError("Opcja --paper-cash-asset wymaga niepustej wartości.")
        cli_paper_cash_asset = normalized_asset

    cli_initial_cash = getattr(args, "paper_initial_cash", None)
    if cli_initial_cash is not None:
        if cli_initial_cash <= 0:
            raise CLIUsageError("Opcja --paper-initial-cash wymaga dodatniej wartości.")
        manager.set_paper_balance(float(cli_initial_cash), asset=cli_paper_cash_asset)
    elif cli_paper_cash_asset is not None:
        manager.set_paper_balance(manager.get_paper_initial_cash(), asset=cli_paper_cash_asset)

    cli_fee_rate = getattr(args, "paper_fee_rate", None)
    if cli_fee_rate is not None:
        if cli_fee_rate < 0:
            raise CLIUsageError("Opcja --paper-fee-rate wymaga nieujemnej wartości.")
        manager.set_paper_fee_rate(float(cli_fee_rate))

    simulator_overrides: dict[str, float] = {}
    cli_leverage_limit = getattr(args, "paper_leverage_limit", None)
    if cli_leverage_limit is not None:
        if cli_leverage_limit <= 0:
            raise CLIUsageError("Opcja --paper-leverage-limit wymaga dodatniej wartości.")
        simulator_overrides["leverage_limit"] = float(cli_leverage_limit)

    cli_maintenance_margin = getattr(args, "paper_maintenance_margin", None)
    if cli_maintenance_margin is not None:
        if cli_maintenance_margin <= 0:
            raise CLIUsageError(
                "Opcja --paper-maintenance-margin wymaga dodatniej wartości."
            )
        simulator_overrides["maintenance_margin_ratio"] = float(cli_maintenance_margin)

    cli_funding_rate = getattr(args, "paper_funding_rate", None)
    if cli_funding_rate is not None:
        simulator_overrides["funding_rate"] = float(cli_funding_rate)

    cli_funding_interval = getattr(args, "paper_funding_interval", None)
    if cli_funding_interval is not None:
        if cli_funding_interval <= 0:
            raise CLIUsageError(
                "Opcja --paper-funding-interval wymaga dodatniej wartości (sekundy)."
            )
        simulator_overrides["funding_interval_seconds"] = float(cli_funding_interval)

    cli_simulator_setting_args = getattr(args, "paper_simulator_settings", None) or []
    for raw_argument in cli_simulator_setting_args:
        key, value = _parse_paper_simulator_setting_argument(raw_argument)
        simulator_overrides[key] = value

    if simulator_overrides:
        try:
            manager.configure_paper_simulator(**simulator_overrides)
        except ValueError as exc:
            raise CLIUsageError(str(exc)) from exc

    combined_watchdog = _merge_mappings(
        profile.get("watchdog") if isinstance(profile.get("watchdog"), Mapping) else None,
        env_manager_cfg.get("watchdog") if isinstance(env_manager_cfg.get("watchdog"), Mapping) else None,
    )
    if combined_watchdog:
        combined_watchdog = dict(combined_watchdog)
    else:
        combined_watchdog = {}

    retry_cfg = combined_watchdog.get("retry_policy")
    retry_policy: dict[str, object] = dict(retry_cfg) if isinstance(retry_cfg, Mapping) else {}
    retry_overridden = False

    cli_retry_exception_args = getattr(args, "watchdog_retry_exceptions", None) or []
    if cli_retry_exception_args:
        normalized_exceptions: list[str] = []
        for raw_exc in cli_retry_exception_args:
            if not isinstance(raw_exc, str) or not raw_exc.strip():
                raise CLIUsageError(
                    "Opcja --watchdog-retry-exception wymaga niepustej nazwy klasy wyjątku."
                )
            normalized_exceptions.append(raw_exc.strip())
        combined_watchdog["retry_exceptions"] = normalized_exceptions

    cli_retry_max_attempts = getattr(args, "watchdog_max_attempts", None)
    if cli_retry_max_attempts is not None:
        if cli_retry_max_attempts <= 0:
            raise CLIUsageError("Opcja --watchdog-max-attempts wymaga dodatniej wartości.")
        retry_policy["max_attempts"] = int(cli_retry_max_attempts)
        retry_overridden = True

    cli_retry_base_delay = getattr(args, "watchdog_base_delay", None)
    if cli_retry_base_delay is not None:
        if cli_retry_base_delay <= 0:
            raise CLIUsageError("Opcja --watchdog-base-delay wymaga dodatniej wartości.")
        retry_policy["base_delay"] = float(cli_retry_base_delay)
        retry_overridden = True

    cli_retry_max_delay = getattr(args, "watchdog_max_delay", None)
    if cli_retry_max_delay is not None:
        if cli_retry_max_delay <= 0:
            raise CLIUsageError("Opcja --watchdog-max-delay wymaga dodatniej wartości.")
        retry_policy["max_delay"] = float(cli_retry_max_delay)
        retry_overridden = True

    jitter_min_arg = getattr(args, "watchdog_jitter_min", None)
    jitter_max_arg = getattr(args, "watchdog_jitter_max", None)
    if jitter_min_arg is not None and jitter_min_arg < 0:
        raise CLIUsageError("Opcja --watchdog-jitter-min wymaga nieujemnej wartości.")
    if jitter_max_arg is not None and jitter_max_arg < 0:
        raise CLIUsageError("Opcja --watchdog-jitter-max wymaga nieujemnej wartości.")
    if jitter_min_arg is not None or jitter_max_arg is not None:
        existing_jitter = _extract_jitter_pair(retry_policy.get("jitter"))
        jitter_min = float(jitter_min_arg) if jitter_min_arg is not None else (existing_jitter[0] if existing_jitter else 0.0)
        jitter_max = float(jitter_max_arg) if jitter_max_arg is not None else (existing_jitter[1] if existing_jitter else 0.2)
        if jitter_max < jitter_min:
            raise CLIUsageError("Opcja --watchdog-jitter-max musi być większa lub równa wartości minimalnej.")
        retry_policy["jitter"] = (jitter_min, jitter_max)
        retry_overridden = True

    if retry_policy or retry_overridden:
        combined_watchdog["retry_policy"] = retry_policy

    circuit_cfg = combined_watchdog.get("circuit_breaker")
    circuit_breaker: dict[str, object] = dict(circuit_cfg) if isinstance(circuit_cfg, Mapping) else {}
    circuit_overridden = False

    cli_failure_threshold = getattr(args, "watchdog_failure_threshold", None)
    if cli_failure_threshold is not None:
        if cli_failure_threshold <= 0:
            raise CLIUsageError("Opcja --watchdog-failure-threshold wymaga dodatniej wartości.")
        circuit_breaker["failure_threshold"] = int(cli_failure_threshold)
        circuit_overridden = True

    cli_recovery_timeout = getattr(args, "watchdog_recovery_timeout", None)
    if cli_recovery_timeout is not None:
        if cli_recovery_timeout <= 0:
            raise CLIUsageError("Opcja --watchdog-recovery-timeout wymaga dodatniej wartości.")
        circuit_breaker["recovery_timeout"] = float(cli_recovery_timeout)
        circuit_overridden = True

    cli_half_open_success = getattr(args, "watchdog_half_open_success", None)
    if cli_half_open_success is not None:
        if cli_half_open_success <= 0:
            raise CLIUsageError("Opcja --watchdog-half-open-success wymaga dodatniej wartości.")
        circuit_breaker["half_open_success_threshold"] = int(cli_half_open_success)
        circuit_overridden = True

    if circuit_breaker or circuit_overridden:
        combined_watchdog["circuit_breaker"] = circuit_breaker

    watchdog_payload = _serialize_watchdog_config(combined_watchdog)

    if combined_watchdog:
        _configure_watchdog(manager, {"watchdog": combined_watchdog})

    native_settings: dict[str, object] = {}
    native_mode_override: Mode | None = None

    profile_settings = _extract_adapter_settings(profile)
    if profile_settings and manager.mode in {Mode.MARGIN, Mode.FUTURES}:
        native_settings.update(profile_settings)

    env_native = env_manager_cfg.get("native_adapter")
    if isinstance(env_native, Mapping):
        native_settings_cfg = env_native.get("settings")
        if isinstance(native_settings_cfg, Mapping):
            native_settings.update(native_settings_cfg)
        native_mode_raw = env_native.get("mode")
        if isinstance(native_mode_raw, str) and native_mode_raw.strip():
            native_mode_override = _normalize_native_mode(
                native_mode_raw,
                context="Konfiguracja natywnego adaptera",
            )

    cli_native_mode = getattr(args, "native_mode", None)
    if isinstance(cli_native_mode, str) and cli_native_mode.strip():
        native_mode_override = _normalize_native_mode(
            cli_native_mode,
            context="Opcja --native-mode",
        )

    cli_native_setting_args = getattr(args, "native_settings", None) or []
    cli_native_settings_provided = False
    for raw_argument in cli_native_setting_args:
        key, value = _parse_native_setting_argument(raw_argument)
        native_settings[key] = value
        cli_native_settings_provided = True

    should_configure_native = bool(native_settings) or native_mode_override is not None or cli_native_settings_provided
    if should_configure_native:
        target_mode = native_mode_override
        if target_mode is None and manager.mode in {Mode.MARGIN, Mode.FUTURES}:
            target_mode = manager.mode
        if target_mode is None:
            raise CLIUsageError(
                "Konfiguracja natywnego adaptera wymaga trybu margin lub futures. "
                "Ustaw --mode, profil lub użyj --native-mode."
            )
        manager.configure_native_adapter(settings=native_settings or {}, mode=target_mode)
        native_adapter_payload = {
            "mode": target_mode.value,
            "settings": dict(native_settings),
        }
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
    private_asset = None
    private_min_balance = None
    if isinstance(health_config, Mapping):
        asset_raw = health_config.get("private_asset")
        if isinstance(asset_raw, str) and asset_raw.strip():
            private_asset = asset_raw.strip().upper()
        private_min_balance = _coerce_optional_float(
            health_config.get("private_min_balance"), key="health_check.private_min_balance"
        )
    cli_public_symbol = getattr(args, "public_symbol", None)
    if isinstance(cli_public_symbol, str) and cli_public_symbol.strip():
        public_symbol = cli_public_symbol.strip()
    cli_private_asset = getattr(args, "private_asset", None)
    if isinstance(cli_private_asset, str) and cli_private_asset.strip():
        private_asset = cli_private_asset.strip().upper()
    cli_private_min_balance = getattr(args, "private_min_balance", None)
    if cli_private_min_balance is not None:
        private_min_balance = float(cli_private_min_balance)
    requested_checks = _normalize_requested_checks(getattr(args, "checks", None))
    if requested_checks:
        invalid = [name for name in requested_checks if name not in _SUPPORTED_HEALTH_CHECKS]
        if invalid:
            raise CLIUsageError(
                "Nieznane testy zdrowia: "
                + ", ".join(sorted(invalid))
                + ". Dostępne: "
                + ", ".join(_SUPPORTED_HEALTH_CHECKS)
                + "."
            )
    requested_set = set(requested_checks)

    skip_public = bool(args.skip_public)
    skip_private = bool(args.skip_private)
    if isinstance(health_config, Mapping):
        skip_public = skip_public or bool(health_config.get("skip_public"))
        skip_private = skip_private or bool(health_config.get("skip_private"))

    include_public = not skip_public and (not requested_checks or "public_api" in requested_set)
    include_private = not skip_private and (not requested_checks or "private_api" in requested_set)
    notes: list[str] = []
    public_disabled_reason: str | None = None
    private_disabled_reason: str | None = None

    if skip_public and (not requested_checks or "public_api" in requested_set):
        public_disabled_reason = "Test public_api został wyłączony (--skip-public lub konfiguracja)."
    if skip_private and (not requested_checks or "private_api" in requested_set):
        private_disabled_reason = "Test private_api został wyłączony (--skip-private lub konfiguracja)."

    if include_private:
        if manager.mode is Mode.PAPER:
            include_private = False
            private_disabled_reason = "Test private_api nie jest obsługiwany w trybie paper."
            notes.append("Pomijam test private_api w trybie paper.")
        elif not (api_key and secret):
            include_private = False
            private_disabled_reason = "Brak kompletnych poświadczeń API."
            notes.append("Pomijam test private_api – brak kompletnych poświadczeń API.")

    if requested_checks:
        if "public_api" in requested_set and not include_public:
            reason = public_disabled_reason or "Test public_api jest niedostępny w tym uruchomieniu."
            raise CLIUsageError(f"Nie można uruchomić testu public_api: {reason}")
        if "private_api" in requested_set and not include_private:
            reason = private_disabled_reason or "Test private_api jest niedostępny w tym uruchomieniu."
            raise CLIUsageError(f"Nie można uruchomić testu private_api: {reason}")

    checks = _build_health_checks(
        manager,
        include_public=include_public,
        include_private=include_private,
        public_symbol=public_symbol,
        private_asset=private_asset,
        private_min_balance=private_min_balance,
    )
    include_public = any(check.name == "public_api" for check in checks)
    include_private = any(check.name == "private_api" for check in checks)
    monitor = manager.create_health_monitor(checks)
    results = monitor.run()

    for note in notes:
        print(note, file=sys.stderr)

    overall = HealthMonitor.overall_status(results)

    output_path_arg = getattr(args, "output_path", None)

    if output_format not in {"json", "json-pretty"} and output_path_arg:
        raise CLIUsageError(
            "Opcja --output-path wymaga formatu wyjścia JSON (ustaw --output-format=json lub json-pretty)."
        )

    if output_format in {"json", "json-pretty"}:
        payload_environment = dict(environment_profile) if environment_profile else None
        if payload_environment is not None:
            payload_environment = dict(payload_environment)
        paper_payload = {
            "variant": manager.get_paper_variant(),
            "initial_cash": manager.get_paper_initial_cash(),
            "cash_asset": manager.get_paper_cash_asset(),
            "fee_rate": manager.get_paper_fee_rate(),
        }
        simulator_settings = manager.get_paper_simulator_settings()
        if simulator_settings:
            paper_payload["simulator"] = simulator_settings
        payload = {
            "exchange": exchange_id,
            "mode": manager.mode.value,
            "testnet": bool(testnet_flag),
            "environment": payload_environment,
            "environment_summary": environment_summary,
            "notes": notes,
            "requested_checks": list(requested_checks) if requested_checks else None,
            "include_public": include_public,
            "include_private": include_private,
            "private_asset": private_asset,
            "private_min_balance": private_min_balance,
            "paper": paper_payload,
            "watchdog": watchdog_payload,
            "native_adapter": native_adapter_payload,
            "results": [
                {
                    "name": result.name,
                    "status": result.status.value,
                    "latency": result.latency,
                    "details": dict(result.details),
                }
                for result in results
            ],
            "overall_status": overall.value,
        }
        json_kwargs: dict[str, object] = {"ensure_ascii": False}
        if output_format == "json-pretty":
            json_kwargs["indent"] = 2
            json_kwargs["sort_keys"] = True
        rendered_payload = json.dumps(payload, **json_kwargs)  # type: ignore[arg-type]
        print(rendered_payload)
        if output_path_arg:
            destination = Path(output_path_arg).expanduser()
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(rendered_payload + "\n", encoding="utf-8")
            except OSError as exc:
                raise CLIUsageError(f"Nie udało się zapisać wyniku do pliku {destination}: {exc}") from exc
    else:
        for result in results:
            latency = f"{result.latency:.3f}s"
            details = _format_details(result.details)
            print(f"{result.name}: {result.status.value} (latency={latency}){details}")
        print(f"Overall status: {overall.value}")

    exit_code_map = {
        HealthStatus.HEALTHY: 0,
        HealthStatus.DEGRADED: 1,
        HealthStatus.UNAVAILABLE: 2,
    }
    return exit_code_map.get(overall, 2)


def list_environments(args: argparse.Namespace) -> int:
    config_path = getattr(args, "environment_config", None) or str(DEFAULT_ENVIRONMENT_CONFIG_PATH)
    storage, defaults, environments = _read_environment_payload(config_path)

    defaults_dict = dict(defaults or {})
    default_environment_raw = defaults_dict.get("environment")
    default_environment = (
        str(default_environment_raw).strip()
        if isinstance(default_environment_raw, str) and default_environment_raw.strip()
        else None
    )
    default_exchange = defaults_dict.get("exchange")
    default_exchange = (
        str(default_exchange).strip()
        if isinstance(default_exchange, str) and str(default_exchange).strip()
        else None
    )
    defaults_manager = (
        defaults_dict.get("exchange_manager")
        if isinstance(defaults_dict.get("exchange_manager"), Mapping)
        else None
    )

    print(f"Zdefiniowane środowiska ({storage}):")
    if not environments:
        print("  (brak środowisk)")
        return 0

    for name in sorted(environments):
        section = environments[name]
        description_raw = section.get("description")
        description = (
            str(description_raw).strip()
            if isinstance(description_raw, str) and description_raw.strip()
            else ""
        )
        exchange_raw = section.get("exchange")
        exchange = (
            str(exchange_raw).strip()
            if isinstance(exchange_raw, str) and exchange_raw.strip()
            else default_exchange
        )

        manager_cfg = section.get("exchange_manager") if isinstance(section.get("exchange_manager"), Mapping) else None
        merged_manager = _merge_mappings(defaults_manager, manager_cfg) if (defaults_manager or manager_cfg) else {}

        mode_raw = merged_manager.get("mode")
        mode = (
            str(mode_raw).strip()
            if isinstance(mode_raw, str) and str(mode_raw).strip()
            else None
        )

        summary_parts: list[str] = []
        if exchange:
            summary_parts.append(f"exchange={exchange}")
        if mode:
            summary_parts.append(f"mode={mode}")

        testnet_flag = merged_manager.get("testnet")
        if testnet_flag is not None:
            summary_parts.append(f"testnet={'true' if bool(testnet_flag) else 'false'}")

        paper_variant = merged_manager.get("paper_variant")
        if isinstance(paper_variant, str) and paper_variant.strip():
            summary_parts.append(f"paper_variant={paper_variant.strip()}")

        default_mark = " (default)" if default_environment and name == default_environment else ""
        summary = " [" + ", ".join(summary_parts) + "]" if summary_parts else ""

        description_suffix = f" - {description}" if description else ""
        print(f"  * {name}{default_mark}{summary}{description_suffix}")

    return 0


def show_environment(args: argparse.Namespace) -> int:
    environment_name = getattr(args, "environment", None)
    if not environment_name:
        raise CLIUsageError("Komenda show-environment wymaga podania nazwy środowiska (--environment).")

    config_path = getattr(args, "environment_config", None) or str(DEFAULT_ENVIRONMENT_CONFIG_PATH)
    profile = _load_environment_profile(config_path, environment_name)

    storage = profile.get("__path__")
    source_path = str(storage) if isinstance(storage, str) else config_path

    print(f"Środowisko '{environment_name}' ({source_path}):")

    printable = dict(profile)
    printable.pop("__path__", None)

    if not printable:
        print("  (pusta konfiguracja)")
        return 0

    if yaml is not None:  # pragma: no cover - fallback dla braku PyYAML
        rendered = yaml.safe_dump(  # type: ignore[union-attr]
            printable,
            sort_keys=True,
            allow_unicode=True,
            default_flow_style=False,
        ).strip()
    else:  # pragma: no cover - środowiska wymagają PyYAML, ale zapewniamy czytelny fallback
        rendered = repr(printable)

    for line in rendered.splitlines():
        print(f"  {line}")

    return 0


def show_strategy_catalog(args: argparse.Namespace) -> int:
    output_format = getattr(args, "output_format", "text") or "text"
    engine_filter = {value.strip().lower() for value in getattr(args, "engines", []) if value}
    capability_filter = {
        value.strip().lower() for value in getattr(args, "capabilities", []) if value
    }
    tag_filter = {value.strip().lower() for value in getattr(args, "tags", []) if value}

    engines = []
    for entry in DEFAULT_STRATEGY_CATALOG.describe_engines():
        engine_name = str(entry.get("engine", ""))
        normalized_engine = engine_name.lower()
        capability = str(entry.get("capability", "") or "")
        normalized_capability = capability.lower()
        tags = [str(tag) for tag in entry.get("default_tags", [])]
        normalized_tags = {tag.lower() for tag in tags}
        if engine_filter and normalized_engine not in engine_filter:
            continue
        if capability_filter and normalized_capability not in capability_filter:
            continue
        if tag_filter and not (normalized_tags & tag_filter):
            continue
        engines.append(entry)

    definitions: list[Mapping[str, object]] = []
    config_path = getattr(args, "config", None)
    scheduler_name = getattr(args, "scheduler", None)
    if config_path:
        try:
            core_config = load_core_config(config_path)
        except Exception as exc:  # pragma: no cover - błędy IO/parsingu
            raise CLIUsageError(f"Nie udało się wczytać konfiguracji {config_path}: {exc}") from exc
        definitions = describe_strategy_definitions(core_config)
        if scheduler_name:
            try:
                plan = describe_multi_strategy_configuration(
                    config_path=config_path,
                    scheduler_name=scheduler_name,
                    include_strategy_definitions=True,
                    only_scheduler_definitions=True,
                )
                definitions = plan.get("strategies", [])  # type: ignore[assignment]
            except Exception as exc:  # pragma: no cover - walidacja konfiguracji
                raise CLIUsageError(str(exc)) from exc
        filtered_definitions: list[Mapping[str, object]] = []
        for entry in definitions:
            engine_name = str(entry.get("engine", ""))
            normalized_engine = engine_name.lower()
            capability = str(entry.get("capability", "") or "")
            normalized_capability = capability.lower()
            tags = [str(tag) for tag in entry.get("tags", [])]
            normalized_tags = {tag.lower() for tag in tags}
            if engine_filter and normalized_engine not in engine_filter:
                continue
            if capability_filter and normalized_capability not in capability_filter:
                continue
            if tag_filter and not (normalized_tags & tag_filter):
                continue
            payload = dict(entry)
            if not getattr(args, "include_parameters", False):
                payload.pop("parameters", None)
                payload.pop("metadata", None)
            filtered_definitions.append(payload)
        definitions = filtered_definitions

    payload: dict[str, object] = {"engines": engines}
    if config_path:
        payload["config_path"] = str(Path(config_path).expanduser())
        if scheduler_name:
            payload["scheduler"] = scheduler_name
        payload["definitions"] = definitions

    if output_format in {"json", "json-pretty"}:
        json_kwargs: dict[str, object] = {"ensure_ascii": False}
        if output_format == "json-pretty":
            json_kwargs["indent"] = 2
            json_kwargs["sort_keys"] = True
        print(json.dumps(payload, **json_kwargs))  # type: ignore[arg-type]
        return 0

    print("Silniki strategii dostępne w katalogu:")
    if not engines:
        print("  (brak wyników po zastosowaniu filtrów)")
    else:
        for entry in engines:
            engine = entry.get("engine", "(nieznany)")
            capability = entry.get("capability") or "-"
            tags = ", ".join(entry.get("default_tags", [])) or "-"
            license_tier = entry.get("license_tier") or "-"
            risk_classes = ", ".join(str(item) for item in entry.get("risk_classes", [])) or "-"
            required_data = ", ".join(str(item) for item in entry.get("required_data", [])) or "-"
            print(
                "  * {engine} (capability={capability}, license={license}, risk_classes=[{risk}], required_data=[{data}], tags={tags})".format(
                    engine=engine,
                    capability=capability,
                    license=license_tier,
                    risk=risk_classes,
                    data=required_data,
                    tags=tags,
                )
            )

    if config_path:
        print()
        if not definitions:
            print("Definicje strategii: (brak wyników po zastosowaniu filtrów)")
        else:
            print("Definicje strategii:")
            for entry in definitions:
                name = entry.get("name", "(bez nazwy)")
                engine = entry.get("engine", "(nieznany)")
                risk_profile = entry.get("risk_profile") or "-"
                tags = ", ".join(entry.get("tags", [])) or "-"
                capability = entry.get("capability") or "-"
                license_tier = entry.get("license_tier") or "-"
                risk_classes = ", ".join(str(item) for item in entry.get("risk_classes", [])) or "-"
                required_data = ", ".join(str(item) for item in entry.get("required_data", [])) or "-"
                print(
                    "  * {name} -> {engine} (risk_profile={risk_profile}, capability={capability}, license={license}, risk_classes=[{risk}], required_data=[{data}], tags={tags})".format(
                        name=name,
                        engine=engine,
                        risk_profile=risk_profile,
                        capability=capability,
                        license=license_tier,
                        risk=risk_classes,
                        data=required_data,
                        tags=tags,
                    )
                )
    return 0


def show_scheduler_plan(args: argparse.Namespace) -> int:
    output_format = getattr(args, "output_format", "text") or "text"
    include_definitions = bool(getattr(args, "include_definitions", True))
    scheduler_name = getattr(args, "scheduler", None)
    try:
        plan = describe_multi_strategy_configuration(
            config_path=args.config,
            scheduler_name=scheduler_name,
            include_strategy_definitions=include_definitions,
            only_scheduler_definitions=include_definitions and bool(scheduler_name),
        )
    except Exception as exc:  # pragma: no cover - walidacja konfiguracji
        raise CLIUsageError(str(exc)) from exc

    filter_tags = {value.strip().lower() for value in getattr(args, "filter_tags", []) if value}
    filter_strategies = {
        value.strip().lower() for value in getattr(args, "filter_strategies", []) if value
    }

    schedules = []
    for entry in plan.get("schedules", []):
        name = str(entry.get("name", ""))
        strategy = str(entry.get("strategy", ""))
        normalized_name = name.lower()
        normalized_strategy = strategy.lower()
        if filter_strategies and normalized_strategy not in filter_strategies and normalized_name not in filter_strategies:
            continue
        tags = {str(tag).lower() for tag in entry.get("tags", [])}
        if filter_tags and not (tags & filter_tags):
            continue
        schedules.append(entry)

    plan["schedules"] = schedules

    if include_definitions and "strategies" in plan:
        used_strategies = {entry.get("strategy") for entry in schedules}
        filtered_definitions = [
            entry
            for entry in plan.get("strategies", [])
            if entry.get("name") in used_strategies or not used_strategies
        ]
        plan["strategies"] = filtered_definitions

    if output_format in {"json", "json-pretty"}:
        json_kwargs: dict[str, object] = {"ensure_ascii": False}
        if output_format == "json-pretty":
            json_kwargs["indent"] = 2
            json_kwargs["sort_keys"] = True
        print(json.dumps(plan, **json_kwargs))  # type: ignore[arg-type]
        return 0

    config_path = plan.get("config_path")
    scheduler_label = plan.get("scheduler")
    print(
        "Plan scheduler-a '{scheduler}' (config={config})".format(
            scheduler=scheduler_label,
            config=config_path,
        )
    )
    policy = plan.get("capital_policy", {})
    policy_name = policy.get("name", "(nieznana)")
    interval = policy.get("configured_rebalance_seconds") or policy.get("policy_interval_seconds")
    interval_text = f", rebalance_interval={interval}" if interval else ""
    print(f"Polityka kapitału: {policy_name}{interval_text}")
    print(f"Harmonogramy ({len(schedules)}):")
    if not schedules:
        print("  (brak wyników po zastosowaniu filtrów)")
    for entry in schedules:
        tags = ", ".join(entry.get("tags", [])) or "-"
        interval = entry.get("interval") or "-"
        print(
            "  * {name}: {strategy} [profile={profile}] cadence={cadence}s drift={drift}s max_signals={max_signals} interval={interval} license={license} risk_classes=[{risk}] required_data=[{data}] tags={tags}".format(
                name=entry.get("name", "(bez nazwy)"),
                strategy=entry.get("strategy", "(nieznana)"),
                profile=entry.get("risk_profile", "-"),
                cadence=entry.get("cadence_seconds", "-"),
                drift=entry.get("max_drift_seconds", "-"),
                max_signals=entry.get("max_signals", "-"),
                interval=interval,
                license=entry.get("license_tier", "-"),
                risk=", ".join(str(item) for item in entry.get("risk_classes", [])) or "-",
                data=", ".join(str(item) for item in entry.get("required_data", [])) or "-",
                tags=tags,
            )
        )

    suspensions = plan.get("initial_suspensions", [])
    if suspensions:
        print("Początkowe zawieszenia:")
        for suspension in suspensions:
            reason = suspension.get("reason") or "-"
            until = suspension.get("until") or "-"
            duration = suspension.get("duration_seconds")
            duration_text = f", duration={duration}s" if duration is not None else ""
            print(
                f"  * {suspension.get('kind')}::{suspension.get('target')} (reason={reason}, until={until}{duration_text})"
            )

    if plan.get("initial_signal_limits"):
        print("Początkowe nadpisania limitów sygnałów:")
        for strategy_name, profiles in plan["initial_signal_limits"].items():
            for profile, payload in profiles.items():
                reason = payload.get("reason") or "-"
                until = payload.get("until") or "-"
                print(
                    f"  * {strategy_name}/{profile}: limit={payload.get('limit')} reason={reason} until={until}"
                )

    if plan.get("signal_limits"):
        print("Stałe limity sygnałów:")
        for strategy_name, profiles in plan["signal_limits"].items():
            for profile, payload in profiles.items():
                print(f"  * {strategy_name}/{profile}: limit={payload.get('limit')}")

    return 0

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
        if args.command == "list-environments":
            return list_environments(args)
        if args.command == "show-environment":
            return show_environment(args)
        if args.command == "strategy-catalog":
            return show_strategy_catalog(args)
        if args.command == "scheduler-plan":
            return show_scheduler_plan(args)
        raise CLIUsageError(f"Nieznana komenda: {args.command}")
    except CLIUsageError as exc:
        print(f"Błąd: {exc}", file=sys.stderr)
        return 2


__all__ = [
    "main",
    "create_parser",
    "run_health_check",
    "CLIUsageError",
    "list_environments",
    "show_environment",
    "show_strategy_catalog",
    "show_scheduler_plan",
]
