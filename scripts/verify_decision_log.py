"""Walidacja podpisanych decision logów telemetrii UI.

Skrypt wczytuje plik JSONL (również w wariancie .gz lub ze standardowego
wejścia) wygenerowany przez `watch_metrics_stream` i weryfikuje podpisy
HMAC-SHA256.  Obsługuje konfigurację przez argumenty CLI oraz zmienne
środowiskowe z prefiksem `BOT_CORE_VERIFY_DECISION_LOG_`.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import codecs
from collections import defaultdict
from copy import deepcopy
import gzip
import hmac
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# --- opcjonalna konfiguracja core.yaml ---------------------------------------
try:  # pragma: no cover - brak modułu konfiguracji
    from bot_core.config import load_core_config  # type: ignore
except Exception:  # pragma: no cover
    load_core_config = None  # type: ignore

# --- presety profili ryzyka (z fallbackiem, patrz scripts.telemetry_risk_profiles) --
from scripts.telemetry_risk_profiles import (
    get_metrics_service_env_overrides,
    get_risk_profile,
    list_risk_profile_names,
    load_risk_profiles_with_metadata,
    reset_risk_profile_store,
    risk_profile_metadata,
    summarize_risk_profile,
)

try:  # pragma: no cover - PyYAML opcjonalny
    import yaml  # type: ignore
except Exception:  # pragma: no cover - środowiska minimalne
    yaml = None  # type: ignore

LOGGER = logging.getLogger("bot_core.scripts.verify_decision_log")

_ENV_PREFIX = "BOT_CORE_VERIFY_DECISION_LOG_"

_ENV_EXPECT_FILTERS = f"{_ENV_PREFIX}EXPECT_FILTERS_JSON"
_ENV_REQUIRE_SCREEN_INFO = f"{_ENV_PREFIX}REQUIRE_SCREEN_INFO"
_ENV_SUMMARY_PATH = f"{_ENV_PREFIX}SUMMARY_JSON"
_ENV_REPORT_OUTPUT = f"{_ENV_PREFIX}REPORT_OUTPUT"
_ENV_MAX_EVENT_COUNTS = f"{_ENV_PREFIX}MAX_EVENT_COUNTS_JSON"
_ENV_MIN_EVENT_COUNTS = f"{_ENV_PREFIX}MIN_EVENT_COUNTS_JSON"
_ENV_RISK_PROFILE = f"{_ENV_PREFIX}RISK_PROFILE"
_ENV_RISK_PROFILES_FILE = f"{_ENV_PREFIX}RISK_PROFILES_FILE"
_ENV_PRINT_RISK_PROFILES = f"{_ENV_PREFIX}PRINT_RISK_PROFILES"
_ENV_CORE_CONFIG = f"{_ENV_PREFIX}CORE_CONFIG"
_ENV_RISK_PROFILE_ENV_SNIPPET = f"{_ENV_PREFIX}RISK_PROFILE_ENV_SNIPPET"
_ENV_RISK_PROFILE_YAML_SNIPPET = f"{_ENV_PREFIX}RISK_PROFILE_YAML_SNIPPET"
_ENV_REQUIRE_TLS_MATERIALS = f"{_ENV_PREFIX}REQUIRE_TLS_MATERIALS"
_ENV_EXPECT_SERVER_SHA256 = f"{_ENV_PREFIX}EXPECT_SERVER_SHA256"
_ENV_EXPECT_SERVER_SHA256_SOURCE = f"{_ENV_PREFIX}EXPECT_SERVER_SHA256_SOURCE"

# rozszerzenia dot. RBAC/TLS dla RiskService i auth-scope
_ENV_REQUIRE_AUTH_SCOPE = f"{_ENV_PREFIX}REQUIRE_AUTH_SCOPE"
_ENV_REQUIRE_RISK_SCOPE = f"{_ENV_PREFIX}REQUIRE_RISK_SERVICE_SCOPE"
_ENV_REQUIRE_RISK_TOKEN_ID = f"{_ENV_PREFIX}REQUIRE_RISK_SERVICE_TOKEN_ID"

_ENV_REQUIRE_RISK_TLS = f"{_ENV_PREFIX}REQUIRE_RISK_SERVICE_TLS"
_ENV_REQUIRE_RISK_TLS_MATERIALS = f"{_ENV_PREFIX}REQUIRE_RISK_SERVICE_TLS_MATERIALS"
_ENV_EXPECT_RISK_SERVER_SHA256 = f"{_ENV_PREFIX}EXPECT_RISK_SERVICE_SERVER_SHA256"
_ENV_REQUIRE_RISK_AUTH_TOKEN = f"{_ENV_PREFIX}REQUIRE_RISK_SERVICE_AUTH_TOKEN"

_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


class VerificationError(RuntimeError):
    """Zgłaszane, gdy walidacja decision logu nie powiedzie się."""


@dataclass
class _SigningKey:
    value: bytes
    key_id: str | None


_SEVERITY_ORDER = [
    "trace",
    "debug",
    "info",
    "notice",
    "warning",
    "error",
    "critical",
    "alert",
    "emergency",
    "fatal",
]

_SEVERITY_RANK = {name: index for index, name in enumerate(_SEVERITY_ORDER)}

_TLS_MATERIAL_CHOICES = (
    "root_cert",
    "client_cert",
    "client_key",
    "server_name",
    "server_sha256",
)

# rozszerzenie: osobna lista materiałów TLS dla RiskService
_RISK_TLS_MATERIAL_CHOICES = (
    "root_cert",
    "client_cert",
    "client_key",
    "client_auth",
)

_HEX_DIGITS = set("0123456789abcdef")


class _SummaryAggregator:
    def __init__(self) -> None:
        self._total = 0
        self._first_ts: datetime | None = None
        self._last_ts: datetime | None = None
        self._severity_totals: defaultdict[str, int] = defaultdict(int)
        self._events: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "count": 0,
                "fps_count": 0,
                "fps_total": 0.0,
                "fps_min": None,
                "fps_max": None,
                "screens": set(),
                "severity_counts": defaultdict(int),
                "first_ts": None,
                "last_ts": None,
            }
        )

    def add_snapshot(self, entry: Mapping[str, Any]) -> None:
        timestamp = entry.get("timestamp")
        ts_dt = _parse_iso_datetime(timestamp)
        if ts_dt:
            if self._first_ts is None or ts_dt < self._first_ts:
                self._first_ts = ts_dt
            if self._last_ts is None or ts_dt > self._last_ts:
                self._last_ts = ts_dt

        event = entry.get("event")
        if not isinstance(event, str) or not event:
            event = "unknown"

        stats = self._events[event]
        stats["count"] += 1

        severity = _normalize_severity(entry.get("severity"))
        if severity:
            stats["severity_counts"][severity] += 1
            self._severity_totals[severity] += 1

        if ts_dt:
            if stats["first_ts"] is None or ts_dt < stats["first_ts"]:
                stats["first_ts"] = ts_dt
            if stats["last_ts"] is None or ts_dt > stats["last_ts"]:
                stats["last_ts"] = ts_dt

        fps_value = entry.get("fps")
        if isinstance(fps_value, (int, float)):
            fps_float = float(fps_value)
            stats["fps_count"] += 1
            stats["fps_total"] += fps_float
            stats["fps_min"] = (
                fps_float
                if stats["fps_min"] is None or fps_float < stats["fps_min"]
                else stats["fps_min"]
            )
            stats["fps_max"] = (
                fps_float
                if stats["fps_max"] is None or fps_float > stats["fps_max"]
                else stats["fps_max"]
            )

        screen_ctx = entry.get("screen") if isinstance(entry.get("screen"), Mapping) else None
        if screen_ctx:
            encoded = json.dumps(screen_ctx, ensure_ascii=False, sort_keys=True)
            stats["screens"].add(encoded)

        self._total += 1

    def as_dict(self) -> dict[str, Any]:
        events_summary: dict[str, Any] = {}
        for event_name in sorted(self._events):
            stats = self._events[event_name]
            payload: dict[str, Any] = {"count": stats["count"]}
            if stats["fps_count"]:
                payload["fps"] = {
                    "min": stats["fps_min"],
                    "max": stats["fps_max"],
                    "avg": stats["fps_total"] / stats["fps_count"],
                    "samples": stats["fps_count"],
                }
            if stats["screens"]:
                payload["screens"] = [
                    json.loads(encoded) for encoded in sorted(stats["screens"])
                ]
            if stats["severity_counts"]:
                payload["severity"] = {
                    "counts": {
                        level: stats["severity_counts"][level]
                        for level in sorted(stats["severity_counts"])
                    }
                }
            if stats["first_ts"]:
                payload["first_timestamp"] = stats["first_ts"].isoformat()
            if stats["last_ts"]:
                payload["last_timestamp"] = stats["last_ts"].isoformat()
            events_summary[event_name] = payload

        summary: dict[str, Any] = {"total_snapshots": self._total, "events": events_summary}
        if self._first_ts:
            summary["first_timestamp"] = self._first_ts.isoformat()
        if self._last_ts:
            summary["last_timestamp"] = self._last_ts.isoformat()
        if self._severity_totals:
            summary["severity_counts"] = {
                level: self._severity_totals[level]
                for level in sorted(self._severity_totals)
            }
        return summary


def _format_env_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if value is None:
        return ""
    return str(value)


def _env_override_key(option: str) -> str:
    return "RUN_TRADING_STUB_METRICS_" + option.replace("-", "_").upper()


def _normalize_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    if not overrides:
        return {}
    normalized: dict[str, Any] = {}
    for key, value in overrides.items():
        normalized[str(key)] = value
    return normalized


def _expected_env_assignments(overrides: Mapping[str, Any]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for key, value in overrides.items():
        expected[_env_override_key(str(key))] = _format_env_override_value(value)
    return expected


def _unescape_env_value(raw: str) -> str:
    if raw.startswith("\"") and raw.endswith("\""):
        inner = raw[1:-1]
        try:
            return codecs.decode(inner, "unicode_escape")
        except Exception:  # pragma: no cover - defensywne
            return inner.replace("\\\"", "\"").replace("\\\\", "\\")
    if raw.startswith("'") and raw.endswith("'"):
        inner = raw[1:-1]
        return inner.replace("\\'", "'").replace("\\\\", "\\")
    return raw


def _parse_env_snippet(path: Path) -> dict[str, str]:
    content = path.read_text(encoding="utf-8")
    assignments: dict[str, str] = {}
    for index, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise VerificationError(
                f"Linia {index} pliku {path} nie zawiera przypisania KEY=VALUE"
            )
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise VerificationError(
                f"Linia {index} pliku {path} zawiera pustą nazwę zmiennej środowiskowej"
            )
        assignments[key] = _unescape_env_value(value.strip())
    return assignments


def _load_yaml_like(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        if yaml is None:
            raise VerificationError(
                f"Plik {path} nie jest poprawnym JSON-em, a PyYAML nie jest dostępny do parsowania"
            )
        data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise VerificationError(f"Plik {path} nie zawiera słownika z konfiguracją snippetów")
    return data


def _validate_risk_profile_snippets(
    *,
    env_path: str | None,
    yaml_path: str | None,
    recommended_overrides: Mapping[str, Any] | None,
    profile_name: str | None,
) -> tuple[list[Mapping[str, Any]], list[str]]:
    validations: list[Mapping[str, Any]] = []
    errors: list[str] = []
    overrides = _normalize_overrides(recommended_overrides)
    profile_label = profile_name or "unknown"

    if env_path:
        entry: dict[str, Any] = {"type": "env", "path": env_path}
        validations.append(entry)
        target = Path(env_path).expanduser()
        try:
            actual = _parse_env_snippet(target)
        except FileNotFoundError:
            entry["status"] = "missing"
            entry["error"] = "file_missing"
            errors.append(f"Brak pliku env snippet: {env_path}")
        except VerificationError as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)
            errors.append(str(exc))
        else:
            if not overrides:
                entry["status"] = "error"
                entry["error"] = "missing_recommended_overrides"
                errors.append(
                    "Brak recommended_overrides w podsumowaniu profilu ryzyka – nie można porównać snippetów"
                )
            else:
                expected = _expected_env_assignments(overrides)
                missing = sorted(key for key in expected if key not in actual)
                extra = sorted(key for key in actual if key not in expected)
                mismatched = {
                    key: {"expected": expected[key], "actual": actual[key]}
                    for key in expected
                    if key in actual and actual[key] != expected[key]
                }
                entry["expected_keys"] = sorted(expected)
                if not missing and not extra and not mismatched:
                    entry["status"] = "ok"
                else:
                    entry["status"] = "mismatch"
                    details: dict[str, Any] = {}
                    if missing:
                        details["missing"] = missing
                    if extra:
                        details["extra"] = extra
                    if mismatched:
                        details["mismatched"] = mismatched
                    if details:
                        entry["details"] = details
                    errors.append(
                        "Plik env snippet nie zgadza się z recommended_overrides profilu "
                        f"{profile_label}"
                    )

    if yaml_path:
        entry = {"type": "yaml", "path": yaml_path}
        validations.append(entry)
        target = Path(yaml_path).expanduser()
        try:
            payload = _load_yaml_like(target)
        except FileNotFoundError:
            entry["status"] = "missing"
            entry["error"] = "file_missing"
            errors.append(f"Brak pliku YAML/JSON snippet: {yaml_path}")
        except VerificationError as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)
            errors.append(str(exc))
        else:
            section: Mapping[str, Any] | None = None
            if isinstance(payload.get("metrics_service_overrides"), Mapping):
                section = payload["metrics_service_overrides"]  # type: ignore[index]
            elif isinstance(payload, Mapping):
                section = payload
            if not overrides:
                entry["status"] = "error"
                entry["error"] = "missing_recommended_overrides"
                errors.append(
                    "Brak recommended_overrides w podsumowaniu profilu ryzyka – nie można porównać snippetów"
                )
            elif not isinstance(section, Mapping):
                entry["status"] = "error"
                entry["error"] = "missing_override_section"
                errors.append(
                    f"Plik {yaml_path} nie zawiera sekcji metrics_service_overrides do weryfikacji"
                )
            else:
                expected_map = _normalize_overrides(overrides)
                actual_map = _normalize_overrides(section)
                missing = sorted(key for key in expected_map if key not in actual_map)
                extra = sorted(key for key in actual_map if key not in expected_map)
                mismatched = {
                    key: {"expected": expected_map[key], "actual": actual_map[key]}
                    for key in expected_map
                    if key in actual_map and actual_map[key] != expected_map[key]
                }
                if not missing and not extra and not mismatched:
                    entry["status"] = "ok"
                else:
                    entry["status"] = "mismatch"
                    details = {}
                    if missing:
                        details["missing"] = missing
                    if extra:
                        details["extra"] = extra
                    if mismatched:
                        details["mismatched"] = mismatched
                    if details:
                        entry["details"] = details
                    errors.append(
                        "Plik YAML/JSON snippet nie zgadza się z recommended_overrides profilu "
                        f"{profile_label}"
                    )

    return validations, errors


def _normalize_severity(candidate: Any) -> str | None:
    if not isinstance(candidate, str):
        return None
    normalized = candidate.strip().lower()
    if not normalized:
        return None
    return normalized


def _severity_at_least(candidate: str, minimum: str) -> bool:
    candidate_rank = _SEVERITY_RANK.get(candidate)
    minimum_rank = _SEVERITY_RANK.get(minimum)
    if candidate_rank is None or minimum_rank is None:
        return False
    return candidate_rank >= minimum_rank


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip()
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _parse_env_bool(value: str, *, variable: str, parser: argparse.ArgumentParser) -> bool:
    lowered = value.strip().lower()
    if lowered in _BOOL_TRUE:
        return True
    if lowered in _BOOL_FALSE:
        return False
    parser.error(f"Nieprawidłowa wartość {variable}={value!r}; oczekiwano wartości bool")
    raise AssertionError("unreachable")


def _parse_env_list(
    value: str,
    *,
    variable: str,
    parser: argparse.ArgumentParser,
) -> list[str]:
    raw = (value or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        candidates = [item.strip() for item in raw.replace(";", ",").split(",")]
        return [item for item in candidates if item]
    if isinstance(parsed, str):
        parsed_list = [parsed.strip()]
    elif isinstance(parsed, list):
        parsed_list = []
        for item in parsed:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    parsed_list.append(stripped)
            elif item is None:
                continue
            else:
                parser.error(
                    f"{variable} może zawierać jedynie wartości tekstowe (lub null w tablicy JSON)"
                )
    else:
        parser.error(f"{variable} musi być listą JSON lub pojedynczym tekstem")
        parsed_list = []
    return [entry for entry in parsed_list if entry]


def _normalize_server_sha256(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip().lower().replace(":", "")
    if not stripped:
        return None
    if any(char not in _HEX_DIGITS for char in stripped):
        return None
    return stripped


def _coerce_expected_value(raw: str) -> Any:
    trimmed = raw.strip()
    lowered = trimmed.lower()
    if lowered in _BOOL_TRUE:
        return True
    if lowered in _BOOL_FALSE:
        return False
    if trimmed.startswith("[") or trimmed.startswith("{"):
        try:
            return json.loads(trimmed)
        except json.JSONDecodeError:
            pass
    try:
        return int(trimmed)
    except ValueError:
        try:
            return float(trimmed)
        except ValueError:
            return trimmed


def _parse_filter_spec(spec: str, *, parser: argparse.ArgumentParser) -> tuple[str, Any]:
    if "=" not in spec:
        parser.error("--expect-filter wymaga formatu klucz=wartość")
    key, raw_value = spec.split("=", 1)
    key = key.strip()
    if not key:
        parser.error("--expect-filter wymaga niepustego klucza")
    value = _coerce_expected_value(raw_value)
    return key, value


def _load_key_from_file(path: str) -> bytes:
    try:
        payload = Path(path).expanduser().read_bytes()
    except OSError as exc:
        raise VerificationError(f"Nie udało się odczytać klucza z {path}: {exc}") from exc
    stripped = payload.strip()
    if not stripped:
        raise VerificationError(f"Plik klucza {path} jest pusty")
    return stripped


def _load_signing_key(
    *,
    cli_value: str | None,
    cli_file: str | None,
    cli_key_id: str | None,
    env_value: str | None,
    env_file: str | None,
    env_key_id: str | None,
    env_allow_unsigned: str | None,
    parser: argparse.ArgumentParser,
) -> tuple[_SigningKey | None, bool]:
    allow_unsigned = False
    if env_allow_unsigned is not None:
        allow_unsigned = _parse_env_bool(env_allow_unsigned, variable=f"{_ENV_PREFIX}ALLOW_UNSIGNED", parser=parser)

    value = cli_value or env_value
    file_path = cli_file or env_file

    if value and file_path:
        parser.error("Użyj tylko jednego źródła klucza: --hmac-key lub --hmac-key-file")
    key_bytes: bytes | None = None
    key_id: str | None = None

    if value:
        key_bytes = value.encode("utf-8")
    elif file_path:
        key_bytes = _load_key_from_file(file_path)

    if cli_value and env_value and cli_value != env_value:
        LOGGER.warning("Wartość --hmac-key nadpisuje zmienną środowiskową %sHMAC_KEY", _ENV_PREFIX)
    if cli_file and env_file and cli_file != env_file:
        LOGGER.warning("Ścieżka --hmac-key-file nadpisuje %sHMAC_KEY_FILE", _ENV_PREFIX)

    if cli_key_id and env_key_id and cli_key_id != env_key_id:
        parser.error("Konflikt pomiędzy --hmac-key-id a %sHMAC_KEY_ID" % _ENV_PREFIX)

    key_id = cli_key_id or env_key_id or None
    if key_bytes is not None:
        return _SigningKey(value=key_bytes, key_id=key_id), allow_unsigned
    if cli_key_id:
        LOGGER.warning("Zignorowano --hmac-key-id bez dostarczonego klucza HMAC")
    return None, allow_unsigned


def _load_expected_filters(
    *,
    cli_filters: Iterable[str],
    parser: argparse.ArgumentParser,
) -> dict[str, Any]:
    expected: dict[str, Any] = {}
    for spec in cli_filters:
        key, value = _parse_filter_spec(spec, parser=parser)
        expected[key] = value
    return expected


def _parse_event_limit_spec(spec: str, *, parser: argparse.ArgumentParser) -> tuple[str, int]:
    if "=" not in spec:
        parser.error(
            "Specyfikacja limitu zdarzeń musi mieć postać NAZWA=LICZBA, np. reduce_motion=5"
        )
    raw_event, raw_limit = spec.split("=", 1)
    event = raw_event.strip()
    if not event:
        parser.error("Nazwa zdarzenia w limicie nie może być pusta")
    raw_limit = raw_limit.strip()
    if not raw_limit:
        parser.error(f"Limit dla zdarzenia {event!r} nie może być pusty")
    try:
        value = int(raw_limit)
    except ValueError:
        parser.error(f"Limit dla zdarzenia {event!r} musi być liczbą całkowitą")
    if value < 0:
        parser.error(f"Limit dla zdarzenia {event!r} nie może być ujemny")
    return event, value


def _load_event_count_specs(
    *, cli_limits: Iterable[str], parser: argparse.ArgumentParser
) -> dict[str, int]:
    limits: dict[str, int] = {}
    for spec in cli_limits:
        event, value = _parse_event_limit_spec(spec, parser=parser)
        limits[event] = value
    return limits


def _open_input(path: str) -> Iterable[str]:
    if path == "-":
        for line in sys.stdin:
            yield line
        return
    candidate = Path(path).expanduser()
    if not candidate.exists():
        raise VerificationError(f"Plik {path} nie istnieje")
    if candidate.suffix == ".gz":
        try:
            with gzip.open(candidate, "rt", encoding="utf-8") as handle:
                for line in handle:
                    yield line
        except OSError as exc:
            raise VerificationError(f"Nie udało się rozpakować {path}: {exc}") from exc
        return
    try:
        with candidate.open("r", encoding="utf-8") as handle:
            for line in handle:
                yield line
    except OSError as exc:
        raise VerificationError(f"Nie udało się odczytać {path}: {exc}") from exc


def _canonical_payload(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _verify_entry(entry: Mapping[str, Any], *, signing_key: _SigningKey | None, allow_unsigned: bool) -> None:
    signature = entry.get("signature")
    if not signature:
        if allow_unsigned:
            return
        raise VerificationError("Wpis decision logu nie ma podpisu")

    if not isinstance(signature, Mapping):
        raise VerificationError("Pole signature ma nieprawidłową strukturę")

    algorithm = signature.get("algorithm")
    if algorithm != "HMAC-SHA256":
        raise VerificationError(f"Nieobsługiwany algorytm podpisu: {algorithm}")

    value = signature.get("value")
    if not isinstance(value, str):
        raise VerificationError("Brak wartości podpisu")
    try:
        expected_digest = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise VerificationError("Nieprawidłowa wartość podpisu base64") from exc

    key_id = signature.get("key_id")
    if signing_key is None:
        raise VerificationError("Oczekiwano klucza do weryfikacji podpisu, ale nie został dostarczony")
    if key_id is not None and signing_key.key_id and key_id != signing_key.key_id:
        raise VerificationError(
            f"Podpis używa identyfikatora klucza {key_id}, ale oczekiwano {signing_key.key_id}"
        )

    payload = dict(entry)
    payload.pop("signature", None)
    digest = hmac.new(signing_key.value, _canonical_payload(payload), digestmod="sha256").digest()
    if not hmac.compare_digest(digest, expected_digest):
        raise VerificationError("Niepoprawny podpis HMAC")


def _load_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = dict(metadata)
    cleaned.setdefault("verified_at", datetime.now(timezone.utc).isoformat())
    return cleaned


def verify_log(
    path: str,
    *,
    signing_key: _SigningKey | None,
    allow_unsigned: bool,
    require_screen_info: bool = False,
    collect_summary: bool = False,
) -> dict[str, Any]:
    metadata: dict[str, Any] | None = None
    total = 0
    verified = 0
    snapshots: list[Mapping[str, Any]] = []
    aggregator = _SummaryAggregator() if collect_summary else None

    for raw_line in _open_input(path):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        total += 1
        try:
            entry = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise VerificationError(f"Nieprawidłowy JSONL (linia {total}): {exc}") from exc
        if not isinstance(entry, Mapping):
            raise VerificationError(f"Oczekiwano obiektu JSON (linia {total})")

        kind = entry.get("kind")
        if kind == "metadata":
            metadata = _load_metadata(entry.get("metadata", {}))
        _verify_entry(entry, signing_key=signing_key, allow_unsigned=allow_unsigned)
        if kind != "metadata":
            if require_screen_info:
                screen = entry.get("screen")
                if not isinstance(screen, Mapping):
                    raise VerificationError("Wpis decision logu nie zawiera sekcji screen z metadanymi monitora")
                if not screen:
                    raise VerificationError("Metadane monitora (screen) są puste")
                important_keys = {
                    "index",
                    "name",
                    "manufacturer",
                    "model",
                    "refresh_hz",
                    "device_pixel_ratio",
                    "resolution",
                }
                if not any(key in screen for key in important_keys):
                    raise VerificationError(
                        "Metadane monitora nie zawierają żadnego z kluczowych pól (index/name/manufacturer/model/refresh_hz/resolution)"
                    )
            snapshots.append(entry)
            if aggregator is not None:
                aggregator.add_snapshot(entry)
        verified += 1

    if total == 0:
        raise VerificationError("Decision log nie zawiera wpisów")

    result: dict[str, Any] = {
        "path": path,
        "entries": total,
        "verified_entries": verified,
        "metadata": metadata,
        "snapshots": snapshots,
    }
    if aggregator is not None:
        result["summary"] = aggregator.as_dict()
    return result


def _validate_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    expect_mode: str | None,
    expect_summary: bool,
    expected_filters: Mapping[str, Any],
    require_auth_token: bool,
    required_auth_scopes: Sequence[str],
    require_tls: bool,
    expect_input_file: str | None,
    expect_endpoint: str | None,
    required_tls_materials: Sequence[str],
    expected_server_sha256: Sequence[str],
    expected_server_sha256_sources: Sequence[str],
) -> None:
    expectations_defined = any(
        [
            expect_mode,
            expect_summary,
            expected_filters,
            require_auth_token,
            required_auth_scopes,
            require_tls,
            expect_input_file,
            expect_endpoint,
            required_tls_materials,
            expected_server_sha256,
            expected_server_sha256_sources,
        ]
    )
    if metadata is None:
        if expectations_defined:
            raise VerificationError("Decision log nie zawiera metadanych, a oczekiwano ich obecności")
        return

    mode = metadata.get("mode")
    if expect_mode and mode != expect_mode:
        raise VerificationError(f"Oczekiwano trybu metadanych {expect_mode!r}, otrzymano {mode!r}")

    if expect_summary and not metadata.get("summary_enabled", False):
        raise VerificationError("Metadane nie mają włączonego podsumowania (summary_enabled)")

    summary_signature = metadata.get("summary_signature")
    if summary_signature is not None and not isinstance(summary_signature, Mapping):
        raise VerificationError("Metadane summary_signature powinny być obiektem JSON")

    if expected_filters:
        filters = metadata.get("filters")
        if not isinstance(filters, Mapping):
            raise VerificationError("Metadane nie zawierają sekcji filters")
        for key, expected_value in expected_filters.items():
            if key not in filters:
                raise VerificationError(f"Metadane nie zawierają filtra {key!r}")
            actual_value = filters[key]
            if isinstance(actual_value, list) and isinstance(expected_value, list):
                if actual_value != expected_value:
                    raise VerificationError(
                        f"Filtr {key!r} ma wartość {actual_value!r}, oczekiwano {expected_value!r}"
                    )
            elif actual_value != expected_value:
                raise VerificationError(
                    f"Filtr {key!r} ma wartość {actual_value!r}, oczekiwano {expected_value!r}"
                )

    if expect_input_file is not None:
        if mode != "jsonl":
            raise VerificationError("Oczekiwano metadanych trybu jsonl dla weryfikacji input_file")
        input_file = metadata.get("input_file")
        if input_file != expect_input_file:
            raise VerificationError(
                f"Metadane input_file mają wartość {input_file!r}, oczekiwano {expect_input_file!r}"
            )

    if expect_endpoint is not None:
        if mode != "grpc":
            raise VerificationError("Oczekiwano metadanych trybu grpc dla weryfikacji endpointu")
        endpoint = metadata.get("endpoint")
        if endpoint != expect_endpoint:
            raise VerificationError(
                f"Metadane endpoint mają wartość {endpoint!r}, oczekiwano {expect_endpoint!r}"
            )

    if require_auth_token:
        if mode != "grpc":
            raise VerificationError("Wymagano auth_token_provided dla logu w trybie grpc")
        if not metadata.get("auth_token_provided", False):
            raise VerificationError("Metadane wskazują brak tokenu autoryzacyjnego")

    required_auth_scopes = tuple(scope for scope in required_auth_scopes if scope)
    if required_auth_scopes:
        scope_checked = metadata.get("auth_token_scope_checked")
        if scope_checked is not True:
            raise VerificationError(
                "Metadane nie potwierdzają weryfikacji scope'ów tokenu autoryzacyjnego"
            )
        scope_match = metadata.get("auth_token_scope_match")
        if scope_match is False:
            raise VerificationError(
                "Token autoryzacyjny nie spełnia wymaganych scope'ów"
            )
        scopes_recorded_raw = metadata.get("auth_token_scopes")
        recorded_scopes: set[str] | None
        if scopes_recorded_raw is None:
            recorded_scopes = None
        elif isinstance(scopes_recorded_raw, (list, tuple, set)):
            recorded_scopes = {
                str(entry).strip().lower()
                for entry in scopes_recorded_raw
                if str(entry).strip()
            }
        else:
            raise VerificationError(
                "Metadane auth_token_scopes powinny być listą lub krotką scope'ów"
            )
        if recorded_scopes:
            missing_scopes = [
                scope for scope in required_auth_scopes if scope not in recorded_scopes
            ]
            if missing_scopes:
                raise VerificationError(
                    "Token autoryzacyjny nie posiada wymaganych scope'ów: "
                    + ", ".join(sorted(missing_scopes))
                )

    if require_tls:
        if mode != "grpc":
            raise VerificationError("Wymagano TLS dla logu w trybie grpc")
        if not metadata.get("use_tls", False):
            raise VerificationError("Metadane wskazują, że połączenie gRPC nie używa TLS")

    tls_requirements = bool(required_tls_materials or expected_server_sha256 or expected_server_sha256_sources)
    tls_materials = metadata.get("tls_materials") if isinstance(metadata, Mapping) else None
    if tls_requirements:
        if not isinstance(tls_materials, Mapping):
            raise VerificationError("Metadane nie zawierają sekcji tls_materials")
    if not isinstance(tls_materials, Mapping):
        tls_materials = {}

    missing_materials: list[str] = []
    for material in required_tls_materials:
        present = tls_materials.get(material)
        if not present:
            missing_materials.append(material)
    if missing_materials:
        raise VerificationError(
            "Brak wymaganych materiałów TLS w metadanych: " + ", ".join(sorted(missing_materials))
        )

    # spójność deklaracji fingerprintu z flagą w tls_materials
    server_material_recorded = "server_sha256" in tls_materials
    server_material_flag = bool(tls_materials.get("server_sha256"))
    server_fingerprint_raw = metadata.get("server_sha256") if isinstance(metadata, Mapping) else None
    normalized_present = _normalize_server_sha256(server_fingerprint_raw)

    if server_material_recorded:
        if server_material_flag and normalized_present is None:
            raise VerificationError(
                "Metadane TLS deklarują obecność server_sha256, lecz fingerprint nie został zapisany"
            )
        if normalized_present is not None and not server_material_flag:
            raise VerificationError(
                "Metadane TLS zawierają fingerprint server_sha256, jednak tls_materials wskazuje jego brak"
            )

    expected_sha = [item for item in expected_server_sha256 if item]
    normalized_expected: dict[str, str] = {}
    for raw in expected_sha:
        normalized = _normalize_server_sha256(raw)
        if normalized is None:
            raise VerificationError(f"Nieprawidłowy oczekiwany fingerprint SHA-256: {raw!r}")
        normalized_expected[normalized] = raw

    expected_sources = tuple({source for source in expected_server_sha256_sources if source})

    if normalized_expected or "server_sha256" in required_tls_materials:
        if normalized_present is None:
            raise VerificationError("Metadane TLS nie zawierają poprawnego odcisku server_sha256")
        if normalized_expected and normalized_present not in normalized_expected:
            expected_labels = ", ".join(sorted(normalized_expected.values()))
            raise VerificationError(
                "Fingerprint server_sha256 w metadanych nie zgadza się z oczekiwanymi wartościami"
                + (f" ({expected_labels})" if expected_labels else "")
            )
        if expected_sources:
            fingerprint_source_raw = metadata.get("server_sha256_source") if isinstance(metadata, Mapping) else None
            fingerprint_source = (
                str(fingerprint_source_raw).strip().lower() if fingerprint_source_raw is not None else None
            )
            if fingerprint_source not in expected_sources:
                allowed = ", ".join(sorted(expected_sources)) or ""
                raise VerificationError(
                    "Źródło fingerprintu TLS w metadanych nie spełnia oczekiwań"
                    + (f" (dozwolone: {allowed})" if allowed else "")
                )
    elif expected_sources:
        fingerprint_source_raw = metadata.get("server_sha256_source") if isinstance(metadata, Mapping) else None
        fingerprint_source = (
            str(fingerprint_source_raw).strip().lower() if fingerprint_source_raw is not None else None
        )
        if fingerprint_source not in expected_sources:
            allowed = ", ".join(sorted(expected_sources)) or ""
            raise VerificationError(
                "Metadane TLS nie zawierają fingerprintu, ale określono oczekiwane źródła"
                + (f" (dozwolone: {allowed})" if allowed else "")
            )


def _matches_screen_name(screen: Mapping[str, Any] | None, *, expected_substring: str) -> bool:
    if not expected_substring:
        return True
    if not isinstance(screen, Mapping):
        return False
    candidate = screen.get("name")
    if not isinstance(candidate, str):
        return False
    return expected_substring.casefold() in candidate.casefold()


def _extract_risk_service_metadata(
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not isinstance(metadata, Mapping):
        return None

    candidates: list[Mapping[str, Any]] = []
    core_cfg = metadata.get("core_config")
    if isinstance(core_cfg, Mapping):
        risk_core = core_cfg.get("risk_service")
        if isinstance(risk_core, Mapping):
            candidates.append(risk_core)
    risk_section = metadata.get("risk_service")
    if isinstance(risk_section, Mapping):
        candidates.append(risk_section)

    if not candidates:
        return None

    merged: dict[str, Any] = {}
    for candidate in candidates:
        for key, value in candidate.items():
            if value is not None:
                merged[key] = value
    return merged


def _validate_risk_service_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    require_tls: bool,
    required_materials: Sequence[str],
    expected_fingerprints: Sequence[str],
    required_scopes: Sequence[str],
    required_token_ids: Sequence[str],
    require_auth_token: bool,
) -> None:
    requirements_defined = any(
        [
            require_tls,
            required_materials,
            expected_fingerprints,
            required_scopes,
            required_token_ids,
            require_auth_token,
        ]
    )
    if not requirements_defined:
        return

    if metadata is None:
        raise VerificationError("Metadane nie zawierają sekcji risk_service")

    if require_tls and not metadata.get("tls_enabled"):
        raise VerificationError("Sekcja risk_service nie wskazuje aktywnego TLS")

    material_field_map = {
        "root_cert": "root_cert_configured",
        "client_cert": "client_cert_configured",
        "client_key": "client_key_configured",
        "client_auth": "client_auth",
    }
    missing_risk_materials: list[str] = []
    for material in required_materials:
        field_name = material_field_map.get(material)
        if not field_name:
            raise VerificationError(
                f"Materiał TLS {material} nie jest obsługiwany dla risk_service"
            )
        if not metadata.get(field_name):
            missing_risk_materials.append(material)
    if missing_risk_materials:
        raise VerificationError(
            "Sekcja risk_service nie zawiera wymaganych materiałów TLS: "
            + ", ".join(sorted(missing_risk_materials))
        )

    normalized_expected_fps = [fp for fp in expected_fingerprints if fp]
    if normalized_expected_fps:
        recorded = metadata.get("pinned_fingerprints")
        recorded_set: set[str] = set()
        if isinstance(recorded, (list, tuple, set)):
            for entry in recorded:
                normalized = _normalize_server_sha256(entry)
                if normalized:
                    recorded_set.add(normalized)
        missing_fp = [
            fingerprint
            for fingerprint in normalized_expected_fps
            if fingerprint not in recorded_set
        ]
        if missing_fp:
            raise VerificationError(
                "Sekcja risk_service nie zawiera wymaganego pinningu TLS: "
                + ", ".join(sorted(missing_fp))
            )

    if required_scopes:
        recorded_scopes: set[str] = set()
        primary_scope = metadata.get("auth_token_scope_required")
        if isinstance(primary_scope, str):
            candidate = primary_scope.strip().lower()
            if candidate:
                recorded_scopes.add(candidate)
        scopes_field = metadata.get("auth_token_scopes")
        if isinstance(scopes_field, (list, tuple, set)):
            for entry in scopes_field:
                if isinstance(entry, str):
                    candidate = entry.strip().lower()
                    if candidate:
                        recorded_scopes.add(candidate)
        required_map = metadata.get("required_scopes")
        if isinstance(required_map, Mapping):
            for scope_name in required_map.keys():
                if isinstance(scope_name, str):
                    candidate = scope_name.strip().lower()
                    if candidate:
                        recorded_scopes.add(candidate)
        missing_scopes = [
            scope for scope in required_scopes if scope not in recorded_scopes
        ]
        if missing_scopes:
            raise VerificationError(
                "Sekcja risk_service nie deklaruje wymaganych scope'ów: "
                + ", ".join(sorted(missing_scopes))
            )

    # token_id (wielokrotne)
    if required_token_ids:
        recorded_tokens: set[str] = set()
        token_id = metadata.get("auth_token_token_id")
        if isinstance(token_id, str):
            candidate = token_id.strip()
            if candidate:
                recorded_tokens.add(candidate)
        token_list = metadata.get("auth_token_tokens")
        if isinstance(token_list, (list, tuple, set)):
            for entry in token_list:
                if isinstance(entry, str):
                    candidate = entry.strip()
                    if candidate:
                        recorded_tokens.add(candidate)
        missing_tokens = [
            token for token in required_token_ids if token not in recorded_tokens
        ]
        if missing_tokens:
            raise VerificationError(
                "Sekcja risk_service nie deklaruje wymaganych token_id: "
                + ", ".join(sorted(missing_tokens))
            )

    if require_auth_token:
        if metadata.get("auth_token_scope_checked") is not True:
            raise VerificationError(
                "Sekcja risk_service nie potwierdza weryfikacji tokenu RBAC"
            )
        if metadata.get("auth_token_scope_match") is False:
            raise VerificationError(
                "Token RBAC dla risk_service nie spełnia wymaganych scope'ów"
            )


def _ensure_filter_matches_snapshots(
    *,
    metadata: Mapping[str, Any],
    snapshots: Sequence[Mapping[str, Any]],
) -> None:
    filters = metadata.get("filters")
    if not isinstance(filters, Mapping):
        return

    severity_filters: set[str] | None = None
    if "severity" in filters:
        raw = filters["severity"]
        if raw is None:
            severity_filters = None
        elif isinstance(raw, list):
            normalized = {_normalize_severity(item) for item in raw}
            if None in normalized:
                raise VerificationError("Filtr severity w metadanych zawiera nieprawidłowe wartości")
            severity_filters = {item for item in normalized if item is not None}
        else:
            raise VerificationError("Filtr severity w metadanych musi być listą wartości tekstowych")

    severity_min_raw = filters.get("severity_min")
    severity_min = None
    if severity_min_raw is not None:
        severity_min = _normalize_severity(severity_min_raw)
        if severity_min is None:
            raise VerificationError("Filtr severity_min w metadanych ma nieprawidłową wartość")

    screen_index = filters.get("screen_index")
    if screen_index is not None and not isinstance(screen_index, int):
        raise VerificationError("Filtr screen_index w metadanych powinien być liczbą całkowitą")

    screen_name = filters.get("screen_name")
    if screen_name is not None and not isinstance(screen_name, str):
        raise VerificationError("Filtr screen_name w metadanych powinien być tekstem")

    event_expected = filters.get("event")
    if event_expected is not None and not isinstance(event_expected, str):
        raise VerificationError("Filtr event w metadanych powinien być tekstem")

    since_dt = None
    if "since" in filters:
        since_dt = _parse_iso_datetime(filters["since"])
        if since_dt is None:
            raise VerificationError("Filtr since w metadanych ma nieprawidłowy format ISO 8601")

    until_dt = None
    if "until" in filters:
        until_dt = _parse_iso_datetime(filters["until"])
        if until_dt is None:
            raise VerificationError("Filtr until w metadanych ma nieprawidłowy format ISO 8601")

    limit = filters.get("limit")
    if limit is not None:
        if isinstance(limit, bool):
            raise VerificationError("Filtr limit w metadanych nie może być wartością bool")
        try:
            limit_int = int(limit)
        except (TypeError, ValueError):
            raise VerificationError("Filtr limit w metadanych powinien być liczbą całkowitą") from None
        if limit_int < 0:
            raise VerificationError("Filtr limit w metadanych nie może być ujemny")
        if len(snapshots) > limit_int:
            raise VerificationError(
                f"Decision log zawiera {len(snapshots)} wpisów, co przekracza limit {limit_int} z metadanych"
            )

    for index, entry in enumerate(snapshots, start=1):
        severity_value = _normalize_severity(entry.get("severity"))
        if severity_filters is not None:
            if severity_value is None or severity_value not in severity_filters:
                raise VerificationError(
                    f"Wpis #{index} posiada severity {severity_value!r}, które nie przechodzi filtra severity"
                )
        if severity_min is not None:
            if severity_value is None or not _severity_at_least(severity_value, severity_min):
                raise VerificationError(
                    f"Wpis #{index} ma severity {severity_value!r} poniżej progu {severity_min!r}"
                )

        if event_expected is not None:
            if entry.get("event") != event_expected:
                raise VerificationError(
                    f"Wpis #{index} posiada event {entry.get('event')!r}, oczekiwano {event_expected!r}"
                )

        screen_info = entry.get("screen") if isinstance(entry.get("screen"), Mapping) else None
        if screen_index is not None:
            if not isinstance(screen_info, Mapping) or screen_info.get("index") != screen_index:
                raise VerificationError(
                    f"Wpis #{index} nie spełnia filtra screen_index={screen_index}; metadane ekranu: {screen_info!r}"
                )
        if screen_name is not None:
            if not _matches_screen_name(screen_info, expected_substring=screen_name):
                raise VerificationError(
                    f"Wpis #{index} nie spełnia filtra screen_name zawierającego {screen_name!r}"
                )

        if since_dt is not None or until_dt is not None:
            timestamp_value = entry.get("timestamp")
            ts_dt = _parse_iso_datetime(timestamp_value)
            if ts_dt is None:
                raise VerificationError(
                    f"Wpis #{index} nie zawiera poprawnego znacznika czasu zgodnego z filtrami since/until"
                )
            if since_dt is not None and ts_dt < since_dt:
                raise VerificationError(
                    f"Wpis #{index} ({timestamp_value}) leży przed granicą since {filters['since']!r}"
                )
            if until_dt is not None and ts_dt > until_dt:
                raise VerificationError(
                    f"Wpis #{index} ({timestamp_value}) leży po granicy until {filters['until']!r}"
                )


def _read_summary_payload(path: str) -> Mapping[str, Any]:
    if path == "-":
        payload = sys.stdin.read()
    else:
        summary_path = Path(path).expanduser()
        try:
            if summary_path.suffix == ".gz":
                with gzip.open(summary_path, "rt", encoding="utf-8") as handle:
                    payload = handle.read()
            else:
                payload = summary_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise VerificationError(f"Nie udało się odczytać podsumowania {path}: {exc}") from exc

    try:
        summary = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise VerificationError(f"Podsumowanie {path} zawiera nieprawidłowy JSON: {exc}") from exc

    if not isinstance(summary, Mapping):
        raise VerificationError("Podsumowanie powinno być obiektem JSON zawierającym pole 'summary'")

    body = summary.get("summary")
    if not isinstance(body, Mapping):
        raise VerificationError("Plik podsumowania musi zawierać obiekt 'summary'")

    return summary


def _floats_close(a: float, b: float, *, tol: float = 1e-6) -> bool:
    diff = abs(a - b)
    return diff <= tol or diff <= tol * max(abs(a), abs(b), 1.0)


def _normalise_screen_set(candidate: Any, *, context: str) -> list[str]:
    if candidate is None:
        return []
    if not isinstance(candidate, list):
        raise VerificationError(f"{context}: sekcja screens powinna być listą")
    encoded: list[str] = []
    for item in candidate:
        if not isinstance(item, Mapping):
            raise VerificationError(f"{context}: każdy wpis screens powinien być obiektem JSON")
        encoded.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
    return sorted(encoded)


def _extract_severity_counts(candidate: Any, *, context: str) -> dict[str, int]:
    if candidate is None:
        return {}
    if not isinstance(candidate, Mapping):
        raise VerificationError(f"{context}: sekcja severity powinna być obiektem zawierającym counts")
    counts = candidate.get("counts")
    if not isinstance(counts, Mapping):
        raise VerificationError(f"{context}: sekcja severity powinna zawierać pole counts")
    normalised: dict[str, int] = {}
    for level, value in counts.items():
        level_normalised = _normalize_severity(level)
        if level_normalised is None:
            raise VerificationError(f"{context}: severity zawiera nieprawidłowy poziom {level!r}")
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            raise VerificationError(f"{context}: liczność severity {level!r} powinna być liczbą całkowitą") from None
        normalised[level_normalised] = numeric
    return normalised


def _compare_summary(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    if not isinstance(actual, Mapping):
        raise VerificationError("Brak obliczonego podsumowania do weryfikacji")

    expected_total = expected.get("total_snapshots")
    actual_total = actual.get("total_snapshots")
    if expected_total != actual_total:
        raise VerificationError(
            f"Podsumowanie oczekuje {expected_total!r} wpisów, ale log zawiera {actual_total!r}"
        )

    expected_events = expected.get("events")
    actual_events = actual.get("events")
    if not isinstance(expected_events, Mapping) or not isinstance(actual_events, Mapping):
        raise VerificationError("Podsumowanie powinno zawierać sekcję events jako obiekt JSON")

    expected_event_keys = set(expected_events.keys())
    actual_event_keys = set(actual_events.keys())
    if expected_event_keys != actual_event_keys:
        missing = expected_event_keys - actual_event_keys
        extra = actual_event_keys - expected_event_keys
        raise VerificationError(
            "Zestaw zdarzeń w podsumowaniu nie zgadza się z decision logiem: "
            f"brakujące={sorted(missing)}, nadmiarowe={sorted(extra)}"
        )

    for event_name in sorted(expected_event_keys):
        context = f"Podsumowanie dla zdarzenia {event_name!r}"
        expected_event = expected_events[event_name]
        actual_event = actual_events[event_name]
        if not isinstance(expected_event, Mapping) or not isinstance(actual_event, Mapping):
            raise VerificationError(f"{context}: oczekiwano obiektów JSON")

        if expected_event.get("count") != actual_event.get("count"):
            raise VerificationError(
                f"{context}: liczba wpisów wynosi {actual_event.get('count')!r}, oczekiwano {expected_event.get('count')!r}"
            )

        expected_fps = expected_event.get("fps")
        actual_fps = actual_event.get("fps")
        if expected_fps is None:
            if actual_fps is not None:
                raise VerificationError(f"{context}: log posiada statystyki FPS, ale podsumowanie ich nie deklaruje")
        else:
            if not isinstance(expected_fps, Mapping) or not isinstance(actual_fps, Mapping):
                raise VerificationError(f"{context}: sekcja fps powinna być obiektem JSON")
            for key in ("min", "max", "avg"):
                exp_value = expected_fps.get(key)
                act_value = actual_fps.get(key)
                if exp_value is None or act_value is None:
                    raise VerificationError(f"{context}: sekcja fps nie zawiera pola {key}")
                if not _floats_close(float(exp_value), float(act_value)):
                    raise VerificationError(
                        f"{context}: wartość fps.{key} wynosi {act_value!r}, oczekiwano {exp_value!r}"
                    )
            if expected_fps.get("samples") != actual_fps.get("samples"):
                raise VerificationError(
                    f"{context}: liczba próbek FPS wynosi {actual_fps.get('samples')!r}, oczekiwano {expected_fps.get('samples')!r}"
                )

        expected_screens = _normalise_screen_set(expected_event.get("screens"), context=context)
        actual_screens = _normalise_screen_set(actual_event.get("screens"), context=context)
        if expected_screens != actual_screens:
            raise VerificationError(f"{context}: zestaw ekranów różni się od oczekiwanego")

        expected_severity = _extract_severity_counts(expected_event.get("severity"), context=context)
        actual_severity = _extract_severity_counts(actual_event.get("severity"), context=context)
        if expected_severity != actual_severity:
            raise VerificationError(
                f"{context}: zestaw liczności severity nie pokrywa się z oczekiwanym"
            )

        for ts_field in ("first_timestamp", "last_timestamp"):
            if ts_field in expected_event:
                if expected_event[ts_field] != actual_event.get(ts_field):
                    raise VerificationError(
                        f"{context}: pole {ts_field} ma wartość {actual_event.get(ts_field)!r}, oczekiwano {expected_event[ts_field]!r}"
                    )
            elif ts_field in actual_event:
                raise VerificationError(f"{context}: log zawiera pole {ts_field}, którego brak w podsumowaniu")

    for ts_field in ("first_timestamp", "last_timestamp"):
        if ts_field in expected:
            if expected[ts_field] != actual.get(ts_field):
                raise VerificationError(
                    f"Podsumowanie {ts_field} ma wartość {actual.get(ts_field)!r}, oczekiwano {expected[ts_field]!r}"
                )
        elif ts_field in actual:
            raise VerificationError(
                f"Decision log raportuje pole {ts_field}, które nie zostało zadeklarowane w podsumowaniu"
            )

    expected_severity_total = expected.get("severity_counts")
    actual_severity_total = actual.get("severity_counts")
    if expected_severity_total is None:
        if actual_severity_total:
            raise VerificationError("Podsumowanie nie deklaruje severity_counts, ale log zawiera globalne liczniki")
    else:
        if not isinstance(expected_severity_total, Mapping) or not isinstance(actual_severity_total, Mapping):
            raise VerificationError("severity_counts powinno być obiektem JSON")
        expected_counts = _extract_severity_counts({"counts": expected_severity_total}, context="severity_counts")
        actual_counts = _extract_severity_counts({"counts": actual_severity_total}, context="severity_counts")
        if expected_counts != actual_counts:
            raise VerificationError("Globalne liczniki severity nie zgadzają się z decision logiem")


def _validate_summary_path(
    path: str,
    computed_summary: Mapping[str, Any] | None,
    *,
    signing_key: _SigningKey | None,
    metadata_signature: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if computed_summary is None:
        raise VerificationError("Brak danych do weryfikacji podsumowania decision logu")

    if metadata_signature is not None and not isinstance(metadata_signature, Mapping):
        raise VerificationError("Metadane summary_signature mają nieprawidłową strukturę")

    payload = _read_summary_payload(path)
    summary_body = payload.get("summary")
    signature_payload = payload.get("signature")

    if signature_payload is None and metadata_signature:
        raise VerificationError("Metadane wymagają podpisanego podsumowania, ale plik nie zawiera podpisu")

    signature_present = signature_payload is not None
    if signature_payload is not None:
        if signing_key is None:
            raise VerificationError("Podsumowanie zawiera podpis, lecz nie dostarczono klucza HMAC do weryfikacji")
        entry = dict(payload)
        _verify_entry(entry, signing_key=signing_key, allow_unsigned=False)
        if metadata_signature:
            expected_algorithm = metadata_signature.get("algorithm")
            actual_algorithm = signature_payload.get("algorithm")
            if expected_algorithm and actual_algorithm != expected_algorithm:
                raise VerificationError(
                    "Podsumowanie podpisane algorytmem innym niż zadeklarowany w metadanych"
                )
            expected_key_id = metadata_signature.get("key_id")
            actual_key_id = signature_payload.get("key_id")
            if expected_key_id and actual_key_id != expected_key_id:
                raise VerificationError(
                    "Podsumowanie podpisane innym identyfikatorem klucza niż wskazano w metadanych"
                )
    elif signing_key and metadata_signature is None:
        LOGGER.warning(
            "Podsumowanie nie zawiera podpisu mimo dostarczenia klucza HMAC – upewnij się, że watcher generuje podpisy."
        )

    _compare_summary(summary_body, computed_summary)

    result: dict[str, Any] = {
        "path": path,
        "signed": signature_present,
    }
    if signature_payload:
        result["signature"] = signature_payload
    if metadata_signature:
        result["metadata_signature"] = metadata_signature
    metadata_section = payload.get("metadata")
    if isinstance(metadata_section, Mapping):
        rp_summary = metadata_section.get("risk_profile_summary")
        if isinstance(rp_summary, Mapping):
            result["risk_profile_summary"] = rp_summary
    return result


def _enforce_event_limits(*, summary: Mapping[str, Any] | None, limits: Mapping[str, int]) -> None:
    if not limits:
        return
    if not isinstance(summary, Mapping):
        raise VerificationError("Brak danych podsumowania decision logu do weryfikacji limitów zdarzeń")
    events_section = summary.get("events")
    if not isinstance(events_section, Mapping):
        raise VerificationError("Podsumowanie decision logu nie zawiera sekcji events")
    for event_name, limit in limits.items():
        stats = events_section.get(event_name)
        count = 0
        if isinstance(stats, Mapping):
            raw_count = stats.get("count")
            if isinstance(raw_count, int):
                count = raw_count
            else:
                # jeżeli brak poprawnego pola count traktujemy jak zero, ale raportujemy ostrzeżenie
                LOGGER.warning(
                    "Sekcja podsumowania dla zdarzenia %s nie zawiera prawidłowego pola count; przyjęto wartość 0",
                    event_name,
                )
        if count > limit:
            raise VerificationError(
                f"Zdarzenie {event_name!r} wystąpiło {count} razy, co przekracza dozwolony limit {limit}"
            )


def _enforce_min_event_counts(
    *, summary: Mapping[str, Any] | None, requirements: Mapping[str, int]
) -> None:
    if not requirements:
        return
    if not isinstance(summary, Mapping):
        raise VerificationError(
            "Brak danych podsumowania decision logu do weryfikacji minimalnych liczności zdarzeń"
        )
    events_section = summary.get("events")
    if not isinstance(events_section, Mapping):
        raise VerificationError("Podsumowanie decision logu nie zawiera sekcji events")
    for event_name, minimum in requirements.items():
        stats = events_section.get(event_name)
        count = 0
        if isinstance(stats, Mapping):
            raw_count = stats.get("count")
            if isinstance(raw_count, int):
                count = raw_count
            else:
                LOGGER.warning(
                    "Sekcja podsumowania dla zdarzenia %s nie zawiera prawidłowego pola count; przyjęto wartość 0",
                    event_name,
                )
        if count < minimum:
            raise VerificationError(
                f"Zdarzenie {event_name!r} wystąpiło {count} razy, co nie spełnia wymaganego minimum {minimum}"
            )


def _build_report_payload(
    verification_result: Mapping[str, Any],
    metadata: Mapping[str, Any],
    summary_validation: Mapping[str, Any] | None,
    *,
    enforced_limits: Mapping[str, int] | None = None,
    enforced_minimums: Mapping[str, int] | None = None,
    risk_profile: Mapping[str, Any] | None = None,
    risk_profile_summary: Mapping[str, Any] | None = None,
    core_config: Mapping[str, Any] | None = None,
    risk_profile_snippet_validation: Sequence[Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "report_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "path": verification_result.get("path"),
        "entries": verification_result.get("entries"),
        "verified_entries": verification_result.get("verified_entries"),
        "metadata": metadata,
    }
    summary_data = verification_result.get("summary")
    if summary_data is not None:
        payload["summary"] = summary_data
    if summary_validation is not None:
        payload["summary_validation"] = summary_validation
    if enforced_limits:
        payload["enforced_event_limits"] = dict(enforced_limits)
    if enforced_minimums:
        payload["enforced_event_minimums"] = dict(enforced_minimums)
    if risk_profile:
        payload["risk_profile"] = dict(risk_profile)
    if risk_profile_summary:
        payload["risk_profile_summary"] = dict(risk_profile_summary)
    if core_config:
        payload["core_config"] = dict(core_config)
    if risk_profile_snippet_validation:
        payload["risk_profile_snippet_validation"] = [
            dict(entry) for entry in risk_profile_snippet_validation
        ]
    return payload


def _write_report_output(destination: str, payload: Mapping[str, Any]) -> None:
    if destination.strip() == "-":
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ------------------------------ CLI -----------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Weryfikacja podpisów decision logu UI")
    parser.add_argument(
        "path",
        nargs="?",
        help="Ścieżka do decision logu (JSONL lub JSONL.GZ). Użyj '-' dla STDIN.",
    )
    parser.add_argument("--hmac-key", help="Sekret HMAC wprost w CLI (UTF-8)")
    parser.add_argument("--hmac-key-file", help="Ścieżka do pliku z sekretem HMAC")
    parser.add_argument("--hmac-key-id", help="Identyfikator klucza HMAC")
    parser.add_argument("--allow-unsigned", action="store_true", help="Zezwól na wpisy bez podpisu")
    parser.add_argument("--expected-key-id", help="Oczekiwany identyfikator klucza podpisującego")
    parser.add_argument("--expect-mode", choices=["grpc", "jsonl"], help="Oczekiwany tryb metadanych")
    parser.add_argument(
        "--expect-summary-enabled",
        action="store_true",
        help="Wymagaj, aby metadane wskazywały włączone podsumowanie",
    )
    parser.add_argument(
        "--expect-filter",
        action="append",
        default=[],
        metavar="KLUCZ=WARTOŚĆ",
        help="Oczekiwany filtr w metadanych (można podawać wielokrotnie)",
    )
    parser.add_argument(
        "--require-auth-token",
        action="store_true",
        help="Wymagaj, by log gRPC zawierał token autoryzacyjny",
    )
    parser.add_argument(
        "--require-auth-scope",
        dest="require_auth_scopes",
        action="append",
        default=[],
        metavar="SCOPE",
        help="Wymagaj, by token autoryzacyjny obejmował wskazany scope (można podawać wielokrotnie)",
    )
    parser.add_argument(
        "--require-tls",
        action="store_true",
        help="Wymagaj, by log gRPC był zarejestrowany przy użyciu TLS",
    )
    parser.add_argument(
        "--require-tls-material",
        action="append",
        default=[],
        choices=_TLS_MATERIAL_CHOICES,
        metavar="MATERIAŁ",
        help="Wymagaj obecności materiału TLS w metadanych (można podawać wielokrotnie)",
    )
    # rozszerzenia RiskService
    parser.add_argument(
        "--require-risk-service-tls",
        action="store_true",
        help="Wymagaj, by konfiguracja RiskService wskazywała aktywne TLS",
    )
    parser.add_argument(
        "--require-risk-service-tls-material",
        dest="require_risk_service_tls_material",
        action="append",
        default=[],
        choices=_RISK_TLS_MATERIAL_CHOICES,
        metavar="MATERIAŁ",
        help="Wymagaj określonych materiałów TLS w sekcji risk_service (można powtarzać)",
    )
    parser.add_argument(
        "--expect-risk-service-server-sha256",
        action="append",
        default=[],
        metavar="FINGERPRINT",
        help="Oczekiwany fingerprint SHA-256 z pinningu RiskService (można podawać wielokrotnie)",
    )
    parser.add_argument(
        "--expect-server-sha256",
        action="append",
        default=[],
        metavar="FINGERPRINT",
        help="Oczekiwany odcisk SHA-256 serwera zapisany w metadanych (można podawać wielokrotnie)",
    )
    parser.add_argument(
        "--expect-server-sha256-source",
        action="append",
        default=[],
        metavar="ŹRÓDŁO",
        help="Dozwolone źródła fingerprintu TLS (np. cli, env, pinned_fingerprint)",
    )
    parser.add_argument(
        "--require-risk-service-scope",
        dest="require_risk_service_scope",
        action="append",
        default=[],
        metavar="SCOPE",
        help="Wymagaj, aby sekcja risk_service deklarowała wskazany scope (można powtarzać)",
    )
    parser.add_argument(
        "--require-risk-service-token-id",
        dest="require_risk_service_token_id",
        action="append",
        default=[],
        metavar="TOKEN_ID",
        help="Wymagaj, aby sekcja risk_service deklarowała wskazany token_id (można powtarzać)",
    )
    parser.add_argument(
        "--require-risk-service-auth-token",
        action="store_true",
        help="Wymagaj potwierdzenia tokenu RBAC w sekcji risk_service",
    )
    parser.add_argument(
        "--require-screen-info",
        action="store_true",
        help="Wymagaj, aby każdy wpis snapshot zawierał metadane monitora (sekcja screen)",
    )
    parser.add_argument(
        "--expect-input-file",
        help="W trybie JSONL wymuś oczekiwaną ścieżkę pliku wejściowego",
    )
    parser.add_argument(
        "--expect-endpoint",
        help="W trybie gRPC wymuś oczekiwany endpoint",
    )
    parser.add_argument(
        "--summary-json",
        help="Ścieżka do pliku JSON z podsumowaniem (obsługa .gz, '-' dla STDIN)",
    )
    parser.add_argument(
        "--report-output",
        help="Zapisz wynik walidacji i podsumowanie decision logu do pliku JSON",
    )
    parser.add_argument(
        "--max-event-count",
        action="append",
        default=[],
        metavar="ZDARZENIE=LIMIT",
        help="Maksymalna dozwolona liczba wystąpień zdarzenia (można podawać wielokrotnie)",
    )
    parser.add_argument(
        "--min-event-count",
        action="append",
        default=[],
        metavar="ZDARZENIE=MINIMUM",
        help="Minimalna wymagana liczba wystąpień zdarzenia (można podawać wielokrotnie)",
    )
    parser.add_argument(
        "--risk-profile",
        help="Zastosuj predefiniowany profil ryzyka (ustawia limity i wymagania audytowe)",
    )
    parser.add_argument(
        "--risk-profiles-file",
        help=(
            "Ścieżka do pliku JSON/YAML z profilami ryzyka telemetrii – pozwala rozszerzyć"
            " lub nadpisać wbudowane presety."
        ),
    )
    parser.add_argument(
        "--print-risk-profiles",
        action="store_true",
        help="Wypisz dostępne profile ryzyka (wraz z metadanymi) i zakończ",
    )
    parser.add_argument(
        "--risk-profile-env-snippet",
        help=(
            "Ścieżka do pliku .env wygenerowanego przez telemetry_risk_profiles render --format env,"
            " który ma zostać porównany z recommended_overrides"
        ),
    )
    parser.add_argument(
        "--risk-profile-yaml-snippet",
        help=(
            "Ścieżka do pliku YAML/JSON wygenerowanego przez telemetry_risk_profiles render --format yaml"
            " (z sekcją metrics_service_overrides) w celu porównania z recommended_overrides"
        ),
    )
    parser.add_argument(
        "--core-config",
        help=(
            "Ścieżka do pliku core.yaml – wczyta domyślne ustawienia MetricsService (profil "
            "ryzyka i plik presetów)."
        ),
    )
    return parser


def _print_available_risk_profiles(
    selected: str | None, *, core_metadata: Mapping[str, Any] | None = None
) -> None:
    payload: dict[str, Any] = {"risk_profiles": {}}
    for name in list_risk_profile_names():
        payload["risk_profiles"][name] = risk_profile_metadata(name)

    if selected:
        normalized = selected.strip().lower()
        payload["selected"] = normalized
        payload["selected_profile"] = payload["risk_profiles"].get(normalized)

    if core_metadata:
        payload["core_config"] = dict(core_metadata)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _load_risk_profile_presets(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    path_value = getattr(args, "risk_profiles_file", None)
    if not path_value:
        return

    target = Path(path_value).expanduser()
    try:
        registered, _meta = load_risk_profiles_with_metadata(target, origin_label=f"verify:{target}")
    except FileNotFoundError as exc:
        parser.error(str(exc))
    except Exception as exc:  # pragma: no cover - zależne od formatu
        parser.error(f"Nie udało się wczytać profili ryzyka z {target}: {exc}")
    else:
        if registered:
            LOGGER.info(
                "Załadowano %s profil(e) ryzyka telemetrii z %s", len(registered), target
            )


def _apply_env_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.path is None:
        env_path = os.getenv(f"{_ENV_PREFIX}PATH")
        if env_path:
            args.path = env_path

    if getattr(args, "risk_profiles_file", None) is None:
        env_profiles = os.getenv(_ENV_RISK_PROFILES_FILE)
        if env_profiles:
            args.risk_profiles_file = env_profiles

    if not getattr(args, "print_risk_profiles", False):
        env_print = os.getenv(_ENV_PRINT_RISK_PROFILES)
        if env_print:
            args.print_risk_profiles = _parse_env_bool(
                env_print, variable=_ENV_PRINT_RISK_PROFILES, parser=parser
            )

    if getattr(args, "core_config", None) is None:
        env_core = os.getenv(_ENV_CORE_CONFIG)
        if env_core:
            args.core_config = env_core

    env_key = os.getenv(f"{_ENV_PREFIX}HMAC_KEY")
    env_key_file = os.getenv(f"{_ENV_PREFIX}HMAC_KEY_FILE")
    env_key_id = os.getenv(f"{_ENV_PREFIX}HMAC_KEY_ID")

    signing_key, allow_unsigned_override = _load_signing_key(
        cli_value=args.hmac_key,
        cli_file=args.hmac_key_file,
        cli_key_id=args.hmac_key_id,
        env_value=env_key,
        env_file=env_key_file,
        env_key_id=env_key_id,
        env_allow_unsigned=os.getenv(f"{_ENV_PREFIX}ALLOW_UNSIGNED"),
        parser=parser,
    )
    if allow_unsigned_override:
        args.allow_unsigned = True
    if signing_key:
        args._signing_key = signing_key
    else:
        args._signing_key = None

    if args.expect_mode is None:
        env_mode = os.getenv(f"{_ENV_PREFIX}EXPECT_MODE")
        if env_mode:
            lowered = env_mode.strip().lower()
            if lowered not in {"grpc", "jsonl"}:
                parser.error(
                    f"{_ENV_PREFIX}EXPECT_MODE ma nieprawidłową wartość {env_mode!r}; dozwolone: grpc/jsonl"
                )
            args.expect_mode = lowered

    if not args.expect_summary_enabled:
        env_summary = os.getenv(f"{_ENV_PREFIX}EXPECT_SUMMARY_ENABLED")
        if env_summary is not None:
            args.expect_summary_enabled = _parse_env_bool(
                env_summary, variable=f"{_ENV_PREFIX}EXPECT_SUMMARY_ENABLED", parser=parser
            )

    if not args.require_auth_token:
        env_auth = os.getenv(f"{_ENV_PREFIX}REQUIRE_AUTH_TOKEN")
        if env_auth is not None:
            args.require_auth_token = _parse_env_bool(
                env_auth, variable=f"{_ENV_PREFIX}REQUIRE_AUTH_TOKEN", parser=parser
            )

    if not args.require_tls:
        env_tls = os.getenv(f"{_ENV_PREFIX}REQUIRE_TLS")
        if env_tls is not None:
            args.require_tls = _parse_env_bool(
                env_tls, variable=f"{_ENV_PREFIX}REQUIRE_TLS", parser=parser
            )

    if args.expect_input_file is None:
        env_input = os.getenv(f"{_ENV_PREFIX}EXPECT_INPUT_FILE")
        if env_input:
            args.expect_input_file = env_input

    if args.expect_endpoint is None:
        env_endpoint = os.getenv(f"{_ENV_PREFIX}EXPECT_ENDPOINT")
        if env_endpoint:
            args.expect_endpoint = env_endpoint

    required_tls_materials = {
        str(item).strip().lower() for item in getattr(args, "require_tls_material", [])
    }
    env_tls_materials = os.getenv(_ENV_REQUIRE_TLS_MATERIALS)
    if env_tls_materials:
        for entry in _parse_env_list(env_tls_materials, variable=_ENV_REQUIRE_TLS_MATERIALS, parser=parser):
            normalized = entry.strip().lower()
            if normalized not in _TLS_MATERIAL_CHOICES:
                parser.error(
                    f"{_ENV_REQUIRE_TLS_MATERIALS} zawiera nieobsługiwany materiał TLS {entry!r};"
                    f" dozwolone: {', '.join(_TLS_MATERIAL_CHOICES)}"
                )
            required_tls_materials.add(normalized)
    args._required_tls_materials = tuple(sorted(required_tls_materials))

    expected_server_sha: list[str] = []
    expected_server_sha.extend(getattr(args, "expect_server_sha256", []) or [])
    env_expected_sha = os.getenv(_ENV_EXPECT_SERVER_SHA256)
    if env_expected_sha:
        expected_server_sha.extend(
            _parse_env_list(env_expected_sha, variable=_ENV_EXPECT_SERVER_SHA256, parser=parser)
        )
    unique_expected_sha: list[str] = []
    seen_sha: set[str] = set()
    for raw in expected_server_sha:
        candidate = str(raw).strip()
        if not candidate or candidate in seen_sha:
            continue
        seen_sha.add(candidate)
        unique_expected_sha.append(candidate)
    args._expected_server_sha256 = tuple(unique_expected_sha)

    expected_sources_raw: list[str] = []
    expected_sources_raw.extend(getattr(args, "expect_server_sha256_source", []) or [])
    env_expected_sources = os.getenv(_ENV_EXPECT_SERVER_SHA256_SOURCE)
    if env_expected_sources:
        expected_sources_raw.extend(
            _parse_env_list(
                env_expected_sources,
                variable=_ENV_EXPECT_SERVER_SHA256_SOURCE,
                parser=parser,
            )
        )
    normalized_sources: list[str] = []
    seen_sources: set[str] = set()
    for source in expected_sources_raw:
        normalized_source = str(source).strip().lower()
        if not normalized_source or normalized_source in seen_sources:
            continue
        seen_sources.add(normalized_source)
        normalized_sources.append(normalized_source)
    args._expected_server_sha256_sources = tuple(normalized_sources)

    # --- RiskService rozszerzenia z ENV/CLI ---------------------------------
    if not getattr(args, "require_risk_service_tls", False):
        env_risk_tls = os.getenv(_ENV_REQUIRE_RISK_TLS)
        if env_risk_tls is not None:
            args.require_risk_service_tls = _parse_env_bool(
                env_risk_tls,
                variable=_ENV_REQUIRE_RISK_TLS,
                parser=parser,
            )

    risk_tls_materials = {
        str(item).strip().lower()
        for item in getattr(args, "require_risk_service_tls_material", [])
    }
    env_risk_tls_materials = os.getenv(_ENV_REQUIRE_RISK_TLS_MATERIALS)
    if env_risk_tls_materials:
        for entry in _parse_env_list(
            env_risk_tls_materials,
            variable=_ENV_REQUIRE_RISK_TLS_MATERIALS,
            parser=parser,
        ):
            normalized = entry.strip().lower()
            if normalized not in _RISK_TLS_MATERIAL_CHOICES:
                parser.error(
                    f"{_ENV_REQUIRE_RISK_TLS_MATERIALS} zawiera nieobsługiwany materiał TLS {entry!r};"
                    f" dozwolone: {', '.join(_RISK_TLS_MATERIAL_CHOICES)}"
                )
            risk_tls_materials.add(normalized)
    args._required_risk_service_tls_materials = tuple(sorted(risk_tls_materials))

    expected_risk_sha: list[str] = []
    expected_risk_sha.extend(getattr(args, "expect_risk_service_server_sha256", []) or [])
    env_expected_risk_sha = os.getenv(_ENV_EXPECT_RISK_SERVER_SHA256)
    if env_expected_risk_sha:
        expected_risk_sha.extend(
            _parse_env_list(
                env_expected_risk_sha,
                variable=_ENV_EXPECT_RISK_SERVER_SHA256,
                parser=parser,
            )
        )
    normalized_risk_sha: list[str] = []
    seen_risk_sha: set[str] = set()
    for raw in expected_risk_sha:
        normalized = _normalize_server_sha256(raw)
        if not normalized or normalized in seen_risk_sha:
            continue
        seen_risk_sha.add(normalized)
        normalized_risk_sha.append(normalized)
    args._expected_risk_service_sha256 = tuple(normalized_risk_sha)

    required_risk_scopes_raw: list[str] = []
    required_risk_scopes_raw.extend(getattr(args, "require_risk_service_scope", []) or [])
    env_required_risk_scopes = os.getenv(_ENV_REQUIRE_RISK_SCOPE)
    if env_required_risk_scopes:
        required_risk_scopes_raw.extend(
            _parse_env_list(
                env_required_risk_scopes,
                variable=_ENV_REQUIRE_RISK_SCOPE,
                parser=parser,
            )
        )
    normalized_risk_scopes: list[str] = []
    seen_risk_scopes: set[str] = set()
    for scope in required_risk_scopes_raw:
        normalized_scope = str(scope).strip().lower()
        if not normalized_scope or normalized_scope in seen_risk_scopes:
            continue
        seen_risk_scopes.add(normalized_scope)
        normalized_risk_scopes.append(normalized_scope)
    args._required_risk_service_scopes = tuple(normalized_risk_scopes)

    required_risk_tokens_raw: list[str] = []
    required_risk_tokens_raw.extend(
        getattr(args, "require_risk_service_token_id", []) or []
    )
    env_required_risk_tokens = os.getenv(_ENV_REQUIRE_RISK_TOKEN_ID)
    if env_required_risk_tokens:
        required_risk_tokens_raw.extend(
            _parse_env_list(
                env_required_risk_tokens,
                variable=_ENV_REQUIRE_RISK_TOKEN_ID,
                parser=parser,
            )
        )
    normalized_risk_tokens: list[str] = []
    seen_risk_tokens: set[str] = set()
    for token_id in required_risk_tokens_raw:
        normalized_token = str(token_id).strip()
        if not normalized_token or normalized_token in seen_risk_tokens:
            continue
        seen_risk_tokens.add(normalized_token)
        normalized_risk_tokens.append(normalized_token)
    args._required_risk_service_token_ids = tuple(normalized_risk_tokens)

    if not getattr(args, "require_risk_service_auth_token", False):
        env_risk_token = os.getenv(_ENV_REQUIRE_RISK_AUTH_TOKEN)
        if env_risk_token is not None:
            args.require_risk_service_auth_token = _parse_env_bool(
                env_risk_token,
                variable=_ENV_REQUIRE_RISK_AUTH_TOKEN,
                parser=parser,
            )

    args._require_risk_service_tls = bool(getattr(args, "require_risk_service_tls", False))
    args._require_risk_service_auth_token = bool(
        getattr(args, "require_risk_service_auth_token", False)
    )

    required_auth_scopes_raw: list[str] = []
    required_auth_scopes_raw.extend(getattr(args, "require_auth_scopes", []) or [])
    env_required_scopes = os.getenv(_ENV_REQUIRE_AUTH_SCOPE)
    if env_required_scopes:
        required_auth_scopes_raw.extend(
            _parse_env_list(
                env_required_scopes,
                variable=_ENV_REQUIRE_AUTH_SCOPE,
                parser=parser,
            )
        )
    normalized_auth_scopes: list[str] = []
    seen_auth_scopes: set[str] = set()
    for scope in required_auth_scopes_raw:
        normalized_scope = str(scope).strip().lower()
        if not normalized_scope or normalized_scope in seen_auth_scopes:
            continue
        seen_auth_scopes.add(normalized_scope)
        normalized_auth_scopes.append(normalized_scope)
    args._required_auth_scopes = tuple(normalized_auth_scopes)
    if normalized_auth_scopes:
        args.require_auth_token = True

    # --- reszta --------------------------------------------------------------
    if getattr(args, "risk_profile", None) is None:
        env_risk_profile = os.getenv(_ENV_RISK_PROFILE)
        if env_risk_profile:
            args.risk_profile = env_risk_profile.strip().lower()
            args._risk_profile_source = "env"

    if not getattr(args, "require_screen_info", False):
        env_screen = os.getenv(_ENV_REQUIRE_SCREEN_INFO)
        if env_screen is not None:
            args.require_screen_info = _parse_env_bool(
                env_screen, variable=_ENV_REQUIRE_SCREEN_INFO, parser=parser
            )

    if getattr(args, "summary_json", None) is None:
        env_summary_path = os.getenv(_ENV_SUMMARY_PATH)
        if env_summary_path:
            args.summary_json = env_summary_path

    if getattr(args, "report_output", None) is None:
        env_report_path = os.getenv(_ENV_REPORT_OUTPUT)
        if env_report_path:
            args.report_output = env_report_path

    if getattr(args, "risk_profile_env_snippet", None) is None:
        env_env_snippet = os.getenv(_ENV_RISK_PROFILE_ENV_SNIPPET)
        if env_env_snippet:
            args.risk_profile_env_snippet = env_env_snippet

    if getattr(args, "risk_profile_yaml_snippet", None) is None:
        env_yaml_snippet = os.getenv(_ENV_RISK_PROFILE_YAML_SNIPPET)
        if env_yaml_snippet:
            args.risk_profile_yaml_snippet = env_yaml_snippet

    max_event_counts: dict[str, int] = {}
    env_event_limits = os.getenv(_ENV_MAX_EVENT_COUNTS)
    if env_event_limits:
        try:
            env_limits_obj = json.loads(env_event_limits)
        except json.JSONDecodeError as exc:
            parser.error(f"{_ENV_MAX_EVENT_COUNTS} zawiera nieprawidłowy JSON: {exc}")
        if not isinstance(env_limits_obj, Mapping):
            parser.error(f"{_ENV_MAX_EVENT_COUNTS} musi być obiektem JSON mapującym zdarzenie na limit")
        for key, value in env_limits_obj.items():
            if not isinstance(value, int):
                parser.error(
                    f"{_ENV_MAX_EVENT_COUNTS} ma nieprawidłową wartość dla zdarzenia {key!r}; oczekiwano liczby całkowitej"
                )
            if value < 0:
                parser.error(
                    f"{_ENV_MAX_EVENT_COUNTS} ma ujemny limit dla zdarzenia {key!r}; oczekiwano wartości >= 0"
                )
            max_event_counts[str(key)] = int(value)

    cli_limits = _load_event_count_specs(cli_limits=args.max_event_count, parser=parser)
    max_event_counts.update(cli_limits)
    args._max_event_counts = max_event_counts

    min_event_counts: dict[str, int] = {}
    env_min_limits = os.getenv(_ENV_MIN_EVENT_COUNTS)
    if env_min_limits:
        try:
            env_min_limits_obj = json.loads(env_min_limits)
        except json.JSONDecodeError as exc:
            parser.error(f"{_ENV_MIN_EVENT_COUNTS} zawiera nieprawidłowy JSON: {exc}")
        if not isinstance(env_min_limits_obj, Mapping):
            parser.error(f"{_ENV_MIN_EVENT_COUNTS} musi być obiektem JSON mapującym zdarzenie na minimalną liczbę wystąpień")
        for key, value in env_min_limits_obj.items():
            if not isinstance(value, int):
                parser.error(
                    f"{_ENV_MIN_EVENT_COUNTS} ma nieprawidłową wartość dla zdarzenia {key!r}; oczekiwano liczby całkowitej"
                )
            if value < 0:
                parser.error(
                    f"{_ENV_MIN_EVENT_COUNTS} ma ujemne minimum dla zdarzenia {key!r}; oczekiwano wartości >= 0"
                )
            min_event_counts[str(key)] = int(value)

    cli_min_limits = _load_event_count_specs(cli_limits=args.min_event_count, parser=parser)
    min_event_counts.update(cli_min_limits)
    args._min_event_counts = min_event_counts

    expected_filters: dict[str, Any] = {}
    env_filters_payload = os.getenv(_ENV_EXPECT_FILTERS)
    if env_filters_payload:
        try:
            env_filters_obj = json.loads(env_filters_payload)
        except json.JSONDecodeError as exc:
            parser.error(f"{_ENV_EXPECT_FILTERS} zawiera nieprawidłowy JSON: {exc}")
        if not isinstance(env_filters_obj, Mapping):
            parser.error(f"{_ENV_EXPECT_FILTERS} musi być obiektem JSON")
        for key, value in env_filters_obj.items():
            expected_filters[str(key)] = value

    cli_filters = _load_expected_filters(cli_filters=args.expect_filter, parser=parser)
    expected_filters.update(cli_filters)
    args._expected_filters = expected_filters


def _apply_core_config_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    config_path = getattr(args, "core_config", None)
    if not config_path:
        return

    if load_core_config is None:  # pragma: no cover - brak modułu konfiguracji
        parser.error("Obsługa --core-config wymaga modułu bot_core.config")

    target = Path(str(config_path)).expanduser()
    if not target.exists():
        parser.error(f"Plik konfiguracji core '{target}' nie istnieje")

    try:
        core_config = load_core_config(target)
    except Exception as exc:  # pragma: no cover - błędna konfiguracja
        parser.error(f"Nie udało się wczytać konfiguracji {target}: {exc}")

    metadata: dict[str, Any] = {"path": str(target)}
    metrics_config = getattr(core_config, "metrics_service", None)
    if metrics_config is None:
        metadata["warning"] = "metrics_service_missing"
        args._core_config_metadata = metadata
        return

    metrics_meta: dict[str, Any] = {
        "host": getattr(metrics_config, "host", None),
        "port": getattr(metrics_config, "port", None),
        "risk_profile": getattr(metrics_config, "ui_alerts_risk_profile", None),
        "risk_profiles_file": getattr(
            metrics_config, "ui_alerts_risk_profiles_file", None
        ),
    }
    metadata["metrics_service"] = {
        key: value for key, value in metrics_meta.items() if value not in (None, "")
    }
    args._core_config_metadata = metadata

    profiles_file = metrics_meta.get("risk_profiles_file")
    if not getattr(args, "risk_profiles_file", None) and profiles_file:
        args.risk_profiles_file = profiles_file

    profile_name = metrics_meta.get("risk_profile")
    if not getattr(args, "risk_profile", None) and profile_name:
        args.risk_profile = profile_name
        args._risk_profile_source = "core_config"


def _apply_risk_profile_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    profile_name = getattr(args, "risk_profile", None)
    if not profile_name:
        args._risk_profile_config = None
        args._risk_profile_summary = None
        return

    normalized = profile_name.strip().lower()

    try:
        base_config = get_risk_profile(normalized)
    except KeyError:
        parser.error(
            f"Profil ryzyka {profile_name!r} nie jest obsługiwany. Dostępne: {', '.join(list_risk_profile_names())}"
        )
    args.risk_profile = normalized
    profile_config = risk_profile_metadata(normalized)
    source_label = getattr(args, "_risk_profile_source", None)
    if source_label:
        profile_config = dict(profile_config)
        profile_config.setdefault("source", source_label)

    if profile_config.get("expect_summary_enabled") and not args.expect_summary_enabled:
        args.expect_summary_enabled = True

    if profile_config.get("expect_mode") and args.expect_mode is None:
        args.expect_mode = profile_config["expect_mode"]

    if profile_config.get("require_tls") and not args.require_tls:
        args.require_tls = True

    if profile_config.get("require_auth_token") and not args.require_auth_token:
        args.require_auth_token = True

    if profile_config.get("require_screen_info"):
        args.require_screen_info = True

    expected_filters = dict(getattr(args, "_expected_filters", {}))
    for key, value in base_config.get("expected_filters", {}).items():
        expected_filters.setdefault(key, value)

    severity_min = base_config.get("severity_min")
    if severity_min:
        normalized_severity = _normalize_severity(severity_min)
        if normalized_severity is None:
            parser.error(
                f"Profil ryzyka {profile_name} ma nieprawidłowy próg severity_min: {severity_min!r}"
            )
        profile_config["severity_min"] = normalized_severity

    args._expected_filters = expected_filters

    max_event_counts = dict(getattr(args, "_max_event_counts", {}))
    for key, value in base_config.get("max_event_counts", {}).items():
        max_event_counts.setdefault(key, value)
    args._max_event_counts = max_event_counts

    min_event_counts = dict(getattr(args, "_min_event_counts", {}))
    for key, value in base_config.get("min_event_counts", {}).items():
        min_event_counts.setdefault(key, value)
    args._min_event_counts = min_event_counts

    profile_summary = summarize_risk_profile(profile_config)
    profile_config = dict(profile_config)
    profile_config["summary"] = profile_summary
    args._risk_profile_summary = profile_summary
    args._risk_profile_config = profile_config


def _enforce_risk_profile_requirements(
    *,
    config: Mapping[str, Any] | None,
    snapshots: Sequence[Mapping[str, Any]],
) -> None:
    if not config:
        return

    severity_min = config.get("severity_min")
    if severity_min:
        for index, entry in enumerate(snapshots, start=1):
            severity_value = _normalize_severity(entry.get("severity"))
            if severity_value is None or not _severity_at_least(severity_value, severity_min):
                raise VerificationError(
                    f"Profil ryzyka {config.get('name', 'unknown')} wymaga severity >= {severity_min}; wpis #{index}"
                    f" posiada severity {severity_value!r}"
                )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    # Zresetuj stan profili ryzyka, aby unikać przecieków pomiędzy wywołaniami.
    reset_risk_profile_store()

    # źródło profilu ryzyka do metadanych
    args._risk_profile_source = None
    provided_flags = {arg for arg in (argv or []) if isinstance(arg, str) and arg.startswith("--")}
    if "--risk-profile" in provided_flags:
        args._risk_profile_source = "cli"

    # ENV → CLI, core.yaml, plik presetów → zastosowanie profilu
    _apply_env_defaults(args, parser)
    _apply_core_config_defaults(args, parser)
    _load_risk_profile_presets(args, parser)
    _apply_risk_profile_defaults(args, parser)

    core_metadata: Mapping[str, Any] | None = getattr(args, "_core_config_metadata", None)

    if getattr(args, "print_risk_profiles", False):
        _print_available_risk_profiles(getattr(args, "risk_profile", None), core_metadata=core_metadata)
        return 0

    if args.path is None:
        parser.error("Nie podano ścieżki do decision logu")

    signing_key: _SigningKey | None = getattr(args, "_signing_key", None)
    if args.expected_key_id:
        if signing_key is None:
            parser.error("--expected-key-id wymaga dostarczenia klucza HMAC")
        if signing_key.key_id and signing_key.key_id != args.expected_key_id:
            parser.error(
                "Identyfikator klucza z CLI/ENV nie zgadza się z oczekiwanym (--expected-key-id)"
            )
        if signing_key.key_id != args.expected_key_id:
            signing_key = _SigningKey(value=signing_key.value, key_id=args.expected_key_id)

    summary_path = getattr(args, "summary_json", None)
    report_output = getattr(args, "report_output", None)
    max_event_counts: Mapping[str, int] = getattr(args, "_max_event_counts", {})
    min_event_counts: Mapping[str, int] = getattr(args, "_min_event_counts", {})
    risk_profile_config: Mapping[str, Any] | None = getattr(args, "_risk_profile_config", None)

    if summary_path == "-" and args.path == "-":
        parser.error("Nie można czytać decision logu i podsumowania jednocześnie ze STDIN")

    collect_summary = bool(
        summary_path
        or report_output
        or max_event_counts
        or min_event_counts
        or getattr(args, "risk_profile_env_snippet", None)
        or getattr(args, "risk_profile_yaml_snippet", None)
    )
    if risk_profile_config and risk_profile_config.get("severity_min"):
        collect_summary = True

    summary_validation: Mapping[str, Any] | None = None
    result: Mapping[str, Any]
    exit_code = 0
    try:
        result = verify_log(
            args.path,
            signing_key=signing_key,
            allow_unsigned=args.allow_unsigned,
            require_screen_info=args.require_screen_info,
            collect_summary=collect_summary,
        )
        _validate_metadata(
            result.get("metadata"),
            expect_mode=args.expect_mode,
            expect_summary=args.expect_summary_enabled,
            expected_filters=getattr(args, "_expected_filters", {}),
            require_auth_token=args.require_auth_token,
            required_auth_scopes=getattr(args, "_required_auth_scopes", ()),
            require_tls=args.require_tls,
            expect_input_file=args.expect_input_file,
            expect_endpoint=args.expect_endpoint,
            required_tls_materials=getattr(args, "_required_tls_materials", ()),
            expected_server_sha256=getattr(args, "_expected_server_sha256", ()),
            expected_server_sha256_sources=getattr(args, "_expected_server_sha256_sources", ()),
        )
        metadata = result.get("metadata")
        _validate_risk_service_metadata(
            _extract_risk_service_metadata(metadata),
            require_tls=getattr(args, "_require_risk_service_tls", False),
            required_materials=getattr(args, "_required_risk_service_tls_materials", ()),
            expected_fingerprints=getattr(args, "_expected_risk_service_sha256", ()),
            required_scopes=getattr(args, "_required_risk_service_scopes", ()),
            required_token_ids=getattr(args, "_required_risk_service_token_ids", ()),
            require_auth_token=getattr(args, "_require_risk_service_auth_token", False),
        )
        if metadata:
            _ensure_filter_matches_snapshots(metadata=metadata, snapshots=result.get("snapshots", []))
        summary_signature_meta = None
        if isinstance(metadata, Mapping):
            summary_signature_meta = metadata.get("summary_signature")
        if summary_path:
            summary_validation = _validate_summary_path(
                summary_path,
                result.get("summary"),
                signing_key=signing_key,
                metadata_signature=summary_signature_meta,
            )
        _enforce_event_limits(summary=result.get("summary"), limits=max_event_counts)
        _enforce_min_event_counts(summary=result.get("summary"), requirements=min_event_counts)
        _enforce_risk_profile_requirements(
            config=risk_profile_config, snapshots=result.get("snapshots", [])
        )
    except VerificationError as exc:
        LOGGER.error("Walidacja nie powiodła się: %s", exc)
        return 2

    metadata_info = result.get("metadata") or {}

    risk_profile_summary_for_report: Mapping[str, Any] | None = None
    if summary_validation and isinstance(summary_validation, Mapping):
        rp_summary = summary_validation.get("risk_profile_summary")
        if isinstance(rp_summary, Mapping):
            risk_profile_summary_for_report = rp_summary
    if not risk_profile_summary_for_report and isinstance(metadata_info, Mapping):
        rp_summary = metadata_info.get("risk_profile_summary")
        if isinstance(rp_summary, Mapping):
            risk_profile_summary_for_report = rp_summary
    if not risk_profile_summary_for_report:
        fallback_summary = getattr(args, "_risk_profile_summary", None)
        if isinstance(fallback_summary, Mapping):
            risk_profile_summary_for_report = fallback_summary

    recommended_overrides: Mapping[str, Any] | None = None
    profile_name = None
    if isinstance(risk_profile_summary_for_report, Mapping):
        recommended_overrides = risk_profile_summary_for_report.get("recommended_overrides")
        profile_name = risk_profile_summary_for_report.get("name")
    if recommended_overrides is None:
        fallback_summary = getattr(args, "_risk_profile_summary", None)
        if isinstance(fallback_summary, Mapping):
            recommended_overrides = fallback_summary.get("recommended_overrides")
            profile_name = profile_name or fallback_summary.get("name")

    snippet_validations, snippet_errors = _validate_risk_profile_snippets(
        env_path=getattr(args, "risk_profile_env_snippet", None),
        yaml_path=getattr(args, "risk_profile_yaml_snippet", None),
        recommended_overrides=recommended_overrides,
        profile_name=profile_name or getattr(args, "risk_profile", None),
    )

    for error_message in snippet_errors:
        LOGGER.error("Walidacja snippetów profilu ryzyka: %s", error_message)
    if snippet_errors:
        exit_code = 2

    if report_output:
        _write_report_output(
            report_output,
            _build_report_payload(
                result,
                metadata_info,
                summary_validation,
                enforced_limits=max_event_counts,
                enforced_minimums=min_event_counts,
                risk_profile=risk_profile_config,
                risk_profile_summary=risk_profile_summary_for_report,
                core_config=core_metadata,
                risk_profile_snippet_validation=snippet_validations,
            ),
        )
    metadata_line = json.dumps(metadata_info, ensure_ascii=False)
    if exit_code == 0:
        LOGGER.info(
            "OK: zweryfikowano %s wpisów (plik=%s)\nMetadane: %s",
            result["verified_entries"],
            result["path"],
            metadata_line,
        )
    else:
        LOGGER.error(
            "Walidacja zakończona z błędami snippetów profilu ryzyka (zweryfikowano %s wpisów, plik=%s).\nMetadane: %s",
            result["verified_entries"],
            result["path"],
            metadata_line,
        )
    return exit_code


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())
