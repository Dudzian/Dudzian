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

try:
    from bot_core.config import load_core_config  # type: ignore
except Exception:  # pragma: no cover - brak modułu konfiguracji
    load_core_config = None  # type: ignore

from scripts.telemetry_risk_profiles import (
    get_risk_profile,
    list_risk_profile_names,
    load_risk_profiles_from_file,
    risk_profile_metadata,
)

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
    require_tls: bool,
    expect_input_file: str | None,
    expect_endpoint: str | None,
) -> None:
    expectations_defined = any(
        [
            expect_mode,
            expect_summary,
            expected_filters,
            require_auth_token,
            require_tls,
            expect_input_file,
            expect_endpoint,
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

    if require_tls:
        if mode != "grpc":
            raise VerificationError("Wymagano TLS dla logu w trybie grpc")
        if not metadata.get("use_tls", False):
            raise VerificationError("Metadane wskazują, że połączenie gRPC nie używa TLS")


def _matches_screen_name(screen: Mapping[str, Any] | None, *, expected_substring: str) -> bool:
    if not expected_substring:
        return True
    if not isinstance(screen, Mapping):
        return False
    candidate = screen.get("name")
    if not isinstance(candidate, str):
        return False
    return expected_substring.casefold() in candidate.casefold()


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
    core_config: Mapping[str, Any] | None = None,
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
    if core_config:
        payload["core_config"] = dict(core_config)
    return payload


def _write_report_output(destination: str, payload: Mapping[str, Any]) -> None:
    if destination.strip() == "-":
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

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
        "--require-tls",
        action="store_true",
        help="Wymagaj, by log gRPC był zarejestrowany przy użyciu TLS",
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
        registered = load_risk_profiles_from_file(target, origin=f"verify:{target}")
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
    args._risk_profile_source = None
    provided_flags = {arg for arg in (argv or []) if arg.startswith("--")}
    if "--risk-profile" in provided_flags:
        args._risk_profile_source = "cli"

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
    # core_metadata already resolved powyżej
    if summary_path == "-" and args.path == "-":
        parser.error("Nie można czytać decision logu i podsumowania jednocześnie ze STDIN")

    collect_summary = bool(summary_path or report_output or max_event_counts or min_event_counts)
    if risk_profile_config and risk_profile_config.get("severity_min"):
        collect_summary = True

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
            require_tls=args.require_tls,
            expect_input_file=args.expect_input_file,
            expect_endpoint=args.expect_endpoint,
        )
        metadata = result.get("metadata")
        if metadata:
            _ensure_filter_matches_snapshots(metadata=metadata, snapshots=result.get("snapshots", []))
        summary_signature_meta = None
        if isinstance(metadata, Mapping):
            summary_signature_meta = metadata.get("summary_signature")
        summary_validation: Mapping[str, Any] | None = None
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
                core_config=core_metadata,
            ),
        )
    metadata_line = json.dumps(metadata_info, ensure_ascii=False)
    LOGGER.info(
        "OK: zweryfikowano %s wpisów (plik=%s)\nMetadane: %s",
        result["verified_entries"],
        result["path"],
        metadata_line,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())
