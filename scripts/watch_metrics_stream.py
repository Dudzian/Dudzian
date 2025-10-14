#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Narzędzie do podglądu strumienia MetricsService.

Skrypt łączy się z serwerem telemetrii `MetricsService` (gRPC) i wypisuje
otrzymane `MetricsSnapshot`. Może filtrować zdarzenia po polu `event` oraz
`severity` w `notes`, ograniczać liczbę pobranych rekordów, filtrować po
metadanych ekranu i formatować wynik w postaci czytelnej tabeli lub surowego
JSON-u.

Przykłady użycia::

    # Podgląd ostatnich 5 rekordów w formacie tabeli
    python -m scripts.watch_metrics_stream --limit 5

    # Wyłącznie zdarzenia reduce_motion jako JSON
    python -m scripts.watch_metrics_stream --event reduce_motion --format json

    # Alerty o severity co najmniej warning
    python -m scripts.watch_metrics_stream --severity-min warning

    # Podłączenie do niestandardowego hosta/portu
    python -m scripts.watch_metrics_stream --host 10.0.0.5 --port 50070

    # Połączenie z serwerem wymagającym tokenu Bearer
    python -m scripts.watch_metrics_stream --auth-token secret --limit 10

    # Filtruj tylko zdarzenia z monitora o indeksie 1
    python -m scripts.watch_metrics_stream --screen-index 1

    # Token z pliku oraz konfiguracja TLS przez zmienne środowiskowe
    export BOT_CORE_WATCH_METRICS_ROOT_CERT=secrets/root.pem
    export BOT_CORE_WATCH_METRICS_SERVER_SHA256=0123deadbeef
    python -m scripts.watch_metrics_stream --auth-token-file secrets/token.txt

    # Wczytaj dodatkowe nagłówki gRPC z pliku poprzez zmienną środowiskową
    export BOT_CORE_WATCH_METRICS_HEADERS_FILE=config/metrics_headers.env
    python -m scripts.watch_metrics_stream --summary

    # Audytuj zdarzenia z określonego okna czasowego
    python -m scripts.watch_metrics_stream \
        --from-jsonl artifacts/metrics.jsonl --since 2024-02-01T00:00:00Z \
        --until 2024-02-01T01:00:00Z

    # Analiza artefaktu skompresowanego gzipem
    python -m scripts.watch_metrics_stream --from-jsonl artifacts/metrics.jsonl.gz --summary

    # Wczytaj snapshoty z STDIN (np. po rozpakowaniu artefaktu w potoku)
    zcat artifacts/metrics.jsonl.gz | python -m scripts.watch_metrics_stream --from-jsonl -

    # Audytuj artefakt JSONL i wypisz podsumowanie zdarzeń wraz z poziomami severity
    python -m scripts.watch_metrics_stream \
        --from-jsonl artifacts/metrics.jsonl --summary --severity warning

    # Zapisz podsumowanie zdarzeń do pliku JSON obok standardowego wypisu
    python -m scripts.watch_metrics_stream \
        --from-jsonl artifacts/metrics.jsonl --summary --summary-output out/summary.json

    # Decision log z podpisem HMAC i identyfikatorem klucza rotacyjnego
    python -m scripts.watch_metrics_stream \
        --decision-log logs/ui_decision.jsonl \
        --decision-log-hmac-key-file secrets/ops_decision.key \
        --decision-log-key-id ops-2024q1

    # Wczytaj ustawienia telemetryczne (host, TLS, profil ryzyka) z core.yaml
    python -m scripts.watch_metrics_stream --core-config config/core.yaml --summary

    # Wypisz scalone nagłówki gRPC bez łączenia z serwerem
    python -m scripts.watch_metrics_stream --headers-report-only --header x-trace=demo

Skrypt wymaga obecności wygenerowanych stubów gRPC (`trading_pb2*.py`) oraz
pakietu `grpcio`.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from collections import OrderedDict, defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import gzip
import hmac
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

MetadataValue = str | bytes

# --- opcjonalna konfiguracja core.yaml ---------------------------------------
try:
    from bot_core.config import load_core_config  # type: ignore
except Exception:  # pragma: no cover - środowisko bez modułu konfiguracyjnego
    load_core_config = None  # type: ignore

try:  # pragma: no cover - moduł bezpieczeństwa jest opcjonalny w części środowisk
    from bot_core.security.tokens import (  # type: ignore
        resolve_service_token,
        resolve_service_token_secret,
    )
except Exception:  # pragma: no cover - brak modułu bezpieczeństwa
    resolve_service_token = None  # type: ignore
    resolve_service_token_secret = None  # type: ignore

# --- presety profili ryzyka (z fallbackiem, patrz scripts.telemetry_risk_profiles) --
try:
    from scripts.telemetry_risk_profiles import (
        get_risk_profile,
        list_risk_profile_names,
        load_risk_profiles_with_metadata,
        risk_profile_metadata,
        summarize_risk_profile,
    )
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Brak modułu scripts.telemetry_risk_profiles. Upewnij się, że jest na PYTHONPATH."
    ) from exc

LOGGER = logging.getLogger("bot_core.scripts.watch_metrics_stream")

_ENV_PREFIX = "BOT_CORE_WATCH_METRICS_"
_CLI_HEADER_SOURCE = "cli:--header"
_ENV_HEADER_SOURCE = f"env:{_ENV_PREFIX}HEADERS"
_CLI_HEADER_DIR_SOURCE = "cli:--headers-dir"
_ENV_HEADER_DIR_SOURCE = f"env:{_ENV_PREFIX}HEADERS_DIRS"
_HEX_DIGITS = set("0123456789abcdef")
_REQUIRED_METRICS_SCOPE = "metrics.read"
_REQUIRED_RISK_SCOPE = "risk.read"
_METADATA_KEY_PATTERN = re.compile(r"^[0-9a-z._-]+$")


def _parse_pinned_fingerprint(entry: object) -> tuple[str, str] | None:
    """Zwraca parę (algorytm, fingerprint) dla wpisu pinningu TLS."""

    if entry in (None, ""):
        return None
    text = str(entry).strip().lower()
    if not text:
        return None
    if ":" in text:
        algorithm, fingerprint = text.split(":", 1)
        algorithm = algorithm.strip() or "sha256"
    else:
        algorithm, fingerprint = "sha256", text
    fingerprint = fingerprint.replace(":", "").strip()
    if not fingerprint:
        return None
    if any(char not in _HEX_DIGITS for char in fingerprint):
        LOGGER.warning(
            "Wpis pinningu TLS '%s' zawiera znaki spoza zakresu hex – pomijam", entry
        )
        return None
    return algorithm, fingerprint


def _select_sha256_fingerprint(entries: Sequence[object]) -> str | None:
    """Wybiera fingerprint SHA-256 spośród wpisów pinningu."""

    for entry in entries:
        parsed = _parse_pinned_fingerprint(entry)
        if not parsed:
            continue
        algorithm, fingerprint = parsed
        if algorithm == "sha256":
            return fingerprint
        LOGGER.warning(
            "Pomijam wpis pinningu TLS %r – obsługiwany jest wyłącznie SHA-256", entry
        )
    return None


def _looks_like_sensitive_metadata_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in ("auth", "token", "secret", "key"))


_METADATA_CLEAR_SENTINELS = {"none", "null"}


def _split_header_entries(raw_value: str) -> list[str]:
    """Dzieli łańcuch nagłówków na poszczególne wpisy.

    Obsługuje zarówno separatory średnikowe, jak i wieloliniowe. Linie zaczynające
    się od `#` są traktowane jako komentarze i ignorowane.
    """

    entries: list[str] = []
    for chunk in raw_value.replace("\r\n", "\n").split("\n"):
        piece = chunk.strip()
        if not piece or piece.startswith("#"):
            continue
        for entry in piece.split(";"):
            normalized = entry.strip()
            if normalized:
                entries.append(normalized)
    return entries


def _split_header_file_specs(raw_value: str) -> list[str]:
    """Zwraca listę ścieżek zdefiniowanych w zmiennej HEADERS_FILE."""

    paths: list[str] = []
    for chunk in raw_value.replace("\r\n", "\n").split("\n"):
        piece = chunk.strip()
        if not piece or piece.startswith("#"):
            continue
        for candidate in piece.split(";"):
            normalized = candidate.strip()
            if normalized:
                paths.append(normalized)
    return paths


def _split_header_directory_specs(raw_value: str) -> list[str]:
    """Zwraca listę katalogów zdefiniowanych w zmiennych HEADERS_DIRS."""

    return _split_header_file_specs(raw_value)


def _resolve_headers_directory(
    path_value: str, *, parser: argparse.ArgumentParser
) -> tuple[list[str], str]:
    """Zwraca posortowaną listę plików nagłówków z katalogu."""

    target = Path(path_value).expanduser()
    if not target.exists():
        parser.error(f"Nie znaleziono katalogu nagłówków gRPC: {target}")
    if not target.is_dir():
        parser.error(f"Ścieżka {target} nie jest katalogiem nagłówków gRPC")
    files = sorted(entry for entry in target.iterdir() if entry.is_file())
    return [str(item) for item in files], str(target)


def _load_headers_file_entries(
    path_value: str, *, parser: argparse.ArgumentParser
) -> tuple[list[str], str]:
    """Wczytuje nagłówki gRPC z pliku i zwraca je wraz z kanoniczną ścieżką."""

    target = Path(path_value).expanduser()
    try:
        content = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        parser.error(f"Nie znaleziono pliku nagłówków gRPC: {target}")
    except OSError as exc:  # pragma: no cover - zależne od środowiska
        parser.error(f"Nie można odczytać pliku nagłówków gRPC {target}: {exc}")
    return _split_header_entries(content), str(target)


def _decode_base64_payload(
    payload: str,
    *,
    parser: argparse.ArgumentParser,
    key: str,
    entry: str,
    source: str,
) -> bytes:
    """Dekoduje wartość w formacie Base64, zgłaszając błąd parsera przy niepowodzeniu."""

    normalized = "".join(payload.split())
    if not normalized:
        parser.error(
            f"Nagłówek '{key}' otrzymał pustą wartość base64 w wpisie '{entry}' (źródło {source})"
        )
    try:
        return base64.b64decode(normalized, validate=True)
    except binascii.Error:
        parser.error(
            f"Nagłówek '{key}' nie zawiera poprawnej wartości base64 w wpisie '{entry}' (źródło {source})"
        )


def _resolve_metadata_value(
    value: str,
    *,
    parser: argparse.ArgumentParser,
    key: str,
    entry: str,
    env: Mapping[str, str] | None = None,
) -> tuple[MetadataValue, str | None]:
    """Zwraca docelową wartość nagłówka oraz opis źródła (jeśli dotyczy).

    Wartości mogą wskazywać na plik (`@file:path`) lub zmienną środowiskową
    (`@env:NAME`). Sekwencja `@@` pozwala wprowadzić dosłowną wartość
    rozpoczynającą się od `@`.
    """

    if not value.startswith("@"):
        return value, None

    if value.startswith("@@"):
        # Escapowany znak `@` – obcinamy pierwszy prefiksowy znak.
        return value[1:], None

    directive, separator, remainder = value[1:].partition(":")
    if not separator:
        # Nieznana dyrektywa – traktujemy jak literalną wartość.
        return value, None

    directive_lower = directive.lower()
    target = remainder.strip()

    if directive_lower == "file":
        if not target:
            parser.error(
                f"Nagłówek '{key}' wskazuje pusty plik w wpisie '{entry}'"
            )
        file_path = Path(target).expanduser()
        try:
            content = file_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            parser.error(
                f"Nie znaleziono pliku '{file_path}' dla nagłówka '{key}'"
            )
        except OSError as exc:  # pragma: no cover - zależne od środowiska
            parser.error(
                f"Nie można odczytać pliku '{file_path}' dla nagłówka '{key}': {exc}"
            )
        # Usuwamy typowe końcówki linii pozostawiając resztę wartości bez zmian.
        return content.rstrip("\r\n"), f"file:{file_path}"

    if directive_lower in {"file64", "fileb64", "file-base64"}:
        if not target:
            parser.error(
                f"Nagłówek '{key}' wskazuje pusty plik w wpisie '{entry}'"
            )
        file_path = Path(target).expanduser()
        try:
            content = file_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            parser.error(
                f"Nie znaleziono pliku '{file_path}' dla nagłówka '{key}'"
            )
        except OSError as exc:  # pragma: no cover - zależne od środowiska
            parser.error(
                f"Nie można odczytać pliku '{file_path}' dla nagłówka '{key}': {exc}"
            )
        decoded = _decode_base64_payload(
            content,
            parser=parser,
            key=key,
            entry=entry,
            source=f"file:{file_path}",
        )
        return decoded, f"file:{file_path}"

    if directive_lower == "env":
        if not target:
            parser.error(
                f"Nagłówek '{key}' wskazuje pustą zmienną środowiskową w wpisie '{entry}'"
            )
        env_map = dict(env or os.environ)
        if target not in env_map:
            parser.error(
                f"Zmiennej środowiskowej '{target}' wymaganej przez nagłówek '{key}' nie znaleziono"
            )
        return env_map[target], f"env:{target}"

    if directive_lower in {"env64", "envb64", "env-base64"}:
        if not target:
            parser.error(
                f"Nagłówek '{key}' wskazuje pustą zmienną środowiskową w wpisie '{entry}'"
            )
        env_map = dict(env or os.environ)
        if target not in env_map:
            parser.error(
                f"Zmiennej środowiskowej '{target}' wymaganej przez nagłówek '{key}' nie znaleziono"
            )
        decoded = _decode_base64_payload(
            env_map[target],
            parser=parser,
            key=key,
            entry=entry,
            source=f"env:{target}",
        )
        return decoded, f"env:{target}"

    if directive_lower == "literal":
        # Pozwala jawnie podać wartość zaczynającą się od '@'.
        return remainder, None

    # Nieznana dyrektywa – pozostawiamy wartość bez zmian.
    return value, None


def _parse_metadata_entries(
    entries: Sequence[str],
    *,
    parser: argparse.ArgumentParser,
    env: Mapping[str, str] | None = None,
    base_source: str | None = None,
) -> tuple[
    list[tuple[str, MetadataValue]],
    set[str],
    dict[str, str],
    dict[str, str],
]:
    metadata: list[tuple[str, MetadataValue]] = []
    removals: set[str] = set()
    sources: dict[str, str] = {}
    removal_sources: dict[str, str] = {}
    env_map = env or os.environ
    for raw_entry in entries:
        entry = raw_entry.strip()
        if not entry:
            continue
        if "=" in entry:
            key, value = entry.split("=", 1)
        elif ":" in entry:
            key, value = entry.split(":", 1)
        else:
            parser.error(
                "Nieprawidłowy nagłówek gRPC '%s'. Użyj formatu klucz=wartość lub klucz:wartość." % raw_entry
            )
        stripped_key = key.strip()
        normalized_key = stripped_key.lower()
        if not normalized_key:
            parser.error("Nagłówek gRPC o pustym kluczu jest niedozwolony")
        if stripped_key != normalized_key:
            parser.error(
                "Nagłówek gRPC '%s' musi używać małych liter" % stripped_key
            )
        if not _METADATA_KEY_PATTERN.fullmatch(normalized_key):
            parser.error(
                "Nagłówek gRPC '%s' zawiera niedozwolone znaki – dozwolone są małe litery, cyfry, '-', '_' i '.'"
                % normalized_key
            )
        normalized_value = value.strip()
        sentinel = normalized_value.lower()
        if sentinel in _METADATA_CLEAR_SENTINELS:
            removals.add(normalized_key)
            if base_source:
                removal_sources[normalized_key] = base_source
            continue
        if sentinel == "default":
            # Przywracamy konfigurację domyślną – brak wpisu CLI/ENV.
            removals.discard(normalized_key)
            removal_sources.pop(normalized_key, None)
            continue
        resolved_value, value_source = _resolve_metadata_value(
            normalized_value,
            parser=parser,
            key=normalized_key,
            entry=raw_entry,
            env=env_map,
        )
        if isinstance(resolved_value, bytes) and not normalized_key.endswith("-bin"):
            try:
                resolved_value = resolved_value.decode("utf-8")
            except UnicodeDecodeError:
                parser.error(
                    "Nagłówek gRPC '%s' oczekuje tekstowej wartości UTF-8, otrzymano dane binarne" % normalized_key
                )
        if isinstance(resolved_value, str) and normalized_key.endswith("-bin"):
            resolved_value = _decode_base64_payload(
                resolved_value,
                parser=parser,
                key=normalized_key,
                entry=raw_entry,
                source=base_source or "inline",
            )
        removals.discard(normalized_key)
        removal_sources.pop(normalized_key, None)
        metadata.append((normalized_key, resolved_value))
        if base_source:
            detail_source = base_source
            if value_source:
                detail_source = f"{base_source}@{value_source}"
            sources[normalized_key] = detail_source
    return metadata, removals, sources, removal_sources


def _merge_metadata_entries(
    entries: Sequence[tuple[str, MetadataValue]]
) -> list[tuple[str, MetadataValue]]:
    """Zwraca listę metadanych z usuniętymi duplikatami kluczy.

    W przypadku powtórzeń ostatnia wartość wygrywa, dzięki czemu wpisy CLI
    nadpisują konfigurację `core.yaml`, a kolejność wynikowa zachowuje ostatnie
    wystąpienie każdego klucza (ważne dla serwera gRPC, który wykorzystuje
    kolejność metadanych).
    """

    deduplicated: "OrderedDict[str, MetadataValue]" = OrderedDict()
    for key, value in entries:
        if key in deduplicated:
            # usuwamy poprzednie wystąpienie, aby odtworzyć kolejność "ostatni wygrywa"
            del deduplicated[key]
        deduplicated[key] = value
    return list(deduplicated.items())


def _finalize_custom_metadata(args: argparse.Namespace) -> None:
    removals = set(getattr(args, "_custom_metadata_remove", set()) or ())
    entries = list(getattr(args, "_custom_metadata", []) or [])

    if removals and entries:
        entries = [pair for pair in entries if pair[0] not in removals]

    merged = _merge_metadata_entries(entries) if entries else []
    args._custom_metadata = merged or None

    sources_map = dict(getattr(args, "_custom_metadata_sources", {}) or {})
    merged_sources: dict[str, str] = {}
    if sources_map and merged:
        for key, _ in merged:
            source = sources_map.get(key)
            if source:
                merged_sources[key] = source

    args._custom_metadata_sources = merged_sources or None

    removal_sources = getattr(args, "_custom_metadata_remove_sources", {}) or {}

    core_meta = getattr(args, "_core_config_metadata", None)
    if isinstance(core_meta, dict):
        metrics_meta = core_meta.get("metrics_service")
        if isinstance(metrics_meta, dict):
            if merged_sources:
                metrics_meta["grpc_metadata_sources"] = dict(merged_sources)
            else:
                metrics_meta.pop("grpc_metadata_sources", None)
            if merged:
                metrics_meta["grpc_metadata_keys"] = [key for key, _ in merged]
                metrics_meta["grpc_metadata_count"] = len(merged)
                metrics_meta["grpc_metadata_enabled"] = True
            else:
                metrics_meta.pop("grpc_metadata_keys", None)
                metrics_meta.pop("grpc_metadata_count", None)
                metrics_meta["grpc_metadata_enabled"] = False
            if removals:
                metrics_meta["grpc_metadata_removed"] = sorted(removals)
                if removal_sources:
                    metrics_meta["grpc_metadata_removed_sources"] = dict(removal_sources)
                else:
                    metrics_meta.pop("grpc_metadata_removed_sources", None)
            else:
                metrics_meta.pop("grpc_metadata_removed", None)
                metrics_meta.pop("grpc_metadata_removed_sources", None)


def _sanitize_metadata_preview(key: str, value: MetadataValue) -> tuple[str, dict[str, object]]:
    """Zwraca bezpieczny do logowania podgląd wartości metadanych."""

    details: dict[str, object] = {"key": key}
    if isinstance(value, bytes):
        details["type"] = "binary"
        details["value_preview"] = f"<{len(value)} bytes>"
        details["length"] = len(value)
        return details["value_preview"], details

    details["type"] = "text"
    masked = _looks_like_sensitive_metadata_key(key)
    rendered = value.replace("\r", "\\r").replace("\n", "\\n")
    truncated = False
    if len(rendered) > 80:
        rendered = rendered[:77] + "..."
        truncated = True
    if masked:
        rendered = "<masked>"
        details["masked"] = True
    if truncated:
        details["truncated"] = True
    details["value_preview"] = rendered
    details["length"] = len(value)
    return rendered, details


def _build_headers_report(args: argparse.Namespace) -> dict[str, object]:
    entries = list(getattr(args, "_custom_metadata", []) or [])
    sources = dict(getattr(args, "_custom_metadata_sources", {}) or {})
    removals = set(getattr(args, "_custom_metadata_remove", set()) or ())
    removal_sources = dict(getattr(args, "_custom_metadata_remove_sources", {}) or {})
    headers_disabled = bool(getattr(args, "_headers_disabled", False))

    report: dict[str, object] = {
        "headers_disabled": headers_disabled,
        "headers_count": 0,
        "headers": [],
        "removed": [],
    }

    if headers_disabled:
        report["removed_count"] = len(removals)
        if removals:
            removed_list: list[dict[str, object]] = []
            for key in sorted(removals):
                entry: dict[str, object] = {"key": key}
                source = removal_sources.get(key)
                if source:
                    entry["source"] = source
                removed_list.append(entry)
            report["removed"] = removed_list
        return report

    summarized_headers: list[dict[str, object]] = []
    for key, value in entries:
        _, details = _sanitize_metadata_preview(key, value)
        source = sources.get(key)
        if source:
            details["source"] = source
        summarized_headers.append(details)

    report["headers"] = summarized_headers
    report["headers_count"] = len(summarized_headers)
    if removals:
        removed_list = []
        for key in sorted(removals):
            entry: dict[str, object] = {"key": key}
            source = removal_sources.get(key)
            if source:
                entry["source"] = source
            removed_list.append(entry)
        report["removed"] = removed_list
        report["removed_count"] = len(removals)
    else:
        report["removed_count"] = 0
    return report


def _print_headers_report(args: argparse.Namespace) -> None:
    report = _build_headers_report(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _load_grpc_components():
    try:
        import grpc  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Pakiet grpcio jest wymagany do połączenia z MetricsService."
        ) from exc

    try:
        from bot_core.generated import trading_pb2, trading_pb2_grpc  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Brak wygenerowanych stubów trading_pb2*.py. Uruchom scripts/generate_trading_stubs.py"
        ) from exc

    return grpc, trading_pb2, trading_pb2_grpc


def _timestamp_to_iso(timestamp) -> str | None:
    if timestamp is None:
        return None
    if isinstance(timestamp, str):
        try:
            dt = datetime.fromisoformat(timestamp)
        except ValueError:
            return timestamp
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    if isinstance(timestamp, Mapping):
        seconds = timestamp.get("seconds")
        nanos = timestamp.get("nanos")
        if seconds is None and nanos is None:
            return None
        seconds = int(seconds or 0)
        nanos = int(nanos or 0)
        dt = datetime.fromtimestamp(seconds + nanos / 1_000_000_000, tz=timezone.utc)
        return dt.isoformat()
    seconds = getattr(timestamp, "seconds", None)
    nanos = getattr(timestamp, "nanos", None)
    if not seconds and not nanos:
        return None
    seconds = seconds or 0
    nanos = nanos or 0
    dt = datetime.fromtimestamp(seconds + nanos / 1_000_000_000, tz=timezone.utc)
    return dt.isoformat()


def _parse_notes(notes: str) -> dict[str, Any]:
    if not notes:
        return {}
    try:
        payload = json.loads(notes)
        if isinstance(payload, dict):
            return payload
        return {"value": payload}
    except json.JSONDecodeError:
        return {"raw": notes}


SEVERITY_ORDER = [
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

SEVERITY_RANK = {name: index for index, name in enumerate(SEVERITY_ORDER)}


def _normalize_severity(candidate: Any) -> str | None:
    if not isinstance(candidate, str):
        return None
    normalized = candidate.strip()
    return normalized.lower() if normalized else None


def _severity_at_least(candidate: str, minimum: str) -> bool:
    cr = SEVERITY_RANK.get(candidate)
    mr = SEVERITY_RANK.get(minimum)
    return cr is not None and mr is not None and cr >= mr


def _screen_context(notes: Any) -> dict[str, Any]:
    if not isinstance(notes, dict):
        return {}
    screen = notes.get("screen")
    if not isinstance(screen, dict):
        return {}

    context: dict[str, Any] = {}

    name = screen.get("name")
    if isinstance(name, str) and name.strip():
        context["name"] = name.strip()

    index = screen.get("index")
    if isinstance(index, (int, float)):
        index_int = int(index)
        if index_int >= 0:
            context["index"] = index_int

    refresh = screen.get("refresh_hz")
    if isinstance(refresh, (int, float)) and refresh > 0:
        context["refresh_hz"] = float(refresh)

    dpr = screen.get("device_pixel_ratio")
    if isinstance(dpr, (int, float)) and dpr > 0:
        context["device_pixel_ratio"] = float(dpr)

    geometry = screen.get("geometry_px")
    if isinstance(geometry, dict):
        width = geometry.get("width")
        height = geometry.get("height")
        if isinstance(width, (int, float)) and isinstance(height, (int, float)):
            context["resolution"] = {"width": int(width), "height": int(height)}

    return context


def _screen_summary(context: dict[str, Any]) -> str:
    if not context:
        return ""
    parts: list[str] = []

    index = context.get("index")
    name = context.get("name")
    if isinstance(index, int):
        label = f"#{index}"
        parts.append(f"{label} ({name})" if isinstance(name, str) and name else label)
    elif isinstance(name, str) and name:
        parts.append(name)

    resolution = context.get("resolution")
    if isinstance(resolution, dict):
        w = resolution.get("width")
        h = resolution.get("height")
        if isinstance(w, int) and isinstance(h, int):
            parts.append(f"{w}x{h} px")

    refresh = context.get("refresh_hz")
    if isinstance(refresh, (int, float)) and refresh > 0:
        parts.append(f"{refresh:.0f} Hz")

    return ", ".join(parts) if parts else ""


def _extract_snapshot_fields(snapshot, notes_payload: Any) -> dict[str, Any]:
    has_field = getattr(snapshot, "HasField", None)
    timestamp = None
    if has_field and snapshot.HasField("generated_at"):
        timestamp = _timestamp_to_iso(snapshot.generated_at)

    fps = None
    if has_field and snapshot.HasField("fps"):
        fps = snapshot.fps

    event = None
    severity = None
    if isinstance(notes_payload, dict):
        raw_event = notes_payload.get("event")
        if isinstance(raw_event, str) and raw_event:
            event = raw_event
        severity = _normalize_severity(notes_payload.get("severity"))

    screen_ctx = _screen_context(notes_payload)

    return {
        "timestamp": timestamp,
        "fps": fps,
        "event": event,
        "severity": severity,
        "screen": screen_ctx or None,
    }


class _OfflineSnapshot:
    def __init__(self, record: Mapping[str, Any]) -> None:
        raw_notes = record.get("notes")
        if isinstance(raw_notes, str):
            self.notes = raw_notes
        else:
            self.notes = json.dumps(raw_notes, ensure_ascii=False) if raw_notes is not None else ""

        raw_fps = record.get("fps")
        self.fps = float(raw_fps) if isinstance(raw_fps, (int, float)) else None

        generated_at = record.get("generated_at")
        self.generated_at = generated_at if isinstance(generated_at, (str, Mapping)) else None

    def HasField(self, field: str) -> bool:  # pragma: no cover - prosta logika
        if field == "generated_at":
            return self.generated_at is not None
        if field == "fps":
            return self.fps is not None
        return False


def _iter_jsonl_snapshots(source: str) -> Iterable[_OfflineSnapshot]:
    source_label = "stdin" if source == "-" else str(Path(source).expanduser())

    handle = None
    close_handle = False
    if source == "-":
        handle = sys.stdin
    else:
        path = Path(source).expanduser()
        try:
            if path.suffix.lower() in {".gz", ".gzip"}:
                handle = gzip.open(path, "rt", encoding="utf-8")
            else:
                handle = path.open("r", encoding="utf-8")
        except FileNotFoundError as exc:
            LOGGER.error("Nie znaleziono pliku JSONL: %s", path)
            raise SystemExit(2) from exc
        except gzip.BadGzipFile as exc:
            LOGGER.error("Nie udało się zdekompresować pliku JSONL %s: %s", path, exc)
            raise SystemExit(2) from exc
        except OSError as exc:  # pragma: no cover - zależne od platformy
            LOGGER.error("Nie udało się odczytać pliku JSONL %s: %s", path, exc)
            raise SystemExit(2) from exc
        close_handle = True

    assert handle is not None  # dla mypy

    try:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                LOGGER.warning(
                    "Pominięto wiersz %s w %s – niepoprawny JSON: %s",
                    line_number,
                    source_label,
                    exc,
                )
                continue
            if not isinstance(record, Mapping):
                LOGGER.warning(
                    "Pominięto wiersz %s w %s – oczekiwano obiektu JSON.",
                    line_number,
                    source_label,
                )
                continue
            yield _OfflineSnapshot(record)
    finally:
        if close_handle and handle is not None:
            handle.close()


class _SummaryCollector:
    def __init__(self) -> None:
        self.total = 0
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
        self._first_ts: datetime | None = None
        self._last_ts: datetime | None = None
        self._severity_totals: defaultdict[str, int] = defaultdict(int)

    def add(self, snapshot, notes_payload: Any) -> None:
        iso_ts = _snapshot_timestamp(snapshot)
        dt_ts = _parse_iso_datetime(iso_ts)
        if dt_ts:
            if self._first_ts is None or dt_ts < self._first_ts:
                self._first_ts = dt_ts
            if self._last_ts is None or dt_ts > self._last_ts:
                self._last_ts = dt_ts

        event = "unknown"
        if isinstance(notes_payload, dict):
            raw_event = notes_payload.get("event")
            if isinstance(raw_event, str) and raw_event:
                event = raw_event

        stats = self._events[event]
        stats["count"] += 1

        severity = None
        if isinstance(notes_payload, dict):
            severity = _normalize_severity(notes_payload.get("severity"))
        if severity:
            stats["severity_counts"][severity] += 1
            self._severity_totals[severity] += 1

        if dt_ts:
            if stats["first_ts"] is None or dt_ts < stats["first_ts"]:
                stats["first_ts"] = dt_ts
            if stats["last_ts"] is None or dt_ts > stats["last_ts"]:
                stats["last_ts"] = dt_ts

        fps_value = _snapshot_fps(snapshot)
        if fps_value is not None:
            stats["fps_count"] += 1
            stats["fps_total"] += fps_value
            stats["fps_min"] = (
                fps_value if stats["fps_min"] is None or fps_value < stats["fps_min"] else stats["fps_min"]
            )
            stats["fps_max"] = (
                fps_value if stats["fps_max"] is None or fps_value > stats["fps_max"] else stats["fps_max"]
            )

        screen_ctx = _screen_context(notes_payload)
        if screen_ctx:
            stats["screens"].add(json.dumps(screen_ctx, sort_keys=True, ensure_ascii=False))

        self.total += 1

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

        summary: dict[str, Any] = {"total_snapshots": self.total, "events": events_summary}
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


def _canonical_payload(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sign_payload(
    payload: Mapping[str, Any],
    *,
    signing_key: bytes | None,
    signing_key_id: str | None,
) -> Mapping[str, Any]:
    if not signing_key:
        return dict(payload)

    digest = hmac.new(signing_key, _canonical_payload(payload), hashlib.sha256).digest()
    signature_payload: dict[str, Any] = {
        "algorithm": "HMAC-SHA256",
        "value": base64.b64encode(digest).decode("ascii"),
    }
    if signing_key_id:
        signature_payload["key_id"] = signing_key_id

    entry = dict(payload)
    entry["signature"] = signature_payload
    return entry


class _DecisionLogger:
    def __init__(
        self,
        path: str | Path,
        *,
        signing_key: bytes | None = None,
        signing_key_id: str | None = None,
    ) -> None:
        self._path = Path(path).expanduser()
        self._handle = None
        self._signing_key = signing_key
        self._signing_key_id = signing_key_id

    def __enter__(self) -> "_DecisionLogger":
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("a", encoding="utf-8")
        except OSError as exc:
            LOGGER.error("Nie udało się otworzyć decision logu %s: %s", self._path, exc)
            raise SystemExit(2) from exc
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle:
            self._handle.close()
            self._handle = None

    def _apply_signature(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return dict(
            _sign_payload(
                payload,
                signing_key=self._signing_key,
                signing_key_id=self._signing_key_id,
            )
        )

    def _write_entry(self, payload: Mapping[str, Any]) -> None:
        if not self._handle:
            raise RuntimeError("Decision logger not initialised")
        prepared = self._apply_signature(payload)
        self._handle.write(json.dumps(prepared, ensure_ascii=False) + "\n")
        self._handle.flush()

    def write_metadata(self, metadata: Mapping[str, Any]) -> None:
        entry = {
            "kind": "metadata",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": dict(metadata),
        }
        self._write_entry(entry)

    def record(self, snapshot, notes_payload: Any, *, source: str) -> None:
        extracted = _extract_snapshot_fields(snapshot, notes_payload)
        entry = {
            "kind": "snapshot",
            "timestamp": extracted["timestamp"],
            "source": source,
            "event": extracted["event"],
            "severity": extracted["severity"],
            "fps": extracted["fps"],
            "screen": extracted["screen"],
            "notes": notes_payload,
        }
        self._write_entry(entry)


def _matches_screen_filters(
    notes_payload: Any,
    *,
    screen_index: int | None,
    screen_name: str | None,
) -> bool:
    if screen_index is None and not screen_name:
        return True

    context = _screen_context(notes_payload)
    if not context:
        return False

    if screen_index is not None and context.get("index") != screen_index:
        return False

    if screen_name:
        candidate = context.get("name")
        if not isinstance(candidate, str):
            return False
        if screen_name.casefold() not in candidate.casefold():
            return False

    return True


def _matches_severity_filter(
    notes_payload: Any,
    *,
    severities: set[str] | None,
    severity_min: str | None,
) -> bool:
    if not severities and not severity_min:
        return True
    if not isinstance(notes_payload, dict):
        return False
    severity = _normalize_severity(notes_payload.get("severity"))
    if severity is None:
        return False
    if severities and severity not in severities:
        return False
    if severity_min and not _severity_at_least(severity, severity_min):
        return False
    return True


def _matches_time_filters(
    snapshot,
    *,
    since: datetime | None,
    until: datetime | None,
) -> bool:
    if since is None and until is None:
        return True
    iso_ts = _snapshot_timestamp(snapshot)
    dt_ts = _parse_iso_datetime(iso_ts)
    if dt_ts is None:
        return False
    if since and dt_ts < since:
        return False
    if until and dt_ts > until:
        return False
    return True


def _snapshot_timestamp(snapshot) -> str | None:
    if getattr(snapshot, "HasField", None) and not snapshot.HasField("generated_at"):
        return None
    generated_at = getattr(snapshot, "generated_at", None)
    if generated_at is None:
        return None
    return _timestamp_to_iso(generated_at)


def _snapshot_fps(snapshot) -> float | None:
    if getattr(snapshot, "HasField", None) and not snapshot.HasField("fps"):
        return None
    fps = getattr(snapshot, "fps", None)
    if fps is None:
        return None
    try:
        return float(fps)
    except (TypeError, ValueError):  # pragma: no cover - zabezpieczenie
        return None


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _parse_cli_datetime(
    value: str,
    *,
    parser: argparse.ArgumentParser,
    flag: str,
) -> datetime:
    candidate = value.strip()
    if not candidate:
        parser.error(f"{flag} wymaga niepustej wartości ISO 8601")
    dt = _parse_iso_datetime(candidate)
    if dt is None:
        parser.error(f"Niepoprawny format czasu dla {flag}; oczekiwano ISO 8601 (np. 2024-02-01T12:00:00Z)")
    return dt


def _decision_log_filters(
    args,
    severity_filters: set[str] | None,
    *,
    severity_min: str | None,
    since_iso: str | None,
    until_iso: str | None,
) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    if args.event:
        filters["event"] = args.event
    if severity_filters:
        filters["severity"] = sorted(severity_filters)
    if severity_min:
        filters["severity_min"] = severity_min
    if args.screen_index is not None:
        filters["screen_index"] = args.screen_index
    if args.screen_name:
        filters["screen_name"] = args.screen_name
    if args.limit is not None:
        filters["limit"] = args.limit
    if since_iso:
        filters["since"] = since_iso
    if until_iso:
        filters["until"] = until_iso
    return filters


def _decision_log_metadata_offline(
    args,
    *,
    severity_filters: set[str] | None,
    severity_min: str | None,
    summary_enabled: bool,
    summary_signature: Mapping[str, Any] | None,
    since_iso: str | None,
    until_iso: str | None,
    signing_info: Mapping[str, Any] | None,
) -> dict[str, Any]:
    input_location = "stdin" if args.from_jsonl == "-" else str(Path(args.from_jsonl).expanduser())
    metadata: dict[str, Any] = {
        "mode": "jsonl",
        "input_file": input_location,
    }
    filters = _decision_log_filters(
        args,
        severity_filters,
        severity_min=severity_min,
        since_iso=since_iso,
        until_iso=until_iso,
    )
    if filters:
        metadata["filters"] = filters
    if summary_enabled:
        metadata["summary_enabled"] = True
        if summary_signature:
            metadata["summary_signature"] = dict(summary_signature)
    if signing_info:
        metadata["signing"] = dict(signing_info)
    risk_profile = getattr(args, "_risk_profile_config", None)
    if risk_profile:
        metadata["risk_profile"] = dict(risk_profile)
        summary = getattr(args, "_risk_profile_summary", None)
        if summary:
            metadata["risk_profile_summary"] = dict(summary)
    core_config = getattr(args, "_core_config_metadata", None)
    if core_config:
        metadata["core_config"] = dict(core_config)
    return metadata


def _decision_log_metadata_grpc(
    args,
    *,
    severity_filters: set[str] | None,
    severity_min: str | None,
    summary_enabled: bool,
    summary_signature: Mapping[str, Any] | None,
    auth_token: str | None,
    tls_enabled: bool,
    since_iso: str | None,
    until_iso: str | None,
    signing_info: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "mode": "grpc",
        "endpoint": f"{args.host}:{args.port}",
        "use_tls": bool(tls_enabled),
        "auth_token_provided": bool(auth_token),
    }
    if args.timeout is not None:
        metadata["timeout"] = args.timeout
    filters = _decision_log_filters(
        args,
        severity_filters,
        severity_min=severity_min,
        since_iso=since_iso,
        until_iso=until_iso,
    )
    if filters:
        metadata["filters"] = filters
    if summary_enabled:
        metadata["summary_enabled"] = True
        if summary_signature:
            metadata["summary_signature"] = dict(summary_signature)
    if signing_info:
        metadata["signing"] = dict(signing_info)
    metadata["tls_materials"] = {
        "root_cert": bool(args.root_cert),
        "client_cert": bool(args.client_cert),
        "client_key": bool(args.client_key),
        "server_name": bool(args.server_name),
        "server_sha256": bool(args.server_sha256),
    }
    custom_metadata = getattr(args, "_custom_metadata", None) or []
    if custom_metadata:
        custom_section: dict[str, Any] = {
            "keys": [key for key, _ in custom_metadata],
            "count": len(custom_metadata),
        }
        if any(_looks_like_sensitive_metadata_key(key) for key, _ in custom_metadata):
            custom_section["contains_sensitive"] = True
        sources_map = getattr(args, "_custom_metadata_sources", None)
        if sources_map:
            custom_section["sources"] = dict(sources_map)
        metadata["custom_metadata"] = custom_section
    risk_profile = getattr(args, "_risk_profile_config", None)
    if risk_profile:
        metadata["risk_profile"] = dict(risk_profile)
        summary = getattr(args, "_risk_profile_summary", None)
        if summary:
            metadata["risk_profile_summary"] = dict(summary)
    core_config = getattr(args, "_core_config_metadata", None)
    if core_config:
        metadata["core_config"] = dict(core_config)
    return metadata


def _format_snapshot(snapshot, *, fmt: str, notes_payload: Any) -> str:
    extracted = _extract_snapshot_fields(snapshot, notes_payload)
    screen_summary = _screen_summary(extracted["screen"] or {})

    if fmt == "json":
        payload = {
            "generated_at": extracted["timestamp"],
            "fps": extracted["fps"],
            "event": extracted["event"],
            "severity": extracted["severity"],
            "notes": notes_payload,
            "screen": extracted["screen"],
        }
        return json.dumps(payload, ensure_ascii=False)

    notes_preview = notes_payload
    if isinstance(notes_preview, dict):
        preview = json.dumps(notes_preview, ensure_ascii=False)
    else:
        preview = str(notes_preview)
    if len(preview) > 160:
        preview = preview[:157] + "..."
    timestamp = extracted["timestamp"] or "-"
    fps = f"{extracted['fps']:.2f}" if extracted["fps"] is not None else "-"
    event_label = extracted["event"] or "-"
    severity_label = extracted["severity"] or "-"
    screen_label = screen_summary or "-"
    return (
        f"{timestamp} | fps={fps} | event={event_label} | severity={severity_label} | "
        f"screen={screen_label} | notes={preview}"
    )


def _iter_stream(
    stub,
    request,
    *,
    timeout: float | None,
    metadata: list[tuple[str, MetadataValue]] | None,
) -> Iterable:
    try:
        stream_kwargs = {"timeout": timeout}
        if metadata:
            stream_kwargs["metadata"] = metadata
        yield from stub.StreamMetrics(request, **stream_kwargs)
    except Exception as exc:  # pragma: no cover
        LOGGER.error("Błąd połączenia z MetricsService: %s", exc)
        raise SystemExit(1) from exc


def _read_file_bytes(path: str | None) -> bytes | None:
    if not path:
        return None
    try:
        data = Path(path).expanduser().read_bytes()
    except FileNotFoundError as exc:
        LOGGER.error("Nie znaleziono pliku TLS: %s", path)
        raise SystemExit(2) from exc
    except OSError as exc:  # pragma: no cover - zależne od platformy
        LOGGER.error("Nie udało się odczytać pliku TLS %s: %s", path, exc)
        raise SystemExit(2) from exc
    return data


def _verify_fingerprint(data: bytes | None, expected_hex: str | None) -> None:
    if not expected_hex:
        return
    if data is None:
        LOGGER.error(
            "Nie można zweryfikować odcisku SHA-256 certyfikatu serwera bez --root-cert"
        )
        raise SystemExit(2)
    digest = hashlib.sha256(data).hexdigest()
    if digest.lower() != expected_hex.lower():
        LOGGER.error(
            "Odcisk SHA-256 certyfikatu serwera nie zgadza się – oczekiwano %s, otrzymano %s",
            expected_hex,
            digest,
        )
        raise SystemExit(2)


def _parse_env_bool(
    value: str,
    *,
    variable: str,
    parser: argparse.ArgumentParser,
) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    parser.error(
        f"Nieprawidłowa wartość '{value}' w zmiennej środowiskowej {variable} – oczekiwano true/false."
    )
    raise AssertionError("parser.error nie przerwał działania")


def _apply_environment_overrides(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser,
    provided_flags: set[str],
) -> tuple[bool, bool]:
    env = os.environ
    env_use_tls_explicit = False

    if "--use-tls" not in provided_flags:
        env_value = env.get(f"{_ENV_PREFIX}USE_TLS")
        if env_value is not None:
            args.use_tls = _parse_env_bool(
                env_value,
                variable=f"{_ENV_PREFIX}USE_TLS",
                parser=parser,
            )
            env_use_tls_explicit = True

    tls_env_present = False

    def _override_simple(
        attr: str,
        suffix: str,
        flag: str,
        *,
        allow_none: bool = False,
        allow_default: bool = True,
        strip_value: bool = False,
    ) -> None:
        nonlocal tls_env_present
        if flag in provided_flags:
            return
        env_key = f"{_ENV_PREFIX}{suffix}"
        if env_key not in env:
            return
        raw_value = env[env_key]
        stripped = raw_value.strip()
        normalized = stripped.lower()
        value: Any
        if allow_none and normalized in {"", "none", "null"}:
            value = None
        elif allow_default and normalized == "default":
            value = parser.get_default(attr)
        else:
            value = stripped if strip_value else raw_value
        setattr(args, attr, value)
        if attr == "risk_profile" and value not in (None, ""):
            args._risk_profile_source = "env"
        if attr == "server_sha256" and value not in (None, ""):
            args._server_sha256_source = "env"
        if attr in {
            "root_cert",
            "client_cert",
            "client_key",
            "server_name",
            "server_sha256",
        }:
            tls_env_present = bool(value)

    def _override_numeric(
        attr: str,
        suffix: str,
        flag: str,
        cast,
        *,
        allow_none: bool = False,
    ) -> None:
        if flag in provided_flags:
            return
        env_key = f"{_ENV_PREFIX}{suffix}"
        if env_key not in env:
            return
        raw_value = env[env_key]
        stripped = raw_value.strip()
        normalized = stripped.lower()
        if allow_none and normalized in {"", "none", "null"}:
            setattr(args, attr, None)
            return
        if normalized == "default":
            setattr(args, attr, parser.get_default(attr))
            return
        try:
            setattr(args, attr, cast(stripped))
        except Exception:
            parser.error(
                f"Nieprawidłowa wartość '{raw_value}' w zmiennej {env_key} dla parametru {attr}."
            )

    def _override_list(attr: str, suffix: str, flag: str) -> None:
        if flag in provided_flags:
            return
        env_key = f"{_ENV_PREFIX}{suffix}"
        if env_key not in env:
            return
        raw_value = env[env_key]
        stripped = raw_value.strip()
        normalized = stripped.lower()
        if normalized in {"", "none", "null"}:
            setattr(args, attr, None)
            return
        if normalized == "default":
            setattr(args, attr, parser.get_default(attr))
            return
        values = [item.strip() for item in raw_value.split(",") if item.strip()]
        setattr(args, attr, values)

    def _override_headers(attr: str, suffix: str, flag: str) -> None:
        if flag in provided_flags:
            return
        env_key = f"{_ENV_PREFIX}{suffix}"
        if env_key not in env:
            return
        raw_value = env[env_key]
        stripped = raw_value.strip()
        normalized = stripped.lower()
        if normalized in {"", "none", "null"}:
            setattr(args, attr, [])
            return
        if normalized == "default":
            setattr(args, attr, parser.get_default(attr))
            return
        setattr(args, attr, _split_header_entries(raw_value))

    def _override_header_files(attr: str, suffix: str, flag: str) -> None:
        if flag in provided_flags:
            return
        env_key = f"{_ENV_PREFIX}{suffix}"
        if env_key not in env:
            return
        raw_value = env[env_key]
        stripped = raw_value.strip()
        normalized = stripped.lower()
        if normalized in {"", "none", "null"}:
            setattr(args, attr, [])
            return
        if normalized == "default":
            setattr(args, attr, parser.get_default(attr))
            return
        paths = _split_header_file_specs(raw_value)
        if not paths:
            parser.error(
                f"Zmienna {env_key} nie zawiera żadnych ścieżek nagłówków gRPC"
            )
        setattr(args, attr, paths)

    def _override_header_dirs(attr: str, suffix: str, flag: str) -> None:
        if flag in provided_flags:
            return
        env_key = f"{_ENV_PREFIX}{suffix}"
        if env_key not in env:
            return
        raw_value = env[env_key]
        stripped = raw_value.strip()
        normalized = stripped.lower()
        if normalized in {"", "none", "null"}:
            setattr(args, attr, [])
            return
        if normalized == "default":
            setattr(args, attr, parser.get_default(attr))
            return
        directories = _split_header_directory_specs(raw_value)
        if not directories:
            parser.error(
                f"Zmienna {env_key} nie zawiera żadnych katalogów nagłówków gRPC"
            )
        setattr(args, attr, directories)

    _override_simple("host", "HOST", "--host")
    _override_numeric("port", "PORT", "--port", int)
    _override_numeric("timeout", "TIMEOUT", "--timeout", float, allow_none=True)
    _override_numeric("limit", "LIMIT", "--limit", int, allow_none=True)
    _override_simple("event", "EVENT", "--event", allow_none=True, strip_value=True)
    _override_list("severity", "SEVERITY", "--severity")
    _override_simple(
        "severity_min",
        "SEVERITY_MIN",
        "--severity-min",
        allow_none=True,
        strip_value=True,
    )
    _override_simple(
        "risk_profile",
        "RISK_PROFILE",
        "--risk-profile",
        allow_none=True,
        strip_value=True,
    )
    _override_simple(
        "risk_profiles_file",
        "RISK_PROFILES_FILE",
        "--risk-profiles-file",
        allow_none=True,
    )
    _override_simple("core_config", "CORE_CONFIG", "--core-config", allow_none=True)
    _override_numeric("screen_index", "SCREEN_INDEX", "--screen-index", int, allow_none=True)
    _override_simple("screen_name", "SCREEN_NAME", "--screen-name", allow_none=True, strip_value=True)
    _override_simple("since", "SINCE", "--since", allow_none=True, strip_value=True)
    _override_simple("until", "UNTIL", "--until", allow_none=True, strip_value=True)
    _override_simple("from_jsonl", "FROM_JSONL", "--from-jsonl", allow_none=True)
    _override_header_files("headers_files", "HEADERS_FILE", "--headers-file")
    _override_header_dirs("headers_dirs", "HEADERS_DIRS", "--headers-dir")

    if "--summary" not in provided_flags and not args.summary:
        env_key = f"{_ENV_PREFIX}SUMMARY"
        if env_key in env:
            args.summary = _parse_env_bool(env[env_key], variable=env_key, parser=parser)

    if "--print-risk-profiles" not in provided_flags and not getattr(args, "print_risk_profiles", False):
        env_key = f"{_ENV_PREFIX}PRINT_RISK_PROFILES"
        if env_key in env:
            args.print_risk_profiles = _parse_env_bool(env[env_key], variable=env_key, parser=parser)

    if "--headers-report" not in provided_flags and not getattr(args, "headers_report", False):
        env_key = f"{_ENV_PREFIX}HEADERS_REPORT"
        if env_key in env:
            args.headers_report = _parse_env_bool(env[env_key], variable=env_key, parser=parser)

    if "--headers-report-only" not in provided_flags and not getattr(args, "headers_report_only", False):
        env_key = f"{_ENV_PREFIX}HEADERS_REPORT_ONLY"
        if env_key in env:
            args.headers_report_only = _parse_env_bool(env[env_key], variable=env_key, parser=parser)

    _override_simple(
        "summary_output",
        "SUMMARY_OUTPUT",
        "--summary-output",
        allow_none=True,
    )
    _override_simple("decision_log", "DECISION_LOG", "--decision-log", allow_none=True)
    _override_simple(
        "decision_log_hmac_key",
        "DECISION_LOG_HMAC_KEY",
        "--decision-log-hmac-key",
        allow_none=True,
        strip_value=True,
    )
    _override_simple(
        "decision_log_hmac_key_file",
        "DECISION_LOG_HMAC_KEY_FILE",
        "--decision-log-hmac-key-file",
        allow_none=True,
    )
    _override_simple(
        "decision_log_key_id",
        "DECISION_LOG_KEY_ID",
        "--decision-log-key-id",
        allow_none=True,
        strip_value=True,
    )
    _override_headers("headers", "HEADERS", "--header")

    if "--format" not in provided_flags:
        env_key = f"{_ENV_PREFIX}FORMAT"
        if env_key in env:
            candidate = env[env_key].strip().lower()
            if candidate not in {"table", "json"}:
                parser.error(
                    f"Nieprawidłowy format '{env[env_key]}' w zmiennej {env_key}. Dozwolone: table/json."
                )
            args.format = candidate

    _override_simple("root_cert", "ROOT_CERT", "--root-cert", allow_none=True, strip_value=True)
    _override_simple("client_cert", "CLIENT_CERT", "--client-cert", allow_none=True, strip_value=True)
    _override_simple("client_key", "CLIENT_KEY", "--client-key", allow_none=True, strip_value=True)
    _override_simple("server_name", "SERVER_NAME", "--server-name", allow_none=True, strip_value=True)
    _override_simple(
        "server_sha256",
        "SERVER_SHA256",
        "--server-sha256",
        allow_none=True,
        strip_value=True,
    )

    if "--auth-token" not in provided_flags and args.auth_token is None:
        env_key = f"{_ENV_PREFIX}AUTH_TOKEN"
        if env_key in env:
            args.auth_token = env[env_key]

    if "--auth-token-file" not in provided_flags and args.auth_token_file is None:
        env_key = f"{_ENV_PREFIX}AUTH_TOKEN_FILE"
        if env_key in env:
            args.auth_token_file = env[env_key]

    return tls_env_present, env_use_tls_explicit


def _apply_core_config_defaults(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser,
    provided_flags: set[str],
) -> None:
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
    metrics_meta["auth_token_scope_required"] = _REQUIRED_METRICS_SCOPE
    metrics_meta["auth_token_scope_checked"] = False

    rbac_tokens = tuple(getattr(metrics_config, "rbac_tokens", ()) or ())
    if rbac_tokens:
        metrics_meta["rbac_tokens"] = len(rbac_tokens)

    tls_config = getattr(metrics_config, "tls", None)
    if tls_config is not None:
        metrics_meta["tls_enabled"] = bool(getattr(tls_config, "enabled", False))
        metrics_meta["client_auth"] = bool(getattr(tls_config, "require_client_auth", False))
        metrics_meta["root_cert_configured"] = bool(getattr(tls_config, "client_ca_path", None))
        metrics_meta["client_cert_configured"] = bool(getattr(tls_config, "certificate_path", None))
        metrics_meta["client_key_configured"] = bool(getattr(tls_config, "private_key_path", None))
        pinned_fingerprints = tuple(getattr(tls_config, "pinned_fingerprints", ()) or ())
        if pinned_fingerprints:
            metrics_meta["pinned_fingerprints"] = list(pinned_fingerprints)
            selected_pin = _select_sha256_fingerprint(pinned_fingerprints)
            if selected_pin and not getattr(args, "server_sha256", None):
                args.server_sha256 = selected_pin
                args._server_sha256_source = "pinned_fingerprint"
            if selected_pin:
                metrics_meta["pinned_fingerprint_selected"] = selected_pin
            else:
                metrics_meta["pinned_fingerprint_selected"] = None

    if getattr(metrics_config, "auth_token", None):
        metrics_meta["auth_token_configured"] = True

    existing_custom_entries = list(getattr(args, "_custom_metadata", []) or [])
    existing_custom_sources = dict(getattr(args, "_custom_metadata_sources", {}) or {})
    existing_removal_keys = set(getattr(args, "_custom_metadata_remove", set()) or ())
    existing_removal_sources = dict(getattr(args, "_custom_metadata_remove_sources", {}) or {})

    config_metadata_entries = tuple(getattr(metrics_config, "grpc_metadata", ()) or ())
    metadata_sources = dict(getattr(metrics_config, "grpc_metadata_sources", {}) or {})
    config_metadata_files = tuple(getattr(metrics_config, "grpc_metadata_files", ()) or ())
    config_metadata_directories = (
        tuple(getattr(metrics_config, "grpc_metadata_directories", ()) or ())
        if hasattr(metrics_config, "grpc_metadata_directories")
        else ()
    )
    if config_metadata_files:
        metrics_meta["grpc_metadata_files"] = list(config_metadata_files)
        metrics_meta["grpc_metadata_files_count"] = len(config_metadata_files)
    if config_metadata_directories:
        metrics_meta["grpc_metadata_directories"] = list(config_metadata_directories)
        metrics_meta["grpc_metadata_directories_count"] = len(config_metadata_directories)

    headers_disabled = getattr(args, "_headers_disabled", False)

    file_metadata_entries: list[tuple[str, MetadataValue]] = []
    file_metadata_sources: dict[str, str] = {}
    file_removal_keys: set[str] = set()
    file_removal_sources: dict[str, str] = {}
    directory_metadata_entries: list[tuple[str, MetadataValue]] = []
    directory_metadata_sources: dict[str, str] = {}
    directory_removal_keys: set[str] = set()
    directory_removal_sources: dict[str, str] = {}
    directory_files_map: dict[str, list[str]] = {}

    if config_metadata_directories and not headers_disabled:
        for dir_path in config_metadata_directories:
            trimmed_dir = str(dir_path).strip()
            if not trimmed_dir:
                continue
            files, resolved_dir = _resolve_headers_directory(trimmed_dir, parser=parser)
            directory_files_map.setdefault(resolved_dir, [])
            if files:
                directory_files_map[resolved_dir].extend(files)
            else:
                continue
            for file_path in files:
                entries, resolved_path = _load_headers_file_entries(file_path, parser=parser)
                if not entries:
                    continue
                (
                    parsed_entries,
                    removal_keys,
                    parsed_sources,
                    parsed_removal_sources,
                ) = _parse_metadata_entries(
                    entries,
                    parser=parser,
                    env=os.environ,
                    base_source=f"config-dir:{resolved_path}",
                )
                if parsed_entries:
                    directory_metadata_entries.extend(parsed_entries)
                    if parsed_sources:
                        directory_metadata_sources.update(parsed_sources)
                if removal_keys:
                    directory_removal_keys.update(removal_keys)
                    if parsed_removal_sources:
                        directory_removal_sources.update(parsed_removal_sources)
                    else:
                        for key in removal_keys:
                            directory_removal_sources.setdefault(
                                key, f"config-dir:{resolved_path}"
                            )

    if config_metadata_files and not headers_disabled:
        for file_path in config_metadata_files:
            trimmed_path = str(file_path).strip()
            if not trimmed_path:
                continue
            entries, resolved_path = _load_headers_file_entries(trimmed_path, parser=parser)
            if not entries:
                continue
            (
                parsed_entries,
                removal_keys,
                parsed_sources,
                parsed_removal_sources,
            ) = _parse_metadata_entries(
                entries,
                parser=parser,
                env=os.environ,
                base_source=f"config-file:{resolved_path}",
            )
            if parsed_entries:
                file_metadata_entries.extend(parsed_entries)
                if parsed_sources:
                    file_metadata_sources.update(parsed_sources)
            if removal_keys:
                file_removal_keys.update(removal_keys)
                if parsed_removal_sources:
                    file_removal_sources.update(parsed_removal_sources)
                else:
                    for key in removal_keys:
                        file_removal_sources.setdefault(key, f"config-file:{resolved_path}")

    config_prepend: list[tuple[str, MetadataValue]] = []
    if not headers_disabled:
        if directory_metadata_entries:
            config_prepend.extend(directory_metadata_entries)
        if file_metadata_entries:
            config_prepend.extend(file_metadata_entries)
        if config_metadata_entries:
            config_prepend.extend(config_metadata_entries)

    if config_prepend:
        metrics_meta["grpc_metadata_keys"] = [key for key, _ in config_prepend]
        metrics_meta["grpc_metadata_count"] = len(config_prepend)
        metrics_meta["grpc_metadata_enabled"] = True
        combined_entries = config_prepend + existing_custom_entries
    else:
        combined_entries = existing_custom_entries
        if config_metadata_entries or config_metadata_files:
            metrics_meta["grpc_metadata_enabled"] = False

    combined_sources: dict[str, str] = {}
    if directory_metadata_sources:
        combined_sources.update(directory_metadata_sources)
    if file_metadata_sources:
        combined_sources.update(file_metadata_sources)
    if metadata_sources:
        combined_sources.update(metadata_sources)
    if existing_custom_sources:
        combined_sources.update(existing_custom_sources)

    combined_removal_keys = set(existing_removal_keys)
    combined_removal_sources = dict(existing_removal_sources)
    if directory_removal_keys and not headers_disabled:
        combined_removal_keys.update(directory_removal_keys)
        for key in directory_removal_keys:
            source = directory_removal_sources.get(key)
            if source:
                combined_removal_sources[key] = source
            else:
                combined_removal_sources.setdefault(key, "config-dir")
    if file_removal_keys and not headers_disabled:
        combined_removal_keys.update(file_removal_keys)
        for key in file_removal_keys:
            source = file_removal_sources.get(key)
            if source:
                combined_removal_sources[key] = source
            else:
                combined_removal_sources.setdefault(key, "config-file")

    args._custom_metadata = combined_entries
    args._custom_metadata_sources = combined_sources or None
    args._custom_metadata_remove = combined_removal_keys
    args._custom_metadata_remove_sources = combined_removal_sources or None

    metadata_applied = bool(config_prepend)
    if combined_sources and (metadata_applied or existing_custom_sources):
        metrics_meta["grpc_metadata_sources"] = dict(combined_sources)
    if directory_files_map:
        metrics_meta["grpc_metadata_directory_files"] = {
            directory: list(files)
            for directory, files in directory_files_map.items()
        }
        metrics_meta["grpc_metadata_directory_files_total"] = sum(
            len(files) for files in directory_files_map.values()
        )

    default_host = parser.get_default("host")
    if (
        "--host" not in provided_flags
        and getattr(args, "host", default_host) == default_host
        and getattr(metrics_config, "host", None)
    ):
        args.host = metrics_config.host

    default_port = parser.get_default("port")
    if (
        "--port" not in provided_flags
        and getattr(args, "port", default_port) == default_port
        and getattr(metrics_config, "port", None)
    ):
        args.port = metrics_config.port

    if not getattr(args, "risk_profiles_file", None) and getattr(
        metrics_config, "ui_alerts_risk_profiles_file", None
    ):
        args.risk_profiles_file = metrics_config.ui_alerts_risk_profiles_file

    if not getattr(args, "risk_profile", None) and getattr(
        metrics_config, "ui_alerts_risk_profile", None
    ):
        args.risk_profile = metrics_config.ui_alerts_risk_profile
        args._risk_profile_source = "core_config"

    tls_cfg = getattr(metrics_config, "tls", None)
    if tls_cfg is not None and getattr(tls_cfg, "enabled", False):
        if not args.use_tls:
            LOGGER.info(
                "Konfiguracja core.yaml wymaga TLS – automatycznie włączam --use-tls"
            )
            args.use_tls = True
        if not args.root_cert and getattr(tls_cfg, "client_ca_path", None):
            args.root_cert = tls_cfg.client_ca_path
        if not args.client_cert and getattr(tls_cfg, "certificate_path", None):
            args.client_cert = tls_cfg.certificate_path
        if not args.client_key and getattr(tls_cfg, "private_key_path", None):
            args.client_key = tls_cfg.private_key_path

    token_scope_match: bool | None = None
    if not getattr(args, "auth_token", None) and not getattr(args, "auth_token_file", None):
        if getattr(metrics_config, "auth_token", None):
            args.auth_token = metrics_config.auth_token
            metrics_meta["auth_token_source"] = "config"
            metrics_meta["auth_token_scope_reason"] = "legacy_token"
            metrics_meta.setdefault("auth_token_configured", True)
        elif getattr(metrics_config, "auth_token_env", None):
            env_name = str(metrics_config.auth_token_env)
            env_value = os.environ.get(env_name)
            metrics_meta["auth_token_source"] = "env"
            metrics_meta["auth_token_env"] = env_name
            if env_value:
                args.auth_token = env_value
                metrics_meta.setdefault("auth_token_configured", True)
                metrics_meta["auth_token_env_present"] = True
                metrics_meta["auth_token_scope_reason"] = "legacy_token_env"
            else:
                metrics_meta.setdefault("auth_token_configured", False)
                metrics_meta["auth_token_env_present"] = False
        elif getattr(metrics_config, "auth_token_file", None):
            file_path = Path(str(metrics_config.auth_token_file)).expanduser()
            metrics_meta["auth_token_source"] = "file"
            metrics_meta["auth_token_file"] = str(file_path)
            try:
                file_value = file_path.read_text(encoding="utf-8").strip()
            except OSError:
                file_value = ""
            if file_value:
                args.auth_token = file_value
                metrics_meta.setdefault("auth_token_configured", True)
                metrics_meta["auth_token_scope_reason"] = "legacy_token_file"
            else:
                metrics_meta.setdefault("auth_token_configured", False)
        elif (
            rbac_tokens
            and resolve_service_token is not None
        ):
            token_entry = resolve_service_token(
                rbac_tokens,
                scope=_REQUIRED_METRICS_SCOPE,
            )
            if token_entry and token_entry.secret:
                args.auth_token = token_entry.secret
                metrics_meta["auth_token_source"] = "rbac_token"
                metrics_meta.setdefault("auth_token_configured", True)
                metrics_meta["auth_token_scope_checked"] = True
                scopes = sorted(token_entry.scopes)
                if scopes:
                    metrics_meta["auth_token_scopes"] = scopes
                token_scope_match = bool(
                    not scopes or _REQUIRED_METRICS_SCOPE in scopes
                )
                metrics_meta["auth_token_scope_match"] = token_scope_match
                metrics_meta["auth_token_token_id"] = token_entry.token_id
            elif resolve_service_token_secret is not None:
                token_secret = resolve_service_token_secret(
                    rbac_tokens, scope=_REQUIRED_METRICS_SCOPE
                )
                if token_secret:
                    args.auth_token = token_secret
                    metrics_meta["auth_token_source"] = "rbac_token"
                    metrics_meta.setdefault("auth_token_configured", True)
                    metrics_meta["auth_token_scope_reason"] = "rbac_secret_only"
    if token_scope_match is False:
        metrics_meta["auth_token_scope_warning"] = "missing_required_scope"

    if args.use_tls:
        metrics_meta["use_tls"] = True
    for attr, key in (
        ("root_cert", "root_cert"),
        ("client_cert", "client_cert"),
        ("client_key", "client_key"),
        ("server_name", "server_name"),
    ):
        value = getattr(args, attr, None)
        if value:
            metrics_meta[key] = value
    server_sha = getattr(args, "server_sha256", None)
    if server_sha:
        metrics_meta["server_sha256"] = server_sha
        source = getattr(args, "_server_sha256_source", None)
        if source:
            metrics_meta["server_sha256_source"] = source

    metadata["metrics_service"] = {
        key: value for key, value in metrics_meta.items() if value not in (None, "")
    }

    # --- risk_service metadane (jeśli obecne w core.yaml) --------------------
    risk_config = getattr(core_config, "risk_service", None)
    risk_meta: dict[str, Any] = {"auth_token_scope_required": _REQUIRED_RISK_SCOPE}
    if risk_config is None:
        risk_meta["warning"] = "risk_service_missing"
    else:
        risk_meta["enabled"] = bool(getattr(risk_config, "enabled", True))
        risk_meta["host"] = getattr(risk_config, "host", None)
        risk_meta["port"] = getattr(risk_config, "port", None)
        risk_meta["history_size"] = getattr(risk_config, "history_size", None)
        risk_meta["publish_interval_seconds"] = getattr(
            risk_config, "publish_interval_seconds", None
        )
        profiles = getattr(risk_config, "profiles", None)
        if profiles:
            risk_meta["profiles"] = sorted({str(profile).strip() for profile in profiles if profile})
        if getattr(risk_config, "auth_token", None):
            risk_meta["auth_token_configured"] = True

        required_scopes: dict[str, list[str]] = {
            _REQUIRED_RISK_SCOPE: ["core_config.risk_service"]
        }

        rbac_tokens_risk = tuple(getattr(risk_config, "rbac_tokens", ()) or ())
        if rbac_tokens_risk:
            risk_meta["rbac_tokens"] = len(rbac_tokens_risk)
            if resolve_service_token is not None:
                token_entry = resolve_service_token(
                    rbac_tokens_risk,
                    scope=_REQUIRED_RISK_SCOPE,
                )
                risk_meta["auth_token_scope_checked"] = True
                if token_entry:
                    risk_meta["auth_token_scope_match"] = bool(
                        not token_entry.scopes or _REQUIRED_RISK_SCOPE in token_entry.scopes
                    )
                    risk_meta["auth_token_token_id"] = token_entry.token_id
                    if token_entry.scopes:
                        risk_meta["auth_token_scopes"] = sorted(token_entry.scopes)
                else:
                    risk_meta["auth_token_scope_match"] = False
        risk_meta["required_scopes"] = required_scopes

        tls_cfg = getattr(risk_config, "tls", None)
        if tls_cfg is not None:
            risk_meta["tls_enabled"] = bool(getattr(tls_cfg, "enabled", False))
            risk_meta["client_auth"] = bool(getattr(tls_cfg, "require_client_auth", False))
            risk_meta["root_cert_configured"] = bool(getattr(tls_cfg, "client_ca_path", None))
            risk_meta["client_cert_configured"] = bool(getattr(tls_cfg, "certificate_path", None))
            risk_meta["client_key_configured"] = bool(getattr(tls_cfg, "private_key_path", None))
            pinned = tuple(getattr(tls_cfg, "pinned_fingerprints", ()) or ())
            if pinned:
                risk_meta["pinned_fingerprints"] = list(pinned)

    metadata["risk_service"] = {
        key: value
        for key, value in risk_meta.items()
        if value not in (None, "", [], {})
    }
    args._core_config_metadata = metadata


def _load_custom_risk_profiles(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    path_value = getattr(args, "risk_profiles_file", None)
    if not path_value:
        return

    target = Path(path_value).expanduser()
    try:
        registered, _meta = load_risk_profiles_with_metadata(target, origin_label=f"watcher:{target}")
    except FileNotFoundError as exc:
        parser.error(str(exc))
    except Exception as exc:
        parser.error(f"Nie udało się wczytać profili ryzyka z {target}: {exc}")
    else:
        if registered:
            LOGGER.info(
                "Załadowano %s profil(e) ryzyka telemetrii z %s", len(registered), target
            )


def _apply_risk_profile_settings(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    profile_name = getattr(args, "risk_profile", None)
    if not profile_name:
        args._risk_profile_config = None
        args._risk_profile_base = None
        args._risk_profile_summary = None
        return

    normalized = profile_name.strip().lower()
    try:
        profile_base = get_risk_profile(normalized)
    except KeyError:
        available = list_risk_profile_names()
        parser.error(
            f"Profil ryzyka {profile_name!r} nie jest obsługiwany." + (f" Dostępne: {', '.join(available)}" if available else "")
        )

    args.risk_profile = normalized
    args._risk_profile_base = profile_base
    profile_metadata = risk_profile_metadata(normalized)
    source_label = getattr(args, "_risk_profile_source", None)
    if source_label:
        profile_metadata = dict(profile_metadata)
        profile_metadata.setdefault("source", source_label)
    args._risk_profile_config = profile_metadata
    args._risk_profile_summary = summarize_risk_profile(profile_metadata)

    severity_min = profile_base.get("severity_min")
    if severity_min and not args.severity_min:
        args.severity_min = severity_min

    if profile_base.get("expect_summary_enabled") and not (args.summary or args.summary_output):
        LOGGER.info(
            "Profil ryzyka %s wymaga aktywnego podsumowania – automatycznie włączam --summary",
            normalized,
        )
        args.summary = True


def _print_available_risk_profiles(
    selected: str | None, *, core_metadata: Mapping[str, Any] | None = None
) -> None:
    profiles: dict[str, Mapping[str, Any]] = {}
    for name in list_risk_profile_names():
        profiles[name] = risk_profile_metadata(name)

    payload: dict[str, Any] = {"risk_profiles": profiles}
    if selected:
        normalized = selected.strip().lower()
        payload["selected"] = normalized
        payload["selected_profile"] = profiles.get(normalized)
    if core_metadata:
        payload["core_config"] = dict(core_metadata)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _emit_summary(
    summary_collector: _SummaryCollector | None,
    *,
    print_to_console: bool,
    output_path: str | None,
    signing_key: bytes | None,
    signing_key_id: str | None,
    risk_profile: Mapping[str, Any] | None = None,
    risk_profile_summary: Mapping[str, Any] | None = None,
    core_metadata: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    if summary_collector is None:
        return None

    summary_payload: dict[str, Any] = {"summary": summary_collector.as_dict()}
    if risk_profile or risk_profile_summary or core_metadata:
        metadata_section = summary_payload.setdefault("metadata", {})
        if risk_profile:
            metadata_section["risk_profile"] = dict(risk_profile)
        if risk_profile_summary:
            metadata_section["risk_profile_summary"] = dict(risk_profile_summary)
        if core_metadata:
            metadata_section["core_config"] = dict(core_metadata)
            metrics_section = core_metadata.get("metrics_service")
            if isinstance(metrics_section, Mapping):
                metadata_section.setdefault(
                    "metrics_service", dict(metrics_section)
                )
            risk_section = core_metadata.get("risk_service")
            if isinstance(risk_section, Mapping):
                metadata_section.setdefault("risk_service", dict(risk_section))
    signed_payload = _sign_payload(
        summary_payload,
        signing_key=signing_key,
        signing_key_id=signing_key_id,
    )

    if print_to_console:
        print(json.dumps(signed_payload, ensure_ascii=False))

    if output_path:
        target_path = Path(output_path).expanduser()
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(
                json.dumps(signed_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            LOGGER.error("Nie udało się zapisać podsumowania do %s: %s", target_path, exc)
            raise SystemExit(2) from exc

        LOGGER.info("Zapisano podsumowanie do %s", target_path)

    return signed_payload


def _load_auth_token(token: str | None, token_file: str | None) -> str | None:
    if token and token_file:
        LOGGER.error("Użyj tylko jednej z opcji autoryzacji: --auth-token lub --auth-token-file")
        raise SystemExit(2)

    if token_file:
        try:
            data = Path(token_file).expanduser().read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            LOGGER.error("Nie znaleziono pliku z tokenem: %s", token_file)
            raise SystemExit(2) from exc
        except OSError as exc:  # pragma: no cover - zależne od platformy
            LOGGER.error("Nie udało się odczytać pliku z tokenem %s: %s", token_file, exc)
            raise SystemExit(2) from exc
        token = data.strip()
        if not token:
            LOGGER.error("Plik %s nie zawiera tokenu autoryzacyjnego", token_file)
            raise SystemExit(2)

    return token


def _load_decision_log_signing_key(
    raw_key: str | None,
    key_file: str | None,
    *,
    parser: argparse.ArgumentParser,
) -> bytes | None:
    if raw_key and key_file:
        parser.error(
            "Użyj tylko jednej z opcji podpisywania decision logu: --decision-log-hmac-key lub "
            "--decision-log-hmac-key-file."
        )

    key_material = raw_key
    if key_file:
        try:
            key_material = Path(key_file).expanduser().read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            LOGGER.error("Nie znaleziono pliku z kluczem HMAC: %s", key_file)
            raise SystemExit(2) from exc
        except OSError as exc:  # pragma: no cover - zależne od platformy
            LOGGER.error("Nie udało się odczytać klucza HMAC z %s: %s", key_file, exc)
            raise SystemExit(2) from exc

    if key_material is None:
        return None

    key_stripped = key_material.strip()
    if not key_stripped:
        parser.error("Klucz HMAC decision logu nie może być pusty")

    key_bytes = key_stripped.encode("utf-8")
    if len(key_bytes) < 16:
        LOGGER.warning(
            "Klucz HMAC dla decision logu ma mniej niż 16 bajtów – rozważ użycie dłuższego klucza."
        )
    return key_bytes


def create_metrics_channel(
    grpc_module,
    address: str,
    *,
    use_tls: bool,
    root_cert: str | None,
    client_cert: str | None,
    client_key: str | None,
    server_name: str | None,
    server_sha256: str | None,
):
    """Buduje kanał gRPC (z TLS/mTLS jeśli wymagane)."""

    if not use_tls:
        return grpc_module.insecure_channel(address)

    root_bytes = _read_file_bytes(root_cert)
    _verify_fingerprint(root_bytes, server_sha256)

    if bool(client_cert) != bool(client_key):
        LOGGER.error("mTLS wymaga jednoczesnego podania --client-cert oraz --client-key")
        raise SystemExit(2)

    certificate_chain = _read_file_bytes(client_cert)
    private_key = _read_file_bytes(client_key)

    credentials = grpc_module.ssl_channel_credentials(
        root_certificates=root_bytes,
        private_key=private_key,
        certificate_chain=certificate_chain,
    )

    options = ()
    if server_name:
        options = (("grpc.ssl_target_name_override", server_name),)

    return grpc_module.secure_channel(address, credentials, options=options)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Adres hosta MetricsService")
    parser.add_argument("--port", type=int, default=50051, help="Port MetricsService")
    parser.add_argument("--timeout", type=float, default=None, help="Timeout RPC w sekundach")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maksymalna liczba snapshotów do wypisania; domyślnie stream bez końca",
    )
    parser.add_argument("--event", default=None, help="Filtruj snapshoty po polu event w notes (np. reduce_motion)")
    parser.add_argument(
        "--severity",
        action="append",
        default=None,
        help=(
            "Filtruj snapshoty po poziomie severity (np. warning). "
            "Można podać wielokrotnie lub jako listę rozdzielaną przecinkami."
        ),
    )
    parser.add_argument(
        "--severity-min",
        default=None,
        help=(
            "Minimalny poziom severity (np. warning) – przepuszcza zdarzenia o równym lub "
            "wyższym poziomie."
        ),
    )
    parser.add_argument(
        "--risk-profile",
        default=None,
        help=(
            "Zastosuj predefiniowany profil ryzyka – ustawia domyślne progi severity i wymagania audytu."
        ),
    )
    parser.add_argument(
        "--risk-profiles-file",
        default=None,
        help=(
            "Ścieżka do pliku JSON/YAML z dodatkowymi profilami ryzyka telemetrii. "
            "Pozwala rozszerzyć lub nadpisać presety wbudowane."
        ),
    )
    parser.add_argument(
        "--core-config",
        default=None,
        help=(
            "Ścieżka do pliku core.yaml – wczyta domyślne ustawienia MetricsService (profil ryzyka, "
            "preset pliku z profilami, TLS, host/port)."
        ),
    )
    parser.add_argument("--since", default=None, help="Odfiltruj snapshoty starsze niż podany znacznik czasu ISO 8601 (UTC)")
    parser.add_argument("--until", default=None, help="Odfiltruj snapshoty nowsze niż podany znacznik czasu ISO 8601 (UTC)")
    parser.add_argument("--screen-index", type=int, default=None, help="Ogranicz snapshoty do monitora o określonym indeksie")
    parser.add_argument("--screen-name", default=None, help="Filtruj snapshoty po fragmencie nazwy monitora (case-insensitive)")
    parser.add_argument("--format", choices=("table", "json"), default="table", help="Format wypisywanych danych")
    parser.add_argument("--auth-token", default=None, help="Opcjonalny token autoryzacyjny (wysyłany w nagłówku authorization)")
    parser.add_argument("--auth-token-file", default=None, help="Ścieżka do pliku z tokenem Bearer (jedna linia). Wyklucza --auth-token")
    parser.add_argument("--use-tls", action="store_true", help="Wymusza połączenie TLS z serwerem")
    parser.add_argument("--root-cert", default=None, help="Ścieżka do zaufanego certyfikatu root CA (PEM) używanego do walidacji")
    parser.add_argument("--client-cert", default=None, help="Certyfikat klienta (PEM) dla mTLS")
    parser.add_argument("--client-key", default=None, help="Klucz prywatny klienta (PEM) dla mTLS")
    parser.add_argument("--server-name", default=None, help="Nazwa serwera TLS (override SNI) – przydatne dla IP lub testów")
    parser.add_argument("--server-sha256", default=None, help="Oczekiwany odcisk SHA-256 certyfikatu serwera (pinning)")
    parser.add_argument(
        "--from-jsonl",
        default=None,
        help=(
            "Odczytaj snapshoty z pliku JSONL (np. artefakt CI) zamiast łączyć się z serwerem. "
            "Flagi TLS i autoryzacji są ignorowane w tym trybie."
        ),
    )
    parser.add_argument(
        "--print-risk-profiles",
        action="store_true",
        help=(
            "Wypisz dostępne profile ryzyka telemetrii (wraz z progami) i zakończ działanie bez "
            "łączenia się z serwerem ani odczytu artefaktów."
        ),
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help=(
            "Wypisz podsumowanie odebranych snapshotów (łączna liczba, zdarzenia, statystyki FPS). "
            "Działa zarówno w trybie online, jak i podczas odczytu z JSONL."
        ),
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help=(
            "Zapisz podsumowanie do wskazanego pliku JSON. Włącza kolekcjonowanie statystyk "
            "również wtedy, gdy nie użyto --summary."
        ),
    )
    parser.add_argument(
        "--decision-log",
        default=None,
        help=(
            "Zapisuj każde zdarzenie telemetryczne do pliku JSONL (decision log). "
            "Działa w trybie online i podczas odczytu z JSONL; katalog zostanie utworzony automatycznie."
        ),
    )
    parser.add_argument(
        "--decision-log-hmac-key",
        default=None,
        help=(
            "Sekretny klucz HMAC (UTF-8) do podpisywania wpisów decision logu. "
            "Nie używaj razem z --decision-log-hmac-key-file."
        ),
    )
    parser.add_argument(
        "--decision-log-hmac-key-file",
        default=None,
        help=(
            "Ścieżka do pliku zawierającego klucz HMAC (UTF-8) dla decision logu. "
            "Wyklucza --decision-log-hmac-key."
        ),
    )
    parser.add_argument(
        "--decision-log-key-id",
        default=None,
        help=(
            "Opcjonalny identyfikator klucza HMAC zapisywany przy podpisach decision logu "
            "(ułatwia rotację kluczy)."
        ),
    )
    parser.add_argument(
        "--header",
        dest="headers",
        action="append",
        default=None,
        help=(
            "Dodatkowe nagłówki gRPC w formacie klucz=wartość lub klucz:wartość. "
            "Można podać wielokrotnie, np. --header x-trace=abc123. Wartości mogą "
            "odwoływać się do @env:NAZWA, @file:ścieżka, @env64:NAZWA, @file64:ścieżka "
            "lub @@prefix (dosłowny '@')."
        ),
    )
    parser.add_argument(
        "--headers-file",
        dest="headers_files",
        action="append",
        default=None,
        help=(
            "Ścieżka do pliku z dodatkowymi nagłówkami gRPC. Każda linia powinna zawierać "
            "wpis klucz=wartość (puste linie i komentarze z # są pomijane). "
            "Można podać wiele plików; wpisy z późniejszych źródeł nadpisują wcześniejsze. "
            "Wariant środowiskowy: zmienna BOT_CORE_WATCH_METRICS_HEADERS_FILE (lista ścieżek)."
        ),
    )
    parser.add_argument(
        "--headers-report",
        action="store_true",
        help=(
            "Wypisz do stdout zagregowane nagłówki gRPC (klucz, typ, źródło) po sca-"
            "leniu konfiguracji core.yaml, zmiennych środowiskowych i parametrów CLI."
        ),
    )
    parser.add_argument(
        "--headers-report-only",
        action="store_true",
        help=(
            "Jak --headers-report, ale zakończ działanie po wypisaniu raportu bez "
            "łączenia z serwerem ani odczytu artefaktów JSONL."
        ),
    )
    parser.add_argument(
        "--headers-dir",
        dest="headers_dirs",
        action="append",
        default=None,
        help=(
            "Katalog zawierający pliki z nagłówkami gRPC (format jak w --headers-file). "
            "Pliki są sortowane alfabetycznie i scalane z innymi źródłami. "
            "Wariant środowiskowy: BOT_CORE_WATCH_METRICS_HEADERS_DIRS (lista katalogów)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # źródło profilu ryzyka do metadanych
    args._risk_profile_source = None
    provided_flags = {arg for arg in (argv or []) if isinstance(arg, str) and arg.startswith("--")}
    if "--risk-profile" in provided_flags:
        args._risk_profile_source = "cli"
    if "--server-sha256" in provided_flags:
        args._server_sha256_source = "cli"

    tls_env_present, env_use_tls_explicit = _apply_environment_overrides(
        args,
        parser=parser,
        provided_flags=provided_flags,
    )
    if getattr(args, "headers_report_only", False):
        args.headers_report = True
    custom_metadata_entries: list[tuple[str, MetadataValue]] = []
    custom_metadata_sources: dict[str, str] = {}
    removal_sources: dict[str, str] = {}
    removal_keys_total: set[str] = set()

    env_headers_obj = getattr(args, "headers", None)
    env_header_files_obj = getattr(args, "headers_files", None)
    env_header_dirs_obj = getattr(args, "headers_dirs", None)
    env_headers_list = env_headers_obj if isinstance(env_headers_obj, list) else None
    env_header_files_list = (
        env_header_files_obj if isinstance(env_header_files_obj, list) else None
    )
    env_header_dirs_list = (
        env_header_dirs_obj if isinstance(env_header_dirs_obj, list) else None
    )

    env_lists = [
        candidate
        for candidate in (env_headers_list, env_header_files_list, env_header_dirs_list)
        if candidate is not None
    ]
    env_requested_disable = bool(env_lists) and all(not candidate for candidate in env_lists)

    headers_disabled = (
        env_requested_disable
        and "--header" not in provided_flags
        and "--headers-file" not in provided_flags
        and "--headers-dir" not in provided_flags
    )

    raw_headers = None if headers_disabled else env_headers_obj
    header_files_args = None if headers_disabled else env_header_files_obj
    header_dirs_args = None if headers_disabled else env_header_dirs_obj

    def _collect_metadata(entries: Sequence[str], base_source: str) -> None:
        if not entries:
            return
        (
            parsed,
            removal_keys,
            metadata_sources,
            removal_source_map,
        ) = _parse_metadata_entries(
            entries,
            parser=parser,
            env=os.environ,
            base_source=base_source,
        )
        if parsed:
            custom_metadata_entries.extend(parsed)
        if metadata_sources:
            custom_metadata_sources.update(metadata_sources)
        if removal_keys:
            removal_keys_total.update(removal_keys)
            if removal_source_map:
                removal_sources.update(removal_source_map)
            else:
                for key in removal_keys:
                    removal_sources.setdefault(key, base_source)

    if raw_headers and "--header" not in provided_flags:
        _collect_metadata(raw_headers, _ENV_HEADER_SOURCE)

    if header_dirs_args:
        dir_source_prefix = (
            _CLI_HEADER_DIR_SOURCE
            if "--headers-dir" in provided_flags
            else _ENV_HEADER_DIR_SOURCE
        )
        for directory_path in header_dirs_args:
            trimmed_dir = directory_path.strip()
            if not trimmed_dir:
                parser.error("Podano pustą ścieżkę w --headers-dir")
            files, resolved_dir = _resolve_headers_directory(trimmed_dir, parser=parser)
            if not files:
                LOGGER.warning(
                    "Katalog nagłówków gRPC %s nie zawiera żadnych plików", resolved_dir
                )
                continue
            for file_path in files:
                entries, resolved_path = _load_headers_file_entries(file_path, parser=parser)
                if entries:
                    _collect_metadata(entries, f"{dir_source_prefix}:{resolved_path}")

    if header_files_args:
        for file_path in header_files_args:
            trimmed_path = file_path.strip()
            if not trimmed_path:
                parser.error("Podano pustą ścieżkę w --headers-file")
            entries, resolved_path = _load_headers_file_entries(trimmed_path, parser=parser)
            if entries:
                _collect_metadata(entries, f"file:{resolved_path}")

    if raw_headers and "--header" in provided_flags:
        _collect_metadata(raw_headers, _CLI_HEADER_SOURCE)

    args._custom_metadata = custom_metadata_entries
    args._custom_metadata_sources = custom_metadata_sources or None
    args._custom_metadata_remove = removal_keys_total
    args._custom_metadata_remove_sources = removal_sources or None
    args._headers_disabled = headers_disabled
    if headers_disabled:
        args.headers = None
        args.headers_files = None
        args.headers_dirs = None
    _apply_core_config_defaults(args, parser=parser, provided_flags=provided_flags)
    _finalize_custom_metadata(args)
    if getattr(args, "headers_report", False):
        _print_headers_report(args)
        if getattr(args, "headers_report_only", False):
            return 0
    _load_custom_risk_profiles(args, parser)
    if (
        tls_env_present
        and not args.use_tls
        and not env_use_tls_explicit
        and "--use-tls" not in provided_flags
    ):
        LOGGER.debug("Wymuszam TLS na podstawie zmiennych środowiskowych z materiałem TLS")
        args.use_tls = True
    _apply_risk_profile_settings(args, parser)
    core_metadata = getattr(args, "_core_config_metadata", None)

    if getattr(args, "print_risk_profiles", False):
        _print_available_risk_profiles(
            getattr(args, "risk_profile", None), core_metadata=core_metadata
        )
        return 0

    if args.since is not None:
        args.since = args.since.strip() or None
    if args.until is not None:
        args.until = args.until.strip() or None
    if args.severity_min is not None:
        args.severity_min = args.severity_min.strip() or None
    if args.decision_log_hmac_key is not None:
        args.decision_log_hmac_key = args.decision_log_hmac_key.strip() or None
    if args.decision_log_hmac_key_file is not None:
        args.decision_log_hmac_key_file = args.decision_log_hmac_key_file.strip() or None
    if args.decision_log_key_id is not None:
        args.decision_log_key_id = args.decision_log_key_id.strip() or None

    since_dt = _parse_cli_datetime(args.since, parser=parser, flag="--since") if args.since else None
    until_dt = _parse_cli_datetime(args.until, parser=parser, flag="--until") if args.until else None
    if since_dt and until_dt and until_dt < since_dt:
        parser.error("--until nie może być wcześniejsze niż --since")

    since_iso = since_dt.isoformat() if since_dt else None
    until_iso = until_dt.isoformat() if until_dt else None

    if args.screen_index is not None and args.screen_index < 0:
        parser.error("--screen-index musi być liczbą nieujemną")

    severity_filters: set[str] | None = None
    if args.severity:
        severity_filters = set()
        for chunk in args.severity:
            for raw_value in chunk.split(","):
                normalized = _normalize_severity(raw_value)
                if normalized is None:
                    parser.error(
                        "Nieprawidłowa wartość severity – użyj np. warning, critical lub info."
                    )
                severity_filters.add(normalized)
        if not severity_filters:
            severity_filters = None

    severity_min: str | None = None
    if args.severity_min:
        normalized_min = _normalize_severity(args.severity_min)
        if normalized_min is None or normalized_min not in SEVERITY_RANK:
            parser.error(
                "Nieprawidłowa wartość --severity-min – użyj jednego z: "
                + ", ".join(SEVERITY_ORDER)
            )
        severity_min = normalized_min
        if severity_filters:
            conflicts = sorted(
                value for value in severity_filters if not _severity_at_least(value, severity_min)
            )
            if conflicts:
                parser.error(
                    "Parametry --severity i --severity-min są sprzeczne: "
                    f"{', '.join(conflicts)} < {severity_min}. Usuń niższe poziomy lub obniż próg."
                )

    risk_profile_base: Mapping[str, Any] | None = getattr(args, "_risk_profile_base", None)
    risk_profile_config: dict[str, Any] | None = getattr(args, "_risk_profile_config", None)
    if risk_profile_base:
        profile_min = risk_profile_base.get("severity_min")
        if profile_min:
            normalized_profile_min = _normalize_severity(profile_min)
            if normalized_profile_min is None:
                parser.error(
                    f"Profil ryzyka {args.risk_profile} ma nieprawidłowy próg severity_min: {profile_min!r}"
                )
            if severity_min is None:
                severity_min = normalized_profile_min
                args.severity_min = normalized_profile_min
            elif not _severity_at_least(severity_min, normalized_profile_min):
                parser.error(
                    f"Profil ryzyka {args.risk_profile} wymaga severity >= {normalized_profile_min};"
                    f" ustawiono {severity_min}"
                )
            if risk_profile_config is not None:
                risk_profile_config = dict(risk_profile_config)
                risk_profile_config["severity_min"] = normalized_profile_min
                args._risk_profile_config = risk_profile_config

    signing_key = _load_decision_log_signing_key(
        args.decision_log_hmac_key,
        args.decision_log_hmac_key_file,
        parser=parser,
    )
    signing_key_id = args.decision_log_key_id if signing_key else None
    if args.decision_log_key_id and not signing_key:
        LOGGER.warning(
            "Podano identyfikator klucza decision logu bez klucza HMAC – ignoruję KEY ID."
        )
    decision_log_signing_info: dict[str, Any] | None = None
    if signing_key:
        decision_log_signing_info = {"algorithm": "HMAC-SHA256"}
        if signing_key_id:
            decision_log_signing_info["key_id"] = signing_key_id

    auth_token = _load_auth_token(args.auth_token, args.auth_token_file)

    tls_args = [
        args.root_cert,
        args.client_cert,
        args.client_key,
        args.server_name,
        args.server_sha256,
    ]

    summary_enabled = bool(args.summary or args.summary_output)
    if signing_key and not args.decision_log and not summary_enabled:
        LOGGER.warning(
            "Podano klucz HMAC bez aktywnego decision logu ani podsumowania – podpisywanie nie będzie użyte."
        )
    summary_collector = _SummaryCollector() if summary_enabled else None

    summary_signature_info: dict[str, Any] | None = None
    if summary_enabled and signing_key:
        summary_signature_info = {"algorithm": "HMAC-SHA256"}
        if signing_key_id:
            summary_signature_info["key_id"] = signing_key_id

    decision_context = (
        _DecisionLogger(
            args.decision_log,
            signing_key=signing_key,
            signing_key_id=signing_key_id,
        )
        if args.decision_log
        else nullcontext()
    )

    with decision_context as decision_logger:
        if args.from_jsonl:
            if args.use_tls or any(tls_args):
                LOGGER.warning("Ignoruję ustawienia TLS w trybie odczytu z pliku JSONL")
            if auth_token:
                LOGGER.warning("Ignoruję token autoryzacyjny w trybie odczytu z pliku JSONL")

            count = 0
            source_label = "stdin" if args.from_jsonl == "-" else str(Path(args.from_jsonl).expanduser())
            if decision_logger:
                decision_logger.write_metadata(
                    _decision_log_metadata_offline(
                        args,
                        severity_filters=severity_filters,
                        severity_min=severity_min,
                        summary_enabled=summary_enabled,
                        summary_signature=summary_signature_info,
                        since_iso=since_iso,
                        until_iso=until_iso,
                        signing_info=decision_log_signing_info,
                    )
                )
            for snapshot in _iter_jsonl_snapshots(args.from_jsonl):
                if not _matches_time_filters(snapshot, since=since_dt, until=until_dt):
                    continue
                notes_payload = _parse_notes(snapshot.notes)
                if args.event and notes_payload.get("event") != args.event:
                    continue
                if not _matches_severity_filter(
                    notes_payload,
                    severities=severity_filters,
                    severity_min=severity_min,
                ):
                    continue
                if not _matches_screen_filters(
                    notes_payload,
                    screen_index=args.screen_index,
                    screen_name=args.screen_name,
                ):
                    continue
                print(
                    _format_snapshot(
                        snapshot,
                        fmt=args.format,
                        notes_payload=notes_payload,
                    )
                )
                if summary_collector:
                    summary_collector.add(snapshot, notes_payload)
                if decision_logger:
                    decision_logger.record(snapshot, notes_payload, source="jsonl")
                count += 1
                if args.limit is not None and count >= args.limit:
                    break

            if count == 0:
                LOGGER.warning("Nie znaleziono snapshotów w źródle %s spełniających filtry", source_label)
            if summary_collector:
                if args.summary and count:
                    print()
                _emit_summary(
                    summary_collector,
                    print_to_console=bool(args.summary),
                    output_path=args.summary_output,
                    signing_key=signing_key,
                    signing_key_id=signing_key_id,
                    risk_profile=risk_profile_config,
                    risk_profile_summary=getattr(args, "_risk_profile_summary", None),
                    core_metadata=core_metadata,
                )
            return 0

        if not args.use_tls and any(tls_args):
            parser.error("Flagi TLS wymagają ustawienia --use-tls")

        grpc, trading_pb2, trading_pb2_grpc = _load_grpc_components()

        address = f"{args.host}:{args.port}"
        tls_enabled = bool(args.use_tls or any(tls_args))
        channel = create_metrics_channel(
            grpc,
            address,
            use_tls=tls_enabled,
            root_cert=args.root_cert,
            client_cert=args.client_cert,
            client_key=args.client_key,
            server_name=args.server_name,
            server_sha256=args.server_sha256,
        )
        stub = trading_pb2_grpc.MetricsServiceStub(channel)

        request = trading_pb2.MetricsRequest()
        request.include_ui_metrics = True

        count = 0
        metadata: list[tuple[str, MetadataValue]] = []
        if auth_token:
            metadata.append(("authorization", f"Bearer {auth_token}"))
        custom_metadata_entries = getattr(args, "_custom_metadata", None)
        if custom_metadata_entries:
            metadata.extend(custom_metadata_entries)
        metadata_to_send = metadata or None

        if decision_logger:
            decision_logger.write_metadata(
                _decision_log_metadata_grpc(
                    args,
                    severity_filters=severity_filters,
                    severity_min=severity_min,
                    summary_enabled=summary_enabled,
                    summary_signature=summary_signature_info,
                    auth_token=auth_token,
                    tls_enabled=tls_enabled,
                    since_iso=since_iso,
                    until_iso=until_iso,
                    signing_info=decision_log_signing_info,
                )
            )

        for snapshot in _iter_stream(
            stub, request, timeout=args.timeout, metadata=metadata_to_send
        ):
            if not _matches_time_filters(snapshot, since=since_dt, until=until_dt):
                continue
            notes_payload = _parse_notes(snapshot.notes)
            if args.event and notes_payload.get("event") != args.event:
                continue
            if not _matches_severity_filter(
                notes_payload,
                severities=severity_filters,
                severity_min=severity_min,
            ):
                continue
            if not _matches_screen_filters(
                notes_payload,
                screen_index=args.screen_index,
                screen_name=args.screen_name,
            ):
                continue
            print(
                _format_snapshot(
                    snapshot,
                    fmt=args.format,
                    notes_payload=notes_payload,
                )
            )
            if summary_collector:
                summary_collector.add(snapshot, notes_payload)
            if decision_logger:
                decision_logger.record(snapshot, notes_payload, source="grpc")
            count += 1
            if args.limit is not None and count >= args.limit:
                break

        if count == 0:
            LOGGER.warning("Nie odebrano żadnych snapshotów z %s", address)
        if summary_collector:
            if args.summary and count:
                print()
            _emit_summary(
                summary_collector,
                print_to_console=bool(args.summary),
                output_path=args.summary_output,
                signing_key=signing_key,
                signing_key_id=signing_key_id,
                risk_profile=risk_profile_config,
                risk_profile_summary=getattr(args, "_risk_profile_summary", None),
                core_metadata=core_metadata,
            )
        return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
