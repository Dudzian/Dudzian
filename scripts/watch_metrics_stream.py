"""Narzędzie do podglądu strumienia MetricsService.

Skrypt łączy się z serwerem telemetrii `MetricsService` (gRPC) i wypisuje
otrzymane `MetricsSnapshot`.  Może filtrować zdarzenia po polu `event` oraz
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

Skrypt wymaga obecności wygenerowanych stubów gRPC (`trading_pb2*.py`) oraz
pakietu `grpcio`.
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import gzip
import hmac
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.telemetry_risk_profiles import (
    get_risk_profile,
    list_risk_profile_names,
    risk_profile_metadata,
)


LOGGER = logging.getLogger("bot_core.scripts.watch_metrics_stream")

_ENV_PREFIX = "BOT_CORE_WATCH_METRICS_"


def _load_grpc_components():
    try:
        import grpc  # type: ignore
    except ImportError as exc:  # pragma: no cover - środowisko bez grpcio
        raise SystemExit(
            "Pakiet grpcio jest wymagany do połączenia z MetricsService."
        ) from exc

    try:
        from bot_core.generated import trading_pb2, trading_pb2_grpc  # type: ignore
    except ImportError as exc:  # pragma: no cover - brak wygenerowanych stubów
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


def _normalize_severity(candidate: Any) -> str | None:
    if not isinstance(candidate, str):
        return None
    normalized = candidate.strip()
    if not normalized:
        return None
    return normalized.lower()


def _severity_at_least(candidate: str, minimum: str) -> bool:
    candidate_rank = _SEVERITY_RANK.get(candidate)
    minimum_rank = _SEVERITY_RANK.get(minimum)
    if candidate_rank is None or minimum_rank is None:
        return False
    return candidate_rank >= minimum_rank


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
        if isinstance(raw_fps, (int, float)):
            self.fps = float(raw_fps)
        else:
            self.fps = None

        generated_at = record.get("generated_at")
        if isinstance(generated_at, (str, Mapping)):
            self.generated_at = generated_at
        else:
            self.generated_at = None

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
        if isinstance(name, str) and name:
            parts.append(f"{label} ({name})")
        else:
            parts.append(label)
    elif isinstance(name, str) and name:
        parts.append(name)

    resolution = context.get("resolution")
    if isinstance(resolution, dict):
        width = resolution.get("width")
        height = resolution.get("height")
        if isinstance(width, int) and isinstance(height, int):
            parts.append(f"{width}x{height} px")

    refresh = context.get("refresh_hz")
    if isinstance(refresh, (int, float)) and refresh > 0:
        parts.append(f"{refresh:.0f} Hz")

    if not parts:
        return ""

    return ", ".join(parts)


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
    except (TypeError, ValueError):  # pragma: no cover - zabezpieczenie przed niestandardowym typem
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
                fps_value
                if stats["fps_min"] is None or fps_value < stats["fps_min"]
                else stats["fps_min"]
            )
            stats["fps_max"] = (
                fps_value
                if stats["fps_max"] is None or fps_value > stats["fps_max"]
                else stats["fps_max"]
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

    if screen_index is not None:
        if context.get("index") != screen_index:
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
    if args.from_jsonl == "-":
        input_location = "stdin"
    else:
        input_location = str(Path(args.from_jsonl).expanduser())
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
    risk_profile = getattr(args, "_risk_profile_config", None)
    if risk_profile:
        metadata["risk_profile"] = dict(risk_profile)
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
    metadata: list[tuple[str, str]] | None,
) -> Iterable:
    try:
        stream_kwargs = {"timeout": timeout}
        if metadata:
            stream_kwargs["metadata"] = metadata
        yield from stub.StreamMetrics(request, **stream_kwargs)
    except Exception as exc:  # pragma: no cover - obsługa błędów połączenia
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
) -> None:
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

    def _override_simple(attr: str, suffix: str, flag: str) -> None:
        nonlocal tls_env_present
        if flag in provided_flags:
            return
        env_key = f"{_ENV_PREFIX}{suffix}"
        if env_key not in env:
            return
        value = env[env_key]
        setattr(args, attr, value)
        if attr in {
            "root_cert",
            "client_cert",
            "client_key",
            "server_name",
            "server_sha256",
        }:
            tls_env_present = True

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
        if allow_none and raw_value.strip() == "":
            setattr(args, attr, None)
            return
        try:
            setattr(args, attr, cast(raw_value))
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
        values = [item.strip() for item in raw_value.split(",") if item.strip()]
        setattr(args, attr, values)

    _override_simple("host", "HOST", "--host")
    _override_numeric("port", "PORT", "--port", int)
    _override_numeric("timeout", "TIMEOUT", "--timeout", float, allow_none=True)
    _override_numeric("limit", "LIMIT", "--limit", int, allow_none=True)
    _override_simple("event", "EVENT", "--event")
    _override_list("severity", "SEVERITY", "--severity")
    _override_simple("severity_min", "SEVERITY_MIN", "--severity-min")
    _override_simple("risk_profile", "RISK_PROFILE", "--risk-profile")
    _override_numeric("screen_index", "SCREEN_INDEX", "--screen-index", int, allow_none=True)
    _override_simple("screen_name", "SCREEN_NAME", "--screen-name")
    _override_simple("since", "SINCE", "--since")
    _override_simple("until", "UNTIL", "--until")
    _override_simple("from_jsonl", "FROM_JSONL", "--from-jsonl")
    if "--summary" not in provided_flags and not args.summary:
        env_key = f"{_ENV_PREFIX}SUMMARY"
        if env_key in env:
            args.summary = _parse_env_bool(env[env_key], variable=env_key, parser=parser)

    _override_simple("summary_output", "SUMMARY_OUTPUT", "--summary-output")
    _override_simple("decision_log", "DECISION_LOG", "--decision-log")
    _override_simple(
        "decision_log_hmac_key",
        "DECISION_LOG_HMAC_KEY",
        "--decision-log-hmac-key",
    )
    _override_simple(
        "decision_log_hmac_key_file",
        "DECISION_LOG_HMAC_KEY_FILE",
        "--decision-log-hmac-key-file",
    )
    _override_simple(
        "decision_log_key_id",
        "DECISION_LOG_KEY_ID",
        "--decision-log-key-id",
    )

    if "--format" not in provided_flags:
        env_key = f"{_ENV_PREFIX}FORMAT"
        if env_key in env:
            candidate = env[env_key].strip().lower()
            if candidate not in {"table", "json"}:
                parser.error(
                    f"Nieprawidłowy format '{env[env_key]}' w zmiennej {env_key}. Dozwolone: table/json."
                )
            args.format = candidate

    _override_simple("root_cert", "ROOT_CERT", "--root-cert")
    _override_simple("client_cert", "CLIENT_CERT", "--client-cert")
    _override_simple("client_key", "CLIENT_KEY", "--client-key")
    _override_simple("server_name", "SERVER_NAME", "--server-name")
    _override_simple("server_sha256", "SERVER_SHA256", "--server-sha256")

    if "--auth-token" not in provided_flags and args.auth_token is None:
        env_key = f"{_ENV_PREFIX}AUTH_TOKEN"
        if env_key in env:
            args.auth_token = env[env_key]

    if "--auth-token-file" not in provided_flags and args.auth_token_file is None:
        env_key = f"{_ENV_PREFIX}AUTH_TOKEN_FILE"
        if env_key in env:
            args.auth_token_file = env[env_key]

    if tls_env_present and not args.use_tls and not env_use_tls_explicit and "--use-tls" not in provided_flags:
        LOGGER.debug("Wymuszam TLS na podstawie zmiennych środowiskowych z materiałem TLS")
        args.use_tls = True


def _apply_risk_profile_settings(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    profile_name = getattr(args, "risk_profile", None)
    if not profile_name:
        args._risk_profile_config = None
        args._risk_profile_base = None
        return

    normalized = profile_name.strip().lower()
    try:
        profile_base = get_risk_profile(normalized)
    except KeyError:
        parser.error(
            f"Profil ryzyka {profile_name!r} nie jest obsługiwany. Dostępne: {', '.join(list_risk_profile_names())}"
        )

    args.risk_profile = normalized
    args._risk_profile_base = profile_base
    args._risk_profile_config = risk_profile_metadata(normalized)

    severity_min = profile_base.get("severity_min")
    if severity_min and not args.severity_min:
        args.severity_min = severity_min

    if profile_base.get("expect_summary_enabled") and not (args.summary or args.summary_output):
        LOGGER.info(
            "Profil ryzyka %s wymaga aktywnego podsumowania – automatycznie włączam --summary",
            normalized,
        )
        args.summary = True

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
    parser.add_argument(
        "--event",
        default=None,
        help="Filtruj snapshoty po polu event w notes (np. reduce_motion)",
    )
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
        choices=list_risk_profile_names(),
        default=None,
        help=(
            "Zastosuj predefiniowany profil ryzyka – ustawia domyślne progi severity i wymagania audytu."
        ),
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Odfiltruj snapshoty starsze niż podany znacznik czasu ISO 8601 (UTC)",
    )
    parser.add_argument(
        "--until",
        default=None,
        help="Odfiltruj snapshoty nowsze niż podany znacznik czasu ISO 8601 (UTC)",
    )
    parser.add_argument(
        "--screen-index",
        type=int,
        default=None,
        help="Ogranicz snapshoty do monitora o określonym indeksie",
    )
    parser.add_argument(
        "--screen-name",
        default=None,
        help="Filtruj snapshoty po fragmencie nazwy monitora (case-insensitive)",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Format wypisywanych danych",
    )
    parser.add_argument(
        "--auth-token",
        default=None,
        help="Opcjonalny token autoryzacyjny (wysyłany w nagłówku authorization)",
    )
    parser.add_argument(
        "--auth-token-file",
        default=None,
        help="Ścieżka do pliku z tokenem Bearer (jedna linia). Wyklucza --auth-token",
    )
    parser.add_argument("--use-tls", action="store_true", help="Wymusza połączenie TLS z serwerem")
    parser.add_argument(
        "--root-cert",
        default=None,
        help="Ścieżka do zaufanego certyfikatu root CA (PEM) używanego do walidacji",
    )
    parser.add_argument(
        "--client-cert",
        default=None,
        help="Certyfikat klienta (PEM) dla mTLS",
    )
    parser.add_argument(
        "--client-key",
        default=None,
        help="Klucz prywatny klienta (PEM) dla mTLS",
    )
    parser.add_argument(
        "--server-name",
        default=None,
        help="Nazwa serwera TLS (override SNI) – przydatne dla IP lub testów",
    )
    parser.add_argument(
        "--server-sha256",
        default=None,
        help="Oczekiwany odcisk SHA-256 certyfikatu serwera (pinning)",
    )
    parser.add_argument(
        "--from-jsonl",
        default=None,
        help=(
            "Odczytaj snapshoty z pliku JSONL (np. artefakt CI) zamiast łączyć się z serwerem. "
            "Flagi TLS i autoryzacji są ignorowane w tym trybie."
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
    return parser


def _emit_summary(
    summary_collector: _SummaryCollector | None,
    *,
    print_to_console: bool,
    output_path: str | None,
    signing_key: bytes | None,
    signing_key_id: str | None,
    risk_profile: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    if summary_collector is None:
        return None

    summary_payload: dict[str, Any] = {"summary": summary_collector.as_dict()}
    if risk_profile:
        summary_payload.setdefault("metadata", {})["risk_profile"] = dict(risk_profile)
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


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    provided_flags = {
        arg.split("=", 1)[0]
        for arg in (argv or [])
        if arg.startswith("--")
    }
    _apply_environment_overrides(args, parser=parser, provided_flags=provided_flags)
    _apply_risk_profile_settings(args, parser)

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
        if normalized_min is None or normalized_min not in _SEVERITY_RANK:
            parser.error(
                "Nieprawidłowa wartość --severity-min – użyj jednego z: "
                + ", ".join(_SEVERITY_ORDER)
            )
        severity_min = normalized_min
        if severity_filters:
            conflicts = sorted(
                value
                for value in severity_filters
                if not _severity_at_least(value, severity_min)
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
            source_label = (
                "stdin" if args.from_jsonl == "-" else str(Path(args.from_jsonl).expanduser())
            )
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
                LOGGER.warning(
                    "Nie znaleziono snapshotów w źródle %s spełniających filtry", source_label
                )
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
        metadata: list[tuple[str, str]] | None = None
        if auth_token:
            metadata = [("authorization", f"Bearer {auth_token}")]

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

        for snapshot in _iter_stream(stub, request, timeout=args.timeout, metadata=metadata):
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
            )
        return 0


if __name__ == "__main__":  # pragma: no cover - punkt wejścia
    sys.exit(main())
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
