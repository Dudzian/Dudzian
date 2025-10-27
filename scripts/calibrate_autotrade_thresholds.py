from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Mapping

import yaml


_STREAM_READ_SIZE = 65536
_DEFAULT_GROUP_SAMPLE_LIMIT = 50000
_DEFAULT_GLOBAL_SAMPLE_LIMIT = 50000
_JSONL_SUFFIXES = frozenset({".jsonl", ".ndjson"})

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.ai.config_loader import load_risk_thresholds


@dataclass
class RiskScoreSources:
    from_files: float | None = None
    from_inline: float | None = None


def _normalize_metric_key(key: str) -> str:
    return key.strip().casefold().replace("-", "_")


_FREEZE_STATUS_PREFIXES = ("risk_freeze", "auto_risk_freeze")
_FREEZE_STATUS_EXTRAS = {"risk_unfreeze", "auto_risk_unfreeze"}
_SUPPORTED_THRESHOLD_METRICS = frozenset(
    _normalize_metric_key(name)
    for name in ("signal_after_adjustment", "signal_after_clamp", "risk_score")
)
_ABSOLUTE_THRESHOLD_METRICS = frozenset(
    _normalize_metric_key(name)
    for name in ("signal_after_adjustment", "signal_after_clamp")
)
_SUPPORTED_THRESHOLD_CANONICAL = {
    _normalize_metric_key(name): name
    for name in ("signal_after_adjustment", "signal_after_clamp", "risk_score")
}

_METRIC_VALUE_DOMAINS: dict[str, tuple[float | None, float | None]] = {
    "signal_after_adjustment": (-1.0, 1.0),
    "signal_after_clamp": (-1.0, 1.0),
    "risk_score": (0.0, 1.0),
    "risk_freeze_duration": (0.0, None),
}
_THRESHOLD_VALUE_KEYS = (
    "current_threshold",
    "threshold",
    "value",
    "current",
    "limit",
    "upper",
    "max",
)

_AMBIGUOUS_SYMBOL_MAPPING: tuple[str, str] = ("__ambiguous__", "__ambiguous__")
_UNKNOWN_IDENTIFIERS: frozenset[str] = frozenset(
    {
        "unknown",
        "__unknown__",
        "n/a",
        "na",
        "none",
        "null",
        "unassigned",
    }
)


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_cli_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = _parse_datetime(value)
    if not parsed:
        raise SystemExit(f"Nie udało się sparsować daty: {value}")
    return parsed


def _normalize_freeze_status(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower()
    if not candidate:
        return None
    if candidate in _FREEZE_STATUS_EXTRAS:
        return candidate
    if any(candidate.startswith(prefix) for prefix in _FREEZE_STATUS_PREFIXES):
        return candidate
    return None


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_threshold_value(candidate: object) -> float | None:
    numeric = _coerce_float(candidate)
    if numeric is not None:
        return numeric
    if isinstance(candidate, Mapping):
        for mapping in _iter_mappings(candidate):
            for key in _THRESHOLD_VALUE_KEYS:
                numeric = _coerce_float(mapping.get(key))
                if numeric is not None:
                    return numeric
    return None


def _resolve_metric_threshold(mapping: Mapping[str, object], metric_name: str) -> float | None:
    direct = _extract_threshold_value(mapping.get(metric_name))
    if direct is not None:
        return direct

    normalized_metric = _normalize_metric_key(metric_name)
    metric_candidate = _normalize_string(mapping.get("metric"))
    if metric_candidate and _normalize_metric_key(metric_candidate) == normalized_metric:
        for key in _THRESHOLD_VALUE_KEYS:
            numeric = _extract_threshold_value(mapping.get(key))
            if numeric is not None:
                return numeric

    name_candidate = _normalize_string(mapping.get("name"))
    if name_candidate and _normalize_metric_key(name_candidate) == normalized_metric:
        for key in _THRESHOLD_VALUE_KEYS:
            numeric = _extract_threshold_value(mapping.get(key))
            if numeric is not None:
                return numeric

    for key, value in mapping.items():
        if not isinstance(key, str):
            continue
        normalized_key = _normalize_metric_key(key)
        if normalized_metric not in normalized_key:
            continue
        if not any(token in normalized_key for token in ("threshold", "limit", "value", "current")):
            continue
        numeric = _extract_threshold_value(value)
        if numeric is not None:
            return numeric

    return None


def _iter_mappings(payload: object) -> Iterable[Mapping[str, object]]:
    if isinstance(payload, Mapping):
        yield payload
        for child in payload.values():
            yield from _iter_mappings(child)
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            yield from _iter_mappings(item)


def _normalize_string(value: object) -> str | None:
    if isinstance(value, str):
        candidate = value.strip()
        return candidate or None
    if value is None or isinstance(value, Mapping):
        return None
    candidate = str(value).strip()
    return candidate or None


def _canonicalize_symbol_key(value: object) -> str | None:
    symbol = _normalize_string(value)
    if not symbol:
        return None
    canonical = symbol.casefold()
    if not canonical:
        canonical = symbol.upper()
    return canonical or None


def _canonicalize_identifier(value: str | None) -> str:
    normalized = _normalize_string(value)
    if not normalized:
        return "unknown"
    canonical = normalized.casefold()
    if canonical in _UNKNOWN_IDENTIFIERS:
        return "unknown"
    return canonical or "unknown"


def _lookup_first_str(payload: Mapping[str, object], keys: Iterable[str]) -> str | None:
    for mapping in _iter_mappings(payload):
        for key in keys:
            value = mapping.get(key)
            if isinstance(value, str):
                normalized = value.strip()
                if normalized:
                    return normalized
    return None


def _lookup_first_float(payload: Mapping[str, object], keys: Iterable[str]) -> float | None:
    for mapping in _iter_mappings(payload):
        for key in keys:
            if key not in mapping:
                continue
            value = _coerce_float(mapping.get(key))
            if value is not None:
                return value
    return None


def _iter_freeze_events(payload: Mapping[str, object]) -> Iterable[dict[str, object]]:
    for mapping in _iter_mappings(payload):
        status_value = mapping.get("status")
        if status_value is None:
            status_value = mapping.get("event")
        normalized_status = _normalize_freeze_status(status_value)
        if not normalized_status:
            continue
        reason = _lookup_first_str(
            mapping,
            (
                "reason",
                "source_reason",
                "freeze_reason",
                "status_reason",
            ),
        ) or "unknown"
        duration = _lookup_first_float(
            mapping,
            (
                "frozen_for",
                "freeze_seconds",
                "freeze_duration",
                "duration",
                "risk_freeze_seconds",
                "expires_in",
                "remaining_seconds",
            ),
        )
        risk_score = _lookup_first_float(mapping, ("risk_score", "score"))
        yield {
            "status": normalized_status,
            "type": "auto" if normalized_status.startswith("auto_") else "manual",
            "reason": reason,
            "duration": duration,
            "risk_score": risk_score,
        }


def _parse_percentiles(raw: str | None) -> list[float]:
    if not raw:
        return [0.5, 0.75, 0.9, 0.95, 0.99]
    percentiles: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as exc:  # noqa: BLE001 - CLI feedback
            raise SystemExit(f"Niepoprawna wartość percentyla: {token}") from exc
        if value <= 0.0 or value >= 1.0:
            raise SystemExit("Percentyle muszą znajdować się w przedziale (0, 1)")
        percentiles.append(value)
    if not percentiles:
        raise SystemExit("Lista percentyli nie może być pusta")
    return sorted(set(percentiles))


def _parse_threshold_mapping(raw: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for token in raw.split(","):
        candidate = token.strip()
        if not candidate:
            continue
        if "=" not in candidate:
            raise SystemExit(
                "Niepoprawny format progu – użyj klucza i wartości w formacie metric=value"
            )
        key, value = candidate.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit("Klucz progu nie może być pusty")
        numeric = _coerce_float(value)
        if numeric is None:
            raise SystemExit(f"Nie udało się zinterpretować progu '{value}' dla metryki {key}")
        normalized_key = _normalize_metric_key(key)
        result[normalized_key] = float(numeric)
    return result


def _load_threshold_payload(path: Path) -> object:
    if path.is_dir():
        raise SystemExit(f"Ścieżka z progami musi wskazywać plik: {path}")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - zależy od środowiska
            raise SystemExit(
                "Obsługa plików YAML wymaga zainstalowania biblioteki PyYAML"
            ) from exc
        return yaml.safe_load(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Nie udało się wczytać pliku z progami: {path}") from exc


def _ensure_finite_threshold(value: float, *, metric: str, source: str) -> float:
    if not math.isfinite(value):
        raise SystemExit(
            f"Metryka {metric} z {source} posiada niefinityczną wartość: {value}"
        )
    return value


def _load_current_signal_thresholds(
    sources: Iterable[str] | None,
) -> tuple[dict[str, float], RiskScoreSources, dict[str, object]]:
    thresholds: dict[str, float] = {}
    risk_score_sources = RiskScoreSources()
    metadata_files: list[str] = []
    metadata_inline: dict[str, float] = {}
    if not sources:
        return thresholds, risk_score_sources, {}

    for source in sources:
        if not source:
            continue
        candidate = source.strip()
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            metadata_files.append(str(path))
            payload = _load_threshold_payload(path)
            if not isinstance(payload, (Mapping, list, tuple)):
                raise SystemExit(
                    "Plik z progami musi zawierać strukturę słownikową lub listę słowników"
                )
            for mapping in _iter_mappings(payload):
                for metric_name in _SUPPORTED_THRESHOLD_METRICS:
                    value = _resolve_metric_threshold(mapping, metric_name)
                    if value is not None:
                        numeric = _ensure_finite_threshold(
                            float(value),
                            metric=metric_name,
                            source=str(path),
                        )
                        if metric_name == "risk_score":
                            risk_score_sources.from_files = numeric
                        else:
                            thresholds[metric_name] = numeric
            continue
        if "=" not in candidate:
            raise SystemExit(f"Ścieżka z progami nie istnieje: {path}")
        mapping = _parse_threshold_mapping(candidate)
        for metric_name, numeric in mapping.items():
            if not isinstance(metric_name, str):
                continue
            metric_name_normalized = _normalize_metric_key(metric_name)
            if metric_name_normalized in _SUPPORTED_THRESHOLD_METRICS:
                finite_value = _ensure_finite_threshold(
                    float(numeric),
                    metric=metric_name_normalized,
                    source="parametru CLI",
                )
                metadata_inline[metric_name_normalized] = finite_value
                if metric_name_normalized == "risk_score":
                    risk_score_sources.from_inline = finite_value
                else:
                    thresholds[metric_name_normalized] = finite_value

    metadata: dict[str, object] = {}
    if metadata_files:
        metadata["files"] = metadata_files
    if metadata_inline:
        metadata["inline"] = metadata_inline

    return thresholds, risk_score_sources, metadata


def _normalize_risk_threshold_paths(sources: Iterable[str] | None) -> list[Path]:
    paths: list[Path] = []
    if not sources:
        return paths

    for source in sources:
        if not source:
            continue
        candidate = Path(source).expanduser()
        if not candidate.exists():
            raise SystemExit(f"Ścieżka z progami ryzyka nie istnieje: {candidate}")
        if candidate.is_dir():
            raise SystemExit(f"Ścieżka z progami ryzyka musi wskazywać plik: {candidate}")
        paths.append(candidate)

    return paths


def _extract_risk_score_threshold(thresholds: Mapping[str, object]) -> float | None:
    auto_trader_cfg = thresholds.get("auto_trader")
    if not isinstance(auto_trader_cfg, Mapping):
        return None
    map_cfg = auto_trader_cfg.get("map_regime_to_signal")
    if not isinstance(map_cfg, Mapping):
        return None
    value = map_cfg.get("risk_score")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _iter_paths(raw_paths: Iterable[str]) -> Iterable[Path]:
    for raw in raw_paths:
        candidate = Path(raw).expanduser()
        if not candidate.exists():
            raise SystemExit(f"Ścieżka nie istnieje: {candidate}")
        if candidate.is_dir():
            yield from sorted(path for path in candidate.glob("*.jsonl"))
        else:
            yield candidate


def _load_journal_events(
    paths: Iterable[Path],
    *,
    since: datetime | None = None,
    until: datetime | None = None,
) -> Iterable[dict[str, object]]:
    def _iterator() -> Iterable[dict[str, object]]:
        for path in paths:
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError as exc:  # noqa: BLE001 - CLI feedback
                            raise SystemExit(
                                f"Nie udało się sparsować JSON w dzienniku {path}: {exc}"
                            ) from exc
                        if isinstance(payload, Mapping):
                            timestamp = _parse_datetime(payload.get("timestamp"))
                            if since and timestamp and timestamp < since:
                                continue
                            if until and timestamp and timestamp > until:
                                continue
                            yield dict(payload)
            except OSError as exc:  # noqa: BLE001 - CLI feedback
                raise SystemExit(f"Nie udało się odczytać dziennika {path}: {exc}") from exc

    return _iterator()


def _extract_entry_timestamp(entry: Mapping[str, object]) -> datetime | None:
    direct = _parse_datetime(entry.get("timestamp"))
    if direct:
        return direct
    decision = entry.get("decision")
    if isinstance(decision, Mapping):
        ts = _parse_datetime(decision.get("timestamp"))
        if ts:
            return ts
    detail = entry.get("detail")
    if isinstance(detail, Mapping):
        ts = _parse_datetime(detail.get("timestamp"))
        if ts:
            return ts
    return None


class _JSONStream:
    __slots__ = ("_handle", "_buffer", "_pos", "_decoder")

    def __init__(self, handle) -> None:  # type: ignore[no-untyped-def]
        self._handle = handle
        self._buffer = ""
        self._pos = 0
        self._decoder = json.JSONDecoder()

    def _refill(self) -> bool:
        chunk = self._handle.read(_STREAM_READ_SIZE)
        if chunk == "":
            return False
        if self._pos > 0:
            self._buffer = self._buffer[self._pos :] + chunk
            self._pos = 0
        else:
            self._buffer += chunk
        return True

    def _ensure(self) -> bool:
        if self._pos < len(self._buffer):
            return True
        return self._refill()

    def compact(self) -> None:
        if self._pos > 0:
            self._buffer = self._buffer[self._pos :]
            self._pos = 0

    def skip_whitespace(self) -> None:
        while True:
            while self._pos < len(self._buffer) and self._buffer[self._pos].isspace():
                self._pos += 1
            if self._pos < len(self._buffer):
                return
            if not self._refill():
                return

    def peek_char(self) -> str | None:
        while True:
            if self._pos < len(self._buffer):
                return self._buffer[self._pos]
            if not self._refill():
                return None

    def consume_char(self, expected: str | None = None) -> str:
        if not self._ensure():
            raise SystemExit("Nieoczekiwany koniec danych podczas parsowania JSON")
        char = self._buffer[self._pos]
        if expected is not None and char != expected:
            raise SystemExit(
                f"Niepoprawny JSON: oczekiwano znaku '{expected}', otrzymano '{char}'"
            )
        self._pos += 1
        return char

    def decode_value(self) -> object:
        while True:
            try:
                value, new_pos = self._decoder.raw_decode(self._buffer, self._pos)
            except json.JSONDecodeError:
                if not self._refill():
                    raise SystemExit("Nieoczekiwany koniec danych podczas parsowania JSON")
                continue
            self._pos = new_pos
            return value

def _should_use_jsonl_parser(path: Path) -> bool:
    suffixes = {suffix.lower() for suffix in path.suffixes}
    if suffixes & _JSONL_SUFFIXES:
        return True

    try:
        with path.open("r", encoding="utf-8") as handle:
            first_line: str | None = None
            second_line: str | None = None
            while True:
                raw_line = handle.readline()
                if raw_line == "":
                    break
                stripped = raw_line.strip()
                if not stripped:
                    continue
                stripped = stripped.lstrip("\ufeff")
                if not stripped:
                    continue
                first_line = stripped
                break
            if first_line is None:
                return False
            while True:
                raw_line = handle.readline()
                if raw_line == "":
                    break
                stripped = raw_line.strip()
                if not stripped:
                    continue
                second_line = stripped
                break
            if second_line is None:
                return False
    except OSError:
        return False

    if first_line.startswith("{") and first_line.endswith("}") and not first_line.endswith("},"):
        return True
    return False


def _stream_autotrade_jsonl(
    path: Path,
    *,
    since: datetime | None,
    until: datetime | None,
) -> Iterable[dict[str, object]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                stripped = stripped.lstrip("\ufeff")
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:  # noqa: BLE001 - CLI feedback
                    raise SystemExit(
                        f"Niepoprawny JSON w eksporcie autotradera {path} (linia {line_number}): {exc}"
                    ) from exc
                if not isinstance(payload, Mapping):
                    continue
                timestamp = _extract_entry_timestamp(payload)
                if since and timestamp and timestamp < since:
                    continue
                if until and timestamp and timestamp > until:
                    continue
                yield dict(payload)
    except OSError as exc:  # noqa: BLE001 - CLI feedback
        raise SystemExit(f"Nie udało się odczytać eksportu {path}: {exc}") from exc


def _load_autotrade_entries(
    paths: Iterable[str],
    *,
    since: datetime | None = None,
    until: datetime | None = None,
) -> Iterable[dict[str, object]]:
    def _iterator() -> Iterable[dict[str, object]]:
        for raw in paths:
            path = Path(raw).expanduser()
            if not path.exists():
                raise SystemExit(f"Eksport autotradera nie istnieje: {path}")
            if _should_use_jsonl_parser(path):
                yield from _stream_autotrade_jsonl(path, since=since, until=until)
                continue
            try:
                with path.open("r", encoding="utf-8") as handle:
                    stream = _JSONStream(handle)
                    stream.skip_whitespace()
                    first = stream.peek_char()
                    if first == "\ufeff":
                        stream.consume_char()
                        stream.compact()
                        stream.skip_whitespace()
                        first = stream.peek_char()
                    if first is None:
                        continue
                    if first == "[":
                        stream.consume_char("[")
                        stream.compact()
                        yield from _stream_autotrade_array(stream, since=since, until=until, enforce_eof=True)
                    elif first == "{":
                        stream.consume_char("{")
                        stream.compact()
                        yield from _stream_autotrade_object(stream, since=since, until=until)
                    else:
                        raise SystemExit(
                            f"Nieobsługiwany format JSON w eksporcie autotradera {path}"
                        )
            except OSError as exc:
                raise SystemExit(f"Nie udało się odczytać eksportu {path}: {exc}") from exc
            except SystemExit:
                raise
            except Exception as exc:  # noqa: BLE001 - CLI feedback
                raise SystemExit(
                    f"Niepoprawny JSON w eksporcie autotradera {path}: {exc}"
                ) from exc

    return _iterator()


def _stream_autotrade_array(
    stream: _JSONStream,
    *,
    since: datetime | None,
    until: datetime | None,
    enforce_eof: bool = False,
) -> Iterable[dict[str, object]]:
    first_value = True
    while True:
        stream.skip_whitespace()
        char = stream.peek_char()
        if char is None:
            raise SystemExit("Niepoprawny JSON: nieoczekiwane zakończenie tablicy")
        if char == "]":
            stream.consume_char("]")
            stream.compact()
            break
        if not first_value:
            if char != ",":
                raise SystemExit("Niepoprawny JSON: oczekiwano ',' w tablicy")
            stream.consume_char(",")
            stream.compact()
            stream.skip_whitespace()
        entry = stream.decode_value()
        stream.compact()
        if isinstance(entry, Mapping):
            timestamp = _extract_entry_timestamp(entry)
            if since and timestamp and timestamp < since:
                first_value = False
                continue
            if until and timestamp and timestamp > until:
                first_value = False
                continue
            yield dict(entry)
        first_value = False
    if enforce_eof:
        stream.skip_whitespace()
        if stream.peek_char() is not None:
            stream.skip_whitespace()
            if stream.peek_char() is not None:
                raise SystemExit("Niepoprawny JSON: dodatkowe dane po tablicy")


def _stream_autotrade_object(
    stream: _JSONStream,
    *,
    since: datetime | None,
    until: datetime | None,
) -> Iterable[dict[str, object]]:
    first_pair = True
    while True:
        stream.skip_whitespace()
        char = stream.peek_char()
        if char is None:
            raise SystemExit("Niepoprawny JSON: nieoczekiwane zakończenie obiektu")
        if char == "}":
            stream.consume_char("}")
            stream.compact()
            break
        if not first_pair:
            if char != ",":
                raise SystemExit("Niepoprawny JSON: oczekiwano ',' w obiekcie")
            stream.consume_char(",")
            stream.compact()
            stream.skip_whitespace()
        key = stream.decode_value()
        stream.compact()
        if not isinstance(key, str):
            raise SystemExit("Niepoprawny JSON: klucz obiektu musi być napisem")
        stream.skip_whitespace()
        stream.consume_char(":")
        stream.compact()
        stream.skip_whitespace()
        if key == "entries":
            char = stream.peek_char()
            if char != "[":
                raise SystemExit("Niepoprawny JSON: pole 'entries' musi być tablicą")
            stream.consume_char("[")
            stream.compact()
            yield from _stream_autotrade_array(stream, since=since, until=until)
        else:
            _ = stream.decode_value()
            stream.compact()
        first_pair = False
    stream.skip_whitespace()
    if stream.peek_char() is not None:
        stream.skip_whitespace()
        if stream.peek_char() is not None:
            raise SystemExit("Niepoprawny JSON: dodatkowe dane po obiekcie")


def _resolve_key(exchange: str | None, strategy: str | None) -> tuple[str, str]:
    normalized_exchange = _canonicalize_identifier(exchange)
    normalized_strategy = _canonicalize_identifier(strategy)
    return normalized_exchange, normalized_strategy


def _is_unknown_token(value: str) -> bool:
    normalized = _normalize_string(value)
    if not normalized:
        return True
    return normalized.casefold() in _UNKNOWN_IDENTIFIERS


def _key_completeness(key: tuple[str, str]) -> int:
    exchange, strategy = key
    score = 0
    if not _is_unknown_token(exchange):
        score += 1
    if not _is_unknown_token(strategy):
        score += 1
    return score


def _has_conflict(existing: tuple[str, str], candidate: tuple[str, str]) -> bool:
    for current, new in zip(existing, candidate):
        if _is_unknown_token(current) or _is_unknown_token(new):
            continue
        if current != new:
            return True
    return False


def _update_symbol_map_entry(
    symbol_map: dict[str, tuple[str, str]],
    event: Mapping[str, object],
) -> None:
    symbol = _canonicalize_symbol_key(event.get("symbol"))
    if not symbol:
        return
    exchange = _normalize_string(event.get("primary_exchange"))
    strategy = _normalize_string(event.get("strategy"))
    key = _resolve_key(exchange, strategy)
    existing = symbol_map.get(symbol)
    if existing is None:
        symbol_map[symbol] = key
        return
    if existing == _AMBIGUOUS_SYMBOL_MAPPING:
        return
    if existing == key:
        return
    if _has_conflict(existing, key):
        symbol_map[symbol] = _AMBIGUOUS_SYMBOL_MAPPING
        return
    merged_exchange = existing[0]
    merged_strategy = existing[1]
    candidate_exchange, candidate_strategy = key
    if not _is_unknown_token(candidate_exchange):
        merged_exchange = candidate_exchange
    if not _is_unknown_token(candidate_strategy):
        merged_strategy = candidate_strategy
    merged = (merged_exchange, merged_strategy)
    if merged == existing:
        return
    if _has_conflict(existing, merged):
        symbol_map[symbol] = _AMBIGUOUS_SYMBOL_MAPPING
        return
    symbol_map[symbol] = merged


def _build_symbol_map(events: Iterable[Mapping[str, object]]) -> dict[str, tuple[str, str]]:
    symbol_map: dict[str, tuple[str, str]] = {}
    for event in events:
        _update_symbol_map_entry(symbol_map, event)
    return symbol_map


def _extract_summary(entry: Mapping[str, object]) -> Mapping[str, object] | None:
    decision = entry.get("decision")
    if isinstance(decision, Mapping):
        details = decision.get("details")
        if isinstance(details, Mapping):
            summary = details.get("summary")
            if isinstance(summary, Mapping):
                return summary
        summary = decision.get("summary")
        if isinstance(summary, Mapping):
            return summary
    detail = entry.get("detail")
    if isinstance(detail, Mapping):
        summary = detail.get("summary")
        if isinstance(summary, Mapping):
            return summary
    summary = entry.get("regime_summary")
    if isinstance(summary, Mapping):
        return summary
    return None


def _extract_symbol(entry: Mapping[str, object]) -> str | None:
    decision = entry.get("decision")
    if isinstance(decision, Mapping):
        details = decision.get("details")
        if isinstance(details, Mapping):
            symbol = details.get("symbol")
            if isinstance(symbol, str) and symbol:
                return symbol
        symbol = decision.get("symbol")
        if isinstance(symbol, str) and symbol:
            return symbol
    detail = entry.get("detail")
    if isinstance(detail, Mapping):
        symbol = detail.get("symbol")
        if isinstance(symbol, str) and symbol:
            return symbol
    symbol = entry.get("symbol")
    if isinstance(symbol, str) and symbol:
        return symbol
    return None


def _resolve_group_from_symbol(
    entry: Mapping[str, object],
    symbol: str | None,
    summary: Mapping[str, object] | None,
    symbol_map: Mapping[str, tuple[str, str]],
) -> tuple[tuple[str, str], tuple[str, str]]:
    def _update_from_value(
        value: object,
        current: tuple[str | None, str | None],
        seen: set[int],
    ) -> tuple[str | None, str | None]:
        exchange, strategy = current
        if exchange is not None and strategy is not None:
            return exchange, strategy
        if isinstance(value, Mapping):
            return _update_from_mapping(value, (exchange, strategy), seen)
        if isinstance(value, (list, tuple)):
            for item in value:
                exchange, strategy = _update_from_value(item, (exchange, strategy), seen)
                if exchange is not None and strategy is not None:
                    break
        return exchange, strategy

    def _update_from_mapping(
        mapping: object,
        current: tuple[str | None, str | None],
        seen: set[int],
    ) -> tuple[str | None, str | None]:
        exchange, strategy = current
        if not isinstance(mapping, Mapping):
            return exchange, strategy
        mapping_id = id(mapping)
        if mapping_id in seen:
            return exchange, strategy
        seen.add(mapping_id)
        if exchange is None:
            exchange = _normalize_string(mapping.get("primary_exchange"))
        if strategy is None:
            strategy = _normalize_string(mapping.get("strategy"))
        if exchange is not None and strategy is not None:
            return exchange, strategy
        for nested_key in ("routing", "route"):
            nested = mapping.get(nested_key)
            exchange, strategy = _update_from_value(nested, (exchange, strategy), seen)
            if exchange is not None and strategy is not None:
                return exchange, strategy
        for candidate in mapping.values():
            exchange, strategy = _update_from_value(candidate, (exchange, strategy), seen)
            if exchange is not None and strategy is not None:
                break
        return exchange, strategy

    detail_payload: Mapping[str, object] | None = None
    decision_payload: Mapping[str, object] | None = None
    decision_details: Mapping[str, object] | None = None
    if isinstance(entry, Mapping):
        raw_detail = entry.get("detail")
        if isinstance(raw_detail, Mapping):
            detail_payload = raw_detail
        raw_decision = entry.get("decision")
        if isinstance(raw_decision, Mapping):
            decision_payload = raw_decision
            raw_details = raw_decision.get("details")
            if isinstance(raw_details, Mapping):
                decision_details = raw_details

    exchange: str | None = None
    strategy: str | None = None
    seen_mappings: set[int] = set()
    for candidate in (
        entry,
        detail_payload,
        decision_details,
        decision_payload,
        summary,
    ):
        exchange, strategy = _update_from_mapping(candidate, (exchange, strategy), seen_mappings)
        if exchange is not None and strategy is not None:
            break

    canonical_symbol = _canonicalize_symbol_key(symbol) if symbol is not None else None
    if canonical_symbol and (exchange is None or strategy is None):
        mapped = symbol_map.get(canonical_symbol)
        if mapped and mapped != _AMBIGUOUS_SYMBOL_MAPPING:
            mapped_exchange, mapped_strategy = mapped
            if exchange is None:
                candidate = _normalize_string(mapped_exchange)
                if candidate is not None:
                    exchange = candidate
            if strategy is None:
                candidate = _normalize_string(mapped_strategy)
                if candidate is not None:
                    strategy = candidate

    key = _resolve_key(exchange, strategy)
    display = (
        exchange if exchange is not None else key[0],
        strategy if strategy is not None else key[1],
    )
    return key, display


def _compute_percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    weight = position - lower
    return lower_value + (upper_value - lower_value) * weight


def _finite_values(values: Iterable[object]) -> list[float]:
    finite: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            finite.append(numeric)
    return finite


@dataclass(slots=True)
class MetricStatisticsResult:
    payload: dict[str, object]
    sorted_samples: list[float]
    sorted_abs_samples: list[float] | None = None
    absolute_percentiles: dict[str, object] | None = None
    approximation_mode: str | None = None


def _merge_sorted_sequences(left: list[float], right: list[float]) -> list[float]:
    result: list[float] = []
    i = 0
    j = 0
    left_len = len(left)
    right_len = len(right)
    while i < left_len and j < right_len:
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    if i < left_len:
        result.extend(left[i:])
    if j < right_len:
        result.extend(right[j:])
    return result


def _sorted_absolute_values(sorted_values: list[float]) -> list[float]:
    if not sorted_values:
        return []
    negatives = [abs(value) for value in sorted_values if value < 0]
    negatives.reverse()
    positives = [value for value in sorted_values if value >= 0]
    return _merge_sorted_sequences(negatives, positives)


class StreamingMetricAggregator:
    __slots__ = (
        "_count",
        "_min",
        "_max",
        "_sum",
        "_sum_sq",
        "_samples",
        "_sample_limit",
        "_rng",
        "_domain_min",
        "_domain_max",
        "_min_abs",
        "_clamp_count",
        "_clamp_absolute_count",
    )

    def __init__(
        self,
        *,
        sample_limit: int | None = None,
        rng: random.Random | None = None,
        domain: tuple[float | None, float | None] | None = None,
    ) -> None:
        self._count = 0
        self._min: float | None = None
        self._max: float | None = None
        self._sum = 0.0
        self._sum_sq = 0.0
        self._sample_limit = sample_limit if sample_limit is None or sample_limit >= 0 else None
        self._samples: list[float] = []
        self._rng = rng or random.Random(0)
        if domain:
            self._domain_min, self._domain_max = domain
        else:
            self._domain_min = None
            self._domain_max = None
        self._min_abs: float | None = None
        self._clamp_count = 0
        self._clamp_absolute_count = 0

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            return
        self._count += 1
        self._sum += value
        self._sum_sq += value * value
        if self._min is None or value < self._min:
            self._min = value
        if self._max is None or value > self._max:
            self._max = value
        abs_value = abs(value)
        if self._min_abs is None or abs_value < self._min_abs:
            self._min_abs = abs_value
        limit = self._sample_limit
        if limit is None or limit < 0:
            self._samples.append(value)
            return
        if limit == 0:
            return
        if len(self._samples) < limit:
            self._samples.append(value)
            return
        index = self._rng.randrange(self._count)
        if index < limit:
            self._samples[index] = value

    def extend(self, values: Iterable[float]) -> None:
        for value in values:
            self.add(value)

    @property
    def count(self) -> int:
        return self._count

    @property
    def sample_size(self) -> int:
        return len(self._samples)

    @property
    def sample_limit(self) -> int | None:
        return self._sample_limit

    @property
    def sample_truncated(self) -> bool:
        limit = self._sample_limit
        if limit is None or limit < 0:
            return False
        if limit == 0:
            return self._count > 0
        return self._count > limit

    @property
    def omitted_samples(self) -> int:
        limit = self._sample_limit
        if limit is None or limit < 0:
            return 0
        retained = len(self._samples) if limit != 0 else 0
        return max(self._count - retained, 0)

    @property
    def clamp_regular_count(self) -> int:
        return self._clamp_count

    @property
    def clamp_absolute_count(self) -> int:
        return self._clamp_absolute_count

    def _sorted_samples(self) -> list[float]:
        if not self._samples:
            return []
        return sorted(self._samples)

    def _mean_stddev(self) -> tuple[float, float]:
        if self._count == 0:
            return (0.0, 0.0)
        mean = self._sum / self._count
        variance = (self._sum_sq / self._count) - (mean * mean)
        if variance < 0:
            variance = 0.0
        stddev = math.sqrt(variance) if self._count > 1 else 0.0
        return (mean, stddev)

    def _value_bounds(self, *, absolute: bool = False) -> tuple[float | None, float | None]:
        domain_min = self._domain_min
        domain_max = self._domain_max
        if not absolute:
            lower = self._min
            upper = self._max
            if domain_min is not None:
                lower = domain_min if lower is None else max(lower, domain_min)
            if domain_max is not None:
                upper = domain_max if upper is None else min(upper, domain_max)
            return (lower, upper)

        lower: float | None = None
        upper: float | None = None
        abs_candidates: list[float] = []
        if self._min_abs is not None:
            abs_candidates.append(self._min_abs)
        if self._min is not None:
            abs_candidates.append(abs(self._min))
        if self._max is not None:
            abs_candidates.append(abs(self._max))
        if abs_candidates:
            lower = min(abs_candidates)
            upper = max(abs_candidates)
        else:
            lower = 0.0
            upper = None

        lower = 0.0 if lower is None else max(lower, 0.0)

        if domain_min is not None:
            domain_min_abs = domain_min if domain_min >= 0 else 0.0
            lower = max(lower, domain_min_abs)
        if domain_max is not None:
            domain_max_abs = abs(domain_max)
            upper = domain_max_abs if upper is None else min(upper, domain_max_abs)

        return (lower, upper)

    def _clamp(
        self,
        value: float,
        bounds: tuple[float | None, float | None],
        *,
        absolute: bool = False,
    ) -> float:
        lower, upper = bounds
        clamped = value
        clamped_flag = False
        if lower is not None and clamped < lower:
            clamped = lower
            clamped_flag = True
        if upper is not None and clamped > upper:
            clamped = upper
            clamped_flag = True
        if clamped_flag:
            if absolute:
                self._clamp_absolute_count += 1
            else:
                self._clamp_count += 1
        return clamped

    def statistics(
        self, percentiles: Iterable[float], *, absolute: bool = False
    ) -> MetricStatisticsResult:
        if self._count == 0:
            payload = {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "stddev": None,
                "percentiles": {
                    _format_percentile_label(p): None for p in percentiles
                },
                "sample_truncated": False,
                "retained_samples": 0,
                "omitted_samples": 0,
                "approximation_mode": None,
                "absolute_percentiles": {
                    _format_percentile_label(p): None for p in percentiles
                }
                if absolute
                else None,
            }
            return MetricStatisticsResult(
                payload=payload,
                sorted_samples=[],
                sorted_abs_samples=[] if absolute else None,
                absolute_percentiles=payload.get("absolute_percentiles")
                if absolute
                else None,
                approximation_mode=None,
            )
        mean, stddev = self._mean_stddev()
        samples = self._sorted_samples()
        percentiles_payload = {}
        approximation_mode: str | None = None
        absolute_percentiles_payload: dict[str, float | None] | None = None
        sorted_abs_samples: list[float] | None = None
        if samples:
            percentiles_payload = {
                _format_percentile_label(p): _compute_percentile(samples, p)
                for p in percentiles
            }
            if absolute:
                sorted_abs_samples = _sorted_absolute_values(samples)
                absolute_percentiles_payload = {
                    _format_percentile_label(p): _compute_percentile(
                        sorted_abs_samples, p
                    )
                    for p in percentiles
                }
        elif self._sample_limit == 0:
            percentiles_payload = _approximate_percentiles_from_moments(
                mean, stddev, percentiles
            )
            approximation_mode = "approximate_from_moments"
            if absolute:
                absolute_percentiles_payload = _approximate_folded_percentiles_from_moments(
                    mean, stddev, percentiles
                )
        else:
            percentiles_payload = {
                _format_percentile_label(p): None for p in percentiles
            }
            if absolute:
                absolute_percentiles_payload = {
                    _format_percentile_label(p): None for p in percentiles
                }
        bounds = self._value_bounds()
        for key, value in list(percentiles_payload.items()):
            if value is None:
                continue
            percentiles_payload[key] = self._clamp(value, bounds, absolute=False)
        if absolute and absolute_percentiles_payload is not None:
            abs_bounds = self._value_bounds(absolute=True)
            for key, value in list(absolute_percentiles_payload.items()):
                if value is None:
                    continue
                absolute_percentiles_payload[key] = self._clamp(
                    value, abs_bounds, absolute=True
                )
        payload = {
            "count": self._count,
            "min": self._min,
            "max": self._max,
            "mean": mean,
            "stddev": stddev,
            "percentiles": percentiles_payload,
            "sample_truncated": self.sample_truncated,
            "retained_samples": len(samples),
            "omitted_samples": self.omitted_samples,
            "approximation_mode": approximation_mode,
            "absolute_percentiles": absolute_percentiles_payload
            if absolute
            else None,
        }
        return MetricStatisticsResult(
            payload=payload,
            sorted_samples=samples,
            sorted_abs_samples=sorted_abs_samples,
            absolute_percentiles=absolute_percentiles_payload,
            approximation_mode=approximation_mode,
        )

    def suggest_threshold(
        self,
        percentile: float,
        *,
        absolute: bool = False,
        statistics_result: MetricStatisticsResult | None = None,
    ) -> float | None:
        percentiles_payload: Mapping[str, object] | None = None
        absolute_percentiles_payload: Mapping[str, object] | None = None
        working_abs: list[float] | None = None
        label = _format_percentile_label(percentile)

        if statistics_result is not None:
            samples = statistics_result.sorted_samples
            percentiles_payload = statistics_result.payload.get("percentiles")
            working_abs = statistics_result.sorted_abs_samples
            absolute_percentiles_payload = statistics_result.absolute_percentiles
            if absolute_percentiles_payload is None:
                raw_absolute = statistics_result.payload.get("absolute_percentiles")
                if isinstance(raw_absolute, Mapping):
                    absolute_percentiles_payload = raw_absolute
        else:
            samples = self._sorted_samples()

        cached_candidate: float | None = None
        if percentiles_payload is not None:
            value = percentiles_payload.get(label)
            if isinstance(value, (int, float)):
                cached_candidate = float(value)

        if (
            cached_candidate is not None
            and not absolute
            and statistics_result is not None
        ):
            return cached_candidate

        if absolute and absolute_percentiles_payload is not None:
            abs_candidate = absolute_percentiles_payload.get(label)
            if isinstance(abs_candidate, (int, float)):
                return float(abs_candidate)

        if samples:
            if absolute:
                if working_abs is None:
                    working_abs = _sorted_absolute_values(samples)
                    if statistics_result is not None:
                        statistics_result.sorted_abs_samples = working_abs
                working = working_abs
            else:
                working = samples
            if working:
                value = _compute_percentile(working, percentile)
                return self._clamp(
                    value,
                    self._value_bounds(absolute=absolute),
                    absolute=absolute,
                )

        if self._count == 0:
            return None

        if percentiles_payload is not None:
            if (
                absolute
                and self._sample_limit == 0
                and (
                    statistics_result is None
                    or statistics_result.payload.get("retained_samples", 0) == 0
                )
            ):
                mean, stddev = self._mean_stddev()
                value = _approximate_folded_normal_percentile(mean, stddev, percentile)
                return self._clamp(
                    value,
                    self._value_bounds(absolute=True),
                    absolute=True,
                )

            if cached_candidate is not None and not absolute:
                return cached_candidate

        mean, stddev = self._mean_stddev()
        if self._sample_limit == 0:
            if absolute:
                value = _approximate_folded_normal_percentile(mean, stddev, percentile)
            else:
                value = _approximate_normal_percentile(mean, stddev, percentile)
            return self._clamp(
                value,
                self._value_bounds(absolute=absolute),
                absolute=absolute,
            )
        return None


def _format_percentile_label(percentile: float) -> str:
    decimal_value = Decimal(str(percentile)) * Decimal(100)
    normalized = decimal_value.normalize()
    text = format(normalized, "f")
    if "." in text:
        integer, fractional = text.split(".", 1)
        fractional = fractional.rstrip("0")
        if fractional:
            return f"p{integer.zfill(2)}_{fractional}"
        return f"p{integer.zfill(2)}"
    return f"p{text.zfill(2)}"


def _approximate_normal_percentile(mean: float, stddev: float, percentile: float) -> float:
    if stddev <= 0 or math.isclose(stddev, 0.0):
        return mean
    dist = statistics.NormalDist(mu=mean, sigma=stddev)
    return dist.inv_cdf(percentile)


def _approximate_folded_normal_percentile(
    mean: float, stddev: float, percentile: float
) -> float:
    percentile = max(0.0, min(1.0, percentile))
    if percentile <= 0.0:
        return 0.0
    if percentile >= 1.0:
        return max(abs(mean), stddev * 10.0)
    if stddev <= 0 or math.isclose(stddev, 0.0):
        return abs(mean)
    dist = statistics.NormalDist(mu=mean, sigma=stddev)

    def _folded_cdf(x: float) -> float:
        if x <= 0:
            return 0.0
        return dist.cdf(x) - dist.cdf(-x)

    high = max(abs(mean), stddev)
    if high <= 0.0:
        high = 1.0
    target = percentile
    # Expand search window until coverage >= target
    for _ in range(64):
        if _folded_cdf(high) >= target:
            break
        high *= 2.0
    low = 0.0
    for _ in range(64):
        mid = (low + high) / 2.0
        if mid <= 0.0 and high <= 0.0:
            break
        if _folded_cdf(mid) < target:
            low = mid
        else:
            high = mid
    return high


def _approximate_percentiles_from_moments(
    mean: float, stddev: float, percentiles: Iterable[float]
) -> dict[str, float]:
    return {
        _format_percentile_label(p): _approximate_normal_percentile(mean, stddev, p)
        for p in percentiles
    }


def _approximate_folded_percentiles_from_moments(
    mean: float, stddev: float, percentiles: Iterable[float]
) -> dict[str, float]:
    return {
        _format_percentile_label(p): _approximate_folded_normal_percentile(mean, stddev, p)
        for p in percentiles
    }


def _metric_statistics(values: list[float], percentiles: Iterable[float]) -> dict[str, object]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "stddev": None,
            "percentiles": {},
        }
    sorted_values = sorted(values)
    count = len(sorted_values)
    mean = statistics.fmean(sorted_values)
    stddev = statistics.pstdev(sorted_values) if count > 1 else 0.0
    percentiles_payload = {
        _format_percentile_label(p): _compute_percentile(sorted_values, p)
        for p in percentiles
    }
    return {
        "count": count,
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "mean": mean,
        "stddev": stddev,
        "percentiles": percentiles_payload,
    }


def _suggest_threshold(values: list[float], percentile: float, *, absolute: bool = False) -> float | None:
    if not values:
        return None
    target_values = [abs(value) for value in values] if absolute else list(values)
    target_values.sort()
    return _compute_percentile(target_values, percentile)


def _build_metrics_section(
    values_map: Mapping[str, object],
    percentiles: Iterable[float],
    suggestion_percentile: float,
    *,
    current_risk_score: float | None,
    current_signal_thresholds: Mapping[str, float] | None = None,
) -> dict[str, dict[str, object]]:
    metrics_payload: dict[str, dict[str, object]] = {}
    for metric_name, values in values_map.items():
        absolute = metric_name in _ABSOLUTE_THRESHOLD_METRICS
        if isinstance(values, StreamingMetricAggregator):
            stats_result = values.statistics(percentiles, absolute=absolute)
            stats_payload = stats_result.payload
            suggested = values.suggest_threshold(
                suggestion_percentile,
                absolute=absolute,
                statistics_result=stats_result,
            )
            stats_payload["clamped_values"] = {
                "regular": int(values.clamp_regular_count),
                "absolute": int(values.clamp_absolute_count),
            }
        else:
            finite_values = _finite_values(values)
            if isinstance(values, list) and len(finite_values) != len(values):
                values[:] = finite_values
            stats_payload = _metric_statistics(finite_values, percentiles)
            suggested = _suggest_threshold(
                finite_values,
                suggestion_percentile,
                absolute=absolute,
            )
            stats_payload["clamped_values"] = {"regular": 0, "absolute": 0}
            stats_payload["approximation_mode"] = None
            if absolute:
                abs_values = sorted(abs(value) for value in finite_values)
                stats_payload["absolute_percentiles"] = {
                    _format_percentile_label(p): _compute_percentile(abs_values, p)
                    if abs_values
                    else None
                    for p in percentiles
                }
            else:
                stats_payload["absolute_percentiles"] = None
        if metric_name == "risk_score":
            current = current_risk_score
        elif current_signal_thresholds:
            current = current_signal_thresholds.get(metric_name)
        else:
            current = None
        stats_payload["suggested_threshold"] = suggested
        stats_payload["current_threshold"] = current
        if isinstance(values, StreamingMetricAggregator) and not absolute:
            if "absolute_percentiles" not in stats_payload:
                stats_payload["absolute_percentiles"] = None
        metrics_payload[metric_name] = stats_payload
    return metrics_payload


def _format_freeze_summary(summary: Mapping[str, object]) -> dict[str, object]:
    type_counts = summary.get("type_counts")
    status_counts = summary.get("status_counts")
    reason_counts = summary.get("reason_counts")
    total = int(summary.get("total") or 0)
    omitted_total = summary.get("omitted_total")
    omitted = int(omitted_total) if omitted_total is not None else 0
    auto_count = 0
    manual_count = 0
    if isinstance(type_counts, Counter):
        auto_count = int(type_counts.get("auto", 0))
        manual_count = int(type_counts.get("manual", 0))
    payload = {
        "total": total,
        "auto": auto_count,
        "manual": manual_count,
        "statuses": [
            {"status": status, "count": int(count)}
            for status, count in sorted(
                (status_counts.items() if isinstance(status_counts, Counter) else []),
                key=lambda item: (-item[1], item[0]),
            )
        ],
        "reasons": [
            {"reason": reason, "count": int(count)}
            for reason, count in sorted(
                (reason_counts.items() if isinstance(reason_counts, Counter) else []),
                key=lambda item: (-item[1], item[0]),
            )
        ],
    }
    payload["omitted"] = omitted
    return payload


def _write_json(report: Mapping[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_threshold_config(report: Mapping[str, object]) -> dict[str, object]:
    auto_trader_config: dict[str, object] = {}

    strategy_thresholds: dict[str, dict[str, dict[str, float]]] = {}
    groups_payload = report.get("groups")
    if isinstance(groups_payload, Iterable):
        for entry in groups_payload:
            if not isinstance(entry, Mapping):
                continue
            exchange = _normalize_string(entry.get("primary_exchange")) or "unknown"
            strategy = _normalize_string(entry.get("strategy")) or "unknown"
            metrics = entry.get("metrics")
            if not isinstance(metrics, Mapping):
                continue
            suggestions: dict[str, float] = {}
            for metric_name, payload in metrics.items():
                if not isinstance(metric_name, str) or not isinstance(payload, Mapping):
                    continue
                normalized_metric = _normalize_metric_key(metric_name)
                if normalized_metric not in _SUPPORTED_THRESHOLD_METRICS:
                    continue
                canonical_name = _SUPPORTED_THRESHOLD_CANONICAL.get(
                    normalized_metric, normalized_metric
                )
                suggestion = payload.get("suggested_threshold")
                if not isinstance(suggestion, (int, float)):
                    continue
                numeric = float(suggestion)
                if not math.isfinite(numeric):
                    continue
                suggestions[canonical_name] = numeric
            if not suggestions:
                continue
            exchange_map = strategy_thresholds.setdefault(exchange, {})
            metrics_map = exchange_map.get(strategy, {})
            metrics_map.update(suggestions)
            exchange_map[strategy] = metrics_map

    if strategy_thresholds:
        ordered_strategy_thresholds: dict[str, dict[str, dict[str, float]]] = {}
        for exchange in sorted(strategy_thresholds):
            strategy_map = strategy_thresholds[exchange]
            ordered_strategy_thresholds[exchange] = {
                strategy: {
                    metric: strategy_map[strategy][metric]
                    for metric in sorted(strategy_map[strategy])
                }
                for strategy in sorted(strategy_map)
            }
        auto_trader_config["strategy_signal_thresholds"] = ordered_strategy_thresholds

    global_signal_thresholds: dict[str, float] = {}
    map_regime_thresholds: dict[str, float] = {}
    global_summary = report.get("global_summary")
    if isinstance(global_summary, Mapping):
        metrics_payload = global_summary.get("metrics")
        if isinstance(metrics_payload, Mapping):
            for metric_name, payload in metrics_payload.items():
                if not isinstance(metric_name, str) or not isinstance(payload, Mapping):
                    continue
                normalized_metric = _normalize_metric_key(metric_name)
                if normalized_metric not in _SUPPORTED_THRESHOLD_METRICS:
                    continue
                canonical_name = _SUPPORTED_THRESHOLD_CANONICAL.get(
                    normalized_metric, normalized_metric
                )
                suggestion = payload.get("suggested_threshold")
                if not isinstance(suggestion, (int, float)):
                    continue
                numeric = float(suggestion)
                if not math.isfinite(numeric):
                    continue
                if normalized_metric == _normalize_metric_key("risk_score"):
                    map_regime_thresholds[canonical_name] = numeric
                else:
                    global_signal_thresholds[canonical_name] = numeric

    if global_signal_thresholds:
        auto_trader_config["signal_thresholds"] = {
            key: global_signal_thresholds[key] for key in sorted(global_signal_thresholds)
        }

    if map_regime_thresholds:
        auto_trader_config.setdefault("map_regime_to_signal", {}).update(
            {key: map_regime_thresholds[key] for key in sorted(map_regime_thresholds)}
        )

    if not auto_trader_config:
        return {}

    return {"auto_trader": auto_trader_config}


def _write_threshold_config(config: Mapping[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        with destination.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    else:
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")


def _write_csv(
    groups: Iterable[dict[str, object]],
    destination: Path,
    *,
    percentiles: Iterable[str],
    global_summary: Mapping[str, object] | None = None,
) -> None:
    import csv

    destination.parent.mkdir(parents=True, exist_ok=True)
    percentile_labels = list(percentiles)
    abs_percentile_columns = [f"abs_{label}" for label in percentile_labels]

    freeze_columns = [
        "freeze_total",
        "freeze_auto",
        "freeze_manual",
        "freeze_omitted",
        "freeze_status_counts",
        "freeze_reason_counts",
        "freeze_truncated",
    ]
    raw_columns = [
        "raw_values_truncated",
        "raw_values_omitted",
    ]
    approximation_columns = ["approximation_mode"]
    sample_columns = [
        "sample_truncated",
        "retained_samples",
        "omitted_samples",
    ]
    clamp_columns = [
        "clamp_regular",
        "clamp_absolute",
    ]
    fieldnames = [
        "primary_exchange",
        "strategy",
        "metric",
        "count",
        "min",
        "max",
        "mean",
        "stddev",
        "suggested_threshold",
        "current_threshold",
    ] + (
        approximation_columns
        + sample_columns
        + raw_columns
        + clamp_columns
        + percentile_labels
        + abs_percentile_columns
        + freeze_columns
    )
    approximation_defaults = {column: "" for column in approximation_columns}
    sample_defaults = {column: "" for column in sample_columns}
    raw_defaults = {column: "" for column in raw_columns}
    clamp_defaults = {column: "" for column in clamp_columns}
    abs_defaults = {column: "" for column in abs_percentile_columns}
    freeze_defaults = {column: "" for column in freeze_columns}

    global_raw_values_omitted: Counter[str] = Counter()
    global_raw_truncated_metrics: set[str] = set()
    def _freeze_row_payload(
        *,
        primary_exchange: str,
        strategy: str,
        summary: Mapping[str, object] | None,
        truncated: bool | None,
    ) -> dict[str, object] | None:
        if summary is None:
            return None
        total = summary.get("total")
        auto = summary.get("auto")
        manual = summary.get("manual")
        omitted = summary.get("omitted")
        statuses = summary.get("statuses")
        reasons = summary.get("reasons")
        row: dict[str, object] = {
            "primary_exchange": primary_exchange,
            "strategy": strategy,
            "metric": "__freeze_summary__",
            "count": total,
            "min": "",
            "max": "",
            "mean": "",
            "stddev": "",
            "suggested_threshold": "",
            "current_threshold": "",
        }
        row.update(approximation_defaults)
        row.update(sample_defaults)
        row.update(raw_defaults)
        row.update(clamp_defaults)
        row.update(abs_defaults)
        row.update(freeze_defaults)
        if total is not None:
            row["freeze_total"] = total
        if auto is not None:
            row["freeze_auto"] = auto
        if manual is not None:
            row["freeze_manual"] = manual
        if omitted is not None:
            row["freeze_omitted"] = omitted
        if statuses:
            row["freeze_status_counts"] = json.dumps(statuses, ensure_ascii=False)
        if reasons:
            row["freeze_reason_counts"] = json.dumps(reasons, ensure_ascii=False)
        if truncated is not None:
            row["freeze_truncated"] = str(bool(truncated)).lower()
        else:
            row["freeze_truncated"] = ""
        return row

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            primary_exchange = group["primary_exchange"]
            strategy = group["strategy"]
            metrics = group["metrics"]
            freeze_summary = group.get("freeze_summary") if isinstance(group.get("freeze_summary"), Mapping) else None
            freeze_truncated = group.get("raw_freeze_events_truncated")
            raw_values_truncated = bool(group.get("raw_values_truncated"))
            raw_values_omitted_map = group.get("raw_values_omitted")
            if not isinstance(raw_values_omitted_map, Mapping):
                raw_values_omitted_map = {}
            for metric_name, metric_payload in metrics.items():
                row = {
                    "primary_exchange": primary_exchange,
                    "strategy": strategy,
                    "metric": metric_name,
                    "count": metric_payload["count"],
                    "min": metric_payload["min"],
                    "max": metric_payload["max"],
                    "mean": metric_payload["mean"],
                    "stddev": metric_payload["stddev"],
                    "suggested_threshold": metric_payload.get("suggested_threshold"),
                    "current_threshold": metric_payload.get("current_threshold"),
                }
                row.update(approximation_defaults)
                row.update(sample_defaults)
                row.update(raw_defaults)
                row.update(clamp_defaults)
                row.update(abs_defaults)
                for percentile_key, percentile_value in metric_payload.get("percentiles", {}).items():
                    row[percentile_key] = percentile_value
                absolute_percentiles = metric_payload.get("absolute_percentiles")
                if isinstance(absolute_percentiles, Mapping):
                    for percentile_key, percentile_value in absolute_percentiles.items():
                        row[f"abs_{percentile_key}"] = percentile_value
                approximation_mode = metric_payload.get("approximation_mode")
                if approximation_mode is not None:
                    row["approximation_mode"] = approximation_mode
                sample_truncated = metric_payload.get("sample_truncated")
                if sample_truncated is not None:
                    row["sample_truncated"] = str(bool(sample_truncated)).lower()
                retained = metric_payload.get("retained_samples")
                if retained is not None:
                    row["retained_samples"] = retained
                omitted = metric_payload.get("omitted_samples")
                if omitted is not None:
                    row["omitted_samples"] = omitted
                row["raw_values_truncated"] = str(raw_values_truncated).lower()
                raw_omitted_for_metric = int(raw_values_omitted_map.get(metric_name, 0))
                row["raw_values_omitted"] = raw_omitted_for_metric
                if raw_omitted_for_metric:
                    global_raw_values_omitted[metric_name] += raw_omitted_for_metric
                    global_raw_truncated_metrics.add(metric_name)
                clamped_payload = metric_payload.get("clamped_values")
                if isinstance(clamped_payload, Mapping):
                    regular = clamped_payload.get("regular")
                    absolute = clamped_payload.get("absolute")
                    if regular is not None:
                        row["clamp_regular"] = regular
                    if absolute is not None:
                        row["clamp_absolute"] = absolute
                row.update(freeze_defaults)
                writer.writerow(row)
            freeze_row = _freeze_row_payload(
                primary_exchange=primary_exchange,
                strategy=strategy,
                summary=freeze_summary,
                truncated=bool(freeze_truncated) if freeze_truncated is not None else None,
            )
            if freeze_row is not None:
                writer.writerow(freeze_row)
        if global_summary:
            metrics = global_summary.get("metrics")
            if isinstance(metrics, Mapping):
                for metric_name, metric_payload in metrics.items():
                    row = {
                        "primary_exchange": "__all__",
                        "strategy": "__all__",
                        "metric": metric_name,
                        "count": metric_payload.get("count"),
                        "min": metric_payload.get("min"),
                        "max": metric_payload.get("max"),
                        "mean": metric_payload.get("mean"),
                        "stddev": metric_payload.get("stddev"),
                        "suggested_threshold": metric_payload.get("suggested_threshold"),
                        "current_threshold": metric_payload.get("current_threshold"),
                    }
                    row.update(approximation_defaults)
                    row.update(sample_defaults)
                    row.update(raw_defaults)
                    row.update(clamp_defaults)
                    row.update(abs_defaults)
                    percentiles_payload = metric_payload.get("percentiles")
                    if isinstance(percentiles_payload, Mapping):
                        for percentile_key, percentile_value in percentiles_payload.items():
                            row[percentile_key] = percentile_value
                    absolute_percentiles = metric_payload.get("absolute_percentiles")
                    if isinstance(absolute_percentiles, Mapping):
                        for percentile_key, percentile_value in absolute_percentiles.items():
                            row[f"abs_{percentile_key}"] = percentile_value
                    approximation_mode = metric_payload.get("approximation_mode")
                    if approximation_mode is not None:
                        row["approximation_mode"] = approximation_mode
                    sample_truncated = metric_payload.get("sample_truncated")
                    if sample_truncated is not None:
                        row["sample_truncated"] = str(bool(sample_truncated)).lower()
                    retained = metric_payload.get("retained_samples")
                    if retained is not None:
                        row["retained_samples"] = retained
                    omitted = metric_payload.get("omitted_samples")
                    if omitted is not None:
                        row["omitted_samples"] = omitted
                    global_raw_omitted = int(global_raw_values_omitted.get(metric_name, 0))
                    row["raw_values_truncated"] = str(
                        metric_name in global_raw_truncated_metrics
                    ).lower()
                    row["raw_values_omitted"] = global_raw_omitted
                    clamped_payload = metric_payload.get("clamped_values")
                    if isinstance(clamped_payload, Mapping):
                        regular = clamped_payload.get("regular")
                        absolute = clamped_payload.get("absolute")
                        if regular is not None:
                            row["clamp_regular"] = regular
                        if absolute is not None:
                            row["clamp_absolute"] = absolute
                    row.update(freeze_defaults)
                    writer.writerow(row)
            freeze_summary = global_summary.get("freeze_summary")
            summary_mapping = freeze_summary if isinstance(freeze_summary, Mapping) else None
            freeze_row = _freeze_row_payload(
                primary_exchange="__all__",
                strategy="__all__",
                summary=summary_mapping,
                truncated=None,
            )
            if freeze_row is not None:
                writer.writerow(freeze_row)


def _maybe_plot(
    groups: Iterable[dict[str, object]],
    destination: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("Matplotlib nie jest dostępny – pomijam generowanie wykresów", file=sys.stderr)
        return

    destination.mkdir(parents=True, exist_ok=True)
    for group in groups:
        primary_exchange = group["primary_exchange"]
        strategy = group["strategy"]
        metrics = group["metrics"]
        raw_values = group["raw_values"]
        for metric_name, values in raw_values.items():
            if not values:
                continue
            figure = plt.figure()
            ax = figure.add_subplot(1, 1, 1)
            ax.hist(values, bins=min(20, len(values)))
            ax.set_title(f"{metric_name}\n{primary_exchange} / {strategy}")
            ax.set_xlabel(metric_name)
            ax.set_ylabel("Liczba obserwacji")
            safe_metric = metric_name.replace("/", "_")
            safe_exchange = primary_exchange.replace("/", "_")
            safe_strategy = strategy.replace("/", "_")
            output_path = destination / f"{safe_exchange}__{safe_strategy}__{safe_metric}.png"
            figure.tight_layout()
            figure.savefig(output_path)
            plt.close(figure)


def _generate_report(
    *,
    journal_events: Iterable[Mapping[str, object]],
    autotrade_entries: Iterable[Mapping[str, object]],
    percentiles: list[float],
    suggestion_percentile: float,
    since: datetime | None = None,
    until: datetime | None = None,
    current_signal_thresholds: Mapping[str, float] | None = None,
    risk_threshold_sources: Iterable[str] | None = None,
    cli_risk_score: float | None = None,
    file_risk_score: float | None = None,
    current_signal_threshold_sources: Mapping[str, object] | None = None,
    max_freeze_events: int | None = None,
    max_raw_values: int | None = None,
    max_group_samples: int | None = _DEFAULT_GROUP_SAMPLE_LIMIT,
    max_global_samples: int | None = _DEFAULT_GLOBAL_SAMPLE_LIMIT,
) -> dict[str, object]:
    symbol_map: dict[str, tuple[str, str]] = {}
    grouped_values: dict[tuple[str, str], dict[str, StreamingMetricAggregator]] = defaultdict(dict)
    grouped_metric_rngs: dict[tuple[str, str], dict[str, random.Random]] = defaultdict(dict)
    raw_value_snapshots: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    raw_value_omitted: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    raw_value_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    raw_value_rngs: dict[tuple[str, str], dict[str, random.Random]] = defaultdict(dict)
    aggregated_metrics: dict[str, StreamingMetricAggregator] = {}
    aggregated_metric_rngs: dict[str, random.Random] = {}

    if max_group_samples is None:
        group_sample_limit: int | None = _DEFAULT_GROUP_SAMPLE_LIMIT
    elif max_group_samples <= 0:
        group_sample_limit = None
    else:
        group_sample_limit = max_group_samples

    def _stable_seed(label: str) -> int:
        digest = hashlib.blake2s(label.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big", signed=False)

    def _group_metric_rng(key: tuple[str, str], metric: str) -> random.Random:
        rng = grouped_metric_rngs[key].get(metric)
        if rng is None:
            seed = f"group::{key[0]}::{key[1]}::{metric}"
            rng = random.Random(_stable_seed(seed))
            grouped_metric_rngs[key][metric] = rng
        return rng

    def _global_metric_rng(metric: str) -> random.Random:
        rng = aggregated_metric_rngs.get(metric)
        if rng is None:
            rng = random.Random(_stable_seed(f"global::{metric}"))
            aggregated_metric_rngs[metric] = rng
        return rng

    def _raw_sample_rng(key: tuple[str, str], metric: str) -> random.Random:
        rng = raw_value_rngs[key].get(metric)
        if rng is None:
            seed = f"raw::{key[0]}::{key[1]}::{metric}"
            rng = random.Random(_stable_seed(seed))
            raw_value_rngs[key][metric] = rng
        return rng

    def _ensure_group_metric(key: tuple[str, str], metric: str) -> StreamingMetricAggregator:
        aggregator = grouped_values[key].get(metric)
        if aggregator is None:
            aggregator = StreamingMetricAggregator(
                sample_limit=group_sample_limit,
                rng=_group_metric_rng(key, metric),
                domain=_METRIC_VALUE_DOMAINS.get(metric),
            )
            grouped_values[key][metric] = aggregator
        return aggregator

    def _ensure_global_metric(metric: str) -> StreamingMetricAggregator:
        aggregator = aggregated_metrics.get(metric)
        if aggregator is None:
            limit = max_global_samples if (max_global_samples is None or max_global_samples >= 0) else None
            aggregator = StreamingMetricAggregator(
                sample_limit=limit,
                rng=_global_metric_rng(metric),
                domain=_METRIC_VALUE_DOMAINS.get(metric),
            )
            aggregated_metrics[metric] = aggregator
        return aggregator

    def _record_metric_value(key: tuple[str, str], metric: str, value: float) -> None:
        _ensure_group_metric(key, metric).add(value)
        _ensure_global_metric(metric).add(value)
        _record_raw_sample(key, metric, value)

    def _record_raw_sample(key: tuple[str, str], metric: str, value: float) -> None:
        if max_raw_values is not None and max_raw_values >= 0:
            limit = max_raw_values
            samples = raw_value_snapshots[key][metric]
            total_tracker = raw_value_counts[key]
            total_tracker[metric] += 1
            seen = total_tracker[metric]

            if limit == 0:
                if samples:
                    samples.clear()
                raw_value_omitted[key][metric] = seen
                return

            if len(samples) < limit:
                samples.append(value)
            else:
                rng = _raw_sample_rng(key, metric)
                index = rng.randrange(seen)
                if index < limit:
                    samples[index] = value
            raw_value_omitted[key][metric] = max(seen - len(samples), 0)
        else:
            raw_value_snapshots[key][metric].append(value)
    freeze_summaries: dict[tuple[str, str], dict[str, object]] = defaultdict(
        lambda: {
            "total": 0,
            "type_counts": Counter(),
            "status_counts": Counter(),
            "reason_counts": Counter(),
        }
    )
    freeze_events: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    display_names: dict[tuple[str, str], tuple[str, str]] = {}

    def _select_display(current: str, candidate: str, canonical: str) -> str:
        if not candidate:
            candidate = canonical
        elif _is_unknown_token(candidate):
            candidate = canonical
        if not current:
            return candidate
        if _is_unknown_token(current) and not _is_unknown_token(candidate):
            return candidate
        if current == canonical and candidate != canonical:
            return candidate
        return current

    def _record_display(
        key: tuple[str, str],
        exchange: str | None,
        strategy: str | None,
    ) -> None:
        candidate_exchange = exchange if exchange is not None else key[0]
        candidate_strategy = strategy if strategy is not None else key[1]
        if _is_unknown_token(candidate_exchange):
            candidate_exchange = key[0]
        if _is_unknown_token(candidate_strategy):
            candidate_strategy = key[1]
        current = display_names.get(key)
        if current is None:
            display_names[key] = (candidate_exchange, candidate_strategy)
            return
        display_exchange = _select_display(current[0], candidate_exchange, key[0])
        display_strategy = _select_display(current[1], candidate_strategy, key[1])
        display_names[key] = (display_exchange, display_strategy)

    def _record_freeze(
        key: tuple[str, str],
        payload: Mapping[str, object],
    ) -> None:
        summary = freeze_summaries[key]
        summary["total"] = int(summary["total"]) + 1
        status = str(payload.get("status") or "unknown")
        freeze_type = str(payload.get("type") or "manual")
        reason = str(payload.get("reason") or "unknown")
        duration = payload.get("duration")
        if isinstance(summary["type_counts"], Counter):
            summary["type_counts"][freeze_type] += 1
        if isinstance(summary["status_counts"], Counter):
            summary["status_counts"][status] += 1
        if isinstance(summary["reason_counts"], Counter):
            summary["reason_counts"][reason] += 1
        if duration is not None:
            try:
                numeric_duration = float(duration)
            except (TypeError, ValueError):
                numeric_duration = None
            if numeric_duration is not None and math.isfinite(numeric_duration):
                _record_metric_value(key, "risk_freeze_duration", numeric_duration)
        should_collect = True
        if max_freeze_events is not None:
            if max_freeze_events <= 0:
                should_collect = False
            else:
                should_collect = len(freeze_events[key]) < max_freeze_events
        if should_collect:
            freeze_events[key].append(
                {
                    "status": status,
                    "type": freeze_type,
                    "reason": reason,
                    "duration": duration,
                    "risk_score": payload.get("risk_score"),
                }
            )

    journal_count = 0
    for event in journal_events:
        journal_count += 1
        _update_symbol_map_entry(symbol_map, event)
        base_exchange = _normalize_string(event.get("primary_exchange"))
        base_strategy = _normalize_string(event.get("strategy"))
        key = _resolve_key(base_exchange, base_strategy)
        _record_display(key, base_exchange, base_strategy)
        for metric in ("signal_after_adjustment", "signal_after_clamp"):
            value = event.get(metric)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric):
                continue
            _record_metric_value(key, metric, numeric)
        for freeze_payload in _iter_freeze_events(event):
            symbol = _extract_symbol(event)
            exchange = base_exchange
            strategy = base_strategy
            if symbol:
                canonical_symbol = _canonicalize_symbol_key(symbol)
                mapped = symbol_map.get(canonical_symbol) if canonical_symbol else None
                if mapped:
                    mapped_exchange, mapped_strategy = mapped
                    if exchange is None and mapped_exchange not in (None, "unknown"):
                        exchange = mapped_exchange
                    if strategy is None and mapped_strategy not in (None, "unknown"):
                        strategy = mapped_strategy
            freeze_key = _resolve_key(exchange, strategy)
            _record_display(freeze_key, exchange, strategy)
            _record_freeze(freeze_key, freeze_payload)

    autotrade_count = 0
    for entry in autotrade_entries:
        autotrade_count += 1
        summary = _extract_summary(entry)
        symbol = _extract_symbol(entry)
        key, display = _resolve_group_from_symbol(entry, symbol, summary, symbol_map)
        _record_display(key, display[0], display[1])

        freeze_payloads = list(_iter_freeze_events(entry))
        for freeze_payload in freeze_payloads:
            _record_freeze(key, freeze_payload)

        if not summary:
            continue

        status_candidates: list[object] = [entry.get("status")]
        detail = entry.get("detail")
        if isinstance(detail, Mapping):
            status_candidates.append(detail.get("status"))
        is_freeze_entry = any(
            _normalize_freeze_status(candidate) for candidate in status_candidates if candidate is not None
        )
        if is_freeze_entry:
            continue

        risk_score = summary.get("risk_score")
        if risk_score is None:
            continue
        try:
            score_value = float(risk_score)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score_value):
            continue
        _record_metric_value(key, "risk_score", score_value)

    risk_threshold_paths = _normalize_risk_threshold_paths(risk_threshold_sources)
    current_risk_score = None
    if risk_threshold_paths:
        for config_path in risk_threshold_paths:
            thresholds = load_risk_thresholds(config_path=config_path)
            value = _extract_risk_score_threshold(thresholds)
            if value is not None:
                current_risk_score = value
    else:
        thresholds = load_risk_thresholds()
        current_risk_score = _extract_risk_score_threshold(thresholds)

    if file_risk_score is not None:
        current_risk_score = float(file_risk_score)

    if cli_risk_score is not None:
        current_risk_score = float(cli_risk_score)

    groups: list[dict[str, object]] = []
    aggregated_freeze_summary = {
        "total": 0,
        "type_counts": Counter(),
        "status_counts": Counter(),
        "reason_counts": Counter(),
        "omitted_total": 0,
    }

    freeze_truncated_group_count = 0
    raw_truncated_group_count = 0
    raw_values_omitted_total = 0
    group_sample_truncated_count = 0
    group_sample_omitted_total = 0
    aggregated_sample_truncated_count = 0
    aggregated_sample_omitted_total = 0
    clamp_group_regular_total = 0
    clamp_group_absolute_total = 0
    clamp_group_regular_metrics: set[tuple[str, str, str]] = set()
    clamp_group_absolute_metrics: set[tuple[str, str, str]] = set()
    clamp_global_regular_total = 0
    clamp_global_absolute_total = 0
    clamp_global_regular_metrics: set[str] = set()
    clamp_global_absolute_metrics: set[str] = set()
    approximation_group_metrics: set[tuple[str, str, str]] = set()
    approximation_global_metrics: set[str] = set()

    all_group_keys = set(grouped_values.keys())
    all_group_keys.update(freeze_summaries.keys())
    all_group_keys.update(display_names.keys())
    all_group_keys.update(freeze_events.keys())

    for exchange, strategy in sorted(all_group_keys):
        key = (exchange, strategy)
        metrics = grouped_values.get(key) or {}
        metrics_payload = _build_metrics_section(
            metrics,
            percentiles,
            suggestion_percentile,
            current_risk_score=current_risk_score,
            current_signal_thresholds=current_signal_thresholds,
        )
        for metric_name, metric_payload in metrics_payload.items():
            approximation_mode = metric_payload.get("approximation_mode")
            if approximation_mode:
                approximation_group_metrics.add((exchange, strategy, metric_name))
        for metric_name, aggregator in metrics.items():
            if isinstance(aggregator, StreamingMetricAggregator):
                if aggregator.sample_truncated:
                    group_sample_truncated_count += 1
                    group_sample_omitted_total += aggregator.omitted_samples
                if aggregator.clamp_regular_count:
                    clamp_group_regular_total += aggregator.clamp_regular_count
                    clamp_group_regular_metrics.add((exchange, strategy, metric_name))
                if aggregator.clamp_absolute_count:
                    clamp_group_absolute_total += aggregator.clamp_absolute_count
                    clamp_group_absolute_metrics.add((exchange, strategy, metric_name))
        freeze_summary = freeze_summaries.get(key) or {
            "total": 0,
            "type_counts": Counter(),
            "status_counts": Counter(),
            "reason_counts": Counter(),
        }
        display_exchange, display_strategy = display_names.get(
            key, (exchange, strategy)
        )
        raw_freeze_entries = list(freeze_events.get(key, []))
        total_freezes = int(freeze_summary.get("total") or 0)
        raw_freeze_omitted = max(total_freezes - len(raw_freeze_entries), 0)
        raw_freeze_truncated = False
        if max_freeze_events is not None and max_freeze_events >= 0:
            if raw_freeze_omitted > 0:
                raw_freeze_truncated = True
                freeze_truncated_group_count += 1
        else:
            raw_freeze_omitted = 0
        freeze_summary["omitted_total"] = int(raw_freeze_omitted)

        raw_values_snapshot = raw_value_snapshots.get(key, {})
        raw_values_omitted_map = raw_value_omitted.get(key, {})
        metric_keys = set(raw_values_snapshot.keys()) | set(raw_values_omitted_map.keys())
        raw_values_payload = {
            metric: _finite_values(raw_values_snapshot.get(metric, []))
            for metric in metric_keys
        }
        raw_values_omitted_payload = {
            metric: int(raw_values_omitted_map.get(metric, 0))
            for metric in metric_keys
        }
        raw_values_truncated = any(value > 0 for value in raw_values_omitted_payload.values())
        raw_values_omitted_sum = sum(raw_values_omitted_payload.values())
        raw_values_omitted_total += raw_values_omitted_sum
        if raw_values_truncated:
            raw_truncated_group_count += 1

        if not metrics and total_freezes == 0 and not raw_freeze_entries:
            continue

        if isinstance(freeze_summary.get("type_counts"), Counter):
            aggregated_freeze_summary["type_counts"].update(freeze_summary["type_counts"])
        if isinstance(freeze_summary.get("status_counts"), Counter):
            aggregated_freeze_summary["status_counts"].update(freeze_summary["status_counts"])
        if isinstance(freeze_summary.get("reason_counts"), Counter):
            aggregated_freeze_summary["reason_counts"].update(freeze_summary["reason_counts"])
        freeze_summary_payload = _format_freeze_summary(freeze_summary)
        aggregated_freeze_summary["total"] = int(aggregated_freeze_summary["total"]) + total_freezes
        aggregated_freeze_summary["omitted_total"] = int(
            aggregated_freeze_summary.get("omitted_total", 0)
        ) + int(raw_freeze_omitted)
        groups.append(
            {
                "primary_exchange": display_exchange,
                "strategy": display_strategy,
                "metrics": metrics_payload,
                "raw_values": raw_values_payload,
                "raw_values_omitted": raw_values_omitted_payload,
                "raw_values_truncated": raw_values_truncated,
                "freeze_summary": freeze_summary_payload,
                "raw_freeze_events": raw_freeze_entries,
                "raw_freeze_events_truncated": raw_freeze_truncated,
                "raw_freeze_events_omitted": raw_freeze_omitted,
            }
        )

    global_metrics = _build_metrics_section(
        aggregated_metrics,
        percentiles,
        suggestion_percentile,
        current_risk_score=current_risk_score,
        current_signal_thresholds=current_signal_thresholds,
    )

    for metric_name, aggregator in aggregated_metrics.items():
        if aggregator.sample_truncated:
            aggregated_sample_truncated_count += 1
            aggregated_sample_omitted_total += aggregator.omitted_samples
        if aggregator.clamp_regular_count:
            clamp_global_regular_total += aggregator.clamp_regular_count
            clamp_global_regular_metrics.add(metric_name)
        if aggregator.clamp_absolute_count:
            clamp_global_absolute_total += aggregator.clamp_absolute_count
            clamp_global_absolute_metrics.add(metric_name)
        metric_payload = global_metrics.get(metric_name)
        if isinstance(metric_payload, Mapping):
            approximation_mode = metric_payload.get("approximation_mode")
            if approximation_mode:
                approximation_global_metrics.add(metric_name)
    global_summary = {
        "metrics": global_metrics,
        "freeze_summary": _format_freeze_summary(aggregated_freeze_summary),
    }

    sources_payload: dict[str, object] = {
        "journal_events": journal_count,
        "autotrade_entries": autotrade_count,
    }
    signal_sources_payload: dict[str, object] = {}
    if current_signal_threshold_sources:
        files = current_signal_threshold_sources.get("files")
        if files:
            signal_sources_payload["files"] = list(files)
        inline_values = current_signal_threshold_sources.get("inline")
        if inline_values:
            signal_sources_payload["inline"] = {
                key: float(value)
                for key, value in inline_values.items()
                if isinstance(key, str)
            }
    if file_risk_score is not None:
        signal_sources_payload["file_risk_score"] = float(file_risk_score)
    if signal_sources_payload:
        sources_payload["current_signal_thresholds"] = signal_sources_payload
    if risk_threshold_paths:
        sources_payload["risk_threshold_files"] = [str(path) for path in risk_threshold_paths]
    if cli_risk_score is not None:
        sources_payload["risk_score_override"] = float(cli_risk_score)
    if max_freeze_events is not None:
        sources_payload["max_freeze_events"] = int(max_freeze_events)
        if freeze_truncated_group_count:
            sources_payload["raw_freeze_events_truncated_groups"] = freeze_truncated_group_count
    if max_raw_values is not None:
        sources_payload["max_raw_values"] = int(max_raw_values)
        if raw_truncated_group_count:
            sources_payload["raw_values_truncated_groups"] = raw_truncated_group_count
        if raw_values_omitted_total:
            sources_payload["raw_values_omitted_total"] = int(raw_values_omitted_total)
    if max_group_samples is not None:
        sources_payload["max_group_samples"] = int(max_group_samples)
    if group_sample_truncated_count:
        sources_payload["group_samples_truncated_metrics"] = group_sample_truncated_count
    if group_sample_omitted_total:
        sources_payload["group_samples_omitted_total"] = int(group_sample_omitted_total)
    if max_global_samples is not None:
        sources_payload["max_global_samples"] = int(max_global_samples)
        if aggregated_sample_truncated_count:
            sources_payload["global_samples_truncated_metrics"] = aggregated_sample_truncated_count
        if aggregated_sample_omitted_total:
            sources_payload["global_samples_omitted_total"] = int(aggregated_sample_omitted_total)
    if approximation_group_metrics:
        sorted_group_metrics = sorted(approximation_group_metrics)
        sources_payload["approximation_metrics"] = len(sorted_group_metrics)
        sources_payload["approximation_metrics_list"] = [
            {
                "primary_exchange": exchange,
                "strategy": strategy,
                "metric": metric,
            }
            for exchange, strategy, metric in sorted_group_metrics
        ]
    if approximation_global_metrics:
        sorted_global_metrics = sorted(approximation_global_metrics)
        sources_payload["approximation_global_metrics"] = len(sorted_global_metrics)
        sources_payload["approximation_global_metrics_list"] = sorted_global_metrics
    clamp_payload: dict[str, object] = {}
    if clamp_group_regular_total or clamp_group_absolute_total:
        group_payload: dict[str, object] = {
            "regular": int(clamp_group_regular_total),
            "absolute": int(clamp_group_absolute_total),
        }
        if clamp_group_regular_metrics:
            group_payload["metrics_with_regular_clamp"] = len(clamp_group_regular_metrics)
        if clamp_group_absolute_metrics:
            group_payload["metrics_with_absolute_clamp"] = len(clamp_group_absolute_metrics)
        clamp_payload["group"] = group_payload
    if clamp_global_regular_total or clamp_global_absolute_total:
        global_payload: dict[str, object] = {
            "regular": int(clamp_global_regular_total),
            "absolute": int(clamp_global_absolute_total),
        }
        if clamp_global_regular_metrics:
            global_payload["metrics_with_regular_clamp"] = len(clamp_global_regular_metrics)
        if clamp_global_absolute_metrics:
            global_payload["metrics_with_absolute_clamp"] = len(clamp_global_absolute_metrics)
        clamp_payload["global"] = global_payload
    if clamp_payload:
        sources_payload["clamped_values"] = clamp_payload

    return {
        "schema": "stage6.autotrade.threshold_calibration",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "percentiles": [_format_percentile_label(p) for p in percentiles],
        "suggestion_percentile": suggestion_percentile,
        "filters": {
            "since": since.isoformat() if since else None,
            "until": until.isoformat() if until else None,
        },
        "groups": groups,
        "global_summary": global_summary,
        "sources": sources_payload,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analizuje dziennik TradingDecisionJournal oraz eksport autotradera, "
            "aby zaproponować progi sygnałów i blokad ryzyka."
        )
    )
    parser.add_argument(
        "--journal",
        required=True,
        nargs="+",
        help="Ścieżki do plików JSONL (lub katalogów) z TradingDecisionJournal",
    )
    parser.add_argument(
        "--autotrade-export",
        required=True,
        nargs="+",
        help="Pliki JSON wygenerowane przez export_risk_evaluations lub eksport statusów autotradera",
    )
    parser.add_argument(
        "--percentiles",
        help="Lista percentyli (np. 0.5,0.9,0.95)",
    )
    parser.add_argument(
        "--suggestion-percentile",
        type=float,
        default=0.95,
        help="Percentyl wykorzystywany do sugestii nowego progu",
    )
    parser.add_argument(
        "--since",
        help="Uwzględniaj zdarzenia od wskazanej daty (ISO 8601)",
    )
    parser.add_argument(
        "--until",
        help="Uwzględniaj zdarzenia do wskazanej daty (ISO 8601)",
    )
    parser.add_argument(
        "--output-json",
        help="Opcjonalna ścieżka do zapisu pełnego raportu w formacie JSON",
    )
    parser.add_argument(
        "--output-csv",
        help="Opcjonalna ścieżka do zapisu raportu w formacie CSV",
    )
    parser.add_argument(
        "--output-threshold-config",
        help=(
            "Opcjonalna ścieżka do pliku YAML/JSON z sugerowanymi progami – zgodna z "
            "formatem load_risk_thresholds"
        ),
    )
    parser.add_argument(
        "--plot-dir",
        help="Opcjonalny katalog na histogramy z rozkładami metryk",
    )
    parser.add_argument(
        "--current-threshold",
        action="append",
        help=(
            "Opcjonalne źródło aktualnych progów sygnału – można podać "
            "plik JSON/YAML lub pary metric=value (np. signal_after_clamp=0.8)"
        ),
    )
    parser.add_argument(
        "--risk-thresholds",
        action="append",
        help=(
            "Opcjonalne pliki z progami ryzyka używanymi przez load_risk_thresholds;"
            " można wskazać wiele ścieżek, aby nadpisywać wartości (ostatnia wygrywa)."
        ),
    )
    parser.add_argument(
        "--max-freeze-events",
        type=int,
        help=(
            "Maksymalna liczba zdarzeń risk_freeze zapisywanych w raporcie dla każdej pary"
            " giełda/strategia (0 oznacza pominięcie sekcji raw_freeze_events)."
        ),
    )
    parser.add_argument(
        "--max-raw-values",
        type=int,
        help=(
            "Maksymalna liczba przechowywanych surowych próbek metryk dla każdej pary"
            " giełda/strategia (0 oznacza brak próbek w raporcie i wyłącznie statystyki)."
        ),
    )
    parser.add_argument(
        "--max-group-samples",
        type=int,
        default=_DEFAULT_GROUP_SAMPLE_LIMIT,
        help=(
            "Limit próbek wykorzystywanych przez agregatory metryk na poziomie pary"
            " giełda/strategia (0 lub wartość ujemna oznacza brak limitu)."
        ),
    )
    parser.add_argument(
        "--max-global-samples",
        type=int,
        default=_DEFAULT_GLOBAL_SAMPLE_LIMIT,
        help=(
            "Limit próbek używanych do percentyli w podsumowaniu globalnym (wartość ujemna "
            "wyłącza limit, 0 powoduje zastosowanie przybliżeń na bazie rozkładu normalnego "
            "wyliczonego z sum i wariancji)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    percentiles = _parse_percentiles(args.percentiles)
    if not (0.0 < args.suggestion_percentile < 1.0):
        raise SystemExit("Percentyl sugerowanego progu musi być w przedziale (0, 1)")

    max_freeze_events = args.max_freeze_events
    if max_freeze_events is not None and max_freeze_events < 0:
        raise SystemExit("Parametr --max-freeze-events musi być nieujemny")

    max_raw_values = args.max_raw_values
    if max_raw_values is not None and max_raw_values < 0:
        raise SystemExit("Parametr --max-raw-values musi być nieujemny")

    max_group_samples = args.max_group_samples
    if max_group_samples is None:
        max_group_samples = _DEFAULT_GROUP_SAMPLE_LIMIT

    max_global_samples = args.max_global_samples
    if max_global_samples is not None and max_global_samples < 0:
        max_global_samples = None

    since = _parse_cli_datetime(args.since)
    until = _parse_cli_datetime(args.until)
    if since and until and since > until:
        raise SystemExit("Data początkowa nie może być późniejsza niż końcowa")

    journal_paths = list(_iter_paths(args.journal))
    if not journal_paths:
        raise SystemExit("Nie znaleziono żadnych plików dziennika")

    journal_events = _load_journal_events(journal_paths, since=since, until=until)
    autotrade_entries = _load_autotrade_entries(args.autotrade_export, since=since, until=until)
    (
        current_signal_thresholds,
        risk_score_sources,
        current_signal_threshold_sources,
    ) = _load_current_signal_thresholds(args.current_threshold)

    cli_risk_score = risk_score_sources.from_inline
    file_risk_score = risk_score_sources.from_files

    report = _generate_report(
        journal_events=journal_events,
        autotrade_entries=autotrade_entries,
        percentiles=percentiles,
        suggestion_percentile=args.suggestion_percentile,
        since=since,
        until=until,
        current_signal_thresholds=current_signal_thresholds,
        risk_threshold_sources=args.risk_thresholds,
        cli_risk_score=cli_risk_score,
        file_risk_score=file_risk_score,
        current_signal_threshold_sources=current_signal_threshold_sources,
        max_freeze_events=max_freeze_events,
        max_raw_values=max_raw_values,
        max_group_samples=max_group_samples,
        max_global_samples=max_global_samples,
    )

    if args.output_json:
        _write_json(report, Path(args.output_json))
        print(f"Zapisano raport JSON: {args.output_json}")

    if args.output_csv:
        percentile_keys = report["percentiles"]
        _write_csv(
            report["groups"],
            Path(args.output_csv),
            percentiles=percentile_keys,
            global_summary=report.get("global_summary"),
        )
        print(f"Zapisano raport CSV: {args.output_csv}")

    if args.output_threshold_config:
        threshold_config = _build_threshold_config(report)
        if not threshold_config:
            print(
                "Brak sugerowanych progów do eksportu – plik nie został utworzony",
                file=sys.stderr,
            )
        else:
            _write_threshold_config(
                threshold_config, Path(args.output_threshold_config)
            )
            print(
                f"Zapisano konfigurację progów: {args.output_threshold_config}"
            )

    if args.plot_dir:
        _maybe_plot(report["groups"], Path(args.plot_dir))

    total_groups = len(report["groups"])
    sources = report.get("sources", {})
    journal_count = int(sources.get("journal_events", 0))
    autotrade_count = int(sources.get("autotrade_entries", 0))
    print(
        f"Przetworzono {journal_count} zdarzeń dziennika i "
        f"{autotrade_count} wpisów autotradera dla {total_groups} kombinacji giełda/strategia."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
