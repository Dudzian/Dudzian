from __future__ import annotations

import argparse
import io
import gzip
import heapq
import json
import math
import sys
from array import array
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, Literal, Mapping, TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.ai.config_loader import load_risk_thresholds


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


_SUPPORTED_JOURNAL_EXTENSIONS = (".jsonl", ".jsonl.gz")
_SUPPORTED_AUTOTRADE_EXTENSIONS = (".json", ".json.gz", ".jsonl", ".jsonl.gz")


_JSON_STREAM_CHUNK_SIZE = 65536


_JSONStreamMode = Literal["entries", "search"]


class _JSONStreamEntriesParser:
    def __init__(
        self,
        handle: TextIO,
        path: Path,
        *,
        chunk_size: int,
        normalizer: Callable[[object], Mapping[str, object] | None],
    ) -> None:
        self._handle = handle
        self._path = path
        self._chunk_size = chunk_size
        self._normalize_entry = normalizer
        self._decoder = json.JSONDecoder()
        self._buffer = ""
        self._position = 0

    def _read_more(self) -> bool:
        chunk = self._handle.read(self._chunk_size)
        if not chunk:
            return False
        if self._position:
            self._buffer = self._buffer[self._position :] + chunk
            self._position = 0
        else:
            self._buffer += chunk
        return True

    def _shrink_buffer(self) -> None:
        if self._position >= len(self._buffer):
            self._buffer = ""
            self._position = 0
        elif self._position > self._chunk_size:
            self._buffer = self._buffer[self._position :]
            self._position = 0

    def _skip_whitespace(self) -> bool:
        while True:
            while self._position < len(self._buffer) and self._buffer[self._position].isspace():
                self._position += 1
            if self._position < len(self._buffer) and self._buffer[self._position] == "\ufeff":
                self._position += 1
                continue
            if self._position < len(self._buffer):
                return True
            if not self._read_more():
                return False

    def _decode_value(self) -> object:
        while True:
            try:
                value, next_position = self._decoder.raw_decode(self._buffer, self._position)
            except json.JSONDecodeError as exc:  # noqa: BLE001 - CLI feedback
                if not self._read_more():
                    raise SystemExit(
                        f"Niepoprawny JSON w eksporcie autotradera {self._path}: {exc}"
                    ) from exc
                continue
            self._position = next_position
            self._shrink_buffer()
            return value

    def _consume_array(self, mode: _JSONStreamMode) -> Iterator[Mapping[str, object]]:
        self._position += 1
        first_item = True
        while True:
            if not self._skip_whitespace():
                raise SystemExit(
                    f"Niepoprawny JSON w eksporcie autotradera {self._path}: niekompletna tablica"
                )
            if self._position >= len(self._buffer) and not self._read_more():
                raise SystemExit(
                    f"Niepoprawny JSON w eksporcie autotradera {self._path}: niekompletna tablica"
                )
            current = self._buffer[self._position]
            if current == "]":
                self._position += 1
                self._shrink_buffer()
                return
            if not first_item:
                if current != ",":
                    raise SystemExit(
                        f"Niepoprawny JSON w eksporcie autotradera {self._path}: oczekiwano przecinka"
                    )
                self._position += 1
                if not self._skip_whitespace():
                    raise SystemExit(
                        f"Niepoprawny JSON w eksporcie autotradera {self._path}: niekompletna tablica"
                    )
                if self._position >= len(self._buffer) and not self._read_more():
                    raise SystemExit(
                        f"Niepoprawny JSON w eksporcie autotradera {self._path}: niekompletna tablica"
                    )
                current = self._buffer[self._position]
                if current == "]":
                    raise SystemExit(
                        f"Niepoprawny JSON w eksporcie autotradera {self._path}: dodatkowy przecinek"
                    )
            first_item = False
            if mode == "entries":
                value = self._decode_value()
                yield from self._iter_nested_entries(value)
                continue
            yield from self._consume_value("search")

    def _consume_value(self, mode: _JSONStreamMode) -> Iterator[Mapping[str, object]]:
        if not self._skip_whitespace():
            raise SystemExit(
                f"Niepoprawny JSON w eksporcie autotradera {self._path}: niekompletna wartość"
            )
        if self._position >= len(self._buffer) and not self._read_more():
            raise SystemExit(
                f"Niepoprawny JSON w eksporcie autotradera {self._path}: niekompletna wartość"
            )
        current = self._buffer[self._position]
        if current == "[":
            yield from self._consume_array(mode)
            return
        if current == "{":
            if mode == "entries":
                value = self._decode_value()
                yield from self._iter_nested_entries(value)
                return
            yield from self._consume_object()
            return
        if current in "\"-0123456789tfn":
            value = self._decode_value()
            if mode == "entries":
                normalized = self._normalize_entry(value)
                if normalized is not None:
                    yield normalized
            return
        raise SystemExit(
            f"Niepoprawny JSON w eksporcie autotradera {self._path}: nieznany typ wartości"
        )

    def _consume_object(self, *, emit_self: bool = False) -> Iterator[Mapping[str, object]]:
        self._position += 1
        collected: dict[str, object] | None = {} if emit_self else None
        pending_entries: list[Mapping[str, object]] | None = [] if emit_self else None
        saw_entries_field = False
        while True:
            if not self._skip_whitespace():
                raise SystemExit(
                    f"Niepoprawny JSON w eksporcie autotradera {self._path}: niekompletny obiekt"
                )
            if self._position >= len(self._buffer) and not self._read_more():
                raise SystemExit(
                    f"Niepoprawny JSON w eksporcie autotradera {self._path}: niekompletny obiekt"
                )
            current = self._buffer[self._position]
            if current == "}":
                self._position += 1
                self._shrink_buffer()
                break
            key = self._decode_value()
            if not isinstance(key, str):
                raise SystemExit(
                    f"Niepoprawny JSON w eksporcie autotradera {self._path}: klucz musi być napisem"
                )
            if not self._skip_whitespace():
                raise SystemExit(
                    f"Niepoprawny JSON w eksporcie autotradera {self._path}: niekompletna para klucz-wartość"
                )
            if self._position >= len(self._buffer) and not self._read_more():
                raise SystemExit(
                    f"Niepoprawny JSON w eksporcie autotradera {self._path}: niekompletna para klucz-wartość"
                )
            if self._buffer[self._position] != ":":
                raise SystemExit(
                    f"Niepoprawny JSON w eksporcie autotradera {self._path}: oczekiwano ':' po kluczu {key}"
                )
            self._position += 1
            if not self._skip_whitespace():
                raise SystemExit(
                    f"Niepoprawny JSON w eksporcie autotradera {self._path}: niekompletna para klucz-wartość"
                )
            if self._position >= len(self._buffer) and not self._read_more():
                raise SystemExit(
                    f"Niepoprawny JSON w eksporcie autotradera {self._path}: niekompletna para klucz-wartość"
                )
            if key == "entries":
                saw_entries_field = True
                yield from self._consume_value("entries")
            elif emit_self and collected is not None and pending_entries is not None:
                value = self._decode_value()
                collected[key] = value
                for nested in self._iter_nested_entries(value):
                    pending_entries.append(nested)
            else:
                yield from self._consume_value("search")
            if not self._skip_whitespace():
                raise SystemExit(
                    f"Niepoprawny JSON w eksporcie autotradera {self._path}: niekompletny obiekt"
                )
            if self._position >= len(self._buffer) and not self._read_more():
                raise SystemExit(
                    f"Niepoprawny JSON w eksporcie autotradera {self._path}: niekompletny obiekt"
                )
            current = self._buffer[self._position]
            if current == ",":
                self._position += 1
                if not self._skip_whitespace():
                    raise SystemExit(
                        f"Niepoprawny JSON w eksporcie autotradera {self._path}: niekompletny obiekt"
                    )
                if self._position >= len(self._buffer) and not self._read_more():
                    raise SystemExit(
                        f"Niepoprawny JSON w eksporcie autotradera {self._path}: niekompletny obiekt"
                    )
                if self._buffer[self._position] == "}":
                    raise SystemExit(
                        f"Niepoprawny JSON w eksporcie autotradera {self._path}: dodatkowy przecinek"
                    )
                continue
            if current == "}":
                self._position += 1
                self._shrink_buffer()
                break
            raise SystemExit(
                f"Niepoprawny JSON w eksporcie autotradera {self._path}: oczekiwano ',' lub '}}'"
            )

        if emit_self and collected is not None and not saw_entries_field:
            normalized = self._normalize_entry(collected)
            if normalized is not None:
                yield normalized
        if emit_self and pending_entries:
            yield from pending_entries

    def iter_entries(self) -> Iterator[Mapping[str, object]]:
        if not self._skip_whitespace():
            return
        if self._position >= len(self._buffer) and not self._read_more():
            return
        current = self._buffer[self._position]
        if current == "[":
            yield from self._consume_array("entries")
            return
        if current == "{":
            yield from self._consume_object(emit_self=True)
            return
        raise SystemExit(
            f"Niepoprawny JSON w eksporcie autotradera {self._path}: oczekiwano obiektu lub tablicy"
        )

    def _iter_nested_entries(self, value: object) -> Iterator[Mapping[str, object]]:
        stack: list[object] = [value]
        while stack:
            current = stack.pop()
            if isinstance(current, Mapping):
                entries_field = current.get("entries")
                if isinstance(entries_field, (Mapping, list)):
                    other_values = [candidate for key, candidate in current.items() if key != "entries"]
                    stack.extend(reversed(other_values))
                    stack.append(entries_field)
                    normalized = self._normalize_entry(current)
                    if normalized is not None:
                        yield normalized
                    continue
                stack.extend(reversed(list(current.values())))
                normalized = self._normalize_entry(current)
                if normalized is not None:
                    yield normalized
            elif isinstance(current, list):
                stack.extend(reversed(current))
            else:
                normalized = self._normalize_entry(current)
                if normalized is not None:
                    yield normalized


_METRIC_APPEND_OBSERVER: Callable[[tuple[str, str], str, int], None] | None = None


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


def _ensure_finite_value(
    metric_name: str,
    value: float,
    *,
    source: str | None = None,
) -> float:
    """Zwraca wartość, jeśli jest skończona, w przeciwnym razie zgłasza błąd."""

    if math.isfinite(value):
        return value

    location = f" w źródle {source}" if source else ""
    raise SystemExit(
        f"Niepoprawna wartość progu dla metryki {metric_name}: {value}"
        f" (musi być skończoną liczbą){location}"
    )


def _normalize_threshold_value(
    metric_name: str,
    raw_value: float | int,
    *,
    source: str | None = None,
) -> float:
    return float(raw_value)


def _normalize_and_validate_threshold(
    metric_name: str,
    raw_value: float | int,
    *,
    source: str | None = None,
) -> float:
    normalized_value = _normalize_threshold_value(
        metric_name,
        raw_value,
        source=source,
    )
    return _ensure_finite_value(
        metric_name,
        normalized_value,
        source=source,
    )
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
        if not math.isfinite(value):
            raise SystemExit(f"Percentyl '{token}' musi być skończoną liczbą")
        if value <= 0.0 or value >= 1.0:
            raise SystemExit("Percentyle muszą znajdować się w przedziale (0, 1)")
        percentiles.append(value)
    if not percentiles:
        raise SystemExit("Lista percentyli nie może być pusta")
    return sorted(set(percentiles))


def _parse_threshold_mapping(raw: str) -> tuple[dict[str, float], dict[str, str]]:
    values: dict[str, float] = {}
    sources: dict[str, str] = {}
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
        value_str = value.strip()
        numeric = _coerce_float(value_str)
        if numeric is None:
            raise SystemExit(
                f"Nie udało się zinterpretować progu '{value}' dla metryki {key}"
            )
        normalized_key = _normalize_metric_key(key)
        pair_repr = f"{key}={value_str}"
        finite_value = _normalize_and_validate_threshold(
            normalized_key,
            numeric,
            source=f"CLI '{pair_repr}'",
        )
        values[normalized_key] = finite_value
        sources[normalized_key] = pair_repr
    return values, sources


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
) -> tuple[dict[str, float], float | None, dict[str, object]]:
    thresholds: dict[str, float] = {}
    current_risk_score: float | None = None
    source_files: list[str] = []
    risk_source_files: list[str] = []
    inline_values: dict[str, float] = {}
    inline_risk_thresholds: dict[str, float] = {}
    risk_score_origin: dict[str, object] | None = None
    inline_risk_source: str | None = None
    inline_risk_value: float | None = None
    file_risk_source: str | None = None
    file_risk_value: float | None = None
    if not sources:
        return (
            thresholds,
            current_risk_score,
            {
                "files": source_files,
                "inline": inline_values,
                "risk_files": risk_source_files,
                "risk_inline": inline_risk_thresholds,
                "risk_score_source": risk_score_origin,
            },
        )

    for source in sources:
        if not source:
            continue
        candidate = source.strip()
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            path_str = str(path)
            if path_str not in source_files:
                source_files.append(path_str)
            payload = _load_threshold_payload(path)
            if not isinstance(payload, (Mapping, list, tuple)):
                raise SystemExit(
                    "Plik z progami musi zawierać strukturę słownikową lub listę słowników"
                )
            found_risk_in_file = False
            for mapping in _iter_mappings(payload):
                for metric_name in _SUPPORTED_THRESHOLD_METRICS:
                    value = _resolve_metric_threshold(mapping, metric_name)
                    if value is not None:
                        finite_value = _normalize_and_validate_threshold(
                            metric_name,
                            raw_value=value,
                            source=path_str,
                        )
                        if metric_name == "risk_score":
                            found_risk_in_file = True
                            current_risk_score = finite_value
                            file_risk_source = path_str
                            file_risk_value = finite_value
                        else:
                            thresholds[metric_name] = finite_value
            if found_risk_in_file and path_str not in risk_source_files:
                risk_source_files.append(path_str)
            continue
        if "=" not in candidate:
            raise SystemExit(f"Ścieżka z progami nie istnieje: {path}")
        mapping, mapping_sources = _parse_threshold_mapping(candidate)
        for metric_name, numeric in mapping.items():
            if not isinstance(metric_name, str):
                continue
            metric_name_normalized = _normalize_metric_key(metric_name)
            if metric_name_normalized in _SUPPORTED_THRESHOLD_METRICS:
                source_repr = mapping_sources.get(metric_name_normalized, candidate)
                validation_source = f"CLI '{source_repr}'"
                finite_value = _ensure_finite_value(
                    metric_name_normalized,
                    numeric,
                    source=validation_source,
                )
                if metric_name_normalized == "risk_score":
                    current_risk_score = finite_value
                    inline_risk_thresholds[metric_name_normalized] = finite_value
                    inline_risk_source = source_repr
                    inline_risk_value = finite_value
                else:
                    thresholds[metric_name_normalized] = finite_value
                    inline_values[metric_name_normalized] = finite_value

    if inline_risk_value is not None:
        inline_risk_value = _ensure_finite_value(
            "risk_score",
            inline_risk_value,
            source=inline_risk_source,
        )
        current_risk_score = inline_risk_value
        risk_score_origin = {
            "kind": "inline",
            "source": inline_risk_source,
            "value": inline_risk_value,
        }
    elif file_risk_value is not None:
        file_risk_value = _ensure_finite_value(
            "risk_score",
            file_risk_value,
            source=file_risk_source,
        )
        current_risk_score = file_risk_value
        risk_score_origin = {
            "kind": "file",
            "source": file_risk_source,
            "value": file_risk_value,
        }

    return (
        thresholds,
        current_risk_score,
        {
            "files": source_files,
            "inline": inline_values,
            "risk_files": risk_source_files,
            "risk_inline": inline_risk_thresholds,
            "risk_score_source": risk_score_origin,
        },
    )


def _has_extension(path: Path, allowed: tuple[str, ...]) -> bool:
    if not path.name:
        return False
    lower = path.name.lower()
    return any(lower.endswith(ext) for ext in allowed)


def _is_json_lines_path(path: Path) -> bool:
    lower_name = path.name.lower()
    return lower_name.endswith(".jsonl") or lower_name.endswith(".jsonl.gz")


def _open_text_file(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


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
            matched = False
            for child in sorted(candidate.iterdir()):
                if not child.is_file() or not _has_extension(child, _SUPPORTED_JOURNAL_EXTENSIONS):
                    continue
                matched = True
                yield child
            if not matched:
                raise SystemExit(
                    "Katalog dzienników nie zawiera plików JSONL ani skompresowanych JSONL (.gz)"
                )
            continue
        if not _has_extension(candidate, _SUPPORTED_JOURNAL_EXTENSIONS):
            raise SystemExit(
                "Dziennik musi być plikiem JSONL (opcjonalnie skompresowanym .gz)"
            )
        yield candidate


def _iter_autotrade_paths(raw_paths: Iterable[Path | str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    directories_without_files: list[Path] = []

    for raw in raw_paths:
        candidate = Path(raw).expanduser()
        if not candidate.exists():
            raise SystemExit(f"Eksport autotradera nie istnieje: {candidate}")
        if candidate.is_dir():
            matched = False
            for child in sorted(candidate.iterdir()):
                if not child.is_file() or not _has_extension(
                    child, _SUPPORTED_AUTOTRADE_EXTENSIONS
                ):
                    continue
                resolved = child.resolve()
                if resolved in seen:
                    continue
                matched = True
                seen.add(resolved)
                paths.append(child)
            if not matched:
                directories_without_files.append(candidate)
            continue

        resolved = candidate.resolve()
        if resolved in seen:
            continue
        if not _has_extension(candidate, _SUPPORTED_AUTOTRADE_EXTENSIONS):
            raise SystemExit(
                "Eksport autotradera musi być plikiem JSON/JSONL (opcjonalnie skompresowanym .gz)"
            )
        seen.add(resolved)
        paths.append(candidate)

    if not paths:
        if directories_without_files:
            directory = directories_without_files[0]
            raise SystemExit(
                f"Katalog eksportów autotradera {directory} nie zawiera plików JSON/JSONL"
                " (również skompresowanych .gz)"
            )
        raise SystemExit("Nie znaleziono żadnych plików eksportu autotradera")

    return paths


def _load_journal_events(
    paths: Iterable[Path],
    *,
    since: datetime | None = None,
    until: datetime | None = None,
) -> Iterator[Mapping[str, object]]:
    def _iter_path(path: Path) -> Iterator[Mapping[str, object]]:
        try:
            with _open_text_file(path) as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as exc:  # noqa: BLE001 - CLI feedback
                        raise SystemExit(
                            f"Nie udało się sparsować JSON w dzienniku {path}: {exc}"
                        ) from exc
                    if not isinstance(payload, Mapping):
                        continue
                    timestamp = _parse_datetime(payload.get("timestamp"))
                    if since and timestamp and timestamp < since:
                        continue
                    if until and timestamp and timestamp > until:
                        continue
                    yield payload
        except OSError as exc:  # noqa: BLE001 - CLI feedback
            raise SystemExit(f"Nie udało się odczytać dziennika {path}: {exc}") from exc

    for path in paths:
        yield from _iter_path(path)


def _extract_entry_timestamp(entry: Mapping[str, object]) -> datetime | None:
    direct = _parse_datetime(entry.get("timestamp"))
    if direct:
        return direct

    stack: list[object] = [entry]
    seen: set[int] = set()

    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            marker = id(current)
            if marker in seen:
                continue
            seen.add(marker)

            candidate = _parse_datetime(current.get("timestamp"))
            if candidate:
                return candidate

            for key, value in current.items():
                if key == "entries":
                    continue
                if isinstance(value, (Mapping, list)):
                    stack.append(value)
        elif isinstance(current, list):
            marker = id(current)
            if marker in seen:
                continue
            seen.add(marker)
            stack.extend(current)

    return None



def _load_autotrade_entries(
    paths: Iterable[Path | str],
    *,
    since: datetime | None = None,
    until: datetime | None = None,
) -> Iterator[Mapping[str, object]]:
    def _is_entry_candidate(mapping: Mapping[str, object]) -> bool:
        if "timestamp" in mapping:
            return True
        for key in ("decision", "detail", "regime_summary", "status"):
            if key in mapping:
                return True
        return False

    def _normalize_entry(item: object) -> Mapping[str, object] | None:
        if not isinstance(item, Mapping):
            return None
        if not _is_entry_candidate(item):
            return None
        timestamp = _extract_entry_timestamp(item)
        if since and timestamp and timestamp < since:
            return None
        if until and timestamp and timestamp > until:
            return None
        return item

    def _iter_json_lines(handle: TextIO, path: Path) -> Iterator[Mapping[str, object]]:
        first_line = True
        for raw_line in handle:
            if first_line and raw_line.startswith("\ufeff"):
                raw_line = raw_line.lstrip("\ufeff")
                first_line = False
            elif first_line:
                first_line = False
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:  # noqa: BLE001 - CLI feedback
                raise SystemExit(
                    f"Nie udało się sparsować JSON w eksporcie autotradera {path}: {exc}"
                ) from exc
            normalized = _normalize_entry(item)
            if normalized is not None:
                yield normalized

    def _iter_json_stream(handle: TextIO, path: Path) -> Iterator[Mapping[str, object]]:
        parser = _JSONStreamEntriesParser(
            handle,
            path,
            chunk_size=_JSON_STREAM_CHUNK_SIZE,
            normalizer=_normalize_entry,
        )
        yield from parser.iter_entries()

    def _iter_path(path: Path) -> Iterator[Mapping[str, object]]:
        try:
            with _open_text_file(path) as handle:
                if _is_json_lines_path(path):
                    yield from _iter_json_lines(handle, path)
                else:
                    yield from _iter_json_stream(handle, path)
        except OSError as exc:  # noqa: BLE001 - CLI feedback
            raise SystemExit(
                f"Nie udało się odczytać eksportu autotradera {path}: {exc}"
            ) from exc

    for path in _iter_autotrade_paths(paths):
        yield from _iter_path(path)


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


def _percentile_target(count: int, percentile: float) -> tuple[int, int, float]:
    position = (count - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return lower, upper, weight


def _select_indices_from_sequences(
    sequences: Iterable[Iterable[float]],
    indices: Iterable[int],
) -> dict[int, float]:
    ordered_indices = sorted(set(index for index in indices if index >= 0))
    if not ordered_indices:
        return {}
    iterables = [iter(sequence) for sequence in sequences]
    if not iterables:
        return {}
    merged = heapq.merge(*iterables)
    results: dict[int, float] = {}
    target_iter = iter(ordered_indices)
    try:
        target = next(target_iter)
    except StopIteration:
        return {}
    for index, value in enumerate(merged):
        while target == index:
            results[target] = float(value)
            try:
                target = next(target_iter)
            except StopIteration:
                return results
        if index > ordered_indices[-1]:
            break
    return results


def _compute_percentiles_from_sequences(
    sequences: Iterable[Iterable[float]],
    percentiles: Iterable[float],
    *,
    count: int,
) -> dict[str, float]:
    unique_percentiles = sorted({p for p in percentiles if 0.0 <= p <= 1.0})
    if not unique_percentiles or count <= 0:
        return {}
    targets = {p: _percentile_target(count, p) for p in unique_percentiles}
    required_indices = [index for triple in targets.values() for index in triple[:2]]
    index_values = _select_indices_from_sequences(sequences, required_indices)
    results: dict[str, float] = {}
    for percentile in unique_percentiles:
        lower, upper, weight = targets[percentile]
        lower_value = index_values.get(lower)
        upper_value = index_values.get(upper)
        if lower_value is None or upper_value is None:
            continue
        if lower == upper:
            value = lower_value
        else:
            value = lower_value + (upper_value - lower_value) * weight
        results[f"p{int(percentile * 100):02d}"] = value
    return results


class _MetricSeries:
    __slots__ = (
        "_values",
        "_sorted_values",
        "_sorted_absolute_values",
        "_sum",
        "_sum_of_squares",
        "_min",
        "_max",
    )

    def __init__(self) -> None:
        self._values = array("d")
        self._sorted_values: array | None = None
        self._sorted_absolute_values: array | None = None
        self._sum = 0.0
        self._sum_of_squares = 0.0
        self._min: float | None = None
        self._max: float | None = None

    def _invalidate_cache(self) -> None:
        self._sorted_values = None
        self._sorted_absolute_values = None

    def append(self, value: float) -> None:
        numeric = float(value)
        self._values.append(numeric)
        self._sum += numeric
        self._sum_of_squares += numeric * numeric
        if self._min is None or numeric < self._min:
            self._min = numeric
        if self._max is None or numeric > self._max:
            self._max = numeric
        self._invalidate_cache()

    def extend(self, values: Iterable[float]) -> None:
        local_min = self._min
        local_max = self._max
        appended = False
        for value in values:
            numeric = float(value)
            self._values.append(numeric)
            self._sum += numeric
            self._sum_of_squares += numeric * numeric
            if local_min is None or numeric < local_min:
                local_min = numeric
            if local_max is None or numeric > local_max:
                local_max = numeric
            appended = True
        if appended:
            self._min = local_min
            self._max = local_max
            self._invalidate_cache()

    def __len__(self) -> int:
        return len(self._values)

    def _ensure_sorted_values(self) -> array:
        cached = self._sorted_values
        if cached is None:
            cached = array("d", sorted(self._values))
            self._sorted_values = cached
        return cached

    def _ensure_sorted_absolute_values(self) -> array:
        cached = self._sorted_absolute_values
        if cached is not None:
            return cached

        sorted_values = self._ensure_sorted_values()
        length = len(sorted_values)
        if length == 0:
            result = array("d")
            self._sorted_absolute_values = result
            return result

        split = bisect_left(sorted_values, 0.0)

        if split == 0:
            self._sorted_absolute_values = sorted_values
            return sorted_values

        if split == length:
            result = array("d", [0.0] * length)
            write_index = 0
            for index in range(length - 1, -1, -1):
                result[write_index] = -sorted_values[index]
                write_index += 1
            self._sorted_absolute_values = result
            return result

        result = array("d", [0.0] * length)
        write_index = 0
        neg_index = split - 1
        pos_index = split

        while neg_index >= 0 and pos_index < length:
            negative_abs = -sorted_values[neg_index]
            positive_abs = sorted_values[pos_index]
            if negative_abs <= positive_abs:
                result[write_index] = negative_abs
                neg_index -= 1
            else:
                result[write_index] = positive_abs
                pos_index += 1
            write_index += 1

        while neg_index >= 0:
            result[write_index] = -sorted_values[neg_index]
            neg_index -= 1
            write_index += 1

        while pos_index < length:
            result[write_index] = sorted_values[pos_index]
            pos_index += 1
            write_index += 1

        self._sorted_absolute_values = result
        return result

    def __iter__(self) -> Iterator[float]:
        return iter(self._ensure_sorted_values())

    def values(self) -> array:
        return self._ensure_sorted_values()

    def absolute_values(self) -> array:
        return self._ensure_sorted_absolute_values()

    @property
    def sum(self) -> float:
        return self._sum

    @property
    def sum_of_squares(self) -> float:
        return self._sum_of_squares

    def min_value(self) -> float | None:
        return self._min

    def max_value(self) -> float | None:
        return self._max

    def _statistics_payload(self, percentiles: Iterable[float]) -> dict[str, object]:
        count = len(self)
        if not count:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "stddev": None,
                "percentiles": {},
            }
        mean = self._sum / count
        if count > 1:
            variance = max(self._sum_of_squares / count - mean * mean, 0.0)
            stddev = math.sqrt(variance)
        else:
            stddev = 0.0
        sorted_values = self._ensure_sorted_values()
        percentiles_payload = _compute_percentiles_from_sequences(
            [sorted_values], percentiles, count=count
        )
        return {
            "count": count,
            "min": float(sorted_values[0]),
            "max": float(sorted_values[-1]),
            "mean": mean,
            "stddev": stddev,
            "percentiles": percentiles_payload,
        }

    def statistics(self, percentiles: Iterable[float]) -> dict[str, object]:
        return self._statistics_payload(percentiles)

    def suggest(self, percentile: float, *, absolute: bool = False) -> float | None:
        if not self._values:
            return None
        sequences: list[array]
        if absolute:
            sequences = [self._ensure_sorted_absolute_values()]
        else:
            sequences = [self._ensure_sorted_values()]
        percentile_key = f"p{int(percentile * 100):02d}"
        result = _compute_percentiles_from_sequences(
            sequences, [percentile], count=len(self)
        )
        return result.get(percentile_key)


def _aggregate_metric_series(
    series_list: Iterable[_MetricSeries],
    percentiles: Iterable[float],
    suggestion_percentile: float,
    *,
    absolute: bool,
    current_threshold: float | None,
) -> dict[str, object]:
    non_empty_series = [series for series in series_list if len(series)]
    if not non_empty_series:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "stddev": None,
            "percentiles": {},
            "suggested_threshold": None,
            "current_threshold": current_threshold,
        }

    total_count = sum(len(series) for series in non_empty_series)
    total_sum = sum(series.sum for series in non_empty_series)
    total_sum_of_squares = sum(series.sum_of_squares for series in non_empty_series)
    mean = total_sum / total_count
    if total_count > 1:
        variance = max(total_sum_of_squares / total_count - mean * mean, 0.0)
        stddev = math.sqrt(variance)
    else:
        stddev = 0.0

    min_value: float | None = None
    max_value: float | None = None
    for series in non_empty_series:
        series_min = series.min_value()
        if series_min is not None:
            min_value = series_min if min_value is None else min(min_value, series_min)
        series_max = series.max_value()
        if series_max is not None:
            max_value = series_max if max_value is None else max(max_value, series_max)

    sequences = [series.values() for series in non_empty_series]
    percentiles_payload = _compute_percentiles_from_sequences(
        sequences, percentiles, count=total_count
    )

    suggestion_sequences = (
        [series.absolute_values() for series in non_empty_series]
        if absolute
        else [series.values() for series in non_empty_series]
    )
    suggestion_key = f"p{int(suggestion_percentile * 100):02d}"
    suggestion_result = _compute_percentiles_from_sequences(
        suggestion_sequences, [suggestion_percentile], count=total_count
    )

    return {
        "count": total_count,
        "min": min_value,
        "max": max_value,
        "mean": mean,
        "stddev": stddev,
        "percentiles": percentiles_payload,
        "suggested_threshold": suggestion_result.get(suggestion_key),
        "current_threshold": current_threshold,
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
    mean = sum(sorted_values) / count
    if count > 1:
        sum_of_squares = sum(value * value for value in sorted_values)
        variance = max(sum_of_squares / count - mean * mean, 0.0)
        stddev = math.sqrt(variance)
    else:
        stddev = 0.0
    percentiles_payload = {
        f"p{int(p * 100):02d}": _compute_percentile(sorted_values, p)
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
    if absolute:
        target_values = sorted(abs(value) for value in values)
    else:
        target_values = sorted(values)
    return _compute_percentile(target_values, percentile)


def _numeric_buffer() -> _MetricSeries:
    return _MetricSeries()


def _build_metrics_section(
    values_map: Mapping[str, Iterable[float]],
    percentiles: Iterable[float],
    suggestion_percentile: float,
    *,
    current_risk_score: float | None,
    current_signal_thresholds: Mapping[str, float] | None = None,
) -> dict[str, dict[str, object]]:
    metrics_payload: dict[str, dict[str, object]] = {}
    for metric_name, values in values_map.items():
        absolute = metric_name in _ABSOLUTE_THRESHOLD_METRICS
        if isinstance(values, _MetricSeries):
            stats_payload = values.statistics(percentiles)
            suggested = values.suggest(suggestion_percentile, absolute=absolute)
        else:
            if isinstance(values, list):
                try:
                    all_finite = all(math.isfinite(value) for value in values)
                except TypeError:
                    all_finite = False
                if all_finite:
                    finite_values = values
                else:
                    finite_values = _finite_values(values)
                    values[:] = finite_values
            elif isinstance(values, array):
                finite_values = list(values)
            else:
                finite_values = _finite_values(values)
            stats_payload = _metric_statistics(finite_values, percentiles)
            suggested = _suggest_threshold(
                finite_values, suggestion_percentile, absolute=absolute
            )
        if metric_name == "risk_score":
            current = current_risk_score
        elif current_signal_thresholds:
            current = current_signal_thresholds.get(metric_name)
        else:
            current = None
        stats_payload["suggested_threshold"] = suggested
        stats_payload["current_threshold"] = current
        metrics_payload[metric_name] = stats_payload
    return metrics_payload


def _format_freeze_summary(summary: Mapping[str, object]) -> dict[str, object]:
    type_counts = summary.get("type_counts")
    status_counts = summary.get("status_counts")
    reason_counts = summary.get("reason_counts")
    total = int(summary.get("total") or 0)
    auto_count = 0
    manual_count = 0
    if isinstance(type_counts, Counter):
        auto_count = int(type_counts.get("auto", 0))
        manual_count = int(type_counts.get("manual", 0))
    return {
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


class _FreezeEventSampler:
    __slots__ = ("_limit", "_events", "_overflow_summary")

    def __init__(self, limit: int) -> None:
        self._limit = max(0, int(limit))
        self._events: list[dict[str, object]] = []
        self._overflow_summary: dict[str, object] = {
            "total": 0,
            "type_counts": Counter(),
            "status_counts": Counter(),
            "reason_counts": Counter(),
        }

    @property
    def limit(self) -> int:
        return self._limit

    def record(self, event: Mapping[str, object]) -> None:
        sanitized_event = {
            "status": event.get("status"),
            "type": event.get("type"),
            "reason": event.get("reason"),
            "duration": event.get("duration"),
            "risk_score": event.get("risk_score"),
        }
        if len(self._events) < self._limit:
            self._events.append(sanitized_event)
            return

        overflow = self._overflow_summary
        overflow["total"] = int(overflow.get("total", 0)) + 1
        type_counts = overflow.get("type_counts")
        status_counts = overflow.get("status_counts")
        reason_counts = overflow.get("reason_counts")
        freeze_type = sanitized_event.get("type")
        status = sanitized_event.get("status")
        reason = sanitized_event.get("reason")
        if isinstance(type_counts, Counter) and isinstance(freeze_type, str):
            type_counts[freeze_type] += 1
        if isinstance(status_counts, Counter) and isinstance(status, str):
            status_counts[status] += 1
        if isinstance(reason_counts, Counter) and isinstance(reason, str):
            reason_counts[reason] += 1

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "limit": self._limit,
            "events": list(self._events),
        }
        overflow_summary = self._overflow_summary
        if overflow_summary.get("total"):
            payload["overflow_summary"] = _format_freeze_summary(overflow_summary)
        else:
            payload["overflow_summary"] = _format_freeze_summary(
                {
                    "total": 0,
                    "type_counts": Counter(),
                    "status_counts": Counter(),
                    "reason_counts": Counter(),
                }
            )
        return payload


def _write_json(report: Mapping[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(
    groups: Iterable[dict[str, object]],
    destination: Path,
    *,
    percentiles: Iterable[str],
    global_summary: Mapping[str, object] | None = None,
) -> None:
    import csv

    destination.parent.mkdir(parents=True, exist_ok=True)
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
    ] + list(percentiles)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            primary_exchange = group["primary_exchange"]
            strategy = group["strategy"]
            metrics = group["metrics"]
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
                for percentile_key, percentile_value in metric_payload.get("percentiles", {}).items():
                    row[percentile_key] = percentile_value
                writer.writerow(row)
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
                    percentiles_payload = metric_payload.get("percentiles")
                    if isinstance(percentiles_payload, Mapping):
                        for percentile_key, percentile_value in percentiles_payload.items():
                            row[percentile_key] = percentile_value
                    writer.writerow(row)


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
        raw_values = group.get("raw_values")
        if not isinstance(raw_values, Mapping):
            continue
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
    current_threshold_sources: Mapping[str, object] | None = None,
    risk_score_override: float | None = None,
    risk_score_source: Mapping[str, object] | None = None,
    risk_threshold_sources: Iterable[str] | None = None,
    cli_risk_score_threshold: float | None = None,
    cli_risk_score: float | None = None,
    include_raw_values: bool = False,
    raw_freeze_events_mode: Literal["omit", "sample"] = "omit",
    limit_freeze_events: int | None = None,
    max_freeze_events: int | None = None,
    omit_raw_freeze_events: bool = False,
    max_raw_freeze_events: int | None = None,
) -> dict[str, object]:
    normalized_signal_thresholds: dict[str, float] | None = None
    if isinstance(current_signal_thresholds, Mapping):
        normalized_signal_thresholds = {}
        for raw_key, raw_value in current_signal_thresholds.items():
            if not isinstance(raw_key, str):
                continue
            normalized_key = _normalize_metric_key(raw_key)
            if normalized_key not in _SUPPORTED_THRESHOLD_METRICS:
                continue
            numeric = _coerce_float(raw_value)
            if numeric is None:
                raise SystemExit(
                    "Nie udało się zinterpretować bieżącego progu "
                    f"'{raw_value}' dla metryki {raw_key}"
                )
            normalized_signal_thresholds[normalized_key] = _ensure_finite_value(
                normalized_key,
                float(numeric),
                source=f"current_thresholds.{normalized_key}",
            )
    current_signal_thresholds = normalized_signal_thresholds

    symbol_map: dict[str, tuple[str, str]] = {}
    grouped_values: dict[tuple[str, str], dict[str, _MetricSeries]] = {}
    global_metric_series: dict[str, _MetricSeries] = {}
    def _empty_freeze_summary() -> dict[str, object]:
        return {
            "total": 0,
            "type_counts": Counter(),
            "status_counts": Counter(),
            "reason_counts": Counter(),
        }

    freeze_summaries: dict[tuple[str, str], dict[str, object]] = defaultdict(
        _empty_freeze_summary
    )
    freeze_event_collections: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    display_names: dict[tuple[str, str], tuple[str, str]] = {}
    aggregated_freeze_summary = _empty_freeze_summary()
    aggregated_freeze_events: list[dict[str, object]] = []
    freeze_event_overflow_summaries: dict[tuple[str, str], dict[str, object]] = defaultdict(
        _empty_freeze_summary
    )
    aggregated_freeze_overflow = _empty_freeze_summary()
    freeze_event_samplers: dict[tuple[str, str], _FreezeEventSampler] = {}
    aggregated_freeze_sampler: _FreezeEventSampler | None = None
    raw_freeze_event_display_limit: int | None

    def _sanitize_optional_limit(raw_limit: object) -> int | None:
        if raw_limit is None:
            return None
        try:
            return max(0, int(raw_limit))
        except (TypeError, ValueError):
            return 0

    raw_freeze_event_display_limit = _sanitize_optional_limit(max_raw_freeze_events)
    normalized_freeze_mode = str(raw_freeze_events_mode or "omit").strip().lower()
    if normalized_freeze_mode not in {"omit", "sample"}:
        normalized_freeze_mode = "omit"
    omit_raw_freeze_events = bool(omit_raw_freeze_events)
    sampling_freeze_events = normalized_freeze_mode == "sample" and not omit_raw_freeze_events
    if sampling_freeze_events:
        if limit_freeze_events is None:
            sampler_limit = 25
        else:
            sampler_limit = max(0, int(limit_freeze_events))
        aggregated_freeze_sampler = _FreezeEventSampler(sampler_limit)
    else:
        sampler_limit = 0
    if max_freeze_events is None:
        freeze_event_limit: int | None = None
    else:
        try:
            freeze_event_limit = max(0, int(max_freeze_events))
        except (TypeError, ValueError):
            freeze_event_limit = 0
    journal_count = 0
    autotrade_count = 0

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

    def _ensure_metrics(key: tuple[str, str]) -> dict[str, _MetricSeries]:
        metrics = grouped_values.get(key)
        if metrics is None:
            metrics = {}
            grouped_values[key] = metrics
        return metrics

    def _ensure_global_series(metric_name: str) -> _MetricSeries:
        series = global_metric_series.get(metric_name)
        if series is None:
            series = _numeric_buffer()
            global_metric_series[metric_name] = series
        return series

    def _metric_buffer(key: tuple[str, str], metric_name: str) -> _MetricSeries:
        metrics = _ensure_metrics(key)
        buffer = metrics.get(metric_name)
        if buffer is None:
            buffer = _numeric_buffer()
            metrics[metric_name] = buffer
        return buffer

    def _record_metric_value(
        key: tuple[str, str],
        metric_name: str,
        numeric: float,
    ) -> None:
        numeric_value = float(numeric)
        values = _metric_buffer(key, metric_name)
        values.append(numeric_value)
        global_series = _ensure_global_series(metric_name)
        global_series.append(numeric_value)
        if _METRIC_APPEND_OBSERVER is not None:
            _METRIC_APPEND_OBSERVER(key, metric_name, len(values))

    def _serialize_counter(counter: Counter | object) -> dict[str, int]:
        if not isinstance(counter, Counter):
            return {}
        return {
            str(key): int(count)
            for key, count in sorted(counter.items(), key=lambda item: item[0])
        }

    def _freeze_event_record(
        status: str,
        freeze_type: str,
        reason: str,
        duration: float | None,
        risk_score: float | None,
    ) -> dict[str, object]:
        record: dict[str, object] = {
            "status": status,
            "type": freeze_type,
            "reason": reason,
        }
        if duration is not None:
            record["duration"] = duration
        if risk_score is not None:
            record["risk_score"] = risk_score
        return record

    def _increment_freeze_summary(
        summary: dict[str, object],
        status: str,
        freeze_type: str,
        reason: str,
    ) -> None:
        summary["total"] = int(summary.get("total", 0)) + 1
        type_counts = summary.get("type_counts")
        status_counts = summary.get("status_counts")
        reason_counts = summary.get("reason_counts")
        if isinstance(type_counts, Counter):
            type_counts[freeze_type] += 1
        if isinstance(status_counts, Counter):
            status_counts[status] += 1
        if isinstance(reason_counts, Counter):
            reason_counts[reason] += 1

    def _clone_formatted_freeze_summary(
        summary_payload: Mapping[str, object] | None,
    ) -> dict[str, object]:
        summary = _empty_freeze_summary()
        if not isinstance(summary_payload, Mapping):
            return summary
        summary["total"] = int(summary_payload.get("total") or 0)
        type_counts = summary.get("type_counts")
        if isinstance(type_counts, Counter):
            auto_count = summary_payload.get("auto")
            manual_count = summary_payload.get("manual")
            if isinstance(auto_count, (int, float)):
                type_counts["auto"] = int(auto_count)
            if isinstance(manual_count, (int, float)):
                type_counts["manual"] = int(manual_count)
        statuses = summary_payload.get("statuses")
        if isinstance(statuses, Iterable):
            for item in statuses:
                if not isinstance(item, Mapping):
                    continue
                status = item.get("status")
                count = item.get("count")
                if status is None or count is None:
                    continue
                try:
                    numeric_count = int(count)
                except (TypeError, ValueError):
                    continue
                summary_status_counts = summary.get("status_counts")
                if isinstance(summary_status_counts, Counter) and isinstance(status, str):
                    summary_status_counts[status] += numeric_count
        reasons = summary_payload.get("reasons")
        if isinstance(reasons, Iterable):
            for item in reasons:
                if not isinstance(item, Mapping):
                    continue
                reason = item.get("reason")
                count = item.get("count")
                if reason is None or count is None:
                    continue
                try:
                    numeric_count = int(count)
                except (TypeError, ValueError):
                    continue
                summary_reason_counts = summary.get("reason_counts")
                if isinstance(summary_reason_counts, Counter) and isinstance(reason, str):
                    summary_reason_counts[reason] += numeric_count
        return summary

    def _limit_raw_freeze_payload(
        payload: Mapping[str, object] | None,
    ) -> dict[str, object] | None:
        if not isinstance(payload, Mapping):
            return None

        normalized_limit = raw_freeze_event_display_limit
        if normalized_limit is not None:
            try:
                normalized_limit = max(0, int(normalized_limit))
            except (TypeError, ValueError):
                normalized_limit = 0
            if normalized_limit == 0:
                return None

        events_payload = payload.get("events")
        events: list[dict[str, object]] = []
        if isinstance(events_payload, Iterable) and not isinstance(events_payload, (str, bytes)):
            for item in events_payload:
                if not isinstance(item, Mapping):
                    continue
                events.append(
                    {
                        "status": item.get("status"),
                        "type": item.get("type"),
                        "reason": item.get("reason"),
                        "duration": item.get("duration"),
                        "risk_score": item.get("risk_score"),
                    }
                )

        original_limit = _sanitize_optional_limit(payload.get("limit"))
        if normalized_limit is None:
            overflow_summary_payload = payload.get("overflow_summary")
            if isinstance(overflow_summary_payload, Mapping):
                overflow_summary = dict(overflow_summary_payload)
            else:
                overflow_summary = _format_freeze_summary(_empty_freeze_summary())
            if original_limit is None:
                computed_limit = len(events)
            else:
                computed_limit = original_limit
            return {
                "limit": computed_limit,
                "events": [dict(event) for event in events],
                "overflow_summary": overflow_summary,
            }

        trimmed_events = events[:normalized_limit]
        overflow_summary_source = payload.get("overflow_summary")
        overflow_summary = _clone_formatted_freeze_summary(
            overflow_summary_source if isinstance(overflow_summary_source, Mapping) else None
        )
        for event in events[normalized_limit:]:
            status = str(event.get("status") or "unknown")
            freeze_type = str(event.get("type") or "manual")
            reason = str(event.get("reason") or "unknown")
            _increment_freeze_summary(overflow_summary, status, freeze_type, reason)
        formatted_overflow = _format_freeze_summary(overflow_summary)
        if original_limit is None:
            computed_limit = normalized_limit
        else:
            computed_limit = min(original_limit, normalized_limit)
        computed_limit = max(computed_limit, len(trimmed_events))
        return {
            "limit": computed_limit,
            "events": [dict(event) for event in trimmed_events],
            "overflow_summary": formatted_overflow,
        }

    def _record_freeze(
        key: tuple[str, str],
        payload: Mapping[str, object],
    ) -> None:
        summary = freeze_summaries[key]
        status = str(payload.get("status") or "unknown")
        freeze_type = str(payload.get("type") or "manual")
        reason = str(payload.get("reason") or "unknown")
        duration_value = payload.get("duration")
        numeric_duration: float | None = None
        if duration_value is not None:
            try:
                numeric_duration = float(duration_value)
            except (TypeError, ValueError):
                numeric_duration = None
        _increment_freeze_summary(summary, status, freeze_type, reason)
        _increment_freeze_summary(aggregated_freeze_summary, status, freeze_type, reason)
        _ensure_metrics(key)
        if numeric_duration is not None and math.isfinite(numeric_duration):
            _record_metric_value(key, "risk_freeze_duration", numeric_duration)
        raw_risk_score = payload.get("risk_score")
        numeric_risk_score: float | None = None
        if raw_risk_score is not None:
            try:
                numeric_risk_score = float(raw_risk_score)
            except (TypeError, ValueError):
                numeric_risk_score = None
            if numeric_risk_score is not None and not math.isfinite(numeric_risk_score):
                numeric_risk_score = None
        if sampling_freeze_events:
            sampler = freeze_event_samplers.get(key)
            if sampler is None:
                sampler = _FreezeEventSampler(sampler_limit)
                freeze_event_samplers[key] = sampler
            sampler.record(
                {
                    "status": status,
                    "type": freeze_type,
                    "reason": reason,
                    "duration": numeric_duration,
                    "risk_score": numeric_risk_score,
                }
            )
            if aggregated_freeze_sampler is not None:
                aggregated_freeze_sampler.record(
                    {
                        "status": status,
                        "type": freeze_type,
                        "reason": reason,
                        "duration": numeric_duration,
                        "risk_score": numeric_risk_score,
                    }
                )
        freeze_event_payload = _freeze_event_record(
            status,
            freeze_type,
            reason,
            numeric_duration if numeric_duration is not None and math.isfinite(numeric_duration) else None,
            numeric_risk_score,
        )
        if freeze_event_limit is None:
            freeze_event_collections[key].append(dict(freeze_event_payload))
            aggregated_freeze_events.append(dict(freeze_event_payload))
            return

        if freeze_event_limit > 0:
            group_events = freeze_event_collections[key]
            if len(group_events) < freeze_event_limit:
                group_events.append(dict(freeze_event_payload))
            else:
                overflow_summary = freeze_event_overflow_summaries[key]
                _increment_freeze_summary(overflow_summary, status, freeze_type, reason)
            if len(aggregated_freeze_events) < freeze_event_limit:
                aggregated_freeze_events.append(dict(freeze_event_payload))
            else:
                _increment_freeze_summary(
                    aggregated_freeze_overflow, status, freeze_type, reason
                )
            return

        overflow_summary = freeze_event_overflow_summaries[key]
        _increment_freeze_summary(overflow_summary, status, freeze_type, reason)
        _increment_freeze_summary(aggregated_freeze_overflow, status, freeze_type, reason)

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

        for freeze_payload in _iter_freeze_events(entry):
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
                current_risk_score = _ensure_finite_value(
                    "risk_score",
                    float(value),
                    source=str(config_path),
                )
    else:
        thresholds = load_risk_thresholds()
        value = _extract_risk_score_threshold(thresholds)
        if value is not None:
            current_risk_score = _ensure_finite_value(
                "risk_score",
                float(value),
                source="load_risk_thresholds()",
            )

    override_source: str | None = None
    if isinstance(risk_score_source, Mapping):
        raw_source = risk_score_source.get("source")
        if raw_source is not None:
            override_source = str(raw_source)
        else:
            raw_kind = risk_score_source.get("kind")
            if isinstance(raw_kind, str) and raw_kind:
                override_source = f"current_thresholds.{raw_kind}"

    if risk_score_override is not None:
        source_hint = override_source or "CLI risk_score_override"
        current_risk_score = _ensure_finite_value(
            "risk_score",
            float(risk_score_override),
            source=source_hint,
        )
    elif current_risk_score is None and isinstance(risk_score_source, Mapping):
        raw_value = _coerce_float(risk_score_source.get("value"))
        if raw_value is not None:
            source_hint = override_source or "current_thresholds.value"
            current_risk_score = _ensure_finite_value(
                "risk_score",
                float(raw_value),
                source=source_hint,
            )

    if cli_risk_score is not None:
        current_risk_score = float(cli_risk_score)

    if cli_risk_score is not None:
        current_risk_score = float(cli_risk_score)

    groups: list[dict[str, object]] = []
    all_keys = set(grouped_values.keys()) | set(freeze_summaries.keys()) | set(display_names.keys())

    for exchange, strategy in sorted(all_keys):
        metrics = grouped_values.get((exchange, strategy))
        if metrics is None:
            metrics = {}
        metrics_payload = _build_metrics_section(
            metrics,
            percentiles,
            suggestion_percentile,
            current_risk_score=current_risk_score,
            current_signal_thresholds=current_signal_thresholds,
        )
        freeze_summary = freeze_summaries.get((exchange, strategy)) or {
            "total": 0,
            "type_counts": Counter(),
            "status_counts": Counter(),
            "reason_counts": Counter(),
        }
        freeze_summary_payload = _format_freeze_summary(freeze_summary)
        has_metrics = any(values for values in metrics.values())
        has_freeze = int(freeze_summary.get("total") or 0) > 0
        if not (has_metrics or has_freeze):
            continue
        display_exchange, display_strategy = display_names.get(
            (exchange, strategy), (exchange, strategy)
        )
        group_payload: dict[str, object] = {
            "primary_exchange": display_exchange,
            "strategy": display_strategy,
            "metrics": metrics_payload,
            "freeze_summary": freeze_summary_payload,
        }
        if freeze_event_limit is None:
            events_sample = freeze_event_collections.get((exchange, strategy))
            if events_sample:
                group_payload["freeze_events"] = {
                    "mode": "all",
                    "events": [dict(item) for item in events_sample],
                    "total": int(freeze_summary.get("total") or 0),
                    "type_counts": _serialize_counter(freeze_summary.get("type_counts")),
                    "status_counts": _serialize_counter(freeze_summary.get("status_counts")),
                    "reason_counts": _serialize_counter(freeze_summary.get("reason_counts")),
                }
        else:
            events_sample = freeze_event_collections.get((exchange, strategy), [])
            overflow_summary = freeze_event_overflow_summaries.get((exchange, strategy))
            if overflow_summary is None:
                overflow_summary = _empty_freeze_summary()
            group_payload["freeze_events"] = {
                "mode": "limit",
                "limit": freeze_event_limit,
                "events": [dict(item) for item in events_sample],
                "total": int(freeze_summary.get("total") or 0),
                "type_counts": _serialize_counter(freeze_summary.get("type_counts")),
                "status_counts": _serialize_counter(freeze_summary.get("status_counts")),
                "reason_counts": _serialize_counter(freeze_summary.get("reason_counts")),
                "overflow_summary": _format_freeze_summary(overflow_summary),
            }
        if sampling_freeze_events:
            sampler = freeze_event_samplers.get((exchange, strategy))
            if sampler is not None:
                payload = _limit_raw_freeze_payload(sampler.to_payload())
                if payload is not None:
                    group_payload["raw_freeze_events"] = payload
        if include_raw_values:
            group_payload["raw_values"] = {
                metric: (
                    values if isinstance(values, list) else list(values)
                )
                for metric, values in metrics.items()
            }
        groups.append(group_payload)

    global_metrics: dict[str, dict[str, object]] = {}
    for metric_name, series in global_metric_series.items():
        if metric_name == "risk_score":
            current_threshold = current_risk_score
        elif current_signal_thresholds:
            current_threshold = current_signal_thresholds.get(metric_name)
        else:
            current_threshold = None
        global_metrics[metric_name] = _aggregate_metric_series(
            [series],
            percentiles,
            suggestion_percentile,
            absolute=metric_name in _ABSOLUTE_THRESHOLD_METRICS,
            current_threshold=current_threshold,
        )
    global_summary: dict[str, object] = {
        "metrics": global_metrics,
        "freeze_summary": _format_freeze_summary(aggregated_freeze_summary),
    }
    aggregated_raw_freeze_payload: dict[str, object] | None = None
    if freeze_event_limit is None:
        if aggregated_freeze_events:
            global_summary["freeze_events"] = {
                "mode": "all",
                "events": [dict(item) for item in aggregated_freeze_events],
                "total": int(aggregated_freeze_summary.get("total") or 0),
                "type_counts": _serialize_counter(aggregated_freeze_summary.get("type_counts")),
                "status_counts": _serialize_counter(aggregated_freeze_summary.get("status_counts")),
                "reason_counts": _serialize_counter(aggregated_freeze_summary.get("reason_counts")),
            }
    else:
        global_summary["freeze_events"] = {
            "mode": "limit",
            "limit": freeze_event_limit,
            "events": [dict(item) for item in aggregated_freeze_events],
            "total": int(aggregated_freeze_summary.get("total") or 0),
            "type_counts": _serialize_counter(aggregated_freeze_summary.get("type_counts")),
            "status_counts": _serialize_counter(aggregated_freeze_summary.get("status_counts")),
            "reason_counts": _serialize_counter(aggregated_freeze_summary.get("reason_counts")),
            "overflow_summary": _format_freeze_summary(aggregated_freeze_overflow),
        }
    if sampling_freeze_events and aggregated_freeze_sampler is not None:
        aggregated_raw_freeze_payload = _limit_raw_freeze_payload(
            aggregated_freeze_sampler.to_payload()
        )
        if aggregated_raw_freeze_payload is not None:
            global_summary["raw_freeze_events"] = aggregated_raw_freeze_payload
    if include_raw_values:
        global_summary["raw_values"] = {
            metric: list(series.values())
            for metric, series in global_metric_series.items()
        }

    current_threshold_files: list[str] = []
    current_threshold_inline: dict[str, float] = {}
    risk_threshold_inline: dict[str, float] = {}
    risk_threshold_files_extra: list[str] = []
    risk_score_metadata_payload: dict[str, object] | None = None
    if isinstance(current_threshold_sources, Mapping):
        raw_files = current_threshold_sources.get("files")
        if isinstance(raw_files, Iterable) and not isinstance(raw_files, (str, bytes)):
            seen_files: set[str] = set()
            for item in raw_files:
                item_str = str(item)
                if item_str in seen_files:
                    continue
                seen_files.add(item_str)
                current_threshold_files.append(item_str)
        raw_inline = current_threshold_sources.get("inline")
        if isinstance(raw_inline, Mapping):
            for key, value in raw_inline.items():
                if not isinstance(key, str):
                    continue
                numeric = _coerce_float(value)
                if numeric is None:
                    continue
                normalized_key = _normalize_metric_key(key)
                validated_value = _normalize_and_validate_threshold(
                    normalized_key,
                    float(numeric),
                    source="current_thresholds.inline",
                )
                current_threshold_inline[normalized_key] = validated_value
        raw_risk_inline = current_threshold_sources.get("risk_inline")
        if isinstance(raw_risk_inline, Mapping):
            for key, value in raw_risk_inline.items():
                if not isinstance(key, str):
                    continue
                numeric = _coerce_float(value)
                if numeric is None:
                    continue
                normalized_key = _normalize_metric_key(key)
                validated_value = _normalize_and_validate_threshold(
                    normalized_key,
                    float(numeric),
                    source="risk_thresholds.inline",
                )
                risk_threshold_inline[normalized_key] = validated_value
        raw_risk_files = current_threshold_sources.get("risk_files")
        if isinstance(raw_risk_files, Iterable) and not isinstance(raw_risk_files, (str, bytes)):
            seen_risk_files: set[str] = set()
            for item in raw_risk_files:
                item_str = str(item)
                if item_str in seen_risk_files:
                    continue
                seen_risk_files.add(item_str)
                risk_threshold_files_extra.append(item_str)
        raw_risk_source = current_threshold_sources.get("risk_score_source")
        if isinstance(raw_risk_source, Mapping):
            metadata: dict[str, object] = {}
            raw_kind = raw_risk_source.get("kind")
            if isinstance(raw_kind, str):
                metadata["kind"] = raw_kind
            raw_source = raw_risk_source.get("source")
            if raw_source is not None:
                metadata["source"] = str(raw_source)
            raw_value = _coerce_float(raw_risk_source.get("value"))
            if raw_value is not None:
                metadata["value"] = _normalize_and_validate_threshold(
                    "risk_score",
                    float(raw_value),
                    source="current_thresholds.risk_score_source",
                )
            if metadata:
                risk_score_metadata_payload = metadata

    combined_risk_files: list[str] = []
    seen_combined_risk_files: set[str] = set()
    for path in risk_threshold_paths:
        path_str = str(path)
        if path_str in seen_combined_risk_files:
            continue
        seen_combined_risk_files.add(path_str)
        combined_risk_files.append(path_str)
    for path_str in risk_threshold_files_extra:
        if path_str in seen_combined_risk_files:
            continue
        seen_combined_risk_files.add(path_str)
        combined_risk_files.append(path_str)

    current_thresholds_payload: dict[str, object] = {
        "files": current_threshold_files,
        "inline": current_threshold_inline,
    }
    if risk_score_metadata_payload is not None:
        current_thresholds_payload["risk_score"] = risk_score_metadata_payload

    sources_payload: dict[str, object] = {
        "journal_events": journal_count,
        "autotrade_entries": autotrade_count,
        "current_thresholds": current_thresholds_payload,
        "risk_thresholds": {
            "files": combined_risk_files,
            "inline": risk_threshold_inline,
        },
    }

    signal_sources_payload: dict[str, object] = {}
    if current_threshold_files:
        signal_sources_payload["files"] = list(current_threshold_files)
    if current_threshold_inline:
        signal_sources_payload["inline"] = dict(current_threshold_inline)
    if signal_sources_payload:
        sources_payload["current_signal_thresholds"] = signal_sources_payload
    if risk_threshold_paths:
        sources_payload["risk_threshold_files"] = [str(path) for path in risk_threshold_paths]
    if cli_risk_score is not None:
        sources_payload["risk_score_override"] = float(cli_risk_score)
    if aggregated_raw_freeze_payload is not None:
        raw_freeze_sources_payload: dict[str, object] = {"mode": "sample"}
        limit_value = aggregated_raw_freeze_payload.get("limit")
        if isinstance(limit_value, (int, float)):
            raw_freeze_sources_payload["limit"] = int(limit_value)
        requested_limit = raw_freeze_event_display_limit
        if isinstance(requested_limit, int):
            raw_freeze_sources_payload["requested_limit"] = requested_limit
        overflow_summary = aggregated_raw_freeze_payload.get("overflow_summary")
        if isinstance(overflow_summary, Mapping):
            raw_freeze_sources_payload["overflow_summary"] = dict(overflow_summary)
        sources_payload["raw_freeze_events"] = raw_freeze_sources_payload
    else:
        omit_payload: dict[str, object] = {"mode": "omit"}
        if isinstance(raw_freeze_event_display_limit, int):
            omit_payload["requested_limit"] = raw_freeze_event_display_limit
            if raw_freeze_event_display_limit == 0:
                omit_payload["reason"] = "limit_zero"
        if omit_raw_freeze_events:
            omit_payload["reason"] = "explicit_omit"
        elif not sampling_freeze_events and "reason" not in omit_payload:
            if limit_freeze_events is None:
                omit_payload["reason"] = "sampling_disabled"
            else:
                omit_payload["reason"] = "no_samples"
        sources_payload["raw_freeze_events"] = omit_payload
    if freeze_event_limit is None:
        sources_payload["freeze_events"] = {"mode": "all"}
    else:
        sources_payload["freeze_events"] = {
            "mode": "limit",
            "limit": freeze_event_limit,
            "overflow_summary": _format_freeze_summary(aggregated_freeze_overflow),
        }

    return {
        "schema": "stage6.autotrade.threshold_calibration",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "percentiles": [f"p{int(p * 100):02d}" for p in percentiles],
        "suggestion_percentile": suggestion_percentile,
        "filters": {
            "since": since.isoformat() if since else None,
            "until": until.isoformat() if until else None,
        },
        "groups": groups,
        "global_summary": global_summary,
        "sources": sources_payload,
    }


def _resolve_freeze_event_limit(
    *,
    limit_freeze_events: int | None,
    raw_freeze_events_mode: str | None,
    raw_freeze_events_limit: int | None,
) -> int | None:
    """Wybiera limit blokad na podstawie nowych i legacyjnych flag CLI."""

    if limit_freeze_events is not None:
        return int(limit_freeze_events)

    if not raw_freeze_events_mode:
        return None

    normalized_mode = str(raw_freeze_events_mode).strip().lower()
    if normalized_mode != "sample":
        return None

    if raw_freeze_events_limit is None:
        return 25

    return int(raw_freeze_events_limit)


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
        help=(
            "Ścieżki do plików JSONL (również skompresowanych .gz) lub katalogów "
            "z TradingDecisionJournal"
        ),
    )
    parser.add_argument(
        "--autotrade-export",
        required=True,
        nargs="+",
        help=(
            "Pliki JSON/JSONL (również skompresowane .gz) wygenerowane przez "
            "export_risk_evaluations lub eksport statusów autotradera"
        ),
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
        "--limit-freeze-events",
        type=int,
        help=(
            "Opcjonalnie ogranicz liczbę zdarzeń blokad zapisywanych w sekcji "
            "raw_freeze_events; gdy ustawione dodaje próbkę pierwszych N wpisów oraz "
            "podsumowanie reszty. Bez parametru sekcja jest pomijana."
        ),
    )
    parser.add_argument(
        "--max-raw-freeze-events",
        type=int,
        help=(
            "Ogranicz liczbę zdarzeń prezentowanych w sekcji raw_freeze_events. "
            "Pozwala skrócić próbkę bez zmiany agregatów ani źródeł danych."
        ),
    )
    parser.add_argument(
        "--omit-raw-freeze-events",
        action="store_true",
        help=(
            "Pomija sekcję raw_freeze_events nawet, gdy dostępny jest limit próbki. "
            "Zawsze pozostawia wyłącznie zagregowane statystyki."
        ),
    )
    parser.add_argument(
        "--freeze-events-limit",
        type=int,
        help=(
            "Ogranicz liczbę zdarzeń zapisywanych w sekcjach freeze_events; "
            "zachowuje jedynie pierwsze N wpisów wraz z agregatami. "
            "Wartość 0 pozostawia wyłącznie podsumowania bez listy zdarzeń."
        ),
    )
    parser.add_argument(
        "--raw-freeze-events",
        choices=("omit", "sample"),
        default="omit",
        help=(
            "[Przestarzałe] Steruje sekcją raw_freeze_events w raporcie: 'sample' dodaje próbkę "
            "zdarzeń wraz z podsumowaniem reszty, 'omit' pozostawia tylko statystyki. "
            "Użyj --limit-freeze-events, aby włączyć próbkę."
        ),
    )
    parser.add_argument(
        "--raw-freeze-events-limit",
        type=int,
        default=25,
        help=(
            "[Przestarzałe] Maksymalna liczba zdarzeń blokad zapisywana w próbce dla każdej "
            "kombinacji giełda/strategia. Zastąpione przez --limit-freeze-events."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    percentiles = _parse_percentiles(args.percentiles)
    if not (0.0 < args.suggestion_percentile < 1.0):
        raise SystemExit("Percentyl sugerowanego progu musi być w przedziale (0, 1)")

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
        provided_risk_score,
        current_threshold_sources_payload,
    ) = _load_current_signal_thresholds(args.current_threshold)

    risk_score_source_metadata: Mapping[str, object] | None = None
    risk_score_override: float | None = None
    signal_thresholds_payload: Mapping[str, float] | None = current_signal_thresholds
    if isinstance(current_threshold_sources_payload, Mapping):
        raw_risk_source = current_threshold_sources_payload.get("risk_score_source")
        if isinstance(raw_risk_source, Mapping):
            risk_score_source_metadata = raw_risk_source
            raw_kind = raw_risk_source.get("kind")
            if provided_risk_score is not None and raw_kind == "inline":
                risk_score_override = provided_risk_score
            elif provided_risk_score is not None and raw_kind == "file":
                existing = dict(current_signal_thresholds or {})
                existing["risk_score"] = provided_risk_score
                signal_thresholds_payload = existing

    limit_freeze_events = _resolve_freeze_event_limit(
        limit_freeze_events=args.limit_freeze_events,
        raw_freeze_events_mode=getattr(args, "raw_freeze_events", None),
        raw_freeze_events_limit=getattr(args, "raw_freeze_events_limit", None),
    )

    report = _generate_report(
        journal_events=journal_events,
        autotrade_entries=autotrade_entries,
        percentiles=percentiles,
        suggestion_percentile=args.suggestion_percentile,
        since=since,
        until=until,
        current_signal_thresholds=signal_thresholds_payload,
        current_threshold_sources=current_threshold_sources_payload,
        risk_score_override=risk_score_override,
        risk_score_source=risk_score_source_metadata,
        risk_threshold_sources=args.risk_thresholds,
        include_raw_values=bool(args.plot_dir),
        raw_freeze_events_mode="sample" if limit_freeze_events is not None else "omit",
        limit_freeze_events=limit_freeze_events,
        max_freeze_events=getattr(args, "freeze_events_limit", None),
        omit_raw_freeze_events=bool(getattr(args, "omit_raw_freeze_events", False)),
        max_raw_freeze_events=getattr(args, "max_raw_freeze_events", None),
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
