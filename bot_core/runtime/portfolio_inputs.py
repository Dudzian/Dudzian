"""Ładowanie artefaktów Stage6 dla PortfolioGovernora w runtime."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from bot_core.observability.slo import SLOStatus
from bot_core.risk import StressOverrideRecommendation

__all__ = [
    "build_slo_status_provider",
    "build_stress_override_provider",
    "load_slo_statuses",
    "load_stress_overrides",
]


_LOGGER = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _candidate_paths(path: Path, fallbacks: Sequence[Path]) -> Iterable[Path]:
    if path.is_absolute():
        yield path
        return
    seen: set[str] = set()
    for candidate in (path,):
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            yield candidate
    for base in fallbacks:
        candidate = Path(base) / path
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            yield candidate


def _parse_datetime(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _to_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _metadata_to_floats(payload: Mapping[str, object] | None) -> dict[str, float]:
    result: dict[str, float] = {}
    if not payload:
        return result
    for key, value in payload.items():
        if isinstance(value, (int, float)):
            result[str(key)] = float(value)
    return result


def _ensure_fresh(path: Path, *, max_age: timedelta | None) -> bool:
    if max_age is None:
        return True
    try:
        stat = path.stat()
    except FileNotFoundError:
        return False
    mtime = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
    age = _utcnow() - mtime
    if age > max_age:
        minutes = age.total_seconds() / 60.0
        _LOGGER.warning(
            "Raport %s jest przestarzały (wiek %.1f min > %.1f min)",
            path,
            minutes,
            max_age.total_seconds() / 60.0,
        )
        return False
    return True


def load_slo_statuses(
    report_path: Path,
    *,
    max_age: timedelta | None = None,
) -> dict[str, SLOStatus]:
    """Ładuje statusy SLO Stage6 z raportu JSON."""

    if not _ensure_fresh(report_path, max_age=max_age):
        return {}
    try:
        content = report_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except Exception:  # pragma: no cover - błędy IO logowane niżej
        _LOGGER.exception("Błąd odczytu raportu SLO %s", report_path)
        return {}

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        _LOGGER.exception("Niepoprawny JSON raportu SLO w %s", report_path)
        return {}

    results = payload.get("results")
    if not isinstance(results, Mapping):
        return {}

    statuses: dict[str, SLOStatus] = {}
    for name, entry in results.items():
        if not isinstance(entry, Mapping):
            continue
        indicator = str(entry.get("indicator", ""))
        target = _to_float(entry.get("target")) or 0.0
        comparison = str(entry.get("comparison", "<="))
        value = _to_float(entry.get("value"))
        warning = _to_float(entry.get("warning_threshold"))
        error_budget = _to_float(entry.get("error_budget_pct"))
        status_text = str(entry.get("status", "unknown"))
        severity = str(entry.get("severity", "warning"))
        metadata = _metadata_to_floats(entry.get("metadata") if isinstance(entry.get("metadata"), Mapping) else None)
        statuses[str(name)] = SLOStatus(
            name=str(name),
            indicator=indicator,
            value=value,
            target=target,
            comparison=comparison,
            status=status_text,
            severity=severity,
            warning_threshold=warning,
            error_budget_pct=error_budget,
            window_start=_parse_datetime(entry.get("window_start")),
            window_end=_parse_datetime(entry.get("window_end")),
            sample_size=int(entry.get("sample_size", 0) or 0),
            reason=(
                str(entry.get("reason"))
                if entry.get("reason") not in (None, "")
                else None
            ),
            metadata=metadata,
        )
    return statuses


def load_stress_overrides(
    report_path: Path,
    *,
    max_age: timedelta | None = None,
) -> tuple[StressOverrideRecommendation, ...]:
    """Ładuje rekomendacje override ze Stage6 Stress Lab."""

    if not _ensure_fresh(report_path, max_age=max_age):
        return ()
    try:
        content = report_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ()
    except Exception:  # pragma: no cover
        _LOGGER.exception("Błąd odczytu raportu Stress Lab %s", report_path)
        return ()

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        _LOGGER.exception("Niepoprawny JSON raportu Stress Lab w %s", report_path)
        return ()

    overrides_section = payload.get("overrides")
    if not isinstance(overrides_section, Sequence):
        return ()

    overrides: list[StressOverrideRecommendation] = []
    for entry in overrides_section:
        if not isinstance(entry, Mapping):
            continue
        reason = entry.get("reason")
        if reason in (None, ""):
            continue
        tags_value = entry.get("tags")
        tags: tuple[str, ...]
        if isinstance(tags_value, Sequence) and not isinstance(tags_value, (str, bytes)):
            tags = tuple(str(tag) for tag in tags_value)
        else:
            tags = ()
        overrides.append(
            StressOverrideRecommendation(
                severity=str(entry.get("severity", "warning")),
                reason=str(reason),
                symbol=(
                    str(entry.get("symbol"))
                    if entry.get("symbol") not in (None, "")
                    else None
                ),
                risk_budget=(
                    str(entry.get("risk_budget"))
                    if entry.get("risk_budget") not in (None, "")
                    else None
                ),
                weight_multiplier=_to_float(entry.get("weight_multiplier")),
                min_weight=_to_float(entry.get("min_weight")),
                max_weight=_to_float(entry.get("max_weight")),
                tags=tags,
                force_rebalance=bool(entry.get("force_rebalance", False)),
            )
        )
    return tuple(overrides)


def build_slo_status_provider(
    path: str | Path,
    *,
    fallback_directories: Sequence[Path] = (),
    max_age: timedelta | None = None,
) -> Callable[[], Mapping[str, SLOStatus]]:
    target = Path(path)
    fallbacks = tuple(Path(item) for item in fallback_directories)

    def _provider() -> Mapping[str, SLOStatus]:
        for candidate in _candidate_paths(target, fallbacks):
            try:
                statuses = load_slo_statuses(candidate, max_age=max_age)
            except Exception:  # pragma: no cover - log diagnostyczny
                _LOGGER.exception("Błąd ładowania statusów SLO z %s", candidate)
                return {}
            if statuses:
                return statuses
        _LOGGER.debug(
            "Nie znaleziono aktualnego raportu SLO dla ścieżki %s (fallbacki: %s)",
            target,
            ", ".join(str(item) for item in fallbacks) or "brak",
        )
        return {}

    return _provider


def build_stress_override_provider(
    path: str | Path,
    *,
    fallback_directories: Sequence[Path] = (),
    max_age: timedelta | None = None,
) -> Callable[[], Sequence[StressOverrideRecommendation]]:
    target = Path(path)
    fallbacks = tuple(Path(item) for item in fallback_directories)

    def _provider() -> Sequence[StressOverrideRecommendation]:
        for candidate in _candidate_paths(target, fallbacks):
            try:
                overrides = load_stress_overrides(candidate, max_age=max_age)
            except Exception:  # pragma: no cover - log diagnostyczny
                _LOGGER.exception("Błąd ładowania override'ów Stress Lab z %s", candidate)
                return ()
            if overrides:
                return overrides
        _LOGGER.debug(
            "Nie znaleziono aktualnego raportu Stress Lab dla ścieżki %s (fallbacki: %s)",
            target,
            ", ".join(str(item) for item in fallbacks) or "brak",
        )
        return ()

    return _provider
