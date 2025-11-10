"""Ładowanie artefaktów Stage6 dla PortfolioGovernora w runtime."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, cast

from bot_core.observability.slo import SLOStatus
from bot_core.risk import StressOverrideRecommendation

__all__ = [
    "build_slo_status_provider",
    "build_stress_override_provider",
    "load_slo_statuses",
    "load_stress_overrides",
    "load_portfolio_stress_summary",
    "build_portfolio_stress_provider",
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


def load_portfolio_stress_summary(
    report_path: Path,
    *,
    max_age: timedelta | None = None,
) -> Mapping[str, Any]:
    """Ładuje skondensowany raport portfolio_stress (scenariusze, drawdowny)."""

    if not _ensure_fresh(report_path, max_age=max_age):
        return {}
    try:
        content = report_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except Exception:  # pragma: no cover
        _LOGGER.exception("Błąd odczytu raportu portfolio_stress %s", report_path)
        return {}

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        _LOGGER.exception("Niepoprawny JSON raportu portfolio_stress w %s", report_path)
        return {}

    scenarios_raw = payload.get("scenarios")
    if not isinstance(scenarios_raw, Sequence):
        return {}

    scenarios: list[dict[str, Any]] = []
    worst_drawdown: float | None = None
    worst_entry: dict[str, Any] | None = None
    for entry in scenarios_raw:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name", "")).strip() or None
        total_return = _to_float(entry.get("total_return_pct"))
        drawdown = _to_float(entry.get("drawdown_pct"))
        total_pnl = _to_float(entry.get("total_pnl_usd"))
        liquidity = _to_float(entry.get("liquidity_impact_usd"))
        cash_pnl = _to_float(entry.get("cash_pnl_usd"))
        worst_section = entry.get("worst_position")
        worst_symbol = None
        worst_symbol_return = None
        worst_symbol_pnl = None
        if isinstance(worst_section, Mapping):
            symbol_value = worst_section.get("symbol")
            worst_symbol = str(symbol_value).strip() if isinstance(symbol_value, str) else None
            worst_symbol_return = _to_float(worst_section.get("return_pct"))
            worst_symbol_pnl = _to_float(worst_section.get("pnl_usd"))
        scenario_summary: dict[str, Any] = {
            "name": name,
            "title": entry.get("title"),
            "horizon_days": _to_float(entry.get("horizon_days")),
            "probability": _to_float(entry.get("probability")),
            "total_return_pct": total_return,
            "drawdown_pct": drawdown,
            "total_pnl_usd": total_pnl,
            "cash_pnl_usd": cash_pnl,
            "liquidity_impact_usd": liquidity,
            "worst_position": {
                "symbol": worst_symbol,
                "return_pct": worst_symbol_return,
                "pnl_usd": worst_symbol_pnl,
            },
        }
        tags = entry.get("tags")
        if isinstance(tags, Sequence) and not isinstance(tags, (str, bytes)):
            scenario_summary["tags"] = [str(tag) for tag in tags]
        scenarios.append(scenario_summary)
        if drawdown is not None:
            if worst_drawdown is None or drawdown > worst_drawdown:
                worst_drawdown = drawdown
                worst_entry = scenario_summary

    summary_raw = payload.get("summary")
    summary_section: Mapping[str, Any] | None
    summary_section = summary_raw if isinstance(summary_raw, Mapping) else None

    scenario_count = int(
        (summary_section.get("scenario_count") if summary_section else None)
        or len(scenarios)
    )
    summary_payload: dict[str, Any] = {
        "portfolio_id": payload.get("portfolio_id"),
        "generated_at": payload.get("generated_at"),
        "scenario_count": scenario_count,
        "scenarios": scenarios,
    }

    summary_details: dict[str, Any] = {"scenario_count": scenario_count}

    max_drawdown = _to_float(summary_section.get("max_drawdown_pct")) if summary_section else None
    if max_drawdown is None:
        max_drawdown = worst_drawdown
    if max_drawdown is not None:
        summary_payload["max_drawdown_pct"] = max_drawdown
        summary_details["max_drawdown_pct"] = max_drawdown

    min_total_return = _to_float(summary_section.get("min_total_return_pct")) if summary_section else None
    if min_total_return is not None:
        summary_payload["min_total_return_pct"] = min_total_return
        summary_details["min_total_return_pct"] = min_total_return

    max_liquidity = _to_float(summary_section.get("max_liquidity_impact_usd")) if summary_section else None
    if max_liquidity is not None:
        summary_payload["max_liquidity_impact_usd"] = max_liquidity
        summary_details["max_liquidity_impact_usd"] = max_liquidity

    worst_summary: Mapping[str, Any] | None = None
    if summary_section and isinstance(summary_section.get("worst_scenario"), Mapping):
        worst_summary = cast(Mapping[str, Any], summary_section.get("worst_scenario"))

    if worst_summary is not None:
        worst_payload = {
            "name": worst_summary.get("name"),
            "title": worst_summary.get("title"),
            "drawdown_pct": _to_float(worst_summary.get("drawdown_pct")),
            "total_return_pct": _to_float(worst_summary.get("total_return_pct")),
            "total_pnl_usd": _to_float(worst_summary.get("total_pnl_usd")),
            "liquidity_impact_usd": _to_float(worst_summary.get("liquidity_impact_usd")),
        }
    elif worst_entry is not None:
        worst_payload = {
            "name": worst_entry.get("name"),
            "drawdown_pct": worst_entry.get("drawdown_pct"),
            "total_return_pct": worst_entry.get("total_return_pct"),
            "total_pnl_usd": worst_entry.get("total_pnl_usd"),
        }
    else:
        worst_payload = None

    if worst_payload is not None:
        summary_payload["worst_scenario"] = worst_payload
        summary_details["worst_scenario"] = worst_payload

    if summary_section is not None:
        total_probability = _to_float(summary_section.get("total_probability"))
        if total_probability is not None:
            summary_details["total_probability"] = total_probability
        expected_pnl = _to_float(summary_section.get("expected_pnl_usd"))
        if expected_pnl is not None:
            summary_details["expected_pnl_usd"] = expected_pnl
        expected_return = _to_float(summary_section.get("expected_return_pct"))
        if expected_return is not None:
            summary_details["expected_return_pct"] = expected_return
        var_return = _to_float(summary_section.get("var_95_return_pct"))
        if var_return is not None:
            summary_details["var_95_return_pct"] = var_return
        var_pnl = _to_float(summary_section.get("var_95_pnl_usd"))
        if var_pnl is not None:
            summary_details["var_95_pnl_usd"] = var_pnl
        cvar_return = _to_float(summary_section.get("cvar_95_return_pct"))
        if cvar_return is not None:
            summary_details["cvar_95_return_pct"] = cvar_return
        cvar_pnl = _to_float(summary_section.get("cvar_95_pnl_usd"))
        if cvar_pnl is not None:
            summary_details["cvar_95_pnl_usd"] = cvar_pnl
        tag_aggregates_raw = summary_section.get("tag_aggregates")
        if isinstance(tag_aggregates_raw, Sequence):
            aggregates: list[dict[str, Any]] = []
            for entry in tag_aggregates_raw:
                if not isinstance(entry, Mapping):
                    continue
                tag_value = entry.get("tag")
                if not isinstance(tag_value, str):
                    continue
                scenario_count_raw = entry.get("scenario_count")
                try:
                    scenario_count = int(scenario_count_raw)
                except (TypeError, ValueError):
                    scenario_count = None
                aggregate: dict[str, Any] = {"tag": tag_value}
                if scenario_count is not None:
                    aggregate["scenario_count"] = scenario_count
                max_drawdown_tag = _to_float(entry.get("max_drawdown_pct"))
                if max_drawdown_tag is not None:
                    aggregate["max_drawdown_pct"] = max_drawdown_tag
                worst_return_tag = _to_float(entry.get("worst_return_pct"))
                if worst_return_tag is not None:
                    aggregate["worst_return_pct"] = worst_return_tag
                total_probability_tag = _to_float(entry.get("total_probability"))
                if total_probability_tag is not None:
                    aggregate["total_probability"] = total_probability_tag
                expected_pnl_tag = _to_float(entry.get("expected_pnl_usd"))
                if expected_pnl_tag is not None:
                    aggregate["expected_pnl_usd"] = expected_pnl_tag
                worst_tag_raw = entry.get("worst_scenario")
                if isinstance(worst_tag_raw, Mapping):
                    aggregate["worst_scenario"] = {
                        "name": worst_tag_raw.get("name"),
                        "title": worst_tag_raw.get("title"),
                        "drawdown_pct": _to_float(worst_tag_raw.get("drawdown_pct")),
                        "total_return_pct": _to_float(
                            worst_tag_raw.get("total_return_pct")
                        ),
                        "total_pnl_usd": _to_float(worst_tag_raw.get("total_pnl_usd")),
                    }
                aggregates.append(aggregate)
            if aggregates:
                summary_details["tag_aggregates"] = aggregates

    summary_payload["summary"] = summary_details

    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        summary_payload["metadata"] = dict(metadata)
    return summary_payload


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


def build_portfolio_stress_provider(
    path: str | Path,
    *,
    fallback_directories: Sequence[Path] = (),
    max_age: timedelta | None = None,
) -> Callable[[], Mapping[str, Any]]:
    target = Path(path)
    fallbacks = tuple(Path(item) for item in fallback_directories)

    def _provider() -> Mapping[str, Any]:
        for candidate in _candidate_paths(target, fallbacks):
            try:
                summary = load_portfolio_stress_summary(candidate, max_age=max_age)
            except Exception:  # pragma: no cover - log diagnostyczny
                _LOGGER.exception(
                    "Błąd ładowania raportu portfolio_stress z %s",
                    candidate,
                )
                return {}
            if summary:
                return summary
        _LOGGER.debug(
            "Nie znaleziono aktualnego raportu portfolio_stress dla ścieżki %s (fallbacki: %s)",
            target,
            ", ".join(str(item) for item in fallbacks) or "brak",
        )
        return {}

    return _provider
