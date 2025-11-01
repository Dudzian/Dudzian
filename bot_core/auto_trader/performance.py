"""Narzędzia do wyliczania metryk krzywej kapitału kontrolera auto-mode."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any, Mapping, Sequence

import pandas as pd


def _normalize_cycle_timestamp(value: Any, tz: tzinfo | None) -> datetime | None:
    """Konwertuje różne reprezentacje czasu cyklu na ``datetime`` w zadanej strefie."""

    target_tz = tz or timezone.utc

    if isinstance(value, datetime):
        candidate = value
    elif isinstance(value, pd.Timestamp):
        candidate = value.to_pydatetime()
    elif isinstance(value, (int, float)) and math.isfinite(value):
        candidate = datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            candidate = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = pd.to_datetime(stripped, utc=True, errors="coerce")
            except Exception:
                return None
            if pd.isna(parsed):
                return None
            candidate = parsed.to_pydatetime()
    else:
        return None

    if candidate.tzinfo is None:
        candidate = candidate.replace(tzinfo=timezone.utc)

    try:
        return candidate.astimezone(target_tz)
    except Exception:
        return None


def _extract_cycle_pnl(cycle: Mapping[str, Any]) -> float:
    """Zwraca PnL cyklu kontrolera z tolerancją na różne pola."""

    pnl_candidate: Any = cycle.get("pnl")
    if pnl_candidate is None and isinstance(cycle.get("orders"), Sequence):
        for order in cycle["orders"]:  # type: ignore[index]
            if not isinstance(order, Mapping):
                continue
            if "pnl" in order:
                pnl_candidate = order.get("pnl")
                break
            if "pnl_usd" in order:
                pnl_candidate = order.get("pnl_usd")
                break
            if "pnl_usdt" in order:
                pnl_candidate = order.get("pnl_usdt")
                break
    try:
        value = float(pnl_candidate) if pnl_candidate is not None else 0.0
    except (TypeError, ValueError):
        value = 0.0
    if not math.isfinite(value):
        return 0.0
    return value


def _window_payload(start: datetime | None, end: datetime | None) -> dict[str, Any]:
    """Buduje opis okna czasowego oparty na znacznikach czasu."""

    if start is None and end is None:
        return {}
    payload: dict[str, Any] = {}
    if start is not None:
        payload["start"] = start.isoformat()
    if end is not None:
        payload["end"] = end.isoformat()
    if start is not None and end is not None:
        duration = (end - start).total_seconds()
        payload["duration_s"] = 0 if duration <= 0 else int(duration)
    return payload


def build_cycle_equity_summary(
    history: Sequence[Mapping[str, Any]] | None,
    *,
    tz: tzinfo | None,
    now: datetime | None = None,
    base_equity: float = 100_000.0,
    window_hours: float | None = 24.0,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Buduje krzywą kapitału oraz metryki z historii cykli kontrolera."""

    if not history:
        return [], {}, {}

    target_tz = tz or timezone.utc
    now_reference = now.astimezone(target_tz) if now else datetime.now(target_tz)
    window_start: datetime | None = None
    if window_hours is not None and window_hours > 0:
        window_start = now_reference - timedelta(hours=float(window_hours))

    sortable: list[tuple[datetime | None, int, Mapping[str, Any], Any]] = []
    for index, entry in enumerate(history):
        if not isinstance(entry, Mapping):
            continue
        timestamp_raw = (
            entry.get("finished_at")
            or entry.get("timestamp")
            or entry.get("ended_at")
            or entry.get("completed_at")
            or entry.get("started_at")
        )
        sortable.append((_normalize_cycle_timestamp(timestamp_raw, target_tz), index, entry, timestamp_raw))

    if not sortable:
        return [], {}, {}

    def sort_key(item: tuple[datetime | None, int, Mapping[str, Any], Any]) -> tuple[datetime, int]:
        timestamp, order, _, _ = item
        if timestamp is None:
            return datetime.min.replace(tzinfo=target_tz), order
        return timestamp, order

    sortable.sort(key=sort_key)

    equity_points: list[dict[str, Any]] = []
    returns: list[float] = []
    max_drawdown = 0.0
    cumulative = base_equity
    peak = cumulative
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    cycle_count = 0

    window_returns: list[float] = []
    window_max_drawdown = 0.0
    window_cycle_count = 0
    window_first_timestamp: datetime | None = None
    window_last_timestamp: datetime | None = None
    window_base_equity: float | None = None
    window_previous_equity: float | None = None
    window_last_equity: float | None = None
    window_peak: float | None = None

    for index, (timestamp, _order, entry, raw_timestamp) in enumerate(sortable):
        pnl_value = _extract_cycle_pnl(entry)
        previous_equity = cumulative
        cumulative = previous_equity + pnl_value
        cycle_count += 1

        if timestamp is not None:
            if first_timestamp is None:
                first_timestamp = timestamp
            last_timestamp = timestamp

        if previous_equity != 0.0 and math.isfinite(previous_equity):
            returns.append((cumulative - previous_equity) / previous_equity)

        if cumulative > peak:
            peak = cumulative
        elif peak > 0.0:
            drawdown = (peak - cumulative) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        in_window = window_start is not None and timestamp is not None and timestamp >= window_start
        if window_start is not None:
            if in_window:
                if window_first_timestamp is None:
                    window_first_timestamp = timestamp
                    window_base_equity = previous_equity
                    window_previous_equity = previous_equity
                    window_peak = max(previous_equity, cumulative)
                if window_previous_equity is not None and window_previous_equity != 0.0 and math.isfinite(window_previous_equity):
                    window_returns.append((cumulative - window_previous_equity) / window_previous_equity)
                window_previous_equity = cumulative
                window_cycle_count += 1
                window_last_equity = cumulative
                window_last_timestamp = timestamp
                if window_peak is None or cumulative > window_peak:
                    window_peak = cumulative
                elif window_peak > 0.0:
                    window_drawdown = (window_peak - cumulative) / window_peak
                    if window_drawdown > window_max_drawdown:
                        window_max_drawdown = window_drawdown
            else:
                if window_first_timestamp is None:
                    window_peak = cumulative
                    window_previous_equity = cumulative
                    window_base_equity = cumulative

        if timestamp is not None:
            timestamp_value: Any = timestamp.isoformat()
        elif isinstance(raw_timestamp, (str, int, float)):
            timestamp_value = raw_timestamp
        else:
            timestamp_value = index

        equity_points.append(
            {
                "timestamp": timestamp_value,
                "value": cumulative,
                "source": "controller",
                "category": "auto-mode",
            }
        )

    metrics: dict[str, Any] = {}
    if cycle_count > 0:
        metrics["cycle_count"] = cycle_count
        if base_equity != 0.0 and math.isfinite(base_equity):
            metrics["net_return_pct"] = (cumulative - base_equity) / base_equity
        if returns:
            mean = sum(returns) / len(returns)
            metrics["avg_return_pct"] = mean
            variance = max(0.0, (sum(value * value for value in returns) / len(returns)) - (mean * mean))
            metrics["volatility_pct"] = math.sqrt(variance)
        metrics["max_drawdown_pct"] = max_drawdown
        window_payload = _window_payload(first_timestamp, last_timestamp)
        if window_payload:
            metrics["window"] = window_payload

    window_metrics: dict[str, Any] = {}
    if (
        window_cycle_count > 0
        and window_base_equity is not None
        and window_last_equity is not None
        and window_base_equity != 0.0
        and math.isfinite(window_base_equity)
    ):
        window_metrics["cycle_count"] = window_cycle_count
        window_metrics["net_return_pct"] = (window_last_equity - window_base_equity) / window_base_equity
        if window_returns:
            mean = sum(window_returns) / len(window_returns)
            window_metrics["avg_return_pct"] = mean
            variance = max(
                0.0,
                (sum(value * value for value in window_returns) / len(window_returns)) - (mean * mean),
            )
            window_metrics["volatility_pct"] = math.sqrt(variance)
        window_metrics["max_drawdown_pct"] = window_max_drawdown
        window_payload = _window_payload(window_first_timestamp, window_last_timestamp)
        if window_payload:
            window_metrics["window"] = window_payload

    return equity_points, metrics, window_metrics


__all__ = ["build_cycle_equity_summary"]
