from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.ai.config_loader import load_risk_thresholds


_FREEZE_STATUS_PREFIXES = ("risk_freeze", "auto_risk_freeze")
_FREEZE_STATUS_EXTRAS = {"risk_unfreeze", "auto_risk_unfreeze"}
_ABSOLUTE_THRESHOLD_METRICS = {"signal_after_adjustment", "signal_after_clamp"}

_AMBIGUOUS_SYMBOL_MAPPING: tuple[str, str] = ("__ambiguous__", "__ambiguous__")


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
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
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
                        events.append(dict(payload))
        except OSError as exc:  # noqa: BLE001 - CLI feedback
            raise SystemExit(f"Nie udało się odczytać dziennika {path}: {exc}") from exc
    return events


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


def _load_autotrade_entries(
    paths: Iterable[str],
    *,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.exists():
            raise SystemExit(f"Eksport autotradera nie istnieje: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # noqa: BLE001 - CLI feedback
            raise SystemExit(f"Niepoprawny JSON w eksporcie autotradera {path}: {exc}") from exc
        if isinstance(payload, Mapping):
            raw_entries = payload.get("entries")
            if isinstance(raw_entries, Iterable):
                for item in raw_entries:
                    if isinstance(item, Mapping):
                        timestamp = _extract_entry_timestamp(item)
                        if since and timestamp and timestamp < since:
                            continue
                        if until and timestamp and timestamp > until:
                            continue
                        entries.append(dict(item))
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, Mapping):
                    timestamp = _extract_entry_timestamp(item)
                    if since and timestamp and timestamp < since:
                        continue
                    if until and timestamp and timestamp > until:
                        continue
                    entries.append(dict(item))
    return entries


def _resolve_key(exchange: str | None, strategy: str | None) -> tuple[str, str]:
    normalized_exchange = (exchange or "unknown").strip() or "unknown"
    normalized_strategy = (strategy or "unknown").strip() or "unknown"
    return normalized_exchange, normalized_strategy


def _is_unknown_token(value: str) -> bool:
    return value.strip().lower() == "unknown"


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


def _build_symbol_map(events: Iterable[Mapping[str, object]]) -> dict[str, tuple[str, str]]:
    symbol_map: dict[str, tuple[str, str]] = {}
    for event in events:
        symbol = _normalize_string(event.get("symbol"))
        if not symbol:
            continue
        exchange = _normalize_string(event.get("primary_exchange"))
        strategy = _normalize_string(event.get("strategy"))
        key = _resolve_key(exchange, strategy)
        existing = symbol_map.get(symbol)
        if existing is None:
            symbol_map[symbol] = key
            continue
        if existing == _AMBIGUOUS_SYMBOL_MAPPING:
            continue
        if existing == key:
            continue
        if _has_conflict(existing, key):
            symbol_map[symbol] = _AMBIGUOUS_SYMBOL_MAPPING
            continue
        merged_exchange = existing[0]
        merged_strategy = existing[1]
        candidate_exchange, candidate_strategy = key
        if not _is_unknown_token(candidate_exchange):
            merged_exchange = candidate_exchange
        if not _is_unknown_token(candidate_strategy):
            merged_strategy = candidate_strategy
        merged = (merged_exchange, merged_strategy)
        if merged == existing:
            continue
        if _has_conflict(existing, merged):
            symbol_map[symbol] = _AMBIGUOUS_SYMBOL_MAPPING
            continue
        symbol_map[symbol] = merged
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
) -> tuple[str, str]:
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

    if symbol and (exchange is None or strategy is None):
        mapped = symbol_map.get(symbol)
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

    return _resolve_key(exchange, strategy)


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
    target_values = [abs(value) for value in values] if absolute else list(values)
    target_values.sort()
    return _compute_percentile(target_values, percentile)


def _build_metrics_section(
    values_map: Mapping[str, list[float]],
    percentiles: Iterable[float],
    suggestion_percentile: float,
    *,
    current_risk_score: float | None,
) -> dict[str, dict[str, object]]:
    metrics_payload: dict[str, dict[str, object]] = {}
    for metric_name, values in values_map.items():
        stats_payload = _metric_statistics(values, percentiles)
        absolute = metric_name in _ABSOLUTE_THRESHOLD_METRICS
        suggested = _suggest_threshold(values, suggestion_percentile, absolute=absolute)
        current = current_risk_score if metric_name == "risk_score" else None
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
    journal_events: list[dict[str, object]],
    autotrade_entries: list[dict[str, object]],
    percentiles: list[float],
    suggestion_percentile: float,
    since: datetime | None = None,
    until: datetime | None = None,
) -> dict[str, object]:
    symbol_map = _build_symbol_map(journal_events)
    grouped_values: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    raw_value_snapshots: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    freeze_summaries: dict[tuple[str, str], dict[str, object]] = defaultdict(
        lambda: {
            "total": 0,
            "type_counts": Counter(),
            "status_counts": Counter(),
            "reason_counts": Counter(),
        }
    )
    freeze_events: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)

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
            if numeric_duration is not None:
                grouped_values[key]["risk_freeze_duration"].append(numeric_duration)
                raw_value_snapshots[key]["risk_freeze_duration"].append(numeric_duration)
        freeze_events[key].append(
            {
                "status": status,
                "type": freeze_type,
                "reason": reason,
                "duration": duration,
                "risk_score": payload.get("risk_score"),
            }
        )

    for event in journal_events:
        base_exchange = _normalize_string(event.get("primary_exchange"))
        base_strategy = _normalize_string(event.get("strategy"))
        key = _resolve_key(base_exchange, base_strategy)
        for metric in ("signal_after_adjustment", "signal_after_clamp"):
            value = event.get(metric)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            grouped_values[key][metric].append(numeric)
            raw_value_snapshots[key][metric].append(numeric)
        for freeze_payload in _iter_freeze_events(event):
            symbol = _extract_symbol(event)
            exchange = base_exchange
            strategy = base_strategy
            if symbol:
                mapped = symbol_map.get(symbol)
                if mapped:
                    mapped_exchange, mapped_strategy = mapped
                    if exchange is None and mapped_exchange not in (None, "unknown"):
                        exchange = mapped_exchange
                    if strategy is None and mapped_strategy not in (None, "unknown"):
                        strategy = mapped_strategy
            freeze_key = _resolve_key(exchange, strategy)
            _record_freeze(freeze_key, freeze_payload)

    for entry in autotrade_entries:
        summary = _extract_summary(entry)
        symbol = _extract_symbol(entry)
        key = _resolve_group_from_symbol(entry, symbol, summary, symbol_map)

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
        grouped_values[key]["risk_score"].append(score_value)
        raw_value_snapshots[key]["risk_score"].append(score_value)

    thresholds = load_risk_thresholds()
    current_risk_score = None
    auto_trader_cfg = thresholds.get("auto_trader")
    if isinstance(auto_trader_cfg, Mapping):
        map_cfg = auto_trader_cfg.get("map_regime_to_signal")
        if isinstance(map_cfg, Mapping):
            value = map_cfg.get("risk_score")
            if isinstance(value, (int, float)):
                current_risk_score = float(value)

    groups: list[dict[str, object]] = []
    aggregated_values: defaultdict[str, list[float]] = defaultdict(list)
    aggregated_freeze_summary = {
        "total": 0,
        "type_counts": Counter(),
        "status_counts": Counter(),
        "reason_counts": Counter(),
    }

    for (exchange, strategy), metrics in sorted(grouped_values.items()):
        metrics_payload = _build_metrics_section(
            metrics,
            percentiles,
            suggestion_percentile,
            current_risk_score=current_risk_score,
        )
        raw_snapshot = raw_value_snapshots.get((exchange, strategy), {})
        freeze_summary = freeze_summaries.get((exchange, strategy)) or {
            "total": 0,
            "type_counts": Counter(),
            "status_counts": Counter(),
            "reason_counts": Counter(),
        }
        freeze_summary_payload = _format_freeze_summary(freeze_summary)
        aggregated_freeze_summary["total"] = int(aggregated_freeze_summary["total"]) + int(
            freeze_summary.get("total") or 0
        )
        if isinstance(freeze_summary.get("type_counts"), Counter):
            aggregated_freeze_summary["type_counts"].update(freeze_summary["type_counts"])
        if isinstance(freeze_summary.get("status_counts"), Counter):
            aggregated_freeze_summary["status_counts"].update(freeze_summary["status_counts"])
        if isinstance(freeze_summary.get("reason_counts"), Counter):
            aggregated_freeze_summary["reason_counts"].update(freeze_summary["reason_counts"])
        for metric_name, values in metrics.items():
            aggregated_values[metric_name].extend(values)
        groups.append(
            {
                "primary_exchange": exchange,
                "strategy": strategy,
                "metrics": metrics_payload,
                "raw_values": {metric: list(values) for metric, values in raw_snapshot.items()},
                "freeze_summary": freeze_summary_payload,
                "raw_freeze_events": list(freeze_events.get((exchange, strategy), [])),
            }
        )

    global_metrics = _build_metrics_section(
        aggregated_values,
        percentiles,
        suggestion_percentile,
        current_risk_score=current_risk_score,
    )
    global_summary = {
        "metrics": global_metrics,
        "freeze_summary": _format_freeze_summary(aggregated_freeze_summary),
        "raw_values": {metric: list(values) for metric, values in aggregated_values.items()},
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
        "sources": {
            "journal_events": len(journal_events),
            "autotrade_entries": len(autotrade_entries),
        },
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
        "--plot-dir",
        help="Opcjonalny katalog na histogramy z rozkładami metryk",
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

    report = _generate_report(
        journal_events=journal_events,
        autotrade_entries=autotrade_entries,
        percentiles=percentiles,
        suggestion_percentile=args.suggestion_percentile,
        since=since,
        until=until,
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
    print(
        f"Przetworzono {len(journal_events)} zdarzeń dziennika i "
        f"{len(autotrade_entries)} wpisów autotradera dla {total_groups} kombinacji giełda/strategia."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
