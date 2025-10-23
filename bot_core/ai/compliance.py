"""Helpery zbierające zbiorczy status zgodności pipeline'u AI."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping, Sequence

from .audit import (
    load_recent_walk_forward_reports,
    load_scheduler_state,
    summarize_walk_forward_reports,
)
from .data_monitoring import (
    load_recent_data_quality_reports,
    load_recent_drift_reports,
    summarize_data_quality_reports,
    summarize_drift_reports,
)

__all__ = ["collect_pipeline_compliance_summary"]


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        candidate = value
    elif isinstance(value, str):
        try:
            candidate = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None
    if candidate.tzinfo is None:
        candidate = candidate.replace(tzinfo=timezone.utc)
    else:
        candidate = candidate.astimezone(timezone.utc)
    return candidate


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _merge_pending_sign_offs(
    *sources: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> Mapping[str, tuple[Mapping[str, Any], ...]]:
    pending: MutableMapping[str, list[Mapping[str, Any]]] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for role, entries in source.items():
            role_key = str(role)
            bucket = pending.setdefault(role_key, [])
            if isinstance(entries, Sequence):
                for entry in entries:
                    if isinstance(entry, Mapping):
                        bucket.append(dict(entry))
    return {role: tuple(entries) for role, entries in pending.items()}


def _add_issue(issues: list[str], label: str) -> None:
    if label not in issues:
        issues.append(label)


def _summarize_scheduler_state(
    state: Mapping[str, Any], *, now: datetime
) -> Mapping[str, Any]:
    snapshot = {str(key): value for key, value in state.items()}
    last_run = _parse_iso_datetime(snapshot.get("last_run"))
    next_run = _parse_iso_datetime(snapshot.get("next_run"))
    cooldown_until = _parse_iso_datetime(snapshot.get("cooldown_until"))
    paused_until = _parse_iso_datetime(snapshot.get("paused_until"))

    seconds_since_last = None
    if last_run is not None:
        seconds_since_last = (now - last_run).total_seconds()
    seconds_until_next = None
    if next_run is not None:
        seconds_until_next = (next_run - now).total_seconds()

    summary: MutableMapping[str, Any] = {
        "state": MappingProxyType(snapshot),
        "is_overdue": bool(seconds_until_next is not None and seconds_until_next <= 0),
        "cooldown_active": bool(cooldown_until is not None and cooldown_until > now),
        "paused": bool(paused_until is not None and paused_until > now),
        "seconds_since_last_run": seconds_since_last,
        "seconds_until_next_run": seconds_until_next,
    }

    interval_seconds = _safe_float(snapshot.get("interval"))
    if interval_seconds is None:
        interval_seconds = _safe_float(snapshot.get("configured_interval"))
    if interval_seconds is not None:
        summary["interval_seconds"] = interval_seconds

    return MappingProxyType(summary)


def collect_pipeline_compliance_summary(
    *,
    audit_root: str | None | Any = None,
    data_quality_limit: int = 5,
    drift_limit: int = 5,
    walk_forward_limit: int = 3,
    include_scheduler: bool = True,
    now: datetime | None = None,
) -> Mapping[str, Any]:
    """Buduje zbiorcze podsumowanie gotowości pipeline'u AI do wdrożenia."""

    reference_time = now.astimezone(timezone.utc) if now else datetime.now(timezone.utc)

    dq_reports = load_recent_data_quality_reports(
        limit=data_quality_limit, audit_root=audit_root
    )
    drift_reports = load_recent_drift_reports(limit=drift_limit, audit_root=audit_root)
    walk_reports = load_recent_walk_forward_reports(
        limit=walk_forward_limit, audit_root=audit_root
    )

    dq_summary = summarize_data_quality_reports(dq_reports)
    drift_summary = summarize_drift_reports(drift_reports)
    walk_summary = summarize_walk_forward_reports(walk_reports)

    pending = _merge_pending_sign_offs(
        dq_summary.get("pending_sign_off"),
        drift_summary.get("pending_sign_off"),
        walk_summary.get("pending_sign_off"),
    )

    issues: list[str] = []
    if int(dq_summary.get("enforced_alerts", 0)) > 0:
        _add_issue(issues, "data_quality_alerts")
    if int(drift_summary.get("exceeds_threshold", 0)) > 0:
        _add_issue(issues, "drift_alerts")
    if any(pending.values()):
        _add_issue(issues, "missing_sign_offs")
    if int(walk_summary.get("total", 0)) == 0:
        _add_issue(issues, "missing_walk_forward_reports")

    scheduler_summary: Mapping[str, Any] | None = None
    if include_scheduler:
        scheduler_state = load_scheduler_state(audit_root=audit_root)
        if scheduler_state is not None:
            scheduler_summary = _summarize_scheduler_state(
                scheduler_state, now=reference_time
            )
            if scheduler_summary.get("is_overdue"):
                _add_issue(issues, "scheduler_overdue")
            if scheduler_summary.get("cooldown_active"):
                _add_issue(issues, "scheduler_cooldown")
            if scheduler_summary.get("paused"):
                _add_issue(issues, "scheduler_paused")

    summary: MutableMapping[str, Any] = {
        "data_quality": dq_summary,
        "drift": drift_summary,
        "walk_forward": walk_summary,
        "pending_sign_off": MappingProxyType(pending),
        "issues": tuple(issues),
        "ready": not issues,
        "scheduler": scheduler_summary,
    }
    return MappingProxyType(summary)
