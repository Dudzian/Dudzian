"""Narzędzia monitoringu jakości danych inference Decision Engine."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from bot_core.alerts import DriftAlertPayload

__all__ = [
    "DataCompletenessWatcher",
    "FeatureBoundsValidator",
    "export_data_quality_report",
    "export_drift_alert_report",
    "DataQualityException",
    "apply_policy_to_report",
    "update_sign_off",
    "load_recent_data_quality_reports",
    "load_recent_drift_reports",
    "summarize_data_quality_reports",
    "summarize_drift_reports",
]


_SAFE_FILENAME = re.compile(r"[^a-z0-9]+", re.IGNORECASE)
_SIGN_OFF_ROLES = frozenset({"risk", "compliance"})
_SIGN_OFF_STATUSES = frozenset(
    {"pending", "approved", "rejected", "escalated", "waived", "investigating"}
)
_COMPLETED_SIGN_OFF_STATUSES = frozenset({"approved", "waived"})


def _audit_root() -> Path:
    root_override = os.environ.get("AI_DECISION_AUDIT_ROOT")
    if root_override:
        return Path(root_override).expanduser().resolve()
    return Path("audit") / "ai_decision"


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_slug(prefix: str) -> str:
    normalized = _SAFE_FILENAME.sub("_", prefix or "report").strip("_") or "report"
    return normalized.lower()


def _timestamp_slug(prefix: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{now}_{_normalize_slug(prefix)}"


def _default_sign_off() -> dict[str, MutableMapping[str, Any]]:
    return {
        "risk": {
            "status": "pending",
            "signed_by": None,
            "timestamp": None,
            "notes": "Awaiting Risk review",
        },
        "compliance": {
            "status": "pending",
            "signed_by": None,
            "timestamp": None,
            "notes": "Awaiting Compliance sign-off",
        },
    }


def _write_report(directory: Path, prefix: str, payload: Mapping[str, Any]) -> Path:
    base = _timestamp_slug(prefix)
    destination = directory / f"{base}.json"
    counter = 1
    while destination.exists():
        destination = directory / f"{base}_{counter}.json"
        counter += 1
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return destination


def _update_report_file(
    report: Mapping[str, Any], mutator: Callable[[MutableMapping[str, Any]], None]
) -> None:
    report_path = report.get("report_path")
    if not isinstance(report_path, (str, PathLike)):
        return
    path = Path(report_path)
    try:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, Mapping):
            payload_dict: MutableMapping[str, Any] = dict(payload)
        else:
            payload_dict = {}
        mutator(payload_dict)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload_dict, handle, ensure_ascii=False, indent=2, sort_keys=True)
    except (OSError, json.JSONDecodeError, TypeError):
        # Nie przerywamy przepływu w razie problemów I/O – raport w pamięci pozostaje aktualny.
        return


def _apply_sign_off_to_payload(
    payload: MutableMapping[str, Any], role: str, entry: Mapping[str, Any]
) -> None:
    raw_sign_off = payload.get("sign_off")
    if isinstance(raw_sign_off, MutableMapping):
        sign_off: MutableMapping[str, Any] = dict(raw_sign_off)
    elif isinstance(raw_sign_off, Mapping):
        sign_off = dict(raw_sign_off)
    else:
        sign_off = _default_sign_off()
    sign_off[role] = dict(entry)
    payload["sign_off"] = sign_off


def export_data_quality_report(
    payload: Mapping[str, Any], *, category: str = "completeness"
) -> Path:
    """Persistuje raport jakości danych do katalogu audytowego."""

    root = _ensure_directory(_audit_root() / "data_quality")
    enriched = dict(payload)
    enriched.setdefault("sign_off", _default_sign_off())
    return _write_report(root, category, enriched)


def apply_policy_to_report(
    report: MutableMapping[str, Any], *, enforce: bool
) -> MutableMapping[str, Any]:
    """Dodaje sekcję polityki do raportu i aktualizuje zapisany plik."""

    policy = {"enforce": bool(enforce)}
    report["policy"] = policy
    _update_report_file(report, lambda payload: payload.__setitem__("policy", policy))
    return report


def update_sign_off(
    report: MutableMapping[str, Any],
    *,
    role: str,
    status: str,
    signed_by: str | None = None,
    notes: str | None = None,
) -> MutableMapping[str, Any]:
    """Aktualizuje sekcję podpisów Risk/Compliance w raporcie i pliku JSON."""

    normalized_role = str(role).strip().lower()
    if normalized_role not in _SIGN_OFF_ROLES:
        raise ValueError(f"unsupported sign-off role: {role!r}")
    normalized_status = str(status).strip().lower()
    if normalized_status not in _SIGN_OFF_STATUSES:
        raise ValueError(f"unsupported sign-off status: {status!r}")

    sign_off = report.get("sign_off")
    if isinstance(sign_off, MutableMapping):
        sign_off_map: MutableMapping[str, Any] = sign_off
    elif isinstance(sign_off, Mapping):
        sign_off_map = dict(sign_off)
    else:
        sign_off_map = _default_sign_off()
    report["sign_off"] = sign_off_map

    entry = sign_off_map.get(normalized_role)
    if not isinstance(entry, MutableMapping):
        entry = {}

    entry = dict(entry)
    entry["status"] = normalized_status
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    if signed_by is not None:
        entry["signed_by"] = str(signed_by)
    else:
        entry.setdefault("signed_by", None)
    if notes is not None:
        entry["notes"] = str(notes)
    else:
        default_notes = _default_sign_off().get(normalized_role, {}).get("notes")
        if default_notes is not None:
            entry.setdefault("notes", default_notes)

    sign_off_map[normalized_role] = entry
    _update_report_file(
        report,
        lambda payload: _apply_sign_off_to_payload(
            payload, normalized_role, entry
        ),
    )
    return report


def export_drift_alert_report(
    payload: DriftAlertPayload | Mapping[str, Any], *, category: str = "drift_alert"
) -> Path:
    """Zapisuje raport alertu dryfu do katalogu audytowego."""

    if isinstance(payload, DriftAlertPayload):
        raw: Mapping[str, Any] = {
            "model_name": payload.model_name,
            "drift_score": payload.drift_score,
            "threshold": payload.threshold,
            "window": payload.window,
            "backend": payload.backend,
            "extra": dict(payload.extra or {}),
        }
    else:
        raw = payload
    enriched = dict(raw)
    enriched.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    enriched.setdefault("sign_off", _default_sign_off())
    extra_payload = enriched.get("extra")
    if isinstance(extra_payload, Mapping):
        enriched["extra"] = _normalize_context(extra_payload)
    root = _ensure_directory(_audit_root() / "drift")
    return _write_report(root, category, enriched)


def _load_recent_reports(
    *,
    subdir: str,
    category: str | None = None,
    limit: int = 20,
) -> tuple[Mapping[str, Any], ...]:
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    root = _audit_root() / subdir
    if not root.exists():
        return ()
    pattern = "*.json"
    if category:
        slug = _normalize_slug(category)
        pattern = f"*_{slug}*.json"
    files = [path for path in root.glob(pattern) if path.is_file()]
    if not files:
        return ()
    files.sort(key=lambda path: (path.stat().st_mtime, path.name))
    selected = files[-limit:]
    reports: list[Mapping[str, Any]] = []
    for path in reversed(selected):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping):
            record: MutableMapping[str, Any] = dict(payload)
        else:
            record = {"raw_payload": payload}
        record.setdefault("report_path", str(path))
        if category is not None:
            record.setdefault("category", _normalize_slug(category))
        reports.append(record)
    return tuple(reports)


def load_recent_data_quality_reports(
    *, category: str | None = None, limit: int = 20
) -> tuple[Mapping[str, Any], ...]:
    """Ładuje najnowsze raporty jakości danych z dysku.

    Zwraca krotkę raportów (najnowszy jako pierwszy), co upraszcza porównanie
    alertów podczas przeglądów zgodności.  Ustawienie ``category`` ogranicza
    wynik do konkretnej kategorii (np. ``"completeness"``).
    """

    return _load_recent_reports(subdir="data_quality", category=category, limit=limit)


def load_recent_drift_reports(*, limit: int = 20) -> tuple[Mapping[str, Any], ...]:
    """Ładuje najnowsze raporty dryfu modelu z audytu."""

    return _load_recent_reports(subdir="drift", limit=limit)


def summarize_data_quality_reports(
    reports: Sequence[Mapping[str, Any]] | None,
) -> Mapping[str, Any]:
    """Buduje podsumowanie alertów jakości danych dla compliance i Risk."""

    normalized: list[Mapping[str, Any]] = [
        report
        for report in reports or ()
        if isinstance(report, Mapping)
    ]

    summary: MutableMapping[str, Any] = {
        "total": len(normalized),
        "alerts": 0,
        "enforced_alerts": 0,
        "by_category": {},
        "pending_sign_off": {role: [] for role in _SIGN_OFF_ROLES},
    }

    for report in normalized:
        category = str(report.get("category") or "unknown")
        status = str(report.get("status") or "").lower()
        policy = report.get("policy")
        enforce = True
        if isinstance(policy, Mapping):
            raw_enforce = policy.get("enforce")
            if isinstance(raw_enforce, bool):
                enforce = raw_enforce

        by_category = summary["by_category"]
        category_stats = by_category.setdefault(
            category,
            {
                "total": 0,
                "alerts": 0,
                "enforced_alerts": 0,
                "latest_status": None,
                "latest_report_path": None,
            },
        )
        category_stats["total"] += 1
        if category_stats["latest_status"] is None:
            category_stats["latest_status"] = status or None
        if category_stats["latest_report_path"] is None:
            report_path = report.get("report_path")
            if isinstance(report_path, (str, PathLike)):
                category_stats["latest_report_path"] = str(report_path)

        if status == "alert":
            summary["alerts"] += 1
            category_stats["alerts"] += 1
            if enforce:
                summary["enforced_alerts"] += 1
                category_stats["enforced_alerts"] += 1
                _collect_pending_sign_off(report, category, summary["pending_sign_off"])

    summary["by_category"] = {
        key: MappingProxyType(value) if not isinstance(value, MappingProxyType) else value
        for key, value in summary["by_category"].items()
    }
    summary["pending_sign_off"] = {
        role: tuple(entries)
        for role, entries in summary["pending_sign_off"].items()
    }
    return MappingProxyType(summary)


def summarize_drift_reports(
    reports: Sequence[Mapping[str, Any]] | None,
) -> Mapping[str, Any]:
    """Zwraca podsumowanie alertów dryfu oraz brakujących podpisów."""

    normalized: list[Mapping[str, Any]] = [
        report
        for report in reports or ()
        if isinstance(report, Mapping)
    ]

    summary: MutableMapping[str, Any] = {
        "total": len(normalized),
        "exceeds_threshold": 0,
        "latest_report_path": None,
        "latest_exceeding_report_path": None,
        "pending_sign_off": {role: [] for role in _SIGN_OFF_ROLES},
    }

    for index, report in enumerate(normalized):
        if index == 0:
            report_path = report.get("report_path")
            if isinstance(report_path, (str, PathLike)):
                summary["latest_report_path"] = str(report_path)
        try:
            drift_score = float(report.get("drift_score"))
            threshold = float(report.get("threshold"))
        except (TypeError, ValueError):
            continue
        exceeds = drift_score >= threshold
        if not exceeds:
            continue
        summary["exceeds_threshold"] += 1
        if summary["latest_exceeding_report_path"] is None:
            report_path = report.get("report_path")
            if isinstance(report_path, (str, PathLike)):
                summary["latest_exceeding_report_path"] = str(report_path)
        _collect_pending_sign_off(
            report,
            str(report.get("category") or "drift_alert"),
            summary["pending_sign_off"],
        )

    summary["pending_sign_off"] = {
        role: tuple(entries)
        for role, entries in summary["pending_sign_off"].items()
    }
    return MappingProxyType(summary)


def _collect_pending_sign_off(
    report: Mapping[str, Any],
    category: str,
    pending: MutableMapping[str, list[Mapping[str, Any]]],
) -> None:
    report_path = report.get("report_path")
    path_str = None
    if isinstance(report_path, (str, PathLike)):
        path_str = str(report_path)
    timestamp = report.get("timestamp")
    sign_off = report.get("sign_off")
    if not isinstance(sign_off, Mapping):
        for role in _SIGN_OFF_ROLES:
            pending[role].append(
                {
                    "category": category,
                    "status": "pending",
                    "report_path": path_str,
                    "timestamp": timestamp,
                }
            )
        return

    for role in _SIGN_OFF_ROLES:
        entry = sign_off.get(role)
        status = "pending"
        if isinstance(entry, Mapping):
            status = str(entry.get("status") or "pending").lower()
        if status not in _COMPLETED_SIGN_OFF_STATUSES:
            pending[role].append(
                {
                    "category": category,
                    "status": status,
                    "report_path": path_str,
                    "timestamp": timestamp,
                }
            )
def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _normalize_context(context: Mapping[str, Any]) -> dict[str, Any]:
    def _normalize(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            return {str(key): _normalize(val) for key, val in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [_normalize(item) for item in value]
        return str(value)

    return {str(key): _normalize(val) for key, val in context.items()}


@dataclass(slots=True)
class DataCompletenessWatcher:
    """Monitoruje brakujące cechy lub wartości null przed scoringiem."""

    expected_features: tuple[str, ...] = ()
    last_report_path: Path | None = field(init=False, default=None, repr=False)

    def configure(self, expected_features: Sequence[str] | None) -> None:
        if not expected_features:
            self.expected_features = ()
            return
        normalized = {
            str(name).strip()
            for name in expected_features
            if isinstance(name, str) and str(name).strip()
        }
        self.expected_features = tuple(sorted(normalized))

    def configure_from_metadata(self, metadata: Mapping[str, Any]) -> None:
        feature_names: Sequence[str] | None = None
        stats = metadata.get("feature_stats")
        if isinstance(stats, Mapping):
            feature_names = tuple(stats.keys())
        if not feature_names:
            raw = metadata.get("feature_names")
            if isinstance(raw, Sequence):
                feature_names = tuple(str(name) for name in raw)
        if not feature_names:
            scalers = metadata.get("feature_scalers")
            if isinstance(scalers, Mapping):
                feature_names = tuple(str(name) for name in scalers.keys())
        self.configure(feature_names)

    @property
    def is_configured(self) -> bool:
        return bool(self.expected_features)

    def observe(
        self, features: Mapping[str, Any], *, context: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        context_payload = _normalize_context(context or {})
        missing = [name for name in self.expected_features if _is_missing(features.get(name))]
        unexpected = [
            str(name)
            for name in features.keys()
            if self.expected_features and str(name) not in self.expected_features
        ]
        null_features = [
            str(name)
            for name, value in features.items()
            if _is_missing(value) and str(name) not in missing
        ]
        status = "alert" if missing or null_features else "ok"
        report = {
            "category": "completeness",
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "expected_features": list(self.expected_features),
            "missing_features": sorted(missing),
            "null_features": sorted(null_features),
            "unexpected_features": sorted(set(unexpected)),
            "context": context_payload,
        }
        self.last_report_path = export_data_quality_report(report, category="completeness")
        report["report_path"] = str(self.last_report_path)
        return report


@dataclass(slots=True)
class FeatureBoundsValidator:
    """Waliduje czy wartości cech mieszczą się w oczekiwanych przedziałach."""

    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    last_report_path: Path | None = field(init=False, default=None, repr=False)

    def configure(self, bounds: Mapping[str, Mapping[str, Any]] | None) -> None:
        if not bounds:
            self.bounds = {}
            return
        parsed: dict[str, tuple[float, float]] = {}
        for name, payload in bounds.items():
            if not isinstance(payload, Mapping):
                continue
            try:
                minimum = float(payload.get("min"))
            except (TypeError, ValueError):
                minimum = float("-inf")
            try:
                maximum = float(payload.get("max"))
            except (TypeError, ValueError):
                maximum = float("inf")
            parsed[str(name)] = (minimum, maximum)
        self.bounds = parsed

    def configure_from_metadata(self, metadata: Mapping[str, Any]) -> None:
        stats = metadata.get("feature_stats")
        if isinstance(stats, Mapping) and stats:
            self.configure(stats)
            return
        scalers = metadata.get("feature_scalers")
        parsed: dict[str, Mapping[str, float]] = {}
        if isinstance(scalers, Mapping):
            for name, payload in scalers.items():
                if not isinstance(payload, Mapping):
                    continue
                mean = float(payload.get("mean", 0.0))
                stdev = float(payload.get("stdev", 0.0))
                span = abs(stdev) * 5 if math.isfinite(stdev) else 0.0
                parsed[str(name)] = {"min": mean - span, "max": mean + span}
        self.configure(parsed)

    @property
    def is_configured(self) -> bool:
        return bool(self.bounds)

    def observe(
        self, features: Mapping[str, Any], *, context: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        context_payload = _normalize_context(context or {})
        violations: list[dict[str, Any]] = []
        for name, (minimum, maximum) in self.bounds.items():
            raw = features.get(name)
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                violations.append({
                    "feature": name,
                    "value": raw,
                    "min": minimum,
                    "max": maximum,
                    "reason": "non_numeric",
                })
                continue
            if value < minimum or value > maximum:
                violations.append({
                    "feature": name,
                    "value": value,
                    "min": minimum,
                    "max": maximum,
                    "reason": "out_of_bounds",
                })
        status = "alert" if violations else "ok"
        report = {
            "category": "bounds",
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": context_payload,
            "violations": violations,
            "monitored_features": sorted(self.bounds.keys()),
        }
        self.last_report_path = export_data_quality_report(report, category="bounds")
        report["report_path"] = str(self.last_report_path)
        return report

class DataQualityException(RuntimeError):
    """Sygnał, że monitoring jakości danych wykrył krytyczne naruszenia."""

    def __init__(
        self,
        reports: Mapping[str, Mapping[str, Any]],
        message: str | None = None,
    ) -> None:
        payload: dict[str, Mapping[str, Any]] = {
            str(name): dict(report)
            for name, report in reports.items()
        }
        super().__init__(message or "Data quality checks failed")
        self._reports: Mapping[str, Mapping[str, Any]] = MappingProxyType(payload)

    @property
    def reports(self) -> Mapping[str, Mapping[str, Any]]:
        """Zwraca raporty monitoringu, które spowodowały wyjątek."""

        return self._reports

