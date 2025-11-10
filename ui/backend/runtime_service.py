"""Serwis runtime dostarczający dane dziennika decyzji do QML."""
from __future__ import annotations

import json
import logging
import os
import math
import queue
import statistics
import threading
import time
from collections import Counter, deque
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from PySide6.QtCore import QObject, Property, QTimer, Signal, Slot

from bot_core.config import load_core_config
from bot_core.portfolio import resolve_decision_log_config
from bot_core.runtime.journal import TradingDecisionJournal
from .demo_data import load_demo_decisions

try:  # pragma: no cover - moduł może nie być dostępny w wersjach light
    from bot_core.ai import ModelRepository
except Exception:  # pragma: no cover - fallback dla dystrybucji bez komponentu AI
    ModelRepository = None  # type: ignore[assignment]

try:  # pragma: no cover - harmonogram retrainingu jest opcjonalny
    from bot_core.runtime.ai_retrain import CronSchedule
except Exception:  # pragma: no cover - fallback dla środowisk bez retrainingu
    CronSchedule = None  # type: ignore[assignment]

try:  # pragma: no cover - funkcja ładowania runtime może nie być dostępna w starszych gałęziach
    from bot_core.config.loader import load_runtime_app_config
except Exception:  # pragma: no cover - fallback gdy brak unified loadera
    load_runtime_app_config = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - adnotacje tylko w czasie statycznym
    from bot_core.config.models import RuntimeAppConfig

_LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - gRPC może nie być dostępne w trybach light
    import grpc
except Exception:  # pragma: no cover - fallback gdy brak gRPC
    grpc = None  # type: ignore[assignment]

DecisionRecord = Mapping[str, str]
DecisionLoader = Callable[[int], Iterable[DecisionRecord]]


@dataclass(slots=True)
class _RuntimeDecisionEntry:
    """Struktura pośrednia używana w konwersji rekordów."""

    event: str
    timestamp: str
    environment: str
    portfolio: str
    risk_profile: str
    schedule: str | None
    strategy: str | None
    symbol: str | None
    side: str | None
    status: str | None
    quantity: str | None
    price: str | None
    market_regime: Mapping[str, object]
    decision: Mapping[str, object]
    ai: Mapping[str, object]
    extras: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "event": self.event,
            "timestamp": self.timestamp,
            "environment": self.environment,
            "portfolio": self.portfolio,
            "riskProfile": self.risk_profile,
            "schedule": self.schedule,
            "strategy": self.strategy,
            "symbol": self.symbol,
            "side": self.side,
            "status": self.status,
            "quantity": self.quantity,
            "price": self.price,
            "marketRegime": dict(self.market_regime),
            "decision": dict(self.decision),
            "ai": dict(self.ai),
            "metadata": dict(self.extras),
        }


def _default_loader(limit: int) -> Iterable[DecisionRecord]:
    """Zapewnia dane demonstracyjne przy pierwszym uruchomieniu."""

    entries = list(load_demo_decisions())
    if not entries:
        return []
    if limit > 0:
        entries = entries[-limit:]
    # Zwracamy w kolejności od najnowszych do najstarszych, aby zachować spójność z dziennikiem
    return reversed(entries)


def _load_from_journal(journal: TradingDecisionJournal, limit: int) -> Iterable[DecisionRecord]:
    exported = list(journal.export())
    if limit > 0:
        exported = exported[-limit:]
    return reversed(exported)


def _normalize_bool(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return value


def _camelize(prefix: str, key: str) -> str:
    suffix = key[len(prefix) :].lstrip("_")
    if not suffix:
        return ""
    parts = [part for part in suffix.split("_") if part]
    if not parts:
        return ""
    head, *tail = parts
    return head + "".join(segment.capitalize() for segment in tail)


_BASE_FIELD_MAP: Mapping[str, str] = {
    "event": "event",
    "timestamp": "timestamp",
    "environment": "environment",
    "portfolio": "portfolio",
    "risk_profile": "risk_profile",
    "schedule": "schedule",
    "strategy": "strategy",
    "symbol": "symbol",
    "side": "side",
    "status": "status",
    "quantity": "quantity",
    "price": "price",
}


def _parse_entry(record: DecisionRecord) -> _RuntimeDecisionEntry:
    base: MutableMapping[str, str | None] = {key: None for key in _BASE_FIELD_MAP.values()}
    decision_payload: MutableMapping[str, object] = {}
    ai_payload: MutableMapping[str, object] = {}
    regime_payload: MutableMapping[str, object] = {}
    extras: MutableMapping[str, object] = {}

    confidence = record.get("confidence")
    latency = record.get("latency_ms")
    if confidence is not None:
        decision_payload["confidence"] = confidence
    if latency is not None:
        decision_payload["latencyMs"] = latency

    for key, value in record.items():
        if key in {"confidence", "latency_ms"}:
            continue
        mapped = _BASE_FIELD_MAP.get(key)
        if mapped is not None:
            base[mapped] = value
            continue
        if key.startswith("decision_"):
            normalized = _camelize("decision_", key)
            if not normalized:
                continue
            payload_value: object = value
            if normalized == "shouldTrade":
                payload_value = _normalize_bool(value)
            decision_payload[normalized] = payload_value
            continue
        if key.startswith("ai_"):
            normalized = _camelize("ai_", key)
            if not normalized:
                continue
            ai_payload[normalized] = value
            continue
        if key.startswith("market_regime"):
            normalized = _camelize("market_regime", key)
            if not normalized:
                continue
            regime_payload[normalized] = value
            continue
        if key == "risk_profile":
            # już zmapowane do base
            continue
        extras[key] = value

    event = str(base["event"] or "")
    timestamp = str(base["timestamp"] or "")
    environment = str(base["environment"] or "")
    portfolio = str(base["portfolio"] or "")
    risk_profile = str(base["risk_profile"] or "")

    schedule = base.get("schedule")
    strategy = base.get("strategy")
    symbol = base.get("symbol")
    side = base.get("side")
    status = base.get("status")
    quantity = base.get("quantity")
    price = base.get("price")

    return _RuntimeDecisionEntry(
        event=event,
        timestamp=timestamp,
        environment=environment,
        portfolio=portfolio,
        risk_profile=risk_profile,
        schedule=schedule,
        strategy=strategy,
        symbol=symbol,
        side=side,
        status=status,
        quantity=quantity,
        price=price,
        market_regime=regime_payload,
        decision=decision_payload,
        ai=ai_payload,
        extras=extras,
    )


def _normalize_sequence(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        sanitized = [segment.strip() for segment in value.replace(";", ",").split(",")]
        return [segment for segment in sanitized if segment]
    if isinstance(value, Mapping):
        reason = value.get("reason")
        if isinstance(reason, str) and reason.strip():
            return [reason.strip()]
        # fallback: reprezentacja tekstowa całego obiektu
        try:
            return [json.dumps(dict(value), ensure_ascii=False)]
        except TypeError:
            return [str(dict(value))]
    if isinstance(value, Iterable):
        result: list[str] = []
        for item in value:
            result.extend(_normalize_sequence(item))
        return result
    return []


def _to_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return {}


def _first_non_empty(*values: object) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_risk_action(
    metadata: Mapping[str, object], decision: Mapping[str, object], status: str
) -> str:
    candidates = (
        metadata.get("riskAction"),
        metadata.get("risk_action"),
        metadata.get("action"),
        decision.get("riskAction"),
        decision.get("risk_action"),
        decision.get("action"),
    )
    action = _first_non_empty(*candidates)
    if action:
        return action
    if status.lower() in {"blocked", "rejected", "risk_block"}:
        return status
    return ""


def _compute_activity_score(is_block: bool, is_freeze: bool, is_override: bool) -> float:
    if is_block:
        return 1.0
    if is_freeze:
        return 0.85
    if is_override:
        return 0.7
    return 0.45


def _build_risk_context(
    entries: Iterable[Mapping[str, object]]
) -> tuple[dict[str, object], list[dict[str, object]]]:
    block_keywords = {"block", "blocked", "risk_block", "reject", "rejected"}
    freeze_keywords = {"freeze", "frozen", "lock"}
    override_keys = {"stressOverride", "stress_override", "stressOverrides", "stress_overrides"}

    blocks = 0
    freezes = 0
    overrides = 0
    total_entries = 0
    latest_stress_failures: list[str] = []
    latest_failure_ts = ""
    latest_risk_flags: list[str] = []
    last_block_entry: dict[str, object] | None = None
    last_freeze_entry: dict[str, object] | None = None
    last_override_entry: dict[str, object] | None = None
    stress_failure_set: set[str] = set()
    risk_flag_set: set[str] = set()
    strategy_set: set[str] = set()
    strategy_counter: Counter[str] = Counter()
    override_reason_set: set[str] = set()
    stress_failure_counter: Counter[str] = Counter()
    risk_flag_counter: Counter[str] = Counter()
    strategy_summaries: dict[str, dict[str, object]] = {}

    timeline: list[dict[str, object]] = []

    for entry in entries:
        total_entries += 1
        metadata = _to_mapping(entry.get("metadata"))
        decision = _to_mapping(entry.get("decision"))
        timestamp = str(entry.get("timestamp") or "")
        event = str(entry.get("event") or "")
        strategy = str(entry.get("strategy") or "").strip()
        risk_profile = str(entry.get("riskProfile") or "").strip()
        status = str(entry.get("status") or "").strip()
        risk_action = _extract_risk_action(metadata, decision, status)

        risk_flags = _normalize_sequence(
            metadata.get("riskFlags")
            or metadata.get("risk_flags")
            or decision.get("riskFlags")
            or decision.get("risk_flags")
        )
        stress_failures = _normalize_sequence(
            metadata.get("stressFailures")
            or metadata.get("stress_failures")
            or decision.get("stressFailures")
            or decision.get("stress_failures")
        )

        stress_overrides_payload = metadata.get("stressOverrides") or metadata.get("stress_overrides")
        if not stress_overrides_payload:
            stress_overrides_payload = decision.get("stressOverrides") or decision.get("stress_overrides")
        stress_overrides = _normalize_sequence(stress_overrides_payload)

        is_block = any(keyword in risk_action.lower() for keyword in block_keywords) if risk_action else False
        if not is_block and status:
            is_block = any(keyword in status.lower() for keyword in block_keywords)

        is_freeze = any(keyword in risk_action.lower() for keyword in freeze_keywords) if risk_action else False
        if not is_freeze and status:
            is_freeze = any(keyword in status.lower() for keyword in freeze_keywords)

        override_indicator = False
        if stress_overrides:
            override_indicator = True
        else:
            for key in override_keys:
                if metadata.get(key) or decision.get(key):
                    override_indicator = True
                    break
        is_override = override_indicator

        activity_score = _compute_activity_score(is_block, is_freeze, is_override)

        summary_bucket: dict[str, object] | None = None
        if strategy:
            strategy_set.add(strategy)
            strategy_counter.update([strategy])
            summary_bucket = strategy_summaries.setdefault(
                strategy,
                {
                    "strategy": strategy,
                    "blockCount": 0,
                    "freezeCount": 0,
                    "stressOverrideCount": 0,
                    "totalEvents": 0,
                    "lastEvent": "",
                    "lastTimestamp": "",
                    "lastRiskFlags": [],
                    "lastStressFailures": [],
                    "lastRiskAction": "",
                    "stressOverrideReasons": [],
                },
            )
        if summary_bucket is not None:
            summary_bucket["totalEvents"] = int(summary_bucket.get("totalEvents", 0)) + 1
        if risk_flags:
            risk_flag_set.update(risk_flags)
            risk_flag_counter.update(risk_flags)
            latest_risk_flags = risk_flags
            if summary_bucket is not None and risk_flags:
                summary_bucket["lastRiskFlags"] = list(risk_flags)
        if stress_failures:
            stress_failure_set.update(stress_failures)
            stress_failure_counter.update(stress_failures)
            latest_stress_failures = stress_failures
            latest_failure_ts = timestamp
            if summary_bucket is not None and stress_failures:
                summary_bucket["lastStressFailures"] = list(stress_failures)
        if stress_overrides:
            override_reason_set.update(stress_overrides)
            if summary_bucket is not None and stress_overrides:
                summary_bucket["stressOverrideReasons"] = list(stress_overrides)

        if is_block:
            blocks += 1
            if timestamp and (
                not last_block_entry
                or str(timestamp) >= str(last_block_entry.get("timestamp", ""))
            ):
                last_block_entry = {
                    "timestamp": timestamp,
                    "event": event,
                    "strategy": strategy,
                }
            if summary_bucket is not None:
                summary_bucket["blockCount"] = int(summary_bucket.get("blockCount", 0)) + 1
        if is_freeze:
            freezes += 1
            if timestamp and (
                not last_freeze_entry
                or str(timestamp) >= str(last_freeze_entry.get("timestamp", ""))
            ):
                last_freeze_entry = {
                    "timestamp": timestamp,
                    "event": event,
                    "strategy": strategy,
                    "riskAction": risk_action,
                }
            if summary_bucket is not None:
                summary_bucket["freezeCount"] = int(summary_bucket.get("freezeCount", 0)) + 1
        if is_override:
            overrides += 1
            if timestamp and (
                not last_override_entry
                or str(timestamp) >= str(last_override_entry.get("timestamp", ""))
            ):
                last_override_entry = {
                    "timestamp": timestamp,
                    "event": event,
                    "strategy": strategy,
                }
            if summary_bucket is not None:
                summary_bucket["stressOverrideCount"] = int(
                    summary_bucket.get("stressOverrideCount", 0)
                ) + 1

        if summary_bucket is not None:
            if risk_action:
                summary_bucket["lastRiskAction"] = risk_action
            if timestamp:
                last_timestamp = str(summary_bucket.get("lastTimestamp", ""))
                if not last_timestamp or str(timestamp) >= last_timestamp:
                    summary_bucket["lastTimestamp"] = timestamp
                    summary_bucket["lastEvent"] = event
                    summary_bucket["lastRiskFlags"] = list(risk_flags)
                    summary_bucket["lastStressFailures"] = list(stress_failures)

        timeline.append(
            {
                "timestamp": timestamp,
                "event": event,
                "strategy": strategy,
                "riskProfile": risk_profile,
                "status": status,
                "riskAction": risk_action,
                "riskFlags": risk_flags,
                "stressFailures": stress_failures,
                "stressOverrides": stress_overrides,
                "isBlock": is_block,
                "isFreeze": is_freeze,
                "isStressOverride": is_override,
                "activityScore": activity_score,
                "record": dict(entry),
            }
        )

    def _sort_key(item: Mapping[str, object]) -> tuple[int, str]:
        timestamp = str(item.get("timestamp") or "")
        return (0 if timestamp else 1, timestamp)

    timeline.sort(key=_sort_key)

    metrics: dict[str, object] = {"totalEntries": total_entries}

    severity_order = {"block": 0, "freeze": 1, "override": 2, "neutral": 3}

    def _classify(summary: Mapping[str, object]) -> str:
        if int(summary.get("blockCount", 0)) > 0:
            return "block"
        if int(summary.get("freezeCount", 0)) > 0:
            return "freeze"
        if int(summary.get("stressOverrideCount", 0)) > 0:
            return "override"
        return "neutral"

    def _timestamp_key(value: object) -> tuple[int, str]:
        if isinstance(value, str) and value:
            candidate = value.replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(candidate)
            except ValueError:
                return (0, candidate)
            return (-int(parsed.timestamp()), candidate)
        return (0, "")

    summaries_payload: list[dict[str, object]] = []
    for summary in strategy_summaries.values():
        payload = dict(summary)
        payload["severity"] = _classify(summary)
        payload.setdefault("stressOverrideReasons", [])
        summaries_payload.append(payload)

    summaries_payload.sort(
        key=lambda item: (
            severity_order.get(str(item.get("severity", "neutral")), 99),
            -int(item.get("blockCount", 0)),
            -int(item.get("freezeCount", 0)),
            -int(item.get("stressOverrideCount", 0)),
            _timestamp_key(item.get("lastTimestamp", "")),
            str(item.get("strategy", "")),
        )
    )

    metrics.update(
        {
            "blockCount": blocks,
            "freezeCount": freezes,
            "stressOverrideCount": overrides,
            "lastBlock": dict(last_block_entry or {}),
            "lastFreeze": dict(last_freeze_entry or {}),
            "lastStressOverride": dict(last_override_entry or {}),
            "latestStressFailures": latest_stress_failures,
            "latestStressFailureAt": latest_failure_ts,
            "latestRiskFlags": latest_risk_flags,
            "uniqueRiskFlags": sorted(risk_flag_set),
            "uniqueStressFailures": sorted(stress_failure_set),
            "strategies": sorted(strategy_set),
            "strategyCounts": dict(strategy_counter),
            "stressOverrideReasons": sorted(override_reason_set),
            "riskFlagCounts": dict(risk_flag_counter),
            "stressFailureCounts": dict(stress_failure_counter),
            "timelineStart": timeline[0]["timestamp"] if timeline else "",
            "timelineEnd": timeline[-1]["timestamp"] if timeline else "",
            "strategySummaries": summaries_payload,
        }
    )

    return metrics, timeline


class RuntimeService(QObject):
    """Zapewnia QML dostęp do najnowszych decyzji autotradera."""

    decisionsChanged = Signal()
    errorMessageChanged = Signal()
    liveSourceChanged = Signal()
    retrainNextRunChanged = Signal()
    adaptiveStrategySummaryChanged = Signal()
    riskMetricsChanged = Signal()
    riskTimelineChanged = Signal()
    operatorActionChanged = Signal()

    def __init__(
        self,
        *,
        journal: TradingDecisionJournal | None = None,
        decision_loader: DecisionLoader | None = None,
        parent: QObject | None = None,
        default_limit: int = 20,
        core_config_path: str | os.PathLike[str] | None = None,
        runtime_config_path: str | os.PathLike[str] | None = None,
    ) -> None:
        super().__init__(parent)
        if decision_loader is not None:
            self._loader: DecisionLoader = decision_loader
        elif journal is not None:
            self._loader = lambda limit: _load_from_journal(journal, limit)
        else:
            self._loader = _default_loader
        self._default_limit = max(1, int(default_limit))
        self._decisions: list[dict[str, object]] = []
        self._error_message = ""
        self._core_config_path = Path(core_config_path).expanduser() if core_config_path else None
        self._cached_core_config = None
        self._active_profile: str | None = None
        self._active_log_path: Path | None = None
        self._active_stream_label: str | None = None
        self._runtime_config_path = Path(runtime_config_path).expanduser() if runtime_config_path else None
        self._runtime_config_cache: "RuntimeAppConfig | None" = None
        self._retrain_next_run: str = ""
        self._adaptive_summary: str = ""
        self._risk_metrics: dict[str, object] = {}
        self._risk_timeline: list[dict[str, object]] = []
        self._last_operator_action: dict[str, object] | None = None
        self._grpc_thread: threading.Thread | None = None
        self._grpc_stop_event: threading.Event | None = None
        self._grpc_queue: "queue.Queue[tuple[str, object]] | None" = None
        self._grpc_timer: QTimer | None = None
        self._grpc_stream_active = False
        self._grpc_target: str | None = None
        self._grpc_metadata: list[tuple[str, str]] = self._load_grpc_metadata()
        self._grpc_limit = self._default_limit
        metrics_path_env = os.environ.get("BOT_CORE_UI_FEED_LATENCY_PATH")
        if metrics_path_env:
            self._feed_metrics_path = Path(metrics_path_env).expanduser()
        else:
            self._feed_metrics_path = Path("reports/ci/decision_feed_metrics.json")
        self._feed_latencies: deque[float] = deque(maxlen=1024)
        try:
            self._update_runtime_metadata(invalidate_cache=False)
        except Exception:  # pragma: no cover - defensywna inicjalizacja
            _LOGGER.debug("Nie udało się zainicjalizować metadanych runtime", exc_info=True)
        self._auto_connect_grpc()

    # ------------------------------------------------------------------
    @Property("QVariantList", notify=decisionsChanged)
    def decisions(self) -> list[dict[str, object]]:  # type: ignore[override]
        return list(self._decisions)

    @Property(str, notify=errorMessageChanged)
    def errorMessage(self) -> str:  # type: ignore[override]
        return self._error_message

    @Property(str, notify=retrainNextRunChanged)
    def retrainNextRun(self) -> str:  # type: ignore[override]
        return self._retrain_next_run

    @Property(str, notify=adaptiveStrategySummaryChanged)
    def adaptiveStrategySummary(self) -> str:  # type: ignore[override]
        return self._adaptive_summary

    @Property("QVariantMap", notify=riskMetricsChanged)
    def riskMetrics(self) -> dict[str, object]:  # type: ignore[override]
        return dict(self._risk_metrics)

    @Property("QVariantList", notify=riskTimelineChanged)
    def riskTimeline(self) -> list[dict[str, object]]:  # type: ignore[override]
        return list(self._risk_timeline)

    @Property("QVariantMap", notify=operatorActionChanged)
    def lastOperatorAction(self) -> dict[str, object]:  # type: ignore[override]
        if self._last_operator_action is None:
            return {}
        return dict(self._last_operator_action)

    # ------------------------------------------------------------------
    @Slot(int, result="QVariantList")
    def loadRecentDecisions(self, limit: int = 0) -> list[dict[str, object]]:  # type: ignore[override]
        """Pobiera najnowsze decyzje z dziennika."""

        size = int(limit)
        if size <= 0:
            size = self._default_limit
        if self._grpc_stream_active:
            subset = self._decisions[:size]
            return [dict(entry) for entry in subset]
        try:
            raw_entries = list(self._loader(size))
        except Exception as exc:  # pragma: no cover - diagnostyka
            self._error_message = str(exc)
            self.errorMessageChanged.emit()
            self._risk_metrics = {}
            self._risk_timeline = []
            self.riskMetricsChanged.emit()
            self.riskTimelineChanged.emit()
            return []

        self._error_message = ""
        self.errorMessageChanged.emit()

        parsed: list[dict[str, object]] = []
        for record in raw_entries:
            entry = _parse_entry(record)
            payload = entry.to_payload()
            parsed.append(payload)
        self._decisions = parsed
        self.decisionsChanged.emit()
        self._apply_risk_context(parsed)
        self._update_runtime_metadata(invalidate_cache=False)
        return list(self._decisions)

    @Slot()
    def refreshRuntimeMetadata(self) -> None:  # type: ignore[override]
        """Wymusza ponowne wczytanie metadanych retrainingu i presetów adaptacyjnych."""

        self._update_runtime_metadata(invalidate_cache=True)

    # ------------------------------------------------------------------
    @Property(str, notify=liveSourceChanged)
    def activeDecisionLogPath(self) -> str:  # type: ignore[override]
        if self._active_stream_label:
            return self._active_stream_label
        if self._active_log_path is None:
            return ""
        return str(self._active_log_path)

    @Slot(str, result=bool)
    def attachToLiveDecisionLog(self, profile: str = "") -> bool:  # type: ignore[override]
        """Przełącza loader na rzeczywisty decision log skonfigurowany w core.yaml."""

        self._stop_grpc_stream()
        sanitized_profile = profile.strip()
        profile_value = sanitized_profile or None
        target = self._resolve_grpc_target(profile_value)
        if target:
            try:
                self._start_grpc_stream(target, self._default_limit)
            except Exception as exc:  # pragma: no cover - diagnostyka
                _LOGGER.debug("attachToLiveDecisionLog gRPC failed", exc_info=True)
                self._active_profile = profile_value
                return self._handle_grpc_error(str(exc), profile=profile_value, silent=False)
            else:
                self._loader = lambda limit: []
                self._active_profile = profile_value
                self._active_log_path = None
                self._active_stream_label = f"grpc://{target}"
                self._error_message = ""
                self.errorMessageChanged.emit()
                self.liveSourceChanged.emit()
                return True

        if self._activate_jsonl_loader(profile_value, silent=False):
            return True
        return False

    # ------------------------------------------------------------------ operator actions --
    @Slot("QVariantMap", result=bool)
    def requestFreeze(self, entry: Mapping[str, object] | None = None) -> bool:  # type: ignore[override]
        return self._record_operator_action("freeze", entry)

    @Slot("QVariantMap", result=bool)
    def requestUnfreeze(self, entry: Mapping[str, object] | None = None) -> bool:  # type: ignore[override]
        return self._record_operator_action("unfreeze", entry)

    @Slot("QVariantMap", result=bool)
    def requestUnblock(self, entry: Mapping[str, object] | None = None) -> bool:  # type: ignore[override]
        return self._record_operator_action("unblock", entry)

    # ------------------------------------------------------------------
    def _build_live_loader(
        self, profile: str | None
    ) -> tuple[DecisionLoader, Path]:
        core_config = self._load_core_config()
        configured_path, _kwargs = resolve_decision_log_config(core_config)
        if configured_path is None:
            raise FileNotFoundError(
                "Decision log portfela nie jest skonfigurowany w pliku core.yaml"
            )

        log_path = Path(configured_path)
        if not log_path.is_absolute():
            config_path = self._resolve_core_config_path()
            if config_path is not None:
                log_path = (config_path.parent / log_path).resolve()
            else:
                log_path = log_path.expanduser().resolve()

        if not log_path.exists():
            raise FileNotFoundError(
                f"Decision log '{log_path}' nie istnieje – uruchom autotradera, aby utworzyć plik"
            )
        if not log_path.is_file():
            raise IsADirectoryError(
                f"Decision log '{log_path}' wskazuje na katalog – oczekiwany plik JSONL"
            )

        loader = self._build_jsonl_loader(log_path)
        return loader, log_path

    def _build_jsonl_loader(self, log_path: Path) -> DecisionLoader:
        def _loader(limit: int) -> Iterable[DecisionRecord]:
            entries: list[DecisionRecord] = []
            try:
                with log_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        payload = line.strip()
                        if not payload:
                            continue
                        try:
                            data = json.loads(payload)
                        except json.JSONDecodeError:
                            _LOGGER.warning(
                                "Pominięto uszkodzony wpis decision logu %s", log_path, exc_info=True
                            )
                            continue
                        if isinstance(data, Mapping):
                            entries.append(data)
            except FileNotFoundError:
                raise
            except OSError as exc:
                raise RuntimeError(
                    f"Nie udało się odczytać decision logu '{log_path}': {exc}"
                ) from exc

            if limit > 0:
                entries = entries[-limit:]
            return entries

        return _loader

    # ------------------------------------------------------------------ risk aggregation helpers --
    def _resolve_grpc_target(self, profile: str | None) -> str | None:
        candidates = (
            os.environ.get("BOT_CORE_UI_GRPC_ENDPOINT"),
            os.environ.get("BOT_CORE_TRADING_GRPC_ADDRESS"),
            os.environ.get("BOT_CORE_RUNTIME_GRPC_ADDRESS"),
        )
        for candidate in candidates:
            if candidate and candidate.strip():
                return candidate.strip()
        return None

    def _load_grpc_metadata(self) -> list[tuple[str, str]]:
        raw = os.environ.get("BOT_CORE_UI_GRPC_METADATA", "")
        if not raw:
            return []
        metadata: list[tuple[str, str]] = []
        for item in raw.split(","):
            if not item:
                continue
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if key:
                metadata.append((key, value))
        return metadata

    def _auto_connect_grpc(self) -> None:
        target = self._resolve_grpc_target(self._active_profile)
        if not target:
            return
        try:
            self._start_grpc_stream(target, self._default_limit)
        except Exception as exc:  # pragma: no cover - defensywne logowanie
            _LOGGER.debug("Auto gRPC bootstrap failed", exc_info=True)
            self._handle_grpc_error(str(exc), profile=self._active_profile, silent=True)
        else:
            self._active_log_path = None
            self._active_stream_label = f"grpc://{target}"
            self._error_message = ""
            self.errorMessageChanged.emit()
            self.liveSourceChanged.emit()

    def _activate_jsonl_loader(self, profile: str | None, *, silent: bool) -> bool:
        try:
            loader, log_path = self._build_live_loader(profile)
        except Exception as exc:
            if not silent:
                self._error_message = str(exc)
                self.errorMessageChanged.emit()
            return False

        self._loader = loader
        self._active_profile = profile
        self._active_log_path = log_path
        self._active_stream_label = None
        self._grpc_stream_active = False
        self.liveSourceChanged.emit()
        self.loadRecentDecisions(self._default_limit)
        if not silent:
            self._error_message = ""
            self.errorMessageChanged.emit()
        return True

    def _use_demo_loader(self, message: str | None, *, profile: str | None, silent: bool) -> None:
        self._loader = _default_loader
        self._active_profile = profile
        self._active_log_path = None
        self._active_stream_label = "offline-demo"
        self._grpc_stream_active = False
        self.liveSourceChanged.emit()
        self.loadRecentDecisions(self._default_limit)
        if not silent:
            self._error_message = message or ""
            self.errorMessageChanged.emit()

    def _handle_grpc_error(self, message: str, *, profile: str | None, silent: bool) -> bool:
        self._stop_grpc_stream()
        if self._activate_jsonl_loader(profile, silent=True):
            if not silent:
                self._error_message = message
                self.errorMessageChanged.emit()
            return True
        self._use_demo_loader(message if not silent else None, profile=profile, silent=silent)
        return True

    def _ensure_grpc_timer(self) -> None:
        if self._grpc_timer is not None:
            return
        timer = QTimer(self)
        timer.setInterval(200)
        timer.setSingleShot(False)
        timer.timeout.connect(self._drain_grpc_queue)
        timer.start()
        self._grpc_timer = timer

    def _start_grpc_stream(self, target: str, limit: int) -> None:
        if grpc is None:
            raise RuntimeError("Pakiet grpcio jest wymagany do połączenia z RuntimeService.")
        try:
            from bot_core.generated import trading_pb2, trading_pb2_grpc
        except ImportError as exc:  # pragma: no cover - brak stubów
            raise RuntimeError(
                "Brak wygenerowanych stubów trading_pb2*_grpc – uruchom scripts/generate_trading_stubs.py"
            ) from exc

        self._grpc_stop_event = threading.Event()
        self._grpc_queue = queue.Queue()
        self._grpc_target = target
        self._grpc_stream_active = True
        self._grpc_limit = max(1, int(limit))
        self._decisions = []
        self._ensure_grpc_timer()
        worker = threading.Thread(
            target=self._grpc_worker,
            args=(target, trading_pb2, trading_pb2_grpc, self._grpc_limit),
            name="RuntimeServiceGrpc",
            daemon=True,
        )
        self._grpc_thread = worker
        worker.start()

    def _grpc_worker(
        self,
        target: str,
        trading_pb2,
        trading_pb2_grpc,
        limit: int,
    ) -> None:
        queue_obj = self._grpc_queue
        if queue_obj is None:
            return
        stop_event = self._grpc_stop_event
        request = trading_pb2.StreamDecisionsRequest(
            limit=max(0, int(limit)),
            skip_snapshot=False,
            poll_interval_seconds=1.0,
        )
        metadata = tuple(self._grpc_metadata)
        channel = None
        try:
            channel = grpc.insecure_channel(target)
            stub = trading_pb2_grpc.RuntimeServiceStub(channel)
            if metadata:
                stream = stub.StreamDecisions(request, metadata=metadata)
            else:
                stream = stub.StreamDecisions(request)
            for update in stream:
                if stop_event is not None and stop_event.is_set():
                    break
                if update.HasField("snapshot"):
                    payload = [dict(entry.fields) for entry in update.snapshot.records]
                    queue_obj.put(("snapshot", payload))
                elif update.HasField("increment"):
                    queue_obj.put(("increment", dict(update.increment.record.fields)))
            queue_obj.put(("done", None))
        except Exception as exc:  # pragma: no cover - diagnostyka
            queue_obj.put(("error", str(exc)))
            queue_obj.put(("done", None))
        finally:
            if channel is not None:
                try:
                    channel.close()
                except Exception:
                    pass

    def _drain_grpc_queue(self) -> None:
        queue_obj = self._grpc_queue
        if queue_obj is None:
            return
        updated = False
        while True:
            try:
                kind, payload = queue_obj.get_nowait()
            except queue.Empty:
                break
            if kind == "snapshot":
                if isinstance(payload, list):
                    self._apply_grpc_snapshot(payload)
                    updated = True
                queue_obj.task_done()
                continue
            if kind == "increment":
                if isinstance(payload, Mapping):
                    self._append_grpc_record(payload)
                    updated = True
                queue_obj.task_done()
                continue
            if kind == "error":
                self._handle_grpc_error(str(payload), profile=self._active_profile, silent=False)
                queue_obj.task_done()
                return
            if kind == "done":
                if self._grpc_stop_event is None or not self._grpc_stop_event.is_set():
                    self._grpc_stream_active = False
                queue_obj.task_done()
                continue
            queue_obj.task_done()

        if updated:
            self.decisionsChanged.emit()
            self._apply_risk_context(self._decisions)
            self._update_runtime_metadata(invalidate_cache=False)
            self._error_message = ""
            self.errorMessageChanged.emit()
            self._write_feed_metrics()

    def _apply_grpc_snapshot(self, records: Iterable[Mapping[str, str]]) -> None:
        collected: list[dict[str, object]] = []
        for record in reversed(list(records)):
            if not isinstance(record, Mapping):
                continue
            try:
                entry = _parse_entry(record)
            except Exception:  # pragma: no cover - diagnostyka
                _LOGGER.debug("Nie udało się sparsować snapshotu gRPC", exc_info=True)
                continue
            payload = entry.to_payload()
            collected.append(payload)
            self._record_feed_latency(record)
        if not collected:
            return
        max_size = max(1, self._default_limit)
        if len(collected) > max_size:
            collected = collected[:max_size]
        self._decisions = collected

    def _append_grpc_record(self, record: Mapping[str, str]) -> None:
        try:
            entry = _parse_entry(record)
        except Exception:  # pragma: no cover - diagnostyka
            _LOGGER.debug("Nie udało się sparsować przyrostu gRPC", exc_info=True)
            return
        payload = entry.to_payload()
        self._record_feed_latency(record)
        self._decisions.insert(0, payload)
        if len(self._decisions) > self._default_limit:
            self._decisions = self._decisions[: self._default_limit]

    def _record_feed_latency(self, record: Mapping[str, str]) -> None:
        timestamp_raw = record.get("timestamp")
        if not timestamp_raw:
            return
        text = str(timestamp_raw).strip()
        if not text:
            return
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        latency_ms = (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds() * 1000.0
        if latency_ms < 0:
            latency_ms = 0.0
        self._feed_latencies.append(float(latency_ms))

    def _write_feed_metrics(self) -> None:
        latencies = list(self._feed_latencies)
        if not latencies:
            return
        stats_payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "avg_ms": sum(latencies) / len(latencies),
            "p50_ms": statistics.median(latencies),
            "p95_ms": self._percentile(latencies, 95.0),
        }
        try:
            self._feed_metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self._feed_metrics_path.write_text(
                json.dumps(stats_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:  # pragma: no cover - zapisy metryk nie powinny blokować UI
            _LOGGER.debug("Nie udało się zapisać metryk latencji feedu", exc_info=True)

    @staticmethod
    def _percentile(values: Iterable[float], percentile: float) -> float:
        data = sorted(values)
        if not data:
            return 0.0
        if percentile <= 0:
            return data[0]
        if percentile >= 100:
            return data[-1]
        rank = (percentile / 100.0) * (len(data) - 1)
        low = math.floor(rank)
        high = math.ceil(rank)
        if low == high:
            return data[int(rank)]
        fraction = rank - low
        return data[low] + (data[high] - data[low]) * fraction

    def _stop_grpc_stream(self) -> None:
        if self._grpc_stop_event is not None:
            self._grpc_stop_event.set()
        if self._grpc_thread is not None and self._grpc_thread.is_alive():
            self._grpc_thread.join(timeout=1.5)
        self._grpc_thread = None
        self._grpc_stop_event = None
        self._grpc_queue = None
        if self._grpc_timer is not None:
            self._grpc_timer.stop()
            self._grpc_timer.deleteLater()
            self._grpc_timer = None
        self._grpc_stream_active = False
        self._active_stream_label = None

    def _apply_risk_context(self, entries: Iterable[Mapping[str, object]]) -> None:
        metrics, timeline = _build_risk_context(entries)
        self._risk_metrics = metrics
        self._risk_timeline = timeline
        self.riskMetricsChanged.emit()
        self.riskTimelineChanged.emit()

    def __del__(self) -> None:  # pragma: no cover - defensywne sprzątanie
        try:
            self._stop_grpc_stream()
        except Exception:
            pass

    def _record_operator_action(
        self, action: str, entry: Mapping[str, object] | None
    ) -> bool:
        sanitized = dict(_to_mapping(entry)) if entry is not None else {}
        timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        self._last_operator_action = {
            "action": action,
            "timestamp": timestamp,
            "entry": sanitized,
        }
        self.operatorActionChanged.emit()
        if sanitized:
            reference = sanitized.get("event") or sanitized.get("timestamp") or sanitized.get("id")
        else:
            reference = None
        if reference:
            _LOGGER.info("Operator action '%s' triggered for %s", action, reference)
        else:
            _LOGGER.info("Operator action '%s' triggered", action)
        return True

    def _load_core_config(self):
        if self._cached_core_config is not None:
            return self._cached_core_config
        config_path = self._resolve_core_config_path()
        if config_path is None:
            raise FileNotFoundError(
                "Nie znaleziono ścieżki do core.yaml – ustaw zmienną BOT_CORE_UI_CORE_CONFIG_PATH"
            )
        self._cached_core_config = load_core_config(config_path)
        return self._cached_core_config

    def _resolve_core_config_path(self) -> Path | None:
        if self._core_config_path is not None:
            return self._core_config_path

        candidates = (
            os.environ.get("BOT_CORE_UI_CORE_CONFIG_PATH"),
            os.environ.get("BOT_CORE_CORE_CONFIG"),
            os.environ.get("BOT_CORE_CONFIG"),
            os.environ.get("DUDZIAN_CORE_CONFIG"),
        )
        for candidate in candidates:
            if candidate:
                path = Path(candidate).expanduser()
                self._core_config_path = path
                return path

        default = Path("config/core.yaml")
        self._core_config_path = default
        return default

    # ------------------------------------------------------------------ runtime metadata helpers --
    def _update_runtime_metadata(self, *, invalidate_cache: bool) -> None:
        if invalidate_cache:
            self._runtime_config_cache = None
        next_run = self._compute_next_retrain()
        if next_run != self._retrain_next_run:
            self._retrain_next_run = next_run
            self.retrainNextRunChanged.emit()
        summary = self._build_adaptive_summary()
        if summary != self._adaptive_summary:
            self._adaptive_summary = summary
            self.adaptiveStrategySummaryChanged.emit()

    def _load_runtime_config(self) -> "RuntimeAppConfig":
        if load_runtime_app_config is None:
            raise RuntimeError("Ładowanie konfiguracji runtime nie jest dostępne w tej dystrybucji")
        if self._runtime_config_cache is not None:
            return self._runtime_config_cache
        config_path = self._resolve_runtime_config_path()
        if config_path is None:
            raise FileNotFoundError(
                "Nie znaleziono pliku runtime.yaml – ustaw zmienną BOT_CORE_UI_RUNTIME_CONFIG_PATH"
            )
        self._runtime_config_cache = load_runtime_app_config(config_path)
        return self._runtime_config_cache

    def _resolve_runtime_config_path(self) -> Path | None:
        if self._runtime_config_path is not None:
            return self._runtime_config_path

        candidates = (
            os.environ.get("BOT_CORE_UI_RUNTIME_CONFIG_PATH"),
            os.environ.get("BOT_CORE_RUNTIME_CONFIG_PATH"),
            os.environ.get("BOT_CORE_RUNTIME_CONFIG"),
            os.environ.get("DUDZIAN_RUNTIME_CONFIG"),
        )
        for candidate in candidates:
            if candidate:
                path = Path(candidate).expanduser()
                self._runtime_config_path = path
                return path

        default = Path("config/runtime.yaml")
        self._runtime_config_path = default
        return default

    def _compute_next_retrain(self) -> str:
        if CronSchedule is None:
            return ""
        try:
            runtime_config = self._load_runtime_config()
        except FileNotFoundError:
            return ""
        except Exception:  # pragma: no cover - diagnostyka środowiska
            _LOGGER.debug("Nie udało się wczytać konfiguracji runtime", exc_info=True)
            return ""

        schedule: str | None = None
        retrain_cfg = getattr(runtime_config.ai, "retrain", None)
        if retrain_cfg and getattr(retrain_cfg, "enabled", False):
            schedule = getattr(retrain_cfg, "schedule", None) or getattr(runtime_config.ai, "retrain_schedule", None)
        else:
            schedule = getattr(runtime_config.ai, "retrain_schedule", None)
        if not schedule:
            return ""
        try:
            cron = CronSchedule(schedule)
            next_run = cron.next_after(datetime.now(timezone.utc))
        except Exception:  # pragma: no cover - niepoprawna składnia lub błąd obliczeń
            _LOGGER.debug("Nie udało się obliczyć najbliższego retrainingu", exc_info=True)
            return ""
        return next_run.astimezone().isoformat(timespec="minutes")

    def _build_adaptive_summary(self) -> str:
        if ModelRepository is None:
            return ""
        try:
            runtime_config = self._load_runtime_config()
        except FileNotFoundError:
            return ""
        except Exception:  # pragma: no cover - diagnostyka środowiska
            _LOGGER.debug("Nie udało się wczytać konfiguracji runtime dla presetów adaptacyjnych", exc_info=True)
            return ""

        registry_path = getattr(runtime_config.ai, "model_registry_path", None)
        if not registry_path:
            return ""
        try:
            repository = ModelRepository(Path(registry_path))  # type: ignore[abstract]
        except Exception:  # pragma: no cover - repozytorium może być nieosiągalne
            _LOGGER.debug("Nie udało się zainicjalizować ModelRepository", exc_info=True)
            return ""
        try:
            artifact = repository.load("adaptive_strategy_policy.json")
        except FileNotFoundError:
            return ""
        except Exception:  # pragma: no cover - uszkodzony plik lub brak dostępu
            _LOGGER.debug("Nie udało się wczytać stanu adaptive learnera", exc_info=True)
            return ""

        state = getattr(artifact, "model_state", None)
        policies = state.get("policies") if isinstance(state, Mapping) else None
        if not isinstance(policies, Mapping) or not policies:
            return ""

        fragments: list[str] = []
        for regime_key, payload in policies.items():
            if not isinstance(payload, Mapping):
                continue
            strategies = payload.get("strategies")
            if not isinstance(strategies, Iterable):
                continue
            best_name: str | None = None
            best_score: float | None = None
            best_plays = 0
            for entry in strategies:
                if not isinstance(entry, Mapping):
                    continue
                name = str(entry.get("name") or "").strip()
                if not name:
                    continue
                plays = int(entry.get("plays", 0) or 0)
                total_reward = float(entry.get("total_reward", 0.0) or 0.0)
                mean_reward = total_reward / plays if plays > 0 else float(entry.get("last_reward", 0.0) or 0.0)
                if best_score is None or mean_reward > best_score:
                    best_name = name
                    best_score = mean_reward
                    best_plays = plays
            if best_name is None or best_score is None:
                continue
            regime_label = str(regime_key).replace("_", " ")
            fragments.append(f"{regime_label}: {best_name} (μ={best_score:.2f}, n={best_plays})")

        if not fragments:
            return ""

        updated_at = ""
        try:
            updated_at = str(artifact.metadata.get("updated_at", "")).strip()
        except Exception:
            updated_at = ""
        summary = "; ".join(fragments)
        if updated_at:
            summary = f"{summary} — aktualizacja {updated_at}"
        return summary


__all__ = ["RuntimeService"]
