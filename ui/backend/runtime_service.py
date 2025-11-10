"""Serwis runtime dostarczający dane dziennika decyzji do QML."""
from __future__ import annotations

import json
import logging
import os
from collections import Counter
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from PySide6.QtCore import QObject, Property, Signal, Slot

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
        self._runtime_config_path = Path(runtime_config_path).expanduser() if runtime_config_path else None
        self._runtime_config_cache: "RuntimeAppConfig | None" = None
        self._retrain_next_run: str = ""
        self._adaptive_summary: str = ""
        self._risk_metrics: dict[str, object] = {}
        self._risk_timeline: list[dict[str, object]] = []
        self._last_operator_action: dict[str, object] | None = None
        try:
            self._update_runtime_metadata(invalidate_cache=False)
        except Exception:  # pragma: no cover - defensywna inicjalizacja
            _LOGGER.debug("Nie udało się zainicjalizować metadanych runtime", exc_info=True)

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
        if self._active_log_path is None:
            return ""
        return str(self._active_log_path)

    @Slot(str, result=bool)
    def attachToLiveDecisionLog(self, profile: str = "") -> bool:  # type: ignore[override]
        """Przełącza loader na rzeczywisty decision log skonfigurowany w core.yaml."""

        try:
            loader, log_path = self._build_live_loader(profile.strip() or None)
        except Exception as exc:  # pragma: no cover - diagnostyka
            _LOGGER.debug("attachToLiveDecisionLog failed", exc_info=True)
            self._error_message = str(exc)
            self.errorMessageChanged.emit()
            return False

        self._loader = loader
        self._active_profile = profile.strip() or None
        self._active_log_path = log_path
        self._error_message = ""
        self.errorMessageChanged.emit()
        self.liveSourceChanged.emit()
        self.loadRecentDecisions(self._default_limit)
        return True

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
    def _apply_risk_context(self, entries: Iterable[Mapping[str, object]]) -> None:
        metrics, timeline = _build_risk_context(entries)
        self._risk_metrics = metrics
        self._risk_timeline = timeline
        self.riskMetricsChanged.emit()
        self.riskTimelineChanged.emit()

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
