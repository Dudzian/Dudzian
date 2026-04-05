"""Kontrakty danych dla shadow trackingu okazji tradingowych."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(slots=True, frozen=True)
class OpportunityThresholdConfig:
    min_expected_edge_bps: float = 0.0
    min_probability: float = 0.5


@dataclass(slots=True, frozen=True)
class OpportunityShadowContext:
    run_id: str | None = None
    environment: str = "shadow"
    notes: Mapping[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class OpportunityShadowRecord:
    record_key: str
    symbol: str
    decision_timestamp: datetime
    model_version: str
    decision_source: str
    expected_edge_bps: float
    success_probability: float
    confidence: float
    proposed_direction: str
    accepted: bool
    rejection_reason: str | None
    rank: int
    provenance: Mapping[str, object] = field(default_factory=dict)
    threshold_config: OpportunityThresholdConfig = field(default_factory=OpportunityThresholdConfig)
    snapshot: Mapping[str, object] = field(default_factory=dict)
    context: OpportunityShadowContext = field(default_factory=OpportunityShadowContext)

    @staticmethod
    def build_record_key(
        *,
        symbol: str,
        decision_timestamp: datetime,
        model_version: str,
        rank: int,
    ) -> str:
        canonical_timestamp = _isoformat_utc(decision_timestamp)
        payload = f"{symbol}|{canonical_timestamp}|{model_version}|{rank}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decision_timestamp"] = _isoformat_utc(self.decision_timestamp)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> OpportunityShadowRecord:
        threshold_payload = payload.get("threshold_config")
        context_payload = payload.get("context")
        return cls(
            record_key=str(payload["record_key"]),
            symbol=str(payload["symbol"]),
            decision_timestamp=_parse_datetime(payload["decision_timestamp"]),
            model_version=str(payload["model_version"]),
            decision_source=str(payload["decision_source"]),
            expected_edge_bps=float(payload["expected_edge_bps"]),
            success_probability=float(payload["success_probability"]),
            confidence=float(payload["confidence"]),
            proposed_direction=str(payload["proposed_direction"]),
            accepted=_parse_bool(payload["accepted"], field_name="accepted"),
            rejection_reason=str(payload["rejection_reason"])
            if payload.get("rejection_reason") is not None
            else None,
            rank=int(payload["rank"]),
            provenance=dict(payload.get("provenance") or {}),
            threshold_config=OpportunityThresholdConfig(**dict(threshold_payload or {})),
            snapshot=dict(payload.get("snapshot") or {}),
            context=OpportunityShadowContext(**dict(context_payload or {})),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> OpportunityShadowRecord:
        return cls.from_dict(dict(json.loads(payload)))


@dataclass(slots=True, frozen=True)
class OpportunityOutcomeLabel:
    symbol: str
    decision_timestamp: datetime
    correlation_key: str
    horizon_minutes: int
    realized_return_bps: float
    max_favorable_excursion_bps: float
    max_adverse_excursion_bps: float
    hit_take_profit: bool | None = None
    hit_stop_loss: bool | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)
    label_quality: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decision_timestamp"] = _isoformat_utc(self.decision_timestamp)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> OpportunityOutcomeLabel:
        return cls(
            symbol=str(payload["symbol"]),
            decision_timestamp=_parse_datetime(payload["decision_timestamp"]),
            correlation_key=str(payload["correlation_key"]),
            horizon_minutes=int(payload["horizon_minutes"]),
            realized_return_bps=float(payload["realized_return_bps"]),
            max_favorable_excursion_bps=float(payload["max_favorable_excursion_bps"]),
            max_adverse_excursion_bps=float(payload["max_adverse_excursion_bps"]),
            hit_take_profit=_parse_optional_bool(
                payload.get("hit_take_profit"), field_name="hit_take_profit"
            ),
            hit_stop_loss=_parse_optional_bool(
                payload.get("hit_stop_loss"), field_name="hit_stop_loss"
            ),
            provenance=dict(payload.get("provenance") or {}),
            label_quality=str(payload.get("label_quality") or "unknown"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> OpportunityOutcomeLabel:
        return cls.from_dict(dict(json.loads(payload)))


class OpportunityShadowRepository:
    """Minimalne repozytorium NDJSON dla shadow decisions i outcome labels."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def shadow_records_path(self) -> Path:
        return self._root / "opportunity_shadow_records.ndjson"

    @property
    def outcome_labels_path(self) -> Path:
        return self._root / "opportunity_outcome_labels.ndjson"

    @property
    def open_outcomes_path(self) -> Path:
        return self._root / "opportunity_open_outcomes.ndjson"

    @dataclass(slots=True, frozen=True)
    class OpenOutcomeState:
        correlation_key: str
        symbol: str
        side: str
        entry_price: float
        decision_timestamp: datetime
        entry_quantity: float = 0.0
        closed_quantity: float = 0.0
        provenance: Mapping[str, object] = field(default_factory=dict)

        def to_dict(self) -> dict[str, Any]:
            payload = asdict(self)
            payload["decision_timestamp"] = _isoformat_utc(self.decision_timestamp)
            return payload

        @classmethod
        def from_dict(
            cls,
            payload: Mapping[str, object],
        ) -> "OpportunityShadowRepository.OpenOutcomeState":
            return cls(
                correlation_key=str(payload["correlation_key"]),
                symbol=str(payload["symbol"]),
                side=str(payload["side"]),
                entry_price=float(payload["entry_price"]),
                decision_timestamp=_parse_datetime(payload["decision_timestamp"]),
                entry_quantity=float(payload.get("entry_quantity") or 0.0),
                closed_quantity=float(payload.get("closed_quantity") or 0.0),
                provenance=dict(payload.get("provenance") or {}),
            )

    def append_shadow_records(self, records: Sequence[OpportunityShadowRecord]) -> Path:
        return _append_ndjson(self.shadow_records_path, (entry.to_dict() for entry in records))

    def append_outcome_labels(self, labels: Sequence[OpportunityOutcomeLabel]) -> Path:
        return _append_ndjson(self.outcome_labels_path, (entry.to_dict() for entry in labels))

    def append_outcome_labels_for_existing_records(
        self,
        labels: Sequence[OpportunityOutcomeLabel],
    ) -> tuple[Path | None, tuple[str, ...]]:
        """Append labels only when correlation keys exist in persisted shadow records."""

        result = self.attach_outcome_labels_idempotent(labels)
        if result.missing_correlation_keys:
            return None, result.missing_correlation_keys
        if result.conflicting_correlation_keys:
            return None, result.conflicting_correlation_keys
        return result.path, ()

    @dataclass(slots=True, frozen=True)
    class OutcomeAttachResult:
        path: Path | None
        attached_correlation_keys: tuple[str, ...]
        upgraded_correlation_keys: tuple[str, ...]
        duplicate_noop_correlation_keys: tuple[str, ...]
        conflicting_correlation_keys: tuple[str, ...]
        missing_correlation_keys: tuple[str, ...]

    def attach_outcome_labels_idempotent(
        self,
        labels: Sequence[OpportunityOutcomeLabel],
    ) -> "OpportunityShadowRepository.OutcomeAttachResult":
        """Attach labels with idempotent correlation policy.

        Contract:
        - one final label per correlation_key,
        - exact duplicate => no-op,
        - conflicting duplicate => rejected,
        - missing shadow correlation => rejected.
        """

        if not labels:
            return self.OutcomeAttachResult(
                path=None,
                attached_correlation_keys=(),
                upgraded_correlation_keys=(),
                duplicate_noop_correlation_keys=(),
                conflicting_correlation_keys=(),
                missing_correlation_keys=(),
            )

        shadow_records = self.load_shadow_records()
        known_shadow_keys = {record.record_key for record in shadow_records}
        shadow_by_key = {record.record_key: record for record in shadow_records}
        existing_labels = {
            str(label.correlation_key): label for label in self.load_outcome_labels()
        }
        pending: dict[str, OpportunityOutcomeLabel] = {}
        attached: list[str] = []
        upgraded: set[str] = set()
        duplicate_noop: set[str] = set()
        conflicting: set[str] = set()
        missing: set[str] = set()

        for label in labels:
            correlation_key = str(label.correlation_key)
            if correlation_key not in known_shadow_keys:
                missing.add(correlation_key)
                continue
            shadow_record = shadow_by_key.get(correlation_key)
            if shadow_record is not None:
                if str(shadow_record.symbol) != str(label.symbol):
                    conflicting.add(correlation_key)
                    continue
                if _isoformat_utc(shadow_record.decision_timestamp) != _isoformat_utc(label.decision_timestamp):
                    conflicting.add(correlation_key)
                    continue
            existing = pending.get(correlation_key) or existing_labels.get(correlation_key)
            if existing is None:
                pending[correlation_key] = label
                attached.append(correlation_key)
                continue
            if existing == label:
                duplicate_noop.add(correlation_key)
                continue
            if self._can_upgrade_label(existing=existing, incoming=label):
                pending[correlation_key] = label
                upgraded.add(correlation_key)
                continue
            conflicting.add(correlation_key)

        if missing or conflicting:
            return self.OutcomeAttachResult(
                path=None,
                attached_correlation_keys=(),
                upgraded_correlation_keys=tuple(sorted(upgraded)),
                duplicate_noop_correlation_keys=tuple(sorted(duplicate_noop)),
                conflicting_correlation_keys=tuple(sorted(conflicting)),
                missing_correlation_keys=tuple(sorted(missing)),
            )

        path: Path | None = None
        if pending:
            existing_rows = self.load_outcome_labels()
            by_key = {str(row.correlation_key): row for row in existing_rows}
            by_key.update(pending)
            ordered_keys = [
                str(row.correlation_key)
                for row in existing_rows
                if str(row.correlation_key) in by_key
            ]
            for key in pending:
                if key not in ordered_keys:
                    ordered_keys.append(key)
            merged_rows = [by_key[key].to_dict() for key in ordered_keys]
            path = _write_ndjson(self.outcome_labels_path, merged_rows)
        return self.OutcomeAttachResult(
            path=path,
            attached_correlation_keys=tuple(attached),
            upgraded_correlation_keys=tuple(sorted(upgraded)),
            duplicate_noop_correlation_keys=tuple(sorted(duplicate_noop)),
            conflicting_correlation_keys=(),
            missing_correlation_keys=(),
        )

    @staticmethod
    def _can_upgrade_label(
        *,
        existing: OpportunityOutcomeLabel,
        incoming: OpportunityOutcomeLabel,
    ) -> bool:
        if str(existing.correlation_key) != str(incoming.correlation_key):
            return False
        if str(existing.symbol) != str(incoming.symbol):
            return False
        if _isoformat_utc(existing.decision_timestamp) != _isoformat_utc(incoming.decision_timestamp):
            return False
        existing_rank = OpportunityShadowRepository._quality_rank(existing.label_quality)
        incoming_rank = OpportunityShadowRepository._quality_rank(incoming.label_quality)
        return incoming_rank > existing_rank

    @staticmethod
    def _quality_rank(value: str) -> int:
        normalized = str(value or "").strip().lower()
        if normalized.startswith("final"):
            return 3
        if normalized.startswith("partial"):
            return 2
        if normalized == "execution_proxy_pending_exit":
            return 1
        return 0

    def load_shadow_records(self) -> list[OpportunityShadowRecord]:
        return [
            OpportunityShadowRecord.from_dict(payload)
            for payload in _read_ndjson(self.shadow_records_path)
        ]

    def load_outcome_labels(self) -> list[OpportunityOutcomeLabel]:
        return [
            OpportunityOutcomeLabel.from_dict(payload)
            for payload in _read_ndjson(self.outcome_labels_path)
        ]

    def load_open_outcomes(self) -> list["OpportunityShadowRepository.OpenOutcomeState"]:
        return [
            self.OpenOutcomeState.from_dict(payload)
            for payload in _read_ndjson(self.open_outcomes_path)
        ]

    def upsert_open_outcome(self, state: "OpportunityShadowRepository.OpenOutcomeState") -> Path:
        existing_rows = self.load_open_outcomes()
        by_key = {row.correlation_key: row for row in existing_rows}
        by_key[state.correlation_key] = state
        ordered_keys = [row.correlation_key for row in existing_rows if row.correlation_key in by_key]
        if state.correlation_key not in ordered_keys:
            ordered_keys.append(state.correlation_key)
        return _write_ndjson(self.open_outcomes_path, (by_key[key].to_dict() for key in ordered_keys))

    def remove_open_outcome(self, correlation_key: str) -> Path | None:
        existing_rows = self.load_open_outcomes()
        filtered = [row for row in existing_rows if row.correlation_key != correlation_key]
        if len(filtered) == len(existing_rows):
            return None
        return _write_ndjson(self.open_outcomes_path, (row.to_dict() for row in filtered))

    def load_shadow_records_for_model(self, model_version: str) -> list[OpportunityShadowRecord]:
        normalized = str(model_version).strip()
        if not normalized:
            raise ValueError("model_version must be a non-empty string")
        return [row for row in self.load_shadow_records() if row.model_version == normalized]


def _isoformat_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _parse_datetime(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _append_ndjson(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return path


def _write_ndjson(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return path


def _read_ndjson(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payloads: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payloads.append(dict(json.loads(stripped)))
    return payloads


def _parse_bool(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"Pole {field_name} musi być typu bool")


def _parse_optional_bool(value: object, *, field_name: str) -> bool | None:
    if value is None:
        return None
    return _parse_bool(value, field_name=field_name)


__all__ = [
    "OpportunityOutcomeLabel",
    "OpportunityShadowContext",
    "OpportunityShadowRecord",
    "OpportunityShadowRepository",
    "OpportunityThresholdConfig",
]
