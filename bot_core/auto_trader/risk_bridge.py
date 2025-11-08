"""Bridges between auto-trader and risk engine payloads."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Mapping, Optional


@dataclass(slots=True)
class RiskDecision:
    """Serializable snapshot describing the outcome of a risk engine check."""

    should_trade: bool
    fraction: float
    state: str
    reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    mode: str = "demo"
    cooldown_active: bool = False
    cooldown_remaining_s: Optional[float] = None
    cooldown_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "should_trade": self.should_trade,
            "fraction": float(self.fraction),
            "state": self.state,
            "reason": self.reason,
            "details": dict(self.details),
            "mode": self.mode,
        }
        if self.stop_loss_pct is not None:
            payload["stop_loss_pct"] = float(self.stop_loss_pct)
        if self.take_profit_pct is not None:
            payload["take_profit_pct"] = float(self.take_profit_pct)
        payload["cooldown_active"] = self.cooldown_active
        if self.cooldown_remaining_s is not None:
            payload["cooldown_remaining_s"] = float(self.cooldown_remaining_s)
        if self.cooldown_reason is not None:
            payload["cooldown_reason"] = self.cooldown_reason
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RiskDecision":
        """Build a :class:`RiskDecision` from a mapping payload."""

        def _to_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return value != 0
            if isinstance(value, str):
                stripped = value.strip().lower()
                if stripped in {"", "0", "false", "no", "off"}:
                    return False
                if stripped in {"1", "true", "yes", "on"}:
                    return True
            return bool(value)

        should_trade = _to_bool(payload.get("should_trade"))

        def _to_optional_float(key: str) -> Optional[float]:
            value = payload.get(key)
            if value is None:
                return None
            try:
                candidate = float(value)
            except (TypeError, ValueError):
                return None
            return candidate

        def _to_float(key: str, default: float = 0.0) -> float:
            value = payload.get(key, default)
            try:
                candidate = float(value)
            except (TypeError, ValueError):
                return default
            return candidate

        state_value = payload.get("state")
        if isinstance(state_value, str):
            state = state_value.strip() or "unknown"
        elif state_value is None:
            state = "unknown"
        else:
            state = str(state_value)

        reason_value = payload.get("reason")
        if isinstance(reason_value, str):
            reason = reason_value.strip() or None
        elif reason_value is None:
            reason = None
        else:
            reason = str(reason_value)

        mode_value = payload.get("mode")
        if isinstance(mode_value, str):
            stripped_mode = mode_value.strip()
            mode = stripped_mode or "demo"
        elif mode_value is None:
            mode = "demo"
        else:
            mode = str(mode_value)

        details_value = payload.get("details")
        details: Dict[str, Any]
        if isinstance(details_value, Mapping):
            details = dict(details_value)
        else:
            details = {}

        cooldown_reason_value = payload.get("cooldown_reason")
        if isinstance(cooldown_reason_value, str):
            cooldown_reason = cooldown_reason_value.strip() or None
        elif cooldown_reason_value is None:
            cooldown_reason = None
        else:
            cooldown_reason = str(cooldown_reason_value)

        return cls(
            should_trade=should_trade,
            fraction=_to_float("fraction", 0.0),
            state=state,
            reason=reason,
            details=details,
            stop_loss_pct=_to_optional_float("stop_loss_pct"),
            take_profit_pct=_to_optional_float("take_profit_pct"),
            mode=mode,
            cooldown_active=_to_bool(payload.get("cooldown_active", False)),
            cooldown_remaining_s=_to_optional_float("cooldown_remaining_s"),
            cooldown_reason=cooldown_reason,
        )


@dataclass(slots=True)
class GuardrailTrigger:
    """Structured details about a guardrail that forced a HOLD signal."""

    name: str
    label: str
    comparator: str
    threshold: float
    unit: Optional[str] = None
    value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "label": self.label,
            "comparator": self.comparator,
            "threshold": float(self.threshold),
        }
        if self.unit is not None:
            payload["unit"] = self.unit
        if self.value is not None:
            payload["value"] = float(self.value)
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GuardrailTrigger":
        """Hydrate :class:`GuardrailTrigger` from mapping data."""

        name_value = payload.get("name")
        if isinstance(name_value, str):
            name = name_value.strip() or "<unknown>"
        elif name_value is None:
            name = "<unknown>"
        else:
            name = str(name_value)

        label_value = payload.get("label")
        if isinstance(label_value, str):
            label = label_value.strip()
        elif label_value is None:
            label = ""
        else:
            label = str(label_value)

        comparator_value = payload.get("comparator")
        if isinstance(comparator_value, str):
            comparator = comparator_value.strip() or "=="
        elif comparator_value is None:
            comparator = "=="
        else:
            comparator = str(comparator_value)

        def _optional_float(key: str) -> Optional[float]:
            value = payload.get(key)
            if value is None:
                return None
            try:
                candidate = float(value)
            except (TypeError, ValueError):
                return None
            return candidate

        unit_value = payload.get("unit")
        if isinstance(unit_value, str):
            unit = unit_value.strip() or None
        elif unit_value is None:
            unit = None
        else:
            unit = str(unit_value)

        threshold_value = payload.get("threshold", 0.0)
        try:
            threshold = float(threshold_value)
        except (TypeError, ValueError):
            threshold = 0.0

        return cls(
            name=name,
            label=label,
            comparator=comparator,
            threshold=threshold,
            unit=unit,
            value=_optional_float("value"),
        )


def normalize_guardrail_triggers(
    candidates: Iterable[Any] | Mapping[str, Any] | None,
) -> list[tuple[GuardrailTrigger, dict[str, Any]]]:
    """Hydrate guardrail triggers from heterogeneous payloads.

    The helper accepts sequences of :class:`GuardrailTrigger` instances,
    arbitrary mappings returned by the risk service as well as lightweight
    objects exposing a ``to_dict`` method.  Each candidate is transformed into a
    normalised :class:`GuardrailTrigger` accompanied by a dictionary snapshot of
    the original payload.  Missing keys are intentionally preserved so that
    downstream analytics can still distinguish between absent and falsy values.
    """

    if candidates is None:
        return []

    if isinstance(candidates, Mapping):
        iterable: Iterable[Any] = candidates.values()
    else:
        iterable = candidates

    normalized: list[tuple[GuardrailTrigger, dict[str, Any]]] = []
    for item in iterable:
        if item is None:
            continue

        if isinstance(item, GuardrailTrigger):
            payload = dict(item.to_dict())
            normalized.append((item, payload))
            continue

        to_dict = getattr(item, "to_dict", None)
        if callable(to_dict):
            try:
                raw_payload = to_dict()
            except Exception:  # pragma: no cover - defensive guard
                raw_payload = None
            else:
                if isinstance(raw_payload, Mapping):
                    payload = dict(raw_payload)
                    trigger = GuardrailTrigger.from_mapping(payload)
                    normalized.append((trigger, payload))
                    continue

        if is_dataclass(item):
            payload = asdict(item)
            trigger = GuardrailTrigger.from_mapping(payload)
            normalized.append((trigger, dict(payload)))
            continue

        if isinstance(item, Mapping):
            payload = dict(item)
            trigger = GuardrailTrigger.from_mapping(payload)
            normalized.append((trigger, payload))
            continue

        name = str(item).strip()
        fallback_payload = {"name": name} if name else {}
        trigger = GuardrailTrigger.from_mapping(fallback_payload)
        normalized.append((trigger, fallback_payload))

    return normalized


__all__ = ["RiskDecision", "GuardrailTrigger", "normalize_guardrail_triggers"]
