"""Mapping of user-facing preference knobs to preset parameter overrides."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from bot_core.marketplace import PresetDocument


class RiskTarget(str, Enum):
    """Supported qualitative risk targets used by the UI."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"

    @classmethod
    def from_value(cls, value: Any) -> "RiskTarget":
        if isinstance(value, RiskTarget):
            return value
        text = str(value or "").strip().lower()
        if not text:
            raise ValueError("Risk target value cannot be empty")
        for member in cls:
            if member.value == text:
                return member
        raise ValueError(f"Unsupported risk target: {value!r}")


@dataclass(slots=True)
class UserPreferenceConfig:
    """Normalized user preference payload collected in the UI."""

    risk_target: RiskTarget
    budget: float | None = None
    max_positions: int | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "UserPreferenceConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("User preferences payload must be a mapping")
        risk_value = payload.get("risk_target")
        if risk_value in (None, ""):
            risk_value = payload.get("riskTarget")
        risk_target = RiskTarget.from_value(risk_value)
        budget_value = payload.get("budget")
        if budget_value in ("", None):
            budget = None
        else:
            try:
                budget = float(budget_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid budget value: {budget_value!r}") from exc
        positions_value = payload.get("max_positions")
        if positions_value in (None, ""):
            positions_value = payload.get("maxPositions")
        if positions_value in (None, ""):
            max_positions = None
        else:
            try:
                max_positions = int(float(positions_value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid max_positions value: {positions_value!r}") from exc
        return cls(risk_target=risk_target, budget=budget, max_positions=max_positions)

    def as_payload(self) -> Mapping[str, Any]:
        payload: dict[str, Any] = {"risk_target": self.risk_target.value}
        if self.budget is not None:
            payload["budget"] = self.budget
        if self.max_positions is not None:
            payload["max_positions"] = self.max_positions
        return payload


class PresetPreferencePersonalizer:
    """Computes parameter overrides for presets based on user preferences."""

    def __init__(
        self,
        *,
        risk_scalars: Mapping[RiskTarget, float] | None = None,
        budget_keys: Sequence[str] | None = None,
        risk_keys: Sequence[str] | None = None,
        leverage_keys: Sequence[str] | None = None,
        position_keys: Sequence[str] | None = None,
    ) -> None:
        self._risk_scalars = dict(
            risk_scalars
            or {
                RiskTarget.CONSERVATIVE: 0.6,
                RiskTarget.BALANCED: 1.0,
                RiskTarget.AGGRESSIVE: 1.5,
            }
        )
        self._budget_keys = tuple(budget_keys or ("budget", "notional", "notional_value", "capital"))
        self._risk_keys = tuple(risk_keys or ("risk_multiplier", "risk", "risk_target"))
        self._leverage_keys = tuple(leverage_keys or ("leverage", "max_leverage"))
        self._position_keys = tuple(position_keys or ("max_positions", "max_trades"))

    def build_overrides(
        self,
        preset: PresetDocument | Mapping[str, Any],
        preferences: UserPreferenceConfig,
    ) -> Mapping[str, Mapping[str, Any]]:
        payload = self._resolve_payload(preset)
        strategies = payload.get("strategies")
        if not isinstance(strategies, Sequence):
            return {}

        overrides: dict[str, dict[str, Any]] = {}
        for entry in strategies:
            if not isinstance(entry, Mapping):
                continue
            params = entry.get("parameters")
            if not isinstance(params, Mapping):
                continue
            strategy_name = str(entry.get("name") or entry.get("engine") or payload.get("name") or "strategy")
            override: dict[str, Any] = {}

            if preferences.budget is not None:
                for key in self._budget_keys:
                    if key in params:
                        override[key] = preferences.budget

            risk_scalar = self._risk_scalars.get(preferences.risk_target, 1.0)
            risk_adjusted = False
            for key in self._risk_keys:
                value = params.get(key)
                if isinstance(value, (int, float)):
                    override[key] = float(value) * risk_scalar
                    risk_adjusted = True
            if not risk_adjusted:
                override.setdefault("risk_multiplier", risk_scalar)

            for key in self._leverage_keys:
                value = params.get(key)
                if isinstance(value, (int, float)):
                    override[key] = max(float(value) * risk_scalar, 0.0)

            if preferences.max_positions is not None:
                for key in self._position_keys:
                    if key in params:
                        override[key] = max(preferences.max_positions, 1)

            if override:
                overrides[strategy_name] = override

        return overrides

    @staticmethod
    def _resolve_payload(preset: PresetDocument | Mapping[str, Any]) -> Mapping[str, Any]:
        if isinstance(preset, PresetDocument):
            return preset.payload
        if isinstance(preset, Mapping):
            if "strategies" in preset:
                return preset
            candidate = preset.get("preset")
            if isinstance(candidate, Mapping):
                return candidate
        raise TypeError("Unsupported preset payload type")


__all__ = [
    "PresetPreferencePersonalizer",
    "RiskTarget",
    "UserPreferenceConfig",
]

